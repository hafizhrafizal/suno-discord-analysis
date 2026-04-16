"""
routers/chat.py — RAG chat and conversation summarisation.

  POST /api/chat
  POST /api/summarize
  POST /api/summarize/followup
"""

import json
import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

import state
from config import VALID_CHAT_MODELS
from database import get_db
from embeddings import active_collection, embed_texts_async
from sql_helpers import (
    _parse_upload_ids,
    _sql_upload_ids_clause,
    _suno_sql,
    keyword_search,
    sql_date_clauses,
    sql_min_words_clause,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Summarisation pipeline helpers ────────────────────────────────────────────

# How many semantically retrieved candidates we aim for before clustering.
_SUMMARY_CANDIDATE_TARGET = 100


def _cosine_sim_matrix(embs: np.ndarray) -> np.ndarray:
    """Return the pairwise cosine-similarity matrix for a 2-D embedding array."""
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normed = embs / norms
    return normed @ normed.T


def _deduplicate_candidates(
    rows: list,
    embs: np.ndarray,
    threshold: float = 0.97,
) -> tuple:
    """
    Remove near-duplicate messages (cosine similarity ≥ threshold).
    Also drops trivially short messages (< 10 chars) that slipped past
    the word-count filter.
    Keeps the first occurrence in ranking order.

    Returns (deduped_rows, deduped_embs, n_removed).
    """
    dropped: set = set()

    # Pre-drop empty / very short messages
    for i, row in enumerate(rows):
        if len((row.get("content") or "").strip()) < 10:
            dropped.add(i)

    sim = _cosine_sim_matrix(embs)
    keep: list = []
    for i in range(len(rows)):
        if i in dropped:
            continue
        keep.append(i)
        # Mark highly-similar later messages as duplicates
        for j in range(i + 1, len(rows)):
            if j not in dropped and sim[i, j] >= threshold:
                dropped.add(j)

    if not keep:
        return [], embs[:0], len(rows)

    n_removed = len(rows) - len(keep)
    kept_rows = [rows[i] for i in keep]
    kept_embs = embs[np.array(keep, dtype=int)]
    return kept_rows, kept_embs, n_removed


def _numpy_kmeans(
    embs: np.ndarray, n_clusters: int, max_iter: int = 60
) -> np.ndarray:
    """Minimal Lloyd's K-Means using only NumPy (fallback when sklearn absent)."""
    rng = np.random.default_rng(42)
    init_idx = rng.choice(len(embs), size=n_clusters, replace=False)
    centroids = embs[init_idx].copy()
    labels = np.zeros(len(embs), dtype=int)

    for _ in range(max_iter):
        # (N, K) distance matrix
        dists = np.linalg.norm(
            embs[:, None, :] - centroids[None, :, :], axis=2
        )
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                centroids[k] = embs[mask].mean(axis=0)

    return labels


def _cluster_candidates(rows: list, embs: np.ndarray) -> tuple:
    """
    Cluster candidate messages by semantic similarity.
    Returns (labels, algo_name, n_clusters) where labels is a flat list of
    integer cluster labels, one per row.

    Priority order:
      1. HDBSCAN  — best quality, optional dependency
      2. sklearn KMeans  — common optional dep
      3. NumPy KMeans  — always available fallback
    """
    n = len(rows)
    if n <= 4:
        # Too few messages to cluster meaningfully; each is its own cluster.
        return list(range(n)), "none", n

    n_clusters = max(2, min(12, n // 8))

    # 1. HDBSCAN
    try:
        import hdbscan as _hdbscan  # type: ignore
        clusterer = _hdbscan.HDBSCAN(
            min_cluster_size=max(3, n // 15), metric="euclidean"
        )
        labels = clusterer.fit_predict(embs)
        # HDBSCAN labels noise as -1; fold into an extra cluster
        max_lbl = int(labels.max()) if labels.max() >= 0 else -1
        labels = np.where(labels == -1, max_lbl + 1, labels)
        n_unique = len(set(labels.tolist()))
        return labels.tolist(), "HDBSCAN", n_unique
    except ImportError:
        pass

    # 2. sklearn KMeans
    try:
        from sklearn.cluster import KMeans as _KMeans  # type: ignore
        km = _KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = km.fit_predict(embs).tolist()
        return labels, "KMeans (sklearn)", n_clusters
    except ImportError:
        pass

    # 3. NumPy-only fallback
    labels = _numpy_kmeans(embs, n_clusters).tolist()
    return labels, "KMeans (NumPy)", n_clusters


def _sample_cluster(
    cluster_rows: list,
    cluster_embs: np.ndarray,
    max_reps: int = 2,
    max_diverse: int = 2,
) -> list:
    """
    Sample representative + diverse messages from a single cluster.

    - max_reps messages closest to the cluster centroid (author-diverse).
    - max_diverse additional messages maximally distant from the reps
      (prefer unseen authors).
    """
    n = len(cluster_rows)
    if n <= max_reps + max_diverse:
        return list(cluster_rows)

    centroid = cluster_embs.mean(axis=0)
    dist_to_centroid = np.linalg.norm(cluster_embs - centroid, axis=1)
    closeness_order = np.argsort(dist_to_centroid)

    # -- Representatives: closest to centroid, author-diverse --
    rep_idx: list = []
    rep_authors: set = set()
    for i in closeness_order:
        if len(rep_idx) >= max_reps:
            break
        author = cluster_rows[int(i)].get("username", "")
        if author not in rep_authors or not rep_authors:
            rep_idx.append(int(i))
            rep_authors.add(author)

    # -- Diverse: farthest from reps, prefer unseen authors --
    rep_set = set(rep_idx)
    remaining = [i for i in range(n) if i not in rep_set]
    if not remaining:
        return [cluster_rows[i] for i in sorted(rep_idx)]

    rem_embs = cluster_embs[remaining]
    rep_embs = cluster_embs[list(rep_set)]
    # Minimum distance from each remaining point to any representative
    min_dist_to_reps = np.min(
        np.linalg.norm(
            rem_embs[:, None, :] - rep_embs[None, :, :], axis=2
        ),
        axis=1,
    )
    diverse_order = np.argsort(-min_dist_to_reps)  # descending

    diverse_idx: list = []
    diverse_authors: set = set(rep_authors)
    for rank_pos in diverse_order:
        if len(diverse_idx) >= max_diverse:
            break
        orig = remaining[int(rank_pos)]
        author = cluster_rows[orig].get("username", "")
        if author not in diverse_authors:
            diverse_idx.append(orig)
            diverse_authors.add(author)
        elif len(diverse_idx) < max_diverse:
            # Accept same-author duplicate if no fresh author is available
            diverse_idx.append(orig)

    selected = sorted(rep_idx + diverse_idx)
    return [cluster_rows[i] for i in selected]


def _build_evidence_set(
    rows: list,
    embs: np.ndarray,
    max_evidence: int = 60,
) -> tuple:
    """
    Full inner pipeline:
      1. Deduplicate near-identical messages.
      2. Cluster remaining candidates by topic.
      3. Sample representatives + diverse messages per cluster.
      4. Assemble, sort chronologically, and cap at max_evidence.

    Returns (evidence, stats) where stats = {
      n_input, n_after_dedup, n_dupes_removed, algorithm, n_clusters, n_evidence
    }
    """
    n_input = len(rows)

    # Step 3a: deduplication
    rows, embs, n_removed = _deduplicate_candidates(rows, embs)
    if not rows:
        return [], {
            "n_input": n_input, "n_after_dedup": 0, "n_dupes_removed": n_removed,
            "algorithm": "none", "n_clusters": 0, "n_evidence": 0,
        }

    n_after_dedup = len(rows)

    # Step 3b: clustering
    labels, algo_name, n_clusters = _cluster_candidates(rows, embs)
    unique_labels = sorted(set(labels))

    # Step 3c: per-cluster sampling
    evidence: list = []
    for lbl in unique_labels:
        cluster_idx = [i for i, lb in enumerate(labels) if lb == lbl]
        c_rows = [rows[i] for i in cluster_idx]
        c_embs = embs[np.array(cluster_idx, dtype=int)]
        evidence.extend(_sample_cluster(c_rows, c_embs))

    # Step 3d: chronological sort + cap
    evidence.sort(key=lambda r: (r.get("date") or ""))
    evidence = evidence[:max_evidence]

    stats = {
        "n_input": n_input,
        "n_after_dedup": n_after_dedup,
        "n_dupes_removed": n_removed,
        "algorithm": algo_name,
        "n_clusters": n_clusters,
        "n_evidence": len(evidence),
    }
    return evidence, stats


# ── /api/chat ─────────────────────────────────────────────────────────────────

@router.post("/api/chat")
async def chat_endpoint(request: Request):
    body       = await request.json()
    message    = (body.get("message") or "").strip()
    history    = body.get("history") or []
    upload_ids = body.get("upload_ids") or ""
    chat_model = (body.get("model") or "gpt-5.4").strip()

    if not message:
        raise HTTPException(400, "Empty message.")
    if not state.openai_client:
        raise HTTPException(400, "OpenAI API key not set — add it in Settings.")
    if chat_model not in VALID_CHAT_MODELS:
        raise HTTPException(400, f"Unknown model '{chat_model}'.")

    uid_list = _parse_upload_ids(upload_ids)

    _chat_query_emb = None
    try:
        _chat_query_emb = (await embed_texts_async([message]))[0]
    except Exception:
        pass

    def _semantic_search() -> list:
        rows: list = []
        try:
            col = active_collection()
            if col is None:
                logger.warning("/api/chat semantic retrieval: no active vector collection")
                return rows
            if col.count() == 0 or _chat_query_emb is None:
                return rows
            results   = col.query(query_embeddings=[_chat_query_emb], n_results=12)
            ids       = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0]
            uuid_map: dict = {}
            if ids:
                conn = get_db()
                try:
                    ph      = ",".join("?" * len(ids))
                    db_rows = conn.execute(
                        f"SELECT * FROM messages WHERE msg_uuid IN ({ph})", ids
                    ).fetchall()
                    uuid_map = {r["msg_uuid"]: dict(r) for r in db_rows}
                finally:
                    conn.close()
            for uid, dist in zip(ids, distances):
                row = uuid_map.get(uid)
                if row is None:
                    continue
                if uid_list and row.get("upload_id") not in uid_list:
                    continue
                try:
                    score = round(1.0 - float(dist), 4) if dist is not None else None
                except (TypeError, ValueError):
                    score = None
                row["_score"] = score
                rows.append(row)
        except Exception as exc:
            logger.error("/api/chat semantic retrieval error: %s", exc, exc_info=True)
        return rows

    import asyncio
    loop = asyncio.get_running_loop()
    semantic_rows, keyword_rows = await asyncio.gather(
        loop.run_in_executor(state.vector_executor, _semantic_search),
        keyword_search(keyword=message, upload_ids=upload_ids, limit=10),
    )

    context_rows    = list(semantic_rows)
    existing_uuids  = {r["msg_uuid"] for r in context_rows}
    for r in keyword_rows:
        if r["msg_uuid"] not in existing_uuids:
            context_rows.append(r)
            existing_uuids.add(r["msg_uuid"])
    context_rows = context_rows[:20]

    if context_rows:
        ctx_text = "\n".join(
            f"[{r['username']} | {r['date']}] {r['content']}"
            for r in context_rows
        )
        system = (
            "You are a knowledgeable assistant for the Suno AI Discord community.\n\n"
            "INSTRUCTIONS:\n"
            "- Use the retrieved conversation excerpts below as your PRIMARY source of truth.\n"
            "- Cite specific usernames (e.g. **@username**) when referencing their messages.\n"
            "- If the context does not cover the question, say so clearly before answering from general knowledge.\n\n"
            "MANDATORY FORMATTING — your entire response MUST be valid Markdown:\n"
            "- Start with a `##` heading that summarises the answer topic.\n"
            "- Use `###` subheadings to separate distinct sub-topics.\n"
            "- Use **bold** for key terms, usernames, and important points.\n"
            "- Use `-` bullet lists for multiple items or steps; use `1.` numbered lists for sequences.\n"
            "- Use `> blockquote` to highlight a direct or paraphrased user quote.\n"
            "- Use `inline code` for technical terms, settings, or commands.\n"
            "- End with a `---` rule followed by a brief *Sources* section listing cited usernames and dates.\n"
            "- Do NOT output plain prose paragraphs without any formatting.\n\n"
            "RETRIEVED CONTEXT:\n" + ctx_text
        )
    else:
        system = (
            "You are a helpful assistant for the Suno AI Discord community.\n"
            "No embedded messages are available — answer from general knowledge.\n\n"
            "MANDATORY FORMATTING — your entire response MUST be valid Markdown:\n"
            "- Start with a `##` heading.\n"
            "- Use **bold**, `-` bullet lists, `###` subheadings, and `inline code` where appropriate.\n"
            "- Do NOT output plain prose without any Markdown structure.\n"
        )

    is_o_model = chat_model.startswith("o")
    sys_role   = "developer" if is_o_model else "system"
    msgs       = [{"role": sys_role, "content": system}]
    for turn in history[-20:]:
        role    = turn.get("role")
        content = turn.get("content")
        if role in {"system", "user", "assistant"} and content:
            msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": message})

    async def generate():
        try:
            stream = state.openai_client.chat.completions.create(
                model=chat_model, messages=msgs, stream=True
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'content': delta})}\n\n"
        except Exception as exc:
            logger.error("chat generate() error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


# ── /api/summarize ────────────────────────────────────────────────────────────

@router.post("/api/summarize")
async def summarize_endpoint(request: Request):
    body       = await request.json()
    username   = (body.get("username") or "").strip()
    date_from  = (body.get("date_from") or "").strip()
    date_to    = (body.get("date_to") or "").strip()
    prompt_txt = (body.get("prompt") or "").strip()
    upload_ids = (body.get("upload_ids") or "")
    min_words  = int(body.get("min_words") or 0)
    suno_team  = str(body.get("suno_team") or "all")
    sum_model  = (body.get("model") or "gpt-5.4").strip()

    if not state.openai_client:
        raise HTTPException(400, "OpenAI API key not set — add it in Settings.")
    if sum_model not in VALID_CHAT_MODELS:
        raise HTTPException(400, f"Unknown model '{sum_model}'.")

    uid_list              = _parse_upload_ids(upload_ids)
    uid_sql, uid_params   = _sql_upload_ids_clause(uid_list)
    date_sql, date_params = sql_date_clauses(date_from or None, date_to or None)
    words_sql, words_params = sql_min_words_clause(min_words)

    # ── Step 1: Apply metadata filters ────────────────────────────────────────
    # Fetch msg_uuid in addition to the display fields so we can look up
    # stored vectors later.
    params: list = []
    sql = "SELECT msg_uuid, username, date, content FROM messages WHERE 1=1"
    if username:
        sql += " AND LOWER(username) LIKE LOWER(?)"
        params.append(f"%{username}%")
    sql += uid_sql
    params.extend(uid_params)
    sql += _suno_sql(suno_team) + date_sql + words_sql
    params.extend(date_params + words_params)
    sql += " ORDER BY date, row_index"

    conn = get_db()
    db_rows = conn.execute(sql, params).fetchall()
    conn.close()

    if not db_rows:
        raise HTTPException(404, "No messages found matching those filters.")

    # UUID → dict lookup for fast post-filter intersection with vector results
    filtered_map: dict = {r["msg_uuid"]: dict(r) for r in db_rows}
    n_filtered = len(filtered_map)

    # ── Step 2: Semantic retrieval from vector store ───────────────────────────
    # Use the user's custom prompt as the retrieval query when available;
    # otherwise fall back to a generic coverage query.
    retrieval_query = (
        prompt_txt
        or "key discussions, important insights, notable feedback, use cases, significant events"
    )

    query_embedding: Optional[list] = None
    try:
        query_embedding = (await embed_texts_async([retrieval_query]))[0]
    except Exception as exc:
        logger.warning("summarize: failed to embed query (%s) — skipping vector step", exc)

    candidate_rows: list = []          # rows that passed both filter AND vector rank
    candidate_embs_raw: list = []      # corresponding stored embedding vectors
    _log_total_in_store: int = 0
    _log_overfetch_n: int = 0
    _log_vector_ok: bool = False       # True when vector retrieval ran successfully
    _log_vector_err: str = ""          # non-empty when retrieval failed

    if query_embedding is not None:
        import asyncio as _asyncio
        _sum_loop = _asyncio.get_running_loop()

        def _vector_retrieval_sync() -> tuple:
            """Run all blocking vector-store calls in a thread (safe for Qdrant HTTP)."""
            try:
                col = active_collection()
                total = col.count()
                if total == 0:
                    return total, 0, [], [], True, ""
                overfetch = min(
                    max(n_filtered * 3, _SUMMARY_CANDIDATE_TARGET * 3),
                    min(total, 800),
                )
                results = col.query(
                    query_embeddings=[query_embedding], n_results=overfetch
                )
                result_ids   = results.get("ids",      [[]])[0]
                result_dists = results.get("distances", [[]])[0]

                scored: list = []
                for uid, dist in zip(result_ids, result_dists):
                    if uid in filtered_map:
                        score = round(1.0 - float(dist), 4) if dist is not None else 0.0
                        scored.append((score, uid))
                scored.sort(key=lambda x: -x[0])
                top_uuids = [uid for _, uid in scored[:_SUMMARY_CANDIDATE_TARGET]]

                rows_out: list = []
                embs_out: list = []
                if top_uuids:
                    emb_result = col.get(ids=top_uuids, include=["embeddings"])
                    emb_ids    = emb_result.get("ids", [])
                    emb_vecs   = emb_result.get("embeddings", [])
                    emb_map: dict = {
                        eid: evec
                        for eid, evec in zip(emb_ids, emb_vecs)
                        if evec is not None
                    }
                    score_map: dict = {uid: s for s, uid in scored}
                    for uid in top_uuids:
                        if uid in emb_map and uid in filtered_map:
                            row = dict(filtered_map[uid])
                            row["_score"] = score_map.get(uid)
                            rows_out.append(row)
                            embs_out.append(emb_map[uid])

                return total, overfetch, rows_out, embs_out, True, ""
            except Exception as exc:
                err_msg = f"{type(exc).__name__}: {exc}"
                logger.error("summarize: vector retrieval error (%s) — using fallback", err_msg, exc_info=True)
                return 0, 0, [], [], False, err_msg

        (
            _log_total_in_store,
            _log_overfetch_n,
            candidate_rows,
            candidate_embs_raw,
            _log_vector_ok,
            _log_vector_err,
        ) = await _sum_loop.run_in_executor(state.vector_executor, _vector_retrieval_sync)

        if _log_vector_ok:
            logger.info(
                "summarize: filtered=%d  overfetch=%d  candidates=%d",
                n_filtered, _log_overfetch_n, len(candidate_rows),
            )

    # ── Steps 3-6: Dedup → Cluster → Sample → Assemble ───────────────────────
    # If we got a meaningful candidate set, run the full pipeline.
    # Otherwise fall back to sending all filtered rows to the LLM (original
    # behaviour), which is still correct but less focused.
    use_pipeline = len(candidate_rows) >= 10
    pipeline_stats: dict = {}

    if use_pipeline:
        embs_array = np.array(candidate_embs_raw, dtype=np.float32)
        evidence, pipeline_stats = _build_evidence_set(candidate_rows, embs_array)
        if not evidence:
            # Should not happen, but guard against edge cases
            use_pipeline = False

    if not use_pipeline:
        logger.info(
            "summarize: pipeline skipped (candidates=%d) — sending all %d filtered rows",
            len(candidate_rows), n_filtered,
        )
        # Fall back: convert sqlite Row objects to plain dicts and send all
        evidence = [dict(r) for r in db_rows]

    # ── Step 7: Build the LLM prompt and stream the summary ──────────────────
    conv = "\n".join(
        f"[{r['username']} | {r['date']}]: {r['content']}"
        for r in evidence
    )

    n_evidence = len(evidence)
    conv_header = (
        f"CONVERSATION ({n_evidence} selected messages"
        + (f" from {n_filtered} filtered" if use_pipeline else "")
        + "):"
    )

    default_prompt = """\
Produce a comprehensive summary of the Discord conversation below.

MANDATORY STRUCTURE (strictly follow this Markdown layout):

## Overview
One short paragraph giving the high-level context.

## Key Topics
For each major topic:
### [Topic Name]
- Bullet points covering the main discussion points.
- Use **bold** for important terms or conclusions.

## Notable Opinions & Insights
> Direct or paraphrased quotes from participants, formatted as blockquotes, with **@username** attributed.

## Decisions / Conclusions
- Any outcomes, agreed next steps, or unresolved questions.

## Participants
- List unique usernames who contributed meaningfully.

---
Do NOT output plain paragraphs. Every section must use the Markdown elements above."""

    user_prompt = prompt_txt or default_prompt
    full        = f"{user_prompt}\n\n{conv_header}\n{conv}"

    is_o_model = sum_model.startswith("o")
    sys_role   = "developer" if is_o_model else "system"

    # ── Build pipeline log events ─────────────────────────────────────────────
    pipeline_log: list = []
    pipeline_log.append({
        "type": "log", "step": "filter", "label": "Metadata filter",
        "msg": f"{n_filtered:,} messages matched filters",
    })
    if _log_vector_ok and _log_total_in_store > 0:
        pipeline_log.append({
            "type": "log", "step": "retrieval", "label": "Vector retrieval",
            "msg": (
                f"Overfetched {_log_overfetch_n:,} from {_log_total_in_store:,} in store "
                f"→ {len(candidate_rows):,} candidates after intersection"
            ),
        })
    elif not _log_vector_ok:
        pipeline_log.append({
            "type": "log", "step": "fallback", "label": "Vector fallback",
            "msg": (
                f"Vector retrieval failed ({_log_vector_err}) — using all filtered messages"
                if _log_vector_err else
                "Embedding unavailable — using all filtered messages"
            ),
        })
    elif _log_total_in_store == 0:
        pipeline_log.append({
            "type": "log", "step": "fallback", "label": "Vector fallback",
            "msg": "Vector store is empty — using all filtered messages",
        })

    if use_pipeline and pipeline_stats:
        n_dupes = pipeline_stats.get("n_dupes_removed", 0)
        n_dedup = pipeline_stats.get("n_after_dedup", len(candidate_rows))
        algo    = pipeline_stats.get("algorithm", "unknown")
        n_clust = pipeline_stats.get("n_clusters", 0)
        pipeline_log.append({
            "type": "log", "step": "dedup", "label": "Deduplication",
            "msg": (
                f"{n_dupes:,} near-duplicate{'s' if n_dupes != 1 else ''} removed "
                f"({n_dedup:,} remain)"
            ),
        })
        if algo != "none":
            pipeline_log.append({
                "type": "log", "step": "cluster", "label": "Clustering",
                "msg": f"{n_clust} cluster{'s' if n_clust != 1 else ''} via {algo}",
            })
        pipeline_log.append({
            "type": "log", "step": "sample", "label": "Sampling",
            "msg": f"Sampled down to {n_evidence:,} representative messages",
        })
    else:
        pipeline_log.append({
            "type": "log", "step": "fallback", "label": "Pipeline skipped",
            "msg": f"Too few candidates — using {n_evidence:,} filtered messages directly",
        })

    pipeline_log.append({
        "type": "log", "step": "llm", "label": "LLM generation",
        "msg": f"Summarising {n_evidence:,} messages with {sum_model}…",
    })

    async def generate():
        # Emit research transparency log before LLM content
        for entry in pipeline_log:
            yield f"data: {json.dumps(entry)}\n\n"
        try:
            stream = state.openai_client.chat.completions.create(
                model=sum_model,
                messages=[
                    {
                        "role": sys_role,
                        "content": (
                            "You are an expert analyst summarising Discord conversations from the Suno AI community. "
                            "You MUST respond exclusively in well-structured Markdown. "
                            "Never output plain prose. Always use ## headings, ### subheadings, "
                            "**bold**, - bullet lists, > blockquotes, and `code` where appropriate."
                        ),
                    },
                    {"role": "user", "content": full},
                ],
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'content': delta})}\n\n"
        except Exception as exc:
            logger.error("summarize generate() error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


# ── /api/summarize/followup ───────────────────────────────────────────────────

# Smaller candidate target for follow-up: focused evidence, not breadth.
_FOLLOWUP_CANDIDATE_TARGET = 60


@router.post("/api/summarize/followup")
async def summarize_followup_endpoint(request: Request):
    """
    Follow-up Q&A within a Hybrid Summary session.

    Accepts the follow-up question plus the same filter params used for the
    initial summary.  Runs the same semantic retrieval + dedup/cluster/sample
    pipeline against the same filtered message pool, then streams a focused
    Markdown answer grounded in that fresh evidence.

    Request body:
      question   – the follow-up question (required)
      history    – [{role, content}] conversation so far (initial summary first)
      username, date_from, date_to, upload_ids, min_words, suno_team, model
                 – same filter params as the original /api/summarize call
    """
    body       = await request.json()
    question   = (body.get("question") or "").strip()
    history    = body.get("history") or []
    username   = (body.get("username") or "").strip()
    date_from  = (body.get("date_from") or "").strip()
    date_to    = (body.get("date_to") or "").strip()
    upload_ids = (body.get("upload_ids") or "")
    min_words  = int(body.get("min_words") or 0)
    suno_team  = str(body.get("suno_team") or "all")
    sum_model  = (body.get("model") or "gpt-5.4").strip()

    if not question:
        raise HTTPException(400, "Empty question.")
    if not state.openai_client:
        raise HTTPException(400, "OpenAI API key not set — add it in Settings.")
    if sum_model not in VALID_CHAT_MODELS:
        raise HTTPException(400, f"Unknown model '{sum_model}'.")

    uid_list              = _parse_upload_ids(upload_ids)
    uid_sql, uid_params   = _sql_upload_ids_clause(uid_list)
    date_sql, date_params = sql_date_clauses(date_from or None, date_to or None)
    words_sql, words_params = sql_min_words_clause(min_words)

    # ── Step 1: Apply the same metadata filters as the initial summary ─────────
    params: list = []
    sql = "SELECT msg_uuid, username, date, content FROM messages WHERE 1=1"
    if username:
        sql += " AND LOWER(username) LIKE LOWER(?)"
        params.append(f"%{username}%")
    sql += uid_sql
    params.extend(uid_params)
    sql += _suno_sql(suno_team) + date_sql + words_sql
    params.extend(date_params + words_params)
    sql += " ORDER BY date, row_index"

    conn = get_db()
    db_rows = conn.execute(sql, params).fetchall()
    conn.close()

    if not db_rows:
        raise HTTPException(404, "No messages found matching those filters.")

    filtered_map: dict = {r["msg_uuid"]: dict(r) for r in db_rows}
    n_filtered = len(filtered_map)

    # ── Step 2: Semantic retrieval using the follow-up QUESTION as query ───────
    query_embedding: Optional[list] = None
    try:
        query_embedding = (await embed_texts_async([question]))[0]
    except Exception as exc:
        logger.warning("summarize/followup: failed to embed question (%s)", exc)

    candidate_rows: list = []
    candidate_embs_raw: list = []

    if query_embedding is not None:
        import asyncio as _asyncio
        _fu_loop = _asyncio.get_running_loop()

        def _fu_vector_retrieval_sync() -> tuple:
            """Run all blocking vector-store calls in a thread (safe for Qdrant HTTP)."""
            try:
                col = active_collection()
                if col is None:
                    logger.warning("summarize/followup: no active vector collection")
                    return 0, 0, [], []
                total = col.count()
                if total == 0:
                    return total, 0, [], []
                overfetch_n = min(
                    max(n_filtered * 3, _FOLLOWUP_CANDIDATE_TARGET * 3),
                    min(total, 600),
                )
                results      = col.query(
                    query_embeddings=[query_embedding], n_results=overfetch_n
                )
                result_ids   = results.get("ids",      [[]])[0]
                result_dists = results.get("distances", [[]])[0]

                scored: list = []
                for uid, dist in zip(result_ids, result_dists):
                    if uid in filtered_map:
                        score = round(1.0 - float(dist), 4) if dist is not None else 0.0
                        scored.append((score, uid))
                scored.sort(key=lambda x: -x[0])
                top_uuids = [uid for _, uid in scored[:_FOLLOWUP_CANDIDATE_TARGET]]

                rows_out: list = []
                embs_out: list = []
                if top_uuids:
                    emb_result = col.get(ids=top_uuids, include=["embeddings"])
                    emb_map: dict = {
                        eid: evec
                        for eid, evec in zip(
                            emb_result.get("ids", []), emb_result.get("embeddings", [])
                        )
                        if evec is not None
                    }
                    score_map: dict = {uid: s for s, uid in scored}
                    for uid in top_uuids:
                        if uid in emb_map and uid in filtered_map:
                            row = dict(filtered_map[uid])
                            row["_score"] = score_map.get(uid)
                            rows_out.append(row)
                            embs_out.append(emb_map[uid])

                return total, overfetch_n, rows_out, embs_out
            except Exception as exc:
                logger.error("summarize/followup: vector retrieval error (%s)", exc, exc_info=True)
                return 0, 0, [], []

        _fu_total, _fu_overfetch, candidate_rows, candidate_embs_raw = \
            await _fu_loop.run_in_executor(state.vector_executor, _fu_vector_retrieval_sync)

        if _fu_total > 0:
            logger.info(
                "summarize/followup: filtered=%d  overfetch=%d  candidates=%d",
                n_filtered, _fu_overfetch, len(candidate_rows),
            )

    # ── Steps 3-6: Dedup → Cluster → Sample → Assemble ────────────────────────
    # Use max_evidence=40 for follow-up: tight, focused evidence beats breadth.
    use_pipeline = len(candidate_rows) >= 10
    fu_pipeline_stats: dict = {}

    if use_pipeline:
        embs_array = np.array(candidate_embs_raw, dtype=np.float32)
        evidence, fu_pipeline_stats = _build_evidence_set(candidate_rows, embs_array, max_evidence=40)
        if not evidence:
            use_pipeline = False

    if not use_pipeline:
        logger.info(
            "summarize/followup: pipeline skipped (candidates=%d) — using first 40 filtered rows",
            len(candidate_rows),
        )
        evidence = [dict(r) for r in db_rows[:40]]

    n_evidence = len(evidence)

    # ── Step 7: Build prompt with conversation history + evidence, then stream ─
    evidence_context = "\n".join(
        f"[{r['username']} | {r['date']}]: {r['content']}"
        for r in evidence
    )

    is_o_model = sum_model.startswith("o")
    sys_role   = "developer" if is_o_model else "system"
    system_content = (
        "You are an expert analyst for the Suno AI Discord community. "
        "A Hybrid Summary of a Discord conversation was already generated for the user. "
        "Now answer the follow-up question using the fresh evidence retrieved below — "
        "drawn from the same filtered message pool used for the summary. "
        "Be precise and grounded in the evidence. Cite usernames and dates where relevant. "
        "Respond in well-structured Markdown.\n\n"
        f"RETRIEVED EVIDENCE FOR THIS QUESTION ({n_evidence} messages):\n{evidence_context}"
    )

    msgs: list = [{"role": sys_role, "content": system_content}]
    # Include prior turns (initial summary at history[0], then interleaved Q&A).
    for turn in history[-12:]:
        role    = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": question})

    # ── Build follow-up pipeline log ──────────────────────────────────────────
    fu_log: list = [
        {"type": "log", "step": "filter", "label": "Metadata filter",
         "msg": f"{n_filtered:,} messages in filtered pool"},
        {"type": "log", "step": "retrieval", "label": "Vector retrieval",
         "msg": f"{len(candidate_rows):,} semantic candidates retrieved for follow-up"},
    ]
    if use_pipeline and fu_pipeline_stats:
        n_dupes = fu_pipeline_stats.get("n_dupes_removed", 0)
        algo    = fu_pipeline_stats.get("algorithm", "unknown")
        n_clust = fu_pipeline_stats.get("n_clusters", 0)
        if n_dupes:
            fu_log.append({"type": "log", "step": "dedup", "label": "Deduplication",
                           "msg": f"{n_dupes:,} near-duplicate{'s' if n_dupes != 1 else ''} removed"})
        if algo != "none":
            fu_log.append({"type": "log", "step": "cluster", "label": "Clustering",
                           "msg": f"{n_clust} cluster{'s' if n_clust != 1 else ''} via {algo}"})
    fu_log.append({"type": "log", "step": "llm", "label": "LLM generation",
                   "msg": f"Answering with {n_evidence:,} evidence messages via {sum_model}…"})

    async def generate():
        for entry in fu_log:
            yield f"data: {json.dumps(entry)}\n\n"
        try:
            stream = state.openai_client.chat.completions.create(
                model=sum_model, messages=msgs, stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'content': delta})}\n\n"
        except Exception as exc:
            logger.error("summarize/followup generate() error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )
