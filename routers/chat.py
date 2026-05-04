"""
routers/chat.py — RAG chat and conversation summarisation.

  POST /api/chat
  POST /api/summarize
  POST /api/summarize/followup
  POST /api/summarize-results
  POST /api/summarize-results/followup
  POST /api/user-profile
  POST /api/user-profile/followup
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

# Minimum candidates to guarantee after threshold filtering (avoids degenerate clustering).
_MIN_CANDIDATES = 15


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


def _noise_to_singletons(labels: list) -> list:
    """Replace noise labels (-1) with unique singleton cluster IDs."""
    next_lbl = max(labels) + 1
    result = []
    for lbl in labels:
        if lbl == -1:
            result.append(next_lbl)
            next_lbl += 1
        else:
            result.append(lbl)
    return result


def _cluster_candidates(rows: list, embs: np.ndarray) -> tuple:
    """
    Cluster candidate messages by semantic similarity.
    Returns (labels, algo_name, n_clusters) where labels is a flat list of
    integer cluster labels, one per row.

    Priority order:
      1. HDBSCAN  — preferred; auto-detects cluster count; noise points become
                    individual singleton clusters so every outlier contributes.
      2. OPTICS   — density-based like HDBSCAN, available via sklearn;
                    also auto-detects cluster count with no fixed k.
      3. KMeans   — last resort only; k scales with n but is uncapped so topic
                    breadth is not artificially limited.
    """
    n = len(rows)
    if n <= 4:
        return list(range(n)), "none", n

    # 1. HDBSCAN — best quality, discovers natural cluster structure
    try:
        import hdbscan as _hdbscan  # type: ignore
        clusterer = _hdbscan.HDBSCAN(
            min_cluster_size=max(2, n // 25),  # small → finer-grained clusters
            min_samples=max(1, n // 50),        # low → fewer noise points
            metric="euclidean",
        )
        labels = _noise_to_singletons(clusterer.fit_predict(embs).tolist())
        return labels, "HDBSCAN", len(set(labels))
    except ImportError:
        pass

    # 2. OPTICS (sklearn) — density-based, auto-detects cluster count, no fixed k
    try:
        from sklearn.cluster import OPTICS as _OPTICS  # type: ignore
        clusterer = _OPTICS(
            min_samples=max(2, n // 50),
            min_cluster_size=max(2, n // 25),
            metric="euclidean",
        )
        labels = _noise_to_singletons(clusterer.fit_predict(embs).tolist())
        return labels, "OPTICS", len(set(labels))
    except ImportError:
        pass

    # 3. KMeans — last resort; k scales with n, no hard upper cap
    n_clusters = max(3, n // 5)
    try:
        from sklearn.cluster import KMeans as _KMeans  # type: ignore
        km = _KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = km.fit_predict(embs).tolist()
        return labels, "KMeans (sklearn)", n_clusters
    except ImportError:
        pass

    labels = _numpy_kmeans(embs, n_clusters).tolist()
    return labels, "KMeans (NumPy)", n_clusters


def _sample_cluster(
    cluster_rows: list,
    cluster_embs: np.ndarray,
    n_closest: int = 5,
    n_furthest: int = 5,
) -> list:
    """
    Sample from a single cluster by centroid distance.

    - n_closest: messages nearest the centroid (core / most representative).
    - n_furthest: messages furthest from the centroid (peripheral / edge cases).
    """
    n = len(cluster_rows)
    if n <= n_closest + n_furthest:
        return list(cluster_rows)

    centroid = cluster_embs.mean(axis=0)
    dist = np.linalg.norm(cluster_embs - centroid, axis=1)
    order = np.argsort(dist)  # ascending = closest first

    closest_idx  = list(order[:n_closest])
    furthest_idx = list(order[-(n_furthest):])

    selected = sorted(set(closest_idx) | set(furthest_idx))
    return [cluster_rows[i] for i in selected]


def _build_evidence_set(
    rows: list,
    embs: np.ndarray,
    max_evidence: int = 120,
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
    prompt_txt      = (body.get("prompt") or "").strip()
    retrieval_mode  = (body.get("retrieval_mode") or "cluster").strip()
    upload_ids      = (body.get("upload_ids") or "")
    min_words       = int(body.get("min_words") or 0)
    suno_team       = str(body.get("suno_team") or "all")
    sum_model       = (body.get("model") or "gpt-5.4").strip()

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

    # ── Shared state for pipeline log (initialised here so "all" mode can skip) ─
    candidate_rows: list = []
    _log_total_in_store: int = 0
    _log_overfetch_n: int = 0
    _log_vector_ok: bool = False
    _log_vector_err: str = ""
    use_pipeline: bool = False
    pipeline_stats: dict = {}
    _fallback_cluster: bool = False
    _fallback_n_rows: int = 0
    evidence: list = []

    if retrieval_mode == "all":
        # ── "All messages" mode: skip retrieval and clustering entirely ───────
        evidence = [dict(r) for r in db_rows]
        logger.info("summarize: mode=all — sending all %d filtered rows to LLM", n_filtered)

    else:
        # ── Step 2: Semantic retrieval from vector store ──────────────────────
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

        candidate_embs_raw: list = []

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
                    # Broad initial fetch — retrieve as many as useful so the
                    # similarity threshold can decide relevance, not a hard cap.
                    overfetch = min(total, max(n_filtered * 5, 2000))
                    results = col.query(
                        query_embeddings=[query_embedding], n_results=overfetch
                    )
                    result_ids   = results.get("ids",      [[]])[0]
                    result_dists = results.get("distances", [[]])[0]

                    # Intersect with metadata-filtered set
                    all_scored: list = []
                    for uid, dist in zip(result_ids, result_dists):
                        if uid in filtered_map:
                            score = round(1.0 - float(dist), 4) if dist is not None else 0.0
                            all_scored.append((score, uid))
                    all_scored.sort(key=lambda x: -x[0])

                    # Adaptive similarity threshold: keep top 70% by score.
                    # Falls back to at least _MIN_CANDIDATES so clustering is viable.
                    if all_scored:
                        scores_arr = np.array([s for s, _ in all_scored])
                        threshold  = float(np.percentile(scores_arr, 30))
                        above = [(s, uid) for s, uid in all_scored if s >= threshold]
                        scored = above if len(above) >= _MIN_CANDIDATES else all_scored[:_MIN_CANDIDATES]
                    else:
                        scored = []

                    top_uuids = [uid for _, uid in scored]

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

        # ── Steps 3-6: Dedup → Cluster → Sample → Assemble ───────────────────
        use_pipeline = len(candidate_rows) >= 10

        if use_pipeline:
            embs_array = np.array(candidate_embs_raw, dtype=np.float32)
            evidence, pipeline_stats = _build_evidence_set(candidate_rows, embs_array)
            if not evidence:
                use_pipeline = False

        if not use_pipeline:
            # Vector retrieval returned too few candidates — fetch stored embeddings
            # for the entire filtered set and run the full pipeline on those.
            import asyncio as _asyncio_fb
            import random as _random

            all_uuids   = list(filtered_map.keys())
            fetch_uuids = (
                _random.sample(all_uuids, 3000) if len(all_uuids) > 3000 else all_uuids
            )

            def _fetch_fallback_embs() -> dict:
                try:
                    col = active_collection()
                    result   = col.get(ids=fetch_uuids, include=["embeddings"])
                    emb_ids  = result.get("ids", [])
                    emb_vecs = result.get("embeddings", [])
                    return {
                        eid: evec
                        for eid, evec in zip(emb_ids, emb_vecs)
                        if evec is not None
                    }
                except Exception as exc:
                    logger.warning("summarize: fallback emb fetch failed (%s)", exc)
                    return {}

            _fb_loop    = _asyncio_fb.get_running_loop()
            all_emb_map: dict = await _fb_loop.run_in_executor(
                state.vector_executor, _fetch_fallback_embs
            )

            if all_emb_map:
                fb_rows: list = []
                fb_embs: list = []
                for row in db_rows:
                    uid = dict(row)["msg_uuid"] if not isinstance(row, dict) else row["msg_uuid"]
                    if uid in all_emb_map:
                        fb_rows.append(dict(row))
                        fb_embs.append(all_emb_map[uid])

                _fallback_n_rows = len(fb_rows)
                if _fallback_n_rows >= 10:
                    embs_array = np.array(fb_embs, dtype=np.float32)
                    evidence, pipeline_stats = _build_evidence_set(fb_rows, embs_array)
                    if evidence:
                        use_pipeline      = True
                        _fallback_cluster = True
                        logger.info(
                            "summarize: fallback cluster — %d/%d filtered rows have embeddings",
                            _fallback_n_rows, n_filtered,
                        )

        if not use_pipeline:
            logger.info(
                "summarize: no embeddings available — sending all %d filtered rows to LLM",
                n_filtered,
            )
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

    if retrieval_mode == "all":
        pipeline_log.append({
            "type": "log", "step": "sample", "label": "Mode: all messages",
            "msg": f"Clustering skipped — all {n_evidence:,} filtered messages sent to LLM",
        })
    else:
        if _log_vector_ok and _log_total_in_store > 0 and not _fallback_cluster:
            pipeline_log.append({
                "type": "log", "step": "retrieval", "label": "Vector retrieval",
                "msg": (
                    f"Fetched {_log_overfetch_n:,} from {_log_total_in_store:,} in store "
                    f"→ {len(candidate_rows):,} candidates above similarity threshold"
                ),
            })
        elif _fallback_cluster:
            pipeline_log.append({
                "type": "log", "step": "retrieval", "label": "Fallback retrieval",
                "msg": (
                    f"Too few vector candidates — fetched embeddings for "
                    f"{_fallback_n_rows:,} of {n_filtered:,} filtered messages"
                ),
            })
        elif not _log_vector_ok:
            pipeline_log.append({
                "type": "log", "step": "fallback", "label": "Vector fallback",
                "msg": (
                    f"Vector retrieval failed ({_log_vector_err}) — clustering all filtered messages"
                    if _log_vector_err else
                    "Embedding unavailable — clustering all filtered messages"
                ),
            })
        elif _log_total_in_store == 0:
            pipeline_log.append({
                "type": "log", "step": "fallback", "label": "Vector fallback",
                "msg": "Vector store is empty — clustering all filtered messages",
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
    prompt_txt = (body.get("prompt") or "").strip()
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
                # Same adaptive approach as the main summarize endpoint,
                # but capped a bit lower (follow-up is focused, not a full sweep).
                overfetch_n = min(total, max(n_filtered * 4, 1000))
                results      = col.query(
                    query_embeddings=[query_embedding], n_results=overfetch_n
                )
                result_ids   = results.get("ids",      [[]])[0]
                result_dists = results.get("distances", [[]])[0]

                all_scored: list = []
                for uid, dist in zip(result_ids, result_dists):
                    if uid in filtered_map:
                        score = round(1.0 - float(dist), 4) if dist is not None else 0.0
                        all_scored.append((score, uid))
                all_scored.sort(key=lambda x: -x[0])

                if all_scored:
                    scores_arr = np.array([s for s, _ in all_scored])
                    threshold  = float(np.percentile(scores_arr, 30))
                    above = [(s, uid) for s, uid in all_scored if s >= threshold]
                    scored = above if len(above) >= _MIN_CANDIDATES else all_scored[:_MIN_CANDIDATES]
                else:
                    scored = []

                top_uuids = [uid for _, uid in scored]

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
    use_pipeline = len(candidate_rows) >= 10
    fu_pipeline_stats: dict = {}

    if use_pipeline:
        embs_array = np.array(candidate_embs_raw, dtype=np.float32)
        evidence, fu_pipeline_stats = _build_evidence_set(candidate_rows, embs_array, max_evidence=80)
        if not evidence:
            use_pipeline = False

    if not use_pipeline:
        # Fetch embeddings for all filtered rows and cluster those instead.
        import asyncio as _asyncio_fu_fb
        import random as _random_fu

        all_uuids_fu  = list(filtered_map.keys())
        fetch_uuids_fu = (
            _random_fu.sample(all_uuids_fu, 2000)
            if len(all_uuids_fu) > 2000 else all_uuids_fu
        )

        def _fetch_fu_fallback_embs() -> dict:
            try:
                col = active_collection()
                result   = col.get(ids=fetch_uuids_fu, include=["embeddings"])
                emb_ids  = result.get("ids", [])
                emb_vecs = result.get("embeddings", [])
                return {eid: evec for eid, evec in zip(emb_ids, emb_vecs) if evec is not None}
            except Exception as exc:
                logger.warning("summarize/followup: fallback emb fetch failed (%s)", exc)
                return {}

        _fu_fb_loop   = _asyncio_fu_fb.get_running_loop()
        fu_emb_map: dict = await _fu_fb_loop.run_in_executor(
            state.vector_executor, _fetch_fu_fallback_embs
        )

        if fu_emb_map:
            fb_rows_fu: list = []
            fb_embs_fu: list = []
            for row in db_rows:
                uid = dict(row)["msg_uuid"] if not isinstance(row, dict) else row["msg_uuid"]
                if uid in fu_emb_map:
                    fb_rows_fu.append(dict(row))
                    fb_embs_fu.append(fu_emb_map[uid])

            if len(fb_rows_fu) >= 10:
                embs_array = np.array(fb_embs_fu, dtype=np.float32)
                evidence, fu_pipeline_stats = _build_evidence_set(
                    fb_rows_fu, embs_array, max_evidence=80
                )
                if evidence:
                    use_pipeline = True
                    logger.info(
                        "summarize/followup: fallback cluster — %d/%d rows with embeddings",
                        len(fb_rows_fu), n_filtered,
                    )

    if not use_pipeline:
        logger.info(
            "summarize/followup: no embeddings — sending first 80 filtered rows to LLM",
            )
        evidence = [dict(r) for r in db_rows[:80]]

    n_evidence = len(evidence)

    # ── Step 7: Build prompt with conversation history + evidence, then stream ─
    evidence_context = "\n".join(
        f"[{r['username']} | {r['date']}]: {r['content']}"
        for r in evidence
    )

    # ── Separate initial summary from subsequent Q&A turns ───────────────────
    # history[0] is the initial summary (assistant turn); the rest are Q&A pairs.
    # We embed the initial summary explicitly in the system prompt so the model
    # treats it as authoritative context, not just an anonymous prior message.
    initial_summary = ""
    qa_history: list = history
    if history and history[0].get("role") == "assistant":
        initial_summary = (history[0].get("content") or "").strip()
        qa_history = history[1:]

    is_o_model = sum_model.startswith("o")
    sys_role   = "developer" if is_o_model else "system"

    system_parts: list = [
        "You are an expert analyst for the Suno AI Discord community. "
        "The user generated a Hybrid Summary and is asking follow-up questions. "
        "Answer using ALL THREE sources of context below:\n"
        "  1. RETRIEVED EVIDENCE — fresh quotes retrieved specifically for this question.\n"
        "  2. INITIAL SUMMARY — the full summary already presented to the user.\n"
        "  3. PRIOR Q&A — any follow-up questions and answers already exchanged.\n"
        "Be precise. Cite usernames and dates where relevant. "
        "Respond in well-structured Markdown.",
    ]
    if prompt_txt:
        system_parts.append(f"\nORIGINAL CUSTOM INSTRUCTIONS:\n{prompt_txt}")
    if initial_summary:
        system_parts.append(f"\nINITIAL SUMMARY:\n{initial_summary}")
    system_parts.append(
        f"\nRETRIEVED EVIDENCE FOR THIS QUESTION ({n_evidence} messages):\n{evidence_context}"
    )
    system_content = "\n".join(system_parts)

    msgs: list = [{"role": sys_role, "content": system_content}]
    # Append prior Q&A turns so the model can see the full conversation thread.
    for turn in qa_history[-20:]:
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


# ── /api/user-profile ─────────────────────────────────────────────────────────

@router.post("/api/user-profile")
async def user_profile_endpoint(request: Request):
    """
    Analyse a specific user's messages to identify persona, attitude evolution,
    entry/exit dates, and other customisable attributes.

    Request body:
      profile_username – the username to analyse (required)
      prompt           – optional focus prompt
      retrieval_mode   – "cluster" (default) or "all"
      date_from, date_to, upload_ids, min_words, suno_team, model
    """
    body             = await request.json()
    profile_username = (body.get("profile_username") or "").strip()
    prompt_txt       = (body.get("prompt") or "").strip()
    retrieval_mode   = (body.get("retrieval_mode") or "cluster").strip()
    date_from        = (body.get("date_from") or "").strip()
    date_to          = (body.get("date_to") or "").strip()
    upload_ids       = (body.get("upload_ids") or "")
    min_words        = int(body.get("min_words") or 0)
    suno_team        = str(body.get("suno_team") or "all")
    sum_model        = (body.get("model") or "gpt-5.4").strip()

    if not profile_username:
        raise HTTPException(400, "profile_username is required.")
    if not state.openai_client:
        raise HTTPException(400, "OpenAI API key not set — add it in Settings.")
    if sum_model not in VALID_CHAT_MODELS:
        raise HTTPException(400, f"Unknown model '{sum_model}'.")

    uid_list              = _parse_upload_ids(upload_ids)
    uid_sql, uid_params   = _sql_upload_ids_clause(uid_list)
    date_sql, date_params = sql_date_clauses(date_from or None, date_to or None)
    words_sql, words_params = sql_min_words_clause(min_words)

    # ── Step 1: Fetch all messages by this user ────────────────────────────────
    params: list = []
    sql = "SELECT msg_uuid, username, date, content FROM messages WHERE 1=1"
    sql += " AND LOWER(username) = LOWER(?)"
    params.append(profile_username)
    sql += uid_sql
    params.extend(uid_params)
    sql += _suno_sql(suno_team) + date_sql + words_sql
    params.extend(date_params + words_params)
    sql += " ORDER BY date, row_index"

    conn = get_db()
    db_rows = conn.execute(sql, params).fetchall()
    conn.close()

    if not db_rows:
        raise HTTPException(404, f"No messages found for user '{profile_username}' matching those filters.")

    filtered_map: dict = {r["msg_uuid"]: dict(r) for r in db_rows}
    n_filtered = len(filtered_map)

    # Entry / exit metadata
    first_row  = dict(db_rows[0])
    last_row   = dict(db_rows[-1])
    entry_date = first_row.get("date", "")
    exit_date  = last_row.get("date", "")

    # ── Shared state for pipeline log ─────────────────────────────────────────
    candidate_rows: list = []
    _log_total_in_store: int = 0
    _log_overfetch_n: int = 0
    _log_vector_ok: bool = False
    _log_vector_err: str = ""
    use_pipeline: bool = False
    pipeline_stats: dict = {}
    _fallback_cluster: bool = False
    _fallback_n_rows: int = 0
    evidence: list = []

    if retrieval_mode == "all":
        evidence = [dict(r) for r in db_rows]
        logger.info("user-profile: mode=all — using all %d rows for '%s'", n_filtered, profile_username)

    else:
        # ── Step 2: Semantic retrieval ─────────────────────────────────────────
        retrieval_query = (
            prompt_txt
            or f"attitude, opinions, concerns, feedback, and persona of {profile_username} regarding Suno AI"
        )

        query_embedding: Optional[list] = None
        try:
            query_embedding = (await embed_texts_async([retrieval_query]))[0]
        except Exception as exc:
            logger.warning("user-profile: failed to embed query (%s)", exc)

        candidate_embs_raw: list = []

        if query_embedding is not None:
            import asyncio as _asyncio_up
            _up_loop = _asyncio_up.get_running_loop()

            def _up_vector_retrieval_sync() -> tuple:
                try:
                    col = active_collection()
                    total = col.count()
                    if total == 0:
                        return total, 0, [], [], True, ""
                    overfetch = min(total, max(n_filtered * 5, 2000))
                    results = col.query(
                        query_embeddings=[query_embedding], n_results=overfetch
                    )
                    result_ids   = results.get("ids",      [[]])[0]
                    result_dists = results.get("distances", [[]])[0]

                    all_scored: list = []
                    for uid, dist in zip(result_ids, result_dists):
                        if uid in filtered_map:
                            score = round(1.0 - float(dist), 4) if dist is not None else 0.0
                            all_scored.append((score, uid))
                    all_scored.sort(key=lambda x: -x[0])

                    if all_scored:
                        scores_arr = np.array([s for s, _ in all_scored])
                        threshold  = float(np.percentile(scores_arr, 30))
                        above = [(s, uid) for s, uid in all_scored if s >= threshold]
                        scored = above if len(above) >= _MIN_CANDIDATES else all_scored[:_MIN_CANDIDATES]
                    else:
                        scored = []

                    top_uuids = [uid for _, uid in scored]

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
                    logger.error("user-profile: vector retrieval error (%s)", err_msg, exc_info=True)
                    return 0, 0, [], [], False, err_msg

            (
                _log_total_in_store,
                _log_overfetch_n,
                candidate_rows,
                candidate_embs_raw,
                _log_vector_ok,
                _log_vector_err,
            ) = await _up_loop.run_in_executor(state.vector_executor, _up_vector_retrieval_sync)

        # ── Steps 3-6: Dedup → Cluster → Sample → Assemble ───────────────────
        use_pipeline = len(candidate_rows) >= 10

        if use_pipeline:
            embs_array = np.array(candidate_embs_raw, dtype=np.float32)
            evidence, pipeline_stats = _build_evidence_set(candidate_rows, embs_array)
            if not evidence:
                use_pipeline = False

        if not use_pipeline:
            import asyncio as _asyncio_fb2
            import random as _random2

            all_uuids   = list(filtered_map.keys())
            fetch_uuids = (
                _random2.sample(all_uuids, 3000) if len(all_uuids) > 3000 else all_uuids
            )

            def _up_fetch_fallback_embs() -> dict:
                try:
                    col = active_collection()
                    result   = col.get(ids=fetch_uuids, include=["embeddings"])
                    emb_ids  = result.get("ids", [])
                    emb_vecs = result.get("embeddings", [])
                    return {
                        eid: evec
                        for eid, evec in zip(emb_ids, emb_vecs)
                        if evec is not None
                    }
                except Exception as exc:
                    logger.warning("user-profile: fallback emb fetch failed (%s)", exc)
                    return {}

            _fb2_loop   = _asyncio_fb2.get_running_loop()
            all_emb_map: dict = await _fb2_loop.run_in_executor(
                state.vector_executor, _up_fetch_fallback_embs
            )

            if all_emb_map:
                fb_rows: list = []
                fb_embs: list = []
                for row in db_rows:
                    uid = dict(row)["msg_uuid"] if not isinstance(row, dict) else row["msg_uuid"]
                    if uid in all_emb_map:
                        fb_rows.append(dict(row))
                        fb_embs.append(all_emb_map[uid])

                _fallback_n_rows = len(fb_rows)
                if _fallback_n_rows >= 10:
                    embs_array = np.array(fb_embs, dtype=np.float32)
                    evidence, pipeline_stats = _build_evidence_set(fb_rows, embs_array)
                    if evidence:
                        use_pipeline      = True
                        _fallback_cluster = True

        if not use_pipeline:
            evidence = [dict(r) for r in db_rows]

    # ── Step 7: Build LLM prompt ──────────────────────────────────────────────
    conv = "\n".join(
        f"[{r['date']}]: {r['content']}"
        for r in evidence
    )
    n_evidence = len(evidence)

    default_prompt = (
        f"Analyse the messages below written by Discord user **{profile_username}** in the Suno AI community server.\n\n"
        "MANDATORY STRUCTURE (strictly follow this Markdown layout):\n\n"
        f"## User Profile: {profile_username}\n\n"
        "### Entry & Exit\n"
        f"- **First message:** {entry_date}\n"
        f"- **Last message:** {exit_date}\n"
        f"- **Total messages analysed:** {n_filtered}\n\n"
        "### Persona\n"
        "Describe this user's overall character, communication style, and role in the community "
        "(e.g. power user, casual listener, critic, advocate, developer).\n\n"
        "### Evolution of Attitude & Concerns\n"
        "Describe how this user's attitude toward Suno (Bark / Chirp / the platform) changed over time. "
        "Use a chronological narrative with approximate time references. Note any inflection points "
        "(e.g. excitement → frustration → departure, or initial scepticism → advocacy).\n\n"
        "### Key Topics & Concerns\n"
        "- Bullet list of recurring themes this user raised.\n\n"
        "### Notable Quotes\n"
        "> Include 2-5 representative verbatim or near-verbatim quotes that best capture their voice, "
        "with approximate dates where possible.\n\n"
        "### Summary Assessment\n"
        "One short paragraph summarising who this user is and their relationship with Suno AI.\n\n"
        "---\n"
        "Use **bold** for important conclusions. Do NOT write plain prose paragraphs outside the sections above."
    )

    user_prompt = prompt_txt or default_prompt
    full = f"{user_prompt}\n\nMESSAGES ({n_evidence} selected from {n_filtered} total):\n{conv}"

    is_o_model = sum_model.startswith("o")
    sys_role   = "developer" if is_o_model else "system"

    # ── Build pipeline log events ─────────────────────────────────────────────
    pipeline_log: list = []
    pipeline_log.append({
        "type": "log", "step": "filter", "label": "User filter",
        "msg": f"{n_filtered:,} messages by '{profile_username}' matched filters",
    })
    pipeline_log.append({
        "type": "log", "step": "meta", "label": "User span",
        "msg": f"Entry: {entry_date}  \u2192  Exit: {exit_date}",
    })

    if retrieval_mode == "all":
        pipeline_log.append({
            "type": "log", "step": "sample", "label": "Mode: all messages",
            "msg": f"Clustering skipped \u2014 all {n_evidence:,} messages sent to LLM",
        })
    else:
        if _log_vector_ok and _log_total_in_store > 0 and not _fallback_cluster:
            pipeline_log.append({
                "type": "log", "step": "retrieval", "label": "Vector retrieval",
                "msg": (
                    f"Fetched {_log_overfetch_n:,} from {_log_total_in_store:,} in store "
                    f"\u2192 {len(candidate_rows):,} candidates above similarity threshold"
                ),
            })
        elif _fallback_cluster:
            pipeline_log.append({
                "type": "log", "step": "retrieval", "label": "Fallback retrieval",
                "msg": (
                    f"Too few vector candidates \u2014 fetched embeddings for "
                    f"{_fallback_n_rows:,} of {n_filtered:,} messages"
                ),
            })
        elif not _log_vector_ok:
            pipeline_log.append({
                "type": "log", "step": "fallback", "label": "Vector fallback",
                "msg": (
                    f"Vector retrieval failed ({_log_vector_err})"
                    if _log_vector_err else
                    "Embedding unavailable \u2014 clustering all messages"
                ),
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
                "msg": f"Too few candidates \u2014 using {n_evidence:,} messages directly",
            })

    pipeline_log.append({
        "type": "log", "step": "llm", "label": "LLM generation",
        "msg": f"Analysing {n_evidence:,} messages with {sum_model}\u2026",
    })

    async def up_generate():
        for entry in pipeline_log:
            yield f"data: {json.dumps(entry)}\n\n"
        try:
            stream = state.openai_client.chat.completions.create(
                model=sum_model,
                messages=[
                    {
                        "role": sys_role,
                        "content": (
                            "You are an expert analyst profiling Discord users in the Suno AI community. "
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
            logger.error("user-profile generate() error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        up_generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


# ── /api/user-profile/followup ────────────────────────────────────────────────

@router.post("/api/user-profile/followup")
async def user_profile_followup_endpoint(request: Request):
    """
    Follow-up Q&A within a User Profile session.

    Request body:
      question         – the follow-up question (required)
      history          – [{role, content}] conversation so far (initial profile first)
      profile_username – username that was profiled
      prompt           – the original focus prompt (if any)
      date_from, date_to, upload_ids, min_words, suno_team, model
    """
    body             = await request.json()
    question         = (body.get("question") or "").strip()
    history          = body.get("history") or []
    profile_username = (body.get("profile_username") or "").strip()
    prompt_txt       = (body.get("prompt") or "").strip()
    date_from        = (body.get("date_from") or "").strip()
    date_to          = (body.get("date_to") or "").strip()
    upload_ids       = (body.get("upload_ids") or "")
    min_words        = int(body.get("min_words") or 0)
    suno_team        = str(body.get("suno_team") or "all")
    sum_model        = (body.get("model") or "gpt-5.4").strip()

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

    params: list = []
    sql = "SELECT msg_uuid, username, date, content FROM messages WHERE 1=1"
    if profile_username:
        sql += " AND LOWER(username) = LOWER(?)"
        params.append(profile_username)
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

    # ── Semantic retrieval using the follow-up QUESTION as query ──────────────
    query_embedding: Optional[list] = None
    try:
        query_embedding = (await embed_texts_async([question]))[0]
    except Exception as exc:
        logger.warning("user-profile/followup: failed to embed question (%s)", exc)

    candidate_rows: list = []
    candidate_embs_raw: list = []

    if query_embedding is not None:
        import asyncio as _asyncio_upfu

        _upfu_loop = _asyncio_upfu.get_running_loop()

        def _upfu_retrieval_sync() -> tuple:
            try:
                col = active_collection()
                total = col.count()
                if total == 0:
                    return total, 0, [], [], True, ""
                overfetch = min(total, max(n_filtered * 5, 2000))
                results = col.query(
                    query_embeddings=[query_embedding], n_results=overfetch
                )
                result_ids   = results.get("ids",      [[]])[0]
                result_dists = results.get("distances", [[]])[0]

                all_scored: list = []
                for uid, dist in zip(result_ids, result_dists):
                    if uid in filtered_map:
                        score = round(1.0 - float(dist), 4) if dist is not None else 0.0
                        all_scored.append((score, uid))
                all_scored.sort(key=lambda x: -x[0])

                if all_scored:
                    scores_arr = np.array([s for s, _ in all_scored])
                    threshold  = float(np.percentile(scores_arr, 30))
                    above = [(s, uid) for s, uid in all_scored if s >= threshold]
                    scored = above if len(above) >= _MIN_CANDIDATES else all_scored[:_MIN_CANDIDATES]
                else:
                    scored = []

                top_uuids = [uid for _, uid in scored]

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
                logger.error("user-profile/followup: retrieval error (%s)", err_msg, exc_info=True)
                return 0, 0, [], [], False, err_msg

        (
            _fu_total,
            _fu_overfetch,
            candidate_rows,
            candidate_embs_raw,
            _fu_ok,
            _fu_err,
        ) = await _upfu_loop.run_in_executor(state.vector_executor, _upfu_retrieval_sync)

    fu_use_pipeline = len(candidate_rows) >= 10
    fu_pipeline_stats: dict = {}

    if fu_use_pipeline:
        embs_array = np.array(candidate_embs_raw, dtype=np.float32)
        fu_evidence, fu_pipeline_stats = _build_evidence_set(candidate_rows, embs_array)
        if not fu_evidence:
            fu_use_pipeline = False
            fu_evidence = [dict(r) for r in db_rows[:200]]
    else:
        fu_evidence = [dict(r) for r in db_rows[:200]]

    n_evidence = len(fu_evidence)
    conv = "\n".join(
        f"[{r['date']}]: {r['content']}"
        for r in fu_evidence
    )

    # Build system message with initial profile as context
    initial_profile = ""
    qa_history: list = []
    if history and history[0].get("role") == "assistant":
        initial_profile = history[0].get("content", "")
        qa_history = history[1:]
    elif history and history[0].get("role") == "user":
        qa_history = history

    is_o_model = sum_model.startswith("o")
    sys_role   = "developer" if is_o_model else "system"

    system_content = (
        "You are an expert analyst answering follow-up questions about a specific Discord user's profile. "
        "Ground your answers in the evidence messages provided AND the initial profile analysis. "
        "Respond in well-structured Markdown.\n\n"
        "You have access to three sources of context:\n"
        "1. INITIAL PROFILE: the full profile analysis generated in this session\n"
        "2. EVIDENCE MESSAGES: fresh semantic matches for this specific question\n"
        "3. Q&A HISTORY: prior follow-up questions and answers in this session\n"
    )
    if initial_profile:
        system_content += f"\n\nINITIAL PROFILE:\n{initial_profile}"
    if prompt_txt:
        system_content += f"\n\nOriginal focus: {prompt_txt}"

    msgs: list = [{"role": sys_role, "content": system_content}]

    for turn in qa_history:
        role    = turn.get("role", "")
        content = turn.get("content", "")
        if role in {"user", "assistant"} and content:
            msgs.append({"role": role, "content": content})

    evidence_block = (
        f"\n\nEVIDENCE MESSAGES ({n_evidence} messages from {n_filtered} filtered):\n{conv}"
    )
    msgs.append({"role": "user", "content": question + evidence_block})

    # ── Follow-up pipeline log ────────────────────────────────────────────────
    fu_log: list = [
        {"type": "log", "step": "filter", "label": "User filter",
         "msg": f"{n_filtered:,} messages by '{profile_username}' in filtered pool"},
        {"type": "log", "step": "retrieval", "label": "Vector retrieval",
         "msg": f"{len(candidate_rows):,} semantic candidates retrieved for follow-up"},
    ]
    if fu_use_pipeline and fu_pipeline_stats:
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
                   "msg": f"Answering with {n_evidence:,} evidence messages via {sum_model}\u2026"})

    async def up_fu_generate():
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
            logger.error("user-profile/followup generate() error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        up_fu_generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


# ── /api/summarize-results ────────────────────────────────────────────────────

@router.post("/api/summarize-results")
async def summarize_results_endpoint(request: Request):
    """
    Summarise messages passed directly from the browser's currentResults array.
    No DB query — the client sends the messages it already has.

    Request body:
      messages        – [{msg_uuid, username, date, content}, ...]  (required; no hard cap — auto-clusters if payload exceeds ~90 k tokens)
      prompt          – optional custom instructions
      model           – OpenAI model ID
      retrieval_mode  – "cluster" (default) | "all"
    """
    import asyncio as _sr_asyncio

    body           = await request.json()
    messages       = body.get("messages") or []
    prompt_txt     = (body.get("prompt") or "").strip()
    sum_model      = (body.get("model") or "gpt-5.4").strip()
    retrieval_mode = (body.get("retrieval_mode") or "cluster").strip()

    if not state.openai_client:
        raise HTTPException(400, "OpenAI API key not set — add it in Settings.")
    if sum_model not in VALID_CHAT_MODELS:
        raise HTTPException(400, f"Unknown model '{sum_model}'.")
    if not messages:
        raise HTTPException(400, "No messages provided.")

    messages.sort(key=lambda r: r.get("date") or "")
    n_input = len(messages)

    # Rough token estimate: 1 token ≈ 4 chars. Reserve ~12 k tokens for system
    # prompt + response headroom. Threshold: 90 k content tokens → ~360 k chars.
    _TOKEN_CHAR_LIMIT = 360_000

    def _est_chars(rows: list) -> int:
        return sum(
            len(r.get("username") or "") + len(r.get("date") or "") + len(r.get("content") or "") + 10
            for r in rows
        )

    async def _run_cluster_pipeline(rows: list, loop, label_prefix: str = "") -> tuple[list, list]:
        """Run embed → dedup → cluster → sample. Returns (evidence, log_entries)."""
        logs: list = []
        uuids      = [r.get("msg_uuid") for r in rows if r.get("msg_uuid")]
        uuid_to_row: dict = {r["msg_uuid"]: r for r in rows if r.get("msg_uuid")}

        if len(uuids) < 10:
            logs.append({
                "type": "log", "step": "fallback", "label": f"{label_prefix}Cluster skipped",
                "msg": f"Too few messages with UUIDs ({len(uuids)}) — sending all {len(rows):,}",
            })
            return rows, logs

        def _fetch_embs() -> dict:
            try:
                col = active_collection()
                if col is None:
                    return {}
                result   = col.get(ids=uuids, include=["embeddings"])
                emb_ids  = result.get("ids", [])
                emb_vecs = result.get("embeddings", [])
                return {eid: evec for eid, evec in zip(emb_ids, emb_vecs) if evec is not None}
            except Exception as exc:
                logger.warning("summarize-results: emb fetch failed (%s)", exc)
                return {}

        emb_map: dict     = await loop.run_in_executor(state.vector_executor, _fetch_embs)
        rows_with_emb     = [uuid_to_row[uid] for uid in uuids if uid in emb_map]
        embs_raw          = [emb_map[uid]      for uid in uuids if uid in emb_map]

        logs.append({
            "type": "log", "step": "retrieval", "label": f"{label_prefix}Embedding lookup",
            "msg": f"Found embeddings for {len(rows_with_emb):,} of {len(rows):,} messages",
        })

        if len(rows_with_emb) < 10:
            logs.append({
                "type": "log", "step": "fallback", "label": f"{label_prefix}Cluster fallback",
                "msg": f"Too few embeddings ({len(rows_with_emb)}) — sending all {len(rows):,}",
            })
            return rows, logs

        embs_array = np.array(embs_raw, dtype=np.float32)
        ev, ps     = _build_evidence_set(rows_with_emb, embs_array)

        if not ev:
            logs.append({
                "type": "log", "step": "fallback", "label": f"{label_prefix}Cluster fallback",
                "msg": f"Clustering produced no output — sending all {len(rows):,}",
            })
            return rows, logs

        n_dupes = ps.get("n_dupes_removed", 0)
        n_dedup = ps.get("n_after_dedup", len(rows_with_emb))
        algo    = ps.get("algorithm", "unknown")
        n_clust = ps.get("n_clusters", 0)
        logs.append({
            "type": "log", "step": "dedup", "label": f"{label_prefix}Deduplication",
            "msg": f"{n_dupes:,} near-duplicate{'s' if n_dupes != 1 else ''} removed ({n_dedup:,} remain)",
        })
        if algo != "none":
            logs.append({
                "type": "log", "step": "cluster", "label": f"{label_prefix}Clustering",
                "msg": f"{n_clust} cluster{'s' if n_clust != 1 else ''} via {algo}",
            })
        logs.append({
            "type": "log", "step": "sample", "label": f"{label_prefix}Sampling",
            "msg": f"Sampled down to {len(ev):,} representative messages",
        })
        return ev, logs

    pipeline_log: list = []
    if prompt_txt:
        pipeline_log.append({
            "type": "log", "step": "instruction", "label": "Custom instructions",
            "msg": prompt_txt,
        })
    pipeline_log.append({
        "type": "log", "step": "filter", "label": "Input",
        "msg": f"{n_input:,} search result messages passed from browser",
    })

    _sr_loop = _sr_asyncio.get_running_loop()

    # ── Cluster & Sample pipeline ─────────────────────────────────────────────
    evidence: list = messages

    if retrieval_mode == "cluster":
        evidence, cluster_logs = await _run_cluster_pipeline(messages, _sr_loop)
        pipeline_log.extend(cluster_logs)
    else:
        pipeline_log.append({
            "type": "log", "step": "sample", "label": "Mode: all messages",
            "msg": f"Clustering skipped — all {n_input:,} messages sent to LLM",
        })

    # ── Token-limit safety net ────────────────────────────────────────────────
    # If the evidence still exceeds ~90 k tokens worth of characters, auto-fall
    # back to cluster+sample regardless of the requested mode.
    if _est_chars(evidence) > _TOKEN_CHAR_LIMIT:
        est_tokens = _est_chars(evidence) // 4
        pipeline_log.append({
            "type": "log", "step": "fallback", "label": "Token limit — auto cluster",
            "msg": (
                f"Estimated payload (~{est_tokens:,} tokens) exceeds safe context limit. "
                f"Auto-switching to cluster+sample."
            ),
        })
        sampled, cluster_logs = await _run_cluster_pipeline(evidence, _sr_loop, label_prefix="Auto-")
        pipeline_log.extend(cluster_logs)
        # If cluster still produces a too-large payload, hard-truncate as last resort
        if _est_chars(sampled) > _TOKEN_CHAR_LIMIT:
            before = len(sampled)
            while sampled and _est_chars(sampled) > _TOKEN_CHAR_LIMIT:
                sampled = sampled[:int(len(sampled) * 0.8)]
            pipeline_log.append({
                "type": "log", "step": "fallback", "label": "Hard truncation",
                "msg": f"Cluster output still too large — truncated from {before:,} to {len(sampled):,} messages",
            })
        evidence = sampled

    n = len(evidence)
    pipeline_log.append({
        "type": "log", "step": "llm", "label": "LLM generation",
        "msg": f"Summarising {n:,} messages with {sum_model}\u2026",
    })

    conv = "\n".join(
        f"[{r.get('username','?')} | {r.get('date','')}]: {r.get('content','')}"
        for r in evidence
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
    full = f"{user_prompt}\n\nCONVERSATION ({n} messages):\n{conv}"

    is_o_model = sum_model.startswith("o")
    sys_role   = "developer" if is_o_model else "system"

    async def sr_generate():
        for entry in pipeline_log:
            yield f"data: {json.dumps(entry)}\n\n"
        try:
            stream = state.openai_client.chat.completions.create(
                model=sum_model,
                messages=[
                    {
                        "role": sys_role,
                        "content": (
                            "You are an expert analyst summarising Discord conversations "
                            "from the Suno AI community. "
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
            logger.error("summarize-results generate() error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sr_generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


# ── /api/summarize-results/followup ──────────────────────────────────────────

@router.post("/api/summarize-results/followup")
async def summarize_results_followup_endpoint(request: Request):
    """
    Stateless follow-up Q&A for the Summarize Results feature.
    All context is in history — no DB or vector retrieval needed.

    Request body:
      question  – follow-up question (required)
      history   – [{role, content}] — index 0 is the initial summary (assistant)
      model     – OpenAI model ID
    """
    body      = await request.json()
    question  = (body.get("question") or "").strip()
    history   = body.get("history") or []
    sum_model = (body.get("model") or "gpt-5.4").strip()

    if not question:
        raise HTTPException(400, "Empty question.")
    if not state.openai_client:
        raise HTTPException(400, "OpenAI API key not set — add it in Settings.")
    if sum_model not in VALID_CHAT_MODELS:
        raise HTTPException(400, f"Unknown model '{sum_model}'.")

    is_o_model = sum_model.startswith("o")
    sys_role   = "developer" if is_o_model else "system"

    initial_summary = ""
    qa_history = history
    if history and history[0].get("role") == "assistant":
        initial_summary = (history[0].get("content") or "").strip()
        qa_history = history[1:]

    system_content = (
        "You are an expert analyst answering follow-up questions about a Discord conversation summary. "
        "Answer based on the initial summary provided. Respond in well-structured Markdown.\n\n"
        + (f"INITIAL SUMMARY:\n{initial_summary}" if initial_summary else "")
    )

    msgs: list = [{"role": sys_role, "content": system_content}]
    for turn in qa_history[-20:]:
        role    = turn.get("role", "")
        content = (turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": question})

    fu_log = [
        {"type": "log", "step": "llm", "label": "LLM generation",
         "msg": f"Answering follow-up with {sum_model}\u2026"},
    ]

    async def sr_fu_generate():
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
            logger.error("summarize-results/followup generate() error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sr_fu_generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )
