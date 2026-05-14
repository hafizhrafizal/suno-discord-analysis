"""
routers/search.py — message search endpoints.

  GET  /api/search/username
  GET  /api/search/keyword
  GET  /api/search/range
  GET  /api/search/semantic
  GET  /api/search/users-in-range
  GET  /api/search/user-messages
  POST /api/search/bulk-context
"""

import logging
from datetime import date as _date, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

import state
from config import EMBEDDING_MODELS
from database import get_db
from embeddings import active_collection, embed_texts_async
from sql_helpers import (
    _parse_upload_ids,
    _sql_upload_ids_clause,
    _suno_sql,
    date_in_range,
    is_suno_team_member,
    keyword_search,
    sql_date_clauses,
    sql_min_words_clause,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/search/username")
async def search_by_username(
    username:   str,
    upload_ids: Optional[str] = None,
    date_from:  Optional[str] = None,
    date_to:    Optional[str] = None,
    suno_team:  str           = "all",
    min_words:  int           = 0,
    limit:      int           = 200,
):
    uid_list             = _parse_upload_ids(upload_ids)
    uid_sql, uid_params  = _sql_upload_ids_clause(uid_list)
    date_sql, date_params = sql_date_clauses(date_from, date_to)
    words_sql, words_params = sql_min_words_clause(min_words)
    sql = (
        "SELECT * FROM messages WHERE LOWER(username) LIKE LOWER(?)"
        + uid_sql + _suno_sql(suno_team) + date_sql + words_sql
        + " ORDER BY date, row_index LIMIT ?"
    )
    conn = get_db()
    rows = conn.execute(
        sql,
        [f"%{username}%"] + uid_params + date_params + words_params + [limit],
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/api/search/keyword")
async def search_by_keyword(
    keyword:    str,
    upload_ids: Optional[str] = None,
    username:   Optional[str] = None,
    date_from:  Optional[str] = None,
    date_to:    Optional[str] = None,
    suno_team:  str           = "all",
    min_words:  int           = 0,
    limit:      int           = 200,
    match_type: str           = "fuzzy",
):
    return await keyword_search(
        keyword=keyword,
        upload_ids=upload_ids,
        username=username,
        date_from=date_from,
        date_to=date_to,
        suno_team=suno_team,
        min_words=min_words,
        limit=limit,
        match_type=match_type,
    )


@router.get("/api/search/range")
async def search_by_range(
    upload_ids: Optional[str] = None,
    username:   Optional[str] = None,
    date_from:  Optional[str] = None,
    date_to:    Optional[str] = None,
    suno_team:  str           = "all",
    min_words:  int           = 0,
    limit:      Optional[int] = None,
):
    uid_list              = _parse_upload_ids(upload_ids)
    uid_sql, uid_params   = _sql_upload_ids_clause(uid_list)
    date_sql, date_params = sql_date_clauses(date_from, date_to)
    words_sql, words_params = sql_min_words_clause(min_words)

    params: list = []
    sql = "SELECT * FROM messages WHERE 1=1"
    if username:
        sql += " AND LOWER(username) LIKE LOWER(?)"
        params.append(f"%{username}%")
    sql += uid_sql
    params.extend(uid_params)
    sql += _suno_sql(suno_team) + date_sql + words_sql
    params.extend(date_params + words_params)
    sql += " ORDER BY date, row_index"
    if limit is not None and limit > 0:
        sql += " LIMIT ?"
        params.append(limit)

    conn = get_db()
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Direct embedding fetch threshold: if the SQL-filtered set is this size or
# smaller, fetch embeddings by ID and score directly (exact, no missed matches).
# Above this, fall back to a broad ANN query + filter intersection.
# 5000 is the crossover where HTTP/batch overhead of direct fetch starts to
# outweigh the ANN index speedup. Below it, Strategy A is both faster and exact.
_SEMANTIC_DIRECT_THRESHOLD = 5000


@router.get("/api/search/semantic")
async def search_semantic(
    query:      str,
    upload_ids: Optional[str] = None,
    n_results:  int           = 20,
    username:   Optional[str] = None,
    date_from:  Optional[str] = None,
    date_to:    Optional[str] = None,
    suno_team:  str           = "all",
    min_words:  int           = 0,
):
    import asyncio as _asyncio
    import numpy as _np

    uid_list = _parse_upload_ids(upload_ids)
    col      = active_collection()
    if col is None:
        raise HTTPException(400, "Vector store is not initialised — check server logs.")

    import httpx as _httpx

    _sem_loop = _asyncio.get_running_loop()
    try:
        total = await _sem_loop.run_in_executor(state.vector_executor, col.count)
    except (_httpx.ReadTimeout, _httpx.ConnectTimeout, _httpx.TimeoutException, Exception) as exc:
        if "timeout" in str(exc).lower() or isinstance(exc, (_httpx.ReadTimeout, _httpx.ConnectTimeout)):
            raise HTTPException(
                504,
                "Vector store timed out while counting documents. "
                "The Qdrant server may be overloaded — please retry in a moment.",
            )
        raise

    if total == 0:
        raise HTTPException(
            400,
            f"No messages are embedded with the current model "
            f"({EMBEDDING_MODELS[state.current_embedding_model]['label']}). "
            "Upload or re-embed data with this model selected first.",
        )

    # ── Step 1: Apply all metadata filters in SQL to get the candidate pool ───
    uid_sql, uid_params   = _sql_upload_ids_clause(uid_list)
    date_sql, date_params = sql_date_clauses(date_from, date_to)
    words_sql, words_params = sql_min_words_clause(min_words)

    sql_params: list = []
    sql = "SELECT id, msg_uuid, username, date, content, upload_id, is_suno_team, row_index FROM messages WHERE 1=1"
    if username:
        sql += " AND LOWER(username) LIKE LOWER(?)"
        sql_params.append(f"%{username}%")
    sql += uid_sql + _suno_sql(suno_team) + date_sql + words_sql
    sql_params.extend(uid_params + date_params + words_params)
    sql += " ORDER BY date, row_index"

    conn = get_db()
    db_rows = conn.execute(sql, sql_params).fetchall()
    conn.close()

    if not db_rows:
        return []

    # Build UUID → full-row map for the filtered set
    filtered_map: dict = {r["msg_uuid"]: dict(r) for r in db_rows}
    filtered_uuids = list(filtered_map.keys())
    n_filtered = len(filtered_uuids)

    # ── Step 2: Embed the query ───────────────────────────────────────────────
    query_emb = (await embed_texts_async([query]))[0]
    q_vec = _np.array(query_emb, dtype=_np.float32)
    # Normalise once for fast cosine similarity via dot product
    q_norm = _np.linalg.norm(q_vec)
    if q_norm > 0:
        q_vec = q_vec / q_norm

    # ── Step 3: Score the filtered set ───────────────────────────────────────
    # Strategy A — small filtered set: fetch embeddings directly by ID and
    #   compute exact cosine similarity. Every filtered message is scored;
    #   no results are missed regardless of how selective the filters are.
    # Strategy B — large filtered set: broad vector query then intersect with
    #   the filtered UUIDs. Uses the ANN index for speed.

    scored: list = []  # [(similarity, msg_uuid), ...]

    if n_filtered <= _SEMANTIC_DIRECT_THRESHOLD:
        # Strategy A: direct lookup
        def _fetch_direct() -> dict:
            try:
                result   = col.get(ids=filtered_uuids, include=["embeddings"])
                emb_ids  = result.get("ids") or []
                emb_vecs = result.get("embeddings") or []
                return {eid: evec for eid, evec in zip(emb_ids, emb_vecs) if evec is not None}
            except Exception as exc:
                logger.warning("semantic search: direct emb fetch failed (%s)", exc)
                return {}

        emb_map: dict = await _sem_loop.run_in_executor(state.vector_executor, _fetch_direct)

        for uid, emb in emb_map.items():
            if uid not in filtered_map:
                continue
            v = _np.array(emb, dtype=_np.float32)
            norm = _np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            sim = float(_np.dot(q_vec, v))
            scored.append((sim, uid))

    else:
        # Strategy B: broad ANN query then intersect
        fetch_n = min(total, max(n_filtered * 2, 2000))

        def _query_broad():
            return col.query(query_embeddings=[query_emb], n_results=fetch_n)

        try:
            results = await _sem_loop.run_in_executor(state.vector_executor, _query_broad)
        except (_httpx.ReadTimeout, _httpx.ConnectTimeout, _httpx.TimeoutException, Exception) as exc:
            if "timeout" in str(exc).lower() or isinstance(exc, (_httpx.ReadTimeout, _httpx.ConnectTimeout)):
                raise HTTPException(
                    504,
                    f"Vector search timed out (queried {fetch_n} results). "
                    "Try a narrower date range or upload filter, then retry.",
                )
            raise

        ids_raw   = results["ids"][0]
        dists_raw = results["distances"][0]
        for uid, dist in zip(ids_raw, dists_raw):
            if uid in filtered_map:
                sim = round(1.0 - float(dist), 4)
                scored.append((sim, uid))

    # ── Step 4: Sort by similarity and return top n_results ──────────────────
    scored.sort(key=lambda x: -x[0])

    messages: list[dict] = []
    for sim, uid in scored:
        msg = dict(filtered_map[uid])
        msg["similarity_score"] = round(sim, 4)
        messages.append(msg)
        if len(messages) >= n_results:
            break

    return messages


@router.get("/api/search/users-in-range")
async def search_users_in_range(
    upload_ids: Optional[str] = None,
    date_from:  Optional[str] = None,
    date_to:    Optional[str] = None,
    suno_team:  str           = "all",
    min_words:  int           = 0,
):
    uid_list              = _parse_upload_ids(upload_ids)
    uid_sql, uid_params   = _sql_upload_ids_clause(uid_list)
    date_sql, date_params = sql_date_clauses(date_from, date_to)
    words_sql, words_params = sql_min_words_clause(min_words)

    sql = """
        SELECT
            username,
            COUNT(*) AS total_messages,
            MIN(date) AS first_message_date,
            MAX(date) AS last_message_date,
            ROUND(AVG(
                CASE WHEN TRIM(COALESCE(content,'')) = '' THEN 0
                     ELSE LENGTH(TRIM(COALESCE(content,'')))
                          - LENGTH(REPLACE(TRIM(COALESCE(content,'')), ' ', ''))
                          + 1
                END
            ), 1) AS avg_word_count,
            COUNT(DISTINCT strftime('%Y-%W', date)) AS weeks_with_messages,
            MAX(is_suno_team) AS is_suno_team
        FROM messages
        WHERE 1=1
    """
    sql += uid_sql + _suno_sql(suno_team) + date_sql + words_sql
    sql += " GROUP BY LOWER(username) ORDER BY total_messages DESC"

    conn = get_db()
    rows = conn.execute(
        sql, uid_params + date_params + words_params
    ).fetchall()
    conn.close()

    # Compute total distinct weeks in the requested date range.
    total_weeks_in_range: Optional[int] = None
    if date_from and date_to:
        try:
            d0 = _date.fromisoformat(date_from)
            d1 = _date.fromisoformat(date_to)
            if d0 <= d1:
                seen: set = set()
                cur = d0
                while cur <= d1:
                    seen.add(cur.strftime("%Y-%W"))
                    cur += timedelta(days=7)
                seen.add(d1.strftime("%Y-%W"))
                total_weeks_in_range = len(seen)
        except ValueError:
            pass

    result = []
    for r in rows:
        row = dict(r)
        weeks_with = row.get("weeks_with_messages") or 0
        if total_weeks_in_range and total_weeks_in_range > 0:
            row["pct_weeks_active"] = round(weeks_with / total_weeks_in_range * 100, 1)
        else:
            row["pct_weeks_active"] = None
        row["total_weeks_in_range"] = total_weeks_in_range
        result.append(row)

    return result


@router.get("/api/search/user-messages")
async def search_user_messages(
    username:   str,
    upload_ids: Optional[str] = None,
    date_from:  Optional[str] = None,
    date_to:    Optional[str] = None,
    keyword:    Optional[str] = None,
    suno_team:  str           = "all",
    min_words:  int           = 0,
    limit:      int           = 0,
):
    uid_list              = _parse_upload_ids(upload_ids)
    uid_sql, uid_params   = _sql_upload_ids_clause(uid_list)
    date_sql, date_params = sql_date_clauses(date_from, date_to)
    words_sql, words_params = sql_min_words_clause(min_words)

    params: list = [username]
    sql = (
        "SELECT * FROM messages WHERE LOWER(username) = LOWER(?)"
        + uid_sql + _suno_sql(suno_team) + date_sql + words_sql
    )
    params += uid_params + date_params + words_params

    if keyword:
        sql += " AND LOWER(content) LIKE LOWER(?)"
        params.append(f"%{keyword}%")

    sql += " ORDER BY date, row_index"
    if limit > 0:
        sql += " LIMIT ?"
        params.append(limit)

    conn = get_db()
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.post("/api/search/bulk-context")
async def bulk_context(request: Request):
    """
    Fetch conversation context for multiple messages in one round-trip.

    Request body:
      msg_ids – list of integer message IDs
      before  – messages before each target (default 5, max 50)
      after   – messages after each target  (default 5, max 50)

    Returns { "<msg_id>": [context_rows...], ... }
    Each row has is_target=True on the target message.
    """
    body    = await request.json()
    msg_ids = [int(i) for i in (body.get("msg_ids") or [])]
    before  = max(0, min(int(body.get("before", 5)), 50))
    after   = max(0, min(int(body.get("after",  5)), 50))

    if not msg_ids:
        return {}

    conn = get_db()
    result: dict = {}
    for msg_id in msg_ids:
        target = conn.execute(
            "SELECT * FROM messages WHERE id = ?", (msg_id,)
        ).fetchone()
        if not target:
            continue
        t = dict(target)
        rows = conn.execute(
            """SELECT * FROM messages
               WHERE upload_id = ? AND row_index BETWEEN ? AND ?
               ORDER BY row_index""",
            (t["upload_id"], max(0, t["row_index"] - before), t["row_index"] + after),
        ).fetchall()
        result[str(msg_id)] = [
            {**dict(r), "is_target": r["id"] == msg_id}
            for r in rows
        ]
    conn.close()
    return result
