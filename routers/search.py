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
    uid_list = _parse_upload_ids(upload_ids)
    col      = active_collection()
    if col is None:
        raise HTTPException(400, "Vector store is not initialised — check server logs.")

    _sem_loop = _asyncio.get_running_loop()
    total = await _sem_loop.run_in_executor(state.vector_executor, col.count)
    if total == 0:
        raise HTTPException(
            400,
            f"No messages are embedded with the current model "
            f"({EMBEDDING_MODELS[state.current_embedding_model]['label']}). "
            "Upload or re-embed data with this model selected first.",
        )

    has_filters = bool(username or date_from or date_to or (suno_team != "all") or uid_list or min_words)
    fetch_n     = min(n_results * 4 if has_filters else n_results, total)
    query_emb   = (await embed_texts_async([query]))[0]

    _query_fn   = lambda: col.query(query_embeddings=[query_emb], n_results=fetch_n)
    results     = await _sem_loop.run_in_executor(state.vector_executor, _query_fn)
    ids         = results["ids"][0]
    distances   = results["distances"][0]

    conn         = get_db()
    uuid_to_dist = dict(zip(ids, distances))
    if ids:
        placeholders = ",".join("?" * len(ids))
        rows         = conn.execute(
            f"SELECT * FROM messages WHERE msg_uuid IN ({placeholders})", ids
        ).fetchall()
        uuid_to_row = {r["msg_uuid"]: dict(r) for r in rows}
    else:
        uuid_to_row = {}
    conn.close()

    messages: list[dict] = []
    for msg_uuid in ids:
        msg = uuid_to_row.get(msg_uuid)
        if msg is None:
            continue
        msg["similarity_score"] = round(1.0 - uuid_to_dist[msg_uuid], 4)

        if uid_list and msg["upload_id"] not in uid_list:
            continue
        if username and username.lower() not in msg["username"].lower():
            continue
        if not date_in_range(msg["date"], date_from, date_to):
            continue
        if suno_team == "only" and not is_suno_team_member(msg["is_suno_team"]):
            continue
        if suno_team == "exclude" and is_suno_team_member(msg["is_suno_team"]):
            continue
        if min_words > 1 and len((msg["content"] or "").split()) < min_words:
            continue

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
