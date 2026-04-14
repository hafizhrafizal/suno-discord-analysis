"""
routers/search.py — message search endpoints.

  GET /api/search/username
  GET /api/search/keyword
  GET /api/search/range
  GET /api/search/semantic
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

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
    uid_list = _parse_upload_ids(upload_ids)
    col      = active_collection()
    total    = col.count()
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

    results   = col.query(query_embeddings=[query_emb], n_results=fetch_n)
    ids       = results["ids"][0]
    distances = results["distances"][0]

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
