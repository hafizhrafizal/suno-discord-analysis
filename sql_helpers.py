"""
sql_helpers.py — SQL fragment builders, FTS5 query builder, suno-team
filters, and the reusable keyword-search function used by both the
/api/search/keyword endpoint and the /api/chat retrieval step.
"""

import logging
import re
from typing import Optional

from database import get_db

logger = logging.getLogger(__name__)

# ── Suno-team SQL fragments ───────────────────────────────────────────────────

SUNO_TEAM_SQL         = " AND LOWER(is_suno_team) IN ('true', '1')"
EXCLUDE_SUNO_TEAM_SQL = " AND (is_suno_team IS NULL OR LOWER(is_suno_team) NOT IN ('true', '1'))"


def _suno_sql(suno_team: str) -> str:
    if suno_team == "only":
        return SUNO_TEAM_SQL
    if suno_team == "exclude":
        return EXCLUDE_SUNO_TEAM_SQL
    return ""


def is_suno_team_member(val: str) -> bool:
    return (val or "").lower() in ("true", "1")


# ── Date helpers ──────────────────────────────────────────────────────────────

def sql_date_clauses(
    date_from: Optional[str], date_to: Optional[str]
) -> tuple[str, list]:
    sql, params = "", []
    if date_from:
        sql += " AND substr(date, 1, 10) >= ?"
        params.append(date_from[:10])
    if date_to:
        sql += " AND substr(date, 1, 10) <= ?"
        params.append(date_to[:10])
    return sql, params


def date_in_range(
    date_str: str,
    date_from: Optional[str],
    date_to: Optional[str],
) -> bool:
    d = (date_str or "")[:10]
    if date_from and d < date_from[:10]:
        return False
    if date_to and d > date_to[:10]:
        return False
    return True


# ── Word-count helper ─────────────────────────────────────────────────────────

def sql_min_words_clause(min_words: int) -> tuple[str, list]:
    """Return SQL fragment filtering by minimum word count (space-count+1 heuristic)."""
    if min_words and min_words > 1:
        return (
            " AND (length(trim(content)) - length(replace(trim(content), ' ', '')) + 1) >= ?",
            [min_words],
        )
    return "", []


# ── Upload-ID filter ──────────────────────────────────────────────────────────

def _parse_upload_ids(upload_ids) -> list[str]:
    """Accept a comma-string, a list, or empty/None. Always returns a clean list."""
    if not upload_ids:
        return []
    if isinstance(upload_ids, list):
        return [str(x).strip() for x in upload_ids if str(x).strip()]
    return [x.strip() for x in str(upload_ids).split(",") if x.strip()]


def _sql_upload_ids_clause(uid_list: list[str]) -> tuple[str, list]:
    if not uid_list:
        return "", []
    placeholders = ",".join("?" * len(uid_list))
    return f" AND upload_id IN ({placeholders})", uid_list


# ── FTS5 query builder ────────────────────────────────────────────────────────

def _build_fts_query(keyword: str) -> str:
    """
    Convert a plain keyword string into a safe FTS5 MATCH expression.

    Strategy:
      - Strip FTS5 syntax chars to prevent injection / parse errors.
      - Single words  → prefix match   (e.g.  feat → "feat"*)
      - Multi-word    → phrase match   (e.g. "new feature" → "new feature")
    """
    safe = re.sub(r'["\'*^()\[\]{};:\\]', ' ', keyword).strip()
    if not safe:
        raise ValueError("Empty keyword after FTS5 sanitization")
    words = safe.split()
    if len(words) == 1:
        return f'"{words[0]}"*'
    return f'"{" ".join(words)}"'


# ── Reusable keyword-search function ─────────────────────────────────────────

async def keyword_search(
    keyword: str,
    upload_ids: Optional[str] = None,
    username: Optional[str]   = None,
    date_from: Optional[str]  = None,
    date_to:   Optional[str]  = None,
    suno_team: str             = "all",
    min_words: int             = 0,
    limit:     int             = 200,
) -> list[dict]:
    """
    Run a keyword search against the messages table.

    Uses FTS5 for speed; falls back to a LIKE scan if the FTS index is not
    yet populated.  Shared by the /api/search/keyword endpoint and the
    /api/chat retrieval step.
    """
    uid_list   = _parse_upload_ids(upload_ids)
    date_sql, date_params   = sql_date_clauses(date_from, date_to)
    words_sql, words_params = sql_min_words_clause(min_words)

    conn = get_db()
    try:
        fts_expr      = _build_fts_query(keyword)
        candidate_cap = limit * 20
        fts_rows = conn.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ? LIMIT ?",
            [fts_expr, candidate_cap],
        ).fetchall()

        if not fts_rows:
            return []

        candidate_ids = [r[0] for r in fts_rows]
        ph     = ",".join("?" * len(candidate_ids))
        params: list = list(candidate_ids)
        sql    = f"SELECT * FROM messages WHERE id IN ({ph})"

        if username:
            sql += " AND LOWER(username) LIKE LOWER(?)"
            params.append(f"%{username}%")
        if uid_list:
            uid_ph = ",".join("?" * len(uid_list))
            sql += f" AND upload_id IN ({uid_ph})"
            params.extend(uid_list)
        sql += _suno_sql(suno_team) + date_sql + words_sql
        params.extend(date_params + words_params)
        sql += " ORDER BY date, row_index LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()

    except Exception as fts_exc:
        logger.warning("FTS5 keyword search failed (%s), falling back to LIKE", fts_exc)
        uid_sql, uid_params = _sql_upload_ids_clause(uid_list)
        params_fb: list = [f"%{keyword}%"]
        sql_fb = "SELECT * FROM messages WHERE LOWER(content) LIKE LOWER(?)"
        if username:
            sql_fb += " AND LOWER(username) LIKE LOWER(?)"
            params_fb.append(f"%{username}%")
        sql_fb += uid_sql
        params_fb.extend(uid_params)
        sql_fb += _suno_sql(suno_team) + date_sql + words_sql
        params_fb.extend(date_params + words_params)
        sql_fb += " ORDER BY date, row_index LIMIT ?"
        params_fb.append(limit)
        rows = conn.execute(sql_fb, params_fb).fetchall()

    finally:
        conn.close()

    return [dict(r) for r in rows]
