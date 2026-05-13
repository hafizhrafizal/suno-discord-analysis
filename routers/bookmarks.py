"""
routers/bookmarks.py — bookmark management and code assignment.

  POST   /api/bookmarks
  GET    /api/bookmarks
  GET    /api/bookmarks/ids
  DELETE /api/bookmarks/{bookmark_id}
  DELETE /api/bookmarks/by-msg/{msg_id}
  POST   /api/bookmarks/{bookmark_id}/codes/{code_id}
  DELETE /api/bookmarks/{bookmark_id}/codes/{code_id}

  Back-compat aliases (old /labels/ paths):
  POST   /api/bookmarks/{bookmark_id}/labels/{label_id}
  DELETE /api/bookmarks/{bookmark_id}/labels/{label_id}
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

import state
from database import get_db, get_session_user

router = APIRouter()


def _uid(request: Request) -> Optional[int]:
    """Return the authenticated user's id in multi mode, else None."""
    if state.app_mode != "multi":
        return None
    token = request.cookies.get("session")
    if not token:
        return None
    user = get_session_user(token)
    return user["id"] if user else None


@router.post("/api/bookmarks")
async def add_bookmark(body: dict, request: Request):
    msg_id = body.get("msg_id")
    if not isinstance(msg_id, int):
        raise HTTPException(400, "msg_id (int) is required")
    ctx_before = int(body.get("ctx_before", 5))
    ctx_after  = int(body.get("ctx_after",  5))
    note       = str(body.get("note", ""))
    user_id    = _uid(request)
    conn = get_db()
    try:
        row = conn.execute("SELECT id FROM messages WHERE id=?", (msg_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Message not found")
        if user_id is not None:
            existing = conn.execute(
                "SELECT id FROM bookmarks WHERE msg_id=? AND user_id=?", (msg_id, user_id)
            ).fetchone()
        else:
            existing = conn.execute(
                "SELECT id FROM bookmarks WHERE msg_id=?", (msg_id,)
            ).fetchone()
        if existing:
            return {"status": "exists", "bookmark_id": existing["id"]}
        cur = conn.execute(
            "INSERT INTO bookmarks (msg_id, ctx_before, ctx_after, note, created_at, user_id) VALUES (?,?,?,?,?,?)",
            (msg_id, ctx_before, ctx_after, note, datetime.utcnow().isoformat(), user_id),
        )
        conn.commit()
        return {"status": "created", "bookmark_id": cur.lastrowid}
    finally:
        conn.close()


@router.get("/api/bookmarks")
async def list_bookmarks(request: Request):
    user_id = _uid(request)
    conn = get_db()
    if user_id is not None:
        rows = conn.execute(
            """SELECT b.id AS bookmark_id, b.ctx_before, b.ctx_after, b.note, b.created_at,
                      m.*
               FROM bookmarks b
               JOIN messages m ON m.id = b.msg_id
               WHERE b.user_id = ?
               ORDER BY b.created_at DESC""",
            (user_id,),
        ).fetchall()
        bm_ids_sub = "SELECT id FROM bookmarks WHERE user_id = ?"
        code_rows = conn.execute(
            f"""SELECT bc.bookmark_id, c.id, c.name, c.color, c.description, c.category_id
               FROM bookmark_codes bc
               JOIN codes c ON c.id = bc.code_id
               WHERE bc.bookmark_id IN ({bm_ids_sub})""",
            (user_id,),
        ).fetchall()
        hl_rows = conn.execute(
            f"""SELECT bch.id, bch.bookmark_id, bch.code_id, bch.highlighted_text,
                       c.name AS code_name, c.color AS code_color
               FROM bookmark_code_highlights bch
               JOIN codes c ON c.id = bch.code_id
               WHERE bch.bookmark_id IN ({bm_ids_sub})""",
            (user_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT b.id AS bookmark_id, b.ctx_before, b.ctx_after, b.note, b.created_at,
                      m.*
               FROM bookmarks b
               JOIN messages m ON m.id = b.msg_id
               ORDER BY b.created_at DESC"""
        ).fetchall()
        code_rows = conn.execute(
            """SELECT bc.bookmark_id, c.id, c.name, c.color, c.description, c.category_id
               FROM bookmark_codes bc
               JOIN codes c ON c.id = bc.code_id"""
        ).fetchall()
        hl_rows = conn.execute(
            """SELECT bch.id, bch.bookmark_id, bch.code_id, bch.highlighted_text,
                      c.name AS code_name, c.color AS code_color
               FROM bookmark_code_highlights bch
               JOIN codes c ON c.id = bch.code_id"""
        ).fetchall()
    conn.close()

    codes_by_bm: dict = {}
    for cr in code_rows:
        codes_by_bm.setdefault(cr["bookmark_id"], []).append({
            "id": cr["id"], "name": cr["name"], "color": cr["color"],
            "description": cr["description"], "category_id": cr["category_id"],
        })

    hl_by_bm: dict = {}
    for hr in hl_rows:
        hl_by_bm.setdefault(hr["bookmark_id"], []).append({
            "id": hr["id"], "code_id": hr["code_id"],
            "code_name": hr["code_name"], "code_color": hr["code_color"],
            "highlighted_text": hr["highlighted_text"],
        })

    result = []
    for r in rows:
        d = dict(r)
        d["codes"]      = codes_by_bm.get(d["bookmark_id"], [])
        d["labels"]     = d["codes"]   # back-compat alias
        d["highlights"] = hl_by_bm.get(d["bookmark_id"], [])
        result.append(d)
    return result


@router.get("/api/bookmarks/ids")
async def list_bookmark_ids(request: Request):
    user_id = _uid(request)
    conn = get_db()
    if user_id is not None:
        rows = conn.execute(
            "SELECT msg_id FROM bookmarks WHERE user_id = ?", (user_id,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT msg_id FROM bookmarks").fetchall()
    conn.close()
    return [r["msg_id"] for r in rows]


@router.patch("/api/bookmarks/{bookmark_id}")
async def update_bookmark(bookmark_id: int, body: dict, request: Request):
    note = body.get("note")
    if note is None:
        raise HTTPException(400, "note is required")
    user_id = _uid(request)
    conn = get_db()
    if user_id is not None:
        cur = conn.execute(
            "UPDATE bookmarks SET note = ? WHERE id = ? AND user_id = ?",
            (str(note), bookmark_id, user_id),
        )
    else:
        cur = conn.execute(
            "UPDATE bookmarks SET note = ? WHERE id = ?",
            (str(note), bookmark_id),
        )
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        raise HTTPException(404, "Bookmark not found")
    return {"status": "updated"}


@router.delete("/api/bookmarks/{bookmark_id}")
async def delete_bookmark(bookmark_id: int, request: Request):
    user_id = _uid(request)
    conn = get_db()
    if user_id is not None:
        cur = conn.execute(
            "DELETE FROM bookmarks WHERE id=? AND user_id=?", (bookmark_id, user_id)
        )
    else:
        cur = conn.execute("DELETE FROM bookmarks WHERE id=?", (bookmark_id,))
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        raise HTTPException(404, "Bookmark not found")
    return {"status": "deleted"}


@router.delete("/api/bookmarks/by-msg/{msg_id}")
async def delete_bookmark_by_msg(msg_id: int, request: Request):
    user_id = _uid(request)
    conn = get_db()
    if user_id is not None:
        cur = conn.execute(
            "DELETE FROM bookmarks WHERE msg_id=? AND user_id=?", (msg_id, user_id)
        )
    else:
        cur = conn.execute("DELETE FROM bookmarks WHERE msg_id=?", (msg_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted", "affected": cur.rowcount}


# ── Code assignment ───────────────────────────────────────────────────────────

@router.post("/api/bookmarks/{bookmark_id}/codes/{code_id}")
async def assign_code(bookmark_id: int, code_id: int):
    conn = get_db()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO bookmark_codes (bookmark_id, code_id) VALUES (?,?)",
            (bookmark_id, code_id),
        )
        conn.commit()
        return {"status": "assigned"}
    finally:
        conn.close()


@router.delete("/api/bookmarks/{bookmark_id}/codes/{code_id}")
async def unassign_code(bookmark_id: int, code_id: int):
    conn = get_db()
    conn.execute(
        "DELETE FROM bookmark_codes WHERE bookmark_id=? AND code_id=?",
        (bookmark_id, code_id),
    )
    conn.commit()
    conn.close()
    return {"status": "unassigned"}


# ── Back-compat: old /labels/ paths ──────────────────────────────────────────

@router.post("/api/bookmarks/{bookmark_id}/labels/{label_id}")
async def assign_label_compat(bookmark_id: int, label_id: int):
    return await assign_code(bookmark_id, label_id)

@router.delete("/api/bookmarks/{bookmark_id}/labels/{label_id}")
async def unassign_label_compat(bookmark_id: int, label_id: int):
    return await unassign_code(bookmark_id, label_id)


# ── Span-level highlight codings ─────────────────────────────────────────────

@router.post("/api/bookmarks/{bookmark_id}/highlights")
async def add_highlight(bookmark_id: int, body: dict):
    code_id          = body.get("code_id")
    highlighted_text = (body.get("highlighted_text") or "").strip()
    if not isinstance(code_id, int) or not highlighted_text:
        raise HTTPException(400, "code_id (int) and highlighted_text (str) are required")
    conn = get_db()
    try:
        cur = conn.execute(
            """INSERT OR IGNORE INTO bookmark_code_highlights
               (bookmark_id, code_id, highlighted_text, created_at)
               VALUES (?,?,?,?)""",
            (bookmark_id, code_id, highlighted_text, datetime.utcnow().isoformat()),
        )
        conn.commit()
        # If INSERT OR IGNORE skipped (duplicate), fetch the existing row's id
        if cur.lastrowid and cur.rowcount > 0:
            return {"status": "created", "id": cur.lastrowid}
        existing = conn.execute(
            """SELECT id FROM bookmark_code_highlights
               WHERE bookmark_id=? AND code_id=? AND highlighted_text=?""",
            (bookmark_id, code_id, highlighted_text),
        ).fetchone()
        return {"status": "exists", "id": existing["id"] if existing else None}
    finally:
        conn.close()


@router.delete("/api/bookmarks/{bookmark_id}/highlights/{highlight_id}")
async def remove_highlight(bookmark_id: int, highlight_id: int):
    conn = get_db()
    conn.execute(
        "DELETE FROM bookmark_code_highlights WHERE id=? AND bookmark_id=?",
        (highlight_id, bookmark_id),
    )
    conn.commit()
    conn.close()
    return {"status": "deleted"}
