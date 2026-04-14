"""
routers/bookmarks.py — bookmark management and label assignment.

  POST   /api/bookmarks
  GET    /api/bookmarks
  GET    /api/bookmarks/ids
  DELETE /api/bookmarks/{bookmark_id}
  DELETE /api/bookmarks/by-msg/{msg_id}
  POST   /api/bookmarks/{bookmark_id}/labels/{label_id}
  DELETE /api/bookmarks/{bookmark_id}/labels/{label_id}
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException

from database import get_db

router = APIRouter()


@router.post("/api/bookmarks")
async def add_bookmark(body: dict):
    msg_id = body.get("msg_id")
    if not isinstance(msg_id, int):
        raise HTTPException(400, "msg_id (int) is required")
    ctx_before = int(body.get("ctx_before", 5))
    ctx_after  = int(body.get("ctx_after",  5))
    note       = str(body.get("note", ""))
    conn = get_db()
    try:
        row = conn.execute("SELECT id FROM messages WHERE id=?", (msg_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Message not found")
        existing = conn.execute(
            "SELECT id FROM bookmarks WHERE msg_id=?", (msg_id,)
        ).fetchone()
        if existing:
            return {"status": "exists", "bookmark_id": existing["id"]}
        cur = conn.execute(
            "INSERT INTO bookmarks (msg_id, ctx_before, ctx_after, note, created_at) VALUES (?,?,?,?,?)",
            (msg_id, ctx_before, ctx_after, note, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return {"status": "created", "bookmark_id": cur.lastrowid}
    finally:
        conn.close()


@router.get("/api/bookmarks")
async def list_bookmarks():
    conn = get_db()
    rows = conn.execute(
        """SELECT b.id AS bookmark_id, b.ctx_before, b.ctx_after, b.note, b.created_at,
                  m.*
           FROM bookmarks b
           JOIN messages m ON m.id = b.msg_id
           ORDER BY b.created_at DESC"""
    ).fetchall()
    label_rows = conn.execute(
        """SELECT bl.bookmark_id, l.id, l.name, l.color
           FROM bookmark_labels bl
           JOIN labels l ON l.id = bl.label_id"""
    ).fetchall()
    conn.close()

    labels_by_bm: dict = {}
    for lr in label_rows:
        labels_by_bm.setdefault(lr["bookmark_id"], []).append(
            {"id": lr["id"], "name": lr["name"], "color": lr["color"]}
        )
    result = []
    for r in rows:
        d          = dict(r)
        d["labels"] = labels_by_bm.get(d["bookmark_id"], [])
        result.append(d)
    return result


@router.get("/api/bookmarks/ids")
async def list_bookmark_ids():
    """Return only the bookmarked message IDs — cheap check for UI state."""
    conn = get_db()
    rows = conn.execute("SELECT msg_id FROM bookmarks").fetchall()
    conn.close()
    return [r["msg_id"] for r in rows]


@router.delete("/api/bookmarks/{bookmark_id}")
async def delete_bookmark(bookmark_id: int):
    conn = get_db()
    cur  = conn.execute("DELETE FROM bookmarks WHERE id=?", (bookmark_id,))
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        raise HTTPException(404, "Bookmark not found")
    return {"status": "deleted"}


@router.delete("/api/bookmarks/by-msg/{msg_id}")
async def delete_bookmark_by_msg(msg_id: int):
    conn = get_db()
    cur  = conn.execute("DELETE FROM bookmarks WHERE msg_id=?", (msg_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted", "affected": cur.rowcount}


@router.post("/api/bookmarks/{bookmark_id}/labels/{label_id}")
async def assign_label(bookmark_id: int, label_id: int):
    conn = get_db()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO bookmark_labels (bookmark_id, label_id) VALUES (?,?)",
            (bookmark_id, label_id),
        )
        conn.commit()
        return {"status": "assigned"}
    finally:
        conn.close()


@router.delete("/api/bookmarks/{bookmark_id}/labels/{label_id}")
async def unassign_label(bookmark_id: int, label_id: int):
    conn = get_db()
    conn.execute(
        "DELETE FROM bookmark_labels WHERE bookmark_id=? AND label_id=?",
        (bookmark_id, label_id),
    )
    conn.commit()
    conn.close()
    return {"status": "unassigned"}
