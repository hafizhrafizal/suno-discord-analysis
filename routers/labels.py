"""
routers/labels.py — label CRUD.

  GET    /api/labels
  POST   /api/labels
  DELETE /api/labels/{label_id}
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException

from database import get_db

router = APIRouter()


@router.get("/api/labels")
async def list_labels():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, color, created_at FROM labels ORDER BY name COLLATE NOCASE"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.post("/api/labels")
async def create_label(body: dict):
    name  = (body.get("name") or "").strip()
    color = (body.get("color") or "#6366f1").strip()
    if not name:
        raise HTTPException(400, "name is required")
    conn = get_db()
    try:
        cur = conn.execute(
            "INSERT INTO labels (name, color, created_at) VALUES (?,?,?)",
            (name, color, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return {"id": cur.lastrowid, "name": name, "color": color}
    except Exception:
        raise HTTPException(409, "Label name already exists")
    finally:
        conn.close()


@router.delete("/api/labels/{label_id}")
async def delete_label(label_id: int):
    conn = get_db()
    cur  = conn.execute("DELETE FROM labels WHERE id=?", (label_id,))
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        raise HTTPException(404, "Label not found")
    return {"status": "deleted"}
