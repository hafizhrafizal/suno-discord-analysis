"""
routers/admin.py — Admin-only user management.

  GET    /api/admin/users
  DELETE /api/admin/users/{user_id}
  POST   /api/admin/users/{user_id}/toggle-admin
"""

from fastapi import APIRouter, Depends, HTTPException

from database import get_db
from routers.deps import require_admin

router = APIRouter()


@router.get("/api/admin/users")
async def list_users(_: dict = Depends(require_admin)):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, username, is_admin, created_at FROM users ORDER BY created_at ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.delete("/api/admin/users/{user_id}")
async def delete_user(user_id: int, admin: dict = Depends(require_admin)):
    if admin.get("id") == user_id:
        raise HTTPException(400, "You cannot delete your own account.")
    conn = get_db()
    row = conn.execute("SELECT id, username FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "User not found.")
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted", "username": row["username"]}


@router.post("/api/admin/users/{user_id}/toggle-admin")
async def toggle_admin(user_id: int, admin: dict = Depends(require_admin)):
    if admin.get("id") == user_id:
        raise HTTPException(400, "You cannot change your own admin status.")
    conn = get_db()
    row = conn.execute("SELECT id, username, is_admin FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "User not found.")
    new_status = 0 if row["is_admin"] else 1
    conn.execute("UPDATE users SET is_admin = ? WHERE id = ?", (new_status, user_id))
    conn.commit()
    conn.close()
    return {"status": "ok", "username": row["username"], "is_admin": bool(new_status)}
