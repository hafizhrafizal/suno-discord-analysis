"""
routers/suno_team.py — Suno-team member management.

  GET    /api/suno-team
  DELETE /api/suno-team/{username}
"""

from fastapi import APIRouter, Depends

from database import get_db
from routers.deps import require_admin

router = APIRouter()


@router.get("/api/suno-team")
async def get_suno_team():
    """Return all usernames currently flagged as Suno Team, with message counts."""
    conn = get_db()
    rows = conn.execute(
        """
        SELECT username, COUNT(*) AS msg_count
        FROM messages
        WHERE LOWER(is_suno_team) IN ('true', '1')
        GROUP BY username
        ORDER BY username COLLATE NOCASE
        """
    ).fetchall()
    conn.close()
    return [{"username": r["username"], "msg_count": r["msg_count"]} for r in rows]


@router.delete("/api/suno-team/{username}")
async def remove_suno_team(username: str, _: dict = Depends(require_admin)):
    """Mark all messages by this username as non-Suno-Team."""
    conn   = get_db()
    result = conn.execute(
        "UPDATE messages SET is_suno_team = 'false' WHERE username = ?",
        (username,),
    )
    conn.commit()
    affected = result.rowcount
    conn.close()
    return {"username": username, "updated": affected}
