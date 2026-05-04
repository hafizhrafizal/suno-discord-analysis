"""
routers/deps.py — Shared FastAPI dependencies for auth and role checks.
"""

from fastapi import Depends, HTTPException, Request

import state
from database import get_session_user


def get_request_user(request: Request) -> dict | None:
    """Return the authenticated user dict in multi mode, or None in single mode."""
    if state.app_mode != "multi":
        return None
    token = request.cookies.get("session")
    if not token:
        return None
    return get_session_user(token)


def require_admin(user: dict | None = Depends(get_request_user)) -> dict:
    """
    Dependency that allows the request only for admin users.
    In single mode, all operations are permitted (returns empty dict).
    In multi mode, raises 403 if the user is not an admin.
    """
    if state.app_mode != "multi":
        return {}
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user
