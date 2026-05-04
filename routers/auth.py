"""
routers/auth.py — Authentication and onboarding endpoints.

  POST /api/auth/set-mode   — choose app mode during onboarding
  POST /api/auth/register   — create account (multi mode)
  POST /api/auth/login      — authenticate and issue session cookie
  POST /api/auth/logout     — destroy session
  GET  /api/auth/me         — return current user info
"""

import hashlib
import secrets
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

import state
from config import APP_MODE
from database import (
    create_session,
    create_user,
    delete_session,
    get_session_user,
    get_user_by_username,
    set_setting,
    update_user_api_key,
    users_exist,
)

router = APIRouter()


# ── Password helpers ──────────────────────────────────────────────────────────

def _hash_password(password: str) -> tuple[str, str]:
    salt = secrets.token_hex(32)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
    return h.hex(), salt


def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
    return secrets.compare_digest(h.hex(), stored_hash)


def _set_session_cookie(response: JSONResponse, token: str, is_https: bool) -> None:
    response.set_cookie(
        "session",
        value=token,
        max_age=2_592_000,
        httponly=True,
        samesite="lax",
        secure=is_https,
        path="/",
    )


# ── Onboarding ────────────────────────────────────────────────────────────────

@router.post("/api/auth/set-mode")
async def set_app_mode(body: dict):
    """Set the application mode during onboarding. Rejected if already configured."""
    if APP_MODE:
        raise HTTPException(409, "App mode is fixed by the APP_MODE environment variable.")
    if state.app_mode != "pending_onboarding":
        raise HTTPException(409, "App mode is already configured.")

    mode = (body.get("mode") or "").strip().lower()
    if mode not in ("single", "multi"):
        raise HTTPException(400, "mode must be 'single' or 'multi'.")

    set_setting("app_mode", mode)
    state.app_mode = mode
    return {"status": "ok", "mode": mode, "redirect": "/login" if mode == "multi" else "/"}


# ── Registration ──────────────────────────────────────────────────────────────

@router.post("/api/auth/register")
async def register(body: dict, request: Request):
    if state.app_mode != "multi":
        raise HTTPException(403, "Registration is only available in multi-user mode.")

    username = (body.get("username") or "").strip()
    password = (body.get("password") or "").strip()
    api_key  = (body.get("openai_api_key") or "").strip()

    if not username:
        raise HTTPException(400, "username is required.")
    if len(username) < 2 or len(username) > 40:
        raise HTTPException(400, "username must be 2–40 characters.")
    if not password or len(password) < 8:
        raise HTTPException(400, "password must be at least 8 characters.")

    if get_user_by_username(username):
        raise HTTPException(409, f"Username '{username}' is already taken.")

    pw_hash, salt = _hash_password(password)
    user_id = create_user(username, pw_hash, salt, api_key)

    if api_key:
        from openai import AsyncOpenAI, OpenAI
        state.user_clients[user_id] = (OpenAI(api_key=api_key), AsyncOpenAI(api_key=api_key))

    token   = secrets.token_hex(32)
    expires = (datetime.utcnow() + timedelta(days=30)).isoformat()
    create_session(token, user_id, expires)

    resp = JSONResponse({"status": "ok", "username": username, "redirect": "/"})
    _set_session_cookie(resp, token, request.url.scheme == "https")
    return resp


# ── Login ─────────────────────────────────────────────────────────────────────

@router.post("/api/auth/login")
async def login(body: dict, request: Request):
    if state.app_mode != "multi":
        raise HTTPException(403, "Login is only available in multi-user mode.")

    username = (body.get("username") or "").strip()
    password = (body.get("password") or "").strip()

    if not username or not password:
        raise HTTPException(400, "username and password are required.")

    user = get_user_by_username(username)
    if user is None or not _verify_password(password, user["password_hash"], user["password_salt"]):
        raise HTTPException(401, "Invalid username or password.")

    uid = user["id"]
    if user.get("openai_api_key") and uid not in state.user_clients:
        from openai import AsyncOpenAI, OpenAI
        key = user["openai_api_key"]
        state.user_clients[uid] = (OpenAI(api_key=key), AsyncOpenAI(api_key=key))

    token   = secrets.token_hex(32)
    expires = (datetime.utcnow() + timedelta(days=30)).isoformat()
    create_session(token, uid, expires)

    resp = JSONResponse({"status": "ok", "username": user["username"], "redirect": "/"})
    _set_session_cookie(resp, token, request.url.scheme == "https")
    return resp


# ── Logout ────────────────────────────────────────────────────────────────────

@router.post("/api/auth/logout")
async def logout(request: Request):
    token = request.cookies.get("session")
    if token:
        delete_session(token)
    resp = JSONResponse({"status": "ok"})
    resp.delete_cookie("session", path="/")
    return resp


# ── Current user ──────────────────────────────────────────────────────────────

@router.get("/api/auth/me")
async def get_me(request: Request):
    if state.app_mode != "multi":
        return {"mode": "single", "user": None}
    token = request.cookies.get("session")
    if not token:
        raise HTTPException(401, "Not authenticated.")
    user = get_session_user(token)
    if not user:
        raise HTTPException(401, "Session expired or invalid.")
    return {
        "id":          user["id"],
        "username":    user["username"],
        "has_api_key": bool(user.get("openai_api_key")),
        "is_admin":    bool(user.get("is_admin")),
    }


# ── Update API key (multi mode) ────────────────────────────────────────────────

@router.post("/api/auth/update-api-key")
async def update_api_key(body: dict, request: Request):
    """Update the current user's OpenAI API key (multi mode)."""
    if state.app_mode != "multi":
        raise HTTPException(403, "Use /api/set-api-key in single-user mode.")

    token = request.cookies.get("session")
    user  = get_session_user(token) if token else None
    if not user:
        raise HTTPException(401, "Not authenticated.")

    key = (body.get("api_key") or "").strip()
    if not key:
        raise HTTPException(400, "api_key is required.")

    uid = user["id"]
    update_user_api_key(uid, key)

    # Evict old client and create new one
    old = state.user_clients.pop(uid, None)
    if old:
        import asyncio
        async def _close_old():
            try:
                await old[1].close()
            except Exception:
                pass
        asyncio.create_task(_close_old())

    from openai import AsyncOpenAI, OpenAI
    state.user_clients[uid] = (OpenAI(api_key=key), AsyncOpenAI(api_key=key))
    state.set_request_clients(*state.user_clients[uid])

    return {"status": "ok", "message": "API key updated."}


# ── Users exist check (for login page initial state) ─────────────────────────

@router.get("/api/auth/users-exist")
async def check_users_exist():
    return {"exists": users_exist()}
