"""
routers/auth.py — Authentication and onboarding endpoints.

  POST /api/auth/set-mode              — choose app mode during onboarding
  POST /api/auth/first-admin           — create the first admin (onboarding, multi mode only)
  POST /api/auth/register              — create account (multi mode)
  POST /api/auth/login                 — authenticate and issue session cookie
  POST /api/auth/logout                — destroy session
  GET  /api/auth/me                    — return current user info
  GET  /api/auth/users-exist           — check whether any users exist (login page hint)
  GET  /api/auth/google                — redirect to Google OAuth consent screen
  GET  /api/auth/google/callback       — handle Google OAuth callback
"""

import hashlib
import re
import secrets
import time
from datetime import datetime, timedelta
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

import state
from config import (
    APP_MODE,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_REDIRECT_URI,
)
from database import (
    create_google_user,
    create_session,
    create_user,
    delete_session,
    get_session_user,
    get_user_by_google_id,
    get_user_by_id,
    get_user_by_username,
    set_setting,
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


def _set_session_cookie(response, token: str, is_https: bool) -> None:
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


# ── First admin setup (runs once, immediately after enabling multi mode) ──────

@router.post("/api/auth/first-admin")
async def create_first_admin(body: dict, request: Request):
    """
    Create the initial admin account. Only succeeds when no users exist at all.
    Called from the onboarding Step 2 form.
    """
    if state.app_mode != "multi":
        raise HTTPException(400, "Only available in multi-user mode.")
    if users_exist():
        raise HTTPException(409, "Setup is already complete. Please log in.")

    username = (body.get("username") or "").strip()
    password = (body.get("password") or "").strip()

    if not username:
        raise HTTPException(400, "username is required.")
    if len(username) < 2 or len(username) > 40:
        raise HTTPException(400, "username must be 2–40 characters.")
    if not password or len(password) < 8:
        raise HTTPException(400, "password must be at least 8 characters.")

    pw_hash, salt = _hash_password(password)
    user_id = create_user(username, pw_hash, salt, is_admin=True)

    token   = secrets.token_hex(32)
    expires = (datetime.utcnow() + timedelta(days=30)).isoformat()
    create_session(token, user_id, expires)

    resp = JSONResponse({"status": "ok", "username": username, "redirect": "/"})
    _set_session_cookie(resp, token, request.url.scheme == "https")
    return resp


# ── Registration ──────────────────────────────────────────────────────────────

@router.post("/api/auth/register")
async def register(body: dict, request: Request):
    if state.app_mode != "multi":
        raise HTTPException(403, "Registration is only available in multi-user mode.")

    username = (body.get("username") or "").strip()
    password = (body.get("password") or "").strip()

    if not username:
        raise HTTPException(400, "username is required.")
    if len(username) < 2 or len(username) > 40:
        raise HTTPException(400, "username must be 2–40 characters.")
    if not password or len(password) < 8:
        raise HTTPException(400, "password must be at least 8 characters.")

    if get_user_by_username(username):
        raise HTTPException(409, f"Username '{username}' is already taken.")

    pw_hash, salt = _hash_password(password)
    user_id = create_user(username, pw_hash, salt)

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

    # Friendly hint for Google-only accounts
    if user and not user.get("password_hash") and user.get("google_id"):
        raise HTTPException(401, "This account uses Google sign-in. Use the 'Continue with Google' button.")

    if user is None or not _verify_password(password, user["password_hash"], user["password_salt"]):
        raise HTTPException(401, "Invalid username or password.")

    uid   = user["id"]
    token = secrets.token_hex(32)
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
    uid = user["id"]
    return {
        "id":          uid,
        "username":    user["username"],
        "has_api_key": uid in state.user_clients,
        "is_admin":    bool(user.get("is_admin")),
    }


# ── Users exist check (for login page initial state) ─────────────────────────

@router.get("/api/auth/users-exist")
async def check_users_exist():
    return {"exists": users_exist()}


# ── Google OAuth ──────────────────────────────────────────────────────────────

# In-memory OAuth state store: token → unix timestamp
_oauth_states: dict[str, float] = {}
_STATE_TTL = 600  # 10 minutes


def _clean_oauth_states() -> None:
    cutoff = time.time() - _STATE_TTL
    for key in list(_oauth_states):
        if _oauth_states[key] < cutoff:
            del _oauth_states[key]


def _sanitize_username(raw: str) -> str:
    """Turn a Google display name into a valid username."""
    clean = re.sub(r"[^a-zA-Z0-9_]", "", raw.replace(" ", "_"))
    return (clean[:30] or "user").lower()


@router.get("/api/auth/google")
async def google_login():
    """Redirect the browser to Google's OAuth consent screen."""
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(501, "Google OAuth is not configured on this server.")
    if state.app_mode != "multi":
        raise HTTPException(400, "Google sign-in is only available in multi-user mode.")

    _clean_oauth_states()
    oauth_state = secrets.token_urlsafe(32)
    _oauth_states[oauth_state] = time.time()

    params = urlencode({
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope":         "openid email profile",
        "state":         oauth_state,
        "access_type":   "online",
    })
    return RedirectResponse(f"https://accounts.google.com/o/oauth2/v2/auth?{params}")


@router.get("/api/auth/google/callback")
async def google_callback(request: Request):
    """Exchange the authorization code for a session."""

    # Google returned an error (e.g. user denied access)
    error = request.query_params.get("error")
    if error:
        return RedirectResponse(f"/login?error={error}")

    code        = request.query_params.get("code")
    oauth_state = request.query_params.get("state")

    _clean_oauth_states()
    if not oauth_state or oauth_state not in _oauth_states:
        return RedirectResponse("/login?error=Invalid+OAuth+state.+Please+try+again.")
    del _oauth_states[oauth_state]

    if not code:
        return RedirectResponse("/login?error=Missing+authorization+code.")

    # Exchange code → tokens
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            token_resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code":          code,
                    "client_id":     GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "redirect_uri":  GOOGLE_REDIRECT_URI,
                    "grant_type":    "authorization_code",
                },
            )
            if token_resp.status_code != 200:
                return RedirectResponse("/login?error=Failed+to+exchange+authorization+code.")

            access_token = token_resp.json().get("access_token")
            if not access_token:
                return RedirectResponse("/login?error=No+access+token+received.")

            # Fetch user profile
            info_resp = await client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if info_resp.status_code != 200:
                return RedirectResponse("/login?error=Failed+to+fetch+Google+profile.")

            userinfo = info_resp.json()
    except Exception:
        return RedirectResponse("/login?error=Google+sign-in+failed.+Please+try+again.")

    google_id = userinfo.get("sub")
    email     = userinfo.get("email", "")
    raw_name  = (userinfo.get("given_name") or userinfo.get("name") or
                 email.split("@")[0] or "user")

    if not google_id:
        return RedirectResponse("/login?error=Could+not+retrieve+Google+user+ID.")

    # Find existing account linked to this Google ID
    user = get_user_by_google_id(google_id)

    if user is None:
        # First time this Google account is used — auto-register
        base     = _sanitize_username(raw_name)
        username = base
        n = 1
        while get_user_by_username(username):
            username = f"{base}{n}"
            n += 1

        is_first = not users_exist()
        user_id  = create_google_user(google_id, email, username, is_admin=is_first)
        user     = get_user_by_id(user_id)

    uid     = user["id"]
    token   = secrets.token_hex(32)
    expires = (datetime.utcnow() + timedelta(days=30)).isoformat()
    create_session(token, uid, expires)

    resp = RedirectResponse("/", status_code=302)
    _set_session_cookie(resp, token, request.url.scheme == "https")
    return resp
