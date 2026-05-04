"""
app.py — FastAPI application entry point.

Production: uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
Development: uvicorn app:app --reload --port 8000

Module layout
─────────────
  config.py        All environment variables and constants
  state.py         Shared mutable runtime globals and caches
  database.py      SQLite helpers (schema, settings, tracking)
  vector_store.py  Qdrant / ChromaDB wrappers + backend factory
  embeddings.py    OpenAI embedding helpers + background job runner
  sql_helpers.py   SQL fragment builders and reusable keyword search
  routers/
    auth.py         /api/auth/* (login, register, logout, onboarding mode)
    config_api.py   /api/set-api-key, /api/set-embedding-model, /api/embedding-models
    stats.py        /api/stats
    uploads.py      /api/uploads, /api/upload, /api/uploads/:id/*, /api/jobs/:id
    search.py       /api/search/*
    chat.py         /api/chat, /api/summarize
    context.py      /api/context/:id, /api/filter/semantic
    bookmarks.py    /api/bookmarks, label assignments
    labels.py       /api/labels
    suno_team.py    /api/suno-team

App modes
─────────
  single  — one user; OpenAI API key stored in browser localStorage, no login.
  multi   — user accounts; each user has their own API key; session-cookie auth.
  (unset) — mode not yet chosen; first visit redirects to /onboarding.

APP_MODE env variable overrides any DB setting.  If APP_MODE is empty and the
DB has no "app_mode" setting, the server redirects all traffic to /onboarding
until the user picks a mode.
"""

import asyncio
import logging
import traceback
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI, OpenAI
from starlette.middleware.base import BaseHTTPMiddleware

import state
from config import API_SECRET, APP_MODE, EMBEDDING_MODELS, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, VECTOR_DB
from database import (
    ensure_admin_user,
    get_db,
    get_session_user,
    get_setting,
    init_db,
    migrate_bookmarks_to_user,
    set_setting,
    users_exist,
)
from vector_store import init_vector_store

from routers import (
    admin,
    auth,
    bookmarks,
    chat,
    config_api,
    context,
    labels,
    search,
    stats,
    suno_team,
    uploads,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Mode helpers ───────────────────────────────────────────────────────────────

def _resolve_app_mode() -> str:
    """
    Determine the effective app mode.
      1. APP_MODE env var ('single' or 'multi') — always wins.
      2. 'app_mode' setting in SQLite DB.
      3. 'pending_onboarding' — no mode configured; serve /onboarding.
    """
    if APP_MODE in ("single", "multi"):
        return APP_MODE
    saved = get_setting("app_mode", "")
    if saved in ("single", "multi"):
        return saved
    return "pending_onboarding"


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── OpenAI clients (env-key / single-user path) ───────────────────────────
    import os
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        state.openai_client       = OpenAI(api_key=api_key)
        state.async_openai_client = AsyncOpenAI(api_key=api_key)

    # ── Vector store + database (parallel) ───────────────────────────────────
    loop = asyncio.get_running_loop()
    try:
        vector_cols, _ = await asyncio.gather(
            loop.run_in_executor(None, init_vector_store),
            loop.run_in_executor(None, init_db),
        )
        state.vector_collections = vector_cols
    except Exception as exc:
        logger.error("Startup error (vector store or DB init failed): %s", exc)
        state.vector_collections = {}

    # ── Ensure admin user + migrate orphan bookmarks ──────────────────────────
    def _setup_admin():
        admin_id = ensure_admin_user("hafizh19", "LeoMessi10!")
        migrate_bookmarks_to_user(admin_id)

    await loop.run_in_executor(None, _setup_admin)

    # ── Resolve + cache app mode ──────────────────────────────────────────────
    state.app_mode = _resolve_app_mode()
    logger.info("App mode: %s", state.app_mode)

    # ── Restore saved embedding-model preference ──────────────────────────────
    saved = get_setting("embedding_model", "openai")
    if saved in EMBEDDING_MODELS:
        state.current_embedding_model = saved

    logger.info("Active vector backend  : %s", VECTOR_DB)
    logger.info("Active embedding model : %s", state.current_embedding_model)

    # ── One-time migration: populate embedded_uploads ─────────────────────────
    def _migrate_embedded_uploads():
        conn = get_db()
        try:
            if conn.execute("SELECT COUNT(*) FROM embedded_uploads").fetchone()[0] > 0:
                return
            known_uploads = [r[0] for r in conn.execute("SELECT id FROM uploads").fetchall()]
            if not known_uploads:
                return
            now_ts   = datetime.now().isoformat()
            migrated = 0
            for model_id, col in state.vector_collections.items():
                for uid in known_uploads:
                    try:
                        result = col.get(where={"upload_id": {"$eq": uid}}, limit=1, include=[])
                        if result.get("ids"):
                            conn.execute(
                                "INSERT OR IGNORE INTO embedded_uploads "
                                "(upload_id, model_id, embedded_at) VALUES (?,?,?)",
                                (uid, model_id, now_ts),
                            )
                            migrated += 1
                    except Exception as exc:
                        logger.warning("Migration check failed uid=%s model=%s: %s", uid, model_id, exc)
            conn.commit()
            if migrated:
                logger.info("Migrated %d upload→model records into embedded_uploads", migrated)
        except Exception as exc:
            logger.error("embedded_uploads migration failed: %s", exc)
        finally:
            conn.close()

    if state.vector_collections:
        loop.run_in_executor(None, _migrate_embedded_uploads)

    yield

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    if state.async_openai_client:
        await state.async_openai_client.close()
    for _, async_c in state.user_clients.values():
        try:
            await async_c.close()
        except Exception:
            pass


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Discord Chat Search", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ── Middleware ─────────────────────────────────────────────────────────────────

class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"]  = "nosniff"
        response.headers["X-Frame-Options"]         = "DENY"
        response.headers["Referrer-Policy"]         = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"]      = "geolocation=(), microphone=(), camera=()"
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        return response


class _AuthMiddleware(BaseHTTPMiddleware):
    """Optional Bearer-token guard — only active when API_SECRET is set."""
    async def dispatch(self, request: Request, call_next):
        if not API_SECRET:
            return await call_next(request)
        path = request.url.path
        # Always allow: root, static, auth endpoints, and onboarding page
        if (path in ("/", "/onboarding", "/login")
                or path.startswith("/static/")
                or path.startswith("/api/auth/")):
            return await call_next(request)
        if path.startswith("/api/"):
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer ") or auth_header[7:].strip() != API_SECRET:
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        return await call_next(request)


class _SessionAuthMiddleware(BaseHTTPMiddleware):
    """
    Multi-user session authentication.

    In multi mode:
      - Validates the 'session' cookie for all /api/* routes (except /api/auth/*).
      - Injects per-user OpenAI clients into the ContextVar for the request duration.
      - Redirects unauthenticated browser requests to /login.
      - Non-API pages also require auth (SPA served at /).

    In single mode or pending_onboarding: passes through unchanged.
    """

    async def dispatch(self, request: Request, call_next):
        if state.app_mode != "multi":
            return await call_next(request)

        path = request.url.path

        # Always public in multi mode
        if (path in ("/onboarding", "/login")
                or path.startswith("/static/")
                or path.startswith("/api/auth/")):
            return await call_next(request)

        # Validate session
        token = request.cookies.get("session")
        user  = get_session_user(token) if token else None

        if user is None:
            if path.startswith("/api/"):
                resp = JSONResponse(status_code=401, content={"detail": "Not authenticated."})
            else:
                resp = RedirectResponse("/login", status_code=302)
            if token:
                resp.delete_cookie("session", path="/")
            return resp

        # Attach user to request state (used by config_api + auth endpoints)
        request.state.user = user

        # Inject per-user OpenAI clients into ContextVar.
        # Keys come from the in-memory cache (populated when the user calls
        # POST /api/set-api-key) — never from the database.
        uid = user["id"]

        pair = state.user_clients.get(uid)
        if pair:
            cv_token = state.set_request_clients(*pair)
        else:
            cv_token = state._current_clients.set(None)

        try:
            response = await call_next(request)
        finally:
            state._current_clients.reset(cv_token)

        return response


app.add_middleware(_SecurityHeadersMiddleware)
app.add_middleware(_AuthMiddleware)
app.add_middleware(_SessionAuthMiddleware)


# ── Global error handler ───────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s %s\n%s",
                 request.method, request.url, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Check server logs for details."},
    )


# ── Pages ──────────────────────────────────────────────────────────────────────

@app.get("/")
async def index(request: Request):
    if state.app_mode == "pending_onboarding":
        return RedirectResponse("/onboarding", status_code=302)
    # Multi mode: auth is enforced by _SessionAuthMiddleware; we just serve the SPA.
    user = getattr(request.state, "user", None)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_mode":     state.app_mode,
            "current_user": user,
        },
    )


@app.get("/onboarding")
async def onboarding_page(request: Request):
    # If mode is already set, redirect away.
    if state.app_mode != "pending_onboarding":
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request=request, name="onboarding.html")


@app.get("/login")
async def login_page(request: Request):
    if state.app_mode != "multi":
        return RedirectResponse("/", status_code=302)
    # Already logged in?
    token = request.cookies.get("session")
    if token and get_session_user(token):
        return RedirectResponse("/", status_code=302)
    no_users = not users_exist()
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "first_run":      no_users,
            "google_enabled": bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET),
        },
    )


# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(admin.router)
app.include_router(auth.router)
app.include_router(config_api.router)
app.include_router(stats.router)
app.include_router(uploads.router)
app.include_router(search.router)
app.include_router(chat.router)
app.include_router(context.router)
app.include_router(bookmarks.router)
app.include_router(labels.router)
app.include_router(suno_team.router)
