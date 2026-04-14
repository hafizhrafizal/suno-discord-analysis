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
    config_api.py  /api/set-api-key, /api/set-embedding-model, /api/embedding-models
    stats.py       /api/stats
    uploads.py     /api/uploads, /api/upload, /api/uploads/:id/*, /api/jobs/:id
    search.py      /api/search/*
    chat.py        /api/chat, /api/summarize
    context.py     /api/context/:id, /api/filter/semantic
    bookmarks.py   /api/bookmarks, label assignments
    labels.py      /api/labels
    suno_team.py   /api/suno-team
"""

import asyncio
import logging
import traceback
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI, OpenAI
from starlette.middleware.base import BaseHTTPMiddleware

import state
from config import API_SECRET, EMBEDDING_MODELS, VECTOR_DB
from database import get_db, get_setting, init_db
from vector_store import init_vector_store

from routers import (
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


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── OpenAI clients ────────────────────────────────────────────────────────
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

    # ── Restore saved embedding-model preference ──────────────────────────────
    saved = get_setting("embedding_model", "openai")
    if saved in EMBEDDING_MODELS:
        state.current_embedding_model = saved

    logger.info("Active vector backend  : %s", VECTOR_DB)
    logger.info("Active embedding model : %s", state.current_embedding_model)

    # ── One-time migration: populate embedded_uploads from the vector store ───
    # Runs only when the tracking table is empty (first startup after upgrade).
    # Runs in a background thread so it never blocks startup or requests.
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
                        result = col.get(
                            where={"upload_id": {"$eq": uid}},
                            limit=1,
                            include=[],
                        )
                        if result.get("ids"):
                            conn.execute(
                                "INSERT OR IGNORE INTO embedded_uploads "
                                "(upload_id, model_id, embedded_at) VALUES (?,?,?)",
                                (uid, model_id, now_ts),
                            )
                            migrated += 1
                    except Exception as exc:
                        logger.warning("Migration check failed uid=%s model=%s: %s",
                                       uid, model_id, exc)
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


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Discord Chat Search", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ── Middleware ────────────────────────────────────────────────────────────────

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
    async def dispatch(self, request: Request, call_next):
        if not API_SECRET:
            return await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static/"):
            return await call_next(request)
        if path.startswith("/api/"):
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[7:].strip() != API_SECRET:
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        return await call_next(request)


app.add_middleware(_SecurityHeadersMiddleware)
app.add_middleware(_AuthMiddleware)


# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s %s\n%s",
                 request.method, request.url, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Check server logs for details."},
    )


# ── Index ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(config_api.router)
app.include_router(stats.router)
app.include_router(uploads.router)
app.include_router(search.router)
app.include_router(chat.router)
app.include_router(context.router)
app.include_router(bookmarks.router)
app.include_router(labels.router)
app.include_router(suno_team.router)
