"""
Discord Chat Search — FastAPI app
Production: uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
Development: uvicorn app:app --reload --port 8000

Environment variables (set in .env or shell):
  DB_PATH          Path to the SQLite database file   (default: discord_data.db)
  CHROMA_PATH      Path to the ChromaDB directory     (default: ./chroma_db)
  MAX_UPLOAD_MB    Maximum CSV upload size in MB       (default: 50)
  API_SECRET       Bearer token to protect all /api/* endpoints (optional)
  OPENAI_API_KEY   Pre-load an OpenAI key at startup   (optional)
"""

import asyncio
import io
import json
import logging
import os
import re
import sqlite3
import traceback
import uuid
from xmlrpc import client

import numpy as np
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import chromadb
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from starlette.middleware.base import BaseHTTPMiddleware

# Load .env file if present (ignored in production where env vars are set directly)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Configuration from environment ─────────────────────────────────────────
DB_PATH          = os.environ.get("DB_PATH", "discord_data.db")
CHROMA_PATH      = os.environ.get("CHROMA_PATH", "./chroma_db")
MAX_UPLOAD_MB    = int(os.environ.get("MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1_048_576
# Optional bearer token — if set, all /api/* requests require Authorization: Bearer <token>
_API_SECRET      = os.environ.get("API_SECRET", "").strip()

# ─── Chat model allowlist ────────────────────────────────────────────────────
# Only permit known OpenAI model IDs to prevent arbitrary model invocations.
_VALID_CHAT_MODELS: frozenset[str] = frozenset({
    # GPT-5.x
    "gpt-5.4", "gpt-5.4-pro", "gpt-5.4-mini", "gpt-5.4-nano",
    "gpt-5", "gpt-5-mini", "gpt-5-nano",
    # GPT-4.1
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    # GPT-4o
    "gpt-4o", "gpt-4o-mini",
    # GPT-4 legacy
    "gpt-4-turbo", "gpt-4",
    # GPT-3.5
    "gpt-3.5-turbo",
    # o-series
    "o4-mini", "o3", "o3-mini", "o1", "o1-mini", "o1-preview",
})

# ─── Embedding model registry ────────────────────────────────────────────────
# Each model gets its own ChromaDB collection so vectors are never mixed.

EMBEDDING_MODELS: dict[str, dict] = {
    "openai": {
        "label": "OpenAI text-embedding-3-small",
        "description": "Best quality · requires API key · cloud",
        "dims": 1536,
        "local": False,
        "collection": "discord_openai",
    },
}

# ─── Runtime globals ─────────────────────────────────────────────────────────

openai_client: Optional[OpenAI] = None
chroma_collections: dict[str, object] = {}   # model_id → chromadb.Collection
current_embedding_model: str = "openai"


# ─── Database helpers ────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")      # enforce FK constraints
    conn.execute("PRAGMA cache_size=-32000")    # 32 MB page cache
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=134217728")  # 128 MB memory-mapped I/O
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            msg_uuid     TEXT    UNIQUE NOT NULL,
            author_id    TEXT,
            username     TEXT,
            date         TEXT,
            content      TEXT,
            attachments  TEXT,
            reactions    TEXT,
            is_suno_team TEXT,
            week         TEXT,
            month        TEXT,
            upload_id    TEXT    NOT NULL,
            row_index    INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS uploads (
            id          TEXT    PRIMARY KEY,
            filename    TEXT    NOT NULL,
            row_count   INTEGER NOT NULL,
            upload_time TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS bookmarks (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            msg_id     INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            ctx_before INTEGER NOT NULL DEFAULT 5,
            ctx_after  INTEGER NOT NULL DEFAULT 5,
            note       TEXT    DEFAULT '',
            created_at TEXT    NOT NULL
        );

        -- Drop the useless B-tree content index (leading-wildcard LIKE cannot use it).
        -- It was wasting ~100MB+ of disk and slowing every INSERT.
        DROP INDEX IF EXISTS idx_content;

        -- Useful structural indexes
        CREATE INDEX IF NOT EXISTS idx_username     ON messages(username COLLATE NOCASE);
        CREATE INDEX IF NOT EXISTS idx_upload_row   ON messages(upload_id, row_index);
        CREATE INDEX IF NOT EXISTS idx_date         ON messages(date);
        CREATE INDEX IF NOT EXISTS idx_msg_uuid     ON messages(msg_uuid);
        CREATE INDEX IF NOT EXISTS idx_suno_team    ON messages(is_suno_team);
        -- Covering index for date-range + suno filter queries
        CREATE INDEX IF NOT EXISTS idx_date_suno    ON messages(date, is_suno_team);
        CREATE INDEX IF NOT EXISTS idx_bookmark_msg ON bookmarks(msg_id);

        CREATE TABLE IF NOT EXISTS labels (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT    NOT NULL UNIQUE,
            color      TEXT    NOT NULL DEFAULT '#6366f1',
            created_at TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS bookmark_labels (
            bookmark_id INTEGER NOT NULL REFERENCES bookmarks(id) ON DELETE CASCADE,
            label_id    INTEGER NOT NULL REFERENCES labels(id)    ON DELETE CASCADE,
            PRIMARY KEY (bookmark_id, label_id)
        );

        -- ── FTS5 full-text search index ──────────────────────────────────────
        -- content='messages' means FTS5 stores no duplicate text; it reads from
        -- the messages table directly.  content_rowid='id' aligns FTS rowids with
        -- messages.id so JOINs are trivial.
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
            USING fts5(content, content='messages', content_rowid='id');

        -- Keep FTS in sync with the messages table via triggers
        CREATE TRIGGER IF NOT EXISTS tg_messages_ai
            AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
            END;
        CREATE TRIGGER IF NOT EXISTS tg_messages_ad
            AFTER DELETE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content)
                    VALUES ('delete', old.id, old.content);
            END;
        CREATE TRIGGER IF NOT EXISTS tg_messages_au
            AFTER UPDATE OF content ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content)
                    VALUES ('delete', old.id, old.content);
                INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
            END;
    """)

    # Rebuild FTS index to cover any rows that existed before FTS was created.
    # This is a fast bulk operation (~1-2 seconds for 600K rows) and is idempotent.
    conn.execute("INSERT INTO messages_fts(messages_fts) VALUES ('rebuild')")
    conn.commit()
    conn.close()


def get_setting(key: str, default: str = "") -> str:
    conn = get_db()
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default


def set_setting(key: str, value: str):
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?,?)", (key, value)
    )
    conn.commit()
    conn.close()


def safe_str(val) -> str:
    """Convert a possibly-NaN pandas value to a clean string."""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip()


# ─── Embedding helpers ───────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using OpenAI text-embedding-3-small.

    This is a synchronous/blocking call — always invoke it via
    ``asyncio.get_event_loop().run_in_executor(None, embed_texts, batch)``
    inside async code so it does not freeze the event loop.
    """
    if not openai_client:
        raise HTTPException(400, "OpenAI API key not configured. Set it in Settings.")
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[t[:8191] for t in texts],
    )
    return [e.embedding for e in response.data]


# Batch size for embedding API calls.
# Smaller batches → more frequent SSE progress events → proxy keepalive.
# OpenAI accepts up to 2048 inputs, but 500 keeps round-trips under ~10 s.
_EMBED_BATCH_SIZE = 500


def active_collection():
    """Return the ChromaDB collection for the currently selected model."""
    col = chroma_collections.get(current_embedding_model)
    if col is None:
        raise HTTPException(503, "Vector store not ready — restart the server and try again.")
    return col


def embedding_model_available() -> bool:
    return openai_client is not None


# ─── App lifecycle ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global openai_client, chroma_collections, current_embedding_model

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        openai_client = OpenAI(api_key=api_key)

    def _init_chroma():
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        logger.info("Using CHROMA_PATH: %s", CHROMA_PATH)
        logger.info("Existing collections: %s", [c.name for c in client.list_collections()])
        cols = {}
        for model_id, cfg in EMBEDDING_MODELS.items():
            try:
                cols[model_id] = client.get_or_create_collection(
                    name=cfg["collection"],
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:
                logger.error("Failed to init ChromaDB collection %s: %s", cfg["collection"], exc)
        return cols

    loop = asyncio.get_running_loop()
    try:
        chroma_cols, _ = await asyncio.gather(
            loop.run_in_executor(None, _init_chroma),
            loop.run_in_executor(None, init_db),
        )
        chroma_collections = chroma_cols
    except Exception as exc:
        logger.error("Startup error (ChromaDB or DB init failed): %s", exc)
        chroma_collections = {}

    # Restore saved model preference
    saved = get_setting("embedding_model", "openai")
    if saved in EMBEDDING_MODELS:
        current_embedding_model = saved

    logger.info("Active embedding model: %s", current_embedding_model)
    yield


app = FastAPI(title="Discord Chat Search", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ─── Security headers middleware ──────────────────────────────────────────────
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

app.add_middleware(_SecurityHeadersMiddleware)


# ─── Optional bearer-token auth middleware ────────────────────────────────────
class _AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not _API_SECRET:
            return await call_next(request)  # auth disabled
        # Always allow the HTML page and static assets
        path = request.url.path
        if path == "/" or path.startswith("/static/"):
            return await call_next(request)
        # Guard all API endpoints
        if path.startswith("/api/"):
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[7:].strip() != _API_SECRET:
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        return await call_next(request)

app.add_middleware(_AuthMiddleware)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s %s\n%s",
                 request.method, request.url,
                 traceback.format_exc())
    # Do NOT expose the exception class or message to the client
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Check server logs for details."},
    )


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/set-api-key")
async def set_api_key(body: dict):
    global openai_client
    key = body.get("api_key", "").strip()
    if not key:
        raise HTTPException(400, "api_key is required")
    openai_client = OpenAI(api_key=key)
    # NOTE: the key is intentionally NOT written to os.environ to avoid
    # leaking it into subprocess environments or log aggregators.
    return {"status": "ok", "message": "API key saved for this session"}


@app.post("/api/set-embedding-model")
async def set_embedding_model(body: dict):
    global current_embedding_model
    model_id = body.get("model_id", "").strip()
    if model_id not in EMBEDDING_MODELS:
        raise HTTPException(400, f"Unknown model '{model_id}'")
    if not embedding_model_available():
        raise HTTPException(
            400,
            "OpenAI API key is not configured. Set it in Settings before selecting an embedding model."
        )
    current_embedding_model = model_id
    set_setting("embedding_model", model_id)
    return {
        "status": "ok",
        "model_id": model_id,
        "label": EMBEDDING_MODELS[model_id]["label"],
    }


@app.get("/api/embedding-models")
async def list_embedding_models():
    result = []
    for mid, cfg in EMBEDDING_MODELS.items():
        try:
            count = chroma_collections[mid].count() if mid in chroma_collections else 0
        except Exception:
            count = 0
        result.append({
            "id": mid,
            **cfg,
            "embedded_count": count,
            "active": mid == current_embedding_model,
            "available": embedding_model_available(),
        })
    return result


@app.get("/api/stats")
async def get_stats():
    conn = get_db()
    total_msgs = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    total_uploads = conn.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
    conn.close()
    try:
        embedded_msgs = active_collection().count()
    except Exception:
        embedded_msgs = 0
    return {
        "total_messages": total_msgs,
        "total_uploads": total_uploads,
        "embedded_messages": embedded_msgs,
        "api_key_set": openai_client is not None,
        "current_model": current_embedding_model,
        "current_model_label": EMBEDDING_MODELS[current_embedding_model]["label"],
    }


def _has_upload_in_collection(collection, upload_id: str) -> bool:
    """Fast presence check: does this upload have any vectors in this collection?"""
    try:
        result = collection.get(where={"upload_id": upload_id}, limit=1, include=["documents"])
        return len(result["ids"]) > 0
    except Exception:
        return False


@app.get("/api/uploads")
async def list_uploads():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM uploads ORDER BY upload_time DESC"
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["embedded_models"] = {
            mid: _has_upload_in_collection(chroma_collections[mid], d["id"])
            for mid in EMBEDDING_MODELS
        }
        result.append(d)
    return result


@app.post("/api/uploads/{upload_id}/reembed")
async def reembed_upload(upload_id: str):
    """Re-embed an existing upload with the currently active model (upsert — safe to re-run)."""
    conn = get_db()
    upload = conn.execute(
        "SELECT * FROM uploads WHERE id = ?", (upload_id,)
    ).fetchone()
    if not upload:
        conn.close()
        raise HTTPException(404, "Upload not found")

    rows = conn.execute(
        "SELECT msg_uuid, content, username, date FROM messages WHERE upload_id = ? AND content != ''",
        (upload_id,),
    ).fetchall()
    conn.close()

    if not rows:
        async def generate():
            yield "data: No messages to embed\n\n"
            yield "data: Completed\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")

    if not embedding_model_available():
        raise HTTPException(
            400,
            "OpenAI API key is required to re-embed. Set it in Settings first."
        )

    texts = [r["content"] for r in rows]
    uuids = [r["msg_uuid"] for r in rows]
    metas = [{"username": r["username"], "date": r["date"], "upload_id": upload_id} for r in rows]

    col = active_collection()

    async def generate():
        try:
            loop = asyncio.get_running_loop()
            yield f"data: Starting embedding {len(texts)} messages with {EMBEDDING_MODELS[current_embedding_model]['label']}\n\n"
            embedded = 0
            batch_num = 0
            for i in range(0, len(texts), _EMBED_BATCH_SIZE):
                b_texts = texts[i : i + _EMBED_BATCH_SIZE]
                b_uuids = uuids[i : i + _EMBED_BATCH_SIZE]
                b_metas = metas[i : i + _EMBED_BATCH_SIZE]
                batch_num += 1
                # SSE comment — ignored by the client but sends bytes to the proxy,
                # resetting its idle-read timer before each blocking API call.
                yield f": batch {batch_num}\n\n"
                try:
                    # Run the blocking OpenAI call in a thread so the event loop
                    # (and SSE flush) is not frozen while waiting for the API.
                    embeddings = await loop.run_in_executor(None, embed_texts, b_texts)
                    await loop.run_in_executor(
                        None,
                        lambda: col.upsert(embeddings=embeddings, documents=b_texts, ids=b_uuids, metadatas=b_metas),
                    )
                    embedded += len(b_texts)
                    yield f"data: Embedded {embedded}/{len(texts)} messages\n\n"
                except Exception as e:
                    logger.warning("Reembed batch %d failed: %s", batch_num, e)
                    yield f"data: Error in batch {batch_num}: {str(e)}\n\n"
            yield f"data: Completed: re-embedded {embedded} messages\n\n"
        except Exception as e:
            logger.error("Reembed generator crashed: %s", e)
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )


@app.delete("/api/uploads/{upload_id}")
async def delete_upload(upload_id: str):
    """Delete an upload and all its messages from SQLite and every ChromaDB collection."""
    conn = get_db()
    upload = conn.execute(
        "SELECT * FROM uploads WHERE id = ?", (upload_id,)
    ).fetchone()
    if not upload:
        conn.close()
        raise HTTPException(404, "Upload not found")

    # Collect all msg_uuids for this upload
    uuid_rows = conn.execute(
        "SELECT msg_uuid FROM messages WHERE upload_id = ?", (upload_id,)
    ).fetchall()
    msg_uuids = [r["msg_uuid"] for r in uuid_rows]

    # Remove from every ChromaDB collection
    if msg_uuids:
        for col in chroma_collections.values():
            try:
                # Batch delete to avoid "too many SQL variables" error
                batch_size = 500
                for i in range(0, len(msg_uuids), batch_size):
                    batch_uuids = msg_uuids[i : i + batch_size]
                    existing = col.get(ids=batch_uuids, include=["documents"])
                    if existing["ids"]:
                        col.delete(ids=existing["ids"])
            except Exception as e:
                logger.error("Chroma delete failed for upload %s: %s", upload_id, e)
                raise HTTPException(500, f"Failed to delete vectors from ChromaDB: {str(e)}")

    # Remove from SQLite
    conn.execute("DELETE FROM messages WHERE upload_id = ?", (upload_id,))
    conn.execute("DELETE FROM uploads WHERE id = ?", (upload_id,))
    conn.commit()
    conn.close()

    return {"status": "ok", "deleted_messages": len(msg_uuids)}


@app.delete("/api/uploads/{upload_id}/sqlite")
async def delete_upload_sqlite(upload_id: str):
    """Delete an upload and its messages from SQLite only — embeddings are preserved."""
    conn = get_db()
    upload = conn.execute(
        "SELECT * FROM uploads WHERE id = ?", (upload_id,)
    ).fetchone()
    if not upload:
        conn.close()
        raise HTTPException(404, "Upload not found")

    msg_count = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE upload_id = ?", (upload_id,)
    ).fetchone()[0]

    conn.execute("DELETE FROM messages WHERE upload_id = ?", (upload_id,))
    conn.execute("DELETE FROM uploads WHERE id = ?", (upload_id,))
    conn.commit()
    conn.close()

    return {"status": "ok", "deleted_messages": msg_count}


@app.delete("/api/uploads/{upload_id}/embeddings")
async def delete_upload_embeddings(upload_id: str):
    """Delete embeddings for an upload from every ChromaDB collection — SQLite untouched."""
    conn = get_db()
    upload = conn.execute(
        "SELECT * FROM uploads WHERE id = ?", (upload_id,)
    ).fetchone()
    if not upload:
        conn.close()
        raise HTTPException(404, "Upload not found")
    conn.close()

    loop = asyncio.get_running_loop()
    deleted_count = 0

    for col in chroma_collections.values():
        try:
            # Use the stored upload_id metadata to delete in one shot.
            # Run in executor so the blocking ChromaDB call doesn't freeze
            # the event loop (and trigger a proxy timeout).
            def _delete_by_upload():
                # Peek at current count so we can report how many were removed.
                before = col.get(
                    where={"upload_id": {"$eq": upload_id}},
                    include=[],          # IDs only — fastest
                ).get("ids", [])
                if before:
                    col.delete(where={"upload_id": {"$eq": upload_id}})
                return len(before)

            n = await loop.run_in_executor(None, _delete_by_upload)
            deleted_count += n
        except Exception as e:
            logger.error("Chroma delete embeddings failed for upload %s: %s", upload_id, e)
            raise HTTPException(500, f"Failed to delete vectors from ChromaDB: {str(e)}")

    return {"status": "ok", "deleted_embeddings": deleted_count}


@app.post("/api/upload")
async def upload_csv(request: Request, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only .csv files are supported")

    # Reject oversized uploads before reading the full body
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB."
        )

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB."
        )

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"Could not parse CSV: {exc}")

    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )

    required = {"author_id", "username", "date", "content"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(400, f"CSV is missing required columns: {missing}")

    for col in ["attachments", "reactions", "is_suno_team", "week", "month"]:
        if col not in df.columns:
            df[col] = ""

    df = df.reset_index(drop=True)
    df["content"] = df["content"].fillna("").astype(str)

    if not embedding_model_available():
        if df["content"].astype(str).str.strip().any():
            raise HTTPException(
                400,
                "OpenAI API key is required to embed messages. Set it in Settings before uploading."
            )

    upload_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    async def generate():
        yield f"data: Processing {len(df)} rows from {file.filename}\n\n"

        conn = get_db()
        conn.execute(
            "INSERT INTO uploads VALUES (?,?,?,?)",
            (upload_id, file.filename, len(df), now),
        )

        rows_inserted = 0
        texts_to_embed: list[str] = []
        uuids_to_embed: list[str] = []
        metas_to_embed: list[dict] = []

        for row_idx, row in df.iterrows():
            msg_uuid = str(uuid.uuid4())
            content = row["content"].strip()

            try:
                conn.execute(
                    """INSERT OR IGNORE INTO messages
                       (msg_uuid, author_id, username, date, content, attachments,
                        reactions, is_suno_team, week, month, upload_id, row_index)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        msg_uuid,
                        safe_str(row.get("author_id")),
                        safe_str(row.get("username")),
                        safe_str(row.get("date")),
                        content,
                        safe_str(row.get("attachments")),
                        safe_str(row.get("reactions")),
                        safe_str(row.get("is_suno_team")),
                        safe_str(row.get("week")),
                        safe_str(row.get("month")),
                        upload_id,
                        int(row_idx),
                    ),
                )
                rows_inserted += 1
            except Exception:
                continue

            if content:
                texts_to_embed.append(content)
                uuids_to_embed.append(msg_uuid)
                metas_to_embed.append({
                    "username": safe_str(row.get("username")),
                    "date": safe_str(row.get("date")),
                    "upload_id": upload_id,
                })

        conn.commit()
        # Update query planner statistics after bulk insert
        conn.execute("ANALYZE messages")
        conn.commit()
        conn.close()

        yield f"data: Inserted {rows_inserted} messages into database\n\n"

        # Embed using the active model into its dedicated collection
        embedded_count = 0
        if embedding_model_available() and texts_to_embed:
            yield f"data: Starting embedding {len(texts_to_embed)} messages with {EMBEDDING_MODELS[current_embedding_model]['label']}\n\n"
            col = active_collection()
            _loop = asyncio.get_running_loop()
            batch_num = 0
            for i in range(0, len(texts_to_embed), _EMBED_BATCH_SIZE):
                b_texts = texts_to_embed[i : i + _EMBED_BATCH_SIZE]
                b_uuids = uuids_to_embed[i : i + _EMBED_BATCH_SIZE]
                b_metas = metas_to_embed[i : i + _EMBED_BATCH_SIZE]
                batch_num += 1
                # SSE keepalive comment — resets proxy idle timer before the blocking call
                yield f": batch {batch_num}\n\n"
                try:
                    embeddings = await _loop.run_in_executor(None, embed_texts, b_texts)
                    await _loop.run_in_executor(
                        None,
                        lambda: col.add(embeddings=embeddings, documents=b_texts, ids=b_uuids, metadatas=b_metas),
                    )
                    embedded_count += len(b_texts)
                    yield f"data: Embedded {embedded_count}/{len(texts_to_embed)} messages\n\n"
                except Exception as e:
                    logger.warning("Embedding batch %d failed: %s", batch_num, e)
                    yield f"data: Error in batch {batch_num}: {str(e)}\n\n"
        elif not embedding_model_available():
            yield "data: Skipping embedding (no OpenAI API key configured)\n\n"
        else:
            yield "data: No messages to embed\n\n"

        yield f"data: Completed: inserted {rows_inserted}, embedded {embedded_count}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )


# ─── FTS5 keyword helper ─────────────────────────────────────────────────────

def _build_fts_query(keyword: str) -> str:
    """
    Convert a plain keyword string into a safe FTS5 MATCH expression.

    Strategy:
      - Strip FTS5 syntax chars to prevent injection/parse errors.
      - Single words   → prefix match  (e.g.  feat → "feat"*)
      - Multi-word     → phrase match  (e.g. "new feature" → "new feature")
    """
    # Remove FTS5 operators / special chars
    safe = re.sub(r'["\'*^()\[\]{};:\\]', ' ', keyword).strip()
    if not safe:
        raise ValueError("Empty keyword after FTS5 sanitization")
    words = safe.split()
    if len(words) == 1:
        return f'"{words[0]}"*'   # prefix wildcard
    # Phrase — double-quote the whole phrase (no wildcard; FTS5 phrase = exact order)
    inner = " ".join(words)
    return f'"{inner}"'


# ─── SQL filter helpers ──────────────────────────────────────────────────────

def sql_date_clauses(date_from: Optional[str], date_to: Optional[str]) -> tuple[str, list]:
    sql, params = "", []
    if date_from:
        sql += " AND substr(date, 1, 10) >= ?"
        params.append(date_from[:10])
    if date_to:
        sql += " AND substr(date, 1, 10) <= ?"
        params.append(date_to[:10])
    return sql, params


def sql_min_words_clause(min_words: int) -> tuple[str, list]:
    """Return SQL fragment filtering by minimum word count (space-count+1 heuristic)."""
    if min_words and min_words > 1:
        return (
            " AND (length(trim(content)) - length(replace(trim(content), ' ', '')) + 1) >= ?",
            [min_words],
        )
    return "", []


def date_in_range(date_str: str, date_from: Optional[str], date_to: Optional[str]) -> bool:
    d = (date_str or "")[:10]
    if date_from and d < date_from[:10]:
        return False
    if date_to and d > date_to[:10]:
        return False
    return True


SUNO_TEAM_SQL         = " AND LOWER(is_suno_team) IN ('true', '1')"
EXCLUDE_SUNO_TEAM_SQL = " AND (is_suno_team IS NULL OR LOWER(is_suno_team) NOT IN ('true', '1'))"


def _suno_sql(suno_team: str) -> str:
    """Return the appropriate SQL fragment for the suno_team filter value."""
    if suno_team == "only":
        return SUNO_TEAM_SQL
    if suno_team == "exclude":
        return EXCLUDE_SUNO_TEAM_SQL
    return ""


def is_suno_team_member(val: str) -> bool:
    return (val or "").lower() in ("true", "1")


# ─── Search endpoints ────────────────────────────────────────────────────────

def _parse_upload_ids(upload_ids) -> list[str]:
    """Accept a comma-string, a list, or empty/None. Always returns a clean list."""
    if not upload_ids:
        return []
    if isinstance(upload_ids, list):
        return [str(x).strip() for x in upload_ids if str(x).strip()]
    return [x.strip() for x in str(upload_ids).split(",") if x.strip()]


def _sql_upload_ids_clause(uid_list: list[str]) -> tuple[str, list]:
    if not uid_list:
        return "", []
    placeholders = ",".join("?" * len(uid_list))
    return f" AND upload_id IN ({placeholders})", uid_list


@app.get("/api/search/username")
async def search_by_username(
    username: str,
    upload_ids: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    suno_team: str = "all",
    min_words: int = 0,
    limit: int = 200,
):
    uid_list = _parse_upload_ids(upload_ids)
    uid_sql, uid_params = _sql_upload_ids_clause(uid_list)
    date_sql, date_params = sql_date_clauses(date_from, date_to)
    words_sql, words_params = sql_min_words_clause(min_words)
    sql = (
        "SELECT * FROM messages WHERE LOWER(username) LIKE LOWER(?)"
        + uid_sql
        + _suno_sql(suno_team)
        + date_sql
        + words_sql
        + " ORDER BY date, row_index LIMIT ?"
    )
    conn = get_db()
    rows = conn.execute(sql, [f"%{username}%"] + uid_params + date_params + words_params + [limit]).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/search/keyword")
async def search_by_keyword(
    keyword: str,
    upload_ids: Optional[str] = None,
    username: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    suno_team: str = "all",
    min_words: int = 0,
    limit: int = 200,
):
    uid_list = _parse_upload_ids(upload_ids)
    date_sql, date_params = sql_date_clauses(date_from, date_to)
    words_sql, words_params = sql_min_words_clause(min_words)

    conn = get_db()
    try:
        # ── Fast path: FTS5 two-step search ───────────────────────────────────
        # Step 1 — inverted index lookup: get candidate row IDs from FTS5.
        #   Over-fetch so post-filtering still returns enough rows.
        fts_expr       = _build_fts_query(keyword)
        candidate_cap  = limit * 20          # fetch up to 20× to survive post-filters
        fts_rows = conn.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ? LIMIT ?",
            [fts_expr, candidate_cap],
        ).fetchall()

        if not fts_rows:
            return []

        # Step 2 — filter candidates with all other predicates.
        candidate_ids = [r[0] for r in fts_rows]
        ph = ",".join("?" * len(candidate_ids))
        params: list = list(candidate_ids)
        sql = f"SELECT * FROM messages WHERE id IN ({ph})"

        if username:
            sql += " AND LOWER(username) LIKE LOWER(?)"
            params.append(f"%{username}%")
        if uid_list:
            uid_ph = ",".join("?" * len(uid_list))
            sql += f" AND upload_id IN ({uid_ph})"
            params.extend(uid_list)
        sql += _suno_sql(suno_team)
        sql += date_sql
        params.extend(date_params)
        sql += words_sql
        params.extend(words_params)
        sql += " ORDER BY date, row_index LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()

    except Exception as fts_exc:
        # ── Slow fallback: LIKE scan (used if FTS5 table is not yet populated) ─
        logger.warning("FTS5 keyword search failed (%s), falling back to LIKE", fts_exc)
        uid_sql, uid_params = _sql_upload_ids_clause(uid_list)
        params_fb: list = [f"%{keyword}%"]
        sql_fb = "SELECT * FROM messages WHERE LOWER(content) LIKE LOWER(?)"
        if username:
            sql_fb += " AND LOWER(username) LIKE LOWER(?)"
            params_fb.append(f"%{username}%")
        sql_fb += uid_sql
        params_fb.extend(uid_params)
        sql_fb += _suno_sql(suno_team) + date_sql + words_sql
        params_fb.extend(date_params + words_params)
        sql_fb += " ORDER BY date, row_index LIMIT ?"
        params_fb.append(limit)
        rows = conn.execute(sql_fb, params_fb).fetchall()

    finally:
        conn.close()

    return [dict(r) for r in rows]


@app.get("/api/search/range")
async def search_by_range(
    upload_ids: Optional[str] = None,
    username: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    suno_team: str = "all",
    min_words: int = 0,
    limit: Optional[int] = None,
):
    uid_list = _parse_upload_ids(upload_ids)
    uid_sql, uid_params = _sql_upload_ids_clause(uid_list)
    date_sql, date_params = sql_date_clauses(date_from, date_to)
    words_sql, words_params = sql_min_words_clause(min_words)

    params: list = []
    sql = "SELECT * FROM messages WHERE 1=1"
    if username:
        sql += " AND LOWER(username) LIKE LOWER(?)"
        params.append(f"%{username}%")
    sql += uid_sql
    params.extend(uid_params)
    sql += _suno_sql(suno_team)
    sql += date_sql
    params.extend(date_params)
    sql += words_sql
    params.extend(words_params)
    sql += " ORDER BY date, row_index"
    if limit is not None and limit > 0:
        sql += " LIMIT ?"
        params.append(limit)

    conn = get_db()
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/search/semantic")
async def search_semantic(
    query: str,
    upload_ids: Optional[str] = None,
    n_results: int = 20,
    username: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    suno_team: str = "all",
    min_words: int = 0,
):
    uid_list = _parse_upload_ids(upload_ids)

    col = active_collection()
    total = col.count()
    if total == 0:
        raise HTTPException(
            400,
            f"No messages are embedded with the current model "
            f"({EMBEDDING_MODELS[current_embedding_model]['label']}). "
            "Upload or re-embed data with this model selected first."
        )

    has_filters = bool(username or date_from or date_to or (suno_team != "all") or uid_list or min_words)
    fetch_n = min(n_results * 4 if has_filters else n_results, total)
    query_emb = embed_texts([query])[0]

    results = col.query(query_embeddings=[query_emb], n_results=fetch_n)
    ids = results["ids"][0]
    distances = results["distances"][0]

    # Batch-fetch all rows in one query instead of N individual lookups
    conn = get_db()
    uuid_to_dist = dict(zip(ids, distances))
    if ids:
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT * FROM messages WHERE msg_uuid IN ({placeholders})", ids
        ).fetchall()
        uuid_to_row = {r["msg_uuid"]: dict(r) for r in rows}
    else:
        uuid_to_row = {}
    conn.close()

    messages = []
    for msg_uuid in ids:
        msg = uuid_to_row.get(msg_uuid)
        if msg is None:
            continue
        msg["similarity_score"] = round(1.0 - uuid_to_dist[msg_uuid], 4)

        if uid_list and msg["upload_id"] not in uid_list:
            continue
        if username and username.lower() not in msg["username"].lower():
            continue
        if not date_in_range(msg["date"], date_from, date_to):
            continue
        if suno_team == "only" and not is_suno_team_member(msg["is_suno_team"]):
            continue
        if suno_team == "exclude" and is_suno_team_member(msg["is_suno_team"]):
            continue
        if min_words > 1:
            wc = len((msg["content"] or "").split())
            if wc < min_words:
                continue

        messages.append(msg)
        if len(messages) >= n_results:
            break

    return messages



@app.post("/api/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    message = (body.get("message") or "").strip()
    history = body.get("history") or []
    upload_ids = body.get("upload_ids") or ""
    chat_model = (body.get("model") or "gpt-5.4").strip()

    if not message:
        raise HTTPException(status_code=400, detail="Empty message.")
    if not openai_client:
        raise HTTPException(status_code=400, detail="OpenAI API key not set — add it in Settings.")
    if chat_model not in _VALID_CHAT_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{chat_model}'.")

    uid_list = _parse_upload_ids(upload_ids)

    def _semantic_search() -> list[dict]:
        rows: list[dict] = []
        try:
            col = active_collection()
            if col.count() == 0:
                return rows
            query_emb = embed_texts([message])[0]
            results = col.query(query_embeddings=[query_emb], n_results=12)
            ids = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0]
            uuid_map: dict[str, dict] = {}
            if ids:
                conn = get_db()
                try:
                    ph = ",".join("?" * len(ids))
                    db_rows = conn.execute(
                        f"SELECT * FROM messages WHERE msg_uuid IN ({ph})", ids
                    ).fetchall()
                    uuid_map = {r["msg_uuid"]: dict(r) for r in db_rows}
                finally:
                    conn.close()
            for uid, dist in zip(ids, distances):
                row = uuid_map.get(uid)
                if row is None:
                    continue
                if uid_list and row.get("upload_id") not in uid_list:
                    continue
                score = None
                if dist is not None:
                    try:
                        score = round(1.0 - float(dist), 4)
                    except (TypeError, ValueError):
                        score = None
                row["_score"] = score
                rows.append(row)
        except Exception as e:
            logger.warning("/api/chat semantic retrieval error: %s", e)
        return rows

    async def _do_keyword_search() -> list[dict]:
        try:
            return await search_by_keyword(keyword=message, upload_ids=upload_ids, limit=10)
        except Exception as e:
            logger.warning("/api/chat keyword search error: %s", e)
            return []

    # Run slow semantic search in thread pool; keyword search is fast SQLite — run as coroutine
    loop = asyncio.get_running_loop()
    semantic_rows, keyword_rows = await asyncio.gather(
        loop.run_in_executor(None, _semantic_search),
        _do_keyword_search(),
    )

    context_rows = list(semantic_rows)
    existing_uuids = {r["msg_uuid"] for r in context_rows}
    for r in keyword_rows:
        if r["msg_uuid"] not in existing_uuids:
            context_rows.append(r)
            existing_uuids.add(r["msg_uuid"])

    # Limit total context to 20 messages
    context_rows = context_rows[:20]

    if context_rows:
        ctx_text = "\n".join(
            f"[{r['username']} | {r['date']}] {r['content']}"
            for r in context_rows
        )
        system = """You are a knowledgeable assistant for the Suno AI Discord community.

INSTRUCTIONS:
- Use the retrieved conversation excerpts below as your PRIMARY source of truth.
- Cite specific usernames (e.g. **@username**) when referencing their messages.
- If the context does not cover the question, say so clearly before answering from general knowledge.

MANDATORY FORMATTING — your entire response MUST be valid Markdown:
- Start with a `##` heading that summarises the answer topic.
- Use `###` subheadings to separate distinct sub-topics.
- Use **bold** for key terms, usernames, and important points.
- Use `-` bullet lists for multiple items or steps; use `1.` numbered lists for sequences.
- Use `> blockquote` to highlight a direct or paraphrased user quote.
- Use `inline code` for technical terms, settings, or commands.
- End with a `---` rule followed by a brief *Sources* section listing the cited usernames and dates.
- Do NOT output plain prose paragraphs without any formatting.

RETRIEVED CONTEXT:
""" + ctx_text
    else:
        system = """You are a helpful assistant for the Suno AI Discord community.
No embedded messages are available — answer from general knowledge.

MANDATORY FORMATTING — your entire response MUST be valid Markdown:
- Start with a `##` heading.
- Use **bold**, `-` bullet lists, `###` subheadings, and `inline code` where appropriate.
- Do NOT output plain prose without any Markdown structure.
"""

    is_o_model = chat_model.startswith("o")
    sys_role = "developer" if is_o_model else "system"
    msgs = [{"role": sys_role, "content": system}]
    for turn in history[-20:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in {"system", "user", "assistant"} and content:
            msgs.append({"role": role, "content": content})

    msgs.append({"role": "user", "content": message})

    async def generate():
        try:
            stream = openai_client.chat.completions.create(
                model=chat_model, messages=msgs, stream=True
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'content': delta})}\n\n"
        except Exception as e:
            logger.error("generate() error: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )

@app.post("/api/summarize")
async def summarize_endpoint(request: Request):
    body           = await request.json()
    username       = (body.get("username") or "").strip()
    date_from      = (body.get("date_from") or "").strip()
    date_to        = (body.get("date_to") or "").strip()
    prompt_txt     = (body.get("prompt") or "").strip()
    upload_ids     = (body.get("upload_ids") or "")
    min_words      = int(body.get("min_words") or 0)
    suno_team = str(body.get("suno_team") or "all")
    sum_model      = (body.get("model") or "gpt-5.4").strip()

    if not openai_client:
        raise HTTPException(400, "OpenAI API key not set — add it in Settings.")
    if sum_model not in _VALID_CHAT_MODELS:
        raise HTTPException(400, f"Unknown model '{sum_model}'.")

    uid_list = _parse_upload_ids(upload_ids)
    uid_sql, uid_params = _sql_upload_ids_clause(uid_list)
    date_sql, date_params = sql_date_clauses(date_from, date_to)
    words_sql, words_params = sql_min_words_clause(min_words)

    params: list = []
    sql = "SELECT username, date, content FROM messages WHERE 1=1"
    if username:
        sql += " AND LOWER(username) LIKE LOWER(?)"
        params.append(f"%{username}%")
    sql += uid_sql
    params.extend(uid_params)
    sql += _suno_sql(suno_team)
    sql += date_sql
    params.extend(date_params)
    sql += words_sql
    params.extend(words_params)
    sql += " ORDER BY date, row_index"

    conn = get_db()
    rows = conn.execute(sql, params).fetchall()
    conn.close()

    if not rows:
        raise HTTPException(404, "No messages found matching those filters.")

    conv = "\n".join(
        f"[{r['username']} | {r['date']}]: {r['content']}"
        for r in rows
    )

    default_prompt = """\
Produce a comprehensive summary of the Discord conversation below.

MANDATORY STRUCTURE (strictly follow this Markdown layout):

## Overview
One short paragraph giving the high-level context.

## Key Topics
For each major topic:
### [Topic Name]
- Bullet points covering the main discussion points.
- Use **bold** for important terms or conclusions.

## Notable Opinions & Insights
> Direct or paraphrased quotes from participants, formatted as blockquotes, with **@username** attributed.

## Decisions / Conclusions
- Any outcomes, agreed next steps, or unresolved questions.

## Participants
- List unique usernames who contributed meaningfully.

---
Do NOT output plain paragraphs. Every section must use the Markdown elements above.\
"""
    user_prompt = prompt_txt or default_prompt
    full = f"{user_prompt}\n\nCONVERSATION ({len(rows)} messages):\n{conv}"

    is_o_model = sum_model.startswith("o")
    sys_role = "developer" if is_o_model else "system"

    async def generate():
        try:
            stream = openai_client.chat.completions.create(
                model=sum_model,
                messages=[
                    {
                        "role": sys_role,
                        "content": (
                            "You are an expert analyst summarising Discord conversations from the Suno AI community. "
                            "You MUST respond exclusively in well-structured Markdown. "
                            "Never output plain prose. Always use ## headings, ### subheadings, "
                            "**bold**, - bullet lists, > blockquotes, and `code` where appropriate."
                        ),
                    },
                    {"role": "user", "content": full},
                ],
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'content': delta})}\n\n"
        except Exception as e:
            logger.error("generate() error: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        yield "data: [DONE]\n\n"
    

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


@app.get("/api/context/{message_id}")
async def get_context(message_id: int, before: int = 5, after: int = 5):
    before = max(0, min(before, 200))   # clamp: 0–200
    after  = max(0, min(after,  200))
    conn = get_db()
    target = conn.execute(
        "SELECT * FROM messages WHERE id = ?", (message_id,)
    ).fetchone()

    if not target:
        conn.close()
        raise HTTPException(404, "Message not found")

    target = dict(target)
    upload_id = target["upload_id"]
    row_idx = target["row_index"]

    context_rows = conn.execute(
        """SELECT * FROM messages
           WHERE upload_id = ? AND row_index BETWEEN ? AND ?
           ORDER BY row_index""",
        (upload_id, max(0, row_idx - before), row_idx + after),
    ).fetchall()
    conn.close()

    return [
        {**dict(r), "is_target": (r["id"] == message_id)}
        for r in context_rows
    ]


# ─── Bookmarks ───────────────────────────────────────────────────────────────

@app.post("/api/bookmarks")
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


@app.get("/api/bookmarks")
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
        d = dict(r)
        d["labels"] = labels_by_bm.get(d["bookmark_id"], [])
        result.append(d)
    return result


@app.get("/api/bookmarks/ids")
async def list_bookmark_ids():
    """Return only the bookmarked message IDs — cheap check for UI state."""
    conn = get_db()
    rows = conn.execute("SELECT msg_id FROM bookmarks").fetchall()
    conn.close()
    return [r["msg_id"] for r in rows]


@app.delete("/api/bookmarks/{bookmark_id}")
async def delete_bookmark(bookmark_id: int):
    conn = get_db()
    cur = conn.execute("DELETE FROM bookmarks WHERE id=?", (bookmark_id,))
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        raise HTTPException(404, "Bookmark not found")
    return {"status": "deleted"}


@app.delete("/api/bookmarks/by-msg/{msg_id}")
async def delete_bookmark_by_msg(msg_id: int):
    conn = get_db()
    cur = conn.execute("DELETE FROM bookmarks WHERE msg_id=?", (msg_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted", "affected": cur.rowcount}


# ─── In-results semantic filter ──────────────────────────────────────────────

# Stop-words + question-opener words stripped before embedding question queries.
_FILTER_STOP = frozenset([
    "what", "how", "why", "when", "where", "who", "which", "whose", "whom",
    "is", "are", "was", "were", "does", "do", "did", "will", "would",
    "can", "could", "should", "shall", "may", "might", "has", "have", "had",
    "the", "a", "an", "of", "to", "for", "in", "on", "at", "by", "with",
    "from", "into", "about", "and", "or", "but", "if", "then", "that", "this",
    "it", "its", "their", "there", "these", "those", "be", "been", "being",
    "i", "you", "we", "they", "he", "she", "me", "him", "her", "us", "them",
    "any", "some", "no", "not", "just", "also", "so", "as", "up",
])

_QUESTION_RE = re.compile(
    r"^(what|how|why|when|where|who|which|is|are|was|were|does|do|did"
    r"|will|would|can|could|should|shall|has|have|had)\b",
    re.IGNORECASE,
)


def _prepare_filter_query(raw: str) -> tuple[str, float]:
    """
    Return (text_to_embed, threshold).

    For question queries (detected by opener word or trailing ?) we strip
    stop/question words to expose the core semantic terms, and use a lower
    similarity threshold (0.30) so that relevant messages are not missed.
    For plain keyword/phrase queries we embed as-is with the standard 0.45
    threshold (slightly relaxed from the old 0.50 to improve recall).
    """
    is_question = bool(_QUESTION_RE.match(raw.strip())) or raw.strip().endswith("?")

    if is_question:
        tokens = re.findall(r"\b\w+\b", raw.lower())
        core   = [t for t in tokens if t not in _FILTER_STOP and len(t) > 1]
        embed_text = " ".join(core) if core else raw
        threshold  = 0.30
    else:
        embed_text = raw
        threshold  = 0.45

    return embed_text, threshold


@app.post("/api/filter/semantic")
async def filter_semantic(body: dict):
    query   = (body.get("query") or "").strip()
    raw_ids = body.get("msg_ids") or []

    if not query:
        raise HTTPException(400, "query is required")
    if not raw_ids:
        return {"results": [], "threshold": 0.45, "query_used": query}

    # Validate that all msg_ids are integers to prevent type confusion attacks
    try:
        msg_ids = [int(x) for x in raw_ids]
    except (TypeError, ValueError):
        raise HTTPException(400, "msg_ids must be a list of integers")

    embed_text, threshold = _prepare_filter_query(query)

    col = active_collection()
    if col.count() == 0:
        raise HTTPException(400, "No embeddings found for the active model. "
                                 "Upload and embed data first, then retry.")

    # Resolve msg_ids → msg_uuids
    conn = get_db()
    placeholders = ",".join("?" * len(msg_ids))
    rows = conn.execute(
        f"SELECT id, msg_uuid FROM messages WHERE id IN ({placeholders})",
        msg_ids,
    ).fetchall()
    conn.close()

    if not rows:
        return {"results": [], "threshold": threshold, "query_used": embed_text}

    id_to_uuid = {r["id"]: r["msg_uuid"] for r in rows}
    uuid_to_id = {v: k for k, v in id_to_uuid.items()}
    uuids      = list(id_to_uuid.values())

    # Embed the query
    loop = asyncio.get_running_loop()
    query_vec = await loop.run_in_executor(
        None, lambda: embed_texts([embed_text])[0]
    )

    # ── Directly fetch stored embeddings by ID (avoids ChromaDB $in filter bugs) ──
    # col.get() is reliable for any list size; col.query() with $in can silently
    # drop results when the candidate list is large.
    stored = col.get(ids=uuids, include=["embeddings"])
    stored_ids  = stored.get("ids") or []
    stored_embs = stored.get("embeddings")
    if stored_embs is None:
        stored_embs = []

    if not stored_ids or len(stored_embs) == 0:
        return {
            "results": [],
            "threshold": threshold,
            "query_used": embed_text,
            "warning": "No embeddings found for these messages. Re-embed the upload in Config first.",
        }

    # Cosine similarity = dot product of L2-normalised vectors (faster than full cosine formula)
    q = np.array(query_vec, dtype=np.float32)
    hits = []
    for uid, emb in zip(stored_ids, stored_embs):
        if uid not in uuid_to_id:
            continue
        sim = float(np.dot(q, np.array(emb, dtype=np.float32)))
        if sim >= threshold:
            hits.append({"id": uuid_to_id[uid], "score": round(sim, 4)})

    hits.sort(key=lambda x: x["score"], reverse=True)
    return {"results": hits, "threshold": threshold, "query_used": embed_text}


# ─── Suno Team management ────────────────────────────────────────────────────

@app.get("/api/suno-team")
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


@app.delete("/api/suno-team/{username}")
async def remove_suno_team(username: str):
    """Mark all messages by this username as non-Suno-Team."""
    conn = get_db()
    result = conn.execute(
        "UPDATE messages SET is_suno_team = 'false' WHERE username = ?",
        (username,),
    )
    conn.commit()
    affected = result.rowcount
    conn.close()
    return {"username": username, "updated": affected}


# ─── Labels ──────────────────────────────────────────────────────────────────

@app.get("/api/labels")
async def list_labels():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, color, created_at FROM labels ORDER BY name COLLATE NOCASE"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/api/labels")
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


@app.delete("/api/labels/{label_id}")
async def delete_label(label_id: int):
    conn = get_db()
    cur = conn.execute("DELETE FROM labels WHERE id=?", (label_id,))
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        raise HTTPException(404, "Label not found")
    return {"status": "deleted"}


@app.post("/api/bookmarks/{bookmark_id}/labels/{label_id}")
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


@app.delete("/api/bookmarks/{bookmark_id}/labels/{label_id}")
async def unassign_label(bookmark_id: int, label_id: int):
    conn = get_db()
    conn.execute(
        "DELETE FROM bookmark_labels WHERE bookmark_id=? AND label_id=?",
        (bookmark_id, label_id),
    )
    conn.commit()
    conn.close()
    return {"status": "unassigned"}
