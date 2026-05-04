"""
state.py — shared mutable runtime state.

All modules that need to read or write the live OpenAI clients,
vector-store collections, embed-job registry, or response caches
import from here.  Using module-level variables keeps references
consistent: `state.openai_client = x` is visible to every importer.

Multi-user mode uses ContextVar to inject per-request OpenAI clients.
`get_openai_client()` / `get_async_openai_client()` are the preferred
accessors — they return the request-scoped client when set, else the
global fallback.
"""

import time as _time
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from typing import Optional, Tuple

# ── App mode ──────────────────────────────────────────────────────────────────
# Set at startup from APP_MODE env or the "app_mode" DB setting.
# Values: "single" | "multi" | "pending_onboarding"
app_mode: str = ""

# ── Global OpenAI clients (single mode / env-key startup) ────────────────────
# Populated during lifespan startup; None until then.
openai_client:        Optional[object] = None   # openai.OpenAI  (sync — chat completions)
async_openai_client:  Optional[object] = None   # openai.AsyncOpenAI (async — embeddings)

# ── Per-user client cache (multi mode) ───────────────────────────────────────
# user_id → (sync OpenAI client, async OpenAI client)
user_clients: dict = {}

# ── Per-request ContextVar (multi mode) ──────────────────────────────────────
# Holds (sync_client, async_client) for the duration of a single HTTP request.
# Set by _SessionAuthMiddleware; reset after call_next completes.
_current_clients: ContextVar[Optional[Tuple[object, object]]] = ContextVar(
    "_current_clients", default=None
)


def get_openai_client():
    """Return the sync OpenAI client for the current request (multi) or global (single)."""
    pair = _current_clients.get()
    return pair[0] if pair is not None else openai_client


def get_async_openai_client():
    """Return the async OpenAI client for the current request (multi) or global (single)."""
    pair = _current_clients.get()
    return pair[1] if pair is not None else async_openai_client


def set_request_clients(sync_c, async_c) -> object:
    """Set per-request clients; returns the ContextVar token for later reset."""
    return _current_clients.set((sync_c, async_c))

# model_id → *CollectionWrapper (Qdrant or Chroma)
vector_collections: dict[str, object] = {}

# Currently active embedding model key (matches EMBEDDING_MODELS keys in config.py)
current_embedding_model: str = "openai"

# Dedicated thread-pool for blocking vector-store calls.
# Kept separate so embedding concurrency never competes with store I/O.
vector_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="vectorstore")

# ── Background embed job registry ─────────────────────────────────────────────
# job_id → {status, embedded, total, skipped, batch_errors, error, traceback}
embed_jobs: dict[str, dict] = {}
# upload_id → job_id  (only while the job is running; cleared on finish)
active_embed: dict[str, str] = {}

# ── Stats / uploads response cache ───────────────────────────────────────────
_stats_cache:     dict | None = None
_stats_cache_ts:  float       = 0.0
_uploads_cache:   list | None = None
_uploads_cache_ts: float      = 0.0
STATS_TTL                     = 30.0   # seconds


def invalidate_stats_cache() -> None:
    global _stats_cache
    _stats_cache = None


def invalidate_uploads_cache() -> None:
    global _uploads_cache
    _uploads_cache = None


def invalidate_all_caches() -> None:
    invalidate_stats_cache()
    invalidate_uploads_cache()


def get_stats_cache():
    return _stats_cache, _stats_cache_ts


def set_stats_cache(value: dict) -> None:
    global _stats_cache, _stats_cache_ts
    _stats_cache    = value
    _stats_cache_ts = _time.monotonic()


def get_uploads_cache():
    return _uploads_cache, _uploads_cache_ts


def set_uploads_cache(value: list) -> None:
    global _uploads_cache, _uploads_cache_ts
    _uploads_cache    = value
    _uploads_cache_ts = _time.monotonic()
