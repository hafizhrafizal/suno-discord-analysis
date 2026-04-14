"""
state.py — shared mutable runtime state.

All modules that need to read or write the live OpenAI clients,
vector-store collections, embed-job registry, or response caches
import from here.  Using module-level variables keeps references
consistent: `state.openai_client = x` is visible to every importer.
"""

import time as _time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# Populated during lifespan startup; None until then.
openai_client:        Optional[object] = None   # openai.OpenAI  (sync — chat completions)
async_openai_client:  Optional[object] = None   # openai.AsyncOpenAI (async — embeddings)

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
