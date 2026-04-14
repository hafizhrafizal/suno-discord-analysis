"""
config.py — all environment variables and application-wide constants.

All other modules import from here; nothing here imports from the project.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH          = os.environ.get("DB_PATH", "discord_data.db")

# ── Upload limits ─────────────────────────────────────────────────────────────
MAX_UPLOAD_MB    = int(os.environ.get("MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1_048_576

# ── API security ──────────────────────────────────────────────────────────────
API_SECRET       = os.environ.get("API_SECRET", "").strip()

# ── Vector backend ────────────────────────────────────────────────────────────
# Valid values: qdrant | chroma_persistent | chroma_http
VECTOR_DB        = os.environ.get("VECTOR_DB", "qdrant").strip().lower()

# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_URL       = os.environ.get("QDRANT_URL", "").strip()
QDRANT_API_KEY   = os.environ.get("QDRANT_API_KEY", "").strip()

# ── ChromaDB persistent ───────────────────────────────────────────────────────
CHROMA_PATH      = os.environ.get("CHROMA_PATH", "./chroma_db").strip()

# ── ChromaDB HTTP ─────────────────────────────────────────────────────────────
CHROMA_HOST       = os.environ.get("CHROMA_HOST", "localhost").strip()
CHROMA_PORT       = int(os.environ.get("CHROMA_PORT", "8001"))
CHROMA_SSL        = os.environ.get("CHROMA_SSL", "false").strip().lower() == "true"
CHROMA_AUTH_TOKEN = os.environ.get("CHROMA_AUTH_TOKEN", "").strip()

# ── Embedding performance ─────────────────────────────────────────────────────
# OpenAI accepts up to 2048 inputs per embedding request.
EMBED_BATCH_SIZE   = int(os.environ.get("EMBED_BATCH_SIZE", "2048"))
# How many embedding API calls to fire concurrently.
EMBED_CONCURRENCY  = int(os.environ.get("EMBED_CONCURRENCY", "10"))

# ── Chat model allowlist ──────────────────────────────────────────────────────
# Only permit known OpenAI model IDs to prevent arbitrary model invocations.
VALID_CHAT_MODELS: frozenset[str] = frozenset({
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

# ── Embedding model registry ──────────────────────────────────────────────────
# Each model gets its own vector-store collection so vectors are never mixed.
EMBEDDING_MODELS: dict[str, dict] = {
    "openai": {
        "label":       "OpenAI text-embedding-3-small",
        "description": "Best quality · requires API key · cloud",
        "dims":        1536,
        "local":       False,
        "collection":  "discord_openai",
    },
}
