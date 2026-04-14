"""
routers/stats.py — server-side stats endpoint.

  GET /api/stats
"""

import asyncio
import time as _time

from fastapi import APIRouter

import state
from config import EMBEDDING_MODELS
from database import get_db
from embeddings import active_collection

router = APIRouter()


@router.get("/api/stats")
async def get_stats():
    cached, ts = state.get_stats_cache()
    if cached and (_time.monotonic() - ts) < state.STATS_TTL:
        return cached

    loop = asyncio.get_running_loop()

    def _db_counts():
        conn = get_db()
        tm = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        tu = conn.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
        conn.close()
        return tm, tu

    def _vector_count():
        try:
            return active_collection().count()
        except Exception:
            return 0

    (total_msgs, total_uploads), embedded_msgs = await asyncio.gather(
        loop.run_in_executor(None, _db_counts),
        loop.run_in_executor(None, _vector_count),
    )

    result = {
        "total_messages":      total_msgs,
        "total_uploads":       total_uploads,
        "embedded_messages":   embedded_msgs,
        "api_key_set":         state.openai_client is not None,
        "current_model":       state.current_embedding_model,
        "current_model_label": EMBEDDING_MODELS[state.current_embedding_model]["label"],
    }
    state.set_stats_cache(result)
    return result
