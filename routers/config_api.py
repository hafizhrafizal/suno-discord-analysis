"""
routers/config_api.py — API-key management and embedding-model selection.

  POST /api/set-api-key         — single mode: set global key; multi mode: update user key
  POST /api/set-embedding-model
  GET  /api/embedding-models
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from openai import AsyncOpenAI, OpenAI

import state
from config import EMBEDDING_MODELS
from database import get_session_user, set_setting
from embeddings import embedding_model_available

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/set-api-key")
async def set_api_key(body: dict, request: Request):
    key = (body.get("api_key") or "").strip()
    if not key:
        raise HTTPException(400, "api_key is required")

    if state.app_mode == "multi":
        # Multi mode: store key in memory only — never persisted to the database.
        # The client (browser localStorage) is the sole durable store.
        token = request.cookies.get("session")
        user  = get_session_user(token) if token else None
        if not user:
            raise HTTPException(401, "Not authenticated.")

        uid = user["id"]

        # Evict old cached client
        old = state.user_clients.pop(uid, None)
        if old:
            import asyncio
            async def _close():
                try: await old[1].close()
                except Exception: pass
            asyncio.create_task(_close())

        new_pair = (OpenAI(api_key=key), AsyncOpenAI(api_key=key))
        state.user_clients[uid] = new_pair
        state.set_request_clients(*new_pair)
        return {"status": "ok", "message": "API key set for this session."}

    # Single mode: set global clients as before.
    if state.async_openai_client:
        await state.async_openai_client.close()
    state.openai_client       = OpenAI(api_key=key)
    state.async_openai_client = AsyncOpenAI(api_key=key)
    return {"status": "ok", "message": "API key saved for this session"}


@router.post("/api/set-embedding-model")
async def set_embedding_model(body: dict):
    model_id = body.get("model_id", "").strip()
    if model_id not in EMBEDDING_MODELS:
        raise HTTPException(400, f"Unknown model '{model_id}'")
    if not embedding_model_available():
        raise HTTPException(
            400,
            "OpenAI API key is not configured. Set it in Settings before selecting an embedding model.",
        )
    state.current_embedding_model = model_id
    set_setting("embedding_model", model_id)
    return {
        "status":   "ok",
        "model_id": model_id,
        "label":    EMBEDDING_MODELS[model_id]["label"],
    }


@router.get("/api/embedding-models")
async def list_embedding_models():
    result = []
    for mid, cfg in EMBEDDING_MODELS.items():
        try:
            count = state.vector_collections[mid].count() if mid in state.vector_collections else 0
        except Exception:
            count = 0
        result.append({
            "id":             mid,
            **cfg,
            "embedded_count": count,
            "active":         mid == state.current_embedding_model,
            "available":      embedding_model_available(),
        })
    return result
