"""
routers/context.py — message context window and in-results semantic filter.

  GET  /api/context/{message_id}
  POST /api/filter/semantic
"""

import logging
import re

import numpy as np
from fastapi import APIRouter, HTTPException

import state
from database import get_db
from embeddings import active_collection, embed_texts_async
from sql_helpers import _parse_upload_ids

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Context window ────────────────────────────────────────────────────────────

@router.get("/api/context/{message_id}")
async def get_context(message_id: int, before: int = 5, after: int = 5):
    before  = max(0, min(before, 200))
    after   = max(0, min(after,  200))
    conn    = get_db()
    target  = conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()

    if not target:
        conn.close()
        raise HTTPException(404, "Message not found")

    target    = dict(target)
    upload_id = target["upload_id"]
    row_idx   = target["row_index"]

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


# ── In-results semantic filter ────────────────────────────────────────────────

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

    Question queries: strip stop/question words → lower threshold (0.30).
    Plain queries:    embed as-is → standard threshold (0.45).
    """
    is_question = bool(_QUESTION_RE.match(raw.strip())) or raw.strip().endswith("?")

    if is_question:
        tokens     = re.findall(r"\b\w+\b", raw.lower())
        core       = [t for t in tokens if t not in _FILTER_STOP and len(t) > 1]
        embed_text = " ".join(core) if core else raw
        threshold  = 0.30
    else:
        embed_text = raw
        threshold  = 0.45

    return embed_text, threshold


@router.post("/api/filter/semantic")
async def filter_semantic(body: dict):
    query   = (body.get("query") or "").strip()
    raw_ids = body.get("msg_ids") or []

    if not query:
        raise HTTPException(400, "query is required")
    if not raw_ids:
        return {"results": [], "threshold": 0.45, "query_used": query}

    try:
        msg_ids = [int(x) for x in raw_ids]
    except (TypeError, ValueError):
        raise HTTPException(400, "msg_ids must be a list of integers")

    embed_text, threshold = _prepare_filter_query(query)

    col = active_collection()
    if col.count() == 0:
        raise HTTPException(
            400,
            "No embeddings found for the active model. "
            "Upload and embed data first, then retry.",
        )

    conn         = get_db()
    placeholders = ",".join("?" * len(msg_ids))
    rows         = conn.execute(
        f"SELECT id, msg_uuid FROM messages WHERE id IN ({placeholders})",
        msg_ids,
    ).fetchall()
    conn.close()

    if not rows:
        return {"results": [], "threshold": threshold, "query_used": embed_text}

    id_to_uuid = {r["id"]: r["msg_uuid"] for r in rows}
    uuid_to_id = {v: k for k, v in id_to_uuid.items()}
    uuids      = list(id_to_uuid.values())

    query_vec = (await embed_texts_async([embed_text]))[0]

    # Fetch stored embeddings by ID for cosine comparison
    stored      = col.get(ids=uuids, include=["embeddings"])
    stored_ids  = stored.get("ids") or []
    stored_embs = stored.get("embeddings")
    if stored_embs is None:
        stored_embs = []

    if not stored_ids or len(stored_embs) == 0:
        return {
            "results":    [],
            "threshold":  threshold,
            "query_used": embed_text,
            "warning":    "No embeddings found for these messages. Re-embed the upload in Config first.",
        }

    q    = np.array(query_vec, dtype=np.float32)
    hits: list[dict] = []
    for uid, emb in zip(stored_ids, stored_embs):
        if uid not in uuid_to_id:
            continue
        sim = float(np.dot(q, np.array(emb, dtype=np.float32)))
        if sim >= threshold:
            hits.append({"id": uuid_to_id[uid], "score": round(sim, 4)})

    hits.sort(key=lambda x: x["score"], reverse=True)
    return {"results": hits, "threshold": threshold, "query_used": embed_text}
