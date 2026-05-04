"""
embeddings.py — OpenAI embedding helpers, active-collection accessor,
and the background embed-job coroutine.
"""

import asyncio
import logging
import traceback

from fastapi import HTTPException

import state
from config import EMBED_BATCH_SIZE, EMBED_CONCURRENCY, EMBEDDING_MODELS
from database import mark_upload_embedded

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def embedding_model_available() -> bool:
    return state.get_async_openai_client() is not None


def active_collection():
    """Return the vector-store wrapper for the currently selected model."""
    col = state.vector_collections.get(state.current_embedding_model)
    if col is None:
        logger.error(
            "active_collection: no collection for model '%s'. "
            "vector_collections keys: %s  VECTOR_DB=%s",
            state.current_embedding_model,
            list(state.vector_collections.keys()),
            __import__("config").VECTOR_DB,
        )
        raise HTTPException(503, "Vector store not ready — restart the server and try again.")
    return col


async def embed_texts_async(texts: list[str]) -> list[list[float]]:
    """Embed texts using the async OpenAI client — pure async I/O."""
    client = state.get_async_openai_client()
    if not client:
        raise HTTPException(400, "OpenAI API key not configured. Set it in Settings.")
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=[t[:8191] for t in texts],
    )
    return [e.embedding for e in response.data]


# ── Background embed job ──────────────────────────────────────────────────────

async def run_embed_job(
    job_id: str,
    upload_id: str,
    col,
    all_texts: list,
    all_uuids: list,
    all_metas: list,
) -> None:
    """
    Background coroutine: embed messages and record progress in state.embed_jobs.

    Speed:    AsyncOpenAI client — pure async I/O, no thread-pool overhead.
              EMBED_CONCURRENCY requests fire simultaneously via asyncio.gather.
              Batch size = EMBED_BATCH_SIZE (OpenAI max) — fewest round-trips.
    Resumable: UUID presence check run concurrently; already-done messages are
              skipped so re-runs pick up where they left off.
    Storage:  Vectors are written to the active vector-store backend.
    """
    job = state.embed_jobs[job_id]

    try:
        # ── 1. Concurrent resumability check ─────────────────────────────────
        job["phase"] = "checking"
        check_sem     = asyncio.Semaphore(4)
        check_batches = [all_uuids[i : i + 500] for i in range(0, len(all_uuids), 500)]

        async def _check_batch(batch):
            async with check_sem:
                result = await asyncio.get_running_loop().run_in_executor(
                    state.vector_executor, lambda: col.get(ids=batch, include=[])
                )
            found = set(result.get("ids", []))
            job["skipped"] += len(found)
            return found

        check_results = await asyncio.gather(*[_check_batch(b) for b in check_batches])
        already_done  = set().union(*check_results) if check_results else set()

        job["skipped"] = len(already_done)
        todo = [
            (t, u, m)
            for t, u, m in zip(all_texts, all_uuids, all_metas)
            if u not in already_done
        ]
        job["total"] = len(todo)

        if not todo:
            job["status"] = "completed"
            state.invalidate_all_caches()
            return

        job["phase"] = "embedding"

        # ── 2. Concurrent embedding ───────────────────────────────────────────
        sem = asyncio.Semaphore(EMBED_CONCURRENCY)

        async def _process_batch(batch_num: int, b_texts, b_uuids, b_metas):
            async with sem:
                try:
                    embeddings = await embed_texts_async(b_texts)
                except Exception as exc:
                    tb = traceback.format_exc()
                    logger.error("Job %s batch %d API error:\n%s", job_id, batch_num, tb)
                    job["batch_errors"].append({
                        "batch": batch_num,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": tb,
                    })
                    return

            try:
                _e, _bt, _bu, _bm = embeddings, b_texts, b_uuids, b_metas
                await asyncio.get_running_loop().run_in_executor(
                    state.vector_executor,
                    lambda: col.upsert(embeddings=_e, documents=_bt, ids=_bu, metadatas=_bm),
                )
            except Exception as exc:
                tb = traceback.format_exc()
                logger.error("Job %s batch %d upsert error:\n%s", job_id, batch_num, tb)
                job["batch_errors"].append({
                    "batch": batch_num,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": tb,
                })
                return

            job["embedded"]     += len(b_texts)
            job["current_batch"] = batch_num

        batch_coros = []
        for batch_num, i in enumerate(range(0, len(todo), EMBED_BATCH_SIZE), start=1):
            chunk = todo[i : i + EMBED_BATCH_SIZE]
            batch_coros.append(_process_batch(
                batch_num,
                [x[0] for x in chunk],
                [x[1] for x in chunk],
                [x[2] for x in chunk],
            ))

        await asyncio.gather(*batch_coros)

        job["status"] = "completed"
        mark_upload_embedded(upload_id, state.current_embedding_model)
        state.invalidate_all_caches()

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Embed job %s crashed:\n%s", job_id, tb)
        job["status"]    = "failed"
        job["error"]     = f"{type(exc).__name__}: {exc}"
        job["traceback"] = tb
    finally:
        state.active_embed.pop(upload_id, None)
