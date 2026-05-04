"""
routers/uploads.py — CSV upload, embed jobs, and upload management.

  GET    /api/uploads
  POST   /api/upload
  POST   /api/uploads/{upload_id}/reembed
  GET    /api/jobs/{job_id}
  DELETE /api/uploads/{upload_id}
  DELETE /api/uploads/{upload_id}/sqlite
  DELETE /api/uploads/{upload_id}/embeddings
"""

import asyncio
import io
import logging
import uuid
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

import state
from config import EMBED_BATCH_SIZE, EMBED_CONCURRENCY, EMBEDDING_MODELS, MAX_UPLOAD_BYTES, MAX_UPLOAD_MB
from database import (
    get_all_embedded_uploads,
    get_db,
    mark_upload_embedded,
    safe_str,
    unmark_upload_embedded,
)
from embeddings import (
    active_collection,
    embed_texts_async,
    embedding_model_available,
    run_embed_job,
)
from routers.deps import require_admin

logger = logging.getLogger(__name__)
router = APIRouter()


# ── List uploads ──────────────────────────────────────────────────────────────

@router.get("/api/uploads")
async def list_uploads():
    import time as _time
    cached, ts = state.get_uploads_cache()
    if cached is not None and (_time.monotonic() - ts) < state.STATS_TTL:
        return cached

    loop = asyncio.get_running_loop()

    def _fetch():
        conn  = get_db()
        rows  = conn.execute("SELECT * FROM uploads ORDER BY upload_time DESC").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    uploads = await loop.run_in_executor(None, _fetch)
    if not uploads:
        state.set_uploads_cache(uploads)
        return uploads

    embedded_state = await loop.run_in_executor(None, get_all_embedded_uploads)
    mid_list = list(EMBEDDING_MODELS)
    for u in uploads:
        u["embedded_models"] = {
            mid: (u["id"] in embedded_state.get(mid, set()))
            for mid in mid_list
        }

    state.set_uploads_cache(uploads)
    return uploads


# ── CSV upload + streaming embed ──────────────────────────────────────────────

@router.post("/api/upload")
async def upload_csv(request: Request, file: UploadFile = File(...), _: dict = Depends(require_admin)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only .csv files are supported")

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB.")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB.")

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"Could not parse CSV: {exc}")

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    required = {"author_id", "username", "date", "content"}
    missing  = required - set(df.columns)
    if missing:
        raise HTTPException(400, f"CSV is missing required columns: {missing}")

    for col in ["attachments", "reactions", "is_suno_team", "week", "month"]:
        if col not in df.columns:
            df[col] = ""

    df = df.reset_index(drop=True)
    df["content"] = df["content"].fillna("").astype(str)

    if not embedding_model_available() and df["content"].astype(str).str.strip().any():
        raise HTTPException(
            400,
            "OpenAI API key is required to embed messages. Set it in Settings before uploading.",
        )

    upload_id = str(uuid.uuid4())
    now       = datetime.now().isoformat()

    async def generate():
        yield f"data: Processing {len(df)} rows from {file.filename}\n\n"

        conn = get_db()
        conn.execute("INSERT INTO uploads VALUES (?,?,?,?)",
                     (upload_id, file.filename, len(df), now))

        rows_inserted   = 0
        texts_to_embed: list[str]  = []
        uuids_to_embed: list[str]  = []
        metas_to_embed: list[dict] = []

        for row_idx, row in df.iterrows():
            msg_uuid = str(uuid.uuid4())
            content  = row["content"].strip()
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
                    "username":  safe_str(row.get("username")),
                    "date":      safe_str(row.get("date")),
                    "upload_id": upload_id,
                })

        conn.commit()
        conn.execute("ANALYZE messages")
        conn.commit()
        conn.close()

        yield f"data: Inserted {rows_inserted} messages into database\n\n"

        embedded_count = 0
        if embedding_model_available() and texts_to_embed:
            yield (
                f"data: Starting embedding {len(texts_to_embed)} messages "
                f"with {EMBEDDING_MODELS[state.current_embedding_model]['label']}\n\n"
            )
            col  = active_collection()
            loop = asyncio.get_running_loop()
            sem  = asyncio.Semaphore(EMBED_CONCURRENCY)

            async def _upload_embed_batch(batch_num: int, b_texts, b_uuids, b_metas):
                nonlocal embedded_count
                async with sem:
                    try:
                        embeddings = await embed_texts_async(b_texts)
                    except Exception as exc:
                        logger.warning("Embedding batch %d failed: %s", batch_num, exc)
                        return
                try:
                    _e, _bt, _bu, _bm = embeddings, b_texts, b_uuids, b_metas
                    await loop.run_in_executor(
                        state.vector_executor,
                        lambda: col.upsert(embeddings=_e, documents=_bt, ids=_bu, metadatas=_bm),
                    )
                    embedded_count += len(b_texts)
                except Exception as exc:
                    logger.warning("Vector store write batch %d failed: %s", batch_num, exc)

            all_batches = [
                _upload_embed_batch(
                    batch_num,
                    texts_to_embed[i : i + EMBED_BATCH_SIZE],
                    uuids_to_embed[i : i + EMBED_BATCH_SIZE],
                    metas_to_embed[i : i + EMBED_BATCH_SIZE],
                )
                for batch_num, i in enumerate(
                    range(0, len(texts_to_embed), EMBED_BATCH_SIZE), start=1
                )
            ]

            for chunk_start in range(0, len(all_batches), EMBED_CONCURRENCY):
                chunk = all_batches[chunk_start : chunk_start + EMBED_CONCURRENCY]
                await asyncio.gather(*chunk)
                yield f"data: Embedded {embedded_count}/{len(texts_to_embed)} messages\n\n"

            if embedded_count > 0:
                mark_upload_embedded(upload_id, state.current_embedding_model)
        elif not embedding_model_available():
            yield "data: Skipping embedding (no OpenAI API key configured)\n\n"
        else:
            yield "data: No messages to embed\n\n"

        state.invalidate_all_caches()
        yield f"data: Completed: inserted {rows_inserted}, embedded {embedded_count}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# ── Re-embed ──────────────────────────────────────────────────────────────────

@router.post("/api/uploads/{upload_id}/reembed")
async def reembed_upload(upload_id: str, _: dict = Depends(require_admin)):
    """
    Start a background embed job for an existing upload.
    Returns immediately with a job_id.  Poll GET /api/jobs/{job_id} for progress.
    """
    existing_jid = state.active_embed.get(upload_id)
    if existing_jid and state.embed_jobs.get(existing_jid, {}).get("status") == "running":
        return {"job_id": existing_jid, "already_running": True}

    conn   = get_db()
    upload = conn.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    if not upload:
        conn.close()
        raise HTTPException(404, "Upload not found")

    rows = conn.execute(
        "SELECT msg_uuid, content, username, date FROM messages WHERE upload_id = ? AND content != ''",
        (upload_id,),
    ).fetchall()
    conn.close()

    if not embedding_model_available():
        raise HTTPException(400, "OpenAI API key is required to re-embed. Set it in Settings first.")

    texts  = [r["content"]  for r in rows]
    uuids  = [r["msg_uuid"] for r in rows]
    metas  = [{"username": r["username"], "date": r["date"], "upload_id": upload_id} for r in rows]
    col    = active_collection()
    job_id = str(uuid.uuid4())

    state.embed_jobs[job_id] = {
        "status":        "running",
        "phase":         "checking",
        "upload_id":     upload_id,
        "model":         EMBEDDING_MODELS[state.current_embedding_model]["label"],
        "embedded":      0,
        "total":         len(texts),
        "skipped":       0,
        "current_batch": 0,
        "batch_errors":  [],
        "error":         None,
        "traceback":     None,
    }
    state.active_embed[upload_id] = job_id

    asyncio.create_task(run_embed_job(job_id, upload_id, col, texts, uuids, metas))

    return {"job_id": job_id, "already_running": False, "total_messages": len(texts)}


# ── Job progress ──────────────────────────────────────────────────────────────

@router.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Poll embed job progress."""
    job = state.embed_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


# ── Delete endpoints ──────────────────────────────────────────────────────────

@router.delete("/api/uploads/{upload_id}")
async def delete_upload(upload_id: str, _: dict = Depends(require_admin)):
    """Delete an upload and all its messages from SQLite and every vector-store collection."""
    conn   = get_db()
    upload = conn.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    if not upload:
        conn.close()
        raise HTTPException(404, "Upload not found")

    uuid_rows  = conn.execute(
        "SELECT msg_uuid FROM messages WHERE upload_id = ?", (upload_id,)
    ).fetchall()
    msg_uuids = [r["msg_uuid"] for r in uuid_rows]

    if msg_uuids:
        for col in state.vector_collections.values():
            try:
                batch_size = 500
                for i in range(0, len(msg_uuids), batch_size):
                    batch_uuids = msg_uuids[i : i + batch_size]
                    existing    = col.get(ids=batch_uuids, include=[])
                    if existing["ids"]:
                        col.delete(ids=existing["ids"])
            except Exception as exc:
                logger.error("Vector store delete failed for upload %s: %s", upload_id, exc)
                raise HTTPException(500, f"Failed to delete vectors: {exc}")

    conn.execute("DELETE FROM messages WHERE upload_id = ?", (upload_id,))
    conn.execute("DELETE FROM uploads WHERE id = ?", (upload_id,))
    conn.commit()
    conn.close()
    state.invalidate_all_caches()
    return {"status": "ok", "deleted_messages": len(msg_uuids)}


@router.delete("/api/uploads/{upload_id}/sqlite")
async def delete_upload_sqlite(upload_id: str, _: dict = Depends(require_admin)):
    """Delete an upload and its messages from SQLite only — embeddings are preserved."""
    conn   = get_db()
    upload = conn.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
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
    state.invalidate_all_caches()
    return {"status": "ok", "deleted_messages": msg_count}


@router.delete("/api/uploads/{upload_id}/embeddings")
async def delete_upload_embeddings(upload_id: str, _: dict = Depends(require_admin)):
    """Delete embeddings for an upload from every vector-store collection — SQLite untouched."""
    conn   = get_db()
    upload = conn.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    if not upload:
        conn.close()
        raise HTTPException(404, "Upload not found")
    conn.close()

    loop          = asyncio.get_running_loop()
    deleted_count = 0

    for col in state.vector_collections.values():
        try:
            def _delete_by_upload():
                before = col.get(
                    where={"upload_id": {"$eq": upload_id}},
                    include=[],
                ).get("ids", [])
                if before:
                    col.delete(where={"upload_id": {"$eq": upload_id}})
                return len(before)

            n = await loop.run_in_executor(None, _delete_by_upload)
            deleted_count += n
        except Exception as exc:
            logger.error("Vector store delete embeddings failed for upload %s: %s", upload_id, exc)
            raise HTTPException(500, f"Failed to delete vectors: {exc}")

    unmark_upload_embedded(upload_id)
    state.invalidate_all_caches()
    return {"status": "ok", "deleted_embeddings": deleted_count}
