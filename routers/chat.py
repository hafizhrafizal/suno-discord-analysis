"""
routers/chat.py — RAG chat and conversation summarisation.

  POST /api/chat
  POST /api/summarize
"""

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

import state
from config import VALID_CHAT_MODELS
from database import get_db
from embeddings import active_collection, embed_texts_async
from sql_helpers import (
    _parse_upload_ids,
    _sql_upload_ids_clause,
    _suno_sql,
    keyword_search,
    sql_date_clauses,
    sql_min_words_clause,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/chat")
async def chat_endpoint(request: Request):
    body       = await request.json()
    message    = (body.get("message") or "").strip()
    history    = body.get("history") or []
    upload_ids = body.get("upload_ids") or ""
    chat_model = (body.get("model") or "gpt-5.4").strip()

    if not message:
        raise HTTPException(400, "Empty message.")
    if not state.openai_client:
        raise HTTPException(400, "OpenAI API key not set — add it in Settings.")
    if chat_model not in VALID_CHAT_MODELS:
        raise HTTPException(400, f"Unknown model '{chat_model}'.")

    uid_list = _parse_upload_ids(upload_ids)

    _chat_query_emb = None
    try:
        _chat_query_emb = (await embed_texts_async([message]))[0]
    except Exception:
        pass

    def _semantic_search() -> list[dict]:
        rows: list[dict] = []
        try:
            col = active_collection()
            if col.count() == 0 or _chat_query_emb is None:
                return rows
            results   = col.query(query_embeddings=[_chat_query_emb], n_results=12)
            ids       = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0]
            uuid_map: dict[str, dict] = {}
            if ids:
                conn = get_db()
                try:
                    ph      = ",".join("?" * len(ids))
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
                try:
                    score = round(1.0 - float(dist), 4) if dist is not None else None
                except (TypeError, ValueError):
                    score = None
                row["_score"] = score
                rows.append(row)
        except Exception as exc:
            logger.warning("/api/chat semantic retrieval error: %s", exc)
        return rows

    import asyncio
    loop = asyncio.get_running_loop()
    semantic_rows, keyword_rows = await asyncio.gather(
        loop.run_in_executor(None, _semantic_search),
        keyword_search(keyword=message, upload_ids=upload_ids, limit=10),
    )

    context_rows    = list(semantic_rows)
    existing_uuids  = {r["msg_uuid"] for r in context_rows}
    for r in keyword_rows:
        if r["msg_uuid"] not in existing_uuids:
            context_rows.append(r)
            existing_uuids.add(r["msg_uuid"])
    context_rows = context_rows[:20]

    if context_rows:
        ctx_text = "\n".join(
            f"[{r['username']} | {r['date']}] {r['content']}"
            for r in context_rows
        )
        system = (
            "You are a knowledgeable assistant for the Suno AI Discord community.\n\n"
            "INSTRUCTIONS:\n"
            "- Use the retrieved conversation excerpts below as your PRIMARY source of truth.\n"
            "- Cite specific usernames (e.g. **@username**) when referencing their messages.\n"
            "- If the context does not cover the question, say so clearly before answering from general knowledge.\n\n"
            "MANDATORY FORMATTING — your entire response MUST be valid Markdown:\n"
            "- Start with a `##` heading that summarises the answer topic.\n"
            "- Use `###` subheadings to separate distinct sub-topics.\n"
            "- Use **bold** for key terms, usernames, and important points.\n"
            "- Use `-` bullet lists for multiple items or steps; use `1.` numbered lists for sequences.\n"
            "- Use `> blockquote` to highlight a direct or paraphrased user quote.\n"
            "- Use `inline code` for technical terms, settings, or commands.\n"
            "- End with a `---` rule followed by a brief *Sources* section listing cited usernames and dates.\n"
            "- Do NOT output plain prose paragraphs without any formatting.\n\n"
            "RETRIEVED CONTEXT:\n" + ctx_text
        )
    else:
        system = (
            "You are a helpful assistant for the Suno AI Discord community.\n"
            "No embedded messages are available — answer from general knowledge.\n\n"
            "MANDATORY FORMATTING — your entire response MUST be valid Markdown:\n"
            "- Start with a `##` heading.\n"
            "- Use **bold**, `-` bullet lists, `###` subheadings, and `inline code` where appropriate.\n"
            "- Do NOT output plain prose without any Markdown structure.\n"
        )

    is_o_model = chat_model.startswith("o")
    sys_role   = "developer" if is_o_model else "system"
    msgs       = [{"role": sys_role, "content": system}]
    for turn in history[-20:]:
        role    = turn.get("role")
        content = turn.get("content")
        if role in {"system", "user", "assistant"} and content:
            msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": message})

    async def generate():
        try:
            stream = state.openai_client.chat.completions.create(
                model=chat_model, messages=msgs, stream=True
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'content': delta})}\n\n"
        except Exception as exc:
            logger.error("chat generate() error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


@router.post("/api/summarize")
async def summarize_endpoint(request: Request):
    body       = await request.json()
    username   = (body.get("username") or "").strip()
    date_from  = (body.get("date_from") or "").strip()
    date_to    = (body.get("date_to") or "").strip()
    prompt_txt = (body.get("prompt") or "").strip()
    upload_ids = (body.get("upload_ids") or "")
    min_words  = int(body.get("min_words") or 0)
    suno_team  = str(body.get("suno_team") or "all")
    sum_model  = (body.get("model") or "gpt-5.4").strip()

    if not state.openai_client:
        raise HTTPException(400, "OpenAI API key not set — add it in Settings.")
    if sum_model not in VALID_CHAT_MODELS:
        raise HTTPException(400, f"Unknown model '{sum_model}'.")

    uid_list              = _parse_upload_ids(upload_ids)
    uid_sql, uid_params   = _sql_upload_ids_clause(uid_list)
    date_sql, date_params = sql_date_clauses(date_from or None, date_to or None)
    words_sql, words_params = sql_min_words_clause(min_words)

    params: list = []
    sql = "SELECT username, date, content FROM messages WHERE 1=1"
    if username:
        sql += " AND LOWER(username) LIKE LOWER(?)"
        params.append(f"%{username}%")
    sql += uid_sql
    params.extend(uid_params)
    sql += _suno_sql(suno_team) + date_sql + words_sql
    params.extend(date_params + words_params)
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
Do NOT output plain paragraphs. Every section must use the Markdown elements above."""

    user_prompt = prompt_txt or default_prompt
    full        = f"{user_prompt}\n\nCONVERSATION ({len(rows)} messages):\n{conv}"

    is_o_model = sum_model.startswith("o")
    sys_role   = "developer" if is_o_model else "system"

    async def generate():
        try:
            stream = state.openai_client.chat.completions.create(
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
        except Exception as exc:
            logger.error("summarize generate() error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )
