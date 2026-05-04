# Suno Discord Analysis — Retrieval & Analysis Platform

A web-based research tool for ingesting, searching, and analysing exported Discord conversation data using keyword search, vector-similarity retrieval, and LLM-powered summarisation. Built for academic research into the Suno AI community.

For the detailed technical implementation of each feature, see [TECHNICAL_FLOW.md](TECHNICAL_FLOW.md).

---

## Table of Contents

- [System Overview](#system-overview)
- [Architecture](#architecture)
- [Setup](#setup)
- [Configuration Reference](#configuration-reference)
- [Features](#features)
  - [Data Upload & Embedding](#1-data-upload--embedding)
  - [Search](#2-search)
  - [Semantic Filter](#3-semantic-filter-in-results)
  - [Context Window](#4-context-window)
  - [RAG Chat](#5-rag-chat)
  - [Hybrid Summary](#6-hybrid-summary)
  - [Summarize Results](#7-summarize-results)
  - [User Profile Analysis](#8-user-profile-analysis)
  - [Bookmarks & Labels](#9-bookmarks--labels)
  - [Suno Team Management](#10-suno-team-management)
  - [Stats](#11-stats)
- [API Reference](#api-reference)
- [Deployment](#deployment)

---

## System Overview

The platform operates as a FastAPI application backed by two storage layers:

| Layer | Technology | Purpose |
|---|---|---|
| Relational store | SQLite (WAL mode) + FTS5 | Structured message storage, keyword search, bookmarks, labels |
| Vector store | Qdrant or ChromaDB | Dense-vector similarity search and embedding storage |

Messages are ingested from CSV exports, stored in SQLite with a full-text index, and optionally embedded with OpenAI `text-embedding-3-small` (1536 dimensions, cosine distance) for semantic retrieval. All LLM calls use the OpenAI chat completions API and are streamed back to the browser as Server-Sent Events.

---

## Architecture

```
Browser (HTML/JS)
      │
      │  HTTP / SSE
      ▼
FastAPI (app.py)
  ├── _AuthMiddleware         Bearer-token guard on /api/*
  ├── _SecurityHeadersMiddleware
  │
  ├── routers/config_api.py   API key & embedding model management
  ├── routers/stats.py        /api/stats
  ├── routers/uploads.py      CSV ingest, background embed jobs
  ├── routers/search.py       Username / keyword / range / semantic search
  ├── routers/chat.py         RAG chat, summarisation, user profiling
  ├── routers/context.py      Context window, in-results semantic filter
  ├── routers/bookmarks.py    Bookmark CRUD + label assignment
  ├── routers/labels.py       Label CRUD
  └── routers/suno_team.py    Suno-team member management
      │
      ├── SQLite (discord_data.db)
      │     messages, uploads, bookmarks, labels, settings,
      │     embedded_uploads, messages_fts (FTS5 virtual table)
      │
      └── Vector Store (Qdrant / ChromaDB)
            Collection: discord_openai  (1536-dim, cosine)
```

**Shared runtime state** (`state.py`) holds the OpenAI client references, the vector collection wrappers, the background embed-job registry, and short-lived response caches (30-second TTL for `/api/stats` and `/api/uploads`).

A dedicated `ThreadPoolExecutor` (`vector_executor`, 4 workers) handles all blocking vector-store calls so they never block the asyncio event loop.

---

## Setup

### Prerequisites

- Python 3.11+
- An OpenAI API key (for embeddings and chat)
- A running Qdrant server *or* ChromaDB (persistent or HTTP)

### Install

```bash
pip install -r requirements.txt
```

### Environment

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `VECTOR_DB` | `qdrant` | Backend: `qdrant`, `chroma_persistent`, or `chroma_http` |
| `QDRANT_URL` | — | Qdrant server URL (required when `VECTOR_DB=qdrant`) |
| `QDRANT_API_KEY` | — | Optional Qdrant auth token |
| `CHROMA_PATH` | `./chroma_db` | Local path for persistent ChromaDB |
| `OPENAI_API_KEY` | — | Pre-load at startup (recommended for production) |
| `API_SECRET` | — | Bearer token protecting all `/api/*` routes |
| `DB_PATH` | `discord_data.db` | SQLite file path |
| `MAX_UPLOAD_MB` | `50` | CSV upload size limit |
| `EMBED_BATCH_SIZE` | `2048` | Messages per OpenAI embedding request |
| `EMBED_CONCURRENCY` | `10` | Concurrent embedding API calls |

### Run

```bash
# Development
uvicorn app:app --reload --port 8000

# Production
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
```

---

## Configuration Reference

### CSV Format

The upload endpoint expects a CSV with at minimum these columns (header names are normalised to lowercase with underscores):

| Column | Required | Description |
|---|---|---|
| `author_id` | Yes | Discord user ID |
| `username` | Yes | Display name |
| `date` | Yes | ISO-8601 timestamp or date string |
| `content` | Yes | Message text |
| `attachments` | No | Attachment URLs or metadata |
| `reactions` | No | Reaction counts |
| `is_suno_team` | No | `"true"` / `"1"` flags Suno team members |
| `week` | No | ISO week label |
| `month` | No | Month label |

### Chat Model Allowlist

Only the following OpenAI model IDs are accepted at the `/api/chat` and summarisation endpoints (configured in `config.py`):

- GPT-5.x series: `gpt-5.4`, `gpt-5.4-pro`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`
- GPT-4.1 series: `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`
- GPT-4o series: `gpt-4o`, `gpt-4o-mini`
- GPT-4 legacy: `gpt-4-turbo`, `gpt-4`
- GPT-3.5: `gpt-3.5-turbo`
- o-series: `o4-mini`, `o3`, `o3-mini`, `o1`, `o1-mini`, `o1-preview`

---

## Features

### 1. Data Upload & Embedding

**Endpoint:** `POST /api/upload`

A CSV file is uploaded and processed in a single streaming request. The server responds with Server-Sent Events so the browser can display live progress.

**Processing steps:**

1. Column headers are normalised (stripped, lowercased, spaces replaced with underscores).
2. Required columns (`author_id`, `username`, `date`, `content`) are validated.
3. Optional columns are added as empty strings if absent.
4. Each row is assigned a random UUID (`msg_uuid`) and inserted into SQLite with `INSERT OR IGNORE` to prevent duplicates.
5. The FTS5 trigger `tg_messages_ai` automatically indexes the `content` column on each insert.
6. Messages with non-empty `content` are embedded concurrently using the OpenAI API (`text-embedding-3-small`) in batches of up to `EMBED_BATCH_SIZE` (default 2048), with up to `EMBED_CONCURRENCY` (default 10) batches in flight simultaneously.
7. Vectors are upserted into the active vector-store collection.
8. A record is written to `embedded_uploads` to track which uploads have been embedded per model.

**Re-embedding:** `POST /api/uploads/{upload_id}/reembed` starts an asynchronous background job. The job checks which UUIDs are already in the vector store (resumability check, 4-way concurrent in batches of 500) and skips those, then embeds only the remainder. Progress is polled via `GET /api/jobs/{job_id}`.

**Deletion options:**
- `DELETE /api/uploads/{upload_id}` — removes from both SQLite and all vector-store collections.
- `DELETE /api/uploads/{upload_id}/sqlite` — removes from SQLite only; vectors are preserved.
- `DELETE /api/uploads/{upload_id}/embeddings` — removes vectors only; SQLite rows are preserved.

---

### 2. Search

All search endpoints accept these common filter parameters:

| Parameter | Type | Description |
|---|---|---|
| `upload_ids` | `string` (CSV) | Restrict to specific upload IDs |
| `date_from` | `YYYY-MM-DD` | Earliest message date |
| `date_to` | `YYYY-MM-DD` | Latest message date |
| `suno_team` | `all` / `only` / `exclude` | Suno team membership filter |
| `min_words` | `int` | Minimum word count per message |
| `limit` | `int` | Maximum results returned |

#### Username Search — `GET /api/search/username`

Case-insensitive `LIKE` query on the `username` column. Supports partial matching (substring). Results are ordered by `(date, row_index)`.

#### Keyword Search — `GET /api/search/keyword`

Uses the SQLite FTS5 virtual table (`messages_fts`) for fast full-text search. The query string is sanitised and compiled into an FTS5 MATCH expression:

- Single word → prefix match: `"word"*`
- Multiple words → phrase match: `"word1 word2"`

FTS5 candidates (up to `limit × 20`) are retrieved by `rowid`, then the full message rows are fetched with `IN (...)` and all additional filters are applied in SQL. If the FTS index is unavailable or returns an error, the search falls back transparently to a `LOWER(content) LIKE LOWER(?)` scan.

#### Date Range Search — `GET /api/search/range`

Returns all messages within the specified date range and other filters, with no keyword constraint. Useful for time-sliced corpus extraction.

#### Semantic Search — `GET /api/search/semantic`

1. The query string is embedded with OpenAI `text-embedding-3-small`.
2. The vector store is queried for the nearest `n_results` neighbours (cosine similarity, ascending distance = descending similarity).
3. When filters are active, `n_results × 4` candidates are fetched to allow post-filtering without missing relevant results.
4. Matching UUIDs are looked up in SQLite with a single `IN (...)` query.
5. Metadata filters (upload, username, date range, Suno team, min words) are applied in Python.
6. Each result carries a `similarity_score` (1 − cosine distance, rounded to 4 decimal places).

#### Users in Range — `GET /api/search/users-in-range`

Aggregates per-user statistics for the filtered message set:

- `total_messages` — message count
- `first_message_date` / `last_message_date`
- `avg_word_count` — average word count (space-count+1 heuristic)
- `weeks_with_messages` — distinct ISO weeks with at least one message
- `pct_weeks_active` — percentage of weeks in the date range with activity (requires both `date_from` and `date_to`)

#### User Messages — `GET /api/search/user-messages`

Exact-match username query (case-insensitive) with optional keyword sub-filter. Returns the full chronological message list for a single user.

#### Bulk Context — `POST /api/search/bulk-context`

Fetches conversation context windows for multiple messages in a single round-trip. For each `msg_id`, returns the target message plus `before` rows before it and `after` rows after it (both capped at 50), using `(upload_id, row_index)` to navigate the original CSV order. The target row has `is_target: true`.

---

### 3. Semantic Filter (In-Results)

**Endpoint:** `POST /api/filter/semantic`

Filters an already-retrieved set of message IDs by semantic relevance to a query without hitting the vector store's ANN index. Instead, it fetches the stored embedding vectors for the specified IDs directly and computes cosine similarity in-process using NumPy.

**Query preprocessing:**

- If the query starts with a question word (`what`, `how`, `why`, etc.) or ends with `?`, stop words and question words are stripped to isolate the semantic core. The similarity threshold is lowered to 0.20.
- Otherwise the query is embedded as-is with a threshold of 0.30.

This distinction prevents question-phrasing from diluting the embedding with low-information tokens.

---

### 4. Context Window

**Endpoint:** `GET /api/context/{message_id}`

Returns up to `before` (default 5, max 200) messages before the target and `after` (default 5, max 200) messages after it, ordered by `row_index` within the same upload. The target row has `is_target: true`. This preserves original chronological order within the CSV without relying on timestamp ordering (which may have gaps or ties).

---

### 5. RAG Chat

**Endpoint:** `POST /api/chat`

Implements Retrieval-Augmented Generation over the message corpus. The response streams as Server-Sent Events.

**Retrieval pipeline:**

1. The user's message is embedded with `text-embedding-3-small`.
2. A semantic search retrieves 12 nearest-neighbour messages from the vector store, filtered by `upload_ids` if specified.
3. A keyword search (`keyword_search()`) retrieves up to 10 additional messages using FTS5.
4. Results from both paths are merged, deduplicated by `msg_uuid`, and capped at 20 messages.

**Prompt construction:**

- The 20 retrieved messages are formatted as `[username | date] content` lines.
- A structured system prompt instructs the model to cite usernames, use Markdown headings, bold key terms, use blockquotes for direct quotes, and close with a sources section.
- If no messages are embedded, the system prompt falls back to a general-knowledge instruction.
- Up to 20 prior conversation turns are included in the messages array.

**o-series model handling:** Models whose ID begins with `o` use `"developer"` as the system role (required by the OpenAI reasoning models API).

---

### 6. Hybrid Summary

**Endpoint:** `POST /api/summarize` (initial) + `POST /api/summarize/followup` (follow-up Q&A)

A multi-stage pipeline that selects a statistically representative evidence set from potentially thousands of messages and summarises it with an LLM.

**Pipeline stages (cluster mode):**

```
Metadata filter
      │  filters by username, date range, uploads, Suno team, min_words
      ▼
Semantic retrieval
      │  query = custom prompt OR generic coverage query
      │  overfetch = min(total_in_store, max(n_filtered × 5, 2000))
      │  intersect with filtered set
      │  adaptive threshold: keep top 70% by score (≥ _MIN_CANDIDATES = 15)
      ▼
Deduplication
      │  cosine similarity matrix on candidate embeddings
      │  drop messages with similarity ≥ 0.97 (keep first in ranking order)
      │  also drop messages < 10 chars
      ▼
Clustering
      │  priority: HDBSCAN → OPTICS (sklearn) → KMeans (sklearn) → KMeans (NumPy)
      │  noise points (-1) are promoted to singleton clusters
      ▼
Per-cluster sampling
      │  for each cluster: 5 messages closest to centroid + 5 furthest
      ▼
Assemble & sort
      │  chronological sort, capped at 120 messages
      ▼
LLM generation (streaming SSE)
```

**Fallback behaviour:**
- If semantic retrieval returns fewer than 10 candidates, the pipeline fetches stored embeddings for the full filtered set (up to 3000 random samples if larger) and runs dedup/cluster/sample on those.
- If no embeddings exist at all, all filtered messages are sent directly to the LLM.
- If `retrieval_mode = "all"`, the clustering pipeline is skipped and all filtered messages are sent.

**Transparency log:** Before the LLM tokens stream, the server emits JSON log events for each pipeline step (filter count, retrieval count, duplicates removed, algorithm used, cluster count, evidence count). These appear in the browser UI as a visible "research pipeline" trace.

**Follow-up Q&A:** The follow-up endpoint re-runs the same retrieval/dedup/cluster pipeline with the *follow-up question* as the retrieval query (not the original prompt). The initial summary is embedded in the system prompt as authoritative context, and the prior Q&A turns are appended to the message list.

---

### 7. Summarize Results

**Endpoint:** `POST /api/summarize-results` (initial) + `POST /api/summarize-results/followup`

Summarises the set of messages the browser currently displays (passed directly in the request body) rather than re-querying the database. This allows summarising any arbitrary selection of search results.

**Token safety net:** The payload character count is estimated (1 token ≈ 4 chars). If the estimated token count exceeds ~90,000 tokens, the pipeline auto-switches to cluster+sample regardless of the requested `retrieval_mode`. If clustering still produces an oversized payload, a hard-truncation loop progressively removes 20% of the message set until it fits within the limit.

**Follow-up:** The follow-up variant is stateless — all context lives in the `history` array sent by the browser; no database or vector-store calls are made.

---

### 8. User Profile Analysis

**Endpoint:** `POST /api/user-profile` (initial) + `POST /api/user-profile/followup`

Analyses all messages from a specific user to produce a structured persona profile. Uses the same semantic retrieval → dedup → cluster → sample pipeline as the Hybrid Summary, but the filter is an exact-match on `username` and the LLM prompt requests:

- Entry date (first message) and exit date (last message)
- Persona description (role, communication style)
- Evolution of attitude over time (chronological narrative with inflection points)
- Key recurring topics and concerns
- Representative verbatim quotes
- Summary assessment

The entry/exit dates are extracted from the first and last rows of the filtered query result and injected into the prompt template before clustering.

---

### 9. Bookmarks & Labels

Bookmarks allow any message to be saved for later review. Each bookmark stores:

- `msg_id` — the saved message
- `ctx_before` / `ctx_after` — how many context rows to show around it
- `note` — free-text annotation
- `created_at` — UTC timestamp

Labels are independent coloured tags (name + hex colour) that can be assigned to bookmarks via a many-to-many join table (`bookmark_labels`). The `GET /api/bookmarks` endpoint returns bookmarks with their labels joined in a single query.

The `GET /api/bookmarks/ids` endpoint returns only the bookmarked `msg_id` list — used by the UI to cheaply mark which messages are already bookmarked without fetching full rows.

---

### 10. Suno Team Management

The `is_suno_team` field in each message row flags Suno AI staff messages. All search and summarisation endpoints accept a `suno_team` parameter:

| Value | Behaviour |
|---|---|
| `all` (default) | No filter |
| `only` | Restrict to Suno team messages |
| `exclude` | Exclude Suno team messages |

`GET /api/suno-team` returns all usernames currently flagged as Suno team members with their message counts. `DELETE /api/suno-team/{username}` sets `is_suno_team = 'false'` for all messages by that username, effectively demoting them from the team roster.

---

### 11. Stats

**Endpoint:** `GET /api/stats`

Returns aggregate counts fetched in parallel:

- `total_messages` — row count in the `messages` table
- `total_uploads` — row count in the `uploads` table
- `embedded_messages` — vector count in the active collection
- `api_key_set` — whether an OpenAI client is configured
- `current_model` / `current_model_label` — active embedding model

Responses are cached in memory for 30 seconds to avoid repeated DB and vector-store hits from the dashboard polling.

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/set-api-key` | Set OpenAI API key for this session |
| `POST` | `/api/set-embedding-model` | Switch active embedding model |
| `GET` | `/api/embedding-models` | List models with embedded vector counts |
| `GET` | `/api/stats` | Aggregate stats (cached 30 s) |
| `GET` | `/api/uploads` | List all uploads with embedding status (cached 30 s) |
| `POST` | `/api/upload` | Upload CSV; streams SSE progress |
| `POST` | `/api/uploads/{id}/reembed` | Start background re-embed job |
| `GET` | `/api/jobs/{job_id}` | Poll embed job progress |
| `DELETE` | `/api/uploads/{id}` | Delete upload from SQLite + vector store |
| `DELETE` | `/api/uploads/{id}/sqlite` | Delete from SQLite only |
| `DELETE` | `/api/uploads/{id}/embeddings` | Delete vectors only |
| `GET` | `/api/search/username` | Search by username |
| `GET` | `/api/search/keyword` | FTS5 keyword search |
| `GET` | `/api/search/range` | Date range fetch |
| `GET` | `/api/search/semantic` | Vector similarity search |
| `GET` | `/api/search/users-in-range` | Per-user activity stats |
| `GET` | `/api/search/user-messages` | All messages by one user |
| `POST` | `/api/search/bulk-context` | Context windows for multiple messages |
| `GET` | `/api/context/{id}` | Context window for one message |
| `POST` | `/api/filter/semantic` | Semantic filter on a result set |
| `POST` | `/api/chat` | RAG chat (streaming SSE) |
| `POST` | `/api/summarize` | Hybrid summary (streaming SSE) |
| `POST` | `/api/summarize/followup` | Hybrid summary follow-up Q&A |
| `POST` | `/api/summarize-results` | Summarise browser result set (streaming SSE) |
| `POST` | `/api/summarize-results/followup` | Summarise results follow-up Q&A |
| `POST` | `/api/user-profile` | User persona profile (streaming SSE) |
| `POST` | `/api/user-profile/followup` | User profile follow-up Q&A |
| `POST` | `/api/bookmarks` | Create bookmark |
| `GET` | `/api/bookmarks` | List bookmarks with labels |
| `GET` | `/api/bookmarks/ids` | List bookmarked message IDs |
| `DELETE` | `/api/bookmarks/{id}` | Delete bookmark |
| `DELETE` | `/api/bookmarks/by-msg/{msg_id}` | Delete bookmark by message ID |
| `POST` | `/api/bookmarks/{id}/labels/{label_id}` | Assign label to bookmark |
| `DELETE` | `/api/bookmarks/{id}/labels/{label_id}` | Remove label from bookmark |
| `POST` | `/api/labels` | Create label |
| `GET` | `/api/labels` | List labels |
| `PUT` | `/api/labels/{id}` | Update label |
| `DELETE` | `/api/labels/{id}` | Delete label |
| `GET` | `/api/suno-team` | List Suno team members |
| `DELETE` | `/api/suno-team/{username}` | Remove Suno team flag |

---

## Deployment

The repository includes a GitHub Actions workflow (`.github/workflows/deploy.yml`) that deploys to a VPS on every push to `main`:

1. SSH into the VPS using a secret key.
2. `git pull origin main` in the app directory.
3. `docker compose up -d --build` rebuilds and restarts the container.
4. `docker image prune -f` removes dangling images.

The workflow uses `concurrency: group: production-deploy` with `cancel-in-progress: true` to prevent overlapping deployments from concurrent pushes.
