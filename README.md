# Suno Discord Analysis — Retrieval & Analysis Platform

A web-based research tool for ingesting, searching, and analysing exported Discord conversation data using keyword search, vector-similarity retrieval, and LLM-powered summarisation. Built for academic research into the Suno AI community.

Supports both **single-user** and **multi-user** deployment modes, with role-based access control, session authentication, and per-user data isolation.

For the detailed technical implementation of each feature, see [TECHNICAL_FLOW.md](TECHNICAL_FLOW.md).

---

## Table of Contents

- [System Overview](#system-overview)
- [Architecture](#architecture)
- [App Modes](#app-modes)
- [Setup](#setup)
- [Configuration Reference](#configuration-reference)
- [Features](#features)
  - [App Mode & Onboarding](#1-app-mode--onboarding)
  - [Authentication](#2-authentication)
  - [Admin Panel](#3-admin-panel)
  - [Role-Based Access Control](#4-role-based-access-control)
  - [Data Upload & Embedding](#5-data-upload--embedding)
  - [Search](#6-search)
  - [Semantic Filter](#7-semantic-filter-in-results)
  - [Context Window](#8-context-window)
  - [RAG Chat](#9-rag-chat)
  - [Hybrid Summary](#10-hybrid-summary)
  - [Summarize Results](#11-summarize-results)
  - [User Profile Analysis](#12-user-profile-analysis)
  - [Bookmarks & Labels](#13-bookmarks--labels)
  - [Suno Team Management](#14-suno-team-management)
  - [Stats](#15-stats)
- [API Reference](#api-reference)
- [Deployment](#deployment)

---

## System Overview

The platform operates as a FastAPI application backed by two storage layers:

| Layer | Technology | Purpose |
|---|---|---|
| Relational store | SQLite (WAL mode) + FTS5 | Structured message storage, keyword search, bookmarks, labels, users, sessions |
| Vector store | Qdrant or ChromaDB | Dense-vector similarity search and embedding storage |

Messages are ingested from CSV exports, stored in SQLite with a full-text index, and optionally embedded with OpenAI `text-embedding-3-small` (1536 dimensions, cosine distance) for semantic retrieval. All LLM calls use the OpenAI chat completions API and are streamed back to the browser as Server-Sent Events.

In **multi-user mode**, each user holds their own OpenAI API key. The server injects the correct per-request client using a Python `ContextVar` so every router and service layer transparently uses the authenticated user's key without signature changes.

---

## Architecture

```
Browser (HTML/JS)
      │
      │  HTTP / SSE
      ▼
FastAPI (app.py)
  ├── _SecurityHeadersMiddleware  X-Frame-Options, CSP headers
  ├── _AuthMiddleware             Bearer-token guard on /api/* (API_SECRET)
  ├── _SessionAuthMiddleware      Cookie-based session auth (multi mode only)
  │
  ├── routers/auth.py         /api/auth/* (login, register, logout, me, set-mode)
  ├── routers/admin.py        /api/admin/* (user list, delete, toggle-admin)
  ├── routers/config_api.py   API key & embedding model management
  ├── routers/stats.py        /api/stats
  ├── routers/uploads.py      CSV ingest, background embed jobs  [admin-only writes]
  ├── routers/search.py       Username / keyword / range / semantic search
  ├── routers/chat.py         RAG chat, summarisation, user profiling
  ├── routers/context.py      Context window, in-results semantic filter
  ├── routers/bookmarks.py    Bookmark CRUD + label assignment  [per-user scoped]
  ├── routers/labels.py       Label CRUD
  └── routers/suno_team.py    Suno-team member management  [admin-only writes]
      │
      ├── SQLite (discord_data.db)
      │     messages, uploads, bookmarks, labels, settings,
      │     embedded_uploads, messages_fts (FTS5),
      │     users, sessions
      │
      └── Vector Store (Qdrant / ChromaDB)
            Collection: discord_openai  (1536-dim, cosine)
```

**Shared runtime state** (`state.py`) holds the OpenAI client references, per-user client registry (`user_clients`), a `ContextVar` for per-request client injection, the vector collection wrappers, the background embed-job registry, and short-lived response caches (30-second TTL for `/api/stats` and `/api/uploads`).

A dedicated `ThreadPoolExecutor` (`vector_executor`, 4 workers) handles all blocking vector-store calls so they never block the asyncio event loop.

---

## App Modes

The platform supports three modes, controlled by the `APP_MODE` environment variable or configured interactively at first launch:

| Mode | Behaviour |
|---|---|
| `single` | One shared user. OpenAI API key is entered once in the browser (stored in `localStorage`). No login required. All data is global. |
| `multi` | Multiple accounts. Each user logs in with a username and password and manages their own OpenAI API key. Bookmarks are scoped to the user. Role-based access restricts dataset management to admins. |
| *(unset)* | On first visit, the server redirects to `/onboarding` where the user chooses the mode. The choice is persisted in SQLite. `APP_MODE` in `.env` always overrides the DB setting. |

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
| `APP_MODE` | — | `single`, `multi`, or unset (shows onboarding on first visit) |
| `API_SECRET` | — | Bearer token protecting all `/api/*` routes |
| `DB_PATH` | `discord_data.db` | SQLite file path |
| `MAX_UPLOAD_MB` | `50` | CSV upload size limit |
| `EMBED_BATCH_SIZE` | `2048` | Messages per OpenAI embedding request |
| `EMBED_CONCURRENCY` | `10` | Concurrent embedding API calls |

### Run

```bash
# Development (auto-reload on file change)
uvicorn app:app --reload --port 8000

# Production
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
```

> **Note:** Always restart the server after adding or modifying router files. Without `--reload`, the process uses the code snapshot taken at startup.

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

### 1. App Mode & Onboarding

**Pages:** `/onboarding`

When `APP_MODE` is not set in the environment and no mode has been saved to the database, all traffic redirects to `/onboarding`. This page presents two cards — **Single User** and **Multi User** — and explains the trade-offs of each.

Selecting a mode calls `POST /api/auth/set-mode`, which persists the choice to SQLite (`settings` table) and updates the in-process `state.app_mode`. After setting the mode the server redirects to `/login` (multi) or `/` (single). The onboarding page is inaccessible once a mode is configured.

---

### 2. Authentication

**Pages:** `/login`  
**Endpoints:** `POST /api/auth/login`, `POST /api/auth/register`, `POST /api/auth/logout`, `GET /api/auth/me`, `POST /api/auth/update-api-key`

Authentication is only active in **multi mode**. In single mode all auth endpoints return errors and the login page redirects to `/`.

#### Login and Registration

The `/login` page renders with two tabs — **Log In** and **Sign Up**. When no users exist (first run), the Sign Up tab is shown automatically.

- **Registration:** `POST /api/auth/register` validates the username (2–40 characters, unique) and password (minimum 8 characters). The password is hashed with PBKDF2-HMAC-SHA256 at 260,000 iterations with a 32-byte random salt. On success, a session cookie is issued and the user is redirected to `/`.
- **Login:** `POST /api/auth/login` performs the same hash verification with `secrets.compare_digest` to prevent timing attacks. A new session token is issued on each successful login.

#### Session Management

Sessions are stored in the `sessions` SQLite table (token, user_id, expires_at). The `_SessionAuthMiddleware` validates the `session` cookie on every request in multi mode:

- Public paths (static files, `/api/auth/*`, `/onboarding`, `/login`) bypass the check.
- Invalid or expired sessions on API routes return `401`. On page routes they redirect to `/login`.
- On valid sessions, `request.state.user` is populated and the user's OpenAI clients are injected into the `ContextVar` so downstream code can call `state.get_openai_client()` without any request object.

Session cookies are `HttpOnly`, `SameSite=Lax`, `Secure` (on HTTPS), and expire after 30 days.

#### Admin Bootstrap

At every startup, `ensure_admin_user("hafizh19", "LeoMessi10!")` runs idempotently:

1. If the username does not exist, creates the account and sets `is_admin = 1`.
2. If the account exists but is not admin, promotes it.
3. `migrate_bookmarks_to_user(admin_id)` assigns any bookmarks with `user_id = NULL` (pre-multi-mode data) to the admin account.

---

### 3. Admin Panel

**Page:** avatar dropdown → **Admin**  
**Endpoints:** `GET /api/admin/users`, `DELETE /api/admin/users/{user_id}`, `POST /api/admin/users/{user_id}/toggle-admin`

Visible only to admin accounts. The admin panel displays all registered users in a table with columns: username, role badge (Admin / User), join date, and an actions column.

Available actions per user (excluding the signed-in admin themselves):

| Action | Endpoint | Effect |
|---|---|---|
| Make Admin / Remove Admin | `POST …/toggle-admin` | Flips `is_admin` 0 ↔ 1 |
| Delete | `DELETE …/{user_id}` | Removes the account; their bookmarks lose the `user_id` reference (set to NULL via ON DELETE SET NULL) |

Admins cannot modify or delete their own account through this panel. The page can be refreshed without navigating away.

---

### 4. Role-Based Access Control

**File:** `routers/deps.py`

Two dependency functions are used across routers:

- `get_request_user(request)` — reads the session cookie and returns the user dict, or `None` in single mode.
- `require_admin(user=Depends(get_request_user))` — raises `403` if the user is not an admin; in single mode returns `{}` (all operations permitted).

**Write operations restricted to admins** in multi mode:

| Router | Restricted endpoints |
|---|---|
| `uploads.py` | Upload CSV, re-embed, delete upload (all 3 variants) |
| `suno_team.py` | Remove Suno team member |

**All users** (including non-admins) can:

- Search, view context windows, view bookmarks, RAG chat, run summaries
- Update their own OpenAI API key (`POST /api/auth/update-api-key`)
- Select the active embedding model (`POST /api/set-embedding-model`)
- Create, edit, and delete their own bookmarks and labels

---

### 5. Data Upload & Embedding

**Endpoint:** `POST /api/upload` *(admin only in multi mode)*

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

### 6. Search

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

### 7. Semantic Filter (In-Results)

**Endpoint:** `POST /api/filter/semantic`

Filters an already-retrieved set of message IDs by semantic relevance to a query without hitting the vector store's ANN index. Instead, it fetches the stored embedding vectors for the specified IDs directly and computes cosine similarity in-process using NumPy.

**Query preprocessing:**

- If the query starts with a question word (`what`, `how`, `why`, etc.) or ends with `?`, stop words and question words are stripped to isolate the semantic core. The similarity threshold is lowered to 0.20.
- Otherwise the query is embedded as-is with a threshold of 0.30.

This distinction prevents question-phrasing from diluting the embedding with low-information tokens.

---

### 8. Context Window

**Endpoint:** `GET /api/context/{message_id}`

Returns up to `before` (default 5, max 200) messages before the target and `after` (default 5, max 200) messages after it, ordered by `row_index` within the same upload. The target row has `is_target: true`. This preserves original chronological order within the CSV without relying on timestamp ordering (which may have gaps or ties).

---

### 9. RAG Chat

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

### 10. Hybrid Summary

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

### 11. Summarize Results

**Endpoint:** `POST /api/summarize-results` (initial) + `POST /api/summarize-results/followup`

Summarises the set of messages the browser currently displays (passed directly in the request body) rather than re-querying the database. This allows summarising any arbitrary selection of search results.

**Token safety net:** The payload character count is estimated (1 token ≈ 4 chars). If the estimated token count exceeds ~90,000 tokens, the pipeline auto-switches to cluster+sample regardless of the requested `retrieval_mode`. If clustering still produces an oversized payload, a hard-truncation loop progressively removes 20% of the message set until it fits within the limit.

**Follow-up:** The follow-up variant is stateless — all context lives in the `history` array sent by the browser; no database or vector-store calls are made.

---

### 12. User Profile Analysis

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

### 13. Bookmarks & Labels

Bookmarks allow any message to be saved for later review. Each bookmark stores:

- `msg_id` — the saved message
- `user_id` — the owning user in multi mode (`NULL` = global, single mode)
- `ctx_before` / `ctx_after` — how many context rows to show around it
- `note` — free-text annotation
- `created_at` — UTC timestamp

In **multi mode**, each user's bookmarks are isolated — `GET /api/bookmarks` and `GET /api/bookmarks/ids` return only the authenticated user's bookmarks. In **single mode**, bookmarks are global (no user scoping). Existing bookmarks without a `user_id` (created before multi-mode was enabled) are migrated to the admin account at startup.

Labels are independent coloured tags (name + hex colour) that can be assigned to bookmarks via a many-to-many join table (`bookmark_labels`). The `GET /api/bookmarks` endpoint returns bookmarks with their labels joined in a single query.

---

### 14. Suno Team Management

The `is_suno_team` field in each message row flags Suno AI staff messages. All search and summarisation endpoints accept a `suno_team` parameter:

| Value | Behaviour |
|---|---|
| `all` (default) | No filter |
| `only` | Restrict to Suno team messages |
| `exclude` | Exclude Suno team messages |

`GET /api/suno-team` returns all usernames currently flagged as Suno team members with their message counts. `DELETE /api/suno-team/{username}` sets `is_suno_team = 'false'` for all messages by that username, effectively demoting them from the team roster.

In multi mode, the delete operation is restricted to admins.

---

### 15. Stats

**Endpoint:** `GET /api/stats`

Returns aggregate counts fetched in parallel:

- `total_messages` — row count in the `messages` table
- `total_uploads` — row count in the `uploads` table
- `embedded_messages` — vector count in the active collection
- `api_key_set` — whether an OpenAI client is configured for the request
- `current_model` / `current_model_label` — active embedding model

Responses are cached in memory for 30 seconds to avoid repeated DB and vector-store hits from the dashboard polling.

---

## API Reference

### Auth

| Method | Path | Auth required | Description |
|---|---|---|---|
| `POST` | `/api/auth/set-mode` | No | Set app mode during onboarding |
| `POST` | `/api/auth/register` | No | Create new account (multi mode) |
| `POST` | `/api/auth/login` | No | Authenticate; set session cookie |
| `POST` | `/api/auth/logout` | No | Destroy session cookie |
| `GET` | `/api/auth/me` | Session | Current user info + is_admin flag |
| `POST` | `/api/auth/update-api-key` | Session | Update own OpenAI API key |
| `GET` | `/api/auth/users-exist` | No | Whether any users are registered |

### Admin

| Method | Path | Auth required | Description |
|---|---|---|---|
| `GET` | `/api/admin/users` | Admin | List all registered users |
| `DELETE` | `/api/admin/users/{user_id}` | Admin | Delete a user account |
| `POST` | `/api/admin/users/{user_id}/toggle-admin` | Admin | Flip admin status |

### Settings

| Method | Path | Auth required | Description |
|---|---|---|---|
| `POST` | `/api/set-api-key` | — | Set OpenAI API key (single mode) |
| `POST` | `/api/set-embedding-model` | — | Switch active embedding model |
| `GET` | `/api/embedding-models` | — | List models with vector counts |

### Data

| Method | Path | Auth required | Description |
|---|---|---|---|
| `GET` | `/api/stats` | — | Aggregate stats (cached 30 s) |
| `GET` | `/api/uploads` | — | List uploads with embedding status (cached 30 s) |
| `POST` | `/api/upload` | Admin | Upload CSV; streams SSE progress |
| `POST` | `/api/uploads/{id}/reembed` | Admin | Start background re-embed job |
| `GET` | `/api/jobs/{job_id}` | — | Poll embed job progress |
| `DELETE` | `/api/uploads/{id}` | Admin | Delete from SQLite + vector store |
| `DELETE` | `/api/uploads/{id}/sqlite` | Admin | Delete from SQLite only |
| `DELETE` | `/api/uploads/{id}/embeddings` | Admin | Delete vectors only |

### Search

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/search/username` | Search by username |
| `GET` | `/api/search/keyword` | FTS5 keyword search |
| `GET` | `/api/search/range` | Date range fetch |
| `GET` | `/api/search/semantic` | Vector similarity search |
| `GET` | `/api/search/users-in-range` | Per-user activity stats |
| `GET` | `/api/search/user-messages` | All messages by one user |
| `POST` | `/api/search/bulk-context` | Context windows for multiple messages |
| `GET` | `/api/context/{id}` | Context window for one message |
| `POST` | `/api/filter/semantic` | Semantic filter on a result set |

### Chat & Summarisation

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chat` | RAG chat (streaming SSE) |
| `POST` | `/api/summarize` | Hybrid summary (streaming SSE) |
| `POST` | `/api/summarize/followup` | Hybrid summary follow-up Q&A |
| `POST` | `/api/summarize-results` | Summarise browser result set (streaming SSE) |
| `POST` | `/api/summarize-results/followup` | Summarise results follow-up Q&A |
| `POST` | `/api/user-profile` | User persona profile (streaming SSE) |
| `POST` | `/api/user-profile/followup` | User profile follow-up Q&A |

### Bookmarks & Labels

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/bookmarks` | Create bookmark |
| `GET` | `/api/bookmarks` | List bookmarks with labels (scoped to user in multi mode) |
| `GET` | `/api/bookmarks/ids` | List bookmarked message IDs |
| `DELETE` | `/api/bookmarks/{id}` | Delete bookmark |
| `DELETE` | `/api/bookmarks/by-msg/{msg_id}` | Delete bookmark by message ID |
| `POST` | `/api/bookmarks/{id}/labels/{label_id}` | Assign label to bookmark |
| `DELETE` | `/api/bookmarks/{id}/labels/{label_id}` | Remove label from bookmark |
| `POST` | `/api/labels` | Create label |
| `GET` | `/api/labels` | List labels |
| `PUT` | `/api/labels/{id}` | Update label |
| `DELETE` | `/api/labels/{id}` | Delete label |

### Suno Team

| Method | Path | Auth required | Description |
|---|---|---|---|
| `GET` | `/api/suno-team` | — | List Suno team members |
| `DELETE` | `/api/suno-team/{username}` | Admin | Remove Suno team flag |

---

## Deployment

The repository includes a GitHub Actions workflow (`.github/workflows/deploy.yml`) that deploys to a VPS on every push to `main`:

1. SSH into the VPS using a secret key.
2. `git pull origin main` in the app directory.
3. `docker compose up -d --build` rebuilds and restarts the container.
4. `docker image prune -f` removes dangling images.

The workflow uses `concurrency: group: production-deploy` with `cancel-in-progress: true` to prevent overlapping deployments from concurrent pushes.

> **Multi-mode deployment note:** The admin account (`hafizh19`) is bootstrapped automatically on every server start via `ensure_admin_user()`. No manual database setup is required. If deploying fresh, the first startup will create the admin user and you can begin registering other accounts via `/login`.
