# Suno Discord Analysis — Retrieval & Analysis Platform

A web-based research tool for ingesting, searching, and analysing exported Discord conversation data using keyword search, vector-similarity retrieval, and LLM-powered summarisation. Built for academic research into the Suno AI community.

Supports both **single-user** and **multi-user** deployment modes, with role-based access control, session authentication, and per-user data isolation.

For the detailed technical implementation of each feature, see [TECHNICAL_FLOW.md](TECHNICAL_FLOW.md).
For deployment instructions, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

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
  - [Hybrid Summary](#9-hybrid-summary)
  - [Summarize Results](#10-summarize-results)
  - [User Profile Analysis](#11-user-profile-analysis)
  - [Bookmarks & Labels](#12-bookmarks--labels)
  - [Qualitative Coding](#13-qualitative-coding)
  - [Suno Team Management](#14-suno-team-management)
  - [Stats](#15-stats)
- [API Reference](#api-reference)
- [Deployment](#deployment)

---

## System Overview

The platform is a full-stack web application with a Rust/Axum backend and a React/TypeScript frontend, backed by two storage layers:

| Layer | Technology | Purpose |
|---|---|---|
| Relational store | PostgreSQL 16 + `tsvector` | Structured message storage, keyword search, bookmarks, labels, users, sessions, qualitative codes |
| Vector store | Qdrant | Dense-vector similarity search and embedding storage (1536-dim, cosine) |

Messages are ingested from CSV exports, stored in PostgreSQL with a full-text index updated by a database trigger, and optionally embedded with OpenAI `text-embedding-3-small` for semantic retrieval. All LLM calls use the OpenAI chat completions API and are streamed back to the browser as Server-Sent Events.

In **multi-user mode**, each user's OpenAI API key is sent as an `X-OpenAI-Key` request header from the browser — the server never persists it to the database.

---

## Architecture

```
Browser (React 18 + TypeScript 5.5 + Vite 5.4)
      │
      │  HTTP / SSE
      ▼
Axum 0.7 (Rust)
  ├── CORSLayer                any origin (dev/nginx handles prod restriction)
  ├── SessionManagerLayer      PostgreSQL-backed sessions (tower-sessions)
  ├── TraceLayer               HTTP request/response tracing
  ├── DefaultBodyLimit         200 MB (for CSV uploads)
  │
  ├── routes/auth.rs           /api/auth/*     (login, register, logout, me, set-mode)
  ├── routes/admin.rs          /api/admin/*    (user list, delete, toggle-admin, set-password)
  ├── routes/config_api.rs     /api/*          (api key, embedding model)
  ├── routes/stats.rs          /api/stats
  ├── routes/uploads.rs        /api/upload, /api/uploads/*  (CSV ingest, embed, delete)
  ├── routes/search.rs         /api/search/*   (keyword, semantic HyDE, range, username)
  ├── routes/chat.rs           /api/summarize, /api/summarize-results, /api/user-profile
  ├── routes/context_route.rs  /api/context/*, /api/filter/semantic
  ├── routes/bookmarks.rs      /api/bookmarks/*, /api/labels/*
  ├── routes/codes.rs          /api/codes/*, /api/code-categories/*, /api/bookmark-codes/*
  └── routes/suno_team.rs      /api/suno-team/*
      │
      ├── PostgreSQL 16
      │     messages, uploads, bookmarks, labels, bookmark_labels,
      │     embedded_uploads, settings, users,
      │     codes, code_categories, bookmark_codes, bookmark_code_highlights
      │
      └── Qdrant
            Collection: discord_openai  (1536-dim, cosine distance)
```

---

## App Modes

| Mode | Behaviour |
|---|---|
| `single` | One shared user. OpenAI API key is entered once in the browser (stored in `localStorage`). No login required. All data is global. |
| `multi` | Multiple accounts. Each user logs in with a username and password and manages their own OpenAI API key. Bookmarks are scoped to the user. Role-based access restricts dataset management to admins. |
| `demo` | Behaves like single mode — auto-authenticates as admin without a login prompt. |
| *(unset)* | On first visit, an onboarding page lets the user choose the mode. The choice is persisted in the `settings` table. `APP_MODE` in `.env` always overrides the database setting. |

---

## Setup

### Prerequisites

- [Rust](https://rustup.rs/) (stable toolchain)
- [Node.js](https://nodejs.org/) 20+
- PostgreSQL 16 (local or remote)
- Qdrant (local Docker or cloud)
- An OpenAI API key (for embeddings and LLM summarisation)

### Install

```bash
# Install frontend dependencies
cd frontend && npm install && cd ..

# Build backend (debug for development)
cd backend && cargo build && cd ..
```

### Environment

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

The `.env` file should be placed in the **project root** (one level above `backend/`). The backend loads it from `../` relative to its working directory.

Key variables:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgres://postgres:password@localhost/discord_db` | PostgreSQL connection string |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_COLLECTION` | `discord_openai` | Qdrant collection name |
| `QDRANT_API_KEY` | *(empty)* | Optional Qdrant authentication token |
| `OPENAI_API_KEY` | *(empty)* | Pre-load an API key at server startup |
| `APP_MODE` | *(unset)* | `single`, `multi`, `demo`, or unset (shows onboarding) |

### Run (Development)

```bash
# Terminal 1 — Backend (auto-compiles and runs)
cd backend && cargo run

# Terminal 2 — Frontend (hot-reload dev server)
cd frontend && npm run dev
```

The backend listens on `http://0.0.0.0:8000`. The Vite dev server proxies `/api/*` requests to the backend.

### Run (Production — single process)

Build the frontend and serve its `dist/` from the backend as static files:

```bash
cd frontend && npm run build
cd backend && cargo build --release && ./target/release/retrieval-backend
```

The backend serves `frontend/dist/` as static files and falls back to `index.html` for all non-API routes (SPA routing).

---

## Configuration Reference

### CSV Format

The upload endpoint expects a CSV with at minimum these columns (headers are normalised to lowercase with underscores):

| Column | Required | Description |
|---|---|---|
| `author_id` | Yes | Discord user ID |
| `username` | Yes | Display name |
| `date` | Yes | ISO-8601 timestamp or date string |
| `content` | Yes | Message text |
| `attachments` | No | Attachment URLs or metadata |
| `reactions` | No | Reaction counts |
| `is_suno_team` | No | `"true"` / `"1"` flags Suno team members |
| `week` | No | ISO week label (auto-computed from date if absent) |
| `month` | No | Month label |

The uploader also recognises alternate spellings such as `msg_uuid`, `message id`, `id` and maps them automatically.

### Chat Model Allowlist

Only the following OpenAI model IDs are accepted at summarisation and profile endpoints:

- GPT-5.x series: `gpt-5.4`, `gpt-5.4-pro`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`
- GPT-4.1 series: `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`
- GPT-4o series: `gpt-4o`, `gpt-4o-mini`
- GPT-4 legacy: `gpt-4-turbo`, `gpt-4`
- GPT-3.5: `gpt-3.5-turbo`
- o-series: `o4-mini`, `o3`, `o3-mini`, `o1`, `o1-mini`, `o1-preview`

---

## Features

### 1. App Mode & Onboarding

**Page:** `/onboarding`

When `APP_MODE` is not set and no mode has been saved to the database, the first visit redirects to `/onboarding`. This page presents two cards — **Single User** and **Multi User** — and explains the trade-offs.

Selecting a mode calls `POST /api/auth/set-mode`, which persists the choice to the `settings` table and redirects to `/login` (multi) or `/` (single/demo). The onboarding page is inaccessible once a mode is configured.

---

### 2. Authentication

**Page:** `/login`  
**Endpoints:** `POST /api/auth/login`, `POST /api/auth/register`, `POST /api/auth/logout`, `GET /api/auth/me`

Authentication is only active in **multi mode**. In single and demo modes the login page redirects to `/`.

#### Login and Registration

- **Registration:** Validates username (2–40 characters, case-insensitive unique) and password (minimum 8 characters). Password is hashed with **Argon2id** (memory 19 MiB, 2 iterations, parallelism 1) with a cryptographically random salt from `OsRng`. On success, a session cookie is issued.
- **Login:** Verifies the Argon2id hash. A new session is issued on each successful login.

#### Session Management

Sessions are stored in a PostgreSQL table managed by `tower-sessions-sqlx-store`.

- Session TTL: **30 days on inactivity** (`Expiry::OnInactivity`)
- Session cookies: `HttpOnly`, `SameSite=Lax`
- Invalid or expired sessions on API routes return `401`; on page routes they redirect to `/login`

#### Admin Password Reset

`POST /api/admin/users/:user_id/password` — allows an admin to set any user's password directly, useful for account recovery.

---

### 3. Admin Panel

**Page:** Avatar dropdown → **Admin**  
**Endpoints:** `GET /api/admin/users`, `DELETE /api/admin/users/:id`, `POST /api/admin/users/:id/toggle-admin`, `POST /api/admin/users/:id/password`

Visible only to admin accounts in multi mode. Displays all registered users with username, admin badge, join date, and action controls.

| Action | Endpoint | Effect |
|---|---|---|
| Make Admin / Remove Admin | `POST .../toggle-admin` | Flips `is_admin` boolean |
| Set Password | `POST .../password` | Resets the user's password |
| Delete | `DELETE .../:id` | Removes account; their bookmarks lose `user_id` (set NULL via `ON DELETE SET NULL`) |

Admins cannot modify or delete their own account through this panel.

---

### 4. Role-Based Access Control

Two Axum extractor types enforce access:

- `AuthUser` — resolves the session, returns the user. In single/demo mode, always returns a default admin object.
- `AdminUser` — wraps `AuthUser`, returns `403 Forbidden` if `is_admin` is false.

**Operations restricted to admins** (in multi mode):

| Route module | Restricted endpoints |
|---|---|
| `uploads.rs` | Upload CSV, re-embed, delete upload (all 3 variants) |
| `suno_team.rs` | Add/remove Suno team members |
| `admin.rs` | All admin endpoints |

All other endpoints (search, summarise, bookmarks, codes, labels) are accessible to any authenticated user.

---

### 5. Data Upload & Embedding

**Endpoint:** `POST /api/upload` *(admin only in multi mode)*

A CSV file (up to 200 MB) is uploaded and processed in a streaming response. The server responds with Server-Sent Events so the browser displays live progress.

**Phase 1 — Insert:**

1. Column headers are normalised.
2. Required columns are validated.
3. Rows are inserted in batches of 500 with `ON CONFLICT (msg_uuid) DO NOTHING` (deduplication guard).
4. `word_count` is computed as the space-count+1 heuristic at insert time.
5. `week` is derived from the `date` column as ISO week format `YYYY-IW`.
6. `is_suno_team` is backfilled across uploads: usernames already flagged in the database are automatically flagged in the new upload.
7. A PostgreSQL trigger fires on each insert to populate the `search_vector` tsvector column from `content` and `username`.

**Phase 2 — Embed** (if OpenAI API key is set):

1. Messages with non-empty content are collected.
2. Already-embedded message UUIDs are identified via a Qdrant scroll query (10,000 points per page) — this is the **resumability check**.
3. Remaining messages are embedded using `text-embedding-3-small` in batches of **500**, with **3** concurrent requests.
4. Rate-limit responses (HTTP 429) are retried with exponential back-off (2s, 4s, 8s), honouring `Retry-After`.
5. Discord Snowflake IDs are converted to UUID v5 (namespace `NAMESPACE_OID`) for stable Qdrant point IDs.
6. Vectors are upserted with payload `{msg_uuid, upload_id, document}`.
7. On completion, `embedded_uploads` is updated.

**Re-embedding:** `POST /api/uploads/:id/reembed` streams the same embed pipeline for an existing upload, always running the resumability check first.

**Deletion options:**

| Endpoint | Effect |
|---|---|
| `DELETE /api/uploads/:id` | Removes from both PostgreSQL and Qdrant |
| `DELETE /api/uploads/:id/sqlite` | Removes from PostgreSQL only; vectors preserved |
| `DELETE /api/uploads/:id/embeddings` | Removes Qdrant vectors only; DB rows preserved |

---

### 6. Search

All search endpoints share these filter parameters:

| Parameter | Type | Description |
|---|---|---|
| `upload_ids` | string (CSV) | Restrict to specific upload IDs |
| `date_from` | `YYYY-MM-DD` | Earliest message date |
| `date_to` | `YYYY-MM-DD` | Latest message date |
| `is_suno_team` | `"true"`/`"only"` or `"false"`/`"exclude"` | Suno team membership filter |
| `min_words` | int | Minimum word count per message |
| `limit` | int | Maximum results (default 200, max 10,000) |
| `username` | string | Filter by username (partial match) |

#### Username Search — `GET /api/search/username`

Case-insensitive `ILIKE` query on the `username` column. Supports partial matching. Results ordered by `(date, row_index)`.

#### Keyword Search — `GET /api/search/keyword`

Uses the PostgreSQL `search_vector` tsvector column (GIN-indexed, maintained by trigger) for fast full-text search. Supports three match modes via the `match_type` parameter:

| Mode | Behaviour |
|---|---|
| `fuzzy` | AND of all prefix-matched tokens: `token:* & token2:*` |
| `exact` | Adjacent phrase match using `phraseto_tsquery` |
| `any_word` | OR of all tokens: `token | token2` |

If the GIN index is not yet ready (startup backfill in progress), search falls back transparently to an `ILIKE` word-by-word scan.

#### Date Range Search — `GET /api/search/range`

Returns all messages within the specified date range and active filters, with no keyword constraint. Ordered by `(date, row_index)`.

#### Semantic Search — `GET /api/search/semantic`

Uses **HyDE (Hypothetical Document Embeddings)** for improved retrieval quality:

1. **In parallel:** embed the raw query and call `gpt-4o-mini` to generate 2-3 hypothetical Discord messages (~80 words) relevant to the query, then embed those.
2. **Blend:** `blended = normalize(raw) × (1 − w) + normalize(hyde) × w`, where `w = 0.75` for queries ≤ 3 words, `0.55` otherwise.
3. **Qdrant ANN search:** fetch `min(limit × 8, 800)` nearest neighbours.
4. **Relative threshold:** keep results with `score ≥ max(0.15, best_score × 0.50)`.
5. **PostgreSQL fetch:** look up matching rows and apply all metadata filters.
6. Return up to `limit` results with `similarity_score`.

#### Users in Range — `GET /api/search/users-in-range`

Returns per-user statistics for the filtered message set: message count, first/last message date, average word count, distinct active weeks, and percentage of weeks active.

#### User Messages — `GET /api/search/user-messages`

Exact-match username query with optional keyword sub-filter. Returns the full chronological message list for a single user.

#### Bulk Context — `POST /api/search/bulk-context`

Fetches context windows for multiple messages in one request. For each `msg_id`, returns the target plus `before`/`after` surrounding messages from the same upload (by `row_index`). The target row carries `is_target: true`.

---

### 7. Semantic Filter (In-Results)

**Endpoint:** `POST /api/filter/semantic`

Filters an already-retrieved message set by semantic relevance to a query without performing a new ANN query. Stored embedding vectors are fetched from Qdrant for the specified IDs and cosine similarity is computed in-process.

**Query preprocessing:** Queries starting with question words or ending with `?` have stop words stripped (threshold 0.20). Non-question queries use threshold 0.30.

---

### 8. Context Window

**Endpoint:** `GET /api/context/:message_id`

Returns up to `before` (default 5, max 200) messages before and `after` (default 5, max 200) after the target, ordered by `row_index` within the same upload. This preserves original CSV order regardless of timestamp precision or duplicates.

---

### 9. Hybrid Summary

**Endpoints:** `POST /api/summarize` · `POST /api/summarize/followup`

A multi-stage pipeline that selects a statistically representative evidence set from the database corpus and summarises it with an LLM.

**Pipeline stages:**

```
1. Metadata filter       SQL filter by username, date, uploads, Suno team, min_words
2. Semantic retrieval    query embedding → Qdrant ANN → intersect with filtered set
3. Adaptive threshold    keep top 70% by score (minimum 15 candidates)
4. Deduplication         cosine similarity ≥ 0.97 → drop near-duplicates
5. HDBSCAN clustering    pure-Rust implementation, cosine distance
6. Per-cluster sampling  5 closest + 5 furthest per cluster centroid
7. Assemble              chronological sort, capped at 120 messages
8. LLM generation        streaming SSE to browser
```

A **transparency log** is emitted as SSE events before the LLM stream, showing filter count, retrieval count, dedup removals, algorithm used, cluster count, and evidence count. This is visible in the browser as a "research pipeline" trace.

**Follow-up Q&A:** Re-runs the same pipeline with the follow-up question as the retrieval query. The initial summary is embedded in the system prompt; prior Q&A turns are appended.

---

### 10. Summarize Results

**Endpoints:** `POST /api/summarize-results` · `POST /api/summarize-results/followup`

Summarises the messages currently displayed in the browser (passed directly in the request body) rather than re-querying the database. This allows summarising any arbitrary selection of search results.

Uses the same HDBSCAN dedup/cluster/sample pipeline as the Hybrid Summary, with stored embeddings fetched from Qdrant by UUID rather than by ANN query. The follow-up variant is fully stateless — all context is in the `history` array; no database or Qdrant calls are made.

---

### 11. User Profile Analysis

**Endpoints:** `POST /api/user-profile` · `POST /api/user-profile/followup`

Analyses all messages from a specific user to produce a structured persona profile using the same retrieval → dedup → cluster → sample pipeline, filtered to an exact-match on `username`. The LLM prompt requests: entry/exit dates, persona description, evolution of attitude (chronological narrative), key recurring topics, representative verbatim quotes, and summary assessment.

---

### 12. Bookmarks & Labels

**Endpoints:** `/api/bookmarks/*`, `/api/labels/*`

Bookmarks allow any message to be saved with context window settings, a free-text note, and colour-coded labels. Each bookmark stores:

- `msg_id` — the saved message
- `user_id` — the owning user in multi mode (`NULL` = global)
- `ctx_before` / `ctx_after` — context row count for display
- `note` — free-text annotation
- `created_at` — UTC timestamp

In **multi mode**, each user's bookmarks are isolated. In **single mode**, bookmarks are global. Labels are coloured tags (name + hex colour) assigned via a many-to-many join table. `GET /api/bookmarks` returns bookmarks with their labels joined in a single query to avoid N+1.

---

### 13. Qualitative Coding

**Endpoints:** `/api/codes/*`, `/api/code-categories/*`, `/api/bookmark-codes/*`
**Page:** `/coding` (manager view and table view)

A qualitative analysis system that enables systematic coding of bookmarked messages, mirroring the grounded-theory research workflow.

**Codes** are named, coloured tags with an optional description. **Code Categories** organise codes into a two-level hierarchy (categories can have a `parent_id`). Codes are assigned to bookmarks at two levels of granularity:

- **Message-level coding** (`bookmark_codes`) — assigns a code to an entire bookmarked message.
- **Passage-level coding** (`bookmark_code_highlights`) — assigns a code to a specific text excerpt within the message, with the `highlighted_text` stored. Multiple highlights per `(bookmark, code)` pair are supported.

The **Coding Manager** view (`/coding/manager`) provides a code-book interface for creating and organising codes and categories. The **Coding Table** view (`/coding/table`) presents all bookmarks and their assigned codes in a spreadsheet-like layout for review and export.

---

### 14. Suno Team Management

The `is_suno_team` field flags Suno AI staff messages. All search and summarisation endpoints accept an `is_suno_team` parameter to include, exclude, or isolate Suno team messages.

`GET /api/suno-team` — list all usernames currently flagged, with message counts.  
`POST /api/suno-team/:username` — flag all messages by that username as Suno team.  
`DELETE /api/suno-team/:username` — remove the Suno team flag for all messages by that username.

In multi mode, write operations are restricted to admins.

---

### 15. Stats

**Endpoint:** `GET /api/stats`

Returns aggregate counts:

- `total_messages` — row count in `messages`
- `total_uploads` — row count in `uploads`
- `embedded_messages` — vector count in the Qdrant collection
- `api_key_set` — whether an OpenAI API key is available for the request
- `vector_db_label` — always `"Qdrant"`

---

## API Reference

### Auth

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/auth/set-mode` | Set app mode during onboarding |
| `POST` | `/api/auth/register` | Create new account (multi mode) |
| `POST` | `/api/auth/login` | Authenticate; set session cookie |
| `POST` | `/api/auth/logout` | Destroy session |
| `GET` | `/api/auth/me` | Current user info + is_admin flag |
| `GET` | `/api/auth/app-mode` | Get current app mode |

### Admin

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/api/admin/users` | Admin | List all registered users |
| `DELETE` | `/api/admin/users/:id` | Admin | Delete a user account |
| `POST` | `/api/admin/users/:id/toggle-admin` | Admin | Flip admin status |
| `POST` | `/api/admin/users/:id/password` | Admin | Reset user password |

### Settings

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/set-api-key` | Set OpenAI API key (single/demo mode) |
| `POST` | `/api/set-embedding-model` | Switch active embedding model |
| `GET` | `/api/embedding-models` | List models with vector counts |

### Data

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/api/stats` | — | Aggregate stats |
| `GET` | `/api/uploads` | — | List uploads with embedding status |
| `POST` | `/api/upload` | Admin | Upload CSV; streams SSE progress (200 MB limit) |
| `POST` | `/api/uploads/:id/reembed` | Admin | Re-embed upload (SSE) |
| `DELETE` | `/api/uploads/:id` | Admin | Delete from PostgreSQL + Qdrant |
| `DELETE` | `/api/uploads/:id/sqlite` | Admin | Delete from PostgreSQL only |
| `DELETE` | `/api/uploads/:id/embeddings` | Admin | Delete Qdrant vectors only |

### Search

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/search/username` | Search by username |
| `GET` | `/api/search/keyword` | PostgreSQL full-text search |
| `GET` | `/api/search/range` | Date range fetch |
| `GET` | `/api/search/semantic` | Vector similarity search (HyDE) |
| `GET` | `/api/search/users-in-range` | Per-user activity stats |
| `GET` | `/api/search/user-messages` | All messages by one user |
| `POST` | `/api/search/bulk-context` | Context windows for multiple messages |
| `GET` | `/api/context/:id` | Context window for one message |
| `POST` | `/api/filter/semantic` | Semantic filter on a result set |

### Summarisation

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/summarize` | Hybrid summary (streaming SSE) |
| `POST` | `/api/summarize/followup` | Hybrid summary follow-up Q&A |
| `POST` | `/api/summarize-results` | Summarise browser result set (SSE) |
| `POST` | `/api/summarize-results/followup` | Summarise results follow-up Q&A |
| `POST` | `/api/user-profile` | User persona profile (SSE) |
| `POST` | `/api/user-profile/followup` | User profile follow-up Q&A |

### Bookmarks & Labels

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/bookmarks` | Create bookmark |
| `GET` | `/api/bookmarks` | List bookmarks with labels (user-scoped in multi) |
| `GET` | `/api/bookmarks/ids` | List bookmarked message IDs |
| `GET` | `/api/bookmarks/meta` | List bookmarks (lightweight, no codes) |
| `DELETE` | `/api/bookmarks/:id` | Delete bookmark |
| `DELETE` | `/api/bookmarks/by-msg/:msg_id` | Delete bookmark by message ID |
| `POST` | `/api/bookmarks/:id/labels/:label_id` | Assign label to bookmark |
| `DELETE` | `/api/bookmarks/:id/labels/:label_id` | Remove label from bookmark |
| `POST` | `/api/bookmarks/:id/highlights` | Add passage-level code highlight |
| `DELETE` | `/api/bookmarks/:id/highlights/:hl_id` | Remove code highlight |
| `POST` | `/api/labels` | Create label |
| `GET` | `/api/labels` | List labels |
| `PUT` | `/api/labels/:id` | Update label |
| `DELETE` | `/api/labels/:id` | Delete label |

### Qualitative Codes

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/codes` | Create code |
| `GET` | `/api/codes` | List codes |
| `PUT` | `/api/codes/:id` | Update code |
| `DELETE` | `/api/codes/:id` | Delete code |
| `POST` | `/api/code-categories` | Create code category |
| `GET` | `/api/code-categories` | List code categories |
| `PUT` | `/api/code-categories/:id` | Update category |
| `DELETE` | `/api/code-categories/:id` | Delete category |
| `POST` | `/api/bookmark-codes` | Assign code to bookmark |
| `GET` | `/api/bookmark-codes` | List all bookmark-code assignments |
| `GET` | `/api/bookmark-codes/:bookmark_id` | List codes for a bookmark |
| `DELETE` | `/api/bookmark-codes/:bookmark_id/:code_id` | Remove code from bookmark |

### Suno Team

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/api/suno-team` | — | List Suno team members |
| `POST` | `/api/suno-team/:username` | Admin | Flag username as Suno team |
| `DELETE` | `/api/suno-team/:username` | Admin | Remove Suno team flag |

---

## Deployment

The repository includes a GitHub Actions workflow that deploys to a VPS on every push to `main`. The pipeline:

1. Compiles the React frontend (`npm run build`)
2. Compiles the Rust backend binary (`cargo build --release`)
3. Packages the binary into a Docker image and pushes to GitHub Container Registry (`ghcr.io`)
4. Rsyncs the frontend `dist/` to the VPS and restarts the Docker Compose stack

**Services running on VPS:**

```
nginx    (port 80/443)    serves frontend + proxies /api/* to backend
backend  (internal)       Rust API on port 8000
postgres (internal)       PostgreSQL 16
qdrant   (internal)       Qdrant vector store on port 6333
```

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for step-by-step instructions including DNS, SSL, GitHub secrets, PostgreSQL migration, and ongoing operations.
