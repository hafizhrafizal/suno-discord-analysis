# Technical Flow — Suno Discord Analysis Platform

This document provides implementation-level descriptions of every feature for academic transparency. It traces data from the HTTP request through each layer to the final response.

**Backend:** Rust 1.x + Axum 0.7, async runtime via Tokio, database access via sqlx 0.7 (compile-time-checked queries).  
**Frontend:** React 18.3 + TypeScript 5.5, built with Vite 5.4, data fetching via TanStack Query 5.56, global state via Zustand 4.5.  
**Relational store:** PostgreSQL 16 with `tsvector` full-text search.  
**Vector store:** Qdrant (REST API, 1536-dim cosine).

---

## Table of Contents

1. [Application Startup](#1-application-startup)
2. [Database Schema](#2-database-schema)
3. [Vector Store Layer (Qdrant)](#3-vector-store-layer-qdrant)
4. [Embedding Pipeline](#4-embedding-pipeline)
5. [Keyword Search (PostgreSQL FTS)](#5-keyword-search-postgresql-fts)
6. [Semantic Search with HyDE](#6-semantic-search-with-hyde)
7. [In-Results Semantic Filter](#7-in-results-semantic-filter)
8. [Context Window](#8-context-window)
9. [Summarize Results (HDBSCAN Pipeline)](#9-summarize-results-hdbscan-pipeline)
10. [Hybrid Summary](#10-hybrid-summary)
11. [User Profile Analysis](#11-user-profile-analysis)
12. [Bookmarks and Labels](#12-bookmarks-and-labels)
13. [Qualitative Coding System](#13-qualitative-coding-system)
14. [Authentication and Sessions](#14-authentication-and-sessions)
15. [Security Model](#15-security-model)
16. [SSE Streaming Pattern](#16-sse-streaming-pattern)

---

## 1. Application Startup

**File:** `backend/src/main.rs`

On startup, the following steps execute in order:

1. **Environment** — `dotenvy::dotenv_override()` loads `.env` from one directory above the backend working directory (the project root). `RUST_LOG` defaults to `"retrieval_backend=debug,tower_http=info"`.

2. **PostgreSQL pool** — `sqlx::postgres::PgPoolOptions::new().max_connections(10).connect_with()` establishes the connection pool.

3. **Schema initialisation** — `init_db(&pool)` runs `CREATE TABLE IF NOT EXISTS` for all 12 tables and creates all indexes and triggers. See §2 for the complete schema.

4. **FTS backfill** — If `search_vector` is NULL for any `messages` rows (migration from a schema version that lacked the column), a `tokio::spawn` background task fills them using:
   ```sql
   UPDATE messages SET search_vector =
     to_tsvector('simple', COALESCE(content,'') || ' ' || COALESCE(username,''))
   WHERE search_vector IS NULL
   ```
   The startup log line `search_vector index is current — fast keyword search active from startup` confirms no backfill was needed.

5. **Word-count backfill** — If `word_count` is NULL for any rows, a similar background task fills it. The computation is the space-count+1 heuristic: `array_length(string_to_array(trim(content), ' '), 1)`.

6. **Week backfill** — If `week` is NULL for any rows, fills it from the `date` column as `to_char(date::date, 'IYYY-IW')`.

7. **Admin seed** — In `single` or `demo` mode, `UPSERT INTO users (id=1, username='admin', is_admin=true)` ensures a default admin account exists.

8. **Session store** — `PostgresStore::new(pool.clone())` is constructed and `.migrate()` is called to create the `tower_sessions` table. Session expiry is set to `Expiry::OnInactivity(Duration::seconds(30 * 24 * 3600))` — **30 days of inactivity**.

9. **Router assembly** — All route modules are merged via `Router::merge()`. The middleware stack is layered (innermost to outermost):
   ```
   DefaultBodyLimit::max(200 * 1024 * 1024)   ← 200 MB upload limit
   TraceLayer::new_for_http()
   SessionManagerLayer::new(session_store)
   CORSLayer::very_permissive()
   ```

10. **Bind** — `axum::serve(TcpListener::bind("0.0.0.0:8000").await?, app)`.

**App state** (`AppState`) holds:
- `db: PgPool` — the database connection pool
- `config: Config` — parsed environment configuration

---

## 2. Database Schema

**File:** `backend/src/models.rs` — `init_db()` function

```sql
-- ── Core data ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS uploads (
    id          TEXT    PRIMARY KEY,               -- UUID assigned at upload time
    filename    TEXT    NOT NULL,
    row_count   BIGINT  DEFAULT 0,
    upload_time TEXT    NOT NULL                   -- ISO-8601 UTC
);

CREATE TABLE IF NOT EXISTS messages (
    id            BIGSERIAL PRIMARY KEY,
    msg_uuid      TEXT    UNIQUE NOT NULL,          -- UUID assigned at upload time
    author_id     TEXT,
    username      TEXT    NOT NULL DEFAULT 'unknown',
    date          TEXT,                             -- ISO-8601 string from CSV
    content       TEXT,
    attachments   TEXT,
    reactions     TEXT,
    is_suno_team  TEXT    DEFAULT 'false',
    week          TEXT,                             -- "YYYY-IW" (ISO week, e.g. "2024-03")
    month         TEXT,
    upload_id     TEXT    REFERENCES uploads(id) ON DELETE CASCADE,
    row_index     BIGINT,                           -- 0-based position in original CSV
    word_count    BIGINT,                           -- space-count + 1 heuristic
    search_vector tsvector                          -- maintained by trigger (see below)
);

-- Indexes on messages
CREATE INDEX IF NOT EXISTS idx_messages_date
    ON messages(date);
CREATE INDEX IF NOT EXISTS idx_messages_username
    ON messages(username);
CREATE INDEX IF NOT EXISTS idx_messages_upload_id
    ON messages(upload_id);
CREATE INDEX IF NOT EXISTS idx_messages_suno_team
    ON messages(is_suno_team);
CREATE INDEX IF NOT EXISTS idx_messages_date_username
    ON messages(date, username);
CREATE INDEX IF NOT EXISTS idx_messages_week
    ON messages(week);
CREATE INDEX IF NOT EXISTS idx_messages_search_vector
    USING GIN ON messages(search_vector);          -- enables O(log n) FTS

-- Trigger: keep search_vector synchronised with content and username
CREATE OR REPLACE FUNCTION update_search_vector() RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector :=
        to_tsvector('simple',
            COALESCE(NEW.content, '') || ' ' || COALESCE(NEW.username, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER tg_messages_sv_update
BEFORE INSERT OR UPDATE OF content, username ON messages
FOR EACH ROW EXECUTE FUNCTION update_search_vector();

-- ── Authentication ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id            BIGSERIAL PRIMARY KEY,
    username      TEXT    NOT NULL,
    password_hash TEXT,                            -- Argon2id PHC string
    password_salt TEXT,                            -- random Base64 salt
    is_admin      BOOLEAN DEFAULT false,
    created_at    TEXT    NOT NULL,                -- ISO-8601 UTC
    google_id     TEXT    UNIQUE,                  -- reserved for future OAuth
    email         TEXT    UNIQUE
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username
    ON users(LOWER(username));                     -- case-insensitive uniqueness

-- ── Bookmarks ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS bookmarks (
    id         BIGSERIAL PRIMARY KEY,
    msg_id     BIGINT  NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    ctx_before BIGINT  DEFAULT 5,
    ctx_after  BIGINT  DEFAULT 5,
    note       TEXT,
    created_at TEXT    NOT NULL,
    user_id    BIGINT  REFERENCES users(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS labels (
    id         BIGSERIAL PRIMARY KEY,
    name       TEXT    UNIQUE NOT NULL,
    color      TEXT    DEFAULT '#6366f1',
    created_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS bookmark_labels (
    bookmark_id BIGINT NOT NULL REFERENCES bookmarks(id) ON DELETE CASCADE,
    label_id    BIGINT NOT NULL REFERENCES labels(id)    ON DELETE CASCADE,
    PRIMARY KEY (bookmark_id, label_id)
);

-- ── Embedding tracking ─────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS embedded_uploads (
    upload_id   TEXT NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
    model_id    TEXT NOT NULL,
    embedded_at TEXT NOT NULL,
    PRIMARY KEY (upload_id, model_id)
);

-- ── Settings ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ── Qualitative coding ─────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS code_categories (
    id         BIGSERIAL PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    color      TEXT NOT NULL DEFAULT '#94a3b8',
    parent_id  BIGINT REFERENCES code_categories(id) ON DELETE SET NULL,  -- self-ref
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS codes (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    color       TEXT DEFAULT '#6366f1',
    description TEXT DEFAULT '',
    category_id BIGINT REFERENCES code_categories(id) ON DELETE SET NULL,
    created_at  TEXT NOT NULL
);

-- Message-level code assignment (one code per bookmark, optional excerpt)
CREATE TABLE IF NOT EXISTS bookmark_codes (
    bookmark_id      BIGINT NOT NULL REFERENCES bookmarks(id) ON DELETE CASCADE,
    code_id          BIGINT NOT NULL REFERENCES codes(id)     ON DELETE CASCADE,
    highlighted_text TEXT,
    PRIMARY KEY (bookmark_id, code_id)
);

-- Passage-level highlights (multiple excerpts per bookmark+code pair)
CREATE TABLE IF NOT EXISTS bookmark_code_highlights (
    id               BIGSERIAL PRIMARY KEY,
    bookmark_id      BIGINT NOT NULL REFERENCES bookmarks(id) ON DELETE CASCADE,
    code_id          BIGINT NOT NULL REFERENCES codes(id)     ON DELETE CASCADE,
    highlighted_text TEXT NOT NULL,
    created_at       TEXT NOT NULL
);

-- ── Session store (auto-managed by tower-sessions-sqlx-store) ──────────────
-- Table: tower_sessions — created by PostgresStore::migrate()
```

**Key design decisions:**

- `row_index` stores the original 0-based CSV row position so context windows reconstruct exact conversation order independent of timestamp precision or ties.
- `search_vector` is a GIN-indexed `tsvector` built from both `content` and `username` using the `simple` dictionary (no language-specific stemming, pure token matching). This means queries match on original word forms including usernames.
- `word_count` is denormalised at insert time to avoid recomputing in every filter query.
- `LOWER(username)` unique index enforces case-insensitive username uniqueness without requiring a normalised column.
- `bookmark_codes` and `bookmark_code_highlights` serve different granularities: the former tracks which code applies to a whole bookmarked message (with an optional short excerpt), the latter supports any number of passage-level highlights per `(bookmark, code)` pair.

---

## 3. Vector Store Layer (Qdrant)

**Files:** `backend/src/routes/uploads.rs`, `backend/src/routes/search.rs`, `backend/src/routes/chat.rs`

All vector operations target the Qdrant REST API at `QDRANT_URL`. The collection is created on first embed if it does not exist.

**Collection parameters:**

| Parameter | Value |
|---|---|
| Vector dimension | 1536 |
| Distance metric | Cosine |
| Point ID format | UUID v5 (derived from Discord Snowflake ID, namespace `NAMESPACE_OID`) |
| Payload fields | `msg_uuid` (string), `upload_id` (string), `document` (string) |

**UUID v5 derivation:** Discord message IDs are 64-bit Snowflake integers stored as text strings. They are converted to stable Qdrant point IDs via `uuid::Uuid::new_v5(&NAMESPACE_OID, id.as_bytes())`. This ensures that re-embedding the same Discord message always produces the same Qdrant point ID, enabling idempotent upserts.

**HTTP client:** `reqwest::Client` with `Authorization: api-key <QDRANT_API_KEY>` header when the key is set.

**Key operations used:**

| Operation | Qdrant endpoint | Notes |
|---|---|---|
| Create collection | `PUT /collections/{name}` | Called if collection does not exist before embedding |
| Upsert vectors | `PUT /collections/{name}/points` | Batch upsert during embedding |
| Scroll (existence check) | `POST /collections/{name}/points/scroll` | 10,000 points per page, `with_payload: false, with_vector: false` |
| Search (ANN) | `POST /collections/{name}/points/search` | Returns `{id, score, payload}` |
| Get by IDs | `POST /collections/{name}/points` | `with_payload: true, with_vector: true` — fetches stored embeddings |
| Delete by filter | `POST /collections/{name}/points/delete` | Filter: `{must: [{key: "upload_id", match: {value: id}}]}` |
| Collection info | `GET /collections/{name}` | `result.vectors_count` used for stats |

---

## 4. Embedding Pipeline

**File:** `backend/src/routes/uploads.rs`

### Upload-time embedding (inline SSE stream)

`POST /api/upload` is an Axum handler returning `Sse<impl Stream<Item = ...>>`. The CSV parsing, DB inserts, and embedding all happen inside a single `async_stream::stream!` block:

**Phase 1 — Database insert:**

1. Multipart body is parsed; the CSV file field is extracted.
2. CSV headers are normalised: stripped, lowercased, spaces → underscores. Alias mappings handle `"message id"` → `"msg_uuid"` etc.
3. Required columns (`author_id`, `username`, `date`, `content`) are validated. Missing required column → immediate `{type: "error"}` SSE and stream close.
4. Rows are collected and inserted in batches of **500**:
   ```sql
   INSERT INTO messages (msg_uuid, author_id, username, date, content,
     attachments, reactions, is_suno_team, week, month, upload_id, row_index,
     word_count)
   VALUES ($1, $2, ...) ON CONFLICT (msg_uuid) DO NOTHING
   ```
5. The database trigger fires on each inserted row, populating `search_vector`.
6. `is_suno_team` cross-upload backfill: any username already present in `messages` with `is_suno_team = 'true'` causes all new messages from that username to be inserted with `is_suno_team = 'true'` as well.
7. SSE event after each batch:
   ```json
   {"type": "progress", "inserted": N, "skipped": S, "total": T}
   ```

**Phase 2 — Embed:**

1. All messages for this upload are fetched from PostgreSQL.
2. A Qdrant scroll over the collection (filtered by `upload_id`, 10,000 per page) collects all existing point IDs. UUIDs already present are marked as `already_embedded`. This is the **resumability check** — re-running an upload or starting a partially-completed embed will not re-embed already-done messages.
3. SSE event: `{"type": "embed_start", "total": T, "already_embedded": A, "model": "text-embedding-3-small"}`
4. New messages (UUIDs not already in Qdrant) are split into batches of **500**.
5. A `tokio::sync::Semaphore` with **3 permits** limits concurrency to 3 simultaneous OpenAI requests.
6. Each batch calls `POST https://api.openai.com/v1/embeddings` with:
   ```json
   {"model": "text-embedding-3-small", "input": ["text1", "text2", ...]}
   ```
   HTTP 429 responses are retried with exponential back-off: 2s, 4s, 8s. The `Retry-After` response header is respected if present.
7. Each batch's 1536-dimensional vectors are immediately upserted to Qdrant with the corresponding `msg_uuid`, `upload_id`, and `document` payload.
8. SSE event after each batch: `{"type": "embed_progress", "embedded": N, "total": T}`
9. On full completion: `{"type": "done", "upload_id": "...", "total_inserted": N, "embedded": E, ...}`
10. `embedded_uploads` is updated: `INSERT INTO embedded_uploads (upload_id, model_id, embedded_at) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING`.

**Constants:**

| Constant | Value |
|---|---|
| CSV insert batch size | 500 rows |
| OpenAI embed batch size | 500 texts |
| Concurrent embed requests | 3 (Semaphore) |
| Qdrant scroll page size | 10,000 points |
| Retry delays (429) | 2s → 4s → 8s |
| Upload body limit | 200 MB |

### Re-embed (SSE)

`POST /api/uploads/:id/reembed` is structurally identical to Phase 2 above, operating on an existing upload. The resumability check always runs first so the operation is idempotent.

---

## 5. Keyword Search (PostgreSQL FTS)

**File:** `backend/src/routes/search.rs` — `search_keyword()`

### tsvector full-text search

The `messages.search_vector` column is a `tsvector` built from `content || ' ' || username` using the `simple` text search dictionary. The `simple` dictionary performs only case-folding and no stemming, meaning searches match on the exact token forms present in the text.

**Query construction** — controlled by the `match_type` request parameter:

| `match_type` | tsquery expression | Behaviour |
|---|---|---|
| `fuzzy` (default) | `to_tsquery('simple', 'tok1:* & tok2:*')` | Prefix AND — matches messages containing all tokens as prefixes |
| `exact` | `phraseto_tsquery('simple', 'tok1 tok2')` | Adjacent phrase — tokens must appear in order and adjacent |
| `any_word` | `to_tsquery('simple', 'tok1 | tok2')` | OR — any of the supplied tokens |

All tokens are lower-cased before query construction. FTS special characters (`'`, `:`, `&`, `|`, `!`, `<`, `>`) are escaped to prevent injection into the `tsquery` expression.

**SQL query (parameterised):**

```sql
SELECT m.id, m.msg_uuid, m.author_id, m.username, m.date, m.content,
       m.attachments, m.reactions, m.is_suno_team, m.upload_id, m.row_index,
       (b.id IS NOT NULL) AS is_bookmark
FROM messages m
LEFT JOIN bookmarks b ON m.id = b.msg_id AND b.user_id = $user_id
WHERE m.search_vector @@ to_tsquery('simple', $tsquery)
  [AND m.upload_id = ANY($upload_ids)]
  [AND LOWER(m.username) LIKE LOWER($username_pattern)]
  [AND m.date >= $date_from]
  [AND m.date <= $date_to]
  [AND m.word_count >= $min_words]
  [AND m.is_suno_team IN ('true','1')]    -- suno_team = "only"
  [AND m.is_suno_team NOT IN ('true','1')]  -- suno_team = "exclude"
ORDER BY m.date ASC, m.row_index ASC
LIMIT $limit
```

**GIN index:** The `idx_messages_search_vector` GIN index on `search_vector` allows the `@@` operator to run in O(log n + k) time where k is the number of matching rows, rather than a full table scan.

**Fallback:** If the `search_vector` column contains NULLs (startup backfill still running) or if the `match_type` cannot be compiled to a valid `tsquery`, the query falls back to:

```sql
WHERE content ILIKE '%token1%' AND content ILIKE '%token2%'
```

---

## 6. Semantic Search with HyDE

**File:** `backend/src/routes/search.rs` — `search_semantic()`

HyDE (Hypothetical Document Embeddings) improves retrieval precision by augmenting the raw query embedding with an embedding derived from a hypothetical answer document generated by `gpt-4o-mini`.

### Full algorithm

**Step 1 — HyDE weight selection:**

```
if query.split_whitespace().count() <= 3:
    w = 0.75   # short query: rely heavily on hypothetical document
else:
    w = 0.55   # standard: roughly equal blend
```

**Step 2 — Parallel embedding and document generation:**

```rust
let (raw_emb, hyde_emb) = tokio::join!(
    embed_text(&query, &openai_key),
    async {
        let hypo_doc = generate_hypothetical_doc(&query, &openai_key).await;
        embed_text(&hypo_doc, &openai_key).await
    }
);
```

Hypothetical document prompt sent to `gpt-4o-mini`:

```
Generate 2-3 hypothetical Discord messages (about 80 words total) from the
Suno AI Discord server that would be relevant to: "<query>".
Write in the style of Discord messages, be concise and specific.
```

Model: `gpt-4o-mini`. Temperature: 0.7. Max tokens: 200.

**Step 3 — Embedding blend:**

Both embeddings are L2-normalised before blending:

```
norm_raw  = raw_emb  / ||raw_emb||
norm_hyde = hyde_emb / ||hyde_emb||
blended   = norm_raw * (1 - w) + norm_hyde * w
blended   = blended  / ||blended||            ← renormalize
```

**Step 4 — Qdrant ANN search:**

```
fetch_n = min(limit * 8, 800)

POST /collections/{name}/points/search
{
  "vector": blended,
  "limit": fetch_n,
  "with_payload": true,
  "with_vector": false
}
```

Response: `[{id, score, payload: {msg_uuid, upload_id, document}}, ...]`

**Step 5 — Relative similarity threshold:**

```
best_sim = results[0].score          (highest similarity)
min_sim  = max(0.15, best_sim * 0.50)
results  = [r for r in results if r.score >= min_sim]
```

This adaptive threshold keeps all results within 50% of the best score, with an absolute floor of 0.15. It avoids returning low-quality results when the best score is already mediocre.

**Step 6 — PostgreSQL fetch:**

```sql
SELECT m.*, (b.id IS NOT NULL) AS is_bookmark
FROM messages m
LEFT JOIN bookmarks b ON m.id = b.msg_id AND b.user_id = $user_id
WHERE m.msg_uuid = ANY($matching_uuids)
  [AND upload_id, username, date, word_count, suno_team filters]
```

**Step 7 — Sort and cap:**

Sorted by `sort_by` parameter:
- `date_asc` — chronological
- `date_desc` — reverse chronological
- *(default)* — Qdrant similarity order (best first)

Each result carries a `similarity_score` field (the raw Qdrant cosine similarity).

---

## 7. In-Results Semantic Filter

**File:** `backend/src/routes/context_route.rs` — `filter_semantic()`

Reranks an already-retrieved set of message IDs by semantic relevance without a new ANN query.

### Query preprocessing

```
if query starts with {what, how, why, when, where, who, which, is, are, do, does, can}
   or query ends with '?':
    strip stop words and question words from query
    threshold = 0.20
else:
    use query as-is
    threshold = 0.30
```

The lower threshold for questions compensates for the semantic dilution caused by function words in the question phrasing.

### Scoring

1. Fetch stored embeddings from Qdrant for all provided UUIDs:
   ```
   POST /collections/{name}/points
   {"ids": [...], "with_payload": false, "with_vector": true}
   ```
2. Embed the preprocessed query with `text-embedding-3-small`.
3. For each `(uuid, stored_embedding)` pair, compute cosine similarity:
   ```
   sim = dot(normalize(query_emb), normalize(stored_emb))
   ```
   (OpenAI embeddings are unit-normalised by the API; normalisation is verified before the dot product.)
4. Return only results where `sim >= threshold`, sorted by score descending.

---

## 8. Context Window

**File:** `backend/src/routes/context_route.rs` — `get_context()`

```sql
SELECT m.*
FROM messages m
WHERE m.upload_id = $1
  AND m.row_index BETWEEN GREATEST(0, $target_row - $before)
                      AND $target_row + $after
ORDER BY m.row_index ASC
```

`before` and `after` are clamped to [0, 200]. `row_index` stores the original 0-based CSV row position, guaranteeing that context is reconstructed in exact source order regardless of timestamp precision or duplicates.

The target message (`id = message_id`) is tagged `is_target: true` in the response list.

---

## 9. Summarize Results (HDBSCAN Pipeline)

**File:** `backend/src/routes/chat.rs` — `summarize_results()`

This endpoint receives the messages currently displayed in the browser (passed in the request body) and summarises them using HDBSCAN clustering for evidence selection.

### Request schema

```rust
struct SummarizeResultsRequest {
    messages: Vec<SummarizeMsg>,  // {username, date, content, msg_uuid?}
    query:    Option<String>,     // optional retrieval/summary guidance
    model:    Option<String>,     // default "gpt-4o"
    retrieval_mode: Option<String>,  // "cluster" (default) or "all"
}
```

### Pipeline (streaming SSE)

**Step 1 — Content deduplication:**

Messages with identical lowercased content are removed before clustering. SSE log event emitted.

**Step 2 — Retrieval mode branch:**

**`retrieval_mode = "cluster"` (default):**

1. Collect all `msg_uuid` values and fetch their stored embeddings from Qdrant:
   ```
   POST /collections/{name}/points
   {"ids": [uuid1, uuid2, ...], "with_vector": true}
   ```
2. If fewer than 3 embeddings are returned (messages not yet embedded), fall back to word-count sampling: take up to 200 messages sorted by `word_count` descending.
3. Run HDBSCAN on the embedding matrix (§9.1).
4. For each cluster, sample the `n_closest=5` and `n_furthest=5` points from the cluster centroid (or all members if cluster size ≤ 10). For noise points (label = -1), take up to 5 total.
5. Assemble all sampled messages, sort chronologically by `date`, cap at 120.

**`retrieval_mode = "all"`:**

Skip clustering. Use all messages, capped at 200.

**Step 3 — LLM generation:**

Evidence messages are formatted as `[username | date]: content` lines. Sent to the specified model with `prompts::SUMMARIZE`. Response streams as SSE `{type: "chunk", content: "..."}` events.

---

### 9.1 HDBSCAN Implementation

**File:** `backend/src/hdbscan.rs`

Pure-Rust implementation. Input: `&[Vec<f32>]` (n × 1536 embeddings). Output: `Vec<i32>` (cluster labels, 0-based; -1 = noise).

**Parameters:**

```rust
let min_cluster_size: usize = 3.min(n / 4).max(2);
// e.g. n=20 → mcs=3; n=40 → mcs=3; n=100 → mcs=3 (min(3,25).max(2))
// ensures at least 2, at most n/4, absolute cap of 3
```

**Algorithm:**

**Stage 1 — Pairwise cosine distance matrix** `D[i][j]` for all pairs:

```
cosine_distance(a, b) = 1.0 - dot(a, b) / (||a|| * ||b||)
```

Time complexity: O(n² · d) where d = 1536.

**Stage 2 — Core distances:**

For each point `i`, `core_dist[i]` = `D[i][k]` where `k` is the index of the `min_cluster_size`-th nearest neighbour of `i`. Core distances prevent low-density regions from being absorbed into clusters.

**Stage 3 — Mutual reachability distance:**

```
mrd(i, j) = max(core_dist[i], core_dist[j], D[i][j])
```

**Stage 4 — Minimum spanning tree:**

Prim's algorithm on the complete graph with `mrd` edge weights. Produces `n-1` edges. Time complexity: O(n²).

**Stage 5 — Single-linkage dendrogram:**

MST edges sorted ascending by weight. Union-find processes them in order. Each merge event `(node_A, node_B, weight)` creates a dendrogram node with `lambda = 1.0 / weight` (higher lambda = tighter merge).

**Stage 6 — Condensed tree construction:**

Depth-first traversal of the dendrogram. At each merge:
- If **both** sub-trees have ≥ `min_cluster_size` members: real cluster split → create two condensed cluster nodes.
- If one or both sub-trees are smaller: the small sub-tree's points become noise at this level; their `lambda` values contribute to the parent cluster's stability score.

**Stage 7 — Excess-of-mass cluster selection:**

For each condensed cluster `C`:
```
stability(C) = Σ_{p ∈ C} (lambda(p) - lambda_birth(C))
```

Where `lambda(p)` is the lambda value at which point `p` falls out of `C`, and `lambda_birth(C)` is the lambda at which `C` was created.

Bottom-up decision: keep cluster `C` if `stability(C) ≥ Σ stability(children)`. Otherwise propagate children upward.

**Stage 8 — Label assignment:**

Each point walks up the condensed tree until it reaches a selected cluster; that cluster's index (0-based) is the point's label. Points not reachable from any selected cluster receive label -1 (noise).

**Utility functions (used by sampling in §9):**

```rust
fn centroid(embeddings: &[Vec<f32>]) -> Vec<f32>
// Element-wise mean of all embedding vectors in the set

fn closest_and_furthest(embeddings: &[Vec<f32>], centroid: &[f32], n: usize)
    -> (Vec<usize>, Vec<usize>)
// Sorts member indices by L2 distance from centroid
// Returns first n (closest) and last n (furthest) indices
```

---

## 10. Hybrid Summary

**File:** `backend/src/routes/chat.rs` — `summarize()`, `summarize_followup()`

The Hybrid Summary runs a full retrieval → cluster → LLM pipeline directly on the database corpus.

### Pipeline stages

**Stage 1 — Metadata filter:**

```sql
SELECT msg_uuid, username, date, content
FROM messages
WHERE [LOWER(username) LIKE LOWER($username)]
  [AND upload_id = ANY($upload_ids)]
  [AND date >= $date_from AND date <= $date_to]
  [AND is_suno_team filter]
  [AND word_count >= $min_words]
ORDER BY date ASC, row_index ASC
```

Result: `filtered_map: HashMap<String, Row>` keyed by `msg_uuid`.

**Stage 2 — Semantic retrieval:**

The user's custom prompt is the retrieval query when provided; otherwise the generic coverage query is used:
```
"key discussions, important insights, notable feedback, use cases, significant events"
```

The query is embedded with `text-embedding-3-small` (**not** HyDE — the corpus is pre-filtered so coverage matters more than precision here). Qdrant is queried:

```
overfetch_n = min(total_qdrant_count, max(n_filtered * 5, 2000))
POST /collections/{name}/points/search {vector: ..., limit: overfetch_n}
```

Results are intersected with `filtered_map` by `msg_uuid`. Each intersecting result gets `score = similarity`.

**Stage 3 — Adaptive threshold:**

```
threshold = 30th percentile of scores
keep: score >= threshold  (top 70% retained)
ensure: at least 15 candidates survive (force-keep top 15 if fewer)
```

Embeddings are then fetched for the surviving UUIDs.

**Stage 4 — Deduplication:**

Near-duplicate removal using pairwise cosine similarity of stored embeddings:
1. Drop messages shorter than 10 characters.
2. Compute normalised embedding matrix: `E_norm = E / ||E||`.
3. Pairwise similarity: `S = E_norm @ E_norm.T` (all values in [-1, 1]).
4. Process in ranking order: for each kept message `i`, mark all later messages `j` with `S[i,j] ≥ 0.97` as duplicates.

Threshold 0.97 targets near-identical messages (copy-pastes, bot reposts) while preserving paraphrases.

**Stage 5 — HDBSCAN clustering:**

Same algorithm as §9.1 with the same `min_cluster_size` formula.

**Stage 6 — Per-cluster sampling:**

For each cluster: 5 closest + 5 furthest from centroid. Clusters with ≤ 10 members contribute all members.

**Stage 7 — Assemble:**

All sampled messages are combined, sorted chronologically by `date`, and capped at **120 messages** (`max_evidence`).

**Stage 8 — Transparency log + LLM generation:**

Before the LLM token stream begins, SSE log events are emitted for each pipeline step:

```json
{"type": "log", "step": "filter",    "count": N}
{"type": "log", "step": "retrieval", "count": R, "overfetch": O}
{"type": "log", "step": "threshold", "kept": K, "dropped": D}
{"type": "log", "step": "dedup",     "kept": K, "removed": R}
{"type": "log", "step": "cluster",   "algorithm": "HDBSCAN", "clusters": C}
{"type": "log", "step": "evidence",  "count": E}
```

These appear in the browser UI as a visible "research pipeline" trace. LLM tokens then stream as `{"type": "chunk", "content": "..."}`.

### Fallback chain

```
Stage 2 returns ≥ 10 candidates?
├── YES → Stages 3–7 (dedup/cluster/sample)
└── NO  → Fetch embeddings for all rows in filtered_map (up to 3000 random sample)
          ≥ 10 rows have stored embeddings?
          ├── YES → Stages 4–7 on those embeddings (no score threshold)
          └── NO  → Send all filtered_map rows directly to LLM (no clustering)
```

### Follow-up (`POST /api/summarize/followup`)

- The follow-up question is the retrieval query (not the original prompt).
- Overfetch: `max(n_filtered * 4, 1000)`.
- Evidence cap: **80** (not 120).
- The initial summary (first assistant turn in `history`) is embedded in the system prompt.
- Up to 20 prior Q&A turns are appended to the message list.

---

## 11. User Profile Analysis

**File:** `backend/src/routes/chat.rs` — `user_profile()`, `user_profile_followup()`

Functionally identical to §10, with these differences:

**Filter:** Exact match on `LOWER(username) = LOWER($1)` (not `ILIKE` partial match).

**Fallback retrieval query:**
```
"attitude, opinions, concerns, feedback, and persona of {username} regarding Suno AI"
```

**Entry/exit dates:** Extracted from `db_rows.first()` and `db_rows.last()` and injected into the LLM prompt template.

**LLM prompt instructs the model to output:**
1. Entry date (first message) and exit date (last message)
2. Persona description — role, communication style, expertise signals
3. Evolution of attitude over time — chronological narrative with identified inflection points
4. Key recurring topics and concerns
5. Representative verbatim quotes (at least 3, with dates)
6. Summary assessment

**Follow-up (`POST /api/user-profile/followup`):** The initial profile is included as a system-level context block. Evidence is appended to the user turn rather than the system prompt.

---

## 12. Bookmarks and Labels

**File:** `backend/src/routes/bookmarks.rs`

### Bookmark creation (`POST /api/bookmarks`)

1. Validate `msg_id` exists in `messages`.
2. Check for existing bookmark: `SELECT id FROM bookmarks WHERE msg_id = $1 AND (user_id = $2 OR user_id IS NULL)`. Return `{status: "exists", bookmark_id: N}` without creating a duplicate.
3. Insert: `INSERT INTO bookmarks (msg_id, ctx_before, ctx_after, note, created_at, user_id)`.

### Bookmark listing (`GET /api/bookmarks`)

Two queries, no N+1:

**Query 1:** All bookmarks for the authenticated user joined with messages:
```sql
SELECT b.*, m.*
FROM bookmarks b
JOIN messages m ON b.msg_id = m.id
WHERE b.user_id = $1
ORDER BY b.created_at DESC
```

**Query 2:** All label assignments for those bookmarks:
```sql
SELECT bl.bookmark_id, l.*
FROM bookmark_labels bl
JOIN labels l ON bl.label_id = l.id
WHERE bl.bookmark_id = ANY($bookmark_ids)
```

Labels are grouped into `HashMap<i64, Vec<Label>>` in Rust and merged into each bookmark struct before JSON serialisation.

### Lightweight ID listing (`GET /api/bookmarks/ids`)

```sql
SELECT msg_id FROM bookmarks WHERE user_id = $1
```

Returns only message IDs. Used by the frontend to mark bookmarked message cards without loading full bookmark data on every search.

### Labels

Labels carry `name` (unique), `color` (hex, default `#6366f1`), `created_at`. Assignment uses `INSERT INTO bookmark_labels ... ON CONFLICT (bookmark_id, label_id) DO NOTHING`. `ON DELETE CASCADE` on `bookmark_labels.bookmark_id` automatically removes label assignments when a bookmark is deleted.

---

## 13. Qualitative Coding System

**File:** `backend/src/routes/codes.rs`

### Data model

**Codes** are qualitative tags with `name`, `color`, `description`, and an optional `category_id`. **Code categories** have `name`, `color`, and an optional `parent_id` (self-referential foreign key) enabling a two-level hierarchy. `ON DELETE SET NULL` on `parent_id` means deleting a parent category does not cascade to its children.

**Two coding granularities:**

| Table | Key | Purpose |
|---|---|---|
| `bookmark_codes` | `(bookmark_id, code_id)` | Message-level: one code per bookmark, optional short excerpt |
| `bookmark_code_highlights` | `id` (auto) | Passage-level: multiple excerpts per `(bookmark, code)` pair |

The distinction enables both coarse-grained tagging (e.g. "this message is about frustration") and fine-grained textual coding (e.g. "this specific sentence expresses concern about pricing").

### Key operations

**Assign code to bookmark (`POST /api/bookmark-codes`):**

```sql
INSERT INTO bookmark_codes (bookmark_id, code_id, highlighted_text)
VALUES ($1, $2, $3)
ON CONFLICT (bookmark_id, code_id) DO UPDATE SET highlighted_text = $3
```

**Add passage highlight (`POST /api/bookmarks/:id/highlights`):**

```sql
INSERT INTO bookmark_code_highlights (bookmark_id, code_id, highlighted_text, created_at)
VALUES ($1, $2, $3, $4)
```

No unique constraint — multiple highlights per `(bookmark_id, code_id)` are allowed by design.

**List codes for bookmark (`GET /api/bookmark-codes/:bookmark_id`):**

Single query joining `bookmark_codes → codes → code_categories` and a second query for `bookmark_code_highlights`. Both are merged in Rust.

---

## 14. Authentication and Sessions

**File:** `backend/src/routes/auth.rs`, `backend/src/main.rs`

### Argon2id password hashing

Registration:
```rust
let salt = SaltString::generate(&mut OsRng);       // cryptographically random
let hash = Argon2::default()
    .hash_password(password.as_bytes(), &salt)?
    .to_string();                                    // PHC string format
```

`Argon2::default()` parameters:
| Parameter | Value |
|---|---|
| Variant | Argon2id |
| Memory cost | 19,456 KiB (19 MiB) |
| Iterations | 2 |
| Parallelism | 1 |
| Salt length | 16 bytes (SaltString generates 128-bit) |
| Output length | 32 bytes |

Verification:
```rust
Argon2::default().verify_password(
    password.as_bytes(),
    &PasswordHash::new(&stored_hash)?
)?
```

`verify_password` runs in constant time (no early-exit on mismatch), preventing timing-based side-channel attacks.

### Session lifecycle

**Login (`POST /api/auth/login`):**
```rust
session.insert("user_id", user.id).await?;
```

**Auth check (`AuthUser` extractor):**
```rust
let user_id: i64 = session.get("user_id").await?  // None → 401
    .ok_or(StatusCode::UNAUTHORIZED)?;
sqlx::query_as!(User, "SELECT * FROM users WHERE id = $1", user_id)
    .fetch_one(&pool).await?
```

In `single`/`demo` mode: the extractor bypasses the session lookup entirely and returns a hard-coded `User { id: 1, username: "admin", is_admin: true }`.

**Logout (`POST /api/auth/logout`):**
```rust
session.delete().await?;
```

**Session TTL:** `Expiry::OnInactivity(Duration::seconds(2_592_000))` — 30 days. Each authenticated request that reads the session refreshes the expiry. `tower-sessions-sqlx-store` handles the expiry check and cleanup automatically.

### Mode-based auth

`GET /api/auth/app-mode` returns the current mode (`"single"`, `"multi"`, or `"demo"`). The frontend fetches this on app load to decide whether to show the login UI and whether to read the API key from session vs. localStorage.

---

## 15. Security Model

### API key handling

The OpenAI API key is never stored in the database. It flows as the `X-OpenAI-Key` request header from the browser's `localStorage` (single/demo mode) or from per-user localStorage (multi mode). The backend reads it per-request, uses it for the duration of the handler, and discards it. This means a database breach does not expose API keys.

### SQL injection prevention

All database access uses sqlx parameterised queries (`$1`, `$2`, ...) with Rust type-level binding. The only dynamic SQL construction is:
- `= ANY($1)` with a `Vec<String>` bind — safe (no string interpolation)
- Keyword search token construction — tokens are extracted from user input and passed to `to_tsquery()` via a bind variable, not interpolated into the SQL string

FTS injection (malformed `tsquery` expressions) is mitigated by lower-casing and stripping special characters before token extraction.

### Chat model allowlist

`VALID_CHAT_MODELS` is a compile-time `const &[&str]` slice. Any request specifying an unlisted model ID returns HTTP 400 before any API call is made.

### CORS

The backend CORS layer is fully permissive (`CORSLayer::very_permissive()`). In production, nginx provides origin restriction and the backend's permissive policy avoids conflicts with the reverse proxy headers. In development, this allows the Vite dev server (port 5173) to call the backend (port 8000) without CORS errors.

### Security headers

Applied by the `tower_http` `SetResponseHeaderLayer` (or equivalent) to all responses:

| Header | Value |
|---|---|
| `X-Content-Type-Options` | `nosniff` |
| `X-Frame-Options` | `DENY` |
| `Referrer-Policy` | `strict-origin-when-cross-origin` |

---

## 16. SSE Streaming Pattern

All streaming endpoints (upload, reembed, summarize, summarize-results, user-profile) use Axum's `Sse<impl Stream>` with the `async_stream::stream!` macro:

```rust
async fn my_handler(/* ... */) -> impl IntoResponse {
    let stream = async_stream::stream! {
        // ... work ...
        yield Ok(Event::default().data(serde_json::to_string(&payload)?));
        // ... more work ...
    };
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
}
```

Each SSE frame is formatted as:
```
data: {"type":"...","field":"..."}\n\n
```

The `X-Accel-Buffering: no` response header is set to disable Nginx proxy buffering, ensuring tokens reach the browser immediately rather than being held until the buffer fills.

The frontend consumes SSE streams using the browser's native `EventSource` API (for simple streams) or a custom `fetch`+`ReadableStream` pipeline (for streams requiring `POST` bodies and custom headers). TanStack Query's mutation state manages the loading/error UI, while a Zustand store accumulates streaming content for real-time display.
