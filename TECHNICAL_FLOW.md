# Technical Flow — Suno Discord Analysis Platform

This document provides implementation-level descriptions of every feature for academic transparency. It traces data from the HTTP request through each layer to the final response.

---

## Table of Contents

1. [Application Startup](#1-application-startup)
2. [Database Schema](#2-database-schema)
3. [Vector Store Layer](#3-vector-store-layer)
4. [Embedding Pipeline](#4-embedding-pipeline)
5. [Keyword Search (FTS5)](#5-keyword-search-fts5)
6. [Semantic Search](#6-semantic-search)
7. [In-Results Semantic Filter](#7-in-results-semantic-filter)
8. [Context Window](#8-context-window)
9. [RAG Chat](#9-rag-chat)
10. [Hybrid Summary Pipeline](#10-hybrid-summary-pipeline)
11. [Summarize Results Pipeline](#11-summarize-results-pipeline)
12. [User Profile Analysis](#12-user-profile-analysis)
13. [Bookmarks and Labels](#13-bookmarks-and-labels)
14. [Security Model](#14-security-model)
15. [Caching Strategy](#15-caching-strategy)

---

## 1. Application Startup

**File:** `app.py` — `lifespan()` async context manager

On startup the following steps run in parallel using `asyncio.gather`:

1. **Vector store initialisation** — `init_vector_store()` (blocking, run in thread executor):
   - Reads `VECTOR_DB` from environment.
   - `qdrant`: connects to `QDRANT_URL`; creates collections defined in `EMBEDDING_MODELS` if they do not exist; wraps each in `QdrantCollectionWrapper`; logs vector counts.
   - `chroma_persistent`: opens a `PersistentClient` at `CHROMA_PATH`; calls `get_or_create_collection` with `hnsw:space=cosine`.
   - `chroma_http`: opens an `HttpClient`; verifies connectivity with `heartbeat()`; creates collections.
   - Returns `{model_id: wrapper}` stored in `state.vector_collections`.

2. **Database initialisation** — `init_db()` (blocking, run in thread executor):
   - Calls `get_db()` which sets SQLite PRAGMAs: WAL journal mode, NORMAL synchronous, foreign keys ON, 32 MB page cache, in-memory temp store, 128 MB memory-mapped I/O.
   - Runs `CREATE TABLE IF NOT EXISTS` for: `messages`, `uploads`, `settings`, `bookmarks`, `bookmark_labels`, `labels`, `embedded_uploads`.
   - Creates indexes: `idx_username`, `idx_upload_row`, `idx_date`, `idx_msg_uuid`, `idx_suno_team`, `idx_date_suno`, `idx_bookmark_msg`, `idx_embedded_model`.
   - Creates the FTS5 virtual table `messages_fts` (content mirror of `messages.content`).
   - Creates three triggers: `tg_messages_ai` (after insert), `tg_messages_ad` (after delete), `tg_messages_au` (after update on content) to keep the FTS index synchronised.
   - If `messages_fts` row count is 0 but `messages` has rows, runs `INSERT INTO messages_fts VALUES ('rebuild')` to rebuild the index from existing data (migration safety net).

3. **API key restoration** — if `OPENAI_API_KEY` is set in the environment, both a synchronous `OpenAI` client and an asynchronous `AsyncOpenAI` client are created and stored in `state`.

4. **Embedding model restoration** — `get_setting("embedding_model")` reads the last-saved model ID from SQLite and applies it to `state.current_embedding_model`.

5. **Migration** — if `embedded_uploads` is empty but `uploads` has rows, a background thread scans each upload's messages against the vector store and backfills `embedded_uploads` records. This runs in `loop.run_in_executor` so it never blocks startup or requests.

---

## 2. Database Schema

**File:** `database.py`

```sql
messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    msg_uuid    TEXT    UNIQUE NOT NULL,   -- UUID assigned at upload time
    author_id   TEXT,
    username    TEXT,
    date        TEXT,                      -- ISO-8601 string from CSV
    content     TEXT,
    attachments TEXT,
    reactions   TEXT,
    is_suno_team TEXT,                     -- "true"/"1" or empty
    week        TEXT,
    month       TEXT,
    upload_id   TEXT    NOT NULL,          -- FK to uploads.id
    row_index   INTEGER NOT NULL           -- original 0-based row position in CSV
)

uploads (
    id          TEXT    PRIMARY KEY,       -- UUID assigned at upload time
    filename    TEXT    NOT NULL,
    row_count   INTEGER NOT NULL,
    upload_time TEXT    NOT NULL
)

bookmarks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    msg_id     INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    ctx_before INTEGER NOT NULL DEFAULT 5,
    ctx_after  INTEGER NOT NULL DEFAULT 5,
    note       TEXT    DEFAULT '',
    created_at TEXT    NOT NULL
)

labels (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL UNIQUE,
    color      TEXT    NOT NULL DEFAULT '#6366f1',
    created_at TEXT    NOT NULL
)

bookmark_labels (
    bookmark_id INTEGER NOT NULL REFERENCES bookmarks(id) ON DELETE CASCADE,
    label_id    INTEGER NOT NULL REFERENCES labels(id)    ON DELETE CASCADE,
    PRIMARY KEY (bookmark_id, label_id)
)

embedded_uploads (
    upload_id   TEXT NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
    model_id    TEXT NOT NULL,
    embedded_at TEXT NOT NULL,
    PRIMARY KEY (upload_id, model_id)
)

settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
)

-- FTS5 virtual table (content mirror of messages.content)
messages_fts USING fts5(content, content='messages', content_rowid='id')
```

**Indexes:** `username COLLATE NOCASE`, `(upload_id, row_index)`, `date`, `msg_uuid`, `is_suno_team`, `(date, is_suno_team)`, `bookmarks.msg_id`, `embedded_uploads.model_id`.

**`row_index`** is the original 0-based position of the row in the uploaded CSV. It is used to reconstruct conversation context windows without relying on timestamp ordering, which can be non-unique or imprecise.

---

## 3. Vector Store Layer

**File:** `vector_store.py`

Both `QdrantCollectionWrapper` and `ChromaCollectionWrapper` expose the same five-method interface:

| Method | Signature | Description |
|---|---|---|
| `count()` | `→ int` | Returns the number of stored vectors |
| `get()` | `(ids?, where?, limit?, include?) → dict` | Fetch points by ID list or filter; optionally return embeddings |
| `upsert()` | `(embeddings, documents, ids, metadatas)` | Insert or update points |
| `query()` | `(query_embeddings, n_results) → dict` | ANN search; returns `{"ids": [[...]], "distances": [[...]]}` |
| `delete()` | `(ids?, where?)` | Delete by ID list or payload filter |

**Qdrant fallback chain for `query()`:**

The client and server may be at different versions. `QdrantCollectionWrapper.query()` tries three strategies in order:

1. `client.query_points()` — modern API (qdrant-client ≥ 1.7, Qdrant server ≥ 1.7). If the server returns HTTP 404, falls through.
2. `client.search()` — legacy API (qdrant-client < 1.7). Used if `query_points` is absent.
3. Direct REST `POST /collections/{name}/points/search` — bypasses the Python client entirely; works on all server versions. Used when `query_points` gets a 404 and `QDRANT_URL` is set.

**ChromaDB SQLite safety:** ChromaDB's persistent backend uses SQLite internally, which raises "too many SQL variables" when bind-variable count exceeds ~999. `ChromaCollectionWrapper` caps `n_results` at 900 in `query()` and chunks large ID lists in `get()` (batches of 900), merging results transparently.

**`_where_to_filter()`:** Converts a Chroma-style `{"field": {"$eq": value}}` dict to a Qdrant `Filter` with `FieldCondition` / `MatchValue` objects.

**Distance convention:** Both backends are configured for cosine distance. The wrappers return `distance = 1 − cosine_similarity`, so smaller distance = higher relevance. The search endpoints convert back: `similarity_score = round(1.0 − distance, 4)`.

---

## 4. Embedding Pipeline

**File:** `embeddings.py`, `routers/uploads.py`

### Embedding function

```python
async def embed_texts_async(texts: list[str]) -> list[list[float]]:
```

Uses `AsyncOpenAI.embeddings.create()` with model `text-embedding-3-small`. Texts are truncated to 8191 characters each (OpenAI token limit approximation). Returns a list of 1536-dimensional float vectors.

### Upload-time embedding (synchronous with streaming progress)

During `POST /api/upload`, embedding happens inline within the SSE generator:

1. All non-empty `content` values are collected with their UUIDs and metadata.
2. They are split into batches of `EMBED_BATCH_SIZE` (default 2048).
3. Batches are processed with `EMBED_CONCURRENCY` (default 10) concurrent `asyncio.gather` calls.
4. After each concurrency chunk, the SSE stream emits a progress message: `"Embedded N/M messages"`.
5. On completion, `mark_upload_embedded(upload_id, model_id)` writes to `embedded_uploads`.

### Re-embed background job

`POST /api/uploads/{upload_id}/reembed` returns immediately with a `job_id`. The actual work runs in `asyncio.create_task(run_embed_job(...))`.

**Resumability check (phase: "checking"):**

- All UUIDs for the upload are split into batches of 500.
- Up to 4 concurrent async tasks call `col.get(ids=batch, include=[])` via `run_in_executor` on `state.vector_executor`.
- UUIDs already present in the vector store are collected into `already_done`.
- Only UUIDs not in `already_done` are included in the embedding work list.

**Embedding (phase: "embedding"):**

- Work is split into `EMBED_BATCH_SIZE` batches.
- Up to `EMBED_CONCURRENCY` batches run concurrently, each acquiring a `Semaphore` slot.
- Each batch: calls `embed_texts_async()`, then upserts to the vector store via `run_in_executor`.
- Per-batch errors are logged but do not abort the job; they are stored in `job["batch_errors"]`.

**Job state** (`state.embed_jobs[job_id]`):

```python
{
    "status":        "running" | "completed" | "failed",
    "phase":         "checking" | "embedding",
    "upload_id":     str,
    "model":         str,       # human-readable label
    "embedded":      int,       # messages successfully embedded so far
    "total":         int,       # messages to embed (after resumability check)
    "skipped":       int,       # already in vector store
    "current_batch": int,
    "batch_errors":  list,
    "error":         str | None,
    "traceback":     str | None,
}
```

---

## 5. Keyword Search (FTS5)

**File:** `sql_helpers.py` — `keyword_search()`, `_build_fts_query()`

### FTS5 query compilation

```python
def _build_fts_query(keyword: str) -> str:
```

1. FTS5 syntax characters (`" ' * ^ ( ) [ ] { } ; : \`) are replaced with spaces.
2. The cleaned string is tokenised on whitespace.
3. Single token → prefix match: `"token"*`
4. Multiple tokens → phrase match: `"token1 token2 ..."`

Prefix match allows partial-word hits (e.g. `feat` matches `feature`, `features`). Phrase match requires the exact sequence of words in adjacent positions.

### Two-phase retrieval

**Phase 1 — FTS5 candidate fetch:**

```sql
SELECT rowid FROM messages_fts WHERE messages_fts MATCH ? LIMIT ?
```

The limit is `limit × 20` to ensure enough candidates survive the metadata filters in phase 2. FTS5 returns `rowid` values corresponding to `messages.id`.

**Phase 2 — Metadata filter and full-row fetch:**

```sql
SELECT * FROM messages
WHERE id IN (?,?,...)
  [AND LOWER(username) LIKE LOWER(?)]
  [AND upload_id IN (?,...)]
  [AND LOWER(is_suno_team) IN ('true','1')]
  [AND substr(date,1,10) >= ?]
  [AND substr(date,1,10) <= ?]
  [AND (length(trim(content))-length(replace(trim(content),' ',''))+1) >= ?]
ORDER BY date, row_index
LIMIT ?
```

**Word-count filter:** `(length(trim(content)) - length(replace(trim(content), ' ', '')) + 1) >= min_words` counts words as space-count + 1. This is a heuristic that avoids the need for a UDF.

**Fallback:** If the FTS query raises any exception (malformed expression, index not ready), the function logs a warning and retries with `LOWER(content) LIKE LOWER('%keyword%')`.

---

## 6. Semantic Search

**File:** `routers/search.py` — `search_semantic()`

```
Request: query string + filters + n_results
         │
         ▼
embed_texts_async([query])   → 1 × 1536 float vector
         │
         ▼
col.query(query_embeddings=[query_emb], n_results=fetch_n)
  fetch_n = min(n_results × 4, total) if filters active
          = n_results                 if no filters
         │
         ▼
results = {"ids": [[uuid, ...]], "distances": [[dist, ...]]}
         │
         ▼
SELECT * FROM messages WHERE msg_uuid IN (?)
         │
         ▼
Post-filter in Python:
  - upload_id in uid_list
  - username substring match
  - date_in_range(date, date_from, date_to)
  - suno_team membership
  - word count
         │
         ▼
Attach similarity_score = round(1.0 − distance, 4)
Keep first n_results after filtering
```

The **4× overfetch factor** when filters are active compensates for the fact that ANN results are not aware of metadata filters: fetching 4 times as many candidates statistically ensures enough survive post-filtering to fill the requested `n_results` quota.

---

## 7. In-Results Semantic Filter

**File:** `routers/context.py` — `filter_semantic()`

This endpoint operates on an already-retrieved set of message IDs (from any search mode) and re-ranks them by semantic relevance without performing a new ANN query.

### Query preprocessing

```python
def _prepare_filter_query(raw: str) -> tuple[str, float]:
```

1. If the raw query matches `^(what|how|why|...)` or ends with `?`, it is classified as a question.
2. For questions: all stop words and question-word tokens are removed; the core semantic tokens are joined and embedded. Threshold = 0.20.
3. For non-question queries: embedded as-is. Threshold = 0.30.

The lower threshold for questions compensates for the fact that the question's semantic embedding is anchored to the core concept, not diluted by function words.

### Scoring

1. `col.get(ids=uuids, include=["embeddings"])` fetches stored vectors directly by ID (no ANN search).
2. For each `(uuid, stored_embedding)` pair, cosine similarity is computed with NumPy: `sim = dot(q_norm, e_norm)` (the vectors as stored are not normalised; a full dot product is used, which equals cosine similarity when both vectors are unit-norm — the OpenAI embedding API returns unit-norm vectors by default).
3. Results with `sim >= threshold` are returned sorted by score descending.

---

## 8. Context Window

**File:** `routers/context.py` — `get_context()`

```sql
SELECT * FROM messages
WHERE upload_id = ?
  AND row_index BETWEEN max(0, target_row_index − before) AND target_row_index + after
ORDER BY row_index
```

The `row_index` column stores the original 0-based position of each row in the uploaded CSV. This guarantees that context is reconstructed in the exact order the messages appeared in the source export, regardless of whether timestamps are precise or have duplicates.

The target message is identified by `id = message_id` and tagged with `is_target: true` in the returned list.

Parameter bounds: `before` and `after` are clamped to [0, 200].

---

## 9. RAG Chat

**File:** `routers/chat.py` — `chat_endpoint()`

### Retrieval (parallel)

```python
semantic_rows, keyword_rows = await asyncio.gather(
    loop.run_in_executor(vector_executor, _semantic_search),
    keyword_search(keyword=message, upload_ids=upload_ids, limit=10),
)
```

**Semantic path** (`_semantic_search()`):

- Queries the active collection for 12 nearest neighbours.
- Filters by `upload_ids` if specified.
- Converts distances to `_score = round(1.0 − dist, 4)`.
- Returns at most 12 rows.

**Keyword path** (`keyword_search()`):

- Runs the FTS5 two-phase search with the user's message as the keyword.
- Returns at most 10 rows.

**Merge:** Both result sets are deduplicated by `msg_uuid` (semantic results take priority). The combined list is capped at 20 messages.

### Prompt construction

**With retrieved context:**

```
system:
  "You are a knowledgeable assistant for the Suno AI Discord community.
   INSTRUCTIONS: Use retrieved excerpts as PRIMARY source...
   MANDATORY FORMATTING: ## heading, ### subheadings, **bold**,
   - bullets, > blockquotes, inline code, --- + *Sources* section.
   RETRIEVED CONTEXT:
   [username | date] content
   [username | date] content
   ..."

history[-20:]   (prior turns, filtered for valid role values)

user: <message>
```

**Without context (no embeddings):**

```
system:
  "No embedded messages available — answer from general knowledge.
   MANDATORY FORMATTING: ..."

history[-20:]
user: <message>
```

### Streaming

`state.openai_client.chat.completions.create(..., stream=True)` uses the synchronous OpenAI client (not async) because the async client is reserved for embedding calls. Each `chunk.choices[0].delta.content` token is serialised as `data: {"content": "<token>"}\n\n` and flushed via `StreamingResponse`.

The `X-Accel-Buffering: no` header disables Nginx proxy buffering so tokens reach the browser immediately.

---

## 10. Hybrid Summary Pipeline

**File:** `routers/chat.py` — `summarize_endpoint()`

### Stage 1: Metadata filter

```sql
SELECT msg_uuid, username, date, content FROM messages
WHERE [username LIKE ?]
  [AND upload_id IN (?,...)]
  [AND is_suno_team filter]
  [AND date range]
  [AND min_words]
ORDER BY date, row_index
```

Produces `filtered_map: dict[uuid → row]` used as the intersection basis for all subsequent stages.

### Stage 2: Semantic retrieval (cluster mode only)

**Retrieval query:** The user's custom prompt is used as the retrieval query if provided; otherwise a generic coverage query is used:  
`"key discussions, important insights, notable feedback, use cases, significant events"`.

The query is embedded with `embed_texts_async`. The vector store is queried with:

```
overfetch_n = min(total_in_store, max(n_filtered × 5, 2000))
```

Results are intersected with `filtered_map` (UUID matching). Each intersecting result gets `score = round(1.0 − dist, 4)`. Scores are sorted descending.

**Adaptive threshold:** The 70th percentile of scores is computed. Results scoring at or above this percentile are kept (i.e., the top 30% by score are discarded as less relevant — wait, actually the bottom 30% are discarded: `threshold = np.percentile(scores, 30)`, so messages scoring at or above the 30th-percentile value are kept). If this leaves fewer than `_MIN_CANDIDATES` (15) results, the top 15 are used unconditionally.

After filtering, stored embeddings are fetched for the surviving UUIDs using `col.get(ids=top_uuids, include=["embeddings"])`.

### Stage 3: Deduplication

**Function:** `_deduplicate_candidates(rows, embs, threshold=0.97)`

1. Pre-drop messages shorter than 10 characters.
2. Compute pairwise cosine similarity matrix: `sim = (embs / |embs|) @ (embs / |embs|).T` using NumPy broadcasting.
3. Process messages in ranking order. For each kept message `i`, mark any later message `j` with `sim[i,j] ≥ 0.97` as a duplicate.
4. Returns filtered rows and embeddings.

A threshold of 0.97 targets near-identical messages (e.g. copy-pastes, bot reposts) while preserving paraphrases and topically related but distinct messages.

### Stage 4: Clustering

**Function:** `_cluster_candidates(rows, embs)`

Tries clustering algorithms in priority order:

| Priority | Algorithm | Conditions | Notes |
|---|---|---|---|
| 1 | HDBSCAN | `import hdbscan` succeeds | `min_cluster_size = max(2, n//25)`, `min_samples = max(1, n//50)`, metric=euclidean |
| 2 | OPTICS | `from sklearn.cluster import OPTICS` succeeds | same parameters |
| 3 | KMeans (sklearn) | `from sklearn.cluster import KMeans` succeeds | `k = max(3, n//5)` |
| 4 | KMeans (NumPy) | fallback | Lloyd's algorithm, seed=42 |

Noise points (HDBSCAN/OPTICS label −1) are promoted to individual singleton clusters so outlier messages are not silently discarded.

For very small inputs (n ≤ 4), each message is its own singleton cluster and no algorithm is run.

### Stage 5: Per-cluster sampling

**Function:** `_sample_cluster(cluster_rows, cluster_embs, n_closest=5, n_furthest=5)`

For each cluster:

1. Compute the centroid as the mean of cluster embeddings.
2. Sort cluster members by L2 distance from the centroid.
3. Select the 5 closest (most representative) and 5 furthest (most peripheral/diverse).
4. If the cluster has ≤ 10 members, all are kept.

This bimodal sampling ensures the LLM receives both the central theme of each cluster and its boundary cases.

### Stage 6: Assemble

Selected messages from all clusters are combined, sorted chronologically by `date`, and capped at 120 messages (`max_evidence`).

### Stage 7: LLM generation

The evidence set is formatted as `[username | date]: content` lines and combined with the user's prompt (or a default structured Markdown prompt) into a single user message. The pipeline log events (one JSON object per step) are emitted as SSE frames *before* the LLM token stream begins, making the retrieval process visible in the browser.

### Fallback chain

```
Semantic retrieval returns ≥ 10 candidates?
├── YES → run dedup/cluster/sample
└── NO  → fetch stored embeddings for entire filtered_map (up to 3000 random)
          ≥ 10 rows have embeddings?
          ├── YES → run dedup/cluster/sample on those (fallback cluster)
          └── NO  → send all filtered rows to LLM directly
```

### Follow-up endpoint

`POST /api/summarize/followup` re-runs the same pipeline but:
- The *follow-up question* (not the original prompt) is used as the retrieval query.
- The overfetch multiplier is `max(n_filtered × 4, 1000)` instead of `× 5 / 2000`.
- The evidence cap is 80 instead of 120.
- The initial summary (first assistant turn in `history`) is embedded in the system prompt.
- Prior Q&A turns (up to 20) are appended to the message list.

---

## 11. Summarize Results Pipeline

**File:** `routers/chat.py` — `summarize_results_endpoint()`

This endpoint receives messages directly from the browser (the current search result set) rather than querying the database. It uses the same dedup/cluster/sample `_build_evidence_set()` pipeline, with two differences:

1. **Embedding lookup instead of ANN search:** Embeddings are fetched by UUID using `col.get(ids=uuids, include=["embeddings"])`. There is no semantic retrieval query — the input set is the full candidate pool.

2. **Token safety net:** After clustering, the character count of the evidence set is estimated:
   ```
   chars = sum(len(username) + len(date) + len(content) + 10 for each message)
   ```
   If `chars > 360,000` (≈ 90,000 tokens at 4 chars/token), the pipeline auto-triggers cluster+sample regardless of `retrieval_mode`. If clustering still yields an oversized payload, a loop progressively truncates the list to 80% of its size until it fits.

The follow-up variant (`/api/summarize-results/followup`) is fully stateless: all context is in the `history` array; no DB or vector calls are made.

---

## 12. User Profile Analysis

**File:** `routers/chat.py` — `user_profile_endpoint()`

Functionally identical to the Hybrid Summary pipeline except:

- The filter is an exact-match on `LOWER(username) = LOWER(?)` rather than a partial match.
- The fallback retrieval query is:  
  `"attitude, opinions, concerns, feedback, and persona of {username} regarding Suno AI"`
- Entry and exit dates are extracted from `db_rows[0]` and `db_rows[-1]`.
- The default LLM prompt requests: persona, entry/exit dates, evolution of attitude (chronological), key topics, notable quotes, summary assessment.
- The `msg_uuid` list for the profile query is capped at 3000 random samples for the fallback embedding fetch (same as Hybrid Summary).

The follow-up endpoint (`/api/user-profile/followup`) appends evidence as a block to the user turn rather than the system prompt, and includes the initial profile as a system-level context block.

---

## 13. Bookmarks and Labels

**File:** `routers/bookmarks.py`, `routers/labels.py`

### Bookmark creation

1. Validate `msg_id` exists in `messages`.
2. Check for existing bookmark on the same `msg_id` — return `{status: "exists"}` rather than creating a duplicate.
3. Insert: `(msg_id, ctx_before, ctx_after, note, created_at)`.

### Bookmark listing

Single query joining `bookmarks` → `messages`. A second query fetches all `(bookmark_id, label)` rows. Labels are grouped into a dict `{bookmark_id: [label, ...]}` in Python and merged into each bookmark dict before returning. This avoids N+1 queries.

### ID-only listing

`GET /api/bookmarks/ids` returns only `[msg_id, ...]` — used by the frontend to cheaply determine which message cards should show the "bookmarked" indicator without loading full bookmark data.

### Labels

Labels are stored with a `name` (unique), a `color` (hex string, default `#6366f1`), and a `created_at` timestamp. Assignment uses `INSERT OR IGNORE` on the `bookmark_labels` junction table.

On bookmark deletion, `ON DELETE CASCADE` automatically removes all `bookmark_labels` rows for that bookmark.

---

## 14. Security Model

**File:** `app.py`

### Authentication middleware (`_AuthMiddleware`)

- If `API_SECRET` is empty, all requests are allowed (development mode).
- The root path `/` and all `/static/*` paths are always public (UI must be accessible without a token).
- All `/api/*` paths require: `Authorization: Bearer <API_SECRET>`.
- Non-matching or missing header → HTTP 401 `{"detail": "Unauthorized"}`.

The middleware is a `BaseHTTPMiddleware` that calls `call_next` after validation, adding no overhead to unauthenticated paths.

### Security headers middleware (`_SecurityHeadersMiddleware`)

Applied to all responses:

| Header | Value |
|---|---|
| `X-Content-Type-Options` | `nosniff` |
| `X-Frame-Options` | `DENY` |
| `Referrer-Policy` | `strict-origin-when-cross-origin` |
| `Permissions-Policy` | `geolocation=(), microphone=(), camera=()` |
| `Strict-Transport-Security` | `max-age=63072000; includeSubDomains` (HTTPS only) |

### Chat model allowlist

`VALID_CHAT_MODELS` in `config.py` is a `frozenset` of permitted OpenAI model IDs. Any request specifying an unlisted model ID receives HTTP 400. This prevents arbitrary model invocation via the chat and summarisation endpoints.

### FTS5 injection prevention

`_build_fts_query()` in `sql_helpers.py` strips FTS5 syntax characters before building the MATCH expression. All SQL parameters are passed as bind variables (parameterised queries), never interpolated into SQL strings except for `IN (?,?,...)` placeholders which are generated from `",".join("?" * len(ids))` and bound normally.

---

## 15. Caching Strategy

**File:** `state.py`

Two in-memory caches reduce repeated DB and vector-store hits from frequent dashboard polling:

| Cache | Variable | TTL | Invalidated by |
|---|---|---|---|
| Stats | `_stats_cache` | 30 s | `invalidate_all_caches()` |
| Uploads list | `_uploads_cache` | 30 s | `invalidate_all_caches()` |

`invalidate_all_caches()` is called after every write operation: CSV upload, re-embed completion, upload deletion. This ensures the UI reflects the updated state immediately after a write, while reads within a 30-second window served from cache avoid redundant vector-count queries (which require a network round-trip to Qdrant).

Cache staleness is checked with `time.monotonic()` (monotonic clock, unaffected by system clock adjustments).
