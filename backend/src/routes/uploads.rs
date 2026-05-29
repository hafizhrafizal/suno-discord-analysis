use axum::{
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::{sse::{Event, Sse}, IntoResponse},
    Json,
};
use serde_json::{json, Value};
use std::convert::Infallible;
use uuid::Uuid;
use chrono::Datelike;
use crate::{error::{AppError, Result}, state::AppState};

const EMBED_MODEL: &str = "text-embedding-3-small";
/// 500 inputs × 1 536 dims × ~12 chars/float ≈ 9 MB per response — reliable ceiling before
/// reqwest body reads become unstable.  Throughput comes from EMBED_CONCURRENCY, not batch size.
const EMBED_BATCH_SIZE: usize = 500;
/// Parallel OpenAI requests per window.  Kept at 3 to stay within default tier rate limits
/// (text-embedding-3-small: 3 000 RPM / 1 000 000 TPM on tier 1 — 3 × 500 = 1 500 per window).
const EMBED_CONCURRENCY: usize = 3;

fn compute_week(date: &str) -> Option<String> {
    let date = date.trim();
    if date.is_empty() {
        return None;
    }
    chrono::NaiveDateTime::parse_from_str(date, "%Y-%m-%d %H:%M:%S%.f")
        .ok()
        .or_else(|| chrono::NaiveDateTime::parse_from_str(date, "%Y-%m-%d %H:%M:%S").ok())
        .map(|dt| {
            let iw = dt.iso_week();
            format!("{}-{:02}", iw.year(), iw.week())
        })
        .or_else(|| {
            chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d")
                .ok()
                .map(|d| {
                    let iw = d.iso_week();
                    format!("{}-{:02}", iw.year(), iw.week())
                })
        })
}

/// Call OpenAI batch embeddings API once.
/// Returns `Ok(embeddings)` on success, `Err(retry_after_secs)` on 429,
/// `Err(u64::MAX)` on permanent errors (4xx), `Err(0)` on transient failures (5xx, network).
async fn embed_batch_once(
    client: &reqwest::Client,
    api_key: &str,
    texts: &[String],
) -> std::result::Result<Vec<Option<Vec<f64>>>, u64> {
    let resp = match client
        .post("https://api.openai.com/v1/embeddings")
        .bearer_auth(api_key)
        .json(&json!({ "model": EMBED_MODEL, "input": texts }))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("OpenAI embed request failed: {}", e);
            return Err(0);
        }
    };

    let status = resp.status();

    if status.as_u16() == 429 {
        // Read Retry-After header BEFORE consuming the body with .text()
        let retry_after = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(5)
            .max(1); // .max(1) prevents Err(0) collision with the transient-failure sentinel
        let body_preview = resp.text().await.unwrap_or_default();
        tracing::warn!(
            "OpenAI embed rate limited (429), will retry in {}s: {}",
            retry_after,
            &body_preview[..body_preview.len().min(200)]
        );
        return Err(retry_after);
    }

    // Read body as text first — avoids opaque "error decoding response body" on failures
    let body = match resp.text().await {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!("OpenAI embed body read failed: {}", e);
            return Err(0);
        }
    };

    // 4xx (other than 429) = permanent — wrong key, bad input, etc. Don't retry.
    if status.is_client_error() {
        tracing::warn!("OpenAI embed permanent error HTTP {} — {}", status, &body[..body.len().min(300)]);
        return Err(u64::MAX);
    }

    // 5xx = transient server error — retry with back-off.
    if !status.is_success() {
        tracing::warn!("OpenAI embed server error HTTP {} — {}", status, &body[..body.len().min(200)]);
        return Err(0);
    }

    if !body.starts_with('{') {
        tracing::warn!("OpenAI embed unexpected body (first 200 chars): {}", &body[..body.len().min(200)]);
        return Err(0);
    }

    let data: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("OpenAI embed JSON parse error: {} — body len {} bytes", e, body.len());
            return Err(0);
        }
    };

    // Surface API-level error objects (e.g. token limit exceeded, invalid model)
    if let Some(err_obj) = data.get("error") {
        tracing::warn!("OpenAI embed API error (permanent): {}", err_obj);
        return Err(u64::MAX);
    }

    let mut result: Vec<Option<Vec<f64>>> = vec![None; texts.len()];
    if let Some(arr) = data["data"].as_array() {
        for item in arr {
            let idx = item["index"].as_u64().unwrap_or(0) as usize;
            if idx < texts.len() {
                let v: Vec<f64> = item["embedding"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
                    .unwrap_or_default();
                if !v.is_empty() {
                    result[idx] = Some(v);
                }
            }
        }
    }
    Ok(result)
}

/// Embed a batch with up to 4 attempts using a *shared* client (TLS connections are reused).
/// On 429 respects `Retry-After`; on transient failures uses exponential back-off (2 s, 4 s, 8 s).
/// On permanent errors (4xx, API error object) fails immediately without retrying.
async fn embed_batch(client: reqwest::Client, api_key: &str, texts: &[String]) -> Vec<Option<Vec<f64>>> {
    for attempt in 0u32..4 {
        match embed_batch_once(&client, api_key, texts).await {
            Ok(embeddings) => return embeddings,
            Err(u64::MAX) => {
                // Permanent error (invalid key, bad request, API-level error) — no point retrying
                tracing::warn!("Embedding batch skipped (permanent error) — {} messages", texts.len());
                return vec![None; texts.len()];
            }
            Err(0) => {
                if attempt < 3 {
                    let delay = 2u64.pow(attempt);
                    tracing::info!("Embed attempt {} failed (transient), retrying in {}s", attempt + 1, delay);
                    tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
                }
            }
            Err(retry_after) => {
                tracing::info!(
                    "OpenAI rate-limited (attempt {}), waiting {}s (Retry-After header)",
                    attempt + 1,
                    retry_after
                );
                tokio::time::sleep(std::time::Duration::from_secs(retry_after)).await;
            }
        }
    }
    tracing::warn!("Embedding batch failed after 4 attempts — skipping {} messages", texts.len());
    vec![None; texts.len()]
}

/// Build the ChromaDB base URL and a shared reqwest client.
fn chroma_setup(state: &AppState) -> (String, reqwest::Client) {
    let base = format!(
        "http://{}:{}/api/v2/tenants/default_tenant/databases/default_database",
        state.config.chroma_host, state.config.chroma_port
    );
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());
    (base, client)
}

/// Delete all ChromaDB vectors whose metadata `upload_id` matches.
/// Returns count of deleted vectors (best-effort; 0 on any failure).
async fn delete_chroma_by_upload_id(
    client: &reqwest::Client,
    base: &str,
    collection_id: &str,
    upload_id: &str,
) -> i64 {
    match client
        .post(format!("{}/collections/{}/delete", base, collection_id))
        .json(&json!({ "where": { "upload_id": { "$eq": upload_id } } }))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => r
            .json::<Value>().await.ok()
            .and_then(|v| v["ids"].as_array().map(|a| a.len() as i64))
            .unwrap_or(0),
        Ok(r) => {
            let s = r.status();
            let b = r.text().await.unwrap_or_default();
            tracing::warn!("ChromaDB delete by upload_id HTTP {}: {}", s, &b[..b.len().min(200)]);
            0
        }
        Err(e) => { tracing::warn!("ChromaDB delete by upload_id failed: {}", e); 0 }
    }
}

/// Delete all Qdrant vectors whose payload `upload_id` matches.
/// Returns an approximate count of deleted vectors (best-effort; 0 on failure).
async fn delete_qdrant_by_upload_id(
    client: &reqwest::Client,
    qdrant_url: &str,
    collection: &str,
    api_key: Option<&str>,
    upload_id: &str,
) -> i64 {
    let filter = json!({
        "filter": {
            "must": [{"key": "upload_id", "match": {"value": upload_id}}]
        }
    });
    // Count first for best-effort reporting
    let count_url = format!("{}/collections/{}/points/count", qdrant_url, collection);
    let mut count_req = client.post(&count_url).json(&filter);
    if let Some(key) = api_key { count_req = count_req.header("api-key", key); }
    let approx_count: i64 = match count_req.send().await {
        Ok(r) if r.status().is_success() => r.json::<Value>().await.ok()
            .and_then(|v| v["result"]["count"].as_i64())
            .unwrap_or(0),
        _ => 0,
    };

    let delete_url = format!("{}/collections/{}/points/delete", qdrant_url, collection);
    let mut del_req = client.post(&delete_url).json(&filter);
    if let Some(key) = api_key { del_req = del_req.header("api-key", key); }
    match del_req.send().await {
        Ok(r) if r.status().is_success() => approx_count,
        Ok(r) => {
            let s = r.status();
            let b = r.text().await.unwrap_or_default();
            tracing::warn!("Qdrant delete by upload_id HTTP {}: {}", s, &b[..b.len().min(200)]);
            0
        }
        Err(e) => { tracing::warn!("Qdrant delete by upload_id failed: {}", e); 0 }
    }
}

/// Get existing ChromaDB collection UUID by name, or create it with cosine distance.
async fn get_or_create_chroma_collection(
    client: &reqwest::Client,
    base: &str,
    name: &str,
) -> Option<String> {
    // Try GET first
    if let Ok(r) = client.get(format!("{}/collections/{}", base, name)).send().await {
        if r.status().is_success() {
            if let Ok(info) = r.json::<Value>().await {
                if let Some(id) = info["id"].as_str() {
                    return Some(id.to_string());
                }
            }
        }
    }
    // Create (ChromaDB get_or_create is idempotent)
    let r = client
        .post(format!("{}/collections", base))
        .json(&json!({
            "name": name,
            "get_or_create": true,
            "metadata": { "hnsw:space": "cosine" }
        }))
        .send()
        .await
        .ok()?;
    let info: Value = r.json().await.ok()?;
    info["id"].as_str().map(String::from)
}

/// Return the set of IDs that already have vectors in a ChromaDB collection.
/// All 1 000-ID chunks are queried concurrently.
async fn get_existing_chroma_ids(
    client: &reqwest::Client,
    base: &str,
    collection_id: &str,
    ids: &[String],
) -> std::collections::HashSet<String> {
    let futs = ids.chunks(1000).map(|chunk| {
        let client = client.clone();
        let url = format!("{}/collections/{}/get", base, collection_id);
        let chunk = chunk.to_vec();
        async move {
            let mut found = Vec::<String>::new();
            if let Ok(r) = client
                .post(&url)
                .json(&serde_json::json!({ "ids": chunk, "include": [] }))
                .send()
                .await
            {
                if let Ok(data) = r.json::<serde_json::Value>().await {
                    if let Some(arr) = data["ids"].as_array() {
                        for v in arr {
                            if let Some(s) = v.as_str() {
                                found.push(s.to_string());
                            }
                        }
                    }
                }
            }
            found
        }
    });
    futures::future::join_all(futs)
        .await
        .into_iter()
        .flatten()
        .collect()
}

/// Ensure a Qdrant collection exists; create with cosine distance if missing.
async fn ensure_qdrant_collection(
    client: &reqwest::Client,
    qdrant_url: &str,
    collection: &str,
    api_key: Option<&str>,
    vector_size: u64,
) -> bool {
    let url = format!("{}/collections/{}", qdrant_url, collection);
    let mut get_req = client.get(&url);
    if let Some(key) = api_key { get_req = get_req.header("api-key", key); }
    if let Ok(r) = get_req.send().await {
        if r.status().is_success() { return true; }
    }
    let mut put_req = client
        .put(&url)
        .json(&json!({ "vectors": { "size": vector_size, "distance": "Cosine" } }));
    if let Some(key) = api_key { put_req = put_req.header("api-key", key); }
    put_req.send().await.map(|r| r.status().is_success()).unwrap_or(false)
}

/// Return the set of IDs already present in a Qdrant collection.
async fn get_existing_qdrant_ids(
    client: &reqwest::Client,
    qdrant_url: &str,
    collection: &str,
    api_key: Option<&str>,
    ids: &[String],
) -> std::collections::HashSet<String> {
    let futs = ids.chunks(1000).map(|chunk| {
        let client = client.clone();
        let url = format!("{}/collections/{}/points", qdrant_url, collection);
        let chunk = chunk.to_vec();
        let api_key = api_key.map(String::from);
        async move {
            let mut found = Vec::<String>::new();
            let mut req = client.post(&url).json(&json!({
                "ids": chunk,
                "with_payload": false,
                "with_vector": false,
            }));
            if let Some(key) = &api_key { req = req.header("api-key", key); }
            if let Ok(r) = req.send().await {
                if let Ok(data) = r.json::<Value>().await {
                    if let Some(arr) = data["result"].as_array() {
                        for p in arr {
                            if let Some(id) = p["id"].as_str() {
                                found.push(id.to_string());
                            }
                        }
                    }
                }
            }
            found
        }
    });
    futures::future::join_all(futs).await.into_iter().flatten().collect()
}

/// Upsert a batch of embeddings into Qdrant.
async fn upsert_qdrant_batch(
    client: &reqwest::Client,
    qdrant_url: &str,
    collection: &str,
    api_key: Option<&str>,
    ids: &[String],
    embeddings: &[Vec<f64>],
    documents: &[String],
    upload_id: &str,
) {
    let points: Vec<Value> = ids
        .iter()
        .zip(embeddings.iter())
        .zip(documents.iter())
        .map(|((id, emb), doc)| json!({
            "id": id,
            "vector": emb,
            "payload": { "msg_uuid": id, "upload_id": upload_id, "document": doc }
        }))
        .collect();
    let url = format!("{}/collections/{}/points", qdrant_url, collection);
    let mut req = client.put(&url).json(&json!({ "points": points }));
    if let Some(key) = api_key { req = req.header("api-key", key); }
    if let Err(e) = req.send().await {
        tracing::warn!("Qdrant upsert failed: {}", e);
    }
}

/// Upsert a batch of embeddings into ChromaDB.
async fn upsert_chroma_batch(
    client: &reqwest::Client,
    base: &str,
    collection_id: &str,
    ids: &[String],
    embeddings: &[Vec<f64>],
    documents: &[String],
    upload_id: &str,
) {
    let metadatas: Vec<Value> = ids
        .iter()
        .map(|id| json!({ "msg_uuid": id, "upload_id": upload_id }))
        .collect();
    let res = client
        .post(format!("{}/collections/{}/upsert", base, collection_id))
        .json(&json!({
            "ids": ids,
            "embeddings": embeddings,
            "documents": documents,
            "metadatas": metadatas,
        }))
        .send()
        .await;
    if let Err(e) = res {
        tracing::warn!("ChromaDB upsert failed: {}", e);
    }
}

pub async fn upload_csv(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse> {
    let mut csv_bytes: Option<Vec<u8>> = None;
    let mut original_filename = String::from("upload.csv");

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::BadRequest(e.to_string()))?
    {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" {
            original_filename = field
                .file_name()
                .unwrap_or("upload.csv")
                .to_string();
            let data = field
                .bytes()
                .await
                .map_err(|e| AppError::BadRequest(e.to_string()))?;
            csv_bytes = Some(data.to_vec());
        }
    }

    let csv_data = csv_bytes.ok_or_else(|| AppError::BadRequest("No file field found".into()))?;

    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(csv_data.as_slice());

    let headers = reader
        .headers()
        .map_err(|e| AppError::BadRequest(format!("CSV parse error: {}", e)))?
        .clone();

    fn normalize_header(h: &str) -> &'static str {
        match h.trim().to_lowercase().as_str() {
            "id" | "message id" | "msg_uuid" => "msg_uuid",
            "author" | "author id" | "username" => "username",
            "content" | "message" | "text" => "content",
            "date" | "timestamp" | "created_at" | "time" => "date",
            "attachments" | "attachment" => "attachments",
            "reactions" | "reaction" => "reactions",
            "is_suno_team" | "suno_team" | "is_team" => "is_suno_team",
            _ => "unknown",
        }
    }

    let header_map: Vec<(&'static str, usize)> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| (normalize_header(h), i))
        .filter(|(name, _)| *name != "unknown")
        .collect();

    let get_field = move |row: &csv::StringRecord, field: &str| -> Option<String> {
        header_map
            .iter()
            .find(|(n, _)| *n == field)
            .and_then(|(_, i)| row.get(*i))
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
    };

    let upload_id = Uuid::new_v4().to_string();
    let now = chrono::Utc::now().to_rfc3339();

    let records: Vec<csv::StringRecord> = reader
        .records()
        .filter_map(|r| r.ok())
        .collect();
    let total = records.len() as i64;

    sqlx::query(
        "INSERT INTO uploads (id, filename, row_count, upload_time) VALUES ($1, $2, $3, $4)",
    )
    .bind(&upload_id)
    .bind(&original_filename)
    .bind(total)
    .bind(&now)
    .execute(&state.db)
    .await?;

    // Capture everything the stream needs before moving into the generator
    let upload_id_clone = upload_id.clone();
    let db = state.db.clone();
    let openai_key_opt = state.openai_key.read().await.clone();
    let chroma_host = state.config.chroma_host.clone();
    let chroma_port = state.config.chroma_port;
    let chroma_collection = state.config.chroma_collection.clone();
    let vector_db_cfg = state.config.vector_db.clone();
    let qdrant_url_cfg = state.config.qdrant_url.clone();
    let qdrant_collection_cfg = state.config.qdrant_collection.clone();
    let qdrant_api_key_cfg = state.config.qdrant_api_key.clone();

    let event_stream = async_stream::stream! {
        // ── Phase 1: insert rows ──────────────────────────────────────────────
        let mut processed: i64 = 0; // rows seen (used for row_index ordering)
        let mut inserted: i64 = 0;  // rows actually inserted (not duplicate-skipped)
        let mut all_csv_uuids: Vec<String> = Vec::new(); // every UUID from the CSV (inserted + skipped)
        let batch_size = 500usize;

        for chunk in records.chunks(batch_size) {
            let mut tx = match db.begin().await {
                Ok(tx) => tx,
                Err(e) => {
                    yield Ok::<Event, Infallible>(Event::default().data(
                        serde_json::to_string(&json!({"type":"error","message":e.to_string()})).unwrap()
                    ));
                    return;
                }
            };

            for record in chunk {
                let msg_uuid = get_field(record, "msg_uuid")
                    .unwrap_or_else(|| Uuid::new_v4().to_string());
                all_csv_uuids.push(msg_uuid.clone()); // track before conflict check
                let username = get_field(record, "username")
                    .unwrap_or_else(|| "unknown".to_string());
                let content = match get_field(record, "content") {
                    Some(c) => c,
                    None => continue,
                };
                let date = get_field(record, "date").unwrap_or_default();
                let attachments = get_field(record, "attachments");
                let reactions = get_field(record, "reactions");
                let is_suno_team = get_field(record, "is_suno_team")
                    .unwrap_or_else(|| "false".to_string());
                let row_idx = processed;
                processed += 1;
                let word_count = content.split_whitespace().count() as i64;
                let week = compute_week(&date);

                if let Ok(r) = sqlx::query(
                    "INSERT INTO messages
                     (msg_uuid, username, date, content, attachments, reactions,
                      is_suno_team, upload_id, row_index, week, word_count)
                     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                     ON CONFLICT DO NOTHING",
                )
                .bind(&msg_uuid)
                .bind(&username)
                .bind(&date)
                .bind(&content)
                .bind(&attachments)
                .bind(&reactions)
                .bind(&is_suno_team)
                .bind(&upload_id_clone)
                .bind(row_idx)
                .bind(&week)
                .bind(word_count)
                .execute(&mut *tx)
                .await {
                    inserted += r.rows_affected() as i64;
                }
            }

            let _ = tx.commit().await;

            yield Ok(Event::default().data(
                serde_json::to_string(&json!({
                    "type": "progress",
                    "inserted": inserted,
                    "skipped": processed - inserted,
                    "total": total,
                })).unwrap()
            ));
        }

        // ── Suno-team backfill ────────────────────────────────────────────────
        let _ = sqlx::query(
            "UPDATE messages
             SET is_suno_team = 'true'
             WHERE upload_id = $1
               AND LOWER(is_suno_team) NOT IN ('true', '1')
               AND username IN (
                   SELECT DISTINCT username FROM messages
                   WHERE LOWER(is_suno_team) IN ('true', '1')
                     AND upload_id != $1
               )",
        )
        .bind(&upload_id_clone)
        .execute(&db)
        .await;

        // ── Phase 2: embed with text-embedding-3-small ────────────────────────
        let mut embedded_count: i64 = 0;
        let mut already_in_chroma: i64 = 0;

        match openai_key_opt {
            None => {
                yield Ok(Event::default().data(
                    serde_json::to_string(&json!({
                        "type": "embed_skip",
                        "reason": "no_api_key",
                    })).unwrap()
                ));
            }
            Some(api_key) => {
                // Fetch msg_uuid + content for all rows in the CSV (both newly inserted and
                // pre-existing ones that were skipped by ON CONFLICT). This handles re-uploads
                // of the same file correctly: even if 0 rows were inserted, the messages still
                // exist in the DB under their original upload_id and need to be embedded.
                let msgs: Vec<(String, String)> = match sqlx::query(
                    "SELECT msg_uuid, content FROM messages
                     WHERE msg_uuid = ANY($1) AND content IS NOT NULL",
                )
                .bind(&all_csv_uuids)
                .fetch_all(&db)
                .await
                {
                    Ok(rows) => {
                        use sqlx::Row;
                        rows.into_iter()
                            .map(|r| (r.get::<String, _>("msg_uuid"), r.get::<String, _>("content")))
                            .collect()
                    }
                    Err(e) => {
                        tracing::warn!("Failed to fetch messages for embedding: {}", e);
                        vec![]
                    }
                };

                if !msgs.is_empty() {
                    let is_qdrant = vector_db_cfg.to_lowercase() == "qdrant";
                    let http_client = reqwest::Client::builder()
                        .timeout(std::time::Duration::from_secs(120))
                        .build()
                        .unwrap_or_else(|_| reqwest::Client::new());
                    let embed_client = http_client.clone();

                    let (chroma_base, collection_id_opt) = if is_qdrant {
                        let ok = ensure_qdrant_collection(
                            &http_client,
                            &qdrant_url_cfg,
                            &qdrant_collection_cfg,
                            qdrant_api_key_cfg.as_deref(),
                            1536,
                        ).await;
                        (String::new(), if ok { Some(String::new()) } else { None })
                    } else {
                        let base = format!(
                            "http://{}:{}/api/v2/tenants/default_tenant/databases/default_database",
                            chroma_host, chroma_port
                        );
                        let cid = get_or_create_chroma_collection(
                            &http_client, &base, &chroma_collection,
                        ).await;
                        (base, cid)
                    };

                    match collection_id_opt {
                        None => {
                            tracing::warn!("Vector store unavailable — embedding skipped");
                            yield Ok(Event::default().data(
                                serde_json::to_string(&json!({
                                    "type": "embed_skip",
                                    "reason": "vector_db_unavailable",
                                })).unwrap()
                            ));
                        }
                        Some(collection_id) => {
                            let all_ids: Vec<String> = msgs.iter().map(|(id, _)| id.clone()).collect();
                            let existing_ids = if is_qdrant {
                                get_existing_qdrant_ids(
                                    &http_client,
                                    &qdrant_url_cfg,
                                    &qdrant_collection_cfg,
                                    qdrant_api_key_cfg.as_deref(),
                                    &all_ids,
                                ).await
                            } else {
                                get_existing_chroma_ids(
                                    &http_client, &chroma_base, &collection_id, &all_ids,
                                ).await
                            };
                            already_in_chroma = existing_ids.len() as i64;

                            let msgs_to_embed: Vec<(String, String)> = msgs
                                .into_iter()
                                .filter(|(id, _)| !existing_ids.contains(id))
                                .collect();
                            let total_to_embed = msgs_to_embed.len() as i64;

                            if total_to_embed == 0 {
                                tracing::info!(
                                    "All {} vectors already in vector store — embedding skipped",
                                    already_in_chroma
                                );
                                yield Ok(Event::default().data(
                                    serde_json::to_string(&json!({
                                        "type": "embed_skip",
                                        "reason": "all_already_embedded",
                                        "count": already_in_chroma,
                                    })).unwrap()
                                ));
                            } else {
                                yield Ok(Event::default().data(
                                    serde_json::to_string(&json!({
                                        "type": "embed_start",
                                        "total": total_to_embed,
                                        "already_embedded": already_in_chroma,
                                        "model": EMBED_MODEL,
                                    })).unwrap()
                                ));

                                let all_batches: Vec<(Vec<String>, Vec<String>)> = msgs_to_embed
                                    .chunks(EMBED_BATCH_SIZE)
                                    .map(|chunk| (
                                        chunk.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>(),
                                        chunk.iter().map(|(_, c)| c.clone()).collect::<Vec<_>>(),
                                    ))
                                    .collect();

                                for window in all_batches.chunks(EMBED_CONCURRENCY) {
                                    let embed_futs = window.iter().map(|(ids, texts)| {
                                        let embed_client = embed_client.clone();
                                        let api_key = api_key.clone();
                                        let ids = ids.clone();
                                        let texts = texts.clone();
                                        async move {
                                            let embeddings = embed_batch(embed_client, &api_key, &texts).await;
                                            ids.into_iter()
                                                .zip(embeddings)
                                                .zip(texts)
                                                .filter_map(|((id, emb_opt), text)| {
                                                    emb_opt.map(|emb| (id, emb, text))
                                                })
                                                .collect::<Vec<_>>()
                                        }
                                    });
                                    let window_results = futures::future::join_all(embed_futs).await;

                                    let upsert_futs = window_results.iter().map(|valid| {
                                        let http_client = http_client.clone();
                                        let chroma_base = chroma_base.clone();
                                        let collection_id = collection_id.clone();
                                        let upload_id_inner = upload_id_clone.clone();
                                        let valid = valid.clone();
                                        let qdrant_url = qdrant_url_cfg.clone();
                                        let qdrant_col = qdrant_collection_cfg.clone();
                                        let qdrant_key = qdrant_api_key_cfg.clone();
                                        async move {
                                            if valid.is_empty() { return 0i64; }
                                            let v_ids: Vec<String> = valid.iter().map(|(id, _, _)| id.clone()).collect();
                                            let v_embs: Vec<Vec<f64>> = valid.iter().map(|(_, emb, _)| emb.clone()).collect();
                                            let v_docs: Vec<String> = valid.iter().map(|(_, _, doc)| doc.clone()).collect();
                                            if is_qdrant {
                                                upsert_qdrant_batch(
                                                    &http_client, &qdrant_url, &qdrant_col,
                                                    qdrant_key.as_deref(),
                                                    &v_ids, &v_embs, &v_docs, &upload_id_inner,
                                                ).await;
                                            } else {
                                                upsert_chroma_batch(
                                                    &http_client, &chroma_base, &collection_id,
                                                    &v_ids, &v_embs, &v_docs, &upload_id_inner,
                                                ).await;
                                            }
                                            valid.len() as i64
                                        }
                                    });
                                    let counts = futures::future::join_all(upsert_futs).await;
                                    embedded_count += counts.iter().sum::<i64>();

                                    yield Ok(Event::default().data(
                                        serde_json::to_string(&json!({
                                            "type": "embed_progress",
                                            "embedded": embedded_count,
                                            "total": total_to_embed,
                                        })).unwrap()
                                    ));
                                }
                            }

                            if embedded_count > 0 || already_in_chroma > 0 {
                                let _ = sqlx::query(
                                    "INSERT INTO embedded_uploads (upload_id, model_id)
                                     VALUES ($1, $2)
                                     ON CONFLICT DO NOTHING",
                                )
                                .bind(&upload_id_clone)
                                .bind(EMBED_MODEL)
                                .execute(&db)
                                .await;
                            }
                        }
                    }
                }
            }
        }

        // ── Done ──────────────────────────────────────────────────────────────
        yield Ok(Event::default().data(
            serde_json::to_string(&json!({
                "type": "done",
                "upload_id": upload_id_clone,
                "total_inserted": inserted,
                "total_skipped": processed - inserted,
                "embedded": embedded_count,
                "already_embedded": already_in_chroma,
                "embed_model": if embedded_count > 0 || already_in_chroma > 0 { EMBED_MODEL } else { "" },
            })).unwrap()
        ));
    };

    Ok(Sse::new(event_stream))
}

pub async fn list_uploads(State(state): State<AppState>) -> Result<Json<Value>> {
    let rows = sqlx::query(
        "SELECT id, filename, row_count, upload_time FROM uploads ORDER BY upload_time DESC",
    )
    .fetch_all(&state.db)
    .await?;

    // Fetch which models each upload has been embedded with
    let embed_rows = sqlx::query("SELECT upload_id, model_id FROM embedded_uploads")
        .fetch_all(&state.db)
        .await
        .unwrap_or_default();

    use sqlx::Row;
    let mut embed_map: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for r in &embed_rows {
        embed_map
            .entry(r.get::<String, _>("upload_id"))
            .or_default()
            .push(r.get::<String, _>("model_id"));
    }

    let uploads: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            let id: String = r.get("id");
            let models = embed_map.get(&id).cloned().unwrap_or_default();
            let embedded_models: serde_json::Map<String, Value> = models
                .into_iter()
                .map(|m| (m, Value::Bool(true)))
                .collect();
            json!({
                "id": id,
                "filename": r.get::<String, _>("filename"),
                "row_count": r.get::<i64, _>("row_count"),
                "upload_time": r.get::<String, _>("upload_time"),
                "embedded_models": embedded_models,
            })
        })
        .collect();

    Ok(Json(json!(uploads)))
}

pub async fn delete_upload(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>> {
    let deleted_messages: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM messages WHERE upload_id = $1")
            .bind(&id)
            .fetch_one(&state.db)
            .await
            .unwrap_or(0);

    // ON DELETE CASCADE removes messages and embedded_uploads automatically
    let affected = sqlx::query("DELETE FROM uploads WHERE id = $1")
        .bind(&id)
        .execute(&state.db)
        .await?
        .rows_affected();

    if affected == 0 {
        return Err(AppError::NotFound(format!("Upload {} not found", id)));
    }

    // Delete vectors for this upload from the configured vector store (best-effort)
    let deleted_vectors = if state.config.vector_db.to_lowercase() == "qdrant" {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        delete_qdrant_by_upload_id(
            &client, &state.config.qdrant_url, &state.config.qdrant_collection,
            state.config.qdrant_api_key.as_deref(), &id,
        ).await
    } else {
        let (chroma_base, chroma_client) = chroma_setup(&state);
        match get_or_create_chroma_collection(
            &chroma_client, &chroma_base, &state.config.chroma_collection,
        ).await {
            Some(cid) => delete_chroma_by_upload_id(&chroma_client, &chroma_base, &cid, &id).await,
            None => 0,
        }
    };

    Ok(Json(json!({
        "deleted": id,
        "deleted_messages": deleted_messages,
        "deleted_vectors": deleted_vectors,
    })))
}

/// Delete only the relational-DB rows (messages + upload record) while leaving
/// ChromaDB vectors in place.  Functionally identical to delete_upload for now
/// since we do not yet implement ChromaDB vector removal on "Delete All".
pub async fn delete_upload_sqlite(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>> {
    let deleted_messages: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM messages WHERE upload_id = $1")
            .bind(&id)
            .fetch_one(&state.db)
            .await
            .unwrap_or(0);

    let affected = sqlx::query("DELETE FROM uploads WHERE id = $1")
        .bind(&id)
        .execute(&state.db)
        .await?
        .rows_affected();

    if affected == 0 {
        return Err(AppError::NotFound(format!("Upload {} not found", id)));
    }
    Ok(Json(json!({ "deleted": id, "deleted_messages": deleted_messages })))
}

pub async fn delete_upload_embeddings(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>> {
    // Remove the tracking record so re-embed can run again
    sqlx::query("DELETE FROM embedded_uploads WHERE upload_id = $1")
        .bind(&id)
        .execute(&state.db)
        .await?;

    // Delete actual vectors for this upload from the configured vector store
    let deleted_vectors = if state.config.vector_db.to_lowercase() == "qdrant" {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        delete_qdrant_by_upload_id(
            &client, &state.config.qdrant_url, &state.config.qdrant_collection,
            state.config.qdrant_api_key.as_deref(), &id,
        ).await
    } else {
        let (chroma_base, chroma_client) = chroma_setup(&state);
        match get_or_create_chroma_collection(
            &chroma_client, &chroma_base, &state.config.chroma_collection,
        ).await {
            Some(cid) => delete_chroma_by_upload_id(&chroma_client, &chroma_base, &cid, &id).await,
            None => 0,
        }
    };

    Ok(Json(json!({ "deleted_vectors": deleted_vectors })))
}

pub async fn reembed(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>> {
    let exists: Option<String> = sqlx::query_scalar("SELECT id FROM uploads WHERE id = $1")
        .bind(&id)
        .fetch_optional(&state.db)
        .await?;
    if exists.is_none() {
        return Err(AppError::NotFound(format!("Upload {} not found", id)));
    }

    let api_key = state
        .get_openai_key(None)
        .await
        .ok_or_else(|| AppError::BadRequest("OpenAI API key not set".into()))?;

    let is_qdrant = state.config.vector_db.to_lowercase() == "qdrant";

    let chroma_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| AppError::Internal(e.to_string()))?;
    let embed_client = chroma_client.clone();

    // Connect to the configured vector store
    let (chroma_base, collection_id) = if is_qdrant {
        let ok = ensure_qdrant_collection(
            &chroma_client,
            &state.config.qdrant_url,
            &state.config.qdrant_collection,
            state.config.qdrant_api_key.as_deref(),
            1536,
        ).await;
        if !ok {
            return Err(AppError::Internal("Qdrant unavailable".into()));
        }
        (String::new(), String::new())
    } else {
        let base = format!(
            "http://{}:{}/api/v2/tenants/default_tenant/databases/default_database",
            state.config.chroma_host, state.config.chroma_port
        );
        let cid = get_or_create_chroma_collection(
            &chroma_client, &base, &state.config.chroma_collection,
        )
        .await
        .ok_or_else(|| AppError::Internal("ChromaDB unavailable".into()))?;
        (base, cid)
    };

    let msgs: Vec<(String, String)> = {
        use sqlx::Row;
        sqlx::query(
            "SELECT msg_uuid, content FROM messages
             WHERE upload_id = $1 AND content IS NOT NULL
             ORDER BY row_index",
        )
        .bind(&id)
        .fetch_all(&state.db)
        .await?
        .into_iter()
        .map(|r| (r.get("msg_uuid"), r.get("content")))
        .collect()
    };

    let total = msgs.len() as i64;
    let all_ids: Vec<String> = msgs.iter().map(|(mid, _)| mid.clone()).collect();

    // Skip already-embedded vectors so re-embed is resumable
    let existing_ids = if is_qdrant {
        get_existing_qdrant_ids(
            &chroma_client,
            &state.config.qdrant_url,
            &state.config.qdrant_collection,
            state.config.qdrant_api_key.as_deref(),
            &all_ids,
        ).await
    } else {
        get_existing_chroma_ids(&chroma_client, &chroma_base, &collection_id, &all_ids).await
    };

    let skipped = existing_ids.len() as i64;
    let msgs_to_embed: Vec<(String, String)> = msgs
        .into_iter()
        .filter(|(mid, _)| !existing_ids.contains(mid))
        .collect();

    let all_batches: Vec<(Vec<String>, Vec<String>)> = msgs_to_embed
        .chunks(EMBED_BATCH_SIZE)
        .map(|chunk| (
            chunk.iter().map(|(mid, _)| mid.clone()).collect::<Vec<_>>(),
            chunk.iter().map(|(_, c)| c.clone()).collect::<Vec<_>>(),
        ))
        .collect();

    let mut embedded: i64 = 0;

    for window in all_batches.chunks(EMBED_CONCURRENCY) {
        let embed_futs = window.iter().map(|(ids, texts)| {
            let embed_client = embed_client.clone();
            let api_key = api_key.clone();
            let ids = ids.clone();
            let texts = texts.clone();
            async move {
                let embeddings = embed_batch(embed_client, &api_key, &texts).await;
                ids.into_iter()
                    .zip(embeddings)
                    .zip(texts)
                    .filter_map(|((mid, emb_opt), text)| emb_opt.map(|emb| (mid, emb, text)))
                    .collect::<Vec<_>>()
            }
        });
        let window_results = futures::future::join_all(embed_futs).await;

        let upsert_futs = window_results.iter().map(|valid| {
            let chroma_client = chroma_client.clone();
            let chroma_base = chroma_base.clone();
            let collection_id = collection_id.clone();
            let upload_id_inner = id.clone();
            let valid = valid.clone();
            let qdrant_url = state.config.qdrant_url.clone();
            let qdrant_collection = state.config.qdrant_collection.clone();
            let qdrant_api_key = state.config.qdrant_api_key.clone();
            async move {
                if valid.is_empty() { return 0i64; }
                let v_ids: Vec<String> = valid.iter().map(|(vid, _, _)| vid.clone()).collect();
                let v_embs: Vec<Vec<f64>> = valid.iter().map(|(_, emb, _)| emb.clone()).collect();
                let v_docs: Vec<String> = valid.iter().map(|(_, _, doc)| doc.clone()).collect();
                if is_qdrant {
                    upsert_qdrant_batch(
                        &chroma_client, &qdrant_url, &qdrant_collection,
                        qdrant_api_key.as_deref(),
                        &v_ids, &v_embs, &v_docs, &upload_id_inner,
                    ).await;
                } else {
                    upsert_chroma_batch(
                        &chroma_client, &chroma_base, &collection_id,
                        &v_ids, &v_embs, &v_docs, &upload_id_inner,
                    ).await;
                }
                valid.len() as i64
            }
        });
        let counts = futures::future::join_all(upsert_futs).await;
        embedded += counts.iter().sum::<i64>();
    }

    sqlx::query(
        "INSERT INTO embedded_uploads (upload_id, model_id) VALUES ($1, $2)
         ON CONFLICT DO NOTHING",
    )
    .bind(&id)
    .bind(EMBED_MODEL)
    .execute(&state.db)
    .await?;

    Ok(Json(json!({
        "upload_id": id,
        "total": total,
        "embedded": embedded,
        "skipped": skipped,
        "model": EMBED_MODEL,
    })))
}

/// Purge orphaned vectors (in vector store but not in PostgreSQL messages table).
/// Dispatches to ChromaDB or Qdrant based on VECTOR_DB config.
/// Streams progress as SSE events so the browser doesn't time out on large collections.
pub async fn sync_chroma(State(state): State<AppState>) -> Result<impl IntoResponse> {
    let db = state.db.clone();
    let is_qdrant = state.config.vector_db.to_lowercase() == "qdrant";
    let qdrant_url = state.config.qdrant_url.clone();
    let qdrant_collection = state.config.qdrant_collection.clone();
    let qdrant_api_key = state.config.qdrant_api_key.clone();
    let (chroma_base, chroma_client) = chroma_setup(&state);
    let chroma_collection = state.config.chroma_collection.clone();

    let event_stream = async_stream::stream! {
        // ── Step 1: verify/resolve vector store collection ────────────────────
        // For Qdrant: check collection exists. For ChromaDB: resolve collection UUID.
        let (qdrant_delete_url, chroma_collection_id): (String, String) = if is_qdrant {
            let collection_url = format!("{}/collections/{}", qdrant_url, qdrant_collection);
            let mut check_req = reqwest::Client::new().get(&collection_url);
            if let Some(key) = &qdrant_api_key { check_req = check_req.header("api-key", key); }
            if !check_req.send().await.map(|r| r.status().is_success()).unwrap_or(false) {
                yield Ok::<Event, Infallible>(Event::default().data(
                    serde_json::to_string(&json!({ "type": "error", "message": "Qdrant collection unavailable" })).unwrap()
                ));
                return;
            }
            (format!("{}/collections/{}/points/delete", qdrant_url, qdrant_collection), String::new())
        } else {
            match get_or_create_chroma_collection(&chroma_client, &chroma_base, &chroma_collection).await {
                Some(id) => (String::new(), id),
                None => {
                    yield Ok::<Event, Infallible>(Event::default().data(
                        serde_json::to_string(&json!({ "type": "error", "message": "ChromaDB unavailable" })).unwrap()
                    ));
                    return;
                }
            }
        };

        // ── Step 2: load embeddable PostgreSQL msg_uuids ──────────────────────
        yield Ok(Event::default().data(
            serde_json::to_string(&json!({ "type": "pg_fetch" })).unwrap()
        ));
        let pg_uuids: std::collections::HashSet<String> = match sqlx::query_scalar(
            "SELECT msg_uuid FROM messages
             WHERE msg_uuid IS NOT NULL AND msg_uuid != ''
               AND content IS NOT NULL AND content != ''",
        )
        .fetch_all(&db)
        .await
        {
            Ok(v) => v.into_iter().collect(),
            Err(e) => {
                yield Ok(Event::default().data(
                    serde_json::to_string(&json!({ "type": "error", "message": e.to_string() })).unwrap()
                ));
                return;
            }
        };
        let total_postgres = pg_uuids.len() as i64;
        yield Ok(Event::default().data(
            serde_json::to_string(&json!({ "type": "pg_done", "count": total_postgres })).unwrap()
        ));

        // ── Step 3: collect all vector store IDs ──────────────────────────────
        let mut all_vec_ids: Vec<String> = Vec::new();
        if is_qdrant {
            let scroll_url = format!("{}/collections/{}/points/scroll", qdrant_url, qdrant_collection);
            let qdrant_client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new());
            let mut next_offset = serde_json::Value::Null;
            loop {
                let body = json!({
                    "limit": 10000,
                    "offset": next_offset,
                    "with_payload": false,
                    "with_vectors": false,
                });
                let mut scroll_req = qdrant_client.post(&scroll_url).json(&body);
                if let Some(key) = &qdrant_api_key { scroll_req = scroll_req.header("api-key", key); }
                let resp: Value = match scroll_req.send().await {
                    Ok(r) if r.status().is_success() => match r.json().await {
                        Ok(v) => v,
                        Err(_) => break,
                    },
                    _ => break,
                };
                let points = resp["result"]["points"].as_array().map(|a| a.to_vec()).unwrap_or_default();
                let count = points.len();
                for p in &points {
                    if let Some(id) = p["id"].as_str() {
                        all_vec_ids.push(id.to_string());
                    }
                }
                yield Ok(Event::default().data(
                    serde_json::to_string(&json!({ "type": "chroma_page", "fetched": all_vec_ids.len() })).unwrap()
                ));
                next_offset = resp["result"]["next_page_offset"].clone();
                if next_offset.is_null() || count == 0 { break; }
            }
        } else {
            let page_size = 10_000usize;
            let mut offset = 0usize;
            loop {
                let page_resp = chroma_client
                    .post(format!("{}/collections/{}/get", chroma_base, chroma_collection_id))
                    .json(&json!({ "limit": page_size, "offset": offset, "include": [] }))
                    .send()
                    .await;
                let page_ids: Vec<String> = match page_resp {
                    Ok(r) if r.status().is_success() => r
                        .json::<Value>().await.ok()
                        .and_then(|v| v["ids"].as_array().map(|arr| {
                            arr.iter().filter_map(|x| x.as_str().map(String::from)).collect()
                        }))
                        .unwrap_or_default(),
                    _ => break,
                };
                let count = page_ids.len();
                all_vec_ids.extend(page_ids);
                yield Ok(Event::default().data(
                    serde_json::to_string(&json!({ "type": "chroma_page", "fetched": all_vec_ids.len() })).unwrap()
                ));
                if count < page_size { break; }
                offset += page_size;
            }
        }

        let total_vec = all_vec_ids.len() as i64;
        let vec_set: std::collections::HashSet<String> = all_vec_ids.iter().cloned().collect();
        let orphans: Vec<String> = all_vec_ids.into_iter().filter(|id| !pg_uuids.contains(id)).collect();
        let orphan_count = orphans.len() as i64;
        let un_embedded: i64 = pg_uuids.iter().filter(|id| !vec_set.contains(*id)).count() as i64;

        // ── Step 4: delete orphans ─────────────────────────────────────────────
        let qdrant_client2 = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        let mut orphans_deleted: i64 = 0;
        for chunk in orphans.chunks(500) {
            let deleted = if is_qdrant {
                let mut del_req = qdrant_client2.post(&qdrant_delete_url)
                    .json(&json!({ "points": chunk }));
                if let Some(key) = &qdrant_api_key { del_req = del_req.header("api-key", key); }
                del_req.send().await.map(|r| r.status().is_success()).unwrap_or(false)
            } else {
                chroma_client
                    .post(format!("{}/collections/{}/delete", chroma_base, chroma_collection_id))
                    .json(&json!({ "ids": chunk }))
                    .send().await
                    .map(|r| r.status().is_success())
                    .unwrap_or(false)
            };
            if deleted { orphans_deleted += chunk.len() as i64; }
            yield Ok(Event::default().data(
                serde_json::to_string(&json!({
                    "type": "delete_progress",
                    "deleted": orphans_deleted,
                    "total_orphans": orphan_count,
                })).unwrap()
            ));
        }

        yield Ok(Event::default().data(
            serde_json::to_string(&json!({
                "type": "done",
                "total_postgres": total_postgres,
                "total_chroma": total_vec,
                "orphans_deleted": orphans_deleted,
                "un_embedded": un_embedded,
            })).unwrap()
        ));
    };

    Ok(Sse::new(event_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    ))
}

pub async fn get_job(Path(job_id): Path<String>) -> (StatusCode, Json<Value>) {
    (
        StatusCode::NOT_FOUND,
        Json(json!({ "error": "No active embedding jobs", "job_id": job_id })),
    )
}
