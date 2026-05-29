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

    let event_stream = async_stream::stream! {
        // ── Phase 1: insert rows ──────────────────────────────────────────────
        let mut processed: i64 = 0; // rows seen (used for row_index ordering)
        let mut inserted: i64 = 0;  // rows actually inserted (not duplicate-skipped)
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
                // Fetch all msg_uuid + content for this upload
                let msgs: Vec<(String, String)> = match sqlx::query(
                    "SELECT msg_uuid, content FROM messages
                     WHERE upload_id = $1 AND content IS NOT NULL
                     ORDER BY row_index",
                )
                .bind(&upload_id_clone)
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
                    let chroma_base = format!(
                        "http://{}:{}/api/v2/tenants/default_tenant/databases/default_database",
                        chroma_host, chroma_port
                    );
                    // One client shared across all ChromaDB and OpenAI calls — TLS connections are reused
                    let chroma_client = reqwest::Client::builder()
                        .timeout(std::time::Duration::from_secs(120))
                        .build()
                        .unwrap_or_else(|_| reqwest::Client::new());
                    let embed_client = chroma_client.clone();

                    let collection_id_opt = get_or_create_chroma_collection(
                        &chroma_client, &chroma_base, &chroma_collection,
                    ).await;

                    match collection_id_opt {
                        None => {
                            tracing::warn!("ChromaDB unavailable — embedding skipped");
                            yield Ok(Event::default().data(
                                serde_json::to_string(&json!({
                                    "type": "embed_skip",
                                    "reason": "chroma_unavailable",
                                })).unwrap()
                            ));
                        }
                        Some(collection_id) => {
                            // Check which IDs already have vectors — enables resumability
                            let all_ids: Vec<String> = msgs.iter().map(|(id, _)| id.clone()).collect();
                            let existing_ids = get_existing_chroma_ids(
                                &chroma_client, &chroma_base, &collection_id, &all_ids,
                            ).await;
                            already_in_chroma = existing_ids.len() as i64;

                            // Only embed messages that are not yet in ChromaDB
                            let msgs_to_embed: Vec<(String, String)> = msgs
                                .into_iter()
                                .filter(|(id, _)| !existing_ids.contains(id))
                                .collect();
                            let total_to_embed = msgs_to_embed.len() as i64;

                            if total_to_embed == 0 {
                                tracing::info!(
                                    "All {} vectors already in ChromaDB — embedding skipped",
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

                                // Pre-build all batches so we can window over them
                                let all_batches: Vec<(Vec<String>, Vec<String>)> = msgs_to_embed
                                    .chunks(EMBED_BATCH_SIZE)
                                    .map(|chunk| (
                                        chunk.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>(),
                                        chunk.iter().map(|(_, c)| c.clone()).collect::<Vec<_>>(),
                                    ))
                                    .collect();

                                for window in all_batches.chunks(EMBED_CONCURRENCY) {
                                    // Embed EMBED_CONCURRENCY batches in parallel, sharing one HTTP client
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

                                    // Upsert each batch's results in parallel
                                    let upsert_futs = window_results.iter().map(|valid| {
                                        let chroma_client = chroma_client.clone();
                                        let chroma_base = chroma_base.clone();
                                        let collection_id = collection_id.clone();
                                        let upload_id_inner = upload_id_clone.clone();
                                        let valid = valid.clone();
                                        async move {
                                            if valid.is_empty() { return 0i64; }
                                            let v_ids: Vec<String> = valid.iter().map(|(id, _, _)| id.clone()).collect();
                                            let v_embs: Vec<Vec<f64>> = valid.iter().map(|(_, emb, _)| emb.clone()).collect();
                                            let v_docs: Vec<String> = valid.iter().map(|(_, _, doc)| doc.clone()).collect();
                                            upsert_chroma_batch(&chroma_client, &chroma_base, &collection_id, &v_ids, &v_embs, &v_docs, &upload_id_inner).await;
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

                            // Record in embedded_uploads if any vectors exist (new or pre-existing)
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
    // Count messages before deletion so we can report it
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
    Ok(Json(json!({ "deleted": id, "deleted_messages": deleted_messages })))
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
    let affected = sqlx::query("DELETE FROM embedded_uploads WHERE upload_id = $1")
        .bind(&id)
        .execute(&state.db)
        .await?
        .rows_affected();
    Ok(Json(json!({
        "deleted_embeddings": affected,
        "note": "ChromaDB vectors for this upload remain; re-embed to regenerate.",
    })))
}

pub async fn reembed(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>> {
    // Check upload exists
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

    let chroma_base = format!(
        "http://{}:{}/api/v2/tenants/default_tenant/databases/default_database",
        state.config.chroma_host, state.config.chroma_port
    );
    let chroma_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| AppError::Internal(e.to_string()))?;
    let embed_client = chroma_client.clone();

    let collection_id = get_or_create_chroma_collection(
        &chroma_client, &chroma_base, &state.config.chroma_collection,
    )
    .await
    .ok_or_else(|| AppError::Internal("ChromaDB unavailable".into()))?;

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
    let mut embedded: i64 = 0;

    // Skip vectors already in ChromaDB so re-embed is resumable
    let all_ids: Vec<String> = msgs.iter().map(|(mid, _)| mid.clone()).collect();
    let existing_ids = get_existing_chroma_ids(&chroma_client, &chroma_base, &collection_id, &all_ids).await;
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
            async move {
                if valid.is_empty() { return 0i64; }
                let v_ids: Vec<String> = valid.iter().map(|(vid, _, _)| vid.clone()).collect();
                let v_embs: Vec<Vec<f64>> = valid.iter().map(|(_, emb, _)| emb.clone()).collect();
                let v_docs: Vec<String> = valid.iter().map(|(_, _, doc)| doc.clone()).collect();
                upsert_chroma_batch(&chroma_client, &chroma_base, &collection_id, &v_ids, &v_embs, &v_docs, &upload_id_inner).await;
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

pub async fn get_job(Path(job_id): Path<String>) -> (StatusCode, Json<Value>) {
    (
        StatusCode::NOT_FOUND,
        Json(json!({ "error": "No active embedding jobs", "job_id": job_id })),
    )
}
