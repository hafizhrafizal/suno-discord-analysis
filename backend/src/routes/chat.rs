use axum::{
    extract::State,
    http::HeaderMap,
    response::{
        sse::{Event, Sse},
        IntoResponse,
    },
    Json,
};

fn key_from_header(headers: &HeaderMap) -> Option<String> {
    headers.get("x-openai-key")
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
        .map(String::from)
}
use serde_json::json;
use std::convert::Infallible;
use crate::{
    error::AppError,
    models::{
        AuthUser, ChatRequest, SummarizeFollowupRequest, SummarizeMsg, SummarizeRequest,
        SummarizeResultsFollowupRequest, SummarizeResultsRequest, UserProfileFollowupRequest,
        UserProfileRequest,
    },
    prompts,
    state::AppState,
};

fn format_messages_as_context(
    rows: &[serde_json::Value],
) -> String {
    rows.iter()
        .enumerate()
        .map(|(i, m)| {
            let username = m["username"].as_str().unwrap_or("unknown");
            let date = m["date"].as_str().unwrap_or("");
            let content = m["content"].as_str().unwrap_or("");
            format!("[{}] {} ({}): {}", i + 1, username, date, content)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_messages_as_context_slim(msgs: &[SummarizeMsg]) -> String {
    msgs.iter()
        .enumerate()
        .map(|(i, m)| format!("[{}] {} ({}): {}", i + 1, m.username, m.date, m.content))
        .collect::<Vec<_>>()
        .join("\n")
}

fn log_event(step: &str, label: &str, message: &str) -> Event {
    Event::default().data(
        serde_json::to_string(&json!({
            "type": "log",
            "step": step,
            "label": label,
            "message": message,
        }))
        .unwrap(),
    )
}

fn dedup_messages(msgs: Vec<SummarizeMsg>) -> (Vec<SummarizeMsg>, usize) {
    let mut seen = std::collections::HashSet::new();
    let before = msgs.len();
    let deduped = msgs
        .into_iter()
        .filter(|m| seen.insert(m.content.trim().to_lowercase()))
        .collect::<Vec<_>>();
    let removed = before - deduped.len();
    (deduped, removed)
}

/// Convert any msg_uuid to a Qdrant-compatible UUID string (mirrors uploads.rs logic).
fn to_qdrant_id(id: &str) -> String {
    if uuid::Uuid::parse_str(id).is_ok() {
        id.to_string()
    } else {
        uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, id.as_bytes()).to_string()
    }
}

/// Fetch embeddings from Qdrant for the given msg_uuids.
/// Converts IDs to Qdrant UUID form for the lookup, then reads original msg_uuid
/// back from the point payload so the returned keys match the database values.
async fn fetch_embeddings_qdrant(
    qdrant_url: &str,
    collection: &str,
    api_key: Option<&str>,
    uuids: &[String],
) -> Option<Vec<(String, Vec<f32>)>> {
    let client = reqwest::Client::new();
    // /points/get is the correct endpoint for fetching points by ID
    let url = format!("{}/collections/{}/points/get", qdrant_url, collection);
    let qdrant_ids: Vec<String> = uuids.iter().map(|id| to_qdrant_id(id)).collect();
    let mut req = client.post(&url).json(&serde_json::json!({
        "ids": qdrant_ids,
        "with_vectors": true,
        "with_payload": true,  // needed to read back original msg_uuid
    }));
    if let Some(key) = api_key { req = req.header("api-key", key); }
    let resp = req.send().await.ok()?;
    if !resp.status().is_success() { return None; }
    let body: serde_json::Value = resp.json().await.ok()?;
    let points = body["result"].as_array()?;
    let result: Vec<(String, Vec<f32>)> = points
        .iter()
        .filter_map(|p| {
            // Use payload.msg_uuid (original ID) as the map key so callers can
            // look up by the same msg_uuid values they have from the database
            let id = p["payload"]["msg_uuid"].as_str()
                .or_else(|| p["id"].as_str())
                .map(String::from)?;
            let vec: Vec<f32> = p["vector"]
                .as_array()?
                .iter()
                .filter_map(|v| v.as_f64().map(|x| x as f32))
                .collect();
            if vec.is_empty() { None } else { Some((id, vec)) }
        })
        .collect();
    Some(result)
}

async fn retrieve_keyword_messages(
    state: &AppState,
    query: &str,
    limit: i64,
) -> Vec<serde_json::Value> {
    if query.is_empty() {
        return vec![];
    }
    let rows = sqlx::query(
        "SELECT id, msg_uuid, username, date, content
         FROM messages
         WHERE search_vector @@ plainto_tsquery('simple', $1)
         ORDER BY date DESC
         LIMIT $2",
    )
    .bind(query)
    .bind(limit)
    .fetch_all(&state.db)
    .await
    .unwrap_or_default();

    rows.into_iter()
        .map(|r| {
            use sqlx::Row;
            json!({
                "id": r.get::<i64, _>("id"),
                "username": r.get::<String, _>("username"),
                "date": r.get::<String, _>("date"),
                "content": r.get::<String, _>("content"),
            })
        })
        .collect()
}

async fn call_openai_stream(
    api_key: &str,
    system_prompt: &str,
    user_message: &str,
    model: &str,
) -> Result<impl futures::Stream<Item = Result<Event, Infallible>>, AppError> {
    let client = reqwest::Client::new();
    let body = json!({
        "model": model,
        "stream": true,
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_message }
        ]
    });

    let resp = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await
        .map_err(|e| AppError::Internal(format!("OpenAI request failed: {}", e)))?;

    if !resp.status().is_success() {
        let err_text = resp.text().await.unwrap_or_default();
        return Err(AppError::Internal(format!("OpenAI error: {}", err_text)));
    }

    use futures::StreamExt;
    let byte_stream = resp.bytes_stream();

    let event_stream = async_stream::stream! {
        let mut buf = String::new();
        tokio::pin!(byte_stream);

        while let Some(chunk) = byte_stream.next().await {
            let bytes = match chunk {
                Ok(b) => b,
                Err(_) => break,
            };
            buf.push_str(&String::from_utf8_lossy(&bytes));
            let parts: Vec<&str> = buf.split("\n\n").collect();
            let last = parts.last().map(|s| s.to_string()).unwrap_or_default();

            for part in &parts[..parts.len().saturating_sub(1)] {
                let line = part.trim();
                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" {
                        yield Ok(Event::default().data(
                            serde_json::to_string(&json!({"type":"done","sources":[]})).unwrap()
                        ));
                        return;
                    }
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(content) = parsed["choices"][0]["delta"]["content"].as_str() {
                            if !content.is_empty() {
                                yield Ok(Event::default().data(
                                    serde_json::to_string(&json!({"type":"chunk","content": content})).unwrap()
                                ));
                            }
                        }
                    }
                }
            }
            buf = last;
        }

        yield Ok(Event::default().data(
            serde_json::to_string(&json!({"type":"done","sources":[]})).unwrap()
        ));
    };

    Ok(event_stream)
}

pub async fn chat(
    State(state): State<AppState>,
    user: AuthUser,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Result<impl IntoResponse, AppError> {
    let api_key = state
        .get_openai_key(key_from_header(&headers))
        .await
        .ok_or_else(|| AppError::BadRequest("OpenAI API key not set".into()))?;

    let messages = retrieve_keyword_messages(&state, &req.query, 10).await;
    let context = format_messages_as_context(&messages);

    let system = req
        .system_prompt
        .clone()
        .unwrap_or_else(|| prompts::CHAT.replace("{context}", &context));

    let event_stream = call_openai_stream(&api_key, &system, &req.query, "gpt-4o").await?;
    Ok(Sse::new(event_stream))
}

pub async fn summarize(
    State(state): State<AppState>,
    user: AuthUser,
    headers: HeaderMap,
    Json(req): Json<SummarizeRequest>,
) -> Result<impl IntoResponse, AppError> {
    let api_key = state
        .get_openai_key(key_from_header(&headers))
        .await
        .ok_or_else(|| AppError::BadRequest("OpenAI API key not set".into()))?;

    let messages = retrieve_keyword_messages(&state, &req.query, 20).await;
    let context = format_messages_as_context(&messages);
    let system = prompts::SUMMARIZE.replace("{context}", &context);

    let event_stream = call_openai_stream(&api_key, &system, &req.query, "gpt-4o").await?;
    Ok(Sse::new(event_stream))
}

pub async fn summarize_followup(
    State(state): State<AppState>,
    user: AuthUser,
    headers: HeaderMap,
    Json(req): Json<SummarizeFollowupRequest>,
) -> Result<impl IntoResponse, AppError> {
    let api_key = state
        .get_openai_key(key_from_header(&headers))
        .await
        .ok_or_else(|| AppError::BadRequest("OpenAI API key not set".into()))?;

    let messages = retrieve_keyword_messages(&state, &req.query, 10).await;
    let context = format_messages_as_context(&messages);
    let system = prompts::SUMMARIZE_FOLLOWUP
        .replace("{context}", &context)
        .replace("{previous_summary}", &req.previous_summary);

    let event_stream = call_openai_stream(&api_key, &system, &req.query, "gpt-4o").await?;
    Ok(Sse::new(event_stream))
}

pub async fn summarize_results(
    State(state): State<AppState>,
    user: AuthUser,
    headers: HeaderMap,
    Json(req): Json<SummarizeResultsRequest>,
) -> Result<impl IntoResponse, AppError> {
    let api_key = state
        .get_openai_key(key_from_header(&headers))
        .await
        .ok_or_else(|| AppError::BadRequest("OpenAI API key not set".into()))?;

    let query = req.query.or(req.prompt).unwrap_or_default();
    let model = req.model.unwrap_or_else(|| "gpt-4o".to_string());
    let retrieval_mode = req.retrieval_mode.unwrap_or_else(|| "cluster".to_string());
    let input_messages = req.messages;
    let qdrant_url = state.config.qdrant_url.clone();
    let qdrant_collection = state.config.qdrant_collection.clone();
    let qdrant_api_key = state.config.qdrant_api_key.clone();

    let stream = async_stream::stream! {
        use futures::StreamExt;
        use crate::hdbscan;

        // ── Step 1: Filter ──────────────────────────────────────────────────
        let total = input_messages.len();
        yield Ok(log_event("filter", "Filter", &format!("{} messages received", total)));
        if total == 0 {
            yield Ok(log_event("fallback", "Warning", "No messages to analyse"));
            return;
        }

        // ── Step 2: Dedup ───────────────────────────────────────────────────
        let (deduped, removed) = dedup_messages(input_messages);
        if removed > 0 {
            yield Ok(log_event("dedup", "Dedup",
                &format!("Removed {} duplicates — {} unique messages remain", removed, deduped.len())));
        } else {
            yield Ok(log_event("dedup", "Dedup",
                &format!("{} messages (no duplicates)", deduped.len())));
        }

        // ── Step 3: Cluster or all ──────────────────────────────────────────
        let final_msgs: Vec<SummarizeMsg> = if retrieval_mode == "cluster" {
            let n = deduped.len();
            let min_cs = 3usize.min(n / 4).max(2);

            // Try to fetch embeddings from ChromaDB
            let uuids: Vec<String> = deduped.iter()
                .filter_map(|m| m.msg_uuid.clone())
                .collect();
            let has_uuids = !uuids.is_empty();

            if has_uuids {
                yield Ok(log_event("retrieval", "Retrieval",
                    &format!("Fetching embeddings for {} messages from Qdrant…", uuids.len())));
            }

            let embedding_map: std::collections::HashMap<String, Vec<f32>> = if has_uuids {
                match fetch_embeddings_qdrant(
                    &qdrant_url, &qdrant_collection, qdrant_api_key.as_deref(), &uuids,
                ).await {
                    Some(pairs) => pairs.into_iter().collect(),
                    None => {
                        yield Ok(log_event("fallback", "Warning",
                            "Could not fetch embeddings from Qdrant — falling back to word-count sampling"));
                        std::collections::HashMap::new()
                    }
                }
            } else {
                yield Ok(log_event("fallback", "Warning",
                    "No UUIDs in messages — falling back to word-count sampling"));
                std::collections::HashMap::new()
            };

            // Build embedding matrix aligned to deduped
            let embeddings: Vec<Option<Vec<f32>>> = deduped.iter()
                .map(|m| m.msg_uuid.as_deref().and_then(|u| embedding_map.get(u)).cloned())
                .collect();

            let embedded_indices: Vec<usize> = embeddings.iter().enumerate()
                .filter_map(|(i, e)| e.as_ref().map(|_| i))
                .collect();
            let embedded_vecs: Vec<Vec<f32>> = embedded_indices.iter()
                .filter_map(|&i| embeddings[i].clone())
                .collect();

            if embedded_vecs.len() >= min_cs * 2 {
                // ── HDBSCAN ────────────────────────────────────────────────
                yield Ok(log_event("cluster", "Cluster",
                    &format!("Running HDBSCAN on {} embedded messages (min_cluster_size={})",
                        embedded_vecs.len(), min_cs)));

                let labels = hdbscan::hdbscan(&embedded_vecs, min_cs);
                let n_clusters = {
                    let mut ids: Vec<i32> = labels.iter().filter(|&&l| l >= 0).cloned().collect();
                    ids.sort_unstable();
                    ids.dedup();
                    ids.len()
                };
                let n_noise = labels.iter().filter(|&&l| l < 0).count();
                yield Ok(log_event("cluster", "Cluster",
                    &format!("Found {} clusters ({} noise points)", n_clusters, n_noise)));

                // ── Sample: closest + furthest from each cluster centroid ──
                let mut sampled_embedded_indices: Vec<usize> = Vec::new();
                let max_label = labels.iter().filter(|&&l| l >= 0).copied().max().unwrap_or(-1);
                for cluster_id in 0..=(max_label as usize) {
                    let members: Vec<usize> = labels.iter().enumerate()
                        .filter(|(_, &l)| l == cluster_id as i32)
                        .map(|(local_i, _)| local_i)
                        .collect();
                    if members.is_empty() { continue; }
                    if let Some((closest, furthest)) = hdbscan::closest_and_furthest(&embedded_vecs, &members) {
                        sampled_embedded_indices.push(closest);
                        if furthest != closest {
                            sampled_embedded_indices.push(furthest);
                        }
                    }
                }
                // Also include noise points (1 per noise, capped)
                let noise_local: Vec<usize> = labels.iter().enumerate()
                    .filter(|(_, &l)| l < 0)
                    .map(|(i, _)| i)
                    .take(5)
                    .collect();
                sampled_embedded_indices.extend(noise_local);
                sampled_embedded_indices.sort_unstable();
                sampled_embedded_indices.dedup();

                yield Ok(log_event("sample", "Sample",
                    &format!("Selected {} messages (closest + furthest from each cluster centroid)",
                        sampled_embedded_indices.len())));

                // Map local embedded indices → indices in deduped
                sampled_embedded_indices.iter()
                    .map(|&local_i| deduped[embedded_indices[local_i]].clone())
                    .collect()
            } else {
                // Not enough embeddings — fall back to word-count top-N per time window
                yield Ok(log_event("fallback", "Warning",
                    "Too few embeddings for HDBSCAN — using word-count sampling"));
                let cap = 60usize;
                let mut msgs = deduped.clone();
                msgs.sort_by(|a, b| a.date.cmp(&b.date));
                let n_buckets = ((msgs.len() / 5).max(2)).min(10);
                let bucket_sz = (msgs.len() + n_buckets - 1) / n_buckets;
                let per_bucket = ((cap + n_buckets - 1) / n_buckets).max(1);
                let mut result = Vec::new();
                for bucket in msgs.chunks(bucket_sz) {
                    let mut b: Vec<&SummarizeMsg> = bucket.iter().collect();
                    b.sort_by_key(|m| std::cmp::Reverse(m.content.split_whitespace().count()));
                    result.extend(b.into_iter().take(per_bucket).cloned());
                }
                result
            }
        } else {
            // "all" mode
            let cap = 200usize;
            let n = deduped.len().min(cap);
            if deduped.len() > cap {
                yield Ok(log_event("retrieval", "Retrieval",
                    &format!("Using all messages (capped at {})", cap)));
            } else {
                yield Ok(log_event("retrieval", "Retrieval",
                    &format!("Using all {} messages as context", n)));
            }
            deduped.into_iter().take(cap).collect()
        };

        // ── Step 4: LLM ─────────────────────────────────────────────────────
        let n_final = final_msgs.len();
        yield Ok(log_event("llm", "LLM",
            &format!("Sending {} messages to {} for analysis…", n_final, model)));

        let context = format_messages_as_context_slim(&final_msgs);
        let prompt = if query.is_empty() {
            "Summarize the key themes, notable discussions, and important insights from these messages.".to_string()
        } else {
            query.clone()
        };
        let system = prompts::SUMMARIZE.replace("{context}", &context);

        match call_openai_stream(&api_key, &system, &prompt, &model).await {
            Ok(llm_stream) => {
                tokio::pin!(llm_stream);
                while let Some(ev) = llm_stream.next().await {
                    yield ev;
                }
            }
            Err(e) => {
                yield Ok(log_event("fallback", "Error", &e.to_string()));
            }
        }
    };

    Ok(Sse::new(stream))
}

pub async fn summarize_results_followup(
    State(state): State<AppState>,
    user: AuthUser,
    headers: HeaderMap,
    Json(req): Json<SummarizeResultsFollowupRequest>,
) -> Result<impl IntoResponse, AppError> {
    let api_key = state
        .get_openai_key(key_from_header(&headers))
        .await
        .ok_or_else(|| AppError::BadRequest("OpenAI API key not set".into()))?;

    let query = req.query.or(req.question).unwrap_or_default();
    let previous_summary = req.previous_summary.or(req.summary).unwrap_or_default();

    let context = if let Some(msgs) = &req.messages {
        format_messages_as_context_slim(msgs)
    } else {
        String::new()
    };
    let system = prompts::SUMMARIZE_FOLLOWUP
        .replace("{context}", &context)
        .replace("{previous_summary}", &previous_summary);

    let event_stream = call_openai_stream(&api_key, &system, &query, "gpt-4o").await?;
    Ok(Sse::new(event_stream))
}

pub async fn user_profile(
    State(state): State<AppState>,
    user: AuthUser,
    headers: HeaderMap,
    Json(req): Json<UserProfileRequest>,
) -> Result<impl IntoResponse, AppError> {
    let api_key = state
        .get_openai_key(key_from_header(&headers))
        .await
        .ok_or_else(|| AppError::BadRequest("OpenAI API key not set".into()))?;

    let rows = sqlx::query(
        "SELECT username, date, content FROM messages WHERE username = ? ORDER BY date ASC LIMIT 50",
    )
    .bind(&req.username)
    .fetch_all(&state.db)
    .await
    .unwrap_or_default();

    let msgs_json: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            use sqlx::Row;
            json!({
                "username": r.get::<String, _>("username"),
                "date": r.get::<String, _>("date"),
                "content": r.get::<String, _>("content"),
            })
        })
        .collect();

    let context = format_messages_as_context(&msgs_json);
    let system = prompts::USER_PROFILE
        .replace("{username}", &req.username)
        .replace("{context}", &context);
    let user_msg = req
        .query
        .as_deref()
        .unwrap_or("Please analyze this user's profile.");

    let event_stream = call_openai_stream(&api_key, &system, user_msg, "gpt-4o").await?;
    Ok(Sse::new(event_stream))
}

pub async fn user_profile_followup(
    State(state): State<AppState>,
    user: AuthUser,
    headers: HeaderMap,
    Json(req): Json<UserProfileFollowupRequest>,
) -> Result<impl IntoResponse, AppError> {
    let api_key = state
        .get_openai_key(key_from_header(&headers))
        .await
        .ok_or_else(|| AppError::BadRequest("OpenAI API key not set".into()))?;

    let system = prompts::USER_PROFILE_FOLLOWUP
        .replace("{previous_profile}", &req.previous_profile);

    let event_stream = call_openai_stream(&api_key, &system, &req.query, "gpt-4o").await?;
    Ok(Sse::new(event_stream))
}
