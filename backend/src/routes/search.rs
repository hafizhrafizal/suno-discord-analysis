use axum::{
    extract::{Query, State},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::atomic::Ordering;
use crate::{error::{AppError, Result}, models::{AuthUser, BulkContextRequest}, state::AppState};

/// Build an FTS5 MATCH string from user input. Each word is double-quoted
/// (to prevent FTS5 operator injection) and suffixed with `*` for prefix matching.
fn build_fts_match(q: &str, match_type: &str) -> String {
    let tokens: Vec<String> = q
        .split_whitespace()
        .filter(|w| !w.is_empty())
        .map(|w| format!("\"{}\"*", w.replace('"', "\"\"")))
        .collect();

    match match_type {
        "exact" => format!("\"{}\"", q.replace('"', "\"\"")),
        "any_word" => {
            if tokens.is_empty() { String::new() } else { tokens.join(" OR ") }
        }
        _ => tokens.join(" "), // fuzzy: all word-prefixes must be present (implicit AND)
    }
}

#[derive(Debug, Deserialize)]
pub struct KeywordParams {
    pub q: Option<String>,
    pub limit: Option<i64>,
    pub upload_ids: Option<String>,
    pub date_from: Option<String>,
    pub date_to: Option<String>,
    pub is_suno_team: Option<String>,
    pub min_words: Option<i64>,
    pub match_type: Option<String>, // "fuzzy" | "exact" | "any_word"
    pub username: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UsernameParams {
    pub q: Option<String>,
    pub limit: Option<i64>,
    pub date_from: Option<String>,
    pub date_to: Option<String>,
    pub is_suno_team: Option<String>,
    pub min_words: Option<i64>,
    pub upload_ids: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RangeParams {
    pub date_from: Option<String>,
    pub date_to: Option<String>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
    pub upload_ids: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UserRangeParams {
    pub date_from: Option<String>,
    pub date_to: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UserMessagesParams {
    pub username: Option<String>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

pub async fn keyword_search(
    State(state): State<AppState>,
    Query(params): Query<KeywordParams>,
) -> Result<Json<Value>> {
    let q = params.q.unwrap_or_default().trim().to_string();
    if q.is_empty() {
        return Ok(Json(json!([])));
    }
    let limit = params.limit.unwrap_or(200).min(10_000);
    let match_type = params.match_type.as_deref().unwrap_or("fuzzy");
    let use_fts = state.fts_ready.load(Ordering::Relaxed);

    let mut sql: String;
    let mut args: Vec<String> = Vec::new();

    if use_fts {
        // FTS5 path — uses the inverted index, orders of magnitude faster than LIKE.
        let fts_match = build_fts_match(&q, match_type);
        sql = String::from(
            "SELECT m.id, m.msg_uuid, m.author_id, m.username, m.date, m.content,
                    m.attachments, m.reactions, m.is_suno_team, m.week, m.month,
                    m.upload_id, m.row_index
             FROM messages_fts
             JOIN messages m ON m.id = messages_fts.rowid
             WHERE messages_fts MATCH ?",
        );
        args.push(fts_match);
    } else {
        // LIKE fallback — used only while the background FTS rebuild is still running.
        let words: Vec<String> = q.split_whitespace()
            .filter(|w| !w.is_empty())
            .map(|w| w.to_string())
            .collect();

        sql = String::from(
            "SELECT m.id, m.msg_uuid, m.author_id, m.username, m.date, m.content,
                    m.attachments, m.reactions, m.is_suno_team, m.week, m.month,
                    m.upload_id, m.row_index
             FROM messages m
             WHERE 1=1",
        );

        match match_type {
            "exact" => {
                sql.push_str(" AND m.content LIKE ? ESCAPE '\\'");
                args.push(format!("%{}%", q.replace('%', "\\%").replace('_', "\\_")));
            }
            "any_word" => {
                if !words.is_empty() {
                    let clauses: Vec<String> = words.iter()
                        .map(|_| "m.content LIKE ? ESCAPE '\\'".to_string())
                        .collect();
                    sql.push_str(&format!(" AND ({})", clauses.join(" OR ")));
                    for w in &words {
                        args.push(format!("%{}%", w.replace('%', "\\%").replace('_', "\\_")));
                    }
                }
            }
            _ => {
                for w in &words {
                    sql.push_str(" AND m.content LIKE ? ESCAPE '\\'");
                    args.push(format!("%{}%", w.replace('%', "\\%").replace('_', "\\_")));
                }
            }
        }
    }

    // Common filters — same column references work in both FTS5 JOIN and plain messages paths.
    if let Some(ref df) = params.date_from {
        sql.push_str(" AND m.date >= ?");
        args.push(df.clone());
    }
    if let Some(ref dt) = params.date_to {
        sql.push_str(" AND m.date <= ?");
        args.push(dt.clone());
    }
    if let Some(ref st) = params.is_suno_team {
        match st.as_str() {
            "true" | "1" | "only"     => sql.push_str(" AND m.is_suno_team IN ('true','1')"),
            "exclude" | "false" | "0" => sql.push_str(" AND (m.is_suno_team IS NULL OR m.is_suno_team NOT IN ('true','1'))"),
            _ => {}
        }
    }
    if let Some(mw) = params.min_words {
        if mw > 0 {
            sql.push_str(&format!(
                " AND (length(m.content) - length(replace(m.content,' ','')) + 1) >= {}",
                mw
            ));
        }
    }
    if let Some(ref un) = params.username {
        if !un.is_empty() {
            sql.push_str(" AND m.username LIKE ?");
            args.push(format!("%{}%", un));
        }
    }
    if let Some(ref uid_str) = params.upload_ids {
        let ids: Vec<&str> = uid_str.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        if !ids.is_empty() {
            let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            sql.push_str(&format!(" AND m.upload_id IN ({})", placeholders));
            for id in ids {
                args.push(id.to_string());
            }
        }
    }

    sql.push_str(" ORDER BY m.date DESC LIMIT ?");
    args.push(limit.to_string());

    let mut q_builder = sqlx::query(&sql);
    for arg in &args {
        q_builder = q_builder.bind(arg);
    }

    let rows = q_builder.fetch_all(&state.db).await?;

    use sqlx::Row;
    let messages: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            json!({
                "id": r.get::<i64, _>("id"),
                "msg_uuid": r.get::<String, _>("msg_uuid"),
                "author_id": r.get::<Option<String>, _>("author_id"),
                "username": r.get::<String, _>("username"),
                "date": r.get::<String, _>("date"),
                "content": r.get::<String, _>("content"),
                "attachments": r.get::<Option<String>, _>("attachments"),
                "reactions": r.get::<Option<String>, _>("reactions"),
                "is_suno_team": r.get::<Option<String>, _>("is_suno_team"),
                "upload_id": r.get::<Option<String>, _>("upload_id"),
                "row_index": r.get::<Option<i64>, _>("row_index"),
            })
        })
        .collect();

    Ok(Json(json!(messages)))
}

/// Returns true when the query reads like a question or open-ended request.
fn is_question_query(q: &str) -> bool {
    let lower = q.trim().to_lowercase();
    if lower.ends_with('?') { return true; }
    let starters = [
        "what ", "how ", "why ", "when ", "who ", "which ", "where ",
        "is ", "are ", "do ", "does ", "can ", "could ", "should ", "would ",
        "will ", "has ", "have ", "had ", "did ", "was ", "were ",
        "tell me", "explain ", "describe ", "find ", "show ", "list ",
    ];
    starters.iter().any(|s| lower.starts_with(s))
}

fn normalize_embedding(v: Vec<f64>) -> Vec<f64> {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 { return v; }
    v.into_iter().map(|x| x / norm).collect()
}

/// Blend two unit-normalised embeddings with weight `w` on `b` (0.5 = equal weight).
fn blend_embeddings(a: &[f64], b: &[f64], w: f64) -> Vec<f64> {
    let blended: Vec<f64> = a.iter().zip(b.iter())
        .map(|(x, y)| x * (1.0 - w) + y * w)
        .collect();
    normalize_embedding(blended)
}

/// HyDE: ask GPT-4o-mini to write a hypothetical Discord reply, then embed it.
/// Returns None on any failure so the caller falls back to the raw query embedding.
async fn hyde_embed(api_key: &str, query: &str, embed_model: &str) -> Option<Vec<f64>> {
    let client = reqwest::Client::new();

    let chat: serde_json::Value = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&json!({
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are searching a Discord community about Suno AI (an AI music generator). \
                                Given a search query, write 2-3 short, realistic Discord messages (~80 words total) \
                                that would directly and relevantly discuss or answer the query. \
                                Write only the message text, no usernames or metadata."
                },
                { "role": "user", "content": query }
            ],
            "max_tokens": 200,
            "temperature": 0.2,
        }))
        .send().await.ok()?
        .json().await.ok()?;

    let hypo_doc = chat["choices"][0]["message"]["content"].as_str()?.to_string();
    tracing::debug!("HyDE document: {}", &hypo_doc[..hypo_doc.len().min(120)]);

    let embed: serde_json::Value = client
        .post("https://api.openai.com/v1/embeddings")
        .bearer_auth(api_key)
        .json(&json!({ "model": embed_model, "input": hypo_doc }))
        .send().await.ok()?
        .json().await.ok()?;

    let vec: Vec<f64> = embed["data"][0]["embedding"]
        .as_array()?
        .iter()
        .filter_map(|v| v.as_f64())
        .collect();

    if vec.is_empty() { None } else { Some(vec) }
}

async fn embed_query(api_key: &str, text: &str, model: &str) -> Result<Vec<f64>> {
    let client = reqwest::Client::new();
    let resp = client
        .post("https://api.openai.com/v1/embeddings")
        .bearer_auth(api_key)
        .json(&json!({ "model": model, "input": text }))
        .send()
        .await
        .map_err(|e| AppError::Internal(format!("OpenAI embed request failed: {}", e)))?;

    if !resp.status().is_success() {
        let err = resp.text().await.unwrap_or_default();
        return Err(AppError::Internal(format!("OpenAI embed error: {}", err)));
    }

    let data: Value = resp.json().await
        .map_err(|e| AppError::Internal(format!("OpenAI embed parse error: {}", e)))?;

    let embedding = data["data"][0]["embedding"]
        .as_array()
        .ok_or_else(|| AppError::Internal("No embedding in OpenAI response".into()))?
        .iter()
        .filter_map(|v| v.as_f64())
        .collect();

    Ok(embedding)
}

#[derive(Debug, Deserialize)]
pub struct SemanticParams {
    pub q: Option<String>,
    pub limit: Option<i64>,
    pub upload_ids: Option<String>,
    pub date_from: Option<String>,
    pub date_to: Option<String>,
    pub is_suno_team: Option<String>,
    pub min_words: Option<i64>,
    pub username: Option<String>,
    pub sort_by: Option<String>, // "similarity" | "date"
}

pub async fn semantic_search(
    State(state): State<AppState>,
    user: AuthUser,
    Query(params): Query<SemanticParams>,
) -> Result<Json<Value>> {
    let q = params.q.as_deref().unwrap_or("").trim().to_string();
    if q.is_empty() {
        return Ok(Json(json!([])));
    }
    let limit = params.limit.unwrap_or(20).min(200) as usize;

    let api_key = state
        .get_openai_key(Some(user.id))
        .await
        .ok_or_else(|| AppError::BadRequest("OpenAI API key not set".into()))?;

    let embed_model = state.get_embedding_model();

    // For question/open-ended queries, run HyDE concurrently with the raw embed and blend results.
    let is_question = is_question_query(&q);
    let embedding = if is_question {
        tracing::debug!("HyDE: question query detected — generating hypothetical document");
        let (raw_res, hyde_opt) = tokio::join!(
            embed_query(&api_key, &q, &embed_model),
            hyde_embed(&api_key, &q, &embed_model)
        );
        let raw = normalize_embedding(raw_res?);
        match hyde_opt {
            Some(hyde) => blend_embeddings(&raw, &normalize_embedding(hyde), 0.5),
            None => {
                tracing::warn!("HyDE failed — falling back to raw query embedding");
                raw
            }
        }
    } else {
        normalize_embedding(embed_query(&api_key, &q, &embed_model).await?)
    };

    let chroma_base = format!(
        "http://{}:{}/api/v2/tenants/default_tenant/databases/default_database",
        state.config.chroma_host, state.config.chroma_port
    );

    let client = reqwest::Client::new();

    // Step 1: resolve collection name → UUID (v2 requires UUID in query path)
    let collection_info_url = format!("{}/collections/{}", chroma_base, state.config.chroma_collection);
    let collection_resp = client.get(&collection_info_url).send().await;
    let collection_id = match collection_resp {
        Err(e) => {
            eprintln!("ChromaDB unreachable: {}", e);
            return Ok(Json(json!({ "error": "ChromaDB unavailable", "results": [] })));
        }
        Ok(r) if !r.status().is_success() => {
            let err = r.text().await.unwrap_or_default();
            eprintln!("ChromaDB collection lookup error: {}", err);
            return Ok(Json(json!({ "error": format!("ChromaDB error: {}", err), "results": [] })));
        }
        Ok(r) => {
            let info: Value = match r.json().await {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("ChromaDB collection parse error: {}", e);
                    return Ok(Json(json!({ "error": "ChromaDB response parse error", "results": [] })));
                }
            };
            match info["id"].as_str().map(String::from) {
                Some(id) => id,
                None => {
                    eprintln!("ChromaDB collection missing id field");
                    return Ok(Json(json!({ "error": "ChromaDB collection id missing", "results": [] })));
                }
            }
        }
    };

    // Step 2: overcollect (5× limit) so post-filtering still yields enough results
    let n_results = (limit * 5).min(500);
    let chroma_url = format!("{}/collections/{}/query", chroma_base, collection_id);
    let chroma_send = client
        .post(&chroma_url)
        .json(&json!({
            "query_embeddings": [embedding],
            "n_results": n_results,
            "include": ["metadatas", "distances"]
        }))
        .send()
        .await;

    let chroma_resp = match chroma_send {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ChromaDB unreachable: {}", e);
            return Ok(Json(json!({ "error": "ChromaDB unavailable", "results": [] })));
        }
    };

    if !chroma_resp.status().is_success() {
        let err = chroma_resp.text().await.unwrap_or_default();
        eprintln!("ChromaDB error response: {}", err);
        return Ok(Json(json!({ "error": format!("ChromaDB error: {}", err), "results": [] })));
    }

    let chroma_data: Value = match chroma_resp.json().await {
        Ok(v) => v,
        Err(e) => {
            eprintln!("ChromaDB parse error: {}", e);
            return Ok(Json(json!({ "error": "ChromaDB response parse error", "results": [] })));
        }
    };

    let empty_vec = vec![];
    let ids = chroma_data["ids"][0].as_array().unwrap_or(&empty_vec);
    if ids.is_empty() {
        return Ok(Json(json!([])));
    }

    // ids are msg_uuid values — fetch full messages from SQLite preserving similarity order
    let msg_uuids: Vec<String> = ids.iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();

    // ChromaDB cosine space: distance = 1 − cosine_similarity → similarity = 1 − distance
    let empty_dists = vec![];
    let distances = chroma_data["distances"][0].as_array().unwrap_or(&empty_dists);
    let uuid_to_similarity: std::collections::HashMap<String, f64> = msg_uuids.iter()
        .zip(distances.iter())
        .filter_map(|(uuid, dist)| {
            dist.as_f64().map(|d| (uuid.clone(), (1.0 - d).clamp(0.0, 1.0)))
        })
        .collect();

    // Adaptive threshold: floor at 0.30, but tighten relative to the best result quality.
    // This keeps high-quality sets focused while preserving recall on low-signal queries.
    let best_sim = uuid_to_similarity.values().cloned().fold(0.0_f64, f64::max);
    let min_similarity = 0.30_f64.max(best_sim * 0.65);
    tracing::debug!("Semantic: best_sim={:.3}, threshold={:.3}, hyde={}", best_sim, min_similarity, is_question);

    let placeholders = msg_uuids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT id, msg_uuid, author_id, username, date, content,
                attachments, reactions, is_suno_team, upload_id, row_index
         FROM messages WHERE msg_uuid IN ({})",
        placeholders
    );
    let mut q_builder = sqlx::query(&sql);
    for uuid in &msg_uuids {
        q_builder = q_builder.bind(uuid);
    }
    let rows = q_builder.fetch_all(&state.db).await?;

    use sqlx::Row;
    use std::collections::HashMap;
    let mut row_map: HashMap<String, Value> = rows
        .into_iter()
        .map(|r| {
            let uuid: String = r.get("msg_uuid");
            let similarity = uuid_to_similarity.get(&uuid).copied();
            let val = json!({
                "id": r.get::<i64, _>("id"),
                "msg_uuid": uuid.clone(),
                "author_id": r.get::<Option<String>, _>("author_id"),
                "username": r.get::<String, _>("username"),
                "date": r.get::<String, _>("date"),
                "content": r.get::<String, _>("content"),
                "attachments": r.get::<Option<String>, _>("attachments"),
                "reactions": r.get::<Option<String>, _>("reactions"),
                "is_suno_team": r.get::<Option<String>, _>("is_suno_team"),
                "upload_id": r.get::<Option<String>, _>("upload_id"),
                "row_index": r.get::<Option<i64>, _>("row_index"),
                "similarity": similarity,
            });
            (uuid, val)
        })
        .collect();

    // Filter by threshold + user params, then truncate to the requested limit
    let mut messages: Vec<Value> = msg_uuids
        .iter()
        .filter_map(|uuid| row_map.remove(uuid))
        .filter(|m| {
            if let Some(sim) = m["similarity"].as_f64() {
                if sim < min_similarity { return false; }
            }
            if let Some(ref df) = params.date_from {
                if m["date"].as_str().unwrap_or("") < df.as_str() { return false; }
            }
            if let Some(ref dt) = params.date_to {
                if m["date"].as_str().unwrap_or("") > dt.as_str() { return false; }
            }
            if let Some(ref un) = params.username {
                if !un.is_empty() {
                    let uname = m["username"].as_str().unwrap_or("").to_lowercase();
                    if !uname.contains(&un.to_lowercase()) { return false; }
                }
            }
            if let Some(ref st) = params.is_suno_team {
                let is_team = matches!(m["is_suno_team"].as_str().unwrap_or(""), "true" | "1");
                match st.as_str() {
                    "true" | "1" | "only"     => { if !is_team { return false; } }
                    "exclude" | "false" | "0" => { if is_team  { return false; } }
                    _ => {}
                }
            }
            if let Some(mw) = params.min_words {
                if mw > 0 {
                    let word_count = m["content"].as_str().unwrap_or("").split_whitespace().count() as i64;
                    if word_count < mw { return false; }
                }
            }
            if let Some(ref uid_str) = params.upload_ids {
                let allowed: Vec<&str> = uid_str.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
                if !allowed.is_empty() {
                    let upload_id = m["upload_id"].as_str().unwrap_or("");
                    if !allowed.contains(&upload_id) { return false; }
                }
            }
            true
        })
        .take(limit)
        .collect();

    // Sort by date if requested; default keeps ChromaDB's similarity-descending order
    match params.sort_by.as_deref() {
        Some("date_asc") => messages.sort_by(|a, b| {
            a["date"].as_str().unwrap_or("").cmp(b["date"].as_str().unwrap_or(""))
        }),
        Some("date_desc") => messages.sort_by(|a, b| {
            b["date"].as_str().unwrap_or("").cmp(a["date"].as_str().unwrap_or(""))
        }),
        _ => {}
    }

    Ok(Json(json!(messages)))
}

pub async fn username_search(
    State(state): State<AppState>,
    Query(params): Query<UsernameParams>,
) -> Result<Json<Value>> {
    let q = params.q.unwrap_or_default();
    let limit = params.limit.unwrap_or(200).min(10_000);
    let pattern = format!("%{}%", q);

    let mut sql = String::from(
        "SELECT id, msg_uuid, author_id, username, date, content,
                attachments, reactions, is_suno_team, upload_id, row_index
         FROM messages WHERE username LIKE ?",
    );
    let mut args: Vec<String> = vec![pattern];

    if let Some(ref df) = params.date_from {
        sql.push_str(" AND date >= ?");
        args.push(df.clone());
    }
    if let Some(ref dt) = params.date_to {
        sql.push_str(" AND date <= ?");
        args.push(dt.clone());
    }
    if let Some(ref st) = params.is_suno_team {
        match st.as_str() {
            "true" | "1" | "only"     => sql.push_str(" AND is_suno_team IN ('true','1')"),
            "exclude" | "false" | "0" => sql.push_str(" AND (is_suno_team IS NULL OR is_suno_team NOT IN ('true','1'))"),
            _ => {}
        }
    }
    if let Some(mw) = params.min_words {
        if mw > 0 {
            sql.push_str(&format!(
                " AND (length(content) - length(replace(content,' ','')) + 1) >= {}",
                mw
            ));
        }
    }
    if let Some(ref uid_str) = params.upload_ids {
        let ids: Vec<&str> = uid_str.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        if !ids.is_empty() {
            let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            sql.push_str(&format!(" AND upload_id IN ({})", placeholders));
            for id in ids { args.push(id.to_string()); }
        }
    }
    sql.push_str(" ORDER BY date ASC LIMIT ?");
    args.push(limit.to_string());

    let mut q_builder = sqlx::query(&sql);
    for arg in &args { q_builder = q_builder.bind(arg); }
    let rows = q_builder.fetch_all(&state.db).await?;

    let messages: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            use sqlx::Row;
            json!({
                "id": r.get::<i64, _>("id"),
                "msg_uuid": r.get::<String, _>("msg_uuid"),
                "username": r.get::<String, _>("username"),
                "date": r.get::<String, _>("date"),
                "content": r.get::<String, _>("content"),
                "upload_id": r.get::<Option<String>, _>("upload_id"),
                "row_index": r.get::<Option<i64>, _>("row_index"),
            })
        })
        .collect();

    Ok(Json(json!(messages)))
}

pub async fn range_search(
    State(state): State<AppState>,
    Query(params): Query<RangeParams>,
) -> Result<Json<Value>> {
    let limit = params.limit.map(|l| l.min(10_000)); // None = no limit
    let offset = params.offset.unwrap_or(0);

    let mut sql = String::from(
        "SELECT id, msg_uuid, author_id, username, date, content,
                attachments, reactions, is_suno_team, upload_id, row_index
         FROM messages WHERE 1=1",
    );
    let mut args: Vec<String> = Vec::new();

    if let Some(ref df) = params.date_from {
        sql.push_str(" AND date >= ?");
        args.push(df.clone());
    }
    if let Some(ref dt) = params.date_to {
        sql.push_str(" AND date <= ?");
        args.push(dt.clone());
    }
    if let Some(ref uid_str) = params.upload_ids {
        let ids: Vec<&str> = uid_str.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        if !ids.is_empty() {
            let ph = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            sql.push_str(&format!(" AND upload_id IN ({})", ph));
            for id in ids {
                args.push(id.to_string());
            }
        }
    }

    // LIMIT -1 in SQLite means no limit; use it when the caller omits the limit param.
    sql.push_str(" ORDER BY date ASC LIMIT ? OFFSET ?");
    args.push(limit.unwrap_or(-1).to_string());
    args.push(offset.to_string());

    let mut q_builder = sqlx::query(&sql);
    for arg in &args {
        q_builder = q_builder.bind(arg);
    }

    let rows = q_builder.fetch_all(&state.db).await?;

    let messages: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            use sqlx::Row;
            json!({
                "id": r.get::<i64, _>("id"),
                "msg_uuid": r.get::<String, _>("msg_uuid"),
                "username": r.get::<String, _>("username"),
                "date": r.get::<String, _>("date"),
                "content": r.get::<String, _>("content"),
                "upload_id": r.get::<Option<String>, _>("upload_id"),
                "row_index": r.get::<Option<i64>, _>("row_index"),
            })
        })
        .collect();

    Ok(Json(json!(messages)))
}

pub async fn users_in_range(
    State(state): State<AppState>,
    Query(params): Query<UserRangeParams>,
) -> Result<Json<Value>> {
    let mut date_filters = String::new();
    let mut args: Vec<String> = Vec::new();

    if let Some(ref df) = params.date_from {
        date_filters.push_str(" AND date >= ?");
        args.push(df.clone());
    }
    if let Some(ref dt) = params.date_to {
        date_filters.push_str(" AND date <= ?");
        args.push(dt.clone());
    }

    // Single round-trip: CTE computes total_weeks, cross-joined into the per-user aggregation
    let sql = format!(
        "WITH tw AS (
             SELECT COUNT(DISTINCT COALESCE(week, strftime('%Y-%W', date))) AS total_weeks
             FROM messages WHERE 1=1{df}
         )
         SELECT m.username,
                COUNT(*) AS message_count,
                MIN(m.date) AS first_date,
                MAX(m.date) AS last_date,
                ROUND(AVG(COALESCE(CAST(m.word_count AS REAL),
                    CAST(length(m.content) - length(replace(m.content, ' ', '')) + 1 AS REAL))), 1) AS avg_words,
                COUNT(DISTINCT COALESCE(m.week, strftime('%Y-%W', m.date))) AS weeks_active,
                MAX(CASE WHEN m.is_suno_team IN ('true', '1', 'True') THEN 1 ELSE 0 END) AS is_team,
                tw.total_weeks
         FROM messages m, tw
         WHERE 1=1{df}
         GROUP BY m.username, tw.total_weeks
         ORDER BY message_count DESC",
        df = date_filters,
    );

    let mut q = sqlx::query(&sql);
    // Bind args twice: once for the CTE WHERE clause, once for the outer WHERE clause
    for a in &args { q = q.bind(a); }
    for a in &args { q = q.bind(a); }
    let rows = q.fetch_all(&state.db).await?;

    use sqlx::Row;
    let users: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            let weeks_active: i64 = r.get("weeks_active");
            let total_weeks: i64 = r.get::<i64, _>("total_weeks").max(1);
            let pct_weeks = (weeks_active as f64 / total_weeks as f64 * 1000.0).round() / 10.0;
            json!({
                "username": r.get::<String, _>("username"),
                "total_messages": r.get::<i64, _>("message_count"),
                "first_date": r.get::<Option<String>, _>("first_date"),
                "last_date": r.get::<Option<String>, _>("last_date"),
                "avg_words": r.get::<Option<f64>, _>("avg_words").unwrap_or(0.0),
                "weeks_active": weeks_active,
                "total_weeks": total_weeks,
                "pct_weeks": pct_weeks,
                "is_suno_team": r.get::<i64, _>("is_team") == 1,
            })
        })
        .collect();

    Ok(Json(json!(users)))
}

pub async fn user_messages(
    State(state): State<AppState>,
    Query(params): Query<UserMessagesParams>,
) -> Result<Json<Value>> {
    let username = params.username.unwrap_or_default();
    let limit = params.limit.unwrap_or(100).min(10_000);
    let offset = params.offset.unwrap_or(0);

    let rows = sqlx::query(
        "SELECT id, msg_uuid, author_id, username, date, content,
                attachments, reactions, is_suno_team, upload_id, row_index
         FROM messages WHERE username = ? ORDER BY date ASC LIMIT ? OFFSET ?",
    )
    .bind(&username)
    .bind(limit)
    .bind(offset)
    .fetch_all(&state.db)
    .await?;

    let messages: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            use sqlx::Row;
            json!({
                "id": r.get::<i64, _>("id"),
                "msg_uuid": r.get::<String, _>("msg_uuid"),
                "username": r.get::<String, _>("username"),
                "date": r.get::<String, _>("date"),
                "content": r.get::<String, _>("content"),
                "upload_id": r.get::<Option<String>, _>("upload_id"),
                "row_index": r.get::<Option<i64>, _>("row_index"),
            })
        })
        .collect();

    Ok(Json(json!(messages)))
}

pub async fn bulk_context(
    State(state): State<AppState>,
    Json(req): Json<BulkContextRequest>,
) -> Result<Json<Value>> {
    let context_size = req.context_size.unwrap_or(5);
    let mut results: Vec<Value> = Vec::new();

    for msg_id in &req.message_ids {
        let row = sqlx::query(
            "SELECT upload_id, row_index FROM messages WHERE id = ?",
        )
        .bind(msg_id)
        .fetch_optional(&state.db)
        .await?;

        if let Some(r) = row {
            use sqlx::Row;
            let upload_id: Option<String> = r.get("upload_id");
            let row_index: Option<i64> = r.get("row_index");

            if let (Some(uid), Some(ri)) = (upload_id, row_index) {
                let context_rows = sqlx::query(
                    "SELECT id, msg_uuid, username, date, content, row_index
                     FROM messages
                     WHERE upload_id = ? AND row_index BETWEEN ? AND ?
                     ORDER BY row_index ASC",
                )
                .bind(&uid)
                .bind(ri - context_size)
                .bind(ri + context_size)
                .fetch_all(&state.db)
                .await?;

                let context_msgs: Vec<Value> = context_rows
                    .into_iter()
                    .map(|cr| {
                        json!({
                            "id": cr.get::<i64, _>("id"),
                            "msg_uuid": cr.get::<String, _>("msg_uuid"),
                            "username": cr.get::<String, _>("username"),
                            "date": cr.get::<String, _>("date"),
                            "content": cr.get::<String, _>("content"),
                            "row_index": cr.get::<Option<i64>, _>("row_index"),
                            "is_target": cr.get::<i64, _>("id") == *msg_id,
                        })
                    })
                    .collect();

                results.push(json!({
                    "message_id": msg_id,
                    "context": context_msgs,
                }));
            }
        }
    }

    Ok(Json(json!(results)))
}
