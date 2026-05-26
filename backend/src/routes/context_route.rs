use axum::{
    extract::{Path, Query, State},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use crate::{error::Result, models::SemanticFilterRequest, state::AppState};

#[derive(Debug, Deserialize)]
pub struct ContextParams {
    pub before: Option<i64>,
    pub after: Option<i64>,
}

pub async fn get_context(
    State(state): State<AppState>,
    Path(message_id): Path<i64>,
    Query(params): Query<ContextParams>,
) -> Result<Json<Value>> {
    let before = params.before.unwrap_or(5).min(200);
    let after = params.after.unwrap_or(5).min(200);

    let row = sqlx::query(
        "SELECT upload_id, row_index FROM messages WHERE id = ?",
    )
    .bind(message_id)
    .fetch_optional(&state.db)
    .await?
    .ok_or_else(|| crate::error::AppError::NotFound(format!("Message {} not found", message_id)))?;

    use sqlx::Row;
    let upload_id: Option<String> = row.get("upload_id");
    let row_index: Option<i64> = row.get("row_index");

    let (uid, ri) = match (upload_id, row_index) {
        (Some(u), Some(r)) => (u, r),
        _ => return Err(crate::error::AppError::NotFound("Message has no upload context".into())),
    };

    let rows = sqlx::query(
        "SELECT id, msg_uuid, username, date, content, row_index
         FROM messages
         WHERE upload_id = ? AND row_index BETWEEN ? AND ?
         ORDER BY row_index ASC",
    )
    .bind(&uid)
    .bind(ri - before)
    .bind(ri + after)
    .fetch_all(&state.db)
    .await?;

    let messages: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            let rid: i64 = r.get("id");
            json!({
                "id": rid,
                "msg_uuid": r.get::<String, _>("msg_uuid"),
                "username": r.get::<String, _>("username"),
                "date": r.get::<String, _>("date"),
                "content": r.get::<String, _>("content"),
                "row_index": r.get::<Option<i64>, _>("row_index"),
                "is_target": rid == message_id,
            })
        })
        .collect();

    Ok(Json(json!(messages)))
}

pub async fn filter_semantic(
    Json(req): Json<SemanticFilterRequest>,
) -> Json<Value> {
    // Naive relevance: count query word occurrences in content (no embeddings yet)
    let query_words: Vec<&str> = req.query.split_whitespace().collect();
    let mut scored: Vec<(usize, &crate::models::Message)> = req
        .messages
        .iter()
        .map(|m| {
            let content_lower = m.content.to_lowercase();
            let score = query_words
                .iter()
                .map(|w| {
                    let w_lower = w.to_lowercase();
                    content_lower.matches(w_lower.as_str()).count()
                })
                .sum::<usize>();
            (score, m)
        })
        .collect();

    scored.sort_by(|a, b| b.0.cmp(&a.0));

    let result: Vec<&crate::models::Message> = scored.into_iter().map(|(_, m)| m).collect();
    Json(json!(result))
}
