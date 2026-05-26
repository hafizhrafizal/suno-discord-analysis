use axum::{extract::State, Json};
use serde_json::{json, Value};
use crate::{error::Result, state::AppState};

pub async fn get_stats(State(state): State<AppState>) -> Result<Json<Value>> {
    let total_messages: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM messages")
            .fetch_one(&state.db)
            .await?;

    let total_uploads: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM uploads")
            .fetch_one(&state.db)
            .await?;

    let embedded_messages: i64 = sqlx::query_scalar(
        "SELECT COUNT(DISTINCT m.id) FROM messages m
         JOIN embedded_uploads eu ON m.upload_id = eu.upload_id",
    )
    .fetch_one(&state.db)
    .await?;

    let api_key_set = state.openai_key.read().await.is_some();

    let current_model: Option<String> =
        sqlx::query_scalar("SELECT value FROM settings WHERE key='embedding_model'")
            .fetch_optional(&state.db)
            .await?;

    Ok(Json(json!({
        "total_messages": total_messages,
        "total_uploads": total_uploads,
        "embedded_messages": embedded_messages,
        "api_key_set": api_key_set,
        "current_model": current_model,
    })))
}
