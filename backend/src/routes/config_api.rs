use axum::{extract::State, Json};
use serde_json::{json, Value};
use crate::{
    error::Result,
    models::{AuthUser, SetApiKeyRequest, SetEmbeddingModelRequest},
    state::AppState,
};

const EMBEDDING_MODEL_ID: &str = "text-embedding-3-small";
const EMBEDDING_MODEL_DIMS: u32 = 1536;

/// The API key is stored in the browser's localStorage only and sent per-request via
/// the `X-OpenAI-Key` header.  This endpoint is kept for backward compatibility but
/// no longer stores anything server-side.
pub async fn set_api_key(
    _state: State<AppState>,
    _user: AuthUser,
    _req: Json<SetApiKeyRequest>,
) -> Result<Json<Value>> {
    Ok(Json(json!({ "status": "ok" })))
}

pub async fn set_embedding_model(
    State(state): State<AppState>,
    _user: AuthUser,
    _req: Json<SetEmbeddingModelRequest>,
) -> Result<Json<Value>> {
    sqlx::query(
        "INSERT INTO settings (key, value) VALUES ('embedding_model', $1)
         ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
    )
    .bind(EMBEDDING_MODEL_ID)
    .execute(&state.db)
    .await?;
    Ok(Json(json!({ "model": EMBEDDING_MODEL_ID })))
}

pub async fn list_embedding_models(State(state): State<AppState>) -> Result<Json<Value>> {
    let vector_count: i64 = state.get_vector_count().await.unwrap_or(0);

    Ok(Json(json!([{
        "id": EMBEDDING_MODEL_ID,
        "label": "text-embedding-3-small",
        "description": "OpenAI text-embedding-3-small · 1536 dims",
        "dims": EMBEDDING_MODEL_DIMS,
        "vector_count": vector_count,
        "active": true,
    }])))
}
