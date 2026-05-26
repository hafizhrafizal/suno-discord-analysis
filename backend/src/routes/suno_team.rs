use axum::{extract::{Path, State}, Json};
use serde_json::{json, Value};
use crate::{error::Result, state::AppState};

pub async fn list_suno_team(State(state): State<AppState>) -> Result<Json<Value>> {
    let rows = sqlx::query(
        "SELECT username, COUNT(*) as message_count
         FROM messages
         WHERE is_suno_team IN ('true','1')
         GROUP BY username
         ORDER BY message_count DESC",
    )
    .fetch_all(&state.db)
    .await?;

    let team: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            use sqlx::Row;
            json!({
                "username": r.get::<String, _>("username"),
                "message_count": r.get::<i64, _>("message_count"),
            })
        })
        .collect();

    Ok(Json(json!(team)))
}

pub async fn remove_suno_team(
    State(state): State<AppState>,
    Path(username): Path<String>,
) -> Result<Json<Value>> {
    let affected = sqlx::query(
        "UPDATE messages SET is_suno_team='false' WHERE username = $1",
    )
    .bind(&username)
    .execute(&state.db)
    .await?
    .rows_affected();

    Ok(Json(json!({ "username": username, "updated": affected })))
}
