use axum::{
    extract::{Path, State},
    Json,
};
use serde_json::{json, Value};
use crate::{
    error::{AppError, Result},
    models::{CreateLabelRequest, UpdateLabelRequest},
    state::AppState,
};

pub async fn create_label(
    State(state): State<AppState>,
    Json(req): Json<CreateLabelRequest>,
) -> Result<Json<Value>> {
    let color = req.color.unwrap_or_else(|| "#6366f1".to_string());
    let id: i64 = sqlx::query_scalar(
        "INSERT INTO labels (name, color) VALUES (?, ?) RETURNING id",
    )
    .bind(&req.name)
    .bind(&color)
    .fetch_one(&state.db)
    .await?;

    Ok(Json(json!({ "id": id, "name": req.name, "color": color })))
}

pub async fn list_labels(State(state): State<AppState>) -> Result<Json<Value>> {
    let rows = sqlx::query("SELECT id, name, color, created_at FROM labels ORDER BY name ASC")
        .fetch_all(&state.db)
        .await?;

    let labels: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            use sqlx::Row;
            json!({
                "id": r.get::<i64, _>("id"),
                "name": r.get::<String, _>("name"),
                "color": r.get::<String, _>("color"),
                "created_at": r.get::<String, _>("created_at"),
            })
        })
        .collect();

    Ok(Json(json!(labels)))
}

pub async fn update_label(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Json(req): Json<UpdateLabelRequest>,
) -> Result<Json<Value>> {
    if let Some(ref name) = req.name {
        sqlx::query("UPDATE labels SET name = ? WHERE id = ?")
            .bind(name)
            .bind(id)
            .execute(&state.db)
            .await?;
    }
    if let Some(ref color) = req.color {
        sqlx::query("UPDATE labels SET color = ? WHERE id = ?")
            .bind(color)
            .bind(id)
            .execute(&state.db)
            .await?;
    }

    let row = sqlx::query("SELECT id, name, color FROM labels WHERE id = ?")
        .bind(id)
        .fetch_optional(&state.db)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Label {} not found", id)))?;

    use sqlx::Row;
    Ok(Json(json!({
        "id": row.get::<i64, _>("id"),
        "name": row.get::<String, _>("name"),
        "color": row.get::<String, _>("color"),
    })))
}

pub async fn delete_label(
    State(state): State<AppState>,
    Path(id): Path<i64>,
) -> Result<Json<Value>> {
    let affected = sqlx::query("DELETE FROM labels WHERE id = ?")
        .bind(id)
        .execute(&state.db)
        .await?
        .rows_affected();

    if affected == 0 {
        return Err(AppError::NotFound(format!("Label {} not found", id)));
    }
    Ok(Json(json!({ "deleted": id })))
}
