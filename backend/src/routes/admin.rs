use axum::{
    extract::{Path, State},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use argon2::{
    password_hash::{rand_core::OsRng, PasswordHasher, SaltString},
    Argon2,
};
use crate::{error::{AppError, Result}, middleware::auth::AdminUser, state::AppState};

#[derive(Deserialize)]
pub struct SetPasswordRequest {
    pub password: String,
}

pub async fn list_users(
    State(state): State<AppState>,
    _admin: AdminUser,
) -> Result<Json<Value>> {
    let rows = sqlx::query(
        "SELECT id, username, is_admin, created_at FROM users ORDER BY created_at DESC",
    )
    .fetch_all(&state.db)
    .await?;

    let users: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            use sqlx::Row;
            json!({
                "id": r.get::<i64, _>("id"),
                "username": r.get::<String, _>("username"),
                "is_admin": r.get::<bool, _>("is_admin"),
                "created_at": r.get::<String, _>("created_at"),
            })
        })
        .collect();

    Ok(Json(json!(users)))
}

pub async fn delete_user(
    State(state): State<AppState>,
    _admin: AdminUser,
    Path(user_id): Path<i64>,
) -> Result<Json<Value>> {
    let affected = sqlx::query("DELETE FROM users WHERE id = $1")
        .bind(user_id)
        .execute(&state.db)
        .await?
        .rows_affected();

    if affected == 0 {
        return Err(crate::error::AppError::NotFound(format!(
            "User {} not found",
            user_id
        )));
    }
    Ok(Json(json!({ "deleted": user_id })))
}

pub async fn toggle_admin(
    State(state): State<AppState>,
    _admin: AdminUser,
    Path(user_id): Path<i64>,
) -> Result<Json<Value>> {
    let current: bool =
        sqlx::query_scalar("SELECT is_admin FROM users WHERE id = $1")
            .bind(user_id)
            .fetch_optional(&state.db)
            .await?
            .ok_or_else(|| {
                crate::error::AppError::NotFound(format!("User {} not found", user_id))
            })?;

    let new_val = !current;
    sqlx::query("UPDATE users SET is_admin = $1 WHERE id = $2")
        .bind(new_val)
        .bind(user_id)
        .execute(&state.db)
        .await?;

    Ok(Json(json!({ "id": user_id, "is_admin": new_val })))
}

pub async fn set_user_password(
    State(state): State<AppState>,
    _admin: AdminUser,
    Path(user_id): Path<i64>,
    Json(req): Json<SetPasswordRequest>,
) -> Result<Json<Value>> {
    if req.password.len() < 8 {
        return Err(AppError::BadRequest("Password must be at least 8 characters".into()));
    }

    let salt = SaltString::generate(&mut OsRng);
    let hash = Argon2::default()
        .hash_password(req.password.as_bytes(), &salt)
        .map_err(|e| AppError::Internal(e.to_string()))?
        .to_string();

    let affected = sqlx::query(
        "UPDATE users SET password_hash = $1, password_salt = $2 WHERE id = $3",
    )
    .bind(&hash)
    .bind(salt.as_str())
    .bind(user_id)
    .execute(&state.db)
    .await?
    .rows_affected();

    if affected == 0 {
        return Err(AppError::NotFound(format!("User {} not found", user_id)));
    }

    Ok(Json(json!({ "ok": true })))
}
