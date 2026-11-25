use axum::{extract::State, http::StatusCode, Json};
use serde_json::{json, Value};
use tower_sessions::Session;
use argon2::{
    password_hash::{PasswordHash, PasswordVerifier},
    Argon2,
};
use crate::{error::{AppError, Result}, models::{AuthUser, LoginRequest, SetModeRequest}, state::AppState};

// Public self-registration is intentionally removed -- accounts are
// created only by an admin via POST /api/admin/users (see admin.rs).
// This handler is kept so any old client still pointing at the route
// gets a clear, explicit error instead of a generic 404.
pub async fn register() -> Result<Json<Value>> {
    Err(AppError::Forbidden)
}


pub async fn login(
    State(state): State<AppState>,
    session: Session,
    Json(req): Json<LoginRequest>,
) -> Result<Json<Value>> {
    let mode = state.get_app_mode().await.unwrap_or_default();
    if mode != "multi" {
        return Err(AppError::Forbidden);
    }
    let row = sqlx::query(
        "SELECT id, username, password_hash, is_admin FROM users WHERE LOWER(username) = LOWER($1)",
    )
    .bind(&req.username)
    .fetch_optional(&state.db)
    .await?
    .ok_or_else(|| AppError::BadRequest("Invalid username or password".into()))?;

    use sqlx::Row;
    let password_hash: String = row.get("password_hash");
    let parsed_hash = PasswordHash::new(&password_hash)
        .map_err(|e| AppError::Internal(e.to_string()))?;
    Argon2::default()
        .verify_password(req.password.as_bytes(), &parsed_hash)
        .map_err(|_| AppError::BadRequest("Invalid username or password".into()))?;

    let id: i64 = row.get("id");
    let username: String = row.get("username");
    let is_admin: bool = row.get("is_admin");
    session.insert("user_id", id).await.map_err(|e| AppError::Internal(e.to_string()))?;

    Ok(Json(json!({
        "id": id,
        "username": username,
        "is_admin": is_admin
    })))
}

pub async fn logout(session: Session) -> StatusCode {
    let _ = session.clear().await;
    StatusCode::NO_CONTENT
}

pub async fn me(user: AuthUser) -> Json<Value> {
    Json(json!({ "id": user.id, "username": user.username, "is_admin": user.is_admin }))
}

pub async fn get_mode(State(state): State<AppState>) -> Result<Json<Value>> {
    let mode = state.get_app_mode().await;
    Ok(Json(json!({ "mode": mode.unwrap_or_else(|| "single".to_string()) })))
}

pub async fn set_mode(
    State(state): State<AppState>,
    Json(req): Json<SetModeRequest>,
) -> Result<Json<Value>> {
    if req.mode != "single" && req.mode != "multi" {
        return Err(AppError::BadRequest("Mode must be 'single' or 'multi'".into()));
    }
    sqlx::query(
        "INSERT INTO settings (key, value) VALUES ('app_mode', $1)
         ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
    )
    .bind(&req.mode)
    .execute(&state.db)
    .await?;
    Ok(Json(json!({ "mode": req.mode })))
}
