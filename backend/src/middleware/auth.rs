use axum::{
    async_trait,
    extract::FromRequestParts,
    http::request::Parts,
};
use tower_sessions::Session;
use crate::{error::AppError, models::AuthUser, state::AppState};

#[async_trait]
impl<S> FromRequestParts<S> for AuthUser
where
    S: Send + Sync + AsRef<AppState>,
{
    type Rejection = AppError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let app_state: &AppState = state.as_ref();

        let mode = app_state.config.app_mode.as_deref().unwrap_or("");
        if mode == "demo" || mode == "single" {
            return Ok(AuthUser { id: 1, username: "admin".to_string(), is_admin: true });
        }

        let session = Session::from_request_parts(parts, state)
            .await
            .map_err(|_| AppError::Unauthorized)?;

        let user_id: Option<i64> =
            session.get("user_id").await.map_err(|_| AppError::Unauthorized)?;
        let user_id = user_id.ok_or(AppError::Unauthorized)?;

        let row = sqlx::query(
            "SELECT id, username, is_admin FROM users WHERE id = $1",
        )
        .bind(user_id)
        .fetch_optional(&app_state.db)
        .await
        .map_err(AppError::Database)?
        .ok_or(AppError::Unauthorized)?;

        use sqlx::Row;
        Ok(AuthUser {
            id: row.get::<i64, _>("id"),
            username: row.get::<String, _>("username"),
            is_admin: row.get::<bool, _>("is_admin"),
        })
    }
}

#[allow(dead_code)]
pub struct AdminUser(pub AuthUser);

#[async_trait]
impl<S> FromRequestParts<S> for AdminUser
where
    S: Send + Sync + AsRef<AppState>,
{
    type Rejection = AppError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let user = AuthUser::from_request_parts(parts, state).await?;
        if !user.is_admin {
            return Err(AppError::Forbidden);
        }
        Ok(AdminUser(user))
    }
}
