use axum::{
    extract::{Path, Query, State},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use crate::{
    error::{AppError, Result},
    models::{
        AssignCodeRequest, CreateCodeCategoryRequest, CreateCodeRequest,
        UpdateCodeCategoryRequest, UpdateCodeRequest,
    },
    state::AppState,
};

// ── Codes ────────────────────────────────────────────────────────────────────

pub async fn create_code(
    State(state): State<AppState>,
    Json(req): Json<CreateCodeRequest>,
) -> Result<Json<Value>> {
    let color = req.color.unwrap_or_else(|| "#6366f1".to_string());
    let description = req.description.unwrap_or_default();
    let id: i64 = sqlx::query_scalar(
        "INSERT INTO codes (name, color, description, category_id) VALUES (?, ?, ?, ?) RETURNING id",
    )
    .bind(&req.name)
    .bind(&color)
    .bind(&description)
    .bind(req.category_id)
    .fetch_one(&state.db)
    .await?;

    Ok(Json(json!({
        "id": id,
        "name": req.name,
        "color": color,
        "description": description,
        "category_id": req.category_id,
    })))
}

pub async fn list_codes(State(state): State<AppState>) -> Result<Json<Value>> {
    let rows = sqlx::query(
        "SELECT id, name, color, description, category_id, created_at FROM codes ORDER BY name ASC",
    )
    .fetch_all(&state.db)
    .await?;

    let codes: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            use sqlx::Row;
            json!({
                "id": r.get::<i64, _>("id"),
                "name": r.get::<String, _>("name"),
                "color": r.get::<String, _>("color"),
                "description": r.get::<Option<String>, _>("description").unwrap_or_default(),
                "category_id": r.get::<Option<i64>, _>("category_id"),
                "created_at": r.get::<String, _>("created_at"),
            })
        })
        .collect();

    Ok(Json(json!(codes)))
}

pub async fn update_code(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Json(req): Json<UpdateCodeRequest>,
) -> Result<Json<Value>> {
    if let Some(ref name) = req.name {
        sqlx::query("UPDATE codes SET name = ? WHERE id = ?")
            .bind(name)
            .bind(id)
            .execute(&state.db)
            .await?;
    }
    if let Some(ref color) = req.color {
        sqlx::query("UPDATE codes SET color = ? WHERE id = ?")
            .bind(color)
            .bind(id)
            .execute(&state.db)
            .await?;
    }
    if let Some(ref description) = req.description {
        sqlx::query("UPDATE codes SET description = ? WHERE id = ?")
            .bind(description)
            .bind(id)
            .execute(&state.db)
            .await?;
    }
    // category_id can be explicitly set to null to uncategorize
    if req.category_id.is_some() || (req.name.is_none() && req.color.is_none() && req.description.is_none()) {
        sqlx::query("UPDATE codes SET category_id = ? WHERE id = ?")
            .bind(req.category_id)
            .bind(id)
            .execute(&state.db)
            .await?;
    }

    let row = sqlx::query(
        "SELECT id, name, color, description, category_id FROM codes WHERE id = ?",
    )
    .bind(id)
    .fetch_optional(&state.db)
    .await?
    .ok_or_else(|| AppError::NotFound(format!("Code {} not found", id)))?;

    use sqlx::Row;
    Ok(Json(json!({
        "id": row.get::<i64, _>("id"),
        "name": row.get::<String, _>("name"),
        "color": row.get::<String, _>("color"),
        "description": row.get::<Option<String>, _>("description").unwrap_or_default(),
        "category_id": row.get::<Option<i64>, _>("category_id"),
    })))
}

pub async fn delete_code(
    State(state): State<AppState>,
    Path(id): Path<i64>,
) -> Result<Json<Value>> {
    let affected = sqlx::query("DELETE FROM codes WHERE id = ?")
        .bind(id)
        .execute(&state.db)
        .await?
        .rows_affected();

    if affected == 0 {
        return Err(AppError::NotFound(format!("Code {} not found", id)));
    }
    Ok(Json(json!({ "deleted": id })))
}

// ── Categories ───────────────────────────────────────────────────────────────

pub async fn create_category(
    State(state): State<AppState>,
    Json(req): Json<CreateCodeCategoryRequest>,
) -> Result<Json<Value>> {
    let color = req.color.unwrap_or_else(|| "#94a3b8".to_string());
    let id: i64 = sqlx::query_scalar(
        "INSERT INTO code_categories (name, color, parent_id) VALUES (?, ?, ?) RETURNING id",
    )
    .bind(&req.name)
    .bind(&color)
    .bind(req.parent_id)
    .fetch_one(&state.db)
    .await?;

    Ok(Json(json!({
        "id": id,
        "name": req.name,
        "color": color,
        "parent_id": req.parent_id,
    })))
}

pub async fn list_categories(State(state): State<AppState>) -> Result<Json<Value>> {
    let rows = sqlx::query(
        "SELECT id, name, color, parent_id, created_at FROM code_categories ORDER BY name ASC",
    )
    .fetch_all(&state.db)
    .await?;

    let cats: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            use sqlx::Row;
            json!({
                "id": r.get::<i64, _>("id"),
                "name": r.get::<String, _>("name"),
                "color": r.get::<Option<String>, _>("color").unwrap_or_else(|| "#94a3b8".to_string()),
                "parent_id": r.get::<Option<i64>, _>("parent_id"),
                "created_at": r.get::<String, _>("created_at"),
            })
        })
        .collect();

    Ok(Json(json!(cats)))
}

pub async fn update_category(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Json(req): Json<UpdateCodeCategoryRequest>,
) -> Result<Json<Value>> {
    if let Some(ref name) = req.name {
        sqlx::query("UPDATE code_categories SET name = ? WHERE id = ?")
            .bind(name)
            .bind(id)
            .execute(&state.db)
            .await?;
    }
    if let Some(ref color) = req.color {
        sqlx::query("UPDATE code_categories SET color = ? WHERE id = ?")
            .bind(color)
            .bind(id)
            .execute(&state.db)
            .await?;
    }
    // parent_id can be explicitly null (root level); only update if field was sent
    if req.parent_id.is_some() || (req.name.is_none() && req.color.is_none()) {
        sqlx::query("UPDATE code_categories SET parent_id = ? WHERE id = ?")
            .bind(req.parent_id)
            .bind(id)
            .execute(&state.db)
            .await?;
    }

    let row = sqlx::query(
        "SELECT id, name, color, parent_id FROM code_categories WHERE id = ?",
    )
    .bind(id)
    .fetch_optional(&state.db)
    .await?
    .ok_or_else(|| AppError::NotFound(format!("Category {} not found", id)))?;

    use sqlx::Row;
    Ok(Json(json!({
        "id": row.get::<i64, _>("id"),
        "name": row.get::<String, _>("name"),
        "color": row.get::<Option<String>, _>("color").unwrap_or_else(|| "#94a3b8".to_string()),
        "parent_id": row.get::<Option<i64>, _>("parent_id"),
    })))
}

pub async fn delete_category(
    State(state): State<AppState>,
    Path(id): Path<i64>,
) -> Result<Json<Value>> {
    let affected = sqlx::query("DELETE FROM code_categories WHERE id = ?")
        .bind(id)
        .execute(&state.db)
        .await?
        .rows_affected();

    if affected == 0 {
        return Err(AppError::NotFound(format!("Category {} not found", id)));
    }
    Ok(Json(json!({ "deleted": id })))
}

// ── Bookmark Codes ────────────────────────────────────────────────────────────

pub async fn assign_code(
    State(state): State<AppState>,
    Json(req): Json<AssignCodeRequest>,
) -> Result<Json<Value>> {
    sqlx::query(
        "INSERT OR REPLACE INTO bookmark_codes (bookmark_id, code_id, highlighted_text)
         VALUES (?, ?, ?)",
    )
    .bind(req.bookmark_id)
    .bind(req.code_id)
    .bind(&req.highlighted_text)
    .execute(&state.db)
    .await?;

    Ok(Json(json!({
        "bookmark_id": req.bookmark_id,
        "code_id": req.code_id,
        "highlighted_text": req.highlighted_text,
    })))
}

#[derive(Debug, Deserialize)]
pub struct BookmarkCodesQuery {
    pub bookmark_id: Option<i64>,
}

pub async fn list_all_bookmark_codes(
    State(state): State<AppState>,
    Query(q): Query<BookmarkCodesQuery>,
) -> Result<Json<Value>> {
    let rows = if let Some(bid) = q.bookmark_id {
        sqlx::query(
            "SELECT bc.bookmark_id, bc.code_id,
                    bc.highlighted_text,
                    c.name as code_name, c.color, 'excerpt' as type
             FROM bookmark_codes bc JOIN codes c ON c.id = bc.code_id
             WHERE bc.bookmark_id = ?
             UNION ALL
             SELECT bch.bookmark_id, bch.code_id,
                    bch.highlighted_text,
                    c.name as code_name, c.color, 'highlight' as type
             FROM bookmark_code_highlights bch JOIN codes c ON c.id = bch.code_id
             WHERE bch.bookmark_id = ?",
        )
        .bind(bid)
        .bind(bid)
        .fetch_all(&state.db)
        .await?
    } else {
        sqlx::query(
            "SELECT bc.bookmark_id, bc.code_id,
                    bc.highlighted_text,
                    c.name as code_name, c.color, 'excerpt' as type
             FROM bookmark_codes bc JOIN codes c ON c.id = bc.code_id
             UNION ALL
             SELECT bch.bookmark_id, bch.code_id,
                    bch.highlighted_text,
                    c.name as code_name, c.color, 'highlight' as type
             FROM bookmark_code_highlights bch JOIN codes c ON c.id = bch.code_id",
        )
        .fetch_all(&state.db)
        .await?
    };

    let result: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            use sqlx::Row;
            json!({
                "bookmark_id": r.get::<i64, _>("bookmark_id"),
                "code_id": r.get::<i64, _>("code_id"),
                "highlighted_text": r.get::<Option<String>, _>("highlighted_text"),
                "code_name": r.get::<String, _>("code_name"),
                "color": r.get::<String, _>("color"),
                "type": r.get::<String, _>("type"),
            })
        })
        .collect();

    Ok(Json(json!(result)))
}

pub async fn list_codes_for_bookmark(
    State(state): State<AppState>,
    Path(bookmark_id): Path<i64>,
) -> Result<Json<Value>> {
    let rows = sqlx::query(
        "SELECT bc.code_id,
                bc.highlighted_text,
                c.name as code_name, c.color, 'excerpt' as type
         FROM bookmark_codes bc JOIN codes c ON c.id = bc.code_id
         WHERE bc.bookmark_id = ?
         UNION ALL
         SELECT bch.code_id,
                bch.highlighted_text,
                c.name as code_name, c.color, 'highlight' as type
         FROM bookmark_code_highlights bch JOIN codes c ON c.id = bch.code_id
         WHERE bch.bookmark_id = ?",
    )
    .bind(bookmark_id)
    .bind(bookmark_id)
    .fetch_all(&state.db)
    .await?;

    let result: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            use sqlx::Row;
            json!({
                "code_id": r.get::<i64, _>("code_id"),
                "highlighted_text": r.get::<Option<String>, _>("highlighted_text"),
                "code_name": r.get::<String, _>("code_name"),
                "color": r.get::<String, _>("color"),
                "type": r.get::<String, _>("type"),
            })
        })
        .collect();

    Ok(Json(json!(result)))
}

pub async fn remove_code_assignment(
    State(state): State<AppState>,
    Path((bookmark_id, code_id)): Path<(i64, i64)>,
) -> Result<Json<Value>> {
    sqlx::query(
        "DELETE FROM bookmark_codes WHERE bookmark_id = ? AND code_id = ?",
    )
    .bind(bookmark_id)
    .bind(code_id)
    .execute(&state.db)
    .await?;

    Ok(Json(json!({ "removed": true, "bookmark_id": bookmark_id, "code_id": code_id })))
}
