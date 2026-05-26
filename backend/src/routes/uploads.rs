use axum::{
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::{sse::{Event, Sse}, IntoResponse},
    Json,
};
use serde_json::{json, Value};
use std::convert::Infallible;
use uuid::Uuid;
use crate::{error::{AppError, Result}, state::AppState};

pub async fn upload_csv(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse> {
    let mut csv_bytes: Option<Vec<u8>> = None;
    let mut original_filename = String::from("upload.csv");

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::BadRequest(e.to_string()))?
    {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" {
            original_filename = field
                .file_name()
                .unwrap_or("upload.csv")
                .to_string();
            let data = field
                .bytes()
                .await
                .map_err(|e| AppError::BadRequest(e.to_string()))?;
            csv_bytes = Some(data.to_vec());
        }
    }

    let csv_data = csv_bytes.ok_or_else(|| AppError::BadRequest("No file field found".into()))?;

    // Parse CSV
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(csv_data.as_slice());

    let headers = reader
        .headers()
        .map_err(|e| AppError::BadRequest(format!("CSV parse error: {}", e)))?
        .clone();

    // Normalize header mapping
    fn normalize_header(h: &str) -> &'static str {
        match h.trim().to_lowercase().as_str() {
            "id" | "message id" | "msg_uuid" => "msg_uuid",
            "author" | "author id" | "username" => "username",
            "content" | "message" | "text" => "content",
            "date" | "timestamp" | "created_at" | "time" => "date",
            "attachments" | "attachment" => "attachments",
            "reactions" | "reaction" => "reactions",
            _ => "unknown",
        }
    }

    let header_map: Vec<(&'static str, usize)> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| (normalize_header(h), i))
        .filter(|(name, _)| *name != "unknown")
        .collect();

    let get_field = move |row: &csv::StringRecord, field: &str| -> Option<String> {
        header_map
            .iter()
            .find(|(n, _)| *n == field)
            .and_then(|(_, i)| row.get(*i))
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
    };

    let upload_id = Uuid::new_v4().to_string();
    let now = chrono::Utc::now().to_rfc3339();

    // Collect all rows first (to count)
    let records: Vec<csv::StringRecord> = reader
        .records()
        .filter_map(|r| r.ok())
        .collect();
    let total = records.len() as i64;

    // Insert upload record
    sqlx::query(
        "INSERT INTO uploads (id, filename, row_count, upload_time) VALUES (?, ?, ?, ?)",
    )
    .bind(&upload_id)
    .bind(&original_filename)
    .bind(total)
    .bind(&now)
    .execute(&state.db)
    .await?;

    // Build SSE stream
    let upload_id_clone = upload_id.clone();
    let db = state.db.clone();

    let event_stream = async_stream::stream! {
        let mut inserted: i64 = 0;
        let batch_size = 500usize;

        for chunk in records.chunks(batch_size) {
            let mut tx = match db.begin().await {
                Ok(tx) => tx,
                Err(e) => {
                    yield Ok::<Event, Infallible>(Event::default().data(
                        serde_json::to_string(&json!({"type":"error","message":e.to_string()})).unwrap()
                    ));
                    return;
                }
            };

            for record in chunk {
                let msg_uuid = get_field(record, "msg_uuid")
                    .unwrap_or_else(|| Uuid::new_v4().to_string());
                let username = get_field(record, "username").unwrap_or_else(|| "unknown".to_string());
                let content = match get_field(record, "content") {
                    Some(c) => c,
                    None => continue,
                };
                let date = get_field(record, "date").unwrap_or_default();
                let attachments = get_field(record, "attachments");
                let reactions = get_field(record, "reactions");
                let row_idx = inserted;

                let word_count = content.split_whitespace().count() as i64;
                let _ = sqlx::query(
                    "INSERT OR IGNORE INTO messages
                     (msg_uuid, username, date, content, attachments, reactions, upload_id, row_index, week, word_count)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, strftime('%Y-%W', ?), ?)",
                )
                .bind(&msg_uuid)
                .bind(&username)
                .bind(&date)
                .bind(&content)
                .bind(&attachments)
                .bind(&reactions)
                .bind(&upload_id_clone)
                .bind(row_idx)
                .bind(&date)
                .bind(word_count)
                .execute(&mut *tx)
                .await;

                inserted += 1;
            }

            let _ = tx.commit().await;

            yield Ok(Event::default().data(
                serde_json::to_string(&json!({
                    "type": "progress",
                    "inserted": inserted,
                    "total": total,
                })).unwrap()
            ));
        }

        // Rebuild FTS index
        let _ = sqlx::query("INSERT INTO messages_fts(messages_fts) VALUES ('rebuild')")
            .execute(&db)
            .await;

        yield Ok(Event::default().data(
            serde_json::to_string(&json!({
                "type": "done",
                "upload_id": upload_id_clone,
                "total_inserted": inserted,
            })).unwrap()
        ));
    };

    Ok(Sse::new(event_stream))
}

pub async fn list_uploads(State(state): State<AppState>) -> Result<Json<Value>> {
    let rows = sqlx::query(
        "SELECT id, filename, row_count, upload_time FROM uploads ORDER BY upload_time DESC",
    )
    .fetch_all(&state.db)
    .await?;

    let uploads: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            use sqlx::Row;
            json!({
                "id": r.get::<String, _>("id"),
                "filename": r.get::<String, _>("filename"),
                "row_count": r.get::<i64, _>("row_count"),
                "upload_time": r.get::<String, _>("upload_time"),
            })
        })
        .collect();

    Ok(Json(json!(uploads)))
}

pub async fn delete_upload(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>> {
    sqlx::query("DELETE FROM messages WHERE upload_id = ?")
        .bind(&id)
        .execute(&state.db)
        .await?;
    sqlx::query("DELETE FROM embedded_uploads WHERE upload_id = ?")
        .bind(&id)
        .execute(&state.db)
        .await?;
    let affected = sqlx::query("DELETE FROM uploads WHERE id = ?")
        .bind(&id)
        .execute(&state.db)
        .await?
        .rows_affected();

    if affected == 0 {
        return Err(AppError::NotFound(format!("Upload {} not found", id)));
    }
    Ok(Json(json!({ "deleted": id })))
}

pub async fn delete_upload_sqlite(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>> {
    let affected = sqlx::query("DELETE FROM messages WHERE upload_id = ?")
        .bind(&id)
        .execute(&state.db)
        .await?
        .rows_affected();
    Ok(Json(json!({ "deleted_messages": affected })))
}

pub async fn delete_upload_embeddings(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>> {
    sqlx::query("DELETE FROM embedded_uploads WHERE upload_id = ?")
        .bind(&id)
        .execute(&state.db)
        .await?;
    Ok(Json(json!({ "status": "ok", "note": "Vector embeddings not yet implemented in Rust backend" })))
}

pub async fn reembed(Path(id): Path<String>) -> (StatusCode, Json<Value>) {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({ "error": "Re-embedding is not yet implemented in the Rust backend.", "upload_id": id })),
    )
}

pub async fn get_job(Path(job_id): Path<String>) -> (StatusCode, Json<Value>) {
    (
        StatusCode::NOT_FOUND,
        Json(json!({ "error": "No active embedding jobs", "job_id": job_id })),
    )
}
