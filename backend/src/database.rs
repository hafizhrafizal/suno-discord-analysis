use sqlx::{sqlite::SqliteConnectOptions, SqlitePool};
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use sqlx::Row;

pub async fn create_pool(db_path: &str) -> Result<SqlitePool, sqlx::Error> {
    let opts = SqliteConnectOptions::from_str(&format!("sqlite:{}", db_path))?
        .create_if_missing(true)
        .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
        .foreign_keys(true);
    SqlitePool::connect_with(opts).await
}

pub async fn init_db(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS uploads (
            id TEXT PRIMARY KEY,
            filename TEXT,
            row_count INTEGER DEFAULT 0,
            upload_time TEXT DEFAULT (datetime('now'))
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            msg_uuid TEXT UNIQUE,
            author_id TEXT,
            username TEXT NOT NULL DEFAULT 'unknown',
            date TEXT,
            content TEXT,
            attachments TEXT,
            reactions TEXT,
            is_suno_team TEXT DEFAULT 'false',
            week TEXT,
            month TEXT,
            upload_id TEXT REFERENCES uploads(id),
            row_index INTEGER
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE COLLATE NOCASE,
            password_hash TEXT,
            password_salt TEXT,
            is_admin INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            google_id TEXT UNIQUE,
            email TEXT UNIQUE
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS bookmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            msg_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            ctx_before INTEGER DEFAULT 5,
            ctx_after INTEGER DEFAULT 5,
            note TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            user_id INTEGER REFERENCES users(id) ON DELETE SET NULL
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            color TEXT DEFAULT '#6366f1',
            created_at TEXT DEFAULT (datetime('now'))
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS bookmark_labels (
            bookmark_id INTEGER NOT NULL REFERENCES bookmarks(id) ON DELETE CASCADE,
            label_id INTEGER NOT NULL REFERENCES labels(id) ON DELETE CASCADE,
            PRIMARY KEY (bookmark_id, label_id)
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS embedded_uploads (
            upload_id TEXT NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
            model_id TEXT NOT NULL,
            embedded_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (upload_id, model_id)
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS code_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            color TEXT NOT NULL DEFAULT '#94a3b8',
            parent_id INTEGER REFERENCES code_categories(id) ON DELETE SET NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            color TEXT DEFAULT '#6366f1',
            description TEXT DEFAULT '',
            category_id INTEGER REFERENCES code_categories(id) ON DELETE SET NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS bookmark_codes (
            bookmark_id INTEGER NOT NULL REFERENCES bookmarks(id) ON DELETE CASCADE,
            code_id INTEGER NOT NULL REFERENCES codes(id) ON DELETE CASCADE,
            highlighted_text TEXT,
            PRIMARY KEY (bookmark_id, code_id)
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS bookmark_code_highlights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bookmark_id INTEGER NOT NULL REFERENCES bookmarks(id) ON DELETE CASCADE,
            code_id INTEGER NOT NULL REFERENCES codes(id) ON DELETE CASCADE,
            highlighted_text TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )",
    )
    .execute(pool)
    .await?;

    // Column migrations — safe to run every startup; errors mean column already exists
    for stmt in &[
        "ALTER TABLE code_categories ADD COLUMN color TEXT NOT NULL DEFAULT '#94a3b8'",
        "ALTER TABLE code_categories ADD COLUMN parent_id INTEGER REFERENCES code_categories(id) ON DELETE SET NULL",
        "ALTER TABLE codes ADD COLUMN description TEXT DEFAULT ''",
        "ALTER TABLE bookmark_codes ADD COLUMN highlighted_text TEXT",
        "ALTER TABLE messages ADD COLUMN word_count INTEGER",
    ] {
        let _ = sqlx::query(stmt).execute(pool).await;
    }

    // FTS5 virtual table
    sqlx::query(
        "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            content,
            username,
            date,
            content='messages',
            content_rowid='id'
        )",
    )
    .execute(pool)
    .await?;

    // FTS triggers
    sqlx::query(
        "CREATE TRIGGER IF NOT EXISTS tg_messages_ai AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content, username, date)
            VALUES (new.id, new.content, new.username, new.date);
        END",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TRIGGER IF NOT EXISTS tg_messages_ad AFTER DELETE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, content, username, date)
            VALUES ('delete', old.id, old.content, old.username, old.date);
        END",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TRIGGER IF NOT EXISTS tg_messages_au AFTER UPDATE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, content, username, date)
            VALUES ('delete', old.id, old.content, old.username, old.date);
            INSERT INTO messages_fts(rowid, content, username, date)
            VALUES (new.id, new.content, new.username, new.date);
        END",
    )
    .execute(pool)
    .await?;

    // B-tree indexes for common filter columns
    for stmt in &[
        "CREATE INDEX IF NOT EXISTS idx_messages_date ON messages(date)",
        "CREATE INDEX IF NOT EXISTS idx_messages_username ON messages(username COLLATE NOCASE)",
        "CREATE INDEX IF NOT EXISTS idx_messages_upload_id ON messages(upload_id)",
        "CREATE INDEX IF NOT EXISTS idx_messages_suno_team ON messages(is_suno_team)",
        "CREATE INDEX IF NOT EXISTS idx_messages_date_username ON messages(date, username)",
        "CREATE INDEX IF NOT EXISTS idx_messages_week ON messages(week)",
    ] {
        sqlx::query(stmt).execute(pool).await?;
    }

    // WAL pragmas
    sqlx::query("PRAGMA synchronous=NORMAL").execute(pool).await?;
    sqlx::query("PRAGMA cache_size=-32000").execute(pool).await?;
    sqlx::query("PRAGMA mmap_size=134217728").execute(pool).await?;
    sqlx::query("PRAGMA foreign_keys=ON").execute(pool).await?;

    Ok(())
}

/// Check whether the FTS5 index needs to be rebuilt and return true if so.
/// Called from main so the rebuild can be spawned as a background task.
pub async fn needs_fts_rebuild(pool: &SqlitePool) -> bool {
    let msg_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM messages")
        .fetch_one(pool)
        .await
        .unwrap_or(0);
    if msg_count == 0 {
        return false;
    }
    let fts_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM messages_fts")
        .fetch_one(pool)
        .await
        .unwrap_or(-1);
    msg_count != fts_count
}

/// Rebuild the FTS5 index. Sets `fts_ready` to true on success.
pub async fn rebuild_fts(pool: &SqlitePool, fts_ready: Arc<AtomicBool>) {
    tracing::info!("FTS5 index out of sync — rebuilding in background…");
    match sqlx::query("INSERT INTO messages_fts(messages_fts) VALUES ('rebuild')")
        .execute(pool)
        .await
    {
        Ok(_) => {
            tracing::info!("FTS5 rebuild complete — fast keyword search now active");
            fts_ready.store(true, Ordering::Relaxed);
        }
        Err(e) => tracing::warn!("FTS5 rebuild failed: {}", e),
    }
}

/// Returns true if any messages row is missing word_count (backfill needed).
pub async fn needs_word_count_backfill(pool: &SqlitePool) -> bool {
    sqlx::query("SELECT EXISTS(SELECT 1 FROM messages WHERE word_count IS NULL AND content IS NOT NULL)")
        .fetch_one(pool)
        .await
        .map(|r| r.get::<i64, _>(0) == 1)
        .unwrap_or(false)
}

/// Populate word_count and week for all existing rows that are missing them.
pub async fn backfill_precomputed_columns(pool: &SqlitePool) {
    tracing::info!("Backfilling word_count and week columns…");
    match sqlx::query(
        "UPDATE messages
         SET word_count = length(trim(content)) - length(replace(trim(content), ' ', '')) + 1
         WHERE word_count IS NULL AND content IS NOT NULL",
    )
    .execute(pool)
    .await
    {
        Ok(r) => tracing::info!("Backfill word_count: {} rows updated", r.rows_affected()),
        Err(e) => tracing::warn!("Backfill word_count failed: {}", e),
    }
    match sqlx::query(
        "UPDATE messages SET week = strftime('%Y-%W', date) WHERE week IS NULL AND date IS NOT NULL",
    )
    .execute(pool)
    .await
    {
        Ok(r) => tracing::info!("Backfill week: {} rows updated", r.rows_affected()),
        Err(e) => tracing::warn!("Backfill week failed: {}", e),
    }
}

pub async fn seed_demo_user(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT OR IGNORE INTO users (id, username, password_hash, password_salt, is_admin)
         VALUES (1, 'admin', '', '', 1)",
    )
    .execute(pool)
    .await?;
    Ok(())
}
