"""
database.py — SQLite helpers: connection, schema init, settings, and
embedded-upload tracking.
"""

import logging
import sqlite3
from datetime import datetime
from typing import Optional

import pandas as pd

from config import DB_PATH

logger = logging.getLogger(__name__)


# ── Connection ────────────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA cache_size=-32000")    # 32 MB page cache
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=134217728")  # 128 MB memory-mapped I/O
    return conn


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db() -> None:
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            msg_uuid     TEXT    UNIQUE NOT NULL,
            author_id    TEXT,
            username     TEXT,
            date         TEXT,
            content      TEXT,
            attachments  TEXT,
            reactions    TEXT,
            is_suno_team TEXT,
            week         TEXT,
            month        TEXT,
            upload_id    TEXT    NOT NULL,
            row_index    INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS uploads (
            id          TEXT    PRIMARY KEY,
            filename    TEXT    NOT NULL,
            row_count   INTEGER NOT NULL,
            upload_time TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS bookmarks (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            msg_id     INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            ctx_before INTEGER NOT NULL DEFAULT 5,
            ctx_after  INTEGER NOT NULL DEFAULT 5,
            note       TEXT    DEFAULT '',
            created_at TEXT    NOT NULL
        );

        DROP INDEX IF EXISTS idx_content;

        CREATE INDEX IF NOT EXISTS idx_username     ON messages(username COLLATE NOCASE);
        CREATE INDEX IF NOT EXISTS idx_upload_row   ON messages(upload_id, row_index);
        CREATE INDEX IF NOT EXISTS idx_date         ON messages(date);
        CREATE INDEX IF NOT EXISTS idx_msg_uuid     ON messages(msg_uuid);
        CREATE INDEX IF NOT EXISTS idx_suno_team    ON messages(is_suno_team);
        CREATE INDEX IF NOT EXISTS idx_date_suno    ON messages(date, is_suno_team);
        CREATE INDEX IF NOT EXISTS idx_bookmark_msg ON bookmarks(msg_id);

        CREATE TABLE IF NOT EXISTS labels (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT    NOT NULL UNIQUE,
            color      TEXT    NOT NULL DEFAULT '#6366f1',
            created_at TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS bookmark_labels (
            bookmark_id INTEGER NOT NULL REFERENCES bookmarks(id) ON DELETE CASCADE,
            label_id    INTEGER NOT NULL REFERENCES labels(id)    ON DELETE CASCADE,
            PRIMARY KEY (bookmark_id, label_id)
        );

        -- Tracks which uploads have been embedded per model.
        -- Replaces O(n_uploads × n_docs) vector-store scans with O(1) SQLite lookups.
        CREATE TABLE IF NOT EXISTS embedded_uploads (
            upload_id   TEXT NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
            model_id    TEXT NOT NULL,
            embedded_at TEXT NOT NULL,
            PRIMARY KEY (upload_id, model_id)
        );
        CREATE INDEX IF NOT EXISTS idx_embedded_model ON embedded_uploads(model_id);

        -- FTS5 full-text search index (content mirror of messages.content)
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
            USING fts5(content, content='messages', content_rowid='id');

        CREATE TRIGGER IF NOT EXISTS tg_messages_ai
            AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
            END;
        CREATE TRIGGER IF NOT EXISTS tg_messages_ad
            AFTER DELETE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content)
                    VALUES ('delete', old.id, old.content);
            END;
        CREATE TRIGGER IF NOT EXISTS tg_messages_au
            AFTER UPDATE OF content ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content)
                    VALUES ('delete', old.id, old.content);
                INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
            END;
    """)

    # Rebuild FTS only when the index appears empty (first run, or after manual wipe).
    fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
    msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    if fts_count == 0 and msg_count > 0:
        logger.info("FTS index empty — rebuilding for %d messages…", msg_count)
        conn.execute("INSERT INTO messages_fts(messages_fts) VALUES ('rebuild')")
    conn.commit()
    conn.close()


# ── Settings ──────────────────────────────────────────────────────────────────

def get_setting(key: str, default: str = "") -> str:
    conn = get_db()
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?,?)", (key, value)
    )
    conn.commit()
    conn.close()


# ── Embedded-upload tracking ──────────────────────────────────────────────────

def mark_upload_embedded(upload_id: str, model_id: str) -> None:
    """Record that upload_id has been embedded with model_id."""
    conn = get_db()
    conn.execute(
        "INSERT OR IGNORE INTO embedded_uploads (upload_id, model_id, embedded_at) VALUES (?,?,?)",
        (upload_id, model_id, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def unmark_upload_embedded(upload_id: str, model_id: Optional[str] = None) -> None:
    """Remove embedding record for an upload (all models if model_id is None)."""
    conn = get_db()
    if model_id:
        conn.execute(
            "DELETE FROM embedded_uploads WHERE upload_id=? AND model_id=?",
            (upload_id, model_id),
        )
    else:
        conn.execute("DELETE FROM embedded_uploads WHERE upload_id=?", (upload_id,))
    conn.commit()
    conn.close()


def get_all_embedded_uploads() -> dict[str, set[str]]:
    """Return {model_id: {upload_id, …}} from the SQLite tracking table."""
    conn = get_db()
    rows = conn.execute("SELECT upload_id, model_id FROM embedded_uploads").fetchall()
    conn.close()
    result: dict[str, set[str]] = {}
    for upload_id, model_id in rows:
        result.setdefault(model_id, set()).add(upload_id)
    return result


# ── Utilities ─────────────────────────────────────────────────────────────────

def safe_str(val) -> str:
    """Convert a possibly-NaN pandas value to a clean string."""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip()
