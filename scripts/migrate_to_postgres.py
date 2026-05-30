"""
Migrate discord_data.db (SQLite) → PostgreSQL.
Reads DATABASE_URL from .env in the same directory.

Usage:
    pip install psycopg2-binary python-dotenv
    python migrate_to_postgres.py
"""

import sqlite3
import sys
import os
import re

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    sys.exit("Missing psycopg2. Run:  pip install psycopg2-binary")

try:
    from dotenv import load_dotenv
except ImportError:
    sys.exit("Missing python-dotenv. Run:  pip install python-dotenv")

# ── Load .env ─────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, ".env"))

DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL or "yourpassword" in DATABASE_URL:
    sys.exit(
        "ERROR: Set a real DATABASE_URL in .env first.\n"
        "Example: DATABASE_URL=postgres://postgres:MyPass123@localhost:5432/discord_db"
    )

SQLITE_PATH = os.path.join(script_dir, "discord_data.db")
if not os.path.exists(SQLITE_PATH):
    sys.exit(f"ERROR: SQLite file not found at {SQLITE_PATH}")

# ── Connect ────────────────────────────────────────────────────────────────────
print(f"Connecting to PostgreSQL: {DATABASE_URL.split('@')[-1]} ...")
sqlite_conn = sqlite3.connect(SQLITE_PATH)
sqlite_conn.row_factory = sqlite3.Row

pg_dsn = re.sub(r"^postgres://", "postgresql://", DATABASE_URL)
pg_conn = psycopg2.connect(pg_dsn)
pg_conn.autocommit = False
pg_cur = pg_conn.cursor()
print("Connected.\n")

# ── Create tables ──────────────────────────────────────────────────────────────
print("Creating tables if they don't exist...")
pg_conn.autocommit = True
for stmt in [
    """CREATE TABLE IF NOT EXISTS uploads (
        id TEXT PRIMARY KEY,
        filename TEXT,
        row_count BIGINT DEFAULT 0,
        upload_time TEXT DEFAULT to_char(NOW() AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')
    )""",
    """CREATE TABLE IF NOT EXISTS messages (
        id BIGSERIAL PRIMARY KEY,
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
        row_index BIGINT,
        word_count BIGINT,
        search_vector tsvector
    )""",
    """CREATE TABLE IF NOT EXISTS users (
        id BIGSERIAL PRIMARY KEY,
        username TEXT NOT NULL,
        password_hash TEXT,
        password_salt TEXT,
        is_admin BOOLEAN DEFAULT false,
        created_at TEXT DEFAULT to_char(NOW() AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS'),
        google_id TEXT UNIQUE,
        email TEXT UNIQUE
    )""",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(LOWER(username))",
    """CREATE TABLE IF NOT EXISTS bookmarks (
        id BIGSERIAL PRIMARY KEY,
        msg_id BIGINT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
        ctx_before BIGINT DEFAULT 5,
        ctx_after BIGINT DEFAULT 5,
        note TEXT,
        created_at TEXT DEFAULT to_char(NOW() AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS'),
        user_id BIGINT REFERENCES users(id) ON DELETE SET NULL
    )""",
    """CREATE TABLE IF NOT EXISTS labels (
        id BIGSERIAL PRIMARY KEY,
        name TEXT UNIQUE,
        color TEXT DEFAULT '#6366f1',
        created_at TEXT DEFAULT to_char(NOW() AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')
    )""",
    """CREATE TABLE IF NOT EXISTS bookmark_labels (
        bookmark_id BIGINT NOT NULL REFERENCES bookmarks(id) ON DELETE CASCADE,
        label_id BIGINT NOT NULL REFERENCES labels(id) ON DELETE CASCADE,
        PRIMARY KEY (bookmark_id, label_id)
    )""",
    """CREATE TABLE IF NOT EXISTS embedded_uploads (
        upload_id TEXT NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
        model_id TEXT NOT NULL,
        embedded_at TEXT DEFAULT to_char(NOW() AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS'),
        PRIMARY KEY (upload_id, model_id)
    )""",
    """CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS code_categories (
        id BIGSERIAL PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        color TEXT NOT NULL DEFAULT '#94a3b8',
        parent_id BIGINT REFERENCES code_categories(id) ON DELETE SET NULL,
        created_at TEXT DEFAULT to_char(NOW() AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')
    )""",
    """CREATE TABLE IF NOT EXISTS codes (
        id BIGSERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        color TEXT DEFAULT '#6366f1',
        description TEXT DEFAULT '',
        category_id BIGINT REFERENCES code_categories(id) ON DELETE SET NULL,
        created_at TEXT DEFAULT to_char(NOW() AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')
    )""",
    """CREATE TABLE IF NOT EXISTS bookmark_codes (
        bookmark_id BIGINT NOT NULL REFERENCES bookmarks(id) ON DELETE CASCADE,
        code_id BIGINT NOT NULL REFERENCES codes(id) ON DELETE CASCADE,
        highlighted_text TEXT,
        PRIMARY KEY (bookmark_id, code_id)
    )""",
    """CREATE TABLE IF NOT EXISTS bookmark_code_highlights (
        id BIGSERIAL PRIMARY KEY,
        bookmark_id BIGINT NOT NULL REFERENCES bookmarks(id) ON DELETE CASCADE,
        code_id BIGINT NOT NULL REFERENCES codes(id) ON DELETE CASCADE,
        highlighted_text TEXT NOT NULL,
        created_at TEXT DEFAULT to_char(NOW() AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')
    )""",
    """CREATE OR REPLACE FUNCTION messages_search_vector_update() RETURNS trigger AS $$
       BEGIN
           NEW.search_vector := to_tsvector('simple',
               coalesce(NEW.content, '') || ' ' || coalesce(NEW.username, ''));
           RETURN NEW;
       END;
       $$ LANGUAGE plpgsql""",
    "DROP TRIGGER IF EXISTS tg_messages_sv_update ON messages",
    """CREATE TRIGGER tg_messages_sv_update
       BEFORE INSERT OR UPDATE ON messages
       FOR EACH ROW EXECUTE FUNCTION messages_search_vector_update()""",
    "CREATE INDEX IF NOT EXISTS idx_messages_date ON messages(date)",
    "CREATE INDEX IF NOT EXISTS idx_messages_username ON messages(username)",
    "CREATE INDEX IF NOT EXISTS idx_messages_upload_id ON messages(upload_id)",
    "CREATE INDEX IF NOT EXISTS idx_messages_suno_team ON messages(is_suno_team)",
    "CREATE INDEX IF NOT EXISTS idx_messages_week ON messages(week)",
    "CREATE INDEX IF NOT EXISTS idx_messages_search_vector ON messages USING GIN(search_vector)",
]:
    pg_cur.execute(stmt)
pg_conn.autocommit = False
print("Tables ready.\n")

# ── Helpers ────────────────────────────────────────────────────────────────────
def sqlite_count(table):
    return sqlite_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

def pg_count(table):
    pg_cur.execute(f"SELECT COUNT(*) FROM {table}")
    return pg_cur.fetchone()[0]

def already_done(table):
    """Returns True and prints a skip message if PG row count matches SQLite."""
    sq = sqlite_count(table)
    pg = pg_count(table)
    if pg >= sq and sq > 0:
        print(f"  {table}: already populated ({pg} rows) — skipping.\n")
        return True
    return False

def batch_insert(table, columns, rows, page=2000):
    if not rows:
        print(f"  {table}: no rows to insert.\n")
        return
    cols = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
    total = 0
    for i in range(0, len(rows), page):
        chunk = rows[i:i+page]
        psycopg2.extras.execute_batch(pg_cur, sql, chunk, page_size=page)
        total += len(chunk)
        print(f"  {table}: {total}/{len(rows)}", end="\r")
    print()

def sync_seq(table, id_col="id"):
    pg_cur.execute(
        f"SELECT setval(pg_get_serial_sequence('{table}', '{id_col}'), "
        f"COALESCE((SELECT MAX({id_col}) FROM {table}), 1))"
    )

# ── 1. uploads ─────────────────────────────────────────────────────────────────
print("Migrating uploads...")
if not already_done("uploads"):
    rows = sqlite_conn.execute(
        "SELECT id, filename, row_count, upload_time FROM uploads"
    ).fetchall()
    batch_insert("uploads", ["id","filename","row_count","upload_time"],
        [(r["id"], r["filename"], r["row_count"], r["upload_time"]) for r in rows])
    pg_conn.commit()
    print(f"  uploads done: {pg_count('uploads')} rows\n")

# ── 2. messages ────────────────────────────────────────────────────────────────
print("Migrating messages...")
sq_msg = sqlite_count("messages")
pg_msg = pg_count("messages")

# If messages exist but max(id) doesn't match SQLite, they were auto-sequenced —
# truncate and re-insert with explicit IDs so bookmarks FK works.
if pg_msg > 0:
    pg_cur.execute("SELECT MAX(id) FROM messages")
    pg_max_id = pg_cur.fetchone()[0] or 0
    sq_max_id = sqlite_conn.execute("SELECT MAX(id) FROM messages").fetchone()[0] or 0
    if pg_max_id != sq_max_id:
        print(f"  messages: IDs mismatch (pg_max={pg_max_id}, sqlite_max={sq_max_id})")
        print("  Truncating messages (and dependent tables) and re-inserting with correct IDs...")
        pg_conn.rollback()
        pg_cur.execute("TRUNCATE TABLE bookmark_code_highlights, bookmark_codes, bookmark_labels, bookmarks, messages RESTART IDENTITY CASCADE")
        pg_conn.commit()
        pg_msg = 0

if pg_msg >= sq_msg and sq_msg > 0:
    print(f"  messages: already populated ({pg_msg} rows) — skipping.\n")
else:
    print(f"  Loading {sq_msg} rows from SQLite (this takes ~1 min)...")
    rows = sqlite_conn.execute(
        """SELECT id, msg_uuid, author_id, username, date, content, attachments,
                  reactions, is_suno_team, week, month, upload_id, row_index, word_count
           FROM messages ORDER BY id"""
    ).fetchall()
    batch_insert(
        "messages",
        ["id","msg_uuid","author_id","username","date","content","attachments",
         "reactions","is_suno_team","week","month","upload_id","row_index","word_count"],
        [(r["id"], r["msg_uuid"], r["author_id"], r["username"], r["date"], r["content"],
          r["attachments"], r["reactions"], r["is_suno_team"], r["week"], r["month"],
          r["upload_id"], r["row_index"], r["word_count"]) for r in rows],
    )
    sync_seq("messages")
    pg_conn.commit()
    print(f"  messages done: {pg_count('messages')} rows\n")

# ── 3. users ───────────────────────────────────────────────────────────────────
print("Migrating users...")
if not already_done("users"):
    rows = sqlite_conn.execute(
        "SELECT id, username, password_hash, password_salt, is_admin, created_at, google_id, email FROM users"
    ).fetchall()
    batch_insert("users",
        ["id","username","password_hash","password_salt","is_admin","created_at","google_id","email"],
        [(r["id"], r["username"], r["password_hash"], r["password_salt"],
          bool(r["is_admin"]), r["created_at"], r["google_id"], r["email"]) for r in rows])
    sync_seq("users")
    pg_conn.commit()
    print(f"  users done: {pg_count('users')} rows\n")

# ── 4. settings ────────────────────────────────────────────────────────────────
print("Migrating settings...")
if not already_done("settings"):
    rows = sqlite_conn.execute("SELECT key, value FROM settings").fetchall()
    for r in rows:
        pg_cur.execute(
            "INSERT INTO settings (key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            (r["key"], r["value"]),
        )
    pg_conn.commit()
    print(f"  settings done: {pg_count('settings')} rows\n")

# ── 5. labels ──────────────────────────────────────────────────────────────────
print("Migrating labels...")
if not already_done("labels"):
    rows = sqlite_conn.execute("SELECT id, name, color, created_at FROM labels").fetchall()
    batch_insert("labels", ["id","name","color","created_at"],
        [(r["id"], r["name"], r["color"], r["created_at"]) for r in rows])
    sync_seq("labels")
    pg_conn.commit()
    print(f"  labels done: {pg_count('labels')} rows\n")

# ── 6. code_categories ─────────────────────────────────────────────────────────
print("Migrating code_categories...")
if not already_done("code_categories"):
    rows = sqlite_conn.execute(
        "SELECT id, name, color, parent_id, created_at FROM code_categories"
    ).fetchall()
    batch_insert("code_categories", ["id","name","color","parent_id","created_at"],
        [(r["id"], r["name"], r["color"], r["parent_id"], r["created_at"]) for r in rows])
    sync_seq("code_categories")
    pg_conn.commit()
    print(f"  code_categories done: {pg_count('code_categories')} rows\n")

# ── 7. codes ───────────────────────────────────────────────────────────────────
print("Migrating codes...")
if not already_done("codes"):
    rows = sqlite_conn.execute(
        "SELECT id, name, color, description, category_id, created_at FROM codes"
    ).fetchall()
    batch_insert("codes", ["id","name","color","description","category_id","created_at"],
        [(r["id"], r["name"], r["color"], r["description"], r["category_id"], r["created_at"]) for r in rows])
    sync_seq("codes")
    pg_conn.commit()
    print(f"  codes done: {pg_count('codes')} rows\n")

# ── 8. bookmarks ───────────────────────────────────────────────────────────────
print("Migrating bookmarks...")
if not already_done("bookmarks"):
    rows = sqlite_conn.execute(
        "SELECT id, msg_id, ctx_before, ctx_after, note, created_at, user_id FROM bookmarks"
    ).fetchall()
    batch_insert("bookmarks",
        ["id","msg_id","ctx_before","ctx_after","note","created_at","user_id"],
        [(r["id"], r["msg_id"], r["ctx_before"], r["ctx_after"],
          r["note"], r["created_at"], r["user_id"]) for r in rows])
    sync_seq("bookmarks")
    pg_conn.commit()
    print(f"  bookmarks done: {pg_count('bookmarks')} rows\n")

# ── 9. bookmark_labels ─────────────────────────────────────────────────────────
print("Migrating bookmark_labels...")
if not already_done("bookmark_labels"):
    rows = sqlite_conn.execute(
        "SELECT bookmark_id, label_id FROM bookmark_labels"
    ).fetchall()
    batch_insert("bookmark_labels", ["bookmark_id","label_id"],
        [(r["bookmark_id"], r["label_id"]) for r in rows])
    pg_conn.commit()
    print(f"  bookmark_labels done: {pg_count('bookmark_labels')} rows\n")

# ── 10. bookmark_codes ─────────────────────────────────────────────────────────
print("Migrating bookmark_codes...")
if not already_done("bookmark_codes"):
    rows = sqlite_conn.execute(
        "SELECT bookmark_id, code_id, highlighted_text FROM bookmark_codes"
    ).fetchall()
    batch_insert("bookmark_codes", ["bookmark_id","code_id","highlighted_text"],
        [(r["bookmark_id"], r["code_id"], r["highlighted_text"]) for r in rows])
    pg_conn.commit()
    print(f"  bookmark_codes done: {pg_count('bookmark_codes')} rows\n")

# ── 11. bookmark_code_highlights ───────────────────────────────────────────────
print("Migrating bookmark_code_highlights...")
if not already_done("bookmark_code_highlights"):
    rows = sqlite_conn.execute(
        "SELECT id, bookmark_id, code_id, highlighted_text, created_at FROM bookmark_code_highlights"
    ).fetchall()
    batch_insert("bookmark_code_highlights",
        ["id","bookmark_id","code_id","highlighted_text","created_at"],
        [(r["id"], r["bookmark_id"], r["code_id"], r["highlighted_text"], r["created_at"]) for r in rows])
    sync_seq("bookmark_code_highlights")
    pg_conn.commit()
    print(f"  bookmark_code_highlights done: {pg_count('bookmark_code_highlights')} rows\n")

# ── 12. embedded_uploads ───────────────────────────────────────────────────────
print("Migrating embedded_uploads...")
if not already_done("embedded_uploads"):
    rows = sqlite_conn.execute(
        "SELECT upload_id, model_id, embedded_at FROM embedded_uploads"
    ).fetchall()
    batch_insert("embedded_uploads", ["upload_id","model_id","embedded_at"],
        [(r["upload_id"], r["model_id"], r["embedded_at"]) for r in rows])
    pg_conn.commit()
    print(f"  embedded_uploads done: {pg_count('embedded_uploads')} rows\n")

# ── Done ───────────────────────────────────────────────────────────────────────
sqlite_conn.close()
pg_conn.close()
print("=" * 50)
print("Migration complete!")
print("=" * 50)
print("\nNext: start the Rust backend to verify:")
print("  cd backend && cargo run")
