"""
routers/codes.py — QDA code management (replaces labels.py).

  GET    /api/codes
  POST   /api/codes
  PUT    /api/codes/{code_id}
  DELETE /api/codes/{code_id}
  POST   /api/codes/merge

  GET    /api/code-categories
  POST   /api/code-categories
  PUT    /api/code-categories/{category_id}
  DELETE /api/code-categories/{category_id}

  Back-compat aliases (used by old JS still in cache):
  GET    /api/labels   → list_codes
  POST   /api/labels   → create_code
  DELETE /api/labels/{id} → delete_code
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException

from database import get_db

router = APIRouter()


# ── Code Categories ───────────────────────────────────────────────────────────

@router.get("/api/code-categories")
async def list_code_categories():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, color, parent_id, created_at FROM code_categories ORDER BY name COLLATE NOCASE"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.post("/api/code-categories")
async def create_code_category(body: dict):
    name      = (body.get("name") or "").strip()
    color     = (body.get("color") or "#94a3b8").strip()
    parent_id = body.get("parent_id")
    if not name:
        raise HTTPException(400, "name is required")
    conn = get_db()
    try:
        cur = conn.execute(
            "INSERT INTO code_categories (name, color, parent_id, created_at) VALUES (?,?,?,?)",
            (name, color, parent_id, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return {"id": cur.lastrowid, "name": name, "color": color, "parent_id": parent_id,
                "created_at": datetime.utcnow().isoformat()}
    except Exception:
        raise HTTPException(409, "Category name already exists")
    finally:
        conn.close()


@router.put("/api/code-categories/{category_id}")
async def update_code_category(category_id: int, body: dict):
    name  = (body.get("name") or "").strip()
    color = body.get("color")
    if not name:
        raise HTTPException(400, "name is required")
    conn = get_db()
    try:
        sets, params = ["name = ?"], [name]
        if color:
            sets.append("color = ?")
            params.append(color)
        if "parent_id" in body:
            pid = body["parent_id"]
            # Prevent cycles: a category cannot be its own ancestor
            if pid is not None and pid == category_id:
                raise HTTPException(400, "A category cannot be its own parent")
            sets.append("parent_id = ?")
            params.append(pid)
        params.append(category_id)
        cur = conn.execute(f"UPDATE code_categories SET {', '.join(sets)} WHERE id = ?", params)
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(404, "Category not found")
        return {"status": "updated"}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(409, "Category name already exists")
    finally:
        conn.close()


@router.delete("/api/code-categories/{category_id}")
async def delete_code_category(category_id: int):
    conn = get_db()
    # Re-parent children to the deleted category's own parent
    parent = conn.execute(
        "SELECT parent_id FROM code_categories WHERE id = ?", (category_id,)
    ).fetchone()
    grandparent_id = parent["parent_id"] if parent else None
    conn.execute(
        "UPDATE code_categories SET parent_id = ? WHERE parent_id = ?",
        (grandparent_id, category_id),
    )
    conn.execute("UPDATE codes SET category_id = NULL WHERE category_id = ?", (category_id,))
    cur = conn.execute("DELETE FROM code_categories WHERE id = ?", (category_id,))
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        raise HTTPException(404, "Category not found")
    return {"status": "deleted"}


# ── Codes ─────────────────────────────────────────────────────────────────────

@router.get("/api/codes")
async def list_codes():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, color, description, category_id, created_at FROM codes ORDER BY name COLLATE NOCASE"
    ).fetchall()

    # Groundedness: bookmarks per code
    g_rows = conn.execute(
        "SELECT code_id, COUNT(*) AS cnt FROM bookmark_codes GROUP BY code_id"
    ).fetchall()
    groundedness = {r["code_id"]: r["cnt"] for r in g_rows}

    # Density: distinct co-occurring codes per code
    d_rows = conn.execute(
        """SELECT bc1.code_id, COUNT(DISTINCT bc2.code_id) AS cnt
           FROM bookmark_codes bc1
           JOIN bookmark_codes bc2
             ON bc1.bookmark_id = bc2.bookmark_id AND bc2.code_id != bc1.code_id
           GROUP BY bc1.code_id"""
    ).fetchall()
    density = {r["code_id"]: r["cnt"] for r in d_rows}

    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["groundedness"] = groundedness.get(r["id"], 0)
        d["density"]      = density.get(r["id"], 0)
        result.append(d)
    return result


@router.post("/api/codes")
async def create_code(body: dict):
    name        = (body.get("name") or "").strip()
    color       = (body.get("color") or "#0d3e7f").strip()
    description = (body.get("description") or "").strip()
    category_id = body.get("category_id")
    if not name:
        raise HTTPException(400, "name is required")
    conn = get_db()
    try:
        cur = conn.execute(
            "INSERT INTO codes (name, color, description, category_id, created_at) VALUES (?,?,?,?,?)",
            (name, color, description, category_id, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return {
            "id": cur.lastrowid, "name": name, "color": color,
            "description": description, "category_id": category_id,
            "groundedness": 0, "density": 0,
        }
    except Exception:
        raise HTTPException(409, "Code name already exists")
    finally:
        conn.close()


@router.put("/api/codes/{code_id}")
async def update_code(code_id: int, body: dict):
    conn = get_db()
    if not conn.execute("SELECT id FROM codes WHERE id = ?", (code_id,)).fetchone():
        conn.close()
        raise HTTPException(404, "Code not found")

    sets, params = [], []
    if "name" in body:
        name = (body["name"] or "").strip()
        if not name:
            conn.close()
            raise HTTPException(400, "name cannot be empty")
        sets.append("name = ?");  params.append(name)
    if "color" in body:
        sets.append("color = ?"); params.append(body["color"])
    if "description" in body:
        sets.append("description = ?"); params.append((body["description"] or "").strip())
    if "category_id" in body:
        sets.append("category_id = ?"); params.append(body["category_id"])

    if not sets:
        conn.close()
        return {"status": "no-op"}

    params.append(code_id)
    try:
        conn.execute(f"UPDATE codes SET {', '.join(sets)} WHERE id = ?", params)
        conn.commit()
    except Exception:
        conn.close()
        raise HTTPException(409, "Code name already exists")
    conn.close()
    return {"status": "updated"}


@router.delete("/api/codes/{code_id}")
async def delete_code(code_id: int):
    conn = get_db()
    cur  = conn.execute("DELETE FROM codes WHERE id = ?", (code_id,))
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        raise HTTPException(404, "Code not found")
    return {"status": "deleted"}


@router.post("/api/codes/merge")
async def merge_codes(body: dict):
    """Merge source_id into target_id: reassign all bookmarks, then delete source."""
    source_id = body.get("source_id")
    target_id = body.get("target_id")
    if not isinstance(source_id, int) or not isinstance(target_id, int):
        raise HTTPException(400, "source_id and target_id (int) are required")
    if source_id == target_id:
        raise HTTPException(400, "source and target must be different")

    conn = get_db()
    if not conn.execute("SELECT id FROM codes WHERE id = ?", (source_id,)).fetchone():
        conn.close(); raise HTTPException(404, f"Source code {source_id} not found")
    if not conn.execute("SELECT id FROM codes WHERE id = ?", (target_id,)).fetchone():
        conn.close(); raise HTTPException(404, f"Target code {target_id} not found")

    src_bms = conn.execute(
        "SELECT bookmark_id FROM bookmark_codes WHERE code_id = ?", (source_id,)
    ).fetchall()

    migrated = 0
    for row in src_bms:
        cur = conn.execute(
            "INSERT OR IGNORE INTO bookmark_codes (bookmark_id, code_id) VALUES (?,?)",
            (row["bookmark_id"], target_id),
        )
        migrated += cur.rowcount

    conn.execute("DELETE FROM codes WHERE id = ?", (source_id,))
    conn.commit()
    conn.close()
    return {"status": "merged", "migrated_bookmarks": migrated}


@router.get("/api/codes/{code_id}/bookmarks")
async def list_code_bookmarks(code_id: int):
    """Return bookmarks (with excerpt text) that carry this code."""
    conn = get_db()
    if not conn.execute("SELECT id FROM codes WHERE id = ?", (code_id,)).fetchone():
        conn.close()
        raise HTTPException(404, "Code not found")
    rows = conn.execute(
        """SELECT b.id AS bookmark_id, b.note, b.created_at,
                  m.content, m.username, m.date, m.author_id
           FROM bookmark_codes bc
           JOIN bookmarks b ON b.id = bc.bookmark_id
           JOIN messages  m ON m.id = b.msg_id
           WHERE bc.code_id = ?
           ORDER BY b.created_at DESC""",
        (code_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Back-compat aliases ───────────────────────────────────────────────────────

@router.get("/api/labels")
async def list_labels_compat():
    return await list_codes()

@router.post("/api/labels")
async def create_label_compat(body: dict):
    return await create_code(body)

@router.delete("/api/labels/{label_id}")
async def delete_label_compat(label_id: int):
    return await delete_code(label_id)
