"""
migrate_chroma_to_qdrant.py

Migrates a local ChromaDB persistent collection to a remote Qdrant instance
via an SSH tunnel.

Step 1 — open the tunnel (keep this terminal running):
    ssh -L 6333:localhost:6333 ubuntu@129.226.210.164 -N

Step 2 — run the migration:
    python migrate_chroma_to_qdrant.py

Traffic to 127.0.0.1:6333 is transparently forwarded to Qdrant running on
the remote host (129.226.210.164). No changes to the script are needed as
long as the tunnel is active.

Override any setting via environment variables or .env file.
Migration-specific vars (take priority over .env values):
    MIGRATION_QDRANT_URL      — destination Qdrant URL  (default: http://127.0.0.1:6333)
    MIGRATION_QDRANT_API_KEY  — destination API key     (default: QDRANT_API_KEY from .env)
"""

import math
import uuid
from typing import Any

import chromadb

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# -----------------------------
# Config
# -----------------------------

CHROMA_PATH       = os.environ.get("CHROMA_PATH", "./chroma_db")
CHROMA_COLLECTION = "discord_openai"

# Migration target: forwards through SSH tunnel  →  ubuntu@129.226.210.164
#   Open the tunnel first (in a separate terminal, keep it running):
#       ssh -L 6333:localhost:6333 ubuntu@129.226.210.164 -N
#   Then run this script — traffic to 127.0.0.1:6333 is transparently forwarded.
#   Override via MIGRATION_QDRANT_URL / MIGRATION_QDRANT_API_KEY env vars if needed.
QDRANT_URL        = os.environ.get("MIGRATION_QDRANT_URL") or os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_API_KEY    = os.environ.get("MIGRATION_QDRANT_API_KEY") or os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = "discord_openai"

BATCH_SIZE = 200        # points per upsert call — reduce if you hit timeouts
DISTANCE   = Distance.COSINE
TIMEOUT    = 60         # seconds per request


# -----------------------------
# Helpers
# -----------------------------

_UUID5_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # UUID namespace URL


def to_qdrant_id(chroma_id: str) -> str:
    """
    Qdrant accepts UUID strings or unsigned integers only.
    If the Chroma ID is already a valid UUID, pass it through.
    Otherwise derive a deterministic UUID5 from it.
    """
    try:
        return str(uuid.UUID(chroma_id))
    except ValueError:
        return str(uuid.uuid5(_UUID5_NS, chroma_id))


def normalize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """
    Qdrant payload values must be JSON-serialisable scalars, lists, or dicts.
    Drop None values; stringify anything else that isn't a supported type.
    """
    if not metadata:
        return {}
    clean: dict[str, Any] = {}
    for k, v in metadata.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool, list, dict)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    print("=" * 60)
    print("ChromaDB  →  Qdrant migration")
    print(f"  Source  : {CHROMA_PATH}  /  {CHROMA_COLLECTION}")
    print(f"  Target  : {QDRANT_URL}  /  {QDRANT_COLLECTION}")
    print("=" * 60)

    # ── 1. Open local Chroma ──────────────────────────────────────
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        chroma_col = chroma_client.get_collection(CHROMA_COLLECTION)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot open Chroma collection '{CHROMA_COLLECTION}' at '{CHROMA_PATH}': {exc}"
        ) from exc

    total = chroma_col.count()
    print(f"\nChroma records: {total}")
    if total == 0:
        print("Nothing to migrate.")
        return

    # ── 2. Fetch all IDs up-front (reliable pagination) ───────────
    print("Fetching all Chroma IDs …", end=" ", flush=True)
    id_result = chroma_col.get(include=[])          # no vectors/docs — fast
    all_chroma_ids: list[str] = id_result.get("ids", [])
    if not all_chroma_ids:
        raise RuntimeError("Chroma returned no IDs — collection may be empty or corrupted.")
    print(f"{len(all_chroma_ids)} IDs loaded")

    # ── 3. Infer vector size from a single sample ─────────────────
    sample = chroma_col.get(
        ids=[all_chroma_ids[0]],
        include=["embeddings"],
    )
    embs = sample.get("embeddings")
    if embs is None or len(embs) == 0:
        raise RuntimeError(
            "No embedding returned for sample record. "
            "Ensure the collection has embeddings stored."
        )
    vector_size = len(embs[0])
    print(f"Vector size   : {vector_size}")

    # ── 4. Connect to Qdrant (with pre-flight check) ──────────────
    print(f"\nConnecting to Qdrant at {QDRANT_URL} …", end=" ", flush=True)
    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
        timeout=TIMEOUT,
    )
    try:
        qdrant.get_collections()          # lightweight connectivity test
        print("OK")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Qdrant at '{QDRANT_URL}': {exc}\n"
            "Make sure the SSH tunnel is open and active:\n"
            "    ssh -L 6333:localhost:6333 ubuntu@129.226.210.164 -N\n"
            "Also check that Qdrant is running on the remote host and the API key is correct."
        ) from exc

    # ── 5. Ensure collection exists with correct dimensions ───────
    existing_collections = {c.name for c in qdrant.get_collections().collections}
    if QDRANT_COLLECTION in existing_collections:
        info = qdrant.get_collection(QDRANT_COLLECTION)
        existing_size = info.config.params.vectors.size  # type: ignore[union-attr]
        if existing_size != vector_size:
            raise RuntimeError(
                f"Qdrant collection '{QDRANT_COLLECTION}' already exists with "
                f"vector size {existing_size}, but Chroma has size {vector_size}. "
                "Delete the Qdrant collection first or choose a different name."
            )
        print(f"Qdrant collection '{QDRANT_COLLECTION}' already exists.")
    else:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=DISTANCE),
        )
        print(f"Created Qdrant collection '{QDRANT_COLLECTION}'.")

    # ── 6. Check which IDs are already in Qdrant (resume support) ─
    print("Checking already-migrated IDs in Qdrant …", end=" ", flush=True)
    already_in_qdrant: set[str] = set()
    scroll_offset = None
    while True:
        scroll_result, scroll_offset = qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=1000,
            offset=scroll_offset,
            with_payload=False,
            with_vectors=False,
        )
        for point in scroll_result:
            already_in_qdrant.add(str(point.id))
        if scroll_offset is None:
            break
    print(f"{len(already_in_qdrant)} already present")

    # Build the list of Chroma IDs that still need to be migrated
    pending_ids = [
        cid for cid in all_chroma_ids
        if to_qdrant_id(cid) not in already_in_qdrant
    ]
    skipped = len(all_chroma_ids) - len(pending_ids)
    if skipped:
        print(f"Skipping {skipped} already-migrated records.")
    if not pending_ids:
        print("Nothing left to migrate — already complete.")
        return
    print(f"Records to migrate: {len(pending_ids)}\n")

    # ── 7. Batch migrate ──────────────────────────────────────────
    num_batches = math.ceil(len(pending_ids) / BATCH_SIZE)
    migrated = 0

    for batch_idx in range(num_batches):
        batch_ids = pending_ids[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

        batch = chroma_col.get(
            ids=batch_ids,
            include=["embeddings", "documents", "metadatas"],
        )

        ids        = batch.get("ids") or []
        embeddings = batch.get("embeddings")
        documents  = batch.get("documents") or []
        metadatas  = batch.get("metadatas") or []

        if not ids or embeddings is None or len(embeddings) == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches}: no data, skipping.")
            continue

        points: list[PointStruct] = []
        for i, chroma_id in enumerate(ids):
            qdrant_id = to_qdrant_id(chroma_id)

            payload = normalize_metadata(metadatas[i] if i < len(metadatas) else None)

            # Always store original Chroma ID so you can cross-reference
            if qdrant_id != chroma_id:
                payload["_chroma_id"] = chroma_id

            doc = documents[i] if i < len(documents) else None
            if doc is not None:
                payload["document"] = doc

            points.append(
                PointStruct(
                    id=qdrant_id,
                    vector=list(embeddings[i]),  # ensure plain list, not numpy array
                    payload=payload,
                )
            )

        try:
            qdrant.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points,
                wait=True,
            )
        except Exception as exc:
            print(f"\n  ERROR on batch {batch_idx + 1}: {exc}")
            print("  Retrying once …", end=" ", flush=True)
            try:
                qdrant.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=points,
                    wait=True,
                )
                print("retry OK")
            except Exception as exc2:
                raise RuntimeError(
                    f"Batch {batch_idx + 1} failed after retry: {exc2}"
                ) from exc2

        migrated += len(points)
        pct = migrated / len(pending_ids) * 100
        print(f"  [{pct:5.1f}%]  {migrated}/{len(pending_ids)}  (batch {batch_idx + 1}/{num_batches})")

    # ── 8. Verify ─────────────────────────────────────────────────
    qdrant_count = qdrant.count(collection_name=QDRANT_COLLECTION).count
    print(f"\nMigration complete.")
    print(f"  Chroma  : {total} records")
    print(f"  Qdrant  : {qdrant_count} records")
    if qdrant_count < total:
        print(f"  WARNING : {total - qdrant_count} records may be missing — re-run to resume.")
    else:
        print("  All records migrated successfully.")


if __name__ == "__main__":
    main()