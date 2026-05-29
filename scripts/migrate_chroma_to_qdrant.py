"""
Migrate vectors from a local ChromaDB persistent collection to a Qdrant server.

Requirements:
    pip install chromadb qdrant-client tqdm

Usage:
    python migrate_chroma_to_qdrant.py \
        --chroma-path ./chroma_db \
        --chroma-collection discord_openai \
        --qdrant-url http://localhost:6333 \
        --qdrant-collection discord_openai \
        --vector-size 1536 \
        --batch-size 200

For remote Qdrant (via SSH tunnel):
    # First open a tunnel on your local machine:
    #   ssh -L 6333:localhost:6333 user@your-vps-ip
    # Then run this script with --qdrant-url http://localhost:6333
"""

import argparse
import sys
import uuid
import time

def main():
    parser = argparse.ArgumentParser(description="Migrate ChromaDB → Qdrant")
    parser.add_argument("--chroma-path", default="./chroma_db", help="Path to ChromaDB persistent directory")
    parser.add_argument("--chroma-collection", default="discord_openai", help="ChromaDB collection name")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant server URL")
    parser.add_argument("--qdrant-api-key", default=None, help="Qdrant API key (leave blank for self-hosted)")
    parser.add_argument("--qdrant-collection", default="discord_openai", help="Qdrant collection name")
    parser.add_argument("--vector-size", type=int, default=1536, help="Embedding dimension (1536 for text-embedding-3-small)")
    parser.add_argument("--batch-size", type=int, default=200, help="Vectors per upsert batch")
    parser.add_argument("--dry-run", action="store_true", help="Count vectors without uploading")
    args = parser.parse_args()

    try:
        import chromadb
    except ImportError:
        print("ERROR: chromadb not installed. Run: pip install chromadb")
        sys.exit(1)

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance, PointStruct
    except ImportError:
        print("ERROR: qdrant-client not installed. Run: pip install qdrant-client")
        sys.exit(1)

    try:
        from tqdm import tqdm
        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False

    # ── Connect to ChromaDB ────────────────────────────────────────────────────
    print(f"Connecting to ChromaDB at {args.chroma_path} ...")
    chroma_client = chromadb.PersistentClient(path=args.chroma_path)
    try:
        collection = chroma_client.get_collection(args.chroma_collection)
    except Exception as e:
        print(f"ERROR: Could not open ChromaDB collection '{args.chroma_collection}': {e}")
        sys.exit(1)

    total = collection.count()
    print(f"  Found {total:,} vectors in ChromaDB collection '{args.chroma_collection}'")

    if args.dry_run:
        print("Dry run complete. No data uploaded.")
        return

    if total == 0:
        print("Collection is empty — nothing to migrate.")
        return

    # ── Connect to Qdrant ─────────────────────────────────────────────────────
    print(f"\nConnecting to Qdrant at {args.qdrant_url} ...")
    qdrant_kwargs = {"url": args.qdrant_url, "timeout": 30}
    if args.qdrant_api_key:
        qdrant_kwargs["api_key"] = args.qdrant_api_key
    qdrant = QdrantClient(**qdrant_kwargs)

    # Create or recreate collection
    existing = [c.name for c in qdrant.get_collections().collections]
    if args.qdrant_collection in existing:
        print(f"  Collection '{args.qdrant_collection}' already exists.")
        confirm = input("  Recreate it (ALL existing Qdrant data will be lost)? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)
        qdrant.delete_collection(args.qdrant_collection)

    print(f"  Creating Qdrant collection '{args.qdrant_collection}' (dim={args.vector_size}, cosine) ...")
    qdrant.create_collection(
        collection_name=args.qdrant_collection,
        vectors_config=VectorParams(size=args.vector_size, distance=Distance.COSINE),
    )

    # ── Migrate in batches ────────────────────────────────────────────────────
    print(f"\nMigrating {total:,} vectors in batches of {args.batch_size} ...")
    offset = 0
    uploaded = 0
    errors = 0
    iterator = range(0, total, args.batch_size)
    if HAS_TQDM:
        iterator = tqdm(iterator, unit="batch")

    while offset < total:
        batch_ids = []
        limit = min(args.batch_size, total - offset)

        # ChromaDB get() with offset/limit
        result = collection.get(
            limit=limit,
            offset=offset,
            include=["embeddings", "documents", "metadatas"],
        )

        points = []
        for chroma_id, embedding, document, metadata in zip(
            result["ids"],
            result["embeddings"],
            result["documents"],
            result["metadatas"],
        ):
            # Convert ChromaDB string ID to a deterministic UUID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, chroma_id))

            payload = {}
            if metadata:
                payload.update(metadata)
            if document:
                payload["document"] = document
            payload["chroma_id"] = chroma_id  # keep original ID for reference

            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

        try:
            qdrant.upsert(collection_name=args.qdrant_collection, points=points)
            uploaded += len(points)
        except Exception as e:
            print(f"\nWARN: Batch at offset {offset} failed: {e}")
            errors += 1

        offset += limit
        if HAS_TQDM:
            iterator.update(1)
        else:
            print(f"  {uploaded:,}/{total:,} uploaded ...", end="\r")

    print(f"\n\nDone! Uploaded {uploaded:,} vectors. Errors: {errors}.")

    # ── Verify ────────────────────────────────────────────────────────────────
    time.sleep(1)
    info = qdrant.get_collection(args.qdrant_collection)
    qdrant_count = info.vectors_count
    print(f"Qdrant reports {qdrant_count:,} vectors in '{args.qdrant_collection}'.")
    if qdrant_count != total:
        print(f"WARNING: Expected {total:,} but got {qdrant_count:,}. Re-run to retry failed batches.")
    else:
        print("Migration verified successfully.")


if __name__ == "__main__":
    main()
