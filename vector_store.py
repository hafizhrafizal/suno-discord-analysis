"""
vector_store.py — vector-store backend wrappers and initialisation.

Exports:
  QdrantCollectionWrapper   — wraps qdrant-client
  ChromaCollectionWrapper   — wraps chromadb (persistent or HTTP)
  init_vector_store()       — factory: returns {model_id: wrapper} for the
                              configured VECTOR_DB backend
"""

import logging
from typing import Optional

from config import (
    CHROMA_AUTH_TOKEN,
    CHROMA_HOST,
    CHROMA_PATH,
    CHROMA_PORT,
    CHROMA_SSL,
    EMBEDDING_MODELS,
    QDRANT_API_KEY,
    QDRANT_URL,
    VECTOR_DB,
)
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)


# ── Qdrant wrapper ────────────────────────────────────────────────────────────

class QdrantCollectionWrapper:
    """
    Thin wrapper around QdrantClient that exposes a uniform collection
    interface (count / get / upsert / query / delete).

    Supports qdrant-client + Qdrant server version mismatches:
    - New client (≥1.9) + new server (≥1.7): uses query_points()
    - New client (≥1.9) + OLD server (<1.7): falls back to REST /search
    - Old client (<1.7): uses search()
    """

    def __init__(self, client: QdrantClient, collection_name: str,
                 url: str = "", api_key: str = ""):
        self._client = client
        self._name = collection_name
        self._url = url.rstrip("/")    # stored for REST fallback
        self._api_key = api_key

    @staticmethod
    def _where_to_filter(where: dict) -> Filter:
        """Convert a Chroma-style where dict to a Qdrant Filter."""
        conditions = []
        for key, condition in where.items():
            value = condition.get("$eq", condition) if isinstance(condition, dict) else condition
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=conditions)

    def count(self) -> int:
        try:
            return self._client.count(collection_name=self._name).count
        except Exception:
            return 0

    def get(self, ids=None, where=None, limit=None, include=None):
        """
        Return a Chroma-like dict with keys such as:
        - ids
        - embeddings (optional)
        """
        include = include or []
        want_vectors = "embeddings" in include

        if ids is not None:
            records = self._client.retrieve(
                collection_name=self._name,
                ids=ids,
                with_payload=True,
                with_vectors=want_vectors,
            )
            result = {"ids": [str(r.id) for r in records]}
            if want_vectors:
                result["embeddings"] = [r.vector for r in records]
            return result

        if where is not None:
            qfilter = self._where_to_filter(where)
            effective_limit = limit if limit is not None else 10_000
            points, _ = self._client.scroll(
                collection_name=self._name,
                scroll_filter=qfilter,
                limit=effective_limit,
                with_payload=True,
                with_vectors=want_vectors,
            )
            result = {"ids": [str(p.id) for p in points]}
            if want_vectors:
                result["embeddings"] = [p.vector for p in points]
            return result

        # fallback: return first N points if requested without filters
        effective_limit = limit if limit is not None else 10_000
        points, _ = self._client.scroll(
            collection_name=self._name,
            limit=effective_limit,
            with_payload=True,
            with_vectors=want_vectors,
        )
        result = {"ids": [str(p.id) for p in points]}
        if want_vectors:
            result["embeddings"] = [p.vector for p in points]
        return result

    def upsert(self, embeddings, documents, ids, metadatas):
        points = []
        for emb, doc, pid, meta in zip(embeddings, documents, ids, metadatas):
            payload = dict(meta or {})
            payload["document"] = doc
            points.append(
                PointStruct(
                    id=str(pid),
                    vector=emb,
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self._name,
            points=points,
            wait=True,
        )

    def _search_via_rest(self, vector: list, n_results: int) -> dict:
        """
        Call the classic POST /collections/{name}/points/search endpoint directly.
        Works on ALL Qdrant server versions and bypasses qdrant-client versioning.
        Uses httpx (already a qdrant-client dependency) for reliable HTTP.
        """
        import httpx
        headers: dict = {}
        if self._api_key:
            headers["api-key"] = self._api_key
        url = f"{self._url}/collections/{self._name}/points/search"
        with httpx.Client(timeout=60.0, verify=False) as client:
            resp = client.post(url, json={
                "vector": vector,
                "limit": n_results,
                "with_payload": False,
                "with_vector": False,
            }, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        hits = data.get("result", [])
        ids = [str(h["id"]) for h in hits]
        distances = [round(1.0 - float(h["score"]), 6) for h in hits]
        return {"ids": [ids], "distances": [distances]}

    def query(self, query_embeddings, n_results):
        """
        Return a Chroma-compatible shape:
        {"ids": [[...]], "distances": [[...]]}

        Fallback chain:
          1. query_points()  — qdrant-client ≥1.7, Qdrant server ≥1.7
          2. search()        — qdrant-client <1.7 (legacy)
          3. REST /search    — new client + old server (server <1.7, returns 404 on /query)
        """
        vector = list(query_embeddings[0])

        # ── Attempt 1: modern query_points (client ≥ 1.7) ────────────────────
        if hasattr(self._client, "query_points"):
            try:
                result = self._client.query_points(
                    collection_name=self._name,
                    query=vector,
                    limit=n_results,
                    with_payload=False,
                    with_vectors=False,
                )
                hits = result.points
                ids = [str(h.id) for h in hits]
                distances = [round(1.0 - float(h.score), 6) for h in hits]
                return {"ids": [ids], "distances": [distances]}
            except Exception as exc:
                # 404 = server too old for /query endpoint → fall through
                if "404" not in str(exc) and "Not Found" not in str(exc):
                    raise
                logger.warning(
                    "query_points() got 404 — Qdrant server is older than 1.7. "
                    "Falling back to REST /search. Consider upgrading your Qdrant server."
                )

        # ── Attempt 2: legacy search() (client < 1.7) ─────────────────────────
        if hasattr(self._client, "search"):
            hits = self._client.search(
                collection_name=self._name,
                query_vector=vector,
                limit=n_results,
                with_payload=False,
                with_vectors=False,
            )
            ids = [str(h.id) for h in hits]
            distances = [round(1.0 - float(h.score), 6) for h in hits]
            return {"ids": [ids], "distances": [distances]}

        # ── Attempt 3: direct REST /search (works on all server versions) ─────
        if self._url:
            return self._search_via_rest(vector, n_results)

        raise RuntimeError(
            "Cannot perform vector search: qdrant-client has neither query_points() "
            "nor search(), and no QDRANT_URL is available for a REST fallback."
        )

    def delete(self, ids=None, where=None):
        if ids is not None:
            self._client.delete(
                collection_name=self._name,
                points_selector=PointIdsList(points=[str(i) for i in ids]),
                wait=True,
            )
        elif where is not None:
            self._client.delete(
                collection_name=self._name,
                points_selector=FilterSelector(filter=self._where_to_filter(where)),
                wait=True,
            )

# ── ChromaDB wrapper (persistent + HTTP share the same Collection API) ────────

# SQLite (used by ChromaDB persistent) caps bind variables at 999.
# Stay well clear of the limit for both query and bulk-get operations.
_CHROMA_SAFE_BATCH = 900


class ChromaCollectionWrapper:
    """
    Thin wrapper around a native ChromaDB Collection.
    Exposes the same interface as QdrantCollectionWrapper.

    SQLite safety: ChromaDB persistent uses SQLite internally, which raises
    "too many SQL variables" when n_results or an id list exceeds ~999.
    query() caps n_results at _CHROMA_SAFE_BATCH; get() chunks large id lists
    and merges the results transparently.
    """

    def __init__(self, col):
        self._col = col

    def count(self) -> int:
        try:
            return self._col.count()
        except Exception:
            return 0

    def get(self, ids=None, where=None, limit=None, include=None):
        include = include or []
        # Chunk large id lists to avoid SQLite variable limit
        if ids is not None and len(ids) > _CHROMA_SAFE_BATCH:
            want_vectors = "embeddings" in include
            merged_ids: list = []
            merged_embs: list = []
            for i in range(0, len(ids), _CHROMA_SAFE_BATCH):
                chunk = ids[i : i + _CHROMA_SAFE_BATCH]
                r = self._col.get(ids=chunk, include=include)
                merged_ids.extend(r.get("ids") or [])
                if want_vectors:
                    embs = r.get("embeddings")
                    if embs is not None:
                        merged_embs.extend(embs)
            result: dict = {"ids": merged_ids}
            if want_vectors:
                result["embeddings"] = merged_embs
            return result

        kwargs: dict = {}
        if ids     is not None: kwargs["ids"]     = ids
        if where   is not None: kwargs["where"]   = where
        if limit   is not None: kwargs["limit"]   = limit
        if include:             kwargs["include"] = include
        return self._col.get(**kwargs)

    def upsert(self, embeddings, documents, ids, metadatas):
        self._col.upsert(
            embeddings=embeddings,
            documents=documents,
            ids=ids,
            metadatas=metadatas,
        )

    def query(self, query_embeddings, n_results):
        # Cap n_results to avoid SQLite "too many SQL variables" in persistent backend
        safe_n = min(n_results, _CHROMA_SAFE_BATCH)
        return self._col.query(query_embeddings=query_embeddings, n_results=safe_n)

    def delete(self, ids=None, where=None):
        kwargs: dict = {}
        if ids   is not None: kwargs["ids"]   = ids
        if where is not None: kwargs["where"] = where
        self._col.delete(**kwargs)


# ── Per-backend init helpers ──────────────────────────────────────────────────

def _init_qdrant() -> dict:
    if not QDRANT_URL:
        logger.error("QDRANT_URL is not set — set it in .env to use the Qdrant backend.")
        return {}

    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None, timeout=60)
        logger.info("Vector backend: Qdrant (%s)", QDRANT_URL)
    except Exception as exc:
        logger.error("Failed to connect to Qdrant: %s", exc)
        return {}

    try:
        existing = {c.name for c in client.get_collections().collections}
        logger.info("Existing Qdrant collections: %s", existing)
    except Exception as exc:
        logger.error("Failed to list Qdrant collections: %s", exc)
        existing = set()

    cols: dict = {}
    for model_id, cfg in EMBEDDING_MODELS.items():
        cname = cfg["collection"]
        if cname not in existing:
            # Collection may not exist yet — try to create it.
            # If it already exists (e.g. get_collections() failed transiently and
            # returned an empty set), the create call will fail; that is fine —
            # we still wrap the existing collection below.
            try:
                client.create_collection(
                    collection_name=cname,
                    vectors_config=VectorParams(
                        size=cfg["dims"],
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection: %s", cname)
            except Exception as exc:
                logger.warning(
                    "Could not create Qdrant collection %s (%s) — "
                    "assuming it already exists and continuing.",
                    cname, exc,
                )

        # Always wrap the collection — even if create_collection() failed above,
        # the collection likely exists from a previous run or migration.
        try:
            wrapper = QdrantCollectionWrapper(
                client, cname,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY or "",
            )
            n = wrapper.count()
            cols[model_id] = wrapper
            logger.info(
                "Registered Qdrant collection: %s → model '%s'  (%d vectors)",
                cname, model_id, n,
            )
        except Exception as exc:
            logger.error("Failed to wrap Qdrant collection %s: %s", cname, exc)

    return cols


def _init_chroma_persistent() -> dict:
    try:
        import chromadb as _chromadb
    except ImportError:
        logger.error("chromadb is not installed. Run: pip install chromadb")
        return {}
    logger.info("Vector backend: ChromaDB persistent  (path=%s)", CHROMA_PATH)
    client = _chromadb.PersistentClient(path=CHROMA_PATH)
    cols: dict = {}
    for model_id, cfg in EMBEDDING_MODELS.items():
        try:
            cols[model_id] = ChromaCollectionWrapper(
                client.get_or_create_collection(
                    name=cfg["collection"],
                    metadata={"hnsw:space": "cosine"},
                )
            )
        except Exception as exc:
            logger.error("Failed to init ChromaDB persistent collection %s: %s",
                         cfg["collection"], exc)
    return cols


def _init_chroma_http() -> dict:
    try:
        import chromadb as _chromadb
    except ImportError:
        logger.error("chromadb is not installed. Run: pip install chromadb")
        return {}

    logger.info("Vector backend: ChromaDB HTTP  (%s:%s  ssl=%s)",
                CHROMA_HOST, CHROMA_PORT, CHROMA_SSL)

    client_kwargs: dict = dict(host=CHROMA_HOST, port=CHROMA_PORT, ssl=CHROMA_SSL)
    if CHROMA_AUTH_TOKEN:
        client_kwargs["headers"] = {"Authorization": f"Bearer {CHROMA_AUTH_TOKEN}"}

    try:
        client = _chromadb.HttpClient(**client_kwargs)
        client.heartbeat()
    except Exception as exc:
        logger.error("Cannot connect to ChromaDB HTTP server: %s", exc)
        return {}

    cols: dict = {}
    for model_id, cfg in EMBEDDING_MODELS.items():
        try:
            cols[model_id] = ChromaCollectionWrapper(
                client.get_or_create_collection(
                    name=cfg["collection"],
                    metadata={"hnsw:space": "cosine"},
                )
            )
        except Exception as exc:
            logger.error("Failed to init ChromaDB HTTP collection %s: %s",
                         cfg["collection"], exc)
    return cols


# ── Public factory ────────────────────────────────────────────────────────────

def init_vector_store() -> dict:
    """Return {model_id: *CollectionWrapper} for the configured VECTOR_DB backend."""
    if VECTOR_DB == "qdrant":
        return _init_qdrant()
    if VECTOR_DB == "chroma_persistent":
        return _init_chroma_persistent()
    if VECTOR_DB == "chroma_http":
        return _init_chroma_http()
    logger.error(
        "Unknown VECTOR_DB='%s'. Valid values: qdrant, chroma_persistent, chroma_http",
        VECTOR_DB,
    )
    return {}
