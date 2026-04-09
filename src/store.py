"""
store.py — Qdrant collection management, upsert, search.

Collection naming:
  PDF:  rag_pdf__<sanitised_stem>   e.g. rag_pdf__my_document
  HTML: rag_html__<sanitised_slug>  e.g. rag_html__example_com_article

Each collection has two named vector spaces:
  "text"  — TEXT_DIM  cosine  (CLIP text or nomic-embed-text)
  "image" — IMAGE_DIM cosine  (CLIP image)

Both text AND image chunks live in the same collection.
content_type payload field ("text" or "image") distinguishes them.
"""

import re
import uuid
import hashlib
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    PayloadSchemaType,
)

from logger import get_logger

log = get_logger(__name__)

from config import (
    QDRANT_HOST, QDRANT_PORT,
    PDF_PREFIX, HTML_PREFIX,
    TEXT_DIM, IMAGE_DIM,
)

CLAUDE_PREFIX = "rag_claude__"


# ── Collection naming ─────────────────────────────────────────────────────────

def _sanitise(name: str, max_len: int = 50) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    if name and name[0].isdigit():
        name = "d_" + name
    return name[:max_len] or "unknown"


def collection_for_pdf(file_path: str) -> str:
    stem = Path(file_path).stem
    return PDF_PREFIX + _sanitise(stem)


def collection_for_html(url: str) -> str:
    try:
        p = urlparse(url)
        slug = (p.netloc + p.path).rstrip("/")
    except Exception:
        slug = url
    return HTML_PREFIX + _sanitise(slug)


def stable_id(source: str, index: int) -> str:
    """Deterministic UUID for a chunk. source = file_path or url."""
    raw = f"{source}::{index}"
    return str(uuid.UUID(hashlib.sha256(raw.encode()).hexdigest()[:32]))


# ── QdrantStore ───────────────────────────────────────────────────────────────

class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            timeout=30,
            check_compatibility=False,
        )

    # ── Collection management ─────────────────────────────────────────────────

    def ensure_collection(self, name: str, text_dim: int = TEXT_DIM, image_dim: int = IMAGE_DIM):
        """Create collection with text + image named vectors if it doesn't exist."""
        try:
            if self.client.collection_exists(name):
                return
        except Exception:
            pass

        log.info("Creating collection", coll=name)
        self.client.create_collection(
            collection_name=name,
            vectors_config={
                "text":  VectorParams(size=text_dim,  distance=Distance.COSINE),
                "image": VectorParams(size=image_dim, distance=Distance.COSINE),
            },
        )
        # Payload indexes for filtering
        for field, schema in [
            ("content_type",    PayloadSchemaType.KEYWORD),
            ("source_type",     PayloadSchemaType.KEYWORD),
            ("page_number",     PayloadSchemaType.INTEGER),
            ("file_path",       PayloadSchemaType.KEYWORD),
            ("url",             PayloadSchemaType.KEYWORD),
        ]:
            try:
                self.client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                pass

    def collection_exists(self, name: str) -> bool:
        try:
            return self.client.collection_exists(name)
        except Exception:
            return False

    def point_count(self, name: str) -> int:
        """
        Returns exact point count using count() API.
        get_collection().points_count can be stale in Qdrant — count() is authoritative.
        """
        try:
            result = self.client.count(collection_name=name, exact=True)
            return result.count
        except Exception:
            return 0

    def all_collections(self) -> list[str]:
        try:
            return [c.name for c in self.client.get_collections().collections]
        except Exception:
            return []

    def pdf_collections(self) -> list[str]:
        return [c for c in self.all_collections() if c.startswith(PDF_PREFIX)]

    def html_collections(self) -> list[str]:
        return [c for c in self.all_collections() if c.startswith(HTML_PREFIX)]

    def claude_collections(self) -> list[str]:
        return [c for c in self.all_collections() if c.startswith(CLAUDE_PREFIX)]

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert(self, collection: str, points: list[PointStruct]) -> int:
        """Upsert a list of pre-built PointStruct objects. Returns count stored."""
        if not points:
            return 0
        BATCH = 64
        stored = 0
        for i in range(0, len(points), BATCH):
            batch = points[i:i + BATCH]
            try:
                self.client.upsert(
                    collection_name=collection,
                    points=batch,
                    wait=True,
                )
                stored += len(batch)
            except Exception as e:
                log.error("Upsert error", coll=collection, error=str(e))
        return stored

    def build_point(
        self,
        point_id: str,
        text_vec: Optional[list],
        image_vec: Optional[list],
        payload: dict,
        text_dim: int = TEXT_DIM,
        image_dim: int = IMAGE_DIM,
    ) -> Optional[PointStruct]:
        """Build a PointStruct with text and/or image vectors."""
        vectors = {}
        if text_vec and len(text_vec) == text_dim:
            vectors["text"] = text_vec
        if image_vec and len(image_vec) == image_dim:
            vectors["image"] = image_vec

        # Must have at least one vector
        if not vectors:
            return None

        # If only one vector type, mirror it to the other space
        # so the collection doesn't reject the point
        if "text" not in vectors and "image" in vectors:
            # Pad/truncate image vec to text_dim if dimensions differ
            if text_dim == image_dim:
                vectors["text"] = vectors["image"]
            else:
                vectors["text"] = (vectors["image"] + [0.0] * text_dim)[:text_dim]
        if "image" not in vectors and "text" in vectors:
            if text_dim == image_dim:
                vectors["image"] = vectors["text"]
            else:
                vectors["image"] = (vectors["text"] + [0.0] * image_dim)[:image_dim]

        return PointStruct(
            id=point_id,
            vector=vectors,
            payload=payload,
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        collection: str,
        query_vec: list,
        vector_name: str = "text",
        top_k: int = 20,
    ) -> list[dict]:
        """
        Search one collection. Returns list of {chunk_id, score, payload}.
        Handles empty/non-existent collections silently — they return [].
        """
        try:
            result = self.client.query_points(
                collection_name=collection,
                query=query_vec,
                using=vector_name,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            return [
                {
                    "chunk_id": str(hit.id).replace("-", ""),
                    "score":    hit.score,
                    "payload":  hit.payload or {},
                }
                for hit in result.points
            ]
        except Exception as e:
            err = str(e)
            # 404 = collection exists but is empty or vector namespace missing — silent skip
            if "404" in err or "Not Found" in err:
                return []
            log.error("Search error", coll=collection, vector=vector_name, error=err)
            return []

    def search_all_pdf(self, query_vec: list, vector_name: str = "text", top_k: int = 20) -> list[dict]:
        """Search across all PDF collections and merge results."""
        results = []
        for coll in self.pdf_collections():
            results.extend(self.search(coll, query_vec, vector_name, top_k))
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def search_all_html(self, query_vec: list, vector_name: str = "text", top_k: int = 20) -> list[dict]:
        """Search across all HTML collections and merge results."""
        results = []
        for coll in self.html_collections():
            results.extend(self.search(coll, query_vec, vector_name, top_k))
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def search_all_claude(self, query_vec: list, vector_name: str = "text", top_k: int = 20) -> list[dict]:
        """Search across all Claude chat collections."""
        results = []
        for coll in self.claude_collections():
            results.extend(self.search(coll, query_vec, vector_name, top_k))
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def fetch_payloads(self, chunk_ids: list[str]) -> dict[str, dict]:
        """Fetch payloads for a list of chunk IDs across all collections."""
        if not chunk_ids:
            return {}

        id_map = {}
        for cid in chunk_ids:
            try:
                uid = str(uuid.UUID(cid[:32]))
                id_map[uid] = cid
            except ValueError:
                uid = str(uuid.UUID(hashlib.sha256(cid.encode()).hexdigest()[:32]))
                id_map[uid] = cid

        result = {}
        for coll in self.all_collections():
            if not (coll.startswith(PDF_PREFIX) or
                    coll.startswith(HTML_PREFIX) or
                    coll.startswith(CLAUDE_PREFIX)):
                continue
            try:
                pts = self.client.retrieve(
                    collection_name=coll,
                    ids=list(id_map.keys()),
                    with_payload=True,
                )
                for pt in pts:
                    key = id_map.get(str(pt.id), str(pt.id).replace("-", ""))
                    result[key] = pt.payload or {}
            except Exception:
                pass
        return result

    def scroll_text_chunks(self) -> list[tuple[str, str]]:
        """
        Scroll ALL text chunks from every collection.
        Returns [(chunk_id, text)] for BM25 rebuild.
        Paginates in batches of 200 — memory efficient for large corpora.
        """
        pairs: list[tuple[str, str]] = []
        for coll in self.all_collections():
            if not (coll.startswith(PDF_PREFIX) or
                    coll.startswith(HTML_PREFIX) or
                    coll.startswith(CLAUDE_PREFIX)):
                continue
            offset = None
            while True:
                try:
                    pts, next_offset = self.client.scroll(
                        collection_name=coll,
                        limit=200,
                        offset=offset,
                        with_payload=["text", "content_type"],
                        with_vectors=False,
                    )
                except Exception as e:
                    log.warning("scroll failed", coll=coll, error=str(e))
                    break
                for pt in pts:
                    p = pt.payload or {}
                    if p.get("content_type") == "text":
                        text = p.get("text", "").strip()
                        if text:
                            pairs.append((str(pt.id).replace("-", ""), text))
                if next_offset is None:
                    break
                offset = next_offset
        return pairs

    def is_healthy(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def total_points(self) -> int:
        return sum(self.point_count(c) for c in self.all_collections())
