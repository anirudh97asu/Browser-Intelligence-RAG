"""
bm25.py — Memory-efficient BM25 keyword index.

Uses bm25s built-in save/load (numpy arrays on disk, not pickle).
Passes chunk_ids directly as corpus to retrieve() — no post-lookup needed.
Does NOT keep corpus_tokens in RAM after build().

Layout on disk:
  data/bm25/          ← bm25s native index files
  data/bm25_ids.json  ← ordered list of chunk_ids matching the index
"""

from logger import get_logger

log = get_logger(__name__)

import json
import re
from pathlib import Path

DATA_DIR  = Path(__file__).parent.parent / "data"
BM25_DIR  = DATA_DIR / "bm25"
IDS_PATH  = DATA_DIR / "bm25_ids.json"


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


class BM25Index:
    def __init__(self):
        self._chunk_ids: list[str] = []   # parallel to index rows
        self._retriever = None
        self._ready = False
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        BM25_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def doc_count(self) -> int:
        return len(self._chunk_ids)

    # ── Build ─────────────────────────────────────────────────────────────────

    def build_from(self, id_text_pairs: list[tuple[str, str]]) -> None:
        """
        Build index from scratch from (chunk_id, text) pairs.
        Preferred entry point — does not hold corpus_tokens in RAM.
        """
        if not id_text_pairs:
            log.warning("BM25 build_from called with empty corpus")
            return

        import bm25s
        chunk_ids = [cid for cid, _ in id_text_pairs]
        texts     = [text for _, text in id_text_pairs]

        log.info("BM25 tokenising", docs=len(texts))
        # bm25s.tokenize handles batching internally
        tokens = bm25s.tokenize(texts, show_progress=False)

        retriever = bm25s.BM25()
        retriever.index(tokens)

        self._chunk_ids = chunk_ids
        self._retriever = retriever
        self._ready     = True
        log.info("BM25 built", docs=self.doc_count)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        """Return [(chunk_id, score)] sorted by score descending."""
        if not self._ready or not self._retriever or not self._chunk_ids:
            return []

        import bm25s
        import numpy as np

        try:
            q_tokens = bm25s.tokenize([query], show_progress=False)
            k_capped = min(k, len(self._chunk_ids))

            # Pass chunk_ids as corpus → retriever returns IDs directly
            results, scores = self._retriever.retrieve(
                q_tokens,
                corpus=self._chunk_ids,
                k=k_capped,
                show_progress=False,
            )

            hits = []
            for cid, score in zip(results[0], scores[0]):
                s = float(score)
                if s > 0:
                    hits.append((str(cid), s))
            return hits

        except Exception as e:
            log.error("BM25 search error", error=str(e))
            return []

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Save index to disk using bm25s native format + JSON id list."""
        if not self._ready or not self._retriever:
            log.warning("BM25 save called but index not ready")
            return
        try:
            self._retriever.save(str(BM25_DIR))
            IDS_PATH.write_text(json.dumps(self._chunk_ids))
            log.info("BM25 saved", docs=self.doc_count, path=str(BM25_DIR))
        except Exception as e:
            log.error("BM25 save error", error=str(e))

    def load(self) -> bool:
        """Load index from disk. Returns True on success."""
        try:
            if not IDS_PATH.exists() or not BM25_DIR.exists():
                return False
            import bm25s
            self._retriever = bm25s.BM25.load(str(BM25_DIR), mmap=True)
            self._chunk_ids = json.loads(IDS_PATH.read_text())
            self._ready     = True
            log.info("BM25 loaded", docs=self.doc_count)
            return True
        except Exception as e:
            log.error("BM25 load error", error=str(e))
            return False

    def clear(self) -> None:
        """Wipe in-memory state. Call before rebuilding."""
        self._chunk_ids = []
        self._retriever = None
        self._ready     = False
