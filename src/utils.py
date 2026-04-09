"""
utils.py — shared helpers used by ingest_pdf and ingest_html.

  - split_sentences(text)       → list of sentences (spaCy)
  - semantic_chunk(text, emb)   → list of text chunks
  - rrf_merge(ranked_lists)     → merged ranked list
  - is_avatar_or_icon(src, alt) → True if image should be skipped
"""

import math
import re
from typing import Optional

from logger import get_logger

log = get_logger(__name__)

from config import (
    SEMANTIC_THRESHOLD, MIN_CHUNK_WORDS,
    SKIP_IMAGE_KEYWORDS, RRF_K,
)


# ── Sentence splitting ────────────────────────────────────────────────────────

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm",
                              disable=["ner", "tagger", "lemmatizer", "attribute_ruler"])
            _nlp.max_length = 2_000_000
        except Exception as e:
            log.warning("spaCy unavailable — using simple sentence splitter", error=str(e))
            _nlp = "simple"
    return _nlp


def split_sentences(text: str) -> list[str]:
    nlp = _get_nlp()
    if nlp == "simple":
        # Fallback: split on . ! ?
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


# ── Semantic chunking ─────────────────────────────────────────────────────────

def cosine_dist(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)


def semantic_chunk(
    text: str,
    embedder,
    threshold: float = SEMANTIC_THRESHOLD,
    min_words: int | None = None,
) -> list[str]:
    """
    Split text into semantic chunks using cosine distance between sentence embeddings.
    min_words: override MIN_CHUNK_WORDS for this call (e.g. lower for OCR/scanned pages)
    """
    effective_min = min_words if min_words is not None else MIN_CHUNK_WORDS

    sentences = split_sentences(text)
    if len(sentences) <= 2:
        cleaned = text.strip()
        if not cleaned:
            return []
        return [cleaned] if len(cleaned.split()) >= effective_min else []

    vecs = embedder.embed_texts(sentences)

    # Fall back to word-count split if embedding failed
    if not vecs or all(v is None for v in vecs):
        return _word_split(text, max_words=250, min_words=effective_min)

    chunks, current, current_words = [], [sentences[0]], len(sentences[0].split())

    for i in range(1, len(sentences)):
        prev_vec = vecs[i - 1]
        curr_vec = vecs[i]

        if prev_vec is None or curr_vec is None:
            current.append(sentences[i])
            current_words += len(sentences[i].split())
            continue

        dist = cosine_dist(prev_vec, curr_vec)
        if dist > threshold and current_words >= effective_min:
            chunks.append(" ".join(current))
            current, current_words = [], 0

        current.append(sentences[i])
        current_words += len(sentences[i].split())

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if len(c.split()) >= effective_min]


def _word_split(text: str, max_words: int = 250, min_words: int = MIN_CHUNK_WORDS) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        if len(chunk.split()) >= min_words:
            chunks.append(chunk)
    return chunks


# ── RRF merge ─────────────────────────────────────────────────────────────────

def rrf_merge(ranked_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    """
    Reciprocal Rank Fusion.
    ranked_lists: each is an ordered list of chunk_ids.
    Returns merged list of chunk_ids ordered by RRF score.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


# ── Image filtering ───────────────────────────────────────────────────────────

def is_avatar_or_icon(src: str, alt: Optional[str] = None) -> bool:
    """Return True if this image is likely an avatar, icon, or decoration."""
    combined = ((alt or "") + " " + (src or "")).lower()
    return any(kw in combined for kw in SKIP_IMAGE_KEYWORDS)
