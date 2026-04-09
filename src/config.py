"""
config.py — loads all settings from .env (or environment variables).

Priority: environment variable > .env file > hardcoded default.
So you can override any setting at runtime:
  PDF_DPI=300 python3 api.py
"""

import os
from pathlib import Path

# Load .env from project root (two levels up from src/)
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().split("#")[0].strip()  # strip inline comments
        if key and key not in os.environ:          # env var wins over .env
            os.environ[key] = val


def _str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()

def _int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default

def _float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


# ── Services ──────────────────────────────────────────────────────────────────
OLLAMA_URL  = _str("OLLAMA_URL",  "http://localhost:11434")
QDRANT_HOST = _str("QDRANT_HOST", "localhost")
QDRANT_PORT = _int("QDRANT_PORT", 6333)
REDIS_URL   = _str("REDIS_URL",   "redis://localhost:6379/0")

# ── Models ────────────────────────────────────────────────────────────────────
EMBED_TEXT_MODEL = _str("EMBED_TEXT_MODEL", "nomic-embed-text")
LLM_MODEL        = _str("LLM_MODEL",        "gemma3:1b")
# VLM_MODEL: kept for potential future use but not required.
# Scanned PDFs use Docling OCR locally instead.
VLM_MODEL        = _str("VLM_MODEL",        "llava-phi3")
CLIP_MODEL       = _str("CLIP_MODEL",       "clip-ViT-B-32")

# ── Embedding dimensions ───────────────────────────────────────────────────────
TEXT_DIM  = 768   # nomic-embed-text output dim
IMAGE_DIM = 512   # CLIP ViT-B/32 output dim

# ── Qdrant collection prefixes ────────────────────────────────────────────────
PDF_PREFIX  = "rag_pdf__"
HTML_PREFIX = "rag_html__"

# ── Chunking ──────────────────────────────────────────────────────────────────
SEMANTIC_THRESHOLD = _float("SEMANTIC_THRESHOLD", 0.3)
MIN_CHUNK_WORDS    = _int("MIN_CHUNK_WORDS",       15)

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_VECTOR  = _int("TOP_K_VECTOR", 20)
TOP_K_BM25    = _int("TOP_K_BM25",   20)
TOP_K_FINAL   = _int("TOP_K_FINAL",   5)
RRF_K         = _int("RRF_K",        60)

# ── PDF rendering ──────────────────────────────────────────────────────────────
PDF_DPI        = _int("PDF_DPI",        150)
MIN_TEXT_CHARS = _int("MIN_TEXT_CHARS",  20)
MIN_IMAGE_PX   = _int("MIN_IMAGE_PX",   80)

# ── HTML image filtering ──────────────────────────────────────────────────────
SKIP_IMAGE_KEYWORDS = [
    "avatar", "user", "profile", "author", "favicon",
    "icon", "logo", "badge", "emoji", "subscriber",
]

# ── Embed batching ────────────────────────────────────────────────────────────
EMBED_BATCH_SIZE = 32

# ── API server ────────────────────────────────────────────────────────────────
API_HOST = _str("API_HOST", "0.0.0.0")
API_PORT = _int("API_PORT", 8000)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = _str("LOG_LEVEL", "INFO")
LOG_FILE  = _str("LOG_FILE",  "logs/rag2.log")
