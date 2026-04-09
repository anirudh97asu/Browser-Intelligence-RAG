"""
embed.py — unified embedding for text and images using CLIP.

CLIP (clip-ViT-B-32) gives a SHARED 512d space for both text and images.
A text query can find image chunks and vice versa.

Text-only fallback: if CLIP is not available, uses nomic-embed-text (768d)
for text and skips image embedding (images stored but not searchable by text).

Usage:
    from embed import Embedder
    emb = Embedder()

    vec = emb.embed_text("what is RAG?")          # → list[float] 512d or 768d
    vec = emb.embed_image_b64("base64png...")      # → list[float] 512d
    vec = emb.embed_image_pil(pil_image)           # → list[float] 512d
"""

import base64
import io
import requests
import json
from typing import Optional

from logger import get_logger

log = get_logger(__name__)

from config import (
    CLIP_MODEL, EMBED_TEXT_MODEL, OLLAMA_URL,
    IMAGE_DIM, TEXT_DIM, EMBED_BATCH_SIZE,
)


def _free_gpu():
    """Release unused GPU memory after an inference call."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class Embedder:
    def __init__(self):
        self._clip = None
        self._clip_dim = IMAGE_DIM
        self._mode = "clip"  # or "ollama"
        self._load_clip()

    def _load_clip(self):
        """Try to load CLIP. Fall back to Ollama text-only if unavailable."""
        try:
            from sentence_transformers import SentenceTransformer
            log.info(f"Loading CLIP model: {CLIP_MODEL}")
            self._clip = SentenceTransformer(CLIP_MODEL)

            # get_sentence_embedding_dimension() returns None for CLIP in newer
            # sentence-transformers versions — probe with an actual encode instead
            probe = self._clip.encode("test", convert_to_numpy=True)
            self._clip_dim = int(probe.shape[0])

            self._mode = "clip"
            log.info(f"CLIP ready", dim=self._clip_dim, model=CLIP_MODEL)
        except Exception as e:
            log.warning(f"CLIP unavailable — falling back to nomic-embed-text (text only)",
                        error=str(e))
            self._mode = "ollama"
            self._clip_dim = TEXT_DIM

    @property
    def text_dim(self) -> int:
        return self._clip_dim

    @property
    def image_dim(self) -> int:
        return self._clip_dim  # same space

    @property
    def using_clip(self) -> bool:
        return self._mode == "clip"

    # ── Text embedding ────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> Optional[list]:
        """Embed a single text string."""
        if not text.strip():
            return None
        if self._mode == "clip":
            return self._clip_embed_text(text)
        else:
            return self._ollama_embed_text(text)

    def embed_texts(self, texts: list[str]) -> list[Optional[list]]:
        """Embed a batch of text strings."""
        if not texts:
            return []
        if self._mode == "clip":
            return self._clip_embed_texts(texts)
        else:
            return self._ollama_embed_texts(texts)

    def _clip_embed_text(self, text: str) -> Optional[list]:
        try:
            vec = self._clip.encode(text, convert_to_numpy=True)
            result = vec.tolist()
            return result
        except Exception as e:
            log.error("CLIP text embed", error=str(e))
            return None
        finally:
            _free_gpu()

    def _clip_embed_texts(self, texts: list[str]) -> list[Optional[list]]:
        results = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i:i + EMBED_BATCH_SIZE]
            try:
                vecs = self._clip.encode(batch, convert_to_numpy=True, batch_size=EMBED_BATCH_SIZE)
                results.extend([v.tolist() for v in vecs])
            except Exception as e:
                log.error("CLIP batch embed", error=str(e))
                results.extend([None] * len(batch))
            finally:
                _free_gpu()
        return results

    def _ollama_embed_text(self, text: str) -> Optional[list]:
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": EMBED_TEXT_MODEL, "input": text},
                timeout=60,
            )
            data = r.json()
            embeds = data.get("embeddings", [])
            return embeds[0] if embeds else None
        except Exception as e:
            log.error("Ollama embed", error=str(e))
            return None

    def _ollama_embed_texts(self, texts: list[str]) -> list[Optional[list]]:
        results = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i:i + EMBED_BATCH_SIZE]
            try:
                r = requests.post(
                    f"{OLLAMA_URL}/api/embed",
                    json={"model": EMBED_TEXT_MODEL, "input": batch},
                    timeout=120,
                )
                data = r.json()
                embeds = data.get("embeddings", [])
                # Pad with None if fewer returned
                while len(embeds) < len(batch):
                    embeds.append(None)
                results.extend(embeds)
            except Exception as e:
                log.error("Ollama batch embed", error=str(e))
                results.extend([None] * len(batch))
        return results

    # ── Image embedding ───────────────────────────────────────────────────────

    def embed_image_b64(self, b64: str, label: str = "") -> Optional[list]:
        """Embed a base64-encoded PNG/JPEG image."""
        if not self.using_clip:
            return None
        try:
            from PIL import Image
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            return self.embed_image_pil(img)
        except Exception as e:
            log.error("Image b64 decode", label=label, error=str(e))
            return None

    def embed_image_pil(self, pil_image) -> Optional[list]:
        """Embed a PIL Image object."""
        if not self.using_clip:
            return None
        try:
            vec = self._clip.encode(pil_image, convert_to_numpy=True)
            return vec.tolist()
        except Exception as e:
            log.error("CLIP image embed", error=str(e))
            return None
        finally:
            _free_gpu()

    def embed_image_url(self, url: str, timeout: int = 15) -> Optional[list]:
        """Download an image from URL and embed it."""
        if not self.using_clip:
            return None
        try:
            from PIL import Image
            resp = requests.get(url, timeout=timeout, stream=True)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            return self.embed_image_pil(img)
        except Exception as e:
            log.error("Image URL embed", url=url[:60], error=str(e))
            return None
