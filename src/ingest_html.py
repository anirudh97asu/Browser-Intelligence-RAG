"""
ingest_html.py — Index an HTML page (by URL or local file) into Qdrant.

Pipeline:
  Fetch HTML → Trafilatura XML mode → text (no tables, no avatars)
                                    → content images (src URLs)
  Text → semantic chunk → text embed → store (content_type="text")
  Images → CLIP image embed → store  (content_type="image")
           payload includes img_src for content.js highlight

Collection: rag_html__<domain_slug>
Run:
  python ingest_html.py --url https://example.com/article
  python ingest_html.py --file /path/to/page.html --url https://original-url.com
"""

import argparse
import sys
from pathlib import Path
from urllib.parse import urljoin

sys.path.insert(0, str(Path(__file__).parent))

from logger import get_logger

log = get_logger(__name__)

from config import TEXT_DIM, IMAGE_DIM
from embed import Embedder
from store import QdrantStore, collection_for_html, stable_id
from bm25 import BM25Index
from utils import semantic_chunk, is_avatar_or_icon


def fetch_html(url: str, timeout: int = 30) -> bytes:
    import requests
    headers = {"User-Agent": "Mozilla/5.0 (compatible; BrowserRAG/1.0)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.content


def parse_html(html_bytes: bytes, url: str) -> dict:
    """
    Parse HTML with Trafilatura XML mode.
    Returns {"text": str, "images": [{"src": str, "alt": str}]}
    """
    import trafilatura
    from lxml import etree

    html_str = html_bytes.decode("utf-8", errors="replace")

    xml_out = trafilatura.extract(
        html_str,
        url=url,
        include_images=True,
        include_tables=True,
        include_links=False,
        favor_precision=True,
        deduplicate=True,
        output_format="xml",
    )

    text = ""
    images = []

    if xml_out:
        try:
            root = etree.fromstring(xml_out.encode("utf-8"))

            # Text — exclude tables
            text_parts = root.xpath(".//main//text()[not(ancestor::table)]")
            text = " ".join(p.strip() for p in text_parts if p.strip())

            # Images — skip avatars/icons
            for g in root.xpath(".//main//graphic"):
                src = g.get("src", "").strip()
                alt = g.get("alt", "").strip()
                if not src:
                    continue
                if src.lower().startswith(("data:", "javascript:")):
                    continue
                if src.lower().endswith(".svg"):
                    continue
                if is_avatar_or_icon(src, alt):
                    continue
                # Resolve relative URLs
                try:
                    src = urljoin(url, src)
                except Exception:
                    pass
                images.append({"src": src, "alt": alt})
        except Exception as e:
            log.warning("XML parse error", error=str(e))

    # Fallback plain text
    if not text.strip():
        plain = trafilatura.extract(html_str, url=url, favor_precision=True)
        text = (plain or "").strip()

    return {"text": text, "images": images}


def index_html(url: str, html_bytes: bytes, embedder: Embedder, store: QdrantStore, bm25: BM25Index):
    """Index one HTML page. Returns number of points stored."""
    log.info("Indexing HTML", url=url)
    collection = collection_for_html(url)
    store.ensure_collection(collection, text_dim=embedder.text_dim, image_dim=embedder.image_dim)

    existing = store.point_count(collection)
    if existing > 0:
        log.info("Already indexed — skipping", points=existing)
        return 0

    parsed = parse_html(html_bytes, url)
    page_text = parsed["text"]
    page_images = parsed["images"]

    if not page_text and not page_images:
        log.warning("No content extracted", url=url)
        return 0

    points = []
    _bm25_pairs = []   # (chunk_id, text) collected for BM25 rebuild
    chunk_idx = 0

    # ── Text chunks ───────────────────────────────────────────────────────────
    if page_text:
        text_chunks = semantic_chunk(page_text, embedder)
        text_vecs   = embedder.embed_texts(text_chunks)

        for chunk_text, text_vec in zip(text_chunks, text_vecs):
            if text_vec is None:
                chunk_idx += 1
                continue

            point_id = stable_id(url, chunk_idx)
            pt = store.build_point(
                point_id=point_id,
                text_vec=text_vec,
                image_vec=None,
                payload={
                    "modality":         "html",
                    "content_type":     "text",
                    "source_type":      "html",
                    "source_id":        collection,
                    "source_path":      "",
                    "source_url":       url,
                    "url":              url,
                    "filename":         url.split("/")[-1][:80] or url.split("/")[-2][:80],
                    "text":             chunk_text,
                    "text_fingerprint": chunk_text[:80],
                    "img_src":          "",
                    "img_alt":          "",
                    "collection":       collection,
                    "section_title":    "",
                    "image_path":       "",
                    "page_number":      0,
                    "bbox":             [],
                },
                text_dim=embedder.text_dim,
                image_dim=embedder.image_dim,
            )
            if pt:
                points.append(pt)
                _bm25_pairs.append((point_id.replace("-", ""), chunk_text))
            chunk_idx += 1

        log.info("Text chunks", count=len(text_chunks))

    # ── Image chunks ──────────────────────────────────────────────────────────
    img_stored = 0
    for img in page_images:
        src = img["src"]
        alt = img["alt"]

        img_vec = embedder.embed_image_url(src)
        if img_vec is None:
            chunk_idx += 1
            continue

        # Also embed alt text so text queries can find this image
        alt_text = alt or f"Image from {url}"
        text_vec = embedder.embed_text(alt_text)

        point_id = stable_id(url, chunk_idx)
        pt = store.build_point(
            point_id=point_id,
            text_vec=text_vec,
            image_vec=img_vec,
            payload={
                "modality":         "html",
                "content_type":     "image",
                "source_type":      "html",
                "source_id":        collection,
                "source_path":      "",
                "source_url":       url,
                "url":              url,
                "filename":         url.split("/")[-1][:80] or url.split("/")[-2][:80],
                "text":             alt_text,
                "text_fingerprint": "",
                "img_src":          src,
                "img_alt":          alt,
                "collection":       collection,
                "section_title":    "",
                "image_path":       src,
                "page_number":      0,
                "bbox":             [],
            },
            text_dim=embedder.text_dim,
            image_dim=embedder.image_dim,
        )
        if pt:
            points.append(pt)
            img_stored += 1
        chunk_idx += 1

    if img_stored:
        log.info("Image chunks", count=img_stored)

    stored = store.upsert(collection, points)
    log.info("HTML indexed", points=stored, collection=collection)
    return stored


def main():
    parser = argparse.ArgumentParser(description="Index an HTML page into Qdrant")
    parser.add_argument("--url",   required=True, help="URL of the page (used as key)")
    parser.add_argument("--file",  help="Local HTML file to read (instead of fetching URL)")
    parser.add_argument("--force", action="store_true", help="Re-index even if already indexed")
    args = parser.parse_args()

    embedder = Embedder()
    store    = QdrantStore()
    bm25     = BM25Index()
    bm25.load()

    if args.force:
        coll = collection_for_html(args.url)
        if store.collection_exists(coll):
            store.client.delete_collection(coll)
            log.info("Deleted collection", coll=coll)

    if args.file:
        html_bytes = Path(args.file).read_bytes()
    else:
        log.info("Fetching URL", url=args.url)
        html_bytes = fetch_html(args.url)

    stored = index_html(args.url, html_bytes, embedder, store, bm25)

    pairs = store.scroll_text_chunks()
    if pairs:
        bm25.build_from(pairs)
        bm25.save()
    log.info("Indexing complete", stored=stored)


if __name__ == "__main__":
    main()
