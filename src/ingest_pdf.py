"""
ingest_pdf.py — Index a PDF into Qdrant.

Pipeline per document:
  1. PyMuPDF opens PDF, detects scanned vs typeset pages
  2. If any scanned pages: Docling OCR runs ONCE for the whole PDF
  3. Per page:

  Typeset page:
    a. PyMuPDF get_text("dict") → text blocks with bboxes
    b. Semantic chunk the text
    c. CLIP ViT-B/32 text embed  → stored as content_type="text"   (with bbox)
    d. PyMuPDF renders page PNG  → CLIP ViT-B/32 image embed
                                 → stored as content_type="image_page" (full-page bbox)
    e. Embedded figures (>80px)  → CLIP ViT-B/32 image embed
                                 → stored as content_type="image_embedded"

  Scanned page:
    a. Docling OCR text (from pre-pass)
    b. Semantic chunk the OCR text
    c. CLIP ViT-B/32 text embed  → stored as content_type="text"   (no bbox)
    d. PyMuPDF renders page PNG  → CLIP ViT-B/32 image embed
                                 → stored as content_type="image_page" (full-page bbox)
    (no embedded figure extraction — scanned pages have no vector images)

All image embeds use CLIP ViT-B/32 (512d) — same space as text embeds.
No VLM is called anywhere in this pipeline.
"""

import argparse
import base64
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logger import get_logger
log = get_logger(__name__)

from config import PDF_DPI, MIN_TEXT_CHARS, MIN_IMAGE_PX
from embed import Embedder
from store import QdrantStore, collection_for_pdf, stable_id
from bm25 import BM25Index
from utils import semantic_chunk


# ── MuPDF noise suppression ───────────────────────────────────────────────────
# Suppress benign format warnings (e.g. "No common ancestor in structure tree")
# that some malformed PDFs emit. These do not affect extraction quality.
# Must be set before any fitz operation.
import fitz as _fitz_init
_fitz_init.TOOLS.mupdf_display_errors(False)
_fitz_init.TOOLS.mupdf_display_warnings(False)
del _fitz_init


# ── Docling OCR ───────────────────────────────────────────────────────────────

def _docling_extract(pdf_path: Path) -> dict[int, dict]:
    """
    Run Docling OCR on a PDF. Returns per-page data (1-based page numbers):

      {
        page_no: {
          "text":     str,              # full page OCR text (joined)
          "elements": [                 # individual text elements with bboxes
            {
              "text": str,
              "bbox": [x0, y0, x1, y1] # PDF points, TOPLEFT origin (PyMuPDF convention)
            },
            ...
          ]
        }
      }

    Docling bbox coords are normalised (0–1) relative to page size.
    We convert to absolute PDF points using PyMuPDF page dimensions so
    the viewer can draw highlight boxes directly.

    Returns empty dict if Docling is unavailable or fails.
    """
    try:
        import fitz
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import PdfFormatOption

        log.info("Docling OCR starting", file=pdf_path.name)

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr             = True
        pipeline_options.do_table_structure = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result  = converter.convert(str(pdf_path))
        doc_obj = result.document

        # Get page dimensions from PyMuPDF for coordinate conversion
        import fitz as _fitz
        fitz_doc = _fitz.open(str(pdf_path))
        page_dims: dict[int, tuple[float, float]] = {}  # page_no → (width_pt, height_pt)
        for i in range(fitz_doc.page_count):
            r = fitz_doc[i].rect
            page_dims[i + 1] = (r.width, r.height)
        fitz_doc.close()

        # Collect elements per page
        page_data: dict[int, dict] = {}

        for element, _level in doc_obj.iterate_items():
            prov = getattr(element, "prov", None) or []
            if not prov:
                continue
            p        = prov[0]
            page_no  = p.page_no
            text     = (getattr(element, "text", "") or "").strip()
            if not text:
                continue

            if page_no not in page_data:
                page_data[page_no] = {"text": "", "elements": []}

            # Convert Docling bbox to PDF points (PyMuPDF TOPLEFT convention)
            bbox_pts = None
            raw_bbox = getattr(p, "bbox", None)
            if raw_bbox is not None and page_no in page_dims:
                pw, ph = page_dims[page_no]
                try:
                    # Docling BoundingBox: l, t, r, b
                    # coord_origin is typically BOTTOMLEFT (PDF standard)
                    # → convert to TOPLEFT by flipping y: y_top = page_h - y_bottom
                    from docling.datamodel.document import CoordOrigin
                    l = float(raw_bbox.l)
                    t = float(raw_bbox.t)
                    r = float(raw_bbox.r)
                    b = float(raw_bbox.b)

                    if hasattr(raw_bbox, "coord_origin") and                        raw_bbox.coord_origin == CoordOrigin.BOTTOMLEFT:
                        # Normalised BOTTOMLEFT → absolute TOPLEFT
                        # Docling normalises to [0,1] range
                        x0 = l * pw
                        y0 = (1.0 - t) * ph   # flip: top becomes bottom-left origin
                        x1 = r * pw
                        y1 = (1.0 - b) * ph
                    else:
                        # Already TOPLEFT normalised
                        x0, y0, x1, y1 = l * pw, t * ph, r * pw, b * ph

                    # Ensure x0 < x1, y0 < y1
                    bbox_pts = [
                        min(x0, x1), min(y0, y1),
                        max(x0, x1), max(y0, y1),
                    ]
                except Exception:
                    bbox_pts = None

            page_data[page_no]["elements"].append({
                "text": text,
                "bbox": bbox_pts,
            })

        # Build full-page text string per page
        for pg, data in page_data.items():
            data["text"] = " ".join(e["text"] for e in data["elements"])

        log.info("Docling OCR complete", file=pdf_path.name, pages=len(page_data))
        return page_data

    except ImportError:
        log.error("Docling not installed — run: uv sync")
        return {}
    except Exception as e:
        log.error("Docling OCR failed", file=pdf_path.name, error=str(e))
        return {}


# ── Block bbox matching ───────────────────────────────────────────────────────

def _find_block_bbox(page_blocks: list[dict], chunk_text: str) -> list | None:
    """
    Match a text chunk against PyMuPDF text blocks and return a merged bbox.
    Uses word-overlap scoring — robust to ligatures and spacing artefacts.
    Returns [x0, y0, x1, y1] in PDF points (top-left origin), or None.
    """
    if not page_blocks or not chunk_text.strip():
        return None

    def _words(t: str) -> set[str]:
        return set(re.sub(r"[^a-z0-9]", " ", t.lower()).split())

    chunk_words = _words(chunk_text)
    if not chunk_words:
        return None

    scored = []
    for b in page_blocks:
        bw = _words(b["text"])
        if not bw:
            continue
        overlap = len(chunk_words & bw) / len(chunk_words)
        if overlap > 0.25:
            scored.append((overlap, b["bbox"]))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [bbox for _, bbox in scored[:3]]
    return [
        min(b[0] for b in top),
        min(b[1] for b in top),
        max(b[2] for b in top),
        max(b[3] for b in top),
    ]


# ── Main index function ───────────────────────────────────────────────────────

def _find_scanned_bbox(page_elements: list[dict], chunk_text: str) -> list | None:
    """
    Match a chunk from Docling OCR text against individual OCR elements.
    Uses the same word-overlap strategy as _find_block_bbox.
    Returns merged bbox [x0, y0, x1, y1] in PDF points (TOPLEFT), or None.
    """
    if not page_elements or not chunk_text.strip():
        return None

    chunk_words = set(re.sub(r"[^a-z0-9]", " ", chunk_text.lower()).split())
    if not chunk_words:
        return None

    scored = []
    for el in page_elements:
        if not el.get("bbox"):
            continue
        el_words = set(re.sub(r"[^a-z0-9]", " ", el["text"].lower()).split())
        if not el_words:
            continue
        overlap = len(chunk_words & el_words) / len(chunk_words)
        if overlap > 0.2:   # slightly lower threshold for OCR (noisier text)
            scored.append((overlap, el["bbox"]))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [bbox for _, bbox in scored[:3]]
    return [
        min(b[0] for b in top),
        min(b[1] for b in top),
        max(b[2] for b in top),
        max(b[3] for b in top),
    ]


def index_pdf(
    pdf_path:  Path,
    embedder:  Embedder,
    store:     QdrantStore,
    bm25:      BM25Index,
    max_pages: int | None = 50,
) -> int:
    """
    Index one PDF into Qdrant. Returns number of points stored.
    Skips if already indexed (check by point count in collection).
    max_pages: process at most this many pages (default 50, None = all pages).
    """
    import fitz  # PyMuPDF

    log.info("Indexing PDF", file=pdf_path.name)
    collection = collection_for_pdf(str(pdf_path))
    store.ensure_collection(
        collection,
        text_dim=embedder.text_dim,
        image_dim=embedder.image_dim,
    )

    if store.point_count(collection) > 0:
        log.info("Already indexed — skipping (use --force to re-index)",
                 file=pdf_path.name)
        return 0

    scale = PDF_DPI / 72.0
    mat   = fitz.Matrix(scale, scale)
    doc   = fitz.open(str(pdf_path))

    # ── Apply page limit ─────────────────────────────────────────────────────
    total_pages = doc.page_count
    if max_pages and max_pages > 0 and total_pages > max_pages:
        # SKIP the entire PDF — do not partially index it.
        # A partial index would give incomplete retrieval results.
        doc.close()
        log.info("PDF skipped — exceeds page limit",
                 file=pdf_path.name, pages=total_pages, limit=max_pages)
        return 0
    page_limit = total_pages

    # ── Detect scanned pages (cheap pre-pass) ─────────────────────────────────
    scanned_pages: set[int] = set()   # 0-based page indices
    for pg_idx in range(page_limit):
        raw = (doc[pg_idx].get_text("text") or "").strip()
        if len(raw) < MIN_TEXT_CHARS:
            scanned_pages.add(pg_idx)

    # ── Docling OCR for scanned pages (one call, whole PDF) ───────────────────
    docling_texts: dict[int, str] = {}   # 1-based page_no → OCR text
    if scanned_pages:
        log.info("Scanned pages found — running Docling OCR",
                 file=pdf_path.name, count=len(scanned_pages))
        docling_texts = _docling_extract(pdf_path)

    # ── Per-page indexing ─────────────────────────────────────────────────────
    points    = []
    chunk_idx = 0

    for page_num in range(page_limit):
        page       = doc[page_num]
        page_label = f"p{page_num + 1}"
        is_scanned = page_num in scanned_pages

        # ── Render page PNG for CLIP image embedding ──────────────────────────
        # Always rendered — used for image_page chunk on every page.
        pix      = page.get_pixmap(matrix=mat, alpha=False)
        page_png = pix.tobytes("png")
        page_b64 = base64.b64encode(page_png).decode("ascii")

        # ── Get text for this page ────────────────────────────────────────────
        if is_scanned:
            page_doc_data = docling_texts.get(page_num + 1, {})
            page_text     = page_doc_data.get("text", "").strip()
            page_ocr_elements = page_doc_data.get("elements", [])
            if not page_text:
                page_text = f"{pdf_path.stem} page {page_num + 1}"
                page_ocr_elements = []
                log.warning("No OCR text for page", page=page_label, file=pdf_path.name)
            else:
                word_count = len(page_text.split())
                log.info("OCR text ready", page=page_label,
                         chars=len(page_text), words=word_count)
        else:
            page_text         = (page.get_text("text") or "").strip()
            page_ocr_elements = []   # not used for typeset pages

        # ── Extract PyMuPDF text blocks with bboxes (typeset only) ───────────
        # Used later to find the precise highlight bbox for each text chunk.
        page_blocks: list[dict] = []
        if not is_scanned:
            try:
                raw_blocks = page.get_text(
                    "dict", flags=fitz.TEXT_PRESERVE_WHITESPACE
                )["blocks"]
                for b in raw_blocks:
                    if b.get("type") == 0:   # 0 = text block
                        block_text = " ".join(
                            span["text"]
                            for line in b.get("lines", [])
                            for span in line.get("spans", [])
                        ).strip()
                        if block_text:
                            page_blocks.append({
                                "text": block_text,
                                "bbox": list(b["bbox"]),
                            })
            except Exception as e:
                log.warning("get_text dict failed", page=page_label, error=str(e))

        # ── Text chunks → CLIP text embed ─────────────────────────────────────
        # Use lower min_words for scanned pages — OCR text is sparser
        min_words = 5 if is_scanned else None   # None = use config default
        text_chunks = semantic_chunk(page_text, embedder, min_words=min_words)
        text_vecs   = embedder.embed_texts(text_chunks)

        for chunk_text, text_vec in zip(text_chunks, text_vecs):
            if text_vec is None:
                chunk_idx += 1
                continue

            # Precise bbox:
            #   Typeset → PyMuPDF block matching (word overlap on get_text dict)
            #   Scanned → Docling OCR element matching (word overlap on OCR elements)
            if not is_scanned:
                bbox = _find_block_bbox(page_blocks, chunk_text)
            else:
                bbox = _find_scanned_bbox(page_ocr_elements, chunk_text)

            pt = store.build_point(
                point_id   = stable_id(str(pdf_path), chunk_idx),
                text_vec   = text_vec,
                image_vec  = None,   # text chunk: no image vector
                payload    = {
                    "content_type":     "text",
                    "source_type":      "pdf",
                    "file_path":        str(pdf_path),
                    "page_number":      page_num + 1,
                    "text":             chunk_text,
                    "text_fingerprint": chunk_text[:80],
                    "bbox":             bbox or [],
                    "is_scanned":       is_scanned,
                    "collection":       collection,
                },
                text_dim   = embedder.text_dim,
                image_dim  = embedder.image_dim,
            )
            if pt:
                points.append(pt)
            chunk_idx += 1

        # ── Full-page image chunk → CLIP image embed ──────────────────────────
        # PyMuPDF renders page at PDF_DPI → PNG → CLIP ViT-B/32 image encoder.
        # This gives a visual representation of the page in the shared 512d space.
        # Stored for BOTH typeset and scanned pages.
        page_img_vec = embedder.embed_image_b64(page_b64, label=f"{page_label} page")
        if page_img_vec is not None:
            mb        = page.mediabox
            full_bbox = [mb.x0, mb.y0, mb.x1, mb.y1]
            pt = store.build_point(
                point_id  = stable_id(str(pdf_path), chunk_idx),
                text_vec  = None,
                image_vec = page_img_vec,   # CLIP image embed
                payload   = {
                    "modality":         "pdf",
                    "content_type":     "image_page",
                    "source_type":      "pdf",
                    "source_id":        collection,
                    "source_path":      str(pdf_path),
                    "source_url":       "",
                    "file_path":        str(pdf_path),
                    "filename":         pdf_path.name,
                    "page_number":      page_num + 1,
                    "text":             page_text[:200],
                    "text_fingerprint": "",
                    "bbox":             full_bbox,
                    "is_scanned":       is_scanned,
                    "collection":       collection,
                    "section_title":    "",
                    "image_path":       f"{pdf_path.name}::page{page_num + 1}",
                },
                text_dim  = embedder.text_dim,
                image_dim = embedder.image_dim,
            )
            if pt:
                points.append(pt)
            chunk_idx += 1

        # ── Embedded figure chunks → CLIP image embed (typeset only) ──────────
        # Extract each figure/diagram embedded in the page by the PDF author.
        # Skip small images (icons, bullets) below MIN_IMAGE_PX threshold.
        if not is_scanned:
            for img_idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                try:
                    base_img = doc.extract_image(xref)
                    if not base_img:
                        continue
                    if base_img.get("width", 0) < MIN_IMAGE_PX or \
                       base_img.get("height", 0) < MIN_IMAGE_PX:
                        continue

                    # Get image bbox on the page (PDF points, top-left origin)
                    img_bbox = []
                    try:
                        rects = page.get_image_rects(xref)
                        if rects:
                            r = rects[0]
                            img_bbox = [r.x0, r.y0, r.x1, r.y1]
                    except Exception:
                        img_bbox = []

                    # Normalise to RGB PNG for CLIP
                    img_bytes = base_img["image"]
                    if (base_img.get("ext") or "").lower() != "png":
                        xpix = fitz.Pixmap(doc, xref)
                        if xpix.colorspace and xpix.colorspace.n != 3:
                            xpix = fitz.Pixmap(fitz.csRGB, xpix)
                        img_bytes = xpix.tobytes("png")

                    img_b64   = base64.b64encode(img_bytes).decode("ascii")
                    img_vec   = embedder.embed_image_b64(
                        img_b64, label=f"{page_label} fig{img_idx}"
                    )
                    if img_vec is None:
                        chunk_idx += 1
                        continue

                    pt = store.build_point(
                        point_id  = stable_id(str(pdf_path), chunk_idx),
                        text_vec  = None,
                        image_vec = img_vec,
                        payload   = {
                            "modality":         "pdf",
                            "content_type":     "image_embedded",
                            "source_type":      "pdf",
                            "source_id":        collection,
                            "source_path":      str(pdf_path),
                            "source_url":       "",
                            "file_path":        str(pdf_path),
                            "filename":         pdf_path.name,
                            "page_number":      page_num + 1,
                            "text":             f"Figure {img_idx + 1} on page {page_num + 1}",
                            "text_fingerprint": "",
                            "bbox":             img_bbox,
                            "is_scanned":       False,
                            "collection":       collection,
                            "section_title":    "",
                            "image_path":       f"{pdf_path.name}::page{page_num+1}::fig{img_idx}",
                        },
                        text_dim  = embedder.text_dim,
                        image_dim = embedder.image_dim,
                    )
                    if pt:
                        points.append(pt)
                    chunk_idx += 1

                except Exception as e:
                    log.debug("Figure extraction failed", page=page_label,
                              img_idx=img_idx, error=str(e))
                    continue

        if is_scanned and not text_chunks:
            # OCR yielded nothing — store a pure image-only chunk so the page
            # is still retrievable via CLIP image search (ViT visual embedding).
            # The page PNG was already rendered and embedded above as image_page;
            # here we additionally store a dedicated fallback chunk that signals
            # "no text available — matched visually" so the UI can show it clearly.
            if page_img_vec is not None:
                mb        = page.mediabox
                full_bbox = [mb.x0, mb.y0, mb.x1, mb.y1]
                fallback_pt = store.build_point(
                    point_id  = stable_id(str(pdf_path), chunk_idx),
                    text_vec  = page_img_vec,   # mirror image vec into text space
                    image_vec = page_img_vec,
                    payload   = {
                        "modality":         "pdf",
                        "content_type":     "image_page",
                        "source_type":      "pdf",
                        "source_id":        collection,
                        "source_path":      str(pdf_path),
                        "source_url":       "",
                        "file_path":        str(pdf_path),
                        "filename":         pdf_path.name,
                        "page_number":      page_num + 1,
                        "text":             f"[No OCR text] Page {page_num + 1} of {pdf_path.stem} (visually indexed)",
                        "text_fingerprint": "",
                        "bbox":             full_bbox,
                        "is_scanned":       True,
                        "ocr_failed":       True,   # flag for UI: pure visual match
                        "collection":       collection,
                        "section_title":    "",
                        "image_path":       f"{pdf_path.name}::page{page_num + 1}::ocr_fallback",
                    },
                    text_dim  = embedder.text_dim,
                    image_dim = embedder.image_dim,
                )
                if fallback_pt:
                    points.append(fallback_pt)
                    chunk_idx += 1
                    log.info("OCR failed — stored image-only fallback chunk",
                             page=page_label, file=pdf_path.name)
            else:
                log.warning("Scanned page: OCR failed AND image embed failed — page lost",
                            page=page_label, file=pdf_path.name)
        else:
            log.debug("Page indexed", page=page_label,
                      scanned=is_scanned, chunks=len(text_chunks))

    doc.close()

    stored = store.upsert(collection, points)
    log.info("PDF indexed", file=pdf_path.name, points=stored, collection=collection)
    return stored


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Index PDFs into Qdrant")
    parser.add_argument("--path",      help="Single PDF file")
    parser.add_argument("--folder",    help="Folder of PDFs (non-recursive by default)")
    parser.add_argument("--force",     action="store_true",
                        help="Delete existing collection and re-index")
    parser.add_argument("--max-pages", type=int, default=10,
                        help="Max pages per PDF to index (default: 10, 0 = all pages)")
    args = parser.parse_args()

    if not args.path and not args.folder:
        parser.print_help()
        sys.exit(1)

    embedder = Embedder()
    store    = QdrantStore()
    bm25     = BM25Index()
    bm25.load()

    pdf_files: list[Path] = []
    if args.path:
        pdf_files.append(Path(args.path))
    if args.folder:
        pdf_files.extend(sorted(Path(args.folder).glob("*.pdf")))

    total = 0
    for pdf_path in pdf_files:
        if not pdf_path.exists():
            log.error("File not found", path=str(pdf_path))
            continue
        if args.force:
            coll = collection_for_pdf(str(pdf_path))
            if store.collection_exists(coll):
                store.client.delete_collection(coll)
                log.info("Deleted existing collection", coll=coll)
        mp = None if args.max_pages == 0 else args.max_pages
        total += index_pdf(pdf_path, embedder, store, bm25, max_pages=mp)

    # Rebuild BM25 from all text chunks in Qdrant
    pairs = store.scroll_text_chunks()
    if pairs:
        bm25.build_from(pairs)
        bm25.save()
    log.info("Done", total_stored=total, bm25_docs=bm25.doc_count)


if __name__ == "__main__":
    main()
