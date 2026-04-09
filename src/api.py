"""
api.py — FastAPI server matching the Browser RAG extension's exact API contract.

Endpoints the extension calls:
  GET  /health
  POST /index/start          body: {index_web, index_pdfs, pdf_folder, ...}
  GET  /index/status/{run_id}
  POST /index/test/url       body: {url}
  POST /index/test/pdf       body: {path}
  POST /index/pdf/folder     body: {folder_path, recursive}
  POST /query                body: {text, top_k, top_urls}
  GET  /pdf/page             ?path=&page=&dpi=
  GET  /pdf/info             ?path=

Run:
  python api.py
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from logger import get_logger

log = get_logger(__name__)

from config import API_HOST, API_PORT, PDF_DPI
from embed import Embedder
from store import QdrantStore, collection_for_pdf, collection_for_html
from bm25 import BM25Index
import query as query_module
from ingest_pdf import index_pdf
from ingest_html import index_html, fetch_html


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Browser RAG", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared services ───────────────────────────────────────────────────────────

_embedder = None
_store    = None
_bm25     = None


def services():
    global _embedder, _store, _bm25
    if _embedder is None:
        log.info("Initialising services")
        _embedder = Embedder()
        _store    = QdrantStore()
        _bm25     = BM25Index()
        _bm25.load()
        log.info("Services ready")
    return _embedder, _store, _bm25


# startup: pre-warm services when server starts
# Using lifespan to avoid FastAPI deprecation warning
from contextlib import asynccontextmanager as _acm

@_acm
async def _lifespan(app):
    services()
    yield

# Patch lifespan onto the already-created app
app.router.lifespan_context = _lifespan


# ── Run tracker ───────────────────────────────────────────────────────────────

_runs: dict = {}


def new_run() -> str:
    run_id = str(uuid.uuid4())[:8]
    _runs[run_id] = {"run_id": run_id, "status": "running",
                     "stored": 0, "parsed": 0, "errors": [],
                     "current_step": "starting"}
    return run_id


def finish_run(run_id, stored=0, errors=None):
    if run_id in _runs:
        _runs[run_id].update({"status": "completed", "stored": stored,
                               "errors": errors or []})


def fail_run(run_id, error):
    if run_id in _runs:
        _runs[run_id].update({"status": "failed", "errors": [error]})


# ── Request models ────────────────────────────────────────────────────────────

class IndexStartRequest(BaseModel):
    index_web:         bool      = False
    index_pdfs:        bool      = False
    index_claude:      bool      = False
    # PDF settings
    browser:           str       = "brave"
    pdf_folder:        str       = ""
    max_files:         int | None = 10     # default 10 PDF files
    max_pages:         int        = 50     # pages per PDF, default 50
    recursive:         bool       = False
    # Web history settings
    history_limit:     int        = 10     # max URLs, default 10
    # Claude settings
    claude_json_path:  str        = ""
    max_conversations: int | None = 10     # default 10 conversations

class IndexTestUrlRequest(BaseModel):
    url: str

class IndexTestPdfRequest(BaseModel):
    path: str

class IndexFolderRequest(BaseModel):
    folder_path: str
    recursive:   bool  = False
    force:       bool  = False
    max_files:   int | None = None   # None = index all files in folder

class QueryRequest(BaseModel):
    text:          str
    top_k:         int  = 5
    top_urls:      int  = 3
    search_pdfs:   bool = True
    search_html:   bool = True
    search_claude: bool = True


# ── Source-specific indexers (called from index_start) ───────────────────────

def _index_browser_history(browser: str, limit: int,
                            emb, store, bm25, run_id: str) -> int:
    """Read browser SQLite history → fetch each URL → Trafilatura → CLIP → Qdrant."""
    from ingest_browser import index_browser_history
    return index_browser_history(
        browser=browser,
        limit=limit or 200,
        embedder=emb,
        store=store,
        bm25=bm25,
        run_id=run_id,
    )


def _index_claude_chats(json_path: str, emb, store, bm25,
                         run_id: str, max_conversations: int | None = None) -> int:
    """Parse Claude conversations.json export → Q+A pairs → CLIP → Qdrant."""
    if not json_path:
        raise ValueError("claude_json_path not provided")
    from ingest_claude import index_claude
    return index_claude(
        json_path=Path(json_path),
        embedder=emb,
        store=store,
        bm25=bm25,
        limit=max_conversations,
    )


# ── BM25 rebuild helper ──────────────────────────────────────────────────────

def _rebuild_bm25(store: QdrantStore, bm25: BM25Index) -> None:
    """
    Rebuild BM25 index from all text chunks currently in Qdrant.
    Called after every indexing operation to keep BM25 in sync.
    Memory efficient: scrolls Qdrant in batches, builds once.
    """
    try:
        pairs = store.scroll_text_chunks()   # [(chunk_id, text), ...]
        if not pairs:
            log.warning("BM25 rebuild: no text chunks found in Qdrant")
            return
        bm25.build_from(pairs)
        bm25.save()
        log.info("BM25 rebuilt", docs=len(pairs))
    except Exception as e:
        log.error("BM25 rebuild failed", error=str(e))


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    emb, store, bm25 = services()
    colls = store.all_collections()
    # Report empty collections so user knows to re-index
    empty = [c for c in colls if store.point_count(c) == 0]
    total = sum(store.point_count(c) for c in colls)
    return {
        "status":             "ok",
        "clip_mode":          emb._mode,
        "text_dim":           emb.text_dim,
        "bm25_docs":          bm25.doc_count,
        "collections":        colls,
        "total_points":       total,
        "empty_collections":  empty,
        "note": (f"{len(empty)} empty collections — run wipe-collections then re-index"
                 if empty else ""),
    }


@app.post("/admin/wipe-empty")
async def wipe_empty_collections():
    """Delete all collections that have zero points (created but never indexed)."""
    _, store, _ = services()
    deleted = []
    for coll in store.all_collections():
        if store.point_count(coll) == 0:
            try:
                store.client.delete_collection(coll)
                deleted.append(coll)
            except Exception as e:
                log.warning("Could not delete empty collection",
                            coll=coll, error=str(e))
    log.info("Wiped empty collections", count=len(deleted))
    return {"deleted": deleted, "count": len(deleted)}


# ── Index start (bulk) ────────────────────────────────────────────────────────

@app.post("/index/start")
async def index_start(bg: BackgroundTasks, req: IndexStartRequest = None):
    run_id = new_run()

    # Default to empty request if body was omitted
    if req is None:
        req = IndexStartRequest()

    async def _run():
        """
        Sequential indexing across all requested source types.
        Each source type runs independently — failure in one does NOT
        stop the others. Errors are collected and reported in the run status.
        """
        emb, store, bm25 = services()
        total  = 0
        errors = []

        # ── 1. PDF folder ─────────────────────────────────────────────────────
        if req.index_pdfs:
            _runs[run_id]["current_step"] = "pdfs"
            if not req.pdf_folder:
                errors.append("PDF: no folder path provided")
                log.warning("index_pdfs=True but pdf_folder is empty", run_id=run_id)
            else:
                folder = Path(req.pdf_folder)
                if not folder.exists():
                    errors.append(f"PDF: folder not found: {req.pdf_folder}")
                    log.error("PDF folder not found", path=req.pdf_folder)
                else:
                    pattern  = "**/*.pdf" if req.recursive else "*.pdf"
                    all_pdfs = sorted(folder.glob(pattern))
                    if req.max_files and req.max_files > 0:
                        all_pdfs = all_pdfs[:req.max_files]
                    log.info("Indexing PDF folder", folder=req.pdf_folder,
                             count=len(all_pdfs), max_files=req.max_files)
                    for pdf_path in all_pdfs:
                        try:
                            n = index_pdf(pdf_path, emb, store, bm25,
                                          max_pages=req.max_pages)
                            total += n
                            _runs[run_id]["stored"] = total
                            _runs[run_id]["parsed"] += 1
                        except Exception as e:
                            err = f"PDF {pdf_path.name}: {e}"
                            errors.append(err)
                            log.error("PDF indexing failed", file=pdf_path.name, error=str(e))
                            # Continue with next PDF — do not abort

        # ── 2. Web history (browser URLs) ─────────────────────────────────────
        if req.index_web:
            _runs[run_id]["current_step"] = "web"
            try:
                n = _index_browser_history(
                    browser=req.browser,
                    limit=req.history_limit,
                    emb=emb, store=store, bm25=bm25,
                    run_id=run_id,
                )
                total += n
                _runs[run_id]["stored"] = total
            except NotImplementedError as e:
                errors.append(f"Web: {e}")
                log.warning("Web history indexing skipped", reason=str(e))
            except Exception as e:
                errors.append(f"Web: {e}")
                log.error("Web history indexing failed", error=str(e))
                # Continue to next source — do not abort

        # ── 3. Claude chat export ─────────────────────────────────────────────
        if req.index_claude:
            _runs[run_id]["current_step"] = "claude"
            try:
                n = _index_claude_chats(
                    json_path=req.claude_json_path,
                    emb=emb, store=store, bm25=bm25,
                    run_id=run_id,
                    max_conversations=req.max_conversations,
                )
                total += n
                _runs[run_id]["stored"] = total
            except NotImplementedError as e:
                errors.append(f"Claude: {e}")
                log.warning("Claude chat indexing skipped", reason=str(e))
            except Exception as e:
                errors.append(f"Claude: {e}")
                log.error("Claude chat indexing failed", error=str(e))
                # Continue — do not abort

        # ── Rebuild BM25 once after all sources ───────────────────────────────
        _runs[run_id]["current_step"] = "bm25"
        _rebuild_bm25(store, bm25)

        # Mark complete regardless of per-source errors
        # (errors are surfaced in the status response, not as a hard failure)
        status = "completed" if not errors else "partial"
        _runs[run_id].update({
            "status":       status,
            "stored":       total,
            "errors":       errors,
            "current_step": "done",
        })
        log.info("Indexing run complete", run_id=run_id,
                 status=status, stored=total, errors=len(errors))

    bg.add_task(_run)
    return {"run_id": run_id, "status": "started"}


@app.get("/index/status/{run_id}")
async def index_status(run_id: str):
    if run_id not in _runs:
        raise HTTPException(404, detail=f"Run {run_id!r} not found")
    return _runs[run_id]


@app.post("/index/test/url")
async def index_test_url(req: IndexTestUrlRequest, bg: BackgroundTasks):
    run_id = new_run()

    async def _run():
        emb, store, bm25 = services()
        try:
            html_bytes = fetch_html(req.url)
            n = index_html(req.url, html_bytes, emb, store, bm25)
            _rebuild_bm25(store, bm25)
            finish_run(run_id, n)
        except Exception as e:
            fail_run(run_id, str(e))

    bg.add_task(_run)
    return {"run_id": run_id, "status": "started"}


@app.post("/index/test/pdf")
async def index_test_pdf(req: IndexTestPdfRequest, bg: BackgroundTasks):
    pdf_path = Path(req.path)
    if not pdf_path.exists():
        raise HTTPException(404, detail=f"File not found: {req.path}")
    run_id = new_run()

    async def _run():
        emb, store, bm25 = services()
        try:
            n = index_pdf(pdf_path, emb, store, bm25)
            _rebuild_bm25(store, bm25)
            finish_run(run_id, n)
        except Exception as e:
            fail_run(run_id, str(e))

    bg.add_task(_run)
    return {"run_id": run_id, "status": "started"}


@app.post("/index/pdf/folder")
async def index_pdf_folder(req: IndexFolderRequest, bg: BackgroundTasks):
    folder = Path(req.folder_path)
    if not folder.exists():
        raise HTTPException(404, detail=f"Folder not found: {req.folder_path}")
    run_id = new_run()

    async def _run():
        emb, store, bm25 = services()
        total, errors = 0, []
        pattern = "**/*.pdf" if req.recursive else "*.pdf"
        all_pdfs = sorted(folder.glob(pattern))
        # Apply max_files limit if set; otherwise index everything
        if req.max_files and req.max_files > 0:
            all_pdfs = all_pdfs[:req.max_files]
        log.info("Indexing PDF folder", folder=str(folder),
                 total=len(all_pdfs), max_files=req.max_files)
        for pdf_path in all_pdfs:
            try:
                if req.force:
                    coll = collection_for_pdf(str(pdf_path))
                    if store.collection_exists(coll):
                        store.client.delete_collection(coll)
                n = index_pdf(pdf_path, emb, store, bm25)
                total += n
                _runs[run_id]["stored"] = total
                _runs[run_id]["parsed"] += 1
            except Exception as e:
                errors.append(f"{pdf_path.name}: {e}")
        _rebuild_bm25(store, bm25)
        finish_run(run_id, total, errors)

    bg.add_task(_run)
    return {"run_id": run_id, "status": "started"}


# ── Query ─────────────────────────────────────────────────────────────────────

def _color_for(content_type: str) -> str:
    return {"text": "#FFD700", "image": "#2196F3",
            "image_page": "#2196F3", "image_embedded": "#2196F3"}.get(content_type, "#FFD700")


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    """
    Returns the shape the extension's renderSourceCard() expects:
    {
      answer: str,
      sources: [{
        url, file_path, doc_type, source_type, top_score, snippet,
        chunks, text_chunks, image_chunks
      }]
    }
    """
    emb, store, bm25 = services()
    raw = query_module.query(
        query_text=req.text,
        embedder=emb, store=store, bm25=bm25,
        top_k_final=req.top_k,
        search_pdfs=req.search_pdfs,
        search_html=req.search_html,
        search_claude=req.search_claude,
    )

    # Group chunks by document
    source_map: dict = {}
    for chunk in raw["chunks"]:
        is_pdf = chunk.get("source_type") == "pdf"
        key    = chunk.get("file_path") if is_pdf else chunk.get("url", "")
        if not key:
            continue

        if key not in source_map:
            source_map[key] = {
                "url":          f"file://{key}" if is_pdf else key,
                "file_path":    key if is_pdf else "",
                "doc_type":     "pdf" if is_pdf else "html",
                "source_type":  chunk.get("source_type", ""),
                "top_score":    0.0,
                "snippet":      "",
                "chunks":       [],
                "text_chunks":  [],
                "image_chunks": [],
            }

        s = source_map[key]
        score            = chunk.get("score", 0.0)            # RRF score
        similarity_score = chunk.get("similarity_score", score) # cosine score
        if score > s["top_score"]:
            s["top_score"] = score
            if chunk.get("content_type") == "text":
                s["snippet"] = chunk["text"][:200]

        c = {
            # Core
            "chunk_id":         chunk["chunk_id"],
            "content_type":     chunk["content_type"],
            "text":             chunk["text"],
            "score":            score,
            "similarity_score": similarity_score,
            # Spec-required fields
            "modality":         chunk.get("modality", chunk.get("source_type", "")),
            "source_id":        chunk.get("source_id", ""),
            "source_path":      chunk.get("source_path", chunk.get("file_path", "")),
            "source_url":       chunk.get("source_url", chunk.get("url", "")),
            "filename":         chunk.get("filename", ""),
            # PDF
            "file_path":        chunk.get("file_path", ""),
            "page_number":      chunk.get("page_number", 0),
            "bbox":             chunk.get("bbox", []),
            "is_scanned":       chunk.get("is_scanned", False),
            "ocr_failed":       chunk.get("ocr_failed", False),
            "section_title":    chunk.get("section_title", ""),
            # Image
            "img_src":          chunk.get("img_src", ""),
            "img_alt":          chunk.get("img_alt", ""),
            "image_path":       chunk.get("image_path", ""),
            # HTML
            "web_url":          "" if is_pdf else chunk.get("url", ""),
            "url":              chunk.get("url", ""),
            # Claude
            "conversation_id":  chunk.get("conversation_id", ""),
            "turn_index":       chunk.get("turn_index", 0),
            "speaker_role":     chunk.get("speaker_role", ""),
            "timestamp":        chunk.get("timestamp", ""),
            "message_id":       chunk.get("message_id", ""),
            # UI
            "text_fingerprint": chunk.get("text_fingerprint", ""),
            "source_type":      chunk.get("source_type", ""),
            "highlight_color":  _color_for(chunk["content_type"]),
        }
        s["chunks"].append(c)
        if chunk["content_type"] == "text":
            s["text_chunks"].append(c)
        else:
            s["image_chunks"].append(c)

    sources = sorted(source_map.values(), key=lambda s: s["top_score"], reverse=True)
    return {
        "answer":                  raw["answer"],
        "sources":                 sources[:req.top_urls],
        "query":                   req.text,
        "retrieval_duration_ms":   raw.get("retrieval_duration_ms", 0.0),
        "generation_duration_ms":  raw.get("generation_duration_ms", 0.0),
        "total_chunks_searched":   raw.get("total_chunks_searched", 0),
    }


# ── PDF rendering ─────────────────────────────────────────────────────────────

@app.get("/pdf/page")
async def pdf_page(path: str, page: int = 1, dpi: int = PDF_DPI):
    try:
        import fitz
    except ImportError:
        raise HTTPException(500, detail="PyMuPDF not installed")
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise HTTPException(404, detail=f"File not found: {path}")
    try:
        doc = fitz.open(str(pdf_path))
        if page < 1 or page > doc.page_count:
            raise HTTPException(400, detail=f"Page {page} out of range")
        pix = doc[page - 1].get_pixmap(
            matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
        png = pix.tobytes("png")
        doc.close()
        return Response(content=png, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/pdf/info")
async def pdf_info(path: str):
    try:
        import fitz
    except ImportError:
        raise HTTPException(500, detail="PyMuPDF not installed")
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise HTTPException(404, detail=f"File not found: {path}")
    try:
        doc  = fitz.open(str(pdf_path))
        meta = doc.metadata or {}
        info = {"file": str(pdf_path), "pages": doc.page_count,
                "title": meta.get("title") or pdf_path.stem}
        doc.close()
        return info
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=True)
