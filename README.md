# Browser RAG 2

> **Local, fully on-device RAG** — hybrid dense + keyword search over your browser history, PDFs, and Claude conversations, with a Brave/Chrome extension that highlights retrieved passages directly in the source page or document.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Screenshots](#screenshots)
3. [Architecture](#architecture)
4. [Retrieval Pipeline (Detailed)](#retrieval-pipeline-detailed)
5. [Ingestion Pipeline (Detailed)](#ingestion-pipeline-detailed)
6. [Highlight Flow](#highlight-flow)
7. [Prerequisites](#prerequisites)
8. [Quick Start](#quick-start)
9. [All Make Commands](#all-make-commands)
10. [API Reference](#api-reference)
11. [Environment Variables](#environment-variables)
12. [Qdrant Collection Naming](#qdrant-collection-naming)
13. [Payload Schema](#payload-schema)
14. [Sample Query Response](#sample-query-response)
15. [File Structure](#file-structure)

---

## What It Does

Browser RAG 2 indexes three source types — **web pages** (from your browser history), **PDFs** (local files), and **Claude conversation exports** — into a local Qdrant vector database. When you search, it runs a six-way hybrid retrieval across all modalities simultaneously, reranks with a cross-encoder, generates a grounded LLM answer via GPU-accelerated Ollama, and then opens every source in its own browser tab with the exact matching passages highlighted in yellow (text) or blue (images).

**Key properties:**
- 100% local — nothing leaves your machine
- CLIP ViT-B/32 shared 512d embedding space for text *and* images — a text query can retrieve image chunks and vice versa
- Per-modality retrieval budget — PDF, HTML, and Claude each get their own `top_k` candidates before merging, so no source type is drowned out
- OCR fallback — if Docling yields no text for a scanned page, the page image is stored as a pure CLIP visual chunk and is still retrievable
- GPU-accelerated LLM — Ollama is called with `num_gpu=-1` (full offload) and streaming enabled for fast first-token latency

---

## Screenshots

### 1. HTML In-Page Highlighter

The extension's `content.js` injects CSS outlines directly onto the matching DOM elements of the live web page. Yellow = text chunk match. The rank badge (`#1`, `#2`, …) appears above each highlighted element.

![HTML in-page highlighter](docs/screenshots/html_highlighter.png)

> *Query: "What is Unicode Codespace?" — the two paragraphs that were retrieved as the top-scoring chunks are outlined in yellow with rank badges on `reedbeta.com/blog/programmers-intro-to-unicode/`.*

---

### 2. Search Tab — Query in Progress

The Search panel inside the Browser Intelligence extension window. A text query has been submitted and the backend is processing it (hybrid retrieval + rerank + LLM generation).

![Search tab — searching](docs/screenshots/search_tab.png)

> *The extension is opened as a full browser tab (`chrome-extension://…/html/popup.html`). The search bar shows "What is Unicode Codespace?" and the spinner reads "Searching…" while the `/query` API call runs.*

---

### 3. Codebase — `ingest_pdf.py` OCR Fallback

The updated `ingest_pdf.py` open in VS Code, showing the OCR fallback block (lines 525–563). When Docling OCR returns no text for a scanned page, a pure image-only chunk is stored with both `text_vec` and `image_vec` set to the CLIP page embedding, and `ocr_failed: true` in the payload. The terminal below shows live structured logs from the ingestion pipeline.

![ingest_pdf.py OCR fallback + live logs](docs/screenshots/codebase_ingest_pdf.png)

> *Editor shows `ingest_pdf.py` at the OCR fallback section. The terminal shows structured logs: `ingest_html` indexing a page from `karpathy.github.io` — fetching → parsing → text chunks (644 chars) → image chunks (1 image) → `HTML indexed` with 14 points stored into Qdrant.*

---

### 4. Search Result — Answer Tab with Source Link

After retrieval completes, the Answer tab is shown automatically. The LLM answer is displayed with timing, a Copy button, and a clickable source citation that opens the source page in a new tab with highlights. The "↗ Open All 1 Source in New Tabs" button opens every retrieved document simultaneously.

![Search result — Answer tab](docs/screenshots/search_result_answer.png)

> *Query: "What is Unicode Codespace?" — LLM answer displayed with 17770ms generation time. Source `[1] www.reedbeta.com/blog/programmers-intro-to-unicode/` is clickable — it opens the page and triggers the CSS highlight injection shown in Screenshot 1.*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION                                   │
│                                                                     │
│  ingest_pdf.py    PyMuPDF extract text + bboxes (typeset)           │
│                   Docling OCR (scanned) → fallback: CLIP image-only │
│                                                                     │
│  ingest_html.py   Trafilatura XML → text chunks + image URLs        │
│                                                                     │
│  ingest_claude.py conversations.json → Q+A pairs → text chunks      │
│                                                                     │
│  ingest_browser.py  Browser SQLite history → fetch → ingest_html    │
│           │                                                         │
│           ▼                                                         │
│  embed.py          CLIP ViT-B/32  →  512d shared text+image space   │
│                    Fallback: nomic-embed-text (768d) via Ollama      │
│           │                                                         │
│           ▼                                                         │
│  store.py          Qdrant  — one collection per document            │
│                    Two named vectors: "text" (512d) + "image" (512d) │
│  bm25.py           BM25 keyword index — disk-persistent             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          RETRIEVAL                                  │
│                                                                     │
│  query.py                                                           │
│    1. CLIP embed query text → 512d vector                           │
│    2. Dense search — k per modality × vector space:                 │
│         PDF   text-space  (top_k)                                   │
│         PDF   image-space (top_k)                                   │
│         HTML  text-space  (top_k)                                   │
│         HTML  image-space (top_k)                                   │
│         Claude text-space (top_k)                                   │
│    3. BM25 keyword search  (top_k)                                  │
│    4. Six-way RRF merge  (k=60)                                     │
│    5. Cross-encoder rerank  (BAAI/bge-reranker-v2-m3, CPU)          │
│    6. GPU-accelerated LLM  (Ollama, num_gpu=-1, streaming)          │
│                                                                     │
│  Returns: answer + chunks + retrieval_ms + generation_ms            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       EXTENSION UI                                  │
│                                                                     │
│  popup.js        Search panel, Answer tab, "Open All Sources" btn   │
│  background.js   Tab routing:                                       │
│                    PDF    → pdf_viewer.html  (bbox highlights)      │
│                    HTML   → live page tab    (content.js highlights)│
│                    Claude → claude_viewer.html (turn highlights)    │
│  content.js      In-page CSS outline injection via text_fingerprint │
│  pdf_viewer.js   Page PNG render + absolute-positioned bbox overlay │
│  claude_viewer.html  Q+A turn cards with matched text highlighted   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Retrieval Pipeline (Detailed)

### Step 1 — Query Embedding

The query string is embedded with CLIP ViT-B/32 into a 512d vector. Because CLIP uses a shared text+image space, the same vector is used for both text and image searches.

### Step 2 — Per-Modality Dense Search

Each source type gets its own independent `TOP_K_VECTOR` budget:

| Ranked list | Vector space | Description |
|---|---|---|
| `pdf_text_ids` | `"text"` | Text chunks from all PDF collections |
| `pdf_image_ids` | `"image"` | Page renders + embedded figures from PDFs |
| `html_text_ids` | `"text"` | Text chunks from all HTML collections |
| `html_image_ids` | `"image"` | Images extracted from web pages |
| `claude_text_ids` | `"text"` | Q+A pairs from Claude conversation exports |

This guarantees that a heavily-indexed source type cannot crowd out other modalities.

### Step 3 — BM25 Keyword Search

Runs in parallel with dense search. The BM25 index is rebuilt from all Qdrant text chunks after every ingest operation, kept on disk, and memory-mapped on startup.

### Step 4 — Six-Way Reciprocal Rank Fusion

All six ranked lists are merged with RRF (k=60):

```
score(chunk) = Σ  1 / (60 + rank_in_list_i)
               i
```

RRF is rank-based — cosine score magnitudes from different vector spaces are not compared directly. A chunk appearing near the top of multiple lists gets a compounding score boost.

### Step 5 — Cross-Encoder Rerank

Top candidates are re-scored with `BAAI/bge-reranker-v2-m3` on CPU (GPU is reserved for CLIP + LLM). Only text chunks are reranked; image chunks bypass reranking and are appended separately (up to 2 per source type).

### Step 6 — GPU-Accelerated LLM Generation

The top `TOP_K_FINAL` text chunks are assembled into a context window and sent to Ollama with:

```json
{
  "num_gpu":    -1,
  "num_thread": 4,
  "num_ctx":    4096,
  "stream":     true
}
```

`num_gpu=-1` offloads all model layers to GPU. `stream=true` returns tokens as they are generated, minimising time-to-first-token. The response is assembled from the token stream and returned with `generation_duration_ms` timing.

---

## Ingestion Pipeline (Detailed)

### PDF Ingestion (`ingest_pdf.py`)

```
PDF file
  │
  ├─ PyMuPDF: detect scanned vs typeset (< MIN_TEXT_CHARS raw text → scanned)
  │
  ├─ Scanned pages → Docling OCR (one pass, entire PDF)
  │     └─ OCR text + element bboxes (normalised → PDF points)
  │     └─ If OCR yields no text → image-only fallback chunk:
  │           text_vec = image_vec = CLIP page embed
  │           ocr_failed = true  (shown as 👁 visual badge in UI)
  │
  ├─ Per page:
  │     ├─ Text → semantic_chunk() → CLIP text embed
  │     │   → store (content_type="text", bbox from word-overlap matching)
  │     │
  │     ├─ Page render (PDF_DPI DPI) → PNG → CLIP image embed
  │     │   → store (content_type="image_page", full-page bbox)
  │     │
  │     └─ Embedded figures (> MIN_IMAGE_PX) → CLIP image embed
  │         → store (content_type="image_embedded", figure bbox)
  │
  └─ BM25 rebuild from all Qdrant text chunks
```

### HTML Ingestion (`ingest_html.py`)

```
URL
  │
  ├─ requests.get() → raw HTML bytes
  ├─ Trafilatura XML mode → main content text + image src/alt list
  │     (avatars, icons, SVGs filtered out via SKIP_IMAGE_KEYWORDS)
  │
  ├─ Text → semantic_chunk() → CLIP text embed → store (content_type="text")
  └─ Images → CLIP image embed (fetched from URL)
              + alt text proxy embed → store (content_type="image")
```

### Claude Ingestion (`ingest_claude.py`)

```
conversations.json
  │
  ├─ Parse human+assistant turn pairs
  ├─ Format: "Q: <human>\nA: <assistant>" (kept atomic)
  ├─ semantic_chunk() on combined text
  └─ CLIP text embed → store
       source_type="claude"
       url="claude://conversation/<uuid>"
       collection="rag_claude__<uuid[:24]>"
```

### Semantic Chunking (`utils.py`)

spaCy splits text into sentences. Consecutive sentences are grouped into chunks by cosine distance between CLIP embeddings — if `distance > SEMANTIC_THRESHOLD` (default 0.3) and the current chunk has ≥ `MIN_CHUNK_WORDS` words, a new chunk begins. Falls back to word-count splitting if embedding fails.

---

## Highlight Flow

### HTML text chunk
1. User clicks "Open + Highlight ↗" in the popup, or "↗ Open All Sources"
2. `background.js` opens the webpage URL in a new tab
3. Once loaded, `executeScript` dispatches a `RAG_HIGHLIGHT` event with the chunks
4. `content.js` receives the event and calls `resolveElement(chunk)` for each chunk
5. `resolveElement` finds the DOM node by matching `text_fingerprint` (first 80 chars) against `<p>`, `<h1>`–`<h6>`, `<li>`, `<td>`, etc., preferring leaf elements
6. Matching element gets CSS class `rag-hl` with `--rag-color: #FFD700` and a rank badge `#1` above it
7. Page scrolls smoothly to the first highlighted element

### HTML image chunk
- Same flow, but `resolveElement` matches `img[src*="filename"]`
- Highlight colour is `#2196F3` (blue)

### PDF text chunk
1. `background.js` calls `openPdfViewer(filePath, chunks)` → opens `viewer/pdf_viewer.html`
2. Viewer calls `GET /pdf/page?path=…&page=N&dpi=150` → receives PNG from FastAPI
3. `drawHighlights()` computes pixel position from stored `bbox` using `ptToPx = dpi/72`
4. Yellow `div.hl-box` absolutely-positioned overlays are rendered over the PNG

### PDF scanned page / OCR fallback
- Same viewer flow, but `is_scanned=true` or `ocr_failed=true` → full-page blue bounding box
- Sidebar shows a `👁 visual` amber badge: matched visually, not by text

### Claude conversation chunk
1. `background.js` detects `url.startsWith("claude://")` → calls `openClaudeViewer(chunks)`
2. Opens `viewer/claude_viewer.html` in a new tab
3. Viewer renders each Q+A pair as a turn card (human bubble + assistant bubble)
4. Matched text fingerprint highlighted in yellow within the question
5. Sidebar lists all chunks with rank, score, turn number, and text preview

---

## Prerequisites

- Docker + Docker Compose (Podman also works — the Makefile uses `docker compose`)
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for dependency management
- Ollama with models:

```bash
ollama pull gemma3:1b          # LLM for answer generation
ollama pull nomic-embed-text   # fallback text embedder if CLIP unavailable
ollama pull llava-phi3         # legacy VLM (optional — Docling OCR is used instead)
```

- Brave or Chrome browser (for the extension)

---

## Quick Start

```bash
# 0. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. Configure environment
cp .env.example .env
# Edit .env if needed (Qdrant host, model names, etc.)

# 2. Start Qdrant + Redis
make up

# 3. Install Python deps + download CLIP weights + spaCy model (~400MB, one-time)
make setup

# 4. Start the FastAPI server
make server
# → http://localhost:8000

# 5. Load the extension
# Open brave://extensions  (or chrome://extensions)
# Enable "Developer mode"
# Click "Load unpacked" → select the extension/ folder

# 6. Index a single URL to verify the pipeline
make index-url URL=https://reedbeta.com/blog/programmers-intro-to-unicode/

# 7. Index a PDF
make index-pdf PDF=/path/to/your/doc.pdf

# 8. Query from the terminal
make query QUERY="What is Unicode Codespace?"

# 9. Or use the Browser Intelligence extension
# Click the 🧠 icon in the toolbar → Search tab → type your query → Search
# Click "↗ Open All Sources in New Tabs" to see highlights in every source
```

---

## All Make Commands

```bash
make help                           # Print all targets

# ── Services ──────────────────────────────────────────────
make up                             # docker compose up -d (Qdrant + Redis)
make down                           # docker compose down
make restart                        # docker compose restart
make logs                           # tail -f docker logs

# ── Setup ─────────────────────────────────────────────────
make install                        # uv sync (creates .venv)
make setup                          # install + download CLIP + spaCy weights

# ── Server ────────────────────────────────────────────────
make server                         # uvicorn api:app --reload on :8000

# ── Indexing ──────────────────────────────────────────────
make index-url URL=https://…        # Index one URL
make index-pdf PDF=/path/to/doc.pdf # Index one PDF
make index-pdf-force PDF=…          # Re-index even if already indexed
make index-pdf-folder FOLDER=…      # Index all PDFs in a folder

# ── Querying ──────────────────────────────────────────────
make query QUERY="your question"    # Run a query, print answer + sources

# ── Maintenance ───────────────────────────────────────────
make health                         # GET /health | jq
make clean                          # Remove BM25 index files + logs
make reset                          # WIPE Qdrant volumes + BM25 + logs (irreversible)
```

---

## API Reference

| Method | Path | Body / Params | Returns |
|--------|------|---------------|---------|
| `GET` | `/health` | — | `{status, clip_mode, bm25_docs, collections, total_points}` |
| `POST` | `/index/start` | `{index_pdfs, index_web, index_claude, pdf_folder, …}` | `{run_id}` |
| `GET` | `/index/status/{run_id}` | — | `{status, stored, errors, current_step}` |
| `POST` | `/index/test/url` | `{url}` | `{run_id}` |
| `POST` | `/index/test/pdf` | `{path}` | `{run_id}` |
| `POST` | `/index/pdf/folder` | `{folder_path, recursive, max_files}` | `{run_id}` |
| `POST` | `/query` | `{text, top_k, top_urls, search_pdfs, search_html, search_claude}` | `{answer, sources, retrieval_duration_ms, generation_duration_ms, total_chunks_searched}` |
| `GET` | `/pdf/page` | `?path=&page=&dpi=` | PNG image |
| `GET` | `/pdf/info` | `?path=` | `{pages, title}` |
| `POST` | `/admin/wipe-empty` | — | `{deleted: […]}` |

---

## Environment Variables

All settings live in `.env`. Environment variables override `.env` values.

| Variable | Default | Description |
|---|---|---|
| `CLIP_MODEL` | `clip-ViT-B-32` | CLIP model name (sentence-transformers) |
| `EMBED_TEXT_MODEL` | `nomic-embed-text` | Fallback Ollama text embedder |
| `LLM_MODEL` | `gemma3:1b` | Ollama model for answer generation |
| `VLM_MODEL` | `llava-phi3` | Legacy VLM (unused — Docling replaced it) |
| `QDRANT_HOST` | `localhost` | Qdrant hostname |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `SEMANTIC_THRESHOLD` | `0.3` | Cosine distance threshold for chunk splitting |
| `MIN_CHUNK_WORDS` | `15` | Minimum words per chunk |
| `TOP_K_VECTOR` | `20` | Dense candidates per modality per vector space |
| `TOP_K_BM25` | `20` | BM25 keyword candidates |
| `TOP_K_FINAL` | `5` | Final chunks after cross-encoder rerank |
| `RRF_K` | `60` | RRF smoothing constant |
| `PDF_DPI` | `150` | PDF page render resolution |
| `MIN_TEXT_CHARS` | `20` | Chars threshold to classify a page as scanned |
| `MIN_IMAGE_PX` | `80` | Minimum image dimension (px) to embed |
| `EMBED_BATCH_SIZE` | `32` | Embedding batch size |
| `LOG_LEVEL` | `INFO` | Logging level |
| `API_HOST` | `0.0.0.0` | FastAPI bind host |
| `API_PORT` | `8000` | FastAPI port |

---

## Qdrant Collection Naming

```
PDF:    rag_pdf__<sanitised_stem>       e.g. rag_pdf__my_document
HTML:   rag_html__<sanitised_slug>      e.g. rag_html__reedbeta_com_blog_unicode
Claude: rag_claude__<uuid_first_24>     e.g. rag_claude__abc123def456789012345678

Each collection has two named vector spaces:
  "text"  — 512d CLIP  (text chunks + image alt-text proxy)
  "image" — 512d CLIP  (page renders + embedded figures + web images)
```

Chunk IDs are deterministic SHA-256-derived UUIDs: `stable_id(source_path, chunk_index)`. Re-indexing the same document produces the same IDs, enabling safe upserts.

---

## Payload Schema

Every point stored in Qdrant carries a flat payload dict.

### PDF text chunk
```json
{
  "content_type": "text",
  "source_type": "pdf",
  "file_path": "/home/user/paper.pdf",
  "filename": "paper.pdf",
  "page_number": 3,
  "text": "Retrieval Augmented Generation combines...",
  "text_fingerprint": "Retrieval Augmented Generation combin",
  "bbox": [72.0, 120.5, 540.0, 145.2],
  "is_scanned": false,
  "ocr_failed": false,
  "collection": "rag_pdf__paper"
}
```

### PDF scanned page — OCR fallback
```json
{
  "content_type": "image_page",
  "source_type": "pdf",
  "file_path": "/home/user/scanned.pdf",
  "page_number": 5,
  "text": "[No OCR text] Page 5 of scanned (visually indexed)",
  "bbox": [0.0, 0.0, 595.0, 842.0],
  "is_scanned": true,
  "ocr_failed": true,
  "image_path": "scanned.pdf::page5::ocr_fallback",
  "collection": "rag_pdf__scanned"
}
```

### HTML text chunk
```json
{
  "content_type": "text",
  "source_type": "html",
  "url": "https://reedbeta.com/blog/programmers-intro-to-unicode/",
  "text": "The set of all possible code points is called the codespace...",
  "text_fingerprint": "The set of all possible code points i",
  "img_src": "",
  "collection": "rag_html__reedbeta_com_blog_programmers_intro_to_unicode"
}
```

### HTML image chunk
```json
{
  "content_type": "image",
  "source_type": "html",
  "url": "https://example.com/article",
  "text": "Unicode codespace map diagram",
  "img_src": "https://example.com/images/codespace-map.png",
  "img_alt": "Unicode codespace map diagram",
  "collection": "rag_html__example_com_article"
}
```

### Claude conversation chunk
```json
{
  "content_type": "text",
  "source_type": "claude",
  "url": "claude://conversation/abc-123-def",
  "text": "Q: What is RAG?\nA: RAG stands for Retrieval Augmented Generation...",
  "text_fingerprint": "Q: What is RAG?\nA: RAG stands for Re",
  "conv_name": "RAG architecture discussion",
  "conv_uuid": "abc-123-def",
  "conversation_id": "abc-123-def",
  "turn_idx": 0,
  "speaker_role": "human+assistant",
  "timestamp": "2025-11-14T10:23:00Z",
  "collection": "rag_claude__abc123def"
}
```

---

## Sample Query Response

`POST /query` — `{"text": "What is Unicode Codespace?", "top_k": 5, "top_urls": 3}`

```json
{
  "answer": "Code points are identified by number, customarily written in hexadecimal with the prefix 'U+', such as U+0041 'A' latin capital letter a or U+03B8 'θ' greek small letter theta. The set of all possible code points is called the codespace, consisting of 1,114,112 code points.",
  "query": "What is Unicode Codespace?",
  "retrieval_duration_ms": 312.4,
  "generation_duration_ms": 17770.2,
  "total_chunks_searched": 14,
  "sources": [
    {
      "url": "https://reedbeta.com/blog/programmers-intro-to-unicode/",
      "file_path": "",
      "doc_type": "html",
      "source_type": "html",
      "top_score": 0.01626,
      "snippet": "The set of all possible code points is called the codespace...",
      "text_chunks": [
        {
          "chunk_id": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
          "content_type": "text",
          "text": "The set of all possible code points is called the codespace...",
          "score": 0.01626,
          "similarity_score": 0.312,
          "source_type": "html",
          "web_url": "https://reedbeta.com/blog/programmers-intro-to-unicode/",
          "text_fingerprint": "The set of all possible code points i",
          "highlight_color": "#FFD700"
        }
      ],
      "image_chunks": []
    }
  ]
}
```

---

## File Structure

```
rag2/
├── src/
│   ├── api.py              FastAPI server — all HTTP endpoints
│   ├── query.py            Retrieval pipeline: embed → dense → BM25 → RRF → rerank → LLM
│   ├── embed.py            CLIP / Ollama embedder
│   ├── store.py            Qdrant collection management + search
│   ├── bm25.py             BM25 keyword index (bm25s, disk-persistent)
│   ├── ingest_pdf.py       PDF indexer (PyMuPDF + Docling OCR + CLIP, OCR fallback)
│   ├── ingest_html.py      HTML indexer (Trafilatura + CLIP)
│   ├── ingest_claude.py    Claude conversation export indexer
│   ├── ingest_browser.py   Browser SQLite history reader → ingest_html
│   ├── config.py           All settings (loaded from .env)
│   ├── utils.py            semantic_chunk, rrf_merge, is_avatar_or_icon
│   └── logger.py           Structured logging (console colour + JSON file)
│
├── extension/
│   ├── manifest.json           MV3 extension manifest
│   ├── js/
│   │   ├── background.js       Service worker: tab routing for PDF / HTML / Claude
│   │   ├── popup.js            Extension UI logic, search, "Open All Sources"
│   │   └── content.js          In-page CSS highlight injection
│   ├── html/
│   │   └── popup.html          Extension window HTML + all CSS
│   ├── viewer/
│   │   ├── pdf_viewer.html     PDF viewer shell
│   │   ├── pdf_viewer.js       Page PNG render + bbox highlight overlay
│   │   ├── claude_viewer.html  Claude conversation turn viewer (Q+A cards)
│   │   └── pdfjs/              PDF.js library (bundled)
│   ├── css/
│   │   ├── popup.css
│   │   └── highlight.css
│   └── icons/
│
├── docs/
│   └── screenshots/
│       ├── html_highlighter.png
│       ├── search_tab.png
│       ├── codebase_ingest_pdf.png
│       └── search_result_answer.png
│
├── eval/
│   ├── run_hf_eval.py      HuggingFace dataset evaluation runner
│   └── README.md
│
├── tests/
│   └── test_pipeline.py
│
├── docker-compose.yml      Qdrant + Redis services
├── Makefile                All developer commands
├── pyproject.toml          Python project + dependency spec (uv)
├── .env.example            Documented environment variable template
└── README.md               This file
```
