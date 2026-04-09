'use strict';

const API = 'http://localhost:8000';

// ── Parse query params ──────────────────────────────────────────────────────
const params   = new URLSearchParams(location.search);
const filePath = params.get('path') || '';   // absolute file path

let chunks = [];
try {
  const raw = params.get('data') || '[]';
  chunks = JSON.parse(decodeURIComponent(raw));
  console.log('[PDF Viewer] filePath:', filePath);
  console.log('[PDF Viewer] chunks loaded:', chunks.length, chunks);
} catch (e) {
  console.warn('[PDF Viewer] failed to parse chunks from URL:', e);
  console.log('[PDF Viewer] raw data param:', params.get('data'));
  chunks = [];
}

// ── State ───────────────────────────────────────────────────────────────────
let totalPages  = 1;
let currentPage = 1;
let zoom        = 1.0;
let activeChunk = null;

// ── DOM refs ────────────────────────────────────────────────────────────────
const pageImg       = document.getElementById('page-img');
const pageContainer = document.getElementById('page-container');
const pageInfo      = document.getElementById('page-info');
const statusBar     = document.getElementById('status-bar');
const fileNameEl    = document.getElementById('file-name');
const chunkList     = document.getElementById('chunk-list');
const chunkCount    = document.getElementById('chunk-count');

// ── Bootstrap ────────────────────────────────────────────────────────────────
async function init() {
  if (!filePath) {
    statusBar.textContent = 'No file path provided.';
    return;
  }

  // Show file name
  fileNameEl.textContent = filePath.split('/').pop();

  // Fetch PDF info (page count, title) — optional, fall back gracefully
  try {
    const infoResp = await fetch(`${API}/pdf/info?path=${encodeURIComponent(filePath)}`);
    if (infoResp.ok) {
      const info = await infoResp.json();
      totalPages = info.pages || 1;
      statusBar.textContent = `${totalPages} pages`;
    } else {
      // Server reachable but error — continue with page 1
      console.warn('[PDF Viewer] pdf/info returned', infoResp.status);
      totalPages = 99;  // allow navigation even without info
      statusBar.textContent = 'Ready';
    }
  } catch (e) {
    // fetch blocked (CSP) or server down — still try to render via img.src
    // img.src requests are NOT blocked by connect-src CSP
    console.warn('[PDF Viewer] fetch blocked or server down:', e.message);
    totalPages = 99;
    statusBar.textContent = 'Ready (offline mode)';
  }

  // Populate sidebar
  buildSidebar();

  // Navigate to the page of the highest-scoring chunk
  if (chunks.length > 0) {
    const best = chunks.reduce((a, b) =>
      (a.similarity_score || a.score || 0) >= (b.similarity_score || b.score || 0) ? a : b
    );
    const startPage = best.page_number || best.page || 1;
    await goToPage(startPage, best);
  } else {
    await goToPage(1);
  }
}

// ── Sidebar ──────────────────────────────────────────────────────────────────
function buildSidebar() {
  if (!chunks.length) return;
  chunkCount.textContent = `(${chunks.length})`;
  document.getElementById('empty-msg')?.remove();

  chunks.forEach((c, i) => {
    const score   = (c.similarity_score || c.score || 0).toFixed(3);
    const pageNum = c.page_number || c.page || '?';
    const text    = (c.text || c.img_alt || '').slice(0, 120);
    const ocrFlag = c.ocr_failed
      ? '<span style="color:#e3b341;font-size:10px;margin-left:4px" title="OCR failed — visual match only">👁 visual</span>'
      : '';

    const item = document.createElement('div');
    item.className = 'chunk-item';
    item.innerHTML = `
      <div><span class="chunk-rank">#${i+1}</span><span class="chunk-score">score ${score}</span>${ocrFlag}</div>
      <div class="chunk-page">📄 Page ${pageNum}</div>
      <div class="chunk-text">${esc(text)}</div>
    `;
    item.addEventListener('click', () => goToPage(pageNum, c, item));
    chunkList.appendChild(item);
  });
}

// ── Page rendering ───────────────────────────────────────────────────────────
async function goToPage(pageNum, focusChunk = null, sidebarItem = null) {
  currentPage = Math.max(1, Math.min(pageNum, totalPages));
  pageInfo.textContent = `${currentPage} / ${totalPages}`;

  // Update sidebar active state
  document.querySelectorAll('.chunk-item.active').forEach(el => el.classList.remove('active'));
  if (sidebarItem) {
    sidebarItem.classList.add('active');
    sidebarItem.scrollIntoView({ block: 'nearest' });
  }

  // Fetch the page PNG from the FastAPI server
  statusBar.textContent = 'Rendering…';
  const dpi = Math.round(150 * zoom);
  const imgUrl = `${API}/pdf/page?path=${encodeURIComponent(filePath)}&page=${currentPage}&dpi=${dpi}`;

  // Remove old highlights
  pageContainer.querySelectorAll('.hl-box').forEach(el => el.remove());

  console.log('[PDF Viewer] loading page image:', imgUrl);
  try {
    await new Promise((resolve, reject) => {
      pageImg.onload  = resolve;
      pageImg.onerror = (e) => {
        console.error('[PDF Viewer] img load failed:', imgUrl, e);
        reject(e);
      };
      pageImg.src     = imgUrl;
    });
    pageContainer.style.width  = pageImg.naturalWidth  + 'px';
    pageContainer.style.height = pageImg.naturalHeight + 'px';

    statusBar.textContent = `Page ${currentPage} ✓`;
    drawHighlights(focusChunk);
  } catch (e) {
    statusBar.textContent = `Render failed — server at ${API} reachable?`;
    console.error('[PDF Viewer] render failed for:', imgUrl);
  }
}

// ── Highlighting ─────────────────────────────────────────────────────────────
function drawHighlights(focusChunk) {
  const imgW = pageImg.naturalWidth;
  const imgH = pageImg.naturalHeight;

  // All chunks on this page
  const pageChunks = chunks.filter(c => (c.page_number || c.page || 0) === currentPage);

  pageChunks.forEach((c, i) => {
    const isFocus   = focusChunk && (c.chunk_id === focusChunk.chunk_id);
    const isImage   = c.content_type !== 'text';
    const isScanned = c.is_scanned === true;
    const bbox      = c.bbox;  // [x0, y0, x1, y1] in PDF points

    if (bbox && bbox.length === 4 && (bbox[2] - bbox[0]) > 1 && (bbox[3] - bbox[1]) > 1) {
      // ── Real bbox — PyMuPDF coordinates ────────────────────────────────
      // PyMuPDF uses top-left origin (unlike standard PDF which is bottom-left).
      // page.search_for() returns Rect(x0, y0, x1, y1) with y increasing downward.
      // The rendered PNG is at PDF_DPI dpi, so scale = dpi/72.
      // We don't know original page pts here, so we use the rendered image dims
      // as a proxy: pdfW ≈ imgW / (dpi/72), same for H.
      // dpi used when this page was rendered = 150 * zoom (see goToPage)
      const renderedDpi = 150 * zoom;
      const ptToPx = renderedDpi / 72;  // 1 PDF point → pixels at render dpi
      // PyMuPDF top-left origin — no Y flip needed
      const cssX0 = bbox[0] * ptToPx;
      const cssY0 = bbox[1] * ptToPx;
      const cssW  = (bbox[2] - bbox[0]) * ptToPx;
      const cssH  = (bbox[3] - bbox[1]) * ptToPx;
      drawBox(cssX0, cssY0, cssW, cssH, i + 1, isFocus, isImage);

    } else if (isScanned || (isImage && (!bbox || bbox.length < 4))) {
      // ── Scanned page or image chunk with no bbox ────────────────────────
      // Draw a full-page blue outline — user can see the whole page is the match
      const pad = 8;
      drawBox(pad, pad, imgW - pad * 2, imgH - pad * 2, i + 1, isFocus, true);

    }
    // Text chunks with no bbox: just navigate to the page, no box drawn
  });
}

function drawBox(x, y, w, h, rank, isFocus, isImage) {
  const box = document.createElement('div');
  box.className = `hl-box ${isFocus ? 'active-hl' : isImage ? 'img-hl' : 'text-hl'}`;
  box.style.cssText = `left:${x}px; top:${y}px; width:${w}px; height:${h}px;`;

  const label = document.createElement('div');
  label.className = 'hl-label';
  label.textContent = `#${rank}`;
  box.appendChild(label);

  pageContainer.appendChild(box);
}

// ── Controls ─────────────────────────────────────────────────────────────────
document.getElementById('btn-prev').addEventListener('click', () => {
  if (currentPage > 1) goToPage(currentPage - 1);
});
document.getElementById('btn-next').addEventListener('click', () => {
  if (currentPage < totalPages) goToPage(currentPage + 1);
});
document.getElementById('btn-zoom-in').addEventListener('click', () => {
  zoom = Math.min(zoom + 0.25, 3.0);
  goToPage(currentPage, null);
});
document.getElementById('btn-zoom-out').addEventListener('click', () => {
  zoom = Math.max(zoom - 0.25, 0.5);
  goToPage(currentPage, null);
});

// Keyboard navigation
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft'  || e.key === 'ArrowUp')   goToPage(currentPage - 1);
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown')  goToPage(currentPage + 1);
});

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                  .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

init();