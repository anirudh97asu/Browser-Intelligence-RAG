/**
 * content.js — Browser RAG in-page highlighter
 *
 * Injected into every HTTP/HTTPS page.
 * Listens for RAG_HIGHLIGHT custom events dispatched by background.js
 * and applies colour-coded outlines to matching DOM elements.
 *
 * Matching strategies (tried in order):
 *   1. XPath         — precise, populated for future content
 *   2. CSS selector  — precise
 *   3. text_fingerprint (first 80 chars) — primary working strategy
 *   4. chunk.text substring search       — fallback
 *   5. img[src] match for image chunks
 *
 * Colour scheme (matches backend ContentType → HighlightColor):
 *   text   → #FFD700  yellow
 *   image  → #2196F3  blue
 *   table  → #4CAF50  green
 *   code   → #F44336  red
 */

'use strict';

const STYLE_ID = 'rag-highlight-styles';
const HIGHLIGHT_CLASS = 'rag-hl';

const CONTENT_COLORS = {
  text:  '#FFD700',
  image: '#2196F3',
  table: '#4CAF50',
  code:  '#F44336',
};

// ── Styles ────────────────────────────────────────────────────────────────────

function injectStyles() {
  if (document.getElementById(STYLE_ID)) return;

  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = `
    .${HIGHLIGHT_CLASS} {
      outline: 3px solid var(--rag-color, #FFD700) !important;
      background-color: color-mix(
        in srgb,
        var(--rag-color, #FFD700) 18%,
        transparent
      ) !important;
      border-radius: 3px;
      position: relative;
      scroll-margin-top: 80px;
      transition: outline 0.2s ease, background-color 0.2s ease;
    }

    /* Rank badge */
    .${HIGHLIGHT_CLASS}::before {
      content: attr(data-rag-rank);
      position: absolute;
      top: -20px;
      left: 0;
      background: var(--rag-color, #FFD700);
      color: #000;
      font-size: 10px;
      font-weight: 700;
      font-family: monospace;
      padding: 1px 5px;
      border-radius: 3px;
      z-index: 2147483647;
      pointer-events: none;
      white-space: nowrap;
    }

    /* Pulse animation on first render */
    @keyframes rag-pulse {
      0%   { box-shadow: 0 0 0 0 var(--rag-color, #FFD700); }
      60%  { box-shadow: 0 0 0 8px transparent; }
      100% { box-shadow: 0 0 0 0 transparent; }
    }
    .${HIGHLIGHT_CLASS}.rag-hl-new {
      animation: rag-pulse 1s ease-out forwards;
    }
  `;
  document.head.appendChild(style);
}


// ── Clear previous highlights ─────────────────────────────────────────────────

function clearHighlights() {
  document.querySelectorAll(`.${HIGHLIGHT_CLASS}`).forEach((el) => {
    el.classList.remove(HIGHLIGHT_CLASS, 'rag-hl-new');
    el.style.removeProperty('--rag-color');
    delete el.dataset.ragRank;
  });
}


// ── Element resolution ────────────────────────────────────────────────────────

// Only leaf-ish text elements — no containers like article/section/div
// that would match the entire page text and create a full-page highlight.
const SEARCH_SELECTORS =
  'p, h1, h2, h3, h4, h5, h6, li, td, th, blockquote, pre, code, figcaption, span';

function resolveElement(chunk) {
  const fingerprint = (chunk.text_fingerprint || '').trim();
  const text        = (chunk.text || '').trim();
  const imgSrc      = chunk.img_src || '';
  const isImage     = chunk.content_type === 'image';

  // ── Image chunk: match by img src filename ──────────────────────────────
  if (isImage && imgSrc) {
    try {
      const filename = imgSrc.split('/').pop().split('?')[0];
      if (filename && filename.length > 3) {
        const img = document.querySelector(`img[src*="${CSS.escape(filename)}"]`);
        if (img) return img;
      }
      const img = document.querySelector(`img[src="${CSS.escape(imgSrc)}"]`);
      if (img) return img;
    } catch (_) {}
    return null;
  }

  // ── Text chunk ───────────────────────────────────────────────────────────
  // Normalise: collapse all whitespace to single space, lowercase.
  const norm = t => t.replace(/\s+/g, ' ').toLowerCase().trim();

  // isLeaf: element has no block-level children (avoids matching containers)
  const isLeaf = el => {
    const blockTags = new Set(['P','H1','H2','H3','H4','H5','H6','LI','TD','TH',
                                'BLOCKQUOTE','PRE','CODE','ARTICLE','SECTION','DIV']);
    for (const child of el.children) {
      if (blockTags.has(child.tagName)) return false;
    }
    return true;
  };

  // isTooLarge: element covers more than 60% of viewport height → skip
  const isTooLarge = el => {
    const r = el.getBoundingClientRect();
    return r.height > window.innerHeight * 0.6;
  };

  const candidates = Array.from(document.querySelectorAll(SEARCH_SELECTORS));

  function findMatch(needle, requireLeaf) {
    if (!needle || needle.length < 8) return null;
    for (const el of candidates) {
      if (requireLeaf && !isLeaf(el)) continue;
      if (norm(el.textContent).includes(needle)) {
        if (isTooLarge(el)) continue;  // skip containers
        return el;
      }
    }
    return null;
  }

  const fp60  = fingerprint.length >= 12 ? norm(fingerprint.slice(0, 60)) : '';
  const fp30  = fingerprint.length >= 8  ? norm(fingerprint.slice(0, 30)) : '';
  const tx80  = text.length >= 12        ? norm(text.slice(0, 80))        : '';
  const tx40  = text.length >= 8         ? norm(text.slice(0, 40))        : '';

  // Try longest needles first with leaf requirement, then relax
  return findMatch(fp60, true)
      || findMatch(fp30, true)
      || findMatch(tx80, true)
      || findMatch(tx40, true)
      || findMatch(fp60, false)
      || findMatch(fp30, false)
      || null;
}


// ── Apply highlights ──────────────────────────────────────────────────────────

function applyHighlights(chunks) {
  if (!chunks || chunks.length === 0) return;

  injectStyles();
  clearHighlights();

  let highlightCount = 0;
  let firstEl       = null;

  chunks.forEach((chunk, rank) => {
    try {
      const el = resolveElement(chunk);
      if (!el) return;

      const color = chunk.highlight_color
        || CONTENT_COLORS[chunk.content_type]
        || CONTENT_COLORS.text;

      el.classList.add(HIGHLIGHT_CLASS, 'rag-hl-new');
      el.style.setProperty('--rag-color', color);
      el.dataset.ragRank = `#${rank + 1}`;

      // Remove pulse class after animation
      el.addEventListener(
        'animationend',
        () => el.classList.remove('rag-hl-new'),
        { once: true }
      );

      if (!firstEl) firstEl = el;
      highlightCount += 1;
    } catch (_) {}
  });

  if (firstEl) {
    setTimeout(() => {
      firstEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 300);
  }

  console.log(`[RAG] Highlighted ${highlightCount} / ${chunks.length} chunks`);
}


// ── Event listeners ───────────────────────────────────────────────────────────

// From background.js via executeScript
window.addEventListener('RAG_HIGHLIGHT', (e) => {
  applyHighlights(e.detail);
});

// Handle chunks that were set before this listener was registered
if (window.__rag_pending_chunks) {
  applyHighlights(window.__rag_pending_chunks);
  window.__rag_pending_chunks = null;
}
