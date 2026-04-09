/**
 * background.js — Browser RAG service worker
 *
 * Handles:
 *   1. Opening the main extension UI as a standalone window.
 *   2. Opening web pages and injecting highlight instructions via content.js.
 *   3. Opening the PDF viewer for local / file:// PDFs with chunk highlights.
 */

'use strict';

const APP_URL    = chrome.runtime.getURL('html/popup.html');
const WIN_ID_KEY = 'bhi_window_id';
const WIN_W      = 1100;
const WIN_H      = 840;


// ── Open / focus extension window on toolbar click ────────────────────────────

chrome.action.onClicked.addListener(async () => {
  const stored     = await chrome.storage.local.get(WIN_ID_KEY);
  const existingId = stored[WIN_ID_KEY];

  if (existingId !== undefined) {
    try {
      const win = await chrome.windows.get(existingId);
      if (win) {
        await chrome.windows.update(existingId, { focused: true });
        return;
      }
    } catch (_) {
      await chrome.storage.local.remove(WIN_ID_KEY);
    }
  }

  await openWindow();
});

async function openWindow() {
  try {
    const win = await chrome.windows.create({
      url:     APP_URL,
      type:    'normal',
      width:   WIN_W,
      height:  WIN_H,
      focused: true,
    });
    await chrome.storage.local.set({ [WIN_ID_KEY]: win.id });
  } catch (err) {
    console.warn('[RAG] window.create failed, opening as tab:', err);
    try {
      await chrome.tabs.create({ url: APP_URL, active: true });
    } catch (tabErr) {
      console.error('[RAG] tab fallback also failed:', tabErr);
    }
  }
}

chrome.windows.onRemoved.addListener(async (windowId) => {
  const stored = await chrome.storage.local.get(WIN_ID_KEY);
  if (stored[WIN_ID_KEY] === windowId) {
    await chrome.storage.local.remove(WIN_ID_KEY);
  }
});


// ── Message router ────────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === 'OPEN_AND_HIGHLIGHT') {
    // docType ("pdf" | "html") explicitly set by popup.js.
    handleOpenAndHighlight(msg.url, msg.chunks, msg.sourceType, msg.docType);
    sendResponse({ ok: true });
  } else if (msg.type === 'OPEN_ALL_SOURCES') {
    // Open every retrieved source in its own tab simultaneously.
    openAllSources(msg.sources).catch(err =>
      console.error('[RAG] openAllSources error:', err)
    );
    sendResponse({ ok: true });
  }
  return true;
});


// ── HTML page highlight ───────────────────────────────────────────────────────

async function handleOpenAndHighlight(url, chunks, sourceType, docType) {
  const isPdf    = docType === 'pdf'    || sourceType === 'pdf';
  const isClaude = docType === 'claude' || sourceType === 'claude'
                   || (url && url.startsWith('claude://'));

  if (isPdf) {
    const filePath = chunks?.[0]?.file_path || '';
    openPdfViewer(filePath, chunks);
  } else if (isClaude) {
    openClaudeViewer(chunks);
  } else {
    const webUrl = chunks?.[0]?.web_url || url;
    openAndHighlightHtml(webUrl, chunks);
  }
}

/**
 * openAllSources — open every retrieved source in its own tab simultaneously.
 *
 * Called by popup.js when the user wants to see ALL sources at once.
 * Groups chunks by source key (file_path for PDFs, url for HTML/Claude),
 * then opens each group in a parallel new tab with appropriate highlights.
 *
 * @param {Array} sources — the sources array from /query response
 */
async function openAllSources(sources) {
  if (!sources || sources.length === 0) return;

  const openPromises = sources.map(source => {
    const docType    = source.doc_type || (source.source_type === 'pdf' ? 'pdf' : 'html');
    const isPdf      = docType === 'pdf';
    const isClaude   = source.source_type === 'claude'
                       || (source.url && source.url.startsWith('claude://'));
    const allChunks  = source.chunks || [];

    if (isPdf) {
      const filePath = allChunks[0]?.file_path || source.file_path || '';
      if (!filePath) return Promise.resolve();
      return openPdfViewer(filePath, allChunks);
    } else if (isClaude) {
      if (!allChunks.length) return Promise.resolve();
      return openClaudeViewer(allChunks);
    } else {
      // HTML — use web_url from first chunk, fall back to source.url
      const webUrl = allChunks[0]?.web_url || source.url || '';
      if (!webUrl) return Promise.resolve();
      return openAndHighlightHtml(webUrl, allChunks);
    }
  });

  await Promise.allSettled(openPromises);
}

async function openAndHighlightHtml(url, chunks) {
  try {
    const tabs = await chrome.tabs.query({ url: url + '*' });
    let tab;

    if (tabs.length > 0) {
      tab = tabs[0];
      await chrome.tabs.update(tab.id, { active: true });
      await chrome.windows.update(tab.windowId, { focused: true });
    } else {
      tab = await chrome.tabs.create({ url });
    }

    const inject = (tabId) => {
      chrome.scripting.executeScript({
        target: { tabId },
        func: (chunkData) => {
          // Guard against double injection
          if (window.__rag_highlight_injected) return;
          window.__rag_highlight_injected = true;
          window.__rag_pending_chunks = chunkData;
          window.dispatchEvent(
            new CustomEvent('RAG_HIGHLIGHT', { detail: chunkData })
          );
        },
        args: [chunks],
      }).catch((err) => {
        console.warn('[RAG] script injection failed:', err.message);
      });
    };

    if (tab.status === 'complete') {
      inject(tab.id);
    } else {
      const listener = (tabId, info) => {
        if (tabId === tab.id && info.status === 'complete') {
          chrome.tabs.onUpdated.removeListener(listener);
          inject(tabId);
        }
      };
      chrome.tabs.onUpdated.addListener(listener);
    }
  } catch (err) {
    console.error('[RAG] openAndHighlightHtml error:', err);
  }
}


// ── PDF viewer ────────────────────────────────────────────────────────────────

function openPdfViewer(filePath, chunks) {
  if (!filePath) {
    console.error('[RAG] openPdfViewer: no file_path in chunks');
    return Promise.resolve();
  }

  // Strip chunks to only what the viewer needs — keeps URL short
  const slim = (chunks || []).map(c => ({
    chunk_id:         c.chunk_id || '',
    content_type:     c.content_type || 'text',
    text:             (c.text || '').slice(0, 120),
    similarity_score: c.similarity_score || c.score || 0,
    page_number:      c.page_number || c.page || 0,
    bbox:             c.bbox || null,
    is_scanned:       c.is_scanned || false,
    ocr_failed:       c.ocr_failed || false,
    text_fingerprint: c.text_fingerprint || '',
    img_alt:          c.img_alt || '',
  }));

  const viewerBase = chrome.runtime.getURL('viewer/pdf_viewer.html');
  const encoded    = encodeURIComponent(JSON.stringify(slim));
  const viewerUrl  = `${viewerBase}?path=${encodeURIComponent(filePath)}&data=${encoded}`;

  return chrome.tabs.create({ url: viewerUrl, active: true }).catch((err) => {
    console.error('[RAG] PDF viewer open failed:', err);
  });
}


// ── Claude conversation viewer ────────────────────────────────────────────────
// Claude source chunks have url = "claude://conversation/<uuid>".
// There is no real page to open — instead we open the in-extension Claude
// viewer which renders the Q+A pairs with highlights.

function openClaudeViewer(chunks) {
  if (!chunks || chunks.length === 0) return Promise.resolve();

  const slim = (chunks || []).map(c => ({
    chunk_id:        c.chunk_id || '',
    content_type:    c.content_type || 'text',
    text:            c.text || '',
    score:           c.similarity_score || c.score || 0,
    conv_name:       c.conv_name || '',
    conversation_id: c.conversation_id || '',
    turn_index:      c.turn_index || 0,
    timestamp:       c.timestamp || '',
  }));

  const viewerBase = chrome.runtime.getURL('viewer/claude_viewer.html');
  const encoded    = encodeURIComponent(JSON.stringify(slim));
  const viewerUrl  = `${viewerBase}?data=${encoded}`;

  return chrome.tabs.create({ url: viewerUrl, active: true }).catch((err) => {
    console.error('[RAG] Claude viewer open failed:', err);
  });
}
