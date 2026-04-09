/**
 * popup.js — Browser Intelligence
 *
 * Runs inside the persistent detached window (not a popup).
 * The window is created by background.js via chrome.windows.create()
 * so it never closes when the user switches tabs or applications.
 *
 * Key design decisions:
 *  - All event listeners attached via addEventListener (MV3 CSP — no inline handlers)
 *  - Checkbox cards: JS exclusively controls checked state (no double-toggle from native label)
 *  - Progress section shown via element.style.display = 'block' (inline style wins over CSS)
 *  - Every fetch catch() logs to console.error AND shows visible UI feedback
 *  - nomic search_document: prefix is added by the backend — frontend just sends plain text
 *  - Claude JSON path field accepts both local paths and http/https URLs
 */

'use strict';

const BACKEND_URL = 'http://localhost:8000';
const POLL_MS     = 2000;

// ── Init ──────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  setupSidebarNav();
  setupCheckCards();
  setupIndexPanel();
  setupSearchPanel();
  setupSourceHandlers();
  setupPDFPanel();
  setupClaudePanel();
  setupImportModal();
  setupStalenessHandlers();
  setupErrorLogToggle();
  setupDashboard();
  setupTestPanel();

  // Async initialisation — run in parallel, failures are non-fatal
  await Promise.allSettled([
    checkBackendHealth(),
    detectClaudeExport(),
    loadDashboardStats(),
  ]);
});


// ── Sidebar navigation ────────────────────────────────────────────────────────

function setupSidebarNav() {
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
      const panelId = item.dataset.panel;
      if (!panelId) return;

      document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
      item.classList.add('active');

      document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
      const panel = document.getElementById(`panel-${panelId}`);
      if (panel) panel.classList.add('active');

      // Lazy-load data when switching panels
      if (panelId === 'dashboard') loadDashboardStats();
      if (panelId === 'claude') detectClaudeExport();
      if (panelId === 'logs') loadRecentRuns();
    });
  });
}

function switchToPanel(panelId) {
  const navItem = document.querySelector(`.nav-item[data-panel="${panelId}"]`);
  if (navItem) navItem.click();
}


// ── Checkbox cards (FIX: JS is sole toggle controller, no double-toggle) ──────

function setupCheckCards() {
  document.querySelectorAll('.check-card').forEach(card => {
    card.addEventListener('click', () => {
      const cb = document.getElementById(card.dataset.target);
      if (!cb) return;
      cb.checked = !cb.checked;
      card.classList.toggle('checked', cb.checked);
    });
  });
}


// ── Backend health ────────────────────────────────────────────────────────────

async function checkBackendHealth() {
  const dot     = document.getElementById('health-dot');
  const offline = document.getElementById('offline-banner');
  try {
    const health = await apiFetch('/health');
    const ok = health.ok === true;

    dot.textContent = '●';
    dot.style.color = ok ? 'var(--accent)' : 'var(--warning)';
    offline.style.display = 'none';

    // Parse checks array: [{name, ok, detail}]
    const checks = Array.isArray(health.checks) ? health.checks : [];
    const qdrant  = checks.find(c => c.name === 'qdrant');
    const bm25    = checks.find(c => c.name === 'bm25');

    // Extract point count from qdrant detail string e.g. "points=54"
    const pointsMatch = qdrant?.detail?.match(/points=(\d+)/);
    const points = pointsMatch ? parseInt(pointsMatch[1]) : 0;

    dot.title = ok ? `Healthy — ${points.toLocaleString()} chunks indexed` : 'Degraded — check /health';
    document.getElementById('vector-count').textContent = `${points.toLocaleString()} chunks`;

    setDashStat('dash-vectors', points.toLocaleString());
    setDashStat('dash-status',  ok ? '✅ OK' : '⚠ Degraded');
    setDashStat('dash-device',  bm25?.detail || '—');

    // Health details panel
    const hd = document.getElementById('health-details');
    if (hd) {
      hd.innerHTML = checks
        .map(c => `${c.name}: ${c.ok ? '✅' : '❌'} ${esc(c.detail || '')}`)
        .join('<br>');
    }

    return health;
  } catch (err) {
    console.error('[RAG] Health check failed:', err);
    dot.textContent = '●';
    dot.style.color = 'var(--danger)';
    dot.title = `Backend offline: ${err.message}`;
    offline.style.display = 'flex';
    setDashStat('dash-status', '❌ Offline');
    return null;
  }
}

function setDashStat(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}


// ── Dashboard ─────────────────────────────────────────────────────────────────

function setupDashboard() {
  document.getElementById('btn-refresh-dash')?.addEventListener('click', () => {
    checkBackendHealth();
    loadRecentRuns();
    detectClaudeExport();
  });
  document.getElementById('btn-quick-index')?.addEventListener('click', () => {
    switchToPanel('index');
    // Small delay so panel renders before we start
    setTimeout(startIndexAll, 100);
  });
  document.getElementById('btn-quick-search')?.addEventListener('click', () => {
    switchToPanel('search');
    document.getElementById('search-input')?.focus();
  });
}

async function loadDashboardStats() {
  await checkBackendHealth();
  await loadRecentRuns();
}

async function loadRecentRuns() {
  const container = document.getElementById('recent-runs-list');
  if (!container) return;
  // Run history is in-memory on the server — not persisted across restarts.
  // Show the health summary instead.
  try {
    const health = await apiFetch('/health');
    const checks = Array.isArray(health.checks) ? health.checks : [];
    container.innerHTML = checks
      .map(c => `<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--border);font-size:12px">
        <span style="color:var(--muted)">${esc(c.name)}</span>
        <span style="color:${c.ok ? 'var(--accent)' : 'var(--danger)'};">${c.ok ? '✅' : '❌'}</span>
        <span style="color:var(--muted);font-family:'IBM Plex Mono',monospace;font-size:11px;margin-left:8px">${esc(c.detail||'')}</span>
      </div>`).join('') || '<span class="text-muted">No health data.</span>';
  } catch (err) {
    container.innerHTML = '<span class="text-muted">Backend offline.</span>';
  }
}


// ── Claude auto-detection ─────────────────────────────────────────────────────

async function detectClaudeExport() {
  // Claude auto-detection not implemented in this version.
  const statusEl = document.getElementById('claude-detect-status');
  const pathEl   = document.getElementById('claude-detect-path');
  if (statusEl) statusEl.innerHTML = '<span style="color:var(--muted)">not supported</span>';
  if (pathEl)   pathEl.textContent = 'Use the custom path field below.';
}


// ── INDEX PANEL ───────────────────────────────────────────────────────────────

function setupIndexPanel() {
  document.getElementById('btn-index-all')?.addEventListener('click', startIndexAll);
}

async function startIndexAll() {
  // Read checkbox states — only checked sources will be indexed
  const indexWeb    = document.getElementById('chk-web')?.checked    ?? false;
  const indexPdfs   = document.getElementById('chk-pdfs')?.checked   ?? false;
  const indexClaude = document.getElementById('chk-claude')?.checked ?? false;

  if (!indexWeb && !indexPdfs && !indexClaude) {
    showToast('Select at least one source to index.', 'warning');
    return;
  }

  // ── Read settings for each source ────────────────────────────────────────
  // PDF
  const pdfFolder   = document.getElementById('pdf-folder-input')?.value.trim() || null;
  const pdfMaxFiles = parseInt(document.getElementById('pdf-max-files')?.value  || '10', 10) || 10;
  const pdfMaxPages = parseInt(document.getElementById('pdf-max-pages')?.value  || '50', 10) || 50;

  // Web history
  const browser     = document.getElementById('browser-select')?.value || 'brave';
  const webMaxUrls  = parseInt(document.getElementById('web-max-urls')?.value   || '10',  10) || 10;

  // Claude
  const claudePath  = document.getElementById('claude-path-input')?.value.trim() || null;
  const claudeMaxC  = parseInt(document.getElementById('claude-max-convs')?.value || '0', 10) || null;

  // ── Guards: required paths ────────────────────────────────────────────────
  if (indexPdfs && !pdfFolder) {
    showToast('PDFs checked but no folder path provided.', 'warning');
    return;
  }
  if (indexClaude && !claudePath) {
    showToast('Claude chats checked but no conversations.json path provided.', 'warning');
    return;
  }

  const btn = document.getElementById('btn-index-all');
  btn.disabled = true;
  btn.textContent = '⏳ Starting…';
  showProgressPanel();

  try {
    // Build body — only include fields for checked sources.
    // Limits are ALWAYS sent for checked sources so backend enforces them.
    const body = {
      index_web:    indexWeb,
      index_pdfs:   indexPdfs,
      index_claude: indexClaude,
    };

    if (indexPdfs) {
      body.pdf_folder   = pdfFolder;
      body.max_files    = pdfMaxFiles;    // null = all files
      body.max_pages    = pdfMaxPages;    // default 10
    }
    if (indexWeb) {
      body.browser       = browser;
      body.history_limit = webMaxUrls;    // default 200
    }
    if (indexClaude) {
      body.claude_json_path  = claudePath;
      body.max_conversations = claudeMaxC; // null = all conversations
    }

    console.log('[RAG] POST /index/start');
    const result = await apiFetch('/index/start', { method: 'POST', body: JSON.stringify(body) });
    console.log('[RAG] /index/start response:', result);

    document.getElementById('progress-label').textContent = 'Indexing started…';
    pollRun(result.run_id);
  } catch (err) {
    console.error('[RAG] /index/start failed:', err);
    btn.disabled = false;
    btn.textContent = '⚡ Index Everything';
    document.getElementById('progress-label').textContent = `❌ Failed to start: ${err.message}`;
    document.getElementById('progress-bar').style.background = 'var(--danger)';
    showToast(`Failed to start indexing: ${err.message}`, 'error');
  }
}

// FIX: inline style wins over CSS display:none
function showProgressPanel() {
  document.getElementById('progress-section').style.display = 'block';
  document.getElementById('progress-label').textContent = 'Connecting to backend…';
  document.getElementById('progress-bar').style.width = '0%';
  document.getElementById('progress-bar').style.background = 'linear-gradient(90deg, var(--accent2), var(--accent))';
  document.getElementById('progress-count').textContent = '0 / —';
  document.getElementById('progress-chunks').textContent = '0 chunks';
  document.getElementById('progress-ready').style.display = 'none';
  document.getElementById('error-log').style.display = 'none';
  document.getElementById('error-list').innerHTML = '';
  document.getElementById('error-count-badge').textContent = '0';
  // Reset step cards
  ['web', 'pdf', 'claude'].forEach(s => {
    const card = document.getElementById(`step-${s}`);
    if (card) {
      card.className = 'step-card';
      document.getElementById(`step-${s}-stat`).textContent = '—';
      document.getElementById(`step-${s}-pct`).textContent  = '';
      document.getElementById(`step-${s}-bar`).style.width  = '0';
    }
  });
}


// ── Run polling ───────────────────────────────────────────────────────────────

let _errorCount   = 0;
let _errorsShown  = false;
let _pollInterval = null;

function setupErrorLogToggle() {
  document.getElementById('error-log-header')?.addEventListener('click', toggleErrors);
  document.getElementById('btn-toggle-errors')?.addEventListener('click', e => {
    e.stopPropagation();
    toggleErrors();
  });
}

function toggleErrors() {
  _errorsShown = !_errorsShown;
  document.getElementById('error-list').style.display = _errorsShown ? 'block' : 'none';
  document.getElementById('btn-toggle-errors').textContent = _errorsShown ? 'Hide' : 'Show';
}

function pollRun(runId) {
  _errorCount  = 0;
  _errorsShown = false;
  if (_pollInterval) clearInterval(_pollInterval);

  _pollInterval = setInterval(async () => {
    try {
      const status = await apiFetch(`/index/status/${runId}`);
      updateProgress(status);
      if (['completed', 'failed', 'partial', 'cancelled'].includes(status.status)) {
        clearInterval(_pollInterval);
        _pollInterval = null;
        finalizeProgress(status);
      }
    } catch (err) {
      console.error('[BHI] pollRun error:', err);
    }
  }, POLL_MS);

  document.getElementById('btn-cancel').onclick = () => {
    if (_pollInterval) { clearInterval(_pollInterval); _pollInterval = null; }
    document.getElementById('progress-label').textContent = 'Cancelled.';
    resetIndexButton();
  };
}

function updateProgress(status) {
  // New IndexStatus fields: total_urls, fetched, parsed, chunked, stored, failed, skipped
  const total = status.total_urls || 1;
  const done  = (status.fetched || 0);
  const pct   = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;

  document.getElementById('progress-bar').style.width = `${pct}%`;
  document.getElementById('progress-count').textContent = `${done.toLocaleString()} / ${total.toLocaleString()} fetched`;
  document.getElementById('progress-chunks').textContent = `${(status.stored || 0).toLocaleString()} chunks stored`;

  // Per-step updates from live Redis progress
  const prog = status.progress;
  if (prog?.steps) {
    Object.entries(prog.steps).forEach(([key, data]) => updateStepCard(key, data));
    if (prog.current_step && prog.current_step !== 'done') {
      const stepLabels = { web: 'Indexing web history', pdf: 'Indexing PDFs', claude: 'Indexing Claude chats' };
      const stepData = prog.steps[prog.current_step] || {};
      let labelText = stepLabels[prog.current_step] || 'Indexing';
      if (stepData.current_item) {
        labelText += ` — ${stepData.current_item}`;
        if (stepData.stage) labelText += ` (${stepData.stage})`;
      } else {
        labelText += '…';
      }
      document.getElementById('progress-label').textContent = labelText;
    }
  } else if (status.step_details) {
    // Fallback when Redis unavailable
    Object.entries(status.step_details).forEach(([key, d]) => {
      updateStepCard(key, {
        status: d.status, success: d.success, failed: d.failed,
        skipped: d.skipped || 0, total: d.total, chunks: d.chunks,
        pct: d.total > 0 ? Math.round((d.success + d.failed) / d.total * 100) : 0,
      });
    });
  }

  // Append new errors — status.errors is List[str]
  const errors = status.errors || [];
  if (errors.length > _errorCount) {
    const newErrors = errors.slice(_errorCount);
    _errorCount = errors.length;
    const list = document.getElementById('error-list');
    newErrors.forEach(errStr => {
      const row = document.createElement('div');
      row.className = 'error-row';
      row.innerHTML = `<span class="error-msg" style="font-size:11px;color:var(--muted)">${esc(errStr)}</span>`;
      list.appendChild(row);
    });
    document.getElementById('error-count-badge').textContent = _errorCount;
    document.getElementById('error-log').style.display = 'block';
  }

  if ((status.stored || 0) > 0) {
    document.getElementById('progress-ready').style.display = 'flex';
    document.getElementById('progress-ready-count').textContent = (status.stored || 0).toLocaleString();
  }
}

function updateStepCard(stepKey, data) {
  const key    = stepKey.replace(/_run_id$/, '');
  const card   = document.getElementById(`step-${key}`);
  const statEl = document.getElementById(`step-${key}-stat`);
  const pctEl  = document.getElementById(`step-${key}-pct`);
  const barEl  = document.getElementById(`step-${key}-bar`);
  const feedEl = document.getElementById(`step-${key}-feed`);
  if (!card || !statEl) return;

  const cls = { running: 'step--active', pending: 'step--active', completed: 'step--done', partial: 'step--done', failed: 'step--error', skipped: 'step--skip' }[data.status] || '';
  card.className = `step-card ${cls}`;

  const s = data.success || 0, f = data.failed || 0, t = data.total || 0, c = data.chunks || 0;
  if (t > 0) {
    statEl.textContent = `${s}/${t}`;
    pctEl.textContent  = `${c} chunks`;
  } else if (data.status === 'skipped') {
    statEl.textContent = 'skipped';
    pctEl.textContent  = (data.error || '').slice(0, 28);
  } else {
    statEl.textContent = data.status === 'running' ? '…' : '—';
  }
  if (barEl) {
    const p = data.pct ?? (t > 0 ? Math.round((s + f) / t * 100) : 0);
    barEl.style.width = `${Math.min(p, 100)}%`;
  }
  // Live text feed — show current item + stage when actively running
  if (feedEl) {
    if (data.status === 'running' && data.current_item) {
      const stage = data.stage ? ` — ${data.stage}` : '';
      feedEl.textContent = data.current_item + stage;
      feedEl.classList.add('step-feed--active');
    } else if (data.status === 'completed' || data.status === 'partial') {
      feedEl.textContent = `✓ ${s} done, ${c} chunks`;
      feedEl.classList.remove('step-feed--active');
    } else if (data.status === 'failed') {
      feedEl.textContent = (data.error || 'failed').slice(0, 48);
      feedEl.classList.remove('step-feed--active');
    } else {
      feedEl.textContent = '';
      feedEl.classList.remove('step-feed--active');
    }
  }
}

function finalizeProgress(status) {
  const chunks = (status.stored || 0).toLocaleString();
  if (status.status === 'completed') {
    document.getElementById('progress-label').textContent = `✅ Done — ${chunks} chunks stored`;
    document.getElementById('progress-bar').style.width = '100%';
    document.getElementById('progress-bar').style.background = 'var(--accent)';
  } else if (status.status === 'partial') {
    document.getElementById('progress-label').textContent = `⚠ Partial — ${chunks} chunks, ${status.failed || 0} errors`;
    document.getElementById('progress-bar').style.width = '100%';
    document.getElementById('progress-bar').style.background = 'var(--warning)';
  } else {
    document.getElementById('progress-label').textContent = `❌ Failed — ${status.failed || 0} errors`;
    document.getElementById('progress-bar').style.background = 'var(--danger)';
  }
  resetIndexButton();
  checkBackendHealth();
  loadRecentRuns();
}

function resetIndexButton() {
  const btn = document.getElementById('btn-index-all');
  btn.disabled = false;
  btn.textContent = '⚡ Index Everything';
}


// ── SEARCH PANEL ──────────────────────────────────────────────────────────────

function setupSearchPanel() {
  // Sub-tabs
  document.getElementById('search-tab-text')?.addEventListener('click', () => {
    document.getElementById('search-tab-text').classList.add('active');
    document.getElementById('search-tab-image').classList.remove('active');
    document.getElementById('text-search-panel').style.display  = 'block';
    document.getElementById('image-search-panel').style.display = 'none';
  });
  document.getElementById('search-tab-image')?.addEventListener('click', () => {
    document.getElementById('search-tab-image').classList.add('active');
    document.getElementById('search-tab-text').classList.remove('active');
    document.getElementById('image-search-panel').style.display = 'block';
    document.getElementById('text-search-panel').style.display  = 'none';
  });

  document.getElementById('btn-search')?.addEventListener('click', runTextSearch);
  document.getElementById('search-input')?.addEventListener('keydown', e => {
    if (e.key === 'Enter') runTextSearch();
  });

  // Example query chips
  document.querySelectorAll('.example-query').forEach(btn => {
    btn.addEventListener('click', () => {
      document.getElementById('search-input').value = btn.textContent;
      runTextSearch();
    });
  });

  // Image search
  const dropZone  = document.getElementById('image-drop-zone');
  const fileInput = document.getElementById('image-file-input');
  dropZone?.addEventListener('click', () => fileInput.click());
  dropZone?.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone?.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone?.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) handleImageFile(e.dataTransfer.files[0]);
  });
  fileInput?.addEventListener('change', () => { if (fileInput.files[0]) handleImageFile(fileInput.files[0]); });
  document.getElementById('btn-image-search')?.addEventListener('click', runImageSearch);
}

async function runTextSearch() {
  const query = document.getElementById('search-input').value.trim();
  if (!query) return;
  showResults({ loading: '🔍 Searching…' });
  const t0 = performance.now();
  try {
    const result = await apiFetch('/query', {
      method: 'POST',
      body: JSON.stringify({ text: query, top_k: 5, top_urls: 3 }),
    });
    const timingMs = Math.round(performance.now() - t0);
    showResults({
      sources:    result.sources,
      answer:     result.answer,
      agentSteps: result.agent_steps,
      timingMs,
    });
  } catch (err) {
    console.error('[RAG] search failed:', err);
    showResults({ error: `❌ Search error: ${err.message}` });
  }
}

let _imageBase64 = null;

function handleImageFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    _imageBase64 = e.target.result.split(',')[1];
    const preview = document.getElementById('image-preview');
    preview.src = e.target.result;
    preview.style.display = 'block';
    document.getElementById('btn-image-search').style.display = 'block';
  };
  reader.readAsDataURL(file);
}

async function runImageSearch() {
  if (!_imageBase64) return;
  showResults({ loading: '🔍 Searching by image…' });
  showResults({ error: '❌ Image search not implemented — use text search.' });
}

function showResults({ loading, error, sources, answer, agentSteps, timingMs }) {
  const section = document.getElementById('results-section');
  section.style.display = 'block';

  if (loading || error) {
    document.getElementById('results-answer').textContent = loading || error;
    document.getElementById('results-answer').style.display = 'block';
    document.getElementById('results-list').innerHTML = '';
    document.getElementById('agent-steps').style.display = 'none';
    hideAnswerTab();
    return;
  }

  // ── Sources tab ──────────────────────────────────────────────────────────
  const listEl = document.getElementById('results-list');
  listEl.innerHTML = '';
  (sources || []).forEach(s => listEl.appendChild(renderSourceCard(s)));

  // ── "Open All Sources" button — opens every source in its own tab ────────
  const existingOpenAll = document.getElementById('btn-open-all-sources');
  if (existingOpenAll) existingOpenAll.remove();
  if (sources && sources.length > 0) {
    const openAllBtn = document.createElement('button');
    openAllBtn.id        = 'btn-open-all-sources';
    openAllBtn.className = 'btn btn--outline btn--small';
    openAllBtn.style.cssText = 'margin:8px 0 4px 0;width:100%;';
    openAllBtn.textContent  = `↗ Open All ${sources.length} Source${sources.length > 1 ? 's' : ''} in New Tabs`;
    openAllBtn.addEventListener('click', () => {
      chrome.runtime.sendMessage({ type: 'OPEN_ALL_SOURCES', sources });
    });
    listEl.after(openAllBtn);
  }

  // Hide legacy answer element (now in Answer tab)
  document.getElementById('results-answer').style.display = 'none';

  if (agentSteps?.length) {
    document.getElementById('agent-steps').textContent = agentSteps.join(' → ');
    document.getElementById('agent-steps').style.display = 'block';
  } else {
    document.getElementById('agent-steps').style.display = 'none';
  }

  // ── Answer tab ───────────────────────────────────────────────────────────
  if (answer) {
    showAnswerTab(answer, sources || [], agentSteps || [], timingMs);
  } else {
    hideAnswerTab();
  }
}

function showAnswerTab(answer, sources, agentSteps, timingMs) {
  // Ensure Answer tab button exists
  ensureAnswerTabButton();

  const tabEl = document.getElementById('answer-tab-content');
  if (!tabEl) return;

  const sourceLinks = sources.slice(0, 5).map((s, i) =>
    `<a class="answer-source-link" href="#" data-url="${esc(s.url)}" data-idx="${i}">
      [${i + 1}] ${esc(shortUrl(s.url))}
    </a>`
  ).join('');

  const timing = timingMs ? `<span class="answer-timing">${timingMs}ms</span>` : '';

  tabEl.innerHTML = `
    <div class="answer-header">
      <span class="answer-label">🤖 LLM Answer ${timing}</span>
      <button class="btn btn--tiny answer-copy-btn" id="answer-copy-btn">Copy</button>
    </div>
    <div class="answer-body" id="answer-body">${esc(answer)}</div>
    ${sourceLinks ? `<div class="answer-sources"><span class="answer-sources-label">Sources used:</span>${sourceLinks}</div>` : ''}
    ${agentSteps.length ? `<div class="answer-steps">${esc(agentSteps.join(' → '))}</div>` : ''}
  `;

  // Copy button
  document.getElementById('answer-copy-btn')?.addEventListener('click', () => {
    navigator.clipboard.writeText(answer).then(() => {
      const btn = document.getElementById('answer-copy-btn');
      const prev = btn.textContent;
      btn.textContent = '✓ Copied';
      setTimeout(() => { btn.textContent = prev; }, 1500);
    });
  });

  // Source links → open + highlight
  tabEl.querySelectorAll('.answer-source-link').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const idx = Number(link.dataset.idx);
      const source = sources[idx];
      if (!source) return;
      // Prefer text chunks for answer tab navigation
      const textChunks = source.text_chunks?.length
        ? source.text_chunks
        : (source.chunks || []).filter(c => !c.content_type?.startsWith('image'));
      const flatChunks = flattenChunks(textChunks.length ? textChunks : source.chunks || []);
      const docType = source.doc_type || (source.source_type === 'pdf' ? 'pdf' : 'html');
      // web_url / file_path come from the chunk payload written at index time
      const routeUrl = docType === 'pdf'
        ? (flatChunks[0]?.file_path || source.file_path || source.url)
        : (flatChunks[0]?.web_url  || source.url);
      chrome.runtime.sendMessage({
        type: 'OPEN_AND_HIGHLIGHT',
        url: routeUrl,
        chunks: flatChunks,
        sourceType: source.source_type,
        docType,
      });
    });
  });

  // Switch to Answer tab automatically on first result
  switchResultTab('answer');
}

function hideAnswerTab() {
  const btn = document.getElementById('result-tab-answer');
  if (btn) btn.style.display = 'none';
}

function ensureAnswerTabButton() {
  if (document.getElementById('result-tab-answer')) {
    document.getElementById('result-tab-answer').style.display = '';
    return;
  }

  // Create tab bar if it doesn't exist
  const section = document.getElementById('results-section');
  if (!section) return;

  // Insert tab bar before the first child
  const tabBar = document.createElement('div');
  tabBar.id = 'result-tabs';
  tabBar.className = 'search-tabs';
  tabBar.style.marginBottom = '10px';
  tabBar.innerHTML = `
    <button class="search-tab" id="result-tab-sources" data-tab="sources">📄 Sources</button>
    <button class="search-tab" id="result-tab-answer"  data-tab="answer">🤖 Answer</button>
  `;
  section.insertBefore(tabBar, section.firstChild);

  // Tab click handlers
  tabBar.querySelectorAll('.search-tab').forEach(btn => {
    btn.addEventListener('click', () => switchResultTab(btn.dataset.tab));
  });

  // Create answer tab content container
  const answerContent = document.createElement('div');
  answerContent.id = 'answer-tab-content';
  answerContent.style.display = 'none';
  section.appendChild(answerContent);
}

function switchResultTab(tab) {
  const sourcesTab = document.getElementById('result-tab-sources');
  const answerTab  = document.getElementById('result-tab-answer');
  const listEl     = document.getElementById('results-list');
  const stepsEl    = document.getElementById('agent-steps');
  const answerEl   = document.getElementById('answer-tab-content');

  if (tab === 'answer') {
    sourcesTab?.classList.remove('active');
    answerTab?.classList.add('active');
    if (listEl)   listEl.style.display = 'none';
    if (stepsEl)  stepsEl.style.display = 'none';
    if (answerEl) answerEl.style.display = 'block';
  } else {
    answerTab?.classList.remove('active');
    sourcesTab?.classList.add('active');
    if (listEl)   listEl.style.display = 'flex';
    if (stepsEl && stepsEl.textContent) stepsEl.style.display = 'block';
    if (answerEl) answerEl.style.display = 'none';
  }
}

function renderSourceCard(source) {
  const docType   = source.doc_type || (source.source_type === 'pdf' ? 'pdf' : 'html');
  const isPdf     = docType === 'pdf';
  const icon      = isPdf ? '📄' : '🌐';
  const score     = source.top_score ? (source.top_score * 100).toFixed(0) + '%' : '?';

  // Separate text and image chunks (use server-provided splits if present)
  const textChunks  = source.text_chunks?.length
    ? source.text_chunks
    : (source.chunks || []).filter(c => !c.content_type?.startsWith('image'));
  const IMAGE_TYPES = new Set(['image', 'image_page', 'image_embedded']);
  const imageChunks = source.image_chunks?.length
    ? source.image_chunks
    : (source.chunks || []).filter(c => IMAGE_TYPES.has(c.content_type));

  const hasImages = imageChunks.length > 0;
  const hasText   = textChunks.length  > 0;

  // Build badge list
  const types  = [...new Set((source.chunks || []).map(c => c.content_type))];
  const badges = types.map(t => `<span class="badge badge--${t}">${t}</span>`).join('');

  // Image thumbnails strip (up to 3, show page number + ocr_failed indicator)
  const imgStrip = hasImages
    ? `<div class="result-img-strip" style="display:flex;gap:4px;margin:6px 0;flex-wrap:wrap">
        ${imageChunks.slice(0, 3).map((ic, idx) => {
          const pageNum  = ic.page_number || ic.page || (idx + 1);
          const ocrFail  = ic.ocr_failed ? ' 👁' : '';
          const label    = ic.content_type === 'image_embedded'
                           ? `🖼 fig p${pageNum}${ocrFail}` : `🖼 p${pageNum}${ocrFail}`;
          const title    = ic.ocr_failed
                           ? `Visual-only match (OCR failed) — page ${pageNum}`
                           : `${ic.content_type} page ${pageNum}`;
          return `<button class="img-thumb-btn btn btn--tiny" data-page="${pageNum}"
                    style="padding:3px 6px;font-size:10px;${ic.ocr_failed ? 'border-color:var(--warning);color:var(--warning);' : ''}"
                    title="${title}">
                    ${label}
                  </button>`;
        }).join('')}
      </div>`
    : '';

  const card = document.createElement('div');
  card.className = 'result-card';
  card.innerHTML = `
    <div class="result-header">
      <span>${icon}</span>
      <span class="result-url" title="${esc(source.url)}">${esc(shortUrl(source.url))}</span>
      <span class="result-score">${score}</span>
    </div>
    <div class="result-snippet">${esc((source.snippet || '').slice(0, 160))}${(source.snippet||'').length > 160 ? '…' : ''}</div>
    ${imgStrip}
    <div style="display:flex;align-items:center;justify-content:space-between;margin-top:8px;flex-wrap:wrap;gap:4px">
      <div class="result-badges">${badges}</div>
      <div style="display:flex;gap:4px">
        ${hasText ? `<button class="btn btn--outline btn--small result-open-text-btn">
          ${isPdf ? '📄 Open Text Chunks ↗' : 'Open + Highlight ↗'}
        </button>` : ''}
        ${hasImages && isPdf ? `<button class="btn btn--outline btn--small result-open-img-btn">
          🖼 Open Page Image ↗
        </button>` : ''}
      </div>
    </div>
  `;

  // ── Text chunks → open and highlight ──────────────────────────────────
  card.querySelector('.result-open-text-btn')?.addEventListener('click', () => {
    const useChunks = textChunks.length ? textChunks : source.chunks || [];
    const flatChunks = flattenChunks(useChunks);
    // web_url comes from the first chunk's payload (set by server at query time)
    const webUrl = flatChunks[0]?.web_url || source.url;
    chrome.runtime.sendMessage({
      type: 'OPEN_AND_HIGHLIGHT',
      url: webUrl,
      chunks: flatChunks,
      sourceType: source.source_type,
      docType,
    });
  });

  // ── Image chunks → open PDF viewer ─────────────────────────────────────
  card.querySelector('.result-open-img-btn')?.addEventListener('click', () => {
    const flatChunks = flattenChunks(imageChunks.length ? imageChunks : source.chunks || []);
    const filePath   = flatChunks[0]?.file_path || source.file_path || '';
    chrome.runtime.sendMessage({
      type: 'OPEN_AND_HIGHLIGHT',
      url: source.url,      // display only
      file_path: filePath,  // actual path for PDF viewer
      chunks: flatChunks,
      sourceType: source.source_type,
      docType: 'pdf',
    });
  });

  // ── Per-page thumbnail buttons ──────────────────────────────────────────
  card.querySelectorAll('.img-thumb-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const pageNum   = Number(btn.dataset.page);
      const pageChunk = imageChunks.find(ic => (ic.page_number || ic.page) === pageNum)
                        || imageChunks[0];
      if (!pageChunk) return;
      chrome.runtime.sendMessage({
        type: 'OPEN_AND_HIGHLIGHT',
        url: source.url,
        chunks: [flattenChunks([pageChunk])[0]],
        sourceType: source.source_type,
        docType: 'pdf',
      });
    });
  });

  return card;
}

// Flatten chunk for content.js (HTML highlight) / pdf_viewer.html (PDF highlight)
// Preserves all routing fields set by the server: web_url, file_path, page_number, bbox
function flattenChunks(chunks) {
  return chunks.map(c => ({
    ...c,
    // HTML highlight fields
    xpath:            c.xpath            || '',
    css_selector:     c.css_selector     || '',
    text_fingerprint: c.text_fingerprint || '',
    // PDF routing fields (from server Qdrant payload)
    web_url:     c.web_url     || '',
    file_path:   c.file_path   || '',
    page_number: c.page_number || c.page || 0,
    bbox:        c.bbox        || null,   // [x0,y0,x1,y1] PDF points TOPLEFT origin
    is_scanned:  c.is_scanned  || false,  // true → Docling OCR page; bbox from Docling
    // Scoring
    score: c.similarity_score || c.score || 0,
  }));
}


// ── WEB HISTORY SOURCES ───────────────────────────────────────────────────────

function setupSourceHandlers() {
  document.querySelectorAll('.source-sync').forEach(btn => {
    btn.addEventListener('click', async () => {
      try {
        const browser = btn.dataset.source || 'brave';
        const syncBody = { index_web: true, browser };
        const result = await apiFetch('/index/start', { method: 'POST', body: JSON.stringify(syncBody) });
        switchToPanel('index');
        showProgressPanel();
        pollRun(result.run_id);
      } catch (err) {
        console.error('[BHI] sync failed:', err);
        showToast(`Sync failed: ${err.message}`, 'error');
      }
    });
  });
}


// ── PDF PANEL ─────────────────────────────────────────────────────────────────

function setupPDFPanel() {
  // Folder indexing
  document.querySelector('.source-folder')?.addEventListener('click', async () => {
    const folder    = document.getElementById('pdf-folder-path')?.value.trim() || '/home/anirudh97/Data/filesnew/files';
    const maxRaw    = document.getElementById('pdf-max-files')?.value.trim();
    const recursive = document.getElementById('pdf-recursive')?.value === 'true';
    const maxFiles  = maxRaw ? parseInt(maxRaw, 10) : null;  // null = all files

    if (!folder) {
      showToast('Enter a PDF folder path first.', 'warning');
      return;
    }
    try {
      const body = { folder_path: folder, recursive };
      if (maxFiles && maxFiles > 0) body.max_files = maxFiles;
      const result = await apiFetch('/index/pdf/folder', {
        method: 'POST',
        body: JSON.stringify(body),
      });
      switchToPanel('index');
      showProgressPanel();
      pollRun(result.run_id);
    } catch (err) {
      console.error('[BHI] PDF folder failed:', err);
      showToast(`PDF folder error: ${err.message}`, 'error');
    }
  });

  // Single PDF upload
  const pdfDrop  = document.getElementById('pdf-drop-zone');
  const pdfInput = document.getElementById('pdf-file-input');
  pdfDrop?.addEventListener('click', () => pdfInput.click());
  pdfDrop?.addEventListener('dragover', e => { e.preventDefault(); pdfDrop.classList.add('drag-over'); });
  pdfDrop?.addEventListener('dragleave', () => pdfDrop.classList.remove('drag-over'));
  pdfDrop?.addEventListener('drop', async e => {
    e.preventDefault();
    pdfDrop.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) await uploadPDF(e.dataTransfer.files[0]);
  });
  pdfInput?.addEventListener('change', async () => {
    if (pdfInput.files[0]) await uploadPDF(pdfInput.files[0]);
  });
}

async function uploadPDF(file) {
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    showToast('Only PDF files are supported.', 'warning');
    return;
  }
  // File upload not supported — use the folder path field instead.
  showToast('Use "Index PDF Folder" with the folder path. File upload is not supported.', 'warning');
}


// ── CLAUDE PANEL ──────────────────────────────────────────────────────────────

function setupClaudePanel() {
  // Auto-detect button
  document.querySelector('.source-claude-auto')?.addEventListener('click', async () => {
    showToast('Claude chat indexing is not implemented in this version.', 'warning');
  });

  // Custom path or URL (accepts both local paths and http/https URLs)
  document.getElementById('btn-claude-custom')?.addEventListener('click', async () => {
    const pathOrUrl = document.getElementById('claude-custom-path')?.value.trim();
    if (!pathOrUrl) {
      showToast('Enter a path or URL to conversations.json', 'warning');
      return;
    }
    showToast('Claude chat indexing is not implemented in this version.', 'warning');
  });

  // ZIP import buttons
  document.querySelectorAll('.source-import').forEach(btn => {
    btn.addEventListener('click', () => openImportModal(btn.dataset.source));
  });
}


// ── IMPORT MODAL ──────────────────────────────────────────────────────────────

const IMPORT_STEPS = {
  chatgpt: [
    '1. Open <a href="https://chatgpt.com" target="_blank">ChatGPT ↗</a>',
    '2. Settings → Data Controls → Export Data → Confirm',
    '3. Wait for email link (~2 min), download the ZIP',
    '4. Drop the ZIP below:',
  ],
  claude: [
    '1. Open <a href="https://claude.ai" target="_blank">Claude.ai ↗</a>',
    '2. Settings → Privacy → Export Data',
    '3. Download the ZIP immediately',
    '4. Drop the ZIP below:',
  ],
};

let _importSource = null;

function setupImportModal() {
  document.getElementById('btn-close-modal')?.addEventListener('click', closeImportModal);
  const fileInput = document.getElementById('import-file-input');
  const dropZone  = document.getElementById('import-drop-zone');
  dropZone?.addEventListener('click', () => fileInput.click());
  dropZone?.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone?.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone?.addEventListener('drop', async e => {
    e.preventDefault(); dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) await uploadExport(e.dataTransfer.files[0]);
  });
  fileInput?.addEventListener('change', async () => {
    if (fileInput.files[0]) await uploadExport(fileInput.files[0]);
  });
}

function openImportModal(source) {
  _importSource = source;
  document.getElementById('import-modal-title').textContent = source === 'chatgpt' ? 'Import ChatGPT Chats' : 'Import Claude Chats';
  document.getElementById('import-steps').innerHTML = IMPORT_STEPS[source].map(s => `<p class="import-step">${s}</p>`).join('');
  document.getElementById('import-modal').classList.add('open');
}

function closeImportModal() {
  document.getElementById('import-modal').classList.remove('open');
  _importSource = null;
}

async function uploadExport(file) {
  showToast('Chat export import is not implemented in this version.', 'warning');
  closeImportModal();
}


// ── STALENESS ─────────────────────────────────────────────────────────────────

function setupStalenessHandlers() {
  document.getElementById('btn-reindex-now')?.addEventListener('click', () => {
    document.getElementById('stale-banner').style.display = 'none';
    startIndexAll();
  });
  document.getElementById('btn-stale-later')?.addEventListener('click', async () => {
    await setLocal('bhi_stale_snoozed_until', Date.now() + 7 * 24 * 3600 * 1000);
    document.getElementById('stale-banner').style.display = 'none';
  });
  document.getElementById('btn-stale-dismiss')?.addEventListener('click', async () => {
    await setLocal('bhi_stale_dismissed', Date.now());
    document.getElementById('stale-banner').style.display = 'none';
  });
}


// ── API helper ────────────────────────────────────────────────────────────────

async function apiFetch(path, options = {}) {
  const url = `${BACKEND_URL}${path}`;
  console.log('[BHI]', options.method || 'GET', url);
  let res;
  try {
    res = await fetch(url, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    });
  } catch (networkErr) {
    throw new Error(`Cannot reach backend at ${BACKEND_URL} — is the server running? (${networkErr.message})`);
  }
  if (!res.ok) {
    let body = {};
    try { body = await res.json(); } catch (_) {}
    throw new Error(body.detail || `HTTP ${res.status} from ${path}`);
  }
  return res.json();
}


// ── Toast notification ────────────────────────────────────────────────────────

function showToast(msg, type = 'info') {
  const banner = document.getElementById('offline-banner');
  const color  = type === 'error' ? 'var(--danger)' : type === 'warning' ? 'var(--warning)' : 'var(--accent2)';
  const prev   = banner.innerHTML;
  const prevDisplay = banner.style.display;
  banner.innerHTML = `<span style="color:${color}">${esc(msg)}</span>`;
  banner.style.display = 'flex';
  setTimeout(() => { banner.style.display = prevDisplay; banner.innerHTML = prev; }, 5000);
}


// ── Utilities ─────────────────────────────────────────────────────────────────

function shortUrl(url) {
  if (!url) return '';
  // pdf:// pseudo-scheme — show just the filename
  if (url.startsWith('pdf://')) return url.slice(6);
  // file:///absolute/path — show just the filename
  if (url.startsWith('file:///')) {
    const parts = url.slice(8).split('/');
    return parts[parts.length - 1] || url.slice(0, 55);
  }
  try {
    const u = new URL(url);
    return (u.hostname + u.pathname).slice(0, 55);
  } catch (_) {
    return url.slice(0, 55);
  }
}

function esc(str) {
  return (str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function getLocal(key) {
  return new Promise(resolve => chrome.storage.local.get(key, r => resolve(r[key])));
}

function setLocal(key, value) {
  return new Promise(resolve => chrome.storage.local.set({ [key]: value }, resolve));
}


// ══════════════════════════════════════════════════════════════════════════════
// TEST PIPELINE PANEL — added for pipeline verification
// ══════════════════════════════════════════════════════════════════════════════

function setupTestPanel() {
  // ── Test URL indexing ──────────────────────────────────────────────────────
  document.getElementById('btn-test-url')?.addEventListener('click', async () => {
    const url = document.getElementById('test-url-input')?.value.trim();
    const statusEl = document.getElementById('test-url-status');
    if (!url) { setTestStatus(statusEl, '⚠ Enter a URL first', 'warning'); return; }

    const btn = document.getElementById('btn-test-url');
    btn.disabled = true;
    btn.textContent = '⏳ Indexing…';
    setTestStatus(statusEl, `Starting indexing of: ${url}`, 'info');

    try {
      const result = await apiFetch('/index/test/url', {
        method: 'POST',
        body: JSON.stringify({ url }),
      });
      pollTestRun(result.run_id, statusEl, btn, '⚡ Index This URL');
    } catch (err) {
      setTestStatus(statusEl, `❌ ${err.message}`, 'error');
      btn.disabled = false;
      btn.textContent = '⚡ Index This URL';
    }
  });

  // ── Test PDF indexing ──────────────────────────────────────────────────────
  document.getElementById('btn-test-pdf')?.addEventListener('click', async () => {
    const path    = document.getElementById('test-pdf-input')?.value.trim();
    const statusEl = document.getElementById('test-pdf-status');
    if (!path) { setTestStatus(statusEl, '⚠ Enter a PDF path first', 'warning'); return; }

    const btn = document.getElementById('btn-test-pdf');
    btn.disabled = true;
    btn.textContent = '⏳ Indexing…';
    setTestStatus(statusEl, `Starting indexing of: ${path}`, 'info');

    try {
      const result = await apiFetch('/index/test/pdf', {
        method: 'POST',
        body: JSON.stringify({ path }),
      });
      pollTestRun(result.run_id, statusEl, btn, '📄 Index This PDF');
    } catch (err) {
      setTestStatus(statusEl, `❌ ${err.message}`, 'error');
      btn.disabled = false;
      btn.textContent = '📄 Index This PDF';
    }
  });

  // ── Test query ─────────────────────────────────────────────────────────────
  document.getElementById('btn-test-query')?.addEventListener('click', runTestQuery);
  document.getElementById('test-query-input')?.addEventListener('keydown', e => {
    if (e.key === 'Enter') runTestQuery();
  });
}

async function runTestQuery() {
  const query    = document.getElementById('test-query-input')?.value.trim();
  const resultsEl = document.getElementById('test-query-results');
  const answerEl  = document.getElementById('test-query-answer');
  const listEl    = document.getElementById('test-query-list');
  const timingEl  = document.getElementById('test-query-timing');
  if (!query) return;

  resultsEl.style.display = 'block';
  answerEl.style.display  = 'block';
  answerEl.textContent    = '🔍 Searching…';
  listEl.innerHTML        = '';
  timingEl.textContent    = '';

  const t0 = performance.now();
  try {
    const result = await apiFetch('/query', {
      method: 'POST',
      body: JSON.stringify({ text: query, top_k: 5, top_urls: 3 }),
    });
    const elapsed = Math.round(performance.now() - t0);

    if (result.answer) {
      answerEl.textContent    = result.answer.slice(0, 600) + (result.answer.length > 600 ? '…' : '');
      answerEl.style.display  = 'block';
    } else {
      answerEl.style.display = 'none';
    }

    listEl.innerHTML = '';
    // Re-use the same renderSourceCard() used by the main search — gives
    // proper "Open Text Chunks" and "Open Page Image" buttons with
    // correct docType routing and file_path / web_url from the payload.
    (result.sources || []).forEach(s => {
      listEl.appendChild(renderSourceCard(s));
    });

    const rt = result.retrieval_duration_ms?.toFixed(0) ?? '?';
    const gt = result.generation_duration_ms?.toFixed(0) ?? '?';
    timingEl.textContent =
      `retrieve ${rt}ms · generate ${gt}ms · total ${elapsed}ms · `+
      `${result.total_chunks_searched ?? '?'} chunks searched`;

  } catch (err) {
    answerEl.textContent = `❌ Query error: ${err.message}`;
    console.error('[RAG] test query failed:', err);
  }
}

// Poll a test run_id and update a status div
async function pollTestRun(runId, statusEl, btn, btnLabel) {
  let attempts = 0;
  const maxAttempts = 120; // 4 minutes at 2s intervals

  const interval = setInterval(async () => {
    attempts++;
    if (attempts > maxAttempts) {
      clearInterval(interval);
      setTestStatus(statusEl, '⚠ Timed out waiting for result', 'warning');
      btn.disabled = false;
      btn.textContent = btnLabel;
      return;
    }
    try {
      const s = await apiFetch(`/index/status/${runId}`);
      const detail = buildTestStatusText(s);
      setTestStatus(statusEl, detail,
        s.status === 'completed' ? 'success' :
        s.status === 'failed'    ? 'error'   : 'info');

      if (['completed', 'failed', 'partial'].includes(s.status)) {
        clearInterval(interval);
        btn.disabled    = false;
        btn.textContent = btnLabel;
      }
    } catch (err) {
      console.error('[RAG] pollTestRun error:', err);
    }
  }, 2000);
}

function buildTestStatusText(s) {
  const lines = [
    `Status: ${s.status}`,
    `Fetched: ${s.fetched}  Parsed: ${s.parsed}  Chunked: ${s.chunked}`,
    `Embedded: ${s.embedded}  Stored: ${s.stored}  Failed: ${s.failed}`,
  ];
  if (s.errors && s.errors.length) {
    lines.push(`Errors: ${s.errors.slice(0, 3).join(' | ')}`);
  }
  return lines.join('\n');
}

function setTestStatus(el, text, type) {
  if (!el) return;
  el.style.display = 'block';
  el.style.whiteSpace = 'pre-line';
  el.style.color =
    type === 'success' ? 'var(--accent)'  :
    type === 'error'   ? 'var(--danger)'  :
    type === 'warning' ? 'var(--warning)' : 'var(--muted)';
  el.textContent = text;
}
