/* ══════════════════════════════════════════════════════════════════════════
   MARKED CONFIGURATION
══════════════════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  if (typeof marked !== 'undefined') {
    marked.use({ gfm: true, breaks: true });
  }
});

/* ══════════════════════════════════════════════════════════════════════════
   FETCH HELPER
   Wraps fetch() so non-JSON responses (proxy 502/504, nginx error pages,
   HTTP→HTTPS redirects) produce a clear error instead of "Unexpected token '<'".
══════════════════════════════════════════════════════════════════════════ */
async function apiFetch(url, options = {}) {
  const r = await fetch(url, options);
  const ct = r.headers.get('content-type') || '';
  if (!r.ok || !ct.includes('application/json')) {
    // Try to extract a detail message from JSON; fall back to status text.
    let detail;
    try {
      const body = await r.json();
      detail = body.detail || body.message || JSON.stringify(body);
    } catch {
      detail = await r.text().catch(() => '');
      // If it looks like HTML (proxy error page), give a friendlier message.
      if (detail.trimStart().startsWith('<')) {
        detail = `Server returned an HTML error page (HTTP ${r.status}). ` +
                 `Check that the backend is running and the proxy is correctly configured.`;
      }
    }
    throw new Error(detail || `HTTP ${r.status}`);
  }
  return r.json();
}

/* ══════════════════════════════════════════════════════════════════════════
   ERROR POPUP
══════════════════════════════════════════════════════════════════════════ */
function showErrorPopup(msg) {
  document.getElementById('error-popup-msg').textContent = msg;
  const popup = document.getElementById('error-popup');
  popup.classList.remove('hidden');
  // close on Escape
  document.addEventListener('keydown', _errorPopupEscHandler);
}
function dismissErrorPopup() {
  document.getElementById('error-popup').classList.add('hidden');
  document.removeEventListener('keydown', _errorPopupEscHandler);
}
function _errorPopupEscHandler(e) {
  if (e.key === 'Escape') dismissErrorPopup();
}

/* ══════════════════════════════════════════════════════════════════════════
   STATE
══════════════════════════════════════════════════════════════════════════ */
let currentResults   = [];
let currentSearchType = '';
let currentKeyword   = '';
let allUploads       = [];           // [{id, filename, row_count, upload_time, embedded_models}]
let selectedUploadIds = new Set();   // empty = search all
let bookmarkedIds    = new Set();    // msg ids that are bookmarked
let activeFilterTokens = [];         // tokens to highlight in filter results (exact mode)
let _allLabels       = [];           // all defined labels [{id, name, color}]
let _bmLabelFilter   = new Set();    // label IDs selected as bookmark filter

/* ══════════════════════════════════════════════════════════════════════════
   USERNAME COLOUR PALETTE
══════════════════════════════════════════════════════════════════════════ */
const _usernameColors = {};
let _colorIdx = 0;
const PALETTE = [
  ['#dbeafe','#1e40af'],['#dcfce7','#166534'],['#ede9fe','#5b21b6'],
  ['#fce7f3','#9d174d'],['#ffedd5','#9a3412'],['#ccfbf1','#0f5f53'],
  ['#fee2e2','#991b1b'],['#fef9c3','#854d0e'],['#f1f5f9','#334155'],
];
function usernameStyle(u) {
  if (!_usernameColors[u]) { _usernameColors[u] = PALETTE[_colorIdx++ % PALETTE.length]; }
  const [bg, text] = _usernameColors[u];
  return `background:${bg};color:${text}`;
}

/* ══════════════════════════════════════════════════════════════════════════
   UTILITIES
══════════════════════════════════════════════════════════════════════════ */
function esc(s) { const d = document.createElement('div'); d.textContent = s ?? ''; return d.innerHTML; }
function highlight(text, kw) {
  if (!kw || !text) return esc(text);
  return esc(text).replace(
    new RegExp(kw.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'), 'gi'),
    m => `<mark>${m}</mark>`
  );
}
// Highlight multiple tokens (longest first so phrases beat their own words)
function highlightTerms(text, tokens) {
  if (!tokens.length || !text) return esc(text);
  const sorted  = [...tokens].sort((a, b) => b.length - a.length);
  const pattern = sorted.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
  return esc(text).replace(new RegExp(`(${pattern})`, 'gi'), m => `<mark>${m}</mark>`);
}
function truthy(v) { return v === 'True' || v === 'true' || v === '1' || v === 1 || v === true; }
function hasContent(v) { return v && v !== '' && v !== 'nan' && v !== '[]' && v !== 'None'; }
function enc(s) { return encodeURIComponent(s); }

/* ══════════════════════════════════════════════════════════════════════════
   PAGE NAVIGATION
══════════════════════════════════════════════════════════════════════════ */
function navigateTo(page) {
  document.querySelectorAll('.nav-tab').forEach(b => b.classList.toggle('nav-active', b.dataset.page === page));
  document.getElementById('page-search').classList.toggle('hidden', page !== 'search');
  document.getElementById('page-settings').classList.toggle('hidden', page !== 'settings');
  document.getElementById('page-chat').classList.toggle('hidden', page !== 'chat');
  document.getElementById('page-bookmarks').classList.toggle('hidden', page !== 'bookmarks');
  if (page === 'settings') loadSettingsPage();
  if (page === 'bookmarks') loadBookmarksPage();
}

document.querySelectorAll('.nav-tab').forEach(btn => {
  btn.addEventListener('click', () => navigateTo(btn.dataset.page));
});

/* ══════════════════════════════════════════════════════════════════════════
   STATS
══════════════════════════════════════════════════════════════════════════ */
async function loadStats() {
  try {
    const d = await apiFetch('/api/stats');
    document.getElementById('stats-bar').innerHTML =
      `${d.total_messages.toLocaleString()} msgs &bull; ` +
      `${d.total_uploads} uploads &bull; ` +
      `${d.embedded_messages.toLocaleString()} embedded &bull; ` +
      `<span style="color:#c4b5fd">${esc(d.current_model_label)}</span>` +
      (d.api_key_set ? ' &bull; <span style="color:#86efac">API key ✓</span>' : '');
  } catch (_) {}
}

/* ══════════════════════════════════════════════════════════════════════════
   SCOPE SELECTOR (Search page — which uploads to search)
══════════════════════════════════════════════════════════════════════════ */
function renderScopeChips() {
  const container = document.getElementById('scope-chips');
  if (!allUploads.length) {
    container.innerHTML = '<span class="text-xs text-gray-400 italic">No uploads yet — go to the Data page to add data.</span>';
    return;
  }
  container.innerHTML = allUploads.map(u => {
    const active = selectedUploadIds.size === 0 || selectedUploadIds.has(u.id);
    return `
      <button class="scope-chip ${active ? 'scope-chip-on' : 'scope-chip-off'}"
              data-id="${u.id}" title="${esc(u.id)}">
        ${esc(u.filename)}
        <span class="text-[10px] opacity-70">${Number(u.row_count).toLocaleString()} rows</span>
      </button>`;
  }).join('');

  container.querySelectorAll('.scope-chip').forEach(btn => {
    btn.addEventListener('click', () => toggleScopeChip(btn.dataset.id, btn));
  });
}

function toggleScopeChip(uploadId, btn) {
  if (selectedUploadIds.size === 0) {
    // Currently "all" — switch to explicitly selecting all except clicked
    allUploads.forEach(u => { if (u.id !== uploadId) selectedUploadIds.add(u.id); });
  } else if (selectedUploadIds.has(uploadId)) {
    selectedUploadIds.delete(uploadId);
    if (selectedUploadIds.size === 0) selectedUploadIds = new Set(); // reset to "all"
  } else {
    selectedUploadIds.add(uploadId);
    // If all are now selected, reset to "all"
    if (selectedUploadIds.size === allUploads.length) selectedUploadIds = new Set();
  }
  renderScopeChips();
}

document.getElementById('scope-select-all').onclick = () => {
  selectedUploadIds = new Set();
  renderScopeChips();
};
document.getElementById('scope-select-none').onclick = () => {
  // Select none = only results section stays empty; keep 1st upload selected for usability
  selectedUploadIds = new Set();
  renderScopeChips();
};

function getScopeParam() {
  if (selectedUploadIds.size === 0) return '';
  return [...selectedUploadIds].join(',');
}

/* ══════════════════════════════════════════════════════════════════════════
   SETTINGS PAGE
══════════════════════════════════════════════════════════════════════════ */
async function loadModelOptions() {
  try {
    const models = await apiFetch('/api/embedding-models');
    const container = document.getElementById('model-options');
    container.innerHTML = models.map(m => {
      const available = m.available !== false;
      const availabilityNote = available
        ? ''
        : '<p class="text-xs text-rose-500 mt-1">Requires OpenAI API key to use.</p>';
      return `
      <label class="model-option ${m.active ? 'model-option-active' : ''}" data-id="${m.id}">
        <div class="flex items-start gap-3">
          <input type="radio" name="embed-model" value="${m.id}"
                 ${m.active ? 'checked' : ''} ${!available ? 'disabled' : ''} class="mt-0.5 accent-indigo-600 shrink-0" />
          <div class="flex-1 min-w-0">
            <div class="flex items-center gap-2 flex-wrap">
              <span class="text-sm font-semibold text-gray-800">${esc(m.label)}</span>
              ${m.local
                ? '<span class="text-xs px-1.5 py-0.5 rounded" style="background:#dcfce7;color:#166534">local</span>'
                : '<span class="text-xs px-1.5 py-0.5 rounded" style="background:#dbeafe;color:#1e40af">cloud</span>'}
              ${m.active ? '<span class="text-xs px-1.5 py-0.5 rounded font-medium" style="background:#ede9fe;color:#5b21b6">active</span>' : ''}
            </div>
            <p class="text-xs text-gray-500 mt-0.5">${esc(m.description)}</p>
            <p class="text-xs text-gray-400">${m.dims}-dim · ${m.embedded_count.toLocaleString()} msgs embedded</p>
            ${availabilityNote}
          </div>
        </div>
      </label>`;
    }).join('');

    container.querySelectorAll('input[name="embed-model"]').forEach(radio => {
      radio.addEventListener('change', async () => {
        try {
          const res = await fetch('/api/set-embedding-model', {
            method: 'POST', headers: {'Content-Type':'application/json'},
            body: JSON.stringify({model_id: radio.value}),
          });
          const d = await res.json();
          if (!res.ok) throw new Error(d.detail);
          const isLocal = models.find(m => m.id === radio.value)?.local;
          const cnt     = models.find(m => m.id === radio.value)?.embedded_count || 0;
          const msg = document.getElementById('model-msg');
          msg.textContent = `Switched to ${d.label}.` +
            (isLocal && cnt === 0 ? ' Weights will download on first use (~0.4–1.3 GB).' : '');
          msg.classList.remove('hidden');
          loadStats();
          loadModelOptions();
        } catch (e) { showErrorPopup('Failed to switch model: ' + e.message); }
      });
    });
  } catch (_) {}
}

/* ══════════════════════════════════════════════════════════════════════════
   API KEY POPUP
══════════════════════════════════════════════════════════════════════════ */
const STORAGE_KEY = 'openai_api_key';

async function _sendKeyToServer(key) {
  const res = await fetch('/api/set-api-key', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({api_key: key}),
  });
  const d = await res.json();
  if (!res.ok) throw new Error(d.detail || 'Failed to save API key');
  return d;
}

function showApiKeyPopup(dismissable = false) {
  const popup     = document.getElementById('apikey-popup');
  const input     = document.getElementById('apikey-popup-input');
  const skipBtn   = document.getElementById('apikey-popup-skip');
  const errorEl   = document.getElementById('apikey-popup-error');

  // Pre-fill if a key is already stored
  const stored = localStorage.getItem(STORAGE_KEY) || '';
  input.value = stored;
  errorEl.textContent = '';
  errorEl.classList.add('hidden');

  // Show/hide Skip button depending on dismissability
  skipBtn.classList.toggle('hidden', !dismissable);

  popup.classList.remove('hidden');
  input.focus();

  // Backdrop click only dismisses if dismissable
  document.getElementById('apikey-backdrop').onclick = dismissable ? hideApiKeyPopup : null;

  document.addEventListener('keydown', _apiKeyEscHandler);
}

function hideApiKeyPopup() {
  document.getElementById('apikey-popup').classList.add('hidden');
  document.removeEventListener('keydown', _apiKeyEscHandler);
}

function _apiKeyEscHandler(e) {
  if (e.key !== 'Escape') return;
  // Only dismiss on Esc if a key is already saved (popup is dismissable)
  if (localStorage.getItem(STORAGE_KEY)) hideApiKeyPopup();
}

function updateSettingsKeyStatus() {
  const statusEl = document.getElementById('settings-key-status');
  if (!statusEl) return;
  const stored = localStorage.getItem(STORAGE_KEY);
  statusEl.textContent = stored
    ? 'API key saved in your browser (localStorage).'
    : 'No API key set. Click "Change Key" to add one.';
}

// Save button
document.getElementById('apikey-popup-save').onclick = async () => {
  const input   = document.getElementById('apikey-popup-input');
  const errorEl = document.getElementById('apikey-popup-error');
  const saveBtn = document.getElementById('apikey-popup-save');
  const key = input.value.trim();

  if (!key) {
    errorEl.textContent = 'Please enter your API key.';
    errorEl.classList.remove('hidden');
    input.focus();
    return;
  }

  saveBtn.disabled = true;
  saveBtn.textContent = 'Saving…';
  errorEl.textContent = '';
  errorEl.classList.add('hidden');

  try {
    await _sendKeyToServer(key);
    localStorage.setItem(STORAGE_KEY, key);
    hideApiKeyPopup();
    updateSettingsKeyStatus();
    loadStats();
  } catch (e) {
    errorEl.textContent = e.message;
    errorEl.classList.remove('hidden');
  } finally {
    saveBtn.disabled = false;
    saveBtn.textContent = 'Save & Continue';
  }
};

// Skip button
document.getElementById('apikey-popup-skip').onclick = () => hideApiKeyPopup();

// Enter key in popup input triggers save
document.getElementById('apikey-popup-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') document.getElementById('apikey-popup-save').click();
});

// "Change Key" button on settings page
document.getElementById('settings-change-key').onclick = () => showApiKeyPopup(true);

/* ══════════════════════════════════════════════════════════════════════════
   DATA PAGE
══════════════════════════════════════════════════════════════════════════ */

/* ── Upload new CSV ── */
document.getElementById('upload-btn').onclick = async () => {
  const input = document.getElementById('csv-file');
  if (!input.files.length) { setUploadStatus('Please select a CSV file.', 'error'); return; }
  const btn = document.getElementById('upload-btn');
  btn.disabled = true; btn.textContent = 'Starting…';
  hideUploadProgress();
  showUploadProgress(0, 'Starting upload…');
  const form = new FormData();
  form.append('file', input.files[0]);
  try {
    const response = await fetch('/api/upload', { method: 'POST', body: form });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim();
          if (data.startsWith('Processing ')) {
            showUploadProgress(5, data);
          } else if (data.startsWith('Inserted ')) {
            showUploadProgress(20, data);
          } else if (data.startsWith('Starting embedding ')) {
            showUploadProgress(30, data);
          } else if (data.startsWith('Embedded ')) {
            const match = data.match(/Embedded\s+(\d+)\/(\d+)/);
            if (match) {
              const current = Number(match[1]);
              const total = Number(match[2]);
              const pct = total ? 30 + Math.min(70, 70 * current / total) : 30;
              showUploadProgress(pct, data);
            } else {
              showUploadProgress(50, data);
            }
            btn.textContent = data;
          } else if (data.startsWith('Completed:')) {
            showUploadProgress(100, data.replace('Completed: ', ''), 'success');
            btn.textContent = 'Upload & Embed';
            refreshUploads();
            loadStats();
          } else if (data.startsWith('Error')) {
            showUploadProgress(100, data, 'error');
          } else {
            setUploadStatus(data, 'info');
          }
        }
      }
    }
  } catch (e) {
    setUploadStatus('Error: ' + e.message, 'error');
    btn.textContent = 'Upload & Embed';
  } finally {
    btn.disabled = false;
  }
};
function setUploadStatus(msg, type) {
  const el = document.getElementById('upload-status');
  el.textContent = msg;
  el.className = 'mt-2 text-sm ' +
    (type==='error' ? 'text-red-600' : type==='success' ? 'text-green-600' : 'text-gray-500');
}

function showUploadProgress(percent, label, type = 'info') {
  const progress = document.getElementById('upload-progress');
  const fill = document.getElementById('upload-progress-fill');
  const labelEl = document.getElementById('upload-progress-label');
  if (!progress || !fill || !labelEl) return;
  progress.classList.remove('hidden');
  fill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  fill.classList.toggle('error', type === 'error');
  labelEl.textContent = `${label} (${Math.round(Math.max(0, Math.min(100, percent)))}%)`;
  setUploadStatus(label, type);
}

function hideUploadProgress() {
  const progress = document.getElementById('upload-progress');
  const fill = document.getElementById('upload-progress-fill');
  const labelEl = document.getElementById('upload-progress-label');
  if (!progress || !fill || !labelEl) return;
  progress.classList.add('hidden');
  fill.style.width = '0%';
  fill.classList.remove('error');
  labelEl.textContent = '';
}

/* ── Uploads table ── */
async function refreshUploads() {
  try {
    allUploads = await apiFetch('/api/uploads');
  } catch (_) { allUploads = []; }
  renderScopeChips();
  if (!document.getElementById('page-settings').classList.contains('hidden')) {
    renderUploadsTable();
  }
  loadStats();
}

function renderUploadsTable() {
  const container = document.getElementById('uploads-table');
  if (!allUploads.length) {
    container.innerHTML = '<p class="text-sm text-gray-400 text-center py-6">No uploads yet.</p>';
    return;
  }
  container.innerHTML = allUploads.map(u => uploadCard(u)).join('');

  container.querySelectorAll('.reembed-btn').forEach(btn => {
    btn.addEventListener('click', () => doReembed(btn.dataset.id, btn.dataset.name, btn));
  });
  container.querySelectorAll('.delete-db-btn').forEach(btn => {
    btn.addEventListener('click', () => confirmDelete(btn.dataset.id, btn.dataset.name, 'sqlite'));
  });
  container.querySelectorAll('.delete-embed-btn').forEach(btn => {
    btn.addEventListener('click', () => confirmDelete(btn.dataset.id, btn.dataset.name, 'embeddings'));
  });
  container.querySelectorAll('.delete-all-btn').forEach(btn => {
    btn.addEventListener('click', () => confirmDelete(btn.dataset.id, btn.dataset.name, 'full'));
  });
}

function uploadCard(u) {
  const modelBadges = Object.entries(u.embedded_models || {}).map(([mid, has]) => {
    const labels = {openai:'OpenAI'};
    return has
      ? `<span class="embed-badge embed-badge-yes">${labels[mid] || mid}</span>`
      : `<span class="embed-badge embed-badge-no">${labels[mid] || mid} —</span>`;
  }).join('');

  const safeId = u.id.replace(/[^a-zA-Z0-9-]/g, '');

  return `
    <div class="border border-gray-200 rounded-xl p-4 hover:border-indigo-200 transition-colors">
      <div class="flex items-start justify-between gap-3">
        <div class="flex-1 min-w-0">
          <p class="font-semibold text-sm text-gray-800 truncate">${esc(u.filename)}</p>
          <p class="text-xs text-gray-500 mt-0.5">
            ${Number(u.row_count).toLocaleString()} rows &bull;
            Uploaded ${u.upload_time.slice(0,16)}
          </p>
          <p class="text-xs text-gray-400 font-mono mt-0.5" title="Upload ID">${u.id}</p>
          <div class="flex flex-wrap gap-1.5 mt-2">${modelBadges}</div>
        </div>
        <div class="flex flex-col gap-2 shrink-0">
          <button class="reembed-btn action-btn-primary"
                  data-id="${u.id}" data-name="${esc(u.filename)}">
            Re-embed
          </button>
          <button class="delete-embed-btn action-btn-warning"
                  data-id="${u.id}" data-name="${esc(u.filename)}">
            Delete Embedding
          </button>
          <button class="delete-db-btn action-btn-danger"
                  data-id="${u.id}" data-name="${esc(u.filename)}">
            Delete DB
          </button>
          <button class="delete-all-btn action-btn-danger"
                  data-id="${u.id}" data-name="${esc(u.filename)}"
                  style="border-color:#dc2626;background:#fff1f2;font-weight:700;">
            Delete All
          </button>
        </div>
      </div>
      <!-- Inline re-embed progress -->
      <div id="reembed-progress-${safeId}" class="hidden mt-3">
        <div class="progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
          <div id="reembed-fill-${safeId}" class="progress-fill"></div>
        </div>
        <p id="reembed-label-${safeId}" class="mt-1 text-xs text-gray-600" aria-live="polite">Starting…</p>
      </div>
    </div>`;
}

// Track active poll timers so clicking Re-embed twice doesn't spawn duplicates.
const _reembedTimers = {};

async function doReembed(uploadId, filename, btn) {
  const safeId    = uploadId.replace(/[^a-zA-Z0-9-]/g, '');
  const progressEl = document.getElementById(`reembed-progress-${safeId}`);
  const fillEl     = document.getElementById(`reembed-fill-${safeId}`);
  const labelEl    = document.getElementById(`reembed-label-${safeId}`);

  // Ensure the inline error <pre> exists inside the progress section.
  let errEl = document.getElementById(`reembed-err-${safeId}`);
  if (!errEl && progressEl) {
    errEl = document.createElement('pre');
    errEl.id        = `reembed-err-${safeId}`;
    errEl.className = 'hidden mt-2 text-xs text-red-700 bg-red-50 border border-red-200 ' +
                      'rounded-lg p-2 max-h-48 overflow-auto whitespace-pre-wrap break-all';
    progressEl.appendChild(errEl);
  }

  function setProgress(pct, label) {
    if (fillEl) { fillEl.style.width = pct + '%'; fillEl.classList.remove('error'); }
    if (labelEl) labelEl.textContent = label;
    if (progressEl) progressEl.classList.remove('hidden');
  }
  function setError(msg) {
    if (fillEl) fillEl.classList.add('error');
    if (labelEl) labelEl.textContent = 'Failed — see details below';
    if (errEl)  { errEl.textContent = msg; errEl.classList.remove('hidden'); }
    if (progressEl) progressEl.classList.remove('hidden');
  }
  function finish() {
    btn.disabled    = false;
    btn.textContent = 'Re-embed';
    clearInterval(_reembedTimers[uploadId]);
    delete _reembedTimers[uploadId];
  }

  // Don't start a second timer if one is already polling.
  if (_reembedTimers[uploadId]) return;

  btn.disabled    = true;
  btn.textContent = 'Starting…';
  if (errEl) errEl.classList.add('hidden');
  setProgress(0, 'Submitting job…');

  // ── 1. POST to start the background job ──────────────────────────────────
  let jobId;
  try {
    const res = await fetch(`/api/uploads/${enc(uploadId)}/reembed`, { method: 'POST' });
    let d;
    try { d = await res.json(); } catch {
      throw new Error(`Server returned non-JSON (HTTP ${res.status})`);
    }
    if (!res.ok) throw new Error(d.detail || `HTTP ${res.status}`);
    jobId = d.job_id;
    if (d.already_running) {
      setProgress(0, 'Job already running — resuming progress display…');
    } else {
      const skip = d.skipped || 0;
      setProgress(0, skip > 0
        ? `Resuming: ${skip.toLocaleString()} already embedded, checking remainder…`
        : `Job started — ${(d.total_messages || 0).toLocaleString()} messages queued`);
    }
  } catch (e) {
    setError(`Failed to start job:\n${e.message}`);
    finish();
    return;
  }

  // ── 2. Poll GET /api/jobs/{jobId} every 1.5 s ────────────────────────────
  btn.textContent = 'Embedding…';

  _reembedTimers[uploadId] = setInterval(async () => {
    let job;
    try {
      const r = await fetch(`/api/jobs/${enc(jobId)}`);
      if (!r.ok) return;   // transient — keep polling
      job = await r.json();
    } catch {
      return;              // network blip — keep polling
    }

    const embedded = job.embedded  || 0;
    const total    = job.total     || 0;
    const skipped  = job.skipped   || 0;
    const pct      = total > 0 ? Math.round(embedded / total * 100) : (job.status === 'completed' ? 100 : 0);

    if (job.status === 'running') {
      if (job.phase === 'checking') {
        // During the skip-check phase: show how many are already embedded (grows over time)
        const checkLabel = skipped > 0
          ? `Checking… ${skipped.toLocaleString()} already embedded so far`
          : 'Checking which messages are already embedded…';
        setProgress(0, checkLabel);
      } else {
        const batchInfo = job.current_batch ? ` (batch ${job.current_batch})` : '';
        const skipNote  = skipped > 0 ? ` · ${skipped.toLocaleString()} skipped` : '';
        setProgress(pct, `Embedding… ${pct}% — ${embedded.toLocaleString()}/${total.toLocaleString()} new messages${skipNote}${batchInfo}`);
      }

    } else if (job.status === 'completed') {
      const skipNote = skipped > 0 ? `, ${skipped.toLocaleString()} already embedded` : '';
      const errNote  = job.batch_errors.length > 0 ? ` (${job.batch_errors.length} batch error(s) — see below)` : '';
      setProgress(100, `Done — ${embedded.toLocaleString()} embedded${skipNote}${errNote}`);

      if (job.batch_errors.length > 0) {
        const detail = job.batch_errors.map(be =>
          `Batch ${be.batch}:\n${be.error}\n\n${be.traceback}`
        ).join('\n─────────────────\n');
        setError(detail);
      }
      refreshUploads();
      loadStats();
      setTimeout(() => {
        if (progressEl && !errEl?.textContent) progressEl.classList.add('hidden');
      }, 4000);
      finish();

    } else if (job.status === 'failed') {
      const detail = `${job.error || 'Unknown error'}\n\n${job.traceback || ''}`.trim();
      setError(detail);
      finish();
    }
  }, 1500);
}

/* ── Delete with confirm modal ── */
let _pendingDeleteId   = null;
let _pendingDeleteType = null; // 'full' | 'sqlite' | 'embeddings'

function confirmDelete(uploadId, filename, type) {
  _pendingDeleteId   = uploadId;
  _pendingDeleteType = type;

  const titleEl = document.getElementById('confirm-modal-title');
  const msgEl   = document.getElementById('confirm-msg');

  if (type === 'sqlite') {
    titleEl.textContent = 'Delete from Database';
    msgEl.textContent   = `Remove "${filename}" messages from SQLite only. Embeddings will be preserved. This cannot be undone.`;
  } else if (type === 'embeddings') {
    titleEl.textContent = 'Delete Embeddings';
    msgEl.textContent   = `Remove all vector embeddings for "${filename}" from the vector store. Messages in the database will be preserved.`;
  } else {
    titleEl.textContent = 'Delete Upload';
    msgEl.textContent   = `Delete "${filename}" and all its messages from both the database and the vector store? This cannot be undone.`;
  }
  document.getElementById('confirm-modal').classList.remove('hidden');
}

document.getElementById('confirm-cancel').onclick = () => {
  document.getElementById('confirm-modal').classList.add('hidden');
  document.getElementById('delete-progress').classList.add('hidden');
  document.getElementById('delete-status').classList.add('hidden');
  document.getElementById('confirm-ok').disabled = false;
  _pendingDeleteId   = null;
  _pendingDeleteType = null;
};
document.getElementById('confirm-ok').onclick = async () => {
  if (!_pendingDeleteId) return;
  const btn = document.getElementById('confirm-ok');
  const statusEl = document.getElementById('delete-status');
  const progressEl = document.getElementById('delete-progress');
  const progressFill = document.getElementById('delete-progress-fill');

  btn.disabled = true;
  progressEl.classList.remove('hidden');
  statusEl.classList.add('hidden');
  progressFill.classList.remove('error');

  let url;
  if (_pendingDeleteType === 'sqlite') {
    url = `/api/uploads/${enc(_pendingDeleteId)}/sqlite`;
  } else if (_pendingDeleteType === 'embeddings') {
    url = `/api/uploads/${enc(_pendingDeleteId)}/embeddings`;
  } else {
    url = `/api/uploads/${enc(_pendingDeleteId)}`;
  }

  try {
    const res = await fetch(url, { method: 'DELETE' });
    let d;
    try {
      d = await res.json();
    } catch {
      throw new Error(`Server returned a non-JSON response (HTTP ${res.status}). The operation may have timed out — check server logs.`);
    }
    if (!res.ok) throw new Error(d.detail || `HTTP ${res.status}`);

    progressFill.style.width = '100%';
    progressEl.querySelector('#delete-progress-label').textContent = 'Done';

    if (_pendingDeleteType === 'sqlite') {
      statusEl.textContent = `Removed ${d.deleted_messages} messages from the database. Embeddings untouched.`;
    } else if (_pendingDeleteType === 'embeddings') {
      statusEl.textContent = `Removed ${d.deleted_embeddings} embeddings from the vector store. Database untouched.`;
    } else {
      statusEl.textContent = `Removed ${d.deleted_messages} messages and all embeddings.`;
    }
    statusEl.className = 'bg-green-50 text-green-700 border border-green-200 rounded-lg p-3 text-sm';
    statusEl.classList.remove('hidden');

    if (_pendingDeleteType !== 'embeddings') {
      selectedUploadIds.delete(_pendingDeleteId);
    }
    setTimeout(() => {
      document.getElementById('confirm-modal').classList.add('hidden');
      refreshUploads();
    }, 1500);
  } catch (e) {
    progressFill.style.width = '100%';
    progressFill.classList.add('error');
    progressEl.querySelector('#delete-progress-label').textContent = 'Failed';
    statusEl.textContent = `Error: ${e.message}`;
    statusEl.className = 'bg-red-50 text-red-700 border border-red-200 rounded-lg p-3 text-sm';
    statusEl.classList.remove('hidden');
  }
  btn.disabled = false;
  _pendingDeleteId   = null;
  _pendingDeleteType = null;
};

document.getElementById('refresh-data-btn').onclick = refreshUploads;
document.getElementById('refresh-suno-btn').onclick = renderSunoTeamTable;

async function loadSettingsPage() {
  loadModelOptions();
  renderUploadsTable();
  renderLabelManager();
  renderSunoTeamTable();
  updateSettingsKeyStatus();
}

async function renderLabelManager() {
  await loadAllLabels();   // refreshes _allLabels + filter chips if bookmark page was open
  const list = document.getElementById('labels-list');
  if (!_allLabels.length) {
    list.innerHTML = '<p class="text-sm text-gray-400">No labels yet. Create one above.</p>';
    return;
  }
  list.innerHTML = _allLabels.map(l => {
    const tc = labelTextColor(l.color);
    return `<span class="inline-flex items-center gap-1.5 text-xs font-medium rounded-full px-3 py-1"
                  style="background:${l.color};color:${tc}">
              ${esc(l.name)}
              <button class="label-delete-btn opacity-70 hover:opacity-100 font-bold leading-none"
                      data-label-id="${l.id}" data-label-name="${esc(l.name)}" title="Delete label">×</button>
            </span>`;
  }).join('');
}

document.getElementById('labels-list').addEventListener('click', async e => {
  const btn = e.target.closest('.label-delete-btn');
  if (!btn) return;
  const id   = parseInt(btn.dataset.labelId);
  const name = btn.dataset.labelName;
  if (!confirm(`Delete label "${name}"? It will be removed from all bookmarks.`)) return;
  btn.disabled = true;
  const res = await fetch(`/api/labels/${id}`, { method: 'DELETE' });
  if (!res.ok) { btn.disabled = false; return; }
  _allLabels = _allLabels.filter(l => l.id !== id);
  _bmLabelFilter.delete(id);
  _cachedBookmarks.forEach(bm => { bm.labels = (bm.labels || []).filter(l => l.id !== id); });
  renderLabelManager();
  renderBmLabelFilterChips();
});

document.getElementById('label-create-form').addEventListener('submit', async e => {
  e.preventDefault();
  const nameInput  = document.getElementById('label-name-input');
  const colorInput = document.getElementById('label-color-input');
  const msgEl      = document.getElementById('label-create-msg');
  const name  = nameInput.value.trim();
  const color = colorInput.value;
  if (!name) return;
  msgEl.classList.add('hidden');
  const res = await fetch('/api/labels', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, color }),
  });
  if (!res.ok) {
    msgEl.textContent = (await res.json()).detail || 'Failed to create label.';
    msgEl.classList.remove('hidden');
    return;
  }
  const newLabel = await res.json();
  _allLabels = [..._allLabels, newLabel].sort((a, b) =>
    a.name.localeCompare(b.name, undefined, { sensitivity: 'base' })
  );
  nameInput.value = '';
  renderLabelManager();
  renderBmLabelFilterChips();
});

async function renderSunoTeamTable() {
  const el = document.getElementById('suno-team-table');
  el.innerHTML = '<p class="text-sm text-gray-400 text-center py-6">Loading…</p>';
  try {
    const members = await apiFetch('/api/suno-team');
    if (!members.length) {
      el.innerHTML = '<p class="text-sm text-gray-400 text-center py-6">No Suno Team members found.</p>';
      return;
    }
    el.innerHTML = `
      <div class="overflow-x-auto">
        <table class="w-full text-sm border-collapse">
          <thead>
            <tr class="bg-gray-50 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
              <th class="px-3 py-2 border-b border-gray-200">Username</th>
              <th class="px-3 py-2 border-b border-gray-200 text-right">Messages</th>
              <th class="px-3 py-2 border-b border-gray-200"></th>
            </tr>
          </thead>
          <tbody>
            ${members.map(m => `
              <tr class="border-b border-gray-100 hover:bg-gray-50" id="suno-row-${esc(m.username)}">
                <td class="px-3 py-2">
                  <span class="ubadge" style="${usernameStyle(m.username)}">${esc(m.username)}</span>
                </td>
                <td class="px-3 py-2 text-right text-gray-600 tabular-nums">${m.msg_count.toLocaleString()}</td>
                <td class="px-3 py-2 text-right">
                  <button class="suno-remove action-btn-danger"
                          data-username="${esc(m.username)}">
                    Remove from team
                  </button>
                </td>
              </tr>`).join('')}
          </tbody>
        </table>
      </div>`;
  } catch (e) {
    el.innerHTML = `<p class="text-sm text-red-500 text-center py-6">Failed to load: ${esc(e.message)}</p>`;
  }
}

document.getElementById('suno-team-table').addEventListener('click', async e => {
  const btn = e.target.closest('.suno-remove');
  if (!btn) return;
  const username = btn.dataset.username;
  btn.disabled = true;
  btn.textContent = 'Removing…';
  try {
    const res = await fetch(`/api/suno-team/${encodeURIComponent(username)}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(await res.text());
    const row = document.getElementById(`suno-row-${username}`);
    if (row) row.remove();
    // if table body is now empty, show empty state
    const tbody = document.querySelector('#suno-team-table tbody');
    if (tbody && !tbody.querySelector('tr')) {
      document.getElementById('suno-team-table').innerHTML =
        '<p class="text-sm text-gray-400 text-center py-6">No Suno Team members found.</p>';
    }
  } catch (err) {
    btn.disabled = false;
    btn.textContent = 'Remove from team';
    alert(`Failed to remove: ${err.message}`);
  }
});

/* ══════════════════════════════════════════════════════════════════════════
   SEARCH TABS
══════════════════════════════════════════════════════════════════════════ */
document.querySelectorAll('.search-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.search-tab').forEach(b => b.classList.remove('tab-active'));
    btn.classList.add('tab-active');
    document.querySelectorAll('.search-panel').forEach(p => p.classList.add('hidden'));
    document.getElementById('tab-' + btn.dataset.tab).classList.remove('hidden');
  });
});

[
  ['username-input',           'username'],
  ['keyword-input',            'keyword'],
  ['keyword-username-filter',  'keyword'],
  ['semantic-input',           'semantic'],
  ['semantic-username-filter', 'semantic'],
].forEach(([id, type]) => {
  document.getElementById(id).addEventListener('keydown', e => {
    if (e.key === 'Enter') doSearch(type);
  });
});

document.getElementById('username-search-btn').addEventListener('click', () => doSearch('username'));
document.getElementById('keyword-search-btn').addEventListener('click', () => doSearch('keyword'));
document.getElementById('semantic-search-btn').addEventListener('click', () => doSearch('semantic'));
const rangeSearchBtn = document.getElementById('range-search-btn');
if (rangeSearchBtn) {
  rangeSearchBtn.addEventListener('click', () => doSearch('range'));
}

document.getElementById('range-mode-exact').addEventListener('click', () => {
  document.getElementById('range-mode-exact').classList.add('range-mode-active');
  document.getElementById('range-mode-month').classList.remove('range-mode-active');
  document.getElementById('range-exact-inputs').classList.remove('hidden');
  document.getElementById('range-month-inputs').classList.add('hidden');
});
document.getElementById('range-mode-month').addEventListener('click', () => {
  document.getElementById('range-mode-month').classList.add('range-mode-active');
  document.getElementById('range-mode-exact').classList.remove('range-mode-active');
  document.getElementById('range-month-inputs').classList.remove('hidden');
  document.getElementById('range-exact-inputs').classList.add('hidden');
});

/* ══════════════════════════════════════════════════════════════════════════
   SEARCH
══════════════════════════════════════════════════════════════════════════ */
function appendDateParams(url, from, to) {
  if (from) url += `&date_from=${enc(from)}`;
  if (to)   url += `&date_to=${enc(to)}`;
  return url;
}

async function doSearch(type) {
  let url, keyword = '';
  const scope = getScopeParam();

  if (type === 'username') {
    const q       = document.getElementById('username-input').value.trim();
    if (!q) return;
    const limit   = document.getElementById('username-limit').value || 200;
    const dFrom   = document.getElementById('username-date-from').value;
    const dTo     = document.getElementById('username-date-to').value;
    const suno    = document.getElementById('username-suno').value;
    const minW    = parseInt(document.getElementById('username-min-words').value) || 0;
    url = `/api/search/username?username=${enc(q)}&limit=${limit}`;
    if (scope)          url += `&upload_ids=${enc(scope)}`;
    if (suno !== 'all') url += `&suno_team=${enc(suno)}`;
    if (minW > 1) url += `&min_words=${minW}`;
    url = appendDateParams(url, dFrom, dTo);

  } else if (type === 'keyword') {
    keyword     = document.getElementById('keyword-input').value.trim();
    if (!keyword) return;
    const uFilter = document.getElementById('keyword-username-filter').value.trim();
    const limit   = document.getElementById('keyword-limit').value || 200;
    const dFrom   = document.getElementById('keyword-date-from').value;
    const dTo     = document.getElementById('keyword-date-to').value;
    const suno    = document.getElementById('keyword-suno').value;
    const minW    = parseInt(document.getElementById('keyword-min-words').value) || 0;
    url = `/api/search/keyword?keyword=${enc(keyword)}&limit=${limit}`;
    if (uFilter)        url += `&username=${enc(uFilter)}`;
    if (scope)          url += `&upload_ids=${enc(scope)}`;
    if (suno !== 'all') url += `&suno_team=${enc(suno)}`;
    if (minW > 1) url += `&min_words=${minW}`;
    url = appendDateParams(url, dFrom, dTo);

  } else if (type === 'semantic') {
    const q       = document.getElementById('semantic-input').value.trim();
    if (!q) return;
    const n       = document.getElementById('semantic-n').value || 20;
    const uFilter = document.getElementById('semantic-username-filter').value.trim();
    const dFrom   = document.getElementById('semantic-date-from').value;
    const dTo     = document.getElementById('semantic-date-to').value;
    const suno    = document.getElementById('semantic-suno').value;
    const minW    = parseInt(document.getElementById('semantic-min-words').value) || 0;
    url = `/api/search/semantic?query=${enc(q)}&n_results=${n}`;
    if (uFilter)        url += `&username=${enc(uFilter)}`;
    if (scope)          url += `&upload_ids=${enc(scope)}`;
    if (suno !== 'all') url += `&suno_team=${enc(suno)}`;
    if (minW > 1) url += `&min_words=${minW}`;
    url = appendDateParams(url, dFrom, dTo);
  } else if (type === 'range') {
    const suno       = document.getElementById('range-suno').value;
    const isMonthMode = document.getElementById('range-mode-month').classList.contains('range-mode-active');
    let dFrom = '', dTo = '';

    if (isMonthMode) {
      const mFrom = document.getElementById('range-month-from').value; // "YYYY-MM"
      const mTo   = document.getElementById('range-month-to').value;
      if (!mFrom && !mTo) {
        renderError('Please set at least a "From" or "To" month.');
        return;
      }
      if (mFrom) dFrom = `${mFrom}-01`;
      if (mTo) {
        const [y, m] = mTo.split('-');
        const lastDay = new Date(parseInt(y), parseInt(m), 0).getDate();
        dTo = `${mTo}-${String(lastDay).padStart(2, '0')}`;
      }
    } else {
      dFrom = document.getElementById('range-date-from').value;
      dTo   = document.getElementById('range-date-to').value;
      if (!dFrom && !dTo) {
        renderError('Please set at least a "From" or "To" date.');
        return;
      }
    }

    const minW = parseInt(document.getElementById('range-min-words').value) || 0;
    url = '/api/search/range?';
    if (scope)          url += `upload_ids=${enc(scope)}&`;
    if (suno !== 'all') url += `suno_team=${enc(suno)}&`;
    if (dFrom)        url += `date_from=${enc(dFrom)}&`;
    if (dTo)          url += `date_to=${enc(dTo)}&`;
    if (minW > 1)     url += `min_words=${minW}&`;
    url = url.replace(/[?&]$/, '');
  }

  currentSearchType = type;
  currentKeyword    = keyword;
  setBtnLoading(type, true);

  try {
    const res = await fetch(url);
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Search failed'); }
    currentResults = await res.json();
    renderResults(currentResults);
  } catch (e) { renderError(e.message); }
  finally { setBtnLoading(type, false); }
}

function setBtnLoading(type, loading) {
  const btn = document.getElementById(`${type}-search-btn`);
  btn.disabled   = loading;
  btn.textContent = loading ? 'Searching…' : 'Search';
}

/* ══════════════════════════════════════════════════════════════════════════
   RESULTS
══════════════════════════════════════════════════════════════════════════ */
function renderError(msg) {
  showErrorPopup(msg);
}

function renderResults(results) {
  const sec = document.getElementById('results-section');
  sec.classList.remove('hidden');
  document.getElementById('results-count').textContent =
    `${results.length.toLocaleString()} result${results.length !== 1 ? 's' : ''}`;
  document.getElementById('export-btn').classList.toggle('hidden', results.length === 0);

  // Reset filter bar on new search
  activeFilterTokens = [];
  document.getElementById('results-filter').value = '';
  document.getElementById('results-filter-clear').classList.add('hidden');
  document.getElementById('results-filter-count').classList.add('hidden');
  document.getElementById('filter-spinner').classList.add('hidden');

  const container = document.getElementById('results-container');
  sec.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  if (!results.length) {
    container.innerHTML = '<p class="text-center text-gray-400 py-10 text-sm">No results found.</p>';
    return;
  }
  container.innerHTML = results.map(msg => msgCard(msg)).join('');
  container.querySelectorAll('.ctx-toggle').forEach(btn => {
    btn.addEventListener('click', () => toggleContext(parseInt(btn.dataset.id), btn));
  });
}

/* ══════════════════════════════════════════════════════════════════════════
   RESULTS FILTER
══════════════════════════════════════════════════════════════════════════ */

let filterMode        = 'exact';  // 'exact' | 'semantic'
let _semanticDebounce = null;

/* ── Set active mode + update UI ── */
const _EXACT_ACTIVE    = ['bg-indigo-700','text-white'];
const _EXACT_INACTIVE  = ['bg-slate-100','text-slate-500'];
const _SEM_ACTIVE      = ['bg-violet-600','text-white'];
const _SEM_INACTIVE    = ['bg-slate-100','text-slate-500'];

function setFilterMode(mode) {
  filterMode = mode;
  const exactBtn = document.getElementById('filter-mode-exact');
  const semBtn   = document.getElementById('filter-mode-semantic');

  if (mode === 'exact') {
    exactBtn.classList.remove(..._EXACT_INACTIVE);  exactBtn.classList.add(..._EXACT_ACTIVE);
    semBtn.classList.remove(..._SEM_ACTIVE);         semBtn.classList.add(..._SEM_INACTIVE);
  } else {
    semBtn.classList.remove(..._SEM_INACTIVE);       semBtn.classList.add(..._SEM_ACTIVE);
    exactBtn.classList.remove(..._EXACT_ACTIVE);     exactBtn.classList.add(..._EXACT_INACTIVE);
  }

  document.getElementById('results-filter').placeholder =
    mode === 'exact'
      ? 'Exact: whole-word match, multi-word phrase scores highest…'
      : 'Semantic: re-rank results by embedding similarity…';
}

document.getElementById('filter-mode-exact')
  .addEventListener('click', () => { setFilterMode('exact');    applyResultsFilter(); });
document.getElementById('filter-mode-semantic')
  .addEventListener('click', () => { setFilterMode('semantic'); applyResultsFilter(); });

/* ── Shared render helpers ── */
function _attachCtxListeners(container) {
  container.querySelectorAll('.ctx-toggle').forEach(btn => {
    btn.addEventListener('click', () => toggleContext(parseInt(btn.dataset.id), btn));
  });
}

function _renderFilteredCards(msgs, total) {
  const countLabel = document.getElementById('results-filter-count');
  const container  = document.getElementById('results-container');
  countLabel.textContent = `${msgs.length} of ${total}`;
  countLabel.classList.remove('hidden');
  if (!msgs.length) {
    container.innerHTML = '<p class="text-center text-gray-400 py-10 text-sm">No results match the filter.</p>';
    return;
  }
  container.innerHTML = msgs.map(m => msgCard(m)).join('');
  _attachCtxListeners(container);
}

function _resetToAllResults() {
  activeFilterTokens = [];
  document.getElementById('results-filter-count').classList.add('hidden');
  const container = document.getElementById('results-container');
  container.innerHTML = currentResults.map(m => msgCard(m)).join('');
  _attachCtxListeners(container);
}

/* ── Exact filter (instant, client-side) ── */
function _escapeRegex(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

function _applyExactFilter(term) {
  if (!term) { _resetToAllResults(); return; }

  const words = term.split(/\s+/).filter(Boolean);
  // Tokens for highlighting: phrase first (if multi-word), then individual words
  activeFilterTokens = words.length > 1 ? [term, ...words] : words;

  // Word-boundary regex per token — "cat" won't match "category"
  const tokenRegexes = words.map(w =>
    new RegExp('\\b' + _escapeRegex(w) + '\\b', 'i')
  );
  // Phrase regex: full sequence with word boundaries (multi-word only)
  const phraseRegex = words.length > 1
    ? new RegExp('\\b' + words.map(_escapeRegex).join('\\s+') + '\\b', 'i')
    : null;

  const scored = currentResults
    .map(m => {
      const user    = m.username || '';
      const content = m.content  || '';

      // Exact phrase match (multi-word) — scores highest
      if (phraseRegex) {
        const inUser    = phraseRegex.test(user);
        const inContent = phraseRegex.test(content);
        if (inUser || inContent) return { m, s: 1000 + (inUser ? 500 : 0) };
      }

      // Per-token whole-word matches
      let s = 0;
      for (const rx of tokenRegexes) {
        if (rx.test(user))    s += 20;
        if (rx.test(content)) s += 10;
      }
      return { m, s };
    })
    .filter(x => x.s > 0)
    .sort((a, b) => new Date(a.m.date) - new Date(b.m.date));

  _renderFilteredCards(scored.map(x => x.m), currentResults.length);
}

/* ── Semantic filter (debounced, API-backed) ── */
async function _applySemanticFilter(term) {
  if (!term) { _resetToAllResults(); return; }
  if (!currentResults.length) return;

  const spinner    = document.getElementById('filter-spinner');
  const countLabel = document.getElementById('results-filter-count');

  spinner.classList.remove('hidden');
  countLabel.classList.add('hidden');

  try {
    const res = await fetch('/api/filter/semantic', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: term, msg_ids: currentResults.map(m => m.id) }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Semantic filter failed' }));
      throw new Error(err.detail || 'Semantic filter failed');
    }

    const data = await res.json();
    const { results: ranked, threshold, query_used, warning } = data;
    if (warning) {
      document.getElementById('results-container').innerHTML =
        `<p class="text-center text-amber-600 py-10 text-sm">⚠ ${esc(warning)}</p>`;
      spinner.classList.add('hidden');
      return;
    }
    const byId = Object.fromEntries(currentResults.map(m => [m.id, m]));

    const hits = ranked
      .map(r => ({ ...byId[r.id], similarity_score: r.score }))
      .filter(m => m.id != null);

    // Show count with threshold label, and query interpretation if it changed
    const interpreted = query_used && query_used !== term
      ? ` · searched: "${query_used}"` : '';
    const countLabel2 = document.getElementById('results-filter-count');
    countLabel2.textContent = `${hits.length} of ${currentResults.length} · similarity ≥ ${threshold}${interpreted}`;
    countLabel2.classList.remove('hidden');

    const container = document.getElementById('results-container');
    if (!hits.length) {
      container.innerHTML = `<p class="text-center text-gray-400 py-10 text-sm">
        No results above the ${threshold} similarity threshold${interpreted}.<br>
        <span class="text-xs">Try a broader query or switch to Exact mode.</span>
      </p>`;
      return;
    }
    container.innerHTML = hits.map(m => msgCard(m)).join('');
    _attachCtxListeners(container);
  } catch (e) {
    showErrorPopup(`Semantic filter error: ${e.message}`);
    _resetToAllResults();
  } finally {
    spinner.classList.add('hidden');
  }
}

/* ── Unified dispatcher ── */
function applyResultsFilter() {
  const term     = document.getElementById('results-filter').value.trim().toLowerCase();
  const clearBtn = document.getElementById('results-filter-clear');
  clearBtn.classList.toggle('hidden', !term);
  if (!currentResults.length) return;

  clearTimeout(_semanticDebounce);
  if (filterMode === 'exact') {
    _applyExactFilter(term);
  } else {
    if (!term) { _applySemanticFilter(''); return; }
    _semanticDebounce = setTimeout(() => _applySemanticFilter(term), 500);
  }
}

document.getElementById('results-filter').addEventListener('input', applyResultsFilter);
document.getElementById('results-filter-clear').addEventListener('click', () => {
  document.getElementById('results-filter').value = '';
  applyResultsFilter();
});

function formatDate(raw) {
  if (!raw) return '';
  const d = new Date(raw);
  if (isNaN(d)) return esc(raw);  // prevent XSS if date field contains HTML
  const MONTHS = ['January','February','March','April','May','June',
                  'July','August','September','October','November','December'];
  const day  = d.getUTCDate();
  const mon  = MONTHS[d.getUTCMonth()];
  const yr   = d.getUTCFullYear();
  const hh   = String(d.getUTCHours()).padStart(2, '0');
  const mm   = String(d.getUTCMinutes()).padStart(2, '0');
  const ss   = String(d.getUTCSeconds()).padStart(2, '0');
  return `${day} ${mon} ${yr} ${hh}:${mm}:${ss} GMT+0`;
}

function msgCard(msg) {
  const score = msg.similarity_score !== undefined
    ? `<span class="text-xs px-2 py-0.5 rounded-full" style="background:#eef2ff;color:#3730a3">
         ${msg.similarity_score}
       </span>` : '';
  const teamBadge = truthy(msg.is_suno_team)
    ? `<span class="text-xs px-2 py-0.5 rounded-full font-medium"
             style="background:#fef3c7;color:#92400e">Suno Team</span>` : '';
  const body = activeFilterTokens.length
    ? highlightTerms(msg.content, activeFilterTokens)
    : currentSearchType === 'keyword'
      ? highlight(msg.content, currentKeyword)
      : esc(msg.content);
  const userHtml = activeFilterTokens.length
    ? highlightTerms(msg.username, activeFilterTokens)
    : esc(msg.username);
  const attachLine = hasContent(msg.attachments)
    ? `<p class="text-xs text-gray-500 mt-1">📎 ${esc(msg.attachments)}</p>` : '';
  const reactLine = hasContent(msg.reactions)
    ? `<p class="text-xs text-gray-500 mt-1">💬 ${esc(msg.reactions)}</p>` : '';

  // Find source filename
  const src = allUploads.find(u => u.id === msg.upload_id);
  const srcLabel = src ? `<span class="text-[10px] text-gray-400 truncate max-w-[12rem]" title="${esc(msg.upload_id)}">${esc(src.filename)}</span>` : '';

  return `
    <div id="card-${msg.id}" class="bg-white rounded-2xl shadow border border-gray-100 overflow-hidden">
      <div class="p-4">
        <div class="flex items-start justify-between gap-2 mb-2">
          <div class="flex flex-col gap-0.5">
            <div class="flex items-center flex-wrap gap-1.5">
              <span class="ubadge" style="${usernameStyle(msg.username)}">${userHtml}</span>
              ${teamBadge}${score}
            </div>
            <span class="text-xs text-gray-400">${formatDate(msg.date)}</span>
          </div>
          ${srcLabel}
        </div>
        <p class="text-sm leading-relaxed text-gray-800 whitespace-pre-wrap break-words">${body}</p>
        ${attachLine}${reactLine}
      </div>
      <div class="border-t bg-gray-50 px-4 py-2 flex items-center justify-between gap-2">
        <button class="bm-toggle text-xs font-medium flex items-center gap-1 transition-colors"
                data-id="${msg.id}"
                title="${bookmarkedIds.has(msg.id) ? 'Remove bookmark' : 'Save bookmark'}">
          ${bookmarkedIds.has(msg.id)
            ? `<svg class="w-3.5 h-3.5 text-amber-500" fill="currentColor" viewBox="0 0 24 24"><path d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"/></svg><span class="text-amber-600">Bookmarked</span>`
            : `<svg class="w-3.5 h-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"/></svg><span class="text-gray-500">Bookmark</span>`}
        </button>
        <button class="ctx-toggle text-xs text-indigo-600 hover:text-indigo-800 font-medium"
                data-id="${msg.id}" data-open="false">
          Show context ↕
        </button>
      </div>
      <div id="ctx-${msg.id}" class="hidden"></div>
    </div>`;
}

/* ══════════════════════════════════════════════════════════════════════════
   CONTEXT EXPANSION
══════════════════════════════════════════════════════════════════════════ */
async function toggleContext(id, btn) {
  const ctxEl = document.getElementById(`ctx-${id}`);
  if (btn.dataset.open === 'true') {
    ctxEl.classList.add('hidden');
    btn.dataset.open = 'false';
    btn.textContent = 'Show context ↕';
    return;
  }
  const before = document.getElementById('ctx-before').value || 5;
  const after  = document.getElementById('ctx-after').value  || 5;
  btn.textContent = 'Loading…'; btn.disabled = true;
  try {
    const res  = await fetch(`/api/context/${id}?before=${before}&after=${after}`);
    if (!res.ok) throw new Error('Failed to load context');
    const msgs = await res.json();
    ctxEl.innerHTML = `
      <div class="border-t bg-slate-50 p-4 space-y-2">
        <p class="text-xs text-gray-500 font-medium mb-3">
          Context — ${msgs.length} messages (${before} before &bull; ${after} after)
        </p>
        ${msgs.map(m => ctxMsg(m)).join('')}
      </div>`;
    ctxEl.classList.remove('hidden');
    btn.dataset.open = 'true';
    btn.textContent = 'Hide context ↕';
  } catch (e) { btn.textContent = 'Show context ↕'; console.error(e); }
  finally { btn.disabled = false; }
}

function ctxMsg(msg) {
  const cls = msg.is_target ? 'ctx-target' : 'ctx-regular';
  const targetBadge = msg.is_target
    ? `<span class="text-xs px-1.5 py-0.5 rounded font-semibold"
             style="background:#fef08a;color:#78350f">★ result</span>` : '';
  const teamBadge = truthy(msg.is_suno_team)
    ? `<span class="text-xs px-1.5 py-0.5 rounded" style="background:#fef3c7;color:#92400e">Team</span>` : '';
  return `
    <div class="${cls} p-3">
      <div class="flex items-center justify-between gap-2 mb-1">
        <div class="flex items-center gap-1.5 flex-wrap">
          <span class="ubadge" style="${usernameStyle(msg.username)}">${esc(msg.username)}</span>
          ${teamBadge}${targetBadge}
        </div>
        <span class="text-xs text-gray-400 shrink-0">${formatDate(msg.date)}</span>
      </div>
      <p class="text-sm text-gray-800 whitespace-pre-wrap break-words">${esc(msg.content)}</p>
    </div>`;
}

/* ══════════════════════════════════════════════════════════════════════════
   EXPORT
══════════════════════════════════════════════════════════════════════════ */
document.getElementById('export-btn').onclick = () => {
  if (!currentResults.length) return;
  const cols = ['id','username','date','content','attachments','reactions',
                'is_suno_team','week','month','author_id','similarity_score'];
  const csv = [
    cols.join(','),
    ...currentResults.map(m =>
      cols.map(c => `"${String(m[c] ?? '').replace(/"/g,'""')}"`).join(',')
    ),
  ].join('\n');
  const a = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(new Blob([csv], {type:'text/csv'})),
    download: `results_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.csv`,
  });
  a.click();
};

/* ══════════════════════════════════════════════════════════════════════════
   BOOKMARKS
══════════════════════════════════════════════════════════════════════════ */

async function loadBookmarkIds() {
  try {
    const ids = await apiFetch('/api/bookmarks/ids');
    bookmarkedIds = new Set(ids);
    updateBmBadge();
  } catch (_) {}
}

function updateBmBadge() {
  const badge = document.getElementById('bm-count-badge');
  if (bookmarkedIds.size > 0) {
    badge.textContent = bookmarkedIds.size;
    badge.classList.remove('hidden');
  } else {
    badge.classList.add('hidden');
  }
}

async function toggleBookmark(msgId) {
  if (bookmarkedIds.has(msgId)) {
    await fetch(`/api/bookmarks/by-msg/${msgId}`, { method: 'DELETE' });
    bookmarkedIds.delete(msgId);
  } else {
    await fetch('/api/bookmarks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ msg_id: msgId }),
    });
    bookmarkedIds.add(msgId);
  }
  updateBmBadge();

  // Re-render the button in the card without re-rendering all results
  const card = document.getElementById(`card-${msgId}`);
  if (card) {
    const btn = card.querySelector('.bm-toggle');
    if (btn) {
      const isNow = bookmarkedIds.has(msgId);
      btn.title = isNow ? 'Remove bookmark' : 'Save bookmark';
      btn.innerHTML = isNow
        ? `<svg class="w-3.5 h-3.5 text-amber-500" fill="currentColor" viewBox="0 0 24 24"><path d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"/></svg><span class="text-amber-600">Bookmarked</span>`
        : `<svg class="w-3.5 h-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"/></svg><span class="text-gray-500">Bookmark</span>`;
    }
  }
}

// Event delegation for bookmark buttons in results
document.getElementById('results-container').addEventListener('click', async e => {
  const btn = e.target.closest('.bm-toggle');
  if (!btn) return;
  btn.disabled = true;
  await toggleBookmark(parseInt(btn.dataset.id));
  btn.disabled = false;
});

function _sortBookmarks(bms) {
  const mode = document.getElementById('bm-sort').value;
  const sorted = [...bms];
  if (mode === 'date') {
    sorted.sort((a, b) => new Date(a.date) - new Date(b.date));
  } else if (mode === 'username') {
    sorted.sort((a, b) => (a.username || '').localeCompare(b.username || ''));
  } else {
    // 'added' — sort by bookmark_id ascending (insertion order)
    sorted.sort((a, b) => a.bookmark_id - b.bookmark_id);
  }
  return sorted;
}

let _cachedBookmarks = [];

async function loadBookmarksPage() {
  const container = document.getElementById('bookmarks-container');
  container.innerHTML = '<p class="text-sm text-gray-400 text-center py-8">Loading…</p>';
  await loadAllLabels();
  try {
    _cachedBookmarks = await apiFetch('/api/bookmarks');
    _renderBookmarksSorted();
  } catch (e) {
    container.innerHTML = `<p class="text-sm text-red-500 text-center py-8">Failed to load bookmarks: ${esc(e.message)}</p>`;
  }
}

// ── Label colour helpers ──────────────────────────────────────────────────────
function labelTextColor(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.55 ? '#1f2937' : '#ffffff';
}

// ── Load & render label filter chips ─────────────────────────────────────────
async function loadAllLabels() {
  try {
    _allLabels = await apiFetch('/api/labels');
  } catch (_) { _allLabels = []; }
  renderBmLabelFilterChips();
}

function renderBmLabelFilterChips() {
  const container = document.getElementById('bm-label-filter-chips');
  if (!_allLabels.length) {
    container.innerHTML = '<span class="text-xs text-gray-400">No labels yet. Create labels in Config → Manage Labels.</span>';
    return;
  }
  container.innerHTML = _allLabels.map(l => {
    const active = _bmLabelFilter.has(l.id);
    const bg     = active ? l.color : '#f1f5f9';
    const tc     = active ? labelTextColor(l.color) : '#64748b';
    const border = active ? l.color : '#e2e8f0';
    return `<button class="bm-label-filter-chip text-xs px-2.5 py-0.5 rounded-full font-medium border transition-all"
                    data-label-id="${l.id}"
                    style="background:${bg};color:${tc};border-color:${border}">
              ${esc(l.name)}
            </button>`;
  }).join('');
}

// ── Bookmark label chips inside card ─────────────────────────────────────────
function _bmLabelChipsHtml(bm) {
  const chips = (bm.labels || []).map(l => {
    const tc = labelTextColor(l.color);
    return `<span class="bm-label-chip text-xs px-2 py-0.5 rounded-full font-medium cursor-pointer select-none"
                  style="background:${l.color};color:${tc}"
                  data-bm-id="${bm.bookmark_id}" data-label-id="${l.id}"
                  title="Remove label">${esc(l.name)} ×</span>`;
  }).join('');
  return chips + `<button class="bm-label-btn text-xs text-gray-400 hover:text-indigo-600 border border-dashed border-gray-300 hover:border-indigo-400 rounded-full px-2 py-0.5 transition-colors"
                          data-bm-id="${bm.bookmark_id}">+ label</button>`;
}

// ── Render the inline label picker panel ─────────────────────────────────────
function _renderBmLabelPanel(bookmarkId) {
  const panel = document.getElementById(`bm-label-panel-${bookmarkId}`);
  const bm = _cachedBookmarks.find(b => b.bookmark_id === bookmarkId);
  if (!panel || !bm) return;
  const assignedIds = new Set((bm.labels || []).map(l => l.id));
  const existingChips = _allLabels.length
    ? `<div class="flex flex-wrap gap-1.5 mb-2">
        ${_allLabels.map(l => {
          const active = assignedIds.has(l.id);
          const bg     = active ? l.color : '#f8fafc';
          const tc     = active ? labelTextColor(l.color) : '#64748b';
          const border = active ? l.color : '#cbd5e1';
          return `<button class="bm-label-toggle text-xs px-2.5 py-0.5 rounded-full font-medium border transition-all"
                          data-bm-id="${bookmarkId}" data-label-id="${l.id}"
                          style="background:${bg};color:${tc};border-color:${border}">
                    ${active ? '✓ ' : ''}${esc(l.name)}
                  </button>`;
        }).join('')}
      </div>`
    : '';
  panel.innerHTML = `
    <div class="p-2">
      ${existingChips}
      <div class="flex items-center gap-1.5 pt-1 border-t border-gray-100">
        <input class="bm-new-label-input flex-1 min-w-0 border rounded-lg px-2 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-300"
               placeholder="New label name…" maxlength="40" data-bm-id="${bookmarkId}" />
        <input class="bm-new-label-color w-7 h-7 rounded cursor-pointer border border-gray-200 p-0.5"
               type="color" value="#6366f1" data-bm-id="${bookmarkId}" title="Pick colour" />
        <button class="bm-new-label-create text-xs px-2.5 py-1 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 shrink-0"
                data-bm-id="${bookmarkId}">Add</button>
      </div>
    </div>`;
}

// ── filter bookmarks ─────────────────────────────────────────────────────────
function _filterBookmarks(bms) {
  const userTerm = (document.getElementById('bm-filter-user').value || '').trim().toLowerCase();
  const sunoMode = document.getElementById('bm-filter-suno').value;
  const textRaw  = (document.getElementById('bm-filter-text').value || '').trim();

  // Build whole-word regexes for the text search term
  const textWords   = textRaw ? textRaw.split(/\s+/).filter(Boolean) : [];
  const textRegexes = textWords.map(w => new RegExp('\\b' + _escapeRegex(w) + '\\b', 'i'));
  const textPhrase  = textWords.length > 1
    ? new RegExp('\\b' + textWords.map(_escapeRegex).join('\\s+') + '\\b', 'i')
    : null;

  return bms.filter(bm => {
    if (userTerm && !(bm.username || '').toLowerCase().includes(userTerm)) return false;
    if (sunoMode === 'only'    && !truthy(bm.is_suno_team)) return false;
    if (sunoMode === 'exclude' &&  truthy(bm.is_suno_team)) return false;
    if (_bmLabelFilter.size > 0) {
      const bmLabelIds = new Set((bm.labels || []).map(l => l.id));
      const hasMatch = [..._bmLabelFilter].some(id => bmLabelIds.has(id));
      if (!hasMatch) return false;
    }
    if (textRegexes.length > 0) {
      const hay = (bm.username || '') + ' ' + (bm.content || '');
      // Phrase match or all individual words must match
      const ok = textPhrase
        ? textPhrase.test(hay)
        : textRegexes.every(rx => rx.test(hay));
      if (!ok) return false;
    }
    return true;
  });
}

function _renderBookmarksSorted() {
  const container = document.getElementById('bookmarks-container');
  if (!_cachedBookmarks.length) {
    container.innerHTML = '<p class="text-sm text-gray-400 text-center py-8">No bookmarks yet. Use the bookmark button on any search result.</p>';
    return;
  }
  const filtered = _filterBookmarks(_sortBookmarks(_cachedBookmarks));
  if (!filtered.length) {
    container.innerHTML = '<p class="text-sm text-gray-400 text-center py-8">No bookmarks match the current filters.</p>';
    return;
  }
  container.innerHTML = filtered.map(bm => bookmarkCard(bm)).join('');
}

function bookmarkCard(bm) {
  const score = bm.similarity_score !== undefined && bm.similarity_score !== null
    ? `<span class="text-xs px-2 py-0.5 rounded-full" style="background:#eef2ff;color:#3730a3">${bm.similarity_score}</span>` : '';
  const teamBadge = truthy(bm.is_suno_team)
    ? `<span class="text-xs px-2 py-0.5 rounded-full font-medium" style="background:#fef3c7;color:#92400e">Suno Team</span>` : '';
  const src = allUploads.find(u => u.id === bm.upload_id);
  const srcLabel = src
    ? `<span class="text-[10px] text-gray-400 truncate max-w-[12rem]" title="${esc(bm.upload_id)}">${esc(src.filename)}</span>` : '';
  const savedAt = new Date(bm.created_at).toLocaleString();

  return `
    <div id="bm-card-${bm.bookmark_id}" class="bg-white rounded-2xl shadow border border-amber-100 overflow-hidden">
      <div class="bg-amber-50 px-4 py-1.5 flex items-center justify-between gap-2 border-b border-amber-100">
        <span class="text-xs text-amber-700 flex items-center gap-1">
          <svg class="w-3 h-3" fill="currentColor" viewBox="0 0 24 24"><path d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"/></svg>
          Saved ${savedAt}
        </span>
        <button class="bm-remove text-xs text-red-400 hover:text-red-600 font-medium"
                data-bm-id="${bm.bookmark_id}" data-msg-id="${bm.id}">
          Remove
        </button>
      </div>
      <div class="p-4">
        <div class="flex items-start justify-between gap-2 mb-2">
          <div class="flex flex-col gap-0.5">
            <div class="flex items-center flex-wrap gap-1.5">
              <span class="ubadge" style="${usernameStyle(bm.username)}">${esc(bm.username)}</span>
              ${teamBadge}${score}
            </div>
            <span class="text-xs text-gray-400">${formatDate(bm.date)}</span>
          </div>
          ${srcLabel}
        </div>
        <p class="text-sm leading-relaxed text-gray-800 whitespace-pre-wrap break-words">${esc(bm.content)}</p>
        ${hasContent(bm.attachments) ? `<p class="text-xs text-gray-500 mt-1">📎 ${esc(bm.attachments)}</p>` : ''}
        ${hasContent(bm.reactions)   ? `<p class="text-xs text-gray-500 mt-1">💬 ${esc(bm.reactions)}</p>`   : ''}
        <!-- Labels -->
        <div class="flex items-center flex-wrap gap-1 mt-2 min-h-[1.5rem]" id="bm-labels-${bm.bookmark_id}">
          ${_bmLabelChipsHtml(bm)}
        </div>
        <!-- Inline label picker (hidden until opened) -->
        <div id="bm-label-panel-${bm.bookmark_id}" class="hidden mt-1 border border-dashed border-gray-200 rounded-xl bg-gray-50"></div>
      </div>
      <div class="border-t bg-gray-50 px-4 py-2 flex justify-end">
        <button class="bm-ctx-toggle text-xs text-indigo-600 hover:text-indigo-800 font-medium"
                data-id="${bm.id}" data-open="false">
          Show context ↕
        </button>
      </div>
      <div id="bmctx-${bm.id}" class="hidden"></div>
    </div>`;
}

// Event delegation for bookmarks page
document.getElementById('bookmarks-container').addEventListener('click', async e => {
  // Remove bookmark
  const removeBtn = e.target.closest('.bm-remove');
  if (removeBtn) {
    const bmId  = parseInt(removeBtn.dataset.bmId);
    const msgId = parseInt(removeBtn.dataset.msgId);
    removeBtn.disabled = true;
    await fetch(`/api/bookmarks/${bmId}`, { method: 'DELETE' });
    bookmarkedIds.delete(msgId);
    updateBmBadge();
    document.getElementById(`bm-card-${bmId}`)?.remove();
    // Update star in results if visible
    const card = document.getElementById(`card-${msgId}`);
    if (card) {
      const btn = card.querySelector('.bm-toggle');
      if (btn) {
        btn.title = 'Save bookmark';
        btn.innerHTML = `<svg class="w-3.5 h-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"/></svg><span class="text-gray-500">Bookmark</span>`;
      }
    }
    const container = document.getElementById('bookmarks-container');
    if (!container.querySelector('[id^="bm-card-"]')) {
      container.innerHTML = '<p class="text-sm text-gray-400 text-center py-8">No bookmarks yet. Use the bookmark button on any search result.</p>';
    }
    return;
  }

  // Remove label chip (× click on assigned label)
  const labelChip = e.target.closest('.bm-label-chip');
  if (labelChip) {
    const bmId    = parseInt(labelChip.dataset.bmId);
    const labelId = parseInt(labelChip.dataset.labelId);
    await fetch(`/api/bookmarks/${bmId}/labels/${labelId}`, { method: 'DELETE' });
    const bm = _cachedBookmarks.find(b => b.bookmark_id === bmId);
    if (bm) bm.labels = (bm.labels || []).filter(l => l.id !== labelId);
    const labelsRow = document.getElementById(`bm-labels-${bmId}`);
    if (labelsRow) labelsRow.innerHTML = _bmLabelChipsHtml(bm);
    return;
  }

  // Open/close label picker panel
  const labelBtn = e.target.closest('.bm-label-btn');
  if (labelBtn) {
    const bmId = parseInt(labelBtn.dataset.bmId);
    const panel = document.getElementById(`bm-label-panel-${bmId}`);
    if (panel.classList.contains('hidden')) {
      _renderBmLabelPanel(bmId);
      panel.classList.remove('hidden');
    } else {
      panel.classList.add('hidden');
    }
    return;
  }

  // Toggle label assignment in picker panel
  const labelToggle = e.target.closest('.bm-label-toggle');
  if (labelToggle) {
    const bmId    = parseInt(labelToggle.dataset.bmId);
    const labelId = parseInt(labelToggle.dataset.labelId);
    const bm      = _cachedBookmarks.find(b => b.bookmark_id === bmId);
    if (!bm) return;
    const isAssigned = (bm.labels || []).some(l => l.id === labelId);
    if (isAssigned) {
      await fetch(`/api/bookmarks/${bmId}/labels/${labelId}`, { method: 'DELETE' });
      bm.labels = (bm.labels || []).filter(l => l.id !== labelId);
    } else {
      await fetch(`/api/bookmarks/${bmId}/labels/${labelId}`, { method: 'POST' });
      const label = _allLabels.find(l => l.id === labelId);
      if (label) bm.labels = [...(bm.labels || []), { id: label.id, name: label.name, color: label.color }];
    }
    // Update chips row and re-render panel
    const labelsRow = document.getElementById(`bm-labels-${bmId}`);
    if (labelsRow) labelsRow.innerHTML = _bmLabelChipsHtml(bm);
    _renderBmLabelPanel(bmId);
    return;
  }

  // Create new label inline and assign it
  const createBtn = e.target.closest('.bm-new-label-create');
  if (createBtn) {
    const bmId  = parseInt(createBtn.dataset.bmId);
    const panel = document.getElementById(`bm-label-panel-${bmId}`);
    const nameInput  = panel.querySelector('.bm-new-label-input');
    const colorInput = panel.querySelector('.bm-new-label-color');
    const name  = (nameInput?.value || '').trim();
    const color = colorInput?.value || '#6366f1';
    if (!name) { nameInput?.focus(); return; }
    createBtn.disabled = true;
    createBtn.textContent = '…';
    try {
      // Create the label
      const labelRes = await fetch('/api/labels', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, color }),
      });
      const newLabel = await labelRes.json();
      if (!labelRes.ok) {
        // Label may already exist — find it in _allLabels
        const existing = _allLabels.find(l => l.name.toLowerCase() === name.toLowerCase());
        if (!existing) { createBtn.disabled = false; createBtn.textContent = 'Add'; return; }
        newLabel.id = existing.id; newLabel.name = existing.name; newLabel.color = existing.color;
      } else {
        // Add to global cache, keep sorted
        _allLabels = [..._allLabels, newLabel].sort((a, b) =>
          a.name.localeCompare(b.name, undefined, { sensitivity: 'base' })
        );
        renderBmLabelFilterChips();
      }
      // Assign to this bookmark
      await fetch(`/api/bookmarks/${bmId}/labels/${newLabel.id}`, { method: 'POST' });
      const bm = _cachedBookmarks.find(b => b.bookmark_id === bmId);
      if (bm && !(bm.labels || []).some(l => l.id === newLabel.id)) {
        bm.labels = [...(bm.labels || []), { id: newLabel.id, name: newLabel.name, color: newLabel.color }];
      }
      const labelsRow = document.getElementById(`bm-labels-${bmId}`);
      if (labelsRow) labelsRow.innerHTML = _bmLabelChipsHtml(bm);
      _renderBmLabelPanel(bmId);
    } catch (_) {
      createBtn.disabled = false;
      createBtn.textContent = 'Add';
    }
    return;
  }

  // Toggle context in bookmark card
  const ctxBtn = e.target.closest('.bm-ctx-toggle');
  if (!ctxBtn) return;
  const id     = parseInt(ctxBtn.dataset.id);
  const ctxEl  = document.getElementById(`bmctx-${id}`);

  if (ctxBtn.dataset.open === 'true') {
    ctxEl.classList.add('hidden');
    ctxBtn.dataset.open = 'false';
    ctxBtn.textContent  = 'Show context ↕';
    return;
  }

  // Read current global controls at click-time
  const before = parseInt(document.getElementById('bm-ctx-before').value) || 5;
  const after  = parseInt(document.getElementById('bm-ctx-after').value)  || 5;

  ctxBtn.textContent = 'Loading…';
  ctxBtn.disabled    = true;
  try {
    const msgs = await apiFetch(`/api/context/${id}?before=${before}&after=${after}`);
    ctxEl.innerHTML = `
      <div class="border-t bg-slate-50 p-4 space-y-2">
        <p class="text-xs text-gray-500 font-medium mb-3">
          Context — ${msgs.length} messages (${before} before &bull; ${after} after)
        </p>
        ${msgs.map(m => ctxMsg(m)).join('')}
      </div>`;
    ctxEl.classList.remove('hidden');
    ctxBtn.dataset.open = 'true';
    ctxBtn.textContent  = 'Hide context ↕';
  } catch (_) {
    ctxBtn.textContent = 'Show context ↕';
  } finally {
    ctxBtn.disabled = false;
  }
});

document.getElementById('bm-refresh-btn').addEventListener('click', loadBookmarksPage);

/* Collapse all open bookmark context panels when ctx values change (no re-render, no network) */
function _collapseAllBmContext() {
  document.querySelectorAll('.bm-ctx-toggle[data-open="true"]').forEach(btn => {
    const ctxEl = document.getElementById(`bmctx-${btn.dataset.id}`);
    if (ctxEl) ctxEl.classList.add('hidden');
    btn.dataset.open = 'false';
    btn.textContent  = 'Show context ↕';
  });
}
document.getElementById('bm-ctx-before').addEventListener('change', _collapseAllBmContext);
document.getElementById('bm-ctx-after') .addEventListener('change', _collapseAllBmContext);
document.getElementById('bm-sort').addEventListener('change', () => {
  _collapseAllBmContext();
  _renderBookmarksSorted();
});
document.getElementById('bm-filter-suno').addEventListener('change', () => {
  _collapseAllBmContext();
  _renderBookmarksSorted();
});

// Enter key in inline label name input → click the Add button
document.getElementById('bookmarks-container').addEventListener('keydown', e => {
  if (e.key !== 'Enter') return;
  const input = e.target.closest('.bm-new-label-input');
  if (!input) return;
  e.preventDefault();
  const bmId = input.dataset.bmId;
  document.querySelector(`.bm-new-label-create[data-bm-id="${bmId}"]`)?.click();
});

// Label filter chip toggle
document.getElementById('bm-label-filter-chips').addEventListener('click', e => {
  const chip = e.target.closest('.bm-label-filter-chip');
  if (!chip) return;
  const id = parseInt(chip.dataset.labelId);
  if (_bmLabelFilter.has(id)) _bmLabelFilter.delete(id);
  else _bmLabelFilter.add(id);
  renderBmLabelFilterChips();
  _collapseAllBmContext();
  _renderBookmarksSorted();
});
let _bmUserFilterDebounce = null;
document.getElementById('bm-filter-user').addEventListener('input', () => {
  clearTimeout(_bmUserFilterDebounce);
  _bmUserFilterDebounce = setTimeout(() => {
    _collapseAllBmContext();
    _renderBookmarksSorted();
  }, 250);
});

let _bmTextFilterDebounce = null;
document.getElementById('bm-filter-text').addEventListener('input', () => {
  clearTimeout(_bmTextFilterDebounce);
  _bmTextFilterDebounce = setTimeout(() => {
    _collapseAllBmContext();
    _renderBookmarksSorted();
  }, 250);
});

/* ══════════════════════════════════════════════════════════════════════════
   HYBRID SUMMARY PAGE
══════════════════════════════════════════════════════════════════════════ */

/* ── Date mode toggle ── */
document.getElementById('sum-mode-exact').addEventListener('click', () => {
  document.getElementById('sum-mode-exact').classList.add('range-mode-active');
  document.getElementById('sum-mode-month').classList.remove('range-mode-active');
  document.getElementById('sum-exact-inputs').classList.remove('hidden');
  document.getElementById('sum-month-inputs').classList.add('hidden');
});
document.getElementById('sum-mode-month').addEventListener('click', () => {
  document.getElementById('sum-mode-month').classList.add('range-mode-active');
  document.getElementById('sum-mode-exact').classList.remove('range-mode-active');
  document.getElementById('sum-month-inputs').classList.remove('hidden');
  document.getElementById('sum-exact-inputs').classList.add('hidden');
});

/* ── Follow-up chat state ── */
// Index 0 is always the initial summary (assistant turn).
// Subsequent turns are user/assistant pairs for follow-up Q&A.
let sumFollowUpHistory = [];
// Filter params stored after a successful summary so follow-up
// can search the same filtered message pool.
let sumLastFilterParams = null;

/* ── Form collapse / expand ── */
function _buildCompactInfo(p) {
  if (!p) return '—';
  const parts = [];
  if (p.date_from || p.date_to) {
    parts.push(`${p.date_from || '…'} → ${p.date_to || '…'}`);
  }
  if (p.username) parts.push(`@${p.username}`);
  if (p.min_words) parts.push(`≥${p.min_words} words`);
  if (p.suno_team !== 'all') parts.push(p.suno_team === 'only' ? 'Suno Team only' : 'excl. Suno Team');
  return parts.length ? parts.join(' · ') : 'All messages';
}

function collapseSumForm(filterParams) {
  document.getElementById('sum-compact-info').textContent = _buildCompactInfo(filterParams);
  document.getElementById('sum-form-full').classList.add('hidden');
  document.getElementById('sum-form-compact').classList.remove('hidden');
}

function expandSumForm() {
  document.getElementById('sum-form-compact').classList.add('hidden');
  document.getElementById('sum-form-full').classList.remove('hidden');
}

document.getElementById('sum-form-expand').addEventListener('click', expandSumForm);
document.getElementById('sum-form-collapse').addEventListener('click', () => collapseSumForm(sumLastFilterParams));

/* ── Process log ── */
const LOG_ICONS = {
  filter:    '🔍',
  retrieval: '📡',
  dedup:     '🧹',
  cluster:   '🔮',
  sample:    '🎯',
  llm:       '✨',
  fallback:  '⚠️',
};

function renderProcessLogEntry(entry) {
  const logEl = document.getElementById('sum-process-log');
  const div = document.createElement('div');
  div.className = `log-entry log-step-${entry.step || 'fallback'}`;
  const icon  = LOG_ICONS[entry.step] || '•';
  const label = entry.label || entry.step || '';
  const msg   = entry.msg   || '';
  div.innerHTML =
    `<span class="log-icon">${icon}</span>` +
    `<span class="log-label">${esc(label)}</span>` +
    `<span class="log-msg">${esc(msg)}</span>`;
  logEl.appendChild(div);
}

/* ── Log panel toggle ── */
document.getElementById('sum-log-toggle').addEventListener('click', () => {
  const logEl    = document.getElementById('sum-process-log');
  const toggleEl = document.getElementById('sum-log-toggle');
  const hidden   = logEl.classList.toggle('hidden');
  toggleEl.innerHTML = hidden ? '&#9660; Show' : '&#9650; Hide';
});

/* ── Follow-up bubble helpers ── */
function appendFollowUpUserBubble(text) {
  const container = document.getElementById('sum-followup-history');
  const wrap = document.createElement('div');
  wrap.className = 'flex justify-end';
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble-user';
  bubble.textContent = text;
  wrap.appendChild(bubble);
  container.appendChild(wrap);
  container.scrollTop = container.scrollHeight;
}

function appendFollowUpAssistantBubble(text) {
  const container = document.getElementById('sum-followup-history');
  const wrap = document.createElement('div');
  wrap.className = 'flex justify-start';
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble-assistant markdown-body';
  bubble.textContent = text;
  wrap.appendChild(bubble);
  container.appendChild(wrap);
  container.scrollTop = container.scrollHeight;
  return bubble; // caller streams content into bubble.innerHTML
}

/* ── Generate Hybrid Summary ── */
async function doSummarize() {
  const btn      = document.getElementById('sum-btn');
  const resultEl = document.getElementById('sum-result');
  const logEl    = document.getElementById('sum-process-log');

  const username    = document.getElementById('sum-username').value.trim();
  const isMonthMode = document.getElementById('sum-mode-month').classList.contains('range-mode-active');
  const minW        = parseInt(document.getElementById('sum-min-words').value) || 0;
  const suno        = document.getElementById('sum-suno').value;
  const prompt      = document.getElementById('sum-prompt').value.trim();
  const scope       = getScopeParam();
  const sumModel    = document.getElementById('sum-model').value;

  let dateFrom = '', dateTo = '';
  if (isMonthMode) {
    const mFrom = document.getElementById('sum-month-from').value;
    const mTo   = document.getElementById('sum-month-to').value;
    if (mFrom) dateFrom = mFrom + '-01';
    if (mTo) {
      const parts = mTo.split('-');
      const lastDay = new Date(parseInt(parts[0]), parseInt(parts[1]), 0).getDate();
      dateTo = mTo + '-' + String(lastDay).padStart(2, '0');
    }
  } else {
    dateFrom = document.getElementById('sum-date-from').value;
    dateTo   = document.getElementById('sum-date-to').value;
  }

  btn.disabled = true;
  btn.textContent = 'Summarizing…';

  // Show results panel and reset its contents
  document.getElementById('sum-results-panel').classList.remove('hidden');
  logEl.innerHTML = '';
  resultEl.innerHTML = '';
  // Make sure log panel is visible
  logEl.classList.remove('hidden');
  document.getElementById('sum-log-toggle').innerHTML = '&#9650; Hide';

  // Reset follow-up section whenever a new summary starts
  document.getElementById('sum-followup-section').classList.add('hidden');
  document.getElementById('sum-followup-history').innerHTML = '';
  sumFollowUpHistory = [];
  sumLastFilterParams = null;

  // Show collapse button while generating
  document.getElementById('sum-form-collapse').classList.remove('hidden');

  try {
    const currentParams = {
      username:   username  || null,
      date_from:  dateFrom  || null,
      date_to:    dateTo    || null,
      upload_ids: scope ? scope.split(',') : [],
      min_words:  minW,
      suno_team:  suno,
      model:      sumModel,
    };

    const res = await fetch('/api/summarize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...currentParams, prompt: prompt || null }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(err.detail || 'Request failed');
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let summaryText = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (raw === '[DONE]') break;
        try {
          const delta = JSON.parse(raw);
          if (delta.type === 'log') {
            renderProcessLogEntry(delta);
          } else if (delta.content) {
            summaryText += delta.content;
            resultEl.innerHTML = marked.parse(summaryText);
          } else if (delta.error) {
            throw new Error(delta.error);
          }
        } catch (parseErr) {
          if (!(parseErr instanceof SyntaxError)) throw parseErr;
        }
      }
    }

    if (!summaryText) {
      showErrorPopup('No response received from the model. The model may be unavailable or the request was rejected. Check your API key and selected model.');
      return;
    }

    // Persist filter params for follow-up retrieval against the same pool
    sumLastFilterParams = currentParams;
    // Seed history with the summary so follow-up LLM has full context
    sumFollowUpHistory = [{ role: 'assistant', content: summaryText }];

    // Reveal follow-up section
    document.getElementById('sum-followup-section').classList.remove('hidden');

    // Collapse the form so results are easily readable
    collapseSumForm(currentParams);

  } catch (e) {
    showErrorPopup(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate Hybrid Summary';
  }
}

document.getElementById('sum-btn').addEventListener('click', doSummarize);

/* ── Follow-up Q&A ── */
async function sendSumFollowUp() {
  const input    = document.getElementById('sum-followup-input');
  const sendBtn  = document.getElementById('sum-followup-send');
  const question = input.value.trim();

  if (!question) return;
  if (!sumLastFilterParams) {
    showErrorPopup('Generate a summary first before asking follow-up questions.');
    return;
  }

  input.value     = '';
  input.disabled  = true;
  sendBtn.disabled = true;
  sendBtn.textContent = '…';

  sumFollowUpHistory.push({ role: 'user', content: question });
  appendFollowUpUserBubble(question);

  const bubble = appendFollowUpAssistantBubble('');
  let answerText = '';

  try {
    const res = await fetch('/api/summarize/followup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        // History excludes the current question (already in 'question' field)
        history: sumFollowUpHistory.slice(0, -1),
        ...sumLastFilterParams,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(err.detail || 'Request failed');
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (raw === '[DONE]') break;
        try {
          const delta = JSON.parse(raw);
          if (delta.type === 'log') {
            // Follow-up log events are silently consumed (not shown in main log)
          } else if (delta.content) {
            answerText += delta.content;
            bubble.innerHTML = marked.parse(answerText);
            document.getElementById('sum-followup-history').scrollTop =
              document.getElementById('sum-followup-history').scrollHeight;
          } else if (delta.error) {
            throw new Error(delta.error);
          }
        } catch (parseErr) {
          if (!(parseErr instanceof SyntaxError)) throw parseErr;
        }
      }
    }

    sumFollowUpHistory.push({ role: 'assistant', content: answerText });

  } catch (e) {
    bubble.remove();
    sumFollowUpHistory.pop(); // roll back the user turn
    showErrorPopup(e.message);
  } finally {
    input.disabled   = false;
    sendBtn.disabled = false;
    sendBtn.textContent = 'Ask';
    input.focus();
  }
}

document.getElementById('sum-followup-send').addEventListener('click', sendSumFollowUp);
document.getElementById('sum-followup-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && e.ctrlKey) { e.preventDefault(); sendSumFollowUp(); }
});
document.getElementById('sum-followup-clear').addEventListener('click', () => {
  // Keep index-0 (initial summary) so follow-up context is preserved
  const init = sumFollowUpHistory[0];
  sumFollowUpHistory = init ? [init] : [];
  document.getElementById('sum-followup-history').innerHTML = '';
});

/* ── PDF Export ── */
function exportSummaryPDF() {
  const summaryHTML = document.getElementById('sum-result').innerHTML;
  const dateStr     = new Date().toLocaleDateString('en-US', {
    year: 'numeric', month: 'long', day: 'numeric',
  });

  // Build Q&A section from history (skip index 0 = initial summary)
  let qaHTML = '';
  const qaHistory = sumFollowUpHistory.slice(1);
  for (const turn of qaHistory) {
    if (turn.role === 'user') {
      const safe = turn.content
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      qaHTML += `<div class="q-block"><span class="q-label">Question</span>${safe}</div>`;
    } else {
      qaHTML += `<div class="a-block">${marked.parse(turn.content)}</div>`;
    }
  }

  // Build a self-contained HTML string and open it via a Blob URL so we
  // avoid the deprecated document.write() API entirely.
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Hybrid Summary \u2013 ${dateStr}</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  max-width: 820px; margin: 40px auto; padding: 0 28px;
  color: #1e293b; line-height: 1.65; font-size: 14px;
}
h1  { font-size: 1.5rem; color: #3730a3; padding-bottom: 10px;
      border-bottom: 2px solid #e2e8f0; margin-bottom: 6px; }
.meta { font-size: 0.75rem; color: #6b7280; margin-bottom: 1.75rem; }
h2  { font-size: 1.15rem; font-weight: 700; color: #1e293b; margin-top: 2rem;
      border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin-bottom: 1rem; }
h3  { font-size: 1rem; font-weight: 600; color: #374151; margin: 1rem 0 0.4rem; }
h4  { font-size: 0.9rem; font-weight: 600; margin: 0.8rem 0 0.3rem; }
p   { margin-bottom: 0.65rem; }
ul, ol { padding-left: 1.4rem; margin-bottom: 0.65rem; }
li  { margin-bottom: 0.2rem; }
blockquote {
  border-left: 3px solid #6366f1; margin: 0.75rem 0;
  padding: 6px 14px; background: #f5f3ff; color: #3730a3;
  border-radius: 0 6px 6px 0; font-style: italic;
}
code {
  background: #f1f5f9; border-radius: 4px;
  padding: 1px 5px; font-size: 0.82em; font-family: monospace;
}
pre  {
  background: #1e293b; color: #e2e8f0; border-radius: 6px;
  padding: 12px; overflow-x: auto; margin-bottom: 0.65rem;
}
pre code { background: none; padding: 0; color: inherit; }
hr   { border: none; border-top: 1px solid #e2e8f0; margin: 1.25rem 0; }
strong { font-weight: 700; }
em     { font-style: italic; }
a      { color: #3730a3; text-decoration: underline; }
table  { border-collapse: collapse; width: 100%; margin-bottom: 0.65rem; font-size: 0.85rem; }
th, td { border: 1px solid #e2e8f0; padding: 6px 10px; text-align: left; }
th     { background: #f8fafc; font-weight: 700; }
.q-block {
  background: #eef2ff; border-radius: 10px 10px 10px 2px;
  padding: 10px 14px; margin: 14px 0 4px; color: #1e1b4b; font-weight: 600;
}
.q-label {
  display: block; font-size: 0.68rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.06em;
  color: #6366f1; margin-bottom: 4px;
}
.a-block {
  background: #f8fafc; border-left: 3px solid #94a3b8;
  border-radius: 0 10px 10px 0; padding: 10px 14px; margin: 4px 0 14px;
}
@media print { body { margin: 16px 28px; } }
</style>
</head>
<body>
<h1>Hybrid Summary</h1>
<p class="meta">Exported ${dateStr}</p>
<div class="summary-body">${summaryHTML}</div>
${qaHTML ? '<h2>Follow-up Q&amp;A</h2><div class="qa-body">' + qaHTML + '</div>' : ''}
<script>window.onload = function() { window.print(); };<\/script>
</body>
</html>`;

  const blob = new Blob([html], { type: 'text/html' });
  const url  = URL.createObjectURL(blob);
  const win  = window.open(url, '_blank', 'width=920,height=750');
  if (!win) {
    URL.revokeObjectURL(url);
    showErrorPopup('Pop-up blocked. Please allow pop-ups for this page, then try again.');
    return;
  }
  // Release the object URL once the new window has loaded the document
  win.addEventListener('load', () => URL.revokeObjectURL(url), { once: true });
}

document.getElementById('sum-export-pdf').addEventListener('click', exportSummaryPDF);

/* ══════════════════════════════════════════════════════════════════════════
   INIT
══════════════════════════════════════════════════════════════════════════ */
(async () => {
  await refreshUploads();   // loads allUploads, renders scope chips, stats
  await loadBookmarkIds();  // populate bookmarkedIds set + badge

  // Restore API key from localStorage → send to server, then check stats.
  // If no key is stored yet, show the popup so the user can enter one.
  const storedKey = localStorage.getItem(STORAGE_KEY);
  if (storedKey) {
    try {
      await _sendKeyToServer(storedKey);
      loadStats();
    } catch (_) {
      // Stored key is invalid/rejected — clear it and prompt again
      localStorage.removeItem(STORAGE_KEY);
      showApiKeyPopup(false);
    }
  } else {
    showApiKeyPopup(false);
  }
})();
