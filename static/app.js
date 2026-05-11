// -- MARKED CONFIGURATION --------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  if (typeof marked !== 'undefined') {
    marked.use({ gfm: true, breaks: true });
  }
});

/*
   FETCH HELPER
   Wraps fetch() so non-JSON responses (proxy 502/504, nginx error pages,
   HTTP→HTTPS redirects) produce a clear error instead of "Unexpected token '<'".
 */
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

// -- ERROR POPUP -----------------------------------------------------------
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

// -- APP MODE --------------------------------------------------------------
const APP_MODE     = document.querySelector('meta[name="app-mode"]')?.content    || 'single';
const CURRENT_USER = document.querySelector('meta[name="current-user"]')?.content || '';
let   currentUserIsAdmin = APP_MODE !== 'multi';  // single mode = always admin

// -- STATE -----------------------------------------------------------------
let currentResults   = [];
let currentSearchType = '';
let currentKeyword   = '';
let _trendBucket = 'month';  // 'month' | 'week' | 'day'
let _trendChart  = null;     // Chart.js instance
let srFollowUpHistory = [];
let srSummaryText     = '';
let srLastPrompt      = '';
let allUploads       = [];           // [{id, filename, row_count, upload_time, embedded_models}]
let selectedUploadIds = new Set();   // empty = search all
let bookmarkedIds    = new Set();    // msg ids that are bookmarked
let activeFilterTokens = [];         // tokens to highlight in filter results (exact mode)
let _allCodes       = [];           // all defined codes [{id, name, color, description, category_id}]
let _bmCodeFilter   = new Set();    // code IDs selected as bookmark filter

// -- USERNAME COLOUR PALETTE -----------------------------------------------
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

// -- UTILITIES -------------------------------------------------------------
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

// -- PAGE NAVIGATION -------------------------------------------------------
function navigateTo(page) {
  document.querySelectorAll('.nav-tab').forEach(b => b.classList.toggle('nav-active', b.dataset.page === page));
  document.getElementById('page-search').classList.toggle('hidden', page !== 'search');
  document.getElementById('page-settings').classList.toggle('hidden', page !== 'settings');
  document.getElementById('page-bookmarks').classList.toggle('hidden', page !== 'bookmarks');
  document.getElementById('page-coding').classList.toggle('hidden', page !== 'coding');
  document.getElementById('page-admin')?.classList.toggle('hidden', page !== 'admin');
  if (page === 'settings')  loadSettingsPage();
  if (page === 'bookmarks') loadBookmarksPage();
  if (page === 'coding')    loadCodingPage();
  if (page === 'admin')     loadAdminPage();
}

document.querySelectorAll('.nav-tab').forEach(btn => {
  btn.addEventListener('click', () => navigateTo(btn.dataset.page));
});

// -- STATS -----------------------------------------------------------------
async function loadStats() {
  try {
    const d = await apiFetch('/api/stats');
    document.getElementById('stats-bar').innerHTML =
      `${d.total_messages.toLocaleString()} msgs &bull; ` +
      `${d.total_uploads} uploads &bull; ` +
      `${d.embedded_messages.toLocaleString()} embedded &bull; ` +
      `<span style="color:#c4b5fd">${esc(d.current_model_label)}</span>` +
      (d.api_key_set ? ' &bull; <span style="color:#86efac">API key ...</span>' : '');
  } catch (_) {}
}

// -- SCOPE SELECTOR (Search page - which uploads to search) ----------------
function renderScopeChips() {
  const container = document.getElementById('scope-chips');
  if (!allUploads.length) {
    container.innerHTML = '<span class="text-xs text-gray-400 italic">No uploads yet€” go to the Data page to add data.</span>';
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
    btn.addEventListener('click', () => toggleScopeChip(btn.dataset.id));
  });
}

function toggleScopeChip(uploadId) {
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

// -- SETTINGS PAGE ---------------------------------------------------------
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
            <p class="text-xs text-gray-400">${m.dims}-dim· ${m.embedded_count.toLocaleString()} msgs embedded</p>
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
            (isLocal && cnt === 0 ? ' Weights will download on first use (~0.4â€“1.3 GB).' : '');
          msg.classList.remove('hidden');
          loadStats();
          loadModelOptions();
        } catch (e) { showErrorPopup('Failed to switch model: ' + e.message); }
      });
    });
  } catch (_) {}
}

// -- API KEY POPUP ---------------------------------------------------------
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
  const descEl    = document.getElementById('apikey-popup-desc');

  // Key is always stored in browser localStorage only€” never on the server.
  const stored = localStorage.getItem(STORAGE_KEY) || '';
  input.value = stored;
  if (descEl) descEl.innerHTML = 'Stored in <strong>your browser\'s localStorage</strong> only€” never saved to the server or database. Sent to your own server per session to make OpenAI requests on your behalf.';

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
  // Only dismiss on Esc if dismissable
  if (localStorage.getItem(STORAGE_KEY)) hideApiKeyPopup();
}

function updateSettingsKeyStatus(apiKeySet) {
  const statusEl = document.getElementById('settings-key-status');
  if (!statusEl) return;
  const stored = localStorage.getItem(STORAGE_KEY);
  statusEl.textContent = stored
    ? 'API key saved in your browser (localStorage)€” not stored on the server.'
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
  saveBtn.textContent = 'Saving...';
  errorEl.textContent = '';
  errorEl.classList.add('hidden');

  try {
    await _sendKeyToServer(key);
    localStorage.setItem(STORAGE_KEY, key);
    hideApiKeyPopup();
    updateSettingsKeyStatus(true);
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

// -- DATA PAGE -------------------------------------------------------------

/* -- Upload new CSV -- */
document.getElementById('upload-btn').onclick = async () => {
  const input = document.getElementById('csv-file');
  if (!input.files.length) { setUploadStatus('Please select a CSV file.', 'error'); return; }
  const btn = document.getElementById('upload-btn');
  btn.disabled = true; btn.textContent = 'Starting...';
  hideUploadProgress();
  showUploadProgress(0, 'Starting upload...');
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

/* -- Uploads table -- */
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
    btn.addEventListener('click', () => doReembed(btn.dataset.id, btn));
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
      : `<span class="embed-badge embed-badge-no">${labels[mid] || mid}€”</span>`;
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
        ${currentUserIsAdmin ? `
        <div class="upload-actions-col flex flex-col gap-2 shrink-0">
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
        </div>` : ''}
      </div>
      <!-- Inline re-embed progress -->
      <div id="reembed-progress-${safeId}" class="hidden mt-3">
        <div class="progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
          <div id="reembed-fill-${safeId}" class="progress-fill"></div>
        </div>
        <p id="reembed-label-${safeId}" class="mt-1 text-xs text-gray-600" aria-live="polite">Starting...</p>
      </div>
    </div>`;
}

// Track active poll timers so clicking Re-embed twice doesn't spawn duplicates.
const _reembedTimers = {};

async function doReembed(uploadId, btn) {
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
    if (labelEl) labelEl.textContent = 'Failed€” see details below';
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
  btn.textContent = 'Starting...';
  if (errEl) errEl.classList.add('hidden');
  setProgress(0, 'Submitting job...');

  // -- 1. POST to start the background job ----------------------------------
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
      setProgress(0, 'Job already running€” resuming progress display...');
    } else {
      const skip = d.skipped || 0;
      setProgress(0, skip > 0
        ? `Resuming: ${skip.toLocaleString()} already embedded, checking remainder...`
        : `Job started€” ${(d.total_messages || 0).toLocaleString()} messages queued`);
    }
  } catch (e) {
    setError(`Failed to start job:\n${e.message}`);
    finish();
    return;
  }

  // -- 2. Poll GET /api/jobs/{jobId} every 1.5 s ----------------------------
  btn.textContent = 'Embedding...';

  _reembedTimers[uploadId] = setInterval(async () => {
    let job;
    try {
      const r = await fetch(`/api/jobs/${enc(jobId)}`);
      if (!r.ok) return;   // transient€” keep polling
      job = await r.json();
    } catch {
      return;              // network blip€” keep polling
    }

    const embedded = job.embedded  || 0;
    const total    = job.total     || 0;
    const skipped  = job.skipped   || 0;
    const pct      = total > 0 ? Math.round(embedded / total * 100) : (job.status === 'completed' ? 100 : 0);

    if (job.status === 'running') {
      if (job.phase === 'checking') {
        // During the skip-check phase: show how many are already embedded (grows over time)
        const checkLabel = skipped > 0
          ? `Checking... ${skipped.toLocaleString()} already embedded so far`
          : 'Checking which messages are already embedded...';
        setProgress(0, checkLabel);
      } else {
        const batchInfo = job.current_batch ? ` (batch ${job.current_batch})` : '';
        const skipNote  = skipped > 0 ? `· ${skipped.toLocaleString()} skipped` : '';
        setProgress(pct, `Embedding... ${pct}%€” ${embedded.toLocaleString()}/${total.toLocaleString()} new messages${skipNote}${batchInfo}`);
      }

    } else if (job.status === 'completed') {
      const skipNote = skipped > 0 ? `, ${skipped.toLocaleString()} already embedded` : '';
      const errNote  = job.batch_errors.length > 0 ? ` (${job.batch_errors.length} batch error(s)€” see below)` : '';
      setProgress(100, `Done€” ${embedded.toLocaleString()} embedded${skipNote}${errNote}`);

      if (job.batch_errors.length > 0) {
        const detail = job.batch_errors.map(be =>
          `Batch ${be.batch}:\n${be.error}\n\n${be.traceback}`
        ).join('\n-----------------\n');
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

/* -- Delete with confirm modal -- */
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
      throw new Error(`Server returned a non-JSON response (HTTP ${res.status}). The operation may have timed out€” check server logs.`);
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

function applyAdminUI() {
  // Show/hide Config sections that are admin-only
  const uploadSection  = document.getElementById('section-upload');
  const sunoSection    = document.getElementById('section-suno-team');
  if (uploadSection) uploadSection.classList.toggle('hidden', !currentUserIsAdmin);
  if (sunoSection)   sunoSection.classList.toggle('hidden', !currentUserIsAdmin);

  // Show Admin dropdown item for admins
  const adminMenuItem = document.getElementById('user-menu-admin');
  if (adminMenuItem) adminMenuItem.classList.toggle('hidden', !currentUserIsAdmin);

  // Account section€” visible in multi mode only
  const accountSection = document.getElementById('section-account');
  if (accountSection && APP_MODE === 'multi') {
    accountSection.classList.remove('hidden');
    const nameEl   = document.getElementById('account-username');
    const roleEl   = document.getElementById('account-role');
    const avatarEl = document.getElementById('account-avatar');
    if (nameEl && CURRENT_USER) {
      nameEl.textContent   = CURRENT_USER;
      if (avatarEl) avatarEl.textContent = CURRENT_USER.charAt(0).toUpperCase();
    }
    if (roleEl) roleEl.textContent = currentUserIsAdmin ? 'Administrator' : 'User';
  }

  // Refresh uploads table so action buttons reflect admin status
  if (!document.getElementById('page-settings').classList.contains('hidden')) {
    renderUploadsTable();
    if (currentUserIsAdmin) renderSunoTeamTable();
  }
}

// Settings-page logout button (multi mode)
document.getElementById('settings-logout-btn').addEventListener('click', async () => {
  const btn = document.getElementById('settings-logout-btn');
  btn.disabled    = true;
  btn.textContent = 'Logging out...';
  try { await fetch('/api/auth/logout', { method: 'POST' }); } catch (_) {}
  window.location.href = '/login';
});

async function loadSettingsPage() {
  applyAdminUI();
  loadModelOptions();
  renderUploadsTable();
  renderLabelManager();
  if (currentUserIsAdmin) renderSunoTeamTable();
  document.getElementById('goto-coding-btn')?.addEventListener('click', () => navigateTo('coding'));
  try {
    const d = await apiFetch('/api/stats');
    updateSettingsKeyStatus(d.api_key_set);
  } catch (_) {
    updateSettingsKeyStatus(false);
  }
}

// -- ADMIN PAGE ------------------------------------------------------------

function _adminMsg(text, type) {
  const el = document.getElementById('admin-msg');
  if (!el) return;
  el.textContent = text;
  el.className   = `text-xs mb-3 rounded-lg px-3 py-2 ${
    type === 'error' ? 'bg-red-50 text-red-700 border border-red-200'
                     : 'bg-green-50 text-green-700 border border-green-200'
  }`;
  el.classList.remove('hidden');
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add('hidden'), 4000);
}

async function loadAdminPage() {
  const table = document.getElementById('admin-users-table');
  if (!table) return;
  table.innerHTML = '<p class="text-sm text-gray-400 text-center py-6">Loading...</p>';
  try {
    const users = await apiFetch('/api/admin/users');
    _renderAdminTable(users);
  } catch (e) {
    table.innerHTML = `<p class="text-sm text-red-500 text-center py-6">Failed to load: ${esc(e.message)}</p>`;
  }
}

function _renderAdminTable(users) {
  const table = document.getElementById('admin-users-table');
  if (!users.length) {
    table.innerHTML = '<p class="text-sm text-gray-400 text-center py-6">No users found.</p>';
    return;
  }

  // Identify self by matching CURRENT_USER name (id not exposed in meta tag)
  table.innerHTML = `
    <table class="w-full text-sm border-collapse">
      <thead>
        <tr class="bg-gray-50 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
          <th class="px-3 py-2.5 border-b border-gray-200">Username</th>
          <th class="px-3 py-2.5 border-b border-gray-200">Role</th>
          <th class="px-3 py-2.5 border-b border-gray-200">Joined</th>
          <th class="px-3 py-2.5 border-b border-gray-200 text-right">Actions</th>
        </tr>
      </thead>
      <tbody>
        ${users.map(u => {
          const isSelf = u.username.toLowerCase() === CURRENT_USER.toLowerCase();
          const roleBadge = u.is_admin
            ? '<span class="inline-flex items-center gap-1 text-[10px] font-bold bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full uppercase tracking-wide">Admin</span>'
            : '<span class="inline-flex items-center gap-1 text-[10px] font-medium bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full uppercase tracking-wide">User</span>';
          const selfTag = isSelf
            ? '<span class="text-[10px] text-gray-400 ml-1">(you)</span>'
            : '';
          const actions = isSelf
            ? '<span class="text-xs text-gray-400 italic">â€”</span>'
            : `<div class="flex gap-2 justify-end flex-wrap">
                 <button class="admin-toggle-btn action-btn-primary text-xs py-1 px-2.5"
                         data-id="${u.id}" data-admin="${u.is_admin ? 1 : 0}">
                   ${u.is_admin ? 'Remove Admin' : 'Make Admin'}
                 </button>
                 <button class="admin-delete-btn action-btn-danger text-xs py-1 px-2.5"
                         data-id="${u.id}" data-name="${esc(u.username)}">
                   Delete
                 </button>
               </div>`;
          return `
            <tr class="border-b border-gray-100 hover:bg-gray-50" id="admin-user-row-${u.id}">
              <td class="px-3 py-2.5">
                <span class="font-medium text-gray-900">${esc(u.username)}</span>${selfTag}
              </td>
              <td class="px-3 py-2.5">${roleBadge}</td>
              <td class="px-3 py-2.5 text-gray-500 text-xs">${(u.created_at || '').slice(0, 10)}</td>
              <td class="px-3 py-2.5 text-right">${actions}</td>
            </tr>`;
        }).join('')}
      </tbody>
    </table>`;

  // Toggle admin
  table.querySelectorAll('.admin-toggle-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const uid       = parseInt(btn.dataset.id, 10);
      const wasAdmin  = btn.dataset.admin === '1';
      btn.disabled    = true;
      btn.textContent = '...';
      try {
        const res = await apiFetch(`/api/admin/users/${uid}/toggle-admin`, { method: 'POST' });
        _adminMsg(`${res.username} is now ${res.is_admin ? 'an Admin' : 'a regular User'}.`, 'ok');
        await loadAdminPage();
      } catch (e) {
        _adminMsg(e.message, 'error');
        btn.disabled    = false;
        btn.textContent = wasAdmin ? 'Remove Admin' : 'Make Admin';
      }
    });
  });

  // Delete user
  table.querySelectorAll('.admin-delete-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const uid  = parseInt(btn.dataset.id, 10);
      const name = btn.dataset.name;
      if (!confirm(`Delete user "${name}"? This cannot be undone.`)) return;
      btn.disabled    = true;
      btn.textContent = 'Deleting...';
      try {
        await apiFetch(`/api/admin/users/${uid}`, { method: 'DELETE' });
        _adminMsg(`User "${name}" deleted.`, 'ok');
        document.getElementById(`admin-user-row-${uid}`)?.remove();
      } catch (e) {
        _adminMsg(e.message, 'error');
        btn.disabled    = false;
        btn.textContent = 'Delete';
      }
    });
  });
}

document.getElementById('admin-refresh-btn')?.addEventListener('click', loadAdminPage);

async function renderLabelManager() {
  await loadAllCodes();   // refreshes _allCodes + filter chips if bookmark page was open
  const list = document.getElementById('labels-list');
  if (!_allCodes.length) {
    list.innerHTML = '<p class="text-sm text-gray-400">No codes yet. Create one above.</p>';
    return;
  }
  list.innerHTML = _allCodes.map(l => {
    const tc = labelTextColor(l.color);
    return `<span class="inline-flex items-center gap-1.5 text-xs font-medium rounded-full px-3 py-1"
                  style="background:${l.color};color:${tc}">
              ${esc(l.name)}
              <button class="label-delete-btn opacity-70 hover:opacity-100 font-bold leading-none"
                      data-code-id="${l.id}" data-code-name="${esc(l.name)}" title="Delete code">Ã—</button>
            </span>`;
  }).join('');
}

document.getElementById('labels-list').addEventListener('click', async e => {
  const btn = e.target.closest('.label-delete-btn');
  if (!btn) return;
  const id   = parseInt(btn.dataset.codeId);
  const name = btn.dataset.codeName;
  if (!confirm(`Delete code "${name}"? It will be removed from all bookmarks.`)) return;
  btn.disabled = true;
  const res = await fetch(`/api/codes/${id}`, { method: 'DELETE' });
  if (!res.ok) { btn.disabled = false; return; }
  _allCodes = _allCodes.filter(l => l.id !== id);
  _bmCodeFilter.delete(id);
  _cachedBookmarks.forEach(bm => { bm.codes = (bm.codes || []).filter(l => l.id !== id); });
  renderLabelManager();
  renderBmCodeFilterChips();
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
  const res = await fetch('/api/codes', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, color }),
  });
  if (!res.ok) {
    msgEl.textContent = (await res.json()).detail || 'Failed to create code.';
    msgEl.classList.remove('hidden');
    return;
  }
  const newLabel = await res.json();
  _allCodes = [..._allCodes, newLabel].sort((a, b) =>
    a.name.localeCompare(b.name, undefined, { sensitivity: 'base' })
  );
  nameInput.value = '';
  renderLabelManager();
  renderBmCodeFilterChips();
});

async function renderSunoTeamTable() {
  const el = document.getElementById('suno-team-table');
  el.innerHTML = '<p class="text-sm text-gray-400 text-center py-6">Loading...</p>';
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
                  ${currentUserIsAdmin ? `<button class="suno-remove action-btn-danger"
                          data-username="${esc(m.username)}">
                    Remove from team
                  </button>` : ''}
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
  btn.textContent = 'Removing...';
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

// -- SEARCH TABS -----------------------------------------------------------
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

// -- SEARCH ----------------------------------------------------------------
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
    renderResults(_sortSemanticResults(currentResults));
  } catch (e) { renderError(e.message); }
  finally { setBtnLoading(type, false); }
}

function _sortSemanticResults(results) {
  if (currentSearchType !== 'semantic') return results;
  const mode = document.getElementById('semantic-sort').value;
  const copy = [...results];
  if (mode === 'date_asc') {
    copy.sort((a, b) => new Date(a.date) - new Date(b.date));
  } else if (mode === 'date_desc') {
    copy.sort((a, b) => new Date(b.date) - new Date(a.date));
  } else {
    // 'score'€” restore original API order (highest similarity first)
    copy.sort((a, b) => (b.similarity_score ?? 0) - (a.similarity_score ?? 0));
  }
  return copy;
}

document.getElementById('semantic-sort').addEventListener('change', () => {
  if (currentSearchType === 'semantic' && currentResults.length) {
    renderResults(_sortSemanticResults(currentResults));
  }
});

function setBtnLoading(type, loading) {
  const btn = document.getElementById(`${type}-search-btn`);
  btn.disabled   = loading;
  btn.textContent = loading ? 'Searching...' : 'Search';
}

// -- RESULTS ---------------------------------------------------------------
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

  // Trend chart + summarize panel
  renderTrendChart(currentResults, _trendBucket);
  _updateSrCountLabel();
  if (results.length) document.getElementById('sr-section').classList.remove('hidden');
}

// -- TREND CHART -----------------------------------------------------------
const _MONTH_NAMES = ['January','February','March','April','May','June',
                      'July','August','September','October','November','December'];

function _binResultsByBucket(results, bucket) {
  // sortKey → { display, count, dateFrom, dateTo }
  const bins = new Map();
  for (const r of results) {
    const d = new Date(r.date);
    if (isNaN(d)) continue;
    let sortKey, display, dateFrom, dateTo;
    if (bucket === 'day') {
      const yr  = d.getUTCFullYear();
      const mo  = String(d.getUTCMonth() + 1).padStart(2, '0');
      const day = String(d.getUTCDate()).padStart(2, '0');
      sortKey  = `${yr}-${mo}-${day}`;
      display  = `${day} ${_MONTH_NAMES[d.getUTCMonth()]} ${yr}`;
      dateFrom = sortKey;
      dateTo   = sortKey;
    } else if (bucket === 'week') {
      const wom   = Math.min(Math.ceil(d.getDate() / 7), 4);  // 1-4
      const yr    = d.getFullYear();
      const moNum = String(d.getMonth() + 1).padStart(2, '0');
      sortKey  = `${yr}-${moNum}-W${wom}`;
      display  = `${_MONTH_NAMES[d.getMonth()]}-${yr}-W${wom}`;
      const weekStarts = [1, 8, 15, 22];
      const fromDay   = weekStarts[wom - 1];
      const toDay     = wom < 4 ? weekStarts[wom] - 1 : new Date(yr, d.getMonth() + 1, 0).getDate();
      dateFrom = `${yr}-${moNum}-${String(fromDay).padStart(2, '0')}`;
      dateTo   = `${yr}-${moNum}-${String(toDay).padStart(2, '0')}`;
    } else {
      const yr    = d.getFullYear();
      const moNum = String(d.getMonth() + 1).padStart(2, '0');
      const lastDay = new Date(yr, d.getMonth() + 1, 0).getDate();
      sortKey  = `${yr}-${moNum}`;
      display  = `${_MONTH_NAMES[d.getMonth()]}-${yr}`;
      dateFrom = `${yr}-${moNum}-01`;
      dateTo   = `${yr}-${moNum}-${String(lastDay).padStart(2, '0')}`;
    }
    if (!bins.has(sortKey)) bins.set(sortKey, { display, count: 0, dateFrom, dateTo });
    bins.get(sortKey).count++;
  }
  const sortKeys = [...bins.keys()].sort();
  return {
    labels:   sortKeys.map(k => bins.get(k).display),
    counts:   sortKeys.map(k => bins.get(k).count),
    ranges:   sortKeys.map(k => ({ from: bins.get(k).dateFrom, to: bins.get(k).dateTo, label: bins.get(k).display })),
  };
}

let _trendRanges = [];  // parallel to chart bars: { from, to, label }

function _getOrCreateChartBanner() {
  let banner = document.getElementById('chart-filter-banner');
  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'chart-filter-banner';
    banner.className = 'hidden flex items-center gap-2 bg-indigo-50 border border-indigo-200 rounded-xl px-3 py-2 mb-3 text-xs text-indigo-800';
    const container = document.getElementById('results-container');
    container.parentNode.insertBefore(banner, container);
  }
  return banner;
}

function _applyChartRangeFilter(range) {
  const filtered = currentResults.filter(r => {
    const d = (r.date || '').slice(0, 10);
    return d >= range.from && d <= range.to;
  });
  activeFilterTokens = [];
  _renderFilteredCards(filtered, currentResults.length);

  const banner = _getOrCreateChartBanner();
  banner.innerHTML = `
    <svg class="w-3.5 h-3.5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2a1 1 0 01-.293.707L13 13.414V19a1 1 0 01-.553.894l-4 2A1 1 0 017 21v-7.586L3.293 6.707A1 1 0 013 6V4z"/>
    </svg>
    <span>Showing <strong>${filtered.length}</strong> of <strong>${currentResults.length}</strong> messages &mdash; <strong>${esc(range.label)}</strong> (${range.from === range.to ? range.from : range.from + ' → ' + range.to})</span>
    <button id="chart-filter-clear" class="ml-auto text-indigo-600 hover:text-indigo-900 font-semibold">âœ• Clear</button>
  `;
  banner.classList.remove('hidden');
  document.getElementById('chart-filter-clear').addEventListener('click', () => {
    banner.classList.add('hidden');
    _resetToAllResults();
    document.getElementById('results-filter').value = '';
    document.getElementById('results-filter-count').classList.add('hidden');
  });

  // Highlight the clicked bar
  if (_trendChart) {
    const idx = _trendRanges.findIndex(r => r.from === range.from && r.to === range.to);
    _trendChart.data.datasets[0].backgroundColor = _trendRanges.map((_, i) =>
      i === idx ? 'rgba(79,70,229,1)' : 'rgba(99,102,241,0.4)'
    );
    _trendChart.update('none');
  }
}

function _resetChartHighlight() {
  if (!_trendChart) return;
  _trendChart.data.datasets[0].backgroundColor = 'rgba(99,102,241,0.7)';
  _trendChart.update('none');
}

function renderTrendChart(results, bucket) {
  const section = document.getElementById('trend-section');
  if (!results.length) { section.classList.add('hidden'); return; }

  const { labels, counts, ranges } = _binResultsByBucket(results, bucket);
  if (!labels.length) { section.classList.add('hidden'); return; }
  _trendRanges = ranges;

  // Clear any active chart filter banner when re-rendering
  const banner = document.getElementById('chart-filter-banner');
  if (banner) banner.classList.add('hidden');

  section.classList.remove('hidden');
  if (_trendChart) { _trendChart.destroy(); _trendChart = null; }

  _trendChart = new Chart(document.getElementById('trend-chart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: counts,
        backgroundColor: 'rgba(99,102,241,0.7)',
        borderColor: '#4338ca',
        borderWidth: 1,
        borderRadius: bucket === 'day' ? 1 : 3,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cursor: 'pointer',
      onClick(_, elements) {
        if (!elements.length) return;
        const range = _trendRanges[elements[0].index];
        if (range) _applyChartRangeFilter(range);
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: items => {
              const r = _trendRanges[items[0].dataIndex];
              if (!r) return items[0].label;
              return r.from === r.to ? r.from : `${r.from} → ${r.to}`;
            },
            label: item => `${item.raw} messages€” click to filter`,
          },
        },
      },
      scales: {
        x: { ticks: { maxRotation: 45, font: { size: 11 } } },
        y: { beginAtZero: true, ticks: { precision: 0, font: { size: 11 } } },
      },
    },
  });

  // Make the canvas show a pointer cursor
  document.getElementById('trend-chart').style.cursor = 'pointer';
}

(function () {
  const _buckets = ['month', 'week', 'day'];
  function _setTrendBucket(b) {
    _trendBucket = b;
    for (const bucket of _buckets) {
      const btn = document.getElementById(`trend-btn-${bucket}`);
      if (!btn) continue;
      btn.classList.toggle('trend-bucket-active', b === bucket);
      btn.setAttribute('aria-pressed', String(b === bucket));
    }
    renderTrendChart(currentResults, _trendBucket);
  }
  document.getElementById('trend-btn-month').addEventListener('click', () => _setTrendBucket('month'));
  document.getElementById('trend-btn-week').addEventListener('click',  () => _setTrendBucket('week'));
  document.getElementById('trend-btn-day').addEventListener('click',   () => _setTrendBucket('day'));
}());

// -- SUMMARIZE RESULTS -----------------------------------------------------
const LOG_ICONS = {
  filter:      'ðŸ”',
  retrieval:   'ðŸ“¡',
  dedup:       'ðŸ§¹',
  cluster:     'ðŸ”®',
  sample:      'ðŸŽ¯',
  llm:         'âœ¨',
  fallback:    'âš ï¸',
  meta:        'ðŸ“…',
  instruction: 'ðŸ“',
};
function _updateSrCountLabel() {
  const el = document.getElementById('sr-count-label');
  if (el) el.textContent = currentResults.length.toLocaleString();
}

function renderSrLogEntry(entry) {
  const logEl = document.getElementById('sr-process-log');
  const div   = document.createElement('div');
  div.className = `log-entry log-step-${entry.step || 'fallback'}`;
  const icon = LOG_ICONS[entry.step] || 'â€¢';
  div.innerHTML =
    `<span class="log-icon">${icon}</span>` +
    `<span class="log-label">${esc(entry.label || '')}</span>` +
    `<span class="log-msg">${esc(entry.msg || '')}</span>`;
  logEl.appendChild(div);
}

async function doSummarizeResults() {
  if (!currentResults.length) {
    showErrorPopup('No results to summarize. Run a search first.');
    return;
  }
  const btn           = document.getElementById('sr-btn');
  const logEl         = document.getElementById('sr-process-log');
  const resultEl      = document.getElementById('sr-result');
  const prompt        = document.getElementById('sr-prompt').value.trim();
  const model         = document.getElementById('sr-model').value;
  const retrievalMode = document.getElementById('sr-retrieval-mode').value;

  btn.disabled    = true;
  btn.textContent = 'Summarizing...';

  document.getElementById('sr-results-panel').classList.remove('hidden');
  logEl.innerHTML    = '';
  resultEl.innerHTML = '';
  document.getElementById('sr-followup-section').classList.add('hidden');
  document.getElementById('sr-followup-history').innerHTML = '';
  document.getElementById('sr-export-pdf').classList.add('hidden');
  srFollowUpHistory = [];
  srSummaryText     = '';
  srLastPrompt      = prompt;

  const messages = currentResults.map(r => ({
    msg_uuid: r.msg_uuid || null,
    username: r.username || '',
    date:     r.date     || '',
    content:  r.content  || '',
  }));

  try {
    const res = await fetch('/api/summarize-results', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages, prompt: prompt || null, model, retrieval_mode: retrievalMode }),
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
            renderSrLogEntry(delta);
          } else if (delta.content) {
            srSummaryText += delta.content;
            resultEl.innerHTML = marked.parse(srSummaryText);
          } else if (delta.error) {
            throw new Error(delta.error);
          }
        } catch (parseErr) {
          if (!(parseErr instanceof SyntaxError)) throw parseErr;
        }
      }
    }

    if (!srSummaryText) {
      showErrorPopup('No response received from the model. Check your API key and selected model.');
      return;
    }
    srFollowUpHistory = [{ role: 'assistant', content: srSummaryText }];
    document.getElementById('sr-followup-section').classList.remove('hidden');
    document.getElementById('sr-export-pdf').classList.remove('hidden');

  } catch (e) {
    showErrorPopup(e.message);
  } finally {
    btn.disabled    = false;
    btn.textContent = 'Summarize Results';
  }
}

/* -- SR follow-up bubble helpers -- */
function _appendSrUserBubble(text) {
  const c = document.getElementById('sr-followup-history');
  const w = document.createElement('div'); w.className = 'flex justify-end';
  const b = document.createElement('div'); b.className = 'chat-bubble-user';
  b.textContent = text;
  w.appendChild(b); c.appendChild(w);
  c.scrollTop = c.scrollHeight;
}

function _appendSrAssistantBubble() {
  const c = document.getElementById('sr-followup-history');
  const w = document.createElement('div'); w.className = 'flex justify-start';
  const b = document.createElement('div'); b.className = 'chat-bubble-assistant markdown-body';
  w.appendChild(b); c.appendChild(w);
  c.scrollTop = c.scrollHeight;
  return b;
}

function _appendSrLogStrip() {
  const c     = document.getElementById('sr-followup-history');
  const strip = document.createElement('div'); strip.className = 'fu-log-strip';
  c.appendChild(strip); c.scrollTop = c.scrollHeight;
  return strip;
}

function _renderSrFuLogEntry(strip, entry) {
  const div = document.createElement('div');
  div.className = `fu-log-entry fu-log-step-${entry.step || 'fallback'}`;
  div.innerHTML =
    `<span class="fu-log-icon">${LOG_ICONS[entry.step] || 'â€¢'}</span>` +
    `<span class="fu-log-label">${esc(entry.label || '')}</span>` +
    `<span class="fu-log-msg">${esc(entry.msg || '')}</span>`;
  strip.appendChild(div);
  const c = document.getElementById('sr-followup-history');
  c.scrollTop = c.scrollHeight;
}

async function sendSrFollowUp() {
  const input    = document.getElementById('sr-followup-input');
  const sendBtn  = document.getElementById('sr-followup-send');
  const question = input.value.trim();
  if (!question || !srSummaryText) return;

  const model = document.getElementById('sr-model').value;
  input.value      = '';
  input.disabled   = true;
  sendBtn.disabled = true;
  sendBtn.textContent = '...';

  srFollowUpHistory.push({ role: 'user', content: question });
  _appendSrUserBubble(question);
  const strip  = _appendSrLogStrip();
  const bubble = _appendSrAssistantBubble();
  let answerText = '';

  try {
    const res = await fetch('/api/summarize-results/followup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, history: srFollowUpHistory.slice(0, -1), model }),
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
            _renderSrFuLogEntry(strip, delta);
          } else if (delta.content) {
            answerText += delta.content;
            bubble.innerHTML = marked.parse(answerText);
            const c = document.getElementById('sr-followup-history');
            c.scrollTop = c.scrollHeight;
          } else if (delta.error) {
            throw new Error(delta.error);
          }
        } catch (parseErr) {
          if (!(parseErr instanceof SyntaxError)) throw parseErr;
        }
      }
    }
    srFollowUpHistory.push({ role: 'assistant', content: answerText });

  } catch (e) {
    bubble.remove();
    srFollowUpHistory.pop();
    showErrorPopup(e.message);
  } finally {
    input.disabled   = false;
    sendBtn.disabled = false;
    sendBtn.textContent = 'Ask';
    input.focus();
  }
}

/* -- SR PDF export -- */
function exportSrPDF() {
  const summaryHTML = document.getElementById('sr-result').innerHTML;
  const dateStr     = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
  const pdfFilename = `Results-Summary-${new Date().toISOString().slice(0, 10)}`;

  let instructionHTML = '';
  if (srLastPrompt) {
    const safe = srLastPrompt.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    instructionHTML = `<div class="custom-instruction"><span class="ci-label">Custom Instructions</span>${safe}</div>`;
  }

  let qaHTML = '';
  for (const turn of srFollowUpHistory.slice(1)) {
    if (turn.role === 'user') {
      const safe = turn.content.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      qaHTML += `<div class="q-block"><span class="q-label">Question</span>${safe}</div>`;
    } else {
      qaHTML += `<div class="a-block">${marked.parse(turn.content)}</div>`;
    }
  }

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>${pdfFilename}</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  max-width: 820px; margin: 40px auto; padding: 0 28px;
  color: #1e293b; line-height: 1.65; font-size: 14px; }
h1 { font-size: 1.5rem; color: #4c1d95; padding-bottom: 10px;
     border-bottom: 2px solid #e2e8f0; margin-bottom: 6px; }
.meta { font-size: 0.75rem; color: #6b7280; margin-bottom: 1.75rem; }
h2 { font-size: 1.15rem; font-weight: 700; color: #1e293b; margin-top: 2rem;
     border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin-bottom: 1rem; }
h3 { font-size: 1rem; font-weight: 600; color: #374151; margin: 1rem 0 0.4rem; }
h4 { font-size: 0.9rem; font-weight: 600; margin: 0.8rem 0 0.3rem; }
p  { margin-bottom: 0.65rem; }
ul, ol { padding-left: 1.4rem; margin-bottom: 0.65rem; }
li { margin-bottom: 0.2rem; }
blockquote { border-left: 3px solid #7c3aed; margin: 0.75rem 0;
  padding: 6px 14px; background: #f5f3ff; color: #3730a3;
  border-radius: 0 6px 6px 0; font-style: italic; }
code { background: #f1f5f9; border-radius: 4px; padding: 1px 5px; font-size: 0.82em; font-family: monospace; }
pre  { background: #1e293b; color: #e2e8f0; border-radius: 6px; padding: 12px; overflow-x: auto; margin-bottom: 0.65rem; }
pre code { background: none; padding: 0; color: inherit; }
hr   { border: none; border-top: 1px solid #e2e8f0; margin: 1.25rem 0; }
strong { font-weight: 700; } em { font-style: italic; } a { color: #4c1d95; text-decoration: underline; }
table { border-collapse: collapse; width: 100%; margin-bottom: 0.65rem; font-size: 0.85rem; }
th, td { border: 1px solid #e2e8f0; padding: 6px 10px; text-align: left; }
th { background: #f8fafc; font-weight: 700; }
.custom-instruction { background: #fefce8; border: 1px solid #fde68a; border-radius: 6px;
  padding: 8px 14px; margin-bottom: 1.75rem; font-size: 0.8rem; color: #713f12; }
.ci-label { display: block; font-size: 0.68rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.06em; color: #92400e; margin-bottom: 4px; }
.q-block { background: #f5f3ff; border-radius: 10px 10px 10px 2px;
  padding: 10px 14px; margin: 14px 0 4px; color: #1e1b4b; font-weight: 600; }
.q-label { display: block; font-size: 0.68rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.06em; color: #7c3aed; margin-bottom: 4px; }
.a-block { background: #f8fafc; border-left: 3px solid #94a3b8;
  border-radius: 0 10px 10px 0; padding: 10px 14px; margin: 4px 0 14px; }
@media print { body { margin: 16px 28px; } }
</style>
</head>
<body>
<h1>Results Summary</h1>
<p class="meta">Exported ${dateStr} &middot; ${currentResults.length.toLocaleString()} search results</p>
${instructionHTML}
<div class="summary-body">${summaryHTML}</div>
${qaHTML ? '<h2>Follow-up Q&amp;A</h2><div class="qa-body">' + qaHTML + '</div>' : ''}
<script>window.onload = function() { window.print(); };<\/script>
</body>
</html>`;

  const blob = new Blob([html], { type: 'text/html' });
  const url  = URL.createObjectURL(blob);
  const win  = window.open(url, '_blank', 'width=920,height=750');
  if (!win) { URL.revokeObjectURL(url); showErrorPopup('Pop-up blocked. Please allow pop-ups for this page.'); return; }
  win.addEventListener('load', () => URL.revokeObjectURL(url), { once: true });
}

/* -- Trend chart PNG export -- */
function exportTrendChartPNG() {
  if (!_trendChart) return;
  const src = document.getElementById('trend-chart');
  const out  = document.createElement('canvas');
  out.width  = src.width;
  out.height = src.height;
  const ctx  = out.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, out.width, out.height);
  ctx.drawImage(src, 0, 0);
  const a = document.createElement('a');
  a.download = `trend-${_trendBucket}-${new Date().toISOString().slice(0, 10)}.png`;
  a.href = out.toDataURL('image/png');
  a.click();
}
document.getElementById('trend-export-png').addEventListener('click', exportTrendChartPNG);

/* -- SR event wiring -- */
document.getElementById('sr-retrieval-mode').addEventListener('change', () => {
  const hint = document.getElementById('sr-mode-hint');
  const mode = document.getElementById('sr-retrieval-mode').value;
  hint.textContent = mode === 'cluster'
    ? 'Deduplicates and samples representative messages across semantic clusters.'
    : 'All messages passed directly to the LLM€” no clustering or deduplication.';
});
document.getElementById('sr-btn').addEventListener('click', doSummarizeResults);
document.getElementById('sr-log-toggle').addEventListener('click', () => {
  const logEl = document.getElementById('sr-process-log');
  const btn   = document.getElementById('sr-log-toggle');
  const hidden = logEl.classList.toggle('hidden');
  btn.innerHTML = hidden ? '&#9660; Show' : '&#9650; Hide';
});
document.getElementById('sr-followup-send').addEventListener('click', sendSrFollowUp);
document.getElementById('sr-followup-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && e.ctrlKey) { e.preventDefault(); sendSrFollowUp(); }
});
document.getElementById('sr-followup-clear').addEventListener('click', () => {
  srFollowUpHistory = srSummaryText ? [{ role: 'assistant', content: srSummaryText }] : [];
  document.getElementById('sr-followup-history').innerHTML = '';
});
document.getElementById('sr-export-pdf').addEventListener('click', exportSrPDF);
document.getElementById('sr-export-pdf-followup').addEventListener('click', exportSrPDF);

// -- RESULTS FILTER --------------------------------------------------------

let filterMode        = 'exact';  // 'exact' | 'any' | 'semantic'
let _semanticDebounce = null;

/* -- Set active mode + update UI -- */
const _EXACT_ACTIVE    = ['bg-indigo-700','text-white'];
const _EXACT_INACTIVE  = ['bg-slate-100','text-slate-500'];
const _ANY_ACTIVE      = ['bg-emerald-600','text-white'];
const _ANY_INACTIVE    = ['bg-slate-100','text-slate-500'];
const _SEM_ACTIVE      = ['bg-violet-600','text-white'];
const _SEM_INACTIVE    = ['bg-slate-100','text-slate-500'];

function setFilterMode(mode) {
  filterMode = mode;
  const exactBtn = document.getElementById('filter-mode-exact');
  const anyBtn   = document.getElementById('filter-mode-any');
  const semBtn   = document.getElementById('filter-mode-semantic');

  exactBtn.classList.remove(..._EXACT_ACTIVE,   ..._EXACT_INACTIVE);
  anyBtn.classList.remove(  ..._ANY_ACTIVE,     ..._ANY_INACTIVE);
  semBtn.classList.remove(  ..._SEM_ACTIVE,     ..._SEM_INACTIVE);

  if (mode === 'exact') {
    exactBtn.classList.add(..._EXACT_ACTIVE);
    anyBtn.classList.add(  ..._ANY_INACTIVE);
    semBtn.classList.add(  ..._SEM_INACTIVE);
  } else if (mode === 'any') {
    exactBtn.classList.add(..._EXACT_INACTIVE);
    anyBtn.classList.add(  ..._ANY_ACTIVE);
    semBtn.classList.add(  ..._SEM_INACTIVE);
  } else {
    exactBtn.classList.add(..._EXACT_INACTIVE);
    anyBtn.classList.add(  ..._ANY_INACTIVE);
    semBtn.classList.add(  ..._SEM_ACTIVE);
  }

  exactBtn.setAttribute('aria-pressed', String(mode === 'exact'));
  anyBtn.setAttribute(  'aria-pressed', String(mode === 'any'));
  semBtn.setAttribute(  'aria-pressed', String(mode === 'semantic'));

  const placeholders = {
    exact:    'Exact: whole-word match, multi-word phrase scores highest...',
    any:      'Any Word: returns messages containing at least one query word...',
    semantic: 'Semantic: re-rank results by embedding similarity...',
  };
  document.getElementById('results-filter').placeholder = placeholders[mode];
}

document.getElementById('filter-mode-exact')
  .addEventListener('click', () => { setFilterMode('exact');    applyResultsFilter(); });
document.getElementById('filter-mode-any')
  .addEventListener('click', () => { setFilterMode('any');      applyResultsFilter(); });
document.getElementById('filter-mode-semantic')
  .addEventListener('click', () => { setFilterMode('semantic'); applyResultsFilter(); });

/* -- Shared render helpers -- */
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

/* -- Exact filter (instant, client-side) -- */
function _escapeRegex(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

function _applyExactFilter(term) {
  if (!term) { _resetToAllResults(); return; }

  const words = term.split(/\s+/).filter(Boolean);
  activeFilterTokens = words.length > 1 ? [term, ...words] : words;

  // Substring inclusion: all words must appear somewhere (AND logic).
  // Keeps results visible while still typing partial words.
  const subRegexes = words.map(w => new RegExp(_escapeRegex(w), 'i'));

  // Whole-word regexes for bonus scoring (exact-word hit ranks higher)
  const wordRegexes = words.map(w =>
    new RegExp('\\b' + _escapeRegex(w) + '\\b', 'i')
  );
  // Phrase regex: full sequence (multi-word only) for highest score
  const phraseRegex = words.length > 1
    ? new RegExp(_escapeRegex(words.join(' ')), 'i')
    : null;

  const scored = currentResults
    .map(m => {
      const user    = m.username || '';
      const content = m.content  || '';
      const text    = user + ' ' + content;

      // All words must be present as substrings (AND gate)
      if (!subRegexes.every(rx => rx.test(text))) return { m, s: 0 };

      // Phrase match€” highest score
      if (phraseRegex) {
        if (phraseRegex.test(user))    return { m, s: 1500 };
        if (phraseRegex.test(content)) return { m, s: 1000 };
      }

      // Whole-word bonus over plain substring
      let s = 1;  // base: passes substring gate
      for (const rx of wordRegexes) {
        if (rx.test(user))    s += 20;
        if (rx.test(content)) s += 10;
      }
      return { m, s };
    })
    .filter(x => x.s > 0)
    .sort((a, b) => new Date(a.m.date) - new Date(b.m.date));

  _renderFilteredCards(scored.map(x => x.m), currentResults.length);
}

/* -- Any-word filter (instant, client-side) -- */
function _applyAnyWordFilter(term) {
  if (!term) { _resetToAllResults(); return; }

  const words = term.split(/\s+/).filter(Boolean);
  // Highlight all individual words
  activeFilterTokens = words;

  // Substring match (OR logic)€” partial words work while typing
  const tokenRegexes = words.map(w => new RegExp(_escapeRegex(w), 'i'));

  const matched = currentResults
    .filter(m => {
      const text = (m.username || '') + ' ' + (m.content || '');
      return tokenRegexes.some(rx => rx.test(text));
    })
    .sort((a, b) => new Date(a.date) - new Date(b.date));

  _renderFilteredCards(matched, currentResults.length);
}

/* -- Semantic filter (debounced, API-backed) -- */
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
        `<p class="text-center text-amber-600 py-10 text-sm">âš  ${esc(warning)}</p>`;
      spinner.classList.add('hidden');
      return;
    }
    const byId = Object.fromEntries(currentResults.map(m => [m.id, m]));

    const hits = ranked
      .map(r => ({ ...byId[r.id], similarity_score: r.score }))
      .filter(m => m.id != null);

    // Show count with threshold label, and query interpretation if it changed
    const interpreted = query_used && query_used !== term
      ? `· searched: "${query_used}"` : '';
    const countLabel2 = document.getElementById('results-filter-count');
    countLabel2.textContent = `${hits.length} of ${currentResults.length}· similarity‰¥ ${threshold}${interpreted}`;
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

/* -- Unified dispatcher -- */
function applyResultsFilter() {
  const term     = document.getElementById('results-filter').value.trim().toLowerCase();
  const clearBtn = document.getElementById('results-filter-clear');
  clearBtn.classList.toggle('hidden', !term);
  if (!currentResults.length) return;

  clearTimeout(_semanticDebounce);
  if (filterMode === 'exact') {
    _applyExactFilter(term);
  } else if (filterMode === 'any') {
    _applyAnyWordFilter(term);
  } else {
    if (!term) { _applySemanticFilter(''); return; }
    _semanticDebounce = setTimeout(() => _applySemanticFilter(term), 500);
  }
}

document.getElementById('results-filter').addEventListener('input', applyResultsFilter);
document.getElementById('results-filter-clear').addEventListener('click', () => {
  document.getElementById('results-filter').value = '';
  const banner = document.getElementById('chart-filter-banner');
  if (banner) banner.classList.add('hidden');
  _resetChartHighlight();
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
    ? `<p class="text-xs text-gray-500 mt-1">ðŸ“Ž ${esc(msg.attachments)}</p>` : '';
  const reactLine = hasContent(msg.reactions)
    ? `<p class="text-xs text-gray-500 mt-1">ðŸ'¬ ${esc(msg.reactions)}</p>` : '';

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

// -- CONTEXT EXPANSION -----------------------------------------------------
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
  btn.textContent = 'Loading...'; btn.disabled = true;
  try {
    const res  = await fetch(`/api/context/${id}?before=${before}&after=${after}`);
    if (!res.ok) throw new Error('Failed to load context');
    const msgs = await res.json();
    ctxEl.innerHTML = `
      <div class="border-t bg-slate-50 p-4 space-y-2">
        <p class="text-xs text-gray-500 font-medium mb-3">
          Context€” ${msgs.length} messages (${before} before &bull; ${after} after)
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
             style="background:#fef08a;color:#78350f">â˜… result</span>` : '';
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

// -- EXPORT ----------------------------------------------------------------
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

// -- BOOKMARKS -------------------------------------------------------------

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
    // 'added'€” sort by bookmark_id ascending (insertion order)
    sorted.sort((a, b) => a.bookmark_id - b.bookmark_id);
  }
  return sorted;
}

let _cachedBookmarks = [];

async function loadBookmarksPage() {
  const container = document.getElementById('bookmarks-container');
  container.innerHTML = '<p class="text-sm text-gray-400 text-center py-8">Loading...</p>';
  await loadAllCodes();
  try {
    _cachedBookmarks = await apiFetch('/api/bookmarks');
    _renderBookmarksSorted();
  } catch (e) {
    container.innerHTML = `<p class="text-sm text-red-500 text-center py-8">Failed to load bookmarks: ${esc(e.message)}</p>`;
  }
}

// -- Label colour helpers ------------------------------------------------------
function labelTextColor(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.55 ? '#1f2937' : '#ffffff';
}

// -- Load & render code filter chips -----------------------------------------
async function loadAllCodes() {
  try {
    _allCodes = await apiFetch('/api/codes');
  } catch (_) { _allCodes = []; }
  renderBmCodeFilterChips();
}

function renderBmCodeFilterChips() {
  const container = document.getElementById('bm-label-filter-chips');
  if (!_allCodes.length) {
    container.innerHTML = '<span class="text-xs text-gray-400">No codes yet. Create codes in Settings or the Coding Manager.</span>';
    return;
  }
  container.innerHTML = _allCodes.map(l => {
    const active = _bmCodeFilter.has(l.id);
    const bg     = active ? l.color : '#f1f5f9';
    const tc     = active ? labelTextColor(l.color) : '#64748b';
    const border = active ? l.color : '#e2e8f0';
    return `<button class="bm-code-filter-chip text-xs px-2.5 py-0.5 rounded-full font-medium border transition-all"
                    data-code-id="${l.id}"
                    style="background:${bg};color:${tc};border-color:${border}">
              ${esc(l.name)}
            </button>`;
  }).join('');
}

// -- Bookmark code chips inside card -----------------------------------------
function _bmCodeChipsHtml(bm) {
  const chips = (bm.codes || []).map(l => {
    const tc = labelTextColor(l.color);
    return `<span class="bm-code-chip text-xs px-2 py-0.5 rounded-full font-medium cursor-pointer select-none"
                  style="background:${l.color};color:${tc}"
                  data-bm-id="${bm.bookmark_id}" data-code-id="${l.id}"
                  title="Remove code">${esc(l.name)} Ã—</span>`;
  }).join('');
  return chips + `<button class="bm-code-btn text-xs text-gray-400 hover:text-indigo-600 border border-dashed border-gray-300 hover:border-indigo-400 rounded-full px-2 py-0.5 transition-colors"
                          data-bm-id="${bm.bookmark_id}">+ code</button>`;
}

// -- Render the inline code picker panel -------------------------------------
function _renderBmCodePanel(bookmarkId) {
  const panel = document.getElementById(`bm-code-panel-${bookmarkId}`);
  const bm = _cachedBookmarks.find(b => b.bookmark_id === bookmarkId);
  if (!panel || !bm) return;
  const assignedIds = new Set((bm.codes || []).map(l => l.id));
  const existingChips = _allCodes.length
    ? `<div class="flex flex-wrap gap-1.5 mb-2">
        ${_allCodes.map(l => {
          const active = assignedIds.has(l.id);
          const bg     = active ? l.color : '#f8fafc';
          const tc     = active ? labelTextColor(l.color) : '#64748b';
          const border = active ? l.color : '#cbd5e1';
          return `<button class="bm-code-toggle text-xs px-2.5 py-0.5 rounded-full font-medium border transition-all"
                          data-bm-id="${bookmarkId}" data-code-id="${l.id}"
                          style="background:${bg};color:${tc};border-color:${border}">
                    ${active ? 'âœ“ ' : ''}${esc(l.name)}
                  </button>`;
        }).join('')}
      </div>`
    : '';
  panel.innerHTML = `
    <div class="p-2">
      ${existingChips}
      <div class="flex items-center gap-1.5 pt-1 border-t border-gray-100">
        <input class="bm-new-code-input flex-1 min-w-0 border rounded-lg px-2 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-300"
               placeholder="New code name..." maxlength="40" data-bm-id="${bookmarkId}" />
        <input class="bm-new-code-color w-7 h-7 rounded cursor-pointer border border-gray-200 p-0.5"
               type="color" value="#6366f1" data-bm-id="${bookmarkId}" title="Pick colour" />
        <button class="bm-new-code-create text-xs px-2.5 py-1 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 shrink-0"
                data-bm-id="${bookmarkId}">Add</button>
      </div>
    </div>`;
}

// -- filter bookmarks ---------------------------------------------------------
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
    if (_bmCodeFilter.size > 0) {
      const bmLabelIds = new Set((bm.codes || []).map(l => l.id));
      const hasMatch = [..._bmCodeFilter].some(id => bmLabelIds.has(id));
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
        ${hasContent(bm.attachments) ? `<p class="text-xs text-gray-500 mt-1">ðŸ“Ž ${esc(bm.attachments)}</p>` : ''}
        ${hasContent(bm.reactions)   ? `<p class="text-xs text-gray-500 mt-1">ðŸ'¬ ${esc(bm.reactions)}</p>`   : ''}
        <!-- Codes -->
        <div class="flex items-center flex-wrap gap-1 mt-2 min-h-[1.5rem]" id="bm-codes-${bm.bookmark_id}">
          ${_bmCodeChipsHtml(bm)}
        </div>
        <!-- Inline code picker (hidden until opened) -->
        <div id="bm-code-panel-${bm.bookmark_id}" class="hidden mt-1 border border-dashed border-gray-200 rounded-xl bg-gray-50"></div>
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

  // Remove code chip (Ã— click on assigned label)
  const labelChip = e.target.closest('.bm-code-chip');
  if (labelChip) {
    const bmId    = parseInt(labelChip.dataset.bmId);
    const codeId = parseInt(labelChip.dataset.codeId);
    await fetch(`/api/bookmarks/${bmId}/codes/${codeId}`, { method: 'DELETE' });
    const bm = _cachedBookmarks.find(b => b.bookmark_id === bmId);
    if (bm) bm.codes = (bm.codes || []).filter(l => l.id !== codeId);
    const labelsRow = document.getElementById(`bm-codes-${bmId}`);
    if (labelsRow) labelsRow.innerHTML = _bmCodeChipsHtml(bm);
    return;
  }

  // Open/close code picker panel
  const labelBtn = e.target.closest('.bm-code-btn');
  if (labelBtn) {
    const bmId = parseInt(labelBtn.dataset.bmId);
    const panel = document.getElementById(`bm-code-panel-${bmId}`);
    if (panel.classList.contains('hidden')) {
      _renderBmCodePanel(bmId);
      panel.classList.remove('hidden');
    } else {
      panel.classList.add('hidden');
    }
    return;
  }

  // Toggle code assignment in picker panel
  const labelToggle = e.target.closest('.bm-code-toggle');
  if (labelToggle) {
    const bmId    = parseInt(labelToggle.dataset.bmId);
    const codeId = parseInt(labelToggle.dataset.codeId);
    const bm      = _cachedBookmarks.find(b => b.bookmark_id === bmId);
    if (!bm) return;
    const isAssigned = (bm.codes || []).some(l => l.id === codeId);
    if (isAssigned) {
      await fetch(`/api/bookmarks/${bmId}/codes/${codeId}`, { method: 'DELETE' });
      bm.codes = (bm.codes || []).filter(l => l.id !== codeId);
    } else {
      await fetch(`/api/bookmarks/${bmId}/codes/${codeId}`, { method: 'POST' });
      const label = _allCodes.find(l => l.id === codeId);
      if (label) bm.codes = [...(bm.codes || []), { id: label.id, name: label.name, color: label.color }];
    }
    // Update chips row and re-render panel
    const labelsRow = document.getElementById(`bm-codes-${bmId}`);
    if (labelsRow) labelsRow.innerHTML = _bmCodeChipsHtml(bm);
    _renderBmCodePanel(bmId);
    return;
  }

  // Create new code inline and assign it
  const createBtn = e.target.closest('.bm-new-code-create');
  if (createBtn) {
    const bmId  = parseInt(createBtn.dataset.bmId);
    const panel = document.getElementById(`bm-code-panel-${bmId}`);
    const nameInput  = panel.querySelector('.bm-new-code-input');
    const colorInput = panel.querySelector('.bm-new-code-color');
    const name  = (nameInput?.value || '').trim();
    const color = colorInput?.value || '#6366f1';
    if (!name) { nameInput?.focus(); return; }
    createBtn.disabled = true;
    createBtn.textContent = '...';
    try {
      // Create the code
      const labelRes = await fetch('/api/codes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, color }),
      });
      const newLabel = await labelRes.json();
      if (!labelRes.ok) {
        // Label may already exist€” find it in _allCodes
        const existing = _allCodes.find(l => l.name.toLowerCase() === name.toLowerCase());
        if (!existing) { createBtn.disabled = false; createBtn.textContent = 'Add'; return; }
        newLabel.id = existing.id; newLabel.name = existing.name; newLabel.color = existing.color;
      } else {
        // Add to global cache, keep sorted
        _allCodes = [..._allCodes, newLabel].sort((a, b) =>
          a.name.localeCompare(b.name, undefined, { sensitivity: 'base' })
        );
        renderBmCodeFilterChips();
      }
      // Assign to this bookmark
      await fetch(`/api/bookmarks/${bmId}/codes/${newLabel.id}`, { method: 'POST' });
      const bm = _cachedBookmarks.find(b => b.bookmark_id === bmId);
      if (bm && !(bm.codes || []).some(l => l.id === newLabel.id)) {
        bm.codes = [...(bm.codes || []), { id: newLabel.id, name: newLabel.name, color: newLabel.color }];
      }
      const labelsRow = document.getElementById(`bm-codes-${bmId}`);
      if (labelsRow) labelsRow.innerHTML = _bmCodeChipsHtml(bm);
      _renderBmCodePanel(bmId);
      document.dispatchEvent(new CustomEvent('codebook-updated'));
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

  ctxBtn.textContent = 'Loading...';
  ctxBtn.disabled    = true;
  try {
    const msgs = await apiFetch(`/api/context/${id}?before=${before}&after=${after}`);
    ctxEl.innerHTML = `
      <div class="border-t bg-slate-50 p-4 space-y-2">
        <p class="text-xs text-gray-500 font-medium mb-3">
          Context€” ${msgs.length} messages (${before} before &bull; ${after} after)
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

// Enter key in inline code name input → click the Add button
document.getElementById('bookmarks-container').addEventListener('keydown', e => {
  if (e.key !== 'Enter') return;
  const input = e.target.closest('.bm-new-code-input');
  if (!input) return;
  e.preventDefault();
  const bmId = input.dataset.bmId;
  document.querySelector(`.bm-new-code-create[data-bm-id="${bmId}"]`)?.click();
});

// Code filter chip toggle
document.getElementById('bm-label-filter-chips').addEventListener('click', e => {
  const chip = e.target.closest('.bm-code-filter-chip');
  if (!chip) return;
  const id = parseInt(chip.dataset.codeId);
  if (_bmCodeFilter.has(id)) _bmCodeFilter.delete(id);
  else _bmCodeFilter.add(id);
  renderBmCodeFilterChips();
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

// -- CODING MANAGER --------------------------------------------------------

let _cmCodes      = [];
let _cmCategories = [];
let _cmSelected   = new Set();
let _cmMergeMode  = false;
let _cmOpenCodeId = null;

async function loadCodingPage() {
  await _cmRefresh();
}

async function _cmRefresh() {
  try {
    [_cmCodes, _cmCategories] = await Promise.all([
      apiFetch('/api/codes'),
      apiFetch('/api/code-categories'),
    ]);
  } catch (e) {
    document.getElementById('cm-code-list').innerHTML =
      `<p class="text-sm text-red-500 text-center py-8">Failed to load: ${esc(e.message)}</p>`;
    return;
  }
  _cmPopulateCategorySelects();
  _cmRenderCodeList();
  if (_cmOpenCodeId !== null) {
    const code = _cmCodes.find(c => c.id === _cmOpenCodeId);
    if (code) _cmOpenDetail(code);
    else _cmCloseDetail();
  }
}

function _cmPopulateCategorySelects() {
  const opts = '<option value="">— Uncategorized —</option>' +
    _cmCategories.map(c => `<option value="${c.id}">${esc(c.name)}</option>`).join('');
  document.getElementById('cm-nc-category').innerHTML = opts;
  document.getElementById('cm-edit-category').innerHTML = opts;
}

function _cmRenderCodeList() {
  const list = document.getElementById('cm-code-list');
  if (!_cmCodes.length) {
    list.innerHTML = '<p class="text-sm text-gray-400 text-center py-8">No codes yet. Click "+ New Code" to create your first code.</p>';
    return;
  }
  const grouped = {};
  _cmCodes.forEach(c => {
    const key = c.category_id ?? '__none__';
    (grouped[key] = grouped[key] || []).push(c);
  });
  const sections = [];
  const sortedCats = [..._cmCategories].sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }));
  sortedCats.forEach(cat => {
    const codes = grouped[cat.id];
    if (!codes) return;
    sections.push(_cmCategorySection(cat, codes));
  });
  if (grouped['__none__']?.length) {
    sections.push(_cmCategorySection(null, grouped['__none__']));
  }
  list.innerHTML = sections.join('');
}

function _cmCategorySection(cat, codes) {
  const catHeader = cat
    ? `<div class="flex items-center gap-2 mb-2">
         <span class="w-2.5 h-2.5 rounded-full shrink-0" style="background:${cat.color}"></span>
         <span class="text-xs font-semibold text-gray-600 uppercase tracking-wide">${esc(cat.name)}</span>
       </div>`
    : `<div class="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">Uncategorized</div>`;
  const cards = codes.map(c => {
    const tc       = labelTextColor(c.color);
    const selected = _cmSelected.has(c.id);
    const selCls   = selected ? 'ring-2 ring-amber-400 bg-amber-50' : 'bg-white hover:border-indigo-200';
    return `
      <div class="cm-code-card border rounded-xl p-3 flex items-start gap-3 cursor-pointer transition-all ${selCls}"
           data-code-id="${c.id}">
        ${_cmMergeMode ? `<input type="checkbox" class="mt-0.5 accent-amber-500 shrink-0 cm-select-cb" ${selected ? 'checked' : ''} data-code-id="${c.id}" />` : ''}
        <span class="w-3 h-3 rounded-full shrink-0 mt-0.5" style="background:${c.color}"></span>
        <div class="flex-1 min-w-0">
          <div class="flex items-center gap-2 flex-wrap">
            <span class="text-sm font-semibold text-gray-800">${esc(c.name)}</span>
            <span class="text-[10px] px-1.5 py-0.5 rounded-full font-medium" style="background:${c.color};color:${tc}">${c.groundedness} quotes</span>
          </div>
          ${c.description ? `<p class="text-xs text-gray-500 mt-0.5 truncate">${esc(c.description)}</p>` : ''}
        </div>
        <span class="text-[10px] text-gray-400 shrink-0 mt-0.5">density ${c.density}</span>
      </div>`;
  }).join('');
  return `<div class="space-y-2 mb-4">${catHeader}${cards}</div>`;
}

function _cmOpenDetail(code) {
  _cmOpenCodeId = code.id;
  const panel = document.getElementById('cm-detail-panel');
  panel.classList.remove('hidden');
  document.getElementById('cm-edit-name').value  = code.name;
  document.getElementById('cm-edit-color').value = code.color;
  document.getElementById('cm-edit-desc').value  = code.description || '';
  document.getElementById('cm-edit-category').value = code.category_id ?? '';
  document.getElementById('cm-edit-ground').textContent  = code.groundedness ?? '—';
  document.getElementById('cm-edit-density').textContent = code.density ?? '—';
  document.getElementById('cm-edit-msg').classList.add('hidden');
}

function _cmCloseDetail() {
  _cmOpenCodeId = null;
  document.getElementById('cm-detail-panel').classList.add('hidden');
}

document.getElementById('cm-code-list').addEventListener('click', e => {
  const cb = e.target.closest('.cm-select-cb');
  if (cb) {
    const id = parseInt(cb.dataset.codeId);
    if (_cmSelected.has(id)) _cmSelected.delete(id); else _cmSelected.add(id);
    _cmUpdateMergeBtn();
    _cmRenderCodeList();
    return;
  }
  const card = e.target.closest('.cm-code-card');
  if (!card) return;
  if (_cmMergeMode) {
    const id = parseInt(card.dataset.codeId);
    if (_cmSelected.has(id)) _cmSelected.delete(id); else _cmSelected.add(id);
    _cmUpdateMergeBtn();
    _cmRenderCodeList();
    return;
  }
  const code = _cmCodes.find(c => c.id === parseInt(card.dataset.codeId));
  if (code) _cmOpenDetail(code);
});

document.getElementById('cm-edit-save').addEventListener('click', async () => {
  if (_cmOpenCodeId === null) return;
  const name        = document.getElementById('cm-edit-name').value.trim();
  const color       = document.getElementById('cm-edit-color').value;
  const description = document.getElementById('cm-edit-desc').value.trim();
  const catVal      = document.getElementById('cm-edit-category').value;
  const category_id = catVal ? parseInt(catVal) : null;
  if (!name) return;
  try {
    await apiFetch(`/api/codes/${_cmOpenCodeId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, color, description, category_id }),
    });
    const msgEl = document.getElementById('cm-edit-msg');
    msgEl.textContent = 'Saved.';
    msgEl.classList.remove('hidden');
    setTimeout(() => msgEl.classList.add('hidden'), 2500);
    _allCodes = _allCodes.map(c => c.id === _cmOpenCodeId ? { ...c, name, color, description, category_id } : c);
    _cachedBookmarks.forEach(bm => { (bm.codes || []).forEach(c => { if (c.id === _cmOpenCodeId) { c.name = name; c.color = color; } }); });
    renderBmCodeFilterChips();
    await _cmRefresh();
    document.dispatchEvent(new CustomEvent('codebook-updated'));
  } catch (e) { showErrorPopup('Failed to save: ' + e.message); }
});

document.getElementById('cm-edit-delete').addEventListener('click', async () => {
  if (_cmOpenCodeId === null) return;
  const code = _cmCodes.find(c => c.id === _cmOpenCodeId);
  if (!code) return;
  if (!confirm(`Delete code "${code.name}"? It will be removed from all bookmarks.`)) return;
  const deletingId = _cmOpenCodeId;
  try {
    await apiFetch(`/api/codes/${deletingId}`, { method: 'DELETE' });
    _cmCloseDetail();
    _allCodes = _allCodes.filter(c => c.id !== deletingId);
    _bmCodeFilter.delete(deletingId);
    _cachedBookmarks.forEach(bm => { bm.codes = (bm.codes || []).filter(c => c.id !== deletingId); });
    renderBmCodeFilterChips();
    await _cmRefresh();
    document.dispatchEvent(new CustomEvent('codebook-updated'));
  } catch (e) { showErrorPopup('Failed to delete: ' + e.message); }
});

document.getElementById('cm-detail-close').addEventListener('click', _cmCloseDetail);

document.getElementById('cm-new-code-btn').addEventListener('click', () => {
  document.getElementById('cm-new-code-panel').classList.toggle('hidden');
  document.getElementById('cm-new-cat-panel').classList.add('hidden');
  document.getElementById('cm-nc-name').focus();
});
document.getElementById('cm-nc-cancel').addEventListener('click', () => {
  document.getElementById('cm-new-code-panel').classList.add('hidden');
});
document.getElementById('cm-nc-save').addEventListener('click', async () => {
  const name        = document.getElementById('cm-nc-name').value.trim();
  const color       = document.getElementById('cm-nc-color').value;
  const catVal      = document.getElementById('cm-nc-category').value;
  const category_id = catVal ? parseInt(catVal) : null;
  const msgEl       = document.getElementById('cm-nc-msg');
  msgEl.classList.add('hidden');
  if (!name) { document.getElementById('cm-nc-name').focus(); return; }
  try {
    const code = await apiFetch('/api/codes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, color, category_id }),
    });
    document.getElementById('cm-nc-name').value  = '';
    document.getElementById('cm-nc-color').value = '#6366f1';
    document.getElementById('cm-nc-category').value = '';
    document.getElementById('cm-new-code-panel').classList.add('hidden');
    _allCodes = [..._allCodes, code].sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }));
    renderBmCodeFilterChips();
    await _cmRefresh();
    document.dispatchEvent(new CustomEvent('codebook-updated'));
  } catch (e) {
    msgEl.textContent = e.message || 'Failed to create code.';
    msgEl.classList.remove('hidden');
  }
});

document.getElementById('cm-new-cat-btn').addEventListener('click', () => {
  document.getElementById('cm-new-cat-panel').classList.toggle('hidden');
  document.getElementById('cm-new-code-panel').classList.add('hidden');
  document.getElementById('cm-cat-name').focus();
});
document.getElementById('cm-cat-cancel').addEventListener('click', () => {
  document.getElementById('cm-new-cat-panel').classList.add('hidden');
});
document.getElementById('cm-cat-save').addEventListener('click', async () => {
  const name  = document.getElementById('cm-cat-name').value.trim();
  const color = document.getElementById('cm-cat-color').value;
  const msgEl = document.getElementById('cm-cat-msg');
  msgEl.classList.add('hidden');
  if (!name) { document.getElementById('cm-cat-name').focus(); return; }
  try {
    await apiFetch('/api/code-categories', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, color }),
    });
    document.getElementById('cm-cat-name').value  = '';
    document.getElementById('cm-cat-color').value = '#94a3b8';
    document.getElementById('cm-new-cat-panel').classList.add('hidden');
    await _cmRefresh();
  } catch (e) {
    msgEl.textContent = e.message || 'Failed to create category.';
    msgEl.classList.remove('hidden');
  }
});

document.getElementById('cm-select-mode-toggle').addEventListener('change', e => {
  _cmMergeMode = e.target.checked;
  _cmSelected.clear();
  document.getElementById('cm-merge-btn').classList.add('hidden');
  document.getElementById('cm-merge-cancel').classList.add('hidden');
  _cmRenderCodeList();
});

function _cmUpdateMergeBtn() {
  const n = _cmSelected.size;
  const mergeBtn  = document.getElementById('cm-merge-btn');
  const cancelBtn = document.getElementById('cm-merge-cancel');
  if (n === 2) {
    mergeBtn.textContent = 'Merge Selected (2)';
    mergeBtn.classList.remove('hidden');
    cancelBtn.classList.remove('hidden');
  } else {
    mergeBtn.classList.add('hidden');
    cancelBtn.classList.toggle('hidden', n === 0);
  }
}

document.getElementById('cm-merge-cancel').addEventListener('click', () => {
  _cmSelected.clear();
  document.getElementById('cm-select-mode-toggle').checked = false;
  _cmMergeMode = false;
  document.getElementById('cm-merge-btn').classList.add('hidden');
  document.getElementById('cm-merge-cancel').classList.add('hidden');
  _cmRenderCodeList();
});

document.getElementById('cm-merge-btn').addEventListener('click', async () => {
  if (_cmSelected.size !== 2) return;
  const [srcId, tgtId] = [..._cmSelected];
  const src = _cmCodes.find(c => c.id === srcId);
  const tgt = _cmCodes.find(c => c.id === tgtId);
  if (!src || !tgt) return;
  if (!confirm(`Merge "${src.name}" into "${tgt.name}"? "${src.name}" will be deleted and its bookmarks reassigned to "${tgt.name}".`)) return;
  try {
    await apiFetch('/api/codes/merge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source_id: srcId, target_id: tgtId }),
    });
    _cmSelected.clear();
    _cmMergeMode = false;
    document.getElementById('cm-select-mode-toggle').checked = false;
    document.getElementById('cm-merge-btn').classList.add('hidden');
    document.getElementById('cm-merge-cancel').classList.add('hidden');
    _allCodes = _allCodes.filter(c => c.id !== srcId);
    _cachedBookmarks.forEach(bm => {
      const hadSrc = (bm.codes || []).some(c => c.id === srcId);
      bm.codes = (bm.codes || []).filter(c => c.id !== srcId);
      if (hadSrc && !bm.codes.some(c => c.id === tgtId)) {
        const tgtCode = _allCodes.find(c => c.id === tgtId);
        if (tgtCode) bm.codes.push({ id: tgtCode.id, name: tgtCode.name, color: tgtCode.color });
      }
    });
    renderBmCodeFilterChips();
    if (_cmOpenCodeId === srcId) _cmCloseDetail();
    await _cmRefresh();
    document.dispatchEvent(new CustomEvent('codebook-updated'));
  } catch (e) { showErrorPopup('Merge failed: ' + e.message); }
});

document.getElementById('cm-refresh-btn').addEventListener('click', _cmRefresh);

document.addEventListener('codebook-updated', () => {
  if (!document.getElementById('page-coding').classList.contains('hidden')) {
    _cmRefresh();
  }
});

// -- USERS IN RANGE --------------------------------------------------------

let _usersData    = [];
let _usersSortCol = 'total_messages';
let _usersSortDir = 'desc';

async function searchUsersInRange() {
  const btn      = document.getElementById('users-search-btn');
  const dateFrom = document.getElementById('users-date-from').value;
  const dateTo   = document.getElementById('users-date-to').value;
  const suno     = document.getElementById('users-suno').value;
  const minWords = parseInt(document.getElementById('users-min-words').value, 10) || 0;

  btn.disabled    = true;
  btn.textContent = 'Searching...';

  try {
    const params = new URLSearchParams();
    if (dateFrom)    params.set('date_from', dateFrom);
    if (dateTo)      params.set('date_to', dateTo);
    if (suno !== 'all') params.set('suno_team', suno);
    if (minWords > 0) params.set('min_words', minWords);
    const scope = getScopeParam();
    if (scope) params.set('upload_ids', scope);

    _usersData = await apiFetch(`/api/search/users-in-range?${params}`);
    _usersSortCol = 'total_messages';
    _usersSortDir = 'desc';
    _sortAndRenderUsers();
    document.getElementById('users-results').classList.remove('hidden');
  } catch (e) {
    showErrorPopup(e.message);
  } finally {
    btn.disabled    = false;
    btn.textContent = 'Search';
  }
}

function _lastDayOfMonth(yearMonth) {
  const [y, m] = yearMonth.split('-').map(Number);
  return new Date(y, m, 0).toISOString().slice(0, 10);
}

function _usersActiveFilters() {
  return {
    q:            (document.getElementById('users-filter-input').value || '').toLowerCase().trim(),
    monthFrom:    document.getElementById('users-month-from').value,
    monthTo:      document.getElementById('users-month-to').value,
    minMsgs:      parseInt(document.getElementById('users-min-msgs').value,      10) || 0,
    minWeeks:     parseInt(document.getElementById('users-min-weeks').value,     10) || 0,
    minAvgWords:  parseFloat(document.getElementById('users-min-avg-words').value) || 0,
    maxAvgWords:  parseFloat(document.getElementById('users-max-avg-words').value) || 0,
  };
}

function _sortAndRenderUsers() {
  let data = [..._usersData];
  const { q, monthFrom, monthTo, minMsgs, minWeeks, minAvgWords, maxAvgWords } = _usersActiveFilters();

  if (q) data = data.filter(u => (u.username || '').toLowerCase().includes(q));

  // Month From: first_message_date must fall within that exact month
  if (monthFrom) {
    const mStart = monthFrom + '-01';
    const mEnd   = _lastDayOfMonth(monthFrom);
    data = data.filter(u => {
      const first = u.first_message_date || '';
      return first >= mStart && first <= mEnd;
    });
  }

  // Month To: last_message_date must be in that month or later
  if (monthTo) {
    const mStart = monthTo + '-01';
    data = data.filter(u => (u.last_message_date || '') >= mStart);
  }

  if (minMsgs     > 0) data = data.filter(u => (u.total_messages      || 0) >= minMsgs);
  if (minWeeks    > 0) data = data.filter(u => (u.weeks_with_messages || 0) >= minWeeks);
  if (minAvgWords > 0) data = data.filter(u => (u.avg_word_count      || 0) >= minAvgWords);
  if (maxAvgWords > 0) data = data.filter(u => (u.avg_word_count      || 0) <= maxAvgWords);

  data.sort((a, b) => {
    let av = a[_usersSortCol];
    let bv = b[_usersSortCol];
    if (av == null) av = _usersSortDir === 'asc' ? '' : '\uffff';
    if (bv == null) bv = _usersSortDir === 'asc' ? '' : '\uffff';
    if (typeof av === 'string') return _usersSortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
    return _usersSortDir === 'asc' ? av - bv : bv - av;
  });

  const tbody = document.getElementById('users-tbody');
  tbody.innerHTML = '';
  const frag = document.createDocumentFragment();

  for (const u of data) {
    const totalWeeks = u.total_weeks_in_range;
    const weeksDisp  = totalWeeks != null
      ? `${u.weeks_with_messages}<span class="text-gray-400">/${totalWeeks}</span>`
      : String(u.weeks_with_messages ?? 'â€”');
    const pctDisp    = u.pct_weeks_active != null ? `${u.pct_weeks_active}%` : 'â€”';
    const avgWords   = u.avg_word_count != null ? Number(u.avg_word_count).toFixed(1) : 'â€”';

    const teamBadge = truthy(u.is_suno_team)
      ? `<span class="ml-1.5 text-[0.6rem] bg-amber-100 text-amber-700 px-1 py-0.5 rounded font-semibold leading-none">Team</span>`
      : '';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="users-td">
        <button class="text-indigo-700 hover:underline hover:text-indigo-900 font-medium text-left
                       focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-600 rounded"
                data-username="${esc(u.username)}">${esc(u.username)}</button>${teamBadge}
      </td>
      <td class="users-td text-right tabular-nums">${(u.total_messages || 0).toLocaleString()}</td>
      <td class="users-td">${u.first_message_date || 'â€”'}</td>
      <td class="users-td">${u.last_message_date || 'â€”'}</td>
      <td class="users-td text-right tabular-nums">${avgWords}</td>
      <td class="users-td text-right tabular-nums">${weeksDisp}</td>
      <td class="users-td text-right tabular-nums">${pctDisp}</td>
    `;
    frag.appendChild(tr);
  }
  tbody.appendChild(frag);

  // Sort indicators
  document.querySelectorAll('#users-table .users-th').forEach(th => {
    const col = th.dataset.col;
    const ind = th.querySelector('.sort-ind');
    if (!ind) return;
    if (col === _usersSortCol) {
      ind.textContent = _usersSortDir === 'asc' ? ' →' : ' ↓';
      ind.style.color = '#4f46e5';
    } else {
      ind.textContent = ' ↕';
      ind.style.color = '';
    }
  });

  const n = data.length;
  const anyFilter = q || monthFrom || monthTo || minMsgs > 0 || minWeeks > 0 || minAvgWords > 0 || maxAvgWords > 0;
  document.getElementById('users-result-count').textContent =
    `${n.toLocaleString()} user${n !== 1 ? 's' : ''}` +
    (anyFilter ? ` (filtered from ${_usersData.length.toLocaleString()})` : '');
}

// Sort on header click
document.getElementById('users-table').addEventListener('click', e => {
  const th = e.target.closest('.users-th');
  if (!th) return;
  const col = th.dataset.col;
  if (!col) return;
  if (_usersSortCol === col) {
    _usersSortDir = _usersSortDir === 'asc' ? 'desc' : 'asc';
  } else {
    _usersSortCol = col;
    _usersSortDir = col === 'username' || col === 'first_message_date' || col === 'last_message_date'
      ? 'asc' : 'desc';
  }
  _sortAndRenderUsers();
});

// Username click → open profile
document.getElementById('users-tbody').addEventListener('click', e => {
  const btn = e.target.closest('[data-username]');
  if (!btn) return;
  const username = btn.dataset.username;
  const row = _usersData.find(u => u.username === username);
  openUserProfile(username, row || null);
});

// Live filters€” month pickers fire 'change'; number/text inputs fire 'input'
['users-month-from', 'users-month-to'].forEach(id => {
  document.getElementById(id).addEventListener('change', _sortAndRenderUsers);
});
['users-filter-input', 'users-min-msgs', 'users-min-weeks',
 'users-min-avg-words', 'users-max-avg-words'].forEach(id => {
  document.getElementById(id).addEventListener('input', _sortAndRenderUsers);
});
document.getElementById('users-refine-apply').addEventListener('click', _sortAndRenderUsers);

document.getElementById('users-refine-clear').addEventListener('click', () => {
  ['users-month-from', 'users-month-to', 'users-min-msgs', 'users-min-weeks',
   'users-min-avg-words', 'users-max-avg-words']
    .forEach(id => { document.getElementById(id).value = ''; });
  _sortAndRenderUsers();
});

document.getElementById('users-search-btn').addEventListener('click', searchUsersInRange);
document.getElementById('users-date-from').addEventListener('keydown', e => { if (e.key === 'Enter') searchUsersInRange(); });
document.getElementById('users-date-to').addEventListener('keydown', e => { if (e.key === 'Enter') searchUsersInRange(); });

// -- USER PROFILE OVERLAY --------------------------------------------------

let _upoUsername    = '';
let _profileMessages = [];

function _statPill(label, value) {
  return `<div class="inline-flex flex-col items-start bg-indigo-50 rounded-xl px-4 py-2.5 mr-2 mb-2">
    <span class="text-[0.65rem] font-semibold uppercase tracking-wide text-indigo-400">${label}</span>
    <span class="text-sm font-bold text-indigo-900 mt-0.5 tabular-nums">${value}</span>
  </div>`;
}

async function openUserProfile(username, stats) {
  _upoUsername = username;

  document.getElementById('upo-username').textContent = username;
  document.getElementById('upo-msg-count').textContent = '';
  document.getElementById('upo-filter-count').textContent = '';

  // Populate stats bar
  if (stats) {
    const totalWeeks = stats.total_weeks_in_range;
    const weeksStr = totalWeeks != null
      ? `${stats.weeks_with_messages} / ${totalWeeks}`
      : String(stats.weeks_with_messages ?? 'â€”');
    const pctStr = stats.pct_weeks_active != null ? `${stats.pct_weeks_active}%` : 'â€”';
    const avgStr = stats.avg_word_count != null ? Number(stats.avg_word_count).toFixed(1) : 'â€”';

    document.getElementById('upo-stats').innerHTML =
      `<div class="flex flex-wrap">` +
      _statPill('Total Messages', (stats.total_messages || 0).toLocaleString()) +
      _statPill('First Message',  stats.first_message_date || 'â€”') +
      _statPill('Last Message',   stats.last_message_date  || 'â€”') +
      _statPill('Avg Words',      avgStr) +
      _statPill('Weeks Active',   weeksStr) +
      _statPill('% Weeks Active', pctStr) +
      `</div>`;
  } else {
    document.getElementById('upo-stats').innerHTML = '';
  }

  // Pre-fill date filters from the users-in-range search
  document.getElementById('upo-date-from').value = document.getElementById('users-date-from').value;
  document.getElementById('upo-date-to').value   = document.getElementById('users-date-to').value;
  document.getElementById('upo-keyword').value   = '';

  // Show overlay
  const overlay = document.getElementById('user-profile-overlay');
  overlay.classList.remove('hidden');
  overlay.classList.add('flex');
  document.body.style.overflow = 'hidden';

  await _fetchProfileMessages();
}

function closeUserProfile() {
  const overlay = document.getElementById('user-profile-overlay');
  overlay.classList.add('hidden');
  overlay.classList.remove('flex');
  document.body.style.overflow = '';

  // Reset summary panel so the next profile starts clean
  document.getElementById('upo-sum-panel').classList.add('hidden');
  document.getElementById('upo-sum-results').classList.add('hidden');
  document.getElementById('upo-sum-output').innerHTML = '';
  document.getElementById('upo-sum-log').innerHTML = '';
  document.getElementById('upo-sum-prompt').value = '';
  document.getElementById('upo-sum-export-pdf').classList.add('hidden');
}

async function _fetchProfileMessages() {
  const msgEl    = document.getElementById('upo-messages');
  const dateFrom = document.getElementById('upo-date-from').value;
  const dateTo   = document.getElementById('upo-date-to').value;
  const keyword  = document.getElementById('upo-keyword').value.trim();
  const filterEl = document.getElementById('upo-filter-count');

  msgEl.innerHTML = '<p class="text-sm text-gray-400 py-6 text-center">Loading...</p>';
  filterEl.textContent = '';

  try {
    const params = new URLSearchParams({ username: _upoUsername });
    if (dateFrom) params.set('date_from', dateFrom);
    if (dateTo)   params.set('date_to', dateTo);
    if (keyword)  params.set('keyword', keyword);
    const scope = getScopeParam();
    if (scope) params.set('upload_ids', scope);

    const msgs = await apiFetch(`/api/search/user-messages?${params}`);

    document.getElementById('upo-msg-count').textContent =
      `${msgs.length.toLocaleString()} message${msgs.length !== 1 ? 's' : ''}`;
    filterEl.textContent = keyword || dateFrom || dateTo
      ? `${msgs.length.toLocaleString()} result${msgs.length !== 1 ? 's' : ''}`
      : '';

    if (!msgs.length) {
      _profileMessages = [];
      msgEl.innerHTML = '<p class="text-sm text-gray-400 py-8 text-center">No messages found.</p>';
      return;
    }

    _profileMessages = msgs;
    msgEl.innerHTML = '';
    const frag = document.createDocumentFragment();
    for (const msg of msgs) {
      const card = document.createElement('div');
      card.className = 'bg-white rounded-xl border border-gray-200 shadow-sm p-3';
      const safeContent = keyword ? highlight(msg.content || '', keyword) : esc(msg.content || '');
      card.innerHTML = `
        <div class="flex items-center justify-between mb-1.5 gap-2">
          <span class="text-xs text-gray-400">${esc(msg.date || '')}</span>
          <div class="flex gap-1.5">
            ${truthy(msg.is_suno_team)
              ? `<span class="text-[0.65rem] bg-indigo-100 text-indigo-700 px-1.5 py-0.5 rounded font-semibold">Suno Team</span>`
              : ''}
            ${msg.upload_id
              ? `<span class="text-[0.65rem] bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded">${esc(String(msg.upload_id))}</span>`
              : ''}
          </div>
        </div>
        <p class="text-sm text-gray-800 whitespace-pre-wrap break-words leading-relaxed">${safeContent}</p>
        <div class="border-t border-gray-100 mt-2 pt-1.5 flex justify-end">
          <button class="upo-ctx-btn text-xs text-indigo-600 hover:text-indigo-800 font-medium
                         focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-600 rounded"
                  data-id="${msg.id}" data-open="false">Show context ↕</button>
        </div>
        <div id="upo-ctx-${msg.id}" class="hidden mt-2"></div>
      `;
      frag.appendChild(card);
    }
    msgEl.appendChild(frag);
  } catch (e) {
    msgEl.innerHTML = `<p class="text-sm text-red-600 py-4 text-center">Error: ${esc(e.message)}</p>`;
  }
}

/* -- User profile summarize ----------------------------------------------- */

document.getElementById('upo-sum-toggle').addEventListener('click', () => {
  const panel  = document.getElementById('upo-sum-panel');
  const hidden = panel.classList.toggle('hidden');
  document.getElementById('upo-sum-toggle').textContent = hidden ? 'âœ¦ Summarize' : 'âœ¦ Hide Summary';
  if (!hidden) {
    const before = document.getElementById('upo-ctx-before').value || 5;
    const after  = document.getElementById('upo-ctx-after').value  || 5;
    document.getElementById('upo-sum-ctx-hint').textContent = `${before} before / ${after} after`;
    // Default limit to all currently loaded messages
    if (_profileMessages.length) {
      document.getElementById('upo-sum-limit').value = _profileMessages.length;
    }
  }
});

document.getElementById('upo-sum-log-toggle').addEventListener('click', () => {
  const logEl = document.getElementById('upo-sum-log');
  const btn   = document.getElementById('upo-sum-log-toggle');
  const hide  = btn.textContent.startsWith('â–²');
  logEl.classList.toggle('hidden', hide);
  btn.textContent = hide ? 'â–¼ Show' : 'â–² Hide';
});

function _upoSumLog(step, label, msg) {
  const icons = { input:'ðŸ“‹', context:'ðŸ“¡', llm:'âœ¨', fallback:'âš ï¸' };
  const div = document.createElement('div');
  div.className = 'text-xs text-gray-600 flex items-start gap-1.5 py-0.5';
  div.innerHTML = `<span class="shrink-0">${icons[step] || 'â€¢'}</span>
    <span><strong>${esc(label)}</strong>€” ${esc(msg)}</span>`;
  document.getElementById('upo-sum-log').appendChild(div);
}

async function doUpoSummarize() {
  const runBtn   = document.getElementById('upo-sum-run');
  const outputEl = document.getElementById('upo-sum-output');
  const logEl    = document.getElementById('upo-sum-log');
  const resultsEl= document.getElementById('upo-sum-results');
  const limit    = parseInt(document.getElementById('upo-sum-limit').value, 10) || _profileMessages.length;
  const model    = document.getElementById('upo-sum-model').value;
  const prompt   = document.getElementById('upo-sum-prompt').value.trim();
  const before   = Math.min(parseInt(document.getElementById('upo-ctx-before').value, 10) || 5, 50);
  const after    = Math.min(parseInt(document.getElementById('upo-ctx-after').value,  10) || 5, 50);

  if (!_profileMessages.length) {
    showErrorPopup('No messages loaded. Filter the profile first.');
    return;
  }

  runBtn.disabled    = true;
  runBtn.textContent = 'Working...';
  logEl.innerHTML    = '';
  outputEl.innerHTML = '';
  resultsEl.classList.remove('hidden');
  document.getElementById('upo-sum-export-pdf').classList.add('hidden');

  const msgs = _profileMessages.slice(0, limit);
  _upoSumLog('input', 'Input', `${msgs.length} messages from ${esc(_upoUsername)}`);

  // Fetch context for each message
  let contextMap = {};
  try {
    const msgIds = msgs.map(m => m.id).filter(Boolean);
    _upoSumLog('context', 'Context fetch', `Fetching ${before}+${after} context msgs for each...`);
    contextMap = await apiFetch('/api/search/bulk-context', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ msg_ids: msgIds, before, after }),
    });
    _upoSumLog('context', 'Context fetch', `Done€” ${Object.keys(contextMap).length} messages enriched`);
  } catch (e) {
    _upoSumLog('fallback', 'Context fetch failed', `${e.message}€” proceeding without context`);
  }

  // Build formatted blocks: context before /˜… user message / context after
  const blocks = msgs.map(m => {
    const ctx      = contextMap[String(m.id)] || [];
    const targetIdx= ctx.findIndex(r => r.is_target);
    const ctxPre   = targetIdx > 0 ? ctx.slice(0, targetIdx) : [];
    const ctxPost  = targetIdx >= 0 ? ctx.slice(targetIdx + 1) : [];
    const fmt      = r => `  [${r.username}]: ${r.content}`;

    let block = '';
    if (ctxPre.length)  block += ctxPre.map(fmt).join('\n') + '\n';
    block += `â˜… [${m.username} | ${m.date}]: ${m.content}`;
    if (ctxPost.length) block += '\n' + ctxPost.map(fmt).join('\n');
    return block;
  });

  const conv   = blocks.join('\n\n---\n\n');
  const n      = msgs.length;
  const header = `USER PROFILE ANALYSIS€” ${_upoUsername} (${n} messages with context)`;

  const defaultPrompt = `Each block below contains one message from **${_upoUsername}** (marked˜…) with surrounding conversation context. Concisely identify persona, topics, attitudes, actions, narratives, and identified changes in attitude and stance if present. Use tight bullet points. No padding, no repetition across sections.`;

  const fullPrompt = (prompt || defaultPrompt) + `\n\n${header}:\n${conv}`;
  _upoSumLog('llm', 'LLM generation', `Summarising with ${model}...`);

  let output = '';
  try {
    const res = await fetch('/api/summarize-results', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages:       [{ username: _upoUsername, date: '', content: fullPrompt }],
        model,
        retrieval_mode: 'all',
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
          if (delta.content) {
            output += delta.content;
            outputEl.innerHTML = marked.parse(output);
          } else if (delta.error) {
            throw new Error(delta.error);
          }
        } catch (parseErr) {
          if (!(parseErr instanceof SyntaxError)) throw parseErr;
        }
      }
    }
  } catch (e) {
    outputEl.innerHTML = `<p class="text-red-600 text-sm">Error: ${esc(e.message)}</p>`;
  } finally {
    runBtn.disabled    = false;
    runBtn.textContent = 'Run Summary';
    if (output) {
      document.getElementById('upo-sum-export-pdf').classList.remove('hidden');
    }
  }
}

function exportUpoSumPDF() {
  const outputEl  = document.getElementById('upo-sum-output');
  const summaryHTML = outputEl.innerHTML;
  if (!summaryHTML.trim()) return;

  const dateStr = new Date().toLocaleDateString('en-US', {
    year: 'numeric', month: 'long', day: 'numeric',
  });
  const username    = _upoUsername || 'User';
  const pdfFilename = `UserSummary_${username}_${new Date().toISOString().slice(0, 10)}`;

  const customInstructions = (document.getElementById('upo-sum-prompt').value || '').trim();
  let instructionHTML = '';
  if (customInstructions) {
    const safe = customInstructions
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    instructionHTML = `<div class="custom-instruction"><span class="ci-label">Custom Instructions</span>${safe}</div>`;
  }

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>${pdfFilename}</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  max-width: 820px; margin: 40px auto; padding: 0 28px;
  color: #1e293b; line-height: 1.65; font-size: 14px;
}
h1  { font-size: 1.5rem; color: #4f46e5; padding-bottom: 10px;
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
  border-left: 3px solid #4f46e5; margin: 0.75rem 0;
  padding: 6px 14px; background: #eef2ff; color: #312e81;
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
a      { color: #4f46e5; text-decoration: underline; }
table  { border-collapse: collapse; width: 100%; margin-bottom: 0.65rem; font-size: 0.85rem; }
th, td { border: 1px solid #e2e8f0; padding: 6px 10px; text-align: left; }
th     { background: #f8fafc; font-weight: 700; }
.custom-instruction {
  background: #fefce8; border: 1px solid #fde68a; border-radius: 6px;
  padding: 8px 14px; margin-bottom: 1.75rem; font-size: 0.8rem; color: #713f12;
}
.ci-label {
  display: block; font-size: 0.68rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.06em;
  color: #92400e; margin-bottom: 4px;
}
@media print { body { margin: 16px 28px; } }
</style>
</head>
<body>
<h1>User Summary: ${username.replace(/</g, '&lt;')}</h1>
<p class="meta">Exported ${dateStr}</p>
${instructionHTML}
<div class="summary-body">${summaryHTML}</div>
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
  win.addEventListener('load', () => URL.revokeObjectURL(url), { once: true });
}

document.getElementById('upo-sum-run').addEventListener('click', doUpoSummarize);
document.getElementById('upo-sum-export-pdf').addEventListener('click', exportUpoSumPDF);

async function upoToggleContext(id, btn) {
  const ctxEl = document.getElementById(`upo-ctx-${id}`);
  if (btn.dataset.open === 'true') {
    ctxEl.classList.add('hidden');
    btn.dataset.open = 'false';
    btn.textContent  = 'Show context ↕';
    return;
  }
  const before = parseInt(document.getElementById('upo-ctx-before').value, 10) || 5;
  const after  = parseInt(document.getElementById('upo-ctx-after').value,  10) || 5;
  btn.textContent = 'Loading...';
  btn.disabled    = true;
  try {
    const msgs = await apiFetch(`/api/context/${id}?before=${before}&after=${after}`);
    ctxEl.innerHTML = `
      <div class="bg-slate-50 rounded-lg border border-slate-200 p-3 space-y-2">
        <p class="text-xs text-gray-500 font-medium mb-2">
          ${msgs.length} messages &mdash; ${before} before &bull; ${after} after
        </p>
        ${msgs.map(m => ctxMsg(m)).join('')}
      </div>`;
    ctxEl.classList.remove('hidden');
    btn.dataset.open = 'true';
    btn.textContent  = 'Hide context ↕';
  } catch (e) {
    btn.textContent = 'Show context ↕';
    console.error(e);
  } finally {
    btn.disabled = false;
  }
}

document.getElementById('upo-messages').addEventListener('click', e => {
  const btn = e.target.closest('.upo-ctx-btn');
  if (!btn) return;
  upoToggleContext(parseInt(btn.dataset.id, 10), btn);
});

document.getElementById('upo-back').addEventListener('click', closeUserProfile);
document.getElementById('upo-filter-btn').addEventListener('click', _fetchProfileMessages);
document.getElementById('upo-date-from').addEventListener('change', _fetchProfileMessages);
document.getElementById('upo-date-to').addEventListener('change', _fetchProfileMessages);
document.getElementById('upo-keyword').addEventListener('keydown', e => { if (e.key === 'Enter') _fetchProfileMessages(); });
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && !document.getElementById('user-profile-overlay').classList.contains('hidden')) {
    closeUserProfile();
  }
});

// -- MULTI-USER: avatar menu (dropdown with Config + Logout) ---------------
if (APP_MODE === 'multi' && CURRENT_USER) {
  const menu        = document.getElementById('user-menu');
  const avatarBtn   = document.getElementById('user-avatar-btn');
  const dropdown    = document.getElementById('user-dropdown');
  const initialEl   = document.getElementById('user-avatar-initial');
  const nameEl      = document.getElementById('user-dropdown-name');
  const adminBtn    = document.getElementById('user-menu-admin');
  const configBtn   = document.getElementById('user-menu-config');
  const logoutBtn   = document.getElementById('logout-btn');

  // Populate avatar
  if (initialEl) initialEl.textContent = CURRENT_USER.charAt(0).toUpperCase();
  if (nameEl)    nameEl.textContent    = CURRENT_USER;

  // Show the avatar menu; hide the standalone Config nav tab (replaced by dropdown)
  menu.classList.remove('hidden');
  menu.classList.add('block');
  const navSettings = document.getElementById('nav-settings');
  if (navSettings) navSettings.classList.add('hidden');

  // Toggle dropdown open/closed
  function _openMenu() {
    dropdown.classList.remove('hidden');
    avatarBtn.setAttribute('aria-expanded', 'true');
  }
  function _closeMenu() {
    dropdown.classList.add('hidden');
    avatarBtn.setAttribute('aria-expanded', 'false');
  }

  avatarBtn.addEventListener('click', e => {
    e.stopPropagation();
    dropdown.classList.contains('hidden') ? _openMenu() : _closeMenu();
  });

  // Close when clicking outside
  document.addEventListener('click', e => {
    if (!menu.contains(e.target)) _closeMenu();
  });

  // Close on Escape
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') _closeMenu();
  });

  // Admin → navigate to admin page
  if (adminBtn) {
    adminBtn.addEventListener('click', () => {
      _closeMenu();
      navigateTo('admin');
    });
  }

  // Config → navigate to settings page
  if (configBtn) {
    configBtn.addEventListener('click', () => {
      _closeMenu();
      navigateTo('settings');
    });
  }

  // Log Out
  if (logoutBtn) {
    logoutBtn.addEventListener('click', async () => {
      logoutBtn.disabled    = true;
      logoutBtn.textContent = 'Logging out...';
      try { await fetch('/api/auth/logout', { method: 'POST' }); } catch (_) {}
      window.location.href = '/login';
    });
  }
}

// -- INIT ------------------------------------------------------------------
(async () => {
  await refreshUploads();   // loads allUploads, renders scope chips, stats
  await loadBookmarkIds();  // populate bookmarkedIds set + badge

  if (APP_MODE === 'multi') {
    // Resolve admin status, then apply UI visibility rules.
    try {
      const me = await apiFetch('/api/auth/me');
      currentUserIsAdmin = !!me.is_admin;
    } catch (_) {}
    applyAdminUI();

    // Restore API key from localStorage into server memory for this session.
    // The key is never stored server-side; localStorage is the sole durable store.
    const storedKey = localStorage.getItem(STORAGE_KEY);
    if (storedKey) {
      try { await _sendKeyToServer(storedKey); } catch (_) {
        localStorage.removeItem(STORAGE_KEY);
      }
    }

    // Load stats and prompt for API key if not set in localStorage.
    try {
      const d = await apiFetch('/api/stats');
      const keySet = !!localStorage.getItem(STORAGE_KEY);
      document.getElementById('stats-bar').innerHTML =
        `${d.total_messages.toLocaleString()} msgs &bull; ` +
        `${d.total_uploads} uploads &bull; ` +
        `${d.embedded_messages.toLocaleString()} embedded &bull; ` +
        `<span style="color:#c4b5fd">${esc(d.current_model_label)}</span>` +
        (keySet ? ' &bull; <span style="color:#86efac">API key ...</span>' : '');
      if (!keySet) showApiKeyPopup(true);
    } catch (_) {}
  } else {
    // Single mode: restore API key from localStorage → send to server.
    // If no key is stored yet, show the popup so the user can enter one.
    const storedKey = localStorage.getItem(STORAGE_KEY);
    if (storedKey) {
      try {
        await _sendKeyToServer(storedKey);
        loadStats();
      } catch (_) {
        // Stored key is invalid/rejected€” clear it and prompt again
        localStorage.removeItem(STORAGE_KEY);
        showApiKeyPopup(false);
      }
    } else {
      showApiKeyPopup(false);
    }
  }
})();
