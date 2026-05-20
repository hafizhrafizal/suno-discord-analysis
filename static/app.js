// -- MARKED CONFIGURATION --------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  if (typeof marked !== 'undefined') {
    marked.use({ gfm: true, breaks: true });
  }
  _initStaticColorPickers();
});

/*
   FETCH HELPER
   Wraps fetch() so non-JSON responses (proxy 502/504, nginx error pages,
   HTTP→HTTPS redirects) produce a clear error instead of "Unexpected token '<'".
 */
async function apiFetch(url, options = {}) {
  if (options.body && typeof options.body === 'string') {
    options = {
      ...options,
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    };
  }
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

// -- GENERIC CONFIRM DIALOG ------------------------------------------------
function cmConfirm(msg, okLabel = 'Confirm', okClass = 'bg-red-600 hover:bg-red-700') {
  return new Promise(resolve => {
    const modal    = document.getElementById('app-confirm-modal');
    const okBtn    = document.getElementById('app-confirm-ok');
    const cancelBtn = document.getElementById('app-confirm-cancel');
    document.getElementById('app-confirm-msg').textContent = msg;
    okBtn.textContent = okLabel;
    okBtn.className   = `px-4 py-2 text-sm font-semibold rounded-lg text-white ${okClass}`;
    modal.classList.remove('hidden');

    function cleanup(result) {
      modal.classList.add('hidden');
      okBtn.removeEventListener('click', onOk);
      cancelBtn.removeEventListener('click', onCancel);
      document.removeEventListener('keydown', onKey);
      resolve(result);
    }
    const onOk     = () => cleanup(true);
    const onCancel = () => cleanup(false);
    const onKey    = e => { if (e.key === 'Escape') cleanup(false); };
    okBtn.addEventListener('click', onOk);
    cancelBtn.addEventListener('click', onCancel);
    document.addEventListener('keydown', onKey);
  });
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

// Demo mode: Settings nav tab stays visible but only shows the API key section

// -- STATS -----------------------------------------------------------------
async function loadStats() {
  try {
    const d = await apiFetch('/api/stats');
    document.getElementById('stats-bar').innerHTML =
      `${d.total_messages.toLocaleString()} msgs &bull; ` +
      `${d.total_uploads} dataset &bull; ` +
      `${d.embedded_messages.toLocaleString()} embedded &bull; ` ;
  } catch (_) {}
}

// -- SCOPE SELECTOR (Search page - which uploads to search) ----------------
function renderScopeChips() {
  const container = document.getElementById('scope-chips');
  if (!allUploads.length) {
    container.innerHTML = '<span class="text-xs text-gray-400 italic">No uploads yet go to the Data page to add data.</span>';
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
            (isLocal && cnt === 0 ? ' Weights will download on first use (~0.4–1.3 GB).' : '');
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

  // Key is always stored in browser localStorage only never on the server.
  const stored = localStorage.getItem(STORAGE_KEY) || '';
  input.value = stored;
  if (descEl) descEl.innerHTML = 'Stored in <strong>your browser\'s localStorage</strong> only never saved to the server or database. Sent to your own server per session to make OpenAI requests on your behalf.';

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
    ? 'API key saved in your browser (localStorage) not stored on the server.'
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
      : `<span class="embed-badge embed-badge-no">${labels[mid] || mid}</span>`;
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
    if (labelEl) labelEl.textContent = 'Failed see details below';
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
      setProgress(0, 'Job already running resuming progress display...');
    } else {
      const skip = d.skipped || 0;
      setProgress(0, skip > 0
        ? `Resuming: ${skip.toLocaleString()} already embedded, checking remainder...`
        : `Job started ${(d.total_messages || 0).toLocaleString()} messages queued`);
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
      if (!r.ok) return;   // transient keep polling
      job = await r.json();
    } catch {
      return;              // network blip keep polling
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
        setProgress(pct, `Embedding... ${pct}% ${embedded.toLocaleString()}/${total.toLocaleString()} new messages${skipNote}${batchInfo}`);
      }

    } else if (job.status === 'completed') {
      const skipNote = skipped > 0 ? `, ${skipped.toLocaleString()} already embedded` : '';
      const errNote  = job.batch_errors.length > 0 ? ` (${job.batch_errors.length} batch error(s) see below)` : '';
      setProgress(100, `Done ${embedded.toLocaleString()} embedded${skipNote}${errNote}`);

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
      throw new Error(`Server returned a non-JSON response (HTTP ${res.status}). The operation may have timed out check server logs.`);
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

  // Account section visible in multi mode only
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
  // Demo mode: show only the OpenAI API key section, hide everything else
  const demoOnlySections = [
    'section-account', 'section-embed-model', 'section-upload',
    'section-uploads', 'section-suno-team',
  ];
  demoOnlySections.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('hidden', APP_MODE === 'demo');
  });

  if (APP_MODE !== 'demo') {
    applyAdminUI();
    loadModelOptions();
    renderUploadsTable();
    renderLabelManager();
    if (currentUserIsAdmin) renderSunoTeamTable();
    document.getElementById('goto-coding-btn')?.addEventListener('click', () => navigateTo('coding'));
  }

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
            ? '<span class="text-xs text-gray-400 italic">â</span>'
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
      if (!await cmConfirm(`Delete user "${name}"? This cannot be undone.`, 'Delete')) return;
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
  if (!list) return;
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
                      data-code-id="${l.id}" data-code-name="${esc(l.name)}" title="Delete code">—</button>
            </span>`;
  }).join('');
}

document.getElementById('labels-list')?.addEventListener('click', async e => {
  const btn = e.target.closest('.label-delete-btn');
  if (!btn) return;
  const id   = parseInt(btn.dataset.codeId);
  const name = btn.dataset.codeName;
  if (!await cmConfirm(`Delete code "${name}"? It will be removed from all bookmarks.`, 'Delete')) return;
  btn.disabled = true;
  const res = await fetch(`/api/codes/${id}`, { method: 'DELETE' });
  if (!res.ok) { btn.disabled = false; return; }
  _allCodes = _allCodes.filter(l => l.id !== id);
  _bmCodeFilter.delete(id);
  _cachedBookmarks.forEach(bm => { bm.codes = (bm.codes || []).filter(l => l.id !== id); });
  renderLabelManager();
  renderBmCodeFilterChips();
});

document.getElementById('label-create-form')?.addEventListener('submit', async e => {
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
    showErrorPopup(`Failed to remove: ${err.message}`);
  }
});

// -- SEARCH TABS -----------------------------------------------------------

// Top-level Chat / Users tab switching
document.querySelectorAll('[data-cat]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-cat]').forEach(b => {
      b.classList.remove('subtab-btn-active');
      b.setAttribute('aria-selected', 'false');
    });
    btn.classList.add('subtab-btn-active');
    btn.setAttribute('aria-selected', 'true');
    document.getElementById('search-cat-panel-chat').classList.add('hidden');
    document.getElementById('search-cat-panel-users').classList.add('hidden');
    document.getElementById('search-cat-panel-' + btn.dataset.cat).classList.remove('hidden');
  });
});

// Inner chat search tab switching
document.querySelectorAll('.search-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.search-tab').forEach(b => b.classList.remove('tab-active'));
    btn.classList.add('tab-active');
    document.querySelectorAll('.search-panel').forEach(p => p.classList.add('hidden'));
    document.getElementById('tab-' + btn.dataset.tab).classList.remove('hidden');

    // Flush results, summary, and stats visualization when switching search mode
    document.getElementById('results-section').classList.add('hidden');
    document.getElementById('results-container').innerHTML = '';
    document.getElementById('sr-section').classList.add('hidden');
    document.getElementById('trend-section').classList.add('hidden');
    const _tabbar = document.getElementById('sr-viz-tabbar');
    if (_tabbar) _tabbar.classList.add('hidden');
    const _banner = document.getElementById('chart-filter-banner');
    if (_banner) _banner.remove();
    if (_trendChart) { _trendChart.destroy(); _trendChart = null; }
    _trendRanges = [];
    currentResults = [];
  });
});

// Keyword match type toggle
let keywordMatchType = 'fuzzy';
['fuzzy', 'exact', 'any'].forEach(mt => {
  document.getElementById(`kw-match-${mt}`).addEventListener('click', () => {
    keywordMatchType = mt;
    ['fuzzy', 'exact', 'any'].forEach(t => {
      const b = document.getElementById(`kw-match-${t}`);
      b.classList.toggle('range-mode-active', t === mt);
      b.setAttribute('aria-pressed', t === mt ? 'true' : 'false');
    });
  });
});

[
  ['username-input',           'username'],
  ['username-date-from',       'username'],
  ['username-date-to',         'username'],
  ['username-limit',           'username'],
  ['username-min-words',       'username'],
  ['keyword-input',            'keyword'],
  ['keyword-username-filter',  'keyword'],
  ['keyword-date-from',        'keyword'],
  ['keyword-date-to',          'keyword'],
  ['keyword-limit',            'keyword'],
  ['keyword-min-words',        'keyword'],
  ['semantic-input',           'semantic'],
  ['semantic-username-filter', 'semantic'],
  ['semantic-date-from',       'semantic'],
  ['semantic-date-to',         'semantic'],
  ['semantic-n',               'semantic'],
  ['semantic-min-words',       'semantic'],
].forEach(([id, type]) => {
  document.getElementById(id)?.addEventListener('keydown', e => {
    if (e.key === 'Enter') doSearch(type);
  });
});

// Collapsible options panels
['keyword', 'username', 'semantic'].forEach(type => {
  const toggle = document.getElementById(`${type}-opts-toggle`);
  const opts   = document.getElementById(`${type}-opts`);
  if (!toggle || !opts) return;
  toggle.addEventListener('click', () => {
    const nowHidden = opts.classList.toggle('hidden');
    toggle.setAttribute('aria-expanded', String(!nowHidden));
    toggle.querySelector('svg').style.transform = nowHidden ? '' : 'rotate(180deg)';
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
    url = `/api/search/keyword?keyword=${enc(keyword)}&limit=${limit}&match_type=${keywordMatchType}`;
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
  currentResults    = [];
  document.getElementById('results-section').classList.add('hidden');
  document.getElementById('results-container').innerHTML = '';
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
    // 'score' restore original API order (highest similarity first)
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

// -- Scroll pagination state -----------------------------------------------
const PAGE_SIZE   = 50;
let _pageItems    = [];   // full list being paged
let _pageOffset   = 0;    // how many have been rendered
let _pageObserver = null; // IntersectionObserver

function _stopPageObserver() {
  if (_pageObserver) { _pageObserver.disconnect(); _pageObserver = null; }
  const s = document.getElementById('scroll-sentinel');
  if (s) s.remove();
}

function _appendResultCards(msgs) {
  const container = document.getElementById('results-container');
  const sentinel  = document.getElementById('scroll-sentinel');
  if (sentinel) sentinel.remove();
  const temp = document.createElement('div');
  temp.innerHTML = msgs.map(m => msgCard(m)).join('');
  temp.querySelectorAll('.ctx-toggle').forEach(btn => {
    btn.addEventListener('click', () => toggleContext(parseInt(btn.dataset.id), btn));
  });
  while (temp.firstChild) container.appendChild(temp.firstChild);
}

function _renderNextPage() {
  const page = _pageItems.slice(_pageOffset, _pageOffset + PAGE_SIZE);
  if (!page.length) { _stopPageObserver(); return; }
  // Disconnect old observer before appending — do NOT call _stopPageObserver()
  // here as that would also remove the sentinel we're about to create.
  if (_pageObserver) { _pageObserver.disconnect(); _pageObserver = null; }
  _appendResultCards(page);  // removes previous sentinel, appends new cards
  _pageOffset += page.length;
  if (_pageOffset < _pageItems.length) {
    const sentinel = document.createElement('div');
    sentinel.id = 'scroll-sentinel';
    sentinel.className = 'py-3 text-center text-xs text-gray-400';
    sentinel.textContent = `Showing ${_pageOffset.toLocaleString()} of ${_pageItems.length.toLocaleString()}…`;
    document.getElementById('results-container').appendChild(sentinel);
    _pageObserver = new IntersectionObserver(
      entries => { if (entries[0].isIntersecting) _renderNextPage(); },
      { rootMargin: '200px' }
    );
    _pageObserver.observe(sentinel);
  } else {
    _stopPageObserver();
  }
}

function _startPaging(items) {
  _stopPageObserver();
  _pageItems  = items;
  _pageOffset = 0;
  document.getElementById('results-container').innerHTML = '';
  _renderNextPage();
}
// --------------------------------------------------------------------------

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
    _stopPageObserver();
    container.innerHTML = '<p class="text-center text-gray-400 py-10 text-sm">No results found.</p>';
    return;
  }
  _startPaging(results);

  // Trend chart + summarize panel
  renderTrendChart(currentResults, _trendBucket);
  _updateSrCountLabel();
  if (results.length) {
    const tabbar = document.getElementById('sr-viz-tabbar');
    if (tabbar) tabbar.classList.remove('hidden');
  }
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
    <button id="chart-filter-clear" class="ml-auto text-indigo-600 hover:text-indigo-900 font-semibold">• Clear</button>
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

  // section visibility controlled by tab clicks — do not auto-show
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
            label: item => `${item.raw} messages click to filter`,
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

// -- ANALYSIS TABS (Summarize / Visualization) ----------------------------
(function () {
  const PANELS = ['sr-section', 'trend-section'];

  document.getElementById('sr-viz-tabbar')?.addEventListener('click', e => {
    const btn = e.target.closest('.sr-viz-tab');
    if (!btn) return;
    const targetId = btn.dataset.target;

    // Toggle: clicking the active tab collapses it
    const targetEl = document.getElementById(targetId);
    const isOpen   = !targetEl?.classList.contains('hidden') && btn.classList.contains('sr-viz-tab-active');

    // Hide all panels and deactivate all tabs
    PANELS.forEach(id => document.getElementById(id)?.classList.add('hidden'));
    document.querySelectorAll('.sr-viz-tab').forEach(b => {
      b.classList.remove('sr-viz-tab-active', 'bg-indigo-100', 'text-indigo-700');
    });

    if (!isOpen && targetEl) {
      targetEl.classList.remove('hidden');
      btn.classList.add('sr-viz-tab-active', 'bg-indigo-100', 'text-indigo-700');
      // Resize chart if the visualization panel just became visible
      if (targetId === 'trend-section' && _trendChart) {
        requestAnimationFrame(() => _trendChart.resize());
      }
    }
  });
}());

// -- SEARCH PANEL COLLAPSE -------------------------------------------------
(function () {
  const section   = document.getElementById('search-section');
  const body      = document.getElementById('search-body');
  const btn       = document.getElementById('search-collapse-btn');
  const icon      = document.getElementById('search-collapse-icon');
  if (!section || !body || !btn || !icon) return;

  let _collapsed = false;

  function _setCollapsed(collapsed) {
    _collapsed = collapsed;
    body.classList.toggle('hidden', collapsed);
    icon.style.transform = collapsed ? 'rotate(180deg)' : '';
    btn.title = collapsed ? 'Expand search' : 'Collapse search';
    // tighten vertical padding when collapsed so the sticky bar is compact
    section.style.paddingBottom = collapsed ? '0.5rem' : '';
    section.style.paddingTop    = collapsed ? '0.5rem' : '';
  }

  btn.addEventListener('click', () => _setCollapsed(!_collapsed));

  // Show toggle button only when scrolled away from top; auto-expand at top
  let _rafPending = false;
  window.addEventListener('scroll', () => {
    if (_rafPending) return;
    _rafPending = true;
    requestAnimationFrame(() => {
      _rafPending = false;
      const atTop = window.scrollY < 10;
      btn.classList.toggle('hidden', atTop);
      if (atTop && _collapsed) _setCollapsed(false);
    });
  }, { passive: true });
}());

// -- SUMMARIZE RESULTS -----------------------------------------------------
const LOG_ICONS = {
  filter:      '🔍',
  retrieval:   '📡',
  dedup:       '🧹',
  cluster:     '🔮',
  sample:      '🎯',
  llm:         '✨',
  fallback:    '⚠️',
  meta:        '📅',
  instruction: '📝',
};
function _updateSrCountLabel() {
  const el = document.getElementById('sr-count-label');
  if (el) el.textContent = currentResults.length.toLocaleString();
}

function renderSrLogEntry(entry) {
  const logEl = document.getElementById('sr-process-log');
  const div   = document.createElement('div');
  div.className = `log-entry log-step-${entry.step || 'fallback'}`;
  const icon = LOG_ICONS[entry.step] || '•';
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
    `<span class="fu-log-icon">${LOG_ICONS[entry.step] || '•'}</span>` +
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
    : 'All messages passed directly to the LLM no clustering or deduplication.';
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

let filterMode        = 'exact';  // 'exact' | 'any' | 'fuzzy' | 'semantic'
let _semanticDebounce = null;

/* -- Set active mode + update UI -- */
const _EXACT_ACTIVE    = ['bg-indigo-700','text-white'];
const _EXACT_INACTIVE  = ['bg-slate-100','text-slate-500'];
const _ANY_ACTIVE      = ['bg-emerald-600','text-white'];
const _ANY_INACTIVE    = ['bg-slate-100','text-slate-500'];
const _FUZZY_ACTIVE    = ['bg-amber-500','text-white'];
const _FUZZY_INACTIVE  = ['bg-slate-100','text-slate-500'];
const _SEM_ACTIVE      = ['bg-indigo-700','text-white'];
const _SEM_INACTIVE    = ['bg-slate-100','text-slate-500'];

function setFilterMode(mode) {
  filterMode = mode;
  const exactBtn  = document.getElementById('filter-mode-exact');
  const anyBtn    = document.getElementById('filter-mode-any');
  const fuzzyBtn  = document.getElementById('filter-mode-fuzzy');
  const semBtn    = document.getElementById('filter-mode-semantic');

  exactBtn.classList.remove( ..._EXACT_ACTIVE,   ..._EXACT_INACTIVE);
  anyBtn.classList.remove(   ..._ANY_ACTIVE,     ..._ANY_INACTIVE);
  fuzzyBtn.classList.remove( ..._FUZZY_ACTIVE,   ..._FUZZY_INACTIVE);
  semBtn.classList.remove(   ..._SEM_ACTIVE,     ..._SEM_INACTIVE);

  if (mode === 'exact') {
    exactBtn.classList.add( ..._EXACT_ACTIVE);
    anyBtn.classList.add(   ..._ANY_INACTIVE);
    fuzzyBtn.classList.add( ..._FUZZY_INACTIVE);
    semBtn.classList.add(   ..._SEM_INACTIVE);
  } else if (mode === 'any') {
    exactBtn.classList.add( ..._EXACT_INACTIVE);
    anyBtn.classList.add(   ..._ANY_ACTIVE);
    fuzzyBtn.classList.add( ..._FUZZY_INACTIVE);
    semBtn.classList.add(   ..._SEM_INACTIVE);
  } else if (mode === 'fuzzy') {
    exactBtn.classList.add( ..._EXACT_INACTIVE);
    anyBtn.classList.add(   ..._ANY_INACTIVE);
    fuzzyBtn.classList.add( ..._FUZZY_ACTIVE);
    semBtn.classList.add(   ..._SEM_INACTIVE);
  } else {
    exactBtn.classList.add( ..._EXACT_INACTIVE);
    anyBtn.classList.add(   ..._ANY_INACTIVE);
    fuzzyBtn.classList.add( ..._FUZZY_INACTIVE);
    semBtn.classList.add(   ..._SEM_ACTIVE);
  }

  exactBtn.setAttribute( 'aria-pressed', String(mode === 'exact'));
  anyBtn.setAttribute(   'aria-pressed', String(mode === 'any'));
  fuzzyBtn.setAttribute( 'aria-pressed', String(mode === 'fuzzy'));
  semBtn.setAttribute(   'aria-pressed', String(mode === 'semantic'));

  const placeholders = {
    exact:    'Exact: whole-word match, multi-word phrase scores highest...',
    any:      'Any Word: returns messages containing at least one query word...',
    fuzzy:    'Fuzzy: characters must appear in order anywhere in the text (fzf-style)...',
    semantic: 'Semantic: re-rank results by embedding similarity...',
  };
  document.getElementById('results-filter').placeholder = placeholders[mode];
}

document.getElementById('filter-mode-exact')
  .addEventListener('click', () => { setFilterMode('exact');    applyResultsFilter(); });
document.getElementById('filter-mode-any')
  .addEventListener('click', () => { setFilterMode('any');      applyResultsFilter(); });
document.getElementById('filter-mode-fuzzy')
  .addEventListener('click', () => { setFilterMode('fuzzy');    applyResultsFilter(); });
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
  countLabel.textContent = `${msgs.length} of ${total}`;
  countLabel.classList.remove('hidden');
  if (!msgs.length) {
    _stopPageObserver();
    document.getElementById('results-container').innerHTML =
      '<p class="text-center text-gray-400 py-10 text-sm">No results match the filter.</p>';
    return;
  }
  _startPaging(msgs);
}

function _resetToAllResults() {
  activeFilterTokens = [];
  document.getElementById('results-filter-count').classList.add('hidden');
  _startPaging(currentResults);
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

      // Phrase match highest score
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

  // Substring match (OR logic) partial words work while typing
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
    _startPaging(hits);
  } catch (e) {
    showErrorPopup(`Semantic filter error: ${e.message}`);
    _resetToAllResults();
  } finally {
    spinner.classList.add('hidden');
  }
}

/* -- Fuzzy filter (instant, client-side, fzf-style) -- */
// Returns a score > 0 if every character of `pattern` appears in order in `str`.
// Consecutive run bonus: each run of consecutive character matches adds extra weight.
function _fuzzyScore(pattern, str) {
  if (!pattern) return 1;
  let si = 0, pi = 0, score = 0, run = 0;
  while (si < str.length && pi < pattern.length) {
    if (str[si] === pattern[pi]) {
      run++;
      score += run * 2;   // consecutive chars score more
      pi++;
    } else {
      run = 0;
    }
    si++;
  }
  return pi === pattern.length ? score : 0;
}

function _applyFuzzyFilter(term) {
  if (!term) { _resetToAllResults(); return; }

  // Each space-separated word must fuzzy-match independently (AND logic)
  const words = term.split(/\s+/).filter(Boolean);
  activeFilterTokens = words;  // used for highlight (best-effort)

  const scored = currentResults
    .map(m => {
      const text = ((m.username || '') + ' ' + (m.content || '')).toLowerCase();
      let total = 0;
      for (const w of words) {
        const s = _fuzzyScore(w, text);
        if (!s) return { m, s: 0 };
        total += s;
      }
      return { m, s: total };
    })
    .filter(x => x.s > 0)
    .sort((a, b) => b.s - a.s);   // highest fuzzy score first

  _renderFilteredCards(scored.map(x => x.m), currentResults.length);
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
  } else if (filterMode === 'fuzzy') {
    _applyFuzzyFilter(term);
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
          Context ${msgs.length} messages (${before} before &bull; ${after} after)
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
  const dir  = document.getElementById('bm-sort-dir').dataset.dir; // 'asc' | 'desc'
  const asc  = dir === 'asc';
  const sorted = [...bms];
  if (mode === 'date') {
    sorted.sort((a, b) => (new Date(a.date) - new Date(b.date)) * (asc ? 1 : -1));
  } else if (mode === 'username') {
    sorted.sort((a, b) => (a.username || '').localeCompare(b.username || '') * (asc ? 1 : -1));
  } else {
    sorted.sort((a, b) => (a.bookmark_id - b.bookmark_id) * (asc ? 1 : -1));
  }
  return sorted;
}

let _cachedBookmarks  = [];
let _bmSelectionState      = null;  // {bmId, text} while a highlight code picker is open
let _bmSelPopover          = null;  // floating DOM element for the selection popover
let _bmAccumulatedSegments = null;  // {bmId, segments: string[]} for Ctrl+multi-select
let _bmPendingHlState      = null;  // {bmId, texts: string[]} active pending highlight in excerpt

const _BM_SEG_SEP = ' ... '; // separator used to join/split multi-segment highlight texts

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

const _CODE_PALETTE = [
  '#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
  '#06b6d4', '#f97316', '#ec4899', '#14b8a6', '#84cc16',
  '#3b82f6', '#a855f7', '#22c55e', '#e11d48', '#0ea5e9',
  '#d97706', '#7c3aed', '#059669', '#dc2626', '#0891b2',
];
function _randomCodeColor() {
  return _CODE_PALETTE[Math.floor(Math.random() * _CODE_PALETTE.length)];
}

// ── Color swatch picker helpers ───────────────────────────────────────────────
// Builds the HTML for a swatch picker that replaces <input type="color">.
// inputId      — the id placed on the hidden <input>; may be empty for inline pickers
// initialColor — hex string
// inputClass   — extra class(es) added to the hidden <input> (e.g. 'bm-new-code-color')
// inputData    — extra HTML attributes for the hidden <input> (e.g. 'data-bm-id="5"')
function _colorPickerHtml(inputId, initialColor, inputClass = '', inputData = '') {
  const uid   = inputId || ('cpk' + Math.random().toString(36).slice(2, 7));
  const color = initialColor || _CODE_PALETTE[0];
  const dots  = _CODE_PALETTE.map(c => {
    const sel = c.toLowerCase() === color.toLowerCase() ? ' active' : '';
    return `<button type="button" class="color-swatch-dot${sel}" style="background:${c}" data-color="${c}"></button>`;
  }).join('');
  return `<div class="color-picker-wrap" data-cpk="${uid}">`
    + `<button type="button" class="color-swatch-btn" style="background:${color}" data-cpk-toggle="${uid}" title="Pick colour"></button>`
    + `<input type="hidden"${inputId ? ` id="${inputId}"` : ''} class="cpk-value${inputClass ? ' ' + inputClass : ''}" value="${color}"${inputData ? ' ' + inputData : ''} />`
    + `<div class="color-swatch-palette hidden" data-cpk-palette="${uid}">${dots}</div>`
    + `</div>`;
}

// Update a named swatch picker (and its button + dot selection) to a new color.
function _setPickerColor(inputId, color) {
  const inp = document.getElementById(inputId);
  if (!inp) return;
  inp.value = color;
  const wrap = inp.closest('.color-picker-wrap');
  if (!wrap) return;
  wrap.querySelector('.color-swatch-btn').style.background = color;
  wrap.querySelectorAll('.color-swatch-dot').forEach(d =>
    d.classList.toggle('active', d.dataset.color.toLowerCase() === color.toLowerCase()));
}

// Convert every remaining <input type="color"> in the static HTML into a swatch picker.
function _initStaticColorPickers() {
  document.querySelectorAll('input[type="color"]').forEach(input => {
    const id    = input.id;
    const color = input.value || _randomCodeColor();
    const tmp   = document.createElement('div');
    tmp.innerHTML = _colorPickerHtml(id, color);
    // Carry over any lg-size class markers from the original input
    if (input.classList.contains('lg')) tmp.firstElementChild.querySelector('.color-swatch-btn')?.classList.add('lg');
    input.replaceWith(tmp.firstElementChild);
  });
}

// Global delegation for all swatch picker interactions
document.addEventListener('click', e => {
  // Toggle dropdown
  const toggleBtn = e.target.closest('[data-cpk-toggle]');
  if (toggleBtn) {
    const uid = toggleBtn.dataset.cpkToggle;
    document.querySelectorAll(`.color-swatch-palette:not([data-cpk-palette="${uid}"])`).forEach(p => p.classList.add('hidden'));
    document.querySelector(`[data-cpk-palette="${uid}"]`)?.classList.toggle('hidden');
    return;
  }
  // Select a color
  const dot = e.target.closest('.color-swatch-dot');
  if (dot) {
    const pal = dot.closest('[data-cpk-palette]');
    if (!pal) return;
    const wrap  = pal.closest('.color-picker-wrap');
    const color = dot.dataset.color;
    const inp   = wrap?.querySelector('.cpk-value');
    const btn   = wrap?.querySelector('.color-swatch-btn');
    if (inp) inp.value = color;
    if (btn) btn.style.background = color;
    pal.querySelectorAll('.color-swatch-dot').forEach(d =>
      d.classList.toggle('active', d.dataset.color === color));
    pal.classList.add('hidden');
    return;
  }
  // Click outside → close all palettes
  if (!e.target.closest('.color-picker-wrap')) {
    document.querySelectorAll('.color-swatch-palette').forEach(p => p.classList.add('hidden'));
  }
});

// -- Load & render code filter chips -----------------------------------------
async function loadAllCodes() {
  try {
    _allCodes = await apiFetch('/api/codes');
  } catch (_) { _allCodes = []; }
  renderBmCodeFilterChips();
}

function renderBmCodeFilterChips() {
  const container = document.getElementById('bm-label-filter-chips');
  if (!container) return;
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
                  title="Remove code">${esc(l.name)} —</span>`;
  }).join('');
  return chips + `<button class="bm-code-btn text-xs text-gray-400 hover:text-indigo-600 border border-dashed border-gray-300 hover:border-indigo-400 rounded-full px-2 py-0.5 transition-colors"
                          data-bm-id="${bm.bookmark_id}">+ code</button>`;
}

// -- Render the inline code picker panel -------------------------------------
function _renderBmCodePanel(bookmarkId) {
  const panel = document.getElementById(`bm-code-panel-${bookmarkId}`);
  if (!panel) return;
  panel.innerHTML = `
    <div class=”p-2 border border-dashed border-indigo-200 rounded-xl bg-indigo-50/30”>
      <div class=”flex items-start justify-between gap-2 mb-2”>
        <p class=”text-xs font-medium text-indigo-700”>Coding whole quote</p>
        <button class=”bm-code-panel-close text-gray-400 hover:text-gray-600 leading-none shrink-0 text-sm” data-bm-id=”${bookmarkId}”>✕</button>
      </div>
      <div class=”flex items-center gap-1.5”>
        <div class=”relative flex-1 min-w-0”>
          <input class=”bm-new-code-input w-full border border-gray-200 rounded-lg px-2 py-1 text-xs bg-white focus:outline-none focus:ring-2 focus:ring-indigo-300”
                 placeholder=”Type to search or create a code…” data-bm-id=”${bookmarkId}” autocomplete=”off” />
          <div class=”bm-code-suggestions hidden absolute left-0 top-full mt-0.5 z-50 bg-white border border-gray-200 rounded-xl shadow-lg py-1 w-full max-h-44 overflow-y-auto”
               data-bm-id=”${bookmarkId}” data-type=”whole”></div>
        </div>
        ${_colorPickerHtml('', _randomCodeColor(), 'bm-new-code-color', `data-bm-id=”${bookmarkId}”`)}
        <button class=”bm-new-code-create text-xs px-2.5 py-1 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 shrink-0”
                data-bm-id=”${bookmarkId}”>Add</button>
      </div>
    </div>`;
  requestAnimationFrame(() => panel.querySelector('.bm-new-code-input')?.focus());
}

// ── Annotate excerpt text with colored highlight spans ────────────────────────
// Handles overlapping spans: splits content at every span boundary, then renders
// each segment with all codes that cover it (stacked bottom-border colors).
// pendingTexts: text strings actively being coded — shown with dashed indigo underline.
function _annotateExcerpt(content, highlights, pendingTexts = []) {
  const lower = content.toLowerCase();

  const rawSpans = [];
  for (const h of (highlights || [])) {
    const needle = (h.highlighted_text || '').toLowerCase();
    if (!needle) continue;
    const idx = lower.indexOf(needle);
    if (idx === -1) continue;
    rawSpans.push({ start: idx, end: idx + needle.length,
                    color: h.code_color || '#6366f1', name: h.code_name || '?', pending: false });
  }
  for (const pt of pendingTexts) {
    const needle = (pt || '').toLowerCase();
    if (!needle) continue;
    const idx = lower.indexOf(needle);
    if (idx === -1) continue;
    rawSpans.push({ start: idx, end: idx + needle.length,
                    color: '#6366f1', name: 'Coding…', pending: true });
  }
  if (!rawSpans.length) return esc(content);

  const pts = new Set([0, content.length]);
  rawSpans.forEach(s => { pts.add(s.start); pts.add(s.end); });
  const sorted = [...pts].sort((a, b) => a - b);

  let out = '';
  for (let i = 0; i < sorted.length - 1; i++) {
    const segStart = sorted[i];
    const segEnd   = sorted[i + 1];
    if (segStart === segEnd) continue;
    const text    = content.slice(segStart, segEnd);
    const active  = rawSpans.filter(s => s.start <= segStart && s.end >= segEnd);
    if (!active.length) { out += esc(text); continue; }

    const coded   = active.filter(s => !s.pending);
    const pending = active.filter(s =>  s.pending);

    if (!coded.length) {
      // Only a pending span — dashed indigo underline + tinted background
      out += `<mark class="bm-hl-inline bm-hl-pending rounded-sm cursor-default" style="background:#e0e7ff;border-bottom:2px dashed #6366f1" title="Coding…">${esc(text)}</mark>`;
    } else {
      const [first, ...rest] = coded;
      let style = `background:${first.color}28;border-bottom:2px solid ${first.color}`;
      if (rest.length) {
        const shadows = rest.map((s, k) => `0 ${(k + 2) * 2 + 1}px 0 ${s.color}`).join(',');
        style += `;box-shadow:${shadows};padding-bottom:${rest.length * 3}px`;
      }
      // pending coexists — no extra decoration, the coded highlight is sufficient
      const title = [...coded.map(s => s.name), ...(pending.length ? ['(Coding…)'] : [])].join(' + ');
      out += `<mark class="bm-hl-inline rounded-sm cursor-default" style="${style}" title="${esc(title)}">${esc(text)}</mark>`;
    }
  }
  return out;
}

// ── Coded-span chips shown below each bookmark card ───────────────────────────
function _bmHlChipsHtml(bm) {
  const hls = bm.highlights || [];
  if (!hls.length) return '';

  // Group by code_id so segments of the same code appear as one chip
  const byCode = {};
  for (const h of hls) {
    if (!byCode[h.code_id]) byCode[h.code_id] = { meta: h, texts: [], ids: [] };
    byCode[h.code_id].texts.push(h.highlighted_text || '');
    byCode[h.code_id].ids.push(h.id);
  }

  const chips = Object.values(byCode).map(({ meta, texts, ids }) => {
    const tc  = labelTextColor(meta.code_color || '#6366f1');
    const txt = texts.join(_BM_SEG_SEP); // no character limit
    return `<div class="flex items-start gap-1.5">
      <span class="w-0.5 self-stretch rounded-full shrink-0 mt-0.5" style="background:${meta.code_color || '#6366f1'}"></span>
      <span class="text-xs italic text-gray-600 flex-1 break-words min-w-0">"${esc(txt)}"</span>
      <span class="text-xs px-1.5 py-0.5 rounded-full font-medium shrink-0 ml-1"
            style="background:${meta.code_color || '#6366f1'};color:${tc}">${esc(meta.code_name || '?')}</span>
      <button class="bm-hl-remove text-gray-300 hover:text-red-500 text-base leading-none shrink-0 transition-colors"
              data-bm-id="${bm.bookmark_id}" data-hl-ids="${ids.join(',')}" title="Remove coding">×</button>
    </div>`;
  });

  return `<div class="mt-2 pt-2 border-t border-dashed border-gray-100 space-y-1">
    <p class="text-[10px] font-semibold text-gray-400 uppercase tracking-wide mb-1">Coded spans</p>
    ${chips.join('')}
  </div>`;
}

// ── Highlight code picker panel (opened when text is selected) ────────────────
function _renderBmHlPanel(bmId, selectedText) {
  const panel = document.getElementById(`bm-hl-panel-${bmId}`);
  const bm    = _cachedBookmarks.find(b => b.bookmark_id === bmId);
  if (!panel || !bm) return;
  panel.dataset.hlText = selectedText;
  _bmSetPendingHl(bmId, selectedText);

  const truncated = selectedText.length > 80 ? selectedText.slice(0, 80) + '…' : selectedText;

  panel.innerHTML = `
    <div class="p-2 border border-dashed border-indigo-200 rounded-xl bg-indigo-50/30">
      <div class="flex items-start justify-between gap-2 mb-2">
        <p class="text-xs font-medium text-indigo-700">Coding: <em class="text-indigo-500 not-italic">"${esc(truncated)}"</em></p>
        <button class="bm-hl-panel-close text-gray-400 hover:text-gray-600 leading-none shrink-0 text-sm" data-bm-id="${bmId}">✕</button>
      </div>
      <div class="flex items-center gap-1.5">
        <div class="relative flex-1 min-w-0">
          <input class="bm-hl-new-code-input w-full border border-gray-200 rounded-lg px-2 py-1 text-xs bg-white focus:outline-none focus:ring-2 focus:ring-indigo-300"
                 placeholder="Type to search or create a code…" data-bm-id="${bmId}" autocomplete="off" />
          <div class="bm-code-suggestions hidden absolute left-0 top-full mt-0.5 z-50 bg-white border border-gray-200 rounded-xl shadow-lg py-1 w-full max-h-44 overflow-y-auto"
               data-bm-id="${bmId}" data-type="span"></div>
        </div>
        ${_colorPickerHtml('', _randomCodeColor(), 'bm-hl-new-code-color', `data-bm-id="${bmId}"`)}
        <button class="bm-hl-new-code-create text-xs px-2.5 py-1 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 shrink-0"
                data-bm-id="${bmId}">Add</button>
      </div>
    </div>`;
  requestAnimationFrame(() => panel.querySelector('.bm-hl-new-code-input')?.focus());
}

// -- filter bookmarks ---------------------------------------------------------
function _filterBookmarks(bms) {
  const userTerm  = (document.getElementById('bm-filter-user').value || '').trim().toLowerCase();
  const sunoMode  = document.getElementById('bm-filter-suno').value;
  const codedMode = document.getElementById('bm-filter-coded')?.value || 'all';
  const textRaw   = (document.getElementById('bm-filter-text').value || '').trim();
  const monthFrom = (document.getElementById('bm-month-from').value || '').trim();
  const monthTo   = (document.getElementById('bm-month-to').value || '').trim();
  const textTerms = textRaw ? textRaw.split(/\s+/).filter(Boolean).map(w => w.toLowerCase()) : [];

  return bms.filter(bm => {
    if (userTerm && !(bm.username || '').toLowerCase().includes(userTerm)) return false;
    if (sunoMode === 'only'    && !truthy(bm.is_suno_team)) return false;
    if (sunoMode === 'exclude' &&  truthy(bm.is_suno_team)) return false;
    if (monthFrom || monthTo) {
      const bmDate = (bm.date || '').slice(0, 10);
      if (monthFrom && bmDate < monthFrom) return false;
      if (monthTo   && bmDate > monthTo)   return false;
    }
    if (textTerms.length > 0) {
      const hay = ((bm.username || '') + ' ' + (bm.content || '') + ' ' + (bm.note || '')).toLowerCase();
      if (!textTerms.every(t => hay.includes(t))) return false;
    }
    if (codedMode !== 'all') {
      const isCoded = (bm.codes?.length > 0) || (bm.highlights?.length > 0);
      if (codedMode === 'coded'   && !isCoded) return false;
      if (codedMode === 'uncoded' &&  isCoded) return false;
    }
    return true;
  });
}

function _renderBookmarksSorted() {
  const container  = document.getElementById('bookmarks-container');
  const countBadge = document.getElementById('bm-result-count');

  const codedBadge     = document.getElementById('bm-coded-count');
  const closeAllBtn    = document.getElementById('bm-close-all-panels');

  if (!_cachedBookmarks.length) {
    container.innerHTML = '<p class="text-sm text-gray-400 text-center py-8">No bookmarks yet. Use the bookmark button on any search result.</p>';
    if (countBadge)  countBadge.classList.add('hidden');
    if (codedBadge)  codedBadge.classList.add('hidden');
    if (closeAllBtn) closeAllBtn.classList.add('hidden');
    return;
  }

  const filtered = _filterBookmarks(_sortBookmarks(_cachedBookmarks));
  const total    = _cachedBookmarks.length;
  const shown    = filtered.length;

  if (countBadge) {
    countBadge.classList.remove('hidden');
    countBadge.textContent = shown === total ? `${total}` : `${shown} / ${total}`;
  }

  if (codedBadge) {
    const codedCount = filtered.filter(bm =>
      (bm.codes?.length > 0) || (bm.highlights?.length > 0)
    ).length;
    codedBadge.classList.toggle('hidden', shown === 0);
    codedBadge.textContent = `coded ${codedCount} / ${shown}`;
  }

  if (closeAllBtn) closeAllBtn.classList.remove('hidden');

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
        <p class="text-sm leading-relaxed text-gray-800 whitespace-pre-wrap break-words bm-excerpt-text"
           data-bm-id="${bm.bookmark_id}">${_annotateExcerpt(bm.content, bm.highlights)}</p>
        ${hasContent(bm.attachments) ? `<p class="text-xs text-gray-500 mt-1">📎 ${esc(bm.attachments)}</p>` : ''}
        ${hasContent(bm.reactions)   ? `<p class="text-xs text-gray-500 mt-1">💬 ${esc(bm.reactions)}</p>`   : ''}
        <!-- Codes -->
        <div class="flex items-center flex-wrap gap-1 mt-2 min-h-[1.5rem]" id="bm-codes-${bm.bookmark_id}">
          ${_bmCodeChipsHtml(bm)}
        </div>
        <!-- Inline code picker (hidden until opened) -->
        <div id="bm-code-panel-${bm.bookmark_id}" class="hidden mt-1 border border-dashed border-gray-200 rounded-xl bg-gray-50"></div>
        <!-- Coded spans list -->
        <div id="bm-hl-chips-${bm.bookmark_id}">${_bmHlChipsHtml(bm)}</div>
        <!-- Highlight code picker (hidden until text is selected) -->
        <div id="bm-hl-panel-${bm.bookmark_id}" class="hidden mt-1"></div>
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
  // Apply a suggested existing code from the autocomplete dropdown
  const sugItem = e.target.closest('.bm-code-suggest-item');
  if (sugItem) {
    const bmId   = parseInt(sugItem.dataset.bmId);
    const codeId = parseInt(sugItem.dataset.codeId);
    const type   = sugItem.dataset.type; // 'whole' | 'span'
    const bm     = _cachedBookmarks.find(b => b.bookmark_id === bmId);
    const code   = _allCodes.find(c => c.id === codeId);
    if (bm && code) {
      if (type === 'whole') {
        if (!(bm.codes || []).some(c => c.id === codeId)) {
          await fetch(`/api/bookmarks/${bmId}/codes/${codeId}`, { method: 'POST' });
          bm.codes = [...(bm.codes || []),
            { id: code.id, name: code.name, color: code.color, description: code.description, category_id: code.category_id }];
          const labelsRow = document.getElementById(`bm-codes-${bmId}`);
          if (labelsRow) labelsRow.innerHTML = _bmCodeChipsHtml(bm);
        }
      } else {
        const panel   = document.getElementById(`bm-hl-panel-${bmId}`);
        const selText = panel?.dataset.hlText || '';
        const segs    = selText.split(_BM_SEG_SEP).map(s => s.trim()).filter(Boolean);
        for (const seg of segs) {
          const res  = await fetch(`/api/bookmarks/${bmId}/highlights`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code_id: codeId, highlighted_text: seg }),
          });
          const data = await res.json();
          if (data.id && !(bm.highlights || []).some(h => h.id === data.id)) {
            bm.highlights = [...(bm.highlights || []),
              { id: data.id, code_id: codeId, code_name: code.name, code_color: code.color, highlighted_text: seg }];
          }
        }
        const hlChips = document.getElementById(`bm-hl-chips-${bmId}`);
        if (hlChips) hlChips.innerHTML = _bmHlChipsHtml(bm);
        if (panel && selText) _renderBmHlPanel(bmId, selText);
      }
    }
    // Clear input + close dropdown
    const rel = sugItem.closest('.relative');
    if (rel) {
      const inp = rel.querySelector('input');
      if (inp) inp.value = '';
      rel.querySelector('.bm-code-suggestions')?.classList.add('hidden');
    }
    return;
  }

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

  // Remove code chip (— click on assigned label)
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
    const color = colorInput?.value || '#0d3e7f';
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
        // Label may already exist find it in _allCodes
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

  // Close whole-quote code picker
  const codeClose = e.target.closest('.bm-code-panel-close');
  if (codeClose) {
    const bmId = parseInt(codeClose.dataset.bmId);
    document.getElementById(`bm-code-panel-${bmId}`)?.classList.add('hidden');
    return;
  }

  // Close highlight code picker
  const hlClose = e.target.closest('.bm-hl-panel-close');
  if (hlClose) {
    const bmId = parseInt(hlClose.dataset.bmId);
    document.getElementById(`bm-hl-panel-${bmId}`)?.classList.add('hidden');
    _bmClearPendingHl(bmId);
    _bmResetAccumulatedSel();
    return;
  }

  // Toggle code assignment in highlight code picker
  const hlToggle = e.target.closest('.bm-hl-code-toggle');
  if (hlToggle) {
    const bmId   = parseInt(hlToggle.dataset.bmId);
    const codeId = parseInt(hlToggle.dataset.codeId);
    const bm     = _cachedBookmarks.find(b => b.bookmark_id === bmId);
    const panel  = document.getElementById(`bm-hl-panel-${bmId}`);
    if (!bm || !panel) return;
    const selText  = panel.dataset.hlText || '';
    // Split combined text into individual segments
    const segs = selText.split(_BM_SEG_SEP).map(s => s.trim()).filter(Boolean);
    // Find existing highlights for this code on any of the segments
    const existingForCode = (bm.highlights || []).filter(
      h => h.code_id === codeId && segs.some(s => s.toLowerCase() === h.highlighted_text.toLowerCase())
    );
    if (existingForCode.length) {
      // Remove all segments for this code
      await Promise.all(existingForCode.map(h =>
        fetch(`/api/bookmarks/${bmId}/highlights/${h.id}`, { method: 'DELETE' })
      ));
      const removedIds = new Set(existingForCode.map(h => h.id));
      bm.highlights = (bm.highlights || []).filter(h => !removedIds.has(h.id));
    } else {
      // Add highlight for each segment
      const code = _allCodes.find(c => c.id === codeId);
      for (const seg of segs) {
        const res  = await fetch(`/api/bookmarks/${bmId}/highlights`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code_id: codeId, highlighted_text: seg }),
        });
        const data = await res.json();
        if (code && data.id && !(bm.highlights || []).some(h => h.id === data.id)) {
          bm.highlights = [...(bm.highlights || []), {
            id: data.id, code_id: codeId, code_name: code.name,
            code_color: code.color, highlighted_text: seg,
          }];
        }
      }
    }
    const hlChips   = document.getElementById(`bm-hl-chips-${bmId}`);
    const excerptEl = document.querySelector(`.bm-excerpt-text[data-bm-id="${bmId}"]`);
    if (hlChips)   hlChips.innerHTML   = _bmHlChipsHtml(bm);
    _renderBmHlPanel(bmId, selText); // re-renders excerpt with pending highlight + re-renders panel
    return;
  }

  // Create new code and assign as highlight
  const hlCreate = e.target.closest('.bm-hl-new-code-create');
  if (hlCreate) {
    const bmId  = parseInt(hlCreate.dataset.bmId);
    const panel = document.getElementById(`bm-hl-panel-${bmId}`);
    const bm    = _cachedBookmarks.find(b => b.bookmark_id === bmId);
    if (!panel || !bm) return;
    const selText    = panel.dataset.hlText || '';
    const nameInput  = panel.querySelector('.bm-hl-new-code-input');
    const colorInput = panel.querySelector('.bm-hl-new-code-color');
    const name  = (nameInput?.value || '').trim();
    const color = colorInput?.value || '#0d3e7f';
    if (!name) { nameInput?.focus(); return; }
    hlCreate.disabled = true; hlCreate.textContent = '...';
    try {
      const labelRes = await fetch('/api/codes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, color }),
      });
      const newCode = await labelRes.json();
      if (!labelRes.ok) {
        const existing = _allCodes.find(l => l.name.toLowerCase() === name.toLowerCase());
        if (!existing) { hlCreate.disabled = false; hlCreate.textContent = 'Add'; return; }
        newCode.id = existing.id; newCode.name = existing.name; newCode.color = existing.color;
      } else {
        _allCodes = [..._allCodes, newCode].sort((a, b) =>
          a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }));
        renderBmCodeFilterChips();
      }
      // Create one highlight entry per segment
      const segs = selText.split(_BM_SEG_SEP).map(s => s.trim()).filter(Boolean);
      for (const seg of segs) {
        const hlRes  = await fetch(`/api/bookmarks/${bmId}/highlights`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code_id: newCode.id, highlighted_text: seg }),
        });
        const hlData = await hlRes.json();
        if (hlData.id && !(bm.highlights || []).some(h => h.id === hlData.id)) {
          bm.highlights = [...(bm.highlights || []), {
            id: hlData.id, code_id: newCode.id, code_name: newCode.name,
            code_color: newCode.color, highlighted_text: seg,
          }];
        }
      }
      const hlChips = document.getElementById(`bm-hl-chips-${bmId}`);
      if (hlChips) hlChips.innerHTML = _bmHlChipsHtml(bm);
      _renderBmHlPanel(bmId, selText); // re-renders excerpt with pending highlight + re-renders panel
      document.dispatchEvent(new CustomEvent('codebook-updated'));
    } catch (_) {
      hlCreate.disabled = false; hlCreate.textContent = 'Add';
    }
    return;
  }

  // Remove a coded span (highlight) — data-hl-ids may be comma-separated for merged chips
  const hlRemove = e.target.closest('.bm-hl-remove');
  if (hlRemove) {
    const bmId  = parseInt(hlRemove.dataset.bmId);
    const hlIds = (hlRemove.dataset.hlIds || hlRemove.dataset.hlId || '')
      .split(',').map(s => parseInt(s.trim())).filter(Boolean);
    const bm    = _cachedBookmarks.find(b => b.bookmark_id === bmId);
    await Promise.all(hlIds.map(id =>
      fetch(`/api/bookmarks/${bmId}/highlights/${id}`, { method: 'DELETE' })
    ));
    if (bm) {
      const removed = new Set(hlIds);
      bm.highlights = (bm.highlights || []).filter(h => !removed.has(h.id));
    }
    const hlChips   = document.getElementById(`bm-hl-chips-${bmId}`);
    const excerptEl = document.querySelector(`.bm-excerpt-text[data-bm-id="${bmId}"]`);
    if (hlChips   && bm) hlChips.innerHTML   = _bmHlChipsHtml(bm);
    const pendingTexts = (_bmPendingHlState?.bmId === bmId) ? _bmPendingHlState.texts : [];
    if (excerptEl && bm) excerptEl.innerHTML = _annotateExcerpt(bm.content, bm.highlights, pendingTexts);
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
          Context ${msgs.length} messages (${before} before &bull; ${after} after)
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

document.getElementById('bm-sort-dir').addEventListener('click', () => {
  const btn = document.getElementById('bm-sort-dir');
  const next = btn.dataset.dir === 'asc' ? 'desc' : 'asc';
  btn.dataset.dir  = next;
  btn.textContent  = next === 'asc' ? '↑ Asc' : '↓ Desc';
  _collapseAllBmContext();
  _renderBookmarksSorted();
});
document.getElementById('bm-filter-suno').addEventListener('change', () => {
  _collapseAllBmContext();
  _renderBookmarksSorted();
});
document.getElementById('bm-filter-coded').addEventListener('change', () => {
  _collapseAllBmContext();
  _renderBookmarksSorted();
});

document.getElementById('bm-close-all-panels').addEventListener('click', () => {
  document.querySelectorAll('[id^="bm-code-panel-"]').forEach(p => p.classList.add('hidden'));
  document.querySelectorAll('[id^="bm-hl-panel-"]').forEach(p => p.classList.add('hidden'));
});

// Keyboard nav for code-name inputs and their suggestion dropdowns
document.getElementById('bookmarks-container').addEventListener('keydown', e => {
  const input   = e.target.closest('.bm-new-code-input, .bm-hl-new-code-input');
  const sugItem = e.target.closest('.bm-code-suggest-item');

  if (e.key === 'ArrowDown') {
    e.preventDefault();
    if (input) {
      input.closest('.relative')?.querySelector('.bm-code-suggest-item')?.focus();
    } else if (sugItem) {
      (sugItem.nextElementSibling)?.focus();
    }
    return;
  }
  if (e.key === 'ArrowUp') {
    e.preventDefault();
    if (sugItem) {
      const prev = sugItem.previousElementSibling;
      if (prev?.classList.contains('bm-code-suggest-item')) prev.focus();
      else sugItem.closest('.relative')?.querySelector('input')?.focus();
    }
    return;
  }
  if (e.key === 'Escape') {
    const rel = (input || sugItem)?.closest('.relative');
    rel?.querySelector('.bm-code-suggestions')?.classList.add('hidden');
    rel?.querySelector('input')?.focus();
    return;
  }
  if (e.key !== 'Enter') return;

  if (sugItem) {
    e.preventDefault();
    sugItem.click();
    return;
  }
  if (input) {
    e.preventDefault();
    const cls = input.classList.contains('bm-new-code-input') ? 'bm-new-code-create' : 'bm-hl-new-code-create';
    document.querySelector(`.${cls}[data-bm-id="${input.dataset.bmId}"]`)?.click();
  }
});

// Live suggestions as user types in code-name input
document.getElementById('bookmarks-container').addEventListener('input', e => {
  const input = e.target.closest('.bm-new-code-input, .bm-hl-new-code-input');
  if (!input) return;
  const term   = input.value.trim().toLowerCase();
  const rel    = input.closest('.relative');
  const sugBox = rel?.querySelector('.bm-code-suggestions');
  if (!sugBox) return;
  if (!term) { sugBox.classList.add('hidden'); sugBox.innerHTML = ''; return; }
  const matches = _allCodes.filter(c => c.name.toLowerCase().includes(term)).slice(0, 8);
  if (!matches.length) { sugBox.classList.add('hidden'); return; }
  const bmId = sugBox.dataset.bmId;
  const type = sugBox.dataset.type;
  sugBox.innerHTML = matches.map(c =>
    `<button type="button"
             class="bm-code-suggest-item w-full text-left px-3 py-1.5 text-xs hover:bg-indigo-50 flex items-center gap-2 transition-colors"
             data-bm-id="${bmId}" data-code-id="${c.id}" data-type="${type}">
       <span class="w-2.5 h-2.5 rounded-full shrink-0" style="background:${c.color}"></span>
       <span class="truncate">${esc(c.name)}</span>
     </button>`
  ).join('');
  sugBox.classList.remove('hidden');
});

// Close suggestions when focus leaves the input+dropdown area
document.getElementById('bookmarks-container').addEventListener('focusout', e => {
  if (!e.target.closest('.bm-new-code-input, .bm-hl-new-code-input, .bm-code-suggest-item')) return;
  const rel = e.target.closest('.relative');
  setTimeout(() => {
    if (rel && !rel.contains(document.activeElement)) {
      rel.querySelector('.bm-code-suggestions')?.classList.add('hidden');
    }
  }, 120);
});

// Period (month) filter
['bm-month-from', 'bm-month-to'].forEach(id => {
  document.getElementById(id).addEventListener('change', () => {
    _collapseAllBmContext();
    _renderBookmarksSorted();
  });
});
document.getElementById('bm-month-clear').addEventListener('click', () => {
  document.getElementById('bm-month-from').value = '';
  document.getElementById('bm-month-to').value   = '';
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

// ── Text selection → open coding popover ─────────────────────────────────────

function _bmCombinedText() {
  return (_bmAccumulatedSegments?.segments || []).join(_BM_SEG_SEP);
}

function _updateBmSelPopoverBtn() {
  const lbl = document.querySelector('#bm-sel-code-btn .bm-sel-label');
  if (!lbl) return;
  const n = _bmAccumulatedSegments?.segments.length || 1;
  lbl.textContent = n > 1 ? `Add open coding (${n} spans)` : 'Add open coding';
}

function _bmResetAccumulatedSel() {
  // Clear the pending highlight only if the coding panel isn't currently open
  if (_bmPendingHlState) {
    const panel = document.getElementById(`bm-hl-panel-${_bmPendingHlState.bmId}`);
    if (!panel || panel.classList.contains('hidden')) {
      _bmClearPendingHl(_bmPendingHlState.bmId);
    }
  }
  _bmAccumulatedSegments = null;
  _bmSelectionState      = null;
  const lbl = document.querySelector('#bm-sel-code-btn .bm-sel-label');
  if (lbl) lbl.textContent = 'Add open coding';
}

// Set or update the active-coding pending highlight for a bookmark excerpt.
function _bmSetPendingHl(bmId, combinedText) {
  // Clear any previous pending highlight on a different bookmark
  if (_bmPendingHlState && _bmPendingHlState.bmId !== bmId) {
    const prevBm = _cachedBookmarks.find(b => b.bookmark_id === _bmPendingHlState.bmId);
    const prevEl = document.querySelector(`.bm-excerpt-text[data-bm-id="${_bmPendingHlState.bmId}"]`);
    if (prevBm && prevEl) prevEl.innerHTML = _annotateExcerpt(prevBm.content, prevBm.highlights);
  }
  const texts = (combinedText || '').split(_BM_SEG_SEP).map(s => s.trim()).filter(Boolean);
  _bmPendingHlState = texts.length ? { bmId, texts } : null;
  const bm = _cachedBookmarks.find(b => b.bookmark_id === bmId);
  const el = document.querySelector(`.bm-excerpt-text[data-bm-id="${bmId}"]`);
  if (bm && el) el.innerHTML = _annotateExcerpt(bm.content, bm.highlights, texts);
}

// Clear the pending highlight, re-rendering the excerpt without it.
function _bmClearPendingHl(bmId) {
  _bmPendingHlState = null;
  const bm = _cachedBookmarks.find(b => b.bookmark_id === bmId);
  const el = document.querySelector(`.bm-excerpt-text[data-bm-id="${bmId}"]`);
  if (bm && el) el.innerHTML = _annotateExcerpt(bm.content, bm.highlights);
}

function _ensureBmSelPopover() {
  if (_bmSelPopover) return _bmSelPopover;
  _bmSelPopover = document.createElement('div');
  _bmSelPopover.id        = 'bm-sel-popover';
  _bmSelPopover.className = 'fixed z-50 bg-white border border-indigo-200 rounded-xl shadow-lg py-1 select-none hidden';
  _bmSelPopover.innerHTML = `
    <button id="bm-sel-code-btn"
            class="flex items-center gap-1.5 w-full px-3 py-1.5 text-xs font-medium text-indigo-700 hover:bg-indigo-50 transition-colors whitespace-nowrap">
      <svg class="w-3 h-3 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
              d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A2 2 0 013 12V7a2 2 0 012-2z"/>
      </svg>
      <span class="bm-sel-label">Add open coding</span>
    </button>
    <p id="bm-sel-hint" class="hidden px-3 pb-1.5 text-[10px] text-indigo-400">Ctrl+select to add more spans</p>`;
  document.body.appendChild(_bmSelPopover);
  document.getElementById('bm-sel-code-btn').addEventListener('click', () => {
    if (!_bmSelectionState) return;
    _bmSelPopover.classList.add('hidden');
    const { bmId, text } = _bmSelectionState;
    const hlPanel = document.getElementById(`bm-hl-panel-${bmId}`);
    if (hlPanel) {
      _renderBmHlPanel(bmId, text);
      hlPanel.classList.remove('hidden');
      hlPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  });
  return _bmSelPopover;
}

// Show the popover anchored near (clientX, clientY)
function _showBmSelPopover(clientX, clientY, bmId, text) {
  _bmSelectionState = { bmId, text };
  const pop  = _ensureBmSelPopover();
  const hint = document.getElementById('bm-sel-hint');
  const n    = _bmAccumulatedSegments?.segments.length || 1;
  if (hint) hint.classList.toggle('hidden', n > 1); // hint only on first selection
  pop.style.left = Math.min(clientX + 4, window.innerWidth  - 200) + 'px';
  pop.style.top  = Math.min(clientY + 4, window.innerHeight -  60) + 'px';
  pop.classList.remove('hidden');
}

// Mouse-up anywhere → show selection popover if selection falls within a bookmark excerpt
document.addEventListener('mouseup', e => {
  setTimeout(() => {
    const sel     = window.getSelection();
    const selText = sel?.toString()?.trim() || '';
    if (!selText) {
      if (!e.ctrlKey) { _bmSelPopover?.classList.add('hidden'); _bmResetAccumulatedSel(); }
      return;
    }
    const anchorEl = sel.anchorNode?.parentElement?.closest('.bm-excerpt-text');
    const focusEl  = sel.focusNode?.parentElement?.closest('.bm-excerpt-text');
    const el = anchorEl || focusEl;
    if (!el) {
      if (!e.ctrlKey) { _bmSelPopover?.classList.add('hidden'); _bmResetAccumulatedSel(); }
      return;
    }
    const bmId = parseInt(el.dataset.bmId);

    if (e.ctrlKey && _bmAccumulatedSegments?.bmId === bmId) {
      // Append new span to the accumulation (skip duplicates)
      if (!_bmAccumulatedSegments.segments.includes(selText)) {
        _bmAccumulatedSegments.segments.push(selText);
      }
      window.getSelection()?.removeAllRanges(); // clear browser selection — span captured
      _bmSelectionState = { bmId, text: _bmCombinedText() };
      _updateBmSelPopoverBtn();
      // Persist all accumulated spans as pending highlights so nothing disappears
      _bmSetPendingHl(bmId, _bmCombinedText());
      // Hide hint once we have multiple spans
      const hint = document.getElementById('bm-sel-hint');
      if (hint) hint.classList.add('hidden');
      // Keep popover visible, reposition near cursor
      const pop = _ensureBmSelPopover();
      pop.style.left = Math.min(e.clientX + 4, window.innerWidth  - 200) + 'px';
      pop.style.top  = Math.min(e.clientY + 4, window.innerHeight -  60) + 'px';
      pop.classList.remove('hidden');
      // If the hl panel is already open for this bookmark, refresh it with combined text
      const hlPanel = document.getElementById(`bm-hl-panel-${bmId}`);
      if (hlPanel && !hlPanel.classList.contains('hidden')) {
        _renderBmHlPanel(bmId, _bmCombinedText());
      }
    } else {
      // Fresh selection — start accumulation (or reset if different bookmark)
      _bmAccumulatedSegments = { bmId, segments: [selText] };
      _bmSelectionState = { bmId, text: selText };
      _updateBmSelPopoverBtn();
      // Show pending highlight immediately so the first span stays marked during Ctrl+drag
      _bmSetPendingHl(bmId, selText);
      _showBmSelPopover(e.clientX, e.clientY, bmId, selText);
    }
  }, 10);
});

// Right-click inside a bookmark excerpt (text selected) → replace native menu
document.getElementById('bookmarks-container').addEventListener('contextmenu', e => {
  const excerptEl = e.target.closest('.bm-excerpt-text');
  if (!excerptEl) return;
  const sel     = window.getSelection();
  const selText = sel?.toString()?.trim() || '';
  if (!selText) return;
  const anchorEl = sel.anchorNode?.parentElement?.closest('.bm-excerpt-text');
  const focusEl  = sel.focusNode?.parentElement?.closest('.bm-excerpt-text');
  if (anchorEl !== excerptEl && focusEl !== excerptEl) return;
  e.preventDefault();
  _showBmSelPopover(e.clientX, e.clientY, parseInt(excerptEl.dataset.bmId), selText);
});

// Click outside popover → hide it; Ctrl+click keeps accumulation alive for next span
document.addEventListener('mousedown', e => {
  if (_bmSelPopover && !_bmSelPopover.contains(e.target)) {
    _bmSelPopover.classList.add('hidden');
    if (!e.ctrlKey) _bmResetAccumulatedSel();
  }
});

// -- CODING MANAGER --------------------------------------------------------

let _cmCodes          = [];
let _cmCategories     = [];
let _cmSelected       = new Set();
let _cmMergeMode      = false;
let _cmOpenCodeId     = null;
let _cmOpenCatId      = null;
let _cmCollapsed      = new Set();  // category ids that are collapsed
let _cmDragCodeId     = null;       // id of code being dragged
let _cmExpandedCodes  = new Set();  // code ids with excerpts expanded inline
let _cmExcerptsCache  = {};         // code id → excerpt rows (or null while loading)
let _cmDragCatId      = null;       // id of coding (category) being dragged
let _cmDragExcerpt    = null;       // { bookmarkId, sourceCodeId } being dragged
let _cmDragCopy       = false;      // true when Ctrl held at dragstart → copy instead of move
let _cmDragScrollRAF  = null;       // rAF id for edge-scroll during drag
let _cmOpenExcBmId    = null;       // bookmark_id open in excerpt panel
let _cmOpenExcSrcId   = null;       // source code id open in excerpt panel
let _cmOpenExcHlId    = null;       // highlight_id when a span-coding row is open (null for whole-bookmark)
let _cmOpenExcHlText  = null;       // highlighted_text for the span-coding row
let _cmExcPanelChangeMode = false;  // true when panel is in "change coding" mode
let _cmCtxMenuData    = null;       // excerpt right-click payload
let _cmFilterDateFrom = '';         // YYYY-MM-DD
let _cmFilterDateTo   = '';         // YYYY-MM-DD
let _cmFilterSuno     = 'all';      // 'all' | 'only' | 'exclude'

// Merge highlight rows that share the same bookmark_id (same code, different segments).
// Produces one row per bookmark with highlighted_text = "seg1 ... seg2".
function _cmMergeHighlightRows(rows) {
  const excerpts  = rows.filter(r => r.type === 'excerpt');
  const hlByBm    = {};
  for (const r of rows) {
    if (r.type !== 'highlight') continue;
    if (!hlByBm[r.bookmark_id]) hlByBm[r.bookmark_id] = { ...r, _segs: [r.highlighted_text] };
    else hlByBm[r.bookmark_id]._segs.push(r.highlighted_text);
  }
  const highlights = Object.values(hlByBm).map(r => ({
    ...r,
    highlighted_text: r._segs.join(_BM_SEG_SEP),
  }));
  return [...excerpts, ...highlights].sort((a, b) =>
    (b.created_at || '') > (a.created_at || '') ? 1 : -1
  );
}

function _cmFilterExcerpts(rows) {
  return rows.filter(r => {
    if (_cmFilterDateFrom || _cmFilterDateTo) {
      const d = (r.date || '').substring(0, 10);
      if (_cmFilterDateFrom && d < _cmFilterDateFrom) return false;
      if (_cmFilterDateTo   && d > _cmFilterDateTo)   return false;
    }
    if (_cmFilterSuno === 'only'    && !truthy(r.is_suno_team)) return false;
    if (_cmFilterSuno === 'exclude' &&  truthy(r.is_suno_team)) return false;
    return true;
  });
}

function _cmFilterActive() {
  return _cmFilterDateFrom || _cmFilterDateTo || _cmFilterSuno !== 'all';
}

async function loadCodingPage() {
  await _cmRefresh();
}

async function _cmRefresh() {
  try {
    [_cmCodes, _cmCategories] = await Promise.all([
      apiFetch('/api/codes'),
      apiFetch('/api/code-categories'),
    ]);
    if (_cmFilterActive()) {
      _cachedBookmarks = await apiFetch('/api/bookmarks');
    }
  } catch (e) {
    document.getElementById('cm-code-list').innerHTML =
      `<p class="text-sm text-red-500 text-center py-8">Failed to load: ${esc(e.message)}</p>`;
    return;
  }
  _cmPopulateCategorySelects();
  _cmRenderTree();
  // Re-fetch excerpts for expanded codes whose cache was invalidated (e.g. after an edit/delete)
  for (const codeId of _cmExpandedCodes) {
    if (_cmExcerptsCache[codeId] === undefined) _cmFetchExcerptsFor(codeId);
  }
  if (_cmOpenCodeId !== null) {
    const code = _cmCodes.find(c => c.id === _cmOpenCodeId);
    if (code) _cmOpenCodeDetail(code);
    else _cmCloseDetail();
  } else if (_cmOpenCatId !== null) {
    const cat = _cmCategories.find(c => c.id === _cmOpenCatId);
    if (cat) _cmOpenCatDetail(cat);
    else _cmCloseDetail();
  } else if (_cmOpenExcBmId !== null) {
    _cmOpenExcerptPanel(_cmOpenExcBmId, _cmOpenExcSrcId, _cmOpenExcHlId, _cmOpenExcHlText);
  }
}

// ── Category select population ────────────────────────────────────────────────

function _cmPopulateCategorySelects() {
  const sorted = [..._cmCategories].sort((a, b) =>
    a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }));

  const codeOpts = '<option value="">— Uncategorized —</option>' +
    sorted.map(c => `<option value="${c.id}">${esc(c.name)}</option>`).join('');
  document.getElementById('cm-nc-category').innerHTML   = codeOpts;
  document.getElementById('cm-edit-category').innerHTML = codeOpts;

  // Parent-coding selects (used for create & edit of codings)
  const parentOpts = '<option value="">— Root level (2nd-order) —</option>' +
    sorted.map(c => `<option value="${c.id}">${esc(c.name)}</option>`).join('');
  document.getElementById('cm-cat-parent').innerHTML      = parentOpts;
  document.getElementById('cm-cat-edit-parent').innerHTML = parentOpts;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

let _cmTreeMaxDepth = 0;

function _cmComputeMaxDepth() {
  let max = 0;
  function walk(catId, d) {
    max = Math.max(max, d);
    _cmCategories.filter(c => c.parent_id === catId).forEach(c => walk(c.id, d + 1));
  }
  _cmCategories.filter(c => c.parent_id == null).forEach(c => walk(c.id, 0));
  _cmTreeMaxDepth = max;
}

// Order label counts from the BOTTOM of the hierarchy:
//   category just above open codings = 2nd-order
//   category above that              = 3rd-order, etc.
// So n = (maxDepth - depth) + 2
function _cmOrderLabel(depth) {
  const n   = (_cmTreeMaxDepth - depth) + 2;
  const sfx = { 2: 'nd', 3: 'rd' }[n] || 'th';
  return `${n}${sfx}-order`;
}

function _cmGetDescendantCatIds(catId) {
  const result = new Set([catId]);
  const stack  = [catId];
  while (stack.length) {
    const id = stack.pop();
    _cmCategories
      .filter(c => c.parent_id === id)
      .forEach(c => { if (!result.has(c.id)) { result.add(c.id); stack.push(c.id); } });
  }
  return result;
}

// ── Tree building ─────────────────────────────────────────────────────────────

function _cmBuildTree() {
  const catMap = {};
  _cmCategories.forEach(c => { catMap[c.id] = { ...c, children: [], codes: [] }; });

  _cmCodes.forEach(code => {
    if (code.category_id != null && catMap[code.category_id]) {
      catMap[code.category_id].codes.push(code);
    }
  });

  const roots = [];
  _cmCategories.forEach(c => {
    if (c.parent_id != null && catMap[c.parent_id]) {
      catMap[c.parent_id].children.push(catMap[c.id]);
    } else {
      roots.push(catMap[c.id]);
    }
  });

  const sortByName = (a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' });
  const sortNode = node => {
    node.children.sort(sortByName);
    node.codes.sort(sortByName);
    node.children.forEach(sortNode);
  };
  roots.sort(sortByName);
  roots.forEach(sortNode);
  return roots;
}

function _cmCountCodes(node) {
  return node.codes.length + node.children.reduce((s, ch) => s + _cmCountCodes(ch), 0);
}

// Returns a Set of code IDs that have at least one quote matching the current filter,
// or null when no filter is active (meaning: show everything).
function _cmBuildFilteredCodeSet() {
  if (!_cmFilterActive() || !(_cachedBookmarks || []).length) return null;
  const ids = new Set();
  _cachedBookmarks.forEach(bm => {
    if (_cmFilterSuno === 'only'    && !truthy(bm.is_suno_team)) return;
    if (_cmFilterSuno === 'exclude' &&  truthy(bm.is_suno_team)) return;
    if (_cmFilterDateFrom || _cmFilterDateTo) {
      const d = (bm.date || '').substring(0, 10);
      if (_cmFilterDateFrom && d < _cmFilterDateFrom) return;
      if (_cmFilterDateTo   && d > _cmFilterDateTo)   return;
    }
    (bm.codes      || []).forEach(c  => ids.add(c.id));
    (bm.highlights || []).forEach(hl => ids.add(hl.code_id));
  });
  return ids;
}

function _cmCountVisibleCodes(node, visibleIds) {
  if (visibleIds === null) return _cmCountCodes(node);
  const direct = node.codes.filter(c => visibleIds.has(c.id)).length;
  return direct + node.children.reduce((s, ch) => s + _cmCountVisibleCodes(ch, visibleIds), 0);
}

// ── Tree rendering ────────────────────────────────────────────────────────────

function _cmRenderTree() {
  _cmComputeMaxDepth();
  const list = document.getElementById('cm-code-list');
  const roots = _cmBuildTree();
  const visibleIds = _cmBuildFilteredCodeSet(); // null = no filter, show all

  const uncategorized = _cmCodes
    .filter(c => c.category_id == null)
    .filter(c => visibleIds === null || visibleIds.has(c.id))
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }));

  let html = '';
  roots.forEach(node => { html += _cmRenderCatNode(node, 0, visibleIds); });

  // Uncategorized / unassigned open codings drop zone
  const ucCards = uncategorized.map(c => _cmRenderCodeCard(c, 0)).join('');
  html += `
    <div class="mt-2">
      <div class="cm-drop-zone flex items-center gap-2 py-1 px-2 rounded-lg"
           data-drop-cat-id="">
        <span class="text-xs font-semibold text-gray-400 uppercase tracking-wide">Uncategorized Open Codings</span>
      </div>
      <div class="space-y-1 mt-1">${ucCards || ''}</div>
    </div>`;

  if (!_cmCodes.length && !_cmCategories.length) {
    list.innerHTML = '<p class="text-sm text-gray-400 text-center py-8">No codes yet. Click "+ New Code" to create your first code.</p>';
    return;
  }
  list.innerHTML = html;
}

function _cmRenderCatNode(node, depth, visibleIds) {
  const visibleCodes = visibleIds === null ? node.codes : node.codes.filter(c => visibleIds.has(c.id));
  const childrenHtmlParts = node.children.map(ch => _cmRenderCatNode(ch, depth + 1, visibleIds));
  const visibleChildrenHtml = childrenHtmlParts.filter(h => h !== '').join('');

  // When filter is active, hide category entirely if it has no visible codes or children
  if (visibleIds !== null && visibleCodes.length === 0 && !visibleChildrenHtml) return '';

  const totalCodes   = _cmCountVisibleCodes(node, visibleIds);
  const hasContent   = visibleCodes.length > 0 || visibleChildrenHtml !== '';
  const isCollapsed  = _cmCollapsed.has(node.id);
  const pl           = 8 + depth * 18;
  const isOpenCat    = _cmOpenCatId === node.id;
  const headerBg     = isOpenCat ? 'bg-slate-100' : 'hover:bg-gray-50';
  const toggleIcon   = hasContent ? (isCollapsed ? '▶' : '▼') : '';

  const header = `
    <div class="cm-cat-header flex items-center gap-2 py-1.5 rounded-lg cursor-grab group transition-colors ${headerBg}"
         style="padding-left:${pl}px;padding-right:8px"
         data-cat-id="${node.id}"
         data-depth="${depth}"
         draggable="true">
      <span class="w-3 text-[9px] text-gray-400 shrink-0 cm-tree-toggle" data-cat-id="${node.id}">${toggleIcon}</span>
      <span class="w-2.5 h-2.5 rounded-full shrink-0" style="background:${node.color}"></span>
      <span class="text-xs font-semibold text-gray-600 uppercase tracking-wide flex-1 truncate">${esc(node.name)}</span>
      <span class="text-[9px] px-1 py-0.5 rounded bg-gray-100 text-gray-400 shrink-0 font-mono">${_cmOrderLabel(depth)}</span>
      <span class="text-[10px] text-gray-400 shrink-0">${totalCodes > 0 ? totalCodes + ' open codings' : ''}</span>
      <button class="hidden group-hover:inline text-[10px] text-indigo-500 hover:text-indigo-700 ml-1 cm-edit-cat-btn shrink-0"
              data-cat-id="${node.id}">edit</button>
    </div>`;

  if (isCollapsed) {
    return `<div class="cm-cat-node" data-cat-id="${node.id}">${header}</div>`;
  }

  const codesHtml = visibleCodes.map(c => _cmRenderCodeCard(c, depth + 1)).join('');

  // Inner drop zone (always rendered so user can drop into an empty category)
  const dropZone = `
    <div class="cm-drop-zone min-h-[6px] rounded" style="margin-left:${pl + 18}px"
         data-drop-cat-id="${node.id}"></div>`;

  return `
    <div class="cm-cat-node" data-cat-id="${node.id}">
      ${header}
      <div class="cm-cat-children space-y-1 mt-0.5">
        ${visibleChildrenHtml}
        <div class="space-y-1">${codesHtml}</div>
        ${dropZone}
      </div>
    </div>`;
}

// Renders a single row in a code's inline excerpt list.
// r.type === 'excerpt'  → whole-bookmark coding (draggable, full snippet)
// r.type === 'highlight' → span-level coding (read-only, shows highlighted text)
function _cmRenderExcerptRow(r, codeId, accent) {
  const meta = `${esc(r.username || '')}${r.date ? ' · ' + esc(r.date.substring(0, 10)) : ''}`;
  if (r.type === 'highlight') {
    const hlSnippet = r.highlighted_text || '';
    return `<div class="border-l-2 pl-2 py-1 cm-excerpt-item group rounded-r-md transition-colors hover:bg-amber-50/60"
                 style="border-color:${accent}"
                 draggable="true"
                 data-bookmark-id="${r.bookmark_id}"
                 data-source-code-id="${codeId}"
                 data-highlight-id="${r.highlight_id}"
                 data-hl-text="${esc(hlSnippet)}"
                 title="Drag to move · Ctrl+drag to copy">
      <div class="flex items-start gap-1.5">
        <div class="flex-1 min-w-0">
          <p class="text-xs italic text-gray-800 leading-relaxed font-medium">"${esc(hlSnippet)}"</p>
          <p class="text-[10px] mt-0.5 flex items-center gap-1">
            <span class="px-1 py-0.5 rounded text-[9px] font-semibold" style="background:${accent}20;color:${accent}">span</span>
            <span class="text-gray-400">${meta}</span>
          </p>
        </div>
        <span class="text-gray-300 shrink-0 mt-0.5 text-[11px] leading-none">⠿</span>
      </div>
    </div>`;
  }
  const snippet = (r.content || '').substring(0, 180);
  const more    = (r.content || '').length > 180 ? '…' : '';
  return `<div class="border-l-2 pl-2 py-1 cm-excerpt-item group rounded-r-md transition-colors hover:bg-indigo-50/60"
               style="border-color:${accent}"
               draggable="true"
               data-bookmark-id="${r.bookmark_id}"
               data-source-code-id="${codeId}"
               title="Drag to move · Ctrl+drag to copy">
    <div class="flex items-start gap-1.5">
      <div class="flex-1 min-w-0">
        <p class="text-xs italic text-gray-700 leading-relaxed">"${esc(snippet)}${more}"</p>
        <p class="text-[10px] text-gray-400 mt-0.5">${meta}</p>
        ${r.note ? `<p class="text-[10px] text-indigo-600">${esc(r.note)}</p>` : ''}
      </div>
      <span class="text-gray-300 shrink-0 mt-0.5 text-[11px] leading-none">⠿</span>
    </div>
  </div>`;
}

function _cmRenderCodeCard(code, depth) {
  const tc         = labelTextColor(code.color);
  const selected   = _cmSelected.has(code.id);
  const isOpen     = _cmOpenCodeId === code.id;
  const isExpanded = _cmExpandedCodes.has(code.id);
  const selCls     = selected ? 'ring-2 ring-amber-400 bg-amber-50'
                   : (isOpen ? 'border-indigo-400 bg-indigo-50' : 'bg-white hover:border-indigo-200');
  const pl         = 8 + depth * 18;

  // Inline excerpts section
  let excerptsHtml = '';
  if (isExpanded) {
    const cached = _cmExcerptsCache[code.id];
    if (cached === undefined) {
      excerptsHtml = `<div id="cm-exc-${code.id}" class="mt-1 text-xs text-gray-400 italic text-center py-1.5 border-l-2 border-indigo-200 pl-2 ml-3">Loading excerpts…</div>`;
    } else {
      const visible = _cmFilterExcerpts(cached);
      if (visible.length === 0) {
        excerptsHtml = `<div id="cm-exc-${code.id}" class="mt-1 text-xs text-gray-400 italic text-center py-1.5">${cached.length ? 'No excerpts match the current filter.' : 'No excerpts yet.'}</div>`;
      } else {
        const items = visible.map(r => _cmRenderExcerptRow(r, code.id, code.color)).join('');
        excerptsHtml = `<div id="cm-exc-${code.id}" class="mt-1 ml-3 space-y-1.5">${items}</div>`;
      }
    }
  }

  return `
    <div class="cm-code-outer" style="margin-left:${pl}px" data-outer-code-id="${code.id}">
      <div class="cm-code-card border rounded-xl p-2.5 flex items-start gap-2 cursor-pointer transition-all ${selCls}"
           data-code-id="${code.id}"
           data-depth="${depth}"
           draggable="${_cmMergeMode ? 'false' : 'true'}">
        ${_cmMergeMode ? `<input type="checkbox" class="mt-0.5 accent-amber-500 shrink-0 cm-select-cb" ${selected ? 'checked' : ''} data-code-id="${code.id}" />` : ''}
        <span class="w-3 h-3 rounded-full shrink-0 mt-0.5" style="background:${code.color}"></span>
        <div class="flex-1 min-w-0">
          <div class="flex items-center gap-2 flex-wrap">
            <span class="text-sm font-semibold text-gray-800">${esc(code.name)}</span>
            <span class="text-[10px] px-1.5 py-0.5 rounded-full font-medium" style="background:${code.color};color:${tc}">${code.groundedness} quotes</span>
          </div>
          ${code.description ? `<p class="text-xs text-gray-500 mt-0.5 truncate">${esc(code.description)}</p>` : ''}
        </div>
        <span class="text-[10px] text-gray-400 shrink-0 mt-0.5 select-none">${isExpanded ? '▲' : '▼'}</span>
      </div>
      ${excerptsHtml}
    </div>`;
}

// ── Detail panel: Code ────────────────────────────────────────────────────────

function _cmOpenCodeDetail(code) {
  _cmFlushDetailPanel();
  _cmOpenCodeId = code.id;
  document.getElementById('cm-detail-title').textContent = 'Edit Open Coding';
  document.getElementById('cm-code-edit-section').classList.remove('hidden');
  document.getElementById('cm-edit-name').value          = code.name;
  _setPickerColor('cm-edit-color', code.color);
  document.getElementById('cm-edit-desc').value          = code.description || '';
  document.getElementById('cm-edit-category').value      = code.category_id ?? '';
  document.getElementById('cm-edit-ground').textContent  = code.groundedness ?? '—';
  document.getElementById('cm-edit-density').textContent = code.density ?? '—';
  document.getElementById('cm-detail-panel').classList.remove('hidden');
}

async function _cmFetchExcerptsFor(codeId) {
  try {
    const rows = await apiFetch(`/api/codes/${codeId}/bookmarks`);
    _cmExcerptsCache[codeId] = _cmMergeHighlightRows(rows);
  } catch (_) {
    _cmExcerptsCache[codeId] = [];
  }
  // Targeted DOM update — avoid full re-render to preserve scroll position
  const container = document.getElementById(`cm-exc-${codeId}`);
  if (!container || !_cmExpandedCodes.has(codeId)) return;
  const cached  = _cmExcerptsCache[codeId];
  const visible = _cmFilterExcerpts(cached);
  if (!cached.length) {
    container.innerHTML = '<p class="text-xs text-gray-400 italic text-center py-1.5">No excerpts yet.</p>';
    return;
  }
  if (!visible.length) {
    container.innerHTML = '<p class="text-xs text-gray-400 italic text-center py-1.5">No excerpts match the current filter.</p>';
    return;
  }
  // Find the code's color for the left-border accent
  const code   = _cmCodes.find(c => c.id === codeId);
  const accent = code ? code.color : '#0d3e7f';
  container.innerHTML = visible.map(r => _cmRenderExcerptRow(r, codeId, accent)).join('');
}

// ── Detail panel: Category ────────────────────────────────────────────────────

function _cmOpenCatDetail(cat) {
  _cmFlushDetailPanel();
  _cmOpenCatId = cat.id;
  document.getElementById('cm-detail-title').textContent = 'Edit Coding';
  document.getElementById('cm-cat-edit-section').classList.remove('hidden');
  document.getElementById('cm-cat-edit-name').value  = cat.name;
  _setPickerColor('cm-cat-edit-color', cat.color);
  // Filter out self and descendants to avoid cycles
  const parentEl = document.getElementById('cm-cat-edit-parent');
  parentEl.innerHTML = '<option value="">— Root level (2nd-order) —</option>' +
    _cmCategories
      .filter(c => c.id !== cat.id)
      .sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }))
      .map(c => `<option value="${c.id}">${esc(c.name)}</option>`)
      .join('');
  parentEl.value = cat.parent_id ?? '';
  document.getElementById('cm-detail-panel').classList.remove('hidden');
}

function _cmFlushDetailPanel() {
  _cmOpenCodeId         = null;
  _cmOpenCatId          = null;
  _cmOpenExcBmId        = null;
  _cmOpenExcSrcId       = null;
  _cmOpenExcHlId        = null;
  _cmOpenExcHlText      = null;
  _cmExcPanelChangeMode = false;
  document.getElementById('cm-code-edit-section').classList.add('hidden');
  document.getElementById('cm-cat-edit-section').classList.add('hidden');
  document.getElementById('cm-exc-edit-section').classList.add('hidden');
  document.getElementById('cm-edit-msg').classList.add('hidden');
  document.getElementById('cm-cat-edit-msg').classList.add('hidden');
  document.getElementById('cm-exc-panel-suggestions').classList.add('hidden');
  const _ctxBtn = document.getElementById('cm-exc-ctx-btn');
  if (_ctxBtn) { _ctxBtn.classList.add('hidden'); _ctxBtn.dataset.msgId = ''; }
}

function _cmCloseDetail() {
  _cmFlushDetailPanel();
  document.getElementById('cm-detail-panel').classList.add('hidden');
}

// ── Excerpt coding panel (sidebar) ───────────────────────────────────────────

async function _cmOpenExcerptPanel(bookmarkId, sourceCodeId, hlId = null, hlText = null, changeMode = false) {
  let bm = _cachedBookmarks.find(b => b.bookmark_id === bookmarkId);
  if (!bm) {
    try {
      _cachedBookmarks = await apiFetch('/api/bookmarks');
      bm = _cachedBookmarks.find(b => b.bookmark_id === bookmarkId);
    } catch (_) {}
  }
  if (!bm) return;

  _cmFlushDetailPanel();
  _cmOpenExcBmId        = bookmarkId;
  _cmOpenExcSrcId       = sourceCodeId;
  _cmOpenExcHlId        = hlId;
  _cmOpenExcHlText      = hlText;
  _cmExcPanelChangeMode = changeMode;

  const textEl   = document.getElementById('cm-exc-panel-text');
  const srcCode  = _cmCodes.find(c => c.id === sourceCodeId);
  const addEl    = document.getElementById('cm-exc-panel-add-section');
  const codesLbl = document.querySelector('#cm-exc-edit-section .text-xs.font-medium');
  const addLbl   = addEl?.querySelector('label');

  document.getElementById('cm-exc-edit-section').classList.remove('hidden');
  document.getElementById('cm-detail-panel').classList.remove('hidden');
  document.getElementById('cm-exc-panel-source').textContent =
    [bm.username, bm.date ? bm.date.substring(0, 10) : null].filter(Boolean).join(' · ');
  textEl.style.borderLeftColor = srcCode ? srcCode.color : '#0d3e7f';

  // Show conversation context button
  const _ctxBtn = document.getElementById('cm-exc-ctx-btn');
  if (_ctxBtn) {
    _ctxBtn.dataset.msgId  = bm.id || '';
    _ctxBtn.dataset.source = [bm.username, bm.date ? bm.date.substring(0, 10) : null].filter(Boolean).join(' · ');
    _ctxBtn.classList.remove('hidden');
  }

  if (changeMode) {
    // ── Change-coding mode: show source chip, search for replacement ─────────
    document.getElementById('cm-detail-title').textContent =
      hlId !== null ? 'Change Span Coding' : 'Change Open Coding';
    textEl.innerHTML = hlId !== null
      ? `<strong class="not-italic font-semibold text-gray-800 block mb-1">"${esc(hlText || '')}"</strong>`
        + `<span class="text-gray-400 text-[10px] block mt-1">Full excerpt: ${esc((bm.content || '').substring(0, 200))}${(bm.content || '').length > 200 ? '…' : ''}</span>`
      : esc(`"${bm.content || ''}"`);
    if (codesLbl) codesLbl.textContent = 'Changing from:';
    // Show source code chip (read-only, no × button)
    const codesContainer = document.getElementById('cm-exc-panel-codes');
    if (srcCode) {
      const tc = labelTextColor(srcCode.color);
      codesContainer.innerHTML =
        `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium"
               style="background:${srcCode.color};color:${tc}">${esc(srcCode.name)}</span>`;
    } else {
      codesContainer.innerHTML = '';
    }
    if (addEl) addEl.classList.remove('hidden');
    if (addLbl) addLbl.textContent = 'Replace with:';
    const searchEl = document.getElementById('cm-exc-panel-search');
    searchEl.value = '';
    searchEl.placeholder = 'Search or create replacement coding…';
    searchEl.focus();
    return;
  }

  // Reset search placeholder in case it was changed by change-mode
  const searchEl = document.getElementById('cm-exc-panel-search');
  searchEl.placeholder = 'Type to search or create…';
  if (addLbl) addLbl.textContent = 'Add Open Coding';

  if (hlId !== null) {
    // ── Span-level coding view ──────────────────────────────────────────────
    document.getElementById('cm-detail-title').textContent = 'Span Coding';
    textEl.innerHTML = `<strong class="not-italic font-semibold text-gray-800 block mb-1">"${esc(hlText || '')}"</strong>`
      + `<span class="text-gray-400 text-[10px] block mt-1">Full excerpt: ${esc((bm.content || '').substring(0, 200))}${(bm.content || '').length > 200 ? '…' : ''}</span>`;
    if (codesLbl) codesLbl.textContent = 'Span coding';
    _cmExcPanelRenderSpanCode(srcCode, bookmarkId, hlId);
    if (addEl) addEl.classList.add('hidden');
  } else {
    // ── Whole-bookmark coding view ──────────────────────────────────────────
    document.getElementById('cm-detail-title').textContent = 'Edit Open Codings';
    textEl.textContent = `"${bm.content || ''}"`;
    if (codesLbl) codesLbl.textContent = 'Open Codings';
    _cmExcPanelRenderCodes(bm, bookmarkId);
    if (addEl) addEl.classList.remove('hidden');
    searchEl.value = '';
    searchEl.focus();
  }
}

function _cmExcPanelRenderSpanCode(code, bookmarkId, hlId) {
  const container = document.getElementById('cm-exc-panel-codes');
  if (!code) {
    container.innerHTML = '<span class="text-xs text-gray-400 italic">Code not found</span>';
    return;
  }
  const tc = labelTextColor(code.color);
  container.innerHTML = `
    <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium"
          style="background:${code.color};color:${tc}">
      ${esc(code.name)}
    </span>
    <p class="text-[10px] text-gray-400 mt-1 w-full">Manage span codings from the <strong>Bookmarks</strong> page.</p>`;
}

function _cmExcPanelRenderCodes(bm, bookmarkId) {
  const container = document.getElementById('cm-exc-panel-codes');
  const codes     = bm ? (bm.codes || []) : [];
  if (!codes.length) {
    container.innerHTML = '<span class="text-xs text-gray-400 italic">No open codings assigned yet</span>';
    return;
  }
  container.innerHTML = codes.map(c => {
    const tc = labelTextColor(c.color);
    return `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium"
                  style="background:${c.color};color:${tc}">
      ${esc(c.name)}
      <button class="cm-exc-panel-remove-code ml-0.5 hover:opacity-70 font-bold leading-none"
              data-code-id="${c.id}" data-bookmark-id="${bookmarkId}"
              title="Remove this coding">×</button>
    </span>`;
  }).join('');
}

function _cmExcPanelShowSuggestions(q) {
  const bm          = _cachedBookmarks.find(b => b.bookmark_id === _cmOpenExcBmId);
  // In change mode exclude only the source code; in add mode exclude all already-assigned codes
  const assignedIds = _cmExcPanelChangeMode
    ? new Set([_cmOpenExcSrcId])
    : new Set((bm ? bm.codes || [] : []).map(c => c.id));
  const sugEl       = document.getElementById('cm-exc-panel-suggestions');

  if (!q) { sugEl.classList.add('hidden'); sugEl.innerHTML = ''; return; }

  const ql      = q.toLowerCase();
  const matches = _cmCodes
    .filter(c => !assignedIds.has(c.id) && c.name.toLowerCase().includes(ql))
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }));
  const exactExists = _cmCodes.some(c => c.name.toLowerCase() === ql);

  let html = matches.map(c =>
    `<div class="cm-exc-sug-item flex items-center gap-2 px-3 py-2 hover:bg-indigo-50 cursor-pointer text-sm"
          data-code-id="${c.id}">
      <span class="w-2.5 h-2.5 rounded-full shrink-0" style="background:${c.color}"></span>
      <span>${esc(c.name)}</span>
    </div>`
  ).join('');

  if (!exactExists) {
    html += `<div class="cm-exc-sug-create flex items-center gap-2 px-3 py-2 hover:bg-green-50 cursor-pointer text-sm text-green-700 ${matches.length ? 'border-t border-gray-100' : ''}">
      <span class="text-base font-bold leading-none">+</span>
      <span>Create <strong>"${esc(q)}"</strong> as new open coding</span>
    </div>`;
  }

  if (html) { sugEl.innerHTML = html; sugEl.classList.remove('hidden'); }
  else       { sugEl.innerHTML = ''; sugEl.classList.add('hidden'); }
}

async function _cmExcPanelAssignCode(codeId) {
  const bookmarkId = _cmOpenExcBmId;
  const bm         = _cachedBookmarks.find(b => b.bookmark_id === bookmarkId);

  if (_cmExcPanelChangeMode) {
    const oldCodeId = _cmOpenExcSrcId;
    try {
      if (_cmOpenExcHlId !== null) {
        // Span: POST new highlight with new code, DELETE old highlight
        await apiFetch(`/api/bookmarks/${bookmarkId}/highlights`, {
          method: 'POST',
          body: JSON.stringify({ code_id: codeId, highlighted_text: _cmOpenExcHlText }),
        });
        await apiFetch(`/api/bookmarks/${bookmarkId}/highlights/${_cmOpenExcHlId}`, { method: 'DELETE' });
        if (bm) bm.highlights = (bm.highlights || []).filter(h => h.highlight_id !== _cmOpenExcHlId);
      } else {
        // Whole-bookmark: DELETE old code assignment, POST new code assignment
        await apiFetch(`/api/bookmarks/${bookmarkId}/codes/${oldCodeId}`, { method: 'DELETE' });
        await apiFetch(`/api/bookmarks/${bookmarkId}/codes/${codeId}`, { method: 'POST' });
        const newCode = _cmCodes.find(c => c.id === codeId);
        if (bm) {
          bm.codes = (bm.codes || []).filter(c => c.id !== oldCodeId);
          if (newCode && !(bm.codes).some(c => c.id === codeId)) {
            bm.codes.push({ id: newCode.id, name: newCode.name, color: newCode.color,
              description: newCode.description, category_id: newCode.category_id });
          }
        }
      }
    } catch (err) {
      showErrorPopup('Failed to change coding: ' + err.message);
      _cmExcPanelChangeMode = false;
      return;
    }
    delete _cmExcerptsCache[oldCodeId];
    delete _cmExcerptsCache[codeId];
    _cmCloseDetail();
    if (_cmExpandedCodes.has(oldCodeId)) _cmFetchExcerptsFor(oldCodeId);
    if (_cmExpandedCodes.has(codeId))    _cmFetchExcerptsFor(codeId);
    return;
  }

  // Normal ADD mode
  try {
    await apiFetch(`/api/bookmarks/${bookmarkId}/codes/${codeId}`, { method: 'POST' });
    const code = _cmCodes.find(c => c.id === codeId);
    if (bm && code && !(bm.codes || []).some(c => c.id === codeId)) {
      bm.codes = [...(bm.codes || []),
        { id: code.id, name: code.name, color: code.color,
          description: code.description, category_id: code.category_id }];
    }
    delete _cmExcerptsCache[codeId];
    _cmExcPanelRenderCodes(bm, bookmarkId);
    document.getElementById('cm-exc-panel-search').value = '';
    document.getElementById('cm-exc-panel-suggestions').classList.add('hidden');
  } catch (err) {
    showErrorPopup('Failed to assign coding: ' + err.message);
  }
}

// Panel search input → live suggestions + Enter to confirm
document.getElementById('cm-exc-panel-search').addEventListener('input', e => {
  _cmExcPanelShowSuggestions(e.target.value.trim());
});
document.getElementById('cm-exc-panel-search').addEventListener('focus', e => {
  if (e.target.value.trim()) _cmExcPanelShowSuggestions(e.target.value.trim());
});
document.getElementById('cm-exc-panel-search').addEventListener('keydown', async e => {
  if (e.key !== 'Enter') return;
  e.preventDefault();
  const q = e.target.value.trim();
  if (!q) return;
  const bm          = _cachedBookmarks.find(b => b.bookmark_id === _cmOpenExcBmId);
  const assignedIds = _cmExcPanelChangeMode
    ? new Set([_cmOpenExcSrcId])
    : new Set((bm ? bm.codes || [] : []).map(c => c.id));
  const match       = _cmCodes.find(c =>
    !assignedIds.has(c.id) && c.name.toLowerCase() === q.toLowerCase()
  );
  if (match) {
    await _cmExcPanelAssignCode(match.id);
  } else {
    const exactExists = _cmCodes.some(c => c.name.toLowerCase() === q.toLowerCase());
    if (!exactExists) {
      try {
        const newCode = await apiFetch('/api/codes', { method: 'POST', body: JSON.stringify({ name: q }) });
        _cmCodes.push(newCode);
        _allCodes.push(newCode);
        await _cmExcPanelAssignCode(newCode.id);
      } catch (err) {
        showErrorPopup('Failed to create coding: ' + err.message);
      }
    }
  }
});

// Suggestions: pick existing or create new
document.getElementById('cm-exc-panel-suggestions').addEventListener('mousedown', async e => {
  const item   = e.target.closest('.cm-exc-sug-item');
  const create = e.target.closest('.cm-exc-sug-create');
  if (item) {
    e.preventDefault();
    await _cmExcPanelAssignCode(parseInt(item.dataset.codeId));
    return;
  }
  if (create) {
    e.preventDefault();
    const name = document.getElementById('cm-exc-panel-search').value.trim();
    if (!name) return;
    try {
      const newCode = await apiFetch('/api/codes', { method: 'POST', body: JSON.stringify({ name }) });
      _cmCodes.push(newCode);
      _allCodes.push(newCode);
      await _cmExcPanelAssignCode(newCode.id);
    } catch (err) {
      showErrorPopup('Failed to create coding: ' + err.message);
    }
  }
});

// Remove code chip
document.getElementById('cm-exc-panel-codes').addEventListener('click', async e => {
  const btn = e.target.closest('.cm-exc-panel-remove-code');
  if (!btn) return;
  const bookmarkId = parseInt(btn.dataset.bookmarkId);
  const codeId     = parseInt(btn.dataset.codeId);
  try {
    await apiFetch(`/api/bookmarks/${bookmarkId}/codes/${codeId}`, { method: 'DELETE' });
    const bm = _cachedBookmarks.find(b => b.bookmark_id === bookmarkId);
    if (bm) bm.codes = (bm.codes || []).filter(c => c.id !== codeId);
    delete _cmExcerptsCache[codeId];
    _cmExcPanelRenderCodes(bm, bookmarkId);
  } catch (err) {
    showErrorPopup('Failed to remove coding: ' + err.message);
  }
});

// ── Excerpt row right-click context menu ─────────────────────────────────────

document.getElementById('cm-code-list').addEventListener('contextmenu', e => {
  const row = e.target.closest('.cm-excerpt-item');
  if (!row) return;
  e.preventDefault();
  _cmCtxMenuData = {
    bookmarkId:   parseInt(row.dataset.bookmarkId),
    sourceCodeId: parseInt(row.dataset.sourceCodeId),
    hlId:         row.dataset.highlightId ? parseInt(row.dataset.highlightId) : null,
    hlText:       row.dataset.hlText || null,
  };
  const menu = document.getElementById('cm-exc-ctx-menu');
  const x = Math.min(e.clientX, window.innerWidth  - 220);
  const y = Math.min(e.clientY, window.innerHeight - 80);
  menu.style.left = x + 'px';
  menu.style.top  = y + 'px';
  menu.classList.remove('hidden');
});

document.getElementById('cm-ctx-add-coding').addEventListener('click', () => {
  if (!_cmCtxMenuData) return;
  const { bookmarkId, sourceCodeId, hlId, hlText } = _cmCtxMenuData;
  _cmCtxMenuData = null;
  _cmOpenExcerptPanel(bookmarkId, sourceCodeId, hlId, hlText, false);
});

document.getElementById('cm-ctx-change-coding').addEventListener('click', () => {
  if (!_cmCtxMenuData) return;
  const { bookmarkId, sourceCodeId, hlId, hlText } = _cmCtxMenuData;
  _cmCtxMenuData = null;
  _cmOpenExcerptPanel(bookmarkId, sourceCodeId, hlId, hlText, true);
});

// Dismiss context menu on any outside click or right-click
document.addEventListener('click', e => {
  if (!e.target.closest('#cm-exc-ctx-menu')) {
    document.getElementById('cm-exc-ctx-menu')?.classList.add('hidden');
  }
});

// ── Tree click delegation ────────────────────────────────────────────────────

document.getElementById('cm-code-list').addEventListener('click', e => {
  // Checkbox in merge mode
  const cb = e.target.closest('.cm-select-cb');
  if (cb) {
    const id = parseInt(cb.dataset.codeId);
    if (_cmSelected.has(id)) _cmSelected.delete(id); else _cmSelected.add(id);
    _cmUpdateMergeBtn();
    _cmRenderTree();
    return;
  }

  // Collapse toggle
  const toggle = e.target.closest('.cm-tree-toggle');
  if (toggle) {
    const id = parseInt(toggle.dataset.catId);
    if (_cmCollapsed.has(id)) _cmCollapsed.delete(id); else _cmCollapsed.add(id);
    _cmRenderTree();
    return;
  }

  // Edit category inline button
  const editBtn = e.target.closest('.cm-edit-cat-btn');
  if (editBtn) {
    e.stopPropagation();
    const cat = _cmCategories.find(c => c.id === parseInt(editBtn.dataset.catId));
    if (cat) _cmOpenCatDetail(cat);
    return;
  }

  // Category header → toggle collapse or open detail
  const catHeader = e.target.closest('.cm-cat-header');
  if (catHeader && !e.target.closest('.cm-tree-toggle') && !e.target.closest('.cm-edit-cat-btn')) {
    const id = parseInt(catHeader.dataset.catId);
    const cat = _cmCategories.find(c => c.id === id);
    if (cat) _cmOpenCatDetail(cat);
    return;
  }

  // Code card
  const card = e.target.closest('.cm-code-card');
  if (!card) return;
  if (_cmMergeMode) {
    const id = parseInt(card.dataset.codeId);
    if (_cmSelected.has(id)) _cmSelected.delete(id); else _cmSelected.add(id);
    _cmUpdateMergeBtn();
    _cmRenderTree();
    return;
  }
  const codeId = parseInt(card.dataset.codeId);
  const code   = _cmCodes.find(c => c.id === codeId);
  if (!code) return;

  // Toggle inline excerpts
  const wasExpanded = _cmExpandedCodes.has(codeId);
  if (wasExpanded) {
    _cmExpandedCodes.delete(codeId);
  } else {
    _cmExpandedCodes.add(codeId);
  }

  _cmOpenCodeDetail(code);
  _cmRenderTree();

  // Fetch excerpts if expanding and not yet cached
  if (!wasExpanded && _cmExcerptsCache[codeId] === undefined) {
    _cmFetchExcerptsFor(codeId);
  }
});

// ── Drag & drop ───────────────────────────────────────────────────────────────

// ── Excerpt item mouse handling: prevent text selection, detect click vs drag ──

let _cmExcMdEl = null;  // element where mousedown started
let _cmExcMdX  = 0;
let _cmExcMdY  = 0;

document.getElementById('cm-code-list').addEventListener('mousedown', e => {
  const item = e.target.closest('.cm-excerpt-item');
  if (!item) return;
  // Do NOT preventDefault here — it would kill dragstart in Chrome/Edge.
  // Text selection is prevented by CSS user-select:none !important on the element.
  _cmExcMdEl = item;
  _cmExcMdX  = e.clientX;
  _cmExcMdY  = e.clientY;
});

document.getElementById('cm-code-list').addEventListener('mouseup', e => {
  const el = _cmExcMdEl;
  _cmExcMdEl = null;
  if (!el) return;
  const moved = Math.abs(e.clientX - _cmExcMdX) > 5 || Math.abs(e.clientY - _cmExcMdY) > 5;
  if (moved) return;   // was a drag, not a click
  // Treat as click → open excerpt / span-coding panel
  const bmId    = parseInt(el.dataset.bookmarkId);
  const srcCode = parseInt(el.dataset.sourceCodeId);
  const hlIdStr = el.dataset.highlightId;
  if (hlIdStr) {
    const hlId   = parseInt(hlIdStr);
    const cached = _cmExcerptsCache[srcCode];
    const row    = cached?.find(r => r.highlight_id === hlId);
    _cmOpenExcerptPanel(bmId, srcCode, hlId, row?.highlighted_text || '');
  } else {
    _cmOpenExcerptPanel(bmId, srcCode);
  }
});

// Belt-and-suspenders: also kill selectstart on excerpt items
document.getElementById('cm-code-list').addEventListener('selectstart', e => {
  if (e.target.closest('.cm-excerpt-item[draggable="true"]')) e.preventDefault();
});

document.getElementById('cm-code-list').addEventListener('dragstart', e => {
  // Excerpt drag — must be checked first (excerpts are inside code-outer wrappers)
  // Skip span-level highlight items (they are not draggable)
  const excerptItem = e.target.closest('.cm-excerpt-item[draggable="true"]');
  if (excerptItem) {
    _cmExcMdEl  = null;   // cancel click detection — this is a real drag
    _cmDragCopy = e.ctrlKey;
    _cmDragExcerpt = {
      bookmarkId:   parseInt(excerptItem.dataset.bookmarkId),
      sourceCodeId: parseInt(excerptItem.dataset.sourceCodeId),
      highlightId:  excerptItem.dataset.highlightId ? parseInt(excerptItem.dataset.highlightId) : null,
      hlText:       excerptItem.dataset.hlText || null,
    };
    _cmDragCodeId = null; _cmDragCatId = null;
    e.dataTransfer.effectAllowed = _cmDragCopy ? 'copy' : 'move';
    e.dataTransfer.setData('text/plain', 'excerpt');
    excerptItem.classList.add('opacity-50');
    return;
  }
  // Open coding card drag
  const card = e.target.closest('.cm-code-card');
  if (card) {
    _cmDragCodeId = parseInt(card.dataset.codeId);
    _cmDragCatId  = null; _cmDragExcerpt = null;
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', 'code:' + _cmDragCodeId);
    card.classList.add('opacity-50');
    return;
  }
  // Coding (category) header drag — not toggle/edit buttons
  const catHeader = e.target.closest('.cm-cat-header');
  if (catHeader && !e.target.closest('.cm-tree-toggle') && !e.target.closest('.cm-edit-cat-btn')) {
    _cmDragCatId  = parseInt(catHeader.dataset.catId);
    _cmDragCodeId = null; _cmDragExcerpt = null;
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', 'cat:' + _cmDragCatId);
    catHeader.classList.add('opacity-50');
  }
});

// Auto-scroll the page when dragging near the top/bottom viewport edge.
function _cmCancelDragScroll() {
  if (_cmDragScrollRAF !== null) { cancelAnimationFrame(_cmDragScrollRAF); _cmDragScrollRAF = null; }
}

document.addEventListener('dragover', e => {
  const active = _cmDragCodeId !== null || _cmDragCatId !== null || _cmDragExcerpt !== null;
  if (!active) { _cmCancelDragScroll(); return; }
  const EDGE = 80, MAX_SPEED = 14;
  const y = e.clientY, h = window.innerHeight;
  _cmCancelDragScroll();
  if (y < EDGE) {
    const speed = Math.max(1, Math.round(MAX_SPEED * (1 - y / EDGE)));
    const tick  = () => { window.scrollBy(0, -speed); _cmDragScrollRAF = requestAnimationFrame(tick); };
    _cmDragScrollRAF = requestAnimationFrame(tick);
  } else if (y > h - EDGE) {
    const speed = Math.max(1, Math.round(MAX_SPEED * (1 - (h - y) / EDGE)));
    const tick  = () => { window.scrollBy(0,  speed); _cmDragScrollRAF = requestAnimationFrame(tick); };
    _cmDragScrollRAF = requestAnimationFrame(tick);
  }
});

document.getElementById('cm-code-list').addEventListener('dragend', () => {
  _cmCancelDragScroll();
  _cmDragCodeId  = null;
  _cmDragCatId   = null;
  _cmDragExcerpt = null;
  _cmDragCopy    = false;
  document.querySelectorAll('.cm-excerpt-item').forEach(i => i.classList.remove('opacity-50'));
  document.querySelectorAll('.cm-code-card').forEach(c => c.classList.remove('opacity-50'));
  document.querySelectorAll('.cm-code-outer').forEach(o => o.classList.remove('ring-2', 'ring-green-400', 'ring-blue-400', 'rounded-xl'));
  document.querySelectorAll('.cm-cat-header').forEach(h => h.classList.remove('opacity-50', 'bg-indigo-50'));
  document.querySelectorAll('.cm-drop-zone').forEach(z => z.classList.remove('bg-indigo-100', 'ring-2', 'ring-indigo-400'));
});

document.getElementById('cm-code-list').addEventListener('dragover', e => {
  // ── Excerpt drag: only open-coding wrappers are valid targets ─────────────
  if (_cmDragExcerpt !== null) {
    const outer = e.target.closest('.cm-code-outer');
    if (outer && parseInt(outer.dataset.outerCodeId) !== _cmDragExcerpt.sourceCodeId) {
      e.preventDefault();
      const isCopy = e.ctrlKey;
      e.dataTransfer.dropEffect = isCopy ? 'copy' : 'move';
      outer.classList.remove('ring-green-400', 'ring-blue-400');
      outer.classList.add('ring-2', isCopy ? 'ring-blue-400' : 'ring-green-400', 'rounded-xl');
    }
    return;
  }
  // ── Code/cat drag: zones and cat headers are valid ────────────────────────
  if (_cmDragCodeId === null && _cmDragCatId === null) return;
  const zone = e.target.closest('.cm-drop-zone');
  if (zone) { e.preventDefault(); zone.classList.add('bg-indigo-100', 'ring-2', 'ring-indigo-400'); return; }
  const catHeader = e.target.closest('.cm-cat-header');
  if (catHeader) {
    const targetCatId = parseInt(catHeader.dataset.catId);
    if (_cmDragCatId !== null && _cmGetDescendantCatIds(_cmDragCatId).has(targetCatId)) return;
    e.preventDefault();
    catHeader.classList.add('bg-indigo-50');
  }
});

document.getElementById('cm-code-list').addEventListener('dragleave', e => {
  if (_cmDragExcerpt !== null) {
    const outer = e.target.closest('.cm-code-outer');
    if (outer) outer.classList.remove('ring-2', 'ring-green-400', 'ring-blue-400', 'rounded-xl');
    return;
  }
  const zone = e.target.closest('.cm-drop-zone');
  if (zone) zone.classList.remove('bg-indigo-100', 'ring-2', 'ring-indigo-400');
  const catHeader = e.target.closest('.cm-cat-header');
  if (catHeader) catHeader.classList.remove('bg-indigo-50');
});

document.getElementById('cm-code-list').addEventListener('drop', async e => {
  e.preventDefault();
  document.querySelectorAll('.cm-code-outer').forEach(o => o.classList.remove('ring-2', 'ring-green-400', 'ring-blue-400', 'rounded-xl'));
  document.querySelectorAll('.cm-drop-zone').forEach(z => z.classList.remove('bg-indigo-100', 'ring-2', 'ring-indigo-400'));
  document.querySelectorAll('.cm-cat-header').forEach(h => h.classList.remove('bg-indigo-50', 'opacity-50'));

  // ── Drop: excerpt → re-assign (move) or copy to another open coding ───────
  if (_cmDragExcerpt !== null) {
    const isCopy = e.ctrlKey;
    const exc    = _cmDragExcerpt;
    _cmDragExcerpt = null;
    _cmDragCopy    = false;
    const outer = e.target.closest('.cm-code-outer');
    if (!outer) return;
    const targetCodeId = parseInt(outer.dataset.outerCodeId);
    if (!targetCodeId || targetCodeId === exc.sourceCodeId) return;
    try {
      if (exc.highlightId !== null) {
        // ── Span highlight drag ──────────────────────────────────────────────
        // Create the highlight under the target code
        await apiFetch(`/api/bookmarks/${exc.bookmarkId}/highlights`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code_id: targetCodeId, highlighted_text: exc.hlText }),
        });
        if (!isCopy) {
          await apiFetch(`/api/bookmarks/${exc.bookmarkId}/highlights/${exc.highlightId}`, { method: 'DELETE' });
        }
        // Update cached bookmark highlights
        const bm = _cachedBookmarks.find(b => b.bookmark_id === exc.bookmarkId);
        if (bm) {
          if (!isCopy) bm.highlights = (bm.highlights || []).filter(h => h.id !== exc.highlightId);
          const tc = _allCodes.find(c => c.id === targetCodeId);
          if (tc) bm.highlights = [...(bm.highlights || []), {
            code_id: targetCodeId, code_name: tc.name, code_color: tc.color,
            highlighted_text: exc.hlText,
          }];
        }
      } else {
        // ── Whole-bookmark coding drag ───────────────────────────────────────
        await apiFetch(`/api/bookmarks/${exc.bookmarkId}/codes/${targetCodeId}`, { method: 'POST' });
        if (!isCopy) {
          await apiFetch(`/api/bookmarks/${exc.bookmarkId}/codes/${exc.sourceCodeId}`, { method: 'DELETE' });
        }
        const bm = _cachedBookmarks.find(b => b.bookmark_id === exc.bookmarkId);
        if (bm) {
          if (!isCopy) bm.codes = (bm.codes || []).filter(c => c.id !== exc.sourceCodeId);
          if (!bm.codes.some(c => c.id === targetCodeId)) {
            const tc = _allCodes.find(c => c.id === targetCodeId);
            if (tc) bm.codes.push({ id: tc.id, name: tc.name, color: tc.color });
          }
        }
      }
      if (!isCopy) delete _cmExcerptsCache[exc.sourceCodeId];
      delete _cmExcerptsCache[targetCodeId];
      if (!isCopy && _cmExpandedCodes.has(exc.sourceCodeId)) _cmFetchExcerptsFor(exc.sourceCodeId);
      if (_cmExpandedCodes.has(targetCodeId)) _cmFetchExcerptsFor(targetCodeId);
      await _cmRefresh();
      document.dispatchEvent(new CustomEvent('codebook-updated'));
    } catch (err) { showErrorPopup((isCopy ? 'Copy' : 'Move') + ' failed: ' + err.message); }
    return;
  }

  const zone      = e.target.closest('.cm-drop-zone');
  const catHeader = e.target.closest('.cm-cat-header');
  const target    = zone || catHeader;
  if (!target) return;

  const rawId       = target.dataset.dropCatId ?? target.dataset.catId;
  const newParentId = rawId ? parseInt(rawId) : null;

  // ── Drop: open coding → assign to a coding level ─────────────────────────
  if (_cmDragCodeId !== null) {
    const codeId = _cmDragCodeId;
    _cmDragCodeId = null;
    const code = _cmCodes.find(c => c.id === codeId);
    if (!code || code.category_id === newParentId) return;
    try {
      await apiFetch(`/api/codes/${codeId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ category_id: newParentId }),
      });
      _allCodes = _allCodes.map(c => c.id === codeId ? { ...c, category_id: newParentId } : c);
      await _cmRefresh();
      document.dispatchEvent(new CustomEvent('codebook-updated'));
    } catch (err) { showErrorPopup('Move failed: ' + err.message); }
    return;
  }

  // ── Drop: coding → re-parent to a higher-order coding ────────────────────
  if (_cmDragCatId !== null) {
    const catId = _cmDragCatId;
    _cmDragCatId = null;
    if (newParentId !== null && _cmGetDescendantCatIds(catId).has(newParentId)) return;
    const cat = _cmCategories.find(c => c.id === catId);
    if (!cat || cat.parent_id === newParentId) return;
    try {
      await apiFetch(`/api/code-categories/${catId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: cat.name, color: cat.color, parent_id: newParentId }),
      });
      await _cmRefresh();
    } catch (err) { showErrorPopup('Move failed: ' + err.message); }
  }
});

// ── Code save / delete ────────────────────────────────────────────────────────

document.getElementById('cm-code-edit-section').addEventListener('keydown', e => {
  if (e.key === 'Enter' && e.target.tagName === 'INPUT') {
    e.preventDefault();
    document.getElementById('cm-edit-save').click();
  }
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
    delete _cmExcerptsCache[_cmOpenCodeId];  // force re-fetch on next expand
    _allCodes = _allCodes.map(c => c.id === _cmOpenCodeId ? { ...c, name, color, description, category_id } : c);
    _cachedBookmarks.forEach(bm => { (bm.codes || []).forEach(c => { if (c.id === _cmOpenCodeId) { c.name = name; c.color = color; } }); });
    renderBmCodeFilterChips();
    await _cmRefresh();
    document.dispatchEvent(new CustomEvent('codebook-updated'));
  } catch (err) { showErrorPopup('Failed to save: ' + err.message); }
});

document.getElementById('cm-edit-delete').addEventListener('click', async () => {
  if (_cmOpenCodeId === null) return;
  const code = _cmCodes.find(c => c.id === _cmOpenCodeId);
  if (!code) return;
  if (!await cmConfirm(`Delete open coding "${code.name}"? It will be removed from all bookmarks.`, 'Delete')) return;
  const deletingId = _cmOpenCodeId;
  try {
    await apiFetch(`/api/codes/${deletingId}`, { method: 'DELETE' });
    _cmCloseDetail();
    _cmExpandedCodes.delete(deletingId);
    delete _cmExcerptsCache[deletingId];
    _allCodes = _allCodes.filter(c => c.id !== deletingId);
    _bmCodeFilter.delete(deletingId);
    _cachedBookmarks.forEach(bm => { bm.codes = (bm.codes || []).filter(c => c.id !== deletingId); });
    renderBmCodeFilterChips();
    await _cmRefresh();
    document.dispatchEvent(new CustomEvent('codebook-updated'));
  } catch (err) { showErrorPopup('Failed to delete: ' + err.message); }
});

// ── Category save / delete ────────────────────────────────────────────────────

document.getElementById('cm-cat-edit-save').addEventListener('click', async () => {
  if (_cmOpenCatId === null) return;
  const name      = document.getElementById('cm-cat-edit-name').value.trim();
  const color     = document.getElementById('cm-cat-edit-color').value;
  const parentVal = document.getElementById('cm-cat-edit-parent').value;
  const parent_id = parentVal ? parseInt(parentVal) : null;
  if (!name) return;
  try {
    await apiFetch(`/api/code-categories/${_cmOpenCatId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, color, parent_id }),
    });
    const msgEl = document.getElementById('cm-cat-edit-msg');
    msgEl.textContent = 'Saved.';
    msgEl.classList.remove('hidden');
    setTimeout(() => msgEl.classList.add('hidden'), 2500);
    await _cmRefresh();
  } catch (err) { showErrorPopup('Failed to save: ' + err.message); }
});

document.getElementById('cm-cat-edit-delete').addEventListener('click', async () => {
  if (_cmOpenCatId === null) return;
  const cat = _cmCategories.find(c => c.id === _cmOpenCatId);
  if (!cat) return;
  if (!await cmConfirm(`Delete coding "${cat.name}"? Child codings will be promoted one level and open codings will become uncategorized.`, 'Delete')) return;
  try {
    await apiFetch(`/api/code-categories/${_cmOpenCatId}`, { method: 'DELETE' });
    _cmCloseDetail();
    await _cmRefresh();
  } catch (err) { showErrorPopup('Failed to delete: ' + err.message); }
});

// ── Detail close ──────────────────────────────────────────────────────────────

document.getElementById('cm-detail-close').addEventListener('click', () => {
  _cmCloseDetail();
  _cmRenderTree();
});

// ── Excerpt panel: conversation context popup ────────────────────────────────

let _cmCtxActiveMsgId  = null;
let _cmCtxActiveSource = '';

async function _cmCtxLoad(msgId, source) {
  const metaEl  = document.getElementById('cm-ctx-modal-meta');
  const bodyEl  = document.getElementById('cm-ctx-modal-body');
  const before  = Math.max(0, parseInt(document.getElementById('cm-ctx-before').value) || 10);
  const after   = Math.max(0, parseInt(document.getElementById('cm-ctx-after').value)  || 10);
  bodyEl.innerHTML = '<p class="text-xs text-gray-400 text-center py-6">Loading…</p>';
  try {
    const msgs = await apiFetch(`/api/context/${msgId}?before=${before}&after=${after}`);
    metaEl.textContent = source
      ? `${source} · ${msgs.length} messages (${before} before · ${after} after)`
      : `${msgs.length} messages (${before} before · ${after} after)`;
    bodyEl.innerHTML = msgs.map(m => ctxMsg(m)).join('');
    // Scroll to the bookmarked (target) message
    requestAnimationFrame(() => {
      const target = bodyEl.querySelector('.ctx-target');
      if (target) target.scrollIntoView({ block: 'center', behavior: 'smooth' });
    });
  } catch (err) {
    bodyEl.innerHTML = `<p class="text-xs text-red-500 text-center py-6">${esc(err.message)}</p>`;
  }
}

document.getElementById('cm-exc-ctx-btn').addEventListener('click', async function () {
  const msgId  = parseInt(this.dataset.msgId);
  const source = this.dataset.source || '';
  if (!msgId) return;
  _cmCtxActiveMsgId  = msgId;
  _cmCtxActiveSource = source;
  document.getElementById('cm-ctx-modal').classList.remove('hidden');
  await _cmCtxLoad(msgId, source);
});

document.getElementById('cm-ctx-reload-btn').addEventListener('click', async () => {
  if (_cmCtxActiveMsgId) await _cmCtxLoad(_cmCtxActiveMsgId, _cmCtxActiveSource);
});

document.getElementById('cm-ctx-modal-close').addEventListener('click', () => {
  document.getElementById('cm-ctx-modal').classList.add('hidden');
});
document.getElementById('cm-ctx-modal').addEventListener('click', e => {
  if (e.target === e.currentTarget) e.currentTarget.classList.add('hidden');
});

// ── New Code panel ────────────────────────────────────────────────────────────

document.getElementById('cm-new-code-btn').addEventListener('click', () => {
  document.getElementById('cm-new-code-panel').classList.toggle('hidden');
  document.getElementById('cm-new-cat-panel').classList.add('hidden');
  _setPickerColor('cm-nc-color', _randomCodeColor());
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
    _setPickerColor('cm-nc-color', _randomCodeColor());
    document.getElementById('cm-nc-category').value = '';
    document.getElementById('cm-new-code-panel').classList.add('hidden');
    _allCodes = [..._allCodes, code].sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }));
    renderBmCodeFilterChips();
    await _cmRefresh();
    document.dispatchEvent(new CustomEvent('codebook-updated'));
  } catch (err) {
    msgEl.textContent = err.message || 'Failed to create code.';
    msgEl.classList.remove('hidden');
  }
});

// ── New Theme panel ────────────────────────────────────────────────────────────

document.getElementById('cm-new-cat-btn').addEventListener('click', () => {
  document.getElementById('cm-new-cat-panel').classList.toggle('hidden');
  document.getElementById('cm-new-code-panel').classList.add('hidden');
  document.getElementById('cm-cat-name').focus();
});
document.getElementById('cm-cat-cancel').addEventListener('click', () => {
  document.getElementById('cm-new-cat-panel').classList.add('hidden');
});
document.getElementById('cm-cat-save').addEventListener('click', async () => {
  const name      = document.getElementById('cm-cat-name').value.trim();
  const color     = document.getElementById('cm-cat-color').value;
  const parentVal = document.getElementById('cm-cat-parent').value;
  const parent_id = parentVal ? parseInt(parentVal) : null;
  const msgEl     = document.getElementById('cm-cat-msg');
  msgEl.classList.add('hidden');
  if (!name) { document.getElementById('cm-cat-name').focus(); return; }
  try {
    await apiFetch('/api/code-categories', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, color, parent_id }),
    });
    document.getElementById('cm-cat-name').value  = '';
    _setPickerColor('cm-cat-color', '#94a3b8');
    document.getElementById('cm-cat-parent').value = '';
    document.getElementById('cm-new-cat-panel').classList.add('hidden');
    await _cmRefresh();
  } catch (err) {
    msgEl.textContent = err.message || 'Failed to create coding.';
    msgEl.classList.remove('hidden');
  }
});

// ── Merge mode ────────────────────────────────────────────────────────────────

document.getElementById('cm-select-mode-toggle').addEventListener('change', e => {
  _cmMergeMode = e.target.checked;
  _cmSelected.clear();
  document.getElementById('cm-merge-btn').classList.add('hidden');
  document.getElementById('cm-merge-cancel').classList.add('hidden');
  _cmRenderTree();
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
  _cmRenderTree();
});

document.getElementById('cm-merge-btn').addEventListener('click', async () => {
  if (_cmSelected.size !== 2) return;
  const [srcId, tgtId] = [..._cmSelected];
  const src = _cmCodes.find(c => c.id === srcId);
  const tgt = _cmCodes.find(c => c.id === tgtId);
  if (!src || !tgt) return;
  if (!await cmConfirm(`Merge "${src.name}" into "${tgt.name}"? "${src.name}" will be deleted and its bookmarks reassigned to "${tgt.name}".`, 'Merge', 'bg-amber-600 hover:bg-amber-700')) return;
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
  } catch (err) { showErrorPopup('Merge failed: ' + err.message); }
});

document.getElementById('cm-refresh-btn').addEventListener('click', _cmRefresh);

document.addEventListener('codebook-updated', () => {
  if (!document.getElementById('page-coding').classList.contains('hidden')) {
    _cmRefresh();
  }
});

// ── Coding page tabs ──────────────────────────────────────────────────────────

let _cmActiveTab = 'manager'; // 'manager' | 'table'

function _cmSwitchTab(tab) {
  _cmActiveTab = tab;
  const isManager = tab === 'manager';
  document.getElementById('cm-section-manager').classList.toggle('hidden', !isManager);
  document.getElementById('cm-section-table').classList.toggle('hidden', isManager);

  const manBtn = document.getElementById('cm-tab-manager');
  const tblBtn = document.getElementById('cm-tab-table');
  const _tabActive   = 'flex-1 py-2 text-xs font-semibold rounded-xl bg-indigo-700 text-white transition-colors';
  const _tabInactive = 'flex-1 py-2 text-xs font-semibold rounded-xl text-gray-600 hover:bg-gray-100 transition-colors';
  manBtn.className = isManager  ? _tabActive : _tabInactive;
  tblBtn.className = !isManager ? _tabActive : _tabInactive;

  if (!isManager) _cmLoadCodingTable();
}

document.getElementById('cm-tab-manager').addEventListener('click', () => _cmSwitchTab('manager'));
document.getElementById('cm-tab-table').addEventListener('click', () => _cmSwitchTab('table'));
document.getElementById('cm-table-reload-btn').addEventListener('click', _cmLoadCodingTable);

// ── Coding filter listeners ───────────────────────────────────────────────────

async function _cmApplyFilter() {
  _cmFilterDateFrom = document.getElementById('cm-filter-month-from').value || '';
  _cmFilterDateTo   = document.getElementById('cm-filter-month-to').value   || '';
  _cmFilterSuno     = document.getElementById('cm-filter-suno').value        || 'all';
  const clearBtn = document.getElementById('cm-filter-clear');
  if (clearBtn) clearBtn.classList.toggle('hidden', !_cmFilterActive());
  if (_cmFilterActive()) {
    _cachedBookmarks = await apiFetch('/api/bookmarks');
  }
  // Re-render both views with updated filter
  _cmRenderTree();
  _cmRenderCodingTable();
}

document.getElementById('cm-filter-month-from').addEventListener('change', _cmApplyFilter);
document.getElementById('cm-filter-month-to').addEventListener('change',   _cmApplyFilter);
document.getElementById('cm-filter-suno').addEventListener('change',        _cmApplyFilter);

document.getElementById('cm-filter-clear').addEventListener('click', () => {
  _cmFilterDateFrom = '';
  _cmFilterDateTo   = '';
  _cmFilterSuno     = 'all';
  document.getElementById('cm-filter-month-from').value = '';
  document.getElementById('cm-filter-month-to').value   = '';
  document.getElementById('cm-filter-suno').value        = 'all';
  document.getElementById('cm-filter-clear').classList.add('hidden');
  _cmRenderTree();
  _cmRenderCodingTable();
});

// ── Coding Manager: right-click "Add higher-level coding" ────────────────────

let _cmCtxTargetType  = null; // 'code' | 'cat'
let _cmCtxTargetId    = null; // int
let _cmCtxNewParentId = null; // parent_id for the new category (null = root)

function _cmOrderLabelFull(n) {
  // n = 2 → "2nd-order", 3 → "3rd-order", etc.
  const s = n === 2 ? '2nd' : n === 3 ? '3rd' : `${n}th`;
  return `${s}-order`;
}

function _cmHideCtxMenu() {
  const menu = document.getElementById('cm-ctx-menu');
  menu.classList.add('hidden');
  document.getElementById('cm-ctx-add-form').classList.add('hidden');
  document.getElementById('cm-ctx-add-err').classList.add('hidden');
  document.getElementById('cm-ctx-add-name').value = '';
  _cmCtxTargetType = null;
  _cmCtxTargetId   = null;
}

document.getElementById('cm-code-list').addEventListener('contextmenu', e => {
  e.preventDefault();

  const card   = e.target.closest('.cm-code-card[data-code-id]');
  const header = e.target.closest('.cm-cat-header[data-cat-id]');
  if (!card && !header) return;

  const menu = document.getElementById('cm-ctx-menu');
  const lbl  = document.getElementById('cm-ctx-add-label');

  if (card) {
    _cmCtxTargetType  = 'code';
    _cmCtxTargetId    = parseInt(card.dataset.codeId);
    _cmCtxNewParentId = null; // new category becomes a root; code re-assigned to it
    lbl.textContent   = 'Add 2nd-order coding above this code';
  } else {
    _cmCtxTargetType = 'cat';
    _cmCtxTargetId   = parseInt(header.dataset.catId);
    const depth      = parseInt(header.dataset.depth) || 0;
    const newOrder   = depth + 3; // depth 0 = 2nd-order → new parent = 3rd-order
    // New category inherits the existing category's current parent (insert in between)
    const existing   = _cmCategories.find(c => c.id === _cmCtxTargetId);
    _cmCtxNewParentId = existing ? existing.parent_id : null;
    lbl.textContent  = `Add ${_cmOrderLabelFull(newOrder)} coding above this category`;
  }

  // Position and show
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  menu.classList.remove('hidden');
  document.getElementById('cm-ctx-add-form').classList.add('hidden');
  const mw = menu.offsetWidth;
  const mh = menu.offsetHeight;
  menu.style.left = Math.min(e.clientX, vw - mw - 8) + 'px';
  menu.style.top  = Math.min(e.clientY, vh - mh - 8) + 'px';
});

document.getElementById('cm-ctx-add-higher').addEventListener('click', () => {
  document.getElementById('cm-ctx-add-form').classList.remove('hidden');
  document.getElementById('cm-ctx-add-name').focus();
});

document.getElementById('cm-ctx-add-cancel').addEventListener('click', _cmHideCtxMenu);

document.getElementById('cm-ctx-add-confirm').addEventListener('click', async () => {
  const name  = document.getElementById('cm-ctx-add-name').value.trim();
  const color = document.getElementById('cm-ctx-add-color').value;
  const errEl = document.getElementById('cm-ctx-add-err');
  errEl.classList.add('hidden');
  if (!name) { errEl.textContent = 'Name is required.'; errEl.classList.remove('hidden'); return; }
  if (!_cmCtxTargetId) return;

  try {
    const newCat = await apiFetch('/api/code-categories', {
      method: 'POST',
      body: JSON.stringify({ name, color, parent_id: _cmCtxNewParentId }),
    });

    if (_cmCtxTargetType === 'code') {
      await apiFetch(`/api/codes/${_cmCtxTargetId}`, {
        method: 'PUT',
        body: JSON.stringify({ category_id: newCat.id }),
      });
    } else {
      const existing = _cmCategories.find(c => c.id === _cmCtxTargetId);
      await apiFetch(`/api/code-categories/${_cmCtxTargetId}`, {
        method: 'PUT',
        body: JSON.stringify({ name: existing.name, color: existing.color, parent_id: newCat.id }),
      });
    }

    _cmHideCtxMenu();
    await _cmRefresh();
  } catch (err) {
    errEl.textContent = err.message || 'Failed to create category.';
    errEl.classList.remove('hidden');
  }
});

// Enter key in name input triggers confirm
document.getElementById('cm-ctx-add-name').addEventListener('keydown', e => {
  if (e.key === 'Enter') document.getElementById('cm-ctx-add-confirm').click();
  if (e.key === 'Escape') _cmHideCtxMenu();
});

// Click outside the menu dismisses it
document.addEventListener('click', e => {
  const menu = document.getElementById('cm-ctx-menu');
  if (!menu.classList.contains('hidden') && !menu.contains(e.target)) _cmHideCtxMenu();
});

// ── Coding Table: load ────────────────────────────────────────────────────────

async function _cmLoadCodingTable() {
  const container = document.getElementById('cm-coding-table-container');
  container.innerHTML = '<p class="text-sm text-gray-400 text-center py-8">Loading…</p>';
  // Ensure codes & categories are loaded
  if (!_cmCodes.length && !_cmCategories.length) await _cmRefresh();
  // Ensure bookmarks are loaded (needed for excerpts)
  if (!_cachedBookmarks.length) {
    try { _cachedBookmarks = await apiFetch('/api/bookmarks'); }
    catch (err) {
      container.innerHTML = `<p class="text-sm text-red-500 text-center py-8">Failed to load bookmarks: ${esc(err.message)}</p>`;
      return;
    }
  }
  _cmRenderCodingTable();
}

// ── Coding Table: render ──────────────────────────────────────────────────────

function _cmGetCatPath(catId) {
  const path = [];
  let cur = _cmCategories.find(c => c.id === catId);
  while (cur) {
    path.unshift(cur);
    cur = cur.parent_id ? _cmCategories.find(c => c.id === cur.parent_id) : null;
  }
  return path;
}

function _cmRenderCodingTable() {
  _cmComputeMaxDepth();
  const container = document.getElementById('cm-coding-table-container');

  const excByCode = {};
  (_cachedBookmarks || []).forEach(bm => {
    // Apply coding filter to each bookmark
    if (_cmFilterSuno === 'only'    && !truthy(bm.is_suno_team)) return;
    if (_cmFilterSuno === 'exclude' &&  truthy(bm.is_suno_team)) return;
    if (_cmFilterDateFrom || _cmFilterDateTo) {
      const d = (bm.date || '').substring(0, 10);
      if (_cmFilterDateFrom && d < _cmFilterDateFrom) return;
      if (_cmFilterDateTo   && d > _cmFilterDateTo)   return;
    }
    const meta = { username: bm.username || '', date: (bm.date || '').substring(0, 10),
                   note: bm.note || '', bookmarkId: bm.bookmark_id };
    // Whole-bookmark codings — show full message content
    (bm.codes || []).forEach(code => {
      (excByCode[code.id] = excByCode[code.id] || []).push({ ...meta, content: bm.content || '' });
    });
    // Span-level highlight codings — merge multiple highlights of same code per bookmark
    const hlByCode = {};
    (bm.highlights || []).forEach(hl => {
      if (!hlByCode[hl.code_id]) hlByCode[hl.code_id] = [];
      hlByCode[hl.code_id].push(hl.highlighted_text || '');
    });
    Object.entries(hlByCode).forEach(([codeId, texts]) => {
      (excByCode[parseInt(codeId)] = excByCode[parseInt(codeId)] || [])
        .push({ ...meta, content: texts.join(_BM_SEG_SEP) });
    });
  });

  if (_cmTreeMaxDepth <= 1) {
    _cmRenderClassicTable(container, excByCode);
  } else {
    _cmRenderHierarchyTable(container, excByCode);
  }
}

// Classic fixed-column table: used when ≤ 3 order levels exist
// maxDepth=0 → [2nd-order | Open Coding | Quote | Source | Note]
// maxDepth=1 → [3rd-order | 2nd-order | Open Coding | Quote | Source | Note]
function _cmRenderClassicTable(container, excByCode) {
  const roots = _cmBuildTree();
  const uncategorized = _cmCodes
    .filter(c => c.category_id == null)
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }));

  // Build a flat list of rows: each row carries the category cells + one excerpt (or null)
  const flatRows = [];

  function addCodeRows(code, cat3, cat2) {
    const excs = excByCode[code.id] || [];
    if (excs.length === 0) {
      if (_cmFilterActive()) return; // hide codes with no matching quotes when filter is on
      flatRows.push({ cat3, cat2, code, exc: null });
    } else {
      excs.forEach(exc => flatRows.push({ cat3, cat2, code, exc }));
    }
  }

  if (_cmTreeMaxDepth === 0) {
    roots.forEach(cat2Node => {
      cat2Node.codes.forEach(code => addCodeRows(code, null, cat2Node));
    });
    uncategorized.forEach(code => addCodeRows(code, null, null));
  } else { // maxDepth === 1
    roots.forEach(cat3Node => {
      cat3Node.children.forEach(cat2Node => {
        cat2Node.codes.forEach(code => addCodeRows(code, cat3Node, cat2Node));
      });
      cat3Node.codes.forEach(code => addCodeRows(code, cat3Node, null));
    });
    uncategorized.forEach(code => addCodeRows(code, null, null));
  }

  if (!flatRows.length) {
    container.innerHTML = '<p class="text-sm text-gray-400 text-center py-8">No codes yet.</p>';
    return;
  }

  // For each position, compute rowspan (> 0 = first in group → render cell; 0 = skip)
  function computeSpans(keyFn) {
    const n = flatRows.length;
    const result = new Array(n).fill(0);
    let gs = 0;
    for (let i = 1; i <= n; i++) {
      if (i === n || keyFn(flatRows[i]) !== keyFn(flatRows[gs])) {
        result[gs] = i - gs;
        gs = i;
      }
    }
    return result;
  }

  const cat3Spans = _cmTreeMaxDepth === 1
    ? computeSpans(r => r.cat3 ? r.cat3.id : '__Z__')
    : null;
  // cat2 key includes cat3 so groups reset when cat3 changes
  const cat2Spans = computeSpans(r =>
    (r.cat3 ? r.cat3.id : '__X__') + '|' + (r.cat2 ? r.cat2.id : '__Y__')
  );
  const codeSpans = computeSpans(r => r.code.id);

  let tbody = '';
  flatRows.forEach((row, i) => {
    const { cat3, cat2, code, exc } = row;
    tbody += '<tr class="border-b border-gray-100 hover:bg-gray-50">';

    // 3rd-order column (maxDepth=1 only)
    if (_cmTreeMaxDepth === 1 && cat3Spans[i] > 0) {
      if (cat3) {
        tbody += `<td class="px-3 py-2 align-top border-r border-gray-100" rowspan="${cat3Spans[i]}"
            style="border-left:3px solid ${cat3.color}">
            <div class="flex items-center gap-1.5 flex-wrap">
              <span class="w-2.5 h-2.5 rounded-full shrink-0" style="background:${cat3.color}"></span>
              <span class="text-xs font-semibold text-gray-800">${esc(cat3.name)}</span>
            </div></td>`;
      } else {
        tbody += `<td class="px-3 py-2 align-top border-r border-gray-100 text-[10px] italic text-gray-400"
            rowspan="${cat3Spans[i]}">—</td>`;
      }
    }

    // 2nd-order column
    if (cat2Spans[i] > 0) {
      if (cat2) {
        tbody += `<td class="px-3 py-2 align-top border-r border-gray-100" rowspan="${cat2Spans[i]}"
            style="border-left:3px solid ${cat2.color}">
            <div class="flex items-center gap-1.5 flex-wrap">
              <span class="w-2.5 h-2.5 rounded-full shrink-0" style="background:${cat2.color}"></span>
              <span class="text-xs font-semibold text-gray-800">${esc(cat2.name)}</span>
            </div></td>`;
      } else {
        tbody += `<td class="px-3 py-2 align-top border-r border-gray-100 text-[10px] italic text-gray-400"
            rowspan="${cat2Spans[i]}">—</td>`;
      }
    }

    // Open Coding column
    if (codeSpans[i] > 0) {
      const tc = labelTextColor(code.color);
      tbody += `<td class="px-3 py-2 align-top border-r border-gray-100" rowspan="${codeSpans[i]}">
          <div class="flex items-center gap-1.5 flex-wrap">
            <span class="w-2.5 h-2.5 rounded-full shrink-0" style="background:${code.color}"></span>
            <span class="text-xs font-semibold text-gray-800">${esc(code.name)}</span>
            <span class="text-[10px] px-1 py-0.5 rounded-full" style="background:${code.color};color:${tc}">${code.groundedness}</span>
          </div>
          ${code.description ? `<p class="text-[10px] text-gray-500 mt-0.5 ml-4 italic">${esc(code.description)}</p>` : ''}
          <div class="mt-1 ml-4 text-[10px] text-gray-400">G:${code.groundedness} D:${code.density}</div>
        </td>`;
    }

    // Excerpt columns
    if (exc) {
      const snippet = exc.content.substring(0, 220);
      const more    = exc.content.length > 220 ? '…' : '';
      tbody += `<td class="px-3 py-2 text-xs italic text-gray-700 leading-relaxed max-w-xs">"${esc(snippet)}${more}"</td>
        <td class="px-3 py-2 text-[10px] text-gray-500 whitespace-nowrap align-top">${esc(exc.username)}<br>${esc(exc.date)}</td>
        <td class="px-3 py-2 text-[10px] text-gray-400 align-top">${esc(exc.note) || '—'}</td>`;
    } else {
      tbody += `<td colspan="3" class="px-3 py-2 text-[10px] italic text-gray-300">No excerpts yet.</td>`;
    }
    tbody += '</tr>';
  });

  const orderHdrs = _cmTreeMaxDepth === 1
    ? `<th class="px-3 py-2 text-left text-xs font-semibold text-gray-600 w-40">3rd-order Coding</th>
       <th class="px-3 py-2 text-left text-xs font-semibold text-gray-600 w-40">2nd-order Coding</th>`
    : `<th class="px-3 py-2 text-left text-xs font-semibold text-gray-600 w-40">2nd-order Coding</th>`;

  container.innerHTML = `
    <table class="w-full text-sm border-collapse">
      <thead>
        <tr class="bg-gray-50 border-b-2 border-gray-200">
          ${orderHdrs}
          <th class="px-3 py-2 text-left text-xs font-semibold text-gray-600 w-44">Open Coding</th>
          <th class="px-3 py-2 text-left text-xs font-semibold text-gray-600">Quote / Excerpt</th>
          <th class="px-3 py-2 text-left text-xs font-semibold text-gray-600 w-28">Source</th>
          <th class="px-3 py-2 text-left text-xs font-semibold text-gray-600 w-28">Note</th>
        </tr>
      </thead>
      <tbody>${tbody}</tbody>
    </table>`;
}

// Hierarchy table: used when 4+ order levels exist (maxDepth >= 2)
// Category header rows span all columns; colspan fixed to 4.
function _cmRenderHierarchyTable(container, excByCode) {
  const roots        = _cmBuildTree();
  const uncategorized = _cmCodes
    .filter(c => c.category_id == null)
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }));

  function catHeaderRow(node, depth) {
    const indent = 8 + depth * 16;
    return `<tr style="background:${node.color}18; border-left:3px solid ${node.color}">
      <td colspan="4" class="px-3 py-2 text-xs font-bold text-gray-700 uppercase tracking-wide"
          style="padding-left:${indent}px">
        <span class="inline-flex items-center gap-2">
          <span class="w-2.5 h-2.5 rounded-full" style="background:${node.color}"></span>
          <span class="font-mono text-[9px] border px-1 rounded bg-white/70 text-gray-500">${_cmOrderLabel(depth)}</span>
          ${esc(node.name)}
        </span>
      </td>
    </tr>`;
  }

  function codeRows(code) {
    const tc   = labelTextColor(code.color);
    const excs = excByCode[code.id] || [];

    const codeCellHtml = rowspan => `
      <td class="px-3 py-2 align-top border-r border-gray-100" rowspan="${rowspan}">
        <div class="flex items-center gap-1.5 flex-wrap">
          <span class="w-2.5 h-2.5 rounded-full shrink-0" style="background:${code.color}"></span>
          <span class="text-xs font-semibold text-gray-800">${esc(code.name)}</span>
          <span class="text-[10px] px-1 py-0.5 rounded-full" style="background:${code.color};color:${tc}">${code.groundedness}</span>
        </div>
        ${code.description ? `<p class="text-[10px] text-gray-500 mt-0.5 ml-4 italic">${esc(code.description)}</p>` : ''}
        <div class="mt-1 ml-4 text-[10px] text-gray-400">G:${code.groundedness} D:${code.density}</div>
      </td>`;

    if (!excs.length) {
      if (_cmFilterActive()) return ''; // hide codes with no matching quotes when filter is on
      return `<tr class="border-b border-gray-100 hover:bg-gray-50">
        ${codeCellHtml(1)}
        <td colspan="3" class="px-3 py-2 text-[10px] italic text-gray-300">No excerpts yet.</td>
      </tr>`;
    }

    return excs.map((exc, i) => {
      const snippet = exc.content.substring(0, 220);
      const more    = exc.content.length > 220 ? '…' : '';
      return `<tr class="border-b border-gray-100 hover:bg-gray-50">
        ${i === 0 ? codeCellHtml(excs.length) : ''}
        <td class="px-3 py-2 text-xs italic text-gray-700 leading-relaxed max-w-xs">"${esc(snippet)}${more}"</td>
        <td class="px-3 py-2 text-[10px] text-gray-500 whitespace-nowrap align-top">${esc(exc.username)}<br>${esc(exc.date)}</td>
        <td class="px-3 py-2 text-[10px] text-gray-400 align-top">${esc(exc.note) || '—'}</td>
      </tr>`;
    }).join('');
  }

  let tbody = '';
  function walkNode(node, depth) {
    // When filter active, collect children/codes into a buffer first so we can
    // skip the category header entirely if nothing inside is visible.
    if (_cmFilterActive()) {
      const savedTbody = tbody;
      tbody = '';
      node.children.forEach(ch => walkNode(ch, depth + 1));
      node.codes.forEach(code => { tbody += codeRows(code); });
      const nodeBody = tbody;
      tbody = savedTbody;
      if (nodeBody) {
        tbody += catHeaderRow(node, depth);
        tbody += nodeBody;
      }
    } else {
      tbody += catHeaderRow(node, depth);
      node.children.forEach(ch => walkNode(ch, depth + 1));
      node.codes.forEach(code => { tbody += codeRows(code); });
    }
  }
  roots.forEach(node => walkNode(node, 0));

  const visibleUncategorized = _cmFilterActive()
    ? uncategorized.filter(c => (excByCode[c.id] || []).length > 0)
    : uncategorized;

  if (visibleUncategorized.length) {
    tbody += `<tr class="bg-gray-50"><td colspan="4" class="px-3 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wide">
      Uncategorized Open Codings</td></tr>`;
    visibleUncategorized.forEach(code => { tbody += codeRows(code); });
  }

  if (!tbody) {
    container.innerHTML = '<p class="text-sm text-gray-400 text-center py-8">No codes yet.</p>';
    return;
  }

  container.innerHTML = `
    <table class="w-full text-sm border-collapse">
      <thead>
        <tr class="bg-gray-50 border-b-2 border-gray-200">
          <th class="px-3 py-2 text-left text-xs font-semibold text-gray-600 w-56">Open Coding</th>
          <th class="px-3 py-2 text-left text-xs font-semibold text-gray-600">Quote / Excerpt</th>
          <th class="px-3 py-2 text-left text-xs font-semibold text-gray-600 w-28">Source</th>
          <th class="px-3 py-2 text-left text-xs font-semibold text-gray-600 w-28">Note</th>
        </tr>
      </thead>
      <tbody>${tbody}</tbody>
    </table>`;
}

// ── Coding Table: CSV export ──────────────────────────────────────────────────

document.getElementById('cm-table-export-btn').addEventListener('click', () => {
  const excByCode = {};
  (_cachedBookmarks || []).forEach(bm => {
    // Apply the same filter as the coding table view
    if (_cmFilterSuno === 'only'    && !truthy(bm.is_suno_team)) return;
    if (_cmFilterSuno === 'exclude' &&  truthy(bm.is_suno_team)) return;
    if (_cmFilterDateFrom || _cmFilterDateTo) {
      const d = (bm.date || '').substring(0, 7);
      if (_cmFilterDateFrom && d < _cmFilterDateFrom) return;
      if (_cmFilterDateTo   && d > _cmFilterDateTo)   return;
    }
    const meta = { username: bm.username || '', date: (bm.date || '').substring(0, 10), note: bm.note || '' };
    (bm.codes || []).forEach(code => {
      (excByCode[code.id] = excByCode[code.id] || []).push({ ...meta, content: bm.content || '' });
    });
    const hlByCodeCsv = {};
    (bm.highlights || []).forEach(hl => {
      if (!hlByCodeCsv[hl.code_id]) hlByCodeCsv[hl.code_id] = [];
      hlByCodeCsv[hl.code_id].push(hl.highlighted_text || '');
    });
    Object.entries(hlByCodeCsv).forEach(([codeId, texts]) => {
      (excByCode[parseInt(codeId)] = excByCode[parseInt(codeId)] || [])
        .push({ ...meta, content: texts.join(_BM_SEG_SEP) });
    });
  });

  const esc2 = v => '"' + String(v).replace(/"/g, '""') + '"';
  const rows = [['Coding Path', 'Open Coding', 'Memo', 'Quote', 'Source', 'Date', 'Note', 'Groundedness', 'Density']];

  const roots = _cmBuildTree();
  const uncategorized = _cmCodes.filter(c => c.category_id == null);

  function csvCode(code) {
    const excs = excByCode[code.id] || [];
    if (_cmFilterActive() && excs.length === 0) return; // hide codes with no matching quotes under filter
    const path = _cmGetCatPath(code.category_id).map(c => c.name).join(' > ');
    (excs.length ? excs : [null]).forEach(exc => {
      rows.push([
        path, code.name, code.description || '',
        exc ? exc.content : '', exc ? exc.username : '', exc ? exc.date : '', exc ? exc.note : '',
        code.groundedness, code.density,
      ]);
    });
  }
  function csvNode(node) {
    node.children.forEach(csvNode);
    node.codes.forEach(csvCode);
  }
  roots.forEach(csvNode);
  uncategorized.forEach(csvCode);

  const csv  = rows.map(r => r.map(esc2).join(',')).join('\n');
  const blob = new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8;' });
  const url  = URL.createObjectURL(blob);
  const a    = Object.assign(document.createElement('a'), { href: url, download: 'coding_table.csv' });
  a.click();
  URL.revokeObjectURL(url);
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

  _usersData    = [];
  currentResults = [];
  document.getElementById('users-results').classList.add('hidden');
  document.getElementById('results-section').classList.add('hidden');
  document.getElementById('results-container').innerHTML = '';
  document.getElementById('sr-section').classList.add('hidden');
  const _trendSec = document.getElementById('trend-section');
  if (_trendSec) _trendSec.classList.add('hidden');
  const _tabbar2 = document.getElementById('sr-viz-tabbar');
  if (_tabbar2) _tabbar2.classList.add('hidden');
  if (_trendChart) { _trendChart.destroy(); _trendChart = null; }
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
      : String(u.weeks_with_messages ?? 'â');
    const pctDisp    = u.pct_weeks_active != null ? `${u.pct_weeks_active}%` : 'â';
    const avgWords   = u.avg_word_count != null ? Number(u.avg_word_count).toFixed(1) : 'â';

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
      <td class="users-td">${u.first_message_date || 'â'}</td>
      <td class="users-td">${u.last_message_date || 'â'}</td>
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

// Live filters month pickers fire 'change'; number/text inputs fire 'input'
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

let _upoUsername       = '';
let _profileMessages   = [];
let _upoSummaryText    = '';
let _upoFollowUpHistory = [];

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
      : String(stats.weeks_with_messages ?? 'â');
    const pctStr = stats.pct_weeks_active != null ? `${stats.pct_weeks_active}%` : '—';
    const avgStr = stats.avg_word_count != null ? Number(stats.avg_word_count).toFixed(1) : '—';

    document.getElementById('upo-stats').innerHTML =
      `<div class="flex flex-wrap">` +
      _statPill('Total Messages', (stats.total_messages || 0).toLocaleString()) +
      _statPill('First Message',  stats.first_message_date || 'â') +
      _statPill('Last Message',   stats.last_message_date  || 'â') +
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
  document.getElementById('upo-fu-section').classList.add('hidden');
  document.getElementById('upo-fu-history').innerHTML = '';
  _upoSummaryText    = '';
  _upoFollowUpHistory = [];
}

async function _fetchProfileMessages() {
  const msgEl    = document.getElementById('upo-messages');
  const dateFrom  = document.getElementById('upo-date-from').value;
  const dateTo    = document.getElementById('upo-date-to').value;
  const keyword   = document.getElementById('upo-keyword').value.trim();
  const minWords  = parseInt(document.getElementById('upo-min-words').value, 10) || 0;
  const filterEl  = document.getElementById('upo-filter-count');

  msgEl.innerHTML = '<p class="text-sm text-gray-400 py-6 text-center">Loading...</p>';
  filterEl.textContent = '';

  try {
    const params = new URLSearchParams({ username: _upoUsername });
    if (dateFrom)    params.set('date_from', dateFrom);
    if (dateTo)      params.set('date_to', dateTo);
    if (keyword)     params.set('keyword', keyword);
    if (minWords > 0) params.set('min_words', minWords);
    const scope = getScopeParam();
    if (scope) params.set('upload_ids', scope);

    const msgs = await apiFetch(`/api/search/user-messages?${params}`);

    document.getElementById('upo-msg-count').textContent =
      `${msgs.length.toLocaleString()} message${msgs.length !== 1 ? 's' : ''}`;
    filterEl.textContent = keyword || dateFrom || dateTo || minWords > 0
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
      card.id        = `upo-card-${msg.id}`;
      card.className = 'bg-white rounded-xl border border-gray-200 shadow-sm p-3';
      const safeContent  = keyword ? highlight(msg.content || '', keyword) : esc(msg.content || '');
      const isBookmarked = bookmarkedIds.has(msg.id);
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
        <div class="border-t border-gray-100 mt-2 pt-1.5 flex items-center justify-between">
          <button class="upo-bm-btn flex items-center gap-1 text-xs font-medium rounded
                         focus-visible:outline focus-visible:outline-2 focus-visible:outline-amber-500"
                  data-id="${msg.id}"
                  title="${isBookmarked ? 'Remove bookmark' : 'Save bookmark'}">
            ${isBookmarked
              ? `<svg class="w-3.5 h-3.5 text-amber-500" fill="currentColor" viewBox="0 0 24 24"><path d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"/></svg><span class="text-amber-600">Bookmarked</span>`
              : `<svg class="w-3.5 h-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"/></svg><span class="text-gray-500">Bookmark</span>`}
          </button>
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
  document.getElementById('upo-sum-toggle').textContent = hidden ? '¦ Summarize' : '¦ Hide Summary';
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
  const hide  = btn.textContent.startsWith('▲');
  logEl.classList.toggle('hidden', hide);
  btn.textContent = hide ? '▼ Show' : '▲ Hide';
});

function _upoSumLog(step, label, msg) {
  const icons = { input:'📋', context:'📡', llm:'✨', fallback:'⚠️' };
  const div = document.createElement('div');
  div.className = 'text-xs text-gray-600 flex items-start gap-1.5 py-0.5';
  div.innerHTML = `<span class="shrink-0">${icons[step] || '•'}</span>
    <span><strong>${esc(label)}</strong> ${esc(msg)}</span>`;
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
    _upoSumLog('context', 'Context fetch', `Done ${Object.keys(contextMap).length} messages enriched`);
  } catch (e) {
    _upoSumLog('fallback', 'Context fetch failed', `${e.message} proceeding without context`);
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
  const header = `USER PROFILE ANALYSIS ${_upoUsername} (${n} messages with context)`;

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
      _upoSummaryText     = output;
      _upoFollowUpHistory = [];
      document.getElementById('upo-fu-history').innerHTML = '';
      document.getElementById('upo-fu-section').classList.remove('hidden');
      document.getElementById('upo-sum-export-pdf').classList.remove('hidden');
    }
  }
}

/* -- User profile follow-up Q&A ------------------------------------------- */

function _appendUpoUserBubble(text) {
  const c = document.getElementById('upo-fu-history');
  const w = document.createElement('div'); w.className = 'flex justify-end';
  const b = document.createElement('div'); b.className = 'chat-bubble-user';
  b.textContent = text;
  w.appendChild(b); c.appendChild(w);
  c.scrollTop = c.scrollHeight;
}

function _appendUpoAssistantBubble() {
  const c = document.getElementById('upo-fu-history');
  const w = document.createElement('div'); w.className = 'flex justify-start';
  const b = document.createElement('div'); b.className = 'chat-bubble-assistant markdown-body';
  w.appendChild(b); c.appendChild(w);
  c.scrollTop = c.scrollHeight;
  return b;
}

function _appendUpoLogStrip() {
  const c     = document.getElementById('upo-fu-history');
  const strip = document.createElement('div'); strip.className = 'fu-log-strip';
  c.appendChild(strip); c.scrollTop = c.scrollHeight;
  return strip;
}

function _renderUpoFuLogEntry(strip, entry) {
  const div = document.createElement('div');
  div.className = `fu-log-entry fu-log-step-${entry.step || 'fallback'}`;
  div.innerHTML =
    `<span class="fu-log-icon">${LOG_ICONS[entry.step] || '•'}</span>` +
    `<span class="fu-log-label">${esc(entry.label || '')}</span>` +
    `<span class="fu-log-msg">${esc(entry.msg || '')}</span>`;
  strip.appendChild(div);
  document.getElementById('upo-fu-history').scrollTop = 9999;
}

async function sendUpoFollowUp() {
  const input    = document.getElementById('upo-fu-input');
  const sendBtn  = document.getElementById('upo-fu-send');
  const question = input.value.trim();
  if (!question || !_upoSummaryText) return;

  const model = document.getElementById('upo-sum-model').value;
  input.value         = '';
  input.disabled      = true;
  sendBtn.disabled    = true;
  sendBtn.textContent = '...';

  _upoFollowUpHistory.push({ role: 'user', content: question });
  _appendUpoUserBubble(question);
  const strip  = _appendUpoLogStrip();
  const bubble = _appendUpoAssistantBubble();
  let answerText = '';

  try {
    const history = [
      { role: 'assistant', content: _upoSummaryText },
      ..._upoFollowUpHistory.slice(0, -1),
    ];
    const res = await fetch('/api/summarize-results/followup', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ question, history, model }),
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
            _renderUpoFuLogEntry(strip, delta);
          } else if (delta.content) {
            answerText += delta.content;
            bubble.innerHTML = marked.parse(answerText);
            document.getElementById('upo-fu-history').scrollTop = 9999;
          } else if (delta.error) {
            throw new Error(delta.error);
          }
        } catch (parseErr) {
          if (!(parseErr instanceof SyntaxError)) throw parseErr;
        }
      }
    }
    _upoFollowUpHistory.push({ role: 'assistant', content: answerText });

  } catch (e) {
    bubble.remove();
    _upoFollowUpHistory.pop();
    showErrorPopup(e.message);
  } finally {
    input.disabled      = false;
    sendBtn.disabled    = false;
    sendBtn.textContent = 'Ask';
    input.focus();
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
document.getElementById('upo-fu-send').addEventListener('click', sendUpoFollowUp);
document.getElementById('upo-fu-clear').addEventListener('click', () => {
  _upoFollowUpHistory = [];
  document.getElementById('upo-fu-history').innerHTML = '';
});
document.getElementById('upo-fu-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) sendUpoFollowUp();
});

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
  const ctx = e.target.closest('.upo-ctx-btn');
  if (ctx) { upoToggleContext(parseInt(ctx.dataset.id, 10), ctx); return; }
});

document.getElementById('upo-messages').addEventListener('click', async e => {
  const btn = e.target.closest('.upo-bm-btn');
  if (!btn) return;
  const msgId = parseInt(btn.dataset.id, 10);
  btn.disabled = true;

  try {
    if (bookmarkedIds.has(msgId)) {
      await fetch(`/api/bookmarks/by-msg/${msgId}`, { method: 'DELETE' });
      bookmarkedIds.delete(msgId);
    } else {
      await fetch('/api/bookmarks', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ msg_id: msgId }),
      });
      bookmarkedIds.add(msgId);
    }
    updateBmBadge();

    const isNow = bookmarkedIds.has(msgId);
    btn.title     = isNow ? 'Remove bookmark' : 'Save bookmark';
    btn.innerHTML = isNow
      ? `<svg class="w-3.5 h-3.5 text-amber-500" fill="currentColor" viewBox="0 0 24 24"><path d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"/></svg><span class="text-amber-600">Bookmarked</span>`
      : `<svg class="w-3.5 h-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"/></svg><span class="text-gray-500">Bookmark</span>`;
  } catch (err) {
    showErrorPopup(err.message || 'Bookmark failed');
  } finally {
    btn.disabled = false;
  }
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
        `${d.total_uploads} datasets &bull; ` +
        `${d.embedded_messages.toLocaleString()} embedded ` +
        (keySet ? ' &bull; <span style="color:#86efac">API key set</span>' : '');
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
        // Stored key is invalid/rejected clear it and prompt again
        localStorage.removeItem(STORAGE_KEY);
        showApiKeyPopup(false);
      }
    } else {
      showApiKeyPopup(false);
    }
  }
})();
