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
   HTTPâ†’HTTPS redirects) produce a clear error instead of "Unexpected token '<'".
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
            (isLocal && cnt === 0 ? ' Weights will download on first use (~0.4""1.3 GB).' : '');
          msg.classList.remove('hidden');
          loadStats();
          loadModelOptions();
        } catch (e) { showErrorPopup('Failed to switch model: ' + e.message); }
      });
    });
  } catch (_) {}
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

// â"€â"€ Color swatch picker helpers â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
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
  // Click outside â†’ close all palettes
  if (!e.target.closest('.color-picker-wrap')) {
    document.querySelectorAll('.color-swatch-palette').forEach(p => p.classList.add('hidden'));
  }
});

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
