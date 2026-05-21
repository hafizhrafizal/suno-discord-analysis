
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
            ? '<span class="text-xs text-gray-400 italic">Ã¢</span>'
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

