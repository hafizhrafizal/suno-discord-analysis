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
  // Apply sub-tab from URL hash (#coding/table or #coding/manager)
  const sub = (location.hash.match(/^#coding\/(\w+)/) || [])[1];
  if (sub === 'table' || sub === 'manager') _cmSwitchTab(sub, { updateHash: false });
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
        <span class="text-[10px] text-gray-400 shrink-0 mt-0.5 select-none">${isExpanded ? 'â–²' : '▼'}</span>
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
    if (codesLbl) codesLbl.textContent = 'View Context';
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
    // Auto-expand the destination code so the moved quote is immediately visible
    _cmExpandedCodes.add(codeId);
    _cmCloseDetail();
    // Re-render tree so filter visibility and expand states update
    _cmRenderTree();
    // Fetch fresh excerpts for both affected codes
    _cmFetchExcerptsFor(oldCodeId);
    _cmFetchExcerptsFor(codeId);
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

document.getElementById('cm-ctx-delete-coding').addEventListener('click', async () => {
  if (!_cmCtxMenuData) return;
  const { bookmarkId, sourceCodeId, hlId } = _cmCtxMenuData;
  _cmCtxMenuData = null;
  const bm = _cachedBookmarks.find(b => b.bookmark_id === bookmarkId);
  try {
    if (hlId !== null) {
      // Span coding: delete the specific highlight
      await apiFetch(`/api/bookmarks/${bookmarkId}/highlights/${hlId}`, { method: 'DELETE' });
      if (bm) bm.highlights = (bm.highlights || []).filter(h => h.highlight_id !== hlId);
    } else {
      // Whole-bookmark coding: remove the code assignment
      await apiFetch(`/api/bookmarks/${bookmarkId}/codes/${sourceCodeId}`, { method: 'DELETE' });
      if (bm) bm.codes = (bm.codes || []).filter(c => c.id !== sourceCodeId);
    }
  } catch (err) {
    showErrorPopup('Failed to delete coding: ' + err.message);
    return;
  }
  delete _cmExcerptsCache[sourceCodeId];
  if (_cmOpenExcBmId === bookmarkId) _cmCloseDetail();
  _cmRenderTree();
  _cmFetchExcerptsFor(sourceCodeId);
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

document.getElementById('cm-collapse-all-btn').addEventListener('click', () => {
  const btn = document.getElementById('cm-collapse-all-btn');
  const isExpanded = btn.dataset.state !== 'collapsed';

  if (isExpanded) {
    // Collapse everything
    _cmExpandedCodes.clear();
    _cmCollapsed.clear();
    _cmCategories.forEach(cat => _cmCollapsed.add(cat.id));
    _cmCloseDetail();
    _cmRenderTree();
    btn.textContent = 'Expand all';
    btn.dataset.state = 'collapsed';
  } else {
    // Expand all categories and all code quote lists
    _cmCollapsed.clear();
    _cmCodes.forEach(c => _cmExpandedCodes.add(c.id));
    _cmRenderTree();
    // Fetch excerpts for any codes not yet cached
    _cmCodes.forEach(c => {
      if (_cmExcerptsCache[c.id] === undefined) _cmFetchExcerptsFor(c.id);
    });
    btn.textContent = 'Collapse all';
    btn.dataset.state = 'expanded';
  }
});

document.addEventListener('codebook-updated', () => {
  if (!document.getElementById('page-coding').classList.contains('hidden')) {
    _cmRefresh();
  }
});

// ── Coding page tabs ──────────────────────────────────────────────────────────

let _cmActiveTab = 'manager'; // 'manager' | 'table'

function _cmSwitchTab(tab, { updateHash = true } = {}) {
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
  if (updateHash) history.pushState(null, '', location.pathname + '#coding/' + tab);
}

document.getElementById('cm-tab-manager').addEventListener('click', () => _cmSwitchTab('manager'));
document.getElementById('cm-tab-table').addEventListener('click', () => _cmSwitchTab('table'));
document.getElementById('cm-table-reload-btn').addEventListener('click', _cmLoadCodingTable);

// ── Coding filter listeners ───────────────────────────────────────────────────

async function _cmApplyFilter() {
  _cmFilterDateFrom = document.getElementById('cm-filter-month-from').value || '';
  _cmFilterDateTo   = document.getElementById('cm-filter-month-to').value   || '';
  // _cmFilterSuno is set directly by suno button clicks
  const clearBtn = document.getElementById('cm-filter-clear');
  if (clearBtn) clearBtn.classList.toggle('hidden', !_cmFilterActive());
  if (_cmFilterActive()) {
    _cachedBookmarks = await apiFetch('/api/bookmarks');
  }
  // Re-render both views with updated filter
  _cmRenderTree();
  _cmRenderCodingTable();
}

function _cmSetSunoActive(val) {
  document.querySelectorAll('.cm-suno-btn').forEach(b => b.classList.toggle('cm-suno-active', b.dataset.suno === val));
}

document.getElementById('cm-filter-month-from').addEventListener('change', () => { _cmClearPhaseActive(); _cmApplyFilter(); });
document.getElementById('cm-filter-month-to').addEventListener('change',   () => { _cmClearPhaseActive(); _cmApplyFilter(); });

document.querySelectorAll('.cm-suno-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    _cmFilterSuno = btn.dataset.suno;
    _cmSetSunoActive(btn.dataset.suno);
    _cmApplyFilter();
  });
});

function _cmClearPhaseActive() {
  document.querySelectorAll('.cm-phase-btn').forEach(b => b.classList.remove('cm-phase-active'));
}

document.querySelectorAll('.cm-phase-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const already = btn.classList.contains('cm-phase-active');
    _cmClearPhaseActive();
    if (already) {
      // Second click on same phase → clear the date filter
      document.getElementById('cm-filter-month-from').value = '';
      document.getElementById('cm-filter-month-to').value   = '';
    } else {
      btn.classList.add('cm-phase-active');
      document.getElementById('cm-filter-month-from').value = btn.dataset.from;
      document.getElementById('cm-filter-month-to').value   = btn.dataset.to;
    }
    _cmApplyFilter();
  });
});

document.getElementById('cm-filter-clear').addEventListener('click', () => {
  _cmFilterDateFrom = '';
  _cmFilterDateTo   = '';
  _cmFilterSuno     = 'all';
  document.getElementById('cm-filter-month-from').value = '';
  document.getElementById('cm-filter-month-to').value   = '';
  _cmSetSunoActive('all');
  document.getElementById('cm-filter-clear').classList.add('hidden');
  _cmClearPhaseActive();
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

// Classic fixed-column table: used when â‰¤ 3 order levels exist
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
      const d = (bm.date || '').substring(0, 10);
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

