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
    <div class="p-2 border border-dashed border-indigo-200 rounded-xl bg-indigo-50/30">
      <div class="flex items-start justify-between gap-2 mb-2">
        <p class="text-xs font-medium text-indigo-700">Coding whole quote</p>
        <button class="bm-code-panel-close text-gray-400 hover:text-gray-600 leading-none shrink-0 text-sm" data-bm-id="${bookmarkId}">✕</button>
      </div>
      <div class="flex items-center gap-1.5">
        <div class="relative flex-1 min-w-0">
          <input class="bm-new-code-input w-full border border-gray-200 rounded-lg px-2 py-1 text-xs bg-white focus:outline-none focus:ring-2 focus:ring-indigo-300"
                 placeholder="Type to search or create a code…" data-bm-id="${bookmarkId}" autocomplete="off" />
          <div class="bm-code-suggestions hidden absolute left-0 top-full mt-0.5 z-50 bg-white border border-gray-200 rounded-xl shadow-lg py-1 w-full max-h-44 overflow-y-auto"
               data-bm-id="${bookmarkId}" data-type="whole"></div>
        </div>
        ${_colorPickerHtml('', _randomCodeColor(), 'bm-new-code-color', `data-bm-id="${bookmarkId}"`)}
        <button class="bm-new-code-create text-xs px-2.5 py-1 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 shrink-0"
                data-bm-id="${bookmarkId}">Add</button>
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

