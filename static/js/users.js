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
      : String(u.weeks_with_messages ?? 'Ã¢');
    const pctDisp    = u.pct_weeks_active != null ? `${u.pct_weeks_active}%` : 'Ã¢';
    const avgWords   = u.avg_word_count != null ? Number(u.avg_word_count).toFixed(1) : 'Ã¢';

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
      <td class="users-td">${u.first_message_date || 'Ã¢'}</td>
      <td class="users-td">${u.last_message_date || 'Ã¢'}</td>
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
      : String(stats.weeks_with_messages ?? 'Ã¢');
    const pctStr = stats.pct_weeks_active != null ? `${stats.pct_weeks_active}%` : '—';
    const avgStr = stats.avg_word_count != null ? Number(stats.avg_word_count).toFixed(1) : '—';

    document.getElementById('upo-stats').innerHTML =
      `<div class="flex flex-wrap">` +
      _statPill('Total Messages', (stats.total_messages || 0).toLocaleString()) +
      _statPill('First Message',  stats.first_message_date || 'Ã¢') +
      _statPill('Last Message',   stats.last_message_date  || 'Ã¢') +
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
  document.getElementById('upo-sum-toggle').textContent = hidden ? 'Â¦ Summarize' : 'Â¦ Hide Summary';
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
  btn.textContent = hide ? '▼ Show' : 'â–² Hide';
});

function _upoSumLog(step, label, msg) {
  const icons = { input:'📋', context:'📡', llm:'✨', fallback:'⚠️' };
  const div = document.createElement('div');
  div.className = 'text-xs text-gray-600 flex items-start gap-1.5 py-0.5';
  div.innerHTML = `<span class="shrink-0">${icons[step] || '"¢'}</span>
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

  // Build formatted blocks: context before /Ëœ… user message / context after
  const blocks = msgs.map(m => {
    const ctx      = contextMap[String(m.id)] || [];
    const targetIdx= ctx.findIndex(r => r.is_target);
    const ctxPre   = targetIdx > 0 ? ctx.slice(0, targetIdx) : [];
    const ctxPost  = targetIdx >= 0 ? ctx.slice(targetIdx + 1) : [];
    const fmt      = r => `  [${r.username}]: ${r.content}`;

    let block = '';
    if (ctxPre.length)  block += ctxPre.map(fmt).join('\n') + '\n';
    block += `Ã¢Ëœ… [${m.username} | ${m.date}]: ${m.content}`;
    if (ctxPost.length) block += '\n' + ctxPost.map(fmt).join('\n');
    return block;
  });

  const conv   = blocks.join('\n\n---\n\n');
  const n      = msgs.length;
  const header = `USER PROFILE ANALYSIS ${_upoUsername} (${n} messages with context)`;

  const defaultPrompt = `Each block below contains one message from **${_upoUsername}** (markedËœ…) with surrounding conversation context. Concisely identify persona, topics, attitudes, actions, narratives, and identified changes in attitude and stance if present. Use tight bullet points. No padding, no repetition across sections.`;

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
    `<span class="fu-log-icon">${LOG_ICONS[entry.step] || '"¢'}</span>` +
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
