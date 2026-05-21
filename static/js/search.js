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
  // sortKey â†’ { display, count, dateFrom, dateTo }
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
    <span>Showing <strong>${filtered.length}</strong> of <strong>${currentResults.length}</strong> messages &mdash; <strong>${esc(range.label)}</strong> (${range.from === range.to ? range.from : range.from + ' â†’ ' + range.to})</span>
    <button id="chart-filter-clear" class="ml-auto text-indigo-600 hover:text-indigo-900 font-semibold">"¢ Clear</button>
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
              return r.from === r.to ? r.from : `${r.from} â†’ ${r.to}`;
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
  filter:      'ðŸ"',
  retrieval:   'ðŸ"¡',
  dedup:       'ðŸ§¹',
  cluster:     'ðŸ"®',
  sample:      'ðŸŽ¯',
  llm:         'âœ¨',
  fallback:    'âš ï¸',
  meta:        'ðŸ"…',
  instruction: 'ðŸ"',
};
function _updateSrCountLabel() {
  const el = document.getElementById('sr-count-label');
  if (el) el.textContent = currentResults.length.toLocaleString();
}

function renderSrLogEntry(entry) {
  const logEl = document.getElementById('sr-process-log');
  const div   = document.createElement('div');
  div.className = `log-entry log-step-${entry.step || 'fallback'}`;
  const icon = LOG_ICONS[entry.step] || '"¢';
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
    `<span class="fu-log-icon">${LOG_ICONS[entry.step] || '"¢'}</span>` +
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
    countLabel2.textContent = `${hits.length} of ${currentResults.length}· similarity"°Â¥ ${threshold}${interpreted}`;
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
    ? `<p class="text-xs text-gray-500 mt-1">ðŸ"Ž ${esc(msg.attachments)}</p>` : '';
  const reactLine = hasContent(msg.reactions)
    ? `<p class="text-xs text-gray-500 mt-1">ðŸ’¬ ${esc(msg.reactions)}</p>` : '';

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
          Show context â†•
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
    btn.textContent = 'Show context â†•';
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
    btn.textContent = 'Hide context â†•';
  } catch (e) { btn.textContent = 'Show context â†•'; console.error(e); }
  finally { btn.disabled = false; }
}

function ctxMsg(msg) {
  const cls = msg.is_target ? 'ctx-target' : 'ctx-regular';
  const targetBadge = msg.is_target
    ? `<span class="text-xs px-1.5 py-0.5 rounded font-semibold"
             style="background:#fef08a;color:#78350f">Ã¢Ëœ… result</span>` : '';
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

