import { useState, useCallback, useEffect, useRef, useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import ReactMarkdown from 'react-markdown'
import { apiFetch, streamEvents } from '../api/client'
import { useSearchStore } from '../store/searchStore'
import { getUserStyle, highlightText, highlightTerms } from '../utils/colors'
import type { Message, Upload, UserInRange } from '../types'

type SearchTab = 'keyword' | 'semantic' | 'username' | 'range'
type MatchType = 'fuzzy' | 'exact' | 'any'
type SearchCat = 'chat' | 'users'
type TrendBucket = 'month' | 'week' | 'day'
type AnalysisTab = 'none' | 'summarize' | 'viz'
type FilterMode = 'exact' | 'any' | 'fuzzy' | 'semantic'

interface TrendPoint {
  period: string
  count: number
  from: string
  to: string
}

interface SrLogEntry {
  step: string
  label: string
  msg: string
}

const LOG_ICONS: Record<string, string> = {
  filter: '🔍',
  retrieval: '📡',
  dedup: '🧹',
  cluster: '🔮',
  sample: '🎯',
  llm: '✨',
  fallback: '⚠️',
  meta: '📅',
  instruction: '📝',
}

function UsernameTag({ username }: { username: string }) {
  return (
    <span className="ubadge text-xs font-semibold" style={getUserStyle(username)}>
      {username}
    </span>
  )
}

interface MessageCardProps {
  msg: Message
  keyword: string
  tokens: string[]
  isBookmarked: boolean
  isSelected: boolean
  onBookmarkToggle: (msg: Message) => void
  onSelectToggle: (id: number) => void
  ctxBefore: number
  ctxAfter: number
}

function MessageCard({ msg, keyword, tokens, isBookmarked, isSelected, onBookmarkToggle, onSelectToggle, ctxBefore, ctxAfter }: MessageCardProps) {
  const [copied, setCopied] = useState(false)
  const [ctxOpen, setCtxOpen] = useState(false)
  const [ctxMessages, setCtxMessages] = useState<Message[]>([])
  const [ctxLoading, setCtxLoading] = useState(false)

  const highlighted = tokens.length > 0
    ? highlightTerms(msg.content, tokens)
    : keyword
      ? highlightText(msg.content, keyword)
      : msg.content

  const handleCopy = () => {
    navigator.clipboard.writeText(msg.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  const handleCtxToggle = async () => {
    if (ctxOpen) { setCtxOpen(false); return }
    setCtxLoading(true)
    setCtxOpen(true)
    try {
      const data = await apiFetch<Message[]>(`/context/${msg.id}?before=${ctxBefore}&after=${ctxAfter}`)
      setCtxMessages(data)
    } catch { } finally { setCtxLoading(false) }
  }

  return (
    <div id={`card-${msg.id}`} className={`bg-white rounded-2xl shadow p-4 border-2 transition-colors ${isSelected ? 'border-indigo-300' : 'border-transparent'}`}>
      <div className="flex items-start gap-2">
        <input
          type="checkbox"
          checked={isSelected}
          onChange={() => onSelectToggle(msg.id)}
          className="mt-1 accent-indigo-600 shrink-0"
        />
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap mb-1">
                <UsernameTag username={msg.username} />
                <span className="text-xs text-gray-500">{msg.date?.slice(0, 16)}</span>
                {(msg.is_suno_team === 'True' || msg.is_suno_team === true || msg.is_suno_team === '1') && (
                  <span className="text-[10px] font-semibold bg-violet-100 text-violet-700 px-1.5 py-0.5 rounded-full">Suno Team</span>
                )}
                {msg.similarity != null && (
                  <span className="inline-flex items-center gap-0.5 text-[10px] font-semibold bg-indigo-50 text-indigo-600 border border-indigo-200 px-1.5 py-0.5 rounded-full tabular-nums">
                    <svg className="w-2.5 h-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
                    {(msg.similarity * 100).toFixed(1)}%
                  </span>
                )}
              </div>
              <p
                className="text-sm text-gray-800 whitespace-pre-wrap break-words"
                dangerouslySetInnerHTML={{ __html: highlighted }}
              />
            </div>
            <div className="flex flex-col gap-1 shrink-0">
              <button
                onClick={() => onBookmarkToggle(msg)}
                className="flex items-center gap-1 text-xs px-2 py-1 rounded-lg hover:bg-gray-100 transition-colors"
                title={isBookmarked ? 'Remove bookmark' : 'Save bookmark'}
              >
                {isBookmarked ? (
                  <>
                    <svg className="w-3.5 h-3.5 text-amber-500" fill="currentColor" viewBox="0 0 24 24"><path d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" /></svg>
                    <span className="text-amber-600">Bookmarked</span>
                  </>
                ) : (
                  <>
                    <svg className="w-3.5 h-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" /></svg>
                    <span className="text-gray-500">Bookmark</span>
                  </>
                )}
              </button>
              <button
                onClick={handleCopy}
                className="flex items-center gap-1 text-xs px-2 py-1 rounded-lg hover:bg-gray-100 transition-colors text-gray-500"
                title="Copy content"
              >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
                <span>{copied ? 'Copied!' : 'Copy'}</span>
              </button>
              <button
                onClick={handleCtxToggle}
                className="flex items-center gap-1 text-xs px-2 py-1 rounded-lg hover:bg-gray-100 transition-colors text-gray-500"
                title="Show context"
              >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" /></svg>
                <span>Context</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {ctxOpen && (
        <div className="mt-3 border-t border-gray-100 pt-3 ml-5">
          {ctxLoading ? (
            <p className="text-xs text-gray-400 text-center py-2">Loading context…</p>
          ) : (
            <div className="space-y-1.5">
              {ctxMessages.map((m) => (
                <div
                  key={m.id}
                  className={`flex gap-2 text-xs px-2 py-1 rounded ${m.id === msg.id ? 'bg-yellow-50 border border-yellow-200' : 'bg-gray-50'}`}
                >
                  <span className="shrink-0 font-semibold" style={getUserStyle(m.username)}>{m.username}</span>
                  <span className="text-gray-400 shrink-0">{m.date?.slice(0, 10)}</span>
                  <span className="text-gray-700 flex-1">{m.content}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// --- Client-side filter helpers ---

function fuzzyMatch(text: string, pattern: string): boolean {
  // Each char in pattern must appear in text in order
  let pi = 0
  const lText = text.toLowerCase()
  const lPat = pattern.toLowerCase()
  for (let i = 0; i < lText.length && pi < lPat.length; i++) {
    if (lText[i] === lPat[pi]) pi++
  }
  return pi === lPat.length
}

function applyExactFilter(messages: Message[], term: string): Message[] {
  if (!term.trim()) return messages
  const words = term.trim().toLowerCase().split(/\s+/)
  return messages.filter((m) => {
    const lc = m.content.toLowerCase()
    return words.every((w) => lc.includes(w))
  })
}

function applyAnyFilter(messages: Message[], term: string): Message[] {
  if (!term.trim()) return messages
  const words = term.trim().toLowerCase().split(/\s+/)
  return messages.filter((m) => {
    const lc = m.content.toLowerCase()
    return words.some((w) => lc.includes(w))
  })
}

function applyFuzzyFilter(messages: Message[], term: string): Message[] {
  if (!term.trim()) return messages
  const words = term.trim().split(/\s+/)
  return messages.filter((m) =>
    words.every((w) => fuzzyMatch(m.content, w))
  )
}

function computeFilterTokens(mode: FilterMode, term: string): string[] {
  if (!term.trim()) return []
  const words = term.trim().split(/\s+/).filter(Boolean)
  if (mode === 'exact') {
    return words.length > 1 ? [term.trim(), ...words] : [term.trim()]
  }
  if (mode === 'any' || mode === 'fuzzy') {
    return words
  }
  // semantic: no highlighting
  return []
}

// --- Main component ---

// Safe sessionStorage helpers
function ssGet<T>(key: string, fallback: T): T {
  try { const v = sessionStorage.getItem(key); return v !== null ? (JSON.parse(v) as T) : fallback } catch { return fallback }
}
function ssSet(key: string, value: unknown) {
  try { sessionStorage.setItem(key, JSON.stringify(value)) } catch {}
}

export default function SearchPage() {
  const { results, setResults, selectedIds, toggleSelected, selectAll, clearSelected, bookmarkedIds, setBookmarkedIds, toggleBookmarked } = useSearchStore()

  // Search state
  const [searchCat, setSearchCat] = useState<SearchCat>(() => (localStorage.getItem('search_cat') as SearchCat) || 'chat')
  const [activeTab, setActiveTab] = useState<SearchTab>(() => (localStorage.getItem('search_tab') as SearchTab) || 'keyword')
  const [matchType, setMatchType] = useState<MatchType>(() => ssGet('sr_match_type', 'fuzzy'))
  const [query, setQuery] = useState(() => ssGet('sr_query', ''))
  const [filterUsername, setFilterUsername] = useState(() => ssGet('sr_filter_username', ''))
  const [dateFrom, setDateFrom] = useState(() => ssGet('sr_date_from', ''))
  const [dateTo, setDateTo] = useState(() => ssGet('sr_date_to', ''))
  const [sunoTeam, setSunoTeam] = useState('all')
  const [maxResults, setMaxResults] = useState(200)
  const [minWords, setMinWords] = useState(0)
  const [ctxBefore, setCtxBefore] = useState(10)
  const [ctxAfter, setCtxAfter] = useState(10)
  const [sortBy, setSortBy] = useState('similarity')
  const [showOptions, setShowOptions] = useState(false)
  const [searching, setSearching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [bookmarkError, setBookmarkError] = useState<string | null>(null)
  const [uploads, setUploads] = useState<Upload[]>([])
  const [selectedUploads, setSelectedUploads] = useState<Set<string>>(new Set())
  const [userStats, setUserStats] = useState<UserInRange[]>(() => ssGet<UserInRange[]>('sr_user_stats', []))
  const [userSortCol, setUserSortCol] = useState<'username' | 'total_messages' | 'first_date' | 'last_date' | 'avg_words' | 'weeks_active' | 'pct_weeks'>('total_messages')
  const [userSortDir, setUserSortDir] = useState<'asc' | 'desc'>('desc')
  const [userNameFilter, setUserNameFilter] = useState('')
  const [refineFirstMsgIn, setRefineFirstMsgIn] = useState('')
  const [refineLastMsgFrom, setRefineLastMsgFrom] = useState('')
  const [refineMinMessages, setRefineMinMessages] = useState('')
  const [refineMinWeeks, setRefineMinWeeks] = useState('')
  const [refineAvgWordsMin, setRefineAvgWordsMin] = useState('')
  const [refineAvgWordsMax, setRefineAvgWordsMax] = useState('')
  const [trendData, setTrendData] = useState<TrendPoint[]>(() => ssGet<TrendPoint[]>('sr_trend_data', []))
  const [trendBucket, setTrendBucket] = useState<TrendBucket>('month')
  const [lastKeyword, setLastKeyword] = useState(() => ssGet('sr_last_keyword', ''))
  const [lastTokens, setLastTokens] = useState<string[]>(() => ssGet<string[]>('sr_last_tokens', []))
  const [showSearchPanel, setShowSearchPanel] = useState(true)
  const [rangeLimit, setRangeLimit] = useState(200)
  const [rangeOffset, setRangeOffset] = useState(0)
  const [rangeMode, setRangeMode] = useState<'exact' | 'month'>('exact')

  // Analysis tabs
  const [analysisTab, setAnalysisTab] = useState<AnalysisTab>('none')

  // Summarize & Analyse state
  const [summarizeQuery, setSummarizeQuery] = useState('')
  const [summarizeResult, setSummarizeResult] = useState('')
  const [summarizeLoading, setSummarizeLoading] = useState(false)
  const [summarizeFollowUp, setSummarizeFollowUp] = useState<{ role: 'user' | 'assistant'; content: string }[]>([])
  const [summarizeFollowUpLoading, setSummarizeFollowUpLoading] = useState(false)
  const [summarizeFollowUpInput, setSummarizeFollowUpInput] = useState('')
  const [srModel, setSrModel] = useState('gpt-4o')
  const [srRetrievalMode, setSrRetrievalMode] = useState<'cluster' | 'all'>('cluster')
  const [srLog, setSrLog] = useState<SrLogEntry[]>([])
  const [srLogVisible, setSrLogVisible] = useState(true)
  const [showSrResultsPanel, setShowSrResultsPanel] = useState(false)
  const stopSummarizeRef = useRef<(() => void) | null>(null)

  // Results filter bar state
  const [filterMode, setFilterMode] = useState<FilterMode>('exact')
  const [filterText, setFilterText] = useState('')
  const [displayedResults, setDisplayedResults] = useState<Message[]>([])
  const [filterCount, setFilterCount] = useState<{ shown: number; total: number } | null>(null)
  const [filterLoading, setFilterLoading] = useState(false)
  const [filterTokens, setFilterTokens] = useState<string[]>([])

  // Chart range filter
  const [chartRange, setChartRange] = useState<{ from: string; to: string; label: string } | null>(null)

  // User detail panel state
  const [selectedUser, setSelectedUser] = useState<UserInRange | null>(null)
  const [userMsgs, setUserMsgs] = useState<Message[]>([])
  const [userMsgsLoading, setUserMsgsLoading] = useState(false)
  const [userDetailDateFrom, setUserDetailDateFrom] = useState('')
  const [userDetailDateTo, setUserDetailDateTo] = useState('')
  const [userDetailContains, setUserDetailContains] = useState('')
  const [userDetailMinWords, setUserDetailMinWords] = useState(0)
  const [userDetailCtxBefore, setUserDetailCtxBefore] = useState(5)
  const [userDetailCtxAfter, setUserDetailCtxAfter] = useState(5)
  const [userSumModel, setUserSumModel] = useState('gpt-5.4')
  const [userSumRetrievalMode, setUserSumRetrievalMode] = useState<'cluster' | 'all'>('cluster')
  const [userSumInstructions, setUserSumInstructions] = useState('')
  const [userSumResult, setUserSumResult] = useState('')
  const [userSumLoading, setUserSumLoading] = useState(false)
  const [userSumVisible, setUserSumVisible] = useState(false)
  const [userSumFollowUp, setUserSumFollowUp] = useState<{ role: 'user' | 'assistant'; content: string }[]>([])
  const [userSumFollowUpInput, setUserSumFollowUpInput] = useState('')
  const [userSumFollowUpLoading, setUserSumFollowUpLoading] = useState(false)
  const [userDetailVisible, setUserDetailVisible] = useState(50)
  const userSumStopRef = useRef<(() => void) | null>(null)

  // Semantic filter debounce ref
  const semanticFilterTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Infinite scroll
  const [visibleCount, setVisibleCount] = useState(50)
  const sentinelRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    apiFetch<Upload[]>('/uploads').then(setUploads).catch(() => {})
    apiFetch<number[]>('/bookmarks/ids').then(setBookmarkedIds).catch(() => {})
  }, [setBookmarkedIds])

  // Restore results to Zustand store on mount if store is empty (page refresh) but sessionStorage has data
  useEffect(() => {
    if (results.length === 0) {
      const saved = ssGet<Message[]>('sr_results', [])
      if (saved.length > 0) setResults(saved)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Sync key state to sessionStorage so it survives navigation and page refresh
  useEffect(() => { ssSet('sr_results', results) }, [results])
  useEffect(() => { ssSet('sr_query', query) }, [query])
  useEffect(() => { ssSet('sr_match_type', matchType) }, [matchType])
  useEffect(() => { ssSet('sr_date_from', dateFrom) }, [dateFrom])
  useEffect(() => { ssSet('sr_date_to', dateTo) }, [dateTo])
  useEffect(() => { ssSet('sr_filter_username', filterUsername) }, [filterUsername])
  useEffect(() => { ssSet('sr_last_keyword', lastKeyword) }, [lastKeyword])
  useEffect(() => { ssSet('sr_last_tokens', lastTokens) }, [lastTokens])
  useEffect(() => { ssSet('sr_trend_data', trendData) }, [trendData])
  useEffect(() => { ssSet('sr_user_stats', userStats) }, [userStats])

  // Reset to first page whenever the filtered result set changes (new search or filter applied)
  useEffect(() => {
    setVisibleCount(50)
  }, [displayedResults])

  // IntersectionObserver: load next 50 when sentinel scrolls into view
  useEffect(() => {
    const el = sentinelRef.current
    if (!el) return
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          setVisibleCount((prev) => prev + 50)
        }
      },
      { rootMargin: '200px' },
    )
    observer.observe(el)
    return () => observer.disconnect()
  // Re-attach whenever displayedResults changes so the cap logic in render stays correct
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [displayedResults.length])

  const scopeParam = selectedUploads.size > 0 ? [...selectedUploads].join(',') : ''

  const buildParams = useCallback(() => {
    const p = new URLSearchParams()
    if (dateFrom) p.set('date_from', dateFrom)
    if (dateTo) p.set('date_to', dateTo)
    if (sunoTeam === 'only') p.set('is_suno_team', 'only')
    else if (sunoTeam === 'exclude') p.set('is_suno_team', 'exclude')
    if (minWords > 0) p.set('min_words', String(minWords))
    p.set('limit', String(maxResults))
    if (scopeParam) p.set('upload_ids', scopeParam)
    return p
  }, [dateFrom, dateTo, sunoTeam, minWords, maxResults, scopeParam])

  const toggleScopeChip = (id: string) => {
    setSelectedUploads((prev) => {
      const next = new Set(prev)
      if (next.size === 0) {
        uploads.forEach((u) => { if (u.id !== id) next.add(u.id) })
      } else if (next.has(id)) {
        next.delete(id)
        if (next.size === 0) return new Set()
      } else {
        next.add(id)
        if (next.size === uploads.length) return new Set()
      }
      return next
    })
  }

  const buildTrend = useCallback((data: Message[], bucket: TrendBucket) => {
    const counts: Record<string, number> = {}
    const mins: Record<string, string> = {}
    const maxs: Record<string, string> = {}
    data.forEach((m) => {
      if (!m.date) return
      let key: string
      if (bucket === 'month') key = m.date.slice(0, 7)
      else if (bucket === 'day') key = m.date.slice(0, 10)
      else {
        const d = new Date(m.date)
        const monday = new Date(d)
        monday.setDate(d.getDate() - ((d.getDay() + 6) % 7))
        key = monday.toISOString().slice(0, 10)
      }
      counts[key] = (counts[key] || 0) + 1
      const day = m.date.slice(0, 10)
      if (!mins[key] || day < mins[key]) mins[key] = day
      if (!maxs[key] || day > maxs[key]) maxs[key] = day
    })
    const sorted = Object.keys(counts).sort()
    setTrendData(sorted.map((period) => ({
      period,
      count: counts[period],
      from: mins[period],
      to: maxs[period],
    })))
  }, [])

  // Apply text filter + optional chart range filter on top of full results
  const applyFilters = useCallback((
    allResults: Message[],
    mode: FilterMode,
    text: string,
    range: { from: string; to: string; label: string } | null,
    tokens: string[],
  ) => {
    let filtered = allResults

    // Apply chart range first
    if (range) {
      filtered = filtered.filter((m) => {
        const d = m.date?.slice(0, 10) ?? ''
        return d >= range.from && d <= range.to
      })
    }

    // Apply text filter
    if (text.trim()) {
      if (mode === 'exact') {
        filtered = applyExactFilter(filtered, text)
      } else if (mode === 'any') {
        filtered = applyAnyFilter(filtered, text)
      } else if (mode === 'fuzzy') {
        filtered = applyFuzzyFilter(filtered, text)
      }
      // semantic is handled separately (async)
    }

    setDisplayedResults(filtered)
    setFilterTokens(tokens)
    const hasFilter = text.trim() || range
    setFilterCount(hasFilter ? { shown: filtered.length, total: allResults.length } : null)
  }, [])

  const runSemanticFilter = useCallback(async (text: string, allResults: Message[]) => {
    if (!text.trim()) {
      setDisplayedResults(allResults)
      setFilterCount(null)
      setFilterTokens([])
      return
    }
    setFilterLoading(true)
    try {
      const data = await apiFetch<Message[]>('/filter/semantic', {
        method: 'POST',
        body: JSON.stringify({ query: text, messages: allResults }),
      })
      setDisplayedResults(data)
      setFilterTokens([])
      setFilterCount({ shown: data.length, total: allResults.length })
    } catch {
      setDisplayedResults(allResults)
      setFilterCount(null)
    } finally {
      setFilterLoading(false)
    }
  }, [])

  // When filter text or mode changes, recompute displayedResults
  useEffect(() => {
    if (results.length === 0) return
    if (filterMode === 'semantic') {
      if (semanticFilterTimerRef.current) clearTimeout(semanticFilterTimerRef.current)
      semanticFilterTimerRef.current = setTimeout(() => {
        runSemanticFilter(filterText, results)
      }, 500)
    } else {
      const tokens = computeFilterTokens(filterMode, filterText)
      applyFilters(results, filterMode, filterText, chartRange, tokens)
    }
  }, [filterText, filterMode, results, chartRange, applyFilters, runSemanticFilter])

  const handleChartBarClick = (pt: TrendPoint) => {
    if (chartRange?.from === pt.from) {
      // Deselect
      setChartRange(null)
      const tokens = computeFilterTokens(filterMode, filterText)
      applyFilters(results, filterMode, filterText, null, tokens)
    } else {
      const range = { from: pt.from, to: pt.to, label: pt.period }
      setChartRange(range)
      const tokens = computeFilterTokens(filterMode, filterText)
      applyFilters(results, filterMode, filterText, range, tokens)
    }
  }

  const doSearch = useCallback(async () => {
    setSearching(true)
    setError(null)
    setTrendData([])
    clearSelected()
    setFilterText('')
    setFilterMode('exact')
    setFilterCount(null)
    setFilterTokens([])
    setChartRange(null)
    setAnalysisTab('none')
    setResults([])
    setDisplayedResults([])

    // Range tab validation
    if (activeTab === 'range' && !dateFrom && !dateTo) {
      setError('Please enter at least one date before searching.')
      setSearching(false)
      return
    }

    try {
      const params = buildParams()
      let data: Message[] = []
      let kw = ''
      let toks: string[] = []

      if (activeTab === 'keyword') {
        if (!query.trim()) { setSearching(false); return }
        params.set('q', query)
        params.set('match_type', matchType === 'any' ? 'any_word' : matchType)
        if (filterUsername) params.set('username', filterUsername)
        data = await apiFetch<Message[]>(`/search/keyword?${params}`)
        kw = query
        if (matchType === 'exact') toks = [query]
        else if (matchType === 'any') toks = query.split(/\s+/).filter(Boolean)
      } else if (activeTab === 'semantic') {
        if (!query.trim()) { setSearching(false); return }
        params.set('q', query)
        if (filterUsername) params.set('username', filterUsername)
        if (sortBy !== 'similarity') params.set('sort_by', sortBy)
        const raw = await apiFetch<Message[] | { error: string; results: [] }>(`/search/semantic?${params}`)
        if (Array.isArray(raw)) {
          data = raw
        } else {
          setError((raw as { error: string }).error || 'Semantic search unavailable (ChromaDB may not be running)')
          data = []
        }
        kw = query
      } else if (activeTab === 'username') {
        if (!query.trim()) { setSearching(false); return }
        params.set('q', query)
        data = await apiFetch<Message[]>(`/search/username?${params}`)
      } else if (activeTab === 'range') {
        const p = buildParams()
        p.delete('limit') // range search returns all messages in the date range — no cap
        p.set('offset', String(rangeOffset))
        data = await apiFetch<Message[]>(`/search/range?${p}`)
      }

      setResults(data)
      setDisplayedResults(data)
      setLastKeyword(kw)
      setLastTokens(toks)
      if (data.length > 0) buildTrend(data, trendBucket)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed')
    } finally {
      setSearching(false)
    }
  }, [query, activeTab, matchType, filterUsername, buildParams, rangeLimit, rangeOffset, sortBy, trendBucket, setResults, clearSelected, buildTrend, dateFrom, dateTo])

  const doUserSearch = async () => {
    setSearching(true)
    setError(null)
    try {
      const params = buildParams()
      const data = await apiFetch<UserInRange[]>(`/search/users-in-range?${params}`)
      setUserStats(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed')
    } finally {
      setSearching(false)
    }
  }

  const handleBookmarkToggle = async (msg: Message) => {
    const wasBookmarked = bookmarkedIds.has(msg.id)
    // Optimistic update — flip the UI immediately so the button responds at once.
    toggleBookmarked(msg.id)
    try {
      if (wasBookmarked) {
        await apiFetch(`/bookmarks/by-msg/${msg.id}`, { method: 'DELETE' })
      } else {
        await apiFetch('/bookmarks', {
          method: 'POST',
          body: JSON.stringify({ msg_id: msg.id }),
        })
      }
    } catch (err) {
      // Revert the optimistic update and surface the error.
      toggleBookmarked(msg.id)
      const msg2 = err instanceof Error ? err.message : 'Bookmark action failed'
      setBookmarkError(msg2)
      setTimeout(() => setBookmarkError(null), 5000)
    }
  }

  const handleTabSwitch = (tab: SearchTab) => {
    localStorage.setItem('search_tab', tab)
    setActiveTab(tab)
    setResults([])
    setDisplayedResults([])
    setTrendData([])
    setLastKeyword('')
    setLastTokens([])
    setFilterText('')
    setFilterCount(null)
    setFilterTokens([])
    setChartRange(null)
    setAnalysisTab('none')
    clearSelected()
  }

  const handleCatSwitch = (cat: SearchCat) => {
    localStorage.setItem('search_cat', cat)
    setSearchCat(cat)
    setResults([])
    setDisplayedResults([])
    setUserStats([])
    setTrendData([])
    setFilterText('')
    setFilterCount(null)
    setFilterTokens([])
    setChartRange(null)
    setAnalysisTab('none')
    clearSelected()
  }

  const exportCSV = () => {
    const sel = results.filter((r) => selectedIds.has(r.id))
    const rows = sel.length > 0 ? sel : results
    if (rows.length === 0) return
    const header = ['id', 'username', 'date', 'content', 'is_suno_team']
    const lines = rows.map((r) =>
      [r.id, r.username, r.date, `"${(r.content || '').replace(/"/g, '""')}"`, r.is_suno_team ?? ''].join(',')
    )
    const csv = [header.join(','), ...lines].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'search_results.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  const copySelected = () => {
    const sel = results.filter((r) => selectedIds.has(r.id))
    if (sel.length === 0) return
    const text = sel.map((r) => `[${r.username} @ ${r.date?.slice(0, 16)}]\n${r.content}`).join('\n\n---\n\n')
    navigator.clipboard.writeText(text)
  }

  const handleSummarize = () => {
    const sel = results.filter((r) => selectedIds.has(r.id))
    const msgs = sel.length > 0 ? sel : results
    if (msgs.length === 0) return
    setSummarizeResult('')
    setSummarizeFollowUp([])
    setSummarizeLoading(true)
    setSrLog([])
    setShowSrResultsPanel(true)

    if (stopSummarizeRef.current) stopSummarizeRef.current()

    stopSummarizeRef.current = streamEvents(
      '/summarize-results',
      {
        messages: msgs.map((m) => ({ username: m.username, date: m.date, content: m.content, msg_uuid: m.msg_uuid })),
        prompt: summarizeQuery,
        model: srModel,
        retrieval_mode: srRetrievalMode,
      },
      (event) => {
        if (event.type === 'log') {
          setSrLog((prev) => [...prev, {
            step: String(event.step ?? ''),
            label: String(event.label ?? event.step ?? ''),
            msg: String(event.message ?? event.msg ?? ''),
          }])
        } else if (event.content) {
          setSummarizeResult((prev) => prev + String(event.content))
        }
      },
      () => setSummarizeLoading(false),
      (err) => {
        setSummarizeResult(`Error: ${err.message}`)
        setSummarizeLoading(false)
      },
    )
  }

  const handleSumFollowUp = () => {
    if (!summarizeFollowUpInput.trim() || summarizeFollowUpLoading) return
    const q = summarizeFollowUpInput.trim()
    setSummarizeFollowUpInput('')
    setSummarizeFollowUp((prev) => [...prev, { role: 'user', content: q }])
    setSummarizeFollowUpLoading(true)
    let answer = ''
    streamEvents(
      '/summarize-results/followup',
      { question: q, summary: summarizeResult, history: summarizeFollowUp },
      (event) => {
        if (event.type === 'chunk') {
          answer += String(event.content ?? '')
          setSummarizeFollowUp((prev) => {
            const last = prev[prev.length - 1]
            if (last?.role === 'assistant') return [...prev.slice(0, -1), { role: 'assistant', content: answer }]
            return [...prev, { role: 'assistant', content: answer }]
          })
        } else if (event.type === 'result') {
          answer = String(event.content ?? answer)
          setSummarizeFollowUp((prev) => {
            const last = prev[prev.length - 1]
            if (last?.role === 'assistant') return [...prev.slice(0, -1), { role: 'assistant', content: answer }]
            return [...prev, { role: 'assistant', content: answer }]
          })
        }
      },
      () => setSummarizeFollowUpLoading(false),
      (err) => {
        setSummarizeFollowUp((prev) => [...prev, { role: 'assistant', content: `Error: ${err.message}` }])
        setSummarizeFollowUpLoading(false)
      },
    )
  }

  const handleExportPDF = () => {
    if (!summarizeResult) return
    const win = window.open('', '_blank')
    if (!win) return
    win.document.write(`<!DOCTYPE html><html><head><meta charset="utf-8"><title>Analysis Export</title>
      <style>
        body { font-family: Georgia, serif; max-width: 800px; margin: 40px auto; padding: 20px; line-height: 1.6; color: #1a1a1a; }
        h1, h2, h3 { color: #1e1b4b; }
        pre { background: #f5f5f5; padding: 12px; border-radius: 4px; overflow-x: auto; }
        blockquote { border-left: 4px solid #818cf8; margin: 0; padding-left: 16px; color: #555; }
        .followup { margin-top: 32px; border-top: 2px solid #e5e7eb; padding-top: 16px; }
        .q { font-weight: bold; color: #3730a3; margin-bottom: 4px; }
        .a { margin-bottom: 16px; }
      </style>
    </head><body>`)
    win.document.write(`<h1>Analysis Results</h1>`)
    win.document.write(`<div>${summarizeResult.replace(/\n/g, '<br>')}</div>`)
    if (summarizeFollowUp.length > 0) {
      win.document.write(`<div class="followup"><h2>Follow-up Q&amp;A</h2>`)
      summarizeFollowUp.forEach((h) => {
        if (h.role === 'user') {
          win.document.write(`<div class="q">Q: ${h.content}</div>`)
        } else {
          win.document.write(`<div class="a">${h.content.replace(/\n/g, '<br>')}</div>`)
        }
      })
      win.document.write(`</div>`)
    }
    win.document.write(`</body></html>`)
    win.document.close()
    win.print()
  }

  const filteredUsers = userStats.filter((u) => {
    if (userNameFilter.trim() && !u.username.toLowerCase().includes(userNameFilter.trim().toLowerCase())) return false
    if (refineFirstMsgIn && (u.first_date ?? '') < refineFirstMsgIn) return false
    if (refineLastMsgFrom && (u.last_date ?? '') < refineLastMsgFrom) return false
    if (refineMinMessages && u.total_messages < parseInt(refineMinMessages)) return false
    if (refineMinWeeks && (u.weeks_active ?? 0) < parseInt(refineMinWeeks)) return false
    if (refineAvgWordsMin && (u.avg_words ?? 0) < parseFloat(refineAvgWordsMin)) return false
    if (refineAvgWordsMax && (u.avg_words ?? 0) > parseFloat(refineAvgWordsMax)) return false
    return true
  })

  const sortedUsers = [...filteredUsers].sort((a, b) => {
    const dir = userSortDir === 'asc' ? 1 : -1
    switch (userSortCol) {
      case 'username': return dir * a.username.localeCompare(b.username)
      case 'first_date': return dir * (a.first_date ?? '').localeCompare(b.first_date ?? '')
      case 'last_date': return dir * (a.last_date ?? '').localeCompare(b.last_date ?? '')
      case 'avg_words': return dir * ((a.avg_words ?? 0) - (b.avg_words ?? 0))
      case 'weeks_active': return dir * ((a.weeks_active ?? 0) - (b.weeks_active ?? 0))
      case 'pct_weeks': return dir * ((a.pct_weeks ?? 0) - (b.pct_weeks ?? 0))
      default: return dir * (a.total_messages - b.total_messages)
    }
  })

  const handleUserSortCol = (col: typeof userSortCol) => {
    if (userSortCol === col) setUserSortDir((d) => d === 'asc' ? 'desc' : 'asc')
    else { setUserSortCol(col); setUserSortDir('desc') }
  }

  const clearRefine = () => {
    setUserNameFilter('')
    setRefineFirstMsgIn('')
    setRefineLastMsgFrom('')
    setRefineMinMessages('')
    setRefineMinWeeks('')
    setRefineAvgWordsMin('')
    setRefineAvgWordsMax('')
  }

  const toggleAnalysisTab = (tab: 'summarize' | 'viz') => {
    setAnalysisTab((prev) => prev === tab ? 'none' : tab)
  }

  // User detail derived state
  const userDetailFiltered = useMemo(() => {
    let msgs = userMsgs
    if (userDetailDateFrom) msgs = msgs.filter(m => (m.date?.slice(0, 10) ?? '') >= userDetailDateFrom)
    if (userDetailDateTo) msgs = msgs.filter(m => (m.date?.slice(0, 10) ?? '') <= userDetailDateTo)
    if (userDetailContains.trim()) {
      const words = userDetailContains.toLowerCase().split(/\s+/).filter(Boolean)
      msgs = msgs.filter(m => words.every(w => m.content.toLowerCase().includes(w)))
    }
    if (userDetailMinWords > 0) {
      msgs = msgs.filter(m => m.content.split(/\s+/).filter(Boolean).length >= userDetailMinWords)
    }
    return msgs
  }, [userMsgs, userDetailDateFrom, userDetailDateTo, userDetailContains, userDetailMinWords])

  const userDetailStats = useMemo(() => {
    if (userDetailFiltered.length === 0) return null
    const dates = userDetailFiltered.map(m => m.date?.slice(0, 19) ?? '').filter(Boolean).sort()
    const first = dates[0] ?? ''
    const last = dates[dates.length - 1] ?? ''
    const avgWords = userDetailFiltered.reduce((s, m) => s + m.content.split(/\s+/).filter(Boolean).length, 0) / userDetailFiltered.length
    const weekKey = (ds: string) => {
      const d = new Date(ds)
      const ys = new Date(d.getFullYear(), 0, 1)
      const doy = Math.floor((d.getTime() - ys.getTime()) / 86400000)
      const w = Math.floor((doy + (ys.getDay() === 0 ? 6 : ys.getDay() - 1)) / 7)
      return `${d.getFullYear()}-${String(w).padStart(2, '0')}`
    }
    const weeksActive = new Set(userDetailFiltered.map(m => m.date ? weekKey(m.date) : null).filter(Boolean)).size
    const totalWeeks = first && last ? Math.max(1, Math.ceil((new Date(last).getTime() - new Date(first).getTime()) / (7 * 86400000)) + 1) : 1
    return {
      total: userDetailFiltered.length,
      first: first.replace('T', ' '),
      last: last.replace('T', ' '),
      avgWords: Math.round(avgWords * 10) / 10,
      weeksActive,
      totalWeeks,
      pctWeeks: Math.round(weeksActive / totalWeeks * 1000) / 10,
    }
  }, [userDetailFiltered])

  const openUserDetail = async (u: UserInRange) => {
    setSelectedUser(u)
    setUserMsgsLoading(true)
    setUserMsgs([])
    setUserDetailDateFrom(dateFrom)
    setUserDetailDateTo(dateTo)
    setUserDetailContains('')
    setUserDetailMinWords(0)
    setUserSumResult('')
    setUserSumVisible(false)
    setUserSumFollowUp([])
    setUserDetailVisible(50)
    try {
      const data = await apiFetch<Message[]>(`/search/user-messages?username=${encodeURIComponent(u.username)}&limit=10000`)
      setUserMsgs(data)
    } finally {
      setUserMsgsLoading(false)
    }
  }

  const runUserSummary = () => {
    if (userDetailFiltered.length === 0) return
    setUserSumResult('')
    setUserSumFollowUp([])
    setUserSumLoading(true)
    setUserSumVisible(true)
    if (userSumStopRef.current) userSumStopRef.current()
    userSumStopRef.current = streamEvents(
      '/summarize-results',
      {
        messages: userDetailFiltered.map(m => ({ username: m.username, date: m.date, content: m.content })),
        prompt: userSumInstructions,
        model: userSumModel,
        retrieval_mode: userSumRetrievalMode,
      },
      (event) => { if (event.content) setUserSumResult(prev => prev + String(event.content)) },
      () => setUserSumLoading(false),
      (err) => { setUserSumResult(`Error: ${err.message}`); setUserSumLoading(false) },
    )
  }

  const runUserSumFollowUp = () => {
    if (!userSumFollowUpInput.trim() || userSumFollowUpLoading) return
    const q = userSumFollowUpInput.trim()
    setUserSumFollowUpInput('')
    setUserSumFollowUp(prev => [...prev, { role: 'user', content: q }])
    setUserSumFollowUpLoading(true)
    let answer = ''
    streamEvents(
      '/summarize-results/followup',
      { question: q, summary: userSumResult, history: userSumFollowUp },
      (event) => {
        if (event.type === 'chunk' || event.content) {
          answer += String(event.content ?? '')
          setUserSumFollowUp(prev => {
            const last = prev[prev.length - 1]
            if (last?.role === 'assistant') return [...prev.slice(0, -1), { role: 'assistant', content: answer }]
            return [...prev, { role: 'assistant', content: answer }]
          })
        }
      },
      () => setUserSumFollowUpLoading(false),
      (err) => {
        setUserSumFollowUp(prev => [...prev, { role: 'assistant', content: `Error: ${err.message}` }])
        setUserSumFollowUpLoading(false)
      },
    )
  }

  const inputCls = 'search-input'
  const tabs: { id: SearchTab; label: string }[] = [
    { id: 'keyword', label: 'Keyword' },
    { id: 'semantic', label: 'Semantic' },
    { id: 'username', label: 'Username' },
    { id: 'range', label: 'Time Range' },
  ]

  const selectedCount = selectedIds.size

  // Determine which tokens to use for card highlighting
  const activeTokens = filterText.trim() ? filterTokens : lastTokens
  const activeKeyword = filterText.trim() ? '' : lastKeyword

  // Filter mode placeholder text
  const filterPlaceholder: Record<FilterMode, string> = {
    exact: 'Filter: all words must match…',
    any: 'Filter: any word matches…',
    fuzzy: 'Filter: fuzzy char-order match…',
    semantic: 'Filter: semantic similarity…',
  }

  // Retrieval mode hint
  const retrievalHint: Record<'cluster' | 'all', string> = {
    cluster: 'Groups messages by topic and samples representatives. Best for large result sets.',
    all: 'Passes all messages directly to the model. Use for small result sets (<100 msgs).',
  }

  // User detail full-panel view
  if (selectedUser) {
    const u = selectedUser
    const visibleMsgs = userDetailFiltered.slice(0, userDetailVisible)
    return (
      <div className="max-w-5xl mx-auto px-3 sm:px-4 py-4 space-y-4">
        {/* Header bar */}
        <div className="bg-indigo-700 text-white rounded-2xl shadow px-4 py-3 flex items-center gap-3">
          <button
            onClick={() => setSelectedUser(null)}
            className="flex items-center gap-1.5 text-sm font-semibold text-indigo-200 hover:text-white transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" /></svg>
            Back
          </button>
          <span className="text-base font-bold flex-1 truncate">{u.username}</span>
          <span className="text-sm text-indigo-200 shrink-0">{userMsgsLoading ? 'Loading…' : `${userMsgs.length.toLocaleString()} messages`}</span>
        </div>

        {/* Stats cards — reflect filtered messages */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {[
            { label: 'TOTAL MESSAGES', value: userDetailStats ? userDetailStats.total.toLocaleString() : '—' },
            { label: 'FIRST MESSAGE', value: userDetailStats?.first.slice(0, 19) ?? '—' },
            { label: 'LAST MESSAGE', value: userDetailStats?.last.slice(0, 19) ?? '—' },
            { label: 'AVG WORDS', value: userDetailStats ? String(userDetailStats.avgWords) : '—' },
            { label: 'WEEKS ACTIVE', value: userDetailStats ? `${userDetailStats.weeksActive} / ${userDetailStats.totalWeeks}` : '—' },
            { label: '% WEEKS ACTIVE', value: userDetailStats ? `${userDetailStats.pctWeeks}%` : '—' },
          ].map(({ label, value }) => (
            <div key={label} className="bg-white rounded-xl shadow px-3 py-2.5">
              <p className="text-[10px] font-semibold text-indigo-600 uppercase tracking-wide">{label}</p>
              <p className="text-sm font-bold text-gray-900 mt-0.5 truncate">{value}</p>
            </div>
          ))}
        </div>

        {/* Filters */}
        <div className="bg-white rounded-2xl shadow px-4 py-3">
          <div className="flex flex-wrap items-end gap-x-4 gap-y-2">
            <div className="flex flex-col gap-1">
              <label className="filter-label">From</label>
              <input type="date" value={userDetailDateFrom} onChange={e => setUserDetailDateFrom(e.target.value)} className="search-input" />
            </div>
            <div className="flex flex-col gap-1">
              <label className="filter-label">To</label>
              <input type="date" value={userDetailDateTo} onChange={e => setUserDetailDateTo(e.target.value)} className="search-input" />
            </div>
            <div className="flex flex-col gap-1 flex-1 min-w-[160px]">
              <label className="filter-label">Contains</label>
              <input
                value={userDetailContains}
                onChange={e => setUserDetailContains(e.target.value)}
                placeholder="keyword..."
                className="search-input"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="filter-label">Min words</label>
              <input
                type="text" inputMode="numeric"
                value={String(userDetailMinWords)}
                onChange={e => setUserDetailMinWords(parseInt(e.target.value.replace(/\D/g, '')) || 0)}
                className="search-input w-20"
              />
            </div>
            <span className="text-sm text-gray-500 pb-1">{userDetailFiltered.length.toLocaleString()} results</span>
          </div>

          <div className="mt-3 pt-3 border-t flex flex-wrap gap-x-4 gap-y-2 items-center text-sm text-gray-700">
            <span className="font-semibold text-gray-800">CONTEXT WINDOW</span>
            <label className="flex items-center gap-1.5">
              <span>Before</span>
              <input type="text" inputMode="numeric" value={String(userDetailCtxBefore)}
                onChange={e => setUserDetailCtxBefore(parseInt(e.target.value.replace(/\D/g, '')) || 0)}
                className="w-14 border border-gray-400 px-2 py-1 text-sm text-center text-gray-900" />
            </label>
            <label className="flex items-center gap-1.5">
              <span>After</span>
              <input type="text" inputMode="numeric" value={String(userDetailCtxAfter)}
                onChange={e => setUserDetailCtxAfter(parseInt(e.target.value.replace(/\D/g, '')) || 0)}
                className="w-14 border border-gray-400 px-2 py-1 text-sm text-center text-gray-900" />
            </label>
            <span className="text-xs text-gray-500">msgs · click "Show context" on any message</span>
          </div>
        </div>

        {/* Summary section */}
        <div className="bg-white rounded-2xl shadow overflow-hidden">
          <button
            onClick={() => setUserSumVisible(v => !v)}
            className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-50 transition-colors"
          >
            <div className="flex items-center gap-2">
              <svg className="w-4 h-4 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span className="text-sm font-semibold text-gray-800">Summarize with LLM</span>
              <span className="text-xs text-gray-500">{userDetailFiltered.length.toLocaleString()} messages selected</span>
            </div>
            <svg
              className={`w-4 h-4 text-gray-400 transition-transform ${userSumVisible ? 'rotate-180' : ''}`}
              fill="none" stroke="currentColor" viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {userSumVisible && (
            <div className="border-t border-gray-100 px-4 pb-4 pt-3 space-y-4">
              {/* Controls row */}
              <div className="flex flex-wrap items-end gap-3">
                <div className="flex flex-col gap-1">
                  <label className="filter-label">Model</label>
                  <select value={userSumModel} onChange={e => setUserSumModel(e.target.value)} className="search-input text-xs py-1">
                    <optgroup label="GPT-5.4">
                      <option value="gpt-5.4">GPT-5.4</option>
                      <option value="gpt-5.4-mini">GPT-5.4 mini</option>
                      <option value="gpt-5.4-nano">GPT-5.4 nano</option>
                    </optgroup>
                    <optgroup label="GPT-4.1">
                      <option value="gpt-4.1">GPT-4.1</option>
                      <option value="gpt-4.1-mini">GPT-4.1 mini</option>
                    </optgroup>
                    <optgroup label="GPT-4o">
                      <option value="gpt-4o">GPT-4o</option>
                      <option value="gpt-4o-mini">GPT-4o mini</option>
                    </optgroup>
                    <optgroup label="o-series">
                      <option value="o4-mini">o4-mini</option>
                      <option value="o3-mini">o3-mini</option>
                    </optgroup>
                  </select>
                </div>
                <div className="flex flex-col gap-1">
                  <label className="filter-label">Mode</label>
                  <div className="flex rounded-lg overflow-hidden border border-gray-300 text-xs font-semibold">
                    {(['cluster', 'all'] as const).map(mode => (
                      <button
                        key={mode}
                        onClick={() => setUserSumRetrievalMode(mode)}
                        className={`px-3 py-1.5 transition-colors ${userSumRetrievalMode === mode ? 'bg-indigo-700 text-white' : 'bg-white text-gray-600 hover:bg-gray-50'}`}
                      >
                        {mode === 'cluster' ? 'Cluster & Sample' : 'All Messages'}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="flex flex-col gap-1 flex-1 min-w-[220px]">
                  <label className="filter-label">Custom instructions (optional)</label>
                  <textarea
                    value={userSumInstructions}
                    onChange={e => setUserSumInstructions(e.target.value)}
                    rows={2}
                    placeholder="e.g. Focus on feature requests. Highlight recurring concerns."
                    className="search-input w-full resize-none"
                  />
                </div>
                <button
                  onClick={runUserSummary}
                  disabled={userSumLoading || userDetailFiltered.length === 0}
                  className="search-btn self-end"
                >
                  {userSumLoading ? 'Running…' : 'Run Summary'}
                </button>
              </div>
              <p className="text-xs text-gray-500">
                {userSumRetrievalMode === 'cluster'
                  ? 'Groups messages by topic and samples representatives. Best for large sets.'
                  : 'Passes all filtered messages directly to the model. Use for small sets (<100 msgs).'}
                {' '}Each message includes <strong>{userDetailCtxBefore} before</strong> / <strong>{userDetailCtxAfter} after</strong> context.
              </p>

              {/* Result */}
              {userSumLoading && !userSumResult && (
                <p className="text-sm text-gray-400 animate-pulse">Generating summary…</p>
              )}
              {userSumResult && (
                <div className="bg-gray-50 rounded-xl p-4 markdown-body" aria-live="polite">
                  <ReactMarkdown>{userSumResult + (userSumLoading ? ' ▋' : '')}</ReactMarkdown>
                </div>
              )}
              {userSumResult && !userSumLoading && (
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-xs font-semibold text-gray-600 uppercase tracking-wide">Follow-up Q&amp;A</h4>
                    {userSumFollowUp.length > 0 && (
                      <button onClick={() => setUserSumFollowUp([])} className="text-xs text-gray-400 hover:text-red-500">Clear Q&amp;A</button>
                    )}
                  </div>
                  <div className="space-y-2 mb-2 max-h-60 overflow-y-auto">
                    {userSumFollowUp.map((h, i) => (
                      <div key={i} className={h.role === 'user' ? 'flex justify-end' : ''}>
                        <div className={`rounded-xl px-3 py-2 text-sm ${h.role === 'user' ? 'bg-indigo-700 text-white max-w-xl' : 'bg-gray-50 border border-gray-200 text-gray-900 markdown-body'}`}>
                          {h.role === 'assistant' ? <ReactMarkdown>{h.content}</ReactMarkdown> : h.content}
                        </div>
                      </div>
                    ))}
                    {userSumFollowUpLoading && <div className="text-xs text-gray-400 animate-pulse">Generating…</div>}
                  </div>
                  <div className="flex gap-2">
                    <input
                      value={userSumFollowUpInput}
                      onChange={e => setUserSumFollowUpInput(e.target.value)}
                      onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runUserSumFollowUp() } }}
                      placeholder="Ask a follow-up question… (Enter to submit)"
                      className="search-input flex-1"
                    />
                    <button onClick={runUserSumFollowUp} disabled={userSumFollowUpLoading || !userSumFollowUpInput.trim()} className="search-btn">Ask</button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Messages list */}
        <div className="space-y-3">
          {userMsgsLoading ? (
            <div className="bg-white rounded-2xl shadow p-8 text-center text-sm text-gray-400">Loading messages…</div>
          ) : userDetailFiltered.length === 0 ? (
            <div className="bg-white rounded-2xl shadow p-8 text-center text-sm text-gray-400">No messages match the current filters.</div>
          ) : (
            <>
              {visibleMsgs.map(msg => (
                <MessageCard
                  key={msg.id}
                  msg={msg}
                  keyword={userDetailContains}
                  tokens={[]}
                  isBookmarked={bookmarkedIds.has(msg.id)}
                  isSelected={selectedIds.has(msg.id)}
                  onBookmarkToggle={handleBookmarkToggle}
                  onSelectToggle={toggleSelected}
                  ctxBefore={userDetailCtxBefore}
                  ctxAfter={userDetailCtxAfter}
                />
              ))}
              {userDetailVisible < userDetailFiltered.length && (
                <button
                  onClick={() => setUserDetailVisible(v => v + 50)}
                  className="w-full py-3 text-sm font-semibold text-indigo-600 hover:text-indigo-800 bg-white rounded-2xl shadow transition-colors"
                >
                  Load more ({userDetailFiltered.length - userDetailVisible} remaining)
                </button>
              )}
            </>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-5xl mx-auto px-3 sm:px-4 py-4 space-y-4">

      {/* Scope selector */}
      <section className="bg-white rounded-2xl shadow p-4">
        <div className="flex items-center justify-between mb-2">
          <h2 className="font-semibold text-sm text-gray-700 uppercase tracking-wide">Search Scope</h2>
          <div className="flex gap-2 items-center">
            <button onClick={() => setSelectedUploads(new Set())} className="text-xs font-semibold text-indigo-700 hover:underline">All</button>
            <span className="text-gray-400">|</span>
            <button onClick={() => setSelectedUploads(new Set(uploads.map(u => u.id)))} className="text-xs font-semibold text-indigo-700 hover:underline">None</button>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {uploads.length === 0 ? (
            <span className="text-xs text-gray-400 italic">No uploads yet — go to the Settings page to add data.</span>
          ) : (
            uploads.map((u) => {
              const active = selectedUploads.size === 0 || selectedUploads.has(u.id)
              return (
                <button
                  key={u.id}
                  onClick={() => toggleScopeChip(u.id)}
                  className={`scope-chip ${active ? 'scope-chip-on' : 'scope-chip-off'}`}
                  title={u.id}
                >
                  {u.filename}
                  <span className="text-[10px] opacity-70">{Number(u.row_count).toLocaleString()} rows</span>
                </button>
              )
            })
          )}
        </div>
        <p className="text-xs text-gray-600 mt-2">Toggle which uploaded files to include. All selected = search everything.</p>
      </section>

      {/* Search section */}
      <section
        className="bg-white rounded-2xl shadow p-5"
        onKeyDown={(e) => {
          if (e.key !== 'Enter') return
          const tag = (e.target as HTMLElement).tagName.toLowerCase()
          if (tag === 'button' || tag === 'textarea') return
          if (searchCat === 'users') doUserSearch()
          else doSearch()
        }}
      >
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-semibold text-sm text-gray-700 uppercase tracking-wide">Search</h2>
          <button
            onClick={() => setShowSearchPanel((v) => !v)}
            className="text-gray-400 hover:text-gray-600 text-xs"
          >
            {showSearchPanel ? '▲ Collapse' : '▼ Expand'}
          </button>
        </div>

        {showSearchPanel && (
          <>
            {/* Chat / Users top toggle */}
            <div className="flex gap-1 mb-4">
              <button onClick={() => handleCatSwitch('chat')} className={`subtab-btn${searchCat === 'chat' ? ' subtab-btn-active' : ''}`}>Search Chat</button>
              <button onClick={() => handleCatSwitch('users')} className={`subtab-btn${searchCat === 'users' ? ' subtab-btn-active' : ''}`}>Search Users</button>
            </div>

            {/* Chat search */}
            {searchCat === 'chat' && (
              <>
                <div className="flex border-b mb-4 overflow-x-auto">
                  {tabs.map((t) => (
                    <button key={t.id} onClick={() => handleTabSwitch(t.id)} className={`search-tab whitespace-nowrap${activeTab === t.id ? ' tab-active' : ''}`}>
                      {t.label}
                    </button>
                  ))}
                </div>

                {/* Keyword tab */}
                {activeTab === 'keyword' && (
                  <>
                    <div className="flex items-center gap-2 mb-3">
                      <span className="text-xs font-semibold text-gray-600">Match:</span>
                      <div className="inline-flex border border-gray-300 overflow-hidden">
                        {(['fuzzy', 'exact', 'any'] as MatchType[]).map((m) => (
                          <button key={m} onClick={() => setMatchType(m)} className={`range-mode-btn${matchType === m ? ' range-mode-active' : ''}`}>
                            {m === 'fuzzy' ? 'Fuzzy' : m === 'exact' ? 'Exact' : 'Any Word'}
                          </button>
                        ))}
                      </div>
                    </div>
                    <div className="flex gap-2 mb-4">
                      <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Keyword or phrase…" className={`${inputCls} flex-1`} />
                      <button onClick={doSearch} disabled={searching} className="search-btn">{searching ? 'Searching…' : 'Search'}</button>
                    </div>
                    <button onClick={() => setShowOptions((v) => !v)} className="flex items-center gap-1 text-xs text-gray-500 hover:text-indigo-600 mb-2 transition-colors">
                      <svg className={`w-3 h-3 transition-transform duration-200${showOptions ? ' rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 9l-7 7-7-7"/>
                      </svg>
                      <span>Options</span>
                    </button>
                    {showOptions && (
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-3">
                        <div className="flex flex-col gap-1">
                          <label className="filter-label">Username</label>
                          <input value={filterUsername} onChange={(e) => setFilterUsername(e.target.value)} placeholder="Optional username…" className={inputCls} />
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="filter-label">From Date</label>
                          <input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} className={inputCls} />
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="filter-label">To Date</label>
                          <input type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} className={inputCls} />
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="filter-label">Suno Team</label>
                          <select value={sunoTeam} onChange={(e) => setSunoTeam(e.target.value)} className={inputCls}>
                            <option value="all">All</option>
                            <option value="only">Only Suno Team</option>
                            <option value="exclude">Exclude Suno Team</option>
                          </select>
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="filter-label">Max Results</label>
                          <input type="text" inputMode="numeric" value={String(maxResults)} onChange={(e) => { const d = e.target.value.replace(/\D/g, ''); setMaxResults(d ? parseInt(d) : 200) }} className={inputCls} />
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="filter-label">Min Word Count</label>
                          <input type="text" inputMode="numeric" value={String(minWords)} onChange={(e) => { const d = e.target.value.replace(/\D/g, ''); setMinWords(d ? parseInt(d) : 0) }} className={inputCls} />
                        </div>
                      </div>
                    )}
                  </>
                )}

                {/* Semantic tab */}
                {activeTab === 'semantic' && (
                  <>
                    <div className="flex gap-2 mb-4">
                      <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Describe what you're looking for…" className={`${inputCls} flex-1`} />
                      <button onClick={doSearch} disabled={searching} className="search-btn">{searching ? 'Searching…' : 'Search'}</button>
                    </div>
                    <button onClick={() => setShowOptions((v) => !v)} className="flex items-center gap-1 text-xs text-gray-500 hover:text-indigo-600 mb-2 transition-colors">
                      <svg className={`w-3 h-3 transition-transform duration-200${showOptions ? ' rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 9l-7 7-7-7"/>
                      </svg>
                      <span>Options</span>
                    </button>
                    {showOptions && (
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-3">
                        <div className="flex flex-col gap-1">
                          <label className="filter-label">Username</label>
                          <input value={filterUsername} onChange={(e) => setFilterUsername(e.target.value)} placeholder="Optional username…" className={inputCls} />
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="filter-label">From Date</label>
                          <input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} className={inputCls} />
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="filter-label">To Date</label>
                          <input type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} className={inputCls} />
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="filter-label">Suno Team</label>
                          <select value={sunoTeam} onChange={(e) => setSunoTeam(e.target.value)} className={inputCls}>
                            <option value="all">All</option>
                            <option value="only">Only Suno Team</option>
                            <option value="exclude">Exclude Suno Team</option>
                          </select>
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="filter-label">Max Results</label>
                          <input type="text" inputMode="numeric" value={String(maxResults)} onChange={(e) => { const d = e.target.value.replace(/\D/g, ''); setMaxResults(d ? parseInt(d) : 200) }} className={inputCls} />
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="filter-label">Min Word Count</label>
                          <input type="text" inputMode="numeric" value={String(minWords)} onChange={(e) => { const d = e.target.value.replace(/\D/g, ''); setMinWords(d ? parseInt(d) : 0) }} className={inputCls} />
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="filter-label">Sort By</label>
                          <select value={sortBy} onChange={(e) => setSortBy(e.target.value)} className={inputCls}>
                            <option value="similarity">Similarity ↓</option>
                            <option value="date_asc">Date (oldest first)</option>
                            <option value="date_desc">Date (newest first)</option>
                          </select>
                        </div>
                      </div>
                    )}
                    <p className="text-xs text-gray-500 mt-3">Requires embedded data. Author &amp; date filter applied after retrieval.</p>
                  </>
                )}

                {/* Username tab */}
                {activeTab === 'username' && (
                  <>
                    <div className="flex gap-2 mb-4">
                      <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Username or partial match…" className={`${inputCls} flex-1`} />
                      <button onClick={doSearch} disabled={searching} className="search-btn">{searching ? 'Searching…' : 'Search'}</button>
                    </div>
                    <button onClick={() => setShowOptions((v) => !v)} className="flex items-center gap-1 text-xs text-gray-500 hover:text-indigo-600 mb-2 transition-colors">
                      <svg className={`w-3 h-3 transition-transform duration-200${showOptions ? ' rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 9l-7 7-7-7"/>
                      </svg>
                      <span>Options</span>
                    </button>
                    {showOptions && (
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-3">
                        <div className="flex flex-col gap-1">
                          <label className="filter-label">From Date</label>
                          <input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} className={inputCls} />
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="filter-label">To Date</label>
                          <input type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} className={inputCls} />
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="filter-label">Suno Team</label>
                          <select value={sunoTeam} onChange={(e) => setSunoTeam(e.target.value)} className={inputCls}>
                            <option value="all">All</option>
                            <option value="only">Only Suno Team</option>
                            <option value="exclude">Exclude Suno Team</option>
                          </select>
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="filter-label">Max Results</label>
                          <input type="text" inputMode="numeric" value={String(maxResults)} onChange={(e) => { const d = e.target.value.replace(/\D/g, ''); setMaxResults(d ? parseInt(d) : 200) }} className={inputCls} />
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="filter-label">Min Word Count</label>
                          <input type="text" inputMode="numeric" value={String(minWords)} onChange={(e) => { const d = e.target.value.replace(/\D/g, ''); setMinWords(d ? parseInt(d) : 0) }} className={inputCls} />
                        </div>
                      </div>
                    )}
                    <p className="text-xs text-gray-500 mt-3">Partial match, case-insensitive.</p>
                  </>
                )}

                {/* Time Range tab */}
                {activeTab === 'range' && (
                  <>
                    <div className="inline-flex border border-gray-300 overflow-hidden mb-3">
                      <button onClick={() => setRangeMode('exact')} className={`range-mode-btn${rangeMode === 'exact' ? ' range-mode-active' : ''}`}>Exact Date Range</button>
                      <button onClick={() => setRangeMode('month')} className={`range-mode-btn${rangeMode === 'month' ? ' range-mode-active' : ''}`}>By Month &amp; Year</button>
                    </div>
                    {rangeMode === 'exact' && (
                      <div className="grid grid-cols-2 gap-x-4 gap-y-3 mb-3">
                        <div className="flex flex-col gap-1">
                          <label className="filter-label">From Date</label>
                          <input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} className={inputCls} />
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="filter-label">To Date</label>
                          <input type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} className={inputCls} />
                        </div>
                      </div>
                    )}
                    {rangeMode === 'month' && (
                      <div className="grid grid-cols-2 gap-x-4 gap-y-3 mb-3">
                        <div className="flex flex-col gap-1">
                          <label className="filter-label">From Month</label>
                          <input type="month" value={dateFrom.slice(0, 7)} onChange={(e) => setDateFrom(e.target.value + '-01')} className={inputCls} />
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="filter-label">To Month</label>
                          <input type="month" value={dateTo.slice(0, 7)} onChange={(e) => setDateTo(e.target.value + '-31')} className={inputCls} />
                        </div>
                      </div>
                    )}
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-3 mb-4">
                      <div className="flex flex-col gap-1">
                        <label className="filter-label">Suno Team</label>
                        <select value={sunoTeam} onChange={(e) => setSunoTeam(e.target.value)} className={inputCls}>
                          <option value="all">All</option>
                          <option value="only">Only Suno Team</option>
                          <option value="exclude">Exclude Suno Team</option>
                        </select>
                      </div>
                      <div className="flex flex-col gap-1">
                        <label className="filter-label">Min Word Count</label>
                        <input type="text" inputMode="numeric" value={String(minWords)} onChange={(e) => { const d = e.target.value.replace(/\D/g, ''); setMinWords(d ? parseInt(d) : 0) }} className={inputCls} />
                      </div>
                    </div>
                    <button onClick={doSearch} disabled={searching} className="search-btn">{searching ? 'Searching…' : 'Search'}</button>
                    <p className="text-xs text-gray-500 mt-3">Returns all messages in the selected range. At least one date is required.</p>
                  </>
                )}
              </>
            )}

            {/* Users search */}
            {searchCat === 'users' && (
              <div className="space-y-3">
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-3 mb-4">
                  <div className="flex flex-col gap-1">
                    <label className="filter-label">From Date</label>
                    <input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} className={inputCls} />
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="filter-label">To Date</label>
                    <input type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} className={inputCls} />
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="filter-label">Suno Team</label>
                    <select value={sunoTeam} onChange={(e) => setSunoTeam(e.target.value)} className={inputCls}>
                      <option value="all">All</option>
                      <option value="only">Only Suno Team</option>
                      <option value="exclude">Exclude Suno Team</option>
                    </select>
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="filter-label">Min Word Count</label>
                    <input type="text" inputMode="numeric" value={String(minWords)} onChange={(e) => { const d = e.target.value.replace(/\D/g, ''); setMinWords(d ? parseInt(d) : 0) }} className={inputCls} />
                  </div>
                </div>
                <button onClick={doUserSearch} disabled={searching} className="search-btn mb-4">{searching ? 'Searching…' : 'Search'}</button>
                <p className="text-xs text-gray-500 mb-4">Returns all users active in the selected date range with message statistics. Leave dates empty to include all data.</p>
              </div>
            )}

            {/* Context window */}
            <div className="mt-4 pt-4 border-t flex flex-wrap gap-x-4 gap-y-2 items-center text-sm text-gray-700">
              <span className="font-semibold text-gray-800 w-full sm:w-auto">Context window:</span>
              <label className="flex items-center gap-1.5">
                <span className="text-gray-700">Before</span>
                <input type="text" inputMode="numeric" value={String(ctxBefore)} onChange={(e) => { const d = e.target.value.replace(/\D/g, ''); setCtxBefore(d ? parseInt(d) : 0) }} className="w-14 border border-gray-400 px-2 py-1 text-sm text-center text-gray-900" />
                <span className="text-gray-600 text-xs">msgs</span>
              </label>
              <label className="flex items-center gap-1.5">
                <span className="text-gray-700">After</span>
                <input type="text" inputMode="numeric" value={String(ctxAfter)} onChange={(e) => { const d = e.target.value.replace(/\D/g, ''); setCtxAfter(d ? parseInt(d) : 0) }} className="w-14 border border-gray-400 px-2 py-1 text-sm text-center text-gray-900" />
                <span className="text-gray-600 text-xs">msgs</span>
              </label>
              <span className="text-xs text-gray-600">(same CSV, original order)</span>
            </div>
          </>
        )}
      </section>

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-2xl px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* No results feedback */}
      {!searching && !error && searchCat === 'chat' && results.length === 0 && lastKeyword && (
        <div className="bg-gray-50 border border-gray-200 rounded-2xl px-4 py-3 text-sm text-gray-500 text-center">
          No results found for <span className="font-semibold text-gray-700">"{lastKeyword}"</span>. Try a different query or match type.
        </div>
      )}

      {/* Users results table */}
      {searchCat === 'users' && userStats.length > 0 && (
        <section className="bg-white rounded-2xl shadow p-5 flex flex-col" style={{ maxHeight: 'calc(100vh - 180px)' }}>
          {/* REFINE bar — always visible, never scrolls */}
          <div className="shrink-0 mb-3 bg-gray-50 rounded-xl border border-gray-200 px-4 py-3">
            <div className="flex flex-wrap gap-x-4 gap-y-2 items-end">
              <div className="flex flex-col gap-1">
                <label className="filter-label">Username</label>
                <input value={userNameFilter} onChange={(e) => setUserNameFilter(e.target.value)} placeholder="any" className="search-input text-xs py-1 w-32" />
              </div>
              <div className="flex flex-col gap-1">
                <label className="filter-label">First msg after</label>
                <input type="date" value={refineFirstMsgIn} onChange={(e) => setRefineFirstMsgIn(e.target.value)} className="search-input text-xs py-1" />
              </div>
              <div className="flex flex-col gap-1">
                <label className="filter-label">Last msg after</label>
                <input type="date" value={refineLastMsgFrom} onChange={(e) => setRefineLastMsgFrom(e.target.value)} className="search-input text-xs py-1" />
              </div>
              <div className="flex flex-col gap-1">
                <label className="filter-label">Min messages</label>
                <input type="text" inputMode="numeric" value={refineMinMessages} onChange={(e) => setRefineMinMessages(e.target.value.replace(/\D/g, ''))} placeholder="any" className="search-input text-xs py-1 w-20" />
              </div>
              <div className="flex flex-col gap-1">
                <label className="filter-label">Min weeks active</label>
                <input type="text" inputMode="numeric" value={refineMinWeeks} onChange={(e) => setRefineMinWeeks(e.target.value.replace(/\D/g, ''))} placeholder="any" className="search-input text-xs py-1 w-20" />
              </div>
              <div className="flex flex-col gap-1">
                <label className="filter-label">Avg words ≥</label>
                <input type="text" inputMode="numeric" value={refineAvgWordsMin} onChange={(e) => setRefineAvgWordsMin(e.target.value.replace(/[^0-9.]/g, ''))} placeholder="any" className="search-input text-xs py-1 w-20" />
              </div>
              <div className="flex flex-col gap-1">
                <label className="filter-label">≤</label>
                <input type="text" inputMode="numeric" value={refineAvgWordsMax} onChange={(e) => setRefineAvgWordsMax(e.target.value.replace(/[^0-9.]/g, ''))} placeholder="any" className="search-input text-xs py-1 w-20" />
              </div>
              <span className="self-end pb-1 text-xs text-gray-500 ml-1">
                {filteredUsers.length.toLocaleString()}{filteredUsers.length !== userStats.length ? ` / ${userStats.length.toLocaleString()}` : ''} users
              </span>
              {(userNameFilter || refineFirstMsgIn || refineLastMsgFrom || refineMinMessages || refineMinWeeks || refineAvgWordsMin || refineAvgWordsMax) && (
                <button
                  onClick={clearRefine}
                  className="self-end pb-1 text-xs text-gray-400 hover:text-red-500 transition-colors flex items-center gap-1"
                >
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                  Clear
                </button>
              )}
            </div>
          </div>

          <div className="flex-1 overflow-auto">
            <table className="w-full text-sm border-collapse">
              <thead className="sticky top-0 z-10">
                <tr className="bg-gray-50 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide select-none">
                  {(
                    [
                      { col: 'username', label: 'Username', cls: '' },
                      { col: 'total_messages', label: 'Messages', cls: 'text-right' },
                      { col: 'first_date', label: 'First Message', cls: '' },
                      { col: 'last_date', label: 'Last Message', cls: '' },
                      { col: 'avg_words', label: 'Avg Words', cls: 'text-right' },
                      { col: 'weeks_active', label: 'Weeks Active', cls: 'text-right' },
                      { col: 'pct_weeks', label: '% Weeks', cls: 'text-right' },
                    ] as { col: typeof userSortCol; label: string; cls: string }[]
                  ).map(({ col, label, cls }) => (
                    <th
                      key={col}
                      className={`px-3 py-2 border-b border-gray-200 cursor-pointer hover:text-gray-800 bg-gray-50 ${cls}`}
                      onClick={() => handleUserSortCol(col)}
                    >
                      {label} {userSortCol === col ? (userSortDir === 'asc' ? '↑' : '↓') : '↕'}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sortedUsers.map((u) => (
                  <tr key={u.username} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="px-3 py-2">
                      <button
                        onClick={() => openUserDetail(u)}
                        className="ubadge text-left hover:underline"
                        style={getUserStyle(u.username)}
                      >
                        {u.username}
                      </button>
                      {u.is_suno_team && (
                        <span className="ml-1.5 text-[10px] font-semibold bg-violet-100 text-violet-700 px-1.5 py-0.5 rounded-full">Team</span>
                      )}
                    </td>
                    <td className="px-3 py-2 text-right text-gray-600 tabular-nums">{u.total_messages.toLocaleString()}</td>
                    <td className="px-3 py-2 text-xs text-gray-500 tabular-nums">{u.first_date?.replace('T', ' ').slice(0, 19)}</td>
                    <td className="px-3 py-2 text-xs text-gray-500 tabular-nums">{u.last_date?.replace('T', ' ').slice(0, 19)}</td>
                    <td className="px-3 py-2 text-right text-gray-600 tabular-nums">{u.avg_words?.toFixed(1) ?? '—'}</td>
                    <td className="px-3 py-2 text-right text-gray-600 tabular-nums">
                      {u.weeks_active}/{u.total_weeks}
                    </td>
                    <td className="px-3 py-2 text-right text-gray-600 tabular-nums">{u.pct_weeks?.toFixed(1) ?? '—'}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {sortedUsers.length === 0 && (
              <p className="text-center text-sm text-gray-400 py-6">No users match the current filters.</p>
            )}
          </div>
          <p className="text-xs text-gray-500 mt-3">Click a username to open their message detail view.</p>
        </section>
      )}

      {/* Chat results */}
      {searchCat === 'chat' && results.length > 0 && (
        <>
          {/* Action toolbar */}
          <div className="bg-white rounded-2xl shadow px-4 py-2.5 flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-1.5 text-sm text-gray-700 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={selectedCount === results.length && results.length > 0}
                onChange={(e) => e.target.checked ? selectAll() : clearSelected()}
                className="accent-indigo-600"
              />
              Select all
            </label>
            <span className="text-xs text-gray-500">
              {selectedCount > 0
                ? `${selectedCount} selected`
                : `Showing ${Math.min(visibleCount, displayedResults.length).toLocaleString()} of ${results.length.toLocaleString()} results`}
            </span>
            <div className="flex gap-2 ml-auto flex-wrap">
              <button onClick={exportCSV} className="action-btn-primary">Export CSV</button>
              <button onClick={copySelected} disabled={selectedCount === 0} className="action-btn-primary">Copy Selected</button>
            </div>
          </div>

          {/* Analysis tab bar */}
          <div className="flex gap-2">
            <button
              onClick={() => toggleAnalysisTab('summarize')}
              className={`px-4 py-2 rounded-xl text-sm font-semibold border-2 transition-colors ${analysisTab === 'summarize' ? 'border-indigo-500 bg-indigo-50 text-indigo-800' : 'border-gray-200 bg-white text-gray-600 hover:border-indigo-300'}`}
            >
              Summarize &amp; Analyse
            </button>
            <button
              onClick={() => toggleAnalysisTab('viz')}
              className={`px-4 py-2 rounded-xl text-sm font-semibold border-2 transition-colors ${analysisTab === 'viz' ? 'border-purple-500 bg-purple-50 text-purple-800' : 'border-gray-200 bg-white text-gray-600 hover:border-purple-300'}`}
            >
              Visualization
            </button>
          </div>

          {/* Summarize & Analyse panel */}
          {analysisTab === 'summarize' && (
            <section className="bg-white rounded-2xl shadow p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-800">Summarize &amp; Analyse Results</h3>
                <span className="text-xs text-gray-500">{results.length.toLocaleString()} current results available for analysis.</span>
              </div>

              {/* Controls */}
              <div className="space-y-3 mb-4">
                <div className="flex flex-col gap-1">
                  <label className="filter-label">Custom instructions (optional)</label>
                  <textarea
                    value={summarizeQuery}
                    onChange={(e) => setSummarizeQuery(e.target.value)}
                    placeholder="Optional focus or instructions for the analysis…"
                    className={`${inputCls} min-h-[60px] resize-y`}
                    rows={2}
                  />
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-3">
                  <div className="flex flex-col gap-1">
                    <label className="filter-label">Retrieval mode</label>
                    <select value={srRetrievalMode} onChange={(e) => setSrRetrievalMode(e.target.value as 'cluster' | 'all')} className={inputCls}>
                      <option value="cluster">Cluster &amp; Sample (recommended)</option>
                      <option value="all">All messages (no clustering)</option>
                    </select>
                    <p className="text-xs text-gray-400 mt-0.5">{retrievalHint[srRetrievalMode]}</p>
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="filter-label">Model</label>
                    <select value={srModel} onChange={(e) => setSrModel(e.target.value)} className={inputCls}>
                      <optgroup label="GPT-5.4">
                        <option value="gpt-5.4">GPT-5.4 (default)</option>
                        <option value="gpt-5.4-pro">gpt-5.4-pro</option>
                        <option value="gpt-5.4-mini">gpt-5.4-mini</option>
                        <option value="gpt-5.4-nano">gpt-5.4-nano</option>
                      </optgroup>
                      <optgroup label="GPT-4.1">
                        <option value="gpt-4.1">GPT-4.1</option>
                      </optgroup>
                      <optgroup label="GPT-4o">
                        <option value="gpt-4o">GPT-4o</option>
                      </optgroup>
                      <optgroup label="o-series">
                        <option value="o4-mini">o4-mini</option>
                        <option value="o3">o3</option>
                        <option value="o3-mini">o3-mini</option>
                      </optgroup>
                    </select>
                  </div>
                </div>
              </div>

              <button
                onClick={handleSummarize}
                disabled={summarizeLoading}
                className="search-btn mb-4"
              >
                {summarizeLoading ? 'Generating…' : 'Summarize Results'}
              </button>

              {/* Pipeline log */}
              {(srLog.length > 0 || summarizeLoading) && (
                <div className="mb-4">
                  <button
                    onClick={() => setSrLogVisible((v) => !v)}
                    className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700 mb-1"
                  >
                    <svg className={`w-3 h-3 transition-transform${srLogVisible ? ' rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 9l-7 7-7-7"/>
                    </svg>
                    Pipeline log ({srLog.length} steps)
                  </button>
                  {srLogVisible && (
                    <div className="bg-gray-50 rounded-xl p-3 space-y-1 text-xs font-mono max-h-48 overflow-y-auto">
                      {srLog.map((entry, i) => (
                        <div key={i} className="log-entry flex gap-2 items-start">
                          <span className="log-icon shrink-0">{LOG_ICONS[entry.step] ?? '•'}</span>
                          <span className="log-label font-semibold text-gray-600 shrink-0 w-20">{entry.label}</span>
                          <span className="log-msg text-gray-500 flex-1">{entry.msg}</span>
                        </div>
                      ))}
                      {summarizeLoading && (
                        <div className="flex gap-2 items-center text-gray-400 animate-pulse">
                          <span>✨</span>
                          <span>Processing…</span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Result area */}
              {showSrResultsPanel && (
                <div>
                  {summarizeResult && (
                    <div className="bg-gray-50 rounded-xl p-4 markdown-body mb-4" aria-live="polite">
                      <ReactMarkdown>{summarizeResult + (summarizeLoading ? ' ▋' : '')}</ReactMarkdown>
                    </div>
                  )}
                  {!summarizeLoading && summarizeResult && (
                    <button
                      onClick={handleExportPDF}
                      className="action-btn-primary mb-4"
                    >
                      Export PDF
                    </button>
                  )}

                  {/* Follow-up Q&A */}
                  {summarizeResult && !summarizeLoading && (
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="text-xs font-semibold text-gray-600 uppercase tracking-wide">Follow-up Q&amp;A</h4>
                        {summarizeFollowUp.length > 0 && (
                          <button
                            onClick={() => setSummarizeFollowUp([])}
                            className="text-xs text-gray-400 hover:text-red-500"
                          >
                            Clear Q&amp;A
                          </button>
                        )}
                      </div>
                      <div className="space-y-2 mb-2 max-h-60 overflow-y-auto">
                        {summarizeFollowUp.map((h, i) => (
                          <div key={i} className={h.role === 'user' ? 'flex justify-end' : ''}>
                            <div className={`rounded-xl px-3 py-2 text-sm ${h.role === 'user' ? 'bg-indigo-700 text-white max-w-xl' : 'bg-gray-50 border border-gray-200 text-gray-900 markdown-body'}`}>
                              {h.role === 'assistant' ? <ReactMarkdown>{h.content}</ReactMarkdown> : h.content}
                            </div>
                          </div>
                        ))}
                        {summarizeFollowUpLoading && <div className="text-xs text-gray-400 animate-pulse">Generating…</div>}
                      </div>
                      <div className="flex gap-2">
                        <input
                          value={summarizeFollowUpInput}
                          onChange={(e) => setSummarizeFollowUpInput(e.target.value)}
                          onKeyDown={(e) => {
                            if ((e.key === 'Enter' && e.ctrlKey) || (e.key === 'Enter' && !e.shiftKey)) {
                              e.preventDefault()
                              handleSumFollowUp()
                            }
                          }}
                          placeholder="Ask a follow-up question… (Ctrl+Enter to submit)"
                          className={`${inputCls} flex-1`}
                        />
                        <button onClick={handleSumFollowUp} disabled={summarizeFollowUpLoading || !summarizeFollowUpInput.trim()} className="search-btn">Ask</button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </section>
          )}

          {/* Visualization panel */}
          {analysisTab === 'viz' && (
            <section className="bg-white rounded-2xl shadow p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-sm text-gray-700 uppercase tracking-wide">Message Volume Over Time</h3>
                <div className="flex items-center gap-2">
                  <div className="flex gap-0.5 bg-gray-100 rounded-lg p-0.5">
                    {(['month', 'week', 'day'] as TrendBucket[]).map((b) => (
                      <button
                        key={b}
                        onClick={() => { setTrendBucket(b); buildTrend(results, b) }}
                        className={`trend-bucket-btn${trendBucket === b ? ' trend-bucket-active' : ''}`}
                      >
                        {b.charAt(0).toUpperCase() + b.slice(1)}
                      </button>
                    ))}
                  </div>
                  <button
                    onClick={() => alert('Chart export requires Chart.js. PNG export not available with Recharts.')}
                    className="action-btn-primary text-xs"
                    title="Export chart as PNG"
                  >
                    PNG
                  </button>
                </div>
              </div>

              {/* Chart range banner */}
              {chartRange && (
                <div className="flex items-center gap-2 mb-3 bg-indigo-50 border border-indigo-200 rounded-xl px-3 py-2 text-sm text-indigo-800">
                  <span>
                    Showing {displayedResults.length} of {results.length} — {chartRange.label} ({chartRange.from} → {chartRange.to})
                  </span>
                  <button
                    onClick={() => {
                      setChartRange(null)
                      const tokens = computeFilterTokens(filterMode, filterText)
                      applyFilters(results, filterMode, filterText, null, tokens)
                    }}
                    className="ml-auto text-indigo-500 hover:text-indigo-800 font-bold"
                  >
                    ✕ Clear
                  </button>
                </div>
              )}

              {trendData.length > 0 ? (
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart
                    data={trendData}
                    onClick={(data) => {
                      if (data?.activePayload?.[0]) {
                        const pt = data.activePayload[0].payload as TrendPoint
                        handleChartBarClick(pt)
                      }
                    }}
                    style={{ cursor: 'pointer' }}
                  >
                    <XAxis dataKey="period" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                      {trendData.map((entry, i) => (
                        <Cell
                          key={i}
                          fill={chartRange?.from === entry.from ? '#4f46e5' : 'rgba(99,102,241,0.7)'}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-sm text-gray-400 text-center py-8">No trend data available.</p>
              )}
            </section>
          )}

          {/* Results filter bar — sticky so it stays visible while scrolling through cards */}
          <div className="bg-white rounded-2xl shadow px-4 py-3 sticky top-0 z-20">
            <div className="flex flex-wrap items-center gap-2">
              {/* Mode buttons */}
              <div className="inline-flex rounded-lg overflow-hidden border border-gray-200 shrink-0">
                <button
                  onClick={() => setFilterMode('exact')}
                  className={`px-3 py-1.5 text-xs font-semibold transition-colors ${filterMode === 'exact' ? 'bg-blue-600 text-white' : 'bg-white text-gray-600 hover:bg-gray-50'}`}
                >
                  Exact
                </button>
                <button
                  onClick={() => setFilterMode('any')}
                  className={`px-3 py-1.5 text-xs font-semibold border-l border-gray-200 transition-colors ${filterMode === 'any' ? 'bg-emerald-600 text-white' : 'bg-white text-gray-600 hover:bg-gray-50'}`}
                >
                  Any Word
                </button>
                <button
                  onClick={() => setFilterMode('fuzzy')}
                  className={`px-3 py-1.5 text-xs font-semibold border-l border-gray-200 transition-colors ${filterMode === 'fuzzy' ? 'bg-amber-500 text-white' : 'bg-white text-gray-600 hover:bg-gray-50'}`}
                >
                  Fuzzy
                </button>
                <button
                  onClick={() => setFilterMode('semantic')}
                  className={`px-3 py-1.5 text-xs font-semibold border-l border-gray-200 transition-colors ${filterMode === 'semantic' ? 'bg-indigo-600 text-white' : 'bg-white text-gray-600 hover:bg-gray-50'}`}
                >
                  Semantic
                </button>
              </div>

              {/* Filter input */}
              <div className="flex items-center gap-1.5 flex-1 min-w-0">
                <svg className="w-3.5 h-3.5 text-gray-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2a1 1 0 01-.293.707L13 13.414V19a1 1 0 01-.553.894l-4 2A1 1 0 017 21v-7.586L3.293 6.707A1 1 0 013 6V4z" />
                </svg>
                <input
                  value={filterText}
                  onChange={(e) => setFilterText(e.target.value)}
                  placeholder={filterPlaceholder[filterMode]}
                  className="search-input flex-1 min-w-0 text-xs py-1.5"
                />
                {filterLoading && (
                  <svg className="w-3.5 h-3.5 text-indigo-500 animate-spin shrink-0" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                )}
                {filterText && (
                  <button
                    onClick={() => {
                      setFilterText('')
                      setDisplayedResults(results)
                      setFilterCount(null)
                      setFilterTokens([])
                    }}
                    className="text-gray-400 hover:text-gray-700 shrink-0 font-bold text-sm"
                    title="Clear filter"
                  >
                    ✕
                  </button>
                )}
              </div>

              {/* Count label */}
              {filterCount && (
                <span className="text-xs text-gray-500 shrink-0 tabular-nums">
                  {filterCount.shown} of {filterCount.total}
                </span>
              )}
            </div>
          </div>

          {/* Bookmark error toast */}
          {bookmarkError && (
            <div className="bg-red-50 border border-red-200 rounded-2xl px-4 py-3 text-sm text-red-700 flex items-center justify-between">
              <span>Bookmark failed: {bookmarkError}</span>
              <button onClick={() => setBookmarkError(null)} className="text-red-400 hover:text-red-700 font-bold ml-3">✕</button>
            </div>
          )}

          {/* Message cards */}
          <div className="space-y-3" id="results-container">
            {displayedResults.slice(0, visibleCount).map((msg) => (
              <MessageCard
                key={msg.id}
                msg={msg}
                keyword={activeKeyword}
                tokens={activeTokens}
                isBookmarked={bookmarkedIds.has(msg.id)}
                isSelected={selectedIds.has(msg.id)}
                onBookmarkToggle={handleBookmarkToggle}
                onSelectToggle={toggleSelected}
                ctxBefore={ctxBefore}
                ctxAfter={ctxAfter}
              />
            ))}
            {displayedResults.length === 0 && filterText.trim() && (
              <div className="text-center py-8 text-gray-400 text-sm">No results match the current filter.</div>
            )}
            {/* Sentinel — IntersectionObserver target for infinite scroll */}
            <div ref={sentinelRef} />
            {visibleCount < displayedResults.length ? (
              <div className="py-4 text-center text-sm text-gray-400 animate-pulse">
                Loading more… ({Math.min(visibleCount, displayedResults.length).toLocaleString()} of {displayedResults.length.toLocaleString()})
              </div>
            ) : displayedResults.length > 50 && (
              <div className="py-3 text-center text-xs text-gray-400">
                All {displayedResults.length.toLocaleString()} results shown
              </div>
            )}
          </div>
        </>
      )}

    </div>
  )
}
