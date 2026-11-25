import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { apiFetch } from '../api/client'

// ── Types ──────────────────────────────────────────────────────────────────

interface SearchHit {
  id: number
  username: string
  date: string
  content: string
}

interface ThreadMessage {
  id: number
  msg_uuid: string
  username: string
  date: string
  content: string
  row_index: number | null
  is_target: boolean
}

interface Code {
  id: number
  name: string
  color: string
  description: string
  category_id: number | null
}

interface Category {
  id: number
  name: string
  color: string
  parent_id: number | null
}

interface Highlight {
  highlight_id: number
  bookmark_id: number
  code_id: number
  highlighted_text: string
  code_name: string
  color: string
}

interface Popover {
  x: number
  y: number
  text: string
  msgId: number
}

// ── Helpers ────────────────────────────────────────────────────────────────

// Builds non-overlapping render segments for a message's content given its
// highlights. Best-effort: matches the first occurrence of each highlighted
// substring. Good enough for an inline "code this span" MVP; doesn't yet
// handle overlapping codes on the exact same characters.
function buildSegments(content: string, highlights: Highlight[]) {
  type Range = { start: number; end: number; hl: Highlight }
  const ranges: Range[] = []
  for (const hl of highlights) {
    if (!hl.highlighted_text) continue
    const idx = content.indexOf(hl.highlighted_text)
    if (idx === -1) continue
    ranges.push({ start: idx, end: idx + hl.highlighted_text.length, hl })
  }
  ranges.sort((a, b) => a.start - b.start)

  const segments: { text: string; hl?: Highlight }[] = []
  let cursor = 0
  for (const r of ranges) {
    if (r.start < cursor) continue // skip overlaps for now
    if (r.start > cursor) segments.push({ text: content.slice(cursor, r.start) })
    segments.push({ text: content.slice(r.start, r.end), hl: r.hl })
    cursor = r.end
  }
  if (cursor < content.length) segments.push({ text: content.slice(cursor) })
  return segments
}

export default function CodingWorkspacePage() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchHit[]>([])
  const [searching, setSearching] = useState(false)
  const [activeId, setActiveId] = useState<number | null>(null)
  const [thread, setThread] = useState<ThreadMessage[]>([])
  const [threadLoading, setThreadLoading] = useState(false)
  const [highlightsByMsg, setHighlightsByMsg] = useState<Record<number, Highlight[]>>({})

  const [codes, setCodes] = useState<Code[]>([])
  const [categories, setCategories] = useState<Category[]>([])
  const [codeFilter, setCodeFilter] = useState('')
  const [focusCodeId, setFocusCodeId] = useState<number | null>(null)

  const [popover, setPopover] = useState<Popover | null>(null)
  const [codePickerFilter, setCodePickerFilter] = useState('')
  const [newCodeName, setNewCodeName] = useState('')
  const [busy, setBusy] = useState(false)

  const containerRef = useRef<HTMLDivElement>(null)

  // Initial codebook load
  useEffect(() => {
    apiFetch<Code[]>('/codes').then(setCodes).catch(() => {})
    apiFetch<Category[]>('/code-categories').then(setCategories).catch(() => {})
  }, [])

  const runSearch = useCallback(async () => {
    if (!query.trim()) { setResults([]); return }
    setSearching(true)
    try {
      const hits = await apiFetch<SearchHit[]>(`/search/keyword?q=${encodeURIComponent(query)}&limit=40`)
      setResults(hits)
    } catch {
      setResults([])
    } finally {
      setSearching(false)
    }
  }, [query])

  const loadHighlightsFor = useCallback(async (msgIds: number[]) => {
    const entries = await Promise.all(
      msgIds.map(async (id) => {
        try {
          const hl = await apiFetch<Highlight[]>(`/codes/by-message/${id}`)
          return [id, hl] as const
        } catch {
          return [id, [] as Highlight[]] as const
        }
      }),
    )
    setHighlightsByMsg((prev) => {
      const next = { ...prev }
      for (const [id, hl] of entries) next[id] = hl
      return next
    })
  }, [])

  const openMessage = useCallback(async (msgId: number) => {
    setActiveId(msgId)
    setThreadLoading(true)
    setPopover(null)
    try {
      const data = await apiFetch<ThreadMessage[]>(`/context/${msgId}?before=20&after=20`)
      setThread(data)
      await loadHighlightsFor(data.map((m) => m.id))
    } catch {
      setThread([])
    } finally {
      setThreadLoading(false)
    }
  }, [loadHighlightsFor])

  // Text-selection handling inside the transcript
  const handleMouseUp = useCallback(() => {
    const sel = window.getSelection()
    if (!sel || sel.isCollapsed || !sel.toString().trim()) { return }
    const text = sel.toString().trim()
    if (text.length < 2) return

    let node = sel.anchorNode as HTMLElement | null
    while (node && !(node instanceof HTMLElement && node.dataset.msgId)) {
      node = node.parentElement
    }
    if (!node) return
    const msgId = Number(node.dataset.msgId)

    const range = sel.getRangeAt(0)
    const rect = range.getBoundingClientRect()
    setPopover({ x: rect.left + rect.width / 2, y: rect.top, text, msgId })
    setCodePickerFilter('')
  }, [])

  const applyCode = useCallback(async (codeId: number) => {
    if (!popover) return
    setBusy(true)
    try {
      const bm = await apiFetch<{ id: number }>(`/bookmarks/by-msg/${popover.msgId}`, { method: 'PUT' })
      await apiFetch(`/bookmarks/${bm.id}/highlights`, {
        method: 'POST',
        body: JSON.stringify({ code_id: codeId, highlighted_text: popover.text }),
      })
      await loadHighlightsFor([popover.msgId])
      window.getSelection()?.removeAllRanges()
      setPopover(null)
    } catch (err) {
      console.error(err)
    } finally {
      setBusy(false)
    }
  }, [popover, loadHighlightsFor])

  const createAndApplyCode = useCallback(async () => {
    if (!newCodeName.trim()) return
    setBusy(true)
    try {
      const created = await apiFetch<Code>('/codes', {
        method: 'POST',
        body: JSON.stringify({ name: newCodeName.trim() }),
      })
      setCodes((prev) => [...prev, created].sort((a, b) => a.name.localeCompare(b.name)))
      setNewCodeName('')
      await applyCode(created.id)
    } finally {
      setBusy(false)
    }
  }, [newCodeName, applyCode])

  const removeHighlight = useCallback(async (hl: Highlight) => {
    await apiFetch(`/bookmarks/${hl.bookmark_id}/highlights/${hl.highlight_id}`, { method: 'DELETE' })
    setHighlightsByMsg((prev) => {
      const next = { ...prev }
      for (const id of Object.keys(next)) {
        next[Number(id)] = next[Number(id)].filter((h) => h.highlight_id !== hl.highlight_id)
      }
      return next
    })
  }, [])

  // Close popover on outside click / escape
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setPopover(null) }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [])

  const filteredCodes = useMemo(
    () => codes.filter((c) => c.name.toLowerCase().includes(codeFilter.toLowerCase())),
    [codes, codeFilter],
  )

  const codesByCategory = useMemo(() => {
    const map = new Map<number | null, Code[]>()
    for (const c of filteredCodes) {
      const key = c.category_id
      if (!map.has(key)) map.set(key, [])
      map.get(key)!.push(c)
    }
    return map
  }, [filteredCodes])

  const quickPickCodes = useMemo(
    () => codes.filter((c) => c.name.toLowerCase().includes(codePickerFilter.toLowerCase())).slice(0, 8),
    [codes, codePickerFilter],
  )

  return (
    <div className="h-[calc(100vh-56px)] sm:h-[calc(100vh-64px)] flex bg-gray-50 -mx-3 sm:-mx-4 -mt-4 overflow-hidden">
      {/* ── Left: Sources ─────────────────────────────────────────────── */}
      <aside className="w-[260px] shrink-0 border-r border-gray-200 bg-white flex flex-col">
        <div className="p-3 border-b border-gray-100">
          <h2 className="text-xs font-bold uppercase tracking-wide text-gray-500 mb-2">Sources</h2>
          <div className="flex gap-1.5">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && runSearch()}
              placeholder="Find a message…"
              className="search-input text-sm flex-1"
            />
            <button onClick={runSearch} className="search-btn px-3 text-sm" disabled={searching}>
              {searching ? '…' : 'Go'}
            </button>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto">
          {results.length === 0 && (
            <p className="text-xs text-gray-400 text-center mt-8 px-4">
              Search for a message above to open it as a document in the center panel.
            </p>
          )}
          {results.map((r) => (
            <button
              key={r.id}
              onClick={() => openMessage(r.id)}
              className={`w-full text-left px-3 py-2.5 border-b border-gray-50 hover:bg-indigo-50 transition-colors ${activeId === r.id ? 'bg-indigo-50 border-l-2 border-l-indigo-700' : ''}`}
            >
              <div className="flex items-center justify-between mb-0.5">
                <span className="text-xs font-semibold text-indigo-700 truncate">{r.username}</span>
                <span className="text-[10px] text-gray-400 shrink-0 ml-1">{r.date?.slice(0, 10)}</span>
              </div>
              <p className="text-xs text-gray-600 line-clamp-2">{r.content}</p>
            </button>
          ))}
        </div>
      </aside>

      {/* ── Center: Document browser ──────────────────────────────────── */}
      <main className="flex-1 flex flex-col min-w-0">
        <div className="bg-white border-b border-gray-200 px-4 py-2 flex items-center justify-between">
          <h1 className="text-sm font-bold text-gray-800">Document Browser</h1>
          {activeId && <span className="text-xs text-gray-400">Message #{activeId} &middot; ±20 context</span>}
        </div>
        <div ref={containerRef} onMouseUp={handleMouseUp} className="flex-1 overflow-y-auto px-4 py-3">
          {!activeId && !threadLoading && (
            <div className="h-full flex items-center justify-center text-center text-gray-400 text-sm px-8">
              Open a message from the Sources panel, then select any text in the transcript<br />to attach a code — just like Atlas.ti's document browser.
            </div>
          )}
          {threadLoading && (
            <div className="h-full flex items-center justify-center">
              <div className="animate-spin rounded-full h-8 w-8 border-4 border-indigo-700 border-t-transparent" />
            </div>
          )}
          {!threadLoading && thread.map((m) => {
            const hl = (highlightsByMsg[m.id] || []).filter((h) => !focusCodeId || h.code_id === focusCodeId)
            const segments = buildSegments(m.content || '', hl)
            return (
              <div
                key={m.id}
                data-msg-id={m.id}
                className={`mb-2.5 rounded-lg px-3 py-2 ${m.is_target ? 'bg-amber-50 ring-1 ring-amber-200' : 'bg-white'} border border-gray-100`}
              >
                <div className="flex items-baseline gap-2 mb-0.5">
                  <span className="text-xs font-bold text-indigo-700">{m.username}</span>
                  <span className="text-[10px] text-gray-400">{m.date}</span>
                </div>
                <p className="text-sm text-gray-800 leading-relaxed whitespace-pre-wrap select-text">
                  {segments.map((seg, i) =>
                    seg.hl ? (
                      <mark
                        key={i}
                        onClick={(e) => { e.stopPropagation(); if (confirm(`Remove code "${seg.hl!.code_name}" from this span?`)) removeHighlight(seg.hl!) }}
                        title={`Coded: ${seg.hl.code_name} (click to remove)`}
                        style={{ backgroundColor: `${seg.hl.color}33`, borderBottom: `2px solid ${seg.hl.color}`, cursor: 'pointer' }}
                        className="rounded-sm px-0.5"
                      >
                        {seg.text}
                      </mark>
                    ) : (
                      <span key={i}>{seg.text}</span>
                    ),
                  )}
                </p>
              </div>
            )
          })}
        </div>
      </main>

      {/* ── Right: Codebook ───────────────────────────────────────────── */}
      <aside className="w-[290px] shrink-0 border-l border-gray-200 bg-white flex flex-col">
        <div className="p-3 border-b border-gray-100">
          <h2 className="text-xs font-bold uppercase tracking-wide text-gray-500 mb-2">Codebook</h2>
          <input
            value={codeFilter}
            onChange={(e) => setCodeFilter(e.target.value)}
            placeholder="Filter codes…"
            className="search-input text-sm w-full"
          />
          {focusCodeId && (
            <button onClick={() => setFocusCodeId(null)} className="mt-2 text-xs text-indigo-700 hover:underline">
              &times; Clear focus filter
            </button>
          )}
        </div>
        <div className="flex-1 overflow-y-auto">
          {[...codesByCategory.entries()].map(([catId, list]) => {
            const cat = categories.find((c) => c.id === catId)
            return (
              <div key={String(catId)} className="border-b border-gray-50">
                <div className="px-3 py-1.5 bg-gray-50 text-[11px] font-bold uppercase tracking-wide text-gray-500">
                  {cat ? cat.name : 'Uncategorized'}
                </div>
                {list.map((c) => (
                  <button
                    key={c.id}
                    onClick={() => setFocusCodeId(focusCodeId === c.id ? null : c.id)}
                    className={`w-full flex items-center gap-2 px-3 py-1.5 text-left hover:bg-gray-50 ${focusCodeId === c.id ? 'bg-indigo-50' : ''}`}
                  >
                    <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: c.color }} />
                    <span className="text-sm text-gray-700 truncate">{c.name}</span>
                  </button>
                ))}
              </div>
            )
          })}
          {codes.length === 0 && (
            <p className="text-xs text-gray-400 text-center mt-6 px-3">No codes yet. Select text in a message to create your first code.</p>
          )}
        </div>
      </aside>

      {/* ── Floating "code this selection" popover ───────────────────── */}
      {popover && (
        <div
          className="fixed z-50 bg-white rounded-xl shadow-2xl border border-gray-200 p-2 w-64"
          style={{ left: Math.max(8, popover.x - 128), top: Math.max(8, popover.y - 10 - 220) }}
        >
          <p className="text-[11px] text-gray-500 px-1 pb-1.5 border-b border-gray-100 mb-1.5 truncate">
            “{popover.text}”
          </p>
          <input
            autoFocus
            value={codePickerFilter}
            onChange={(e) => setCodePickerFilter(e.target.value)}
            placeholder="Search codes…"
            className="search-input text-xs w-full mb-1.5"
          />
          <div className="max-h-40 overflow-y-auto">
            {quickPickCodes.map((c) => (
              <button
                key={c.id}
                disabled={busy}
                onClick={() => applyCode(c.id)}
                className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-gray-50 text-left disabled:opacity-50"
              >
                <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: c.color }} />
                <span className="text-sm text-gray-700 truncate">{c.name}</span>
              </button>
            ))}
            {quickPickCodes.length === 0 && codePickerFilter && (
              <p className="text-xs text-gray-400 px-2 py-1">No matching code.</p>
            )}
          </div>
          <div className="border-t border-gray-100 mt-1.5 pt-1.5 flex gap-1.5">
            <input
              value={newCodeName}
              onChange={(e) => setNewCodeName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && createAndApplyCode()}
              placeholder="+ New code name…"
              className="search-input text-xs flex-1"
            />
            <button onClick={createAndApplyCode} disabled={busy || !newCodeName.trim()} className="action-btn-primary text-xs px-2">
              Add
            </button>
          </div>
          <button onClick={() => setPopover(null)} className="w-full text-center text-[11px] text-gray-400 hover:text-gray-600 mt-1">
            Cancel
          </button>
        </div>
      )}
    </div>
  )
}
