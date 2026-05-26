import { useState, useRef, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import { streamEvents } from '../api/client'

type SubTab = 'chat' | 'profile'
type DateMode = 'exact' | 'month'

interface LogEntry {
  type: 'info' | 'ok' | 'warn' | 'err'
  text: string
}

function LogPanel({ entries, onToggle, visible }: { entries: LogEntry[]; onToggle: () => void; visible: boolean }) {
  return (
    <section className="bg-white rounded-2xl shadow p-5">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Retrieval &amp; Pipeline Log</h3>
        <button onClick={onToggle} className="text-xs font-semibold text-gray-500 hover:text-gray-800">
          {visible ? '▲ Hide' : '▼ Show'}
        </button>
      </div>
      {visible && (
        <div className="space-y-0.5" aria-live="polite">
          {entries.map((e, i) => (
            <div key={i} className={`log-${e.type} block`}>{e.text}</div>
          ))}
          {entries.length === 0 && <p className="text-xs text-gray-400">No log entries yet.</p>}
        </div>
      )}
    </section>
  )
}

interface FollowUpSectionProps {
  history: { role: 'user' | 'assistant'; content: string }[]
  onAsk: (q: string) => void
  onClear: () => void
  loading: boolean
}

function FollowUpSection({ history, onAsk, onClear, loading }: FollowUpSectionProps) {
  const [input, setInput] = useState('')

  const handleAsk = () => {
    if (!input.trim() || loading) return
    onAsk(input.trim())
    setInput('')
  }

  return (
    <section className="bg-white rounded-2xl shadow p-5">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-700">Follow-up Questions</h3>
        <button
          onClick={onClear}
          className="text-xs font-semibold text-red-700 hover:text-red-900 border border-red-300 hover:border-red-500 px-3 py-1 rounded-lg transition-colors"
        >
          Clear Q&amp;A
        </button>
      </div>

      <div className="space-y-3 mb-3 max-h-[28rem] overflow-y-auto pr-1">
        {history.map((h, i) => (
          <div key={i} className={h.role === 'user' ? 'flex justify-end' : ''}>
            <div className={`rounded-xl px-4 py-3 text-sm ${h.role === 'user' ? 'bg-indigo-700 text-white max-w-xl' : 'bg-gray-50 border border-gray-200 text-gray-900 markdown-body w-full'}`}>
              {h.role === 'assistant' ? (
                <ReactMarkdown>{h.content}</ReactMarkdown>
              ) : h.content}
            </div>
          </div>
        ))}
        {loading && (
          <div className="text-xs text-gray-400 animate-pulse">Generating answer…</div>
        )}
      </div>

      <div className="flex gap-2 items-end">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && e.ctrlKey) { e.preventDefault(); handleAsk() } }}
          rows={2}
          placeholder="Ask a follow-up question…"
          className="search-input flex-1 resize-none min-w-0"
        />
        <button onClick={handleAsk} disabled={loading || !input.trim()} className="search-btn self-end shrink-0">
          Ask
        </button>
      </div>
      <p className="text-xs text-gray-500 mt-1.5">Press Ctrl+Enter or click Ask</p>
    </section>
  )
}

const MODEL_OPTIONS = (
  <>
    <optgroup label="GPT-5.4">
      <option value="gpt-5.4">GPT-5.4</option>
      <option value="gpt-5.4-pro">GPT-5.4 pro</option>
      <option value="gpt-5.4-mini">GPT-5.4 mini</option>
      <option value="gpt-5.4-nano">GPT-5.4 nano</option>
    </optgroup>
    <optgroup label="GPT-5">
      <option value="gpt-5">GPT-5</option>
      <option value="gpt-5-mini">GPT-5 mini</option>
      <option value="gpt-5-nano">GPT-5 nano</option>
    </optgroup>
    <optgroup label="GPT-4.1">
      <option value="gpt-4.1">GPT-4.1</option>
      <option value="gpt-4.1-mini">GPT-4.1 mini</option>
      <option value="gpt-4.1-nano">GPT-4.1 nano</option>
    </optgroup>
    <optgroup label="GPT-4o">
      <option value="gpt-4o">GPT-4o</option>
      <option value="gpt-4o-mini">GPT-4o mini</option>
    </optgroup>
    <optgroup label="o-series (reasoning)">
      <option value="o4-mini">o4 mini</option>
      <option value="o3">o3</option>
      <option value="o3-mini">o3 mini</option>
      <option value="o1">o1</option>
      <option value="o1-mini">o1 mini</option>
    </optgroup>
  </>
)

export default function ChatPage() {
  const [subTab, setSubTab] = useState<SubTab>('chat')

  // RAG Chat state
  const [sumUsername, setSumUsername] = useState('')
  const [sumDateMode, setSumDateMode] = useState<DateMode>('exact')
  const [sumDateFrom, setSumDateFrom] = useState('')
  const [sumDateTo, setSumDateTo] = useState('')
  const [sumMonthFrom, setSumMonthFrom] = useState('')
  const [sumMonthTo, setSumMonthTo] = useState('')
  const [sumMinWords, setSumMinWords] = useState(0)
  const [sumSuno, setSumSuno] = useState('all')
  const [sumMode, setSumMode] = useState('cluster')
  const [sumPrompt, setSumPrompt] = useState('')
  const [sumModel, setSumModel] = useState('gpt-5.4')
  const [sumLoading, setSumLoading] = useState(false)
  const [sumLog, setSumLog] = useState<LogEntry[]>([])
  const [sumLogVisible, setSumLogVisible] = useState(true)
  const [sumResult, setSumResult] = useState('')
  const [sumResultVisible, setSumResultVisible] = useState(false)
  const [sumFollowUpHistory, setSumFollowUpHistory] = useState<{ role: 'user' | 'assistant'; content: string }[]>([])
  const [sumFollowUpLoading, setSumFollowUpLoading] = useState(false)
  const [sumFormCollapsed, setSumFormCollapsed] = useState(false)
  const sumStopRef = useRef<(() => void) | null>(null)

  // Profile state
  const [profUsername, setProfUsername] = useState('')
  const [profDateMode, setProfDateMode] = useState<DateMode>('exact')
  const [profDateFrom, setProfDateFrom] = useState('')
  const [profDateTo, setProfDateTo] = useState('')
  const [profMonthFrom, setProfMonthFrom] = useState('')
  const [profMonthTo, setProfMonthTo] = useState('')
  const [profMode, setProfMode] = useState('cluster')
  const [profPrompt, setProfPrompt] = useState('')
  const [profModel, setProfModel] = useState('gpt-5.4')
  const [profLoading, setProfLoading] = useState(false)
  const [profLog, setProfLog] = useState<LogEntry[]>([])
  const [profLogVisible, setProfLogVisible] = useState(true)
  const [profResult, setProfResult] = useState('')
  const [profResultVisible, setProfResultVisible] = useState(false)
  const [profFollowUpHistory, setProfFollowUpHistory] = useState<{ role: 'user' | 'assistant'; content: string }[]>([])
  const [profFollowUpLoading, setProfFollowUpLoading] = useState(false)
  const [profFormCollapsed, setProfFormCollapsed] = useState(false)
  const profStopRef = useRef<(() => void) | null>(null)

  const appendSumLog = (type: LogEntry['type'], text: string) =>
    setSumLog((prev) => [...prev, { type, text }])
  const appendProfLog = (type: LogEntry['type'], text: string) =>
    setProfLog((prev) => [...prev, { type, text }])

  const handleSumGenerate = useCallback(() => {
    setSumLog([])
    setSumResult('')
    setSumResultVisible(false)
    setSumLoading(true)

    const body: Record<string, unknown> = {
      model: sumModel,
      retrieval_mode: sumMode,
    }
    if (sumUsername) body.username = sumUsername
    if (sumPrompt) body.custom_instructions = sumPrompt
    if (sumDateMode === 'exact') {
      if (sumDateFrom) body.date_from = sumDateFrom
      if (sumDateTo) body.date_to = sumDateTo
    } else {
      if (sumMonthFrom) body.date_from = sumMonthFrom + '-01'
      if (sumMonthTo) body.date_to = sumMonthTo + '-31'
    }
    if (sumMinWords > 0) body.min_words = sumMinWords
    if (sumSuno !== 'all') body.suno_team = sumSuno

    let accumulated = ''
    sumStopRef.current = streamEvents(
      '/summarize',
      body,
      (event) => {
        const type = event.type as string
        if (type === 'log') {
          const level = (event.level as string) || 'info'
          appendSumLog(
            level === 'error' ? 'err' : level === 'warning' ? 'warn' : level === 'success' ? 'ok' : 'info',
            (event.message as string) || '',
          )
        } else if (type === 'chunk') {
          accumulated += (event.content as string) || ''
          setSumResult(accumulated)
          setSumResultVisible(true)
        } else if (type === 'result') {
          accumulated = (event.content as string) || accumulated
          setSumResult(accumulated)
          setSumResultVisible(true)
        }
      },
      () => {
        setSumLoading(false)
        setSumFormCollapsed(true)
      },
      (err) => {
        appendSumLog('err', err.message)
        setSumLoading(false)
      },
    )
  }, [sumModel, sumMode, sumUsername, sumPrompt, sumDateMode, sumDateFrom, sumDateTo, sumMonthFrom, sumMonthTo, sumMinWords, sumSuno])

  const handleSumFollowUp = useCallback((q: string) => {
    setSumFollowUpHistory((prev) => [...prev, { role: 'user', content: q }])
    setSumFollowUpLoading(true)
    let answer = ''
    streamEvents(
      '/summarize/followup',
      { question: q, summary: sumResult, history: sumFollowUpHistory },
      (event) => {
        if (event.type === 'chunk') {
          answer += (event.content as string) || ''
          setSumFollowUpHistory((prev) => {
            const last = prev[prev.length - 1]
            if (last?.role === 'assistant') {
              return [...prev.slice(0, -1), { role: 'assistant', content: answer }]
            }
            return [...prev, { role: 'assistant', content: answer }]
          })
        } else if (event.type === 'result') {
          answer = (event.content as string) || answer
          setSumFollowUpHistory((prev) => {
            const last = prev[prev.length - 1]
            if (last?.role === 'assistant') {
              return [...prev.slice(0, -1), { role: 'assistant', content: answer }]
            }
            return [...prev, { role: 'assistant', content: answer }]
          })
        }
      },
      () => setSumFollowUpLoading(false),
      (err) => {
        setSumFollowUpHistory((prev) => [...prev, { role: 'assistant', content: `Error: ${err.message}` }])
        setSumFollowUpLoading(false)
      },
    )
  }, [sumResult, sumFollowUpHistory])

  const handleProfAnalyse = useCallback(() => {
    if (!profUsername.trim()) return
    setProfLog([])
    setProfResult('')
    setProfResultVisible(false)
    setProfLoading(true)

    const body: Record<string, unknown> = {
      username: profUsername,
      model: profModel,
      retrieval_mode: profMode,
    }
    if (profPrompt) body.focus_prompt = profPrompt
    if (profDateMode === 'exact') {
      if (profDateFrom) body.date_from = profDateFrom
      if (profDateTo) body.date_to = profDateTo
    } else {
      if (profMonthFrom) body.date_from = profMonthFrom + '-01'
      if (profMonthTo) body.date_to = profMonthTo + '-31'
    }

    let accumulated = ''
    profStopRef.current = streamEvents(
      '/user-profile',
      body,
      (event) => {
        const type = event.type as string
        if (type === 'log') {
          const level = (event.level as string) || 'info'
          appendProfLog(
            level === 'error' ? 'err' : level === 'warning' ? 'warn' : level === 'success' ? 'ok' : 'info',
            (event.message as string) || '',
          )
        } else if (type === 'chunk') {
          accumulated += (event.content as string) || ''
          setProfResult(accumulated)
          setProfResultVisible(true)
        } else if (type === 'result') {
          accumulated = (event.content as string) || accumulated
          setProfResult(accumulated)
          setProfResultVisible(true)
        }
      },
      () => {
        setProfLoading(false)
        setProfFormCollapsed(true)
      },
      (err) => {
        appendProfLog('err', err.message)
        setProfLoading(false)
      },
    )
  }, [profUsername, profModel, profMode, profPrompt, profDateMode, profDateFrom, profDateTo, profMonthFrom, profMonthTo])

  const handleProfFollowUp = useCallback((q: string) => {
    setProfFollowUpHistory((prev) => [...prev, { role: 'user', content: q }])
    setProfFollowUpLoading(true)
    let answer = ''
    streamEvents(
      '/user-profile/followup',
      { question: q, profile: profResult, history: profFollowUpHistory, username: profUsername },
      (event) => {
        if (event.type === 'chunk') {
          answer += (event.content as string) || ''
          setProfFollowUpHistory((prev) => {
            const last = prev[prev.length - 1]
            if (last?.role === 'assistant') {
              return [...prev.slice(0, -1), { role: 'assistant', content: answer }]
            }
            return [...prev, { role: 'assistant', content: answer }]
          })
        } else if (event.type === 'result') {
          answer = (event.content as string) || answer
          setProfFollowUpHistory((prev) => {
            const last = prev[prev.length - 1]
            if (last?.role === 'assistant') {
              return [...prev.slice(0, -1), { role: 'assistant', content: answer }]
            }
            return [...prev, { role: 'assistant', content: answer }]
          })
        }
      },
      () => setProfFollowUpLoading(false),
      (err) => {
        setProfFollowUpHistory((prev) => [...prev, { role: 'assistant', content: `Error: ${err.message}` }])
        setProfFollowUpLoading(false)
      },
    )
  }, [profResult, profFollowUpHistory, profUsername])

  const inputCls = 'search-input'

  return (
    <div className="max-w-5xl mx-auto px-3 sm:px-4 py-4 space-y-4">
      {/* Sub-tab switcher */}
      <div className="flex gap-1 bg-white rounded-2xl shadow border border-gray-200 p-1 overflow-x-auto">
        <button
          onClick={() => setSubTab('chat')}
          className={`subtab-btn flex items-center gap-1.5 text-sm font-semibold ${subTab === 'chat' ? 'subtab-btn-active' : ''}`}
        >
          <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-3 3v-3z" />
          </svg>
          RAG Chat
        </button>
        <button
          onClick={() => setSubTab('profile')}
          className={`subtab-btn flex items-center gap-1.5 text-sm font-semibold ${subTab === 'profile' ? 'subtab-btn-active' : ''}`}
        >
          <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
          </svg>
          User Profile
        </button>
      </div>

      {/* ── RAG Chat sub-tab ── */}
      {subTab === 'chat' && (
        <div className="space-y-4">
          {/* Sticky form panel */}
          <div className="sticky top-0 z-20">
            {sumFormCollapsed ? (
              <div className="bg-white rounded-2xl shadow border border-indigo-100 px-4 py-2.5 flex items-center gap-3 flex-wrap">
                <span className="text-xs font-semibold text-indigo-800 shrink-0">RAG Chat</span>
                <span className="text-xs text-gray-500 flex-1 min-w-0 truncate">
                  {[sumUsername && `@${sumUsername}`, sumDateFrom && `from ${sumDateFrom}`, sumDateTo && `to ${sumDateTo}`].filter(Boolean).join(' · ') || 'All messages'}
                </span>
                <button
                  onClick={() => setSumFormCollapsed(false)}
                  className="text-xs font-semibold text-gray-600 hover:text-gray-900 border border-gray-300 hover:border-gray-500 px-3 py-1.5 rounded-lg transition-colors"
                >
                  ▼ Filters
                </button>
              </div>
            ) : (
              <div className="bg-white rounded-2xl shadow p-5">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h2 className="text-base font-semibold text-indigo-800">RAG Chat</h2>
                    <p className="text-xs text-gray-500 mt-0.5">Semantic retrieval + clustering over your filtered messages.</p>
                  </div>
                  {sumResultVisible && (
                    <button
                      onClick={() => setSumFormCollapsed(true)}
                      className="text-xs font-semibold text-gray-500 hover:text-gray-800 border border-gray-300 hover:border-gray-500 px-3 py-1.5 rounded-lg transition-colors"
                    >
                      ▲ Collapse
                    </button>
                  )}
                </div>

                {/* Username + date mode */}
                <div className="flex flex-wrap gap-2 items-center mb-3">
                  <label className="filter-label">Username</label>
                  <input value={sumUsername} onChange={(e) => setSumUsername(e.target.value)} placeholder="Optional username…" className={`${inputCls} w-44`} />
                  <div className="inline-flex rounded-lg border border-gray-300 overflow-hidden">
                    <button onClick={() => setSumDateMode('exact')} className={`range-mode-btn ${sumDateMode === 'exact' ? 'range-mode-active' : ''}`}>Exact Date</button>
                    <button onClick={() => setSumDateMode('month')} className={`range-mode-btn ${sumDateMode === 'month' ? 'range-mode-active' : ''}`}>By Month</button>
                  </div>
                </div>

                {/* Date inputs */}
                {sumDateMode === 'exact' ? (
                  <div className="flex flex-wrap gap-2 items-center mb-3">
                    <label className="filter-label">From</label>
                    <input type="date" value={sumDateFrom} onChange={(e) => setSumDateFrom(e.target.value)} className={inputCls} />
                    <label className="filter-label">To</label>
                    <input type="date" value={sumDateTo} onChange={(e) => setSumDateTo(e.target.value)} className={inputCls} />
                  </div>
                ) : (
                  <div className="flex flex-wrap gap-2 items-center mb-3">
                    <label className="filter-label">From</label>
                    <input type="month" value={sumMonthFrom} onChange={(e) => setSumMonthFrom(e.target.value)} className={inputCls} />
                    <label className="filter-label">To</label>
                    <input type="month" value={sumMonthTo} onChange={(e) => setSumMonthTo(e.target.value)} className={inputCls} />
                  </div>
                )}

                {/* Min words + suno */}
                <div className="flex flex-wrap gap-2 items-center mb-3">
                  <input type="number" value={sumMinWords} onChange={(e) => setSumMinWords(parseInt(e.target.value) || 0)} min={0} placeholder="Min words" className={`${inputCls} w-24`} />
                  <label className="filter-label">Suno Team</label>
                  <select value={sumSuno} onChange={(e) => setSumSuno(e.target.value)} className={inputCls}>
                    <option value="all">All</option>
                    <option value="only">Only Suno Team</option>
                    <option value="exclude">Exclude Suno Team</option>
                  </select>
                </div>

                {/* Retrieval mode */}
                <div className="flex flex-wrap gap-2 items-center mb-3">
                  <label className="filter-label">Mode</label>
                  <select value={sumMode} onChange={(e) => setSumMode(e.target.value)} className={`${inputCls} text-xs py-1`}>
                    <option value="cluster">Cluster &amp; Sample</option>
                    <option value="all">All filtered messages</option>
                  </select>
                  <span className="text-xs text-gray-400">Semantic retrieval → HDBSCAN clustering → representative sampling</span>
                </div>

                {/* Custom instructions */}
                <div className="mb-3">
                  <label className="block text-xs font-semibold text-gray-700 mb-1">Custom instructions / focus (optional)</label>
                  <textarea
                    value={sumPrompt}
                    onChange={(e) => setSumPrompt(e.target.value)}
                    rows={2}
                    placeholder="e.g. Focus on feature requests and bug reports. Bullet-point format."
                    className={`${inputCls} w-full resize-none`}
                  />
                </div>

                {/* Model + submit */}
                <div className="flex flex-wrap items-center gap-2">
                  <label className="text-xs font-semibold text-gray-700 shrink-0">Model</label>
                  <select value={sumModel} onChange={(e) => setSumModel(e.target.value)} className={`${inputCls} text-xs py-1 flex-1 min-w-[10rem] max-w-xs`}>
                    {MODEL_OPTIONS}
                  </select>
                  <button onClick={handleSumGenerate} disabled={sumLoading} className="search-btn">
                    {sumLoading ? 'Generating…' : 'Generate RAG Summary'}
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Results */}
          {(sumLog.length > 0 || sumResultVisible) && (
            <div className="space-y-4 pb-8">
              <LogPanel entries={sumLog} visible={sumLogVisible} onToggle={() => setSumLogVisible((v) => !v)} />

              {sumResultVisible && (
                <div className="bg-white rounded-2xl shadow p-5 markdown-body" aria-live="polite">
                  <ReactMarkdown>{sumResult || (sumLoading ? '▋' : '')}</ReactMarkdown>
                </div>
              )}

              {sumResultVisible && (
                <FollowUpSection
                  history={sumFollowUpHistory}
                  onAsk={handleSumFollowUp}
                  onClear={() => setSumFollowUpHistory([])}
                  loading={sumFollowUpLoading}
                />
              )}
            </div>
          )}
        </div>
      )}

      {/* ── User Profile sub-tab ── */}
      {subTab === 'profile' && (
        <div className="space-y-4">
          {/* Sticky form panel */}
          <div className="sticky top-0 z-20">
            {profFormCollapsed ? (
              <div className="bg-white rounded-2xl shadow border border-indigo-100 px-4 py-2.5 flex items-center gap-3 flex-wrap">
                <span className="text-xs font-semibold text-indigo-800 shrink-0">User Profile</span>
                <span className="text-xs text-gray-500 flex-1 min-w-0 truncate">
                  {profUsername || '—'}
                </span>
                <button
                  onClick={() => setProfFormCollapsed(false)}
                  className="text-xs font-semibold text-gray-600 hover:text-gray-900 border border-gray-300 hover:border-gray-500 px-3 py-1.5 rounded-lg transition-colors"
                >
                  ▼ Filters
                </button>
              </div>
            ) : (
              <div className="bg-white rounded-2xl shadow p-5">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h2 className="text-base font-semibold text-indigo-800">User Profile Analysis</h2>
                    <p className="text-xs text-gray-500 mt-0.5">Persona, attitude evolution, and concerns for a specific user.</p>
                  </div>
                  {profResultVisible && (
                    <button
                      onClick={() => setProfFormCollapsed(true)}
                      className="text-xs font-semibold text-gray-500 hover:text-gray-800 border border-gray-300 hover:border-gray-500 px-3 py-1.5 rounded-lg transition-colors"
                    >
                      ▲ Collapse
                    </button>
                  )}
                </div>

                {/* Username + date mode */}
                <div className="flex flex-wrap gap-2 items-center mb-3">
                  <label className="filter-label">Username <span className="text-red-500">*</span></label>
                  <input value={profUsername} onChange={(e) => setProfUsername(e.target.value)} placeholder="Exact or partial username…" className={`${inputCls} w-52`} />
                  <div className="inline-flex rounded-lg border border-gray-300 overflow-hidden">
                    <button onClick={() => setProfDateMode('exact')} className={`range-mode-btn ${profDateMode === 'exact' ? 'range-mode-active' : ''}`}>Exact Date</button>
                    <button onClick={() => setProfDateMode('month')} className={`range-mode-btn ${profDateMode === 'month' ? 'range-mode-active' : ''}`}>By Month</button>
                  </div>
                </div>

                {/* Date inputs */}
                {profDateMode === 'exact' ? (
                  <div className="flex flex-wrap gap-2 items-center mb-3">
                    <label className="filter-label">From</label>
                    <input type="date" value={profDateFrom} onChange={(e) => setProfDateFrom(e.target.value)} className={inputCls} />
                    <label className="filter-label">To</label>
                    <input type="date" value={profDateTo} onChange={(e) => setProfDateTo(e.target.value)} className={inputCls} />
                  </div>
                ) : (
                  <div className="flex flex-wrap gap-2 items-center mb-3">
                    <label className="filter-label">From</label>
                    <input type="month" value={profMonthFrom} onChange={(e) => setProfMonthFrom(e.target.value)} className={inputCls} />
                    <label className="filter-label">To</label>
                    <input type="month" value={profMonthTo} onChange={(e) => setProfMonthTo(e.target.value)} className={inputCls} />
                  </div>
                )}

                {/* Retrieval mode */}
                <div className="flex flex-wrap gap-2 items-center mb-3">
                  <label className="filter-label">Mode</label>
                  <select value={profMode} onChange={(e) => setProfMode(e.target.value)} className={`${inputCls} text-xs py-1`}>
                    <option value="cluster">Cluster &amp; Sample</option>
                    <option value="all">All filtered messages</option>
                  </select>
                </div>

                {/* Focus prompt */}
                <div className="mb-3">
                  <label className="block text-xs font-semibold text-gray-700 mb-1">Analysis focus (optional)</label>
                  <textarea
                    value={profPrompt}
                    onChange={(e) => setProfPrompt(e.target.value)}
                    rows={2}
                    placeholder="e.g. Focus on their feedback about the music generation quality and pricing concerns."
                    className={`${inputCls} w-full resize-none`}
                  />
                </div>

                {/* Model + submit */}
                <div className="flex flex-wrap items-center gap-2">
                  <label className="text-xs font-semibold text-gray-700 shrink-0">Model</label>
                  <select value={profModel} onChange={(e) => setProfModel(e.target.value)} className={`${inputCls} text-xs py-1 flex-1 min-w-[10rem] max-w-xs`}>
                    {MODEL_OPTIONS}
                  </select>
                  <button onClick={handleProfAnalyse} disabled={profLoading || !profUsername.trim()} className="search-btn">
                    {profLoading ? 'Analysing…' : 'Analyse User'}
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Results */}
          {(profLog.length > 0 || profResultVisible) && (
            <div className="space-y-4 pb-8">
              <LogPanel entries={profLog} visible={profLogVisible} onToggle={() => setProfLogVisible((v) => !v)} />

              {profResultVisible && (
                <div className="bg-white rounded-2xl shadow p-5 markdown-body" aria-live="polite">
                  <ReactMarkdown>{profResult || (profLoading ? '▋' : '')}</ReactMarkdown>
                </div>
              )}

              {profResultVisible && (
                <FollowUpSection
                  history={profFollowUpHistory}
                  onAsk={handleProfFollowUp}
                  onClear={() => setProfFollowUpHistory([])}
                  loading={profFollowUpLoading}
                />
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
