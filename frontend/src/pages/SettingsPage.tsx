import { useState, useRef } from 'react'
import { createPortal } from 'react-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { apiFetch } from '../api/client'
import { useAuthStore } from '../store/authStore'
import type { Stats, EmbeddingModel, Upload, SunoTeamMember } from '../types'

interface DeleteProgress {
  pct: number
  label: string
  type: 'info' | 'success' | 'error'
}

function UploadCard({
  upload,
  isAdmin,
  onRefresh,
}: {
  upload: Upload
  isAdmin: boolean
  onRefresh: () => void
}) {
  const [reembedProgress, setReembedProgress] = useState<{ pct: number; label: string; error?: string } | null>(null)
  const [reembedLoading, setReembedLoading] = useState(false)
  const [confirmType, setConfirmType] = useState<'full' | 'sqlite' | 'embeddings' | null>(null)
  const [deleteProgress, setDeleteProgress] = useState<DeleteProgress | null>(null)
  const [deleteMsg, setDeleteMsg] = useState<{ text: string; type: 'success' | 'error' } | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const doReembed = async () => {
    if (pollRef.current) return
    setReembedLoading(true)
    setReembedProgress({ pct: 0, label: 'Submitting job…' })

    let jobId: string
    try {
      const res = await apiFetch<{ job_id: string; already_running?: boolean; skipped?: number; total_messages?: number }>(
        `/uploads/${encodeURIComponent(upload.id)}/reembed`,
        { method: 'POST' },
      )
      jobId = res.job_id
      if (res.already_running) {
        setReembedProgress({ pct: 0, label: 'Job already running — resuming progress display…' })
      } else {
        const skip = res.skipped || 0
        setReembedProgress({
          pct: 0,
          label: skip > 0
            ? `Resuming: ${skip.toLocaleString()} already embedded, checking remainder…`
            : `Job started — ${(res.total_messages || 0).toLocaleString()} messages queued`,
        })
      }
    } catch (e) {
      setReembedLoading(false)
      setReembedProgress({ pct: 0, label: 'Failed to start', error: e instanceof Error ? e.message : String(e) })
      return
    }

    pollRef.current = setInterval(async () => {
      try {
        const job = await apiFetch<{
          status: string; phase?: string; embedded: number; total: number; skipped: number;
          batch_errors: { batch: number; error: string; traceback: string }[]
          error?: string; traceback?: string; current_batch?: number
        }>(`/jobs/${encodeURIComponent(jobId)}`)

        const embedded = job.embedded || 0
        const total = job.total || 0
        const skipped = job.skipped || 0
        const pct = total > 0 ? Math.round(embedded / total * 100) : (job.status === 'completed' ? 100 : 0)

        if (job.status === 'running') {
          if (job.phase === 'checking') {
            const checkLabel = skipped > 0
              ? `Checking… ${skipped.toLocaleString()} already embedded so far`
              : 'Checking which messages are already embedded…'
            setReembedProgress({ pct: 0, label: checkLabel })
          } else {
            const batchInfo = job.current_batch ? ` (batch ${job.current_batch})` : ''
            const skipNote = skipped > 0 ? ` · ${skipped.toLocaleString()} skipped` : ''
            setReembedProgress({ pct, label: `Embedding… ${pct}% — ${embedded.toLocaleString()}/${total.toLocaleString()} new messages${skipNote}${batchInfo}` })
          }
        } else if (job.status === 'completed') {
          clearInterval(pollRef.current!); pollRef.current = null
          const skipNote = skipped > 0 ? `, ${skipped.toLocaleString()} already embedded` : ''
          const errNote = job.batch_errors.length > 0 ? ` (${job.batch_errors.length} batch error(s) — see below)` : ''
          const errorDetail = job.batch_errors.length > 0
            ? job.batch_errors.map(be => `Batch ${be.batch}:\n${be.error}\n\n${be.traceback}`).join('\n-----------------\n')
            : undefined
          setReembedLoading(false)
          setReembedProgress({ pct: 100, label: `Done — ${embedded.toLocaleString()} embedded${skipNote}${errNote}`, error: errorDetail })
          onRefresh()
          if (!errorDetail) setTimeout(() => setReembedProgress(null), 4000)
        } else if (job.status === 'failed') {
          clearInterval(pollRef.current!); pollRef.current = null
          const detail = `${job.error || 'Unknown error'}\n\n${job.traceback || ''}`.trim()
          setReembedLoading(false)
          setReembedProgress({ pct: 100, label: 'Failed', error: detail })
        }
      } catch { /* ignore blips */ }
    }, 1500)
  }

  const doDelete = async (type: 'full' | 'sqlite' | 'embeddings') => {
    setConfirmType(null)
    setDeleteProgress({ pct: 0, label: 'Deleting…', type: 'info' })
    setDeleteMsg(null)
    const suffix = type === 'full' ? '' : `/${type}`
    try {
      const res = await apiFetch<{ deleted_messages?: number; deleted_embeddings?: number }>(
        `/uploads/${encodeURIComponent(upload.id)}${suffix}`,
        { method: 'DELETE' },
      )
      setDeleteProgress({ pct: 100, label: 'Done', type: 'success' })
      const msg = type === 'sqlite'
        ? `Removed ${res.deleted_messages} messages from the database. Embeddings untouched.`
        : type === 'embeddings'
          ? `Removed ${res.deleted_embeddings} embeddings from the vector store. Database untouched.`
          : `Removed ${res.deleted_messages} messages and all embeddings.`
      setDeleteMsg({ text: msg, type: 'success' })
      setTimeout(() => onRefresh(), 1500)
    } catch (e) {
      setDeleteProgress({ pct: 100, label: 'Failed', type: 'error' })
      setDeleteMsg({ text: `Error: ${e instanceof Error ? e.message : String(e)}`, type: 'error' })
    }
  }

  const modelBadges = Object.entries(upload.embedded_models || {}).map(([mid, has]) => (
    <span key={mid} className={`embed-badge ${has ? 'embed-badge-yes' : 'embed-badge-no'}`}>
      {mid}
    </span>
  ))

  return (
    <div className="border border-gray-200 rounded-xl p-4 hover:border-indigo-200 transition-colors">
      {confirmType && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-3 text-sm">
          <p className="text-yellow-900 mb-2">
            {confirmType === 'sqlite'
              ? `Remove "${upload.filename}" messages from SQLite only. Embeddings will be preserved.`
              : confirmType === 'embeddings'
                ? `Remove all vector embeddings for "${upload.filename}". Messages in the database will be preserved.`
                : `Delete "${upload.filename}" and all its messages from both the database and the vector store?`}
          </p>
          <div className="flex gap-2">
            <button onClick={() => doDelete(confirmType)} className="action-btn-danger">Confirm Delete</button>
            <button onClick={() => setConfirmType(null)} className="action-btn-primary">Cancel</button>
          </div>
        </div>
      )}

      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <p className="font-semibold text-sm text-gray-800 truncate">{upload.filename}</p>
          <p className="text-xs text-gray-500 mt-0.5">
            {Number(upload.row_count).toLocaleString()} rows &bull; Uploaded {upload.upload_time.slice(0, 16)}
          </p>
          <p className="text-xs text-gray-400 font-mono mt-0.5">{upload.id}</p>
          <div className="flex flex-wrap gap-1.5 mt-2">{modelBadges}</div>
        </div>

        {isAdmin && (
          <div className="flex flex-col gap-2 shrink-0">
            <button onClick={doReembed} disabled={reembedLoading} className="action-btn-primary">
              {reembedLoading ? 'Embedding…' : 'Re-embed'}
            </button>
            <button onClick={() => setConfirmType('embeddings')} className="action-btn-warning">Delete Embedding</button>
            <button onClick={() => setConfirmType('sqlite')} className="action-btn-danger">Delete DB</button>
            <button onClick={() => setConfirmType('full')} className="action-btn-danger" style={{ borderColor: '#dc2626', fontWeight: 700 }}>Delete All</button>
          </div>
        )}
      </div>

      {reembedProgress && (
        <div className="mt-3">
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${reembedProgress.pct}%` }} />
          </div>
          <p className="mt-1 text-xs text-gray-600">{reembedProgress.label}</p>
          {reembedProgress.error && (
            <pre className="mt-2 text-xs text-red-700 bg-red-50 border border-red-200 rounded p-2 max-h-40 overflow-auto whitespace-pre-wrap break-all">
              {reembedProgress.error}
            </pre>
          )}
        </div>
      )}

      {deleteProgress && (
        <div className="mt-2">
          <div className="progress-track">
            <div className={`progress-fill ${deleteProgress.type === 'error' ? 'error' : ''}`} style={{ width: `${deleteProgress.pct}%` }} />
          </div>
        </div>
      )}
      {deleteMsg && (
        <div className={`mt-2 rounded-lg p-3 text-sm ${deleteMsg.type === 'success' ? 'bg-green-50 text-green-700 border border-green-200' : 'bg-red-50 text-red-700 border border-red-200'}`}>
          {deleteMsg.text}
        </div>
      )}
    </div>
  )
}

export default function SettingsPage() {
  const { user, appMode } = useAuthStore()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const isAdmin = appMode !== 'multi' || user?.is_admin === true

  // API Key
  const [showKeyPopup, setShowKeyPopup] = useState(false)
  const [keyInput, setKeyInput] = useState('')
  const [keySaving, setKeySaving] = useState(false)
  const [keyError, setKeyError] = useState('')

  // Upload
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [uploadProgress, setUploadProgress] = useState<{ pct: number; label: string; type: 'info' | 'success' | 'error' } | null>(null)
  const [uploadStatus, setUploadStatus] = useState('')
  const [uploadBtnLoading, setUploadBtnLoading] = useState(false)

  const { data: stats } = useQuery<Stats>({
    queryKey: ['stats'],
    queryFn: () => apiFetch('/stats'),
  })

  const { data: models = [] } = useQuery<EmbeddingModel[]>({
    queryKey: ['embedding-models'],
    queryFn: () => apiFetch('/embedding-models'),
  })

  const { data: uploads = [], refetch: refetchUploads } = useQuery<Upload[]>({
    queryKey: ['uploads'],
    queryFn: () => apiFetch('/uploads'),
  })

  const { data: sunoTeam = [], refetch: refetchSuno } = useQuery<SunoTeamMember[]>({
    queryKey: ['suno-team'],
    queryFn: () => apiFetch('/suno-team'),
    enabled: isAdmin,
  })

  const handleSaveKey = async () => {
    if (!keyInput.trim()) { setKeyError('Please enter your API key.'); return }
    setKeySaving(true)
    setKeyError('')
    try {
      await apiFetch('/set-api-key', {
        method: 'POST',
        body: JSON.stringify({ api_key: keyInput.trim() }),
      })
      localStorage.setItem('openai_api_key', keyInput.trim())
      setShowKeyPopup(false)
      setKeyInput('')
      queryClient.invalidateQueries({ queryKey: ['stats'] })
    } catch (e) {
      setKeyError(e instanceof Error ? e.message : 'Failed to save API key')
    } finally {
      setKeySaving(false)
    }
  }

  const handleUpload = async () => {
    if (!uploadFile) { setUploadStatus('Please select a CSV file.'); return }
    setUploadBtnLoading(true)
    setUploadProgress({ pct: 0, label: 'Starting upload…', type: 'info' })
    setUploadStatus('')
    const form = new FormData()
    form.append('file', uploadFile)
    try {
      const response = await fetch('/api/upload', { method: 'POST', body: form })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const data = line.slice(6).trim()
          if (data.startsWith('Processing ')) setUploadProgress({ pct: 5, label: data, type: 'info' })
          else if (data.startsWith('Inserted ')) setUploadProgress({ pct: 20, label: data, type: 'info' })
          else if (data.startsWith('Starting embedding')) setUploadProgress({ pct: 30, label: data, type: 'info' })
          else if (data.startsWith('Embedded ')) {
            const match = data.match(/Embedded\s+(\d+)\/(\d+)/)
            if (match) {
              const pct = 30 + Math.min(70, 70 * parseInt(match[1]) / parseInt(match[2]))
              setUploadProgress({ pct, label: data, type: 'info' })
            }
          } else if (data.startsWith('Completed:')) {
            setUploadProgress({ pct: 100, label: data.replace('Completed: ', ''), type: 'success' })
            refetchUploads()
            queryClient.invalidateQueries({ queryKey: ['stats'] })
          } else if (data.startsWith('Error')) {
            setUploadProgress({ pct: 100, label: data, type: 'error' })
          } else {
            setUploadStatus(data)
          }
        }
      }
    } catch (e) {
      setUploadStatus(`Error: ${e instanceof Error ? e.message : String(e)}`)
      setUploadProgress(null)
    } finally {
      setUploadBtnLoading(false)
    }
  }

  const handleRemoveSuno = async (username: string) => {
    try {
      await apiFetch(`/suno-team/${encodeURIComponent(username)}`, { method: 'DELETE' })
      refetchSuno()
    } catch (e) {
      alert(`Failed to remove: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const apiKeyStatus = stats?.api_key_set
    ? 'API key is set and active.'
    : 'No API key set. Click "Change Key" to add one.'

  return (
    <div className="max-w-5xl mx-auto px-3 sm:px-4 py-4 space-y-5">

      {/* Account section (multi mode only) */}
      {appMode === 'multi' && user && (
        <section className="bg-white rounded-2xl shadow p-5">
          <h2 className="font-semibold text-sm text-gray-700 uppercase tracking-wide mb-4">Account</h2>
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div className="flex items-center gap-3 min-w-0">
              <div className="shrink-0 w-10 h-10 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-700 font-bold text-base select-none">
                {user.username.charAt(0).toUpperCase()}
              </div>
              <div className="min-w-0">
                <p className="font-semibold text-sm text-gray-900 truncate">{user.username}</p>
                <p className="text-xs text-gray-500 mt-0.5">{user.is_admin ? 'Administrator' : 'User'}</p>
              </div>
            </div>
            <button
              onClick={async () => {
                try { await apiFetch('/auth/logout', { method: 'POST' }) } catch { /* ignore */ }
                navigate('/login')
              }}
              className="shrink-0 flex items-center gap-2 px-4 py-2 text-sm font-semibold bg-red-50 text-red-700 border border-red-200 rounded-lg hover:bg-red-100 hover:border-red-300 transition-colors"
            >
              <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" /></svg>
              Log Out
            </button>
          </div>
        </section>
      )}

      {/* Embedding model — hidden in demo mode */}
      {appMode !== 'demo' && <section className="bg-white rounded-2xl shadow p-5">
        <h2 className="font-semibold text-sm text-gray-700 uppercase tracking-wide mb-3">Embedding Model</h2>
        <div className="flex items-center gap-3 p-3 rounded-xl border border-indigo-100 bg-indigo-50">
          <svg className="w-5 h-5 text-indigo-500 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18" />
          </svg>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-sm font-semibold text-gray-800">text-embedding-3-small</span>
              <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: '#dbeafe', color: '#1e40af' }}>cloud</span>
              <span className="text-xs px-1.5 py-0.5 rounded font-medium" style={{ background: '#ede9fe', color: '#5b21b6' }}>active</span>
            </div>
            <p className="text-xs text-gray-500 mt-0.5">OpenAI text-embedding-3-small · 1536 dims</p>
            {models[0] && <p className="text-xs text-gray-400">{(models[0].vector_count || 0).toLocaleString()} uploads embedded</p>}
          </div>
        </div>
      </section>}

      {/* OpenAI API key */}
      <section className="bg-white rounded-2xl shadow p-5">
        <div className="flex items-center justify-between gap-4">
          <div className="min-w-0">
            <h2 className="font-semibold text-sm text-gray-700 uppercase tracking-wide">OpenAI API Key</h2>
            <p className="text-xs text-gray-600 mt-1">{apiKeyStatus}</p>
            <p className="text-xs text-gray-400 mt-1.5 flex items-start gap-1">
              <svg className="w-3.5 h-3.5 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" /></svg>
              Stored in your browser's localStorage only — never sent to or saved on the server.
            </p>
          </div>
          <button onClick={() => { setKeyInput(localStorage.getItem('openai_api_key') || ''); setShowKeyPopup(true) }} className="shrink-0 px-4 py-2 text-sm font-semibold bg-indigo-700 text-white rounded-lg hover:bg-indigo-800 transition-colors">
            Change Key
          </button>
        </div>
      </section>

      {/* Upload new CSV (admin only, hidden in demo mode) */}
      {isAdmin && appMode !== 'demo' && (
        <section className="bg-white rounded-2xl shadow p-5">
          <h2 className="font-semibold text-sm text-gray-700 uppercase tracking-wide mb-3">Upload New CSV</h2>
          <p className="text-xs text-gray-600 mb-3">
            Required columns: <code className="bg-gray-100 text-gray-800 px-1 rounded">author_id, username, date, content</code>.
            Optional: <code className="bg-gray-100 text-gray-800 px-1 rounded">attachments, reactions, is_suno_team, week, month</code>.
          </p>
          <div className="flex flex-wrap gap-3 items-center">
            <label htmlFor="csv-file" className="sr-only">Choose CSV file</label>
            <input
              id="csv-file"
              type="file"
              accept=".csv"
              onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)}
              className="text-sm text-gray-700 file:mr-3 file:py-1.5 file:px-4 file:rounded-lg file:border-0 file:bg-indigo-100 file:text-indigo-800 file:font-semibold file:text-sm hover:file:bg-indigo-200 cursor-pointer"
            />
            <button
              onClick={handleUpload}
              disabled={uploadBtnLoading || !uploadFile}
              className="bg-indigo-700 text-white px-4 py-1.5 rounded-lg text-sm font-semibold hover:bg-indigo-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Upload &amp; Embed
            </button>
          </div>
          {uploadStatus && <p className="mt-2 text-sm text-gray-500">{uploadStatus}</p>}
          {uploadProgress && (
            <div className="mt-3">
              <div className="progress-track">
                <div
                  className={`progress-fill ${uploadProgress.type === 'error' ? 'error' : ''}`}
                  style={{ width: `${uploadProgress.pct}%` }}
                />
              </div>
              <p className={`mt-2 text-xs ${uploadProgress.type === 'error' ? 'text-red-600' : uploadProgress.type === 'success' ? 'text-green-600' : 'text-gray-600'}`}>
                {uploadProgress.label} ({Math.round(uploadProgress.pct)}%)
              </p>
            </div>
          )}
        </section>
      )}

      {/* Active dataset — hidden in demo mode */}
      {appMode !== 'demo' && <section className="bg-white rounded-2xl shadow p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-semibold text-sm text-gray-700 uppercase tracking-wide">Active Dataset</h2>
          <button onClick={() => refetchUploads()} className="text-xs font-semibold text-indigo-700 hover:underline">Refresh</button>
        </div>
        <div className="space-y-3" aria-live="polite">
          {uploads.length === 0 ? (
            <p className="text-sm text-gray-600 text-center py-6">No dataset uploads yet.</p>
          ) : (
            uploads.map((u) => (
              <UploadCard key={u.id} upload={u} isAdmin={isAdmin} onRefresh={() => { refetchUploads(); queryClient.invalidateQueries({ queryKey: ['stats'] }) }} />
            ))
          )}
        </div>
      </section>}

      {/* Suno Team management (admin only, hidden in demo mode) */}
      {isAdmin && appMode !== 'demo' && (
        <section className="bg-white rounded-2xl shadow p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-semibold text-sm text-gray-700 uppercase tracking-wide">Suno Team Members</h2>
            <button onClick={() => refetchSuno()} className="text-xs font-semibold text-indigo-700 hover:underline">Refresh</button>
          </div>
          <p className="text-xs text-gray-600 mb-3">Usernames currently flagged as Suno Team in the database. Removing a user marks all their messages as non-team.</p>
          <div className="overflow-x-auto">
            {sunoTeam.length === 0 ? (
              <p className="text-sm text-gray-400 text-center py-6">No Suno Team members found.</p>
            ) : (
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="bg-gray-50 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                    <th className="px-3 py-2 border-b border-gray-200">Username</th>
                    <th className="px-3 py-2 border-b border-gray-200 text-right">Messages</th>
                    <th className="px-3 py-2 border-b border-gray-200"></th>
                  </tr>
                </thead>
                <tbody>
                  {sunoTeam.map((m) => (
                    <tr key={m.username} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="px-3 py-2">
                        <span className="ubadge" style={{ background: '#ede9fe', color: '#5b21b6' }}>{m.username}</span>
                      </td>
                      <td className="px-3 py-2 text-right text-gray-600 tabular-nums">{m.msg_count.toLocaleString()}</td>
                      <td className="px-3 py-2 text-right">
                        <button onClick={() => handleRemoveSuno(m.username)} className="action-btn-danger">
                          Remove from team
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </section>
      )}

      {/* API Key Popup — rendered via portal directly on document.body so fixed positioning
          is never clipped by ancestor transforms or overflow */}
      {showKeyPopup && createPortal(
        <>
          <div className="fixed inset-0 z-[9999] bg-black/50" onClick={() => setShowKeyPopup(false)} />
          <div className="fixed inset-0 z-[10000] flex items-center justify-center p-4 pointer-events-none">
            <div className="pointer-events-auto w-full max-w-md rounded-xl shadow-2xl overflow-hidden">
              {/* Header */}
              <div className="bg-[#0d3e7f] px-5 py-4">
                <div className="flex items-center gap-2.5 mb-0.5">
                  <svg className="w-5 h-5 text-white shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
                  </svg>
                  <h2 className="text-lg font-bold text-white">OpenAI API Key</h2>
                </div>
                <p className="text-xs text-blue-200 ml-[30px]">Required for Chat, Summarize, and the OpenAI embedding model.</p>
              </div>
              {/* Body */}
              <div className="bg-white px-5 py-5">
                <label className="block text-sm font-semibold text-gray-800 mb-1.5">Your API key</label>
                <input
                  type="text"
                  value={keyInput}
                  onChange={(e) => setKeyInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSaveKey()}
                  placeholder="sk-…"
                  className="w-full border-2 border-[#0d3e7f] rounded px-3 py-2.5 text-sm font-mono text-gray-900 focus:outline-none focus:ring-2 focus:ring-[#0d3e7f]/30 mb-3"
                  autoFocus
                />
                <p className="text-xs text-gray-600 mb-4 leading-relaxed">
                  Stored in <strong>your browser&apos;s localStorage</strong> only — never saved to the server or
                  database. Sent to your own server per session to make OpenAI requests on your behalf.
                </p>
                {keyError && <p className="text-xs text-red-600 mb-3">{keyError}</p>}
                <div className="flex gap-2.5">
                  <button
                    onClick={handleSaveKey}
                    disabled={keySaving}
                    className="flex-1 py-2.5 text-sm font-bold bg-[#0d3e7f] text-white rounded hover:bg-[#0a2f60] disabled:opacity-50 transition-colors"
                  >
                    {keySaving ? 'Saving…' : 'Save & Continue'}
                  </button>
                  <button
                    onClick={() => setShowKeyPopup(false)}
                    className="px-6 py-2.5 text-sm font-semibold border border-gray-300 text-gray-700 rounded hover:bg-gray-50 transition-colors"
                  >
                    Skip
                  </button>
                </div>
              </div>
            </div>
          </div>
        </>,
        document.body,
      )}
    </div>
  )
}
