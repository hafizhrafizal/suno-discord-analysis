import { useState } from 'react'
import Modal from '../ui/Modal'
import Button from '../ui/Button'
import { apiFetch } from '../../api/client'
import type { Message } from '../../types'

interface BookmarkModalProps {
  isOpen: boolean
  onClose: () => void
  message: Message | null
  onSuccess: () => void
}

export default function BookmarkModal({
  isOpen,
  onClose,
  message,
  onSuccess,
}: BookmarkModalProps) {
  const [note, setNote] = useState('')
  const [ctxBefore, setCtxBefore] = useState(5)
  const [ctxAfter, setCtxAfter] = useState(5)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSave = async () => {
    if (!message) return
    setSaving(true)
    setError(null)
    try {
      await apiFetch('/bookmarks', {
        method: 'POST',
        body: JSON.stringify({
          msg_id: message.id,
          ctx_before: ctxBefore,
          ctx_after: ctxAfter,
          note: note || undefined,
        }),
      })
      onSuccess()
      onClose()
      setNote('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save bookmark')
    } finally {
      setSaving(false)
    }
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Add Bookmark">
      {message && (
        <div className="space-y-4">
          <div className="bg-slate-50 rounded-md p-3 text-sm">
            <span className="font-semibold text-navy-700 mr-2">{message.username}</span>
            <span className="text-slate-400 text-xs">{message.date}</span>
            <p className="mt-1 text-slate-700 line-clamp-3">{message.content}</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Note (optional)
            </label>
            <textarea
              value={note}
              onChange={(e) => setNote(e.target.value)}
              rows={3}
              placeholder="Add a note about this message..."
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-navy-500"
            />
          </div>

          <div className="flex gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">
                Context before
              </label>
              <input
                type="number"
                value={ctxBefore}
                onChange={(e) => setCtxBefore(Math.max(0, parseInt(e.target.value) || 0))}
                min={0}
                max={50}
                className="w-20 rounded-md border border-slate-300 px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-navy-500"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">
                Context after
              </label>
              <input
                type="number"
                value={ctxAfter}
                onChange={(e) => setCtxAfter(Math.max(0, parseInt(e.target.value) || 0))}
                min={0}
                max={50}
                className="w-20 rounded-md border border-slate-300 px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-navy-500"
              />
            </div>
          </div>

          {error && (
            <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-md px-3 py-2">
              {error}
            </p>
          )}

          <div className="flex gap-3 justify-end">
            <Button variant="secondary" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={handleSave} loading={saving}>
              Save Bookmark
            </Button>
          </div>
        </div>
      )}
    </Modal>
  )
}
