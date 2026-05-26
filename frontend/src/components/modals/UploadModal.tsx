import { useRef, useState } from 'react'
import { Upload } from 'lucide-react'
import Modal from '../ui/Modal'
import Button from '../ui/Button'

interface UploadModalProps {
  isOpen: boolean
  onClose: () => void
  onSuccess: () => void
}

export default function UploadModal({ isOpen, onClose, onSuccess }: UploadModalProps) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [progress, setProgress] = useState<string | null>(null)
  const [done, setDone] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)

  const handleUpload = async () => {
    const file = fileRef.current?.files?.[0]
    if (!file) return

    setUploading(true)
    setProgress(null)
    setDone(null)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
        credentials: 'include',
      })

      if (!res.ok || !res.body) {
        throw new Error(`HTTP ${res.status}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done: streamDone, value } = await reader.read()
        if (streamDone) break

        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop() ?? ''

        for (const part of parts) {
          const line = part.trim()
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6))
              if (event.type === 'progress') {
                setProgress(`Inserting rows: ${event.inserted} / ${event.total}`)
              } else if (event.type === 'done') {
                setDone(`Upload complete! ID: ${event.upload_id} (${event.total_inserted} rows)`)
                onSuccess()
              } else if (event.type === 'error') {
                setError(event.message)
              }
            } catch {
              // ignore
            }
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Upload CSV">
      <div className="space-y-4">
        <div className="border-2 border-dashed border-slate-300 rounded-lg p-8 text-center">
          <Upload className="mx-auto mb-3 text-slate-400" size={32} />
          <p className="text-sm text-slate-600 mb-2">Select a CSV file to upload</p>
          <input
            ref={fileRef}
            type="file"
            accept=".csv"
            className="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-medium file:bg-navy-800 file:text-white hover:file:bg-navy-700"
          />
        </div>

        <p className="text-xs text-slate-500">
          Required columns: <code>content</code>, <code>date</code>. Optional: <code>id</code>, <code>author</code>, <code>attachments</code>, <code>reactions</code>.
        </p>

        {progress && (
          <div className="bg-blue-50 border border-blue-200 rounded-md p-3 text-sm text-blue-700">
            {progress}
          </div>
        )}
        {done && (
          <div className="bg-green-50 border border-green-200 rounded-md p-3 text-sm text-green-700">
            {done}
          </div>
        )}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-3 text-sm text-red-700">
            {error}
          </div>
        )}

        <div className="flex gap-3 justify-end">
          <Button variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleUpload} loading={uploading} disabled={uploading}>
            Upload
          </Button>
        </div>
      </div>
    </Modal>
  )
}
