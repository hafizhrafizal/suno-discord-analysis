import { useState, useEffect } from 'react'
import { createPortal } from 'react-dom'
import { useQueryClient } from '@tanstack/react-query'
import { useAuthStore } from '../../store/authStore'

export default function ApiKeyModal() {
  const { showKeyModal, setShowKeyModal } = useAuthStore()
  const queryClient = useQueryClient()
  const [keyInput, setKeyInput] = useState('')
  const [keyError, setKeyError] = useState('')

  useEffect(() => {
    if (showKeyModal) {
      setKeyInput(localStorage.getItem('openai_api_key') || '')
      setKeyError('')
    }
  }, [showKeyModal])

  if (!showKeyModal) return null

  const handleSave = () => {
    if (!keyInput.trim()) { setKeyError('Please enter your API key.'); return }
    try {
      localStorage.setItem('openai_api_key', keyInput.trim())
      setShowKeyModal(false)
      setKeyInput('')
      queryClient.invalidateQueries({ queryKey: ['stats'] })
    } catch (e) {
      setKeyError(e instanceof Error ? e.message : 'Failed to save API key')
    }
  }

  return createPortal(
    <>
      <div className="fixed inset-0 z-[9999] bg-black/50" onClick={() => setShowKeyModal(false)} />
      <div className="fixed inset-0 z-[10000] flex items-end sm:items-center justify-center p-0 sm:p-4 pointer-events-none">
        <div className="pointer-events-auto w-full sm:max-w-md rounded-t-2xl sm:rounded-xl shadow-2xl overflow-hidden">
          {/* Header */}
          <div className="bg-[#0d3e7f] px-5 py-4">
            <div className="flex items-center gap-2.5 mb-0.5">
              <svg className="w-5 h-5 text-white shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
              </svg>
              <h2 className="text-lg font-bold text-white">OpenAI API Key Required</h2>
            </div>
            <p className="text-xs text-blue-200 ml-[30px]">Required for Chat, Summarize, and the embedding model.</p>
          </div>
          {/* Body */}
          <div className="bg-white px-5 py-5">
            <label className="block text-sm font-semibold text-gray-800 mb-1.5">Your API key</label>
            <input
              type="text"
              value={keyInput}
              onChange={(e) => setKeyInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSave()}
              placeholder="sk-…"
              className="w-full border-2 border-[#0d3e7f] rounded-lg px-3 py-3 text-sm font-mono text-gray-900 focus:outline-none focus:ring-2 focus:ring-[#0d3e7f]/30 mb-3"
              autoFocus
            />
            <p className="text-xs text-gray-500 mb-4 leading-relaxed">
              Stored in <strong>your browser only</strong> — never sent to or saved on the server.
            </p>
            {keyError && <p className="text-xs text-red-600 mb-3">{keyError}</p>}
            <div className="flex gap-2.5">
              <button
                onClick={handleSave}
                className="flex-1 py-3 text-sm font-bold bg-[#0d3e7f] text-white rounded-lg hover:bg-[#0a2f60] transition-colors"
              >
                Save &amp; Continue
              </button>
              <button
                onClick={() => setShowKeyModal(false)}
                className="px-5 py-3 text-sm font-semibold border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      </div>
    </>,
    document.body,
  )
}
