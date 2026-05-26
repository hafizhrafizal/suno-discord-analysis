import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ChevronUp, ChevronDown } from 'lucide-react'
import Modal from '../ui/Modal'
import Button from '../ui/Button'
import { apiFetch } from '../../api/client'
import type { Message } from '../../types'

interface ContextModalProps {
  isOpen: boolean
  onClose: () => void
  messageId: number | null
  ctxBefore?: number
  ctxAfter?: number
}

export default function ContextModal({ isOpen, onClose, messageId, ctxBefore = 5, ctxAfter = 5 }: ContextModalProps) {
  const [before, setBefore] = useState(ctxBefore)
  const [after, setAfter] = useState(ctxAfter)

  const { data: messages, isLoading } = useQuery<Message[]>({
    queryKey: ['context', messageId, before, after],
    queryFn: () =>
      apiFetch(`/context/${messageId}?before=${before}&after=${after}`),
    enabled: isOpen && messageId != null,
  })

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Message Context" size="xl">
      <div className="flex items-center gap-3 mb-4 pb-4 border-b border-slate-200">
        <div className="flex items-center gap-2">
          <label className="text-sm text-slate-600">Before:</label>
          <div className="flex items-center gap-1">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setBefore((v) => Math.max(1, v - 5))}
              icon={<ChevronDown size={12} />}
            />
            <span className="w-8 text-center text-sm font-medium">{before}</span>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setBefore((v) => Math.min(50, v + 5))}
              icon={<ChevronUp size={12} />}
            />
          </div>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm text-slate-600">After:</label>
          <div className="flex items-center gap-1">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setAfter((v) => Math.max(1, v - 5))}
              icon={<ChevronDown size={12} />}
            />
            <span className="w-8 text-center text-sm font-medium">{after}</span>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setAfter((v) => Math.min(50, v + 5))}
              icon={<ChevronUp size={12} />}
            />
          </div>
        </div>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-4 border-navy-800 border-t-transparent" />
        </div>
      ) : (
        <div className="space-y-1">
          {messages?.map((msg) => {
            const isTarget = (msg as Message & { is_target?: boolean }).is_target
            return (
              <div
                key={msg.id}
                className={`p-3 rounded-md text-sm ${
                  isTarget
                    ? 'bg-yellow-100 border border-yellow-300 font-medium'
                    : 'bg-slate-50 border border-slate-200'
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-semibold text-navy-700">{msg.username}</span>
                  <span className="text-slate-400 text-xs">{msg.date}</span>
                  {isTarget && (
                    <span className="text-xs bg-yellow-500 text-white px-1.5 py-0.5 rounded">
                      TARGET
                    </span>
                  )}
                </div>
                <p className="text-slate-700 whitespace-pre-wrap break-words">
                  {msg.content}
                </p>
              </div>
            )
          })}
        </div>
      )}
    </Modal>
  )
}
