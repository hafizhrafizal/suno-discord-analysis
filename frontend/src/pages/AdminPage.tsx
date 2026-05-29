import { useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Navigate } from 'react-router-dom'
import { apiFetch } from '../api/client'
import { useAuthStore } from '../store/authStore'
import type { User } from '../types'

export default function AdminPage() {
  const { user, appMode } = useAuthStore()
  const queryClient = useQueryClient()
  const isAllowed = !(appMode === 'multi' && !user?.is_admin)

  const { data: users = [], isLoading, error } = useQuery<User[]>({
    queryKey: ['admin-users'],
    queryFn: () => apiFetch('/admin/users'),
    enabled: isAllowed,
  })

  const [pwUserId, setPwUserId] = useState<number | null>(null)
  const [pwInput, setPwInput] = useState('')
  const [pwLoading, setPwLoading] = useState(false)
  const [pwError, setPwError] = useState('')
  const [pwSuccess, setPwSuccess] = useState(false)

  if (!isAllowed) {
    return <Navigate to="/search" replace />
  }

  const refresh = () => queryClient.invalidateQueries({ queryKey: ['admin-users'] })

  const toggleAdmin = async (userId: number) => {
    try {
      await apiFetch(`/admin/users/${userId}/toggle-admin`, { method: 'POST' })
      refresh()
    } catch (err) {
      alert(`Failed: ${err instanceof Error ? err.message : String(err)}`)
    }
  }

  const deleteUser = async (userId: number, username: string) => {
    if (!confirm(`Delete user "${username}"? This cannot be undone.`)) return
    try {
      await apiFetch(`/admin/users/${userId}`, { method: 'DELETE' })
      refresh()
    } catch (err) {
      alert(`Failed: ${err instanceof Error ? err.message : String(err)}`)
    }
  }

  const openPwForm = (userId: number) => {
    setPwUserId(userId)
    setPwInput('')
    setPwError('')
    setPwSuccess(false)
  }

  const closePwForm = () => {
    setPwUserId(null)
    setPwInput('')
    setPwError('')
    setPwSuccess(false)
  }

  const submitPassword = async () => {
    if (!pwUserId || !pwInput.trim()) return
    if (pwInput.length < 8) { setPwError('Password must be at least 8 characters.'); return }
    setPwLoading(true)
    setPwError('')
    setPwSuccess(false)
    try {
      await apiFetch(`/admin/users/${pwUserId}/password`, {
        method: 'POST',
        body: JSON.stringify({ password: pwInput }),
      })
      setPwSuccess(true)
      setPwInput('')
      setTimeout(closePwForm, 1500)
    } catch (err) {
      setPwError(err instanceof Error ? err.message : 'Failed to set password')
    } finally {
      setPwLoading(false)
    }
  }

  return (
    <div className="max-w-5xl mx-auto px-3 sm:px-4 py-4 space-y-4">
      <section className="bg-white rounded-2xl shadow p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-semibold text-sm text-gray-700 uppercase tracking-wide">User Management</h2>
          <button onClick={refresh} className="text-xs font-semibold text-indigo-700 hover:underline">Refresh</button>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700 mb-4">
            {error instanceof Error ? error.message : 'Failed to load users'}
          </div>
        )}

        {isLoading ? (
          <div className="flex justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-4 border-indigo-700 border-t-transparent" />
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="bg-gray-50 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                  <th className="px-3 py-2 border-b border-gray-200">Username</th>
                  <th className="px-3 py-2 border-b border-gray-200">Role</th>
                  <th className="px-3 py-2 border-b border-gray-200">Joined</th>
                  <th className="px-3 py-2 border-b border-gray-200 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.map((u) => (
                  <>
                    <tr key={u.id} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="px-3 py-2 font-medium text-gray-800">{u.username}</td>
                      <td className="px-3 py-2">
                        {u.is_admin ? (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-yellow-100 text-yellow-800 rounded-full text-xs font-semibold">
                            Admin
                          </span>
                        ) : (
                          <span className="inline-flex items-center px-2 py-0.5 bg-gray-100 text-gray-600 rounded-full text-xs">
                            User
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-2 text-xs text-gray-500">
                        {u.created_at?.slice(0, 10) ?? '—'}
                      </td>
                      <td className="px-3 py-2">
                        <div className="flex justify-end gap-2">
                          <button
                            onClick={() => pwUserId === u.id ? closePwForm() : openPwForm(u.id)}
                            className="action-btn-warning"
                          >
                            {pwUserId === u.id ? 'Cancel' : 'Set Password'}
                          </button>
                          <button
                            onClick={() => toggleAdmin(u.id)}
                            disabled={u.id === user?.id}
                            className="action-btn-primary"
                          >
                            {u.is_admin ? 'Remove Admin' : 'Make Admin'}
                          </button>
                          <button
                            onClick={() => deleteUser(u.id, u.username)}
                            disabled={u.id === user?.id}
                            className="action-btn-danger"
                          >
                            Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                    {pwUserId === u.id && (
                      <tr key={`pw-${u.id}`} className="bg-amber-50 border-b border-amber-100">
                        <td colSpan={4} className="px-3 py-3">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="text-xs font-semibold text-amber-800">New password for <em>{u.username}</em>:</span>
                            <input
                              type="password"
                              value={pwInput}
                              onChange={(e) => { setPwInput(e.target.value); setPwError(''); setPwSuccess(false) }}
                              onKeyDown={(e) => e.key === 'Enter' && submitPassword()}
                              placeholder="Min 8 characters…"
                              className="search-input text-sm flex-1 min-w-[180px]"
                              autoFocus
                            />
                            <button
                              onClick={submitPassword}
                              disabled={pwLoading || !pwInput.trim()}
                              className="action-btn-primary disabled:opacity-50"
                            >
                              {pwLoading ? 'Saving…' : 'Save'}
                            </button>
                          </div>
                          {pwError && <p className="mt-1.5 text-xs text-red-600">{pwError}</p>}
                          {pwSuccess && <p className="mt-1.5 text-xs text-green-600 font-semibold">Password updated successfully.</p>}
                        </td>
                      </tr>
                    )}
                  </>
                ))}
                {users.length === 0 && (
                  <tr>
                    <td colSpan={4} className="px-3 py-8 text-center text-sm text-gray-400">No users found.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}
