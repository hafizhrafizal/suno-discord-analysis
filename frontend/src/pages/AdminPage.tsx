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
