import { useState, FormEvent, useEffect } from 'react'
import { useNavigate, Navigate } from 'react-router-dom'
import { apiFetch, ApiError } from '../api/client'
import { useAuthStore } from '../store/authStore'
import type { User } from '../types'

export default function LoginPage() {
  const [tab, setTab] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [modeChecked, setModeChecked] = useState(false)
  const navigate = useNavigate()
  const { setUser, setAppMode, appMode, user } = useAuthStore()

  useEffect(() => {
    apiFetch<{ mode: string }>('/auth/app-mode')
      .then(d => {
        const mode = d?.mode ?? 'single'
        setAppMode(mode)
        setModeChecked(true)
        if (mode !== 'multi') navigate('/', { replace: true })
      })
      .catch(() => {
        setAppMode('single')
        navigate('/', { replace: true })
      })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)

    if (tab === 'register' && password !== confirmPassword) {
      setError('Passwords do not match')
      return
    }

    setLoading(true)
    try {
      const endpoint = tab === 'login' ? '/auth/login' : '/auth/register'
      const u = await apiFetch<User>(endpoint, {
        method: 'POST',
        body: JSON.stringify({ username, password }),
      })
      setUser(u)
      navigate('/')
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message)
      } else {
        setError('Something went wrong')
      }
    } finally {
      setLoading(false)
    }
  }

  // Already authenticated — go to app
  if (user) return <Navigate to="/" replace />

  // While fetching mode show a spinner
  if (!modeChecked) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-100">
        <div className="animate-spin rounded-full h-10 w-10 border-4 border-indigo-700 border-t-transparent" />
      </div>
    )
  }

  // Guard: if mode resolved to non-multi, navigation already happened; render nothing
  if (appMode !== 'multi') return null

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <div className="w-full max-w-sm bg-white rounded-2xl shadow-lg overflow-hidden">
        {/* Header */}
        <div className="bg-indigo-700 px-8 py-6 text-center">
          <svg className="mx-auto mb-2 text-white w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <h1 className="text-xl font-bold text-white">Suno Discord Analysis</h1>
          <p className="text-indigo-200 text-sm mt-1">Discord message research tool</p>
        </div>

        <div className="p-8">
          {/* Tabs */}
          <div className="flex mb-6 border-b border-gray-200">
            <button
              onClick={() => { setTab('login'); setError(null) }}
              className={`flex-1 pb-2.5 text-sm font-semibold transition-colors border-b-2 -mb-px ${
                tab === 'login'
                  ? 'border-indigo-700 text-indigo-700'
                  : 'border-transparent text-gray-500 hover:text-gray-800'
              }`}
            >
              Log In
            </button>
            <button
              onClick={() => { setTab('register'); setError(null) }}
              className={`flex-1 pb-2.5 text-sm font-semibold transition-colors border-b-2 -mb-px ${
                tab === 'register'
                  ? 'border-indigo-700 text-indigo-700'
                  : 'border-transparent text-gray-500 hover:text-gray-800'
              }`}
            >
              Sign Up
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Username</label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                autoFocus
                className="search-input w-full"
                placeholder="Enter username"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="search-input w-full"
                placeholder="Enter password"
              />
            </div>
            {tab === 'register' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Confirm Password</label>
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                  className="search-input w-full"
                  placeholder="Confirm password"
                />
              </div>
            )}

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg px-3 py-2 text-sm text-red-700">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full search-btn justify-center py-2"
            >
              {loading ? 'Please wait…' : tab === 'login' ? 'Log In' : 'Create Account'}
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}
