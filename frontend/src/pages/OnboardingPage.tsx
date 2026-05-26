import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { apiFetch } from '../api/client'
import { useAuthStore } from '../store/authStore'
import type { User } from '../types'

export default function OnboardingPage() {
  const [step, setStep] = useState<1 | 2>(1)
  const [loading, setLoading] = useState<string | null>(null)
  const [adminUsername, setAdminUsername] = useState('')
  const [adminPassword, setAdminPassword] = useState('')
  const [adminConfirm, setAdminConfirm] = useState('')
  const [error, setError] = useState<string | null>(null)
  const { setAppMode, setUser } = useAuthStore()
  const navigate = useNavigate()

  const selectSingle = async () => {
    setLoading('single')
    try {
      await apiFetch('/auth/set-mode', {
        method: 'POST',
        body: JSON.stringify({ mode: 'single' }),
      })
      setAppMode('single')
      navigate('/search')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to set mode')
    } finally {
      setLoading(null)
    }
  }

  const selectMulti = () => {
    setStep(2)
  }

  const createAdmin = async () => {
    setError(null)
    if (!adminUsername.trim()) { setError('Username is required'); return }
    if (!adminPassword) { setError('Password is required'); return }
    if (adminPassword !== adminConfirm) { setError('Passwords do not match'); return }
    setLoading('multi')
    try {
      await apiFetch('/auth/set-mode', {
        method: 'POST',
        body: JSON.stringify({ mode: 'multi' }),
      })
      const user = await apiFetch<User>('/auth/register', {
        method: 'POST',
        body: JSON.stringify({ username: adminUsername, password: adminPassword, is_admin: true }),
      })
      setAppMode('multi')
      setUser(user)
      navigate('/search')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create admin account')
    } finally {
      setLoading(null)
    }
  }

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <div className="w-full max-w-2xl">
        {step === 1 && (
          <>
            <div className="text-center mb-10">
              <h1 className="text-3xl font-bold text-indigo-800">Welcome to Suno Discord Analysis</h1>
              <p className="text-gray-600 mt-2">Choose how you'd like to use this application</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Single User card */}
              <button
                onClick={selectSingle}
                disabled={loading !== null}
                className="bg-white rounded-2xl shadow-md p-8 text-left hover:shadow-lg transition-shadow border-2 border-transparent hover:border-indigo-500 focus:outline-none focus:border-indigo-500"
              >
                <div className="w-14 h-14 bg-indigo-100 rounded-full flex items-center justify-center mb-4">
                  <svg className="w-7 h-7 text-indigo-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                </div>
                <h2 className="text-xl font-bold text-gray-900 mb-2">Single User</h2>
                <ul className="text-gray-600 text-sm space-y-1">
                  <li>• Just you — no login required</li>
                  <li>• Provide your own OpenAI API key</li>
                  <li>• Full access to all features</li>
                  <li>• Perfect for personal research</li>
                </ul>
                {loading === 'single' && (
                  <div className="mt-4 flex items-center gap-2 text-indigo-700 text-sm">
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-indigo-600 border-t-transparent" />
                    Setting up…
                  </div>
                )}
              </button>

              {/* Multi User card */}
              <button
                onClick={selectMulti}
                disabled={loading !== null}
                className="bg-white rounded-2xl shadow-md p-8 text-left hover:shadow-lg transition-shadow border-2 border-transparent hover:border-indigo-500 focus:outline-none focus:border-indigo-500"
              >
                <div className="w-14 h-14 bg-indigo-100 rounded-full flex items-center justify-center mb-4">
                  <svg className="w-7 h-7 text-indigo-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                  </svg>
                </div>
                <h2 className="text-xl font-bold text-gray-900 mb-2">Multi User</h2>
                <ul className="text-gray-600 text-sm space-y-1">
                  <li>• For teams and collaborators</li>
                  <li>• Individual accounts with own bookmarks</li>
                  <li>• Admin manages users and settings</li>
                  <li>• Login required</li>
                </ul>
              </button>
            </div>
          </>
        )}

        {step === 2 && (
          <div className="bg-white rounded-2xl shadow-lg p-8 max-w-md mx-auto">
            <button onClick={() => setStep(1)} className="text-xs text-gray-400 hover:text-gray-700 mb-4 flex items-center gap-1">
              ← Back
            </button>
            <h2 className="text-xl font-bold text-gray-900 mb-2">Create Admin Account</h2>
            <p className="text-sm text-gray-500 mb-6">This will be the administrator account for your team.</p>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Username</label>
                <input
                  type="text"
                  value={adminUsername}
                  onChange={(e) => setAdminUsername(e.target.value)}
                  placeholder="Admin username"
                  autoFocus
                  className="search-input w-full"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
                <input
                  type="password"
                  value={adminPassword}
                  onChange={(e) => setAdminPassword(e.target.value)}
                  placeholder="Strong password"
                  className="search-input w-full"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Confirm Password</label>
                <input
                  type="password"
                  value={adminConfirm}
                  onChange={(e) => setAdminConfirm(e.target.value)}
                  placeholder="Confirm password"
                  onKeyDown={(e) => e.key === 'Enter' && createAdmin()}
                  className="search-input w-full"
                />
              </div>
              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg px-3 py-2 text-sm text-red-700">{error}</div>
              )}
              <button
                onClick={createAdmin}
                disabled={loading !== null}
                className="w-full search-btn py-2 justify-center"
              >
                {loading === 'multi' ? 'Creating…' : 'Create Admin & Continue'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
