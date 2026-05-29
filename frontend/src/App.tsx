import { useEffect, useState } from 'react'
import {
  createBrowserRouter,
  RouterProvider,
  Navigate,
  Outlet,
} from 'react-router-dom'
import { apiFetch, ApiError } from './api/client'
import { useAuthStore } from './store/authStore'
import type { User } from './types'

import Layout from './components/layout/Layout'
import LoginPage from './pages/LoginPage'
import OnboardingPage from './pages/OnboardingPage'
import SearchPage from './pages/SearchPage'
import BookmarksPage from './pages/BookmarksPage'
import AdminPage from './pages/AdminPage'
import SettingsPage from './pages/SettingsPage'
import CodingPage from './pages/CodingPage'
import NotFoundPage from './pages/NotFoundPage'
import ErrorPage from './pages/ErrorPage'

function CodingRedirect() {
  const last = localStorage.getItem('coding_tab') || 'manager'
  return <Navigate to={`/coding/${last}`} replace />
}

function ProtectedRoute() {
  const { user, setUser, setAppMode } = useAuthStore()
  const [loading, setLoading] = useState(true)
  const [redirect, setRedirect] = useState<string | null>(null)

  useEffect(() => {
    if (user) {
      setLoading(false)
      return
    }
    const check = async () => {
      try {
        const me = await apiFetch<User>('/auth/me')
        setUser(me)
        try {
          const modeData = await apiFetch<{ mode: string }>('/auth/app-mode').catch(() => ({ mode: 'single' }))
          setAppMode(modeData?.mode ?? 'single')
        } catch {
          setAppMode('single')
        }
      } catch (err) {
        if (err instanceof ApiError && err.status === 401) {
          setRedirect('/login')
        }
      } finally {
        setLoading(false)
      }
    }
    check()
  }, [user, setUser, setAppMode])

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-100">
        <div className="animate-spin rounded-full h-10 w-10 border-4 border-indigo-700 border-t-transparent" />
      </div>
    )
  }

  if (redirect) return <Navigate to={redirect} replace />

  return <Outlet />
}

const router = createBrowserRouter([
  {
    path: '/login',
    element: <LoginPage />,
    errorElement: <ErrorPage />,
  },
  {
    path: '/onboarding',
    element: <OnboardingPage />,
    errorElement: <ErrorPage />,
  },
  {
    element: <ProtectedRoute />,
    errorElement: <ErrorPage />,
    children: [
      {
        element: <Layout />,
        errorElement: <ErrorPage />,
        children: [
          { path: '/', element: <Navigate to="/search" replace /> },
          { path: '/search', element: <SearchPage /> },
          { path: '/bookmarks', element: <BookmarksPage /> },
          { path: '/coding', element: <CodingRedirect /> },
          { path: '/coding/manager', element: <CodingPage /> },
          { path: '/coding/table', element: <CodingPage /> },
          { path: '/settings', element: <SettingsPage /> },
          { path: '/admin', element: <AdminPage /> },
        ],
      },
    ],
  },
  {
    path: '*',
    element: <NotFoundPage />,
  },
])

export default function App() {
  return <RouterProvider router={router} future={{ v7_startTransition: true }} />
}
