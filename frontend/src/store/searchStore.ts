import { create } from 'zustand'
import type { Message } from '../types'

interface SearchState {
  results: Message[]
  selectedIds: Set<number>
  bookmarkedIds: Set<number>
  setResults: (results: Message[]) => void
  toggleSelected: (id: number) => void
  selectAll: () => void
  clearSelected: () => void
  setBookmarkedIds: (ids: number[]) => void
  toggleBookmarked: (id: number) => void
}

export const useSearchStore = create<SearchState>((set, get) => ({
  results: [],
  selectedIds: new Set(),
  bookmarkedIds: new Set(),

  setResults: (results) => set({ results }),

  toggleSelected: (id) =>
    set((s) => {
      const next = new Set(s.selectedIds)
      next.has(id) ? next.delete(id) : next.add(id)
      return { selectedIds: next }
    }),

  selectAll: () =>
    set((s) => ({ selectedIds: new Set(s.results.map((r) => r.id)) })),

  clearSelected: () => set({ selectedIds: new Set() }),

  setBookmarkedIds: (ids) => set({ bookmarkedIds: new Set(ids) }),

  toggleBookmarked: (id) =>
    set((s) => {
      const next = new Set(s.bookmarkedIds)
      next.has(id) ? next.delete(id) : next.add(id)
      return { bookmarkedIds: next }
    }),
}))
