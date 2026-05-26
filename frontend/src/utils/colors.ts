const PALETTE: [string, string][] = [
  ['#dbeafe', '#1e40af'],
  ['#dcfce7', '#166534'],
  ['#ede9fe', '#5b21b6'],
  ['#fce7f3', '#9d174d'],
  ['#ffedd5', '#9a3412'],
  ['#ccfbf1', '#0f5f53'],
  ['#fee2e2', '#991b1b'],
  ['#fef9c3', '#854d0e'],
  ['#f1f5f9', '#334155'],
]

const userColorMap: Record<string, [string, string]> = {}
let colorIdx = 0

export function getUserColor(username: string): [string, string] {
  if (!userColorMap[username]) {
    userColorMap[username] = PALETTE[colorIdx++ % PALETTE.length]
  }
  return userColorMap[username]
}

export function getUserStyle(username: string): React.CSSProperties {
  const [bg, color] = getUserColor(username)
  return { background: bg, color }
}

// Import React for the CSSProperties type
import type React from 'react'

export function highlightText(text: string, keyword: string): string {
  if (!keyword || !text) return escHtml(text)
  const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  return escHtml(text).replace(
    new RegExp(escaped, 'gi'),
    (m) => `<mark>${m}</mark>`,
  )
}

export function highlightTerms(text: string, tokens: string[]): string {
  if (!tokens.length || !text) return escHtml(text)
  const sorted = [...tokens].sort((a, b) => b.length - a.length)
  const pattern = sorted.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')
  return escHtml(text).replace(
    new RegExp(`(${pattern})`, 'gi'),
    (m) => `<mark>${m}</mark>`,
  )
}

function escHtml(s: string): string {
  const d = document.createElement('div')
  d.textContent = s ?? ''
  return d.innerHTML
}
