export const PALETTE: string[] = [
  // Indigo — site primary
  '#4f46e5', '#6366f1', '#818cf8', '#7c3aed', '#8b5cf6', '#a78bfa',
  // Blue
  '#1d4ed8', '#2563eb', '#3b82f6', '#60a5fa', '#0369a1', '#0ea5e9',
  // Teal / Cyan
  '#0f766e', '#0d9488', '#14b8a6', '#0891b2', '#06b6d4', '#22d3ee',
  // Green
  '#15803d', '#16a34a', '#22c55e', '#65a30d', '#84cc16', '#4ade80',
  // Amber / Yellow
  '#92400e', '#b45309', '#d97706', '#f59e0b', '#eab308', '#fbbf24',
  // Orange / Red
  '#b91c1c', '#dc2626', '#ef4444', '#c2410c', '#ea580c', '#f97316',
  // Pink / Rose
  '#9f1239', '#be123c', '#e11d48', '#f43f5e', '#db2777', '#ec4899',
  // Slate / Gray
  '#1e293b', '#334155', '#475569', '#64748b', '#6b7280', '#94a3b8',
]

export function randomPaletteColor(): string {
  return PALETTE[Math.floor(Math.random() * PALETTE.length)]
}
