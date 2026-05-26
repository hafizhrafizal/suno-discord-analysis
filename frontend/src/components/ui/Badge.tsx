import { X } from 'lucide-react'

interface BadgeProps {
  label: string
  color?: string
  onRemove?: () => void
  textColor?: string
}

function hexToRgb(hex: string) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
  return result
    ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
      }
    : null
}

function isLightColor(hex: string): boolean {
  const rgb = hexToRgb(hex)
  if (!rgb) return true
  const luminance = (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b) / 255
  return luminance > 0.5
}

export default function Badge({ label, color = '#6366f1', onRemove }: BadgeProps) {
  const light = isLightColor(color)
  const textClass = light ? 'text-slate-800' : 'text-white'

  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${textClass}`}
      style={{ backgroundColor: color }}
    >
      {label}
      {onRemove && (
        <button
          onClick={onRemove}
          className="hover:opacity-75 transition-opacity"
          type="button"
          aria-label={`Remove ${label}`}
        >
          <X size={10} />
        </button>
      )}
    </span>
  )
}
