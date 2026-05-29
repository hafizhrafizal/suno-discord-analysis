import { useState, useRef, useEffect, useCallback } from 'react'
import { PALETTE } from '../../utils/palette'

interface ColorPickerProps {
  value: string
  onChange: (color: string) => void
}

const POP_WIDTH = 196

export default function ColorPicker({ value, onChange }: ColorPickerProps) {
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState({ top: 0, left: 0 })
  const btnRef = useRef<HTMLButtonElement>(null)
  const popRef = useRef<HTMLDivElement>(null)

  const recompute = useCallback(() => {
    if (!btnRef.current) return
    const r = btnRef.current.getBoundingClientRect()
    let left = r.left
    if (left + POP_WIDTH > window.innerWidth - 8) left = r.right - POP_WIDTH
    setPos({ top: r.bottom + 4, left })
  }, [])

  const handleToggle = () => {
    if (!open) recompute()
    setOpen((v) => !v)
  }

  useEffect(() => {
    if (!open) return
    const onMouseDown = (e: MouseEvent) => {
      if (
        popRef.current && !popRef.current.contains(e.target as Node) &&
        btnRef.current && !btnRef.current.contains(e.target as Node)
      ) setOpen(false)
    }
    document.addEventListener('mousedown', onMouseDown)
    // capture:true catches scroll from any scrollable ancestor
    window.addEventListener('scroll', recompute, true)
    return () => {
      document.removeEventListener('mousedown', onMouseDown)
      window.removeEventListener('scroll', recompute, true)
    }
  }, [open, recompute])

  return (
    <div className="inline-block">
      <button
        ref={btnRef}
        type="button"
        onClick={handleToggle}
        className="h-8 w-10 border border-gray-300 cursor-pointer hover:ring-2 hover:ring-indigo-400 transition-all shrink-0"
        style={{ background: value }}
        title={value}
      />
      {open && (
        <div
          ref={popRef}
          className="fixed z-[9999] p-2 bg-white border border-gray-200 shadow-lg"
          style={{ top: pos.top, left: pos.left }}
        >
          <div className="grid grid-cols-6 gap-1">
            {PALETTE.map((color) => (
              <button
                key={color}
                type="button"
                onClick={() => { onChange(color); setOpen(false) }}
                className="w-6 h-6 border-2 transition-transform hover:scale-110"
                style={{
                  background: color,
                  borderColor: color === value ? '#fff' : 'transparent',
                  outline: color === value ? `2px solid ${color}` : 'none',
                }}
                title={color}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
