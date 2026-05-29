import { useNavigate } from 'react-router-dom'

const NOTES = ['♪', '♫', '♩', '♬', '♪', '♫']
const NOTE_STYLES: React.CSSProperties[] = [
  { left: '12%',  animationDelay: '0s',    animationDuration: '3.0s', fontSize: '1.4rem', color: '#4a90d9' },
  { left: '28%',  animationDelay: '0.6s',  animationDuration: '3.8s', fontSize: '1.1rem', color: '#93bde5' },
  { left: '45%',  animationDelay: '1.1s',  animationDuration: '2.9s', fontSize: '1.6rem', color: '#0d3e7f' },
  { left: '62%',  animationDelay: '0.3s',  animationDuration: '3.4s', fontSize: '1.0rem', color: '#4a90d9' },
  { left: '77%',  animationDelay: '1.5s',  animationDuration: '3.1s', fontSize: '1.3rem', color: '#93bde5' },
  { left: '88%',  animationDelay: '0.8s',  animationDuration: '4.0s', fontSize: '1.2rem', color: '#0d3e7f' },
]

// Waveform bars — first few alive, then dead (flatline at 404)
const BARS = [1, 0.7, 0.9, 0.5, 0.8, 0.3, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.05, 0]

export default function NotFoundPage() {
  const navigate = useNavigate()

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-6 select-none">

      {/* Floating notes */}
      <div className="relative w-full max-w-lg h-20 mb-2 overflow-hidden pointer-events-none">
        {NOTES.map((note, i) => (
          <span
            key={i}
            className="note-float absolute bottom-0 font-bold"
            style={NOTE_STYLES[i]}
          >
            {note}
          </span>
        ))}
      </div>

      {/* Card */}
      <div className="w-full max-w-lg bg-white shadow-lg overflow-hidden">

        {/* Header — Discord-style deleted message bar */}
        <div className="bg-[#0d3e7f] px-6 py-4 flex items-center gap-3">
          <div className="w-8 h-8 bg-[#4a90d9] flex items-center justify-center flex-shrink-0">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
            </svg>
          </div>
          <div>
            <p className="text-white font-semibold text-sm">Suno Analysis Tool</p>
            <p className="text-[#93bde5] text-xs">Discord Message Research</p>
          </div>
          <span className="ml-auto text-[#93bde5] text-xs font-mono opacity-70">404</span>
        </div>

        {/* Waveform — alive then flatlines */}
        <div className="bg-[#f8fafc] border-b border-[#e2e8f0] px-6 py-4 flex items-end gap-[3px] h-14">
          {BARS.map((h, i) => {
            const isDead = h === 0 || i >= 7
            const height = Math.max(h * 28, isDead ? 2 : 4)
            return (
              <div
                key={i}
                className={isDead ? 'bar-dead' : 'bar-live'}
                style={{
                  width: '12px',
                  height: `${height}px`,
                  background: isDead ? '#cbd5e1' : '#0d3e7f',
                  animationDelay: isDead ? '0s' : `${i * 0.07}s`,
                  animationDuration: isDead ? '2.4s' : `${0.9 + i * 0.04}s`,
                  flexShrink: 0,
                }}
              />
            )
          })}
        </div>

        {/* Body — looks like a Discord "message deleted" notice */}
        <div className="px-6 py-7 text-center">
          <div className="glitch">
            <p className="text-[5rem] font-black leading-none text-[#0d3e7f] tracking-tight">404</p>
          </div>
          <p className="mt-2 text-lg font-semibold text-gray-800">
            Ooops... Page Not Found!
          </p>
          <p className="mt-1 text-sm text-gray-500 max-w-xs mx-auto">
            The page you're looking for doesn't exist in this Discord server — or it never did.
          </p>

          
        </div>

        {/* Actions */}
        <div className="border-t border-[#e2e8f0] px-6 py-4 flex gap-3 justify-center">
          <button
            onClick={() => navigate(-1)}
            className="action-btn-primary px-5 py-2 text-sm"
          >
            ← Go back
          </button>
          <button
            onClick={() => navigate('/search')}
            className="search-btn px-5 py-2 text-sm"
          >
            Search messages
          </button>
        </div>
      </div>

      <p className="mt-5 text-xs text-gray-400">
        Lost? Try searching the Discord archive above.
      </p>
    </div>
  )
}
