import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        navy: {
          50: '#e8edf5',
          100: '#d1dcea',
          200: '#a3b9d5',
          300: '#7596c0',
          400: '#4773ab',
          500: '#2d5f9e',
          600: '#1e4a8a',
          700: '#183d75',
          800: '#1e3a5f',
          900: '#0a1628',
        },
      },
    },
  },
  plugins: [],
} satisfies Config
