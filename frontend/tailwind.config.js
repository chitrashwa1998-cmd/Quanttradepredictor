/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Cyberpunk theme colors matching original Streamlit app
        'cyber-dark': '#0a0b0d',
        'cyber-blue': '#00d4ff',
        'cyber-purple': '#9f7aea',
        'cyber-green': '#4ade80',
        'cyber-red': '#f87171',
        'cyber-yellow': '#fbbf24',
        'cyber-gray': '#374151',
      },
      fontFamily: {
        'mono': ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'pulse-glow': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'cyber-flicker': 'flicker 1.5s linear infinite',
      },
      keyframes: {
        flicker: {
          '0%, 19.999%, 22%, 62.999%, 64%, 64.999%, 70%, 100%': {
            opacity: 1,
          },
          '20%, 21.999%, 63%, 63.999%, 65%, 69.999%': {
            opacity: 0.4,
          },
        },
      },
    },
  },
  plugins: [],
}