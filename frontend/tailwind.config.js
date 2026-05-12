/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#040B16',
        surface: '#0A1428',
        surfaceHighlight: '#112240',
        primary: '#00F0FF',
        primaryDark: '#007A8A',
        secondary: '#6E00FF',
        danger: '#FF003C',
        success: '#00FF66',
        accent: '#FF9F00',
        textMain: '#E2E8F0',
        textMuted: '#94A3B8'
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'cyber-grid': 'linear-gradient(to right, #00F0FF11 1px, transparent 1px), linear-gradient(to bottom, #00F0FF11 1px, transparent 1px)',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['Fira Code', 'monospace'],
      },
      boxShadow: {
        'neon': '0 0 5px theme("colors.primary"), 0 0 20px theme("colors.primary")',
        'neon-danger': '0 0 5px theme("colors.danger"), 0 0 20px theme("colors.danger")',
        'neon-accent': '0 0 5px theme("colors.accent"), 0 0 20px theme("colors.accent")',
        'glass': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
      }
    },
  },
  plugins: [],
}
