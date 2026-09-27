/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#090d16',
        surface: '#111827',
        'surface-highlight': '#1f293d',
        accent: '#10b981', // Emerald green
        'accent-red': '#ef4444',
      }
    },
  },
  plugins: [],
}
