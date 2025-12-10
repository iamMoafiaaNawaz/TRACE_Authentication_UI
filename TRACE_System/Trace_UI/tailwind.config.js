/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        trace: {
          bg: '#90D5FF',
          accent: '#1E90FF',
        }
      }
    },
  },
  plugins: [],
}