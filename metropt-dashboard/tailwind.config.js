/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'], // For engineering data
      },
      colors: {
        background: "#0f172a", // Slate 900 (Deep Navy)
        sidebar: "#1e293b",    // Slate 800
        card: "#1e293b",       // Slate 800 (Base for glass)
        primary: "#06b6d4",    // Cyan 500 (Electric Blue)
        secondary: "#6366f1",  // Indigo 500 (Accent)
        success: "#10b981",    // Emerald
        danger: "#f43f5e",     // Rose
        text: "#f1f5f9"        // Slate 100
      },
      boxShadow: {
        'glow': '0 0 20px -5px rgba(6, 182, 212, 0.5)', // Neon glow effect
      }
    },
  },
  plugins: [],
}