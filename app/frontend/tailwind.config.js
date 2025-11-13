/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{vue,js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        "dark-bg": "#0d011f",
        "main-purple": "#7e22ce",
        "main-blue": "#4f46e5",
        "light-purple": "#a78bfa",
        "border-blue-light": "#2e8ec3",
      },
    },
  },
  plugins: [],
};
