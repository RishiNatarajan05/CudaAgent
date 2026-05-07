/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        nvidia: { 500: "#76b900", 400: "#90d300" },
      },
    },
  },
  plugins: [],
};
