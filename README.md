# CNN Digit Explorer

This repo now ships as a Vite app that is easy to deploy on Vercel.

The new experience is designed to be:

- public and easy to share
- visually polished and mobile-friendly
- educational, with a clear explanation of what CNNs are and why they matter
- interactive, with a drawing canvas and live demo-style prediction feedback

## Local Development

```bash
npm install
npm run dev
```

## Production Build

```bash
npm run build
```

## Deployment

This project is Vercel-friendly out of the box:

- `index.html` is the app entry point
- `src/main.js` contains the interactive UI
- `src/style.css` contains the full visual system
- `vercel.json` configures the static build output

## Notes

- The old Streamlit files are still present for reference, but the active app is now the Vite frontend.
- I did not use or store the Gemini API key that was pasted in chat. Please rotate it and add it as an environment variable if you want to integrate Gemini later.
- The current canvas prediction is a lightweight browser-side MVP so the site stays fast and deployable everywhere.

## Next Upgrade

If you want, the next step can be adding a real model-backed inference path so the canvas uses your trained CNN instead of the heuristic demo output.
