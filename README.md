# CNN Digit Explorer

Live app: [cnn-digit-recognition-app.vercel.app](https://cnn-digit-recognition-app.vercel.app/)

This repo now ships as a Vite app that is easy to deploy on Vercel and can use a Gemini-backed
serverless function for digit predictions.

The new experience is designed to be:

- public and easy to share
- visually polished and mobile-friendly
- educational, with a clear explanation of what CNNs are and why they matter
- interactive, with a drawing canvas and live demo-style prediction feedback
- model-backed through a Vercel function, with Gemini as the classifier when configured

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
- `public/model/` is where the TensorFlow.js model files should live

## Predictions

The app now sends the canvas image to `/api/predict`.

That serverless function:

- accepts the drawn digit as a PNG data URL
- calls Gemini using the `GEMINI_API_KEY` environment variable
- returns a predicted digit plus probability bars

If `GEMINI_API_KEY` is not set in Vercel, the app will still deploy, but prediction requests will
show an error message until the key is added.

## Notes

- The old Streamlit files are still present for reference, but the active app is now the Vite frontend.
- I did not use or store the Gemini API key that was pasted in chat. Please rotate it and add it as
  the `GEMINI_API_KEY` environment variable in Vercel.

## Next Upgrade

If you want, the next step can be generating the TensorFlow.js model files locally and dropping them
into `public/model/` so the live app uses your trained CNN instead of a placeholder state.
