# CNN Digit Explorer

Live app: [cnn-digit-recognition-app.vercel.app](https://cnn-digit-recognition-app.vercel.app/)

This repo now ships as a Vite app that is easy to deploy on Vercel and is ready to load a
TensorFlow.js model in the browser.

The new experience is designed to be:

- public and easy to share
- visually polished and mobile-friendly
- educational, with a clear explanation of what CNNs are and why they matter
- interactive, with a drawing canvas and live demo-style prediction feedback
- model-ready, so a converted CNN can be loaded from `public/model/model.json`

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

## TensorFlow.js Model

To make predictions with your trained CNN, convert `mnist_cnn_model.keras` into TensorFlow.js
artifacts and place them in `public/model/`:

- `model.json`
- `group1-shard*.bin`

The app loads the model from `/model/model.json` in the browser using `tf.loadLayersModel(...)`.

If the converted files are missing, the app will still deploy, but the canvas will show a loading
or unavailable state instead of a real prediction.

## Notes

- The old Streamlit files are still present for reference, but the active app is now the Vite frontend.
- I did not use or store the Gemini API key that was pasted in chat. Please rotate it and add it as an environment variable if you want to integrate Gemini later.
- The app now expects a TensorFlow.js model in `public/model/` to produce real predictions.

## Next Upgrade

If you want, the next step can be generating the TensorFlow.js model files locally and dropping them
into `public/model/` so the live app uses your trained CNN instead of a placeholder state.
