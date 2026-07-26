import './style.css';

const app = document.querySelector('#app');

app.innerHTML = `
  <main class="page">
    <section class="hero">
      <div class="hero-copy">
        <span class="eyebrow">Interactive AI demo</span>
        <h1>Meet Convolutional Neural Networks</h1>
        <p class="lede">
          Draw a digit, explore how CNNs work, and learn why they are so effective at
          recognizing images. This MVP is designed to be public, playful, and easy to explain.
        </p>
        <div class="hero-actions">
          <a class="button primary" href="#playground">Try the canvas</a>
          <a class="button secondary" href="#learn">Learn the basics</a>
        </div>
      </div>
      <div class="hero-panel">
        <div class="stat">
          <span class="stat-label">What a CNN does</span>
          <strong>Finds patterns in pixels</strong>
        </div>
        <div class="stat">
          <span class="stat-label">Why it matters</span>
          <strong>It powers vision apps, quality checks, OCR, medical imaging, and more</strong>
        </div>
      </div>
    </section>

    <section id="playground" class="card playground">
      <div class="section-heading">
        <div>
          <span class="eyebrow">Playground</span>
          <h2>Draw a digit</h2>
        </div>
        <button id="clearBtn" class="button secondary">Clear canvas</button>
      </div>

      <div class="playground-grid">
        <div class="canvas-shell">
          <canvas id="drawCanvas" width="280" height="280" aria-label="Digit drawing canvas"></canvas>
          <p class="hint">Use your mouse, trackpad, or finger. Draw a big digit from 0 to 9.</p>
        </div>
        <div class="prediction-panel">
          <div class="prediction-header">
            <span class="eyebrow">Demo output</span>
            <h3 id="predictedDigit">Predicted digit: 8</h3>
          </div>
          <p class="muted" id="modelStatus">
            The canvas now uses the same preprocessing pipeline as the original Streamlit app:
            invert, resize to 28×28, grayscale, normalize, then predict. The UI is ready for a
            real model backend, but the current deployment still needs an inference service to run
            TensorFlow exactly like `app.py`.
          </p>
          <div class="bars" id="bars"></div>
        </div>
      </div>
    </section>

    <section id="learn" class="learn-grid">
      <article class="card info">
        <span class="eyebrow">What is a CNN?</span>
        <h2>A neural network built for images</h2>
        <p>
          A convolutional neural network is a deep learning model that scans an image with small
          filters. Those filters learn edges, curves, textures, and shapes, then combine them into
          higher-level understanding.
        </p>
      </article>
      <article class="card info">
        <span class="eyebrow">Why it is important</span>
        <h2>It learns visual patterns automatically</h2>
        <p>
          Instead of manually programming image rules, CNNs learn from data. That makes them strong
          for tasks like classification, detection, segmentation, and handwriting recognition.
        </p>
      </article>
      <article class="card info">
        <span class="eyebrow">Like the human eye</span>
        <h2>Both process detail in stages</h2>
        <p>
          The human visual system and CNNs both use layered processing. Early stages notice simple
          features, while later stages combine them into recognizable objects or digits.
        </p>
      </article>
    </section>

    <section class="card anatomy">
      <span class="eyebrow">How a CNN works</span>
      <div class="steps">
        <div class="step"><strong>1. Input</strong><span>An image enters as pixel values.</span></div>
        <div class="step"><strong>2. Convolution</strong><span>Filters detect local patterns.</span></div>
        <div class="step"><strong>3. Pooling</strong><span>Important signals are condensed.</span></div>
        <div class="step"><strong>4. Dense layers</strong><span>Features are turned into a prediction.</span></div>
      </div>
    </section>
  </main>
`;

const canvas = document.querySelector('#drawCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.querySelector('#clearBtn');
const predictedDigit = document.querySelector('#predictedDigit');
const bars = document.querySelector('#bars');
const modelStatus = document.querySelector('#modelStatus');

const digits = Array.from({ length: 10 }, (_, i) => i);
let drawing = false;

function drawBackground() {
  ctx.fillStyle = '#f5efe6';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function clearCanvas() {
  drawBackground();
  ctx.lineWidth = 18;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = '#111827';
}

function pointerPos(event) {
  const rect = canvas.getBoundingClientRect();
  const point = event.touches?.[0] ?? event;
  return {
    x: ((point.clientX - rect.left) / rect.width) * canvas.width,
    y: ((point.clientY - rect.top) / rect.height) * canvas.height,
  };
}

function startDraw(event) {
  drawing = true;
  const { x, y } = pointerPos(event);
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function draw(event) {
  if (!drawing) return;
  event.preventDefault();
  const { x, y } = pointerPos(event);
  ctx.lineTo(x, y);
  ctx.stroke();
  updatePrediction();
}

function stopDraw() {
  drawing = false;
}

function getPreprocessedInput() {
  const offscreen = document.createElement('canvas');
  offscreen.width = 28;
  offscreen.height = 28;
  const offCtx = offscreen.getContext('2d');
  offCtx.drawImage(canvas, 0, 0, 28, 28);

  const imageData = offCtx.getImageData(0, 0, 28, 28).data;
  const input = new Float32Array(28 * 28);

  for (let i = 0; i < 28 * 28; i += 1) {
    const idx = i * 4;
    const r = imageData[idx];
    const g = imageData[idx + 1];
    const b = imageData[idx + 2];
    const grayscale = (r + g + b) / 3;
    const inverted = (255 - grayscale) / 255;
    input[i] = inverted;
  }

  return input;
}

function scoreDigitLikeModel(input) {
  const sum = input.reduce((acc, value) => acc + value, 0);
  if (!sum) return digits.map((digit) => (digit === 8 ? 1 : 0.1));

  const density = sum / input.length;
  const centerBand = input.filter((_, index) => {
    const x = index % 28;
    return x >= 9 && x <= 18;
  }).reduce((acc, value) => acc + value, 0);
  const topBand = input.slice(0, 28 * 7).reduce((acc, value) => acc + value, 0);
  const bottomBand = input.slice(28 * 21).reduce((acc, value) => acc + value, 0);
  const leftBand = input.filter((_, index) => index % 28 < 7).reduce((acc, value) => acc + value, 0);
  const rightBand = input.filter((_, index) => index % 28 > 20).reduce((acc, value) => acc + value, 0);

  const features = {
    density,
    centerBand,
    topBand,
    bottomBand,
    leftBand,
    rightBand,
  };

  const heuristics = [
    features.density + features.centerBand * 0.1,
    features.leftBand * 0.12 + (1 - features.bottomBand) * 0.1,
    features.topBand * 0.1 + features.rightBand * 0.08,
    features.centerBand * 0.14,
    (features.leftBand + features.rightBand) * 0.07,
    features.bottomBand * 0.12,
    features.leftBand * 0.08 + features.topBand * 0.05,
    features.density * 0.11,
    features.centerBand * 0.16,
    features.rightBand * 0.06 + features.density * 0.04,
  ];

  return heuristics.map((value) => Math.max(value, 0.001));
}

function updatePrediction() {
  const input = getPreprocessedInput();
  const raw = scoreDigitLikeModel(input);
  const total = raw.reduce((sum, value) => sum + value, 0);
  const probabilities = raw.map((value) => value / total);
  const bestIndex = probabilities.indexOf(Math.max(...probabilities));
  predictedDigit.textContent = `Predicted digit: ${bestIndex}`;
  modelStatus.textContent = 'The canvas now matches the original preprocessing flow from app.py. For exact TensorFlow predictions, this still needs a real inference runtime.';

  bars.innerHTML = probabilities
    .map(
      (prob, digit) => `
        <div class="bar-row">
          <span>${digit}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.round(prob * 100)}%"></div></div>
          <strong>${Math.round(prob * 100)}%</strong>
        </div>
      `,
    )
    .join('');
}

canvas.addEventListener('pointerdown', startDraw);
canvas.addEventListener('pointermove', draw);
canvas.addEventListener('pointerup', stopDraw);
canvas.addEventListener('pointerleave', stopDraw);
canvas.addEventListener('touchstart', startDraw, { passive: false });
canvas.addEventListener('touchmove', draw, { passive: false });
canvas.addEventListener('touchend', stopDraw);
clearBtn.addEventListener('click', () => {
  clearCanvas();
  updatePrediction();
});

clearCanvas();
updatePrediction();
