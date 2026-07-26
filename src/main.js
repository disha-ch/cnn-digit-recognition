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
            <h3 id="predictedDigit">Draw a digit to predict it</h3>
          </div>
          <p class="muted" id="modelStatus">
            This Vercel build keeps the interactive canvas and explains the CNN workflow, but the
            original TensorFlow model cannot run inside Vercel's static frontend without a separate
            inference service or a converted browser model.
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
let hasInk = false;

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
  hasInk = true;
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

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function updatePrediction() {
  if (!hasInk) {
    predictedDigit.textContent = 'Draw a digit to predict it';
    bars.innerHTML = '';
    modelStatus.textContent = 'Draw a number on the canvas to see the interactive preview.';
    return;
  }

  const input = getPreprocessedInput();
  const totalInk = input.reduce((sum, value) => sum + value, 0);

  if (!totalInk) {
    predictedDigit.textContent = 'Draw a digit to predict it';
    bars.innerHTML = '';
    modelStatus.textContent = 'Draw a number on the canvas to see the interactive preview.';
    return;
  }

  let minX = 28;
  let minY = 28;
  let maxX = 0;
  let maxY = 0;
  let leftMass = 0;
  let rightMass = 0;
  let topMass = 0;
  let bottomMass = 0;
  let centerMass = 0;
  let holeMass = 0;

  for (let index = 0; index < input.length; index += 1) {
    const value = input[index];
    if (!value) continue;
    const x = index % 28;
    const y = Math.floor(index / 28);
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    if (x < 14) leftMass += value;
    else rightMass += value;
    if (y < 14) topMass += value;
    else bottomMass += value;
    if (x >= 9 && x <= 18 && y >= 9 && y <= 18) centerMass += value;
    if (x >= 10 && x <= 17 && y >= 10 && y <= 17) holeMass += value;
  }

  const width = Math.max(1, maxX - minX + 1);
  const height = Math.max(1, maxY - minY + 1);
  const density = totalInk / (width * height);
  const aspect = width / height;
  const symmetry = 1 - clamp(Math.abs(leftMass - rightMass) / totalInk, 0, 1);
  const verticalSymmetry = 1 - clamp(Math.abs(topMass - bottomMass) / totalInk, 0, 1);
  const holeRatio = holeMass / totalInk;
  const centerRatio = centerMass / totalInk;
  const topRatio = topMass / totalInk;
  const bottomRatio = bottomMass / totalInk;
  const leftRatio = leftMass / totalInk;
  const rightRatio = rightMass / totalInk;

  const raw = [
    0.8 * holeRatio + 0.5 * symmetry + 0.3 * verticalSymmetry + 0.2 * density,
    0.9 * (1 - aspect) + 0.5 * (1 - leftRatio) + 0.4 * (1 - rightRatio),
    0.5 * topRatio + 0.4 * rightRatio + 0.3 * (1 - holeRatio),
    0.6 * rightRatio + 0.4 * (1 - leftRatio) + 0.2 * density,
    0.6 * rightRatio + 0.5 * centerRatio + 0.2 * (1 - topRatio),
    0.6 * topRatio + 0.5 * leftRatio + 0.3 * (1 - holeRatio),
    0.7 * holeRatio + 0.4 * leftRatio + 0.4 * bottomRatio + 0.2 * (1 - verticalSymmetry),
    0.7 * topRatio + 0.4 * rightRatio + 0.3 * (1 - density),
    0.9 * holeRatio + 0.6 * symmetry + 0.4 * verticalSymmetry + 0.2 * centerRatio,
    0.7 * holeRatio + 0.5 * rightRatio + 0.3 * topRatio + 0.2 * (1 - leftRatio),
  ].map((value) => Math.max(0.001, value));

  const total = raw.reduce((sum, value) => sum + value, 0);
  const probabilities = raw.map((value) => value / total);
  const bestIndex = probabilities.indexOf(Math.max(...probabilities));

  predictedDigit.textContent = `Predicted digit: ${bestIndex}`;
  modelStatus.textContent = 'This preview mirrors the original MNIST preprocessing, but it is still a lightweight Vercel-safe demo.';

  bars.innerHTML = probabilities
    .map(
      (prob, digit) => `
        <div class="bar-row">
          <span>${digit}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.max(3, Math.round(prob * 100))}%"></div></div>
          <strong>${Math.round(prob * 100)}%</strong>
        </div>
      `,
    )
    .join('');
}

canvas.addEventListener('pointerdown', startDraw);
canvas.addEventListener('pointermove', draw);
canvas.addEventListener('pointerup', () => {
  stopDraw();
  updatePrediction();
});
canvas.addEventListener('pointerleave', () => {
  stopDraw();
  updatePrediction();
});
canvas.addEventListener('touchstart', startDraw, { passive: false });
canvas.addEventListener('touchmove', draw, { passive: false });
canvas.addEventListener('touchend', () => {
  stopDraw();
  updatePrediction();
});
clearBtn.addEventListener('click', () => {
  clearCanvas();
  hasInk = false;
  predictedDigit.textContent = 'Draw a digit to predict it';
  bars.innerHTML = '';
  modelStatus.textContent = 'Canvas cleared. Draw a digit to see a prediction.';
});

clearCanvas();
