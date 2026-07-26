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
          <p class="muted">
            This MVP currently uses an in-browser heuristic preview. It keeps the app fast and
            Vercel-friendly while we prepare a production-grade model backend.
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

function getImageData() {
  const image = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  let ink = 0;
  let centroidX = 0;
  let centroidY = 0;
  let top = canvas.height;
  let bottom = 0;
  let left = canvas.width;
  let right = 0;

  for (let y = 0; y < canvas.height; y += 1) {
    for (let x = 0; x < canvas.width; x += 1) {
      const idx = (y * canvas.width + x) * 4;
      const brightness = image[idx];
      const dark = 255 - brightness;
      if (dark > 30) {
        ink += dark;
        centroidX += x * dark;
        centroidY += y * dark;
        left = Math.min(left, x);
        right = Math.max(right, x);
        top = Math.min(top, y);
        bottom = Math.max(bottom, y);
      }
    }
  }

  if (!ink) {
    return { ink: 0, centroidX: 0, centroidY: 0, top: 0, bottom: 0, left: 0, right: 0 };
  }

  return {
    ink,
    centroidX: centroidX / ink,
    centroidY: centroidY / ink,
    top,
    bottom,
    left,
    right,
  };
}

function scoreDigit(metrics, digit) {
  if (!metrics.ink) return digit === 8 ? 1 : 0.1;
  const { centroidX, centroidY, top, bottom, left, right } = metrics;
  const width = Math.max(1, right - left);
  const height = Math.max(1, bottom - top);
  const centerBias = 1 - (Math.abs(centroidX - 140) + Math.abs(centroidY - 140)) / 280;
  const tallness = height / width;
  const widthness = width / height;
  const vertical = 1 - Math.abs(tallness - 1.1);
  const round = 1 - Math.abs(widthness - 0.9);

  const baseScores = [0.9, 0.5, 0.6, 0.7, 0.45, 0.8, 0.55, 0.65, 0.95, 0.75];
  const shapeInfluence = [
    centerBias,
    round,
    vertical,
    centerBias * 0.8,
    widthness,
    round * 0.95,
    vertical * 0.9,
    0.6 + centerBias * 0.2,
    1 - Math.abs(widthness - 1),
    0.7 + centerBias * 0.15,
  ];

  return Math.max(0.01, baseScores[digit] * shapeInfluence[digit]);
}

function updatePrediction() {
  const metrics = getImageData();
  const raw = digits.map((digit) => scoreDigit(metrics, digit));
  const total = raw.reduce((sum, value) => sum + value, 0);
  const probabilities = raw.map((value) => value / total);
  const bestIndex = probabilities.indexOf(Math.max(...probabilities));
  predictedDigit.textContent = `Predicted digit: ${bestIndex}`;

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
