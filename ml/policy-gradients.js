/* Vol. XXVIII — Policy Gradients */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK_FADE } = V;

  // 2D continuous action space. Policy is a Gaussian N(μ, σ²I).
  // Reward is a smooth bump centered at the target.
  function reward(a, target) {
    const dx = a[0] - target[0], dy = a[1] - target[1];
    return Math.exp(-(dx * dx + dy * dy) / 1.5);
  }

  function gaussian() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  const TARGETS = {
    'upper right': [1.5, 1.5],
    'lower left':  [-1.5, -1.5],
    'right':       [2.0, 0],
    'origin':      [0, 0],
  };

  const state = {
    targetKey: 'upper right',
    target: [1.5, 1.5],
    mu: [-1.5, -1.5],
    sigma: 0.5,
    lr: 0.05,
    samples: [],
    training: false,
    timer: null,
    step: 0,
    rewardTotal: 0,
  };

  const targetsEl = document.getElementById('targets');
  const lrSlider = document.getElementById('lr-slider');
  const lrLabel = document.getElementById('lr-label');
  const sigmaSlider = document.getElementById('sigma-slider');
  const sigmaLabel = document.getElementById('sigma-label');
  const trainBtn = document.getElementById('train-btn');
  const resetBtn = document.getElementById('reset-btn');
  const readoutEl = document.getElementById('readout');
  const canvas = document.getElementById('canvas');

  Object.entries(TARGETS).forEach(([k, t]) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = k;
    b.dataset.target = k;
    b.addEventListener('click', () => {
      state.targetKey = k;
      state.target = t.slice();
      render();
    });
    targetsEl.appendChild(b);
  });
  lrSlider.addEventListener('input', (e) => {
    state.lr = +e.target.value;
    lrLabel.textContent = `η = ${state.lr.toFixed(3)}`;
  });
  sigmaSlider.addEventListener('input', (e) => {
    state.sigma = +e.target.value;
    sigmaLabel.textContent = `σ = ${state.sigma.toFixed(2)}`;
    render();
  });
  trainBtn.addEventListener('click', () => { if (state.training) stopTrain(); else startTrain(); });
  resetBtn.addEventListener('click', () => {
    stopTrain();
    state.mu = [-1.5, -1.5];
    state.samples = [];
    state.step = 0;
    state.rewardTotal = 0;
    render();
  });

  // Click on canvas to set target
  canvas.addEventListener('click', (e) => {
    const r = canvas.getBoundingClientRect();
    const size = canvasCtl.getSize();
    const R = 3;
    state.target = [
      (e.clientX - r.left) / r.width * 2 * R - R,
      R - (e.clientY - r.top) / r.height * 2 * R,
    ];
    state.targetKey = 'custom';
    render();
  });

  function startTrain() {
    state.training = true;
    trainBtn.textContent = '⏸ Pause';
    state.timer = setInterval(() => {
      // Sample 8 actions, get rewards, compute gradient w.r.t. μ.
      const samples = [];
      let sumGrad = [0, 0];
      let totalR = 0;
      const baseline = state.samples.length > 0
        ? state.samples.reduce((s, p) => s + p.r, 0) / state.samples.length
        : 0;
      for (let i = 0; i < 8; i++) {
        const a = [state.mu[0] + state.sigma * gaussian(), state.mu[1] + state.sigma * gaussian()];
        const r = reward(a, state.target);
        const advantage = r - baseline;
        // ∇_μ log p(a|μ, σ²) = (a - μ) / σ²
        const gx = (a[0] - state.mu[0]) / (state.sigma * state.sigma) * advantage;
        const gy = (a[1] - state.mu[1]) / (state.sigma * state.sigma) * advantage;
        sumGrad[0] += gx; sumGrad[1] += gy;
        totalR += r;
        samples.push({ a, r });
      }
      // Take a gradient step
      state.mu = [state.mu[0] + state.lr * sumGrad[0] / 8,
                  state.mu[1] + state.lr * sumGrad[1] / 8];
      state.samples = samples;
      state.step++;
      state.rewardTotal = totalR / 8;
      render();
    }, 80);
  }
  function stopTrain() {
    state.training = false;
    trainBtn.textContent = '▷ Train';
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: true });
  canvas.style.cursor = 'crosshair';

  function draw(ctx, size) {
    const R = 3;
    const tx = (x) => (x + R) / (2 * R) * size.w;
    const ty = (y) => size.h - (y + R) / (2 * R) * size.h;

    // Reward heatmap (faint)
    const cells = 50;
    for (let i = 0; i < cells; i++) {
      for (let j = 0; j < cells; j++) {
        const x = -R + (j + 0.5) / cells * 2 * R;
        const y = R - (i + 0.5) / cells * 2 * R;
        const r = reward([x, y], state.target);
        ctx.fillStyle = `rgba(122,31,36,${0.04 + r * 0.30})`;
        ctx.fillRect(j * size.w / cells, i * size.h / cells, size.w / cells + 1, size.h / cells + 1);
      }
    }

    // Grid
    ctx.strokeStyle = 'rgba(38,35,32,0.08)';
    ctx.lineWidth = 1;
    for (let i = -3; i <= 3; i++) {
      ctx.beginPath(); ctx.moveTo(tx(i), 0); ctx.lineTo(tx(i), size.h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, ty(i)); ctx.lineTo(size.w, ty(i)); ctx.stroke();
    }

    // Policy contour (1σ, 2σ)
    ctx.strokeStyle = 'rgba(38,35,32,0.55)';
    ctx.lineWidth = 1.5;
    for (const factor of [1, 2]) {
      ctx.beginPath();
      for (let t = 0; t <= Math.PI * 2 + 0.1; t += 0.05) {
        const x = state.mu[0] + state.sigma * factor * Math.cos(t);
        const y = state.mu[1] + state.sigma * factor * Math.sin(t);
        const px = tx(x), py = ty(y);
        if (t === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    // Samples (recent batch)
    for (const { a } of state.samples) {
      ctx.fillStyle = 'rgba(38,35,32,0.45)';
      ctx.beginPath();
      ctx.arc(tx(a[0]), ty(a[1]), 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // μ marker
    ctx.fillStyle = '#262320';
    ctx.beginPath();
    ctx.arc(tx(state.mu[0]), ty(state.mu[1]), 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#fffdf6';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(tx(state.mu[0]), ty(state.mu[1]), 5, 0, Math.PI * 2);
    ctx.stroke();

    // Target star
    ctx.fillStyle = ACCENT;
    const starX = tx(state.target[0]);
    const starY = ty(state.target[1]);
    drawStar(ctx, starX, starY, 9);
  }

  function drawStar(ctx, x, y, r) {
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const a = (i / 10) * Math.PI * 2 - Math.PI / 2;
      const radius = i % 2 === 0 ? r : r * 0.45;
      const px = x + radius * Math.cos(a);
      const py = y + radius * Math.sin(a);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fill();
  }

  function render() {
    Array.from(targetsEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.target === state.targetKey);
    });
    lrLabel.textContent = `η = ${state.lr.toFixed(3)}`;
    sigmaLabel.textContent = `σ = ${state.sigma.toFixed(2)}`;
    lrSlider.value = state.lr;
    sigmaSlider.value = state.sigma;
    readoutEl.innerHTML = `
      <div class="row"><span>μ</span><b>(${formatNum(state.mu[0])}, ${formatNum(state.mu[1])})</b></div>
      <div class="row"><span>target</span><b>(${formatNum(state.target[0])}, ${formatNum(state.target[1])})</b></div>
      <div class="row"><span>step</span><b>${state.step}</b></div>
      <div class="row"><span>last reward</span><b>${formatNum(state.rewardTotal)}</b></div>`;
    canvasCtl.redraw();
  }

  render();
})();
