/* Vol. XXII — The Adversaries (GANs) */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK_FADE } = V;

  function gaussian() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // Real-data distributions
  const DISTS = {
    ring: () => {
      const a = Math.random() * Math.PI * 2;
      const r = 1.6 + gaussian() * 0.05;
      return [r * Math.cos(a), r * Math.sin(a)];
    },
    'two clusters': () => {
      const cls = Math.random() < 0.5 ? -1 : 1;
      return [cls * 1.4 + gaussian() * 0.25, cls * 1.4 + gaussian() * 0.25];
    },
    spiral: () => {
      const t = Math.random() * Math.PI * 4;
      const r = t * 0.18;
      return [r * Math.cos(t) + gaussian() * 0.05, r * Math.sin(t) + gaussian() * 0.05];
    },
    grid: () => {
      const i = Math.floor(Math.random() * 3) - 1;
      const j = Math.floor(Math.random() * 3) - 1;
      return [i * 1.2 + gaussian() * 0.1, j * 1.2 + gaussian() * 0.1];
    },
  };

  // We don't actually train networks — we simulate the adversarial dynamic by
  // moving generator points toward real-data density gradients.
  const state = {
    distKey: 'ring',
    real: [],
    fake: [],
    dStrength: 1.0,
    training: false,
    timer: null,
    step: 0,
    accuracy: 0.5,
  };

  function resampleReal() {
    state.real = Array.from({ length: 80 }, () => DISTS[state.distKey]());
  }
  function resampleFake() {
    state.fake = Array.from({ length: 80 }, () => [gaussian() * 0.3, gaussian() * 0.3]);
  }
  resampleReal();
  resampleFake();

  // Discriminator: a soft RBF density estimate of "realness" at any point.
  // Real points contribute positive density; fake points contribute negative.
  function discriminator(x, y) {
    let s = 0;
    const sigma = 0.4 / state.dStrength;
    const eps = 1e-6;
    for (const [rx, ry] of state.real) s += Math.exp(-((x - rx) ** 2 + (y - ry) ** 2) / (2 * sigma * sigma));
    for (const [fx, fy] of state.fake) s -= Math.exp(-((x - fx) ** 2 + (y - fy) ** 2) / (2 * sigma * sigma));
    return s / Math.max(state.real.length, 1);
  }

  // One adversarial step: each fake point moves up the gradient of D (toward "more real").
  function stepGAN() {
    const lr = 0.07;
    const eps = 0.05;
    state.fake = state.fake.map(([x, y]) => {
      const dx = (discriminator(x + eps, y) - discriminator(x - eps, y)) / (2 * eps);
      const dy = (discriminator(x, y + eps) - discriminator(x, y - eps)) / (2 * eps);
      // Add some noise to avoid mode collapse
      return [
        x + lr * dx + (Math.random() - 0.5) * 0.04,
        y + lr * dy + (Math.random() - 0.5) * 0.04,
      ];
    });
    state.step++;
    // Compute accuracy: fraction of points where D's classification matches truth
    let c = 0;
    for (const [rx, ry] of state.real) if (discriminator(rx, ry) > 0) c++;
    for (const [fx, fy] of state.fake) if (discriminator(fx, fy) < 0) c++;
    state.accuracy = c / (state.real.length + state.fake.length);
  }

  const distsEl = document.getElementById('dists');
  const dStrength = document.getElementById('d-strength');
  const dLabel = document.getElementById('d-label');
  const trainBtn = document.getElementById('train-btn');
  const resetBtn = document.getElementById('reset-btn');
  const readoutEl = document.getElementById('readout');
  const canvas = document.getElementById('canvas');

  Object.keys(DISTS).forEach((k) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = k;
    b.dataset.dist = k;
    b.addEventListener('click', () => {
      state.distKey = k;
      resampleReal(); resampleFake();
      state.step = 0;
      stopTrain();
      render();
    });
    distsEl.appendChild(b);
  });

  dStrength.addEventListener('input', (e) => {
    state.dStrength = +e.target.value;
    dLabel.textContent = `D strength = ${state.dStrength.toFixed(1)}`;
    render();
  });

  trainBtn.addEventListener('click', () => {
    if (state.training) stopTrain(); else startTrain();
  });
  resetBtn.addEventListener('click', () => {
    stopTrain();
    resampleReal(); resampleFake();
    state.step = 0;
    render();
  });

  function startTrain() {
    state.training = true;
    trainBtn.textContent = '⏸ Pause';
    state.timer = setInterval(() => { stepGAN(); render(); }, 80);
  }
  function stopTrain() {
    state.training = false;
    trainBtn.textContent = '▷ Train';
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: true });

  function draw(ctx, size) {
    const R = 3;
    const tx = (x) => (x + R) / (2 * R) * size.w;
    const ty = (y) => size.h - (y + R) / (2 * R) * size.h;

    // Discriminator heatmap
    const cells = 50;
    let dMax = 0;
    const grid = [];
    for (let i = 0; i < cells; i++) {
      grid.push([]);
      for (let j = 0; j < cells; j++) {
        const x = -R + (j + 0.5) / cells * 2 * R;
        const y = R - (i + 0.5) / cells * 2 * R;
        const d = discriminator(x, y);
        grid[i].push(d);
        if (Math.abs(d) > dMax) dMax = Math.abs(d);
      }
    }
    for (let i = 0; i < cells; i++) for (let j = 0; j < cells; j++) {
      const d = grid[i][j];
      const t = Math.max(0, Math.min(1, Math.abs(d) / Math.max(0.001, dMax)));
      const color = d > 0
        ? `rgba(122,31,36,${0.05 + t * 0.30})`
        : `rgba(38,35,32,${0.04 + t * 0.20})`;
      ctx.fillStyle = color;
      ctx.fillRect(j * size.w / cells, i * size.h / cells, size.w / cells + 1, size.h / cells + 1);
    }

    // Grid
    ctx.strokeStyle = 'rgba(38,35,32,0.08)';
    ctx.lineWidth = 1;
    for (let i = -3; i <= 3; i++) {
      ctx.beginPath(); ctx.moveTo(tx(i), 0); ctx.lineTo(tx(i), size.h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, ty(i)); ctx.lineTo(size.w, ty(i)); ctx.stroke();
    }

    // Real points
    for (const [x, y] of state.real) {
      ctx.fillStyle = '#262320';
      ctx.beginPath();
      ctx.arc(tx(x), ty(y), 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
    // Fake points
    for (const [x, y] of state.fake) {
      ctx.fillStyle = ACCENT;
      ctx.beginPath();
      ctx.arc(tx(x), ty(y), 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function render() {
    Array.from(distsEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.dist === state.distKey);
    });
    dStrength.value = state.dStrength;
    dLabel.textContent = `D strength = ${state.dStrength.toFixed(1)}`;
    readoutEl.innerHTML = `
      <div class="row"><span>step</span><b>${state.step}</b></div>
      <div class="row"><span>D accuracy</span><b>${(state.accuracy * 100).toFixed(0)}%</b></div>
      <div class="row"><span>real points</span><b>${state.real.length}</b></div>
      <div class="row"><span>fake points</span><b>${state.fake.length}</b></div>`;
    canvasCtl.redraw();
  }

  render();
})();
