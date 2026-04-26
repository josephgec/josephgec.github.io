/* Vol. XXI — A Continent of Latent Space (VAEs) */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK_FADE } = V;

  // Synthetic 2D-latent → 8×8 image decoder. We hand-design 4 "anchor" patterns
  // and blend them according to (z1, z2). This stands in for what a trained decoder
  // would learn from a small dataset of clusters.
  const ANCHORS = {
    // (z1<0, z2<0): cross
    cross:    [[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0]],
    // (z1>0, z2<0): square
    square:   [[1,1,1,1,1,1,1,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,1,1,1,1,1,1,1]],
    // (z1<0, z2>0): circle
    circle:   [[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[1,1,0,0,0,0,1,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,1,0,0,0,0,1,1],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0]],
    // (z1>0, z2>0): triangle
    triangle: [[0,0,0,1,1,0,0,0],[0,0,1,1,1,1,0,0],[0,0,1,1,1,1,0,0],[0,1,1,1,1,1,1,0],[0,1,1,1,1,1,1,0],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]],
  };

  // Decoder: bilinear blend of the four anchors based on (z1, z2) sigmoid-mapped to (0..1).
  function decode(z1, z2) {
    const u = 1 / (1 + Math.exp(-z1)); // 0..1, anchored at z1=0 → 0.5
    const w = 1 / (1 + Math.exp(-z2));
    const out = [];
    for (let i = 0; i < 8; i++) {
      const row = [];
      for (let j = 0; j < 8; j++) {
        const v00 = ANCHORS.cross[i][j];
        const v10 = ANCHORS.square[i][j];
        const v01 = ANCHORS.circle[i][j];
        const v11 = ANCHORS.triangle[i][j];
        const v = (1 - u) * (1 - w) * v00 + u * (1 - w) * v10 + (1 - u) * w * v01 + u * w * v11;
        row.push(v);
      }
      out.push(row);
    }
    return out;
  }

  function gaussian() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  const state = {
    z1: 0,
    z2: 0,
    view: 'manifold', // 'manifold' shows a 2D grid; 'single' shows just the chosen z
  };

  const z1Slider = document.getElementById('z1-slider');
  const z2Slider = document.getElementById('z2-slider');
  const z1Label = document.getElementById('z1-label');
  const z2Label = document.getElementById('z2-label');
  const sampleBtn = document.getElementById('sample-btn');
  const viewToggleEl = document.getElementById('view-toggle');
  const readoutEl = document.getElementById('readout');
  const captionEl = document.getElementById('caption');
  const canvas = document.getElementById('canvas');

  ['Manifold', 'Single'].forEach((lbl, i) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = lbl;
    b.dataset.view = i === 0 ? 'manifold' : 'single';
    b.addEventListener('click', () => { state.view = b.dataset.view; render(); });
    viewToggleEl.appendChild(b);
  });

  z1Slider.addEventListener('input', (e) => { state.z1 = +e.target.value; render(); });
  z2Slider.addEventListener('input', (e) => { state.z2 = +e.target.value; render(); });
  sampleBtn.addEventListener('click', () => {
    state.z1 = Math.max(-3, Math.min(3, gaussian()));
    state.z2 = Math.max(-3, Math.min(3, gaussian()));
    render();
  });

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: true });

  function draw(ctx, size) {
    if (state.view === 'manifold') drawManifold(ctx, size);
    else drawSingle(ctx, size);
  }

  function drawCell(ctx, map, x0, y0, cell) {
    for (let i = 0; i < 8; i++) {
      for (let j = 0; j < 8; j++) {
        const t = Math.max(0, Math.min(1, map[i][j]));
        const r = Math.round(247 - t * 200);
        const g = Math.round(244 - t * 200);
        const b = Math.round(236 - t * 220);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x0 + j * cell, y0 + i * cell, cell + 0.5, cell + 0.5);
      }
    }
  }

  function drawManifold(ctx, size) {
    // 7×7 grid of small decoded outputs across z1, z2 ∈ [-2.5, 2.5]
    const N = 7;
    const tileMargin = 4;
    const tileSize = (size.w - tileMargin * (N + 1)) / N;
    const cell = tileSize / 8;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const z1 = -2.5 + j / (N - 1) * 5;
        const z2 = 2.5 - i / (N - 1) * 5;
        const out = decode(z1, z2);
        const x0 = tileMargin + j * (tileSize + tileMargin);
        const y0 = tileMargin + i * (tileSize + tileMargin);
        drawCell(ctx, out, x0, y0, cell);
      }
    }
    // Highlight the current (z1, z2) cell
    const u = (state.z1 + 2.5) / 5;
    const w = (2.5 - state.z2) / 5;
    const j = Math.round(u * (N - 1));
    const i = Math.round(w * (N - 1));
    const hx = tileMargin + j * (tileSize + tileMargin);
    const hy = tileMargin + i * (tileSize + tileMargin);
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 2.5;
    ctx.strokeRect(hx - 1, hy - 1, tileSize + 2, tileSize + 2);

    // Axis labels
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('z₁ →', size.w - 30, size.h - 6);
    ctx.save();
    ctx.translate(12, 36);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('z₂ →', 0, 0);
    ctx.restore();
  }

  function drawSingle(ctx, size) {
    const out = decode(state.z1, state.z2);
    const cell = (Math.min(size.w, size.h) - 60) / 8;
    const x0 = (size.w - cell * 8) / 2;
    const y0 = (size.h - cell * 8) / 2;
    drawCell(ctx, out, x0, y0, cell);
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x0, y0, cell * 8, cell * 8);
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText(`decoded · z = (${formatNum(state.z1)}, ${formatNum(state.z2)})`,
      size.w / 2, size.h - 14);
  }

  function render() {
    Array.from(viewToggleEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.view === state.view);
    });
    z1Slider.value = state.z1;
    z2Slider.value = state.z2;
    z1Label.textContent = `z₁ = ${formatNum(state.z1)}`;
    z2Label.textContent = `z₂ = ${formatNum(state.z2)}`;
    captionEl.innerHTML = state.view === 'manifold'
      ? '<span class="ml-cap-num">·</span> A grid of decoded outputs across the 2D latent. The whole continent is filled — every (z₁, z₂) decodes to <em>something</em>. The accent box marks the current sample.'
      : '<span class="ml-cap-num">·</span> The decoded output at the chosen latent point. Drag the sliders to navigate the manifold; hit <em>Sample</em> to draw randomly from the prior.';

    // Likelihood under N(0, I)
    const logp = -0.5 * (state.z1 ** 2 + state.z2 ** 2) - Math.log(2 * Math.PI);
    readoutEl.innerHTML = `
      <div class="row"><span>z₁</span><b>${formatNum(state.z1)}</b></div>
      <div class="row"><span>z₂</span><b>${formatNum(state.z2)}</b></div>
      <div class="row"><span>log p(z)</span><b>${formatNum(logp)}</b></div>
      <div class="row"><span>‖z‖</span><b>${formatNum(Math.hypot(state.z1, state.z2))}</b></div>`;

    canvasCtl.redraw();
  }

  render();
})();
