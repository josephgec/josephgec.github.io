/* Vol. XIX — Holding it Together (Dropout / BatchNorm) */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, INK_FADE } = V;

  const state = {
    p: 0.3,
    bnOn: true,
    seed: 1,
  };

  const pSlider = document.getElementById('p-slider');
  const pLabel = document.getElementById('p-label');
  const bnToggleEl = document.getElementById('bn-toggle');
  const resampleBtn = document.getElementById('resample');
  const readoutEl = document.getElementById('readout');
  const canvas = document.getElementById('canvas');

  ['BN on', 'BN off'].forEach((lbl, i) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = lbl;
    b.dataset.bn = i === 0 ? 'on' : 'off';
    b.addEventListener('click', () => { state.bnOn = i === 0; render(); });
    bnToggleEl.appendChild(b);
  });

  pSlider.addEventListener('input', (e) => {
    state.p = +e.target.value;
    pLabel.textContent = `p = ${state.p.toFixed(2)}`;
    render();
  });
  resampleBtn.addEventListener('click', () => {
    state.seed = Math.floor(Math.random() * 1000);
    render();
  });

  // ───────── Synthetic 4-layer activation simulation ─────────
  function seeded(seed) {
    let s = seed;
    return () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
  }
  function gaussianFromUniform(rnd) {
    let u = 0, w = 0;
    while (u === 0) u = rnd();
    while (w === 0) w = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * w);
  }

  function simulate() {
    const rnd = seeded(state.seed);
    const layers = 4;
    const units = 12;
    const acts = [];
    // Layer 0: roughly N(0,1)
    const a0 = Array.from({ length: units }, () => gaussianFromUniform(rnd));
    acts.push(a0);
    for (let li = 1; li < layers; li++) {
      const prev = acts[li - 1];
      // Random affine: scale and shift in a way that drifts (without BN)
      const scale = 1.2;  // slightly amplifies
      const shift = 0.4;
      const ax = prev.map((v) => Math.max(0, v * scale + shift + gaussianFromUniform(rnd) * 0.3));
      let result = ax;
      if (state.bnOn) {
        // Standardize: subtract mean, divide by std
        const m = ax.reduce((a, b) => a + b, 0) / ax.length;
        const v = ax.reduce((a, b) => a + (b - m) ** 2, 0) / ax.length;
        const sd = Math.sqrt(v + 1e-6);
        result = ax.map((x) => (x - m) / sd);
      }
      acts.push(result);
    }
    // Apply dropout mask to displayed acts (training-time view)
    const masks = acts.map((layer) => layer.map(() => (rnd() > state.p ? 1 : 0)));
    const masked = acts.map((layer, li) => layer.map((v, i) => v * masks[li][i] / Math.max(0.001, 1 - state.p)));
    return { acts, masked, masks };
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    const sim = simulate();
    const layers = sim.acts.length;
    const units = sim.acts[0].length;

    // Top half: activation grid (units × layers, with dropout mask shown)
    const topH = size.h * 0.55;
    const padX = 60;
    const cellW = (size.w - padX - 24) / layers;
    const cellH = (topH - 30) / units;

    ctx.fillStyle = INK_FADE;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    for (let li = 0; li < layers; li++) {
      ctx.fillText(li === 0 ? 'input' : `layer ${li}`, padX + li * cellW + cellW / 2, 16);
    }

    for (let li = 0; li < layers; li++) {
      for (let i = 0; i < units; i++) {
        const v = sim.masked[li][i];
        const masked = sim.masks[li][i] === 0;
        const intensity = Math.min(1, Math.abs(v) / 2);
        ctx.fillStyle = masked
          ? 'rgba(38,35,32,0.08)'
          : (v >= 0 ? `rgba(122,31,36,${0.10 + intensity * 0.7})` : `rgba(38,35,32,${0.10 + intensity * 0.7})`);
        ctx.fillRect(padX + li * cellW + 4, 26 + i * cellH, cellW - 8, cellH - 2);
        if (masked) {
          ctx.strokeStyle = 'rgba(38,35,32,0.5)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(padX + li * cellW + 6, 26 + i * cellH + cellH / 2);
          ctx.lineTo(padX + li * cellW + cellW - 8, 26 + i * cellH + cellH / 2);
          ctx.stroke();
        }
      }
    }

    // Bottom half: distributions per layer
    const bottomY = topH + 24;
    const bottomH = size.h - bottomY - 8;
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('activation distribution per layer', size.w / 2, bottomY - 6);

    for (let li = 0; li < layers; li++) {
      const layer = sim.acts[li];
      const xMin = -3, xMax = 6;
      const x0 = padX + li * cellW + 4;
      const w = cellW - 8;
      const tx = (x) => x0 + (x - xMin) / (xMax - xMin) * w;
      ctx.strokeStyle = 'rgba(38,35,32,0.20)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x0, bottomY, w, bottomH);
      // Zero line
      ctx.strokeStyle = 'rgba(38,35,32,0.40)';
      ctx.beginPath();
      ctx.moveTo(tx(0), bottomY); ctx.lineTo(tx(0), bottomY + bottomH);
      ctx.stroke();
      // Plot the layer's individual values as ticks
      layer.forEach((v) => {
        const px = tx(Math.max(xMin, Math.min(xMax, v)));
        ctx.strokeStyle = 'rgba(122,31,36,0.65)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(px, bottomY + bottomH - 4);
        ctx.lineTo(px, bottomY + bottomH - 14);
        ctx.stroke();
      });
      // Mean and std summary
      const m = layer.reduce((a, b) => a + b, 0) / layer.length;
      const sd = Math.sqrt(layer.reduce((a, b) => a + (b - m) ** 2, 0) / layer.length);
      ctx.fillStyle = INK_FADE;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`μ=${formatNum(m)} σ=${formatNum(sd)}`, x0 + w / 2, bottomY + bottomH + 14);
    }
  }

  function render() {
    pLabel.textContent = `p = ${state.p.toFixed(2)}`;
    Array.from(bnToggleEl.children).forEach((btn) => {
      btn.classList.toggle('active', (btn.dataset.bn === 'on') === state.bnOn);
    });
    const sim = simulate();
    const liveCount = sim.masks.flat().filter((m) => m === 1).length;
    const total = sim.masks.flat().length;
    readoutEl.innerHTML = `
      <div class="row"><span>active units</span><b>${liveCount} / ${total}</b></div>
      <div class="row"><span>BN</span><b>${state.bnOn ? 'on' : 'off'}</b></div>`;
    canvasCtl.redraw();
  }

  render();
})();
