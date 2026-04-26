/* Vol. XXXII — Bias-Variance Tradeoff */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK_FADE } = V;

  const trueF = (x) => Math.sin(Math.PI * x);

  function gaussian() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // Generate n training samples uniformly in [-1, 1]
  function makeData(n, sigma) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const x = -1 + 2 * Math.random();
      pts.push({ x, y: trueF(x) + sigma * gaussian() });
    }
    return pts;
  }

  // Fit a polynomial of given degree using normal equations.
  // Solves (X^T X) w = X^T y where X has columns 1, x, x^2, ..., x^degree.
  function fitPoly(pts, deg) {
    const n = pts.length;
    const dim = deg + 1;
    // Build X^T X (dim×dim) and X^T y (dim)
    const A = Array.from({ length: dim }, () => new Array(dim).fill(0));
    const b = new Array(dim).fill(0);
    for (let k = 0; k < n; k++) {
      const x = pts[k].x, y = pts[k].y;
      const phi = new Array(dim);
      for (let j = 0; j < dim; j++) phi[j] = Math.pow(x, j);
      for (let i = 0; i < dim; i++) {
        for (let j = 0; j < dim; j++) A[i][j] += phi[i] * phi[j];
        b[i] += phi[i] * y;
      }
    }
    // Add tiny ridge to keep matrix invertible at high degrees
    for (let i = 0; i < dim; i++) A[i][i] += 1e-6;
    return solveLinear(A, b);
  }

  // Gauss-Jordan elimination
  function solveLinear(A, b) {
    const n = A.length;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let i = 0; i < n; i++) {
      let pivot = i;
      for (let k = i + 1; k < n; k++) if (Math.abs(M[k][i]) > Math.abs(M[pivot][i])) pivot = k;
      [M[i], M[pivot]] = [M[pivot], M[i]];
      const p = M[i][i];
      if (Math.abs(p) < 1e-12) return new Array(n).fill(0);
      for (let k = 0; k <= n; k++) M[i][k] /= p;
      for (let r = 0; r < n; r++) {
        if (r === i) continue;
        const f = M[r][i];
        for (let k = 0; k <= n; k++) M[r][k] -= f * M[i][k];
      }
    }
    return M.map((row) => row[n]);
  }

  function evalPoly(w, x) {
    let s = 0;
    for (let j = 0; j < w.length; j++) s += w[j] * Math.pow(x, j);
    return s;
  }

  const state = {
    deg: 3,
    sigma: 0.25,
    n: 20,
    seed: 1,
  };

  const degSlider = document.getElementById('deg-slider');
  const degLabel = document.getElementById('deg-label');
  const noiseSlider = document.getElementById('noise-slider');
  const noiseLabel = document.getElementById('noise-label');
  const nSlider = document.getElementById('n-slider');
  const nLabel = document.getElementById('n-label');
  const resampleBtn = document.getElementById('resample');
  const readoutEl = document.getElementById('readout');
  const canvas = document.getElementById('canvas');

  degSlider.addEventListener('input', (e) => { state.deg = +e.target.value; render(); });
  noiseSlider.addEventListener('input', (e) => { state.sigma = +e.target.value; render(); });
  nSlider.addEventListener('input', (e) => { state.n = +e.target.value; render(); });
  resampleBtn.addEventListener('click', () => { state.seed++; render(); });

  // Use seed to make Math.random deterministic per render? Not strictly needed; just bump on resample.
  // For consistency we reuse JS Math.random but reseed implicitly via repeated calls per render.

  // ───────── Compute multiple fits for variance display ─────────
  function computeFits() {
    // Generate 10 different training sets and fit each
    const fits = [];
    for (let i = 0; i < 10; i++) {
      const pts = makeData(state.n, state.sigma);
      const w = fitPoly(pts, state.deg);
      fits.push({ pts, w });
    }
    return fits;
  }

  // Bias/variance decomposition per degree
  function bvDecomposition() {
    const xs = [];
    for (let i = 0; i < 50; i++) xs.push(-1 + 2 * i / 49);
    const trueY = xs.map(trueF);
    const out = [];
    for (let d = 1; d <= 15; d++) {
      // Fit on 30 different datasets, average the predictions
      const preds = [];
      for (let k = 0; k < 30; k++) {
        const pts = makeData(state.n, state.sigma);
        const w = fitPoly(pts, d);
        preds.push(xs.map((x) => evalPoly(w, x)));
      }
      const avg = xs.map((_, j) => preds.reduce((s, p) => s + p[j], 0) / preds.length);
      let bias2 = 0, variance = 0;
      for (let j = 0; j < xs.length; j++) {
        bias2 += (avg[j] - trueY[j]) ** 2;
        let v = 0;
        for (const p of preds) v += (p[j] - avg[j]) ** 2;
        variance += v / preds.length;
      }
      bias2 /= xs.length; variance /= xs.length;
      out.push({ degree: d, bias2, variance, total: bias2 + variance + state.sigma * state.sigma });
    }
    return out;
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    const padX = 36;
    const topH = size.h * 0.55;
    const botY = topH + 28;
    const botH = size.h - botY - 36;

    // Top: data + fits
    const tx = (x) => padX + (x + 1) * (size.w - padX * 2) / 2;
    const ty = (y) => topH - 16 + 16 - (y + 1.5) * (topH - 32) / 3;

    // Frame
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.strokeRect(padX, 16, size.w - padX * 2, topH - 32);

    // True function
    ctx.strokeStyle = 'rgba(38,35,32,0.45)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    for (let i = 0; i <= 100; i++) {
      const x = -1 + 2 * i / 100;
      const y = trueF(x);
      const px = tx(x), py = ty(y);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Fits
    const fits = computeFits();
    fits.forEach((fit, idx) => {
      ctx.strokeStyle = idx === 0 ? ACCENT : `rgba(122,31,36,0.18)`;
      ctx.lineWidth = idx === 0 ? 2 : 1;
      ctx.beginPath();
      for (let i = 0; i <= 100; i++) {
        const x = -1 + 2 * i / 100;
        const y = evalPoly(fit.w, x);
        const px = tx(x), py = ty(Math.max(-1.5, Math.min(1.5, y)));
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
    });
    // Data points (from first dataset)
    fits[0].pts.forEach(({ x, y }) => {
      ctx.fillStyle = '#262320';
      ctx.beginPath();
      ctx.arc(tx(x), ty(y), 3, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText(`degree-${state.deg} polynomial · 10 fits on different training sets · accent = current`, padX, 12);

    // Bottom: bias / variance / total over degrees
    const bv = bvDecomposition();
    const plotX0 = padX + 30;
    const plotW = size.w - plotX0 - padX;
    const maxErr = Math.max(...bv.map((d) => Math.max(d.bias2, d.variance, d.total))) + 0.05;
    const xMap = (deg) => plotX0 + (deg - 1) / 14 * plotW;
    const yMap = (e) => botY + botH - (e / maxErr) * botH;

    // Frame
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.beginPath();
    ctx.moveTo(plotX0, botY); ctx.lineTo(plotX0, botY + botH);
    ctx.lineTo(plotX0 + plotW, botY + botH);
    ctx.stroke();

    // Y labels
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    [0, maxErr / 2, maxErr].forEach((y) => {
      ctx.fillText(formatNum(y), plotX0 - 6, yMap(y) + 3);
    });

    // X labels (degrees)
    ctx.textAlign = 'center';
    [1, 5, 10, 15].forEach((d) => {
      ctx.fillText(d, xMap(d), botY + botH + 14);
    });

    // bias² (ink)
    ctx.strokeStyle = '#262320';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    bv.forEach((d, i) => {
      const x = xMap(d.degree), y = yMap(d.bias2);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // variance (oxblood)
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    bv.forEach((d, i) => {
      const x = xMap(d.degree), y = yMap(d.variance);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // total (dashed)
    ctx.strokeStyle = 'rgba(38,35,32,0.55)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    bv.forEach((d, i) => {
      const x = xMap(d.degree), y = yMap(d.total);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // Mark current degree
    const cx = xMap(state.deg);
    ctx.strokeStyle = 'rgba(122,31,36,0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 4]);
    ctx.beginPath();
    ctx.moveTo(cx, botY); ctx.lineTo(cx, botY + botH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Legend
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    let legX = plotX0 + plotW - 220, legY = botY + 14;
    const legend = [
      ['bias²', '#262320'],
      ['variance', ACCENT],
      ['total', 'rgba(38,35,32,0.55)'],
    ];
    legend.forEach(([lbl, color], i) => {
      ctx.fillStyle = color;
      ctx.fillRect(legX, legY + i * 14 - 8, 20, 2);
      ctx.fillStyle = INK_FADE;
      ctx.fillText(lbl, legX + 28, legY + i * 14 - 4);
    });

    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('error decomposition over polynomial degree', plotX0, botY - 6);
  }

  function render() {
    degSlider.value = state.deg;
    degLabel.textContent = `degree ${state.deg}`;
    noiseSlider.value = state.sigma;
    noiseLabel.textContent = `σ = ${state.sigma.toFixed(2)}`;
    nSlider.value = state.n;
    nLabel.textContent = `n = ${state.n}`;

    // Approximate bias-variance at current degree (cheap recomputation)
    const sample = bvDecomposition().find((d) => d.degree === state.deg);
    readoutEl.innerHTML = `
      <div class="row"><span>bias²</span><b>${formatNum(sample.bias2)}</b></div>
      <div class="row"><span>variance</span><b>${formatNum(sample.variance)}</b></div>
      <div class="row"><span>noise σ²</span><b>${formatNum(state.sigma * state.sigma)}</b></div>
      <div class="row"><span>total err</span><b>${formatNum(sample.total)}</b></div>`;
    canvasCtl.redraw();
  }

  render();
})();
