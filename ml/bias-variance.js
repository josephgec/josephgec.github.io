/* Vol. XXXII — Bias-Variance Tradeoff */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK_FADE } = V;

  const trueF = (x) => Math.sin(Math.PI * x);

  // Mulberry32 deterministic RNG so resampling is bumped by `seed` only,
  // and noise residuals match the data points across renders.
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussFromRng(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // Generate n training samples uniformly in [-1, 1] using a seeded RNG.
  function makeData(n, sigma, seed) {
    const rng = mulberry32(seed);
    const pts = [];
    for (let i = 0; i < n; i++) {
      const x = -1 + 2 * rng();
      const noise = sigma * gaussFromRng(rng);
      pts.push({ x, y: trueF(x) + noise, noise });
    }
    return pts;
  }

  // Fit a polynomial of given degree using normal equations.
  function fitPoly(pts, deg) {
    const n = pts.length;
    const dim = deg + 1;
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
    for (let i = 0; i < dim; i++) A[i][i] += 1e-6;
    return solveLinear(A, b);
  }

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
    // Default to a moderate degree where both bias² and variance are visibly
    // non-trivial — the sweet middle of the U is the most instructive landing
    // state. Degree 6 sits past the underfit shoulder; the spread band is
    // already widening and the bottom plot's bias and variance lines are both
    // legibly above zero.
    deg: 6,
    sigma: 0.25,
    n: 20,
    seed: 1,
    showNoise: false,
  };

  const degSlider = document.getElementById('deg-slider');
  const degLabel = document.getElementById('deg-label');
  const noiseSlider = document.getElementById('noise-slider');
  const noiseLabel = document.getElementById('noise-label');
  const nSlider = document.getElementById('n-slider');
  const nLabel = document.getElementById('n-label');
  const resampleBtn = document.getElementById('resample');
  const showNoiseEl = document.getElementById('show-noise');
  const readoutEl = document.getElementById('readout');
  const canvas = document.getElementById('canvas');

  degSlider.addEventListener('input', (e) => { state.deg = +e.target.value; render(); });
  noiseSlider.addEventListener('input', (e) => { state.sigma = +e.target.value; render(); });
  nSlider.addEventListener('input', (e) => { state.n = +e.target.value; render(); });
  resampleBtn.addEventListener('click', () => { state.seed += 17; render(); });
  if (showNoiseEl) showNoiseEl.addEventListener('change', (e) => { state.showNoise = e.target.checked; render(); });

  // ───────── Compute multiple fits for variance display ─────────
  function computeFits() {
    const fits = [];
    for (let i = 0; i < 10; i++) {
      const pts = makeData(state.n, state.sigma, state.seed * 1000 + i);
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
      const preds = [];
      for (let k = 0; k < 30; k++) {
        const pts = makeData(state.n, state.sigma, state.seed * 1000 + 100 + d * 31 + k);
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

    // Compute spread-band of fits at each x: capture min/max y across the 10 fits.
    // We'll use this band to draw the "spread footprint" as a faint translucent envelope —
    // this makes "variance = the spread-of-fits" visually concrete.
    const bandXs = [];
    const bandHi = [];
    const bandLo = [];
    for (let i = 0; i <= 100; i++) {
      const x = -1 + 2 * i / 100;
      let lo = Infinity, hi = -Infinity;
      for (const fit of fits) {
        const yv = Math.max(-1.5, Math.min(1.5, evalPoly(fit.w, x)));
        if (yv < lo) lo = yv; if (yv > hi) hi = yv;
      }
      bandXs.push(x); bandLo.push(lo); bandHi.push(hi);
    }
    // Fill the band
    ctx.fillStyle = 'rgba(122,31,36,0.10)';
    ctx.beginPath();
    for (let i = 0; i < bandXs.length; i++) {
      const px = tx(bandXs[i]), py = ty(bandHi[i]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    for (let i = bandXs.length - 1; i >= 0; i--) {
      const px = tx(bandXs[i]), py = ty(bandLo[i]);
      ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fill();

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

    // Data points (from first dataset). When show-noise is ON, draw noise as vertical sticks
    // from the true sin(πx) curve to the noisy sample — making σ² visible.
    fits[0].pts.forEach(({ x, y, noise }) => {
      if (state.showNoise) {
        const yTrue = trueF(x);
        ctx.strokeStyle = 'rgba(38,35,32,0.45)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(tx(x), ty(yTrue));
        ctx.lineTo(tx(x), ty(y));
        ctx.stroke();
        // small open dot at the true point
        ctx.fillStyle = 'rgba(38,35,32,0.45)';
        ctx.beginPath();
        ctx.arc(tx(x), ty(yTrue), 1.6, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.fillStyle = '#262320';
      ctx.beginPath();
      ctx.arc(tx(x), ty(y), 3, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    const headerSuffix = state.showNoise
      ? `· dashed = true sin(πx); sticks = noise (σ = ${state.sigma.toFixed(2)}, σ² = ${formatNum(state.sigma*state.sigma)})`
      : '· faint band = spread across 10 fits = variance';
    ctx.fillText(`degree-${state.deg} polynomial · 10 fits on different training sets ${headerSuffix}`, padX, 12);

    // Mini legend in the top panel (top-right): three swatches for true / fits / band.
    {
      const legX0 = size.w - padX - 168;
      const legY0 = 16 + 4;
      const items = [
        { label: 'true f(x) = sin(πx)', dash: true,  color: 'rgba(38,35,32,0.55)' },
        { label: 'one fit (this sample)', dash: false, color: ACCENT },
        { label: 'spread over 10 fits',   dash: false, color: 'rgba(122,31,36,0.30)', band: true },
      ];
      ctx.font = 'italic 10px "Source Serif 4", Georgia, serif';
      items.forEach((it, i) => {
        const yy = legY0 + i * 13;
        if (it.band) {
          ctx.fillStyle = 'rgba(122,31,36,0.18)';
          ctx.fillRect(legX0, yy - 4, 22, 8);
        } else {
          ctx.strokeStyle = it.color;
          ctx.lineWidth = it.dash ? 1.4 : 2;
          if (it.dash) ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(legX0, yy); ctx.lineTo(legX0 + 22, yy);
          ctx.stroke();
          if (it.dash) ctx.setLineDash([]);
        }
        ctx.fillStyle = INK_FADE;
        ctx.fillText(it.label, legX0 + 28, yy + 3);
      });
    }

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
    ctx.fillText('polynomial degree', plotX0 + plotW / 2, botY + botH + 28);

    // Irreducible noise σ² baseline (horizontal dotted line)
    const sigma2 = state.sigma * state.sigma;
    if (sigma2 < maxErr) {
      const yS2 = yMap(sigma2);
      ctx.strokeStyle = 'rgba(38,35,32,0.30)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(plotX0, yS2); ctx.lineTo(plotX0 + plotW, yS2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(38,35,32,0.55)';
      ctx.font = 'italic 10px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'left';
      ctx.fillText(`σ² = ${formatNum(sigma2)}  (label noise — no model beats this floor)`, plotX0 + 6, yS2 - 4);
    }

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

    // Mark current degree (bottom panel) — extend up into top panel as a faint vertical guide.
    const cx = xMap(state.deg);
    const currentBV = bv.find((d) => d.degree === state.deg);
    const yVar = currentBV ? yMap(currentBV.variance) : (botY + botH);
    ctx.strokeStyle = 'rgba(122,31,36,0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 4]);
    ctx.beginPath();
    ctx.moveTo(cx, botY); ctx.lineTo(cx, botY + botH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Spotlight dot on the variance curve at current degree
    if (currentBV) {
      ctx.fillStyle = ACCENT;
      ctx.beginPath();
      ctx.arc(cx, yVar, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#fffdf6';
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }

    // ─── Cross-panel connector lines: link the spread band (top, around x=0) to the variance dot (bottom).
    // Find the band's vertical extent at x ≈ 0 (mid-canvas) so the connectors physically tie the
    // "spread of fits" to "the variance value at this degree."
    const linkMidX = tx(0);
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < bandXs.length; i++) {
      if (Math.abs(bandXs[i]) < 0.05) {
        if (bandLo[i] < lo) lo = bandLo[i];
        if (bandHi[i] > hi) hi = bandHi[i];
      }
    }
    if (isFinite(lo) && isFinite(hi)) {
      ctx.strokeStyle = 'rgba(122,31,36,0.35)';
      ctx.setLineDash([3, 4]);
      ctx.lineWidth = 1;
      // Top of spread → variance dot
      ctx.beginPath();
      ctx.moveTo(linkMidX, ty(hi));
      ctx.bezierCurveTo(linkMidX, (ty(hi) + yVar) / 2, cx, (ty(hi) + yVar) / 2, cx, yVar);
      ctx.stroke();
      // Bottom of spread → variance dot
      ctx.beginPath();
      ctx.moveTo(linkMidX, ty(lo));
      ctx.bezierCurveTo(linkMidX, (ty(lo) + yVar) / 2, cx, (ty(lo) + yVar) / 2, cx, yVar);
      ctx.stroke();
      ctx.setLineDash([]);

      // Tiny annotation
      ctx.fillStyle = 'rgba(122,31,36,0.7)';
      ctx.font = 'italic 10px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'left';
      ctx.fillText('spread → variance', cx + 6, yVar - 6);
    }

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
    // The formula, anchored to the picture: bias² + variance + σ² right above the curves.
    ctx.fillStyle = '#262320';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.fillText('Error(x) = bias²(x) + variance(x) + σ²', plotX0 + plotW, botY - 6);
  }

  function render() {
    degSlider.value = state.deg;
    degLabel.textContent = `degree ${state.deg}`;
    noiseSlider.value = state.sigma;
    noiseLabel.textContent = `σ = ${state.sigma.toFixed(2)}`;
    nSlider.value = state.n;
    nLabel.textContent = `n = ${state.n}`;
    if (showNoiseEl) showNoiseEl.checked = state.showNoise;

    // Approximate bias-variance at current degree (cheap recomputation)
    const sample = bvDecomposition().find((d) => d.degree === state.deg);
    readoutEl.innerHTML = `
      <div class="row"><span>bias²</span><b>${formatNum(sample.bias2)}</b></div>
      <div class="row"><span>variance</span><b style="color:var(--accent)">${formatNum(sample.variance)}</b></div>
      <div class="row"><span>noise σ²</span><b>${formatNum(state.sigma * state.sigma)}</b></div>
      <div class="row"><span>total err</span><b>${formatNum(sample.total)}</b></div>`;
    canvasCtl.redraw();
  }

  render();
})();
