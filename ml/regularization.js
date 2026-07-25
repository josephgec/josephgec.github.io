/* Vol. XIX — Holding it Together (Dropout / BatchNorm) */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK, INK_FADE } = V;

  const VIEWS = [
    { key: 'dropout',   label: 'Dropout · ensemble of sub-nets' },
    { key: 'testtime',  label: 'Train vs test time' },
    { key: 'batchnorm', label: 'BatchNorm · standardize the input' },
  ];
  const CAPTIONS = {
    dropout: 'Three forward passes with three different dropout masks — each is a different sub-network. Diagonal stripes mark silenced neurons. Bottom: the average activation across the three passes ≈ the full network. That\'s the ensemble effect.',
    testtime: 'Left: training time, dropout active. Random neurons are silenced; surviving outputs scaled up by 1 / (1 − p) so the mean stays the same. Right: test time, every neuron fires, no scaling, no randomness — what you actually deploy.',
    batchnorm: 'Top: the input distribution (orange ticks) drifts left/right with the per-layer μ. BatchNorm subtracts μ (centers at 0) then divides by σ (variance to 1), giving a clean standardized distribution. Then γ and β can optionally re-shape it.',
  };

  const state = {
    // Default p = 0.5 so a striking number of cells are silenced on first paint —
    // the dropout pattern is immediately legible.
    p: 0.5,
    bnOn: true,
    seed: 1,
    view: 'dropout',
    bnPhase: 0,    // 0..3 stages of the BN animation
    bnTimer: null,
    dropAnim: 0,   // 0..1 fade-in for stripes after Resample
    dropAnimTimer: null,
  };

  const pSlider = document.getElementById('p-slider');
  const pLabel = document.getElementById('p-label');
  const bnToggleEl = document.getElementById('bn-toggle');
  const resampleBtn = document.getElementById('resample');
  const readoutEl = document.getElementById('readout');
  const tabsEl = document.getElementById('view-tabs');
  const captionEl = document.getElementById('caption-text');
  const archCanvas = document.getElementById('arch-canvas');
  const canvas = document.getElementById('canvas');

  ['BN on', 'BN off'].forEach((lbl, i) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = lbl;
    b.dataset.bn = i === 0 ? 'on' : 'off';
    b.addEventListener('click', () => { state.bnOn = i === 0; render(); });
    bnToggleEl.appendChild(b);
  });
  VIEWS.forEach((v) => {
    const b = document.createElement('button');
    b.textContent = v.label;
    b.dataset.view = v.key;
    b.addEventListener('click', () => {
      state.view = v.key;
      if (v.key === 'batchnorm') stepBN();
      render();
    });
    tabsEl.appendChild(b);
  });

  pSlider.addEventListener('input', (e) => {
    state.p = +e.target.value;
    pLabel.textContent = `p = ${state.p.toFixed(2)}`;
    render();
  });
  resampleBtn.addEventListener('click', () => {
    state.seed = Math.floor(Math.random() * 1000);
    animateDropout();
    render();
  });

  function animateDropout() {
    state.dropAnim = 0;
    if (state.dropAnimTimer) clearInterval(state.dropAnimTimer);
    const t0 = performance.now();
    state.dropAnimTimer = setInterval(() => {
      const t = (performance.now() - t0) / 200;
      state.dropAnim = Math.min(1, t);
      if (state.dropAnim >= 1) { clearInterval(state.dropAnimTimer); state.dropAnimTimer = null; }
      render();
    }, 25);
  }
  function stepBN() {
    state.bnPhase = 0;
    if (state.bnTimer) clearInterval(state.bnTimer);
    state.bnTimer = setInterval(() => {
      state.bnPhase = (state.bnPhase + 1) % 4;
      render();
    }, 1500);
  }

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

  function simulate(seedOffset = 0) {
    const rnd = seeded(state.seed + seedOffset);
    const layers = 4;
    const units = 12;
    const acts = [];
    const a0 = Array.from({ length: units }, () => gaussianFromUniform(rnd));
    acts.push(a0);
    for (let li = 1; li < layers; li++) {
      const prev = acts[li - 1];
      const scale = 1.2;
      const shift = 0.4;
      const ax = prev.map((v) => Math.max(0, v * scale + shift + gaussianFromUniform(rnd) * 0.3));
      let result = ax;
      if (state.bnOn) {
        const m = ax.reduce((a, b) => a + b, 0) / ax.length;
        const v = ax.reduce((a, b) => a + (b - m) ** 2, 0) / ax.length;
        const sd = Math.sqrt(v + 1e-6);
        result = ax.map((x) => (x - m) / sd);
      }
      acts.push(result);
    }
    const masks = acts.map((layer) => layer.map(() => (rnd() > state.p ? 1 : 0)));
    const masked = acts.map((layer, li) => layer.map((v, i) => v * masks[li][i] / Math.max(0.001, 1 - state.p)));
    return { acts, masked, masks };
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    if (state.view === 'testtime') drawTestTime(ctx, size);
    else if (state.view === 'batchnorm') drawBatchNorm(ctx, size);
    else drawDropout(ctx, size);
  }

  function drawCellFill(ctx, x, y, w, h, v, masked, animT) {
    const intensity = Math.min(1, Math.abs(v) / 2);
    if (masked) {
      // base muted background
      ctx.fillStyle = 'rgba(38,35,32,0.05)';
      ctx.fillRect(x, y, w, h);
      // diagonal stripes pattern, alpha grows with animT
      const stripeAlpha = 0.40 * Math.min(1, animT);
      ctx.save();
      ctx.beginPath();
      ctx.rect(x, y, w, h);
      ctx.clip();
      ctx.strokeStyle = `rgba(38,35,32,${stripeAlpha})`;
      ctx.lineWidth = 1;
      const sp = 4;
      for (let s = -h; s < w + h; s += sp) {
        ctx.beginPath();
        ctx.moveTo(x + s, y);
        ctx.lineTo(x + s + h, y + h);
        ctx.stroke();
      }
      ctx.restore();
      // border (faint)
      ctx.strokeStyle = 'rgba(38,35,32,0.18)';
      ctx.strokeRect(x, y, w, h);
    } else {
      ctx.fillStyle = v >= 0
        ? `rgba(122,31,36,${0.12 + intensity * 0.7})`
        : `rgba(38,35,32,${0.12 + intensity * 0.7})`;
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = 'rgba(38,35,32,0.18)';
      ctx.strokeRect(x, y, w, h);
    }
  }

  // ───────── View 1: Dropout — three masks side by side ─────────
  function drawDropout(ctx, size) {
    const passes = [
      simulate(0),
      simulate(7),
      simulate(13),
    ];
    const layers = 4;
    const units = 12;

    const margin = 24;
    const gap = 16;
    const colW = (size.w - margin * 2 - gap * (passes.length)) / (passes.length + 1);  // +1 for the average column
    const cellH = (size.h * 0.75 - 50) / units;
    const startY = 36;

    // Title row
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('PASS 1 (mask A)', margin + colW / 2, 20);
    ctx.fillText('PASS 2 (mask B)', margin + colW + gap + colW / 2, 20);
    ctx.fillText('PASS 3 (mask C)', margin + 2 * (colW + gap) + colW / 2, 20);
    ctx.fillStyle = ACCENT;
    ctx.fillText('AVERAGE ≈ FULL NET', margin + 3 * (colW + gap) + colW / 2, 20);

    // Each pass column: mini grid showing all 4 layers stacked horizontally
    passes.forEach((sim, ci) => {
      const x0 = margin + ci * (colW + gap);
      const layerW = (colW - 4) / layers;
      for (let li = 0; li < layers; li++) {
        // Layer label
        ctx.fillStyle = INK_FADE;
        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText(li === 0 ? 'in' : `L${li}`, x0 + li * layerW + layerW / 2, startY - 4);
        for (let i = 0; i < units; i++) {
          const v = sim.masked[li][i];
          const masked = sim.masks[li][i] === 0;
          drawCellFill(ctx, x0 + li * layerW + 2, startY + i * cellH, layerW - 4, cellH - 1, v, masked, state.dropAnim || 1);
        }
      }
    });

    // Average column
    const x0 = margin + 3 * (colW + gap);
    const layerW = (colW - 4) / layers;
    for (let li = 0; li < layers; li++) {
      ctx.fillStyle = INK_FADE;
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(li === 0 ? 'in' : `L${li}`, x0 + li * layerW + layerW / 2, startY - 4);
      for (let i = 0; i < units; i++) {
        const avg = (passes[0].masked[li][i] + passes[1].masked[li][i] + passes[2].masked[li][i]) / 3;
        drawCellFill(ctx, x0 + li * layerW + 2, startY + i * cellH, layerW - 4, cellH - 1, avg, false, 1);
      }
    }
    // Surround with a soft accent border
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x0 - 2, startY - 2, layerW * layers + 4, cellH * units + 4);

    // Bottom: count of active units per pass + invariance equation
    const summaryY = size.h - 28;
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    passes.forEach((sim, ci) => {
      const live = sim.masks.flat().filter((m) => m === 1).length;
      const total = sim.masks.flat().length;
      ctx.fillText(`active: ${live}/${total}`, margin + ci * (colW + gap) + colW / 2, summaryY);
    });
    ctx.fillStyle = ACCENT;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.fillText('= ensemble of sub-networks', margin + 3 * (colW + gap) + colW / 2, summaryY);

    // Click hint
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText(`p = ${state.p.toFixed(2)}  ·  hit Resample for fresh masks`, size.w / 2, size.h - 10);

    // (The striped = silenced / filled = active key lives in the caption below the
    // canvas; an inline top-right legend used to collide with the AVERAGE column title.)
  }

  // ───────── View 2: Train vs test time ─────────
  function drawTestTime(ctx, size) {
    const sim = simulate(0);
    const layers = 4;
    const units = 12;

    const margin = 30;
    const gap = 30;
    const colW = (size.w - margin * 2 - gap) / 2;
    const cellH = (size.h * 0.75 - 80) / units;
    const startY = 60;
    const layerW = (colW - 8) / layers;

    // Headers
    ctx.fillStyle = INK;
    ctx.font = 'italic 14px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('Training time', margin + colW / 2, 28);
    ctx.fillText('Test time', margin + colW + gap + colW / 2, 28);
    ctx.fillStyle = INK_FADE;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillText('mask m, scale by 1/(1−p)', margin + colW / 2, 46);
    ctx.fillText('every neuron fires, no scaling', margin + colW + gap + colW / 2, 46);

    // Left: training time — silent neurons
    let leftLive = 0, leftTotal = 0;
    for (let li = 0; li < layers; li++) {
      ctx.fillStyle = INK_FADE;
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(li === 0 ? 'in' : `L${li}`, margin + li * layerW + layerW / 2, startY - 6);
      for (let i = 0; i < units; i++) {
        const v = sim.masked[li][i];
        const masked = sim.masks[li][i] === 0;
        drawCellFill(ctx, margin + li * layerW + 4, startY + i * cellH, layerW - 8, cellH - 2, v, masked, 1);
        leftTotal++;
        if (!masked) leftLive++;
      }
    }
    // Right: test time — all live
    const xR = margin + colW + gap;
    for (let li = 0; li < layers; li++) {
      ctx.fillStyle = INK_FADE;
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(li === 0 ? 'in' : `L${li}`, xR + li * layerW + layerW / 2, startY - 6);
      for (let i = 0; i < units; i++) {
        const v = sim.acts[li][i] * (1 - state.p);  // un-scale to show "no scaling needed at test time"
        drawCellFill(ctx, xR + li * layerW + 4, startY + i * cellH, layerW - 8, cellH - 2, v, false, 1);
      }
    }

    // Footer formula
    const fy = size.h - 28;
    ctx.fillStyle = INK;
    ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('a\' = (m ⊙ a) / (1 − p)', margin + colW / 2, fy);
    ctx.fillText('a\' = a', xR + colW / 2, fy);
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.fillText(`active: ${leftLive}/${leftTotal} (≈ ${((1-state.p)*100).toFixed(0)}%)`, margin + colW / 2, fy + 14);
    ctx.fillText(`active: ${units * 4}/${units * 4} (100%)`, xR + colW / 2, fy + 14);
  }

  // ───────── View 3: BatchNorm animation ─────────
  function drawBatchNorm(ctx, size) {
    // Generate a synthetic input distribution (drifted away from 0)
    const rnd = seeded(state.seed + 99);
    const N = 40;
    const inputs = Array.from({ length: N }, () => 1.4 + gaussianFromUniform(rnd) * 1.1);
    // Compute μ, σ
    const mu = inputs.reduce((a, b) => a + b, 0) / N;
    const sigma = Math.sqrt(inputs.reduce((a, b) => a + (b - mu) ** 2, 0) / N + 1e-6);
    const gamma = 1.6;
    const beta = -0.4;
    // Decide stage based on bnPhase
    // 0 = raw inputs · 1 = − μ (centered) · 2 = ÷ σ (standardized) · 3 = γ x̂ + β

    const stages = [
      { lbl: 'input',          fn: (x) => x,                                       gloss: 'raw activations from previous layer · drifted: μ ≠ 0, σ ≠ 1' },
      { lbl: 'subtract μ',      fn: (x) => x - mu,                                  gloss: `subtract the batch mean μ = ${mu.toFixed(2)} · now centered at 0` },
      { lbl: 'divide by σ',     fn: (x) => (x - mu) / sigma,                        gloss: `divide by σ = ${sigma.toFixed(2)} · now σ̂ = 1 (standardized)` },
      { lbl: 'γ · x̂ + β',       fn: (x) => gamma * ((x - mu) / sigma) + beta,        gloss: `learned rescale: γ = ${gamma.toFixed(1)}, β = ${beta.toFixed(1)} · network can re-shift if it helps` },
    ];

    const margin = 30;
    const yAxis = size.h * 0.55;
    const trackW = size.w - margin * 2;
    const xMin = -4, xMax = 6;
    const tx = (x) => margin + (x - xMin) / (xMax - xMin) * trackW;

    // Title and description
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('BATCH NORMALIZATION · standardize, then learn an affine map back', margin, 22);

    // Highlighted stage label
    const stage = stages[state.bnPhase];
    ctx.fillStyle = ACCENT;
    ctx.font = 'italic 16px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText(`Step ${state.bnPhase + 1} / 4 · ${stage.lbl}`, margin, 50);
    ctx.fillStyle = INK;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.fillText(stage.gloss, margin, 70);

    // Track baseline
    ctx.strokeStyle = 'rgba(38,35,32,0.20)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(margin, yAxis);
    ctx.lineTo(size.w - margin, yAxis);
    ctx.stroke();
    // Tick at 0
    ctx.strokeStyle = 'rgba(38,35,32,0.40)';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(tx(0), yAxis - 50); ctx.lineTo(tx(0), yAxis + 50); ctx.stroke();
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('0', tx(0), yAxis + 65);

    // Plot transformed values as ticks
    const transformed = inputs.map(stage.fn);
    transformed.forEach((v) => {
      const x = tx(Math.max(xMin, Math.min(xMax, v)));
      ctx.strokeStyle = 'rgba(122,31,36,0.65)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, yAxis - 12);
      ctx.lineTo(x, yAxis + 12);
      ctx.stroke();
    });

    // μ marker (vertical line)
    const meanT = transformed.reduce((a, b) => a + b, 0) / N;
    const sdT = Math.sqrt(transformed.reduce((a, b) => a + (b - meanT) ** 2, 0) / N + 1e-6);

    // μ line
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(tx(meanT), yAxis - 30); ctx.lineTo(tx(meanT), yAxis + 30); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = ACCENT;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`μ = ${meanT.toFixed(2)}`, tx(meanT), yAxis - 40);
    // σ band
    ctx.strokeStyle = 'rgba(122,31,36,0.40)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(tx(meanT - sdT), yAxis + 30);
    ctx.lineTo(tx(meanT + sdT), yAxis + 30);
    ctx.stroke();
    ctx.beginPath(); ctx.moveTo(tx(meanT - sdT), yAxis + 24); ctx.lineTo(tx(meanT - sdT), yAxis + 36); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(tx(meanT + sdT), yAxis + 24); ctx.lineTo(tx(meanT + sdT), yAxis + 36); ctx.stroke();
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillText(`σ = ${sdT.toFixed(2)}`, tx(meanT), yAxis + 52);

    // Phase progression boxes
    const pbY = size.h - 36;
    const pbW = (size.w - margin * 2) / stages.length;
    stages.forEach((s, i) => {
      const x0 = margin + i * pbW;
      ctx.fillStyle = i === state.bnPhase ? 'rgba(122,31,36,0.20)' : 'rgba(38,35,32,0.05)';
      ctx.strokeStyle = i === state.bnPhase ? ACCENT : 'rgba(38,35,32,0.20)';
      ctx.lineWidth = i === state.bnPhase ? 2 : 1;
      ctx.fillRect(x0 + 4, pbY, pbW - 8, 24);
      ctx.strokeRect(x0 + 4, pbY, pbW - 8, 24);
      ctx.fillStyle = i === state.bnPhase ? ACCENT : INK;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(s.lbl, x0 + pbW / 2, pbY + 12);
      ctx.textBaseline = 'alphabetic';
    });
  }

  // ───────── Architecture canvas (BN + Dropout block diagrams) ─────────
  function drawArch() {
    const cv = archCanvas;
    const dpr = window.devicePixelRatio || 1;
    const W = cv.parentElement.getBoundingClientRect().width, H = 200;
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = W + 'px'; cv.style.height = H + 'px';
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const margin = 20;
    const halfW = (W - margin * 3) / 2;

    // Common helpers
    function box(ctx, x, y, w, h, label, accent = false, sub = '') {
      ctx.fillStyle = accent ? 'rgba(122,31,36,0.10)' : '#fffdf6';
      ctx.strokeStyle = INK;
      ctx.lineWidth = 1.2;
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = INK;
      ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, x + w / 2, y + h / 2);
      ctx.textBaseline = 'alphabetic';
      if (sub) {
        ctx.fillStyle = INK_FADE;
        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.fillText(sub, x + w / 2, y + h + 12);
      }
    }
    function arrow(ctx, x1, y1, x2) {
      ctx.strokeStyle = INK_FADE;
      ctx.fillStyle = INK_FADE;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y1);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x2, y1);
      ctx.lineTo(x2 - 5, y1 - 3);
      ctx.lineTo(x2 - 5, y1 + 3);
      ctx.closePath(); ctx.fill();
    }

    const baseY = H * 0.55;

    // ── Left: BatchNorm block ──
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('BATCHNORM BLOCK', margin, 18);
    const bnSteps = [
      { lbl: 'input x', sub: '' },
      { lbl: '− μ',      sub: 'center' },
      { lbl: '÷ σ',      sub: 'standardize' },
      { lbl: '× γ',      sub: 'rescale' },
      { lbl: '+ β',      sub: 'shift' },
      { lbl: 'output y', sub: '' },
    ];
    const bnStepW = halfW / bnSteps.length;
    const bnBoxW = bnStepW - 12;
    bnSteps.forEach((s, i) => {
      const x = margin + i * bnStepW + 6;
      const isAccent = i > 0 && i < bnSteps.length - 1;
      box(ctx, x, baseY - 18, bnBoxW, 32, s.lbl, isAccent, s.sub);
      if (i < bnSteps.length - 1) {
        arrow(ctx, x + bnBoxW, baseY - 2, x + bnStepW + 6);
      }
    });
    // Surround the (μ, σ, γ, β) span with a faint bracket
    const bx0 = margin + bnStepW + 6;
    const bx1 = margin + 5 * bnStepW + 6 - 6;
    ctx.strokeStyle = 'rgba(122,31,36,0.40)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.strokeRect(bx0 - 4, baseY - 28, bx1 - bx0 + 8, 64);
    ctx.setLineDash([]);
    ctx.fillStyle = ACCENT;
    ctx.font = 'italic 10px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('the BN transform', (bx0 + bx1) / 2, baseY - 32);

    // ── Right: Dropout block ──
    const rx = margin * 2 + halfW;
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('DROPOUT BLOCK', rx, 18);
    const drSteps = [
      { lbl: 'input a', sub: '' },
      { lbl: 'mask m',  sub: 'sample 0/1 per neuron' },
      { lbl: 'm ⊙ a',   sub: 'silence dropped units' },
      { lbl: '÷ (1−p)', sub: 'rescale survivors' },
      { lbl: 'output a\'', sub: '' },
    ];
    const drStepW = halfW / drSteps.length;
    const drBoxW = drStepW - 12;
    drSteps.forEach((s, i) => {
      const x = rx + i * drStepW + 6;
      const isAccent = i > 0 && i < drSteps.length - 1;
      box(ctx, x, baseY - 18, drBoxW, 32, s.lbl, isAccent, s.sub);
      if (i < drSteps.length - 1) {
        arrow(ctx, x + drBoxW, baseY - 2, x + drStepW + 6);
      }
    });
    const dx0 = rx + drStepW + 6;
    const dx1 = rx + 4 * drStepW + 6 - 6;
    ctx.strokeStyle = 'rgba(122,31,36,0.40)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.strokeRect(dx0 - 4, baseY - 28, dx1 - dx0 + 8, 64);
    ctx.setLineDash([]);
    ctx.fillStyle = ACCENT;
    ctx.font = 'italic 10px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('train-time only · removed at test', (dx0 + dx1) / 2, baseY - 32);
  }
  new ResizeObserver(drawArch).observe(archCanvas.parentElement);

  function render() {
    pLabel.textContent = `p = ${state.p.toFixed(2)}`;
    Array.from(bnToggleEl.children).forEach((btn) => {
      btn.classList.toggle('active', (btn.dataset.bn === 'on') === state.bnOn);
    });
    Array.from(tabsEl.children).forEach((btn) => {
      btn.classList.toggle('on', btn.dataset.view === state.view);
    });
    if (captionEl) captionEl.textContent = CAPTIONS[state.view];

    const sim = simulate();
    const liveCount = sim.masks.flat().filter((m) => m === 1).length;
    const total = sim.masks.flat().length;
    readoutEl.innerHTML = `
      <div class="row"><span>active units</span><b>${liveCount} / ${total}</b></div>
      <div class="row"><span>BN</span><b>${state.bnOn ? 'on' : 'off'}</b></div>
      <div class="row"><span>scale 1/(1−p)</span><b>${(1 / Math.max(0.001, 1 - state.p)).toFixed(2)}×</b></div>`;
    canvasCtl.redraw();
    drawArch();
  }

  render();
})();
