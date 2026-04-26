/* Vol. IX — A Field Guide to Activations */
(function () {
  'use strict';
  const V = window.MLViz;
  const { INK_FADE } = V;

  // Colours chosen to read against the cream paper bg without competing with the oxblood accent.
  const FNS = {
    sigmoid: {
      name: 'Sigmoid',
      formula: 'σ(x) = 1 / (1 + e⁻ˣ)',
      dformula: "σ'(x) = σ(x) · (1 − σ(x))",
      f:  (x) => 1 / (1 + Math.exp(-x)),
      df: (x) => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); },
      range: '(0, 1)',
      dPeak: 0.25,
      color: '#7a1f24',
      blurb: "The classic. Squashes everything to (0, 1) — useful for probabilities. But the gradient flatlines at the tails: train deep networks with this and the signal dies on the way back. They called it the vanishing gradient problem.",
      use: 'Output layer for binary classification. Almost never in hidden layers anymore.',
    },
    tanh: {
      name: 'Tanh',
      formula: 'tanh(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)',
      dformula: "tanh'(x) = 1 − tanh²(x)",
      f:  (x) => Math.tanh(x),
      df: (x) => 1 - Math.tanh(x) ** 2,
      range: '(−1, 1)',
      dPeak: 1.0,
      color: '#3a6b5e',
      blurb: "Sigmoid's zero-centered cousin. Outputs in (−1, 1), so activations don't all push in the same direction. Still saturates at the tails — same vanishing gradient, just with better posture.",
      use: 'Recurrent networks, occasionally. Mostly displaced by ReLU.',
    },
    relu: {
      name: 'ReLU',
      formula: 'ReLU(x) = max(0, x)',
      dformula: "ReLU'(x) = 1 if x > 0 else 0",
      f:  (x) => Math.max(0, x),
      df: (x) => (x > 0 ? 1 : 0),
      range: '[0, ∞)',
      dPeak: 1.0,
      color: '#262320',
      blurb: "Brutally simple. If positive, pass through. If negative, zero. The kink at the origin is technically non-differentiable but nobody cares. Cheap to compute, doesn't saturate on the positive side, and somehow makes deep networks actually trainable.",
      use: 'Default for hidden layers in feedforward and convolutional networks since ~2012.',
    },
    leaky: {
      name: 'Leaky ReLU',
      formula: 'f(x) = max(αx, x),  α = 0.1',
      dformula: "f'(x) = 1 if x > 0 else 0.1",
      f:  (x) => (x > 0 ? x : 0.1 * x),
      df: (x) => (x > 0 ? 1 : 0.1),
      range: '(−∞, ∞)',
      dPeak: 1.0,
      color: '#8b5a2b',
      blurb: "Fix for ReLU's \"dying neurons\": when too many units get stuck at zero, gradients can't flow back through them. A small slope on the negative side keeps the gradient alive.",
      use: 'When standard ReLU is killing too many neurons. Variants: PReLU, ELU, GELU.',
    },
    gelu: {
      name: 'GELU',
      formula: 'GELU(x) ≈ x · σ(1.702x)',
      dformula: "GELU'(x) ≈ σ(1.702x) + 1.702 x σ(1.702x)(1−σ(1.702x))",
      f:  (x) => x * (1 / (1 + Math.exp(-1.702 * x))),
      df: (x) => { const s = 1 / (1 + Math.exp(-1.702 * x)); return s + x * s * (1 - s) * 1.702; },
      range: '(−ε, ∞)',
      dPeak: 1.05,
      color: '#5a3a8e',
      blurb: "Smooth, probabilistic ReLU — weights inputs by their cumulative normal probability. The activation in modern transformers (GPT, BERT). Slight dip below zero, then asymptotes to identity.",
      use: 'Transformers, large language models.',
    },
  };

  const VIEWS = [
    { key: 'split',   label: 'Split · ƒ above ƒ′' },
    { key: 'compare', label: 'Compare · small multiples' },
  ];

  const state = {
    focused: 'relu',
    hoverX: null,
    view: 'split',
  };

  const focusEl = document.getElementById('focus');
  const captionEl = document.getElementById('caption');
  const marginEl = document.getElementById('margin');
  const miniGridEl = document.getElementById('mini-grid');
  const canvas = document.getElementById('canvas');
  const tabsEl = document.getElementById('view-tabs');
  const deadReadoutEl = document.getElementById('dead-readout');

  // View tabs
  VIEWS.forEach((v) => {
    const b = document.createElement('button');
    b.textContent = v.label;
    b.dataset.view = v.key;
    b.addEventListener('click', () => {
      state.view = v.key;
      render();
    });
    tabsEl.appendChild(b);
  });

  // Focus presets — each acts as the "active activation" tab strip.
  Object.entries(FNS).forEach(([k, fn]) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.dataset.fn = k;
    b.textContent = fn.name;
    b.addEventListener('click', () => {
      state.focused = k;
      render();
    });
    focusEl.appendChild(b);
  });

  // Build mini-grid (always visible at the bottom — small multiples for comparison).
  Object.entries(FNS).forEach(([k, fn]) => {
    const wrap = document.createElement('div');
    wrap.className = 'mini';
    wrap.dataset.fn = k;
    wrap.title = `Click to focus ${fn.name} in the split view above`;
    wrap.innerHTML = `
      <canvas></canvas>
      <div class="name" style="color:${fn.color}">${fn.name}</div>
      <div class="range">${fn.range}</div>`;
    wrap.style.cursor = 'pointer';
    wrap.addEventListener('click', () => {
      state.focused = k;
      // Smooth-scroll back up to the focused split view.
      document.querySelector('.ml-canvas-wrap').scrollIntoView({ behavior: 'smooth', block: 'center' });
      render();
    });
    miniGridEl.appendChild(wrap);
  });

  // Mini-grid: each panel shows ƒ AND ƒ′ overlaid (comparison view).
  function drawMinis() {
    const W = 180, H = 120;
    Array.from(miniGridEl.children).forEach((el) => {
      const k = el.dataset.fn;
      const fn = FNS[k];
      const cv = el.querySelector('canvas');
      const dpr = window.devicePixelRatio || 1;
      cv.width = W * dpr; cv.height = H * dpr;
      cv.style.width = W + 'px'; cv.style.height = H + 'px';
      const ctx = cv.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const xMin = -5, xMax = 5;
      const yMin = -1.5, yMax = 2.5;
      const tx = (x) => (x - xMin) / (xMax - xMin) * W;
      const ty = (y) => H - (y - yMin) / (yMax - yMin) * H;
      // Highlight focused panel
      if (k === state.focused) {
        ctx.fillStyle = 'rgba(122,31,36,0.05)';
        ctx.fillRect(0, 0, W, H);
      }
      // Axes
      ctx.strokeStyle = 'rgba(38,35,32,0.18)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(tx(0), 0); ctx.lineTo(tx(0), H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, ty(0)); ctx.lineTo(W, ty(0)); ctx.stroke();
      // Derivative — light dashed
      ctx.strokeStyle = fn.color;
      ctx.globalAlpha = 0.45;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      let first = true;
      for (let px = 0; px <= W; px++) {
        const x = xMin + (px / W) * (xMax - xMin);
        const y = fn.df(x);
        if (first) { ctx.moveTo(px, ty(y)); first = false; }
        else ctx.lineTo(px, ty(y));
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      // Activation — solid
      ctx.strokeStyle = fn.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      first = true;
      for (let px = 0; px <= W; px++) {
        const x = xMin + (px / W) * (xMax - xMin);
        const y = fn.f(x);
        if (first) { ctx.moveTo(px, ty(y)); first = false; }
        else ctx.lineTo(px, ty(y));
      }
      ctx.stroke();
    });
  }

  // Main canvas (rectangular)
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });
  canvas.style.cursor = 'crosshair';

  function drawCurve(ctx, fn, useDeriv, xMin, xMax, x2px, y2py, w0, w1) {
    ctx.strokeStyle = fn.color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    let first = true;
    for (let px = w0; px <= w1; px += 1) {
      const x = xMin + ((px - w0) / (w1 - w0)) * (xMax - xMin);
      const y = useDeriv ? fn.df(x) : fn.f(x);
      if (first) { ctx.moveTo(px, y2py(y)); first = false; }
      else ctx.lineTo(px, y2py(y));
    }
    ctx.stroke();
  }

  function drawAxes(ctx, x2px, y2py, xMin, xMax, yMin, yMax, w0, w1, h0, h1) {
    // Faint gridlines
    ctx.strokeStyle = 'rgba(38,35,32,0.06)';
    ctx.lineWidth = 1;
    for (let i = Math.ceil(xMin); i <= Math.floor(xMax); i++) {
      ctx.beginPath(); ctx.moveTo(x2px(i), h0); ctx.lineTo(x2px(i), h1); ctx.stroke();
    }
    for (let y = Math.ceil(yMin); y <= Math.floor(yMax); y++) {
      ctx.beginPath(); ctx.moveTo(w0, y2py(y)); ctx.lineTo(w1, y2py(y)); ctx.stroke();
    }
    // Bold axes (x = 0, y = 0 if visible)
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x2px(0), h0); ctx.lineTo(x2px(0), h1); ctx.stroke();
    if (yMin <= 0 && yMax >= 0) {
      ctx.beginPath(); ctx.moveTo(w0, y2py(0)); ctx.lineTo(w1, y2py(0)); ctx.stroke();
    }
  }

  function draw(ctx, size) {
    if (state.view === 'compare') {
      drawCompare(ctx, size);
      return;
    }
    drawSplit(ctx, size);
  }

  // SPLIT VIEW — top half = ƒ(x); bottom half = ƒ′(x), with DEAD ZONE band.
  function drawSplit(ctx, size) {
    const fn = FNS[state.focused];
    const xMin = -6, xMax = 6;
    const padL = 50, padR = 60;
    const padT = 32, padG = 36, padB = 28;
    const halfH = (size.h - padT - padG - padB) / 2;
    const w0 = padL, w1 = size.w - padR;
    const top0 = padT, top1 = padT + halfH;
    const bot0 = padT + halfH + padG, bot1 = padT + halfH + padG + halfH;
    const x2px = (x) => w0 + (x - xMin) / (xMax - xMin) * (w1 - w0);

    // ── Top: ƒ(x) ──
    const yMinF = -1.5, yMaxF = 3;
    const fy = (y) => top1 - (y - yMinF) / (yMaxF - yMinF) * halfH;
    drawAxes(ctx, x2px, fy, xMin, xMax, yMinF, yMaxF, w0, w1, top0, top1);
    // Frame
    ctx.strokeStyle = 'rgba(38,35,32,0.20)';
    ctx.lineWidth = 1;
    ctx.strokeRect(w0, top0, w1 - w0, top1 - top0);
    // Curve
    drawCurve(ctx, fn, false, xMin, xMax, x2px, fy, w0, w1);
    // Pane label & formula
    ctx.fillStyle = INK_FADE;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    ctx.fillText('ƒ(x)', w0 + 6, top0 + 14);
    ctx.fillStyle = fn.color;
    ctx.font = 'italic 14px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'right';
    ctx.fillText(fn.formula, w1 - 6, top0 + 16);
    // Y-tick labels
    ctx.fillStyle = 'rgba(38,35,32,0.5)';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    [-1, 0, 1, 2].forEach((v) => {
      if (v >= yMinF && v <= yMaxF) ctx.fillText(v.toString(), w0 - 4, fy(v) + 3);
    });

    // ── Bottom: ƒ′(x) ──
    const yMinD = -0.2, yMaxD = 1.3;
    const dy = (y) => bot1 - (y - yMinD) / (yMaxD - yMinD) * halfH;
    // Dead-zone band where |ƒ′| < 0.1
    const dt = 0.1;
    const dy0 = dy(dt), dy1 = dy(-dt);
    ctx.fillStyle = 'rgba(38,35,32,0.07)';
    ctx.fillRect(w0, dy0, w1 - w0, dy1 - dy0);
    drawAxes(ctx, x2px, dy, xMin, xMax, yMinD, yMaxD, w0, w1, bot0, bot1);
    // Frame
    ctx.strokeStyle = 'rgba(38,35,32,0.20)';
    ctx.lineWidth = 1;
    ctx.strokeRect(w0, bot0, w1 - w0, bot1 - bot0);
    // Dead-zone label
    ctx.fillStyle = 'rgba(38,35,32,0.55)';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText('DEAD ZONE  |ƒ′| < 0.1', w0 + 8, (dy0 + dy1) / 2);
    // Curve
    drawCurve(ctx, fn, true, xMin, xMax, x2px, dy, w0, w1);
    // Pane label & formula
    ctx.fillStyle = INK_FADE;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    ctx.fillText("ƒ'(x)", w0 + 6, bot0 + 14);
    ctx.fillStyle = fn.color;
    ctx.font = 'italic 14px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'right';
    ctx.fillText(fn.dformula, w1 - 6, bot0 + 16);
    // Y-tick labels
    ctx.fillStyle = 'rgba(38,35,32,0.5)';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    [0, 0.5, 1].forEach((v) => {
      if (v >= yMinD && v <= yMaxD) ctx.fillText(v.toString(), w0 - 4, dy(v) + 3);
    });

    // X-tick labels (shared, under bottom panel)
    ctx.fillStyle = 'rgba(38,35,32,0.5)';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let i = xMin + 2; i <= xMax - 2; i += 2) {
      if (i !== 0) ctx.fillText(i.toString(), x2px(i), bot1 + 4);
    }
    ctx.fillText('x', w1 + 4, bot1 + 4);

    // Affordance hint when not hovering — invites the user to interact.
    if (state.hoverX === null) {
      ctx.fillStyle = 'rgba(38,35,32,0.45)';
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'right'; ctx.textBaseline = 'alphabetic';
      ctx.fillText('hover to read ƒ(x), ƒ′(x) at any x →', w1 - 6, bot1 - 6);
    }

    // Story annotation on the derivative pane: each activation gets one short remark
    // pinned to its most-revealing region.
    {
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
      let note = null, ax = null, ay = null;
      if (state.focused === 'sigmoid' || state.focused === 'tanh') {
        note = 'tails saturate → gradient ≈ 0';
        ax = x2px(3.5); ay = dy(0.05);
      } else if (state.focused === 'relu') {
        note = 'left half is dead — slope is exactly 0';
        ax = x2px(-5.5); ay = dy(0.18);
      } else if (state.focused === 'leaky') {
        note = 'tiny α slope keeps gradients alive';
        ax = x2px(-5.5); ay = dy(0.18);
      } else if (state.focused === 'gelu') {
        note = 'smooth curve — no kink for backprop to trip on';
        ax = x2px(-1.0); ay = dy(0.7);
      }
      if (note) {
        ctx.fillStyle = fn.color;
        ctx.fillText(note, ax, ay);
      }
    }

    // Hover crosshair
    if (state.hoverX !== null && state.hoverX >= xMin && state.hoverX <= xMax) {
      const px = x2px(state.hoverX);
      ctx.strokeStyle = 'rgba(38,35,32,0.4)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(px, top0); ctx.lineTo(px, bot1);
      ctx.stroke();
      ctx.setLineDash([]);
      // Dots + values on each pane
      const fv = fn.f(state.hoverX), dv = fn.df(state.hoverX);
      ctx.fillStyle = fn.color;
      ctx.beginPath(); ctx.arc(px, fy(fv), 4, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(px, dy(dv), 4, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#262320';
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
      ctx.fillText(`x=${state.hoverX.toFixed(2)}`, px + 6, top0 + 14);
      ctx.fillText(`ƒ=${fv.toFixed(3)}`, px + 6, fy(fv) - 6);
      ctx.fillText(`ƒ'=${dv.toFixed(3)}`, px + 6, dy(dv) - 6);
    }
  }

  // COMPARE VIEW — five small panels, ƒ + ƒ′ overlaid.
  function drawCompare(ctx, size) {
    const xMin = -5, xMax = 5;
    const yMin = -1.5, yMax = 2.5;
    const keys = Object.keys(FNS);
    const cols = 5;
    const padL = 24, padR = 24, padT = 36, padB = 28;
    const gap = 14;
    const cellW = (size.w - padL - padR - gap * (cols - 1)) / cols;
    const cellH = size.h - padT - padB;
    keys.forEach((k, i) => {
      const fn = FNS[k];
      const cx = padL + i * (cellW + gap);
      const cy = padT;
      const x2px = (x) => cx + (x - xMin) / (xMax - xMin) * cellW;
      const y2py = (y) => cy + cellH - (y - yMin) / (yMax - yMin) * cellH;
      // Frame
      ctx.fillStyle = (k === state.focused) ? 'rgba(122,31,36,0.04)' : 'rgba(38,35,32,0.015)';
      ctx.fillRect(cx, cy, cellW, cellH);
      ctx.strokeStyle = 'rgba(38,35,32,0.18)';
      ctx.lineWidth = 1;
      ctx.strokeRect(cx, cy, cellW, cellH);
      // Axes
      ctx.strokeStyle = 'rgba(38,35,32,0.25)';
      ctx.beginPath(); ctx.moveTo(x2px(0), cy); ctx.lineTo(x2px(0), cy + cellH); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx, y2py(0)); ctx.lineTo(cx + cellW, y2py(0)); ctx.stroke();
      // ƒ′ — light dashed
      ctx.strokeStyle = fn.color;
      ctx.globalAlpha = 0.45;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      let first = true;
      for (let px = cx; px <= cx + cellW; px++) {
        const x = xMin + ((px - cx) / cellW) * (xMax - xMin);
        const y = fn.df(x);
        if (first) { ctx.moveTo(px, y2py(y)); first = false; }
        else ctx.lineTo(px, y2py(y));
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      // ƒ — solid
      ctx.strokeStyle = fn.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      first = true;
      for (let px = cx; px <= cx + cellW; px++) {
        const x = xMin + ((px - cx) / cellW) * (xMax - xMin);
        const y = fn.f(x);
        if (first) { ctx.moveTo(px, y2py(y)); first = false; }
        else ctx.lineTo(px, y2py(y));
      }
      ctx.stroke();
      // Title
      ctx.fillStyle = fn.color;
      ctx.font = 'italic 14px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
      ctx.fillText(fn.name, cx + cellW / 2, cy - 8);
      // Range
      ctx.fillStyle = INK_FADE;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillText(fn.range, cx + cellW / 2, cy + cellH + 14);
    });
    // Legend
    ctx.fillStyle = INK_FADE;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('— ƒ(x)    - - ƒ′(x)', padL, size.h - 6);
    ctx.textAlign = 'right';
    ctx.fillText('click a panel to focus', size.w - padR, size.h - 6);
  }

  // Hover only meaningful in split view.
  canvas.addEventListener('mousemove', (e) => {
    const r = canvas.getBoundingClientRect();
    const xMin = -6, xMax = 6;
    const px = (e.clientX - r.left) / r.width * canvas.clientWidth;
    // Approximate: x = lerp from left padding boundary through right padding boundary.
    const padLfrac = 50 / canvas.clientWidth;
    const padRfrac = 60 / canvas.clientWidth;
    const t = (px / canvas.clientWidth - padLfrac) / (1 - padLfrac - padRfrac);
    state.hoverX = xMin + t * (xMax - xMin);
    canvasCtl.redraw();
  });
  canvas.addEventListener('mouseleave', () => {
    state.hoverX = null;
    canvasCtl.redraw();
  });

  // Render
  function render() {
    Array.from(focusEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.fn === state.focused);
    });
    Array.from(tabsEl.children).forEach((btn) => {
      btn.classList.toggle('on', btn.dataset.view === state.view);
    });

    const fn = FNS[state.focused];
    captionEl.innerHTML = state.view === 'split'
      ? `<span class="ml-cap-num">·</span> Above: <em>${fn.name}</em>'s shape — <em>ƒ(x)</em>. Below: its derivative <em>ƒ′(x)</em>, the slope of the curve above. Backprop multiplies this slope through every layer, so wherever it lies in the shaded "dead zone" (|ƒ′| &lt; 0.1) the gradient gets nearly killed and learning stalls.`
      : `<span class="ml-cap-num">·</span> All five activations side-by-side. <span style="font-family:var(--mono); font-size:11px">— solid = ƒ(x), - - dashed = ƒ′(x)</span>. Click any panel to focus it in the split view.`;

    // Side-rail readout
    const peakDeriv = fn.dPeak;
    deadReadoutEl.innerHTML = `
      <div class="row"><span>ƒ′ peak</span><b>${peakDeriv.toFixed(2)}</b></div>
      <div class="row"><span>ƒ′(0)</span><b>${fn.df(0).toFixed(3)}</b></div>
      <div class="row"><span>ƒ′(±5)</span><b>${fn.df(5).toFixed(3)} / ${fn.df(-5).toFixed(3)}</b></div>`;

    marginEl.innerHTML = `
      <div class="ml-margin-rule"></div>
      <div class="ml-margin-h" style="color:${fn.color}">${fn.name}</div>
      <div class="ml-equation" style="font-size:14px">${fn.formula}</div>
      <div style="font-family:var(--mono); font-size:12px; color:var(--ml-ink-fade); margin-bottom:14px">range: ${fn.range}</div>
      <p>${fn.blurb}</p>
      <div class="ml-margin-rule thin"></div>
      <div class="ml-margin-h sm">In practice</div>
      <p class="quiet">${fn.use}</p>`;

    canvasCtl.redraw();
    drawMinis();
  }

  render();
})();
