/* Vol. IX — A Field Guide to Activations */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, INK_FADE } = V;

  // Colours chosen to read against the cream paper bg without competing with the oxblood accent.
  const FNS = {
    sigmoid: {
      name: 'Sigmoid',
      formula: 'σ(x) = 1 / (1 + e⁻ˣ)',
      f:  (x) => 1 / (1 + Math.exp(-x)),
      df: (x) => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); },
      range: '(0, 1)',
      color: '#7a1f24',
      blurb: "The classic. Squashes everything to (0, 1) — useful for probabilities. But the gradient flatlines at the tails: train deep networks with this and the signal dies on the way back. They called it the vanishing gradient problem.",
      use: 'Output layer for binary classification. Almost never in hidden layers anymore.',
    },
    tanh: {
      name: 'Tanh',
      formula: 'tanh(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)',
      f:  (x) => Math.tanh(x),
      df: (x) => 1 - Math.tanh(x) ** 2,
      range: '(−1, 1)',
      color: '#3a6b5e',
      blurb: "Sigmoid's zero-centered cousin. Outputs in (−1, 1), so activations don't all push in the same direction. Still saturates at the tails — same vanishing gradient, just with better posture.",
      use: 'Recurrent networks, occasionally. Mostly displaced by ReLU.',
    },
    relu: {
      name: 'ReLU',
      formula: 'ReLU(x) = max(0, x)',
      f:  (x) => Math.max(0, x),
      df: (x) => (x > 0 ? 1 : 0),
      range: '[0, ∞)',
      color: '#262320',
      blurb: "Brutally simple. If positive, pass through. If negative, zero. The kink at the origin is technically non-differentiable but nobody cares. Cheap to compute, doesn't saturate on the positive side, and somehow makes deep networks actually trainable.",
      use: 'Default for hidden layers in feedforward and convolutional networks since ~2012.',
    },
    leaky: {
      name: 'Leaky ReLU',
      formula: 'f(x) = max(αx, x),  α = 0.1',
      f:  (x) => (x > 0 ? x : 0.1 * x),
      df: (x) => (x > 0 ? 1 : 0.1),
      range: '(−∞, ∞)',
      color: '#8b5a2b',
      blurb: "Fix for ReLU's \"dying neurons\": when too many units get stuck at zero, gradients can't flow back through them. A small slope on the negative side keeps the gradient alive.",
      use: 'When standard ReLU is killing too many neurons. Variants: PReLU, ELU, GELU.',
    },
    gelu: {
      name: 'GELU',
      formula: 'GELU(x) ≈ x · σ(1.702x)',
      f:  (x) => x * (1 / (1 + Math.exp(-1.702 * x))),
      df: (x) => { const s = 1 / (1 + Math.exp(-1.702 * x)); return s + x * s * (1 - s) * 1.702; },
      range: '(−ε, ∞)',
      color: '#5a3a8e',
      blurb: "Smooth, probabilistic ReLU — weights inputs by their cumulative normal probability. The activation in modern transformers (GPT, BERT). Slight dip below zero, then asymptotes to identity.",
      use: 'Transformers, large language models.',
    },
  };

  const state = {
    active: ['sigmoid', 'tanh', 'relu', 'leaky'],
    showDeriv: false,
    hoverX: null,
    focused: 'relu',
  };

  const checksEl = document.getElementById('checks');
  const toggleEl = document.getElementById('toggle-deriv');
  const focusEl = document.getElementById('focus');
  const captionEl = document.getElementById('caption');
  const marginEl = document.getElementById('margin');
  const miniGridEl = document.getElementById('mini-grid');
  const canvas = document.getElementById('canvas');

  // Build checkboxes
  Object.entries(FNS).forEach(([k, fn]) => {
    const lbl = document.createElement('label');
    lbl.innerHTML = `
      <input type="checkbox" ${state.active.includes(k) ? 'checked' : ''} />
      <span class="swatch" style="background:${fn.color}"></span>
      <span>${fn.name}</span>`;
    const cb = lbl.querySelector('input');
    cb.addEventListener('change', () => {
      if (cb.checked) state.active.push(k); else state.active = state.active.filter(x => x !== k);
      render();
    });
    lbl.dataset.fn = k;
    checksEl.appendChild(lbl);
  });

  toggleEl.addEventListener('click', () => {
    state.showDeriv = !state.showDeriv;
    render();
  });

  // Focus presets
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

  // Build mini-grid (always visible at the bottom)
  Object.entries(FNS).forEach(([k, fn]) => {
    const wrap = document.createElement('div');
    wrap.className = 'mini';
    wrap.dataset.fn = k;
    wrap.innerHTML = `
      <canvas></canvas>
      <div class="name" style="color:${fn.color}">${fn.name}</div>
      <div class="range">${fn.range}</div>`;
    miniGridEl.appendChild(wrap);
  });

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
      const yMin = state.showDeriv ? -0.2 : -1.2;
      const yMax = state.showDeriv ? 1.1 : 2.5;
      const tx = (x) => (x - xMin) / (xMax - xMin) * W;
      const ty = (y) => H - (y - yMin) / (yMax - yMin) * H;
      ctx.strokeStyle = 'rgba(38,35,32,0.18)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(tx(0), 0); ctx.lineTo(tx(0), H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, ty(0)); ctx.lineTo(W, ty(0)); ctx.stroke();
      ctx.strokeStyle = fn.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      let first = true;
      for (let px = 0; px <= W; px++) {
        const x = xMin + (px / W) * (xMax - xMin);
        const y = state.showDeriv ? fn.df(x) : fn.f(x);
        if (first) { ctx.moveTo(px, ty(y)); first = false; }
        else ctx.lineTo(px, ty(y));
      }
      ctx.stroke();
    });
  }

  // Main chart (rectangular)
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    const xMin = -6, xMax = 6;
    const yMin = state.showDeriv ? -0.2 : -1.5;
    const yMax = state.showDeriv ? 1.2  :  3;
    const tx = (x) => (x - xMin) / (xMax - xMin) * size.w;
    const ty = (y) => size.h - (y - yMin) / (yMax - yMin) * size.h;

    // Grid
    ctx.strokeStyle = 'rgba(38,35,32,0.06)';
    ctx.lineWidth = 1;
    for (let i = xMin; i <= xMax; i++) {
      ctx.beginPath(); ctx.moveTo(tx(i), 0); ctx.lineTo(tx(i), size.h); ctx.stroke();
    }
    for (let y = Math.ceil(yMin); y <= yMax; y++) {
      ctx.beginPath(); ctx.moveTo(0, ty(y)); ctx.lineTo(size.w, ty(y)); ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(tx(0), 0); ctx.lineTo(tx(0), size.h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, ty(0)); ctx.lineTo(size.w, ty(0)); ctx.stroke();

    // Axis labels
    ctx.fillStyle = 'rgba(38,35,32,0.5)';
    ctx.font = '11px "JetBrains Mono", monospace';
    for (let i = xMin + 2; i <= xMax - 2; i += 2) {
      if (i !== 0) ctx.fillText(i, tx(i) + 3, ty(0) - 4);
    }

    // Curves
    for (const k of state.active) {
      const fn = FNS[k];
      ctx.strokeStyle = fn.color;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      let first = true;
      for (let px = 0; px <= size.w; px += 1) {
        const x = xMin + (px / size.w) * (xMax - xMin);
        const y = state.showDeriv ? fn.df(x) : fn.f(x);
        if (first) { ctx.moveTo(px, ty(y)); first = false; }
        else ctx.lineTo(px, ty(y));
      }
      ctx.stroke();
      // Right-edge label
      const yEnd = state.showDeriv ? fn.df(xMax - 0.3) : fn.f(xMax - 0.3);
      ctx.fillStyle = fn.color;
      ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
      ctx.fillText(fn.name, tx(xMax - 0.3) + 6, ty(yEnd) + 4);
    }

    // Hover line
    if (state.hoverX !== null) {
      ctx.strokeStyle = 'rgba(38,35,32,0.4)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(tx(state.hoverX), 0); ctx.lineTo(tx(state.hoverX), size.h);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#262320';
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.fillText('x = ' + state.hoverX.toFixed(2), tx(state.hoverX) + 6, 14);
      for (const k of state.active) {
        const fn = FNS[k];
        const y = state.showDeriv ? fn.df(state.hoverX) : fn.f(state.hoverX);
        ctx.fillStyle = fn.color;
        ctx.beginPath();
        ctx.arc(tx(state.hoverX), ty(y), 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  canvas.addEventListener('mousemove', (e) => {
    const r = canvas.getBoundingClientRect();
    state.hoverX = -6 + (e.clientX - r.left) / r.width * 12;
    canvasCtl.redraw();
  });
  canvas.addEventListener('mouseleave', () => {
    state.hoverX = null;
    canvasCtl.redraw();
  });

  // Render
  function render() {
    // Update checkboxes (for programmatic state changes — not strictly needed here but keeps code uniform).
    Array.from(checksEl.children).forEach((lbl) => {
      lbl.querySelector('input').checked = state.active.includes(lbl.dataset.fn);
    });
    Array.from(focusEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.fn === state.focused);
    });
    toggleEl.style.background = state.showDeriv ? 'var(--ink)' : 'transparent';
    toggleEl.style.color = state.showDeriv ? 'var(--bg)' : 'var(--ink)';
    toggleEl.textContent = state.showDeriv ? "Showing derivatives ƒ′(x)" : 'Show derivatives';

    captionEl.innerHTML = state.showDeriv
      ? '<span class="ml-cap-num">·</span> Derivatives — where these are flat, gradients vanish during backprop. Sigmoid and tanh both saturate at their tails; ReLU is a simple zero/one step; GELU stays smooth.'
      : '<span class="ml-cap-num">·</span> The shape of each function on (−6, 6). Hover the chart to read off values for every active curve at the same x.';

    const fn = FNS[state.focused];
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
