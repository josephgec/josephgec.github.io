/* Vol. XIII — A Race of Optimizers */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, INK_FADE } = V;

  // Order matters — terrain buttons render in this order. Ravine + Saddle first, since they
  // are the cases where Adam clearly wins.
  const SURFACES = {
    ravine: {
      name: 'Narrow ravine',
      blurb: "A long, curved valley. Vanilla SGD oscillates across the steep walls; momentum smooths the descent; adaptive methods find a path quickly.",
      f: (x, y) => 0.05 * (1 - x) ** 2 + 0.5 * (y - x * x) ** 2,
      g: (x, y) => [-0.1 * (1 - x) - 2 * x * (y - x * x), (y - x * x)],
      range: 2.5, init: [-1.8, 2.2],
      target: 0,            // approximate min loss
    },
    saddle: {
      name: 'Saddle',
      blurb: "A flat ridge crossed by a descending spine. Plain SGD hangs around the saddle for ages; momentum eventually breaks free; Adam barely notices.",
      f: (x, y) => 0.3 * x * x - 0.3 * y * y,
      g: (x, y) => [0.6 * x, -0.6 * y],
      range: 3, init: [0.05, 0.05],
      target: -2.7,
    },
    bowl: {
      name: 'Bowl',
      blurb: "The easy case. Everyone arrives — but momentum methods overshoot before settling, while plain SGD walks straight in.",
      f: (x, y) => 0.5 * x * x + 0.5 * y * y,
      g: (x, y) => [x, y],
      range: 3, init: [-2.4, 2.0],
      target: 0,
    },
    rough: {
      name: 'Rugged',
      blurb: "Sinusoidal ripples on a gentle bowl. Optimizers sometimes get pinned in shallow local basins. Adam's adaptive scaling helps it skate over the wrinkles.",
      f: (x, y) => 0.05 * (x * x + y * y) + 0.4 * Math.sin(x * 1.5) * Math.cos(y * 1.5) + 0.2 * Math.sin(x * 3) * Math.sin(y * 2.5),
      g: (x, y) => [
        0.1 * x + 0.6 * Math.cos(x * 1.5) * Math.cos(y * 1.5) + 0.6 * Math.cos(x * 3) * Math.sin(y * 2.5),
        0.1 * y - 0.6 * Math.sin(x * 1.5) * Math.sin(y * 1.5) + 0.5 * Math.sin(x * 3) * Math.cos(y * 2.5),
      ],
      range: 3, init: [-2.5, -2.5],
      target: -0.5,
    },
  };

  // Per-optimizer state extractors for the inline mini-bar widgets.
  // Each returns up to two pairs (label, magnitude) — one per axis or one per slot.
  const OPTIMIZERS = {
    sgd: {
      name: 'SGD', color: '#262320',
      defaultLr: 0.08,
      lrMin: 0.005, lrMax: 0.4,
      blurb: 'Plain steepest descent.',
      rule: 'No memory. Each step uses only the current gradient.',
      ruleEq: 'w ← w − η ∇L',
      use:  'Convex problems, or when you want simplicity. Often a strong baseline if you tune the schedule.',
      stateBars: () => [],   // SGD has no internal state
      init: (pos) => ({ pos: pos.slice() }),
      step: (s, g, lr) => ({ pos: [s.pos[0] - lr * g[0], s.pos[1] - lr * g[1]] }),
    },
    momentum: {
      name: 'Momentum', color: '#7a1f24',
      defaultLr: 0.04,
      lrMin: 0.002, lrMax: 0.2,
      blurb: 'Accumulates a velocity in consistent gradient directions.',
      rule: 'Remembers a running velocity — keeps moving even when the gradient flips.',
      ruleEq: 'v ← β v + ∇L,   w ← w − η v',
      use:  "Anywhere SGD oscillates. Standard for training computer-vision networks (often with Nesterov's tweak).",
      // Show |v_x|, |v_y| as two side-by-side bars.
      stateBars: (st) => [
        { label: 'vₓ', val: st.v ? st.v[0] : 0, color: '#7a1f24' },
        { label: 'vᵧ', val: st.v ? st.v[1] : 0, color: '#7a1f24' },
      ],
      init: (pos) => ({ pos: pos.slice(), v: [0, 0] }),
      step: (s, g, lr) => {
        const beta = 0.9;
        const v = [beta * s.v[0] + g[0], beta * s.v[1] + g[1]];
        return { pos: [s.pos[0] - lr * v[0], s.pos[1] - lr * v[1]], v };
      },
    },
    rmsprop: {
      name: 'RMSprop', color: '#3a6b5e',
      defaultLr: 0.05,
      lrMin: 0.002, lrMax: 0.3,
      blurb: 'Per-coordinate scaling by recent squared gradient.',
      rule: 'Remembers per-axis gradient magnitudes — shrinks steps where gradients are large, lengthens them where small.',
      ruleEq: 's ← β s + (1−β) g²,   w ← w − η g/√s',
      use:  'Recurrent networks and any setting where gradient scales vary wildly across parameters.',
      stateBars: (st) => [
        { label: 'sₓ', val: st.s ? Math.sqrt(st.s[0]) : 0, color: '#3a6b5e' },
        { label: 'sᵧ', val: st.s ? Math.sqrt(st.s[1]) : 0, color: '#3a6b5e' },
      ],
      init: (pos) => ({ pos: pos.slice(), s: [0, 0] }),
      step: (s, g, lr) => {
        const beta = 0.9, eps = 1e-7;
        const ns = [beta * s.s[0] + (1 - beta) * g[0] * g[0],
                    beta * s.s[1] + (1 - beta) * g[1] * g[1]];
        return {
          pos: [s.pos[0] - lr * g[0] / Math.sqrt(ns[0] + eps),
                s.pos[1] - lr * g[1] / Math.sqrt(ns[1] + eps)],
          s: ns,
        };
      },
    },
    adam: {
      name: 'Adam', color: '#5a3a8e',
      defaultLr: 0.05,
      lrMin: 0.002, lrMax: 0.3,
      blurb: 'Momentum × RMSprop with bias correction.',
      rule: 'Combines both: a velocity AND a per-axis scale, with bias correction for early steps.',
      ruleEq: 'm ← β₁m + (1−β₁)g,   v ← β₂v + (1−β₂)g²,   w ← w − η m̂/(√v̂ + ε)',
      use:  'The default for transformers and most language modeling. Robust across hyperparameters, fast to converge.',
      stateBars: (st) => [
        { label: 'm', val: st.m ? Math.hypot(st.m[0], st.m[1]) : 0, color: '#5a3a8e' },
        { label: 'v', val: st.v ? Math.sqrt(Math.hypot(st.v[0], st.v[1])) : 0, color: '#8b6abf' },
      ],
      init: (pos) => ({ pos: pos.slice(), m: [0, 0], v: [0, 0], t: 0 }),
      step: (s, g, lr) => {
        const b1 = 0.9, b2 = 0.999, eps = 1e-8;
        const t = s.t + 1;
        const m = [b1 * s.m[0] + (1 - b1) * g[0], b1 * s.m[1] + (1 - b1) * g[1]];
        const v = [b2 * s.v[0] + (1 - b2) * g[0] * g[0], b2 * s.v[1] + (1 - b2) * g[1] * g[1]];
        const mh = [m[0] / (1 - Math.pow(b1, t)), m[1] / (1 - Math.pow(b1, t))];
        const vh = [v[0] / (1 - Math.pow(b2, t)), v[1] / (1 - Math.pow(b2, t))];
        return {
          pos: [s.pos[0] - lr * mh[0] / (Math.sqrt(vh[0]) + eps),
                s.pos[1] - lr * mh[1] / (Math.sqrt(vh[1]) + eps)],
          m, v, t,
        };
      },
    },
  };

  const state = {
    surfKey: 'ravine',
    active: ['sgd', 'momentum', 'rmsprop', 'adam'],
    // Per-optimizer learning rates.
    lrs: Object.fromEntries(Object.entries(OPTIMIZERS).map(([k, o]) => [k, o.defaultLr])),
    states: null,
    paths: null,
    step: 0,
    running: false,
    timer: null,
    focused: 'adam',
    // Per-optimizer step at which loss first dropped close to the minimum (or null).
    nearMinStep: {},
  };
  function initStates() {
    const surf = SURFACES[state.surfKey];
    const o = {};
    for (const k of Object.keys(OPTIMIZERS)) o[k] = OPTIMIZERS[k].init(surf.init);
    return o;
  }
  function initPaths() {
    const surf = SURFACES[state.surfKey];
    const p = {};
    for (const k of Object.keys(OPTIMIZERS)) p[k] = [surf.init.slice()];
    return p;
  }
  state.states = initStates();
  state.paths = initPaths();

  // ───────── DOM ─────────
  const racersEl = document.getElementById('racers');
  const terrainsEl = document.getElementById('terrains');
  const runBtn = document.getElementById('run-btn');
  const resetBtn = document.getElementById('reset-btn');
  const standingsEl = document.getElementById('standings');
  const stepLabel = document.getElementById('step-label');
  const captionEl = document.getElementById('caption');
  const marginEl = document.getElementById('margin');
  const canvas = document.getElementById('canvas');
  const lossCanvas = document.getElementById('loss-canvas');

  // Per-racer rows: [checkbox] [color swatch] [name] [η slider] [η value]
  Object.entries(OPTIMIZERS).forEach(([k, o]) => {
    const row = document.createElement('div');
    row.dataset.opt = k;
    row.style.display = 'grid';
    row.style.gridTemplateColumns = 'auto auto 1fr';
    row.style.alignItems = 'center';
    row.style.columnGap = '8px';
    row.style.rowGap = '2px';
    row.innerHTML = `
      <label style="display:flex; align-items:center; gap:8px; cursor:pointer; grid-column: 1 / 4">
        <input type="checkbox" ${state.active.includes(k) ? 'checked' : ''} />
        <span style="width:14px; height:3px; background:${o.color}; display:inline-block"></span>
        <span style="font-family:var(--serif); font-size:15px; color:${o.color}">${o.name}</span>
        <span class="lr-readout" style="margin-left:auto; font-family:var(--mono); font-size:11px; color:var(--ml-ink-fade)">η=${state.lrs[k].toFixed(3)}</span>
      </label>
      <input type="range" class="lr-slider"
             min="${o.lrMin}" max="${o.lrMax}" step="0.001"
             value="${state.lrs[k]}"
             aria-label="${o.name} learning rate"
             style="grid-column: 1 / 4; width:100%; accent-color:${o.color}; height: 14px" />`;
    const cb = row.querySelector('input[type=checkbox]');
    cb.addEventListener('change', () => {
      if (cb.checked) state.active.push(k);
      else state.active = state.active.filter((x) => x !== k);
      render();
    });
    const lrSlider = row.querySelector('input.lr-slider');
    const lrReadout = row.querySelector('.lr-readout');
    lrSlider.addEventListener('input', (e) => {
      state.lrs[k] = +e.target.value;
      lrReadout.textContent = `η=${state.lrs[k].toFixed(3)}`;
    });
    racersEl.appendChild(row);
  });

  // Terrains
  Object.entries(SURFACES).forEach(([k, s]) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = s.name;
    b.dataset.surf = k;
    b.addEventListener('click', () => {
      state.surfKey = k;
      reset();
    });
    terrainsEl.appendChild(b);
  });

  runBtn.addEventListener('click', () => {
    if (state.running) stopRun(); else startRun();
  });
  resetBtn.addEventListener('click', reset);

  function nearMinThresh(surf) {
    // "near-min loss" = within 5% of the surface range above its minimum.
    const lvls = sampleLossRange(surf);
    return surf.target + 0.05 * (lvls.max - surf.target);
  }
  function sampleLossRange(surf) {
    const R = surf.range, N = 40;
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const x = -R + (j + 0.5) / N * 2 * R;
      const y = R - (i + 0.5) / N * 2 * R;
      const v = surf.f(x, y);
      if (v < mn) mn = v; if (v > mx) mx = v;
    }
    return { min: mn, max: mx };
  }

  function startRun() {
    state.running = true;
    runBtn.textContent = '⏸ Pause';
    state.timer = setInterval(() => {
      const surf = SURFACES[state.surfKey];
      const thresh = nearMinThresh(surf);
      const next = {};
      const newPaths = {};
      for (const k of Object.keys(OPTIMIZERS)) {
        const st = state.states[k];
        const g = surf.g(st.pos[0], st.pos[1]);
        const lr = state.lrs[k];
        next[k] = OPTIMIZERS[k].step(st, g, lr);
        newPaths[k] = [...state.paths[k], next[k].pos.slice()].slice(-300);
        // Has this optimizer just reached near-min?
        const nowLoss = surf.f(next[k].pos[0], next[k].pos[1]);
        if (state.nearMinStep[k] == null && nowLoss <= thresh) {
          state.nearMinStep[k] = state.step + 1;
        }
      }
      state.states = next;
      state.paths = newPaths;
      state.step++;
      render();
    }, 50);
  }
  function stopRun() {
    state.running = false;
    runBtn.textContent = '▷ Race';
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }
  function reset() {
    stopRun();
    state.states = initStates();
    state.paths = initPaths();
    state.step = 0;
    state.nearMinStep = {};
    render();
  }

  // ───────── Race canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, drawRace, { square: true });

  function drawRace(ctx, size) {
    const surf = SURFACES[state.surfKey];
    const R = surf.range;
    const tx = (x) => (x + R) / (2 * R) * size.w;
    const ty = (y) => size.h - (y + R) / (2 * R) * size.h;

    // Heatmap + light contours
    const cells = 90;
    let lmin = Infinity, lmax = -Infinity;
    const grid = [];
    for (let i = 0; i < cells; i++) {
      grid[i] = [];
      for (let j = 0; j < cells; j++) {
        const x = -R + (j + 0.5) / cells * 2 * R;
        const y = R - (i + 0.5) / cells * 2 * R;
        const v = surf.f(x, y);
        grid[i][j] = v;
        if (v < lmin) lmin = v; if (v > lmax) lmax = v;
      }
    }
    for (let i = 0; i < cells; i++) for (let j = 0; j < cells; j++) {
      const tRaw = (grid[i][j] - lmin) / (lmax - lmin + 1e-9);
      const t = Math.pow(tRaw, 0.55);
      const r = Math.round(247 - t * 200), g = Math.round(244 - t * 200), b = Math.round(236 - t * 210);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(j * size.w / cells, i * size.h / cells, size.w / cells + 1, size.h / cells + 1);
    }
    const levels = 10;
    for (let k = 0; k < levels; k++) {
      const tLevel = Math.pow((k + 0.5) / levels, 1.7);
      const lvl = lmin + tLevel * (lmax - lmin);
      ctx.strokeStyle = 'rgba(38,35,32,0.18)';
      ctx.lineWidth = 0.6;
      for (let i = 0; i < cells - 1; i++) for (let j = 0; j < cells - 1; j++) {
        const a = grid[i][j], b2 = grid[i][j+1], c2 = grid[i+1][j], d2 = grid[i+1][j+1];
        const idx = (a > lvl ? 1 : 0) | (b2 > lvl ? 2 : 0) | (d2 > lvl ? 4 : 0) | (c2 > lvl ? 8 : 0);
        if (idx === 0 || idx === 15) continue;
        const x0 = j * size.w / cells, y0 = i * size.h / cells, dx = size.w / cells, dy = size.h / cells;
        const lerp = (v1, v2) => (lvl - v1) / (v2 - v1 + 1e-9);
        const top = [x0 + dx * lerp(a, b2), y0];
        const right = [x0 + dx, y0 + dy * lerp(b2, d2)];
        const bot = [x0 + dx * lerp(c2, d2), y0 + dy];
        const left = [x0, y0 + dy * lerp(a, c2)];
        ctx.beginPath();
        const seg = (p1, p2) => { ctx.moveTo(...p1); ctx.lineTo(...p2); };
        if (idx === 1 || idx === 14) seg(top, left);
        else if (idx === 2 || idx === 13) seg(top, right);
        else if (idx === 4 || idx === 11) seg(right, bot);
        else if (idx === 8 || idx === 7) seg(bot, left);
        else if (idx === 3 || idx === 12) seg(left, right);
        else if (idx === 6 || idx === 9) seg(top, bot);
        else { seg(top, left); seg(right, bot); }
        ctx.stroke();
      }
    }

    // Paths
    for (const k of state.active) {
      const o = OPTIMIZERS[k];
      const path = state.paths[k];
      if (path.length < 2) continue;
      ctx.strokeStyle = o.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      path.forEach((p, i) => {
        const px = tx(p[0]), py = ty(p[1]);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
    }
    // Markers + state bars + flag-if-finished
    for (const k of state.active) {
      const o = OPTIMIZERS[k];
      const pos = state.states[k].pos;
      const px = tx(pos[0]), py = ty(pos[1]);
      // State bars (right of marble)
      const bars = o.stateBars(state.states[k]);
      drawStateBars(ctx, px + 10, py, bars);
      // Marble
      ctx.fillStyle = o.color;
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#fffdf6';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.stroke();
      // "Near-min reached" flag.
      if (state.nearMinStep[k] != null) {
        ctx.strokeStyle = o.color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(px, py - 6);
        ctx.lineTo(px, py - 18);
        ctx.stroke();
        ctx.fillStyle = o.color;
        ctx.beginPath();
        ctx.moveTo(px, py - 18);
        ctx.lineTo(px + 10, py - 14);
        ctx.lineTo(px, py - 10);
        ctx.closePath();
        ctx.fill();
        ctx.fillStyle = o.color;
        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(`@${state.nearMinStep[k]}`, px + 12, py - 14);
      }
    }
    // Start marker
    ctx.strokeStyle = 'rgba(38,35,32,0.5)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.arc(tx(surf.init[0]), ty(surf.init[1]), 8, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Legend for the state-bar widget — bottom-left corner.
    ctx.fillStyle = INK_FADE;
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('▮▮ next to a marble = its memory', 8, size.h - 22);
    ctx.fillText('  (velocity / per-axis scale)', 8, size.h - 8);
    ctx.textAlign = 'right';
    ctx.fillText('▷ flag = first step that reached near-min', size.w - 8, size.h - 8);

    // "First to the minimum" headline — only when at least one racer has finished.
    const finished = Object.entries(state.nearMinStep)
      .filter(([k, v]) => v != null && state.active.includes(k))
      .sort((a, b) => a[1] - b[1]);
    if (finished.length > 0) {
      const [winK, winStep] = finished[0];
      const w = OPTIMIZERS[winK];
      const txt = `${w.name} reached the minimum first — step ${winStep}`;
      ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
      const tw = ctx.measureText(txt).width + 18;
      const tx0 = (size.w - tw) / 2;
      ctx.fillStyle = 'rgba(255,253,246,0.94)';
      ctx.strokeStyle = w.color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.rect(tx0, 8, tw, 22);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = w.color;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(txt, size.w / 2, 19);
    }
  }

  function drawStateBars(ctx, x, y, bars) {
    if (!bars.length) return;
    const barH = 3;
    const spacing = 5;
    const maxBarLen = 22;
    bars.forEach((b, i) => {
      const w = Math.min(maxBarLen, Math.abs(b.val) * 6);
      const yy = y - (bars.length - 1) * spacing / 2 + i * spacing - 1;
      // Subtle backing line
      ctx.fillStyle = 'rgba(38,35,32,0.15)';
      ctx.fillRect(x, yy, maxBarLen, barH);
      // Value
      ctx.fillStyle = b.color;
      ctx.fillRect(x, yy, w, barH);
    });
  }

  // ───────── Loss-over-time chart ─────────
  function drawLossRace() {
    const cv = lossCanvas;
    const dpr = window.devicePixelRatio || 1;
    const W = cv.parentElement.getBoundingClientRect().width, H = 160;
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = W + 'px'; cv.style.height = H + 'px';
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const surf = SURFACES[state.surfKey];
    let maxLen = 0, maxLoss = -Infinity, minLoss = Infinity;
    for (const k of state.active) {
      const path = state.paths[k];
      maxLen = Math.max(maxLen, path.length);
      for (const p of path) {
        const v = surf.f(p[0], p[1]);
        if (v > maxLoss) maxLoss = v;
        if (v < minLoss) minLoss = v;
      }
    }
    if (maxLen < 2) return;
    const range = Math.max(0.001, maxLoss - minLoss);

    // Axes
    ctx.strokeStyle = 'rgba(38,35,32,0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(40, H - 20); ctx.lineTo(W - 10, H - 20); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(40, 8);      ctx.lineTo(40, H - 20); ctx.stroke();
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillText('loss', 6, 14);
    ctx.fillText('step', W - 30, H - 6);

    for (const k of state.active) {
      const o = OPTIMIZERS[k];
      const path = state.paths[k];
      ctx.strokeStyle = o.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      path.forEach((p, i) => {
        const v = surf.f(p[0], p[1]);
        const px = 40 + (i / (maxLen - 1)) * (W - 50);
        const py = (H - 20) - ((v - minLoss) / range) * (H - 30);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
      // Mark "near-min reached" flag on this curve
      const ms = state.nearMinStep[k];
      if (ms != null && ms < path.length) {
        const v = surf.f(path[ms][0], path[ms][1]);
        const fpx = 40 + (ms / (maxLen - 1)) * (W - 50);
        const fpy = (H - 20) - ((v - minLoss) / range) * (H - 30);
        ctx.strokeStyle = o.color;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(fpx, fpy);
        ctx.lineTo(fpx, fpy - 12);
        ctx.stroke();
        ctx.fillStyle = o.color;
        ctx.beginPath();
        ctx.moveTo(fpx, fpy - 12);
        ctx.lineTo(fpx + 8, fpy - 9);
        ctx.lineTo(fpx, fpy - 6);
        ctx.closePath();
        ctx.fill();
      }
    }
  }

  new ResizeObserver(drawLossRace).observe(lossCanvas.parentElement);

  // ───────── Recipe card (margin) ─────────
  // One-liner plain-English gloss for each rule, paired with its formula.
  const RECIPE_GLOSS = {
    sgd: 'just take a step against the gradient',
    momentum: 'roll a heavy ball — it remembers velocity',
    rmsprop: 'normalize each axis by how loud it has been',
    adam: 'do both: momentum on top, per-axis scaling underneath',
  };
  function recipeCardHTML() {
    const rows = Object.entries(OPTIMIZERS).map(([k, o]) => `
      <div style="padding: 6px 0; border-top: 1px solid var(--rule)">
        <div style="display:grid; grid-template-columns: 78px 1fr; gap: 6px">
          <div style="font-family:var(--serif); font-style:italic; font-size:13px; color:${o.color}">${o.name}</div>
          <div style="font-family:var(--mono); font-size:10px; color:var(--ml-ink-soft); line-height:1.5">${o.ruleEq}</div>
        </div>
        <div style="font-family:var(--serif); font-style:italic; font-size:11px; color:var(--ml-ink-fade); padding-left: 84px; margin-top: 2px; line-height: 1.4">${RECIPE_GLOSS[k]}</div>
      </div>
    `).join('');
    return `
      <div class="ml-margin-rule"></div>
      <div class="ml-margin-h">Recipe card</div>
      ${rows}
      <p class="quiet" style="margin-top:8px">The only difference is the update line. Everything else — gradient, loss, weights — is the same.</p>`;
  }

  // ───────── Render ─────────
  function render() {
    Array.from(racersEl.children).forEach((row) => {
      const cb = row.querySelector('input[type=checkbox]');
      cb.checked = state.active.includes(row.dataset.opt);
    });
    Array.from(terrainsEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.surf === state.surfKey);
    });
    const surf = SURFACES[state.surfKey];

    // Standings
    let html = '';
    // Sort by loss for the actual standings
    const ranked = [...state.active].sort((a, b) => {
      const la = surf.f(state.states[a].pos[0], state.states[a].pos[1]);
      const lb = surf.f(state.states[b].pos[0], state.states[b].pos[1]);
      return la - lb;
    });
    ranked.forEach((k, idx) => {
      const o = OPTIMIZERS[k];
      const pos = state.states[k].pos;
      const loss = surf.f(pos[0], pos[1]);
      const tag = state.nearMinStep[k] != null ? ` · @${state.nearMinStep[k]}` : '';
      html += `<div class="row" data-opt="${k}" style="cursor:pointer">
        <span style="color:${o.color}; font-family:var(--serif); font-style:italic">${idx + 1}. ${o.name}</span>
        <b style="font-family:var(--mono)">${formatNum(loss)}${tag}</b>
      </div>`;
    });
    standingsEl.innerHTML = html;
    Array.from(standingsEl.children).forEach((row) => {
      row.addEventListener('click', () => {
        state.focused = row.dataset.opt;
        render();
      });
    });
    stepLabel.textContent = state.step;

    captionEl.innerHTML = `<span class="ml-cap-num">·</span> ${surf.blurb}`;

    const f = OPTIMIZERS[state.focused];
    marginEl.innerHTML = `
      <div class="ml-margin-rule"></div>
      <div class="ml-margin-h" style="color:${f.color}">${f.name}</div>
      <div class="ml-equation" style="font-size:12px; line-height:1.5">${f.ruleEq}</div>
      <p>${f.blurb}</p>
      <div class="ml-margin-rule thin"></div>
      <div class="ml-margin-h sm">In one line</div>
      <p class="quiet">${f.rule}</p>
      <div class="ml-margin-rule thin"></div>
      <div class="ml-margin-h sm">When to use it</div>
      <p class="quiet">${f.use}</p>
      ${recipeCardHTML()}
      <div class="ml-margin-rule thin"></div>
      <div class="ml-margin-h sm">Symbols, plainly</div>
      <p class="quiet"><b>η</b> (eta) = learning rate, the step size. <b>β</b> (beta) = a memory-decay constant in [0,1] — closer to 1 means more weight on the past. <b>v</b> = velocity (running average of gradient). <b>g</b> = the current gradient ∇L. <b>s</b> (in RMSprop) = running average of squared gradient — measures axis-by-axis how loud signals are. <b>m̂, v̂</b> = bias-corrected estimates in Adam. <b>ε</b> (epsilon) = a tiny constant (~10⁻⁸) added under the square root to keep us from dividing by zero.</p>`;

    canvasCtl.redraw();
    drawLossRace();
  }

  render();
})();
