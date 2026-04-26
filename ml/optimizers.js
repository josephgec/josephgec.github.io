/* Vol. XIII — A Race of Optimizers */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, INK_FADE } = V;

  const SURFACES = {
    ravine: {
      name: 'Narrow ravine',
      blurb: "A long, curved valley. Vanilla SGD oscillates across the steep walls; momentum smooths the descent; adaptive methods find a path quickly.",
      f: (x, y) => 0.05 * (1 - x) ** 2 + 0.5 * (y - x * x) ** 2,
      g: (x, y) => [-0.1 * (1 - x) - 2 * x * (y - x * x), (y - x * x)],
      range: 2.5, init: [-1.8, 2.2],
    },
    saddle: {
      name: 'Saddle',
      blurb: "A flat ridge crossed by a descending spine. Plain SGD hangs around the saddle for ages; momentum eventually breaks free; Adam barely notices.",
      f: (x, y) => 0.3 * x * x - 0.3 * y * y,
      g: (x, y) => [0.6 * x, -0.6 * y],
      range: 3, init: [0.05, 0.05],
    },
    bowl: {
      name: 'Bowl',
      blurb: "The easy case. Everyone arrives — but momentum methods overshoot before settling, while plain SGD walks straight in.",
      f: (x, y) => 0.5 * x * x + 0.5 * y * y,
      g: (x, y) => [x, y],
      range: 3, init: [-2.4, 2.0],
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
    },
  };

  const OPTIMIZERS = {
    sgd: {
      name: 'SGD', color: '#262320',
      blurb: 'Plain steepest descent. <em>w ← w − η ∇L.</em>',
      rule: 'No memory. Each step uses only the current gradient.',
      use:  'Convex problems, or when you want simplicity. Often a strong baseline if you tune the schedule.',
      init: (pos) => ({ pos: pos.slice() }),
      step: (s, g, lr) => ({ pos: [s.pos[0] - lr * g[0], s.pos[1] - lr * g[1]] }),
    },
    momentum: {
      name: 'Momentum', color: '#7a1f24',
      blurb: 'Accumulates velocity in consistent directions. <em>v ← βv + ∇L; w ← w − ηv</em>. β = 0.9.',
      rule: 'Remembers a running velocity — keeps moving even when the gradient flips.',
      use:  "Anywhere SGD oscillates. Standard for training computer-vision networks (often with Nesterov's tweak).",
      init: (pos) => ({ pos: pos.slice(), v: [0, 0] }),
      step: (s, g, lr) => {
        const beta = 0.9;
        const v = [beta * s.v[0] + g[0], beta * s.v[1] + g[1]];
        return { pos: [s.pos[0] - lr * v[0], s.pos[1] - lr * v[1]], v };
      },
    },
    rmsprop: {
      name: 'RMSprop', color: '#3a6b5e',
      blurb: 'Per-coordinate scaling by recent gradient magnitude. <em>s ← βs + (1−β)g²; w ← w − η g/√s</em>. β = 0.9.',
      rule: 'Remembers per-axis gradient magnitudes — shrinks steps where gradients are large, lengthens them where small.',
      use:  'Recurrent networks and any setting where gradient scales vary wildly across parameters.',
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
      blurb: 'Momentum × RMSprop with bias correction. The default optimizer for most modern deep learning.',
      rule: 'Combines both: a velocity AND a per-axis scale, with bias correction for early steps.',
      use:  'The default for transformers and most language modeling. Robust across hyperparameters, fast to converge.',
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
    lr: 0.08,
    states: null,
    paths: null,
    step: 0,
    running: false,
    timer: null,
    focused: 'adam',
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
  const lrSlider = document.getElementById('lr-slider');
  const lrLabel = document.getElementById('lr-label');
  const runBtn = document.getElementById('run-btn');
  const resetBtn = document.getElementById('reset-btn');
  const standingsEl = document.getElementById('standings');
  const stepLabel = document.getElementById('step-label');
  const captionEl = document.getElementById('caption');
  const marginEl = document.getElementById('margin');
  const canvas = document.getElementById('canvas');
  const lossCanvas = document.getElementById('loss-canvas');

  // Racer checkboxes
  Object.entries(OPTIMIZERS).forEach(([k, o]) => {
    const lbl = document.createElement('label');
    lbl.style.display = 'flex';
    lbl.style.alignItems = 'center';
    lbl.style.gap = '10px';
    lbl.style.cursor = 'pointer';
    lbl.dataset.opt = k;
    lbl.innerHTML = `
      <input type="checkbox" ${state.active.includes(k) ? 'checked' : ''} />
      <span style="width:14px; height:3px; background:${o.color}; display:inline-block"></span>
      <span style="font-family:var(--serif); font-size:15px">${o.name}</span>`;
    const cb = lbl.querySelector('input');
    cb.addEventListener('change', () => {
      if (cb.checked) state.active.push(k);
      else state.active = state.active.filter((x) => x !== k);
      render();
    });
    racersEl.appendChild(lbl);
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

  lrSlider.addEventListener('input', (e) => {
    state.lr = +e.target.value;
    lrLabel.textContent = `η = ${state.lr.toFixed(3)}`;
  });
  runBtn.addEventListener('click', () => {
    if (state.running) stopRun(); else startRun();
  });
  resetBtn.addEventListener('click', reset);

  function startRun() {
    state.running = true;
    runBtn.textContent = '⏸ Pause';
    state.timer = setInterval(() => {
      const surf = SURFACES[state.surfKey];
      const next = {};
      const newPaths = {};
      for (const k of Object.keys(OPTIMIZERS)) {
        const st = state.states[k];
        const g = surf.g(st.pos[0], st.pos[1]);
        next[k] = OPTIMIZERS[k].step(st, g, state.lr);
        newPaths[k] = [...state.paths[k], next[k].pos.slice()].slice(-300);
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
    // Markers
    for (const k of state.active) {
      const o = OPTIMIZERS[k];
      const pos = state.states[k].pos;
      ctx.fillStyle = o.color;
      ctx.beginPath();
      ctx.arc(tx(pos[0]), ty(pos[1]), 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#fffdf6';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(tx(pos[0]), ty(pos[1]), 6, 0, Math.PI * 2);
      ctx.stroke();
    }
    // Start marker
    ctx.strokeStyle = 'rgba(38,35,32,0.5)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.arc(tx(surf.init[0]), ty(surf.init[1]), 8, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
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
    }
  }

  new ResizeObserver(drawLossRace).observe(lossCanvas.parentElement);

  // ───────── Render ─────────
  function render() {
    Array.from(racersEl.children).forEach((lbl) => {
      lbl.querySelector('input').checked = state.active.includes(lbl.dataset.opt);
    });
    Array.from(terrainsEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.surf === state.surfKey);
    });
    const surf = SURFACES[state.surfKey];

    // Standings
    let html = '';
    for (const k of state.active) {
      const o = OPTIMIZERS[k];
      const pos = state.states[k].pos;
      const loss = surf.f(pos[0], pos[1]);
      html += `<div class="row" data-opt="${k}" style="cursor:pointer">
        <span style="color:${o.color}; font-family:var(--serif); font-style:italic">${o.name}</span>
        <b style="font-family:var(--mono)">${formatNum(loss)}</b>
      </div>`;
    }
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
      <p>${f.blurb}</p>
      <div class="ml-margin-rule thin"></div>
      <div class="ml-margin-h sm">In one line</div>
      <p class="quiet">${f.rule}</p>
      <div class="ml-margin-rule thin"></div>
      <div class="ml-margin-h sm">When to use it</div>
      <p class="quiet">${f.use}</p>`;

    canvasCtl.redraw();
    drawLossRace();
  }

  render();
})();
