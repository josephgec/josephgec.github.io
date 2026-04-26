/* Vol. XII — Loss Landscapes */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, INK_FADE } = V;

  const SURFACES = {
    bowl: {
      name: 'Convex bowl',
      blurb: "A clean quadratic. Anywhere you start, gradient descent slides to the unique minimum. Real loss landscapes almost never look like this.",
      f: (x, y) => 0.5 * x * x + 0.5 * y * y,
      g: (x, y) => [x, y],
      range: 3, init: [-2.4, 2.0],
    },
    ravine: {
      name: 'Narrow ravine',
      blurb: "Steep in one direction, shallow in the other — Rosenbrock's valley. Vanilla gradient descent zig-zags miserably down the slope. Momentum helps.",
      f: (x, y) => 0.05 * (1 - x) ** 2 + 0.5 * (y - x * x) ** 2,
      g: (x, y) => [-0.1 * (1 - x) - 2 * x * (y - x * x), (y - x * x)],
      range: 2.5, init: [-1.8, 2.2],
    },
    saddle: {
      name: 'Saddle point',
      blurb: "Flat in one direction, descending in another. Naive descent stalls here. In high dimensions, saddles vastly outnumber local minima — and they're where networks really get stuck.",
      f: (x, y) => 0.3 * x * x - 0.3 * y * y,
      g: (x, y) => [0.6 * x, -0.6 * y],
      range: 3, init: [0.05, -0.05],
    },
    dual: {
      name: 'Two minima',
      blurb: "A non-convex landscape with a global and a local minimum, and a ridge between. Where you start matters.",
      f: (x, y) => {
        const a = Math.exp(-(((x - 1.2) ** 2) + ((y - 0.5) ** 2)) / 1.5);
        const b = Math.exp(-(((x + 1.5) ** 2) + ((y + 1.0) ** 2)) / 1.2);
        return -1.2 * a - 0.8 * b + 0.05 * (x * x + y * y);
      },
      g: (x, y) => {
        const a = Math.exp(-(((x - 1.2) ** 2) + ((y - 0.5) ** 2)) / 1.5);
        const b = Math.exp(-(((x + 1.5) ** 2) + ((y + 1.0) ** 2)) / 1.2);
        const dax = -1.2 * a * (-2 * (x - 1.2) / 1.5);
        const day = -1.2 * a * (-2 * (y - 0.5) / 1.5);
        const dbx = -0.8 * b * (-2 * (x + 1.5) / 1.2);
        const dby = -0.8 * b * (-2 * (y + 1.0) / 1.2);
        return [dax + dbx + 0.1 * x, day + dby + 0.1 * y];
      },
      range: 3, init: [-2.5, 2.5],
    },
    rough: {
      name: 'Rugged terrain',
      blurb: "Many shallow local minima — closer to what real deep-learning loss landscapes look like in low-dim slices. Optimizers bounce between basins.",
      f: (x, y) => 0.05 * (x * x + y * y) + 0.4 * Math.sin(x * 1.5) * Math.cos(y * 1.5) + 0.2 * Math.sin(x * 3) * Math.sin(y * 2.5),
      g: (x, y) => [
        0.1 * x + 0.6 * Math.cos(x * 1.5) * Math.cos(y * 1.5) + 0.6 * Math.cos(x * 3) * Math.sin(y * 2.5),
        0.1 * y - 0.6 * Math.sin(x * 1.5) * Math.sin(y * 1.5) + 0.5 * Math.sin(x * 3) * Math.cos(y * 2.5),
      ],
      range: 3, init: [-2.5, -2.5],
    },
  };

  const state = {
    surfKey: 'ravine',
    pos: SURFACES.ravine.init.slice(),
    path: [SURFACES.ravine.init.slice()],
    running: false,
    timer: null,
    lr: 0.1,
  };

  const terrainsEl = document.getElementById('terrains');
  const lrSlider = document.getElementById('lr-slider');
  const lrLabel = document.getElementById('lr-label');
  const runBtn = document.getElementById('run-btn');
  const resetBtn = document.getElementById('reset-btn');
  const readoutEl = document.getElementById('readout');
  const marginEl = document.getElementById('margin');
  const canvas = document.getElementById('canvas');

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
      const [gx, gy] = surf.g(state.pos[0], state.pos[1]);
      state.pos = [state.pos[0] - state.lr * gx, state.pos[1] - state.lr * gy];
      state.path = [...state.path, state.pos.slice()].slice(-200);
      if (Math.hypot(gx, gy) < 0.01) stopRun();
      // Diverge protection — if marble flies off-screen, stop
      if (Math.abs(state.pos[0]) > 6 || Math.abs(state.pos[1]) > 6) stopRun();
      render();
    }, 60);
  }
  function stopRun() {
    state.running = false;
    runBtn.textContent = '▷ Descend';
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }
  function reset() {
    stopRun();
    const surf = SURFACES[state.surfKey];
    state.pos = surf.init.slice();
    state.path = [surf.init.slice()];
    render();
  }

  // Click on canvas to drop marble
  canvas.addEventListener('click', (e) => {
    const surf = SURFACES[state.surfKey];
    const r = canvas.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    const R = surf.range;
    const x = -R + (px / r.width) * 2 * R;
    const y = R - (py / r.height) * 2 * R;
    stopRun();
    state.pos = [x, y];
    state.path = [[x, y]];
    render();
  });

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: true });
  canvas.style.cursor = 'crosshair';

  function draw(ctx, size) {
    const surf = SURFACES[state.surfKey];
    const R = surf.range;
    const tx = (x) => (x + R) / (2 * R) * size.w;
    const ty = (y) => size.h - (y + R) / (2 * R) * size.h;

    // Sample loss grid
    const cells = 100;
    let lmin = Infinity, lmax = -Infinity;
    const grid = [];
    for (let i = 0; i < cells; i++) {
      grid[i] = [];
      for (let j = 0; j < cells; j++) {
        const x = -R + (j + 0.5) / cells * 2 * R;
        const y = R - (i + 0.5) / cells * 2 * R;
        const v = surf.f(x, y);
        grid[i][j] = v;
        if (v < lmin) lmin = v;
        if (v > lmax) lmax = v;
      }
    }
    // Heatmap with gamma compression so skewed surfaces (one corner with huge loss)
    // still show structure across the low-loss region.
    for (let i = 0; i < cells; i++) {
      for (let j = 0; j < cells; j++) {
        const tRaw = (grid[i][j] - lmin) / (lmax - lmin + 1e-9);
        const t = Math.pow(tRaw, 0.55);
        const r = Math.round(247 - t * 200);
        const g = Math.round(244 - t * 200);
        const b = Math.round(236 - t * 210);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(j * size.w / cells, i * size.h / cells, size.w / cells + 1, size.h / cells + 1);
      }
    }
    // Contour lines (marching squares).
    // Levels are placed by gamma-compressed value too so dense low-loss regions get more lines.
    const levels = 12;
    for (let k = 0; k < levels; k++) {
      const tLevel = Math.pow((k + 0.5) / levels, 1.7); // bias contour density toward low loss
      const lvl = lmin + tLevel * (lmax - lmin);
      ctx.strokeStyle = `rgba(38,35,32,${0.18 + (k % 3 === 0 ? 0.12 : 0)})`;
      ctx.lineWidth = k % 3 === 0 ? 0.9 : 0.5;
      for (let i = 0; i < cells - 1; i++) {
        for (let j = 0; j < cells - 1; j++) {
          const a = grid[i][j], b2 = grid[i][j+1], c2 = grid[i+1][j], d2 = grid[i+1][j+1];
          const idx = (a > lvl ? 1 : 0) | (b2 > lvl ? 2 : 0) | (d2 > lvl ? 4 : 0) | (c2 > lvl ? 8 : 0);
          if (idx === 0 || idx === 15) continue;
          const x0 = j * size.w / cells, y0 = i * size.h / cells;
          const dx = size.w / cells, dy = size.h / cells;
          const lerp = (v1, v2) => (lvl - v1) / (v2 - v1 + 1e-9);
          const top = [x0 + dx * lerp(a, b2), y0];
          const right = [x0 + dx, y0 + dy * lerp(b2, d2)];
          const bot = [x0 + dx * lerp(c2, d2), y0 + dy];
          const left = [x0, y0 + dy * lerp(a, c2)];
          ctx.beginPath();
          const drawSeg = (p1, p2) => { ctx.moveTo(...p1); ctx.lineTo(...p2); };
          if (idx === 1 || idx === 14) drawSeg(top, left);
          else if (idx === 2 || idx === 13) drawSeg(top, right);
          else if (idx === 4 || idx === 11) drawSeg(right, bot);
          else if (idx === 8 || idx === 7)  drawSeg(bot, left);
          else if (idx === 3 || idx === 12) drawSeg(left, right);
          else if (idx === 6 || idx === 9)  drawSeg(top, bot);
          else { drawSeg(top, left); drawSeg(right, bot); }
          ctx.stroke();
        }
      }
    }

    // Path
    if (state.path.length > 1) {
      ctx.strokeStyle = 'rgba(122,31,36,0.85)';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      state.path.forEach((p, i) => {
        const px = tx(p[0]), py = ty(p[1]);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
      // Step dots
      for (let i = 0; i < state.path.length; i += Math.max(1, Math.floor(state.path.length / 40))) {
        ctx.fillStyle = 'rgba(122,31,36,0.6)';
        ctx.beginPath();
        ctx.arc(tx(state.path[i][0]), ty(state.path[i][1]), 1.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Marble
    ctx.fillStyle = '#262320';
    ctx.beginPath();
    ctx.arc(tx(state.pos[0]), ty(state.pos[1]), 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#fffdf6';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(tx(state.pos[0]), ty(state.pos[1]), 6, 0, Math.PI * 2);
    ctx.stroke();

    // Gradient arrow at marble
    const [gx, gy] = surf.g(state.pos[0], state.pos[1]);
    const gnorm = Math.hypot(gx, gy);
    if (gnorm > 0.01) {
      const sc = Math.min(0.5, 0.3 / gnorm) * R * 0.5;
      const tip = [state.pos[0] - gx * sc, state.pos[1] - gy * sc];
      ctx.strokeStyle = '#3a6b5e';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(tx(state.pos[0]), ty(state.pos[1]));
      ctx.lineTo(tx(tip[0]), ty(tip[1]));
      ctx.stroke();
      const ang = Math.atan2(ty(tip[1]) - ty(state.pos[1]), tx(tip[0]) - tx(state.pos[0]));
      ctx.fillStyle = '#3a6b5e';
      ctx.beginPath();
      ctx.moveTo(tx(tip[0]), ty(tip[1]));
      ctx.lineTo(tx(tip[0]) - 7 * Math.cos(ang - 0.4), ty(tip[1]) - 7 * Math.sin(ang - 0.4));
      ctx.lineTo(tx(tip[0]) - 7 * Math.cos(ang + 0.4), ty(tip[1]) - 7 * Math.sin(ang + 0.4));
      ctx.fill();
    }
  }

  function render() {
    const surf = SURFACES[state.surfKey];
    Array.from(terrainsEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.surf === state.surfKey);
    });
    const lossNow = surf.f(state.pos[0], state.pos[1]);
    const gradNow = surf.g(state.pos[0], state.pos[1]);
    readoutEl.innerHTML = `
      <div class="row"><span>position</span><b>(${formatNum(state.pos[0])}, ${formatNum(state.pos[1])})</b></div>
      <div class="row"><span>loss</span><b>${formatNum(lossNow)}</b></div>
      <div class="row"><span>‖∇L‖</span><b>${formatNum(Math.hypot(gradNow[0], gradNow[1]))}</b></div>
      <div class="row"><span>steps</span><b>${state.path.length - 1}</b></div>`;
    marginEl.innerHTML = `
      <div class="ml-margin-rule"></div>
      <div class="ml-margin-h">${surf.name}</div>
      <p>${surf.blurb}</p>
      <div class="ml-margin-rule thin"></div>
      <div class="ml-margin-h sm">Step size matters</div>
      <p class="quiet">Too small: glacial. Too large: overshoot, oscillate, diverge. Try cranking <em>η</em> up on the bowl and watch the marble fly off.</p>
      <div class="ml-margin-rule thin"></div>
      <div class="ml-margin-h sm">In high dimensions</div>
      <p class="quiet">Real networks live in spaces with millions of axes. Empirical work suggests local minima are mostly fine — it's saddle points that slow training to a crawl.</p>`;
    canvasCtl.redraw();
  }

  render();
})();
