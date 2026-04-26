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
    // Default landing state: the convex bowl — the simplest case to read at a glance.
    // The marble sits up the slope, the green descent arrow is clearly visible, and
    // contour rings show the geometry that gradient descent will follow.
    surfKey: 'bowl',
    pos: SURFACES.bowl.init.slice(),
    path: [SURFACES.bowl.init.slice()],
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
    // Account for legend strip on right (24px reserved at draw time)
    const usableW = r.width - 36;
    if (px > usableW) return;
    const R = surf.range;
    const x = -R + (px / usableW) * 2 * R;
    const y = R - (py / r.height) * 2 * R;
    stopRun();
    state.pos = [x, y];
    state.path = [[x, y]];
    render();
  });

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: true });
  canvas.style.cursor = 'crosshair';

  // Map a normalized loss t∈[0,1] → cream→oxblood color (consistent across heatmap and legend).
  function lossColor(t) {
    const tg = Math.pow(t, 0.55);
    const r = Math.round(247 - tg * 200);
    const g = Math.round(244 - tg * 200);
    const b = Math.round(236 - tg * 210);
    return [r, g, b];
  }

  function draw(ctx, size) {
    const surf = SURFACES[state.surfKey];
    const R = surf.range;
    // Reserve a strip on the right for the vertical color legend.
    const legendW = 36;
    const mapW = size.w - legendW;
    const tx = (x) => (x + R) / (2 * R) * mapW;
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
    const lrange = lmax - lmin + 1e-9;

    // Phong-style shading. Light direction (from upper-left) in 2D projection.
    // Compute a numerical surface gradient at each cell to derive a "normal" component.
    // Treat (x, y, z=loss). Shade = ambient + diffuse · max(0, n·L) on a heightfield.
    const lightX = -1, lightY = -1; // upper-left light direction in image-space (y axis is image-down → upper-left)
    const lightLen = Math.hypot(lightX, lightY);
    const lx = lightX / lightLen, ly = lightY / lightLen;

    // Heatmap with shading
    for (let i = 0; i < cells; i++) {
      for (let j = 0; j < cells; j++) {
        const tRaw = (grid[i][j] - lmin) / lrange;
        const [r, g, b] = lossColor(tRaw);

        // Approx slope in image coords: positive when terrain goes up to the right / down.
        const right = grid[i][Math.min(cells - 1, j + 1)];
        const left  = grid[i][Math.max(0, j - 1)];
        const down  = grid[Math.min(cells - 1, i + 1)][j];
        const up    = grid[Math.max(0, i - 1)][j];
        const dzdx = (right - left) / 2;          // image-x axis
        const dzdy = (down - up) / 2;             // image-y axis
        // Slope normalized by approximate range scale of the loss
        const sScale = Math.max(1e-6, lrange / cells) * 4;
        const sx = dzdx / sScale;
        const sy = dzdy / sScale;
        // Gradient direction in image space points uphill — light hitting it = -dot with light dir.
        // Tilt factor in [-1, 1]: positive = lit (slope faces light), negative = shaded.
        const tilt = -(sx * lx + sy * ly);
        const tiltClamp = Math.max(-1, Math.min(1, tilt));
        // Apply shading: brighten lit slopes, darken shaded slopes.
        const shade = 1 + 0.18 * tiltClamp;
        const sR = Math.max(0, Math.min(255, r * shade));
        const sG = Math.max(0, Math.min(255, g * shade));
        const sB = Math.max(0, Math.min(255, b * shade));
        ctx.fillStyle = `rgb(${sR | 0},${sG | 0},${sB | 0})`;
        ctx.fillRect(j * mapW / cells, i * size.h / cells, mapW / cells + 1, size.h / cells + 1);
      }
    }

    // Contour lines (marching squares) — drawn ON TOP of shaded heatmap.
    const levels = 12;
    for (let k = 0; k < levels; k++) {
      const tLevel = Math.pow((k + 0.5) / levels, 1.7); // bias contour density toward low loss
      const lvl = lmin + tLevel * lrange;
      ctx.strokeStyle = `rgba(38,35,32,${0.18 + (k % 3 === 0 ? 0.12 : 0)})`;
      ctx.lineWidth = k % 3 === 0 ? 0.9 : 0.5;
      for (let i = 0; i < cells - 1; i++) {
        for (let j = 0; j < cells - 1; j++) {
          const a = grid[i][j], b2 = grid[i][j+1], c2 = grid[i+1][j], d2 = grid[i+1][j+1];
          const idx = (a > lvl ? 1 : 0) | (b2 > lvl ? 2 : 0) | (d2 > lvl ? 4 : 0) | (c2 > lvl ? 8 : 0);
          if (idx === 0 || idx === 15) continue;
          const x0 = j * mapW / cells, y0 = i * size.h / cells;
          const dx = mapW / cells, dy = size.h / cells;
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

    // SADDLE preset annotation: crossed arrows showing descent vs. ascent axes.
    if (state.surfKey === 'saddle') {
      const [sx, sy] = [tx(0), ty(0)];
      const len = 36;
      // Descending axis (along y for this saddle: f decreases as |y| grows) — accent red.
      ctx.strokeStyle = 'rgba(122,31,36,0.85)';
      ctx.lineWidth = 2;
      drawArrow(ctx, sx, sy, sx, sy - len);
      drawArrow(ctx, sx, sy, sx, sy + len);
      // Ascending axis (along x) — faded.
      ctx.strokeStyle = 'rgba(38,35,32,0.45)';
      ctx.setLineDash([4, 3]);
      drawArrow(ctx, sx, sy, sx - len, sy);
      drawArrow(ctx, sx, sy, sx + len, sy);
      ctx.setLineDash([]);
      // Annotation
      ctx.fillStyle = '#262320';
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'left';
      ctx.fillText('saddle: gradient ≈ 0,', sx + len + 6, sy - 4);
      ctx.fillText('but not a minimum.', sx + len + 6, sy + 10);
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

    // Gradient arrow at marble (shows -∇L direction, i.e. where the next step goes)
    const [gx, gy] = surf.g(state.pos[0], state.pos[1]);
    const gnorm = Math.hypot(gx, gy);
    // "Arrived" annotation: when ‖∇L‖ is tiny, the marble has parked. Tag it so the user
    // knows the run finished (and isn't just drifting).
    if (gnorm < 0.02 && state.path.length > 3) {
      const [px, py] = [tx(state.pos[0]), ty(state.pos[1])];
      const tag = state.surfKey === 'saddle' ? 'stalled at saddle' : 'arrived · ‖∇L‖ ≈ 0';
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      const tw = ctx.measureText(tag).width + 12;
      let bx = px + 14, by = py - 22;
      if (bx + tw > size.w - legendW - 4) bx = px - tw - 14;
      ctx.fillStyle = 'rgba(255,253,246,0.94)';
      ctx.strokeStyle = 'rgba(58,107,94,0.7)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.rect(bx, by, tw, 18);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#3a6b5e';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(tag, bx + tw / 2, by + 9);
    }
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

    // ── Vertical color legend on the right edge ──
    const legendX = mapW + 8;
    const legendBarW = 14;
    const legendTop = 28;
    const legendBot = size.h - 28;
    const legendH = legendBot - legendTop;
    // Bar (gradient strip — bottom = low, top = high)
    for (let py = 0; py < legendH; py++) {
      const t = 1 - py / legendH;
      const [r, g, b] = lossColor(t);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(legendX, legendTop + py, legendBarW, 1);
    }
    ctx.strokeStyle = 'rgba(38,35,32,0.4)';
    ctx.lineWidth = 1;
    ctx.strokeRect(legendX, legendTop, legendBarW, legendH);
    // Tick marks + numbers (5 ticks)
    ctx.fillStyle = '#262320';
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 4; i++) {
      const t = i / 4;
      const yy = legendTop + (1 - t) * legendH;
      ctx.strokeStyle = 'rgba(38,35,32,0.4)';
      ctx.beginPath();
      ctx.moveTo(legendX + legendBarW, yy);
      ctx.lineTo(legendX + legendBarW + 3, yy);
      ctx.stroke();
      const lvl = lmin + t * lrange;
      ctx.fillStyle = INK_FADE;
      ctx.fillText(lvl.toFixed(1), legendX - 2, yy);
    }
    // Header
    ctx.fillStyle = INK_FADE;
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('LOSS', legendX + legendBarW / 2, legendTop - 8);
    // Hi / Lo annotations
    ctx.font = 'italic 9px "Source Serif 4", Georgia, serif';
    ctx.fillText('hi', legendX + legendBarW / 2, legendTop - 16);
    ctx.fillText('lo', legendX + legendBarW / 2, legendBot + 18);
  }

  function drawArrow(ctx, x1, y1, x2, y2) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    const ang = Math.atan2(y2 - y1, x2 - x1);
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - 7 * Math.cos(ang - 0.4), y2 - 7 * Math.sin(ang - 0.4));
    ctx.lineTo(x2 - 7 * Math.cos(ang + 0.4), y2 - 7 * Math.sin(ang + 0.4));
    ctx.closePath();
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fill();
  }

  // ───────── Margin: 1D-slice diagram (drawn into an SVG-ish margin canvas via inline SVG) ─────────
  // We render a small SVG inline in marginEl during render(). It shows a curve = f(x, 0) along
  // the x-axis of the chosen surface, with a dot at the marble's x-position. This anchors the
  // 2D map as the 2D version of the simpler 1D picture.
  function sliceSVG(surf, posX) {
    const R = surf.range;
    const W = 200, H = 70, pad = 8;
    const N = 80;
    const pts = [];
    let lmin = Infinity, lmax = -Infinity;
    for (let i = 0; i <= N; i++) {
      const x = -R + (i / N) * 2 * R;
      const v = surf.f(x, 0);
      pts.push([x, v]);
      if (v < lmin) lmin = v;
      if (v > lmax) lmax = v;
    }
    const xR = (x) => pad + ((x + R) / (2 * R)) * (W - 2 * pad);
    const yR = (v) => H - pad - ((v - lmin) / (lmax - lmin + 1e-9)) * (H - 2 * pad);
    let path = '';
    pts.forEach(([x, v], i) => {
      const cmd = i === 0 ? 'M' : 'L';
      path += `${cmd}${xR(x).toFixed(1)} ${yR(v).toFixed(1)} `;
    });
    // Marble at (posX, f(posX, 0))
    const my = surf.f(posX, 0);
    const mx = xR(posX);
    const myPx = yR(my);
    return `
      <svg viewBox="0 0 ${W} ${H}" style="width:100%; height:auto" aria-hidden="true">
        <line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" stroke="rgba(38,35,32,0.30)" stroke-width="0.8"/>
        <path d="${path}" fill="none" stroke="#262320" stroke-width="1.2"/>
        <circle cx="${mx.toFixed(1)}" cy="${myPx.toFixed(1)}" r="3.5" fill="#262320"/>
        <text x="${pad}" y="${pad + 6}" font-family="JetBrains Mono, monospace" font-size="8" fill="rgba(38,35,32,0.55)">f(x, 0)</text>
        <text x="${W - pad}" y="${H - pad - 2}" text-anchor="end" font-family="JetBrains Mono, monospace" font-size="8" fill="rgba(38,35,32,0.55)">x</text>
      </svg>`;
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
      <div class="ml-margin-h sm">1-D slice</div>
      ${sliceSVG(surf, state.pos[0])}
      <p class="quiet">A vertical slice of the 2-D landscape, taken along <em>y = 0</em>. Imagine the 2-D map as that curve, swept around. The marble rolls to the lowest point of this 1-D picture; the canvas is the same idea in two axes.</p>
      <div class="ml-margin-rule thin"></div>
      <div class="ml-margin-h sm">Symbols, plainly</div>
      <p class="quiet"><b>η</b> (eta) = learning rate, the step size. <b>∇L</b> (read "nabla L" or "grad L") is a vector — its components are the partial derivatives of L with respect to each weight. ∇L points <em>uphill</em> on the loss surface; training takes <em>−∇L</em> steps to roll downhill. <b>‖∇L‖</b> = its length. The update rule <em>w ← w − η ∇L</em> is the heart of gradient descent.</p>
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
