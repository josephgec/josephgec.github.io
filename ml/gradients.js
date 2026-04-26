/* Vol. III — Gradients & Derivatives */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, arrow,
          ACCENT, INK, INK_FADE, RULE_FAINT, RULE_AXIS, ACCENT_SOFT, ACCENT_FAINT,
          SERIF_LABEL_SM, MONO_LABEL } = V;

  const FUNCS_1D = [
    // Critical points are listed so we can mark them on the curve and auto-pause near them.
    { name: 'Parabola',  f: (x) => x*x*0.4 - 1,           df: (x) => 0.8*x,                          range: [-3, 3], crits: [0] },
    { name: 'Cubic',     f: (x) => 0.15*x*x*x - 0.6*x,    df: (x) => 0.45*x*x - 0.6,                 range: [-3, 3],
                          crits: [-Math.sqrt(0.6/0.45), Math.sqrt(0.6/0.45)] },
    { name: 'Sine',      f: (x) => 1.4*Math.sin(x*1.2),   df: (x) => 1.4*1.2*Math.cos(x*1.2),        range: [-3, 3],
                          crits: [-Math.PI/(2*1.2), Math.PI/(2*1.2)] },
    { name: 'Bell',      f: (x) => 2*Math.exp(-x*x*0.5),  df: (x) => 2*Math.exp(-x*x*0.5)*(-x),      range: [-3, 3], crits: [0] },
    { name: 'Loss-like', f: (x) => 0.3*x*x + Math.sin(x*2)*0.4 + 1,
                          df: (x) => 0.6*x + 0.8*Math.cos(x*2),                                       range: [-3, 3],
                          // Found by inspection / numeric solve of 0.6x + 0.8cos(2x) = 0
                          crits: [-2.40, -1.16, 0.10, 1.50] },
  ];

  const FUNCS_2D = [
    { name: 'Bowl',      f: (x,y) => 0.4*(x*x + y*y),     grad: (x,y) => [0.8*x, 0.8*y] },
    { name: 'Saddle',    f: (x,y) => 0.5*(x*x - y*y),     grad: (x,y) => [x, -y] },
    { name: 'Two wells', f: (x,y) => 0.15*((x*x-2)*(x*x-2) + y*y*2),
                          grad: (x,y) => [0.6*(x*x-2)*x, 0.6*y] },
    { name: 'Banana',    f: (x,y) => 0.05*((1-x)*(1-x) + 8*(y - x*x)*(y - x*x)),
                          grad: (x,y) => [0.05*(-2*(1-x) - 32*x*(y - x*x)), 0.05*16*(y - x*x)] },
    { name: 'Ripples',   f: (x,y) => 1 - Math.exp(-(x*x+y*y)*0.15) * Math.cos(Math.sqrt(x*x+y*y)*1.5),
                          grad: (x,y) => {
                            const r2 = x*x + y*y, r = Math.sqrt(r2) + 1e-9;
                            const e = Math.exp(-r2*0.15);
                            const c = Math.cos(r*1.5), s = Math.sin(r*1.5);
                            const dfx = -e*(-0.3*x)*c + e*s*1.5*(x/r);
                            const dfy = -e*(-0.3*y)*c + e*s*1.5*(y/r);
                            return [-dfx, -dfy];
                          } },
  ];

  const state = {
    mode: '1d',
    // Land on Cubic, not Parabola: it has two critical points so the dashed
    // f'=0 rings and Play's auto-pause-at-zero are visible from the first frame.
    func1d: 1,
    func2d: 0,
    x: 0.8,
    p: [1.2, -0.6],
    eta: 0.18,
    descentPath: null, // array of [x,y] when running
    descending: false,
    descentStep: 0,
    descentRaf: null,
    playing: false,
    rafId: null,
  };

  // ───────── DOM ─────────
  const modeListEl = document.getElementById('mode-list');
  const funcsEl = document.getElementById('funcs');
  const funcLabelEl = document.getElementById('func-label');
  const readoutEl = document.getElementById('readout');
  const canvasFrameEl = document.getElementById('canvas-frame');
  const canvas = document.getElementById('canvas');
  const captionEl = document.getElementById('caption');
  const marginEl = document.getElementById('margin');
  const playControls = document.getElementById('play-controls');
  const playBtn = document.getElementById('play-btn');
  const xSlider = document.getElementById('x-slider');
  const xLabel = document.getElementById('x-label');
  const descendControls = document.getElementById('descend-controls');
  const descendBtn = document.getElementById('descend-btn');
  const etaSlider = document.getElementById('eta-slider');
  const etaLabel = document.getElementById('eta-label');

  // ───────── Mode buttons ─────────
  Array.from(modeListEl.children).forEach((btn) => {
    btn.addEventListener('click', () => {
      state.mode = btn.dataset.mode;
      stopPlay();
      stopDescend();
      render();
    });
  });

  // ───────── Function presets ─────────
  function renderFuncList() {
    funcsEl.innerHTML = '';
    const list = state.mode === '1d' ? FUNCS_1D : FUNCS_2D;
    const idx = state.mode === '1d' ? state.func1d
                                    : state.func2d; // 2d list also used for descend
    list.forEach((fn, i) => {
      const b = document.createElement('button');
      b.className = 'ml-preset' + (i === idx ? ' active' : '');
      b.textContent = fn.name;
      b.addEventListener('click', () => {
        if (state.mode === '1d') state.func1d = i;
        else state.func2d = i;
        stopDescend();
        state.descentPath = null;
        render();
      });
      funcsEl.appendChild(b);
    });
    funcLabelEl.textContent = state.mode === '1d' ? 'Function' : 'Surface';
  }

  // ───────── 1D play / slider ─────────
  xSlider.addEventListener('input', (e) => {
    stopPlay();
    state.x = +e.target.value;
    render();
  });
  playBtn.addEventListener('click', () => {
    if (state.playing) stopPlay(); else startPlay();
  });
  function startPlay() {
    state.playing = true;
    playBtn.textContent = 'Pause';
    cancelAnimationFrame(state.rafId);
    const tick = () => {
      const fn = FUNCS_1D[state.func1d];
      state.x += 0.02;
      if (state.x > fn.range[1]) state.x = fn.range[0];
      // Auto-pause near a critical point so the user can read "f'(x) = 0".
      const m = fn.df(state.x);
      const nearCrit = fn.crits.some((c) => Math.abs(c - state.x) < 0.02);
      if (nearCrit && Math.abs(m) < 0.1) {
        stopPlay();
        render();
        // Auto-resume after a short pause so the animation keeps sweeping.
        setTimeout(() => { if (!state.playing) startPlay(); }, 1100);
        return;
      }
      render();
      if (state.playing) state.rafId = requestAnimationFrame(tick);
    };
    state.rafId = requestAnimationFrame(tick);
  }
  function stopPlay() {
    state.playing = false;
    playBtn.textContent = 'Play';
    cancelAnimationFrame(state.rafId);
  }

  // ───────── 2D drag (in 2d AND descend modes — drag sets the start point) ─────────
  canvas.addEventListener('pointerdown', (e) => {
    if (state.mode === '1d') return;
    const apply = (ev) => {
      const r = canvas.getBoundingClientRect();
      const range = 3.5;
      const sx = canvasCtl.getSize().w / (range * 2);
      const x = (ev.clientX - r.left - r.width/2) / sx;
      const y = -(ev.clientY - r.top - r.height/2) / sx;
      state.p = [Math.max(-3.4, Math.min(3.4, x)), Math.max(-3.4, Math.min(3.4, y))];
      // In descend mode, dragging clears any old path so the user picks a new start.
      if (state.mode === 'descend') {
        stopDescend();
        state.descentPath = null;
      }
      render();
    };
    apply(e);
    const onMove = (ev) => apply(ev);
    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  });

  // ───────── Descend controls ─────────
  etaSlider.addEventListener('input', (e) => {
    state.eta = +e.target.value;
    etaLabel.textContent = `η = ${formatNum(state.eta)}`;
    // Don't restart automatically — just re-label.
  });
  descendBtn.addEventListener('click', () => {
    if (state.descending) stopDescend();
    else startDescend();
  });
  function startDescend() {
    const fn = FUNCS_2D[state.func2d];
    state.descending = true;
    state.descentStep = 0;
    state.descentPath = [state.p.slice()];
    descendBtn.textContent = 'Pause';
    let stepIdx = 0;
    const step = () => {
      if (!state.descending) return;
      const last = state.descentPath[state.descentPath.length - 1];
      const [gx, gy] = fn.grad(last[0], last[1]);
      const mag = Math.hypot(gx, gy);
      // Shrink step size when gradient is small (close to a minimum) — visualizes "step shrinks near minima".
      const damp = mag < 0.5 ? Math.max(0.25, mag / 0.5) : 1;
      const eta = state.eta * damp;
      const nx = last[0] - gx * eta;
      const ny = last[1] - gy * eta;
      // Stop when essentially flat or out of bounds.
      if (mag < 0.005 || Math.abs(nx) > 3.4 || Math.abs(ny) > 3.4 || stepIdx > 200) {
        stopDescend();
        return;
      }
      state.descentPath.push([nx, ny]);
      stepIdx += 1;
      state.descentStep = stepIdx;
      render();
      state.descentRaf = requestAnimationFrame(step);
    };
    state.descentRaf = requestAnimationFrame(step);
  }
  function stopDescend() {
    state.descending = false;
    descendBtn.textContent = 'Drop marble';
    cancelAnimationFrame(state.descentRaf);
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    if (state.mode === '1d') draw1d(ctx, size);
    else draw2d(ctx, size);
  }

  function draw1d(ctx, size) {
    const fn = FUNCS_1D[state.func1d];
    const xMin = fn.range[0] - 0.2, xMax = fn.range[1] + 0.2;
    const yMin = -3, yMax = 3;
    const sx = size.w / (xMax - xMin);
    const sy = size.h / (yMax - yMin);
    const toPx = (x, y) => [(x - xMin) * sx, size.h - (y - yMin) * sy];

    // Grid
    ctx.strokeStyle = RULE_FAINT;
    ctx.lineWidth = 1;
    for (let gx = Math.ceil(xMin); gx <= xMax; gx++) {
      ctx.beginPath();
      ctx.moveTo(...toPx(gx, yMin)); ctx.lineTo(...toPx(gx, yMax)); ctx.stroke();
    }
    for (let gy = Math.ceil(yMin); gy <= yMax; gy++) {
      ctx.beginPath();
      ctx.moveTo(...toPx(xMin, gy)); ctx.lineTo(...toPx(xMax, gy)); ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = RULE_AXIS;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(...toPx(xMin, 0)); ctx.lineTo(...toPx(xMax, 0)); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(...toPx(0, yMin)); ctx.lineTo(...toPx(0, yMax)); ctx.stroke();

    // Curve
    ctx.strokeStyle = INK;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) {
      const xx = xMin + (xMax - xMin) * i / 200;
      const yy = fn.f(xx);
      const [px, py] = toPx(xx, yy);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // ── Critical-point ring markers (where f'(x) = 0). ──
    fn.crits.forEach((cx) => {
      if (cx < fn.range[0] || cx > fn.range[1]) return;
      const cy = fn.f(cx);
      const [px, py] = toPx(cx, cy);
      ctx.strokeStyle = 'rgba(122,31,36,0.55)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(px, py, 7, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(122,31,36,0.55)';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText("f'=0", px, py - 12);
    });

    // Tangent line at state.x
    const x0 = state.x, y0 = fn.f(x0), m = fn.df(x0);
    const tLen = 1.6;
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(...toPx(x0 - tLen, y0 - m*tLen));
    ctx.lineTo(...toPx(x0 + tLen, y0 + m*tLen));
    ctx.stroke();

    // Rise/run triangle (1 unit run, m unit rise) — visualizes "slope = m"
    ctx.strokeStyle = 'rgba(122,31,36,0.45)';
    ctx.fillStyle = 'rgba(122,31,36,0.10)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(...toPx(x0, y0));
    ctx.lineTo(...toPx(x0 + 1, y0));
    ctx.lineTo(...toPx(x0 + 1, y0 + m));
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.setLineDash([]);

    // Point on curve
    ctx.fillStyle = ACCENT;
    ctx.beginPath();
    ctx.arc(...toPx(x0, y0), 5, 0, Math.PI*2);
    ctx.fill();
    ctx.fillStyle = '#fffdf6';
    ctx.beginPath();
    ctx.arc(...toPx(x0, y0), 2.2, 0, Math.PI*2);
    ctx.fill();

    // Readouts (top-left)
    ctx.fillStyle = INK_FADE;
    ctx.font = SERIF_LABEL_SM;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(`x = ${formatNum(x0)}`, 18, 22);
    ctx.fillText(`f(x) = ${formatNum(y0)}`, 18, 42);
    ctx.fillStyle = ACCENT;
    ctx.fillText(`f ′(x) = ${formatNum(m)}`, 18, 62);
  }

  function draw2d(ctx, size) {
    const fn = FUNCS_2D[state.func2d];
    const range = 3.5;
    const sx = size.w / (range * 2);
    const toPx = (x, y) => [size.w/2 + x*sx, size.h/2 - y*sx];

    // Sample the surface
    const cells = 90;
    const step = (range * 2) / cells;
    let zMin = Infinity, zMax = -Infinity;
    const grid = [];
    for (let i = 0; i <= cells; i++) {
      grid.push([]);
      for (let j = 0; j <= cells; j++) {
        const x = -range + j*step, y = -range + i*step;
        const z = fn.f(x, y);
        grid[i].push(z);
        if (z < zMin) zMin = z;
        if (z > zMax) zMax = z;
      }
    }

    // ── Phong-ish shaded heatmap. Light = upper-left in math coords (so
    //    in canvas pixel coords too, since y is just flipped). Lambertian
    //    shading on the surface normal n = (-fx, -fy, 1) normalized. ──
    const cellPx = size.w / cells;
    const lx = -0.55, ly = 0.6, lz = 0.7; // light direction (math frame, not normalized but close)
    const lLen = Math.hypot(lx, ly, lz);
    for (let i = 0; i < cells; i++) {
      for (let j = 0; j < cells; j++) {
        const z = grid[i][j];
        const t = (z - zMin) / (zMax - zMin + 1e-9);
        // Base color: cream → ink
        let r = 247 - t*210;
        let g = 244 - t*210;
        let b = 236 - t*204;
        // Shading: numerical gradient from neighboring grid cells
        const zR = grid[i][Math.min(cells, j + 1)];
        const zU = grid[Math.min(cells, i + 1)][j];
        const fx = (zR - z) / step;
        const fy = (zU - z) / step;
        // Surface normal (math frame). Dot with light, clamp.
        const nLen = Math.hypot(-fx, -fy, 1);
        const dot = (-fx * lx + -fy * ly + 1 * lz) / (nLen * lLen);
        const ambient = 0.65;
        const diffuse = 0.45;
        const shade = ambient + diffuse * Math.max(0, dot);
        // Apply shading multiplicatively.
        r = Math.max(0, Math.min(255, r * shade));
        g = Math.max(0, Math.min(255, g * shade));
        b = Math.max(0, Math.min(255, b * shade));
        ctx.fillStyle = `rgb(${r|0},${g|0},${b|0})`;
        const [px, py] = toPx(-range + j*step, -range + (i + 1)*step);
        ctx.fillRect(px, py, cellPx + 1, cellPx + 1);
      }
    }

    // Contour lines (marching squares)
    const levels = 10;
    ctx.strokeStyle = 'rgba(38,35,32,0.35)';
    ctx.lineWidth = 0.7;
    const contourSegs = []; // [{lv, p1, p2}] — collected for the perpendicularity demo.
    for (let l = 1; l < levels; l++) {
      const lv = zMin + (zMax - zMin) * (l / levels);
      for (let i = 0; i < cells; i++) {
        for (let j = 0; j < cells; j++) {
          const a = grid[i][j], b = grid[i][j+1], c = grid[i+1][j+1], d = grid[i+1][j];
          const x0 = -range + j*step, y0 = -range + i*step;
          const x1 = x0 + step, y1 = y0 + step;
          let idx = 0;
          if (a > lv) idx |= 1;
          if (b > lv) idx |= 2;
          if (c > lv) idx |= 4;
          if (d > lv) idx |= 8;
          if (idx === 0 || idx === 15) continue;
          const interp = (va, vb, xa, ya, xb, yb) => {
            const t = (lv - va) / (vb - va + 1e-9);
            return [xa + t*(xb - xa), ya + t*(yb - ya)];
          };
          const eAB = () => interp(a, b, x0, y0, x1, y0);
          const eBC = () => interp(b, c, x1, y0, x1, y1);
          const eCD = () => interp(c, d, x1, y1, x0, y1);
          const eDA = () => interp(d, a, x0, y1, x0, y0);
          const segs = [];
          if      (idx === 1 || idx === 14) segs.push([eDA(), eAB()]);
          else if (idx === 2 || idx === 13) segs.push([eAB(), eBC()]);
          else if (idx === 3 || idx === 12) segs.push([eDA(), eBC()]);
          else if (idx === 4 || idx === 11) segs.push([eBC(), eCD()]);
          else if (idx === 5)               { segs.push([eDA(), eAB()]); segs.push([eBC(), eCD()]); }
          else if (idx === 6 || idx === 9)  segs.push([eAB(), eCD()]);
          else if (idx === 7 || idx === 8)  segs.push([eDA(), eCD()]);
          else if (idx === 10)              { segs.push([eAB(), eBC()]); segs.push([eDA(), eCD()]); }
          for (const [p1, p2] of segs) {
            ctx.beginPath();
            ctx.moveTo(...toPx(p1[0], p1[1]));
            ctx.lineTo(...toPx(p2[0], p2[1]));
            ctx.stroke();
            contourSegs.push({ lv, p1, p2 });
          }
        }
      }
    }

    // Sparse gradient field
    const fieldN = 9;
    for (let i = 0; i <= fieldN; i++) {
      for (let j = 0; j <= fieldN; j++) {
        const x = -range*0.85 + j*(range*1.7)/fieldN;
        const y = -range*0.85 + i*(range*1.7)/fieldN;
        const [gx, gy] = fn.grad(x, y);
        const mag = Math.hypot(gx, gy);
        if (mag < 1e-3) continue;
        const scl = 0.25 / Math.max(0.4, mag*0.4);
        arrow(ctx, toPx(x, y), toPx(x + gx*scl, y + gy*scl), 'rgba(38,35,32,0.35)', 0.9, 5);
      }
    }

    // ── Descent path (when in descend mode) — fading scallop trail. ──
    if (state.mode === 'descend' && state.descentPath && state.descentPath.length > 1) {
      const path = state.descentPath;
      ctx.lineWidth = 2;
      for (let i = 0; i < path.length - 1; i++) {
        const a = i / (path.length - 1);
        ctx.strokeStyle = `rgba(122,31,36,${0.25 + a * 0.55})`;
        ctx.beginPath();
        ctx.moveTo(...toPx(path[i][0], path[i][1]));
        ctx.lineTo(...toPx(path[i+1][0], path[i+1][1]));
        ctx.stroke();
        // Tiny dots at each step
        ctx.fillStyle = `rgba(122,31,36,${0.20 + a * 0.5})`;
        const [px, py] = toPx(path[i+1][0], path[i+1][1]);
        ctx.beginPath();
        ctx.arc(px, py, 1.5, 0, Math.PI * 2);
        ctx.fill();
      }
      // Marble (current head)
      const head = path[path.length - 1];
      const [hpx, hpy] = toPx(head[0], head[1]);
      ctx.fillStyle = ACCENT;
      ctx.beginPath();
      ctx.arc(hpx, hpy, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#fffdf6';
      ctx.beginPath();
      ctx.arc(hpx, hpy, 2.5, 0, Math.PI * 2);
      ctx.fill();
      // State.p tracks the head while descending, so the dot is the same place as state.p.
      state.p = [head[0], head[1]];
    }

    // Current point's gradient (prominent)
    const [g1, g2] = fn.grad(state.p[0], state.p[1]);
    const arrowEnd = [state.p[0] + g1*0.6, state.p[1] + g2*0.6];
    arrow(ctx, toPx(state.p[0], state.p[1]),
              toPx(arrowEnd[0], arrowEnd[1]), ACCENT, 3, 11);

    // ── Perpendicularity demo: in 2D mode, draw a thin perpendicular guideline
    //    from the gradient arrow tip to the nearest contour piece, with a small
    //    right-angle marker at the contact. ──
    if (state.mode === '2d' && contourSegs.length) {
      const px0 = state.p[0], py0 = state.p[1];
      // Find nearest contour segment to the foot of the arrow.
      let best = null, bestDist = Infinity, bestProj = null;
      for (const seg of contourSegs) {
        // Distance from (px0,py0) to segment p1-p2
        const [a, b] = [seg.p1, seg.p2];
        const ax = b[0] - a[0], ay = b[1] - a[1];
        const t = ((px0 - a[0]) * ax + (py0 - a[1]) * ay) / (ax*ax + ay*ay + 1e-9);
        const tc = Math.max(0, Math.min(1, t));
        const cx = a[0] + ax * tc, cy = a[1] + ay * tc;
        const d = Math.hypot(cx - px0, cy - py0);
        if (d < bestDist) { bestDist = d; best = seg; bestProj = [cx, cy]; }
      }
      if (best && bestDist > 0.01 && bestDist < 1.4) {
        ctx.strokeStyle = 'rgba(38,35,32,0.55)';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(...toPx(state.p[0], state.p[1]));
        ctx.lineTo(...toPx(bestProj[0], bestProj[1]));
        ctx.stroke();
        ctx.setLineDash([]);

        // Right-angle marker at the contact: a tiny square along the contour and along the perpendicular.
        const [tx, ty] = toPx(bestProj[0], bestProj[1]);
        // Direction along contour:
        const ax = best.p2[0] - best.p1[0];
        const ay = best.p2[1] - best.p1[1];
        const aLen = Math.hypot(ax, ay) + 1e-9;
        const ux = ax / aLen, uy = ay / aLen;
        // Direction back toward the gradient origin (perpendicular, in math coords):
        const perpXm = state.p[0] - bestProj[0];
        const perpYm = state.p[1] - bestProj[1];
        const pLen = Math.hypot(perpXm, perpYm) + 1e-9;
        const pxn = perpXm / pLen, pyn = perpYm / pLen;
        const sLen = 8; // pixels
        // Compute pixel directions (canvas y is flipped vs math y)
        const dux = ux * sLen, duy = -uy * sLen;
        const dpx = pxn * sLen, dpy = -pyn * sLen;
        ctx.strokeStyle = 'rgba(38,35,32,0.7)';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(tx + dux, ty + duy);
        ctx.lineTo(tx + dux + dpx, ty + duy + dpy);
        ctx.lineTo(tx + dpx, ty + dpy);
        ctx.stroke();

        // Caption near the contact
        ctx.fillStyle = INK_FADE;
        ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
        ctx.textAlign = 'left';
        ctx.fillText('∇f ⊥ contour', tx + 12, ty + 4);
      }
    }

    ctx.fillStyle = ACCENT;
    const [ppx, ppy] = toPx(state.p[0], state.p[1]);
    ctx.beginPath(); ctx.arc(ppx, ppy, 6, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = '#fffdf6';
    ctx.beginPath(); ctx.arc(ppx, ppy, 2.5, 0, Math.PI*2); ctx.fill();

    // ── Readouts ──
    ctx.fillStyle = INK_FADE;
    ctx.font = SERIF_LABEL_SM;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(`f(${formatNum(state.p[0])}, ${formatNum(state.p[1])}) = ${formatNum(fn.f(state.p[0], state.p[1]))}`, 16, 22);
    ctx.fillStyle = ACCENT;
    ctx.fillText(`∂f/∂x = ${formatNum(g1)}`, 16, 42);
    ctx.fillText(`∂f/∂y = ${formatNum(g2)}`, 16, 62);
    ctx.fillText(`|∇f| = ${formatNum(Math.hypot(g1, g2))}`, 16, 82);

    if (state.mode === 'descend') {
      ctx.fillStyle = INK_FADE;
      ctx.font = SERIF_LABEL_SM;
      ctx.textAlign = 'right';
      ctx.fillText(`step ${state.descentStep}   ·   η = ${formatNum(state.eta)}`,
        size.w - 16, 22);
      if (state.descentPath && state.descentPath.length > 1) {
        const head = state.descentPath[state.descentPath.length - 1];
        ctx.fillText(`f at marble = ${formatNum(fn.f(head[0], head[1]))}`,
          size.w - 16, 42);
      }
    }

    if (state.mode === 'descend' && (!state.descentPath || state.descentPath.length < 2)) {
      ctx.fillStyle = INK_FADE;
      ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText('drag to place a marble · then press "Drop marble"', size.w / 2, size.h - 18);
    }

    // ── Bottom-right legend: orient newcomer to colors / shapes on the canvas. ──
    const legX = size.w - 188;
    const legY = size.h - 108;
    const legW = 176;
    const legH = 96;
    ctx.fillStyle = 'rgba(255,253,246,0.92)';
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.fillRect(legX, legY, legW, legH);
    ctx.strokeRect(legX, legY, legW, legH);
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('LEGEND', legX + 8, legY + 14);
    // big accent arrow = ∇f at this point
    arrow(ctx, [legX + 12, legY + 30], [legX + 36, legY + 30], ACCENT, 3, 10);
    ctx.fillStyle = ACCENT;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.fillText('∇f  here  (uphill)', legX + 44, legY + 33);
    // small grey field arrow = gradient at sampled grid points
    arrow(ctx, [legX + 12, legY + 48], [legX + 30, legY + 48], 'rgba(38,35,32,0.45)', 1, 5);
    ctx.fillStyle = INK_FADE;
    ctx.fillText('∇f  field  (sampled)', legX + 44, legY + 51);
    // contour lines
    ctx.strokeStyle = 'rgba(38,35,32,0.55)';
    ctx.lineWidth = 0.9;
    ctx.beginPath();
    ctx.moveTo(legX + 12, legY + 66); ctx.lineTo(legX + 36, legY + 66);
    ctx.stroke();
    ctx.fillStyle = INK_FADE;
    ctx.fillText('contour  (constant f)', legX + 44, legY + 69);
    // marble / descent path
    if (state.mode === 'descend') {
      ctx.fillStyle = ACCENT;
      ctx.beginPath(); ctx.arc(legX + 22, legY + 84, 4, 0, Math.PI*2); ctx.fill();
      ctx.fillStyle = '#fffdf6';
      ctx.beginPath(); ctx.arc(legX + 22, legY + 84, 1.6, 0, Math.PI*2); ctx.fill();
      ctx.fillStyle = ACCENT;
      ctx.fillText('marble  (descent step)', legX + 44, legY + 87);
    } else {
      // ⊥ marker swatch
      ctx.strokeStyle = 'rgba(38,35,32,0.7)';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(legX + 14, legY + 80); ctx.lineTo(legX + 22, legY + 80);
      ctx.lineTo(legX + 22, legY + 88);
      ctx.stroke();
      ctx.fillStyle = INK_FADE;
      ctx.fillText('⊥  ∇f meets contour', legX + 44, legY + 87);
    }
  }

  // ───────── Caption + marginalia ─────────
  function captionFor(mode) {
    if (mode === '1d') return `<span class="ml-cap-num">·</span> The <span class="ml-ink-orange">accent line</span> is the tangent — it touches the curve at one point with the same slope. That slope is the derivative, <em>f ′(x)</em>. Dashed circles mark the <em>critical points</em> where <em>f ′(x) = 0</em>; Play sweeps through and pauses there.`;
    if (mode === '2d') return `<span class="ml-cap-num">·</span> Drag the dot anywhere on the surface. The big <span class="ml-ink-orange">accent arrow</span> is the gradient <em>∇f</em>. The thin dashed line drops from your dot to the nearest contour at a right angle — that's <em>∇f ⊥ contour</em> made visible.`;
    return `<span class="ml-cap-num">·</span> <em>Gradient descent</em>: drag to place a marble, press Drop. The marble takes steps of size <em>η · ∇f</em> in the downhill direction. Steps shrink near minima; the trail fades from past to present.`;
  }

  function marginFor(mode) {
    if (mode === '1d') return `
      <div class="ml-margin-rule"></div>
      <div class="ml-margin-h">In the margin</div>
      <div class="ml-equation">f ′(x) = lim<sub style="font-style:normal">h→0</sub> [f(x+h) − f(x)] / h</div>
      <p class="quiet" style="margin-top:-8px"><em>f ′(x)</em> (read "<em>f-prime of x</em>") is the slope. <em>lim</em> means "the value as <em>h</em> shrinks toward 0". <em>h</em> is just a tiny step.</p>
      <p>The slope of the line you'd see if you zoomed infinitely far into the curve at <em>x</em>. Up if positive, down if negative, flat at minima and maxima.</p>
      <p class="quiet">At a minimum, <em>f ′(x) = 0</em> — the dashed ring marks this. Gradient descent searches for these flat spots.</p>
      <div class="ml-margin-rule thin"></div>
      <div class="ml-margin-h sm">Up next</div>
      <p class="quiet">Switch to the <em>Descend</em> mode to see <em>x ← x − η ∇f</em> trace a path downhill.</p>`;
    if (mode === '2d') return `
      <div class="ml-margin-rule"></div>
      <div class="ml-margin-h">In the margin</div>
      <div class="ml-equation">∇f = (∂f/∂x, ∂f/∂y)</div>
      <p class="quiet" style="margin-top:-8px"><em>∇f</em> (read "<em>nabla f</em>" or "<em>gradient of f</em>") is a vector pointing in the steepest-uphill direction. <em>∂f/∂x</em> (read "partial-f partial-x") is the partial derivative — slope along x while y is held still. <em>∂</em> is the curly version of <em>d</em> reserved for partial derivatives.</p>
      <p>Two derivatives glued into a vector — one for each coordinate. The vector points <em>uphill, fastest</em>, and its length tells you how steep the climb is.</p>
      <p>The dashed perpendicular and ⊥ marker on the canvas are a visual proof that <em>∇f is perpendicular to the level set</em>. This is why moving along a contour line keeps f the same.</p>
      <p class="quiet">At a minimum the gradient is zero. That is the goal of training: drive ∇L to 0.</p>
      <div class="ml-margin-rule thin"></div>
      <div class="ml-margin-h sm">Up next</div>
      <p class="quiet">Reverse the gradient and you have <em>gradient descent</em> — try the next mode.</p>`;
    return `
      <div class="ml-margin-rule"></div>
      <div class="ml-margin-h">In the margin</div>
      <div class="ml-equation">x ← x − η ∇f(x)</div>
      <p class="quiet" style="margin-top:-8px"><em>x</em> is your position. <em>η</em> (read "<em>eta</em>") is the <em>learning rate</em> — the step size. <em>∇f(x)</em> is the gradient at <em>x</em>. The minus sign sends you downhill. The arrow <em>←</em> means "is replaced by".</p>
      <p>The single line of math behind every neural-net training loop. Small η = patient and stable; large η = fast but liable to overshoot.</p>
      <p>In the <em>Banana</em> surface, the path zigzags down a long elongated valley. That zigzag is exactly why momentum and Adam exist: smarter optimizers cut across the bowl instead of bouncing off the walls.</p>
      <p class="quiet">The marble's step shrinks automatically as <em>|∇f|</em> shrinks — a free annealing that real optimizers replicate explicitly.</p>
      <div class="ml-margin-rule thin"></div>
      <div class="ml-margin-h sm">Why this works</div>
      <p class="quiet">−∇f is the direction of <em>steepest descent</em>. Take small enough steps and you can never go up. Get close to a minimum and the gradient (and so the step) shrink to zero on their own.</p>`;
  }

  function renderReadout() {
    if (state.mode === '1d') {
      const fn = FUNCS_1D[state.func1d];
      const m = fn.df(state.x);
      const angle = Math.atan(m) * 180 / Math.PI;
      readoutEl.innerHTML = `
        <div class="row"><span>x</span><b>${formatNum(state.x)}</b></div>
        <div class="row"><span>f(x)</span><b>${formatNum(fn.f(state.x))}</b></div>
        <div class="row"><span>f ′(x)</span><b>${formatNum(m)}</b></div>
        <div class="row"><span>angle</span><b>${formatNum(angle)}°</b></div>`;
    } else {
      const fn = FUNCS_2D[state.func2d];
      const [gx, gy] = fn.grad(state.p[0], state.p[1]);
      readoutEl.innerHTML = `
        <div class="row"><span>x</span><b>${formatNum(state.p[0])}</b></div>
        <div class="row"><span>y</span><b>${formatNum(state.p[1])}</b></div>
        <div class="row"><span>f</span><b>${formatNum(fn.f(state.p[0], state.p[1]))}</b></div>
        <div class="row"><span>|∇f|</span><b>${formatNum(Math.hypot(gx, gy))}</b></div>`;
    }
  }

  // ───────── Render ─────────
  function render() {
    Array.from(modeListEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.mode === state.mode);
    });
    renderFuncList();
    renderReadout();
    captionEl.innerHTML = captionFor(state.mode);
    marginEl.innerHTML = marginFor(state.mode);

    // 1D plot is wider; 2D / descend are square.
    if (state.mode === '1d') {
      canvasFrameEl.style.aspectRatio = '1.25 / 1';
      canvas.style.cursor = 'default';
      playControls.style.display = 'flex';
      descendControls.style.display = 'none';
      const fn = FUNCS_1D[state.func1d];
      xSlider.min = fn.range[0]; xSlider.max = fn.range[1];
      xSlider.value = state.x;
      xLabel.textContent = `x = ${formatNum(state.x)}`;
    } else {
      canvasFrameEl.style.aspectRatio = '1 / 1';
      canvas.style.cursor = 'grab';
      playControls.style.display = 'none';
      descendControls.style.display = state.mode === 'descend' ? 'flex' : 'none';
      stopPlay();
      etaSlider.value = state.eta;
      etaLabel.textContent = `η = ${formatNum(state.eta)}`;
    }
    canvasCtl.redraw();
  }

  render();
})();
