/* Vol. III — Gradients & Derivatives */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, arrow,
          ACCENT, INK, INK_FADE, RULE_FAINT, RULE_AXIS, ACCENT_SOFT, ACCENT_FAINT,
          SERIF_LABEL_SM, MONO_LABEL } = V;

  const FUNCS_1D = [
    { name: 'Parabola',  f: (x) => x*x*0.4 - 1,           df: (x) => 0.8*x,                          range: [-3, 3] },
    { name: 'Cubic',     f: (x) => 0.15*x*x*x - 0.6*x,    df: (x) => 0.45*x*x - 0.6,                 range: [-3, 3] },
    { name: 'Sine',      f: (x) => 1.4*Math.sin(x*1.2),   df: (x) => 1.4*1.2*Math.cos(x*1.2),        range: [-3, 3] },
    { name: 'Bell',      f: (x) => 2*Math.exp(-x*x*0.5),  df: (x) => 2*Math.exp(-x*x*0.5)*(-x),      range: [-3, 3] },
    { name: 'Loss-like', f: (x) => 0.3*x*x + Math.sin(x*2)*0.4 + 1,
                          df: (x) => 0.6*x + 0.8*Math.cos(x*2),                                       range: [-3, 3] },
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
    func1d: 0,
    func2d: 0,
    x: 0.8,
    p: [1.2, -0.6],
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

  // ───────── Mode buttons ─────────
  Array.from(modeListEl.children).forEach((btn) => {
    btn.addEventListener('click', () => {
      state.mode = btn.dataset.mode;
      stopPlay();
      render();
    });
  });

  // ───────── Function presets (re-rendered when mode changes) ─────────
  function renderFuncList() {
    funcsEl.innerHTML = '';
    const list = state.mode === '1d' ? FUNCS_1D : FUNCS_2D;
    const idx = state.mode === '1d' ? state.func1d : state.func2d;
    list.forEach((fn, i) => {
      const b = document.createElement('button');
      b.className = 'ml-preset' + (i === idx ? ' active' : '');
      b.textContent = fn.name;
      b.addEventListener('click', () => {
        if (state.mode === '1d') state.func1d = i; else state.func2d = i;
        render();
      });
      funcsEl.appendChild(b);
    });
    funcLabelEl.textContent = state.mode === '1d' ? 'Function' : 'Surface';
  }

  // ───────── Slider + play (1D only) ─────────
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

  // ───────── 2D drag handler ─────────
  canvas.addEventListener('pointerdown', (e) => {
    if (state.mode !== '2d') return;
    const apply = (ev) => {
      const r = canvas.getBoundingClientRect();
      const range = 3.5;
      const sx = canvasCtl.getSize().w / (range * 2);
      const x = (ev.clientX - r.left - r.width/2) / sx;
      const y = -(ev.clientY - r.top - r.height/2) / sx;
      state.p = [Math.max(-3.4, Math.min(3.4, x)), Math.max(-3.4, Math.min(3.4, y))];
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

  // ───────── Canvas — let the parent .ml-canvas-frame's aspect-ratio drive the shape
  //    (1.25:1 in 1D mode, 1:1 in 2D). ─────────
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

    // Heatmap (cream → ink as height grows)
    const cellPx = size.w / cells;
    for (let i = 0; i < cells; i++) {
      for (let j = 0; j < cells; j++) {
        const z = grid[i][j];
        const t = (z - zMin) / (zMax - zMin + 1e-9);
        const r = Math.round(247 - t*210);
        const g = Math.round(244 - t*210);
        const b = Math.round(236 - t*204);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        const [px, py] = toPx(-range + j*step, -range + (i + 1)*step);
        ctx.fillRect(px, py, cellPx + 1, cellPx + 1);
      }
    }

    // Contour lines (marching squares)
    const levels = 10;
    ctx.strokeStyle = 'rgba(38,35,32,0.35)';
    ctx.lineWidth = 0.7;
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

    // Current point's gradient (prominent)
    const [g1, g2] = fn.grad(state.p[0], state.p[1]);
    arrow(ctx, toPx(state.p[0], state.p[1]),
              toPx(state.p[0] + g1*0.6, state.p[1] + g2*0.6), ACCENT, 3, 11);

    ctx.fillStyle = ACCENT;
    const [ppx, ppy] = toPx(state.p[0], state.p[1]);
    ctx.beginPath(); ctx.arc(ppx, ppy, 6, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = '#fffdf6';
    ctx.beginPath(); ctx.arc(ppx, ppy, 2.5, 0, Math.PI*2); ctx.fill();

    // Readouts
    ctx.fillStyle = INK_FADE;
    ctx.font = SERIF_LABEL_SM;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(`f(${formatNum(state.p[0])}, ${formatNum(state.p[1])}) = ${formatNum(fn.f(state.p[0], state.p[1]))}`, 16, 22);
    ctx.fillStyle = ACCENT;
    ctx.fillText(`∂f/∂x = ${formatNum(g1)}`, 16, 42);
    ctx.fillText(`∂f/∂y = ${formatNum(g2)}`, 16, 62);
    ctx.fillText(`|∇f| = ${formatNum(Math.hypot(g1, g2))}`, 16, 82);
  }

  // ───────── Caption + marginalia ─────────
  function captionFor(mode) {
    if (mode === '1d') return `<span class="ml-cap-num">·</span> The <span class="ml-ink-orange">accent line</span> is the tangent — it touches the curve at one point with the same slope. That slope is the derivative, <em>f ′(x)</em>. Drag the slider, or hit Play.`;
    return `<span class="ml-cap-num">·</span> Drag the dot anywhere on the surface. The big <span class="ml-ink-orange">accent arrow</span> is the gradient <em>∇f</em> — it points uphill, perpendicular to the level curves. Faint arrows show the gradient field everywhere.`;
  }

  function marginFor(mode) {
    if (mode === '1d') return `
      <div class="ml-margin-rule"></div>
      <div class="ml-margin-h">In the margin</div>
      <div class="ml-equation">f ′(x) = lim<sub style="font-style:normal">h→0</sub> [f(x+h) − f(x)] / h</div>
      <p>The slope of the line you'd see if you zoomed infinitely far into the curve at <em>x</em>. Up if positive, down if negative, flat at minima and maxima.</p>
      <p class="quiet">At a minimum, <em>f ′(x) = 0</em> — the tangent goes flat. Gradient descent searches for these flat spots.</p>
      <div class="ml-margin-rule thin"></div>
      <div class="ml-margin-h sm">Up next</div>
      <p class="quiet">Reverse the gradient and you have <em>gradient descent</em> — walk downhill, repeat.</p>`;
    return `
      <div class="ml-margin-rule"></div>
      <div class="ml-margin-h">In the margin</div>
      <div class="ml-equation">∇f = (∂f/∂x, ∂f/∂y)</div>
      <p>Two derivatives glued into a vector — one for each coordinate. The vector points <em>uphill, fastest</em>, and its length tells you how steep the climb is.</p>
      <p>At a minimum the gradient is zero — every direction is flat. That is the goal of training: drive ∇L to 0.</p>
      <p class="quiet">The gradient is always perpendicular to the level curves. This is why moving along a contour line keeps f the same.</p>
      <div class="ml-margin-rule thin"></div>
      <div class="ml-margin-h sm">Up next</div>
      <p class="quiet">Reverse the gradient and you have <em>gradient descent</em> — walk downhill, repeat.</p>`;
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

    // 1D plot is wider; 2D plot is square. Aspect ratio + canvas re-attach.
    if (state.mode === '1d') {
      canvasFrameEl.style.aspectRatio = '1.25 / 1';
      canvas.style.cursor = 'default';
      playControls.style.display = 'flex';
      const fn = FUNCS_1D[state.func1d];
      xSlider.min = fn.range[0]; xSlider.max = fn.range[1];
      xSlider.value = state.x;
      xLabel.textContent = `x = ${formatNum(state.x)}`;
    } else {
      canvasFrameEl.style.aspectRatio = '1 / 1';
      canvas.style.cursor = 'grab';
      playControls.style.display = 'none';
      stopPlay();
    }
    canvasCtl.redraw();
  }

  render();
})();
