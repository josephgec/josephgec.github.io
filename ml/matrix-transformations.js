/* Vol. I — Matrix Transformations */
(function () {
  'use strict';
  const V = window.MLViz;
  const { mul, formatNum, arrow, matEq, ACCENT, INK, INK_FADE, ACCENT_SOFT,
          RULE_FAINT, RULE_DARK, SERIF_LABEL, SERIF_LABEL_SM } = V;

  const PRESETS = [
    { name: 'Identity',   M: [[1, 0], [0, 1]],   blurb: 'The do-nothing transform. Every vector returns to itself.' },
    { name: 'Scale × 2',  M: [[2, 0], [0, 2]],   blurb: 'Uniform stretch. Areas grow by det(A) = 4.' },
    { name: 'Rotate 45°', M: [[Math.cos(Math.PI/4), -Math.sin(Math.PI/4)], [Math.sin(Math.PI/4), Math.cos(Math.PI/4)]], blurb: 'Pure rotation. Lengths preserved, det = 1.' },
    { name: 'Shear x',    M: [[1, 1], [0, 1]],   blurb: 'Slide horizontal lines proportionally to their height.' },
    { name: 'Reflect y',  M: [[-1, 0], [0, 1]],  blurb: 'Mirror across the y-axis. det = −1.' },
    { name: 'Squash',     M: [[1, 0], [0, 0.3]], blurb: 'Compress vertically.' },
    { name: 'Singular',   M: [[1, 2], [0.5, 1]], blurb: 'det = 0 — collapses the plane onto a line.' },
    { name: 'General',    M: [[1.4, 0.6], [-0.3, 1.1]], blurb: 'Rotation, stretch, and shear all at once.' },
  ];

  // Optional shapes overlaid on top of the (always-visible) unit-square parallelogram.
  const SHAPES = {
    none:     null,
    letterF:  [[0,0],[0,2],[1.2,2],[1.2,1.6],[0.4,1.6],[0.4,1.1],[1,1.1],[1,0.7],[0.4,0.7],[0.4,0]],
    triangle: [[0,0],[1.5,0],[0.75,1.3]],
  };
  const SHAPE_LABEL = { none: 'Just the unit square', letterF: '+ Letter F', triangle: '+ Triangle' };

  // The unit square's four corners, written as the convex combinations of î and ĵ.
  // Under the matrix Mt, this rectangle becomes the parallelogram (0, î′, î′+ĵ′, ĵ′).
  const UNIT_SQUARE = [[0,0],[1,0],[1,1],[0,1]];

  // ───────── State ─────────
  const state = {
    M: [[1.4, 0.6], [-0.3, 1.1]],
    t: 1,
    shape: 'none',
    playing: false,
    rafId: null,
    lastTs: 0,
  };

  // ───────── DOM ─────────
  const matrixEl = document.getElementById('matrix');
  const readoutEl = document.getElementById('readout');
  const presetsEl = document.getElementById('presets');
  const blurbEl = document.getElementById('preset-blurb');
  const tabsEl = document.getElementById('shape-tabs');
  const tSlider = document.getElementById('t-slider');
  const tLabel = document.getElementById('t-label');
  const replayBtn = document.getElementById('replay');
  const canvas = document.getElementById('canvas');

  // ───────── Matrix editor ─────────
  const matrixCtl = V.mountMatrixCells(matrixEl,
    () => state.M,
    (M) => { state.M = M; render(); }
  );

  // ───────── Presets ─────────
  PRESETS.forEach((p) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = p.name;
    b.addEventListener('click', () => {
      state.M = p.M.map((row) => row.slice());
      state.t = 0;
      startAnim();
      render();
    });
    presetsEl.appendChild(b);
  });

  // ───────── Shape tabs ─────────
  Object.keys(SHAPES).forEach((key) => {
    const b = document.createElement('button');
    b.textContent = SHAPE_LABEL[key];
    b.addEventListener('click', () => { state.shape = key; render(); });
    b.dataset.shape = key;
    tabsEl.appendChild(b);
  });

  // ───────── Slider + replay ─────────
  tSlider.addEventListener('input', (e) => {
    stopAnim();
    state.t = +e.target.value;
    render();
  });
  replayBtn.addEventListener('click', () => {
    state.t = 0;
    startAnim();
    render();
  });

  // ───────── Animation ─────────
  function startAnim() {
    state.playing = true;
    state.lastTs = performance.now();
    cancelAnimationFrame(state.rafId);
    const tick = (now) => {
      const dt = (now - state.lastTs) / 1000;
      state.lastTs = now;
      state.t = Math.min(1, state.t + dt * 0.6);
      render();
      if (state.t < 1 && state.playing) {
        state.rafId = requestAnimationFrame(tick);
      } else {
        state.playing = false;
      }
    };
    state.rafId = requestAnimationFrame(tick);
  }
  function stopAnim() {
    state.playing = false;
    cancelAnimationFrame(state.rafId);
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw);

  function draw(ctx, size) {
    const { M, t, shape } = state;
    const I = [[1,0],[0,1]];
    const Mt = [
      [I[0][0]*(1-t) + M[0][0]*t, I[0][1]*(1-t) + M[0][1]*t],
      [I[1][0]*(1-t) + M[1][0]*t, I[1][1]*(1-t) + M[1][1]*t],
    ];
    const scale = size.w / 8;
    const toPx = ([x, y]) => [size.w/2 + x*scale, size.h/2 - y*scale];

    // Faint original grid
    ctx.strokeStyle = RULE_FAINT;
    ctx.lineWidth = 1;
    for (let i = -4; i <= 4; i++) {
      ctx.beginPath();
      ctx.moveTo(...toPx([i, -4])); ctx.lineTo(...toPx([i, 4]));
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(...toPx([-4, i])); ctx.lineTo(...toPx([4, i]));
      ctx.stroke();
    }

    // Transformed grid
    ctx.strokeStyle = RULE_DARK;
    for (let i = -4; i <= 4; i++) {
      ctx.beginPath();
      for (let s = -4; s <= 4; s += 0.25) {
        const [px, py] = toPx(mul(Mt, [i, s]));
        if (s === -4) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.beginPath();
      for (let s = -4; s <= 4; s += 0.25) {
        const [px, py] = toPx(mul(Mt, [s, i]));
        if (s === -4) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    // ── Original unit square (faint, dashed) — what we started with ──
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    UNIT_SQUARE.forEach((p, i) => {
      const [px, py] = toPx(p);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.closePath();
    ctx.stroke();
    ctx.setLineDash([]);

    // ── Always-visible: the unit square morphs into the parallelogram (0, î′, î′+ĵ′, ĵ′). ──
    // Its area is exactly |det(A)| — the area scaling is the visual.
    const paraPts = UNIT_SQUARE.map((p) => mul(Mt, p));
    ctx.fillStyle = ACCENT_SOFT;
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1.75;
    ctx.beginPath();
    paraPts.forEach((p, i) => {
      const [px, py] = toPx(p);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // ── Optional decorative shape overlaid on top (Letter F / Triangle) ──
    const shapePts = SHAPES[shape];
    if (shapePts) {
      const pts = shapePts.map((p) => mul(Mt, p));
      ctx.fillStyle = 'rgba(38,35,32,0.08)';
      ctx.strokeStyle = INK;
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      pts.forEach((p, i) => {
        const [px, py] = toPx(p);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }

    // ── Original î and ĵ as ghost arrows (where they started) ──
    arrow(ctx, toPx([0,0]), toPx([1, 0]), 'rgba(38,35,32,0.28)', 1.5, 7);
    arrow(ctx, toPx([0,0]), toPx([0, 1]), 'rgba(122,31,36,0.28)', 1.5, 7);

    // ── Live î′ and ĵ′ (the columns of the matrix, made visible) ──
    const i_t = mul(Mt, [1, 0]);
    const j_t = mul(Mt, [0, 1]);
    arrow(ctx, toPx([0,0]), toPx(i_t), INK, 2.75, 11);
    arrow(ctx, toPx([0,0]), toPx(j_t), ACCENT, 2.75, 11);

    // Labels: î′ and ĵ′ at their tips
    ctx.font = SERIF_LABEL;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = INK;
    ctx.fillText('î′', ...toPx([i_t[0]*1.18, i_t[1]*1.18]));
    ctx.fillStyle = ACCENT;
    ctx.fillText('ĵ′', ...toPx([j_t[0]*1.18, j_t[1]*1.18]));

    // Original-position labels (small, faint)
    ctx.font = SERIF_LABEL_SM;
    ctx.fillStyle = 'rgba(38,35,32,0.45)';
    ctx.fillText('î', ...toPx([1.12, -0.08]));
    ctx.fillStyle = 'rgba(122,31,36,0.45)';
    ctx.fillText('ĵ', ...toPx([-0.08, 1.12]));

    // Determinant footer (with explicit unit-square framing)
    const det = M[0][0]*M[1][1] - M[0][1]*M[1][0];
    ctx.fillStyle = INK_FADE;
    ctx.font = SERIF_LABEL_SM;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(`unit square area: 1 → ${formatNum(Math.abs(det))}`, 18, size.h - 22);
    ctx.textAlign = 'right';
    ctx.fillText(`det = ${formatNum(det)}${det < 0 ? '  (orientation flipped)' : ''}`,
      size.w - 18, size.h - 22);
  }

  // ───────── Render (DOM updates + canvas redraw) ─────────
  function render() {
    matrixCtl.refresh();
    const M = state.M;
    const tr = M[0][0] + M[1][1];
    const det = M[0][0]*M[1][1] - M[0][1]*M[1][0];
    readoutEl.querySelector('[data-k="tr"]').textContent = formatNum(tr);
    readoutEl.querySelector('[data-k="det"]').textContent = formatNum(det);
    readoutEl.querySelector('[data-k="iHat"]').textContent = `(${formatNum(M[0][0])}, ${formatNum(M[1][0])})`;
    readoutEl.querySelector('[data-k="jHat"]').textContent = `(${formatNum(M[0][1])}, ${formatNum(M[1][1])})`;

    // Presets active state
    const matched = PRESETS.find((p) => matEq(p.M, M));
    Array.from(presetsEl.children).forEach((btn, i) => {
      btn.classList.toggle('active', PRESETS[i] === matched);
    });
    blurbEl.textContent = matched ? matched.blurb : 'Custom matrix.';

    // Shape tab active
    Array.from(tabsEl.children).forEach((btn) => {
      btn.classList.toggle('on', btn.dataset.shape === state.shape);
    });

    // Slider
    tSlider.value = state.t;
    tLabel.textContent = `t = ${state.t.toFixed(2)}`;

    canvasCtl.redraw();
  }

  render();
})();
