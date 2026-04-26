/* Vol. V — Singular Value Decomposition */
(function () {
  'use strict';
  const V = window.MLViz;
  const { mul, matmul, transpose, svd2, formatNum, arrow, matEq,
          ACCENT, INK, INK_FADE, RULE_FAINT, RULE_DARK,
          ACCENT_FILL, SERIF_LABEL, SERIF_LABEL_SM } = V;

  const PRESETS = [
    { name: 'Slanted',     M: [[1.6, 0.4], [0.6, 1.2]],  blurb: 'A rotation, a stretch, another rotation.' },
    { name: 'Strong axis', M: [[2, 0.5], [0.4, 0.6]],    blurb: 'One singular value dominates — rank-1 approximation will be excellent.' },
    { name: 'Singular',    M: [[1, 2], [0.5, 1]],        blurb: 'σ₂ = 0. The matrix has rank 1 — it crushes the plane to a line.' },
    { name: 'Symmetric',   M: [[2, 0.8], [0.8, 1.2]],    blurb: 'When A is symmetric, U = V (up to sign). SVD coincides with eigendecomposition.' },
    { name: 'Rotation',    M: [[Math.cos(0.7), -Math.sin(0.7)], [Math.sin(0.7), Math.cos(0.7)]], blurb: 'Pure rotation: σ₁ = σ₂ = 1. The unit circle stays a unit circle.' },
    { name: 'Reflection',  M: [[1, 0], [0, -1]],          blurb: 'U or V will carry a negative determinant — the orientation flips.' },
  ];

  const STAGE_NAMES = ['start', 'after Vᵀ (rotate)', 'after Σ Vᵀ (rotate + stretch)', 'after U Σ Vᵀ = A'];
  const TAB_LABELS  = ['Start', 'Vᵀ', 'Σ Vᵀ', 'U Σ Vᵀ'];

  const state = {
    M: [[1.6, 0.4], [0.6, 1.2]],
    stage: (new URLSearchParams(location.search).get('stage') | 0) || 0,
    playing: false,
    timer: null,
  };

  // ───────── DOM ─────────
  const matrixEl = document.getElementById('matrix');
  const readoutEl = document.getElementById('readout');
  const presetsEl = document.getElementById('presets');
  const blurbEl = document.getElementById('preset-blurb');
  const tabsEl = document.getElementById('stage-tabs');
  const canvas = document.getElementById('canvas');
  const replayBtn = document.getElementById('replay');
  const stageSlider = document.getElementById('stage-slider');
  const stageLabel = document.getElementById('stage-label');

  // ───────── Matrix editor ─────────
  const matrixCtl = V.mountMatrixCells(matrixEl,
    () => state.M,
    (M) => { state.M = M; render(); }
  );

  // ───────── Stage tabs ─────────
  TAB_LABELS.forEach((label, i) => {
    const b = document.createElement('button');
    b.textContent = label;
    b.dataset.stage = i;
    b.addEventListener('click', () => {
      stopAuto();
      state.stage = i;
      render();
    });
    tabsEl.appendChild(b);
  });

  // ───────── Presets ─────────
  PRESETS.forEach((p) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = p.name;
    b.addEventListener('click', () => {
      state.M = p.M.map((row) => row.slice());
      state.stage = 0;
      startAuto();
      render();
    });
    presetsEl.appendChild(b);
  });

  // ───────── Slider + replay ─────────
  stageSlider.addEventListener('input', (e) => {
    stopAuto();
    state.stage = +e.target.value;
    render();
  });
  replayBtn.addEventListener('click', () => {
    state.stage = 0;
    startAuto();
    render();
  });

  // ───────── Auto-advance ─────────
  function startAuto() {
    state.playing = true;
    if (state.timer) clearInterval(state.timer);
    state.timer = setInterval(() => {
      if (state.stage >= 3) {
        stopAuto();
        return;
      }
      state.stage += 1;
      render();
    }, 1100);
  }
  function stopAuto() {
    state.playing = false;
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw);

  function draw(ctx, size) {
    const A = state.M;
    const svd = svd2(A);
    if (!svd) return;

    const scale = size.w / 8;
    const toPx = ([x, y]) => [size.w/2 + x*scale, size.h/2 - y*scale];

    const { U, S, V: Vmat } = svd;
    const Vt = transpose(Vmat);
    const Sigma = [[S[0], 0], [0, S[1]]];
    const stage = state.stage;

    let M;
    if (stage === 0) M = [[1,0],[0,1]];
    else if (stage === 1) M = Vt;
    else if (stage === 2) M = matmul(Sigma, Vt);
    else M = A;

    // Faint original grid
    ctx.strokeStyle = RULE_FAINT;
    ctx.lineWidth = 1;
    for (let i = -4; i <= 4; i++) {
      ctx.beginPath(); ctx.moveTo(...toPx([i, -4])); ctx.lineTo(...toPx([i, 4])); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(...toPx([-4, i])); ctx.lineTo(...toPx([4, i])); ctx.stroke();
    }

    // Transformed grid
    ctx.strokeStyle = RULE_DARK;
    for (let i = -3; i <= 3; i++) {
      ctx.beginPath();
      for (let s = -3; s <= 3; s += 0.25) {
        const [px, py] = toPx(mul(M, [i, s]));
        if (s === -3) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.beginPath();
      for (let s = -3; s <= 3; s += 0.25) {
        const [px, py] = toPx(mul(M, [s, i]));
        if (s === -3) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    // Unit circle → ellipse
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 2;
    ctx.fillStyle = ACCENT_FILL;
    ctx.beginPath();
    for (let t = 0; t <= Math.PI*2 + 0.01; t += 0.02) {
      const [px, py] = toPx(mul(M, [Math.cos(t), Math.sin(t)]));
      if (t === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.fill();
    ctx.stroke();

    // v₁, v₂: V's columns. These are the input directions where A acts most simply.
    const v1 = [Vmat[0][0], Vmat[1][0]];
    const v2 = [Vmat[0][1], Vmat[1][1]];
    const Mv1 = mul(M, v1);
    const Mv2 = mul(M, v2);

    // ── Stage 3: draw σ₁ and σ₂ as the ellipse's semi-axes (dashed, with labels)
    //    so the singular values become visible as actual lengths. ──
    if (stage === 3) {
      const u1 = [U[0][0], U[1][0]];
      const u2 = [U[0][1], U[1][1]];
      ctx.strokeStyle = 'rgba(122,31,36,0.55)';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 4]);
      // σ₁·u₁
      ctx.beginPath();
      ctx.moveTo(...toPx([0, 0]));
      ctx.lineTo(...toPx([u1[0]*S[0], u1[1]*S[0]]));
      ctx.stroke();
      // σ₂·u₂
      ctx.beginPath();
      ctx.moveTo(...toPx([0, 0]));
      ctx.lineTo(...toPx([u2[0]*S[1], u2[1]*S[1]]));
      ctx.stroke();
      ctx.setLineDash([]);

      // σ value labels at the midpoints of each semi-axis
      ctx.fillStyle = ACCENT;
      ctx.font = SERIF_LABEL_SM;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const off1 = 0.18; // perpendicular offset so the label doesn't sit on the line
      const off2 = 0.18;
      ctx.fillText(`σ₁ = ${formatNum(S[0])}`,
        ...toPx([u1[0]*S[0]*0.55 + (-u1[1])*off1, u1[1]*S[0]*0.55 + u1[0]*off1]));
      ctx.fillStyle = 'rgba(122,31,36,0.7)';
      ctx.fillText(`σ₂ = ${formatNum(S[1])}`,
        ...toPx([u2[0]*S[1]*0.55 + (-u2[1])*off2, u2[1]*S[1]*0.55 + u2[0]*off2]));
    }

    // v-arrows (image of v₁, v₂ under the current stage's matrix)
    arrow(ctx, toPx([0,0]), toPx(Mv1), INK, 2.5, 10);
    arrow(ctx, toPx([0,0]), toPx(Mv2), 'rgba(38,35,32,0.55)', 2, 8);

    // ── Stage-aware labels for the two arrows ──
    ctx.font = SERIF_LABEL_SM;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const labelsByStage = [
      ['v₁',          'v₂'],            // stage 0: original input directions
      ['Vᵀv₁ = e₁',   'Vᵀv₂ = e₂'],      // stage 1: rotated onto the standard axes
      ['σ₁e₁',        'σ₂e₂'],           // stage 2: stretched along the standard axes
      ['σ₁u₁',        'σ₂u₂'],           // stage 3: rotated by U into the output
    ];
    const [lab1, lab2] = labelsByStage[stage];
    ctx.fillStyle = INK;
    ctx.fillText(lab1, ...toPx([Mv1[0]*1.18, Mv1[1]*1.18]));
    ctx.fillStyle = 'rgba(38,35,32,0.6)';
    ctx.fillText(lab2, ...toPx([Mv2[0]*1.22, Mv2[1]*1.22]));

    // Footer
    ctx.fillStyle = INK_FADE;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.font = SERIF_LABEL_SM;
    ctx.fillText(STAGE_NAMES[stage], 18, size.h - 22);
    ctx.textAlign = 'right';
    ctx.fillText(`σ₁ = ${formatNum(S[0])}, σ₂ = ${formatNum(S[1])}`,
      size.w - 18, size.h - 22);
  }

  // ───────── Render ─────────
  function render() {
    matrixCtl.refresh();
    const A = state.M;
    const svd = svd2(A);
    const det = A[0][0]*A[1][1] - A[0][1]*A[1][0];
    if (svd) {
      readoutEl.querySelector('[data-k="s1"]').textContent = formatNum(svd.S[0]);
      readoutEl.querySelector('[data-k="s2"]').textContent = formatNum(svd.S[1]);
      readoutEl.querySelector('[data-k="cond"]').textContent =
        svd.S[1] > 1e-6 ? formatNum(svd.S[0]/svd.S[1]) : '∞';
      readoutEl.querySelector('[data-k="rank"]').textContent = svd.S[1] < 1e-3 ? '1' : '2';
      readoutEl.querySelector('[data-k="det"]').textContent = formatNum(det);
    }

    const matched = PRESETS.find((p) => matEq(p.M, A));
    Array.from(presetsEl.children).forEach((btn, i) => {
      btn.classList.toggle('active', PRESETS[i] === matched);
    });
    blurbEl.textContent = matched ? matched.blurb : 'Custom matrix.';

    Array.from(tabsEl.children).forEach((btn) => {
      btn.classList.toggle('on', +btn.dataset.stage === state.stage);
    });

    stageSlider.value = state.stage;
    stageLabel.textContent = `step ${state.stage}/3`;

    canvasCtl.redraw();
  }

  render();
})();
