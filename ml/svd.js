/* Vol. V — Singular Value Decomposition */
(function () {
  'use strict';
  const V = window.MLViz;
  const { mul, matmul, transpose, svd2, formatNum, arrow, matEq,
          ACCENT, INK, INK_FADE, RULE_FAINT, RULE_DARK,
          ACCENT_FILL, ACCENT_FAINT, ACCENT_SOFT,
          SERIF_LABEL, SERIF_LABEL_SM } = V;

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
    mode: 'choreography', // 'choreography' | 'lowrank'
    stage: (new URLSearchParams(location.search).get('stage') | 0) || 0,
    k: 1,
    playing: false,
    timer: null,
  };

  // ───────── DOM ─────────
  const matrixEl = document.getElementById('matrix');
  const readoutEl = document.getElementById('readout');
  const presetsEl = document.getElementById('presets');
  const blurbEl = document.getElementById('preset-blurb');
  const modeTabsEl = document.getElementById('mode-tabs');
  const stageTabsEl = document.getElementById('stage-tabs');
  const captionEl = document.getElementById('caption');
  const marginEl = document.getElementById('margin');
  const canvas = document.getElementById('canvas');
  const replayBtn = document.getElementById('replay');
  const stageControlsEl = document.getElementById('stage-controls');
  const stageSlider = document.getElementById('stage-slider');
  const stageLabel = document.getElementById('stage-label');
  const rankControlsEl = document.getElementById('rank-controls');
  const rankSlider = document.getElementById('rank-slider');
  const rankLabel = document.getElementById('rank-label');
  const residualLabel = document.getElementById('residual-label');

  // ───────── Matrix editor ─────────
  const matrixCtl = V.mountMatrixCells(matrixEl,
    () => state.M,
    (M) => { state.M = M; render(); }
  );

  // ───────── Mode tabs ─────────
  Array.from(modeTabsEl.children).forEach((btn) => {
    btn.addEventListener('click', () => {
      stopAuto();
      state.mode = btn.dataset.mode;
      // Reset stage-specific state when switching.
      if (state.mode === 'choreography') {
        state.stage = 0;
        startAuto();
      }
      render();
    });
  });

  // ───────── Stage tabs (choreography) ─────────
  TAB_LABELS.forEach((label, i) => {
    const b = document.createElement('button');
    b.textContent = label;
    b.dataset.stage = i;
    b.addEventListener('click', () => {
      stopAuto();
      state.stage = i;
      render();
    });
    stageTabsEl.appendChild(b);
  });

  // ───────── Presets ─────────
  PRESETS.forEach((p) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = p.name;
    b.addEventListener('click', () => {
      state.M = p.M.map((row) => row.slice());
      if (state.mode === 'choreography') {
        state.stage = 0;
        startAuto();
      }
      render();
    });
    presetsEl.appendChild(b);
  });

  // ───────── Stage slider + replay ─────────
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

  // ───────── Rank slider ─────────
  rankSlider.addEventListener('input', (e) => {
    state.k = +e.target.value;
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
    if (state.mode === 'choreography') drawChoreography(ctx, size);
    else drawLowRank(ctx, size);
  }

  // ───────── View I: 4-stage choreography ─────────
  function drawChoreography(ctx, size) {
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

    drawGrids(ctx, size, toPx, M);
    drawEllipse(ctx, toPx, M, ACCENT, ACCENT_FILL);

    const v1 = [Vmat[0][0], Vmat[1][0]];
    const v2 = [Vmat[0][1], Vmat[1][1]];
    const Mv1 = mul(M, v1);
    const Mv2 = mul(M, v2);

    // ── Stage 3: σ₁ and σ₂ as ruler-style measurements on dashed semi-axes ──
    if (stage === 3) {
      const u1 = [U[0][0], U[1][0]];
      const u2 = [U[0][1], U[1][1]];
      drawRulerSemiAxis(ctx, toPx, u1, S[0], 'σ₁', ACCENT);
      drawRulerSemiAxis(ctx, toPx, u2, S[1], 'σ₂', 'rgba(122,31,36,0.7)');
    }

    arrow(ctx, toPx([0,0]), toPx(Mv1), INK, 2.5, 10);
    arrow(ctx, toPx([0,0]), toPx(Mv2), 'rgba(38,35,32,0.55)', 2, 8);

    // Stage-aware labels
    ctx.font = SERIF_LABEL_SM;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const labelsByStage = [
      ['v₁',          'v₂'],
      ['Vᵀv₁ = e₁',   'Vᵀv₂ = e₂'],
      ['σ₁e₁',        'σ₂e₂'],
      ['σ₁u₁',        'σ₂u₂'],
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

  // ── Ruler-style semi-axis: dashed segment with two short tick marks at the
  //    ends, σ value pinned at the midpoint perpendicular off-axis.
  function drawRulerSemiAxis(ctx, toPx, u, sigma, label, color) {
    if (sigma < 1e-3) {
      // Degenerate: just put the label at origin
      ctx.fillStyle = color;
      ctx.font = SERIF_LABEL_SM;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${label} = ${formatNum(sigma)}`, ...toPx([u[0]*0.3, u[1]*0.3]));
      return;
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(...toPx([0, 0]));
    ctx.lineTo(...toPx([u[0]*sigma, u[1]*sigma]));
    ctx.stroke();
    ctx.setLineDash([]);

    // Tick marks at the two ends
    const perp = [-u[1], u[0]];
    const tlen = 0.12;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    // origin tick
    ctx.moveTo(...toPx([perp[0]*tlen, perp[1]*tlen]));
    ctx.lineTo(...toPx([-perp[0]*tlen, -perp[1]*tlen]));
    // end tick
    ctx.moveTo(...toPx([u[0]*sigma + perp[0]*tlen, u[1]*sigma + perp[1]*tlen]));
    ctx.lineTo(...toPx([u[0]*sigma - perp[0]*tlen, u[1]*sigma - perp[1]*tlen]));
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.font = SERIF_LABEL_SM;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const off = 0.30;
    ctx.fillText(`${label} = ${formatNum(sigma)}`,
      ...toPx([u[0]*sigma*0.5 + perp[0]*off, u[1]*sigma*0.5 + perp[1]*off]));
  }

  // ───────── View II: low-rank approximation (Eckart–Young) ─────────
  function drawLowRank(ctx, size) {
    const A = state.M;
    const svd = svd2(A);
    if (!svd) return;
    const { U, S, V: Vmat } = svd;
    const k = state.k;

    const scale = size.w / 8;
    const toPx = ([x, y]) => [size.w/2 + x*scale, size.h/2 - y*scale];

    // A_k = sum over i<k of σ_i u_i v_i^T
    // For 2x2: A_1 = σ₁ u₁ v₁ᵀ ; A_2 = full A.
    const u1 = [U[0][0], U[1][0]];
    const u2 = [U[0][1], U[1][1]];
    const v1 = [Vmat[0][0], Vmat[1][0]];
    const v2 = [Vmat[0][1], Vmat[1][1]];
    const outer = (u, v) => [[u[0]*v[0], u[0]*v[1]], [u[1]*v[0], u[1]*v[1]]];
    const scaleM = (M, s) => [[M[0][0]*s, M[0][1]*s], [M[1][0]*s, M[1][1]*s]];
    const addM = (A, B) => [[A[0][0]+B[0][0], A[0][1]+B[0][1]], [A[1][0]+B[1][0], A[1][1]+B[1][1]]];
    const subM = (A, B) => [[A[0][0]-B[0][0], A[0][1]-B[0][1]], [A[1][0]-B[1][0], A[1][1]-B[1][1]]];

    const Ak1 = scaleM(outer(u1, v1), S[0]);
    const Ak  = (k === 1) ? Ak1 : addM(Ak1, scaleM(outer(u2, v2), S[1]));
    const E   = subM(A, Ak); // residual matrix
    // Frobenius norm of residual (= σ_{k+1}, by Eckart–Young)
    const eFro = Math.sqrt(E[0][0]*E[0][0] + E[0][1]*E[0][1] + E[1][0]*E[1][0] + E[1][1]*E[1][1]);

    // Faint original grid
    drawGrids(ctx, size, toPx, Ak);

    // Faint outline of true A's image (so the residual is visible as a gap).
    ctx.strokeStyle = 'rgba(38,35,32,0.32)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    for (let t = 0; t <= Math.PI*2 + 0.01; t += 0.02) {
      const [px, py] = toPx(mul(A, [Math.cos(t), Math.sin(t)]));
      if (t === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Approximate image (solid).
    drawEllipse(ctx, toPx, Ak, ACCENT, ACCENT_FILL);

    // ── Residual lines: for ~24 sample directions on the unit circle, draw a
    //    thin line from A·x to A_k·x — these are exactly the columns of (A − A_k)
    //    times each direction, the lost information made visible. ──
    const N = 24;
    ctx.strokeStyle = 'rgba(38,35,32,0.25)';
    ctx.lineWidth = 1;
    for (let i = 0; i < N; i++) {
      const t = (i / N) * Math.PI * 2;
      const x = [Math.cos(t), Math.sin(t)];
      const Ax = mul(A, x);
      const Akx = mul(Ak, x);
      ctx.beginPath();
      ctx.moveTo(...toPx(Akx));
      ctx.lineTo(...toPx(Ax));
      ctx.stroke();
    }

    // ── σ₁ u₁ as the principal axis of the rank-1 ellipse (always shown). ──
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(...toPx([-u1[0]*S[0], -u1[1]*S[0]]));
    ctx.lineTo(...toPx([u1[0]*S[0], u1[1]*S[0]]));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = ACCENT;
    ctx.font = SERIF_LABEL_SM;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    {
      const perp = [-u1[1], u1[0]];
      ctx.fillText(`σ₁ u₁ = ${formatNum(S[0])}`,
        ...toPx([u1[0]*S[0]*0.55 + perp[0]*0.28, u1[1]*S[0]*0.55 + perp[1]*0.28]));
    }
    if (k === 2) {
      // Show σ₂ u₂ as the minor semi-axis when full rank.
      ctx.strokeStyle = 'rgba(122,31,36,0.55)';
      ctx.lineWidth = 1.2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(...toPx([-u2[0]*S[1], -u2[1]*S[1]]));
      ctx.lineTo(...toPx([u2[0]*S[1], u2[1]*S[1]]));
      ctx.stroke();
      ctx.setLineDash([]);
      const perp2 = [-u2[1], u2[0]];
      ctx.fillStyle = 'rgba(122,31,36,0.7)';
      ctx.fillText(`σ₂ u₂ = ${formatNum(S[1])}`,
        ...toPx([u2[0]*S[1]*0.55 + perp2[0]*0.28, u2[1]*S[1]*0.55 + perp2[1]*0.28]));
    }

    // Equation banner.
    ctx.fillStyle = INK;
    ctx.font = 'italic 15px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    if (k === 1) {
      ctx.fillText('A₁ = σ₁ u₁ v₁ᵀ', size.w/2, 14);
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.fillStyle = INK_FADE;
      ctx.fillText('rank 1 — the ellipse degenerates to a line segment', size.w/2, 36);
    } else {
      ctx.fillText('A₂ = σ₁ u₁ v₁ᵀ + σ₂ u₂ v₂ᵀ = A', size.w/2, 14);
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.fillStyle = INK_FADE;
      ctx.fillText('full rank — the approximation is exact', size.w/2, 36);
    }

    // ── Eckart–Young readouts (footer) ──
    ctx.fillStyle = INK_FADE;
    ctx.font = SERIF_LABEL_SM;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(`k = ${k}   ·   keeping ${k} of 2 components`, 18, size.h - 22);
    ctx.textAlign = 'right';
    ctx.fillText(`‖A − A${sub(k)}‖_F = ${formatNum(eFro)}   (= σ${sub(k+1)} when k<2)`,
      size.w - 18, size.h - 22);

    // ── Top-left legend: solid = approximation, dashed = true A, lines = lost. ──
    const legX = 14;
    const legY = 60;
    const legW = 196;
    const legH = 76;
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
    // solid orange = A_k image
    ctx.fillStyle = ACCENT_FILL;
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.ellipse(legX + 24, legY + 30, 14, 5, 0, 0, Math.PI*2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = ACCENT;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.fillText(`A${sub(k)}  approximation`, legX + 48, legY + 33);
    // dashed dark = true A image
    ctx.strokeStyle = 'rgba(38,35,32,0.55)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.ellipse(legX + 24, legY + 50, 14, 7, 0, 0, Math.PI*2); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = INK_FADE;
    ctx.fillText('A   true image (target)', legX + 48, legY + 53);
    // residual swatch
    ctx.strokeStyle = 'rgba(38,35,32,0.45)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(legX + 14, legY + 68); ctx.lineTo(legX + 36, legY + 68);
    ctx.stroke();
    ctx.fillStyle = INK_FADE;
    ctx.fillText('residual  (what was lost)', legX + 48, legY + 71);
  }

  // Subscript helper (we need it in this file too).
  const SUB_DIGITS = '₀₁₂₃₄₅₆₇₈₉';
  function sub(n) { return String(n).replace(/[0-9]/g, (d) => SUB_DIGITS[+d]); }

  // ───────── Common drawing helpers ─────────
  function drawGrids(ctx, size, toPx, M) {
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
  }

  function drawEllipse(ctx, toPx, M, stroke, fill) {
    ctx.strokeStyle = stroke;
    ctx.fillStyle = fill;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let t = 0; t <= Math.PI*2 + 0.01; t += 0.02) {
      const [px, py] = toPx(mul(M, [Math.cos(t), Math.sin(t)]));
      if (t === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.fill();
    ctx.stroke();
  }

  // ───────── Caption + marginalia ─────────
  function captionFor(mode, stage, S) {
    if (mode === 'choreography') {
      const stageCap = [
        `<span class="ml-cap-num">·</span> The starting unit circle. The dark <em>v₁</em> arrow marks the input direction A acts most simply on — the one that will become the long axis after the dance.`,
        `<span class="ml-cap-num">·</span> <em>Vᵀ</em> rotates the plane so v₁ lands on the x-axis. (<em>V</em> is a rotation matrix; <em>Vᵀ</em> is its transpose, which here is also its inverse.) The unit circle is unchanged — rotations don't stretch.`,
        `<span class="ml-cap-num">·</span> <em>Σ</em> stretches by σ₁ along x and σ₂ along y. Now the circle is an axis-aligned ellipse with semi-axes σ₁ and σ₂.`,
        `<span class="ml-cap-num">·</span> <em>U</em> rotates the ellipse to its final orientation: <em>σ₁u₁</em> and <em>σ₂u₂</em> are the dashed semi-axes. Read their lengths off as ruler measurements — those are the singular values.`,
      ];
      return stageCap[stage];
    }
    return `<span class="ml-cap-num">·</span> Set <em>k = 1</em>: the rank-1 approximation <em>A₁ = σ₁ u₁ v₁ᵀ</em> crushes the unit circle to a line segment along <em>u₁</em>. The dashed outline is the true A; the grey lines from approximate to true are residuals. By Eckart–Young, <em>‖A − A₁‖_F = σ₂</em> — no rank-1 matrix gets closer.`;
  }

  function marginFor(mode, S) {
    if (mode === 'choreography') {
      return `
        <div class="ml-margin-rule"></div>
        <div class="ml-margin-h">In the margin</div>
        <div class="ml-equation">A = U Σ Vᵀ</div>
        <p class="quiet" style="margin-top:-8px"><em>U</em> and <em>V</em> are <em>rotation matrices</em> — orthonormal, length-preserving. <em>Σ</em> ("<em>capital sigma</em>") is a diagonal matrix of <em>singular values</em> σ₁, σ₂ (lowercase sigma) — the stretch factors. <em>Vᵀ</em> is "V transpose"; for a rotation it's the inverse rotation. So SVD says: every linear map = rotate, stretch along axes, rotate again.</p>
        <p>Watch the canvas. <em>V</em> finds the input directions where A acts most simply. <em>Σ</em> stretches them. <em>U</em> says where they land.</p>
        <p class="quiet">Unlike eigen-decomposition, SVD works for <em>any</em> matrix — non-square, non-symmetric, even singular. It is the most general structure theorem in linear algebra.</p>

        <div class="ml-margin-rule thin"></div>
        <div class="ml-margin-h sm">Condition number</div>
        <p class="quiet"><em>σ₁ / σ₂</em> — the ratio of biggest stretch to smallest. High = the matrix amplifies along one axis far more than another, so small input errors blow up. Bad for numerical stability. A condition number of 1 means perfect; ∞ means the matrix is singular.</p>

        <div class="ml-margin-rule thin"></div>
        <div class="ml-margin-h sm">Connections</div>
        <p class="quiet">When A is symmetric, U = V and the σ's are the absolute values of the eigenvalues. The covariance matrix you saw in PCA is symmetric, so its SVD <em>is</em> its eigendecomposition.</p>`;
    }
    return `
      <div class="ml-margin-rule"></div>
      <div class="ml-margin-h">In the margin</div>
      <div class="ml-equation">A = σ₁ u₁ v₁ᵀ + σ₂ u₂ v₂ᵀ</div>
      <p class="quiet" style="margin-top:-8px">Each term <em>σᵢ uᵢ vᵢᵀ</em> is a rank-1 matrix — a single direction in input mapping to a single direction in output. SVD writes A as a sum of these, ordered by importance (σ's descending).</p>
      <p>Truncate the sum at <em>k</em> terms: <em>Aₖ = Σᵢ₌₁..ₖ σᵢ uᵢ vᵢᵀ</em>. By the <em>Eckart–Young theorem</em>, no other rank-<em>k</em> matrix is closer to A in Frobenius (or spectral) norm. The error is exactly <em>σₖ₊₁</em> in spectral norm — what you threw away.</p>
      <p class="quiet">In 2D this is binary: k=1 collapses everything to a line; k=2 is exact. Real applications run in 1000s of dimensions, where keeping the top 50 singular values reconstructs nearly perfectly.</p>

      <div class="ml-margin-rule thin"></div>
      <div class="ml-margin-h sm">Where it lives in ML</div>
      <p class="quiet">Image compression, latent semantic analysis, recommender-system factorizations, LoRA fine-tuning of LLMs — all the same idea: keep only the top-<em>k</em> components and you have a compact summary that loses almost nothing.</p>`;
  }

  // ───────── Initial auto-play ─────────
  // Land on stage 0, then dance through the four steps automatically so the
  // visitor sees what SVD *does* before reading anything.
  if (state.mode === 'choreography' && state.stage === 0) {
    setTimeout(() => { if (!state.playing && state.stage === 0) startAuto(); }, 600);
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

    // Mode tabs
    Array.from(modeTabsEl.children).forEach((btn) => {
      btn.classList.toggle('on', btn.dataset.mode === state.mode);
    });

    // Choreography-specific UI
    if (state.mode === 'choreography') {
      stageTabsEl.style.display = '';
      stageControlsEl.style.display = '';
      rankControlsEl.style.display = 'none';
      Array.from(stageTabsEl.children).forEach((btn) => {
        btn.classList.toggle('on', +btn.dataset.stage === state.stage);
      });
      stageSlider.value = state.stage;
      stageLabel.textContent = `step ${state.stage}/3`;
    } else {
      stageTabsEl.style.display = 'none';
      stageControlsEl.style.display = 'none';
      rankControlsEl.style.display = '';
      rankSlider.value = state.k;
      rankLabel.textContent = `k = ${state.k}`;
      if (svd) {
        // residual Frobenius norm
        const u1 = [svd.U[0][0], svd.U[1][0]];
        const u2 = [svd.U[0][1], svd.U[1][1]];
        const v1 = [svd.V[0][0], svd.V[1][0]];
        const v2 = [svd.V[0][1], svd.V[1][1]];
        const outer = (u, v) => [[u[0]*v[0], u[0]*v[1]], [u[1]*v[0], u[1]*v[1]]];
        const Ak1 = outer(u1, v1).map((row) => row.map((x) => x*svd.S[0]));
        let Ak;
        if (state.k === 1) Ak = Ak1;
        else {
          const Ak2 = outer(u2, v2).map((row) => row.map((x) => x*svd.S[1]));
          Ak = [[Ak1[0][0]+Ak2[0][0], Ak1[0][1]+Ak2[0][1]], [Ak1[1][0]+Ak2[1][0], Ak1[1][1]+Ak2[1][1]]];
        }
        const E = [[A[0][0]-Ak[0][0], A[0][1]-Ak[0][1]], [A[1][0]-Ak[1][0], A[1][1]-Ak[1][1]]];
        const eFro = Math.sqrt(E[0][0]*E[0][0] + E[0][1]*E[0][1] + E[1][0]*E[1][0] + E[1][1]*E[1][1]);
        residualLabel.textContent = `‖A − A_k‖ = ${formatNum(eFro)}`;
      }
    }

    captionEl.innerHTML = captionFor(state.mode, state.stage, svd ? svd.S : [0,0]);
    marginEl.innerHTML = marginFor(state.mode, svd ? svd.S : [0,0]);

    canvasCtl.redraw();
  }

  render();
})();
