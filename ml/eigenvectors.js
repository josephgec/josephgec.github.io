/* Vol. II — Eigenvalues & Eigenvectors */
(function () {
  'use strict';
  const V = window.MLViz;
  const { mul, normalize, norm, eigen2, formatNum, sub, arrow, matEq,
          ACCENT, INK, INK_FADE, RULE_DARK, RULE_FAINT, RULE_AXIS,
          ACCENT_SOFT, ACCENT_FAINT,
          SERIF_LABEL, SERIF_LABEL_SM, MONO_LABEL } = V;

  const PRESETS = [
    { name: 'Identity',     M: [[1,0],[0,1]],     blurb: 'Nothing moves. Every vector is an eigenvector with λ = 1.' },
    { name: 'Scale',        M: [[2,0],[0,0.5]],   blurb: 'Pure stretch along the axes — they are the eigenvectors.' },
    { name: 'Shear',        M: [[1,1],[0,1]],     blurb: 'A repeated eigenvalue λ = 1 with only one eigenvector direction.' },
    { name: 'Rotation',     M: [[0,-1],[1,0]],    blurb: '90° rotation. No real eigenvectors — every direction changes.' },
    { name: 'Reflection',   M: [[1,0],[0,-1]],    blurb: 'Flip across x. Eigenvalues +1 and −1 along the axes.' },
    { name: 'Symmetric',    M: [[2,1],[1,2]],     blurb: 'Symmetric matrices have orthogonal eigenvectors.' },
    { name: 'PageRank-ish', M: [[1.2,0.4],[0.3,0.9]], blurb: 'Iterate this and watch every vector collapse to the dominant direction.' },
    { name: 'Squeeze',      M: [[1.5,0],[0,1/1.5]], blurb: 'Stretch one axis, squish the other — area preserved.' },
  ];

  const state = {
    M: [[1.2, 0.4], [0.3, 0.9]],
    mode: 'explore',
    draggedVec: [Math.cos(0.6), Math.sin(0.6)],
    iter: 0,
    iterPlaying: false,
    iterTimer: null,
  };

  // ───────── DOM ─────────
  const matrixEl = document.getElementById('matrix');
  const readoutEl = document.getElementById('readout');
  const presetsEl = document.getElementById('presets');
  const blurbEl = document.getElementById('preset-blurb');
  const tabsEl = document.getElementById('mode-tabs');
  const captionEl = document.getElementById('caption');
  const marginEl = document.getElementById('margin');
  const iterEl = document.getElementById('iter-controls');
  const playBtn = document.getElementById('play-btn');
  const iterSlider = document.getElementById('iter-slider');
  const iterLabel = document.getElementById('iter-label');
  const canvas = document.getElementById('canvas');

  // ───────── Matrix editor ─────────
  const matrixCtl = V.mountMatrixCells(matrixEl,
    () => state.M,
    (M) => { state.M = M; resetIter(); render(); }
  );

  // ───────── Presets ─────────
  PRESETS.forEach((p) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = p.name;
    b.addEventListener('click', () => {
      state.M = p.M.map((row) => row.slice());
      resetIter();
      render();
    });
    presetsEl.appendChild(b);
  });

  // ───────── Mode tabs ─────────
  Array.from(tabsEl.children).forEach((btn) => {
    btn.addEventListener('click', () => {
      state.mode = btn.dataset.mode;
      resetIter();
      render();
    });
  });

  // ───────── Iterate controls ─────────
  playBtn.addEventListener('click', () => {
    if (state.iterPlaying) stopIter(); else startIter();
  });
  iterSlider.addEventListener('input', (e) => {
    stopIter();
    state.iter = +e.target.value;
    render();
  });
  function startIter() {
    state.iterPlaying = true;
    state.iterTimer = setInterval(() => {
      state.iter = state.iter >= 30 ? 0 : state.iter + 1;
      render();
    }, 350);
    render();
  }
  function stopIter() {
    state.iterPlaying = false;
    if (state.iterTimer) { clearInterval(state.iterTimer); state.iterTimer = null; }
    render();
  }
  function resetIter() { stopIter(); state.iter = 0; }

  // ───────── Drag in rotate mode ─────────
  canvas.addEventListener('pointerdown', (e) => {
    if (state.mode !== 'rotate') return;
    const size = canvasCtl.getSize();
    const rect = canvas.getBoundingClientRect();
    const scale = size.w / 8;
    const toMath = (px, py) => [(px - size.w/2)/scale, -(py - size.h/2)/scale];
    const apply = (ev) => {
      const r = canvas.getBoundingClientRect();
      state.draggedVec = normalize(toMath(ev.clientX - r.left, ev.clientY - r.top));
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

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw);

  function drawTransformedGrid(ctx, M, toPx, eigs) {
    ctx.strokeStyle = RULE_FAINT;
    ctx.lineWidth = 1;
    for (let i = -4; i <= 4; i++) {
      ctx.beginPath(); ctx.moveTo(...toPx([i,-4])); ctx.lineTo(...toPx([i,4])); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(...toPx([-4,i])); ctx.lineTo(...toPx([4,i])); ctx.stroke();
    }
    ctx.strokeStyle = RULE_DARK;
    for (let i = -4; i <= 4; i++) {
      ctx.beginPath();
      for (let t = -4; t <= 4; t += 0.25) {
        const [px, py] = toPx(mul(M, [i, t]));
        if (t === -4) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.beginPath();
      for (let t = -4; t <= 4; t += 0.25) {
        const [px, py] = toPx(mul(M, [t, i]));
        if (t === -4) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
    if (eigs.length) {
      ctx.strokeStyle = ACCENT_SOFT;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      eigs.forEach(({ vector }) => {
        const [vx, vy] = vector;
        ctx.beginPath();
        ctx.moveTo(...toPx([-vx*8, -vy*8]));
        ctx.lineTo(...toPx([vx*8, vy*8]));
        ctx.stroke();
      });
      ctx.setLineDash([]);
    }
  }

  function draw(ctx, size) {
    const { M, mode } = state;
    const scale = size.w / 8;
    const toPx = ([x, y]) => [size.w/2 + x*scale, size.h/2 - y*scale];
    const eigs = eigen2(M);

    drawTransformedGrid(ctx, M, toPx, eigs);

    // Axes
    ctx.strokeStyle = RULE_AXIS;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, size.h/2); ctx.lineTo(size.w, size.h/2);
    ctx.moveTo(size.w/2, 0); ctx.lineTo(size.w/2, size.h);
    ctx.stroke();

    if (mode === 'rotate') drawRotate(ctx, size, M, eigs, toPx);
    else if (mode === 'iterate') drawIterate(ctx, size, M, eigs, toPx);
    else drawExplore(ctx, size, M, eigs, toPx);
  }

  function drawExplore(ctx, size, M, eigs, toPx) {
    // ── Unit circle: where every input direction lives ──
    const [cx, cy] = toPx([0, 0]);
    const scl = toPx([1, 0])[0] - cx;
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 4]);
    ctx.beginPath();
    ctx.arc(cx, cy, scl, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // ── Image of the unit circle under M (always an ellipse, possibly degenerate) ──
    ctx.strokeStyle = 'rgba(38,35,32,0.55)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let t = 0; t <= Math.PI * 2 + 0.01; t += 0.02) {
      const [px, py] = toPx(mul(M, [Math.cos(t), Math.sin(t)]));
      if (t === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // ── Correspondence chords: 12 input directions (on unit circle) → their images
    //    For an eigenvector, the chord points radially (toward/away from origin).
    //    For everything else, the chord is sideways — that is the rotation. ──
    const N = 12;
    for (let k = 0; k < N; k++) {
      const a = (k / N) * Math.PI * 2;
      const v = [Math.cos(a), Math.sin(a)];
      const Mv = mul(M, v);
      ctx.strokeStyle = 'rgba(38,35,32,0.22)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(...toPx(v));
      ctx.lineTo(...toPx(Mv));
      ctx.stroke();

      // Input dot (on circle)
      ctx.fillStyle = 'rgba(38,35,32,0.40)';
      const [pvx, pvy] = toPx(v);
      ctx.beginPath();
      ctx.arc(pvx, pvy, 2, 0, Math.PI * 2);
      ctx.fill();

      // Output dot (on ellipse) — slightly larger so the eye reads input → output
      ctx.fillStyle = 'rgba(38,35,32,0.70)';
      const [pmx, pmy] = toPx(Mv);
      ctx.beginPath();
      ctx.arc(pmx, pmy, 2.6, 0, Math.PI * 2);
      ctx.fill();
    }

    // ── Eigenvectors: highlighted rays where the chord IS radial ──
    if (eigs.length) {
      eigs.forEach(({ value, vector }, i) => {
        const [vx, vy] = vector;
        // Eigen-line through origin (more prominent than the faint one in the grid)
        ctx.strokeStyle = ACCENT_FAINT;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(...toPx([-vx * 5, -vy * 5]));
        ctx.lineTo(...toPx([vx * 5, vy * 5]));
        ctx.stroke();
        ctx.setLineDash([]);

        // v on the unit circle (start)
        arrow(ctx, toPx([0, 0]), toPx([vx, vy]), ACCENT_FAINT, 2);
        // λ·v (where M takes it — still on the same line, only stretched)
        arrow(ctx, toPx([0, 0]), toPx([vx * value, vy * value]), ACCENT, 3, 11);

        ctx.fillStyle = ACCENT;
        ctx.font = SERIF_LABEL;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`λ${sub(i + 1)} = ${formatNum(value)}`,
          ...toPx([vx * value * 1.18, vy * value * 1.18]));
      });
    } else {
      ctx.fillStyle = INK_FADE;
      ctx.font = SERIF_LABEL_SM;
      ctx.textAlign = 'center';
      ctx.fillText('no real eigenvectors — every direction rotates', size.w / 2, size.h - 22);
    }

    // Footer caption: tells the reader what the chords mean
    ctx.fillStyle = INK_FADE;
    ctx.font = SERIF_LABEL_SM;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('chord = where M sends each direction', 18, size.h - 22);
  }

  function drawRotate(ctx, size, M, eigs, toPx) {
    // Unit circle
    const [cx, cy] = toPx([0,0]);
    const scl = toPx([1,0])[0] - cx;
    ctx.strokeStyle = 'rgba(38,35,32,0.25)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, scl, 0, Math.PI*2);
    ctx.stroke();

    if (eigs.length) {
      eigs.forEach(({ vector }) => {
        const [vx, vy] = vector;
        ctx.strokeStyle = 'rgba(122,31,36,0.55)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(...toPx([-vx*5, -vy*5]));
        ctx.lineTo(...toPx([vx*5, vy*5]));
        ctx.stroke();
        ctx.setLineDash([]);
      });
    }

    const v = state.draggedVec;
    const Mv = mul(M, v);
    arrow(ctx, toPx([0,0]), toPx(v), 'rgba(38,35,32,0.85)', 2.5);
    arrow(ctx, toPx([0,0]), toPx(Mv), ACCENT, 3, 11);

    const a1 = Math.atan2(v[1], v[0]);
    const a2 = Math.atan2(Mv[1], Mv[0]);
    ctx.strokeStyle = 'rgba(38,35,32,0.4)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, scl * 0.45, -a1, -a2, a2 < a1);
    ctx.stroke();

    ctx.fillStyle = INK;
    ctx.font = MONO_LABEL;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText('v', ...toPx([v[0]*1.18, v[1]*1.18]));
    ctx.fillStyle = ACCENT;
    ctx.fillText('Av', ...toPx([Mv[0]*1.10, Mv[1]*1.10]));

    let dAng = ((a2 - a1) * 180 / Math.PI);
    while (dAng > 180) dAng -= 360;
    while (dAng < -180) dAng += 360;
    ctx.fillStyle = INK_FADE;
    ctx.font = SERIF_LABEL_SM;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(`Δθ = ${dAng.toFixed(1)}°`, 18, size.h - 22);
    ctx.textAlign = 'right';
    ctx.fillText(`|Av| / |v| = ${norm(Mv).toFixed(3)}`, size.w - 18, size.h - 22);
  }

  function drawIterate(ctx, size, M, eigs, toPx) {
    const N = 24;
    for (let k = 0; k < N; k++) {
      const a = (k / N) * Math.PI * 2;
      let v = [Math.cos(a), Math.sin(a)];
      const trail = [v.slice()];
      for (let i = 0; i < state.iter; i++) {
        v = mul(M, v);
        const n = norm(v);
        if (n > 1e-9) v = [v[0]/n, v[1]/n];
        trail.push(v.slice());
      }
      ctx.strokeStyle = 'rgba(38,35,32,0.12)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      trail.forEach(([x, y], idx) => {
        const [px, py] = toPx([x, y]);
        if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
      arrow(ctx, toPx([0,0]), toPx(v), 'rgba(38,35,32,0.6)', 1.5, 7);
    }

    if (eigs.length) {
      const sorted = [...eigs].sort((a,b) => Math.abs(b.value) - Math.abs(a.value));
      const dom = sorted[0];
      const [vx, vy] = dom.vector;
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(...toPx([-vx*5, -vy*5]));
      ctx.lineTo(...toPx([vx*5, vy*5]));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = ACCENT;
      ctx.font = SERIF_LABEL_SM;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`dominant eigenvector  λ = ${formatNum(dom.value)}`,
        ...toPx([vx*4.3, vy*4.3]));
    }

    ctx.fillStyle = INK_FADE;
    ctx.font = SERIF_LABEL_SM;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(`step ${state.iter}`, 18, size.h - 22);
  }

  // ───────── Caption + marginalia ─────────
  function captionFor(mode) {
    if (mode === 'explore') return `<span class="ml-cap-num">I.</span> The dashed circle is the unit circle; the solid curve is its image — where M sends every direction. Each grey chord connects an input direction to its destination. Almost every chord points sideways (the direction got rotated). The <span class="ml-ink-orange">accent arrows</span> mark the rare directions whose chord points purely along the radius — those are the eigenvectors.`;
    if (mode === 'rotate') return `<span class="ml-cap-num">II.</span> Drag the dark arrow around the unit circle. Watch the <span class="ml-ink-orange">accent arrow</span> — it is <span class="ml-ink">A</span> applied to your vector. Almost everywhere it twists away. Cross an eigen-line and it falls perfectly in step.`;
    return `<span class="ml-cap-num">III.</span> Apply <span class="ml-ink">A</span> over and over, normalizing each step. No matter where vectors start, they fold onto the <span class="ml-ink-orange">dominant eigenvector</span>. This is the seed of power iteration — and of PageRank.`;
  }

  function marginFor(mode, det) {
    let body = '';
    if (mode === 'explore') {
      body = `
        <p>The defining equation:</p>
        <div class="ml-equation">A v = λ v</div>
        <p>Read it as: <em>A applied to v gives back v itself, only longer or shorter.</em> The number λ tells you by how much.</p>
        <p class="quiet">For a 2×2 matrix, λ solves <span class="ml-ink-mono">λ² − tr(A)·λ + det(A) = 0</span>. When the discriminant is negative, no real λ exists — the transform is essentially a rotation.</p>`;
    } else if (mode === 'rotate') {
      body = `
        <p><em>Δθ</em> is the angle the vector rotated. On an eigen-line, Δθ collapses to 0 (or 180°, for a negative λ).</p>
        <p><em>|Av|/|v|</em> is the local stretch factor. On an eigen-line it equals exactly |λ|.</p>`;
    } else {
      body = `
        <p>Multiplying by A and re-normalizing is called <em>power iteration</em>. It converges to the eigenvector of the largest eigenvalue.</p>
        <p>Google's original PageRank used this on a matrix of web links: pages were vectors, the link structure was A, and the dominant eigenvector ranked the web.</p>
        <p class="quiet">If the two eigenvalues have equal magnitude, convergence stalls. Try the rotation preset.</p>`;
    }
    return `
      <div class="ml-margin-rule"></div>
      <div class="ml-margin-h">In the margin</div>
      ${body}
      <div class="ml-margin-rule thin"></div>
      <div class="ml-margin-h sm">Determinant</div>
      <p class="quiet">det(A) = λ₁·λ₂ = ${formatNum(det)}. It is the factor by which areas scale. Negative means the plane was flipped.</p>`;
  }

  // ───────── Render ─────────
  function render() {
    matrixCtl.refresh();
    const M = state.M;
    const tr = M[0][0] + M[1][1];
    const det = M[0][0]*M[1][1] - M[0][1]*M[1][0];
    const eigs = eigen2(M);
    readoutEl.querySelector('[data-k="tr"]').textContent = formatNum(tr);
    readoutEl.querySelector('[data-k="det"]').textContent = formatNum(det);
    readoutEl.querySelector('[data-k="l1"]').textContent = eigs.length ? formatNum(eigs[0].value) : 'complex';
    readoutEl.querySelector('[data-k="l2"]').textContent = eigs.length ? formatNum(eigs[1].value) : 'complex';

    const matched = PRESETS.find((p) => matEq(p.M, M));
    Array.from(presetsEl.children).forEach((btn, i) => {
      btn.classList.toggle('active', PRESETS[i] === matched);
    });
    blurbEl.textContent = matched ? matched.blurb : 'Custom matrix.';

    Array.from(tabsEl.children).forEach((btn) => {
      btn.classList.toggle('on', btn.dataset.mode === state.mode);
    });

    captionEl.innerHTML = captionFor(state.mode);
    marginEl.innerHTML = marginFor(state.mode, det);

    iterEl.style.display = state.mode === 'iterate' ? 'flex' : 'none';
    playBtn.textContent = state.iterPlaying ? 'Pause' : 'Play';
    iterSlider.value = state.iter;
    iterLabel.textContent = `step ${state.iter}`;

    canvas.style.cursor = state.mode === 'rotate' ? 'grab' : 'default';
    canvasCtl.redraw();
  }

  render();
})();
