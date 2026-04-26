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

    // ── Correspondence chords: 12 input directions (on unit circle) → their images.
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

      ctx.fillStyle = 'rgba(38,35,32,0.40)';
      const [pvx, pvy] = toPx(v);
      ctx.beginPath();
      ctx.arc(pvx, pvy, 2, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = 'rgba(38,35,32,0.70)';
      const [pmx, pmy] = toPx(Mv);
      ctx.beginPath();
      ctx.arc(pmx, pmy, 2.6, 0, Math.PI * 2);
      ctx.fill();
    }

    // ── Eigenvectors: highlighted rays where the chord IS radial.
    //    Each gets λv tip-label + Av = λv anchored along the line. ──
    if (eigs.length) {
      eigs.forEach(({ value, vector }, i) => {
        const [vx, vy] = vector;
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

        // λ value pinned right at the tip
        ctx.fillStyle = ACCENT;
        ctx.font = SERIF_LABEL;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const tipDir = value >= 0 ? 1 : -1;
        ctx.fillText(`λ${sub(i + 1)} = ${formatNum(value)}`,
          ...toPx([vx * (Math.abs(value) + 0.55) * tipDir, vy * (Math.abs(value) + 0.55) * tipDir]));
      });

      // Anchor "Av = λv" along the dominant eigen-line, with a side caption.
      const dom = [...eigs].sort((a,b) => Math.abs(b.value) - Math.abs(a.value))[0];
      const [evx, evy] = dom.vector;
      // Place equation along the line, slightly off-axis (perpendicular offset).
      const px = -evy, py = evx; // perpendicular unit
      const t = 3.0;
      const off = 0.25;
      ctx.save();
      const [eqPx, eqPy] = toPx([evx * t + px * off, evy * t + py * off]);
      let ang = Math.atan2(-evy, evx); // canvas y is flipped
      if (ang > Math.PI/2) ang -= Math.PI;
      if (ang < -Math.PI/2) ang += Math.PI;
      ctx.translate(eqPx, eqPy);
      ctx.rotate(ang);
      ctx.fillStyle = ACCENT;
      ctx.font = 'italic 15px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('A v = λ v', 0, 0);
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.fillStyle = 'rgba(122,31,36,0.7)';
      ctx.fillText('no rotation — only stretch', 0, 18);
      ctx.restore();
    } else {
      // ── Complex eigenvalue case: show a spiral so "no real eigenvector" is
      //    pictorial, not just a missing label. ──
      const N2 = 4;
      const steps = 18;
      for (let k = 0; k < N2; k++) {
        const a = (k / N2) * Math.PI * 2;
        let v = [Math.cos(a) * 0.7, Math.sin(a) * 0.7];
        ctx.strokeStyle = 'rgba(122,31,36,0.45)';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(...toPx(v));
        for (let s = 0; s < steps; s++) {
          v = mul(M, v);
          ctx.lineTo(...toPx(v));
        }
        ctx.stroke();
        // Tip dot
        ctx.fillStyle = ACCENT;
        const [tpx, tpy] = toPx(v);
        ctx.beginPath();
        ctx.arc(tpx, tpy, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.fillStyle = ACCENT;
      ctx.font = 'italic 14px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText('every direction rotates — no real eigenvector', size.w / 2, 28);
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.fillStyle = INK_FADE;
      ctx.fillText('λ is complex; vectors trace spirals instead of staying on a line', size.w / 2, 46);
    }

    // Footer caption
    ctx.fillStyle = INK_FADE;
    ctx.font = SERIF_LABEL_SM;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('chord = where M sends each direction', 18, size.h - 22);

    // ── Top-left legend: small swatches for chord / dots / eigen-line ──
    const legX = 14;
    const legY = 14;
    const legW = 178;
    const legH = eigs.length ? 78 : 60;
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
    // small dot = input direction v
    ctx.fillStyle = 'rgba(38,35,32,0.40)';
    ctx.beginPath(); ctx.arc(legX + 18, legY + 30, 2, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.fillText('input  v  on unit circle', legX + 30, legY + 33);
    // bigger dot = Mv
    ctx.fillStyle = 'rgba(38,35,32,0.70)';
    ctx.beginPath(); ctx.arc(legX + 18, legY + 47, 2.6, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = INK_FADE;
    ctx.fillText('output  Mv  (after matrix)', legX + 30, legY + 50);
    if (eigs.length) {
      // dashed accent = eigen-line
      ctx.strokeStyle = ACCENT_FAINT;
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(legX + 10, legY + 64); ctx.lineTo(legX + 28, legY + 64);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = ACCENT;
      ctx.fillText('eigen-line  (no rotation)', legX + 30, legY + 67);
    }
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

    // ── Closeness to nearest eigen-direction (cos of angle, abs of dot product). ──
    let nearestAngle = null;
    let nearestEig = null;
    if (eigs.length) {
      let best = -Infinity;
      eigs.forEach((e) => {
        const c = Math.abs(v[0]*e.vector[0] + v[1]*e.vector[1]);
        if (c > best) { best = c; nearestEig = e; }
      });
      nearestAngle = Math.acos(Math.min(1, best)) * 180 / Math.PI;
    }
    const onEigen = nearestAngle !== null && nearestAngle < 4.0;

    arrow(ctx, toPx([0,0]), toPx(v), 'rgba(38,35,32,0.85)', 2.5);
    arrow(ctx, toPx([0,0]), toPx(Mv), ACCENT, 3, 11);

    // ── Discoverability hint: gently telegraph that the dark arrow is draggable.
    //    Sits as faint text near the unit circle's edge so the user can't miss it. ──
    ctx.fillStyle = 'rgba(38,35,32,0.45)';
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('drag anywhere to rotate v', size.w / 2, 18);

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

    // ── "AT EIGENVECTOR" indicator when v is aligned with an eigen-direction. ──
    if (onEigen && nearestEig) {
      // Box + λ readout.
      const boxX = size.w - 220;
      const boxY = 16;
      const boxW = 200;
      const boxH = 60;
      ctx.fillStyle = 'rgba(122,31,36,0.10)';
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 1.5;
      ctx.fillRect(boxX, boxY, boxW, boxH);
      ctx.strokeRect(boxX, boxY, boxW, boxH);
      ctx.fillStyle = ACCENT;
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText('AT EIGENVECTOR', boxX + 12, boxY + 22);
      ctx.font = 'italic 16px "Source Serif 4", Georgia, serif';
      ctx.fillText(`Av = ${formatNum(nearestEig.value)} · v`, boxX + 12, boxY + 46);
    }

    let dAng = ((a2 - a1) * 180 / Math.PI);
    while (dAng > 180) dAng -= 360;
    while (dAng < -180) dAng += 360;
    ctx.fillStyle = INK_FADE;
    ctx.font = SERIF_LABEL_SM;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(`Δθ = ${dAng.toFixed(1)}°`, 18, size.h - 22);
    if (nearestAngle !== null) {
      ctx.textAlign = 'center';
      ctx.fillText(`angle from nearest eigen-line: ${nearestAngle.toFixed(1)}°`, size.w / 2, size.h - 22);
    }
    ctx.textAlign = 'right';
    ctx.fillText(`|Av| / |v| = ${norm(Mv).toFixed(3)}`, size.w - 18, size.h - 22);
  }

  function drawIterate(ctx, size, M, eigs, toPx) {
    const N = 24;

    // ── Complex-eigenvalue case: power iteration on raw vectors traces a spiral
    //    on the plane (not on the unit circle). Show the unnormalized walk. ──
    if (!eigs.length) {
      for (let k = 0; k < N; k++) {
        const a = (k / N) * Math.PI * 2;
        let v = [Math.cos(a), Math.sin(a)];
        const trail = [v.slice()];
        for (let i = 0; i < state.iter; i++) {
          v = mul(M, v);
          // Don't normalize — the spiral is the point.
          trail.push(v.slice());
        }
        ctx.strokeStyle = 'rgba(122,31,36,0.25)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        trail.forEach(([x, y], idx) => {
          const [px, py] = toPx([x, y]);
          if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        });
        ctx.stroke();
        arrow(ctx, toPx([0,0]), toPx(v), 'rgba(122,31,36,0.6)', 1.4, 6);
      }
      ctx.fillStyle = ACCENT;
      ctx.font = 'italic 14px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText('complex λ — vectors spiral instead of converging', size.w / 2, 28);
      ctx.font = SERIF_LABEL_SM;
      ctx.fillStyle = INK_FADE;
      ctx.fillText(`step ${state.iter}`, size.w / 2, size.h - 22);
      return;
    }

    // ── Real-eigen case: normalized power iteration. ──
    const sorted = [...eigs].sort((a,b) => Math.abs(b.value) - Math.abs(a.value));
    const dom = sorted[0];
    const sub2 = sorted[1];
    const [vx, vy] = dom.vector;

    // Track convergence: |v_t · v₁| over iterations, for one representative seed.
    const convSeed = [Math.cos(0.4), Math.sin(0.4)];
    const conv = [Math.abs(convSeed[0]*vx + convSeed[1]*vy)];

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

    // Build the FULL 30-step convergence curve once for the inset (not capped to state.iter).
    {
      let v = convSeed.slice();
      for (let i = 0; i < 30; i++) {
        v = mul(M, v);
        const n = norm(v); if (n > 1e-9) v = [v[0]/n, v[1]/n];
        conv.push(Math.abs(v[0]*vx + v[1]*vy));
      }
    }

    // Dominant eigen-line
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
    ctx.fillText(`dominant eigenvector  λ${sub(1)} = ${formatNum(dom.value)}`,
      ...toPx([vx*4.3, vy*4.3]));

    // ── Convergence inset (top-right corner): |v_t · v₁| → 1 ──
    const insetW = 180;
    const insetH = 92;
    const insetX = size.w - insetW - 16;
    const insetY = 16;
    ctx.fillStyle = 'rgba(255,253,246,0.92)';
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.fillRect(insetX, insetY, insetW, insetH);
    ctx.strokeRect(insetX, insetY, insetW, insetH);

    // Title
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('CONVERGENCE', insetX + 8, insetY + 14);

    // Plot area
    const plotX = insetX + 10;
    const plotY = insetY + 22;
    const plotW = insetW - 20;
    const plotH = insetH - 36;
    // Axis: y goes 0 → 1
    ctx.strokeStyle = 'rgba(38,35,32,0.20)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plotX, plotY); ctx.lineTo(plotX, plotY + plotH);
    ctx.lineTo(plotX + plotW, plotY + plotH);
    ctx.stroke();
    // 1.0 reference line
    ctx.strokeStyle = 'rgba(38,35,32,0.20)';
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(plotX, plotY); ctx.lineTo(plotX + plotW, plotY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Curve
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const stepsShown = Math.min(conv.length - 1, 30);
    for (let i = 0; i <= stepsShown; i++) {
      const x = plotX + (i / 30) * plotW;
      const y = plotY + (1 - conv[i]) * plotH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    // Current step marker
    if (state.iter <= 30 && state.iter < conv.length) {
      const cx = plotX + (state.iter / 30) * plotW;
      const cy = plotY + (1 - conv[state.iter]) * plotH;
      ctx.fillStyle = ACCENT;
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // Decay-rate annotation
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('|vₜ·v₁|', plotX, plotY - 2);
    ctx.textAlign = 'right';
    ctx.fillText('1.0', plotX + plotW, plotY - 2);

    // Rate label below the inset (rate ≈ |λ₂/λ₁|)
    const rate = Math.abs(sub2.value) / Math.max(1e-9, Math.abs(dom.value));
    ctx.font = 'italic 10px "Source Serif 4", Georgia, serif';
    ctx.fillStyle = INK_FADE;
    ctx.textAlign = 'left';
    ctx.fillText(`rate ≈ |λ₂/λ₁| = ${rate.toFixed(2)}`, insetX + 8, insetY + insetH - 4);

    // Footer
    ctx.fillStyle = INK_FADE;
    ctx.font = SERIF_LABEL_SM;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(`step ${state.iter}`, 18, size.h - 22);
  }

  // ───────── Caption + marginalia ─────────
  function captionFor(mode) {
    if (mode === 'explore') return `<span class="ml-cap-num">I.</span> The dashed circle is the unit circle; the solid curve is its image — where M sends every direction. Each grey chord connects an input direction to its destination. Almost every chord points sideways (the direction got rotated). The <span class="ml-ink-orange">accent arrows</span> mark the rare directions whose chord points purely along the radius — those are the eigenvectors. The orange equation is anchored on the eigen-line itself: <em>Av = λv</em>, no rotation — only stretch.`;
    if (mode === 'rotate') return `<span class="ml-cap-num">II.</span> Drag the dark arrow around the unit circle. The <span class="ml-ink-orange">accent arrow</span> is <em>Av</em>. Almost everywhere it twists away from <em>v</em>. Cross an eigen-line and a small badge lights up — at that orientation A acts as pure scalar multiplication.`;
    return `<span class="ml-cap-num">III.</span> Apply <em>A</em> over and over, normalizing each step. Vectors fold onto the <span class="ml-ink-orange">dominant eigenvector</span>. The inset tracks how fast: |vₜ · v₁| → 1 at a rate equal to <em>|λ₂/λ₁|</em>. This is power iteration — and the seed of PageRank.`;
  }

  function marginFor(mode, det) {
    let body = '';
    if (mode === 'explore') {
      body = `
        <p>The defining equation:</p>
        <div class="ml-equation">A v = λ v</div>
        <p class="quiet" style="margin-top:-8px"><em>A</em> = the matrix you set above. <em>v</em> = a vector (an arrow). <em>λ</em> (read "<em>lambda</em>") = a number. So <em>Av = λv</em> means: A turns v into a scaled copy of itself — same direction, just stretched by a factor of λ.</p>
        <p>Read it as: <em>A applied to v gives back v itself, only longer or shorter.</em> Watch the equation pinned to the eigen-line on the canvas — the <span class="ml-ink-orange">accent arrow</span> is the same line as <em>v</em>, just rescaled.</p>
        <p class="quiet">For a 2×2 matrix, λ solves <span class="ml-ink-mono">λ² − tr(A)·λ + det(A) = 0</span>. (<em>tr</em> is the trace — the sum of the diagonal entries. <em>det</em> is the determinant.) When the discriminant is negative, no real λ exists — the transform is essentially a rotation, and the canvas shows a faint spiral instead of a line.</p>`;
    } else if (mode === 'rotate') {
      body = `
        <p><em>Δθ</em> is the angle the vector rotated. On an eigen-line, Δθ collapses to 0 (or 180°, for a negative λ).</p>
        <p><em>|Av|/|v|</em> is the local stretch factor. On an eigen-line it equals exactly |λ|.</p>
        <p class="quiet">The <em>angle from nearest eigen-line</em> readout is the live "are we there yet" — drag toward zero and a badge lights up confirming you've found <em>v</em>.</p>`;
    } else {
      body = `
        <p>Multiplying by A and re-normalizing is called <em>power iteration</em>. It converges to the eigenvector of the largest eigenvalue.</p>
        <p>The <em>convergence inset</em> plots <span class="ml-ink-mono">|vₜ · v₁|</span>, the alignment with the true answer. (The dot <em>·</em> is dot product; subscript <em>t</em> means "at step t".) Each step it shrinks the gap by roughly the factor <em>|λ₂/λ₁|</em>: the closer that ratio is to 1, the slower convergence is.</p>
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
