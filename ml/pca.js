/* Vol. IV — Principal Components */
(function () {
  'use strict';
  const V = window.MLViz;
  const { eigen2, formatNum, arrow,
          ACCENT, INK, INK_FADE, RULE_FAINT, RULE_AXIS,
          ACCENT_FAINT, SERIF_LABEL, SERIF_LABEL_SM } = V;

  function gaussian() {
    let u = 0, w = 0;
    while (u === 0) u = Math.random();
    while (w === 0) w = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * w);
  }
  function makeCloud(n, sx, sy, theta, cx = 0, cy = 0) {
    const c = Math.cos(theta), s = Math.sin(theta);
    const pts = [];
    for (let i = 0; i < n; i++) {
      const x0 = gaussian() * sx, y0 = gaussian() * sy;
      pts.push([cx + x0*c - y0*s, cy + x0*s + y0*c]);
    }
    return pts;
  }
  function covariance(pts) {
    const n = pts.length;
    let mx = 0, my = 0;
    for (const [x, y] of pts) { mx += x; my += y; }
    mx /= n; my /= n;
    let cxx = 0, cxy = 0, cyy = 0;
    for (const [x, y] of pts) {
      const dx = x - mx, dy = y - my;
      cxx += dx*dx; cxy += dx*dy; cyy += dy*dy;
    }
    return { mean: [mx, my], cov: [[cxx/n, cxy/n], [cxy/n, cyy/n]] };
  }

  const PRESETS = [
    { name: 'Slanted blob',   make: () => makeCloud(180, 2.4, 0.7, 0.55) },
    { name: 'Rounder cloud',  make: () => makeCloud(180, 1.6, 1.3, 0.3) },
    { name: 'Tight diagonal', make: () => makeCloud(180, 2.6, 0.35, -0.7) },
    { name: 'Two clusters',   make: () => [...makeCloud(90, 0.5, 0.5, 0, -1.4, -0.6), ...makeCloud(90, 0.5, 0.5, 0, 1.4, 0.6)] },
    { name: 'Banana',         make: () => {
      const pts = [];
      for (let i = 0; i < 180; i++) {
        const t = (Math.random() - 0.5) * 3.5;
        pts.push([t + gaussian()*0.2, 0.45*t*t - 0.9 + gaussian()*0.2]);
      }
      return pts;
    } },
    { name: 'Isotropic',      make: () => makeCloud(180, 1.4, 1.4, 0) },
  ];

  const state = {
    presetIdx: 0,
    pts: PRESETS[0].make(),
    mode: 'explore',
    projT: 0,
    testAngle: 0, // radians, 0..π. user-controlled candidate axis in 'explore' mode.
    rafId: null,
  };

  function varianceAlong(pts, mean, theta) {
    const dx = Math.cos(theta), dy = Math.sin(theta);
    let sum = 0;
    for (const [x, y] of pts) {
      const proj = (x - mean[0]) * dx + (y - mean[1]) * dy;
      sum += proj * proj;
    }
    return pts.length ? sum / pts.length : 0;
  }

  // ───────── DOM ─────────
  const covEl = document.getElementById('cov-matrix');
  const readoutEl = document.getElementById('readout');
  const presetsEl = document.getElementById('presets');
  const resampleBtn = document.getElementById('resample');
  const tabsEl = document.getElementById('mode-tabs');
  const captionEl = document.getElementById('caption');
  const marginEl = document.getElementById('margin');
  const canvas = document.getElementById('canvas');
  const angleControls = document.getElementById('angle-controls');
  const angleSlider = document.getElementById('angle-slider');
  const angleLabel = document.getElementById('angle-label');
  const varianceLabel = document.getElementById('variance-label');

  // Read-only covariance display.
  function renderCovMatrix(cov) {
    covEl.innerHTML = '';
    covEl.classList.add('ml-matrix');
    const left = document.createElement('div');
    left.className = 'ml-bracket';
    const right = document.createElement('div');
    right.className = 'ml-bracket right';
    const grid = document.createElement('div');
    grid.className = 'ml-entries';
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 2; c++) {
        const wrap = document.createElement('div');
        wrap.className = 'ml-cell-wrap';
        const cell = document.createElement('div');
        cell.className = 'ml-cell';
        cell.style.cursor = 'default';
        cell.textContent = formatNum(cov[r][c]);
        wrap.appendChild(cell);
        grid.appendChild(wrap);
      }
    }
    covEl.appendChild(left);
    covEl.appendChild(grid);
    covEl.appendChild(right);
  }

  // ───────── Presets ─────────
  PRESETS.forEach((p, i) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = p.name;
    b.addEventListener('click', () => {
      state.presetIdx = i;
      state.pts = PRESETS[i].make();
      restartProjection();
      render();
    });
    presetsEl.appendChild(b);
  });
  resampleBtn.addEventListener('click', () => {
    state.pts = PRESETS[state.presetIdx].make();
    restartProjection();
    render();
  });

  // ───────── Mode tabs ─────────
  Array.from(tabsEl.children).forEach((btn) => {
    btn.addEventListener('click', () => {
      state.mode = btn.dataset.mode;
      restartProjection();
      render();
    });
  });

  // ───────── Angle slider (only used in 'explore' mode) ─────────
  angleSlider.addEventListener('input', (e) => {
    state.testAngle = (+e.target.value) * Math.PI / 180;
    render();
  });

  // ───────── Animation ─────────
  function restartProjection() {
    cancelAnimationFrame(state.rafId);
    state.projT = 0;
    if (state.mode !== 'reduce') return;
    let last = performance.now();
    const tick = (now) => {
      const dt = (now - last) / 1000;
      last = now;
      state.projT = Math.min(1, state.projT + dt * 0.4);
      render();
      if (state.projT < 1) state.rafId = requestAnimationFrame(tick);
    };
    state.rafId = requestAnimationFrame(tick);
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw);

  function draw(ctx, size) {
    const pts = state.pts;
    const { mean, cov } = covariance(pts);
    const eigs = eigen2(cov);
    if (eigs.length) eigs.sort((a, b) => b.value - a.value);

    const scale = size.w / 9;
    const toPx = ([x, y]) => [size.w/2 + x*scale, size.h/2 - y*scale];

    // Faint grid + axes
    ctx.strokeStyle = RULE_FAINT;
    ctx.lineWidth = 1;
    for (let i = -4; i <= 4; i++) {
      ctx.beginPath(); ctx.moveTo(...toPx([i, -4])); ctx.lineTo(...toPx([i, 4])); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(...toPx([-4, i])); ctx.lineTo(...toPx([4, i])); ctx.stroke();
    }
    ctx.strokeStyle = RULE_AXIS;
    ctx.beginPath();
    ctx.moveTo(0, size.h/2); ctx.lineTo(size.w, size.h/2);
    ctx.moveTo(size.w/2, 0); ctx.lineTo(size.w/2, size.h);
    ctx.stroke();

    const [mx, my] = mean;

    // ── Reference: PC1 dashed line ──
    if (eigs.length && state.mode !== 'explore') {
      const pc1 = eigs[0].vector;
      ctx.strokeStyle = ACCENT_FAINT;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(...toPx([mx - pc1[0]*6, my - pc1[1]*6]));
      ctx.lineTo(...toPx([mx + pc1[0]*6, my + pc1[1]*6]));
      ctx.stroke();
      ctx.setLineDash([]);

      if (state.mode === 'project') {
        ctx.strokeStyle = 'rgba(38,35,32,0.18)';
        ctx.lineWidth = 0.8;
        for (const [x, y] of pts) {
          const dx = x - mx, dy = y - my;
          const t = dx*pc1[0] + dy*pc1[1];
          ctx.beginPath();
          ctx.moveTo(...toPx([x, y]));
          ctx.lineTo(...toPx([mx + t*pc1[0], my + t*pc1[1]]));
          ctx.stroke();
        }
      }
    }

    // ── 'explore' mode: candidate axis ──
    if (eigs.length && state.mode === 'explore') {
      const pc1 = eigs[0].vector;
      ctx.strokeStyle = 'rgba(122,31,36,0.18)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 5]);
      ctx.beginPath();
      ctx.moveTo(...toPx([mx - pc1[0]*6, my - pc1[1]*6]));
      ctx.lineTo(...toPx([mx + pc1[0]*6, my + pc1[1]*6]));
      ctx.stroke();
      ctx.setLineDash([]);

      const ax = Math.cos(state.testAngle);
      const ay = Math.sin(state.testAngle);

      ctx.strokeStyle = 'rgba(38,35,32,0.22)';
      ctx.lineWidth = 0.8;
      for (const [x, y] of pts) {
        const dx = x - mx, dy = y - my;
        const t = dx*ax + dy*ay;
        ctx.beginPath();
        ctx.moveTo(...toPx([x, y]));
        ctx.lineTo(...toPx([mx + t*ax, my + t*ay]));
        ctx.stroke();
      }

      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(...toPx([mx - ax*6, my - ay*6]));
      ctx.lineTo(...toPx([mx + ax*6, my + ay*6]));
      ctx.stroke();
    }

    // ── Mode III: residuals from projected → original (visualizes Eckart–Young / MSE). ──
    let mseVal = null;
    if (eigs.length && state.mode === 'reduce') {
      const pc1 = eigs[0].vector;
      const tt = state.projT;
      let sumSq = 0;
      // After the slide settles (tt ≥ 1) draw thin residuals — what was lost.
      if (tt >= 1) {
        ctx.strokeStyle = 'rgba(38,35,32,0.22)';
        ctx.lineWidth = 0.8;
        for (const [x, y] of pts) {
          const dx = x - mx, dy = y - my;
          const t = dx*pc1[0] + dy*pc1[1];
          const projX = mx + t*pc1[0];
          const projY = my + t*pc1[1];
          ctx.beginPath();
          ctx.moveTo(...toPx([x, y]));
          ctx.lineTo(...toPx([projX, projY]));
          ctx.stroke();
        }
      }
      // MSE is always defined (use the original→projection distance).
      for (const [x, y] of pts) {
        const dx = x - mx, dy = y - my;
        const t = dx*pc1[0] + dy*pc1[1];
        const projX = mx + t*pc1[0], projY = my + t*pc1[1];
        sumSq += (x - projX)*(x - projX) + (y - projY)*(y - projY);
      }
      mseVal = sumSq / pts.length;
    }

    // Points (animated for 'reduce')
    ctx.fillStyle = INK;
    if (eigs.length && state.mode === 'reduce') {
      const pc1 = eigs[0].vector;
      const tt = state.projT;
      // Originals (faded, only after the slide so it doesn't double up at start)
      if (tt >= 1) {
        ctx.fillStyle = 'rgba(38,35,32,0.30)';
        for (const [x, y] of pts) {
          const [px, py] = toPx([x, y]);
          ctx.beginPath();
          ctx.arc(px, py, 1.7, 0, Math.PI*2);
          ctx.fill();
        }
      }
      // Sliding / projected points
      ctx.fillStyle = INK;
      for (const [x, y] of pts) {
        const dx = x - mx, dy = y - my;
        const t = dx*pc1[0] + dy*pc1[1];
        const projX = mx + t*pc1[0];
        const projY = my + t*pc1[1];
        const fx = x*(1-tt) + projX*tt;
        const fy = y*(1-tt) + projY*tt;
        const [px, py] = toPx([fx, fy]);
        ctx.beginPath();
        ctx.arc(px, py, 2.2, 0, Math.PI*2);
        ctx.fill();
      }
    } else {
      for (const [x, y] of pts) {
        const [px, py] = toPx([x, y]);
        ctx.beginPath();
        ctx.arc(px, py, 2.2, 0, Math.PI*2);
        ctx.fill();
      }
    }

    // Mean
    ctx.fillStyle = ACCENT;
    const [mpx, mpy] = toPx([mx, my]);
    ctx.beginPath();
    ctx.arc(mpx, mpy, 4, 0, Math.PI*2);
    ctx.fill();

    // PC arrows + the cov·v = λ·v anchor (in project / reduce).
    if (eigs.length && state.mode !== 'explore') {
      const pc1 = eigs[0].vector, pc2 = eigs[1].vector;
      const s1 = Math.sqrt(Math.max(0, eigs[0].value)) * 2;
      const s2 = Math.sqrt(Math.max(0, eigs[1].value)) * 2;
      arrow(ctx, toPx([mx, my]), toPx([mx + pc1[0]*s1, my + pc1[1]*s1]), ACCENT, 3, 11);
      arrow(ctx, toPx([mx, my]), toPx([mx + pc2[0]*s2, my + pc2[1]*s2]),
            'rgba(122,31,36,0.55)', 2.5, 9);
      ctx.fillStyle = ACCENT;
      ctx.font = SERIF_LABEL_SM;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`PC₁  λ=${formatNum(eigs[0].value)}`,
        ...toPx([mx + pc1[0]*(s1+0.45), my + pc1[1]*(s1+0.45)]));
      ctx.fillStyle = 'rgba(122,31,36,0.7)';
      ctx.fillText(`PC₂  λ=${formatNum(eigs[1].value)}`,
        ...toPx([mx + pc2[0]*(s2+0.45), my + pc2[1]*(s2+0.45)]));

      // Anchored equation: cov · v = λ · v, sitting along PC1.
      if (state.mode === 'project') {
        const px = -pc1[1], py = pc1[0]; // perpendicular unit
        const off = 0.35;
        const t = -2.4;
        const [eqPx, eqPy] = toPx([mx + pc1[0]*t + px*off, my + pc1[1]*t + py*off]);
        let ang = Math.atan2(-pc1[1], pc1[0]);
        if (ang > Math.PI/2) ang -= Math.PI;
        if (ang < -Math.PI/2) ang += Math.PI;
        ctx.save();
        ctx.translate(eqPx, eqPy);
        ctx.rotate(ang);
        ctx.fillStyle = ACCENT;
        ctx.font = 'italic 14px "Source Serif 4", Georgia, serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('cov · v = λ · v', 0, 0);
        ctx.restore();
      }
    }

    // ── Scree plot inset (top-left, in all modes when eigs valid) ──
    if (eigs.length) drawScree(ctx, size, eigs);

    // ── Rotated PC₁/PC₂ inset (project + reduce): same data in PCA's frame. ──
    if (eigs.length && state.mode !== 'explore') drawRotatedInset(ctx, size, pts, mean, eigs);

    // ── Below-SCREE legend: small swatches for PC1, PC2, residual. ──
    if (eigs.length) {
      const legX = 16;
      const legY = 116; // SCREE ends at 16+92=108
      const legW = 170;
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
      // PC1 swatch (full accent arrow)
      arrow(ctx, [legX + 12, legY + 28], [legX + 36, legY + 28], ACCENT, 3, 10);
      ctx.fillStyle = ACCENT;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.fillText('PC₁  (most variance)', legX + 44, legY + 31);
      // PC2 swatch (faint accent)
      arrow(ctx, [legX + 12, legY + 44], [legX + 32, legY + 44], 'rgba(122,31,36,0.55)', 2, 8);
      ctx.fillStyle = 'rgba(122,31,36,0.7)';
      ctx.fillText('PC₂  (perpendicular)', legX + 44, legY + 47);
      // residual swatch
      ctx.strokeStyle = 'rgba(38,35,32,0.6)';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(legX + 12, legY + 60); ctx.lineTo(legX + 36, legY + 60);
      ctx.stroke();
      ctx.fillStyle = INK_FADE;
      ctx.fillText('residual  (lost detail)', legX + 44, legY + 63);
    }

    // Footer
    ctx.fillStyle = INK_FADE;
    ctx.font = SERIF_LABEL_SM;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(`${pts.length} points`, 18, size.h - 22);
    if (eigs.length) {
      ctx.textAlign = 'right';
      if (state.mode === 'explore') {
        const v = varianceAlong(pts, mean, state.testAngle);
        const optimum = eigs[0].value;
        const pct = optimum > 0 ? (v / optimum * 100) : 0;
        ctx.fillText(`your var = ${formatNum(v)}   ·   PC₁ optimum = ${formatNum(optimum)}   (${pct.toFixed(0)}%)`,
          size.w - 18, size.h - 22);
      } else if (state.mode === 'reduce' && mseVal !== null) {
        const total = eigs[0].value + eigs[1].value;
        const explained = total > 0 ? eigs[0].value / total : 0;
        ctx.fillText(`PC₁ explains ${(explained*100).toFixed(1)}%   ·   reconstruction MSE = ${formatNum(mseVal)}`,
          size.w - 18, size.h - 22);
      } else {
        const total = eigs[0].value + eigs[1].value;
        const explained = total > 0 ? eigs[0].value / total : 0;
        ctx.fillText(`PC₁ explains ${(explained*100).toFixed(1)}% of variance`,
          size.w - 18, size.h - 22);
      }
    }
  }

  // Scree plot: a tiny bar chart of λ₁ / λ₂ summing to total variance, with %.
  function drawScree(ctx, size, eigs) {
    const w = 130, h = 92;
    const x = 16;
    const y = 16;
    ctx.fillStyle = 'rgba(255,253,246,0.92)';
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('SCREE', x + 8, y + 14);

    const total = Math.max(1e-9, eigs[0].value + eigs[1].value);
    const baseY = y + h - 18;
    const maxBarH = h - 32;
    const barW = 28;
    const gap = 14;
    const startX = x + (w - 2*barW - gap) / 2;
    const lambdas = [eigs[0].value, eigs[1].value];
    const colors = [ACCENT, 'rgba(122,31,36,0.55)'];
    for (let i = 0; i < 2; i++) {
      const frac = Math.max(0, lambdas[i]) / total;
      const barH = maxBarH * (lambdas[i] / total);
      const bx = startX + i * (barW + gap);
      ctx.fillStyle = colors[i];
      ctx.fillRect(bx, baseY - barH, barW, barH);
      // Label: PCi underneath, % above
      ctx.fillStyle = INK;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText(`PC${i+1}`, bx + barW/2, baseY + 12);
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillStyle = colors[i];
      ctx.fillText(`${(frac*100).toFixed(0)}%`, bx + barW/2, baseY - barH - 4);
    }
  }

  // Rotated coordinate frame: each datum plotted by PC1 / PC2 score.
  function drawRotatedInset(ctx, size, pts, mean, eigs) {
    const w = 150, h = 110;
    const x = size.w - w - 16;
    const y = 16;
    ctx.fillStyle = 'rgba(255,253,246,0.92)';
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('PC₁ / PC₂ FRAME', x + 8, y + 14);

    const pc1 = eigs[0].vector, pc2 = eigs[1].vector;
    const [mx, my] = mean;

    // Compute scores; find their range to autoscale.
    let maxR = 0;
    const scores = pts.map(([px, py]) => {
      const dx = px - mx, dy = py - my;
      const s1 = dx*pc1[0] + dy*pc1[1];
      const s2 = dx*pc2[0] + dy*pc2[1];
      maxR = Math.max(maxR, Math.abs(s1), Math.abs(s2));
      return [s1, s2];
    });
    if (maxR < 1e-6) return;

    const cx = x + w/2;
    const cy = y + h/2 + 6;
    const plotR = Math.min(w, h - 24) * 0.42;
    const scale = plotR / maxR;

    // Axes
    ctx.strokeStyle = 'rgba(38,35,32,0.20)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - plotR, cy); ctx.lineTo(cx + plotR, cy);
    ctx.moveTo(cx, cy - plotR); ctx.lineTo(cx, cy + plotR);
    ctx.stroke();

    // Points
    ctx.fillStyle = INK;
    for (const [s1, s2] of scores) {
      ctx.beginPath();
      ctx.arc(cx + s1*scale, cy - s2*scale, 1.5, 0, Math.PI*2);
      ctx.fill();
    }
    // Axis labels
    ctx.fillStyle = ACCENT;
    ctx.font = 'italic 10px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'right';
    ctx.fillText('PC₁', cx + plotR - 2, cy + 12);
    ctx.fillStyle = 'rgba(122,31,36,0.7)';
    ctx.textAlign = 'left';
    ctx.fillText('PC₂', cx + 4, cy - plotR + 4);
  }

  // ───────── Caption + marginalia ─────────
  function captionFor(mode) {
    if (mode === 'explore') return `<span class="ml-cap-num">I.</span> Drag the slider below to rotate a candidate axis through the data. The thin grey lines are residuals — what each point loses by being projected onto your axis. Watch the variance readout: it peaks at one specific angle, and that angle is <span class="ml-ink-orange">PC₁</span>.`;
    if (mode === 'project') return `<span class="ml-cap-num">II.</span> Each point's distance to PC₁ is what we lose if we keep only its coordinate along that axis. The grey lines are these "errors" — by construction, no other line through the mean would make them shorter. The orange equation pinned along the line: <em>cov · v = λ · v</em> — PC₁ is literally the dominant eigenvector of the covariance matrix, with the data's variance along it as the eigenvalue.`;
    return `<span class="ml-cap-num">III.</span> Watch every point slide onto PC₁. The faded grey dots are the originals; the thin grey lines are the residuals — what was lost. <em>MSE</em> totals them up. By the Eckart–Young theorem, no other 1D projection beats this one.`;
  }
  function marginFor(mode) {
    let body = '';
    if (mode === 'explore') {
      body = `
        <p>You are doing PCA <em>by hand</em>: rotating a line and reading off how much variance falls along it. The angle that maximizes that number is, by definition, <em>PC₁</em>.</p>
        <p>The shortcut is to skip the sliding and compute it directly. The recipe:</p>
        <p class="quiet">(1) center the data on its mean.<br />(2) build the covariance matrix.<br />(3) find its eigenvectors.<br />(4) sort them by eigenvalue.</p>
        <p>Each eigenvalue is the variance along its eigenvector — the same number you watch climb in the readout. The little SCREE bars in the corner show λ₁ and λ₂ at a glance.</p>
        <div class="ml-margin-rule thin"></div>
        <div class="ml-margin-h sm">What is covariance?</div>
        <p class="quiet">The formula in the rail: <em>cov(X) = E[(x − μ)(x − μ)ᵀ]</em>. <em>μ</em> ("<em>mu</em>") is the mean. <em>x − μ</em> is "how far each point sits from the mean". <em>E[…]</em> is "average over the dataset". The diagonal entries are the variance of each feature on its own; the off-diagonals say how much each pair moves together. Its eigenvectors are the directions of greatest spread — and that is the entire trick.</p>`;
    } else if (mode === 'project') {
      body = `
        <p>PC₁ is the line that <em>minimizes total squared distance</em> from the points to itself. It is also, equivalently, the line of <em>maximum variance</em>. Both definitions give the same line.</p>
        <p>The orange equation pinned to PC₁ on the canvas — <em>cov · v = λ · v</em> — says PC₁ is exactly the dominant eigenvector of the covariance matrix, with eigenvalue λ = variance along the line.</p>
        <p class="quiet"><em>cov</em>: the covariance matrix, which captures how features co-vary. <em>v</em>: a candidate direction. <em>λ</em> ("<em>lambda</em>"): how much variance lies along that direction — bigger means more spread, more important. So the equation says: the directions where <em>cov · v</em> just rescales <em>v</em> are exactly the principal axes of the cloud.</p>
        <p class="quiet">The little PC₁/PC₂ frame in the corner is the same data plotted in PCA's coordinates: each point becomes (PC₁ score, PC₂ score). The cloud is now axis-aligned.</p>`;
    } else {
      body = `
        <p>We've done <em>dimensionality reduction</em>: 2 → 1. The thin grey lines show what was thrown away — the <em>reconstruction error</em>. The MSE readout sums it.</p>
        <p>The Eckart–Young theorem promises: no other 1D projection has lower MSE than projecting onto PC₁. Real PCA pipelines do this in 1000 dimensions instead of 2.</p>
        <p class="quiet">For a face dataset the leading principal components are called <em>eigenfaces</em>. For text, they reveal latent topics. Same algorithm, different data.</p>`;
    }
    return `
      <div class="ml-margin-rule"></div>
      <div class="ml-margin-h">In the margin</div>
      ${body}
      <div class="ml-margin-rule thin"></div>
      <div class="ml-margin-h sm">From eigenvectors</div>
      <p class="quiet">Covariance matrices are always symmetric — and symmetric matrices always have <em>orthogonal</em> eigenvectors. That's why the two arrows meet at a right angle.</p>`;
  }

  // ───────── Render ─────────
  function render() {
    const { mean, cov } = covariance(state.pts);
    const eigs = eigen2(cov);
    if (eigs.length) eigs.sort((a, b) => b.value - a.value);
    renderCovMatrix(cov);

    if (eigs.length) {
      readoutEl.querySelector('[data-k="l1"]').textContent = formatNum(eigs[0].value);
      readoutEl.querySelector('[data-k="l2"]').textContent = formatNum(eigs[1].value);
      const total = eigs[0].value + eigs[1].value;
      const ratio = total > 0 ? (eigs[0].value / total * 100) : 0;
      readoutEl.querySelector('[data-k="ratio"]').textContent = `${ratio.toFixed(0)}%`;
    } else {
      readoutEl.querySelector('[data-k="l1"]').textContent = '—';
      readoutEl.querySelector('[data-k="l2"]').textContent = '—';
      readoutEl.querySelector('[data-k="ratio"]').textContent = '—';
    }

    Array.from(presetsEl.children).forEach((btn, i) => {
      btn.classList.toggle('active', i === state.presetIdx);
    });
    Array.from(tabsEl.children).forEach((btn) => {
      btn.classList.toggle('on', btn.dataset.mode === state.mode);
    });

    captionEl.innerHTML = captionFor(state.mode);
    marginEl.innerHTML = marginFor(state.mode);

    if (state.mode === 'explore') {
      angleControls.style.display = 'flex';
      const deg = Math.round(state.testAngle * 180 / Math.PI);
      angleSlider.value = deg;
      angleLabel.textContent = `θ = ${deg}°`;
      const v = varianceAlong(state.pts, mean, state.testAngle);
      varianceLabel.textContent = `var = ${formatNum(v)}`;
    } else {
      angleControls.style.display = 'none';
    }

    canvasCtl.redraw();
  }

  render();
})();
