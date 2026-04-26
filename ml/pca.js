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

  // Read-only covariance display (no scrubbing).
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

  // ───────── Animation: project sliding 0 → 1 in 'reduce' mode ─────────
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

    // ── Reference: PC1 dashed line, faint in 'explore' (so user can compare),
    //    visible in 'project' / 'reduce' as the actual answer. ──
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

    // ── 'explore' mode: candidate axis at state.testAngle, with residuals to it.
    //    PC1's optimal angle is shown as a much fainter dashed reference for comparison. ──
    if (eigs.length && state.mode === 'explore') {
      const pc1 = eigs[0].vector;
      // Faint PC1 reference (the optimum the user is searching for)
      ctx.strokeStyle = 'rgba(122,31,36,0.18)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 5]);
      ctx.beginPath();
      ctx.moveTo(...toPx([mx - pc1[0]*6, my - pc1[1]*6]));
      ctx.lineTo(...toPx([mx + pc1[0]*6, my + pc1[1]*6]));
      ctx.stroke();
      ctx.setLineDash([]);

      // Candidate axis
      const ax = Math.cos(state.testAngle);
      const ay = Math.sin(state.testAngle);

      // Residuals from each point to the candidate axis (visible "loss")
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

      // Candidate axis line itself (prominent)
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(...toPx([mx - ax*6, my - ay*6]));
      ctx.lineTo(...toPx([mx + ax*6, my + ay*6]));
      ctx.stroke();
    }

    // Points (animated for 'reduce')
    ctx.fillStyle = INK;
    if (eigs.length && state.mode === 'reduce') {
      const pc1 = eigs[0].vector;
      const tt = state.projT;
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

    // PC1/PC2 arrows: shown in 'project' and 'reduce'. In 'explore' the user's
    // candidate axis is what they're driving, so the static arrows would give the answer away.
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
    }

    // Footer info
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
      } else {
        const total = eigs[0].value + eigs[1].value;
        const explained = total > 0 ? eigs[0].value / total : 0;
        ctx.fillText(`PC₁ explains ${(explained*100).toFixed(1)}% of variance`,
          size.w - 18, size.h - 22);
      }
    }
  }

  // ───────── Caption + marginalia ─────────
  function captionFor(mode) {
    if (mode === 'explore') return `<span class="ml-cap-num">I.</span> Drag the slider below to rotate a candidate axis through the data. The thin grey lines are residuals — what each point loses by being projected onto your axis. Watch the variance readout: it peaks at one specific angle, and that angle is <span class="ml-ink-orange">PC₁</span>.`;
    if (mode === 'project') return `<span class="ml-cap-num">II.</span> Each point's distance to PC₁ is what we lose if we keep only its coordinate along that axis. The grey lines are these "errors" — by construction, no other line through the mean would make them shorter.`;
    return `<span class="ml-cap-num">III.</span> Watch every point slide onto the principal axis. That single number per point is its <em>PC₁ score</em> — a 2D dataset compressed to 1D with the least information lost.`;
  }
  function marginFor(mode) {
    let body = '';
    if (mode === 'explore') {
      body = `
        <p>You are doing PCA <em>by hand</em>: rotating a line and reading off how much variance falls along it. The angle that maximizes that number is, by definition, <em>PC₁</em>.</p>
        <p>The shortcut is to skip the sliding and compute it directly. The recipe:</p>
        <p class="quiet">(1) center the data on its mean.<br />(2) build the covariance matrix.<br />(3) find its eigenvectors.<br />(4) sort them by eigenvalue.</p>
        <p>Each eigenvalue is the variance along its eigenvector — the same number you watch climb in the readout.</p>`;
    } else if (mode === 'project') {
      body = `
        <p>PC₁ is the line that <em>minimizes total squared distance</em> from the points to itself. It is also, equivalently, the line of <em>maximum variance</em>. Both definitions give the same line.</p>
        <p class="quiet">This connection between "best fit" and "most variance" is one of the quiet beauties of linear algebra.</p>`;
    } else {
      body = `
        <p>We've done <em>dimensionality reduction</em>: 2 → 1. In real applications the same idea takes 10,000-dimensional images down to 50.</p>
        <p class="quiet">For a face dataset, the leading principal components are called <em>eigenfaces</em>. For text, they reveal latent topics. Same algorithm, different data.</p>`;
    }
    return `
      <div class="ml-margin-rule"></div>
      <div class="ml-margin-h">In the margin</div>
      ${body}
      <div class="ml-margin-rule thin"></div>
      <div class="ml-margin-h sm">From the previous chapter</div>
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

    // Angle slider only in 'explore' mode.
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
