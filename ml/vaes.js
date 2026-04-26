/* Vol. XXI — A Continent of Latent Space (VAEs) */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK, INK_FADE, RULE_DARK } = V;

  // Synthetic 2D-latent → 8×8 image decoder. Bilinear blend of four anchor shapes
  // — stands in for what a trained decoder would learn from a small dataset.
  const ANCHORS = {
    cross:    [[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0]],
    square:   [[1,1,1,1,1,1,1,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,1,1,1,1,1,1,1]],
    circle:   [[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[1,1,0,0,0,0,1,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,1,0,0,0,0,1,1],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0]],
    triangle: [[0,0,0,1,1,0,0,0],[0,0,1,1,1,1,0,0],[0,0,1,1,1,1,0,0],[0,1,1,1,1,1,1,0],[0,1,1,1,1,1,1,0],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]],
  };

  function decode(z1, z2) {
    const u = 1 / (1 + Math.exp(-z1));
    const w = 1 / (1 + Math.exp(-z2));
    const out = [];
    for (let i = 0; i < 8; i++) {
      const row = [];
      for (let j = 0; j < 8; j++) {
        const v00 = ANCHORS.cross[i][j];
        const v10 = ANCHORS.square[i][j];
        const v01 = ANCHORS.circle[i][j];
        const v11 = ANCHORS.triangle[i][j];
        const v = (1 - u) * (1 - w) * v00 + u * (1 - w) * v10 + (1 - u) * w * v01 + u * w * v11;
        row.push(v);
      }
      out.push(row);
    }
    return out;
  }

  function gaussian() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // ───────── Views ─────────
  const VIEWS = [
    { key: 'kl',      label: 'KL · two distributions' },
    { key: 'reparam', label: 'Reparameterization' },
    { key: 'walk',    label: 'Walk the manifold' },
  ];
  const CAPTIONS = {
    kl:      'The orange ellipse is the encoder\'s distribution q(z|x) = N(μ, σ²) for the current input. The dashed circle is the prior N(0, I). KL divergence is the spring between them — it pulls μ toward 0 and σ toward 1, so all encoder outputs eventually overlap and tile latent space cleanly.',
    reparam: 'Sampling rebuilt as arithmetic. Draw ε ~ N(0, I) on the right, scale it by σ (stretch), translate by μ (shift) — and you land at z. The randomness is an "input," not part of the graph, so gradients flow through μ and σ during training.',
    walk:    'Two endpoints in latent space, connected by a line. Slide t to walk along it; the decoder paints each step. Smoothness means every intermediate point is plausible — that\'s the payoff for filling the space.',
  };

  // Two endpoint inputs for the walk; map to anchor positions in latent space.
  const ENDPOINTS = {
    A: { name: 'cross',  z: [-1.6, -1.6] },
    B: { name: 'triangle', z: [ 1.6,  1.6] },
  };

  const state = {
    view: 'kl',
    z1: 0,
    z2: 0,
    walkT: 0.5,
    // For KL view: encoder distribution starts off-prior, KL-step nudges it
    encMu: [1.4, 0.9],
    encSigma: [0.45, 0.65],
    // Reparameterization animation
    epsilon: [0.6, -0.4],
  };

  // ───────── DOM ─────────
  const viewToggleEl = document.getElementById('view-toggle');
  const z1Slider = document.getElementById('z1-slider');
  const z2Slider = document.getElementById('z2-slider');
  const z1Label = document.getElementById('z1-label');
  const z2Label = document.getElementById('z2-label');
  const sampleBtn = document.getElementById('sample-btn');
  const klControls = document.getElementById('kl-controls');
  const klStepBtn = document.getElementById('kl-step');
  const klResetBtn = document.getElementById('kl-reset');
  const walkControls = document.getElementById('walk-controls');
  const walkSlider = document.getElementById('walk-slider');
  const walkLabel = document.getElementById('walk-label');
  const readoutEl = document.getElementById('readout');
  const captionEl = document.getElementById('caption');
  const canvas = document.getElementById('canvas');

  VIEWS.forEach((v) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = v.label;
    b.dataset.view = v.key;
    b.addEventListener('click', () => { state.view = v.key; render(); });
    viewToggleEl.appendChild(b);
  });

  z1Slider.addEventListener('input', (e) => { state.z1 = +e.target.value; render(); });
  z2Slider.addEventListener('input', (e) => { state.z2 = +e.target.value; render(); });
  sampleBtn.addEventListener('click', () => {
    state.epsilon = [gaussian(), gaussian()];
    state.z1 = Math.max(-3, Math.min(3, state.encMu[0] + state.encSigma[0] * state.epsilon[0]));
    state.z2 = Math.max(-3, Math.min(3, state.encMu[1] + state.encSigma[1] * state.epsilon[1]));
    render();
  });

  klStepBtn && klStepBtn.addEventListener('click', () => {
    // Pull μ → 0, σ → 1 (the gradient of KL wrt these is well-known and goes that way)
    const r = 0.32;
    state.encMu = state.encMu.map((m) => m * (1 - r));
    state.encSigma = state.encSigma.map((s) => s + (1 - s) * r);
    render();
  });
  klResetBtn && klResetBtn.addEventListener('click', () => {
    state.encMu = [1.4, 0.9];
    state.encSigma = [0.45, 0.65];
    render();
  });
  walkSlider && walkSlider.addEventListener('input', (e) => {
    state.walkT = +e.target.value;
    render();
  });

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: true });

  function drawCell(ctx, map, x0, y0, cell) {
    for (let i = 0; i < 8; i++) {
      for (let j = 0; j < 8; j++) {
        const t = Math.max(0, Math.min(1, map[i][j]));
        const r = Math.round(247 - t * 200);
        const g = Math.round(244 - t * 200);
        const b = Math.round(236 - t * 220);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x0 + j * cell, y0 + i * cell, cell + 0.5, cell + 0.5);
      }
    }
    ctx.strokeStyle = 'rgba(38,35,32,0.25)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x0, y0, cell * 8, cell * 8);
  }

  function draw(ctx, size) {
    if (state.view === 'kl') return drawKL(ctx, size);
    if (state.view === 'reparam') return drawReparam(ctx, size);
    return drawWalk(ctx, size);
  }

  // ─────────────── View 1: Two distributions + KL spring ───────────────
  function drawKL(ctx, size) {
    // Center the latent plane in canvas
    const margin = 56;
    const W = size.w - margin * 2;
    const H = size.h - margin * 2;
    const cx = margin + W / 2;
    const cy = margin + H / 2;
    // Map z ∈ [-3, 3] to pixels
    const sx = W / 6;
    const sy = H / 6;
    const tx = (z) => cx + z * sx;
    const ty = (z) => cy - z * sy;

    // Title
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('encoder distribution q(z|x) vs prior N(0, I)', size.w / 2, 22);

    // Light grid
    ctx.strokeStyle = 'rgba(38,35,32,0.06)';
    ctx.lineWidth = 1;
    for (let g = -3; g <= 3; g++) {
      ctx.beginPath(); ctx.moveTo(tx(g), ty(-3)); ctx.lineTo(tx(g), ty(3)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(tx(-3), ty(g)); ctx.lineTo(tx(3), ty(g)); ctx.stroke();
    }
    // Axes
    ctx.strokeStyle = 'rgba(38,35,32,0.18)';
    ctx.beginPath(); ctx.moveTo(tx(-3), cy); ctx.lineTo(tx(3), cy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx, ty(-3)); ctx.lineTo(cx, ty(3)); ctx.stroke();

    // Prior N(0,I): unit-radius dashed circle (one std)
    ctx.strokeStyle = 'rgba(38,35,32,0.55)';
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.ellipse(cx, cy, sx, sy, 0, 0, Math.PI * 2);
    ctx.stroke();
    // 2σ outer
    ctx.setLineDash([2, 4]);
    ctx.strokeStyle = 'rgba(38,35,32,0.25)';
    ctx.beginPath();
    ctx.ellipse(cx, cy, sx * 2, sy * 2, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    // Prior label
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('N(0, I)  prior', cx + sx + 6, cy - sy - 6);
    // Origin dot
    ctx.fillStyle = 'rgba(38,35,32,0.55)';
    ctx.beginPath(); ctx.arc(cx, cy, 2.5, 0, Math.PI * 2); ctx.fill();

    // Encoder ellipse: filled at 1σ (orange tinted)
    const muX = tx(state.encMu[0]), muY = ty(state.encMu[1]);
    const eSx = state.encSigma[0] * sx;
    const eSy = state.encSigma[1] * sy;
    ctx.fillStyle = 'rgba(122,31,36,0.18)';
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.ellipse(muX, muY, eSx, eSy, 0, 0, Math.PI * 2);
    ctx.fill(); ctx.stroke();
    // 2σ
    ctx.strokeStyle = 'rgba(122,31,36,0.30)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.ellipse(muX, muY, eSx * 2, eSy * 2, 0, 0, Math.PI * 2);
    ctx.stroke();
    // mean dot
    ctx.fillStyle = ACCENT;
    ctx.beginPath(); ctx.arc(muX, muY, 3.5, 0, Math.PI * 2); ctx.fill();

    // μ label
    ctx.fillStyle = ACCENT;
    ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('μ', muX + 8, muY - 8);
    // q(z|x) label
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.fillText('q(z|x) = N(μ, σ²)', muX + eSx + 8, muY + eSy + 14);

    // Spring/arrow from encoder mean → origin (the "KL pull")
    drawSpring(ctx, muX, muY, cx, cy, 8);
    // KL value (computed exactly for diagonal Gaussians vs N(0,I))
    const klVal = klDiagonalGauss(state.encMu, state.encSigma);
    const midX = (muX + cx) / 2;
    const midY = (muY + cy) / 2;
    ctx.fillStyle = INK;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText(`KL = ${formatNum(klVal)}`, midX + 18, midY - 6);
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillText('KULLBACK–LEIBLER', midX + 18, midY + 8);
    // Math anchored to the picture: the closed-form KL for diag-Gauss vs N(0,I)
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.fillStyle = INK_FADE;
    ctx.fillText('½ Σ ( σ² + μ² − 1 − log σ² )', midX + 18, midY + 24);

    // Hint
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.fillStyle = INK_FADE;
    ctx.textAlign = 'center';
    ctx.fillText('press ⇨ Improve KL — watch the orange ellipse get pulled onto the dashed prior',
      size.w / 2, size.h - 10);

    // Symbol gloss (small inline)
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillStyle = INK_FADE;
    ctx.fillText('μ MEAN · σ SPREAD · q(z|x) ENCODER OUTPUT', 12, size.h - 28);
  }

  function klDiagonalGauss(mu, sigma) {
    // KL(N(mu, sigma^2) || N(0,I)) = 0.5 * Σ ( σ² + μ² − 1 − log σ² )
    let s = 0;
    for (let i = 0; i < mu.length; i++) {
      const v = sigma[i] * sigma[i];
      s += v + mu[i] * mu[i] - 1 - Math.log(v + 1e-9);
    }
    return 0.5 * s;
  }

  function drawSpring(ctx, x1, y1, x2, y2, coils) {
    // A simple zigzag spring
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    if (len < 4) return;
    const ux = dx / len, uy = dy / len;
    const px = -uy, py = ux;
    const amp = 5;
    ctx.strokeStyle = 'rgba(38,35,32,0.45)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    for (let i = 1; i <= coils; i++) {
      const t = i / (coils + 1);
      const cxp = x1 + dx * t + px * amp * (i % 2 === 0 ? 1 : -1);
      const cyp = y1 + dy * t + py * amp * (i % 2 === 0 ? 1 : -1);
      ctx.lineTo(cxp, cyp);
    }
    ctx.lineTo(x2, y2);
    ctx.stroke();
    // Arrowhead at origin end
    const ang = Math.atan2(-uy, -ux);
    ctx.fillStyle = 'rgba(38,35,32,0.55)';
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - 7 * Math.cos(ang - 0.4), y2 - 7 * Math.sin(ang - 0.4));
    ctx.lineTo(x2 - 7 * Math.cos(ang + 0.4), y2 - 7 * Math.sin(ang + 0.4));
    ctx.fill();
  }

  // ─────────────── View 2: Reparameterization (vector sum) ───────────────
  function drawReparam(ctx, size) {
    // Title
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('reparameterization · z = μ + σ · ε', size.w / 2, 22);

    // Layout: three "vector" panels stacked horizontally
    // [ μ vector ]  +  [ σ ⊙ ε vector ]  =  [ z ]
    const padX = 28;
    const panelW = (size.w - padX * 4) / 3;
    const panelH = 90;
    const yTop = 80;

    const labels = ['μ', 'σ · ε', 'z = μ + σ · ε'];
    const colors = [ACCENT, 'rgba(38,35,32,0.85)', INK];

    // Compute the two vectors per dim
    const dim = 2;
    const muVec = state.encMu;
    const epsVec = state.epsilon;
    const sigVec = state.encSigma;
    const sigEps = sigVec.map((s, i) => s * epsVec[i]);
    const zVec = muVec.map((m, i) => m + sigEps[i]);

    const vecs = [muVec, sigEps, zVec];

    for (let p = 0; p < 3; p++) {
      const x0 = padX + p * (panelW + padX);
      // panel border
      ctx.strokeStyle = 'rgba(38,35,32,0.20)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x0, yTop, panelW, panelH);
      // header label
      ctx.fillStyle = colors[p];
      ctx.font = 'italic 18px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText(labels[p], x0 + panelW / 2, yTop - 14);
      // bars for each entry
      const v = vecs[p];
      const vMax = 3;
      for (let i = 0; i < dim; i++) {
        const cy = yTop + 24 + i * 32;
        // axis line
        ctx.strokeStyle = 'rgba(38,35,32,0.18)';
        ctx.beginPath();
        ctx.moveTo(x0 + 12, cy);
        ctx.lineTo(x0 + panelW - 12, cy);
        ctx.stroke();
        const midX = x0 + panelW / 2;
        // tick
        ctx.beginPath();
        ctx.moveTo(midX, cy - 4); ctx.lineTo(midX, cy + 4);
        ctx.stroke();
        // bar
        const lenScale = (panelW / 2 - 18) / vMax;
        const blen = Math.max(-vMax, Math.min(vMax, v[i])) * lenScale;
        ctx.fillStyle = colors[p];
        ctx.fillRect(midX, cy - 5, blen, 10);
        // value
        ctx.fillStyle = INK_FADE;
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`${i === 0 ? 'z₁' : 'z₂'}`, x0 + 6, cy + 3);
        ctx.textAlign = 'right';
        ctx.fillText(formatNum(v[i]), x0 + panelW - 6, cy + 3);
      }
    }

    // Plus / equals between panels
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 22px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('+', padX + panelW + padX / 2, yTop + panelH / 2 + 6);
    ctx.fillText('=', padX * 2 + panelW * 2 + padX / 2, yTop + panelH / 2 + 6);

    // Below: 2D plane showing μ → z arrow on the latent plane
    const planeY = yTop + panelH + 30;
    const planeH = size.h - planeY - 80;
    const planeMargin = 40;
    const W = size.w - planeMargin * 2;
    const H = planeH;
    const pcx = planeMargin + W / 2;
    const pcy = planeY + H / 2;
    const sx = (Math.min(W, H) - 20) / 6;
    const sy = sx;
    const tx = (z) => pcx + z * sx;
    const ty = (z) => pcy - z * sy;

    // grid + axes
    ctx.strokeStyle = 'rgba(38,35,32,0.06)';
    for (let g = -3; g <= 3; g++) {
      ctx.beginPath(); ctx.moveTo(tx(g), ty(-3)); ctx.lineTo(tx(g), ty(3)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(tx(-3), ty(g)); ctx.lineTo(tx(3), ty(g)); ctx.stroke();
    }
    ctx.strokeStyle = 'rgba(38,35,32,0.18)';
    ctx.beginPath(); ctx.moveTo(tx(-3), pcy); ctx.lineTo(tx(3), pcy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pcx, ty(-3)); ctx.lineTo(pcx, ty(3)); ctx.stroke();

    // prior N(0,I) circle
    ctx.strokeStyle = 'rgba(38,35,32,0.45)';
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.ellipse(pcx, pcy, sx, sy, 0, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);

    // ε dot in left side of the plot at small scale (we draw it twice — once at "raw" location, once shown being scaled)
    const epsX = tx(epsVec[0]), epsY = ty(epsVec[1]);
    ctx.fillStyle = 'rgba(38,35,32,0.7)';
    ctx.beginPath(); ctx.arc(epsX, epsY, 3, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('ε ~ N(0, I)', epsX + 6, epsY - 4);

    // arrow from origin → μ
    drawArrow(ctx, pcx, pcy, tx(muVec[0]), ty(muVec[1]), ACCENT);
    ctx.fillStyle = ACCENT;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.fillText('μ', tx(muVec[0]) + 6, ty(muVec[1]) - 4);

    // arrow from μ → z (this is + σ·ε)
    drawArrow(ctx, tx(muVec[0]), ty(muVec[1]), tx(zVec[0]), ty(zVec[1]), 'rgba(38,35,32,0.70)');
    ctx.fillStyle = INK;
    ctx.beginPath(); ctx.arc(tx(zVec[0]), ty(zVec[1]), 4, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = INK;
    ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
    ctx.fillText('z', tx(zVec[0]) + 8, ty(zVec[1]) + 4);
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.fillText('  ← μ + σ·ε', tx(zVec[0]) + 8, ty(zVec[1]) + 18);

    // hint
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('press ↻ Sample to draw a fresh ε from the prior', size.w / 2, size.h - 10);

    // glossary
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillStyle = INK_FADE;
    ctx.fillText('ε EPSILON · NOISE FROM PRIOR  ·  σ ⊙ ε STRETCH NOISE BY ENCODER SPREAD', 12, size.h - 42);
    ctx.fillText('WHY: GRADIENTS CAN\'T FLOW THROUGH A RANDOM DRAW — BUT THEY CAN FLOW THROUGH μ + σ·ε', 12, size.h - 28);
  }

  function drawArrow(ctx, x1, y1, x2, y2, color) {
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    ctx.stroke();
    const ang = Math.atan2(y2 - y1, x2 - x1);
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - 8 * Math.cos(ang - 0.4), y2 - 8 * Math.sin(ang - 0.4));
    ctx.lineTo(x2 - 8 * Math.cos(ang + 0.4), y2 - 8 * Math.sin(ang + 0.4));
    ctx.fill();
  }

  // ─────────────── View 3: Walk along latent line ───────────────
  function drawWalk(ctx, size) {
    // Title
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('latent-space line · decode each step', size.w / 2, 22);

    // Top half: 2D latent plane with line A→B and current t mark
    const topH = Math.floor(size.h * 0.50);
    const padY = 36;
    const W = size.w - 80;
    const H = topH - padY - 8;
    const cx = 40 + W / 2;
    const cy = padY + H / 2;
    const sx = W / 6, sy = H / 4;
    const tx = (z) => cx + z * sx;
    const ty = (z) => cy - z * sy;

    // grid
    ctx.strokeStyle = 'rgba(38,35,32,0.06)';
    for (let g = -3; g <= 3; g++) {
      ctx.beginPath(); ctx.moveTo(tx(g), ty(-2)); ctx.lineTo(tx(g), ty(2)); ctx.stroke();
    }
    for (let g = -2; g <= 2; g++) {
      ctx.beginPath(); ctx.moveTo(tx(-3), ty(g)); ctx.lineTo(tx(3), ty(g)); ctx.stroke();
    }
    // axes
    ctx.strokeStyle = 'rgba(38,35,32,0.18)';
    ctx.beginPath(); ctx.moveTo(tx(-3), cy); ctx.lineTo(tx(3), cy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx, ty(-2)); ctx.lineTo(cx, ty(2)); ctx.stroke();

    // prior dashed circle
    ctx.strokeStyle = 'rgba(38,35,32,0.40)';
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.ellipse(cx, cy, sx, sy, 0, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);

    // endpoints A, B
    const A = ENDPOINTS.A.z, B = ENDPOINTS.B.z;
    const Ax = tx(A[0]), Ay = ty(A[1]);
    const Bx = tx(B[0]), By = ty(B[1]);
    // line
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(Ax, Ay); ctx.lineTo(Bx, By);
    ctx.stroke();
    // endpoints
    ctx.fillStyle = ACCENT;
    ctx.beginPath(); ctx.arc(Ax, Ay, 5, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(Bx, By, 5, 0, Math.PI * 2); ctx.fill();
    // labels
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.fillStyle = ACCENT;
    ctx.textAlign = 'right';
    ctx.fillText(`A · ${ENDPOINTS.A.name}`, Ax - 8, Ay + 4);
    ctx.textAlign = 'left';
    ctx.fillText(`B · ${ENDPOINTS.B.name}`, Bx + 8, By - 6);

    // current point at t
    const t = state.walkT;
    const cur = [A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t];
    const curX = tx(cur[0]), curY = ty(cur[1]);
    ctx.fillStyle = INK;
    ctx.beginPath(); ctx.arc(curX, curY, 5, 0, Math.PI * 2); ctx.fill();
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillStyle = INK;
    ctx.fillText(`t = ${formatNum(t)}`, curX + 8, curY + 14);

    // Bottom half: filmstrip — 7 decoded outputs along the line
    const stripY = topH + 24;
    const stripH = size.h - stripY - 30;
    const N = 7;
    const tileM = 6;
    const tileSize = (size.w - tileM * (N + 1)) / N;
    const cellSz = tileSize / 8;
    for (let i = 0; i < N; i++) {
      const ti = i / (N - 1);
      const z1 = A[0] + (B[0] - A[0]) * ti;
      const z2 = A[1] + (B[1] - A[1]) * ti;
      const out = decode(z1, z2);
      const x0 = tileM + i * (tileSize + tileM);
      const y0 = stripY;
      drawCell(ctx, out, x0, y0, cellSz);
      // highlight closest to t
      const closest = Math.round(t * (N - 1));
      if (i === closest) {
        ctx.strokeStyle = ACCENT;
        ctx.lineWidth = 2;
        ctx.strokeRect(x0 - 2, y0 - 2, tileSize + 4, tileSize + 4);
      }
      ctx.fillStyle = INK_FADE;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`t=${formatNum(ti)}`, x0 + tileSize / 2, y0 + tileSize + 14);
    }

    // hint
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('every intermediate decode is plausible — that\'s the smoothness KL bought you',
      size.w / 2, size.h - 8);
  }

  // ───────── Render ─────────
  function render() {
    Array.from(viewToggleEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.view === state.view);
    });

    // Show/hide controls per view
    if (klControls) klControls.style.display = state.view === 'kl' ? 'block' : 'none';
    if (walkControls) walkControls.style.display = state.view === 'walk' ? 'block' : 'none';
    // z sliders are useful for reparam (manual μ-like control); for kl/walk we just leave them
    // Hide sample button when KL view (it samples z which doesn't apply mid-tug)
    sampleBtn.style.display = state.view === 'reparam' ? 'block' : 'none';

    if (z1Slider) z1Slider.value = state.z1;
    if (z2Slider) z2Slider.value = state.z2;
    if (z1Label) z1Label.textContent = `z₁ = ${formatNum(state.z1)}`;
    if (z2Label) z2Label.textContent = `z₂ = ${formatNum(state.z2)}`;
    if (walkSlider) walkSlider.value = state.walkT;
    if (walkLabel) walkLabel.textContent = `t = ${formatNum(state.walkT)}`;

    if (captionEl) captionEl.innerHTML = '<span class="ml-cap-num">·</span> ' + CAPTIONS[state.view];

    // Readout — different per view
    if (state.view === 'kl') {
      const klVal = klDiagonalGauss(state.encMu, state.encSigma);
      readoutEl.innerHTML = `
        <div class="row"><span>μ₁ (mean)</span><b>${formatNum(state.encMu[0])}</b></div>
        <div class="row"><span>μ₂ (mean)</span><b>${formatNum(state.encMu[1])}</b></div>
        <div class="row"><span>σ₁ (spread)</span><b>${formatNum(state.encSigma[0])}</b></div>
        <div class="row"><span>σ₂ (spread)</span><b>${formatNum(state.encSigma[1])}</b></div>
        <div class="row"><span>KL divergence</span><b>${formatNum(klVal)}</b></div>`;
    } else if (state.view === 'reparam') {
      const z = [state.encMu[0] + state.encSigma[0] * state.epsilon[0],
                 state.encMu[1] + state.encSigma[1] * state.epsilon[1]];
      readoutEl.innerHTML = `
        <div class="row"><span>ε₁ (noise)</span><b>${formatNum(state.epsilon[0])}</b></div>
        <div class="row"><span>ε₂ (noise)</span><b>${formatNum(state.epsilon[1])}</b></div>
        <div class="row"><span>μ + σ·ε</span><b>(${formatNum(z[0])}, ${formatNum(z[1])})</b></div>`;
    } else {
      const t = state.walkT;
      const A = ENDPOINTS.A.z, B = ENDPOINTS.B.z;
      const cur = [A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t];
      readoutEl.innerHTML = `
        <div class="row"><span>endpoint A</span><b>${ENDPOINTS.A.name}</b></div>
        <div class="row"><span>endpoint B</span><b>${ENDPOINTS.B.name}</b></div>
        <div class="row"><span>t</span><b>${formatNum(t)}</b></div>
        <div class="row"><span>z(t)</span><b>(${formatNum(cur[0])}, ${formatNum(cur[1])})</b></div>`;
    }

    canvasCtl.redraw();
  }

  render();
})();
