/* Vol. XX — The Bottleneck (Autoencoders) */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK, INK_FADE, RULE_DARK } = V;

  // Five 8×8 binary digits. Just enough variety to demo encoder/decoder structure.
  const SHAPES = {
    cross: [[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0]],
    circle:[[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[1,1,0,0,0,0,1,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,1,0,0,0,0,1,1],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0]],
    square:[[1,1,1,1,1,1,1,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,1,1,1,1,1,1,1]],
    triangle:[[0,0,0,1,1,0,0,0],[0,0,1,1,1,1,0,0],[0,0,1,1,1,1,0,0],[0,1,1,1,1,1,1,0],[0,1,1,1,1,1,1,0],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]],
    noisy_circle: null,
  };

  function genNoisy(rngSeed) {
    let s = rngSeed;
    const rnd = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
    const base = SHAPES.circle;
    return base.map((row) => row.map((v) => Math.max(0, Math.min(1, v + (rnd() - 0.5) * 0.6))));
  }

  // Simulated AE via truncated low-frequency basis (acts like PCA on this toy data).
  function svdLikeReconstruct(matrix, k) {
    const N = matrix.length;
    const M = matrix[0].length;
    const flat = matrix.flat();
    const basis = [];
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        const b = [];
        for (let r = 0; r < N; r++) {
          for (let c = 0; c < M; c++) {
            b.push(Math.cos((Math.PI * (i + 0.5) * (r + 0.5)) / N) *
                   Math.cos((Math.PI * (j + 0.5) * (c + 0.5)) / M));
          }
        }
        const norm = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
        basis.push(b.map((v) => v / norm));
      }
    }
    const used = basis.slice(0, k);
    const coeffs = used.map((b) => flat.reduce((s, v, i) => s + v * b[i], 0));
    const recon = new Array(flat.length).fill(0);
    used.forEach((b, idx) => {
      const c = coeffs[idx];
      for (let i = 0; i < recon.length; i++) recon[i] += c * b[i];
    });
    const out = [];
    for (let i = 0; i < N; i++) out.push(recon.slice(i * M, (i + 1) * M));
    return { recon: out, coeffs };
  }

  // Project a shape into 2 latent dims (for scatter).
  function latent2D(matrix) {
    const { coeffs } = svdLikeReconstruct(matrix, 2);
    return coeffs;
  }

  const VIEWS = [
    { key: 'funnel',   label: 'Funnel · forward pass' },
    { key: 'denoise',  label: 'Denoising triptych' },
    { key: 'scatter',  label: 'Latent scatter (k = 2)' },
  ];
  const CAPTIONS = {
    funnel:  'A funnel: input pixels condense through narrowing layers to a tight latent of k numbers, then expand back out. The waist is exactly k wide — that is the bottleneck.',
    denoise: 'Add noise to the input; the AE projects the noisy version back onto the manifold of plausible shapes. The latent has no room for noise — only for signal.',
    scatter: 'With k = 2 the latent space is a plane. Each preset lands at one dot. Classes separate without ever being told the labels — unsupervised structure.',
  };

  const state = {
    inputKey: 'cross',
    k: 4,
    noiseSeed: 1,
    view: 'funnel',
    flowT: 1,        // 0..1 animation parameter for the funnel
    flowing: false,
    flowTimer: null,
  };

  const inputsEl = document.getElementById('inputs');
  const kSlider = document.getElementById('k-slider');
  const kLabel = document.getElementById('k-label');
  const runBtn = document.getElementById('run-btn');
  const tabsEl = document.getElementById('view-tabs');
  const captionEl = document.getElementById('caption-text');
  const readoutEl = document.getElementById('readout');
  const canvas = document.getElementById('canvas');

  ['cross', 'circle', 'square', 'triangle', 'noisy_circle'].forEach((k) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = k.replace('_', ' ');
    b.dataset.input = k;
    b.addEventListener('click', () => {
      state.inputKey = k;
      // auto-switch to denoise view if user picks noisy
      if (k === 'noisy_circle') state.view = 'denoise';
      runFlow();
    });
    inputsEl.appendChild(b);
  });

  VIEWS.forEach((v) => {
    const b = document.createElement('button');
    b.textContent = v.label;
    b.dataset.view = v.key;
    b.addEventListener('click', () => {
      state.view = v.key;
      if (v.key === 'funnel') runFlow();
      else render();
    });
    tabsEl.appendChild(b);
  });

  kSlider.addEventListener('input', (e) => {
    state.k = +e.target.value;
    kLabel.textContent = `k = ${state.k}`;
    render();
  });

  runBtn.addEventListener('click', () => { state.view = 'funnel'; runFlow(); });

  function runFlow() {
    if (state.flowTimer) { clearInterval(state.flowTimer); state.flowTimer = null; }
    state.flowing = true;
    state.flowT = 0;
    state.flowTimer = setInterval(() => {
      state.flowT += 0.025;
      if (state.flowT >= 1) {
        state.flowT = 1;
        state.flowing = false;
        clearInterval(state.flowTimer);
        state.flowTimer = null;
      }
      render();
    }, 30);
  }

  function getInput() {
    if (state.inputKey === 'noisy_circle') return genNoisy(state.noiseSeed);
    return SHAPES[state.inputKey];
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function drawGrid(ctx, map, x0, y0, cell) {
    const N = map.length;
    const M = map[0].length;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < M; j++) {
        const v = Math.max(-1, Math.min(2, map[i][j]));
        const t = Math.max(0, Math.min(1, v));
        const r = Math.round(247 - t * 200);
        const g = Math.round(244 - t * 200);
        const b = Math.round(236 - t * 220);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x0 + j * cell, y0 + i * cell, cell + 0.5, cell + 0.5);
      }
    }
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x0, y0, M * cell, N * cell);
  }

  function draw(ctx, size) {
    if (state.view === 'denoise') return drawDenoise(ctx, size);
    if (state.view === 'scatter') return drawScatter(ctx, size);
    return drawFunnel(ctx, size);
  }

  // ───────── View 1: Funnel architecture with flowing particles ─────────
  function drawFunnel(ctx, size) {
    const input = getInput();
    const { recon, coeffs } = svdLikeReconstruct(input, state.k);

    // Layout: input grid · funnel encoder · bottleneck · funnel decoder · output grid
    const padX = 18;
    const inputCell = (size.h - 90) / 8;
    const inputW = inputCell * 8;
    const yMid = size.h / 2;
    const funnelY = 30;
    const funnelH = size.h - 90;

    const inputX = padX;
    const encX1 = inputX + inputW + 8;
    const encX2 = encX1 + (size.w - 2 * padX - 2 * inputW - 16) * 0.42;
    const decX1 = encX2 + (size.w - 2 * padX - 2 * inputW - 16) * 0.16;
    const decX2 = decX1 + (size.w - 2 * padX - 2 * inputW - 16) * 0.42;
    const reconX = decX2 + 8;
    const waistY1 = yMid - 16;
    const waistY2 = yMid + 16;
    // bottleneck center
    const bX = (encX2 + decX1) / 2;

    // 1. INPUT grid
    drawGrid(ctx, input, inputX, funnelY, inputCell);
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('input · 8×8 = 64', inputX + inputW / 2, funnelY + inputW + 16);
    ctx.font = 'italic 14px "Source Serif 4", Georgia, serif';
    ctx.fillStyle = INK;
    ctx.fillText('x', inputX + inputW / 2, funnelY - 8);

    // 2. ENCODER funnel — wide left → narrow right
    const encWidthLeft = inputW;
    const waistH = Math.max(state.k * 4, 12);
    ctx.fillStyle = 'rgba(122,31,36,0.07)';
    ctx.strokeStyle = 'rgba(122,31,36,0.45)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(encX1, funnelY);
    ctx.lineTo(encX2, yMid - waistH / 2);
    ctx.lineTo(encX2, yMid + waistH / 2);
    ctx.lineTo(encX1, funnelY + funnelH);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Tier markers (suggesting layers)
    ctx.strokeStyle = 'rgba(122,31,36,0.18)';
    ctx.setLineDash([2, 3]);
    for (let t = 0.25; t < 1; t += 0.25) {
      const x = encX1 + (encX2 - encX1) * t;
      const topY = funnelY + (yMid - waistH / 2 - funnelY) * t;
      const botY = (funnelY + funnelH) - (funnelY + funnelH - yMid - waistH / 2) * t;
      ctx.beginPath();
      ctx.moveTo(x, topY);
      ctx.lineTo(x, botY);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('ENCODER', (encX1 + encX2) / 2, funnelY + funnelH + 28);

    // 3. BOTTLENECK — k vertical cells
    const latMax = Math.max(...coeffs.slice(0, state.k).map(Math.abs)) + 1e-6;
    const latCellH = waistH * 1.6 / Math.max(state.k, 4);
    const latH = latCellH * state.k;
    const latX = bX - 6;
    const latY = yMid - latH / 2;
    coeffs.slice(0, state.k).forEach((c, i) => {
      const intensity = Math.min(1, Math.abs(c) / latMax);
      ctx.fillStyle = c >= 0
        ? `rgba(122,31,36,${0.18 + intensity * 0.7})`
        : `rgba(38,35,32,${0.18 + intensity * 0.7})`;
      ctx.fillRect(latX, latY + i * latCellH, 12, latCellH - 1);
    });
    ctx.strokeStyle = 'rgba(38,35,32,0.45)';
    ctx.lineWidth = 1;
    ctx.strokeRect(latX, latY, 12, latH);

    ctx.fillStyle = ACCENT;
    ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('z', bX, latY - 6);
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillText(`k = ${state.k}`, bX, latY + latH + 14);

    // 4. DECODER funnel — narrow → wide
    ctx.fillStyle = 'rgba(122,31,36,0.07)';
    ctx.strokeStyle = 'rgba(122,31,36,0.45)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(decX1, yMid - waistH / 2);
    ctx.lineTo(decX2, funnelY);
    ctx.lineTo(decX2, funnelY + funnelH);
    ctx.lineTo(decX1, yMid + waistH / 2);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.strokeStyle = 'rgba(122,31,36,0.18)';
    ctx.setLineDash([2, 3]);
    for (let t = 0.25; t < 1; t += 0.25) {
      const x = decX1 + (decX2 - decX1) * t;
      const topY = (yMid - waistH / 2) - (yMid - waistH / 2 - funnelY) * t;
      const botY = (yMid + waistH / 2) + (funnelY + funnelH - yMid - waistH / 2) * t;
      ctx.beginPath();
      ctx.moveTo(x, topY);
      ctx.lineTo(x, botY);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('DECODER', (decX1 + decX2) / 2, funnelY + funnelH + 28);

    // 5. RECONSTRUCTION grid — fade in proportional to flowT
    ctx.save();
    ctx.globalAlpha = Math.min(1, state.flowT * 1.3);
    drawGrid(ctx, recon, reconX, funnelY, inputCell);
    ctx.restore();
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('reconstruction', reconX + inputW / 2, funnelY + inputW + 16);
    ctx.font = 'italic 14px "Source Serif 4", Georgia, serif';
    ctx.fillStyle = INK;
    ctx.fillText('x̂', reconX + inputW / 2, funnelY - 8);

    // 6. PARTICLES — animated flow
    const T = state.flowT;
    const NPARTS = 28;
    for (let p = 0; p < NPARTS; p++) {
      const pT = (T - p * 0.018) * 1.4;
      if (pT < 0 || pT > 1.2) continue;
      const phase = Math.max(0, Math.min(1, pT));
      // x-position interpolates input grid → encoder → bottleneck → decoder → reconstruction
      let px, py;
      const lane = (p / (NPARTS - 1)) - 0.5; // -0.5 .. 0.5
      const inputCY = funnelY + inputW / 2 + lane * inputW * 0.85;
      const reconCY = funnelY + inputW / 2 + lane * inputW * 0.85;
      if (phase < 0.18) {
        const tt = phase / 0.18;
        px = inputX + inputW + 4 + (encX1 - (inputX + inputW + 4)) * tt;
        py = inputCY;
      } else if (phase < 0.5) {
        const tt = (phase - 0.18) / 0.32;
        px = encX1 + (encX2 - encX1) * tt;
        py = inputCY + (yMid - inputCY) * tt;
      } else if (phase < 0.55) {
        const tt = (phase - 0.5) / 0.05;
        px = encX2 + (decX1 - encX2) * tt;
        py = yMid;
      } else if (phase < 0.87) {
        const tt = (phase - 0.55) / 0.32;
        px = decX1 + (decX2 - decX1) * tt;
        py = yMid + (reconCY - yMid) * tt;
      } else {
        const tt = (phase - 0.87) / 0.13;
        px = decX2 + 4 + (reconX - decX2 - 4) * tt;
        py = reconCY;
      }
      const r = phase > 0.5 && phase < 0.55 ? 1.6 : 2.6;
      const alpha = phase < 0.5 ? 0.35 + phase * 0.7 : 0.85 - (phase - 0.5) * 0.4;
      ctx.fillStyle = `rgba(122,31,36,${Math.max(0.15, alpha)})`;
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fill();
    }

    // 7. Loss anchor: draw faint dotted bracket from x to x̂ with L = ‖x − x̂‖²
    ctx.strokeStyle = 'rgba(38,35,32,0.35)';
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 1;
    const lossY = funnelY + inputW + 36;
    ctx.beginPath();
    ctx.moveTo(inputX + inputW / 2, lossY);
    ctx.lineTo(inputX + inputW / 2, lossY + 10);
    ctx.lineTo(reconX + inputW / 2, lossY + 10);
    ctx.lineTo(reconX + inputW / 2, lossY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    const lossMidX = (inputX + inputW / 2 + reconX + inputW / 2) / 2;
    ctx.fillText('L = ‖x − x̂‖²', lossMidX, lossY + 24);
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = INK_FADE;
    ctx.fillText('SQUARED PIXEL DISTANCE — TRAINING MINIMIZES THIS', lossMidX, lossY + 38);

    // Squeeze ratio annotation pinned near the bottleneck
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText(`64 → ${state.k}  (${(64 / state.k).toFixed(1)}× squeeze)`, bX, latY + latH + 30);
  }

  // ───────── View 2: Denoising triptych ─────────
  function drawDenoise(ctx, size) {
    const clean = SHAPES.circle;
    const noisy = genNoisy(state.noiseSeed);
    const { recon } = svdLikeReconstruct(noisy, state.k);

    const cell = (size.h - 100) / 8;
    const panelW = cell * 8;
    const gap = (size.w - 3 * panelW) / 4;
    const yOff = 36;

    const labels = ['clean original', 'noisy input', 'AE reconstruction'];
    const maps = [clean, noisy, recon];
    const xs = [gap, gap * 2 + panelW, gap * 3 + panelW * 2];

    maps.forEach((m, i) => {
      drawGrid(ctx, m, xs[i], yOff, cell);
      ctx.fillStyle = i === 2 ? ACCENT : INK_FADE;
      ctx.font = i === 2 ? 'italic 13px "Source Serif 4", Georgia, serif'
                          : 'italic 12px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText(labels[i], xs[i] + panelW / 2, yOff + panelW + 22);
    });

    // Connector arrows: noisy → AE → reconstruction
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('+ noise', xs[0] + panelW + gap / 2, yOff + panelW / 2 - 8);
    drawArrow(ctx, xs[0] + panelW + 4, yOff + panelW / 2, xs[1] - 4, yOff + panelW / 2);
    ctx.fillText('autoencoder', xs[1] + panelW + gap / 2, yOff + panelW / 2 - 8);
    drawArrow(ctx, xs[1] + panelW + 4, yOff + panelW / 2, xs[2] - 4, yOff + panelW / 2);

    // Caption-equation
    ctx.fillStyle = INK;
    ctx.font = 'italic 14px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('latent has room for the signal — not the noise', size.w / 2, yOff + panelW + 56);
  }

  function drawArrow(ctx, x1, y1, x2, y2) {
    ctx.strokeStyle = 'rgba(38,35,32,0.45)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    ctx.stroke();
    const ang = Math.atan2(y2 - y1, x2 - x1);
    ctx.fillStyle = 'rgba(38,35,32,0.45)';
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - 8 * Math.cos(ang - 0.4), y2 - 8 * Math.sin(ang - 0.4));
    ctx.lineTo(x2 - 8 * Math.cos(ang + 0.4), y2 - 8 * Math.sin(ang + 0.4));
    ctx.fill();
  }

  // ───────── View 3: Latent scatter (k=2) ─────────
  function drawScatter(ctx, size) {
    // Force k=2 perspective regardless; show all presets in 2D
    const presets = ['cross', 'circle', 'square', 'triangle'];
    const points = presets.map((p) => ({ name: p, z: latent2D(SHAPES[p]) }));
    // Scale to canvas
    const xs = points.map((p) => p.z[0]);
    const ys = points.map((p) => p.z[1]);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const ymin = Math.min(...ys), ymax = Math.max(...ys);
    const margin = 90;
    const tx = (x) => margin + (x - xmin) / (xmax - xmin + 1e-6) * (size.w - 2 * margin);
    const ty = (y) => size.h - margin - (y - ymin) / (ymax - ymin + 1e-6) * (size.h - 2 * margin);

    // Title
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('each preset → its (z₁, z₂) latent · classes separate without labels', size.w / 2, 22);

    // Axes
    ctx.strokeStyle = 'rgba(38,35,32,0.18)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(margin - 12, size.h - margin); ctx.lineTo(size.w - margin + 12, size.h - margin);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(margin, margin - 12); ctx.lineTo(margin, size.h - margin + 12);
    ctx.stroke();
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('z₁', size.w - margin + 4, size.h - margin + 4);
    ctx.fillText('z₂', margin - 16, margin - 4);

    // Each preset: small thumbnail + dot
    points.forEach((p) => {
      const px = tx(p.z[0]), py = ty(p.z[1]);
      // Thumbnail
      const thumbCell = 4;
      const thumbW = thumbCell * 8;
      const tx0 = px - thumbW / 2;
      const ty0 = py - thumbW / 2 - 26;
      const isCurrent = p.name === state.inputKey;
      ctx.save();
      ctx.globalAlpha = isCurrent ? 1 : 0.85;
      drawGrid(ctx, SHAPES[p.name], tx0, ty0, thumbCell);
      ctx.restore();

      // Dot
      ctx.fillStyle = isCurrent ? ACCENT : INK;
      ctx.beginPath();
      ctx.arc(px, py, isCurrent ? 6 : 4, 0, Math.PI * 2);
      ctx.fill();

      // Label
      ctx.fillStyle = isCurrent ? ACCENT : INK_FADE;
      ctx.font = isCurrent
        ? 'italic 13px "Source Serif 4", Georgia, serif'
        : 'italic 12px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText(p.name, px, py + 18);
    });

    // Annotation
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('no labels needed — the bottleneck found this organization on its own',
      size.w / 2, size.h - 18);
  }

  function render() {
    Array.from(inputsEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.input === state.inputKey);
    });
    Array.from(tabsEl.children).forEach((btn) => {
      btn.classList.toggle('on', btn.dataset.view === state.view);
    });
    kLabel.textContent = `k = ${state.k}`;
    kSlider.value = state.k;

    const input = getInput();
    const { recon, coeffs } = svdLikeReconstruct(input, state.k);
    let mse = 0, count = 0;
    for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) {
      mse += (input[i][j] - recon[i][j]) ** 2; count++;
    }
    mse /= count;

    readoutEl.innerHTML = `
      <div class="row"><span>latent dims (k)</span><b>${state.k}</b></div>
      <div class="row"><span>input pixels</span><b>64</b></div>
      <div class="row"><span>compression</span><b>${(64 / state.k).toFixed(1)}×</b></div>
      <div class="row"><span>MSE (per pixel)</span><b>${formatNum(mse)}</b></div>`;

    if (captionEl) captionEl.textContent = CAPTIONS[state.view];

    canvasCtl.redraw();
  }

  // First flow on mount
  runFlow();
})();
