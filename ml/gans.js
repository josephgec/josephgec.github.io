/* Vol. XXII — The Adversaries (GANs) */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK, INK_FADE } = V;

  function gaussian() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // Real-data distributions. Each one has a known set of "modes" we can check
  // against to detect mode collapse.
  const DISTS = {
    ring: {
      sample: () => {
        const a = Math.random() * Math.PI * 2;
        const r = 1.6 + gaussian() * 0.05;
        return [r * Math.cos(a), r * Math.sin(a)];
      },
      // Ring is continuous — we treat 8 angle bins as "modes" for collapse detection.
      modes: Array.from({ length: 8 }, (_, k) => {
        const a = (k / 8) * Math.PI * 2;
        return [1.6 * Math.cos(a), 1.6 * Math.sin(a)];
      }),
    },
    'two clusters': {
      sample: () => {
        const cls = Math.random() < 0.5 ? -1 : 1;
        return [cls * 1.4 + gaussian() * 0.25, cls * 1.4 + gaussian() * 0.25];
      },
      modes: [[-1.4, -1.4], [1.4, 1.4]],
    },
    spiral: {
      sample: () => {
        const t = Math.random() * Math.PI * 4;
        const r = t * 0.18;
        return [r * Math.cos(t) + gaussian() * 0.05, r * Math.sin(t) + gaussian() * 0.05];
      },
      modes: Array.from({ length: 6 }, (_, k) => {
        const t = ((k + 1) / 6) * Math.PI * 4;
        const r = t * 0.18;
        return [r * Math.cos(t), r * Math.sin(t)];
      }),
    },
    grid: {
      sample: () => {
        const i = Math.floor(Math.random() * 3) - 1;
        const j = Math.floor(Math.random() * 3) - 1;
        return [i * 1.2 + gaussian() * 0.1, j * 1.2 + gaussian() * 0.1];
      },
      modes: (() => {
        const m = [];
        for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) m.push([i * 1.2, j * 1.2]);
        return m;
      })(),
    },
  };

  const state = {
    distKey: 'ring',
    real: [],
    fake: [],
    dStrength: 1.0,
    training: false,
    timer: null,
    step: 0,
    accuracy: 0.5,
    gLoss: 1.0,
    dLoss: 1.0,
    lossHist: [],   // array of { step, gLoss, dLoss }
    modesCovered: 0,
    totalModes: 0,
  };

  function resampleReal() {
    state.real = Array.from({ length: 80 }, () => DISTS[state.distKey].sample());
  }
  function resampleFake() {
    state.fake = Array.from({ length: 80 }, () => [gaussian() * 0.3, gaussian() * 0.3]);
  }
  resampleReal();
  resampleFake();

  // Discriminator: a soft RBF density estimate of "realness" at any point.
  function discriminator(x, y) {
    let s = 0;
    const sigma = 0.4 / state.dStrength;
    for (const [rx, ry] of state.real) s += Math.exp(-((x - rx) ** 2 + (y - ry) ** 2) / (2 * sigma * sigma));
    for (const [fx, fy] of state.fake) s -= Math.exp(-((x - fx) ** 2 + (y - fy) ** 2) / (2 * sigma * sigma));
    return s / Math.max(state.real.length, 1);
  }

  // Map raw discriminator score to a probability-like value in (0, 1).
  function dProb(x, y) {
    const z = discriminator(x, y) * 8 * state.dStrength;
    return 1 / (1 + Math.exp(-z));
  }

  function stepGAN() {
    const lr = 0.07;
    const eps = 0.05;
    state.fake = state.fake.map(([x, y]) => {
      const dx = (discriminator(x + eps, y) - discriminator(x - eps, y)) / (2 * eps);
      const dy = (discriminator(x, y + eps) - discriminator(x, y - eps)) / (2 * eps);
      return [
        x + lr * dx + (Math.random() - 0.5) * 0.04,
        y + lr * dy + (Math.random() - 0.5) * 0.04,
      ];
    });
    state.step++;

    // D accuracy
    let c = 0;
    for (const [rx, ry] of state.real) if (discriminator(rx, ry) > 0) c++;
    for (const [fx, fy] of state.fake) if (discriminator(fx, fy) < 0) c++;
    state.accuracy = c / (state.real.length + state.fake.length);

    // Loss approximations:
    //   D-loss ≈ −E[log D(x)] − E[log(1 − D(G(z)))]
    //   G-loss ≈ −E[log D(G(z))]            (non-saturating form, common in practice)
    let dl = 0, gl = 0;
    const eps2 = 1e-6;
    for (const [rx, ry] of state.real) dl += -Math.log(dProb(rx, ry) + eps2);
    for (const [fx, fy] of state.fake) {
      const p = dProb(fx, fy);
      dl += -Math.log(1 - p + eps2);
      gl += -Math.log(p + eps2);
    }
    dl /= (state.real.length + state.fake.length);
    gl /= state.fake.length;
    state.dLoss = dl;
    state.gLoss = gl;
    state.lossHist.push({ step: state.step, gLoss: gl, dLoss: dl });
    if (state.lossHist.length > 220) state.lossHist.shift();

    // Mode coverage detection
    const modes = DISTS[state.distKey].modes;
    state.totalModes = modes.length;
    let covered = 0;
    const tol = 0.55;
    for (const m of modes) {
      for (const [fx, fy] of state.fake) {
        if (Math.hypot(fx - m[0], fy - m[1]) < tol) { covered++; break; }
      }
    }
    state.modesCovered = covered;
  }

  // ───────── DOM ─────────
  const distsEl = document.getElementById('dists');
  const dStrength = document.getElementById('d-strength');
  const dLabel = document.getElementById('d-label');
  const dSubLabel = document.getElementById('d-sublabel');
  const trainBtn = document.getElementById('train-btn');
  const resetBtn = document.getElementById('reset-btn');
  const readoutEl = document.getElementById('readout');
  const canvas = document.getElementById('canvas');
  const lossCanvas = document.getElementById('loss-canvas');

  Object.keys(DISTS).forEach((k) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = k;
    b.dataset.dist = k;
    b.addEventListener('click', () => {
      state.distKey = k;
      resampleReal(); resampleFake();
      state.step = 0;
      state.lossHist = [];
      stopTrain();
      // Pre-warm to mid-training so heatmap is informative on selection
      for (let i = 0; i < 22; i++) stepGAN();
      render();
    });
    distsEl.appendChild(b);
  });

  function strengthLabel(v) {
    if (v < 0.85) return { code: 'TOO WEAK', desc: 'D barely tells real from fake — G has no useful gradient signal.' };
    if (v > 1.6)  return { code: 'TOO STRONG', desc: 'D wins by a wide margin — its gradients flatten and G stalls.' };
    return { code: 'EQUILIBRIUM', desc: 'D and G push back evenly — the only regime that learns.' };
  }

  dStrength.addEventListener('input', (e) => {
    state.dStrength = +e.target.value;
    render();
  });

  trainBtn.addEventListener('click', () => {
    if (state.training) stopTrain(); else startTrain();
  });
  resetBtn.addEventListener('click', () => {
    stopTrain();
    resampleReal(); resampleFake();
    state.step = 0;
    state.lossHist = [];
    // Re-prime to mid-training default so the heatmap stays informative
    for (let i = 0; i < 22; i++) stepGAN();
    render();
  });

  function startTrain() {
    state.training = true;
    trainBtn.textContent = '⏸ Pause';
    state.timer = setInterval(() => { stepGAN(); render(); }, 80);
  }
  function stopTrain() {
    state.training = false;
    trainBtn.textContent = '▷ Train';
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: true });
  const lossCtl = V.attachCanvas(lossCanvas, drawLoss, { square: false });

  function drawLoss(ctx, size) {
    const padL = 36, padR = 14, padT = 14, padB = 22;
    const W = size.w - padL - padR;
    const H = size.h - padT - padB;

    // Title left
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('LOSS', padL, padT - 2);

    // Find scale
    const hist = state.lossHist;
    let yMax = 1;
    if (hist.length > 0) {
      yMax = Math.max(0.6, Math.max(...hist.map(h => Math.max(h.gLoss, h.dLoss))) * 1.05);
    }
    const xMin = hist.length > 0 ? hist[0].step : 0;
    const xMax = Math.max(xMin + 50, hist.length > 0 ? hist[hist.length - 1].step : 50);
    const tx = (s) => padL + (s - xMin) / Math.max(1, (xMax - xMin)) * W;
    const ty = (l) => padT + H - (l / yMax) * H;

    // Grid
    ctx.strokeStyle = 'rgba(38,35,32,0.08)';
    ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const y = padT + H - (g / 4) * H;
      ctx.beginPath();
      ctx.moveTo(padL, y); ctx.lineTo(padL + W, y);
      ctx.stroke();
    }
    // Y-axis labels
    ctx.fillStyle = INK_FADE;
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.fillText('0', padL - 4, padT + H + 2);
    ctx.fillText(yMax.toFixed(1), padL - 4, padT + 8);

    if (hist.length < 2) {
      ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
      ctx.fillStyle = INK_FADE;
      ctx.textAlign = 'center';
      ctx.fillText('press ▷ Train to start the game — losses will plot here', padL + W / 2, padT + H / 2);
      return;
    }

    // D-loss (ink line)
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    hist.forEach((h, i) => {
      const x = tx(h.step), y = ty(h.dLoss);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // G-loss (orange line)
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    hist.forEach((h, i) => {
      const x = tx(h.step), y = ty(h.gLoss);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Legend
    const legY = padT + 2;
    const legX = padL + W - 138;
    ctx.fillStyle = ACCENT;
    ctx.fillRect(legX, legY + 4, 14, 2);
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('G-LOSS', legX + 18, legY + 8);
    ctx.fillStyle = INK;
    ctx.fillRect(legX + 70, legY + 4, 14, 2);
    ctx.fillStyle = INK_FADE;
    ctx.fillText('D-LOSS', legX + 88, legY + 8);

    // step axis
    ctx.fillStyle = INK_FADE;
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`step ${xMin}`, padL, padT + H + 14);
    ctx.textAlign = 'right';
    ctx.fillText(`step ${xMax}`, padL + W, padT + H + 14);
  }

  function draw(ctx, size) {
    const R = 3;
    const tx = (x) => (x + R) / (2 * R) * size.w;
    const ty = (y) => size.h - (y + R) / (2 * R) * size.h;

    // Discriminator heatmap
    const cells = 50;
    let dMax = 0;
    const grid = [];
    for (let i = 0; i < cells; i++) {
      grid.push([]);
      for (let j = 0; j < cells; j++) {
        const x = -R + (j + 0.5) / cells * 2 * R;
        const y = R - (i + 0.5) / cells * 2 * R;
        const d = discriminator(x, y);
        grid[i].push(d);
        if (Math.abs(d) > dMax) dMax = Math.abs(d);
      }
    }
    for (let i = 0; i < cells; i++) for (let j = 0; j < cells; j++) {
      const d = grid[i][j];
      const t = Math.max(0, Math.min(1, Math.abs(d) / Math.max(0.001, dMax)));
      const color = d > 0
        ? `rgba(122,31,36,${0.05 + t * 0.30})`
        : `rgba(38,35,32,${0.04 + t * 0.20})`;
      ctx.fillStyle = color;
      ctx.fillRect(j * size.w / cells, i * size.h / cells, size.w / cells + 1, size.h / cells + 1);
    }

    // Grid
    ctx.strokeStyle = 'rgba(38,35,32,0.08)';
    ctx.lineWidth = 1;
    for (let i = -3; i <= 3; i++) {
      ctx.beginPath(); ctx.moveTo(tx(i), 0); ctx.lineTo(tx(i), size.h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, ty(i)); ctx.lineTo(size.w, ty(i)); ctx.stroke();
    }

    // Real points
    for (const [x, y] of state.real) {
      ctx.fillStyle = '#262320';
      ctx.beginPath();
      ctx.arc(tx(x), ty(y), 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
    // Fake points
    for (const [x, y] of state.fake) {
      ctx.fillStyle = ACCENT;
      ctx.beginPath();
      ctx.arc(tx(x), ty(y), 3.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Mode coverage: glow uncovered modes
    const modes = DISTS[state.distKey].modes;
    const tol = 0.55;
    const uncovered = modes.filter((m) => {
      for (const [fx, fy] of state.fake) {
        if (Math.hypot(fx - m[0], fy - m[1]) < tol) return false;
      }
      return true;
    });
    if (state.step > 8) {
      for (const m of uncovered) {
        const px = tx(m[0]), py = ty(m[1]);
        ctx.strokeStyle = 'rgba(122,31,36,0.55)';
        ctx.lineWidth = 1.5;
        const pulse = 0.5 + 0.5 * Math.sin(state.step * 0.4);
        ctx.beginPath();
        ctx.arc(px, py, 12 + pulse * 6, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    // First-load storytelling: small annotation explaining the scene before user hits Train
    if (!state.training && state.step <= 25 && state.modesCovered < state.totalModes) {
      ctx.fillStyle = INK_FADE;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'right';
      ctx.fillText(`step ${state.step} · mid-training — press ▷ Train to continue`, size.w - 10, size.h - 10);
    }

    // Mode collapse warning banner — only after enough training that the
    // generator has had a real chance to spread, so the pre-warmed initial
    // state isn't mistaken for a failure.
    if (state.totalModes > 1 && state.step > 60 &&
        state.modesCovered <= Math.max(1, Math.floor(state.totalModes / 3))) {
      ctx.save();
      const bw = 320, bh = 36;
      const bx = (size.w - bw) / 2;
      const by = 14;
      ctx.fillStyle = 'rgba(122,31,36,0.92)';
      ctx.fillRect(bx, by, bw, bh);
      ctx.fillStyle = '#fffdf6';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('MODE COLLAPSE', size.w / 2, by + 12);
      ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
      ctx.fillText(`G covers ${state.modesCovered} of ${state.totalModes} modes`, size.w / 2, by + 26);
      ctx.restore();
    }
  }

  function render() {
    Array.from(distsEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.dist === state.distKey);
    });
    dStrength.value = state.dStrength;
    const sl = strengthLabel(state.dStrength);
    if (dLabel) {
      dLabel.textContent = sl.code;
      dLabel.style.color = sl.code === 'EQUILIBRIUM' ? 'var(--accent)' : 'var(--ml-ink-fade)';
    }
    if (dSubLabel) dSubLabel.textContent = sl.desc;

    readoutEl.innerHTML = `
      <div class="row"><span>step</span><b>${state.step}</b></div>
      <div class="row"><span>G-loss</span><b>${formatNum(state.gLoss)}</b></div>
      <div class="row"><span>D-loss</span><b>${formatNum(state.dLoss)}</b></div>
      <div class="row"><span>D accuracy</span><b>${(state.accuracy * 100).toFixed(0)}%</b></div>
      <div class="row"><span>modes covered</span><b>${state.modesCovered} / ${state.totalModes}</b></div>`;
    canvasCtl.redraw();
    lossCtl.redraw();
  }

  // initialize totalModes for the initial dist
  state.totalModes = DISTS[state.distKey].modes.length;
  // Pre-warm a partial training so the first-load heatmap is informative —
  // D has separated real ring from a still-tight fake blob, gradient field is visible.
  for (let i = 0; i < 22; i++) stepGAN();
  render();
})();
