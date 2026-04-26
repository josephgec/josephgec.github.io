/* Vol. VIII — The Perceptron */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, arrow,
          ACCENT, INK, INK_FADE } = V;

  const DATASETS = {
    linear: () => {
      const pts = [];
      for (let i = 0; i < 30; i++) {
        pts.push({ x: -2 + Math.random() * 1.5, y: -2 + Math.random() * 4, label: 0 });
        pts.push({ x: 0.5 + Math.random() * 1.5, y: -2 + Math.random() * 4, label: 1 });
      }
      return pts;
    },
    diagonal: () => {
      const pts = [];
      for (let i = 0; i < 60; i++) {
        const x = -2.5 + Math.random() * 5, y = -2.5 + Math.random() * 5;
        pts.push({ x, y, label: x + y > 0 ? 1 : 0 });
      }
      return pts;
    },
    xor: () => {
      const pts = [];
      for (let q = 0; q < 4; q++) {
        const cx = (q % 2) * 2 - 1, cy = Math.floor(q / 2) * 2 - 1;
        const lab = (q === 0 || q === 3) ? 1 : 0;
        for (let i = 0; i < 15; i++) {
          pts.push({ x: cx + (Math.random() - 0.5) * 0.8, y: cy + (Math.random() - 0.5) * 0.8, label: lab });
        }
      }
      return pts;
    },
  };

  const state = {
    // Default landing state: a partially-trained perceptron on the diagonal dataset.
    // The line slants the wrong way for x+y>0 — visibly close-but-wrong, so the user
    // sees a few orange "misclassified" rings and wants to press Train.
    w1: 0.9, w2: -0.4, b: -0.3,
    datasetKey: 'diagonal',
    data: DATASETS.diagonal(),
    training: false,
    timer: null,
    stepCount: 0,
    // Fading trail of past decision lines (most recent at end).
    history: [],
    // Most recent corrected point (for anchored update annotation).
    lastCorrection: null, // { p, dw1, dw2, db, ttl }
    // Easter-egg overlay: two manually-placed lines that solve XOR with an MLP.
    showMlpSolution: false,
    // Set to the step count when training converges (for the "Converged after N" annotation).
    convergedAt: null,
  };

  // ───────── DOM ─────────
  const weightsEl = document.getElementById('weights');
  const readoutEl = document.getElementById('readout');
  const datasetsEl = document.getElementById('datasets');
  const resampleBtn = document.getElementById('resample');
  const trainBtn = document.getElementById('train-btn');
  const randomBtn = document.getElementById('random-init');
  const warningEl = document.getElementById('warning');
  const canvas = document.getElementById('canvas');
  const mlpWrap = document.getElementById('mlp-toggle-wrap');
  const mlpToggle = document.getElementById('mlp-toggle');

  // Build weight sliders
  const sliders = [
    { key: 'w1',    label: 'w₁',         min: -3, max: 3, step: 0.05 },
    { key: 'w2',    label: 'w₂',         min: -3, max: 3, step: 0.05 },
    { key: 'b',     label: 'b (bias)',   min: -3, max: 3, step: 0.05 },
  ];
  function renderWeights() {
    weightsEl.innerHTML = '';
    sliders.forEach((sl) => {
      const row = document.createElement('div');
      row.innerHTML = `
        <div class="ml-param-head">
          <span class="label">${sl.label}</span>
          <b>${formatNum(state[sl.key])}</b>
        </div>
        <input type="range" min="${sl.min}" max="${sl.max}" step="${sl.step}" value="${state[sl.key]}" />`;
      const inp = row.querySelector('input');
      inp.addEventListener('input', (e) => {
        stopTrain();
        state[sl.key] = +e.target.value;
        render();
      });
      weightsEl.appendChild(row);
    });
  }

  // Datasets
  Object.keys(DATASETS).forEach((k) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = k;
    b.dataset.dataset = k;
    b.addEventListener('click', () => {
      stopTrain();
      state.datasetKey = k;
      state.data = DATASETS[k]();
      state.stepCount = 0;
      state.history = [];
      state.lastCorrection = null;
      state.convergedAt = null;
      render();
    });
    datasetsEl.appendChild(b);
  });

  resampleBtn.addEventListener('click', () => {
    stopTrain();
    state.data = DATASETS[state.datasetKey]();
    state.stepCount = 0;
    state.history = [];
    state.lastCorrection = null;
    state.convergedAt = null;
    render();
  });

  // Train + random init
  trainBtn.addEventListener('click', () => {
    if (state.training) stopTrain(); else startTrain();
  });
  randomBtn.addEventListener('click', () => {
    stopTrain();
    state.w1 = Math.random() * 2 - 1;
    state.w2 = Math.random() * 2 - 1;
    state.b = 0;
    state.stepCount = 0;
    state.history = [];
    state.lastCorrection = null;
    state.convergedAt = null;
    render();
  });

  mlpToggle.addEventListener('change', () => {
    state.showMlpSolution = mlpToggle.checked;
    render();
  });

  function startTrain() {
    state.training = true;
    trainBtn.textContent = 'Pause';
    state.timer = setInterval(() => {
      const lr = 0.1;
      let updates = 0;
      let nw1 = state.w1, nw2 = state.w2, nb = state.b;
      let lastP = null, lastDw1 = 0, lastDw2 = 0, lastDb = 0;
      // Snapshot the line about to be replaced — used as a fading trail entry.
      state.history.push({ w1: state.w1, w2: state.w2, b: state.b });
      if (state.history.length > 24) state.history.shift();
      for (const p of state.data) {
        const z = nw1 * p.x + nw2 * p.y + nb;
        const yhat = z > 0 ? 1 : 0;
        const err = p.label - yhat;
        if (err !== 0) {
          const dw1 = lr * err * p.x, dw2 = lr * err * p.y, db = lr * err;
          nw1 += dw1; nw2 += dw2; nb += db;
          lastP = p; lastDw1 = dw1; lastDw2 = dw2; lastDb = db;
          updates++;
        }
      }
      state.w1 = nw1; state.w2 = nw2; state.b = nb;
      state.stepCount++;
      if (lastP) {
        state.lastCorrection = { p: lastP, dw1: lastDw1, dw2: lastDw2, db: lastDb, ttl: 8 };
      } else if (state.lastCorrection) {
        state.lastCorrection.ttl--;
        if (state.lastCorrection.ttl <= 0) state.lastCorrection = null;
      }
      if (updates === 0) {
        // Convergence: zero misclassifications this sweep.
        if (state.convergedAt == null) state.convergedAt = state.stepCount;
        stopTrain();
      }
      render();
    }, 250);
  }
  function stopTrain() {
    state.training = false;
    trainBtn.textContent = 'Train';
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: true });

  // Helper: draw the line w1·x + w2·y + b = 0 clipped to the visible square.
  function drawLine(ctx, w1, w2, b, toPx, color, width, dashed) {
    if (Math.abs(w2) > 0.001) {
      const x1 = -4, y1 = -(w1 * x1 + b) / w2;
      const x2 = 4,  y2 = -(w1 * x2 + b) / w2;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      if (dashed) ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(...toPx([x1, y1])); ctx.lineTo(...toPx([x2, y2]));
      ctx.stroke();
      if (dashed) ctx.setLineDash([]);
    } else if (Math.abs(w1) > 0.001) {
      const x = -b / w1;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      if (dashed) ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(...toPx([x, -4])); ctx.lineTo(...toPx([x, 4]));
      ctx.stroke();
      if (dashed) ctx.setLineDash([]);
    }
  }

  function draw(ctx, size) {
    const sc = size.w / 8;
    const toPx = ([x, y]) => [size.w / 2 + x * sc, size.h / 2 - y * sc];

    // Decision regions
    const cells = 60;
    for (let i = 0; i < cells; i++) {
      for (let j = 0; j < cells; j++) {
        const x = -4 + (j + 0.5) / cells * 8;
        const y = 4 - (i + 0.5) / cells * 8;
        const z = state.w1 * x + state.w2 * y + state.b;
        ctx.fillStyle = z > 0 ? 'rgba(122,31,36,0.10)' : 'rgba(38,35,32,0.04)';
        ctx.fillRect(j * size.w / cells, i * size.h / cells,
                     size.w / cells + 1, size.h / cells + 1);
      }
    }

    // Grid
    ctx.strokeStyle = 'rgba(38,35,32,0.08)';
    ctx.lineWidth = 1;
    for (let i = -3; i <= 3; i++) {
      ctx.beginPath();
      ctx.moveTo(...toPx([i, -4])); ctx.lineTo(...toPx([i, 4])); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(...toPx([-4, i])); ctx.lineTo(...toPx([4, i])); ctx.stroke();
    }

    // Trail of past decision lines (older = fainter).
    const H = state.history;
    for (let i = 0; i < H.length; i++) {
      const t = (i + 1) / H.length; // 0..1, newer = larger t
      const alpha = 0.04 + t * 0.14;
      drawLine(ctx, H[i].w1, H[i].w2, H[i].b, toPx,
               `rgba(122,31,36,${alpha.toFixed(3)})`, 1.2, false);
    }

    // Current decision line
    drawLine(ctx, state.w1, state.w2, state.b, toPx, ACCENT, 2.5, false);

    // Weight vector + perpendicularity ⊥ marker.
    // The weight vector (w1, w2) is normal to the decision line.
    // Foot-of-perpendicular from origin to the line is at:
    //    f = -b/(w1²+w2²) · (w1, w2)
    const wn2 = state.w1 * state.w1 + state.w2 * state.w2;
    if (wn2 > 1e-4) {
      const fx = -state.b * state.w1 / wn2;
      const fy = -state.b * state.w2 / wn2;
      // Weight vector originating from the foot — visually shows "perpendicular to the line."
      const tipMag = 0.7; // visible scale for arrow head, regardless of weight magnitude
      const wLen = Math.sqrt(wn2);
      const ux = state.w1 / wLen, uy = state.w2 / wLen;
      const tipx = fx + ux * tipMag, tipy = fy + uy * tipMag;
      arrow(ctx, toPx([fx, fy]), toPx([tipx, tipy]),
                'rgba(38,35,32,0.7)', 2, 9);
      // Right-angle ⊥ marker: small square at the foot, oriented along (u, n_along_line).
      const lx = -uy, ly = ux; // along-line tangent (rotate (u) by 90°)
      const s = 0.18; // size in world units
      const aPx = toPx([fx, fy]);
      const bPx = toPx([fx + lx * s, fy + ly * s]);
      const cPx = toPx([fx + lx * s + ux * s, fy + ly * s + uy * s]);
      const dPx = toPx([fx + ux * s, fy + uy * s]);
      ctx.strokeStyle = 'rgba(38,35,32,0.55)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(...aPx); ctx.lineTo(...bPx); ctx.lineTo(...cPx); ctx.lineTo(...dPx);
      ctx.stroke();
      // Tiny "w" label at arrow tip.
      ctx.fillStyle = 'rgba(38,35,32,0.7)';
      ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      const [tipPxX, tipPxY] = toPx([tipx, tipy]);
      ctx.fillText('w', tipPxX + 6, tipPxY - 6);
    }

    // Easter-egg: 2-perceptron MLP solution overlay for XOR.
    if (state.showMlpSolution && state.datasetKey === 'xor') {
      // Two diagonal separators that, when AND'd by a hidden layer, solve XOR.
      // Lines: x + y = -1.0  and  x + y = 1.0  (equivalently the two ridges of XOR).
      // We illustrate two clearly-readable lines crossing through the input space.
      drawLine(ctx, 1, 1,  1.0, toPx, 'rgba(58,107,94,0.9)', 2, true);
      drawLine(ctx, 1, 1, -1.0, toPx, 'rgba(58,107,94,0.9)', 2, true);
      ctx.fillStyle = 'rgba(58,107,94,0.9)';
      ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      const [tx, ty] = toPx([-3.6, 3.6]);
      ctx.fillText('two lines, hidden layer combines them', tx, ty);
    }

    // Points
    for (const p of state.data) {
      const z = state.w1 * p.x + state.w2 * p.y + state.b;
      const pred = z > 0 ? 1 : 0;
      const correct = pred === p.label;
      const [px, py] = toPx([p.x, p.y]);
      ctx.fillStyle = p.label === 1 ? ACCENT : INK;
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.fill();
      if (!correct) {
        ctx.strokeStyle = INK;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(px, py, 8, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    // XOR "linearly inseparable" annotation on the always-misclassified corners.
    if (state.datasetKey === 'xor' && !state.showMlpSolution) {
      ctx.fillStyle = 'rgba(38,35,32,0.65)';
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      const [lx, ly] = toPx([-3.7, 3.6]);
      ctx.fillText('linearly inseparable —', lx, ly);
      ctx.fillText('no single line can split these classes', lx, ly + 14);
    }

    // Convergence annotation — top-center, only when not training and converged.
    if (state.convergedAt != null && !state.training && state.datasetKey !== 'xor') {
      const txt = `Converged after ${state.convergedAt} step${state.convergedAt === 1 ? '' : 's'} — line found a separator.`;
      ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
      const tw = ctx.measureText(txt).width + 16;
      const tx0 = (size.w - tw) / 2;
      ctx.fillStyle = 'rgba(255,253,246,0.92)';
      ctx.strokeStyle = 'rgba(58,107,94,0.7)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.rect(tx0, 8, tw, 22);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#3a6b5e';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(txt, size.w / 2, 19);
    }

    // Anchored update rule: Δw = η · err · x  next to the most-recent corrected point.
    if (state.lastCorrection && state.training) {
      const lc = state.lastCorrection;
      const [px, py] = toPx([lc.p.x, lc.p.y]);
      // Tiny callout box positioned to the upper-right of the point (with edge clamp).
      const txt1 = 'Δw = η · err · x';
      const txt2 = `(${formatNum(lc.dw1)}, ${formatNum(lc.dw2)})`;
      const txt3 = 'this point pushed the weights this way';
      ctx.font = '11px "JetBrains Mono", monospace';
      const w1 = ctx.measureText(txt1).width;
      const w2 = ctx.measureText(txt2).width;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      const w3 = ctx.measureText(txt3).width;
      const boxW = Math.max(w1, w2, w3) + 16;
      const boxH = 52;
      let bx = px + 14, by = py - boxH - 8;
      if (bx + boxW > size.w - 4) bx = px - boxW - 14;
      if (by < 4) by = py + 14;
      ctx.fillStyle = 'rgba(255,253,246,0.92)';
      ctx.strokeStyle = 'rgba(122,31,36,0.7)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.rect(bx, by, boxW, boxH);
      ctx.fill(); ctx.stroke();
      // Connector to the point
      ctx.strokeStyle = 'rgba(122,31,36,0.45)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px, py); ctx.lineTo(bx + boxW / 2, by + boxH);
      ctx.stroke();
      // Halo on the point
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px, py, 11, 0, Math.PI * 2);
      ctx.stroke();
      // Text
      ctx.fillStyle = ACCENT;
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText(txt1, bx + 8, by + 6);
      ctx.fillStyle = INK;
      ctx.fillText(txt2, bx + 8, by + 20);
      ctx.fillStyle = INK_FADE;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.fillText(txt3, bx + 8, by + 35);
    }
  }

  function accuracy() {
    let c = 0;
    for (const p of state.data) {
      const z = state.w1 * p.x + state.w2 * p.y + state.b;
      const yhat = z > 0 ? 1 : 0;
      if (yhat === p.label) c++;
    }
    return c / state.data.length;
  }

  // ───────── Render ─────────
  function render() {
    renderWeights();
    Array.from(datasetsEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.dataset === state.datasetKey);
    });
    readoutEl.innerHTML = `
      <div class="row"><span>decision</span><b style="font-family:var(--mono);font-size:11px">${formatNum(state.w1)}·x + ${formatNum(state.w2)}·y + ${formatNum(state.b)} = 0</b></div>
      <div class="row"><span>accuracy</span><b>${(accuracy() * 100).toFixed(1)}%</b></div>
      <div class="row"><span>step</span><b>${state.stepCount}</b></div>`;
    warningEl.innerHTML = state.datasetKey === 'xor'
      ? '<em style="color:var(--accent)">XOR is not linearly separable — a single perceptron can\'t solve it. Watch it fail.</em>'
      : '';
    // Show / hide MLP toggle (only meaningful for XOR).
    mlpWrap.style.display = (state.datasetKey === 'xor') ? 'flex' : 'none';
    canvasCtl.redraw();
  }

  render();
})();
