/* Vol. VIII — The Perceptron */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, arrow,
          ACCENT, INK, INK_FADE, RULE_FAINT, ACCENT_SOFT,
          SERIF_LABEL_SM } = V;

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
    w1: 1, w2: 0.5, b: 0,
    datasetKey: 'diagonal',
    data: DATASETS.diagonal(),
    training: false,
    timer: null,
    stepCount: 0,
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
      render();
    });
    datasetsEl.appendChild(b);
  });

  resampleBtn.addEventListener('click', () => {
    stopTrain();
    state.data = DATASETS[state.datasetKey]();
    state.stepCount = 0;
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
    render();
  });

  function startTrain() {
    state.training = true;
    trainBtn.textContent = 'Pause';
    state.timer = setInterval(() => {
      const lr = 0.1;
      let updates = 0;
      let nw1 = state.w1, nw2 = state.w2, nb = state.b;
      for (const p of state.data) {
        const z = nw1 * p.x + nw2 * p.y + nb;
        const yhat = z > 0 ? 1 : 0;
        const err = p.label - yhat;
        if (err !== 0) {
          nw1 += lr * err * p.x;
          nw2 += lr * err * p.y;
          nb  += lr * err;
          updates++;
        }
      }
      state.w1 = nw1; state.w2 = nw2; state.b = nb;
      state.stepCount++;
      if (updates === 0) stopTrain();
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

    // Decision line
    if (Math.abs(state.w2) > 0.001) {
      const x1 = -4, y1 = -(state.w1 * x1 + state.b) / state.w2;
      const x2 = 4,  y2 = -(state.w1 * x2 + state.b) / state.w2;
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(...toPx([x1, y1])); ctx.lineTo(...toPx([x2, y2]));
      ctx.stroke();
    } else if (Math.abs(state.w1) > 0.001) {
      const x = -state.b / state.w1;
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(...toPx([x, -4])); ctx.lineTo(...toPx([x, 4]));
      ctx.stroke();
    }

    // Weight vector
    arrow(ctx, toPx([0, 0]), toPx([state.w1 * 0.5, state.w2 * 0.5]),
              'rgba(38,35,32,0.55)', 2, 8);

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
    canvasCtl.redraw();
  }

  render();
})();
