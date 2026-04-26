/* Vol. XI — Backpropagation */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK_FADE } = V;

  const tanh = Math.tanh;
  const dtanh = (a) => 1 - a * a;
  function softmax(arr) {
    const m = Math.max(...arr);
    const ex = arr.map((v) => Math.exp(v - m));
    const s = ex.reduce((a, b) => a + b, 0);
    return ex.map((v) => v / s);
  }

  function makeNet(seed = 1) {
    let s = seed;
    const rnd = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
    const rn = () => (rnd() - 0.5) * 1.6;
    return {
      W1: [[rn(), rn()], [rn(), rn()], [rn(), rn()], [rn(), rn()]],
      b1: [0, 0, 0, 0],
      W2: [[rn(), rn(), rn(), rn()], [rn(), rn(), rn(), rn()]],
      b2: [0, 0],
    };
  }

  function forward(net, x) {
    const z1 = net.W1.map((row, i) => row.reduce((s, w, j) => s + w * x[j], 0) + net.b1[i]);
    const a1 = z1.map(tanh);
    const z2 = net.W2.map((row, i) => row.reduce((s, w, j) => s + w * a1[j], 0) + net.b2[i]);
    const a2 = softmax(z2);
    return { x, z1, a1, z2, a2 };
  }

  function backprop(net, fwd, target) {
    const dz2 = fwd.a2.map((p, i) => p - (i === target ? 1 : 0));
    const dW2 = net.W2.map((row, i) => row.map((_, j) => dz2[i] * fwd.a1[j]));
    const db2 = dz2.slice();
    const da1 = fwd.a1.map((_, j) => net.W2.reduce((s, row, i) => s + row[j] * dz2[i], 0));
    const dz1 = da1.map((g, i) => g * dtanh(fwd.a1[i]));
    const dW1 = net.W1.map((row, i) => row.map((_, j) => dz1[i] * fwd.x[j]));
    const db1 = dz1.slice();
    return { dz1, dz2, dW1, dW2, db1, db2 };
  }

  function applyGrads(net, grads, lr) {
    return {
      W1: net.W1.map((row, i) => row.map((w, j) => w - lr * grads.dW1[i][j])),
      b1: net.b1.map((b, i) => b - lr * grads.db1[i]),
      W2: net.W2.map((row, i) => row.map((w, j) => w - lr * grads.dW2[i][j])),
      b2: net.b2.map((b, i) => b - lr * grads.db2[i]),
    };
  }

  function makeData() {
    const pts = [];
    for (let i = 0; i < 80; i++) {
      const x = (Math.random() - 0.5) * 3;
      const y = (Math.random() - 0.5) * 3;
      const label = (y > 0.5 * Math.sin(x * 1.5)) ? 1 : 0;
      pts.push({ x, y, label });
    }
    return pts;
  }

  const state = {
    net: makeNet(3),
    data: makeData(),
    pickIdx: 12,
    phase: 'idle',
    training: false,
    timer: null,
    epoch: 0,
    lossHist: [],
  };

  // ───────── DOM ─────────
  const readoutEl = document.getElementById('readout');
  const nextBtn = document.getElementById('next-btn');
  const animBtn = document.getElementById('anim-btn');
  const phaseLabel = document.getElementById('phase-label');
  const trainBtn = document.getElementById('train-btn');
  const epochLabel = document.getElementById('epoch-label');
  const accLabel = document.getElementById('acc-label');
  const resetBtn = document.getElementById('reset-btn');
  const lossCanvas = document.getElementById('loss-canvas');
  const canvas = document.getElementById('canvas');
  const decisionCanvas = document.getElementById('decision-canvas');
  const captionEl = document.getElementById('caption');

  nextBtn.addEventListener('click', () => {
    state.pickIdx++;
    render();
  });
  resetBtn.addEventListener('click', () => {
    stopTrain();
    state.net = makeNet(Math.floor(Math.random() * 100) + 1);
    state.epoch = 0;
    state.lossHist = [];
    render();
  });

  // Step animation
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  animBtn.addEventListener('click', async () => {
    if (state.phase !== 'idle') return;
    const sample = state.data[state.pickIdx % state.data.length];
    const fwd = forward(state.net, [sample.x, sample.y]);
    const grads = backprop(state.net, fwd, sample.label);

    state.phase = 'forward';   render(); await sleep(800);
    state.phase = 'forward-done'; render(); await sleep(500);
    state.phase = 'backward';  render(); await sleep(800);
    state.phase = 'backward-done'; render(); await sleep(500);
    state.net = applyGrads(state.net, grads, 0.1);
    state.phase = 'idle'; render();
  });

  trainBtn.addEventListener('click', () => {
    if (state.training) stopTrain(); else startTrain();
  });
  function startTrain() {
    state.training = true;
    trainBtn.textContent = '⏸ Pause';
    state.timer = setInterval(() => {
      let nn = state.net;
      let totalLoss = 0;
      for (const p of state.data) {
        const f = forward(nn, [p.x, p.y]);
        const g = backprop(nn, f, p.label);
        nn = applyGrads(nn, g, 0.05);
        totalLoss += -Math.log(f.a2[p.label] + 1e-9);
      }
      state.net = nn;
      state.lossHist = [...state.lossHist.slice(-99), totalLoss / state.data.length];
      state.epoch++;
      render();
    }, 80);
  }
  function stopTrain() {
    state.training = false;
    trainBtn.textContent = '▷ Train continuously';
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  // ───────── Main canvas (the network diagram) ─────────
  const canvasCtl = V.attachCanvas(canvas, drawNet, { square: false });

  function drawNet(ctx, size) {
    const sample = state.data[state.pickIdx % state.data.length];
    const fwd = forward(state.net, [sample.x, sample.y]);
    const grads = backprop(state.net, fwd, sample.label);
    const target = sample.label;

    ctx.fillStyle = '#fffdf6';
    ctx.fillRect(0, 0, size.w, size.h);

    const cols = [
      { n: 2, x: size.w * 0.13, label: 'input',  sub: 'x' },
      { n: 4, x: size.w * 0.43, label: 'hidden', sub: 'a⁽¹⁾' },
      { n: 2, x: size.w * 0.73, label: 'output', sub: 'ŷ' },
    ];
    const colY = (n, i) => size.h * 0.5 + (i - (n - 1) / 2) * Math.min(80, size.h * 0.18);

    const showFwd = ['forward', 'forward-done', 'backward', 'backward-done'].includes(state.phase);
    const showBwd = state.phase === 'backward' || state.phase === 'backward-done';

    function drawConns(fromIdx, toIdx, W, dW) {
      const fromCol = cols[fromIdx], toCol = cols[toIdx];
      for (let i = 0; i < toCol.n; i++) {
        for (let j = 0; j < fromCol.n; j++) {
          const w = W[i][j], dw = dW[i][j];
          const x1 = fromCol.x + 24, y1 = colY(fromCol.n, j);
          const x2 = toCol.x - 24,    y2 = colY(toCol.n, i);
          let stroke;
          if (showBwd) {
            const gmag = Math.min(1, Math.abs(dw) * 4);
            stroke = -dw > 0
              ? `rgba(122,31,36,${0.2 + gmag * 0.7})`
              : `rgba(58,107,94,${0.2 + gmag * 0.7})`;
          } else {
            const alpha = 0.2 + Math.min(0.5, Math.abs(w) * 0.4);
            stroke = w >= 0 ? `rgba(122,31,36,${alpha})` : `rgba(38,35,32,${alpha})`;
          }
          ctx.strokeStyle = stroke;
          ctx.lineWidth = showBwd ? 0.5 + Math.abs(dw) * 8 : 0.5 + Math.abs(w) * 0.6;
          ctx.beginPath();
          ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
          ctx.stroke();
        }
      }
    }

    drawConns(0, 1, state.net.W1, grads.dW1);
    drawConns(1, 2, state.net.W2, grads.dW2);

    // Neurons
    for (let ci = 0; ci < cols.length; ci++) {
      const col = cols[ci];
      const values = ci === 0 ? fwd.x : (ci === 1 ? fwd.a1 : fwd.a2);
      const dvals = ci === 0 ? null : (ci === 1 ? grads.dz1 : grads.dz2);
      for (let i = 0; i < col.n; i++) {
        const x = col.x, y = colY(col.n, i);
        const v = values[i];
        const isLit = ci === 0 || showFwd;
        const mag = Math.min(1, Math.abs(v));
        ctx.fillStyle = isLit
          ? (v >= 0 ? `rgba(122,31,36,${0.15 + mag * 0.7})` : `rgba(38,35,32,${0.15 + mag * 0.7})`)
          : '#fffdf6';
        ctx.strokeStyle = '#262320';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, y, 24, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();

        // Gradient halo
        if (showBwd && dvals) {
          const gmag = Math.min(1, Math.abs(dvals[i]) * 3);
          ctx.strokeStyle = `rgba(58,107,94,${0.3 + gmag * 0.7})`;
          ctx.lineWidth = 2 + gmag * 4;
          ctx.beginPath();
          ctx.arc(x, y, 30, 0, Math.PI * 2);
          ctx.stroke();
        }

        if (isLit) {
          ctx.fillStyle = mag > 0.5 ? '#fffdf6' : '#262320';
          ctx.font = 'bold 11px "JetBrains Mono", monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(formatNum(v), x, y);
        }
      }
      ctx.fillStyle = INK_FADE;
      ctx.font = '11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(col.label, col.x, colY(col.n, col.n - 1) + 50);
      ctx.fillStyle = 'rgba(38,35,32,0.4)';
      ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
      ctx.fillText(col.sub, col.x, colY(col.n, 0) - 38);
    }

    // Target indicator on output
    const tCol = cols[2];
    const ty = colY(tCol.n, target);
    ctx.strokeStyle = '#3a6b5e';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(tCol.x, ty, 36, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#3a6b5e';
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('← target', tCol.x + 42, ty + 4);

    // Phase indicator
    ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    if (state.phase === 'forward' || state.phase === 'forward-done') {
      ctx.fillStyle = ACCENT;
      ctx.fillText('forward →', size.w / 2, 24);
    } else if (state.phase === 'backward' || state.phase === 'backward-done') {
      ctx.fillStyle = '#3a6b5e';
      ctx.fillText('← backward', size.w / 2, 24);
    }
  }

  // ───────── Decision boundary canvas ─────────
  function drawDecision() {
    const cv = decisionCanvas;
    const dpr = window.devicePixelRatio || 1;
    const r = cv.parentElement.getBoundingClientRect();
    const W = r.width, H = r.height;
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = W + 'px'; cv.style.height = H + 'px';
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const xMin = -2, xMax = 2, yMin = -2, yMax = 2;
    const tx = (x) => (x - xMin) / (xMax - xMin) * W;
    const ty = (y) => H - (y - yMin) / (yMax - yMin) * H;

    const cells = 50;
    const cellsX = Math.floor(cells * W / H);
    for (let i = 0; i < cells; i++) {
      for (let j = 0; j < cellsX; j++) {
        const x = xMin + (j + 0.5) / cellsX * (xMax - xMin);
        const y = yMax - (i + 0.5) / cells * (yMax - yMin);
        const f = forward(state.net, [x, y]);
        const conf = f.a2[1] - f.a2[0];
        ctx.fillStyle = conf > 0
          ? `rgba(122,31,36,${0.05 + Math.abs(conf) * 0.25})`
          : `rgba(38,35,32,${0.05 + Math.abs(conf) * 0.18})`;
        ctx.fillRect(j * W / cellsX, i * H / cells, W / cellsX + 1, H / cells + 1);
      }
    }
    for (const p of state.data) {
      ctx.fillStyle = p.label === 1 ? ACCENT : '#262320';
      ctx.beginPath();
      ctx.arc(tx(p.x), ty(p.y), 4, 0, Math.PI * 2);
      ctx.fill();
    }
    const sample = state.data[state.pickIdx % state.data.length];
    ctx.strokeStyle = '#3a6b5e';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(tx(sample.x), ty(sample.y), 9, 0, Math.PI * 2);
    ctx.stroke();
  }

  // ───────── Loss curve ─────────
  function drawLossCurve() {
    const cv = lossCanvas;
    const dpr = window.devicePixelRatio || 1;
    const r = cv.parentElement.getBoundingClientRect();
    const W = r.width, H = 80;
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = W + 'px'; cv.style.height = H + 'px';
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    if (state.lossHist.length < 2) return;
    const max = Math.max(...state.lossHist);
    const min = Math.min(...state.lossHist);
    const range = Math.max(0.01, max - min);
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    state.lossHist.forEach((v, i) => {
      const x = (i / (state.lossHist.length - 1)) * W;
      const y = H - ((v - min) / range) * (H - 10) - 5;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillText('loss ' + state.lossHist[state.lossHist.length - 1].toFixed(3), 6, 12);
  }

  // ───────── Render ─────────
  function captionFor(phase) {
    switch (phase) {
      case 'forward': return 'Forward pass — input flowing right, layer by layer, into a prediction.';
      case 'forward-done': return 'Prediction made. Compare to the truth — that gap is the loss.';
      case 'backward': return 'Backward pass — error flows right to left. Each weight learns how much it contributed.';
      case 'backward-done': return 'Gradients computed. Now nudge each weight a tiny step against its gradient.';
    }
    return 'Press "Forward → Backward" to step through one update — or hit "Train continuously" and watch the decision boundary form.';
  }

  function accuracy() {
    let c = 0;
    for (const p of state.data) {
      const f = forward(state.net, [p.x, p.y]);
      if ((f.a2[1] > f.a2[0] ? 1 : 0) === p.label) c++;
    }
    return c / state.data.length;
  }

  function render() {
    const sample = state.data[state.pickIdx % state.data.length];
    const fwd = forward(state.net, [sample.x, sample.y]);
    const loss = -Math.log(fwd.a2[sample.label] + 1e-9);
    readoutEl.innerHTML = `
      <div class="row"><span>x</span><b>(${formatNum(sample.x)}, ${formatNum(sample.y)})</b></div>
      <div class="row"><span>true class</span><b style="color:${sample.label ? 'var(--accent)' : 'var(--ink)'}">${sample.label}</b></div>
      <div class="row"><span>predicted</span><b>${(fwd.a2[1] * 100).toFixed(0)}% / ${(fwd.a2[0] * 100).toFixed(0)}%</b></div>
      <div class="row"><span>loss</span><b>${formatNum(loss)}</b></div>`;
    phaseLabel.textContent = state.phase;
    epochLabel.textContent = state.epoch;
    accLabel.textContent = `${(accuracy() * 100).toFixed(0)}%`;
    captionEl.innerHTML = `<span class="ml-cap-num">·</span> ${captionFor(state.phase)}`;

    canvasCtl.redraw();
    drawDecision();
    drawLossCurve();
  }

  // Resize handling for the secondary canvases (loss + decision).
  new ResizeObserver(() => { drawDecision(); drawLossCurve(); }).observe(decisionCanvas.parentElement);

  render();
})();
