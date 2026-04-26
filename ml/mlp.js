/* Vol. XIV — Build a Network (MLP Builder) */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK_FADE } = V;

  function genData(kind) {
    const pts = [];
    if (kind === 'circle') {
      for (let i = 0; i < 100; i++) {
        const r = Math.random() < 0.5 ? Math.random() * 0.8 : 1.4 + Math.random() * 0.8;
        const a = Math.random() * Math.PI * 2;
        pts.push({ x: r * Math.cos(a), y: r * Math.sin(a), label: r < 1 ? 1 : 0 });
      }
    } else if (kind === 'xor') {
      for (let i = 0; i < 100; i++) {
        const x = (Math.random() - 0.5) * 4, y = (Math.random() - 0.5) * 4;
        pts.push({ x, y, label: x * y > 0 ? 1 : 0 });
      }
    } else if (kind === 'spiral') {
      for (let cls = 0; cls < 2; cls++) {
        for (let i = 0; i < 50; i++) {
          const r = i / 50 * 2;
          const t = i / 50 * 3.5 + cls * Math.PI + (Math.random() - 0.5) * 0.3;
          pts.push({ x: r * Math.cos(t), y: r * Math.sin(t), label: cls });
        }
      }
    } else {
      for (let i = 0; i < 100; i++) {
        const cls = Math.random() < 0.5 ? 0 : 1;
        const cx = cls ? 1.2 : -1.2, cy = cls ? 1.2 : -1.2;
        pts.push({ x: cx + (Math.random() - 0.5) * 1.5, y: cy + (Math.random() - 0.5) * 1.5, label: cls });
      }
    }
    return pts;
  }

  const tanh = Math.tanh;
  const dtanh = (a) => 1 - a * a;
  const relu = (x) => Math.max(0, x);
  const drelu = (z) => (z > 0 ? 1 : 0);
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));

  function makeNet(layerSizes, seed = 1) {
    let s = seed;
    const rnd = () => { s = (s * 9301 + 49297) % 233280; return (s / 233280 - 0.5) * 1.6; };
    const layers = [];
    for (let i = 1; i < layerSizes.length; i++) {
      const W = [];
      for (let j = 0; j < layerSizes[i]; j++) {
        const row = [];
        for (let k = 0; k < layerSizes[i - 1]; k++) row.push(rnd());
        W.push(row);
      }
      layers.push({ W, b: new Array(layerSizes[i]).fill(0) });
    }
    return { layers, sizes: layerSizes };
  }

  function forward(net, x, activation) {
    const acts = [x.slice()];
    const zs = [];
    let a = x.slice();
    for (let li = 0; li < net.layers.length; li++) {
      const L = net.layers[li];
      const z = L.W.map((row, i) => row.reduce((s, w, j) => s + w * a[j], 0) + L.b[i]);
      zs.push(z);
      a = li === net.layers.length - 1
        ? z.map(sigmoid)
        : (activation === 'relu' ? z.map(relu) : z.map(tanh));
      acts.push(a);
    }
    return { acts, zs };
  }

  function backprop(net, fwd, target, activation) {
    const dWs = [], dbs = [];
    let dz = [fwd.acts[fwd.acts.length - 1][0] - target];
    for (let li = net.layers.length - 1; li >= 0; li--) {
      const a_prev = fwd.acts[li];
      const L = net.layers[li];
      const dW = L.W.map((row, i) => row.map((_, j) => dz[i] * a_prev[j]));
      const db = dz.slice();
      dWs.unshift(dW); dbs.unshift(db);
      if (li > 0) {
        const da = a_prev.map((_, j) => L.W.reduce((s, row, i) => s + row[j] * dz[i], 0));
        const z_prev = fwd.zs[li - 1];
        dz = da.map((g, i) => g * (activation === 'relu' ? drelu(z_prev[i]) : dtanh(fwd.acts[li][i])));
      }
    }
    return { dWs, dbs };
  }

  function applyGrads(net, grads, lr) {
    return {
      ...net,
      layers: net.layers.map((L, i) => ({
        W: L.W.map((row, j) => row.map((w, k) => w - lr * grads.dWs[i][j][k])),
        b: L.b.map((b, j) => b - lr * grads.dbs[i][j]),
      })),
    };
  }

  const state = {
    hidden: [4, 4],
    activation: 'tanh',
    datasetKey: 'circle',
    data: { circle: genData('circle'), xor: genData('xor'), spiral: genData('spiral'), gaussian: genData('gaussian') },
    net: makeNet([2, 4, 4, 1], 7),
    training: false,
    timer: null,
    epoch: 0,
    lr: 0.05,
    loss: 0,
  };

  const datasetsEl = document.getElementById('datasets');
  const archLabel = document.getElementById('arch-label');
  const layerSlidersEl = document.getElementById('layer-sliders');
  const addLayerBtn = document.getElementById('add-layer');
  const removeLayerBtn = document.getElementById('remove-layer');
  const activationsEl = document.getElementById('activations');
  const lrSlider = document.getElementById('lr-slider');
  const lrLabel = document.getElementById('lr-label');
  const trainBtn = document.getElementById('train-btn');
  const resetBtn = document.getElementById('reset-btn');
  const readoutEl = document.getElementById('readout');
  const canvas = document.getElementById('canvas');
  const netCanvas = document.getElementById('net-canvas');

  ['gaussian', 'circle', 'xor', 'spiral'].forEach((k) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = k;
    b.dataset.dataset = k;
    b.addEventListener('click', () => {
      state.datasetKey = k;
      state.epoch = 0;
      render();
    });
    datasetsEl.appendChild(b);
  });
  ['tanh', 'relu'].forEach((k) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = k;
    b.dataset.act = k;
    b.addEventListener('click', () => {
      state.activation = k;
      rebuildNet();
    });
    activationsEl.appendChild(b);
  });

  function rebuildNet() {
    stopTrain();
    state.net = makeNet([2, ...state.hidden, 1], Math.floor(Math.random() * 100) + 1);
    state.epoch = 0;
    state.loss = 0;
    render();
  }
  resetBtn.addEventListener('click', rebuildNet);

  function renderLayerSliders() {
    layerSlidersEl.innerHTML = '';
    state.hidden.forEach((n, i) => {
      const row = document.createElement('div');
      row.style.marginBottom = '10px';
      row.innerHTML = `
        <div class="ml-param-head">
          <span class="label">layer ${i + 1}</span><b>${n} neurons</b>
        </div>
        <input type="range" min="1" max="8" step="1" value="${n}" />`;
      const inp = row.querySelector('input');
      inp.addEventListener('input', (e) => {
        state.hidden = state.hidden.map((s, j) => (j === i ? +e.target.value : s));
        rebuildNet();
      });
      layerSlidersEl.appendChild(row);
    });
  }

  addLayerBtn.addEventListener('click', () => {
    if (state.hidden.length >= 4) return;
    state.hidden = [...state.hidden, 4];
    rebuildNet();
  });
  removeLayerBtn.addEventListener('click', () => {
    if (state.hidden.length <= 1) return;
    state.hidden = state.hidden.slice(0, -1);
    rebuildNet();
  });

  lrSlider.addEventListener('input', (e) => {
    state.lr = +e.target.value;
    lrLabel.textContent = `η = ${state.lr.toFixed(3)}`;
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
      const pts = state.data[state.datasetKey];
      for (const p of pts) {
        const f = forward(nn, [p.x, p.y], state.activation);
        const g = backprop(nn, f, p.label, state.activation);
        nn = applyGrads(nn, g, state.lr);
        const out = f.acts[f.acts.length - 1][0];
        totalLoss += -(p.label * Math.log(out + 1e-9) + (1 - p.label) * Math.log(1 - out + 1e-9));
      }
      state.net = nn;
      state.loss = totalLoss / pts.length;
      state.epoch++;
      render();
    }, 50);
  }
  function stopTrain() {
    state.training = false;
    trainBtn.textContent = '▷ Train';
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  function accuracy() {
    const pts = state.data[state.datasetKey];
    let c = 0;
    for (const p of pts) {
      const f = forward(state.net, [p.x, p.y], state.activation);
      if ((f.acts[f.acts.length - 1][0] > 0.5 ? 1 : 0) === p.label) c++;
    }
    return c / pts.length;
  }

  // ───────── Boundary canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, drawBoundary, { square: true });

  function drawBoundary(ctx, size) {
    const R = 3;
    const tx = (x) => (x + R) / (2 * R) * size.w;
    const ty = (y) => size.h - (y + R) / (2 * R) * size.h;

    const cells = 60;
    for (let i = 0; i < cells; i++) {
      for (let j = 0; j < cells; j++) {
        const x = -R + (j + 0.5) / cells * 2 * R;
        const y = R - (i + 0.5) / cells * 2 * R;
        const f = forward(state.net, [x, y], state.activation);
        const conf = f.acts[f.acts.length - 1][0] * 2 - 1;
        ctx.fillStyle = conf > 0
          ? `rgba(122,31,36,${0.05 + Math.abs(conf) * 0.4})`
          : `rgba(38,35,32,${0.04 + Math.abs(conf) * 0.25})`;
        ctx.fillRect(j * size.w / cells, i * size.h / cells, size.w / cells + 1, size.h / cells + 1);
      }
    }
    ctx.strokeStyle = 'rgba(38,35,32,0.08)';
    ctx.lineWidth = 1;
    for (let i = -3; i <= 3; i++) {
      ctx.beginPath(); ctx.moveTo(tx(i), 0); ctx.lineTo(tx(i), size.h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, ty(i)); ctx.lineTo(size.w, ty(i)); ctx.stroke();
    }
    for (const p of state.data[state.datasetKey]) {
      ctx.fillStyle = p.label === 1 ? ACCENT : '#262320';
      ctx.beginPath();
      ctx.arc(tx(p.x), ty(p.y), 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#fffdf6';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(tx(p.x), ty(p.y), 4, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // ───────── Network anatomy canvas ─────────
  function drawNet() {
    const cv = netCanvas;
    const dpr = window.devicePixelRatio || 1;
    const W = cv.parentElement.getBoundingClientRect().width, H = 200;
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = W + 'px'; cv.style.height = H + 'px';
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const sizes = state.net.sizes;
    const colXs = sizes.map((_, i) => 40 + (i / (sizes.length - 1)) * (W - 80));
    const colY = (n, i) => H * 0.5 + (i - (n - 1) / 2) * Math.min(28, (H - 40) / Math.max(...sizes));
    const r = 10;

    for (let li = 0; li < state.net.layers.length; li++) {
      const L = state.net.layers[li];
      for (let i = 0; i < sizes[li + 1]; i++) {
        for (let j = 0; j < sizes[li]; j++) {
          const w = L.W[i][j];
          const a = Math.min(0.7, Math.abs(w) * 0.5 + 0.1);
          ctx.strokeStyle = w >= 0 ? `rgba(122,31,36,${a})` : `rgba(38,35,32,${a})`;
          ctx.lineWidth = 0.4 + Math.min(2, Math.abs(w) * 0.6);
          ctx.beginPath();
          ctx.moveTo(colXs[li] + r, colY(sizes[li], j));
          ctx.lineTo(colXs[li + 1] - r, colY(sizes[li + 1], i));
          ctx.stroke();
        }
      }
    }
    for (let li = 0; li < sizes.length; li++) {
      for (let i = 0; i < sizes[li]; i++) {
        ctx.fillStyle = '#fffdf6';
        ctx.strokeStyle = '#262320';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(colXs[li], colY(sizes[li], i), r, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();
      }
      ctx.fillStyle = INK_FADE;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      const lbl = li === 0 ? 'input' : (li === sizes.length - 1 ? 'output' : `hidden ${li}`);
      ctx.fillText(lbl, colXs[li], H - 8);
    }
  }
  new ResizeObserver(drawNet).observe(netCanvas.parentElement);

  function render() {
    Array.from(datasetsEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.dataset === state.datasetKey);
    });
    Array.from(activationsEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.act === state.activation);
    });
    archLabel.textContent = `2 → ${state.hidden.join(' → ')} → 1`;
    renderLayerSliders();
    readoutEl.innerHTML = `
      <div class="row"><span>epoch</span><b>${state.epoch}</b></div>
      <div class="row"><span>loss</span><b>${formatNum(state.loss)}</b></div>
      <div class="row"><span>accuracy</span><b>${(accuracy() * 100).toFixed(1)}%</b></div>`;
    canvasCtl.redraw();
    drawNet();
  }

  render();
})();
