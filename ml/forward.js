/* Vol. X — Forward Propagation */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK_FADE } = V;

  // 2 → 4 → 4 → 2 network with hand-tuned weights (no training; we visualize a fixed model).
  const NET = {
    layers: [
      {
        W: [[ 1.2, -0.8], [0.5,  1.4], [-1.1, -0.4], [-0.3,  1.0]],
        b: [0.2, -0.1, 0.3, -0.2], act: 'tanh',
      },
      {
        W: [[ 0.8, -0.6,  0.4, -0.2], [-0.5,  1.1, -0.3,  0.7],
            [ 0.6,  0.4, -0.9,  0.5], [-0.4, -0.7,  0.6,  0.9]],
        b: [0.1, -0.2, 0.3, 0.0], act: 'tanh',
      },
      {
        W: [[ 1.2, -0.9,  0.6, -0.4], [-0.8,  1.0, -0.5,  0.7]],
        b: [0.0, 0.0], act: 'softmax',
      },
    ],
  };

  const tanh = Math.tanh;
  function softmax(arr) {
    const m = Math.max(...arr);
    const ex = arr.map((v) => Math.exp(v - m));
    const s = ex.reduce((a, b) => a + b, 0);
    return ex.map((v) => v / s);
  }

  function forward(input) {
    const trace = [{ values: input.slice(), kind: 'input' }];
    let a = input.slice();
    for (const L of NET.layers) {
      const z = L.W.map((row, i) => row.reduce((s, w, j) => s + w * a[j], 0) + L.b[i]);
      const out = L.act === 'softmax' ? softmax(z) : z.map(tanh);
      trace.push({ values: z, kind: 'pre' });
      trace.push({ values: out, kind: 'post' });
      a = out;
    }
    return trace;
  }

  const state = {
    x1: 0.7, x2: -0.4,
    step: 6,
    playing: false,
    timer: null,
  };

  // ───────── DOM ─────────
  const inputsEl = document.getElementById('inputs');
  const replayBtn = document.getElementById('replay-btn');
  const stepSlider = document.getElementById('step-slider');
  const stepLabel = document.getElementById('step-label');
  const presetsEl = document.getElementById('presets');
  const captionEl = document.getElementById('caption');
  const probReadoutEl = document.getElementById('prob-readout');
  const canvas = document.getElementById('canvas');

  // Build input sliders
  function renderInputs() {
    inputsEl.innerHTML = '';
    [['x1', 'x₁'], ['x2', 'x₂']].forEach(([k, lab]) => {
      const row = document.createElement('div');
      row.innerHTML = `
        <div class="ml-param-head">
          <span class="label">${lab}</span><b>${formatNum(state[k])}</b>
        </div>
        <input type="range" min="-2" max="2" step="0.05" value="${state[k]}" />`;
      const inp = row.querySelector('input');
      inp.addEventListener('input', (e) => {
        stopPlay();
        state[k] = +e.target.value;
        state.step = 6;
        render();
      });
      inputsEl.appendChild(row);
    });
  }

  // Presets — corner regions
  const PRESETS = [
    { name: '↗ upper-right', x1: 0.8,  x2: 0.6 },
    { name: '↖ upper-left',  x1: -1.2, x2: 0.7 },
    { name: '↙ lower-left',  x1: -0.8, x2: -0.9 },
    { name: '↘ lower-right', x1: 1.0,  x2: -0.6 },
  ];
  PRESETS.forEach((p) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = p.name;
    b.addEventListener('click', () => {
      stopPlay();
      state.x1 = p.x1; state.x2 = p.x2;
      state.step = 0;
      startPlay();
    });
    presetsEl.appendChild(b);
  });

  replayBtn.addEventListener('click', () => {
    state.step = 0;
    startPlay();
  });

  stepSlider.addEventListener('input', (e) => {
    stopPlay();
    state.step = +e.target.value;
    render();
  });

  function startPlay() {
    state.playing = true;
    if (state.timer) clearInterval(state.timer);
    state.timer = setInterval(() => {
      if (state.step >= 6) { stopPlay(); return; }
      state.step++;
      render();
    }, 600);
    render();
  }
  function stopPlay() {
    state.playing = false;
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    ctx.fillStyle = '#fffdf6';
    ctx.fillRect(0, 0, size.w, size.h);

    const trace = forward([state.x1, state.x2]);
    const cols = [
      { idx: 0, n: 2, x: size.w * 0.10, label: 'input',    sub: 'x' },
      { idx: 2, n: 4, x: size.w * 0.36, label: 'hidden 1', sub: 'a⁽¹⁾' },
      { idx: 4, n: 4, x: size.w * 0.62, label: 'hidden 2', sub: 'a⁽²⁾' },
      { idx: 6, n: 2, x: size.w * 0.88, label: 'output',   sub: 'ŷ' },
    ];
    const colY = (n, i) => size.h * 0.5 + (i - (n - 1) / 2) * Math.min(80, size.h * 0.18);

    // Determine which connection-set is "active" (mid-flow) based on step.
    // step 1: pre1 → col 0→1; step 3: pre2 → col 1→2; step 5: pre3 → col 2→3.
    let activeConn = -1;
    if (state.step === 1) activeConn = 0;
    if (state.step === 3) activeConn = 1;
    if (state.step === 5) activeConn = 2;

    // Draw connections
    for (let li = 0; li < 3; li++) {
      const fromCol = cols[li], toCol = cols[li + 1];
      const W = NET.layers[li].W;
      const isActive = activeConn === li;
      const isPast = state.step > li * 2 + 2 || (activeConn !== li && state.step > li * 2);
      for (let i = 0; i < toCol.n; i++) {
        for (let j = 0; j < fromCol.n; j++) {
          const w = W[i][j];
          const x1 = fromCol.x + 22, y1 = colY(fromCol.n, j);
          const x2 = toCol.x - 22,    y2 = colY(toCol.n, i);
          const baseAlpha = isActive ? 0.45 : (isPast ? 0.20 : 0.08);
          const alpha = baseAlpha * Math.min(1, Math.abs(w) * 0.8 + 0.3);
          ctx.strokeStyle = w > 0 ? `rgba(122,31,36,${alpha})` : `rgba(38,35,32,${alpha})`;
          ctx.lineWidth = isActive ? 1.2 + Math.abs(w) * 0.8 : 0.6 + Math.abs(w) * 0.5;
          ctx.beginPath();
          ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
          ctx.stroke();
        }
      }
    }

    // Neurons
    cols.forEach((col, ci) => {
      const isLit = state.step >= col.idx;
      const values = isLit ? trace[col.idx].values : null;
      for (let i = 0; i < col.n; i++) {
        const x = col.x, y = colY(col.n, i);
        const v = values ? values[i] : 0;
        const mag = Math.min(1, Math.abs(v));
        ctx.fillStyle = isLit
          ? (v >= 0 ? `rgba(122,31,36,${0.15 + mag * 0.7})` : `rgba(38,35,32,${0.15 + mag * 0.7})`)
          : '#fffdf6';
        ctx.strokeStyle = isLit ? '#262320' : 'rgba(38,35,32,0.30)';
        ctx.lineWidth = isLit ? 1.5 : 1;
        ctx.beginPath();
        ctx.arc(x, y, 22, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();
        if (isLit) {
          ctx.fillStyle = mag > 0.5 ? '#fffdf6' : '#262320';
          ctx.font = 'bold 11px "JetBrains Mono", monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(formatNum(v), x, y);
        }
      }
      // Column labels
      ctx.fillStyle = INK_FADE;
      ctx.font = '11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(col.label, col.x, colY(col.n, col.n - 1) + Math.min(46, size.h * 0.10));
      ctx.fillStyle = 'rgba(38,35,32,0.4)';
      ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
      ctx.fillText(col.sub, col.x, colY(col.n, 0) - 38);
    });
  }

  function describe(step) {
    if (step === 0) return 'The input vector — two raw numbers entering the network.';
    if (step === 1) return 'Layer 1 pre-activation: weighted sum + bias, no bend yet.';
    if (step === 2) return 'Layer 1 post-activation: bent through tanh, squashed into (−1, 1).';
    if (step === 3) return 'Layer 2 pre-activation: each new neuron mixes all four previous activations.';
    if (step === 4) return 'Layer 2 post-activation: another tanh, signals further reshaped.';
    if (step === 5) return 'Output pre-activation: two raw scores (logits).';
    return 'Output post-softmax: a probability distribution over the two classes.';
  }

  // ───────── Render ─────────
  function render() {
    renderInputs();
    stepSlider.value = state.step;
    stepLabel.textContent = `${state.step}/6`;

    captionEl.innerHTML = `<span class="ml-cap-num">·</span> ${describe(state.step)}`;

    const trace = forward([state.x1, state.x2]);
    const out = trace[trace.length - 1].values;
    const c0 = out[0], c1 = out[1];
    probReadoutEl.innerHTML = `
      <div class="row"><span>P(class 0)</span><b style="color:${c0 > c1 ? 'var(--accent)' : 'var(--ink)'}">${(c0 * 100).toFixed(1)}%</b></div>
      <div class="row"><span>P(class 1)</span><b style="color:${c1 > c0 ? 'var(--accent)' : 'var(--ink)'}">${(c1 * 100).toFixed(1)}%</b></div>`;

    canvasCtl.redraw();
  }

  render();
})();
