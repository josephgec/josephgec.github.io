/* Vol. X — Forward Propagation */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, INK_FADE } = V;

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
    const logits = []; // pre-softmax outputs of last layer
    for (let li = 0; li < NET.layers.length; li++) {
      const L = NET.layers[li];
      const z = L.W.map((row, i) => row.reduce((s, w, j) => s + w * a[j], 0) + L.b[i]);
      const out = L.act === 'softmax' ? softmax(z) : z.map(tanh);
      trace.push({ values: z, kind: 'pre' });
      trace.push({ values: out, kind: 'post' });
      if (li === NET.layers.length - 1) logits.push(...z);
      a = out;
    }
    trace.logits = logits;
    return trace;
  }

  const state = {
    x1: 0.7, x2: -0.4,
    step: 6,
    playing: false,
    timer: null,
    // Animation: smooth particle progress in [0, 3]; integer part = which layer's particles, fractional = travel.
    flow: 3,    // 0 = pre-flow, 3 = all flowed
    flowTimer: null,
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
        <input type="range" min="-2" max="2" step="0.05" value="${state[k]}" aria-label="${lab}" />`;
      const inp = row.querySelector('input');
      inp.addEventListener('input', (e) => {
        stopPlay();
        state[k] = +e.target.value;
        state.step = 6;
        state.flow = 3;
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
      state.flow = 0;
      startPlay();
    });
    presetsEl.appendChild(b);
  });

  replayBtn.addEventListener('click', () => {
    state.step = 0;
    state.flow = 0;
    startPlay();
  });

  stepSlider.addEventListener('input', (e) => {
    stopPlay();
    state.step = +e.target.value;
    state.flow = state.step / 2;
    render();
  });

  // Smooth animation: particles travel layer-by-layer, each layer takes ~700ms.
  function startPlay() {
    state.playing = true;
    if (state.flowTimer) cancelAnimationFrame(state.flowTimer);
    let lastT = performance.now();
    const tick = (t) => {
      const dt = (t - lastT) / 1000;
      lastT = t;
      // 3 segments, ~0.7s each → speed ≈ 1.4 segments/sec
      state.flow = Math.min(3, state.flow + dt * 1.4);
      // Snap step to nearest integer so the existing UI / readout still works.
      state.step = Math.min(6, Math.floor(state.flow * 2 + 0.001));
      render();
      if (state.flow < 3) {
        state.flowTimer = requestAnimationFrame(tick);
      } else {
        state.playing = false;
        state.flowTimer = null;
      }
    };
    state.flowTimer = requestAnimationFrame(tick);
  }
  function stopPlay() {
    state.playing = false;
    if (state.flowTimer) { cancelAnimationFrame(state.flowTimer); state.flowTimer = null; }
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
      { idx: 6, n: 2, x: size.w * 0.86, label: 'output',   sub: 'ŷ' },
    ];
    const colY = (n, i) => size.h * 0.5 + (i - (n - 1) / 2) * Math.min(80, size.h * 0.18);

    // Determine flow per-layer: layer li animates while flow ∈ [li, li+1].
    // segProg: progress within the current layer, 0..1; -1 means not started.
    function layerSegProg(li) {
      if (state.flow >= li + 1) return 1;        // fully past
      if (state.flow <= li) return -1;            // not yet
      return state.flow - li;                     // in flight
    }

    // Draw connections (always drawn faintly so the topology is visible).
    for (let li = 0; li < 3; li++) {
      const fromCol = cols[li], toCol = cols[li + 1];
      const W = NET.layers[li].W;
      const seg = layerSegProg(li);
      const isActive = seg > 0 && seg < 1;
      const isPast = seg >= 1;
      for (let i = 0; i < toCol.n; i++) {
        for (let j = 0; j < fromCol.n; j++) {
          const w = W[i][j];
          const x1 = fromCol.x + 22, y1 = colY(fromCol.n, j);
          const x2 = toCol.x - 22,    y2 = colY(toCol.n, i);
          const baseAlpha = isActive ? 0.45 : (isPast ? 0.20 : 0.10);
          const alpha = baseAlpha * Math.min(1, Math.abs(w) * 0.8 + 0.3);
          ctx.strokeStyle = w > 0 ? `rgba(122,31,36,${alpha})` : `rgba(38,35,32,${alpha})`;
          ctx.lineWidth = isActive ? 1.2 + Math.abs(w) * 0.7 : 0.6 + Math.abs(w) * 0.5;
          ctx.beginPath();
          ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
          ctx.stroke();
        }
      }
    }

    // Particles flowing along edges of the active layer.
    for (let li = 0; li < 3; li++) {
      const seg = layerSegProg(li);
      if (seg <= 0 || seg >= 1) continue;
      const fromCol = cols[li], toCol = cols[li + 1];
      const W = NET.layers[li].W;
      const fromVals = trace[cols[li].idx].values; // post-activations of source col (or input)
      for (let i = 0; i < toCol.n; i++) {
        for (let j = 0; j < fromCol.n; j++) {
          const w = W[i][j];
          const a = fromVals[j];
          const contrib = Math.abs(w * a);
          if (contrib < 0.02) continue;
          const x1 = fromCol.x + 22, y1 = colY(fromCol.n, j);
          const x2 = toCol.x - 22,    y2 = colY(toCol.n, i);
          const px = x1 + (x2 - x1) * seg;
          const py = y1 + (y2 - y1) * seg;
          const r = 1.5 + Math.min(3, contrib * 2.5);
          ctx.fillStyle = (w * a) >= 0 ? 'rgba(122,31,36,0.95)' : 'rgba(38,35,32,0.95)';
          ctx.beginPath();
          ctx.arc(px, py, r, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // Neurons
    cols.forEach((col, ci) => {
      // A column "lights" when the particles arriving at it have completed their travel.
      // ci=0 is always lit (input); ci=k lit when state.flow ≥ k.
      const isLit = ci === 0 || state.flow >= ci;
      // Brief halo while particles are arriving (segProg of the upstream layer is 0..1).
      const incomingSeg = ci > 0 ? layerSegProg(ci - 1) : 1;
      const arriving = ci > 0 && incomingSeg > 0 && incomingSeg < 1;
      const values = isLit ? trace[col.idx].values : null;
      for (let i = 0; i < col.n; i++) {
        const x = col.x, y = colY(col.n, i);
        const v = values ? values[i] : 0;
        const mag = Math.min(1, Math.abs(v));
        // Pulse halo while particles arrive (grows with segProg).
        if (arriving) {
          const a = 0.10 + 0.25 * incomingSeg;
          ctx.fillStyle = `rgba(122,31,36,${a})`;
          ctx.beginPath();
          ctx.arc(x, y, 22 + 6 * incomingSeg, 0, Math.PI * 2);
          ctx.fill();
        }
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

    // Logits → probabilities mini-chart at the right edge.
    if (state.flow >= 3 || state.step >= 5) {
      drawLogitToProb(ctx, size, trace);
    }

    // "Predicted class" callout at the top — only when the forward pass has fully landed.
    if (state.flow >= 3) {
      const probs = trace[trace.length - 1].values;
      const winIdx = probs[0] >= probs[1] ? 0 : 1;
      const winPct = (probs[winIdx] * 100).toFixed(0);
      const txt = `Prediction: class c${winIdx} (${winPct}%)`;
      ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
      const tw = ctx.measureText(txt).width + 18;
      const tx0 = (size.w - tw) / 2;
      ctx.fillStyle = 'rgba(255,253,246,0.92)';
      ctx.strokeStyle = 'rgba(122,31,36,0.6)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.rect(tx0, 8, tw, 22);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#7a1f24';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(txt, size.w / 2, 19);
    }
  }

  // Small bar chart adjacent to the output column showing logits → softmax → probs.
  function drawLogitToProb(ctx, size, trace) {
    const logits = trace.logits;
    const probs = trace[trace.length - 1].values;
    const xR = size.w - 6;
    const xL = size.w - 96;
    const yC = size.h * 0.5;
    // Background card
    ctx.fillStyle = 'rgba(255,253,246,0.85)';
    ctx.strokeStyle = 'rgba(38,35,32,0.18)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(xL, yC - 76, xR - xL, 130);
    ctx.fill(); ctx.stroke();
    // Heading
    ctx.fillStyle = INK_FADE;
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    ctx.fillText('LOGITS  →  PROBS', xL + 6, yC - 64);
    // Bars
    const maxLogit = Math.max(0.5, ...logits.map(Math.abs));
    const labelW = 22;
    const colWLogit = 28, colWProb = 28, gapMid = 8;
    const groupX = xL + labelW + 6;
    const colY0 = yC - 50;
    const colY1 = yC + 40;
    const colH = colY1 - colY0;
    for (let i = 0; i < 2; i++) {
      const yMid = colY0 + colH * (0.25 + i * 0.5);
      // Class label
      ctx.fillStyle = '#262320';
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(`c${i}`, xL + 6, yMid);
      // Logit bar — faint, signed (centered baseline)
      const lx = groupX;
      const lcx = lx + colWLogit / 2;
      const lh = (logits[i] / maxLogit) * 18;
      ctx.fillStyle = 'rgba(38,35,32,0.30)';
      if (lh >= 0) ctx.fillRect(lx, yMid - lh, colWLogit, lh);
      else         ctx.fillRect(lx, yMid,      colWLogit, -lh);
      ctx.strokeStyle = 'rgba(38,35,32,0.45)';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(lx - 1, yMid); ctx.lineTo(lx + colWLogit + 1, yMid);
      ctx.stroke();
      ctx.fillStyle = INK_FADE;
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(logits[i].toFixed(2), lcx, yMid + (lh >= 0 ? 16 : -14));
      // Arrow
      ctx.strokeStyle = 'rgba(38,35,32,0.45)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(lx + colWLogit + 2, yMid); ctx.lineTo(lx + colWLogit + gapMid - 2, yMid);
      ctx.stroke();
      // Prob bar — accent, magnitude only
      const px = lx + colWLogit + gapMid;
      const ph = probs[i] * 32;
      const isWin = probs[i] === Math.max(probs[0], probs[1]);
      ctx.fillStyle = isWin ? 'rgba(122,31,36,0.85)' : 'rgba(122,31,36,0.30)';
      ctx.fillRect(px, yMid - ph / 2, colWProb, ph);
      ctx.strokeStyle = 'rgba(38,35,32,0.4)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px, yMid + ph / 2 + 1); ctx.lineTo(px + colWProb, yMid + ph / 2 + 1);
      ctx.stroke();
      ctx.fillStyle = isWin ? '#7a1f24' : '#262320';
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`${(probs[i] * 100).toFixed(0)}%`, px + colWProb / 2, yMid + ph / 2 + 12);
    }
    // Footnote
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 10px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('softmax', xL + 6, yC + 50);
  }

  function describe(step) {
    if (step === 0) return 'The input vector — two raw numbers entering the network.';
    if (step === 1) return 'Layer 1 pre-activation: weighted sum + bias, no bend yet.';
    if (step === 2) return 'Layer 1 post-activation: bent through tanh — the s-shaped curve that squashes any real number into (−1, 1).';
    if (step === 3) return 'Layer 2 pre-activation: each new neuron mixes all four previous activations.';
    if (step === 4) return 'Layer 2 post-activation: another tanh, signals further reshaped.';
    if (step === 5) return 'Output pre-activation: two raw scores (logits — pre-softmax linear outputs).';
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
    const l0 = trace.logits[0], l1 = trace.logits[1];
    probReadoutEl.innerHTML = `
      <div class="row"><span>logit c0</span><b style="font-family:var(--mono)">${formatNum(l0)}</b></div>
      <div class="row"><span>logit c1</span><b style="font-family:var(--mono)">${formatNum(l1)}</b></div>
      <div class="row"><span>P(c0)</span><b style="color:${c0 > c1 ? 'var(--accent)' : 'var(--ink)'}">${(c0 * 100).toFixed(1)}%</b></div>
      <div class="row"><span>P(c1)</span><b style="color:${c1 > c0 ? 'var(--accent)' : 'var(--ink)'}">${(c1 * 100).toFixed(1)}%</b></div>`;

    canvasCtl.redraw();
  }

  render();
})();
