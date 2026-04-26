/* Vol. XVIII — Gates & Long Memory (LSTMs) */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, INK, INK_FADE, ACCENT } = V;

  const VOCAB = ['a', 'b', 'r', 'c', 'd', ' '];
  const oneHot = (c) => VOCAB.map((v) => (v === c ? 1 : 0));
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const tanh = Math.tanh;

  function seeded(seed) {
    let s = seed;
    return () => { s = (s * 9301 + 49297) % 233280; return (s / 233280 - 0.5) * 1.4; };
  }
  function makeLSTM(H, D, seed) {
    const r = seeded(seed);
    function W(rows, cols) { return Array.from({ length: rows }, () => Array.from({ length: cols }, r)); }
    return {
      Wf: W(H, D + H), Wi: W(H, D + H), Wo: W(H, D + H), Wc: W(H, D + H),
      bf: new Array(H).fill(1.0),  // bias forget gate slightly open by default (common practice)
      bi: new Array(H).fill(0), bo: new Array(H).fill(0), bc: new Array(H).fill(0),
      H, D,
    };
  }
  function lstmStep(net, x, h, c, force) {
    const concat = h.concat(x);
    const apply = (W, b) => W.map((row, i) => row.reduce((s, w, j) => s + w * concat[j], 0) + b[i]);
    let f = apply(net.Wf, net.bf).map(sigmoid);
    let i = apply(net.Wi, net.bi).map(sigmoid);
    let o = apply(net.Wo, net.bo).map(sigmoid);
    const cTilde = apply(net.Wc, net.bc).map(tanh);
    if (force === 'f-open')   f = f.map(() => 1);
    if (force === 'f-closed') f = f.map(() => 0);
    if (force === 'i-open')   i = i.map(() => 1);
    if (force === 'i-closed') i = i.map(() => 0);
    if (force === 'o-open')   o = o.map(() => 1);
    if (force === 'o-closed') o = o.map(() => 0);
    const cNew = c.map((cv, k) => f[k] * cv + i[k] * cTilde[k]);
    const hNew = cNew.map((cv, k) => o[k] * tanh(cv));
    return { h: hNew, c: cNew, f, i, o, cTilde };
  }

  // Vanilla RNN — used for the LSTM-vs-RNN comparison.
  function makeRNN(H, D, seed) {
    const r = seeded(seed + 100);
    function W(rows, cols) { return Array.from({ length: rows }, () => Array.from({ length: cols }, r)); }
    // Scale up Whh on purpose so vanilla RNN cell-state-equivalent drifts/explodes.
    const Whh = W(H, H).map((row) => row.map((v) => v * 1.6));
    return { Wxh: W(H, D), Whh, b: new Array(H).fill(0), H, D };
  }
  function rnnStep(net, x, h) {
    const out = new Array(net.H);
    for (let i = 0; i < net.H; i++) {
      let s = net.b[i];
      for (let j = 0; j < net.D; j++) s += net.Wxh[i][j] * x[j];
      for (let j = 0; j < net.H; j++) s += net.Whh[i][j] * h[j];
      out[i] = tanh(s);
    }
    return out;
  }

  const VIEWS = [
    { key: 'cell',    label: 'Inside the cell · schematic' },
    { key: 'tape',    label: 'Across time · gates over t' },
    { key: 'compare', label: 'LSTM vs vanilla RNN' },
  ];
  const CAPTIONS = {
    cell: 'The cell-state bus c runs across the top — a near-linear conveyor belt. Three sigmoid valves pinch or open it: forget (multiplies the bus), input (gates new candidate memory before it merges), output (gates what leaves as h_t). Each valve\'s fill shows how open it is for the current step. Watch them open and close as you scrub through the sequence.',
    tape: 'Each row is one component of the cell over time. Top to bottom: forget f, input i, output o, candidate, cell c, hidden h. The gates are 0–1 valves; the cell c drifts gently while the hidden state h fluctuates per step.',
    compare: 'Same input, two architectures. The LSTM\'s cell state stays bounded — its gates regulate inflow. The vanilla RNN has no such regulator: every step is multiplied by the same recurrent matrix, and the hidden state drifts toward saturation. This is why long sequences need gating.',
  };

  const state = {
    seq: 'abracadabra',
    seed: 7,
    step: 11,
    force: 'none',
    view: 'cell',
    showRNN: false,
    timer: null,
  };

  const seqInput = document.getElementById('seq-input');
  const stepSlider = document.getElementById('step-slider');
  const stepLabel = document.getElementById('step-label');
  const replayBtn = document.getElementById('replay-btn');
  const seedBtn = document.getElementById('seed-btn');
  const gateTogglesEl = document.getElementById('gate-toggles');
  const compareTogglesEl = document.getElementById('compare-toggles');
  const tabsEl = document.getElementById('view-tabs');
  const captionEl = document.getElementById('caption-text');
  const canvas = document.getElementById('canvas');

  ['none', 'f-closed', 'f-open', 'i-closed', 'i-open', 'o-closed', 'o-open'].forEach((k) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = k === 'none' ? 'normal' : k.replace('-', ' ');
    b.dataset.force = k;
    b.addEventListener('click', () => { state.force = k; render(); });
    gateTogglesEl.appendChild(b);
  });

  ['LSTM only', 'LSTM + RNN'].forEach((lbl, idx) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = lbl;
    b.dataset.cmp = idx === 0 ? 'off' : 'on';
    b.addEventListener('click', () => {
      state.showRNN = idx === 1;
      if (state.showRNN) state.view = 'compare';
      render();
    });
    compareTogglesEl.appendChild(b);
  });

  VIEWS.forEach((v) => {
    const b = document.createElement('button');
    b.textContent = v.label;
    b.dataset.view = v.key;
    b.addEventListener('click', () => {
      state.view = v.key;
      if (v.key === 'compare') state.showRNN = true;
      render();
    });
    tabsEl.appendChild(b);
  });

  seqInput.addEventListener('input', (e) => {
    state.seq = e.target.value.replace(/[^abrcd ]/g, '').slice(0, 14);
    state.step = state.seq.length;
    seqInput.value = state.seq;
    stepSlider.max = state.seq.length;
    render();
  });
  stepSlider.addEventListener('input', (e) => { state.step = +e.target.value; render(); });
  replayBtn.addEventListener('click', () => {
    state.step = 0;
    if (state.timer) clearInterval(state.timer);
    state.timer = setInterval(() => {
      if (state.step >= state.seq.length) { clearInterval(state.timer); state.timer = null; return; }
      state.step++;
      render();
    }, 600);
    render();
  });
  seedBtn.addEventListener('click', () => { state.seed = Math.floor(Math.random() * 100) + 1; render(); });

  function trace() {
    const H = 6;
    const net = makeLSTM(H, VOCAB.length, state.seed);
    let h = new Array(H).fill(0);
    let c = new Array(H).fill(0);
    const out = [{ h, c, f: new Array(H).fill(0), i: new Array(H).fill(0), o: new Array(H).fill(0), cTilde: new Array(H).fill(0) }];
    for (let t = 0; t < state.seq.length; t++) {
      const x = oneHot(state.seq[t]);
      const r = lstmStep(net, x, h, c, state.force === 'none' ? null : state.force);
      h = r.h; c = r.c;
      out.push({ h, c, f: r.f, i: r.i, o: r.o, cTilde: r.cTilde });
    }
    return out;
  }
  function traceRNN() {
    const H = 6;
    const net = makeRNN(H, VOCAB.length, state.seed);
    let h = new Array(H).fill(0);
    const out = [h.slice()];
    for (let t = 0; t < state.seq.length; t++) {
      const x = oneHot(state.seq[t]);
      h = rnnStep(net, x, h);
      out.push(h.slice());
    }
    return out;
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    if (state.view === 'cell') drawCell(ctx, size);
    else if (state.view === 'compare') drawCompare(ctx, size);
    else drawTape(ctx, size);
  }

  // ───────── View 1: The canonical LSTM cell schematic ─────────
  function drawCell(ctx, size) {
    const trc = trace();
    const idx = Math.max(0, Math.min(state.step, state.seq.length));
    const cur = trc[idx];

    // Aggregate gate openness as a single mean for valve fill (0..1).
    const meanFn = (a) => a.reduce((s, v) => s + v, 0) / Math.max(1, a.length);
    const fOpen = idx === 0 ? 0 : meanFn(cur.f);
    const iOpen = idx === 0 ? 0 : meanFn(cur.i);
    const oOpen = idx === 0 ? 0 : meanFn(cur.o);

    // Layout
    const margin = 22;
    const W = size.w - margin * 2;
    const H = size.h - margin * 2;
    const x0 = margin, y0 = margin;

    // Cell-state bus: top horizontal line.
    const busY = y0 + H * 0.18;
    const busLeft = x0 + 30;
    const busRight = x0 + W - 30;

    // Hidden output bus: just below cell bus.
    const hOutY = y0 + H * 0.40;

    // Input lane (h_{t-1} concat x_t): bottom horizontal lane.
    const inY = y0 + H * 0.84;

    // Gate column x positions (forget, input/candidate, output)
    const fX = x0 + W * 0.27;
    const iX = x0 + W * 0.50;
    const oX = x0 + W * 0.78;

    // ── Frame the LSTM cell box ──
    ctx.strokeStyle = 'rgba(38,35,32,0.18)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(x0 + 14, y0 + H * 0.08, W - 28, H * 0.82);
    ctx.setLineDash([]);
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('LSTM CELL', x0 + 22, y0 + H * 0.08 + 14);
    // Storytelling: σ = sigmoid maps the input to a 0..1 valve openness.
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'right';
    ctx.fillText('σ = sigmoid · maps to (0, 1) valve openness · ⊙ = element-wise', x0 + W - 22, y0 + H * 0.08 + 14);

    // ── Cell-state bus (the conveyor belt) ──
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(busLeft, busY); ctx.lineTo(busRight, busY);
    ctx.stroke();
    // Bus label and incoming/outgoing carry
    ctx.fillStyle = INK;
    ctx.font = 'italic 14px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('c_{t−1}', busLeft - 60, busY + 4);
    ctx.textAlign = 'right';
    ctx.fillText('c_t →', busRight + 56, busY + 4);
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = INK_FADE;
    ctx.textAlign = 'center';
    ctx.fillText('CELL-STATE BUS · memory conveyor belt · old memory mostly passes through', (busLeft + busRight) / 2, busY - 12);

    // Carry-in and carry-out tiny arrows
    drawArrow(ctx, busLeft - 24, busY, busLeft - 4, busY, ACCENT, 2, 7);
    drawArrow(ctx, busRight + 4, busY, busRight + 24, busY, ACCENT, 2, 7);

    // ── h_{t-1} → h_t arrow above the bus on the left, and out h_t at right ──
    // Actually keep h flow as separate lower line below the bus.
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(busLeft - 24, hOutY);
    ctx.lineTo(busLeft - 4, hOutY);
    ctx.stroke();
    ctx.fillStyle = INK;
    ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('h_{t−1}', busLeft - 60, hOutY + 4);

    // ── Input concat x_t and h_{t-1} on the bottom ──
    ctx.strokeStyle = INK_FADE;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(busLeft - 24, inY);
    ctx.lineTo(busRight + 4, inY);
    ctx.stroke();
    ctx.fillStyle = INK;
    ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('x_t', busLeft - 24, inY + 18);
    ctx.fillText('+ h_{t−1}', busLeft + 18, inY + 18);

    // Current input character indicator
    if (idx > 0 && idx <= state.seq.length) {
      const ch = state.seq[idx - 1] === ' ' ? '·' : state.seq[idx - 1];
      ctx.fillStyle = ACCENT;
      ctx.font = 'italic 22px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'left';
      ctx.fillText(`"${ch}"`, busRight - 60, inY + 18);
      ctx.fillStyle = INK_FADE;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillText(`t = ${idx}`, busRight - 60, inY + 32);
    }

    // ── Forget gate ── multiplicative pinch on the bus
    drawValve(ctx, fX, busY, fOpen, 'σ', '1 · forget');
    // tap from input lane up to gate
    ctx.strokeStyle = 'rgba(38,35,32,0.45)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(fX, inY);
    ctx.lineTo(fX, busY + 16);
    ctx.stroke();
    ctx.setLineDash([]);
    // gate label
    ctx.fillStyle = INK;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('f_t = σ(W_f·[h, x] + b_f)', fX, busY + 60);
    ctx.fillStyle = ACCENT;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillText(`f ≈ ${fOpen.toFixed(2)}`, fX, busY + 76);

    // ── Input gate + candidate (tanh) ──
    // candidate node below bus
    const candY = busY + 50;
    drawNode(ctx, iX - 30, candY, 'tanh', 'candidate');
    // input gate on the bus
    drawValve(ctx, iX, busY, iOpen, 'σ', '2 · input');
    // taps from input lane up to gate and candidate
    ctx.strokeStyle = 'rgba(38,35,32,0.45)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(iX, inY);
    ctx.lineTo(iX, busY + 16);
    ctx.moveTo(iX - 30, inY);
    ctx.lineTo(iX - 30, candY + 14);
    ctx.stroke();
    ctx.setLineDash([]);
    // arrow candidate → bus through input gate (the addition point)
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 2;
    drawArrow(ctx, iX - 30, candY - 14, iX - 6, busY + 4, ACCENT, 2, 7);
    // labels
    ctx.fillStyle = INK;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('i_t = σ(W_i·[h, x])', iX, busY + 96);
    ctx.fillText('c̃_t = tanh(W_c·[h, x])', iX - 30, busY + 110);
    ctx.fillStyle = ACCENT;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillText(`i ≈ ${iOpen.toFixed(2)}`, iX, busY + 76);

    // ⊕ symbol on the bus to indicate addition right of input gate
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(iX + 24, busY, 9, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = ACCENT;
    ctx.font = 'bold 12px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('+', iX + 24, busY);
    ctx.textBaseline = 'alphabetic';
    // tiny annotation: merge old kept memory + new gated candidate
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 9px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('merge: kept + new', iX + 24, busY - 14);

    // ── Output gate ──
    // tanh of cell state, then output gate, into hidden output line.
    const tanhY = busY + 24;
    drawNode(ctx, oX - 28, tanhY + 4, 'tanh', '');
    // line from bus down to tanh
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(oX - 28, busY);
    ctx.lineTo(oX - 28, tanhY - 6);
    ctx.stroke();

    drawValve(ctx, oX, busY, oOpen, 'σ', '3 · output');
    // tap from input lane up to output gate
    ctx.strokeStyle = 'rgba(38,35,32,0.45)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(oX, inY);
    ctx.lineTo(oX, busY + 16);
    ctx.stroke();
    ctx.setLineDash([]);
    // arrow tanh → output gate point on the hOutY line
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(oX - 28, tanhY + 18);
    ctx.lineTo(oX - 28, hOutY);
    ctx.lineTo(oX + 24, hOutY);
    ctx.stroke();
    drawArrow(ctx, oX + 24, hOutY, oX + 50, hOutY, INK, 1.5, 7);
    // h_t output exit
    ctx.fillStyle = INK;
    ctx.font = 'italic 14px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('h_t →', oX + 56, hOutY + 4);
    ctx.fillStyle = INK;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('o_t = σ(W_o·[h, x])', oX, busY + 76);
    ctx.fillText('h_t = o_t · tanh(c_t)', oX, busY + 92);
    ctx.fillStyle = ACCENT;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillText(`o ≈ ${oOpen.toFixed(2)}`, oX, busY + 60);

    // ── Tiny per-dimension bus tape underneath, anchored to bus position ──
    // shows c values across cell dims at this step.
    const tapeY = y0 + H - 14;
    const tapeLeft = x0 + W * 0.18;
    const tapeW = W * 0.64;
    const cellsN = cur.c.length;
    const cw = tapeW / cellsN;
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('c_t (per-dim):', x0 + W * 0.04, tapeY - 2);
    for (let k = 0; k < cellsN; k++) {
      const v = cur.c[k];
      const a = Math.min(1, Math.abs(v));
      ctx.fillStyle = v >= 0 ? `rgba(122,31,36,${0.15 + a * 0.7})` : `rgba(38,35,32,${0.15 + a * 0.7})`;
      ctx.fillRect(tapeLeft + k * cw + 2, tapeY - 12, cw - 4, 14);
    }
    ctx.strokeStyle = 'rgba(38,35,32,0.20)';
    ctx.lineWidth = 1;
    ctx.strokeRect(tapeLeft, tapeY - 12, tapeW, 14);
  }

  function drawValve(ctx, x, y, openness, glyph, label) {
    // Sigmoid valve drawn as a circle with a horizontal bar fill = openness.
    const r = 16;
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.5;
    ctx.fillStyle = '#fffdf6';
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    // Fill proportional to openness
    const fillH = r * 2 * openness;
    ctx.save();
    ctx.beginPath(); ctx.arc(x, y, r - 1, 0, Math.PI * 2); ctx.clip();
    ctx.fillStyle = `rgba(122,31,36,${0.25 + openness * 0.5})`;
    ctx.fillRect(x - r, y + r - fillH, 2 * r, fillH);
    ctx.restore();
    // glyph (σ)
    ctx.fillStyle = INK;
    ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(glyph, x, y);
    ctx.textBaseline = 'alphabetic';
    // multiplicative ⊗ symbol on the bus immediately right of the gate
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(x + 22, y, 9, 0, Math.PI * 2);
    ctx.moveTo(x + 16, y - 6); ctx.lineTo(x + 28, y + 6);
    ctx.moveTo(x + 28, y - 6); ctx.lineTo(x + 16, y + 6);
    ctx.stroke();
    // Label below
    ctx.fillStyle = INK;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText(label.toUpperCase(), x, y - r - 6);
  }

  function drawNode(ctx, x, y, glyph, label) {
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.2;
    ctx.fillStyle = '#fffdf6';
    ctx.beginPath();
    ctx.rect(x - 22, y - 12, 44, 24);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = INK;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(glyph, x, y);
    ctx.textBaseline = 'alphabetic';
    if (label) {
      ctx.fillStyle = INK_FADE;
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.fillText(label.toUpperCase(), x, y + 22);
    }
  }

  function drawArrow(ctx, x1, y1, x2, y2, color, width = 1.5, head = 7) {
    const ang = Math.atan2(y2 - y1, x2 - x1);
    ctx.strokeStyle = color; ctx.fillStyle = color;
    ctx.lineWidth = width; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - head * Math.cos(ang - Math.PI / 7), y2 - head * Math.sin(ang - Math.PI / 7));
    ctx.lineTo(x2 - head * Math.cos(ang + Math.PI / 7), y2 - head * Math.sin(ang + Math.PI / 7));
    ctx.closePath(); ctx.fill();
  }

  // ───────── View 2: Tape (the original heatmap) ─────────
  function drawTape(ctx, size) {
    const trc = trace();
    const N = state.seq.length;
    if (N === 0) return;
    const H = trc[0].h.length;
    const labels = ['f', 'i', 'o', 'c̃', 'c', 'h'];
    const channels = ['f', 'i', 'o', 'cTilde', 'c', 'h'];
    const headerH = 36;
    const totalRows = labels.length * H + (labels.length - 1);
    const cellH = (size.h - headerH - 20) / totalRows;
    const colW = size.w / (N + 1);

    ctx.fillStyle = INK_FADE;
    ctx.font = '12px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    for (let t = 0; t < N; t++) {
      ctx.fillStyle = t < state.step ? '#262320' : 'rgba(38,35,32,0.30)';
      ctx.font = 'bold 13px "JetBrains Mono", monospace';
      ctx.fillText(state.seq[t] === ' ' ? '·' : state.seq[t], colW * (t + 1) + colW * 0.5, 18);
    }

    let y = headerH;
    labels.forEach((lbl, ci) => {
      ctx.fillStyle = INK_FADE;
      ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'right';
      ctx.fillText(lbl, colW - 6, y + 12);

      for (let i = 0; i < H; i++) {
        for (let t = 0; t <= state.step; t++) {
          const v = trc[t][channels[ci]][i];
          const intensity = Math.min(1, Math.abs(v));
          let color;
          if (channels[ci] === 'f' || channels[ci] === 'i' || channels[ci] === 'o') {
            const g = Math.round(247 - intensity * 200);
            const grn = Math.round(244 - intensity * 200);
            const b = Math.round(236 - intensity * 220);
            color = `rgb(${g},${grn},${b})`;
          } else {
            color = v >= 0
              ? `rgba(122,31,36,${0.10 + intensity * 0.7})`
              : `rgba(38,35,32,${0.10 + intensity * 0.7})`;
          }
          ctx.fillStyle = color;
          ctx.fillRect(colW * (t + 0.5), y + i * cellH, colW, cellH + 0.5);
        }
      }
      ctx.strokeStyle = 'rgba(38,35,32,0.30)';
      ctx.lineWidth = 1;
      ctx.strokeRect(colW * 0.5, y, colW * (N + 0.5), H * cellH);
      y += H * cellH + 1;
    });
  }

  // ───────── View 3: LSTM vs vanilla RNN ─────────
  function drawCompare(ctx, size) {
    const lstm = trace();
    const rnn = traceRNN();
    const N = state.seq.length;
    if (N === 0) return;
    const H = lstm[0].h.length;

    const headerH = 36;
    const colW = size.w / (N + 1);
    const sectionH = (size.h - headerH - 36) / 2;
    const cellH = (sectionH - 18) / H;

    // header
    ctx.fillStyle = INK_FADE;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    for (let t = 0; t < N; t++) {
      ctx.fillStyle = t < state.step ? '#262320' : 'rgba(38,35,32,0.30)';
      ctx.font = 'bold 13px "JetBrains Mono", monospace';
      ctx.fillText(state.seq[t] === ' ' ? '·' : state.seq[t], colW * (t + 1) + colW * 0.5, 18);
    }

    // LSTM cell-state row
    let y = headerH;
    ctx.fillStyle = INK;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('LSTM cell-state c_t — bounded, gently changing', colW * 0.5, y - 4);
    for (let i = 0; i < H; i++) {
      for (let t = 0; t <= state.step; t++) {
        const v = lstm[t].c[i];
        const intensity = Math.min(1, Math.abs(v) / 2);
        ctx.fillStyle = v >= 0
          ? `rgba(122,31,36,${0.10 + intensity * 0.7})`
          : `rgba(38,35,32,${0.10 + intensity * 0.7})`;
        ctx.fillRect(colW * (t + 0.5), y + i * cellH, colW, cellH + 0.5);
      }
    }
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.strokeRect(colW * 0.5, y, colW * (N + 0.5), H * cellH);
    // mean magnitude readout
    const lstmMag = avgMag(lstm.slice(1, state.step + 1).map((s) => s.c));
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`mean |c| = ${lstmMag.toFixed(2)}`, colW * (N + 0.5), y + H * cellH + 12);

    // Vanilla RNN row
    y = headerH + sectionH + 16;
    ctx.fillStyle = INK;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('Vanilla RNN h_t — no gating, drifts toward saturation', colW * 0.5, y - 4);
    for (let i = 0; i < H; i++) {
      for (let t = 0; t <= state.step; t++) {
        const v = rnn[t][i];
        const intensity = Math.min(1, Math.abs(v));
        ctx.fillStyle = v >= 0
          ? `rgba(122,31,36,${0.10 + intensity * 0.7})`
          : `rgba(38,35,32,${0.10 + intensity * 0.7})`;
        ctx.fillRect(colW * (t + 0.5), y + i * cellH, colW, cellH + 0.5);
      }
    }
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.strokeRect(colW * 0.5, y, colW * (N + 0.5), H * cellH);
    const rnnMag = avgMag(rnn.slice(1, state.step + 1));
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`mean |h| = ${rnnMag.toFixed(2)}`, colW * (N + 0.5), y + H * cellH + 12);
  }
  function avgMag(arr) {
    if (!arr.length) return 0;
    let s = 0, n = 0;
    arr.forEach((v) => v.forEach((x) => { s += Math.abs(x); n++; }));
    return n === 0 ? 0 : s / n;
  }

  function render() {
    seqInput.value = state.seq;
    stepSlider.max = state.seq.length;
    stepSlider.value = state.step;
    stepLabel.textContent = `t = ${state.step}`;
    Array.from(gateTogglesEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.force === state.force);
    });
    Array.from(compareTogglesEl.children).forEach((btn) => {
      const isOn = (btn.dataset.cmp === 'on') === state.showRNN;
      btn.classList.toggle('active', isOn);
    });
    Array.from(tabsEl.children).forEach((btn) => {
      btn.classList.toggle('on', btn.dataset.view === state.view);
    });
    if (captionEl) captionEl.textContent = CAPTIONS[state.view];
    canvasCtl.redraw();
  }

  render();
})();
