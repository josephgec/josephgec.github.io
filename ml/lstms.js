/* Vol. XVIII — Gates & Long Memory (LSTMs) */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, INK_FADE } = V;

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
    return { h: hNew, c: cNew, f, i, o };
  }

  const state = {
    seq: 'abracadabra',
    seed: 7,
    step: 11,
    force: 'none',
    timer: null,
  };

  const seqInput = document.getElementById('seq-input');
  const stepSlider = document.getElementById('step-slider');
  const stepLabel = document.getElementById('step-label');
  const replayBtn = document.getElementById('replay-btn');
  const seedBtn = document.getElementById('seed-btn');
  const gateTogglesEl = document.getElementById('gate-toggles');
  const canvas = document.getElementById('canvas');

  ['none', 'f-closed', 'f-open', 'i-closed', 'i-open', 'o-closed', 'o-open'].forEach((k) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = k === 'none' ? 'normal' : k.replace('-', ' ');
    b.dataset.force = k;
    b.addEventListener('click', () => { state.force = k; render(); });
    gateTogglesEl.appendChild(b);
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
    const out = [{ h, c, f: new Array(H).fill(0), i: new Array(H).fill(0), o: new Array(H).fill(0) }];
    for (let t = 0; t < state.seq.length; t++) {
      const x = oneHot(state.seq[t]);
      const r = lstmStep(net, x, h, c, state.force === 'none' ? null : state.force);
      h = r.h; c = r.c;
      out.push({ h, c, f: r.f, i: r.i, o: r.o });
    }
    return out;
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    const trc = trace();
    const N = state.seq.length;
    if (N === 0) return;
    const H = trc[0].h.length;
    const labels = ['f', 'i', 'o', 'c', 'h'];
    const channels = ['f', 'i', 'o', 'c', 'h'];
    const headerH = 36;
    const rowsPerCh = H;
    const totalRows = labels.length * rowsPerCh + (labels.length - 1);
    const cellH = (size.h - headerH - 20) / totalRows;
    const colW = size.w / (N + 1);

    // Header chars
    ctx.fillStyle = INK_FADE;
    ctx.font = '12px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    for (let t = 0; t < N; t++) {
      ctx.fillStyle = t < state.step ? '#262320' : 'rgba(38,35,32,0.30)';
      ctx.font = 'bold 13px "JetBrains Mono", monospace';
      ctx.fillText(state.seq[t] === ' ' ? '·' : state.seq[t], colW * (t + 1) + colW * 0.5, 18);
    }

    // Rows by channel
    let y = headerH;
    labels.forEach((lbl, ci) => {
      ctx.fillStyle = INK_FADE;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'right';
      ctx.fillText(lbl, colW - 6, y + 12);

      for (let i = 0; i < H; i++) {
        for (let t = 0; t <= state.step; t++) {
          const v = trc[t][channels[ci]][i];
          const intensity = Math.min(1, Math.abs(v));
          let color;
          if (channels[ci] === 'f' || channels[ci] === 'i' || channels[ci] === 'o') {
            // Gates are 0..1. Use intensity directly as t.
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
      // Frame the section
      ctx.strokeStyle = 'rgba(38,35,32,0.30)';
      ctx.lineWidth = 1;
      ctx.strokeRect(colW * 0.5, y, colW * (N + 0.5), H * cellH);
      y += H * cellH + 1;
    });
  }

  function render() {
    seqInput.value = state.seq;
    stepSlider.max = state.seq.length;
    stepSlider.value = state.step;
    stepLabel.textContent = `t = ${state.step}`;
    Array.from(gateTogglesEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.force === state.force);
    });
    canvasCtl.redraw();
  }

  render();
})();
