/* Vol. XVII — Recurrent Networks */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, INK_FADE } = V;

  const VOCAB = ['a', 'b', 'r', 'c', 'd', ' '];
  const oneHot = (c) => VOCAB.map((v) => (v === c ? 1 : 0));

  function seeded(seed) {
    let s = seed;
    return () => { s = (s * 9301 + 49297) % 233280; return (s / 233280 - 0.5) * 1.4; };
  }
  function makeRNN(H = 8, V = 6, seed = 42) {
    const r = seeded(seed);
    const Wxh = Array.from({ length: H }, () => Array.from({ length: V }, r));
    const Whh = Array.from({ length: H }, () => Array.from({ length: H }, r));
    return { Wxh, Whh, bh: new Array(H).fill(0), H, V };
  }
  function step(net, x, hPrev) {
    const z = new Array(net.H);
    for (let i = 0; i < net.H; i++) {
      let s = net.bh[i];
      for (let j = 0; j < net.V; j++) s += net.Wxh[i][j] * x[j];
      for (let j = 0; j < net.H; j++) s += net.Whh[i][j] * hPrev[j];
      z[i] = Math.tanh(s);
    }
    return z;
  }

  const state = {
    seq: 'abracadabra',
    seed: 7,
    step: 11,
    playing: false,
    timer: null,
  };

  const seqInput = document.getElementById('seq-input');
  const stepSlider = document.getElementById('step-slider');
  const stepLabel = document.getElementById('step-label');
  const replayBtn = document.getElementById('replay-btn');
  const seedBtn = document.getElementById('seed-btn');
  const readoutEl = document.getElementById('readout');
  const canvas = document.getElementById('canvas');

  seqInput.addEventListener('input', (e) => {
    state.seq = e.target.value.replace(/[^abrcd ]/g, '').slice(0, 14);
    state.step = state.seq.length;
    seqInput.value = state.seq;
    stepSlider.max = state.seq.length;
    render();
  });
  stepSlider.addEventListener('input', (e) => {
    stopPlay();
    state.step = +e.target.value;
    render();
  });
  replayBtn.addEventListener('click', () => {
    state.step = 0;
    state.playing = true;
    if (state.timer) clearInterval(state.timer);
    state.timer = setInterval(() => {
      if (state.step >= state.seq.length) { stopPlay(); return; }
      state.step++;
      render();
    }, 600);
    render();
  });
  seedBtn.addEventListener('click', () => {
    state.seed = Math.floor(Math.random() * 100) + 1;
    render();
  });
  function stopPlay() {
    state.playing = false;
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  function trace() {
    const net = makeRNN(8, 6, state.seed);
    let h = new Array(net.H).fill(0);
    const states = [h.slice()];
    for (let t = 0; t < state.seq.length; t++) {
      const x = oneHot(state.seq[t]);
      h = step(net, x, h);
      states.push(h.slice());
    }
    return { net, states };
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    const { states } = trace();
    const N = state.seq.length;
    if (N === 0) return;
    const H = states[0].length;
    const colW = size.w / (N + 1);
    const headerH = 50;
    const cellH = (size.h - headerH - 20) / H;

    // Header: input characters
    ctx.fillStyle = INK_FADE;
    ctx.font = '12px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('t', colW * 0.5, 18);
    for (let t = 0; t < N; t++) {
      const x = colW * (t + 1) + colW * 0.5;
      ctx.fillStyle = t < state.step ? '#262320' : 'rgba(38,35,32,0.30)';
      ctx.font = 'bold 14px "JetBrains Mono", monospace';
      ctx.fillText(state.seq[t] === ' ' ? '·' : state.seq[t], x, 18);
      ctx.fillStyle = 'rgba(38,35,32,0.45)';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillText(`${t + 1}`, x, 36);
    }

    // Hidden state heatmap. Show columns up to state.step (pre-state at column 0).
    for (let i = 0; i < H; i++) {
      // Row label
      ctx.fillStyle = INK_FADE;
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`h${i}`, colW - 6, headerH + i * cellH + cellH / 2 + 3);

      for (let t = 0; t <= state.step; t++) {
        const v = states[t][i];
        const intensity = Math.min(1, Math.abs(v));
        ctx.fillStyle = v >= 0
          ? `rgba(122,31,36,${0.10 + intensity * 0.7})`
          : `rgba(38,35,32,${0.10 + intensity * 0.7})`;
        ctx.fillRect(colW * (t + 0.5), headerH + i * cellH, colW, cellH + 0.5);
      }
    }
    // Frame
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.strokeRect(colW * 0.5, headerH, colW * (N + 0.5), H * cellH);
  }

  function render() {
    seqInput.value = state.seq;
    stepSlider.max = state.seq.length;
    stepSlider.value = state.step;
    stepLabel.textContent = `t = ${state.step} / ${state.seq.length}`;
    const { states } = trace();
    const h = states[state.step];
    let html = '';
    h.forEach((v, i) => {
      html += `<div class="row"><span>h${i}</span><b style="color:${v >= 0 ? 'var(--accent)' : 'var(--ink)'}">${formatNum(v)}</b></div>`;
    });
    readoutEl.innerHTML = html;
    canvasCtl.redraw();
  }

  render();
})();
