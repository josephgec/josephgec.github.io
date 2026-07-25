/* Vol. XVII — Recurrent Networks */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK, INK_FADE } = V;

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

  const VIEWS = [
    { key: 'unrolled', label: 'Unrolled across time' },
    { key: 'schematic', label: 'Rolled vs unrolled' },
    { key: 'heatmap', label: 'Hidden state · heatmap' },
    { key: 'bptt',    label: 'Vanishing gradient' },
  ];
  const CAPTIONS = {
    unrolled: 'Same recurrent cell, drawn out across time. The hidden state h flows left-to-right; each step takes a new input x_t and the previous h_{t−1}, and the same weight matrices W_xh and W_hh are reused at every timestep — the architectural definition of "recurrent."',
    schematic: 'Left: the network in its rolled form — one cell with a self-loop labeled h. Right: that same loop unrolled. The arrows W_hh in the unrolled view all share the same weights with the loop on the left. The unrolled picture is what backprop "sees."',
    heatmap: 'Heat map of every hidden-state dimension over time. Orange = positive activation, ink = negative. Different rows specialize for different patterns in the input — recurring symbols, position, vowels.',
    bptt: 'Backprop through time runs the chain rule backwards through the unrolled network. Each backward hop multiplies by the local Jacobian, which in a tanh RNN has spectral radius ≤ 1. The gradient signal fades — the network can\'t learn long-range dependencies. LSTMs are the fix.',
  };

  const state = {
    seq: 'abracadabra',
    seed: 7,
    step: 11,
    view: 'unrolled',
    playing: false,
    timer: null,
  };

  const seqInput = document.getElementById('seq-input');
  const stepSlider = document.getElementById('step-slider');
  const stepLabel = document.getElementById('step-label');
  const replayBtn = document.getElementById('replay-btn');
  const seedBtn = document.getElementById('seed-btn');
  const readoutEl = document.getElementById('readout');
  const tabsEl = document.getElementById('view-tabs');
  const captionEl = document.getElementById('caption-text');
  const canvas = document.getElementById('canvas');

  VIEWS.forEach((v) => {
    const b = document.createElement('button');
    b.textContent = v.label;
    b.dataset.view = v.key;
    b.addEventListener('click', () => { state.view = v.key; render(); });
    tabsEl.appendChild(b);
  });

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
    if (state.view === 'unrolled') drawUnrolled(ctx, size);
    else if (state.view === 'schematic') drawSchematic(ctx, size);
    else if (state.view === 'bptt') drawBPTT(ctx, size);
    else drawHeatmap(ctx, size);
  }

  // ───────── View 1: Unrolled across time ─────────
  function drawUnrolled(ctx, size) {
    const { states } = trace();
    const N = state.seq.length;
    if (N === 0) return;
    const Hd = states[0].length;

    const margin = 24;
    const yIn = size.h - margin - 24;
    const yCell = size.h * 0.55;
    const yH = size.h * 0.32;
    const yOut = size.h * 0.10;
    const W = size.w - margin * 2;
    const colW = W / (N + 1);

    // Title
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('UNROLLED RNN · one cell, reused each step', margin, margin);
    // Storytelling: anchor the eye to the recurring rhythm.
    // Only drawn when the canvas is wide enough to clear the left-hand title.
    if (size.w > 520) {
      ctx.fillStyle = ACCENT;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'right';
      ctx.fillText('repeating chars → a recurring rhythm', size.w - margin, margin);
    }

    // h_0 starting box on the far left
    drawHbar(ctx, margin + colW * 0.35, yH, 36, 18, new Array(Hd).fill(0));
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('h_0', margin + colW * 0.35 + 18, yH - 4);

    for (let t = 0; t < N; t++) {
      const cx = margin + colW * (t + 1.0) + colW * 0.5;
      const isPast = t < state.step;
      const cellAlpha = isPast ? 1 : 0.35;

      // input x_t below
      ctx.fillStyle = isPast ? INK : 'rgba(38,35,32,0.30)';
      ctx.font = 'italic 22px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      const ch = state.seq[t] === ' ' ? '·' : state.seq[t];
      ctx.fillText(ch, cx, yIn + 6);
      ctx.fillStyle = INK_FADE;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillText(`x_${t + 1}`, cx, yIn + 22);

      // arrow up from input to cell
      drawArrow(ctx, cx, yIn - 16, cx, yCell + 22, isPast ? INK_FADE : 'rgba(38,35,32,0.20)', 1.2, 6);

      // cell box
      ctx.globalAlpha = cellAlpha;
      ctx.strokeStyle = INK;
      ctx.fillStyle = '#fffdf6';
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.rect(cx - 24, yCell - 16, 48, 36);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = INK;
      ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('tanh', cx, yCell + 2);
      ctx.textBaseline = 'alphabetic';
      ctx.globalAlpha = 1;

      // hidden bar above cell
      const hBarX = cx - 18, hBarY = yH - 6;
      drawHbar(ctx, hBarX, hBarY, 36, 18, isPast ? states[t + 1] : null);
      ctx.fillStyle = isPast ? INK : 'rgba(38,35,32,0.30)';
      ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText(`h_${t + 1}`, cx, yH - 16);

      // arrow from cell up to hidden bar
      drawArrow(ctx, cx, yCell - 16, cx, hBarY + 18, isPast ? INK : 'rgba(38,35,32,0.30)', 1.2, 6);

      // arrow from hidden bar up to output y_t
      ctx.fillStyle = isPast ? ACCENT : 'rgba(122,31,36,0.30)';
      ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText(`y_${t + 1}`, cx, yOut);
      drawArrow(ctx, cx, hBarY, cx, yOut + 6, isPast ? ACCENT : 'rgba(122,31,36,0.30)', 1.2, 6);

      // recurrent arrow from previous h to this cell, labeled W_hh
      const prevX = t === 0 ? margin + colW * 0.35 + 18 : margin + colW * (t) + colW * 0.5 - 18;
      const fromX = prevX + 18;
      const toX = cx - 24;
      const midY = yH;
      ctx.strokeStyle = isPast ? ACCENT : 'rgba(122,31,36,0.25)';
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(fromX, midY);
      ctx.bezierCurveTo(fromX + 14, midY, cx - 30, yCell - 6, toX, yCell - 6);
      ctx.stroke();
      // label W_hh
      if (t === Math.min(2, N - 1)) {
        ctx.fillStyle = isPast ? ACCENT : 'rgba(122,31,36,0.30)';
        ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
        ctx.textAlign = 'center';
        ctx.fillText('W_hh (shared)', (fromX + toX) / 2, yH + 8);
      }

      // W_xh label on the input arrow (only on first step to avoid clutter)
      if (t === 0) {
        ctx.fillStyle = INK_FADE;
        ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
        ctx.textAlign = 'left';
        ctx.fillText('W_xh', cx + 6, (yIn + yCell) / 2);
      }
    }

    // Footer hint
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('h_t = tanh(W_xh · x_t + W_hh · h_{t−1} + b)', size.w / 2, size.h - 4);
  }

  // Tiny bar chart for a hidden-state vector.
  function drawHbar(ctx, x, y, w, h, vec) {
    ctx.strokeStyle = 'rgba(38,35,32,0.35)';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#fffdf6';
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
    if (!vec) return;
    const cw = w / vec.length;
    vec.forEach((v, i) => {
      const a = Math.min(1, Math.abs(v));
      ctx.fillStyle = v >= 0
        ? `rgba(122,31,36,${0.20 + a * 0.7})`
        : `rgba(38,35,32,${0.20 + a * 0.7})`;
      const bh = h - 2;
      const ch = bh * Math.min(1, Math.abs(v));
      ctx.fillRect(x + i * cw + 0.5, y + (bh - ch) / 2 + 1, cw - 1, ch);
    });
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

  // ───────── View 2: Rolled vs unrolled side-by-side schematic ─────────
  function drawSchematic(ctx, size) {
    const margin = 32;
    const halfW = (size.w - margin * 3) / 2;

    // ── Left: rolled ──
    const lx = margin, ly = size.h * 0.5;
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('ROLLED', lx + halfW / 2, margin + 4);
    // Cell square
    const cellSize = 70;
    ctx.fillStyle = '#fffdf6';
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.5;
    ctx.fillRect(lx + halfW / 2 - cellSize / 2, ly - cellSize / 2, cellSize, cellSize);
    ctx.strokeRect(lx + halfW / 2 - cellSize / 2, ly - cellSize / 2, cellSize, cellSize);
    ctx.fillStyle = INK;
    ctx.font = 'italic 18px "Source Serif 4", Georgia, serif';
    ctx.textBaseline = 'middle';
    ctx.fillText('RNN', lx + halfW / 2, ly);
    ctx.textBaseline = 'alphabetic';
    // Self-loop arrow (recurrent)
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(lx + halfW / 2 + cellSize / 2 + 26, ly, 26, -Math.PI * 0.7, Math.PI * 0.7);
    ctx.stroke();
    drawArrow(ctx, lx + halfW / 2 + cellSize / 2 + 14, ly + 22, lx + halfW / 2 + cellSize / 2 - 2, ly + 14, ACCENT, 2, 8);
    ctx.fillStyle = ACCENT;
    ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('h', lx + halfW / 2 + cellSize / 2 + 50, ly + 5);
    // Input arrow
    drawArrow(ctx, lx + halfW / 2, ly + cellSize / 2 + 30, lx + halfW / 2, ly + cellSize / 2 + 4, INK_FADE, 1.5, 7);
    ctx.fillStyle = INK;
    ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('x', lx + halfW / 2, ly + cellSize / 2 + 46);
    // Output arrow
    drawArrow(ctx, lx + halfW / 2, ly - cellSize / 2 - 4, lx + halfW / 2, ly - cellSize / 2 - 30, INK, 1.5, 7);
    ctx.fillStyle = INK;
    ctx.fillText('y', lx + halfW / 2, ly - cellSize / 2 - 38);

    // ── Right: unrolled ──
    const rx = margin * 2 + halfW;
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('UNROLLED', rx + halfW / 2, margin + 4);
    const T = 4;
    const stepW = halfW / T;
    for (let t = 0; t < T; t++) {
      const cx = rx + stepW * (t + 0.5);
      const cy = ly;
      // Cell
      ctx.fillStyle = '#fffdf6';
      ctx.strokeStyle = INK;
      ctx.lineWidth = 1.2;
      ctx.fillRect(cx - 22, cy - 22, 44, 44);
      ctx.strokeRect(cx - 22, cy - 22, 44, 44);
      ctx.fillStyle = INK;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('RNN', cx, cy);
      ctx.textBaseline = 'alphabetic';
      // Input below
      drawArrow(ctx, cx, cy + 50, cx, cy + 24, INK_FADE, 1.2, 6);
      ctx.fillStyle = INK_FADE;
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.fillText(`x_${t + 1}`, cx, cy + 64);
      // Output above
      drawArrow(ctx, cx, cy - 24, cx, cy - 48, INK_FADE, 1.2, 6);
      ctx.fillStyle = INK_FADE;
      ctx.fillText(`y_${t + 1}`, cx, cy - 56);
      // Recurrent arrow to next
      if (t < T - 1) {
        const nextX = rx + stepW * (t + 1.5);
        drawArrow(ctx, cx + 22, cy, nextX - 22, cy, ACCENT, 2, 8);
        if (t === 0) {
          ctx.fillStyle = ACCENT;
          ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
          ctx.textAlign = 'center';
          ctx.fillText('W_hh', (cx + 22 + nextX - 22) / 2, cy - 6);
        }
      }
    }
    // h_0 entering on the left
    drawArrow(ctx, rx + 4, ly, rx + stepW * 0.5 - 22, ly, ACCENT, 1.5, 7);
    ctx.fillStyle = ACCENT;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('h_0', rx + 4, ly - 6);
    // h_T leaving on the right
    drawArrow(ctx, rx + stepW * (T - 0.5) + 22, ly, rx + halfW - 4, ly, ACCENT, 1.5, 7);
    ctx.fillStyle = ACCENT;
    ctx.textAlign = 'right';
    ctx.fillText(`h_T`, rx + halfW - 4, ly - 6);

    // Caption underneath
    ctx.fillStyle = INK;
    ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('Reusing the same cell each step — that\'s why it can read sequences of any length.', size.w / 2, size.h - 18);
  }

  // ───────── View 3: Heatmap (legacy) ─────────
  function drawHeatmap(ctx, size) {
    const { states } = trace();
    const N = state.seq.length;
    if (N === 0) return;
    const H = states[0].length;
    const colW = size.w / (N + 1);
    const headerH = 50;
    const cellH = (size.h - headerH - 20) / H;

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

    for (let i = 0; i < H; i++) {
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
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.strokeRect(colW * 0.5, headerH, colW * (N + 0.5), H * cellH);

    // Legend strip below the heatmap: −1 (ink) ··· 0 ··· +1 (oxblood)
    const legY = headerH + H * cellH + 18;
    const legW = Math.min(220, size.w * 0.4);
    const legX = colW * 0.5;
    const legSteps = 24;
    for (let s = 0; s < legSteps; s++) {
      const t = s / (legSteps - 1);            // 0..1
      const v = t * 2 - 1;                      // −1..+1
      const a = Math.min(1, Math.abs(v));
      ctx.fillStyle = v >= 0
        ? `rgba(122,31,36,${0.10 + a * 0.7})`
        : `rgba(38,35,32,${0.10 + a * 0.7})`;
      ctx.fillRect(legX + s * (legW / legSteps), legY, legW / legSteps + 0.5, 10);
    }
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.strokeRect(legX, legY, legW, 10);
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('−1 negative', legX, legY + 22);
    ctx.textAlign = 'center';
    ctx.fillText('0', legX + legW / 2, legY + 22);
    ctx.textAlign = 'right';
    ctx.fillText('+1 positive', legX + legW, legY + 22);
  }

  // ───────── View 4: Vanishing gradient through time ─────────
  function drawBPTT(ctx, size) {
    const N = state.seq.length;
    if (N === 0) return;
    const margin = 32;
    const W = size.w - margin * 2;
    const stepW = W / Math.max(N, 6);
    const yCell = size.h * 0.45;
    const yArrow = size.h * 0.20;

    // Cells along the bottom
    for (let t = 0; t < N; t++) {
      const cx = margin + stepW * (t + 0.5);
      ctx.fillStyle = '#fffdf6';
      ctx.strokeStyle = INK;
      ctx.lineWidth = 1.2;
      ctx.fillRect(cx - 18, yCell - 14, 36, 28);
      ctx.strokeRect(cx - 18, yCell - 14, 36, 28);
      ctx.fillStyle = INK;
      ctx.font = 'italic 10px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('tanh', cx, yCell);
      ctx.textBaseline = 'alphabetic';
      ctx.fillStyle = INK_FADE;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillText(`t = ${t + 1}`, cx, yCell + 26);
      // forward arrows
      if (t < N - 1) {
        drawArrow(ctx, cx + 18, yCell, margin + stepW * (t + 1.5) - 18, yCell, INK_FADE, 1.2, 6);
      }
    }

    // The loss is at the right end — we backprop from there.
    const lossX = margin + stepW * (N - 0.5) + 30;
    ctx.fillStyle = ACCENT;
    ctx.font = 'italic 14px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('L', lossX, yCell + 4);
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillText('loss', lossX, yCell + 18);

    // Backward gradient arrows fading exponentially with distance.
    ctx.fillStyle = INK_FADE;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('∂L/∂h_t  ←  backprop through time', margin, yArrow - 14);

    const decay = 0.55;
    let mag = 1;
    for (let t = N - 1; t >= 0; t--) {
      const cx = margin + stepW * (t + 0.5);
      const cxNext = margin + stepW * (t + 1.5);
      const xFrom = t === N - 1 ? lossX - 4 : cxNext;
      const yFromArrow = yArrow + (1 - mag) * 60;
      const yToArrow = yArrow + (1 - mag * decay) * 60;
      // Fading curved arrow from right cell to this cell
      const alpha = Math.max(0.05, mag);
      ctx.strokeStyle = `rgba(122,31,36,${alpha})`;
      ctx.lineWidth = Math.max(1, mag * 4);
      ctx.beginPath();
      ctx.moveTo(xFrom, yFromArrow);
      ctx.bezierCurveTo(xFrom - stepW * 0.3, yFromArrow - 8, cx + stepW * 0.3, yToArrow - 8, cx, yToArrow);
      ctx.stroke();
      // arrowhead
      drawArrow(ctx, cx + 10, yToArrow + 3, cx, yToArrow, `rgba(122,31,36,${alpha})`, Math.max(1, mag * 2), 6);
      // magnitude label every 2 steps
      if (t % 2 === 0) {
        ctx.fillStyle = `rgba(38,35,32,${alpha + 0.2})`;
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`|grad|≈${mag.toFixed(2)}`, cx, yToArrow - 8);
      }
      mag *= decay;
    }

    // Arrows from each cell up to its gradient track
    ctx.strokeStyle = 'rgba(38,35,32,0.18)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    for (let t = 0; t < N; t++) {
      const cx = margin + stepW * (t + 0.5);
      ctx.beginPath();
      ctx.moveTo(cx, yArrow + 60);
      ctx.lineTo(cx, yCell - 16);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Caption box at the bottom
    ctx.fillStyle = INK;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('Each backward hop multiplies by W_hh · diag(tanh\'). Both factors are ≤ 1 in magnitude, so the gradient shrinks geometrically — a vanilla RNN can\'t learn beyond ~10 steps.', size.w / 2, size.h - 14);
  }

  function render() {
    seqInput.value = state.seq;
    stepSlider.max = state.seq.length;
    stepSlider.value = state.step;
    stepLabel.textContent = `t = ${state.step} / ${state.seq.length}`;
    Array.from(tabsEl.children).forEach((btn) => {
      btn.classList.toggle('on', btn.dataset.view === state.view);
    });
    if (captionEl) captionEl.textContent = CAPTIONS[state.view];

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
