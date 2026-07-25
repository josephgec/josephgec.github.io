/* Vol. XXX — Transfer Learning */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK, INK_FADE } = V;

  // Simulate three training curves: scratch, frozen, fine-tune.
  // Curve shapes are functions of (frozenLayers, dataSize, epoch).
  const NUM_LAYERS = 4;
  // Each layer's parameter count (millions). Backbone is heavy at the bottom, head is small.
  const LAYER_PARAMS_M = [4.2, 4.0, 3.0, 1.4]; // L1..L4
  const HEAD_PARAMS_M = 0.2;
  const TOTAL_PARAMS_M = LAYER_PARAMS_M.reduce((a, b) => a + b, 0) + HEAD_PARAMS_M;

  // Per-epoch wall-clock cost (seconds) — full fine-tune costs the most because every layer is updated.
  const SECONDS_PER_EPOCH = {
    scratch:  1.20, // training every layer from random init takes longest
    frozen:   0.25, // only the head updates, gradient skips backbone
    finetune: 0.85, // most layers update, slightly faster than scratch (warm start)
  };

  function simulate(strategy, frozen, dataSize, maxEpoch) {
    const curve = [];
    // Asymptotes depend on data size (more data → higher achievable accuracy).
    const dataFactor = Math.log(dataSize + 10) / Math.log(510);
    let asymptote, rate, startAcc;
    if (strategy === 'scratch') {
      asymptote = 0.55 + dataFactor * 0.30;
      rate = 0.04 + dataFactor * 0.08;
      startAcc = 0.5; // chance level for binary
    } else if (strategy === 'frozen') {
      asymptote = 0.65 + dataFactor * 0.18;
      rate = 0.30; // fast convergence — only the last layer trains
      startAcc = 0.55;
      // Asymptote also shaped by how many layers are frozen (more frozen = less flexible)
      asymptote -= (frozen / NUM_LAYERS) * 0.08;
    } else {
      // fine-tune
      asymptote = 0.78 + dataFactor * 0.18;
      rate = 0.18;
      startAcc = 0.55;
      asymptote -= (frozen / NUM_LAYERS) * 0.05;
      asymptote = Math.min(0.97, asymptote);
    }
    for (let e = 0; e <= maxEpoch; e++) {
      const v = asymptote - (asymptote - startAcc) * Math.exp(-rate * e);
      // Add small noise
      const noise = (Math.sin(e * (strategy === 'scratch' ? 1.7 : 0.7)) +
                     Math.cos(e * 0.31)) * 0.012 * (1 - dataFactor * 0.4);
      curve.push(Math.max(0, Math.min(1, v + noise)));
    }
    return curve;
  }

  const state = {
    strategy: 'all',
    frozen: 3,
    dataSize: 100,
    maxEpoch: 30,
    // Default to mid-training (epoch 14/30) so first paint shows the three strategies
    // already separated: frozen has converged fast and flatlined, fine-tune is climbing
    // past it, scratch is still well below. Instructive without needing to press Run.
    epoch: 14,
    training: false,
    timer: null,
    curves: { scratch: [], frozen: [], finetune: [] },
  };

  function recompute() {
    state.curves.scratch = simulate('scratch', state.frozen, state.dataSize, state.maxEpoch);
    state.curves.frozen = simulate('frozen', state.frozen, state.dataSize, state.maxEpoch);
    state.curves.finetune = simulate('finetune', state.frozen, state.dataSize, state.maxEpoch);
  }
  recompute();

  // Trainable-parameter accounting based on freeze depth.
  // For "scratch": every layer + head trains. For "frozen": only the head. For "fine-tune": layers above the freeze line + head.
  function trainableParams(strategy, frozen) {
    if (strategy === 'scratch') return TOTAL_PARAMS_M;
    if (strategy === 'frozen')  return HEAD_PARAMS_M;
    // fine-tune: layers >= frozen are trainable, plus head
    let s = HEAD_PARAMS_M;
    for (let i = frozen; i < NUM_LAYERS; i++) s += LAYER_PARAMS_M[i];
    return s;
  }

  const strategiesEl = document.getElementById('strategies');
  const frozenSlider = document.getElementById('frozen-slider');
  const frozenLabel = document.getElementById('frozen-label');
  const dataSlider = document.getElementById('data-slider');
  const dataLabel = document.getElementById('data-label');
  const trainBtn = document.getElementById('train-btn');
  const resetBtn = document.getElementById('reset-btn');
  const readoutEl = document.getElementById('readout');
  const canvas = document.getElementById('canvas');

  ['All three', 'Scratch only', 'Frozen only', 'Fine-tune only'].forEach((lbl, i) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = lbl;
    b.dataset.strategy = ['all', 'scratch', 'frozen', 'finetune'][i];
    b.addEventListener('click', () => { state.strategy = b.dataset.strategy; render(); });
    strategiesEl.appendChild(b);
  });
  frozenSlider.addEventListener('input', (e) => {
    state.frozen = +e.target.value;
    recompute();
    render();
  });
  dataSlider.addEventListener('input', (e) => {
    state.dataSize = +e.target.value;
    recompute();
    render();
  });
  trainBtn.addEventListener('click', () => {
    state.epoch = 0;
    state.training = true;
    if (state.timer) clearInterval(state.timer);
    state.timer = setInterval(() => {
      state.epoch++;
      if (state.epoch >= state.maxEpoch) {
        state.training = false;
        clearInterval(state.timer);
        state.timer = null;
      }
      render();
    }, 100);
    trainBtn.textContent = '▷ Training…';
  });
  resetBtn.addEventListener('click', () => {
    state.epoch = 0;
    state.training = false;
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    trainBtn.textContent = '▷ Train all three';
    render();
  });

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  // Color per strategy — kept in one place for legend, curves, time counters.
  const STRAT_COLOR = {
    scratch:  '#262320',
    frozen:   '#3a6b5e',
    finetune: ACCENT,
  };

  function draw(ctx, size) {
    // ───── Top: layer architecture diagram with frozen vs trainable ─────
    const archH = 130;
    const padX = 36;
    const layerW = (size.w - padX * 2 - 80) / (NUM_LAYERS + 1);
    const layerY = 40;
    const layerHpx = 56;

    // Header banner: "PRE-TRAINED BACKBONE → NEW HEAD"
    ctx.fillStyle = INK_FADE;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    // Terser labels on narrow canvases so the backbone/head banners never collide.
    const wideHdr = size.w > 500;
    ctx.fillText(wideHdr ? 'PRE-TRAINED BACKBONE · ImageNet' : 'BACKBONE', padX, 22);
    ctx.textAlign = 'right';
    ctx.fillText(wideHdr ? 'NEW HEAD · target task' : 'HEAD', padX + NUM_LAYERS * layerW + layerW - 8, 22);

    // Brace under backbone
    ctx.strokeStyle = 'rgba(38,35,32,0.35)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    const braceY = layerY + layerHpx + 14;
    ctx.moveTo(padX, braceY);
    ctx.lineTo(padX + NUM_LAYERS * layerW - 8, braceY);
    ctx.stroke();

    for (let i = 0; i < NUM_LAYERS; i++) {
      const x = padX + i * layerW;
      const isFrozen = i < state.frozen;
      ctx.fillStyle = isFrozen ? 'rgba(38,35,32,0.18)' : 'rgba(122,31,36,0.45)';
      ctx.fillRect(x, layerY, layerW - 8, layerHpx);
      ctx.strokeStyle = 'rgba(38,35,32,0.40)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x, layerY, layerW - 8, layerHpx);
      ctx.fillStyle = isFrozen ? 'rgba(38,35,32,0.7)' : '#fffdf6';
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`L${i + 1}`, x + layerW / 2 - 4, layerY + 22);
      ctx.fillText(isFrozen ? '🔒 frozen' : 'trainable', x + layerW / 2 - 4, layerY + 40);
      // params readout below
      ctx.fillStyle = INK_FADE;
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.fillText(`${LAYER_PARAMS_M[i].toFixed(1)}M`, x + layerW / 2 - 4, layerY + layerHpx + 28);
    }
    // Head (always trainable)
    const headX = padX + NUM_LAYERS * layerW;
    ctx.fillStyle = 'rgba(122,31,36,0.85)';
    ctx.fillRect(headX, layerY, layerW - 8, layerHpx);
    ctx.fillStyle = '#fffdf6';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('head', headX + layerW / 2 - 4, layerY + 22);
    ctx.fillText('new', headX + layerW / 2 - 4, layerY + 40);
    ctx.fillStyle = INK_FADE;
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillText(`${HEAD_PARAMS_M.toFixed(1)}M`, headX + layerW / 2 - 4, layerY + layerHpx + 28);

    // Trainable parameter pill (right of head)
    const trainable = trainableParams(
      state.strategy === 'all' ? 'finetune' : state.strategy,
      state.frozen
    );
    ctx.fillStyle = INK;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText(
      `${trainable.toFixed(1)}M trainable / ${TOTAL_PARAMS_M.toFixed(1)}M total`,
      padX, archH + 6
    );

    // ───── Bottom: training curves ─────
    // Reserve space below the curves for: counter strip (≈46px) + storytelling line (≈20px) + bottom padding.
    // Without this reservation the wall-clock counters and headline tradeoff line render below the canvas.
    const plotY0 = archH + 50;
    const plotH = size.h - plotY0 - 130;
    const plotX0 = 60;
    const plotW = size.w - plotX0 - 24;

    // Frame
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plotX0, plotY0);
    ctx.lineTo(plotX0, plotY0 + plotH);
    ctx.lineTo(plotX0 + plotW, plotY0 + plotH);
    ctx.stroke();

    // Y labels
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    [0, 0.5, 1].forEach((y) => {
      const py = plotY0 + plotH - y * plotH;
      ctx.fillText((y * 100).toFixed(0) + '%', plotX0 - 6, py + 3);
      ctx.strokeStyle = 'rgba(38,35,32,0.08)';
      ctx.beginPath();
      ctx.moveTo(plotX0, py); ctx.lineTo(plotX0 + plotW, py);
      ctx.stroke();
    });
    // X label
    ctx.textAlign = 'right';
    ctx.fillText(`epoch ${state.epoch} / ${state.maxEpoch}`, plotX0 + plotW, plotY0 + plotH + 16);

    function drawCurve(curve, color, label) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      const n = Math.min(curve.length, state.training ? state.epoch + 1 : curve.length);
      for (let i = 0; i < n; i++) {
        const x = plotX0 + (i / state.maxEpoch) * plotW;
        const y = plotY0 + plotH - curve[i] * plotH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
      // End label
      const endIdx = n - 1;
      if (endIdx >= 0) {
        const x = plotX0 + (endIdx / state.maxEpoch) * plotW;
        const y = plotY0 + plotH - curve[endIdx] * plotH;
        ctx.fillStyle = color;
        ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
        ctx.textAlign = 'left';
        ctx.fillText(`${label} · ${(curve[endIdx] * 100).toFixed(0)}%`, x + 6, y + 4);
      }
    }

    if (state.strategy === 'all' || state.strategy === 'scratch') {
      drawCurve(state.curves.scratch, STRAT_COLOR.scratch, 'scratch');
    }
    if (state.strategy === 'all' || state.strategy === 'frozen') {
      drawCurve(state.curves.frozen, STRAT_COLOR.frozen, 'frozen');
    }
    if (state.strategy === 'all' || state.strategy === 'finetune') {
      drawCurve(state.curves.finetune, STRAT_COLOR.finetune, 'fine-tune');
    }
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('validation accuracy', plotX0, plotY0 - 6);

    // ───── Wall-clock counters strip ─────
    // Three columns side by side, each: strategy name, accumulated wall-clock seconds.
    const counterY = plotY0 + plotH + 36;
    const epoch = Math.min(state.epoch, state.maxEpoch);
    const counterRow = [
      { key: 'scratch',  label: 'scratch',   sec: SECONDS_PER_EPOCH.scratch  * epoch },
      { key: 'frozen',   label: 'frozen',    sec: SECONDS_PER_EPOCH.frozen   * epoch },
      { key: 'finetune', label: 'fine-tune', sec: SECONDS_PER_EPOCH.finetune * epoch },
    ];
    const cellW = (plotW + (plotX0 - padX)) / 3;
    const cellX0 = padX;
    counterRow.forEach((row, i) => {
      const cx = cellX0 + i * cellW;
      // Left swatch
      ctx.fillStyle = STRAT_COLOR[row.key];
      ctx.fillRect(cx, counterY - 10, 4, 28);
      ctx.fillStyle = INK_FADE;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      ctx.fillText('WALL-CLOCK', cx + 12, counterY - 2);
      ctx.fillStyle = INK;
      ctx.font = 'italic 18px "Source Serif 4", Georgia, serif';
      const mm = Math.floor(row.sec / 60);
      const ss = (row.sec % 60).toFixed(1);
      const tt = mm > 0 ? `${mm}m ${ss}s` : `${ss}s`;
      ctx.fillText(tt, cx + 12, counterY + 18);
      ctx.fillStyle = INK_FADE;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.fillText(row.label, cx + 12, counterY + 36);
    });

    // ─── Live mid-training annotation: point out frozen's early lead over scratch ───
    // The default state lands at mid-training; this caption makes the *current* gap
    // legible without waiting for the run to finish.
    if (epoch >= 4 && epoch < state.maxEpoch && state.strategy === 'all') {
      const accFrozenNow  = state.curves.frozen[epoch]   * 100;
      const accScratchNow = state.curves.scratch[epoch]  * 100;
      ctx.fillStyle = 'rgba(38,35,32,0.65)';
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'left';
      ctx.fillText(
        `at epoch ${epoch}: frozen already ${(accFrozenNow - accScratchNow).toFixed(0)}% ahead of scratch — and fine-tune is climbing past frozen.`,
        plotX0, plotY0 - 22
      );
    }

    // ─── Storytelling line: tie the wall-clock numbers to the accuracy gap ───
    // Once enough epochs have passed for the comparison to be meaningful, surface
    // the headline tradeoff in one line so the user doesn't have to read the
    // counters and the curves separately.
    if (epoch >= 8 && state.strategy === 'all') {
      const totalFrozen = SECONDS_PER_EPOCH.frozen   * state.maxEpoch;
      const totalFT     = SECONDS_PER_EPOCH.finetune * state.maxEpoch;
      const accFrozen = state.curves.frozen[state.maxEpoch]   * 100;
      const accFT     = state.curves.finetune[state.maxEpoch] * 100;
      const xMore = (totalFT / totalFrozen).toFixed(1);
      const accGap = (accFT - accFrozen).toFixed(1);
      const fmt = (s) => {
        const m = Math.floor(s / 60), x = Math.round(s % 60);
        return m > 0 ? `${m}m ${x}s` : `${s.toFixed(0)}s`;
      };
      ctx.fillStyle = 'rgba(122,31,36,0.78)';
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'left';
      ctx.fillText(
        `Frozen: ${fmt(totalFrozen)} total → ${accFrozen.toFixed(0)}%.  Fine-tune: ${fmt(totalFT)} → ${accFT.toFixed(0)}% — ${xMore}× longer for ${accGap}% more accuracy.`,
        padX, counterY + 56
      );
    }
  }

  function render() {
    Array.from(strategiesEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.strategy === state.strategy);
    });
    frozenLabel.textContent = `${state.frozen} / ${NUM_LAYERS} layers frozen`;
    dataLabel.textContent = `${state.dataSize} examples`;
    const finalScratch = state.curves.scratch[Math.min(state.epoch, state.maxEpoch)];
    const finalFrozen = state.curves.frozen[Math.min(state.epoch, state.maxEpoch)];
    const finalFT = state.curves.finetune[Math.min(state.epoch, state.maxEpoch)];
    const trainableFT = trainableParams('finetune', state.frozen);
    readoutEl.innerHTML = `
      <div class="row"><span>scratch</span><b>${(finalScratch * 100).toFixed(1)}%</b></div>
      <div class="row"><span>frozen</span><b>${(finalFrozen * 100).toFixed(1)}%</b></div>
      <div class="row"><span>fine-tune</span><b>${(finalFT * 100).toFixed(1)}%</b></div>
      <div class="row"><span>trainable</span><b>${trainableFT.toFixed(1)}M / ${TOTAL_PARAMS_M.toFixed(1)}M</b></div>
      <div class="row"><span>data</span><b>${state.dataSize}</b></div>`;
    canvasCtl.redraw();
  }

  render();
})();
