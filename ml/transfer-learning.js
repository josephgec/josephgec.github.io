/* Vol. XXX — Transfer Learning */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK_FADE } = V;

  // Simulate three training curves: scratch, frozen, fine-tune.
  // Curve shapes are functions of (frozenLayers, dataSize, epoch).
  const NUM_LAYERS = 4;
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
    epoch: 0,
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

  function draw(ctx, size) {
    // Top section: layer architecture diagram with frozen vs trainable
    const archH = 110;
    const padX = 36;
    const layerW = (size.w - padX * 2 - 80) / (NUM_LAYERS + 1);
    const layerY = 24;
    const layerHpx = 56;

    // Source pretrain banner
    ctx.fillStyle = INK_FADE;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('pretrained network (source task) →', padX, 16);

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

    // Bottom section: training curves
    const plotY0 = archH + 30;
    const plotH = size.h - plotY0 - 30;
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
      drawCurve(state.curves.scratch, '#262320', 'scratch');
    }
    if (state.strategy === 'all' || state.strategy === 'frozen') {
      drawCurve(state.curves.frozen, '#3a6b5e', 'frozen');
    }
    if (state.strategy === 'all' || state.strategy === 'finetune') {
      drawCurve(state.curves.finetune, ACCENT, 'fine-tune');
    }
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('validation accuracy', plotX0, plotY0 - 6);
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
    readoutEl.innerHTML = `
      <div class="row"><span>scratch</span><b>${(finalScratch * 100).toFixed(1)}%</b></div>
      <div class="row"><span>frozen</span><b>${(finalFrozen * 100).toFixed(1)}%</b></div>
      <div class="row"><span>fine-tune</span><b>${(finalFT * 100).toFixed(1)}%</b></div>
      <div class="row"><span>data</span><b>${state.dataSize}</b></div>`;
    canvasCtl.redraw();
  }

  render();
})();
