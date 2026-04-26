/* Vol. XXXIII — Model Interpretability */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK_FADE } = V;

  // Hand-crafted SHAP-like attributions for demo cases.
  const CASES = {
    'fraud — high risk': {
      baseline: 0.05,
      prediction: 0.83,
      label: 'P(fraud)',
      features: [
        { name: 'transaction amount', value: '$2,500', shap: 0.31 },
        { name: 'device new',         value: 'yes',    shap: 0.22 },
        { name: 'distance from home', value: '4,200 mi', shap: 0.16 },
        { name: 'merchant risk',      value: 'high',   shap: 0.09 },
        { name: 'time of day',        value: '03:42',  shap: 0.08 },
        { name: 'user age',           value: '34',     shap: -0.04 },
        { name: 'country match',      value: 'no',     shap: -0.04 },
      ],
    },
    'fraud — low risk': {
      baseline: 0.05,
      prediction: 0.04,
      label: 'P(fraud)',
      features: [
        { name: 'transaction amount', value: '$45',    shap: -0.06 },
        { name: 'device new',         value: 'no',     shap: -0.05 },
        { name: 'merchant risk',      value: 'low',    shap: -0.04 },
        { name: 'distance from home', value: '0.2 mi', shap: -0.03 },
        { name: 'time of day',        value: '14:18',  shap: 0.01 },
        { name: 'user age',           value: '52',     shap: 0.02 },
        { name: 'country match',      value: 'yes',    shap: 0.04 },
      ],
    },
    'credit — denied': {
      baseline: 0.40,
      prediction: 0.18,
      label: 'P(approve)',
      features: [
        { name: 'income',          value: '$28k', shap: -0.12 },
        { name: 'debt-to-income',  value: '0.55', shap: -0.10 },
        { name: 'credit history',  value: '6 mo', shap: -0.09 },
        { name: 'inquiries',       value: '7',    shap: -0.06 },
        { name: 'utilization',     value: '78%',  shap: -0.04 },
        { name: 'employment',      value: '3 yr', shap: 0.05 },
        { name: 'on-time payments',value: '98%',  shap: 0.06 },
      ],
    },
    'credit — approved': {
      baseline: 0.40,
      prediction: 0.91,
      label: 'P(approve)',
      features: [
        { name: 'income',          value: '$140k', shap: 0.18 },
        { name: 'credit history',  value: '12 yr', shap: 0.14 },
        { name: 'on-time payments',value: '100%',  shap: 0.13 },
        { name: 'utilization',     value: '8%',    shap: 0.10 },
        { name: 'debt-to-income',  value: '0.18',  shap: 0.08 },
        { name: 'inquiries',       value: '1',     shap: 0.04 },
        { name: 'employment',      value: '8 yr',  shap: 0.04 },
      ],
    },
  };

  const state = {
    caseKey: 'fraud — high risk',
    view: 'shap',
  };

  const casesEl = document.getElementById('cases');
  const viewsEl = document.getElementById('views');
  const readoutEl = document.getElementById('readout');
  const canvas = document.getElementById('canvas');

  Object.keys(CASES).forEach((k) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = k;
    b.dataset.case = k;
    b.addEventListener('click', () => { state.caseKey = k; render(); });
    casesEl.appendChild(b);
  });
  ['SHAP waterfall', 'LIME bars'].forEach((lbl, i) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = lbl;
    b.dataset.view = i === 0 ? 'shap' : 'lime';
    b.addEventListener('click', () => { state.view = b.dataset.view; render(); });
    viewsEl.appendChild(b);
  });

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    const c = CASES[state.caseKey];
    if (state.view === 'shap') drawWaterfall(ctx, size, c);
    else drawLime(ctx, size, c);
  }

  function drawWaterfall(ctx, size, c) {
    const padX = 36, padTop = 30, padBot = 30;
    const labelW = 200; // space for feature name+value on the left
    const plotX0 = padX + labelW;
    const plotW = size.w - plotX0 - padX;
    const rows = c.features.length + 2; // baseline + features + final
    const rowH = (size.h - padTop - padBot) / rows;
    const minVal = 0, maxVal = 1;
    const tx = (v) => plotX0 + (v - minVal) / (maxVal - minVal) * plotW;

    // Title
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText(`SHAP waterfall · ${c.label}`, padX, 18);

    // X axis ticks
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = INK_FADE;
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
      ctx.textAlign = 'center';
      ctx.fillText(v.toFixed(2), tx(v), size.h - padBot + 16);
      ctx.strokeStyle = 'rgba(38,35,32,0.10)';
      ctx.beginPath();
      ctx.moveTo(tx(v), padTop); ctx.lineTo(tx(v), size.h - padBot);
      ctx.stroke();
    });

    // Walk through baseline → each feature → prediction
    let acc = c.baseline;
    function drawRow(label, value, prevAcc, contribution, rowIdx, kind) {
      const y = padTop + rowIdx * rowH + rowH / 2;
      // Feature label
      ctx.fillStyle = '#262320';
      ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'left';
      ctx.fillText(label, padX, y - 4);
      if (value) {
        ctx.fillStyle = INK_FADE;
        ctx.font = '11px "JetBrains Mono", monospace';
        ctx.fillText(value, padX, y + 12);
      }
      // Bar
      const fromX = tx(prevAcc);
      const toX = tx(prevAcc + contribution);
      if (kind === 'baseline' || kind === 'final') {
        // Reference bar
        ctx.fillStyle = 'rgba(38,35,32,0.45)';
        const h = 6;
        ctx.fillRect(plotX0, y - h / 2, tx(prevAcc + (contribution || 0)) - plotX0, h);
      } else {
        ctx.fillStyle = contribution > 0 ? 'rgba(122,31,36,0.7)' : 'rgba(38,35,32,0.55)';
        const h = 18;
        const x = Math.min(fromX, toX);
        const w = Math.abs(toX - fromX);
        ctx.fillRect(x, y - h / 2, w, h);
        // Value label
        ctx.fillStyle = '#262320';
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`${contribution > 0 ? '+' : ''}${formatNum(contribution)}`, (fromX + toX) / 2, y + 4);
      }
    }

    // Baseline row
    drawRow('E[f(x)]', `baseline = ${formatNum(c.baseline)}`, 0, c.baseline, 0, 'baseline');

    // Each feature
    c.features.forEach((f, i) => {
      drawRow(f.name, f.value, acc, f.shap, i + 1);
      acc += f.shap;
    });

    // Final row
    drawRow('f(x)', `prediction = ${formatNum(c.prediction)}`, 0, acc, c.features.length + 1, 'final');
  }

  function drawLime(ctx, size, c) {
    const padX = 36, padTop = 30, padBot = 30;
    const labelW = 200;
    const plotX0 = padX + labelW;
    const plotW = size.w - plotX0 - padX;
    // For LIME we just plot the SHAP magnitudes as horizontal bars (local-linear coefficients are functionally similar in this demo)
    const sorted = [...c.features].sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
    const rowH = (size.h - padTop - padBot) / sorted.length;
    const maxAbs = Math.max(...sorted.map((f) => Math.abs(f.shap))) + 1e-6;
    const cx = plotX0 + plotW / 2;

    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('LIME local coefficients · sorted by importance', padX, 18);

    sorted.forEach((f, i) => {
      const y = padTop + i * rowH + rowH / 2;
      ctx.fillStyle = '#262320';
      ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'left';
      ctx.fillText(f.name, padX, y - 4);
      ctx.fillStyle = INK_FADE;
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.fillText(f.value, padX, y + 12);

      // Center axis
      ctx.strokeStyle = 'rgba(38,35,32,0.30)';
      ctx.beginPath();
      ctx.moveTo(cx, padTop); ctx.lineTo(cx, size.h - padBot);
      ctx.stroke();

      const w = (Math.abs(f.shap) / maxAbs) * (plotW / 2 - 8);
      const x = f.shap >= 0 ? cx : cx - w;
      ctx.fillStyle = f.shap >= 0 ? 'rgba(122,31,36,0.7)' : 'rgba(38,35,32,0.55)';
      ctx.fillRect(x, y - 8, w, 16);

      ctx.fillStyle = '#262320';
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.textAlign = f.shap >= 0 ? 'left' : 'right';
      ctx.fillText(`${f.shap > 0 ? '+' : ''}${formatNum(f.shap)}`,
        f.shap >= 0 ? x + w + 4 : x - 4, y + 4);
    });
  }

  function render() {
    Array.from(casesEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.case === state.caseKey);
    });
    Array.from(viewsEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.view === state.view);
    });
    const c = CASES[state.caseKey];
    let html = `<div style="font-family:var(--mono); font-size:11px; color:var(--ml-ink-fade); margin-bottom:6px">${c.label}</div>
                <div class="row"><span>baseline E[f]</span><b>${formatNum(c.baseline)}</b></div>
                <div class="row"><span>prediction f(x)</span><b style="color:var(--accent)">${formatNum(c.prediction)}</b></div>
                <div class="row"><span>net SHAP</span><b>${formatNum(c.prediction - c.baseline)}</b></div>`;
    readoutEl.innerHTML = html;
    canvasCtl.redraw();
  }

  render();
})();
