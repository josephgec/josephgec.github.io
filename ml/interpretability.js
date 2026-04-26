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
    view: 'shap', // 'shap' | 'whatif' | 'lime'
    // What-if: per-feature on/off (one map per case key, populated lazily).
    enabled: {},
  };

  // Initialize all features as enabled for all cases.
  Object.keys(CASES).forEach((k) => {
    state.enabled[k] = CASES[k].features.map(() => true);
  });

  const casesEl = document.getElementById('cases');
  const viewsEl = document.getElementById('views');
  const readoutEl = document.getElementById('readout');
  const canvas = document.getElementById('canvas');
  const whatifLabel = document.getElementById('whatif-rail-label');
  const whatifEl = document.getElementById('whatif-toggles');

  Object.keys(CASES).forEach((k) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = k;
    b.dataset.case = k;
    b.addEventListener('click', () => { state.caseKey = k; render(); });
    casesEl.appendChild(b);
  });
  ['SHAP waterfall', 'What-if', 'LIME bars'].forEach((lbl, i) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = lbl;
    b.dataset.view = ['shap', 'whatif', 'lime'][i];
    b.addEventListener('click', () => { state.view = b.dataset.view; render(); });
    viewsEl.appendChild(b);
  });

  // Build what-if toggles for the active case.
  function rebuildWhatIfToggles() {
    if (!whatifEl) return;
    whatifEl.innerHTML = '';
    const c = CASES[state.caseKey];
    const en = state.enabled[state.caseKey];
    c.features.forEach((f, i) => {
      const wrap = document.createElement('label');
      wrap.style.cssText = 'display:flex; align-items:center; gap:8px; font-family:var(--serif); font-style:italic; font-size:13px; color:var(--ink); cursor:pointer';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = en[i];
      cb.style.accentColor = 'var(--accent)';
      cb.addEventListener('change', () => {
        en[i] = cb.checked;
        render();
      });
      const name = document.createElement('span');
      name.textContent = f.name;
      name.style.flex = '1';
      const num = document.createElement('span');
      num.style.cssText = 'font-family:var(--mono); font-style:normal; font-size:11px; color:var(--ml-ink-fade)';
      num.textContent = (f.shap > 0 ? '+' : '') + formatNum(f.shap);
      wrap.appendChild(cb);
      wrap.appendChild(name);
      wrap.appendChild(num);
      whatifEl.appendChild(wrap);
    });
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  // Build the active feature list (sorted by |shap| desc, biggest pushers first)
  // and respect the what-if enabled mask in 'whatif' mode.
  function activeFeatures() {
    const c = CASES[state.caseKey];
    const en = state.enabled[state.caseKey];
    const feats = c.features.map((f, i) => ({ ...f, idx: i, on: en[i] }));
    // Sort by |shap| descending so biggest contributors are on top.
    feats.sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
    return feats;
  }

  function effectivePrediction() {
    const c = CASES[state.caseKey];
    const en = state.enabled[state.caseKey];
    let p = c.baseline;
    c.features.forEach((f, i) => { if (en[i]) p += f.shap; });
    return p;
  }

  function draw(ctx, size) {
    if (state.view === 'lime') drawLime(ctx, size);
    else drawWaterfall(ctx, size, state.view === 'whatif');
  }

  function drawWaterfall(ctx, size, isWhatIf) {
    const c = CASES[state.caseKey];
    const padX = 36, padTop = 30, padBot = 30;
    const labelW = 220;
    const plotX0 = padX + labelW;
    const plotW = size.w - plotX0 - padX;
    const feats = activeFeatures();
    const rows = feats.length + 2; // baseline + features + final
    const rowH = (size.h - padTop - padBot) / rows;
    const minVal = 0, maxVal = 1;
    const tx = (v) => plotX0 + (v - minVal) / (maxVal - minVal) * plotW;

    // Title
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    const sub = isWhatIf ? ' · what-if · toggles in left rail' : ' · sorted by |φᵢ|';
    ctx.fillText(`SHAP waterfall · ${c.label}${sub}`, padX, 18);

    // Math anchored to the picture: f(x) = E[f(x)] + Σ φᵢ — the accounting identity
    // that the waterfall is literally drawing, sitting right above the bars.
    ctx.fillStyle = '#262320';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('f(x) = E[f(x)] + Σ φᵢ', padX, 34);

    // ─── Mini legend (top-right): bar color → push direction ──
    // Newcomers should see at a glance: oxblood = pushes prediction up, ink = pushes down.
    {
      const legX0 = size.w - padX - 200;
      const legY0 = 12;
      const items = [
        { color: 'rgba(122,31,36,0.7)', label: '+ φᵢ pushes up' },
        { color: 'rgba(38,35,32,0.55)', label: '− φᵢ pushes down' },
      ];
      ctx.font = 'italic 10px "Source Serif 4", Georgia, serif';
      items.forEach((it, i) => {
        const xx = legX0 + i * 100;
        ctx.fillStyle = it.color;
        ctx.fillRect(xx, legY0, 14, 8);
        ctx.fillStyle = INK_FADE;
        ctx.textAlign = 'left';
        ctx.fillText(it.label, xx + 18, legY0 + 7);
      });
    }

    // What-if delta annotation: show how the toggled prediction differs from the full one.
    if (isWhatIf) {
      const fullPred = c.prediction;
      let toggledPred = c.baseline;
      const en = state.enabled[state.caseKey];
      c.features.forEach((f, i) => { if (en[i]) toggledPred += f.shap; });
      const delta = toggledPred - fullPred;
      const onCount = en.filter(Boolean).length;
      ctx.fillStyle = 'rgba(38,35,32,0.65)';
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'left';
      const sign = delta > 0 ? '+' : '';
      ctx.fillText(
        `${onCount}/${c.features.length} features on · prediction shifts ${sign}${formatNum(delta)} vs full f(x)=${formatNum(fullPred)}`,
        padX, size.h - 8
      );
    }

    // X axis ticks (vertical guidelines)
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

    // Walk through baseline → each (enabled) feature → prediction
    let acc = c.baseline;

    function drawRow(label, value, prevAcc, contribution, rowIdx, kind, isOff) {
      const y = padTop + rowIdx * rowH + rowH / 2;
      // Feature label
      ctx.fillStyle = isOff ? 'rgba(38,35,32,0.40)' : '#262320';
      ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'left';
      ctx.fillText((isOff ? '∅ ' : '') + label, padX, y - 4);
      if (value) {
        ctx.fillStyle = INK_FADE;
        ctx.font = '11px "JetBrains Mono", monospace';
        ctx.fillText(value, padX, y + 12);
      }
      // Bar
      if (kind === 'baseline' || kind === 'final') {
        ctx.fillStyle = kind === 'final' ? 'rgba(122,31,36,0.5)' : 'rgba(38,35,32,0.45)';
        const h = 6;
        const endX = tx(prevAcc + (contribution || 0));
        ctx.fillRect(plotX0, y - h / 2, endX - plotX0, h);
        // Marker
        ctx.fillStyle = kind === 'final' ? '#7a1f24' : '#262320';
        ctx.beginPath();
        ctx.arc(endX, y, 4, 0, Math.PI * 2);
        ctx.fill();
        // Numeric label at end
        ctx.fillStyle = '#262320';
        ctx.font = '11px "JetBrains Mono", monospace';
        ctx.textAlign = 'left';
        ctx.fillText(formatNum(prevAcc + (contribution || 0)), endX + 6, y + 4);
      } else if (isOff) {
        // Faded skipped row — show what would have been pushed, but don't accumulate.
        ctx.strokeStyle = 'rgba(38,35,32,0.25)';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 3]);
        const fromX = tx(prevAcc);
        const toX = tx(prevAcc + contribution);
        ctx.beginPath();
        ctx.moveTo(fromX, y);
        ctx.lineTo(toX, y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(38,35,32,0.40)';
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`(off · ${contribution > 0 ? '+' : ''}${formatNum(contribution)})`, (fromX + toX) / 2, y + 4);
      } else {
        const fromX = tx(prevAcc);
        const toX = tx(prevAcc + contribution);
        ctx.fillStyle = contribution > 0 ? 'rgba(122,31,36,0.7)' : 'rgba(38,35,32,0.55)';
        const h = 18;
        const x = Math.min(fromX, toX);
        const w = Math.abs(toX - fromX);
        ctx.fillRect(x, y - h / 2, w, h);
        // Light arrow indicating direction
        ctx.strokeStyle = '#fffdf6';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        if (contribution > 0) {
          ctx.moveTo(toX - 5, y - 4); ctx.lineTo(toX, y); ctx.lineTo(toX - 5, y + 4);
        } else {
          ctx.moveTo(toX + 5, y - 4); ctx.lineTo(toX, y); ctx.lineTo(toX + 5, y + 4);
        }
        ctx.stroke();
        // Value label
        ctx.fillStyle = '#fffdf6';
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        if (w > 32) {
          ctx.fillText(`${contribution > 0 ? '+' : ''}${formatNum(contribution)}`, (fromX + toX) / 2, y + 4);
        } else {
          ctx.fillStyle = '#262320';
          ctx.fillText(`${contribution > 0 ? '+' : ''}${formatNum(contribution)}`,
            contribution > 0 ? toX + 18 : toX - 18, y + 4);
        }
      }
    }

    // Baseline row
    drawRow('E[f(x)]  baseline', `model average = ${formatNum(c.baseline)}`,
            0, c.baseline, 0, 'baseline');

    // Each feature
    feats.forEach((f, i) => {
      const isOff = isWhatIf && !f.on;
      drawRow(f.name, f.value, acc, f.shap, i + 1, 'feature', isOff);
      if (!isOff) acc += f.shap;
    });

    // Final row
    const finalLabel = isWhatIf ? 'f(x)  prediction (with toggles)' : 'f(x)  prediction';
    drawRow(finalLabel, `final = ${formatNum(acc)}`, 0, acc, feats.length + 1, 'final');
  }

  function drawLime(ctx, size) {
    const c = CASES[state.caseKey];
    const padX = 36, padTop = 30, padBot = 30;
    const labelW = 220;
    const plotX0 = padX + labelW;
    const plotW = size.w - plotX0 - padX;
    // Sort by magnitude (largest at top)
    const sorted = [...c.features].sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
    const rowH = (size.h - padTop - padBot) / sorted.length;
    const maxAbs = Math.max(...sorted.map((f) => Math.abs(f.shap))) + 1e-6;
    const cx = plotX0 + plotW / 2;

    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText(`LIME local coefficients · ${c.label} · sorted by |coefficient|`, padX, 18);

    // Center axis
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, padTop); ctx.lineTo(cx, size.h - padBot);
    ctx.stroke();

    // Center axis label
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = INK_FADE;
    ctx.textAlign = 'center';
    ctx.fillText('0', cx, size.h - padBot + 14);
    ctx.textAlign = 'right';
    ctx.fillText('← pushes prediction down', cx - 10, size.h - padBot + 14);
    ctx.textAlign = 'left';
    ctx.fillText('pushes prediction up →', cx + 10, size.h - padBot + 14);

    sorted.forEach((f, i) => {
      const y = padTop + i * rowH + rowH / 2;
      ctx.fillStyle = '#262320';
      ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'left';
      ctx.fillText(f.name, padX, y - 4);
      ctx.fillStyle = INK_FADE;
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.fillText(f.value, padX, y + 12);

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
    // What-if visibility
    const showWI = state.view === 'whatif';
    if (whatifLabel) whatifLabel.style.display = showWI ? 'block' : 'none';
    if (whatifEl) whatifEl.style.display = showWI ? 'flex' : 'none';
    if (showWI) rebuildWhatIfToggles();

    const c = CASES[state.caseKey];
    const effPred = state.view === 'whatif' ? effectivePrediction() : c.prediction;
    let html = `<div style="font-family:var(--mono); font-size:11px; color:var(--ml-ink-fade); margin-bottom:6px">${c.label}</div>
                <div class="row"><span>baseline E[f]</span><b>${formatNum(c.baseline)}</b></div>
                <div class="row"><span>prediction f(x)</span><b style="color:var(--accent)">${formatNum(effPred)}</b></div>
                <div class="row"><span>net SHAP</span><b>${formatNum(effPred - c.baseline)}</b></div>`;
    if (state.view === 'whatif') {
      const en = state.enabled[state.caseKey];
      const onCount = en.filter(Boolean).length;
      html += `<div class="row"><span>features on</span><b>${onCount} / ${c.features.length}</b></div>`;
      html += `<div class="row"><span>full f(x)</span><b>${formatNum(c.prediction)}</b></div>`;
    }
    readoutEl.innerHTML = html;
    canvasCtl.redraw();
  }

  render();
})();
