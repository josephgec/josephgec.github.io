/* Vol. XXIV — The Transformer */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK, INK_FADE } = V;

  const PROMPTS = {
    'the cat sat': ['the', 'cat', 'sat', 'on', 'the', 'mat'],
    'time flies': ['time', 'flies', 'like', 'an', 'arrow'],
    'I love ML': ['I', 'love', 'machine', 'learning'],
  };

  const NUM_LAYERS = 6;
  const HEAD_NAMES = ['previous-token', 'first-token (BOS-bias)', 'content match', 'local window'];

  function wordSim(a, b) {
    if (a === b) return 1;
    let c = 0;
    const la = a.length, lb = b.length;
    for (let i = 0; i < Math.min(la, lb); i++) if (a[i] === b[i]) c++;
    return c / Math.max(la, lb);
  }

  function attnHead(words, headIdx, layerIdx) {
    const N = words.length;
    const M = Array.from({ length: N }, () => new Array(N).fill(0));
    for (let q = 0; q < N; q++) {
      for (let k = 0; k < N; k++) {
        let score = 0;
        if (headIdx === 0) score = (k === q - 1) ? 2.5 : 0;
        else if (headIdx === 1) score = (k === 0) ? 2 : 0;
        else if (headIdx === 2) score = wordSim(words[q], words[k]) * 1.5;
        else score = -Math.abs(q - k) * 0.6;
        score += Math.sin((q + k + layerIdx) * 0.7) * 0.2;
        M[q][k] = score;
      }
      for (let k = q + 1; k < N; k++) M[q][k] = -Infinity;
      const valid = M[q].slice(0, q + 1);
      const m = Math.max(...valid);
      const ex = valid.map((v) => Math.exp(v - m));
      const z = ex.reduce((a, b) => a + b, 0);
      for (let k = 0; k <= q; k++) M[q][k] = ex[k] / z;
    }
    return M;
  }

  // For each head, compute a label based on the argmax-attended token across queries.
  function describeHead(words, headIdx, layerIdx) {
    const M = attnHead(words, headIdx, layerIdx);
    const N = words.length;
    let prevHits = 0, firstHits = 0;
    for (let q = 1; q < N; q++) {
      let best = -1, bv = -1;
      for (let k = 0; k <= q; k++) {
        if (M[q][k] > bv) { bv = M[q][k]; best = k; }
      }
      if (best === q - 1) prevHits++;
      if (best === 0) firstHits++;
    }
    if (prevHits > (N - 1) * 0.6) return 'attends to previous';
    if (firstHits > (N - 1) * 0.6) return 'attends to first';
    if (headIdx === 2) return 'content match';
    return 'local window';
  }

  // ───────── Views ─────────
  const VIEWS = [
    { key: 'block',  label: 'One block · step through' },
    { key: 'heads',  label: 'Heads at one layer' },
    { key: 'layers', label: 'Across layers · one query' },
  ];
  const CAPTIONS = {
    block:  'Step through one transformer block: residual stream → LayerNorm → attention writes its update → ADD back to residual → LayerNorm → FFN writes its update → ADD back. The "+" steps are why information persists across hundreds of layers.',
    heads:  'At one layer, each head is one Q × Kᵀ softmax — its own learned attention pattern. Rows are queries, columns are keys; orange cells are high weights. The lower-triangle-only fill is the causal mask: a token can\'t attend to anything in the future.',
    layers: 'Same query word, different layers. Early layers tend to attend locally (previous token, position); later layers form longer-range, more semantic patterns. Watch the focus shift as you scrub through the layer slider.',
  };

  const state = {
    promptKey: 'the cat sat',
    heads: 4,
    layer: 0,
    view: 'block',
    blockStep: 5,    // 0..6 steps through the block animation
    blockTimer: null,
    queryIdx: 1,     // which query to track in 'layers' view
  };

  // ───────── DOM ─────────
  const promptsEl = document.getElementById('prompts');
  const tabsEl = document.getElementById('view-tabs');
  const headsSlider = document.getElementById('heads-slider');
  const headsLabel = document.getElementById('heads-label');
  const layerSlider = document.getElementById('layer-slider');
  const layerLabel = document.getElementById('layer-label');
  const captionEl = document.getElementById('caption');
  const canvas = document.getElementById('canvas');

  Object.keys(PROMPTS).forEach((k) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = k;
    b.dataset.prompt = k;
    b.addEventListener('click', () => {
      state.promptKey = k;
      const words = PROMPTS[k];
      if (state.queryIdx >= words.length) state.queryIdx = words.length - 1;
      render();
    });
    promptsEl.appendChild(b);
  });

  VIEWS.forEach((v) => {
    const b = document.createElement('button');
    b.textContent = v.label;
    b.dataset.view = v.key;
    b.addEventListener('click', () => {
      state.view = v.key;
      if (v.key === 'block') runBlockAnim();
      render();
    });
    tabsEl.appendChild(b);
  });

  headsSlider.addEventListener('input', (e) => {
    state.heads = +e.target.value;
    render();
  });
  layerSlider.addEventListener('input', (e) => {
    state.layer = +e.target.value;
    render();
  });

  function runBlockAnim() {
    if (state.blockTimer) { clearInterval(state.blockTimer); state.blockTimer = null; }
    state.blockStep = 0;
    state.blockTimer = setInterval(() => {
      state.blockStep += 0.05;
      if (state.blockStep >= 6) {
        state.blockStep = 6;
        clearInterval(state.blockTimer);
        state.blockTimer = null;
      }
      render();
    }, 40);
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });
  canvas.addEventListener('click', (e) => {
    if (state.view === 'block') {
      // Re-trigger the animation on click
      runBlockAnim();
    } else if (state.view === 'layers') {
      // Click to set query idx; we'll find which row on the layout
      // For simplicity, cycle through queries
      const words = PROMPTS[state.promptKey];
      state.queryIdx = (state.queryIdx + 1) % words.length;
      render();
    }
  });

  function draw(ctx, size) {
    if (state.view === 'heads') return drawHeads(ctx, size);
    if (state.view === 'layers') return drawLayers(ctx, size);
    return drawBlock(ctx, size);
  }

  // ─────────── View 1: Step through one block ───────────
  function drawBlock(ctx, size) {
    const words = PROMPTS[state.promptKey];
    const N = words.length;

    // Vertical timeline of stages: x-line residual stream, branches off and rejoins
    const padX = 80;
    const padY = 50;
    const W = size.w - padX * 2;
    const H = size.h - padY * 2;
    const cx = padX + W * 0.18;        // residual highway x
    const branchX = padX + W * 0.55;   // branches go right
    const lnAtnY = padY + H * 0.12;
    const atnY   = padY + H * 0.30;
    const addAtnY = padY + H * 0.48;
    const lnFfnY = padY + H * 0.62;
    const ffnY   = padY + H * 0.78;
    const addFfnY = padY + H * 0.96;

    // Title
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('one block · residual stream picks up two updates', size.w / 2, 22);

    // Residual stream backbone
    ctx.strokeStyle = 'rgba(122,31,36,0.55)';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(cx, padY);
    ctx.lineTo(cx, padY + H);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(122,31,36,0.7)';
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('residual stream', cx + 8, padY - 6);

    // Stages
    const stages = [
      { y: lnAtnY,  label: 'LayerNorm', side: 'left',  step: 0 },
      { y: atnY,    label: 'Attention', side: 'branch', step: 1 },
      { y: addAtnY, label: '+ residual', side: 'add',  step: 2 },
      { y: lnFfnY,  label: 'LayerNorm', side: 'left',  step: 3 },
      { y: ffnY,    label: 'FFN (MLP)', side: 'branch', step: 4 },
      { y: addFfnY, label: '+ residual', side: 'add',  step: 5 },
    ];

    // Draw each stage
    const t = state.blockStep;
    stages.forEach((s, i) => {
      const active = t >= s.step;
      const alpha = active ? 1 : 0.22;
      ctx.globalAlpha = alpha;
      if (s.side === 'branch') {
        // Box on the right with attention heatmap (for atn) or FFN bars (for ffn)
        const boxX = branchX;
        const boxY = s.y - 28;
        const boxW = W - (branchX - padX) - 20;
        const boxH = 56;
        ctx.fillStyle = 'rgba(122,31,36,0.06)';
        ctx.strokeStyle = ACCENT;
        ctx.lineWidth = 1;
        ctx.fillRect(boxX, boxY, boxW, boxH);
        ctx.strokeRect(boxX, boxY, boxW, boxH);
        ctx.fillStyle = ACCENT;
        ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
        ctx.textAlign = 'left';
        ctx.fillText(s.label, boxX + 8, boxY + 16);

        if (s.label === 'Attention') {
          // Mini heatmap: head 0 (or whatever)
          const M = attnHead(words, 0, state.layer);
          const cell = Math.min((boxW - 16) / N, (boxH - 24) / N);
          const mx0 = boxX + boxW - cell * N - 8;
          const my0 = boxY + 22;
          for (let q = 0; q < N; q++) {
            for (let k = 0; k < N; k++) {
              const w = M[q][k];
              if (!isFinite(w)) ctx.fillStyle = 'rgba(38,35,32,0.04)';
              else ctx.fillStyle = `rgba(122,31,36,${0.05 + w * 0.7})`;
              ctx.fillRect(mx0 + k * cell, my0 + q * cell, cell + 0.5, cell + 0.5);
            }
          }
          ctx.strokeStyle = 'rgba(38,35,32,0.30)';
          ctx.strokeRect(mx0, my0, cell * N, cell * N);
          ctx.fillStyle = INK_FADE;
          ctx.font = '9px "JetBrains Mono", monospace';
          ctx.textAlign = 'left';
          ctx.fillText('softmax(QKᵀ/√d)', boxX + 8, boxY + 32);
          ctx.fillText('mixes across tokens', boxX + 8, boxY + 44);
        } else {
          // FFN: token-wise bars
          const cell = Math.min((boxW - 16) / N, 12);
          const mx0 = boxX + boxW - cell * N - 8;
          const my0 = boxY + 22;
          for (let i2 = 0; i2 < N; i2++) {
            const h = 24 + Math.sin((i2 + state.layer) * 1.3) * 6;
            ctx.fillStyle = 'rgba(122,31,36,0.55)';
            ctx.fillRect(mx0 + i2 * cell, my0 + 28 - h, cell - 1, h);
          }
          ctx.fillStyle = INK_FADE;
          ctx.font = '9px "JetBrains Mono", monospace';
          ctx.textAlign = 'left';
          ctx.fillText('per-token MLP', boxX + 8, boxY + 32);
          ctx.fillText('no token mixing', boxX + 8, boxY + 44);
        }

        // Branch arrow from residual → box
        ctx.strokeStyle = active ? ACCENT : 'rgba(122,31,36,0.25)';
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(cx, s.y);
        ctx.lineTo(boxX - 6, s.y);
        ctx.stroke();
        // arrowhead
        ctx.fillStyle = active ? ACCENT : 'rgba(122,31,36,0.25)';
        ctx.beginPath();
        ctx.moveTo(boxX, s.y);
        ctx.lineTo(boxX - 8, s.y - 4);
        ctx.lineTo(boxX - 8, s.y + 4);
        ctx.fill();

      } else if (s.side === 'left') {
        // LayerNorm marker
        const lx = cx - 80;
        const ly = s.y - 10;
        ctx.fillStyle = active ? 'rgba(38,35,32,0.85)' : 'rgba(38,35,32,0.4)';
        ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
        ctx.textAlign = 'right';
        ctx.fillText(s.label, cx - 14, s.y + 4);
        // tick on the residual stream
        ctx.fillStyle = active ? '#262320' : 'rgba(38,35,32,0.4)';
        ctx.fillRect(cx - 4, s.y - 1, 8, 2);
      } else {
        // ADD node — circle on stream
        ctx.fillStyle = active ? ACCENT : 'rgba(122,31,36,0.25)';
        ctx.beginPath();
        ctx.arc(cx, s.y, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#fffdf6';
        ctx.font = 'bold 9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('+', cx, s.y + 1);
        ctx.textBaseline = 'alphabetic';
        // label to right of stream
        ctx.fillStyle = active ? INK : 'rgba(38,35,32,0.4)';
        ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
        ctx.textAlign = 'left';
        ctx.fillText(s.label, cx + 14, s.y + 4);
        // arrow back into stream
        if (i > 0) {
          const prev = stages[i - 1];
          if (prev.side === 'branch') {
            const boxX = branchX;
            ctx.strokeStyle = active ? ACCENT : 'rgba(122,31,36,0.25)';
            ctx.lineWidth = 1.4;
            ctx.beginPath();
            ctx.moveTo(boxX - 6, prev.y);
            ctx.lineTo(boxX - 30, prev.y);
            ctx.lineTo(boxX - 30, s.y);
            ctx.lineTo(cx + 8, s.y);
            ctx.stroke();
            ctx.fillStyle = active ? ACCENT : 'rgba(122,31,36,0.25)';
            ctx.beginPath();
            ctx.moveTo(cx + 6, s.y);
            ctx.lineTo(cx + 14, s.y - 4);
            ctx.lineTo(cx + 14, s.y + 4);
            ctx.fill();
          }
        }
      }
      ctx.globalAlpha = 1;
    });

    // Traveling vector marker on the residual stream
    const totalT = Math.min(t, 6);
    const yPos = padY + (totalT / 6) * H;
    ctx.fillStyle = ACCENT;
    ctx.beginPath();
    ctx.arc(cx, yPos, 5, 0, Math.PI * 2);
    ctx.fill();

    // Storytelling annotation — what the dot is doing right now
    const stepLabels = [
      'normalizing for attention',
      'attention mixes context across tokens',
      'attention\'s update added back',
      'normalizing for FFN',
      'FFN transforms each token in place',
      'FFN\'s update added back — block done',
      'block done — output goes to next block',
    ];
    const stageIdx = Math.min(stepLabels.length - 1, Math.floor(totalT));
    ctx.fillStyle = ACCENT;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText(stepLabels[stageIdx], cx + 12, yPos + 4);

    // Hint
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('click canvas to replay', size.w / 2, size.h - 8);
  }

  // ─────────── View 2: Heads at one layer ───────────
  function drawHeads(ctx, size) {
    const words = PROMPTS[state.promptKey];
    const N = words.length;
    const layerIdx = state.layer;
    const heads = state.heads;

    const padTop = 60;
    const padBot = 50;
    const padLeft = 60;
    const padRight = 24;
    const gap = 24;
    const tileW = (size.w - padLeft - padRight - gap * (heads - 1)) / heads;
    const tileH = size.h - padTop - padBot;
    const cellSize = Math.min(tileW / N, tileH / N);

    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText(`layer ${layerIdx + 1} · ${heads} attention head${heads === 1 ? '' : 's'}`, size.w / 2, 20);

    ctx.fillStyle = '#262320';
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'right';
    for (let i = 0; i < N; i++) {
      ctx.fillText(words[i], padLeft - 6, padTop + i * cellSize + cellSize / 2 + 3);
    }

    for (let h = 0; h < heads; h++) {
      const M = attnHead(words, h, layerIdx);
      const x0 = padLeft + h * (tileW + gap);
      const y0 = padTop;

      for (let q = 0; q < N; q++) {
        for (let k = 0; k < N; k++) {
          const w = M[q][k];
          if (!isFinite(w)) ctx.fillStyle = 'rgba(38,35,32,0.04)';
          else ctx.fillStyle = `rgba(122,31,36,${0.05 + w * 0.7})`;
          ctx.fillRect(x0 + k * cellSize, y0 + q * cellSize, cellSize + 0.5, cellSize + 0.5);
        }
      }
      ctx.strokeStyle = 'rgba(38,35,32,0.30)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x0, y0, N * cellSize, N * cellSize);

      ctx.fillStyle = INK_FADE;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      for (let k = 0; k < N; k++) {
        ctx.save();
        ctx.translate(x0 + k * cellSize + cellSize / 2, y0 + N * cellSize + 8);
        ctx.rotate(-Math.PI / 4);
        ctx.textAlign = 'right';
        ctx.fillText(words[k], 0, 0);
        ctx.restore();
      }
      // Head label — empirical
      const desc = describeHead(words, h, layerIdx);
      ctx.fillStyle = INK_FADE;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText(`head ${h + 1} · ${desc}`, x0 + N * cellSize / 2, y0 - 8);
    }
  }

  // ─────────── View 3: Same query across layers ───────────
  function drawLayers(ctx, size) {
    const words = PROMPTS[state.promptKey];
    const N = words.length;
    const qIdx = Math.min(state.queryIdx, N - 1);
    // Four-panel window centered on the slider's current layer (so scrubbing
    // really moves the view). Clamp into [0, NUM_LAYERS-1].
    const center = state.layer;
    let start = Math.max(0, Math.min(NUM_LAYERS - 4, center - 1));
    const showLayers = [start, start + 1, start + 2, start + 3];
    const headIdx = 2; // content head — varies most across layers

    const padTop = 70;
    const padBot = 30;
    const padLeft = 30;
    const padRight = 24;
    const gap = 18;
    const numPanels = showLayers.length;
    const tileW = (size.w - padLeft - padRight - gap * (numPanels - 1)) / numPanels;
    const tileH = size.h - padTop - padBot;
    const barH = Math.min(tileH * 0.7, 220);

    // Title
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText(`query word "${words[qIdx]}" — attention pattern across layers`, size.w / 2, 22);

    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('click canvas to cycle query word', size.w / 2, 42);

    showLayers.forEach((L, idx) => {
      const M = attnHead(words, headIdx, L);
      const row = M[qIdx];
      const x0 = padLeft + idx * (tileW + gap);
      const y0 = padTop + 16;
      // Panel label — highlight the slider-selected layer
      const isActive = L === state.layer;
      ctx.fillStyle = isActive ? ACCENT : INK;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`LAYER ${L + 1}${isActive ? ' ◄' : ''}`, x0 + tileW / 2, y0 - 6);

      // Bar chart of attention from query → keys
      const barTop = y0 + 12;
      const barBottom = y0 + barH;
      const colW = tileW / N;
      for (let k = 0; k < N; k++) {
        const w = isFinite(row[k]) ? row[k] : 0;
        const h = w * (barBottom - barTop);
        ctx.fillStyle = `rgba(122,31,36,${0.18 + w * 0.7})`;
        ctx.fillRect(x0 + k * colW + 2, barBottom - h, colW - 4, h);
      }
      // Word labels under bars
      ctx.fillStyle = INK;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      for (let k = 0; k < N; k++) {
        const isQ = k === qIdx;
        ctx.fillStyle = isQ ? ACCENT : INK;
        ctx.fillText(words[k], x0 + k * colW + colW / 2, barBottom + 14);
      }
      // Frame
      ctx.strokeStyle = 'rgba(38,35,32,0.18)';
      ctx.strokeRect(x0, y0, tileW, barH + 4);
    });
  }

  function render() {
    Array.from(promptsEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.prompt === state.promptKey);
    });
    Array.from(tabsEl.children).forEach((btn) => {
      btn.classList.toggle('on', btn.dataset.view === state.view);
    });
    headsSlider.value = state.heads;
    layerSlider.value = state.layer;
    headsLabel.textContent = `heads = ${state.heads}`;
    layerLabel.textContent = `layer ${state.layer} / ${NUM_LAYERS - 1}`;

    captionEl.innerHTML = '<span class="ml-cap-num">·</span> ' + CAPTIONS[state.view];
    canvasCtl.redraw();
  }

  render();
  // Kick the block animation on first mount
  runBlockAnim();
})();
