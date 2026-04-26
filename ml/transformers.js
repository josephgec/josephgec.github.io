/* Vol. XXIV — The Transformer */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK_FADE } = V;

  const PROMPTS = {
    'the cat sat': ['the', 'cat', 'sat', 'on', 'the', 'mat'],
    'time flies': ['time', 'flies', 'like', 'an', 'arrow'],
    'I love ML': ['I', 'love', 'machine', 'learning'],
  };

  const NUM_LAYERS = 6;

  // Synthetic attention patterns per layer / head — we hand-design each head to
  // attend to a different relationship (previous-token, content, position-bias, uniform).
  function attnHead(words, headIdx, layerIdx) {
    const N = words.length;
    const M = Array.from({ length: N }, () => new Array(N).fill(0));
    for (let q = 0; q < N; q++) {
      for (let k = 0; k < N; k++) {
        let score = 0;
        if (headIdx === 0) {
          // Previous-token attention: strong attention to k = q-1
          score = (k === q - 1) ? 2.5 : 0;
        } else if (headIdx === 1) {
          // First-token attention (BOS-bias)
          score = (k === 0) ? 2 : 0;
        } else if (headIdx === 2) {
          // Content-similarity (string overlap as a stand-in)
          score = wordSim(words[q], words[k]) * 1.5;
        } else {
          // Local window
          score = -Math.abs(q - k) * 0.6;
        }
        // Layer-dependent perturbation
        score += Math.sin((q + k + layerIdx) * 0.7) * 0.2;
        M[q][k] = score;
      }
      // Causal mask: q can only attend to k <= q
      for (let k = q + 1; k < N; k++) M[q][k] = -Infinity;
      // Softmax
      const valid = M[q].slice(0, q + 1);
      const m = Math.max(...valid);
      const ex = valid.map((v) => Math.exp(v - m));
      const z = ex.reduce((a, b) => a + b, 0);
      for (let k = 0; k <= q; k++) M[q][k] = ex[k] / z;
    }
    return M;
  }
  function wordSim(a, b) {
    if (a === b) return 1;
    let c = 0;
    const la = a.length, lb = b.length;
    for (let i = 0; i < Math.min(la, lb); i++) if (a[i] === b[i]) c++;
    return c / Math.max(la, lb);
  }

  const state = {
    promptKey: 'the cat sat',
    heads: 4,
    layer: 0,
  };

  const promptsEl = document.getElementById('prompts');
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
    b.addEventListener('click', () => { state.promptKey = k; render(); });
    promptsEl.appendChild(b);
  });

  headsSlider.addEventListener('input', (e) => {
    state.heads = +e.target.value;
    headsLabel.textContent = `heads = ${state.heads}`;
    render();
  });
  layerSlider.addEventListener('input', (e) => {
    state.layer = +e.target.value;
    layerLabel.textContent = `layer ${state.layer} / ${NUM_LAYERS - 1}`;
    render();
  });

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    const words = PROMPTS[state.promptKey];
    const N = words.length;
    const layerIdx = state.layer;
    const heads = state.heads;

    // Layout: heads attention matrices side by side, with words labeling rows/cols
    const padTop = 60;
    const padBot = 40;
    const padLeft = 60;
    const padRight = 24;
    const gap = 24;
    const tileW = (size.w - padLeft - padRight - gap * (heads - 1)) / heads;
    const tileH = size.h - padTop - padBot;
    const cellSize = Math.min(tileW / N, tileH / N);

    // Title
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText(`layer ${layerIdx + 1} · ${heads} attention head${heads === 1 ? '' : 's'}`, size.w / 2, 20);

    // Row labels (left of first head)
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

      // Cells
      for (let q = 0; q < N; q++) {
        for (let k = 0; k < N; k++) {
          const w = M[q][k];
          if (!isFinite(w)) {
            ctx.fillStyle = 'rgba(38,35,32,0.04)';
          } else {
            const t = Math.max(0, Math.min(1, w));
            ctx.fillStyle = `rgba(122,31,36,${0.05 + t * 0.7})`;
          }
          ctx.fillRect(x0 + k * cellSize, y0 + q * cellSize, cellSize + 0.5, cellSize + 0.5);
        }
      }
      // Frame
      ctx.strokeStyle = 'rgba(38,35,32,0.30)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x0, y0, N * cellSize, N * cellSize);

      // Column labels under each head
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
      // Head label
      ctx.fillStyle = INK_FADE;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      const headDesc = ['previous', 'first-token', 'content', 'local'][h] || `head ${h}`;
      ctx.fillText(`head ${h + 1} · ${headDesc}`, x0 + N * cellSize / 2, y0 - 8);
    }
  }

  function render() {
    Array.from(promptsEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.prompt === state.promptKey);
    });
    headsSlider.value = state.heads;
    layerSlider.value = state.layer;
    headsLabel.textContent = `heads = ${state.heads}`;
    layerLabel.textContent = `layer ${state.layer} / ${NUM_LAYERS - 1}`;
    captionEl.innerHTML = `<span class="ml-cap-num">·</span> Each square is one attention head's <em>Q × Kᵀ</em> matrix at layer ${state.layer + 1}. Rows = queries, columns = keys. <em>Accent</em> means high attention weight. The lower triangle is filled (causal mask: token <em>q</em> can't attend to anything in the future).`;
    canvasCtl.redraw();
  }

  render();
})();
