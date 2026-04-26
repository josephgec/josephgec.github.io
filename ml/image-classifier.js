/* Vol. XVI — An Image Classifier, end to end */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK_FADE } = V;

  // Same image generators as Vol XV, three classes: circle, cross, diag.
  function genImage(kind, size = 32) {
    const img = [];
    for (let i = 0; i < size; i++) {
      const row = [];
      for (let j = 0; j < size; j++) {
        let v = 0.05;
        if (kind === 'cross') v = (Math.abs(i - size/2) < 2 || Math.abs(j - size/2) < 2) ? 1 : 0.05;
        else if (kind === 'circle') {
          const r = Math.hypot(j - size/2, i - size/2);
          v = r < size * 0.3 ? 1 : (r < size * 0.4 ? 0.5 : 0.05);
        } else if (kind === 'diag') v = (j > i - 4 && j < i + 4) ? 1 : 0.05;
        row.push(v);
      }
      img.push(row);
    }
    return img;
  }

  // Hand-tuned filters for class signatures. Channel 0 detects horizontal lines;
  // 1 detects vertical lines; 2 detects diagonals; 3 detects rings.
  const FILTERS_L1 = [
    [[-1,-1,-1],[2,2,2],[-1,-1,-1]],   // horizontal
    [[-1,2,-1],[-1,2,-1],[-1,2,-1]],   // vertical
    [[2,-1,-1],[-1,2,-1],[-1,-1,2]],   // diag /
    [[-1,1,-1],[1,0,1],[-1,1,-1]],      // ring-ish
  ];
  // Layer 2 filters mix L1 channels. Each L2 filter is 3×3×4. We'll just average pairs.
  function convolve(map, k) {
    const H = map.length, W = map[0].length;
    const KH = k.length, KW = k[0].length;
    const py = Math.floor(KH/2), px = Math.floor(KW/2);
    const out = [];
    for (let i = 0; i < H; i++) {
      const row = [];
      for (let j = 0; j < W; j++) {
        let s = 0;
        for (let ki = 0; ki < KH; ki++) for (let kj = 0; kj < KW; kj++) {
          const ii = i + ki - py, jj = j + kj - px;
          if (ii >= 0 && ii < H && jj >= 0 && jj < W) s += map[ii][jj] * k[ki][kj];
        }
        row.push(Math.max(0, s));
      }
      out.push(row);
    }
    return out;
  }
  function pool(map, sz = 2) {
    const H = map.length, W = map[0].length;
    const out = [];
    for (let i = 0; i < H; i += sz) {
      const row = [];
      for (let j = 0; j < W; j += sz) {
        let m = -Infinity;
        for (let ki = 0; ki < sz; ki++) for (let kj = 0; kj < sz; kj++) {
          const ii = i + ki, jj = j + kj;
          if (ii < H && jj < W) m = Math.max(m, map[ii][jj]);
        }
        row.push(m);
      }
      out.push(row);
    }
    return out;
  }
  function sumMap(map) {
    let s = 0;
    for (const row of map) for (const v of row) s += v;
    return s;
  }
  function softmax(arr) {
    const m = Math.max(...arr);
    const ex = arr.map(v => Math.exp(v - m));
    const sum = ex.reduce((a, b) => a + b, 0);
    return ex.map(v => v / sum);
  }

  function pipeline(img) {
    // Layer 1: 4 conv channels + ReLU + pool
    const l1 = FILTERS_L1.map(k => pool(convolve(img, k)));
    // Layer 2: 8 channels = 4 originals + 4 mixed pairs (cheap stand-in for learned mixing)
    const l2 = [];
    for (let i = 0; i < 4; i++) l2.push(pool(convolve(l1[i], [[0,1,0],[1,-2,1],[0,1,0]])));
    for (let i = 0; i < 4; i++) {
      const mixed = l1[i].map((row, ri) => row.map((v, ci) => v + l1[(i+1)%4][ri][ci]));
      l2.push(pool(convolve(mixed, [[1,1,1],[1,-1,1],[1,1,1]])));
    }
    // Flatten and score: hand-set weights so each class lights up for its signature filter.
    // Class scores are a weighted sum of channel-sums. Channel 0 (horiz) → cross+diag; ch1 (vert) → cross; ch2 (diag) → diag; ch3 (ring) → circle.
    const ch = l1.map(sumMap);
    const logits = [
      0.6 * ch[3] - 0.2 * ch[0] - 0.2 * ch[1] - 0.2 * ch[2],     // circle
      0.5 * ch[0] + 0.5 * ch[1] - 0.3 * ch[3] - 0.3 * ch[2],     // cross
      0.7 * ch[2] - 0.2 * ch[0] - 0.2 * ch[1] - 0.2 * ch[3],     // diag
    ];
    return { l1, l2, logits, probs: softmax(logits.map(v => v * 0.05)) };
  }

  const state = {
    imageKind: 'circle',
    step: 5,
    playing: false,
    timer: null,
  };

  const imagesEl = document.getElementById('images');
  const replayBtn = document.getElementById('replay-btn');
  const stepSlider = document.getElementById('step-slider');
  const stepLabel = document.getElementById('step-label');
  const captionEl = document.getElementById('caption');
  const probsEl = document.getElementById('probs');
  const canvas = document.getElementById('canvas');

  ['circle', 'cross', 'diag'].forEach((k) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = k;
    b.dataset.image = k;
    b.addEventListener('click', () => {
      state.imageKind = k;
      state.step = 5;
      render();
    });
    imagesEl.appendChild(b);
  });
  replayBtn.addEventListener('click', () => {
    state.step = 0; state.playing = true;
    if (state.timer) clearInterval(state.timer);
    state.timer = setInterval(() => {
      if (state.step >= 5) { state.playing = false; clearInterval(state.timer); state.timer = null; return; }
      state.step++;
      render();
    }, 600);
    render();
  });
  stepSlider.addEventListener('input', (e) => {
    state.playing = false;
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    state.step = +e.target.value;
    render();
  });

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    const img = genImage(state.imageKind);
    const out = pipeline(img);

    // 6 stages: input → conv1 → pool1 → conv2 → pool2 → output
    // Layout: stack of small thumbnails per stage in a row.
    function drawMap(map, x0, y0, cell) {
      let max = 0;
      for (const r of map) for (const v of r) if (v > max) max = v;
      for (let i = 0; i < map.length; i++) {
        for (let j = 0; j < map[0].length; j++) {
          const t = Math.max(0, Math.min(1, map[i][j] / Math.max(0.01, max)));
          const r = Math.round(247 - t * 130);
          const g = Math.round(244 - t * 195);
          const b = Math.round(236 - t * 195);
          ctx.fillStyle = `rgb(${r},${g},${b})`;
          ctx.fillRect(x0 + j * cell, y0 + i * cell, cell + 0.5, cell + 0.5);
        }
      }
      ctx.strokeStyle = 'rgba(38,35,32,0.30)';
      ctx.lineWidth = 0.8;
      ctx.strokeRect(x0, y0, map[0].length * cell, map.length * cell);
    }

    const stageX = [
      0,
      size.w * 0.20,
      size.w * 0.40,
      size.w * 0.60,
      size.w * 0.80,
    ];

    // Stage 0: input
    if (state.step >= 0) {
      const c = (size.w * 0.18) / 32;
      drawMap(img, stageX[0], (size.h - c * 32) / 2, c);
      label('input · 32×32', stageX[0] + (c * 32) / 2, size.h - 8);
    }
    // Stage 1: l1 (4 maps stacked vertically)
    if (state.step >= 1) {
      const ch = out.l1; // each is 16×16 after pool already applied above? No — l1 contains pool output. Wait: my pipeline ran convolve THEN pool, so l1 = 16×16. Re-display.
      const cell = (size.w * 0.16) / 16;
      const stackH = ch.length * (cell * 16 + 4);
      const yOff = (size.h - stackH) / 2;
      ch.forEach((m, i) => {
        drawMap(m, stageX[1], yOff + i * (cell * 16 + 4), cell);
      });
      label('conv1 + pool · 16×16×4', stageX[1] + (cell * 16) / 2, size.h - 8);
    }
    // Stage 2: pool1 actually = l1, label differently
    if (state.step >= 2) {
      const ch = out.l2; // each is 8×8
      const cell = (size.w * 0.14) / 8;
      const cols = 4, rows = 2;
      const tileW = cell * 8 + 4;
      const totalH = rows * tileW;
      const yOff = (size.h - totalH) / 2;
      ch.forEach((m, i) => {
        const col = i % cols, row = Math.floor(i / cols);
        drawMap(m, stageX[2] + col * (cell * 8 / 4), yOff + row * tileW, cell);
      });
      label('conv2 + pool · 8×8×8', stageX[2] + (cell * 8) / 2, size.h - 8);
    }
    // Stage 3: flatten (a tall thin strip)
    if (state.step >= 3) {
      const flat = out.l2.flatMap(m => m.flat());
      const N = flat.length;
      const w = size.w * 0.05;
      const h = size.h * 0.7;
      const x0 = stageX[3], y0 = (size.h - h) / 2;
      const cellH = h / N;
      const max = Math.max(...flat.map(Math.abs)) + 1e-6;
      for (let i = 0; i < N; i++) {
        const t = Math.max(0, Math.min(1, flat[i] / max));
        const r = Math.round(247 - t * 130);
        const g = Math.round(244 - t * 195);
        const b = Math.round(236 - t * 195);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x0, y0 + i * cellH, w, cellH + 0.5);
      }
      ctx.strokeStyle = 'rgba(38,35,32,0.30)';
      ctx.strokeRect(x0, y0, w, h);
      label('flatten · 512', x0 + w / 2, size.h - 8);
    }
    // Stage 5: probabilities (3 horizontal bars)
    if (state.step >= 5) {
      const probs = out.probs;
      const labels = ['circle', 'cross', 'diag'];
      const x0 = stageX[4];
      const w = size.w * 0.16;
      const h = size.h * 0.6;
      const y0 = (size.h - h) / 2;
      const barH = (h - 30) / 3;
      labels.forEach((lbl, i) => {
        const p = probs[i];
        const bw = w * p;
        ctx.fillStyle = i === probs.indexOf(Math.max(...probs)) ? ACCENT : 'rgba(38,35,32,0.35)';
        ctx.fillRect(x0, y0 + i * barH + 4, bw, barH - 12);
        ctx.fillStyle = INK_FADE;
        ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
        ctx.textAlign = 'left';
        ctx.fillText(`${lbl} · ${(p * 100).toFixed(0)}%`, x0, y0 + i * barH + barH - 4);
      });
      label('softmax · 3 classes', x0 + w / 2, size.h - 8);
    }

    function label(text, x, y) {
      ctx.fillStyle = INK_FADE;
      ctx.font = '11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText(text, x, y);
    }
  }

  function captionFor(step) {
    return [
      'Stage 0 — input image. 32×32 grayscale pixels.',
      'Stage 1 — conv layer 1 produces 4 feature maps. Each filter responds to a different orientation of edge.',
      'Stage 2 — conv layer 2 mixes the previous channels into 8 deeper features. Spatial size halves to 8×8.',
      'Stage 3 — flatten. 8×8×8 = 512 numbers in one long vector.',
      'Stage 4 — dense layer turns the vector into 3 logits.',
      'Stage 5 — softmax. The largest score is the prediction.',
    ][step];
  }

  function render() {
    Array.from(imagesEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.image === state.imageKind);
    });
    stepSlider.value = state.step;
    stepLabel.textContent = `${state.step}/5`;
    captionEl.innerHTML = `<span class="ml-cap-num">·</span> ${captionFor(state.step)}`;

    const img = genImage(state.imageKind);
    const out = pipeline(img);
    const probs = out.probs;
    const labels = ['circle', 'cross', 'diag'];
    let html = '';
    labels.forEach((lbl, i) => {
      html += `<div class="row"><span>P(${lbl})</span><b style="color:${i === probs.indexOf(Math.max(...probs)) ? 'var(--accent)' : 'var(--ink)'}">${(probs[i] * 100).toFixed(1)}%</b></div>`;
    });
    probsEl.innerHTML = html;

    canvasCtl.redraw();
  }

  render();
})();
