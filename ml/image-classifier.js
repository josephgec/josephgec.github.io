/* Vol. XVI — An Image Classifier, end to end */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK, INK_FADE } = V;

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

  const FILTERS_L1 = [
    [[-1,-1,-1],[2,2,2],[-1,-1,-1]],   // 0 horizontal
    [[-1,2,-1],[-1,2,-1],[-1,2,-1]],   // 1 vertical
    [[2,-1,-1],[-1,2,-1],[-1,-1,2]],   // 2 diag
    [[-1,1,-1],[1,0,1],[-1,1,-1]],     // 3 ring-ish
  ];
  const FILTERS_L1_NAMES = ['horizontal', 'vertical', 'diagonal', 'ring'];

  function convolveValid(map, k) {
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
    const l1 = FILTERS_L1.map(k => pool(convolveValid(img, k)));
    const l2 = [];
    for (let i = 0; i < 4; i++) l2.push(pool(convolveValid(l1[i], [[0,1,0],[1,-2,1],[0,1,0]])));
    for (let i = 0; i < 4; i++) {
      const mixed = l1[i].map((row, ri) => row.map((v, ci) => v + l1[(i+1)%4][ri][ci]));
      l2.push(pool(convolveValid(mixed, [[1,1,1],[1,-1,1],[1,1,1]])));
    }
    const ch = l1.map(sumMap);
    const logitsRaw = [
      170 - 2 * ch[3],                                // circle — the ring filter fires least on a filled disk
      0.5 * ch[0] + 0.5 * ch[1] - 0.35 * ch[2],       // cross  — strong horizontal + vertical edges, weak diagonal
      0.7 * ch[2] - 0.15 * ch[0] - 0.15 * ch[1],      // diag   — dominant diagonal channel
    ];
    // Scale before softmax to keep things tame
    const logits = logitsRaw.map((v) => v * 0.05);
    return { l1, l2, logits, probs: softmax(logits), ch };
  }

  // ───────── State + UI wiring ─────────
  const VIEWS = [
    { key: 'pipeline', label: 'Stage-by-stage pipeline' },
    { key: 'softmax',  label: 'Logits → softmax' },
  ];
  const CAPTIONS = {
    pipeline: 'Stage-by-stage walkthrough. Click any feature-map thumbnail to overlay its hot-region map back onto the original input — that\'s where the filter responded most. The flatten step animates the spatial cube collapsing into a long vector.',
    softmax: 'How three raw scores (logits) become three probabilities. Step 1: exponentiate each — eˣ blows up the largest score. Step 2: divide each by the sum. Step 3: a probability distribution that sums to 1.',
  };

  const state = {
    imageKind: 'circle',
    step: 5,
    playing: false,
    timer: null,
    view: 'pipeline',
    selectedFeature: null,    // {layer:1|2, idx:int}
    flattenAnim: 0,           // 0..1 animation progress for flatten
    flattenTimer: null,
  };

  const imagesEl = document.getElementById('images');
  const replayBtn = document.getElementById('replay-btn');
  const stepSlider = document.getElementById('step-slider');
  const stepLabel = document.getElementById('step-label');
  const captionEl = document.getElementById('caption');
  const probsEl = document.getElementById('probs');
  const tabsEl = document.getElementById('view-tabs');
  const archCanvas = document.getElementById('arch-canvas');
  const canvas = document.getElementById('canvas');

  ['circle', 'cross', 'diag'].forEach((k) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = k;
    b.dataset.image = k;
    b.addEventListener('click', () => {
      state.imageKind = k;
      state.step = 5;
      state.selectedFeature = null;
      render();
    });
    imagesEl.appendChild(b);
  });
  VIEWS.forEach((v) => {
    const b = document.createElement('button');
    b.textContent = v.label;
    b.dataset.view = v.key;
    b.addEventListener('click', () => { state.view = v.key; render(); });
    tabsEl.appendChild(b);
  });

  replayBtn.addEventListener('click', () => {
    state.step = 0; state.playing = true;
    state.selectedFeature = null;
    if (state.timer) clearInterval(state.timer);
    state.timer = setInterval(() => {
      if (state.step >= 5) { state.playing = false; clearInterval(state.timer); state.timer = null; return; }
      state.step++;
      // Trigger flatten animation when arriving at stage 3
      if (state.step === 3) animateFlatten();
      render();
    }, 600);
    render();
  });
  stepSlider.addEventListener('input', (e) => {
    state.playing = false;
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    const newStep = +e.target.value;
    if (newStep === 3 && state.step !== 3) animateFlatten();
    state.step = newStep;
    render();
  });

  function animateFlatten() {
    state.flattenAnim = 0;
    if (state.flattenTimer) clearInterval(state.flattenTimer);
    const tStart = performance.now();
    state.flattenTimer = setInterval(() => {
      const t = (performance.now() - tStart) / 600;
      state.flattenAnim = Math.min(1, t);
      if (state.flattenAnim >= 1) { clearInterval(state.flattenTimer); state.flattenTimer = null; }
      render();
    }, 30);
  }

  // ───────── Canvas (hit-test for feature maps) ─────────
  let hitRects = [];
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  canvas.addEventListener('click', (e) => {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    for (const h of hitRects) {
      if (x >= h.x && x <= h.x + h.w && y >= h.y && y <= h.y + h.h) {
        state.selectedFeature = h.feature;
        render();
        return;
      }
    }
    // Click outside → clear
    if (state.selectedFeature) { state.selectedFeature = null; render(); }
  });

  function draw(ctx, size) {
    if (state.view === 'softmax') drawSoftmax(ctx, size);
    else drawPipeline(ctx, size);
  }

  function drawMap(ctx, map, x0, y0, cell) {
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

  // ───────── View 1: Pipeline ─────────
  function drawPipeline(ctx, size) {
    hitRects = [];
    const img = genImage(state.imageKind);
    const out = pipeline(img);

    const stageX = [
      0,
      size.w * 0.20,
      size.w * 0.40,
      size.w * 0.60,
      size.w * 0.80,
    ];

    function label(text, x, y) {
      ctx.fillStyle = INK_FADE;
      ctx.font = '11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText(text, x, y);
    }

    // Stage 0: input. If a feature is selected, overlay the hot-region map.
    if (state.step >= 0) {
      const c = (size.w * 0.18) / 32;
      const x0 = stageX[0];
      const y0 = (size.h - c * 32) / 2;
      drawMap(ctx, img, x0, y0, c);
      // Hot-region overlay if selected
      if (state.selectedFeature) {
        const fmap = state.selectedFeature.layer === 1
          ? out.l1[state.selectedFeature.idx]
          : out.l2[state.selectedFeature.idx];
        // Upsample fmap (size H×W) onto 32×32 by nearest neighbor scaled by 32/H
        const FH = fmap.length, FW = fmap[0].length;
        const sx = 32 / FW, sy = 32 / FH;
        let mx = 0, hotI = 0, hotJ = 0;
        for (let i = 0; i < FH; i++) for (let j = 0; j < FW; j++) {
          if (fmap[i][j] > mx) { mx = fmap[i][j]; hotI = i; hotJ = j; }
        }
        ctx.save();
        for (let i = 0; i < FH; i++) {
          for (let j = 0; j < FW; j++) {
            const t = Math.max(0, Math.min(1, fmap[i][j] / Math.max(0.01, mx)));
            ctx.fillStyle = `rgba(122,31,36,${t * 0.55})`;
            ctx.fillRect(x0 + j * sx * c, y0 + i * sy * c, sx * c + 0.5, sy * c + 0.5);
          }
        }
        ctx.restore();
        // Border
        ctx.strokeStyle = ACCENT;
        ctx.lineWidth = 2;
        ctx.strokeRect(x0, y0, c * 32, c * 32);
        // Hot-spot crosshair on the upsampled location
        const hx = x0 + (hotJ + 0.5) * sx * c;
        const hy = y0 + (hotI + 0.5) * sy * c;
        ctx.strokeStyle = ACCENT;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(hx, hy, 6, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = ACCENT;
        ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
        ctx.textAlign = 'left';
        ctx.fillText(`hot regions · L${state.selectedFeature.layer}#${state.selectedFeature.idx}`, x0, y0 - 6);
        // Storytelling annotation under the input
        const filterNames = ['horizontal', 'vertical', 'diagonal', 'ring'];
        const fname = state.selectedFeature.layer === 1
          ? filterNames[state.selectedFeature.idx]
          : `deep #${state.selectedFeature.idx}`;
        const region = (hotI < FH / 2 ? 'upper' : 'lower') + '-' + (hotJ < FW / 2 ? 'left' : 'right');
        ctx.fillStyle = ACCENT;
        ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
        ctx.textAlign = 'left';
        ctx.fillText(`Filter ${state.selectedFeature.idx} (${fname}) fires strongest in the ${region}.`, x0, y0 + c * 32 + 22);
      }
      label('input · 32×32', x0 + (c * 32) / 2, size.h - 8);
    }

    // Stage 1: l1 (4 maps stacked)
    if (state.step >= 1) {
      const ch = out.l1; // 16×16
      const cell = (size.w * 0.16) / 16;
      const stackH = ch.length * (cell * 16 + 4);
      const yOff = (size.h - stackH) / 2;
      ch.forEach((m, i) => {
        const yy = yOff + i * (cell * 16 + 4);
        drawMap(ctx, m, stageX[1], yy, cell);
        // Highlight if selected
        if (state.selectedFeature && state.selectedFeature.layer === 1 && state.selectedFeature.idx === i) {
          ctx.strokeStyle = ACCENT;
          ctx.lineWidth = 2;
          ctx.strokeRect(stageX[1] - 2, yy - 2, cell * 16 + 4, cell * 16 + 4);
        }
        // Hit rect
        hitRects.push({ x: stageX[1], y: yy, w: cell * 16, h: cell * 16, feature: { layer: 1, idx: i } });
      });
      label('conv1+pool · 16×16×4', stageX[1] + (cell * 16) / 2, size.h - 8);
    }
    // Stage 2: l2
    if (state.step >= 2) {
      const ch = out.l2;
      const cell = (size.w * 0.14) / 8;
      const cols = 4, rows = 2;
      const tileW = cell * 8 + 4;
      const totalH = rows * tileW;
      const yOff = (size.h - totalH) / 2;
      ch.forEach((m, i) => {
        const col = i % cols, row = Math.floor(i / cols);
        const xx = stageX[2] + col * (cell * 8 / 4);
        const yy = yOff + row * tileW;
        drawMap(ctx, m, xx, yy, cell);
        if (state.selectedFeature && state.selectedFeature.layer === 2 && state.selectedFeature.idx === i) {
          ctx.strokeStyle = ACCENT;
          ctx.lineWidth = 2;
          ctx.strokeRect(xx - 1, yy - 1, cell * 8 + 2, cell * 8 + 2);
        }
        hitRects.push({ x: xx, y: yy, w: cell * 8, h: cell * 8, feature: { layer: 2, idx: i } });
      });
      label('conv2+pool · 8×8×8', stageX[2] + (cell * 8) / 2, size.h - 8);
    }
    // Stage 3: flatten — animate particles flying from spatial cube to a column
    if (state.step >= 3) {
      const flat = out.l2.flatMap(m => m.flat());
      const N = flat.length;
      const w = size.w * 0.05;
      const h = size.h * 0.7;
      const x0 = stageX[3], y0 = (size.h - h) / 2;
      const cellH = h / N;
      const max = Math.max(...flat.map(Math.abs)) + 1e-6;

      // Background column (faded if anim < 1)
      const a = state.flattenAnim;
      ctx.globalAlpha = a;
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
      ctx.globalAlpha = 1;

      // Particles flying from each l2 cell to its slot
      const ch = out.l2;
      const cell = (size.w * 0.14) / 8;
      const cols = 4, rows = 2;
      const tileW = cell * 8 + 4;
      const totalH2 = rows * tileW;
      const yOffl2 = (size.h - totalH2) / 2;
      const samples = Math.min(N, 64);
      for (let s = 0; s < samples; s++) {
        const idx = Math.floor((s / samples) * N);
        const ci = Math.floor(idx / 64);
        const local = idx % 64;
        const li = Math.floor(local / 8), lj = local % 8;
        const col = ci % cols, row = Math.floor(ci / cols);
        const sx = stageX[2] + col * (cell * 8 / 4) + lj * cell + cell / 2;
        const sy = yOffl2 + row * tileW + li * cell + cell / 2;
        const tx = x0 + w / 2;
        const ty = y0 + idx * cellH + cellH / 2;
        const px = sx + (tx - sx) * a;
        const py = sy + (ty - sy) * a;
        ctx.fillStyle = `rgba(122,31,36,${0.6 * (1 - a) + 0.1})`;
        ctx.beginPath();
        ctx.arc(px, py, 1.6, 0, Math.PI * 2);
        ctx.fill();
      }

      label('flatten · 512', x0 + w / 2, size.h - 8);
    }
    // Stage 4: dense (logits) — small bar diagram
    if (state.step >= 4) {
      const x0 = stageX[3] + size.w * 0.05 + 12;
      const w = size.w * 0.08;
      const h = 90;
      const y0 = (size.h - h) / 2;
      const labels = ['circle', 'cross', 'diag'];
      const logits = out.logits;
      // Scale for display
      const maxAbs = Math.max(0.01, ...logits.map(Math.abs));
      ctx.fillStyle = INK_FADE;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('logits', x0 + w / 2, y0 - 8);
      labels.forEach((lbl, i) => {
        const yi = y0 + (h / 3) * i;
        ctx.fillStyle = '#fffdf6';
        ctx.strokeStyle = 'rgba(38,35,32,0.20)';
        ctx.lineWidth = 1;
        ctx.fillRect(x0, yi + 4, w, h / 3 - 8);
        ctx.strokeRect(x0, yi + 4, w, h / 3 - 8);
        const v = logits[i];
        const fillW = (w / 2) * Math.min(1, Math.abs(v) / maxAbs);
        ctx.fillStyle = v >= 0 ? `rgba(122,31,36,0.65)` : `rgba(38,35,32,0.55)`;
        if (v >= 0) ctx.fillRect(x0 + w / 2, yi + 6, fillW, h / 3 - 12);
        else ctx.fillRect(x0 + w / 2 - fillW, yi + 6, fillW, h / 3 - 12);
        ctx.fillStyle = INK;
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'left';
        ctx.fillText(lbl, x0, yi + 2);
        ctx.textAlign = 'right';
        ctx.fillText(formatNum(v), x0 + w, yi + h / 3 - 4);
      });
    }

    // Stage 5: softmax → probability bars
    if (state.step >= 5) {
      const probs = out.probs;
      const labels = ['circle', 'cross', 'diag'];
      const x0 = stageX[4];
      const w = size.w * 0.16;
      const h = size.h * 0.6;
      const y0 = (size.h - h) / 2;
      const barH = (h - 30) / 3;
      const winner = probs.indexOf(Math.max(...probs));
      labels.forEach((lbl, i) => {
        const p = probs[i];
        const bw = w * p;
        ctx.fillStyle = i === winner ? ACCENT : 'rgba(38,35,32,0.35)';
        ctx.fillRect(x0, y0 + i * barH + 4, bw, barH - 12);
        ctx.fillStyle = INK_FADE;
        ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
        ctx.textAlign = 'left';
        ctx.fillText(`${lbl} · ${(p * 100).toFixed(0)}%`, x0, y0 + i * barH + barH - 4);
      });
      // Storytelling: name the winner explicitly with confidence.
      ctx.fillStyle = ACCENT;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'left';
      ctx.fillText(`predicted: ${labels[winner]} (${(probs[winner] * 100).toFixed(0)}%)`, x0, y0 - 8);
      label('softmax · 3 classes', x0 + w / 2, size.h - 8);
    }

    // Hint
    if (state.step >= 1 && !state.selectedFeature) {
      ctx.fillStyle = INK_FADE;
      ctx.font = 'italic 10px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'left';
      ctx.fillText('click any feature map to overlay its hot regions on the input', 0, 14);
    }
  }

  // ───────── View 2: Logits → softmax close-up ─────────
  function drawSoftmax(ctx, size) {
    const img = genImage(state.imageKind);
    const out = pipeline(img);
    const labels = ['circle', 'cross', 'diag'];
    const logits = out.logits;
    const exps = logits.map(Math.exp);
    const sumE = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map((v) => v / sumE);

    const margin = 40;
    const colW = (size.w - margin * 4) / 3;
    const y0 = 60;
    const yEnd = size.h - 60;
    const colH = yEnd - y0;

    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('FROM SCORES TO PROBABILITIES', margin, 22);

    // Three columns
    const cols = [
      { title: 'logits z',         note: 'raw scores · any real number',   values: logits, color: 'mono' },
      { title: 'exp(z)',           note: 'always positive · big z → much bigger e^z', values: exps, color: 'mono' },
      { title: 'p = exp(z) / Σ',   note: 'divide each by their sum · adds to 1',     values: probs,  color: 'accent' },
    ];

    const labMaxAbs = (vals) => Math.max(0.01, ...vals.map(Math.abs));
    const maxes = cols.map((c) => labMaxAbs(c.values));

    cols.forEach((col, ci) => {
      const x0 = margin + ci * (colW + margin);
      ctx.fillStyle = INK;
      ctx.font = 'italic 14px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText(col.title, x0 + colW / 2, y0 - 26);
      ctx.fillStyle = INK_FADE;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.fillText(col.note, x0 + colW / 2, y0 - 10);

      const barH = (colH - 16) / 3;
      const winner = probs.indexOf(Math.max(...probs));
      labels.forEach((lbl, i) => {
        const v = col.values[i];
        const yi = y0 + i * barH;
        ctx.fillStyle = '#fffdf6';
        ctx.strokeStyle = 'rgba(38,35,32,0.20)';
        ctx.lineWidth = 1;
        ctx.fillRect(x0, yi + 4, colW, barH - 12);
        ctx.strokeRect(x0, yi + 4, colW, barH - 12);
        const t = Math.min(1, Math.abs(v) / maxes[ci]);
        const useAccent = col.color === 'accent' || (col.color === 'mono' && v >= 0);
        if (col.title.startsWith('logits')) {
          // bipolar fill from center
          const fw = (colW / 2) * t;
          ctx.fillStyle = v >= 0 ? `rgba(122,31,36,${0.25 + t * 0.55})` : `rgba(38,35,32,${0.25 + t * 0.55})`;
          if (v >= 0) ctx.fillRect(x0 + colW / 2, yi + 6, fw, barH - 16);
          else ctx.fillRect(x0 + colW / 2 - fw, yi + 6, fw, barH - 16);
        } else {
          // left-anchored fill (positive only)
          const fw = colW * t;
          ctx.fillStyle = (col.title.startsWith('p') && i === winner)
            ? ACCENT
            : `rgba(122,31,36,${0.25 + t * 0.55})`;
          ctx.fillRect(x0 + 1, yi + 6, fw, barH - 16);
        }
        // Label and value
        ctx.fillStyle = INK;
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'left';
        ctx.fillText(lbl, x0 + 4, yi + 2);
        ctx.textAlign = 'right';
        const txt = col.title.startsWith('p')
          ? (v * 100).toFixed(1) + '%'
          : formatNum(v);
        ctx.fillText(txt, x0 + colW - 4, yi + barH - 6);
      });

      // Arrow to next column
      if (ci < cols.length - 1) {
        const ax = x0 + colW + 4;
        const bx = x0 + colW + margin - 4;
        const ay = y0 + colH / 2;
        ctx.strokeStyle = INK_FADE;
        ctx.fillStyle = INK_FADE;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, ay);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(bx, ay);
        ctx.lineTo(bx - 6, ay - 4);
        ctx.lineTo(bx - 6, ay + 4);
        ctx.closePath(); ctx.fill();
        ctx.fillStyle = INK_FADE;
        ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
        ctx.textAlign = 'center';
        ctx.fillText(ci === 0 ? 'eˣ' : '÷ Σ', (ax + bx) / 2, ay - 6);
      }
    });

    // Bottom: sum-check
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText(`sum of probabilities = ${probs.reduce((a,b)=>a+b,0).toFixed(2)}  ·  largest input wins`, size.w / 2, size.h - 14);
  }

  // ───────── Architecture canvas ─────────
  function drawArch() {
    const cv = archCanvas;
    const dpr = window.devicePixelRatio || 1;
    const W = cv.parentElement.getBoundingClientRect().width, H = 120;
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = W + 'px'; cv.style.height = H + 'px';
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const stages = [
      { lbl: 'Input',    shape: '32×32×1' },
      { lbl: 'Conv 3×3', shape: '32×32×4' },
      { lbl: 'Pool 2×2', shape: '16×16×4' },
      { lbl: 'Conv 3×3', shape: '16×16×8' },
      { lbl: 'Pool 2×2', shape: '8×8×8'   },
      { lbl: 'Flatten',  shape: '512'     },
      { lbl: 'Dense',    shape: '3'       },
      { lbl: 'Softmax',  shape: 'P'       },
    ];

    const margin = 18;
    const stepW = (W - margin * 2) / stages.length;
    const baseY = H * 0.50;

    stages.forEach((s, i) => {
      const cx = margin + (i + 0.5) * stepW;
      const boxW = Math.min(stepW - 6, 84);
      const boxH = 30;
      const isOutput = i === stages.length - 1;
      ctx.fillStyle = isOutput ? 'rgba(122,31,36,0.10)' : '#fffdf6';
      ctx.strokeStyle = INK;
      ctx.lineWidth = 1.2;
      ctx.fillRect(cx - boxW / 2, baseY - boxH / 2, boxW, boxH);
      ctx.strokeRect(cx - boxW / 2, baseY - boxH / 2, boxW, boxH);
      ctx.fillStyle = INK;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(s.lbl, cx, baseY);
      ctx.textBaseline = 'alphabetic';
      ctx.fillStyle = INK_FADE;
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.fillText(s.shape, cx, baseY + boxH / 2 + 12);

      if (i < stages.length - 1) {
        ctx.strokeStyle = INK_FADE;
        ctx.fillStyle = INK_FADE;
        ctx.lineWidth = 1;
        const ax = cx + boxW / 2 + 1;
        const bx = margin + (i + 1.5) * stepW - boxW / 2 - 1;
        ctx.beginPath();
        ctx.moveTo(ax, baseY);
        ctx.lineTo(bx, baseY);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(bx, baseY);
        ctx.lineTo(bx - 5, baseY - 3);
        ctx.lineTo(bx - 5, baseY + 3);
        ctx.closePath(); ctx.fill();
      }

      // Highlight the current pipeline stage if step view
      if (state.view === 'pipeline') {
        const stageMap = [0, 1, 2, 3, 4, 5, 6, 7]; // step 0-5 maps roughly here
        // Step 0 → input(0); 1 → conv1+pool(2); 2 → conv2+pool(4); 3 → flatten(5); 4 → dense(6); 5 → softmax(7)
        const stepIdx = [0, 2, 4, 5, 6, 7][state.step];
        if (i === stepIdx) {
          ctx.strokeStyle = ACCENT;
          ctx.lineWidth = 2;
          ctx.strokeRect(cx - boxW / 2 - 2, baseY - boxH / 2 - 2, boxW + 4, boxH + 4);
        }
      }
    });

    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('PIPELINE', margin, 14);
  }
  new ResizeObserver(drawArch).observe(archCanvas.parentElement);

  function captionFor(step) {
    return [
      'Stage 0 — input image. 32×32 grayscale pixels.',
      'Stage 1 — conv layer 1 produces 4 feature maps. Each filter responds to a different orientation of edge.',
      'Stage 2 — conv layer 2 mixes the previous channels into 8 deeper features. Spatial size halves to 8×8.',
      'Stage 3 — flatten. 8×8×8 = 512 numbers in one long vector. (Watch the particles fly into the column.)',
      'Stage 4 — dense layer turns the 512-vector into 3 logits, one raw score per class.',
      'Stage 5 — softmax. Exponentiate, normalize, get probabilities. Largest probability is the prediction.',
    ][step];
  }

  function render() {
    Array.from(imagesEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.image === state.imageKind);
    });
    Array.from(tabsEl.children).forEach((btn) => {
      btn.classList.toggle('on', btn.dataset.view === state.view);
    });
    stepSlider.value = state.step;
    stepLabel.textContent = `${state.step}/5`;
    if (state.view === 'softmax') {
      captionEl.innerHTML = `<span class="ml-cap-num">·</span> ${CAPTIONS.softmax}`;
    } else {
      captionEl.innerHTML = `<span class="ml-cap-num">·</span> ${captionFor(state.step)}`;
    }

    const img = genImage(state.imageKind);
    const out = pipeline(img);
    const probs = out.probs;
    const labels = ['circle', 'cross', 'diag'];
    const winner = probs.indexOf(Math.max(...probs));
    let html = '';
    labels.forEach((lbl, i) => {
      html += `<div class="row"><span>P(${lbl})</span><b style="color:${i === winner ? 'var(--accent)' : 'var(--ink)'}">${(probs[i] * 100).toFixed(1)}%</b></div>`;
    });
    probsEl.innerHTML = html;

    canvas.style.cursor = 'pointer';
    canvasCtl.redraw();
    drawArch();
  }

  render();
})();
