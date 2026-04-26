/* Vol. XV — Convolutional Networks */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK, INK_FADE } = V;

  const FILTERS = {
    identity: { name: 'Identity',         k: [[0,0,0],[0,1,0],[0,0,0]],          blurb: 'Passes the image through unchanged.', meaning: 'No change. Useful as a baseline.' },
    edge_h:   { name: 'Horizontal edges', k: [[-1,-1,-1],[0,0,0],[1,1,1]],       blurb: 'Detects horizontal transitions — bright above, dark below.', meaning: 'Difference along y: top row negative, bottom row positive → fires at horizontal edges (top-to-bottom intensity change).' },
    edge_v:   { name: 'Vertical edges',   k: [[-1,0,1],[-1,0,1],[-1,0,1]],       blurb: 'Vertical-edge detector. Together with the horizontal version, the building blocks of edge detection.', meaning: 'Difference along x: left column negative, right column positive → fires at vertical edges.' },
    sobel_x:  { name: 'Sobel X',          k: [[-1,0,1],[-2,0,2],[-1,0,1]],       blurb: 'A weighted vertical edge detector — more emphasis on the central row. Classic computer vision.', meaning: 'A weighted x-derivative. Center row counts double — robust vertical-edge detector.' },
    laplacian:{ name: 'Laplacian',        k: [[0,1,0],[1,-4,1],[0,1,0]],          blurb: 'Second derivative — fires anywhere intensity changes, regardless of direction.', meaning: 'Approximates the 2D Laplacian (∂²/∂x² + ∂²/∂y²). Fires at any intensity change — direction-agnostic.' },
    blur:     { name: 'Box blur',         k: [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]], blurb: 'Average of the 3×3 neighborhood. Smooths noise; loses detail.', meaning: 'Plain mean over the 3×3 window. Smooths noise at the cost of sharpness.' },
    sharpen:  { name: 'Sharpen',          k: [[0,-1,0],[-1,5,-1],[0,-1,0]],      blurb: 'Emphasizes the center pixel against its neighbors. Increases local contrast.', meaning: 'Boosts the center, subtracts neighbors. Equivalent to image + (image − blurred): pushes contrast up.' },
    emboss:   { name: 'Emboss',           k: [[-2,-1,0],[-1,1,1],[0,1,2]],        blurb: 'A 3D-relief feel — directional difference.', meaning: 'Diagonal directional difference — produces a faux-3D shaded look from gradients along the main diagonal.' },
  };

  const IMAGES = ['face', 'cross', 'circle', 'diag', 'grid'];

  function genImage(kind, size = 32) {
    const img = [];
    for (let i = 0; i < size; i++) {
      const row = [];
      for (let j = 0; j < size; j++) {
        let v = 0.05;
        if (kind === 'cross') {
          v = (Math.abs(i - size/2) < 2 || Math.abs(j - size/2) < 2) ? 1 : 0.05;
        } else if (kind === 'circle') {
          const dx = j - size/2, dy = i - size/2;
          const r = Math.hypot(dx, dy);
          v = r < size*0.3 ? 1 : (r < size*0.4 ? 0.5 : 0.05);
        } else if (kind === 'diag') {
          v = (j > i - 4 && j < i + 4) ? 1 : 0.05;
        } else if (kind === 'face') {
          const dx = j - size/2, dy = i - size/2;
          const inHead = Math.hypot(dx, dy) < size * 0.4;
          const eyeL = Math.hypot(j - size*0.38, i - size*0.42) < 1.5;
          const eyeR = Math.hypot(j - size*0.62, i - size*0.42) < 1.5;
          const mouth = (i > size*0.6 && i < size*0.65 && Math.abs(j - size/2) < size*0.12);
          if (eyeL || eyeR || mouth) v = 0;
          else if (inHead) v = 0.85;
          else v = 0.1;
        } else if (kind === 'grid') {
          v = (i % 5 === 0 || j % 5 === 0) ? 0.85 : 0.1;
        }
        row.push(v);
      }
      img.push(row);
    }
    return img;
  }

  function convolve(img, kernel) {
    const H = img.length, W = img[0].length;
    const KH = kernel.length, KW = kernel[0].length;
    const py = Math.floor(KH / 2), px = Math.floor(KW / 2);
    const out = [];
    for (let i = 0; i < H; i++) {
      const row = [];
      for (let j = 0; j < W; j++) {
        let s = 0;
        for (let ki = 0; ki < KH; ki++) {
          for (let kj = 0; kj < KW; kj++) {
            const ii = i + ki - py, jj = j + kj - px;
            if (ii >= 0 && ii < H && jj >= 0 && jj < W) s += img[ii][jj] * kernel[ki][kj];
          }
        }
        row.push(s);
      }
      out.push(row);
    }
    return out;
  }
  const reluMap = (m) => m.map((r) => r.map((v) => Math.max(0, v)));
  function maxpool(map, sz = 2) {
    const H = map.length, W = map[0].length;
    const out = [];
    for (let i = 0; i < H; i += sz) {
      const row = [];
      for (let j = 0; j < W; j += sz) {
        let best = -Infinity;
        for (let ki = 0; ki < sz; ki++) for (let kj = 0; kj < sz; kj++) {
          const ii = i + ki, jj = j + kj;
          if (ii < H && jj < W) best = Math.max(best, map[ii][jj]);
        }
        row.push(best);
      }
      out.push(row);
    }
    return out;
  }

  const VIEWS = [
    { key: 'slide',    label: 'Slide one kernel' },
    { key: 'fourmaps', label: 'Four kernels · feature maps' },
  ];
  const CAPTIONS = {
    slide: 'The yellow box is the kernel\'s current 3×3 window. The lit cell on the conv+ReLU map is the output value computed from that window. Drag the slider — or hit auto-slide — to scan the kernel across the whole image.',
    fourmaps: 'Four different kernels run on the same input — each produces its own feature map. Real CNNs have dozens of these per layer, learned from data. Each row: one kernel, one feature map, one pooled summary.',
  };

  const state = {
    imageKind: 'face',
    filterKey: 'edge_h',
    view: 'slide',
    // Position the kernel right over the eye row of the face — a dramatic firing
    // location that makes the edge detector's purpose obvious on first paint.
    pos: 405,
    playing: false,
    timer: null,
  };

  // One-line annotation tied to each filter — anchors math to picture.
  const FILTER_STORY = {
    identity:  'K = δ(0,0): output = input.',
    edge_h:    'This kernel approximates ∂I/∂y — a horizontal-edge detector.',
    edge_v:    'This kernel approximates ∂I/∂x — a vertical-edge detector.',
    sobel_x:   'Sobel-X: a noise-robust ∂I/∂x with a wider central-row weight.',
    laplacian: 'Laplacian ≈ ∂²I/∂x² + ∂²I/∂y² — fires at any intensity change.',
    blur:      '⅑·Σ over a 3×3 patch — local average smooths high-frequency detail.',
    sharpen:   'Center − neighbours: amplifies what the blur removed.',
    emboss:    'Diagonal directional difference: gives a 3D-relief shading.',
  };

  const imagesEl = document.getElementById('images');
  const filtersEl = document.getElementById('filters');
  const filterBlurbEl = document.getElementById('filter-blurb');
  const kernelGridEl = document.getElementById('kernel-grid');
  const kernelMeaningEl = document.getElementById('kernel-meaning');
  const tabsEl = document.getElementById('view-tabs');
  const captionEl = document.getElementById('caption-text');
  const posSlider = document.getElementById('pos-slider');
  const posLabel = document.getElementById('pos-label');
  const playBtn = document.getElementById('play-btn');
  const archCanvas = document.getElementById('arch-canvas');
  const canvas = document.getElementById('canvas');

  IMAGES.forEach((k) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = k;
    b.dataset.image = k;
    b.addEventListener('click', () => { state.imageKind = k; render(); });
    imagesEl.appendChild(b);
  });
  Object.entries(FILTERS).forEach(([k, f]) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = f.name;
    b.dataset.filter = k;
    b.addEventListener('click', () => { state.filterKey = k; render(); });
    filtersEl.appendChild(b);
  });
  VIEWS.forEach((v) => {
    const b = document.createElement('button');
    b.textContent = v.label;
    b.dataset.view = v.key;
    b.addEventListener('click', () => { state.view = v.key; render(); });
    tabsEl.appendChild(b);
  });
  posSlider.addEventListener('input', (e) => {
    state.pos = +e.target.value;
    stopPlay();
    render();
  });
  playBtn.addEventListener('click', () => {
    if (state.playing) { stopPlay(); render(); return; }
    state.playing = true;
    playBtn.textContent = '⏸ Pause';
    state.timer = setInterval(() => {
      state.pos = (state.pos + 12) % 900;
      posSlider.value = state.pos;
      render();
    }, 80);
  });
  function stopPlay() {
    state.playing = false;
    playBtn.textContent = '▷ Auto-slide';
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    if (state.view === 'fourmaps') drawFourMaps(ctx, size);
    else drawSlide(ctx, size);
  }

  function colorMono(t) {
    const r = Math.round(247 - t * 200);
    const g = Math.round(244 - t * 200);
    const b = Math.round(236 - t * 220);
    return `rgb(${r},${g},${b})`;
  }
  function colorOxblood(t) {
    const r = Math.round(247 - t * 130);
    const g = Math.round(244 - t * 195);
    const b = Math.round(236 - t * 195);
    return `rgb(${r},${g},${b})`;
  }
  function drawMap(ctx, map, x0, y0, cell, colorFn = colorMono) {
    const H = map.length, W = map[0].length;
    let max = 0;
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) if (Math.abs(map[i][j]) > max) max = Math.abs(map[i][j]);
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) {
        const t = Math.max(0, Math.min(1, Math.abs(map[i][j]) / Math.max(0.01, max)));
        ctx.fillStyle = colorFn(t);
        ctx.fillRect(x0 + j * cell, y0 + i * cell, cell + 0.5, cell + 0.5);
      }
    }
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x0, y0, W * cell, H * cell);
  }

  // ───────── View 1: Slide one kernel ─────────
  function drawSlide(ctx, size) {
    const img = genImage(state.imageKind, 32);
    const filter = FILTERS[state.filterKey];
    const conv = convolve(img, filter.k);
    const post = reluMap(conv);
    const pool = maxpool(post, 2);

    const gap = 24;
    const panelW = (size.w - gap * 2) / 3;
    const cellSize = Math.min(panelW / 32, size.h / 32);
    const inputW = cellSize * 32;

    const yOff = (size.h - cellSize * 32) / 2;

    // Sliding window position on a 30×30 valid grid (interior of 32×32 with 3×3 kernel)
    const total = 30 * 30;
    const idx = Math.min(total - 1, Math.floor((state.pos / 900) * total));
    const cy = Math.floor(idx / 30) + 1;  // 1..30
    const cx = (idx % 30) + 1;

    // Panel 1: input + sliding kernel box
    drawMap(ctx, img, 0, yOff, cellSize, colorMono);
    // 3×3 highlight
    ctx.strokeStyle = '#c79a2b';
    ctx.lineWidth = 2;
    ctx.strokeRect((cx - 1) * cellSize, yOff + (cy - 1) * cellSize, 3 * cellSize, 3 * cellSize);
    // dim everything else slightly using a darken pass on the input — actually just leave; box is enough.

    ctx.fillStyle = INK_FADE;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('input · 32×32', inputW / 2, yOff + cellSize * 32 + 18);
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.fillText(`window @ (${cx}, ${cy})`, inputW / 2, yOff + cellSize * 32 + 32);

    // Panel 2: conv+ReLU map, with the just-computed cell highlighted
    const x1 = panelW + gap;
    drawMap(ctx, post, x1, yOff, cellSize, colorOxblood);
    // Highlight cell at (cx, cy) of the conv output
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1 + cx * cellSize, yOff + cy * cellSize, cellSize, cellSize);
    ctx.fillStyle = INK_FADE;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('conv + ReLU · 32×32', x1 + inputW / 2, yOff + cellSize * 32 + 18);
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    const v = post[cy][cx];
    ctx.fillText(`out[${cx},${cy}] = ${formatNum(v)}`, x1 + inputW / 2, yOff + cellSize * 32 + 32);

    // Panel 3: pool
    const x2 = (panelW + gap) * 2;
    const poolH = pool.length;
    const poolDisplayCell = cellSize * 2;
    const poolYOff = (size.h - poolDisplayCell * poolH) / 2;
    drawMap(ctx, pool, x2, poolYOff, poolDisplayCell, colorOxblood);
    ctx.fillStyle = INK_FADE;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('2×2 max-pool · 16×16', x2 + (poolDisplayCell * poolH) / 2, poolYOff + poolDisplayCell * poolH + 18);

    // Connecting hint arrow
    ctx.strokeStyle = 'rgba(38,35,32,0.35)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(cellSize * (cx + 1), yOff + cellSize * (cy - 1) + cellSize * 1.5);
    ctx.lineTo(x1 + cellSize * cx, yOff + cellSize * cy + cellSize * 0.5);
    ctx.stroke();
    ctx.setLineDash([]);

    // On-canvas storytelling: anchor the math to the picture.
    const story = FILTER_STORY[state.filterKey];
    if (story) {
      ctx.fillStyle = ACCENT;
      ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText(story, size.w / 2, 18);
    }
    // Affordance hint: encourage the slider/auto-slide.
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('drag the slider or hit ▷ Auto-slide to scan the kernel', 0, size.h - 4);
  }

  // ───────── View 2: Four kernels feature maps ─────────
  function drawFourMaps(ctx, size) {
    const img = genImage(state.imageKind, 32);
    const set = ['edge_h', 'edge_v', 'sobel_x', 'laplacian'];
    const rows = set.length;
    const cellSize = Math.min((size.w * 0.18) / 32, (size.h - 30) / rows / 32);
    const rowH = cellSize * 32 + 8;
    const startY = (size.h - rowH * rows) / 2;

    set.forEach((fk, ri) => {
      const F = FILTERS[fk];
      const conv = convolve(img, F.k);
      const post = reluMap(conv);
      const pool = maxpool(post, 2);
      const y = startY + ri * rowH;

      // Kernel preview
      const kxc = 0;
      const ksz = Math.min(rowH * 0.7, 60) / 3;
      const kx0 = kxc + 4;
      const ky0 = y + (rowH - ksz * 3) / 2;
      F.k.forEach((row, i) => row.forEach((v, j) => {
        const t = Math.min(1, Math.abs(v) / 2);
        ctx.fillStyle = v >= 0 ? `rgba(122,31,36,${0.15 + t * 0.7})` : `rgba(38,35,32,${0.15 + t * 0.7})`;
        ctx.fillRect(kx0 + j * ksz, ky0 + i * ksz, ksz, ksz);
        ctx.strokeStyle = 'rgba(38,35,32,0.20)';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(kx0 + j * ksz, ky0 + i * ksz, ksz, ksz);
      }));
      ctx.fillStyle = INK_FADE;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      ctx.fillText(F.name, kx0, ky0 - 4);

      // Input thumbnail
      const inX = kx0 + ksz * 3 + 24;
      drawMap(ctx, img, inX, y, cellSize, colorMono);

      // Conv map
      const cmX = inX + cellSize * 32 + 18;
      drawMap(ctx, post, cmX, y, cellSize, colorOxblood);

      // Pooled
      const pX = cmX + cellSize * 32 + 18;
      const pCell = cellSize * 2;
      drawMap(ctx, pool, pX, y + (cellSize * 32 - pCell * 16) / 2, pCell, colorOxblood);

      // Arrow indicators
      ctx.fillStyle = INK_FADE;
      ctx.font = 'italic 10px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      if (ri === 0) {
        ctx.fillText('kernel', kx0 + ksz * 1.5, y - 14);
        ctx.fillText('input', inX + cellSize * 16, y - 14);
        ctx.fillText('conv + ReLU', cmX + cellSize * 16, y - 14);
        ctx.fillText('pool', pX + pCell * 8, y - 14);
      }
    });
  }

  // ───────── Architecture schematic (separate canvas below) ─────────
  function drawArch() {
    const cv = archCanvas;
    const dpr = window.devicePixelRatio || 1;
    const W = cv.parentElement.getBoundingClientRect().width, H = 140;
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = W + 'px'; cv.style.height = H + 'px';
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const stages = [
      { label: 'Input', shape: '32×32×1', kind: 'image' },
      { label: 'Conv 3×3', shape: '32×32×8', kind: 'conv' },
      { label: 'ReLU', shape: '·', kind: 'op' },
      { label: 'Pool 2×2', shape: '16×16×8', kind: 'pool' },
      { label: 'Conv 3×3', shape: '16×16×16', kind: 'conv' },
      { label: 'ReLU', shape: '·', kind: 'op' },
      { label: 'Pool 2×2', shape: '8×8×16', kind: 'pool' },
      { label: 'Flatten', shape: '1024', kind: 'flat' },
      { label: 'Dense', shape: '10', kind: 'dense' },
      { label: 'Softmax', shape: 'P', kind: 'soft' },
    ];

    const margin = 20;
    const totalW = W - margin * 2;
    const stepW = totalW / stages.length;
    const baseY = H * 0.55;

    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('TYPICAL CNN PIPELINE · input → [conv → activation → pool] × N → flatten → dense → output', margin, 16);

    stages.forEach((s, i) => {
      const cx = margin + (i + 0.5) * stepW;
      let boxW = 64, boxH = 36;
      let fill = '#fffdf6';
      if (s.kind === 'op') { boxW = 38; boxH = 24; fill = 'rgba(38,35,32,0.05)'; }
      if (s.kind === 'soft' || s.kind === 'dense') fill = 'rgba(122,31,36,0.10)';

      ctx.fillStyle = fill;
      ctx.strokeStyle = INK;
      ctx.lineWidth = 1.2;
      ctx.fillRect(cx - boxW / 2, baseY - boxH / 2, boxW, boxH);
      ctx.strokeRect(cx - boxW / 2, baseY - boxH / 2, boxW, boxH);
      ctx.fillStyle = INK;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(s.label, cx, baseY);
      ctx.textBaseline = 'alphabetic';
      // Shape annotation below
      ctx.fillStyle = INK_FADE;
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.fillText(s.shape, cx, baseY + boxH / 2 + 14);

      // Arrow
      if (i < stages.length - 1) {
        const ax = cx + boxW / 2;
        const bx = margin + (i + 1.5) * stepW - 64 / 2;
        ctx.strokeStyle = INK_FADE;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(ax + 1, baseY);
        ctx.lineTo(bx - 1, baseY);
        ctx.stroke();
      }
    });

    // Loop bracket over the [conv → activation → pool] block
    const startX = margin + 1.5 * stepW - 32;
    const endX = margin + 6.5 * stepW + 32;
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(startX, baseY - 38);
    ctx.lineTo(startX, baseY - 46);
    ctx.lineTo(endX, baseY - 46);
    ctx.lineTo(endX, baseY - 38);
    ctx.stroke();
    ctx.fillStyle = ACCENT;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('repeated block', (startX + endX) / 2, baseY - 50);
  }
  new ResizeObserver(drawArch).observe(archCanvas.parentElement);

  function render() {
    Array.from(imagesEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.image === state.imageKind);
    });
    Array.from(filtersEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.filter === state.filterKey);
    });
    Array.from(tabsEl.children).forEach((btn) => {
      btn.classList.toggle('on', btn.dataset.view === state.view);
    });
    if (captionEl) captionEl.textContent = CAPTIONS[state.view];

    const f = FILTERS[state.filterKey];
    filterBlurbEl.textContent = f.blurb;
    kernelMeaningEl.textContent = f.meaning;
    kernelGridEl.innerHTML = '';
    let kmax = 0;
    f.k.forEach((row) => row.forEach((v) => { if (Math.abs(v) > kmax) kmax = Math.abs(v); }));
    f.k.forEach((row) => {
      row.forEach((v) => {
        const cell = document.createElement('div');
        const t = kmax > 0 ? Math.min(1, Math.abs(v) / kmax) : 0;
        const bg = v > 0
          ? `rgba(122,31,36,${0.10 + t * 0.45})`
          : (v < 0 ? `rgba(38,35,32,${0.10 + t * 0.40})` : 'var(--paper)');
        cell.style.cssText = `background:${bg}; border:1px solid var(--rule); padding:6px; text-align:center`;
        cell.textContent = formatNum(v);
        kernelGridEl.appendChild(cell);
      });
    });

    // Sliding window position label
    const idx = Math.min(30 * 30 - 1, Math.floor((state.pos / 900) * 30 * 30));
    const cy = Math.floor(idx / 30) + 1;
    const cx = (idx % 30) + 1;
    posLabel.textContent = `window @ (${cx}, ${cy})`;
    posSlider.value = state.pos;

    canvasCtl.redraw();
    drawArch();
  }

  render();
})();
