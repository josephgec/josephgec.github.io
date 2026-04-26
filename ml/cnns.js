/* Vol. XV — Convolutional Networks */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, INK_FADE } = V;

  const FILTERS = {
    identity: { name: 'Identity',         k: [[0,0,0],[0,1,0],[0,0,0]],          blurb: 'Passes the image through unchanged.' },
    edge_h:   { name: 'Horizontal edges', k: [[-1,-1,-1],[0,0,0],[1,1,1]],       blurb: 'Detects horizontal transitions — bright above, dark below.' },
    edge_v:   { name: 'Vertical edges',   k: [[-1,0,1],[-1,0,1],[-1,0,1]],       blurb: 'Vertical-edge detector. Together with the horizontal version, the building blocks of edge detection.' },
    sobel_x:  { name: 'Sobel X',          k: [[-1,0,1],[-2,0,2],[-1,0,1]],       blurb: 'A weighted vertical edge detector — more emphasis on the central row. Classic computer vision.' },
    laplacian:{ name: 'Laplacian',        k: [[0,1,0],[1,-4,1],[0,1,0]],          blurb: 'Second derivative — fires anywhere intensity changes, regardless of direction.' },
    blur:     { name: 'Box blur',         k: [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]], blurb: 'Average of the 3×3 neighborhood. Smooths noise; loses detail.' },
    sharpen:  { name: 'Sharpen',          k: [[0,-1,0],[-1,5,-1],[0,-1,0]],      blurb: 'Emphasizes the center pixel against its neighbors. Increases local contrast.' },
    emboss:   { name: 'Emboss',           k: [[-2,-1,0],[-1,1,1],[0,1,2]],        blurb: 'A 3D-relief feel — directional difference.' },
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

  const state = {
    imageKind: 'face',
    filterKey: 'edge_v',
  };

  const imagesEl = document.getElementById('images');
  const filtersEl = document.getElementById('filters');
  const filterBlurbEl = document.getElementById('filter-blurb');
  const kernelGridEl = document.getElementById('kernel-grid');
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

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    const img = genImage(state.imageKind, 32);
    const filter = FILTERS[state.filterKey];
    const conv = convolve(img, filter.k);
    const post = reluMap(conv);
    const pool = maxpool(post, 2);

    // Three panels: input | conv+ReLU | pooled
    const gap = 24;
    const panelW = (size.w - gap * 2) / 3;
    const cellSize = Math.min(panelW / 32, size.h / 32);
    const inputW = cellSize * 32;
    const poolCell = Math.min(panelW / 16, size.h / 16);

    function drawMap(map, x0, y0, cell, color = (v, max) => {
      // Linear cream→ink mapping. v expected 0..1
      const t = Math.max(0, Math.min(1, v / Math.max(0.01, max)));
      const r = Math.round(247 - t * 200);
      const g = Math.round(244 - t * 200);
      const b = Math.round(236 - t * 220);
      return `rgb(${r},${g},${b})`;
    }) {
      const H = map.length, W = map[0].length;
      let max = 0;
      for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) if (map[i][j] > max) max = map[i][j];
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < W; j++) {
          ctx.fillStyle = color(map[i][j], max);
          ctx.fillRect(x0 + j * cell, y0 + i * cell, cell + 0.5, cell + 0.5);
        }
      }
      ctx.strokeStyle = 'rgba(38,35,32,0.30)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x0, y0, W * cell, H * cell);
    }

    const yOff = (size.h - cellSize * 32) / 2;
    drawMap(img, 0, yOff, cellSize);

    // Panel labels
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('input', inputW / 2, yOff + cellSize * 32 + 18);

    // Convolution output: bipolar mapping (positive = oxblood, negative = ink). Use ReLU result so negatives are 0.
    const x1 = panelW + gap;
    drawMap(post, x1, yOff, cellSize, (v, max) => {
      const t = Math.max(0, Math.min(1, v / Math.max(0.01, max)));
      const r = Math.round(247 - t * 130);
      const g = Math.round(244 - t * 195);
      const b = Math.round(236 - t * 195);
      return `rgb(${r},${g},${b})`;
    });
    ctx.fillStyle = INK_FADE;
    ctx.fillText('conv + ReLU', x1 + inputW / 2, yOff + cellSize * 32 + 18);

    // Pooled output
    const x2 = (panelW + gap) * 2;
    const poolH = pool.length;
    const poolDisplayCell = cellSize * 2;
    const poolYOff = (size.h - poolDisplayCell * poolH) / 2;
    drawMap(pool, x2, poolYOff, poolDisplayCell, (v, max) => {
      const t = Math.max(0, Math.min(1, v / Math.max(0.01, max)));
      const r = Math.round(247 - t * 130);
      const g = Math.round(244 - t * 195);
      const b = Math.round(236 - t * 195);
      return `rgb(${r},${g},${b})`;
    });
    ctx.fillStyle = INK_FADE;
    ctx.fillText('2×2 max-pool', x2 + (poolDisplayCell * poolH) / 2, poolYOff + poolDisplayCell * poolH + 18);
  }

  function render() {
    Array.from(imagesEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.image === state.imageKind);
    });
    Array.from(filtersEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.filter === state.filterKey);
    });
    const f = FILTERS[state.filterKey];
    filterBlurbEl.textContent = f.blurb;
    kernelGridEl.innerHTML = '';
    f.k.forEach((row) => {
      row.forEach((v) => {
        const cell = document.createElement('div');
        cell.style.cssText = 'background:var(--paper); border:1px solid var(--rule); padding:6px; text-align:center';
        cell.textContent = formatNum(v);
        kernelGridEl.appendChild(cell);
      });
    });
    canvasCtl.redraw();
  }

  render();
})();
