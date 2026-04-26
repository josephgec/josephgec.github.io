/* Vol. XXV — A Geometry of Meaning (Embeddings) */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK_FADE } = V;

  // Hand-tuned 2D embeddings. Designed so analogies look correct as vector arrows.
  const EMB = {
    // Axis 1 (right): formal/regal, gendered male
    // Axis 2 (up): female direction (pulled away from male axis)
    king:  [1.0,  0.5],
    queen: [1.0,  1.5],
    man:   [0.4,  0.5],
    woman: [0.4,  1.5],

    Paris:  [-1.4, -0.6],
    France: [-1.0, -1.4],
    Rome:   [-0.4, -0.6],
    Italy:  [0.0,  -1.4],
    Berlin: [-0.7,  0.4],
    Germany:[-0.3, -0.4],

    cat:    [-1.5,  1.6],
    dog:    [-1.2,  1.7],
    pet:    [-1.0,  1.4],
    animal: [-0.8,  1.2],

    car:    [1.5, -1.6],
    truck:  [1.7, -1.8],
    vehicle:[1.3, -1.3],
    bicycle:[1.1, -1.0],

    walk:   [-0.4, -2.0],
    walked: [-0.6, -2.2],
    run:    [-0.1, -2.0],
    ran:    [-0.3, -2.2],

    big:    [2.0,  0.0],
    bigger: [2.4, -0.3],
    small:  [-2.0, 0.0],
    smaller:[-2.4,-0.3],
  };

  const ANALOGIES = {
    'king − man + woman = ?':         { a: 'king',   b: 'man',    c: 'woman',  expect: 'queen' },
    'Paris − France + Italy = ?':     { a: 'Paris',  b: 'France', c: 'Italy',  expect: 'Rome' },
    'walk − walked + ran = ?':        { a: 'walk',   b: 'walked', c: 'ran',    expect: 'run' },
    'big − bigger + smaller = ?':     { a: 'big',    b: 'bigger', c: 'smaller',expect: 'small' },
  };

  const state = {
    analogyKey: 'king − man + woman = ?',
    selected: null,
  };

  const analogiesEl = document.getElementById('analogies');
  const hoverInfo = document.getElementById('hover-info');
  const readoutEl = document.getElementById('readout');
  const captionEl = document.getElementById('caption');
  const canvas = document.getElementById('canvas');

  Object.keys(ANALOGIES).forEach((k) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = k;
    b.dataset.an = k;
    b.addEventListener('click', () => { state.analogyKey = k; render(); });
    analogiesEl.appendChild(b);
  });

  function cosine(a, b) {
    const dot = a[0]*b[0] + a[1]*b[1];
    const na = Math.hypot(...a), nb = Math.hypot(...b);
    return dot / (na * nb + 1e-9);
  }
  function neighbors(word, n = 5) {
    const v = EMB[word]; if (!v) return [];
    const results = [];
    for (const [w, vec] of Object.entries(EMB)) {
      if (w === word) continue;
      results.push({ word: w, sim: cosine(v, vec) });
    }
    results.sort((a, b) => b.sim - a.sim);
    return results.slice(0, n);
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: true });

  let wordRects = [];
  canvas.addEventListener('mousemove', (e) => {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    let hit = null;
    for (const wr of wordRects) {
      if (x >= wr.x && x <= wr.x + wr.w && y >= wr.y && y <= wr.y + wr.h) { hit = wr.word; break; }
    }
    if (hit !== state.selected) { state.selected = hit; render(); }
  });
  canvas.addEventListener('mouseleave', () => { state.selected = null; render(); });

  function draw(ctx, size) {
    const R = 3;
    const tx = (x) => (x + R) / (2 * R) * size.w;
    const ty = (y) => size.h - (y + R) / (2 * R) * size.h;

    // Light grid
    ctx.strokeStyle = 'rgba(38,35,32,0.06)';
    ctx.lineWidth = 1;
    for (let i = -3; i <= 3; i++) {
      ctx.beginPath(); ctx.moveTo(tx(i), 0); ctx.lineTo(tx(i), size.h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, ty(i)); ctx.lineTo(size.w, ty(i)); ctx.stroke();
    }
    // Axes
    ctx.strokeStyle = 'rgba(38,35,32,0.18)';
    ctx.beginPath();
    ctx.moveTo(0, ty(0)); ctx.lineTo(size.w, ty(0)); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(tx(0), 0); ctx.lineTo(tx(0), size.h); ctx.stroke();

    // Analogy arrows
    const an = ANALOGIES[state.analogyKey];
    if (an && EMB[an.a] && EMB[an.b] && EMB[an.c] && EMB[an.expect]) {
      const arrow = (from, to, color) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(tx(from[0]), ty(from[1]));
        ctx.lineTo(tx(to[0]), ty(to[1]));
        ctx.stroke();
        const ang = Math.atan2(ty(to[1]) - ty(from[1]), tx(to[0]) - tx(from[0]));
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(tx(to[0]), ty(to[1]));
        ctx.lineTo(tx(to[0]) - 8 * Math.cos(ang - 0.4), ty(to[1]) - 8 * Math.sin(ang - 0.4));
        ctx.lineTo(tx(to[0]) - 8 * Math.cos(ang + 0.4), ty(to[1]) - 8 * Math.sin(ang + 0.4));
        ctx.fill();
      };
      arrow(EMB[an.b], EMB[an.a], 'rgba(122,31,36,0.5)'); // man → king
      arrow(EMB[an.c], EMB[an.expect], ACCENT);            // woman → queen
    }

    // Words
    wordRects = [];
    ctx.font = 'italic 14px "Source Serif 4", Georgia, serif';
    Object.entries(EMB).forEach(([w, v]) => {
      const px = tx(v[0]), py = ty(v[1]);
      const isAnalogy = an && (w === an.a || w === an.b || w === an.c || w === an.expect);
      const isSelected = w === state.selected;
      ctx.fillStyle = isAnalogy ? ACCENT : '#262320';
      ctx.beginPath();
      ctx.arc(px, py, isSelected ? 4 : 2.5, 0, Math.PI * 2);
      ctx.fill();
      const fontSize = isAnalogy || isSelected ? 14 : 12;
      ctx.font = `italic ${fontSize}px "Source Serif 4", Georgia, serif`;
      ctx.fillStyle = isSelected ? ACCENT : (isAnalogy ? ACCENT : '#262320');
      const m = ctx.measureText(w);
      ctx.textAlign = 'left';
      ctx.fillText(w, px + 6, py + 4);
      wordRects.push({ x: px - 4, y: py - 10, w: m.width + 14, h: 22, word: w });
    });
  }

  function render() {
    Array.from(analogiesEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.an === state.analogyKey);
    });
    const an = ANALOGIES[state.analogyKey];
    captionEl.innerHTML = `<span class="ml-cap-num">·</span> The faint accent arrow shows <em>${an.b} → ${an.a}</em>; the bright arrow shows <em>${an.c} → ${an.expect}</em>. The two are roughly parallel — that's the analogy direction encoded as a literal vector. Hover any word to read its nearest neighbors.`;
    hoverInfo.textContent = state.selected || '— hover a word —';
    if (state.selected && EMB[state.selected]) {
      const nbrs = neighbors(state.selected, 5);
      let html = '';
      nbrs.forEach((n) => {
        html += `<div class="row"><span>${n.word}</span><b>${(n.sim).toFixed(3)}</b></div>`;
      });
      readoutEl.innerHTML = html;
    } else {
      readoutEl.innerHTML = '<div style="font-family:var(--serif); font-style:italic; font-size:13px; color:var(--ml-ink-fade)">Hover any word to see its 5 closest neighbors by cosine similarity.</div>';
    }
    canvas.style.cursor = 'pointer';
    canvasCtl.redraw();
  }

  render();
})();
