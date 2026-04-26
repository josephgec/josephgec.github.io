/* Vol. XXV — A Geometry of Meaning (Embeddings) */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK, INK_FADE } = V;

  const EMB = {
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

  // Loose semantic clusters for hull annotation
  const CLUSTERS = [
    { name: 'royalty / gender', words: ['king', 'queen', 'man', 'woman'] },
    { name: 'places', words: ['Paris', 'France', 'Rome', 'Italy', 'Berlin', 'Germany'] },
    { name: 'animals', words: ['cat', 'dog', 'pet', 'animal'] },
    { name: 'vehicles', words: ['car', 'truck', 'vehicle', 'bicycle'] },
    { name: 'verbs', words: ['walk', 'walked', 'run', 'ran'] },
    { name: 'sizes', words: ['big', 'bigger', 'small', 'smaller'] },
  ];

  const ANALOGIES = {
    'king − man + woman = ?':         { a: 'king',   b: 'man',    c: 'woman',  expect: 'queen' },
    'Paris − France + Italy = ?':     { a: 'Paris',  b: 'France', c: 'Italy',  expect: 'Rome' },
    'walk − walked + ran = ?':        { a: 'walk',   b: 'walked', c: 'ran',    expect: 'run' },
    'big − bigger + smaller = ?':     { a: 'big',    b: 'bigger', c: 'smaller',expect: 'small' },
  };

  const VIEWS = [
    { key: 'analogy', label: 'Preset analogy' },
    { key: 'builder', label: 'Build your own · A − B + C' },
    { key: 'cosine',  label: 'Cosine similarity · hover' },
  ];
  const CAPTIONS = {
    analogy: 'Word vectors live in 2D here (real ones live in hundreds of dimensions). Faint arrow: <em>B → A</em>. Bright arrow: <em>C → expected</em>. The two arrows are roughly parallel — that\'s the analogy direction encoded as a literal vector.',
    builder: 'Pick three words on the canvas to set <em>A</em>, <em>B</em>, <em>C</em>. We compute the result vector <em>A − B + C</em> as plain arithmetic, then snap to the nearest word in the space — the analogy answer.',
    cosine:  'Hover any word. We hold <em>king</em> as a fixed reference and draw the angle between it and the hovered word. The cosine of that angle is the similarity score — 1 means same direction, 0 perpendicular, −1 opposite.',
  };

  const state = {
    view: 'analogy',
    analogyKey: 'king − man + woman = ?',
    selected: null,
    builder: { a: null, b: null, c: null },
    hovered: null,
  };

  // ───────── DOM ─────────
  const analogiesEl = document.getElementById('analogies');
  const tabsEl = document.getElementById('view-tabs');
  const hoverInfo = document.getElementById('hover-info');
  const readoutEl = document.getElementById('readout');
  const captionEl = document.getElementById('caption');
  const canvas = document.getElementById('canvas');
  const builderPane = document.getElementById('builder-pane');
  const builderReadout = document.getElementById('builder-readout');
  const builderClear = document.getElementById('builder-clear');

  Object.keys(ANALOGIES).forEach((k) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = k;
    b.dataset.an = k;
    b.addEventListener('click', () => {
      state.view = 'analogy';
      state.analogyKey = k;
      render();
    });
    analogiesEl.appendChild(b);
  });

  VIEWS.forEach((v) => {
    const b = document.createElement('button');
    b.textContent = v.label;
    b.dataset.view = v.key;
    b.addEventListener('click', () => { state.view = v.key; render(); });
    tabsEl.appendChild(b);
  });

  builderClear && builderClear.addEventListener('click', () => {
    state.builder = { a: null, b: null, c: null };
    render();
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
  function nearestVec(v, exclude = []) {
    let best = null, bSim = -Infinity;
    for (const [w, vec] of Object.entries(EMB)) {
      if (exclude.includes(w)) continue;
      const s = cosine(v, vec);
      if (s > bSim) { bSim = s; best = w; }
    }
    return { word: best, sim: bSim };
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
    if (hit !== state.hovered) { state.hovered = hit; state.selected = hit; render(); }
  });
  canvas.addEventListener('mouseleave', () => {
    state.hovered = null;
    if (state.view !== 'builder') state.selected = null;
    render();
  });

  canvas.addEventListener('click', (e) => {
    if (state.view !== 'builder') return;
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    let hit = null;
    for (const wr of wordRects) {
      if (x >= wr.x && x <= wr.x + wr.w && y >= wr.y && y <= wr.y + wr.h) { hit = wr.word; break; }
    }
    if (!hit) return;
    if (!state.builder.a) state.builder.a = hit;
    else if (!state.builder.b) state.builder.b = hit;
    else if (!state.builder.c) state.builder.c = hit;
    else state.builder = { a: hit, b: null, c: null };
    render();
  });

  function project(size) {
    const R = 3;
    return {
      tx: (x) => (x + R) / (2 * R) * size.w,
      ty: (y) => size.h - (y + R) / (2 * R) * size.h,
    };
  }

  function drawArrow(ctx, x1, y1, x2, y2, color, w) {
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = w || 2;
    ctx.beginPath();
    ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    ctx.stroke();
    const ang = Math.atan2(y2 - y1, x2 - x1);
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - 8 * Math.cos(ang - 0.4), y2 - 8 * Math.sin(ang - 0.4));
    ctx.lineTo(x2 - 8 * Math.cos(ang + 0.4), y2 - 8 * Math.sin(ang + 0.4));
    ctx.fill();
  }

  // Convex hull (gift wrapping) — handles small clusters
  function convexHull(pts) {
    if (pts.length < 3) return pts.slice();
    const points = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const cross = (O, A, B) => (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0]);
    const lower = [];
    for (const p of points) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
      lower.push(p);
    }
    const upper = [];
    for (let i = points.length - 1; i >= 0; i--) {
      const p = points[i];
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
      upper.push(p);
    }
    upper.pop(); lower.pop();
    return lower.concat(upper);
  }

  function drawClusters(ctx, size, p) {
    CLUSTERS.forEach((cl, idx) => {
      const pts = cl.words.filter((w) => EMB[w]).map((w) => [p.tx(EMB[w][0]), p.ty(EMB[w][1])]);
      if (pts.length < 2) return;
      // Inflate via average and pad
      const cx = pts.reduce((s, q) => s + q[0], 0) / pts.length;
      const cy = pts.reduce((s, q) => s + q[1], 0) / pts.length;
      const padded = pts.map(([x, y]) => {
        const dx = x - cx, dy = y - cy;
        const d = Math.hypot(dx, dy) || 1;
        return [x + dx / d * 22, y + dy / d * 22];
      });
      const hull = convexHull(padded);
      ctx.beginPath();
      hull.forEach((pt, i) => {
        if (i === 0) ctx.moveTo(pt[0], pt[1]); else ctx.lineTo(pt[0], pt[1]);
      });
      ctx.closePath();
      ctx.fillStyle = 'rgba(122,31,36,0.04)';
      ctx.strokeStyle = 'rgba(122,31,36,0.25)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.fill();
      ctx.stroke();
      ctx.setLineDash([]);
      // Cluster label near top of hull
      const top = hull.reduce((a, b) => (b[1] < a[1] ? b : a));
      ctx.fillStyle = INK_FADE;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText(cl.name, top[0], top[1] - 4);
    });
  }

  function drawAxesAndGrid(ctx, size, p) {
    ctx.strokeStyle = 'rgba(38,35,32,0.06)';
    ctx.lineWidth = 1;
    for (let i = -3; i <= 3; i++) {
      ctx.beginPath(); ctx.moveTo(p.tx(i), 0); ctx.lineTo(p.tx(i), size.h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p.ty(i)); ctx.lineTo(size.w, p.ty(i)); ctx.stroke();
    }
    ctx.strokeStyle = 'rgba(38,35,32,0.18)';
    ctx.beginPath(); ctx.moveTo(0, p.ty(0)); ctx.lineTo(size.w, p.ty(0)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(p.tx(0), 0); ctx.lineTo(p.tx(0), size.h); ctx.stroke();
  }

  function drawWords(ctx, size, p, options = {}) {
    wordRects = [];
    const an = ANALOGIES[state.analogyKey];
    Object.entries(EMB).forEach(([w, v]) => {
      const px = p.tx(v[0]), py = p.ty(v[1]);
      const isAnalogy = state.view === 'analogy' && an && (w === an.a || w === an.b || w === an.c || w === an.expect);
      const isBuilder = state.view === 'builder' && (w === state.builder.a || w === state.builder.b || w === state.builder.c);
      const isSelected = w === state.selected;
      const accent = isAnalogy || isBuilder || isSelected;
      ctx.fillStyle = accent ? ACCENT : '#262320';
      ctx.beginPath();
      ctx.arc(px, py, accent ? 4 : 2.5, 0, Math.PI * 2);
      ctx.fill();
      const fontSize = accent ? 14 : 12;
      ctx.font = `italic ${fontSize}px "Source Serif 4", Georgia, serif`;
      ctx.fillStyle = accent ? ACCENT : '#262320';
      const m = ctx.measureText(w);
      ctx.textAlign = 'left';
      ctx.fillText(w, px + 6, py + 4);
      wordRects.push({ x: px - 4, y: py - 10, w: m.width + 14, h: 22, word: w });
    });
  }

  function draw(ctx, size) {
    const p = project(size);
    drawAxesAndGrid(ctx, size, p);
    drawClusters(ctx, size, p);

    if (state.view === 'analogy') {
      const an = ANALOGIES[state.analogyKey];
      if (an && EMB[an.a] && EMB[an.b] && EMB[an.c] && EMB[an.expect]) {
        drawArrow(ctx, p.tx(EMB[an.b][0]), p.ty(EMB[an.b][1]),
                  p.tx(EMB[an.a][0]), p.ty(EMB[an.a][1]), 'rgba(122,31,36,0.5)');
        drawArrow(ctx, p.tx(EMB[an.c][0]), p.ty(EMB[an.c][1]),
                  p.tx(EMB[an.expect][0]), p.ty(EMB[an.expect][1]), ACCENT);
      }
    } else if (state.view === 'builder') {
      drawBuilder(ctx, size, p);
    } else if (state.view === 'cosine') {
      drawCosine(ctx, size, p);
    }

    drawWords(ctx, size, p);
  }

  function drawBuilder(ctx, size, p) {
    const { a, b, c } = state.builder;
    if (!a || !b || !c) {
      // hint
      ctx.fillStyle = INK_FADE;
      ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      const slot = !a ? 'A' : (!b ? 'B' : 'C');
      ctx.fillText(`click any word on the canvas to set ${slot}`, size.w / 2, size.h - 14);
      // Show what's picked
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.fillText(`A = ${a || '—'}     B = ${b || '—'}     C = ${c || '—'}`, size.w / 2, 22);
      return;
    }
    // Compute result vector
    const va = EMB[a], vb = EMB[b], vc = EMB[c];
    const result = [va[0] - vb[0] + vc[0], va[1] - vb[1] + vc[1]];
    const nearest = nearestVec(result, [a, b, c]);

    // Draw vector construction:
    //   start at C, add (A − B) → end at result
    //   visualize as: arrow from C to (C + (A − B))
    ctx.strokeStyle = 'rgba(38,35,32,0.55)';
    ctx.lineWidth = 1.4;
    // (A − B) shown at origin → small dotted helper arrow on top of the canvas
    // Then the same delta moved from C
    const px0 = p.tx(vc[0]), py0 = p.ty(vc[1]);
    const px1 = p.tx(result[0]), py1 = p.ty(result[1]);
    drawArrow(ctx, px0, py0, px1, py1, ACCENT, 2);

    // Faint arrow B → A as the "direction" being added
    drawArrow(ctx, p.tx(vb[0]), p.ty(vb[1]),
              p.tx(va[0]), p.ty(va[1]), 'rgba(122,31,36,0.45)', 1.5);

    // Result point
    ctx.fillStyle = INK;
    ctx.beginPath();
    ctx.arc(px1, py1, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#fffdf6';
    ctx.lineWidth = 1.6;
    ctx.stroke();
    // Label
    ctx.fillStyle = INK;
    ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText(`= ${a} − ${b} + ${c}`, px1 + 10, py1 - 6);
    ctx.fillStyle = ACCENT;
    ctx.fillText(`≈ ${nearest.word}  (cos ${nearest.sim.toFixed(2)})`, px1 + 10, py1 + 12);

    // Snap line from result → nearest word
    const np = EMB[nearest.word];
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(px1, py1);
    ctx.lineTo(p.tx(np[0]), p.ty(np[1]));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawCosine(ctx, size, p) {
    // Stable reference: 'king' (first cluster anchor). Hovering any other
    // word draws the angle between it and the reference.
    const refWord = 'king';
    if (!state.hovered || !EMB[state.hovered] || state.hovered === refWord) {
      ctx.fillStyle = INK_FADE;
      ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText(`reference word: ${refWord} — hover any other word to see the angle`, size.w / 2, size.h - 14);
      return;
    }
    const v1 = EMB[refWord];
    const v2 = EMB[state.hovered];
    const ox = p.tx(0), oy = p.ty(0);
    // origin
    ctx.fillStyle = INK_FADE;
    ctx.beginPath(); ctx.arc(ox, oy, 3, 0, Math.PI * 2); ctx.fill();

    // draw two vectors as arrows from origin
    drawArrow(ctx, ox, oy, p.tx(v1[0]), p.ty(v1[1]), 'rgba(38,35,32,0.65)', 1.5);
    drawArrow(ctx, ox, oy, p.tx(v2[0]), p.ty(v2[1]), ACCENT, 1.8);

    // arc between them
    const a1 = Math.atan2(p.ty(v1[1]) - oy, p.tx(v1[0]) - ox);
    const a2 = Math.atan2(p.ty(v2[1]) - oy, p.tx(v2[0]) - ox);
    const r = 40;
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    // pick the shorter arc direction
    let aStart = a1, aEnd = a2;
    let delta = aEnd - aStart;
    while (delta > Math.PI) delta -= 2 * Math.PI;
    while (delta < -Math.PI) delta += 2 * Math.PI;
    ctx.arc(ox, oy, r, aStart, aStart + delta, delta < 0);
    ctx.stroke();

    // cos value label near arc midpoint
    const aMid = aStart + delta / 2;
    const labelX = ox + Math.cos(aMid) * (r + 18);
    const labelY = oy + Math.sin(aMid) * (r + 18);
    const c = cosine(v1, v2);
    ctx.fillStyle = INK;
    ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText(`cos = ${c.toFixed(2)}`, labelX, labelY);
    // Math anchored to the picture: cosine equation next to the arc
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.fillStyle = INK_FADE;
    ctx.fillText('cos θ = (a · b) / (‖a‖ ‖b‖)', labelX, labelY + 16);
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = INK_FADE;
    ctx.textAlign = 'center';
    ctx.fillText('1 = same dir, 0 = ⊥, −1 = opposite', size.w / 2, size.h - 14);
  }

  function render() {
    Array.from(analogiesEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.an === state.analogyKey && state.view === 'analogy');
    });
    Array.from(tabsEl.children).forEach((btn) => {
      btn.classList.toggle('on', btn.dataset.view === state.view);
    });

    // Show/hide the builder pane
    if (builderPane) builderPane.style.display = state.view === 'builder' ? 'block' : 'none';
    if (builderReadout) {
      const { a, b, c } = state.builder;
      let txt = `A = ${a || '—'}, B = ${b || '—'}, C = ${c || '—'}`;
      if (a && b && c) {
        const va = EMB[a], vb = EMB[b], vc = EMB[c];
        const result = [va[0] - vb[0] + vc[0], va[1] - vb[1] + vc[1]];
        const nearest = nearestVec(result, [a, b, c]);
        txt += `<br />→ <em style="color:var(--accent); font-style:italic">${nearest.word}</em> (cos ${nearest.sim.toFixed(2)})`;
      }
      builderReadout.innerHTML = txt;
    }

    if (captionEl) {
      captionEl.innerHTML = '<span class="ml-cap-num">·</span> ' + CAPTIONS[state.view];
    }

    if (hoverInfo) hoverInfo.textContent = state.selected || '— hover a word —';
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
