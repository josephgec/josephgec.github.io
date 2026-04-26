/* Vol. XXIII — Attention */
(function () {
  'use strict';
  const V = window.MLViz;
  const { ACCENT, INK, INK_FADE } = V;

  const SENTENCES = {
    bank:  ['The', 'river', 'bank', 'flooded', 'after', 'the', 'storm'],
    bat:   ['She', 'swung', 'the', 'bat', 'and', 'hit', 'a', 'homer'],
    light: ['The', 'light', 'package', 'arrived', 'in', 'the', 'morning'],
  };

  // 2D semantic vectors hand-tuned so attention bars look meaningful.
  const SEM = {
    The: [0.2, 0.0], the: [0.2, 0.0],
    river: [1.0, 0.6], bank: [0.85, 0.55], flooded: [1.0, 0.7], storm: [0.9, 0.8], after: [0.1, 0.3],
    She: [0.0, 0.2], swung: [-0.6, 0.9], bat: [-0.7, 0.8], hit: [-0.5, 0.95], homer: [-0.7, 0.95],
    and: [0.0, 0.0], a: [0.1, 0.0],
    light: [0.4, -0.7], package: [0.6, -0.6], arrived: [0.5, -0.4], in: [0.0, 0.0], morning: [0.7, -0.5],
  };

  function attnFor(words, qIdx) {
    const get = (w) => SEM[w] || [0, 0];
    const q = get(words[qIdx]);
    const scores = words.map((w, i) => {
      if (i === qIdx) return 1.5;
      const k = get(w);
      return q[0] * k[0] + q[1] * k[1] + 0.05;
    });
    const m = Math.max(...scores);
    const exps = scores.map((s) => Math.exp((s - m) * 3));
    const z = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / z);
  }

  // ───────── View modes ─────────
  const VIEWS = [
    { key: 'bars',   label: 'One query · bars' },
    { key: 'matrix', label: 'All pairs · matrix' },
    { key: 'blend',  label: 'The output · weighted blend' },
  ];
  const CAPTIONS = {
    bars:   'The query word "looks at" the others through learned similarity (dot product), softmax turns the scores into weights, and the query gets re-described as a weighted blend of values.',
    matrix: 'The full attention matrix. Each row is one query; each cell is the softmax weight from that query to the column\'s key. Diagonal cells are the query attending to itself; off-diagonal mass is the context being absorbed.',
    blend:  'The query\'s output is literally a weighted mix of the value vectors. Each segment of the bar is one key\'s contribution, sized by its softmax weight — read it left-to-right to see what the query becomes.',
  };

  const state = {
    sentKey: 'bank',
    qIdx: 2,
    viewMode: 'bars',
  };

  // ───────── DOM ─────────
  const sentencesEl = document.getElementById('sentences');
  const readoutEl = document.getElementById('readout');
  const tabsEl = document.getElementById('view-tabs');
  const captionEl = document.getElementById('caption-text');
  const canvas = document.getElementById('canvas');

  Object.keys(SENTENCES).forEach((k) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = `"${k}"`;
    b.dataset.sent = k;
    b.addEventListener('click', () => {
      state.sentKey = k;
      // pick the most "content-y" word as default query for the new sentence
      const words = SENTENCES[k];
      const defaultIdx = {
        bank: words.indexOf('bank'),
        bat:  words.indexOf('bat'),
        light: words.indexOf('light'),
      }[k];
      state.qIdx = defaultIdx >= 0 ? defaultIdx : 0;
      render();
    });
    sentencesEl.appendChild(b);
  });

  VIEWS.forEach((v) => {
    const b = document.createElement('button');
    b.textContent = v.label;
    b.dataset.view = v.key;
    b.addEventListener('click', () => {
      state.viewMode = v.key;
      render();
    });
    tabsEl.appendChild(b);
  });

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  // Click hit-test rectangles — populated during draw, used in click handler.
  // Each entry: { x, y, w, h, qIdx } — clicking sets state.qIdx.
  let hitRects = [];
  canvas.addEventListener('click', (e) => {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    for (const wr of hitRects) {
      if (x >= wr.x && x <= wr.x + wr.w && y >= wr.y && y <= wr.y + wr.h) {
        if (typeof wr.qIdx === 'number') state.qIdx = wr.qIdx;
        render();
        return;
      }
    }
  });

  function draw(ctx, size) {
    const words = SENTENCES[state.sentKey];
    hitRects = [];
    if (state.viewMode === 'matrix') {
      drawMatrix(ctx, size, words);
    } else if (state.viewMode === 'blend') {
      drawBlend(ctx, size, words, attnFor(words, state.qIdx));
    } else {
      drawBars(ctx, size, words, attnFor(words, state.qIdx));
    }
  }

  // ───────── View 1: Bars (one query → all keys) ─────────
  function drawBars(ctx, size, words, weights) {
    const wordY = 60;
    const barTop = 110;
    const barBottom = size.h - 30;
    const barH = barBottom - barTop;

    ctx.font = 'italic 22px "Source Serif 4", Georgia, serif';
    const widths = words.map((w) => ctx.measureText(w).width);
    const gap = 18;
    const totalW = widths.reduce((a, b) => a + b, 0) + gap * (words.length - 1);
    let x = (size.w - totalW) / 2;
    const wordRects = [];
    words.forEach((w, i) => {
      wordRects.push({ x: x - 4, y: wordY - 26, w: widths[i] + 8, h: 36, qIdx: i });
      x += widths[i] + gap;
    });

    wordRects.forEach((wr, i) => {
      const wt = weights[i];
      const colW = wr.w + gap - 8;
      const barX = wr.x + (wr.w - colW) / 2 + 4;
      const h = wt * barH;
      ctx.fillStyle = i === state.qIdx ? ACCENT : `rgba(122,31,36,${0.12 + wt * 0.6})`;
      ctx.fillRect(barX, barBottom - h, colW, h);
      ctx.fillStyle = INK_FADE;
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText((wt * 100).toFixed(0) + '%', barX + colW / 2, barBottom + 14);
      // Hit rect = full column (word label + bar) so users can click anywhere
      hitRects.push({ x: wr.x, y: wordY - 28, w: wr.w, h: barBottom - wordY + 30, qIdx: i });
    });

    words.forEach((w, i) => {
      const wr = wordRects[i];
      const isQuery = i === state.qIdx;
      ctx.font = `${isQuery ? 'italic ' : ''}22px "Source Serif 4", Georgia, serif`;
      ctx.fillStyle = isQuery ? ACCENT : INK;
      ctx.textAlign = 'left';
      ctx.fillText(w, wr.x + 4, wordY);
    });

    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('attention from "' + words[state.qIdx] + '" → others (softmax of dot products)', size.w / 2, 22);
  }

  // ───────── View 2: Full attention matrix (all queries × all keys) ─────────
  function drawMatrix(ctx, size, words) {
    const N = words.length;
    const allWeights = words.map((_, i) => attnFor(words, i));

    // Title
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('softmax(Q·Kᵀ/√d) — every query (row) attending to every key (col)', size.w / 2, 22);

    // Layout
    const padTop = 90;
    const padLeft = 88;
    const padRight = 24;
    const padBottom = 56;
    const availW = size.w - padLeft - padRight;
    const availH = size.h - padTop - padBottom;
    const cell = Math.floor(Math.min(availW, availH) / N);
    const matW = cell * N;
    const matH = cell * N;
    const x0 = padLeft + (availW - matW) / 2;
    const y0 = padTop;

    // Faint grid background
    ctx.fillStyle = 'rgba(38,35,32,0.03)';
    ctx.fillRect(x0, y0, matW, matH);

    // Cells
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const wt = allWeights[r][c];
        ctx.fillStyle = `rgba(122,31,36,${0.06 + wt * 0.9})`;
        ctx.fillRect(x0 + c * cell + 1, y0 + r * cell + 1, cell - 2, cell - 2);
      }
    }

    // Row labels (italic, right-aligned), and capture row hit rects
    ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
    ctx.textBaseline = 'middle';
    words.forEach((w, r) => {
      const cy = y0 + r * cell + cell / 2;
      ctx.fillStyle = r === state.qIdx ? ACCENT : INK;
      ctx.textAlign = 'right';
      ctx.fillText(w, x0 - 8, cy);
      hitRects.push({ x: 0, y: y0 + r * cell, w: x0 + matW, h: cell, qIdx: r });
    });

    // Column labels (rotated -45° above the matrix)
    ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
    ctx.textBaseline = 'alphabetic';
    words.forEach((w, c) => {
      const cx = x0 + c * cell + cell / 2;
      ctx.save();
      ctx.translate(cx, y0 - 8);
      ctx.rotate(-Math.PI / 4);
      ctx.textAlign = 'left';
      ctx.fillStyle = c === state.qIdx ? ACCENT : INK_FADE;
      ctx.fillText(w, 0, 0);
      ctx.restore();
    });

    // Highlight current query row
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 2;
    ctx.strokeRect(x0 - 1, y0 + state.qIdx * cell - 1, matW + 2, cell + 2);

    // Tiny axis labels
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.save();
    ctx.translate(28, y0 + matH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('QUERY', 0, 0);
    ctx.restore();
    ctx.textAlign = 'center';
    ctx.fillText('KEY', x0 + matW / 2, y0 - 64);

    // Legend strip (gradient from faint to dark)
    const legW = 130;
    const legH = 8;
    const legX = x0 + matW - legW;
    const legY = y0 + matH + 18;
    for (let i = 0; i < legW; i++) {
      const t = i / legW;
      ctx.fillStyle = `rgba(122,31,36,${0.06 + t * 0.9})`;
      ctx.fillRect(legX + i, legY, 1, legH);
    }
    ctx.strokeStyle = 'rgba(38,35,32,0.25)';
    ctx.lineWidth = 1;
    ctx.strokeRect(legX, legY, legW, legH);
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('0', legX, legY + legH + 12);
    ctx.textAlign = 'right';
    ctx.fillText('1', legX + legW, legY + legH + 12);
    ctx.textAlign = 'center';
    ctx.fillText('weight', legX + legW / 2, legY + legH + 12);

    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillStyle = INK_FADE;
    ctx.fillText('click any row to make that word the query', x0, legY + legH + 12);
  }

  // ───────── View 3: Weighted blend (the output as a literal mix) ─────────
  function drawBlend(ctx, size, words, weights) {
    // Title
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('output = Σᵢ weightᵢ · valueᵢ — what the query becomes after attention', size.w / 2, 22);

    // Query header
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('QUERY', size.w / 2, 52);
    ctx.fillStyle = ACCENT;
    ctx.font = 'italic 30px "Source Serif 4", Georgia, serif';
    ctx.fillText('"' + words[state.qIdx] + '"', size.w / 2, 86);

    // Stacked bar — each segment = one key's contribution
    const barH = 64;
    const padX = 32;
    const barLeft = padX;
    const barW = size.w - 2 * padX;
    const barTop = 130;

    let xCursor = barLeft;
    const segLayout = [];
    words.forEach((w, i) => {
      const wt = weights[i];
      const segW = wt * barW;
      segLayout.push({ x: xCursor, w: segW, wt, word: w, i });
      xCursor += segW;
    });

    // Segments
    segLayout.forEach((seg) => {
      const isQ = seg.i === state.qIdx;
      ctx.fillStyle = isQ
        ? 'rgba(122,31,36,0.85)'
        : `rgba(122,31,36,${0.18 + seg.wt * 0.55})`;
      ctx.fillRect(seg.x, barTop, seg.w, barH);

      hitRects.push({ x: seg.x, y: barTop, w: seg.w, h: barH, qIdx: seg.i });

      // Inline label if there's room
      if (seg.w > 42) {
        ctx.fillStyle = (seg.wt > 0.25 || isQ) ? '#fffdf6' : INK;
        ctx.font = 'italic 14px "Source Serif 4", Georgia, serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(seg.word, seg.x + seg.w / 2, barTop + barH / 2 - 8);
        ctx.font = '11px "JetBrains Mono", monospace';
        ctx.fillText((seg.wt * 100).toFixed(0) + '%', seg.x + seg.w / 2, barTop + barH / 2 + 10);
      }
    });

    // Outer border
    ctx.strokeStyle = 'rgba(38,35,32,0.35)';
    ctx.lineWidth = 1;
    ctx.strokeRect(barLeft, barTop, barW, barH);

    // Tick + label below for thin segments that didn't fit a label inline
    ctx.textBaseline = 'top';
    segLayout.forEach((seg) => {
      if (seg.w > 42 || seg.w < 4) return;
      const cx = seg.x + seg.w / 2;
      ctx.strokeStyle = INK_FADE;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx, barTop + barH);
      ctx.lineTo(cx, barTop + barH + 10);
      ctx.stroke();
      ctx.fillStyle = INK_FADE;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText(seg.word, cx, barTop + barH + 12);
    });

    // "= weighted sum" reading line below
    const eqY = barTop + barH + 60;
    const top3 = weights
      .map((w, i) => ({ w, word: words[i], i }))
      .sort((a, b) => b.w - a.w)
      .slice(0, 3);
    const remainder = 1 - top3.reduce((s, p) => s + p.w, 0);

    ctx.fillStyle = INK_FADE;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('READING THE BAR', size.w / 2, eqY - 22);

    ctx.font = 'italic 16px "Source Serif 4", Georgia, serif';
    ctx.fillStyle = INK;
    let parts = top3.map((p) => `${(p.w * 100).toFixed(0)}% × ${p.word}`);
    if (remainder > 0.01) parts.push(`${(remainder * 100).toFixed(0)}% × rest`);
    const eqText = 'output ≈ ' + parts.join('  +  ');
    ctx.fillText(eqText, size.w / 2, eqY);

    // Hint
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.fillStyle = INK_FADE;
    ctx.fillText('click any segment to make that word the query', size.w / 2, size.h - 14);
  }

  // ───────── Render (DOM updates + redraw) ─────────
  function render() {
    Array.from(sentencesEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.sent === state.sentKey);
    });
    Array.from(tabsEl.children).forEach((btn) => {
      btn.classList.toggle('on', btn.dataset.view === state.viewMode);
    });

    const words = SENTENCES[state.sentKey];
    const weights = attnFor(words, state.qIdx);
    let html = '<div style="font-family:var(--mono); font-size:11px; color:var(--ml-ink-fade); margin-bottom:8px">attention from <em style="color:var(--accent)">' + words[state.qIdx] + '</em></div>';
    weights.forEach((w, i) => {
      html += `<div class="row"><span style="${i === state.qIdx ? 'color:var(--accent)' : ''}">${words[i]}</span><b>${(w * 100).toFixed(1)}%</b></div>`;
    });
    readoutEl.innerHTML = html;

    if (captionEl) captionEl.textContent = CAPTIONS[state.viewMode];

    canvas.style.cursor = 'pointer';
    canvasCtl.redraw();
  }

  render();
})();
