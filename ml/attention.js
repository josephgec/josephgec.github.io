/* Vol. XXIII — Attention */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK_FADE } = V;

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

  const state = {
    sentKey: 'bank',
    qIdx: 2,
  };

  const sentencesEl = document.getElementById('sentences');
  const readoutEl = document.getElementById('readout');
  const canvas = document.getElementById('canvas');

  Object.keys(SENTENCES).forEach((k) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = `"${k}"`;
    b.dataset.sent = k;
    b.addEventListener('click', () => {
      state.sentKey = k;
      state.qIdx = 2;
      render();
    });
    sentencesEl.appendChild(b);
  });

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  // Word click hit-test rectangles — populated during draw, used in pointerdown.
  let wordRects = [];
  canvas.addEventListener('click', (e) => {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    for (const wr of wordRects) {
      if (x >= wr.x && x <= wr.x + wr.w && y >= wr.y && y <= wr.y + wr.h) {
        state.qIdx = wr.i;
        render();
        return;
      }
    }
  });

  function draw(ctx, size) {
    const words = SENTENCES[state.sentKey];
    const weights = attnFor(words, state.qIdx);
    const padX = 24;
    const wordY = 60;
    const barTop = 110;
    const barBottom = size.h - 30;
    const barH = barBottom - barTop;

    // Compute word positions (centered)
    ctx.font = 'italic 22px "Source Serif 4", Georgia, serif';
    const widths = words.map((w) => ctx.measureText(w).width);
    const gap = 18;
    const totalW = widths.reduce((a, b) => a + b, 0) + gap * (words.length - 1);
    let x = (size.w - totalW) / 2;
    wordRects = [];
    words.forEach((w, i) => {
      wordRects.push({ x: x - 4, y: wordY - 26, w: widths[i] + 8, h: 36, i });
      x += widths[i] + gap;
    });

    // Bars below each word
    wordRects.forEach((wr, i) => {
      const wt = weights[i];
      const colW = wr.w + gap - 8;
      const barX = wr.x + (wr.w - colW) / 2 + 4;
      const h = wt * barH;
      ctx.fillStyle = i === state.qIdx ? ACCENT : `rgba(122,31,36,${0.12 + wt * 0.6})`;
      ctx.fillRect(barX, barBottom - h, colW, h);
      // weight label below
      ctx.fillStyle = INK_FADE;
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText((wt * 100).toFixed(0) + '%', barX + colW / 2, barBottom + 14);
    });

    // Words on top
    words.forEach((w, i) => {
      const wr = wordRects[i];
      const isQuery = i === state.qIdx;
      ctx.font = `${isQuery ? 'italic ' : ''}22px "Source Serif 4", Georgia, serif`;
      ctx.fillStyle = isQuery ? ACCENT : '#262320';
      ctx.textAlign = 'left';
      ctx.fillText(w, wr.x + 4, wordY);
    });

    // Title
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('attention weights · ' + words[state.qIdx] + ' → others', size.w / 2, 22);
  }

  function render() {
    Array.from(sentencesEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.sent === state.sentKey);
    });
    const words = SENTENCES[state.sentKey];
    const weights = attnFor(words, state.qIdx);
    let html = '<div style="font-family:var(--mono); font-size:11px; color:var(--ml-ink-fade); margin-bottom:8px">attention from <em style="color:var(--accent)">' + words[state.qIdx] + '</em></div>';
    weights.forEach((w, i) => {
      html += `<div class="row"><span style="${i === state.qIdx ? 'color:var(--accent)' : ''}">${words[i]}</span><b>${(w * 100).toFixed(1)}%</b></div>`;
    });
    readoutEl.innerHTML = html;
    canvas.style.cursor = 'pointer';
    canvasCtl.redraw();
  }

  render();
})();
