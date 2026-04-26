/* Vol. XXXI — Hyperparameter Tuning */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK_FADE } = V;

  // 2D score landscapes — multiple bumps for "Bumpy", single peak for "Easy", ridge for "Ridge"
  const LANDSCAPES = {
    'Bumpy':  (x, y) => {
      const a = Math.exp(-((x - 0.65)**2 + (y - 0.7)**2) / 0.05);
      const b = Math.exp(-((x - 0.25)**2 + (y - 0.3)**2) / 0.08) * 0.7;
      const c = Math.exp(-((x - 0.8)**2 + (y - 0.2)**2) / 0.04) * 0.55;
      return a + b + c;
    },
    'Easy peak': (x, y) => Math.exp(-((x - 0.6)**2 + (y - 0.55)**2) / 0.10),
    'Ridge':  (x, y) => Math.exp(-((y - 0.55 - 0.3 * (x - 0.5))**2) / 0.02) * 0.95,
  };

  const state = {
    strategy: 'random',
    landscapeKey: 'Bumpy',
    trials: 20,
    points: [], // {x, y, score, order}
    running: false,
    timer: null,
  };

  const strategiesEl = document.getElementById('strategies');
  const trialsSlider = document.getElementById('trials-slider');
  const trialsLabel = document.getElementById('trials-label');
  const landscapesEl = document.getElementById('landscapes');
  const runBtn = document.getElementById('run-btn');
  const resetBtn = document.getElementById('reset-btn');
  const readoutEl = document.getElementById('readout');
  const canvas = document.getElementById('canvas');

  ['Grid', 'Random', 'Bayesian'].forEach((lbl) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = lbl;
    b.dataset.strategy = lbl.toLowerCase();
    b.addEventListener('click', () => { state.strategy = b.dataset.strategy; reset(); render(); });
    strategiesEl.appendChild(b);
  });
  Object.keys(LANDSCAPES).forEach((k) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = k;
    b.dataset.landscape = k;
    b.addEventListener('click', () => { state.landscapeKey = k; reset(); render(); });
    landscapesEl.appendChild(b);
  });

  trialsSlider.addEventListener('input', (e) => {
    state.trials = +e.target.value;
    trialsLabel.textContent = `${state.trials} trials`;
  });
  runBtn.addEventListener('click', () => { if (state.running) stopRun(); else startRun(); });
  resetBtn.addEventListener('click', () => { reset(); render(); });

  function reset() {
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    state.points = [];
    state.running = false;
    runBtn.textContent = '▷ Run search';
  }

  function nextProbe() {
    const f = LANDSCAPES[state.landscapeKey];
    if (state.strategy === 'grid') {
      const n = Math.ceil(Math.sqrt(state.trials));
      const i = state.points.length;
      const r = Math.floor(i / n);
      const c = i % n;
      const x = (c + 0.5) / n;
      const y = (r + 0.5) / n;
      return { x, y, score: f(x, y), order: i };
    } else if (state.strategy === 'random') {
      const x = Math.random();
      const y = Math.random();
      return { x, y, score: f(x, y), order: state.points.length };
    } else {
      // Bayesian-ish: first 4 random, then exploit + slight noise around best so far + occasionally explore
      if (state.points.length < 4) {
        const x = Math.random(), y = Math.random();
        return { x, y, score: f(x, y), order: state.points.length };
      }
      // Pick a candidate that maximizes "expected improvement" — simulated by taking 30 random
      // candidates, computing score under our nearest-neighbor surrogate, plus uncertainty bonus.
      const candidates = Array.from({ length: 30 }, () => [Math.random(), Math.random()]);
      const best = state.points.reduce((m, p) => Math.max(m, p.score), 0);
      let chosen = candidates[0], bestEI = -Infinity;
      for (const [cx, cy] of candidates) {
        // Surrogate: weighted nearest-neighbor of past points
        let num = 0, den = 0, minDist = Infinity;
        for (const p of state.points) {
          const d = Math.hypot(cx - p.x, cy - p.y);
          minDist = Math.min(minDist, d);
          const w = Math.exp(-d * d / 0.05);
          num += w * p.score; den += w;
        }
        const surrMean = den > 0 ? num / den : 0.5;
        const uncertainty = Math.min(1, minDist * 4);
        const improv = Math.max(0, surrMean - best);
        const ei = improv + 0.3 * uncertainty;
        if (ei > bestEI) { bestEI = ei; chosen = [cx, cy]; }
      }
      return { x: chosen[0], y: chosen[1], score: f(chosen[0], chosen[1]), order: state.points.length };
    }
  }

  function startRun() {
    state.running = true;
    runBtn.textContent = '⏸ Pause';
    state.timer = setInterval(() => {
      if (state.points.length >= state.trials) { stopRun(); return; }
      state.points.push(nextProbe());
      render();
    }, 200);
  }
  function stopRun() {
    state.running = false;
    runBtn.textContent = state.points.length >= state.trials ? '▷ Run search' : '▷ Resume';
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    const f = LANDSCAPES[state.landscapeKey];
    const topH = size.h * 0.65;
    const botH = size.h - topH - 24;
    const padX = 24;

    // Top: heatmap with probes
    const W = size.w - padX * 2, H = topH - 16;
    const x0 = padX, y0 = 16;

    // Heatmap
    const cells = 60;
    for (let i = 0; i < cells; i++) {
      for (let j = 0; j < cells; j++) {
        const xx = (j + 0.5) / cells;
        const yy = (i + 0.5) / cells;
        const v = f(xx, yy);
        const t = Math.max(0, Math.min(1, v));
        const r = Math.round(247 - t * 130);
        const g = Math.round(244 - t * 195);
        const b = Math.round(236 - t * 195);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x0 + j * W / cells, y0 + (cells - 1 - i) * H / cells, W / cells + 1, H / cells + 1);
      }
    }
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x0, y0, W, H);

    // Probes
    state.points.forEach((p, i) => {
      const px = x0 + p.x * W;
      const py = y0 + (1 - p.y) * H;
      const isLatest = i === state.points.length - 1;
      ctx.fillStyle = isLatest ? ACCENT : '#262320';
      ctx.beginPath();
      ctx.arc(px, py, isLatest ? 5 : 3.5, 0, Math.PI * 2);
      ctx.fill();
    });
    // Best so far
    if (state.points.length > 0) {
      const best = state.points.reduce((b, p) => p.score > b.score ? p : b);
      const px = x0 + best.x * W;
      const py = y0 + (1 - best.y) * H;
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px, py, 9, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Bottom: best-score-over-trials curve
    const yPlot = topH + 12;
    const plotX0 = 60;
    const plotW = size.w - plotX0 - padX;
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.beginPath();
    ctx.moveTo(plotX0, yPlot);
    ctx.lineTo(plotX0, yPlot + botH);
    ctx.lineTo(plotX0 + plotW, yPlot + botH);
    ctx.stroke();
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    [0, 0.5, 1].forEach((y) => {
      ctx.fillText(y.toFixed(1), plotX0 - 6, yPlot + botH - y * botH + 3);
    });
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('best score so far', plotX0, yPlot - 4);

    if (state.points.length > 0) {
      let best = 0;
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 2;
      ctx.beginPath();
      state.points.forEach((p, i) => {
        if (p.score > best) best = p.score;
        const x = plotX0 + ((i + 1) / state.trials) * plotW;
        const y = yPlot + botH - best * botH;
        if (i === 0) ctx.moveTo(plotX0, yPlot + botH); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
  }

  function render() {
    Array.from(strategiesEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.strategy === state.strategy);
    });
    Array.from(landscapesEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.landscape === state.landscapeKey);
    });
    trialsLabel.textContent = `${state.trials} trials`;
    trialsSlider.value = state.trials;
    const best = state.points.length > 0 ? state.points.reduce((m, p) => Math.max(m, p.score), 0) : 0;
    readoutEl.innerHTML = `
      <div class="row"><span>trials run</span><b>${state.points.length} / ${state.trials}</b></div>
      <div class="row"><span>best score</span><b>${formatNum(best)}</b></div>
      <div class="row"><span>strategy</span><b>${state.strategy}</b></div>`;
    canvasCtl.redraw();
  }

  render();
})();
