/* Vol. XXXI — Hyperparameter Tuning */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK, INK_FADE } = V;

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

  // Per-trial GPU minutes. Bayesian has a small surrogate-fit overhead but otherwise the same per-trial cost.
  const MIN_PER_TRIAL = {
    grid:     6.0,
    random:   6.0,
    bayesian: 6.5, // tiny GP-fit overhead
  };

  const STRATEGIES = ['grid', 'random', 'bayesian'];

  const state = {
    landscapeKey: 'Bumpy',
    trials: 20,
    // Per-strategy probe arrays so all three race side by side.
    points: { grid: [], random: [], bayesian: [] },
    // Default to mid-race (12/20 trials) so the three strategies are visibly different
    // on first paint: grid is methodically marching across the lattice, random is
    // scattered, Bayesian has clustered its later picks around the discovered peak.
    // Instructive without needing to press Run.
    cursor: 12,
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

  // Pre-existing strategy buttons in the HTML are irrelevant for the racing layout, but
  // we keep them for "All / focus" filtering — clicking selects which panel to highlight.
  state.focus = 'all';
  ['All three', 'Grid', 'Random', 'Bayesian'].forEach((lbl) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = lbl;
    b.dataset.focus = lbl === 'All three' ? 'all' : lbl.toLowerCase();
    b.addEventListener('click', () => { state.focus = b.dataset.focus; render(); });
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
    reset(); render();
  });
  runBtn.addEventListener('click', () => { if (state.running) stopRun(); else startRun(); });
  resetBtn.addEventListener('click', () => { reset(); render(); });

  function reset() {
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    state.points = { grid: [], random: [], bayesian: [] };
    state.cursor = 0;
    state.running = false;
    runBtn.textContent = '▷ Run search';
  }

  // Mulberry32 — deterministic per (strategy, landscape, trials) so the race is fair on resize.
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Pre-compute all probes for a given strategy up to `trials`.
  // This way the three panels are deterministic — same landscape, same trial count → same trail.
  function precomputeAll() {
    const seedKey = (state.landscapeKey + state.trials).split('').reduce((s, c) => s + c.charCodeAt(0), 0);
    const f = LANDSCAPES[state.landscapeKey];
    const result = { grid: [], random: [], bayesian: [] };

    // Grid
    {
      const n = Math.ceil(Math.sqrt(state.trials));
      for (let i = 0; i < state.trials; i++) {
        const r = Math.floor(i / n);
        const c = i % n;
        const x = (c + 0.5) / n;
        const y = (r + 0.5) / n;
        result.grid.push({ x, y, score: f(x, y), order: i });
      }
    }

    // Random
    {
      const rng = mulberry32(seedKey + 13);
      for (let i = 0; i < state.trials; i++) {
        const x = rng(), y = rng();
        result.random.push({ x, y, score: f(x, y), order: i });
      }
    }

    // Bayesian-ish: first 4 random, then exploit + slight noise around best so far + occasionally explore.
    {
      const rng = mulberry32(seedKey + 27);
      const points = [];
      for (let i = 0; i < state.trials; i++) {
        if (points.length < 4) {
          const x = rng(), y = rng();
          points.push({ x, y, score: f(x, y), order: i });
          continue;
        }
        // Pick the candidate that maximizes a tiny EI-ish acquisition (mean + uncertainty bonus).
        const candidates = [];
        for (let k = 0; k < 30; k++) candidates.push([rng(), rng()]);
        const best = points.reduce((m, p) => Math.max(m, p.score), 0);
        let chosen = candidates[0], bestEI = -Infinity;
        for (const [cx, cy] of candidates) {
          let num = 0, den = 0, minDist = Infinity;
          for (const p of points) {
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
        points.push({ x: chosen[0], y: chosen[1], score: f(chosen[0], chosen[1]), order: i });
      }
      result.bayesian = points;
    }

    return result;
  }

  let allProbes = precomputeAll();

  function startRun() {
    if (state.cursor >= state.trials) reset();
    if (state.cursor === 0) allProbes = precomputeAll();
    state.running = true;
    runBtn.textContent = '⏸ Pause';
    state.timer = setInterval(() => {
      if (state.cursor >= state.trials) { stopRun(); return; }
      // advance one trial in each strategy in parallel
      STRATEGIES.forEach((s) => state.points[s].push(allProbes[s][state.cursor]));
      state.cursor++;
      render();
    }, 200);
  }
  function stopRun() {
    state.running = false;
    runBtn.textContent = state.cursor >= state.trials ? '▷ Run search' : '▷ Resume';
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  // Recompute probes if landscape/trials change between runs.
  // Already triggered by reset() in event handlers; precompute here for first render.
  function ensureProbes() {
    allProbes = precomputeAll();
  }
  ensureProbes();

  // Populate the initial mid-race probes so the default render shows all three
  // strategies already in motion (see comment on state.cursor above).
  STRATEGIES.forEach((s) => {
    state.points[s] = allProbes[s].slice(0, state.cursor);
  });
  // Match the button label to the mid-race state — "Resume" is more honest than
  // "Run search" once trials are already on the board.
  if (runBtn) runBtn.textContent = '▷ Resume';

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    const padX = 18;
    const titleH = 22;
    const heatTop = titleH + 18;
    // Three side-by-side panels for the three landscapes
    const panelGap = 10;
    const totalPanelsW = size.w - padX * 2;
    const panelW = (totalPanelsW - panelGap * 2) / 3;
    const heatH = panelW; // square heatmaps
    // Bottom: best-score-vs-GPU-time plot (full width)
    const bottomTop = heatTop + heatH + 56; // room for budget bar + title under each panel
    const bottomH = size.h - bottomTop - 30;

    // Title
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText(`three search strategies racing on the same landscape · ${state.landscapeKey}`, padX, 14);

    // ─── Color-scale legend (top-right): low score → high score gradient ──
    // Newcomers need to know oxblood means "good" without reading the caption.
    {
      const legW = 110, legH = 8;
      const legX = size.w - padX - legW;
      const legY = 6;
      const steps = 24;
      for (let s = 0; s < steps; s++) {
        const t = s / (steps - 1);
        const r = Math.round(247 - t * 130);
        const g = Math.round(244 - t * 195);
        const b = Math.round(236 - t * 195);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(legX + (s * legW) / steps, legY, legW / steps + 0.5, legH);
      }
      ctx.strokeStyle = 'rgba(38,35,32,0.30)';
      ctx.lineWidth = 0.6;
      ctx.strokeRect(legX, legY, legW, legH);
      ctx.fillStyle = INK_FADE;
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText('low f(θ)', legX - 4, legY + legH);
      ctx.textAlign = 'left';
      ctx.fillText('high f(θ)', legX + legW + 4, legY + legH);
    }

    STRATEGIES.forEach((strat, i) => {
      const x0 = padX + i * (panelW + panelGap);
      const y0 = heatTop;
      drawHeatPanel(ctx, x0, y0, panelW, heatH, strat);
    });

    // ─── Mid-race storytelling annotation ───
    // When the race is partway through (default landing state and during a run),
    // surface the headline story so the visual pattern is named, not just shown.
    if (state.cursor > 0 && state.cursor < state.trials) {
      const bestOf = (s) => state.points[s].length
        ? state.points[s].reduce((m, p) => Math.max(m, p.score), 0) : 0;
      const ranking = STRATEGIES
        .map((s) => ({ s, b: bestOf(s) }))
        .sort((a, b) => b.b - a.b);
      const lead = ranking[0];
      ctx.fillStyle = 'rgba(38,35,32,0.65)';
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'left';
      ctx.fillText(
        `${state.cursor}/${state.trials} trials in — ${lead.s} leading at ${formatNum(lead.b)}; watch how each method spends the remaining budget.`,
        padX, bottomTop - 18
      );
    }

    drawBottomPlot(ctx, padX, bottomTop, size.w - padX * 2, bottomH);
  }

  function drawHeatPanel(ctx, x0, y0, W, H, strat) {
    const f = LANDSCAPES[state.landscapeKey];
    const dim = state.focus === 'all' || state.focus === strat ? 1 : 0.3;
    ctx.save();
    ctx.globalAlpha = dim;

    // Heatmap
    const cells = 36;
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
    state.points[strat].forEach((p, i) => {
      const px = x0 + p.x * W;
      const py = y0 + (1 - p.y) * H;
      const isLatest = i === state.points[strat].length - 1 && state.running;
      ctx.fillStyle = isLatest ? ACCENT : '#262320';
      ctx.beginPath();
      ctx.arc(px, py, isLatest ? 4 : 2.6, 0, Math.PI * 2);
      ctx.fill();
    });
    // Best so far ring
    if (state.points[strat].length > 0) {
      const best = state.points[strat].reduce((b, p) => p.score > b.score ? p : b);
      const px = x0 + best.x * W;
      const py = y0 + (1 - best.y) * H;
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.arc(px, py, 7, 0, Math.PI * 2);
      ctx.stroke();
    }

    ctx.restore();

    // Panel label
    ctx.fillStyle = INK;
    ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    const niceLabel = { grid: 'grid · lattice', random: 'random · scattered', bayesian: 'Bayesian · smart' }[strat];
    ctx.fillText(niceLabel, x0, y0 - 6);

    // Axis labels: each heatmap is 2D θ-space (θ₁ horizontal, θ₂ vertical) — only
    // label the leftmost panel (where x0 is smallest) to avoid clutter.
    if (strat === 'grid') {
      ctx.fillStyle = INK_FADE;
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      ctx.fillText('θ₁ (e.g., learning rate) →', x0 + 2, y0 + H - 4);
      ctx.save();
      ctx.translate(x0 + 10, y0 + H - 6);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('θ₂ (e.g., weight decay) →', 0, 0);
      ctx.restore();
    }

    // Budget bar
    const barY = y0 + H + 10;
    const barH = 8;
    ctx.fillStyle = 'rgba(38,35,32,0.10)';
    ctx.fillRect(x0, barY, W, barH);
    const minutesUsed = state.points[strat].length * MIN_PER_TRIAL[strat];
    const minutesBudget = state.trials * Math.max(...Object.values(MIN_PER_TRIAL));
    const fillW = Math.min(1, minutesUsed / minutesBudget) * W;
    ctx.fillStyle = ACCENT;
    ctx.fillRect(x0, barY, fillW, barH);
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 0.6;
    ctx.strokeRect(x0, barY, W, barH);
    // Numbers
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`${state.points[strat].length}/${state.trials} trials`, x0, barY + barH + 12);
    ctx.textAlign = 'right';
    ctx.fillText(`${minutesUsed.toFixed(1)} GPU·min`, x0 + W, barY + barH + 12);
  }

  function drawBottomPlot(ctx, x0, y0, W, H) {
    // best-score over GPU minutes (instead of trial count) — Bayesian's compute-saving shows up here.
    const plotX0 = x0 + 50;
    const plotW = W - 50 - 10;
    const plotH = H - 16;

    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plotX0, y0); ctx.lineTo(plotX0, y0 + plotH);
    ctx.lineTo(plotX0 + plotW, y0 + plotH);
    ctx.stroke();

    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('best score so far · vs · GPU minutes spent', plotX0, y0 - 6);

    // Y axis
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    [0, 0.5, 1].forEach((y) => {
      ctx.fillText(y.toFixed(1), plotX0 - 6, y0 + plotH - y * plotH + 3);
    });

    // X axis: max minutes spent across strategies (so curves share scale)
    const maxMinutes = state.trials * Math.max(...Object.values(MIN_PER_TRIAL));
    ctx.textAlign = 'center';
    [0, 0.5, 1].forEach((tx) => {
      ctx.fillText((tx * maxMinutes).toFixed(0) + ' min', plotX0 + tx * plotW, y0 + plotH + 14);
    });

    const colorOf = { grid: '#262320', random: '#3a6b5e', bayesian: ACCENT };

    STRATEGIES.forEach((strat) => {
      const dim = state.focus === 'all' || state.focus === strat ? 1 : 0.25;
      ctx.save();
      ctx.globalAlpha = dim;
      ctx.strokeStyle = colorOf[strat];
      ctx.lineWidth = 2;
      ctx.beginPath();
      let best = 0;
      let prevX = plotX0, prevY = y0 + plotH;
      ctx.moveTo(prevX, prevY);
      state.points[strat].forEach((p, i) => {
        if (p.score > best) best = p.score;
        const x = plotX0 + ((i + 1) * MIN_PER_TRIAL[strat] / maxMinutes) * plotW;
        const y = y0 + plotH - best * plotH;
        // step curve (horizontal then vertical)
        ctx.lineTo(x, prevY);
        ctx.lineTo(x, y);
        prevX = x; prevY = y;
      });
      ctx.stroke();

      // End label
      if (state.points[strat].length > 0) {
        ctx.fillStyle = colorOf[strat];
        ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
        ctx.textAlign = 'left';
        ctx.fillText(strat, prevX + 4, prevY + 3);
      }
      ctx.restore();
    });
  }

  function render() {
    Array.from(strategiesEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.focus === state.focus);
    });
    Array.from(landscapesEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.landscape === state.landscapeKey);
    });
    trialsLabel.textContent = `${state.trials} trials`;
    trialsSlider.value = state.trials;
    const bestOf = (s) => state.points[s].length ? state.points[s].reduce((m, p) => Math.max(m, p.score), 0) : 0;
    readoutEl.innerHTML = `
      <div class="row"><span>trials run</span><b>${state.cursor} / ${state.trials}</b></div>
      <div class="row"><span>grid · best</span><b>${formatNum(bestOf('grid'))}</b></div>
      <div class="row"><span>random · best</span><b>${formatNum(bestOf('random'))}</b></div>
      <div class="row"><span>bayesian · best</span><b>${formatNum(bestOf('bayesian'))}</b></div>`;
    canvasCtl.redraw();
  }

  render();
})();
