/* Vol. XXIX — Multi-Armed Bandits */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK_FADE } = V;

  const NUM_ARMS = 10;
  // True means hand-selected so the gap structure is interesting.
  const TRUE_MEANS = [0.20, 0.55, 0.40, 0.65, 0.45, 0.80, 0.35, 0.50, 0.70, 0.30];
  const BEST_MEAN = Math.max(...TRUE_MEANS);

  function pullArm(i) {
    // Bernoulli with p = TRUE_MEANS[i]
    return Math.random() < TRUE_MEANS[i] ? 1 : 0;
  }

  const state = {
    strategy: 'eps',
    eps: 0.1,
    speed: 50,
    pulls: new Array(NUM_ARMS).fill(0),
    sums: new Array(NUM_ARMS).fill(0),
    // For Thompson sampling: alpha/beta of Beta posterior
    alpha: new Array(NUM_ARMS).fill(1),
    beta:  new Array(NUM_ARMS).fill(1),
    t: 0,
    cumulativeRegret: [0],
    running: false,
    timer: null,
  };

  function reset() {
    state.pulls = new Array(NUM_ARMS).fill(0);
    state.sums = new Array(NUM_ARMS).fill(0);
    state.alpha = new Array(NUM_ARMS).fill(1);
    state.beta = new Array(NUM_ARMS).fill(1);
    state.t = 0;
    state.cumulativeRegret = [0];
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    state.running = false;
    runBtn.textContent = '▷ Run';
  }

  function pickArm() {
    if (state.strategy === 'eps') {
      if (Math.random() < state.eps) return Math.floor(Math.random() * NUM_ARMS);
      // Greedy with optimistic initialization for unsampled arms.
      let best = -Infinity, ix = 0;
      for (let i = 0; i < NUM_ARMS; i++) {
        const m = state.pulls[i] === 0 ? 1 : state.sums[i] / state.pulls[i];
        if (m > best) { best = m; ix = i; }
      }
      return ix;
    } else if (state.strategy === 'ucb') {
      // Pull each arm once first
      for (let i = 0; i < NUM_ARMS; i++) if (state.pulls[i] === 0) return i;
      let best = -Infinity, ix = 0;
      for (let i = 0; i < NUM_ARMS; i++) {
        const mean = state.sums[i] / state.pulls[i];
        const bonus = Math.sqrt(2 * Math.log(state.t + 1) / state.pulls[i]);
        const ucb = mean + bonus;
        if (ucb > best) { best = ucb; ix = i; }
      }
      return ix;
    } else {
      // Thompson sampling
      let best = -Infinity, ix = 0;
      for (let i = 0; i < NUM_ARMS; i++) {
        // Sample from Beta(alpha, beta) using rejection — for small parameters use approx via gamma ratio.
        // Use simple "random uniform with mean = alpha/(alpha+beta)" as a quick approximation,
        // then add a small noise. This is approximate but visually correct.
        const a = state.alpha[i], b = state.beta[i];
        const mean = a / (a + b);
        const stddev = Math.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)));
        const sample = mean + stddev * gaussian();
        if (sample > best) { best = sample; ix = sample; ix = i; }
      }
      return ix;
    }
  }
  function gaussian() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function step() {
    const arm = pickArm();
    const r = pullArm(arm);
    state.pulls[arm]++;
    state.sums[arm] += r;
    if (r === 1) state.alpha[arm]++; else state.beta[arm]++;
    state.t++;
    const lastReg = state.cumulativeRegret[state.cumulativeRegret.length - 1];
    state.cumulativeRegret.push(lastReg + (BEST_MEAN - TRUE_MEANS[arm]));
    if (state.cumulativeRegret.length > 600) state.cumulativeRegret.shift();
  }

  const strategiesEl = document.getElementById('strategies');
  const epsSlider = document.getElementById('eps-slider');
  const epsLabel = document.getElementById('eps-label');
  const speedSlider = document.getElementById('speed-slider');
  const speedLabel = document.getElementById('speed-label');
  const runBtn = document.getElementById('run-btn');
  const resetBtn = document.getElementById('reset-btn');
  const readoutEl = document.getElementById('readout');
  const canvas = document.getElementById('canvas');

  ['ε-greedy', 'UCB', 'Thompson'].forEach((lbl, i) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = lbl;
    b.dataset.strategy = ['eps', 'ucb', 'thompson'][i];
    b.addEventListener('click', () => {
      state.strategy = b.dataset.strategy;
      reset();
      render();
    });
    strategiesEl.appendChild(b);
  });

  epsSlider.addEventListener('input', (e) => {
    state.eps = +e.target.value;
    epsLabel.textContent = `ε = ${state.eps.toFixed(2)}`;
  });
  speedSlider.addEventListener('input', (e) => {
    state.speed = +e.target.value;
    speedLabel.textContent = `${state.speed} pulls / s`;
    if (state.running) { stopRun(); startRun(); }
  });
  runBtn.addEventListener('click', () => { if (state.running) stopRun(); else startRun(); });
  resetBtn.addEventListener('click', () => { reset(); render(); });

  function startRun() {
    state.running = true;
    runBtn.textContent = '⏸ Pause';
    const interval = Math.max(5, 1000 / state.speed);
    state.timer = setInterval(() => { step(); render(); }, interval);
  }
  function stopRun() {
    state.running = false;
    runBtn.textContent = '▷ Run';
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    const padX = 24, padTop = 24;
    const armPaneH = size.h * 0.62;
    const regretPaneY = size.h * 0.70;
    const regretPaneH = size.h * 0.25;

    const colW = (size.w - padX * 2) / NUM_ARMS;
    const maxPulls = Math.max(1, ...state.pulls);

    // Arms — for each, draw bar (pulls) and tick (estimated mean) with dashed true-mean line
    for (let i = 0; i < NUM_ARMS; i++) {
      const x0 = padX + i * colW;
      const w = colW * 0.7;
      const x = x0 + (colW - w) / 2;

      const trueMean = TRUE_MEANS[i];
      const estMean = state.pulls[i] > 0 ? state.sums[i] / state.pulls[i] : 0;
      const pullFrac = state.pulls[i] / maxPulls;

      const barTop = padTop + (1 - pullFrac) * armPaneH * 0.45;
      const barBot = padTop + armPaneH * 0.55;
      // Pulls bar
      ctx.fillStyle = 'rgba(38,35,32,0.18)';
      ctx.fillRect(x, barTop, w, barBot - barTop);
      ctx.strokeStyle = 'rgba(38,35,32,0.30)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x, barTop, w, barBot - barTop);

      // True mean line (dashed across the column)
      const trueY = padTop + armPaneH - trueMean * (armPaneH - 24);
      ctx.strokeStyle = 'rgba(38,35,32,0.40)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x - 4, trueY);
      ctx.lineTo(x + w + 4, trueY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Estimated mean tick (orange)
      const estY = padTop + armPaneH - estMean * (armPaneH - 24);
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(x - 3, estY);
      ctx.lineTo(x + w + 3, estY);
      ctx.stroke();

      // Arm label
      ctx.fillStyle = INK_FADE;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`arm ${i}`, x + w / 2, padTop + armPaneH + 12);
      ctx.fillText(`${state.pulls[i]}p`, x + w / 2, padTop + armPaneH + 24);
    }

    // Y-axis labels (mean reward)
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    [0, 0.5, 1].forEach((m) => {
      const y = padTop + armPaneH - m * (armPaneH - 24);
      ctx.fillText(m.toFixed(1), 4, y + 3);
    });

    // Regret pane
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('cumulative regret over time', padX, regretPaneY - 6);

    const reg = state.cumulativeRegret;
    if (reg.length > 1) {
      const maxReg = Math.max(...reg, 1);
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      reg.forEach((v, i) => {
        const x = padX + (i / Math.max(1, reg.length - 1)) * (size.w - padX * 2);
        const y = regretPaneY + regretPaneH - (v / maxReg) * regretPaneH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      // Label
      ctx.fillStyle = INK_FADE;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`R = ${formatNum(reg[reg.length - 1])}`, size.w - padX, regretPaneY + 12);
    }
    // Frame
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.strokeRect(padX, regretPaneY, size.w - padX * 2, regretPaneH);
  }

  function render() {
    Array.from(strategiesEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.strategy === state.strategy);
    });
    epsSlider.value = state.eps;
    speedSlider.value = state.speed;
    epsLabel.textContent = `ε = ${state.eps.toFixed(2)}`;
    speedLabel.textContent = `${state.speed} pulls / s`;
    const totalPulls = state.pulls.reduce((a, b) => a + b, 0);
    const totalReward = state.sums.reduce((a, b) => a + b, 0);
    const bestArm = TRUE_MEANS.indexOf(BEST_MEAN);
    const optimalFrac = totalPulls === 0 ? 0 : state.pulls[bestArm] / totalPulls;
    readoutEl.innerHTML = `
      <div class="row"><span>pulls</span><b>${totalPulls}</b></div>
      <div class="row"><span>reward</span><b>${formatNum(totalReward)}</b></div>
      <div class="row"><span>optimal arm</span><b>arm ${bestArm} (${BEST_MEAN.toFixed(2)})</b></div>
      <div class="row"><span>% optimal</span><b>${(optimalFrac * 100).toFixed(0)}%</b></div>
      <div class="row"><span>regret</span><b>${formatNum(state.cumulativeRegret[state.cumulativeRegret.length - 1])}</b></div>`;
    canvasCtl.redraw();
  }

  render();
})();
