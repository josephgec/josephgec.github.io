/* Vol. XXIX — Multi-Armed Bandits */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK, INK_FADE } = V;

  const NUM_ARMS = 10;
  // True means hand-selected so the gap structure is interesting.
  const TRUE_MEANS = [0.20, 0.55, 0.40, 0.65, 0.45, 0.80, 0.35, 0.50, 0.70, 0.30];
  const BEST_MEAN = Math.max(...TRUE_MEANS);
  const BEST_ARM = TRUE_MEANS.indexOf(BEST_MEAN);

  function pull(i) { return Math.random() < TRUE_MEANS[i] ? 1 : 0; }
  function gaussian() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // Per-strategy state. Each tracks pulls/sums/alpha/beta independently.
  function makeAgent() {
    return {
      pulls: new Array(NUM_ARMS).fill(0),
      sums:  new Array(NUM_ARMS).fill(0),
      alpha: new Array(NUM_ARMS).fill(1),
      beta:  new Array(NUM_ARMS).fill(1),
      cumRegret: [0],
      lastArm: -1,
      lastReward: 0,
      lastExploring: false, // for ε-greedy
    };
  }

  // Strategy implementations. Each returns the chosen arm index.
  function pickEps(agent, eps, t) {
    if (Math.random() < eps) {
      agent.lastExploring = true;
      return Math.floor(Math.random() * NUM_ARMS);
    }
    agent.lastExploring = false;
    let best = -Infinity, ix = 0;
    for (let i = 0; i < NUM_ARMS; i++) {
      const m = agent.pulls[i] === 0 ? 1.0 : agent.sums[i] / agent.pulls[i];
      if (m > best) { best = m; ix = i; }
    }
    return ix;
  }
  function pickUCB(agent, t) {
    for (let i = 0; i < NUM_ARMS; i++) if (agent.pulls[i] === 0) return i;
    let best = -Infinity, ix = 0;
    for (let i = 0; i < NUM_ARMS; i++) {
      const mean = agent.sums[i] / agent.pulls[i];
      const bonus = Math.sqrt(2 * Math.log(t + 1) / agent.pulls[i]);
      const v = mean + bonus;
      if (v > best) { best = v; ix = i; }
    }
    return ix;
  }
  function pickThompson(agent) {
    let best = -Infinity, ix = 0;
    for (let i = 0; i < NUM_ARMS; i++) {
      const a = agent.alpha[i], b = agent.beta[i];
      const mean = a / (a + b);
      const stddev = Math.sqrt((a * b) / ((a + b) * (a + b) * (a + b + 1)));
      const sample = mean + stddev * gaussian();
      if (sample > best) { best = sample; ix = i; }
    }
    return ix;
  }

  function applyPull(agent, arm, t) {
    const r = pull(arm);
    agent.pulls[arm]++;
    agent.sums[arm] += r;
    if (r === 1) agent.alpha[arm]++; else agent.beta[arm]++;
    agent.lastArm = arm;
    agent.lastReward = r;
    const lastReg = agent.cumRegret[agent.cumRegret.length - 1];
    agent.cumRegret.push(lastReg + (BEST_MEAN - TRUE_MEANS[arm]));
    if (agent.cumRegret.length > 1200) agent.cumRegret.shift();
  }

  // ───────── Views ─────────
  const VIEWS = [
    { key: 'race', label: 'Race · all 3 head-to-head' },
    { key: 'eps',  label: 'ε-greedy · explore vs exploit' },
    { key: 'ucb',  label: 'UCB · confidence bonus' },
    { key: 'thomp', label: 'Thompson · Beta posteriors' },
  ];
  const CAPTIONS = {
    race:  'All three strategies pulling from the same 10 arms in parallel. The right panel shows their cumulative-regret curves head-to-head — UCB and Thompson typically beat ε-greedy, especially as time goes on.',
    eps:   'ε-greedy: most of the time pull the arm with the highest estimated mean (greedy = orange flash). With probability ε, ignore that and pull a random arm (explore = gray flash). Simple. Surprisingly hard to beat at short horizons.',
    ucb:   'UCB ("upper confidence bound"): above each arm\'s estimated mean (orange tick), draw a vertical "confidence bonus" segment — large for arms pulled few times, small for well-known ones. UCB pulls whichever arm\'s tick + bonus reaches highest. The bonus shrinks visibly as you sample.',
    thomp: 'Thompson sampling: each arm has a Beta posterior over its true rate (the bell curve above its column). Wide bells = uncertain → likely to throw a high sample → likely to be pulled. Bells narrow around the truth as evidence accumulates — exploration becomes exploitation automatically.',
  };

  const COLORS = {
    eps:   { name: 'ε-greedy',  color: 'rgba(58,107,74,0.85)',  rgb: '58,107,74' },   // green
    ucb:   { name: 'UCB',        color: 'rgba(122,31,36,0.85)',  rgb: '122,31,36' },   // accent
    thomp: { name: 'Thompson',   color: 'rgba(38,35,32,0.85)',   rgb: '38,35,32' },    // ink
  };

  const state = {
    view: 'race',
    eps: 0.1,
    speed: 50,
    t: 0,
    running: false,
    timer: null,
    agents: { eps: makeAgent(), ucb: makeAgent(), thomp: makeAgent() },
    flashTs: 0, // last pull, for animation
  };

  function reset() {
    state.t = 0;
    state.agents = { eps: makeAgent(), ucb: makeAgent(), thomp: makeAgent() };
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    state.running = false;
    runBtn.textContent = '▷ Run race';
  }

  function step() {
    state.t++;
    // ε-greedy
    const ae = pickEps(state.agents.eps, state.eps, state.t);
    applyPull(state.agents.eps, ae, state.t);
    // UCB
    const au = pickUCB(state.agents.ucb, state.t);
    applyPull(state.agents.ucb, au, state.t);
    // Thompson
    const at = pickThompson(state.agents.thomp);
    applyPull(state.agents.thomp, at, state.t);
    state.flashTs = performance.now();
  }

  // ───────── DOM ─────────
  const strategiesEl = document.getElementById('strategies');
  const epsSlider = document.getElementById('eps-slider');
  const epsLabel = document.getElementById('eps-label');
  const speedSlider = document.getElementById('speed-slider');
  const speedLabel = document.getElementById('speed-label');
  const runBtn = document.getElementById('run-btn');
  const resetBtn = document.getElementById('reset-btn');
  const readoutEl = document.getElementById('readout');
  const captionEl = document.getElementById('caption-text');
  const canvas = document.getElementById('canvas');

  VIEWS.forEach((v) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = v.label;
    b.dataset.view = v.key;
    b.addEventListener('click', () => { state.view = v.key; render(); });
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
    runBtn.textContent = '▷ Run race';
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    if (state.view === 'race') {
      drawRace(ctx, size);
    } else if (state.view === 'eps') {
      drawSingle(ctx, size, 'eps');
    } else if (state.view === 'ucb') {
      drawSingle(ctx, size, 'ucb');
    } else {
      drawSingle(ctx, size, 'thomp');
    }
  }

  // ───────── Race view: arms grid (top) + 3-line regret chart (bottom) ─────────
  function drawRace(ctx, size) {
    const padX = 24;
    const headerH = 18;
    const armPaneH = size.h * 0.50;
    const armPaneY = headerH;
    const regretPaneY = headerH + armPaneH + 30;
    const regretPaneH = size.h - regretPaneY - 12;

    // Header
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('arms · pull counts (bars), true means (dashed), each agent\'s estimate (color tick)', size.w / 2, 4);

    const colW = (size.w - padX * 2) / NUM_ARMS;
    const maxPulls = Math.max(1,
      ...state.agents.eps.pulls,
      ...state.agents.ucb.pulls,
      ...state.agents.thomp.pulls);

    for (let i = 0; i < NUM_ARMS; i++) {
      const x0 = padX + i * colW;
      const w = colW * 0.78;
      const x = x0 + (colW - w) / 2;

      // Background of column for best arm
      if (i === BEST_ARM) {
        ctx.fillStyle = 'rgba(122,31,36,0.04)';
        ctx.fillRect(x0 + 2, armPaneY, colW - 4, armPaneH);
      }

      // Pull-count bars: three side-by-side mini-bars under the column,
      // one per agent, height = pulls / maxPulls.
      const sub = w / 3;
      const subPadY = armPaneH * 0.50;
      const subBaseY = armPaneY + armPaneH - 4;
      const subTopY = armPaneY + subPadY;
      const subH = subBaseY - subTopY;
      ['eps', 'ucb', 'thomp'].forEach((k, ai) => {
        const bx = x + ai * sub + sub * 0.10;
        const bw = sub * 0.80;
        const p = state.agents[k].pulls[i];
        const h = (p / maxPulls) * subH;
        ctx.fillStyle = COLORS[k].color.replace('0.85', '0.55');
        ctx.fillRect(bx, subBaseY - h, bw, h);
        ctx.strokeStyle = COLORS[k].color.replace('0.85', '0.85');
        ctx.lineWidth = 0.8;
        ctx.strokeRect(bx, subBaseY - h, bw, h);
      });

      // True mean dashed line (in the upper portion)
      const upperTopY = armPaneY + 4;
      const upperBotY = armPaneY + subPadY - 4;
      const meanH = upperBotY - upperTopY;
      const trueMean = TRUE_MEANS[i];
      const trueY = upperBotY - trueMean * meanH;
      ctx.strokeStyle = 'rgba(38,35,32,0.45)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x - 2, trueY); ctx.lineTo(x + w + 2, trueY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Each agent's estimated mean as a colored tick at slightly different x offsets
      ['eps', 'ucb', 'thomp'].forEach((k, ai) => {
        const ag = state.agents[k];
        if (ag.pulls[i] === 0) return;
        const m = ag.sums[i] / ag.pulls[i];
        const y = upperBotY - m * meanH;
        const tx0 = x + (ai * w / 3) + 2;
        const tx1 = tx0 + w / 3 - 4;
        ctx.strokeStyle = COLORS[k].color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(tx0, y); ctx.lineTo(tx1, y);
        ctx.stroke();
      });

      // Mark "last pulled" with a small dot above the column for whichever agent just picked it
      const flashAlpha = Math.max(0, 1 - (performance.now() - state.flashTs) / 250);
      ['eps', 'ucb', 'thomp'].forEach((k, ai) => {
        if (state.agents[k].lastArm === i && flashAlpha > 0) {
          const cx = x + (ai * w / 3) + (w / 6);
          ctx.fillStyle = COLORS[k].color.replace('0.85', String(0.85 * flashAlpha));
          ctx.beginPath();
          ctx.arc(cx, armPaneY - 2, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      // Arm label
      ctx.fillStyle = INK_FADE;
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(`a${i}`, x + w / 2, armPaneY + armPaneH + 2);
      if (i === BEST_ARM) {
        ctx.fillStyle = ACCENT;
        ctx.font = 'italic 9px "Source Serif 4", Georgia, serif';
        ctx.fillText('best', x + w / 2, armPaneY + armPaneH + 13);
      }
    }

    // Y-axis labels (mean reward)
    ctx.fillStyle = INK_FADE;
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    [0, 0.5, 1].forEach((m) => {
      const upperTopY = armPaneY + 4;
      const upperBotY = armPaneY + armPaneH * 0.50 - 4;
      const meanH = upperBotY - upperTopY;
      const y = upperBotY - m * meanH;
      ctx.fillText(m.toFixed(1), 4, y);
    });

    // ───── Regret pane ─────
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.strokeRect(padX, regretPaneY, size.w - padX * 2, regretPaneH);
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('cumulative regret over time · lower = better', padX, regretPaneY - 16);

    const allCurves = [state.agents.eps.cumRegret, state.agents.ucb.cumRegret, state.agents.thomp.cumRegret];
    const maxReg = Math.max(1, ...allCurves.flatMap((a) => a));
    ['eps', 'ucb', 'thomp'].forEach((k) => {
      const reg = state.agents[k].cumRegret;
      ctx.strokeStyle = COLORS[k].color;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      reg.forEach((v, i) => {
        const x = padX + 4 + (i / Math.max(1, reg.length - 1)) * (size.w - padX * 2 - 8);
        const y = regretPaneY + regretPaneH - 4 - (v / maxReg) * (regretPaneH - 8);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });

    // Legend (top-left of regret pane)
    let lx = padX + 8;
    const ly = regretPaneY + 12;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textBaseline = 'middle';
    ['eps', 'ucb', 'thomp'].forEach((k) => {
      ctx.fillStyle = COLORS[k].color;
      ctx.fillRect(lx, ly - 5, 12, 2);
      ctx.fillStyle = INK_FADE;
      ctx.textAlign = 'left';
      const reg = state.agents[k].cumRegret;
      const last = reg[reg.length - 1];
      const txt = `${COLORS[k].name}  R=${last.toFixed(1)}`;
      ctx.fillText(txt, lx + 16, ly);
      lx += ctx.measureText(txt).width + 36;
    });
  }

  // ───────── Single-strategy view ─────────
  function drawSingle(ctx, size, key) {
    const ag = state.agents[key];
    const padX = 24;
    const headerH = 26;
    const armPaneY = headerH;
    const armPaneH = size.h * 0.62;
    const regretPaneY = armPaneY + armPaneH + 32;
    const regretPaneH = size.h - regretPaneY - 12;

    // Header
    ctx.fillStyle = COLORS[key].color;
    ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(COLORS[key].name + ' — focused view', padX, 6);

    // ε-greedy: show last action's tag (explore vs exploit)
    if (key === 'eps') {
      const tag = ag.lastExploring ? 'EXPLORE · random arm chosen' : 'EXPLOIT · greedy on max estimate';
      const tagColor = ag.lastExploring ? 'rgba(38,35,32,0.7)' : COLORS.eps.color;
      ctx.fillStyle = tagColor;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(tag, size.w - padX, 8);
    }

    const colW = (size.w - padX * 2) / NUM_ARMS;
    const maxPulls = Math.max(1, ...ag.pulls);

    const upperTopY = armPaneY + 8;
    const upperBotY = armPaneY + armPaneH * 0.55 - 4;
    const meanH = upperBotY - upperTopY;
    const subBaseY = armPaneY + armPaneH - 4;
    const subTopY  = upperBotY + 12;
    const subH = subBaseY - subTopY;

    for (let i = 0; i < NUM_ARMS; i++) {
      const x0 = padX + i * colW;
      const w = colW * 0.74;
      const x = x0 + (colW - w) / 2;
      const cx = x + w / 2;

      // best-arm tint
      if (i === BEST_ARM) {
        ctx.fillStyle = 'rgba(122,31,36,0.04)';
        ctx.fillRect(x0 + 1, armPaneY, colW - 2, armPaneH);
      }

      const trueMean = TRUE_MEANS[i];
      const trueY = upperBotY - trueMean * meanH;
      // Pull-count bar
      const p = ag.pulls[i];
      const bh = (p / maxPulls) * subH;
      const isLastPicked = ag.lastArm === i;
      ctx.fillStyle = isLastPicked ? COLORS[key].color.replace('0.85', '0.50') : 'rgba(38,35,32,0.18)';
      ctx.fillRect(x, subBaseY - bh, w, bh);
      ctx.strokeStyle = 'rgba(38,35,32,0.30)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x, subBaseY - bh, w, bh);

      // True mean dashed line
      ctx.strokeStyle = 'rgba(38,35,32,0.45)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x - 4, trueY); ctx.lineTo(x + w + 4, trueY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Estimated mean tick
      const estMean = p > 0 ? ag.sums[i] / p : 0;
      const estY = upperBotY - estMean * meanH;
      if (p > 0) {
        ctx.strokeStyle = COLORS[key].color;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(x - 3, estY); ctx.lineTo(x + w + 3, estY);
        ctx.stroke();
      }

      // ─── UCB-specific: confidence-bonus segment above the tick ───
      if (key === 'ucb' && p > 0) {
        const bonus = Math.sqrt(2 * Math.log(state.t + 1) / p);
        const ucbY = upperBotY - Math.min(1, estMean + bonus) * meanH;
        // vertical segment from estY (down) up to ucbY
        ctx.strokeStyle = 'rgba(122,31,36,0.45)';
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(cx, estY);
        ctx.lineTo(cx, ucbY);
        ctx.stroke();
        // small cap at top
        ctx.strokeStyle = 'rgba(122,31,36,0.85)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(cx - 5, ucbY); ctx.lineTo(cx + 5, ucbY);
        ctx.stroke();
        // bonus value annotation for the picked arm
        if (isLastPicked) {
          ctx.fillStyle = ACCENT;
          ctx.font = '9px "JetBrains Mono", monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'bottom';
          ctx.fillText(`+${bonus.toFixed(2)}`, cx, ucbY - 2);
        }
      }

      // ─── Thompson-specific: Beta posterior bell curve drawn above the column ───
      if (key === 'thomp') {
        const a = ag.alpha[i], b = ag.beta[i];
        // Draw the Beta(a, b) curve over the upper region [0,1] mapped to the same vertical axis
        const samples = 32;
        const bellMaxH = meanH * 0.92;
        // Find peak for normalization
        let peak = 0;
        const ys = new Array(samples);
        for (let s = 0; s < samples; s++) {
          const u = (s + 0.5) / samples;
          const v = Math.pow(u, a - 1) * Math.pow(1 - u, b - 1);
          ys[s] = v;
          if (v > peak) peak = v;
        }
        if (peak > 0) {
          ctx.strokeStyle = COLORS.thomp.color.replace('0.85', '0.55');
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          for (let s = 0; s < samples; s++) {
            const u = (s + 0.5) / samples;
            // x maps the [0,1] mean axis to the column horizontally? No — we want
            // density vs mean to share the y-axis (since true_mean line is on y-axis).
            // So: vertical = mean (u), horizontal = density (ys[s]/peak * w/2 around cx).
            const px = cx + ((ys[s] / peak) * (w * 0.42));
            const py = upperBotY - u * meanH;
            if (s === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
          }
          ctx.stroke();
          // mirror to make a symmetric bell
          ctx.beginPath();
          for (let s = 0; s < samples; s++) {
            const u = (s + 0.5) / samples;
            const px = cx - ((ys[s] / peak) * (w * 0.42));
            const py = upperBotY - u * meanH;
            if (s === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
          }
          ctx.stroke();
        }
      }

      // Arm label
      ctx.fillStyle = INK_FADE;
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(`a${i} · ${p}p`, cx, subBaseY + 4);
    }

    // Y-axis labels
    ctx.fillStyle = INK_FADE;
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    [0, 0.5, 1].forEach((m) => {
      const y = upperBotY - m * meanH;
      ctx.fillText(m.toFixed(1), 4, y);
    });

    // Sub-pane label
    ctx.fillStyle = INK_FADE;
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText('PULLS', 4, (subTopY + subBaseY) / 2);

    // Regret pane
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.strokeRect(padX, regretPaneY, size.w - padX * 2, regretPaneH);
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('cumulative regret', padX, regretPaneY - 16);

    const reg = ag.cumRegret;
    if (reg.length >= 2) {
      const allCurves = [state.agents.eps.cumRegret, state.agents.ucb.cumRegret, state.agents.thomp.cumRegret];
      const maxReg = Math.max(1, ...allCurves.flatMap((a) => a));
      ctx.strokeStyle = COLORS[key].color;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      reg.forEach((v, i) => {
        const x = padX + 4 + (i / Math.max(1, reg.length - 1)) * (size.w - padX * 2 - 8);
        const y = regretPaneY + regretPaneH - 4 - (v / maxReg) * (regretPaneH - 8);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      // last value
      const last = reg[reg.length - 1];
      ctx.fillStyle = COLORS[key].color;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillText(`R = ${last.toFixed(2)}`, size.w - padX - 6, regretPaneY + 4);
    }

    // For UCB / Thompson: explanatory annotation
    if (key === 'ucb') {
      ctx.fillStyle = ACCENT;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillText('vertical segment above each tick = confidence bonus · shrinks with samples', size.w - padX, 8);
    } else if (key === 'thomp') {
      ctx.fillStyle = INK_FADE;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillText('bell curve above each arm = Beta posterior · narrows with evidence', size.w - padX, 8);
    }
  }

  function render() {
    Array.from(strategiesEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.view === state.view);
    });
    epsSlider.value = state.eps;
    speedSlider.value = state.speed;
    epsLabel.textContent = `ε = ${state.eps.toFixed(2)}`;
    speedLabel.textContent = `${state.speed} pulls / s`;
    if (captionEl) captionEl.textContent = CAPTIONS[state.view];

    const totalPulls = state.t;
    const epsR = state.agents.eps.cumRegret[state.agents.eps.cumRegret.length - 1];
    const ucbR = state.agents.ucb.cumRegret[state.agents.ucb.cumRegret.length - 1];
    const thR  = state.agents.thomp.cumRegret[state.agents.thomp.cumRegret.length - 1];
    readoutEl.innerHTML = `
      <div class="row"><span>pulls each</span><b>${totalPulls}</b></div>
      <div class="row"><span>best arm</span><b>a${BEST_ARM} (${BEST_MEAN.toFixed(2)})</b></div>
      <div class="row"><span style="color:rgb(58,107,74)">ε-greedy regret</span><b>${formatNum(epsR)}</b></div>
      <div class="row"><span style="color:var(--accent)">UCB regret</span><b>${formatNum(ucbR)}</b></div>
      <div class="row"><span style="color:var(--ink)">Thompson regret</span><b>${formatNum(thR)}</b></div>`;
    canvasCtl.redraw();
  }

  // Warm-start: run ~150 pulls per agent so the race is *already in progress* on
  // first paint — pull-count bars are populated, regret curves have visibly diverged,
  // and the reader sees the punchline (UCB/Thompson pulling ahead) immediately.
  (function warmStart() {
    for (let i = 0; i < 150; i++) step();
    state.flashTs = 0; // suppress the last-pull flash on first paint
  })();

  render();
})();
