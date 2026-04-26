/* Vol. XXVIII — Policy Gradients */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK, INK_FADE } = V;

  // ───────── Grid world (3×3) ─────────
  const GW = 3, GH = 3;
  // start = bottom-left (2,0); goal = top-right (0,2); pit = (1,1)
  // Grid cells encoded by (row, col).
  function makeWorld(goalRow, goalCol) {
    return {
      goal: [goalRow, goalCol],
      pit:  [1, 1],
      start: [GH - 1, 0],
    };
  }
  // Action order: 0=up, 1=right, 2=down, 3=left
  const ACTIONS = [[-1, 0], [0, 1], [1, 0], [0, -1]];
  const ACTION_GLYPHS = ['↑', '→', '↓', '←'];

  // Policy parameters: θ[r][c][a]; π(a|s) = softmax(θ[s]).
  function makeTheta() {
    const t = [];
    for (let r = 0; r < GH; r++) {
      const row = [];
      for (let c = 0; c < GW; c++) row.push([0, 0, 0, 0]);
      t.push(row);
    }
    return t;
  }
  function softmax(zs) {
    const m = Math.max(...zs);
    const e = zs.map((z) => Math.exp(z - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / s);
  }
  function sampleFrom(probs) {
    const u = Math.random();
    let acc = 0;
    for (let i = 0; i < probs.length; i++) {
      acc += probs[i];
      if (u <= acc) return i;
    }
    return probs.length - 1;
  }

  function inBounds(r, c) { return r >= 0 && r < GH && c >= 0 && c < GW; }

  function stepEnv(world, r, c, a) {
    const [dr, dc] = ACTIONS[a];
    let nr = r + dr, nc = c + dc;
    if (!inBounds(nr, nc)) { nr = r; nc = c; }
    let rew = -0.05; // small step cost
    let done = false;
    if (nr === world.goal[0] && nc === world.goal[1]) { rew = 1.0; done = true; }
    else if (nr === world.pit[0] && nc === world.pit[1]) { rew = -1.0; done = true; }
    return { nr, nc, rew, done };
  }

  // One Monte-Carlo REINFORCE rollout, then update θ in place.
  function rollout(theta, world, gamma, lr, baseline) {
    const trace = [];
    let r = world.start[0], c = world.start[1];
    let done = false;
    let steps = 0;
    const maxSteps = 30;
    while (!done && steps < maxSteps) {
      const probs = softmax(theta[r][c]);
      const a = sampleFrom(probs);
      const out = stepEnv(world, r, c, a);
      trace.push({ r, c, a, rew: out.rew, probs });
      r = out.nr; c = out.nc;
      done = out.done;
      steps++;
    }
    // Discounted returns G_t for each step, going backwards.
    const Gs = new Array(trace.length).fill(0);
    let G = 0;
    for (let i = trace.length - 1; i >= 0; i--) {
      G = trace[i].rew + gamma * G;
      Gs[i] = G;
    }
    const totalUndisc = trace.reduce((s, t) => s + t.rew, 0);
    // REINFORCE update: θ[s][a] += α (G − b) (1 − π(a|s))
    //                   θ[s][a'] -= α (G − b) π(a'|s)   for a' ≠ a
    // Using softmax gradient: ∇_θ_a log π(a|s) = δ_{a,a'} − π(a'|s)
    for (let i = 0; i < trace.length; i++) {
      const t = trace[i];
      const adv = Gs[i] - baseline;
      for (let a = 0; a < 4; a++) {
        const grad = (a === t.a ? 1 : 0) - t.probs[a];
        theta[t.r][t.c][a] += lr * adv * grad;
      }
    }
    return { trace, totalReward: totalUndisc, terminalReward: trace.length > 0 ? trace[trace.length - 1].rew : 0 };
  }

  // ───────── Continuous (Gaussian) view — kept as 2nd mode ─────────
  function gaussian() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function gaussianReward(a, target) {
    const dx = a[0] - target[0], dy = a[1] - target[1];
    return Math.exp(-(dx * dx + dy * dy) / 1.5);
  }

  // ───────── State ─────────
  const VIEWS = [
    { key: 'grid',  label: 'Grid policy · bars per state' },
    { key: 'traj',  label: 'Trajectories · arrows by reward' },
    { key: 'cont',  label: 'Continuous · Gaussian policy' },
  ];
  const CAPTIONS = {
    grid:  'A 3×3 grid. Above each cell, four stacked bars show the agent\'s probability of moving ↑→↓← from that cell. Train it: each rollout starts at the bottom-left and ends at the goal (+1) or pit (−1). The bars tilt toward whichever moves led to high reward.',
    traj:  'The same policy, drawn as a vector field. From each state, four arrows fan out — arrow length = π(a|s). The most recent rollout\'s sampled actions are colored by the rollout\'s outcome (green = reached goal, red = fell in pit).',
    cont:  'Continuous 2D action space. The policy is a Gaussian centered at μ; click anywhere to set a target reward. Each step samples 8 actions, weights them by reward, and shifts μ. This view is what "policy gradients" usually look like in continuous-action papers.',
  };

  const state = {
    view: 'grid',
    theta: makeTheta(),
    world: makeWorld(0, GW - 1),
    lr: 0.18,
    gamma: 0.9,
    speed: 10,
    training: false,
    timer: null,
    rollouts: 0,
    lastTrace: null,
    lastReward: 0,
    rewardHistory: [],
    baseline: 0, // running average reward
    flashStep: -1, // index into trace, animates which step is showing
    flashStart: 0,
    // Continuous view state
    cMu: [-1.5, -1.5],
    cSigma: 0.5,
    cTarget: [1.5, 1.5],
    cSamples: [],
    cStep: 0,
    cReward: 0,
  };

  // ───────── DOM ─────────
  const viewToggleEl = document.getElementById('view-toggle');
  const lrSlider = document.getElementById('lr-slider');
  const lrLabel = document.getElementById('lr-label');
  const gammaSlider = document.getElementById('gamma-slider');
  const gammaLabel = document.getElementById('gamma-label');
  const speedSlider = document.getElementById('speed-slider');
  const speedLabel = document.getElementById('speed-label');
  const trainBtn = document.getElementById('train-btn');
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
    viewToggleEl.appendChild(b);
  });

  lrSlider.addEventListener('input', (e) => {
    state.lr = +e.target.value;
    lrLabel.textContent = `α = ${state.lr.toFixed(2)}`;
  });
  gammaSlider.addEventListener('input', (e) => {
    state.gamma = +e.target.value;
    gammaLabel.textContent = `γ = ${state.gamma.toFixed(2)}`;
  });
  speedSlider.addEventListener('input', (e) => {
    state.speed = +e.target.value;
    speedLabel.textContent = `${state.speed} rollouts / s`;
    if (state.training) { stopTrain(); startTrain(); }
  });
  trainBtn.addEventListener('click', () => { if (state.training) stopTrain(); else startTrain(); });
  resetBtn.addEventListener('click', () => {
    stopTrain();
    state.theta = makeTheta();
    state.rollouts = 0;
    state.lastTrace = null;
    state.lastReward = 0;
    state.rewardHistory = [];
    state.baseline = 0;
    state.cMu = [-1.5, -1.5];
    state.cSamples = [];
    state.cStep = 0;
    state.cReward = 0;
    render();
  });

  function startTrain() {
    state.training = true;
    trainBtn.textContent = '⏸ Pause';
    const interval = Math.max(20, 1000 / state.speed);
    state.timer = setInterval(tickTrain, interval);
  }
  function stopTrain() {
    state.training = false;
    trainBtn.textContent = '▷ Train';
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  function tickTrain() {
    if (state.view === 'cont') {
      // Continuous Gaussian step
      const samples = [];
      const sumGrad = [0, 0];
      let totalR = 0;
      const baseline = state.cSamples.length > 0
        ? state.cSamples.reduce((s, p) => s + p.r, 0) / state.cSamples.length
        : 0;
      for (let i = 0; i < 8; i++) {
        const a = [state.cMu[0] + state.cSigma * gaussian(), state.cMu[1] + state.cSigma * gaussian()];
        const r = gaussianReward(a, state.cTarget);
        const adv = r - baseline;
        sumGrad[0] += (a[0] - state.cMu[0]) / (state.cSigma * state.cSigma) * adv;
        sumGrad[1] += (a[1] - state.cMu[1]) / (state.cSigma * state.cSigma) * adv;
        totalR += r;
        samples.push({ a, r });
      }
      state.cMu = [state.cMu[0] + state.lr * sumGrad[0] / 8,
                   state.cMu[1] + state.lr * sumGrad[1] / 8];
      state.cSamples = samples;
      state.cStep++;
      state.cReward = totalR / 8;
      render();
      return;
    }
    // Grid REINFORCE rollout
    const out = rollout(state.theta, state.world, state.gamma, state.lr, state.baseline);
    state.rollouts++;
    state.lastTrace = out.trace;
    state.lastReward = out.totalReward;
    state.rewardHistory.push(out.totalReward);
    if (state.rewardHistory.length > 80) state.rewardHistory.shift();
    // Update baseline as running mean
    state.baseline = state.baseline * 0.9 + out.totalReward * 0.1;
    state.flashStep = 0;
    state.flashStart = performance.now();
    render();
    runFlashAnim();
  }

  // Click on canvas — different behavior per view.
  canvas.addEventListener('click', (e) => {
    const r = canvas.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width;
    const y = (e.clientY - r.top) / r.height;
    if (state.view === 'cont') {
      const R = 3;
      state.cTarget = [x * 2 * R - R, R - y * 2 * R];
    } else {
      // Grid: clicking moves the goal to the clicked cell.
      const layout = computeGridLayout(canvasCtl.getSize());
      const col = Math.floor((x * layout.size.w - layout.gridX0) / layout.cellSize);
      const row = Math.floor((y * layout.size.h - layout.gridY0) / layout.cellSize);
      if (inBounds(row, col) && !(row === state.world.pit[0] && col === state.world.pit[1])) {
        state.world.goal = [row, col];
        // Soft reset of policy so it can re-learn (but keep some learned value)
      }
    }
    render();
  });

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function computeGridLayout(size) {
    // Reserve top space for stacked-bars over each cell.
    const padX = 24, padTop = 70, padBottom = 24;
    const availW = size.w - padX * 2;
    const availH = size.h - padTop - padBottom;
    const cellSize = Math.min(availW / GW, availH / GH);
    const totalW = cellSize * GW, totalH = cellSize * GH;
    const gridX0 = (size.w - totalW) / 2;
    const gridY0 = padTop + (availH - totalH) / 2;
    return { size, cellSize, gridX0, gridY0, totalW, totalH };
  }

  function runFlashAnim() {
    if (!state.lastTrace || state.lastTrace.length === 0) return;
    const elapsed = performance.now() - state.flashStart;
    const stepDur = 90; // ms per step in the trace
    state.flashStep = Math.min(state.lastTrace.length, Math.floor(elapsed / stepDur));
    canvasCtl.redraw();
    if (state.flashStep < state.lastTrace.length + 4) {
      requestAnimationFrame(runFlashAnim);
    }
  }

  function draw(ctx, size) {
    if (state.view === 'cont') return drawContinuous(ctx, size);
    const layout = computeGridLayout(size);
    drawGrid(ctx, layout);
    if (state.view === 'grid') drawProbBars(ctx, layout);
    if (state.view === 'traj') drawArrows(ctx, layout);
    drawAgent(ctx, layout);
    drawHeader(ctx, layout);
    drawRewardCurve(ctx, layout);
  }

  function drawHeader(ctx, layout) {
    const { size } = layout;
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const label = state.view === 'grid'
      ? 'π(a|s) — action probabilities above each state'
      : 'rollouts — arrows colored by reward';
    ctx.fillText(label, size.w / 2, 6);
  }

  function drawGrid(ctx, layout) {
    const { cellSize, gridX0, gridY0 } = layout;
    for (let r = 0; r < GH; r++) {
      for (let c = 0; c < GW; c++) {
        const x = gridX0 + c * cellSize;
        const y = gridY0 + r * cellSize;
        const isGoal = r === state.world.goal[0] && c === state.world.goal[1];
        const isPit  = r === state.world.pit[0]  && c === state.world.pit[1];
        let bg = '#fffdf6';
        if (isGoal) bg = 'rgba(122,31,36,0.55)';
        else if (isPit) bg = 'rgba(38,35,32,0.50)';
        ctx.fillStyle = bg;
        ctx.fillRect(x, y, cellSize, cellSize);
        ctx.strokeStyle = 'rgba(38,35,32,0.30)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, cellSize, cellSize);
        if (isGoal) {
          ctx.fillStyle = '#fffdf6';
          ctx.font = 'bold 14px "JetBrains Mono", monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText('+1', x + cellSize / 2, y + cellSize / 2);
        } else if (isPit) {
          ctx.fillStyle = '#fffdf6';
          ctx.font = 'bold 14px "JetBrains Mono", monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText('−1', x + cellSize / 2, y + cellSize / 2);
        }
        // Mark start
        if (r === state.world.start[0] && c === state.world.start[1] && !isGoal && !isPit) {
          ctx.fillStyle = 'rgba(38,35,32,0.45)';
          ctx.font = '9px "JetBrains Mono", monospace';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'top';
          ctx.fillText('start', x + 4, y + 4);
        }
      }
    }
  }

  // Inside each non-terminal cell: four mini bars showing π(↑), π(→), π(↓), π(←),
  // plus a directional arrow for the most likely action.
  function drawProbBars(ctx, layout) {
    const { cellSize, gridX0, gridY0 } = layout;
    for (let r = 0; r < GH; r++) {
      for (let c = 0; c < GW; c++) {
        const isGoal = r === state.world.goal[0] && c === state.world.goal[1];
        const isPit  = r === state.world.pit[0]  && c === state.world.pit[1];
        if (isGoal || isPit) continue;
        const probs = softmax(state.theta[r][c]);
        const x = gridX0 + c * cellSize;
        const y = gridY0 + r * cellSize;
        // Draw 4 vertical mini-bars in the top portion of the cell, labeled ↑→↓←
        const padInner = cellSize * 0.10;
        const barAreaH = cellSize * 0.42;
        const barAreaW = cellSize - padInner * 2;
        const slot = barAreaW / 4;
        const baseY = y + padInner + barAreaH;
        for (let a = 0; a < 4; a++) {
          const p = probs[a];
          const bx = x + padInner + a * slot + slot * 0.15;
          const bw = slot * 0.7;
          const bh = p * barAreaH;
          ctx.fillStyle = `rgba(122,31,36,${0.18 + p * 0.65})`;
          ctx.fillRect(bx, baseY - bh, bw, bh);
          // glyph
          ctx.fillStyle = INK_FADE;
          ctx.font = '10px "JetBrains Mono", monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'top';
          ctx.fillText(ACTION_GLYPHS[a], bx + bw / 2, baseY + 2);
        }
        // Faint divider between bars area and lower (reserved) area.
        ctx.strokeStyle = 'rgba(38,35,32,0.10)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x + padInner, y + padInner + barAreaH + 14);
        ctx.lineTo(x + cellSize - padInner, y + padInner + barAreaH + 14);
        ctx.stroke();
        // The "best action so far" arrow drawn larger in the lower half
        let bestA = 0;
        for (let a = 1; a < 4; a++) if (probs[a] > probs[bestA]) bestA = a;
        const cx = x + cellSize / 2;
        const cy = y + cellSize * 0.78;
        const ar = cellSize * 0.16;
        const [dr, dc] = ACTIONS[bestA];
        drawArrow(ctx, cx - dc * ar, cy - dr * ar, cx + dc * ar, cy + dr * ar,
          `rgba(122,31,36,${0.4 + probs[bestA] * 0.6})`,
          1 + probs[bestA] * 4);
      }
    }
  }

  // Trajectory view: from each state, fan out 4 arrows with length = π(a|s).
  function drawArrows(ctx, layout) {
    const { cellSize, gridX0, gridY0 } = layout;
    for (let r = 0; r < GH; r++) {
      for (let c = 0; c < GW; c++) {
        const isGoal = r === state.world.goal[0] && c === state.world.goal[1];
        const isPit  = r === state.world.pit[0]  && c === state.world.pit[1];
        if (isGoal || isPit) continue;
        const probs = softmax(state.theta[r][c]);
        const cx = gridX0 + c * cellSize + cellSize / 2;
        const cy = gridY0 + r * cellSize + cellSize / 2;
        for (let a = 0; a < 4; a++) {
          const p = probs[a];
          const len = p * cellSize * 0.42;
          const [dr, dc] = ACTIONS[a];
          const ex = cx + dc * len;
          const ey = cy + dr * len;
          const alpha = 0.30 + p * 0.55;
          drawArrow(ctx, cx, cy, ex, ey, `rgba(122,31,36,${alpha})`, 1 + p * 3.5);
        }
        // Tiny dot at center
        ctx.fillStyle = 'rgba(38,35,32,0.45)';
        ctx.beginPath();
        ctx.arc(cx, cy, 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Overlay: most recent rollout colored by reward
    if (state.lastTrace && state.lastTrace.length > 0) {
      const trace = state.lastTrace;
      const finalRew = trace[trace.length - 1].rew;
      const winning = finalRew > 0;
      const losing = finalRew < -0.5;
      let color;
      if (winning) color = 'rgba(58,107,74,0.85)';   // green
      else if (losing) color = 'rgba(122,31,36,0.85)'; // accent
      else color = 'rgba(38,35,32,0.55)';
      const visible = state.flashStep > 0 ? Math.min(trace.length, state.flashStep) : trace.length;
      for (let i = 0; i < visible; i++) {
        const t = trace[i];
        const cx = gridX0 + t.c * cellSize + cellSize / 2;
        const cy = gridY0 + t.r * cellSize + cellSize / 2;
        const [dr, dc] = ACTIONS[t.a];
        const ex = cx + dc * cellSize * 0.45;
        const ey = cy + dr * cellSize * 0.45;
        drawArrow(ctx, cx, cy, ex, ey, color, 3.5);
      }
      // Outcome label
      if (state.flashStep >= trace.length) {
        ctx.fillStyle = winning ? '#3a6b4a' : losing ? ACCENT : INK_FADE;
        ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        const lbl = winning ? 'rollout reached the goal · push these actions UP'
                  : losing  ? 'rollout fell in the pit · push these actions DOWN'
                  : 'rollout timed out · weak signal';
        ctx.fillText(lbl, layout.gridX0, layout.gridY0 + layout.totalH + 6);
      }
    }
  }

  function drawAgent(ctx, layout) {
    if (!state.lastTrace || state.lastTrace.length === 0) return;
    const trace = state.lastTrace;
    const i = Math.max(0, Math.min(trace.length - 1, state.flashStep));
    const t = trace[i];
    const cx = layout.gridX0 + t.c * layout.cellSize + layout.cellSize / 2;
    const cy = layout.gridY0 + t.r * layout.cellSize + layout.cellSize / 2;
    // Trail
    for (let k = 0; k < i; k++) {
      const tk = trace[k];
      const tx = layout.gridX0 + tk.c * layout.cellSize + layout.cellSize / 2;
      const ty = layout.gridY0 + tk.r * layout.cellSize + layout.cellSize / 2;
      const a = (k + 1) / (i + 1) * 0.40;
      ctx.fillStyle = `rgba(38,35,32,${a})`;
      ctx.beginPath();
      ctx.arc(tx, ty, layout.cellSize * 0.06, 0, Math.PI * 2);
      ctx.fill();
    }
    // Agent dot
    ctx.fillStyle = ACCENT;
    ctx.beginPath();
    ctx.arc(cx, cy, layout.cellSize * 0.10, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#fffdf6';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  function drawArrow(ctx, sx, sy, ex, ey, color, lw) {
    const dx = ex - sx, dy = ey - sy;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 1) return;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = lw;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    const ang = Math.atan2(dy, dx);
    const ah = Math.max(4, lw * 1.8);
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex - ah * Math.cos(ang - 0.45), ey - ah * Math.sin(ang - 0.45));
    ctx.lineTo(ex - ah * Math.cos(ang + 0.45), ey - ah * Math.sin(ang + 0.45));
    ctx.closePath();
    ctx.fill();
  }

  function drawRewardCurve(ctx, layout) {
    const { size } = layout;
    const panelH = 28;
    const panelW = size.w - 32;
    const panelX = 16;
    const panelY = size.h - panelH - 4;
    ctx.fillStyle = 'rgba(38,35,32,0.04)';
    ctx.fillRect(panelX, panelY, panelW, panelH);
    ctx.strokeStyle = 'rgba(38,35,32,0.20)';
    ctx.lineWidth = 1;
    ctx.strokeRect(panelX, panelY, panelW, panelH);
    ctx.fillStyle = INK_FADE;
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('REWARD PER ROLLOUT', panelX + 6, panelY + 3);

    const hist = state.rewardHistory;
    if (hist.length >= 2) {
      const minV = -1.5, maxV = 1.2;
      const range = maxV - minV;
      const zeroY = panelY + panelH - 4 - ((0 - minV) / range) * (panelH - 12);
      ctx.strokeStyle = 'rgba(38,35,32,0.25)';
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(panelX + 4, zeroY);
      ctx.lineTo(panelX + panelW - 4, zeroY);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      hist.forEach((v, i) => {
        const px = panelX + 4 + (i / Math.max(1, hist.length - 1)) * (panelW - 8);
        const py = panelY + panelH - 4 - ((v - minV) / range) * (panelH - 12);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
      const last = hist[hist.length - 1];
      ctx.fillStyle = ACCENT;
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`last R = ${last.toFixed(2)}`, panelX + panelW - 6, panelY + 3);
    } else {
      ctx.fillStyle = 'rgba(38,35,32,0.45)';
      ctx.font = 'italic 10px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('train to see reward curve', panelX + panelW / 2, panelY + panelH / 2 + 2);
    }
  }

  function drawContinuous(ctx, size) {
    const R = 3;
    const tx = (x) => (x + R) / (2 * R) * size.w;
    const ty = (y) => size.h - (y + R) / (2 * R) * size.h;

    // Reward heatmap
    const cells = 50;
    for (let i = 0; i < cells; i++) {
      for (let j = 0; j < cells; j++) {
        const x = -R + (j + 0.5) / cells * 2 * R;
        const y = R - (i + 0.5) / cells * 2 * R;
        const r = gaussianReward([x, y], state.cTarget);
        ctx.fillStyle = `rgba(122,31,36,${0.04 + r * 0.30})`;
        ctx.fillRect(j * size.w / cells, i * size.h / cells, size.w / cells + 1, size.h / cells + 1);
      }
    }
    // Grid
    ctx.strokeStyle = 'rgba(38,35,32,0.08)';
    ctx.lineWidth = 1;
    for (let i = -3; i <= 3; i++) {
      ctx.beginPath(); ctx.moveTo(tx(i), 0); ctx.lineTo(tx(i), size.h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, ty(i)); ctx.lineTo(size.w, ty(i)); ctx.stroke();
    }
    // Policy contour
    ctx.strokeStyle = 'rgba(38,35,32,0.55)';
    ctx.lineWidth = 1.5;
    for (const factor of [1, 2]) {
      ctx.beginPath();
      for (let t = 0; t <= Math.PI * 2 + 0.1; t += 0.05) {
        const x = state.cMu[0] + state.cSigma * factor * Math.cos(t);
        const y = state.cMu[1] + state.cSigma * factor * Math.sin(t);
        const px = tx(x), py = ty(y);
        if (t === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
    // Recent samples — colored by reward
    for (const { a, r } of state.cSamples) {
      const ax = tx(a[0]), ay = ty(a[1]);
      // Arrow from μ to sample
      const mx = tx(state.cMu[0]), my = ty(state.cMu[1]);
      const intensity = Math.max(0, Math.min(1, r));
      // Green for high reward, gray for low
      const color = r > 0.4 ? `rgba(58,107,74,${0.30 + intensity * 0.55})` : `rgba(38,35,32,${0.20 + r * 0.25})`;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1 + intensity * 1.5;
      ctx.beginPath();
      ctx.moveTo(mx, my);
      ctx.lineTo(ax, ay);
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(ax, ay, 3, 0, Math.PI * 2);
      ctx.fill();
    }
    // μ marker
    ctx.fillStyle = '#262320';
    ctx.beginPath();
    ctx.arc(tx(state.cMu[0]), ty(state.cMu[1]), 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#fffdf6';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    // Target star
    ctx.fillStyle = ACCENT;
    drawStar(ctx, tx(state.cTarget[0]), ty(state.cTarget[1]), 9);
    // Header
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('continuous Gaussian policy · μ drifts toward high-reward samples', size.w / 2, 6);
  }

  function drawStar(ctx, x, y, r) {
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const a = (i / 10) * Math.PI * 2 - Math.PI / 2;
      const radius = i % 2 === 0 ? r : r * 0.45;
      const px = x + radius * Math.cos(a);
      const py = y + radius * Math.sin(a);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fill();
  }

  function render() {
    Array.from(viewToggleEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.view === state.view);
    });
    lrLabel.textContent = `α = ${state.lr.toFixed(2)}`;
    gammaLabel.textContent = `γ = ${state.gamma.toFixed(2)}`;
    speedLabel.textContent = `${state.speed} rollouts / s`;
    lrSlider.value = state.lr;
    gammaSlider.value = state.gamma;
    speedSlider.value = state.speed;
    if (captionEl) captionEl.textContent = CAPTIONS[state.view];

    if (state.view === 'cont') {
      readoutEl.innerHTML = `
        <div class="row"><span>μ</span><b>(${formatNum(state.cMu[0])}, ${formatNum(state.cMu[1])})</b></div>
        <div class="row"><span>target</span><b>(${formatNum(state.cTarget[0])}, ${formatNum(state.cTarget[1])})</b></div>
        <div class="row"><span>step</span><b>${state.cStep}</b></div>
        <div class="row"><span>last reward</span><b>${formatNum(state.cReward)}</b></div>`;
    } else {
      const recent = state.rewardHistory.slice(-10);
      const avg = recent.length ? recent.reduce((a, b) => a + b, 0) / recent.length : 0;
      const wins = state.rewardHistory.filter((r) => r > 0.5).length;
      readoutEl.innerHTML = `
        <div class="row"><span>rollouts</span><b>${state.rollouts}</b></div>
        <div class="row"><span>last R</span><b>${formatNum(state.lastReward)}</b></div>
        <div class="row"><span>avg R (last 10)</span><b>${formatNum(avg)}</b></div>
        <div class="row"><span>wins (R&gt;0.5)</span><b>${wins} / ${state.rewardHistory.length}</b></div>
        <div class="row"><span>baseline b</span><b>${formatNum(state.baseline)}</b></div>`;
    }
    canvasCtl.redraw();
  }

  // Warm-start: run a handful of REINFORCE rollouts so the bars on first paint
  // are slightly off-uniform — the *direction* of learning is visible immediately,
  // but the policy is far from converged, leaving plenty for the reader to watch.
  (function warmStart() {
    for (let i = 0; i < 30; i++) {
      const out = rollout(state.theta, state.world, state.gamma, state.lr, state.baseline);
      state.rollouts++;
      state.lastTrace = out.trace;
      state.lastReward = out.totalReward;
      state.rewardHistory.push(out.totalReward);
      if (state.rewardHistory.length > 80) state.rewardHistory.shift();
      state.baseline = state.baseline * 0.9 + out.totalReward * 0.1;
    }
    // Don't trigger the trace-flash animation on first paint — keep the canvas calm.
    state.flashStep = state.lastTrace ? state.lastTrace.length : 0;
  })();

  canvas.style.cursor = 'pointer';
  render();
})();
