/* Vol. XXVI — Markov Decision Processes */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK_FADE } = V;

  const W = 5, H = 4; // grid columns × rows
  // Cell types: ' '=normal, 'W'=wall, 'G'=goal (+1), 'B'=bad (−1)
  function makeGrid() {
    const g = Array.from({ length: H }, () => new Array(W).fill(' '));
    g[1][1] = 'W';
    g[0][W - 1] = 'G';
    g[1][W - 1] = 'B';
    return g;
  }
  // up, right, down, left  (dx, dy)
  const ACTIONS = [[0, -1], [1, 0], [0, 1], [-1, 0]];

  // Stochastic transitions: with prob 0.8 you go where you intended; with prob 0.1 each
  // you slip to one of the perpendicular directions.  Realistic, and finally makes
  // the Σ P(s′|s,a) V(s′) term in Bellman *do something*.
  const SLIP = 0.10;
  const INTENT = 1 - 2 * SLIP;

  function reward(grid, r, c) {
    const cell = grid[r][c];
    if (cell === 'G') return 1;
    if (cell === 'B') return -1;
    return 0;
  }
  function isTerminal(grid, r, c) {
    return grid[r][c] === 'G' || grid[r][c] === 'B';
  }
  function inBounds(r, c) { return r >= 0 && r < H && c >= 0 && c < W; }

  // For action a from (r,c): returns array of {r, c, p} weighted next states.
  // If a candidate next cell is OOB or a wall, agent stays in place for that prob mass.
  function transitionsFor(grid, r, c, a) {
    const intended = ACTIONS[a];
    const left = ACTIONS[(a + 3) % 4];
    const right = ACTIONS[(a + 1) % 4];
    const candidates = [
      { d: intended, p: INTENT },
      { d: left, p: SLIP },
      { d: right, p: SLIP },
    ];
    const out = [];
    for (const { d, p } of candidates) {
      const nr = r + d[1], nc = c + d[0];
      if (!inBounds(nr, nc) || grid[nr][nc] === 'W') {
        out.push({ r, c, p });
      } else {
        out.push({ r: nr, c: nc, p });
      }
    }
    return out;
  }

  // Expected discounted return for taking action a from (r,c) given current V.
  function actionValue(grid, Vgrid, r, c, a, gamma, stepReward) {
    const trans = transitionsFor(grid, r, c, a);
    let total = 0;
    for (const t of trans) {
      const r_imm = isTerminal(grid, t.r, t.c) ? reward(grid, t.r, t.c) : stepReward;
      total += t.p * (r_imm + gamma * Vgrid[t.r][t.c]);
    }
    return total;
  }

  // ONE Bellman backup over all states.  Returns new V array.
  function bellmanBackup(grid, Vgrid, gamma, stepReward) {
    const newV = Vgrid.map((row) => row.slice());
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        if (grid[r][c] === 'W') continue;
        if (grid[r][c] === 'G') { newV[r][c] = 1; continue; }
        if (grid[r][c] === 'B') { newV[r][c] = -1; continue; }
        let best = -Infinity;
        for (let a = 0; a < 4; a++) {
          const v = actionValue(grid, Vgrid, r, c, a, gamma, stepReward);
          if (v > best) best = v;
        }
        newV[r][c] = best;
      }
    }
    return newV;
  }

  function emptyV(grid) {
    const Vg = Array.from({ length: H }, () => new Array(W).fill(0));
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      if (grid[r][c] === 'G') Vg[r][c] = 1;
      else if (grid[r][c] === 'B') Vg[r][c] = -1;
    }
    return Vg;
  }

  function policyFromV(grid, Vgrid, gamma, stepReward) {
    const policy = Array.from({ length: H }, () => new Array(W).fill(-1));
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        if (grid[r][c] === 'W' || isTerminal(grid, r, c)) continue;
        let best = -Infinity, bestA = -1;
        for (let a = 0; a < 4; a++) {
          const v = actionValue(grid, Vgrid, r, c, a, gamma, stepReward);
          if (v > best) { best = v; bestA = a; }
        }
        policy[r][c] = bestA;
      }
    }
    return policy;
  }

  const state = {
    grid: makeGrid(),
    gamma: 0.9,
    stepReward: -0.04,
    view: 'value', // 'value' or 'policy'
    V: null,
    Vprev: null,
    sweep: 0,
    pulses: [], // { r, c, t: time of pulse start in ms }
    speed: 3,
    timer: null,
    autoRunning: false,
    annotation: null, // { text, ts } shown on-canvas
  };
  state.V = emptyV(state.grid);
  // Warm-start: run 3 Bellman backups so the value gradient is visible on first paint.
  for (let i = 0; i < 3; i++) {
    state.V = bellmanBackup(state.grid, state.V, state.gamma, state.stepReward);
    state.sweep++;
  }
  // Initial annotation explains the partial-sweep landing state.
  state.annotation = {
    text: '3 sweeps in — value already leaking outward from the goal. Press Step to continue.',
    ts: performance.now(),
    ttl: 5200,
  };

  const gammaSlider = document.getElementById('gamma-slider');
  const gammaLabel = document.getElementById('gamma-label');
  const stepRewardSlider = document.getElementById('step-reward');
  const stepRewardLabel = document.getElementById('step-reward-label');
  const viewToggleEl = document.getElementById('view-toggle');
  const stepBtn = document.getElementById('step-btn');
  const runBtn = document.getElementById('run-btn');
  const resetBtn = document.getElementById('reset-btn');
  const speedSlider = document.getElementById('speed-slider');
  const speedLabel = document.getElementById('speed-label');
  const readoutEl = document.getElementById('readout');
  const canvas = document.getElementById('canvas');

  ['Values', 'Policy only'].forEach((lbl, i) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = lbl;
    b.dataset.view = i === 0 ? 'value' : 'policy';
    b.addEventListener('click', () => { state.view = b.dataset.view; render(); });
    viewToggleEl.appendChild(b);
  });

  function resetV() {
    state.V = emptyV(state.grid);
    state.Vprev = null;
    state.sweep = 0;
    state.pulses = [];
  }

  function doStep() {
    state.Vprev = state.V.map((row) => row.slice());
    state.V = bellmanBackup(state.grid, state.V, state.gamma, state.stepReward);
    state.sweep++;
    // Add pulses for cells whose value just changed
    const now = performance.now();
    state.pulses = [];
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        if (state.grid[r][c] === 'W') continue;
        if (Math.abs(state.V[r][c] - state.Vprev[r][c]) > 1e-4) {
          state.pulses.push({ r, c, t: now });
        }
      }
    }
    render();
  }

  gammaSlider.addEventListener('input', (e) => {
    state.gamma = +e.target.value;
    gammaLabel.textContent = `γ = ${state.gamma.toFixed(2)}`;
    resetV();
    render();
  });
  stepRewardSlider.addEventListener('input', (e) => {
    state.stepReward = +e.target.value;
    stepRewardLabel.textContent = `step = ${state.stepReward.toFixed(2)}`;
    resetV();
    render();
  });
  speedSlider.addEventListener('input', (e) => {
    state.speed = +e.target.value;
    const labels = ['', '0.25×', '0.5×', '1×', '1.5×', '2×', '3×', '4×', '6×', '8×', '12×'];
    speedLabel.textContent = labels[state.speed] || `${state.speed}×`;
    if (state.autoRunning) { stopAuto(); startAuto(); }
  });

  stepBtn.addEventListener('click', () => { stopAuto(); doStep(); });

  function startAuto() {
    state.autoRunning = true;
    runBtn.textContent = '⏸ Pause auto-step';
    const interval = Math.max(60, 1400 / state.speed);
    state.timer = setInterval(() => { doStep(); }, interval);
  }
  function stopAuto() {
    state.autoRunning = false;
    runBtn.textContent = '▷ Auto-step';
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  runBtn.addEventListener('click', () => { if (state.autoRunning) stopAuto(); else startAuto(); });
  resetBtn.addEventListener('click', () => {
    stopAuto();
    state.grid = makeGrid();
    resetV();
    render();
  });

  // Click on canvas to toggle wall
  canvas.style.cursor = 'pointer';
  canvas.addEventListener('click', (e) => {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    const size = canvasCtl.getSize();
    const cellW = size.w / W;
    const cellH = size.h / H;
    const c = Math.floor(x / cellW);
    const row = Math.floor(y / cellH);
    if (row < 0 || row >= H || c < 0 || c >= W) return;
    let toggled = null;
    if (state.grid[row][c] === 'W') { state.grid[row][c] = ' '; toggled = 'removed'; }
    else if (state.grid[row][c] === ' ') { state.grid[row][c] = 'W'; toggled = 'added'; }
    if (toggled) {
      state.annotation = {
        text: toggled === 'added'
          ? 'Wall added — values will redistribute on next Step.'
          : 'Wall removed — values will redistribute on next Step.',
        ts: performance.now(),
      };
    }
    resetV();
    render();
  });

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  // Animate pulses
  function pulseAlpha(pulseT) {
    const dt = (performance.now() - pulseT) / 1000;
    if (dt < 0 || dt > 0.9) return 0;
    return (1 - dt / 0.9) * 0.55;
  }
  function anyPulseLive() {
    const now = performance.now();
    return state.pulses.some((p) => now - p.t < 900);
  }
  function annotationLive() {
    const ttl = state.annotation && state.annotation.ttl ? state.annotation.ttl : 2400;
    return state.annotation && (performance.now() - state.annotation.ts) < ttl;
  }
  function tickAnim() {
    if (anyPulseLive() || annotationLive()) {
      canvasCtl.redraw();
      requestAnimationFrame(tickAnim);
    }
  }

  function draw(ctx, size) {
    const Vgrid = state.V;
    const policy = policyFromV(state.grid, Vgrid, state.gamma, state.stepReward);

    const cellW = size.w / W;
    const cellH = size.h / H;

    // Background fills
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const cell = state.grid[r][c];
        const x = c * cellW, y = r * cellH;

        if (cell === 'W') {
          ctx.fillStyle = 'rgba(38,35,32,0.20)';
        } else if (cell === 'G') {
          ctx.fillStyle = 'rgba(122,31,36,0.55)';
        } else if (cell === 'B') {
          ctx.fillStyle = 'rgba(38,35,32,0.50)';
        } else {
          const v = Vgrid[r][c];
          const t = Math.max(-1, Math.min(1, v));
          if (state.view === 'value') {
            if (t > 0) {
              ctx.fillStyle = `rgba(122,31,36,${0.05 + t * 0.45})`;
            } else {
              ctx.fillStyle = `rgba(38,35,32,${0.04 + Math.abs(t) * 0.3})`;
            }
          } else {
            ctx.fillStyle = '#fffdf6';
          }
        }
        ctx.fillRect(x, y, cellW, cellH);

        // Pulse overlay
        const pulse = state.pulses.find((p) => p.r === r && p.c === c);
        if (pulse) {
          const a = pulseAlpha(pulse.t);
          if (a > 0) {
            ctx.fillStyle = `rgba(122,31,36,${a})`;
            ctx.fillRect(x, y, cellW, cellH);
            // Expanding ring
            const dt = (performance.now() - pulse.t) / 900;
            const rad = Math.min(cellW, cellH) * (0.2 + dt * 0.55);
            ctx.strokeStyle = `rgba(122,31,36,${a * 0.9})`;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(x + cellW / 2, y + cellH / 2, rad, 0, Math.PI * 2);
            ctx.stroke();
          }
        }

        ctx.strokeStyle = 'rgba(38,35,32,0.30)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, cellW, cellH);
      }
    }

    // Transition arrows: for each non-terminal cell, draw all 3 possible
    // outcomes of the optimal action with arrow width = transition probability.
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        if (state.grid[r][c] !== ' ') continue;
        const a = policy[r][c];
        if (a === -1) continue;
        const trans = transitionsFor(state.grid, r, c, a);
        const cx = c * cellW + cellW / 2;
        const cy = r * cellH + cellH / 2;
        for (const t of trans) {
          if (t.r === r && t.c === c) continue; // bounce-back: skip drawing
          const tx = t.c * cellW + cellW / 2;
          const ty = t.r * cellH + cellH / 2;
          const isOptimal = Math.abs(t.p - INTENT) < 1e-6;
          const lw = Math.max(1, t.p * 4.5);
          // Shrink endpoints toward each other
          const dx = tx - cx, dy = ty - cy;
          const len = Math.sqrt(dx * dx + dy * dy);
          const ux = dx / len, uy = dy / len;
          const sx = cx + ux * cellW * 0.18;
          const sy = cy + uy * cellH * 0.18;
          const ex = tx - ux * cellW * 0.32;
          const ey = ty - uy * cellH * 0.32;

          ctx.strokeStyle = isOptimal
            ? 'rgba(122,31,36,0.85)'
            : 'rgba(38,35,32,0.35)';
          ctx.lineWidth = lw;
          ctx.beginPath();
          ctx.moveTo(sx, sy);
          ctx.lineTo(ex, ey);
          ctx.stroke();
          // Arrowhead
          const ang = Math.atan2(ey - sy, ex - sx);
          ctx.fillStyle = ctx.strokeStyle;
          ctx.beginPath();
          ctx.moveTo(ex, ey);
          ctx.lineTo(ex - 7 * Math.cos(ang - 0.45), ey - 7 * Math.sin(ang - 0.45));
          ctx.lineTo(ex - 7 * Math.cos(ang + 0.45), ey - 7 * Math.sin(ang + 0.45));
          ctx.closePath();
          ctx.fill();
          // Probability tag (only for non-optimal, to avoid clutter on the wide arrow)
          if (!isOptimal && t.p > 0) {
            ctx.fillStyle = 'rgba(38,35,32,0.55)';
            ctx.font = '9px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(t.p.toFixed(2), (sx + ex) / 2, (sy + ey) / 2 - 8);
          }
        }
      }
    }

    // Value text on top of everything
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const cell = state.grid[r][c];
        if (cell === 'W') continue;
        if (state.view !== 'value') continue;
        const x = c * cellW, y = r * cellH;
        ctx.fillStyle = cell === 'G' || cell === 'B' || Math.abs(Vgrid[r][c]) > 0.5
          ? '#fffdf6' : '#262320';
        ctx.font = 'bold 12px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        let txt;
        if (cell === 'G') txt = '+1';
        else if (cell === 'B') txt = '−1';
        else txt = formatNum(Vgrid[r][c]);
        ctx.fillText(txt, x + cellW / 2, y + 6);
      }
    }

    // Sweep counter overlay (bottom-left corner)
    ctx.fillStyle = 'rgba(38,35,32,0.55)';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText(`SWEEP ${state.sweep}`, 6, size.h - 4);
    ctx.textAlign = 'right';
    ctx.fillText('thin = stochastic slip · thick = intended', size.w - 6, size.h - 4);

    // On-canvas annotation (e.g. wall toggled) — fades over its TTL.
    if (state.annotation) {
      const ttl = state.annotation.ttl || 2400;
      const dt = (performance.now() - state.annotation.ts) / ttl;
      if (dt >= 0 && dt < 1) {
        const a = Math.min(1, (1 - dt) * 1.4);
        ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        const txt = state.annotation.text;
        const tw = ctx.measureText(txt).width;
        const bx = size.w / 2 - tw / 2 - 8;
        const by = 6;
        ctx.fillStyle = `rgba(255,253,246,${0.92 * a})`;
        ctx.fillRect(bx, by, tw + 16, 20);
        ctx.strokeStyle = `rgba(122,31,36,${0.55 * a})`;
        ctx.lineWidth = 1;
        ctx.strokeRect(bx, by, tw + 16, 20);
        ctx.fillStyle = `rgba(122,31,36,${a})`;
        ctx.fillText(txt, size.w / 2, by + 3);
      }
    }
  }

  function render() {
    Array.from(viewToggleEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.view === state.view);
    });
    gammaLabel.textContent = `γ = ${state.gamma.toFixed(2)}`;
    stepRewardLabel.textContent = `step = ${state.stepReward.toFixed(2)}`;
    const labels = ['', '0.25×', '0.5×', '1×', '1.5×', '2×', '3×', '4×', '6×', '8×', '12×'];
    speedLabel.textContent = labels[state.speed] || `${state.speed}×`;

    let totalCells = 0, totalV = 0, walls = 0, maxDelta = 0;
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      if (state.grid[r][c] === 'W') { walls++; continue; }
      totalCells++; totalV += state.V[r][c];
      if (state.Vprev) maxDelta = Math.max(maxDelta, Math.abs(state.V[r][c] - state.Vprev[r][c]));
    }
    readoutEl.innerHTML = `
      <div class="row"><span>states</span><b>${totalCells}</b></div>
      <div class="row"><span>walls</span><b>${walls}</b></div>
      <div class="row"><span>sweep</span><b>${state.sweep}</b></div>
      <div class="row"><span>avg V</span><b>${formatNum(totalV / Math.max(1, totalCells))}</b></div>
      <div class="row"><span>max ΔV</span><b>${state.Vprev ? formatNum(maxDelta) : '—'}</b></div>`;
    canvasCtl.redraw();
    if (anyPulseLive() || annotationLive()) requestAnimationFrame(tickAnim);
  }

  render();
})();
