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
  const ACTIONS = [[0, -1], [1, 0], [0, 1], [-1, 0]]; // up, right, down, left

  function reward(grid, r, c) {
    const cell = grid[r][c];
    if (cell === 'G') return 1;
    if (cell === 'B') return -1;
    return 0;
  }
  function isTerminal(grid, r, c) {
    return grid[r][c] === 'G' || grid[r][c] === 'B';
  }

  // Value iteration: deterministic transitions for simplicity.
  function valueIterate(grid, gamma, stepReward, iters = 50) {
    const Vgrid = Array.from({ length: H }, () => new Array(W).fill(0));
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      if (grid[r][c] === 'G') Vgrid[r][c] = 1;
      else if (grid[r][c] === 'B') Vgrid[r][c] = -1;
    }
    for (let iter = 0; iter < iters; iter++) {
      const newV = Vgrid.map((row) => row.slice());
      for (let r = 0; r < H; r++) {
        for (let c = 0; c < W; c++) {
          if (grid[r][c] === 'W' || isTerminal(grid, r, c)) continue;
          let best = -Infinity;
          for (const [dx, dy] of ACTIONS) {
            const nr = r + dy, nc = c + dx;
            if (nr < 0 || nr >= H || nc < 0 || nc >= W || grid[nr][nc] === 'W') {
              // Bounce back
              const v = stepReward + gamma * Vgrid[r][c];
              if (v > best) best = v;
            } else {
              const r_imm = isTerminal(grid, nr, nc) ? reward(grid, nr, nc) : stepReward;
              const v = r_imm + gamma * Vgrid[nr][nc];
              if (v > best) best = v;
            }
          }
          newV[r][c] = best;
        }
      }
      // Snap goal cells to their reward
      for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
        if (grid[r][c] === 'G') newV[r][c] = 1;
        else if (grid[r][c] === 'B') newV[r][c] = -1;
      }
      // Convergence check
      let maxDiff = 0;
      for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
        maxDiff = Math.max(maxDiff, Math.abs(newV[r][c] - Vgrid[r][c]));
      }
      Vgrid.length = 0;
      Vgrid.push(...newV);
      if (maxDiff < 1e-6) break;
    }
    // Policy
    const policy = Array.from({ length: H }, () => new Array(W).fill(-1));
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        if (grid[r][c] === 'W' || isTerminal(grid, r, c)) continue;
        let best = -Infinity, bestA = -1;
        ACTIONS.forEach(([dx, dy], a) => {
          const nr = r + dy, nc = c + dx;
          let v;
          if (nr < 0 || nr >= H || nc < 0 || nc >= W || grid[nr][nc] === 'W') {
            v = stepReward + gamma * Vgrid[r][c];
          } else {
            const r_imm = isTerminal(grid, nr, nc) ? reward(grid, nr, nc) : stepReward;
            v = r_imm + gamma * Vgrid[nr][nc];
          }
          if (v > best) { best = v; bestA = a; }
        });
        policy[r][c] = bestA;
      }
    }
    return { V: Vgrid, policy };
  }

  const state = {
    grid: makeGrid(),
    gamma: 0.9,
    stepReward: -0.04,
    view: 'value', // 'value' or 'policy'
    iters: 50,
    runStep: 0,
    timer: null,
  };

  const gammaSlider = document.getElementById('gamma-slider');
  const gammaLabel = document.getElementById('gamma-label');
  const stepRewardSlider = document.getElementById('step-reward');
  const stepRewardLabel = document.getElementById('step-reward-label');
  const viewToggleEl = document.getElementById('view-toggle');
  const runBtn = document.getElementById('run-btn');
  const resetBtn = document.getElementById('reset-btn');
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

  gammaSlider.addEventListener('input', (e) => {
    state.gamma = +e.target.value;
    gammaLabel.textContent = `γ = ${state.gamma.toFixed(2)}`;
    render();
  });
  stepRewardSlider.addEventListener('input', (e) => {
    state.stepReward = +e.target.value;
    stepRewardLabel.textContent = `step = ${state.stepReward.toFixed(2)}`;
    render();
  });
  runBtn.addEventListener('click', () => {
    state.runStep = 0;
    if (state.timer) clearInterval(state.timer);
    state.timer = setInterval(() => {
      state.runStep++;
      state.iters = Math.min(state.runStep * 4, 60);
      render();
      if (state.iters >= 60) { clearInterval(state.timer); state.timer = null; }
    }, 60);
  });
  resetBtn.addEventListener('click', () => {
    state.grid = makeGrid();
    state.iters = 50;
    state.runStep = 0;
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    render();
  });

  // Click on canvas to toggle wall
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
    if (state.grid[row][c] === 'W') state.grid[row][c] = ' ';
    else if (state.grid[row][c] === ' ') state.grid[row][c] = 'W';
    render();
  });

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    const { V: Vgrid, policy } = valueIterate(state.grid, state.gamma, state.stepReward, state.iters);

    const cellW = size.w / W;
    const cellH = size.h / H;

    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const cell = state.grid[r][c];
        const x = c * cellW, y = r * cellH;

        // Background by value
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

        // Cell border
        ctx.strokeStyle = 'rgba(38,35,32,0.30)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, cellW, cellH);

        // Value text
        if (state.view === 'value' && cell !== 'W') {
          ctx.fillStyle = cell === 'G' || cell === 'B' || Math.abs(Vgrid[r][c]) > 0.5
            ? '#fffdf6' : '#262320';
          ctx.font = '13px "JetBrains Mono", monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          let txt;
          if (cell === 'G') txt = '+1';
          else if (cell === 'B') txt = '−1';
          else txt = formatNum(Vgrid[r][c]);
          ctx.fillText(txt, x + cellW / 2, y + cellH / 2);
        }

        // Policy arrow
        if (cell === ' ' && policy[r][c] !== -1) {
          const [dx, dy] = ACTIONS[policy[r][c]];
          const cx = x + cellW / 2;
          const cy = y + cellH / 2;
          const len = Math.min(cellW, cellH) * 0.32;
          const offset = state.view === 'value' ? cellH * 0.28 : 0;
          const fromX = cx - dx * len * 0.5;
          const fromY = cy + offset - dy * len * 0.5;
          const toX = cx + dx * len * 0.5;
          const toY = cy + offset + dy * len * 0.5;
          ctx.strokeStyle = state.view === 'value' ? 'rgba(38,35,32,0.65)' : ACCENT;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(fromX, fromY);
          ctx.lineTo(toX, toY);
          ctx.stroke();
          // Arrowhead
          const ang = Math.atan2(toY - fromY, toX - fromX);
          ctx.fillStyle = state.view === 'value' ? 'rgba(38,35,32,0.65)' : ACCENT;
          ctx.beginPath();
          ctx.moveTo(toX, toY);
          ctx.lineTo(toX - 8 * Math.cos(ang - 0.4), toY - 8 * Math.sin(ang - 0.4));
          ctx.lineTo(toX - 8 * Math.cos(ang + 0.4), toY - 8 * Math.sin(ang + 0.4));
          ctx.fill();
        }
      }
    }
  }

  function render() {
    Array.from(viewToggleEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.view === state.view);
    });
    gammaLabel.textContent = `γ = ${state.gamma.toFixed(2)}`;
    stepRewardLabel.textContent = `step = ${state.stepReward.toFixed(2)}`;
    const { V: Vgrid } = valueIterate(state.grid, state.gamma, state.stepReward, state.iters);
    let totalCells = 0, totalV = 0, walls = 0;
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      if (state.grid[r][c] === 'W') walls++;
      else { totalCells++; totalV += Vgrid[r][c]; }
    }
    readoutEl.innerHTML = `
      <div class="row"><span>states</span><b>${totalCells}</b></div>
      <div class="row"><span>walls</span><b>${walls}</b></div>
      <div class="row"><span>iter</span><b>${state.iters}</b></div>
      <div class="row"><span>avg V</span><b>${formatNum(totalV / Math.max(1, totalCells))}</b></div>`;
    canvasCtl.redraw();
  }

  render();
})();
