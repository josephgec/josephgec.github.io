/* Vol. XXVII — Q-Learning */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK_FADE } = V;

  const W = 5, H = 4;
  function makeGrid() {
    const g = Array.from({ length: H }, () => new Array(W).fill(' '));
    g[1][1] = 'W';
    g[0][W - 1] = 'G';
    g[1][W - 1] = 'B';
    return g;
  }
  const ACTIONS = [[0, -1], [1, 0], [0, 1], [-1, 0]]; // up, right, down, left
  function reward(grid, r, c) {
    if (grid[r][c] === 'G') return 1;
    if (grid[r][c] === 'B') return -1;
    return -0.04;
  }
  function isTerminal(grid, r, c) {
    return grid[r][c] === 'G' || grid[r][c] === 'B';
  }

  function newQ() {
    const Q = [];
    for (let r = 0; r < H; r++) {
      const row = [];
      for (let c = 0; c < W; c++) row.push([0, 0, 0, 0]);
      Q.push(row);
    }
    return Q;
  }

  const state = {
    grid: makeGrid(),
    Q: newQ(),
    pos: [H - 1, 0],
    eps: 0.2,
    alpha: 0.4,
    gamma: 0.9,
    speed: 50,
    training: false,
    timer: null,
    steps: 0,
    episodes: 0,
    lastReward: 0,
  };

  function reset() {
    state.pos = [H - 1, 0];
    state.episodes++;
  }

  function takeStep() {
    const [r, c] = state.pos;
    if (isTerminal(state.grid, r, c)) { reset(); return; }
    let action;
    if (Math.random() < state.eps) {
      action = Math.floor(Math.random() * 4);
    } else {
      // Greedy
      let best = -Infinity, bestA = 0;
      for (let a = 0; a < 4; a++) {
        if (state.Q[r][c][a] > best) { best = state.Q[r][c][a]; bestA = a; }
      }
      action = bestA;
    }
    const [dx, dy] = ACTIONS[action];
    let nr = r + dy, nc = c + dx;
    if (nr < 0 || nr >= H || nc < 0 || nc >= W || state.grid[nr][nc] === 'W') {
      nr = r; nc = c;
    }
    const rew = isTerminal(state.grid, nr, nc) ? reward(state.grid, nr, nc) : -0.04;
    const maxQNext = isTerminal(state.grid, nr, nc) ? 0 : Math.max(...state.Q[nr][nc]);
    const oldQ = state.Q[r][c][action];
    state.Q[r][c][action] = oldQ + state.alpha * (rew + state.gamma * maxQNext - oldQ);
    state.pos = [nr, nc];
    state.steps++;
    state.lastReward = rew;
  }

  const epsSlider = document.getElementById('eps-slider');
  const epsLabel = document.getElementById('eps-label');
  const alphaSlider = document.getElementById('alpha-slider');
  const alphaLabel = document.getElementById('alpha-label');
  const speedSlider = document.getElementById('speed-slider');
  const speedLabel = document.getElementById('speed-label');
  const trainBtn = document.getElementById('train-btn');
  const resetBtn = document.getElementById('reset-btn');
  const readoutEl = document.getElementById('readout');
  const canvas = document.getElementById('canvas');

  epsSlider.addEventListener('input', (e) => { state.eps = +e.target.value; epsLabel.textContent = `ε = ${state.eps.toFixed(2)}`; });
  alphaSlider.addEventListener('input', (e) => { state.alpha = +e.target.value; alphaLabel.textContent = `α = ${state.alpha.toFixed(2)}`; });
  speedSlider.addEventListener('input', (e) => {
    state.speed = +e.target.value;
    speedLabel.textContent = `${state.speed} steps / s`;
    if (state.training) { stopTrain(); startTrain(); }
  });
  trainBtn.addEventListener('click', () => { if (state.training) stopTrain(); else startTrain(); });
  resetBtn.addEventListener('click', () => {
    stopTrain();
    state.Q = newQ();
    state.pos = [H - 1, 0];
    state.steps = 0;
    state.episodes = 0;
    state.lastReward = 0;
    render();
  });

  function startTrain() {
    state.training = true;
    trainBtn.textContent = '⏸ Pause';
    const interval = Math.max(5, 1000 / state.speed);
    state.timer = setInterval(() => { takeStep(); render(); }, interval);
  }
  function stopTrain() {
    state.training = false;
    trainBtn.textContent = '▷ Train';
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    const gap = 24;
    const halfW = (size.w - gap) / 2;
    const cellW1 = halfW / W;
    const cellH1 = size.h / H;
    const cellSize1 = Math.min(cellW1, cellH1);
    const totalGridW = cellSize1 * W;
    const totalGridH = cellSize1 * H;
    const gridX0 = (halfW - totalGridW) / 2;
    const gridY0 = (size.h - totalGridH) / 2;

    // Left: world
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const cell = state.grid[r][c];
        const x = gridX0 + c * cellSize1, y = gridY0 + r * cellSize1;
        let bg = '#fffdf6';
        if (cell === 'W') bg = 'rgba(38,35,32,0.20)';
        else if (cell === 'G') bg = 'rgba(122,31,36,0.55)';
        else if (cell === 'B') bg = 'rgba(38,35,32,0.50)';
        ctx.fillStyle = bg;
        ctx.fillRect(x, y, cellSize1, cellSize1);
        ctx.strokeStyle = 'rgba(38,35,32,0.30)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, cellSize1, cellSize1);
        if (cell === 'G') {
          ctx.fillStyle = '#fffdf6';
          ctx.font = '13px "JetBrains Mono", monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText('+1', x + cellSize1 / 2, y + cellSize1 / 2);
        } else if (cell === 'B') {
          ctx.fillStyle = '#fffdf6';
          ctx.font = '13px "JetBrains Mono", monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText('−1', x + cellSize1 / 2, y + cellSize1 / 2);
        }
      }
    }
    // Agent
    const [pr, pc] = state.pos;
    ctx.fillStyle = ACCENT;
    ctx.beginPath();
    ctx.arc(gridX0 + pc * cellSize1 + cellSize1 / 2,
            gridY0 + pr * cellSize1 + cellSize1 / 2,
            cellSize1 * 0.28, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#fffdf6';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(gridX0 + pc * cellSize1 + cellSize1 / 2,
            gridY0 + pr * cellSize1 + cellSize1 / 2,
            cellSize1 * 0.28, 0, Math.PI * 2);
    ctx.stroke();

    // World label
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('world', gridX0 + totalGridW / 2, gridY0 + totalGridH + 18);

    // Right: Q-table — each cell shows 4 triangle wedges colored by Q
    const rightX = halfW + gap;
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const x = rightX + gridX0 + c * cellSize1;
        const y = gridY0 + r * cellSize1;
        const cx = x + cellSize1 / 2, cy = y + cellSize1 / 2;
        const cell = state.grid[r][c];
        if (cell === 'W') {
          ctx.fillStyle = 'rgba(38,35,32,0.20)';
          ctx.fillRect(x, y, cellSize1, cellSize1);
        } else if (isTerminal(state.grid, r, c)) {
          ctx.fillStyle = cell === 'G' ? 'rgba(122,31,36,0.55)' : 'rgba(38,35,32,0.50)';
          ctx.fillRect(x, y, cellSize1, cellSize1);
        } else {
          // Background
          ctx.fillStyle = '#fffdf6';
          ctx.fillRect(x, y, cellSize1, cellSize1);
          // Four wedges (triangles to corners)
          const Qs = state.Q[r][c];
          const maxAbs = Math.max(0.01, ...Qs.map(Math.abs));
          // Up: top triangle
          // Right: right triangle
          // Down: bottom triangle
          // Left: left triangle
          const triangles = [
            [[x, y], [x + cellSize1, y], [cx, cy]],         // up
            [[x + cellSize1, y], [x + cellSize1, y + cellSize1], [cx, cy]], // right
            [[x + cellSize1, y + cellSize1], [x, y + cellSize1], [cx, cy]], // down
            [[x, y + cellSize1], [x, y], [cx, cy]],         // left
          ];
          Qs.forEach((q, i) => {
            const intensity = Math.min(1, Math.abs(q) / maxAbs);
            ctx.fillStyle = q >= 0
              ? `rgba(122,31,36,${0.05 + intensity * 0.55})`
              : `rgba(38,35,32,${0.04 + intensity * 0.40})`;
            ctx.beginPath();
            ctx.moveTo(...triangles[i][0]);
            ctx.lineTo(...triangles[i][1]);
            ctx.lineTo(...triangles[i][2]);
            ctx.closePath();
            ctx.fill();
          });
          // Cross lines
          ctx.strokeStyle = 'rgba(38,35,32,0.10)';
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.moveTo(x, y); ctx.lineTo(x + cellSize1, y + cellSize1);
          ctx.moveTo(x + cellSize1, y); ctx.lineTo(x, y + cellSize1);
          ctx.stroke();
        }
        ctx.strokeStyle = 'rgba(38,35,32,0.30)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, cellSize1, cellSize1);
      }
    }
    // Q-table label
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('Q-table (4 actions per cell)', rightX + gridX0 + totalGridW / 2, gridY0 + totalGridH + 18);
  }

  function render() {
    epsLabel.textContent = `ε = ${state.eps.toFixed(2)}`;
    alphaLabel.textContent = `α = ${state.alpha.toFixed(2)}`;
    speedLabel.textContent = `${state.speed} steps / s`;
    epsSlider.value = state.eps;
    alphaSlider.value = state.alpha;
    speedSlider.value = state.speed;
    readoutEl.innerHTML = `
      <div class="row"><span>steps</span><b>${state.steps}</b></div>
      <div class="row"><span>episodes</span><b>${state.episodes}</b></div>
      <div class="row"><span>last reward</span><b>${formatNum(state.lastReward)}</b></div>
      <div class="row"><span>γ</span><b>${state.gamma.toFixed(2)}</b></div>`;
    canvasCtl.redraw();
  }

  render();
})();
