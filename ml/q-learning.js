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
  // Action order: 0=up, 1=right, 2=down, 3=left.
  const ACTIONS = [[0, -1], [1, 0], [0, 1], [-1, 0]];
  const ACTION_GLYPHS = ['↑', '→', '↓', '←'];

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

  const TRAIL_MAX = 12;
  const REWARD_HISTORY_MAX = 60;

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
    lastStep: null,        // { r, c, a, td, exploring, reward, ts }
    flash: null,           // { r, c, a, ts }  triggers wedge flash
    trail: [],             // [{r, c}, ...] recent agent positions
    epReturn: 0,           // running return for current episode
    rewardHistory: [],     // returns of recent finished episodes (running avg input)
  };

  // Warm-start: run a few hundred Q-learning steps before first paint so the wedges
  // are partially filled, the agent is mid-training, and the reward curve has shape.
  // Then quiet the visual side-effects so the first paint is calm — the populated
  // Q-table is the story, not a stale flash from the final warm-step.
  (function warmStart() {
    for (let i = 0; i < 600; i++) takeStep();
    state.flash = null;
    state.lastStep = null;
    state.trail = [];
    state.epReturn = 0;
    state.pos = [H - 1, 0]; // park agent at start, ready for the user's first Train click
  })();

  function reset() {
    state.pos = [H - 1, 0];
    state.episodes++;
    state.rewardHistory.push(state.epReturn);
    if (state.rewardHistory.length > REWARD_HISTORY_MAX) state.rewardHistory.shift();
    state.epReturn = 0;
    state.trail = [];
  }

  function takeStep() {
    const [r, c] = state.pos;
    if (isTerminal(state.grid, r, c)) { reset(); return; }
    let action;
    let exploring = false;
    if (Math.random() < state.eps) {
      action = Math.floor(Math.random() * 4);
      exploring = true;
    } else {
      let best = -Infinity, bestA = 0;
      const ties = [];
      for (let a = 0; a < 4; a++) {
        if (state.Q[r][c][a] > best) { best = state.Q[r][c][a]; bestA = a; ties.length = 0; ties.push(a); }
        else if (state.Q[r][c][a] === best) ties.push(a);
      }
      action = ties.length ? ties[Math.floor(Math.random() * ties.length)] : bestA;
    }
    const [dx, dy] = ACTIONS[action];
    let nr = r + dy, nc = c + dx;
    if (nr < 0 || nr >= H || nc < 0 || nc >= W || state.grid[nr][nc] === 'W') {
      nr = r; nc = c;
    }
    const rew = isTerminal(state.grid, nr, nc) ? reward(state.grid, nr, nc) : -0.04;
    const maxQNext = isTerminal(state.grid, nr, nc) ? 0 : Math.max(...state.Q[nr][nc]);
    const oldQ = state.Q[r][c][action];
    const td = rew + state.gamma * maxQNext - oldQ;
    state.Q[r][c][action] = oldQ + state.alpha * td;

    // Trail
    state.trail.push({ r, c });
    if (state.trail.length > TRAIL_MAX) state.trail.shift();

    state.pos = [nr, nc];
    state.steps++;
    state.lastReward = rew;
    state.epReturn += rew;
    const ts = performance.now();
    state.lastStep = { r, c, a: action, td, exploring, reward: rew, ts };
    state.flash = { r, c, a: action, ts };
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
    state.lastStep = null;
    state.flash = null;
    state.trail = [];
    state.epReturn = 0;
    state.rewardHistory = [];
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

  // Drive flash animation
  function flashAlive() {
    return state.flash && (performance.now() - state.flash.ts) < 700;
  }
  function tickAnim() {
    if (flashAlive()) {
      canvasCtl.redraw();
      requestAnimationFrame(tickAnim);
    }
  }

  function draw(ctx, size) {
    const gap = 24;
    const halfW = (size.w - gap) / 2;
    const cellW1 = halfW / W;
    const cellH1 = size.h / H;
    const cellSize1 = Math.min(cellW1, cellH1) * 0.92;
    const totalGridW = cellSize1 * W;
    const totalGridH = cellSize1 * H;
    const gridX0 = (halfW - totalGridW) / 2;
    const gridY0 = (size.h - totalGridH) / 2 - 6;

    // ───── Left: world ─────
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
    // Trail (fading dots)
    state.trail.forEach((pt, i) => {
      const alpha = (i + 1) / (state.trail.length + 1) * 0.55;
      const cx = gridX0 + pt.c * cellSize1 + cellSize1 / 2;
      const cy = gridY0 + pt.r * cellSize1 + cellSize1 / 2;
      ctx.fillStyle = `rgba(122,31,36,${alpha})`;
      ctx.beginPath();
      ctx.arc(cx, cy, cellSize1 * 0.10, 0, Math.PI * 2);
      ctx.fill();
    });
    // Agent
    const [pr, pc] = state.pos;
    const acx = gridX0 + pc * cellSize1 + cellSize1 / 2;
    const acy = gridY0 + pr * cellSize1 + cellSize1 / 2;
    ctx.fillStyle = ACCENT;
    ctx.beginPath();
    ctx.arc(acx, acy, cellSize1 * 0.26, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#fffdf6';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(acx, acy, cellSize1 * 0.26, 0, Math.PI * 2);
    ctx.stroke();

    // World label + step-type indicator
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('world', gridX0 + totalGridW / 2, gridY0 + totalGridH + 14);

    if (state.lastStep) {
      const tag = state.lastStep.exploring
        ? 'EXPLORE · ε-greedy chose random'
        : 'EXPLOIT · greedy on max Q';
      const tagColor = state.lastStep.exploring ? 'rgba(38,35,32,0.7)' : 'rgba(122,31,36,0.85)';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillStyle = tagColor;
      ctx.fillText(tag, gridX0 + totalGridW / 2, gridY0 - 8);
    }

    // ───── Right: Q-table ─────
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
          ctx.fillStyle = '#fffdf6';
          ctx.fillRect(x, y, cellSize1, cellSize1);
          const Qs = state.Q[r][c];
          const maxAbs = Math.max(0.01, ...Qs.map(Math.abs));
          // Triangles in action order: up, right, down, left
          const triangles = [
            [[x, y], [x + cellSize1, y], [cx, cy]],
            [[x + cellSize1, y], [x + cellSize1, y + cellSize1], [cx, cy]],
            [[x + cellSize1, y + cellSize1], [x, y + cellSize1], [cx, cy]],
            [[x, y + cellSize1], [x, y], [cx, cy]],
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
          // Flash on the last-updated wedge
          if (state.flash && state.flash.r === r && state.flash.c === c) {
            const dt = (performance.now() - state.flash.ts) / 700;
            if (dt >= 0 && dt < 1) {
              const a = (1 - dt) * 0.85;
              ctx.fillStyle = `rgba(122,31,36,${a})`;
              ctx.beginPath();
              ctx.moveTo(...triangles[state.flash.a][0]);
              ctx.lineTo(...triangles[state.flash.a][1]);
              ctx.lineTo(...triangles[state.flash.a][2]);
              ctx.closePath();
              ctx.fill();
              ctx.strokeStyle = `rgba(122,31,36,${a})`;
              ctx.lineWidth = 2;
              ctx.beginPath();
              ctx.moveTo(...triangles[state.flash.a][0]);
              ctx.lineTo(...triangles[state.flash.a][1]);
              ctx.lineTo(...triangles[state.flash.a][2]);
              ctx.closePath();
              ctx.stroke();
            }
          }
          // Cross lines
          ctx.strokeStyle = 'rgba(38,35,32,0.12)';
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.moveTo(x, y); ctx.lineTo(x + cellSize1, y + cellSize1);
          ctx.moveTo(x + cellSize1, y); ctx.lineTo(x, y + cellSize1);
          ctx.stroke();
          // Wedge direction labels (small arrows at the four edge midpoints)
          ctx.fillStyle = 'rgba(38,35,32,0.55)';
          ctx.font = `${Math.floor(cellSize1 * 0.20)}px "JetBrains Mono", monospace`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          // up
          ctx.fillText(ACTION_GLYPHS[0], cx, y + cellSize1 * 0.14);
          // right
          ctx.fillText(ACTION_GLYPHS[1], x + cellSize1 * 0.86, cy);
          // down
          ctx.fillText(ACTION_GLYPHS[2], cx, y + cellSize1 * 0.86);
          // left
          ctx.fillText(ACTION_GLYPHS[3], x + cellSize1 * 0.14, cy);
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
    ctx.fillText('Q-table  (4 wedges per cell, one per direction)', rightX + gridX0 + totalGridW / 2, gridY0 + totalGridH + 14);

    // TD-error annotation next to flashed cell
    if (state.flash && state.lastStep) {
      const dt = (performance.now() - state.flash.ts) / 700;
      if (dt >= 0 && dt < 1) {
        const a = 1 - dt;
        const fx = rightX + gridX0 + state.flash.c * cellSize1 + cellSize1 + 4;
        const fy = gridY0 + state.flash.r * cellSize1 + cellSize1 / 2;
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = `rgba(122,31,36,${a})`;
        ctx.fillText(`TD = ${state.lastStep.td >= 0 ? '+' : ''}${state.lastStep.td.toFixed(3)}`, fx, fy);
      }
    }

    // ───── Convergence panel (running avg of episode return) ─────
    const panelH = 38;
    const panelW = size.w - 32;
    const panelX = 16;
    const panelY = size.h - panelH - 6;
    ctx.fillStyle = 'rgba(38,35,32,0.04)';
    ctx.fillRect(panelX, panelY, panelW, panelH);
    ctx.strokeStyle = 'rgba(38,35,32,0.20)';
    ctx.lineWidth = 1;
    ctx.strokeRect(panelX, panelY, panelW, panelH);
    ctx.fillStyle = INK_FADE;
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('AVG RETURN PER EPISODE', panelX + 6, panelY + 4);

    const hist = state.rewardHistory;
    if (hist.length >= 2) {
      const minV = Math.min(...hist, -1);
      const maxV = Math.max(...hist, 1);
      const range = Math.max(0.5, maxV - minV);
      // baseline (zero)
      const zeroY = panelY + panelH - 6 - ((0 - minV) / range) * (panelH - 16);
      ctx.strokeStyle = 'rgba(38,35,32,0.18)';
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(panelX + 4, zeroY); ctx.lineTo(panelX + panelW - 4, zeroY);
      ctx.stroke();
      ctx.setLineDash([]);
      // line
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      hist.forEach((v, i) => {
        const px = panelX + 4 + (i / Math.max(1, hist.length - 1)) * (panelW - 8);
        const py = panelY + panelH - 6 - ((v - minV) / range) * (panelH - 16);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
      // last value
      const last = hist[hist.length - 1];
      ctx.fillStyle = ACCENT;
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillText(`last = ${last.toFixed(2)}`, panelX + panelW - 6, panelY + 4);
    } else {
      ctx.fillStyle = 'rgba(38,35,32,0.45)';
      ctx.font = 'italic 10px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('finish a few episodes to see learning curve', panelX + panelW / 2, panelY + panelH / 2 + 4);
    }
  }

  function render() {
    epsLabel.textContent = `ε = ${state.eps.toFixed(2)}`;
    alphaLabel.textContent = `α = ${state.alpha.toFixed(2)}`;
    speedLabel.textContent = `${state.speed} steps / s`;
    epsSlider.value = state.eps;
    alphaSlider.value = state.alpha;
    speedSlider.value = state.speed;
    const tdStr = state.lastStep ? formatNum(state.lastStep.td) : '—';
    readoutEl.innerHTML = `
      <div class="row"><span>steps</span><b>${state.steps}</b></div>
      <div class="row"><span>episodes</span><b>${state.episodes}</b></div>
      <div class="row"><span>last reward</span><b>${formatNum(state.lastReward)}</b></div>
      <div class="row"><span>last TD</span><b>${tdStr}</b></div>
      <div class="row"><span>γ</span><b>${state.gamma.toFixed(2)}</b></div>`;
    canvasCtl.redraw();
    if (flashAlive()) requestAnimationFrame(tickAnim);
  }

  render();
})();
