/* Vol. XXXIV — Federated Learning */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK_FADE } = V;

  // Simulate FedAvg on a 1D quadratic loss. Each "client" has data with a slightly
  // different optimum; the global average converges to the population optimum.
  // With non-IID flag, the spread of per-client optima widens.
  const state = {
    numClients: 6,
    distribution: 'iid',
    dropout: 0,
    clients: [],
    serverW: 0,
    round: 0,
    activeUploads: [], // for animation
    activeBroadcast: false,
    lossHistory: [],
    running: false,
    timer: null,
  };

  function makeClients() {
    const out = [];
    for (let i = 0; i < state.numClients; i++) {
      let target;
      if (state.distribution === 'iid') {
        // All clients have target near 0.5
        target = 0.5 + (Math.random() - 0.5) * 0.05;
      } else {
        // Non-IID: clients have very different targets
        target = (i / Math.max(1, state.numClients - 1)) * 1.4 - 0.2;
      }
      out.push({ target, w: 0 });
    }
    return out;
  }

  function reset() {
    state.clients = makeClients();
    state.serverW = 0;
    state.round = 0;
    state.lossHistory = [];
    state.activeUploads = [];
    state.activeBroadcast = false;
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    state.running = false;
    if (typeof runBtn !== 'undefined' && runBtn) runBtn.textContent = '▷ Run rounds';
  }
  // Initialize clients (button text is set later once runBtn is defined)
  state.clients = makeClients();

  function globalLoss() {
    // Population-average loss: average over all clients of (serverW - target)²
    let s = 0;
    for (const c of state.clients) s += (state.serverW - c.target) ** 2;
    return s / state.clients.length;
  }

  function fedRound() {
    // Broadcast: each client receives serverW
    state.activeBroadcast = true;
    // Each client trains locally (a few SGD steps toward its target)
    const lr = 0.5;
    const localUpdates = [];
    state.clients.forEach((c, i) => {
      // Simulate dropout
      if (Math.random() < state.dropout) return;
      let w = state.serverW;
      for (let k = 0; k < 3; k++) {
        w = w - lr * (w - c.target);
      }
      c.w = w;
      localUpdates.push({ idx: i, w });
    });
    state.activeUploads = localUpdates.map((u) => u.idx);
    // Aggregate (FedAvg = simple mean since all clients have same data size in this toy)
    if (localUpdates.length > 0) {
      const avg = localUpdates.reduce((s, u) => s + u.w, 0) / localUpdates.length;
      state.serverW = avg;
    }
    state.round++;
    state.lossHistory.push(globalLoss());
    if (state.lossHistory.length > 50) state.lossHistory.shift();
  }

  const clientsSlider = document.getElementById('clients-slider');
  const clientsLabel = document.getElementById('clients-label');
  const distributionEl = document.getElementById('distribution');
  const dropoutSlider = document.getElementById('dropout-slider');
  const dropoutLabel = document.getElementById('dropout-label');
  const runBtn = document.getElementById('run-btn');
  const resetBtn = document.getElementById('reset-btn');
  const readoutEl = document.getElementById('readout');
  const canvas = document.getElementById('canvas');

  ['IID', 'Non-IID'].forEach((lbl, i) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = lbl;
    b.dataset.dist = i === 0 ? 'iid' : 'non-iid';
    b.addEventListener('click', () => { state.distribution = b.dataset.dist; reset(); render(); });
    distributionEl.appendChild(b);
  });

  clientsSlider.addEventListener('input', (e) => {
    state.numClients = +e.target.value;
    clientsLabel.textContent = `${state.numClients} clients`;
    reset();
    render();
  });
  dropoutSlider.addEventListener('input', (e) => {
    state.dropout = +e.target.value;
    dropoutLabel.textContent = `${(state.dropout * 100).toFixed(0)}% drop per round`;
  });
  runBtn.addEventListener('click', () => { if (state.running) stopRun(); else startRun(); });
  resetBtn.addEventListener('click', () => { reset(); render(); });

  function startRun() {
    state.running = true;
    runBtn.textContent = '⏸ Pause';
    state.timer = setInterval(() => {
      fedRound();
      render();
    }, 700);
  }
  function stopRun() {
    state.running = false;
    runBtn.textContent = '▷ Run rounds';
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    const padX = 24;
    const serverY = 60;
    const clientsY = size.h - 110;
    const lossPaneX = size.w * 0.7;

    // Server (top center, but to the left of loss pane)
    const serverCx = (lossPaneX - padX) / 2 + padX;
    ctx.fillStyle = ACCENT;
    ctx.beginPath();
    ctx.arc(serverCx, serverY, 26, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fffdf6';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('server', serverCx, serverY);
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(`global w = ${formatNum(state.serverW)}`, serverCx, serverY + 42);

    // Clients
    const clientAreaW = lossPaneX - padX - padX;
    const cw = clientAreaW / state.numClients;
    state.clients.forEach((c, i) => {
      const cx = padX + cw * i + cw / 2;
      const cy = clientsY;

      // Connection line (color depends on phase)
      const isUp = state.activeUploads.includes(i);
      ctx.strokeStyle = isUp ? ACCENT : 'rgba(38,35,32,0.20)';
      ctx.lineWidth = isUp ? 1.5 : 0.7;
      ctx.beginPath();
      ctx.moveTo(serverCx, serverY + 26);
      ctx.lineTo(cx, cy - 18);
      ctx.stroke();

      // Client circle
      ctx.fillStyle = 'rgba(38,35,32,0.20)';
      ctx.beginPath();
      ctx.arc(cx, cy, 18, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = 'rgba(38,35,32,0.50)';
      ctx.lineWidth = 1;
      ctx.stroke();
      // ID
      ctx.fillStyle = '#262320';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`C${i + 1}`, cx, cy);
      ctx.fillStyle = INK_FADE;
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(`t=${formatNum(c.target)}`, cx, cy + 32);

      // Local data dot for this client (target shown as accent on a small bar)
      const barY = cy + 46;
      const barH = 6;
      ctx.fillStyle = 'rgba(38,35,32,0.10)';
      ctx.fillRect(cx - 16, barY, 32, barH);
      // local target marker
      const localX = cx - 16 + (c.target / 1.5) * 32;
      ctx.fillStyle = ACCENT;
      ctx.fillRect(localX - 1, barY - 2, 2, barH + 4);
    });

    // Loss curve (right pane)
    const lossPaneW = size.w - lossPaneX - padX;
    const lossPaneY = 60;
    const lossPaneH = size.h - lossPaneY - 30;
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.strokeRect(lossPaneX, lossPaneY, lossPaneW, lossPaneH);
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('global loss / round', lossPaneX, lossPaneY - 6);

    if (state.lossHistory.length > 1) {
      const maxL = Math.max(...state.lossHistory);
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 2;
      ctx.beginPath();
      state.lossHistory.forEach((l, i) => {
        const x = lossPaneX + (i / (state.lossHistory.length - 1)) * lossPaneW;
        const y = lossPaneY + lossPaneH - (l / maxL) * lossPaneH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.fillStyle = INK_FADE;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`loss = ${formatNum(state.lossHistory[state.lossHistory.length - 1])}`, lossPaneX + lossPaneW - 6, lossPaneY + 14);
    }
  }

  function render() {
    Array.from(distributionEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.dist === state.distribution);
    });
    clientsSlider.value = state.numClients;
    clientsLabel.textContent = `${state.numClients} clients`;
    dropoutSlider.value = state.dropout;
    dropoutLabel.textContent = `${(state.dropout * 100).toFixed(0)}% drop per round`;
    readoutEl.innerHTML = `
      <div class="row"><span>round</span><b>${state.round}</b></div>
      <div class="row"><span>clients</span><b>${state.numClients}</b></div>
      <div class="row"><span>global w</span><b>${formatNum(state.serverW)}</b></div>
      <div class="row"><span>global loss</span><b>${formatNum(globalLoss())}</b></div>`;
    canvasCtl.redraw();
  }

  render();
})();
