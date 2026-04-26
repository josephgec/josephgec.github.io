/* Vol. XXXIV — Federated Learning */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK_FADE } = V;

  // ─── Toy training problem ──────────────────────────────────────────────
  // Each client trains a 1D weight w toward its own per-client target.
  // With IID, all targets are near 0.5 (clients see roughly the same data);
  // with non-IID, targets are spread across [-0.2, 1.2] (clients see different classes).
  // Each client also has a local data histogram (5 bins) which shows what its data
  // distribution looks like — uniform for IID, peaked-on-one-class for non-IID.

  // Phase machine for a single round, with animated packets.
  // PHASES (loop): 'broadcast' → 'local' → 'upload' → 'aggregate' → 'idle'
  const PHASE_DURS = {
    broadcast: 28,   // frames
    local:     22,
    upload:    32,
    aggregate: 16,
    idle:      8,
  };
  const PHASE_ORDER = ['broadcast', 'local', 'upload', 'aggregate', 'idle'];

  const state = {
    numClients: 6,
    // Default to non-IID so client histograms visibly differ on first paint —
    // the picture makes "data stays local" instantly meaningful (each device
    // sees a *different* slice of the world, but the global model still
    // converges through gradient averaging).
    distribution: 'non-iid', // 'iid' | 'non-iid'
    dropout: 0,
    dpSigma: 0,          // differential-privacy noise stddev
    clients: [],         // {target, w, dataSize, hist[5], dropped}
    serverW: 0,
    round: 0,
    lossHistory: [],
    running: false,
    timer: null,

    // Animation phase state
    phase: 'idle',
    phaseFrame: 0,
    // Per-round bookkeeping (set when round begins)
    roundDropped: [],     // bools per client for *this* round
    roundLocalW: [],      // each client's local w after local SGD this round
    roundLocalNoise: [],  // jitter added to each client's update (for visual)
  };

  // ─── Build clients (targets + data histograms) ────────────────────────
  function makeClients() {
    const out = [];
    for (let i = 0; i < state.numClients; i++) {
      let target;
      let hist; // 5-bin histogram
      // Slight variation in data sizes so n_k weighting actually matters.
      const dataSize = 60 + Math.floor(Math.random() * 80);
      if (state.distribution === 'iid') {
        target = 0.5 + (Math.random() - 0.5) * 0.05;
        // Roughly uniform over 5 bins, with mild noise.
        hist = [0.18, 0.20, 0.22, 0.20, 0.20].map((v) => v + (Math.random() - 0.5) * 0.06);
      } else {
        target = (i / Math.max(1, state.numClients - 1)) * 1.4 - 0.2;
        // Peaked on one class — bin index proportional to position
        const peak = Math.floor((i / Math.max(1, state.numClients - 1)) * 4 + 0.001);
        hist = [0.05, 0.05, 0.05, 0.05, 0.05];
        hist[peak] = 0.7;
        // Distribute the rest a bit
        for (let b = 0; b < 5; b++) hist[b] += (Math.random() * 0.04);
      }
      // Normalize
      const s = hist.reduce((a, b) => a + b, 0);
      hist = hist.map((v) => Math.max(0.005, v / s));

      // Local data cloud: a deterministic scatter of dots that physically embodies
      // "the client's data lives here, on this device, and never moves." About one
      // dot per ~5 examples, jittered around the client circle. Dots never animate
      // — that's the whole point. Color tracks the histogram peak (so non-IID
      // clouds look visibly different across clients).
      const dotCount = Math.min(28, Math.max(8, Math.round(dataSize / 5)));
      const dots = [];
      // Deterministic per-client jitter using a tiny LCG seeded by index.
      let seed = (i + 1) * 9301 + 49297;
      const rng = () => { seed = (seed * 9301 + 49297) % 233280; return seed / 233280; };
      for (let d = 0; d < dotCount; d++) {
        // Place dots on the LEFT and RIGHT flanks of the client circle — leaving
        // the top open so broadcast/upload packets enter+exit cleanly without
        // visually passing through the data cloud, and the bottom open for the
        // n_k label and histogram.
        const onLeft = rng() < 0.5;
        // Right flank: angle in (-π/4, π/4); left flank: angle in (3π/4, 5π/4).
        const center = onLeft ? Math.PI : 0;
        const angle = center + (rng() - 0.5) * (Math.PI / 2);
        const r = 24 + rng() * 14;
        dots.push({ dx: Math.cos(angle) * r, dy: Math.sin(angle) * r * 0.55 });
      }
      out.push({ target, w: 0, dataSize, hist, dots, dropped: false });
    }
    return out;
  }

  function reset() {
    state.clients = makeClients();
    state.serverW = 0;
    state.round = 0;
    state.lossHistory = [];
    state.phase = 'idle';
    state.phaseFrame = 0;
    state.roundDropped = state.clients.map(() => false);
    state.roundLocalW = state.clients.map(() => 0);
    state.roundLocalNoise = state.clients.map(() => 0);
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    state.running = false;
    if (typeof runBtn !== 'undefined' && runBtn) runBtn.textContent = '▷ Run rounds';
  }
  state.clients = makeClients();
  state.roundDropped = state.clients.map(() => false);
  state.roundLocalW = state.clients.map(() => 0);
  state.roundLocalNoise = state.clients.map(() => 0);

  function globalLoss() {
    if (!state.clients.length) return 0;
    let s = 0;
    for (const c of state.clients) s += (state.serverW - c.target) ** 2;
    return s / state.clients.length;
  }

  // ─── DOM ──────────────────────────────────────────────────────────────
  const clientsSlider = document.getElementById('clients-slider');
  const clientsLabel = document.getElementById('clients-label');
  const distributionEl = document.getElementById('distribution');
  const dpSlider = document.getElementById('dp-slider');
  const dpLabel = document.getElementById('dp-label');
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
  if (dpSlider) dpSlider.addEventListener('input', (e) => {
    state.dpSigma = +e.target.value;
    dpLabel.textContent = state.dpSigma === 0
      ? 'σ = 0.00 (off)'
      : `σ = ${state.dpSigma.toFixed(2)}`;
  });
  dropoutSlider.addEventListener('input', (e) => {
    state.dropout = +e.target.value;
    dropoutLabel.textContent = `${(state.dropout * 100).toFixed(0)}% drop per round`;
  });
  runBtn.addEventListener('click', () => { if (state.running) stopRun(); else startRun(); });
  resetBtn.addEventListener('click', () => { reset(); render(); });

  // ─── Round logic ─────────────────────────────────────────────────────
  // Begin a round: pick which clients drop, simulate local SGD outcomes (we'll commit to serverW
  // at the 'aggregate' phase). The animation just walks through phases visually.
  function beginRound() {
    state.roundDropped = state.clients.map(() => Math.random() < state.dropout);
    const lr = 0.5;
    state.roundLocalW = state.clients.map((c) => {
      let w = state.serverW;
      for (let k = 0; k < 3; k++) w = w - lr * (w - c.target);
      return w;
    });
    // Sample DP noise once per round (so visual jitter is consistent within a round).
    state.roundLocalNoise = state.clients.map(() => {
      // Box-Muller
      let u = 0, v = 0;
      while (u === 0) u = Math.random();
      while (v === 0) v = Math.random();
      const g = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      return state.dpSigma * g;
    });
  }

  // Commit the round to serverW (called at 'aggregate' phase)
  function commitRound() {
    let num = 0, den = 0;
    state.clients.forEach((c, i) => {
      if (state.roundDropped[i]) return;
      const wWithNoise = state.roundLocalW[i] + state.roundLocalNoise[i];
      num += c.dataSize * wWithNoise;
      den += c.dataSize;
    });
    if (den > 0) state.serverW = num / den;
    state.round++;
    state.lossHistory.push(globalLoss());
    if (state.lossHistory.length > 60) state.lossHistory.shift();
  }

  function startRun() {
    state.running = true;
    runBtn.textContent = '⏸ Pause';
    // 60ms tick — phases tick by frame counts
    state.timer = setInterval(tick, 60);
  }
  function stopRun() {
    state.running = false;
    runBtn.textContent = '▷ Run rounds';
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  function tick() {
    // Phase machine
    if (state.phase === 'idle') {
      // Begin a new round
      beginRound();
      state.phase = 'broadcast';
      state.phaseFrame = 0;
    } else {
      state.phaseFrame++;
      if (state.phaseFrame >= PHASE_DURS[state.phase]) {
        // Transition
        const idx = PHASE_ORDER.indexOf(state.phase);
        const nextPhase = PHASE_ORDER[(idx + 1) % PHASE_ORDER.length];
        if (state.phase === 'aggregate') {
          // Apply the round at end of aggregate phase
          commitRound();
        }
        state.phase = nextPhase;
        state.phaseFrame = 0;
      }
    }
    render();
  }

  // ─── Canvas ──────────────────────────────────────────────────────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  // Easing for packet animations
  function easeInOut(t) { return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; }
  function lerp(a, b, t) { return a + (b - a) * t; }

  function draw(ctx, size) {
    const padX = 24;
    const lossPaneX = size.w * 0.70;
    const lossPaneW = size.w - lossPaneX - padX;
    const stageW = lossPaneX - padX * 2;

    const serverY = 70;
    const clientCenterY = size.h - 130;

    // Phase header
    ctx.fillStyle = INK_FADE;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    const phaseDescription = {
      idle:      state.distribution === 'non-iid'
        ? 'idle · each client below has a DIFFERENT local data histogram — yet they will train one shared model. click ▷ Run rounds.'
        : 'idle · click ▷ Run rounds',
      broadcast: 'phase 1 · server broadcasts global w to all clients',
      local:     'phase 2 · each client trains locally on its private data',
      upload:    'phase 3 · clients upload Δw (with optional DP noise)',
      aggregate: 'phase 4 · server averages, weighted by data size n_k',
    };
    ctx.fillText(phaseDescription[state.phase] || '', padX, 18);

    // ─── Tiny packet-color key (top, right-of-stage) ──
    // Two swatches: white-with-oxblood-ring = global weights; oxblood = gradient updates.
    // Helps newcomers read the animation without flipping back to the caption.
    {
      const keyY = 14;
      const keyX0 = padX + stageW - 230;
      // White packet swatch
      ctx.fillStyle = '#fffdf6';
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(keyX0, keyY, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = INK_FADE;
      ctx.font = 'italic 10px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'left';
      ctx.fillText('w (down)', keyX0 + 8, keyY + 3);
      // Orange packet swatch
      ctx.fillStyle = ACCENT;
      ctx.beginPath();
      ctx.arc(keyX0 + 70, keyY, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = INK_FADE;
      ctx.fillText('Δw (up)', keyX0 + 78, keyY + 3);
      // Note: data stays local
      ctx.fillStyle = 'rgba(122,31,36,0.65)';
      ctx.font = 'italic 10px "Source Serif 4", Georgia, serif';
      ctx.fillText('(raw data never leaves the client)', keyX0 + 130, keyY + 3);
    }

    // Server
    const serverCx = padX + stageW / 2;
    ctx.fillStyle = ACCENT;
    ctx.beginPath();
    ctx.arc(serverCx, serverY, 28, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#fffdf6';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#fffdf6';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('server', serverCx, serverY - 4);
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillText(`w=${formatNum(state.serverW)}`, serverCx, serverY + 8);
    ctx.textBaseline = 'alphabetic';

    // Clients
    const cw = stageW / state.numClients;
    const clientPositions = [];
    state.clients.forEach((c, i) => {
      const cx = padX + cw * i + cw / 2;
      const cy = clientCenterY;
      clientPositions.push({ cx, cy });
    });

    // Phase-progress fraction (0..1)
    const phaseDur = PHASE_DURS[state.phase];
    const phaseT = phaseDur > 0 ? state.phaseFrame / phaseDur : 0;
    const phaseTE = easeInOut(phaseT);

    // Background connection lines — always drawn very faint
    state.clients.forEach((c, i) => {
      const { cx, cy } = clientPositions[i];
      ctx.strokeStyle = 'rgba(38,35,32,0.10)';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(serverCx, serverY + 28);
      ctx.lineTo(cx, cy - 22);
      ctx.stroke();
    });

    // ─── Phase animations: packets ────────────────────────────────────
    // BROADCAST: white packets fan out from server to each client.
    if (state.phase === 'broadcast') {
      state.clients.forEach((c, i) => {
        const { cx, cy } = clientPositions[i];
        const sx = serverCx, sy = serverY + 28;
        const ex = cx, ey = cy - 22;
        const px = lerp(sx, ex, phaseTE);
        const py = lerp(sy, ey, phaseTE);
        // Packet
        ctx.fillStyle = '#fffdf6';
        ctx.strokeStyle = ACCENT;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(px, py, 4.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        // Trailing dim line
        ctx.strokeStyle = 'rgba(122,31,36,0.30)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(px, py);
        ctx.stroke();
      });
    }

    // LOCAL: subtle pulse on each client to indicate training
    if (state.phase === 'local') {
      state.clients.forEach((c, i) => {
        if (state.roundDropped[i]) return;
        const { cx, cy } = clientPositions[i];
        const pulse = 0.5 + 0.5 * Math.sin(state.phaseFrame * 0.6);
        ctx.strokeStyle = `rgba(122,31,36,${0.25 + 0.5 * pulse})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(cx, cy, 22 + pulse * 4, 0, Math.PI * 2);
        ctx.stroke();
      });
    }

    // UPLOAD: orange gradient packets travel client → server.
    // Each non-dropped client emits a packet; if DP enabled, add visible jitter.
    if (state.phase === 'upload') {
      state.clients.forEach((c, i) => {
        if (state.roundDropped[i]) return;
        const { cx, cy } = clientPositions[i];
        const sx = cx, sy = cy - 22;
        const ex = serverCx, ey = serverY + 28;
        const px = lerp(sx, ex, phaseTE);
        const py = lerp(sy, ey, phaseTE);
        // DP jitter
        const jitterAmp = state.dpSigma * 18; // visual amplification
        const jx = state.dpSigma > 0
          ? jitterAmp * Math.sin(state.phaseFrame * 0.9 + i * 1.3) * (1 - phaseTE * 0.5)
          : 0;
        const jy = state.dpSigma > 0
          ? jitterAmp * Math.cos(state.phaseFrame * 1.1 + i * 0.7) * (1 - phaseTE * 0.5)
          : 0;
        // Packet
        ctx.fillStyle = ACCENT;
        ctx.beginPath();
        ctx.arc(px + jx, py + jy, 4, 0, Math.PI * 2);
        ctx.fill();
        // Trail
        ctx.strokeStyle = 'rgba(122,31,36,0.50)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(px + jx, py + jy);
        ctx.stroke();
        // If DP enabled, draw a tiny "+noise" wisp around the source
        if (state.dpSigma > 0 && phaseT < 0.3) {
          ctx.strokeStyle = 'rgba(38,35,32,0.40)';
          ctx.lineWidth = 0.6;
          ctx.beginPath();
          ctx.arc(sx, sy, 6 + Math.sin(state.phaseFrame * 0.8) * 2, 0, Math.PI * 2);
          ctx.stroke();
        }
      });
    }

    // AGGREGATE: packets pile near server, then a single white packet emerges
    if (state.phase === 'aggregate') {
      // First half: packets piling up (concentric small dots near server)
      const piling = phaseT < 0.5;
      if (piling) {
        const liveClients = state.clients.filter((c, i) => !state.roundDropped[i]);
        liveClients.forEach((c, k) => {
          const angle = (k / liveClients.length) * Math.PI * 2;
          const r = 22 - phaseT * 20;
          ctx.fillStyle = ACCENT;
          ctx.beginPath();
          ctx.arc(serverCx + Math.cos(angle) * r, serverY + Math.sin(angle) * r, 3, 0, Math.PI * 2);
          ctx.fill();
        });
      } else {
        // Single white packet emerging
        const t = (phaseT - 0.5) * 2;
        const r = t * 36;
        ctx.fillStyle = '#fffdf6';
        ctx.strokeStyle = ACCENT;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(serverCx, serverY, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        // Outward ring
        ctx.strokeStyle = `rgba(122,31,36,${1 - t})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(serverCx, serverY, r, 0, Math.PI * 2);
        ctx.stroke();
      }
      // Label
      ctx.fillStyle = 'rgba(38,35,32,0.55)';
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText('weighted average', serverCx, serverY - 44);
      // FedAvg formula anchored to the picture: at the moment of aggregation,
      // show the equation the server is literally executing.
      ctx.fillStyle = '#262320';
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.fillText('w ← Σₖ (nₖ / n) · wₖ', serverCx, serverY - 56);
    }

    // ─── Draw clients on top of lines/packets ─────────────────────────
    state.clients.forEach((c, i) => {
      const { cx, cy } = clientPositions[i];
      const dropped = state.roundDropped[i] && (state.phase === 'local' || state.phase === 'upload' || state.phase === 'aggregate');

      // Local data cloud — static dots that NEVER animate, encircling the client.
      // This is the "data stays here" picture: while w packets and Δw packets fly
      // overhead, the actual data points sit on the device unchanged.
      // Color tracks the histogram peak so non-IID clouds look class-distinct.
      const peakBin = c.hist.indexOf(Math.max(...c.hist));
      const cloudColor = state.distribution === 'non-iid'
        ? `hsla(${(peakBin / 5) * 300 + 10}, 38%, 35%, ${dropped ? 0.18 : 0.55})`
        : `rgba(38,35,32,${dropped ? 0.18 : 0.45})`;
      ctx.fillStyle = cloudColor;
      c.dots.forEach((d) => {
        ctx.beginPath();
        ctx.arc(cx + d.dx, cy + d.dy, 1.4, 0, Math.PI * 2);
        ctx.fill();
      });

      // Client circle (drawn ON TOP of dots so the circle visually "owns" them).
      ctx.save();
      ctx.globalAlpha = dropped ? 0.35 : 1;
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
      ctx.restore();

      // Dropped X mark + label
      if (dropped) {
        ctx.strokeStyle = 'rgba(38,35,32,0.65)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(cx - 14, cy - 14); ctx.lineTo(cx + 14, cy + 14);
        ctx.moveTo(cx + 14, cy - 14); ctx.lineTo(cx - 14, cy + 14);
        ctx.stroke();
        ctx.fillStyle = 'rgba(38,35,32,0.65)';
        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'alphabetic';
        ctx.fillText('offline', cx, cy + 32);
      } else {
        ctx.fillStyle = INK_FADE;
        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'alphabetic';
        ctx.fillText(`n_k=${c.dataSize}`, cx, cy + 32);
      }

      // Per-client local data histogram
      const histY = cy + 42;
      const histW = 32;
      const histH = 16;
      const binW = histW / 5;
      const histX = cx - histW / 2;
      // Frame
      ctx.strokeStyle = 'rgba(38,35,32,0.20)';
      ctx.lineWidth = 0.6;
      ctx.strokeRect(histX, histY, histW, histH);
      // Bars
      const maxBin = Math.max(...c.hist);
      for (let b = 0; b < 5; b++) {
        const bh = (c.hist[b] / maxBin) * (histH - 2);
        ctx.fillStyle = dropped
          ? 'rgba(38,35,32,0.20)'
          : (state.distribution === 'non-iid' ? 'rgba(122,31,36,0.50)' : 'rgba(38,35,32,0.55)');
        ctx.fillRect(histX + b * binW + 1, histY + (histH - bh) - 1, binW - 2, bh);
      }
      // Histogram label
      ctx.fillStyle = INK_FADE;
      ctx.font = '8px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('local data', cx, histY + histH + 10);

      // Influence weight (n_k / n) — appears during aggregate so newcomers see WHY
      // "weighted by data size" matters. The client with the biggest fraction has
      // the biggest say in the next w.
      if (!dropped && (state.phase === 'aggregate' || state.phase === 'upload')) {
        const liveTotal = state.clients.reduce(
          (s, cc, ii) => s + (state.roundDropped[ii] ? 0 : cc.dataSize), 0
        );
        const w_i = liveTotal > 0 ? c.dataSize / liveTotal : 0;
        ctx.fillStyle = 'rgba(122,31,36,0.85)';
        ctx.font = 'italic 10px "Source Serif 4", Georgia, serif';
        ctx.textAlign = 'center';
        ctx.fillText(`weight ${(w_i * 100).toFixed(0)}%`, cx, histY + histH + 22);
      }
    });

    // ─── Loss curve (right pane) ────────────────────────────────────
    const lossPaneY = 50;
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
      // Y/X axis labels
      ctx.fillStyle = INK_FADE;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(formatNum(maxL), lossPaneX - 4, lossPaneY + 10);
      ctx.fillText('0', lossPaneX - 4, lossPaneY + lossPaneH);
      ctx.textAlign = 'left';
      ctx.fillText(`loss = ${formatNum(state.lossHistory[state.lossHistory.length - 1])}`, lossPaneX + 6, lossPaneY + lossPaneH - 8);
      ctx.fillText(`round = ${state.round}`, lossPaneX + 6, lossPaneY + lossPaneH - 22);
    } else {
      ctx.fillStyle = INK_FADE;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText('no rounds yet', lossPaneX + lossPaneW / 2, lossPaneY + lossPaneH / 2);
    }

    // DP / non-IID banner under loss pane (small status indicators)
    let yBan = lossPaneY + lossPaneH + 14;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    if (state.dpSigma > 0) {
      ctx.fillStyle = 'rgba(122,31,36,0.85)';
      ctx.fillText(`DP σ = ${state.dpSigma.toFixed(2)}`, lossPaneX, yBan);
    }
    if (state.distribution === 'non-iid') {
      ctx.fillStyle = 'rgba(122,31,36,0.85)';
      ctx.fillText('non-IID', lossPaneX + 80, yBan);
    }
  }

  function render() {
    Array.from(distributionEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.dist === state.distribution);
    });
    clientsSlider.value = state.numClients;
    clientsLabel.textContent = `${state.numClients} clients`;
    if (dpSlider) dpSlider.value = state.dpSigma;
    if (dpLabel) dpLabel.textContent = state.dpSigma === 0
      ? 'σ = 0.00 (off)'
      : `σ = ${state.dpSigma.toFixed(2)}`;
    dropoutSlider.value = state.dropout;
    dropoutLabel.textContent = `${(state.dropout * 100).toFixed(0)}% drop per round`;

    const totalN = state.clients.reduce((s, c) => s + c.dataSize, 0);
    readoutEl.innerHTML = `
      <div class="row"><span>round</span><b>${state.round}</b></div>
      <div class="row"><span>phase</span><b>${state.phase}</b></div>
      <div class="row"><span>clients</span><b>${state.numClients}</b></div>
      <div class="row"><span>total n</span><b>${totalN}</b></div>
      <div class="row"><span>global w</span><b>${formatNum(state.serverW)}</b></div>
      <div class="row"><span>global loss</span><b>${formatNum(globalLoss())}</b></div>`;
    canvasCtl.redraw();
  }

  render();
})();
