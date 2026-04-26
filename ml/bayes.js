/* Vol. VII — Bayes' Theorem */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK, INK_FADE, RULE_FAINT,
          SERIF_LABEL_SM, MONO_LABEL } = V;

  const SCENARIOS = [
    {
      key: 'medical',
      name: 'Medical test',
      A: { name: 'Has disease', prior: 0.01 },
      notA: 'Healthy',
      B: 'Test positive',
      pBgivenA: 0.95,
      pBgivenNotA: 0.05,
      blurb: "A rare disease (1%) and a test that's 95% accurate. The famous result: even a positive test makes disease only ~16% likely.",
    },
    {
      key: 'spam',
      name: 'Spam filter',
      A: { name: 'Spam', prior: 0.4 },
      notA: 'Ham',
      B: 'Contains "free"',
      pBgivenA: 0.7,
      pBgivenNotA: 0.05,
      blurb: "40% of incoming mail is spam. The word 'free' appears in 70% of spam, 5% of ham. So 'free' → 90%+ spam.",
    },
    {
      key: 'rain',
      name: 'Weather',
      A: { name: 'Rain today', prior: 0.2 },
      notA: 'No rain',
      B: 'Cloudy morning',
      pBgivenA: 0.85,
      pBgivenNotA: 0.3,
      blurb: 'Rains 20% of days. Cloudy mornings precede 85% of rainy days, 30% of dry ones.',
    },
    {
      key: 'detector',
      name: 'Particle detector',
      A: { name: 'Real signal', prior: 0.001 },
      notA: 'Background',
      B: 'Detector fires',
      pBgivenA: 0.99,
      pBgivenNotA: 0.001,
      blurb: 'Vanishingly rare events with a near-perfect detector. Even tiny false-positive rates dominate.',
    },
  ];

  const state = {
    scenarioIdx: 0,
    prior: SCENARIOS[0].A.prior,
    sens: SCENARIOS[0].pBgivenA,
    fpr:  SCENARIOS[0].pBgivenNotA,
    mode: 'tree',
  };

  // ───────── DOM ─────────
  const scenariosEl = document.getElementById('scenarios');
  const knobsEl = document.getElementById('knobs');
  const readoutEl = document.getElementById('readout');
  const blurbEl = document.getElementById('scenario-blurb');
  const tabsEl = document.getElementById('mode-tabs');
  const captionEl = document.getElementById('caption');
  const canvasFrameEl = document.getElementById('canvas-frame');
  const canvas = document.getElementById('canvas');

  // Scenarios
  SCENARIOS.forEach((s, i) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = s.name;
    b.addEventListener('click', () => {
      state.scenarioIdx = i;
      state.prior = s.A.prior;
      state.sens = s.pBgivenA;
      state.fpr = s.pBgivenNotA;
      render();
    });
    scenariosEl.appendChild(b);
  });

  // Mode tabs
  Array.from(tabsEl.children).forEach((btn) => {
    btn.addEventListener('click', () => {
      state.mode = btn.dataset.mode;
      render();
    });
  });

  // Knobs (built per-scenario, since the labels depend on event names)
  function renderKnobs() {
    const sc = SCENARIOS[state.scenarioIdx];
    const knobs = [
      { key: 'prior', label: `P(${sc.A.name})`,                  sub: 'prior',          min: 0.001, max: 0.999, step: 0.001 },
      { key: 'sens',  label: `P(${sc.B} | ${sc.A.name})`,        sub: 'sensitivity',    min: 0,     max: 1,     step: 0.01  },
      { key: 'fpr',   label: `P(${sc.B} | ¬${sc.A.name})`,       sub: 'false positive', min: 0,     max: 1,     step: 0.01  },
    ];
    knobsEl.innerHTML = '';
    knobs.forEach((k) => {
      const row = document.createElement('div');
      row.innerHTML = `
        <div class="ml-param-head">
          <span class="label">${k.label} <em>· ${k.sub}</em></span>
          <b>${formatNum(state[k.key])}</b>
        </div>
        <input type="range" min="${k.min}" max="${k.max}" step="${k.step}" value="${state[k.key]}" />
      `;
      const slider = row.querySelector('input');
      slider.addEventListener('input', (e) => {
        state[k.key] = +e.target.value;
        render();
      });
      knobsEl.appendChild(row);
    });
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    if (state.mode === 'tree') drawTree(ctx, size);
    else drawSquare(ctx, size);
  }

  // ───────── Tree mode ─────────
  function drawTree(ctx, size) {
    const sc = SCENARIOS[state.scenarioIdx];
    const A = sc.A.name, notA = sc.notA, B = sc.B;
    const pA = state.prior, pNotA = 1 - pA;
    const pBA = state.sens, pNotBA = 1 - pBA;
    const pBNotA = state.fpr, pNotBNotA = 1 - pBNotA;

    const j_AB    = pA * pBA;
    const j_AnotB = pA * pNotBA;
    const j_notAB = pNotA * pBNotA;
    const j_notAnotB = pNotA * pNotBNotA;

    const padX = 24, padY = 30;
    const root = { x: padX + 10, y: size.h / 2 };
    const l1x  = padX + size.w * 0.32;
    const l2x  = padX + size.w * 0.7;
    const yA = size.h * 0.25, yNotA = size.h * 0.75;
    const ll1 = { x: l2x, y: yA - size.h * 0.12 };       // A & B
    const ll2 = { x: l2x, y: yA + size.h * 0.12 };       // A & ¬B
    const ll3 = { x: l2x, y: yNotA - size.h * 0.12 };    // ¬A & B
    const ll4 = { x: l2x, y: yNotA + size.h * 0.12 };    // ¬A & ¬B

    function branch(from, to, label, prob, weight, hot) {
      ctx.strokeStyle = hot ? ACCENT : 'rgba(38,35,32,0.40)';
      ctx.lineWidth = hot ? 3 : Math.max(1, weight * 8);
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
      const mx = (from.x + to.x) / 2, my = (from.y + to.y) / 2;
      ctx.fillStyle = hot ? ACCENT : 'rgba(38,35,32,0.65)';
      ctx.font = '12px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(label, mx, my - 6);
      ctx.font = SERIF_LABEL_SM;
      ctx.fillText(formatNum(prob), mx, my + 12);
    }

    function node(n, label, sublabel, fill) {
      ctx.fillStyle = fill || '#fffdf6';
      ctx.strokeStyle = 'rgba(38,35,32,0.50)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(n.x, n.y, 8, 0, Math.PI * 2);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = INK;
      ctx.font = SERIF_LABEL_SM;
      ctx.textAlign = 'left';
      ctx.fillText(label, n.x + 14, n.y - 2);
      if (sublabel) {
        ctx.fillStyle = 'rgba(38,35,32,0.55)';
        ctx.font = '11px "JetBrains Mono", monospace';
        ctx.fillText(sublabel, n.x + 14, n.y + 14);
      }
    }

    // Branches: highlight the paths leading to B (the evidence we conditioned on)
    branch(root, { x: l1x, y: yA },     `P(${A})`,         pA,        pA,        true);
    branch(root, { x: l1x, y: yNotA },  `P(¬${A})`,        pNotA,     pNotA,     false);
    branch({ x: l1x, y: yA },    ll1, `P(${B}|${A})`,    pBA,       j_AB,      true);
    branch({ x: l1x, y: yA },    ll2, `P(¬${B}|${A})`,   pNotBA,    j_AnotB,   false);
    branch({ x: l1x, y: yNotA }, ll3, `P(${B}|¬${A})`,   pBNotA,    j_notAB,   true);
    branch({ x: l1x, y: yNotA }, ll4, `P(¬${B}|¬${A})`,  pNotBNotA, j_notAnotB, false);

    // Nodes
    node(root, 'start', '', ACCENT);
    node({ x: l1x, y: yA }, A, '');
    node({ x: l1x, y: yNotA }, '¬' + A, '');
    node(ll1, B,        formatNum(j_AB));
    node(ll2, '¬' + B,  formatNum(j_AnotB));
    node(ll3, B,        formatNum(j_notAB));
    node(ll4, '¬' + B,  formatNum(j_notAnotB));

    // Section title
    ctx.fillStyle = INK_FADE;
    ctx.font = SERIF_LABEL_SM;
    ctx.textAlign = 'left';
    ctx.fillText('joint probabilities', l2x + 16, padY - 6);

    // Posterior summary at the bottom
    const posterior = j_AB / (j_AB + j_notAB + 1e-12);
    ctx.fillStyle = ACCENT;
    ctx.font = 'italic 14px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText(`P(${A} | ${B}) = ${(posterior * 100).toFixed(2)}%`,
      size.w / 2, size.h - 8);
  }

  // ───────── Square mode ─────────
  function drawSquare(ctx, size) {
    const sc = SCENARIOS[state.scenarioIdx];
    const A = sc.A.name, B = sc.B;
    const prior = state.prior, sens = state.sens, fpr = state.fpr;

    const pad = 36;
    const W = size.w - pad * 2;
    const H = size.h - pad * 2 - 30;

    const wA    = W * prior;
    const wNotA = W * (1 - prior);
    const hBA   = H * sens;
    const hBNotA = H * fpr;

    // Frame fill
    ctx.fillStyle = 'rgba(255,253,246,0.5)';
    ctx.fillRect(pad, pad, W, H);
    // A column (bigger background tint)
    ctx.fillStyle = 'rgba(38,35,32,0.10)';
    ctx.fillRect(pad, pad, wA, H);
    // ¬A column
    ctx.fillStyle = 'rgba(38,35,32,0.05)';
    ctx.fillRect(pad + wA, pad, wNotA, H);
    // B regions (oxblood)
    ctx.fillStyle = ACCENT;
    ctx.fillRect(pad, pad, wA, hBA);
    ctx.fillStyle = 'rgba(122,31,36,0.55)';
    ctx.fillRect(pad + wA, pad, wNotA, hBNotA);

    // Frame
    ctx.strokeStyle = 'rgba(38,35,32,0.50)';
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, pad, W, H);
    // Vertical divider
    ctx.strokeStyle = 'rgba(38,35,32,0.40)';
    ctx.beginPath();
    ctx.moveTo(pad + wA, pad); ctx.lineTo(pad + wA, pad + H);
    ctx.stroke();

    // Column labels
    ctx.fillStyle = 'rgba(38,35,32,0.75)';
    ctx.font = SERIF_LABEL_SM;
    ctx.textAlign = 'center';
    ctx.fillText(A,        pad + wA / 2,        pad + H + 18);
    ctx.fillText('¬' + A,  pad + wA + wNotA / 2, pad + H + 18);
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillStyle = INK_FADE;
    ctx.fillText(formatNum(prior),     pad + wA / 2,        pad + H + 32);
    ctx.fillText(formatNum(1 - prior), pad + wA + wNotA / 2, pad + H + 32);

    // Top label
    ctx.font = SERIF_LABEL_SM;
    ctx.fillStyle = ACCENT;
    ctx.textAlign = 'left';
    ctx.fillText(`${B} (orange area)`, pad, pad - 8);

    // Posterior at the bottom
    const j_AB = prior * sens;
    const j_notAB = (1 - prior) * fpr;
    const posterior = j_AB / (j_AB + j_notAB + 1e-12);
    ctx.fillStyle = ACCENT;
    ctx.font = 'italic 16px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText(`P(${A} | ${B}) = ${(posterior * 100).toFixed(2)}%`,
      size.w / 2, size.h - 8);
  }

  // ───────── Render ─────────
  function render() {
    const sc = SCENARIOS[state.scenarioIdx];

    Array.from(scenariosEl.children).forEach((btn, i) => {
      btn.classList.toggle('active', i === state.scenarioIdx);
    });
    Array.from(tabsEl.children).forEach((btn) => {
      btn.classList.toggle('on', btn.dataset.mode === state.mode);
    });

    renderKnobs();

    // Live posterior + readout
    const j_AB = state.prior * state.sens;
    const j_notAB = (1 - state.prior) * state.fpr;
    const pB = j_AB + j_notAB;
    const posterior = j_AB / (pB + 1e-12);

    readoutEl.innerHTML = `
      <div class="row"><span>P(${sc.B})</span><b>${formatNum(pB)}</b></div>
      <div class="row"><span>P(${sc.A.name}, ${sc.B})</span><b>${formatNum(j_AB)}</b></div>
      <div class="row"><span>P(¬${sc.A.name}, ${sc.B})</span><b>${formatNum(j_notAB)}</b></div>
      <div class="row" style="border-top:1px solid var(--rule); margin-top:6px; padding-top:8px">
        <span style="color:var(--accent); font-style:italic">P(${sc.A.name} | ${sc.B})</span>
        <b style="color:var(--accent)">${(posterior * 100).toFixed(2)}%</b>
      </div>`;

    blurbEl.textContent = sc.blurb;

    captionEl.innerHTML = state.mode === 'tree'
      ? `<span class="ml-cap-num">I.</span> The thicker the branch, the more likely that path. The <span class="ml-ink-orange">accent paths</span> end at "${sc.B}" — the evidence we observed. The posterior is the accent-A path divided by all accent paths.`
      : `<span class="ml-cap-num">II.</span> The whole square is the world. Width = prior; height = test rate. The <span class="ml-ink-orange">accent area</span> is "${sc.B}" everywhere it occurs. The posterior is the dark-accent slice divided by all accent.`;

    canvasFrameEl.style.aspectRatio = state.mode === 'tree' ? '1.15 / 1' : '1 / 1';
    canvasCtl.redraw();
  }

  render();
})();
