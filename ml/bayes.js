/* Vol. VII — Bayes' Theorem */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK, INK_FADE, RULE_FAINT,
          ACCENT_SOFT, ACCENT_FAINT, ACCENT_FILL,
          SERIF_LABEL_SM, MONO_LABEL } = V;

  const SCENARIOS = [
    {
      key: 'medical',
      name: 'Medical test',
      // The "subject" of the population (matches A/notA)
      population: 'people',
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
      population: 'emails',
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
      population: 'days',
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
      population: 'collisions',
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
    // Default to Population view: the base-rate-fallacy "gasp" lives there
    // (1% prior · 95% sensitivity → ~16% posterior visualised as bright vs faint dots).
    // Tree and Square modes are the natural follow-ups, not the entry point.
    mode: 'population',
    annotation: null,        // { text, until }
    annotationTimer: null,
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
    b.style.cursor = 'pointer';
    b.addEventListener('click', () => {
      state.scenarioIdx = i;
      state.prior = s.A.prior;
      state.sens = s.pBgivenA;
      state.fpr = s.pBgivenNotA;
      if (state.annotationTimer) { clearTimeout(state.annotationTimer); state.annotationTimer = null; }
      state.annotation = null;
      render();
    });
    scenariosEl.appendChild(b);
  });

  // Mode tabs
  Array.from(tabsEl.children).forEach((btn) => {
    btn.style.cursor = 'pointer';
    btn.addEventListener('click', () => {
      state.mode = btn.dataset.mode;
      render();
    });
  });

  // Storytelling annotation — a brief plain-English description of the last knob change,
  // overlaid on the canvas and faded out after a few seconds.
  function setAnnotation(text) {
    if (state.annotationTimer) clearTimeout(state.annotationTimer);
    state.annotation = { text, until: performance.now() + 2800 };
    state.annotationTimer = setTimeout(() => {
      state.annotation = null;
      state.annotationTimer = null;
      canvasCtl.redraw();
    }, 2900);
  }

  function describeKnobChange(knobKey, oldV, newV) {
    const sc = SCENARIOS[state.scenarioIdx];
    const pct = (v) => `${(v * 100).toFixed(1)}%`;
    // Compute posterior at old and new state to express the shift.
    const post = (prior, sens, fpr) => {
      const j = prior * sens, k = (1 - prior) * fpr;
      return j / (j + k + 1e-12);
    };
    const sBefore = { prior: state.prior, sens: state.sens, fpr: state.fpr };
    const sAfter  = { ...sBefore, [knobKey]: newV };
    const _b = { ...sAfter, [knobKey]: oldV };
    const before = post(_b.prior, _b.sens, _b.fpr);
    const after  = post(sAfter.prior, sAfter.sens, sAfter.fpr);
    const arrow = after > before ? '↑' : (after < before ? '↓' : '·');
    const head = (() => {
      if (knobKey === 'prior') return `Prior P(${sc.A.name}) = ${pct(newV)}`;
      if (knobKey === 'sens')  return `Sensitivity P(${sc.B}|${sc.A.name}) = ${pct(newV)}`;
      return `False-positive rate P(${sc.B}|¬${sc.A.name}) = ${pct(newV)}`;
    })();
    return `${head} → posterior ${arrow} ${pct(after)}`;
  }

  // Knobs — labels are now plain-language, with the math-name in a sub-row.
  function renderKnobs() {
    const sc = SCENARIOS[state.scenarioIdx];
    const knobs = [
      {
        key: 'prior',
        plain: `How common is "${sc.A.name}"?`,
        math:  `prior · P(${sc.A.name})`,
        min: 0.001, max: 0.999, step: 0.001,
      },
      {
        key: 'sens',
        plain: `True-positive rate (${sc.B} when ${sc.A.name})`,
        math:  `sensitivity · P(${sc.B} | ${sc.A.name})`,
        min: 0, max: 1, step: 0.01,
      },
      {
        key: 'fpr',
        plain: `False-positive rate (${sc.B} when ¬${sc.A.name})`,
        math:  `P(${sc.B} | ¬${sc.A.name})`,
        min: 0, max: 1, step: 0.01,
      },
    ];
    knobsEl.innerHTML = '';
    knobs.forEach((k) => {
      const row = document.createElement('div');
      row.innerHTML = `
        <div class="ml-param-head">
          <span class="label">${k.plain}</span>
          <b>${formatNum(state[k.key])}</b>
        </div>
        <div style="font-family:var(--mono); font-size:10px; letter-spacing:0.04em; color:var(--ml-ink-fade); margin:-4px 0 6px">${k.math}</div>
        <input type="range" min="${k.min}" max="${k.max}" step="${k.step}" value="${state[k.key]}" />
      `;
      const slider = row.querySelector('input');
      slider.style.cursor = 'ew-resize';
      slider.title = `drag to change ${k.plain}`;
      slider.addEventListener('input', (e) => {
        const oldV = state[k.key];
        const newV = +e.target.value;
        if (oldV !== newV) {
          state[k.key] = newV;
          setAnnotation(describeKnobChange(k.key, oldV, newV));
        }
        render();
      });
      knobsEl.appendChild(row);
    });
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    if (state.mode === 'tree')           drawTree(ctx, size);
    else if (state.mode === 'population') drawPopulation(ctx, size);
    else                                  drawSquare(ctx, size);
    drawAnnotation(ctx, size);
  }

  function drawAnnotation(ctx, size) {
    const a = state.annotation;
    if (!a) return;
    const now = performance.now();
    const remaining = a.until - now;
    if (remaining <= 0) return;
    const alpha = remaining > 700 ? 1 : Math.max(0, remaining / 700);

    const txt = a.text;
    ctx.save();
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    const w = ctx.measureText(txt).width;
    const padX = 12, padY = 7;
    const boxW = w + padX * 2;
    const boxH = 16 + padY;
    const x = (size.w - boxW) / 2;
    const y = 8;
    ctx.globalAlpha = alpha;
    ctx.fillStyle = 'rgba(255,253,246,0.96)';
    ctx.fillRect(x, y, boxW, boxH);
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, boxW, boxH);
    // accent left edge
    ctx.fillStyle = ACCENT;
    ctx.fillRect(x, y, 3, boxH);
    ctx.fillStyle = INK;
    ctx.fillText(txt, size.w / 2, y + boxH / 2 + 4);
    ctx.restore();

    if (remaining < 800 && !state._fadeRaf) {
      state._fadeRaf = requestAnimationFrame(() => {
        state._fadeRaf = null;
        canvasCtl.redraw();
      });
    }
  }

  // ───────── Mode I: Tree mode ─────────
  // Probability tree with branch widths proportional to mass, plus a *giant fraction overlay*
  // pulling the two evidence-leaves into a single visible posterior ratio.
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

    // Layout: left half is the tree, right half is the giant fraction overlay.
    const treeW = size.w * 0.55;
    const padX = 28, padY = 20;
    const root = { x: padX + 6, y: size.h / 2 };
    const l1x  = padX + treeW * 0.34;
    const l2x  = padX + treeW * 0.84;
    const yA = size.h * 0.26, yNotA = size.h * 0.74;
    const ll1 = { x: l2x, y: yA - size.h * 0.13 };       // A & B   ← numerator
    const ll2 = { x: l2x, y: yA + size.h * 0.13 };       // A & ¬B
    const ll3 = { x: l2x, y: yNotA - size.h * 0.13 };    // ¬A & B  ← also in denominator
    const ll4 = { x: l2x, y: yNotA + size.h * 0.13 };    // ¬A & ¬B

    function branch(from, to, label, prob, weight, hot) {
      ctx.strokeStyle = hot ? ACCENT : 'rgba(38,35,32,0.40)';
      ctx.lineWidth = hot ? Math.max(2, weight * 10) : Math.max(1, weight * 8);
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
      const mx = (from.x + to.x) / 2, my = (from.y + to.y) / 2;
      ctx.fillStyle = hot ? ACCENT : 'rgba(38,35,32,0.65)';
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(label, mx, my - 6);
      ctx.font = SERIF_LABEL_SM;
      ctx.fillText(formatNum(prob), mx, my + 12);
    }

    function node(n, label, fill) {
      ctx.fillStyle = fill || '#fffdf6';
      ctx.strokeStyle = 'rgba(38,35,32,0.50)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(n.x, n.y, 7, 0, Math.PI * 2);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = INK;
      ctx.font = SERIF_LABEL_SM;
      ctx.textAlign = 'left';
      ctx.fillText(label, n.x + 12, n.y - 2);
    }

    // Branches: highlight the paths leading to B (the evidence we conditioned on)
    branch(root, { x: l1x, y: yA },     `P(${A})`,         pA,        pA,        true);
    branch(root, { x: l1x, y: yNotA },  `P(¬${A})`,        pNotA,     pNotA,     false);
    branch({ x: l1x, y: yA },    ll1, `P(${B}|${A})`,    pBA,       j_AB,      true);
    branch({ x: l1x, y: yA },    ll2, `P(¬${B}|${A})`,   pNotBA,    j_AnotB,   false);
    branch({ x: l1x, y: yNotA }, ll3, `P(${B}|¬${A})`,   pBNotA,    j_notAB,   true);
    branch({ x: l1x, y: yNotA }, ll4, `P(¬${B}|¬${A})`,  pNotBNotA, j_notAnotB, false);

    // Nodes
    node(root, 'start', ACCENT);
    node({ x: l1x, y: yA }, A);
    node({ x: l1x, y: yNotA }, '¬' + A);
    // Leaf "boxes" — boxed leaves so the next step (pull into fraction) reads as moving objects.
    drawLeafBox(ctx, ll1, B,        j_AB,       /* highlight num */ true,  /* highlight den */ true);
    drawLeafBox(ctx, ll2, '¬' + B,  j_AnotB,    false, false);
    drawLeafBox(ctx, ll3, B,        j_notAB,    false, true);
    drawLeafBox(ctx, ll4, '¬' + B,  j_notAnotB, false, false);

    // Section title
    ctx.fillStyle = INK_FADE;
    ctx.font = SERIF_LABEL_SM;
    ctx.textAlign = 'left';
    ctx.fillText('joint probabilities (leaves)', padX, padY - 4);

    // ───── Giant fraction overlay (right half) ─────
    const fx = size.w * 0.74;
    const fy = size.h / 2;
    const fw = size.w - fx - 24;

    // Title above the fraction
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('POSTERIOR  =', fx + fw / 2, padY + 4);

    // Numerator pill: P(A∩B)
    const pillH = 30;
    const pillW = Math.min(fw, 180);
    const numY = fy - pillH - 10;
    const denY = fy + 10;
    const pillX = fx + (fw - pillW) / 2;

    // Numerator
    drawPill(ctx, pillX, numY, pillW, pillH,
      `P(${A} ∩ ${B})`, j_AB, ACCENT, '#fffdf6');
    // Numerator pull-line from leaf ll1 → numerator
    drawCurvePull(ctx, ll1.x + 8, ll1.y, pillX, numY + pillH / 2, ACCENT, 0.55);

    // Fraction bar
    ctx.strokeStyle = INK;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(pillX - 6, fy);
    ctx.lineTo(pillX + pillW + 6, fy);
    ctx.stroke();

    // Denominator: P(A∩B) + P(¬A∩B)
    const subPillW = (pillW - 16) / 2;
    drawPill(ctx, pillX, denY, subPillW, pillH,
      `P(${A} ∩ ${B})`, j_AB, ACCENT, '#fffdf6');
    ctx.fillStyle = INK;
    ctx.font = 'italic 18px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('+', pillX + subPillW + 8, denY + pillH / 2 + 6);
    drawPill(ctx, pillX + subPillW + 16, denY, subPillW, pillH,
      `P(¬${A} ∩ ${B})`, j_notAB, 'rgba(122,31,36,0.55)', '#fffdf6');

    // Denominator pull-lines from ll1 + ll3 → denominator
    drawCurvePull(ctx, ll1.x + 8, ll1.y, pillX, denY + pillH / 2, ACCENT, 0.30);
    drawCurvePull(ctx, ll3.x + 8, ll3.y, pillX + subPillW + 16, denY + pillH / 2, 'rgba(122,31,36,0.55)', 0.30);

    // Equals + posterior result
    const posterior = j_AB / (j_AB + j_notAB + 1e-12);
    const resY = denY + pillH + 30;
    ctx.fillStyle = INK;
    ctx.font = 'italic 16px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('=', fx + fw / 2, resY);
    ctx.fillStyle = ACCENT;
    ctx.font = 'italic 28px "Source Serif 4", Georgia, serif';
    ctx.fillText(`${(posterior * 100).toFixed(2)}%`, fx + fw / 2, resY + 32);
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillText(`P(${A} | ${B})`, fx + fw / 2, resY + 50);
  }

  function drawLeafBox(ctx, n, eventLabel, joint, isNum, isDen) {
    const w = 88, h = 30;
    const x = n.x - 2, y = n.y - h / 2;
    // Numerator-and-denominator leaf gets full accent fill; denominator-only gets soft.
    let fill = '#fffdf6', stroke = 'rgba(38,35,32,0.45)';
    if (isNum) { fill = ACCENT_SOFT; stroke = ACCENT; }
    else if (isDen) { fill = 'rgba(122,31,36,0.06)'; stroke = ACCENT_FAINT; }
    ctx.fillStyle = fill;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1;
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = INK;
    ctx.font = SERIF_LABEL_SM;
    ctx.textAlign = 'left';
    ctx.fillText(eventLabel, x + 6, y + 12);
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillStyle = isNum || isDen ? ACCENT : INK_FADE;
    ctx.fillText(formatNum(joint), x + 6, y + 24);
  }

  function drawPill(ctx, x, y, w, h, label, value, stroke, fill) {
    const r = h / 2;
    ctx.fillStyle = fill;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    if (typeof ctx.roundRect === 'function') {
      ctx.roundRect(x, y, w, h, r);
    } else {
      ctx.moveTo(x + r, y);
      ctx.lineTo(x + w - r, y);
      ctx.arcTo(x + w, y, x + w, y + r, r);
      ctx.lineTo(x + w, y + h - r);
      ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
      ctx.lineTo(x + r, y + h);
      ctx.arcTo(x, y + h, x, y + h - r, r);
      ctx.lineTo(x, y + r);
      ctx.arcTo(x, y, x + r, y, r);
    }
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = stroke;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText(label, x + 10, y + h / 2 - 1);
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.fillText(formatNum(value), x + w - 10, y + h / 2 + 11);
  }

  function drawCurvePull(ctx, x0, y0, x1, y1, color, alpha) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    const cx = (x0 + x1) / 2;
    ctx.bezierCurveTo(cx, y0, cx, y1, x1, y1);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }

  // ───────── Mode II: Population view ─────────
  // 100×100 dot grid = 10,000 dots, scaled to whatever total population we want.
  // Dots represent units; oxblood = A; ink-pale = ¬A. Then we ring the test-positives.
  function drawPopulation(ctx, size) {
    const sc = SCENARIOS[state.scenarioIdx];
    const A = sc.A.name, notA = sc.notA, B = sc.B;
    const prior = state.prior, sens = state.sens, fpr = state.fpr;

    // Grid sizing — fit a 100×100 grid in the canvas.
    const ROWS = 100, COLS = 100;
    const padTop = 36, padBottom = 90, padX = 30;
    const availW = size.w - padX * 2;
    const availH = size.h - padTop - padBottom;
    const cell = Math.max(2, Math.floor(Math.min(availW / COLS, availH / ROWS)));
    const gridW = cell * COLS, gridH = cell * ROWS;
    const x0 = (size.w - gridW) / 2;
    const y0 = padTop;

    const TOTAL = ROWS * COLS;             // 10,000 dots in the grid
    const nA    = Math.round(prior * TOTAL);
    const nNotA = TOTAL - nA;
    const nTPdots = Math.round(sens * nA);              // true positives
    const nFPdots = Math.round(fpr * nNotA);            // false positives

    // Categorise each dot index.
    // Order: A then ¬A — left-to-right, top-to-bottom raster.
    // Within A: first nTPdots are TP (B), rest are A & ¬B.
    // Within ¬A: first nFPdots are FP (B), rest are ¬A & ¬B.
    function classify(i) {
      if (i < nA) return i < nTPdots ? 'TP' : 'AN';        // A & B / A & ¬B
      const j = i - nA;
      return j < nFPdots ? 'FP' : 'NN';                    // ¬A & B / ¬A & ¬B
    }

    // Colors:
    //   TP — bright oxblood (filled)
    //   AN — soft oxblood (A but no positive test)
    //   FP — faint oxblood (the false positives — the *base-rate fallacy* dots)
    //   NN — pale ink (healthy / negative)
    const COLOR = {
      TP: ACCENT,
      AN: 'rgba(122,31,36,0.30)',
      FP: 'rgba(122,31,36,0.55)',
      NN: 'rgba(38,35,32,0.10)',
    };

    // Draw all dots
    for (let i = 0; i < TOTAL; i++) {
      const r = Math.floor(i / COLS);
      const c = i % COLS;
      const cls = classify(i);
      ctx.fillStyle = COLOR[cls];
      ctx.fillRect(x0 + c * cell + 0.5, y0 + r * cell + 0.5, cell - 1, cell - 1);
    }

    // Headline numbers above the grid
    const totalPop = 100000;
    const peoA = Math.round(prior * totalPop);
    const peoTP = Math.round(prior * sens * totalPop);
    const peoFP = Math.round((1 - prior) * fpr * totalPop);
    const peoPos = peoTP + peoFP;
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`POPULATION ${formatBig(totalPop)} ${sc.population.toUpperCase()}`, x0, y0 - 18);
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.fillStyle = INK;
    ctx.fillText(`each square = ${(totalPop / TOTAL).toFixed(0)} ${sc.population}`, x0 + 230, y0 - 18);

    // Frame around the grid
    ctx.strokeStyle = 'rgba(38,35,32,0.32)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x0 - 0.5, y0 - 0.5, gridW + 1, gridH + 1);

    // ── Bayes formula card, anchored to the right of the grid ──
    // Math next to the picture, not buried in marginalia. The card uses the
    // same TP / (TP + FP) decomposition the dots make visible.
    const cardX = x0 + gridW + 18;
    if (cardX + 220 < size.w) {
      const cardW = Math.min(220, size.w - cardX - 8);
      const cardY = y0 + 6;
      ctx.fillStyle = 'rgba(255,253,246,0.94)';
      ctx.fillRect(cardX, cardY, cardW, 132);
      ctx.strokeStyle = 'rgba(122,31,36,0.35)';
      ctx.lineWidth = 1;
      ctx.strokeRect(cardX, cardY, cardW, 132);
      ctx.fillStyle = ACCENT;
      ctx.fillRect(cardX, cardY, 3, 132);

      ctx.fillStyle = INK_FADE;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      ctx.fillText('BAYES, ANCHORED', cardX + 12, cardY + 16);

      ctx.fillStyle = INK;
      ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
      ctx.fillText(`P(${A} | ${B})  =`, cardX + 12, cardY + 38);

      // Numerator: bright dots
      ctx.fillStyle = ACCENT;
      ctx.fillRect(cardX + 12, cardY + 50, 9, 9);
      ctx.fillStyle = INK;
      ctx.fillText('bright dots',  cardX + 26, cardY + 59);
      ctx.fillStyle = INK_FADE;
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.fillText(formatBig(peoTP), cardX + cardW - 12 - ctx.measureText(formatBig(peoTP)).width, cardY + 59);

      // Fraction bar
      ctx.strokeStyle = INK;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(cardX + 12, cardY + 68);
      ctx.lineTo(cardX + cardW - 12, cardY + 68);
      ctx.stroke();

      // Denominator: bright + faint
      ctx.fillStyle = ACCENT;
      ctx.fillRect(cardX + 12, cardY + 76, 9, 9);
      ctx.fillStyle = 'rgba(122,31,36,0.55)';
      ctx.fillRect(cardX + 24, cardY + 76, 9, 9);
      ctx.fillStyle = INK;
      ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
      ctx.fillText('all positives',  cardX + 38, cardY + 85);
      ctx.fillStyle = INK_FADE;
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.fillText(formatBig(peoPos), cardX + cardW - 12 - ctx.measureText(formatBig(peoPos)).width, cardY + 85);

      // Result
      ctx.fillStyle = ACCENT;
      ctx.font = 'italic 22px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      const postPct = ((peoTP) / Math.max(1, peoTP + peoFP) * 100).toFixed(1) + '%';
      ctx.fillText(`= ${postPct}`, cardX + cardW / 2, cardY + 118);
    }

    // ── Legend below the grid ──
    // Scenario-aware labels — use A / ¬A / B from the active scenario rather
    // than hardcoded "disease/healthy" copy that only fits the medical case.
    const legY = y0 + gridH + 22;
    drawLegendChip(ctx, x0,            legY,      `TRUE POS · ${A} & ${B}`,         COLOR.TP, formatBig(peoTP));
    drawLegendChip(ctx, x0 + 260,      legY,      `FALSE POS · ${notA} & ${B}`,     COLOR.FP, formatBig(peoFP));
    drawLegendChip(ctx, x0 + 520,      legY,      `${A} & ¬${B}`,                   COLOR.AN, formatBig(peoA - peoTP));
    drawLegendChip(ctx, x0,            legY + 22, `${notA} & ¬${B}`,                COLOR.NN, formatBig(totalPop - peoA - peoFP));

    // ── Posterior reading: TP / (TP + FP) framed as a fraction ──
    const posterior = (peoTP) / Math.max(1, (peoTP + peoFP));
    const fy = legY + 50;
    ctx.fillStyle = INK;
    ctx.font = 'italic 14px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText(`Of ${formatBig(peoPos)} ${sc.population} where "${B}", ${formatBig(peoTP)} are actually "${A}".`, x0, fy);

    ctx.fillStyle = ACCENT;
    ctx.font = 'italic 14px "Source Serif 4", Georgia, serif';
    const r2 = `P(${A} | ${B}) = ${formatBig(peoTP)} / ${formatBig(peoPos)} = ${(posterior * 100).toFixed(2)}%`;
    ctx.fillText(r2, x0, fy + 18);

    // Editorial "wait, really?" line — fires when the posterior is dramatically
    // lower than the test's sensitivity (the base-rate-fallacy gasp).
    // Phrased generically so it reads correctly across all scenarios, not just medical.
    // Split across two lines so it fits the canvas at every reasonable width.
    if (posterior < 0.5 && sens > 0.8) {
      ctx.fillStyle = INK_FADE;
      ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'left';
      ctx.fillText(
        `Wait, really? P(${B} | ${A}) = ${(sens * 100).toFixed(0)}%, yet P(${A} | ${B}) = ${(posterior * 100).toFixed(0)}%?`,
        x0, fy + 38
      );
      ctx.fillText(
        `Yes — "${notA}" vastly outnumbers "${A}", so most "${B}" cases are ${notA.toLowerCase()}.`,
        x0, fy + 56
      );
    }
  }

  function drawLegendChip(ctx, x, y, label, color, count) {
    ctx.fillStyle = color;
    ctx.fillRect(x, y - 9, 12, 12);
    ctx.strokeStyle = 'rgba(38,35,32,0.25)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y - 8.5, 11, 11);
    ctx.fillStyle = INK;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText(label, x + 18, y);
    ctx.fillStyle = INK_FADE;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillText(count, x + 18, y + 14);
  }

  function formatBig(n) {
    if (n >= 1000) return n.toLocaleString('en-US');
    return String(n);
  }

  // ───────── Mode III: Square mode (unchanged in spirit, slight polish) ─────────
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

    ctx.fillStyle = 'rgba(255,253,246,0.5)';
    ctx.fillRect(pad, pad, W, H);
    ctx.fillStyle = 'rgba(38,35,32,0.10)';
    ctx.fillRect(pad, pad, wA, H);
    ctx.fillStyle = 'rgba(38,35,32,0.05)';
    ctx.fillRect(pad + wA, pad, wNotA, H);
    ctx.fillStyle = ACCENT;
    ctx.fillRect(pad, pad, wA, hBA);
    ctx.fillStyle = 'rgba(122,31,36,0.55)';
    ctx.fillRect(pad + wA, pad, wNotA, hBNotA);

    ctx.strokeStyle = 'rgba(38,35,32,0.50)';
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, pad, W, H);
    ctx.strokeStyle = 'rgba(38,35,32,0.40)';
    ctx.beginPath();
    ctx.moveTo(pad + wA, pad); ctx.lineTo(pad + wA, pad + H);
    ctx.stroke();

    ctx.fillStyle = 'rgba(38,35,32,0.75)';
    ctx.font = SERIF_LABEL_SM;
    ctx.textAlign = 'center';
    ctx.fillText(A,        pad + wA / 2,        pad + H + 18);
    ctx.fillText('¬' + A,  pad + wA + wNotA / 2, pad + H + 18);
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillStyle = INK_FADE;
    ctx.fillText(formatNum(prior),     pad + wA / 2,        pad + H + 32);
    ctx.fillText(formatNum(1 - prior), pad + wA + wNotA / 2, pad + H + 32);

    ctx.font = SERIF_LABEL_SM;
    ctx.fillStyle = ACCENT;
    ctx.textAlign = 'left';
    ctx.fillText(`${B} (orange area)`, pad, pad - 8);

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

    // Color-coded equation that anchors directly to the visualization above.
    // P(A∩B) — accent strong; P(B) — accent medium; result — accent strong.
    const colorEq = `
      <div style="margin-top:10px; padding-top:10px; border-top:1px solid var(--rule); font-family:var(--serif); font-style:italic; font-size:14px; line-height:1.6">
        <span style="color:var(--accent)">P(${sc.A.name} | ${sc.B})</span>
        <span style="color:var(--ml-ink-fade)"> = </span>
        <span style="color:var(--accent)">P(${sc.A.name} ∩ ${sc.B})</span>
        <span style="color:var(--ml-ink-fade)"> / </span>
        <span style="color:rgba(122,31,36,0.65)">P(${sc.B})</span>
      </div>`;

    readoutEl.innerHTML = `
      <div class="row"><span style="color:rgba(122,31,36,0.65)">P(${sc.B})</span><b>${formatNum(pB)}</b></div>
      <div class="row"><span style="color:var(--accent)">P(${sc.A.name}, ${sc.B})</span><b>${formatNum(j_AB)}</b></div>
      <div class="row"><span>P(¬${sc.A.name}, ${sc.B})</span><b>${formatNum(j_notAB)}</b></div>
      <div class="row" style="border-top:1px solid var(--rule); margin-top:6px; padding-top:8px">
        <span style="color:var(--accent); font-style:italic">P(${sc.A.name} | ${sc.B})</span>
        <b style="color:var(--accent)">${(posterior * 100).toFixed(2)}%</b>
      </div>
      ${colorEq}`;

    blurbEl.textContent = sc.blurb;

    let cap;
    if (state.mode === 'tree') {
      cap = `<span class="ml-cap-num">I.</span> The thicker the branch, the more likely that path. The two highlighted leaves get pulled into a literal fraction on the right — the posterior is <em>that ratio</em>, no more.`;
    } else if (state.mode === 'population') {
      cap = `<span class="ml-cap-num">II.</span> 100,000 ${sc.population} as a 100×100 grid. <span class="ml-ink-orange">Bright oxblood</span> = true positives; <span style="color:rgba(122,31,36,0.55); font-style:italic">faint oxblood</span> = false positives. The posterior is the literal ratio of bright to all-positive squares — the visceral base-rate-fallacy view.`;
    } else {
      cap = `<span class="ml-cap-num">III.</span> The whole square is the world. Width = prior; height = test rate. The accent area is "${sc.B}" everywhere it occurs. The posterior is the bright-accent slice divided by all accent.`;
    }
    captionEl.innerHTML = cap;

    if (state.mode === 'tree') canvasFrameEl.style.aspectRatio = '1.5 / 1';
    else if (state.mode === 'population') canvasFrameEl.style.aspectRatio = '1.05 / 1';
    else canvasFrameEl.style.aspectRatio = '1 / 1';
    canvasCtl.redraw();
  }

  render();
})();
