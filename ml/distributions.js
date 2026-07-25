/* Vol. VI — Probability Distributions */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK, INK_FADE, RULE_FAINT, RULE_AXIS,
          ACCENT_SOFT, ACCENT_FAINT, ACCENT_FILL,
          SERIF_LABEL_SM, MONO_LABEL } = V;

  // ───────── Math: pdfs / pmfs / samplers ─────────
  const gaussianPdf = (x, mu, sigma) => {
    const z = (x - mu) / sigma;
    return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
  };
  const bernoulliPmf = (k, p) => (k === 1 ? p : (k === 0 ? 1 - p : 0));
  const poissonPmf = (k, lam) => {
    if (k < 0) return 0;
    let logp = -lam + k * Math.log(lam);
    for (let i = 2; i <= k; i++) logp -= Math.log(i);
    return Math.exp(logp);
  };
  const binomialPmf = (k, n, p) => {
    if (k < 0 || k > n) return 0;
    const logp = k * Math.log(p + 1e-12) + (n - k) * Math.log(1 - p + 1e-12);
    let logC = 0;
    for (let i = 1; i <= k; i++) logC += Math.log(n - i + 1) - Math.log(i);
    return Math.exp(logC + logp);
  };
  const expoPdf = (x, lam) => (x < 0 ? 0 : lam * Math.exp(-lam * x));
  const uniformPdf = (x, a, b) => (x >= a && x <= b ? 1 / (b - a) : 0);

  function sampleGaussian(mu, sigma) {
    let u = 0, w = 0;
    while (u === 0) u = Math.random();
    while (w === 0) w = Math.random();
    return mu + sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * w);
  }
  const sampleBernoulli = (p) => (Math.random() < p ? 1 : 0);
  function samplePoisson(lam) {
    const L = Math.exp(-lam);
    let k = 0, p = 1;
    do { k++; p *= Math.random(); } while (p > L);
    return k - 1;
  }
  function sampleBinomial(n, p) {
    let k = 0;
    for (let i = 0; i < n; i++) if (Math.random() < p) k++;
    return k;
  }
  const sampleExpo = (lam) => -Math.log(1 - Math.random()) / lam;
  const sampleUniform = (a, b) => a + Math.random() * (b - a);

  // ───────── Distribution definitions ─────────
  const DISTS = {
    gaussian: {
      name: 'Gaussian',
      type: 'continuous',
      params: [
        { key: 'mu',    label: 'μ (mean)', min: -3, max: 3, step: 0.1,  default: 0 },
        { key: 'sigma', label: 'σ (std)',  min: 0.2, max: 2.5, step: 0.05, default: 1 },
      ],
      pdf: (x, p) => gaussianPdf(x, p.mu, p.sigma),
      sample: (p) => sampleGaussian(p.mu, p.sigma),
      range: () => [-5, 5],
      stats: (p) => ({ Mean: p.mu, Variance: p.sigma * p.sigma, Mode: p.mu }),
      blurb: "The bell curve. Sums of many small effects converge to it (Central Limit Theorem). The default model when you don't know better.",
      eq: 'f(x) = (1/σ√2π) · e<sup>−(x−μ)²/2σ²</sup>',
      eqInline: 'f(x) = (1/σ√2π) · exp(−(x−μ)²/2σ²)',
      eqGloss: 'Read it: density at <em>x</em> equals a height-scaling constant (<em>1/σ√2π</em>) times <em>e</em> raised to a negative-squared-distance from the mean. The exponent <em>−(x−μ)²/2σ²</em> is what makes the bell — far from <em>μ</em>, the value plummets.',
      note: 'Two parameters tell you everything: where the bell sits (<em>μ</em>) and how wide it is (<em>σ</em>). Almost every "errors are noisy" assumption in statistics secretly assumes a Gaussian.',
    },
    bernoulli: {
      name: 'Bernoulli',
      type: 'discrete',
      params: [{ key: 'p', label: 'p', min: 0, max: 1, step: 0.01, default: 0.3 }],
      pmf: (k, p) => bernoulliPmf(k, p.p),
      sample: (p) => sampleBernoulli(p.p),
      support: () => [0, 1],
      stats: (p) => ({ Mean: p.p, Variance: p.p * (1 - p.p) }),
      blurb: "A single coin flip. p chance of 1, (1−p) chance of 0. The atom of probability — every binary outcome is a Bernoulli.",
      eq: 'P(X=1) = p, &nbsp;&nbsp; P(X=0) = 1−p',
      eqInline: 'P(X=1) = p,   P(X=0) = 1−p',
      eqGloss: 'Read it: <em>P(X=1)</em> is "the probability that the random variable <em>X</em> equals 1." Two outcomes, two probabilities, summing to 1.',
      note: 'The single most important distribution in classification: every class label probability is a Bernoulli or its multinomial sibling.',
    },
    binomial: {
      name: 'Binomial',
      type: 'discrete',
      params: [
        { key: 'n', label: 'n (trials)', min: 1, max: 40, step: 1, default: 12 },
        { key: 'p', label: 'p',          min: 0, max: 1,  step: 0.01, default: 0.5 },
      ],
      pmf: (k, p) => binomialPmf(k, p.n, p.p),
      sample: (p) => sampleBinomial(p.n, p.p),
      support: (p) => [0, p.n],
      stats: (p) => ({ Mean: p.n * p.p, Variance: p.n * p.p * (1 - p.p) }),
      blurb: "Count successes in n independent Bernoulli trials. As n grows, the shape becomes Gaussian.",
      eq: 'P(k) = C(n,k) p<sup>k</sup>(1−p)<sup>n−k</sup>',
      eqInline: 'P(k) = C(n,k) · p^k · (1−p)^(n−k)',
      eqGloss: 'Read it: chance of exactly <em>k</em> successes in <em>n</em> trials. <em>C(n,k)</em> ("n choose k") counts the orderings; <em>p<sup>k</sup>(1−p)<sup>n−k</sup></em> is the probability of any one such ordering.',
      note: 'Sum of n independent Bernoulli trials. As n grows, the histogram smooths into a Gaussian (de Moivre–Laplace).',
    },
    poisson: {
      name: 'Poisson',
      type: 'discrete',
      params: [{ key: 'lam', label: 'λ (rate)', min: 0.1, max: 20, step: 0.1, default: 4 }],
      pmf: (k, p) => poissonPmf(k, p.lam),
      sample: (p) => samplePoisson(p.lam),
      support: (p) => [0, Math.max(20, Math.ceil(p.lam * 3))],
      stats: (p) => ({ Mean: p.lam, Variance: p.lam }),
      blurb: "Count of rare events in a fixed window — emails per hour, photons per second. Mean equals variance.",
      eq: 'P(k) = λ<sup>k</sup> e<sup>−λ</sup> / k!',
      eqInline: 'P(k) = λ^k · e^(−λ) / k!',
      eqGloss: 'Read it: chance of <em>exactly k</em> events when the average rate is <em>λ</em> ("lambda"). <em>k!</em> is <em>k</em>-factorial — divides out repeated orderings. <em>e<sup>−λ</sup></em> is the chance of zero events.',
      note: 'Models counts of independent rare events in a fixed interval. Mean equals variance — a defining peculiarity.',
    },
    exponential: {
      name: 'Exponential',
      type: 'continuous',
      params: [{ key: 'lam', label: 'λ (rate)', min: 0.1, max: 3, step: 0.05, default: 1 }],
      pdf: (x, p) => expoPdf(x, p.lam),
      sample: (p) => sampleExpo(p.lam),
      range: (p) => [0, 8 / p.lam],
      stats: (p) => ({ Mean: 1 / p.lam, Variance: 1 / (p.lam * p.lam) }),
      blurb: "Waiting time between Poisson events. Memoryless: how long you've waited tells you nothing about how long is left.",
      eq: 'f(x) = λ e<sup>−λx</sup>',
      eqInline: 'f(x) = λ · e^(−λx)',
      eqGloss: 'Read it: density at wait-time <em>x</em>. The <em>e<sup>−λx</sup></em> term decays — long waits get exponentially rarer at rate <em>λ</em>.',
      note: 'The <em>memoryless</em> distribution. Wait time between Poisson arrivals.',
    },
    uniform: {
      name: 'Uniform',
      type: 'continuous',
      params: [
        { key: 'a', label: 'a', min: -3, max: 0, step: 0.1, default: -1 },
        { key: 'b', label: 'b', min: 0,  max: 3, step: 0.1, default: 1 },
      ],
      pdf: (x, p) => uniformPdf(x, p.a, p.b),
      sample: (p) => sampleUniform(p.a, p.b),
      range: (p) => [p.a - 1, p.b + 1],
      stats: (p) => ({ Mean: (p.a + p.b) / 2, Variance: (p.b - p.a) * (p.b - p.a) / 12 }),
      blurb: "Every value in [a, b] equally likely. The 'no preference' prior.",
      eq: 'f(x) = 1/(b−a) on [a, b]',
      eqInline: 'f(x) = 1/(b−a)  on [a,b]',
      eqGloss: 'Read it: a flat slab of height <em>1/(b−a)</em> from <em>a</em> to <em>b</em>, zero elsewhere. The height is whatever it must be so the rectangle\'s area is 1.',
      note: 'Every weight initialization in a neural network starts here.',
    },
  };

  const DIST_KEYS = Object.keys(DISTS);

  // ───────── State ─────────
  const state = {
    distKey: 'gaussian',
    params: {},
    samples: [],
    runningMean: [],   // running sample-mean trace; one entry per accepted sample
    pendingQueue: [],  // queued samples being dropped one-frame-at-a-time
    animId: null,
    // Short on-canvas message describing what the last slider change did.
    // Cleared after a few seconds so it doesn't hang around.
    annotation: null,        // { text: string, until: ms-timestamp }
    annotationTimer: null,
  };
  resetParams();

  function resetParams() {
    const p = {};
    DISTS[state.distKey].params.forEach((prm) => { p[prm.key] = prm.default; });
    state.params = p;
  }

  // ───────── DOM ─────────
  const distListEl = document.getElementById('dist-list');
  const paramsEl = document.getElementById('params');
  const readoutEl = document.getElementById('readout');
  const blurbEl = document.getElementById('dist-blurb');
  const marginEl = document.getElementById('margin');
  const canvas = document.getElementById('canvas');
  const sampleBtns = document.querySelectorAll('.ml-sample-row .ml-sbtn[data-n]');
  const resetBtn = document.getElementById('reset-samples');

  DIST_KEYS.forEach((k) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.dataset.dist = k;
    b.innerHTML = `${DISTS[k].name} <span style="font-family:var(--mono);font-size:11px;margin-left:6px;color:var(--muted)">${DISTS[k].type === 'discrete' ? 'discrete' : 'continuous'}</span>`;
    b.addEventListener('click', () => {
      stopAnim();
      state.distKey = k;
      resetParams();
      state.samples = [];
      state.runningMean = [];
      state.pendingQueue = [];
      if (state.annotationTimer) { clearTimeout(state.annotationTimer); state.annotationTimer = null; }
      state.annotation = null;
      // Re-seed a modest sample on every switch so the histogram-vs-curve
      // story remains visible — without this, picking a new distribution
      // shows only the analytic curve and the "watch it converge" promise dies.
      const dist = DISTS[k];
      const seeded = [];
      for (let i = 0; i < 250; i++) seeded.push(dist.sample(state.params));
      absorbSamples(seeded);
      // Brief storytelling annotation introducing the new distribution.
      setAnnotation(`${dist.name}: 250 samples already drawn — slide parameters to see histogram chase the curve.`);
      render();
    });
    b.style.cursor = 'pointer';
    distListEl.appendChild(b);
  });

  // Sample buttons: pre-generate the values, push to a pending queue,
  // then drop them ~50/frame so the convergence is *visible*, not instant.
  sampleBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const n = +btn.dataset.n;
      const dist = DISTS[state.distKey];
      const generated = [];
      for (let i = 0; i < n; i++) generated.push(dist.sample(state.params));
      // Small batches drop instantly for snappiness.
      if (n <= 10) {
        absorbSamples(generated);
        canvasCtl.redraw();
      } else {
        state.pendingQueue.push(...generated);
        startAnim();
      }
    });
  });
  resetBtn.addEventListener('click', () => {
    stopAnim();
    state.samples = [];
    state.runningMean = [];
    state.pendingQueue = [];
    canvasCtl.redraw();
    updateReadout();
  });

  function absorbSamples(arr) {
    let runSum = state.samples.length
      ? state.runningMean[state.runningMean.length - 1] * state.samples.length
      : 0;
    for (const s of arr) {
      state.samples.push(s);
      runSum += s;
      state.runningMean.push(runSum / state.samples.length);
    }
  }

  function startAnim() {
    if (state.animId) return;
    const tick = () => {
      const PER_FRAME = 50;
      const next = state.pendingQueue.splice(0, PER_FRAME);
      if (next.length) {
        absorbSamples(next);
        canvasCtl.redraw();
        updateReadout();
      }
      if (state.pendingQueue.length) {
        state.animId = requestAnimationFrame(tick);
      } else {
        state.animId = null;
      }
    };
    state.animId = requestAnimationFrame(tick);
  }
  function stopAnim() {
    if (state.animId) cancelAnimationFrame(state.animId);
    state.animId = null;
  }

  // Plain-language annotation for a slider change: "Now σ = 0.5 → narrower curve, ±1σ band shrinks."
  function describeChange(distKey, key, oldV, newV) {
    const d = (newV - oldV);
    const dir = d > 0 ? 'wider' : 'narrower';
    const f = (v) => formatNum(v);
    if (distKey === 'gaussian') {
      if (key === 'mu')    return `μ = ${f(newV)} → bell shifts ${d > 0 ? 'right' : 'left'}; ±1σ band travels with it.`;
      if (key === 'sigma') return `σ = ${f(newV)} → ${dir} curve, ±1σ band ${d > 0 ? 'expands' : 'shrinks'}.`;
    }
    if (distKey === 'bernoulli') {
      return `p = ${f(newV)} → 1-stem ${d > 0 ? 'taller' : 'shorter'}; mean = p moves to ${f(newV)}.`;
    }
    if (distKey === 'binomial') {
      if (key === 'n') return `n = ${Math.round(newV)} trials → support widens; shape smooths toward Gaussian as n grows.`;
      if (key === 'p') return `p = ${f(newV)} → mass shifts ${d > 0 ? 'right' : 'left'}; mean = np moves to ${f(state.params.n * newV)}.`;
    }
    if (distKey === 'poisson') {
      return `λ = ${f(newV)} → mean and variance both = λ; mass ${d > 0 ? 'spreads to higher counts' : 'pulls toward zero'}.`;
    }
    if (distKey === 'exponential') {
      return `λ = ${f(newV)} → mean wait = 1/λ = ${f(1 / newV)}; faster rate, shorter waits.`;
    }
    if (distKey === 'uniform') {
      const a = state.params.a, b = state.params.b;
      return `[a, b] = [${f(a)}, ${f(b)}] → height = 1/(b−a) = ${f(1 / (b - a))}.`;
    }
    return `${key} = ${f(newV)}.`;
  }

  function setAnnotation(text) {
    if (state.annotationTimer) clearTimeout(state.annotationTimer);
    state.annotation = { text, until: performance.now() + 2600 };
    state.annotationTimer = setTimeout(() => {
      state.annotation = null;
      state.annotationTimer = null;
      canvasCtl.redraw();
    }, 2700);
  }

  // Build parameter sliders
  function renderParams() {
    paramsEl.innerHTML = '';
    const dist = DISTS[state.distKey];
    dist.params.forEach((prm) => {
      const wrap = document.createElement('div');
      wrap.innerHTML = `
        <div class="ml-param-head">
          <span class="label">${prm.label}</span>
          <b>${formatNum(state.params[prm.key])}</b>
        </div>
        <input type="range" min="${prm.min}" max="${prm.max}" step="${prm.step}" value="${state.params[prm.key]}" title="drag to change ${prm.label}" />
      `;
      const slider = wrap.querySelector('input');
      slider.style.cursor = 'ew-resize';
      slider.addEventListener('input', (e) => {
        const oldV = state.params[prm.key];
        const newV = +e.target.value;
        state.params[prm.key] = newV;
        // Reset samples when params change so the histogram doesn't visually lag the curve.
        stopAnim();
        state.samples = [];
        state.runningMean = [];
        state.pendingQueue = [];
        // Set storytelling annotation describing what just changed.
        if (oldV !== newV) setAnnotation(describeChange(state.distKey, prm.key, oldV, newV));
        render();
      });
      paramsEl.appendChild(wrap);
    });
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    const dist = DISTS[state.distKey];
    const params = state.params;
    const samples = state.samples;

    // Reserve a strip at the top for the structural micro-schematic.
    const headerH = 38;
    const pad = { l: 50, r: 24, t: 24 + headerH, b: 44 };
    const W = size.w - pad.l - pad.r;
    const H = size.h - pad.t - pad.b;

    drawHeaderSchema(ctx, size, headerH, dist);

    let xMin, xMax;
    if (dist.type === 'continuous') {
      [xMin, xMax] = dist.range(params);
    } else {
      [xMin, xMax] = dist.support(params);
      xMin -= 0.5; xMax += 0.5;
    }

    // Compute curve max
    let yMax = 0;
    if (dist.type === 'continuous') {
      for (let i = 0; i <= 200; i++) {
        const x = xMin + (xMax - xMin) * i / 200;
        const v = dist.pdf(x, params);
        if (v > yMax) yMax = v;
      }
    } else {
      for (let k = Math.ceil(xMin); k <= Math.floor(xMax); k++) {
        const v = dist.pmf(k, params);
        if (v > yMax) yMax = v;
      }
    }
    yMax = yMax * 1.18 + 1e-6;

    // Histogram bins
    let bins = [];
    let histYMax = 0;
    if (samples.length) {
      if (dist.type === 'continuous') {
        const nb = 40;
        bins = new Array(nb).fill(0);
        for (const s of samples) {
          const b = Math.floor((s - xMin) / (xMax - xMin) * nb);
          if (b >= 0 && b < nb) bins[b]++;
        }
        const binW = (xMax - xMin) / nb;
        for (let i = 0; i < nb; i++) {
          const density = bins[i] / samples.length / binW;
          if (density > histYMax) histYMax = density;
        }
      } else {
        const lo = Math.ceil(xMin), hi = Math.floor(xMax);
        bins = new Array(hi - lo + 1).fill(0);
        for (const s of samples) {
          const idx = s - lo;
          if (idx >= 0 && idx < bins.length) bins[idx]++;
        }
        for (let i = 0; i < bins.length; i++) {
          const f = bins[i] / samples.length;
          if (f > histYMax) histYMax = f;
        }
      }
    }
    yMax = Math.max(yMax, histYMax * 1.05);

    const toPx = (x, y) => [pad.l + (x - xMin) / (xMax - xMin) * W, pad.t + (1 - y / yMax) * H];

    // Horizontal grid
    ctx.strokeStyle = 'rgba(38,35,32,0.07)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = yMax * i / 4;
      ctx.beginPath();
      ctx.moveTo(...toPx(xMin, y)); ctx.lineTo(...toPx(xMax, y)); ctx.stroke();
    }

    // Frame (left + bottom)
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t);
    ctx.lineTo(pad.l, pad.t + H);
    ctx.lineTo(pad.l + W, pad.t + H);
    ctx.stroke();

    // X-axis ticks + labels
    ctx.fillStyle = INK_FADE;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    const nTicks = dist.type === 'discrete' ? Math.min(11, Math.floor(xMax - xMin + 1)) : 6;
    for (let i = 0; i <= nTicks; i++) {
      const x = xMin + (xMax - xMin) * i / nTicks;
      const [px, py] = toPx(x, 0);
      ctx.fillText(formatNum(x), px, py + 16);
    }
    // Y-axis labels
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const y = yMax * i / 4;
      const [px, py] = toPx(xMin, y);
      ctx.fillText(y.toFixed(2), px - 6, py + 4);
    }
    // Y-axis title — names the function type
    ctx.save();
    ctx.translate(pad.l - 36, pad.t + H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillStyle = INK_FADE;
    ctx.fillText(dist.type === 'continuous' ? 'density  f(x)' : 'mass  P(k)', 0, 0);
    ctx.restore();

    // ── Variance band: shade ±1σ (and ±2σ for Gaussian) ──
    const stats = dist.stats(params);
    const mean = stats.Mean;
    const std  = Math.sqrt(stats.Variance);
    if (isFinite(mean) && isFinite(std) && std > 0) {
      // Outer ±2σ band — only for Gaussian, where the 95%-mass framing is canonical
      if (state.distKey === 'gaussian') {
        const a = Math.max(xMin, mean - 2 * std);
        const b = Math.min(xMax, mean + 2 * std);
        if (b > a) {
          const [x0] = toPx(a, 0);
          const [x1] = toPx(b, 0);
          ctx.fillStyle = 'rgba(122,31,36,0.04)';
          ctx.fillRect(x0, pad.t, x1 - x0, H);
        }
      }
      // ±1σ band
      const a = Math.max(xMin, mean - std);
      const b = Math.min(xMax, mean + std);
      if (b > a) {
        const [x0] = toPx(a, 0);
        const [x1] = toPx(b, 0);
        ctx.fillStyle = 'rgba(122,31,36,0.08)';
        ctx.fillRect(x0, pad.t, x1 - x0, H);

        // Top-of-band annotation
        ctx.strokeStyle = 'rgba(122,31,36,0.45)';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(x0, pad.t + 10); ctx.lineTo(x0, pad.t + H);
        ctx.moveTo(x1, pad.t + 10); ctx.lineTo(x1, pad.t + H);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = ACCENT;
        ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
        ctx.textAlign = 'center';
        ctx.fillText('±1σ', (x0 + x1) / 2, pad.t + 12);
      }
    }

    // Sample histogram (grey bars, drawn under the theoretical curve)
    if (samples.length) {
      ctx.fillStyle = 'rgba(38,35,32,0.20)';
      if (dist.type === 'continuous') {
        const nb = bins.length;
        const binW = (xMax - xMin) / nb;
        for (let i = 0; i < nb; i++) {
          const density = bins[i] / samples.length / binW;
          const x0 = xMin + i * binW;
          const [bx0, by0] = toPx(x0, density);
          const [bx1, by1] = toPx(x0 + binW, 0);
          ctx.fillRect(bx0 + 0.5, by0, bx1 - bx0 - 1, by1 - by0);
        }
      } else {
        const lo = Math.ceil(xMin);
        const w = (W / (xMax - xMin)) * 0.55;
        for (let i = 0; i < bins.length; i++) {
          const k = lo + i;
          const f = bins[i] / samples.length;
          const [px, py] = toPx(k, f);
          // grey histogram offset slightly to the right of stems
          ctx.fillRect(px + 2, py, w, pad.t + H - py);
        }
      }
    }

    // ── Theoretical: continuous = filled area + curve; discrete = lollipops ──
    if (dist.type === 'continuous') {
      ctx.fillStyle = ACCENT_FILL;
      ctx.beginPath();
      ctx.moveTo(...toPx(xMin, 0));
      for (let i = 0; i <= 300; i++) {
        const x = xMin + (xMax - xMin) * i / 300;
        ctx.lineTo(...toPx(x, dist.pdf(x, params)));
      }
      ctx.lineTo(...toPx(xMax, 0));
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i <= 300; i++) {
        const x = xMin + (xMax - xMin) * i / 300;
        const [px, py] = toPx(x, dist.pdf(x, params));
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
    } else {
      // Discrete: lollipops. Thin stem + filled circle head, with visible gaps.
      const lo = Math.ceil(xMin), hi = Math.floor(xMax);
      const wLolly = Math.min(14, (W / (xMax - xMin)) * 0.18);
      const stemW  = Math.max(2, wLolly * 0.45);
      const baseY  = pad.t + H;
      ctx.lineWidth = stemW;
      for (let k = lo; k <= hi; k++) {
        const v = dist.pmf(k, params);
        if (v < 1e-9) continue;
        const [px, py] = toPx(k, v);
        // Stem
        ctx.strokeStyle = ACCENT;
        ctx.beginPath();
        ctx.moveTo(px, baseY);
        ctx.lineTo(px, py);
        ctx.stroke();
        // Head
        ctx.fillStyle = ACCENT;
        ctx.beginPath();
        ctx.arc(px, py, wLolly / 2, 0, Math.PI * 2);
        ctx.fill();
      }
      // Discreteness hint under axis
      ctx.fillStyle = 'rgba(122,31,36,0.55)';
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'left';
      ctx.fillText('discrete: probability lives only on integer outcomes',
        pad.l + 4, pad.t + H + 32);
    }

    // ── Anchored annotations: μ at peak, σ as horizontal arrow at inflection (Gaussian only) ──
    if (state.distKey === 'gaussian' && isFinite(mean) && std > 0) {
      // μ marker at peak
      const peakY = dist.pdf(mean, params);
      const [mx, my] = toPx(mean, peakY);
      ctx.strokeStyle = 'rgba(38,35,32,0.45)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(mx, my); ctx.lineTo(mx, pad.t + H);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = INK;
      ctx.font = 'italic 13px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText('μ', mx, my - 6);

      // σ arrow at inflection (mean → mean+σ at half-peak height region)
      const yArrow = pad.t + H * 0.40;
      const [ax0] = toPx(mean, 0);
      const [ax1] = toPx(mean + std, 0);
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(ax0, yArrow); ctx.lineTo(ax1, yArrow);
      ctx.stroke();
      // Arrow heads
      const ah = 4;
      ctx.beginPath();
      ctx.moveTo(ax0, yArrow); ctx.lineTo(ax0 + ah, yArrow - ah); ctx.moveTo(ax0, yArrow); ctx.lineTo(ax0 + ah, yArrow + ah);
      ctx.moveTo(ax1, yArrow); ctx.lineTo(ax1 - ah, yArrow - ah); ctx.moveTo(ax1, yArrow); ctx.lineTo(ax1 - ah, yArrow + ah);
      ctx.stroke();
      ctx.fillStyle = ACCENT;
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText('σ', (ax0 + ax1) / 2, yArrow - 5);
    }

    // ── Anchored equation, just under the curve, near the peak ──
    drawAnchoredEquation(ctx, size, pad, H, dist);

    // ── Mean line + label ──
    if (isFinite(mean) && mean >= xMin && mean <= xMax) {
      ctx.strokeStyle = 'rgba(38,35,32,0.55)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(...toPx(mean, 0));
      ctx.lineTo(...toPx(mean, yMax * 0.97));
      ctx.stroke();
      ctx.setLineDash([]);
      // The mean is already marked 'μ' at the peak and shown in the rail readout,
      // so no numeric label here — it used to collide with the σ arrow and sample badge.
    }

    // ── Sample-count badge top-right ──
    if (samples.length) {
      ctx.fillStyle = INK_FADE;
      ctx.font = SERIF_LABEL_SM;
      ctx.textAlign = 'right';
      ctx.fillText(`${samples.length} samples`, size.w - pad.r, pad.t + 14);
    }

    // ── Convergence inset (running sample mean → true mean) ──
    drawConvergenceInset(ctx, size, pad, mean);

    // ── Storytelling annotation: short on-canvas message describing the last slider change ──
    drawSliderAnnotation(ctx, size, pad);
  }

  function drawSliderAnnotation(ctx, size, pad) {
    const a = state.annotation;
    if (!a) return;
    const now = performance.now();
    const remaining = a.until - now;
    if (remaining <= 0) return;
    // Fade out in the last 600ms.
    const alpha = remaining > 600 ? 1 : Math.max(0, remaining / 600);

    const txt = a.text;
    ctx.save();
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    const w = ctx.measureText(txt).width;
    const padX = 10, padY = 7;
    const x = pad.l + 16;
    const y = pad.t + 54;
    ctx.globalAlpha = alpha;
    ctx.fillStyle = 'rgba(255,253,246,0.95)';
    ctx.fillRect(x - padX, y - padY, w + padX * 2, 16 + padY);
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1;
    ctx.strokeRect(x - padX, y - padY, w + padX * 2, 16 + padY);
    // Small arrow-tab on the left edge to point at the curve change
    ctx.fillStyle = ACCENT;
    ctx.fillRect(x - padX - 3, y - padY, 3, 16 + padY);
    ctx.fillStyle = INK;
    ctx.fillText(txt, x, y + 6);
    ctx.restore();

    // Schedule a repaint while the annotation is fading so the alpha animates.
    if (remaining < 700 && !state._fadeRaf) {
      state._fadeRaf = requestAnimationFrame(() => {
        state._fadeRaf = null;
        canvasCtl.redraw();
      });
    }
  }

  // Tiny structural schematic: outcomes axis → probability axis.
  function drawHeaderSchema(ctx, size, headerH, dist) {
    const x0 = 18, y0 = 12;
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('DISTRIBUTION', x0, y0 + 10);
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.fillStyle = INK;
    ctx.fillText('a function from outcomes to probability', x0, y0 + 26);

    // Mini schematic on the right: outcomes ─→ probability
    const sx = size.w - 280;
    const sy = y0 + 18;
    if (sx > x0 + 320) {
      ctx.font = 'italic 11px "Source Serif 4", Georgia, serif';
      ctx.fillStyle = INK_FADE;
      ctx.textAlign = 'right';
      ctx.fillText('outcomes', sx + 60, sy);
      ctx.fillText('probability', size.w - 18, sy);
      ctx.strokeStyle = 'rgba(38,35,32,0.45)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx + 70, sy - 4); ctx.lineTo(size.w - 80, sy - 4);
      ctx.stroke();
      // Arrowhead
      ctx.beginPath();
      ctx.moveTo(size.w - 80, sy - 4);
      ctx.lineTo(size.w - 86, sy - 8);
      ctx.moveTo(size.w - 80, sy - 4);
      ctx.lineTo(size.w - 86, sy);
      ctx.stroke();
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillStyle = INK_FADE;
      ctx.fillText(dist.type === 'continuous' ? 'f(x)' : 'P(k)',
        (sx + 70 + size.w - 80) / 2, sy - 8);
    }
  }

  function drawAnchoredEquation(ctx, size, pad, H, dist) {
    if (!dist.eqInline) return;
    const txt = dist.eqInline;
    const x = pad.l + 16;
    const y = pad.t + 30;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    const w = ctx.measureText(txt).width;
    ctx.fillStyle = 'rgba(255,253,246,0.85)';
    ctx.fillRect(x - 6, y - 12, w + 12, 18);
    ctx.strokeStyle = 'rgba(122,31,36,0.25)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x - 6, y - 12, w + 12, 18);
    ctx.fillStyle = ACCENT;
    ctx.fillText(txt, x, y);
  }

  function drawConvergenceInset(ctx, size, pad, trueMean) {
    if (state.runningMean.length < 2) return;
    const ins = {
      w: 168,
      h: 78,
      x: size.w - 168 - 24,
      y: pad.t + 36,
    };
    // Background card
    ctx.fillStyle = 'rgba(255,253,246,0.92)';
    ctx.fillRect(ins.x, ins.y, ins.w, ins.h);
    ctx.strokeStyle = 'rgba(38,35,32,0.32)';
    ctx.lineWidth = 1;
    ctx.strokeRect(ins.x, ins.y, ins.w, ins.h);

    // Title
    ctx.fillStyle = INK_FADE;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('SAMPLE MEAN → μ', ins.x + 8, ins.y + 14);

    // Compute y-range around trueMean
    const data = state.runningMean;
    let lo = trueMean, hi = trueMean;
    for (const v of data) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const span = Math.max(1e-6, (hi - lo) * 1.1);
    const yc = (trueMean - lo + (hi - trueMean)) / 2 + lo;
    const yLo = yc - span / 2;
    const yHi = yc + span / 2;

    const plotL = ins.x + 8, plotR = ins.x + ins.w - 8;
    const plotT = ins.y + 22, plotB = ins.y + ins.h - 8;
    const N = data.length;
    const xAt = (i) => plotL + (i / Math.max(1, N - 1)) * (plotR - plotL);
    const yAt = (v) => plotB - (v - yLo) / (yHi - yLo) * (plotB - plotT);

    // True-mean reference line
    ctx.strokeStyle = ACCENT_FAINT;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(plotL, yAt(trueMean)); ctx.lineTo(plotR, yAt(trueMean)); ctx.stroke();
    ctx.setLineDash([]);
    // Label μ on inset
    ctx.fillStyle = ACCENT;
    ctx.font = 'italic 10px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('μ', plotL + 1, yAt(trueMean) - 2);

    // Running mean trace — subsample if many points to avoid path explosion
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    const stride = Math.max(1, Math.floor(N / 240));
    for (let i = 0; i < N; i += stride) {
      const px = xAt(i), py = yAt(data[i]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    if ((N - 1) % stride !== 0) {
      ctx.lineTo(xAt(N - 1), yAt(data[N - 1]));
    }
    ctx.stroke();

    // x-axis label
    ctx.fillStyle = INK_FADE;
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`n=${N}`, plotR, plotB + 8);
  }

  // Lightweight readout-only update (for animation frames — avoids slider re-render)
  function updateReadout() {
    const dist = DISTS[state.distKey];
    const stats = dist.stats(state.params);
    const std = Math.sqrt(stats.Variance);
    let html = '';
    Object.entries(stats).forEach(([k, v]) => {
      html += `<div class="row"><span>${k}</span><b>${formatNum(v)}</b></div>`;
    });
    html += `<div class="row"><span>Std</span><b>${formatNum(std)}</b></div>`;
    if (state.samples.length) {
      const sm = state.runningMean[state.runningMean.length - 1];
      html += `<div class="row" style="border-top:1px solid var(--rule); margin-top:6px; padding-top:8px"><span>Sample mean (n=${state.samples.length})</span><b>${formatNum(sm)}</b></div>`;
    }
    readoutEl.innerHTML = html;
  }

  // ───────── Render ─────────
  function render() {
    Array.from(distListEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.dist === state.distKey);
    });
    renderParams();
    updateReadout();

    const dist = DISTS[state.distKey];
    blurbEl.innerHTML = `<em>${dist.name}.</em> ${dist.blurb}`;

    marginEl.innerHTML = `
      <div class="ml-margin-rule"></div>
      <div class="ml-margin-h" style="font-family:var(--mono); font-size:11px; letter-spacing:0.22em; text-transform:uppercase; color:var(--accent)">Before this</div>
      <p class="quiet" style="font-style:italic">Before formal distributions (Bernoulli, 1700s; Gauss, 1809), every probability problem was solved with one-off tricks. The unified language — a function from outcome to probability, parameterized by a few numbers — gave a single way to reason about coin flips, measurement errors, queue waits, biology, and physics.</p>
      <div class="ml-margin-rule thin"></div>
      <div class="ml-margin-h">In the margin</div>
      <div class="ml-equation">${dist.eq}</div>
      ${dist.eqGloss ? `<p class="quiet" style="margin-top:6px">${dist.eqGloss}</p>` : ''}
      <p>${dist.note}</p>
      <div class="ml-margin-rule thin"></div>
      <div class="ml-margin-h sm">Variance, visually</div>
      <p class="quiet">The shaded oxblood band is <em>μ ± σ</em> — roughly where the mass concentrates. For a Gaussian about 68% of samples land inside; for a uniform, exactly 58%; Chebyshev's inequality guarantees at least 0% within ±1σ but at least 75% within ±2σ for <em>any</em> distribution.</p>
      <div class="ml-margin-rule thin"></div>
      <div class="ml-margin-h sm">Law of large numbers</div>
      <p class="quiet">Watch the inset trace. As n grows, the running sample mean converges to <em>μ</em>. That convergence is the empirical heartbeat of statistics — and the reason a histogram of 1000 samples looks like the curve.</p>`;

    canvasCtl.redraw();
  }

  // Pre-populate a modest number of samples on first load so the
  // histogram-vs-curve story is visible at landing — a Gaussian with no
  // samples just looks like a curve, which doesn't pay off the "watch it converge" promise.
  (function seedInitialSamples() {
    const dist = DISTS[state.distKey];
    const seeded = [];
    for (let i = 0; i < 250; i++) seeded.push(dist.sample(state.params));
    absorbSamples(seeded);
  })();

  render();
})();
