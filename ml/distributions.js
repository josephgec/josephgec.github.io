/* Vol. VI — Probability Distributions */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, ACCENT, INK, INK_FADE, RULE_FAINT, RULE_AXIS,
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
      note: 'Every weight initialization in a neural network starts here.',
    },
  };

  const DIST_KEYS = Object.keys(DISTS);

  // ───────── State ─────────
  const state = {
    distKey: 'gaussian',
    params: {},
    samples: [],
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

  // Distribution list
  DIST_KEYS.forEach((k) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.dataset.dist = k;
    b.innerHTML = `${DISTS[k].name} <span style="font-family:var(--mono);font-size:11px;margin-left:6px;color:var(--muted)">${DISTS[k].type === 'discrete' ? 'discrete' : 'continuous'}</span>`;
    b.addEventListener('click', () => {
      state.distKey = k;
      resetParams();
      state.samples = [];
      render();
    });
    distListEl.appendChild(b);
  });

  // Sample / reset buttons
  sampleBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const n = +btn.dataset.n;
      const dist = DISTS[state.distKey];
      const next = state.samples.slice();
      for (let i = 0; i < n; i++) next.push(dist.sample(state.params));
      state.samples = next;
      render();
    });
  });
  resetBtn.addEventListener('click', () => {
    state.samples = [];
    render();
  });

  // Build parameter sliders for the current distribution
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
        <input type="range" min="${prm.min}" max="${prm.max}" step="${prm.step}" value="${state.params[prm.key]}" />
      `;
      const slider = wrap.querySelector('input');
      slider.addEventListener('input', (e) => {
        state.params[prm.key] = +e.target.value;
        // Reset samples when params change so the histogram doesn't visually lag the curve.
        state.samples = [];
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

    const pad = { l: 50, r: 24, t: 24, b: 40 };
    const W = size.w - pad.l - pad.r;
    const H = size.h - pad.t - pad.b;

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
    yMax = yMax * 1.15 + 1e-6;

    // Build histogram bins
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

    // Frame (left + bottom only — newspaper-y)
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

    // Sample histogram (grey bars)
    if (samples.length) {
      ctx.fillStyle = 'rgba(38,35,32,0.18)';
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
        const w = (W / (xMax - xMin)) * 0.8;
        for (let i = 0; i < bins.length; i++) {
          const k = lo + i;
          const f = bins[i] / samples.length;
          const [px, py] = toPx(k, f);
          ctx.fillRect(px - w / 2, py, w, pad.t + H - py);
        }
      }
    }

    // Theoretical curve / bars (oxblood)
    if (dist.type === 'continuous') {
      ctx.fillStyle = 'rgba(122,31,36,0.15)';
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
      const lo = Math.ceil(xMin), hi = Math.floor(xMax);
      ctx.fillStyle = 'rgba(122,31,36,0.55)';
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 2;
      const w = (W / (xMax - xMin)) * 0.55;
      for (let k = lo; k <= hi; k++) {
        const v = dist.pmf(k, params);
        if (v < 1e-9) continue;
        const [px, py] = toPx(k, v);
        ctx.fillRect(px - w / 2, py, w, pad.t + H - py);
        ctx.beginPath();
        ctx.moveTo(px - w / 2, py);
        ctx.lineTo(px + w / 2, py);
        ctx.stroke();
      }
    }

    // Mean line
    const stats = dist.stats(params);
    const mean = stats.Mean;
    if (isFinite(mean) && mean >= xMin && mean <= xMax) {
      ctx.strokeStyle = 'rgba(38,35,32,0.4)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(...toPx(mean, 0));
      ctx.lineTo(...toPx(mean, yMax * 0.95));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = INK_FADE;
      ctx.font = SERIF_LABEL_SM;
      ctx.textAlign = 'left';
      const [mx, my] = toPx(mean, yMax * 0.95);
      ctx.fillText(`mean = ${formatNum(mean)}`, mx + 4, my + 4);
    }

    if (samples.length) {
      ctx.fillStyle = INK_FADE;
      ctx.font = SERIF_LABEL_SM;
      ctx.textAlign = 'right';
      ctx.fillText(`${samples.length} samples`, size.w - pad.r, pad.t + 14);
    }
  }

  // ───────── Render ─────────
  function render() {
    // Active distribution highlight
    Array.from(distListEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.dist === state.distKey);
    });
    renderParams();

    const dist = DISTS[state.distKey];
    const stats = dist.stats(state.params);
    const std = Math.sqrt(stats.Variance);
    let html = '';
    Object.entries(stats).forEach(([k, v]) => {
      html += `<div class="row"><span>${k}</span><b>${formatNum(v)}</b></div>`;
    });
    html += `<div class="row"><span>Std</span><b>${formatNum(std)}</b></div>`;
    readoutEl.innerHTML = html;

    blurbEl.innerHTML = `<em>${dist.name}.</em> ${dist.blurb}`;

    marginEl.innerHTML = `
      <div class="ml-margin-rule"></div>
      <div class="ml-margin-h">In the margin</div>
      <div class="ml-equation">${dist.eq}</div>
      <p>${dist.note}</p>
      <div class="ml-margin-rule thin"></div>
      <div class="ml-margin-h sm">Try this</div>
      <p class="quiet">Drag the parameter sliders and watch the curve breathe. Then click <em>+1000</em> and see the grey histogram pour into the accent shape.</p>`;

    canvasCtl.redraw();
  }

  render();
})();
