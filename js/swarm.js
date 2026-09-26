/* Swarm stage — a glowworm swarm that re-forms into a figure for each
   section marked data-shape. Agents are assigned points in the figure;
   their glow follows a luciferin-style rule (rises in formation, decays
   out of it) modulated by a travelling wave. The cursor is a torch. */
(function () {
  var root = document.documentElement;
  root.classList.remove('no-js');
  var cv = document.getElementById('stage');
  if (!cv || !cv.getContext) return;
  var cx = cv.getContext('2d');
  var reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduce) root.classList.add('reduce');

  var css = getComputedStyle(root);
  function tok(n) { return css.getPropertyValue(n).trim(); }
  var BG = tok('--bg'), DOT = tok('--swarm-dot'), GLOW = tok('--swarm-glow');
  var FONT = '"Bricolage Grotesque", sans-serif', WEIGHT = 700;

  var W, H, dpr = Math.min(window.devicePixelRatio || 1, 2), N = 0, ag = [];
  var shapeName = '', filled = 0, running = false, t0 = 0;
  var mouse = { x: -1e4, y: -1e4 };

  function rgb(h) { return [1, 3, 5].map(function (i) { return parseInt(h.slice(i, i + 2), 16); }); }
  var trail = 'rgba(' + rgb(BG).join(',') + ',0.42)';

  // Glow sprite, drawn once.
  var spr = document.createElement('canvas'); spr.width = spr.height = 32;
  (function () {
    var g = spr.getContext('2d'), c = rgb(GLOW), r = g.createRadialGradient(16, 16, 0, 16, 16, 16);
    r.addColorStop(0, 'rgba(' + c.map(function (v) { return v * 0.85 | 0; }).join(',') + ',1)');
    r.addColorStop(0.18, 'rgba(' + c.join(',') + ',.95)');
    r.addColorStop(0.5, 'rgba(' + c.join(',') + ',.2)');
    r.addColorStop(1, 'rgba(' + c.join(',') + ',0)');
    g.fillStyle = r; g.fillRect(0, 0, 32, 32);
  })();

  // ───── Figures ─────
  function narrow() { return W < 800; }
  function region() {
    return narrow() ? { x: 20, y: 76, w: W - 40, h: H * 0.3 }
                    : { x: W * 0.5, y: H * 0.16, w: W * 0.44, h: H * 0.7 };
  }
  function sample(draw, step) {
    var o = document.createElement('canvas'); o.width = W; o.height = H;
    var g = o.getContext('2d', { willReadFrequently: true });
    g.fillStyle = g.strokeStyle = '#000'; draw(g);
    var d = g.getImageData(0, 0, W, H).data, pts = [];
    for (var y = 0; y < H; y += step)
      for (var x = (y / step) % 2 ? step / 2 : 0; x < W; x += step)
        if (d[((y * W + x) | 0) * 4 + 3] > 120) pts.push([x, y]);
    return pts;
  }
  function fitText(g, lines, x, y, maxW, maxH) {
    var fs = 400; g.font = WEIGHT + ' ' + fs + 'px ' + FONT;
    var widest = Math.max.apply(null, lines.map(function (l) { return g.measureText(l).width; }));
    fs = Math.min(fs * maxW / widest, maxH / (lines.length * 1.02));
    g.font = WEIGHT + ' ' + fs + 'px ' + FONT; g.textBaseline = 'top';
    lines.forEach(function (l, i) { g.fillText(l, x, y + i * fs * 1.02); });
  }

  var SHAPES = {
    hero: function (g) {
      var pad = Math.max(20, Math.min(56, W * 0.04));
      if (narrow()) fitText(g, ['Emergent', 'behavior,'], pad, H * 0.13, W - pad * 2, H * 0.3);
      else fitText(g, ['Emergent behavior,'], pad - W * 0.006, H * 0.17, W - pad * 2, H * 0.3);
    },
    years: function (g) { var r = region(); fitText(g, ['2008', '2014', '2026'], r.x, r.y, r.w * 0.8, r.h); },
    ratchet: function (g) {   // "a ratchet that only moves forward"
      var r = region(), n = 6, sw = r.w / n, sh = r.h / (n + 1), x = r.x, y = r.y + r.h, i;
      g.lineWidth = narrow() ? 5 : 8; g.lineJoin = 'miter'; g.beginPath(); g.moveTo(x, y);
      for (i = 0; i < n; i++) { x += sw * 0.72; g.lineTo(x, y); y -= sh; g.lineTo(x, y); x += sw * 0.28; g.lineTo(x, y); }
      g.stroke();
      for (i = 0; i < n; i++) {
        var px = r.x + sw * (i + 0.72), py = r.y + r.h - sh * (i + 1);
        g.beginPath(); g.moveTo(px, py); g.lineTo(px - sw * 0.25, py + sh * 0.5); g.lineTo(px, py + sh * 0.5); g.fill();
      }
    },
    network: function (g) {
      var r = region(), layers = [4, 6, 6, 3], nodes = [];
      layers.forEach(function (n, li) {
        var col = [];
        for (var k = 0; k < n; k++) col.push([r.x + r.w * (0.06 + 0.88 * li / (layers.length - 1)), r.y + r.h * (k + 0.5) / n]);
        nodes.push(col);
      });
      g.lineWidth = narrow() ? 1.2 : 1.6;
      for (var li = 0; li < nodes.length - 1; li++)
        nodes[li].forEach(function (a) { nodes[li + 1].forEach(function (b) { g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke(); }); });
      var rad = Math.min(r.w, r.h) * 0.045;
      nodes.forEach(function (col) { col.forEach(function (p) { g.beginPath(); g.arc(p[0], p[1], rad, 0, 7); g.fill(); }); });
    },
    // The glowworm problem itself: agents settle around several sources at once.
    sources: function () {
      var r = region(), c = [[0.28, 0.3, 0.2], [0.72, 0.42, 0.16], [0.4, 0.78, 0.14]], pts = [];
      for (var i = 0; i < N; i++) {
        var s = c[i % 3], a = Math.random() * 6.283, d = Math.pow(Math.random(), 1.5) * s[2] * Math.min(r.w, r.h);
        pts.push([r.x + r.w * s[0] + Math.cos(a) * d, r.y + r.h * s[1] + Math.sin(a) * d]);
      }
      return pts;
    },
    mail: function (g) { var r = region(); fitText(g, ['hello@', 'jthomas', '.site'], r.x, r.y + r.h * 0.05, r.w * 0.95, r.h * 0.9); }
  };

  function targetsFor(name) {
    var f = SHAPES[name], pts = f.length === 0 ? f() : sample(f, narrow() ? 5 : 6);
    if (!pts.length) return { pts: [], filled: 0 };
    for (var i = pts.length - 1; i > 0; i--) { var j = Math.random() * (i + 1) | 0, t = pts[i]; pts[i] = pts[j]; pts[j] = t; }
    var out = [];
    for (i = 0; i < N; i++) {
      var p = pts[i % pts.length];
      out.push(i < pts.length ? p : [p[0] + (Math.random() - 0.5) * 3, p[1] + (Math.random() - 0.5) * 3]);
    }
    return { pts: out, filled: Math.min(pts.length, N) };
  }

  // Break formation, scatter, then regroup on the new figure with a staggered call.
  function setShape(name) {
    shapeName = name;
    var ambient = name === 'ambient';
    cv.classList.toggle('ambient', ambient);
    var now = performance.now(), sh = ambient ? null : targetsFor(name), byX = name === 'hero';
    filled = sh ? sh.filled : 0;
    for (var i = 0; i < N; i++) {
      var a = ag[i];
      if (!sh) { a.tx = null; a.extra = false; continue; }
      var t = sh.pts[i];
      a.tx = t[0]; a.ty = t[1]; a.extra = i >= sh.filled;
      if (reduce) { a.x = t[0]; a.y = t[1]; a.l = 1; continue; }
      a.l *= 0.3;
      a.vx += (Math.random() - 0.5) * 6; a.vy += (Math.random() - 0.5) * 6;
      a.at = now + 350 + (byX ? (t[0] / W) * 1400 + Math.random() * 300 : Math.random() * 900);
    }
    if (reduce) drawStatic();
  }

  function glow(a, now) {
    var w = 0.5 + 0.5 * Math.sin(now * 0.0016 - a.x * 0.006 - a.y * 0.002 + a.ph * 0.35);
    return a.l * (0.45 + 0.55 * w * w) + (a.l > 0.9 ? 0.04 * Math.sin(now * 0.004 * a.sp + a.ph) : 0);
  }

  function step(now) {
    var mx = mouse.x, my = mouse.y;
    for (var i = 0; i < N; i++) {
      var a = ag[i];
      if (a.tx !== null && now >= a.at) {
        var dx = a.tx - a.x, dy = a.ty - a.y, near = dx * dx + dy * dy < 9;
        a.vx += dx * 0.011; a.vy += dy * 0.011;
        a.l += ((near ? 1 : 0.2) - a.l) * (near ? 0.04 : 0.02);
      } else {
        a.vx += (Math.random() - 0.5) * 0.5; a.vy += (Math.random() - 0.5) * 0.5;
        a.l += (0.12 - a.l) * 0.02;
        if (a.x < -20) a.x = W + 20; else if (a.x > W + 20) a.x = -20;
        if (a.y < -20) a.y = H + 20; else if (a.y > H + 20) a.y = -20;
      }
      var ex = a.x - mx, ey = a.y - my, d2 = ex * ex + ey * ey;
      if (d2 < 12000) {
        var d = Math.sqrt(d2) || 1, f = (110 - d) * 0.035;
        a.vx += ex / d * f; a.vy += ey / d * f; a.l = Math.min(1, a.l + 0.08);
      }
      a.vx *= 0.86; a.vy *= 0.86; a.x += a.vx; a.y += a.vy;
    }
  }

  function paint(now, fillBg) {
    cx.globalAlpha = 1;
    cx.fillStyle = fillBg; cx.fillRect(0, 0, W, H);
    var intro = reduce ? 1 : Math.min((now - t0) / 1200, 1), i, a, tw;
    cx.fillStyle = DOT;
    for (i = 0; i < N; i++) {
      a = ag[i];
      var born = reduce ? 1 : Math.min(1, Math.max(0, intro * 1.6 - (i / N) * 0.6));
      if (born <= 0) continue;
      tw = glow(a, now);
      if (tw < 0.78 || a.extra) { cx.globalAlpha = born * (0.35 + tw * 0.65); cx.fillRect(a.x, a.y, 1.9, 1.9); }
    }
    for (i = 0; i < N; i++) {
      a = ag[i]; if (a.extra) continue;
      tw = glow(a, now);
      if (tw >= 0.78) {
        var s = 6 + tw * 8;
        cx.globalAlpha = Math.min(1, (tw - 0.7) * 2.2) * Math.min(1, intro * 1.6);
        cx.drawImage(spr, a.x - s / 2, a.y - s / 2, s, s);
      }
    }
    cx.globalAlpha = 1;
  }
  function drawStatic() { paint(performance.now(), BG); }

  function loop(now) {
    if (!running) return;
    step(now); paint(now, trail);
    requestAnimationFrame(loop);
  }
  function start() { if (!running && !reduce && !document.hidden) { running = true; requestAnimationFrame(loop); } }

  function resize() {
    W = innerWidth; H = innerHeight;
    cv.width = W * dpr; cv.height = H * dpr; cx.setTransform(dpr, 0, 0, dpr, 0, 0);
    N = Math.round(Math.min(1800, Math.max(700, W * H / 560)));   // density tracks screen area
    while (ag.length < N) ag.push({ x: Math.random() * W, y: Math.random() * H, vx: 0, vy: 0, l: 0, tx: null, ty: null, at: 0, ph: Math.random() * 6.28, sp: 0.6 + Math.random() * 1.6, extra: false });
    ag.length = N;
    if (shapeName) setShape(shapeName);
  }

  // ───── Wiring ─────
  var navLinks = [].slice.call(document.querySelectorAll('.top nav a[href^="#"]'));
  var io = new IntersectionObserver(function (es) {
    es.forEach(function (e) {
      if (!e.isIntersecting) return;
      var s = e.target.dataset.shape;
      if (s !== shapeName) setShape(s);
      navLinks.forEach(function (l) { l.classList.toggle('on', l.getAttribute('href') === '#' + e.target.id); });
    });
  }, { rootMargin: '-45% 0px -45% 0px' });

  addEventListener('pointermove', function (e) { mouse.x = e.clientX; mouse.y = e.clientY; }, { passive: true });
  document.addEventListener('pointerleave', function () { mouse.x = mouse.y = -1e4; });
  document.addEventListener('visibilitychange', function () { if (document.hidden) running = false; else start(); });
  var rt; addEventListener('resize', function () { clearTimeout(rt); rt = setTimeout(function () { var o = W; resize(); if (reduce || Math.abs(o - W) > 0) drawStatic(); }, 150); });

  var reveals = [].slice.call(document.querySelectorAll('.reveal'));
  setTimeout(function () { reveals.forEach(function (el) { el.classList.add('in'); }); }, 5000);  // never strand the copy
  Promise.all([document.fonts.ready, document.fonts.load(WEIGHT + ' 100px "Bricolage Grotesque"')]).then(function () {
    resize();
    cx.fillStyle = BG; cx.fillRect(0, 0, W, H);
    t0 = performance.now();
    setShape('hero');
    [].forEach.call(document.querySelectorAll('[data-shape]'), function (s) { io.observe(s); });
    // The hero copy lands once the headline has assembled.
    reveals.forEach(function (el) {
      if (reduce) return el.classList.add('in');
      setTimeout(function () { el.classList.add('in'); }, 2400 + (+el.dataset.delay || 0));
    });
    if (reduce) drawStatic(); else start();
  });
})();
