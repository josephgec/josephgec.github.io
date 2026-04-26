/* ML, Visualized — shared math + UI helpers (vanilla JS).
   Exposed on window.MLViz so the per-page scripts can grab what they need. */

window.MLViz = (function () {
  'use strict';

  // ───────── Tokens (must mirror styles.css / shared.css for canvas drawing) ─────────
  const ACCENT = '#7a1f24';
  const INK = '#262320';
  const INK_FADE = 'rgba(38,35,32,0.55)';
  const RULE_DARK = 'rgba(38,35,32,0.32)';
  const RULE_FAINT = 'rgba(38,35,32,0.06)';
  const RULE_AXIS = 'rgba(38,35,32,0.18)';
  const ACCENT_SOFT = 'rgba(122,31,36,0.18)';
  const ACCENT_FAINT = 'rgba(122,31,36,0.35)';
  const ACCENT_FILL = 'rgba(122,31,36,0.10)';
  const SERIF_LABEL = 'italic 16px "Source Serif 4", Georgia, serif';
  const SERIF_LABEL_SM = 'italic 14px "Source Serif 4", Georgia, serif';
  const MONO_LABEL = '13px "JetBrains Mono", ui-monospace, monospace';

  // ───────── Math ─────────
  const mul = (M, [x, y]) => [M[0][0]*x + M[0][1]*y, M[1][0]*x + M[1][1]*y];
  const norm = ([x, y]) => Math.hypot(x, y);
  const normalize = ([x, y]) => {
    const n = Math.hypot(x, y);
    return n < 1e-9 ? [0, 0] : [x/n, y/n];
  };
  const dot = ([a, b], [c, d]) => a*c + b*d;
  const transpose = (M) => [[M[0][0], M[1][0]], [M[0][1], M[1][1]]];
  const matmul = (A, B) => [
    [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
    [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]],
  ];
  const matEq = (A, B, eps = 1e-6) =>
    Math.abs(A[0][0]-B[0][0]) < eps && Math.abs(A[0][1]-B[0][1]) < eps &&
    Math.abs(A[1][0]-B[1][0]) < eps && Math.abs(A[1][1]-B[1][1]) < eps;

  function eigen2(M) {
    const a = M[0][0], b = M[0][1], c = M[1][0], d = M[1][1];
    const tr = a + d, det = a*d - b*c;
    const disc = tr*tr - 4*det;
    if (disc < -1e-9) return [];
    const s = Math.sqrt(Math.max(0, disc));
    const l1 = (tr + s) / 2, l2 = (tr - s) / 2;
    const eigVec = (lam) => {
      const m00 = a - lam, m01 = b, m10 = c, m11 = d - lam;
      let v;
      if (Math.abs(m01) > 1e-9 || Math.abs(m00) > 1e-9) {
        v = [m01, -m00];
        if (norm(v) < 1e-9) v = [-m11, m10];
      } else {
        v = [m11, -m10];
        if (norm(v) < 1e-9) v = [1, 0];
      }
      return normalize(v);
    };
    return [
      { value: l1, vector: eigVec(l1) },
      { value: l2, vector: eigVec(l2) },
    ];
  }

  function svd2(A) {
    const At = transpose(A);
    const AtA = matmul(At, A);
    const eigs = eigen2(AtA);
    if (!eigs.length) return null;
    eigs.sort((a, b) => b.value - a.value);
    const sigma1 = Math.sqrt(Math.max(0, eigs[0].value));
    const sigma2 = Math.sqrt(Math.max(0, eigs[1].value));
    const v1 = eigs[0].vector;
    const v2 = [-v1[1], v1[0]];
    let u1, u2;
    if (sigma1 > 1e-9) {
      const Av1 = mul(A, v1);
      u1 = [Av1[0]/sigma1, Av1[1]/sigma1];
    } else u1 = [1, 0];
    if (sigma2 > 1e-9) {
      const Av2 = mul(A, v2);
      u2 = [Av2[0]/sigma2, Av2[1]/sigma2];
    } else {
      u2 = [-u1[1], u1[0]];
    }
    return {
      U: [[u1[0], u2[0]], [u1[1], u2[1]]],
      S: [sigma1, sigma2],
      V: [[v1[0], v2[0]], [v1[1], v2[1]]],
    };
  }

  // ───────── Formatting ─────────
  function formatNum(n) {
    if (!isFinite(n)) return '∞';
    if (Math.abs(n) < 1e-9) return '0';
    return (Math.round(n * 100) / 100).toString();
  }
  const SUB_DIGITS = '₀₁₂₃₄₅₆₇₈₉';
  function sub(n) { return String(n).replace(/[0-9]/g, (d) => SUB_DIGITS[+d]); }

  // ───────── Drawing ─────────
  function arrow(ctx, fromPx, toPx, color, width = 2, head = 9) {
    const [x1, y1] = fromPx, [x2, y2] = toPx;
    const ang = Math.atan2(y2 - y1, x2 - x1);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - head*Math.cos(ang - Math.PI/7), y2 - head*Math.sin(ang - Math.PI/7));
    ctx.lineTo(x2 - head*Math.cos(ang + Math.PI/7), y2 - head*Math.sin(ang + Math.PI/7));
    ctx.closePath(); ctx.fill();
  }

  // ───────── Canvas size handling ─────────
  // Sets up a HiDPI canvas inside its parent, calling `redraw(ctx, size)` whenever
  // the parent resizes or whenever the returned `redraw()` is called.
  // opts.square (default true) clamps to the smaller dimension; pass false for
  // rectangular canvases that should fill the parent.
  function attachCanvas(canvas, redraw, opts = {}) {
    const square = opts.square !== false;
    let size = { w: 580, h: 580 };
    const setupCtx = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = size.w * dpr;
      canvas.height = size.h * dpr;
      canvas.style.width = size.w + 'px';
      canvas.style.height = size.h + 'px';
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return ctx;
    };
    const draw = () => {
      const ctx = setupCtx();
      ctx.clearRect(0, 0, size.w, size.h);
      redraw(ctx, size);
    };
    const ro = new ResizeObserver(() => {
      const r = canvas.parentElement.getBoundingClientRect();
      if (r.width < 1 || r.height < 1) return;
      if (square) {
        const s = Math.min(r.width, r.height);
        size = { w: s, h: s };
      } else {
        size = { w: r.width, h: r.height };
      }
      draw();
    });
    ro.observe(canvas.parentElement);
    return { redraw: draw, getSize: () => size };
  }

  // ───────── Drag-to-scrub matrix entry ─────────
  // Renders a 2x2 grid of editable cells inside `container`. `getMatrix()` returns
  // the current matrix, `setMatrix(M)` updates it. `onChange` fires after edits.
  function mountMatrixCells(container, getMatrix, setMatrix) {
    container.innerHTML = '';
    container.classList.add('ml-matrix');
    const left = document.createElement('div');
    left.className = 'ml-bracket';
    const right = document.createElement('div');
    right.className = 'ml-bracket right';
    const grid = document.createElement('div');
    grid.className = 'ml-entries';

    const cells = [];
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 2; c++) {
        const wrap = document.createElement('div');
        wrap.className = 'ml-cell-wrap';
        const cell = document.createElement('div');
        cell.className = 'ml-cell';
        cell.title = 'drag to scrub · click to type';
        cell.dataset.r = r; cell.dataset.c = c;
        wrap.appendChild(cell);
        grid.appendChild(wrap);
        cells.push(cell);
        attachScrub(cell, r, c, getMatrix, setMatrix);
      }
    }
    container.appendChild(left);
    container.appendChild(grid);
    container.appendChild(right);

    function refresh() {
      const M = getMatrix();
      cells.forEach((cell) => {
        if (cell.dataset.editing === '1') return;
        const r = +cell.dataset.r, c = +cell.dataset.c;
        cell.textContent = formatNum(M[r][c]);
      });
    }
    refresh();
    return { refresh };
  }

  function attachScrub(cell, r, c, getMatrix, setMatrix) {
    cell.addEventListener('mousedown', (e) => {
      if (cell.dataset.editing === '1') return;
      e.preventDefault();
      const startX = e.clientX;
      const startVal = getMatrix()[r][c];
      let moved = false;
      const onMove = (ev) => {
        const dx = ev.clientX - startX;
        if (!moved && Math.abs(dx) > 3) moved = true;
        if (moved) {
          const v = Math.round((startVal + dx * 0.02) * 100) / 100;
          const M = getMatrix();
          const next = M.map((row) => row.slice());
          next[r][c] = v;
          setMatrix(next);
        }
      };
      const onUp = () => {
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('mouseup', onUp);
        if (!moved) startEdit(cell, r, c, getMatrix, setMatrix);
      };
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onUp);
    });
  }

  function startEdit(cell, r, c, getMatrix, setMatrix) {
    const M = getMatrix();
    const input = document.createElement('input');
    input.className = 'ml-cell-input';
    input.value = formatNum(M[r][c]);
    cell.replaceWith(input);
    cell.dataset.editing = '1';
    input.focus();
    input.select();
    const commit = () => {
      const n = parseFloat(input.value);
      if (!isNaN(n)) {
        const M2 = getMatrix().map((row) => row.slice());
        M2[r][c] = n;
        setMatrix(M2);
      }
      input.replaceWith(cell);
      cell.dataset.editing = '0';
      cell.textContent = formatNum(getMatrix()[r][c]);
    };
    input.addEventListener('blur', commit);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { commit(); }
      else if (e.key === 'Escape') {
        input.replaceWith(cell);
        cell.dataset.editing = '0';
        cell.textContent = formatNum(getMatrix()[r][c]);
      }
    });
  }

  return {
    // tokens
    ACCENT, INK, INK_FADE, RULE_DARK, RULE_FAINT, RULE_AXIS,
    ACCENT_SOFT, ACCENT_FAINT, ACCENT_FILL,
    SERIF_LABEL, SERIF_LABEL_SM, MONO_LABEL,
    // math
    mul, norm, normalize, dot, transpose, matmul, matEq, eigen2, svd2,
    // formatting
    formatNum, sub,
    // drawing
    arrow,
    // dom
    attachCanvas, mountMatrixCells,
  };
})();
