/* Vol. XX — The Bottleneck (Autoencoders) */
(function () {
  'use strict';
  const V = window.MLViz;
  const { formatNum, INK_FADE } = V;

  // Five 8×8 binary digits. Just enough variety to demo encoder/decoder structure.
  const SHAPES = {
    cross: [[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0]],
    circle:[[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[1,1,0,0,0,0,1,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,1,0,0,0,0,1,1],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0]],
    square:[[1,1,1,1,1,1,1,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[1,1,1,1,1,1,1,1]],
    triangle:[[0,0,0,1,1,0,0,0],[0,0,1,1,1,1,0,0],[0,0,1,1,1,1,0,0],[0,1,1,1,1,1,1,0],[0,1,1,1,1,1,1,0],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]],
    noisy_circle: null,
  };
  // Generate noisy variant on demand.
  function genNoisy(rngSeed) {
    let s = rngSeed;
    const rnd = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
    const base = SHAPES.circle;
    return base.map((row) => row.map((v) => Math.max(0, Math.min(1, v + (rnd() - 0.5) * 0.6))));
  }

  // We can't actually train an autoencoder here, but we simulate compression by
  // taking the top-k singular components of the input matrix (truncated SVD =
  // best rank-k approximation = what a linear autoencoder converges to).
  function svdLikeReconstruct(matrix, k) {
    // 8×8 matrix. Power iteration would be heavy; instead we deterministically
    // project onto k principal directions of a fixed basis (DCT-like) — the result
    // is visually similar to PCA and conveys "low-rank approximation."
    const N = matrix.length;
    const M = matrix[0].length;
    const flat = matrix.flat();
    // Simple basis: 2D cosines. The first few capture low-frequency structure.
    const basis = [];
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        const b = [];
        for (let r = 0; r < N; r++) {
          for (let c = 0; c < M; c++) {
            b.push(Math.cos((Math.PI * (i + 0.5) * (r + 0.5)) / N) *
                   Math.cos((Math.PI * (j + 0.5) * (c + 0.5)) / M));
          }
        }
        // Normalize
        const norm = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
        basis.push(b.map((v) => v / norm));
      }
    }
    // Take the top k bases (i.e. lowest-frequency); compute coefficients
    const used = basis.slice(0, k);
    const coeffs = used.map((b) => flat.reduce((s, v, i) => s + v * b[i], 0));
    // Reconstruct
    const recon = new Array(flat.length).fill(0);
    used.forEach((b, idx) => {
      const c = coeffs[idx];
      for (let i = 0; i < recon.length; i++) recon[i] += c * b[i];
    });
    const out = [];
    for (let i = 0; i < N; i++) out.push(recon.slice(i * M, (i + 1) * M));
    return { recon: out, coeffs };
  }

  const state = {
    inputKey: 'cross',
    k: 4,
    noiseSeed: 1,
  };

  const inputsEl = document.getElementById('inputs');
  const kSlider = document.getElementById('k-slider');
  const kLabel = document.getElementById('k-label');
  const latentSlidersEl = document.getElementById('latent-sliders');
  const readoutEl = document.getElementById('readout');
  const canvas = document.getElementById('canvas');

  ['cross', 'circle', 'square', 'triangle', 'noisy_circle'].forEach((k) => {
    const b = document.createElement('button');
    b.className = 'ml-preset';
    b.textContent = k.replace('_', ' ');
    b.dataset.input = k;
    b.addEventListener('click', () => { state.inputKey = k; render(); });
    inputsEl.appendChild(b);
  });

  kSlider.addEventListener('input', (e) => {
    state.k = +e.target.value;
    kLabel.textContent = `k = ${state.k}`;
    render();
  });

  function getInput() {
    if (state.inputKey === 'noisy_circle') return genNoisy(state.noiseSeed);
    return SHAPES[state.inputKey];
  }

  // ───────── Canvas ─────────
  const canvasCtl = V.attachCanvas(canvas, draw, { square: false });

  function draw(ctx, size) {
    const input = getInput();
    const { recon, coeffs } = svdLikeReconstruct(input, state.k);

    function drawGrid(map, x0, y0, cell, max = 1) {
      const N = map.length;
      const M = map[0].length;
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < M; j++) {
          const v = Math.max(-1, Math.min(2, map[i][j]));
          const t = Math.max(0, Math.min(1, v / max));
          const r = Math.round(247 - t * 200);
          const g = Math.round(244 - t * 200);
          const b = Math.round(236 - t * 220);
          ctx.fillStyle = `rgb(${r},${g},${b})`;
          ctx.fillRect(x0 + j * cell, y0 + i * cell, cell + 0.5, cell + 0.5);
        }
      }
      ctx.strokeStyle = 'rgba(38,35,32,0.30)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x0, y0, M * cell, N * cell);
    }

    const cell = (size.h - 80) / 8;
    const mapW = cell * 8;
    const yOff = 30;

    // 1. Input
    drawGrid(input, 24, yOff, cell);
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('input · 8×8', 24 + mapW / 2, yOff + mapW + 18);

    // 2. Encoder (sketch)
    const arrowX1 = 24 + mapW + 14;
    const arrowX2 = arrowX1 + 70;
    ctx.strokeStyle = 'rgba(38,35,32,0.4)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(arrowX1, yOff + mapW / 2); ctx.lineTo(arrowX2, yOff + mapW / 2);
    ctx.stroke();
    ctx.fillStyle = INK_FADE;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('encoder', (arrowX1 + arrowX2) / 2, yOff + mapW / 2 - 10);

    // 3. Latent vector
    const latentX = arrowX2 + 20;
    const latentW = 60;
    const latentCell = mapW / Math.max(state.k, 4);
    const latentYOff = yOff + (mapW - latentCell * state.k) / 2;
    coeffs.slice(0, state.k).forEach((c, i) => {
      const max = Math.max(...coeffs.map(Math.abs)) + 1e-6;
      const intensity = Math.min(1, Math.abs(c) / max);
      ctx.fillStyle = c >= 0
        ? `rgba(122,31,36,${0.15 + intensity * 0.7})`
        : `rgba(38,35,32,${0.15 + intensity * 0.7})`;
      ctx.fillRect(latentX, latentYOff + i * latentCell, latentW, latentCell);
    });
    ctx.strokeStyle = 'rgba(38,35,32,0.30)';
    ctx.lineWidth = 1;
    ctx.strokeRect(latentX, latentYOff, latentW, state.k * latentCell);
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText(`latent · ${state.k} dims`, latentX + latentW / 2, yOff + mapW + 18);

    // 4. Decoder
    const arrowX3 = latentX + latentW + 14;
    const arrowX4 = arrowX3 + 70;
    ctx.strokeStyle = 'rgba(38,35,32,0.4)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(arrowX3, yOff + mapW / 2); ctx.lineTo(arrowX4, yOff + mapW / 2);
    ctx.stroke();
    ctx.fillStyle = INK_FADE;
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('decoder', (arrowX3 + arrowX4) / 2, yOff + mapW / 2 - 10);

    // 5. Reconstruction
    const reconX = arrowX4 + 20;
    drawGrid(recon, reconX, yOff, cell);
    ctx.fillStyle = INK_FADE;
    ctx.font = 'italic 12px "Source Serif 4", Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('reconstruction', reconX + mapW / 2, yOff + mapW + 18);
  }

  function render() {
    Array.from(inputsEl.children).forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.input === state.inputKey);
    });
    kLabel.textContent = `k = ${state.k}`;
    kSlider.value = state.k;

    // MSE between input and reconstruction
    const input = getInput();
    const { recon, coeffs } = svdLikeReconstruct(input, state.k);
    let mse = 0, count = 0;
    for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) {
      mse += (input[i][j] - recon[i][j]) ** 2; count++;
    }
    mse /= count;

    readoutEl.innerHTML = `
      <div class="row"><span>latent dims</span><b>${state.k}</b></div>
      <div class="row"><span>input pixels</span><b>64</b></div>
      <div class="row"><span>compression</span><b>${(64 / state.k).toFixed(1)}×</b></div>
      <div class="row"><span>reconstruction MSE</span><b>${formatNum(mse)}</b></div>`;

    // Show latent values as readouts
    let html = '<div style="font-family:var(--mono); font-size:11px; color:var(--ml-ink-fade); margin-bottom:6px">latent values</div>';
    coeffs.slice(0, state.k).forEach((c, i) => {
      const max = Math.max(...coeffs.map(Math.abs)) + 1e-6;
      html += `<div style="display:flex; justify-content:space-between; font-family:var(--mono); font-size:11px; padding:2px 0"><span style="color:var(--muted)">z${i}</span><b>${formatNum(c / max * 1.5)}</b></div>`;
    });
    latentSlidersEl.innerHTML = html;

    canvasCtl.redraw();
  }

  render();
})();
