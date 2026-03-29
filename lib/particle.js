/**
 * Particle Swarm simulation extracted from inline JS for testability.
 * The inline <script> in index.html contains the same logic.
 */

const CONNECT_DIST = 140;
const PARTICLE_COUNT = 80;
const ATTRACTION_RADIUS = 300;
const ATTRACTION_FORCE = 0.0003;
const DAMPING = 0.995;

function createParticle(W, H) {
  return {
    x: Math.random() * W,
    y: Math.random() * H,
    vx: (Math.random() - 0.5) * 0.6,
    vy: (Math.random() - 0.5) * 0.6,
    r: Math.random() * 2 + 1,
  };
}

function updateParticle(p, mouse, W, H) {
  const dx = mouse.x - p.x;
  const dy = mouse.y - p.y;
  const dist = Math.sqrt(dx * dx + dy * dy);
  if (dist < ATTRACTION_RADIUS && dist > 1) {
    const force = ATTRACTION_FORCE * (ATTRACTION_RADIUS - dist);
    p.vx += (dx / dist) * force;
    p.vy += (dy / dist) * force;
  }
  p.vx *= DAMPING;
  p.vy *= DAMPING;
  p.x += p.vx;
  p.y += p.vy;
  if (p.x < 0) p.x = W;
  if (p.x > W) p.x = 0;
  if (p.y < 0) p.y = H;
  if (p.y > H) p.y = 0;
}

function particleDistance(p1, p2) {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function shouldConnect(p1, p2) {
  return particleDistance(p1, p2) < CONNECT_DIST;
}

function connectionAlpha(p1, p2) {
  const d = particleDistance(p1, p2);
  return (1 - d / CONNECT_DIST) * 0.15;
}

module.exports = {
  CONNECT_DIST,
  PARTICLE_COUNT,
  ATTRACTION_RADIUS,
  ATTRACTION_FORCE,
  DAMPING,
  createParticle,
  updateParticle,
  particleDistance,
  shouldConnect,
  connectionAlpha,
};
