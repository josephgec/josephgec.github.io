const {
  CONNECT_DIST,
  PARTICLE_COUNT,
  ATTRACTION_RADIUS,
  DAMPING,
  createParticle,
  updateParticle,
  particleDistance,
  shouldConnect,
  connectionAlpha,
} = require('../lib/particle');

describe('constants', () => {
  test('CONNECT_DIST is 140', () => {
    expect(CONNECT_DIST).toBe(140);
  });

  test('PARTICLE_COUNT is 80', () => {
    expect(PARTICLE_COUNT).toBe(80);
  });

  test('ATTRACTION_RADIUS is 300', () => {
    expect(ATTRACTION_RADIUS).toBe(300);
  });

  test('DAMPING is 0.995', () => {
    expect(DAMPING).toBe(0.995);
  });
});

describe('createParticle', () => {
  test('position is within bounds', () => {
    const p = createParticle(800, 600);
    expect(p.x).toBeGreaterThanOrEqual(0);
    expect(p.x).toBeLessThanOrEqual(800);
    expect(p.y).toBeGreaterThanOrEqual(0);
    expect(p.y).toBeLessThanOrEqual(600);
  });

  test('velocity is within expected range', () => {
    const p = createParticle(800, 600);
    expect(p.vx).toBeGreaterThanOrEqual(-0.3);
    expect(p.vx).toBeLessThanOrEqual(0.3);
    expect(p.vy).toBeGreaterThanOrEqual(-0.3);
    expect(p.vy).toBeLessThanOrEqual(0.3);
  });

  test('radius is between 1 and 3', () => {
    const p = createParticle(800, 600);
    expect(p.r).toBeGreaterThanOrEqual(1);
    expect(p.r).toBeLessThanOrEqual(3);
  });

  test('returns object with all required properties', () => {
    const p = createParticle(100, 100);
    expect(p).toHaveProperty('x');
    expect(p).toHaveProperty('y');
    expect(p).toHaveProperty('vx');
    expect(p).toHaveProperty('vy');
    expect(p).toHaveProperty('r');
  });
});

describe('updateParticle', () => {
  test('applies damping to velocity', () => {
    const p = { x: 400, y: 300, vx: 1, vy: 1, r: 2 };
    const mouse = { x: -9999, y: -9999 };
    updateParticle(p, mouse, 800, 600);
    expect(p.vx).toBeCloseTo(1 * DAMPING, 5);
    expect(p.vy).toBeCloseTo(1 * DAMPING, 5);
  });

  test('moves particle by velocity', () => {
    const p = { x: 100, y: 100, vx: 2, vy: 3, r: 2 };
    const mouse = { x: -9999, y: -9999 };
    updateParticle(p, mouse, 800, 600);
    expect(p.x).toBeCloseTo(100 + 2 * DAMPING, 2);
    expect(p.y).toBeCloseTo(100 + 3 * DAMPING, 2);
  });

  test('wraps particle from left to right', () => {
    const p = { x: -1, y: 300, vx: -0.5, vy: 0, r: 2 };
    const mouse = { x: -9999, y: -9999 };
    updateParticle(p, mouse, 800, 600);
    expect(p.x).toBe(800);
  });

  test('wraps particle from right to left', () => {
    const p = { x: 801, y: 300, vx: 0.5, vy: 0, r: 2 };
    const mouse = { x: -9999, y: -9999 };
    updateParticle(p, mouse, 800, 600);
    expect(p.x).toBe(0);
  });

  test('wraps particle from top to bottom', () => {
    const p = { x: 400, y: -1, vx: 0, vy: -0.5, r: 2 };
    const mouse = { x: -9999, y: -9999 };
    updateParticle(p, mouse, 800, 600);
    expect(p.y).toBe(600);
  });

  test('wraps particle from bottom to top', () => {
    const p = { x: 400, y: 601, vx: 0, vy: 0.5, r: 2 };
    const mouse = { x: -9999, y: -9999 };
    updateParticle(p, mouse, 800, 600);
    expect(p.y).toBe(0);
  });

  test('attracts particle toward nearby mouse', () => {
    const p = { x: 400, y: 300, vx: 0, vy: 0, r: 2 };
    const mouse = { x: 500, y: 300 };
    updateParticle(p, mouse, 800, 600);
    expect(p.vx).toBeGreaterThan(0);
  });

  test('does not attract when mouse is far away', () => {
    const p = { x: 100, y: 100, vx: 0, vy: 0, r: 2 };
    const mouse = { x: 700, y: 500 };
    updateParticle(p, mouse, 800, 600);
    // Only damping applied, velocity stays ~0
    expect(Math.abs(p.vx)).toBeLessThan(0.001);
    expect(Math.abs(p.vy)).toBeLessThan(0.001);
  });

  test('does not attract when mouse distance is <= 1', () => {
    const p = { x: 400, y: 300, vx: 0, vy: 0, r: 2 };
    const mouse = { x: 400.5, y: 300 };
    updateParticle(p, mouse, 800, 600);
    // Mouse is ~0.5 away which is <= 1, no attraction
    expect(Math.abs(p.vx)).toBeLessThan(0.001);
  });

  test('velocity converges toward zero over many updates without mouse', () => {
    const p = { x: 400, y: 300, vx: 10, vy: 10, r: 2 };
    const mouse = { x: -9999, y: -9999 };
    for (let i = 0; i < 2000; i++) {
      updateParticle(p, mouse, 800, 600);
    }
    expect(Math.abs(p.vx)).toBeLessThan(0.01);
    expect(Math.abs(p.vy)).toBeLessThan(0.01);
  });
});

describe('particleDistance', () => {
  test('returns 0 for same position', () => {
    const p = { x: 100, y: 100 };
    expect(particleDistance(p, p)).toBe(0);
  });

  test('returns correct distance for known values', () => {
    const p1 = { x: 0, y: 0 };
    const p2 = { x: 3, y: 4 };
    expect(particleDistance(p1, p2)).toBe(5);
  });

  test('is symmetric', () => {
    const p1 = { x: 10, y: 20 };
    const p2 = { x: 50, y: 80 };
    expect(particleDistance(p1, p2)).toBe(particleDistance(p2, p1));
  });
});

describe('shouldConnect', () => {
  test('returns true for nearby particles', () => {
    const p1 = { x: 100, y: 100 };
    const p2 = { x: 200, y: 100 };
    expect(shouldConnect(p1, p2)).toBe(true);
  });

  test('returns false for distant particles', () => {
    const p1 = { x: 0, y: 0 };
    const p2 = { x: 500, y: 500 };
    expect(shouldConnect(p1, p2)).toBe(false);
  });

  test('returns false at exactly CONNECT_DIST', () => {
    const p1 = { x: 0, y: 0 };
    const p2 = { x: CONNECT_DIST, y: 0 };
    expect(shouldConnect(p1, p2)).toBe(false);
  });

  test('returns true just under CONNECT_DIST', () => {
    const p1 = { x: 0, y: 0 };
    const p2 = { x: CONNECT_DIST - 1, y: 0 };
    expect(shouldConnect(p1, p2)).toBe(true);
  });
});

describe('connectionAlpha', () => {
  test('returns max alpha for overlapping particles', () => {
    const p = { x: 100, y: 100 };
    expect(connectionAlpha(p, p)).toBeCloseTo(0.15);
  });

  test('returns 0 at CONNECT_DIST', () => {
    const p1 = { x: 0, y: 0 };
    const p2 = { x: CONNECT_DIST, y: 0 };
    expect(connectionAlpha(p1, p2)).toBeCloseTo(0);
  });

  test('returns value between 0 and 0.15', () => {
    const p1 = { x: 0, y: 0 };
    const p2 = { x: 70, y: 0 };
    const alpha = connectionAlpha(p1, p2);
    expect(alpha).toBeGreaterThan(0);
    expect(alpha).toBeLessThanOrEqual(0.15);
  });

  test('decreases with distance', () => {
    const p1 = { x: 0, y: 0 };
    const near = { x: 50, y: 0 };
    const far = { x: 120, y: 0 };
    expect(connectionAlpha(p1, near)).toBeGreaterThan(connectionAlpha(p1, far));
  });
});
