const {
  calcCategoryScore,
  calcWeightedScore,
  evaluateLighthouse,
} = require('../lib/scoring');

describe('calcCategoryScore', () => {
  test('converts 1.0 to 100', () => {
    expect(calcCategoryScore(1.0)).toBe(100);
  });

  test('converts 0 to 0', () => {
    expect(calcCategoryScore(0)).toBe(0);
  });

  test('converts 0.95 to 95', () => {
    expect(calcCategoryScore(0.95)).toBe(95);
  });

  test('rounds fractional scores', () => {
    expect(calcCategoryScore(0.876)).toBe(88);
  });

  test('rounds 0.5 up', () => {
    expect(calcCategoryScore(0.005)).toBe(1);
  });
});

describe('calcWeightedScore', () => {
  test('returns 100.00 for perfect scores', () => {
    expect(calcWeightedScore(100, 100, 100)).toBe('100.00');
  });

  test('returns 0.00 for zero scores', () => {
    expect(calcWeightedScore(0, 0, 0)).toBe('0.00');
  });

  test('applies correct weights (50% SEO, 30% perf, 20% a11y)', () => {
    // SEO=100, perf=0, a11y=0 → 50
    expect(calcWeightedScore(100, 0, 0)).toBe('50.00');
    // SEO=0, perf=100, a11y=0 → 30
    expect(calcWeightedScore(0, 100, 0)).toBe('30.00');
    // SEO=0, perf=0, a11y=100 → 20
    expect(calcWeightedScore(0, 0, 100)).toBe('20.00');
  });

  test('returns string with two decimal places', () => {
    const result = calcWeightedScore(90, 80, 70);
    expect(result).toMatch(/^\d+\.\d{2}$/);
  });

  test('calculates example correctly', () => {
    // 90*0.5 + 80*0.3 + 70*0.2 = 45 + 24 + 14 = 83
    expect(calcWeightedScore(90, 80, 70)).toBe('83.00');
  });
});

describe('evaluateLighthouse', () => {
  test('evaluates perfect Lighthouse scores', () => {
    const categories = {
      seo: { score: 1.0 },
      performance: { score: 1.0 },
      accessibility: { score: 1.0 },
    };
    const result = evaluateLighthouse(categories);
    expect(result.seo).toBe(100);
    expect(result.perf).toBe(100);
    expect(result.a11y).toBe(100);
    expect(result.score).toBe('100.00');
  });

  test('evaluates mixed scores', () => {
    const categories = {
      seo: { score: 0.92 },
      performance: { score: 0.85 },
      accessibility: { score: 0.78 },
    };
    const result = evaluateLighthouse(categories);
    expect(result.seo).toBe(92);
    expect(result.perf).toBe(85);
    expect(result.a11y).toBe(78);
    // 92*0.5 + 85*0.3 + 78*0.2 = 46 + 25.5 + 15.6 = 87.10
    expect(result.score).toBe('87.10');
  });

  test('evaluates zero scores', () => {
    const categories = {
      seo: { score: 0 },
      performance: { score: 0 },
      accessibility: { score: 0 },
    };
    const result = evaluateLighthouse(categories);
    expect(result.score).toBe('0.00');
  });

  test('returns all expected keys', () => {
    const categories = {
      seo: { score: 0.5 },
      performance: { score: 0.5 },
      accessibility: { score: 0.5 },
    };
    const result = evaluateLighthouse(categories);
    expect(result).toHaveProperty('score');
    expect(result).toHaveProperty('seo');
    expect(result).toHaveProperty('perf');
    expect(result).toHaveProperty('a11y');
  });
});
