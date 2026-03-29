const {
  cubicEaseOut,
  calcScrollProgress,
  calcAnimationProgress,
  calcCounterValue,
  formatCounterText,
} = require('../lib/animations');

describe('cubicEaseOut', () => {
  test('returns 0 at t=0', () => {
    expect(cubicEaseOut(0)).toBe(0);
  });

  test('returns 1 at t=1', () => {
    expect(cubicEaseOut(1)).toBe(1);
  });

  test('returns value between 0 and 1 for midpoint', () => {
    const val = cubicEaseOut(0.5);
    expect(val).toBeGreaterThan(0);
    expect(val).toBeLessThan(1);
  });

  test('is monotonically increasing', () => {
    let prev = 0;
    for (let t = 0.1; t <= 1; t += 0.1) {
      const val = cubicEaseOut(t);
      expect(val).toBeGreaterThanOrEqual(prev);
      prev = val;
    }
  });

  test('eases out (faster at start, slower at end)', () => {
    const firstHalf = cubicEaseOut(0.5) - cubicEaseOut(0);
    const secondHalf = cubicEaseOut(1) - cubicEaseOut(0.5);
    expect(firstHalf).toBeGreaterThan(secondHalf);
  });

  test('matches expected formula at t=0.25', () => {
    const expected = 1 - Math.pow(1 - 0.25, 3);
    expect(cubicEaseOut(0.25)).toBeCloseTo(expected);
  });
});

describe('calcScrollProgress', () => {
  test('returns 0 at top of page', () => {
    expect(calcScrollProgress(0, 2000, 800)).toBe(0);
  });

  test('returns 100 at bottom of page', () => {
    expect(calcScrollProgress(1200, 2000, 800)).toBe(100);
  });

  test('returns 50 at midpoint', () => {
    expect(calcScrollProgress(600, 2000, 800)).toBe(50);
  });

  test('handles small page (scrollHeight equals clientHeight)', () => {
    const result = calcScrollProgress(0, 800, 800);
    expect(result).toBeNaN();
  });

  test('returns correct percentage for partial scroll', () => {
    const result = calcScrollProgress(300, 2000, 800);
    expect(result).toBe(25);
  });
});

describe('calcAnimationProgress', () => {
  test('returns 0 at start', () => {
    expect(calcAnimationProgress(1000, 1000, 1200)).toBe(0);
  });

  test('returns 1 at end', () => {
    expect(calcAnimationProgress(2200, 1000, 1200)).toBe(1);
  });

  test('clamps to 1 when past duration', () => {
    expect(calcAnimationProgress(5000, 1000, 1200)).toBe(1);
  });

  test('returns 0.5 at halfway', () => {
    expect(calcAnimationProgress(1600, 1000, 1200)).toBe(0.5);
  });

  test('never exceeds 1', () => {
    for (let t = 0; t < 10000; t += 500) {
      expect(calcAnimationProgress(t, 0, 1200)).toBeLessThanOrEqual(1);
    }
  });
});

describe('calcCounterValue', () => {
  test('returns 0 at progress=0', () => {
    expect(calcCounterValue(0, 21)).toBe(0);
  });

  test('returns target at progress=1', () => {
    expect(calcCounterValue(1, 21)).toBe(21);
  });

  test('returns integer values', () => {
    const val = calcCounterValue(0.5, 21);
    expect(Number.isInteger(val)).toBe(true);
  });

  test('handles target=1', () => {
    expect(calcCounterValue(1, 1)).toBe(1);
    expect(calcCounterValue(0, 1)).toBe(0);
  });

  test('handles large targets', () => {
    expect(calcCounterValue(1, 1000)).toBe(1000);
  });

  test('intermediate value is between 0 and target', () => {
    const val = calcCounterValue(0.5, 27);
    expect(val).toBeGreaterThan(0);
    expect(val).toBeLessThanOrEqual(27);
  });
});

describe('formatCounterText', () => {
  test('appends suffix when progress >= 1', () => {
    expect(formatCounterText(10, 1, 'B+')).toBe('10B+');
  });

  test('omits suffix when progress < 1', () => {
    expect(formatCounterText(5, 0.5, 'B+')).toBe('5');
  });

  test('handles empty suffix', () => {
    expect(formatCounterText(21, 1, '')).toBe('21');
  });

  test('works with zero value', () => {
    expect(formatCounterText(0, 0, 'B+')).toBe('0');
  });

  test('appends suffix at exactly progress=1', () => {
    expect(formatCounterText(10, 1.0, 'B+')).toBe('10B+');
  });
});
