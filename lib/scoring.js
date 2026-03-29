/**
 * Lighthouse scoring logic extracted from eval.js for testability.
 */

function calcCategoryScore(rawScore) {
  return Math.round(rawScore * 100);
}

function calcWeightedScore(seo, perf, a11y) {
  return (seo * 0.50 + perf * 0.30 + a11y * 0.20).toFixed(2);
}

function evaluateLighthouse(categories) {
  const seo = calcCategoryScore(categories.seo.score);
  const perf = calcCategoryScore(categories.performance.score);
  const a11y = calcCategoryScore(categories.accessibility.score);
  const score = calcWeightedScore(seo, perf, a11y);
  return { score, seo, perf, a11y };
}

module.exports = {
  calcCategoryScore,
  calcWeightedScore,
  evaluateLighthouse,
};
