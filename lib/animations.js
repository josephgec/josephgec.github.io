/**
 * Pure animation utilities extracted from inline JS for testability.
 * The inline <script> in index.html contains the same logic.
 */

function cubicEaseOut(t) {
  return 1 - Math.pow(1 - t, 3);
}

function calcScrollProgress(scrollTop, scrollHeight, clientHeight) {
  return (scrollTop / (scrollHeight - clientHeight)) * 100;
}

function calcAnimationProgress(now, start, duration) {
  return Math.min((now - start) / duration, 1);
}

function calcCounterValue(progress, target) {
  const ease = cubicEaseOut(progress);
  return Math.round(ease * target);
}

function formatCounterText(value, progress, suffix) {
  return value + (progress >= 1 ? suffix : '');
}

module.exports = {
  cubicEaseOut,
  calcScrollProgress,
  calcAnimationProgress,
  calcCounterValue,
  formatCounterText,
};
