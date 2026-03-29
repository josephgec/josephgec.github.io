module.exports = {
  projects: [
    {
      displayName: 'unit',
      testMatch: ['<rootDir>/__tests__/animations.test.js', '<rootDir>/__tests__/particle.test.js', '<rootDir>/__tests__/scoring.test.js'],
      testEnvironment: 'node',
    },
    {
      displayName: 'dom',
      testMatch: ['<rootDir>/__tests__/html.test.js', '<rootDir>/__tests__/accessibility.test.js', '<rootDir>/__tests__/dom-interactions.test.js'],
      testEnvironment: 'jsdom',
    },
    {
      displayName: 'seo',
      testMatch: ['<rootDir>/__tests__/seo.test.js'],
      testEnvironment: 'node',
    },
  ],
  collectCoverageFrom: [
    'lib/**/*.js',
  ],
};
