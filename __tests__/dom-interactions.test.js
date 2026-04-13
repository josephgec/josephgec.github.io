const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');

let doc;
beforeAll(() => {
  doc = new DOMParser().parseFromString(html, 'text/html');
});

describe('Navigation DOM structure', () => {
  test('nav has mainNav id', () => {
    const nav = doc.getElementById('mainNav');
    expect(nav).not.toBeNull();
    expect(nav.tagName.toLowerCase()).toBe('nav');
  });

  test('nav contains nav-links list', () => {
    const list = doc.querySelector('.nav-links');
    expect(list).not.toBeNull();
    expect(list.tagName.toLowerCase()).toBe('ul');
  });

  test('nav-links contain 5 section links', () => {
    const links = doc.querySelectorAll('.nav-links a[href^="#"]');
    expect(links.length).toBe(5);
  });

  test('each nav link targets an existing section', () => {
    const links = doc.querySelectorAll('.nav-links a[href^="#"]');
    const ids = Array.from(links).map(l => l.getAttribute('href').slice(1));
    expect(ids).toEqual(['experience', 'education', 'publications', 'projects', 'writing']);
    ids.forEach(id => {
      expect(doc.getElementById(id)).not.toBeNull();
    });
  });
});

describe('Hero section', () => {
  test('has hero element with id', () => {
    const hero = doc.getElementById('hero');
    expect(hero).not.toBeNull();
  });

  test('has hero headline', () => {
    const headline = doc.querySelector('.hero-headline');
    expect(headline).not.toBeNull();
    expect(headline.textContent.length).toBeGreaterThan(0);
  });

  test('has hero tagline', () => {
    const tagline = doc.querySelector('.hero-tagline');
    expect(tagline).not.toBeNull();
    expect(tagline.textContent.length).toBeGreaterThan(0);
  });

  test('has swarm canvas', () => {
    const canvas = doc.getElementById('swarmCanvas');
    expect(canvas).not.toBeNull();
    expect(canvas.tagName.toLowerCase()).toBe('canvas');
  });

  test('has hero action links', () => {
    const actions = doc.querySelector('.hero-actions');
    expect(actions).not.toBeNull();
    const links = actions.querySelectorAll('a');
    expect(links.length).toBeGreaterThanOrEqual(1);
  });
});

describe('Stats ribbon', () => {
  test('has statsRibbon element', () => {
    const ribbon = doc.getElementById('statsRibbon');
    expect(ribbon).not.toBeNull();
  });

  test('has 5 stat items', () => {
    const items = doc.querySelectorAll('.stat-item');
    expect(items.length).toBe(5);
  });

  test('all stat items have data-target attributes', () => {
    const counters = doc.querySelectorAll('.stat-number[data-target]');
    expect(counters.length).toBe(5);
  });

  test('stat targets match expected values', () => {
    const counters = doc.querySelectorAll('.stat-number[data-target]');
    const targets = Array.from(counters).map(el => el.getAttribute('data-target'));
    expect(targets).toEqual(['21', '27', '1', '4', '10']);
  });

  test('stat labels exist', () => {
    const labels = doc.querySelectorAll('.stat-label');
    expect(labels.length).toBe(5);
    const texts = Array.from(labels).map(el => el.textContent);
    expect(texts).toContain('Google Awards');
    expect(texts).toContain('Citations');
    expect(texts).toContain('US Patent');
    expect(texts).toContain('Publications');
    expect(texts).toContain('Events / Day');
  });
});

describe('Scroll progress bar', () => {
  test('has scrollProgress element', () => {
    const bar = doc.getElementById('scrollProgress');
    expect(bar).not.toBeNull();
  });
});

describe('Section structure', () => {
  const expectedSections = ['experience', 'education', 'publications', 'projects'];

  expectedSections.forEach(id => {
    test(`section "${id}" exists`, () => {
      expect(doc.getElementById(id)).not.toBeNull();
    });
  });

  test('experience section has timeline items', () => {
    const items = doc.querySelectorAll('#experience .tl-entry');
    expect(items.length).toBeGreaterThanOrEqual(1);
  });

  test('education section has education cards', () => {
    const cards = doc.querySelectorAll('#education .edu-card');
    expect(cards.length).toBeGreaterThanOrEqual(3);
  });

  test('publications section has pub cards', () => {
    const cards = doc.querySelectorAll('#publications .pub-card');
    expect(cards.length).toBeGreaterThanOrEqual(1);
  });

  test('projects section has project cards', () => {
    const cards = doc.querySelectorAll('#projects .proj-card');
    expect(cards.length).toBeGreaterThanOrEqual(1);
  });
});

describe('Footer', () => {
  test('has footer element', () => {
    const footer = doc.querySelector('footer');
    expect(footer).not.toBeNull();
  });

  test('footer has social links', () => {
    const footer = doc.querySelector('footer');
    const links = footer.querySelectorAll('a[href]');
    expect(links.length).toBeGreaterThanOrEqual(1);
  });
});

describe('Resource hints', () => {
  test('has preconnect for Google Fonts', () => {
    const link = doc.querySelector('link[rel="preconnect"][href*="fonts.googleapis.com"]');
    expect(link).not.toBeNull();
  });

  test('has preconnect for gstatic', () => {
    const link = doc.querySelector('link[rel="preconnect"][href*="fonts.gstatic.com"]');
    expect(link).not.toBeNull();
  });

  test('has dns-prefetch for CDN resources', () => {
    const prefetches = doc.querySelectorAll('link[rel="dns-prefetch"]');
    expect(prefetches.length).toBeGreaterThanOrEqual(2);
  });
});

describe('Skills section', () => {
  test('has skill tags with icons', () => {
    const tags = doc.querySelectorAll('.tag img');
    expect(tags.length).toBeGreaterThanOrEqual(5);
  });

  test('all skill icons have alt text', () => {
    const icons = doc.querySelectorAll('.tag img');
    icons.forEach(img => {
      expect(img.getAttribute('alt')).toBeTruthy();
    });
  });
});
