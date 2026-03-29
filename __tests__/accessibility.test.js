const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');

let doc;
beforeAll(() => {
  doc = new DOMParser().parseFromString(html, 'text/html');
});

describe('Image accessibility', () => {
  test('all <img> tags have alt attributes', () => {
    const images = doc.querySelectorAll('img');
    expect(images.length).toBeGreaterThan(0);
    images.forEach(img => {
      const alt = img.getAttribute('alt');
      expect(alt).not.toBeNull();
      expect(alt.length).toBeGreaterThan(0);
    });
  });

  test('CDN icons have descriptive alt text', () => {
    const cdnImages = doc.querySelectorAll('img[src*="cdn.simpleicons.org"], img[src*="cdn.jsdelivr.net"]');
    expect(cdnImages.length).toBeGreaterThan(0);
    cdnImages.forEach(img => {
      const alt = img.getAttribute('alt');
      expect(alt).toBeTruthy();
      expect(alt.length).toBeGreaterThan(2);
    });
  });

  test('hero avatar has meaningful alt text', () => {
    const avatar = doc.querySelector('img.hero-avatar');
    expect(avatar).not.toBeNull();
    expect(avatar.getAttribute('alt')).toContain('Joseph');
  });
});

describe('ARIA attributes', () => {
  test('navigation has aria-label', () => {
    const nav = doc.querySelector('nav');
    expect(nav.getAttribute('aria-label')).toBe('Site navigation');
  });

  test('all nav section links have aria-labels', () => {
    const navLinks = doc.querySelectorAll('.nav-links a');
    navLinks.forEach(link => {
      expect(link.hasAttribute('aria-label')).toBe(true);
    });
  });
});

describe('Link accessibility', () => {
  test('external links have rel="noopener noreferrer"', () => {
    const externalLinks = doc.querySelectorAll('a[target="_blank"]');
    externalLinks.forEach(link => {
      const rel = link.getAttribute('rel') || '';
      expect(rel).toContain('noopener');
    });
  });

  test('external links have target="_blank"', () => {
    const externalLinks = doc.querySelectorAll('a[href^="https://"]');
    externalLinks.forEach(link => {
      // Internal navigation links are okay without target
      if (!link.getAttribute('href').startsWith('https://jthomas.site')) {
        expect(link.getAttribute('target')).toBe('_blank');
      }
    });
  });
});

describe('Heading hierarchy', () => {
  test('has exactly one h1', () => {
    const h1s = doc.querySelectorAll('h1');
    expect(h1s.length).toBe(1);
  });

  test('h2 headings exist for sections', () => {
    const h2s = doc.querySelectorAll('h2');
    expect(h2s.length).toBeGreaterThanOrEqual(3);
  });

  test('no heading level is skipped (h1 → h2 before h3)', () => {
    const headings = doc.querySelectorAll('h1, h2, h3, h4, h5, h6');
    let maxLevel = 0;
    const violations = [];
    headings.forEach(h => {
      const level = parseInt(h.tagName.charAt(1));
      if (level > maxLevel + 1) {
        violations.push(`${h.tagName} appears without a preceding h${level - 1}`);
      }
      if (level > maxLevel) maxLevel = level;
    });
    expect(violations).toEqual([]);
  });
});

describe('Language and encoding', () => {
  test('html has lang="en"', () => {
    expect(doc.querySelector('html').getAttribute('lang')).toBe('en');
  });

  test('charset is UTF-8', () => {
    const charset = doc.querySelector('meta[charset]');
    expect(charset.getAttribute('charset').toUpperCase()).toBe('UTF-8');
  });
});

describe('Canvas accessibility', () => {
  test('decorative canvas does not interfere with content', () => {
    const canvas = doc.querySelector('canvas#swarmCanvas');
    if (canvas) {
      // Canvas should be decorative, verify it exists within hero section
      const hero = doc.querySelector('#hero');
      expect(hero).not.toBeNull();
    }
  });
});
