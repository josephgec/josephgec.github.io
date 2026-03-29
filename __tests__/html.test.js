const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');

let doc;
beforeAll(() => {
  doc = new DOMParser().parseFromString(html, 'text/html');
});

describe('Primary SEO meta tags', () => {
  test('has a title tag', () => {
    const title = doc.querySelector('title');
    expect(title).not.toBeNull();
    expect(title.textContent).toContain('Joseph Thomas');
  });

  test('has meta description', () => {
    const desc = doc.querySelector('meta[name="description"]');
    expect(desc).not.toBeNull();
    expect(desc.getAttribute('content').length).toBeGreaterThan(50);
    expect(desc.getAttribute('content').length).toBeLessThanOrEqual(160);
  });

  test('has canonical URL', () => {
    const canonical = doc.querySelector('link[rel="canonical"]');
    expect(canonical).not.toBeNull();
    expect(canonical.getAttribute('href')).toBe('https://jthomas.site/');
  });

  test('has robots meta tag', () => {
    const robots = doc.querySelector('meta[name="robots"]');
    expect(robots).not.toBeNull();
    expect(robots.getAttribute('content')).toContain('index');
  });

  test('has theme-color', () => {
    const theme = doc.querySelector('meta[name="theme-color"]');
    expect(theme).not.toBeNull();
    expect(theme.getAttribute('content')).toMatch(/^#[0-9a-f]{6}$/i);
  });

  test('has lang attribute on html', () => {
    const htmlEl = doc.querySelector('html');
    expect(htmlEl.getAttribute('lang')).toBe('en');
  });

  test('has charset meta', () => {
    const charset = doc.querySelector('meta[charset]');
    expect(charset).not.toBeNull();
    expect(charset.getAttribute('charset').toLowerCase()).toBe('utf-8');
  });

  test('has viewport meta', () => {
    const viewport = doc.querySelector('meta[name="viewport"]');
    expect(viewport).not.toBeNull();
    expect(viewport.getAttribute('content')).toContain('width=device-width');
  });
});

describe('Open Graph meta tags', () => {
  test('has og:type', () => {
    const og = doc.querySelector('meta[property="og:type"]');
    expect(og).not.toBeNull();
    expect(og.getAttribute('content')).toBe('profile');
  });

  test('has og:url', () => {
    const og = doc.querySelector('meta[property="og:url"]');
    expect(og).not.toBeNull();
    expect(og.getAttribute('content')).toBe('https://jthomas.site/');
  });

  test('has og:title', () => {
    const og = doc.querySelector('meta[property="og:title"]');
    expect(og).not.toBeNull();
    expect(og.getAttribute('content').length).toBeGreaterThan(0);
  });

  test('has og:description', () => {
    const og = doc.querySelector('meta[property="og:description"]');
    expect(og).not.toBeNull();
  });

  test('has og:image with absolute URL', () => {
    const og = doc.querySelector('meta[property="og:image"]');
    expect(og).not.toBeNull();
    expect(og.getAttribute('content')).toMatch(/^https:\/\//);
  });

  test('has og:locale without backslash escapes', () => {
    const og = doc.querySelector('meta[property="og:locale"]');
    expect(og).not.toBeNull();
    const content = og.getAttribute('content');
    expect(content).toBe('en_US');
    expect(content).not.toContain('\\');
  });

  test('has og:site_name without backslash escapes', () => {
    const og = doc.querySelector('meta[property="og:site_name"]');
    expect(og).not.toBeNull();
    expect(og.getAttribute('content')).toBe('Joseph Thomas');
  });

  test('has profile:first_name without backslash escapes', () => {
    const og = doc.querySelector('meta[property="profile:first_name"]');
    expect(og).not.toBeNull();
    expect(og.getAttribute('content')).toBe('Joseph');
  });

  test('has profile:last_name without backslash escapes', () => {
    const og = doc.querySelector('meta[property="profile:last_name"]');
    expect(og).not.toBeNull();
    expect(og.getAttribute('content')).toBe('Thomas');
  });

  test('og property attributes have no backslash-escaped underscores', () => {
    const ogTags = doc.querySelectorAll('meta[property]');
    ogTags.forEach(tag => {
      const prop = tag.getAttribute('property');
      expect(prop).not.toContain('\\');
    });
  });
});

describe('Twitter Card meta tags', () => {
  test('has twitter:card', () => {
    const tc = doc.querySelector('meta[name="twitter:card"]');
    expect(tc).not.toBeNull();
    expect(tc.getAttribute('content')).toBe('summary');
  });

  test('has twitter:title', () => {
    const tc = doc.querySelector('meta[name="twitter:title"]');
    expect(tc).not.toBeNull();
  });

  test('has twitter:description', () => {
    const tc = doc.querySelector('meta[name="twitter:description"]');
    expect(tc).not.toBeNull();
  });

  test('has twitter:image', () => {
    const tc = doc.querySelector('meta[name="twitter:image"]');
    expect(tc).not.toBeNull();
    expect(tc.getAttribute('content')).toMatch(/^https:\/\//);
  });
});

describe('JSON-LD Structured Data', () => {
  let schema;
  beforeAll(() => {
    const script = doc.querySelector('script[type="application/ld+json"]');
    schema = JSON.parse(script.textContent);
  });

  test('has @context schema.org', () => {
    expect(schema['@context']).toBe('https://schema.org');
  });

  test('has @type Person', () => {
    expect(schema['@type']).toBe('Person');
  });

  test('has name', () => {
    expect(schema.name).toBe('Joseph Thomas');
  });

  test('has jobTitle', () => {
    expect(schema.jobTitle).toBeTruthy();
  });

  test('has worksFor Google', () => {
    expect(schema.worksFor.name).toBe('Google');
  });

  test('sameAs includes Google Scholar', () => {
    expect(schema.sameAs).toEqual(
      expect.arrayContaining([
        expect.stringContaining('scholar.google.com'),
      ])
    );
  });

  test('sameAs includes GitHub', () => {
    expect(schema.sameAs).toEqual(
      expect.arrayContaining([
        expect.stringContaining('github.com'),
      ])
    );
  });

  test('sameAs includes LinkedIn', () => {
    expect(schema.sameAs).toEqual(
      expect.arrayContaining([
        expect.stringContaining('linkedin.com'),
      ])
    );
  });

  test('has alumniOf array', () => {
    expect(Array.isArray(schema.alumniOf)).toBe(true);
    expect(schema.alumniOf.length).toBeGreaterThanOrEqual(1);
  });

  test('has knowsAbout array', () => {
    expect(Array.isArray(schema.knowsAbout)).toBe(true);
    expect(schema.knowsAbout.length).toBeGreaterThan(5);
  });

  test('has speakable specification', () => {
    expect(schema.speakable).toBeDefined();
    expect(schema.speakable['@type']).toBe('SpeakableSpecification');
  });
});

describe('Hero image attributes', () => {
  test('has alt text', () => {
    const img = doc.querySelector('img.hero-avatar');
    expect(img).not.toBeNull();
    expect(img.getAttribute('alt')).toBeTruthy();
  });

  test('has loading attribute', () => {
    const img = doc.querySelector('img.hero-avatar');
    expect(img.getAttribute('loading')).toBe('lazy');
  });

  test('has width and height', () => {
    const img = doc.querySelector('img.hero-avatar');
    expect(img.getAttribute('width')).toBe('190');
    expect(img.getAttribute('height')).toBe('190');
  });
});

describe('Stats counter noscript fallback', () => {
  test('counter elements show real values by default (not 0)', () => {
    const counters = doc.querySelectorAll('[data-target]');
    expect(counters.length).toBeGreaterThan(0);
    counters.forEach(el => {
      const target = el.getAttribute('data-target');
      const suffix = el.getAttribute('data-suffix') || '';
      const expected = target + suffix;
      expect(el.textContent).toBe(expected);
    });
  });
});

describe('Font preload', () => {
  test('has preload link for Syne font', () => {
    const preload = doc.querySelector('link[rel="preload"][as="style"]');
    expect(preload).not.toBeNull();
    expect(preload.getAttribute('href')).toContain('Syne');
  });
});

describe('Navigation', () => {
  test('nav has aria-label', () => {
    const nav = doc.querySelector('nav');
    expect(nav).not.toBeNull();
    expect(nav.getAttribute('aria-label')).toBeTruthy();
  });

  test('nav links have aria-label attributes', () => {
    const links = doc.querySelectorAll('.nav-links a[href^="#"]');
    expect(links.length).toBeGreaterThanOrEqual(4);
    links.forEach(link => {
      expect(link.getAttribute('aria-label')).toBeTruthy();
    });
  });

  test('nav links point to valid section IDs', () => {
    const links = doc.querySelectorAll('.nav-links a[href^="#"]');
    links.forEach(link => {
      const id = link.getAttribute('href').slice(1);
      const section = doc.getElementById(id);
      expect(section).not.toBeNull();
    });
  });
});

describe('Semantic HTML structure', () => {
  test('has <header> element', () => {
    expect(doc.querySelector('header')).not.toBeNull();
  });

  test('has <main> element', () => {
    expect(doc.querySelector('main')).not.toBeNull();
  });

  test('has <footer> element', () => {
    expect(doc.querySelector('footer')).not.toBeNull();
  });

  test('has <nav> element', () => {
    expect(doc.querySelector('nav')).not.toBeNull();
  });

  test('has exactly one <h1>', () => {
    const h1s = doc.querySelectorAll('h1');
    expect(h1s.length).toBe(1);
  });
});
