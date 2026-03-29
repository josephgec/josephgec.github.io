const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');

describe('robots.txt', () => {
  const content = fs.readFileSync(path.join(root, 'robots.txt'), 'utf8');

  test('has User-agent directive', () => {
    expect(content).toContain('User-agent: *');
  });

  test('has Disallow directive (empty = allow all)', () => {
    expect(content).toMatch(/^Disallow:\s*$/m);
  });

  test('has Sitemap pointing to canonical domain', () => {
    expect(content).toContain('Sitemap: https://jthomas.site/sitemap.xml');
  });

  test('does not reference GitHub Pages origin', () => {
    expect(content).not.toContain('josephgec.github.io');
  });
});

describe('sitemap.xml', () => {
  const content = fs.readFileSync(path.join(root, 'sitemap.xml'), 'utf8');

  test('is valid XML with urlset', () => {
    expect(content).toContain('<?xml');
    expect(content).toContain('<urlset');
    expect(content).toContain('</urlset>');
  });

  test('contains canonical URL', () => {
    expect(content).toContain('<loc>https://jthomas.site/</loc>');
  });

  test('has lastmod date', () => {
    expect(content).toMatch(/<lastmod>\d{4}-\d{2}-\d{2}<\/lastmod>/);
  });

  test('has changefreq', () => {
    expect(content).toContain('<changefreq>monthly</changefreq>');
  });

  test('has priority', () => {
    expect(content).toContain('<priority>1.0</priority>');
  });

  test('uses sitemaps.org schema', () => {
    expect(content).toContain('xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"');
  });
});

describe('google81dc4accb9d1182d.html', () => {
  const content = fs.readFileSync(path.join(root, 'google81dc4accb9d1182d.html'), 'utf8');

  test('contains verification string', () => {
    expect(content).toContain('google-site-verification: google81dc4accb9d1182d.html');
  });

  test('has noindex meta tag', () => {
    expect(content).toContain('noindex');
    expect(content).toContain('nofollow');
  });

  test('has proper HTML structure', () => {
    expect(content).toContain('<!DOCTYPE html>');
    expect(content).toContain('<head>');
    expect(content).toContain('</head>');
  });
});

describe('CNAME', () => {
  const content = fs.readFileSync(path.join(root, 'CNAME'), 'utf8');

  test('contains custom domain', () => {
    expect(content.trim()).toBe('jthomas.site');
  });
});
