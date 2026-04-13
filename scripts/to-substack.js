#!/usr/bin/env node
/*
 * scripts/to-substack.js
 *
 * Convert a blog post HTML file from blog/ into Substack-ready Markdown.
 *
 * Usage:
 *   node scripts/to-substack.js blog/claude-agent.html
 *   node scripts/to-substack.js blog/claude-agent.html --copy
 *   node scripts/to-substack.js blog/claude-agent.html --out=out/claude.md
 *
 * Flags:
 *   --copy        copy the body to the clipboard (macOS pbcopy)
 *   --out=PATH    write to PATH instead of stdout
 *
 * Paste the body into Substack's editor. The Title and Subtitle shown
 * in the header block go into Substack's dedicated fields above the
 * body. SVG diagrams in the original post are replaced with a
 * placeholder; screenshot them from the live site and drop them in as
 * images at the placeholder locations.
 */

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');
const { JSDOM } = require('jsdom');

// ── Argument parsing ──────────────────────────────────────────────
const args = process.argv.slice(2);
const fileArg = args.find(a => !a.startsWith('--'));
const copyFlag = args.includes('--copy');
const outFlagRaw = args.find(a => a.startsWith('--out='));
const outPath = outFlagRaw ? outFlagRaw.split('=').slice(1).join('=') : null;

if (!fileArg || args.includes('--help') || args.includes('-h')) {
  process.stderr.write([
    'Usage: node scripts/to-substack.js <blog/file.html> [--copy] [--out=PATH]',
    '',
    'Converts a blog post HTML file into Substack-ready Markdown.',
    '  --copy        copy the body to the clipboard (macOS pbcopy)',
    '  --out=PATH    write to PATH instead of stdout',
    '',
  ].join('\n'));
  process.exit(fileArg ? 0 : 1);
}

const absPath = path.resolve(fileArg);
if (!fs.existsSync(absPath)) {
  process.stderr.write(`error: file not found: ${absPath}\n`);
  process.exit(1);
}

// ── Parse ─────────────────────────────────────────────────────────
const html = fs.readFileSync(absPath, 'utf8');
const dom = new JSDOM(html);
const doc = dom.window.document;

const title = (doc.querySelector('.post-title')?.textContent || '').trim();
const lede = (doc.querySelector('.post-lede')?.textContent || '').trim();
const metaText = Array.from(doc.querySelectorAll('.post-meta > span'))
  .map(el => el.textContent.trim())
  .filter(Boolean)
  .join(' · ');
const canonical =
  doc.querySelector('link[rel="canonical"]')?.getAttribute('href') || '';

const body = doc.querySelector('.post-body');
if (!body) {
  process.stderr.write(`error: no .post-body element found in ${fileArg}\n`);
  process.exit(1);
}

// ── Inline (within-paragraph) conversion ─────────────────────────
function inline(node) {
  let out = '';
  for (const child of node.childNodes) {
    if (child.nodeType === 3) {
      // text node — collapse internal whitespace
      out += child.textContent.replace(/\s+/g, ' ');
    } else if (child.nodeType === 1) {
      const tag = child.tagName.toLowerCase();
      if (tag === 'strong' || tag === 'b') {
        const inner = inline(child).trim();
        if (inner) out += `**${inner}**`;
      } else if (tag === 'em' || tag === 'i') {
        const inner = inline(child).trim();
        if (inner) out += `*${inner}*`;
      } else if (tag === 'code') {
        out += '`' + child.textContent + '`';
      } else if (tag === 'a') {
        const href = child.getAttribute('href') || '';
        const text = inline(child).trim();
        if (text) out += `[${text}](${href})`;
      } else if (tag === 'br') {
        out += '  \n';
      } else {
        // Unknown tag (span, etc.) — walk children as text
        out += inline(child);
      }
    }
  }
  return out;
}

// ── Language detection for fenced code blocks ────────────────────
function detectLang(code) {
  if (/^\s*(from |import |def |class |@|print\()/m.test(code)) return 'python';
  if (/^\s*(git|pip|cd|npm|yarn|python3?|curl|wget|ls|echo|local-tinker)\b/m.test(code)) return 'bash';
  if (/^\$ /m.test(code)) return 'bash';
  return '';
}

// ── Block-level conversion ───────────────────────────────────────
function block(node) {
  if (node.nodeType !== 1) return '';
  const tag = node.tagName.toLowerCase();

  if (tag === 'p') {
    const text = inline(node).trim();
    return text ? text + '\n\n' : '';
  }

  if (tag === 'h2') return `## ${inline(node).trim()}\n\n`;
  if (tag === 'h3') return `### ${inline(node).trim()}\n\n`;
  if (tag === 'h4') return `#### ${inline(node).trim()}\n\n`;

  if (tag === 'ul' || tag === 'ol') {
    let out = '';
    const items = Array.from(node.children).filter(
      c => c.tagName && c.tagName.toLowerCase() === 'li'
    );
    items.forEach((li, i) => {
      const bullet = tag === 'ul' ? '-' : `${i + 1}.`;
      out += `${bullet} ${inline(li).trim()}\n`;
    });
    return out + '\n';
  }

  if (tag === 'pre') {
    const codeEl = node.querySelector('code') || node;
    const code = codeEl.textContent.replace(/\n+$/, '');
    const lang = detectLang(code);
    return '```' + lang + '\n' + code + '\n```\n\n';
  }

  if (tag === 'figure') {
    const caption = (node.querySelector('figcaption')?.textContent || '').trim();
    const src = canonical || 'the blog URL';
    return (
      `> **📊 Diagram** — ${caption}\n` +
      `> *(inline SVG in the original post at ${src} — screenshot and paste here as an image)*\n\n`
    );
  }

  if (tag === 'table') {
    const rows = Array.from(node.querySelectorAll('tr'));
    if (rows.length === 0) return '';
    const rowCells = tr => Array.from(tr.children).map(cell => inline(cell).trim());
    const first = rows[0];
    const isHeaderRow = first.querySelector('th') !== null;
    const headerCells = isHeaderRow ? rowCells(first) : null;
    const bodyStart = isHeaderRow ? 1 : 0;
    const bodyRows = rows.slice(bodyStart).map(rowCells);

    let out = '';
    if (headerCells) {
      out += `| ${headerCells.join(' | ')} |\n`;
      out += `| ${headerCells.map(() => '---').join(' | ')} |\n`;
    }
    bodyRows.forEach(row => {
      out += `| ${row.join(' | ')} |\n`;
    });
    return out + '\n';
  }

  if (tag === 'blockquote') {
    const text = inline(node).trim();
    return text
      .split('\n')
      .map(l => `> ${l}`)
      .join('\n') + '\n\n';
  }

  if (tag === 'div') {
    // Tech-stack pill row → bullet list
    if (node.classList.contains('tech-stack-pills')) {
      const pills = Array.from(node.querySelectorAll('.proj-lang')).map(p =>
        p.textContent.trim()
      );
      return pills.map(p => `- **${p}**`).join('\n') + '\n\n';
    }
    // Unknown div — walk children
    let out = '';
    for (const child of node.children) out += block(child);
    return out;
  }

  return '';
}

// ── Build body markdown ──────────────────────────────────────────
let bodyMd = '';
for (const child of body.children) bodyMd += block(child);
bodyMd = bodyMd.replace(/\n{3,}/g, '\n\n').trimEnd() + '\n';

// ── Header block (metadata for Substack fields) ──────────────────
const rule = '─'.repeat(78);
let header = '';
header += rule + '\n';
if (title) header += `  Title:    ${title}\n`;
if (lede) header += `  Subtitle: ${lede}\n`;
if (metaText) header += `  Meta:     ${metaText}\n`;
if (canonical) header += `  Source:   ${canonical}\n`;
header += rule + '\n\n';
header += '  Paste the body below into Substack. Copy the Title and Subtitle\n';
header += "  above into Substack's dedicated fields. SVG diagrams need to be\n";
header += '  screenshotted from the live site and inserted as images.\n\n';
header += rule + '\n\n';

// ── Output routing ───────────────────────────────────────────────
if (outPath) {
  fs.mkdirSync(path.dirname(path.resolve(outPath)), { recursive: true });
  fs.writeFileSync(outPath, header + bodyMd);
  process.stderr.write(
    `wrote ${outPath} (${(header + bodyMd).length} chars, ${bodyMd.split('\n').length - 1} body lines)\n`
  );
} else if (copyFlag) {
  const result = spawnSync('pbcopy', [], { input: bodyMd });
  if (result.status === 0) {
    process.stderr.write(rule + '\n');
    if (title) process.stderr.write(`  Title:    ${title}\n`);
    if (lede) process.stderr.write(`  Subtitle: ${lede}\n`);
    if (metaText) process.stderr.write(`  Meta:     ${metaText}\n`);
    process.stderr.write(rule + '\n');
    process.stderr.write(
      `  Copied ${bodyMd.length} chars of body markdown to the clipboard.\n`
    );
    process.stderr.write(
      '  Paste into Substack\'s body, then enter the title/subtitle above.\n'
    );
    process.stderr.write(rule + '\n');
  } else {
    process.stderr.write('pbcopy failed; writing to stdout instead\n');
    process.stdout.write(header + bodyMd);
  }
} else {
  process.stdout.write(header + bodyMd);
}
