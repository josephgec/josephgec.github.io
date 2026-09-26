# jthomas.site

Static personal site (no framework, no build step), served from the repo root
via GitHub Pages at https://jthomas.site. Sections: `/` (home), `/blog/`,
`/ml/` (interactive encyclopedia), `/consulting.html`, `blog/feed.xml` (RSS).

Only the home page uses the design below. `/blog/`, `/ml/` and
`/consulting.html` still use the older "Notebook" stylesheet (`styles.css`).

## Home page design: "Swarm stage"

A live glowworm swarm (`js/swarm.js`, drawn on a fixed `<canvas id="stage">`)
sits behind the page. Each element with `data-shape="…"` names a figure. When
that element crosses the middle of the viewport, the swarm breaks formation,
dims, and regroups into the figure:

| Section       | `data-shape` | Figure                                          |
|---------------|--------------|-------------------------------------------------|
| hero          | `hero`       | “Emergent behavior,” in particles (left to right) |
| throughline   | `years`      | 2008 / 2014 / 2026                              |
| writing       | `ratchet`    | a ratchet staircase                             |
| tutorial (ML) | `network`    | a 4-6-6-3 neural net                            |
| selected work | `sources`    | agents settled around three sources (GSO)       |
| appendix      | `ambient`    | no figure; swarm wanders, canvas fades to 45%   |
| contact       | `mail`       | hello@jthomas.site                              |

To add a figure, add a function to `SHAPES` in `js/swarm.js`. A `(g) => {}`
function draws into an offscreen canvas that gets sampled; a zero-argument
function returns points directly. Desktop figures go in `region()` (the right
half); below 800px they sit in the top ~30% of the screen, above the text.

Behaviour rules:
- The cursor is a torch: agents scatter from it and brighten.
- Glow = luciferin-style level (rises in formation, decays out of it) times a
  travelling wave. Agents are *assigned* points, so never claim in copy that
  the figures emerge unaided.
- Agent count scales with screen area (700–1800). The loop pauses on
  hidden tabs.
- `prefers-reduced-motion`: figures are drawn fully formed with no animation,
  and the hero copy shows immediately.
- The hero copy (`.reveal`) fades in after the headline assembles, with a
  5-second safety timeout.

### Tokens (`css/home.css`)

Palette “Sky”:

| Token          | Value     | Use                                    |
|----------------|-----------|----------------------------------------|
| `--bg`         | `#EAF3FB` | page                                   |
| `--surface`    | `#F6FAFE` | cards                                  |
| `--ink`        | `#0F2238` | text, 14.3:1                           |
| `--ink-2`      | `#4A5E75` | secondary text, 5.9:1                  |
| `--accent`     | `#1F5FCC` | links, buttons, 5.2:1 (white on it 5.9:1) |
| `--swarm-glow` | `#2F7BFF` | canvas only, never text                |
| `--swarm-dot`  | `#8FA6C0` | canvas only                            |

Type:
- **Bricolage Grotesque** (variable, wght 400–700, opsz) for everything
  except metadata.
- **IBM Plex Mono** 400/500 for kickers (`§ label`), dates, tags, citation
  counts and the figure caption.
- Both are self-hosted in `/fonts/` (Latin subsets, OFL licences alongside).
  Bricolage is preloaded, with a metric-adjusted Arial fallback.

Shape: buttons have a 10px radius (`--radius`), cards and the photo 16px
(`--radius-card`), and tags are fully rounded.

### Don't
- change the palette or fonts without asking; the user picked Sky +
  Bricolage from live comparisons (Sep 2026)
- add scroll-jacking, parallax, or a second attention-grabbing animation;
  the swarm is the show and everything else stays quiet
- invent stats, credentials or copy; copy on the page is the user's own
- load fonts from Google Fonts (self-hosted on purpose)

### Checking changes
Serve the repo root (`python3 -m http.server`) and screenshot at 1440, 768
and 390px. Chrome won't shrink below ~500px, so render the page inside
fixed-width iframes for narrow widths. Check that `scrollWidth` equals the
viewport width (no sideways scroll).

## Tests

`__tests__/` (Jest + jsdom) predates the redesigns, and many assertions
target old markup (Syne font, stats ribbon, particle canvas IDs, Google
Fonts preconnect). When a test fails after a design change, rewrite the test
for the new design. Don't bend the HTML to satisfy an old assertion.
