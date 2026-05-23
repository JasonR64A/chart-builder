"""
Render two stand-alone scatter charts for the USC + Mercer 2026 article:

1. usc_vs_hosts_2013_2025.png
   X = RPI rank (1 on right, inverted), Y = Q1+Q2 win pct
   Backdrop = all 192 regional hosts 2013-2025 (no 2020), gray
   Highlight = USC 2026 (red), dot size = overall wins

2. mercer_vs_at_large_2013_2025.png
   Same axes
   Backdrop = entrants 2013-2025 who were NOT hosts AND from a multi-bid
              conference (>=2 entrants from same conf that year), gray
   Highlight = Mercer 2026 (red)

Conference lookup uses teams.csv (current 2026 alignment) applied to all
historical years. Note in footnote.
"""
import csv
import re
from pathlib import Path
from collections import defaultdict, Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO = Path(r'C:\Dev\chart-builder-app')
BRACK = REPO / 'data' / 'bracketology'

# 64 Analytics — match the Selection Committee Study series brand
BG       = '#F4EFE3'   # cream, slightly warmer than #FAF8F2
INK      = '#1F1B17'   # near-black with warm undertone
SUBINK   = '#7A736A'   # muted gray for subtitle/source
BACKDROP = '#C7BFAF'   # warm beige for backdrop dots
RED      = '#C41230'   # highlight / target
GOLD     = '#C9A96B'   # accent (mid emphasis)
GREEN    = '#5B7C3E'   # positive accent

plt.rcParams['font.family'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.edgecolor'] = SUBINK
plt.rcParams['axes.labelcolor'] = INK
plt.rcParams['xtick.color'] = SUBINK
plt.rcParams['ytick.color'] = SUBINK


def norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def load_historical():
    rows = []
    with open(BRACK / 'historical_q1q2_baseball.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['note'] == 'no_yearid_match':  # skip only unmatched rows; keep overrides
                continue
            if not r['rpi_rank'] or not r['q1q2_pct']:
                continue
            rows.append({
                'year': int(r['year']),
                'team': r['team'],
                'rpi_rank': int(r['rpi_rank']),
                'in_tournament': r['in_tournament'] == 'True',
                'is_host': r['is_host'] == 'True',
                'w': int(r['w']), 'l': int(r['l']),
                'q1q2_w': int(r['q1q2_w']),
                'q1q2_g': int(r['q1q2_g']),
                'q1q2_pct': float(r['q1q2_pct']),
            })
    return rows


def load_current_targets():
    out = {}
    with open(BRACK / 'current_q1q2_baseball.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            out[r['team']] = {
                'team': r['team'],
                'conference': r['conference'],
                'rpi_rank': int(r['rpi_rank']),
                'record': r['record'],
                'w': int(r['w']), 'l': int(r['l']),
                'q1q2_pct': float(r['q1q2_pct']),
                'q1q2_w': int(r['q1q2_w']),
                'q1q2_g': int(r['q1q2_g']),
            }
    return out


def load_team_conferences():
    """name_norm -> conference name. Uses current chart-builder teams.csv."""
    confs = {}
    with open(REPO / 'data' / 'conferences.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            confs[r['id']] = r['name']
    out = {}
    with open(REPO / 'data' / 'teams.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['sport'] != 'Baseball':
                continue
            out[norm(r['name'])] = confs.get(r['conference_id'], '')
    return out


def size_for_wins(wins):
    """Map overall wins to dot area. Range: ~28W -> 28, ~52W -> 200."""
    base = 28
    scale = 6
    s = base + max(0, wins - 28) * scale
    return max(20, min(s, 260))


def _letter_spaced(s, spaces=2):
    """Apply visual letter-spacing by inserting spaces between characters.
    spaces=2 between letters and 4 between separators reproduces the
    'D 1   B A S E B A L L     ·     ...' look of the snubs chart."""
    out = []
    for word in s.split(' '):
        out.append((' ' * spaces).join(list(word)))
    return ('   ' * 2).join(out)


def render_chart(out_path, title, subtitle, backdrop, target, cohort_label,
                 footer_tag, highlight_color=RED,
                 label_offset=(0.0, 0.0), label_anchor='auto',
                 x_max=75, x_ticks=None):
    fig, ax = plt.subplots(figsize=(13.5, 8.6), facecolor=BG)
    ax.set_facecolor(BG)

    # Subtle "above .500" band — green tint mirrors the snubs chart Q1+Q2 % column
    ax.axhspan(0.500, 0.95, color=GREEN, alpha=0.04, zorder=0)

    # Backdrop dots — warm beige to match the series palette
    xs = [r['rpi_rank'] for r in backdrop]
    ys = [r['q1q2_pct'] for r in backdrop]
    ss = [size_for_wins(r['w']) for r in backdrop]
    ax.scatter(xs, ys, s=ss, color=BACKDROP, alpha=0.55,
               edgecolors='none', zorder=2)

    # Target
    tx, ty, tsize = target['rpi_rank'], target['q1q2_pct'], size_for_wins(target['w'])
    ax.scatter([tx], [ty], s=tsize * 2.0, color=highlight_color,
               edgecolors=INK, linewidths=1.0, zorder=4)

    # Target label
    pct_color = GREEN if target['q1q2_pct'] >= 0.500 else RED
    label_line1 = f"{target['team']} 2026"
    label_line2 = (f"RPI #{target['rpi_rank']}  ·  {target['record']}  ·  "
                   f"Q1+Q2 {target['q1q2_w']}-{target['q1q2_g']-target['q1q2_w']} "
                   f"({target['q1q2_pct']:.3f})")
    lx_dx, lx_dy = label_offset
    if label_anchor == 'auto':
        label_ha = 'left' if lx_dx >= 0 else 'right'
        label_va = 'bottom' if lx_dy >= 0 else 'top'
    else:
        label_ha, label_va = label_anchor.split('_')
    label_text = f"{label_line1}\n{label_line2}"
    ax.annotate(label_text, xy=(tx, ty), xytext=(tx + lx_dx, ty + lx_dy),
                ha=label_ha, va=label_va, fontsize=11.5, color=INK,
                fontweight='bold', zorder=5,
                arrowprops=dict(arrowstyle='-', color=highlight_color,
                                lw=1.0, alpha=0.85,
                                connectionstyle='arc3,rad=0.0',
                                shrinkA=8, shrinkB=10),
                bbox=dict(boxstyle='round,pad=0.55', fc=BG,
                          ec=highlight_color, lw=1.3))

    # Axes — quiet, minimal, like a polished editorial chart
    ax.set_xlim(x_max + 1, -2)
    ax.set_ylim(-0.02, 0.92)
    ax.set_xlabel('RPI rank  (#1 on right)', fontsize=10.5, color=SUBINK,
                  labelpad=10)
    ax.set_ylabel('Q1 + Q2 win %', fontsize=10.5, color=SUBINK, labelpad=12)
    if x_ticks is None:
        x_ticks = [1, 10, 20, 30, 40, 50, 60, 75]
        x_ticks = [t for t in x_ticks if t <= x_max]
    ax.set_xticks(x_ticks)
    ax.set_yticks([0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_yticklabels(['.000', '.200', '.400', '.500', '.600', '.700', '.800'])
    ax.tick_params(axis='both', labelsize=9.5, length=3, width=0.7)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color(SUBINK)
        ax.spines[spine].set_linewidth(0.7)
    ax.grid(True, axis='y', color=BACKDROP, alpha=0.35, linewidth=0.55, zorder=1)
    # Host line + 0.500 line — gold for emphasis like the snubs DROP bars
    ax.axvline(16.5, color=GOLD, linewidth=1.0, alpha=0.7, linestyle='-', zorder=1)
    ax.axhline(0.500, color=SUBINK, linewidth=0.5, alpha=0.4, linestyle='--', zorder=1)
    ax.text(16.5, 0.88, '  host line (RPI 16)', fontsize=8.5, color=GOLD,
            ha='left', va='top', fontweight='bold')

    # Title block
    fig.text(0.06, 0.93, title, fontsize=23, color=INK, weight='bold')
    fig.text(0.06, 0.888, subtitle, fontsize=11, color=SUBINK)

    # Dot-size legend — discreet, mirrors the snubs chart's quiet legend strip
    for wins in (30, 42, 54):
        ax.scatter([], [], s=size_for_wins(wins), color=BACKDROP, alpha=0.6,
                   label=f'{wins} W', edgecolors='none')
    leg = ax.legend(loc='lower left', frameon=False, fontsize=9,
                    labelcolor=INK, title='dot size = overall wins',
                    title_fontsize=8.5, handletextpad=1.0, borderaxespad=1.0,
                    bbox_to_anchor=(0.005, 0.005))
    leg.get_title().set_color(SUBINK)

    # Footer (three lines, all left-aligned, generous vertical breathing room)
    # Row 1: small-caps row-count tag
    fig.text(0.06, 0.072, footer_tag, fontsize=9.5, color=INK, weight='bold')
    # Row 2: cohort note
    fig.text(0.06, 0.046, cohort_label, fontsize=8.5, color=SUBINK)
    # Row 3: thresholds (chart methodology)
    fig.text(0.06, 0.022,
             'Q1: H≤30, N≤50, A≤75  ·  Q2: H≤75, N≤100, A≤135  ·  Records as of Selection Monday',
             fontsize=8, color=SUBINK)

    plt.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.18)
    fig.savefig(out_path, dpi=180, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')


def write_cohort_csv(rows, target, path, extra_fields=()):
    """Write the data behind one chart: backdrop rows + the target row tagged."""
    fields = ['kind', 'year', 'team', 'rpi_rank', 'w', 'l',
              'q1q2_w', 'q1q2_g', 'q1q2_pct'] + list(extra_fields)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            row = {'kind': 'historical', 'year': r['year'], 'team': r['team'],
                   'rpi_rank': r['rpi_rank'], 'w': r['w'], 'l': r['l'],
                   'q1q2_w': r['q1q2_w'], 'q1q2_g': r['q1q2_g'],
                   'q1q2_pct': r['q1q2_pct']}
            for k in extra_fields:
                row[k] = r.get(k, '')
            wr.writerow(row)
        # Target row
        trow = {'kind': 'target_2026', 'year': 2026, 'team': target['team'],
                'rpi_rank': target['rpi_rank'], 'w': target['w'], 'l': target['l'],
                'q1q2_w': target['q1q2_w'], 'q1q2_g': target['q1q2_g'],
                'q1q2_pct': target['q1q2_pct']}
        for k in extra_fields:
            trow[k] = target.get(k, '')
        wr.writerow(trow)


def main():
    hist = load_historical()
    targets = load_current_targets()
    name_conf = load_team_conferences()

    # ── USC chart: all hosts ─────────────────────────────────────────────
    hosts = [r for r in hist if r['is_host']]
    usc = targets['Southern California']
    render_chart(
        BRACK / 'usc_vs_hosts_2013_2025.png',
        title='Does USC look like a host?',
        subtitle=("Southern California 2026 plotted against every regional host  ·  "
                  f"12 NCAA tournaments, 2013-2025  ·  n={len(hosts)} host-seasons"),
        backdrop=hosts,
        target=usc,
        cohort_label='Backdrop: every regional host named by the NCAA D1 baseball committee 2013-2025 (no 2020).',
        footer_tag=f'{len(hosts)} HOSTS',
        # USC at (8, .600). Inverted x: rank=8 is on the right. Anchor label upper-LEFT (toward x>8).
        label_offset=(15, 0.20), label_anchor='center_bottom',
        # Hosts almost never come from below RPI ~22 (only outlier: Louisiana Tech 2021 #33).
        # Cap x at 35 to drop the empty 36-75 range.
        x_max=35, x_ticks=[1, 5, 10, 16, 20, 25, 30, 35],
    )
    write_cohort_csv(hosts, usc, BRACK / 'usc_vs_hosts_2013_2025.csv',
                     extra_fields=('is_host',))

    # ── Mercer chart: at-large from multi-bid conferences ────────────────
    # First, attach conference to historical rows
    for r in hist:
        r['conference'] = name_conf.get(norm(r['team']), '')

    # Count entrants per (year, conference) — only counting in_tournament=True
    entrant_conf_counts = Counter()
    for r in hist:
        if r['in_tournament'] and r['conference']:
            entrant_conf_counts[(r['year'], r['conference'])] += 1

    # Cohort: in_tournament=True, NOT is_host, multi-bid conference (>=2)
    at_large_cohort = [r for r in hist
                       if r['in_tournament']
                       and not r['is_host']
                       and r['conference']
                       and entrant_conf_counts[(r['year'], r['conference'])] >= 2]

    mercer = targets['Mercer']
    render_chart(
        BRACK / 'mercer_vs_at_large_2013_2025.png',
        title='Does Mercer fit the multi-bid-league field?',
        subtitle=("Mercer 2026 plotted against non-host entrants from multi-bid conferences  ·  "
                  f"12 NCAA tournaments, 2013-2025  ·  n={len(at_large_cohort)} team-seasons"),
        backdrop=at_large_cohort,
        target=mercer,
        cohort_label=('Backdrop: non-host entrants from conferences that placed 2+ teams in the field '
                      'that year. Includes conference-tournament auto-bid winners from multi-bid leagues '
                      '(e.g. Nebraska 2025) — not pure at-large picks. Conference assignments use 2026 alignment.'),
        footer_tag=f'{len(at_large_cohort)} ENTRANT SEASONS',
        # Mercer at (28, .409). Anchor label upper-right (toward higher rpi_rank = leftward on inverted axis).
        label_offset=(18, 0.22), label_anchor='center_bottom',
    )
    write_cohort_csv(at_large_cohort, mercer,
                     BRACK / 'mercer_vs_at_large_2013_2025.csv',
                     extra_fields=('conference',))


if __name__ == '__main__':
    main()
