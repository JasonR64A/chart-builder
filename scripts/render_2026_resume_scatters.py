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

# 64 Analytics brand
BG = '#FAF8F2'
INK = '#2D2926'
GRAY = '#B8B0A4'
RED = '#C41230'
GOLD = '#C9A96B'
SUBINK = '#6B6359'

plt.rcParams['font.family'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.edgecolor'] = INK
plt.rcParams['axes.labelcolor'] = INK
plt.rcParams['xtick.color'] = INK
plt.rcParams['ytick.color'] = INK


def norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def load_historical():
    rows = []
    with open(BRACK / 'historical_q1q2_baseball.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['note']:  # skip unmatched rows
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


def render_chart(out_path, title, subtitle, backdrop, target, cohort_label,
                 highlight_color=RED, label_offset=(0.0, 0.0), label_anchor='auto'):
    fig, ax = plt.subplots(figsize=(13, 8.5), facecolor=BG)
    ax.set_facecolor(BG)

    # Backdrop dots
    xs = [r['rpi_rank'] for r in backdrop]
    ys = [r['q1q2_pct'] for r in backdrop]
    ss = [size_for_wins(r['w']) for r in backdrop]
    ax.scatter(xs, ys, s=ss, color=GRAY, alpha=0.45, edgecolors='none', zorder=2)

    # Target
    tx, ty, tsize = target['rpi_rank'], target['q1q2_pct'], size_for_wins(target['w'])
    ax.scatter([tx], [ty], s=tsize * 1.8, color=highlight_color,
               edgecolors=INK, linewidths=1.2, zorder=4)

    # Target label
    label = (f"{target['team']} 2026\n"
             f"RPI #{target['rpi_rank']}  ·  {target['record']}  ·  "
             f"Q1+Q2 {target['q1q2_w']}-{target['q1q2_g']-target['q1q2_w']} "
             f"({target['q1q2_pct']:.3f})")
    # Use caller-supplied label offset (axis-data units). x positive = away from
    # rank #1 (leftward on the inverted axis); y positive = upward.
    lx_dx, lx_dy = label_offset
    if label_anchor == 'auto':
        label_ha = 'left' if lx_dx >= 0 else 'right'
        label_va = 'bottom' if lx_dy >= 0 else 'top'
    else:
        label_ha, label_va = label_anchor.split('_')
    # Draw connector line from dot to label box
    ax.annotate(label, xy=(tx, ty), xytext=(tx + lx_dx, ty + lx_dy),
                ha=label_ha, va=label_va, fontsize=11.5, color=INK,
                fontweight='bold', zorder=5,
                arrowprops=dict(arrowstyle='-', color=highlight_color,
                                lw=1.0, alpha=0.8,
                                connectionstyle='arc3,rad=0.0'),
                bbox=dict(boxstyle='round,pad=0.45', fc=BG, ec=highlight_color, lw=1.3))

    # Axes
    ax.set_xlim(76, -2)  # inverted: RPI #1 on right
    ax.set_ylim(-0.02, 0.92)
    ax.set_xlabel('RPI rank  (#1 on right)', fontsize=11, color=INK, labelpad=8)
    ax.set_ylabel('Q1 + Q2 win %', fontsize=12, color=INK, labelpad=10)
    ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 75])
    ax.set_yticks([0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_yticklabels(['.000', '.200', '.400', '.500', '.600', '.700', '.800'])
    ax.tick_params(axis='both', labelsize=10)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color(SUBINK)
        ax.spines[spine].set_linewidth(0.8)
    ax.grid(True, axis='y', color=GRAY, alpha=0.35, linewidth=0.6, zorder=1)
    ax.axhline(0.500, color=SUBINK, linewidth=0.5, alpha=0.5, linestyle='--', zorder=1)
    ax.axvline(16.5, color=SUBINK, linewidth=0.5, alpha=0.5, linestyle='--', zorder=1)
    # Annotate the host line
    ax.text(16.5, 0.89, 'host line (RPI 16)', fontsize=8.5, color=SUBINK,
            ha='right', va='top', rotation=0)

    # Title block
    fig.text(0.06, 0.95, 'D 1   B A S E B A L L     ·     6 4   A N A L Y T I C S',
             fontsize=9.5, color=SUBINK, weight='bold')
    fig.text(0.06, 0.905, title, fontsize=22, color=INK, weight='bold')
    fig.text(0.06, 0.873, subtitle, fontsize=11.5, color=SUBINK)

    # Legend strip (bottom-right of plot area): dot size legend
    leg_x = 0.78
    leg_y_top = 0.20
    for i, (wins, lbl) in enumerate([(30, '30 W'), (42, '42 W'), (54, '54 W')]):
        y = leg_y_top - i * 0.04
        ax.scatter([], [], s=size_for_wins(wins), color=GRAY, alpha=0.6,
                   label=lbl, edgecolors='none')
    leg = ax.legend(loc='lower left', frameon=False, fontsize=9.5,
                    labelcolor=INK, title='dot size = overall wins',
                    title_fontsize=9, handletextpad=1.2, borderaxespad=1.2,
                    bbox_to_anchor=(0.01, 0.01))
    leg.get_title().set_color(SUBINK)

    # Footer
    fig.text(0.06, 0.045, cohort_label, fontsize=9, color=SUBINK)
    fig.text(0.06, 0.020,
             'Q1: H≤30 · N≤50 · A≤75    Q2: H≤75 · N≤100 · A≤135    '
             'RPI + records as of Selection Monday (2026 snapshot: 2026-05-22)',
             fontsize=8, color=SUBINK)
    fig.text(0.97, 0.020, 'Source: NCAA team schedules + selection-week RPI',
             fontsize=8, color=SUBINK, ha='right')

    plt.subplots_adjust(left=0.07, right=0.97, top=0.83, bottom=0.16)
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
        subtitle=f"Southern California 2026 vs every regional host, 2013-2025 (n={len(hosts)})",
        backdrop=hosts,
        target=usc,
        cohort_label=f'Backdrop: {len(hosts)} regional hosts across 12 NCAA tournaments (2013-2025, no 2020).',
        # USC at (8, .600). Inverted x: rank=8 is on the right. Anchor label to upper-LEFT (toward x>8).
        label_offset=(25, 0.18), label_anchor='center_bottom',
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
        title='Does Mercer look like an at-large pick?',
        subtitle=f"Mercer 2026 vs at-large-track entrants 2013-2025 (n={len(at_large_cohort)})",
        backdrop=at_large_cohort,
        target=mercer,
        cohort_label=('Backdrop: non-host entrants from conferences that placed 2+ teams '
                      'in the field that year (filter excludes one-bid auto-bid leagues). '
                      'Conference assignments use current 2026 alignment for all years.'),
        # Mercer at (27, .500). Anchor label upper-right (toward higher rpi_rank = leftward on inverted axis).
        label_offset=(18, 0.18), label_anchor='center_bottom',
    )
    write_cohort_csv(at_large_cohort, mercer,
                     BRACK / 'mercer_vs_at_large_2013_2025.csv',
                     extra_fields=('conference',))


if __name__ == '__main__':
    main()
