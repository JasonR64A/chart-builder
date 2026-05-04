"""Weekly Awards 'Top 10' graphic — 1080×1080 SVG using the pre-rendered
grunge backdrop at assets/branding/weekly_awards_background.png.

The backdrop has the frame, flags (1–10), 'TOP 10' headline,
'PRESENTED BY 64 ANALYTICS' lockup, side rails, hero panel border,
bottom stats strip border, and decorative elements all baked in.

This module ONLY overlays dynamic content:
  • Sport / division / role subtitle (covers the baked 'D3 BASEBALL
    PITCHERS' text).
  • Week tag (covers the baked 'WEEK XX | DATE RANGE' label).
  • For each of the 10 numbered pills: team logo, player name, team
    subline, and rank-stat value.
  • Hero image fitted inside the hero panel (no spill).
  • Six bottom-strip stat cells: label + value + leader.
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd

_APP_DIR = Path(__file__).resolve().parent.parent
LOGO_DIR = _APP_DIR / 'team_logos_512'
BACKGROUND = _APP_DIR / 'assets' / 'branding' / 'weekly_awards_background.png'

W = 1080
H = 1080

# ── Backdrop-derived coordinates (pixel-measured from the grunge art) ──
# Each pill is ~44 px tall, stride 57 px between pill tops, 10 rows starting
# at y=406. Pill body spans x≈140 (after flag chevron) to x≈530.
PILL_X = 140
PILL_RIGHT = 530
PILL_Y0 = 406
PILL_STRIDE = 57
PILL_H = 44

# Hero panel — empty bordered region on the right
HERO_X = 575
HERO_Y = 145
HERO_W = 440
HERO_H = 808

# Bottom stats strip — empty bordered region with two horizontal red/white lines
STATS_X = 575
STATS_Y = 985
STATS_W = 440
STATS_H = 70

# Static-text cover regions, sized to fully obscure the baked text.
# Subtitle ('D3 BASEBALL PITCHERS') white pixels live at y≈282-306, x≈78-427.
SUB_X = 72
SUB_Y = 274
SUB_W = 380
SUB_H = 40

# Week tag (dark cell with red left border + 'WEEK XX | DATE RANGE') sits in
# the brushstroke bar at y≈68-104, x≈790-1020.
WEEK_X = 790
WEEK_Y = 67
WEEK_W = 235
WEEK_H = 38


def _xe(s):
    return html.escape(str(s) if s is not None else '')


def _b64(p: Path) -> str | None:
    try:
        return f'data:image/png;base64,{base64.b64encode(p.read_bytes()).decode("ascii")}'
    except Exception:
        return None


def _fmt(val, decimals: int) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return '—'
    if isinstance(val, (int, float)) and decimals is not None:
        try:
            return f'{float(val):.{int(decimals)}f}'
        except Exception:
            return str(val)
    return str(val)


def _initials(name: str) -> str:
    if not name:
        return '?'
    parts = [w for w in str(name).split() if w]
    if not parts:
        return '?'
    return ''.join(p[0] for p in parts[:2]).upper()


def _row_overlay(row: dict, idx: int, *, stat_decimals: int,
                  stat_suffix: str, show_team: bool) -> str:
    """Just the in-pill content (logo + name + team + stat). No frame —
    the backdrop already has the chevron flag and pill body."""
    py = PILL_Y0 + idx * PILL_STRIDE
    pill_cy = py + PILL_H / 2

    parts = []
    logo_size = 28
    logo_x = PILL_X + 6
    logo_y = py + (PILL_H - logo_size) / 2

    logo_b64 = row.get('logo_b64')
    if logo_b64:
        parts.append(
            f'<image href="{logo_b64}" xlink:href="{logo_b64}" '
            f'x="{logo_x}" y="{logo_y}" width="{logo_size}" height="{logo_size}" '
            f'preserveAspectRatio="xMidYMid meet"/>'
        )
    else:
        parts.append(
            f'<circle cx="{logo_x + logo_size/2:.1f}" cy="{pill_cy:.1f}" '
            f'r="{logo_size/2}" fill="rgba(255,255,255,0.10)"/>'
            f'<text x="{logo_x + logo_size/2:.1f}" y="{pill_cy + 3.5:.1f}" '
            f'text-anchor="middle" font-family="Barlow Condensed,sans-serif" '
            f'font-style="italic" font-weight="800" font-size="11" '
            f'fill="rgba(255,255,255,0.7)">'
            f'{_xe(_initials(row.get("player") or row.get("team") or "?"))}</text>'
        )

    name_x = logo_x + logo_size + 10
    player = (row.get('player') or '').upper()
    team = (row.get('team') or '').upper()

    if show_team and team:
        parts.append(
            f'<text x="{name_x}" y="{pill_cy - 2:.1f}" text-anchor="start" '
            f'font-family="Barlow Condensed,Oswald,sans-serif" font-style="italic" '
            f'font-weight="700" font-size="15" fill="#ffffff" '
            f'letter-spacing="0.4">{_xe(player)}</text>'
        )
        parts.append(
            f'<text x="{name_x}" y="{pill_cy + 11:.1f}" text-anchor="start" '
            f'font-family="Oswald,sans-serif" font-weight="500" font-size="9" '
            f'fill="rgba(255,255,255,0.55)" letter-spacing="0.9">'
            f'{_xe(team)}</text>'
        )
    else:
        parts.append(
            f'<text x="{name_x}" y="{pill_cy + 5:.1f}" text-anchor="start" '
            f'font-family="Barlow Condensed,sans-serif" font-style="italic" '
            f'font-weight="700" font-size="16" fill="#ffffff" '
            f'letter-spacing="0.4">{_xe(player)}</text>'
        )

    stat_str = _fmt(row.get('stat'), stat_decimals)
    if stat_suffix:
        stat_str = f'{stat_str} {stat_suffix}'
    parts.append(
        f'<text x="{PILL_RIGHT - 14}" y="{pill_cy + 5:.1f}" text-anchor="end" '
        f'font-family="Barlow Condensed,sans-serif" font-style="italic" '
        f'font-weight="800" font-size="17" fill="#ffffff" '
        f'letter-spacing="0.3">{_xe(stat_str)}</text>'
    )
    return ''.join(parts)


def _stat_cell_overlay(stat: dict, x: float, y: float, width: float, height: float) -> str:
    """One bottom-strip stat cell — drawn inside the empty bordered area."""
    label = (stat.get('label') or '').upper()
    value = stat.get('value')
    decimals = stat.get('decimals', 0)
    leader = (stat.get('leader') or '').upper()
    cx = x + width / 2

    parts = [
        # Label (red, on top)
        f'<text x="{cx}" y="{y + 14}" text-anchor="middle" '
        f'font-family="Oswald,sans-serif" font-weight="700" font-size="11" '
        f'letter-spacing="2.0" fill="#d72638">{_xe(label)}</text>',
        # Value (bold italic, large)
        f'<text x="{cx}" y="{y + 40}" text-anchor="middle" '
        f'font-family="Barlow Condensed,sans-serif" font-style="italic" '
        f'font-weight="800" font-size="24" fill="#ffffff">{_xe(_fmt(value, decimals))}</text>',
    ]
    if leader:
        parts.append(
            f'<text x="{cx}" y="{y + 58}" text-anchor="middle" '
            f'font-family="Oswald,sans-serif" font-weight="500" font-size="9" '
            f'letter-spacing="0.9" fill="rgba(255,255,255,0.55)">'
            f'{_xe(leader)}</text>'
        )
    return ''.join(parts)


def build_weekly_awards_svg(rows: list[dict], top_stats: list[dict], *,
                             sport: str, division: str, stat_type: str,
                             week_label: str = '',
                             headline_top: str = 'TOP 10',
                             headline_sub: str = '',
                             stat_suffix: str = '',
                             stat_decimals: int = 2,
                             show_team_subline: bool = True,
                             rail_text: str = '',
                             hero_b64: str | None = None) -> str:
    """Build the 1080x1080 SVG by layering dynamic overlays on the
    pre-rendered grunge backdrop. `rows` should be 10 dicts with keys:
    rank, player, team, stat, logo_b64. `top_stats` should be 6 dicts
    with keys: label, value, decimals, leader."""
    while len(rows) < 10:
        rows.append({'rank': len(rows) + 1, 'player': '—', 'team': '',
                      'stat': None, 'logo_b64': None})
    rows = rows[:10]
    while len(top_stats) < 6:
        top_stats.append({'label': '', 'value': None, 'decimals': 0, 'leader': ''})
    top_stats = top_stats[:6]

    bg_b64 = _b64(BACKGROUND) or ''

    parts = [
        f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink">',
    ]

    # ── 1. Backdrop ──
    if bg_b64:
        parts.append(
            f'<image href="{bg_b64}" xlink:href="{bg_b64}" '
            f'x="0" y="0" width="{W}" height="{H}" '
            f'preserveAspectRatio="xMidYMid slice"/>'
        )
    else:
        parts.append(f'<rect width="{W}" height="{H}" fill="#08080a"/>')

    # ── 2. Hero image — fits inside the panel; clipped so it can't spill
    #       into the rank list on the left. ──
    if hero_b64:
        parts.append(
            f'<defs><clipPath id="heroClip">'
            f'<rect x="{HERO_X}" y="{HERO_Y}" width="{HERO_W}" height="{HERO_H}"/>'
            f'</clipPath></defs>'
            f'<image href="{hero_b64}" xlink:href="{hero_b64}" '
            f'x="{HERO_X}" y="{HERO_Y}" width="{HERO_W}" height="{HERO_H}" '
            f'preserveAspectRatio="xMidYMid slice" clip-path="url(#heroClip)"/>'
        )

    # ── 3. Cover the baked 'D3 BASEBALL PITCHERS' subtitle and redraw ──
    if headline_sub:
        parts.append(
            f'<rect x="{SUB_X - 6}" y="{SUB_Y}" width="{SUB_W}" height="{SUB_H}" '
            f'fill="#08080a"/>'
        )
        parts.append(
            f'<text x="{SUB_X}" y="{SUB_Y + 32}" text-anchor="start" '
            f'font-family="Barlow Condensed,Oswald,sans-serif" '
            f'font-style="italic" font-weight="800" font-size="32" '
            f'fill="#ffffff" letter-spacing="0.4">'
            f'{_xe(headline_sub.upper())}</text>'
        )

    # ── 4. Cover baked 'WEEK XX | DATE RANGE' tag and redraw ──
    if week_label:
        parts.append(
            f'<rect x="{WEEK_X - 6}" y="{WEEK_Y}" width="{WEEK_W + 12}" '
            f'height="{WEEK_H}" fill="#08080a"/>'
        )
        parts.append(
            f'<line x1="{WEEK_X - 4}" y1="{WEEK_Y + 2}" '
            f'x2="{WEEK_X - 4}" y2="{WEEK_Y + WEEK_H - 2}" '
            f'stroke="#d72638" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{WEEK_X + WEEK_W - 8}" y="{WEEK_Y + WEEK_H*0.7:.1f}" '
            f'text-anchor="end" font-family="Barlow Condensed,sans-serif" '
            f'font-style="italic" font-weight="800" font-size="18" '
            f'letter-spacing="2.4" fill="#ffffff">'
            f'{_xe(week_label.upper())}</text>'
        )

    # ── 5. The 10 pill overlays ──
    for i, row in enumerate(rows):
        parts.append(_row_overlay(row, i, stat_decimals=stat_decimals,
                                    stat_suffix=stat_suffix,
                                    show_team=show_team_subline))

    # ── 6. Bottom stats strip — 6 cells inside the empty bordered area ──
    cell_w = STATS_W / 6
    for i, s in enumerate(top_stats):
        parts.append(
            _stat_cell_overlay(s, STATS_X + i * cell_w, STATS_Y,
                                 cell_w, STATS_H)
        )

    parts.append('</svg>')
    return ''.join(parts)


def build_rows_payload(df_sorted: pd.DataFrame, *, name_col: str,
                        stat_col: str, team_col: str, top_n: int = 10,
                        sport_key: str = 'baseball',
                        teams_df: pd.DataFrame | None = None) -> list[dict]:
    """Convert a sorted top-N DataFrame into the 10-row payload, with
    softball→baseball logo fallback (mirrors top25_render._build_logo_id_map)."""
    if df_sorted.empty:
        return []
    head = df_sorted.head(top_n).copy().reset_index(drop=True)

    logo_map = {}
    if teams_df is not None and len(teams_df):
        bb = teams_df[teams_df['sport'] == 'Baseball']
        bb_map = {n: int(i) for n, i in zip(bb['name'].astype(str), bb['id'])
                   if pd.notna(i)}
        if sport_key.lower() == 'baseball':
            logo_map = bb_map
        else:
            sb = teams_df[teams_df['sport'] == 'Softball']
            for n, i in zip(sb['name'].astype(str), sb['id']):
                if pd.isna(i):
                    continue
                sb_id = int(i)
                if (LOGO_DIR / f'{sb_id}.png').exists():
                    logo_map[n] = sb_id
                elif n in bb_map:
                    logo_map[n] = bb_map[n]

    out = []
    for i, r in head.iterrows():
        team_name = str(r.get(team_col, '') or '').strip()
        logo_id = logo_map.get(team_name)
        logo_b64 = None
        if logo_id is not None:
            p = LOGO_DIR / f'{int(logo_id)}.png'
            if p.exists():
                logo_b64 = _b64(p)
        out.append({
            'rank': i + 1,
            'player': str(r.get(name_col, '') or '').strip(),
            'team': team_name,
            'stat': r.get(stat_col),
            'logo_b64': logo_b64,
        })
    return out
