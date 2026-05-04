"""Weekly Awards 'Top 10' graphic — 1080×1080 SVG renderer.

Mirrors the Claude-Design hand-off `Top 10 Pitchers Graphic.html`:
- 1080x1080 black canvas with grunge texture
- Top tagline rail + red brushstroke slash + WEEK | DATE tag
- Left column: huge "TOP 10" italic headline, sport-division-position
  subhead, PRESENTED BY 64 Analytics lockup, then 10 numbered rows
  (red chevron-tab + dark pill with logo / name / team / stat)
- Right column: hero image panel (caller supplies a base64 PNG)
- Bottom: 6-cell stat strip (label / value / leader)
- Bottom tagline rail + decorative baseball seam in bottom-left

Public API:
  build_weekly_awards_svg(rows, top_stats, *, sport, division,
                          stat_type, week_label, headline_sub,
                          stat_suffix, stat_decimals, hero_b64) -> str
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd

_APP_DIR = Path(__file__).resolve().parent.parent
LOGO_DIR = _APP_DIR / 'team_logos_512'
EMBLEM_WHITE = _APP_DIR / 'assets' / 'branding' / 'emblem.png'

W = 1080
H = 1080


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


def _row_svg(row: dict, idx: int, x: float, y: float, width: float,
              height: float, stat_decimals: int, stat_suffix: str,
              show_team: bool) -> str:
    """One ranked row: red chevron tab on the left + dark pill with
    logo, player name, team subline, and stat value on the right."""
    rank = row.get('rank', idx + 1)
    player = row.get('player', '')
    team = row.get('team', '')
    stat_val = row.get('stat')
    logo_b64 = row.get('logo_b64')

    tab_w = 46
    chevron = 12  # arrow-tail length
    pill_x = x + tab_w - 8  # negative margin to overlap flag
    pill_w = width - tab_w + 8
    pill_h = height
    pill_radius = pill_h / 2

    parts = [f'<g transform="translate({x},{y})">']

    # Red chevron tab: rectangle on left with right-pointing arrow
    parts.append(
        f'<polygon points="0,0 {tab_w},0 {tab_w - chevron},{height/2:.1f} '
        f'{tab_w},{height} 0,{height}" fill="#d72638"/>'
    )
    parts.append(
        f'<text x="{(tab_w - chevron + 2)/2:.1f}" y="{height*0.68:.1f}" '
        f'text-anchor="middle" font-family="Barlow Condensed,Oswald,sans-serif" '
        f'font-style="italic" font-weight="900" font-size="22" '
        f'fill="#ffffff">{_xe(rank)}</text>'
    )

    # Pill: rounded right end, square left end (the chevron sits to the left)
    px_local = tab_w - 8
    parts.append(
        f'<path d="M {px_local} 0 H {width - pill_radius} '
        f'A {pill_radius} {pill_radius} 0 0 1 {width} {pill_radius} '
        f'A {pill_radius} {pill_radius} 0 0 1 {width - pill_radius} {pill_h} '
        f'H {px_local} Z" fill="#15151a" '
        f'stroke="rgba(255,255,255,0.10)" stroke-width="0.6"/>'
    )

    # Logo area
    logo_size = 30
    logo_x = px_local + 6
    logo_y = (pill_h - logo_size) / 2
    if logo_b64:
        parts.append(
            f'<image href="{logo_b64}" xlink:href="{logo_b64}" '
            f'x="{logo_x}" y="{logo_y}" width="{logo_size}" height="{logo_size}" '
            f'preserveAspectRatio="xMidYMid meet"/>'
        )
    else:
        parts.append(
            f'<circle cx="{logo_x + logo_size/2:.1f}" cy="{pill_h/2:.1f}" '
            f'r="{logo_size/2}" fill="rgba(255,255,255,0.08)"/>'
            f'<text x="{logo_x + logo_size/2:.1f}" y="{pill_h/2 + 4:.1f}" '
            f'text-anchor="middle" font-family="Barlow Condensed,sans-serif" '
            f'font-style="italic" font-weight="800" font-size="11" '
            f'fill="rgba(255,255,255,0.7)">{_xe(_initials(player or team))}</text>'
        )

    # Name + team subline
    name_x = logo_x + logo_size + 8
    if show_team and team:
        parts.append(
            f'<text x="{name_x}" y="{pill_h/2 - 1:.1f}" text-anchor="start" '
            f'font-family="Barlow Condensed,sans-serif" font-style="italic" '
            f'font-weight="700" font-size="15" fill="#ffffff" '
            f'letter-spacing="0.4">{_xe(player.upper())}</text>'
        )
        parts.append(
            f'<text x="{name_x}" y="{pill_h/2 + 11:.1f}" text-anchor="start" '
            f'font-family="Oswald,sans-serif" font-weight="500" font-size="9" '
            f'fill="rgba(255,255,255,0.55)" letter-spacing="1.0">'
            f'{_xe(team.upper())}</text>'
        )
    else:
        parts.append(
            f'<text x="{name_x}" y="{pill_h*0.66:.1f}" text-anchor="start" '
            f'font-family="Barlow Condensed,sans-serif" font-style="italic" '
            f'font-weight="700" font-size="16" fill="#ffffff" '
            f'letter-spacing="0.4">{_xe(player.upper())}</text>'
        )

    # Stat value (right side of pill)
    stat_str = _fmt(stat_val, stat_decimals)
    if stat_suffix:
        stat_str = f'{stat_str} {stat_suffix}'
    parts.append(
        f'<text x="{width - 12}" y="{pill_h*0.66:.1f}" text-anchor="end" '
        f'font-family="Barlow Condensed,sans-serif" font-style="italic" '
        f'font-weight="800" font-size="17" fill="#ffffff" '
        f'letter-spacing="0.3">{_xe(stat_str)}</text>'
    )

    parts.append('</g>')
    return ''.join(parts)


def _stat_cell_svg(stat: dict, x: float, y: float, width: float, height: float,
                    is_last: bool) -> str:
    """One bottom-stat cell with label / value / leader."""
    label = stat.get('label', '')
    value = stat.get('value')
    decimals = stat.get('decimals', 0)
    leader = stat.get('leader', '')

    parts = [f'<g transform="translate({x},{y})">']
    if not is_last:
        parts.append(
            f'<line x1="{width}" y1="6" x2="{width}" y2="{height-6}" '
            f'stroke="rgba(255,255,255,0.10)" stroke-width="1"/>'
        )
    cx = width / 2
    parts.append(
        f'<text x="{cx}" y="22" text-anchor="middle" '
        f'font-family="Oswald,sans-serif" font-weight="700" font-size="13" '
        f'letter-spacing="2.4" fill="#d72638">{_xe(label.upper())}</text>'
    )
    parts.append(
        f'<text x="{cx}" y="62" text-anchor="middle" '
        f'font-family="Barlow Condensed,sans-serif" font-style="italic" '
        f'font-weight="800" font-size="32" fill="#ffffff">{_xe(_fmt(value, decimals))}</text>'
    )
    if leader:
        parts.append(
            f'<text x="{cx}" y="82" text-anchor="middle" '
            f'font-family="Oswald,sans-serif" font-weight="500" font-size="9" '
            f'letter-spacing="1.0" fill="rgba(255,255,255,0.55)">'
            f'{_xe(leader.upper())}</text>'
        )
    parts.append('</g>')
    return ''.join(parts)


def build_weekly_awards_svg(rows: list[dict], top_stats: list[dict], *,
                             sport: str, division: str, stat_type: str,
                             week_label: str = '',
                             headline_top: str = 'TOP 10',
                             headline_sub: str = '',
                             stat_suffix: str = '',
                             stat_decimals: int = 2,
                             show_team_subline: bool = True,
                             rail_text: str = ('64 ANALYTICS IS ONE OF THE '
                                                'INDUSTRY LEADERS IN COLLEGE '
                                                'SPORTS ANALYTICS'),
                             hero_b64: str | None = None) -> str:
    """Build the 1080x1080 SVG. `rows` should be 10 dicts with keys:
    rank, player, team, stat, logo_b64. `top_stats` should be 6 dicts with
    keys: label, value, decimals, leader."""
    # Pad rows to exactly 10
    while len(rows) < 10:
        rows.append({'rank': len(rows) + 1, 'player': '—', 'team': '',
                      'stat': None, 'logo_b64': None})
    rows = rows[:10]

    # Pad/trim stats to 6
    while len(top_stats) < 6:
        top_stats.append({'label': '', 'value': None, 'decimals': 0, 'leader': ''})
    top_stats = top_stats[:6]

    emblem_b64 = _b64(EMBLEM_WHITE) or ''

    # ── Layout constants (mirrors the HTML prototype) ──
    HEADLINE_X = 80
    HEADLINE_Y = 142
    HEADLINE_W = 380
    RANK_X = 80
    RANK_Y = 392
    RANK_W = 380
    RANK_H = 40
    RANK_GAP = 6
    HERO_X = 500
    HERO_Y = 142
    HERO_W = W - HERO_X - 80
    HERO_H = 720
    STATS_X = 500
    STATS_Y = H - 70 - 130
    STATS_W = W - STATS_X - 80
    STATS_H = 130
    STATS_HEADER_H = 22
    RAIL_TOP_Y = 24
    RAIL_BOTTOM_Y = H - 36

    parts = [
        f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink">',
        '<defs>'
        # Brushstroke gradient
        '<linearGradient id="brush" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0" stop-color="#d72638" stop-opacity="0"/>'
        '<stop offset="0.2" stop-color="#d72638" stop-opacity="0.95"/>'
        '<stop offset="0.55" stop-color="#a01827" stop-opacity="0.85"/>'
        '<stop offset="0.8" stop-color="#d72638" stop-opacity="0.4"/>'
        '<stop offset="1" stop-color="#d72638" stop-opacity="0"/>'
        '</linearGradient>'
        # Stats-bar header gradient
        '<linearGradient id="statsHdr" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0" stop-color="#d72638" stop-opacity="0.16"/>'
        '<stop offset="0.6" stop-color="#d72638" stop-opacity="0"/>'
        '</linearGradient>'
        # Baseball-seam gradient
        '<radialGradient id="ballGrad" cx="40%" cy="40%" r="60%">'
        '<stop offset="0" stop-color="#1c1c1f"/>'
        '<stop offset="1" stop-color="#08080a"/>'
        '</radialGradient>'
        '</defs>',
        # Background
        f'<rect width="{W}" height="{H}" fill="#08080a"/>',
        # Subtle radial highlights (replaces CSS grunge)
        f'<rect width="{W}" height="{H}" fill="url(#statsHdr)" opacity="0.0"/>',
    ]

    # ── Decorative corner strokes ──
    parts.append(
        '<g opacity="0.65" fill="#d72638">'
        '<polygon points="0,0 90,0 30,60 0,60"/>'
        '<polygon points="0,80 70,80 20,130 0,130" opacity="0.5"/>'
        '</g>'
        '<g opacity="0.6" fill="#d72638">'
        f'<polygon points="{W},{H-80} {W},{H} {W-100},{H}"/>'
        f'<polygon points="{W-90},{H} {W-30},{H-60} {W},{H-60} {W},{H}" '
        'opacity="0.55"/>'
        '</g>'
    )

    # ── Top rail tagline ──
    parts.append(
        f'<text x="{W/2}" y="{RAIL_TOP_Y + 13}" text-anchor="middle" '
        f'font-family="Oswald,sans-serif" font-weight="600" font-size="13" '
        f'letter-spacing="5.4" fill="#ffffff" fill-opacity="0.85">'
        f'{_xe(rail_text.upper())}</text>'
    )

    # ── Top brushstroke + WEEK tag (right-aligned) ──
    bar_y = 56
    bar_h = 56
    bar_left = 60
    bar_right = W - 60
    bar_w = bar_right - bar_left
    parts.append(
        f'<g transform="translate({bar_left},{bar_y})">'
        # Brushstroke main bar
        f'<polygon points="0,12 {bar_w},12 {bar_w-15},{bar_h-0} 15,{bar_h-0}" '
        f'fill="url(#brush)"/>'
        # Splatter shapes
        '<polygon points="60,8 130,8 100,28 30,28" fill="#d72638" opacity="0.85"/>'
        '<polygon points="180,4 320,4 280,30 140,30" fill="#a01827" opacity="0.9"/>'
        '<polygon points="350,8 460,8 430,28 320,28" fill="#d72638" opacity="0.7"/>'
        # Drop shadow under bar
        f'<polygon points="20,{bar_h} {bar_w-30},{bar_h} {bar_w-60},{bar_h+12} 50,{bar_h+12}" '
        'fill="#3a0a12" opacity="0.7"/>'
        '</g>'
    )

    # WEEK tag (right edge of bar). Slanted parallelogram with red left edge.
    if week_label:
        wt_h = 36
        wt_pad_x = 22
        # Approximate width = chars * 12 (Barlow Condensed avg)
        wt_text_w = max(220, len(week_label) * 11 + 24)
        wt_x = bar_right - wt_text_w - 6
        wt_y = bar_y + (bar_h - wt_h) / 2 + 2
        parts.append(
            f'<polygon points="{wt_x + 14},{wt_y} '
            f'{wt_x + wt_text_w},{wt_y} '
            f'{wt_x + wt_text_w},{wt_y + wt_h} '
            f'{wt_x},{wt_y + wt_h}" fill="#0a0a0c"/>'
            f'<line x1="{wt_x + 14}" y1="{wt_y}" '
            f'x2="{wt_x}" y2="{wt_y + wt_h}" '
            f'stroke="#d72638" stroke-width="3"/>'
            f'<text x="{wt_x + wt_text_w - wt_pad_x}" y="{wt_y + wt_h*0.68:.1f}" '
            f'text-anchor="end" font-family="Barlow Condensed,sans-serif" '
            f'font-style="italic" font-weight="800" font-size="22" '
            f'letter-spacing="2.6" fill="#ffffff">{_xe(week_label.upper())}</text>'
        )

    # ── Headline block (left) ──
    parts.append(
        f'<text x="{HEADLINE_X}" y="{HEADLINE_Y + 92}" text-anchor="start" '
        f'font-family="Barlow Condensed,sans-serif" font-style="italic" '
        f'font-weight="900" font-size="102" fill="#f0f0f0" '
        f'letter-spacing="-0.5" filter="url(#dropShadow)">'
        f'{_xe(headline_top.upper())}</text>'
    )
    if headline_sub:
        parts.append(
            f'<text x="{HEADLINE_X}" y="{HEADLINE_Y + 130}" text-anchor="start" '
            f'font-family="Barlow Condensed,sans-serif" font-style="italic" '
            f'font-weight="800" font-size="32" fill="#ffffff" '
            f'letter-spacing="0.3">{_xe(headline_sub.upper())}</text>'
        )
    # PRESENTED BY emblem + text
    pres_y = HEADLINE_Y + 160
    if emblem_b64:
        parts.append(
            f'<image href="{emblem_b64}" xlink:href="{emblem_b64}" '
            f'x="{HEADLINE_X}" y="{pres_y}" width="36" height="36" '
            f'preserveAspectRatio="xMidYMid meet"/>'
        )
    parts.append(
        f'<text x="{HEADLINE_X + 46}" y="{pres_y + 14}" text-anchor="start" '
        f'font-family="Oswald,sans-serif" font-weight="700" font-size="10" '
        f'letter-spacing="2.8" fill="#ffffff" fill-opacity="0.9">PRESENTED BY</text>'
    )
    parts.append(
        f'<text x="{HEADLINE_X + 46}" y="{pres_y + 32}" text-anchor="start" '
        f'font-family="Barlow Condensed,sans-serif" font-style="italic" '
        f'font-weight="800" font-size="22" letter-spacing="0.4" fill="#d72638">'
        f'64 ANALYTICS</text>'
    )

    # ── Hero panel (right) — image or empty placeholder ──
    parts.append(
        f'<g transform="translate({HERO_X},{HERO_Y})">'
        f'<rect width="{HERO_W}" height="{HERO_H}" fill="#0c0c0e" '
        f'stroke="rgba(255,255,255,0.18)" stroke-width="1"/>'
    )
    if hero_b64:
        parts.append(
            f'<image href="{hero_b64}" xlink:href="{hero_b64}" '
            f'x="0" y="0" width="{HERO_W}" height="{HERO_H}" '
            f'preserveAspectRatio="xMidYMid slice"/>'
        )
    else:
        # Diagonal stripe fill — empty hint
        for i in range(-int(HERO_H), int(HERO_W) + int(HERO_H), 48):
            parts.append(
                f'<polygon points="{i},0 {i+24},0 {i+24+HERO_H},{HERO_H} '
                f'{i+HERO_H},{HERO_H}" fill="#131316"/>'
            )
        parts.append(
            f'<text x="{HERO_W/2}" y="{HERO_H/2}" text-anchor="middle" '
            f'font-family="JetBrains Mono,monospace" font-size="12" '
            f'letter-spacing="2.2" fill="rgba(255,255,255,0.45)">'
            f'DROP HERO IMAGE HERE</text>'
        )
    parts.append('</g>')

    # Hero corner emblem (top-right of canvas)
    if emblem_b64:
        parts.append(
            f'<image href="{emblem_b64}" xlink:href="{emblem_b64}" '
            f'x="{W - 100}" y="{HERO_Y - 38}" width="96" height="96" '
            f'preserveAspectRatio="xMidYMid meet"/>'
        )

    # ── Rank list (10 rows) ──
    for i, row in enumerate(rows):
        ry = RANK_Y + i * (RANK_H + RANK_GAP)
        parts.append(_row_svg(row, i, RANK_X, ry, RANK_W, RANK_H,
                                stat_decimals, stat_suffix, show_team_subline))

    # ── Bottom stats bar ──
    parts.append(
        f'<g transform="translate({STATS_X},{STATS_Y})">'
        f'<rect width="{STATS_W}" height="{STATS_H}" fill="#0a0a0c" '
        f'stroke="rgba(255,255,255,0.22)" stroke-width="1"/>'
        # Header strip
        f'<rect width="{STATS_W}" height="{STATS_HEADER_H}" '
        f'fill="url(#statsHdr)"/>'
        f'<line x1="0" y1="{STATS_HEADER_H}" x2="{STATS_W}" y2="{STATS_HEADER_H}" '
        f'stroke="#d72638" stroke-width="1"/>'
        f'<text x="14" y="15" text-anchor="start" '
        f'font-family="Oswald,sans-serif" font-weight="600" font-size="10" '
        f'letter-spacing="3.2" fill="rgba(255,255,255,0.7)">TOP STATS</text>'
        f'</g>'
    )
    # 6 stat cells
    cell_w = STATS_W / 6
    cell_h = STATS_H - STATS_HEADER_H
    for i, s in enumerate(top_stats):
        cx = STATS_X + i * cell_w
        cy = STATS_Y + STATS_HEADER_H
        parts.append(_stat_cell_svg(s, cx, cy, cell_w, cell_h, i == 5))

    # ── Decorative baseball seam (bottom-left) ──
    parts.append(
        f'<g transform="translate(-80,{H - 220})" opacity="0.4">'
        f'<circle cx="100" cy="100" r="96" fill="url(#ballGrad)" '
        f'stroke="rgba(255,255,255,0.08)" stroke-width="1"/>'
        f'<path d="M 14 120 Q 80 80 186 130" stroke="#d72638" '
        f'stroke-width="2" fill="none" opacity="0.95"/>'
        f'<g stroke="#d72638" stroke-width="1.4" opacity="0.9">'
        f'<line x1="22" y1="118" x2="28" y2="108"/>'
        f'<line x1="40" y1="108" x2="46" y2="98"/>'
        f'<line x1="60" y1="100" x2="66" y2="90"/>'
        f'<line x1="82" y1="92" x2="88" y2="82"/>'
        f'<line x1="106" y1="90" x2="112" y2="80"/>'
        f'<line x1="130" y1="94" x2="136" y2="84"/>'
        f'<line x1="154" y1="106" x2="160" y2="96"/>'
        f'<line x1="172" y1="118" x2="178" y2="108"/>'
        f'</g>'
        f'<path d="M 18 80 Q 100 130 188 70" stroke="#d72638" '
        f'stroke-width="1.5" fill="none" opacity="0.5"/>'
        f'</g>'
    )

    # ── Bottom rail ──
    parts.append(
        f'<text x="{W/2}" y="{RAIL_BOTTOM_Y}" text-anchor="middle" '
        f'font-family="Oswald,sans-serif" font-weight="600" font-size="13" '
        f'letter-spacing="5.4" fill="#ffffff" fill-opacity="0.85">'
        f'{_xe(rail_text.upper())}</text>'
    )

    parts.append('</svg>')
    return ''.join(parts)


def build_rows_payload(df_sorted: pd.DataFrame, *, name_col: str,
                        stat_col: str, team_col: str, top_n: int = 10,
                        sport_key: str = 'baseball',
                        teams_df: pd.DataFrame | None = None) -> list[dict]:
    """Convert a sorted top-N DataFrame into the 10-row payload, with
    softball→baseball logo fallback (mirrors top25_render._build_logo_id_map).
    """
    if df_sorted.empty:
        return []
    head = df_sorted.head(top_n).copy().reset_index(drop=True)

    # Logo id map: name -> logo_id (BB id with SB fallback to BB id of same school)
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
