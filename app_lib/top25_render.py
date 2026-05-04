"""Top-25 weekly rankings share graphic — 1080×1350 SVG.

Mirrors the V1 'Faithful match' design from the Claude Design hand-off:
red gradient frame, dark interior with angular shards + glow, vertical
side rails repeating the brand tagline, big OSWALD headline, 5x5 tile
grid with red rank tabs.

Data source: any per-team ranked DataFrame with at least these columns:
  rank · teamName · record (optional, blank if missing)
The tile renderer uses the team_logos_512 directory via team_id_64a if
the caller passes it; otherwise the team name's first 3 letters render
as a monogram.
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd

_APP_DIR = Path(__file__).resolve().parent.parent
LOGO_DIR = _APP_DIR / 'team_logos_512'
EMBLEM = _APP_DIR / 'assets' / 'branding' / 'emblem.png'


def _xe(s):
    return html.escape(str(s) if s is not None else '')


def _b64(p: Path) -> str | None:
    try:
        return f'data:image/png;base64,{base64.b64encode(p.read_bytes()).decode("ascii")}'
    except Exception:
        return None


def _team_short(name: str) -> str:
    if not name:
        return '—'
    parts = ''.join(c for c in name if c.isalpha() or c == ' ').split()
    if len(parts) == 1:
        return parts[0][:3].upper()
    return ''.join(p[0] for p in parts[:3]).upper()


def _tile_svg(team: dict, idx: int, x: float, y: float, size: float) -> str:
    """One paper tile (~size × size). Rank tab top-left, logo center."""
    rank = team.get('rank', idx + 1)
    name = team.get('name', '—')
    record = team.get('record', '')
    logo_b64 = team.get('logo_b64')

    rank_text = f'{int(rank):02d}'

    # Tile background — paper texture (SVG can't do CSS radial-gradient easily;
    # use a pale rect + faint dot pattern for the eggshell-paper feel).
    tile = (
        f'<g transform="translate({x:.1f},{y:.1f})">'
        # Drop shadow
        f'<rect x="2" y="6" width="{size:.1f}" height="{size:.1f}" rx="14" '
        f'fill="rgba(0,0,0,0.45)"/>'
        # Paper body
        f'<rect width="{size:.1f}" height="{size:.1f}" rx="14" fill="#f3efe7"/>'
        # Subtle dot grid
        f'<rect width="{size:.1f}" height="{size:.1f}" rx="14" '
        f'fill="url(#paperDots)"/>'
    )

    # Logo or monogram
    if logo_b64:
        logo_size = size * 0.62
        lx = (size - logo_size) / 2
        ly = (size - logo_size) / 2
        tile += (
            f'<image href="{logo_b64}" xlink:href="{logo_b64}" '
            f'x="{lx:.1f}" y="{ly:.1f}" width="{logo_size:.1f}" height="{logo_size:.1f}" '
            f'preserveAspectRatio="xMidYMid meet"/>'
        )
    else:
        mono = _team_short(name)
        tile += (
            f'<text x="{size/2:.1f}" y="{size/2 + 12:.1f}" text-anchor="middle" '
            f'font-family="Oswald,sans-serif" font-weight="700" font-size="38" '
            f'fill="#1a1a1a" opacity="0.65">{_xe(mono)}</text>'
        )

    # Rank tab top-left (angled red tab)
    tab_w = 40
    tab_h = 22
    tile += (
        f'<polygon points="0,0 {tab_w},0 {tab_w-6},{tab_h} 0,{tab_h}" '
        f'fill="#9E1B32"/>'
        f'<text x="{tab_w/2 - 3}" y="{tab_h - 6}" text-anchor="middle" '
        f'font-family="Inter,sans-serif" font-weight="800" font-size="13" '
        f'fill="#ffffff" letter-spacing="0.6">{rank_text}</text>'
    )

    # Record at the bottom of the tile (below the logo) so users get context
    if record:
        tile += (
            f'<text x="{size/2:.1f}" y="{size - 10:.1f}" text-anchor="middle" '
            f'font-family="Inter,sans-serif" font-weight="700" font-size="11" '
            f'fill="#9E1B32" letter-spacing="0.8">{_xe(record)}</text>'
        )

    tile += '</g>'
    return tile


def build_top25_svg(teams: list[dict], sport: str, division: str,
                     week_label: str = '', date_label: str = '') -> str:
    """Build the full 1080×1350 SVG. `teams` should be 25 dicts with
    keys: rank, name, record, logo_b64 (base64 data URL or None)."""
    W, H = 1080, 1350

    # Pad team list to 25 if shorter
    while len(teams) < 25:
        teams.append({'rank': len(teams) + 1, 'name': '—', 'record': '', 'logo_b64': None})
    teams = teams[:25]

    rail_text = ('64 ANALYTICS IS ONE OF THE INDUSTRY LEADERS '
                 'IN COLLEGE SPORTS ANALYTICS · ') * 4

    # Layout
    GRID_LEFT = 64
    GRID_RIGHT = 64
    GRID_TOP = 480     # below header
    GRID_BOTTOM = 80
    grid_w = W - GRID_LEFT - GRID_RIGHT
    GAP = 14
    cols, rows = 5, 5
    tile_size = (grid_w - (cols - 1) * GAP) / cols  # equals (grid_h - 4*GAP)/5 if proportional

    # Recompute grid_h to match square tiles
    grid_h = rows * tile_size + (rows - 1) * GAP
    GRID_TOP = H - GRID_BOTTOM - grid_h - 30

    emblem_b64 = _b64(EMBLEM) or ''

    # --- SVG body ---
    parts = [
        f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink">',
        # Definitions: gradients, paper-dot pattern
        '<defs>'
        # Frame gradient
        '<linearGradient id="frameGrad" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#5a0a18"/>'
        '<stop offset="0.3" stop-color="#9E1B32"/>'
        '<stop offset="0.7" stop-color="#9E1B32"/>'
        '<stop offset="1" stop-color="#5a0a18"/>'
        '</linearGradient>'
        # Frame mask cuts out interior
        '<mask id="frameMask">'
        f'<rect width="{W}" height="{H}" fill="white"/>'
        f'<rect x="44" y="44" width="{W-88}" height="{H-88}" fill="black"/>'
        '</mask>'
        # Top glow
        '<radialGradient id="glowTop" cx="0.5" cy="0" r="0.6">'
        '<stop offset="0" stop-color="#9E1B32" stop-opacity="0.55"/>'
        '<stop offset="1" stop-color="#0d0d0d" stop-opacity="0"/>'
        '</radialGradient>'
        # Bottom glow
        '<radialGradient id="glowBot" cx="0.5" cy="1" r="0.6">'
        '<stop offset="0" stop-color="#9E1B32" stop-opacity="0.45"/>'
        '<stop offset="1" stop-color="#0d0d0d" stop-opacity="0"/>'
        '</radialGradient>'
        # Paper-tile dot pattern
        '<pattern id="paperDots" x="0" y="0" width="6" height="6" patternUnits="userSpaceOnUse">'
        '<circle cx="3" cy="3" r="0.4" fill="#000" opacity="0.06"/>'
        '</pattern>'
        '</defs>',
        # Background
        f'<rect width="{W}" height="{H}" fill="#0d0d0d"/>',
        # Top + bottom glow
        f'<rect width="{W}" height="{H}" fill="url(#glowTop)"/>',
        f'<rect width="{W}" height="{H}" fill="url(#glowBot)"/>',
        # Angular shards
        '<g opacity="0.85">'
        '<polygon points="44,74 174,74 114,254 44,224" fill="#9E1B32" opacity="0.42"/>'
        '<polygon points="1036,74 906,74 966,254 1036,224" fill="#9E1B32" opacity="0.42"/>'
        '<polygon points="164,104 264,74 314,244 214,264" fill="#9E1B32" opacity="0.20"/>'
        '<polygon points="816,74 916,104 866,264 766,244" fill="#9E1B32" opacity="0.20"/>'
        '<polygon points="44,1124 134,1144 104,1306 44,1306" fill="#9E1B32" opacity="0.36"/>'
        '<polygon points="1036,1124 946,1144 976,1306 1036,1306" fill="#9E1B32" opacity="0.36"/>'
        '</g>',
        # Red frame
        f'<rect width="{W}" height="{H}" fill="url(#frameGrad)" mask="url(#frameMask)"/>',
    ]

    # Side rails — vertical text repeating tagline
    # SVG doesn't easily support vertical text layout; approximate by rotating
    # a single line. Two side rails, one each side.
    parts.append(
        f'<g transform="translate(28 {H/2}) rotate(-90)">'
        f'<text text-anchor="middle" font-family="Inter,sans-serif" '
        f'font-weight="800" font-size="11" letter-spacing="3.6" '
        f'fill="#fff" fill-opacity="0.85">{_xe(rail_text)}</text>'
        f'</g>'
    )
    parts.append(
        f'<g transform="translate({W-28} {H/2}) rotate(90)">'
        f'<text text-anchor="middle" font-family="Inter,sans-serif" '
        f'font-weight="800" font-size="11" letter-spacing="3.6" '
        f'fill="#fff" fill-opacity="0.85">{_xe(rail_text)}</text>'
        f'</g>'
    )

    # Header — sport pill (skewed) + headline + presented by
    header_x_center = W / 2

    pill_text = f'{division.upper()} {sport.upper()}'
    pill_w = max(360, len(pill_text) * 22)
    pill_h = 60
    pill_y = 100
    pill_x = header_x_center - pill_w / 2
    # Skewed parallelogram-ish red pill
    pill_pts = (
        f'{pill_x + pill_w*0.03},{pill_y} '
        f'{pill_x + pill_w*0.97},{pill_y} '
        f'{pill_x + pill_w},{pill_y + pill_h} '
        f'{pill_x},{pill_y + pill_h}'
    )
    parts.append(f'<polygon points="{pill_pts}" fill="#9E1B32"/>')
    parts.append(
        f'<text x="{header_x_center}" y="{pill_y + pill_h*0.7}" '
        f'text-anchor="middle" font-family="Oswald,sans-serif" '
        f'font-weight="700" font-size="38" letter-spacing="1.5" '
        f'fill="#fff">{_xe(pill_text)}</text>'
    )

    # Headline
    parts.append(
        f'<text x="{header_x_center}" y="{pill_y + pill_h + 100}" '
        f'text-anchor="middle" font-family="Oswald,sans-serif" '
        f'font-weight="700" font-size="86" letter-spacing="0.4" '
        f'fill="#fff">TOP 25 RANKINGS</text>'
    )

    # Presented by emblem + text
    pres_y = pill_y + pill_h + 130
    if emblem_b64:
        parts.append(
            f'<image href="{emblem_b64}" xlink:href="{emblem_b64}" '
            f'x="{header_x_center - 130}" y="{pres_y}" '
            f'width="64" height="64" preserveAspectRatio="xMidYMid meet"/>'
        )
    parts.append(
        f'<text x="{header_x_center - 50}" y="{pres_y + 24}" '
        f'text-anchor="start" font-family="Inter,sans-serif" '
        f'font-weight="700" font-size="13" letter-spacing="3.6" '
        f'fill="#fff">PRESENTED BY</text>'
    )
    parts.append(
        f'<text x="{header_x_center - 50}" y="{pres_y + 56}" '
        f'text-anchor="start" font-family="Oswald,sans-serif" '
        f'font-weight="700" font-size="34" letter-spacing="1" '
        f'fill="#fff">64 ANALYTICS</text>'
    )

    # Optional week/date subtitle
    if week_label or date_label:
        sub = ' · '.join(s for s in [week_label, date_label] if s)
        parts.append(
            f'<text x="{header_x_center}" y="{pres_y + 100}" '
            f'text-anchor="middle" font-family="Inter,sans-serif" '
            f'font-weight="600" font-size="14" letter-spacing="2.6" '
            f'fill="#bbb">{_xe(sub.upper())}</text>'
        )

    # 5x5 grid of tiles
    for i, team in enumerate(teams):
        col = i % cols
        row = i // cols
        x = GRID_LEFT + col * (tile_size + GAP)
        y = GRID_TOP + row * (tile_size + GAP)
        parts.append(_tile_svg(team, i, x, y, tile_size))

    parts.append('</svg>')
    return ''.join(parts)


def fetch_team_logo_b64(team_id_64a: int | None) -> str | None:
    """Return base64 data URL for a team's logo PNG, or None."""
    if team_id_64a is None:
        return None
    p = LOGO_DIR / f'{int(team_id_64a)}.png'
    if not p.exists():
        return None
    return _b64(p)


def build_teams_payload(rpi_df: pd.DataFrame, teams_df: pd.DataFrame, top_n: int = 25) -> list[dict]:
    """Convert a sport/div RPI DataFrame into the payload `build_top25_svg` expects.
    Joins to teams.csv to fetch the 64A team_id for logo lookup."""
    if rpi_df.empty:
        return []
    df = rpi_df.head(top_n).copy()
    # Build name → 64A id lookup
    name_to_id = dict(zip(teams_df['name'].astype(str), teams_df['id']))
    out = []
    for _, r in df.iterrows():
        name = str(r.get('teamName', '')).strip()
        tid = name_to_id.get(name)
        if pd.isna(tid):
            tid = None
        else:
            try:
                tid = int(tid)
            except (TypeError, ValueError):
                tid = None
        out.append({
            'rank': int(r.get('rank', len(out) + 1)),
            'name': name,
            'record': str(r.get('record', '')),
            'logo_b64': fetch_team_logo_b64(tid),
        })
    return out
