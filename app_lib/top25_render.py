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
BACKGROUND = _APP_DIR / 'assets' / 'branding' / 'top25_background.png'


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
            f'font-family="Inter,sans-serif" font-weight="700" font-size="38" '
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
    """Build the full 1080×1350 SVG. Uses pre-rendered background PNG with
    baked-in frame, side rails, headline, and 'Presented by 64 Analytics'
    lockup. Only the dynamic sport pill is overlaid.

    `teams` should be 25 dicts with keys: rank, name, record, logo_b64."""
    W, H = 1080, 1350

    # Pad team list to 25 if shorter
    while len(teams) < 25:
        teams.append({'rank': len(teams) + 1, 'name': '—', 'record': '', 'logo_b64': None})
    teams = teams[:25]

    # Layout — the background art reserves the top ~28% (≈ 380 px) for the
    # baked-in pill / headline / presented-by lockup. Tiles fill the rest.
    GRID_TOP = 380
    GRID_BOTTOM = 50
    GAP = 14
    cols, rows = 5, 5
    avail_h = H - GRID_TOP - GRID_BOTTOM
    tile_size = (avail_h - (rows - 1) * GAP) / rows
    grid_w = cols * tile_size + (cols - 1) * GAP
    GRID_LEFT = (W - grid_w) / 2

    bg_b64 = _b64(BACKGROUND) or ''

    parts = [
        f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink">',
        '<defs>'
        # Paper-tile dot pattern (used by tile renderer)
        '<pattern id="paperDots" x="0" y="0" width="6" height="6" patternUnits="userSpaceOnUse">'
        '<circle cx="3" cy="3" r="0.4" fill="#000" opacity="0.06"/>'
        '</pattern>'
        '</defs>',
    ]

    # Background — pre-rendered grunge artwork with header text baked in.
    if bg_b64:
        parts.append(
            f'<image href="{bg_b64}" xlink:href="{bg_b64}" '
            f'x="0" y="0" width="{W}" height="{H}" '
            f'preserveAspectRatio="xMidYMid slice"/>'
        )
    else:
        parts.append(f'<rect width="{W}" height="{H}" fill="#0d0d0d"/>')

    # Sport pill — covers the baked-in "D3 BASEBALL" white text at y=96-146.
    # Drawn as a red rectangle that blends into the surrounding angled red
    # banner; fresh white text is drawn on top.
    header_x_center = W / 2
    pill_text = f'{division.upper()} {sport.upper()}'.strip()
    pill_w = 380
    pill_h = 60
    pill_y = 92
    pill_x = header_x_center - pill_w / 2
    parts.append(
        f'<rect x="{pill_x}" y="{pill_y}" width="{pill_w}" height="{pill_h}" '
        f'fill="#c8102e"/>'
    )
    parts.append(
        f'<text x="{header_x_center}" y="{pill_y + pill_h*0.72}" '
        f'text-anchor="middle" font-family="Inter,sans-serif" '
        f'font-weight="700" font-size="40" letter-spacing="1.6" '
        f'fill="#fff">{_xe(pill_text)}</text>'
    )

    # Optional week/date subtitle — fits between "PRESENTED BY 64 ANALYTICS"
    # (ends ~y=355) and the tile grid (starts y=380).
    if week_label or date_label:
        sub = ' · '.join(s for s in [week_label, date_label] if s)
        parts.append(
            f'<text x="{header_x_center}" y="372" '
            f'text-anchor="middle" font-family="Inter,sans-serif" '
            f'font-weight="700" font-size="11" letter-spacing="2.6" '
            f'fill="#fff" fill-opacity="0.62">{_xe(sub.upper())}</text>'
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


def _build_logo_id_map(teams_df: pd.DataFrame, sport_key: str) -> dict:
    """team_name -> logo_id. Logos in team_logos_512/ are named by the
    BASEBALL team_id; softball entries usually don't have their own file.
    For softball: prefer the SB-specific logo if it exists, otherwise fall
    back to the baseball id for the same-named school. Mirrors the pattern
    in app_lib/ncaat_resume_data.py."""
    bb_rows = teams_df[teams_df['sport'] == 'Baseball']
    bb_map = {n: int(i) for n, i in zip(bb_rows['name'].astype(str), bb_rows['id'])
              if pd.notna(i)}
    if sport_key.lower() == 'baseball':
        return bb_map
    sb_rows = teams_df[teams_df['sport'] == 'Softball']
    out = {}
    for n, i in zip(sb_rows['name'].astype(str), sb_rows['id']):
        if pd.isna(i):
            continue
        sb_id = int(i)
        if (LOGO_DIR / f'{sb_id}.png').exists():
            out[n] = sb_id
        elif n in bb_map:
            out[n] = bb_map[n]
    return out


def build_teams_payload(rpi_df: pd.DataFrame, teams_df: pd.DataFrame,
                         top_n: int = 25, sport_key: str = 'baseball') -> list[dict]:
    """Convert a sport/div ranked DataFrame into the payload `build_top25_svg`
    expects. `teams_df` should be the full teams.csv (both sports) so the
    softball→baseball logo fallback works."""
    if rpi_df.empty:
        return []
    df = rpi_df.head(top_n).copy()
    name_to_logo_id = _build_logo_id_map(teams_df, sport_key)
    out = []
    for _, r in df.iterrows():
        name = str(r.get('teamName', '')).strip()
        tid = name_to_logo_id.get(name)
        out.append({
            'rank': int(r.get('rank', len(out) + 1)),
            'name': name,
            'record': str(r.get('record', '')),
            'logo_b64': fetch_team_logo_b64(tid),
        })
    return out
