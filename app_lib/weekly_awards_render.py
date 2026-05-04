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
FONTS_DIR = _APP_DIR / 'assets' / 'fonts'


def _register_bundled_fonts() -> None:
    """Make sure cairo/fontconfig (used by cairosvg PNG export) can find
    the bundled TTF files. On Linux, fontconfig auto-scans ~/.fonts; we
    symlink/copy bundled fonts there once per process and then prime the
    cache via fc-cache. Silent no-op on platforms where these tools
    aren't available — on Render this is a one-time cost at startup."""
    try:
        if not FONTS_DIR.exists():
            return
        import os
        if os.name != 'posix':  # Linux/macOS only
            return
        user_fonts = Path.home() / '.fonts'
        user_fonts.mkdir(parents=True, exist_ok=True)
        import shutil
        for f in FONTS_DIR.glob('*.ttf'):
            dst = user_fonts / f.name
            if not dst.exists():
                try:
                    dst.symlink_to(f)
                except Exception:
                    shutil.copy2(f, dst)
        import subprocess
        try:
            subprocess.run(['fc-cache', '-f', str(user_fonts)],
                            timeout=15, capture_output=True, check=False)
        except FileNotFoundError:
            pass
    except Exception:
        pass


_register_bundled_fonts()

W = 1080
H = 1080

# ── Backdrop-derived coordinates (pixel-measured from the grunge art) ──
# Each pill is ~44 px tall, stride 57 px between pill tops, 10 rows starting
# at y=406. Pill body spans x≈140 (after flag chevron) to x≈460.
PILL_X = 140
PILL_RIGHT = 460
PILL_Y0 = 406
PILL_STRIDE = 57
PILL_H = 44

# Hero panel — fills the bordered hexagon panel on the right (x=550-1020,
# y=150-895). A circular cutout at the 64A emblem location keeps the
# baked red circle visible "on top of" the hero image, matching the design.
HERO_X = 550
HERO_Y = 150
HERO_W = 470
HERO_H = 740
# Baked 64A circle: x=873-1079 (clipped at canvas edge), y=100-268.
# Center ~(975, 185), visible radius ~85. Mask a slightly larger disk so
# the circle has a small breathing-room ring.
EMBLEM_CX = 975
EMBLEM_CY = 185
EMBLEM_R = 92

# Bottom stats strip — bordered region with two decorative horizontal
# lines (red at y≈940, white at y≈970). Cells span the full strip width;
# label sits above the red line, value between red and white.
STATS_X = 555
STATS_Y = 915
STATS_W = 460
STATS_H = 90

# Subtitle cover. Baked 'D3 BASEBALL PITCHERS' is centered horizontally
# under 'TOP 10' (both at center_x ≈ 253). We cover the full text band and
# redraw centered at the same axis so the dynamic subtitle aligns with
# the headline above it.
SUB_CX = 253
SUB_Y = 274
SUB_W = 380
SUB_H = 40
SUB_X = SUB_CX - SUB_W // 2

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


def _row_overlay(row: dict, idx: int, *, show_team: bool) -> str:
    """In-pill content: logo (with white highlight ring so dark logos
    aren't lost on the dark pill body) + player name + team subline.
    The backdrop already provides the chevron flag and the pill itself.
    Per-pill rank-stat values are NOT rendered — the user found them
    visually noisy; the bottom strip carries the leader-board values."""
    py = PILL_Y0 + idx * PILL_STRIDE
    pill_cy = py + PILL_H / 2

    parts = []
    logo_size = 30
    logo_x = PILL_X + 6
    logo_y = py + (PILL_H - logo_size) / 2
    logo_cx = logo_x + logo_size / 2

    logo_b64 = row.get('logo_b64')

    # White circular highlight behind every logo so dark/navy crests don't
    # disappear into the pill background. Slight padding around the logo.
    parts.append(
        f'<circle cx="{logo_cx:.1f}" cy="{pill_cy:.1f}" '
        f'r="{logo_size/2 + 2:.1f}" fill="#ffffff" fill-opacity="0.92"/>'
    )

    if logo_b64:
        parts.append(
            f'<image href="{logo_b64}" xlink:href="{logo_b64}" '
            f'x="{logo_x}" y="{logo_y}" width="{logo_size}" height="{logo_size}" '
            f'preserveAspectRatio="xMidYMid meet"/>'
        )
    else:
        parts.append(
            f'<text x="{logo_cx:.1f}" y="{pill_cy + 4.5:.1f}" '
            f'text-anchor="middle" font-family="Barlow Condensed,sans-serif" '
            f'font-style="italic" font-weight="800" font-size="13" '
            f'fill="#1a1a1d">'
            f'{_xe(_initials(row.get("player") or row.get("team") or "?"))}</text>'
        )

    name_x = logo_x + logo_size + 12
    player = (row.get('player') or '').upper()
    team = (row.get('team') or '').upper()

    if show_team and team:
        parts.append(
            f'<text x="{name_x}" y="{pill_cy - 2:.1f}" text-anchor="start" '
            f'font-family="Barlow Condensed,Oswald,sans-serif" font-style="italic" '
            f'font-weight="700" font-size="16" fill="#ffffff" '
            f'letter-spacing="0.4">{_xe(player)}</text>'
        )
        parts.append(
            f'<text x="{name_x}" y="{pill_cy + 12:.1f}" text-anchor="start" '
            f'font-family="Oswald,sans-serif" font-weight="500" font-size="10" '
            f'fill="rgba(255,255,255,0.6)" letter-spacing="1.0">'
            f'{_xe(team)}</text>'
        )
    else:
        parts.append(
            f'<text x="{name_x}" y="{pill_cy + 5:.1f}" text-anchor="start" '
            f'font-family="Barlow Condensed,sans-serif" font-style="italic" '
            f'font-weight="700" font-size="17" fill="#ffffff" '
            f'letter-spacing="0.4">{_xe(player)}</text>'
        )

    return ''.join(parts)


def _stat_cell_overlay(stat: dict, x: float, y: float, width: float, height: float) -> str:
    """One bottom-strip stat cell — drawn inside the bordered strip
    region. Backdrop has decorative red+white horizontal lines at strip-y
    +25 (red) and +55 (white). We place the LABEL above the red line and
    the VALUE between the two lines so the cell sits cleanly within the
    strip bounds. No leader text — the strip is too thin to fit it."""
    label = (stat.get('label') or '').upper()
    value = stat.get('value')
    decimals = stat.get('decimals', 0)
    cx = x + width / 2

    # Strip is 90 px tall starting at strip-top (y). Decorative red line at
    # +25, white line at +55. Place LABEL above the red line and VALUE
    # below the white line so each cell fills the strip vertically and the
    # decorative lines act as a divider band between them.
    return ''.join([
        # Label (red, above the red line)
        f'<text x="{cx}" y="{y + 18}" text-anchor="middle" '
        f'font-family="Oswald,sans-serif" font-weight="700" font-size="13" '
        f'letter-spacing="2.4" fill="#d72638">{_xe(label)}</text>',
        # Value (white, below the white line — bottom zone)
        f'<text x="{cx}" y="{y + 78}" text-anchor="middle" '
        f'font-family="Barlow Condensed,sans-serif" font-style="italic" '
        f'font-weight="800" font-size="26" fill="#ffffff">{_xe(_fmt(value, decimals))}</text>',
    ])


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

    # ── 2. Hero image — fills the bordered panel. A mask punches a hole
    # for the baked 64A red circle so it stays visible above the image. ──
    if hero_b64:
        parts.append(
            '<defs>'
            '<mask id="heroMask">'
            f'<rect x="{HERO_X}" y="{HERO_Y}" width="{HERO_W}" height="{HERO_H}" fill="white"/>'
            f'<circle cx="{EMBLEM_CX}" cy="{EMBLEM_CY}" r="{EMBLEM_R}" fill="black"/>'
            '</mask>'
            '</defs>'
            f'<image href="{hero_b64}" xlink:href="{hero_b64}" '
            f'x="{HERO_X}" y="{HERO_Y}" width="{HERO_W}" height="{HERO_H}" '
            f'preserveAspectRatio="xMidYMid slice" mask="url(#heroMask)"/>'
        )

    # ── 3. Cover the baked 'D3 BASEBALL PITCHERS' subtitle and redraw,
    # center-aligned with TOP 10 (center_x = 253). ──
    if headline_sub:
        parts.append(
            f'<rect x="{SUB_X}" y="{SUB_Y}" width="{SUB_W}" height="{SUB_H}" '
            f'fill="#08080a"/>'
        )
        parts.append(
            f'<text x="{SUB_CX}" y="{SUB_Y + 32}" text-anchor="middle" '
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

    # ── 5. The 10 pill overlays (logo + name + team subline; no per-pill stat) ──
    for i, row in enumerate(rows):
        parts.append(_row_overlay(row, i, show_team=show_team_subline))

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
