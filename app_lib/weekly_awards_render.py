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
EMBLEM_64 = _APP_DIR / 'assets' / 'branding' / '64_emblem.png'
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

# Hero panel — fills the bordered octagon on the right.
# CORRECTED 2026-05-04 from user markup: the panel's LEFT border is at
# x=470 (verified by gray-border pixel scan at y=400, y=600), NOT x=545
# as previously assumed. Right border x=1027, top y=145, bottom y=894.
# A circular mask preserves the baked 64A emblem in the top-right.
HERO_X = 478
HERO_Y = 152
HERO_W = 545
HERO_H = 705   # trimmed from 738 — ends at y=857 instead of 890 to leave
               # breathing room above the stats strip at y=894
# 64A circle: center (976, 185), r ≈ 85; mask a slightly larger disk.
EMBLEM_CX = 976
EMBLEM_CY = 185
EMBLEM_R = 92

# Bottom stats strip (pentagon) — same horizontal span as hero panel
# (CORRECTED to match real left border at x=470). Pixel-measured edges:
#   top border:   y=894
#   red line:     y=941
#   white line:   y=977
#   bottom border:y=1006
# Label in the TOP zone (above red line), value in the BOTTOM zone
# (below white line); the red+white line band divides them.
STATS_X = 478
STATS_Y = 894
STATS_W = 545
STATS_H = 112
STATS_LABEL_DY = 30   # baseline of label inside the top zone (y ≈ 924)
STATS_VALUE_DY = 73   # baseline of value between red and white lines (y ≈ 967)

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
            f'<text x="{name_x}" y="{pill_cy - 3:.1f}" text-anchor="start" '
            f'font-family="Barlow Condensed,Oswald,sans-serif" font-style="italic" '
            f'font-weight="700" font-size="20" fill="#ffffff" '
            f'letter-spacing="0.5">{_xe(player)}</text>'
        )
        parts.append(
            f'<text x="{name_x}" y="{pill_cy + 14:.1f}" text-anchor="start" '
            f'font-family="Oswald,sans-serif" font-weight="500" font-size="12" '
            f'fill="rgba(255,255,255,0.65)" letter-spacing="1.0">'
            f'{_xe(team)}</text>'
        )
    else:
        parts.append(
            f'<text x="{name_x}" y="{pill_cy + 6:.1f}" text-anchor="start" '
            f'font-family="Barlow Condensed,sans-serif" font-style="italic" '
            f'font-weight="700" font-size="21" fill="#ffffff" '
            f'letter-spacing="0.5">{_xe(player)}</text>'
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

    # Strip is 112 px tall (894-1006). Red decorative line at +47, white
    # line at +83. Label sits in the top zone (above red line), value
    # sits in the bottom zone (below white line). The red+white line
    # band between them visually separates label from value.
    # Strip is 112px tall starting at y. Decorative red line at +47, white
    # at +83. Place LABEL above red and VALUE between red+white (the wide
    # mid zone, not buried under the white line). Each cell is ~91 px wide.
    return ''.join([
        # Label (red, in top zone above the red line)
        f'<text x="{cx}" y="{y + STATS_LABEL_DY}" text-anchor="middle" '
        f'font-family="Oswald,sans-serif" font-weight="700" font-size="13" '
        f'letter-spacing="2.4" fill="#d72638">{_xe(label)}</text>',
        # Value (white, between the red and white lines — bigger font)
        f'<text x="{cx}" y="{y + STATS_VALUE_DY}" text-anchor="middle" '
        f'font-family="Barlow Condensed,sans-serif" font-style="italic" '
        f'font-weight="800" font-size="28" fill="#ffffff">{_xe(_fmt(value, decimals))}</text>',
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

    # ── 2. Hero panel — always visible. When an image is uploaded it fills
    # the panel; when empty, a striped placeholder (matching Share
    # Graphic's pattern) shows where the image will go. The 64A circle is
    # masked out either way so the baked red emblem stays on top. ──
    # Hero panel is a true octagon: diagonal cuts at all 4 corners. The
    # top-right cut is much larger to clear the baked 64A circle that
    # we redraw on top. The other three corners use ~32px chamfers.
    CHAMFER = 32
    BIG_CUT_W = 200   # top-right cut goes far inward to clear the 64A circle
    BIG_CUT_H = 170
    octagon_pts = (
        f'{HERO_X + CHAMFER},{HERO_Y} '                                  # top edge starts after TL cut
        f'{HERO_X + HERO_W - BIG_CUT_W},{HERO_Y} '                       # top edge before 64A cut
        f'{HERO_X + HERO_W},{HERO_Y + BIG_CUT_H} '                       # diagonal across the 64A corner
        f'{HERO_X + HERO_W},{HERO_Y + HERO_H - CHAMFER} '                # right edge ends before BR cut
        f'{HERO_X + HERO_W - CHAMFER},{HERO_Y + HERO_H} '                # bottom-right cut
        f'{HERO_X + CHAMFER},{HERO_Y + HERO_H} '                         # bottom edge ends before BL cut
        f'{HERO_X},{HERO_Y + HERO_H - CHAMFER} '                         # bottom-left cut
        f'{HERO_X},{HERO_Y + CHAMFER}'                                   # left edge ends before TL cut
    )
    # Radial vignette inside the octagon so the image fades softly toward
    # the panel edges instead of having a hard rectangular boundary.
    HERO_CX_VG = HERO_X + HERO_W / 2
    HERO_CY_VG = HERO_Y + HERO_H / 2
    HERO_R_VG = max(HERO_W, HERO_H) * 0.62
    parts.append(
        '<defs>'
        # Feather filter — Gaussian blurs the mask polygon so the octagon
        # edges aren't a hard pixel cut. Result: image dissolves softly
        # along the panel boundary instead of having visible crisp edges.
        '<filter id="featherEdge" x="-5%" y="-5%" width="110%" height="110%">'
        '<feGaussianBlur stdDeviation="6"/>'
        '</filter>'
        '<mask id="heroMask" maskUnits="userSpaceOnUse" '
        f'x="0" y="0" width="{W}" height="{H}">'
        f'<polygon points="{octagon_pts}" fill="white" filter="url(#featherEdge)"/>'
        '</mask>'
        # Multi-stop vignette: image full at center, gradually darkening
        # through 3 intermediate stops, fully blended into the backdrop
        # color at the corners. The earlier the first non-zero stop, the
        # more dramatic the fade — 0.30 means the fade starts only 30%
        # out from the center.
        f'<radialGradient id="heroFade" cx="{HERO_CX_VG:.0f}" cy="{HERO_CY_VG:.0f}" '
        f'r="{HERO_R_VG:.0f}" gradientUnits="userSpaceOnUse">'
        '<stop offset="0" stop-color="#08080a" stop-opacity="0"/>'
        '<stop offset="0.30" stop-color="#08080a" stop-opacity="0"/>'
        '<stop offset="0.65" stop-color="#08080a" stop-opacity="0.35"/>'
        '<stop offset="0.85" stop-color="#08080a" stop-opacity="0.75"/>'
        '<stop offset="1" stop-color="#08080a" stop-opacity="1"/>'
        '</radialGradient>'
        '<pattern id="heroPlaceholder" patternUnits="userSpaceOnUse" '
        'width="48" height="48" patternTransform="rotate(135)">'
        '<rect width="48" height="48" fill="#18181c"/>'
        '<rect x="24" width="24" height="48" fill="#1e1e23"/>'
        '</pattern>'
        '</defs>'
    )
    if hero_b64:
        parts.append(
            f'<image href="{hero_b64}" xlink:href="{hero_b64}" '
            f'x="{HERO_X}" y="{HERO_Y}" width="{HERO_W}" height="{HERO_H}" '
            f'preserveAspectRatio="xMidYMid slice" mask="url(#heroMask)"/>'
        )
        # Radial vignette overlay so the image fades softly toward the
        # octagon edges instead of having a hard cut. Same mask so it
        # follows the octagon shape.
        parts.append(
            f'<rect x="{HERO_X}" y="{HERO_Y}" width="{HERO_W}" height="{HERO_H}" '
            f'fill="url(#heroFade)" mask="url(#heroMask)"/>'
        )
    else:
        # Striped placeholder fills the entire panel area, masked so the
        # placeholder follows the octagon shape.
        parts.append(
            f'<rect x="{HERO_X}" y="{HERO_Y}" width="{HERO_W}" height="{HERO_H}" '
            f'fill="url(#heroPlaceholder)" mask="url(#heroMask)"/>'
        )
        parts.append(
            f'<text x="{HERO_X + HERO_W/2:.0f}" y="{HERO_Y + HERO_H/2:.0f}" '
            f'text-anchor="middle" font-family="JetBrains Mono,monospace" '
            f'font-weight="600" font-size="14" letter-spacing="3" '
            f'fill="rgba(255,255,255,0.55)">DROP HERO IMAGE HERE</text>'
        )

    # ── 2b. 64A emblem PNG drawn ON TOP of the hero. Uses the official
    # brand logo (assets/branding/64_emblem.png) instead of recreating it
    # in SVG, so the proportions and angled "64" mark match the brand.
    emblem_size = (EMBLEM_R - 12) * 2  # diameter
    emblem_b64 = _b64(EMBLEM_64)
    if emblem_b64:
        parts.append(
            f'<image href="{emblem_b64}" xlink:href="{emblem_b64}" '
            f'x="{EMBLEM_CX - emblem_size/2:.0f}" '
            f'y="{EMBLEM_CY - emblem_size/2:.0f}" '
            f'width="{emblem_size:.0f}" height="{emblem_size:.0f}" '
            f'preserveAspectRatio="xMidYMid meet"/>'
        )
    else:
        # Fallback: simple circle if the asset is missing
        parts.append(
            f'<circle cx="{EMBLEM_CX}" cy="{EMBLEM_CY}" r="{EMBLEM_R - 12}" '
            f'fill="#a01827"/>'
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
