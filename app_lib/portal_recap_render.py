"""Portal Recap share graphic — 1080×1350 SVG (draft-recap style, 64A brand).

Three panels:
  1. Teams with the Most Top-250 Commits  (count tiers + team logos)
  2. Where the Top 250 Went                (donut: P4 confs + Drafted/Pro + Other + Uncommitted)
  3. % Committing to Power 4 by Portal Rank (horizontal bars per rank tier)

Pure builder: build_portal_recap_svg(payload) -> svg string. No Streamlit import, so it
can be unit-rendered. Colors are categorical, fixed order; every slice/bar is direct-labeled
(identity never color-alone). Render to PNG with app_lib.safe_render.safe_svg2png.
"""
from __future__ import annotations
import base64
import html
import math
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
LOGO_DIR = _APP_DIR / 'team_logos_512'
BRAND64 = _APP_DIR / 'assets' / 'portal-entrant' / 'logo-64-analytics.png'

# ---- palette (dark surface #0e1116) ----
BG = '#0e1116'
PANEL = '#161b22'
PANEL_EDGE = '#242c37'
INK = '#ffffff'
INK_SOFT = '#93a1b2'
RED = '#c8102e'
GOLD = '#d9a94a'
# categorical pie colors, fixed order (P4 first, then Other/Pro/Uncommitted)
CONF_COLOR = {
    'SEC': '#c8102e', 'ACC': '#1f6feb', 'Big 12': '#e8863c',
    'Big Ten': '#7d5cc6', 'Other': '#5b6672',
    'Drafted/Pro': '#d9a94a', 'Uncommitted': '#39414c',
}


def _xe(s):
    return html.escape(str(s) if s is not None else '')


def _b64(p: Path):
    try:
        return f'data:image/png;base64,{base64.b64encode(p.read_bytes()).decode("ascii")}'
    except Exception:
        return None


def logo_b64(tid):
    return _b64(LOGO_DIR / f'{int(tid)}.png') if tid else None


def _arc(cx, cy, R, r, a0, a1):
    """Donut segment path from angle a0->a1 (radians, 0 = +x, clockwise-negative)."""
    large = 1 if (a1 - a0) % (2 * math.pi) > math.pi else 0
    x1o, y1o = cx + R * math.cos(a0), cy + R * math.sin(a0)
    x2o, y2o = cx + R * math.cos(a1), cy + R * math.sin(a1)
    x1i, y1i = cx + r * math.cos(a1), cy + r * math.sin(a1)
    x2i, y2i = cx + r * math.cos(a0), cy + r * math.sin(a0)
    return (f'M{x1o:.2f},{y1o:.2f} A{R:.2f},{R:.2f} 0 {large} 1 {x2o:.2f},{y2o:.2f} '
            f'L{x1i:.2f},{y1i:.2f} A{r:.2f},{r:.2f} 0 {large} 0 {x2i:.2f},{y2i:.2f} Z')


def build_portal_recap_svg(payload: dict) -> str:
    """payload keys:
       title, subtitle,
       tiers: [(count:int, [logo_b64 or None, ...]), ...]   # panel 1, sorted desc
       pie:   [(label, count, color), ...]                  # panel 2 (already ordered)
       bars:  [(label, pct_float), ...]                     # panel 3 (rank tiers)
       total_top250: int
    """
    W, H = 1080, 1350
    FS = 'font-family="Arial,Helvetica,sans-serif"'
    p = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" '
         f'xmlns:xlink="http://www.w3.org/1999/xlink">']
    p.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    # faint diagonal texture
    p.append('<defs><pattern id="tex" width="46" height="46" patternUnits="userSpaceOnUse" '
             'patternTransform="rotate(125)"><rect width="46" height="46" fill="#0e1116"/>'
             '<rect width="23" height="46" fill="#0c0f13"/></pattern></defs>')
    p.append(f'<rect width="{W}" height="{H}" fill="url(#tex)" opacity="0.5"/>')

    # ---------- header ----------
    p.append(f'<text x="52" y="112" {FS} font-weight="bold" font-size="76" font-style="italic" '
             f'fill="{RED}">2026<tspan fill="{INK}" font-style="normal"> PORTAL</tspan></text>')
    p.append(f'<text x="56" y="150" {FS} font-weight="bold" font-size="22" letter-spacing="5" '
             f'fill="{INK_SOFT}">{_xe(payload.get("subtitle", "TOP 250 TRANSFER RECAP"))}</text>')
    p.append(f'<rect x="52" y="168" width="{W-104}" height="4" fill="{RED}"/>')

    # ---------- panel 1: most top-250 commits ----------
    P1X, P1Y, P1W, P1H = 40, 196, 496, 470
    p.append(f'<rect x="{P1X}" y="{P1Y}" width="{P1W}" height="{P1H}" rx="16" fill="{PANEL}" stroke="{PANEL_EDGE}"/>')
    p.append(f'<text x="{P1X+24}" y="{P1Y+40}" {FS} font-weight="bold" font-size="24" fill="{INK}">'
             f'MOST TOP-250 <tspan fill="{RED}">COMMITS</tspan></text>')
    tiers = payload['tiers'][:6]
    row_h = (P1H - 64) / max(len(tiers), 1)
    ty = P1Y + 64
    for count, logos in tiers:
        cy = ty + row_h / 2
        p.append(f'<text x="{P1X+40}" y="{cy+11}" {FS} font-weight="bold" font-size="40" '
                 f'fill="{RED}" text-anchor="middle">{count}</text>')
        lx = P1X + 92
        lsz = min(46, (P1W - 110) / max(len(logos), 1) - 8)
        for lg in logos[:6]:
            if lg:
                p.append(f'<rect x="{lx}" y="{cy-lsz/2-4}" width="{lsz+8}" height="{lsz+8}" rx="7" '
                         f'fill="#f3efe7"/>')
                p.append(f'<image href="{lg}" xlink:href="{lg}" x="{lx+4}" y="{cy-lsz/2}" '
                         f'width="{lsz}" height="{lsz}" preserveAspectRatio="xMidYMid meet"/>')
            lx += lsz + 16
        if ty + row_h < P1Y + P1H - 6:
            p.append(f'<line x1="{P1X+24}" y1="{ty+row_h:.0f}" x2="{P1X+P1W-24}" y2="{ty+row_h:.0f}" '
                     f'stroke="{PANEL_EDGE}"/>')
        ty += row_h

    # ---------- panel 2: donut of destinations ----------
    P2X, P2Y, P2W, P2H = 556, 196, 484, 470
    p.append(f'<rect x="{P2X}" y="{P2Y}" width="{P2W}" height="{P2H}" rx="16" fill="{PANEL}" stroke="{PANEL_EDGE}"/>')
    p.append(f'<text x="{P2X+24}" y="{P2Y+40}" {FS} font-weight="bold" font-size="24" fill="{INK}">'
             f'WHERE THE <tspan fill="{RED}">TOP 250</tspan> WENT</text>')
    pie = payload['pie']
    total = sum(c for _, c, _ in pie) or 1
    cx, cy, R, r = P2X + 148, P2Y + 250, 118, 66
    a = -math.pi / 2
    for label, cnt, color in pie:
        frac = cnt / total
        a1 = a + frac * 2 * math.pi
        p.append(f'<path d="{_arc(cx, cy, R, r, a, a1)}" fill="{color}" stroke="{PANEL}" stroke-width="2"/>')
        # leader % label on slices >= 5%
        if frac >= 0.05:
            mid = (a + a1) / 2
            lx, ly = cx + (R + 16) * math.cos(mid), cy + (R + 16) * math.sin(mid)
            anch = 'start' if math.cos(mid) >= 0 else 'end'
            p.append(f'<text x="{lx:.0f}" y="{ly:.0f}" {FS} font-weight="bold" font-size="15" '
                     f'fill="{INK}" text-anchor="{anch}">{100*frac:.0f}%</text>')
        a = a1
    p.append(f'<text x="{cx}" y="{cy-4}" {FS} font-weight="bold" font-size="34" fill="{INK}" text-anchor="middle">'
             f'{payload.get("total_top250", total)}</text>')
    p.append(f'<text x="{cx}" y="{cy+20}" {FS} font-size="13" letter-spacing="1.5" fill="{INK_SOFT}" '
             f'text-anchor="middle">PLAYERS</text>')
    # legend (right of donut)
    lgx, lgy = P2X + 292, P2Y + 96
    for label, cnt, color in pie:
        p.append(f'<rect x="{lgx}" y="{lgy-13}" width="15" height="15" rx="3" fill="{color}"/>')
        p.append(f'<text x="{lgx+24}" y="{lgy}" {FS} font-size="17" font-weight="bold" fill="{INK}">'
                 f'{_xe(label)}</text>')
        p.append(f'<text x="{lgx+24}" y="{lgy+18}" {FS} font-size="13" fill="{INK_SOFT}">'
                 f'{cnt}  |  {100*cnt/total:.1f}%</text>')
        lgy += 47

    # ---------- panel 3: % to Power 4 by portal rank ----------
    P3X, P3Y, P3W, P3H = 40, 690, 1000, 560
    p.append(f'<rect x="{P3X}" y="{P3Y}" width="{P3W}" height="{P3H}" rx="16" fill="{PANEL}" stroke="{PANEL_EDGE}"/>')
    p.append(f'<text x="{P3X+28}" y="{P3Y+46}" {FS} font-weight="bold" font-size="28" fill="{INK}">'
             f'% COMMITTING TO A <tspan fill="{RED}">POWER 4</tspan> BY PORTAL RANK</text>')
    p.append(f'<text x="{P3X+28}" y="{P3Y+72}" {FS} font-size="15" fill="{INK_SOFT}">'
             f'Share of committed portal players at an SEC / ACC / Big 12 / Big Ten school</text>')
    bars = payload['bars']
    bx0 = P3X + 250
    bx1 = P3X + P3W - 90
    bmax = bx1 - bx0
    top_y = P3Y + 108
    bh = (P3H - 150) / max(len(bars), 1)
    for i, (label, pct) in enumerate(bars):
        by = top_y + i * bh
        cyb = by + bh / 2
        p.append(f'<text x="{P3X+232}" y="{cyb+7}" {FS} font-weight="bold" font-size="21" '
                 f'fill="{INK}" text-anchor="end">{_xe(label)}</text>')
        # track
        p.append(f'<rect x="{bx0}" y="{cyb-16}" width="{bmax:.0f}" height="32" rx="6" fill="#0e1116"/>')
        w = max(4, bmax * pct / 100)
        p.append(f'<rect x="{bx0}" y="{cyb-16}" width="{w:.0f}" height="32" rx="6" fill="{RED}"/>')
        p.append(f'<text x="{bx0+w+12:.0f}" y="{cyb+7}" {FS} font-weight="bold" font-size="21" '
                 f'fill="{INK}">{pct:.1f}%</text>')
    p.append(f'<text x="{P3X+28}" y="{P3Y+P3H-24}" {FS} font-size="14" fill="{INK_SOFT}">'
             f'Rank tiers by 64A portal rank (1 = best)</text>')

    # ---------- footer ----------
    p.append(f'<text x="52" y="1324" {FS} font-weight="bold" font-size="20" fill="{RED}">'
             f'2026 PORTAL <tspan fill="{INK}">RECAP</tspan></text>')
    b = _b64(BRAND64)
    if b:
        p.append(f'<image href="{b}" xlink:href="{b}" x="{W-232}" y="1296" width="180" height="44" '
                 f'preserveAspectRatio="xMidYMid meet"/>')
    p.append('</svg>')
    return ''.join(p)
