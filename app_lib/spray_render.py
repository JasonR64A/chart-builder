"""Reusable SVG renderers for the Spray Charts visuals so other pages
(Regional Preview, etc.) can embed the team-grid and single-team spray
diagram without duplicating the ~500 lines of geometry math that lives
on the Spray Charts page itself.

Two public entry points:
  * build_team_grid_svg(sport, division, team_name, team_id, perspective='hitting')
    — 3x3 grid of the team's top-9 hitters (or pitchers) by BIP.
  * build_team_spray_svg(sport, division, team_name, team_id, perspective='hitting')
    — single team-level wedge spray heatmap (the bigger field diagram).

Both return a complete <svg>...</svg> string ready to drop into a page.
The renderers default to no situational filters and metric_choice = '% of BIP'
so callers can render with sensible defaults; future iterations can expose
the filter set.
"""
from __future__ import annotations
import base64
import math
from pathlib import Path

import pandas as pd

from app_lib.spray_data import (
    compute_spray_distribution,
    compute_field_side_buckets,
    add_zone_metrics,
    list_players,
    list_pitchers,
    ZONE_NAMES,
)

_APP_DIR = Path(__file__).resolve().parent.parent
LOGO_DIR = _APP_DIR / 'team_logos_512'
HEADSHOT_DIR = _APP_DIR / 'assets' / 'player_headshots'
BRAND_64A_WIDE = _APP_DIR / 'assets' / 'logo-64a-wide.png'


# ── Small utilities ─────────────────────────────────────────────────────────
def _xe(s) -> str:
    if s is None:
        return ''
    return (str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def _embed(path: Path, x: float, y: float, w: float, h: float, opacity: float = 1.0) -> str:
    if not Path(path).exists():
        return ''
    try:
        b64 = base64.b64encode(Path(path).read_bytes()).decode('ascii')
    except Exception:
        return ''
    href = f'data:image/png;base64,{b64}'
    return (f'<image href="{href}" xlink:href="{href}" '
            f'x="{x}" y="{y}" width="{w}" height="{h}" opacity="{opacity}" '
            f'preserveAspectRatio="xMidYMid meet"/>')


def _embed_headshot_circle(cb_id, cx: float, cy: float, r: float) -> str:
    """Profile bubble: PNG from assets/player_headshots/{cb_id}.png if it
    exists, otherwise an empty placeholder circle."""
    if cb_id is not None:
        p = HEADSHOT_DIR / f'{cb_id}.png'
        if p.exists():
            try:
                b64 = base64.b64encode(p.read_bytes()).decode('ascii')
                href = f'data:image/png;base64,{b64}'
                clip_id = f'clip_p_{cb_id}'
                return (
                    f'<defs><clipPath id="{clip_id}">'
                    f'<circle cx="{cx}" cy="{cy}" r="{r}"/></clipPath></defs>'
                    f'<image href="{href}" xlink:href="{href}" '
                    f'x="{cx - r}" y="{cy - r}" width="{r*2}" height="{r*2}" '
                    f'preserveAspectRatio="xMidYMid slice" '
                    f'clip-path="url(#{clip_id})"/>'
                    f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" '
                    f'stroke="#0F2A4D" stroke-width="0.5"/>'
                )
            except Exception:
                pass
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#F0EAD6" '
            f'stroke="#0F2A4D" stroke-width="0.5"/>')


def _ro(intensity: float) -> str:
    """Eggshell → deep-red ramp used for spray-density coloring."""
    i = max(0.05, min(1.0, intensity))
    r = int(255 - (255 - 198) * i)
    g = int(224 - (224 - 40) * i)
    b = int(204 - (204 - 40) * i)
    return f'#{r:02X}{g:02X}{b:02X}'


def _tc(intensity: float) -> str:
    return '#FFFFFF' if intensity > 0.55 else '#0F2A4D'


def _metric_fmt(v: float, choice: str = '% of BIP') -> str:
    if choice == '% of BIP':
        return f'{v:.0f}%'
    if choice == 'TB':
        return f'{int(v)}'
    if 0 <= v < 1:
        return f'.{int(round(v*1000)):03d}'
    return f'{v:.3f}'


def _compact(n) -> str:
    n = int(n)
    if abs(n) >= 1_000_000:
        return f'{n/1_000_000:.1f}M'
    if abs(n) >= 10_000:
        return f'{n/1_000:.0f}K'
    return f'{n:,}'


# ── Cell renderer (3x3 grid) ────────────────────────────────────────────────
# Cell viewBox is 0..100 × 0..90; parent translates each cell to its slot.
_C_HOME = (50, 76)
_C_RINNER = 7
_C_RMID = 22
_C_ROUTER = 42
_C_LINE_HALF = 15
_C_OF = [
    ((7, 10),  -45, -9),
    ((8, 14),   -9,  9),
    ((9, 11),    9, 45),
]
_C_IF = [
    ((5,), -45,   -22.5),
    ((6,), -22.5,   0),
    ((4,),   0,    22.5),
    ((3,),  22.5,  45),
]
_C_LN = [
    ((12,), -45 - _C_LINE_HALF, -45),
    ((13,),  45,  45 + _C_LINE_HALF),
]


def _cpolar(a, r):
    rad = math.radians(a)
    return (_C_HOME[0] + r * math.sin(rad), _C_HOME[1] - r * math.cos(rad))


def _cwedge(a1, a2, ri, ro):
    x1i, y1i = _cpolar(a1, ri); x1o, y1o = _cpolar(a1, ro)
    x2i, y2i = _cpolar(a2, ri); x2o, y2o = _cpolar(a2, ro)
    return (f'M {x1i:.2f},{y1i:.2f} L {x1o:.2f},{y1o:.2f} '
            f'A {ro},{ro} 0 0 1 {x2o:.2f},{y2o:.2f} '
            f'L {x2i:.2f},{y2i:.2f} A {ri},{ri} 0 0 0 {x1i:.2f},{y1i:.2f} Z')


def _cpie(a1, a2, ro):
    x1, y1 = _cpolar(a1, ro); x2, y2 = _cpolar(a2, ro)
    return (f'M {_C_HOME[0]:.2f},{_C_HOME[1]:.2f} L {x1:.2f},{y1:.2f} '
            f'A {ro},{ro} 0 0 1 {x2:.2f},{y2:.2f} Z')


def _clabelpos(a1, a2, ri, ro):
    ang = (a1 + a2) / 2
    r = (ri + ro) / 2
    return _cpolar(ang, r)


def _ccombined(spray, codes):
    sub = spray[spray['hitLocation'].isin(codes)]
    if sub.empty:
        return None
    n = int(sub['total'].sum())
    if n == 0:
        return None
    get = lambda c: int(sub[c].sum()) if c in sub.columns else 0
    c1, c2, c3, ch = get('1B'), get('2B'), get('3B'), get('HR')
    hits = c1 + c2 + c3 + ch
    tb = c1 + 2*c2 + 3*c3 + 4*ch
    woba = 0.888*c1 + 1.271*c2 + 1.616*c3 + 2.101*ch
    return {'BIP': n, 'pct': float(sub['pct'].sum()),
            '1B': c1, '2B': c2, '3B': c3, 'HR': ch,
            'AVG': hits / n, 'SLG': tb / n, 'wOBA': woba / n, 'TB': tb}


def _build_cell(c, metric_choice='% of BIP') -> str:
    spray = c['spray']
    b = c['buckets']
    parts_c = []
    parts_c.append(
        '<rect x="2" y="2" width="96" height="91" rx="3" ry="3" '
        'fill="#FFFFFF" stroke="#0F2A4D" stroke-width="0.4"/>'
    )
    parts_c.append(
        f'<text x="5" y="8" font-family="Inter,sans-serif" font-size="4.2" '
        f'font-weight="800" fill="#0F2A4D">{_xe(c["name"])}</text>'
    )
    sub_bits = _xe(' · '.join(x for x in [c.get('pos', ''), c.get('cls', '')] if x))
    if sub_bits:
        parts_c.append(
            f'<text x="95" y="8" text-anchor="end" font-family="Inter,sans-serif" '
            f'font-size="2.8" font-weight="600" fill="#0F2A4D">{sub_bits}</text>'
        )
    if spray is None or spray.empty:
        parts_c.append('<text x="50" y="50" text-anchor="middle" '
                       'font-family="Inter,sans-serif" font-size="3" '
                       'font-weight="600" fill="#0F2A4D">No batted-ball data</text>')
        return ''.join(parts_c)

    # Per-zone metric values for the wedge fills
    val_by, fmt_by = {}, {}
    for zc, *_ in _C_OF + _C_IF + _C_LN + [((1,), 0, 0), ((2,), 0, 0)]:
        bd = _ccombined(spray, zc)
        primary = zc[0]
        if bd is None:
            val_by[primary] = 0.0
            fmt_by[primary] = '–'
        else:
            v = (bd['pct'] if metric_choice == '% of BIP'
                 else (float(bd['TB']) if metric_choice == 'TB' else bd[metric_choice]))
            val_by[primary] = v
            fmt_by[primary] = _metric_fmt(v, metric_choice)
    fair_max = max((val_by.get(zc[0], 0) for zc, *_ in _C_OF + _C_IF + _C_LN),
                   default=1.0) or 1.0

    # L/M/R mini diamond
    m_home = (22, 33); m_r = 21.0
    side_pcts = {'L': float(b['left_pct']),
                 'C': float(b['middle_pct']),
                 'R': float(b['right_pct'])}
    side_max = max(side_pcts.values()) or 1.0
    SIDE = [('L', -45, -15), ('C', -15, 15), ('R', 15, 45)]
    def _mp(a1, a2, r):
        r1, r2 = math.radians(a1), math.radians(a2)
        x1 = m_home[0] + r * math.sin(r1); y1 = m_home[1] - r * math.cos(r1)
        x2 = m_home[0] + r * math.sin(r2); y2 = m_home[1] - r * math.cos(r2)
        return (f'M {m_home[0]},{m_home[1]} L {x1:.2f},{y1:.2f} '
                f'A {r},{r} 0 0 1 {x2:.2f},{y2:.2f} Z')
    for k, a1, a2 in SIDE:
        pct = side_pcts[k]
        inten = pct / side_max
        parts_c.append(f'<path d="{_mp(a1, a2, m_r)}" fill="{_ro(inten)}" '
                       f'stroke="#FFFFFF" stroke-width="0.5"/>')
        mid = math.radians((a1 + a2) / 2)
        lx = m_home[0] + (m_r * 0.72) * math.sin(mid)
        ly = m_home[1] - (m_r * 0.72) * math.cos(mid)
        parts_c.append(
            f'<text x="{lx:.1f}" y="{ly+0.6:.1f}" text-anchor="middle" '
            f'font-family="Inter,sans-serif" font-size="2.4" font-weight="800" '
            f'fill="{_tc(inten)}">{pct:.0f}%</text>'
        )
    parts_c.append(f'<path d="{_mp(-45, 45, m_r)}" fill="none" '
                   f'stroke="#0F2A4D" stroke-width="0.5"/>')

    # 2x2 hit-type grid (top-right)
    get = lambda col: int(spray[col].sum()) if col in spray.columns else 0
    ht = [[('1B', get('1B')), ('2B', get('2B'))],
          [('3B', get('3B')), ('HR', get('HR'))]]
    tr_x0, tr_y0 = 52, 10
    tr_cw, tr_ch = 22, 6
    for ri, row in enumerate(ht):
        for ci, (lab, val) in enumerate(row):
            cxv = tr_x0 + tr_cw / 2 + ci * tr_cw
            cyv = tr_y0 + ri * tr_ch
            parts_c.append(
                f'<text x="{cxv:.1f}" y="{cyv+1.6:.1f}" text-anchor="middle" '
                f'font-family="Inter,sans-serif" font-size="2.8" font-weight="600" '
                f'fill="#0F2A4D">{lab}</text>'
            )
            parts_c.append(
                f'<text x="{cxv:.1f}" y="{cyv+5.1:.1f}" text-anchor="middle" '
                f'font-family="Inter,sans-serif" font-size="4.2" font-weight="800" '
                f'fill="#0F2A4D">{_compact(val)}</text>'
            )

    # Diamond wedges
    def _draw(zc, a1, a2, ri, ro, lf, *, pie=False):
        primary = zc[0]
        v = val_by.get(primary, 0)
        inten = v / fair_max if fair_max else 0
        d = _cpie(a1, a2, ro) if pie else _cwedge(a1, a2, ri, ro)
        parts_c.append(f'<path d="{d}" fill="{_ro(inten)}" '
                       f'stroke="#FFFFFF" stroke-width="0.4"/>')
        lx, ly = _clabelpos(a1, a2, ri, ro)
        parts_c.append(
            f'<text x="{lx:.1f}" y="{ly+0.7:.1f}" text-anchor="middle" '
            f'font-family="Inter,sans-serif" font-size="{lf}" font-weight="800" '
            f'fill="{_tc(inten)}">{fmt_by.get(primary, "")}</text>'
        )
    for zc, a1, a2 in _C_OF: _draw(zc, a1, a2, _C_RMID, _C_ROUTER, 3.5)
    for zc, a1, a2 in _C_LN: _draw(zc, a1, a2, _C_RINNER, _C_ROUTER, 2.6, pie=True)
    for zc, a1, a2 in _C_IF: _draw(zc, a1, a2, _C_RINNER, _C_RMID, 2.0)

    # Pitcher dot + catcher pentagon
    p_pct = fmt_by.get(1, '')
    px, py = _C_HOME[0], _C_HOME[1] - 4.5
    parts_c.append(f'<circle cx="{px}" cy="{py}" r="2.6" fill="#F8E8E2" '
                   f'stroke="#0F2A4D" stroke-width="0.4"/>')
    parts_c.append(f'<text x="{px}" y="{py+0.8:.1f}" text-anchor="middle" '
                   f'font-family="Inter,sans-serif" font-size="2.0" font-weight="800" '
                   f'fill="#0F2A4D">{p_pct}</text>')
    c_pct = fmt_by.get(2, '')
    pw, ph = 2.5, 2.3
    cx2, cy2 = _C_HOME[0], _C_HOME[1] + ph
    plate = [(cx2-pw, cy2-ph), (cx2+pw, cy2-ph),
             (cx2+pw, cy2+ph*0.10), (cx2, cy2+ph), (cx2-pw, cy2+ph*0.10)]
    parts_c.append('<polygon points="' + ' '.join(f'{x:.2f},{y:.2f}' for x, y in plate)
                   + '" fill="#F8E8E2" stroke="#0F2A4D" stroke-width="0.3"/>')
    parts_c.append(f'<text x="{cx2}" y="{cy2+0.4:.1f}" text-anchor="middle" '
                   f'font-family="Inter,sans-serif" font-size="1.4" font-weight="800" '
                   f'fill="#0F2A4D">{c_pct}</text>')

    # Field outline
    fo = 45 + _C_LINE_HALF
    fxL, fyL = _cpolar(-fo, _C_ROUTER); fxR, fyR = _cpolar(fo, _C_ROUTER)
    parts_c.append(f'<path d="M {_C_HOME[0]},{_C_HOME[1]} L {fxL:.2f},{fyL:.2f} '
                   f'A {_C_ROUTER},{_C_ROUTER} 0 0 1 {fxR:.2f},{fyR:.2f} Z" '
                   f'fill="none" stroke="#0F2A4D" stroke-width="0.5"/>')

    # Bottom-left aggregate stats grid (AVG/SLG, wOBA/TB)
    n_bip = int(spray['total'].sum())
    c1 = get('1B'); c2 = get('2B'); c3 = get('3B'); chh = get('HR')
    hits = c1 + c2 + c3 + chh
    tb_t = c1 + 2*c2 + 3*c3 + 4*chh
    woba_n = 0.888*c1 + 1.271*c2 + 1.616*c3 + 2.101*chh
    avg = hits / n_bip if n_bip else 0
    slg = tb_t / n_bip if n_bip else 0
    woba = woba_n / n_bip if n_bip else 0
    overall = [('AVG', _metric_fmt(avg, 'AVG')),
               ('SLG', _metric_fmt(slg, 'SLG')),
               ('wOBA', _metric_fmt(woba, 'wOBA')),
               ('TB', _compact(tb_t))]
    bl_x0 = 4
    bl_cell_w, bl_cell_h = 22, 6
    bl_y0 = 80
    for i, (lab, val) in enumerate(overall):
        row, col = divmod(i, 2)
        cxv = bl_x0 + bl_cell_w / 2 + col * bl_cell_w
        cyv = bl_y0 + row * bl_cell_h
        parts_c.append(
            f'<text x="{cxv:.1f}" y="{cyv+1.6:.1f}" text-anchor="middle" '
            f'font-family="Inter,sans-serif" font-size="2.8" font-weight="600" '
            f'fill="#0F2A4D">{lab}</text>'
        )
        parts_c.append(
            f'<text x="{cxv:.1f}" y="{cyv+5.1:.1f}" text-anchor="middle" '
            f'font-family="Inter,sans-serif" font-size="4.2" font-weight="800" '
            f'fill="#0F2A4D">{val}</text>'
        )

    # Profile bubble (bottom-right)
    parts_c.append(_embed_headshot_circle(c.get('cb_id'), cx=85, cy=85, r=7.0))
    return ''.join(parts_c)


# ── Public: 3x3 Team Grid ───────────────────────────────────────────────────
def build_team_grid_svg(sport: str, division: str, team_name: str,
                        team_id: int | None = None,
                        perspective: str = 'hitting',
                        metric_choice: str = '% of BIP') -> str:
    """Return SVG string for a 3x3 grid of the team's top-9 hitters (or
    pitchers) by BIP, mirroring the Spray_Charts page's Team Grid view."""
    if perspective == 'pitching':
        plist = list_pitchers(sport, division, team_name)
        id_col = 'pitcherId'
    else:
        plist = list_players(sport, division, team_name)
        id_col = 'playerId'
    if plist.empty:
        return ('<svg viewBox="0 0 304 309" xmlns="http://www.w3.org/2000/svg">'
                '<rect width="100%" height="100%" fill="#F0EAD6"/>'
                f'<text x="152" y="160" text-anchor="middle" font-family="Inter,sans-serif" '
                f'font-size="6" font-weight="700" fill="#0F2A4D">'
                f'No batted-ball data for {_xe(team_name)}</text></svg>')

    top9 = plist.head(9).reset_index(drop=True)

    cells = []
    for _, p in top9.iterrows():
        sp = compute_spray_distribution(
            sport, division, team_name=team_name,
            player_id=p[id_col], perspective=perspective,
        )
        sp = add_zone_metrics(sp)
        b = compute_field_side_buckets(sp)
        full_name = p.get('player_name')
        disp = full_name if isinstance(full_name, str) and full_name.strip() else str(p.get('player', ''))
        cells.append({
            'name': disp,
            'pos': p.get('position') if isinstance(p.get('position'), str) else '',
            'cls': p.get('classification') if isinstance(p.get('classification'), str) else '',
            'cb_id': int(p['cb_id']) if pd.notna(p.get('cb_id')) else None,
            'spray': sp,
            'buckets': b,
        })

    HDR_H = 20
    CELL_W, CELL_H = 100, 95
    COLS, ROWS = 3, 3
    GRID_W = COLS * CELL_W
    GRID_H = ROWS * CELL_H
    VB_W = GRID_W + 4
    VB_H = HDR_H + GRID_H + 4

    parts = [
        f'<svg viewBox="0 0 {VB_W} {VB_H}" width="{VB_W}" height="{VB_H}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink">'
        f'<rect x="0" y="0" width="{VB_W}" height="{VB_H}" fill="#F0EAD6"/>',
        _embed(BRAND_64A_WIDE, x=VB_W/2 - 18, y=2, w=36, h=8.5),
    ]

    sport_label = 'Baseball' if sport.lower() == 'baseball' else 'Softball'
    role_word = 'Top 9 by BIP allowed' if perspective == 'pitching' else 'Top 9 by BIP'
    parts.append(
        f'<text x="{VB_W/2}" y="16" text-anchor="middle" font-family="Inter,sans-serif" '
        f'font-size="3.3" font-weight="700" fill="#0F2A4D">'
        f'{_xe(team_name)} · {sport_label} {division} · {role_word} · Coloring by {_xe(metric_choice)}</text>'
    )

    GRID_X0 = (VB_W - GRID_W) / 2
    for i, c in enumerate(cells):
        row, col = divmod(i, COLS)
        ox = GRID_X0 + col * CELL_W
        oy = HDR_H + row * CELL_H
        parts.append(f'<g transform="translate({ox},{oy})">')
        parts.append(_build_cell(c, metric_choice=metric_choice))
        parts.append('</g>')
    for i in range(len(cells), 9):
        row, col = divmod(i, COLS)
        ox = GRID_X0 + col * CELL_W
        oy = HDR_H + row * CELL_H
        parts.append(
            f'<rect x="{ox+2}" y="{oy+2}" width="96" height="91" rx="3" ry="3" '
            f'fill="#FFFFFF" stroke="#D0C9B0" stroke-width="0.3"/>'
        )

    if team_id is not None:
        team_logo_path = LOGO_DIR / f'{team_id}.png'
        if team_logo_path.exists():
            try:
                tlb64 = base64.b64encode(team_logo_path.read_bytes()).decode('ascii')
                tlhref = f'data:image/png;base64,{tlb64}'
                tlw = 140; tlh = 140
                tlx = VB_W/2 - tlw/2
                tly = HDR_H + GRID_H/2 - tlh/2
                parts.append(
                    f'<image href="{tlhref}" xlink:href="{tlhref}" '
                    f'x="{tlx}" y="{tly}" width="{tlw}" height="{tlh}" '
                    f'opacity="0.1" preserveAspectRatio="xMidYMid meet" '
                    f'pointer-events="none"/>'
                )
            except Exception:
                pass

    parts.append('</svg>')
    return ''.join(parts)


# ── Public: Single-team field diagram ───────────────────────────────────────
def build_team_spray_svg(sport: str, division: str, team_name: str,
                         team_id: int | None = None,
                         perspective: str = 'hitting',
                         metric_choice: str = '% of BIP') -> str:
    """Larger single-team wedge spray heatmap. Stripped of the on-page
    interactivity (tooltips, drill-downs) — just the field diagram."""
    spray = compute_spray_distribution(sport, division, team_name=team_name,
                                       perspective=perspective)
    spray = add_zone_metrics(spray)
    if spray.empty:
        return ('<svg viewBox="0 0 100 75" xmlns="http://www.w3.org/2000/svg">'
                '<rect width="100%" height="100%" fill="#F0EAD6"/>'
                f'<text x="50" y="40" text-anchor="middle" font-family="Inter,sans-serif" '
                f'font-size="3" font-weight="700" fill="#0F2A4D">'
                f'No batted-ball data for {_xe(team_name)}</text></svg>')

    HOME = (50, 65)
    R_INNER = 8; R_MID = 25; R_OUTER = 48
    LINE_HALF = 17
    OUTFIELD = [((7, 10), -45, -9), ((8, 14), -9, 9), ((9, 11), 9, 45)]
    INFIELD  = [((5,), -45, -22.5), ((6,), -22.5, 0), ((4,), 0, 22.5), ((3,), 22.5, 45)]
    LINE     = [((12,), -45 - LINE_HALF, -45), ((13,), 45, 45 + LINE_HALF)]

    def _polar(a, r):
        rad = math.radians(a)
        return (HOME[0] + r * math.sin(rad), HOME[1] - r * math.cos(rad))

    def _wedge(a1, a2, ri, ro):
        x1i, y1i = _polar(a1, ri); x1o, y1o = _polar(a1, ro)
        x2i, y2i = _polar(a2, ri); x2o, y2o = _polar(a2, ro)
        return (f'M {x1i:.2f},{y1i:.2f} L {x1o:.2f},{y1o:.2f} '
                f'A {ro},{ro} 0 0 1 {x2o:.2f},{y2o:.2f} '
                f'L {x2i:.2f},{y2i:.2f} A {ri},{ri} 0 0 0 {x1i:.2f},{y1i:.2f} Z')

    def _pie(a1, a2, ro):
        x1, y1 = _polar(a1, ro); x2, y2 = _polar(a2, ro)
        return (f'M {HOME[0]:.2f},{HOME[1]:.2f} L {x1:.2f},{y1:.2f} '
                f'A {ro},{ro} 0 0 1 {x2:.2f},{y2:.2f} Z')

    def _label_pos(a1, a2, ri, ro):
        return _polar((a1 + a2) / 2, (ri + ro) / 2)

    def _combined(codes):
        sub = spray[spray['hitLocation'].isin(codes)]
        if sub.empty: return None
        n = int(sub['total'].sum())
        if n == 0: return None
        get = lambda c: int(sub[c].sum()) if c in sub.columns else 0
        c1, c2, c3, ch = get('1B'), get('2B'), get('3B'), get('HR')
        hits = c1 + c2 + c3 + ch
        tb = c1 + 2*c2 + 3*c3 + 4*ch
        woba = 0.888*c1 + 1.271*c2 + 1.616*c3 + 2.101*ch
        return {'pct': float(sub['pct'].sum()), 'TB': tb,
                'AVG': hits / n, 'SLG': tb / n, 'wOBA': woba / n}

    val_by, fmt_by = {}, {}
    for zc, *_ in OUTFIELD + INFIELD + LINE + [((1,), 0, 0), ((2,), 0, 0)]:
        bd = _combined(zc)
        primary = zc[0]
        if bd is None:
            val_by[primary] = 0.0
            fmt_by[primary] = '–'
        else:
            v = (bd['pct'] if metric_choice == '% of BIP'
                 else (float(bd['TB']) if metric_choice == 'TB' else bd[metric_choice]))
            val_by[primary] = v
            fmt_by[primary] = _metric_fmt(v, metric_choice)
    fair_max = max((val_by.get(zc[0], 0) for zc, *_ in OUTFIELD + INFIELD + LINE),
                   default=1.0) or 1.0

    VB_W = 100; VB_H = 75
    parts = [
        f'<svg viewBox="0 0 {VB_W} {VB_H}" width="{VB_W}" height="{VB_H}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink">'
        f'<rect x="0" y="0" width="{VB_W}" height="{VB_H}" fill="#F0EAD6"/>',
    ]

    def _draw(zc, a1, a2, ri, ro, lf, *, pie=False):
        primary = zc[0]
        v = val_by.get(primary, 0)
        inten = v / fair_max if fair_max else 0
        d = _pie(a1, a2, ro) if pie else _wedge(a1, a2, ri, ro)
        parts.append(f'<path d="{d}" fill="{_ro(inten)}" '
                     f'stroke="#FFFFFF" stroke-width="0.5"/>')
        lx, ly = _label_pos(a1, a2, ri, ro)
        parts.append(
            f'<text x="{lx:.1f}" y="{ly+0.8:.1f}" text-anchor="middle" '
            f'font-family="Inter,sans-serif" font-size="{lf}" font-weight="800" '
            f'fill="{_tc(inten)}">{fmt_by.get(primary, "")}</text>'
        )
    for zc, a1, a2 in OUTFIELD: _draw(zc, a1, a2, R_MID, R_OUTER, 3.6)
    for zc, a1, a2 in LINE:     _draw(zc, a1, a2, R_INNER, R_OUTER, 2.6, pie=True)
    for zc, a1, a2 in INFIELD:  _draw(zc, a1, a2, R_INNER, R_MID, 2.4)

    # Pitcher + catcher
    px, py = HOME[0], HOME[1] - 4
    parts.append(f'<circle cx="{px}" cy="{py}" r="2.5" fill="#F8E8E2" '
                 f'stroke="#0F2A4D" stroke-width="0.4"/>')
    parts.append(f'<text x="{px}" y="{py+0.8:.1f}" text-anchor="middle" '
                 f'font-family="Inter,sans-serif" font-size="1.8" font-weight="800" '
                 f'fill="#0F2A4D">{fmt_by.get(1, "")}</text>')
    pw, ph = 2.4, 2.0
    cx2, cy2 = HOME[0], HOME[1] + ph
    plate = [(cx2-pw, cy2-ph), (cx2+pw, cy2-ph),
             (cx2+pw, cy2+ph*0.10), (cx2, cy2+ph), (cx2-pw, cy2+ph*0.10)]
    parts.append('<polygon points="' + ' '.join(f'{x:.2f},{y:.2f}' for x, y in plate)
                 + '" fill="#F8E8E2" stroke="#0F2A4D" stroke-width="0.3"/>')

    # Field outline
    fo = 45 + LINE_HALF
    fxL, fyL = _polar(-fo, R_OUTER); fxR, fyR = _polar(fo, R_OUTER)
    parts.append(f'<path d="M {HOME[0]},{HOME[1]} L {fxL:.2f},{fyL:.2f} '
                 f'A {R_OUTER},{R_OUTER} 0 0 1 {fxR:.2f},{fyR:.2f} Z" '
                 f'fill="none" stroke="#0F2A4D" stroke-width="0.5"/>')

    # Subtle team-logo watermark (8% opacity, behind pitcher)
    if team_id is not None:
        team_logo_path = LOGO_DIR / f'{team_id}.png'
        if team_logo_path.exists():
            try:
                tlb64 = base64.b64encode(team_logo_path.read_bytes()).decode('ascii')
                tlhref = f'data:image/png;base64,{tlb64}'
                tlw = 22
                parts.insert(2,  # before the wedges so the logo sits in back
                    f'<image href="{tlhref}" xlink:href="{tlhref}" '
                    f'x="{HOME[0]-tlw/2}" y="{HOME[1]-32}" width="{tlw}" height="{tlw}" '
                    f'opacity="0.08" preserveAspectRatio="xMidYMid meet"/>'
                )
            except Exception:
                pass

    parts.append('</svg>')
    return ''.join(parts)


def build_player_spray_svg(sport: str, division: str, ncaa_player_id,
                            player_name: str = '',
                            perspective: str = 'hitting',
                            metric_choice: str = '% of BIP') -> str:
    """Single-player wedge spray, mirroring build_team_spray_svg's visual
    language but filtered to one batter (or pitcher) by NCAA season pid."""
    spray = compute_spray_distribution(sport, division,
                                       player_id=str(ncaa_player_id),
                                       perspective=perspective)
    spray = add_zone_metrics(spray)
    if spray.empty:
        return ('<svg viewBox="0 0 100 75" xmlns="http://www.w3.org/2000/svg">'
                '<rect width="100%" height="100%" fill="#F0EAD6"/>'
                f'<text x="50" y="40" text-anchor="middle" font-family="Inter,sans-serif" '
                f'font-size="3" font-weight="700" fill="#0F2A4D">'
                f'No batted-ball data{(" for " + _xe(player_name)) if player_name else ""}</text></svg>')

    HOME = (50, 65)
    R_INNER = 8; R_MID = 25; R_OUTER = 48
    LINE_HALF = 17
    OUTFIELD = [((7, 10), -45, -9), ((8, 14), -9, 9), ((9, 11), 9, 45)]
    INFIELD  = [((5,), -45, -22.5), ((6,), -22.5, 0), ((4,), 0, 22.5), ((3,), 22.5, 45)]
    LINE     = [((12,), -45 - LINE_HALF, -45), ((13,), 45, 45 + LINE_HALF)]

    def _polar(a, r):
        rad = math.radians(a)
        return (HOME[0] + r * math.sin(rad), HOME[1] - r * math.cos(rad))

    def _wedge(a1, a2, ri, ro):
        x1i, y1i = _polar(a1, ri); x1o, y1o = _polar(a1, ro)
        x2i, y2i = _polar(a2, ri); x2o, y2o = _polar(a2, ro)
        return (f'M {x1i:.2f},{y1i:.2f} L {x1o:.2f},{y1o:.2f} '
                f'A {ro},{ro} 0 0 1 {x2o:.2f},{y2o:.2f} '
                f'L {x2i:.2f},{y2i:.2f} A {ri},{ri} 0 0 0 {x1i:.2f},{y1i:.2f} Z')

    def _pie(a1, a2, ro):
        x1, y1 = _polar(a1, ro); x2, y2 = _polar(a2, ro)
        return (f'M {HOME[0]:.2f},{HOME[1]:.2f} L {x1:.2f},{y1:.2f} '
                f'A {ro},{ro} 0 0 1 {x2:.2f},{y2:.2f} Z')

    def _label_pos(a1, a2, ri, ro):
        return _polar((a1 + a2) / 2, (ri + ro) / 2)

    def _combined(codes):
        sub = spray[spray['hitLocation'].isin(codes)]
        if sub.empty: return None
        n = int(sub['total'].sum())
        if n == 0: return None
        get = lambda c: int(sub[c].sum()) if c in sub.columns else 0
        c1, c2, c3, ch = get('1B'), get('2B'), get('3B'), get('HR')
        hits = c1 + c2 + c3 + ch
        tb = c1 + 2*c2 + 3*c3 + 4*ch
        woba = 0.888*c1 + 1.271*c2 + 1.616*c3 + 2.101*ch
        return {'pct': float(sub['pct'].sum()), 'TB': tb,
                'AVG': hits / n, 'SLG': tb / n, 'wOBA': woba / n}

    val_by, fmt_by = {}, {}
    for zc, *_ in OUTFIELD + INFIELD + LINE:
        bd = _combined(zc)
        primary = zc[0]
        if bd is None:
            val_by[primary] = 0.0
            fmt_by[primary] = '–'
        else:
            v = (bd['pct'] if metric_choice == '% of BIP'
                 else (float(bd['TB']) if metric_choice == 'TB' else bd[metric_choice]))
            val_by[primary] = v
            fmt_by[primary] = _metric_fmt(v, metric_choice)
    fair_max = max((val_by.get(zc[0], 0) for zc, *_ in OUTFIELD + INFIELD + LINE),
                   default=1.0) or 1.0

    VB_W = 100; VB_H = 75
    parts = [
        f'<svg viewBox="0 0 {VB_W} {VB_H}" width="{VB_W}" height="{VB_H}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<rect x="0" y="0" width="{VB_W}" height="{VB_H}" fill="#F0EAD6"/>',
    ]

    def _draw(zc, a1, a2, ri, ro, lf, *, pie=False):
        primary = zc[0]
        v = val_by.get(primary, 0)
        inten = v / fair_max if fair_max else 0
        d = _pie(a1, a2, ro) if pie else _wedge(a1, a2, ri, ro)
        parts.append(f'<path d="{d}" fill="{_ro(inten)}" '
                     f'stroke="#FFFFFF" stroke-width="0.5"/>')
        lx, ly = _label_pos(a1, a2, ri, ro)
        parts.append(
            f'<text x="{lx:.1f}" y="{ly+0.8:.1f}" text-anchor="middle" '
            f'font-family="Inter,sans-serif" font-size="{lf}" font-weight="800" '
            f'fill="{_tc(inten)}">{fmt_by.get(primary, "")}</text>'
        )
    for zc, a1, a2 in OUTFIELD: _draw(zc, a1, a2, R_MID, R_OUTER, 3.6)
    for zc, a1, a2 in LINE:     _draw(zc, a1, a2, R_INNER, R_OUTER, 2.6, pie=True)
    for zc, a1, a2 in INFIELD:  _draw(zc, a1, a2, R_INNER, R_MID, 2.4)

    # Pitcher disc + home plate — small diamond detail that the team-spray
    # has but the player version was missing. Mirrors build_team_spray_svg
    # lines 553-565 so the visual language stays consistent across pages.
    px, py = HOME[0], HOME[1] - 4
    parts.append(f'<circle cx="{px}" cy="{py}" r="2.5" fill="#F8E8E2" '
                 f'stroke="#0F2A4D" stroke-width="0.4"/>')
    pw, ph = 2.4, 2.0
    cx2, cy2 = HOME[0], HOME[1] + ph
    plate = [(cx2-pw, cy2-ph), (cx2+pw, cy2-ph),
             (cx2+pw, cy2+ph*0.10), (cx2, cy2+ph), (cx2-pw, cy2+ph*0.10)]
    parts.append('<polygon points="' + ' '.join(f'{x:.2f},{y:.2f}' for x, y in plate)
                 + '" fill="#F8E8E2" stroke="#0F2A4D" stroke-width="0.3"/>')

    fo = 45 + LINE_HALF
    fxL, fyL = _polar(-fo, R_OUTER); fxR, fyR = _polar(fo, R_OUTER)
    parts.append(f'<path d="M {HOME[0]},{HOME[1]} L {fxL:.2f},{fyL:.2f} '
                 f'A {R_OUTER},{R_OUTER} 0 0 1 {fxR:.2f},{fyR:.2f} Z" '
                 f'fill="none" stroke="#0F2A4D" stroke-width="0.5"/>')

    # Bottom-left aggregate stats block — same layout as Spray_Charts page.
    # AVG/SLG/wOBA are BIP-conditional (denominator = total BIP).
    def _all(c): return int(spray[c].sum()) if c in spray.columns else 0
    n_bip = int(spray['total'].sum())
    a1, a2, a3, ahr = _all('1B'), _all('2B'), _all('3B'), _all('HR')
    a_hits = a1 + a2 + a3 + ahr
    a_tb = a1 + 2*a2 + 3*a3 + 4*ahr
    a_woba_num = 0.888*a1 + 1.271*a2 + 1.616*a3 + 2.101*ahr
    if n_bip > 0:
        agg_avg, agg_slg, agg_woba = a_hits / n_bip, a_tb / n_bip, a_woba_num / n_bip
    else:
        agg_avg = agg_slg = agg_woba = 0.0
    overall = [
        ('AVG',  _metric_fmt(agg_avg, 'AVG')),
        ('SLG',  _metric_fmt(agg_slg, 'SLG')),
        ('wOBA', _metric_fmt(agg_woba, 'wOBA')),
        ('TB',   _compact(a_tb)),
    ]
    block_w = 9.5
    x0 = 1.5
    for i, (lab, val) in enumerate(overall):
        cx = x0 + block_w/2 + i * block_w
        parts.append(
            f'<text x="{cx:.1f}" y="66.5" text-anchor="middle" '
            f'font-family="Inter,sans-serif" font-size="2.0" font-weight="600" '
            f'fill="#0F2A4D" letter-spacing="0.3">{lab}</text>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="70" text-anchor="middle" '
            f'font-family="Inter,sans-serif" font-size="2.4" font-weight="800" '
            f'fill="#0F2A4D">{val}</text>'
        )

    # Bottom-right caption — scope + metric choice + BIP count
    sport_label = 'Baseball' if sport.lower() == 'baseball' else 'Softball'
    scope_txt = str(player_name) if player_name else f'Player {ncaa_player_id}'
    caption_line1 = f'{scope_txt} · {sport_label} {division}'
    caption_line2 = f'{metric_choice} · {n_bip:,} balls in play'
    parts.append(
        f'<text x="98" y="66.5" text-anchor="end" font-family="Inter,sans-serif" '
        f'font-size="2.0" font-weight="600" fill="#0F2A4D">{_xe(caption_line1)}</text>'
    )
    parts.append(
        f'<text x="98" y="70" text-anchor="end" font-family="Inter,sans-serif" '
        f'font-size="2.4" font-weight="800" fill="#0F2A4D">{_xe(caption_line2)}</text>'
    )

    parts.append('</svg>')
    return ''.join(parts)
