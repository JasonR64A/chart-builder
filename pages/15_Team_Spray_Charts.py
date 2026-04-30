"""
Team Spray Charts — 3×3 grid of the nine highest-BIP batters on a team.

Reuses the data layer from app_lib/spray_data.py. Each cell is a compact
spray diamond (no per-cell logo, no per-cell scope caption); a single
64 Analytics logo sits at the top of the page along with the team name.
"""
import base64
import io
import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR))
from app_lib.spray_data import (
    add_zone_metrics,
    compute_spray_distribution,
    list_players,
    list_teams,
)

BRAND_64A_WIDE = _APP_DIR / 'assets' / 'logo-64a-wide.png'

st.set_page_config(page_title='Team Spray Charts — 64 Analytics', layout='wide')


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## Team Spray Charts')
    sport = st.selectbox('Sport', ['baseball', 'softball'], format_func=str.title)
    division = st.selectbox('Division', ['D1', 'D2', 'D3'])

    teams_df = list_teams(sport, division)
    if teams_df.empty:
        st.error(f'No PBP data found for {sport} {division}.')
        st.stop()

    def _team_label(row):
        short = row['short_name'] if pd.notna(row.get('short_name')) else row['ncaa_team']
        return f"{short}  ({int(row['bip']):,} BIP)"
    teams_df = teams_df.copy()
    teams_df['_label'] = teams_df.apply(_team_label, axis=1)
    teams_df = teams_df.sort_values('bip', ascending=False)

    label_to_team = dict(zip(teams_df['_label'], teams_df['ncaa_team']))
    label_to_short = dict(zip(teams_df['_label'], teams_df['short_name']))
    label_to_tid   = dict(zip(teams_df['_label'], teams_df['team_id']))

    selected_label = st.selectbox('Team', teams_df['_label'].tolist())
    team_filter = label_to_team[selected_label]
    short_name = label_to_short.get(selected_label) or team_filter
    if pd.isna(short_name):
        short_name = team_filter
    selected_team_id = label_to_tid.get(selected_label)
    if pd.notna(selected_team_id):
        selected_team_id = int(selected_team_id)
    else:
        selected_team_id = None

    metric_choice = st.radio(
        'Color & label by:', ['% of BIP', 'AVG', 'SLG', 'wOBA', 'TB'], horizontal=True
    )

    st.markdown('---')
    st.markdown('### Filters')
    f_hand = st.radio('Pitcher hand', ['Any', 'vs LHP', 'vs RHP'], horizontal=True)
    f_two_strikes = st.checkbox('2-strike counts only')
    f_two_outs = st.checkbox('2-out situations only')
    f_risp = st.checkbox('Runners in scoring position')

vs_hand = {'vs LHP': 'L', 'vs RHP': 'R'}.get(f_hand)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _embed_image(path: Path, x: float, y: float, w: float, h: float,
                 opacity: float = 1.0) -> str:
    if not path.exists():
        return ''
    try:
        b64 = base64.b64encode(path.read_bytes()).decode('ascii')
    except Exception:
        return ''
    href = f'data:image/png;base64,{b64}'
    return (f'<image href="{href}" xlink:href="{href}" '
            f'x="{x}" y="{y}" width="{w}" height="{h}" '
            f'opacity="{opacity}" preserveAspectRatio="xMidYMid meet" '
            f'pointer-events="none"/>')


def _red_orange(intensity: float) -> str:
    i = max(0.05, min(1.0, intensity))
    r = int(255 - (255 - 198) * i)
    g = int(224 - (224 - 40)  * i)
    b = int(204 - (204 - 40)  * i)
    return f'#{r:02X}{g:02X}{b:02X}'


def _text_color(intensity: float) -> str:
    return '#FFFFFF' if intensity > 0.55 else '#0F2A4D'


def _metric_value(row, choice: str) -> float:
    if choice == '% of BIP': return float(row['pct'])
    if choice == 'TB':       return float(row['TB'])
    return float(row[choice])


def _metric_fmt(v: float, choice: str) -> str:
    if choice == '% of BIP': return f"{v:.0f}%"
    if choice == 'TB':       return f"{int(v)}"
    if 0 <= v < 1:           return f".{int(round(v*1000)):03d}"
    return f"{v:.3f}"


# Each cell is 100×84 viewBox: y=0..10 = player name strip, y=10..82 = diamond.
# Diamond constants are local to the cell (HOME at y=72, R_OUTER=44).
HOME   = (50, 72)
R_INNER = 7
R_MID   = 22
R_OUTER = 44
LINE_HALF = 15

OUTFIELD = [
    ((7, 10),  -45, -9),
    ((8, 14),   -9,  9),
    ((9, 11),    9, 45),
]
INFIELD = [
    ((5,), -45,   -22.5),
    ((6,), -22.5,   0),
    ((4,),   0,    22.5),
    ((3,),  22.5,  45),
]
LINE = [
    ((12,), -45 - LINE_HALF, -45),
    ((13,),  45,  45 + LINE_HALF),
]


def _polar(angle_deg, radius):
    rad = math.radians(angle_deg)
    return (HOME[0] + radius * math.sin(rad),
            HOME[1] - radius * math.cos(rad))


def _wedge_path(a1, a2, r_in, r_out):
    x1i, y1i = _polar(a1, r_in)
    x1o, y1o = _polar(a1, r_out)
    x2i, y2i = _polar(a2, r_in)
    x2o, y2o = _polar(a2, r_out)
    large = 1 if (a2 - a1) > 180 else 0
    return (f"M {x1i:.2f},{y1i:.2f} L {x1o:.2f},{y1o:.2f} "
            f"A {r_out},{r_out} 0 {large} 1 {x2o:.2f},{y2o:.2f} "
            f"L {x2i:.2f},{y2i:.2f} A {r_in},{r_in} 0 {large} 0 {x1i:.2f},{y1i:.2f} Z")


def _pie_path(a1, a2, r_out):
    x1, y1 = _polar(a1, r_out)
    x2, y2 = _polar(a2, r_out)
    large = 1 if (a2 - a1) > 180 else 0
    return (f"M {HOME[0]:.2f},{HOME[1]:.2f} L {x1:.2f},{y1:.2f} "
            f"A {r_out},{r_out} 0 {large} 1 {x2:.2f},{y2:.2f} Z")


def _label_pos(a1, a2, r_in, r_out):
    ang = (a1 + a2) / 2
    radius = (r_in + r_out) / 2
    return _polar(ang, radius)


def _combined(spray, zone_codes):
    sub = spray[spray['hitLocation'].isin(zone_codes)]
    if sub.empty: return None
    n = int(sub['total'].sum())
    if n == 0: return None
    get = lambda c: int(sub[c].sum()) if c in sub.columns else 0
    c1, c2, c3, ch = get('1B'), get('2B'), get('3B'), get('HR')
    tb = c1 + 2*c2 + 3*c3 + 4*ch
    hits = c1 + c2 + c3 + ch
    woba = 0.888*c1 + 1.271*c2 + 1.616*c3 + 2.101*ch
    return {
        'BIP': n, 'pct': float(sub['pct'].sum()),
        'AVG': hits/n, 'SLG': tb/n, 'wOBA': woba/n, 'TB': tb,
    }


def _build_cell(spray: pd.DataFrame, name: str, sub_label: str,
                metric: str) -> str:
    """Return the SVG <g> fragment for one player cell (in the cell's
    own 100×84 coordinate space; the parent SVG positions the cell)."""
    parts = []

    # Player name (top-left) + BIP count (top-right)
    parts.append(
        f'<text x="3" y="6" font-family="Inter,sans-serif" font-size="4" '
        f'font-weight="800" fill="#0F2A4D">{name}</text>'
    )
    if sub_label:
        parts.append(
            f'<text x="97" y="6" text-anchor="end" font-family="Inter,sans-serif" '
            f'font-size="3" font-weight="600" fill="#0F2A4D">{sub_label}</text>'
        )

    # Eggshell background for the cell area
    parts.append('<rect x="2" y="9" width="96" height="74" '
                 'rx="2" ry="2" fill="#F0EAD6" stroke="#E0D9C0" stroke-width="0.3"/>')

    if spray is None or spray.empty:
        parts.append(
            f'<text x="50" y="48" text-anchor="middle" font-family="Inter,sans-serif" '
            f'font-size="3" font-weight="600" fill="#0F2A4D">No batted-ball data</text>'
        )
        return ''.join(parts)

    # Per-zone metric values
    val_by_zone = {}
    fmt_by_zone = {}
    for zc, *_ in OUTFIELD + INFIELD + LINE + [((1,), 0, 0), ((2,), 0, 0)]:
        bd = _combined(spray, zc)
        primary = zc[0]
        if bd is None:
            val_by_zone[primary] = 0.0
            fmt_by_zone[primary] = '–'
        else:
            v = bd['pct'] if metric == '% of BIP' else (
                float(bd['TB']) if metric == 'TB' else bd[metric]
            )
            val_by_zone[primary] = v
            fmt_by_zone[primary] = _metric_fmt(v, metric)

    fair_max = max((val_by_zone.get(zc[0], 0)
                    for zc, *_ in OUTFIELD + INFIELD + LINE), default=1.0) or 1.0

    def _draw(zc, a1, a2, r_in, r_out, label_font, *, pie=False):
        primary = zc[0]
        v = val_by_zone.get(primary, 0)
        intensity = v / fair_max if fair_max else 0
        fill = _red_orange(intensity)
        d = _pie_path(a1, a2, r_out) if pie else _wedge_path(a1, a2, r_in, r_out)
        parts.append(
            f'<path d="{d}" fill="{fill}" stroke="#FFFFFF" stroke-width="0.5"/>'
        )
        lx, ly = _label_pos(a1, a2, r_in, r_out)
        tc = _text_color(intensity)
        parts.append(
            f'<text x="{lx:.1f}" y="{ly+0.9:.1f}" text-anchor="middle" '
            f'font-family="Inter,sans-serif" font-size="{label_font}" '
            f'font-weight="800" fill="{tc}">{fmt_by_zone.get(primary, "")}</text>'
        )

    for zc, a1, a2 in OUTFIELD:
        _draw(zc, a1, a2, R_MID, R_OUTER, 3.5)
    for zc, a1, a2 in LINE:
        _draw(zc, a1, a2, R_INNER, R_OUTER, 2.4, pie=True)
    for zc, a1, a2 in INFIELD:
        _draw(zc, a1, a2, R_INNER, R_MID, 2.0)

    # Pitcher (zone 1) circle
    p_pct = fmt_by_zone.get(1, '')
    px, py = HOME[0], HOME[1] - 4.5
    parts.append(
        f'<circle cx="{px}" cy="{py}" r="2.6" fill="#F8E8E2" '
        f'stroke="#0F2A4D" stroke-width="0.4"/>'
    )
    parts.append(
        f'<text x="{px}" y="{py+0.8:.1f}" text-anchor="middle" '
        f'font-family="Inter,sans-serif" font-size="2.0" font-weight="800" '
        f'fill="#0F2A4D">{p_pct}</text>'
    )

    # Catcher (zone 2) home-plate pentagon
    c_pct = fmt_by_zone.get(2, '')
    pw, ph = 2.5, 2.3
    cx, cy = HOME[0], HOME[1] + ph
    plate_pts = [
        (cx - pw, cy - ph),
        (cx + pw, cy - ph),
        (cx + pw, cy + ph * 0.10),
        (cx,      cy + ph),
        (cx - pw, cy + ph * 0.10),
    ]
    pts_str = ' '.join(f'{x:.2f},{y:.2f}' for x, y in plate_pts)
    parts.append(
        f'<polygon points="{pts_str}" fill="#F8E8E2" stroke="#0F2A4D" stroke-width="0.3"/>'
    )
    parts.append(
        f'<text x="{cx}" y="{cy+0.4:.1f}" text-anchor="middle" '
        f'font-family="Inter,sans-serif" font-size="1.4" font-weight="800" '
        f'fill="#0F2A4D">{c_pct}</text>'
    )

    # Single field outline
    foul_outer = 45 + LINE_HALF
    fxL, fyL = _polar(-foul_outer, R_OUTER)
    fxR, fyR = _polar( foul_outer, R_OUTER)
    parts.append(
        f'<path d="M {HOME[0]},{HOME[1]} L {fxL:.2f},{fyL:.2f} '
        f'A {R_OUTER},{R_OUTER} 0 0 1 {fxR:.2f},{fyR:.2f} Z" '
        f'fill="none" stroke="#0F2A4D" stroke-width="0.5"/>'
    )

    return ''.join(parts)


# ── Compute per-player sprays ───────────────────────────────────────────────
plist = list_players(sport, division, team_filter)
if plist.empty:
    st.error(f'No players with batted balls found for {short_name}.')
    st.stop()

top9 = plist.head(9).reset_index(drop=True)

players_data = []
for _, p in top9.iterrows():
    pid = p['playerId']
    spray = compute_spray_distribution(
        sport, division,
        team_name=team_filter, player_id=pid,
        vs_hand=vs_hand,
        two_strikes=f_two_strikes, two_outs=f_two_outs, risp=f_risp,
    )
    spray = add_zone_metrics(spray)
    full_name = p.get('player_name')
    name = full_name if isinstance(full_name, str) and full_name.strip() else p['player']
    pos = p.get('position')
    cls = p.get('classification')
    bits = []
    if isinstance(pos, str) and pos: bits.append(pos)
    if isinstance(cls, str) and cls: bits.append(cls)
    bits.append(f"{int(spray['total'].sum()) if not spray.empty else 0} BIP")
    sub_label = ' · '.join(bits)
    players_data.append({'name': name, 'sub': sub_label, 'spray': spray})


# ── Build parent SVG ────────────────────────────────────────────────────────
# Layout: header (y=0..28) + 3 rows × 84 = 252 → total 280.
# Cell width = 100 → 3 cols = 300 wide. Add 2u margin each side = 304.
HDR_H = 28
CELL_W, CELL_H = 100, 84
COLS, ROWS = 3, 3
GRID_W = COLS * CELL_W
GRID_H = ROWS * CELL_H
VB_W   = GRID_W + 4   # 2u margin each side
VB_H   = HDR_H + GRID_H + 4

parts = [
    f'<svg viewBox="0 0 {VB_W} {VB_H}" width="{VB_W}" height="{VB_H}" '
    f'xmlns="http://www.w3.org/2000/svg" '
    f'xmlns:xlink="http://www.w3.org/1999/xlink">'
    f'<rect x="0" y="0" width="{VB_W}" height="{VB_H}" fill="#FFFFFF"/>'
]

# 64A logo top-center
parts.append(_embed_image(BRAND_64A_WIDE, x=VB_W/2 - 19, y=2, w=38, h=9))

# Team name + scope
sport_label = 'Baseball' if sport.lower() == 'baseball' else 'Softball'
filter_bits = []
if vs_hand: filter_bits.append('vs LHP' if vs_hand == 'L' else 'vs RHP')
if f_two_strikes: filter_bits.append('2 strikes')
if f_two_outs:    filter_bits.append('2 outs')
if f_risp:        filter_bits.append('RISP')
filter_txt = ' · ' + ', '.join(filter_bits) if filter_bits else ''
header_line = f'{short_name} · {sport_label} {division}{filter_txt} · Top 9 by BIP'
parts.append(
    f'<text x="{VB_W/2}" y="18" text-anchor="middle" font-family="Inter,sans-serif" '
    f'font-size="4.5" font-weight="800" fill="#0F2A4D">{header_line}</text>'
)
parts.append(
    f'<text x="{VB_W/2}" y="24" text-anchor="middle" font-family="Inter,sans-serif" '
    f'font-size="3" font-weight="500" fill="#0F2A4D">'
    f'Coloring by {metric_choice}</text>'
)

# 3×3 cells
GRID_X0 = (VB_W - GRID_W) / 2
for i, pdata in enumerate(players_data):
    row, col = divmod(i, COLS)
    cx = GRID_X0 + col * CELL_W
    cy = HDR_H + row * CELL_H
    parts.append(f'<g transform="translate({cx},{cy})">')
    parts.append(_build_cell(pdata['spray'], pdata['name'], pdata['sub'], metric_choice))
    parts.append('</g>')

# Pad with empty cells if fewer than 9 players
for i in range(len(players_data), 9):
    row, col = divmod(i, COLS)
    cx = GRID_X0 + col * CELL_W
    cy = HDR_H + row * CELL_H
    parts.append(
        f'<rect x="{cx + 2}" y="{cy + 9}" width="96" height="74" '
        f'rx="2" ry="2" fill="#F8F8F8" stroke="#E0E0E0" stroke-width="0.3"/>'
    )

parts.append('</svg>')
svg_str = ''.join(parts)

# Display SVG with responsive width
display_svg = svg_str.replace(
    '<svg ',
    '<svg style="width:100%;max-width:1200px;height:auto;display:block;margin:0 auto;border-radius:8px;" ',
    1,
)
st.markdown(display_svg, unsafe_allow_html=True)


# ── PNG download ────────────────────────────────────────────────────────────
try:
    import cairosvg
    png_bytes = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'), output_width=2000)
    safe_team = ''.join(c if c.isalnum() else '_' for c in str(short_name))[:40]
    fname = f'spray_team_{sport}_{division}_{safe_team}.png'
    st.download_button('Download PNG', data=png_bytes, file_name=fname,
                       mime='image/png', use_container_width=False)
except Exception as e:
    st.caption(f'PNG export unavailable in this environment ({type(e).__name__}).')


# ── Player table below ──────────────────────────────────────────────────────
st.markdown('### Top 9 batters by balls in play')
table_rows = []
for p in players_data:
    sp = p['spray']
    if sp is None or sp.empty:
        table_rows.append({'Player': p['name'], 'BIP': 0, 'AVG': '-', 'SLG': '-', 'wOBA': '-', 'TB': 0})
        continue
    g = lambda c: int(sp[c].sum()) if c in sp.columns else 0
    n = int(sp['total'].sum())
    c1, c2, c3, ch = g('1B'), g('2B'), g('3B'), g('HR')
    hits = c1 + c2 + c3 + ch
    tb = c1 + 2*c2 + 3*c3 + 4*ch
    woba = 0.888*c1 + 1.271*c2 + 1.616*c3 + 2.101*ch
    table_rows.append({
        'Player': p['name'],
        'BIP': n,
        'AVG': _metric_fmt(hits/n if n else 0, 'AVG'),
        'SLG': _metric_fmt(tb/n if n else 0, 'SLG'),
        'wOBA': _metric_fmt(woba/n if n else 0, 'wOBA'),
        'TB':  tb,
    })
st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)
