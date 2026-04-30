"""
Spray Charts — visualize where batted balls go for a team or player.

Uses the play_by_play data's hitLocation column (NCAA position codes 1-14)
to render a baseball-field heatmap, a bar chart, a result-by-zone matrix,
and a Left/Middle/Right side-split summary.
"""
import streamlit as st
import pandas as pd
from pathlib import Path

import sys
_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR))
from app_lib.spray_data import (
    compute_spray_distribution,
    compute_field_side_buckets,
    add_zone_metrics,
    list_teams,
    list_players,
    ZONE_COORDS,
    ZONE_NAMES,
)

BRAND_LOGO = _APP_DIR / 'assets' / 'logo-circle-black.png'

st.set_page_config(page_title='Spray Charts — 64 Analytics', layout='wide')

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    if BRAND_LOGO.exists():
        st.image(str(BRAND_LOGO), width=180)
    st.markdown('## Spray Charts')

    sport = st.selectbox('Sport', ['baseball', 'softball'], format_func=str.title)
    division = st.selectbox('Division', ['D1', 'D2', 'D3'])

    view_mode = st.radio('View', ['Team', 'Player'], horizontal=True)

    teams_df = list_teams(sport, division)
    if teams_df.empty:
        st.error(f'No PBP data found for {sport} {division}. '
                 f'Expected pbp_data/play_by_play/{sport}_play_by_play_{division}.csv '
                 f'(or .csv.gz on Render).')
        st.stop()

    # Build dropdown labels with team_id when known: "Texas Longhorns (id 856 · 5,510 BIP)"
    def team_label(row):
        tid = f"id {int(row['team_id'])}" if pd.notna(row['team_id']) else 'no id'
        return f"{row['ncaa_team']}  ({tid} · {int(row['bip']):,} BIP)"
    teams_df = teams_df.copy()
    teams_df['_label'] = teams_df.apply(team_label, axis=1)
    options = ['(all teams)'] + teams_df['_label'].tolist()
    name_by_label = dict(zip(teams_df['_label'], teams_df['ncaa_team']))
    selected_label = st.selectbox('Team', options)
    team_filter = None if selected_label == '(all teams)' else name_by_label.get(selected_label)

    selected_player_id = None
    selected_player_name = None
    if view_mode == 'Player':
        plist = list_players(sport, division, team_filter)
        if plist.empty:
            st.warning('No players with batted balls found for this scope.')
            st.stop()
        # Show top 50 by BIP volume
        labels = [f"{r['player']}  ({r['balls_in_play']} BIP)"
                  for _, r in plist.head(80).iterrows()]
        ids = plist.head(80)['playerId'].tolist()
        names = plist.head(80)['player'].tolist()
        choice = st.selectbox('Player', range(len(labels)), format_func=lambda i: labels[i])
        selected_player_id = ids[choice]
        selected_player_name = names[choice]

# ── Compute ──────────────────────────────────────────────────────────────────
spray = compute_spray_distribution(
    sport, division,
    team_name=team_filter if view_mode == 'Team' else None,
    player_id=selected_player_id if view_mode == 'Player' else None,
)
spray = add_zone_metrics(spray)
buckets = compute_field_side_buckets(spray)

# ── Header ───────────────────────────────────────────────────────────────────
title_scope = (
    f"Player: {selected_player_name}" if view_mode == 'Player'
    else (f"Team: {team_filter}" if team_filter else f"All {division} {sport}")
)
st.markdown(f"## {sport.title()} {division} — Spray Chart")
st.caption(title_scope + f" · {buckets['total']:,} balls in play")

if spray.empty:
    st.warning('No batted-ball data for this selection.')
    st.stop()

# ── Metric toggle ───────────────────────────────────────────────────────────
METRIC_OPTIONS = ['% of BIP', 'AVG', 'SLG', 'wOBA', 'TB']
metric_choice = st.radio(
    'Color & label by:', METRIC_OPTIONS, horizontal=True,
    help=('% of BIP = share of balls in play landing in that zone. '
          'AVG/SLG/wOBA are conditional on the ball reaching that zone '
          '(denominator = zone BIP). TB = raw total bases.'),
)

def _metric_value(row, choice: str) -> float:
    if choice == '% of BIP': return float(row['pct'])
    if choice == 'TB':       return float(row['TB'])
    return float(row[choice])  # AVG / SLG / wOBA

def _metric_fmt(v: float, choice: str) -> str:
    if choice == '% of BIP': return f"{v:.0f}%"
    if choice == 'TB':       return f"{int(v)}"
    # AVG/SLG/wOBA — drop the leading zero baseball-style (.350 not 0.350)
    if 0 <= v < 1:           return f".{int(round(v*1000)):03d}"
    return f"{v:.3f}"

# ── Field-diagram SVG (heatmap) + summary side-by-side ──────────────────────
col_field, col_summary = st.columns([3, 2])

with col_field:
    # ── Wedge-based spray-chart heatmap ────────────────────────────────────
    # Layout: home plate at apex (bottom). Fair territory = 90° wedge from
    # -45° to +45°. Outer ring = 5 outfield wedges (CF merges 8+14). Inner
    # ring = 4 infield wedges (3B/SS/2B/1B). Slim down-the-line wedges sit
    # just outside ±45°. Pitcher = centerline circle, catcher = home-plate
    # pentagon at the apex. Single red/orange ramp across all fair zones.
    import math

    def _combined(zone_codes):
        """Sum raw counts across one or more zones and recompute rate stats
        from the totals. Returns None if no BIP."""
        sub = spray[spray['hitLocation'].isin(zone_codes)]
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
        return {
            'BIP': n, 'pct': float(sub['pct'].sum()),
            '1B': c1, '2B': c2, '3B': c3, 'HR': ch,
            'Out': get('Out'), 'FC/Err': get('FC/Err'),
            'AVG': hits / n, 'SLG': tb / n, 'wOBA': woba / n, 'TB': tb,
        }

    def _value_for(bd, choice):
        if choice == '% of BIP': return bd['pct']
        if choice == 'TB':       return float(bd['TB'])
        return bd[choice]

    def _label_for(zone_codes):
        if zone_codes == (8, 14):
            return 'CF + Deep CF'
        return ZONE_NAMES.get(zone_codes[0], str(zone_codes[0]))

    def _tooltip_for(zone_codes) -> str:
        bd = _combined(zone_codes)
        if bd is None:
            return f"{_label_for(zone_codes)}: no BIP"
        zlist = '+'.join(str(z) for z in zone_codes)
        bits = [f"{_label_for(zone_codes)} (zone {zlist})",
                f"BIP: {bd['BIP']}  ({bd['pct']:.1f}%)"]
        for c in ['1B','2B','3B','HR','Out','FC/Err']:
            if bd.get(c, 0) > 0:
                bits.append(f"{c}: {bd[c]}")
        bits.append(f"AVG {_metric_fmt(bd['AVG'],'AVG')} · "
                    f"SLG {_metric_fmt(bd['SLG'],'SLG')} · "
                    f"wOBA {_metric_fmt(bd['wOBA'],'wOBA')} · "
                    f"TB {bd['TB']}")
        return '\n'.join(bits)

    # SVG geometry — viewBox 0 0 100 72 leaves the line wedges' outer
    # corners (at polar(-65°, 48)) just inside the left/right edges.
    HOME = (50, 65)
    R_INNER = 8    # pitcher / inner apex
    R_MID   = 25   # infield / outfield boundary
    R_OUTER = 48   # field edge
    # Angles are in degrees from straight up (toward CF). Negative = left.
    # Zone 14 (Deep CF) is merged INTO zone 8 (CF) for the chart — they
    # describe the same line of sight and zone 14 is too small (<2% BIP)
    # to read as its own wedge. The combined CF wedge sums both zones'
    # raw counts; zone 14 still appears separately in the tables and the
    # drill-down so granular detail isn't lost.
    OUTFIELD = [   # (zone_codes, ang_start, ang_end)
        ((7,),     -45, -27),
        ((10,),    -27,  -9),
        ((8, 14),   -9,   9),   # CF + Deep CF merged
        ((11,),      9,  27),
        ((9,),      27,  45),
    ]
    INFIELD = [
        ((5,), -45,   -22.5),
        ((6,), -22.5,   0),
        ((4,),   0,    22.5),
        ((3,),  22.5,  45),
    ]
    # "Down the line" wedges — NCAA codes 12/13 are FAIR balls that landed
    # close to the foul line (often XBH), not actual foul balls. Slim 20°
    # slivers just outside the foul line; their outer edges align with the
    # field-outline foul lines.
    LINE_HALF = 20
    LINE = [
        ((12,), -45 - LINE_HALF, -45),
        ((13,),  45,  45 + LINE_HALF),
    ]

    # Build value/format lookups for every wedge group + pitcher + catcher.
    # Must come AFTER the geometry (OUTFIELD/INFIELD/LINE) is defined.
    breakdowns = {}
    val_by_zone = {}
    fmt_by_zone = {}
    for zone_codes, *_ in OUTFIELD + INFIELD + LINE + [((1,), 0, 0), ((2,), 0, 0)]:
        bd = _combined(zone_codes)
        primary = zone_codes[0]
        breakdowns[primary] = (zone_codes, bd)
        if bd is None:
            val_by_zone[primary] = 0.0
            fmt_by_zone[primary] = '–'
        else:
            v = _value_for(bd, metric_choice)
            val_by_zone[primary] = v
            fmt_by_zone[primary] = _metric_fmt(v, metric_choice)

    def polar(angle_deg, radius):
        # angle 0 = straight up; positive = clockwise (right)
        rad = math.radians(angle_deg)
        return (HOME[0] + radius * math.sin(rad), HOME[1] - radius * math.cos(rad))

    def wedge_path(a1, a2, r_in, r_out):
        x1i, y1i = polar(a1, r_in)
        x1o, y1o = polar(a1, r_out)
        x2i, y2i = polar(a2, r_in)
        x2o, y2o = polar(a2, r_out)
        large = 1 if (a2 - a1) > 180 else 0
        return (f"M {x1i:.2f},{y1i:.2f} L {x1o:.2f},{y1o:.2f} "
                f"A {r_out},{r_out} 0 {large} 1 {x2o:.2f},{y2o:.2f} "
                f"L {x2i:.2f},{y2i:.2f} A {r_in},{r_in} 0 {large} 0 {x1i:.2f},{y1i:.2f} Z")

    def pie_path(a1, a2, r_out):
        """Pie wedge from the apex (HOME) outward — used for the foul zones
        so they form proper diamond corners instead of annular sectors with
        a curved gap near the apex."""
        x1, y1 = polar(a1, r_out)
        x2, y2 = polar(a2, r_out)
        large = 1 if (a2 - a1) > 180 else 0
        return (f"M {HOME[0]:.2f},{HOME[1]:.2f} L {x1:.2f},{y1:.2f} "
                f"A {r_out},{r_out} 0 {large} 1 {x2:.2f},{y2:.2f} Z")

    def label_pos(a1, a2, r_in, r_out):
        ang = (a1 + a2) / 2
        radius = (r_in + r_out) / 2
        return polar(ang, radius)

    # Single fair-territory color ramp: light orange → deep red.
    def red_orange(intensity):
        i = max(0.05, min(1.0, intensity))
        r = int(255 - (255 - 198) * i)
        g = int(224 - (224 - 40)  * i)
        b = int(204 - (204 - 40)  * i)
        return f'#{r:02X}{g:02X}{b:02X}'
    def text_color(intensity):
        return '#FFFFFF' if intensity > 0.55 else '#0F2A4D'

    # Single intensity scale across ALL fair-territory zones (infield,
    # outfield, down-the-line) so identical metric values render identical
    # shades.
    def max_val(zones):
        return max((val_by_zone.get(zc[0], 0) for zc, *_ in zones), default=1.0) or 1.0
    fair_max = max_val(OUTFIELD + INFIELD + LINE)

    parts = ['''<svg viewBox="0 0 100 72" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;background:#F0EAD6;border-radius:8px;">''']
    parts.append('<text x="50" y="7" text-anchor="middle" font-family="Inter,sans-serif" font-size="5" font-weight="800" fill="#0F2A4D">Spray Chart</text>')

    def _draw_wedge(zone_codes, a1, a2, r_in, r_out, label_font, *, pie=False):
        primary = zone_codes[0]
        v = val_by_zone.get(primary, 0)
        intensity = v / fair_max if fair_max else 0
        fill = red_orange(intensity)
        path_d = pie_path(a1, a2, r_out) if pie else wedge_path(a1, a2, r_in, r_out)
        parts.append(f'<path d="{path_d}" fill="{fill}" stroke="#FFFFFF" stroke-width="0.6"><title>{_tooltip_for(zone_codes)}</title></path>')
        lx, ly = label_pos(a1, a2, r_in, r_out)
        tc = text_color(intensity)
        parts.append(f'<text x="{lx:.1f}" y="{ly+1.0:.1f}" text-anchor="middle" font-family="Inter,sans-serif" font-size="{label_font}" font-weight="800" fill="{tc}" pointer-events="none">{fmt_by_zone.get(primary, "")}</text>')

    # Outfield (CF wedge spans 8+14)
    for zc, a1, a2 in OUTFIELD:
        _draw_wedge(zc, a1, a2, R_MID, R_OUTER, 4.5)
    # Down-the-line wedges (zones 12/13) — fair territory, pie wedges from apex
    for zc, a1, a2 in LINE:
        _draw_wedge(zc, a1, a2, R_INNER, R_OUTER, 3.5, pie=True)
    # Infield
    for zc, a1, a2 in INFIELD:
        _draw_wedge(zc, a1, a2, R_INNER, R_MID, 2.8)

    # Pitcher (zone 1) — circle on the centerline, sized so its bottom edge
    # meets the catcher's top edge for a clean stacked diamond apex.
    r_p = 3.0
    px, py = HOME[0], HOME[1] - 5   # center 5 units above HOME → bottom at y=63
    parts.append(f'<circle cx="{px}" cy="{py}" r="{r_p}" fill="#F8E8E2" stroke="#0F2A4D" stroke-width="0.5"><title>{_tooltip_for((1,))}</title></circle>')
    parts.append(f'<text x="{px}" y="{py+0.9:.1f}" text-anchor="middle" font-family="Inter,sans-serif" font-size="2.4" font-weight="800" fill="#0F2A4D" pointer-events="none">{fmt_by_zone.get(1,"")}</text>')

    # Catcher (zone 2) — home-plate pentagon, positioned ENTIRELY BELOW the
    # foul-line apex so its number stays clear of the V where the foul
    # wedges meet. Front (flat) edge sits at HOME, point extends downward.
    pw, ph = 2.8, 2.6   # half-width, half-height
    cx, cy = HOME[0], HOME[1] + ph   # top of plate aligns with HOME y
    plate_pts = [
        (cx - pw, cy - ph),         # front-left (flat edge facing pitcher)
        (cx + pw, cy - ph),         # front-right
        (cx + pw, cy + ph * 0.10),  # right shoulder
        (cx,      cy + ph),         # back point (toward catcher)
        (cx - pw, cy + ph * 0.10),  # left shoulder
    ]
    pts_str = ' '.join(f'{x:.2f},{y:.2f}' for x, y in plate_pts)
    parts.append(f'<polygon points="{pts_str}" fill="#F8E8E2" stroke="#0F2A4D" stroke-width="0.4"><title>{_tooltip_for((2,))}</title></polygon>')
    parts.append(f'<text x="{cx}" y="{cy+0.4:.1f}" text-anchor="middle" font-family="Inter,sans-serif" font-size="2.2" font-weight="800" fill="#0F2A4D" pointer-events="none">{fmt_by_zone.get(2,"")}</text>')

    # Single continuous field outline — foul line L, outfield arc, foul line R,
    # all sharing the same endpoints as the foul-wedge outer corners. This is
    # the merged diamond/semi-circle outline the design calls for.
    foul_outer = 45 + LINE_HALF
    fxL, fyL = polar(-foul_outer, R_OUTER)
    fxR, fyR = polar( foul_outer, R_OUTER)
    parts.append(
        f'<path d="M {HOME[0]},{HOME[1]} L {fxL:.2f},{fyL:.2f} '
        f'A {R_OUTER},{R_OUTER} 0 0 1 {fxR:.2f},{fyR:.2f} Z" '
        f'fill="none" stroke="#0F2A4D" stroke-width="0.7"/>'
    )

    parts.append('</svg>')
    st.markdown(''.join(parts), unsafe_allow_html=True)

with col_summary:
    st.markdown('### Field-side splits')
    side_df = pd.DataFrame([
        ['Left side',  '3B/SS/LF/LCF/L Line', buckets['left'],   f"{buckets['left_pct']}%"],
        ['Middle',     'P/2B/CF/Deep CF',     buckets['middle'], f"{buckets['middle_pct']}%"],
        ['Right side', '1B/RF/RCF/R Line',    buckets['right'],  f"{buckets['right_pct']}%"],
        ['Other (C)',  '2',                   buckets['other'],  f"{buckets['other_pct']}%"],
    ], columns=['Side', 'Zones', 'Count', 'Share'])
    st.dataframe(side_df, hide_index=True, use_container_width=True)

    st.markdown('### Top zones')
    top = spray.sort_values('total', ascending=False).head(5)[
        ['zone_name', 'total', 'pct']
    ].rename(columns={'zone_name': 'Zone', 'total': 'Count', 'pct': '%'})
    st.dataframe(top, hide_index=True, use_container_width=True)

# ── Zone drill-down card ────────────────────────────────────────────────────
st.markdown('### Zone detail')
zone_options = spray.sort_values('total', ascending=False)
zone_labels = [f"{r['zone_name']} (zone {int(r['hitLocation'])}) — {int(r['total'])} BIP"
               for _, r in zone_options.iterrows()]
zone_ids = zone_options['hitLocation'].tolist()
if zone_labels:
    pick = st.selectbox('Show details for:', range(len(zone_labels)),
                        format_func=lambda i: zone_labels[i], key='zone_drill')
    z = int(zone_ids[pick])
    row = spray[spray['hitLocation'] == z].iloc[0]
    cols = st.columns(8)
    metrics = [
        ('BIP',  f"{int(row['total'])}", f"{row['pct']:.1f}% of all"),
        ('1B',   f"{int(row.get('1B', 0))}", ''),
        ('2B',   f"{int(row.get('2B', 0))}", ''),
        ('3B',   f"{int(row.get('3B', 0))}", ''),
        ('HR',   f"{int(row.get('HR', 0))}", ''),
        ('AVG',  _metric_fmt(float(row['AVG']),  'AVG'),  ''),
        ('SLG',  _metric_fmt(float(row['SLG']),  'SLG'),  ''),
        ('wOBA', _metric_fmt(float(row['wOBA']), 'wOBA'), f"TB {int(row['TB'])}"),
    ]
    for col, (label, val, sub) in zip(cols, metrics):
        col.metric(label, val, sub if sub else None, delta_color='off')

# ── Result-by-zone matrix ────────────────────────────────────────────────────
st.markdown('### Hit results by zone')
matrix_cols = [c for c in ['1B', '2B', '3B', 'HR', 'Out', 'FC/Err', 'Other'] if c in spray.columns]
if matrix_cols:
    display = spray[['zone_name', 'total'] + matrix_cols + ['AVG', 'SLG', 'wOBA', 'TB']].copy()
    display = display.rename(columns={'zone_name': 'Zone', 'total': 'BIP'})
    for c in ['AVG', 'SLG', 'wOBA']:
        display[c] = display[c].apply(lambda v: _metric_fmt(float(v), c))
    st.dataframe(display, hide_index=True, use_container_width=True)

# ── Bar chart of zone distribution ──────────────────────────────────────────
st.markdown('### Distribution by zone')
chart_df = spray[['zone_name', 'total']].rename(columns={'zone_name': 'Zone', 'total': 'Count'})
st.bar_chart(chart_df, x='Zone', y='Count', height=240)

# ── Footer note ─────────────────────────────────────────────────────────────
st.caption(
    'Zone codes: 1=P, 2=C, 3=1B, 4=2B, 5=3B, 6=SS, 7=LF, 8=CF, 9=RF, '
    '10=LCF gap, 11=RCF gap, 12=L Line, 13=R Line, 14=Deep CF. '
    'Zones 12 / 13 are FAIR (down-the-line hits), not foul; zone 14 is '
    'over-the-fence / deep CF. Without batter handedness we can\'t '
    'classify Pull/Oppo, so Left/Middle/Right is the unbiased split.'
)
