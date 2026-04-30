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

# ── Field-diagram SVG (heatmap) + summary side-by-side ──────────────────────
col_field, col_summary = st.columns([3, 2])

with col_field:
    # ── Wedge-based spray-chart heatmap ────────────────────────────────────
    # Layout (matches the reference design provided 2026-04-29):
    #   Home plate at apex (bottom). Fair territory = 90° wedge from -45° to +45°.
    #   Outer ring (outfield): 5 slices LF/LCF/CF/RCF/RF — red gradient
    #   Inner ring (infield):  5 slices 3B/SS/14-UTM/2B/1B — pink gradient
    #   Foul wedges (12,13):   outside the foul lines — blue gradient
    #   Pitcher (1):           circle on the mound
    #   Catcher (2):           octagon at home plate
    import math
    pct_by_zone = dict(zip(spray['hitLocation'], spray['pct']))

    # SVG geometry — viewBox 0 0 100 72 so the field fits horizontally:
    # the foul wedges extend from -90° to -45°, which means the leftmost
    # outer corner of L FOUL is at polar(-90°, R_OUTER). Setting R_OUTER=48
    # puts that corner at x=2 (just inside the left edge).
    HOME = (50, 65)
    R_INNER = 8    # pitcher / inner apex
    R_MID   = 25   # infield / outfield boundary
    R_OUTER = 48   # field edge
    # Angles are in degrees from straight up (toward CF). Negative = left.
    OUTFIELD = [   # (zone, ang_start, ang_end, label)
        (7,  -45, -27, 'LF'),
        (10, -27,  -9, 'LCF'),
        (8,   -9,   9, 'CF'),
        (11,   9,  27, 'RCF'),
        (9,   27,  45, 'RF'),
    ]
    INFIELD = [
        (5,  -45, -27, '3B'),
        (6,  -27,  -9, 'SS'),
        (14,  -9,   9, 'UTM'),
        (4,    9,  27, '2B'),
        (3,   27,  45, '1B'),
    ]
    FOUL = [
        # Narrower foul wedges per reference design — 30° each, not 45°.
        (12, -75, -45, 'L FOUL'),
        (13,  45,  75, 'R FOUL'),
    ]

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

    # Color helpers — 0..1 intensity blends from a soft tint to a deep base
    def red_orange(intensity):
        # Light orange (#FFE0CC) → deep red (#C62828)
        i = max(0.05, min(1.0, intensity))
        r = int(255 - (255 - 198) * i)
        g = int(224 - (224 - 40)  * i)
        b = int(204 - (204 - 40)  * i)
        return f'#{r:02X}{g:02X}{b:02X}'
    def pink(intensity):
        i = max(0.05, min(1.0, intensity))
        r = int(252 - (252 - 230) * i)
        g = int(228 - (228 - 170) * i)
        b = int(228 - (228 - 170) * i)
        return f'#{r:02X}{g:02X}{b:02X}'
    def blue(intensity):
        i = max(0.05, min(1.0, intensity))
        r = int(220 - (220 - 130) * i)
        g = int(232 - (232 - 175) * i)
        b = int(248 - (248 - 220) * i)
        return f'#{r:02X}{g:02X}{b:02X}'
    def text_color(intensity):
        return '#FFFFFF' if intensity > 0.55 else '#0F2A4D'

    # Compute intensity per ring relative to that ring's max
    def max_pct(zones):
        return max((pct_by_zone.get(z, 0) for z, *_ in zones), default=1.0) or 1.0
    of_max = max_pct(OUTFIELD)
    if_max = max_pct(INFIELD)
    foul_max = max_pct(FOUL)

    parts = ['''<svg viewBox="0 0 100 72" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;background:#F0EAD6;border-radius:8px;">''']
    parts.append('<text x="50" y="7" text-anchor="middle" font-family="Inter,sans-serif" font-size="5" font-weight="800" fill="#0F2A4D">Spray Chart</text>')

    # Outfield wedges
    for z, a1, a2, label in OUTFIELD:
        pct = pct_by_zone.get(z, 0)
        intensity = pct / of_max if of_max else 0
        fill = red_orange(intensity)
        parts.append(f'<path d="{wedge_path(a1, a2, R_MID, R_OUTER)}" fill="{fill}" stroke="#FFFFFF" stroke-width="0.6"/>')
        lx, ly = label_pos(a1, a2, R_MID, R_OUTER)
        tc = text_color(intensity)
        parts.append(f'<text x="{lx:.1f}" y="{ly-1.5:.1f}" text-anchor="middle" font-family="Inter,sans-serif" font-size="3.3" font-weight="700" fill="{tc}">{label}</text>')
        parts.append(f'<text x="{lx:.1f}" y="{ly+3.0:.1f}" text-anchor="middle" font-family="Inter,sans-serif" font-size="4.5" font-weight="800" fill="{tc}">{pct:.0f}%</text>')

    # Foul wedges — pie wedges from the apex (true diamond corners)
    for z, a1, a2, label in FOUL:
        pct = pct_by_zone.get(z, 0)
        intensity = pct / foul_max if foul_max else 0
        fill = blue(intensity)
        parts.append(f'<path d="{pie_path(a1, a2, R_OUTER)}" fill="{fill}" stroke="#FFFFFF" stroke-width="0.6"/>')
        lx, ly = label_pos(a1, a2, R_INNER, R_OUTER)
        tc = text_color(intensity * 0.7)
        parts.append(f'<text x="{lx:.1f}" y="{ly-1.5:.1f}" text-anchor="middle" font-family="Inter,sans-serif" font-size="3" font-weight="700" fill="{tc}">{label}</text>')
        parts.append(f'<text x="{lx:.1f}" y="{ly+2.5:.1f}" text-anchor="middle" font-family="Inter,sans-serif" font-size="4" font-weight="800" fill="{tc}">{pct:.0f}%</text>')

    # Infield wedges
    for z, a1, a2, label in INFIELD:
        pct = pct_by_zone.get(z, 0)
        intensity = pct / if_max if if_max else 0
        fill = pink(intensity)
        parts.append(f'<path d="{wedge_path(a1, a2, R_INNER, R_MID)}" fill="{fill}" stroke="#FFFFFF" stroke-width="0.6"/>')
        lx, ly = label_pos(a1, a2, R_INNER, R_MID)
        tc = '#0F2A4D'
        parts.append(f'<text x="{lx:.1f}" y="{ly-1.0:.1f}" text-anchor="middle" font-family="Inter,sans-serif" font-size="2.8" font-weight="700" fill="{tc}">{label}</text>')
        parts.append(f'<text x="{lx:.1f}" y="{ly+2.5:.1f}" text-anchor="middle" font-family="Inter,sans-serif" font-size="3.6" font-weight="800" fill="{tc}">{pct:.0f}%</text>')

    # Pitcher (zone 1) — circle on the centerline, sized so its bottom edge
    # meets the catcher's top edge for a clean stacked diamond apex.
    p_pct = pct_by_zone.get(1, 0)
    r_p = 3.0
    px, py = HOME[0], HOME[1] - 5   # center 5 units above HOME → bottom at y=63
    parts.append(f'<circle cx="{px}" cy="{py}" r="{r_p}" fill="#F8E8E2" stroke="#0F2A4D" stroke-width="0.5"/>')
    parts.append(f'<text x="{px}" y="{py-0.4:.1f}" text-anchor="middle" font-family="Inter,sans-serif" font-size="1.8" font-weight="700" fill="#0F2A4D">P</text>')
    parts.append(f'<text x="{px}" y="{py+1.9:.1f}" text-anchor="middle" font-family="Inter,sans-serif" font-size="2.2" font-weight="800" fill="#0F2A4D">{p_pct:.0f}%</text>')

    # Catcher (zone 2) — small octagon at the apex (home plate). The foul
    # lines converge here and the pitcher sits directly above with no gap.
    c_pct = pct_by_zone.get(2, 0)
    cx, cy = HOME
    r_c = 2.0
    pts = []
    for i in range(8):
        ang = math.radians(22.5 + i * 45)
        pts.append(f'{cx + r_c*math.cos(ang):.1f},{cy + r_c*math.sin(ang):.1f}')
    parts.append(f'<polygon points="{" ".join(pts)}" fill="#F8E8E2" stroke="#0F2A4D" stroke-width="0.4"/>')
    parts.append(f'<text x="{cx}" y="{cy-0.1:.1f}" text-anchor="middle" font-family="Inter,sans-serif" font-size="1.2" font-weight="700" fill="#0F2A4D">C</text>')
    parts.append(f'<text x="{cx}" y="{cy+1.5:.1f}" text-anchor="middle" font-family="Inter,sans-serif" font-size="1.5" font-weight="800" fill="#0F2A4D">{c_pct:.0f}%</text>')

    # Outer field-outline arc — connects the L FOUL outer corner to the
    # R FOUL outer corner with the outfield arc in between.
    fxL, fyL = polar(-90, R_OUTER)
    fxR, fyR = polar( 90, R_OUTER)
    parts.append(f'<path d="M {fxL:.2f},{fyL:.2f} A {R_OUTER},{R_OUTER} 0 0 1 {fxR:.2f},{fyR:.2f}" fill="none" stroke="#0F2A4D" stroke-width="0.7"/>')
    # Bottom edge of the field (along the horizontal at HOME y) connects
    # the foul outer corners back through HOME — forming the diamond base.
    parts.append(f'<line x1="{fxL:.2f}" y1="{fyL:.2f}" x2="{HOME[0]}" y2="{HOME[1]}" stroke="#0F2A4D" stroke-width="0.5"/>')
    parts.append(f'<line x1="{fxR:.2f}" y1="{fyR:.2f}" x2="{HOME[0]}" y2="{HOME[1]}" stroke="#0F2A4D" stroke-width="0.5"/>')

    parts.append('</svg>')
    st.markdown(''.join(parts), unsafe_allow_html=True)

with col_summary:
    st.markdown('### Field-side splits')
    side_df = pd.DataFrame([
        ['Left side',  '5/6/7/LCF', buckets['left'],   f"{buckets['left_pct']}%"],
        ['Middle',     'P/2B/CF',   buckets['middle'], f"{buckets['middle_pct']}%"],
        ['Right side', '1B/9/RCF',  buckets['right'],  f"{buckets['right_pct']}%"],
        ['Other',      'C / foul',  buckets['other'],  f"{buckets['other_pct']}%"],
    ], columns=['Side', 'Zones', 'Count', 'Share'])
    st.dataframe(side_df, hide_index=True, use_container_width=True)

    st.markdown('### Top zones')
    top = spray.sort_values('total', ascending=False).head(5)[
        ['zone_name', 'total', 'pct']
    ].rename(columns={'zone_name': 'Zone', 'total': 'Count', 'pct': '%'})
    st.dataframe(top, hide_index=True, use_container_width=True)

# ── Result-by-zone matrix ────────────────────────────────────────────────────
st.markdown('### Hit results by zone')
matrix_cols = [c for c in ['1B', '2B', '3B', 'HR', 'Out', 'FC/Err', 'Other'] if c in spray.columns]
if matrix_cols:
    display = spray[['zone_name', 'total'] + matrix_cols].rename(columns={'zone_name': 'Zone', 'total': 'Total'})
    st.dataframe(display, hide_index=True, use_container_width=True)

# ── Bar chart of zone distribution ──────────────────────────────────────────
st.markdown('### Distribution by zone')
chart_df = spray[['zone_name', 'total']].rename(columns={'zone_name': 'Zone', 'total': 'Count'})
st.bar_chart(chart_df, x='Zone', y='Count', height=240)

# ── Footer note ─────────────────────────────────────────────────────────────
st.caption(
    'Zone codes: 1=P, 2=C, 3=1B, 4=2B, 5=3B, 6=SS, 7=LF, 8=CF, 9=RF, '
    '10=LCF gap, 11=RCF gap, 12-14=foul/deep zones. '
    'Without batter handedness in the data we can\'t classify Pull/Oppo; '
    'Left/Middle/Right is the unbiased field-side split.'
)
