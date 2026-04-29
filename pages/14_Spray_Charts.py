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

    teams = list_teams(sport, division)
    if not teams:
        st.error(f'No PBP data found for {sport} {division}. '
                 f'Expected pbp_data/play_by_play/{sport}_play_by_play_{division}.csv')
        st.stop()

    selected_team = st.selectbox('Team', ['(all teams)'] + teams)
    team_filter = None if selected_team == '(all teams)' else selected_team

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
    # Build SVG: baseball field with a colored circle at each zone whose
    # radius + opacity scale with the zone's % of total balls in play.
    max_pct = spray['pct'].max() if not spray.empty else 1.0
    svg_parts = ['''
    <svg viewBox="0 0 100 120" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;background:#0a3d0a;border-radius:8px;">
      <!-- Outfield arc -->
      <path d="M 5,55 Q 50,-5 95,55" fill="#0e5a0e" stroke="#1a8a1a" stroke-width="0.4"/>
      <!-- Infield diamond -->
      <polygon points="50,108 76,82 50,56 24,82" fill="#8B6F3E" stroke="#aa8c4f" stroke-width="0.5"/>
      <!-- Pitcher's mound circle -->
      <circle cx="50" cy="80" r="3" fill="#aa8c4f"/>
      <!-- Bases -->
      <rect x="48" y="106" width="4" height="4" fill="#fff"/>
      <rect x="74" y="80" width="4" height="4" fill="#fff" transform="rotate(45 76 82)"/>
      <rect x="48" y="54" width="4" height="4" fill="#fff" transform="rotate(45 50 56)"/>
      <rect x="22" y="80" width="4" height="4" fill="#fff" transform="rotate(45 24 82)"/>
      <!-- Foul lines -->
      <line x1="50" y1="110" x2="5" y2="55" stroke="#fff" stroke-width="0.4"/>
      <line x1="50" y1="110" x2="95" y2="55" stroke="#fff" stroke-width="0.4"/>
    ''']
    for _, row in spray.iterrows():
        z = int(row['hitLocation'])
        if z not in ZONE_COORDS:
            continue
        x, y = ZONE_COORDS[z]
        pct = row['pct']
        # Bubble radius scales 2 (min) → 12 (max) by pct ratio
        rad = 2 + (pct / max_pct) * 10 if max_pct > 0 else 2
        opacity = 0.4 + 0.55 * (pct / max_pct) if max_pct > 0 else 0.4
        svg_parts.append(
            f'<circle cx="{x}" cy="{y}" r="{rad:.2f}" fill="#FFD93D" opacity="{opacity:.2f}" stroke="#fff" stroke-width="0.3"/>'
        )
        # Label inside the bubble
        svg_parts.append(
            f'<text x="{x}" y="{y+1}" font-family="Inter,sans-serif" font-size="3" fill="#000" '
            f'text-anchor="middle" font-weight="700">{row["zone_name"]}</text>'
        )
        svg_parts.append(
            f'<text x="{x}" y="{y+5}" font-family="Inter,sans-serif" font-size="2.5" fill="#222" '
            f'text-anchor="middle">{pct:.1f}%</text>'
        )
    svg_parts.append('</svg>')
    st.markdown(''.join(svg_parts), unsafe_allow_html=True)

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
