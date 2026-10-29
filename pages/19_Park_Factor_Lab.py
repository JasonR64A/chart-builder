"""
Park Factor Lab — interactive park-factor sandbox (baseball 2026).

Three reference points per park, side by side:
  - Pro     : classic runs-based park factor (home RPG / road RPG), like MLB.
  - Ours     : our expected-vs-actual model at its neutral baseline
               (hits, both sides, all players) — "what we have in theory".
  - Adjusted : Ours recomputed with the objects you toggle in the sidebar
               (stat, hitter/pitcher side, home/visitor split, regression).

Baseline is always shown next to Adjusted so you can see where our decisions
moved the number, and how both compare to the pros.

Parks are keyed by play-by-play `venue`; neutral-site and garbled-venue games
are excluded (a game counts toward a park only when its venue matches that
team's usual home park). Physical objects (field dimensions, altitude) arrive
in v2.

Built 2026-10-29. Data: 2026 D-I/II/III play-by-play.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = _APP_DIR / 'data' / 'park_factor_lab_baseball_2026.csv'

DIVISION_LABELS = {'D1': 'D-I', 'D2': 'D-II', 'D3': 'D-III'}
STAT_LABELS = {'Hits': 'h', 'Extra-base hits': 'xbh', 'Home runs': 'hr'}
SIDE_CELLS = {'Both': ['bat', 'pit'], 'Hitters': ['bat'], 'Pitchers': ['pit']}
LOC_CELLS = {'All players': ['home', 'opp'], 'Visitors only': ['opp'], 'Home team only': ['home']}

st.set_page_config(page_title='Park Factor Lab', layout='wide')


@st.cache_data
def load():
    return pd.read_csv(DATA_FILE)


def _cell(row, stat, side, loc, which):
    v = row.get(f'{stat}_{side}_{loc}_{which}', 0.0)
    return 0.0 if pd.isna(v) else float(v)


def ours_pf(row, stat, sides, locs, regress):
    """actual / expected over the selected cells, shrunk toward 1.0 by `regress`."""
    a = sum(_cell(row, stat, sd, lc, 'act') for sd in sides for lc in locs)
    e = sum(_cell(row, stat, sd, lc, 'exp') for sd in sides for lc in locs)
    if e <= 0:
        return np.nan
    raw = a / e
    return 1.0 + (raw - 1.0) * (1.0 - regress)


# ── sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header('Park Factor Lab')
div = st.sidebar.selectbox('Division', list(DIVISION_LABELS), index=0,
                           format_func=lambda d: DIVISION_LABELS[d])
min_g = st.sidebar.slider('Min home games (reliability)', 0, 30, 10)

st.sidebar.markdown('---')
st.sidebar.caption('OUR MODEL — adjust the objects')
stat_label = st.sidebar.radio('Stat', list(STAT_LABELS), index=0)
side_label = st.sidebar.radio('Side', list(SIDE_CELLS), index=0,
                              help='Which players to read the park from.')
loc_label = st.sidebar.radio('Who', list(LOC_CELLS), index=0,
                             help='"Visitors only" is the cleanest neutral read — no home-team familiarity.')
regress = st.sidebar.slider('Regression to 1.0', 0, 100, 0,
                            help='Shrink each park toward neutral. Higher = more conservative.') / 100.0

st.sidebar.markdown('---')
st.sidebar.caption('v2 (coming): physical objects — field dimensions, wall height, altitude.')

stat = STAT_LABELS[stat_label]
sides = SIDE_CELLS[side_label]
locs = LOC_CELLS[loc_label]

# ── compute ────────────────────────────────────────────────────────────────────
df = load()
d = df[df['division'] == div].copy()
d = d[d['home_games'].fillna(0) >= min_g]

d['Ours'] = d.apply(lambda r: ours_pf(r, 'h', ['bat', 'pit'], ['home', 'opp'], 0.0), axis=1)
d['Adjusted'] = d.apply(lambda r: ours_pf(r, stat, sides, locs, regress), axis=1)
d = d.rename(columns={'pro_pf': 'Pro'})
d = d.dropna(subset=['Pro', 'Ours', 'Adjusted'])
d['Δ vs Ours'] = d['Adjusted'] - d['Ours']
d['Δ vs Pro'] = d['Adjusted'] - d['Pro']

# ── header ──────────────────────────────────────────────────────────────────────
st.title('🏟️ Park Factor Lab')
st.markdown(
    'Compare **what the pros measure** (runs-based), **what our model sees** '
    '(expected vs actual), and **your adjusted version** — then read the gaps. '
    '1.00 plays neutral; 1.10 means ~10% more than expected.'
)
setting = f'{stat_label} · {side_label} · {loc_label}' + (f' · {int(regress*100)}% regressed' if regress else '')
st.caption(f'Adjusted = {setting}  |  {DIVISION_LABELS[div]} baseball 2026  |  {len(d)} parks')

if d.empty:
    st.warning('No parks meet the current filters.')
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric('Parks', len(d))
c2.metric('Pro ↔ Adjusted corr', f"{d['Pro'].corr(d['Adjusted']):.2f}")
c3.metric('Most hitter-friendly (Adj)', f"{d['Adjusted'].max():.2f}")
c4.metric('Most pitcher-friendly (Adj)', f"{d['Adjusted'].min():.2f}")

# ── leaderboard ─────────────────────────────────────────────────────────────────
show = (d[['park_name', 'team', 'Pro', 'Ours', 'Adjusted', 'Δ vs Ours', 'Δ vs Pro', 'home_games']]
        .sort_values('Adjusted', ascending=False)
        .rename(columns={'park_name': 'Park', 'team': 'Team', 'home_games': 'Home G'}))
show['Home G'] = show['Home G'].astype('Int64')


def _delta_color(v):
    if pd.isna(v) or v == 0:
        return ''
    return 'color: #16a34a' if v > 0 else 'color: #dc2626'


styled = (show.style
          .format({c: '{:.3f}' for c in ['Pro', 'Ours', 'Adjusted', 'Δ vs Ours', 'Δ vs Pro']})
          .map(_delta_color, subset=['Δ vs Ours', 'Δ vs Pro'])
          .hide(axis='index'))
st.dataframe(styled, use_container_width=True, height=560)

# ── Pro vs Adjusted scatter (the "gleam from the differences" view) ─────────────
with st.expander('Pro vs Adjusted — where the two philosophies disagree', expanded=False):
    try:
        import plotly.graph_objects as go
        lo = float(min(d['Pro'].min(), d['Adjusted'].min())) - 0.03
        hi = float(max(d['Pro'].max(), d['Adjusted'].max())) + 0.03
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines',
                                 line=dict(dash='dash', color='#9ca3af'), showlegend=False,
                                 hoverinfo='skip'))
        fig.add_trace(go.Scatter(
            x=d['Pro'], y=d['Adjusted'], mode='markers',
            marker=dict(size=7, color=d['Δ vs Pro'], colorscale='RdYlGn', cmid=0,
                        line=dict(width=0.5, color='#555')),
            text=d['park_name'], showlegend=False,
            hovertemplate='%{text}<br>Pro %{x:.3f}<br>Adjusted %{y:.3f}<extra></extra>'))
        fig.update_layout(xaxis_title='Pro (runs-based)', yaxis_title='Adjusted (ours)',
                          height=560, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption('On the dashed line = the two agree. Above it = our model reads more '
                   'hitter-friendly than the pros; below = more pitcher-friendly.')
    except Exception as e:
        st.info(f'Scatter unavailable: {e}')

with st.expander('How these are calculated'):
    st.markdown(
        '- **Pro** — (home runs/game) ÷ (road runs/game) for each team, neutral-site games removed. '
        'Needs 5+ home and 5+ road games.\n'
        '- **Ours / Adjusted** — each player gets a season rate (per PA for hitters, per batter '
        'faced for pitchers); at a park we sum **actual** vs **expected** for the selected stat and '
        'players, and take the ratio. Regression shrinks each park toward 1.00.\n'
        '- **Parks** are identified by play-by-play `venue`; a game only counts toward a park when '
        'its venue matches that team\'s usual home park, so neutral sites (conference tournaments, '
        'showcase games) and garbled venues are excluded.\n'
        '- **v2** will add physical objects — field dimensions, wall height, altitude.'
    )
