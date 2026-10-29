"""
Park Factor Lab — interactive park-factor sandbox (baseball 2025 + 2026).

Three reference points per park, side by side:
  - Pro     : classic runs-based park factor (home RPG / road RPG), like MLB.
  - Ours     : our expected-vs-actual model at its neutral baseline
               (hits, both sides, all players) — "what we have in theory".
  - Adjusted : Ours recomputed with the model objects you set (stat, side, who,
               regression).

All three respond to the CONTEXT filters ("thoughts") you add/remove in the
sidebar — conference vs non-conf, midweek vs weekend, month, opponent RPI tier —
so you can slice the whole comparison by game context.

Years: 2026 (venue-clean, neutral sites excluded) and 2025 (no venue that season,
so neutral/tournament games can't be filtered out — flagged below). Parks keyed
by venue. Physical objects (dimensions, altitude) arrive in a later pass.

Built 2026-10-29. Data: 2025-26 D-I/II/III play-by-play.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
CELLS_FILE = _APP_DIR / 'data' / 'park_factor_cells.csv'
PRO_FILE = _APP_DIR / 'data' / 'park_factor_progames.csv'
TEAMS_FILE = _APP_DIR / 'data' / 'teams.csv'
CONF_FILE = _APP_DIR / 'data' / 'conferences.csv'

DIVISION_LABELS = {'D1': 'D-I', 'D2': 'D-II', 'D3': 'D-III'}
STAT_LABELS = {'Hits': 'h', 'Extra-base hits': 'xbh', 'Home runs': 'hr'}
SIDE_CELLS = {'Both': ['bat', 'pit'], 'Hitters': ['bat'], 'Pitchers': ['pit']}
LOC_CELLS = {'All players': ['home', 'opp'], 'Visitors only': ['opp'], 'Home team only': ['home']}
YEAR_OPTS = {'2026': [2026], '2025': [2025], 'Both (2025+26)': [2025, 2026]}
MONTHS = {2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 1: 'Jan'}
TIER_ORDER = ['Top 50', 'Mid (51-150)', 'Low (151+)', 'Unranked']

st.set_page_config(page_title='Park Factor Lab', layout='wide')


@st.cache_data
def load():
    return pd.read_csv(CELLS_FILE), pd.read_csv(PRO_FILE)


cells, progames = load()


@st.cache_data
def conf_map():
    tm = pd.read_csv(TEAMS_FILE)
    tm = tm[tm['sport'].astype(str).str.lower().str.contains('base')]
    cf = pd.read_csv(CONF_FILE)
    id2name = dict(zip(cf['id'], cf['name']))
    id2ab = dict(zip(cf['id'], cf['abbreviation']))
    nm = {str(n).strip(): id2name.get(c) for n, c in zip(tm['name'], tm['conference_id'])}
    ab = {str(n).strip(): id2ab.get(c) for n, c in zip(tm['name'], tm['conference_id'])}
    return nm, ab


TEAM2CONF, TEAM2ABBR = conf_map()

# ── sidebar ─────────────────────────────────────────────────────────────────────
st.sidebar.header('Park Factor Lab')
year_label = st.sidebar.radio('Season', list(YEAR_OPTS), index=2, horizontal=True)
years = YEAR_OPTS[year_label]
div = st.sidebar.selectbox('Division', list(DIVISION_LABELS), index=0,
                           format_func=lambda d: DIVISION_LABELS[d])
conf_present = sorted({TEAM2CONF.get(t) for t in cells[cells['division'] == div]['team'].dropna().unique()
                       if TEAM2CONF.get(t) and pd.notna(TEAM2CONF.get(t))})
conf_pick = st.sidebar.selectbox('Conference (filter parks)', ['All conferences'] + conf_present, index=0)
min_g = st.sidebar.slider('Min home games (reliability)', 0, 25, 6)

st.sidebar.markdown('---')
st.sidebar.caption('CONTEXT — add / remove thoughts')
conf_sel = st.sidebar.multiselect('Game type', ['Conf', 'Non-conf'], default=['Conf', 'Non-conf'])
day_sel = st.sidebar.multiselect('Day', ['Midweek', 'Weekend'], default=['Midweek', 'Weekend'])
months_present = sorted([m for m in cells['month'].dropna().unique() if m in MONTHS], key=lambda m: list(MONTHS).index(m))
month_sel_lbl = st.sidebar.multiselect('Month', [MONTHS[m] for m in months_present],
                                        default=[MONTHS[m] for m in months_present])
month_sel = [m for m in MONTHS if MONTHS[m] in month_sel_lbl]
tier_sel = st.sidebar.multiselect('Opponent tier (RPI)', TIER_ORDER, default=TIER_ORDER)

st.sidebar.markdown('---')
st.sidebar.caption('OUR MODEL — objects')
stat_label = st.sidebar.radio('Stat', list(STAT_LABELS), index=1,
                              help='A park shows its character in power (XBH/HR), not singles.')
side_label = st.sidebar.radio('Side', list(SIDE_CELLS), index=0)
loc_label = st.sidebar.radio('Who', list(LOC_CELLS), index=0,
                             help='"Visitors only" is the cleanest neutral read.')
regress = st.sidebar.slider('Regression to 1.0', 0, 100, 0,
                            help='Shrink each park toward neutral.') / 100.0

stat = STAT_LABELS[stat_label]

# ── apply context filters ───────────────────────────────────────────────────────
def ctx(df):
    return df[df['year'].isin(years) & (df['division'] == div)
              & df['conf_game'].isin(conf_sel) & df['midweek'].isin(day_sel)
              & df['month'].isin(month_sel) & df['opp_tier'].isin(tier_sel)]


fc, fp = ctx(cells), ctx(progames)


def ours_ratio(df, stat, sides, locs):
    f = df[df['side'].isin(sides) & df['loc'].isin(locs)]
    g = f.groupby('park').agg(a=(f'act_{stat}', 'sum'), e=(f'exp_{stat}', 'sum'))
    g['pf'] = np.where(g['e'] > 0, g['a'] / g['e'], np.nan)
    return g['pf']


def pro_pf(df):
    def rpg(sub):
        s = sub.groupby('park').agg(rr=('rf', 'sum'), ra=('ra', 'sum'), n=('n', 'sum'))
        return (s['rr'] + s['ra']) / s['n'], s['n']
    hr, hn = rpg(df[df['homeroad'] == 'home'])
    rr, rn = rpg(df[df['homeroad'] == 'road'])
    out = pd.DataFrame({'home_rpg': hr, 'road_rpg': rr, 'home_g': hn, 'road_g': rn})
    out['Pro'] = np.where((out['home_g'] >= 5) & (out['road_g'] >= 5) & (out['road_rpg'] > 0),
                          out['home_rpg'] / out['road_rpg'], np.nan)
    return out


ours = ours_ratio(fc, stat, ['bat', 'pit'], ['home', 'opp']).rename('Ours')
adj_raw = ours_ratio(fc, stat, SIDE_CELLS[side_label], LOC_CELLS[loc_label])
adjusted = (1.0 + (adj_raw - 1.0) * (1.0 - regress)).rename('Adjusted')
prod = pro_pf(fp)

names = fc.groupby('park')['team'].first()
d = pd.concat([prod[['Pro', 'home_g']], ours, adjusted, names], axis=1)
d = d[d['home_g'].fillna(0) >= min_g].dropna(subset=['Ours', 'Adjusted'])
d['Conf'] = d['team'].map(lambda t: TEAM2ABBR.get(t) or '')
if conf_pick != 'All conferences':
    d = d[d['team'].map(TEAM2CONF) == conf_pick]
d['Δ vs Ours'] = d['Adjusted'] - d['Ours']
d['Δ vs Pro'] = d['Adjusted'] - d['Pro']

# ── header ──────────────────────────────────────────────────────────────────────
st.title('🏟️ Park Factor Lab')
st.markdown('Compare **what the pros measure** (runs-based), **what our model sees** '
            '(expected vs actual), and **your adjusted version** — sliced by whatever game '
            'context you choose. 1.00 plays neutral; 1.10 ≈ 10% more than expected.')

thoughts = []
if set(conf_sel) != {'Conf', 'Non-conf'}: thoughts.append('+'.join(conf_sel))
if set(day_sel) != {'Midweek', 'Weekend'}: thoughts.append('+'.join(day_sel))
if len(month_sel) != len(months_present): thoughts.append('/'.join(month_sel_lbl))
if set(tier_sel) != set(TIER_ORDER): thoughts.append('vs ' + '/'.join(tier_sel))
ctx_txt = ' · '.join(thoughts) if thoughts else 'all games'
div_txt = DIVISION_LABELS[div] + ('' if conf_pick == 'All conferences' else f' · {conf_pick}')
st.caption(f'{year_label} · {div_txt}  |  context: **{ctx_txt}**  |  '
           f'Adjusted = {stat_label} · {side_label} · {loc_label}'
           f'{f" · {int(regress*100)}% regressed" if regress else ""}  |  {len(d)} parks')

if 2025 in years:
    st.info('Heads up: 2025 had no venue data, so neutral-site/tournament games can\'t be '
            'excluded that season (2026 is clean).', icon='ℹ️')

if d.empty:
    st.warning('No parks meet the current filters — try loosening the context or lowering min games.')
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric('Parks', len(d))
c2.metric('Pro ↔ Adjusted corr', f"{d['Pro'].corr(d['Adjusted']):.2f}" if d['Pro'].notna().sum() > 2 else '—')
c3.metric('Most hitter-friendly', f"{d['Adjusted'].max():.2f}")
c4.metric('Most pitcher-friendly', f"{d['Adjusted'].min():.2f}")

# ── leaderboard ─────────────────────────────────────────────────────────────────
show = (d.reset_index().rename(columns={'park': 'Park', 'team': 'Team', 'home_g': 'Home G'})
        [['Park', 'Team', 'Conf', 'Pro', 'Ours', 'Adjusted', 'Δ vs Ours', 'Δ vs Pro', 'Home G']]
        .sort_values('Adjusted', ascending=False))
show['Home G'] = show['Home G'].astype('Int64')


def _dc(v):
    if pd.isna(v) or v == 0:
        return ''
    return 'color: #16a34a' if v > 0 else 'color: #dc2626'


styled = (show.style
          .format({c: '{:.3f}' for c in ['Pro', 'Ours', 'Adjusted', 'Δ vs Ours', 'Δ vs Pro']}, na_rep='—')
          .map(_dc, subset=['Δ vs Ours', 'Δ vs Pro'])
          .hide(axis='index'))
event = st.dataframe(styled, use_container_width=True, height=560,
                     on_select='rerun', selection_mode='single-row', key='pf_lb')

# ── per-park both-sides diagnostic ──────────────────────────────────────────────
sel = []
try:
    sel = list(event.selection.rows)
except Exception:
    try:
        sel = list(event['selection']['rows'])
    except Exception:
        sel = []
sel_park = show.iloc[sel[0]]['Park'] if sel else show.iloc[0]['Park']

st.subheader('🔬 Park diagnostic — both sides')
st.caption(f'**{sel_park}** · {stat_label} · {ctx_txt}  —  click any row above to inspect a different park. '
           'A real park effect moves hitters AND pitchers the same way; a one-sided move is usually '
           'competition or roster, not the fences.')


def _pc(side, loc):
    s = fc[(fc['park'] == sel_park) & (fc['side'] == side) & (fc['loc'] == loc)]
    a, e = s[f'act_{stat}'].sum(), s[f'exp_{stat}'].sum()
    return (a / e, e) if e > 0 else (np.nan, 0.0)


cellmap = [('Home-team hitters', 'bat', 'home'), ('Visiting hitters', 'bat', 'opp'),
           ('Home pitchers (allowed)', 'pit', 'home'), ('Visiting pitchers (allowed)', 'pit', 'opp')]
cols = st.columns(4)
ratios = []
for col, (lab, side, loc) in zip(cols, cellmap):
    r, e = _pc(side, loc)
    ratios.append(r)
    lean = '' if pd.isna(r) else ('↑ favors hitters' if r > 1.03 else '↓ favors pitchers' if r < 0.97 else '≈ neutral')
    col.metric(lab, f'{r:.3f}' if pd.notna(r) else '—', delta=lean, delta_color='off')
    col.caption(f'expected {e:.0f}')

valid = [r for r in ratios if pd.notna(r)]
up = sum(1 for r in valid if r > 1.03)
down = sum(1 for r in valid if r < 0.97)
if len(valid) >= 3 and up >= 3:
    st.success('Consistent **hitter-friendly** signal — both hitters and pitchers move the same way. '
               'This looks like a real park effect.')
elif len(valid) >= 3 and down >= 3:
    st.success('Consistent **pitcher-friendly** signal — both sides move the same way. '
               'This looks like a real park effect.')
elif len(valid) >= 2 and up >= 1 and down >= 1:
    st.warning('Mixed signal — the sides disagree, so this is more likely **competition or roster** than '
               'the park (e.g., a strong home lineup beating weaker arms). The **visiting-hitter** cell is '
               'the cleanest park read — it has no home-team bias.')
else:
    st.info('Roughly neutral across the board.')

with st.expander('Pro vs Adjusted — where the two philosophies disagree'):
    dd = d.dropna(subset=['Pro', 'Adjusted'])
    if len(dd) > 2:
        import plotly.graph_objects as go
        lo = float(min(dd['Pro'].min(), dd['Adjusted'].min())) - 0.03
        hi = float(max(dd['Pro'].max(), dd['Adjusted'].max())) + 0.03
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines',
                                 line=dict(dash='dash', color='#9ca3af'), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=dd['Pro'], y=dd['Adjusted'], mode='markers',
                                 marker=dict(size=7, color=dd['Δ vs Pro'], colorscale='RdYlGn', cmid=0,
                                             line=dict(width=0.5, color='#555')),
                                 text=dd.index, showlegend=False,
                                 hovertemplate='%{text}<br>Pro %{x:.3f}<br>Adjusted %{y:.3f}<extra></extra>'))
        fig.update_layout(xaxis_title='Pro (runs-based)', yaxis_title='Adjusted (ours)',
                          height=560, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Not enough parks with both Pro and Adjusted under the current filters.')

with st.expander('How these are calculated'):
    st.markdown(
        '- **Ours / Adjusted** — the honest park read. Each player\'s expected is their rate at '
        '**every park except this one** (leave-park-out). That removes the circularity where a '
        'bandbox inflates its own home players\' season baselines and cancels itself out. At a park '
        'we sum actual vs that expected for the chosen stat/side/who. Ours is the neutral baseline '
        '(both sides · all players); Adjusted applies your side/who/regression.\n'
        '- **Stat matters:** a park shows its character in **power (XBH/HR)**, not singles — total '
        'hits barely move with the fences, so XBH is the default.\n'
        '- **Pro** — the classic runs method, (home runs/game) ÷ (road runs/game) per team. Kept as '
        'a reference, but it is **noisy for college**: lopsided home/road schedules and the home '
        'team\'s own pitching distort it (it can call a real bandbox pitcher-friendly), so trust '
        'Ours over Pro.\n'
        '- **Competition control:** the **visitors-only** and **opponent-tier** filters are how you '
        'check that a park reads the same against good competition and without the home roster.\n'
        '- **Parks** keyed by play-by-play venue; 2026 excludes neutral sites via the venue match, '
        '2025 has no venue so all home games count. **2025 + 2026 pooled** by default (our full PBP).'
    )
