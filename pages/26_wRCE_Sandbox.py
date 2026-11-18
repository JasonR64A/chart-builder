"""
wRCE Sandbox (hitting) — chart-builder page 26.

Interactive methodology bench: take our complete hitting stats and compute a
production rating with the linear-weight refinements that are "under consideration"
for the wRCE engine, then watch the board re-rank live and compare to the CURRENT
published wRCE (Bill James RC/AB engine).

Levers (the three flagged refinements + park):
  - run values : our RE-matrix linear weights (adjustable per event)
  - wOBA scale : college-specific (ours) vs held at 1.0 (current engine)
  - baseline   : division vs conference-relative (100 = that group's average)
  - park       : park-adjust on/off

Season-level (2026). Counts come from hitting.csv (complete); weights/constants from
sandbox_const.json. This bench covers the hitting refinements, not the full per-game
opponent-differential engine.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='wRCE Sandbox', layout='wide')
_APP_DIR = Path(__file__).resolve().parent.parent
HFILE = _APP_DIR / 'data' / 'sandbox_hitters.csv'
CFILE = _APP_DIR / 'data' / 'sandbox_const.json'
EVT = ['BB', 'HBP', '1B', '2B', '3B', 'HR']
DIV_ORDER = ['D-I', 'D-II', 'D-III']


@st.cache_data
def load():
    h = pd.read_csv(HFILE)
    for c in ['PF', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'IBB', 'HBP', 'SF', 'R', 'PA', 'cur_wRCE', 'cur_RC_AB']:
        h[c] = pd.to_numeric(h[c], errors='coerce').fillna(0.0)
    for c in ['name', 'team', 'div', 'conf']:
        h[c] = h[c].fillna('').astype(str)
    const = json.load(open(CFILE))
    return h, const


H, CONST = load()
DIVW = {d: CONST['div'][d]['w'] for d in CONST['div']}
DIV_SCALE = {d: CONST['div'][d]['scale'] for d in CONST['div']}
DIV_LGRPA = {d: CONST['div'][d]['lgRPA'] for d in CONST['div']}
CONF_LGRPA = {c: CONST['conf'][c]['lgRPA'] for c in CONST['conf']}
CONF_SET = set(CONST['conf'])


# --------------------------------------------------------------------------- engine
def compute(min_ab, scale_mode, baseline, park, wmult):
    df = H[H['AB'] >= min_ab].copy()
    df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']
    df['uBB'] = df['BB'] - df['IBB']
    df['PAden'] = df['AB'] + df['uBB'] + df['HBP'] + df['SF']
    df = df[df['PAden'] > 0]
    # adjusted linear weights per event (division weight x user multiplier)
    num = np.zeros(len(df))
    for e in EVT:
        w = df['div'].map(lambda d: DIVW.get(d, DIVW['D-I'])[e]).to_numpy() * wmult[e]
        cnt = df[e].to_numpy() if e != 'BB' else df['uBB'].to_numpy()
        num += w * cnt
    df['wOBA'] = num / df['PAden'].to_numpy()

    # baseline group: conference if known & chosen, else division
    if baseline == 'Conference':
        df['grp'] = np.where(df['conf'].isin(CONF_SET), df['conf'], df['div'])
        lgrpa = df['grp'].map(lambda g: CONF_LGRPA.get(g, DIV_LGRPA.get(g, 0.165)))
    else:
        df['grp'] = df['div']
        lgrpa = df['grp'].map(lambda g: DIV_LGRPA.get(g, 0.165))
    # league-average wOBA per group, recomputed under the current weights (PA-weighted)
    agg = df.assign(_n=df['wOBA'] * df['PAden']).groupby('grp').agg(_n=('_n', 'sum'), _d=('PAden', 'sum'))
    agg['lg'] = agg['_n'] / agg['_d']
    df['lgwOBA'] = df['grp'].map(agg['lg'])
    df['lgRPA'] = lgrpa.to_numpy()
    df['scale'] = df['div'].map(DIV_SCALE).to_numpy() if scale_mode == 'College (ours)' else 1.0
    df['pf'] = df['PF'].to_numpy() if park else 1.0

    raapa = (df['wOBA'] - df['lgwOBA']) / df['scale']
    lg = df['lgRPA']
    df['wRCplus'] = ((raapa + lg) + (lg - df['pf'] * lg)) / lg * 100
    df['wRAA'] = raapa * df['PAden']

    df = df.sort_values('wRCplus', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    # fair comparison = current Bill James RC/AB (a per-AB rate, like our wRC+); isolates the run-value refinement
    df['cur_rank'] = df['cur_RC_AB'].rank(ascending=False, method='min').astype(int)
    df['move'] = df['cur_rank'] - df['rank']   # + = our method ranks them higher than RC/AB does
    return df


# --------------------------------------------------------------------------- sidebar
st.sidebar.header('Methodology levers')
scale_mode = st.sidebar.radio('wOBA scale', ['College (ours)', 'Held at 1.0 (current)'],
                              help='Our college-specific scale vs the engine currently holding it at 1.0.')
baseline = st.sidebar.radio('Baseline (100 = average of…)', ['Division', 'Conference'],
                            help='Index wRC+ to the division or re-baseline to the player’s conference.')
park = st.sidebar.checkbox('Park-adjust (use our park factors)', value=True)
min_ab = st.sidebar.slider('Qualifying floor — minimum AB', 10, 300, 50, 5)
st.sidebar.markdown('---')
with st.sidebar.expander('Advanced — per-event run values (× multiplier)'):
    wmult = {e: st.slider(e, 0.0, 2.0, 1.0, 0.05, key=f'wm_{e}') for e in EVT}
if st.sidebar.button('Reset multipliers'):
    for e in EVT:
        st.session_state[f'wm_{e}'] = 1.0
    st.rerun()

BOARD = compute(min_ab, scale_mode, baseline, park, wmult)

# --------------------------------------------------------------------------- main
st.title('⚾ wRCE Sandbox — hitting')
st.caption('Our linear-weight production rating vs the current published wRCE. 2026 · counts from hitting.csv (complete). '
           '100 = the baseline group average.')

c1, c2, c3, c4 = st.columns(4)
c1.metric('Players', f'{len(BOARD):,}')
sp = BOARD[['wRCplus', 'cur_RC_AB']].corr(method='spearman').iloc[0, 1]
c2.metric('Rank corr vs current RC/AB', f'{sp:.3f}')
# movers restricted to regulars (PA>=150, current RC/AB present) so they're meaningful, not bottom-of-list noise
movers = BOARD[(BOARD['cur_RC_AB'] > 0) & (BOARD['PA'] >= 150)].copy()
c3.metric('Regulars (PA ≥ 150)', f'{len(movers):,}')
c4.metric('Median rank shift (regulars)', f"{int(movers['move'].abs().median())}" if len(movers) else '—')

fcol1, fcol2, fcol3 = st.columns([2, 2, 3])
dsel = fcol1.multiselect('Division', DIV_ORDER, default=DIV_ORDER)
confs = ['(all)'] + sorted(c for c in BOARD['conf'].unique() if c)
csel = fcol2.selectbox('Conference', confs)
search = fcol3.text_input('Search player / team')

view = BOARD[BOARD['div'].isin(dsel)].copy()
if csel != '(all)':
    view = view[view['conf'] == csel]
if search:
    s = search.lower()
    view = view[view['name'].str.lower().str.contains(s) | view['team'].str.lower().str.contains(s)]

show = view[['rank', 'name', 'team', 'div', 'conf', 'PA', 'HR', 'wOBA', 'wRCplus', 'cur_RC_AB', 'cur_rank', 'move']].copy()
show.columns = ['Rank', 'Player', 'Team', 'Div', 'Conference', 'PA', 'HR', 'wOBA', 'wRC+', 'Cur RC/AB', 'RC/AB rank', 'Δ rank']
st.dataframe(
    show.style.format({'wOBA': '{:.3f}', 'wRC+': '{:.0f}', 'Cur RC/AB': '{:.3f}'})
              .background_gradient(subset=['Δ rank'], cmap='RdYlGn', vmin=-80, vmax=80),
    use_container_width=True, height=560, hide_index=True)

with st.expander('Biggest movers vs the current wRCE engine'):
    m = movers.copy()
    up = m.nlargest(12, 'move')[['name', 'team', 'div', 'wRCplus', 'cur_rank', 'rank', 'move']]
    dn = m.nsmallest(12, 'move')[['name', 'team', 'div', 'wRCplus', 'cur_rank', 'rank', 'move']]
    cc1, cc2 = st.columns(2)
    cc1.caption('Our method ranks higher'); cc1.dataframe(up, use_container_width=True, hide_index=True)
    cc2.caption('Our method ranks lower'); cc2.dataframe(dn, use_container_width=True, hide_index=True)

st.download_button('Download current view (CSV)', view.to_csv(index=False).encode(),
                   file_name='wrce_sandbox_view.csv', mime='text/csv')
