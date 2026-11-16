"""
Softball Portal Ranking Sandbox — chart-builder page 25.

Interactive methodology tool: tune the weights/levers behind the 64A softball portal
ranking and watch the board re-rank live. Default state reproduces the published 2026
softball board (Isa Torres #3). Fed by two precomputed CSVs from the portal engine
(`export_softball_sandbox.py`), so the page does only light math — no 31MB stat loads.

Formula (rank_players.py, softball):
  rating = stat_percentile + conf_adjust + twp + mlb_adjust + hs_adjust + prod_term + fielding
  prod_term = prod_coef * max(pct_wRCE, pct_wRAE) * division_multiplier
Stat percentile is re-weighted from each player's 3-season (2024-2026) stat percentiles,
then delta-anchored to the frozen published value so defaults match the real board exactly.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title='Softball Portal Sandbox', layout='wide')

_APP_DIR = Path(__file__).resolve().parent.parent
PLAYERS_FILE = _APP_DIR / 'data' / 'softball_sandbox_players.csv'
YEARS_FILE = _APP_DIR / 'data' / 'softball_sandbox_years.csv'

ANCHOR = 2026
TORRES = 87082
YEAR_LABELS = {0: '2024', 1: '2025', 2: '2026'}

# component -> (label, color, help)
COMPS = [
    ('stat',  'Stat %ile',  '#52B37C', 'batting / pitching rate percentile'),
    ('prod',  'Production',  '#CAD72B', 'wRCE / wRAE run efficiency'),
    ('conf',  'Conference',  '#49B6CB', 'strength of competition'),
    ('mlb',   'Pro proj.',   '#EAA347', 'SA college / pro rank'),
    ('hs',    'HS',          '#E0795C', 'SA high-school rank'),
    ('twp',   'Two-way',     '#9D8CEA', 'hits AND pitches'),
    ('field', 'Fielding',    '#8A99A0', 'fielding grade (softball = 0)'),
]
CCOLOR = {k: c for k, _, c in [(a, b, d) for a, b, d, e in COMPS]}
CLABEL = {k: lb for k, lb, _, _ in COMPS}

DEFAULTS = {
    'w_ops': 1.0, 'w_woba': 1.0, 'w_iso': 1.0,
    'w_opsa': 1.0, 'w_babip': 1.0, 'w_kbb': 1.0,
    'm_stat': 1.0, 'm_prod': 1.0, 'prod_coef': 0.50, 'm_conf': 1.0,
    'm_mlb': 1.0, 'm_hs': 1.0, 'twp_coef': 0.10, 'm_field': 1.0,
    'div_D1': 1.0, 'div_D2': 0.75, 'div_D3': 0.50,
    'yw_2024': 1.0, 'yw_2025': 1.0, 'yw_2026': 2.0,
    'min_pa': 0, 'min_bf': 0,
}


# ----------------------------------------------------------------------------- data
@st.cache_data
def load():
    pl = pd.read_csv(PLAYERS_FILE)
    yr = pd.read_csv(YEARS_FILE)
    yr['div'] = yr['div'].fillna('')
    for c in ['stored_stat', 'stored_twp', 'stat_def', 'minrank_def', 'conf_adjust',
              'mlb_adjust', 'hs_adjust', 'fielding_rank', 'wrce', 'wrae', 'prod_divadj']:
        pl[c] = pd.to_numeric(pl[c], errors='coerce').fillna(0.0)
    pl['official_rank'] = pd.to_numeric(pl['official_rank'], errors='coerce').fillna(0).astype(int)
    for c in ['name', 'institution', 'division', 'position', 'class_year', 'new_team']:
        pl[c] = pl[c].fillna('').astype(str)
    return pl, yr


PLAYERS, YEARSDF = load()


# ------------------------------------------------------------------------- engine
def compute_board(w):
    y = YEARSDF
    divmap = {'D-I': w['div_D1'], 'D-II': w['div_D2'], 'D-III': w['div_D3'], '': 1.0}
    is_h = (y['side'] == 'H').to_numpy()
    sw1 = np.where(is_h, w['w_ops'], w['w_opsa'])
    sw2 = np.where(is_h, w['w_woba'], w['w_babip'])
    sw3 = np.where(is_h, w['w_iso'], w['w_kbb'])
    dadj = y['div'].map(lambda d: divmap.get(d, 1.0)).to_numpy()
    yw = y['off'].map({0: w['yw_2024'], 1: w['yw_2025'], 2: w['yw_2026']}).to_numpy()
    floor_ok = np.where(is_h, y['pt'].to_numpy() >= w['min_pa'], y['pt'].to_numpy() >= w['min_bf'])
    contrib = (sw1 * y['s1'] + sw2 * y['s2'] + sw3 * y['s3']) * y['pt_pct'] * yw * dadj
    contrib = np.where(floor_ok, contrib, 0.0)
    tmp = pd.DataFrame({'id': y['id'].to_numpy(), 'side': y['side'].to_numpy(), 'c': contrib})
    hsum = tmp[tmp['side'] == 'H'].groupby('id')['c'].sum() / 3.0
    psum = tmp[tmp['side'] == 'P'].groupby('id')['c'].sum() / 3.0

    pl = PLAYERS.copy()
    pl['hsum'] = pl['id'].map(hsum).fillna(0.0)
    pl['psum'] = pl['id'].map(psum).fillna(0.0)
    pl['hrk'] = 0.0
    pl['prk'] = 0.0
    hm = pl['hsum'] > 0
    pm = pl['psum'] > 0
    if hm.any():
        pl.loc[hm, 'hrk'] = pl.loc[hm, 'hsum'].rank(pct=True).round(4)
    if pm.any():
        pl.loc[pm, 'prk'] = pl.loc[pm, 'psum'].rank(pct=True).round(4)
    recStat = pl[['hrk', 'prk']].max(axis=1)
    both = (pl['hrk'] > 0) & (pl['prk'] > 0)
    recMin = np.where(both, pl[['hrk', 'prk']].min(axis=1), 0.0)

    effStat = (pl['stored_stat'] + (recStat - pl['stat_def'])).clip(0, 1)
    effMin = (pl['minrank_def'] + (recMin - pl['minrank_def'])).clip(lower=0)
    twoway = (pl['has_hit'] == 1) & (pl['has_pit'] == 1)
    proddadj = pl['division'].map(lambda d: divmap.get(d, 1.0))

    pl['c_stat'] = w['m_stat'] * effStat
    pl['c_conf'] = w['m_conf'] * pl['conf_adjust']
    pl['c_mlb'] = w['m_mlb'] * pl['mlb_adjust']
    pl['c_hs'] = w['m_hs'] * pl['hs_adjust']
    pl['c_prod'] = w['m_prod'] * w['prod_coef'] * pl[['wrce', 'wrae']].max(axis=1) * proddadj
    pl['c_twp'] = np.where(twoway, w['twp_coef'] * effMin, 0.0)
    pl['c_field'] = w['m_field'] * pl['fielding_rank']
    pl['eff_stat'] = effStat.round(4)
    pl['rating'] = (pl['c_stat'] + pl['c_conf'] + pl['c_mlb'] + pl['c_hs']
                    + pl['c_prod'] + pl['c_twp'] + pl['c_field']).round(4)

    board = pl[pl['rating'] > 0].sort_values(
        ['rating', 'eff_stat', 'id'], ascending=[False, False, True]).reset_index(drop=True)
    board['rank'] = board.index + 1
    board['delta'] = np.where(board['official_rank'] > 0,
                              board['official_rank'] - board['rank'], np.nan)
    return board


# ----------------------------------------------------------------------- presets
st.session_state.setdefault('presets', {'Jason': None, 'Claude': None})
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault('sel_id', TORRES)


def load_weights(d):
    for k in DEFAULTS:
        st.session_state[k] = d[k]


def current_weights():
    return {k: st.session_state[k] for k in DEFAULTS}


def is_default():
    return all(abs(st.session_state[k] - DEFAULTS[k]) < 1e-9 for k in DEFAULTS)


# ------------------------------------------------------------------------- sidebar
sb = st.sidebar
sb.header('🥎 Weights')
pc = sb.columns(3)
if pc[0].button('Published', use_container_width=True):
    load_weights(DEFAULTS); st.rerun()
if pc[1].button('Jason', use_container_width=True, disabled=st.session_state.presets['Jason'] is None):
    load_weights(st.session_state.presets['Jason']); st.rerun()
if pc[2].button('Claude', use_container_width=True, disabled=st.session_state.presets['Claude'] is None):
    load_weights(st.session_state.presets['Claude']); st.rerun()
sc = sb.columns(2)
if sc[0].button('Save → Jason', use_container_width=True):
    st.session_state.presets['Jason'] = current_weights(); sb.success('Saved to Jason', icon='✅')
if sc[1].button('Save → Claude', use_container_width=True):
    st.session_state.presets['Claude'] = current_weights(); sb.success('Saved to Claude', icon='✅')
sb.caption('Published = the live formula. Save a tuning to Jason or Claude to compare.')

sb.markdown('---')
sb.caption('STAT WEIGHTS · hitting (inside the batting percentile)')
sb.slider('OPS', 0.0, 2.0, step=0.05, key='w_ops')
sb.slider('wOBA', 0.0, 2.0, step=0.05, key='w_woba')
sb.slider('ISO', 0.0, 2.0, step=0.05, key='w_iso')
sb.caption('STAT WEIGHTS · pitching')
sb.slider('OPS-against', 0.0, 2.0, step=0.05, key='w_opsa')
sb.slider('BABIP', 0.0, 2.0, step=0.05, key='w_babip')
sb.slider('K/BB', 0.0, 2.0, step=0.05, key='w_kbb')

sb.markdown('---')
sb.caption('TERM WEIGHTS · how much each component counts')
sb.slider('Stat percentile', 0.0, 2.0, step=0.05, key='m_stat')
sb.slider('Production', 0.0, 2.0, step=0.05, key='m_prod')
sb.slider('  ↳ production coefficient', 0.0, 1.0, step=0.05, key='prod_coef',
          help='0.50 × max(wRCE, wRAE) × division')
sb.slider('Conference', 0.0, 2.0, step=0.05, key='m_conf')
sb.slider('Pro projection', 0.0, 2.0, step=0.05, key='m_mlb')
sb.slider('HS pedigree', 0.0, 2.0, step=0.05, key='m_hs')
sb.slider('Two-way bonus', 0.0, 0.30, step=0.01, key='twp_coef', help='× the lesser of the two ranks')
sb.slider('Fielding', 0.0, 2.0, step=0.05, key='m_field')

sb.markdown('---')
sb.caption('PERCENTILE BASIS · division, seasons, playing-time floors')
sb.slider('D-I multiplier', 0.0, 1.25, step=0.05, key='div_D1')
sb.slider('D-II multiplier', 0.0, 1.25, step=0.05, key='div_D2')
sb.slider('D-III multiplier', 0.0, 1.25, step=0.05, key='div_D3')
sb.slider('2024 weight', 0.0, 3.0, step=0.25, key='yw_2024')
sb.slider('2025 weight', 0.0, 3.0, step=0.25, key='yw_2025')
sb.slider('2026 weight (anchor)', 0.0, 3.0, step=0.25, key='yw_2026')
sb.slider('Min plate appearances', 0, 100, step=5, key='min_pa')
sb.slider('Min batters faced', 0, 100, step=5, key='min_bf')

W = current_weights()
BOARD = compute_board(W)
if BOARD.empty:
    st.title('🥎 Softball Portal Ranking Sandbox')
    st.warning('Every rating is zero with the current weights. Raise a weight, or hit '
               '**Published** in the sidebar to reset.', icon='⚠️')
    st.stop()
BYID = {int(r.id): r for r in BOARD.itertuples()}

# ---------------------------------------------------------------------------- head
st.title('🥎 Softball Portal Ranking Sandbox')
st.markdown(
    '**Tune the methodology and watch the board move.** Three-season window '
    '`2024–2026` (2026 anchored at weight 2). The default state reproduces the published '
    '2026 softball portal board — **Isa Torres at #3**, behind Jayden Heavener and Sydney Schwartz.')
tag = 'Published formula' if is_default() else 'Custom weights'
st.caption(f'{tag} · {len(BOARD)} ranked players · rating = stat%ile + conf + two-way + '
           'pro + HS + production + fielding')

# ------------------------------------------------------------------------ spotlight
def comp_parts(r):
    return {'stat': r.c_stat, 'prod': r.c_prod, 'conf': r.c_conf, 'mlb': r.c_mlb,
            'hs': r.c_hs, 'twp': r.c_twp, 'field': r.c_field}


sel_id = st.session_state.sel_id
if sel_id not in BYID:
    sel_id = TORRES if TORRES in BYID else int(BOARD.iloc[0]['id'])
srow = BYID[sel_id]

left, right = st.columns([2, 3])
with left:
    st.subheader(srow.name)
    meta = f"{srow.institution}"
    if srow.division:
        meta += f" · {srow.division}"
    if srow.position:
        meta += f" · {srow.position}"
    if srow.class_year:
        meta += f" · {srow.class_year}"
    st.caption(meta + (f"  →  {srow.new_team}" if srow.new_team else ""))
    m1, m2, m3 = st.columns(3)
    m1.metric('Current rank', f"#{int(srow.rank)}")
    m2.metric('Rating', f"{srow.rating:.3f}")
    dval = srow.official_rank - srow.rank if srow.official_rank > 0 else None
    m3.metric('Published #', f"#{int(srow.official_rank)}" if srow.official_rank > 0 else '—',
              delta=(None if dval is None else (f"+{int(dval)}" if dval > 0 else (f"{int(dval)}" if dval < 0 else "—"))),
              help='Move vs the published 2026 rank')
    # preset comparison
    rowsline = []
    for nm in ('Jason', 'Claude'):
        pw = st.session_state.presets[nm]
        if pw is not None:
            b = compute_board(pw)
            rr = b[b['id'] == sel_id]
            rowsline.append(f"{nm}: **#{int(rr['rank'].iloc[0])}**" if len(rr) else f"{nm}: —")
    if rowsline:
        st.caption('Under saved presets → ' + ' · '.join(rowsline))

with right:
    parts = comp_parts(srow)
    total = srow.rating
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        for k, lb, c, _ in COMPS:
            v = parts[k]
            if v <= 0.0001:
                continue
            fig.add_bar(x=[v], y=['rating'], orientation='h', name=lb, marker_color=c,
                        hovertemplate=f'{lb}: %{{x:.4f}}<extra></extra>')
        fig.update_layout(barmode='stack', height=120, margin=dict(l=0, r=0, t=8, b=0),
                          showlegend=True, legend=dict(orientation='h', y=-0.5, font=dict(size=11)),
                          xaxis=dict(showgrid=False, title='', tickformat='.1f'),
                          yaxis=dict(showticklabels=False),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    except Exception:
        pass
    # why line
    top = sorted(((CLABEL[k], parts[k]) for k, *_ in COMPS if parts[k] > 0.0001),
                 key=lambda x: -x[1])[:2]
    whytop = ', '.join(f'{lb} `{v:.3f}`' for lb, v in top)
    above = BOARD[BOARD['rank'] < srow.rank].tail(2)
    lead = ''
    if len(above):
        names = ' & '.join(above['name'].str.split().str[-1].tolist())
        gap = above['rating'].iloc[-1] - srow.rating
        lead = f" {names} edge {srow.name.split()[0]} by `{abs(gap):.4f}`."
    st.markdown(f"**#{int(srow.rank)}** on a rating of `{total:.3f}`. Biggest lifts: {whytop}.{lead}")

# comparison table vs neighbors above
nb = BOARD[BOARD['rank'] <= srow.rank].tail(3) if srow.rank > 1 else BOARD.head(1)
if len(nb) > 1:
    cmp = pd.DataFrame({
        'Rank': nb['rank'].astype(int).map(lambda x: f'#{x}'),
        'Player': nb['name'].values,
        'Stat %ile': nb['c_stat'].round(3).values,
        'Production': nb['c_prod'].round(3).values,
        'Conf': nb['c_conf'].round(3).values,
        'Pro': nb['c_mlb'].round(3).values,
        'HS': nb['c_hs'].round(3).values,
        'Two-way': nb['c_twp'].round(3).values,
        'Rating': nb['rating'].round(3).values,
    })
    st.dataframe(cmp, use_container_width=True, hide_index=True)

st.markdown('---')

# ---------------------------------------------------------------------------- board
hc1, hc2 = st.columns([3, 1])
hc1.subheader('Live board')
topn = hc2.number_input('Show top', min_value=10, max_value=500, value=100, step=10)
q = st.text_input('Search player or school', '', placeholder='e.g. Torres, Florida St.')

view = BOARD
if q.strip():
    ql = q.strip().lower()
    view = view[view['name'].str.lower().str.contains(ql) | view['institution'].str.lower().str.contains(ql)]
view = view.head(int(topn))


def comp_block(r):
    # unicode mini stacked bar via colored blocks isn't reliable; show the top driver tag instead
    parts = {'stat': r.c_stat, 'prod': r.c_prod, 'conf': r.c_conf, 'mlb': r.c_mlb,
             'hs': r.c_hs, 'twp': r.c_twp, 'field': r.c_field}
    k = max(parts, key=parts.get)
    return CLABEL[k]


disp = pd.DataFrame({
    'Rank': view['rank'].astype(int),
    'Δ': view['delta'],
    'Player': view['name'],
    'School → Dest': [f"{i}{(' → ' + n) if n else ''}" for i, n in zip(view['institution'], view['new_team'])],
    'Pos': view['position'],
    'Rating': view['rating'].round(3),
    'Stat%': view['c_stat'].round(3),
    'Prod': view['c_prod'].round(3),
    'Conf': view['c_conf'].round(3),
    'Pro': view['c_mlb'].round(3),
    'HS': view['c_hs'].round(3),
    'TWP': view['c_twp'].round(3),
    'Pub.': view['official_rank'].map(lambda x: int(x) if x > 0 else None),
})


def _dc(v):
    if pd.isna(v):
        return 'color:#777'
    if v > 0:
        return 'color:#16a34a;font-weight:600'
    if v < 0:
        return 'color:#dc2626;font-weight:600'
    return 'color:#999'


styled = (disp.style
          .format({'Δ': lambda v: '' if pd.isna(v) else (f'▲{int(v)}' if v > 0 else (f'▼{int(-v)}' if v < 0 else '•')),
                   'Rating': '{:.3f}', 'Stat%': '{:.3f}', 'Prod': '{:.3f}', 'Conf': '{:.3f}',
                   'Pro': '{:.3f}', 'HS': '{:.3f}', 'TWP': '{:.3f}', 'Pub.': '{:.0f}'}, na_rep='—')
          .map(_dc, subset=['Δ'])
          .hide(axis='index'))

event = st.dataframe(styled, use_container_width=True, height=560,
                     on_select='rerun', selection_mode='single-row', key='board_lb')
if event.selection.rows:
    picked = view.iloc[event.selection.rows[0]]
    if int(picked['id']) != st.session_state.sel_id:
        st.session_state.sel_id = int(picked['id'])
        st.rerun()
st.caption('Click any row to make that player the spotlight above. '
           'Δ shows the move versus the published 2026 rank.')

# ------------------------------------------------------------------------- expanders
with st.expander('Show the formula'):
    st.latex(r'\text{rating} = m_{s}\cdot \text{stat\%ile} + m_{c}\cdot \text{conf} + '
             r'c_{twp}\cdot \min(\text{ranks}) + m_{p}\cdot \text{pro} + m_{h}\cdot \text{HS} + '
             r'm_{prod}\cdot \big[\,c_{prod}\cdot \max(wRCE,wRAE)\cdot \text{div}\,\big] + m_{f}\cdot \text{field}')
    st.markdown(
        f"**Stat percentile** = `max(hitting, pitching)`, each the percentile — within the "
        f"{len(PLAYERS)}-player pool — of a 3-season weighted sum: "
        f"`(w·OPS + w·wOBA + w·ISO) × PA%ile × season-weight × division`, summed over "
        f"2024/2025/2026 and divided by 3 (pitching uses OPS-against / BABIP / K-BB and BF%ile).")
    st.markdown('Current multipliers: '
                + ' · '.join(f'{CLABEL[k]} `{W[m]:.2f}×`' for k, m in
                             [('stat', 'm_stat'), ('prod', 'm_prod'), ('conf', 'm_conf'),
                              ('mlb', 'm_mlb'), ('hs', 'm_hs'), ('field', 'm_field')])
                + f" · prod coef `{W['prod_coef']:.2f}` · two-way `{W['twp_coef']:.2f}`")

with st.expander('How faithful is this to the published board?'):
    st.markdown(
        "- **At default weights the top of the board is exact** — Heavener (1), Schwartz (2), "
        "Torres (3), Hartley (4), Maestretti (5), Beachum (6) all match the published 2026 board.\n"
        "- **Stat percentile is delta-anchored.** Each player starts from their frozen published "
        "stat percentile; moving a stat weight re-percentiles the pool and shifts them *relative* to "
        "that anchor, so the baseline stays exact and your changes are the signal.\n"
        "- **Deep mid-board can differ from the frozen published list by a few spots** because the "
        "production term reads the nightly-refreshed wRCE/wRAE, while the published board was a "
        "point-in-time snapshot. That drift is data, not formula.\n"
        "- Pool = every 2026 softball portal entrant. Source: `rank_players.py` via "
        "`export_softball_sandbox.py`.")
