"""
wRCE Sandbox (hitting) — chart-builder page 26.

Transparent methodology bench: take our complete hitting stats, compute a production
rating with the refinements under consideration for the wRCE engine, and SHOW the work —
division vs conference wRC+ side by side, the park factor's point-effect per player, the
current published wRCE in view, and a per-player breakdown of how the levers move a ranking.

Levers: wOBA scale (college vs 1.0), park on/off, AB floor, per-event run-value multipliers,
and which baseline drives the rank (division or conference). Season-level, 2026, hitting only.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='wRCE Sandbox', layout='wide')
_APP_DIR = Path(__file__).resolve().parent.parent
FILES = {'Baseball': ('sandbox_hitters.csv', 'sandbox_const.json'),
         'Softball': ('sandbox_hitters_softball.csv', 'sandbox_const_softball.json')}
EVT = ['BB', 'HBP', '1B', '2B', '3B', 'HR']
DIV_ORDER = ['D-I', 'D-II', 'D-III']
DEFW = {e: 1.0 for e in EVT}


@st.cache_data
def load(sport):
    hf, cf = FILES[sport]
    h = pd.read_csv(_APP_DIR / 'data' / hf)
    for c in ['PF', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'IBB', 'HBP', 'SF', 'R', 'PA', 'cur_wRCE', 'cur_RC_AB']:
        h[c] = pd.to_numeric(h[c], errors='coerce').fillna(0.0)
    for c in ['name', 'team', 'div', 'conf']:
        h[c] = h[c].fillna('').astype(str)
    return h, json.load(open(_APP_DIR / 'data' / cf))


sport = st.sidebar.radio('Sport', list(FILES), horizontal=True)
st.sidebar.markdown('---')
H, CONST = load(sport)
DIVW = {d: CONST['div'][d]['w'] for d in CONST['div']}
DIV_SCALE = {d: CONST['div'][d]['scale'] for d in CONST['div']}
DIV_LGRPA = {d: CONST['div'][d]['lgRPA'] for d in CONST['div']}
CONF_LGRPA = {c: CONST['conf'][c]['lgRPA'] for c in CONST['conf']}
CONF_SET = set(CONST['conf'])


def wrcp(raapa, lgrpa, pf):
    return ((raapa + lgrpa) + (lgrpa - pf * lgrpa)) / lgrpa * 100.0


def pa_wmean(df, col):
    g = df.assign(_n=df['wOBA'] * df['PAden']).groupby(col).agg(_n=('_n', 'sum'), _d=('PAden', 'sum'))
    return (g['_n'] / g['_d']).to_dict()


def metrics(df, scale_mode, wmult):
    """Compute wOBA + division & conference wRC+ (park-adjusted and park-neutral)."""
    df = df.copy()
    df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']
    df['uBB'] = df['BB'] - df['IBB']
    df['PAden'] = df['AB'] + df['uBB'] + df['HBP'] + df['SF']
    df = df[df['PAden'] > 0].copy()
    num = np.zeros(len(df))
    for e in EVT:
        w = df['div'].map(lambda d: DIVW.get(d, DIVW['D-I'])[e]).to_numpy() * wmult[e]
        cnt = df['uBB'].to_numpy() if e == 'BB' else df[e].to_numpy()
        num += w * cnt
    df['wOBA'] = num / df['PAden'].to_numpy()
    scale = df['div'].map(DIV_SCALE).to_numpy() if scale_mode == 'College (ours)' else np.ones(len(df))

    # division baseline (lgwOBA recomputed under the current weights)
    lgw = pa_wmean(df, 'div')
    df['lgw_div'] = df['div'].map(lgw)
    df['lgrpa_div'] = df['div'].map(DIV_LGRPA)
    raapa = (df['wOBA'] - df['lgw_div']) / scale
    df['wRC_div_pk'] = wrcp(raapa, df['lgrpa_div'], df['PF'])
    df['wRC_div_np'] = wrcp(raapa, df['lgrpa_div'], 1.0)
    df['parkD_div'] = df['wRC_div_pk'] - df['wRC_div_np']

    # conference baseline (fall back to division where conference unknown)
    df['grp'] = np.where(df['conf'].isin(CONF_SET), df['conf'], df['div'])
    lgwc = pa_wmean(df, 'grp')
    df['lgw_conf'] = df['grp'].map(lgwc)
    df['lgrpa_conf'] = df['grp'].map(lambda g: CONF_LGRPA.get(g, DIV_LGRPA.get(g, 0.165)))
    raapac = (df['wOBA'] - df['lgw_conf']) / scale
    df['wRC_conf_pk'] = wrcp(raapac, df['lgrpa_conf'], df['PF'])
    df['wRC_conf_np'] = wrcp(raapac, df['lgrpa_conf'], 1.0)
    df['parkD_conf'] = df['wRC_conf_pk'] - df['wRC_conf_np']
    df['_scale'] = scale
    return df


# ------------------------------------------------------------------- sidebar levers
st.sidebar.header('Levers')
scale_mode = st.sidebar.radio('wOBA scale', ['College (ours)', 'Held at 1.0 (current)'],
                              help='Our college-specific scale vs the engine holding it at 1.0.')
rank_by = st.sidebar.radio('Rank by baseline', ['Division', 'Conference'],
                           help='Which wRC+ drives the ranking. Both are always shown in the table.')
park = st.sidebar.checkbox('Park-adjust the ranking', value=True,
                           help='Park effect is always shown in the Park Δ column; this decides whether the ranking uses it.')
min_ab = st.sidebar.slider('Qualifying floor — minimum AB', 10, 300, 50, 5)
st.sidebar.markdown('---')
with st.sidebar.expander('Per-event run values (× multiplier)'):
    wmult = {e: st.slider(e, 0.0, 2.0, 1.0, 0.05, key=f'wm_{e}') for e in EVT}
if st.sidebar.button('Reset multipliers'):
    for e in EVT:
        st.session_state[f'wm_{e}'] = 1.0
    st.rerun()

base_df = H[H['AB'] >= min_ab]
CUR = metrics(base_df, scale_mode, wmult)
BASE = metrics(base_df, 'College (ours)', DEFW)   # default config, for the "tinkering" delta

pk = park
CUR['wRC_div'] = CUR['wRC_div_pk'] if pk else CUR['wRC_div_np']
CUR['wRC_conf'] = CUR['wRC_conf_pk'] if pk else CUR['wRC_conf_np']
CUR['parkD'] = CUR['parkD_div'] if rank_by == 'Division' else CUR['parkD_conf']
rc = 'wRC_div' if rank_by == 'Division' else 'wRC_conf'
CUR['our_rank'] = CUR[rc].rank(ascending=False, method='min').astype(int)
BASE['base_rank'] = BASE['wRC_div_pk'].rank(ascending=False, method='min').astype(int)
CUR = CUR.merge(BASE[['pid', 'base_rank']], on='pid', how='left')
CUR['tinkerD'] = (CUR['base_rank'] - CUR['our_rank']).fillna(0).astype(int)
CUR = CUR.sort_values('our_rank').reset_index(drop=True)

# ------------------------------------------------------------------------- header
st.title(f"{'🥎' if sport == 'Softball' else '⚾'} wRCE Sandbox — {sport.lower()} hitting")
st.caption('Our linear-weight production rating with the work shown: division vs conference wRC+, the park '
           'factor’s point-effect, and the current published wRCE. 2026 · counts from hitting.csv. 100 = baseline-group average. '
           f'Weights and league baselines are {sport.lower()}-specific.')

m1, m2, m3, m4 = st.columns(4)
m1.metric('Players', f'{len(CUR):,}')
split = CUR['wRC_div'].sub(CUR['wRC_conf']).abs().median()
m2.metric('Median |Div − Conf| wRC+', f'{split:.0f}')
reg = CUR[CUR['PA'] >= 150]
m3.metric('Median park swing (regulars)', f"{reg['parkD'].abs().median():.1f}")
m4.metric('Median rank move from tinkering', f"{int(reg['tinkerD'].abs().median())}" if len(reg) else '—')

# ------------------------------------------------------------------- player inspect
st.subheader('Inspect a player — where conference, park, and wRCE come in')
names = CUR['name'] + '  ·  ' + CUR['team'] + '  (' + CUR['div'] + ')'
idx = st.selectbox('Player', range(len(CUR)), format_func=lambda i: names.iloc[i])
r = CUR.iloc[idx]
a, b, c = st.columns(3)
with a:
    st.markdown(f"**{r['name']}** — {r['team']}  \n{r['div']} · {r['conf'] or '—'}")
    st.markdown(f"PA {int(r['PA'])} · 1B {int(r['1B'])} · 2B {int(r['2B'])} · 3B {int(r['3B'])} · HR {int(r['HR'])} · uBB {int(r['uBB'])} · HBP {int(r['HBP'])}")
    st.metric('wOBA (current weights)', f"{r['wOBA']:.3f}")
with b:
    st.markdown('**Division baseline**  \n'
                f"lgwOBA {r['lgw_div']:.3f} · scale {r['_scale']:.3f} · lgR/PA {r['lgrpa_div']:.3f}")
    st.metric('wRC+ (Division)', f"{r['wRC_div']:.0f}")
    st.markdown(f"park factor **{r['PF']:.3f}** → park effect **{r['parkD_div']:+.1f}** wRC+  \n"
                f"(before park {r['wRC_div_np']:.0f} → after park {r['wRC_div_pk']:.0f})")
with c:
    st.markdown('**Conference baseline**  \n'
                f"{r['conf'] or 'n/a'} · lgwOBA {r['lgw_conf']:.3f} · lgR/PA {r['lgrpa_conf']:.3f}")
    st.metric('wRC+ (Conference)', f"{r['wRC_conf']:.0f}", delta=f"{r['wRC_conf'] - r['wRC_div']:+.0f} vs Division")
    st.markdown(f"**Current engine** · wRCE **{r['cur_wRCE']:.1f}** · RC/AB {r['cur_RC_AB']:.3f}  \n"
                f"our rank **{int(r['our_rank'])}** · default-config rank {int(r['base_rank'])} → tinkering moved it **{int(r['tinkerD']):+d}**")

# ----------------------------------------------------------------------- full board
st.subheader('Board')
f1, f2, f3 = st.columns([2, 2, 3])
dsel = f1.multiselect('Division', DIV_ORDER, default=DIV_ORDER)
confs = ['(all)'] + sorted(x for x in CUR['conf'].unique() if x)
csel = f2.selectbox('Conference', confs)
search = f3.text_input('Search player / team')
view = CUR[CUR['div'].isin(dsel)]
if csel != '(all)':
    view = view[view['conf'] == csel]
if search:
    s = search.lower()
    view = view[view['name'].str.lower().str.contains(s) | view['team'].str.lower().str.contains(s)]

cols = ['our_rank', 'name', 'team', 'div', 'conf', 'PA', 'HR', 'wOBA', 'wRC_div', 'wRC_conf', 'PF', 'parkD', 'cur_wRCE', 'base_rank', 'tinkerD']
show = view[cols].copy()
show.columns = ['Rank', 'Player', 'Team', 'Div', 'Conference', 'PA', 'HR', 'wOBA', 'wRC+ (Div)', 'wRC+ (Conf)', 'PF', 'Park Δ', 'Cur wRCE', 'Default rank', 'Tinker Δ']
st.dataframe(
    show.style.format({'wOBA': '{:.3f}', 'wRC+ (Div)': '{:.0f}', 'wRC+ (Conf)': '{:.0f}',
                       'PF': '{:.3f}', 'Park Δ': '{:+.1f}', 'Cur wRCE': '{:.1f}', 'Tinker Δ': '{:+d}'})
        .background_gradient(subset=['Park Δ'], cmap='PuOr', vmin=-6, vmax=6)
        .background_gradient(subset=['Tinker Δ'], cmap='RdYlGn', vmin=-100, vmax=100),
    use_container_width=True, height=520, hide_index=True)
st.caption('wRC+ (Div) vs (Conf): same player against the division vs their conference (100 = that group’s avg). '
           'Park Δ: wRC+ points added/removed by the park factor. Cur wRCE: the current published metric. '
           'Tinker Δ: rank move vs the default config (college scale, park on, division, weights ×1) — i.e. the effect of your current lever settings.')
st.download_button('Download current view (CSV)', view[cols].to_csv(index=False).encode(),
                   file_name='wrce_sandbox_view.csv', mime='text/csv')
