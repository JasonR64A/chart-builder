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
TEAMS_FILE = _APP_DIR / 'data' / 'teams.csv'
CONF_FILE = _APP_DIR / 'data' / 'conferences.csv'
# Sport -> (cells csv, progames csv, teams.csv sport keyword). Baseball unchanged;
# softball reads the *_softball.csv files from export-park-factor-lab-v3.py.
SPORT_FILES = {
    'Baseball': ('park_factor_cells.csv', 'park_factor_progames.csv', 'park_factor_runs.csv', 'park_factor_homeroad.csv', 'base'),
    'Softball': ('park_factor_cells_softball.csv', 'park_factor_progames_softball.csv', 'park_factor_runs_softball.csv', 'park_factor_homeroad_softball.csv', 'soft'),
}

DIVISION_LABELS = {'D1': 'D-I', 'D2': 'D-II', 'D3': 'D-III'}
STAT_LABELS = {'Single (1B)': 'singles', 'Double (2B)': 'doubles', 'Triple (3B)': 'triples', 'Home run (HR)': 'hr'}
SIDE_CELLS = {'Both': ['bat', 'pit'], 'Hitters': ['bat'], 'Pitchers': ['pit']}
LOC_CELLS = {'All players': ['home', 'opp'], 'Visitors only': ['opp'], 'Home team only': ['home']}
YEAR_OPTS = {'2026': [2026], '2025': [2025], 'Both (2025+26)': [2025, 2026]}
MONTHS = {2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 1: 'Jan'}
TIER_ORDER = ['Top 50', 'Mid (51-150)', 'Low (151+)', 'Unranked']

st.set_page_config(page_title='Park Factor Lab', layout='wide')

st.sidebar.header('Park Factor Lab')
sport = st.sidebar.radio('Sport', list(SPORT_FILES), index=0, horizontal=True)
_cells_name, _pro_name, _runs_name, _hr_name, _sport_kw = SPORT_FILES[sport]


@st.cache_data
def load(cells_name, pro_name, runs_name, hr_name):
    return (pd.read_csv(_APP_DIR / 'data' / cells_name),
            pd.read_csv(_APP_DIR / 'data' / pro_name),
            pd.read_csv(_APP_DIR / 'data' / runs_name),
            pd.read_csv(_APP_DIR / 'data' / hr_name))


cells, progames, runs_df, homeroad = load(_cells_name, _pro_name, _runs_name, _hr_name)


@st.cache_data
def conf_map(sport_kw):
    tm = pd.read_csv(TEAMS_FILE)
    tm = tm[tm['sport'].astype(str).str.lower().str.contains(sport_kw)]
    cf = pd.read_csv(CONF_FILE)
    id2name = dict(zip(cf['id'], cf['name']))
    id2ab = dict(zip(cf['id'], cf['abbreviation']))
    nm = {str(n).strip(): id2name.get(c) for n, c in zip(tm['name'], tm['conference_id'])}
    ab = {str(n).strip(): id2ab.get(c) for n, c in zip(tm['name'], tm['conference_id'])}
    return nm, ab


TEAM2CONF, TEAM2ABBR = conf_map(_sport_kw)

# ── sidebar ─────────────────────────────────────────────────────────────────────
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
stat_label = st.sidebar.radio('Stat', list(STAT_LABELS), index=3,
                              help='Each event has its own park factor. HR and triples move most with the park; '
                                   'singles barely move. Pick one to inspect a single component — the leaderboard '
                                   'True PF combines all four (+ Score) regardless of this pick.')
side_label = st.sidebar.radio('Side', list(SIDE_CELLS), index=0)
loc_label = st.sidebar.radio('Who', list(LOC_CELLS), index=0,
                             help='"Visitors only" is the cleanest neutral read.')
hand_label = st.sidebar.radio('Batter handedness', ['All', 'Left (L)', 'Right (R)'], index=0,
                              help="Splits the HITTING side by batter handedness — a park's lefty vs righty factor. Switch hitters are excluded from L/R; picking L or R makes it a hitters-only read.")
HAND = {'All': None, 'Left (L)': 'L', 'Right (R)': 'R'}[hand_label]
regress = st.sidebar.slider('Regression to 1.0', 0, 100, 0,
                            help='Shrink each park toward neutral.') / 100.0
st.sidebar.caption('Context weight — value better competition more')
wk_w = st.sidebar.slider('Midweek weight (vs weekend)', 0, 100, 60,
                         help='How much a midweek game counts relative to a weekend game. 100 = equal.') / 100.0
nc_w = st.sidebar.slider('Non-conf weight (vs conference)', 0, 100, 70,
                         help='How much a non-conference game counts relative to a conference game. 100 = equal.') / 100.0
st.sidebar.caption('Finalize — blend Pro in as a guardrail')
pro_blend = st.sidebar.slider('Pro blend into Final', 0, 100, 33,
                              help='Default 33 = 2×Ours + 1×Pro (our model 2x, Pro 1x). 0 = pure Ours; 50 = even split.') / 100.0
st.sidebar.caption('True PF — HR emphasis')
hr_emph = st.sidebar.slider('HR weight in True PF', 0, 100, 70,
                            help='How much the clean HR factor drives True PF. The remaining weight splits across '
                                 '2B/1B/3B (50/35/15). 100 = pure HR; 70 = HR-led with the other events nudging.') / 100.0

stat = STAT_LABELS[stat_label]

# ── apply context filters ───────────────────────────────────────────────────────
def ctx(df):
    return df[df['year'].isin(years) & (df['division'] == div)
              & df['conf_game'].isin(conf_sel) & df['midweek'].isin(day_sel)
              & df['month'].isin(month_sel) & df['opp_tier'].isin(tier_sel)]


fc, fp, fr = ctx(cells), ctx(progames), ctx(runs_df)


def ours_ratio(df, stat, sides, locs, wk=1.0, nc=1.0, bat=None):
    if bat is not None:   # handedness split: hitters only, chosen bat side
        f = df[(df['side'] == 'bat') & (df['bat'] == bat) & df['loc'].isin(locs)].copy()
    else:
        f = df[df['side'].isin(sides) & df['loc'].isin(locs)].copy()
    # context weight — value better competition more; multiplies actual AND expected
    # equally per game-cell, so a neutral park still reads 1.00 at any weighting.
    w = (np.where(f['midweek'] == 'Midweek', wk, 1.0)
         * np.where(f['conf_game'] == 'Non-conf', nc, 1.0))
    f['_a'] = f[f'act_{stat}'] * w
    f['_e'] = f[f'exp_{stat}'] * w
    g = f.groupby('park').agg(a=('_a', 'sum'), e=('_e', 'sum'))
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


def runs_ratio(df):
    # park-clean runs factor: Σ actual team-runs ÷ Σ leave-park-out expected runs
    # (each team vs its own runs/game at every OTHER venue), pooling home + visiting
    # offenses. Replaces the Pro home/road Score inside True PF — no staff/schedule confound.
    g = df.groupby('park').agg(a=('act_runs', 'sum'), e=('exp_runs', 'sum'))
    return (g['a'] / g['e']).where(g['e'] > 0)


ours = ours_ratio(fc, stat, ['bat', 'pit'], ['home', 'opp'], bat=HAND).rename('Ours')  # neutral: equal weights
adj_raw = ours_ratio(fc, stat, SIDE_CELLS[side_label], LOC_CELLS[loc_label], wk=wk_w, nc=nc_w, bat=HAND)
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
# Final PF — Ours (Adjusted) is primary; Pro blends in only as a guardrail (pro_blend).
# Where Pro is undefined (too few home/road games), Final = Adjusted (no blend).
d['Final'] = np.where(d['Pro'].notna(), (1 - pro_blend) * d['Adjusted'] + pro_blend * d['Pro'], d['Adjusted'])

# ── True PF — home/road-vs-competition (HR-driven) ───────────────────────────────
# Per-event park factors (1B/2B/3B/HR) from the home/road-vs-competition engine:
# for each event, (team's home performance ÷ what the competition it faced normally
# allows) ÷ (same on the road). Cancels team quality (same team home/road) AND
# competition (÷ each opponent's road rate). Full-sample both-year fingerprint joined
# by park — so these four columns reflect ALL games, NOT the year/context filters
# (those still drive Pro/Ours/Adjusted). True PF = HR-driven weighted geometric mean;
# the HR-weight slider sets how much HR leads, the rest splits 2B/1B/3B = 50/35/15.
_hrp = homeroad[homeroad['division'] == div].drop_duplicates('park').set_index('park')
for _lab, _col in [('HR', 'pf_hr'), ('2B', 'pf_doubles'), ('1B', 'pf_singles'), ('3B', 'pf_triples')]:
    d[_lab] = d.index.map(_hrp[_col]) if _col in _hrp.columns else np.nan
_rem = 1.0 - hr_emph
EVW = {'HR': hr_emph, '2B': _rem * 0.50, '1B': _rem * 0.35, '3B': _rem * 0.15}
_lognum = pd.Series(0.0, index=d.index)
_wacc = pd.Series(0.0, index=d.index)
for _lab, _w in EVW.items():
    _v = np.log(d[_lab].clip(lower=0.05))
    _ok = d[_lab].notna()
    _lognum = _lognum.add((_v * _w).where(_ok, 0.0), fill_value=0.0)
    _wacc = _wacc.add(pd.Series(_w, index=d.index).where(_ok, 0.0), fill_value=0.0)
d['True PF'] = np.exp(_lognum / _wacc.replace(0, np.nan))

# ── header ──────────────────────────────────────────────────────────────────────
st.title(f'🏟️ Park Factor Lab · {sport}')
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
           f'{f" · {int(regress*100)}% regressed" if regress else ""}'
           f'{"" if (wk_w == 1.0 and nc_w == 1.0) else f" · wt mid×{wk_w:.2f}/non-conf×{nc_w:.2f}"}'
           f'{"" if HAND is None else f" · {hand_label} hitters only"}  |  {len(d)} parks')

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
        [['Park', 'Team', 'Conf', 'True PF', '1B', '2B', '3B', 'HR', 'Pro', 'Ours', 'Adjusted', 'Final', 'Home G']]
        .sort_values('True PF', ascending=False))
show['Home G'] = show['Home G'].astype('Int64')


styled = (show.style
          .format({c: '{:.3f}' for c in ['True PF', 'Pro', 'Ours', 'Adjusted', 'Final']}, na_rep='—')
          .format({c: '{:.2f}' for c in ['1B', '2B', '3B', 'HR']}, na_rep='—')
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

# ── park fingerprint — home/road-vs-competition by event ─────────────────────────
st.subheader('🧬 Park fingerprint — home/road vs competition')
_fp = _hrp.loc[sel_park] if sel_park in _hrp.index else None
if _fp is not None:
    st.caption(f'**{sel_park}** — how much more (or less) this park yields each event than the '
               f'competition normally allows, home vs road. Full-sample (both years). '
               f'{int(_fp.get("home_g", 0))} home / {int(_fp.get("road_g", 0))} road games.')
    fcols = st.columns(5)
    _tp = show.loc[show['Park'] == sel_park, 'True PF']
    fcols[0].metric('True PF', f'{_tp.iloc[0]:.2f}' if len(_tp) and pd.notna(_tp.iloc[0]) else '—')
    for col, (lab, c) in zip(fcols[1:], [('1B', 'pf_singles'), ('2B', 'pf_doubles'), ('3B', 'pf_triples'), ('HR', 'pf_hr')]):
        v = _fp.get(c)
        lean = '' if pd.isna(v) else ('↑' if v > 1.03 else '↓' if v < 0.97 else '≈')
        col.metric(lab, f'{v:.2f}' if pd.notna(v) else '—', delta=lean, delta_color='off')
    st.caption('HR is the cleanest park signal; 3B is the noisiest (triples are rare — read it softly).')
else:
    st.caption(f'**{sel_park}** — no home/road fingerprint (park below the 30-home / 20-road game floor).')

with st.expander(f'📐 Show the math — True PF for {sel_park}'):
    if _fp is not None:
        st.markdown(
            'True PF uses the **home/road-vs-competition** method. For each event, compare what the team '
            '(and its pitchers at this park) did vs **what the competition it faced normally allows** — '
            'at home, then on the road — and divide. Expected = Σ (each opponent\'s **road** rate for the '
            'event × opportunities that game), pooling the home team\'s hitting + pitching, both years.')
        st.latex(r'PF_{\text{event}}=\frac{\text{home actual}\,/\,\text{home expected}}{\text{road actual}\,/\,\text{road expected}}')
        _mrows = []
        for _lab, _e in [('1B', 'singles'), ('2B', 'doubles'), ('3B', 'triples'), ('HR', 'hr')]:
            _ha, _hx, _ra, _rx = _fp.get('ha_' + _e), _fp.get('hx_' + _e), _fp.get('ra_' + _e), _fp.get('rx_' + _e)
            _hrr = _ha / _hx if (pd.notna(_hx) and _hx > 0) else float('nan')
            _rrr = _ra / _rx if (pd.notna(_rx) and _rx > 0) else float('nan')
            _mrows.append({'Event': _lab,
                           'home act': None if pd.isna(_ha) else round(_ha, 1), 'home exp': None if pd.isna(_hx) else round(_hx, 1),
                           'home ratio': None if pd.isna(_hrr) else round(_hrr, 2),
                           'road act': None if pd.isna(_ra) else round(_ra, 1), 'road exp': None if pd.isna(_rx) else round(_rx, 1),
                           'road ratio': None if pd.isna(_rrr) else round(_rrr, 2),
                           'PF = home/road': None if pd.isna(_fp.get('pf_' + _e)) else round(_fp.get('pf_' + _e), 2)})
        st.dataframe(pd.DataFrame(_mrows), hide_index=True, use_container_width=True)
        _r = 1.0 - hr_emph
        _w = {'HR': hr_emph, '2B': _r * 0.50, '1B': _r * 0.35, '3B': _r * 0.15}
        _ln, _ws, _terms = 0.0, 0.0, []
        for _lab, _e in [('HR', 'hr'), ('2B', 'doubles'), ('1B', 'singles'), ('3B', 'triples')]:
            _pf = _fp.get('pf_' + _e)
            if pd.notna(_pf):
                _ln += _w[_lab] * np.log(max(_pf, 0.05)); _ws += _w[_lab]
                _terms.append(f'{_lab} {_pf:.2f}^{_w[_lab]:.2f}')
        _tpf = np.exp(_ln / _ws) if _ws > 0 else float('nan')
        st.markdown(f'**True PF = HR-driven weighted geometric mean** (HR emphasis {int(hr_emph * 100)}%): '
                    f'weights HR {_w["HR"]:.2f} · 2B {_w["2B"]:.2f} · 1B {_w["1B"]:.2f} · 3B {_w["3B"]:.2f}')
        st.markdown(f'= exp( Σ wₑ·ln(PFₑ) ÷ Σ wₑ )  =  **{_tpf:.2f}**')
        st.caption('Dividing by each opponent\'s road rate cancels *who* they faced; home÷road for the same '
                   'team cancels *how good the team is*. 3B is the soft number — triples are rare.')
    else:
        st.caption('No home/road fingerprint for this park (below the game floor), so there\'s no True PF math to show.')

st.subheader('🔬 Park diagnostic — both sides')
st.caption(f'**{sel_park}** · {stat_label} · {ctx_txt}  —  click any row above to inspect a different park. '
           'A real park effect moves hitters AND pitchers the same way; a one-sided move is usually '
           'competition or roster, not the fences.')


def _pc(side, loc):
    s = fc[(fc['park'] == sel_park) & (fc['side'] == side) & (fc['loc'] == loc)]
    if HAND is not None and side == 'bat':
        s = s[s['bat'] == HAND]
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

with st.expander(f'📐 Show the math — Ours / Adjusted ({stat_label}) for {sel_park}'):
    if HAND is not None:
        sub = fc[(fc['park'] == sel_park) & (fc['side'] == 'bat') & (fc['bat'] == HAND) & fc['loc'].isin(LOC_CELLS[loc_label])]
    else:
        sub = fc[(fc['park'] == sel_park) & fc['side'].isin(SIDE_CELLS[side_label]) & fc['loc'].isin(LOC_CELLS[loc_label])]
    wsub = (np.where(sub['midweek'] == 'Midweek', wk_w, 1.0) * np.where(sub['conf_game'] == 'Non-conf', nc_w, 1.0))
    A = float((sub[f'act_{stat}'] * wsub).sum())
    E = float((sub[f'exp_{stat}'] * wsub).sum())
    if wk_w != 1.0 or nc_w != 1.0:
        st.caption(f'Context-weighted: midweek ×{wk_w:.2f}, non-conference ×{nc_w:.2f} — applied to both actual and expected before summing.')
    raw = A / E if E > 0 else float('nan')
    adj = 1.0 + (raw - 1.0) * (1.0 - regress)
    st.markdown(
        f'The park factor is **total actual {stat_label.lower()} ÷ total expected {stat_label.lower()}** over the '
        'games your filters kept. Each player\'s *expected* is their rate at **every other park** (leave-park-out) '
        'times their plate appearances here — so the number only reflects the park, not the player or the roster.')
    st.latex(r'PF=\frac{\sum \text{actual}}{\sum \text{expected}}\,,\qquad \text{expected}_i=r_i^{\ \text{elsewhere}}\times PA_i')
    if E > 0:
        st.markdown(f'### PF = {A:,.0f} ÷ {E:,.1f} = **{raw:.3f}**')
        if regress > 0:
            st.markdown(f'Then regressed {int(regress*100)}% toward 1.00:  '
                        f'1 + ({raw:.3f} − 1) × (1 − {regress:.2f}) = **{adj:.3f}**')
    st.caption('The four cells that pool into it (✓ = included by your Side / Who):')
    rows = []
    for lab, side, loc in cellmap:
        s = fc[(fc['park'] == sel_park) & (fc['side'] == side) & (fc['loc'] == loc)]
        a = float(s[f'act_{stat}'].sum())
        e = float(s[f'exp_{stat}'].sum())
        rows.append({'Cell': lab, 'Actual': round(a, 1), 'Expected': round(e, 1),
                     'Ratio': round(a / e, 3) if e > 0 else None,
                     'In this view': '✓' if (side in SIDE_CELLS[side_label] and loc in LOC_CELLS[loc_label]) else ''})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.markdown(
        f'**What moves the number:** change an *object* — Side / Who / Regression — and different cells are '
        f'summed or shrunk. Change a *context filter* (now: {ctx_txt}) and games that don\'t match are dropped '
        'before summing. That is the whole mechanism; nothing else touches it.')

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
        '- **Stat:** each event (1B / 2B / 3B / HR) has its own park factor — **HR and triples move '
        'most** with the fences, singles barely move. The Stat radio picks which one Ours/Adjusted show.\n'
        '- **True PF + the 1B/2B/3B/HR columns** (new) use the **home/road-vs-competition** method: for '
        'each event, take how much the team beat (or trailed) *what the competition it faced normally '
        'allows* — **at home**, divided by the same thing **on the road**. Dividing by each opponent\'s '
        'own road rate cancels *who* they faced; the home÷road ratio for the same team cancels *how good '
        'the team is*. What\'s left is the park. **True PF** is the HR-driven weighted geometric mean of '
        'the four (HR-weight slider; the rest splits 2B/1B/3B = 50/35/15). These are a **full-sample '
        'both-year fingerprint joined by park**, so they do NOT move with the year/context filters (those '
        'still drive Pro/Ours/Adjusted). **3B is the noisy one** — triples are rare, so read it softly.\n'
        '- **Pro** — the classic runs method, (home runs/game) ÷ (road runs/game) per team. Kept as '
        'a reference, but it is **noisy for college**: lopsided home/road schedules and the home '
        'team\'s own pitching distort it (it can call a real bandbox pitcher-friendly), so trust '
        'Ours over Pro.\n'
        '- **Competition control:** the **visitors-only** and **opponent-tier** filters are how you '
        'check that a park reads the same against good competition and without the home roster.\n'
        '- **Parks** keyed by play-by-play venue; 2026 excludes neutral sites via the venue match, '
        '2025 has no venue so all home games count. **2025 + 2026 pooled** by default (our full PBP).'
    )
