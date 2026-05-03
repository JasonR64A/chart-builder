"""
Regional Preview — 4-team double-elim regional snapshot.

Six sections:
  1. Header with 4 team logos + RPI badges
  2. Outcome donut — Bradley-Terry win probability from RPI
  3. Head-to-head comparison cards (6 unique team pairs)
  4. Bullpen comparison (4 columns)
  5. Who's hot — top hitters + pitchers per team
  6. Starter spider charts (3 expected weekend starters per team)
"""
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _APP_DIR / 'data'
PBP_DIR = _APP_DIR / 'pbp_data'
LOGOS = _APP_DIR / 'team_logos_512'

st.set_page_config(page_title='Regional Preview — 64 Analytics', layout='wide')


# ── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_teams():
    return pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False).fillna('')


@st.cache_data(show_spinner=False)
def load_rpi(sport, division):
    p = DATA_DIR / f'{sport}_rpi_{division}.csv'
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=False)


@st.cache_data(show_spinner=False)
def load_player_rank():
    return pd.read_csv(DATA_DIR / 'player_rank.csv', low_memory=False, encoding='utf-8-sig')


@st.cache_data(show_spinner=False)
def load_players():
    try:
        return pd.read_csv(DATA_DIR / 'players.csv', low_memory=False, encoding='cp1252')
    except UnicodeDecodeError:
        return pd.read_csv(DATA_DIR / 'players.csv', low_memory=False, encoding='latin-1')


@st.cache_data(show_spinner=False)
def load_hitting_pbp(sport, division):
    p = PBP_DIR / sport / f'hitting_pbp_{division}.csv'
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    df['date_parsed'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    return df


@st.cache_data(show_spinner=False)
def load_pitching_pbp(sport, division):
    p = PBP_DIR / sport / f'pitching_pbp_{division}.csv'
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    df['date_parsed'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    return df


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## Regional Preview')
    sport = st.selectbox('Sport', ['baseball', 'softball'], format_func=str.title)
    division = st.selectbox('Division', ['D1', 'D2', 'D3'])

    teams_df = load_teams()
    sport_label = 'Baseball' if sport == 'baseball' else 'Softball'
    sport_teams = teams_df[teams_df['sport'] == sport_label].copy()

    # Sort by RPI (best first) for default selection
    rpi_df = load_rpi(sport, division)
    if not rpi_df.empty:
        # rpi_df has team-name + rpi rank
        rpi_col = next((c for c in rpi_df.columns if 'rpi' in c.lower() and 'rank' in c.lower()), None)
        if rpi_col is None:
            rpi_col = next((c for c in rpi_df.columns if c.lower() in ('rpi', 'rank')), None)
        name_col = next((c for c in rpi_df.columns if c.lower() in ('team', 'name')), None)
        if rpi_col and name_col:
            rpi_lookup = dict(zip(rpi_df[name_col], pd.to_numeric(rpi_df[rpi_col], errors='coerce')))
            sport_teams['_rpi'] = sport_teams['name'].map(rpi_lookup)
            sport_teams = sport_teams.sort_values('_rpi', na_position='last')

    team_options = sport_teams['name'].tolist()
    if not team_options:
        st.error('No teams found for this sport.')
        st.stop()

    seed1 = st.selectbox('1-Seed', team_options, key='seed1')
    seed2 = st.selectbox('2-Seed', [t for t in team_options if t != seed1], key='seed2')
    seed3 = st.selectbox('3-Seed', [t for t in team_options if t not in (seed1, seed2)], key='seed3')
    seed4 = st.selectbox('4-Seed', [t for t in team_options if t not in (seed1, seed2, seed3)], key='seed4')

    regional_name = st.text_input('Regional name', value=f'{seed1} Regional')
    lookback_days = st.slider('Hot-list lookback (days)', 7, 30, 14)


teams = [seed1, seed2, seed3, seed4]
seeds = [1, 2, 3, 4]

# Resolve team metadata
def _team_row(name):
    r = sport_teams[sport_teams['name'] == name]
    return r.iloc[0] if not r.empty else None

team_meta = {t: _team_row(t) for t in teams}
team_ids = {t: int(team_meta[t]['id']) if team_meta[t] is not None and pd.notna(team_meta[t]['id']) else None
            for t in teams}
team_rpi = {t: team_meta[t].get('current_rpi') if team_meta[t] is not None else None for t in teams}


# ── View selector — Bracket Preview (existing) vs Top Hitters (new) ─────────
view = st.radio('View', ['Bracket Preview', 'Top Hitters'],
                horizontal=True, label_visibility='collapsed', key='rp_view')

if view == 'Top Hitters':
    from app_lib.regionals_top_hitters import render_tab as _render_top_hitters_tab
    _SEED_PALETTE = ['#C41230', '#29335c', '#F5A623', '#0F8A5F']
    _hitting_df_th = pd.read_csv(DATA_DIR / 'hitting.csv', low_memory=False)
    _players_df_th = load_players()
    _confs_df_th = pd.read_csv(DATA_DIR / 'conferences.csv', low_memory=False)
    _player_rank_df_th = load_player_rank()
    _render_top_hitters_tab(
        teams, seeds, team_ids, sport, division, regional_name,
        _hitting_df_th, _players_df_th, sport_teams,
        accent_for=lambda _tid, seed: _SEED_PALETTE[(seed - 1) % 4],
        conferences_df=_confs_df_th,
        player_rank_df=_player_rank_df_th,
    )
    st.stop()


# ── Header — 4 team cards ───────────────────────────────────────────────────
st.markdown(f'## {regional_name}')
st.caption(f'{sport.title()} {division} · 4-team double-elimination · Lookback {lookback_days}d')

hdr_cols = st.columns(4)
for col, team, seed in zip(hdr_cols, teams, seeds):
    with col:
        tid = team_ids[team]
        if tid is not None:
            logo_path = LOGOS / f'{tid}.png'
            if logo_path.exists():
                st.image(str(logo_path), width=120)
        st.markdown(f"**#{seed} · {team}**")
        rpi_val = team_rpi[team]
        rpi_txt = f"{int(float(rpi_val))}" if rpi_val and str(rpi_val).strip() else '—'
        conf = team_meta[team]['conference_id'] if team_meta[team] is not None else '—'
        st.caption(f"RPI #{rpi_txt}")


# ── Section 2: Outcome donut (Bradley-Terry from RPI) ───────────────────────
st.markdown('---')
st.markdown('### Win-the-Regional probability')

def _bt_strength(rpi_rank):
    """Lower RPI rank = better team. Convert rank → strength via 1 / sqrt(rank)."""
    try:
        r = float(rpi_rank)
        if r <= 0: return 1.0
        return 1.0 / np.sqrt(r)
    except (TypeError, ValueError):
        return 0.5


strengths = {t: _bt_strength(team_rpi[t]) for t in teams}

def _p_win(a, b):
    """Bradley-Terry P(a beats b)."""
    sa, sb = strengths[a], strengths[b]
    return sa / (sa + sb) if (sa + sb) else 0.5


def _simulate_regional(n=20000, rng=None):
    """Monte-Carlo sample n bracket realizations; return P(team wins regional)."""
    if rng is None:
        rng = np.random.default_rng(42)
    wins = {t: 0 for t in teams}

    for _ in range(n):
        # G1: 1 v 4, G2: 2 v 3
        def play(a, b):
            return a if rng.random() < _p_win(a, b) else b

        s1, s2, s3, s4 = teams
        g1w = play(s1, s4); g1l = s1 if g1w == s4 else s4
        g2w = play(s2, s3); g2l = s2 if g2w == s3 else s3
        # G3: losers
        g3w = play(g1l, g2l)
        # G4: winners
        g4w = play(g1w, g2w); g4l = g1w if g4w == g2w else g2w
        # G5: G3 winner vs G4 loser
        g5w = play(g3w, g4l)
        # G6: G4 winner vs G5 winner
        g6w = play(g4w, g5w)
        # G7 (only if G5 winner won G6 — G4 winner needs second loss to be eliminated)
        if g6w == g5w:
            champ = play(g4w, g5w)
        else:
            champ = g4w
        wins[champ] += 1
    return {t: w / n for t, w in wins.items()}


with st.spinner('Simulating bracket…'):
    win_p = _simulate_regional(20000)

donut_colors = ['#C41230', '#29335c', '#F5A623', '#0F8A5F']
fig = go.Figure(go.Pie(
    labels=[f'#{s} {t}' for t, s in zip(teams, seeds)],
    values=[win_p[t] for t in teams],
    hole=0.55,
    marker=dict(colors=donut_colors[:4]),
    textinfo='label+percent',
    sort=False,
))
fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                  showlegend=False)
st.plotly_chart(fig, use_container_width=True)
st.caption('Bradley-Terry on RPI rank, 20k Monte-Carlo bracket sims. Higher RPI rank = stronger.')


# ── Helpers for the remaining sections ──────────────────────────────────────
def _pbp_team_match(pbp_df, team_name):
    """PBP teamName has mascot suffix; pick the shortest one that startswith()."""
    if pbp_df is None or pbp_df.empty:
        return None
    candidates = [n for n in pbp_df['teamName'].dropna().unique()
                  if isinstance(n, str) and n.startswith(team_name)]
    candidates.sort(key=len)
    return candidates[0] if candidates else None


@st.cache_data(show_spinner=False)
def _team_hitting_stats(team_name, sport, division, days):
    pbp = load_hitting_pbp(sport, division)
    if pbp.empty:
        return {}
    pbp_name = _pbp_team_match(pbp, team_name)
    if pbp_name is None:
        return {}
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    h = pbp[(pbp['teamName'] == pbp_name) & (pbp['date_parsed'] >= cutoff)].copy()
    if h.empty:
        h = pbp[pbp['teamName'] == pbp_name].copy()
    for c in ['ab', 'h', 'hr', 'bb', 'hbp', 'sf', 'tb', 'r', 'rbi', 'doubles', 'triples']:
        if c in h.columns:
            h[c] = pd.to_numeric(h[c], errors='coerce').fillna(0)
    ab = float(h['ab'].sum()) if 'ab' in h.columns else 0
    hits = float(h['h'].sum()) if 'h' in h.columns else 0
    bb = float(h['bb'].sum()) if 'bb' in h.columns else 0
    hbp = float(h['hbp'].sum()) if 'hbp' in h.columns else 0
    sf = float(h['sf'].sum()) if 'sf' in h.columns else 0
    tb = float(h['tb'].sum()) if 'tb' in h.columns else hits
    pa = ab + bb + hbp + sf
    return {
        'BA':  hits / ab if ab else 0,
        'OBP': (hits + bb + hbp) / pa if pa else 0,
        'SLG': tb / ab if ab else 0,
        'OPS': ((hits + bb + hbp) / pa if pa else 0) + (tb / ab if ab else 0),
        'HR':  int(h['hr'].sum()) if 'hr' in h.columns else 0,
        'R':   int(h['r'].sum()) if 'r' in h.columns else 0,
        'AB':  int(ab),
    }


@st.cache_data(show_spinner=False)
def _team_pitching_stats(team_name, sport, division, days):
    pbp = load_pitching_pbp(sport, division)
    if pbp.empty:
        return {}
    pbp_name = _pbp_team_match(pbp, team_name)
    if pbp_name is None:
        return {}
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    p = pbp[(pbp['teamName'] == pbp_name) & (pbp['date_parsed'] >= cutoff)].copy()
    if p.empty:
        p = pbp[pbp['teamName'] == pbp_name].copy()
    for c in ['ip', 'h', 'er', 'bb', 'so', 'bf', 'hrA']:
        if c in p.columns:
            p[c] = pd.to_numeric(p[c], errors='coerce').fillna(0)
    # Convert IP notation (3.2 = 3 2/3) to outs
    def _ip_to_outs(ip_val):
        if pd.isna(ip_val): return 0
        whole = int(ip_val); frac = round((ip_val - whole) * 10)
        return whole * 3 + frac
    outs = p['ip'].apply(_ip_to_outs).sum() if 'ip' in p.columns else 0
    ip = outs / 3.0
    h_a = float(p['h'].sum()) if 'h' in p.columns else 0
    er = float(p['er'].sum()) if 'er' in p.columns else 0
    bb_a = float(p['bb'].sum()) if 'bb' in p.columns else 0
    so = float(p['so'].sum()) if 'so' in p.columns else 0
    return {
        'ERA':  9 * er / ip if ip else 0,
        'WHIP': (h_a + bb_a) / ip if ip else 0,
        'K/9':  9 * so / ip if ip else 0,
        'BAA':  h_a / float(p['bf'].sum()) if 'bf' in p.columns and p['bf'].sum() else 0,
        'IP':   ip,
        'HR_a': int(p['hrA'].sum()) if 'hrA' in p.columns else 0,
    }


# ── Section 3: 6 head-to-head comparison cards ──────────────────────────────
st.markdown('---')
st.markdown('### Head-to-Head Comparisons')

# Pre-compute everyone's stats once
team_h_stats = {t: _team_hitting_stats(t, sport, division, lookback_days) for t in teams}
team_p_stats = {t: _team_pitching_stats(t, sport, division, lookback_days) for t in teams}

# All 6 unique pairs by seed index
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def _stat_row(label, va, vb, fmt='{:.3f}', better='high'):
    """One row of a comparison card: label, both team values, with the leader bolded."""
    try:
        a = float(va); b = float(vb)
        a_better = (a > b) if better == 'high' else (a < b)
    except (TypeError, ValueError):
        a_better = False; b_better = False
    sa = fmt.format(va) if va is not None else '—'
    sb = fmt.format(vb) if vb is not None else '—'
    if a_better:
        sa = f'**{sa}**'
    elif not a_better and va != vb:
        sb = f'**{sb}**'
    return f'| {label} | {sa} | {sb} |'


for i in range(0, 6, 2):
    cols = st.columns(2)
    for j, col in enumerate(cols):
        if i + j >= 6: break
        ai, bi = pairs[i + j]
        a, b = teams[ai], teams[bi]
        sa, sb = seeds[ai], seeds[bi]
        with col:
            st.markdown(f'#### #{sa} {a}  vs  #{sb} {b}')
            ha, hb = team_h_stats.get(a, {}), team_h_stats.get(b, {})
            pa, pb = team_p_stats.get(a, {}), team_p_stats.get(b, {})
            md = ['| Stat | ' + a + ' | ' + b + ' |', '|---|---|---|']
            md.append(_stat_row('AVG',  ha.get('BA'),  hb.get('BA')))
            md.append(_stat_row('OBP',  ha.get('OBP'), hb.get('OBP')))
            md.append(_stat_row('SLG',  ha.get('SLG'), hb.get('SLG')))
            md.append(_stat_row('OPS',  ha.get('OPS'), hb.get('OPS')))
            md.append(_stat_row('HR',   ha.get('HR'),  hb.get('HR'),  fmt='{:d}'))
            md.append(_stat_row('ERA',  pa.get('ERA'), pb.get('ERA'), fmt='{:.2f}', better='low'))
            md.append(_stat_row('WHIP', pa.get('WHIP'),pb.get('WHIP'),fmt='{:.2f}', better='low'))
            md.append(_stat_row('K/9',  pa.get('K/9'), pb.get('K/9'), fmt='{:.1f}'))
            md.append(_stat_row('BAA',  pa.get('BAA'), pb.get('BAA'), fmt='{:.3f}', better='low'))
            st.markdown('\n'.join(md))


# ── Section 4: Bullpen comparison (4 columns) ───────────────────────────────
st.markdown('---')
st.markdown('### Bullpen Comparison (last {}d)'.format(lookback_days))


def _team_relievers(team_name, sport, division, days, n=5):
    pbp = load_pitching_pbp(sport, division)
    if pbp.empty:
        return pd.DataFrame()
    pbp_name = _pbp_team_match(pbp, team_name)
    if pbp_name is None:
        return pd.DataFrame()
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    p = pbp[(pbp['teamName'] == pbp_name) & (pbp['date_parsed'] >= cutoff)].copy()
    if p.empty or 'playerName' not in p.columns:
        return pd.DataFrame()
    for c in ['ip', 'h', 'er', 'bb', 'so']:
        if c in p.columns:
            p[c] = pd.to_numeric(p[c], errors='coerce').fillna(0)
    # Identify relievers as pitchers who DIDN'T start most of their appearances.
    # Simplification: any pitcher with at least 1 outing in window, sorted by appearances.
    # 'is_starter' may not exist in box-score schema — fall back to GS=0 / count appearances.
    def _ip_to_outs(ip_val):
        if pd.isna(ip_val): return 0
        whole = int(ip_val); frac = round((ip_val - whole) * 10)
        return whole * 3 + frac
    p['_outs'] = p['ip'].apply(_ip_to_outs) if 'ip' in p.columns else 0
    grp = p.groupby('playerName').agg(
        App=('ip', 'count'),
        Outs=('_outs', 'sum'),
        H=('h', 'sum'), ER=('er', 'sum'),
        BB=('bb', 'sum'), SO=('so', 'sum'),
    ).reset_index()
    # Filter to "relievers": short outings (avg outs < 9 = avg < 3 IP)
    grp['avg_outs'] = grp['Outs'] / grp['App'].replace(0, 1)
    relievers = grp[grp['avg_outs'] < 9].copy()
    if relievers.empty:
        relievers = grp.copy()
    relievers['IP'] = relievers['Outs'] / 3.0
    relievers['ERA'] = (9 * relievers['ER'] / relievers['IP'].replace(0, np.nan)).round(2)
    relievers['K/9'] = (9 * relievers['SO'] / relievers['IP'].replace(0, np.nan)).round(1)
    return relievers.sort_values('App', ascending=False).head(n)[
        ['playerName', 'App', 'IP', 'ERA', 'K/9', 'SO', 'BB']
    ]


bp_cols = st.columns(4)
for col, team, seed in zip(bp_cols, teams, seeds):
    with col:
        st.markdown(f"**#{seed} {team}**")
        bp = _team_relievers(team, sport, division, lookback_days, n=5)
        if bp.empty:
            st.caption('No reliever data')
        else:
            bp_disp = bp.copy()
            bp_disp['IP'] = bp_disp['IP'].round(1)
            bp_disp.columns = ['Player', 'App', 'IP', 'ERA', 'K/9', 'K', 'BB']
            st.dataframe(bp_disp, hide_index=True, use_container_width=True)


# ── Section 5: Who's hot ────────────────────────────────────────────────────
st.markdown('---')
st.markdown(f'### Who\'s Hot (last {lookback_days}d)')


def _hot_hitters(team_name, sport, division, days, n=3):
    pbp = load_hitting_pbp(sport, division)
    if pbp.empty:
        return pd.DataFrame()
    pbp_name = _pbp_team_match(pbp, team_name)
    if pbp_name is None:
        return pd.DataFrame()
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    h = pbp[(pbp['teamName'] == pbp_name) & (pbp['date_parsed'] >= cutoff)].copy()
    if h.empty or 'playerName' not in h.columns:
        return pd.DataFrame()
    for c in ['ab', 'h', 'hr', 'bb', 'hbp', 'sf', 'tb', 'rbi', 'r']:
        if c in h.columns:
            h[c] = pd.to_numeric(h[c], errors='coerce').fillna(0)
    grp = h.groupby('playerName').agg(
        AB=('ab', 'sum'), H=('h', 'sum'), HR=('hr', 'sum'),
        BB=('bb', 'sum'), HBP=('hbp', 'sum'), SF=('sf', 'sum'),
        TB=('tb', 'sum'), RBI=('rbi', 'sum'), R=('r', 'sum'),
    ).reset_index()
    grp = grp[grp['AB'] >= 8]
    if grp.empty:
        return pd.DataFrame()
    grp['AVG'] = (grp['H'] / grp['AB']).round(3)
    grp['OBP'] = ((grp['H'] + grp['BB'] + grp['HBP']) / (grp['AB'] + grp['BB'] + grp['HBP'] + grp['SF']).replace(0, np.nan)).round(3)
    grp['SLG'] = (grp['TB'] / grp['AB']).round(3)
    grp['OPS'] = (grp['OBP'] + grp['SLG']).round(3)
    return grp.sort_values('OPS', ascending=False).head(n)[
        ['playerName', 'AB', 'AVG', 'OBP', 'SLG', 'OPS', 'HR', 'RBI']
    ]


def _hot_pitcher(team_name, sport, division, days):
    pbp = load_pitching_pbp(sport, division)
    if pbp.empty:
        return pd.DataFrame()
    pbp_name = _pbp_team_match(pbp, team_name)
    if pbp_name is None:
        return pd.DataFrame()
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    p = pbp[(pbp['teamName'] == pbp_name) & (pbp['date_parsed'] >= cutoff)].copy()
    if p.empty or 'playerName' not in p.columns:
        return pd.DataFrame()
    for c in ['ip', 'h', 'er', 'bb', 'so']:
        if c in p.columns:
            p[c] = pd.to_numeric(p[c], errors='coerce').fillna(0)
    def _ip_to_outs(ip_val):
        if pd.isna(ip_val): return 0
        whole = int(ip_val); frac = round((ip_val - whole) * 10)
        return whole * 3 + frac
    p['_outs'] = p['ip'].apply(_ip_to_outs)
    grp = p.groupby('playerName').agg(
        App=('ip', 'count'), Outs=('_outs', 'sum'),
        H=('h', 'sum'), ER=('er', 'sum'), BB=('bb', 'sum'), SO=('so', 'sum'),
    ).reset_index()
    grp = grp[grp['Outs'] >= 9]   # at least 3 IP in window
    if grp.empty:
        return pd.DataFrame()
    grp['IP'] = (grp['Outs'] / 3.0).round(1)
    grp['ERA'] = (9 * grp['ER'] / (grp['Outs'] / 3.0).replace(0, np.nan)).round(2)
    grp['K/9'] = (9 * grp['SO'] / (grp['Outs'] / 3.0).replace(0, np.nan)).round(1)
    return grp.sort_values('ERA', ascending=True).head(1)[
        ['playerName', 'App', 'IP', 'ERA', 'K/9', 'H', 'BB']
    ]


hot_cols = st.columns(4)
for col, team, seed in zip(hot_cols, teams, seeds):
    with col:
        st.markdown(f"**#{seed} {team}**")
        st.caption('Hot Hitters')
        hh = _hot_hitters(team, sport, division, lookback_days, n=3)
        if hh.empty:
            st.caption('— no qualifying hitters —')
        else:
            hh.columns = ['Player', 'AB', 'AVG', 'OBP', 'SLG', 'OPS', 'HR', 'RBI']
            st.dataframe(hh, hide_index=True, use_container_width=True)
        st.caption('Hot Pitcher')
        hp = _hot_pitcher(team, sport, division, lookback_days)
        if hp.empty:
            st.caption('— no qualifying pitchers —')
        else:
            hp.columns = ['Player', 'App', 'IP', 'ERA', 'K/9', 'H', 'BB']
            st.dataframe(hp, hide_index=True, use_container_width=True)


# ── Section 6: Spider charts for 3 expected starters per team ───────────────
st.markdown('---')
st.markdown('### Expected Weekend Starters')


def _team_starters(team_name, sport, division, days, n=3):
    """Pitchers who started ≥1 game in the lookback window, ranked by total outs."""
    pbp = load_pitching_pbp(sport, division)
    if pbp.empty:
        return pd.DataFrame()
    pbp_name = _pbp_team_match(pbp, team_name)
    if pbp_name is None:
        return pd.DataFrame()
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days * 1.5)  # wider for starters (weekly cadence)
    p = pbp[(pbp['teamName'] == pbp_name) & (pbp['date_parsed'] >= cutoff)].copy()
    if p.empty or 'playerName' not in p.columns:
        return pd.DataFrame()
    for c in ['ip', 'h', 'er', 'bb', 'so', 'bf']:
        if c in p.columns:
            p[c] = pd.to_numeric(p[c], errors='coerce').fillna(0)
    def _ip_to_outs(ip_val):
        if pd.isna(ip_val): return 0
        whole = int(ip_val); frac = round((ip_val - whole) * 10)
        return whole * 3 + frac
    p['_outs'] = p['ip'].apply(_ip_to_outs)
    # Heuristic: starter = average ≥9 outs per appearance (3+ IP/start)
    grp = p.groupby('playerName').agg(
        GS=('ip', 'count'), Outs=('_outs', 'sum'),
        H=('h', 'sum'), ER=('er', 'sum'),
        BB=('bb', 'sum'), SO=('so', 'sum'),
        BF=('bf', 'sum') if 'bf' in p.columns else ('h', 'sum'),
    ).reset_index()
    grp['avg_outs'] = grp['Outs'] / grp['GS'].replace(0, 1)
    starters = grp[grp['avg_outs'] >= 9].copy()
    if starters.empty:
        return pd.DataFrame()
    starters['IP'] = (starters['Outs'] / 3.0).round(1)
    starters['ERA'] = (9 * starters['ER'] / (starters['Outs'] / 3.0).replace(0, np.nan)).round(2)
    starters['WHIP'] = ((starters['H'] + starters['BB']) / (starters['Outs'] / 3.0).replace(0, np.nan)).round(2)
    starters['K/9'] = (9 * starters['SO'] / (starters['Outs'] / 3.0).replace(0, np.nan)).round(1)
    starters['BB/9'] = (9 * starters['BB'] / (starters['Outs'] / 3.0).replace(0, np.nan)).round(1)
    starters['BAA'] = (starters['H'] / starters['BF'].replace(0, np.nan)).round(3)
    return starters.sort_values('Outs', ascending=False).head(n)


def _build_starter_spider(starters_df, team_name):
    """Plotly radar comparing up to 3 starters on 5 axes."""
    if starters_df.empty:
        return None
    axes = ['ERA', 'WHIP', 'K/9', 'BB/9', 'BAA']
    # Normalize each axis 0-100 (lower-is-better axes inverted)
    invert = {'ERA': True, 'WHIP': True, 'BB/9': True, 'BAA': True, 'K/9': False}
    def norm(val, axis):
        # rough scale: ERA 0-9, WHIP 0.5-2.5, K/9 0-15, BB/9 0-7, BAA 0.150-0.400
        scales = {'ERA': (0, 9), 'WHIP': (0.5, 2.5), 'K/9': (0, 15),
                  'BB/9': (0, 7), 'BAA': (0.150, 0.400)}
        lo, hi = scales[axis]
        v = max(min(float(val), hi), lo)
        pct = (v - lo) / (hi - lo) if hi != lo else 0.5
        return (1 - pct) * 100 if invert[axis] else pct * 100
    fig = go.Figure()
    palette = ['#C41230', '#29335c', '#F5A623']
    for i, (_, p) in enumerate(starters_df.iterrows()):
        vals = [norm(p[a], a) for a in axes]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=axes + [axes[0]],
            fill='toself',
            name=p['playerName'],
            line=dict(color=palette[i % 3]),
            opacity=0.55,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False)),
        showlegend=True, height=320, margin=dict(l=10, r=10, t=20, b=10),
        title=dict(text=team_name, font=dict(size=14)),
    )
    return fig


sp_cols = st.columns(2)
for i, (team, seed) in enumerate(zip(teams, seeds)):
    with sp_cols[i % 2]:
        st.markdown(f"**#{seed} {team}**")
        starters = _team_starters(team, sport, division, lookback_days, n=3)
        if starters.empty:
            st.caption('No qualifying starters in window')
        else:
            sp_fig = _build_starter_spider(starters, team)
            if sp_fig is not None:
                st.plotly_chart(sp_fig, use_container_width=True)
            disp = starters[['playerName', 'GS', 'IP', 'ERA', 'WHIP', 'K/9', 'BB/9', 'BAA']].copy()
            disp.columns = ['Pitcher', 'GS', 'IP', 'ERA', 'WHIP', 'K/9', 'BB/9', 'BAA']
            st.dataframe(disp, hide_index=True, use_container_width=True)


st.markdown('---')
st.markdown('### Shareable graphic')

# ── Single-panel 1080×1350 IG-feed design ──────────────────────────────────
# Sports Reference / FiveThirtyEight density. Eggshell over navy stage, tabular
# nums, ruled tables, JetBrains Mono labels, 2pt closing rule.
import base64
import math

BRAND_64A_WIDE     = _APP_DIR / 'assets' / 'logo-64a-wide.png'
BRAND_WORDMARK     = _APP_DIR / 'assets' / 'branding' / 'wordmark_red_black.png'
BRAND_CIRCLE_RB    = _APP_DIR / 'assets' / 'branding' / 'circle_red_black.png'
BRAND_CIRCLE_RW    = _APP_DIR / 'assets' / 'branding' / 'circle_red_white.png'
BRAND_EMBLEM       = _APP_DIR / 'assets' / 'branding' / 'emblem.png'
BRAND_EMBLEM_MONO  = _APP_DIR / 'assets' / 'branding' / 'emblem_mono.png'


def _xe(s):
    if s is None: return ''
    return (str(s).replace('&', '&amp;')
                  .replace('<', '&lt;')
                  .replace('>', '&gt;'))


def _embed_png(path, x, y, w, h, opacity=1.0):
    if not Path(path).exists():
        return ''
    try:
        b64 = base64.b64encode(Path(path).read_bytes()).decode('ascii')
    except Exception:
        return ''
    href = f'data:image/png;base64,{b64}'
    return (f'<image href="{href}" xlink:href="{href}" '
            f'x="{x}" y="{y}" width="{w}" height="{h}" '
            f'opacity="{opacity}" preserveAspectRatio="xMidYMid meet" '
            f'pointer-events="none"/>')


# ── Tokens ─────────────────────────────────────────────────────────────────
INK_900 = '#0F1B2D'
INK_700 = '#2A3550'
INK_500 = 'rgba(15,27,45,0.62)'
INK_400 = 'rgba(15,27,45,0.45)'
INK_300 = 'rgba(15,27,45,0.28)'
INK_200 = 'rgba(15,27,45,0.14)'
INK_100 = 'rgba(15,27,45,0.07)'
INK_RULE = 'rgba(15,27,45,0.10)'
BRAND_RED = '#C41230'
BRAND_NAVY = '#0F2A4D'
BG_EGG = '#F0EAD6'

# Seed-based accent palette (real team colors not in teams.csv yet → use seeds)
SEED_ACCENTS = {
    1: '#0A2240',  # navy
    2: '#C84A1E',  # burnt orange
    3: '#3D8AB8',  # carolina blue
    4: '#2E7D5B',  # forest green
}


def _accent_for_seed(seed):
    return SEED_ACCENTS.get(seed, BRAND_NAVY)


def _monogram_for(name):
    """Single-letter monogram from team short name (skip 'St.', '&', etc.)."""
    if not name: return '?'
    s = name.strip()
    return s[0].upper()


# ── Team-color + team-logo helpers ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _team_dominant_color(tid: int | None) -> str | None:
    """Extract a team's primary brand color from its logo PNG. Skips
    near-white, near-black, and near-grayscale (silver-ish) pixels so the
    chromatic primary wins. Returns hex like '#8C1D40'. Mirrors the helper
    used in the PBP_Analytics Share Graphic so colors stay consistent
    across pages."""
    if tid is None:
        return None
    try:
        from PIL import Image
        from collections import Counter
    except Exception:
        return None
    p = LOGOS / f'{int(tid)}.png'
    if not p.exists():
        return None
    try:
        img = Image.open(p).convert('RGBA').resize((96, 96), Image.LANCZOS)
        cleaned = []
        for r, g, b, a in img.getdata():
            if a < 220:
                continue
            if r > 235 and g > 235 and b > 235:
                continue
            if r < 18 and g < 18 and b < 18:
                continue
            if max(r, g, b) - min(r, g, b) < 20:
                continue
            cleaned.append((r // 16 * 16, g // 16 * 16, b // 16 * 16))
        if not cleaned:
            return None
        top = Counter(cleaned).most_common(1)[0][0]
        return f'#{top[0]:02x}{top[1]:02x}{top[2]:02x}'
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _team_logo_b64(tid: int | None) -> str | None:
    """Return data:image/png;base64,... URL for a team logo, or None."""
    if tid is None:
        return None
    p = LOGOS / f'{int(tid)}.png'
    if not p.exists():
        return None
    try:
        import base64 as _b64
        return f'data:image/png;base64,{_b64.b64encode(p.read_bytes()).decode("ascii")}'
    except Exception:
        return None


def _accent_for_team(tid, seed):
    """Team's brand color when available, else seed-palette fallback so we
    never render an empty accent. Same pattern Share Graphic uses."""
    return _team_dominant_color(tid) or _accent_for_seed(seed)


# ── Round-by-round Monte Carlo (extends _simulate_regional) ────────────────
def _simulate_regional_full(n=20000, rng=None):
    """Returns per-team round survival fractions: r1 (post-G1/G2), r2 (W-bracket
    or losers-G3 alive), r3 (reach regional final), r4 (champ). Last value is
    same as win_p; included for completeness."""
    if rng is None:
        rng = np.random.default_rng(42)
    cnt = {t: {'r1': 0, 'r2': 0, 'r3': 0, 'r4': 0} for t in teams}
    s1, s2, s3, s4 = teams
    for _ in range(n):
        def play(a, b):
            return a if rng.random() < _p_win(a, b) else b
        g1w = play(s1, s4); g1l = s1 if g1w == s4 else s4
        g2w = play(s2, s3); g2l = s2 if g2w == s3 else s3
        # r1: alive after G1/G2 = winners of openers
        for t in (g1w, g2w): cnt[t]['r1'] += 1
        # plus losers play G3 and the loser is eliminated (so g1l/g2l counted at r1 too — they're alive heading INTO opener's loss bracket)
        # Better definition: r1 = "won opener" only. Losers go to elim-G3.
        g3w = play(g1l, g2l)  # G3 loser eliminated
        # r2: alive heading into G4 = G1w, G2w, G3w (the 3-team set that remains)
        for t in (g1w, g2w, g3w): cnt[t]['r2'] += 1
        # G4: winners' final
        g4w = play(g1w, g2w); g4l = g1w if g4w == g2w else g2w
        # G5: G3 winner vs G4 loser (loser-out)
        g5w = play(g3w, g4l)
        # r3: reach regional final = G4w + G5w (the two who play G6)
        for t in (g4w, g5w): cnt[t]['r3'] += 1
        # G6: regional final game 1
        g6w = play(g4w, g5w)
        # G7 (only if G5 winner won G6 — G4 winner needs second loss to be eliminated)
        if g6w == g5w:
            champ = play(g4w, g5w)
        else:
            champ = g4w
        cnt[champ]['r4'] += 1
    return {t: {k: v / n for k, v in d.items()} for t, d in cnt.items()}


# ── Season-long aggregates from PBP for radar / depth tables ────────────────
@st.cache_data(show_spinner=False)
def _team_pbp_full(team_name, sport, division):
    """All season hitting + pitching + fielding rows for a team."""
    h_pbp = load_hitting_pbp(sport, division)
    p_pbp = load_pitching_pbp(sport, division)
    pbp_name = _pbp_team_match(h_pbp, team_name)
    h = h_pbp[h_pbp['teamName'] == pbp_name].copy() if pbp_name else pd.DataFrame()
    p_pbp_name = _pbp_team_match(p_pbp, team_name)
    p = p_pbp[p_pbp['teamName'] == p_pbp_name].copy() if p_pbp_name else pd.DataFrame()
    return h, p


def _ip_to_outs(ip_val):
    if pd.isna(ip_val): return 0
    whole = int(ip_val); frac = round((ip_val - whole) * 10)
    return whole * 3 + frac


@st.cache_data(show_spinner=False)
def _team_radar_perf(team_name, sport, division, last_n_games=None):
    """Per-team radar inputs. Nine raw metrics now: RUNS/G, OPS, OBP, ERA,
    WHIP, K%, FLD%, K/BB (hitting — fewer is better), K/BB (pitching —
    more is better). If last_n_games is set, restrict to that window's
    most recent games (by date) for the L25 overlay."""
    h, p = _team_pbp_full(team_name, sport, division)
    if h.empty or p.empty:
        return None
    if last_n_games is not None:
        recent_dates = h['date_parsed'].dropna().sort_values().unique()[-last_n_games:]
        h = h[h['date_parsed'].isin(recent_dates)]
        p = p[p['date_parsed'].isin(recent_dates)]
    if h.empty or p.empty:
        return None
    for c in ['ab','h','hr','bb','hbp','sf','tb','r','k']:
        if c in h.columns: h[c] = pd.to_numeric(h[c], errors='coerce').fillna(0)
    for c in ['ip','er','so','bf','bb','h']:
        if c in p.columns: p[c] = pd.to_numeric(p[c], errors='coerce').fillna(0)
    games = h['date_parsed'].nunique() if 'date_parsed' in h.columns else 1
    ab = float(h['ab'].sum()); hits = float(h['h'].sum())
    bb = float(h['bb'].sum()); hbp = float(h['hbp'].sum())
    sf = float(h['sf'].sum()); tb = float(h['tb'].sum()) if 'tb' in h.columns else hits
    runs = float(h['r'].sum()) if 'r' in h.columns else 0
    h_k = float(h['k'].sum()) if 'k' in h.columns else 0
    pa = ab + bb + hbp + sf
    obp = (hits + bb + hbp) / pa if pa else 0
    slg = tb / ab if ab else 0
    ops = obp + slg
    h_kbb = (h_k / bb) if bb > 0 else 0  # lower = better for hitters
    outs = p['ip'].apply(_ip_to_outs).sum() if 'ip' in p.columns else 0
    ip = outs / 3.0
    er = float(p['er'].sum()); so = float(p['so'].sum())
    bf = float(p['bf'].sum()) if 'bf' in p.columns else 0
    p_h  = float(p['h'].sum())  if 'h'  in p.columns else 0
    p_bb = float(p['bb'].sum()) if 'bb' in p.columns else 0
    era  = 9 * er / ip if ip else 0
    whip = (p_bb + p_h) / ip if ip else 0
    kpct = so / bf if bf else 0
    p_kbb = (so / p_bb) if p_bb > 0 else 0  # higher = better for pitchers
    # Fielding %: (PO + A) / (PO + A + E) from fielding PBP
    f_path = PBP_DIR / sport / f'fielding_pbp_{division}.csv'
    fld = 0.96
    if f_path.exists():
        try:
            f_df = pd.read_csv(f_path, low_memory=False, usecols=lambda c: c in ('teamName','po','a','e','date'))
            f_df['date_parsed'] = pd.to_datetime(f_df['date'], format='mixed', errors='coerce')
            f_team = f_df[f_df['teamName'].fillna('').str.startswith(team_name)]
            if last_n_games is not None and not f_team.empty:
                rd = f_team['date_parsed'].dropna().sort_values().unique()[-last_n_games:]
                f_team = f_team[f_team['date_parsed'].isin(rd)]
            for c in ['po','a','e']:
                if c in f_team.columns:
                    f_team[c] = pd.to_numeric(f_team[c], errors='coerce').fillna(0)
            tot = float(f_team['po'].sum()) + float(f_team['a'].sum()) + float(f_team['e'].sum())
            if tot > 0:
                fld = (float(f_team['po'].sum()) + float(f_team['a'].sum())) / tot
        except Exception:
            pass
    return {
        'RUNS': runs / games if games else 0,
        'OPS': ops, 'OBP': obp,
        'ERA': era, 'WHIP': whip, 'KPCT': kpct, 'FLD': fld,
        'HKBB': h_kbb, 'PKBB': p_kbb,
    }


@st.cache_data(show_spinner=False)
def _division_metric_distributions(sport, division):
    """Sorted per-metric value list across every team in the sport+division.
    Defined here (right after _team_radar_perf) because the percentile lookup
    is invoked at top-level page-render time *before* the radar geometry
    helpers section further down — moving it earlier keeps Python's
    forward-reference rules happy."""
    teams_csv = load_teams()
    sport_label = 'Baseball' if sport == 'baseball' else 'Softball'
    confs = pd.read_csv(DATA_DIR / 'conferences.csv', low_memory=False)
    div_label = {'D1': 'D-I', 'D2': 'D-II', 'D3': 'D-III'}[division]
    valid_conf_ids = set(confs[(confs['division'] == div_label) & (confs['name'] != 'Big Sky Conference')]['id'])
    div_teams = teams_csv[(teams_csv['sport'] == sport_label) & (teams_csv['conference_id'].isin(valid_conf_ids))]['name'].dropna().unique().tolist()
    dist = {k: [] for k in ('RUNS', 'OPS', 'OBP', 'HKBB', 'ERA', 'WHIP', 'PKBB', 'KPCT', 'FLD')}
    for t in div_teams:
        perf = _team_radar_perf(t, sport, division, last_n_games=None)
        if not perf: continue
        for k in dist:
            v = perf.get(k)
            if v is not None:
                dist[k].append(float(v))
    for k in dist:
        dist[k].sort()
    return dist


def _percentile_rank(value, sorted_vals, lower_better=False):
    """0..1 percentile rank for value within sorted_vals. Inverts when
    lower_better=True so a low ERA maps to a high radar fraction."""
    if not sorted_vals:
        return 0.5
    import bisect
    pos = bisect.bisect_left(sorted_vals, value)
    pct = pos / len(sorted_vals)
    return 1.0 - pct if lower_better else pct


@st.cache_data(show_spinner=False)
def _team_top_pitchers(team_name, sport, division, n=8):
    """Top N pitchers by IP. First 3 (by IP) get role SP1/SP2/SP3 highlight.
    Returns IP, FIP, WHIP per the user's preferred staff display."""
    _, p = _team_pbp_full(team_name, sport, division)
    if p.empty or 'playerName' not in p.columns:
        return []
    for c in ['ip','er','so','h','bb','hrA','hb']:
        if c in p.columns: p[c] = pd.to_numeric(p[c], errors='coerce').fillna(0)
    p['_outs'] = p['ip'].apply(_ip_to_outs) if 'ip' in p.columns else 0
    grp = p.groupby('playerName').agg(
        Outs=('_outs','sum'), App=('ip','count'),
        ER=('er','sum'), SO=('so','sum'), H=('h','sum'),
        BB=('bb','sum'), HRA=('hrA','sum'), HBP=('hb','sum'),
    ).reset_index()
    grp['IP'] = grp['Outs'] / 3.0
    grp = grp[grp['IP'] >= 1.0]
    ip_safe = grp['IP'].replace(0, np.nan)
    grp['WHIP'] = ((grp['BB'] + grp['H']) / ip_safe).fillna(0)
    # FIP = (13*HR + 3*(BB+HBP) - 2*K) / IP + FIP_constant (3.0 college approx)
    grp['FIP']  = ((13 * grp['HRA'] + 3 * (grp['BB'] + grp['HBP'])
                    - 2 * grp['SO']) / ip_safe).fillna(0) + 3.0
    grp['avg_outs'] = grp['Outs'] / grp['App'].replace(0, 1)
    grp = grp.sort_values('IP', ascending=False).head(n)
    out = []
    for idx, (_, row) in enumerate(grp.iterrows()):
        if idx < 3:
            role = f'SP{idx+1}'
        elif row['avg_outs'] < 3:
            role = 'CL'
        elif row['avg_outs'] < 9:
            role = 'RP'
        else:
            role = 'SP'
        out.append({'name': row['playerName'], 'role': role,
                    'ip': float(row['IP']),
                    'fip': float(row['FIP']),
                    'whip': float(row['WHIP'])})
    return out


@st.cache_data(show_spinner=False)
def _league_woba(sport, division):
    """League-wide wOBA for the sport+division. Used as the wRAA baseline.
    Falls back to a sensible college-ball average if PBP is unavailable."""
    h_pbp = load_hitting_pbp(sport, division)
    if h_pbp.empty:
        return 0.330
    df = h_pbp.copy()
    for c in ('ab','h','doubles','triples','hr','bb','hbp','sf'):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    AB = df['ab'].sum(); H = df['h'].sum()
    DBL = df.get('doubles', pd.Series(0)).sum()
    TRP = df.get('triples', pd.Series(0)).sum()
    HR = df.get('hr', pd.Series(0)).sum()
    BB = df.get('bb', pd.Series(0)).sum(); HBP = df.get('hbp', pd.Series(0)).sum()
    SF = df.get('sf', pd.Series(0)).sum()
    SINGLES = H - DBL - TRP - HR
    denom = AB + BB + HBP + SF
    if denom == 0:
        return 0.330
    return float((0.690*BB + 0.722*HBP + 0.888*SINGLES + 1.271*DBL
                  + 1.616*TRP + 2.101*HR) / denom)


@st.cache_data(show_spinner=False)
def _team_top_hitters(team_name, sport, division, n=9):
    """Top N hitters by PA. Returns PA, OPS, wRAA per user-spec display."""
    h, _ = _team_pbp_full(team_name, sport, division)
    if h.empty or 'playerName' not in h.columns:
        return []
    for c in ['ab','h','doubles','triples','hr','rbi','bb','hbp','sf','sh','tb']:
        if c in h.columns: h[c] = pd.to_numeric(h[c], errors='coerce').fillna(0)
    grp = h.groupby('playerName').agg(
        AB=('ab','sum'), H=('h','sum'),
        DBL=('doubles','sum'), TRP=('triples','sum'),
        HR=('hr','sum'), RBI=('rbi','sum'),
        BB=('bb','sum'), HBP=('hbp','sum'),
        SF=('sf','sum'), SH=('sh','sum'),
    ).reset_index()
    # Plate appearances = AB + BB + HBP + SF + SH (NCAA convention)
    grp['PA'] = grp['AB'] + grp['BB'] + grp['HBP'] + grp['SF'] + grp['SH']
    grp = grp[grp['PA'] >= 25]  # qualifying threshold
    ab_safe = grp['AB'].replace(0, np.nan)
    obp_denom = (grp['AB'] + grp['BB'] + grp['HBP'] + grp['SF']).replace(0, np.nan)
    grp['OBP'] = ((grp['H'] + grp['BB'] + grp['HBP']) / obp_denom).fillna(0)
    SINGLES = grp['H'] - grp['DBL'] - grp['TRP'] - grp['HR']
    TB = SINGLES + 2*grp['DBL'] + 3*grp['TRP'] + 4*grp['HR']
    grp['SLG'] = (TB / ab_safe).fillna(0)
    grp['OPS'] = grp['OBP'] + grp['SLG']
    # wOBA + wRAA
    woba_num = (0.690*grp['BB'] + 0.722*grp['HBP'] + 0.888*SINGLES
                + 1.271*grp['DBL'] + 1.616*grp['TRP'] + 2.101*grp['HR'])
    grp['wOBA'] = (woba_num / obp_denom).fillna(0)
    league_w = _league_woba(sport, division)
    grp['wRAA'] = ((grp['wOBA'] - league_w) / 1.6) * grp['PA']
    grp = grp.sort_values('PA', ascending=False).head(n)
    return [{'name': r['playerName'],
             'pa': int(r['PA']),
             'ops': float(r['OPS']) if pd.notna(r['OPS']) else 0,
             'wraa': float(r['wRAA']) if pd.notna(r['wRAA']) else 0}
            for _, r in grp.iterrows()]


# ── Compute all data needed for the graphic ────────────────────────────────
with st.spinner('Aggregating season + L25 data…'):
    survival = _simulate_regional_full(20000)
    perf_full = {t: _team_radar_perf(t, sport, division, last_n_games=None) for t in teams}
    perf_l25 = {t: _team_radar_perf(t, sport, division, last_n_games=25) for t in teams}
    radar_dist = _division_metric_distributions(sport, division)
    top_p = {t: _team_top_pitchers(t, sport, division, n=8) for t in teams}
    top_h = {t: _team_top_hitters(t, sport, division, n=9) for t in teams}


# ── Radar geometry helpers ─────────────────────────────────────────────────
# 9 axes, percentile-based normalization (Similar Entities style). Each axis
# is computed as the team's percentile rank within the full sport+division
# team set, so polygons compare apples-to-apples across sports/divisions.
RADAR_AXES = ['RUNS', 'OPS', 'OBP', 'H K/BB',
              'ERA', 'WHIP', 'P K/BB', 'K%', 'FLD']
# Map axis label → metric key from _team_radar_perf, plus inversion flag for
# lower-is-better metrics (ERA, WHIP, hitting K/BB).
_AXIS_KEY = {
    'RUNS':   ('RUNS',   False),
    'OPS':    ('OPS',    False),
    'OBP':    ('OBP',    False),
    'H K/BB': ('HKBB',   True),   # lower K/BB = better discipline at the plate
    'ERA':    ('ERA',    True),
    'WHIP':   ('WHIP',   True),
    'P K/BB': ('PKBB',   False),  # higher K/BB = better stuff on the mound
    'K%':     ('KPCT',   False),
    'FLD':    ('FLD',    False),
}
_STRENGTH_LABELS = {
    'RUNS': 'RUNS/G', 'OPS': 'OPS', 'OBP': 'OBP', 'HKBB': 'HITTER K/BB',
    'ERA': 'TEAM ERA', 'WHIP': 'WHIP', 'PKBB': 'STAFF K/BB',
    'KPCT': 'K%', 'FLD': 'FIELD %',
}
_STRENGTH_FMT = {
    'RUNS': lambda v: f'{v:.1f}',
    'OPS':  lambda v: f'{v:.3f}'.lstrip('0') or '.000',
    'OBP':  lambda v: f'{v:.3f}'.lstrip('0') or '.000',
    'HKBB': lambda v: f'{v:.2f}',
    'ERA':  lambda v: f'{v:.2f}',
    'WHIP': lambda v: f'{v:.2f}',
    'PKBB': lambda v: f'{v:.2f}',
    'KPCT': lambda v: f'{v*100:.1f}%',
    'FLD':  lambda v: f'{v:.3f}'.lstrip('0') or '.000',
}


def _radar_pts(cx, cy, r, perf, frac_scale=1.0, dist=None):
    pts = []
    n = len(RADAR_AXES)
    for i, axis in enumerate(RADAR_AXES):
        angle = -math.pi / 2 + i * 2 * math.pi / n
        key, lower_better = _AXIS_KEY[axis]
        val = perf.get(key, 0) if perf else 0
        if dist is not None:
            frac = _percentile_rank(val, dist.get(key, []), lower_better=lower_better)
        else:
            frac = 0.5
        frac = max(0.0, min(1.0, frac)) * frac_scale
        x = cx + math.cos(angle) * r * frac
        y = cy + math.sin(angle) * r * frac
        pts.append(f'{x:.2f},{y:.2f}')
    return ' '.join(pts)


def _radar_grid_pts(cx, cy, r, frac):
    pts = []
    n = len(RADAR_AXES)
    for i in range(n):
        angle = -math.pi / 2 + i * 2 * math.pi / n
        x = cx + math.cos(angle) * r * frac
        y = cy + math.sin(angle) * r * frac
        pts.append(f'{x:.2f},{y:.2f}')
    return ' '.join(pts)


def _radar_label_pt(cx, cy, r, i):
    angle = -math.pi / 2 + i * 2 * math.pi / len(RADAR_AXES)
    return cx + math.cos(angle) * r * 1.18, cy + math.sin(angle) * r * 1.18


def _strength_for(perf, dist=None):
    if not perf: return None, None
    if dist is None:
        return None, None
    ranked = []
    for axis_label, (key, lower_better) in _AXIS_KEY.items():
        v = perf.get(key, 0)
        pr = _percentile_rank(v, dist.get(key, []), lower_better=lower_better)
        ranked.append((key, pr))
    ranked.sort(key=lambda x: -x[1])
    top_k = ranked[0][0]
    return _STRENGTH_LABELS[top_k], _STRENGTH_FMT[top_k](perf.get(top_k, 0))


# ── Build the SVG (1080×1350 IG-feed canvas) ──────────────────────────────
VB_W, VB_H = 1080, 1350
PAD_X, PAD_TOP = 24, 16

# Vertical layout (y positions)
Y_HEADER = PAD_TOP                       # Header start
H_HEADER = 92
Y_STRIP  = Y_HEADER + H_HEADER           # Team identity strip
H_STRIP  = 84
Y_PROB   = Y_STRIP + H_STRIP             # Path to the Title
H_PROB   = 154
Y_RADAR  = Y_PROB + H_PROB               # Radar hero
H_RADAR  = 408
Y_PITCH  = Y_RADAR + H_RADAR             # Pitching depth
H_PITCH  = 254
Y_HIT    = Y_PITCH + H_PITCH             # Hitting depth
H_HIT    = 282
Y_FOOT   = Y_HIT + H_HIT                 # Footer

parts = [
    f'<svg viewBox="0 0 {VB_W} {VB_H}" width="{VB_W}" height="{VB_H}" '
    f'xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">',
    # font import (works in browsers; cairosvg falls back to system fonts)
    # CDATA-wrap so the unescaped `&` query separators in the @import URL
    # don't break cairosvg's XML parser.
    '<defs><style><![CDATA['
    '@import url("https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700;800&family=JetBrains+Mono:wght@500;600;700;800&display=swap");'
    '.in{font-family:Inter,system-ui,sans-serif}.mn{font-family:"JetBrains Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}'
    ']]></style></defs>',
    f'<rect x="0" y="0" width="{VB_W}" height="{VB_H}" fill="{BG_EGG}"/>',
]

# ── HEADER ────────────────────────────────────────────────────────────────
hx_l = PAD_X
hx_r = VB_W - PAD_X
parts.extend([
    # eyebrow
    f'<text x="{hx_l}" y="{Y_HEADER + 14}" class="in" font-size="10" font-weight="700" '
    f'fill="{BRAND_RED}" letter-spacing="2.2">REGIONAL PREVIEW · {sport.upper()}</text>',
    # title
    f'<text x="{hx_l}" y="{Y_HEADER + 56}" class="in" font-size="40" font-weight="800" '
    f'fill="{INK_900}" letter-spacing="-0.8">{_xe(regional_name)}</text>',
    # mono metaline
    f'<text x="{hx_l}" y="{Y_HEADER + 78}" class="mn" font-size="11" font-weight="600" '
    f'fill="{INK_500}" letter-spacing="0.22">'
    f'NCAA {division} {sport.upper()} · LOOKBACK {lookback_days}D'
    f'</text>',
    # Right side: 64A wordmark + REGIONAL / DOUBLE-ELIM caption.
    # Wordmark image replaces the text "64Analytics" so the page uses the
    # canonical brand mark (red+black variant from Graphics/Logo).
    _embed_png(BRAND_WORDMARK, hx_r - 130, Y_HEADER + 12, 130, 28),
    f'<text x="{hx_r}" y="{Y_HEADER + 56}" class="mn" font-size="10" font-weight="600" '
    f'fill="{INK_500}" text-anchor="end" letter-spacing="0.6">REGIONAL · 4-TEAM</text>',
    f'<text x="{hx_r}" y="{Y_HEADER + 74}" class="mn" font-size="10" font-weight="600" '
    f'fill="{INK_500}" text-anchor="end" letter-spacing="0.6">DOUBLE-ELIM</text>',
    # 2pt closing rule
    f'<line x1="{PAD_X}" y1="{Y_HEADER + H_HEADER - 2}" x2="{VB_W - PAD_X}" '
    f'y2="{Y_HEADER + H_HEADER - 2}" stroke="{INK_900}" stroke-width="2"/>',
])

# ── TEAM IDENTITY STRIP ───────────────────────────────────────────────────
strip_inner_w = VB_W - 2 * PAD_X
cell_w = strip_inner_w / 4
for i, (team, seed) in enumerate(zip(teams, seeds)):
    cell_x = PAD_X + i * cell_w
    tid = team_ids.get(team)
    accent = _accent_for_team(tid, seed)
    rpi_val = team_rpi[team]
    rpi_txt = f'RPI {int(float(rpi_val))}' if rpi_val and str(rpi_val).strip() else 'RPI —'
    # Team record (W-L) — derive from PBP wins/losses if available, else blank.
    rec_str = ''
    # accent bar
    parts.append(f'<rect x="{cell_x}" y="{Y_STRIP}" width="{cell_w}" height="3" fill="{accent}"/>')
    # cell separator
    if i > 0:
        parts.append(f'<line x1="{cell_x}" y1="{Y_STRIP + 12}" x2="{cell_x}" y2="{Y_STRIP + H_STRIP - 8}" '
                     f'stroke="{INK_RULE}" stroke-width="1"/>')
    # Team logo on a white portrait circle with team-color ring (matches the
    # eggshell/clipped pattern used on the Spray Charts and Share Graphic
    # pages so dark / transparent marks read correctly on light backgrounds).
    disk_cx = cell_x + 30
    disk_cy = Y_STRIP + 44
    logo_b64 = _team_logo_b64(tid)
    parts.append(f'<circle cx="{disk_cx}" cy="{disk_cy}" r="22" fill="{accent}"/>')
    parts.append(f'<circle cx="{disk_cx}" cy="{disk_cy}" r="19" fill="#FFFFFF"/>')
    if logo_b64:
        clip_id = f'rp_logo_clip_{i}'
        parts.append(f'<defs><clipPath id="{clip_id}">'
                     f'<circle cx="{disk_cx}" cy="{disk_cy}" r="19"/></clipPath></defs>')
        parts.append(f'<image href="{logo_b64}" xlink:href="{logo_b64}" '
                     f'x="{disk_cx - 17}" y="{disk_cy - 17}" width="34" height="34" '
                     f'preserveAspectRatio="xMidYMid meet" clip-path="url(#{clip_id})"/>')
    else:
        # Fallback: monogram if logo missing
        parts.append(f'<text x="{disk_cx}" y="{disk_cy + 7}" class="in" font-size="20" '
                     f'font-weight="800" fill="{accent}" text-anchor="middle">'
                     f'{_monogram_for(team)}</text>')
    # text block to the right
    tx = disk_cx + 32
    parts.append(f'<text x="{tx}" y="{Y_STRIP + 30}" class="mn" font-size="10" font-weight="700" '
                 f'fill="{accent}" letter-spacing="0.6">#{seed} SEED · {_xe(rpi_txt)}</text>')
    parts.append(f'<text x="{tx}" y="{Y_STRIP + 52}" class="in" font-size="20" font-weight="800" '
                 f'fill="{INK_900}" letter-spacing="-0.2">{_xe(team)}</text>')
    parts.append(f'<text x="{tx}" y="{Y_STRIP + 70}" class="mn" font-size="9" font-weight="600" '
                 f'fill="{INK_500}" letter-spacing="0.4">{_xe(rec_str)}{("· " if rec_str else "")}'
                 f'{_xe(team).upper()}</text>')
parts.append(f'<line x1="{PAD_X}" y1="{Y_STRIP + H_STRIP}" x2="{VB_W - PAD_X}" '
             f'y2="{Y_STRIP + H_STRIP}" stroke="{INK_RULE}" stroke-width="1"/>')

# ── PATH TO THE TITLE ─────────────────────────────────────────────────────
PROB_X = PAD_X + 4
PROB_Y_HEAD = Y_PROB + 18
PROB_Y_GRID = Y_PROB + 38
parts.extend([
    f'<text x="{PROB_X}" y="{PROB_Y_HEAD}" class="in" font-size="11" font-weight="700" '
    f'fill="{INK_900}" letter-spacing="2.0">PATH TO THE TITLE</text>',
    f'<text x="{VB_W - PAD_X - 4}" y="{PROB_Y_HEAD}" class="mn" font-size="10" font-weight="600" '
    f'fill="{INK_500}" text-anchor="end" letter-spacing="0.4">'
    f'20,000 BRADLEY-TERRY MONTE CARLO · SUMS TO 100%</text>',
])

# Grid columns: Team(112) | Survive(1fr) | W-Bracket Live(1fr) | Reach Final(1fr) | Title Game(1fr) | Champ %(84)
team_col_w = 116
champ_col_w = 86
stage_w = (VB_W - 2 * PAD_X - 8 - team_col_w - champ_col_w) / 4
prob_grid_x = PAD_X + 4
col_x = [
    prob_grid_x,
    prob_grid_x + team_col_w,
    prob_grid_x + team_col_w + stage_w,
    prob_grid_x + team_col_w + 2 * stage_w,
    prob_grid_x + team_col_w + 3 * stage_w,
    prob_grid_x + team_col_w + 4 * stage_w,
]
# Column header row
hdr_y = PROB_Y_GRID
for label, x_l, x_r, anchor, color in [
    ('TEAM', col_x[0], col_x[1], 'start', INK_400),
    ('SURVIVE OPENER', col_x[1], col_x[2], 'middle', INK_400),
    ('W-BRACKET LIVE', col_x[2], col_x[3], 'middle', INK_400),
    ('REACH FINAL', col_x[3], col_x[4], 'middle', INK_400),
    ('TITLE GAME', col_x[4], col_x[5], 'middle', INK_400),
    ('CHAMP %', col_x[5], VB_W - PAD_X - 4, 'end', BRAND_RED),
]:
    if anchor == 'start':
        tx = x_l
    elif anchor == 'middle':
        tx = (x_l + x_r) / 2
    else:
        tx = x_r
    parts.append(f'<text x="{tx}" y="{hdr_y}" class="in" font-size="9" font-weight="700" '
                 f'fill="{color}" letter-spacing="1.0" text-anchor="{anchor}">{label}</text>')
parts.append(f'<line x1="{prob_grid_x}" y1="{hdr_y + 6}" x2="{VB_W - PAD_X - 4}" y2="{hdr_y + 6}" '
             f'stroke="{INK_RULE}" stroke-width="1"/>')

# Data rows
ROW_PITCH = 22
for ti, (team, seed) in enumerate(zip(teams, seeds)):
    accent = _accent_for_team(team_ids.get(team), seed)
    s = survival[team]
    row_cy = hdr_y + 14 + ti * ROW_PITCH + ROW_PITCH/2
    # team col: small seed badge + team short name
    badge_x = col_x[0]
    badge_y = row_cy - 9
    parts.append(f'<rect x="{badge_x}" y="{badge_y}" width="18" height="18" rx="3" fill="{accent}"/>')
    parts.append(f'<text x="{badge_x + 9}" y="{badge_y + 13}" class="in" font-size="10" font-weight="800" '
                 f'fill="#FFFFFF" text-anchor="middle">{seed}</text>')
    parts.append(f'<text x="{badge_x + 26}" y="{row_cy + 3}" class="in" font-size="12" font-weight="700" '
                 f'fill="{INK_900}">{_xe(team)}</text>')
    # 4 stage bars
    pct_vals = [s['r1'], s['r2'], s['r3'], s['r4']]
    bar_h = 14
    for si, p_val in enumerate(pct_vals):
        bx_l = col_x[1 + si] + 4
        bx_r = col_x[2 + si] - 4
        bw = bx_r - bx_l
        bar_y = row_cy - bar_h / 2
        # track
        parts.append(f'<rect x="{bx_l}" y="{bar_y}" width="{bw}" height="{bar_h}" rx="2" '
                     f'fill="{INK_100}"/>')
        # fill
        fill_w = bw * min(1.0, max(0.0, p_val))
        parts.append(f'<rect x="{bx_l}" y="{bar_y}" width="{fill_w:.2f}" height="{bar_h}" rx="2" '
                     f'fill="{accent}" fill-opacity="0.92"/>')
        # Label sits at bar center. When fill covers the center (pct>=50)
        # we'd be drawing dark text on a dark team color and lose contrast —
        # switch to white in that case so the percentile is always readable.
        pct_int = round(p_val * 100)
        if pct_int >= 50:
            label_color = '#FFFFFF'; label_weight = 700
        elif pct_int >= 25:
            label_color = INK_900;   label_weight = 700
        else:
            label_color = INK_700;   label_weight = 600
        parts.append(f'<text x="{(bx_l + bx_r)/2:.2f}" y="{row_cy + 3}" class="mn" font-size="10" '
                     f'font-weight="{label_weight}" fill="{label_color}" text-anchor="middle">{pct_int}%</text>')
    # champ %
    champ_pct = s['r4'] * 100
    parts.append(f'<text x="{VB_W - PAD_X - 6}" y="{row_cy + 5}" class="in" font-size="16" font-weight="800" '
                 f'fill="{BRAND_RED}" text-anchor="end" letter-spacing="-0.2">'
                 f'{champ_pct:.1f}<tspan font-size="10">%</tspan></text>')
parts.append(f'<line x1="{PAD_X}" y1="{Y_PROB + H_PROB}" x2="{VB_W - PAD_X}" y2="{Y_PROB + H_PROB}" '
             f'stroke="{INK_RULE}" stroke-width="1"/>')

# ── RADAR HERO ────────────────────────────────────────────────────────────
RH_X = PAD_X + 4
RH_HEAD_Y = Y_RADAR + 18
parts.extend([
    f'<text x="{RH_X}" y="{RH_HEAD_Y}" class="in" font-size="11" font-weight="700" '
    f'fill="{INK_900}" letter-spacing="2.0">TEAM PERFORMANCE PROFILE</text>',
    f'<text x="{VB_W - PAD_X - 4}" y="{RH_HEAD_Y}" class="mn" font-size="9" font-weight="600" '
    f'fill="{INK_500}" text-anchor="end" letter-spacing="1.0">'
    f'RUNS · OPS · OBP · K/BB · ERA · WHIP · K% · FLD% · DIVISION %ILES</text>',
    # Inline dashed-line marker for the L25 legend
    f'<line x1="{VB_W - PAD_X - 70}" y1="{RH_HEAD_Y - 3}" x2="{VB_W - PAD_X - 56}" y2="{RH_HEAD_Y - 3}" '
    f'stroke="{INK_700}" stroke-width="1.5" stroke-dasharray="2.5 2"/>',
    f'<text x="{VB_W - PAD_X - 4}" y="{RH_HEAD_Y + 12}" class="mn" font-size="9" font-weight="600" '
    f'fill="{INK_500}" text-anchor="end" letter-spacing="1.0">LAST 25</text>',
])

# Radar geometry
RH_BODY_Y = Y_RADAR + 36                  # body (radars + tiles) start
CENTER_R = 138                            # center radar half-size = ~280px diam
CENTER_CX = VB_W / 2
CENTER_CY = RH_BODY_Y + CENTER_R + 18
MINI_R = 64                               # mini radar half-size = 128px
TILE_GAP_Y = 18
# Left/right tile column geometry
side_col_w = 200
left_col_x = PAD_X + 12
right_col_x = VB_W - PAD_X - 12 - side_col_w

# Center radar — concentric grid + spokes + 4 team polygons.
# Subtle emblem watermark sits at the geometric center, below the grid so
# the polygons + axis labels read on top of it.
parts.append('<g>')
parts.append(_embed_png(BRAND_EMBLEM,
                        CENTER_CX - 36, CENTER_CY - 36, 72, 72, opacity=0.08))
for f_lev in (0.25, 0.5, 0.75, 1.0):
    parts.append(f'<polygon points="{_radar_grid_pts(CENTER_CX, CENTER_CY, CENTER_R, f_lev)}" '
                 f'fill="{"rgba(15,27,45,0.025)" if f_lev == 1.0 else "none"}" '
                 f'stroke="{INK_RULE}" stroke-width="{1 if f_lev == 1.0 else 0.6}"/>')
for i in range(len(RADAR_AXES)):
    angle = -math.pi / 2 + i * 2 * math.pi / len(RADAR_AXES)
    x_end = CENTER_CX + math.cos(angle) * CENTER_R
    y_end = CENTER_CY + math.sin(angle) * CENTER_R
    parts.append(f'<line x1="{CENTER_CX}" y1="{CENTER_CY}" x2="{x_end:.2f}" y2="{y_end:.2f}" '
                 f'stroke="{INK_RULE}" stroke-width="0.6"/>')
# Tick labels at 25/50/75/100 along axis 0 (top)
for f_lev in (0.25, 0.5, 0.75, 1.0):
    angle = -math.pi / 2
    x = CENTER_CX + math.cos(angle) * CENTER_R * f_lev
    y = CENTER_CY + math.sin(angle) * CENTER_R * f_lev
    parts.append(f'<text x="{x + 4:.2f}" y="{y - 2:.2f}" class="mn" font-size="7" font-weight="600" '
                 f'fill="{INK_300}" letter-spacing="0.4">{int(f_lev*100)}</text>')
# Each team polygon
for i, (team, seed) in enumerate(zip(teams, seeds)):
    perf = perf_full.get(team)
    if perf is None: continue
    accent = _accent_for_team(team_ids.get(team), seed)
    pts = _radar_pts(CENTER_CX, CENTER_CY, CENTER_R, perf, dist=radar_dist)
    parts.append(f'<polygon points="{pts}" fill="{accent}" fill-opacity="0.14" '
                 f'stroke="{accent}" stroke-width="2" stroke-linejoin="round"/>')
    # corner dots
    for pt in pts.split(' '):
        x_s, y_s = pt.split(',')
        parts.append(f'<circle cx="{x_s}" cy="{y_s}" r="3" fill="{accent}" stroke="#FFFFFF" stroke-width="0.5"/>')
# Axis labels
for i, axis in enumerate(RADAR_AXES):
    lx, ly = _radar_label_pt(CENTER_CX, CENTER_CY, CENTER_R, i)
    parts.append(f'<text x="{lx:.2f}" y="{ly:.2f}" class="mn" font-size="11" font-weight="800" '
                 f'fill="{INK_700}" text-anchor="middle" dominant-baseline="middle" '
                 f'letter-spacing="1.0">{axis}</text>')
parts.append('</g>')

# Corner tiles — left has seeds 1 and 3, right has 2 and 4
def _draw_tile(team, seed, side, tile_top_y):
    accent = _accent_for_team(team_ids.get(team), seed)
    perf = perf_full.get(team)
    perf25 = perf_l25.get(team)
    s_lab, s_val = _strength_for(perf, dist=radar_dist)
    out = []
    # mini radar position
    if side == 'left':
        mini_cx = left_col_x + side_col_w - MINI_R - 4
    else:
        mini_cx = right_col_x + MINI_R + 4
    mini_cy = tile_top_y + MINI_R + 12
    # info block position
    if side == 'left':
        info_x = left_col_x + 4
        text_anchor = 'start'
    else:
        info_x = right_col_x + side_col_w - 4
        text_anchor = 'end'

    # info text
    seed_box_size = 16
    if side == 'left':
        seed_x = info_x
        team_x = info_x + seed_box_size + 6
    else:
        seed_x = info_x - seed_box_size
        team_x = info_x - seed_box_size - 6
    out.append(f'<rect x="{seed_x}" y="{tile_top_y + 14}" width="{seed_box_size}" height="{seed_box_size}" '
               f'rx="3" fill="{accent}"/>')
    out.append(f'<text x="{seed_x + seed_box_size/2}" y="{tile_top_y + 26}" class="in" font-size="10" '
               f'font-weight="800" fill="#FFFFFF" text-anchor="middle">{seed}</text>')
    out.append(f'<text x="{team_x}" y="{tile_top_y + 26}" class="mn" font-size="10" font-weight="800" '
               f'fill="{accent}" letter-spacing="0.6" text-anchor="{"start" if side == "left" else "end"}">'
               f'{_xe(team).upper()}</text>')
    # (STRENGTH callout removed per user feedback — the team's top axis was
    # surfacing in the corner tile but cluttered the radar layout. The
    # underlying _strength_for helper stays in case it gets reused later.)

    # mini radar grid + spokes
    for f_lev in (0.33, 0.66, 1.0):
        out.append(f'<polygon points="{_radar_grid_pts(mini_cx, mini_cy, MINI_R, f_lev)}" '
                   f'fill="none" stroke="{INK_RULE}" stroke-width="0.6"/>')
    for i in range(len(RADAR_AXES)):
        angle = -math.pi / 2 + i * 2 * math.pi / len(RADAR_AXES)
        x_end = mini_cx + math.cos(angle) * MINI_R
        y_end = mini_cy + math.sin(angle) * MINI_R
        out.append(f'<line x1="{mini_cx}" y1="{mini_cy}" x2="{x_end:.2f}" y2="{y_end:.2f}" '
                   f'stroke="{INK_RULE}" stroke-width="0.6"/>')
    # season fill
    if perf:
        season_pts = _radar_pts(mini_cx, mini_cy, MINI_R, perf, dist=radar_dist)
        out.append(f'<polygon points="{season_pts}" fill="{accent}" fill-opacity="0.32" '
                   f'stroke="{accent}" stroke-width="1.5" stroke-linejoin="round"/>')
        # corner dots on season
        for pt in season_pts.split(' '):
            x_s, y_s = pt.split(',')
            out.append(f'<circle cx="{x_s}" cy="{y_s}" r="2" fill="{accent}" stroke="#FFFFFF" stroke-width="0.4"/>')
    # L25 dashed overlay
    if perf25:
        l25_pts = _radar_pts(mini_cx, mini_cy, MINI_R, perf25, dist=radar_dist)
        out.append(f'<polygon points="{l25_pts}" fill="none" stroke="{accent}" stroke-width="1.25" '
                   f'stroke-dasharray="2.5 2" stroke-linejoin="round"/>')
    # mini radar axis labels
    for i, axis in enumerate(RADAR_AXES):
        lx, ly = _radar_label_pt(mini_cx, mini_cy, MINI_R, i)
        out.append(f'<text x="{lx:.2f}" y="{ly:.2f}" class="mn" font-size="7" font-weight="700" '
                   f'fill="{INK_400}" text-anchor="middle" dominant-baseline="middle" '
                   f'letter-spacing="0.6">{axis}</text>')
    return out

# Two tiles per side, vertically stacked
TILE_H = 168
# left col: seeds 0 and 2 (=#1 and #3 by zero-index)
left_top_a = RH_BODY_Y + 4
left_top_b = left_top_a + TILE_H + TILE_GAP_Y
right_top_a = left_top_a
right_top_b = left_top_b
parts.extend(_draw_tile(teams[0], seeds[0], 'left',  left_top_a))
parts.extend(_draw_tile(teams[2], seeds[2], 'left',  left_top_b))
parts.extend(_draw_tile(teams[1], seeds[1], 'right', right_top_a))
parts.extend(_draw_tile(teams[3], seeds[3], 'right', right_top_b))

parts.append(f'<line x1="{PAD_X}" y1="{Y_RADAR + H_RADAR}" x2="{VB_W - PAD_X}" y2="{Y_RADAR + H_RADAR}" '
             f'stroke="{INK_RULE}" stroke-width="1"/>')

# ── PITCHING DEPTH ────────────────────────────────────────────────────────
PD_X = PAD_X + 4
PD_HEAD_Y = Y_PITCH + 18
parts.extend([
    f'<text x="{PD_X}" y="{PD_HEAD_Y}" class="in" font-size="11" font-weight="700" '
    f'fill="{INK_900}" letter-spacing="2.0">PITCHING STAFF · TOP 8 BY IP</text>',
    f'<text x="{VB_W - PAD_X - 4}" y="{PD_HEAD_Y}" class="mn" font-size="10" font-weight="600" '
    f'fill="{INK_500}" text-anchor="end" letter-spacing="0.4">'
    f'WEEKEND STARTERS HIGHLIGHTED · IP / FIP / WHIP</text>',
])
pd_inner_w = VB_W - 2 * PAD_X - 8
pd_col_w = pd_inner_w / 4
pd_col_x = [PAD_X + 4 + i * pd_col_w for i in range(4)]
PD_BODY_Y = Y_PITCH + 32
for ci, (team, seed) in enumerate(zip(teams, seeds)):
    accent = _accent_for_team(team_ids.get(team), seed)
    cx_l = pd_col_x[ci]
    cx_r = cx_l + pd_col_w - 4
    # column separator
    if ci > 0:
        parts.append(f'<line x1="{cx_l}" y1="{PD_BODY_Y - 8}" x2="{cx_l}" y2="{Y_PITCH + H_PITCH - 8}" '
                     f'stroke="{INK_RULE}" stroke-width="1"/>')
    # team header inside column
    th_x = cx_l + 8
    parts.append(f'<rect x="{th_x}" y="{PD_BODY_Y + 2}" width="8" height="8" fill="{accent}"/>')
    parts.append(f'<text x="{th_x + 14}" y="{PD_BODY_Y + 10}" class="mn" font-size="10" font-weight="700" '
                 f'fill="{INK_700}" letter-spacing="0.6">{_xe(team).upper()}</text>')
    parts.append(f'<line x1="{th_x}" y1="{PD_BODY_Y + 18}" x2="{cx_r - 4}" y2="{PD_BODY_Y + 18}" '
                 f'stroke="{INK_RULE}" stroke-width="1"/>')
    # column headers — three stat cols (IP / FIP / WHIP)
    pd_ip_x   = cx_r - 4 - 70
    pd_fip_x  = cx_r - 4 - 35
    pd_whip_x = cx_r - 4
    parts.append(f'<text x="{pd_ip_x}" y="{PD_BODY_Y + 32}" class="mn" font-size="8" '
                 f'font-weight="700" fill="{INK_400}" text-anchor="end" letter-spacing="0.6">IP</text>')
    parts.append(f'<text x="{pd_fip_x}" y="{PD_BODY_Y + 32}" class="mn" font-size="8" '
                 f'font-weight="700" fill="{INK_400}" text-anchor="end" letter-spacing="0.6">FIP</text>')
    parts.append(f'<text x="{pd_whip_x}" y="{PD_BODY_Y + 32}" class="mn" font-size="8" '
                 f'font-weight="700" fill="{INK_400}" text-anchor="end" letter-spacing="0.6">WHIP</text>')
    pitchers = top_p.get(team, [])
    for pi, p in enumerate(pitchers[:8]):
        is_top3 = pi < 3
        row_y = PD_BODY_Y + 48 + pi * 20
        text_color = accent if is_top3 else INK_700
        text_weight = 800 if is_top3 else 600
        role_color = accent if is_top3 else INK_400
        parts.append(f'<text x="{th_x}" y="{row_y}" class="mn" font-size="8" font-weight="700" '
                     f'fill="{role_color}" letter-spacing="0.6">{p["role"]}</text>')
        parts.append(f'<text x="{th_x + 26}" y="{row_y}" class="in" font-size="11" font-weight="{text_weight}" '
                     f'fill="{text_color}">{_xe(p["name"])[:18]}</text>')
        parts.append(f'<text x="{pd_ip_x}" y="{row_y}" class="mn" font-size="10" '
                     f'font-weight="{text_weight}" fill="{text_color}" text-anchor="end">{p["ip"]:.1f}</text>')
        parts.append(f'<text x="{pd_fip_x}" y="{row_y}" class="mn" font-size="10" '
                     f'font-weight="{text_weight}" fill="{text_color}" text-anchor="end">{p["fip"]:.2f}</text>')
        parts.append(f'<text x="{pd_whip_x}" y="{row_y}" class="mn" font-size="10" '
                     f'font-weight="{text_weight}" fill="{text_color}" text-anchor="end">{p["whip"]:.2f}</text>')
        if pi < 7:
            parts.append(f'<line x1="{th_x}" y1="{row_y + 4}" x2="{cx_r - 4}" y2="{row_y + 4}" '
                         f'stroke="{INK_RULE}" stroke-width="0.5" stroke-dasharray="2 2"/>')
parts.append(f'<line x1="{PAD_X}" y1="{Y_PITCH + H_PITCH}" x2="{VB_W - PAD_X}" y2="{Y_PITCH + H_PITCH}" '
             f'stroke="{INK_RULE}" stroke-width="1"/>')

# ── HITTING DEPTH ─────────────────────────────────────────────────────────
HD_X = PAD_X + 4
HD_HEAD_Y = Y_HIT + 18
parts.extend([
    f'<text x="{HD_X}" y="{HD_HEAD_Y}" class="in" font-size="11" font-weight="700" '
    f'fill="{INK_900}" letter-spacing="2.0">HITTING ORDER · TOP 9 BY PA</text>',
    f'<text x="{VB_W - PAD_X - 4}" y="{HD_HEAD_Y}" class="mn" font-size="10" font-weight="600" '
    f'fill="{INK_500}" text-anchor="end" letter-spacing="0.4">'
    f'PA / OPS / wRAA</text>',
])
HD_BODY_Y = Y_HIT + 32
hd_col_w = pd_inner_w / 4
hd_col_x = [PAD_X + 4 + i * hd_col_w for i in range(4)]
for ci, (team, seed) in enumerate(zip(teams, seeds)):
    accent = _accent_for_team(team_ids.get(team), seed)
    cx_l = hd_col_x[ci]
    cx_r = cx_l + hd_col_w - 4
    if ci > 0:
        parts.append(f'<line x1="{cx_l}" y1="{HD_BODY_Y}" x2="{cx_l}" y2="{Y_HIT + H_HIT - 4}" '
                     f'stroke="{INK_RULE}" stroke-width="1"/>')
    # accent bar at top of column
    parts.append(f'<rect x="{cx_l + 4}" y="{HD_BODY_Y}" width="{hd_col_w - 8}" height="3" fill="{accent}"/>')
    th_x = cx_l + 8
    parts.append(f'<rect x="{th_x}" y="{HD_BODY_Y + 12}" width="8" height="8" fill="{accent}"/>')
    parts.append(f'<text x="{th_x + 14}" y="{HD_BODY_Y + 20}" class="mn" font-size="10" font-weight="700" '
                 f'fill="{INK_700}" letter-spacing="0.6">{_xe(team).upper()}</text>')
    parts.append(f'<line x1="{th_x}" y1="{HD_BODY_Y + 28}" x2="{cx_r - 4}" y2="{HD_BODY_Y + 28}" '
                 f'stroke="{INK_RULE}" stroke-width="1"/>')
    # column header row — PA / OPS / wRAA
    col_pa_x   = cx_r - 4 - 76
    col_ops_x  = cx_r - 4 - 38
    col_wraa_x = cx_r - 4
    parts.append(f'<text x="{col_pa_x}" y="{HD_BODY_Y + 42}" class="mn" font-size="8" '
                 f'font-weight="700" fill="{INK_400}" text-anchor="end" letter-spacing="0.6">PA</text>')
    parts.append(f'<text x="{col_ops_x}" y="{HD_BODY_Y + 42}" class="mn" font-size="8" '
                 f'font-weight="700" fill="{INK_400}" text-anchor="end" letter-spacing="0.6">OPS</text>')
    parts.append(f'<text x="{col_wraa_x}" y="{HD_BODY_Y + 42}" class="mn" font-size="8" '
                 f'font-weight="700" fill="{INK_400}" text-anchor="end" letter-spacing="0.6">wRAA</text>')
    hitters = top_h.get(team, [])
    for hi, h in enumerate(hitters[:9]):
        # User asked for no per-row highlighting in the hitting order — every
        # row gets the same neutral weight + ink color.
        row_y = HD_BODY_Y + 60 + hi * 20
        text_color = INK_700
        text_weight = 600
        ops_str  = f'{h["ops"]:.3f}'.lstrip('0') if h['ops'] > 0 else '.000'
        wraa_str = f'{h["wraa"]:+.1f}'  # +/- signed, one decimal
        parts.append(f'<text x="{th_x}" y="{row_y}" class="in" font-size="11" font-weight="{text_weight}" '
                     f'fill="{text_color}">{_xe(h["name"])[:18]}</text>')
        parts.append(f'<text x="{col_pa_x}" y="{row_y}" class="mn" font-size="10" '
                     f'font-weight="{text_weight}" fill="{text_color}" text-anchor="end">{h["pa"]}</text>')
        parts.append(f'<text x="{col_ops_x}" y="{row_y}" class="mn" font-size="10" '
                     f'font-weight="{text_weight}" fill="{text_color}" text-anchor="end">{ops_str}</text>')
        parts.append(f'<text x="{col_wraa_x}" y="{row_y}" class="mn" font-size="10" '
                     f'font-weight="{text_weight}" fill="{text_color}" text-anchor="end">{wraa_str}</text>')
        if hi < 8:
            parts.append(f'<line x1="{th_x}" y1="{row_y + 4}" x2="{cx_r - 4}" y2="{row_y + 4}" '
                         f'stroke="{INK_RULE}" stroke-width="0.5" stroke-dasharray="2 2"/>')

# ── FOOTER ────────────────────────────────────────────────────────────────
foot_y = Y_FOOT + 4
parts.append(f'<line x1="{PAD_X}" y1="{foot_y}" x2="{VB_W - PAD_X}" y2="{foot_y}" '
             f'stroke="{INK_900}" stroke-width="2"/>')
foot_text_y = foot_y + 18
parts.append(f'<rect x="{PAD_X}" y="{foot_text_y - 11}" width="62" height="14" rx="2" fill="{INK_900}"/>')
parts.append(f'<text x="{PAD_X + 31}" y="{foot_text_y - 1}" class="mn" font-size="9" font-weight="700" '
             f'fill="{BG_EGG}" text-anchor="middle" letter-spacing="1.0">METHOD</text>')
parts.append(f'<text x="{PAD_X + 70}" y="{foot_text_y}" class="mn" font-size="9" font-weight="600" '
             f'fill="{INK_500}" letter-spacing="0.6">BRADLEY-TERRY · 20K SIM · FULL SEASON · LAST 25 OVERLAY</text>')
parts.append(f'<text x="{VB_W - PAD_X}" y="{foot_text_y}" class="mn" font-size="9" font-weight="800" '
             f'fill="{BRAND_RED}" text-anchor="end" letter-spacing="0.6">64ANALYTICS.COM</text>')
# Brand circle next to the URL — small, uses the red+black circle logo.
parts.append(_embed_png(BRAND_CIRCLE_RB, VB_W - PAD_X - 110, foot_text_y - 11, 16, 16))

parts.append('</svg>')
graphic_svg = ''.join(parts)

# Render in page
display_svg = graphic_svg.replace(
    '<svg ',
    '<svg style="width:100%;max-width:1080px;height:auto;display:block;'
    'margin:0 auto;border-radius:8px;box-shadow:0 12px 36px rgba(0,0,0,.18);" ', 1,
)
st.markdown(display_svg, unsafe_allow_html=True)

# PNG download for the main page
try:
    import cairosvg
    png_bytes = cairosvg.svg2png(bytestring=graphic_svg.encode('utf-8'), output_width=2160)
    safe_name = ''.join(c if c.isalnum() else '_' for c in str(regional_name))[:40]
    fname = f'regional_preview_{sport}_{division}_{safe_name}.png'
    st.download_button('Download PNG (2160w)', data=png_bytes, file_name=fname,
                       mime='image/png', use_container_width=False)
except Exception as e:
    st.caption(f'PNG export unavailable in this environment ({type(e).__name__}: {str(e)[:80]}).')


st.markdown('---')
st.caption('1080×1350 IG-feed graphic. Sports Reference / FiveThirtyEight density: '
           'eggshell over navy, Bradley-Terry round-by-round survival, hero radar with '
           'L25 dotted overlay, season-long top-8 pitchers / top-9 hitters per team.')
