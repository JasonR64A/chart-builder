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
st.caption('v1 — RPI-driven Bradley-Terry sim, 21d hot lists, weekend-cadence starter detection. '
           'Stats pulled from chart-builder PBP box scores. Iterate on inputs and visualization to taste.')
