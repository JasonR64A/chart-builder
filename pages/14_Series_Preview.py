"""
Weekend Series Preview — head-to-head matchup breakdown for upcoming series.

Combines: team rankings (64A/RPI/Massey/DSR), spider chart comparison,
game-by-game predicted WP with starter matchups, bullpen depth, and
"who's hot" last-14-day performers.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta

from pages._win_prob_model import (
    build_team_profiles, adjusted_team_pct, pre_game_wp,
    blend_with_static, build_rank_pct_map, TEAM_RANK_BLEND,
)

_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _APP_DIR / 'data'
PBP_DIR = _APP_DIR / 'pbp_data'

DIVISION_LABELS = {'D1': 'D-I', 'D2': 'D-II', 'D3': 'D-III'}

# 64A brand colors
BG_COLOR = '#FAF8F2'
CARD_RED = '#C41230'
NAVY = '#29335c'
TEXT_COLOR = '#2D2926'

st.set_page_config(page_title='Series Preview', layout='wide')
st.title('Weekend Series Preview')


# ── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_teams_data():
    t = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False).fillna('')
    c = pd.read_csv(DATA_DIR / 'conferences.csv', low_memory=False).fillna('')
    t['id'] = pd.to_numeric(t['id'], errors='coerce').fillna(0).astype(int)
    c_map = dict(zip(c['id'], c['name']))
    c_div = dict(zip(c['id'], c['division']))
    t['conference_name'] = t['conference_id'].map(c_map)
    t['division'] = t['conference_id'].map(c_div)
    return t, c


@st.cache_data
def load_team_ranks(sport_label, div_label):
    tr = pd.read_csv(DATA_DIR / 'team_rank.csv', low_memory=False)
    tr26 = tr[tr['year'] == 2026].copy()
    tr26['rank_64a'] = pd.to_numeric(tr26['integer_64_rank_total'], errors='coerce')
    return tr26


@st.cache_data
def load_external_ranks(sport):
    sport_l = sport.lower()
    ranks = {}
    # RPI
    rpi_f = DATA_DIR / f'{sport_l}_rpi_D1.csv'
    if rpi_f.exists():
        r = pd.read_csv(rpi_f, low_memory=False)
        ranks['rpi'] = dict(zip(r['teamName'], pd.to_numeric(r['rank'], errors='coerce')))
    # Name map
    nm_f = DATA_DIR / 'rankings' / 'name_map.csv'
    ext_map = {}
    if nm_f.exists():
        nm = pd.read_csv(nm_f)
        ext_map = dict(zip(nm['external_name'], nm['our_name']))
    # Massey
    m_f = DATA_DIR / 'rankings' / f'massey_{sport_l}.csv'
    if m_f.exists():
        m = pd.read_csv(m_f, low_memory=False)
        ranks['massey'] = {ext_map.get(t, t): r for t, r in zip(m['team'], m['rank'])}
    # DSR
    d_f = DATA_DIR / 'rankings' / f'dsr_{sport_l}.csv'
    if d_f.exists():
        d = pd.read_csv(d_f, low_memory=False)
        ranks['dsr'] = {ext_map.get(t, t): r for t, r in zip(d['team'], d['rank'])}
    return ranks


@st.cache_data
def load_records(sport):
    sport_l = sport.lower()
    sf = DATA_DIR / f'schedules_full_{sport_l}.csv'
    if not sf.exists():
        return {}
    s = pd.read_csv(sf, low_memory=False).fillna('')
    played = s[s['result'].notna() & (s['result'] != '')]
    records = {}
    for tn, grp in played.groupby('teamName'):
        w = int(grp['result'].str.startswith('W').sum())
        l = len(grp) - w
        # Conference record
        # We don't have a clean conf flag, so just report overall
        records[tn] = {'wins': w, 'losses': l}
    return records


@st.cache_data
def load_player_rank():
    pr = pd.read_csv(DATA_DIR / 'player_rank.csv', low_memory=False,
                     usecols=['player_id', 'team_id', 'year',
                              'percentile_rank_weighted_run_created_efficiency',
                              'percentile_rank_weighted_run_allowed_efficiency'])
    pr['year'] = pd.to_numeric(pr['year'], errors='coerce')
    pr = pr.sort_values('year').drop_duplicates('player_id', keep='last')
    pr['team_id'] = pd.to_numeric(pr['team_id'], errors='coerce').astype('Int64')
    pr['hit_pct'] = pd.to_numeric(pr['percentile_rank_weighted_run_created_efficiency'], errors='coerce')
    pr['pit_pct'] = pd.to_numeric(pr['percentile_rank_weighted_run_allowed_efficiency'], errors='coerce')
    return pr


@st.cache_data
def load_players():
    p = pd.read_csv(DATA_DIR / 'players.csv', low_memory=False, encoding='latin-1', dtype=str).fillna('')
    p['id_int'] = pd.to_numeric(p['id'], errors='coerce').astype('Int64')
    return p


@st.cache_data
def load_hitting_pbp(sport, division):
    sport_l = sport.lower()
    f = PBP_DIR / sport_l / f'hitting_pbp_{division}.csv'
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
    return df


@st.cache_data
def load_pitching_pbp(sport, division):
    sport_l = sport.lower()
    f = PBP_DIR / sport_l / f'pitching_pbp_{division}.csv'
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
    return df


def get_team_id(team_name, teams_df, sport_label):
    match = teams_df[(teams_df['name'] == team_name) & (teams_df['sport'] == sport_label)]
    if len(match):
        return int(match.iloc[0]['id'])
    return None


def get_starters(team_id, player_rank_df, players_df, n=3):
    """Top N pitchers by pitching percentile for a team."""
    team_pr = player_rank_df[player_rank_df['team_id'] == team_id].copy()
    pitchers = team_pr.dropna(subset=['pit_pct'])
    pitchers = pitchers[pitchers['pit_pct'] > 0].sort_values('pit_pct', ascending=False)
    result = []
    for _, r in pitchers.head(n).iterrows():
        pid = int(r['player_id'])
        p_row = players_df[players_df['id_int'] == pid]
        name = p_row.iloc[0]['player_name'] if len(p_row) else f'Player {pid}'
        pos = p_row.iloc[0].get('position', '') if len(p_row) else ''
        result.append({'player_id': pid, 'name': name, 'position': pos,
                       'pit_pct': float(r['pit_pct'])})
    return result


def get_bullpen(team_id, player_rank_df, players_df):
    """Pitchers 4-10 by pitching percentile."""
    team_pr = player_rank_df[player_rank_df['team_id'] == team_id].copy()
    pitchers = team_pr.dropna(subset=['pit_pct'])
    pitchers = pitchers[pitchers['pit_pct'] > 0].sort_values('pit_pct', ascending=False)
    bp = pitchers.iloc[3:10]
    if bp.empty:
        return {'mean_pct': 0, 'count': 0, 'names': []}
    names = []
    for _, r in bp.iterrows():
        p_row = players_df[players_df['id_int'] == int(r['player_id'])]
        n = p_row.iloc[0]['player_name'] if len(p_row) else '?'
        names.append(n)
    return {'mean_pct': float(bp['pit_pct'].mean()), 'count': len(bp), 'names': names}


def get_hot_hitters(team_name, hitting_pbp, n=3, days=14):
    """Top N hitters by games played in last `days` days for a team."""
    if hitting_pbp.empty:
        return []
    cutoff = hitting_pbp['date'].max() - pd.Timedelta(days=days)
    recent = hitting_pbp[(hitting_pbp['date'] >= cutoff) &
                          (hitting_pbp['teamName'] == team_name)]
    if recent.empty:
        return []
    # Group by player, compute stats
    stats = recent.groupby(['playerId', 'playerName']).agg(
        games=('gameId', 'nunique'),
        ab=('ab', 'sum'),
        h=('h', 'sum'),
        hr=('hr', 'sum'),
        rbi=('rbi', 'sum'),
        bb=('bb', 'sum'),
    ).reset_index()
    stats['ab'] = pd.to_numeric(stats['ab'], errors='coerce').fillna(0)
    stats['h'] = pd.to_numeric(stats['h'], errors='coerce').fillna(0)
    stats['hr'] = pd.to_numeric(stats['hr'], errors='coerce').fillna(0)
    stats['rbi'] = pd.to_numeric(stats['rbi'], errors='coerce').fillna(0)
    stats['bb'] = pd.to_numeric(stats['bb'], errors='coerce').fillna(0)
    stats = stats[stats['ab'] >= 10]  # minimum plate appearances
    if stats.empty:
        return []
    stats['avg'] = stats['h'] / stats['ab']
    stats['obp'] = (stats['h'] + stats['bb']) / (stats['ab'] + stats['bb'])
    stats = stats.sort_values('avg', ascending=False)
    result = []
    for _, r in stats.head(n).iterrows():
        result.append({
            'name': r['playerName'], 'games': int(r['games']),
            'avg': f".{int(r['avg']*1000):03d}" if r['avg'] < 1 else '1.000',
            'hr': int(r['hr']), 'rbi': int(r['rbi']),
            'obp': f".{int(r['obp']*1000):03d}" if r['obp'] < 1 else '1.000',
        })
    return result


def get_hot_pitcher(team_name, pitching_pbp, days=14):
    """Top pitcher by IP in last `days` days."""
    if pitching_pbp.empty:
        return None
    cutoff = pitching_pbp['date'].max() - pd.Timedelta(days=days)
    recent = pitching_pbp[(pitching_pbp['date'] >= cutoff) &
                           (pitching_pbp['teamName'] == team_name)]
    if recent.empty:
        return None
    stats = recent.groupby(['playerId', 'playerName']).agg(
        games=('gameId', 'nunique'),
        ip=('ip', 'sum'),
        h=('h', 'sum'),
        er=('er', 'sum'),
        so=('so', 'sum'),
        bb=('bb', 'sum'),
    ).reset_index()
    for c in ['ip', 'h', 'er', 'so', 'bb']:
        stats[c] = pd.to_numeric(stats[c], errors='coerce').fillna(0)
    stats = stats[stats['ip'] >= 3]
    if stats.empty:
        return None
    stats['era'] = (stats['er'] / stats['ip']) * 9 if stats['ip'].sum() > 0 else 99
    stats['era'] = stats.apply(lambda r: (r['er'] / r['ip']) * 9 if r['ip'] > 0 else 99, axis=1)
    best = stats.sort_values('era').iloc[0]
    return {
        'name': best['playerName'], 'games': int(best['games']),
        'ip': f"{best['ip']:.1f}", 'era': f"{best['era']:.2f}",
        'so': int(best['so']), 'bb': int(best['bb']),
    }


def build_spider_chart(team_a_name, team_b_name, team_a_id, team_b_id,
                        player_rank_df, hitting_pbp, pitching_pbp,
                        team_a_color, team_b_color):
    """Radar/spider chart comparing two teams on 6 dimensions."""
    categories = ['Hitting', 'Pitching', 'Power', 'Bullpen', 'Speed', 'Discipline']

    def team_metrics(tid, tname):
        pr = player_rank_df[player_rank_df['team_id'] == tid]
        hit = pr.dropna(subset=['hit_pct']); hit = hit[hit['hit_pct'] > 0]
        pit = pr.dropna(subset=['pit_pct']); pit = pit[pit['pit_pct'] > 0]
        hitting_val = float(hit['hit_pct'].mean()) if len(hit) else 0.5
        pitching_val = float(pit['pit_pct'].mean()) if len(pit) else 0.5
        # Bullpen: pitchers 4-10
        pit_sorted = pit.sort_values('pit_pct', ascending=False)
        bp = pit_sorted.iloc[3:10]
        bullpen_val = float(bp['pit_pct'].mean()) if len(bp) else 0.5

        # Recent hitting stats for power/speed/discipline
        recent = hitting_pbp[hitting_pbp['teamName'] == tname].copy() if not hitting_pbp.empty else pd.DataFrame()
        if not recent.empty and len(recent) > 0:
            for c in ['ab', 'hr', 'sb', 'bb', 'h']:
                if c in recent.columns:
                    recent[c] = pd.to_numeric(recent[c], errors='coerce').fillna(0)
            ab = recent['ab'].sum() if 'ab' in recent.columns else 1
            if ab > 0:
                power_val = min(1.0, (recent['hr'].sum() if 'hr' in recent.columns else 0) / ab * 20)
                speed_val = min(1.0, (recent['sb'].sum() if 'sb' in recent.columns else 0) / ab * 15)
                disc_val = min(1.0, (recent['bb'].sum() if 'bb' in recent.columns else 0) / ab * 5)
            else:
                power_val = speed_val = disc_val = 0.5
        else:
            power_val = speed_val = disc_val = 0.5

        return [hitting_val, pitching_val, power_val, bullpen_val, speed_val, disc_val]

    vals_a = team_metrics(team_a_id, team_a_name)
    vals_b = team_metrics(team_b_id, team_b_name)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_a + [vals_a[0]], theta=categories + [categories[0]],
        fill='toself', fillcolor=f'rgba({int(team_a_color[1:3],16)},{int(team_a_color[3:5],16)},{int(team_a_color[5:7],16)},0.15)',
        line=dict(color=team_a_color, width=2),
        name=team_a_name,
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_b + [vals_b[0]], theta=categories + [categories[0]],
        fill='toself', fillcolor=f'rgba({int(team_b_color[1:3],16)},{int(team_b_color[3:5],16)},{int(team_b_color[5:7],16)},0.15)',
        line=dict(color=team_b_color, width=2),
        name=team_b_name,
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=BG_COLOR,
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
        ),
        showlegend=True,
        legend=dict(orientation='h', y=-0.1),
        paper_bgcolor=BG_COLOR,
        margin=dict(l=60, r=60, t=30, b=60),
        height=380,
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown('### Settings')
sel_sport = st.sidebar.selectbox('Sport', ['Baseball', 'Softball'], key='sp_sport')
sel_div = st.sidebar.selectbox('Division', ['D1', 'D2', 'D3'], key='sp_div')

teams_df, confs_df = load_teams_data()
div_label = DIVISION_LABELS.get(sel_div, 'D-I')
div_ids = set(confs_df[confs_df['division'] == div_label]['id'])
sport_teams = teams_df[(teams_df['sport'] == sel_sport) & (teams_df['conference_id'].isin(div_ids))]
team_list = sorted(sport_teams['name'].tolist())

if not team_list:
    st.warning(f'No teams found for {sel_sport} {sel_div}.')
    st.stop()

st.sidebar.markdown('### Matchup')
team_a = st.sidebar.selectbox('Team A (away)', team_list, index=0, key='sp_team_a')
default_b = min(1, len(team_list) - 1)
team_b = st.sidebar.selectbox('Team B (home)', team_list, index=default_b, key='sp_team_b')

if team_a == team_b:
    st.warning('Pick two different teams.')
    st.stop()


# ── Load Data ────────────────────────────────────────────────────────────────
tr26 = load_team_ranks(sel_sport, div_label)
ext_ranks = load_external_ranks(sel_sport)
records = load_records(sel_sport)
pr = load_player_rank()
players = load_players()
profiles = build_team_profiles(sel_sport, sel_div)
hitting_pbp = load_hitting_pbp(sel_sport, sel_div)
pitching_pbp = load_pitching_pbp(sel_sport, sel_div)

team_a_id = get_team_id(team_a, teams_df, sel_sport)
team_b_id = get_team_id(team_b, teams_df, sel_sport)

# Build rank_pct map for WP computation
ranked_df = tr26[tr26['team_id'].isin(sport_teams['id'])].copy()
ranked_df = ranked_df.merge(sport_teams[['id', 'name']], left_on='team_id', right_on='id', how='inner')
ranked_df['rank'] = ranked_df['rank_64a']
rank_pct_map = build_rank_pct_map(ranked_df.rename(columns={'name': 'team_name'}), 'rank')


# ── Header: Rankings Comparison ──────────────────────────────────────────────
st.markdown(f'## {team_a}  vs  {team_b}')

rec_a = records.get(team_a, {'wins': 0, 'losses': 0})
rec_b = records.get(team_b, {'wins': 0, 'losses': 0})

def get_rank_val(team_name, team_id):
    r64 = tr26[tr26['team_id'] == team_id]['rank_64a'].iloc[0] if len(tr26[tr26['team_id'] == team_id]) else None
    rpi = ext_ranks.get('rpi', {}).get(team_name)
    massey = ext_ranks.get('massey', {}).get(team_name)
    dsr = ext_ranks.get('dsr', {}).get(team_name)
    vals = [v for v in [r64, rpi, massey, dsr] if v is not None and not np.isnan(v)]
    true_rank = int(round(np.mean(vals))) if vals else None
    return {'64A': r64, 'RPI': rpi, 'Massey': massey, 'DSR': dsr, 'True Rank': true_rank}

ranks_a = get_rank_val(team_a, team_a_id)
ranks_b = get_rank_val(team_b, team_b_id)

h1, h2, h3 = st.columns([2, 1, 2])
with h1:
    st.markdown(f'### {team_a}')
    st.markdown(f'**Record: {rec_a["wins"]}-{rec_a["losses"]}**')
    rank_parts = [f'{k}: #{int(v)}' for k, v in ranks_a.items() if v is not None]
    st.caption(' | '.join(rank_parts))
with h2:
    st.markdown('<h2 style="text-align:center; color:#888;">vs</h2>', unsafe_allow_html=True)
with h3:
    st.markdown(f'### {team_b}')
    st.markdown(f'**Record: {rec_b["wins"]}-{rec_b["losses"]}**')
    rank_parts = [f'{k}: #{int(v)}' for k, v in ranks_b.items() if v is not None]
    st.caption(' | '.join(rank_parts))

st.markdown('---')


# ── Spider Chart ─────────────────────────────────────────────────────────────
st.markdown('### Team Comparison')
spider = build_spider_chart(team_a, team_b, team_a_id, team_b_id,
                             pr, hitting_pbp, pitching_pbp,
                             CARD_RED, NAVY)
st.plotly_chart(spider, use_container_width=True)


# ── Game-by-Game Predictions ─────────────────────────────────────────────────
st.markdown('---')
st.markdown('### Game-by-Game Predictions')

starters_a = get_starters(team_a_id, pr, players, n=3)
starters_b = get_starters(team_b_id, pr, players, n=3)
bp_a = get_bullpen(team_a_id, pr, players)
bp_b = get_bullpen(team_b_id, pr, players)

for game_num in [1, 2, 3]:
    g1, g2, g3 = st.columns([2, 1, 2])
    sp_a = starters_a[game_num - 1] if len(starters_a) >= game_num else None
    sp_b = starters_b[game_num - 1] if len(starters_b) >= game_num else None

    # Compute WP for this game
    a_static = rank_pct_map.get(team_a, 0.5)
    b_static = rank_pct_map.get(team_b, 0.5)
    a_player = adjusted_team_pct(team_a, profiles, 'Weekend', game_num, a_static)
    b_player = adjusted_team_pct(team_b, profiles, 'Weekend', game_num, b_static)
    a_p = blend_with_static(a_player, a_static)
    b_p = blend_with_static(b_player, b_static)
    # team_b is home
    wp_home = pre_game_wp(b_p, a_p)
    wp_away = 1 - wp_home

    with g1:
        st.markdown(f'**Game {game_num} — {team_a} ({wp_away*100:.1f}%)**')
        if sp_a:
            st.write(f'SP: {sp_a["name"]} ({sp_a["position"]})')
            st.caption(f'Pitcher percentile: {sp_a["pit_pct"]:.1%}')
        else:
            st.write('SP: TBD')
    with g2:
        # Visual WP bar
        st.markdown(f'<div style="text-align:center; font-size:1.5em; font-weight:bold; margin-top:10px;">'
                    f'{wp_away*100:.0f}% — {wp_home*100:.0f}%</div>',
                    unsafe_allow_html=True)
    with g3:
        st.markdown(f'**Game {game_num} — {team_b} ({wp_home*100:.1f}%)**')
        if sp_b:
            st.write(f'SP: {sp_b["name"]} ({sp_b["position"]})')
            st.caption(f'Pitcher percentile: {sp_b["pit_pct"]:.1%}')
        else:
            st.write('SP: TBD')

# Bullpen comparison
st.markdown('---')
st.markdown('### Bullpen Comparison')
bp1, bp2 = st.columns(2)
with bp1:
    st.markdown(f'**{team_a} Bullpen** (arms 4-10)')
    if bp_a['count']:
        st.metric('Mean Percentile', f"{bp_a['mean_pct']:.1%}")
        st.caption(', '.join(bp_a['names'][:5]))
    else:
        st.write('Insufficient data')
with bp2:
    st.markdown(f'**{team_b} Bullpen** (arms 4-10)')
    if bp_b['count']:
        st.metric('Mean Percentile', f"{bp_b['mean_pct']:.1%}")
        st.caption(', '.join(bp_b['names'][:5]))
    else:
        st.write('Insufficient data')


# ── Who's Hot (last 14 days) ─────────────────────────────────────────────────
st.markdown('---')
st.markdown("### Who's Hot (last 14 days)")

# Need to map team names to PBP team names (PBP has mascot suffix)
# Build reverse map: short name -> PBP full name
if not hitting_pbp.empty:
    pbp_names = set(hitting_pbp['teamName'].dropna())
    def find_pbp_name(short):
        for full in pbp_names:
            if isinstance(full, str) and full.startswith(short):
                return full
        return short
    pbp_a = find_pbp_name(team_a)
    pbp_b = find_pbp_name(team_b)
else:
    pbp_a = team_a
    pbp_b = team_b

hot1, hot2 = st.columns(2)
with hot1:
    st.markdown(f'**{team_a} — Hot Hitters**')
    hot_h_a = get_hot_hitters(pbp_a, hitting_pbp)
    if hot_h_a:
        for h in hot_h_a:
            st.write(f"**{h['name']}** — {h['avg']} AVG, {h['hr']} HR, {h['rbi']} RBI ({h['games']}G)")
    else:
        st.caption('No recent data')

    st.markdown(f'**{team_a} — Hot Pitcher**')
    hot_p_a = get_hot_pitcher(pbp_a, pitching_pbp)
    if hot_p_a:
        st.write(f"**{hot_p_a['name']}** — {hot_p_a['era']} ERA, {hot_p_a['so']} K, {hot_p_a['bb']} BB ({hot_p_a['ip']} IP, {hot_p_a['games']}G)")
    else:
        st.caption('No recent data')

with hot2:
    st.markdown(f'**{team_b} — Hot Hitters**')
    hot_h_b = get_hot_hitters(pbp_b, hitting_pbp)
    if hot_h_b:
        for h in hot_h_b:
            st.write(f"**{h['name']}** — {h['avg']} AVG, {h['hr']} HR, {h['rbi']} RBI ({h['games']}G)")
    else:
        st.caption('No recent data')

    st.markdown(f'**{team_b} — Hot Pitcher**')
    hot_p_b = get_hot_pitcher(pbp_b, pitching_pbp)
    if hot_p_b:
        st.write(f"**{hot_p_b['name']}** — {hot_p_b['era']} ERA, {hot_p_b['so']} K, {hot_p_b['bb']} BB ({hot_p_b['ip']} IP, {hot_p_b['games']}G)")
    else:
        st.caption('No recent data')


# ── Series Win Probability ───────────────────────────────────────────────────
st.markdown('---')
st.markdown('### Series Outcome Probabilities')
# Monte Carlo: probability of sweeping, winning 2-1, etc.
n_sims = 5000
game_wps = []
for gn in [1, 2, 3]:
    a_pl = adjusted_team_pct(team_a, profiles, 'Weekend', gn, a_static)
    b_pl = adjusted_team_pct(team_b, profiles, 'Weekend', gn, b_static)
    a_bl = blend_with_static(a_pl, a_static)
    b_bl = blend_with_static(b_pl, b_static)
    game_wps.append(pre_game_wp(b_bl, a_bl))  # P(home=team_b wins)

rng = np.random.default_rng(42)
a_series_wins = 0
b_series_wins = 0
outcomes = {'3-0 away': 0, '2-1 away': 0, '2-1 home': 0, '3-0 home': 0}
for _ in range(n_sims):
    a_w = sum(1 for gw in game_wps if rng.random() > gw)  # away wins when home loses
    b_w = 3 - a_w
    if a_w >= 2:
        a_series_wins += 1
    else:
        b_series_wins += 1
    if a_w == 3:
        outcomes['3-0 away'] += 1
    elif a_w == 2:
        outcomes['2-1 away'] += 1
    elif b_w == 2:
        outcomes['2-1 home'] += 1
    else:
        outcomes['3-0 home'] += 1

s1, s2, s3, s4 = st.columns(4)
s1.metric(f'{team_a} sweep', f"{outcomes['3-0 away']/n_sims:.1%}")
s2.metric(f'{team_a} 2-1', f"{outcomes['2-1 away']/n_sims:.1%}")
s3.metric(f'{team_b} 2-1', f"{outcomes['2-1 home']/n_sims:.1%}")
s4.metric(f'{team_b} sweep', f"{outcomes['3-0 home']/n_sims:.1%}")

st.caption(f'Series win: **{team_a} {a_series_wins/n_sims:.1%}** — **{team_b} {b_series_wins/n_sims:.1%}** ({n_sims:,} simulations)')
