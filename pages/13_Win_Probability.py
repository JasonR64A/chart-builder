"""
Win Probability Pace Chart — play-by-play WP curve for any game.

Starting WP uses log5 of 64 integer ranks ± 4% home field advantage.
In-game WP is an empirical state lookup (inning, half, outs, runners,
score_diff) blended with the pre-game anchor: early-game weight on
pre-game, late-game weight on state.

Built 2026-04-15. Data: 2026 D1 play-by-play (~5,200 games, 520K plays).
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _APP_DIR / 'data'
PBP_FILE = _APP_DIR / 'pbp_data' / 'play_by_play' / 'baseball_play_by_play_D1.csv'
LOOKUP_FILE = _APP_DIR / 'pbp_data' / 'wp_state_lookup_bb_d1.pkl'
LOGO_DIR = _APP_DIR / 'team_logos_512'

HOME_FIELD_ADVANTAGE = 0.04  # ±4% bump
CLAMP_MIN = 0.01
CLAMP_MAX = 0.99


# ── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_pbp():
    cols = ['gameId', 'date', 'awayTeam', 'homeTeam', 'inning', 'halfInning',
            'outs', 'runner1B', 'runner2B', 'runner3B',
            'awayScore', 'homeScore', 'player', 'playDescription']
    df = pd.read_csv(PBP_FILE, low_memory=False, usecols=cols)
    df['inning'] = pd.to_numeric(df['inning'], errors='coerce').fillna(0).astype(int)
    df['outs'] = pd.to_numeric(df['outs'], errors='coerce').fillna(0).astype(int)
    df['awayScore'] = pd.to_numeric(df['awayScore'], errors='coerce').fillna(0).astype(int)
    df['homeScore'] = pd.to_numeric(df['homeScore'], errors='coerce').fillna(0).astype(int)
    return df


@st.cache_data
def load_lookup():
    with open(LOOKUP_FILE, 'rb') as f:
        return pickle.load(f)


@st.cache_data
def load_teams():
    t = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False)
    t['id'] = pd.to_numeric(t['id'], errors='coerce').fillna(0).astype(int)
    tr = pd.read_csv(DATA_DIR / 'team_rank.csv', low_memory=False)
    tr = tr[tr['year'] == 2026].copy()
    tr['rank'] = pd.to_numeric(tr['integer_64_rank_total'], errors='coerce')
    confs = pd.read_csv(DATA_DIR / 'conferences.csv', low_memory=False)
    di_ids = set(confs[confs['division'] == 'D-I']['id'])
    bb_d1 = t[(t['sport'] == 'Baseball') & (t['conference_id'].isin(di_ids))]
    tr_d1 = tr[tr['team_id'].isin(bb_d1['id'])]
    # rank → percentile (1 = best → 1.0)
    N = len(tr_d1)
    tr_d1 = tr_d1.dropna(subset=['rank']).copy()
    tr_d1['rank_pct'] = 1 - (tr_d1['rank'] - 1) / (N - 1)
    name_to_pct = dict(zip(bb_d1.set_index('id').loc[tr_d1['team_id'].values, 'name'],
                            tr_d1['rank_pct']))
    name_to_rank = dict(zip(bb_d1.set_index('id').loc[tr_d1['team_id'].values, 'name'],
                             tr_d1['rank'].astype(int)))
    name_to_id = dict(zip(bb_d1['name'], bb_d1['id']))
    return name_to_pct, name_to_rank, name_to_id


def log5(wa, wb):
    denom = wa + wb - 2 * wa * wb
    if denom == 0:
        return 0.5
    return (wa - wa * wb) / denom


def pre_game_wp(home_pct, away_pct):
    """Home team pre-game WP. Log5 + home field bump, clamped."""
    if pd.isna(home_pct) or pd.isna(away_pct):
        return 0.5
    base = log5(home_pct, away_pct)  # log5(home, away) = P(home wins)
    return max(CLAMP_MIN, min(CLAMP_MAX, base + HOME_FIELD_ADVANTAGE))


def state_key(row):
    inning = min(int(row['inning']), 10) if pd.notna(row['inning']) else 1
    half = 1 if row.get('halfInning') == 'bottom' else 0
    outs = min(int(row['outs']), 2) if pd.notna(row['outs']) else 0
    bases = int(pd.notna(row.get('runner1B'))) + \
            int(pd.notna(row.get('runner2B'))) * 2 + \
            int(pd.notna(row.get('runner3B'))) * 4
    sd = int(row['homeScore']) - int(row['awayScore'])
    sd = max(-10, min(10, sd))
    return (inning, half, outs, bases, sd)


def game_progress_weight(play_idx, total_plays):
    """Weight on state-based WP. 0 = all pre-game. 1 = all state."""
    if total_plays <= 0:
        return 0.5
    return min(1.0, play_idx / total_plays)


# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='Win Probability', layout='wide')
st.title('Win Probability — Pace Chart')
st.caption('Play-by-play win probability curve. Starting point = log5 of 64 integer ranks ± 4% home-field bump. '
           'In-game = empirical state lookup blended with pre-game anchor (early weight → pre-game, late weight → state).')

pbp = load_pbp()
lookup = load_lookup()
team_pct, team_rank, team_id_map = load_teams()

# Game picker
st.sidebar.markdown('### Game selection')
teams_in_data = sorted(set(pbp['awayTeam'].dropna()) | set(pbp['homeTeam'].dropna()))
sel_team = st.sidebar.selectbox('Team', teams_in_data, index=teams_in_data.index('Boston College Eagles')
                                 if 'Boston College Eagles' in teams_in_data else 0)

team_games = pbp[(pbp['awayTeam'] == sel_team) | (pbp['homeTeam'] == sel_team)]
game_list = team_games.groupby('gameId').first().reset_index()
game_list['date_parsed'] = pd.to_datetime(game_list['date'], format='mixed', errors='coerce')
game_list = game_list.sort_values('date_parsed', ascending=False)
game_list['label'] = game_list.apply(
    lambda r: f"{r['date']}  {r['awayTeam']} @ {r['homeTeam']}", axis=1)
sel_label = st.sidebar.selectbox('Game', game_list['label'].tolist())
sel_gid = int(game_list[game_list['label'] == sel_label]['gameId'].iloc[0])

# Pull full game
game = pbp[pbp['gameId'] == sel_gid].copy().reset_index(drop=True)
home = game['homeTeam'].iloc[0]
away = game['awayTeam'].iloc[0]

# Short names (strip common suffixes for matching team_rank)
def short_name(full_name):
    for sfx in [' Eagles', ' Tigers', ' Bulldogs', ' Wildcats', ' Hokies', ' Crimson Tide',
                ' Volunteers', ' Razorbacks', ' Commodores', ' Rebels', ' Gators', ' Longhorns',
                ' Aggies', ' Bears', ' Hawks', ' Owls', ' Ducks', ' Beavers', ' Huskies',
                ' Jayhawks', ' Cornhuskers', ' Sooners', ' Cowboys', ' Cyclones']:
        if full_name.endswith(sfx):
            return full_name[:-len(sfx)]
    # Split on last space and try the name without the final word
    parts = full_name.rsplit(' ', 1)
    if len(parts) == 2:
        return parts[0]
    return full_name

def find_team_pct(full_name):
    # Exact, short, or anything that starts with the key
    for cand in [full_name, short_name(full_name)]:
        if cand in team_pct:
            return cand, team_pct[cand], team_rank.get(cand)
    for k in team_pct.keys():
        if full_name.startswith(k) or k.startswith(short_name(full_name)):
            return k, team_pct[k], team_rank.get(k)
    return full_name, None, None

home_key, home_p, home_r = find_team_pct(home)
away_key, away_p, away_r = find_team_pct(away)

# Show game header + rankings
c1, c2, c3 = st.columns([2, 1, 2])
c1.markdown(f"### {away}")
c1.caption(f"64 Rank: {away_r if away_r else 'N/A'} ({away_p:.3f} pct)" if away_p else "No rank data")
c2.markdown('### vs')
c3.markdown(f"### {home}")
c3.caption(f"64 Rank: {home_r if home_r else 'N/A'} ({home_p:.3f} pct)" if home_p else "No rank data")

# Pre-game WP
if home_p is not None and away_p is not None:
    pg_home = pre_game_wp(home_p, away_p)
else:
    pg_home = 0.5  # fallback

c1, c2, c3, c4 = st.columns(4)
c1.metric('Pre-game WP (home)', f"{pg_home*100:.1f}%")
c2.metric('Pre-game WP (away)', f"{(1-pg_home)*100:.1f}%")
c3.metric('Final score', f"{int(game['awayScore'].max())} - {int(game['homeScore'].max())}")
c4.metric('Plays in game', f"{len(game)}")

# Build WP curve
total_plays = len(game)
home_wp_curve = [pg_home]
for i, row in game.iterrows():
    key = state_key(row)
    state_info = lookup.get(key)
    w = game_progress_weight(i + 1, total_plays)
    if state_info is None or state_info[1] < 10:  # fall back to pre-game if sample too small
        wp = pg_home
    else:
        state_wp = state_info[0]
        wp = w * state_wp + (1 - w) * pg_home
    home_wp_curve.append(max(CLAMP_MIN, min(CLAMP_MAX, wp)))

# Plot
fig, ax = plt.subplots(figsize=(14, 6), facecolor='#FAF8F2')
ax.set_facecolor('#FAF8F2')
ax.fill_between(range(len(home_wp_curve)), home_wp_curve, 0.5,
                where=[w >= 0.5 for w in home_wp_curve], color='#2E5C8A', alpha=0.25, label=f'{home} ahead')
ax.fill_between(range(len(home_wp_curve)), home_wp_curve, 0.5,
                where=[w < 0.5 for w in home_wp_curve], color='#C41230', alpha=0.25, label=f'{away} ahead')
ax.plot(range(len(home_wp_curve)), home_wp_curve, color='#2D2926', linewidth=2.2)
ax.axhline(0.5, color='#666', linewidth=0.8, linestyle='--', alpha=0.5)

# Annotate start and end
ax.annotate(f"Start: {pg_home*100:.1f}%", xy=(0, pg_home), xytext=(5, pg_home + 0.05),
            fontsize=10, color='#333')
ax.annotate(f"End: {home_wp_curve[-1]*100:.1f}%", xy=(len(home_wp_curve)-1, home_wp_curve[-1]),
            xytext=(len(home_wp_curve)-30, home_wp_curve[-1] + 0.05), fontsize=10, color='#333')

# Mark inning transitions (top → bottom of each inning)
inning_changes = game[['inning', 'halfInning']].drop_duplicates().index.tolist()
for idx in inning_changes[1:]:  # skip first
    ax.axvline(idx + 1, color='#aaa', linewidth=0.4, alpha=0.4)

ax.set_xlim(0, len(home_wp_curve) - 1)
ax.set_ylim(0, 1)
ax.set_xlabel(f'Play #  ({len(game)} plays in game)', fontsize=11)
ax.set_ylabel(f'{home} Win Probability', fontsize=11)
ax.set_title(f'{away} @ {home} — {game["date"].iloc[0]}', fontsize=14, fontweight='bold')
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
ax.legend(loc='upper left', frameon=False)
ax.grid(True, alpha=0.2)

buf = BytesIO()
fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='#FAF8F2')
buf.seek(0)
plt.close(fig)
st.image(buf, use_container_width=True)

st.download_button('Download Chart PNG', data=buf,
                   file_name=f'wp_{sel_gid}_{home}_vs_{away}.png',
                   mime='image/png')

# Optional: show the raw play log
with st.expander('Play-by-play log with WP'):
    game_display = game.copy()
    game_display['Home WP'] = [f'{w*100:.1f}%' for w in home_wp_curve[1:]]
    st.dataframe(game_display[['inning', 'halfInning', 'outs', 'awayScore', 'homeScore',
                                'player', 'playDescription', 'Home WP']],
                 use_container_width=True, hide_index=True)
