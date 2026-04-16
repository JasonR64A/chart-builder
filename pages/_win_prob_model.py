"""
Single source of truth for the pre-game win-probability model.

Used by:
  - pages/13_Win_Probability.py  (individual-game WP display)
  - pages/5_Win_Generator.py     (season projections + single matchup)
  - pages/_shared_rpi.py         (Predicted RPI feeds Bracketology)

Any tweak to the WP formula (edge compression, HFA, blend weights, etc.)
should happen here so all pages move together.
"""
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _APP_DIR / 'data'

# ── Model constants ──────────────────────────────────────────────────────────
HOME_FIELD_ADVANTAGE = 0.04
# Compress the log5 edge so best-vs-worst lands around 90% including HFA.
PREGAME_EDGE_SCALE = 0.72
# Safety clamp — the math above should keep us inside this range.
PREGAME_CLAMP_MIN = 0.10
PREGAME_CLAMP_MAX = 0.90
# Blend of team-rank pct with player-driven pct (0 = pure player, 1 = pure team).
TEAM_RANK_BLEND = 0.5
# Pitching weight in the player-driven talent split (rest goes to hitting).
PITCHING_WEIGHT = 0.60


# ── Core math ────────────────────────────────────────────────────────────────
def log5(wa, wb):
    denom = wa + wb - 2 * wa * wb
    return (wa - wa * wb) / denom if denom else 0.5


def pre_game_wp(home_pct, away_pct, include_hfa=True):
    """Compressed log5 pre-game WP. Returns P(home wins)."""
    if home_pct is None or away_pct is None:
        return 0.5
    try:
        if np.isnan(home_pct) or np.isnan(away_pct):
            return 0.5
    except TypeError:
        return 0.5
    base = log5(home_pct, away_pct)
    scaled = 0.5 + (base - 0.5) * PREGAME_EDGE_SCALE
    adjusted = scaled + (HOME_FIELD_ADVANTAGE if include_hfa else 0.0)
    return max(PREGAME_CLAMP_MIN, min(PREGAME_CLAMP_MAX, adjusted))


# ── Player-driven talent profiles ────────────────────────────────────────────
@st.cache_data
def build_team_profiles(sport, division):
    """
    Per-team talent profiles for the player-adjusted component.
    Returns dict keyed by team name:
      { team_name: {'pitchers': [pit_pct,...], 'hitters_by_pa': [hit_pct,...]} }
    'sport' can be 'Baseball'/'Softball' OR 'baseball'/'softball' (case-insensitive).
    'division' is 'D1' / 'D2' / 'D3'.
    """
    sport_label = 'Baseball' if str(sport).lower().startswith('base') else 'Softball'
    sport_folder = sport_label.lower()
    try:
        pr = pd.read_csv(DATA_DIR / 'player_rank.csv', low_memory=False,
                         usecols=['player_id','team_id','year',
                                  'percentile_rank_weighted_run_created_efficiency',
                                  'percentile_rank_weighted_run_allowed_efficiency'])
        pr['year'] = pd.to_numeric(pr['year'], errors='coerce')
        pr = pr.sort_values('year').drop_duplicates('player_id', keep='last')
        pr['team_id'] = pd.to_numeric(pr['team_id'], errors='coerce').astype('Int64')
        pr['hit'] = pd.to_numeric(pr['percentile_rank_weighted_run_created_efficiency'], errors='coerce')
        pr['pit'] = pd.to_numeric(pr['percentile_rank_weighted_run_allowed_efficiency'], errors='coerce')

        teams = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False, dtype=str).fillna('')
        teams = teams[teams['sport'] == sport_label]
        teams['id_int'] = pd.to_numeric(teams['id'], errors='coerce').astype('Int64')
        id_to_name = dict(zip(teams['id_int'], teams['name']))

        # Hitter usage — last 30 days games-played from hitting_pbp for this sport+division
        hit_pbp_path = _APP_DIR / 'pbp_data' / sport_folder / f'hitting_pbp_{division}.csv'
        pa_counts = {}
        if hit_pbp_path.exists():
            hp = pd.read_csv(hit_pbp_path, low_memory=False, usecols=['date','playerId'])
            hp['date'] = pd.to_datetime(hp['date'], errors='coerce', format='mixed')
            cutoff = hp['date'].max() - pd.Timedelta(days=30)
            hp = hp[hp['date'] >= cutoff]
            pa_counts = hp.groupby('playerId').size().to_dict()

        profiles = {}
        for tid, grp in pr.groupby('team_id'):
            name = id_to_name.get(tid)
            if not name:
                continue
            pit_df = grp.dropna(subset=['pit']); pit_df = pit_df[pit_df['pit'] > 0]
            pit_list = pit_df.sort_values('pit', ascending=False)['pit'].tolist()
            hit_df = grp.dropna(subset=['hit']).copy(); hit_df = hit_df[hit_df['hit'] > 0]
            hit_df['usage'] = hit_df['player_id'].map(pa_counts).fillna(0)
            hit_list = hit_df.sort_values(['usage','hit'], ascending=[False, False])['hit'].tolist()
            profiles[name] = {'pitchers': pit_list, 'hitters_by_pa': hit_list}
        return profiles
    except Exception as e:
        print(f'build_team_profiles failed: {e}')
        return {}


def adjusted_team_pct(name, profiles, game_type, game_num, static_pct):
    """
    Player-driven team talent pct.
      - Weekend: starter = P1/P2/P3 by game_num; hitting = mean top-9 by PA
      - Midweek: starter = mean of pitchers 5-12; hitting = bench lineup
                 (top-6 summed + 3 * mean of 7-12) / 9
    Falls back to static_pct if profile missing.
    """
    if not name or name not in profiles:
        return static_pct
    prof = profiles[name]
    pitchers = prof.get('pitchers', [])
    hitters = prof.get('hitters_by_pa', [])

    pit_pct = None
    if game_type == 'Weekend':
        idx = max(0, min(2, int(game_num) - 1))
        if len(pitchers) > idx:
            pit_pct = pitchers[idx]
    else:
        mw_pool = pitchers[4:12]
        if mw_pool:
            pit_pct = sum(mw_pool) / len(mw_pool)

    hit_pct = None
    if game_type == 'Weekend':
        top9 = hitters[:9]
        if top9:
            hit_pct = sum(top9) / len(top9)
    else:
        top6 = hitters[:6]
        bench_pool = hitters[6:12]
        if top6 and bench_pool:
            bench_mean = sum(bench_pool) / len(bench_pool)
            hit_pct = (sum(top6) + 3 * bench_mean) / 9
        elif hitters:
            hit_pct = sum(hitters[:9]) / min(9, len(hitters))

    if pit_pct is None or hit_pct is None:
        return static_pct
    return (1 - PITCHING_WEIGHT) * hit_pct + PITCHING_WEIGHT * pit_pct


def blend_with_static(player_pct, static_pct, w=TEAM_RANK_BLEND):
    """Blend player-driven pct with static team-rank pct (w = weight on static)."""
    if player_pct is None:
        return static_pct
    if static_pct is None:
        return player_pct
    return w * static_pct + (1 - w) * player_pct


def detect_game_context(game_date, opp_name, team_schedule):
    """
    Given a scheduled game's date + opponent and the team's full schedule,
    detect Weekend series (>=2 games vs same opponent within 4-day window)
    vs Midweek. Returns (game_type, game_num) where game_num ∈ {1,2,3} for
    Weekend games, 0 for Midweek.
    """
    sel = pd.to_datetime(game_date, format='mixed', errors='coerce')
    if pd.isna(sel):
        return 'Midweek', 0
    opp_clean = str(opp_name).split('@')[0].strip()
    sched = team_schedule.copy()
    sched['_opp'] = sched['opponentName'].astype(str).str.split('@').str[0].str.strip()
    sched['_d'] = pd.to_datetime(sched['date'], format='mixed', errors='coerce')
    window = sched[(sched['_opp'] == opp_clean) &
                   (sched['_d'] >= sel - pd.Timedelta(days=2)) &
                   (sched['_d'] <= sel + pd.Timedelta(days=2))]
    # Count ROWS (not unique dates) so doubleheaders on the same date
    # get counted as 2 games, not 1.
    n_games = len(window)
    if n_games >= 2:
        # Position within the window (sorted by date) — for doubleheaders
        # on the same date, row order determines game number.
        window_sorted = window.sort_values('_d').reset_index(drop=True)
        pos = window_sorted.index[window_sorted['_d'] == sel]
        game_num = max(1, min(3, int(pos[0]) + 1 if len(pos) else 1))
        return 'Weekend', game_num
    return 'Midweek', 0


def compute_matchup_wp(home_name, away_name, is_home_for_selected,
                        game_type, game_num, rank_pct_map, profiles):
    """Full pre-game WP pipeline. Returns P(selected team wins)."""
    home_static = rank_pct_map.get(home_name)
    away_static = rank_pct_map.get(away_name)
    home_player = adjusted_team_pct(home_name, profiles, game_type, game_num, home_static)
    away_player = adjusted_team_pct(away_name, profiles, game_type, game_num, away_static)
    home_p = blend_with_static(home_player, home_static)
    away_p = blend_with_static(away_player, away_static)
    p_home_wins = pre_game_wp(home_p, away_p)
    return p_home_wins if is_home_for_selected else (1 - p_home_wins)


def build_rank_pct_map(ranks_df, rank_col='rank'):
    """
    Convert a DataFrame of teams with a numeric rank column into a
    name -> rank_pct dict where 1.0 = best in the filtered set.
    Uses dense ordinal rank so ties collapse cleanly.
    """
    rs = ranks_df.dropna(subset=[rank_col, 'team_name']).copy() if 'team_name' in ranks_df.columns \
        else ranks_df.dropna(subset=[rank_col]).copy()
    rs['_ord'] = rs[rank_col].rank(method='min', ascending=True)
    N = len(rs)
    if N < 2:
        rs['_pct'] = 0.5
    else:
        rs['_pct'] = 1 - (rs['_ord'] - 1) / (N - 1)
    name_col = 'team_name' if 'team_name' in rs.columns else 'name'
    return dict(zip(rs[name_col], rs['_pct']))
