"""
64 Analytics — Win Generator
Season win projections using 64 Rank + remaining schedule.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO

# ── Path setup (works locally and on Streamlit Cloud) ─────────────────────────
_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _APP_DIR / 'data'
# No longer needed — schedules loaded from DATA_DIR
BRAND_LOGO = _APP_DIR / 'assets' / 'brand_logo_dark.png'

# Legacy logistic model — kept for reference / fallback
LOGISTIC_A = 0.006355505043163222
LOGISTIC_B = 151.44616903215248
RESIDUAL_STD = 0.09449904970441943

# Pre-game WP model (same calibration as Win Probability page)
HOME_FIELD_ADVANTAGE = 0.04
PREGAME_EDGE_SCALE = 0.72        # compress log5 edge so best-vs-worst ≈ 90%
PREGAME_CLAMP_MIN = 0.10
PREGAME_CLAMP_MAX = 0.90
TEAM_RANK_BLEND = 0.5            # 50/50 team-rank + player-driven
PITCHING_WEIGHT = 0.60           # 60/40 pitching-to-hitting


# ── Model Functions ───────────────────────────────────────────────────────────
def rank_to_win_pct(rank):
    """Legacy: convert 64 integer rank to expected win percentage."""
    return 1.0 / (1.0 + np.exp(LOGISTIC_A * (rank - LOGISTIC_B)))


def log5(wa, wb):
    denom = wa + wb - 2 * wa * wb
    return (wa - wa * wb) / denom if denom else 0.5


def pre_game_wp(home_pct, away_pct, include_hfa=True):
    """Compressed log5 pre-game WP matching the Win Probability page."""
    if home_pct is None or away_pct is None or np.isnan(home_pct) or np.isnan(away_pct):
        return 0.5
    base = log5(home_pct, away_pct)
    scaled = 0.5 + (base - 0.5) * PREGAME_EDGE_SCALE
    adjusted = scaled + (HOME_FIELD_ADVANTAGE if include_hfa else 0.0)
    return max(PREGAME_CLAMP_MIN, min(PREGAME_CLAMP_MAX, adjusted))


def log5_win_prob(rank_a, rank_b):
    """Legacy raw log5 (for back-compat). Prefer pre_game_wp + rank_pct pipeline."""
    pA = np.clip(rank_to_win_pct(rank_a), 0.01, 0.99)
    pB = np.clip(rank_to_win_pct(rank_b), 0.01, 0.99)
    return (pA * (1 - pB)) / (pA * (1 - pB) + pB * (1 - pA))


@st.cache_data
def build_team_profiles(sport, division):
    """
    Per-team talent profiles for player-adjusted WP. Mirrors the Win
    Probability page (pages/13_Win_Probability.py). Returns dict keyed by
    team name:
      { team_name: {'pitchers': [pit_pct,...], 'hitters_by_pa': [hit_pct,...]} }
    """
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

        sport_label = 'Baseball' if sport == 'baseball' else 'Softball'
        teams = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False, dtype=str).fillna('')
        teams = teams[teams['sport'] == sport_label]
        teams['id_int'] = pd.to_numeric(teams['id'], errors='coerce').astype('Int64')
        id_to_name = dict(zip(teams['id_int'], teams['name']))

        # Hitter usage — last 30 days games-played from hitting_pbp
        hit_pbp_path = _APP_DIR / 'pbp_data' / sport / f'hitting_pbp_{division}.csv'
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
    """Player-driven team talent pct. Falls back to static_pct if profile missing."""
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
    Given a scheduled game's date and opponent, detect whether it's part
    of a Weekend series (>=2 games vs same opponent within a 4-day
    window) or a Midweek single game. Returns (game_type, game_num).
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
    dates = sorted(window['_d'].dropna().unique())
    if len(dates) >= 2:
        game_num = max(1, min(3, sum(1 for d in dates if d <= sel)))
        return 'Weekend', game_num
    return 'Midweek', 0


def compute_matchup_wp(home_name, away_name, is_home_for_selected, game_type, game_num,
                       rank_pct_map, profiles):
    """Full pre-game WP pipeline. Returns P(selected team wins)."""
    home_static = rank_pct_map.get(home_name)
    away_static = rank_pct_map.get(away_name)
    home_player = adjusted_team_pct(home_name, profiles, game_type, game_num, home_static)
    away_player = adjusted_team_pct(away_name, profiles, game_type, game_num, away_static)
    home_p = blend_with_static(home_player, home_static)
    away_p = blend_with_static(away_player, away_static)
    p_home_wins = pre_game_wp(home_p, away_p)
    return p_home_wins if is_home_for_selected else (1 - p_home_wins)


def project_season_probs(win_probs, current_wins, current_losses, n_simulations=1000):
    """
    Monte Carlo projection when per-game win probabilities are already
    computed (e.g. via the full compressed-log5 + HFA + player-blend
    pipeline). Faster than re-computing per game per sim.
    """
    probs = np.asarray(win_probs, dtype=float)
    rng = np.random.default_rng()
    total_games = current_wins + current_losses + len(probs)
    sim_wins = []
    for _ in range(n_simulations):
        rolls = rng.random(len(probs))
        wins = current_wins + int((rolls < probs).sum())
        sim_wins.append(wins)
    sim_wins = np.array(sim_wins)
    expected = float(np.mean(sim_wins))
    expected_losses = total_games - expected
    return {
        'expected_wins': round(expected, 1),
        'expected_losses': round(expected_losses, 1),
        'expected_win_pct': round(expected / total_games, 3) if total_games > 0 else 0,
        'win_low': int(np.percentile(sim_wins, 10)),
        'win_high': int(np.percentile(sim_wins, 90)),
        'win_floor': int(np.min(sim_wins)),
        'win_ceiling': int(np.max(sim_wins)),
        'remaining_games': len(probs),
        'projected_remaining_wins': round(expected - current_wins, 1),
    }


def project_season(team_rank, opponent_ranks, current_wins, current_losses, n_simulations=1000):
    """Legacy projection using static rank-to-wp logistic + raw log5."""
    sim_wins = []
    for _ in range(n_simulations):
        wins = current_wins
        losses = current_losses
        for opp_rank in opponent_ranks:
            prob = log5_win_prob(team_rank, opp_rank)
            if np.random.random() < prob:
                wins += 1
            else:
                losses += 1
        sim_wins.append(wins)

    sim_wins = np.array(sim_wins)
    expected = np.mean(sim_wins)
    total_games = current_wins + current_losses + len(opponent_ranks)
    expected_losses = total_games - expected

    return {
        'expected_wins': round(expected, 1),
        'expected_losses': round(expected_losses, 1),
        'expected_win_pct': round(expected / total_games, 3) if total_games > 0 else 0,
        'win_low': int(np.percentile(sim_wins, 10)),
        'win_high': int(np.percentile(sim_wins, 90)),
        'win_floor': int(np.min(sim_wins)),
        'win_ceiling': int(np.max(sim_wins)),
        'remaining_games': len(opponent_ranks),
        'projected_remaining_wins': round(expected - current_wins, 1),
    }


# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_team_ranks(sport, year):
    """Load True Rank for all teams: average of 64A Rank, RPI, Massey, DSR."""
    tr = pd.read_csv(DATA_DIR / 'team_rank.csv', low_memory=False)
    tr = tr[tr['year'] == int(year)]
    tr['team_id'] = tr['team_id'].astype(str)

    teams = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False)
    confs = pd.read_csv(DATA_DIR / 'conferences.csv', low_memory=False)
    teams = teams.merge(confs[['id', 'name', 'division']], left_on='conference_id',
                        right_on='id', suffixes=('', '_conf'))
    teams = teams.rename(columns={'id': 'team_db_id', 'name': 'team_name', 'name_conf': 'conference_name'})
    teams['team_db_id'] = teams['team_db_id'].astype(str)

    sport_label = 'Baseball' if sport == 'baseball' else 'Softball'
    sport_teams = teams[teams['sport'] == sport_label]

    merged = tr.merge(sport_teams[['team_db_id', 'team_name', 'division', 'conference_name']],
                      left_on='team_id', right_on='team_db_id', how='inner')
    merged['rank_64a'] = pd.to_numeric(merged['integer_64_rank_total'], errors='coerce')

    # Load RPI ranks
    rpi_file = DATA_DIR / f'{sport}_rpi_D1.csv'
    if rpi_file.exists():
        rpi = pd.read_csv(rpi_file, low_memory=False)
        rpi_lookup = dict(zip(rpi['teamName'], rpi['rank']))
        merged['rank_rpi'] = merged['team_name'].map(rpi_lookup)
    else:
        merged['rank_rpi'] = np.nan

    # Load external name mapping
    name_map_file = DATA_DIR / 'rankings' / 'name_map.csv'
    ext_to_our: dict[str, str] = {}
    if name_map_file.exists():
        nm = pd.read_csv(name_map_file)
        ext_to_our = dict(zip(nm['external_name'], nm['our_name']))

    def _map_external(ext_name: str) -> str:
        """Map external ranking name to our team name."""
        return ext_to_our.get(ext_name, ext_name)

    # Load Massey ranks
    massey_file = DATA_DIR / 'rankings' / f'massey_{sport}.csv'
    if massey_file.exists():
        massey = pd.read_csv(massey_file, low_memory=False)
        massey_lookup = {_map_external(t): r for t, r in zip(massey['team'], massey['rank'])}
        merged['rank_massey'] = merged['team_name'].map(massey_lookup)
    else:
        merged['rank_massey'] = np.nan

    # Load DSR ranks
    dsr_file = DATA_DIR / 'rankings' / f'dsr_{sport}.csv'
    if dsr_file.exists():
        dsr = pd.read_csv(dsr_file, low_memory=False)
        dsr_lookup = {_map_external(t): r for t, r in zip(dsr['team'], dsr['rank'])}
        merged['rank_dsr'] = merged['team_name'].map(dsr_lookup)
    else:
        merged['rank_dsr'] = np.nan

    # True Rank = average of available rankings
    rank_cols = ['rank_64a', 'rank_rpi', 'rank_massey', 'rank_dsr']
    merged['rank'] = merged[rank_cols].mean(axis=1)
    # For teams missing some rankings, use what's available
    merged['rank'] = merged['rank'].fillna(merged['rank_64a'])
    merged['rankings_used'] = merged[rank_cols].notna().sum(axis=1)

    return merged.dropna(subset=['rank', 'team_name'])


@st.cache_data
def load_schedules(sport):
    """Load scraped schedule data."""
    schedule_file = DATA_DIR / f'schedules_full_{sport}.csv'
    if not schedule_file.exists():
        return pd.DataFrame()
    df = pd.read_csv(schedule_file, low_memory=False)
    return df


@st.cache_data
def load_current_records(year):
    """Load current W-L records from schedules.csv."""
    sched = pd.read_csv(DATA_DIR / 'schedules.csv', low_memory=False)
    if 'Year' in sched.columns:
        sched = sched.rename(columns={'Year': 'year'})
    sched = sched[sched['year'] == int(year)]
    sched['team_id'] = sched['team_id'].astype(str)
    sched['current_wins'] = sched['Conf_Win'].fillna(0) + sched['OOC_Win'].fillna(0)
    sched['current_losses'] = sched['Conf_Loss'].fillna(0) + sched['OOC_Loss'].fillna(0)
    return sched[['team_id', 'current_wins', 'current_losses']]


@st.cache_data
def compute_predicted_rpi(sport, _ranks_data, _schedules_data, _name_to_rank):
    """
    Predict every remaining game's W/L using True Rank Log5, then compute
    the full RPI formula (0.25*WP + 0.50*OWP + 0.25*OOWP) on projected records.
    Returns DataFrame with team, projected record, predicted RPI rank.
    """
    schedules = _schedules_data
    name_to_rank_local = _name_to_rank

    played = schedules[schedules['result'].notna() & (schedules['result'] != '')].copy()
    remaining = schedules[(schedules['result'].isna()) | (schedules['result'] == '')].copy()

    # Step 1: Build projected WP for EVERY team in the schedule
    # Actual results for played games + predicted W/L for remaining games
    team_wins: dict[str, float] = {}
    team_games: dict[str, float] = {}
    team_opponents: dict[str, list[str]] = {}

    # Count actual wins from played games
    for team_name, group in played.groupby('teamName'):
        wins = group['result'].str.startswith('W').sum()
        team_wins[team_name] = float(wins)
        team_games[team_name] = float(len(group))
        opps = group['opponentName'].apply(lambda x: str(x).split('@')[0].strip()).tolist()
        team_opponents[team_name] = opps

    # Add predicted wins from remaining games using Log5
    for team_name, group in remaining.groupby('teamName'):
        team_rank = name_to_rank_local.get(team_name)
        if team_rank is None:
            team_rank = LOGISTIC_B  # median

        for _, game in group.iterrows():
            opp_name = str(game.get('opponentName', '')).split('@')[0].strip()
            opp_rank = name_to_rank_local.get(opp_name, LOGISTIC_B)
            win_prob = log5_win_prob(team_rank, opp_rank)

            team_wins[team_name] = team_wins.get(team_name, 0) + win_prob
            team_games[team_name] = team_games.get(team_name, 0) + 1

            if team_name not in team_opponents:
                team_opponents[team_name] = []
            team_opponents[team_name].append(opp_name)

    # Step 2: Compute projected WP with NCAA location weighting
    # Home W=0.7, Away W=1.3, Neutral W=1.0
    team_home_games: dict[str, int] = {}
    team_away_games: dict[str, int] = {}
    team_neutral_games: dict[str, int] = {}
    for team_name, group in schedules.groupby('teamName'):
        for _, g in group.iterrows():
            if pd.notna(g.get('isAway')) and g['isAway'] == 1.0:
                team_away_games[team_name] = team_away_games.get(team_name, 0) + 1
            elif '@' in str(g.get('opponentName', '')):
                team_neutral_games[team_name] = team_neutral_games.get(team_name, 0) + 1
            else:
                team_home_games[team_name] = team_home_games.get(team_name, 0) + 1

    wp_lookup: dict[str, float] = {}
    for team in team_games:
        total_w = team_wins.get(team, 0)
        total_g = team_games.get(team, 0)
        if total_g == 0:
            wp_lookup[team] = 0.5
            continue
        wp_raw = total_w / total_g
        home_g = team_home_games.get(team, 0)
        away_g = team_away_games.get(team, 0)
        neutral_g = team_neutral_games.get(team, 0)
        if home_g + away_g + neutral_g > 0:
            w_credit = wp_raw * (home_g * 0.7 + away_g * 1.3 + neutral_g * 1.0)
            l_credit = (1 - wp_raw) * (home_g * 1.3 + away_g * 0.7 + neutral_g * 1.0)
            wp_lookup[team] = w_credit / (w_credit + l_credit) if (w_credit + l_credit) > 0 else 0.5
        else:
            wp_lookup[team] = wp_raw

    # Step 3: Compute OWP for all teams
    owp_lookup: dict[str, float] = {}
    for team in wp_lookup:
        opps = team_opponents.get(team, [])
        opp_wps = [wp_lookup.get(o, 0.5) for o in opps if o in wp_lookup]
        owp_lookup[team] = float(np.mean(opp_wps)) if opp_wps else 0.5

    # Step 4: Compute OOWP and RPI for all teams
    rpi_results = []
    for team in wp_lookup:
        wp = wp_lookup[team]
        owp = owp_lookup.get(team, 0.5)
        opps = team_opponents.get(team, [])
        opp_owps = [owp_lookup.get(o, 0.5) for o in opps if o in owp_lookup]
        oowp = float(np.mean(opp_owps)) if opp_owps else 0.5
        pred_rpi = 0.25 * wp + 0.50 * owp + 0.25 * oowp

        total_w = team_wins.get(team, 0)
        total_g = team_games.get(team, 0)
        total_l = total_g - total_w

        rpi_results.append({
            'team': team,
            'proj_wins': round(total_w),
            'proj_losses': round(total_l),
            'proj_wp': round(wp, 3),
            'owp': round(owp, 3),
            'oowp': round(oowp, 3),
            'pred_rpi': round(pred_rpi, 5),
        })

    df = pd.DataFrame(rpi_results)
    df = df.sort_values('pred_rpi', ascending=False).reset_index(drop=True)
    df['pred_rpi_rank'] = range(1, len(df) + 1)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='64 Analytics — Win Generator', layout='wide',
                   initial_sidebar_state='expanded')

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded" rel="stylesheet">
<style>
html, body, .stApp, .stApp *:not([class*="icon"]):not([class*="Icon"]):not([data-testid*="icon"]):not([data-testid*="Icon"]):not([data-testid*="arrow"]):not(.material-icons):not(.material-symbols):not(.material-symbols-rounded){ font-family: 'Inter', sans-serif !important; }
.stApp { background-color: #1a1a1a; }
h1, h2, h3, p, label, .stMarkdown { color: #C8C8C8 !important; }
</style>
""", unsafe_allow_html=True)

st.sidebar.image(str(BRAND_LOGO), width=80)
st.sidebar.markdown('## Win Generator')
if st.sidebar.button('Reload data'):
    st.cache_data.clear()
    st.rerun()

sport = st.sidebar.selectbox('Sport', ['baseball', 'softball'])
year = '2026'

ranks = load_team_ranks(sport, year)
schedules = load_schedules(sport)
records = load_current_records(year)

if len(schedules) == 0:
    st.warning(f'No schedule data found. Run: `npx ts-node scripts/scrape-schedules.ts {sport} {year}`')
    st.stop()

# Division filter
divisions = ['All'] + sorted(ranks['division'].dropna().unique().tolist())
division = st.sidebar.selectbox('Division', divisions)
if division != 'All':
    ranks = ranks[ranks['division'] == division]

# Build name -> rank lookup for matching schedule opponents
name_to_rank = dict(zip(ranks['team_name'], ranks['rank']))

# Build rank_pct map (1 - (rank-1)/(N-1)) for the new compressed-log5 pipeline.
# Ranks here are "True Rank" (mean of up to 4 sources, float). Use dense ranking
# within the current division filter to convert to a percentile.
ranks_sorted = ranks.dropna(subset=['rank']).copy()
ranks_sorted['rank_ordinal'] = ranks_sorted['rank'].rank(method='min', ascending=True)
N_div = len(ranks_sorted)
if N_div >= 2:
    ranks_sorted['rank_pct'] = 1 - (ranks_sorted['rank_ordinal'] - 1) / (N_div - 1)
else:
    ranks_sorted['rank_pct'] = 0.5
rank_pct_map = dict(zip(ranks_sorted['team_name'], ranks_sorted['rank_pct']))

# Build player-talent profiles for the selected sport(+division).
DIV_LABEL_TO_CODE = {'D-I': 'D1', 'D-II': 'D2', 'D-III': 'D3'}
if division == 'All':
    profiles = {}
    for _code in ('D1', 'D2', 'D3'):
        profiles.update(build_team_profiles(sport, _code))
else:
    _code = DIV_LABEL_TO_CODE.get(division, 'D1')
    profiles = build_team_profiles(sport, _code)

# Merge current records
ranks = ranks.merge(records, on='team_id', how='left')
ranks['current_wins'] = ranks['current_wins'].fillna(0).astype(int)
ranks['current_losses'] = ranks['current_losses'].fillna(0).astype(int)

mode = st.sidebar.radio('Mode', ['Team Projection', 'Full Rankings', 'Predicted RPI', 'Single Matchup'], horizontal=False)

if mode == 'Single Matchup':
    st.markdown('### Single Matchup')
    team_list = sorted(ranks['team_name'].dropna().unique())

    c1, c2 = st.columns(2)
    team_a = c1.selectbox('Team A', team_list, key='match_a')
    team_b = c2.selectbox('Team B', team_list, index=min(1, len(team_list)-1), key='match_b')

    if team_a == team_b:
        st.warning('Pick two different teams.')
        st.stop()

    row_a = ranks[ranks['team_name'] == team_a].iloc[0]
    row_b = ranks[ranks['team_name'] == team_b].iloc[0]

    rank_a = int(row_a['rank'])
    rank_b = int(row_b['rank'])

    mc1, mc2 = st.columns(2)
    host = mc1.radio('Home team', [team_a, team_b], horizontal=True, key='match_host')
    gtype = mc2.radio('Game type', ['Weekend', 'Midweek'], horizontal=True, key='match_type')
    gnum = 1
    if gtype == 'Weekend':
        gnum = st.radio('Series game #', [1, 2, 3], horizontal=True, key='match_gnum')

    is_home_for_a = (host == team_a)
    prob_a = compute_matchup_wp(
        home_name=team_a if is_home_for_a else team_b,
        away_name=team_b if is_home_for_a else team_a,
        is_home_for_selected=is_home_for_a,
        game_type=gtype, game_num=gnum,
        rank_pct_map=rank_pct_map, profiles=profiles,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(f'{team_a} (True Rank #{rank_a})', f'{prob_a:.1%} win prob')
    c2.metric('vs', '')
    c3.metric(f'{team_b} (True Rank #{rank_b})', f'{1-prob_a:.1%} win prob')

elif mode == 'Team Projection':
    st.markdown('### Team Win Projection')
    team_list = sorted(ranks['team_name'].dropna().unique())
    selected_team = st.selectbox('Select team', team_list)

    team_row = ranks[ranks['team_name'] == selected_team].iloc[0]
    team_rank = int(team_row['rank'])
    current_w = int(team_row['current_wins'])
    current_l = int(team_row['current_losses'])

    # Show True Rank breakdown
    rank_parts = []
    for label, col in [('64A', 'rank_64a'), ('RPI', 'rank_rpi'), ('Massey', 'rank_massey'), ('DSR', 'rank_dsr')]:
        val = team_row.get(col)
        if pd.notna(val):
            rank_parts.append(f'{label}: #{int(val)}')
    st.caption(f'True Rank components: {" | ".join(rank_parts)} → **True Rank: #{team_rank}**')

    # Get future games by team name
    team_sched = schedules[
        (schedules['teamName'] == selected_team) &
        ((schedules['result'].isna()) | (schedules['result'] == ''))
    ]

    if len(team_sched) == 0:
        st.info(f'No future games found for {selected_team}. Season may be complete.')
        st.stop()

    # Compute per-game WP using the full pipeline (compressed log5 + HFA
    # + 50/50 team-rank/player blend, with auto-detected Weekend/Midweek).
    win_probs = []
    unranked_opps = []
    game_details = []  # parallel list, for the Remaining-schedule expander
    for _, game in team_sched.iterrows():
        opp_name = str(game.get('opponentName', '')).split('@')[0].strip()
        is_home_for_selected = not (pd.notna(game.get('isAway')) and game.get('isAway') == 1.0)
        if is_home_for_selected:
            home, away = selected_team, opp_name
        else:
            home, away = opp_name, selected_team
        gtype, gnum = detect_game_context(game['date'], opp_name, team_sched)
        # If either team has no rank_pct, fall back to 50%
        if home not in rank_pct_map or away not in rank_pct_map:
            wp = 0.5
            if opp_name not in rank_pct_map:
                unranked_opps.append(opp_name)
        else:
            wp = compute_matchup_wp(home, away, is_home_for_selected,
                                     gtype, gnum, rank_pct_map, profiles)
        win_probs.append(wp)
        game_details.append({'opp': opp_name, 'is_home': is_home_for_selected,
                              'game_type': gtype, 'game_num': gnum, 'wp': wp,
                              'date': game.get('date')})

    projection = project_season_probs(win_probs, current_w, current_l)

    # Display
    st.markdown(f'**{selected_team}** — Rank #{team_rank}')
    st.markdown(f'Current record: **{current_w}-{current_l}** | Remaining games: **{projection["remaining_games"]}**')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Projected Wins', projection['expected_wins'])
    c2.metric('Projected Losses', projection['expected_losses'])
    c3.metric('Projected Win%', f"{projection['expected_win_pct']:.3f}")
    c4.metric('Remaining W', f"+{projection['projected_remaining_wins']}")

    st.markdown(f"""
    **80% Confidence Range**: {projection['win_low']}-{projection['win_high']} wins
    **Full Range**: {projection['win_floor']}-{projection['win_ceiling']} wins
    """)

    # Remaining schedule detail — uses per-game WPs already computed above
    with st.expander('Remaining schedule'):
        sched_rows = []
        for d, (_, game) in zip(game_details, team_sched.iterrows()):
            opp_name = d['opp']
            opp_rank = name_to_rank.get(opp_name)
            venue = 'Home' if d['is_home'] else 'Away'
            if '@' in str(game.get('opponentName', '')):
                venue = 'Neutral'
            type_label = f"Wknd G{d['game_num']}" if d['game_type'] == 'Weekend' else 'Midweek'
            sched_rows.append({
                'Date': d['date'],
                'Opponent': opp_name,
                'Venue': venue,
                'Type': type_label,
                'Opp Rank': int(opp_rank) if opp_rank is not None and not np.isnan(opp_rank) else 'N/A',
                'Win Prob': f'{d["wp"]:.1%}',
            })
        st.dataframe(pd.DataFrame(sched_rows), use_container_width=True, hide_index=True)

    if unranked_opps:
        st.caption(f'{len(set(unranked_opps))} opponents without True Rank (assumed median): {", ".join(set(unranked_opps))}')

elif mode == 'Full Rankings':
    st.markdown('### Projected Season Standings')
    st.caption('Projecting remaining games using True Rank (64A + RPI + Massey + DSR) matchup model (1,000 simulations per team)')

    with st.spinner('Running projections...'):
        results = []
        for _, team_row in ranks.iterrows():
            team_name = team_row['team_name']
            team_rank = int(team_row['rank'])
            current_w = int(team_row['current_wins'])
            current_l = int(team_row['current_losses'])

            # Get future games
            team_sched = schedules[
                (schedules['teamName'] == team_name) &
                ((schedules['result'].isna()) | (schedules['result'] == ''))
            ]

            # Per-game WP with the new compressed-log5 + blend model
            win_probs = []
            for _, game in team_sched.iterrows():
                opp_name = str(game.get('opponentName', '')).split('@')[0].strip()
                is_home = not (pd.notna(game.get('isAway')) and game.get('isAway') == 1.0)
                home = team_name if is_home else opp_name
                away = opp_name if is_home else team_name
                gtype, gnum = detect_game_context(game['date'], opp_name, team_sched)
                if home in rank_pct_map and away in rank_pct_map:
                    wp = compute_matchup_wp(home, away, is_home, gtype, gnum,
                                             rank_pct_map, profiles)
                else:
                    wp = 0.5
                win_probs.append(wp)

            if len(win_probs) == 0:
                results.append({
                    'Team': team_name, 'Rank': team_rank,
                    'Current': f'{current_w}-{current_l}',
                    'Proj W': current_w, 'Proj L': current_l,
                    'Proj Win%': current_w / max(current_w + current_l, 1),
                    'Remaining': 0, 'Range': f'{current_w}',
                })
                continue

            proj = project_season_probs(win_probs, current_w, current_l, n_simulations=500)
            results.append({
                'Team': team_name, 'Rank': team_rank,
                'Current': f'{current_w}-{current_l}',
                'Proj W': proj['expected_wins'], 'Proj L': proj['expected_losses'],
                'Proj Win%': proj['expected_win_pct'],
                'Remaining': proj['remaining_games'],
                'Range': f"{proj['win_low']}-{proj['win_high']}",
            })

    df = pd.DataFrame(results).sort_values('Proj Win%', ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = '#'

    st.dataframe(df, use_container_width=True)

    csv_buf = df.to_csv()
    st.download_button('Download CSV', data=csv_buf,
                      file_name=f'win_projections_{sport}_{year}.csv', mime='text/csv')

elif mode == 'Predicted RPI':
    st.markdown('### Predicted RPI Rankings')
    st.caption('Each remaining game predicted W/L using True Rank Log5, then full RPI formula applied to projected records.')

    from pages._shared_rpi import compute_predicted_rpi_for_bracketology

    with st.spinner('Computing predicted RPI (all D1 teams)...'):
        pred_rpi_df = compute_predicted_rpi_for_bracketology(sport, DATA_DIR)

    if len(pred_rpi_df) == 0:
        st.warning('No data available.')
        st.stop()

    # Merge with current RPI for comparison
    rpi_file = DATA_DIR / f'{sport}_rpi_D1.csv'
    if rpi_file.exists():
        current_rpi = pd.read_csv(rpi_file, low_memory=False)
        current_rpi_lookup = dict(zip(current_rpi['teamName'], current_rpi['rank']))
        pred_rpi_df['current_rpi_rank'] = pred_rpi_df['team'].map(current_rpi_lookup)
        pred_rpi_df['rpi_delta'] = pred_rpi_df['current_rpi_rank'] - pred_rpi_df['final_rank']

    # Further filter by division if selected
    if division != 'All':
        div_teams = set(ranks['team_name'])
        pred_rpi_df = pred_rpi_df[pred_rpi_df['team'].isin(div_teams)]

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Teams Ranked', len(pred_rpi_df))
    if 'current_rpi_rank' in pred_rpi_df.columns:
        top_riser = pred_rpi_df.dropna(subset=['rpi_delta']).nlargest(1, 'rpi_delta')
        top_faller = pred_rpi_df.dropna(subset=['rpi_delta']).nsmallest(1, 'rpi_delta')
        if len(top_riser) > 0:
            r = top_riser.iloc[0]
            c2.metric('Biggest Riser', r['team'], delta=f"+{int(r['rpi_delta'])} spots")
        if len(top_faller) > 0:
            r = top_faller.iloc[0]
            c3.metric('Biggest Faller', r['team'], delta=f"{int(r['rpi_delta'])} spots")

    # Display table
    st.markdown('---')
    display_cols = ['final_rank', 'team', 'true_rank_d1', 'pred_rpi_rank', 'proj_wins', 'proj_losses', 'pred_rpi']
    col_names = {
        'final_rank': 'Final Rank', 'team': 'Team',
        'true_rank_d1': 'True Rank', 'pred_rpi_rank': 'Pred RPI Rank',
        'proj_wins': 'Proj W', 'proj_losses': 'Proj L', 'pred_rpi': 'Pred RPI',
    }
    if 'current_rpi_rank' in pred_rpi_df.columns:
        display_cols.insert(2, 'current_rpi_rank')
        display_cols.insert(3, 'rpi_delta')
        col_names['current_rpi_rank'] = 'Current RPI'
        col_names['rpi_delta'] = 'Delta'
    available_cols = [c for c in display_cols if c in pred_rpi_df.columns]
    display = pred_rpi_df[available_cols].rename(columns=col_names)
    st.dataframe(display, use_container_width=True, hide_index=True, height=1050)

    csv_buf = display.to_csv(index=False)
    st.download_button('Download Predicted RPI CSV', data=csv_buf,
                      file_name=f'predicted_rpi_{sport}_{year}.csv', mime='text/csv')
