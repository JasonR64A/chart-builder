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

# Logistic model params: win_pct = 1 / (1 + exp(A * (rank - B)))
# Fitted on 3,052 team-seasons (2024-2026) with R² = 0.69
LOGISTIC_A = 0.006355505043163222
LOGISTIC_B = 151.44616903215248
RESIDUAL_STD = 0.09449904970441943


# ── Model Functions ───────────────────────────────────────────────────────────
def rank_to_win_pct(rank):
    """Convert 64 integer rank to expected win percentage."""
    return 1.0 / (1.0 + np.exp(LOGISTIC_A * (rank - LOGISTIC_B)))


def log5_win_prob(rank_a, rank_b):
    """Win probability for team A vs team B using Log5 method."""
    pA = np.clip(rank_to_win_pct(rank_a), 0.01, 0.99)
    pB = np.clip(rank_to_win_pct(rank_b), 0.01, 0.99)
    return (pA * (1 - pB)) / (pA * (1 - pB) + pB * (1 - pA))


def project_season(team_rank, opponent_ranks, current_wins, current_losses, n_simulations=1000):
    """
    Project final W-L using Monte Carlo simulation.
    Returns: expected_wins, expected_losses, win_range_low, win_range_high
    """
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

    # Load Massey ranks
    massey_file = DATA_DIR / 'rankings' / f'massey_{sport}.csv'
    if massey_file.exists():
        massey = pd.read_csv(massey_file, low_memory=False)
        massey_lookup = dict(zip(massey['team'], massey['rank']))
        merged['rank_massey'] = merged['team_name'].map(massey_lookup)
    else:
        merged['rank_massey'] = np.nan

    # Load DSR ranks
    dsr_file = DATA_DIR / 'rankings' / f'dsr_{sport}.csv'
    if dsr_file.exists():
        dsr = pd.read_csv(dsr_file, low_memory=False)
        dsr_lookup = dict(zip(dsr['team'], dsr['rank']))
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
    prob_a = log5_win_prob(rank_a, rank_b)

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

    # Get opponent ranks for future games
    opp_ranks = []
    unranked_opps = []
    for _, game in team_sched.iterrows():
        opp_name = str(game.get('opponentName', '')).split('@')[0].strip()
        opp_rank = name_to_rank.get(opp_name)
        if opp_rank is not None and not np.isnan(opp_rank):
            opp_ranks.append(int(opp_rank))
        else:
            # Unknown opponent — use median rank as estimate
            opp_ranks.append(int(LOGISTIC_B))
            unranked_opps.append(opp_name)

    # Run projection
    projection = project_season(team_rank, opp_ranks, current_w, current_l)

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

    # Remaining schedule detail
    with st.expander('Remaining schedule'):
        sched_rows = []
        for _, game in team_sched.iterrows():
            opp_name = str(game.get('opponentName', '?')).split('@')[0].strip()
            opp_rank = name_to_rank.get(opp_name)
            venue = 'Away' if game.get('isAway') == 1.0 else ('Neutral' if '@' in str(game.get('opponentName', '')) else 'Home')
            if opp_rank is not None and not np.isnan(opp_rank):
                prob = log5_win_prob(team_rank, int(opp_rank))
                sched_rows.append({
                    'Date': game['date'],
                    'Opponent': opp_name,
                    'Venue': venue,
                    'Opp Rank': int(opp_rank),
                    'Win Prob': f'{prob:.1%}',
                })
            else:
                sched_rows.append({
                    'Date': game['date'],
                    'Opponent': opp_name,
                    'Venue': venue,
                    'Opp Rank': 'N/A',
                    'Win Prob': '~50%',
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

            opp_ranks = []
            for _, game in team_sched.iterrows():
                opp_name = str(game.get('opponentName', '')).split('@')[0].strip()
                opp_rank = name_to_rank.get(opp_name)
                if opp_rank is not None and not np.isnan(opp_rank):
                    opp_ranks.append(int(opp_rank))
                else:
                    opp_ranks.append(int(LOGISTIC_B))

            if len(opp_ranks) == 0:
                results.append({
                    'Team': team_name, 'Rank': team_rank,
                    'Current': f'{current_w}-{current_l}',
                    'Proj W': current_w, 'Proj L': current_l,
                    'Proj Win%': current_w / max(current_w + current_l, 1),
                    'Remaining': 0, 'Range': f'{current_w}',
                })
                continue

            proj = project_season(team_rank, opp_ranks, current_w, current_l, n_simulations=500)
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

    # Need full name_to_rank for ALL teams (not division-filtered)
    all_ranks = load_team_ranks(sport, year)
    all_name_to_rank = dict(zip(all_ranks['team_name'], all_ranks['rank']))

    with st.spinner('Computing predicted RPI (all teams)...'):
        pred_rpi_df = compute_predicted_rpi(sport, all_ranks, schedules, all_name_to_rank)

    if len(pred_rpi_df) == 0:
        st.warning('No data available.')
        st.stop()

    # Merge with current RPI for comparison
    rpi_file = DATA_DIR / f'{sport}_rpi_D1.csv'
    if rpi_file.exists():
        current_rpi = pd.read_csv(rpi_file, low_memory=False)
        current_rpi_lookup = dict(zip(current_rpi['teamName'], current_rpi['rank']))
        pred_rpi_df['current_rpi_rank'] = pred_rpi_df['team'].map(current_rpi_lookup)
        pred_rpi_df['rpi_delta'] = pred_rpi_df['current_rpi_rank'] - pred_rpi_df['pred_rpi_rank']

    # Filter to D1 teams only (RPI is a D1 concept)
    # Use the full team list to identify D1 teams via conference
    all_teams_db = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False)
    all_confs = pd.read_csv(DATA_DIR / 'conferences.csv', low_memory=False)
    sport_label = 'Baseball' if sport == 'baseball' else 'Softball'
    d1_conf_ids = set(all_confs[(all_confs['division'] == 'D-I') & (all_confs['name'] != 'Big Sky Conference')]['id'])
    d1_team_names = set(all_teams_db[(all_teams_db['sport'] == sport_label) & (all_teams_db['conference_id'].isin(d1_conf_ids))]['name'])
    pred_rpi_df = pred_rpi_df[pred_rpi_df['team'].isin(d1_team_names)]
    # Re-rank after filtering
    pred_rpi_df = pred_rpi_df.sort_values('pred_rpi', ascending=False).reset_index(drop=True)
    pred_rpi_df['pred_rpi_rank'] = range(1, len(pred_rpi_df) + 1)

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
            tr = top_riser.iloc[0]
            c2.metric('Biggest Riser', tr['team'], delta=f"+{int(tr['rpi_delta'])} spots")
        if len(top_faller) > 0:
            tf = top_faller.iloc[0]
            c3.metric('Biggest Faller', tf['team'], delta=f"{int(tf['rpi_delta'])} spots")

    # Display table
    st.markdown('---')
    display_cols = ['pred_rpi_rank', 'team', 'proj_wins', 'proj_losses', 'proj_wp', 'owp', 'pred_rpi']
    col_names = {'pred_rpi_rank': 'Pred RPI Rank', 'team': 'Team', 'proj_wins': 'Proj W',
                 'proj_losses': 'Proj L', 'proj_wp': 'Proj WP', 'owp': 'OWP', 'pred_rpi': 'Pred RPI'}
    if 'current_rpi_rank' in pred_rpi_df.columns:
        display_cols.insert(2, 'current_rpi_rank')
        display_cols.insert(3, 'rpi_delta')
        col_names['current_rpi_rank'] = 'Current RPI'
        col_names['rpi_delta'] = 'Delta'
    display = pred_rpi_df[display_cols].rename(columns=col_names)
    st.dataframe(display, use_container_width=True, hide_index=True, height=1050)

    csv_buf = display.to_csv(index=False)
    st.download_button('Download Predicted RPI CSV', data=csv_buf,
                      file_name=f'predicted_rpi_{sport}_{year}.csv', mime='text/csv')
