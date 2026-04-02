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

mode = st.sidebar.radio('Mode', ['Team Projection', 'Full Rankings', 'Single Matchup'], horizontal=False)

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

else:  # Full Rankings
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
