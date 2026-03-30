"""
64 Analytics — Series Review
Weekend series recap with AI-generated summaries.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import os

# Load API key from environment
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / '.env')
except:
    pass

import anthropic

# ── Path setup ────────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent.parent
PBP_DIR = _APP_DIR / 'pbp_data'
DATA_DIR = _APP_DIR / 'data'
BRAND_LOGO = _APP_DIR / 'assets' / 'brand_logo_dark.png'

# wOBA weights
WOBA_BB, WOBA_HBP, WOBA_1B, WOBA_2B, WOBA_3B, WOBA_HR = 0.690, 0.722, 0.888, 1.271, 1.616, 2.101
WOBA_SCALE = 1.6
FIP_CONSTANT = 3.0


# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_pbp(sport, division, stat_type):
    filepath = PBP_DIR / sport / f'{stat_type}_pbp_{division}.csv'
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath, low_memory=False)
    if 'playerId' in df.columns:
        df['playerId'] = pd.to_numeric(df['playerId'], errors='coerce').fillna(0).astype(int).astype(str)
    if 'date' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    # Mark starters for pitching
    if stat_type == 'pitching' and 'gameId' in df.columns and 'teamName' in df.columns:
        df['is_starter'] = df.duplicated(subset=['gameId', 'teamName'], keep='first') == False
    return df


@st.cache_data
def load_schedules(sport):
    # Try local scrape path first, then check for bundled copy
    for p in [
        Path(f'C:/Users/sixty/OneDrive/Desktop/scrape_final/output/2026/{sport}/schedules_full.csv'),
        _APP_DIR / 'data' / f'schedules_full_{sport}.csv',
    ]:
        if p.exists():
            return pd.read_csv(p, low_memory=False)
    return None


@st.cache_data
def load_conferences():
    teams = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False)
    confs = pd.read_csv(DATA_DIR / 'conferences.csv', low_memory=False)
    merged = teams.merge(confs[['id', 'name']], left_on='conference_id', right_on='id', suffixes=('', '_conf'))
    return merged


# ── Series Detection ──────────────────────────────────────────────────────────
def find_weekend_series(hitting_df, team_name, end_date=None):
    """Find the most recent weekend series for a team (Thu-Sun window)."""
    if end_date is None:
        end_date = hitting_df['date_parsed'].max()

    team_games = hitting_df[hitting_df['teamName'] == team_name].copy()
    if len(team_games) == 0:
        return None, None, []

    # Look backwards from end_date to find the last series
    team_dates = team_games.groupby('date_parsed')['gameId'].first().reset_index().sort_values('date_parsed', ascending=False)

    if len(team_dates) == 0:
        return None, None, []

    # Find opponents in recent games (last 4 days)
    recent_cutoff = end_date - timedelta(days=4)
    recent = team_games[team_games['date_parsed'] >= recent_cutoff]

    if len(recent) == 0:
        # Try last 7 days
        recent_cutoff = end_date - timedelta(days=7)
        recent = team_games[team_games['date_parsed'] >= recent_cutoff]

    if len(recent) == 0:
        return None, None, []

    # Find the most common opponent in recent games (that's the series opponent)
    # Get opponents from the same games
    recent_game_ids = set(recent['gameId'].unique())
    all_recent = hitting_df[hitting_df['gameId'].isin(recent_game_ids)]
    opponents = all_recent[all_recent['teamName'] != team_name]['teamName'].value_counts()

    if len(opponents) == 0:
        return None, None, []

    # The series opponent is the one with most games
    series_opponent = opponents.index[0]

    # Get all games between these two teams in the recent window
    series_games = all_recent[
        ((all_recent['teamName'] == team_name) | (all_recent['teamName'] == series_opponent))
    ]['gameId'].unique()

    # Filter to games where both teams appear
    valid_games = []
    for gid in series_games:
        game_teams = set(all_recent[all_recent['gameId'] == gid]['teamName'].unique())
        if team_name in game_teams and series_opponent in game_teams:
            valid_games.append(gid)

    return team_name, series_opponent, valid_games


def get_all_weekend_series(hitting_df, end_date=None):
    """Find all weekend series in the data."""
    if end_date is None:
        end_date = hitting_df['date_parsed'].max()

    recent_cutoff = end_date - timedelta(days=4)
    recent = hitting_df[hitting_df['date_parsed'] >= recent_cutoff]

    # Group games by the two teams playing
    series_list = []
    seen = set()

    for gid in recent['gameId'].unique():
        game_teams = sorted(recent[recent['gameId'] == gid]['teamName'].unique())
        if len(game_teams) == 2:
            key = tuple(game_teams)
            if key not in seen:
                seen.add(key)
                # Count games between these two
                team_a, team_b = game_teams
                games = recent[
                    ((recent['teamName'] == team_a) | (recent['teamName'] == team_b))
                ]['gameId'].unique()
                valid = []
                for g in games:
                    gt = set(recent[recent['gameId'] == g]['teamName'].unique())
                    if team_a in gt and team_b in gt:
                        valid.append(g)
                if len(valid) >= 2:  # At least 2 games = a series
                    series_list.append((team_a, team_b, valid))

    return series_list


# ── Stats Computation ─────────────────────────────────────────────────────────
def compute_hitter_stats(df):
    """Compute hitting stats for a set of games."""
    rows = []
    for name, group in df.groupby('playerName'):
        ab = group['ab'].sum(); h = group['h'].sum(); bb = group['bb'].sum()
        hbp = group['hbp'].sum(); sf = group['sf'].sum(); sh = group['sh'].sum()
        tb = group['tb'].sum(); hr = group['hr'].sum()
        doubles = group['doubles'].sum(); triples = group['triples'].sum()
        k = group['k'].sum(); r = group['r'].sum(); rbi = group['rbi'].sum()
        sb = group['sb'].sum()
        singles = h - doubles - triples - hr
        pa = ab + bb + hbp + sf + sh
        if pa == 0:
            continue
        obp = (h + bb + hbp) / (ab + bb + hbp + sf) if (ab + bb + hbp + sf) > 0 else 0
        slg = tb / ab if ab > 0 else 0
        ba = h / ab if ab > 0 else 0
        woba_d = ab + bb + sf + hbp
        woba = (WOBA_BB*bb + WOBA_HBP*hbp + WOBA_1B*singles + WOBA_2B*doubles +
                WOBA_3B*triples + WOBA_HR*hr) / woba_d if woba_d > 0 else 0

        rows.append({
            'Player': name, 'Team': group['teamName'].iloc[0],
            'PA': pa, 'AB': ab, 'H': h, '2B': doubles, '3B': triples,
            'HR': hr, 'RBI': rbi, 'R': r, 'BB': bb, 'K': k, 'SB': sb,
            'BA': round(ba, 3), 'OBP': round(obp, 3), 'SLG': round(slg, 3),
            'OPS': round(obp + slg, 3), 'wOBA': round(woba, 3),
        })
    return pd.DataFrame(rows).sort_values('PA', ascending=False) if rows else pd.DataFrame()


def compute_pitcher_stats(df, sport='baseball'):
    """Compute pitching stats for a set of games."""
    rows = []
    inn_per_game = 7 if sport == 'softball' else 9

    for name, group in df.groupby('playerName'):
        ip_raw = group['ip'].sum()  # baseball notation
        # Convert to actual innings
        whole = np.floor(group['ip'])
        frac = np.round((group['ip'] - whole) * 10).astype(int)
        total_outs = (whole * 3 + frac).sum()
        ip = total_outs / 3

        h = group['h'].sum(); bb = group['bb'].sum(); hb = group['hb'].sum()
        so = group['so'].sum(); hr = group['hrA'].sum(); bf = group['bf'].sum()
        er = group['er'].sum(); r = group['r'].sum()
        games = group['gameId'].nunique()
        gs = int(group['is_starter'].sum()) if 'is_starter' in group.columns else 0

        if bf == 0:
            continue

        era = (er / ip) * inn_per_game if ip > 0 else 0
        fip = ((13*hr + 3*(bb+hb) - 2*so) / ip + FIP_CONSTANT) if ip > 0 else 0
        whip = (bb + h) / ip if ip > 0 else 0
        k_rate = so / bf * 100 if bf > 0 else 0

        ip_display = f"{int(total_outs // 3)}.{int(total_outs % 3)}"

        rows.append({
            'Player': name, 'Team': group['teamName'].iloc[0],
            'App': games, 'GS': gs, 'IP': ip_display,
            'H': h, 'R': r, 'ER': er, 'BB': bb, 'SO': so, 'HR': hr,
            'ERA': round(era, 2), 'FIP': round(fip, 2), 'WHIP': round(whip, 2),
            'K%': round(k_rate, 1),
            'is_starter': gs > 0,
        })
    return pd.DataFrame(rows).sort_values('IP', ascending=False, key=lambda x: x.map(lambda v: float(v.replace('.', '') if isinstance(v, str) else v))) if rows else pd.DataFrame()


# ── AI Summary ────────────────────────────────────────────────────────────────
def generate_series_summary(team_a, team_b, game_results, hitter_stats_a, hitter_stats_b,
                            pitcher_stats_a, pitcher_stats_b, conf_standings, next_opponent,
                            sport='baseball'):
    """Generate an AI-powered series summary using Claude."""
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not api_key:
        return "API key not configured. Set ANTHROPIC_API_KEY in environment."

    # Build context
    games_text = ""
    for i, (date, winner, score_w, score_l, loser) in enumerate(game_results):
        games_text += f"Game {i+1} ({date}): {winner} {score_w}, {loser} {score_l}\n"

    # Series result
    a_wins = sum(1 for _, w, _, _, _ in game_results if w == team_a)
    b_wins = sum(1 for _, w, _, _, _ in game_results if w == team_b)

    # Top hitters
    top_a = hitter_stats_a.head(5).to_string(index=False) if len(hitter_stats_a) > 0 else "No data"
    top_b = hitter_stats_b.head(5).to_string(index=False) if len(hitter_stats_b) > 0 else "No data"

    # Top pitchers
    pitch_a = pitcher_stats_a.head(4).to_string(index=False) if len(pitcher_stats_a) > 0 else "No data"
    pitch_b = pitcher_stats_b.head(4).to_string(index=False) if len(pitcher_stats_b) > 0 else "No data"

    prompt = f"""You are a college {sport} analyst for 64 Analytics. Write a concise, engaging series recap (3-5 paragraphs) for the weekend series between {team_a} and {team_b}.

SERIES RESULTS:
{games_text}
{team_a} won {a_wins} game(s), {team_b} won {b_wins} game(s).

{team_a} TOP HITTERS:
{top_a}

{team_b} TOP HITTERS:
{top_b}

{team_a} PITCHING:
{pitch_a}

{team_b} PITCHING:
{pitch_b}

CONFERENCE STANDINGS:
{conf_standings}

NEXT OPPONENT:
{next_opponent}

Write the recap highlighting:
- Who won the series and the key moments/players
- Standout offensive performances (cite specific stats like going 8-for-12 with 3 HR)
- Pitching performances (ERA, strikeouts, innings pitched)
- Conference standing implications
- What's ahead for both teams

Be specific with numbers. Keep it professional but engaging. Don't use generic filler."""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error generating summary: {e}"


# ── Conference Standings ──────────────────────────────────────────────────────
def get_conference_standings(team_name, sport):
    """Get conference standings for a team's conference."""
    conf_data = load_conferences()
    sched = pd.read_csv(DATA_DIR / 'schedules.csv', low_memory=False)

    team_info = conf_data[(conf_data['name'] == team_name) & (conf_data['sport'] == sport.title())]
    if len(team_info) == 0:
        return "Conference data not available."

    conf_name = team_info['name_conf'].iloc[0]
    conf_teams = conf_data[(conf_data['name_conf'] == conf_name) & (conf_data['sport'] == sport.title())]
    conf_team_ids = set(conf_teams['id'].astype(int))

    standings = []
    for tid in conf_team_ids:
        team_sched = sched[(sched['team_id'] == tid) & (sched['Year'] == 2026)]
        if len(team_sched) == 0:
            continue
        row = team_sched.iloc[0]
        t_name = conf_teams[conf_teams['id'] == tid]['name'].iloc[0]
        conf_w = int(row.get('Conf_Win', 0))
        conf_l = int(row.get('Conf_Loss', 0))
        standings.append((t_name, conf_w, conf_l))

    standings.sort(key=lambda x: (-x[1], x[2]))
    text = f"{conf_name} Standings:\n"
    for i, (name, w, l) in enumerate(standings[:10]):
        marker = " <--" if name == team_name else ""
        text += f"  {i+1}. {name} ({w}-{l}){marker}\n"
    return text


def get_next_opponent(team_name, sport):
    """Get the next opponent from schedule."""
    sched = load_schedules(sport)
    if sched is None:
        return "Schedule not available."

    team_sched = sched[(sched['teamName'] == team_name) & (sched['isFuture'] == 1)]
    if len(team_sched) == 0:
        return f"{team_name} has no upcoming games scheduled."

    # Sort by date and get next
    team_sched = team_sched.sort_values('date')
    next_game = team_sched.iloc[0]
    return f"{team_name} plays {next_game['opponentName']} next on {next_game['date']}."


# ── Game Results ──────────────────────────────────────────────────────────────
def get_game_results(hitting_df, game_ids, team_a, team_b):
    """Get W/L and scores for each game in the series."""
    results = []
    for gid in sorted(game_ids):
        game = hitting_df[hitting_df['gameId'] == gid]
        date = game['date'].iloc[0] if len(game) > 0 else '?'

        a_runs = game[game['teamName'] == team_a]['r'].sum()
        b_runs = game[game['teamName'] == team_b]['r'].sum()

        if a_runs > b_runs:
            results.append((date, team_a, int(a_runs), int(b_runs), team_b))
        else:
            results.append((date, team_b, int(b_runs), int(a_runs), team_a))

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='64 Analytics — Series Review', layout='wide',
                   initial_sidebar_state='expanded')

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, .stApp, .stApp *:not([class*="icon"]):not([data-testid*="icon"]):not(.material-icons):not(.material-symbols) { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: #1a1a1a; }
h1, h2, h3, p, label, .stMarkdown { color: #C8C8C8 !important; }
</style>
""", unsafe_allow_html=True)

st.sidebar.image(str(BRAND_LOGO), width=80)
st.sidebar.markdown('## Series Review')

if st.sidebar.button('Reload data'):
    st.cache_data.clear()
    st.rerun()

# Controls
st.sidebar.markdown('---')
sport = st.sidebar.selectbox('Sport', ['baseball', 'softball'])
division = st.sidebar.selectbox('Division', ['D1', 'D2', 'D3'])

# Load data
hitting = load_pbp(sport, division, 'hitting')
pitching = load_pbp(sport, division, 'pitching')

if hitting is None or pitching is None:
    st.error('PBP data not found.')
    st.stop()

# Selection mode
mode = st.sidebar.radio('Find Series By', ['Select Team', 'Browse Weekend Series'])

if mode == 'Select Team':
    all_teams = sorted(hitting['teamName'].dropna().unique())
    selected_team = st.sidebar.selectbox('Team', all_teams)

    if selected_team:
        team_a, team_b, game_ids = find_weekend_series(hitting, selected_team)
        if team_b:
            st.sidebar.success(f'Found series: {team_a} vs {team_b} ({len(game_ids)} games)')
        else:
            st.warning(f'No recent series found for {selected_team}.')
            st.stop()
else:
    series_list = get_all_weekend_series(hitting)
    if not series_list:
        st.warning('No weekend series found in recent data.')
        st.stop()

    series_options = [f"{a} vs {b} ({len(g)} games)" for a, b, g in series_list]
    selected_idx = st.sidebar.selectbox('Series', range(len(series_options)), format_func=lambda i: series_options[i])
    team_a, team_b, game_ids = series_list[selected_idx]

if not game_ids:
    st.warning('No games found for this series.')
    st.stop()

# Filter data to series games
series_hitting = hitting[hitting['gameId'].isin(game_ids)]
series_pitching = pitching[pitching['gameId'].isin(game_ids)]

# Game results
game_results = get_game_results(series_hitting, game_ids, team_a, team_b)
a_wins = sum(1 for _, w, _, _, _ in game_results if w == team_a)
b_wins = sum(1 for _, w, _, _, _ in game_results if w == team_b)

# Header
if a_wins > b_wins:
    st.markdown(f'### {team_a} wins series {a_wins}-{b_wins} over {team_b}')
elif b_wins > a_wins:
    st.markdown(f'### {team_b} wins series {b_wins}-{a_wins} over {team_a}')
else:
    st.markdown(f'### {team_a} and {team_b} split series {a_wins}-{b_wins}')

# Game scores
for date, winner, score_w, score_l, loser in game_results:
    st.markdown(f'**{date}**: {winner} **{score_w}**, {loser} {score_l}')

st.markdown('---')

# Stats for both teams
col1, col2 = st.columns(2)

with col1:
    st.markdown(f'#### {team_a} — Top 10 Hitters')
    hitters_a = compute_hitter_stats(series_hitting[series_hitting['teamName'] == team_a])
    if len(hitters_a) > 0:
        show_cols = ['Player', 'PA', 'AB', 'H', '2B', '3B', 'HR', 'RBI', 'R', 'BB', 'K', 'SB',
                     'BA', 'OBP', 'SLG', 'OPS', 'wOBA']
        show_cols = [c for c in show_cols if c in hitters_a.columns]
        st.dataframe(hitters_a[show_cols].head(10), use_container_width=True, hide_index=True)

with col2:
    st.markdown(f'#### {team_b} — Top 10 Hitters')
    hitters_b = compute_hitter_stats(series_hitting[series_hitting['teamName'] == team_b])
    if len(hitters_b) > 0:
        show_cols = ['Player', 'PA', 'AB', 'H', '2B', '3B', 'HR', 'RBI', 'R', 'BB', 'K', 'SB',
                     'BA', 'OBP', 'SLG', 'OPS', 'wOBA']
        show_cols = [c for c in show_cols if c in hitters_b.columns]
        st.dataframe(hitters_b[show_cols].head(10), use_container_width=True, hide_index=True)

st.markdown('---')

col3, col4 = st.columns(2)

with col3:
    st.markdown(f'#### {team_a} — Pitching')
    pitchers_a = compute_pitcher_stats(series_pitching[series_pitching['teamName'] == team_a], sport)
    if len(pitchers_a) > 0:
        show_cols = ['Player', 'App', 'GS', 'IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR',
                     'ERA', 'FIP', 'WHIP', 'K%']
        show_cols = [c for c in show_cols if c in pitchers_a.columns]
        st.dataframe(pitchers_a[show_cols].head(8), use_container_width=True, hide_index=True)

with col4:
    st.markdown(f'#### {team_b} — Pitching')
    pitchers_b = compute_pitcher_stats(series_pitching[series_pitching['teamName'] == team_b], sport)
    if len(pitchers_b) > 0:
        show_cols = ['Player', 'App', 'GS', 'IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR',
                     'ERA', 'FIP', 'WHIP', 'K%']
        show_cols = [c for c in show_cols if c in pitchers_b.columns]
        st.dataframe(pitchers_b[show_cols].head(8), use_container_width=True, hide_index=True)

# AI Summary
st.markdown('---')
st.markdown('### Series Recap')

if st.button('Generate AI Summary', type='primary'):
    with st.spinner('Generating recap...'):
        # Get conference standings for both teams
        conf_a = get_conference_standings(team_a, sport)
        conf_b = get_conference_standings(team_b, sport)
        conf_text = f"{team_a}:\n{conf_a}\n\n{team_b}:\n{conf_b}"

        # Get next opponents
        next_a = get_next_opponent(team_a, sport)
        next_b = get_next_opponent(team_b, sport)
        next_text = f"{next_a}\n{next_b}"

        summary = generate_series_summary(
            team_a, team_b, game_results,
            hitters_a, hitters_b, pitchers_a, pitchers_b,
            conf_text, next_text, sport
        )
        st.markdown(summary)
