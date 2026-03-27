"""
64 Analytics — PBP Stat Calculator
Calendar-dependent stats from enriched play-by-play data.
Compute OPS, wRAA, FIP, Allowed OPS for any player/team over any date range.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO

# ── Path setup (works locally and on Streamlit Cloud) ─────────────────────────
_APP_DIR = Path(__file__).resolve().parent.parent
BRAND_LOGO = _APP_DIR / 'assets' / 'brand_logo_dark.png'

# wOBA weights (college-specific)
WOBA_BB  = 0.690
WOBA_HBP = 0.722
WOBA_1B  = 0.888
WOBA_2B  = 1.271
WOBA_3B  = 1.616
WOBA_HR  = 2.101

# FIP constant
FIP_CONSTANT = 3.0

# wOBA scale
WOBA_SCALE = 1.0


# ── Stat Computation ──────────────────────────────────────────────────────────
def compute_hitting_stats(df):
    """Compute hitting stats from summed enriched PBP columns."""
    ab = df['hit_AB'].sum()
    h = df['hit_H'].sum()
    bb = df['hit_BB'].sum()
    hbp = df['hit_HBP'].sum()
    sf = df['hit_SF'].sum()
    sh = df['hit_SH'].sum()
    tb = df['hit_TB'].sum()
    hr = df['hit_HR'].sum()
    doubles = df['hit_2B'].sum()
    triples = df['hit_3B'].sum()
    k = df['hit_K'].sum()
    sb = df['hit_SB'].sum()
    cs = df['hit_CS'].sum()
    gdp = df['hit_GDP'].sum()
    r = df['hit_R'].sum()
    rbi = df['hit_RBI'].sum()
    singles = h - doubles - triples - hr

    pa = ab + bb + hbp + sf + sh

    # OBP
    obp_denom = ab + bb + hbp + sf
    obp = (h + bb + hbp) / obp_denom if obp_denom > 0 else 0

    # SLG
    slg = tb / ab if ab > 0 else 0

    # OPS
    ops = obp + slg

    # BA
    ba = h / ab if ab > 0 else 0

    # wOBA
    woba_denom = ab + bb + sf + hbp  # IBB excluded but we don't track it separately
    woba = (WOBA_BB * bb + WOBA_HBP * hbp + WOBA_1B * singles +
            WOBA_2B * doubles + WOBA_3B * triples + WOBA_HR * hr) / woba_denom if woba_denom > 0 else 0

    return {
        'PA': int(pa), 'AB': int(ab), 'H': int(h), '1B': int(singles),
        '2B': int(doubles), '3B': int(triples), 'HR': int(hr), 'TB': int(tb),
        'R': int(r), 'RBI': int(rbi), 'BB': int(bb), 'HBP': int(hbp),
        'SF': int(sf), 'SH': int(sh), 'K': int(k),
        'SB': int(sb), 'CS': int(cs), 'GDP': int(gdp),
        'BA': round(ba, 3), 'OBP': round(obp, 3), 'SLG': round(slg, 3),
        'OPS': round(ops, 3), 'wOBA': round(woba, 3),
    }


def compute_pitching_stats(df):
    """Compute pitching stats from summed enriched PBP columns."""
    ip_outs = df['pitch_IP_outs'].sum()
    ip = ip_outs / 3
    h = df['pitch_H'].sum()
    bb = df['pitch_BB'].sum()
    hb = df['pitch_HB'].sum()
    so = df['pitch_SO'].sum()
    hr = df['pitch_HR_A'].sum()
    bf = df['pitch_BF'].sum()
    go = df['pitch_GO'].sum()
    fo = df['pitch_FO'].sum()
    wp = df['pitch_WP'].sum()
    bk = df['pitch_Bk'].sum()
    doubles = df['pitch_2B_A'].sum()
    triples = df['pitch_3B_A'].sum()
    singles = h - doubles - triples - hr
    p_oab = df['pitch_P_OAB'].sum()
    sha = df['pitch_SHA'].sum()
    sfa = df['pitch_SFA'].sum()

    # FIP
    fip = ((13 * hr) + (3 * (bb + hb)) - (2 * so)) / ip + FIP_CONSTANT if ip > 0 else 0

    # Allowed OBP
    obp_denom = p_oab + bb + hb + sfa
    obp_against = (h + bb + hb) / obp_denom if obp_denom > 0 else 0

    # Allowed SLG
    tb_against = singles + 2 * doubles + 3 * triples + 4 * hr
    slg_against = tb_against / p_oab if p_oab > 0 else 0

    # Allowed OPS
    ops_against = obp_against + slg_against

    # Allowed BA
    ba_against = h / p_oab if p_oab > 0 else 0

    # K rate
    k_pct = so / bf if bf > 0 else 0
    bb_pct = bb / bf if bf > 0 else 0

    return {
        'IP': round(ip, 1), 'BF': int(bf), 'H': int(h), 'BB': int(bb),
        'HB': int(hb), 'SO': int(so), 'HR': int(hr),
        '2B-A': int(doubles), '3B-A': int(triples),
        'GO': int(go), 'FO': int(fo), 'WP': int(wp), 'Bk': int(bk),
        'FIP': round(fip, 2),
        'BAA': round(ba_against, 3),
        'OBP Against': round(obp_against, 3),
        'SLG Against': round(slg_against, 3),
        'OPS Against': round(ops_against, 3),
        'K%': round(k_pct, 3), 'BB%': round(bb_pct, 3),
    }


def compute_wraa(woba, league_woba, pa):
    """wRAA = ((wOBA - league_wOBA) / wOBA_scale) * PA"""
    return round(((woba - league_woba) / WOBA_SCALE) * pa, 1)


def compute_grouped_hitting(df, group_col, league_woba, min_pa=1):
    """Compute hitting stats grouped by a column."""
    rows = []
    for name, group in df.groupby(group_col):
        stats = compute_hitting_stats(group)
        if stats['PA'] >= min_pa:
            stats['wRAA'] = compute_wraa(stats['wOBA'], league_woba, stats['PA'])
            stats[group_col] = name
            rows.append(stats)
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    cols = [group_col] + [c for c in result.columns if c != group_col]
    return result[cols].sort_values('OPS', ascending=False).reset_index(drop=True)


def compute_grouped_pitching(df, group_col, min_bf=1):
    """Compute pitching stats grouped by a column."""
    rows = []
    for name, group in df.groupby(group_col):
        stats = compute_pitching_stats(group)
        if stats['BF'] >= min_bf:
            stats[group_col] = name
            rows.append(stats)
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    cols = [group_col] + [c for c in result.columns if c != group_col]
    return result[cols].sort_values('FIP', ascending=True).reset_index(drop=True)


# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_enriched_pbp(filepath):
    """Load enriched PBP file."""
    df = pd.read_csv(filepath, low_memory=False)
    # Normalize IDs
    for col in ['playerId', 'pitcherId']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int).astype(str)
    # Parse dates
    if 'date' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='64 Analytics — PBP Stat Calculator', layout='wide',
                   initial_sidebar_state='expanded')

st.markdown("""
<style>
.stApp { background-color: #1a1a1a; }
h1, h2, h3, p, label, .stMarkdown { color: #C8C8C8 !important; }
</style>
""", unsafe_allow_html=True)

st.sidebar.image(str(BRAND_LOGO), width=80)
st.sidebar.markdown('## PBP Stat Calculator')
if st.sidebar.button('Reload data'):
    st.cache_data.clear()
    st.rerun()

# File selection
st.sidebar.markdown('---')
st.sidebar.markdown('### Data Source')
pbp_file = st.sidebar.text_input('Enriched PBP file path',
    value='C:/Users/sixty/OneDrive/Desktop/scrape_final/output/2026/03/20/baseball/play_by_play/baseball_play_by_play_2026_D1_full_enriched.csv')

if not Path(pbp_file).exists():
    st.error(f'File not found: {pbp_file}')
    st.stop()

with st.spinner('Loading PBP data...'):
    pbp = load_enriched_pbp(pbp_file)

st.sidebar.markdown(f'**{len(pbp):,}** events loaded')

# Date range filter
st.sidebar.markdown('---')
st.sidebar.markdown('### Date Range')
if 'date_parsed' in pbp.columns:
    min_date = pbp['date_parsed'].min()
    max_date = pbp['date_parsed'].max()
    if pd.notna(min_date) and pd.notna(max_date):
        date_range = st.sidebar.date_input('Date range',
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(), max_value=max_date.date())
        if len(date_range) == 2:
            start, end = date_range
            pbp_filtered = pbp[(pbp['date_parsed'].dt.date >= start) & (pbp['date_parsed'].dt.date <= end)]
        else:
            pbp_filtered = pbp
    else:
        pbp_filtered = pbp
        st.sidebar.warning('Could not parse dates')
else:
    pbp_filtered = pbp

st.sidebar.markdown(f'**{len(pbp_filtered):,}** events in range')

# View mode
st.sidebar.markdown('---')
st.sidebar.markdown('### View')
view = st.sidebar.radio('Mode', ['Hitter Stats', 'Pitcher Stats'], horizontal=True)

# Team filter
all_batting_teams = sorted(pbp_filtered['battingTeam'].dropna().unique()) if 'battingTeam' in pbp_filtered.columns else []
all_pitching_teams = sorted(pbp_filtered['fieldingTeam'].dropna().unique()) if 'fieldingTeam' in pbp_filtered.columns else []

if view == 'Hitter Stats':
    team_list = ['All'] + all_batting_teams
    selected_team = st.sidebar.selectbox('Team', team_list)
    if selected_team != 'All':
        pbp_filtered = pbp_filtered[pbp_filtered['battingTeam'] == selected_team]
else:
    team_list = ['All'] + all_pitching_teams
    selected_team = st.sidebar.selectbox('Team', team_list)
    if selected_team != 'All':
        pbp_filtered = pbp_filtered[pbp_filtered['fieldingTeam'] == selected_team]

# Min PA/BF
if view == 'Hitter Stats':
    min_threshold = st.sidebar.number_input('Min PA', value=10, min_value=1, step=5)
else:
    min_threshold = st.sidebar.number_input('Min BF', value=10, min_value=1, step=5)

# Player filter
if view == 'Hitter Stats':
    player_col = 'player'
    available_players = sorted(pbp_filtered[player_col].dropna().unique())
else:
    player_col = 'pitcher'
    available_players = sorted(pbp_filtered[player_col].dropna().unique())

selected_players = st.sidebar.multiselect(f'Filter {view.split()[0].lower()}s', available_players,
                                           help='Leave empty for all')
if selected_players:
    pbp_filtered = pbp_filtered[pbp_filtered[player_col].isin(selected_players)]

if len(pbp_filtered) == 0:
    st.warning('No events match your filters.')
    st.stop()

# ── Compute and Display ──────────────────────────────────────────────────────
if view == 'Hitter Stats':
    st.markdown('### Hitter Stats')

    # Compute league wOBA for wRAA
    league_stats = compute_hitting_stats(pbp_filtered)
    league_woba = league_stats['wOBA']

    # Overall summary
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('League wOBA', f"{league_woba:.3f}")
    c2.metric('League OPS', f"{league_stats['OPS']:.3f}")
    c3.metric('Total PA', f"{league_stats['PA']:,}")
    c4.metric('Total H', f"{league_stats['H']:,}")
    c5.metric('Total HR', f"{league_stats['HR']:,}")

    # Per-player table
    st.markdown('---')
    player_stats = compute_grouped_hitting(pbp_filtered, player_col, league_woba, min_pa=min_threshold)

    if len(player_stats) == 0:
        st.info(f'No players meet the {min_threshold} PA minimum.')
    else:
        # Format percentages
        display = player_stats.copy()
        show_cols = [player_col, 'PA', 'AB', 'H', '1B', '2B', '3B', 'HR', 'TB',
                     'R', 'RBI', 'BB', 'HBP', 'K', 'SB', 'CS', 'GDP',
                     'BA', 'OBP', 'SLG', 'OPS', 'wOBA', 'wRAA']
        show_cols = [c for c in show_cols if c in display.columns]
        st.dataframe(display[show_cols], use_container_width=True, hide_index=True)

        csv_buf = display[show_cols].to_csv(index=False)
        st.download_button('Download CSV', data=csv_buf,
                          file_name='pbp_hitting_stats.csv', mime='text/csv')

    # Single player deep dive
    if selected_players and len(selected_players) == 1:
        st.markdown(f'### {selected_players[0]} — Game Log')
        player_data = pbp_filtered[pbp_filtered[player_col] == selected_players[0]]

        if 'date_parsed' in player_data.columns:
            game_log = compute_grouped_hitting(player_data, 'date', league_woba, min_pa=1)
            if len(game_log) > 0:
                game_log = game_log.rename(columns={'date': 'Date'})
                game_log = game_log.sort_values('Date')
                gl_cols = ['Date', 'PA', 'AB', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'K',
                          'BA', 'OBP', 'SLG', 'OPS', 'wOBA', 'wRAA']
                gl_cols = [c for c in gl_cols if c in game_log.columns]
                st.dataframe(game_log[gl_cols], use_container_width=True, hide_index=True)

else:  # Pitcher Stats
    st.markdown('### Pitcher Stats')

    # Overall summary
    overall = compute_pitching_stats(pbp_filtered)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('League FIP', f"{overall['FIP']:.2f}")
    c2.metric('League OPS Against', f"{overall['OPS Against']:.3f}")
    c3.metric('Total BF', f"{overall['BF']:,}")
    c4.metric('Total IP', f"{overall['IP']:.0f}")
    c5.metric('Total SO', f"{overall['SO']:,}")

    # Per-pitcher table
    st.markdown('---')
    pitcher_stats = compute_grouped_pitching(pbp_filtered, player_col, min_bf=min_threshold)

    if len(pitcher_stats) == 0:
        st.info(f'No pitchers meet the {min_threshold} BF minimum.')
    else:
        show_cols = [player_col, 'IP', 'BF', 'H', 'BB', 'HB', 'SO', 'HR',
                     'GO', 'FO', 'WP',
                     'FIP', 'BAA', 'OBP Against', 'SLG Against', 'OPS Against',
                     'K%', 'BB%']
        show_cols = [c for c in show_cols if c in pitcher_stats.columns]
        st.dataframe(pitcher_stats[show_cols], use_container_width=True, hide_index=True)

        csv_buf = pitcher_stats[show_cols].to_csv(index=False)
        st.download_button('Download CSV', data=csv_buf,
                          file_name='pbp_pitching_stats.csv', mime='text/csv')

    # Single pitcher deep dive
    if selected_players and len(selected_players) == 1:
        st.markdown(f'### {selected_players[0]} — Game Log')
        pitcher_data = pbp_filtered[pbp_filtered[player_col] == selected_players[0]]

        if 'date_parsed' in pitcher_data.columns:
            game_log = compute_grouped_pitching(pitcher_data, 'date', min_bf=1)
            if len(game_log) > 0:
                game_log = game_log.rename(columns={'date': 'Date'})
                game_log = game_log.sort_values('Date')
                gl_cols = ['Date', 'IP', 'BF', 'H', 'BB', 'SO', 'HR',
                          'FIP', 'OPS Against']
                gl_cols = [c for c in gl_cols if c in game_log.columns]
                st.dataframe(game_log[gl_cols], use_container_width=True, hide_index=True)
