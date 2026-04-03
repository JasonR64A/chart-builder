"""
64 Analytics -- Bracketology
NCAA Tournament field prediction with seeding, regional placement, and super regional pairings.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

# ── Path setup ───────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _APP_DIR / 'data'
BRACKET_DIR = DATA_DIR / 'bracketology'
LOGO_DIR = _APP_DIR / 'team_logos_512'
BRAND_LOGO = _APP_DIR / 'assets' / 'brand_logo_dark.png'

RED = '#C41230'

# ── Page config & style ─────────────────────────────────────────────────────
st.set_page_config(
    page_title='64 Analytics \u2014 Bracketology',
    layout='wide',
    initial_sidebar_state='expanded',
)
st.markdown('''
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, .stApp, .stApp *:not([class*="icon"]):not([data-testid*="icon"]) {
    font-family: 'Inter', sans-serif !important;
}
.stApp { background-color: #1a1a1a; }
h1, h2, h3, p, label, .stMarkdown { color: #C8C8C8 !important; }
div[data-testid="stSidebar"] { background-color: #222222; }
</style>
''', unsafe_allow_html=True)


# ── Haversine ────────────────────────────────────────────────────────────────
def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3959
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_conferences():
    df = pd.read_csv(DATA_DIR / 'conferences.csv')
    return df


@st.cache_data(show_spinner=False)
def load_teams():
    df = pd.read_csv(DATA_DIR / 'teams.csv')
    return df


@st.cache_data(show_spinner=False)
def load_schedules():
    df = pd.read_csv(DATA_DIR / 'schedules.csv')
    return df


@st.cache_data(show_spinner=False)
def load_team_rank():
    df = pd.read_csv(DATA_DIR / 'team_rank.csv')
    return df


@st.cache_data(show_spinner=False)
def load_team_locations():
    df = pd.read_csv(BRACKET_DIR / 'team_locations.csv')
    return df


@st.cache_data(show_spinner=False)
def load_rpi(sport):
    fname = 'baseball_rpi_D1.csv' if sport == 'baseball' else 'softball_rpi_D1.csv'
    df = pd.read_csv(DATA_DIR / fname)
    return df


@st.cache_data(show_spinner=False)
def load_historical_brackets():
    df = pd.read_csv(BRACKET_DIR / 'historical_brackets.csv')
    return df


@st.cache_data(show_spinner=False)
def load_historical_rpi():
    df = pd.read_csv(BRACKET_DIR / 'historical_selection_rpi.csv')
    return df


@st.cache_data(show_spinner=False)
def load_previous_snapshot(sport):
    """Load the most recent bracketology snapshot for movement comparison.
    Returns a dict: team_name -> {overall_seed, seed_tier, national_seed}
    """
    snapshot_dir = BRACKET_DIR / 'snapshots'
    if not snapshot_dir.exists():
        return {}
    # Find all snapshot files for this sport, sorted by date descending
    import glob
    pattern = str(snapshot_dir / f'{sport}_bracketology_2*.csv')
    files = sorted(glob.glob(pattern), reverse=True)
    if not files:
        return {}
    # Load the most recent snapshot
    df = pd.read_csv(files[0])
    lookup = {}
    for _, row in df.iterrows():
        lookup[row['name']] = {
            'overall_seed': int(row.get('overall_seed', 0)),
            'seed_tier': str(row.get('seed_tier', '')),
            'national_seed': row.get('national_seed', None),
        }
    return lookup


def compute_movement(team_name, current_seed, current_tier, previous_snapshot):
    """Compute movement indicator for a team.
    Returns (direction, label) where direction is 'up', 'down', or None.
    'up' means the team improved (lower seed number = better).
    """
    if not previous_snapshot or team_name not in previous_snapshot:
        return None, ''
    prev = previous_snapshot[team_name]
    prev_seed = prev['overall_seed']

    if current_seed < prev_seed:
        diff = prev_seed - current_seed
        return 'up', f'+{diff}'
    elif current_seed > prev_seed:
        diff = current_seed - prev_seed
        return 'down', f'-{diff}'
    return None, ''


def movement_html(direction, label):
    """Return HTML for a movement arrow indicator."""
    if direction == 'up':
        return (f'<span style="color:#4CAF50;font-size:11px;font-weight:700;margin-left:6px;">'
                f'&#9650; {label}</span>')
    elif direction == 'down':
        return (f'<span style="color:#F44336;font-size:11px;font-weight:700;margin-left:6px;">'
                f'&#9660; {label}</span>')
    return ''


# ── Conference abbreviation mapping (RPI conference names -> conferences.csv) ─
def build_conf_abbrev_map(conferences_df):
    """Map RPI conference strings to conference ids."""
    abbrev_to_id = {}
    for _, row in conferences_df.iterrows():
        abbrev_to_id[row['abbreviation']] = row['id']
        abbrev_to_id[row['name']] = row['id']
    # Manual overrides for RPI data discrepancies
    rpi_to_abbrev = {
        'American': 'AAC',
        'Mountain West': 'MW',
        'Big Sky': 'Big Sky',
    }
    for rpi_name, abbrev in rpi_to_abbrev.items():
        if abbrev in abbrev_to_id:
            abbrev_to_id[rpi_name] = abbrev_to_id[abbrev]
    return abbrev_to_id


# ── Name normalization for matching across datasets ──────────────────────────
def normalize_name(name):
    """Normalize team name for matching across datasets."""
    if not isinstance(name, str):
        return ''
    # Strip (AQ) suffix from historical RPI data
    n = name.strip()
    if n.endswith('(AQ)'):
        n = n[:-4].strip()
    # Common variations
    replacements = {
        'Lamar University': 'Lamar',
    }
    return replacements.get(n, n)


# ── Build the 64-team field ─────────────────────────────────────────────────
def build_field(sport):
    """
    Returns (field_df, auto_bids_df, at_large_df, bubble_in, bubble_out)
    field_df has columns: teamName, conference, rpi, conf_win_pct, is_auto_bid,
                          team_id, logo_url, lat, lon, sixty_four_rank, record_str
    """
    conferences = load_conferences()
    teams = load_teams()
    schedules = load_schedules()
    team_rank = load_team_rank()
    locations = load_team_locations()
    rpi_df = load_rpi(sport)

    # Filter to D-I conferences
    di_confs = conferences[conferences['division'] == 'D-I']
    di_conf_ids = set(di_confs['id'].tolist())

    # Build conference abbreviation -> id map
    conf_abbrev_map = build_conf_abbrev_map(di_confs)

    # Filter teams to sport
    sport_label = 'Baseball' if sport == 'baseball' else 'Softball'
    sport_teams = teams[teams['sport'] == sport_label].copy()

    # Filter to D-I teams
    sport_teams = sport_teams[sport_teams['conference_id'].isin(di_conf_ids)].copy()

    # Build conference abbreviation lookup from conference id
    conf_id_to_abbrev = dict(zip(di_confs['id'], di_confs['abbreviation']))
    conf_id_to_name = dict(zip(di_confs['id'], di_confs['name']))

    # Get 2026 schedules
    sched_2026 = schedules[schedules['Year'] == 2026].copy()

    # Merge schedule with sport_teams
    team_sched = sport_teams.merge(sched_2026, left_on='id', right_on='team_id', how='inner')

    # Calculate conference win percentage
    team_sched['conf_games'] = team_sched['Conf_Win'] + team_sched['Conf_Loss']
    team_sched['conf_win_pct'] = np.where(
        team_sched['conf_games'] > 0,
        team_sched['Conf_Win'] / team_sched['conf_games'],
        0.0
    )
    # Total record
    team_sched['total_wins'] = team_sched['Conf_Win'] + team_sched['OOC_Win'] + team_sched['Post_Win']
    team_sched['total_losses'] = team_sched['Conf_Loss'] + team_sched['OOC_Loss'] + team_sched['Post_Loss']
    team_sched['record_str'] = team_sched['total_wins'].fillna(0).astype(int).astype(str) + '-' + team_sched['total_losses'].fillna(0).astype(int).astype(str)

    # Merge RPI data by team name matching
    rpi_df = rpi_df.copy()
    rpi_df['teamName_clean'] = rpi_df['teamName'].apply(normalize_name)

    # Build name -> rpi lookup
    rpi_lookup = dict(zip(rpi_df['teamName_clean'], rpi_df['rpi']))
    rpi_rank_lookup = dict(zip(rpi_df['teamName_clean'], rpi_df['rank']))
    rpi_conf_lookup = dict(zip(rpi_df['teamName_clean'], rpi_df['conference']))

    team_sched['rpi'] = team_sched['name'].map(rpi_lookup)
    team_sched['rpi_rank'] = team_sched['name'].map(rpi_rank_lookup)
    team_sched['rpi_conference'] = team_sched['name'].map(rpi_conf_lookup)

    # Get 64 Rank for 2026
    tr_2026 = team_rank[team_rank['year'] == 2026][['team_id', 'sixty_four_rank_weighted_run_efficiency']].copy()
    tr_2026 = tr_2026.rename(columns={'sixty_four_rank_weighted_run_efficiency': 'sixty_four_rank'})
    # team_rank.team_id -> teams.id
    team_sched = team_sched.merge(
        tr_2026, left_on='id_x', right_on='team_id', how='left', suffixes=('', '_tr')
    )

    # Merge locations
    locations_clean = locations.copy()
    locations_clean['team_name_clean'] = locations_clean['team_name'].apply(normalize_name)
    loc_lookup_lat = dict(zip(locations_clean['team_name_clean'], locations_clean['lat']))
    loc_lookup_lon = dict(zip(locations_clean['team_name_clean'], locations_clean['lon']))
    team_sched['lat'] = team_sched['name'].map(loc_lookup_lat)
    team_sched['lon'] = team_sched['name'].map(loc_lookup_lon)

    # ── Auto-bids: best conference win% per D-I conference ──
    # Only consider teams that have played conference games
    eligible = team_sched[team_sched['conf_games'] > 0].copy()

    auto_bids = []
    for conf_id in di_conf_ids:
        conf_teams = eligible[eligible['conference_id'] == conf_id]
        if len(conf_teams) == 0:
            continue
        # Best conference win pct, break ties with RPI
        conf_teams_sorted = conf_teams.sort_values(
            ['conf_win_pct', 'rpi'], ascending=[False, False]
        )
        best = conf_teams_sorted.iloc[0]
        auto_bids.append(best['name'])

    auto_bids_set = set(auto_bids)

    # ── Pull Predicted RPI directly from Win Generator computation ──
    try:
        from pages._shared_rpi import compute_predicted_rpi_for_bracketology
        pred_rpi_df = compute_predicted_rpi_for_bracketology(sport, DATA_DIR)
    except Exception as e:
        st.error(f'Failed to compute predicted RPI: {e}')
        pred_rpi_df = pd.DataFrame()

    if len(pred_rpi_df) > 0:
        pred_rpi_lookup = dict(zip(pred_rpi_df['team'], pred_rpi_df['pred_rpi']))
        pred_rpi_rank_lookup = dict(zip(pred_rpi_df['team'], pred_rpi_df['pred_rpi_rank']))
        final_rank_lookup = dict(zip(pred_rpi_df['team'], pred_rpi_df['final_rank']))
        proj_w_lookup = dict(zip(pred_rpi_df['team'], pred_rpi_df['proj_wins']))
        proj_l_lookup = dict(zip(pred_rpi_df['team'], pred_rpi_df['proj_losses']))

        team_sched['projected_rpi'] = team_sched['name'].map(pred_rpi_lookup)
        team_sched['projected_rpi_rank'] = team_sched['name'].map(final_rank_lookup).fillna(999).astype(int)
        team_sched['proj_record_str'] = team_sched['name'].apply(
            lambda n: f"{int(proj_w_lookup.get(n, 0))}-{int(proj_l_lookup.get(n, 0))}"
        )
    else:
        team_sched['projected_rpi'] = team_sched['rpi']
        team_sched['projected_rpi_rank'] = 999
        team_sched['proj_record_str'] = team_sched['record_str']

    team_sched['rpi_display'] = team_sched['rpi']

    # ── At-large: fill remaining spots from best PROJECTED RPI ──
    total_field = 64
    n_auto = len(auto_bids_set)
    n_at_large = total_field - n_auto

    # All teams with RPI, not auto-bid, sorted by Final Rank (projected_rpi_rank)
    at_large_pool = team_sched[
        (~team_sched['name'].isin(auto_bids_set)) &
        (team_sched['rpi'].notna())
    ].sort_values('projected_rpi_rank', ascending=True)

    at_large_teams = at_large_pool.head(n_at_large)['name'].tolist()

    # Bubble
    bubble_in = at_large_pool.head(n_at_large).tail(4)['name'].tolist()  # last 4 in
    bubble_out = at_large_pool.iloc[n_at_large:n_at_large + 4]['name'].tolist()  # first 4 out

    # Full field
    field_names = set(auto_bids_set) | set(at_large_teams)
    field_df = team_sched[team_sched['name'].isin(field_names)].copy()
    field_df['is_auto_bid'] = field_df['name'].isin(auto_bids_set)

    # Use rpi_conference if available (from RPI data), else use conference_id
    field_df['display_conference'] = field_df.apply(
        lambda r: r['rpi_conference'] if pd.notna(r.get('rpi_conference')) else conf_id_to_abbrev.get(r['conference_id'], ''),
        axis=1
    )

    # Logo URL
    field_df['logo_url'] = field_df['logo_url']
    field_df['team_db_id'] = field_df['id_x']

    # Ensure we have exactly 64 (or fewer if data is short)
    field_df = field_df.head(64)

    # Build bubble dataframes
    bubble_in_df = team_sched[team_sched['name'].isin(bubble_in)].copy()
    bubble_in_df['display_conference'] = bubble_in_df.apply(
        lambda r: r['rpi_conference'] if pd.notna(r.get('rpi_conference')) else conf_id_to_abbrev.get(r['conference_id'], ''),
        axis=1
    )
    bubble_in_df['is_auto_bid'] = bubble_in_df['name'].isin(auto_bids_set)

    bubble_out_df = team_sched[team_sched['name'].isin(bubble_out)].copy()
    bubble_out_df['display_conference'] = bubble_out_df.apply(
        lambda r: r['rpi_conference'] if pd.notna(r.get('rpi_conference')) else conf_id_to_abbrev.get(r['conference_id'], ''),
        axis=1
    )
    bubble_out_df['is_auto_bid'] = bubble_out_df['name'].isin(auto_bids_set)

    # Auto-bids dataframe for display
    auto_bids_df = team_sched[team_sched['name'].isin(auto_bids_set)].copy()
    auto_bids_df['display_conference'] = auto_bids_df.apply(
        lambda r: r['rpi_conference'] if pd.notna(r.get('rpi_conference')) else conf_id_to_abbrev.get(r['conference_id'], ''),
        axis=1
    )
    auto_bids_df['conf_name'] = auto_bids_df['conference_id'].map(conf_id_to_name)

    return field_df, auto_bids_df, bubble_in_df, bubble_out_df


# ── Seed the field ───────────────────────────────────────────────────────────
def seed_field(field_df):
    """
    Assign national seeds 1-16 (hosts) and place 2/3/4 seeds in regionals.
    Returns list of 16 regional dicts, each with keys:
        national_seed, host, seed_1, seed_2, seed_3, seed_4
    Each seed entry is a dict: {name, rpi, record_str, conference, lat, lon, logo_url, team_db_id, distance}
    """
    df = field_df.copy()

    # Sort by projected_rpi_rank (matches Win Generator Predicted RPI exactly)
    # Rank 1-16 = national seeds/hosts, 17-32 = 2 seeds, 33-48 = 3 seeds, 49-64 = 4 seeds
    if 'projected_rpi_rank' in df.columns:
        df = df.sort_values('projected_rpi_rank', ascending=True).reset_index(drop=True)
    elif 'projected_rpi' in df.columns:
        df = df.sort_values('projected_rpi', ascending=False).reset_index(drop=True)
    else:
        df = df.sort_values('rpi', ascending=False, na_position='last').reset_index(drop=True)

    def team_dict(row, distance=0.0):
        return {
            'name': row.get('name', ''),
            'rpi': row.get('rpi', 0.0),
            'projected_rpi': row.get('projected_rpi', row.get('rpi', 0.0)),
            'projected_rpi_rank': int(row.get('projected_rpi_rank', 0)) if pd.notna(row.get('projected_rpi_rank')) else 0,
            'record_str': row.get('record_str', ''),
            'proj_record_str': row.get('proj_record_str', row.get('record_str', '')),
            'sixty_four_rank': row.get('sixty_four_rank', 0.0),
            'conference': row.get('display_conference', ''),
            'lat': row.get('lat', 0.0),
            'lon': row.get('lon', 0.0),
            'logo_url': row.get('logo_url', ''),
            'team_db_id': int(row.get('team_db_id', 0)) if pd.notna(row.get('team_db_id')) else 0,
            'distance': distance,
            'adjusted_rpi': row.get('adjusted_rpi', 0.0),
            'is_auto_bid': row.get('is_auto_bid', False),
        }

    # Seeds 1-16 are the top 16 teams (regional hosts / 1-seeds)
    hosts = []
    for i in range(min(16, len(df))):
        row = df.iloc[i]
        hosts.append(team_dict(row))

    # Seeds 17-32 are the next 16 teams (2-seeds)
    two_seeds_pool = []
    for i in range(16, min(32, len(df))):
        row = df.iloc[i]
        two_seeds_pool.append(team_dict(row))

    # Remaining 32 teams are 3/4 seeds
    remaining_pool = []
    for i in range(32, len(df)):
        row = df.iloc[i]
        remaining_pool.append(team_dict(row))

    # ── Build conflict sets: conference + head-to-head opponents ──
    # NCAA avoids placing conference opponents or teams that played each
    # other during the season in the same regional.
    team_conference = {}
    teams_db = load_teams()
    confs_db = load_conferences()
    conf_name_by_id = dict(zip(confs_db['id'], confs_db['name']))
    for _, row in teams_db.iterrows():
        cid = row.get('conference_id')
        if pd.notna(cid):
            team_conference[row['name']] = conf_name_by_id.get(int(cid), '')

    # Head-to-head: load full schedule to find who played whom
    opponents_map: dict[str, set[str]] = {}  # team_name -> set of opponent names
    for sport_file in ['schedules_full_baseball.csv', 'schedules_full_softball.csv']:
        sched_path = DATA_DIR / sport_file
        if sched_path.exists():
            sched = pd.read_csv(sched_path, low_memory=False)
            for _, row in sched.iterrows():
                tn = str(row.get('teamName', ''))
                opp = str(row.get('opponentName', ''))
                if tn and opp:
                    # Clean opponent name (remove venue suffixes like "@City, ST")
                    opp_clean = opp.split('@')[0].strip().rstrip()
                    opponents_map.setdefault(tn, set()).add(opp_clean)
                    opponents_map.setdefault(opp_clean, set()).add(tn)

    def has_conflict(team_name, regional_teams):
        """Check if team_name conflicts with any team already in the regional."""
        t_conf = team_conference.get(team_name, '')
        t_opps = opponents_map.get(team_name, set())
        for rt in regional_teams:
            rn = rt.get('name', '')
            # Same conference
            if t_conf and t_conf == team_conference.get(rn, ''):
                return True
            # Played each other
            if rn in t_opps:
                return True
        return False

    # ── Place 2-seeds: global optimization to minimize total rank distance ──
    # Natural pairing: seed 1 gets pool[15] (rank 32), seed 16 gets pool[0] (rank 17)
    # But conference/opponent conflicts require swaps. Instead of greedy sequential
    # assignment (which cascades badly with SEC-heavy top 16), find the globally
    # optimal assignment that minimizes total rank distance while respecting conflicts.

    n_hosts = min(16, len(hosts))
    n_pool = min(16, len(two_seeds_pool))

    # Build cost matrix: cost[host_idx][pool_idx] = rank distance penalty
    # Natural pairing has 0 cost, deviations add cost proportional to rank distance
    # Conflicts get infinite cost
    INF_COST = 10000
    cost = []
    for h_idx in range(n_hosts):
        row = []
        natural_pool_idx = (n_pool - 1) - h_idx  # seed 1 -> pool[15], seed 16 -> pool[0]
        for p_idx in range(n_pool):
            # Check conflict
            if has_conflict(two_seeds_pool[p_idx]['name'], [hosts[h_idx]]):
                row.append(INF_COST)
            else:
                # Cost = absolute distance from natural pairing position
                row.append(abs(p_idx - natural_pool_idx))
        cost.append(row)

    # Solve assignment problem using Hungarian algorithm (scipy) or brute force for 16x16
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        assignment = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
    except ImportError:
        # Fallback: greedy from most constrained host first
        assignment = {}
        used_pools = set()
        # Sort hosts by how many valid (non-infinite) options they have (most constrained first)
        host_options = [(h_idx, sum(1 for c in cost[h_idx] if c < INF_COST)) for h_idx in range(n_hosts)]
        host_options.sort(key=lambda x: x[1])
        for h_idx, _ in host_options:
            natural = (n_pool - 1) - h_idx
            # Sort pool indices by cost for this host
            ranked = sorted(range(n_pool), key=lambda p: cost[h_idx][p])
            for p_idx in ranked:
                if p_idx not in used_pools and cost[h_idx][p_idx] < INF_COST:
                    assignment[h_idx] = p_idx
                    used_pools.add(p_idx)
                    break
            if h_idx not in assignment:
                # All conflict — take cheapest unused
                for p_idx in ranked:
                    if p_idx not in used_pools:
                        assignment[h_idx] = p_idx
                        used_pools.add(p_idx)
                        break

    # Build regionals from assignment
    regionals = []
    for idx in range(n_hosts):
        host = hosts[idx]
        pool_idx = assignment.get(idx, (n_pool - 1) - idx)
        seed2 = two_seeds_pool[pool_idx].copy()
        if pd.notna(host.get('lat')) and pd.notna(seed2.get('lat')):
            seed2['distance'] = round(haversine_miles(host['lat'], host['lon'], seed2['lat'], seed2['lon']), 0)
        else:
            seed2['distance'] = 0
        seed2['national_seed_num'] = 17 + pool_idx

        regionals.append({
            'national_seed': idx + 1,
            'host': host,
            'seed_1': host,
            'seed_2': seed2,
            'seed_3': None,
            'seed_4': None,
        })

    # ── Place 3/4 seeds ──
    # Sort remaining pool by RPI ascending (worst first).
    # The 16 worst RPI teams become 4-seeds, next 16 become 3-seeds.
    # Within each tier, assign to the closest host that doesn't create
    # a conference or head-to-head conflict.
    remaining_pool.sort(key=lambda x: x.get('rpi', 0))

    # Power conference teams (ACC, Big 12, Big Ten, SEC, Sun Belt) are never
    # 4-seeds. Those 16 spots go to non-power conference teams.
    POWER_CONFS = {'ACC', 'Big 12', 'Big Ten', 'SEC', 'Sun Belt'}
    non_power = [t for t in remaining_pool if t.get('conference', '') not in POWER_CONFS]
    power = [t for t in remaining_pool if t.get('conference', '') in POWER_CONFS]

    # 4-seeds: worst 16 RPIs from non-power teams
    four_seed_pool = non_power[:16]
    # 3-seeds: all power conference remainders + leftover non-power
    three_seed_pool = power + non_power[16:]
    three_seed_pool.sort(key=lambda x: x.get('rpi', 0), reverse=True)  # best RPI first

    for seed_key, pool in [('seed_3', three_seed_pool), ('seed_4', four_seed_pool)]:
        # Use Hungarian algorithm for 3/4 seeds too — minimizes distance while
        # respecting conference/opponent conflicts (no 2 SEC teams in same regional)
        n_reg = len(regionals)
        n_pool = len(pool)
        if n_reg > 0 and n_pool > 0:
            try:
                from scipy.optimize import linear_sum_assignment
                cost_34 = []
                for r_idx, reg in enumerate(regionals):
                    if reg.get(seed_key) is not None:
                        # Already assigned — make entire row infinite
                        cost_34.append([10000] * n_pool)
                        continue
                    regional_teams = [reg['seed_1'], reg['seed_2']]
                    if reg.get('seed_3'):
                        regional_teams.append(reg['seed_3'])
                    if reg.get('seed_4'):
                        regional_teams.append(reg['seed_4'])
                    row = []
                    for p_idx, tm in enumerate(pool):
                        conflict = has_conflict(tm['name'], regional_teams)
                        if pd.notna(reg['host'].get('lat')) and pd.notna(tm.get('lat')):
                            dist = haversine_miles(reg['host']['lat'], reg['host']['lon'], tm['lat'], tm['lon'])
                        else:
                            dist = 2000
                        # Cost: distance + heavy penalty for conflicts
                        row.append(dist + (50000 if conflict else 0))
                    cost_34.append(row)
                row_ind, col_ind = linear_sum_assignment(cost_34)
                for ri, ci in zip(row_ind, col_ind):
                    if cost_34[ri][ci] >= 50000:
                        continue  # skip conflict assignments (shouldn't happen with enough options)
                    entry = pool[ci].copy()
                    if pd.notna(regionals[ri]['host'].get('lat')) and pd.notna(entry.get('lat')):
                        entry['distance'] = round(haversine_miles(
                            regionals[ri]['host']['lat'], regionals[ri]['host']['lon'],
                            entry['lat'], entry['lon']), 0)
                    else:
                        entry['distance'] = 0
                    regionals[ri][seed_key] = entry
                continue  # skip the greedy fallback below
            except ImportError:
                pass  # fall through to greedy

        assigned = set()
        for reg in regionals:
            regional_teams = [reg['seed_1'], reg['seed_2']]
            if reg['seed_3']:
                regional_teams.append(reg['seed_3'])
            if reg['seed_4']:
                regional_teams.append(reg['seed_4'])

            # Priority: within 400mi and no conflict > within 400mi with conflict >
            # outside 400mi no conflict > outside 400mi with conflict
            candidates = []
            for j, tm in enumerate(pool):
                if j in assigned:
                    continue
                if pd.notna(reg['host'].get('lat')) and pd.notna(tm.get('lat')):
                    dist = haversine_miles(reg['host']['lat'], reg['host']['lon'], tm['lat'], tm['lon'])
                else:
                    dist = 3000
                conflict = has_conflict(tm['name'], regional_teams)
                within_400 = dist <= 400
                # Sort key: (has conflict, not within 400, distance)
                # No conflict always beats conflict, then prefer within 400mi
                sort_key = (conflict, not within_400, dist)
                candidates.append((j, dist, sort_key, tm))

            candidates.sort(key=lambda x: x[2])
            best_j, best_dist, _, _ = candidates[0] if candidates else (0, 3000, (True, True, 3000), None)

            assigned.add(best_j)
            entry = pool[best_j].copy()
            entry['distance'] = round(best_dist, 0)
            reg[seed_key] = entry

    return regionals


# ── Build super regional pairings ────────────────────────────────────────────
def build_supers(regionals):
    """
    Pair regionals into super regionals: 1v16, 2v15, 3v14, ..., 8v9.
    Returns list of 8 tuples: (higher_seed_regional, lower_seed_regional).
    """
    # Sort by national seed to be safe
    by_seed = sorted(regionals, key=lambda r: r['national_seed'])
    pairings = []
    for i in range(min(8, len(by_seed))):
        high = by_seed[i]
        low_idx = 15 - i
        if low_idx < len(by_seed):
            low = by_seed[low_idx]
        else:
            low = None
        pairings.append((high, low))
    return pairings


# ── Render functions ─────────────────────────────────────────────────────────
def team_logo_path(team_db_id):
    """Return the local path to a team logo, or None."""
    p = LOGO_DIR / f'{team_db_id}.png'
    return str(p) if p.exists() else None


def render_team_row_html(team, seed_num, is_host=False, move_direction=None, move_label=''):
    """Return HTML for a single team row inside a regional card."""
    if team is None:
        return '<div style="padding:6px 10px;color:#666;">TBD</div>'

    name = team.get('name', 'TBD')
    rpi = team.get('rpi', 0)
    proj_rpi_rank = team.get('projected_rpi_rank', 0)
    rpi_str = f'{rpi:.5f}' if isinstance(rpi, (int, float)) and pd.notna(rpi) else 'N/A'
    proj_rpi_rank_str = f'#{proj_rpi_rank}' if proj_rpi_rank > 0 else 'N/A'
    record = team.get('record_str', '')
    proj_record = team.get('proj_record_str', record)
    sixty_four = team.get('sixty_four_rank', 0)
    sixty_four_str = f'{sixty_four:.3f}' if isinstance(sixty_four, (int, float)) and pd.notna(sixty_four) else 'N/A'
    conf = team.get('conference', '')
    dist = team.get('distance', 0)
    logo_url = team.get('logo_url', '')

    dist_html = ''
    if not is_host and dist and dist > 0:
        dist_html = f'<span style="color:#888;font-size:11px;margin-left:6px;">{int(dist)} mi</span>'

    host_badge = ''
    if is_host:
        host_badge = '<span style="background:#C41230;color:#fff;font-size:9px;padding:1px 5px;border-radius:3px;margin-left:6px;">HOST</span>'

    move_arrow = movement_html(move_direction, move_label)

    logo_html = ''
    if logo_url:
        logo_html = f'<img src="{logo_url}" style="width:24px;height:24px;border-radius:4px;margin-right:8px;vertical-align:middle;object-fit:contain;" onerror="this.style.display=\'none\'">'

    # Row highlight for movement
    bg_color = ''
    if move_direction == 'up':
        bg_color = 'background:rgba(76,175,80,0.08);'
    elif move_direction == 'down':
        bg_color = 'background:rgba(244,67,54,0.08);'

    return f'''
    <div style="display:flex;align-items:center;padding:6px 10px;border-bottom:1px solid #333;{bg_color}">
        <div style="min-width:24px;font-weight:700;color:{RED};font-size:16px;margin-right:8px;">{seed_num}</div>
        {logo_html}
        <div style="flex:1;">
            <div style="color:#e0e0e0;font-weight:600;font-size:13px;">
                {name}{host_badge}{dist_html}{move_arrow}
            </div>
            <div style="color:#999;font-size:11px;">
                {conf} &middot; {record} &middot; RPI: {rpi_str} &middot; 64A: {sixty_four_str}
            </div>
            <div style="color:#6dbf6d;font-size:10px;">
                Proj: {proj_record} &middot; Proj RPI: {proj_rpi_rank_str}
            </div>
        </div>
    </div>
    '''


def render_regional_card(regional, previous_snapshot=None):
    """Return full HTML for a regional card."""
    if regional is None:
        return '<div style="background:#252525;border-radius:10px;padding:16px;text-align:center;color:#666;">No data</div>'

    ns = regional['national_seed']
    host = regional['host']
    host_name = host.get('name', 'TBD')
    logo_url = host.get('logo_url', '')

    logo_html = ''
    if logo_url:
        logo_html = f'<img src="{logo_url}" style="width:40px;height:40px;border-radius:6px;margin-right:12px;object-fit:contain;" onerror="this.style.display=\'none\'">'

    def _team_move(team, current_overall_seed):
        if team is None or not previous_snapshot:
            return None, ''
        return compute_movement(
            team.get('name', ''), current_overall_seed,
            '', previous_snapshot
        )

    rows_html = ''
    s1 = regional.get('seed_1')
    s2 = regional.get('seed_2')
    s3 = regional.get('seed_3')
    s4 = regional.get('seed_4')

    # Overall seed: host is national_seed, seed_2 has national_seed_num, etc.
    s1_overall = ns
    s2_overall = s2.get('national_seed_num', 0) if s2 else 0
    s3_overall = s3.get('national_seed_num', 0) if s3 else 0
    s4_overall = s4.get('national_seed_num', 0) if s4 else 0

    d1, l1 = _team_move(s1, s1_overall)
    d2, l2 = _team_move(s2, s2_overall)
    d3, l3 = _team_move(s3, s3_overall)
    d4, l4 = _team_move(s4, s4_overall)

    rows_html += render_team_row_html(s1, 1, is_host=True, move_direction=d1, move_label=l1)
    rows_html += render_team_row_html(s2, 2, move_direction=d2, move_label=l2)
    rows_html += render_team_row_html(s3, 3, move_direction=d3, move_label=l3)
    rows_html += render_team_row_html(s4, 4, move_direction=d4, move_label=l4)

    return f'''
    <div style="background:#252525;border-radius:10px;overflow:hidden;border:1px solid #333;margin-bottom:8px;">
        <div style="display:flex;align-items:center;padding:12px 14px;background:#1e1e1e;border-bottom:2px solid {RED};">
            <div style="font-size:28px;font-weight:800;color:{RED};margin-right:12px;min-width:36px;text-align:center;">
                {ns}
            </div>
            {logo_html}
            <div>
                <div style="color:#e0e0e0;font-weight:700;font-size:15px;">{host_name} Regional</div>
                <div style="color:#888;font-size:11px;">National Seed #{ns}</div>
            </div>
        </div>
        {rows_html}
    </div>
    '''


def render_bubble_card(team_row, label_prefix='', previous_snapshot=None, current_seed=0):
    """Render a bubble team card from a DataFrame row."""
    name = team_row.get('name', '')
    rpi = team_row.get('rpi', 0)
    rpi_str = f'{rpi:.5f}' if isinstance(rpi, (int, float)) and pd.notna(rpi) else 'N/A'
    record = team_row.get('record_str', '')
    conf = team_row.get('display_conference', team_row.get('rpi_conference', ''))
    is_ab = team_row.get('is_auto_bid', False)
    logo_url = team_row.get('logo_url', '')

    ab_badge = ''
    if is_ab:
        ab_badge = '<span style="background:#2a6e2a;color:#fff;font-size:9px;padding:1px 5px;border-radius:3px;margin-left:6px;">AUTO-BID</span>'

    move_direction, move_label = None, ''
    if previous_snapshot and current_seed > 0:
        move_direction, move_label = compute_movement(name, current_seed, '', previous_snapshot)
    move_arrow = movement_html(move_direction, move_label)

    logo_html = ''
    if isinstance(logo_url, str) and logo_url:
        logo_html = f'<img src="{logo_url}" style="width:28px;height:28px;border-radius:4px;margin-right:10px;object-fit:contain;" onerror="this.style.display=\'none\'">'

    bg_color = ''
    if move_direction == 'up':
        bg_color = 'background:rgba(76,175,80,0.12);'
    elif move_direction == 'down':
        bg_color = 'background:rgba(244,67,54,0.12);'

    return f'''
    <div style="display:flex;align-items:center;padding:10px 14px;background:#252525;border-radius:8px;margin-bottom:6px;border:1px solid #333;{bg_color}">
        {logo_html}
        <div style="flex:1;">
            <div style="color:#e0e0e0;font-weight:600;font-size:13px;">{name}{ab_badge}{move_arrow}</div>
            <div style="color:#999;font-size:11px;">{conf} &middot; {record} &middot; RPI: {rpi_str}</div>
        </div>
    </div>
    '''


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    if BRAND_LOGO.exists():
        st.image(str(BRAND_LOGO), width=200)
    st.markdown('---')
    st.markdown('### Bracketology Settings')

    sport = st.selectbox('Sport', ['Baseball', 'Softball'], index=0)
    sport_key = sport.lower()

    scenario_mode = st.toggle('Scenario Mode', value=False, help='Override model predictions with custom field selections.')

    if st.button('Refresh / Reset', use_container_width=True):
        st.cache_data.clear()
        for key in list(st.session_state.keys()):
            if key.startswith('scenario_'):
                del st.session_state[key]
        st.rerun()

    st.markdown('---')
    st.markdown(
        '<div style="color:#888;font-size:11px;text-align:center;">'
        'Bracketology model by 64 Analytics.<br>'
        'Predictions based on RPI, 64 Rank, and geographic placement.'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Main content ─────────────────────────────────────────────────────────────
st.markdown(
    f'<h1 style="text-align:center;margin-bottom:0;">NCAA {sport} Bracketology</h1>'
    f'<p style="text-align:center;color:#888;margin-top:4px;">2026 Tournament Field Projection</p>',
    unsafe_allow_html=True,
)

# Build the field
try:
    field_df, auto_bids_df, bubble_in_df, bubble_out_df = build_field(sport_key)
except Exception as e:
    st.error(f'Error building field: {e}')
    import traceback
    st.code(traceback.format_exc())
    st.stop()


# ── Section 1: Summary metrics ──────────────────────────────────────────────
st.markdown('---')
m1, m2, m3, m4 = st.columns(4)

n_field = len(field_df)
n_auto = int(field_df['is_auto_bid'].sum()) if 'is_auto_bid' in field_df.columns else 0
n_at_large = n_field - n_auto
avg_rpi = field_df['rpi'].mean() if field_df['rpi'].notna().any() else 0

def metric_card(label, value, color=RED):
    return f'''
    <div style="background:#252525;border-radius:10px;padding:18px 16px;text-align:center;border-top:3px solid {color};">
        <div style="font-size:28px;font-weight:800;color:#e0e0e0;">{value}</div>
        <div style="font-size:12px;color:#999;margin-top:4px;text-transform:uppercase;letter-spacing:1px;">{label}</div>
    </div>
    '''

with m1:
    st.markdown(metric_card('Teams in Field', n_field), unsafe_allow_html=True)
with m2:
    st.markdown(metric_card('Auto-Bids', n_auto, '#2a6e2a'), unsafe_allow_html=True)
with m3:
    st.markdown(metric_card('At-Large', n_at_large, '#4A7FB5'), unsafe_allow_html=True)
with m4:
    st.markdown(metric_card('Avg RPI', f'{avg_rpi:.5f}', '#888'), unsafe_allow_html=True)


# ── Scenario Mode ───────────────────────────────────────────────────────────
if scenario_mode:
    st.markdown('---')
    st.markdown('## Scenario Mode')
    st.markdown(
        '<p style="color:#999;font-size:13px;">Add or remove teams from the projected field. '
        'Changes are reflected in the bracket below.</p>',
        unsafe_allow_html=True,
    )

    # All D1 teams for selection
    conferences = load_conferences()
    teams_all = load_teams()
    sport_label = 'Baseball' if sport_key == 'baseball' else 'Softball'
    di_conf_ids = set(conferences[conferences['division'] == 'D-I']['id'].tolist())
    all_d1_teams = sorted(
        teams_all[(teams_all['sport'] == sport_label) & (teams_all['conference_id'].isin(di_conf_ids))]['name'].tolist()
    )

    current_field = sorted(field_df['name'].tolist())

    scenario_field = st.multiselect(
        'Teams in Field (add/remove)',
        options=all_d1_teams,
        default=current_field,
        key='scenario_field_select',
    )

    if st.button('Reset to Model', key='reset_scenario'):
        st.session_state['scenario_field_select'] = current_field
        st.rerun()

    # Rebuild field_df with scenario overrides
    if set(scenario_field) != set(current_field):
        # Reconstruct from the full team data
        schedules = load_schedules()
        team_rank_data = load_team_rank()
        locations = load_team_locations()
        rpi_data = load_rpi(sport_key)

        override_teams = teams_all[
            (teams_all['sport'] == sport_label) &
            (teams_all['name'].isin(scenario_field))
        ].copy()

        sched_2026 = schedules[schedules['Year'] == 2026].copy()
        override_teams = override_teams.merge(sched_2026, left_on='id', right_on='team_id', how='left')

        override_teams['conf_games'] = override_teams['Conf_Win'].fillna(0) + override_teams['Conf_Loss'].fillna(0)
        override_teams['conf_win_pct'] = np.where(
            override_teams['conf_games'] > 0,
            override_teams['Conf_Win'] / override_teams['conf_games'], 0.0
        )
        override_teams['total_wins'] = override_teams['Conf_Win'].fillna(0) + override_teams['OOC_Win'].fillna(0) + override_teams['Post_Win'].fillna(0)
        override_teams['total_losses'] = override_teams['Conf_Loss'].fillna(0) + override_teams['OOC_Loss'].fillna(0) + override_teams['Post_Loss'].fillna(0)
        override_teams['record_str'] = override_teams['total_wins'].astype(int).astype(str) + '-' + override_teams['total_losses'].astype(int).astype(str)

        rpi_data_c = rpi_data.copy()
        rpi_data_c['teamName_clean'] = rpi_data_c['teamName'].apply(normalize_name)
        rpi_lookup = dict(zip(rpi_data_c['teamName_clean'], rpi_data_c['rpi']))
        rpi_conf_lookup = dict(zip(rpi_data_c['teamName_clean'], rpi_data_c['conference']))

        override_teams['rpi'] = override_teams['name'].map(rpi_lookup)
        override_teams['rpi_conference'] = override_teams['name'].map(rpi_conf_lookup)

        tr_2026 = team_rank_data[team_rank_data['year'] == 2026][['team_id', 'sixty_four_rank_weighted_run_efficiency']].copy()
        tr_2026 = tr_2026.rename(columns={'sixty_four_rank_weighted_run_efficiency': 'sixty_four_rank'})
        override_teams = override_teams.merge(tr_2026, left_on='id_x', right_on='team_id', how='left', suffixes=('', '_tr'))

        locations_clean = locations.copy()
        locations_clean['team_name_clean'] = locations_clean['team_name'].apply(normalize_name)
        loc_lat = dict(zip(locations_clean['team_name_clean'], locations_clean['lat']))
        loc_lon = dict(zip(locations_clean['team_name_clean'], locations_clean['lon']))
        override_teams['lat'] = override_teams['name'].map(loc_lat)
        override_teams['lon'] = override_teams['name'].map(loc_lon)

        conf_id_to_abbrev = dict(zip(
            conferences[conferences['division'] == 'D-I']['id'],
            conferences[conferences['division'] == 'D-I']['abbreviation']
        ))
        override_teams['display_conference'] = override_teams.apply(
            lambda r: r['rpi_conference'] if pd.notna(r.get('rpi_conference')) else conf_id_to_abbrev.get(r['conference_id'], ''),
            axis=1
        )
        override_teams['is_auto_bid'] = override_teams['name'].isin(auto_bids_df['name'].tolist())
        override_teams['team_db_id'] = override_teams['id_x']

        field_df = override_teams.head(64)


# ── Section 2: Bracket ──────────────────────────────────────────────────────
st.markdown('---')
st.markdown('## Projected Bracket')
st.markdown(
    '<p style="color:#999;font-size:13px;">16 regionals paired into 8 super regionals. '
    'Teams seeded by adjusted score (80% RPI percentile + 20% 64 Rank).</p>',
    unsafe_allow_html=True,
)

# Load previous snapshot for movement indicators
prev_snapshot = load_previous_snapshot(sport_key)

regionals = seed_field(field_df)
supers = build_supers(regionals)


# Display order for Omaha bracket view:
# Super 1 vs Super 8, Super 4 vs Super 5, Super 2 vs Super 7, Super 3 vs Super 6
# This mirrors the CWS bracket where these winners would play each other
omaha_order = [0, 7, 3, 4, 1, 6, 2, 5]  # indices into supers list

for display_idx, super_idx in enumerate(omaha_order):
    if super_idx >= len(supers):
        continue
    high, low = supers[super_idx]
    i = super_idx  # original super regional number

    st.markdown(
        f'<div style="background:#1e1e1e;border-radius:8px;padding:8px 14px;margin:16px 0 8px 0;'
        f'border-left:4px solid {RED};">'
        f'<span style="color:{RED};font-weight:700;font-size:14px;">SUPER REGIONAL {i + 1}</span>'
        f'<span style="color:#666;font-size:12px;margin-left:12px;">'
        f'Seed {high["national_seed"]} vs Seed {low["national_seed"] if low else "TBD"}'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(render_regional_card(high, previous_snapshot=prev_snapshot), unsafe_allow_html=True)
    with col_right:
        st.markdown(render_regional_card(low, previous_snapshot=prev_snapshot), unsafe_allow_html=True)


# ── Bracket PNG export ─────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from io import BytesIO

def render_bracket_png(supers_data, sport_label, bg_path, brand_path):
    """Render the full bracket as a branded PNG with 64A backdrop."""
    fig = plt.figure(figsize=(20, 28), facecolor='#2c2c2a')

    # Background pattern
    if bg_path.exists():
        bg_img = Image.open(bg_path).convert('RGB')
        bg_ax = fig.add_axes([-0.05, -0.05, 1.1, 1.1])
        bg_ax.imshow(np.array(bg_img), aspect='auto', extent=[0, 1, 0, 1], zorder=0)
        bg_ax.axis('off')

    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 140)
    ax.set_facecolor('none')
    ax.patch.set_alpha(0)
    ax.axis('off')
    ax.invert_yaxis()

    # Title
    ax.text(50, 2, f'NCAA {sport_label} Bracketology', ha='center', va='top',
            fontsize=22, fontweight='bold', color='#e0e0e0', zorder=5)
    ax.text(50, 6, '2026 Tournament Field Projection | 64 Analytics', ha='center', va='top',
            fontsize=11, color='#888888', zorder=5)

    # Brand logo top-right
    if brand_path.exists():
        logo_img = Image.open(brand_path).convert('RGBA')
        logo_img.thumbnail((300, 80), Image.LANCZOS)
        logo_im = OffsetImage(np.array(logo_img), zoom=0.35, alpha=0.8)
        logo_ab = AnnotationBbox(logo_im, (88, 3), xycoords='data', frameon=False, zorder=10)
        ax.add_artist(logo_ab)

    # Draw 8 super regionals, 2 columns of 4
    # Omaha bracket order: 1v8, 4v5, 2v7, 3v6
    png_order = [0, 7, 3, 4, 1, 6, 2, 5]
    for display_idx, si in enumerate(png_order):
        if si >= len(supers_data):
            continue
        high, low = supers_data[si]
        i = si
        col = 0 if display_idx < 4 else 1
        row = display_idx % 4
        x_base = 2 + col * 50
        y_base = 12 + row * 32

        # Super regional header
        ax.add_patch(plt.Rectangle((x_base, y_base), 46, 3, facecolor='#C41230', edgecolor='none', zorder=3, alpha=0.9))
        ax.text(x_base + 1, y_base + 1.5, f'SUPER REGIONAL {i+1}', fontsize=8, fontweight='bold',
                color='white', va='center', zorder=5)
        seed_text = f'Seed {high["national_seed"]} vs Seed {low["national_seed"] if low else "TBD"}'
        ax.text(x_base + 45, y_base + 1.5, seed_text, fontsize=7, color='#ffcccc',
                ha='right', va='center', zorder=5)

        # Two regional cards side by side
        for j, regional in enumerate([high, low]):
            if regional is None:
                continue
            rx = x_base + j * 23
            ry = y_base + 3.5

            # Regional card background
            ax.add_patch(plt.Rectangle((rx, ry), 22, 27, facecolor='#252525',
                                       edgecolor='#3a3a3a', linewidth=0.5, zorder=2, alpha=0.95))

            # National seed badge
            ns = regional['national_seed']
            ax.add_patch(plt.Rectangle((rx, ry), 4, 3, facecolor='#C41230', edgecolor='none', zorder=3))
            ax.text(rx + 2, ry + 1.5, f'#{ns}', fontsize=9, fontweight='bold',
                    color='white', ha='center', va='center', zorder=5)

            # Host name
            host_name = regional['seed_1']['name'] if regional['seed_1'] else 'TBD'
            ax.text(rx + 5, ry + 1.5, f'{host_name} Regional', fontsize=8, fontweight='bold',
                    color='#e0e0e0', va='center', zorder=5)

            # 4 seed rows
            for s_idx, seed_key in enumerate(['seed_1', 'seed_2', 'seed_3', 'seed_4']):
                team = regional.get(seed_key)
                sy = ry + 4 + s_idx * 5.8

                if team is None:
                    ax.text(rx + 1, sy + 2, 'TBD', fontsize=7, color='#666', va='center', zorder=5)
                    continue

                # Seed number
                ax.text(rx + 1.5, sy + 2, str(s_idx + 1), fontsize=10, fontweight='bold',
                        color='#C41230', va='center', zorder=5)

                # Team name
                name = team.get('name', '')
                ax.text(rx + 4, sy + 1.2, name, fontsize=7.5, fontweight='600',
                        color='#e0e0e0', va='center', zorder=5)

                # Conference + record
                conf = team.get('conference', '')
                record = team.get('record_str', '')
                proj_record = team.get('proj_record_str', record)
                rpi_rank = team.get('projected_rpi_rank', '')
                info = f'{conf} | {record}'
                ax.text(rx + 4, sy + 3, info, fontsize=5.5, color='#999', va='center', zorder=5)

                # Projected info
                proj_info = f'Proj: {proj_record} | Proj RPI: #{rpi_rank}'
                ax.text(rx + 4, sy + 4.3, proj_info, fontsize=5, color='#6dbf6d', va='center', zorder=5)

                # Distance (for non-hosts)
                if s_idx > 0:
                    dist = team.get('distance', 0)
                    if dist and dist > 0:
                        ax.text(rx + 21, sy + 1.2, f'{int(dist)}mi', fontsize=5,
                                color='#888', ha='right', va='center', zorder=5)

                # Host badge
                if s_idx == 0:
                    ax.add_patch(plt.Rectangle((rx + 18, sy + 0.5), 3, 1.5, facecolor='#C41230',
                                               edgecolor='none', zorder=3, alpha=0.8))
                    ax.text(rx + 19.5, sy + 1.25, 'HOST', fontsize=4.5, fontweight='bold',
                            color='white', ha='center', va='center', zorder=5)

    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, facecolor='#2c2c2a', edgecolor='none',
                bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return buf


bracket_png = render_bracket_png(
    supers, sport_key.title(),
    _APP_DIR / 'assets' / 'bg_pattern.jpg',
    _APP_DIR / 'assets' / 'brand_logo_dark.png'
)
st.download_button('Download Bracket PNG', data=bracket_png,
                  file_name=f'bracketology_{sport_key}_2026.png', mime='image/png')

# ── Section 3: Bubble ───────────────────────────────────────────────────────
st.markdown('---')
st.markdown('## The Bubble')

b_col1, b_col2 = st.columns(2)

with b_col1:
    st.markdown(
        f'<div style="background:#1e1e1e;border-radius:8px;padding:10px 14px;margin-bottom:10px;'
        f'border-left:4px solid #2a6e2a;">'
        f'<span style="color:#2a6e2a;font-weight:700;font-size:14px;">LAST 4 IN</span></div>',
        unsafe_allow_html=True,
    )
    if len(bubble_in_df) > 0:
        for idx, (_, row) in enumerate(bubble_in_df.iterrows()):
            seed_est = 61 + idx  # last 4 in = seeds 61-64
            st.markdown(render_bubble_card(row, previous_snapshot=prev_snapshot, current_seed=seed_est), unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#666;">No bubble data available.</p>', unsafe_allow_html=True)

with b_col2:
    st.markdown(
        f'<div style="background:#1e1e1e;border-radius:8px;padding:10px 14px;margin-bottom:10px;'
        f'border-left:4px solid {RED};">'
        f'<span style="color:{RED};font-weight:700;font-size:14px;">FIRST 4 OUT</span></div>',
        unsafe_allow_html=True,
    )
    if len(bubble_out_df) > 0:
        for idx, (_, row) in enumerate(bubble_out_df.iterrows()):
            seed_est = 65 + idx  # first 4 out = seeds 65-68
            st.markdown(render_bubble_card(row, previous_snapshot=prev_snapshot, current_seed=seed_est), unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#666;">No bubble data available.</p>', unsafe_allow_html=True)


# ── Section 4: Conference Auto-Bids ─────────────────────────────────────────
st.markdown('---')
st.markdown('## Conference Auto-Bids')
st.markdown(
    '<p style="color:#999;font-size:13px;">Each D-I conference\'s projected automatic qualifier, '
    'determined by best conference win percentage.</p>',
    unsafe_allow_html=True,
)

if len(auto_bids_df) > 0:
    ab_display = auto_bids_df[['name', 'conf_name', 'record_str', 'rpi', 'conf_win_pct']].copy()
    ab_display = ab_display.rename(columns={
        'name': 'Team',
        'conf_name': 'Conference',
        'record_str': 'Record',
        'rpi': 'RPI',
        'conf_win_pct': 'Conf Win%',
    })
    ab_display = ab_display.sort_values('RPI', ascending=False).reset_index(drop=True)
    ab_display.index = ab_display.index + 1
    ab_display['RPI'] = ab_display['RPI'].apply(lambda x: f'{x:.5f}' if pd.notna(x) else 'N/A')
    ab_display['Conf Win%'] = ab_display['Conf Win%'].apply(lambda x: f'{x:.3f}' if pd.notna(x) else 'N/A')

    # Render as styled HTML table
    header = '<tr>' + ''.join(
        f'<th style="padding:8px 12px;text-align:left;color:#999;font-size:11px;'
        f'text-transform:uppercase;letter-spacing:1px;border-bottom:2px solid {RED};">{c}</th>'
        for c in ab_display.columns
    ) + '</tr>'

    rows_html = ''
    for _, row in ab_display.iterrows():
        cells = ''.join(
            f'<td style="padding:6px 12px;color:#ccc;font-size:13px;border-bottom:1px solid #333;">{v}</td>'
            for v in row.values
        )
        rows_html += f'<tr style="background:#252525;">{cells}</tr>'

    table_html = f'''
    <div style="overflow-x:auto;border-radius:10px;border:1px solid #333;">
        <table style="width:100%;border-collapse:collapse;background:#1e1e1e;">
            <thead>{header}</thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    '''
    st.markdown(table_html, unsafe_allow_html=True)
else:
    st.markdown('<p style="color:#666;">No auto-bid data available.</p>', unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown('---')
st.markdown(
    '<div style="text-align:center;color:#555;font-size:11px;padding:20px 0;">'
    '64 Analytics Bracketology &middot; Model uses RPI + 64 Rank with geographic placement &middot; '
    'Updated with latest available data'
    '</div>',
    unsafe_allow_html=True,
)
