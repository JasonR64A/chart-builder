"""
64 Analytics — Player Projection Cards
FiveThirtyEight-inspired player cards with percentile bars, key stats,
rolling game-log sparkline, and a season game log table.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# ── Path setup ──────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent.parent
PBP_DIR = _APP_DIR / 'pbp_data'
DATA_DIR = _APP_DIR / 'data'
LOGO_DIR = _APP_DIR / 'team_logos_512'
BRAND_LOGO = _APP_DIR / 'assets' / 'brand_logo_dark.png'

# ── Constants ───────────────────────────────────────────────────────────────
WOBA_BB  = 0.690
WOBA_HBP = 0.722
WOBA_1B  = 0.888
WOBA_2B  = 1.271
WOBA_3B  = 1.616
WOBA_HR  = 2.101
FIP_CONSTANT = 3.0
WOBA_SCALE = 1.6

ACCENT_RED = '#C41230'
CARD_BG = '#252525'
PAGE_BG = '#1a1a1a'


# ── IP conversion ───────────────────────────────────────────────────────────
def baseball_ip_to_outs(ip_col):
    """Convert baseball IP notation (5.2 = 5 innings + 2 outs) to total outs."""
    whole = np.floor(ip_col)
    frac = np.round((ip_col - whole) * 10).astype(int)
    return (whole * 3 + frac).astype(int)


def outs_to_ip_display(total_outs):
    """Convert total outs back to display IP (e.g., 17 outs -> '5.2')."""
    return f"{int(total_outs // 3)}.{int(total_outs % 3)}"


def outs_to_actual_innings(total_outs):
    """Convert total outs to actual innings for rate calculations."""
    return total_outs / 3


def ip_to_outs_scalar(ip_val):
    """Convert a single IP value to outs."""
    whole = int(ip_val)
    frac = round((ip_val - whole) * 10)
    return whole * 3 + frac


# ── Clean position ──────────────────────────────────────────────────────────
def _clean_position(pos):
    if not pos or (isinstance(pos, float) and np.isnan(pos)):
        return pos
    pos = str(pos).strip()
    if pos.startswith('/'):
        parts = pos.lstrip('/').split('/')
        return parts[0] if parts else pos
    return pos.split('/')[0]


# ── Stat computation ────────────────────────────────────────────────────────
def compute_hitting_stats(df):
    """Compute hitting stats from game-level PBP hitting data."""
    ab = df['ab'].sum()
    h = df['h'].sum()
    bb = df['bb'].sum()
    hbp = df['hbp'].sum()
    sf = df['sf'].sum()
    sh = df['sh'].sum()
    tb = df['tb'].sum()
    hr = df['hr'].sum()
    doubles = df['doubles'].sum()
    triples = df['triples'].sum()
    k = df['k'].sum()
    sb = df['sb'].sum()
    cs = df['cs'].sum()
    r = df['r'].sum()
    rbi = df['rbi'].sum()
    singles = h - doubles - triples - hr
    pa = ab + bb + hbp + sf + sh
    ibb = df['ibb'].sum() if 'ibb' in df.columns else 0

    obp_denom = ab + bb + hbp + sf
    obp = (h + bb + hbp) / obp_denom if obp_denom > 0 else 0
    slg = tb / ab if ab > 0 else 0
    ops = obp + slg
    ba = h / ab if ab > 0 else 0

    woba_denom = ab + bb + sf + hbp
    woba = (WOBA_BB * bb + WOBA_HBP * hbp + WOBA_1B * singles +
            WOBA_2B * doubles + WOBA_3B * triples + WOBA_HR * hr) / woba_denom if woba_denom > 0 else 0

    k_pct = (k / pa * 100) if pa > 0 else 0
    bb_pct = (bb / pa * 100) if pa > 0 else 0
    iso = slg - ba
    babip_denom = ab - k - hr + sf
    babip = (h - hr) / babip_denom if babip_denom > 0 else 0

    return {
        'PA': int(pa), 'AB': int(ab), 'H': int(h), '1B': int(singles),
        '2B': int(doubles), '3B': int(triples), 'HR': int(hr), 'TB': int(tb),
        'R': int(r), 'RBI': int(rbi), 'BB': int(bb), 'HBP': int(hbp),
        'SF': int(sf), 'SH': int(sh), 'IBB': int(ibb), 'K': int(k),
        'SB': int(sb), 'CS': int(cs),
        'BA': round(ba, 3), 'OBP': round(obp, 3), 'SLG': round(slg, 3),
        'OPS': round(ops, 3), 'ISO': round(iso, 3), 'BABIP': round(babip, 3),
        'wOBA': round(woba, 3), 'K%': round(k_pct, 1), 'BB%': round(bb_pct, 1),
    }


def compute_pitching_stats(df):
    """Compute pitching stats from game-level PBP pitching data."""
    total_outs = baseball_ip_to_outs(df['ip']).sum()
    ip = outs_to_actual_innings(total_outs)
    h = df['h'].sum()
    r = df['r'].sum()
    bb = df['bb'].sum()
    hb = df['hb'].sum()
    so = df['so'].sum()
    hr = df['hrA'].sum()
    bf = df['bf'].sum()
    er = df['er'].sum()
    doubles = df['doublesA'].sum()
    triples = df['triplesA'].sum()
    singles = h - doubles - triples - hr
    sha = df['sha'].sum()
    sfa = df['sfa'].sum()
    ibb = df['ibb'].sum() if 'ibb' in df.columns else 0

    p_oab = bf - bb - hb - sfa - sha
    fip = ((13 * hr) + (3 * (bb + hb)) - (2 * so)) / ip + FIP_CONSTANT if ip > 0 else 0
    era = (er / ip) * 9 if ip > 0 else 0

    obp_denom = p_oab + bb + hb + sfa
    obp_against = (h + bb + hb) / obp_denom if obp_denom > 0 else 0
    tb_against = singles + 2 * doubles + 3 * triples + 4 * hr
    slg_against = tb_against / p_oab if p_oab > 0 else 0
    ops_against = obp_against + slg_against

    k_pct = (so / bf * 100) if bf > 0 else 0
    bb_pct = (bb / bf * 100) if bf > 0 else 0
    whip = (bb + h) / ip if ip > 0 else 0

    return {
        'IP_display': outs_to_ip_display(total_outs),
        'IP_outs': int(total_outs), 'BF': int(bf),
        'H': int(h), 'R': int(r), 'ER': int(er),
        'BB': int(bb), 'HB': int(hb), 'SO': int(so),
        'HR': int(hr), 'IBB': int(ibb),
        'ERA': round(era, 2), 'FIP': round(fip, 2),
        'OPS Against': round(ops_against, 3),
        'K%': round(k_pct, 1), 'BB%': round(bb_pct, 1),
        'WHIP': round(whip, 2),
    }


def compute_wraa(woba, league_woba, pa):
    return round(((woba - league_woba) / WOBA_SCALE) * pa, 1)


# ── Game-level stat computation (for sparkline) ────────────────────────────
def compute_hitting_game_stats(row):
    """Compute per-game stats from a single PBP row."""
    ab = row['ab']; h = row['h']; bb = row['bb']; hbp = row['hbp']
    sf = row['sf']; sh = row['sh']; tb = row['tb']
    obp_d = ab + bb + hbp + sf
    obp = (h + bb + hbp) / obp_d if obp_d > 0 else 0
    slg = tb / ab if ab > 0 else 0
    return obp + slg


def compute_pitching_game_fip(row):
    """Compute per-game FIP from a single pitching PBP row."""
    outs = ip_to_outs_scalar(row['ip'])
    ip = outs / 3
    hr = row['hrA']; bb = row['bb']; hb = row['hb']; so = row['so']
    if ip > 0:
        return ((13 * hr) + (3 * (bb + hb)) - (2 * so)) / ip + FIP_CONSTANT
    return np.nan


# ── Data loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_division_teams(sport, division):
    """Get set of team names belonging to a specific division via conferences."""
    teams_path = DATA_DIR / 'teams.csv'
    confs_path = DATA_DIR / 'conferences.csv'
    if not teams_path.exists() or not confs_path.exists():
        return None
    teams = pd.read_csv(teams_path, low_memory=False)
    confs = pd.read_csv(confs_path, low_memory=False)
    sport_label = sport.title() if sport != 'softball' else 'Softball'
    div_label = {'D1': 'D-I', 'D2': 'D-II', 'D3': 'D-III'}[division]
    div_conf_ids = set(confs[confs['division'] == div_label]['id'])
    sport_teams = teams[teams['sport'] == sport_label]
    div_teams = sport_teams[sport_teams['conference_id'].isin(div_conf_ids)]
    return set(div_teams['name'].dropna())


@st.cache_data
def load_player_lookup():
    """Build playerId (ncaa_season_id) -> {position, school, classification, team_id} from rosters + players + teams."""
    rosters_path = DATA_DIR / 'rosters.csv'
    players_path = DATA_DIR / 'players.csv'
    teams_path = DATA_DIR / 'teams.csv'
    if not all(p.exists() for p in [rosters_path, players_path, teams_path]):
        return {}
    rosters = pd.read_csv(rosters_path, low_memory=False)
    players = pd.read_csv(players_path, low_memory=False, encoding='latin-1')
    teams = pd.read_csv(teams_path, low_memory=False)

    r = rosters[['player_id', 'player_ncaa_season_id']].dropna().copy()
    r['player_ncaa_season_id'] = r['player_ncaa_season_id'].astype(int).astype(str)
    r['player_id'] = r['player_id'].astype(int)

    p = players[['id', 'position', 'classification', 'team_id']].copy()
    p['id'] = p['id'].astype(int)
    p['team_id'] = pd.to_numeric(p['team_id'], errors='coerce').fillna(0).astype(int)

    t = teams[['id', 'name']].copy()
    t.columns = ['team_db_id', 'school']

    merged = r.merge(p, left_on='player_id', right_on='id')
    merged = merged.merge(t, left_on='team_id', right_on='team_db_id')
    merged = merged.drop_duplicates('player_ncaa_season_id')
    return {
        'position': dict(zip(merged['player_ncaa_season_id'], merged['position'])),
        'school': dict(zip(merged['player_ncaa_season_id'], merged['school'])),
        'classification': dict(zip(merged['player_ncaa_season_id'], merged['classification'])),
        'team_id': dict(zip(merged['player_ncaa_season_id'], merged['team_db_id'])),
    }


@st.cache_data
def load_pbp(sport, division, stat_type):
    """Load a PBP file with division filtering and enrichment."""
    filepath = PBP_DIR / sport / f'{stat_type}_pbp_{division}.csv'
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath, low_memory=False)
    div_teams = load_division_teams(sport, division)
    if div_teams and 'teamName' in df.columns:
        df = df[df['teamName'].isin(div_teams)].copy()
    if 'playerId' in df.columns:
        df['playerId'] = pd.to_numeric(df['playerId'], errors='coerce').fillna(0).astype(int).astype(str)
    if 'playerPosition' in df.columns:
        df['playerPosition'] = df['playerPosition'].apply(_clean_position)
    if 'playerId' in df.columns:
        lookup = load_player_lookup()
        if lookup:
            df['formalPosition'] = df['playerId'].map(lookup.get('position', {}))
            df['school'] = df['playerId'].map(lookup.get('school', {}))
            df['classification'] = df['playerId'].map(lookup.get('classification', {}))
            df['team_db_id'] = df['playerId'].map(lookup.get('team_id', {}))
            if 'teamName' in df.columns:
                df['school'] = df['school'].fillna(df['teamName'])
    if 'date' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    return df


@st.cache_data
def load_team_logo_map():
    """Build team_name -> logo_id mapping from teams.csv."""
    teams_path = DATA_DIR / 'teams.csv'
    if not teams_path.exists():
        return {}
    teams = pd.read_csv(teams_path, low_memory=False)
    teams['id'] = pd.to_numeric(teams['id'], errors='coerce').fillna(0).astype(int)
    bb = teams[teams['sport'] == 'Baseball'][['name', 'id']].drop_duplicates('name')
    name_to_id = dict(zip(bb['name'], bb['id']))
    sb = teams[teams['sport'] == 'Softball'][['name', 'id']].drop_duplicates('name')
    for _, row in sb.iterrows():
        if row['name'] not in name_to_id:
            name_to_id[row['name']] = row['id']
    return name_to_id


# ── Build division-wide stats for percentile computation ───────────────────
@st.cache_data
def build_division_hitting_stats(sport, division, min_pa=30):
    """Compute hitting stats for all qualified hitters in the division."""
    df = load_pbp(sport, division, 'hitting')
    if df is None or df.empty:
        return pd.DataFrame()
    rows = []
    league_ab = df['ab'].sum()
    league_h = df['h'].sum()
    league_bb = df['bb'].sum()
    league_hbp = df['hbp'].sum()
    league_sf = df['sf'].sum()
    league_sh = df['sh'].sum()
    league_pa = league_ab + league_bb + league_hbp + league_sf + league_sh
    league_r = df['r'].sum()
    league_r_pa = league_r / league_pa if league_pa > 0 else 0

    # League wOBA
    lg_singles = league_h - df['doubles'].sum() - df['triples'].sum() - df['hr'].sum()
    lg_woba_denom = league_ab + league_bb + league_sf + league_hbp
    league_woba = (WOBA_BB * league_bb + WOBA_HBP * league_hbp + WOBA_1B * lg_singles +
                   WOBA_2B * df['doubles'].sum() + WOBA_3B * df['triples'].sum() +
                   WOBA_HR * df['hr'].sum()) / lg_woba_denom if lg_woba_denom > 0 else 0

    for pid, group in df.groupby('playerId'):
        stats = compute_hitting_stats(group)
        if stats['PA'] >= min_pa:
            stats['playerId'] = pid
            stats['playerName'] = group['playerName'].iloc[0]
            stats['school'] = group['school'].iloc[0] if 'school' in group.columns else group['teamName'].iloc[0]
            stats['wRAA'] = compute_wraa(stats['wOBA'], league_woba, stats['PA'])
            pos = group['formalPosition'].dropna()
            stats['position'] = pos.iloc[0] if len(pos) > 0 else group['playerPosition'].value_counts().index[0] if len(group['playerPosition'].dropna()) > 0 else ''
            cls = group['classification'].dropna()
            stats['classification'] = cls.iloc[0] if len(cls) > 0 else ''
            stats['team_db_id'] = group['team_db_id'].dropna().iloc[0] if 'team_db_id' in group.columns and len(group['team_db_id'].dropna()) > 0 else None
            rows.append(stats)
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    result['league_woba'] = league_woba
    result['league_r_pa'] = league_r_pa
    return result


@st.cache_data
def build_division_pitching_stats(sport, division, min_bf=30):
    """Compute pitching stats for all qualified pitchers in the division."""
    df = load_pbp(sport, division, 'pitching')
    if df is None or df.empty:
        return pd.DataFrame()
    rows = []
    for pid, group in df.groupby('playerId'):
        stats = compute_pitching_stats(group)
        if stats['BF'] >= min_bf:
            stats['playerId'] = pid
            stats['playerName'] = group['playerName'].iloc[0]
            stats['school'] = group['school'].iloc[0] if 'school' in group.columns else group['teamName'].iloc[0]
            pos = group['formalPosition'].dropna()
            stats['position'] = pos.iloc[0] if len(pos) > 0 else 'P'
            cls = group['classification'].dropna()
            stats['classification'] = cls.iloc[0] if len(cls) > 0 else ''
            stats['team_db_id'] = group['team_db_id'].dropna().iloc[0] if 'team_db_id' in group.columns and len(group['team_db_id'].dropna()) > 0 else None
            rows.append(stats)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ── Percentile computation ──────────────────────────────────────────────────
def get_percentile(values, player_val):
    """Return percentile (0-100) of player_val within values (no scipy needed)."""
    clean = values.dropna()
    n = len(clean)
    if n == 0:
        return 50.0
    below = (clean < player_val).sum()
    equal = (clean == player_val).sum()
    # "rank" kind: (below + 0.5 * equal) / n * 100
    return (below + 0.5 * equal) / n * 100


# ── Logo loading ────────────────────────────────────────────────────────────
def load_logo_image(team_db_id):
    """Load team logo as PIL Image, trying multiple paths."""
    if team_db_id is None or pd.isna(team_db_id):
        return None
    tid = int(team_db_id)
    for ext in ['png', 'webp']:
        p = LOGO_DIR / f'{tid}.{ext}'
        if p.exists():
            return Image.open(p).convert('RGBA')
    # Try New_Logos subfolder
    for ext in ['png', 'webp']:
        p = LOGO_DIR / 'New_Logos' / f'{tid}.{ext}'
        if p.exists():
            return Image.open(p).convert('RGBA')
    return None


# ── Card rendering ──────────────────────────────────────────────────────────
def _percentile_color(pct):
    """Return color from red (0) -> yellow (50) -> green (100)."""
    if pct <= 50:
        t = pct / 50.0
        r = int(220 * (1 - t) + 230 * t)
        g = int(50 * (1 - t) + 190 * t)
        b = int(50 * (1 - t) + 50 * t)
    else:
        t = (pct - 50) / 50.0
        r = int(230 * (1 - t) + 50 * t)
        g = int(190 * (1 - t) + 200 * t)
        b = int(50 * (1 - t) + 80 * t)
    return f'#{r:02x}{g:02x}{b:02x}'


def render_player_card(player_stats, player_name, school, position, classification,
                       percentile_data, game_log_values, game_dates, league_avg_line,
                       team_db_id, player_type, rolling_label):
    """
    Render a FiveThirtyEight-style player card as a matplotlib figure.

    percentile_data: list of (label, percentile_value) tuples
    game_log_values: list/array of rolling stat values over time
    league_avg_line: float for the dashed league-average line on the sparkline
    """
    fig = plt.figure(figsize=(8, 10), facecolor=CARD_BG, dpi=120)

    # We use a single axes that covers the whole figure for manual layout
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 125)
    ax.set_facecolor(CARD_BG)
    ax.axis('off')

    # ── Top section: Logo + Name + Info ─────────────────────────────────
    # Red accent bar at top
    ax.fill_between([0, 100], [125, 125], [122, 122], color=ACCENT_RED)

    # Team logo
    logo_img = load_logo_image(team_db_id)
    if logo_img is not None:
        logo_img.thumbnail((128, 128))
        imagebox = OffsetImage(np.array(logo_img), zoom=0.22)
        ab = AnnotationBbox(imagebox, (10, 114), frameon=False, zorder=5)
        ax.add_artist(ab)

    # Player name
    ax.text(22, 117, player_name, fontsize=20, fontweight='bold',
            color='white', va='center', fontfamily='sans-serif')

    # Info line: School | Position | Classification
    info_parts = []
    if school:
        info_parts.append(school)
    if position:
        info_parts.append(position)
    if classification:
        info_parts.append(classification)
    info_text = '  |  '.join(info_parts)
    ax.text(22, 112, info_text, fontsize=9, color='#999999', va='center', fontfamily='sans-serif')

    # ── Key stats row ───────────────────────────────────────────────────
    if player_type == 'Hitter':
        stat_keys = ['BA', 'OBP', 'SLG', 'OPS', 'wOBA', 'wRAA']
        stat_formats = {
            'BA': '.3f', 'OBP': '.3f', 'SLG': '.3f', 'OPS': '.3f',
            'wOBA': '.3f', 'wRAA': '.1f',
        }
    else:
        stat_keys = ['ERA', 'FIP', 'K%', 'BB%', 'WHIP', 'OPS Against']
        stat_formats = {
            'ERA': '.2f', 'FIP': '.2f', 'K%': '.1f', 'BB%': '.1f',
            'WHIP': '.2f', 'OPS Against': '.3f',
        }

    n_stats = len(stat_keys)
    stat_width = 80 / n_stats
    start_x = 10

    # Background band for stats
    stat_bg = FancyBboxPatch((5, 100), 90, 8, boxstyle="round,pad=0.5",
                  facecolor='#1e1e1e', edgecolor='#333333', linewidth=0.5)
    ax.add_patch(stat_bg)

    for i, key in enumerate(stat_keys):
        x = start_x + i * stat_width + stat_width / 2
        val = player_stats.get(key, 0)
        fmt = stat_formats.get(key, '.2f')

        # Display label
        display_label = key
        if key == 'OPS Against':
            display_label = 'OPS Ag.'
        ax.text(x, 106.5, display_label, fontsize=7, color='#888888',
                ha='center', va='center', fontfamily='sans-serif', fontweight='500')

        # Display value
        if key == 'wRAA':
            val_str = f"{val:+.1f}" if val != 0 else "0.0"
        elif '%' in key:
            val_str = f"{val:{fmt}}%"
        else:
            val_str = f"{val:{fmt}}"
        ax.text(x, 102.5, val_str, fontsize=12, color='white',
                ha='center', va='center', fontfamily='sans-serif', fontweight='bold')

    # ── Percentile bars ─────────────────────────────────────────────────
    ax.text(10, 96, 'PERCENTILE RANKINGS', fontsize=8, color='#888888',
            va='center', fontfamily='sans-serif', fontweight='600', style='italic')

    bar_y_start = 92
    bar_height = 2.8
    bar_gap = 1.8
    bar_left = 28
    bar_right = 90
    bar_width = bar_right - bar_left

    # Create the gradient colormap: red -> yellow -> green
    cmap_colors = ['#dc3232', '#e6be32', '#32c850']
    pctl_cmap = LinearSegmentedColormap.from_list('pctl', cmap_colors, N=256)

    for i, (label, pctl_val) in enumerate(percentile_data):
        y = bar_y_start - i * (bar_height + bar_gap)

        # Label
        ax.text(bar_left - 1.5, y - bar_height / 2, label, fontsize=8,
                color='#cccccc', ha='right', va='center', fontfamily='sans-serif',
                fontweight='500')

        # Draw gradient bar background
        n_segments = 100
        for j in range(n_segments):
            x0 = bar_left + j * bar_width / n_segments
            w = bar_width / n_segments + 0.1  # slight overlap to avoid gaps
            color = pctl_cmap(j / n_segments)
            ax.fill_between([x0, x0 + w],
                            [y, y],
                            [y - bar_height, y - bar_height],
                            color=color, alpha=0.3)

        # Player marker
        marker_x = bar_left + (pctl_val / 100) * bar_width
        marker_color = _percentile_color(pctl_val)

        # Marker dot
        ax.plot(marker_x, y - bar_height / 2, 'o', color=marker_color,
                markersize=9, markeredgecolor='white', markeredgewidth=1.5, zorder=10)

        # Percentile text inside the marker
        pctl_int = int(round(pctl_val))
        ax.text(marker_x, y - bar_height / 2, f'{pctl_int}',
                fontsize=5.5, color='white', ha='center', va='center',
                fontweight='bold', fontfamily='sans-serif', zorder=11)

    # ── Game log sparkline ──────────────────────────────────────────────
    sparkline_top = bar_y_start - len(percentile_data) * (bar_height + bar_gap) - 2
    ax.text(10, sparkline_top + 1, f'ROLLING 5-GAME {rolling_label}',
            fontsize=8, color='#888888', va='center', fontfamily='sans-serif',
            fontweight='600', style='italic')

    spark_ax = fig.add_axes([0.12, 0.08, 0.78, 0.18])
    spark_ax.set_facecolor('#1e1e1e')
    for spine in spark_ax.spines.values():
        spine.set_color('#444444')
        spine.set_linewidth(0.5)

    if len(game_log_values) > 0:
        x_vals = np.arange(len(game_log_values))
        spark_ax.plot(x_vals, game_log_values, color=ACCENT_RED, linewidth=2, zorder=5)
        spark_ax.fill_between(x_vals, game_log_values, alpha=0.15, color=ACCENT_RED)

        # League average dashed line
        if league_avg_line is not None:
            spark_ax.axhline(y=league_avg_line, color='#888888', linestyle='--',
                             linewidth=1, alpha=0.7, label=f'Div. Avg: {league_avg_line:.3f}')
            spark_ax.legend(fontsize=7, loc='upper right',
                            facecolor='#2a2a2a', edgecolor='#555555',
                            labelcolor='#cccccc')

        # Style
        spark_ax.tick_params(colors='#888888', labelsize=7)
        spark_ax.set_xlim(-0.5, len(game_log_values) - 0.5)

        # X-axis labels: use date strings if available
        if game_dates is not None and len(game_dates) == len(game_log_values):
            # Show a subset of labels to avoid crowding
            n = len(game_dates)
            if n <= 10:
                tick_indices = list(range(n))
            else:
                step = max(1, n // 8)
                tick_indices = list(range(0, n, step))
                if n - 1 not in tick_indices:
                    tick_indices.append(n - 1)
            spark_ax.set_xticks(tick_indices)
            spark_ax.set_xticklabels([game_dates[i] for i in tick_indices],
                                     rotation=45, ha='right', fontsize=6, color='#888888')
        else:
            spark_ax.set_xlabel('Game', fontsize=7, color='#888888')
    else:
        spark_ax.text(0.5, 0.5, 'No game log data available',
                      transform=spark_ax.transAxes, ha='center', va='center',
                      fontsize=10, color='#666666')

    spark_ax.set_ylabel(rolling_label, fontsize=7, color='#888888')

    # ── Brand watermark (bottom right corner) ───────────────────────────
    ax.text(95, 2, '64analytics.com', fontsize=6, color='#555555',
            ha='right', va='bottom', fontfamily='sans-serif', style='italic')

    return fig


# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title='64 Analytics — Player Cards', layout='wide',
                   initial_sidebar_state='expanded')

st.markdown('''
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, .stApp, .stApp *:not([class*="icon"]):not([data-testid*="icon"]):not(.material-icons):not(.material-symbols) { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: #1a1a1a; }
h1, h2, h3, p, label, .stMarkdown { color: #C8C8C8 !important; }
</style>
''', unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.image(str(BRAND_LOGO), width=80)
st.sidebar.markdown('## Player Cards')
if st.sidebar.button('Reload data'):
    st.cache_data.clear()
    st.rerun()

sport = st.sidebar.selectbox('Sport', ['baseball', 'softball'], format_func=str.title)
division = st.sidebar.selectbox('Division', ['D1', 'D2', 'D3'])
player_type = st.sidebar.radio('Player type', ['Hitter', 'Pitcher'], horizontal=True)

# ── Load data ───────────────────────────────────────────────────────────────
if player_type == 'Hitter':
    all_stats = build_division_hitting_stats(sport, division, min_pa=30)
    min_label = 'Min PA'
    min_default = 30
else:
    all_stats = build_division_pitching_stats(sport, division, min_bf=30)
    min_label = 'Min BF'
    min_default = 30

if all_stats is None or all_stats.empty:
    st.warning(f'No {player_type.lower()} data found for {sport.title()} {division}.')
    st.stop()

# Min PA/BF filter in sidebar
min_threshold = st.sidebar.number_input(min_label, min_value=1, max_value=300,
                                         value=min_default, step=5)

if player_type == 'Hitter':
    qualified = all_stats[all_stats['PA'] >= min_threshold].copy()
else:
    qualified = all_stats[all_stats['BF'] >= min_threshold].copy()

if qualified.empty:
    st.warning(f'No qualified players found with {min_label} >= {min_threshold}.')
    st.stop()

# Build player options: "Name (School)"
qualified['display_label'] = qualified['playerName'] + ' (' + qualified['school'].fillna('') + ')'
player_options = sorted(qualified['display_label'].unique())

selected_label = st.sidebar.selectbox(
    'Search player',
    options=player_options,
    index=0,
    help='Type to search for a player',
)

if not selected_label:
    st.info('Select a player from the sidebar.')
    st.stop()

player_row = qualified[qualified['display_label'] == selected_label].iloc[0]
player_id = player_row['playerId']
player_name = player_row['playerName']
school = player_row.get('school', '')
position = player_row.get('position', '')
classification = player_row.get('classification', '')
team_db_id = player_row.get('team_db_id', None)

# ── Compute percentiles ────────────────────────────────────────────────────
if player_type == 'Hitter':
    pctl_ops = get_percentile(qualified['OPS'], player_row['OPS'])
    pctl_woba = get_percentile(qualified['wOBA'], player_row['wOBA'])
    pctl_k = 100 - get_percentile(qualified['K%'], player_row['K%'])  # inverted: lower K% is better
    pctl_bb = get_percentile(qualified['BB%'], player_row['BB%'])
    pctl_iso = get_percentile(qualified['ISO'], player_row['ISO'])
    pctl_babip = get_percentile(qualified['BABIP'], player_row['BABIP'])

    percentile_data = [
        ('OPS', pctl_ops),
        ('wOBA', pctl_woba),
        ('K%', pctl_k),
        ('BB%', pctl_bb),
        ('ISO', pctl_iso),
        ('BABIP', pctl_babip),
    ]
else:
    pctl_fip = 100 - get_percentile(qualified['FIP'], player_row['FIP'])  # inverted
    pctl_ops_ag = 100 - get_percentile(qualified['OPS Against'], player_row['OPS Against'])  # inverted
    pctl_k = get_percentile(qualified['K%'], player_row['K%'])
    pctl_bb = 100 - get_percentile(qualified['BB%'], player_row['BB%'])  # inverted
    pctl_whip = 100 - get_percentile(qualified['WHIP'], player_row['WHIP'])  # inverted
    pctl_era = 100 - get_percentile(qualified['ERA'], player_row['ERA'])  # inverted

    percentile_data = [
        ('FIP', pctl_fip),
        ('OPS Ag.', pctl_ops_ag),
        ('K%', pctl_k),
        ('BB%', pctl_bb),
        ('WHIP', pctl_whip),
        ('ERA', pctl_era),
    ]

# ── Game log sparkline data ─────────────────────────────────────────────────
if player_type == 'Hitter':
    raw_pbp = load_pbp(sport, division, 'hitting')
    rolling_label = 'OPS'
else:
    raw_pbp = load_pbp(sport, division, 'pitching')
    rolling_label = 'FIP'

game_log_values = []
game_dates = []
league_avg_line = None

if raw_pbp is not None and not raw_pbp.empty:
    player_games = raw_pbp[raw_pbp['playerId'] == player_id].copy()

    if 'date_parsed' in player_games.columns:
        player_games = player_games.sort_values('date_parsed')

    if not player_games.empty:
        if player_type == 'Hitter':
            # Compute per-game OPS
            per_game_stat = player_games.apply(compute_hitting_game_stats, axis=1).values
            # Rolling 5-game average
            s = pd.Series(per_game_stat)
            rolling = s.rolling(window=5, min_periods=1).mean()
            game_log_values = rolling.values

            # League average OPS
            league_total = raw_pbp.copy()
            lg_ab = league_total['ab'].sum()
            lg_h = league_total['h'].sum()
            lg_bb = league_total['bb'].sum()
            lg_hbp = league_total['hbp'].sum()
            lg_sf = league_total['sf'].sum()
            lg_tb = league_total['tb'].sum()
            lg_obp_d = lg_ab + lg_bb + lg_hbp + lg_sf
            lg_obp = (lg_h + lg_bb + lg_hbp) / lg_obp_d if lg_obp_d > 0 else 0
            lg_slg = lg_tb / lg_ab if lg_ab > 0 else 0
            league_avg_line = lg_obp + lg_slg
        else:
            # Compute per-game FIP
            per_game_stat = player_games.apply(compute_pitching_game_fip, axis=1).values
            s = pd.Series(per_game_stat)
            rolling = s.rolling(window=5, min_periods=1).mean()
            game_log_values = rolling.dropna().values

            # League average FIP
            lg_total_outs = baseball_ip_to_outs(raw_pbp['ip']).sum()
            lg_ip = outs_to_actual_innings(lg_total_outs)
            lg_hr = raw_pbp['hrA'].sum()
            lg_bb = raw_pbp['bb'].sum()
            lg_hb = raw_pbp['hb'].sum()
            lg_so = raw_pbp['so'].sum()
            league_avg_line = ((13 * lg_hr) + (3 * (lg_bb + lg_hb)) - (2 * lg_so)) / lg_ip + FIP_CONSTANT if lg_ip > 0 else None

        # Date labels
        if 'date_parsed' in player_games.columns:
            dates = player_games['date_parsed'].dt.strftime('%m/%d')
            if player_type == 'Pitcher':
                # Drop NaN FIP games from dates too
                valid_mask = ~pd.Series(per_game_stat).isna()
                dates = dates[valid_mask.values]
            game_dates = dates.tolist()

# ── Render card ─────────────────────────────────────────────────────────────
st.markdown('# Player Projection Card')

fig = render_player_card(
    player_stats=player_row.to_dict(),
    player_name=player_name,
    school=school if school else '',
    position=position if position else '',
    classification=classification if classification and not pd.isna(classification) else '',
    percentile_data=percentile_data,
    game_log_values=game_log_values,
    game_dates=game_dates if game_dates else None,
    league_avg_line=league_avg_line,
    team_db_id=team_db_id,
    player_type=player_type,
    rolling_label=rolling_label,
)

# Center the card
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.pyplot(fig, use_container_width=False)
plt.close(fig)

# ── Download button ─────────────────────────────────────────────────────────
buf = BytesIO()
fig2 = render_player_card(
    player_stats=player_row.to_dict(),
    player_name=player_name,
    school=school if school else '',
    position=position if position else '',
    classification=classification if classification and not pd.isna(classification) else '',
    percentile_data=percentile_data,
    game_log_values=game_log_values,
    game_dates=game_dates if game_dates else None,
    league_avg_line=league_avg_line,
    team_db_id=team_db_id,
    player_type=player_type,
    rolling_label=rolling_label,
)
fig2.savefig(buf, format='png', bbox_inches='tight', facecolor=CARD_BG, dpi=200)
plt.close(fig2)
buf.seek(0)

col2.download_button(
    label='Download Card (PNG)',
    data=buf,
    file_name=f"player_card_{player_name.replace(' ', '_')}.png",
    mime='image/png',
)

# ── Season game log table ───────────────────────────────────────────────────
st.markdown('---')
st.markdown(f'### Game Log — {player_name}')

if raw_pbp is not None and not raw_pbp.empty:
    game_log = raw_pbp[raw_pbp['playerId'] == player_id].copy()
    if 'date_parsed' in game_log.columns:
        game_log = game_log.sort_values('date_parsed')

    if not game_log.empty:
        if player_type == 'Hitter':
            display_cols = ['date', 'teamName', 'ab', 'r', 'h', 'doubles', 'triples',
                            'hr', 'rbi', 'bb', 'hbp', 'k', 'sb', 'cs']
            col_rename = {
                'date': 'Date', 'teamName': 'Team',
                'ab': 'AB', 'r': 'R', 'h': 'H', 'doubles': '2B',
                'triples': '3B', 'hr': 'HR', 'rbi': 'RBI', 'bb': 'BB',
                'hbp': 'HBP', 'k': 'K', 'sb': 'SB', 'cs': 'CS',
            }
            # Add per-game OPS column
            game_log['OPS'] = game_log.apply(compute_hitting_game_stats, axis=1).round(3)
            display_cols.append('OPS')
            col_rename['OPS'] = 'OPS'
        else:
            display_cols = ['date', 'teamName', 'ip', 'h', 'r', 'er', 'bb',
                            'hb', 'so', 'hrA', 'bf']
            col_rename = {
                'date': 'Date', 'teamName': 'Team',
                'ip': 'IP', 'h': 'H', 'r': 'R', 'er': 'ER', 'bb': 'BB',
                'hb': 'HB', 'so': 'SO', 'hrA': 'HR', 'bf': 'BF',
            }
            # Add per-game FIP/ERA columns
            game_log['FIP'] = game_log.apply(compute_pitching_game_fip, axis=1).round(2)
            game_log['ERA_game'] = game_log.apply(
                lambda row: round((row['er'] / outs_to_actual_innings(ip_to_outs_scalar(row['ip']))) * 9, 2)
                if ip_to_outs_scalar(row['ip']) > 0 else np.nan, axis=1
            )
            display_cols.extend(['FIP', 'ERA_game'])
            col_rename['FIP'] = 'FIP'
            col_rename['ERA_game'] = 'ERA'

        available = [c for c in display_cols if c in game_log.columns]
        display_df = game_log[available].rename(columns=col_rename).reset_index(drop=True)
        display_df.index = display_df.index + 1

        # Style the dataframe
        st.dataframe(
            display_df,
            use_container_width=True,
            height=min(400, 35 * len(display_df) + 38),
        )

        # Season totals
        st.markdown('**Season totals:**')
        if player_type == 'Hitter':
            stats = player_row.to_dict()
            totals = (
                f"**{stats.get('PA', 0)}** PA  |  "
                f"**{stats.get('BA', 0):.3f}** BA  |  "
                f"**{stats.get('OBP', 0):.3f}** OBP  |  "
                f"**{stats.get('SLG', 0):.3f}** SLG  |  "
                f"**{stats.get('OPS', 0):.3f}** OPS  |  "
                f"**{stats.get('wOBA', 0):.3f}** wOBA  |  "
                f"**{stats.get('HR', 0)}** HR  |  "
                f"**{stats.get('SB', 0)}** SB"
            )
        else:
            stats = player_row.to_dict()
            totals = (
                f"**{stats.get('IP_display', '0.0')}** IP  |  "
                f"**{stats.get('ERA', 0):.2f}** ERA  |  "
                f"**{stats.get('FIP', 0):.2f}** FIP  |  "
                f"**{stats.get('K%', 0):.1f}%** K%  |  "
                f"**{stats.get('BB%', 0):.1f}%** BB%  |  "
                f"**{stats.get('WHIP', 0):.2f}** WHIP  |  "
                f"**{stats.get('OPS Against', 0):.3f}** OPS Against"
            )
        st.markdown(totals)
    else:
        st.info('No game log data found for this player.')
else:
    st.info('No play-by-play data available.')
