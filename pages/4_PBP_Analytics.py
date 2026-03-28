"""
64 Analytics — PBP Stat Calculator
Game-level stats from play-by-play data.
Compute OPS, wRAA, FIP, Allowed OPS for any player/team over any date range.
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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw

# ── Path setup (works locally and on Streamlit Cloud) ─────────────────────────
_APP_DIR = Path(__file__).resolve().parent.parent
PBP_DIR = _APP_DIR / 'pbp_data'
DATA_DIR = _APP_DIR / 'data'
LOGO_DIR = _APP_DIR / 'team_logos_512'
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


# ── IP conversion ────────────────────────────────────────────────────────────
def baseball_ip_to_outs(ip_col):
    """Convert baseball IP notation (5.2 = 5 innings + 2 outs) to total outs."""
    whole = np.floor(ip_col)
    frac = np.round((ip_col - whole) * 10).astype(int)
    return (whole * 3 + frac).astype(int)


def outs_to_ip(total_outs):
    """Convert total outs back to display IP (e.g., 17 outs = 5.2)."""
    innings = total_outs // 3
    remainder = total_outs % 3
    return float(f"{int(innings)}.{int(remainder)}")


def outs_to_actual_innings(total_outs):
    """Convert total outs to actual innings for rate calculations."""
    return total_outs / 3


# ── Stat Computation ─────────────────────────────────────────────────────────
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
    gdp = df['oppDp'].sum()
    r = df['r'].sum()
    rbi = df['rbi'].sum()
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
    woba_denom = ab + bb + sf + hbp
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
    """Compute pitching stats from game-level PBP pitching data."""
    total_outs = baseball_ip_to_outs(df['ip']).sum()
    ip = outs_to_actual_innings(total_outs)
    h = df['h'].sum()
    bb = df['bb'].sum()
    hb = df['hb'].sum()
    so = df['so'].sum()
    hr = df['hrA'].sum()
    bf = df['bf'].sum()
    wp = df['wp'].sum()
    bk = df['bk'].sum()
    doubles = df['doublesA'].sum()
    triples = df['triplesA'].sum()
    singles = h - doubles - triples - hr
    sha = df['sha'].sum()
    sfa = df['sfa'].sum()
    er = df['er'].sum()

    # Opponent AB = BF - BB - HB - SFA - SHA
    p_oab = bf - bb - hb - sfa - sha

    # FIP
    fip = ((13 * hr) + (3 * (bb + hb)) - (2 * so)) / ip + FIP_CONSTANT if ip > 0 else 0

    # ERA
    era = (er / ip) * 9 if ip > 0 else 0

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

    # K/BB rates
    k_pct = so / bf if bf > 0 else 0
    bb_pct = bb / bf if bf > 0 else 0

    return {
        'IP': outs_to_ip(total_outs), 'BF': int(bf), 'H': int(h),
        'ER': int(er), 'BB': int(bb), 'HB': int(hb), 'SO': int(so), 'HR': int(hr),
        '2B-A': int(doubles), '3B-A': int(triples),
        'WP': int(wp), 'Bk': int(bk),
        'ERA': round(era, 2), 'FIP': round(fip, 2),
        'BAA': round(ba_against, 3),
        'OBP Against': round(obp_against, 3),
        'SLG Against': round(slg_against, 3),
        'OPS Against': round(ops_against, 3),
        'K%': round(k_pct, 3), 'BB%': round(bb_pct, 3),
    }


def compute_fielding_stats(df):
    """Compute fielding stats from game-level PBP fielding data."""
    po = df['po'].sum()
    a = df['a'].sum()
    tc = df['tc'].sum()
    e = df['e'].sum()
    pb = df['pb'].sum()
    sba = df['sba'].sum()
    csb = df['csb'].sum()
    idp = df['idp'].sum()
    tp = df['tp'].sum()

    fpct = (po + a) / tc if tc > 0 else 0
    cs_pct = csb / (sba + csb) if (sba + csb) > 0 else 0

    return {
        'PO': int(po), 'A': int(a), 'TC': int(tc), 'E': int(e),
        'FPCT': round(fpct, 3),
        'PB': int(pb), 'SBA': int(sba), 'CSB': int(csb),
        'CS%': round(cs_pct, 3),
        'IDP': int(idp), 'TP': int(tp),
    }


def compute_wraa(woba, league_woba, pa):
    """wRAA = ((wOBA - league_wOBA) / wOBA_scale) * PA"""
    return round(((woba - league_woba) / WOBA_SCALE) * pa, 1)


def _primary_position(group):
    """Return the most common position for a player group."""
    if 'playerPosition' in group.columns:
        pos = group['playerPosition'].dropna()
        if len(pos) > 0:
            return pos.value_counts().index[0]
    return ''


def compute_grouped_hitting(df, group_col, league_woba, min_pa=1):
    """Compute hitting stats grouped by a column."""
    rows = []
    for name, group in df.groupby(group_col):
        stats = compute_hitting_stats(group)
        if stats['PA'] >= min_pa:
            stats['wRAA'] = compute_wraa(stats['wOBA'], league_woba, stats['PA'])
            stats[group_col] = name
            stats['Pos'] = _primary_position(group)
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
            stats['Pos'] = _primary_position(group)
            rows.append(stats)
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    cols = [group_col] + [c for c in result.columns if c != group_col]
    return result[cols].sort_values('FIP', ascending=True).reset_index(drop=True)


def compute_grouped_fielding(df, group_col):
    """Compute fielding stats grouped by a column."""
    rows = []
    for name, group in df.groupby(group_col):
        stats = compute_fielding_stats(group)
        if stats['TC'] > 0:
            stats[group_col] = name
            stats['Pos'] = _primary_position(group)
            rows.append(stats)
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    cols = [group_col] + [c for c in result.columns if c != group_col]
    return result[cols].sort_values('FPCT', ascending=False).reset_index(drop=True)


def _clean_position(pos):
    """Clean position field: 'PH/LF' → 'PH', '/LF/CF' → 'LF'."""
    if not pos or (isinstance(pos, float) and np.isnan(pos)):
        return pos
    pos = str(pos).strip()
    if pos.startswith('/'):
        # Subbed in from bench — take first position after the leading /
        parts = pos.lstrip('/').split('/')
        return parts[0] if parts else pos
    else:
        # Take position before the first /
        return pos.split('/')[0]


# ── Lineup Card helpers ───────────────────────────────────────────────────────
FIELD_POSITIONS = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH']


@st.cache_data
def load_team_logo_map():
    """Build team_name -> logo_id mapping from teams.csv."""
    teams_path = DATA_DIR / 'teams.csv'
    if not teams_path.exists():
        return {}
    teams = pd.read_csv(teams_path, low_memory=False)
    teams['id'] = pd.to_numeric(teams['id'], errors='coerce').fillna(0).astype(int)
    # Baseball IDs are canonical for logos
    bb = teams[teams['sport'] == 'Baseball'][['name', 'id']].drop_duplicates('name')
    name_to_id = dict(zip(bb['name'], bb['id']))
    # Add softball teams mapped to baseball counterparts
    sb = teams[teams['sport'] == 'Softball'][['name', 'id']].drop_duplicates('name')
    for _, row in sb.iterrows():
        if row['name'] not in name_to_id:
            name_to_id[row['name']] = row['id']
    return name_to_id


@st.cache_data
def load_team_conference_map():
    """Build team_name -> conference_name mapping from teams.csv + conferences.csv."""
    teams_path = DATA_DIR / 'teams.csv'
    confs_path = DATA_DIR / 'conferences.csv'
    if not teams_path.exists() or not confs_path.exists():
        return {}
    teams = pd.read_csv(teams_path, low_memory=False)
    confs = pd.read_csv(confs_path, low_memory=False)
    merged = teams.merge(confs[['id', 'name']], left_on='conference_id', right_on='id', suffixes=('', '_conf'))
    return dict(zip(merged['name'], merged['name_conf']))


@st.cache_data
def get_logo_base64(logo_id):
    """Load a team logo as base64 string for SVG embedding."""
    # Try png first, then webp
    for ext in ['png', 'webp']:
        p = LOGO_DIR / f'{logo_id}.{ext}'
        if p.exists():
            data = p.read_bytes()
            mime = 'image/png' if ext == 'png' else 'image/webp'
            return f'data:{mime};base64,{base64.b64encode(data).decode()}'
    return None
LC_RED = '#C41230'
LC_DH_COLOR = '#22d3a0'
LC_RELIEVER_COLOR = '#a855f7'


def _ip_to_outs(ip_col):
    whole = np.floor(ip_col)
    frac = np.round((ip_col - whole) * 10).astype(int)
    return (whole * 3 + frac).astype(int)


def _outs_to_ip_display(total_outs):
    return f"{int(total_outs // 3)}.{int(total_outs % 3)}"


def compute_pitching_for_lineup(df):
    total_outs = _ip_to_outs(df['ip']).sum()
    ip = total_outs / 3
    h = df['h'].sum(); bb = df['bb'].sum(); hb = df['hb'].sum()
    so = df['so'].sum(); hr = df['hrA'].sum(); bf = df['bf'].sum()
    er = df['er'].sum()
    doubles = df['doublesA'].sum(); triples = df['triplesA'].sum()
    sha = df['sha'].sum(); sfa = df['sfa'].sum()
    singles = h - doubles - triples - hr
    games = df['gameId'].nunique() if 'gameId' in df.columns else len(df)
    fip = ((13*hr) + (3*(bb+hb)) - (2*so)) / ip + FIP_CONSTANT if ip > 0 else 99
    era = (er / ip) * 9 if ip > 0 else 99
    k_pct = so / bf if bf > 0 else 0
    # Allowed OPS
    p_oab = bf - bb - hb - sfa - sha
    obp_d = p_oab + bb + hb + sfa
    obp_a = (h + bb + hb) / obp_d if obp_d > 0 else 0
    tb_a = singles + 2*doubles + 3*triples + 4*hr
    slg_a = tb_a / p_oab if p_oab > 0 else 0
    ops_a = obp_a + slg_a
    k9 = (so / ip) * 9 if ip > 0 else 0
    bb_pct = bb / bf if bf > 0 else 0
    whip = (bb + h) / ip if ip > 0 else 0
    return {'IP': _outs_to_ip_display(total_outs), 'IP_actual': ip,
            'BF': int(bf), 'H': int(h), 'ER': int(er), 'BB': int(bb),
            'SO': int(so), 'HR': int(hr), 'A': int(games),
            'ERA': round(era, 2), 'FIP': round(fip, 2), 'K%': round(k_pct, 3),
            'K/9': round(k9, 2), 'BB%': round(bb_pct, 3), 'WHIP': round(whip, 2),
            'OPS Against': round(ops_a, 3)}


def get_best_hitters(hitting_df, league_woba, min_pa=10):
    best = {}
    for pos in FIELD_POSITIONS:
        pos_df = hitting_df[hitting_df['playerPosition'] == pos]
        if len(pos_df) == 0:
            continue
        rows = []
        for name, group in pos_df.groupby('playerName'):
            stats = compute_hitting_stats(group)
            if stats['PA'] >= min_pa:
                stats['playerName'] = name
                stats['teamName'] = group['teamName'].mode().iloc[0] if len(group['teamName'].mode()) > 0 else ''
                stats['wRAA'] = compute_wraa(stats['wOBA'], league_woba, stats['PA'])
                rows.append(stats)
        if rows:
            df = pd.DataFrame(rows).sort_values('OPS', ascending=False)
            best[pos] = df.iloc[0].to_dict()
    return best


def _combined_rank(df):
    """Rank pitchers by combined FIP + OPS Against score.
    Best (lowest) in each stat gets n/n = 1.000, 2nd gets (n-1)/n, etc.
    Combined score = FIP rank score + OPS Against rank score. Higher = better.
    """
    n = len(df)
    if n == 0:
        return df
    df = df.copy()
    # rank(ascending=True) gives 1 to lowest value — that's the best for both FIP and OPS Against
    df['fip_score'] = (n - df['FIP'].rank(method='min') + 1) / n
    df['ops_a_score'] = (n - df['OPS Against'].rank(method='min') + 1) / n
    df['combined_score'] = df['fip_score'] + df['ops_a_score']
    return df.sort_values('combined_score', ascending=False)


def get_best_pitchers(pitching_df, min_bf_sp=50, min_bf_rp=15, n_starters=3, n_relievers=3):
    rows = []
    for name, group in pitching_df.groupby('playerName'):
        stats = compute_pitching_for_lineup(group)
        stats['playerName'] = name
        stats['teamName'] = group['teamName'].mode().iloc[0] if len(group['teamName'].mode()) > 0 else ''
        stats['is_starter'] = stats['IP_actual'] / max(stats['A'], 1) >= 3.0
        rows.append(stats)
    if not rows:
        return [], []
    df = pd.DataFrame(rows)
    starter_df = _combined_rank(df[(df['is_starter']) & (df['BF'] >= min_bf_sp)])
    reliever_df = _combined_rank(df[(~df['is_starter']) & (df['BF'] >= min_bf_rp)])
    starters = starter_df.head(n_starters).to_dict('records')
    relievers = reliever_df.head(n_relievers).to_dict('records')
    return starters, relievers


def _initials(name):
    parts = name.split()
    return (parts[0][0] + parts[-1][0]) if len(parts) >= 2 else name[:2].upper()


def _last_name(name):
    parts = name.split()
    return parts[-1] if parts else name


POS_COORDS = {
    'CF': (230, 24), 'LF': (108, 97), 'RF': (352, 97),
    'SS': (196, 162), '2B': (264, 162),
    '3B': (152, 222), '1B': (308, 222),
    'C': (230, 277), 'DH': (230, 345),
}


EGGSHELL = '#f5efe0'


def _logo_node(x, y, player, pos, team_map, ring_color, r=22, r_inner=19):
    """Render a player node with team logo in circle and full name below."""
    name = player['playerName']
    team = player.get('teamName', '')
    logo_id = team_map.get(team)
    logo_b64 = get_logo_base64(logo_id) if logo_id else None
    clip_id = f"clip-{pos}-{x}-{y}"

    if logo_b64:
        return f'''<g transform="translate({x},{y})">
    <circle r="{r}" fill="{ring_color}"/>
    <circle r="{r_inner}" fill="{EGGSHELL}"/>
    <clipPath id="{clip_id}"><circle r="{r_inner}"/></clipPath>
    <image href="{logo_b64}" x="-{r_inner}" y="-{r_inner}" width="{r_inner*2}" height="{r_inner*2}" clip-path="url(#{clip_id})" preserveAspectRatio="xMidYMid slice"/>
    <text font-size="7" fill="#c8a880" text-anchor="middle" y="{r+7}" font-family="sans-serif">{name}</text>
    <text font-size="6.5" fill="#9a8060" text-anchor="middle" y="{r+15}" font-family="sans-serif">{pos}</text></g>'''
    else:
        ini = _initials(name)
        return f'''<g transform="translate({x},{y})">
    <circle r="{r}" fill="{ring_color}"/><circle r="{r_inner}" fill="{EGGSHELL}"/>
    <text font-size="10" font-weight="500" fill="#333" text-anchor="middle" dominant-baseline="central" font-family="sans-serif">{ini}</text>
    <text font-size="7" fill="#c8a880" text-anchor="middle" y="{r+7}" font-family="sans-serif">{name}</text>
    <text font-size="6.5" fill="#9a8060" text-anchor="middle" y="{r+15}" font-family="sans-serif">{pos}</text></g>'''


def _pitcher_logo_node(x, y, player, team_map, ring_color, r=20, r_inner=17):
    """Render a pitcher node with team logo and full name."""
    name = player['playerName']
    team = player.get('teamName', '')
    logo_id = team_map.get(team)
    logo_b64 = get_logo_base64(logo_id) if logo_id else None
    clip_id = f"clip-p-{x}-{y}"

    if logo_b64:
        return f'''<g transform="translate({x},{y})">
    <circle r="{r}" fill="{ring_color}"/>
    <circle r="{r_inner}" fill="{EGGSHELL}"/>
    <clipPath id="{clip_id}"><circle r="{r_inner}"/></clipPath>
    <image href="{logo_b64}" x="-{r_inner}" y="-{r_inner}" width="{r_inner*2}" height="{r_inner*2}" clip-path="url(#{clip_id})" preserveAspectRatio="xMidYMid slice"/>
    <text font-size="7" fill="#c8a880" text-anchor="middle" y="25" font-family="sans-serif">{name}</text></g>'''
    else:
        ini = _initials(name)
        return f'''<g transform="translate({x},{y})">
    <circle r="{r}" fill="{ring_color}"/><circle r="{r_inner}" fill="{EGGSHELL}"/>
    <text font-size="9" font-weight="500" fill="#333" text-anchor="middle" dominant-baseline="central" font-family="sans-serif">{ini}</text>
    <text font-size="7" fill="#c8a880" text-anchor="middle" y="25" font-family="sans-serif">{name}</text></g>'''


def render_lineup_svg(best_hitters, starters, relievers, title, subtitle, team_map):
    nodes = []
    for pos, (x, y) in POS_COORDS.items():
        if pos in best_hitters:
            color = LC_DH_COLOR if pos == 'DH' else LC_RED
            nodes.append(_logo_node(x, y, best_hitters[pos], pos, team_map, color))
        else:
            nodes.append(f'''<g transform="translate({x},{y})">
    <circle r="22" fill="#555"/><circle r="19" fill="#1c2a38"/>
    <text font-size="9" fill="#666" text-anchor="middle" dominant-baseline="central" font-family="sans-serif">—</text>
    <text font-size="7" fill="#666" text-anchor="middle" y="37" font-family="sans-serif">{pos}</text></g>''')

    # Pitcher sidebar
    nodes.append('<line x1="416" y1="10" x2="416" y2="390" stroke="#3a3a3a" stroke-width="1"/>')
    nodes.append('<text x="438" y="26" font-size="8" fill="#a89880" text-anchor="middle" letter-spacing="0.08em" font-family="sans-serif">STARTERS</text>')
    for i in range(3):
        y = 58 + i * 56
        if i < len(starters):
            nodes.append(_pitcher_logo_node(438, y, starters[i], team_map, LC_RED))
        else:
            nodes.append(f'<g transform="translate(438,{y})"><circle r="20" fill="#555"/><circle r="17" fill="#1c2a38"/><text font-size="9" fill="#666" text-anchor="middle" dominant-baseline="central" font-family="sans-serif">—</text></g>')

    nodes.append('<text x="438" y="218" font-size="8" fill="#a89880" text-anchor="middle" letter-spacing="0.08em" font-family="sans-serif">RELIEVERS</text>')
    for i in range(3):
        y = 246 + i * 56
        if i < len(relievers):
            nodes.append(_pitcher_logo_node(438, y, relievers[i], team_map, LC_RELIEVER_COLOR))
        else:
            nodes.append(f'<g transform="translate(438,{y})"><circle r="20" fill="#555"/><circle r="17" fill="#1c2a38"/><text font-size="9" fill="#666" text-anchor="middle" dominant-baseline="central" font-family="sans-serif">—</text></g>')

    # Brand logo for center field
    brand_b64 = get_logo_base64('brand') if False else None
    brand_path = _APP_DIR / 'assets' / 'brand_logo_dark.png'
    if brand_path.exists():
        brand_data = brand_path.read_bytes()
        brand_b64 = f'data:image/png;base64,{base64.b64encode(brand_data).decode()}'

    svg = f'''<svg width="100%" viewBox="0 -20 460 420" xmlns="http://www.w3.org/2000/svg">
  <text x="230" y="-10" font-size="14" font-weight="600" fill="#C8C8C8" text-anchor="middle" font-family="sans-serif">{title}</text>
  <text x="230" y="6" font-size="9" fill="#888" text-anchor="middle" font-family="sans-serif">{subtitle}</text>
  <!-- Field shifted up 30px -->
  <path d="M230,310 L50,60 Q230,-30 410,60 Z" fill="#2d8a45"/>
  <path d="M50,60 Q230,-30 410,60" fill="none" stroke="{LC_RED}" stroke-width="10"/>
  <line x1="230" y1="310" x2="50" y2="60" stroke="{LC_RED}" stroke-width="8"/>
  <line x1="230" y1="310" x2="410" y2="60" stroke="{LC_RED}" stroke-width="8"/>
  <path d="M230,310 L128,190 Q230,120 332,190 Z" fill="#c8883a"/>
  <path d="M230,310 L144,200 Q230,136 316,200 Z" fill="#2d8a45"/>
  <rect x="222" y="138" width="16" height="16" rx="2" fill="#f5efe0" transform="rotate(45 230 146)"/>
  <rect x="308" y="198" width="14" height="14" rx="2" fill="#f5efe0" transform="rotate(45 315 205)"/>
  <rect x="148" y="198" width="14" height="14" rx="2" fill="#f5efe0" transform="rotate(45 155 205)"/>
  <polygon points="230,296 220,286 220,276 240,276 240,286" fill="#f5efe0"/>
  <!-- Mound -->
  <circle cx="230" cy="220" r="9" fill="#b87830" opacity="0.9"/>
  <circle cx="230" cy="220" r="4" fill="#a06820"/>
  <!-- Brand logo in center field -->
  {f'<image href="{brand_b64}" x="168" y="45" width="125" height="125" opacity="0.9" preserveAspectRatio="xMidYMid meet"/>' if brand_b64 else ''}
  {chr(10).join(nodes)}
</svg>
<div style="display:flex;gap:16px;justify-content:center;padding:6px 0;flex-wrap:wrap;">
  <div style="display:flex;align-items:center;gap:5px;font-size:11px;color:#888;">
    <div style="width:10px;height:10px;border-radius:50%;background:#C41230;"></div>Fielder / Starter</div>
  <div style="display:flex;align-items:center;gap:5px;font-size:11px;color:#888;">
    <div style="width:10px;height:10px;border-radius:50%;background:#22d3a0;"></div>DH</div>
  <div style="display:flex;align-items:center;gap:5px;font-size:11px;color:#888;">
    <div style="width:10px;height:10px;border-radius:50%;background:#a855f7;"></div>Reliever</div>
</div>'''
    return svg


def render_hitter_card_html(p, pos):
    ini = _initials(p['playerName'])
    return f'''<div style="background:#222;border-radius:16px;border:1px solid #3a3a3a;padding:18px 16px;max-width:360px;margin:8px auto;font-family:sans-serif;">
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px;padding-bottom:14px;border-bottom:1px solid #3a3a3a;">
    <div style="width:52px;height:52px;border-radius:50%;background:#1c2a38;border:2.5px solid #C41230;display:flex;align-items:center;justify-content:center;font-size:15px;font-weight:500;color:#e8d0b0;flex-shrink:0;">{ini}</div>
    <div><div style="font-size:15px;font-weight:500;color:#C8C8C8;">{p['playerName']}</div>
    <div style="font-size:11px;color:#888;margin-top:2px;">{p.get('teamName','')} · {pos}</div></div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:6px;">
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:17px;font-weight:500;color:#C41230;">{p['OPS']:.3f}</div><div style="font-size:9px;color:#888;">OPS</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:17px;font-weight:500;color:#C41230;">{p['wOBA']:.3f}</div><div style="font-size:9px;color:#888;">wOBA</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:17px;font-weight:500;color:#C41230;">{p['wRAA']:.1f}</div><div style="font-size:9px;color:#888;">wRAA</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:17px;font-weight:500;color:#C8C8C8;">{p['BA']:.3f}</div><div style="font-size:9px;color:#888;">AVG</div></div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;">
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:15px;color:#C8C8C8;">{p['HR']}</div><div style="font-size:9px;color:#888;">HR</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:15px;color:#C8C8C8;">{p['RBI']}</div><div style="font-size:9px;color:#888;">RBI</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:15px;color:#C8C8C8;">{p['R']}</div><div style="font-size:9px;color:#888;">R</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:15px;color:#C8C8C8;">{p['BB']}</div><div style="font-size:9px;color:#888;">BB</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:15px;color:#C8C8C8;">{p['SB']}</div><div style="font-size:9px;color:#888;">SB</div></div>
  </div>
</div>'''


def render_pitcher_card_html(p, role='Starter'):
    ini = _initials(p['playerName'])
    ring = LC_RED if role == 'Starter' else LC_RELIEVER_COLOR
    return f'''<div style="background:#222;border-radius:16px;border:1px solid #3a3a3a;padding:18px 16px;max-width:360px;margin:8px auto;font-family:sans-serif;">
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px;padding-bottom:14px;border-bottom:1px solid #3a3a3a;">
    <div style="width:52px;height:52px;border-radius:50%;background:#1c2a38;border:2.5px solid {ring};display:flex;align-items:center;justify-content:center;font-size:15px;font-weight:500;color:#e8d0b0;flex-shrink:0;">{ini}</div>
    <div><div style="font-size:15px;font-weight:500;color:#C8C8C8;">{p['playerName']}</div>
    <div style="font-size:11px;color:#888;margin-top:2px;">{p.get('teamName','')} · {role}</div></div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-bottom:6px;">
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:17px;font-weight:500;color:#C41230;">{p['ERA']:.2f}</div><div style="font-size:9px;color:#888;">ERA</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:17px;font-weight:500;color:#C41230;">{p['FIP']:.2f}</div><div style="font-size:9px;color:#888;">FIP</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:17px;font-weight:500;color:#C41230;">{p['OPS Against']:.3f}</div><div style="font-size:9px;color:#888;">OPS-A</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:17px;color:#C8C8C8;">{p['IP']}</div><div style="font-size:9px;color:#888;">IP</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:17px;color:#C8C8C8;">{p['K%']:.3f}</div><div style="font-size:9px;color:#888;">K%</div></div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;">
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:15px;color:#C8C8C8;">{p['K/9']:.2f}</div><div style="font-size:9px;color:#888;">K/9</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:15px;color:#C8C8C8;">{p['BB%']:.3f}</div><div style="font-size:9px;color:#888;">BB%</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:15px;color:#C8C8C8;">{p['WHIP']:.2f}</div><div style="font-size:9px;color:#888;">WHIP</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:15px;color:#C8C8C8;">{p['A']}</div><div style="font-size:9px;color:#888;">A</div></div>
  </div>
</div>'''


# ── PNG Rendering ─────────────────────────────────────────────────────────────
def _load_logo_pil(team_name, team_map, size=80):
    """Load a team logo as a circular PIL image with eggshell background."""
    logo_id = team_map.get(team_name)
    if not logo_id:
        return None
    for ext in ['png', 'webp']:
        p = LOGO_DIR / f'{logo_id}.{ext}'
        if p.exists():
            img = Image.open(p).convert('RGBA')
            img.thumbnail((size, size), Image.LANCZOS)
            # Paste onto eggshell circle
            bg = Image.new('RGBA', (size, size), (245, 239, 224, 255))
            mask = Image.new('L', (size, size), 0)
            ImageDraw.Draw(mask).ellipse([0, 0, size-1, size-1], fill=255)
            bg.paste(img, ((size - img.width) // 2, (size - img.height) // 2), img)
            bg.putalpha(mask)
            return bg
    return None


def _draw_circle_logo(ax, x, y, team_name, team_map, ring_color, zoom=0.08):
    """Draw a circular team logo with ring on a matplotlib axes."""
    logo = _load_logo_pil(team_name, team_map, size=120)
    if logo:
        # Draw ring circle
        ring = plt.Circle((x, y), 0.045, color=ring_color, transform=ax.transAxes, zorder=5)
        ax.add_patch(ring)
        arr = np.array(logo)
        im = OffsetImage(arr, zoom=zoom)
        ab = AnnotationBbox(im, (x, y), xycoords='axes fraction', frameon=False, zorder=6)
        ax.add_artist(ab)
        return True
    return False


def render_diamond_png(best_hitters, starters, relievers, title, subtitle, team_map):
    """Render the lineup card diamond as a matplotlib PNG."""
    fig = plt.figure(figsize=(12, 10), facecolor='#1a1a1a')
    ax = fig.add_axes([0, 0, 0.85, 1])
    ax.set_xlim(0, 460)
    ax.set_ylim(0, 420)
    ax.set_facecolor('#1a1a1a')
    ax.set_aspect('equal')
    ax.axis('off')

    # Flip y so 0 is top (match SVG coords)
    ax.invert_yaxis()

    # Field geometry (shifted up 30px from original)
    from matplotlib.patches import Polygon, Wedge
    # Outfield grass
    outfield = Polygon([(230, 310), (50, 60), (230, -30), (410, 60)], closed=True, fc='#2d8a45', ec='none', zorder=1)
    ax.add_patch(outfield)
    # Foul lines
    ax.plot([230, 50], [310, 60], color=LC_RED, linewidth=6, zorder=2)
    ax.plot([230, 410], [310, 60], color=LC_RED, linewidth=6, zorder=2)
    # Outfield arc
    theta = np.linspace(np.radians(25), np.radians(155), 100)
    arc_r = 280
    arc_x = 230 + arc_r * np.cos(theta)
    arc_y = 310 - arc_r * np.sin(theta)
    ax.plot(arc_x, arc_y, color=LC_RED, linewidth=8, zorder=2)
    # Infield dirt
    infield = Polygon([(230, 310), (128, 190), (230, 120), (332, 190)], closed=True, fc='#c8883a', ec='none', zorder=2)
    ax.add_patch(infield)
    # Infield grass
    inner_grass = Polygon([(230, 310), (144, 200), (230, 136), (316, 200)], closed=True, fc='#2d8a45', ec='none', zorder=2)
    ax.add_patch(inner_grass)
    # Bases
    for bx, by, s in [(230, 146, 12), (315, 205, 10), (155, 205, 10)]:
        base = Polygon([(bx, by-s), (bx+s, by), (bx, by+s), (bx-s, by)], closed=True, fc='#f5efe0', zorder=3)
        ax.add_patch(base)
    # Home plate
    ax.add_patch(Polygon([(220, 276), (220, 296), (230, 306), (240, 296), (240, 276)], closed=True, fc='#f5efe0', zorder=3))
    # Mound
    mound = plt.Circle((230, 220), 7, color='#b87830', zorder=3)
    ax.add_patch(mound)

    # Brand logo in CF
    brand_path = _APP_DIR / 'assets' / 'brand_logo_dark.png'
    if brand_path.exists():
        brand_img = Image.open(brand_path).convert('RGBA')
        brand_arr = np.array(brand_img)
        brand_im = OffsetImage(brand_arr, zoom=0.12)
        brand_ab = AnnotationBbox(brand_im, (230, 107), frameon=False, zorder=4)
        ax.add_artist(brand_ab)

    # Position players
    pos_xy = {
        'CF': (230, 24), 'LF': (108, 97), 'RF': (352, 97),
        'SS': (196, 162), '2B': (264, 162),
        '3B': (152, 222), '1B': (308, 222),
        'C': (230, 277), 'DH': (230, 345),
    }
    for pos, (px, py) in pos_xy.items():
        if pos in best_hitters:
            p = best_hitters[pos]
            color = LC_DH_COLOR if pos == 'DH' else LC_RED
            logo = _load_logo_pil(p.get('teamName', ''), team_map, size=120)
            if logo:
                ring = plt.Circle((px, py), 22, color=color, zorder=5)
                ax.add_patch(ring)
                eggshell = plt.Circle((px, py), 19, color=EGGSHELL, zorder=5)
                ax.add_patch(eggshell)
                arr = np.array(logo)
                im = OffsetImage(arr, zoom=0.30)
                ab = AnnotationBbox(im, (px, py), frameon=False, zorder=6)
                ax.add_artist(ab)
            else:
                ring = plt.Circle((px, py), 22, color=color, zorder=5)
                ax.add_patch(ring)
                inner = plt.Circle((px, py), 19, color=EGGSHELL, zorder=5)
                ax.add_patch(inner)
                ax.text(px, py, _initials(p['playerName']), ha='center', va='center',
                        fontsize=10, fontweight='bold', color='#333', zorder=7)
            ax.text(px, py + 29, p['playerName'], ha='center', va='top',
                    fontsize=7, color='#c8a880', zorder=7)
            ax.text(px, py + 37, pos, ha='center', va='top',
                    fontsize=6.5, color='#9a8060', zorder=7)
        else:
            ring = plt.Circle((px, py), 22, color='#555', zorder=5)
            ax.add_patch(ring)
            inner = plt.Circle((px, py), 19, color='#1c2a38', zorder=5)
            ax.add_patch(inner)
            ax.text(px, py, '—', ha='center', va='center', fontsize=9, color='#666', zorder=7)
            ax.text(px, py + 37, pos, ha='center', va='top', fontsize=7, color='#666', zorder=7)

    # Pitcher sidebar
    sidebar_ax = fig.add_axes([0.85, 0, 0.15, 1])
    sidebar_ax.set_xlim(0, 1)
    sidebar_ax.set_ylim(0, 1)
    sidebar_ax.set_facecolor('#1a1a1a')
    sidebar_ax.axis('off')

    sidebar_ax.axvline(x=0.05, color='#3a3a3a', linewidth=1)
    sidebar_ax.text(0.5, 0.97, 'STARTERS', ha='center', va='top', fontsize=8,
                    color='#a89880', fontweight='bold', )

    for i in range(3):
        cy = 0.88 - i * 0.13
        if i < len(starters):
            sp = starters[i]
            logo = _load_logo_pil(sp.get('teamName', ''), team_map, size=100)
            ring = plt.Circle((0.5, cy), 0.045, color=LC_RED, transform=sidebar_ax.transAxes, zorder=5)
            sidebar_ax.add_patch(ring)
            eg = plt.Circle((0.5, cy), 0.038, color=EGGSHELL, transform=sidebar_ax.transAxes, zorder=5)
            sidebar_ax.add_patch(eg)
            if logo:
                arr = np.array(logo)
                im = OffsetImage(arr, zoom=0.08)
                ab = AnnotationBbox(im, (0.5, cy), xycoords='axes fraction', frameon=False, zorder=6)
                sidebar_ax.add_artist(ab)
            sidebar_ax.text(0.5, cy - 0.06, sp['playerName'], ha='center', va='top',
                           fontsize=7, color='#c8a880', transform=sidebar_ax.transAxes, zorder=7)
        else:
            ring = plt.Circle((0.5, cy), 0.045, color='#555', transform=sidebar_ax.transAxes, zorder=5)
            sidebar_ax.add_patch(ring)

    sidebar_ax.text(0.5, 0.50, 'RELIEVERS', ha='center', va='top', fontsize=8,
                    color='#a89880', fontweight='bold')

    for i in range(3):
        cy = 0.41 - i * 0.13
        if i < len(relievers):
            rp = relievers[i]
            logo = _load_logo_pil(rp.get('teamName', ''), team_map, size=100)
            ring = plt.Circle((0.5, cy), 0.045, color=LC_RELIEVER_COLOR, transform=sidebar_ax.transAxes, zorder=5)
            sidebar_ax.add_patch(ring)
            eg = plt.Circle((0.5, cy), 0.038, color=EGGSHELL, transform=sidebar_ax.transAxes, zorder=5)
            sidebar_ax.add_patch(eg)
            if logo:
                arr = np.array(logo)
                im = OffsetImage(arr, zoom=0.08)
                ab = AnnotationBbox(im, (0.5, cy), xycoords='axes fraction', frameon=False, zorder=6)
                sidebar_ax.add_artist(ab)
            sidebar_ax.text(0.5, cy - 0.06, rp['playerName'], ha='center', va='top',
                           fontsize=7, color='#c090e8', transform=sidebar_ax.transAxes, zorder=7)
        else:
            ring = plt.Circle((0.5, cy), 0.045, color='#555', transform=sidebar_ax.transAxes, zorder=5)
            sidebar_ax.add_patch(ring)

    # Title
    fig.text(0.42, 0.97, title, ha='center', va='top', fontsize=16, fontweight='bold', color='#C8C8C8')
    fig.text(0.42, 0.94, subtitle, ha='center', va='top', fontsize=10, color='#888')

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=180, facecolor='#1a1a1a', edgecolor='none', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


def render_cards_png(best_hitters, starters, relievers, title, subtitle, team_map):
    """Render player detail cards as a matplotlib PNG."""
    # Build card data
    cards = []
    for pos in FIELD_POSITIONS:
        if pos in best_hitters:
            p = best_hitters[pos]
            top = [(f"{p['OPS']:.3f}", 'OPS'), (f"{p['wOBA']:.3f}", 'wOBA'),
                   (f"{p['wRAA']:.1f}", 'wRAA'), (f"{p['BA']:.3f}", 'AVG')]
            bot = [(p['HR'], 'HR'), (p['RBI'], 'RBI'), (p['R'], 'R'),
                   (p['BB'], 'BB'), (p['SB'], 'SB')]
            cards.append(('hitter', p['playerName'], p.get('teamName', ''), pos, top, bot))
    for sp in starters[:3]:
        top = [(f"{sp['ERA']:.2f}", 'ERA'), (f"{sp['FIP']:.2f}", 'FIP'),
               (f"{sp['OPS Against']:.3f}", 'OPS-A'), (sp['IP'], 'IP'), (f"{sp['K%']:.3f}", 'K%')]
        bot = [(f"{sp['K/9']:.2f}", 'K/9'), (f"{sp['BB%']:.3f}", 'BB%'),
               (f"{sp['WHIP']:.2f}", 'WHIP'), (sp['A'], 'A')]
        cards.append(('pitcher', sp['playerName'], sp.get('teamName', ''), 'Starter', top, bot))
    for rp in relievers[:3]:
        top = [(f"{rp['ERA']:.2f}", 'ERA'), (f"{rp['FIP']:.2f}", 'FIP'),
               (f"{rp['OPS Against']:.3f}", 'OPS-A'), (rp['IP'], 'IP'), (f"{rp['K%']:.3f}", 'K%')]
        bot = [(f"{rp['K/9']:.2f}", 'K/9'), (f"{rp['BB%']:.3f}", 'BB%'),
               (f"{rp['WHIP']:.2f}", 'WHIP'), (rp['A'], 'A')]
        cards.append(('pitcher', rp['playerName'], rp.get('teamName', ''), 'Reliever', top, bot))

    n_cards = len(cards)
    n_cols = 3
    n_rows = (n_cards + n_cols - 1) // n_cols

    card_w_in = 4.2
    card_h_in = 2.0
    gap_x_in = 0.3
    gap_y_in = 0.3
    fig_w = n_cols * card_w_in + (n_cols - 1) * gap_x_in + 0.4
    fig_h = n_rows * card_h_in + (n_rows - 1) * gap_y_in + 0.8
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='#1a1a1a')

    # Subtitle only, centered at top
    fig.text(0.5, 1 - 0.25 / fig_h, subtitle, ha='center', va='top', fontsize=11, color='#888')

    def draw_card(ax, name, team, role_label, stats_top, stats_bottom, team_map, ring_color):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.set_facecolor('#222222')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#3a3a3a')
            spine.set_linewidth(1.5)

        # Logo circle — use data coords so it's a true circle
        ring = plt.Circle((1.2, 3.7), 0.55, color=ring_color, zorder=5)
        ax.add_patch(ring)
        eg = plt.Circle((1.2, 3.7), 0.45, color=EGGSHELL, zorder=5)
        ax.add_patch(eg)
        logo = _load_logo_pil(team, team_map, size=120)
        if logo:
            im = OffsetImage(np.array(logo), zoom=0.28)
            ab = AnnotationBbox(im, (1.2, 3.7), frameon=False, zorder=6)
            ax.add_artist(ab)
        else:
            ax.text(1.2, 3.7, _initials(name), ha='center', va='center', fontsize=11,
                    fontweight='bold', color='#333', zorder=7)

        # Name and role
        ax.text(2.5, 4.05, name, ha='left', va='center', fontsize=11,
                fontweight='bold', color='#C8C8C8')
        ax.text(2.5, 3.35, f'{team} · {role_label}', ha='left', va='center',
                fontsize=8, color='#888')

        # Divider
        ax.axhline(y=2.8, color='#3a3a3a', linewidth=0.5)

        # Top stats row
        n_top = len(stats_top)
        for j, (val, label) in enumerate(stats_top):
            cx = (j + 0.5) / n_top * 10
            ax.text(cx, 2.1, str(val), ha='center', va='center', fontsize=13,
                    fontweight='bold', color=LC_RED)
            ax.text(cx, 1.5, label, ha='center', va='center', fontsize=7, color='#888')

        # Bottom stats row
        n_bot = len(stats_bottom)
        for j, (val, label) in enumerate(stats_bottom):
            cx = (j + 0.5) / n_bot * 10
            ax.text(cx, 0.8, str(val), ha='center', va='center', fontsize=11, color='#C8C8C8')
            ax.text(cx, 0.2, label, ha='center', va='center', fontsize=7, color='#888')

    for idx, (ctype, name, team, role, top_stats, bot_stats) in enumerate(cards):
        r = idx // n_cols
        c = idx % n_cols
        left = (0.2 + c * (card_w_in + gap_x_in)) / fig_w
        bottom = 1.0 - (0.6 + (r + 1) * card_h_in + r * gap_y_in) / fig_h
        w_frac = card_w_in / fig_w
        h_frac = card_h_in / fig_h
        card_ax = fig.add_axes([left, bottom, w_frac, h_frac])
        ring_color = LC_RELIEVER_COLOR if role == 'Reliever' else (LC_DH_COLOR if role == 'DH' else LC_RED)
        draw_card(card_ax, name, team, role, top_stats, bot_stats, team_map, ring_color)

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=180, facecolor='#1a1a1a', edgecolor='none', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


# ── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_pbp(sport, division, stat_type):
    """Load a PBP file: sport=baseball|softball, division=D1|D2|D3, stat_type=hitting|pitching|fielding."""
    filepath = PBP_DIR / sport / f'{stat_type}_pbp_{division}.csv'
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath, low_memory=False)
    # Normalize IDs
    if 'playerId' in df.columns:
        df['playerId'] = pd.to_numeric(df['playerId'], errors='coerce').fillna(0).astype(int).astype(str)
    # Clean positions: "PH/LF" → "PH", "/LF/CF" → "LF"
    if 'playerPosition' in df.columns:
        df['playerPosition'] = df['playerPosition'].apply(_clean_position)
    # Parse dates
    if 'date' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    return df


# ── Main ─────────────────────────────────────────────────────────────────────
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

# Sport / Division / View selectors
st.sidebar.markdown('---')
st.sidebar.markdown('### Data Source')
sport = st.sidebar.selectbox('Sport', ['baseball', 'softball'])
division = st.sidebar.selectbox('Division', ['D1', 'D2', 'D3'])
view = st.sidebar.radio('Mode', ['Hitter Stats', 'Pitcher Stats', 'Fielding Stats', 'Lineup Card'], horizontal=True)

# Load data — Lineup Card needs both hitting + pitching
if view == 'Lineup Card':
    hitting_pbp = load_pbp(sport, division, 'hitting')
    pitching_pbp = load_pbp(sport, division, 'pitching')
    if hitting_pbp is None or pitching_pbp is None:
        st.error(f'PBP data not found for {sport} {division}')
        st.stop()
    # Use hitting for date range reference
    pbp = hitting_pbp
else:
    stat_type = {'Hitter Stats': 'hitting', 'Pitcher Stats': 'pitching', 'Fielding Stats': 'fielding'}[view]
    pbp = load_pbp(sport, division, stat_type)
    if pbp is None:
        st.error(f'No {stat_type} PBP data found for {sport} {division}')
        st.stop()

st.sidebar.markdown(f'**{len(pbp):,}** game lines loaded')

# Date range filter
st.sidebar.markdown('---')
st.sidebar.markdown('### Date Range')
date_start, date_end = None, None
if 'date_parsed' in pbp.columns:
    min_date = pbp['date_parsed'].min()
    max_date = pbp['date_parsed'].max()
    if pd.notna(min_date) and pd.notna(max_date):
        date_range = st.sidebar.date_input('Date range',
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(), max_value=max_date.date())
        if len(date_range) == 2:
            date_start, date_end = date_range
            pbp = pbp[(pbp['date_parsed'].dt.date >= date_start) & (pbp['date_parsed'].dt.date <= date_end)]
            if view == 'Lineup Card':
                hitting_pbp = hitting_pbp[(hitting_pbp['date_parsed'].dt.date >= date_start) & (hitting_pbp['date_parsed'].dt.date <= date_end)]
                pitching_pbp = pitching_pbp[(pitching_pbp['date_parsed'].dt.date >= date_start) & (pitching_pbp['date_parsed'].dt.date <= date_end)]
    else:
        st.sidebar.warning('Could not parse dates')

st.sidebar.markdown(f'**{len(pbp):,}** lines in range')

# Conference filter (applies to all views)
conf_map = load_team_conference_map()
if conf_map and 'teamName' in pbp.columns:
    pbp_conferences = pbp['teamName'].map(conf_map).dropna().unique()
    all_conferences = sorted(pbp_conferences)
    selected_conferences = st.sidebar.multiselect('Conference', all_conferences,
                                                    help='Leave empty for all conferences')
    if selected_conferences:
        conf_teams = {t for t, c in conf_map.items() if c in selected_conferences}
        pbp = pbp[pbp['teamName'].isin(conf_teams)]
        if view == 'Lineup Card':
            hitting_pbp = hitting_pbp[hitting_pbp['teamName'].isin(conf_teams)]
            pitching_pbp = pitching_pbp[pitching_pbp['teamName'].isin(conf_teams)]

# Team / Position / Player filters — not shown for Lineup Card
if view != 'Lineup Card':
    # Team filter
    all_teams = sorted(pbp['teamName'].dropna().unique()) if 'teamName' in pbp.columns else []
    team_list = ['All'] + all_teams
    selected_team = st.sidebar.selectbox('Team', team_list)
    if selected_team != 'All':
        pbp = pbp[pbp['teamName'] == selected_team]

    # Position filter
    if 'playerPosition' in pbp.columns:
        all_positions = sorted(pbp['playerPosition'].dropna().unique())
        selected_positions = st.sidebar.multiselect('Position', all_positions,
                                                     help='Leave empty for all positions')
        if selected_positions:
            pbp = pbp[pbp['playerPosition'].isin(selected_positions)]

    # Min PA/BF
    st.sidebar.markdown('---')
    if view == 'Hitter Stats':
        min_threshold = st.sidebar.number_input('Min PA', value=10, min_value=1, step=5)
    elif view == 'Pitcher Stats':
        min_threshold = st.sidebar.number_input('Min BF', value=10, min_value=1, step=5)

    # Player filter
    player_col = 'playerName'
    available_players = sorted(pbp[player_col].dropna().unique()) if player_col in pbp.columns else []
    selected_players = st.sidebar.multiselect('Filter players', available_players,
                                               help='Leave empty for all')
    if selected_players:
        pbp = pbp[pbp[player_col].isin(selected_players)]

    if len(pbp) == 0:
        st.warning('No events match your filters.')
        st.stop()
else:
    # Lineup Card specific controls
    st.sidebar.markdown('---')
    min_pa_lc = st.sidebar.number_input('Min PA (hitters)', value=10, min_value=1, step=5)
    min_bf_sp = st.sidebar.number_input('Min BF (starters)', value=50, min_value=1, step=10)
    min_bf_rp = st.sidebar.number_input('Min BF (relievers)', value=15, min_value=1, step=5)
    player_col = 'playerName'

# ── Compute and Display ─────────────────────────────────────────────────────
if view == 'Hitter Stats':
    st.markdown(f'### Hitter Stats — {sport.title()} {division}')

    # Compute league wOBA for wRAA
    league_stats = compute_hitting_stats(pbp)
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
    player_stats = compute_grouped_hitting(pbp, player_col, league_woba, min_pa=min_threshold)

    if len(player_stats) == 0:
        st.info(f'No players meet the {min_threshold} PA minimum.')
    else:
        show_cols = [player_col, 'Pos', 'PA', 'AB', 'H', '1B', '2B', '3B', 'HR', 'TB',
                     'R', 'RBI', 'BB', 'HBP', 'K', 'SB', 'CS', 'GDP',
                     'BA', 'OBP', 'SLG', 'OPS', 'wOBA', 'wRAA']
        show_cols = [c for c in show_cols if c in player_stats.columns]
        st.dataframe(player_stats[show_cols], use_container_width=True, hide_index=True)

        csv_buf = player_stats[show_cols].to_csv(index=False)
        st.download_button('Download CSV', data=csv_buf,
                          file_name=f'pbp_hitting_{sport}_{division}.csv', mime='text/csv')

    # Single player deep dive
    if selected_players and len(selected_players) == 1:
        st.markdown(f'### {selected_players[0]} — Game Log')
        player_data = pbp[pbp[player_col] == selected_players[0]]

        if 'date_parsed' in player_data.columns:
            game_log = compute_grouped_hitting(player_data, 'date', league_woba, min_pa=1)
            if len(game_log) > 0:
                game_log = game_log.rename(columns={'date': 'Date'})
                game_log = game_log.sort_values('Date')
                gl_cols = ['Date', 'PA', 'AB', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'K',
                          'BA', 'OBP', 'SLG', 'OPS', 'wOBA', 'wRAA']
                gl_cols = [c for c in gl_cols if c in game_log.columns]
                st.dataframe(game_log[gl_cols], use_container_width=True, hide_index=True)

elif view == 'Pitcher Stats':
    st.markdown(f'### Pitcher Stats — {sport.title()} {division}')

    # Overall summary
    overall = compute_pitching_stats(pbp)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('League FIP', f"{overall['FIP']:.2f}")
    c2.metric('League ERA', f"{overall['ERA']:.2f}")
    c3.metric('League OPS Against', f"{overall['OPS Against']:.3f}")
    c4.metric('Total BF', f"{overall['BF']:,}")
    c5.metric('Total SO', f"{overall['SO']:,}")

    # Per-pitcher table
    st.markdown('---')
    pitcher_stats = compute_grouped_pitching(pbp, player_col, min_bf=min_threshold)

    if len(pitcher_stats) == 0:
        st.info(f'No pitchers meet the {min_threshold} BF minimum.')
    else:
        show_cols = [player_col, 'Pos', 'IP', 'BF', 'H', 'ER', 'BB', 'HB', 'SO', 'HR',
                     'WP', 'ERA', 'FIP', 'BAA', 'OBP Against', 'SLG Against', 'OPS Against',
                     'K%', 'BB%']
        show_cols = [c for c in show_cols if c in pitcher_stats.columns]
        st.dataframe(pitcher_stats[show_cols], use_container_width=True, hide_index=True)

        csv_buf = pitcher_stats[show_cols].to_csv(index=False)
        st.download_button('Download CSV', data=csv_buf,
                          file_name=f'pbp_pitching_{sport}_{division}.csv', mime='text/csv')

    # Single pitcher deep dive
    if selected_players and len(selected_players) == 1:
        st.markdown(f'### {selected_players[0]} — Game Log')
        pitcher_data = pbp[pbp[player_col] == selected_players[0]]

        if 'date_parsed' in pitcher_data.columns:
            game_log = compute_grouped_pitching(pitcher_data, 'date', min_bf=1)
            if len(game_log) > 0:
                game_log = game_log.rename(columns={'date': 'Date'})
                game_log = game_log.sort_values('Date')
                gl_cols = ['Date', 'IP', 'BF', 'H', 'ER', 'BB', 'SO', 'HR',
                          'ERA', 'FIP', 'OPS Against']
                gl_cols = [c for c in gl_cols if c in game_log.columns]
                st.dataframe(game_log[gl_cols], use_container_width=True, hide_index=True)

elif view == 'Fielding Stats':
    st.markdown(f'### Fielding Stats — {sport.title()} {division}')

    # Overall summary
    overall = compute_fielding_stats(pbp)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total TC', f"{overall['TC']:,}")
    c2.metric('Total E', f"{overall['E']:,}")
    c3.metric('League FPCT', f"{overall['FPCT']:.3f}")
    c4.metric('League CS%', f"{overall['CS%']:.3f}")

    # Per-player table
    st.markdown('---')
    fielding_stats = compute_grouped_fielding(pbp, player_col)

    if len(fielding_stats) == 0:
        st.info('No fielding data available.')
    else:
        show_cols = [player_col, 'Pos', 'PO', 'A', 'TC', 'E', 'FPCT',
                     'PB', 'SBA', 'CSB', 'CS%', 'IDP', 'TP']
        show_cols = [c for c in show_cols if c in fielding_stats.columns]
        st.dataframe(fielding_stats[show_cols], use_container_width=True, hide_index=True)

        csv_buf = fielding_stats[show_cols].to_csv(index=False)
        st.download_button('Download CSV', data=csv_buf,
                          file_name=f'pbp_fielding_{sport}_{division}.csv', mime='text/csv')

elif view == 'Lineup Card':
    # Build period label from date range
    if date_start and date_end:
        period_label = f"{date_start.strftime('%b %d')} – {date_end.strftime('%b %d, %Y')}"
    else:
        period_label = 'Full Season'

    if len(hitting_pbp) == 0 or len(pitching_pbp) == 0:
        st.warning('No data for selected period.')
        st.stop()

    # Compute best players
    league_stats = compute_hitting_stats(hitting_pbp)
    league_woba = league_stats['wOBA']
    best_hitters = get_best_hitters(hitting_pbp, league_woba, min_pa=min_pa_lc)
    starters, relievers = get_best_pitchers(pitching_pbp, min_bf_sp=min_bf_sp, min_bf_rp=min_bf_rp)

    # Load team logo map
    team_map = load_team_logo_map()

    # Render SVG
    title = f"Players of the Period"
    subtitle = f"{sport.title()} {division} · {period_label}"
    svg = render_lineup_svg(best_hitters, starters, relievers, title, subtitle, team_map)
    st.markdown(svg, unsafe_allow_html=True)

    # Detail cards
    st.markdown('---')
    st.markdown('### Position Players')
    cols = st.columns(3)
    for i, pos in enumerate(FIELD_POSITIONS):
        if pos in best_hitters:
            with cols[i % 3]:
                st.markdown(render_hitter_card_html(best_hitters[pos], pos), unsafe_allow_html=True)

    st.markdown('### Pitchers')
    if starters:
        cols2 = st.columns(min(len(starters), 3))
        for i, sp in enumerate(starters[:3]):
            with cols2[i]:
                st.markdown(render_pitcher_card_html(sp, 'Starter'), unsafe_allow_html=True)

    if relievers:
        cols3 = st.columns(min(len(relievers), 3))
        for i, rp in enumerate(relievers[:3]):
            with cols3[i]:
                st.markdown(render_pitcher_card_html(rp, 'Reliever'), unsafe_allow_html=True)

    # Download PNGs
    st.markdown('---')
    st.markdown('### Download')
    with st.spinner('Rendering PNGs...'):
        diamond_buf = render_diamond_png(best_hitters, starters, relievers, title, subtitle, team_map)
        cards_buf = render_cards_png(best_hitters, starters, relievers, title, subtitle, team_map)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button('Download Diamond PNG', data=diamond_buf,
                          file_name=f'lineup_diamond_{sport}_{division}.png', mime='image/png')
    with dl2:
        st.download_button('Download Cards PNG', data=cards_buf,
                          file_name=f'lineup_cards_{sport}_{division}.png', mime='image/png')
