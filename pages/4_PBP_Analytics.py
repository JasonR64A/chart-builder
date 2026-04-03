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
WOBA_SCALE = 1.6


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

    ibb = df['ibb'].sum() if 'ibb' in df.columns else 0

    # Additional rate stats
    k_pct = (k / pa * 100) if pa > 0 else 0
    bb_pct = (bb / pa * 100) if pa > 0 else 0
    k_bb = k / bb if bb > 0 else k
    iso = slg - ba
    babip_denom = ab - k - hr + sf
    babip = (h - hr) / babip_denom if babip_denom > 0 else 0
    r_pa = r / pa if pa > 0 else 0

    # wRC = (((wOBA - league_wOBA) / wOBA_scale) + (league_R/PA)) * PA
    # We'll store wOBA raw and compute wRC in the grouped function where we have league context

    return {
        'PA': int(pa), 'AB': int(ab), 'H': int(h), '1B': int(singles),
        '2B': int(doubles), '3B': int(triples), 'HR': int(hr), 'TB': int(tb),
        'R': int(r), 'RBI': int(rbi), 'BB': int(bb), 'HBP': int(hbp),
        'SF': int(sf), 'SH': int(sh), 'IBB': int(ibb), 'K': int(k),
        'SB': int(sb), 'CS': int(cs), 'GDP': int(gdp),
        'BA': round(ba, 3), 'OBP': round(obp, 3), 'SLG': round(slg, 3),
        'OPS': round(ops, 3), 'ISO': round(iso, 3), 'BABIP': round(babip, 3),
        'wOBA': round(woba, 3), 'K%': round(k_pct, 1), 'BB%': round(bb_pct, 1),
        'K/BB': round(k_bb, 2), 'R/PA': round(r_pa, 3),
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
    wp = df['wp'].sum()
    bk = df['bk'].sum()
    doubles = df['doublesA'].sum()
    triples = df['triplesA'].sum()
    singles = h - doubles - triples - hr
    sha = df['sha'].sum()
    sfa = df['sfa'].sum()
    er = df['er'].sum()
    ibb = df['ibb'].sum() if 'ibb' in df.columns else 0
    app = df['gameId'].nunique() if 'gameId' in df.columns else len(df)

    # GS: count games where this pitcher was the starter (first listed for their team)
    gs = int(df['is_starter'].sum()) if 'is_starter' in df.columns else 0

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

    # Rate stats
    k_pct = (so / bf * 100) if bf > 0 else 0
    bb_pct = (bb / bf * 100) if bf > 0 else 0
    k9 = (so / ip) * 9 if ip > 0 else 0
    k7 = (so / ip) * 7 if ip > 0 else 0
    k_bb = so / bb if bb > 0 else so
    whip = (bb + h) / ip if ip > 0 else 0

    # BABIP against
    babip_denom = p_oab - so - hr + sfa
    babip = (h - hr) / babip_denom if babip_denom > 0 else 0

    return {
        'IP': outs_to_ip(total_outs), 'App': int(app), 'GS': int(gs),
        'BF': int(bf), 'OAB': int(p_oab),
        'H': int(h), 'R': int(r), 'ER': int(er),
        'BB': int(bb), 'HB': int(hb), 'SO': int(so),
        'HR': int(hr), '2B-A': int(doubles), '3B-A': int(triples),
        'WP': int(wp), 'Bk': int(bk), 'IBB': int(ibb),
        'SHA': int(sha), 'SFA': int(sfa),
        'ERA': round(era, 2), 'FIP': round(fip, 2),
        'BAA': round(ba_against, 3), 'BABIP': round(babip, 3),
        'OBP Against': round(obp_against, 3),
        'SLG Against': round(slg_against, 3),
        'OPS Against': round(ops_against, 3),
        'K%': round(k_pct, 1), 'BB%': round(bb_pct, 1),
        'K/9': round(k9, 2), 'K/7': round(k7, 2),
        'K/BB': round(k_bb, 2), 'WHIP': round(whip, 2),
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


def compute_wrc(woba, league_woba, league_r_pa, pa):
    """wRC = (((wOBA - league_wOBA) / wOBA_scale) + league_R/PA) * PA"""
    return round((((woba - league_woba) / WOBA_SCALE) + league_r_pa) * pa, 1)


def compute_wrc_plus(wrc_per_pa, league_r_pa):
    """wRC+ = (wRC/PA) / (league R/PA) * 100"""
    return round((wrc_per_pa / league_r_pa) * 100, 0) if league_r_pa > 0 else 100


def _primary_position(group):
    """Return formal position from players.csv, fallback to most common game position."""
    if 'formalPosition' in group.columns:
        formal = group['formalPosition'].dropna()
        if len(formal) > 0:
            return formal.iloc[0]
    if 'playerPosition' in group.columns:
        pos = group['playerPosition'].dropna()
        if len(pos) > 0:
            return pos.value_counts().index[0]
    return ''


def _player_school(group):
    """Return school from players.csv lookup."""
    if 'school' in group.columns:
        school = group['school'].dropna()
        if len(school) > 0:
            return school.iloc[0]
    return ''


def compute_grouped_hitting(df, group_col, league_woba, league_r_pa=0, min_pa=1):
    """Compute hitting stats grouped by a column."""
    rows = []
    for name, group in df.groupby(group_col):
        stats = compute_hitting_stats(group)
        if stats['PA'] >= min_pa:
            stats['wRAA'] = compute_wraa(stats['wOBA'], league_woba, stats['PA'])
            wrc = compute_wrc(stats['wOBA'], league_woba, league_r_pa, stats['PA'])
            stats['wRC'] = wrc
            wrc_per_pa = wrc / stats['PA'] if stats['PA'] > 0 else 0
            stats['wRC+'] = int(compute_wrc_plus(wrc_per_pa, league_r_pa))
            stats[group_col] = name
            stats['Pos'] = _primary_position(group)
            stats['School'] = _player_school(group)
            rows.append(stats)
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    # Combined rank: percentile rank of wRAA + percentile rank of OPS + percentile rank of TB
    n = len(result)
    if n > 0:
        result['wraa_pctl'] = result['wRAA'].rank(pct=True, method='min')
        result['ops_pctl'] = result['OPS'].rank(pct=True, method='min')
        result['tb_pctl'] = result['TB'].rank(pct=True, method='min')
        result['Rank'] = round(result['wraa_pctl'] + result['ops_pctl'] + result['tb_pctl'], 6)
        result = result.drop(columns=['wraa_pctl', 'ops_pctl', 'tb_pctl'])
    cols = ['Rank'] + [group_col] + [c for c in result.columns if c not in ['Rank', group_col]]
    return result[cols].sort_values('Rank', ascending=False).reset_index(drop=True)


def compute_game_score(row):
    """Bill James Game Score for a single game appearance.
    50 + 1(outs) + 2(innings after 4th) + 1(K) - 2(H) - 4(ER) - 2(UER) - 1(BB)"""
    ip = float(row.get('ip', 0))
    whole = int(ip)
    frac = round((ip - whole) * 10)
    outs = whole * 3 + frac
    innings_complete = outs // 3
    innings_after_4th = max(0, innings_complete - 4)

    so = int(row.get('so', 0))
    h = int(row.get('h', 0))
    er = int(row.get('er', 0))
    r = int(row.get('r', 0))
    uer = r - er
    bb = int(row.get('bb', 0))

    return 50 + outs + 2 * innings_after_4th + so - 2 * h - 4 * er - 2 * uer - bb


def compute_grouped_pitching(df, group_col, min_bf=1):
    """Compute pitching stats grouped by a column."""
    rows = []
    for name, group in df.groupby(group_col):
        stats = compute_pitching_stats(group)
        if stats['BF'] >= min_bf:
            stats[group_col] = name
            stats['Pos'] = _primary_position(group)
            stats['School'] = _player_school(group)
            # Average Game Score across all appearances
            game_scores = group.apply(compute_game_score, axis=1)
            stats['GmSc'] = round(game_scores.mean(), 1)
            rows.append(stats)
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    # Pitcher Rank: FIP rank + A-OPS rank + 2 * Game Score rank
    # Lower FIP/OPS = better (rank descending), higher GmSc = better (rank ascending)
    n = len(result)
    if n > 0:
        result['fip_score'] = (n - result['FIP'].rank(method='min') + 1) / n
        result['ops_a_score'] = (n - result['OPS Against'].rank(method='min') + 1) / n
        result['gmsc_score'] = result['GmSc'].rank(method='min', ascending=True) / n
        result['Rank'] = round(
            result['fip_score'] + result['ops_a_score'] + 2 * result['gmsc_score'], 6
        )
        result = result.drop(columns=['fip_score', 'ops_a_score', 'gmsc_score'])
    cols = ['Rank'] + [group_col] + [c for c in result.columns if c not in ['Rank', group_col]]
    return result[cols].sort_values('Rank', ascending=False).reset_index(drop=True)


def compute_grouped_fielding(df, group_col):
    """Compute fielding stats grouped by a column."""
    rows = []
    for name, group in df.groupby(group_col):
        stats = compute_fielding_stats(group)
        if stats['TC'] > 0:
            stats[group_col] = name
            stats['Pos'] = _primary_position(group)
            stats['School'] = _player_school(group)
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
def load_team_logo_map(prefer_sport='baseball'):
    """Build team_name -> logo_id mapping from teams.csv.
    For the preferred sport, use that sport's team ID if a logo file exists.
    Otherwise fall back to baseball ID (canonical)."""
    teams_path = DATA_DIR / 'teams.csv'
    if not teams_path.exists():
        return {}
    teams = pd.read_csv(teams_path, low_memory=False)
    teams['id'] = pd.to_numeric(teams['id'], errors='coerce').fillna(0).astype(int)
    # Baseball IDs are canonical baseline
    bb = teams[teams['sport'] == 'Baseball'][['name', 'id']].drop_duplicates('name')
    name_to_id = dict(zip(bb['name'], bb['id']))
    # Add softball teams for schools without baseball
    sb = teams[teams['sport'] == 'Softball'][['name', 'id']].drop_duplicates('name')
    for _, row in sb.iterrows():
        if row['name'] not in name_to_id:
            name_to_id[row['name']] = row['id']
    # If preferred sport is softball, override with softball ID where a logo file exists
    if prefer_sport == 'softball':
        for _, row in sb.iterrows():
            logo_path = LOGO_DIR / f"{row['id']}.png"
            if logo_path.exists():
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


@st.cache_data
def get_team_color(team_name):
    """Extract dominant color from a team's logo."""
    from collections import Counter
    team_map = load_team_logo_map()
    logo_id = team_map.get(team_name)
    if not logo_id:
        return '#C41230'
    for ext in ['png', 'webp']:
        p = LOGO_DIR / f'{logo_id}.{ext}'
        if p.exists():
            img = Image.open(p).convert('RGBA')
            img.thumbnail((64, 64))
            pixels = np.array(img)
            mask = pixels[:, :, 3] > 128
            rgb = pixels[mask][:, :3]
            if len(rgb) == 0:
                return '#C41230'
            filtered = []
            for r, g, b in rgb:
                brightness = (int(r) + int(g) + int(b)) / 3
                if brightness > 220 or brightness < 35:
                    continue
                filtered.append((r, g, b))
            if not filtered:
                return '#C41230'
            quantized = [(r // 16 * 16, g // 16 * 16, b // 16 * 16) for r, g, b in filtered]
            most_common = Counter(quantized).most_common(1)[0][0]
            return f'#{most_common[0]:02x}{most_common[1]:02x}{most_common[2]:02x}'
    return '#C41230'


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
    gs = int(df['is_starter'].sum()) if 'is_starter' in df.columns else 0
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
    k7 = (so / ip) * 7 if ip > 0 else 0
    bb_pct = (bb / bf * 100) if bf > 0 else 0
    k_pct_100 = (so / bf * 100) if bf > 0 else 0
    whip = (bb + h) / ip if ip > 0 else 0
    k_bb = so / bb if bb > 0 else so
    return {'IP': _outs_to_ip_display(total_outs), 'IP_actual': ip,
            'BF': int(bf), 'H': int(h), 'ER': int(er), 'BB': int(bb),
            'SO': int(so), 'HR': int(hr), 'A': int(games), 'GS': int(gs),
            'ERA': round(era, 2), 'FIP': round(fip, 2),
            'K%': round(k_pct_100, 1), 'BB%': round(bb_pct, 1),
            'K/9': round(k9, 2), 'K/7': round(k7, 2), 'K/BB': round(k_bb, 2), 'WHIP': round(whip, 2),
            'OPS Against': round(ops_a, 3)}


def get_best_hitters(hitting_df, league_woba, min_pa=10, sport='baseball'):
    best = {}
    dh_label = 'DP' if sport == 'softball' else 'DH'

    # Step 1: Determine each player's predominant position and compute stats from ALL their PA
    player_data = []
    for name, group in hitting_df.groupby('playerName'):
        stats = compute_hitting_stats(group)
        if stats['PA'] >= min_pa:
            # Predominant position = most games played at that position
            pos_counts = group['playerPosition'].value_counts()
            predominant_pos = pos_counts.index[0] if len(pos_counts) > 0 else ''
            # Normalize DP to DH for softball
            if sport == 'softball' and predominant_pos == 'DP':
                predominant_pos = 'DH'
            stats['playerName'] = name
            stats['teamName'] = group['teamName'].mode().iloc[0] if len(group['teamName'].mode()) > 0 else ''
            stats['wRAA'] = compute_wraa(stats['wOBA'], league_woba, stats['PA'])
            stats['_pos'] = predominant_pos
            player_data.append(stats)

    if not player_data:
        return best

    # Step 2: Compute combined rank using the FULL player population (min 1 PA)
    # so rankings match the Hitter Stats table regardless of lineup card min PA.
    # Then filter to min_pa for eligibility.
    all_players_full = []
    for name, group in hitting_df.groupby('playerName'):
        stats = compute_hitting_stats(group)
        if stats['PA'] >= 1:
            pos_counts = group['playerPosition'].value_counts()
            predominant_pos = pos_counts.index[0] if len(pos_counts) > 0 else ''
            if sport == 'softball' and predominant_pos == 'DP':
                predominant_pos = 'DH'
            stats['playerName'] = name
            stats['teamName'] = group['teamName'].mode().iloc[0] if len(group['teamName'].mode()) > 0 else ''
            stats['wRAA'] = compute_wraa(stats['wOBA'], league_woba, stats['PA'])
            stats['_pos'] = predominant_pos
            all_players_full.append(stats)

    if not all_players_full:
        return best

    all_df = pd.DataFrame(all_players_full)
    n = len(all_df)
    if n > 0:
        all_df['wraa_pctl'] = all_df['wRAA'].rank(pct=True, method='min')
        all_df['ops_pctl'] = all_df['OPS'].rank(pct=True, method='min')
        all_df['tb_pctl'] = all_df['TB'].rank(pct=True, method='min')
        all_df['Rank'] = all_df['wraa_pctl'] + all_df['ops_pctl'] + all_df['tb_pctl']
        all_df = all_df.drop(columns=['wraa_pctl', 'ops_pctl', 'tb_pctl'])

    # Step 3: Filter to min_pa eligible, then pick highest-ranked per position
    eligible = all_df[all_df['PA'] >= min_pa]
    for pos in FIELD_POSITIONS:
        pos_players = eligible[eligible['_pos'] == pos]
        if len(pos_players) == 0:
            continue
        top = pos_players.sort_values('Rank', ascending=False).iloc[0]
        best[pos] = top.to_dict()
    return best


def _combined_rank(df, pitching_raw_df=None):
    """Rank pitchers by FIP rank + OPS Against rank + 2 * Game Score rank.
    Best (lowest) FIP/OPS gets n/n = 1.000. Best (highest) GmSc gets n/n = 1.000.
    """
    n = len(df)
    if n == 0:
        return df
    df = df.copy()
    # Compute average Game Score for each pitcher from raw game data
    if pitching_raw_df is not None:
        gmsc_map = {}
        for name, group in pitching_raw_df.groupby('playerName'):
            scores = group.apply(compute_game_score, axis=1)
            gmsc_map[name] = scores.mean()
        df['GmSc'] = df['playerName'].map(gmsc_map).fillna(50.0)
    elif 'GmSc' not in df.columns:
        df['GmSc'] = 50.0
    df['fip_score'] = (n - df['FIP'].rank(method='min') + 1) / n
    df['ops_a_score'] = (n - df['OPS Against'].rank(method='min') + 1) / n
    df['gmsc_score'] = df['GmSc'].rank(method='min', ascending=True) / n
    df['combined_score'] = df['fip_score'] + df['ops_a_score'] + 2 * df['gmsc_score']
    return df.sort_values('combined_score', ascending=False)


def get_best_pitchers(pitching_df, min_bf_sp=50, min_bf_rp=15, n_starters=3, n_relievers=3):
    rows = []
    for name, group in pitching_df.groupby('playerName'):
        stats = compute_pitching_for_lineup(group)
        stats['playerName'] = name
        stats['teamName'] = group['teamName'].mode().iloc[0] if len(group['teamName'].mode()) > 0 else ''
        # Starter = majority of appearances were starts (first pitcher for team in game)
        starts = int(group['is_starter'].sum()) if 'is_starter' in group.columns else 0
        stats['GS'] = starts
        stats['is_starter'] = starts > 0  # anyone with even 1 start is a starter, not a reliever
        rows.append(stats)
    if not rows:
        return [], []
    df = pd.DataFrame(rows)
    starter_df = _combined_rank(df[(df['is_starter']) & (df['BF'] >= min_bf_sp)], pitching_df)
    reliever_df = _combined_rank(df[(~df['is_starter']) & (df['BF'] >= min_bf_rp)], pitching_df)
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
    'CF': (190, 35), 'LF': (55, 125), 'RF': (325, 125),
    'SS': (148, 205), '2B': (232, 205),
    '3B': (100, 275), '1B': (280, 275),
    'C': (190, 340), 'DH': (325, 340),
}


EGGSHELL = '#f5efe0'


def _name_with_stroke(name, y_offset, font_size=9):
    """SVG text with white stroke outline for pop, then black fill on top. Auto-sizes for long names."""
    n = len(name)
    if n > 18:
        font_size = max(font_size - 2, 6)
    elif n > 14:
        font_size = max(font_size - 1, 7)
    return f'''<text font-size="{font_size}" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linejoin="round"
          text-anchor="middle" y="{y_offset}" font-family="sans-serif" font-weight="bold">{name}</text>
    <text font-size="{font_size}" fill="#111111" text-anchor="middle" y="{y_offset}"
          font-family="sans-serif" font-weight="bold">{name}</text>'''


def _logo_node(x, y, player, pos, team_map, ring_color, r=22, r_inner=19, sport='baseball'):
    """Render a player node with team logo in circle and name below. DH/DP gets position label."""
    name = player['playerName']
    team = player.get('teamName', '')
    logo_id = team_map.get(team)
    logo_b64 = get_logo_base64(logo_id) if logo_id else None
    clip_id = f"clip-{pos}-{x}-{y}"
    dh_label = 'DP' if sport == 'softball' else 'DH'
    pos_label = f'''<text font-size="7" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linejoin="round"
          text-anchor="middle" y="{r+19}" font-family="sans-serif" font-weight="bold">{dh_label}</text>
    <text font-size="7" fill="#111111" text-anchor="middle" y="{r+19}"
          font-family="sans-serif" font-weight="bold">{dh_label}</text>''' if pos == 'DH' else ''

    if logo_b64:
        return f'''<g transform="translate({x},{y})">
    <circle r="{r}" fill="{ring_color}"/>
    <circle r="{r_inner}" fill="{EGGSHELL}"/>
    <clipPath id="{clip_id}"><circle r="{r_inner}"/></clipPath>
    <image href="{logo_b64}" x="-{r_inner}" y="-{r_inner}" width="{r_inner*2}" height="{r_inner*2}" clip-path="url(#{clip_id})" preserveAspectRatio="xMidYMid slice"/>
    {_name_with_stroke(name, r + 10, font_size=9)}
    {pos_label}</g>'''
    else:
        ini = _initials(name)
        return f'''<g transform="translate({x},{y})">
    <circle r="{r}" fill="{ring_color}"/><circle r="{r_inner}" fill="{EGGSHELL}"/>
    <text font-size="10" font-weight="500" fill="#333" text-anchor="middle" dominant-baseline="central" font-family="sans-serif">{ini}</text>
    {_name_with_stroke(name, r + 10, font_size=9)}
    {pos_label}</g>'''


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
    {_name_with_stroke(name, 27, font_size=8)}</g>'''
    else:
        ini = _initials(name)
        return f'''<g transform="translate({x},{y})">
    <circle r="{r}" fill="{ring_color}"/><circle r="{r_inner}" fill="{EGGSHELL}"/>
    <text font-size="9" font-weight="500" fill="#333" text-anchor="middle" dominant-baseline="central" font-family="sans-serif">{ini}</text>
    {_name_with_stroke(name, 27, font_size=8)}</g>'''


def render_lineup_svg(best_hitters, starters, relievers, title, subtitle, team_map, date_label='', sport='baseball'):
    nodes = []
    for pos, (x, y) in POS_COORDS.items():
        if pos in best_hitters:
            color = LC_DH_COLOR if pos == 'DH' else LC_RED
            nodes.append(_logo_node(x, y, best_hitters[pos], pos, team_map, color, sport=sport))
        else:
            nodes.append(f'''<g transform="translate({x},{y})">
    <circle r="22" fill="#555"/><circle r="19" fill="#1c2a38"/>
    <text font-size="9" fill="#666" text-anchor="middle" dominant-baseline="central" font-family="sans-serif">—</text>
    <text font-size="7" fill="#666" text-anchor="middle" y="37" font-family="sans-serif">{pos}</text></g>''')

    # Pitcher sidebar (right column)
    sx = 423
    nodes.append(f'<line x1="385" y1="10" x2="385" y2="390" stroke="#3a3a3a" stroke-width="1"/>')
    nodes.append(f'''<text x="{sx}" y="26" font-size="8" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linejoin="round" font-weight="bold" text-anchor="middle" letter-spacing="0.08em" font-family="sans-serif">STARTERS</text>
    <text x="{sx}" y="26" font-size="8" fill="#111111" font-weight="bold" text-anchor="middle" letter-spacing="0.08em" font-family="sans-serif">STARTERS</text>''')
    for i in range(3):
        y = 58 + i * 56
        if i < len(starters):
            nodes.append(_pitcher_logo_node(sx, y, starters[i], team_map, LC_RED))
        else:
            nodes.append(f'<g transform="translate({sx},{y})"><circle r="20" fill="#555"/><circle r="17" fill="#1c2a38"/><text font-size="9" fill="#666" text-anchor="middle" dominant-baseline="central" font-family="sans-serif">—</text></g>')

    nodes.append(f'''<text x="{sx}" y="218" font-size="8" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linejoin="round" font-weight="bold" text-anchor="middle" letter-spacing="0.08em" font-family="sans-serif">RELIEVERS</text>
    <text x="{sx}" y="218" font-size="8" fill="#111111" font-weight="bold" text-anchor="middle" letter-spacing="0.08em" font-family="sans-serif">RELIEVERS</text>''')
    for i in range(3):
        y = 246 + i * 56
        if i < len(relievers):
            nodes.append(_pitcher_logo_node(sx, y, relievers[i], team_map, LC_RELIEVER_COLOR))
        else:
            nodes.append(f'<g transform="translate({sx},{y})"><circle r="20" fill="#555"/><circle r="17" fill="#1c2a38"/><text font-size="9" fill="#666" text-anchor="middle" dominant-baseline="central" font-family="sans-serif">—</text></g>')

    # Brand logos
    brand_b64 = None
    brand_path = _APP_DIR / 'assets' / 'brand_logo_dark.png'
    if brand_path.exists():
        brand_data = brand_path.read_bytes()
        brand_b64 = f'data:image/png;base64,{base64.b64encode(brand_data).decode()}'

    wide_logo_b64 = None
    wide_logo_path = _APP_DIR / 'assets' / 'brand_logo_wide.png'
    if wide_logo_path.exists():
        wide_data = wide_logo_path.read_bytes()
        wide_logo_b64 = f'data:image/png;base64,{base64.b64encode(wide_data).decode()}'

    svg = f'''<svg width="100%" viewBox="0 -10 560 430" xmlns="http://www.w3.org/2000/svg">
  <!-- Diamond (left column, centered at x=190, shifted down 20px) -->
  <path d="M190,365 L-5,80 Q190,-20 385,80 Z" fill="#2d8a45"/>
  <path d="M-5,80 Q190,-20 385,80" fill="none" stroke="{LC_RED}" stroke-width="10"/>
  <line x1="190" y1="365" x2="-5" y2="80" stroke="{LC_RED}" stroke-width="8"/>
  <line x1="190" y1="365" x2="385" y2="80" stroke="{LC_RED}" stroke-width="8"/>
  <path d="M190,365 L78,230 Q190,155 302,230 Z" fill="#c8883a"/>
  <path d="M190,365 L94,240 Q190,170 286,240 Z" fill="#2d8a45"/>
  <!-- 2nd base bag: centered between SS(148,205) and 2B(232,205), above them -->
  <rect x="182" y="178" width="16" height="16" rx="2" fill="#f5efe0" transform="rotate(45 190 186)"/>
  <!-- 1st base bag -->
  <rect x="278" y="238" width="14" height="14" rx="2" fill="#f5efe0" transform="rotate(45 285 245)"/>
  <!-- 3rd base bag -->
  <rect x="98" y="238" width="14" height="14" rx="2" fill="#f5efe0" transform="rotate(45 105 245)"/>
  <!-- Home plate -->
  <polygon points="190,352 180,342 180,332 200,332 200,342" fill="#f5efe0"/>
  <!-- Mound -->
  <circle cx="190" cy="270" r="9" fill="#b87830" opacity="0.9"/>
  <circle cx="190" cy="270" r="4" fill="#a06820"/>
  <!-- Brand logo in center field -->
  {f'<image href="{brand_b64}" x="128" y="75" width="125" height="125" opacity="0.9" preserveAspectRatio="xMidYMid meet"/>' if brand_b64 else ''}
  {chr(10).join(nodes)}
  <!-- Branded info tile centered on LF x-axis (x=55) at C/DH height -->
  <rect x="15" y="318" rx="8" ry="8" width="80" height="45" fill="#222222" stroke="#FFFFFF" stroke-width="1.5" opacity="0.9"/>
  {f'<image href="{wide_logo_b64}" x="20" y="323" width="70" height="18" opacity="0.95" preserveAspectRatio="xMidYMid meet"/>' if wide_logo_b64 else ''}
  <text x="55" y="352" font-size="6" fill="#aaaaaa" text-anchor="middle" font-family="sans-serif">{date_label}</text>
</svg>
'''
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


def render_pitcher_card_html(p, role='Starter', sport='baseball'):
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
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:17px;color:#C8C8C8;">{p['K%']:.1f}</div><div style="font-size:9px;color:#888;">K%</div></div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;">
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:15px;color:#C8C8C8;">{p['K/7' if sport == 'softball' else 'K/9']:.2f}</div><div style="font-size:9px;color:#888;">{'K/7' if sport == 'softball' else 'K/9'}</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:15px;color:#C8C8C8;">{p['K/BB']:.2f}</div><div style="font-size:9px;color:#888;">K/BB</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:15px;color:#C8C8C8;">{p['BB%']:.1f}</div><div style="font-size:9px;color:#888;">BB%</div></div>
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


def _load_bg_image():
    """Load the branded background pattern."""
    bg_path = _APP_DIR / 'assets' / 'bg_pattern.jpg'
    if bg_path.exists():
        return Image.open(bg_path).convert('RGB')
    return None


def render_diamond_png(best_hitters, starters, relievers, title, subtitle, team_map):
    """Render the lineup card diamond as a matplotlib PNG."""
    fig = plt.figure(figsize=(12, 10), facecolor='#2c2c2a')

    # Background pattern — cover entire figure including title area
    bg_img = _load_bg_image()
    if bg_img:
        bg_ax = fig.add_axes([-0.05, -0.05, 1.1, 1.1])
        bg_ax.imshow(np.array(bg_img), aspect='auto', extent=[0, 1, 0, 1], zorder=0)
        bg_ax.axis('off')

    ax = fig.add_axes([0, 0, 0.85, 1])
    ax.set_xlim(0, 460)
    ax.set_ylim(0, 420)
    ax.set_facecolor('none')
    ax.patch.set_alpha(0)
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

    # Brand logo in CF — match the website SVG size (~125px)
    brand_path = _APP_DIR / 'assets' / 'brand_logo_dark.png'
    if brand_path.exists():
        brand_img = Image.open(brand_path).convert('RGBA')
        brand_img.thumbnail((250, 250), Image.LANCZOS)
        brand_arr = np.array(brand_img)
        brand_im = OffsetImage(brand_arr, zoom=0.45)
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
                # Resize logo to fit circle and center it
                logo_resized = logo.resize((42, 42), Image.LANCZOS)
                arr = np.array(logo_resized)
                im = OffsetImage(arr, zoom=1.0)
                ab = AnnotationBbox(im, (px, py - 1), frameon=False, zorder=6)
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

    # Pitcher sidebar — use data coords with equal aspect for true circles
    sidebar_ax = fig.add_axes([0.85, 0.02, 0.14, 0.96])
    sidebar_ax.set_xlim(0, 60)
    sidebar_ax.set_ylim(0, 420)
    sidebar_ax.set_aspect('equal')
    sidebar_ax.set_facecolor('none')
    sidebar_ax.patch.set_alpha(0)
    sidebar_ax.axis('off')

    sidebar_ax.plot([2, 2], [10, 410], color='#3a3a3a', linewidth=1)
    sidebar_ax.text(30, 405, 'STARTERS', ha='center', va='top', fontsize=8,
                    color='#a89880', fontweight='bold')

    for i in range(3):
        cy = 370 - i * 65
        if i < len(starters):
            sp = starters[i]
            logo = _load_logo_pil(sp.get('teamName', ''), team_map, size=100)
            ring = plt.Circle((30, cy), 18, color=LC_RED, zorder=5)
            sidebar_ax.add_patch(ring)
            eg = plt.Circle((30, cy), 15, color=EGGSHELL, zorder=5)
            sidebar_ax.add_patch(eg)
            if logo:
                logo_r = logo.resize((28, 28), Image.LANCZOS)
                im = OffsetImage(np.array(logo_r), zoom=1.0)
                ab = AnnotationBbox(im, (30, cy), frameon=False, zorder=6)
                sidebar_ax.add_artist(ab)
            sidebar_ax.text(30, cy - 22, sp['playerName'], ha='center', va='top',
                           fontsize=6, color='#c8a880', zorder=7)
        else:
            ring = plt.Circle((30, cy), 18, color='#555', zorder=5)
            sidebar_ax.add_patch(ring)

    sidebar_ax.text(30, 200, 'RELIEVERS', ha='center', va='top', fontsize=8,
                    color='#a89880', fontweight='bold')

    for i in range(3):
        cy = 170 - i * 65
        if i < len(relievers):
            rp = relievers[i]
            logo = _load_logo_pil(rp.get('teamName', ''), team_map, size=100)
            ring = plt.Circle((30, cy), 18, color=LC_RELIEVER_COLOR, zorder=5)
            sidebar_ax.add_patch(ring)
            eg = plt.Circle((30, cy), 15, color=EGGSHELL, zorder=5)
            sidebar_ax.add_patch(eg)
            if logo:
                logo_r = logo.resize((28, 28), Image.LANCZOS)
                im = OffsetImage(np.array(logo_r), zoom=1.0)
                ab = AnnotationBbox(im, (30, cy), frameon=False, zorder=6)
                sidebar_ax.add_artist(ab)
            sidebar_ax.text(30, cy - 22, rp['playerName'], ha='center', va='top',
                           fontsize=6, color='#c090e8', zorder=7)
        else:
            ring = plt.Circle((30, cy), 18, color='#555', zorder=5)
            sidebar_ax.add_patch(ring)

    # Title
    fig.text(0.42, 0.97, title, ha='center', va='top', fontsize=16, fontweight='bold', color='#C8C8C8')
    fig.text(0.42, 0.94, subtitle, ha='center', va='top', fontsize=10, color='#888')

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=180, facecolor='#2c2c2a', edgecolor='none',
                bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return buf


def render_cards_png(best_hitters, starters, relievers, title, subtitle, team_map, sport='baseball'):
    """Render player detail cards as a matplotlib PNG."""
    # Build card data
    k_key = 'K/7' if sport == 'softball' else 'K/9'
    k_label = 'K/7' if sport == 'softball' else 'K/9'
    dh_label = 'DP' if sport == 'softball' else 'DH'
    cards = []
    for pos in FIELD_POSITIONS:
        if pos in best_hitters:
            p = best_hitters[pos]
            display_pos = dh_label if pos == 'DH' else pos
            top = [(f"{p['OPS']:.3f}", 'OPS'), (f"{p['wOBA']:.3f}", 'wOBA'),
                   (f"{p['wRAA']:.1f}", 'wRAA'), (f"{p['BA']:.3f}", 'AVG')]
            bot = [(p['HR'], 'HR'), (p['RBI'], 'RBI'), (p['R'], 'R'),
                   (p['BB'], 'BB'), (p['SB'], 'SB')]
            cards.append(('hitter', p['playerName'], p.get('teamName', ''), display_pos, top, bot))
    for sp in starters[:3]:
        top = [(f"{sp['ERA']:.2f}", 'ERA'), (f"{sp['FIP']:.2f}", 'FIP'),
               (f"{sp['OPS Against']:.3f}", 'OPS-A'), (sp['IP'], 'IP'), (f"{sp['K%']:.1f}", 'K%')]
        bot = [(f"{sp[k_key]:.2f}", k_label), (f"{sp['K/BB']:.2f}", 'K/BB'),
               (f"{sp['WHIP']:.2f}", 'WHIP'), (sp['A'], 'A')]
        cards.append(('pitcher', sp['playerName'], sp.get('teamName', ''), 'Starter', top, bot))
    for rp in relievers[:3]:
        top = [(f"{rp['ERA']:.2f}", 'ERA'), (f"{rp['FIP']:.2f}", 'FIP'),
               (f"{rp['OPS Against']:.3f}", 'OPS-A'), (rp['IP'], 'IP'), (f"{rp['K%']:.1f}", 'K%')]
        bot = [(f"{rp[k_key]:.2f}", k_label), (f"{rp['K/BB']:.2f}", 'K/BB'),
               (f"{rp['WHIP']:.2f}", 'WHIP'), (rp['A'], 'A')]
        cards.append(('pitcher', rp['playerName'], rp.get('teamName', ''), 'Reliever', top, bot))

    n_cards = len(cards)
    n_cols = 3
    n_rows = (n_cards + n_cols - 1) // n_cols

    card_w_in = 4.2
    card_h_in = 2.0
    gap_x_in = 0.15
    gap_y_in = 0.15
    fig_w = n_cols * card_w_in + (n_cols - 1) * gap_x_in + 0.4
    fig_h = n_rows * card_h_in + (n_rows - 1) * gap_y_in + 0.8
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='#1a1a1a')

    # Background pattern
    bg_img = _load_bg_image()
    if bg_img:
        bg_ax = fig.add_axes([0, 0, 1, 1])
        bg_ax.imshow(np.array(bg_img), aspect='auto', extent=[0, 1, 0, 1], zorder=0)
        bg_ax.axis('off')

    # Subtitle only, centered at top
    fig.text(0.5, 1 - 0.25 / fig_h, subtitle, ha='center', va='top', fontsize=11, color='#C8C8C8')

    # Map role labels for display
    ROLE_DISPLAY = {'Starter': 'SP', 'Reliever': 'RP'}

    def draw_card(fig, left, bottom, w_frac, h_frac, name, team, role_label, stats_top, stats_bottom, team_map, ring_color):
        display_role = ROLE_DISPLAY.get(role_label, role_label)

        # Shadow rect (slightly offset)
        shadow_ax = fig.add_axes([left + 0.004, bottom - 0.006, w_frac, h_frac])
        shadow_ax.set_xlim(-0.5, 10.5); shadow_ax.set_ylim(-0.5, 5.5)
        shadow_ax.axis('off')
        shadow_bg = mpatches.FancyBboxPatch((0, 0), 10, 5,
            boxstyle='round,pad=0.3', facecolor='#000000', alpha=0.5, edgecolor='none',
            zorder=1, clip_on=False)
        shadow_ax.add_patch(shadow_bg)

        # Main card — expand limits so rounded corners + stroke aren't clipped
        ax = fig.add_axes([left, bottom, w_frac, h_frac])
        ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.5, 5.5)
        ax.axis('off')

        # Rounded card background with white stroke
        card_bg = mpatches.FancyBboxPatch((0, 0), 10, 5,
            boxstyle='round,pad=0.3', facecolor='#222222', edgecolor='#FFFFFF',
            linewidth=2.5, zorder=2, clip_on=False)
        ax.add_patch(card_bg)

        # Logo circle — centered in the name/school row (between divider at 2.85 and top at 5.0)
        logo_cy = 3.95
        ring = plt.Circle((1.2, logo_cy), 0.55, color=ring_color, zorder=5, clip_on=False)
        ax.add_patch(ring)
        eg = plt.Circle((1.2, logo_cy), 0.45, color=EGGSHELL, zorder=5, clip_on=False)
        ax.add_patch(eg)
        logo = _load_logo_pil(team, team_map, size=120)
        if logo:
            im = OffsetImage(np.array(logo), zoom=0.28)
            ab = AnnotationBbox(im, (1.2, logo_cy), frameon=False, zorder=6)
            ax.add_artist(ab)
        else:
            ax.text(1.2, logo_cy, _initials(name), ha='center', va='center', fontsize=11,
                    fontweight='bold', color='#333', zorder=7)

        # Dynamic name size: shorter names get bigger font, long names shrink to fit
        # Reserve space for position label on the right (~2 chars for pos like SS, SP)
        max_name_chars = len(name)
        if max_name_chars <= 12:
            name_size = 18
        elif max_name_chars <= 16:
            name_size = 16
        elif max_name_chars <= 20:
            name_size = 14
        else:
            name_size = 12

        # Name and team (left column)
        ax.text(2.0, 4.15, name, ha='left', va='center', fontsize=name_size,
                fontweight='bold', color='#FFFFFF', zorder=7)
        ax.text(2.0, 3.35, team, ha='left', va='center',
                fontsize=8, color='#aaa', zorder=7)

        # Position/role (right column, large)
        ax.text(9.5, 3.85, display_role, ha='right', va='center', fontsize=22,
                fontweight='bold', color='#FFFFFF', alpha=0.9, zorder=7)

        # Divider
        ax.plot([0.5, 9.5], [2.85, 2.85], color='#444', linewidth=0.8, zorder=3)

        # Top stats row
        n_top = len(stats_top)
        for j, (val, label) in enumerate(stats_top):
            cx = (j + 0.5) / n_top * 10
            ax.text(cx, 2.1, str(val), ha='center', va='center', fontsize=13,
                    fontweight='bold', color=LC_RED, zorder=7)
            ax.text(cx, 1.5, label, ha='center', va='center', fontsize=7, color='#888', zorder=7)

        # Bottom stats row
        n_bot = len(stats_bottom)
        for j, (val, label) in enumerate(stats_bottom):
            cx = (j + 0.5) / n_bot * 10
            ax.text(cx, 0.8, str(val), ha='center', va='center', fontsize=11, color='#C8C8C8', zorder=7)
            ax.text(cx, 0.2, label, ha='center', va='center', fontsize=7, color='#888', zorder=7)

    for idx, (ctype, name, team, role, top_stats, bot_stats) in enumerate(cards):
        r = idx // n_cols
        c = idx % n_cols
        left = (0.2 + c * (card_w_in + gap_x_in)) / fig_w
        bottom = 1.0 - (0.6 + (r + 1) * card_h_in + r * gap_y_in) / fig_h
        w_frac = card_w_in / fig_w
        h_frac = card_h_in / fig_h
        ring_color = LC_RELIEVER_COLOR if role == 'Reliever' else (LC_DH_COLOR if role == 'DH' else LC_RED)
        draw_card(fig, left, bottom, w_frac, h_frac, name, team, role, top_stats, bot_stats, team_map, ring_color)

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=180, facecolor='#1a1a1a', edgecolor='none', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


# ── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_player_lookup():
    """Build playerId (ncaa_season_id) → {position, school} from rosters + players + teams."""
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

    p = players[['id', 'position', 'team_id']].copy()
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
    }


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
    # Exclude Big Sky (id 123) — it's a catch-all bucket for unmapped NAIA teams
    div_conf_ids = set(confs[(confs['division'] == div_label) & (confs['name'] != 'Big Sky Conference')]['id'])
    sport_teams = teams[teams['sport'] == sport_label]
    div_teams = sport_teams[sport_teams['conference_id'].isin(div_conf_ids)]
    return set(div_teams['name'].dropna())


@st.cache_data
def load_pbp(sport, division, stat_type):
    """Load a PBP file: sport=baseball|softball, division=D1|D2|D3, stat_type=hitting|pitching|fielding."""
    filepath = PBP_DIR / sport / f'{stat_type}_pbp_{division}.csv'
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath, low_memory=False)
    # Filter to only teams in the selected division (cross-division opponents appear in PBP files)
    div_teams = load_division_teams(sport, division)
    if div_teams and 'teamName' in df.columns:
        df = df[df['teamName'].isin(div_teams)].copy()
    # Normalize IDs
    if 'playerId' in df.columns:
        df['playerId'] = pd.to_numeric(df['playerId'], errors='coerce').fillna(0).astype(int).astype(str)
    # Clean positions: "PH/LF" → "PH", "/LF/CF" → "LF"
    if 'playerPosition' in df.columns:
        df['playerPosition'] = df['playerPosition'].apply(_clean_position)
    # Enrich with formal position and school from players.csv
    if 'playerId' in df.columns:
        lookup = load_player_lookup()
        if lookup:
            df['formalPosition'] = df['playerId'].map(lookup['position'])
            df['school'] = df['playerId'].map(lookup['school'])
            # Fall back to teamName for players not in rosters
            if 'teamName' in df.columns:
                df['school'] = df['school'].fillna(df['teamName'])
    # Mark starters for pitching data: first pitcher per team per game
    if stat_type == 'pitching' and 'gameId' in df.columns and 'teamName' in df.columns:
        df['is_starter'] = df.duplicated(subset=['gameId', 'teamName'], keep='first') == False
    # Parse dates
    if 'date' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    return df


# ── Main ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='64 Analytics — PBP Stat Calculator', layout='wide',
                   initial_sidebar_state='expanded')

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, .stApp, .stApp *:not([class*="icon"]):not([class*="Icon"]):not([data-testid*="icon"]):not([data-testid*="Icon"]):not([data-testid*="arrow"]):not(.material-icons):not(.material-symbols):not(.material-symbols-rounded){ font-family: 'Inter', sans-serif !important; }
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
view = st.sidebar.radio('Mode', ['Hitter Stats', 'Pitcher Stats', 'Fielding Stats', 'Pace Chart', 'Lineup Card'], horizontal=True)

# Load data — Lineup Card and Who's Hot need both hitting + pitching
if view == 'Lineup Card':
    hitting_pbp = load_pbp(sport, division, 'hitting')
    pitching_pbp = load_pbp(sport, division, 'pitching')
    if hitting_pbp is None or pitching_pbp is None:
        st.error(f'PBP data not found for {sport} {division}')
        st.stop()
    # Use hitting for date range reference
    pbp = hitting_pbp
elif view == 'Pace Chart':
    # Will choose hitting or pitching based on sidebar selection
    pace_type = 'hitting'  # placeholder, will be set in sidebar
    pbp = load_pbp(sport, division, 'hitting')
    if pbp is None:
        st.error(f'PBP data not found for {sport} {division}')
        st.stop()
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

# Team / Position / Player filters — not shown for Lineup Card or Who's Hot
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
        min_ip = st.sidebar.number_input('Min IP', value=0.0, min_value=0.0, step=5.0, format='%.1f')

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
elif view == 'Lineup Card':
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
    league_r_pa = league_stats['R/PA']

    # Overall summary
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('League wOBA', f"{league_woba:.3f}")
    c2.metric('League OPS', f"{league_stats['OPS']:.3f}")
    c3.metric('Total PA', f"{league_stats['PA']:,}")
    c4.metric('Total H', f"{league_stats['H']:,}")
    c5.metric('Total HR', f"{league_stats['HR']:,}")

    # Per-player table
    st.markdown('---')
    player_stats = compute_grouped_hitting(pbp, player_col, league_woba, league_r_pa=league_r_pa, min_pa=min_threshold)

    if len(player_stats) == 0:
        st.info(f'No players meet the {min_threshold} PA minimum.')
    else:
        show_cols = ['Rank', player_col, 'School', 'Pos', 'PA', 'AB', 'H', '1B', '2B', '3B', 'HR', 'TB',
                     'R', 'RBI', 'BB', 'HBP', 'SF', 'SH', 'IBB', 'K', 'SB', 'CS', 'GDP',
                     'BA', 'OBP', 'SLG', 'OPS', 'ISO', 'BABIP',
                     'K%', 'BB%', 'K/BB', 'R/PA',
                     'wOBA', 'wRAA', 'wRC', 'wRC+']
        show_cols = [c for c in show_cols if c in player_stats.columns]
        st.dataframe(player_stats[show_cols], use_container_width=True, hide_index=True, height=1050)

        csv_buf = player_stats[show_cols].to_csv(index=False)
        st.download_button('Download CSV', data=csv_buf,
                          file_name=f'pbp_hitting_{sport}_{division}.csv', mime='text/csv')

    # Single player deep dive
    if selected_players and len(selected_players) == 1:
        st.markdown(f'### {selected_players[0]} — Game Log')
        player_data = pbp[pbp[player_col] == selected_players[0]]

        if 'date_parsed' in player_data.columns:
            game_log = compute_grouped_hitting(player_data, 'date', league_woba, league_r_pa=league_r_pa, min_pa=1)
            if len(game_log) > 0:
                game_log = game_log.rename(columns={'date': 'Date'})
                game_log = game_log.sort_values('Date')
                gl_cols = ['Date', 'PA', 'AB', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'K',
                          'BA', 'OBP', 'SLG', 'OPS', 'ISO', 'BABIP',
                          'K%', 'BB%', 'wOBA', 'wRAA']
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

    # Apply cumulative IP filter
    if min_ip > 0 and len(pitcher_stats) > 0 and 'IP' in pitcher_stats.columns:
        pitcher_stats = pitcher_stats[pitcher_stats['IP'].apply(
            lambda x: int(x) + (round((x - int(x)) * 10) / 3) >= min_ip
        )].reset_index(drop=True)

    if len(pitcher_stats) == 0:
        st.info(f'No pitchers meet the minimum filters.')
    else:
        k_col = 'K/7' if sport == 'softball' else 'K/9'
        show_cols = ['Rank', player_col, 'School', 'Pos', 'App', 'GS', 'IP', 'BF', 'OAB',
                     'H', 'R', 'ER', 'BB', 'HB', 'SO',
                     'HR', '2B-A', '3B-A', 'Bk', 'IBB', 'SHA', 'SFA',
                     'ERA', 'FIP', 'GmSc', 'BAA', 'BABIP',
                     'OBP Against', 'SLG Against', 'OPS Against',
                     'K%', 'BB%', k_col, 'K/BB', 'WHIP']
        show_cols = [c for c in show_cols if c in pitcher_stats.columns]
        st.dataframe(pitcher_stats[show_cols], use_container_width=True, hide_index=True, height=1050)

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
                gl_cols = ['Date', 'IP', 'BF', 'H', 'R', 'ER', 'BB', 'SO', 'HR',
                          'ERA', 'FIP', 'BABIP', 'OPS Against', 'K%', 'BB%', 'WHIP']
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
        show_cols = [player_col, 'School', 'Pos', 'PO', 'A', 'TC', 'E', 'FPCT',
                     'PB', 'SBA', 'CSB', 'CS%', 'IDP', 'TP']
        show_cols = [c for c in show_cols if c in fielding_stats.columns]
        st.dataframe(fielding_stats[show_cols], use_container_width=True, hide_index=True, height=1050)

        csv_buf = fielding_stats[show_cols].to_csv(index=False)
        st.download_button('Download CSV', data=csv_buf,
                          file_name=f'pbp_fielding_{sport}_{division}.csv', mime='text/csv')

elif view == 'Pace Chart':
    st.markdown(f'### Pace Chart — {sport.title()} {division}')

    # Pace chart controls
    pace_stat_type = st.sidebar.radio('Stat Type', ['Hitting', 'Pitching'], horizontal=True)
    pace_level = st.sidebar.radio('Level', ['Player', 'Team'], horizontal=True)
    pace_theme = st.sidebar.radio('Theme', ['Dark', 'Light'], horizontal=True)

    if pace_stat_type == 'Hitting':
        pace_pbp = load_pbp(sport, division, 'hitting')
        cum_stats = {'HR': 'hr', 'H': 'h', 'R': 'r', 'RBI': 'rbi', 'BB': 'bb',
                     'K': 'k', 'SB': 'sb', '2B': 'doubles', '3B': 'triples', 'TB': 'tb',
                     'HBP': 'hbp', 'AB': 'ab'}
        # Advanced stats computed as running values
        advanced_hitting = ['wRAA', 'wRC', 'OPS', 'wOBA', 'ISO', 'BABIP']
    else:
        pace_pbp = load_pbp(sport, division, 'pitching')
        cum_stats = {'SO': 'so', 'BB': 'bb', 'H': 'h', 'ER': 'er', 'R': 'r',
                     'HR': 'hrA', 'HB': 'hb'}
        advanced_pitching = ['FIP', 'WHIP', 'ERA', 'K/9', 'K/7', 'OPS Against', 'BAA', 'K%', 'BB%']

    if pace_pbp is None:
        st.error('Data not found.')
        st.stop()

    # Apply date filter
    if 'date_parsed' in pace_pbp.columns and date_start and date_end:
        pace_pbp = pace_pbp[(pace_pbp['date_parsed'].dt.date >= date_start) & (pace_pbp['date_parsed'].dt.date <= date_end)]

    # Apply conference filter
    try:
        if conf_map and selected_conferences:
            c_teams = {t for t, c in conf_map.items() if c in selected_conferences}
            pace_pbp = pace_pbp[pace_pbp['teamName'].isin(c_teams)]
    except NameError:
        pass

    # Apply team filter
    try:
        if selected_team and selected_team != 'All':
            pace_pbp = pace_pbp[pace_pbp['teamName'] == selected_team]
    except NameError:
        pass

    # Build stat options
    if pace_stat_type == 'Hitting':
        all_stat_options = list(cum_stats.keys()) + advanced_hitting
    else:
        all_stat_options = list(cum_stats.keys()) + advanced_pitching

    stat_choice = st.sidebar.selectbox('Stat to Track', all_stat_options)
    is_advanced = stat_choice not in cum_stats
    min_games_pace = st.sidebar.number_input('Min Games', value=5, min_value=1, step=1)

    if 'date_parsed' not in pace_pbp.columns:
        st.warning('Date column not available.')
        st.stop()

    # Group key: player or team (use player+team combo to avoid merging same-name players)
    if pace_level == 'Player':
        pace_pbp = pace_pbp.copy()
        pace_pbp['_player_team'] = pace_pbp['playerName'] + '|||' + pace_pbp['teamName']
        group_key = '_player_team'
    else:
        group_key = 'teamName'

    # Compute league stats for advanced hitting metrics
    if pace_stat_type == 'Hitting' and is_advanced:
        league_stats_pace = compute_hitting_stats(pace_pbp)
        league_woba_pace = league_stats_pace['wOBA']
        league_r_pa_pace = league_stats_pace['R/PA']

    # Build running stats per entity per game date
    entity_games = []
    for entity_name, edata in pace_pbp.groupby(group_key):
        edata_sorted = edata.sort_values('date_parsed')
        if pace_level == 'Player':
            player_name = entity_name.split('|||')[0]
            team = entity_name.split('|||')[1]
            display_name = player_name
        else:
            player_name = entity_name
            team = entity_name
            display_name = entity_name
        entity_name = display_name

        if not is_advanced:
            # Simple cumulative stat — group by gameId to handle doubleheaders
            stat_col = cum_stats[stat_choice]
            if stat_col not in edata_sorted.columns:
                continue
            per_game = edata_sorted.groupby(['gameId', 'date_parsed'])[stat_col].sum().reset_index().sort_values('date_parsed')
            per_game['game_num'] = range(1, len(per_game) + 1)
            per_game['cum_stat'] = per_game[stat_col].cumsum()
            per_game['entity'] = entity_name
            per_game['team'] = team
            entity_games.append(per_game[['entity', 'team', 'date_parsed', 'game_num', 'cum_stat']])
        else:
            # Advanced: compute running stat through each game — use gameId for doubleheaders
            game_order = edata_sorted.drop_duplicates('gameId')[['gameId', 'date_parsed']].sort_values('date_parsed')
            game_ids_ordered = game_order['gameId'].tolist()
            rows = []
            for i, gid in enumerate(game_ids_ordered):
                d = game_order[game_order['gameId'] == gid]['date_parsed'].iloc[0]
                # Include all games up to and including this one
                included_gids = set(game_ids_ordered[:i+1])
                window = edata_sorted[edata_sorted['gameId'].isin(included_gids)]
                if pace_stat_type == 'Hitting':
                    stats = compute_hitting_stats(window)
                    if stat_choice == 'wRAA':
                        val = compute_wraa(stats['wOBA'], league_woba_pace, stats['PA'])
                    elif stat_choice == 'wRC':
                        val = compute_wrc(stats['wOBA'], league_woba_pace, league_r_pa_pace, stats['PA'])
                    else:
                        val = stats.get(stat_choice, 0)
                else:
                    stats = compute_pitching_stats(window)
                    val = stats.get(stat_choice, 0)
                rows.append({'entity': entity_name, 'team': team, 'date_parsed': d,
                            'game_num': i + 1, 'cum_stat': val})
            if rows:
                entity_games.append(pd.DataFrame(rows))

    if not entity_games:
        st.warning('No data available.')
        st.stop()

    all_games = pd.concat(entity_games, ignore_index=True)

    # Extend all entities to the last date in the range (flat line if no data)
    max_date = all_games['date_parsed'].max()
    if date_end:
        max_date = max(max_date, pd.Timestamp(date_end))
    extensions = []
    for entity_name, edata in all_games.groupby('entity'):
        last_row = edata.sort_values('date_parsed').iloc[-1]
        if last_row['date_parsed'] < max_date:
            extensions.append({
                'entity': entity_name,
                'team': last_row['team'],
                'date_parsed': max_date,
                'game_num': last_row['game_num'],
                'cum_stat': last_row['cum_stat'],
            })
    if extensions:
        all_games = pd.concat([all_games, pd.DataFrame(extensions)], ignore_index=True)

    # Filter by min games
    game_counts = all_games.groupby('entity')['game_num'].max()
    qualified = game_counts[game_counts >= min_games_pace].index
    all_games = all_games[all_games['entity'].isin(qualified)]

    if len(all_games) == 0:
        st.warning('No entities meet the minimum games threshold.')
        st.stop()

    # Sort by final value for highlight selection
    final_stats = all_games.groupby(['entity', 'team']).agg(
        total=('cum_stat', 'last'), games=('game_num', 'max')).reset_index()
    # For pitching rate stats, lower is better (sort ascending so best = first).
    # For cumulative counting stats (ER, H, BB, SO, HR, etc.), higher volume = "top".
    lower_is_better = stat_choice in ['FIP', 'WHIP', 'ERA', 'BAA', 'OPS Against', 'BB%']
    final_stats = final_stats.sort_values('total', ascending=lower_is_better)

    if pace_level == 'Player':
        options = [f"{row['entity']} ({row['team']})" for _, row in final_stats.iterrows()]
    else:
        options = [row['entity'] for _, row in final_stats.iterrows()]
    name_map = dict(zip(options, final_stats['entity']))

    top_n = st.sidebar.number_input('Show Top N', value=20, min_value=5, max_value=2000, step=5)
    top_entities = set(final_stats.head(top_n)['entity'])
    filtered = all_games[all_games['entity'].isin(top_entities)]

    highlighted = st.sidebar.multiselect(f'Highlight {pace_level}s', options,
                                          help=f'Select {pace_level.lower()}s to highlight in red')
    highlight_names = {name_map[p] for p in highlighted}

    # Theme colors
    if pace_theme == 'Dark':
        bg = '#1a1a1a'; line_color = '#666666'; label_color = '#888888'
        text_color = '#C8C8C8'; grid_color = '#444444'; spine_color = '#333333'
    else:
        bg = '#FFFFFF'; line_color = '#BBBBBB'; label_color = '#999999'
        text_color = '#1a1a1a'; grid_color = '#E0E0E0'; spine_color = '#CCCCCC'

    # Render chart
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=bg)
    ax.set_facecolor(bg)

    for ename, pdata in filtered.groupby('entity'):
        if ename in highlight_names:
            continue
        ax.plot(pdata['date_parsed'], pdata['cum_stat'],
                color=line_color, alpha=0.35, linewidth=1, zorder=1)
        last = pdata.iloc[-1]
        label = ename if pace_level == 'Team' else ename.split()[-1] if len(ename.split()) > 1 else ename
        ax.annotate(label, xy=(last['date_parsed'], last['cum_stat']),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=6, color=label_color, va='center', alpha=0.6, zorder=1)

    for ename in highlight_names:
        pdata = filtered[filtered['entity'] == ename]
        if len(pdata) == 0:
            pdata = all_games[all_games['entity'] == ename]
        if len(pdata) == 0:
            continue
        # Get team color
        team_for_color = pdata['team'].iloc[0] if pace_level == 'Player' else ename
        h_color = get_team_color(team_for_color)
        ax.plot(pdata['date_parsed'], pdata['cum_stat'],
                color=h_color, alpha=1.0, linewidth=3, zorder=3)
        last = pdata.iloc[-1]
        val_fmt = f"{last['cum_stat']:.2f}" if is_advanced else f"{int(last['cum_stat'])}"
        ax.annotate(f"{ename}\n{val_fmt} {stat_choice} in {int(last['game_num'])} G",
                    xy=(last['date_parsed'], last['cum_stat']),
                    xytext=(10, 0), textcoords='offset points',
                    fontsize=8, fontweight='bold', color=h_color,
                    va='center', zorder=4)

    y_label = f'Running {stat_choice}' if is_advanced else f'Cumulative {stat_choice}'
    ax.set_xlabel('Date', fontsize=11, color=text_color, labelpad=10)
    fig.autofmt_xdate(rotation=45)
    ax.set_ylabel(y_label, fontsize=11, color=text_color, labelpad=10)
    ax.tick_params(colors=label_color)
    ax.grid(True, alpha=0.15, color=grid_color)
    for spine in ax.spines.values():
        spine.set_color(spine_color)

    # Force whole-number y-axis ticks for counting stats
    if not is_advanced:
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Brand logo bottom-right
    wide_logo_path = _APP_DIR / 'assets' / 'brand_logo_wide.png'
    if wide_logo_path.exists():
        logo_img = Image.open(wide_logo_path).convert('RGBA')
        logo_img.thumbnail((200, 50), Image.LANCZOS)
        logo_arr = np.array(logo_img)
        logo_im = OffsetImage(logo_arr, zoom=0.4, alpha=0.6)
        logo_ab = AnnotationBbox(logo_im, (0.98, 0.02), xycoords='axes fraction',
                                  frameon=False, zorder=10, box_alignment=(1, 0))
        ax.add_artist(logo_ab)

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=180, facecolor=bg, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    st.image(buf, use_container_width=True)

    buf.seek(0)
    st.download_button('Download Pace Chart PNG', data=buf,
                      file_name=f'pace_chart_{stat_choice}_{sport}_{division}.png', mime='image/png')

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
    best_hitters = get_best_hitters(hitting_pbp, league_woba, min_pa=min_pa_lc, sport=sport)
    starters, relievers = get_best_pitchers(pitching_pbp, min_bf_sp=min_bf_sp, min_bf_rp=min_bf_rp)

    # Load team logo map (prefer softball logos when viewing softball)
    team_map = load_team_logo_map(prefer_sport=sport)

    # Render SVG
    title = f"Players of the Period"
    subtitle = f"{sport.title()} {division} · {period_label}"
    svg = render_lineup_svg(best_hitters, starters, relievers, title, subtitle, team_map, date_label=period_label, sport=sport)
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
                st.markdown(render_pitcher_card_html(sp, 'Starter', sport=sport), unsafe_allow_html=True)

    if relievers:
        cols3 = st.columns(min(len(relievers), 3))
        for i, rp in enumerate(relievers[:3]):
            with cols3[i]:
                st.markdown(render_pitcher_card_html(rp, 'Reliever', sport=sport), unsafe_allow_html=True)

    # Download PNGs
    st.markdown('---')
    st.markdown('### Download')
    with st.spinner('Rendering PNGs...'):
        # Try cairosvg for pixel-perfect SVG→PNG, fall back to matplotlib
        diamond_buf = None
        try:
            import cairosvg
            # Extract the inner SVG content after the opening tag
            inner_start = svg.find('>') + 1
            inner_end = svg.rfind('</svg>')
            inner_content = svg[inner_start:inner_end]

            # Build background
            bg_element = '<rect x="0" y="-20" width="460" height="440" fill="#1a1a1a"/>'
            bg_path = _APP_DIR / 'assets' / 'bg_pattern.jpg'
            if bg_path.exists():
                bg_data = bg_path.read_bytes()
                bg_b64 = base64.b64encode(bg_data).decode()
                bg_element = f'<image href="data:image/jpeg;base64,{bg_b64}" x="0" y="-20" width="460" height="440" preserveAspectRatio="xMidYMid slice"/>'

            svg_full = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="460" height="440" viewBox="0 -20 460 420"
     xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
{bg_element}
{inner_content}
</svg>'''
            diamond_buf = BytesIO()
            cairosvg.svg2png(bytestring=svg_full.encode('utf-8'), write_to=diamond_buf,
                             output_width=1840, output_height=1760)
            diamond_buf.seek(0)
        except Exception as e:
            st.warning(f'cairosvg failed ({e}), using matplotlib fallback')
            diamond_buf = render_diamond_png(best_hitters, starters, relievers, title, subtitle, team_map)

        cards_buf = render_cards_png(best_hitters, starters, relievers, title, subtitle, team_map, sport=sport)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button('Download Diamond PNG', data=diamond_buf,
                          file_name=f'lineup_diamond_{sport}_{division}.png', mime='image/png')
    with dl2:
        st.download_button('Download Cards PNG', data=cards_buf,
                          file_name=f'lineup_cards_{sport}_{division}.png', mime='image/png')

