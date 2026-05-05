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
    """Compute hitting stats grouped by a column. Vectorized for speed —
    matches the per-group compute_hitting_stats output exactly (same rounding,
    same formulas, same column names)."""
    if len(df) == 0:
        return pd.DataFrame()

    # Step 1: Sum all counting stats in one groupby pass
    sum_cols = ['ab', 'h', 'bb', 'hbp', 'sf', 'sh', 'tb', 'hr', 'doubles', 'triples',
                'k', 'sb', 'cs', 'oppDp', 'r', 'rbi']
    if 'ibb' in df.columns:
        sum_cols.append('ibb')
    g = df.groupby(group_col, sort=False)
    agg = g[sum_cols].sum().reset_index()
    # When grouping by playerId, add playerName for display
    if group_col == 'playerId' and 'playerName' in df.columns:
        name_map = df.dropna(subset=['playerName']).groupby(group_col, sort=False)['playerName'].first()
        agg['playerName'] = agg[group_col].map(name_map)
    if 'ibb' not in agg.columns:
        agg['ibb'] = 0

    # Step 2: Vectorized derived stats (raw, no rounding yet)
    ab = agg['ab']; h = agg['h']; bb = agg['bb']; hbp = agg['hbp']
    sf = agg['sf']; sh = agg['sh']; tb = agg['tb']; hr = agg['hr']
    doubles = agg['doubles']; triples = agg['triples']; k = agg['k']
    singles = h - doubles - triples - hr
    pa = ab + bb + hbp + sf + sh

    obp_d = ab + bb + hbp + sf
    obp = np.where(obp_d > 0, (h + bb + hbp) / obp_d, 0.0)
    slg = np.where(ab > 0, tb / ab, 0.0)
    ba = np.where(ab > 0, h / ab, 0.0)
    ops = obp + slg
    iso = slg - ba
    babip_d = ab - k - hr + sf
    babip = np.where(babip_d > 0, (h - hr) / babip_d, 0.0)

    woba_d = ab + bb + sf + hbp
    woba_num = (WOBA_BB * bb + WOBA_HBP * hbp + WOBA_1B * singles +
                WOBA_2B * doubles + WOBA_3B * triples + WOBA_HR * hr)
    woba = np.where(woba_d > 0, woba_num / woba_d, 0.0)

    k_pct = np.where(pa > 0, k / pa * 100, 0.0)
    bb_pct = np.where(pa > 0, bb / pa * 100, 0.0)
    k_bb = np.where(bb > 0, k / bb, k.astype(float))
    r_pa = np.where(pa > 0, agg['r'] / pa, 0.0)

    # Step 3: Build result dataframe with rounding to match compute_hitting_stats
    result = pd.DataFrame({
        group_col: agg[group_col],
        'PA': pa.astype(int),
        'AB': ab.astype(int), 'H': h.astype(int), '1B': singles.astype(int),
        '2B': doubles.astype(int), '3B': triples.astype(int), 'HR': hr.astype(int),
        'TB': tb.astype(int),
        'R': agg['r'].astype(int), 'RBI': agg['rbi'].astype(int),
        'BB': bb.astype(int), 'HBP': hbp.astype(int),
        'SF': sf.astype(int), 'SH': sh.astype(int),
        'IBB': agg['ibb'].astype(int), 'K': k.astype(int),
        'SB': agg['sb'].astype(int), 'CS': agg['cs'].astype(int),
        'GDP': agg['oppDp'].astype(int),
        'BA': np.round(ba, 3), 'OBP': np.round(obp, 3),
        'SLG': np.round(slg, 3), 'OPS': np.round(ops, 3),
        'ISO': np.round(iso, 3), 'BABIP': np.round(babip, 3),
        'wOBA': np.round(woba, 3),
        'K%': np.round(k_pct, 1), 'BB%': np.round(bb_pct, 1),
        'K/BB': np.round(k_bb, 2), 'R/PA': np.round(r_pa, 3),
    })

    # Carry playerName into result when grouping by playerId
    if 'playerName' in agg.columns:
        result['playerName'] = agg['playerName'].values

    # Filter to min_pa
    result = result[result['PA'] >= min_pa].copy()
    if len(result) == 0:
        return pd.DataFrame()

    # Step 4: wRAA / wRC / wRC+ — use ROUNDED wOBA to match original behavior
    result['wRAA'] = np.round(((result['wOBA'] - league_woba) / WOBA_SCALE) * result['PA'], 1)
    wrc_raw = (((result['wOBA'] - league_woba) / WOBA_SCALE) + league_r_pa) * result['PA']
    result['wRC'] = np.round(wrc_raw, 1)
    wrc_per_pa = np.where(result['PA'] > 0, result['wRC'] / result['PA'], 0.0)
    if league_r_pa > 0:
        result['wRC+'] = np.round((wrc_per_pa / league_r_pa) * 100, 0).astype(int)
    else:
        result['wRC+'] = 100

    # Step 5: Position and school enrichment (per-group lookups)
    if 'formalPosition' in df.columns:
        formal = df.dropna(subset=['formalPosition']).groupby(group_col, sort=False)['formalPosition'].first()
        pos_map = formal.to_dict()
    else:
        pos_map = {}
    if 'playerPosition' in df.columns:
        # Most-common game position as fallback
        pp = df.dropna(subset=['playerPosition'])
        fallback = pp.groupby(group_col, sort=False)['playerPosition'].agg(
            lambda s: s.value_counts().index[0] if len(s) > 0 else '')
        fallback_map = fallback.to_dict()
    else:
        fallback_map = {}
    result['Pos'] = result[group_col].map(lambda n: pos_map.get(n) or fallback_map.get(n, ''))

    if 'school' in df.columns:
        school = df.dropna(subset=['school']).groupby(group_col, sort=False)['school'].first()
        school_map = school.to_dict()
        result['School'] = result[group_col].map(school_map).fillna('')
    else:
        result['School'] = ''

    # Step 6: Combined rank — percentile rank of wRAA + OPS + TB
    n = len(result)
    if n > 0:
        wraa_pctl = result['wRAA'].rank(pct=True, method='min')
        ops_pctl = result['OPS'].rank(pct=True, method='min')
        tb_pctl = result['TB'].rank(pct=True, method='min')
        result['Rank'] = (wraa_pctl + ops_pctl + tb_pctl).round(6)

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
    """Compute pitching stats grouped by a column. Vectorized."""
    if len(df) == 0:
        return pd.DataFrame()

    df = df.copy()
    # Convert IP notation to total outs (vectorized)
    df['_outs'] = baseball_ip_to_outs(df['ip'])

    # Compute game score per row (vectorized)
    whole = df['_outs'] // 3
    innings_after_4th = np.maximum(0, whole - 4)
    so_row = df['so'].astype(int)
    h_row = df['h'].astype(int)
    er_row = df['er'].astype(int)
    r_row = df['r'].astype(int)
    bb_row = df['bb'].astype(int)
    df['_gmsc'] = 50 + df['_outs'] + 2 * innings_after_4th + so_row - 2 * h_row - 4 * er_row - 2 * (r_row - er_row) - bb_row

    # Aggregate per pitcher
    sum_cols = ['_outs', 'h', 'r', 'bb', 'hb', 'so', 'hrA', 'bf', 'wp', 'bk',
                'doublesA', 'triplesA', 'sha', 'sfa', 'er']
    if 'ibb' in df.columns:
        sum_cols.append('ibb')
    if 'is_starter' in df.columns:
        sum_cols.append('is_starter')

    g = df.groupby(group_col, sort=False)
    agg = g[sum_cols].sum().reset_index()
    # When grouping by playerId, add playerName for display
    if group_col == 'playerId' and 'playerName' in df.columns:
        name_map = df.dropna(subset=['playerName']).groupby(group_col, sort=False)['playerName'].first()
        agg['playerName'] = agg[group_col].map(name_map)
    if 'ibb' not in agg.columns:
        agg['ibb'] = 0
    if 'is_starter' not in agg.columns:
        agg['is_starter'] = 0
    agg['_gmsc_mean'] = g['_gmsc'].mean().values
    if 'gameId' in df.columns:
        agg['_app'] = g['gameId'].nunique().values
    else:
        agg['_app'] = g.size().values

    # Vectorized derived stats
    outs = agg['_outs']
    ip_actual = outs / 3.0
    h = agg['h']; r = agg['r']; bb = agg['bb']; hb = agg['hb']
    so = agg['so']; hr = agg['hrA']; bf = agg['bf']
    er = agg['er']; sha = agg['sha']; sfa = agg['sfa']
    doubles = agg['doublesA']; triples = agg['triplesA']
    singles = h - doubles - triples - hr
    p_oab = bf - bb - hb - sfa - sha

    fip = np.where(ip_actual > 0, ((13 * hr) + (3 * (bb + hb)) - (2 * so)) / ip_actual + FIP_CONSTANT, 0.0)
    era = np.where(ip_actual > 0, (er / ip_actual) * 9, 0.0)
    obp_d = p_oab + bb + hb + sfa
    obp_against = np.where(obp_d > 0, (h + bb + hb) / obp_d, 0.0)
    tb_against = singles + 2 * doubles + 3 * triples + 4 * hr
    slg_against = np.where(p_oab > 0, tb_against / p_oab, 0.0)
    ops_against = obp_against + slg_against
    ba_against = np.where(p_oab > 0, h / p_oab, 0.0)
    k_pct = np.where(bf > 0, so / bf * 100, 0.0)
    bb_pct = np.where(bf > 0, bb / bf * 100, 0.0)
    k_bb_pct = k_pct - bb_pct
    k9 = np.where(ip_actual > 0, (so / ip_actual) * 9, 0.0)
    k7 = np.where(ip_actual > 0, (so / ip_actual) * 7, 0.0)
    bb9 = np.where(ip_actual > 0, (bb / ip_actual) * 9, 0.0)
    k_bb = np.where(bb > 0, so / bb, so.astype(float))
    whip = np.where(ip_actual > 0, (bb + h) / ip_actual, 0.0)
    babip_d = p_oab - so - hr + sfa
    babip = np.where(babip_d > 0, (h - hr) / babip_d, 0.0)

    # IP display: convert outs back to baseball notation
    ip_display = (outs // 3).astype(int).astype(str) + '.' + (outs % 3).astype(int).astype(str)
    ip_display = ip_display.astype(float)

    result = pd.DataFrame({
        group_col: agg[group_col],
        'IP': ip_display,
        'App': agg['_app'].astype(int),
        'GS': agg['is_starter'].astype(int),
        'BF': bf.astype(int), 'OAB': p_oab.astype(int),
        'H': h.astype(int), 'R': r.astype(int), 'ER': er.astype(int),
        'BB': bb.astype(int), 'HB': hb.astype(int), 'SO': so.astype(int),
        'HR': hr.astype(int), '2B-A': doubles.astype(int), '3B-A': triples.astype(int),
        'WP': agg['wp'].astype(int), 'Bk': agg['bk'].astype(int),
        'IBB': agg['ibb'].astype(int),
        'SHA': sha.astype(int), 'SFA': sfa.astype(int),
        'ERA': np.round(era, 2), 'FIP': np.round(fip, 2),
        'BAA': np.round(ba_against, 3), 'BABIP': np.round(babip, 3),
        'OBP Against': np.round(obp_against, 3),
        'SLG Against': np.round(slg_against, 3),
        'OPS Against': np.round(ops_against, 3),
        'K%': np.round(k_pct, 1), 'BB%': np.round(bb_pct, 1),
        'K-BB%': np.round(k_bb_pct, 1),
        'K/9': np.round(k9, 2), 'K/7': np.round(k7, 2),
        'BB/9': np.round(bb9, 2),
        'K/BB': np.round(k_bb, 2), 'WHIP': np.round(whip, 2),
        'GmSc': np.round(agg['_gmsc_mean'], 1),
    })

    # Carry playerName into result when grouping by playerId
    if 'playerName' in agg.columns:
        result['playerName'] = agg['playerName'].values

    # Filter to min_bf
    result = result[result['BF'] >= min_bf].copy()
    if len(result) == 0:
        return pd.DataFrame()

    # Position and school enrichment
    if 'formalPosition' in df.columns:
        formal = df.dropna(subset=['formalPosition']).groupby(group_col, sort=False)['formalPosition'].first()
        pos_map = formal.to_dict()
    else:
        pos_map = {}
    if 'playerPosition' in df.columns:
        pp = df.dropna(subset=['playerPosition'])
        fallback = pp.groupby(group_col, sort=False)['playerPosition'].agg(
            lambda s: s.value_counts().index[0] if len(s) > 0 else '')
        fallback_map = fallback.to_dict()
    else:
        fallback_map = {}
    result['Pos'] = result[group_col].map(lambda n: pos_map.get(n) or fallback_map.get(n, ''))

    if 'school' in df.columns:
        school = df.dropna(subset=['school']).groupby(group_col, sort=False)['school'].first()
        school_map = school.to_dict()
        result['School'] = result[group_col].map(school_map).fillna('')
    else:
        result['School'] = ''

    # Pitcher Rank: FIP rank + A-OPS rank + 2 * Game Score rank
    n = len(result)
    if n > 0:
        fip_score = (n - result['FIP'].rank(method='min') + 1) / n
        ops_a_score = (n - result['OPS Against'].rank(method='min') + 1) / n
        gmsc_score = result['GmSc'].rank(method='min', ascending=True) / n
        result['Rank'] = (fip_score + ops_a_score + 2 * gmsc_score).round(6)

    cols = ['Rank'] + [group_col] + [c for c in result.columns if c not in ['Rank', group_col]]
    return result[cols].sort_values('Rank', ascending=False).reset_index(drop=True)


def compute_grouped_fielding(df, group_col):
    """Compute fielding stats grouped by a column. Vectorized."""
    if len(df) == 0:
        return pd.DataFrame()

    sum_cols = ['po', 'a', 'tc', 'e', 'pb', 'sba', 'csb', 'idp', 'tp']
    sum_cols = [c for c in sum_cols if c in df.columns]
    g = df.groupby(group_col, sort=False)
    agg = g[sum_cols].sum().reset_index()
    # When grouping by playerId, add playerName for display
    if group_col == 'playerId' and 'playerName' in df.columns:
        name_map = df.dropna(subset=['playerName']).groupby(group_col, sort=False)['playerName'].first()
        agg['playerName'] = agg[group_col].map(name_map)

    po = agg['po']; a = agg['a']; tc = agg['tc']; e = agg['e']
    sba = agg['sba']; csb = agg['csb']
    fpct = np.where(tc > 0, (po + a) / tc, 0.0)
    cs_pct = np.where((sba + csb) > 0, csb / (sba + csb), 0.0)

    result = pd.DataFrame({
        group_col: agg[group_col],
        'PO': po.astype(int), 'A': a.astype(int), 'TC': tc.astype(int), 'E': e.astype(int),
        'FPCT': np.round(fpct, 3),
        'PB': agg['pb'].astype(int),
        'SBA': sba.astype(int), 'CSB': csb.astype(int),
        'CS%': np.round(cs_pct, 3),
        'IDP': agg['idp'].astype(int), 'TP': agg['tp'].astype(int),
    })

    # Carry playerName into result when grouping by playerId
    if 'playerName' in agg.columns:
        result['playerName'] = agg['playerName'].values

    # Filter to TC > 0
    result = result[result['TC'] > 0].copy()
    if len(result) == 0:
        return pd.DataFrame()

    # Position and school enrichment
    if 'formalPosition' in df.columns:
        formal = df.dropna(subset=['formalPosition']).groupby(group_col, sort=False)['formalPosition'].first()
        pos_map = formal.to_dict()
    else:
        pos_map = {}
    if 'playerPosition' in df.columns:
        pp = df.dropna(subset=['playerPosition'])
        fallback = pp.groupby(group_col, sort=False)['playerPosition'].agg(
            lambda s: s.value_counts().index[0] if len(s) > 0 else '')
        fallback_map = fallback.to_dict()
    else:
        fallback_map = {}
    result['Pos'] = result[group_col].map(lambda n: pos_map.get(n) or fallback_map.get(n, ''))

    if 'school' in df.columns:
        school = df.dropna(subset=['school']).groupby(group_col, sort=False)['school'].first()
        school_map = school.to_dict()
        result['School'] = result[group_col].map(school_map).fillna('')
    else:
        result['School'] = ''

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
    Otherwise fall back to baseball ID (canonical).

    Also registers PBP teamName variants (e.g. "Lamar University" for the
    canonical "Lamar") so logo lookups against the PBP data find the right
    logo even when the PBP file spells the team differently than teams.csv.
    """
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

    # Register PBP teamName variants (e.g. "Lamar University" -> "Lamar" logo)
    # so players like Chris Olivier get their logo even though the PBP files
    # spell Lamar with the "University" suffix.
    norm_to_canonical = {_norm_team(k): k for k in name_to_id.keys()}
    for sport_dir in ('baseball', 'softball'):
        pbp_path = PBP_DIR / sport_dir / 'hitting_pbp_D1.csv'
        if not pbp_path.exists():
            continue
        try:
            pbp_names = pd.read_csv(pbp_path, low_memory=False, usecols=['teamName'])['teamName'].dropna().unique()
            for pn in pbp_names:
                if pn in name_to_id:
                    continue  # already canonical
                norm = _norm_team(pn)
                canonical = norm_to_canonical.get(norm)
                if canonical:
                    name_to_id[pn] = name_to_id[canonical]
        except Exception:
            pass
    return name_to_id


@st.cache_data
def load_team_conference_map():
    """Build team_name -> conference_name mapping from teams.csv + conferences.csv.
    Also registers PBP teamName variants so conference lookups against PBP data
    resolve even for suffix-drift cases like "Lamar University".
    """
    teams_path = DATA_DIR / 'teams.csv'
    confs_path = DATA_DIR / 'conferences.csv'
    if not teams_path.exists() or not confs_path.exists():
        return {}
    teams = pd.read_csv(teams_path, low_memory=False)
    confs = pd.read_csv(confs_path, low_memory=False)
    merged = teams.merge(confs[['id', 'name']], left_on='conference_id', right_on='id', suffixes=('', '_conf'))
    name_to_conf = dict(zip(merged['name'], merged['name_conf']))
    norm_to_canonical = {_norm_team(k): k for k in name_to_conf.keys()}
    for sport_dir in ('baseball', 'softball'):
        pbp_path = PBP_DIR / sport_dir / 'hitting_pbp_D1.csv'
        if not pbp_path.exists():
            continue
        try:
            pbp_names = pd.read_csv(pbp_path, low_memory=False, usecols=['teamName'])['teamName'].dropna().unique()
            for pn in pbp_names:
                if pn in name_to_conf:
                    continue
                canonical = norm_to_canonical.get(_norm_team(pn))
                if canonical:
                    name_to_conf[pn] = name_to_conf[canonical]
        except Exception:
            pass
    return name_to_conf


@st.cache_data
def load_team_classification_map():
    """Build team_name -> conference classification (upper/middle/lower) map.
    Only D-I conferences carry a classification; teams without one are omitted,
    so the tier filter naturally excludes D-II / D-III teams.
    """
    teams_path = DATA_DIR / 'teams.csv'
    confs_path = DATA_DIR / 'conferences.csv'
    if not teams_path.exists() or not confs_path.exists():
        return {}
    teams = pd.read_csv(teams_path, low_memory=False)
    confs = pd.read_csv(confs_path, low_memory=False)
    merged = teams.merge(confs[['id', 'classification']], left_on='conference_id', right_on='id')
    merged = merged[merged['classification'].notna()]
    name_to_class = dict(zip(merged['name'], merged['classification']))
    norm_to_canonical = {_norm_team(k): k for k in name_to_class.keys()}
    for sport_dir in ('baseball', 'softball'):
        pbp_path = PBP_DIR / sport_dir / 'hitting_pbp_D1.csv'
        if not pbp_path.exists():
            continue
        try:
            pbp_names = pd.read_csv(pbp_path, low_memory=False, usecols=['teamName'])['teamName'].dropna().unique()
            for pn in pbp_names:
                if pn in name_to_class:
                    continue
                canonical = norm_to_canonical.get(_norm_team(pn))
                if canonical:
                    name_to_class[pn] = name_to_class[canonical]
        except Exception:
            pass
    return name_to_class


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


def get_best_hitters(hitting_df, league_woba, min_pa=10, sport='baseball', rank_min_pa=None):
    """
    min_pa: PA threshold for eligibility (must have this many PA to appear in the lineup card)
    rank_min_pa: PA threshold for the rank calculation population. If set, percentile ranks
                  are computed within the PA >= rank_min_pa population so that the lineup
                  card agrees with the Hitter Stats table (which uses min_threshold for both
                  filtering and ranking). Defaults to min_pa.
    """
    best = {}
    dh_label = 'DP' if sport == 'softball' else 'DH'
    if rank_min_pa is None:
        rank_min_pa = min_pa

    def _pos_for_group(group):
        # Prefer formalPosition from players.csv (matches Hitter Stats table)
        if 'formalPosition' in group.columns:
            formal = group['formalPosition'].dropna()
            if len(formal) > 0:
                p = formal.iloc[0]
                if sport == 'softball' and p == 'DP':
                    p = 'DH'
                return p
        # Fallback: most-common game position
        if 'playerPosition' in group.columns:
            pos = group['playerPosition'].dropna()
            if len(pos) > 0:
                p = pos.value_counts().index[0]
                if sport == 'softball' and p == 'DP':
                    p = 'DH'
                return p
        return ''

    # Step 1: Build full player pool — anyone meeting EITHER threshold needs to be considered
    pa_floor = min(min_pa, rank_min_pa)
    player_data = []
    hit_group_col = 'playerId' if 'playerId' in hitting_df.columns else 'playerName'
    for key, group in hitting_df.groupby(hit_group_col):
        stats = compute_hitting_stats(group)
        if stats['PA'] >= pa_floor:
            stats['playerName'] = group['playerName'].iloc[0]
            stats['playerId'] = key if hit_group_col == 'playerId' else ''
            stats['teamName'] = group['teamName'].mode().iloc[0] if len(group['teamName'].mode()) > 0 else ''
            stats['wRAA'] = compute_wraa(stats['wOBA'], league_woba, stats['PA'])
            stats['_pos'] = _pos_for_group(group)
            player_data.append(stats)

    if not player_data:
        return best

    all_df = pd.DataFrame(player_data)

    # Step 2: Compute the rank within the rank_min_pa population only (matches Hitter Stats)
    rank_pop = all_df[all_df['PA'] >= rank_min_pa].copy()
    if len(rank_pop) > 0:
        rank_pop['wraa_pctl'] = rank_pop['wRAA'].rank(pct=True, method='min')
        rank_pop['ops_pctl'] = rank_pop['OPS'].rank(pct=True, method='min')
        rank_pop['tb_pctl'] = rank_pop['TB'].rank(pct=True, method='min')
        rank_pop['Rank'] = rank_pop['wraa_pctl'] + rank_pop['ops_pctl'] + rank_pop['tb_pctl']
    # Map player → rank
    rank_map = dict(zip(rank_pop.get('playerName', []), rank_pop.get('Rank', [])))
    all_df['Rank'] = all_df['playerName'].map(rank_map)

    # Step 3: Filter to eligible (PA >= min_pa) AND has a rank (PA >= rank_min_pa)
    # Players with PA in [min_pa, rank_min_pa) are eligible by min_pa but have no rank,
    # so they get excluded — they'd have no comparable rank to Hitter Stats anyway.
    eligible = all_df[(all_df['PA'] >= min_pa) & all_df['Rank'].notna()].copy()
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
        raw_group_col = 'playerId' if 'playerId' in pitching_raw_df.columns else 'playerName'
        for key, group in pitching_raw_df.groupby(raw_group_col):
            scores = group.apply(compute_game_score, axis=1)
            gmsc_map[key] = scores.mean()
        df['GmSc'] = df[raw_group_col].map(gmsc_map).fillna(50.0) if raw_group_col in df.columns else 50.0
    elif 'GmSc' not in df.columns:
        df['GmSc'] = 50.0
    df['fip_score'] = (n - df['FIP'].rank(method='min') + 1) / n
    df['ops_a_score'] = (n - df['OPS Against'].rank(method='min') + 1) / n
    df['gmsc_score'] = df['GmSc'].rank(method='min', ascending=True) / n
    df['combined_score'] = df['fip_score'] + df['ops_a_score'] + 2 * df['gmsc_score']
    return df.sort_values('combined_score', ascending=False)


def get_best_pitchers(pitching_df, min_bf_sp=50, min_bf_rp=15, n_starters=3, n_relievers=3):
    rows = []
    pit_group_col = 'playerId' if 'playerId' in pitching_df.columns else 'playerName'
    for key, group in pitching_df.groupby(pit_group_col):
        stats = compute_pitching_for_lineup(group)
        stats['playerName'] = group['playerName'].iloc[0]
        stats['playerId'] = key if pit_group_col == 'playerId' else ''
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


def _norm_team(name):
    """Normalize a team name for cross-file matching: strip Saint/State/
    Univ. suffix variants so "Lamar" and "Lamar University" collapse to
    the same key. Kept local to this page to avoid cross-page imports.
    """
    import re as _re
    if not isinstance(name, str):
        return ''
    n = name.lower().strip()
    n = _re.sub(r'\bsaint\b', 'st', n)
    n = _re.sub(r'\bstate\b', 'st', n)
    n = _re.sub(r'\buniversity\b|\buniv\.?\b', '', n)
    return _re.sub(r'[^a-z0-9]+', '', n)


@st.cache_data
def load_portal_2026_ncaa_pids():
    """NCAA player season IDs (as strings) for players in the 2026 transfer
    portal. Used by the 'In 2026 portal' sidebar filter. Bridges
    portal_rank_player.player_id (cb_id) -> rosters.player_id -> the
    player_ncaa_season_id PBP files key on. Players in the portal but
    without a 2026 roster row simply won't be present in PBP data anyway.
    """
    portal_path = DATA_DIR / 'portal_rank_player.csv'
    rosters_path = DATA_DIR / 'rosters.csv'
    if not portal_path.exists() or not rosters_path.exists():
        return set()
    portal = pd.read_csv(portal_path, encoding='latin-1', low_memory=False)
    portal['year'] = pd.to_numeric(portal['year'], errors='coerce')
    cb_ids = (pd.to_numeric(portal.loc[portal['year'] == 2026, 'player_id'],
                            errors='coerce')
                .dropna().astype(int).unique())
    if len(cb_ids) == 0:
        return set()
    rosters = pd.read_csv(rosters_path, low_memory=False,
                          usecols=lambda c: c in ('player_id', 'player_ncaa_season_id', 'Year'))
    if 'Year' in rosters.columns:
        rosters['Year'] = pd.to_numeric(rosters['Year'], errors='coerce')
        rosters = rosters[rosters['Year'] == 2026]
    rosters['player_id'] = pd.to_numeric(rosters['player_id'], errors='coerce').astype('Int64')
    rosters['player_ncaa_season_id'] = pd.to_numeric(rosters['player_ncaa_season_id'],
                                                       errors='coerce').astype('Int64')
    rosters = rosters.dropna(subset=['player_id', 'player_ncaa_season_id'])
    matched = rosters[rosters['player_id'].isin(cb_ids)]
    return set(matched['player_ncaa_season_id'].astype(int).astype(str))


@st.cache_data
def load_division_teams(sport, division):
    """Get set of team names belonging to a specific division via conferences.
    Returns the union of teams.csv names AND any PBP-file teamName variants
    that normalize to a canonical teams.csv name (handles "Lamar" vs
    "Lamar University" style suffix drift).
    """
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
    canonical = set(div_teams['name'].dropna())

    # Also include any PBP teamName variants that normalize to a canonical
    # name. Without this, teams like Lamar ("Lamar" in teams.csv, "Lamar
    # University" in PBP files) get filtered out of the whole PBP dataset.
    norm_canonical = {_norm_team(n) for n in canonical}
    pbp_path = PBP_DIR / sport / 'hitting_pbp_D1.csv'
    if pbp_path.exists():
        try:
            pbp_names = pd.read_csv(pbp_path, low_memory=False, usecols=['teamName'])['teamName'].dropna().unique()
            for pn in pbp_names:
                if _norm_team(pn) in norm_canonical:
                    canonical.add(pn)
        except Exception:
            pass
    return canonical


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
view = st.sidebar.radio('Mode', ['Hitter Stats', 'Pitcher Stats', 'Fielding Stats', 'Pace Chart', 'Lineup Card', 'Share Graphic', 'Top 25 Rankings', 'Weekly Awards'], horizontal=True)

# Top 25 Rankings — early-exit branch so the rest of the page (which is built
# around per-player PBP) doesn't run.
if view == 'Top 25 Rankings':
    from app_lib.top25_render import build_top25_svg, build_teams_payload
    st.title('Top 25 Rankings')
    st.caption('1080×1350 weekly graphic. Sources from the 64 Analytics company ranking (team_rank.csv).')

    week_label = st.sidebar.text_input('Week label', value=f'Week of {pd.Timestamp.now().strftime("%b %d, %Y")}')

    sport_label = sport.title()
    teams_df_raw = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False)

    # Pull the 64A company rank, filter to this season + sport/div, attach
    # team name + W-L from elo_custom_all (canonical schedule-derived record).
    team_rank = pd.read_csv(_APP_DIR / 'data' / 'team_rank.csv', low_memory=False)
    team_rank = team_rank[team_rank['year'] == 2026]
    elo = pd.read_csv(_APP_DIR / 'data' / 'rankings' / 'elo_custom_all.csv')
    elo_sd = elo[(elo['sport'] == sport) & (elo['division'] == division)][
        ['team_id', 'wins', 'losses']
    ]
    rank_df = team_rank.merge(elo_sd, on='team_id', how='inner')
    rank_df = rank_df.merge(teams_df_raw[['id', 'name']], left_on='team_id',
                              right_on='id', how='left')
    rank_df = rank_df.rename(columns={
        'integer_64_rank_total': 'rank',
        'name': 'teamName',
    })
    rank_df = rank_df.dropna(subset=['rank', 'teamName']).sort_values('rank')
    rank_df['record'] = (rank_df['wins'].fillna(0).astype(int).astype(str)
                          + '-' + rank_df['losses'].fillna(0).astype(int).astype(str))

    teams_payload = build_teams_payload(rank_df, teams_df_raw, top_n=25,
                                          sport_key=sport)
    if not teams_payload:
        st.error('No ranking data available for this sport/division.')
        st.stop()

    svg = build_top25_svg(teams_payload, sport_label, division,
                           week_label=week_label)

    # On-page preview — scale via CSS but keep viewBox intrinsic
    display_svg = svg.replace(
        '<svg ',
        '<svg style="width:100%;max-width:540px;height:auto;display:block;'
        'margin:0 auto;border-radius:8px;box-shadow:0 8px 24px rgba(0,0,0,.35);" ', 1,
    )
    st.markdown(display_svg, unsafe_allow_html=True)

    # PNG export
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(bytestring=svg.encode('utf-8'), output_width=1080)
        fname = f'top25_{sport}_{division}_{pd.Timestamp.now().strftime("%Y%m%d")}.png'
        st.download_button('Download PNG (1080×1350)', data=png_bytes,
                            file_name=fname, mime='image/png',
                            use_container_width=False)
    except Exception as e:
        st.caption(f'PNG export unavailable in this environment ({type(e).__name__}: {str(e)[:80]}).')

    st.stop()

# Weekly Awards — 1080x1080 'Top 10' graphic per the Claude-Design hand-off.
if view == 'Weekly Awards':
    from app_lib.weekly_awards_render import build_weekly_awards_svg, build_rows_payload
    import base64 as _b64

    st.title('Weekly Awards — Top 10')
    st.caption('1080×1080 weekly award graphic (pitching or hitting). Pick the rank stat, '
               'optionally upload a hero image, then download the PNG.')

    # ── Sidebar controls ──
    wa_stat_type = st.sidebar.selectbox('Stat type', ['pitching', 'hitting'],
                                          key='wa_stat_type')

    # Stat catalog — same shape as the Share-Graphic catalog. The 'Rank'
    # entry is the composite score from compute_grouped_hitting /
    # compute_grouped_pitching (percentile-rank blend used throughout the
    # PBP Analytics page) — listed first so it's the default sort.
    HIT_CAT = [
        ('Rank','RANK',3,False),
        ('BA','AVG',3,False),('OBP','OBP',3,False),('SLG','SLG',3,False),
        ('OPS','OPS',3,False),('ISO','ISO',3,False),('wOBA','wOBA',3,False),
        ('wRC+','wRC+',0,False),('wRAA','wRAA',1,False),
        ('HR','HR',0,False),('H','H',0,False),('TB','TB',0,False),
        ('R','R',0,False),('RBI','RBI',0,False),('SB','SB',0,False),
        ('K%','K%',1,True),('BB%','BB%',1,False),
    ]
    PIT_CAT = [
        ('Rank','RANK',3,False),
        ('ERA','ERA',2,True),('FIP','FIP',2,True),('WHIP','WHIP',2,True),
        ('K/9','K/9',2,False),('BB/9','BB/9',2,True),('K-BB%','K-BB%',1,False),
        ('K%','K%',1,False),('BB%','BB%',1,True),('BAA','BAA',3,True),
        ('SO','SO',0,False),('IP','IP',1,False),('GmSc','GmSc',1,False),
    ]
    catalog = PIT_CAT if wa_stat_type == 'pitching' else HIT_CAT
    stat_labels = [c[0] for c in catalog]
    wa_rank_stat = st.sidebar.selectbox('Rank by', stat_labels, key='wa_rank_stat')
    wa_meta = next(c for c in catalog if c[0] == wa_rank_stat)
    _, wa_default_suffix, wa_default_dec, wa_sort_asc = wa_meta

    wa_min = st.sidebar.number_input(
        'Min BF / PA', value=50, min_value=1, step=10, key='wa_min',
        help='Qualifying threshold — BF for pitching, PA for hitting.')

    today = pd.Timestamp.now()
    # ── Week / date-range selection ──
    # Pulls dates from the PBP data so the slider only spans real games.
    _pbp_for_dates = load_pbp(sport, division, wa_stat_type)
    if _pbp_for_dates is not None and not _pbp_for_dates.empty and 'date_parsed' in _pbp_for_dates.columns:
        _min_d = _pbp_for_dates['date_parsed'].min()
        _max_d = _pbp_for_dates['date_parsed'].max()
        if pd.notna(_min_d) and pd.notna(_max_d):
            _today = pd.Timestamp.now().normalize()
            _max_avail = min(pd.Timestamp(_max_d).normalize(), _today)
            _default_start = _max_avail - pd.Timedelta(days=6)
            if _default_start < pd.Timestamp(_min_d).normalize():
                _default_start = pd.Timestamp(_min_d).normalize()
            wa_date_range = st.sidebar.date_input(
                'Week range (Mon–Sun typical)',
                value=(_default_start.date(), _max_avail.date()),
                min_value=pd.Timestamp(_min_d).date(),
                max_value=pd.Timestamp(_max_d).date(),
                key='wa_date_range',
                help='Filters the leaderboard to games in this window only. '
                     'Defaults to the last 7 days of available data.',
            )
        else:
            wa_date_range = None
    else:
        wa_date_range = None

    if wa_date_range and len(wa_date_range) == 2:
        wa_start, wa_end = wa_date_range
        wk_default = (f'WEEK {pd.Timestamp(wa_end).isocalendar().week} | '
                       f'{pd.Timestamp(wa_start).strftime("%b %d").upper()} – '
                       f'{pd.Timestamp(wa_end).strftime("%b %d").upper()}')
    else:
        wa_start = wa_end = None
        wk_default = (f'WEEK {today.isocalendar().week} | '
                       f'{(today - pd.Timedelta(days=6)).strftime("%b %d").upper()} – '
                       f'{today.strftime("%b %d").upper()}')
    wa_week = st.sidebar.text_input('Week tag (auto-fills from range, editable)',
                                      value=wk_default, key='wa_week')

    sport_label = sport.upper()
    div_label = division.upper()
    role = 'PITCHERS' if wa_stat_type == 'pitching' else 'HITTERS'
    sub_default = f'{div_label} {sport_label} {role}'
    wa_sub = st.sidebar.text_input('Headline subtitle', value=sub_default, key='wa_sub')

    wa_suffix = st.sidebar.text_input('Stat suffix (after value)',
                                       value=wa_default_suffix if wa_rank_stat in ('ERA','WHIP','FIP') else '',
                                       key='wa_suffix')
    wa_decimals = st.sidebar.number_input('Decimals (rank stat)',
                                            value=int(wa_default_dec),
                                            min_value=0, max_value=4, step=1,
                                            key='wa_decimals')
    wa_show_team = st.sidebar.toggle('Show team under player name',
                                       value=True, key='wa_show_team')

    # Hero upload — placed in the main content above the preview so it's
    # discoverable. Drops into the right-panel slot of the rendered SVG.
    wa_hero = st.file_uploader(
        'Hero image — drops into the right panel of the graphic',
        type=['png', 'jpg', 'jpeg'], key='wa_hero',
        help='Upload an action shot, team photo, or chart. The image is '
             'cover-fit into the empty right panel; nothing renders without it.',
    )

    # ── Compute leaderboard ──
    pbp_data = load_pbp(sport, division, wa_stat_type)
    if pbp_data is None or pbp_data.empty:
        st.error(f'No {wa_stat_type} PBP data found for {sport} {division}')
        st.stop()

    # Restrict to the selected week's date range so the leaderboard
    # reflects only that week's games, not season-to-date.
    if wa_start is not None and wa_end is not None and 'date_parsed' in pbp_data.columns:
        _ws = pd.Timestamp(wa_start)
        _we = pd.Timestamp(wa_end) + pd.Timedelta(days=1)  # end-inclusive
        pbp_data = pbp_data[
            (pbp_data['date_parsed'] >= _ws) & (pbp_data['date_parsed'] < _we)
        ].copy()
        if pbp_data.empty:
            st.warning(f'No {wa_stat_type} games in the selected week range.')
            st.stop()
        st.sidebar.caption(f'{len(pbp_data):,} game lines in this week')

    rank_col = 'playerId' if 'playerId' in pbp_data.columns else 'playerName'
    if wa_stat_type == 'hitting':
        league = compute_hitting_stats(pbp_data)
        df_top = compute_grouped_hitting(pbp_data, rank_col, league['wOBA'],
                                          league_r_pa=league.get('R/PA', 0),
                                          min_pa=wa_min)
    else:
        df_top = compute_grouped_pitching(pbp_data, rank_col, min_bf=wa_min)

    if wa_rank_stat not in df_top.columns:
        st.error(f"Stat '{wa_rank_stat}' not in computed columns.")
        st.stop()

    df_top[wa_rank_stat] = pd.to_numeric(df_top[wa_rank_stat], errors='coerce')
    df_top = df_top.dropna(subset=[wa_rank_stat])
    df_top = df_top.sort_values(wa_rank_stat, ascending=wa_sort_asc).reset_index(drop=True)
    if df_top.empty:
        st.warning('No qualifying players for that stat / threshold.')
        st.stop()

    # Bridge playerId -> playerName + school (from chart-builder players.csv)
    name_col = 'playerName' if 'playerName' in df_top.columns else rank_col
    if 'School' not in df_top.columns:
        df_top['School'] = ''
    team_col = 'School'

    teams_df = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False)
    rows_payload = build_rows_payload(df_top, name_col=name_col,
                                        stat_col=wa_rank_stat,
                                        team_col=team_col, top_n=10,
                                        sport_key=sport, teams_df=teams_df)

    # Auto-pick 6 bottom stats per stat-type. Pitching uses A-OPS (OPS
    # Against), sourced from the 'OPS Against' column in
    # compute_grouped_pitching but rendered with the 'A-OPS' label.
    # Tuple: (display_label, dataframe_column, decimals, sort_ascending)
    if wa_stat_type == 'pitching':
        bottom_picks = [('IP','IP',1,False), ('H','H',0,False),
                         ('ER','ER',0,False), ('SO','SO',0,False),
                         ('A-OPS','OPS Against',3,True),
                         ('FIP','FIP',2,True)]
    else:
        bottom_picks = [('AB','AB',0,False), ('H','H',0,False),
                         ('2B','2B',0,False), ('HR','HR',0,False),
                         ('OPS','OPS',3,False), ('wRAA','wRAA',1,False)]
    top_stats_payload = []
    for label, col, dec, asc in bottom_picks:
        if col not in df_top.columns:
            top_stats_payload.append({'label': label, 'value': None,
                                       'decimals': dec, 'leader': ''})
            continue
        s = df_top[[name_col, col]].dropna()
        s[col] = pd.to_numeric(s[col], errors='coerce')
        s = s.dropna(subset=[col]).sort_values(col, ascending=asc)
        if s.empty:
            val, leader = None, ''
        else:
            top = s.iloc[0]
            val = float(top[col])
            leader = str(top[name_col])
            parts = leader.split()
            if len(parts) >= 2:
                leader = f'{parts[0][0]}. {parts[-1]}'
        top_stats_payload.append({'label': label, 'value': val,
                                   'decimals': dec, 'leader': leader})

    hero_b64 = None
    if wa_hero is not None:
        mime = 'image/png' if wa_hero.name.lower().endswith('.png') else 'image/jpeg'
        hero_b64 = f'data:{mime};base64,' + _b64.b64encode(wa_hero.read()).decode('ascii')

    svg = build_weekly_awards_svg(
        rows_payload, top_stats_payload,
        sport=sport, division=division, stat_type=wa_stat_type,
        week_label=wa_week, headline_sub=wa_sub,
        stat_suffix=wa_suffix, stat_decimals=int(wa_decimals),
        show_team_subline=wa_show_team, hero_b64=hero_b64,
    )

    # On-page preview (responsive scale)
    display_svg = svg.replace(
        '<svg ',
        '<svg style="width:100%;max-width:600px;height:auto;display:block;'
        'margin:0 auto;border-radius:8px;box-shadow:0 8px 24px rgba(0,0,0,.35);" ', 1,
    )
    st.markdown(display_svg, unsafe_allow_html=True)

    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(bytestring=svg.encode('utf-8'), output_width=1080)
        fname = f'weekly_awards_{sport}_{division}_{wa_stat_type}_{today.strftime("%Y%m%d")}.png'
        st.download_button('Download PNG (1080×1080)', data=png_bytes,
                            file_name=fname, mime='image/png',
                            use_container_width=False)
    except Exception as e:
        st.caption(f'PNG export unavailable here ({type(e).__name__}: {str(e)[:80]}).')

    st.stop()

# Team aggregation toggle for the three stat-table views (Pace Chart has its
# own Player/Team selector built in; Lineup Card is per-team by definition).
if view in ('Hitter Stats', 'Pitcher Stats', 'Fielding Stats'):
    group_by = st.sidebar.radio('Group by', ['Player', 'Team'], horizontal=True,
                                help='Team aggregates all players on each team — use it to compare team-level wRC+, ERA, FPCT, etc.')
elif view == 'Share Graphic':
    group_by = st.sidebar.radio('Group by', ['Team', 'Player'], horizontal=True,
                                help='Top-N teams or players for the chosen stat.')
else:
    group_by = 'Player'

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
elif view == 'Share Graphic':
    sg_stat_type = st.sidebar.selectbox('Stat type', ['hitting', 'pitching', 'fielding'],
                                        key='sg_stat_type')
    pbp = load_pbp(sport, division, sg_stat_type)
    if pbp is None:
        st.error(f'No {sg_stat_type} PBP data found for {sport} {division}')
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

# Home / Away filter
st.sidebar.markdown('---')
st.sidebar.markdown('### Game Context')
venue_filter = st.sidebar.selectbox('Venue', ['All', 'Home', 'Away'], key='venue_filter')
if venue_filter == 'Home':
    pbp = pbp[pbp['isHome'] == 1]
    if view == 'Lineup Card':
        hitting_pbp = hitting_pbp[hitting_pbp['isHome'] == 1]
        pitching_pbp = pitching_pbp[pitching_pbp['isHome'] == 1]
elif venue_filter == 'Away':
    pbp = pbp[pbp['isHome'] != 1]
    if view == 'Lineup Card':
        hitting_pbp = hitting_pbp[hitting_pbp['isHome'] != 1]
        pitching_pbp = pitching_pbp[pitching_pbp['isHome'] != 1]

# Conference / Non-Conference game filter
conf_map = load_team_conference_map()
game_type_filter = st.sidebar.selectbox('Game Type', ['All', 'Conference', 'Non-Conference'], key='game_type_filter')
if game_type_filter != 'All' and conf_map and 'teamName' in pbp.columns and 'gameId' in pbp.columns:
    # For each game, check if both teams are in the same conference
    game_teams = pbp.groupby('gameId')['teamName'].apply(set).reset_index()
    conf_games = set()
    non_conf_games = set()
    for _, row in game_teams.iterrows():
        teams_in_game = list(row['teamName'])
        if len(teams_in_game) == 2:
            c1 = conf_map.get(teams_in_game[0], '')
            c2 = conf_map.get(teams_in_game[1], '')
            if c1 and c2 and c1 == c2:
                conf_games.add(row['gameId'])
            else:
                non_conf_games.add(row['gameId'])
        else:
            non_conf_games.add(row['gameId'])
    if game_type_filter == 'Conference':
        pbp = pbp[pbp['gameId'].isin(conf_games)]
        if view == 'Lineup Card':
            hitting_pbp = hitting_pbp[hitting_pbp['gameId'].isin(conf_games)]
            pitching_pbp = pitching_pbp[pitching_pbp['gameId'].isin(conf_games)]
    else:
        pbp = pbp[pbp['gameId'].isin(non_conf_games)]
        if view == 'Lineup Card':
            hitting_pbp = hitting_pbp[hitting_pbp['gameId'].isin(non_conf_games)]
            pitching_pbp = pitching_pbp[pitching_pbp['gameId'].isin(non_conf_games)]

# Weekend / Midweek filter (Fri-Sun vs Mon-Thu)
weekday_filter = st.sidebar.selectbox(
    'Day of Week',
    ['All', 'Weekend (Fri-Sun)', 'Midweek (Mon-Thu)'],
    key='weekday_filter',
    help='Weekend is conference-series games; midweek is typically non-con single games.',
)
if weekday_filter != 'All' and 'date_parsed' in pbp.columns:
    weekend_days = {4, 5, 6}  # Fri=4, Sat=5, Sun=6
    if weekday_filter == 'Weekend (Fri-Sun)':
        pbp = pbp[pbp['date_parsed'].dt.weekday.isin(weekend_days)]
        if view == 'Lineup Card':
            hitting_pbp = hitting_pbp[hitting_pbp['date_parsed'].dt.weekday.isin(weekend_days)]
            pitching_pbp = pitching_pbp[pitching_pbp['date_parsed'].dt.weekday.isin(weekend_days)]
    else:
        pbp = pbp[~pbp['date_parsed'].dt.weekday.isin(weekend_days)]
        if view == 'Lineup Card':
            hitting_pbp = hitting_pbp[~hitting_pbp['date_parsed'].dt.weekday.isin(weekend_days)]
            pitching_pbp = pitching_pbp[~pitching_pbp['date_parsed'].dt.weekday.isin(weekend_days)]

# Conference filter (applies to all views)
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

# Conference Strength filter (D-I only: Upper / Middle / Lower)
class_map = load_team_classification_map()
if class_map and 'teamName' in pbp.columns:
    strength_filter = st.sidebar.selectbox(
        'Conference Strength',
        ['All', 'Upper', 'Middle', 'Lower'],
        help='D-I conference tier from conferences.csv. D-II / D-III teams have no tier and will be excluded when a specific tier is selected.',
    )
    if strength_filter != 'All':
        target = strength_filter.lower()
        strength_teams = {t for t, c in class_map.items() if c == target}
        pbp = pbp[pbp['teamName'].isin(strength_teams)]
        if view == 'Lineup Card':
            hitting_pbp = hitting_pbp[hitting_pbp['teamName'].isin(strength_teams)]
            pitching_pbp = pitching_pbp[pitching_pbp['teamName'].isin(strength_teams)]

# 2026 portal filter — applies to any Player-mode view (Hitter / Pitcher /
# Fielding Stats + Share Graphic). Surfaced before the per-view filter blocks
# so the same checkbox state survives across views.
if (view in ('Hitter Stats', 'Pitcher Stats', 'Fielding Stats', 'Share Graphic')
        and group_by == 'Player' and 'playerId' in pbp.columns):
    portal_only = st.sidebar.checkbox(
        'In 2026 portal',
        key='in_2026_portal',
        help='Restrict to players in the 2026 transfer portal '
             '(portal_rank_player.csv year=2026 → rosters → playerId).',
    )
    if portal_only:
        portal_pids = load_portal_2026_ncaa_pids()
        before = len(pbp)
        pbp = pbp[pbp['playerId'].astype(str).isin(portal_pids)]
        st.sidebar.caption(
            f'{len(pbp):,} of {before:,} lines · '
            f'{len(portal_pids):,} 2026 portal players in roster bridge'
        )

# Team / Position / Player filters — not shown for Lineup Card, Who's Hot, or Share Graphic
if view not in ('Lineup Card', 'Share Graphic'):
    # Team filter
    all_teams = sorted(pbp['teamName'].dropna().unique()) if 'teamName' in pbp.columns else []
    team_list = ['All'] + all_teams
    selected_team = st.sidebar.selectbox('Team', team_list)
    if selected_team != 'All':
        pbp = pbp[pbp['teamName'] == selected_team]

    # Position filter — only meaningful when grouping by Player
    if group_by == 'Player' and 'playerPosition' in pbp.columns:
        all_positions = sorted(pbp['playerPosition'].dropna().unique())
        selected_positions = st.sidebar.multiselect('Position', all_positions,
                                                     help='Leave empty for all positions')
        if selected_positions:
            pbp = pbp[pbp['playerPosition'].isin(selected_positions)]

    # Min PA/BF — defaults shift up in Team mode where aggregates are larger
    st.sidebar.markdown('---')
    if view == 'Hitter Stats':
        default_min = 200 if group_by == 'Team' else 10
        min_threshold = st.sidebar.number_input('Min PA', value=default_min, min_value=1, step=5)
    elif view == 'Pitcher Stats':
        default_min = 100 if group_by == 'Team' else 10
        min_threshold = st.sidebar.number_input('Min BF', value=default_min, min_value=1, step=5)
        min_ip = st.sidebar.number_input('Min IP', value=0.0, min_value=0.0, step=5.0, format='%.1f')

    # Player filter only meaningful in Player mode; teamName grouping in Team mode
    if group_by == 'Team':
        player_col = 'teamName'
        selected_players = []
    else:
        player_col = 'playerId' if 'playerId' in pbp.columns else 'playerName'
        available_players = sorted(pbp['playerName'].dropna().unique()) if 'playerName' in pbp.columns else []
        selected_players = st.sidebar.multiselect('Filter players', available_players,
                                                   help='Leave empty for all')
        if selected_players:
            pbp = pbp[pbp['playerName'].isin(selected_players)]

    if len(pbp) == 0:
        st.warning('No events match your filters.')
        st.stop()
elif view == 'Lineup Card':
    # Lineup Card specific controls
    st.sidebar.markdown('---')
    min_pa_lc = st.sidebar.number_input('Min PA (eligibility)', value=10, min_value=1, step=1,
                                         help='Minimum PA to be eligible for the lineup card.')
    rank_min_pa_lc = st.sidebar.number_input('Min PA (rank pop)', value=10, min_value=1, step=1,
                                              help='PA threshold for the rank calculation population. Set to match the Hitter Stats Min PA so the lineup card and Hitter Stats agree on rankings.')
    min_bf_sp = st.sidebar.number_input('Min BF (starters)', value=50, min_value=1, step=10)
    min_bf_rp = st.sidebar.number_input('Min BF (relievers)', value=15, min_value=1, step=5)
    player_col = 'playerId' if 'playerId' in pbp.columns else 'playerName'

# ── Compute and Display ─────────────────────────────────────────────────────
if view == 'Hitter Stats':
    st.markdown(f'### Hitter Stats — {sport.title()} {division}')

    # Compute league wOBA for wRAA
    league_stats = compute_hitting_stats(pbp)
    league_woba = league_stats['wOBA']
    league_r_pa = league_stats['R/PA']

    # Overall summary — sample-wide rate + counting stats
    st.markdown('**Rate Stats (sample)**')
    r1 = st.columns(7)
    r1[0].metric('BA', f"{league_stats['BA']:.3f}")
    r1[1].metric('OBP', f"{league_stats['OBP']:.3f}")
    r1[2].metric('SLG', f"{league_stats['SLG']:.3f}")
    r1[3].metric('OPS', f"{league_stats['OPS']:.3f}")
    r1[4].metric('ISO', f"{league_stats['ISO']:.3f}")
    r1[5].metric('BABIP', f"{league_stats['BABIP']:.3f}")
    r1[6].metric('wOBA', f"{league_woba:.3f}")

    r2 = st.columns(7)
    r2[0].metric('K%', f"{league_stats['K%']:.1f}")
    r2[1].metric('BB%', f"{league_stats['BB%']:.1f}")
    r2[2].metric('K/BB', f"{league_stats['K/BB']:.2f}")
    r2[3].metric('R/PA', f"{league_stats['R/PA']:.3f}")
    r2[4].metric('PA', f"{league_stats['PA']:,}")
    r2[5].metric('AB', f"{league_stats['AB']:,}")
    r2[6].metric('H', f"{league_stats['H']:,}")

    st.markdown('**Counting Stats (sample)**')
    r3 = st.columns(8)
    r3[0].metric('1B', f"{league_stats['1B']:,}")
    r3[1].metric('2B', f"{league_stats['2B']:,}")
    r3[2].metric('3B', f"{league_stats['3B']:,}")
    r3[3].metric('HR', f"{league_stats['HR']:,}")
    r3[4].metric('TB', f"{league_stats['TB']:,}")
    r3[5].metric('R', f"{league_stats['R']:,}")
    r3[6].metric('RBI', f"{league_stats['RBI']:,}")
    r3[7].metric('K', f"{league_stats['K']:,}")

    r4 = st.columns(8)
    r4[0].metric('BB', f"{league_stats['BB']:,}")
    r4[1].metric('HBP', f"{league_stats['HBP']:,}")
    r4[2].metric('IBB', f"{league_stats['IBB']:,}")
    r4[3].metric('SF', f"{league_stats['SF']:,}")
    r4[4].metric('SH', f"{league_stats['SH']:,}")
    r4[5].metric('SB', f"{league_stats['SB']:,}")
    r4[6].metric('CS', f"{league_stats['CS']:,}")
    r4[7].metric('GDP', f"{league_stats['GDP']:,}")

    # Per-player table
    st.markdown('---')
    player_stats = compute_grouped_hitting(pbp, player_col, league_woba, league_r_pa=league_r_pa, min_pa=min_threshold)

    if len(player_stats) == 0:
        st.info(f'No {"teams" if group_by == "Team" else "players"} meet the {min_threshold} PA minimum.')
    else:
        if group_by == 'Team':
            display_col = 'teamName'
            show_cols = ['Rank', 'teamName', 'PA', 'AB', 'H', '1B', '2B', '3B', 'HR', 'TB',
                         'R', 'RBI', 'BB', 'HBP', 'SF', 'SH', 'IBB', 'K', 'SB', 'CS', 'GDP',
                         'BA', 'OBP', 'SLG', 'OPS', 'ISO', 'BABIP',
                         'K%', 'BB%', 'K/BB', 'R/PA',
                         'wOBA', 'wRAA', 'wRC', 'wRC+']
        else:
            display_col = 'playerName' if 'playerName' in player_stats.columns else player_col
            show_cols = ['Rank', display_col, 'School', 'Pos', 'PA', 'AB', 'H', '1B', '2B', '3B', 'HR', 'TB',
                         'R', 'RBI', 'BB', 'HBP', 'SF', 'SH', 'IBB', 'K', 'SB', 'CS', 'GDP',
                         'BA', 'OBP', 'SLG', 'OPS', 'ISO', 'BABIP',
                         'K%', 'BB%', 'K/BB', 'R/PA',
                         'wOBA', 'wRAA', 'wRC', 'wRC+']
        show_cols = [c for c in show_cols if c in player_stats.columns]
        st.dataframe(player_stats[show_cols], use_container_width=True, hide_index=True, height=1050)

        csv_buf = player_stats[show_cols].to_csv(index=False)
        scope_label = 'team' if group_by == 'Team' else 'player'
        st.download_button('Download CSV', data=csv_buf,
                          file_name=f'pbp_hitting_{scope_label}_{sport}_{division}.csv', mime='text/csv')

    # Single player deep dive
    if selected_players and len(selected_players) == 1:
        st.markdown(f'### {selected_players[0]} — Game Log')
        player_data = pbp[pbp['playerName'] == selected_players[0]]

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

    # Overall summary — sample-wide rate + counting stats
    overall = compute_pitching_stats(pbp)
    k_per_label = 'K/7' if sport == 'softball' else 'K/9'
    k_per_val = overall['K/7'] if sport == 'softball' else overall['K/9']

    st.markdown('**Rate Stats (sample)**')
    r1 = st.columns(7)
    r1[0].metric('ERA', f"{overall['ERA']:.2f}")
    r1[1].metric('FIP', f"{overall['FIP']:.2f}")
    r1[2].metric('WHIP', f"{overall['WHIP']:.2f}")
    r1[3].metric('BAA', f"{overall['BAA']:.3f}")
    r1[4].metric('BABIP', f"{overall['BABIP']:.3f}")
    r1[5].metric('OPS-A', f"{overall['OPS Against']:.3f}")
    r1[6].metric('OBP-A', f"{overall['OBP Against']:.3f}")

    r2 = st.columns(7)
    r2[0].metric('SLG-A', f"{overall['SLG Against']:.3f}")
    r2[1].metric('K%', f"{overall['K%']:.1f}")
    r2[2].metric('BB%', f"{overall['BB%']:.1f}")
    r2[3].metric(k_per_label, f"{k_per_val:.2f}")
    r2[4].metric('K/BB', f"{overall['K/BB']:.2f}")
    r2[5].metric('IP', f"{overall['IP']}")
    r2[6].metric('App', f"{overall['App']:,}")

    st.markdown('**Counting Stats (sample)**')
    r3 = st.columns(8)
    r3[0].metric('BF', f"{overall['BF']:,}")
    r3[1].metric('OAB', f"{overall['OAB']:,}")
    r3[2].metric('H', f"{overall['H']:,}")
    r3[3].metric('R', f"{overall['R']:,}")
    r3[4].metric('ER', f"{overall['ER']:,}")
    r3[5].metric('BB', f"{overall['BB']:,}")
    r3[6].metric('SO', f"{overall['SO']:,}")
    r3[7].metric('HR', f"{overall['HR']:,}")

    r4 = st.columns(8)
    r4[0].metric('HB', f"{overall['HB']:,}")
    r4[1].metric('IBB', f"{overall['IBB']:,}")
    r4[2].metric('2B-A', f"{overall['2B-A']:,}")
    r4[3].metric('3B-A', f"{overall['3B-A']:,}")
    r4[4].metric('WP', f"{overall['WP']:,}")
    r4[5].metric('Bk', f"{overall['Bk']:,}")
    r4[6].metric('SHA', f"{overall['SHA']:,}")
    r4[7].metric('SFA', f"{overall['SFA']:,}")

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
        if group_by == 'Team':
            show_cols = ['Rank', 'teamName', 'App', 'GS', 'IP', 'BF', 'OAB',
                         'H', 'R', 'ER', 'BB', 'HB', 'SO',
                         'HR', '2B-A', '3B-A', 'Bk', 'IBB', 'SHA', 'SFA',
                         'ERA', 'FIP', 'GmSc', 'BAA', 'BABIP',
                         'OBP Against', 'SLG Against', 'OPS Against',
                         'K%', 'BB%', k_col, 'K/BB', 'WHIP']
        else:
            pit_display_col = 'playerName' if 'playerName' in pitcher_stats.columns else player_col
            show_cols = ['Rank', pit_display_col, 'School', 'Pos', 'App', 'GS', 'IP', 'BF', 'OAB',
                         'H', 'R', 'ER', 'BB', 'HB', 'SO',
                         'HR', '2B-A', '3B-A', 'Bk', 'IBB', 'SHA', 'SFA',
                         'ERA', 'FIP', 'GmSc', 'BAA', 'BABIP',
                         'OBP Against', 'SLG Against', 'OPS Against',
                         'K%', 'BB%', k_col, 'K/BB', 'WHIP']
        show_cols = [c for c in show_cols if c in pitcher_stats.columns]
        st.dataframe(pitcher_stats[show_cols], use_container_width=True, hide_index=True, height=1050)

        csv_buf = pitcher_stats[show_cols].to_csv(index=False)
        scope_label = 'team' if group_by == 'Team' else 'player'
        st.download_button('Download CSV', data=csv_buf,
                          file_name=f'pbp_pitching_{scope_label}_{sport}_{division}.csv', mime='text/csv')

    # Single pitcher deep dive
    if selected_players and len(selected_players) == 1:
        st.markdown(f'### {selected_players[0]} — Game Log')
        pitcher_data = pbp[pbp['playerName'] == selected_players[0]]

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

    # Overall summary — sample-wide fielding stats
    overall = compute_fielding_stats(pbp)

    st.markdown('**Rate Stats (sample)**')
    r1 = st.columns(2)
    r1[0].metric('FPCT', f"{overall['FPCT']:.3f}")
    r1[1].metric('CS%', f"{overall['CS%']:.3f}")

    st.markdown('**Counting Stats (sample)**')
    r2 = st.columns(8)
    r2[0].metric('PO', f"{overall['PO']:,}")
    r2[1].metric('A', f"{overall['A']:,}")
    r2[2].metric('TC', f"{overall['TC']:,}")
    r2[3].metric('E', f"{overall['E']:,}")
    r2[4].metric('PB', f"{overall['PB']:,}")
    r2[5].metric('SBA', f"{overall['SBA']:,}")
    r2[6].metric('CSB', f"{overall['CSB']:,}")
    r2[7].metric('IDP', f"{overall['IDP']:,}")

    r3 = st.columns(8)
    r3[0].metric('TP', f"{overall['TP']:,}")

    # Per-player table
    st.markdown('---')
    fielding_stats = compute_grouped_fielding(pbp, player_col)

    if len(fielding_stats) == 0:
        st.info('No fielding data available.')
    else:
        if group_by == 'Team':
            show_cols = ['teamName', 'PO', 'A', 'TC', 'E', 'FPCT',
                         'PB', 'SBA', 'CSB', 'CS%', 'IDP', 'TP']
        else:
            fld_display_col = 'playerName' if 'playerName' in fielding_stats.columns else player_col
            show_cols = [fld_display_col, 'School', 'Pos', 'PO', 'A', 'TC', 'E', 'FPCT',
                         'PB', 'SBA', 'CSB', 'CS%', 'IDP', 'TP']
        show_cols = [c for c in show_cols if c in fielding_stats.columns]
        st.dataframe(fielding_stats[show_cols], use_container_width=True, hide_index=True, height=1050)

        csv_buf = fielding_stats[show_cols].to_csv(index=False)
        scope_label = 'team' if group_by == 'Team' else 'player'
        st.download_button('Download CSV', data=csv_buf,
                          file_name=f'pbp_fielding_{scope_label}_{sport}_{division}.csv', mime='text/csv')

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

    # Apply Day of Week filter (sidebar Game Context) — pace_pbp is a fresh
    # load of the hitting file and doesn't automatically inherit the global
    # sidebar filter applied to `pbp` earlier, so re-apply here.
    try:
        if weekday_filter != 'All' and 'date_parsed' in pace_pbp.columns:
            _weekend_days = {4, 5, 6}
            if weekday_filter == 'Weekend (Fri-Sun)':
                pace_pbp = pace_pbp[pace_pbp['date_parsed'].dt.weekday.isin(_weekend_days)]
            else:
                pace_pbp = pace_pbp[~pace_pbp['date_parsed'].dt.weekday.isin(_weekend_days)]
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
    # Run Diff is a team-level-only stat: cumulative (runs scored - runs allowed).
    # Offer it for both Hitting and Pitching tabs when viewing Team data.
    if pace_level == 'Team':
        all_stat_options.append('Run Diff')

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

    # Build running stats per entity per game date — VECTORIZED
    # Step 1: Aggregate per (entity, gameId, date_parsed) using sums of underlying counts.
    # This handles doubleheaders (multiple rows for same gameId) by summing them.
    if pace_stat_type == 'Hitting':
        sum_cols = ['ab', 'h', 'bb', 'hbp', 'sf', 'sh', 'tb', 'hr', 'doubles', 'triples',
                    'k', 'sb', 'cs', 'oppDp', 'r', 'rbi']
    else:
        sum_cols = ['h', 'r', 'er', 'bb', 'so', 'hrA', 'hb', 'bf', 'doublesA', 'triplesA',
                    'sha', 'sfa', 'wp', 'bk']
    sum_cols = [c for c in sum_cols if c in pace_pbp.columns]

    # For pitching, also need ip (special handling)
    if pace_stat_type == 'Pitching' and 'ip' in pace_pbp.columns:
        pace_pbp = pace_pbp.copy()
        pace_pbp['_outs'] = baseball_ip_to_outs(pace_pbp['ip'])
        sum_cols.append('_outs')

    if 'teamName' not in pace_pbp.columns or 'gameId' not in pace_pbp.columns:
        st.warning('Required columns missing.')
        st.stop()

    # Group key + team in one aggregation
    agg_cols = sum_cols
    grouped = pace_pbp.groupby([group_key, 'gameId', 'date_parsed'], sort=False)
    per_game = grouped[agg_cols].sum().reset_index()

    # Get team name per (entity, gameId) — first occurrence.
    # When group_key IS 'teamName', per_game already has the column from reset_index
    # above; skip the self-merge to avoid a ValueError (cannot insert teamName).
    if group_key != 'teamName':
        team_lookup = pace_pbp.groupby([group_key, 'gameId'], sort=False)['teamName'].first().reset_index()
        per_game = per_game.merge(team_lookup, on=[group_key, 'gameId'], how='left')

    # Step 2: Sort by date within each entity, assign game numbers, compute cumsums
    per_game = per_game.sort_values([group_key, 'date_parsed']).reset_index(drop=True)
    per_game['game_num'] = per_game.groupby(group_key, sort=False).cumcount() + 1

    cum_df = per_game[sum_cols].groupby(per_game[group_key], sort=False).cumsum()
    cum_df.columns = [f'cum_{c}' for c in sum_cols]
    per_game = pd.concat([per_game, cum_df], axis=1)

    # Step 3: Compute the cum_stat for the requested stat
    if stat_choice == 'Run Diff':
        # Team-level Run Differential = cumulative (runs scored - runs allowed).
        # Need both teams' per-game runs, so reload the HITTING file (full dataset)
        # and apply only the date / conference filters — the team filter would
        # strip the opponent and make the diff impossible to compute.
        full_hit = load_pbp(sport, division, 'hitting')
        if full_hit is None or 'r' not in full_hit.columns:
            st.warning('Run Diff requires hitting data.')
            st.stop()
        fh = full_hit
        if date_start and date_end and 'date_parsed' in fh.columns:
            fh = fh[(fh['date_parsed'].dt.date >= date_start) & (fh['date_parsed'].dt.date <= date_end)]
        try:
            if conf_map and selected_conferences:
                c_teams = {t for t, c in conf_map.items() if c in selected_conferences}
                fh = fh[fh['teamName'].isin(c_teams) | fh['gameId'].isin(fh[fh['teamName'].isin(c_teams)]['gameId'])]
        except NameError:
            pass
        # Apply Day of Week filter so weekend-only Run Diff excludes midweek
        # games (and midweek-only excludes weekends). Same rule as pace_pbp.
        try:
            if weekday_filter != 'All' and 'date_parsed' in fh.columns:
                _rd_weekend_days = {4, 5, 6}
                if weekday_filter == 'Weekend (Fri-Sun)':
                    fh = fh[fh['date_parsed'].dt.weekday.isin(_rd_weekend_days)]
                else:
                    fh = fh[~fh['date_parsed'].dt.weekday.isin(_rd_weekend_days)]
        except NameError:
            pass
        # Per-game per-team runs scored (hitting 'r' is runs scored)
        tr = fh.groupby(['gameId', 'teamName', 'date_parsed'], sort=False)['r'].sum().reset_index()
        game_totals = tr.groupby('gameId')['r'].sum().to_dict()
        tr['opp_r'] = tr['gameId'].map(game_totals).fillna(0).astype(int) - tr['r'].astype(int)
        tr['run_diff_per_game'] = tr['r'].astype(int) - tr['opp_r']
        # Restrict to the teams currently in per_game (applies team filter)
        wanted_teams = set(per_game['teamName'].unique())
        tr = tr[tr['teamName'].isin(wanted_teams)].copy()
        tr = tr.sort_values(['teamName', 'date_parsed']).reset_index(drop=True)
        tr['game_num'] = tr.groupby('teamName', sort=False).cumcount() + 1
        tr['cum_stat'] = tr.groupby('teamName', sort=False)['run_diff_per_game'].cumsum()
        # Replace per_game with the run-diff frame so the downstream chart
        # render uses these cum values. Ensure all columns the renderer
        # expects exist.
        per_game = tr.rename(columns={})
        per_game[group_key] = per_game['teamName']
    elif not is_advanced:
        stat_col = cum_stats[stat_choice]
        if stat_col not in pace_pbp.columns:
            st.warning(f'{stat_choice} not in data.')
            st.stop()
        per_game['cum_stat'] = per_game[f'cum_{stat_col}']
    else:
        if pace_stat_type == 'Hitting':
            ab = per_game['cum_ab']; h = per_game['cum_h']; bb = per_game['cum_bb']
            hbp = per_game['cum_hbp']; sf = per_game['cum_sf']; sh = per_game['cum_sh']
            tb = per_game['cum_tb']; hr = per_game['cum_hr']
            doubles = per_game['cum_doubles']; triples = per_game['cum_triples']
            k = per_game['cum_k']
            singles = h - doubles - triples - hr
            pa = ab + bb + hbp + sf + sh
            obp_d = ab + bb + hbp + sf
            obp = np.where(obp_d > 0, (h + bb + hbp) / obp_d, 0.0)
            slg = np.where(ab > 0, tb / ab, 0.0)
            ba = np.where(ab > 0, h / ab, 0.0)
            woba_d = ab + bb + sf + hbp
            woba_num = (WOBA_BB * bb + WOBA_HBP * hbp + WOBA_1B * singles +
                        WOBA_2B * doubles + WOBA_3B * triples + WOBA_HR * hr)
            woba_raw = np.where(woba_d > 0, woba_num / woba_d, 0.0)
            woba = np.round(woba_raw, 3)  # match compute_hitting_stats rounding
            babip_d = ab - k - hr + sf
            babip = np.where(babip_d > 0, (h - hr) / babip_d, 0.0)

            if stat_choice == 'wRAA':
                per_game['cum_stat'] = np.round(((woba - league_woba_pace) / WOBA_SCALE) * pa, 1)
            elif stat_choice == 'wRC':
                per_game['cum_stat'] = np.round((((woba - league_woba_pace) / WOBA_SCALE) + league_r_pa_pace) * pa, 1)
            elif stat_choice == 'OPS':
                per_game['cum_stat'] = obp + slg
            elif stat_choice == 'wOBA':
                per_game['cum_stat'] = woba
            elif stat_choice == 'ISO':
                per_game['cum_stat'] = slg - ba
            elif stat_choice == 'BABIP':
                per_game['cum_stat'] = babip
            else:
                per_game['cum_stat'] = 0
        else:
            # Pitching
            outs = per_game['cum__outs']
            ip = outs / 3.0
            h = per_game['cum_h']; bb = per_game['cum_bb']
            so = per_game['cum_so']; hr = per_game['cum_hrA']
            hb = per_game['cum_hb']; bf = per_game['cum_bf']
            er = per_game['cum_er']
            sha = per_game.get('cum_sha', 0); sfa = per_game.get('cum_sfa', 0)
            doubles = per_game.get('cum_doublesA', 0); triples = per_game.get('cum_triplesA', 0)
            singles = h - doubles - triples - hr
            p_oab = bf - bb - hb - sfa - sha
            obp_d = p_oab + bb + hb + sfa
            obp_against = np.where(obp_d > 0, (h + bb + hb) / obp_d, 0.0)
            tb_against = singles + 2 * doubles + 3 * triples + 4 * hr
            slg_against = np.where(p_oab > 0, tb_against / p_oab, 0.0)
            ba_against = np.where(p_oab > 0, h / p_oab, 0.0)
            fip = np.where(ip > 0, ((13 * hr) + (3 * (bb + hb)) - (2 * so)) / ip + FIP_CONSTANT, 0.0)
            era = np.where(ip > 0, (er / ip) * 9, 0.0)
            whip = np.where(ip > 0, (bb + h) / ip, 0.0)
            k9 = np.where(ip > 0, (so / ip) * 9, 0.0)
            k7 = np.where(ip > 0, (so / ip) * 7, 0.0)
            k_pct = np.where(bf > 0, so / bf * 100, 0.0)
            bb_pct = np.where(bf > 0, bb / bf * 100, 0.0)

            if stat_choice == 'FIP':
                per_game['cum_stat'] = fip
            elif stat_choice == 'WHIP':
                per_game['cum_stat'] = whip
            elif stat_choice == 'ERA':
                per_game['cum_stat'] = era
            elif stat_choice == 'K/9':
                per_game['cum_stat'] = k9
            elif stat_choice == 'K/7':
                per_game['cum_stat'] = k7
            elif stat_choice == 'OPS Against':
                per_game['cum_stat'] = obp_against + slg_against
            elif stat_choice == 'BAA':
                per_game['cum_stat'] = ba_against
            elif stat_choice == 'K%':
                per_game['cum_stat'] = k_pct
            elif stat_choice == 'BB%':
                per_game['cum_stat'] = bb_pct
            else:
                per_game['cum_stat'] = 0

    # Step 4: Build the entity column and assemble final dataframe
    # _key uniquely identifies (player+team) so name collisions don't merge
    if pace_level == 'Player':
        per_game['entity'] = per_game[group_key].str.split('\\|\\|\\|').str[0]
        per_game['team'] = per_game[group_key].str.split('\\|\\|\\|').str[1]
        per_game['_key'] = per_game[group_key]
    else:
        per_game['entity'] = per_game[group_key]
        per_game['team'] = per_game['teamName']
        per_game['_key'] = per_game[group_key]

    all_games = per_game[['_key', 'entity', 'team', 'date_parsed', 'game_num', 'cum_stat']].copy()

    if len(all_games) == 0:
        st.warning('No data available.')
        st.stop()

    # Anchor all entities to the start date at 0 (so lines originate from the same point)
    min_date = all_games['date_parsed'].min()
    if date_start:
        min_date = min(min_date, pd.Timestamp(date_start))
    anchors = []
    for key, edata in all_games.groupby('_key', sort=False):
        first_row = edata.sort_values('date_parsed').iloc[0]
        if first_row['date_parsed'] > min_date:
            anchors.append({
                '_key': key,
                'entity': first_row['entity'],
                'team': first_row['team'],
                'date_parsed': min_date,
                'game_num': 0,
                'cum_stat': 0,
            })
    if anchors:
        all_games = pd.concat([all_games, pd.DataFrame(anchors)], ignore_index=True)

    # Extend all entities to the last date in the range (flat line if no data)
    max_date = all_games['date_parsed'].max()
    if date_end:
        max_date = max(max_date, pd.Timestamp(date_end))
    extensions = []
    for key, edata in all_games.groupby('_key', sort=False):
        last_row = edata.sort_values('date_parsed').iloc[-1]
        if last_row['date_parsed'] < max_date:
            extensions.append({
                '_key': key,
                'entity': last_row['entity'],
                'team': last_row['team'],
                'date_parsed': max_date,
                'game_num': last_row['game_num'],
                'cum_stat': last_row['cum_stat'],
            })
    if extensions:
        all_games = pd.concat([all_games, pd.DataFrame(extensions)], ignore_index=True)

    # Filter by min games — use _key so name collisions don't merge
    game_counts = all_games.groupby('_key')['game_num'].max()
    qualified = game_counts[game_counts >= min_games_pace].index
    all_games = all_games[all_games['_key'].isin(qualified)]

    if len(all_games) == 0:
        st.warning('No entities meet the minimum games threshold.')
        st.stop()

    # Sort within each entity by date + game_num. The game_num tiebreaker is
    # essential for doubleheaders: both games share a date, so a date-only sort
    # can flip them and make the line look like cum_HR drops (impossible).
    # game_num was assigned in cumsum order above, so sorting by it matches
    # the cumulative order.
    all_games = all_games.sort_values(['_key', 'date_parsed', 'game_num'])
    final_stats = all_games.groupby(['_key', 'entity', 'team']).agg(
        total=('cum_stat', 'last'), games=('game_num', 'max')).reset_index()
    # For pitching rate stats, lower is better (sort ascending so best = first).
    # For cumulative counting stats (ER, H, BB, SO, HR, etc.), higher volume = "top".
    lower_is_better = stat_choice in ['FIP', 'WHIP', 'ERA', 'BAA', 'OPS Against', 'BB%']
    final_stats = final_stats.sort_values('total', ascending=lower_is_better)

    if pace_level == 'Player':
        options = [f"{row['entity']} ({row['team']})" for _, row in final_stats.iterrows()]
    else:
        options = [row['entity'] for _, row in final_stats.iterrows()]
    # Map dropdown option string -> _key (unique)
    key_map = dict(zip(options, final_stats['_key']))

    top_n = st.sidebar.number_input('Show Top N', value=20, min_value=5, max_value=2000, step=5)
    top_keys = set(final_stats.head(top_n)['_key'])
    filtered = all_games[all_games['_key'].isin(top_keys)]

    highlighted = st.sidebar.multiselect(f'Highlight {pace_level}s', options,
                                          help=f'Select {pace_level.lower()}s to highlight in red')
    highlight_keys = {key_map[p] for p in highlighted}

    # Font setup (match main chart builder)
    _has_font = lambda name: any(name.lower() in f.name.lower() for f in matplotlib.font_manager.fontManager.ttflist)
    TITLE_FONT = 'Franklin Gothic Heavy' if _has_font('Franklin Gothic') else 'DejaVu Sans'
    SUBTITLE_FONT = 'Franklin Gothic Medium' if _has_font('Franklin Gothic') else 'DejaVu Sans'
    BODY_FONT = 'Calibri' if _has_font('Calibri') else 'DejaVu Sans'

    # Theme colors
    if pace_theme == 'Dark':
        bg = '#1a1a1a'; plot_bg = '#1a1a1a'
        line_color = '#4a5568'; label_color = '#718096'
        text_color = '#e2e8f0'; text_md = '#a0aec0'; grid_color = '#2e2e2e'
        spine_color = '#2d3748'; avg_line_color = '#63b3ed'
    else:
        bg = '#FAF8F2'; plot_bg = '#FAF8F2'
        line_color = '#B0A898'; label_color = '#8C8278'
        text_color = '#2D2926'; text_md = '#4A4540'; grid_color = '#E2DCCC'
        spine_color = '#D6D0C0'; avg_line_color = '#3182CE'

    # Render chart
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=bg)
    ax.set_facecolor(plot_bg)

    # Draw non-highlighted lines
    for key, pdata in filtered.groupby('_key', sort=False):
        if key in highlight_keys:
            continue
        ename = pdata['entity'].iloc[0]
        ax.plot(pdata['date_parsed'], pdata['cum_stat'],
                color=line_color, alpha=0.3, linewidth=1, zorder=1)
        last = pdata.iloc[-1]
        label = ename if pace_level == 'Team' else ename.split()[-1] if len(ename.split()) > 1 else ename
        ax.annotate(label, xy=(last['date_parsed'], last['cum_stat']),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=6, fontfamily=BODY_FONT, color=label_color,
                    va='center', alpha=0.5, zorder=1)

    # Draw highlighted lines with team colors
    legend_entries = []
    for key in highlight_keys:
        pdata = filtered[filtered['_key'] == key]
        if len(pdata) == 0:
            pdata = all_games[all_games['_key'] == key]
        if len(pdata) == 0:
            continue
        ename = pdata['entity'].iloc[0]
        team_for_color = pdata['team'].iloc[0] if pace_level == 'Player' else ename
        h_color = get_team_color(team_for_color)
        ax.plot(pdata['date_parsed'], pdata['cum_stat'],
                color=h_color, alpha=1.0, linewidth=2.5, zorder=3)
        last = pdata.iloc[-1]
        val_fmt = f"{last['cum_stat']:.2f}" if is_advanced else f"{int(last['cum_stat'])}"
        legend_label = f"{ename} — {val_fmt} {stat_choice} in {int(last['game_num'])} G"
        legend_entries.append((legend_label, h_color))

    # Division average line (dashed) — uses ALL qualified entities, not just top N
    if len(all_games) > 0:
        avg_val = all_games.groupby('_key')['cum_stat'].last().mean()
        ax.axhline(y=avg_val, color=avg_line_color, linestyle='--',
                   linewidth=1.5, alpha=0.5, zorder=2)
        avg_fmt = f"{avg_val:.2f}" if is_advanced else f"{int(avg_val)}"
        ax.text(0.50, avg_val, f'  {division} average = {avg_fmt}',
                transform=ax.get_yaxis_transform(), fontsize=9,
                fontfamily=BODY_FONT, color=avg_line_color, alpha=0.7,
                va='bottom', ha='center', zorder=5)

    # Title
    y_label = f'Running {stat_choice}' if is_advanced else f'Cumulative {stat_choice}'
    title_main = f'{stat_choice} Pace Chart'
    title_sub = f'{sport.title()} {division} - 2025-26 Season'
    fig.text(0.5, 0.97, title_main, fontsize=18, fontfamily=TITLE_FONT,
             fontweight='bold', color=text_color, ha='center', va='top')
    fig.text(0.5, 0.93, title_sub, fontsize=12, fontfamily=SUBTITLE_FONT,
             color=text_md, ha='center', va='top')

    # Legend (top-left, clean)
    if legend_entries:
        for i, (lname, lcolor) in enumerate(legend_entries):
            ax.plot([], [], color=lcolor, linewidth=2.5,
                    label=lname)
        legend = ax.legend(loc='upper left', frameon=False, fontsize=10,
                          labelcolor=text_color, handlelength=2)
        for text in legend.get_texts():
            text.set_fontfamily(SUBTITLE_FONT)
            text.set_fontweight('bold')

    # Axis styling
    ax.set_xlabel('Date', fontsize=11, fontfamily=BODY_FONT, color=text_md, labelpad=10)
    fig.autofmt_xdate(rotation=45)
    ax.set_ylabel(y_label, fontsize=11, fontfamily=BODY_FONT, color=text_md, labelpad=10)
    ax.tick_params(colors=text_md, labelsize=9)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(BODY_FONT)
    ax.grid(True, alpha=0.12, color=grid_color)
    for spine in ax.spines.values():
        spine.set_color(spine_color)

    fig.subplots_adjust(top=0.88)

    # Force whole-number y-axis ticks for counting stats
    if not is_advanced:
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Brand logo bottom-left
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
    best_hitters = get_best_hitters(hitting_pbp, league_woba, min_pa=min_pa_lc, sport=sport, rank_min_pa=rank_min_pa_lc)
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

            # Build background — square 460x460 viewBox for 1:1 Twitter/X aspect.
            # Content's natural top is ~y=10 (sidebar line) and y=24 (CF), and natural
            # bottom is ~y=365 (home plate). Using viewBox y=-5 to y=455 trims 15 units
            # of empty top padding (vs the previous -20) and adds 55 units of bottom
            # padding to reach a square 460x460 frame.
            bg_element = '<rect x="0" y="-5" width="460" height="460" fill="#1a1a1a"/>'
            bg_path = _APP_DIR / 'assets' / 'bg_pattern.jpg'
            if bg_path.exists():
                bg_data = bg_path.read_bytes()
                bg_b64 = base64.b64encode(bg_data).decode()
                bg_element = f'<image href="data:image/jpeg;base64,{bg_b64}" x="0" y="-5" width="460" height="460" preserveAspectRatio="xMidYMid slice"/>'

            svg_full = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="460" height="460" viewBox="0 -5 460 460"
     xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
{bg_element}
{inner_content}
</svg>'''
            diamond_buf = BytesIO()
            cairosvg.svg2png(bytestring=svg_full.encode('utf-8'), write_to=diamond_buf,
                             output_width=1200, output_height=1200)
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


# ── Share Graphic — 1080×1350 IG portrait social card ──────────────────────
elif view == 'Share Graphic':
    st.markdown(f'### Top Teams Graphic — {sport.title()} {division}')
    st.caption('1080×1350 IG portrait. Pick a stat, optionally upload a hero image, then download the PNG for Twitter/Instagram.')

    import base64 as _b64

    # ── Stat catalog per stat-type ────────────────────────────────────────
    # All hitting columns produced by compute_grouped_hitting. (label, suffix,
    # decimals, sort_ascending). Lower-is-better for the hitter perspective:
    # K, K%, K/BB, GDP, CS.
    HIT_STATS = [
        # Counting
        ('PA',   'PA',   0, False), ('AB',    'AB',   0, False),
        ('H',    'H',    0, False), ('1B',    '1B',   0, False),
        ('2B',   '2B',   0, False), ('3B',    '3B',   0, False),
        ('HR',   'HR',   0, False), ('TB',    'TB',   0, False),
        ('R',    'R',    0, False), ('RBI',   'RBI',  0, False),
        ('BB',   'BB',   0, False), ('HBP',   'HBP',  0, False),
        ('SF',   'SF',   0, False), ('SH',    'SH',   0, False),
        ('IBB',  'IBB',  0, False),
        ('K',    'K',    0, True),
        ('SB',   'SB',   0, False),
        ('CS',   'CS',   0, True),
        ('GDP',  'GDP',  0, True),
        # Rate
        ('BA',   'AVG',  3, False), ('OBP',   'OBP',  3, False),
        ('SLG',  'SLG',  3, False), ('OPS',   'OPS',  3, False),
        ('ISO',  'ISO',  3, False), ('BABIP', 'BABIP',3, False),
        ('wOBA', 'wOBA', 3, False),
        ('K%',   'K%',   1, True),  ('BB%',   'BB%',  1, False),
        ('K/BB', 'K/BB', 2, True),  ('R/PA',  'R/PA', 3, False),
        # Advanced
        ('wRAA', 'wRAA', 1, False), ('wRC',   'wRC',  1, False),
        ('wRC+', 'wRC+', 0, False),
    ]
    # All pitching columns produced by compute_grouped_pitching. Pitcher
    # perspective: lower-is-better for hits/runs/walks/HR allowed and ratio
    # stats (ERA, WHIP, FIP, BAA, etc.); higher-is-better for SO/K rates.
    PIT_STATS = [
        # Volume
        ('IP',   'IP',   1, False), ('App',  'App',  0, False),
        ('GS',   'GS',   0, False), ('BF',   'BF',   0, False),
        ('OAB',  'OAB',  0, False), ('SO',   'SO',   0, False),
        # Allowed
        ('H',    'H',    0, True),  ('R',    'R',    0, True),
        ('ER',   'ER',   0, True),  ('BB',   'BB',   0, True),
        ('HB',   'HB',   0, True),  ('HR',   'HR',   0, True),
        ('2B-A', '2B-A', 0, True),  ('3B-A', '3B-A', 0, True),
        ('WP',   'WP',   0, True),  ('Bk',   'Bk',   0, True),
        ('IBB',  'IBB',  0, True),
        ('SHA',  'SHA',  0, True),  ('SFA',  'SFA',  0, True),
        # Rate
        ('ERA',  'ERA',  2, True),  ('WHIP', 'WHIP', 2, True),
        ('FIP',  'FIP',  2, True),
        ('K/9',  'K/9',  2, False), ('K/7',  'K/7',  2, False),
        ('BB/9', 'BB/9', 2, True),
        ('K/BB', 'K/BB', 2, False),
        ('K%',   'K%',   1, False), ('BB%',  'BB%',  1, True),
        ('K-BB%','K-BB%', 1, False),
        ('BAA',  'BAA',  3, True),  ('BABIP','BABIP',3, True),
        ('OBP Against', 'OBPa', 3, True),
        ('SLG Against', 'SLGa', 3, True),
        ('OPS Against', 'OPSa', 3, True),
        ('GmSc', 'GmSc', 1, False),
    ]
    FLD_STATS = [
        ('FPCT',   'FPCT', 3, False), ('PO',     'PO',   0, False),
        ('A',      'A',    0, False), ('E',      'E',    0, True),
        ('DP',     'DP',   0, False), ('TC',     'TC',   0, False),
    ]
    catalog = {'hitting': HIT_STATS, 'pitching': PIT_STATS, 'fielding': FLD_STATS}[sg_stat_type]
    stat_labels = [c[0] for c in catalog]

    # ── Controls ─────────────────────────────────────────────────────────
    sg_top = st.columns([1, 1, 1, 1])
    with sg_top[0]:
        sg_stat = st.selectbox('Rank by', stat_labels, key='sg_stat')
    sg_meta = next(c for c in catalog if c[0] == sg_stat)
    _, default_suffix, default_decimals, sort_asc = sg_meta
    with sg_top[1]:
        sg_count = st.slider('Row count', 1, 10, 5, key='sg_count')
    with sg_top[2]:
        sg_min = st.number_input('Min PA / BF / TC', value=50, min_value=1, step=10, key='sg_min',
                                  help='Qualifying threshold — PA for hitting, BF for pitching, TC for fielding.')
    with sg_top[3]:
        sg_show_names = st.toggle('Show team names', value=False, key='sg_names')

    sg_text1, sg_text2 = st.columns(2)
    with sg_text1:
        sg_eyebrow = st.text_input('Eyebrow', value='SEASON-TO-DATE', key='sg_eyebrow').upper()
        sg_headline = st.text_input('Headline (use \\n for line break)',
                                     value=f'TOP {sg_count} {division} TEAMS\\nBY {sg_stat.upper()}',
                                     key='sg_headline').replace('\\n', '\n').upper()
    with sg_text2:
        sg_suffix = st.text_input('Stat suffix', value=default_suffix, key='sg_suffix').upper()
        sg_decimals = st.number_input('Stat decimals', value=int(default_decimals),
                                       min_value=0, max_value=4, step=1, key='sg_decimals')

    sg_text3, sg_text4 = st.columns(2)
    with sg_text3:
        sg_cta_label = st.text_input('CTA label', value='SIGN UP TO SEE THE LATEST UPDATED STATS',
                                      key='sg_cta_label').upper()
    with sg_text4:
        sg_cta_url = st.text_input('CTA URL', value='WWW.64ANALYTICS.COM', key='sg_cta_url').upper()

    sg_rail = st.text_input('Side-rail tagline',
                             value='64 ANALYTICS IS THE INDUSTRY LEADER FOR DIAMOND SPORTS ANALYTICS',
                             key='sg_rail').upper()

    sg_hero = st.file_uploader('Hero image (optional, right column — 4:5 portrait recommended)',
                                type=['png', 'jpg', 'jpeg'], key='sg_hero')

    # ── Compute top-N rows ───────────────────────────────────────────────
    pbp_rank = pbp.copy()
    if group_by == 'Team':
        rank_col = 'teamName'
    else:
        rank_col = 'playerId' if 'playerId' in pbp_rank.columns else 'playerName'

    if sg_stat_type == 'hitting':
        league_stats = compute_hitting_stats(pbp_rank)
        league_woba = league_stats['wOBA']
        league_r_pa = league_stats.get('R/PA', 0)
        df_top = compute_grouped_hitting(pbp_rank, rank_col, league_woba,
                                          league_r_pa=league_r_pa, min_pa=sg_min)
    elif sg_stat_type == 'pitching':
        df_top = compute_grouped_pitching(pbp_rank, rank_col, min_bf=sg_min)
    else:
        df_top = compute_grouped_fielding(pbp_rank, rank_col)
        if 'TC' in df_top.columns:
            df_top = df_top[df_top['TC'] >= sg_min]

    if sg_stat not in df_top.columns:
        st.error(f"Stat '{sg_stat}' not in computed columns.")
        st.stop()

    df_top[sg_stat] = pd.to_numeric(df_top[sg_stat], errors='coerce')
    df_top = df_top.dropna(subset=[sg_stat]).sort_values(sg_stat, ascending=sort_asc).head(sg_count).reset_index(drop=True)

    if df_top.empty:
        st.warning('No qualifying rows for the chosen stat / threshold.')
        st.stop()

    # Resolve display name for each row + team metadata for color/logo
    teams_meta = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False).fillna('')
    sport_label = 'Baseball' if sport == 'baseball' else 'Softball'
    sport_teams_lookup = teams_meta[teams_meta['sport'] == sport_label]

    def _team_lookup(tname):
        # Match shortest team-name prefix (memory rule)
        cands = sport_teams_lookup[sport_teams_lookup['name'] == tname]
        if cands.empty:
            cands = sport_teams_lookup[sport_teams_lookup['name'].apply(
                lambda n: isinstance(tname, str) and tname.startswith(n))]
            if not cands.empty:
                cands = cands.sort_values('name', key=lambda s: s.str.len()).head(1)
        if cands.empty:
            return None
        return cands.iloc[0]

    # Fallback palette when logo color extraction fails.
    PILL_PALETTE = ['#1d4ed8', '#2a2a2f', '#d72638', '#3a3f47', '#1f3a8a',
                    '#0f5132', '#7c2d12', '#312e81', '#0e7490', '#7f1d1d']

    @st.cache_data(show_spinner=False)
    def _team_dominant_color(cb_id):
        """Extract the team's primary brand color from its logo PNG. Skips
        near-white, near-black, and near-grayscale pixels (silvers, etc.) so
        the chromatic primary wins. Returns hex like '#8C1D40'."""
        if cb_id is None:
            return None
        try:
            from PIL import Image
            from collections import Counter
        except Exception:
            return None
        p = LOGO_DIR / f'{int(cb_id)}.png'
        if not p.exists():
            return None
        try:
            img = Image.open(p).convert('RGBA').resize((96, 96), Image.LANCZOS)
            cleaned = []
            for r, g, b, a in img.getdata():
                if a < 220:
                    continue
                if r > 235 and g > 235 and b > 235:        # near white
                    continue
                if r < 18 and g < 18 and b < 18:           # near black
                    continue
                if max(r, g, b) - min(r, g, b) < 20:       # near gray (silver, etc.)
                    continue
                cleaned.append((r // 16 * 16, g // 16 * 16, b // 16 * 16))
            if not cleaned:
                return None
            top = Counter(cleaned).most_common(1)[0][0]
            return f'#{top[0]:02x}{top[1]:02x}{top[2]:02x}'
        except Exception:
            return None

    # Bridge from PBP playerId (NCAA season-id) to chart-builder cb_id +
    # team_id, used to look up player headshots and the player's team color.
    @st.cache_data(show_spinner=False)
    def _ncaa_pid_to_cb_bridge():
        """{ncaa_pid: {cb_id, team_id}} for the latest roster year."""
        try:
            rosters = pd.read_csv(DATA_DIR / 'rosters.csv', low_memory=False)
        except Exception:
            return {}
        if 'Year' in rosters.columns:
            rosters['Year'] = pd.to_numeric(rosters['Year'], errors='coerce')
            current_year = int(rosters['Year'].max())
            r = rosters[rosters['Year'] == current_year]
        else:
            r = rosters
        cols_needed = [c for c in ('player_id', 'player_ncaa_season_id', 'team_id') if c in r.columns]
        if 'player_ncaa_season_id' not in cols_needed or 'player_id' not in cols_needed:
            return {}
        r = r[cols_needed].copy()
        r['ncaa_pid'] = pd.to_numeric(r['player_ncaa_season_id'], errors='coerce').astype('Int64')
        r = r.dropna(subset=['ncaa_pid', 'player_id'])
        bridge = {}
        for _, row in r.iterrows():
            try:
                bridge[int(row['ncaa_pid'])] = {
                    'cb_id': int(row['player_id']),
                    'team_id': int(row['team_id']) if 'team_id' in row and pd.notna(row['team_id']) else None,
                }
            except Exception:
                continue
        return bridge

    HEADSHOT_DIR = _APP_DIR / 'assets' / 'player_headshots'
    pid_bridge = _ncaa_pid_to_cb_bridge() if group_by == 'Player' else {}

    rows_data = []
    for idx, row in df_top.iterrows():
        if group_by == 'Team':
            tname = row.get('teamName', '') or row.get(rank_col, '')
            display = tname
            team_row = _team_lookup(tname)
            ncaa_pid = None
        else:
            display = row.get('playerName', '') or row.get(rank_col, '')
            tname = row.get('teamName', '') or row.get('school', '') or ''
            team_row = _team_lookup(tname) if tname else None
            ncaa_pid = row.get('playerId', None)
            try:
                ncaa_pid = int(ncaa_pid) if ncaa_pid is not None and not pd.isna(ncaa_pid) else None
            except Exception:
                ncaa_pid = None

        # Resolve cb player + team via bridge for player rows
        cb_player_id = None
        if ncaa_pid is not None and ncaa_pid in pid_bridge:
            cb_player_id = pid_bridge[ncaa_pid].get('cb_id')
            # If team_row didn't resolve via name match, try by team_id from bridge
            br_team_id = pid_bridge[ncaa_pid].get('team_id')
            if (team_row is None or (team_row is not None and not team_row.get('id'))) and br_team_id:
                cands = sport_teams_lookup[sport_teams_lookup['id'] == br_team_id]
                if not cands.empty:
                    team_row = cands.iloc[0]

        team_cb_id = None
        if team_row is not None:
            try:
                team_cb_id = int(team_row['id']) if team_row.get('id') else None
            except Exception:
                team_cb_id = None

        # Logo: team logo (fallback for placeholder when no headshot)
        logo_b64 = None
        if team_cb_id is not None:
            try:
                logo_path = LOGO_DIR / f'{team_cb_id}.png'
                if logo_path.exists():
                    logo_b64 = _b64.b64encode(logo_path.read_bytes()).decode('ascii')
            except Exception:
                pass

        # Player headshot for Player mode
        headshot_b64 = None
        if cb_player_id is not None:
            try:
                hs_path = HEADSHOT_DIR / f'{cb_player_id}.png'
                if hs_path.exists():
                    headshot_b64 = _b64.b64encode(hs_path.read_bytes()).decode('ascii')
            except Exception:
                pass

        # Pill color — extracted from team logo PNG (palette fallback)
        pill_color = _team_dominant_color(team_cb_id) if team_cb_id is not None else None
        if not pill_color:
            try:
                if team_row is not None and team_row.get('team_id_ncaa') not in (None, '', 'nan'):
                    seed_idx = int(float(team_row['team_id_ncaa'])) % len(PILL_PALETTE)
                else:
                    seed_idx = idx % len(PILL_PALETTE)
            except Exception:
                seed_idx = idx % len(PILL_PALETTE)
            pill_color = PILL_PALETTE[seed_idx]

        # Team abbreviation (first letters of words, or first 3 chars)
        twords = (tname or '').split()
        if len(twords) >= 2:
            tabbr = ''.join(w[0] for w in twords[:3]).upper()
        elif tname:
            tabbr = tname[:3].upper()
        else:
            tabbr = (display or '')[:3].upper()
        # Player initials (e.g., "Landon Hairston" -> "LH")
        pwords = (display or '').split()
        if len(pwords) >= 2:
            pabbr = (pwords[0][0] + pwords[-1][0]).upper()
        elif pwords:
            pabbr = pwords[0][:2].upper()
        else:
            pabbr = '?'
        rows_data.append({
            'rank': idx + 1, 'name': display, 'team': tname,
            'tabbr': tabbr, 'pabbr': pabbr, 'stat': float(row[sg_stat]),
            'color': pill_color, 'logo_b64': logo_b64,
            'headshot_b64': headshot_b64, 'cb_player_id': cb_player_id,
            'team_cb_id': team_cb_id,
        })

    # ── Player-photo upload (Player mode, surfaced before the render so it's
    # actually discoverable). Same store the spray-chart page uses: a PNG at
    # assets/player_headshots/{cb_id}.png replaces the initials-in-a-circle
    # placeholder for that player on the next render.
    if group_by == 'Player':
        any_missing = any(rd.get('headshot_b64') is None and rd.get('cb_player_id') is not None
                          for rd in rows_data)
        with st.expander('Add player photos (replaces initials in the pill)',
                          expanded=any_missing):
            st.caption(
                'Upload a square PNG / JPG for any player; saved as '
                '`assets/player_headshots/{cb_id}.png`. Same store the Spray '
                'Charts page uses, so a photo uploaded once shows up on both.'
            )
            HEADSHOT_DIR.mkdir(parents=True, exist_ok=True)
            for rd in rows_data:
                cb_id = rd.get('cb_player_id')
                pname = rd.get('name', '?')
                if cb_id is None:
                    st.caption(f"`{pname}` — no chart-builder id (rosters bridge missed); upload disabled.")
                    continue
                target = HEADSHOT_DIR / f'{cb_id}.png'
                present = '✓ photo on file' if target.exists() else '— no photo yet'
                up = st.file_uploader(
                    f"{pname}  (cb_id {cb_id})  {present}",
                    type=['png', 'jpg', 'jpeg', 'webp'],
                    key=f'sg_photo_{cb_id}',
                )
                if up is not None:
                    new_bytes = up.getvalue()
                    existing = target.read_bytes() if target.exists() else None
                    if existing != new_bytes:
                        try:
                            from PIL import Image as _PILImage
                            import io as _io
                            img = _PILImage.open(_io.BytesIO(new_bytes)).convert('RGBA')
                            img.save(target, 'PNG')
                            # Refresh in-memory so the SVG below picks it up
                            # without a second rerun.
                            rd['headshot_b64'] = _b64.b64encode(target.read_bytes()).decode('ascii')
                            st.success(f"Saved `{target.name}` — re-rendering below.")
                        except Exception as e:
                            st.error(f'Could not save image: {e}')

    # ── Build SVG ─────────────────────────────────────────────────────────
    def _xe(s):
        if s is None: return ''
        return (str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))

    VB_W, VB_H = 1080, 1350
    BG = '#0a0a0c'
    BG_RAIL = '#15151a'
    RED = '#d72638'
    WHITE = '#ffffff'

    # Layout per row count (mirrors HTML CSS scaling rules)
    row_layout = {
        1:  (88, 18, 78, 64), 2:  (88, 18, 78, 64), 3:  (88, 18, 78, 64),
        4:  (88, 18, 78, 64), 5:  (88, 18, 78, 64), 6:  (78, 14, 68, 64),
        7:  (70, 12, 60, 58), 8:  (64, 11, 54, 54), 9:  (58, 10, 48, 50),
        10: (54, 8,  44, 46),
    }
    ROW_H, ROW_GAP, RANK_SIZE, HEAD_SIZE = row_layout[max(1, min(10, sg_count))]

    # Hero image as data URL — manual upload wins; otherwise for Player mode,
    # auto-fill from the top-1 player's headshot if available.
    hero_data_url = None
    if sg_hero is not None:
        try:
            mime = sg_hero.type or 'image/png'
            hero_b64 = _b64.b64encode(sg_hero.getvalue()).decode('ascii')
            hero_data_url = f'data:{mime};base64,{hero_b64}'
        except Exception:
            hero_data_url = None
    if hero_data_url is None and group_by == 'Player' and rows_data:
        top_hs = rows_data[0].get('headshot_b64')
        if top_hs:
            hero_data_url = f'data:image/png;base64,{top_hs}'

    # 64 emblem from chart-builder/assets if it exists
    EMBLEM_PATHS = [_APP_DIR / 'assets' / '64-emblem-white.png',
                    _APP_DIR / 'assets' / 'logo-64a-mono-white.png',
                    _APP_DIR / 'assets' / 'logo-64a-wide.png']
    emblem_b64 = None
    for ep in EMBLEM_PATHS:
        if ep.exists():
            try:
                emblem_b64 = _b64.b64encode(ep.read_bytes()).decode('ascii')
                break
            except Exception:
                pass

    parts = [
        f'<svg viewBox="0 0 {VB_W} {VB_H}" width="{VB_W}" height="{VB_H}" '
        f'xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">',
        '<defs>',
        # CDATA-wrap the @import so the unescaped `&` query separators don't
        # break the XML parser cairosvg uses.
        '<style><![CDATA['
        '@import url("https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Barlow+Condensed:ital,wght@0,500;0,600;0,700;1,600;1,700;1,800&family=JetBrains+Mono:wght@500;600&display=swap");'
        '.os{font-family:Oswald,sans-serif}.bc{font-family:"Barlow Condensed",sans-serif}.mn{font-family:"JetBrains Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}'
        ']]></style>',
        # turbulence filter for distortion bleed
        '<filter id="distort" x="-10%" y="-10%" width="120%" height="120%">'
        '<feTurbulence type="fractalNoise" baseFrequency="0.012 0.022" numOctaves="2" seed="7" result="noise"/>'
        '<feDisplacementMap in="SourceGraphic" in2="noise" scale="34" xChannelSelector="R" yChannelSelector="G"/>'
        '</filter>',
        # left-bleed mask for the hero image distortion
        '<mask id="bleed-mask" maskUnits="userSpaceOnUse">'
        f'<linearGradient id="bg-grad" x1="0" y1="0" x2="{VB_W}" y2="0" gradientUnits="userSpaceOnUse">'
        '<stop offset="0%" stop-color="black"/>'
        '<stop offset="18%" stop-color="rgb(140,140,140)"/>'
        '<stop offset="32%" stop-color="rgb(204,204,204)"/>'
        '<stop offset="48%" stop-color="black"/>'
        '<stop offset="100%" stop-color="black"/>'
        '</linearGradient>'
        f'<rect x="0" y="0" width="{VB_W}" height="{VB_H}" fill="url(#bg-grad)"/>'
        '</mask>',
        '</defs>',
        # background
        f'<rect x="0" y="0" width="{VB_W}" height="{VB_H}" fill="{BG}"/>',
    ]

    # Distortion bleed layer (only when hero image present)
    if hero_data_url:
        # Image stretched 2x and shifted left so the "bleed" covers the rows
        parts.append(
            f'<g mask="url(#bleed-mask)" opacity="0.55" style="mix-blend-mode:screen">'
            f'<image href="{hero_data_url}" xlink:href="{hero_data_url}" '
            f'x="{-VB_W * 0.5}" y="{-VB_H * 0.5}" width="{VB_W * 2}" height="{VB_H * 2}" '
            f'preserveAspectRatio="xMidYMid slice" filter="url(#distort)"/>'
            f'</g>'
        )

    # Dark scrim over left half so text stays readable
    parts.append(
        '<defs><linearGradient id="scrim" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0%" stop-color="#0a0a0c" stop-opacity="0.96"/>'
        '<stop offset="45%" stop-color="#0a0a0c" stop-opacity="0.85"/>'
        '<stop offset="80%" stop-color="#0a0a0c" stop-opacity="0.55"/>'
        '<stop offset="100%" stop-color="#0a0a0c" stop-opacity="0"/>'
        '</linearGradient></defs>'
        f'<rect x="0" y="0" width="{int(VB_W * 0.6)}" height="{VB_H}" fill="url(#scrim)"/>'
    )

    # Right column hero image
    if hero_data_url:
        parts.append(
            f'<image href="{hero_data_url}" xlink:href="{hero_data_url}" '
            f'x="{VB_W // 2}" y="0" width="{VB_W // 2}" height="{VB_H}" '
            f'preserveAspectRatio="xMidYMid slice"/>'
        )
    else:
        # placeholder pattern
        parts.append('<defs><pattern id="empty-pat" patternUnits="userSpaceOnUse" '
                     'width="48" height="48" patternTransform="rotate(135)">'
                     '<rect width="48" height="48" fill="#18181c"/>'
                     '<rect x="24" width="24" height="48" fill="#1e1e23"/>'
                     '</pattern></defs>')
        parts.append(f'<rect x="{VB_W // 2}" y="0" width="{VB_W // 2}" height="{VB_H}" fill="url(#empty-pat)"/>')

    # Side rails — left + right
    parts.append(f'<rect x="0" y="0" width="38" height="{VB_H}" fill="{BG_RAIL}"/>')
    parts.append(f'<rect x="38" y="0" width="2" height="{VB_H}" fill="{RED}"/>')
    parts.append(f'<rect x="{VB_W - 38}" y="0" width="38" height="{VB_H}" fill="{BG_RAIL}"/>')
    parts.append(f'<rect x="{VB_W - 40}" y="0" width="2" height="{VB_H}" fill="{RED}"/>')

    # Rail text — vertical, both sides
    rail_clean = _xe(sg_rail)
    # Left rail: read top-to-bottom along the rail. Use rotation -90 about center.
    parts.append(
        f'<g transform="translate(19 {VB_H // 2}) rotate(-90)">'
        f'<text x="0" y="5" text-anchor="middle" class="os" font-size="13" font-weight="600" '
        f'fill="{RED}" letter-spacing="4.2">{rail_clean}</text>'
        f'</g>'
    )
    parts.append(
        f'<g transform="translate({VB_W - 19} {VB_H // 2}) rotate(90)">'
        f'<text x="0" y="5" text-anchor="middle" class="os" font-size="13" font-weight="600" '
        f'fill="{RED}" letter-spacing="4.2">{rail_clean}</text>'
        f'</g>'
    )

    # ── LEFT COLUMN — content ────────────────────────────────────────────
    # Pulled tighter to the left rail than the original design so the pills
    # have more horizontal room without bleeding into the hero column.
    LEFT_PAD_X = 54
    content_x = LEFT_PAD_X
    content_top_y = 80

    # Eyebrow
    parts.append(
        f'<text x="{content_x}" y="{content_top_y + 14}" class="os" font-size="22" '
        f'font-weight="600" fill="{WHITE}" letter-spacing="3.96">{_xe(sg_eyebrow)}</text>'
    )

    # Headline (multi-line; italic). Scale font down so the longest line fits
    # inside the left content column — without this, a long stat name like
    # "TOP 5 D1 TEAMS BY wRC+" can spill over the seam into the hero image.
    head_y = content_top_y + 14 + 28
    head_lines = sg_headline.split('\n')
    headline_max_w = (VB_W / 2) - content_x - 16
    longest_line = max((len(line) for line in head_lines), default=1)
    # Barlow Condensed Italic 800: ~0.46 of font-size per char + letter-spacing.
    est_w = longest_line * HEAD_SIZE * 0.46 + max(0, longest_line - 1) * 0.32
    if est_w > headline_max_w and longest_line > 0:
        HEAD_SIZE = max(28, int(HEAD_SIZE * headline_max_w / est_w))
    for i, line in enumerate(head_lines):
        parts.append(
            f'<text x="{content_x}" y="{head_y + (i + 1) * HEAD_SIZE * 0.95:.0f}" '
            f'class="bc" font-size="{HEAD_SIZE}" font-weight="800" font-style="italic" '
            f'fill="{WHITE}" letter-spacing="0.32">{_xe(line)}</text>'
        )

    # Rows — vertically center the block in the available space between
    # headline-bottom and footer-top. Scale ROW_H so the block fills ~78% of
    # available height, capped to a sane range so type stays balanced.
    headline_bottom_y = head_y + len(head_lines) * HEAD_SIZE * 0.95 + 24
    footer_top_y = VB_H - 180
    available_h = footer_top_y - headline_bottom_y
    _gap_ratio = (ROW_GAP / ROW_H) if ROW_H else 0.205
    _rank_ratio = (RANK_SIZE / ROW_H) if ROW_H else 0.886
    _target_block = available_h * 0.82
    _new_row_h = _target_block / max(1.0, sg_count + (sg_count - 1) * _gap_ratio)
    _new_row_h = max(54, min(118, _new_row_h))
    ROW_H = _new_row_h
    ROW_GAP = ROW_H * _gap_ratio
    RANK_SIZE = ROW_H * _rank_ratio

    block_h = sg_count * ROW_H + max(0, sg_count - 1) * ROW_GAP
    rows_top_y = headline_bottom_y + (available_h - block_h) / 2

    # Pill geometry — pills extend almost to the seam between the left
    # content column and the right hero column, with a small gap so the hero
    # image stays visible.
    PILL_RIGHT_EDGE = VB_W * 0.5 - 12

    for idx, rd in enumerate(rows_data):
        row_y = rows_top_y + idx * (ROW_H + ROW_GAP)
        # Rank number
        rank_x = content_x + 2
        parts.append(
            f'<text x="{rank_x}" y="{row_y + ROW_H * 0.78:.0f}" class="bc" '
            f'font-size="{RANK_SIZE:.0f}" font-weight="800" font-style="italic" '
            f'fill="{WHITE}" letter-spacing="-0.15">{rd["rank"]}</text>'
        )
        # Pill — left edge after rank, right edge at PILL_RIGHT_EDGE
        pill_x = content_x + RANK_SIZE * 0.85 + 14
        pill_w = PILL_RIGHT_EDGE - pill_x
        pill_h = ROW_H
        pill_r = pill_h / 2
        parts.append(
            f'<rect x="{pill_x:.1f}" y="{row_y:.1f}" width="{pill_w:.1f}" height="{pill_h}" '
            f'rx="{pill_r}" ry="{pill_r}" fill="{rd["color"]}"/>'
        )

        # Logo position. Player mode: use headshot if available, else show
        # the player's initials in a circle (NEVER fall back to team logo —
        # that's confusing when ranking players). Team mode: team logo if
        # available, else team abbreviation.
        logo_size = pill_h * 0.8
        logo_cx = pill_x + pill_h / 2
        logo_cy = row_y + pill_h / 2
        # Light stroke ring around the logo / headshot — keeps dark logos
        # (e.g. Nevada navy) readable on dark pills.
        ring_r = logo_size / 2 + 1
        ring_attr = (f'<circle cx="{logo_cx:.1f}" cy="{logo_cy:.1f}" r="{ring_r:.1f}" '
                     f'fill="none" stroke="rgba(255,255,255,0.65)" stroke-width="2"/>')
        if group_by == 'Player':
            if rd.get('headshot_b64'):
                clip_id = f'sg_clip_{idx}'
                parts.append(
                    f'<defs><clipPath id="{clip_id}">'
                    f'<circle cx="{logo_cx:.1f}" cy="{logo_cy:.1f}" r="{logo_size / 2:.1f}"/>'
                    f'</clipPath></defs>'
                    f'<circle cx="{logo_cx:.1f}" cy="{logo_cy:.1f}" r="{logo_size / 2:.1f}" '
                    f'fill="rgba(255,255,255,0.12)"/>'
                    f'<image href="data:image/png;base64,{rd["headshot_b64"]}" '
                    f'xlink:href="data:image/png;base64,{rd["headshot_b64"]}" '
                    f'x="{logo_cx - logo_size / 2:.1f}" y="{logo_cy - logo_size / 2:.1f}" '
                    f'width="{logo_size:.1f}" height="{logo_size:.1f}" '
                    f'preserveAspectRatio="xMidYMid slice" clip-path="url(#{clip_id})"/>'
                    + ring_attr
                )
            else:
                parts.append(
                    f'<circle cx="{logo_cx:.1f}" cy="{logo_cy:.1f}" r="{logo_size / 2:.1f}" '
                    f'fill="rgba(255,255,255,0.18)"/>'
                    f'<text x="{logo_cx:.1f}" y="{logo_cy + 7:.1f}" class="bc" '
                    f'font-size="{int(pill_h * 0.32)}" font-weight="800" font-style="italic" '
                    f'fill="{WHITE}" text-anchor="middle" letter-spacing="0.6">{_xe(rd["pabbr"])}</text>'
                    + ring_attr
                )
        else:
            if rd['logo_b64']:
                # Eggshell disc + clipPath so wide logos (e.g. Texas Tech's
                # winged T) get cropped at the circle edge instead of bleeding
                # over the team-color pill.
                clip_id = f'sg_logoclip_{idx}'
                parts.append(
                    f'<defs><clipPath id="{clip_id}">'
                    f'<circle cx="{logo_cx:.1f}" cy="{logo_cy:.1f}" r="{logo_size / 2:.1f}"/>'
                    f'</clipPath></defs>'
                    f'<circle cx="{logo_cx:.1f}" cy="{logo_cy:.1f}" r="{logo_size / 2:.1f}" '
                    f'fill="{EGGSHELL}"/>'
                    f'<image href="data:image/png;base64,{rd["logo_b64"]}" '
                    f'xlink:href="data:image/png;base64,{rd["logo_b64"]}" '
                    f'x="{logo_cx - logo_size / 2:.1f}" y="{logo_cy - logo_size / 2:.1f}" '
                    f'width="{logo_size:.1f}" height="{logo_size:.1f}" '
                    f'preserveAspectRatio="xMidYMid meet" clip-path="url(#{clip_id})"/>'
                    + ring_attr
                )
            else:
                parts.append(
                    f'<circle cx="{logo_cx:.1f}" cy="{logo_cy:.1f}" r="{logo_size / 2:.1f}" '
                    f'fill="rgba(255,255,255,0.12)"/>'
                    f'<text x="{logo_cx:.1f}" y="{logo_cy + 6:.1f}" class="bc" '
                    f'font-size="{int(pill_h * 0.26)}" font-weight="800" font-style="italic" '
                    f'fill="{WHITE}" text-anchor="middle" letter-spacing="0.4">{_xe(rd["tabbr"])}</text>'
                    + ring_attr
                )
        # Stat + name — right-anchored to the pill's right edge so they hug
        # that side regardless of pill height. Avoids the dead-space-in-the-
        # middle look when the pill count is high and pill_h is short.
        text_right = pill_x + pill_w - max(14, pill_h * 0.22)
        stat_str = f'{rd["stat"]:.{int(sg_decimals)}f}'
        stat_size = max(20, min(34, int(pill_h * 0.38)))
        if sg_show_names:
            stat_y = logo_cy - 2
        else:
            stat_y = logo_cy + 8
        parts.append(
            f'<text x="{text_right:.1f}" y="{stat_y:.1f}" class="bc" '
            f'font-size="{stat_size}" font-weight="800" font-style="italic" '
            f'fill="{WHITE}" letter-spacing="0.5" text-anchor="end">{stat_str}{(" " + _xe(sg_suffix)) if sg_suffix else ""}</text>'
        )
        # Team / player name as a bottom-aligned caption beneath the stat
        if sg_show_names:
            name_size = max(13, min(20, int(pill_h * 0.22)))
            name_y = row_y + pill_h - max(8, pill_h * 0.12)
            parts.append(
                f'<text x="{text_right:.1f}" y="{name_y:.1f}" class="bc" '
                f'font-size="{name_size}" font-weight="600" font-style="italic" '
                f'fill="rgba(255,255,255,0.82)" letter-spacing="0.5" text-anchor="end">{_xe(rd["name"]).upper()}</text>'
            )

    # ── FOOTER ────────────────────────────────────────────────────────────
    foot_top = VB_H - 180
    # 64 emblem
    if emblem_b64:
        parts.append(
            f'<image href="data:image/png;base64,{emblem_b64}" '
            f'xlink:href="data:image/png;base64,{emblem_b64}" '
            f'x="{content_x}" y="{foot_top}" width="70" height="70" '
            f'preserveAspectRatio="xMidYMid meet"/>'
        )
    else:
        parts.append(
            f'<circle cx="{content_x + 35}" cy="{foot_top + 35}" r="33" fill="{RED}"/>'
            f'<text x="{content_x + 35}" y="{foot_top + 47}" class="bc" '
            f'font-size="32" font-weight="800" font-style="italic" '
            f'fill="{WHITE}" text-anchor="middle">64</text>'
        )
    # PRESENTED BY label
    parts.append(
        f'<text x="{content_x + 84}" y="{foot_top + 22}" class="os" font-size="11" '
        f'font-weight="600" fill="{WHITE}" letter-spacing="2.42" '
        f'opacity="0.85">PRESENTED BY</text>'
    )
    parts.append(
        f'<text x="{content_x + 84}" y="{foot_top + 50}" class="bc" font-size="28" '
        f'font-weight="800" font-style="italic" fill="{WHITE}" letter-spacing="0.5">'
        f'64 ANALYTICS</text>'
    )
    # Divider
    parts.append(
        f'<line x1="{content_x}" y1="{foot_top + 88}" x2="{VB_W // 2 - 28}" y2="{foot_top + 88}" '
        f'stroke="rgba(255,255,255,0.35)" stroke-width="1"/>'
    )
    # CTA label
    parts.append(
        f'<text x="{content_x}" y="{foot_top + 116}" class="os" font-size="16" '
        f'font-weight="600" fill="{WHITE}" letter-spacing="2.88">{_xe(sg_cta_label)}</text>'
    )
    # CTA URL (red)
    parts.append(
        f'<text x="{content_x}" y="{foot_top + 152}" class="bc" font-size="28" '
        f'font-weight="800" font-style="italic" fill="{RED}" letter-spacing="0.5">'
        f'{_xe(sg_cta_url)}</text>'
    )

    # Corner mark on right column (over hero image area)
    corner_size = 92
    corner_x = VB_W - 38 - 64 - corner_size
    corner_y = 36
    if emblem_b64:
        parts.append(
            f'<image href="data:image/png;base64,{emblem_b64}" '
            f'xlink:href="data:image/png;base64,{emblem_b64}" '
            f'x="{corner_x}" y="{corner_y}" width="{corner_size}" height="{corner_size}" '
            f'preserveAspectRatio="xMidYMid meet"/>'
        )
    else:
        parts.append(
            f'<circle cx="{corner_x + corner_size / 2:.0f}" cy="{corner_y + corner_size / 2:.0f}" '
            f'r="{corner_size / 2}" fill="{RED}"/>'
            f'<text x="{corner_x + corner_size / 2:.0f}" y="{corner_y + corner_size / 2 + 12:.0f}" '
            f'class="bc" font-size="36" font-weight="800" font-style="italic" '
            f'fill="{WHITE}" text-anchor="middle" letter-spacing="-0.5">64</text>'
        )

    parts.append('</svg>')
    sg_svg = ''.join(parts)

    # Display
    sg_display = sg_svg.replace(
        '<svg ',
        '<svg style="width:100%;max-width:1080px;height:auto;display:block;'
        'margin:0 auto;border-radius:8px;box-shadow:0 12px 36px rgba(0,0,0,.5);" ', 1,
    )
    st.markdown(sg_display, unsafe_allow_html=True)

    # PNG download
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(bytestring=sg_svg.encode('utf-8'), output_width=2160)
        safe_stat = ''.join(c if c.isalnum() else '_' for c in str(sg_stat))[:20]
        fname = f'top_{sg_count}_{group_by.lower()}_{safe_stat}_{sport}_{division}.png'
        st.download_button('Download PNG (2160w)', data=png_bytes, file_name=fname,
                           mime='image/png', use_container_width=False)
    except Exception as e:
        st.caption(f'PNG export unavailable in this environment ({type(e).__name__}: {str(e)[:80]}).')

    st.markdown('---')
    st.caption('1080×1350 IG portrait. Pill colors are extracted from each team\'s logo PNG. '
               'Player headshots: drop a PNG named `{cb_id}.png` into `assets/player_headshots/`, '
               'or use the upload section above. Hero image (right column) stays clear; if no '
               'manual hero is uploaded in Player mode, the top-1 player\'s headshot fills it.')

