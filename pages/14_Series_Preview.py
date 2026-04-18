"""
Weekend Series Preview — head-to-head matchup breakdown for upcoming series.

Combines: team rankings (64A/RPI/Massey/DSR), spider chart comparison,
game-by-game predicted WP with starter matchups, bullpen depth, and
"who's hot" last-14-day performers.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta

from pages._win_prob_model import (
    build_team_profiles, adjusted_team_pct, pre_game_wp,
    blend_with_static, build_rank_pct_map, TEAM_RANK_BLEND,
)

_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _APP_DIR / 'data'
PBP_DIR = _APP_DIR / 'pbp_data'

DIVISION_LABELS = {'D1': 'D-I', 'D2': 'D-II', 'D3': 'D-III'}

# 64A brand colors
BG_COLOR = '#FAF8F2'
CARD_RED = '#C41230'
NAVY = '#29335c'
TEXT_COLOR = '#2D2926'

st.set_page_config(page_title='Series Preview', layout='wide')
st.title('Weekend Series Preview')


# ── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_teams_data():
    t = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False).fillna('')
    c = pd.read_csv(DATA_DIR / 'conferences.csv', low_memory=False).fillna('')
    t['id'] = pd.to_numeric(t['id'], errors='coerce').fillna(0).astype(int)
    c_map = dict(zip(c['id'], c['name']))
    c_div = dict(zip(c['id'], c['division']))
    t['conference_name'] = t['conference_id'].map(c_map)
    t['division'] = t['conference_id'].map(c_div)
    return t, c


@st.cache_data
def load_team_ranks(sport_label, div_label):
    tr = pd.read_csv(DATA_DIR / 'team_rank.csv', low_memory=False)
    tr26 = tr[tr['year'] == 2026].copy()
    tr26['rank_64a'] = pd.to_numeric(tr26['integer_64_rank_total'], errors='coerce')
    return tr26


@st.cache_data
def load_external_ranks(sport):
    sport_l = sport.lower()
    ranks = {}
    # RPI
    rpi_f = DATA_DIR / f'{sport_l}_rpi_D1.csv'
    if rpi_f.exists():
        r = pd.read_csv(rpi_f, low_memory=False)
        ranks['rpi'] = dict(zip(r['teamName'], pd.to_numeric(r['rank'], errors='coerce')))
    # Name map
    nm_f = DATA_DIR / 'rankings' / 'name_map.csv'
    ext_map = {}
    if nm_f.exists():
        nm = pd.read_csv(nm_f)
        ext_map = dict(zip(nm['external_name'], nm['our_name']))
    # Massey
    m_f = DATA_DIR / 'rankings' / f'massey_{sport_l}.csv'
    if m_f.exists():
        m = pd.read_csv(m_f, low_memory=False)
        ranks['massey'] = {ext_map.get(t, t): r for t, r in zip(m['team'], m['rank'])}
    # DSR
    d_f = DATA_DIR / 'rankings' / f'dsr_{sport_l}.csv'
    if d_f.exists():
        d = pd.read_csv(d_f, low_memory=False)
        ranks['dsr'] = {ext_map.get(t, t): r for t, r in zip(d['team'], d['rank'])}
    return ranks


@st.cache_data
@st.cache_data
def load_schedules(sport):
    sport_l = sport.lower()
    f = DATA_DIR / f'schedules_full_{sport_l}.csv'
    if not f.exists():
        return pd.DataFrame()
    return pd.read_csv(f, low_memory=False).fillna('')


def load_records(sport):
    sport_l = sport.lower()
    sf = DATA_DIR / f'schedules_full_{sport_l}.csv'
    if not sf.exists():
        return {}
    s = pd.read_csv(sf, low_memory=False).fillna('')
    played = s[s['result'].notna() & (s['result'] != '')]
    records = {}
    for tn, grp in played.groupby('teamName'):
        w = int(grp['result'].str.startswith('W').sum())
        l = len(grp) - w
        # Conference record
        # We don't have a clean conf flag, so just report overall
        records[tn] = {'wins': w, 'losses': l}
    return records


@st.cache_data
def load_player_rank():
    pr = pd.read_csv(DATA_DIR / 'player_rank.csv', low_memory=False,
                     usecols=['player_id', 'team_id', 'year',
                              'percentile_rank_weighted_run_created_efficiency',
                              'percentile_rank_weighted_run_allowed_efficiency'])
    pr['year'] = pd.to_numeric(pr['year'], errors='coerce')
    # Current year only — prevents graduated/transferred players from appearing
    max_year = pr['year'].max()
    pr = pr[pr['year'] == max_year]
    pr['team_id'] = pd.to_numeric(pr['team_id'], errors='coerce').astype('Int64')
    pr['hit_pct'] = pd.to_numeric(pr['percentile_rank_weighted_run_created_efficiency'], errors='coerce')
    pr['pit_pct'] = pd.to_numeric(pr['percentile_rank_weighted_run_allowed_efficiency'], errors='coerce')
    return pr


@st.cache_data
def load_players():
    p = pd.read_csv(DATA_DIR / 'players.csv', low_memory=False, encoding='latin-1', dtype=str).fillna('')
    p['id_int'] = pd.to_numeric(p['id'], errors='coerce').astype('Int64')
    return p


@st.cache_data
def load_hitting_pbp(sport, division):
    sport_l = sport.lower()
    f = PBP_DIR / sport_l / f'hitting_pbp_{division}.csv'
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
    return df


@st.cache_data
def load_pitching_pbp(sport, division):
    sport_l = sport.lower()
    f = PBP_DIR / sport_l / f'pitching_pbp_{division}.csv'
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
    return df


def get_team_id(team_name, teams_df, sport_label):
    match = teams_df[(teams_df['name'] == team_name) & (teams_df['sport'] == sport_label)]
    if len(match):
        return int(match.iloc[0]['id'])
    return None


def get_starters(team_id, player_rank_df, players_df, n=3):
    """Top N pitchers by pitching percentile for a team."""
    team_pr = player_rank_df[player_rank_df['team_id'] == team_id].copy()
    pitchers = team_pr.dropna(subset=['pit_pct'])
    pitchers = pitchers[pitchers['pit_pct'] > 0].sort_values('pit_pct', ascending=False)
    result = []
    for _, r in pitchers.head(n).iterrows():
        pid = int(r['player_id'])
        p_row = players_df[players_df['id_int'] == pid]
        name = p_row.iloc[0]['player_name'] if len(p_row) else f'Player {pid}'
        pos = p_row.iloc[0].get('position', '') if len(p_row) else ''
        result.append({'player_id': pid, 'name': name, 'position': pos,
                       'pit_pct': float(r['pit_pct'])})
    return result


def get_bullpen(team_id, player_rank_df, players_df):
    """Pitchers 4-10 by pitching percentile."""
    team_pr = player_rank_df[player_rank_df['team_id'] == team_id].copy()
    pitchers = team_pr.dropna(subset=['pit_pct'])
    pitchers = pitchers[pitchers['pit_pct'] > 0].sort_values('pit_pct', ascending=False)
    bp = pitchers.iloc[3:10]
    if bp.empty:
        return {'mean_pct': 0, 'count': 0, 'names': []}
    names = []
    for _, r in bp.iterrows():
        p_row = players_df[players_df['id_int'] == int(r['player_id'])]
        n = p_row.iloc[0]['player_name'] if len(p_row) else '?'
        names.append(n)
    return {'mean_pct': float(bp['pit_pct'].mean()), 'count': len(bp), 'names': names}


def get_hot_hitters(team_name, hitting_pbp, n=3, days=14):
    """Top N hitters by games played in last `days` days for a team."""
    if hitting_pbp.empty:
        return []
    cutoff = hitting_pbp['date'].max() - pd.Timedelta(days=days)
    recent = hitting_pbp[(hitting_pbp['date'] >= cutoff) &
                          (hitting_pbp['teamName'] == team_name)]
    if recent.empty:
        return []
    # Group by player, compute stats
    stats = recent.groupby(['playerId', 'playerName']).agg(
        games=('gameId', 'nunique'),
        ab=('ab', 'sum'),
        h=('h', 'sum'),
        hr=('hr', 'sum'),
        rbi=('rbi', 'sum'),
        bb=('bb', 'sum'),
    ).reset_index()
    stats['ab'] = pd.to_numeric(stats['ab'], errors='coerce').fillna(0)
    stats['h'] = pd.to_numeric(stats['h'], errors='coerce').fillna(0)
    stats['hr'] = pd.to_numeric(stats['hr'], errors='coerce').fillna(0)
    stats['rbi'] = pd.to_numeric(stats['rbi'], errors='coerce').fillna(0)
    stats['bb'] = pd.to_numeric(stats['bb'], errors='coerce').fillna(0)
    stats = stats[stats['ab'] >= 10]  # minimum plate appearances
    if stats.empty:
        return []
    stats['avg'] = stats['h'] / stats['ab']
    stats['obp'] = (stats['h'] + stats['bb']) / (stats['ab'] + stats['bb'])
    stats = stats.sort_values('avg', ascending=False)
    result = []
    for _, r in stats.head(n).iterrows():
        result.append({
            'name': r['playerName'], 'games': int(r['games']),
            'avg': f".{int(r['avg']*1000):03d}" if r['avg'] < 1 else '1.000',
            'hr': int(r['hr']), 'rbi': int(r['rbi']),
            'obp': f".{int(r['obp']*1000):03d}" if r['obp'] < 1 else '1.000',
        })
    return result


def get_last_5(team_name, schedules_df):
    """Get last 5 completed games for a team."""
    sched = schedules_df[schedules_df['teamName'] == team_name].copy()
    sched = sched[sched['result'].notna() & (sched['result'] != '')]
    sched['_d'] = pd.to_datetime(sched['date'], errors='coerce', format='mixed')
    sched = sched.sort_values('_d', ascending=False).head(5)
    results = []
    for _, g in sched.iterrows():
        result_str = str(g.get('result', ''))
        # Handle postponed/canceled as grey
        if any(k in result_str.lower() for k in ['ppd', 'canc', 'postpone', 'cancel']):
            wl = 'P'  # postponed — rendered grey in JS
            score = 'PPD'
        elif result_str.startswith('W'):
            wl = 'W'
            score = result_str.replace('W ', '').strip()
        else:
            wl = 'L'
            score = result_str.replace('L ', '').strip()
        opp = str(g.get('opponentName', '')).split('@')[0].strip()
        opp_short = opp[:3].upper() if len(opp) > 3 else opp.upper()
        venue = '@ ' if pd.notna(g.get('isAway')) and g.get('isAway') == 1.0 else 'vs '
        results.append({'wl': wl, 'score': score, 'opp': f'{venue}{opp_short}'})
    return results


def get_hot_pitcher(team_name, pitching_pbp, days=14):
    """Top pitcher by IP in last `days` days."""
    if pitching_pbp.empty:
        return None
    cutoff = pitching_pbp['date'].max() - pd.Timedelta(days=days)
    recent = pitching_pbp[(pitching_pbp['date'] >= cutoff) &
                           (pitching_pbp['teamName'] == team_name)]
    if recent.empty:
        return None
    stats = recent.groupby(['playerId', 'playerName']).agg(
        games=('gameId', 'nunique'),
        ip=('ip', 'sum'),
        h=('h', 'sum'),
        er=('er', 'sum'),
        so=('so', 'sum'),
        bb=('bb', 'sum'),
    ).reset_index()
    for c in ['ip', 'h', 'er', 'so', 'bb']:
        stats[c] = pd.to_numeric(stats[c], errors='coerce').fillna(0)
    stats = stats[stats['ip'] >= 3]
    if stats.empty:
        return None
    stats['era'] = (stats['er'] / stats['ip']) * 9 if stats['ip'].sum() > 0 else 99
    stats['era'] = stats.apply(lambda r: (r['er'] / r['ip']) * 9 if r['ip'] > 0 else 99, axis=1)
    best = stats.sort_values('era').iloc[0]
    return {
        'name': best['playerName'], 'games': int(best['games']),
        'ip': f"{best['ip']:.1f}", 'era': f"{best['era']:.2f}",
        'so': int(best['so']), 'bb': int(best['bb']),
    }


def compute_team_stats(team_name, hitting_pbp, pitching_pbp):
    """Aggregate season hitting + pitching stats for a team from PBP box scores."""
    stats = {}
    # Map PBP team name (with mascot) to short name — LONGEST PREFIX wins
    # to avoid "Georgia" matching "Georgia St." or "Georgia Tech"
    if not hitting_pbp.empty:
        pbp_names = [n for n in hitting_pbp['teamName'].dropna().unique() if isinstance(n, str)]
        candidates = [n for n in pbp_names if n.startswith(team_name)]
        # Prefer shortest match (exact team name + mascot, not a longer team like "Georgia St.")
        candidates.sort(key=len)
        pbp_name = candidates[0] if candidates else team_name
    else:
        pbp_name = team_name

    # Hitting
    if not hitting_pbp.empty:
        h = hitting_pbp[hitting_pbp['teamName'] == pbp_name].copy()
        for c in ['ab', 'h', 'hr', 'bb', 'hbp', 'doubles', 'triples', 'r', 'rbi', 'sf', 'tb']:
            if c in h.columns:
                h[c] = pd.to_numeric(h[c], errors='coerce').fillna(0)
        ab = h['ab'].sum()
        hits = h['h'].sum()
        hr = h['hr'].sum() if 'hr' in h.columns else 0
        bb = h['bb'].sum() if 'bb' in h.columns else 0
        hbp = h['hbp'].sum() if 'hbp' in h.columns else 0
        sf = h['sf'].sum() if 'sf' in h.columns else 0
        tb = h['tb'].sum() if 'tb' in h.columns else hits
        doubles = h['doubles'].sum() if 'doubles' in h.columns else 0
        triples = h['triples'].sum() if 'triples' in h.columns else 0
        k_h = h[h.columns[h.columns.str.lower().isin(['k', 'so'])]].sum().sum() if any(c.lower() in ('k', 'so') for c in h.columns) else 0
        pa = ab + bb + hbp + sf
        stats['BA'] = hits / ab if ab else 0
        stats['OBP'] = (hits + bb + hbp) / pa if pa else 0
        stats['SLG'] = tb / ab if ab else 0
        stats['OPS'] = stats['OBP'] + stats['SLG']
        stats['ISO'] = stats['SLG'] - stats['BA']
        stats['HR'] = int(hr)
        stats['R'] = int(h['r'].sum()) if 'r' in h.columns else 0
        stats['RBI'] = int(h['rbi'].sum()) if 'rbi' in h.columns else 0
        stats['K_rate_h'] = k_h / ab if ab else 0
        stats['BB_rate_h'] = bb / ab if ab else 0
        stats['HR_rate'] = hr / ab if ab else 0
        stats['AB'] = int(ab)
        stats['PA'] = int(pa)
    else:
        for k in ['BA','OBP','SLG','OPS','ISO','HR','R','RBI','K_rate_h','BB_rate_h','HR_rate','AB','PA']:
            stats[k] = 0

    # Pitching — compute all 7 stats to match the 14-axis radar template
    if not pitching_pbp.empty:
        p_cands = sorted([n for n in pitching_pbp['teamName'].dropna().unique()
                          if isinstance(n, str) and n.startswith(team_name)], key=len)
        p_name = p_cands[0] if p_cands else team_name
        p = pitching_pbp[pitching_pbp['teamName'] == p_name].copy()
        for c in ['ip', 'h', 'r', 'er', 'bb', 'so', 'bf', 'hrA', 'doublesA', 'triplesA', 'hb']:
            if c in p.columns:
                p[c] = pd.to_numeric(p[c], errors='coerce').fillna(0)
        ip = p['ip'].sum()
        p_h = p['h'].sum()
        p_er = p['er'].sum()
        p_bb = p['bb'].sum()
        p_so = p['so'].sum() if 'so' in p.columns else 0
        p_hr = p['hrA'].sum() if 'hrA' in p.columns else 0
        p_bf = p['bf'].sum() if 'bf' in p.columns else 0
        p_hb = p['hb'].sum() if 'hb' in p.columns else 0
        p_2b = p['doublesA'].sum() if 'doublesA' in p.columns else 0
        p_3b = p['triplesA'].sum() if 'triplesA' in p.columns else 0
        p_1b = p_h - p_2b - p_3b - p_hr
        # Standard pitching stats
        stats['ERA'] = (p_er / ip * 9) if ip else 0
        stats['WHIP'] = (p_h + p_bb) / ip if ip else 0
        stats['K9'] = (p_so / ip * 9) if ip else 0
        stats['BB9'] = (p_bb / ip * 9) if ip else 0
        stats['K_BB'] = p_so / p_bb if p_bb else 0
        stats['IP'] = round(ip, 1)
        stats['H_allowed'] = int(p_h)
        stats['SO'] = int(p_so)
        # OPS-against (for radar)
        ab_against = p_bf - p_bb - p_hb  # approximate AB
        if ab_against > 0:
            ba_against = p_h / ab_against
            obp_against = (p_h + p_bb + p_hb) / p_bf if p_bf else 0
            tb_against = p_1b + 2*p_2b + 3*p_3b + 4*p_hr
            slg_against = tb_against / ab_against
            stats['OPS_against'] = obp_against + slg_against
        else:
            stats['OPS_against'] = 0
        # BB% and K% (per BF, not per 9)
        stats['BB_pct'] = p_bb / p_bf if p_bf else 0
        stats['K_pct'] = p_so / p_bf if p_bf else 0
        # HR/AB rate (proxy for HR/FB% since we don't have fly ball data)
        stats['HR_AB'] = p_hr / ab_against if ab_against > 0 else 0
    else:
        for k in ['ERA','WHIP','K9','BB9','K_BB','IP','H_allowed','SO',
                   'OPS_against','BB_pct','K_pct','HR_AB']:
            stats[k] = 0

    return stats


def compute_percentile(val, all_vals, higher_is_better=True):
    """Percentile of val within all_vals. 100 = best."""
    arr = sorted([v for v in all_vals if pd.notna(v) and v != 0])
    if not arr:
        return 50.0
    pos = sum(1 for v in arr if v <= val)
    pct = pos / len(arr) * 100
    return pct if higher_is_better else (100 - pct)


def pct_color(pct):
    """Color based on percentile: green=top, yellow=mid, red=bottom."""
    if pct >= 75:
        return '#2d8a4e'  # green
    elif pct >= 50:
        return '#e8a735'  # yellow/gold
    elif pct >= 25:
        return '#d97030'  # orange
    else:
        return '#c41230'  # red


def render_matchup_card(team_a, team_b, stats_a, stats_b, all_team_stats,
                         ranks_a, ranks_b, rec_a, rec_b, tr_a, tr_b):
    """
    Render a publication-quality matchup stat card as a PNG.
    5-column mirrored layout matching the 64 Analytics Twitter Graphics template:
    [A Pitching] [A Hitting] [CENTER wRCE/wRAE] [B Hitting] [B Pitching]
    """
    BG = '#FAF8F2'
    TXT = '#2D2926'
    RED = '#C41230'
    NAVY = '#29335c'
    GRAY = '#888888'
    CELL_BG = '#FFFFFF'
    CELL_BORDER = '#CCCCCC'

    fig, ax = plt.subplots(1, 1, figsize=(18, 14), facecolor=BG)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_facecolor(BG)

    # ── Header: Team names ───────────────────────────────────────────────────
    ax.text(4.5, 13.3, team_a, fontsize=24, fontweight='bold', ha='center', color=TXT)
    ax.text(9, 13.3, '64 ANALYTICS', fontsize=13, fontweight='bold', ha='center',
            color=RED, fontstyle='italic')
    ax.text(13.5, 13.3, team_b, fontsize=24, fontweight='bold', ha='center', color=TXT)

    # ── Sub-header: Record / 64A Rank / RPI ──────────────────────────────────
    def draw_header_stats(cx, ranks, rec):
        y = 12.5
        r64 = f"#{int(ranks['64A'])}" if ranks.get('64A') else '—'
        rpi = f"#{int(ranks['RPI'])}" if ranks.get('RPI') else '—'
        rec_str = f"{rec['wins']}-{rec['losses']}"
        for label, val, dx in [('Record', rec_str, -2.2), ('64A Rank', r64, 0), ('RPI', rpi, 2.2)]:
            ax.text(cx + dx, y + 0.3, label, fontsize=9, ha='center', color=GRAY, fontweight='bold')
            ax.text(cx + dx, y - 0.15, val, fontsize=16, ha='center', color=TXT, fontweight='bold')

    draw_header_stats(4.5, ranks_a, rec_a)
    draw_header_stats(13.5, ranks_b, rec_b)

    # Divider line
    ax.plot([0.5, 17.5], [11.9, 11.9], color='#ddd', linewidth=1)

    # ── Stat cell helper ─────────────────────────────────────────────────────
    all_vals = {}
    for key in list(stats_a.keys()):
        all_vals[key] = [s.get(key, 0) for s in all_team_stats if s.get(key, 0) != 0]

    def draw_cell(cx, y, label, val, all_v, higher_is_better, w=3.2, h=0.95):
        """Single stat cell: label above, box with percentile + value."""
        # Label
        ax.text(cx, y + h / 2 + 0.25, label, fontsize=10, ha='center', fontweight='bold', color=TXT)
        # Box
        rect = mpatches.FancyBboxPatch((cx - w / 2, y - h / 2), w, h,
                                        boxstyle='round,pad=0.04', facecolor=CELL_BG,
                                        edgecolor=CELL_BORDER, linewidth=0.8)
        ax.add_patch(rect)
        # Value (centered, bold)
        fmt = '.3f' if isinstance(val, float) and abs(val) < 10 else '.1f'
        ax.text(cx, y + 0.12, f'{val:{fmt}}', fontsize=14, ha='center', va='center',
                fontweight='bold', color=TXT)
        # Percentile below value
        pct = compute_percentile(val, all_v, higher_is_better) if all_v else 50
        ax.text(cx, y - 0.25, f'{pct:.0f}%', fontsize=9, ha='center', va='center',
                fontweight='bold', color=pct_color(pct))

    def draw_center_cell(cx, y, label, val_a, val_b, w=3.8, h=0.95):
        """Center column cell with both teams' values side by side."""
        ax.text(cx, y + h / 2 + 0.25, label, fontsize=10, ha='center', fontweight='bold', color=TXT)
        rect = mpatches.FancyBboxPatch((cx - w / 2, y - h / 2), w, h,
                                        boxstyle='round,pad=0.04', facecolor=CELL_BG,
                                        edgecolor=CELL_BORDER, linewidth=0.8)
        ax.add_patch(rect)
        fmt_a = '.1f' if isinstance(val_a, float) and abs(val_a) >= 1 else '.3f'
        fmt_b = '.1f' if isinstance(val_b, float) and abs(val_b) >= 1 else '.3f'
        ax.text(cx - w * 0.22, y, f'{val_a:{fmt_a}}', fontsize=13, ha='center', va='center',
                fontweight='bold', color=TXT)
        ax.plot([cx, cx], [y - h / 2 + 0.1, y + h / 2 - 0.1], color='#ddd', linewidth=0.8)
        ax.text(cx + w * 0.22, y, f'{val_b:{fmt_b}}', fontsize=13, ha='center', va='center',
                fontweight='bold', color=TXT)

    # ── Column positions ─────────────────────────────────────────────────────
    COL_A_PIT = 2.0    # Team A Pitching (far left)
    COL_A_HIT = 5.5    # Team A Hitting
    COL_CENTER = 9.0   # Center metrics
    COL_B_HIT = 12.5   # Team B Hitting
    COL_B_PIT = 16.0   # Team B Pitching (far right)
    ROW_START = 10.8
    ROW_STEP = 1.35

    # ── Section headers ──────────────────────────────────────────────────────
    ax.text(COL_A_PIT, 11.5, 'Pitching', fontsize=14, ha='center', fontweight='bold', color=NAVY)
    ax.text(COL_A_HIT, 11.5, 'Hitting', fontsize=14, ha='center', fontweight='bold', color=RED)
    ax.text(COL_B_HIT, 11.5, 'Hitting', fontsize=14, ha='center', fontweight='bold', color=RED)
    ax.text(COL_B_PIT, 11.5, 'Pitching', fontsize=14, ha='center', fontweight='bold', color=NAVY)

    # ── Pitching stats (cols 1 and 5) ────────────────────────────────────────
    pit_stats = [
        ('ERA', 'ERA', False),
        ('WHIP', 'WHIP', False),
        ('K9', 'K/9', True),
        ('BB9', 'BB/9', False),
        ('K_BB', 'K/BB', True),
    ]
    for i, (key, label, hib) in enumerate(pit_stats):
        y = ROW_START - i * ROW_STEP
        draw_cell(COL_A_PIT, y, label, stats_a.get(key, 0), all_vals.get(key, []), hib)
        draw_cell(COL_B_PIT, y, label, stats_b.get(key, 0), all_vals.get(key, []), hib)

    # ── Hitting stats (cols 2 and 4) ─────────────────────────────────────────
    hit_stats = [
        ('OPS', 'OPS', True),
        ('OBP', 'OBP', True),
        ('SLG', 'SLG', True),
        ('ISO', 'ISO', True),
        ('BA', 'AVG', True),
        ('BB_rate_h', 'BB%', True),
        ('K_rate_h', 'K%', False),
        ('HR_rate', 'HR Rate', True),
    ]
    for i, (key, label, hib) in enumerate(hit_stats):
        y = ROW_START - i * ROW_STEP
        draw_cell(COL_A_HIT, y, label, stats_a.get(key, 0), all_vals.get(key, []), hib)
        draw_cell(COL_B_HIT, y, label, stats_b.get(key, 0), all_vals.get(key, []), hib)

    # ── Center column (col 3) ────────────────────────────────────────────────
    center_stats = [
        ('wRCE', tr_a.get('wRCE', 0), tr_b.get('wRCE', 0)),
        ('wRAE', tr_a.get('wRAE', 0), tr_b.get('wRAE', 0)),
        ('Total Rank', float(tr_a.get('rank', 0)), float(tr_b.get('rank', 0))),
    ]
    for i, (label, va, vb) in enumerate(center_stats):
        y = ROW_START - i * ROW_STEP
        draw_center_cell(COL_CENTER, y, label, va, vb)

    # ── Team labels at bottom ────────────────────────────────────────────────
    bottom_y = ROW_START - max(len(hit_stats), len(pit_stats)) * ROW_STEP - 0.3
    ax.text(COL_A_PIT, bottom_y, team_a, fontsize=9, ha='center', color=RED, fontweight='bold')
    ax.text(COL_A_HIT, bottom_y, team_a, fontsize=9, ha='center', color=RED, fontweight='bold')
    ax.text(COL_B_HIT, bottom_y, team_b, fontsize=9, ha='center', color=NAVY, fontweight='bold')
    ax.text(COL_B_PIT, bottom_y, team_b, fontsize=9, ha='center', color=NAVY, fontweight='bold')

    # Footer
    ax.text(9, 0.3, '*Percentages are division rankings', fontsize=8, ha='center',
            color=GRAY, fontstyle='italic')

    plt.tight_layout(pad=0.3)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def build_spider_chart(team_a_name, team_b_name, team_a_id, team_b_id,
                        player_rank_df, hitting_pbp, pitching_pbp,
                        team_a_color, team_b_color):
    """Radar/spider chart comparing two teams on 6 dimensions."""
    categories = ['Hitting', 'Pitching', 'Power', 'Bullpen', 'Speed', 'Discipline']

    def team_metrics(tid, tname):
        pr = player_rank_df[player_rank_df['team_id'] == tid]
        hit = pr.dropna(subset=['hit_pct']); hit = hit[hit['hit_pct'] > 0]
        pit = pr.dropna(subset=['pit_pct']); pit = pit[pit['pit_pct'] > 0]
        hitting_val = float(hit['hit_pct'].mean()) if len(hit) else 0.5
        pitching_val = float(pit['pit_pct'].mean()) if len(pit) else 0.5
        # Bullpen: pitchers 4-10
        pit_sorted = pit.sort_values('pit_pct', ascending=False)
        bp = pit_sorted.iloc[3:10]
        bullpen_val = float(bp['pit_pct'].mean()) if len(bp) else 0.5

        # Recent hitting stats for power/speed/discipline
        recent = hitting_pbp[hitting_pbp['teamName'] == tname].copy() if not hitting_pbp.empty else pd.DataFrame()
        if not recent.empty and len(recent) > 0:
            for c in ['ab', 'hr', 'sb', 'bb', 'h']:
                if c in recent.columns:
                    recent[c] = pd.to_numeric(recent[c], errors='coerce').fillna(0)
            ab = recent['ab'].sum() if 'ab' in recent.columns else 1
            if ab > 0:
                power_val = min(1.0, (recent['hr'].sum() if 'hr' in recent.columns else 0) / ab * 20)
                speed_val = min(1.0, (recent['sb'].sum() if 'sb' in recent.columns else 0) / ab * 15)
                disc_val = min(1.0, (recent['bb'].sum() if 'bb' in recent.columns else 0) / ab * 5)
            else:
                power_val = speed_val = disc_val = 0.5
        else:
            power_val = speed_val = disc_val = 0.5

        return [hitting_val, pitching_val, power_val, bullpen_val, speed_val, disc_val]

    vals_a = team_metrics(team_a_id, team_a_name)
    vals_b = team_metrics(team_b_id, team_b_name)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_a + [vals_a[0]], theta=categories + [categories[0]],
        fill='toself', fillcolor=f'rgba({int(team_a_color[1:3],16)},{int(team_a_color[3:5],16)},{int(team_a_color[5:7],16)},0.15)',
        line=dict(color=team_a_color, width=2),
        name=team_a_name,
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_b + [vals_b[0]], theta=categories + [categories[0]],
        fill='toself', fillcolor=f'rgba({int(team_b_color[1:3],16)},{int(team_b_color[3:5],16)},{int(team_b_color[5:7],16)},0.15)',
        line=dict(color=team_b_color, width=2),
        name=team_b_name,
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=BG_COLOR,
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
        ),
        showlegend=True,
        legend=dict(orientation='h', y=-0.1),
        paper_bgcolor=BG_COLOR,
        margin=dict(l=60, r=60, t=30, b=60),
        height=380,
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown('### Settings')
sel_sport = st.sidebar.selectbox('Sport', ['Baseball', 'Softball'], key='sp_sport')
sel_div = st.sidebar.selectbox('Division', ['D1', 'D2', 'D3'], key='sp_div')

teams_df, confs_df = load_teams_data()
div_label = DIVISION_LABELS.get(sel_div, 'D-I')
div_ids = set(confs_df[confs_df['division'] == div_label]['id'])
sport_teams = teams_df[(teams_df['sport'] == sel_sport) & (teams_df['conference_id'].isin(div_ids))]
team_list = sorted(sport_teams['name'].tolist())

if not team_list:
    st.warning(f'No teams found for {sel_sport} {sel_div}.')
    st.stop()

st.sidebar.markdown('### Matchup')
team_a = st.sidebar.selectbox('Team A (away)', team_list, index=0, key='sp_team_a')
default_b = min(1, len(team_list) - 1)
team_b = st.sidebar.selectbox('Team B (home)', team_list, index=default_b, key='sp_team_b')

if team_a == team_b:
    st.warning('Pick two different teams.')
    st.stop()


# ── Load Data ────────────────────────────────────────────────────────────────
tr26 = load_team_ranks(sel_sport, div_label)
ext_ranks = load_external_ranks(sel_sport)
records = load_records(sel_sport)
pr = load_player_rank()
players = load_players()
profiles = build_team_profiles(sel_sport, sel_div)
hitting_pbp = load_hitting_pbp(sel_sport, sel_div)
pitching_pbp = load_pitching_pbp(sel_sport, sel_div)

team_a_id = get_team_id(team_a, teams_df, sel_sport)
team_b_id = get_team_id(team_b, teams_df, sel_sport)

# Build rank_pct map for WP computation
ranked_df = tr26[tr26['team_id'].isin(sport_teams['id'])].copy()
ranked_df = ranked_df.merge(sport_teams[['id', 'name']], left_on='team_id', right_on='id', how='inner')
ranked_df['rank'] = ranked_df['rank_64a']
rank_pct_map = build_rank_pct_map(ranked_df.rename(columns={'name': 'team_name'}), 'rank')


# ── Header: Rankings Comparison ──────────────────────────────────────────────
st.markdown(f'## {team_a}  vs  {team_b}')

rec_a = records.get(team_a, {'wins': 0, 'losses': 0})
rec_b = records.get(team_b, {'wins': 0, 'losses': 0})

def get_rank_val(team_name, team_id):
    r64 = tr26[tr26['team_id'] == team_id]['rank_64a'].iloc[0] if len(tr26[tr26['team_id'] == team_id]) else None
    rpi = ext_ranks.get('rpi', {}).get(team_name)
    massey = ext_ranks.get('massey', {}).get(team_name)
    dsr = ext_ranks.get('dsr', {}).get(team_name)
    vals = [v for v in [r64, rpi, massey, dsr] if v is not None and not np.isnan(v)]
    true_rank = int(round(np.mean(vals))) if vals else None
    return {'64A': r64, 'RPI': rpi, 'Massey': massey, 'DSR': dsr, 'True Rank': true_rank}

ranks_a = get_rank_val(team_a, team_a_id)
ranks_b = get_rank_val(team_b, team_b_id)

h1, h2, h3 = st.columns([2, 1, 2])
with h1:
    st.markdown(f'### {team_a}')
    st.markdown(f'**Record: {rec_a["wins"]}-{rec_a["losses"]}**')
    rank_parts = [f'{k}: #{int(v)}' for k, v in ranks_a.items() if v is not None]
    st.caption(' | '.join(rank_parts))
with h2:
    st.markdown('<h2 style="text-align:center; color:#888;">vs</h2>', unsafe_allow_html=True)
with h3:
    st.markdown(f'### {team_b}')
    st.markdown(f'**Record: {rec_b["wins"]}-{rec_b["losses"]}**')
    rank_parts = [f'{k}: #{int(v)}' for k, v in ranks_b.items() if v is not None]
    st.caption(' | '.join(rank_parts))

st.markdown('---')


# ── Matchup Card (Claude Design template) ────────────────────────────────────
st.markdown('### Team Comparison')

# Compute stats for both teams + percentiles
stats_a = compute_team_stats(team_a, hitting_pbp, pitching_pbp)
stats_b = compute_team_stats(team_b, hitting_pbp, pitching_pbp)
all_team_names = sorted(sport_teams['name'].tolist())
all_team_stats_list = []
for tn in all_team_names:
    ts = compute_team_stats(tn, hitting_pbp, pitching_pbp)
    if ts.get('AB', 0) > 0:
        all_team_stats_list.append(ts)

tr26_data = load_team_ranks(sel_sport, div_label)
def get_team_rank_metrics(tid):
    row = tr26_data[tr26_data['team_id'] == tid]
    if row.empty:
        return {'rank': 0, 'wRCE': 0, 'wRAE': 0}
    r = row.iloc[0]
    return {
        'rank': int(r['rank_64a']) if pd.notna(r.get('rank_64a')) else 0,
        'wRCE': float(r.get('weighted_run_created_efficiency', 0) or 0),
        'wRAE': float(r.get('weighted_run_allowed_efficiency', 0) or 0),
    }
tr_metrics_a = get_team_rank_metrics(team_a_id)
tr_metrics_b = get_team_rank_metrics(team_b_id)

# Render via the Claude Design HTML template
import streamlit.components.v1 as components
import json

template_path = _APP_DIR / 'assets' / 'series-preview' / 'template_inline.html'
if template_path.exists():
    html = template_path.read_text(encoding='utf-8')

    def fmt_val(v, is_rate=False):
        if isinstance(v, float):
            if is_rate: return f"{v*100:.1f}%"
            if abs(v) < 1: return f".{int(v*1000):03d}" if v >= 0 else f"-.{int(abs(v)*1000):03d}"
            if abs(v) < 10: return f"{v:.2f}"
            return f"{v:.1f}"
        return str(v)

    def pct_for(key, stats, higher_is_better=True):
        val = stats.get(key, 0)
        all_v = [s.get(key, 0) for s in all_team_stats_list if s.get(key, 0) != 0]
        if not all_v: return 50.0
        pos = sum(1 for v in all_v if v <= val)
        pct = pos / len(all_v) * 100
        return round(pct, 1) if higher_is_better else round(100 - pct, 1)

    # Build DATA object matching the template's 14-axis format (7 hitting + 7 pitching)
    hitting_keys = [('OPS','OPS',True),('OBP','wOBA',True),('ISO','ISO',True),
                    ('BA','AVG',True),('BB_rate_h','BB%',True),('K_rate_h','K%',False),
                    ('HR_rate','HR Rate',True)]
    pitching_keys = [('OPS_against','OPS',False),('BB_pct','BB%',False),('K_pct','K%',True),
                     ('WHIP','WHIP',False),('ERA','ERA',False),('K9','K/9',True),
                     ('HR_AB','HR/AB%',False)]

    def build_stats_array(stats_dict, keys):
        arr = []
        for key, label, hib in keys:
            val = stats_dict.get(key, 0)
            arr.append({"label": label, "value": fmt_val(val), "pct": pct_for(key, stats_dict, hib)})
        return arr

    # Last 5 games
    schedules_df = load_schedules(sel_sport)
    last5_a = get_last_5(team_a, schedules_df) if not schedules_df.empty else []
    last5_b = get_last_5(team_b, schedules_df) if not schedules_df.empty else []

    # Records + ranks for header
    r64a = f"#{int(ranks_a['64A'])}" if ranks_a.get('64A') else '—'
    r64b = f"#{int(ranks_b['64A'])}" if ranks_b.get('64A') else '—'
    rpia = f"#{int(ranks_a['RPI'])}" if ranks_a.get('RPI') else '—'
    rpib = f"#{int(ranks_b['RPI'])}" if ranks_b.get('RPI') else '—'

    data_js = f"""
    window.DATA = {{
      teamA: {{
        name: {json.dumps(team_a.upper())},
        hitting: {json.dumps(build_stats_array(stats_a, hitting_keys))},
        pitching: {json.dumps(build_stats_array(stats_a, pitching_keys))}
      }},
      teamB: {{
        name: {json.dumps(team_b.upper())},
        hitting: {json.dumps(build_stats_array(stats_b, hitting_keys))},
        pitching: {json.dumps(build_stats_array(stats_b, pitching_keys))}
      }}
    }};
    """

    # Compute D1 percentiles for bullet bar widths (OUTSIDE the f-string)
    all_tr = tr26_data[tr26_data['team_id'].isin(sport_teams['id'])].copy()
    all_tr['wRCE'] = pd.to_numeric(all_tr.get('weighted_run_created_efficiency'), errors='coerce')
    all_tr['wRAE'] = pd.to_numeric(all_tr.get('weighted_run_allowed_efficiency'), errors='coerce')
    all_tr['rank'] = pd.to_numeric(all_tr.get('rank_64a'), errors='coerce')

    def div_pct(val, col, higher_better=True):
        vals = all_tr[col].dropna().tolist()
        if not vals or val is None: return 50
        pos = sum(1 for v in vals if v <= val)
        pct = pos / len(vals) * 100
        return round(pct if higher_better else (100 - pct), 1)

    wrce_a_pct = div_pct(tr_metrics_a['wRCE'], 'wRCE', True)
    wrce_b_pct = div_pct(tr_metrics_b['wRCE'], 'wRCE', True)
    wrae_a_pct = div_pct(tr_metrics_a['wRAE'], 'wRAE', False)  # lower wRAE = better pitching
    wrae_b_pct = div_pct(tr_metrics_b['wRAE'], 'wRAE', False)
    rank_a_pct = div_pct(tr_metrics_a['rank'], 'rank', False)
    rank_b_pct = div_pct(tr_metrics_b['rank'], 'rank', False)

    cum_rank_a = f"#{int(ranks_a['True Rank'])}" if ranks_a.get('True Rank') else '—'
    cum_rank_b = f"#{int(ranks_b['True Rank'])}" if ranks_b.get('True Rank') else '—'
    bullets_data = [
        {"label": "wRCE", "a": {"v": fmt_val(tr_metrics_a['wRCE']), "pct": wrce_a_pct}, "b": {"v": fmt_val(tr_metrics_b['wRCE']), "pct": wrce_b_pct}},
        {"label": "wRAE", "a": {"v": fmt_val(tr_metrics_a['wRAE']), "pct": wrae_a_pct}, "b": {"v": fmt_val(tr_metrics_b['wRAE']), "pct": wrae_b_pct}},
        {"label": "CUMULATIVE RANK", "a": {"v": cum_rank_a, "pct": rank_a_pct}, "b": {"v": cum_rank_b, "pct": rank_b_pct}},
    ]
    data_js += f"""
    window.BULLETS = {json.dumps(bullets_data)};
    window.LAST5 = {{
      a: {json.dumps(last5_a)},
      b: {json.dumps(last5_b)}
    }};
    window.RADAR_AXES = [
      {{ key: "OPS", group: "H" }},
      {{ key: "wOBA", group: "H" }},
      {{ key: "ISO", group: "H" }},
      {{ key: "AVG", group: "H" }},
      {{ key: "BB%", group: "H" }},
      {{ key: "K%", group: "H" }},
      {{ key: "HR Rate", group: "H" }},
      {{ key: "OPS", label: "OPS-A", group: "P", source: "pitching", lookup: "OPS" }},
      {{ key: "BB%", group: "P", source: "pitching", lookup: "BB%" }},
      {{ key: "K%", group: "P", source: "pitching", lookup: "K%" }},
      {{ key: "WHIP", group: "P" }},
      {{ key: "ERA", group: "P" }},
      {{ key: "K/9", group: "P" }},
      {{ key: "HR/AB%", group: "P" }}
    ];
    """

    # ── Extract team colors from logos ──
    from PIL import Image
    from collections import Counter as ImgCounter
    def get_team_color(team_name_key, fallback='#C41230'):
        tid = get_team_id(team_name_key, teams_df, sel_sport)
        if not tid: return fallback
        for ext in ('png', 'webp'):
            lp = _APP_DIR / 'team_logos_512' / f'{tid}.{ext}'
            if lp.exists():
                try:
                    img = Image.open(lp).convert('RGBA')
                    img.thumbnail((64, 64))
                    px = np.array(img)
                    mask = px[:, :, 3] > 128
                    rgb = px[mask][:, :3]
                    filtered = [(r,g,b) for r,g,b in rgb if 35 < (int(r)+int(g)+int(b))/3 < 220]
                    if not filtered: return fallback
                    quant = [(r//16*16, g//16*16, b//16*16) for r,g,b in filtered]
                    top = ImgCounter(quant).most_common(1)[0][0]
                    return f'#{top[0]:02x}{top[1]:02x}{top[2]:02x}'
                except: pass
        return fallback

    color_a = get_team_color(team_a, '#C41230')
    color_b = get_team_color(team_b, '#29335C')

    # Inject team colors as CSS overrides
    color_css = f"""
    .team-block.left .team-name {{ color: {color_a} !important; }}
    .team-block.right .team-name {{ color: {color_b} !important; }}
    .wordmark-logo, .wordmark-sub, .wordmark-title {{ filter: none !important; }}
    #zoneB .micro-group {{ direction: ltr; }}
    #zoneB .zone-head {{ justify-content: flex-end; }}
    .footer-team.a {{ color: {color_a} !important; }}
    .footer-team.b {{ color: {color_b} !important; }}
    .micro-group-label {{ color: {color_a} !important; }}
    #zoneB .micro-group-label {{ color: {color_b} !important; }}
    .zone-title {{ color: {color_a}; }}
    #zoneB .zone-title {{ color: {color_b} !important; }}
    .bullet-bar.a {{ background: {color_a} !important; }}
    .bullet-bar.b {{ background: {color_b} !important; }}
    .bullet-val.a {{ color: {color_a} !important; }}
    .bullet-val.b {{ color: {color_b} !important; }}
    .radar-legend .sw.a {{ background: {color_a}33 !important; border-color: {color_a} !important; }}
    .radar-legend .sw.b {{ background: {color_b}33 !important; border-color: {color_b} !important; }}
    .team-logo.a {{ background: {color_a} !important; }}
    .team-logo.b {{ background: {color_b} !important; }}
    """
    html = html.replace('</style>', f'{color_css}</style>')

    # Inject team color JS variables for radar/pace/bullet charts
    color_js = f"""
    var TEAM_A_COLOR = '{color_a}';
    var TEAM_B_COLOR = '{color_b}';
    """
    # Replace pace chart color params (targeted — only in renderPaceSmall calls)
    import re as re_mod
    html = re_mod.sub(r"(renderPaceSmall\(\$\('#paceA[^']*'\),\s*PACE\.a\.\w+,\s*PACE\.meta\.\w+,\s*)'red'",
                       rf"\1'{color_a}'", html)
    html = re_mod.sub(r"(renderPaceSmall\(\$\('#paceB[^']*'\),\s*PACE\.b\.\w+,\s*PACE\.meta\.\w+,\s*)'navy'",
                       rf"\1'{color_b}'", html)

    # Replace team names, records, ranks in the HTML
    html = html.replace('NORTH GEORGIA', team_a.upper())
    html = html.replace('North Georgia', team_a)
    html = html.replace('CATAWBA', team_b.upper())
    html = html.replace('Catawba', team_b)
    html = html.replace('NIGHTHAWKS · DAHLONEGA, GA', f'{sel_sport.upper()} · {sel_div}')
    html = html.replace('INDIANS · SALISBURY, NC', f'{sel_sport.upper()} · {sel_div}')
    html = html.replace('>19‑4<', f">{rec_a['wins']}‑{rec_a['losses']}<")
    html = html.replace('>19‑6<', f">{rec_b['wins']}‑{rec_b['losses']}<")
    html = html.replace('>#13<', f'>{r64a}<')
    html = html.replace('>#22<', f'>{r64b}<')
    html = html.replace('>#8<', f'>{rpia}<')
    html = html.replace('>#19<', f'>{rpib}<')
    # Conference record — compute from schedule (games vs same-conference opponents)
    def get_conf_record(team_name_key, sched_df, teams_all, confs_all):
        tid = get_team_id(team_name_key, teams_all, sel_sport)
        if not tid or sched_df.empty: return '—'
        team_conf = teams_all[teams_all['id']==tid]['conference_id'].iloc[0] if len(teams_all[teams_all['id']==tid]) else None
        if not team_conf: return '—'
        conf_teams = set(teams_all[teams_all['conference_id']==team_conf]['name'])
        team_sched = sched_df[(sched_df['teamName']==team_name_key) & (sched_df['result'].notna()) & (sched_df['result']!='')]
        conf_games = team_sched[team_sched['opponentName'].apply(lambda x: str(x).split('@')[0].strip() in conf_teams)]
        if conf_games.empty: return '—'
        cw = int(conf_games['result'].str.startswith('W').sum())
        cl = len(conf_games) - cw
        return f'{cw}‑{cl}'

    conf_a = get_conf_record(team_a, schedules_df, teams_df, confs_df)
    conf_b = get_conf_record(team_b, schedules_df, teams_df, confs_df)
    html = html.replace('>9‑3<', f'>{conf_a}<')
    html = html.replace('>10‑4<', f'>{conf_b}<')
    html = html.replace('Peach Belt Conference', f'{sel_sport} {sel_div}')
    html = html.replace('HOME · 19‑4', f"HOME · {rec_a['wins']}‑{rec_a['losses']}")
    html = html.replace('AWAY · 19‑6', f"AWAY · {rec_b['wins']}‑{rec_b['losses']}")

    # Add "2 WEEK REVIEW" header above pace charts
    html = html.replace('<div class="pace-grid-4">',
                        '<div style="font-size:11px;font-weight:800;letter-spacing:0.2em;text-transform:uppercase;color:var(--gray);text-align:center;margin-bottom:4px;">2 WEEK REVIEW</div><div class="pace-grid-4">')

    # Override ALL hardcoded template data with our dynamic versions
    html = html.replace('window.DATA = {', f'window._ORIG_DATA = {{')
    html = html.replace('window.LAST5 = {', f'window._ORIG_LAST5 = {{')
    html = html.replace('window.RADAR_AXES = [', f'window._ORIG_RADAR_AXES = [')
    html = html.replace('window.BULLETS = [', f'window._ORIG_BULLETS = [')

    # Replace xFIP/SIERA labels with FIP/WHIP in the pace chart HTML
    html = html.replace('>xFIP<', '>FIP<')
    html = html.replace('>SIERA<', '>WHIP<')
    html = html.replace('paceAXFIP', 'paceAFIP')
    html = html.replace('paceASIERA', 'paceAWHIP')
    html = html.replace('paceBXFIP', 'paceBFIP')
    html = html.replace('paceBSIERA', 'paceBWHIP')

    # Build pace data from real PBP (rolling 7-game averages)
    def compute_pace(team_name, hitting_pbp_df, pitching_pbp_df, n_games=14):
        """Compute rolling 7-game pace for OPS, FIP, wRC+, WHIP."""
        pace = {'ops': [], 'fip': [], 'wrc': [], 'whip': []}
        if hitting_pbp_df.empty:
            return pace

        h_cands = sorted([n for n in hitting_pbp_df['teamName'].dropna().unique()
                          if isinstance(n, str) and n.startswith(team_name)], key=len)
        pbp_name = h_cands[0] if h_cands else team_name

        # Get recent games
        h = hitting_pbp_df[hitting_pbp_df['teamName'] == pbp_name].copy()
        h['date'] = pd.to_datetime(h['date'], errors='coerce', format='mixed')
        for c in ['ab','h','hr','bb','hbp','tb','doubles','triples']:
            if c in h.columns: h[c] = pd.to_numeric(h[c], errors='coerce').fillna(0)

        game_stats = h.groupby('gameId').agg(
            date=('date','first'), ab=('ab','sum'), hits=('h','sum'),
            hr=('hr','sum'), bb=('bb','sum'), hbp=('hbp','sum'),
            tb=('tb','sum')
        ).sort_values('date', ascending=False).head(n_games).iloc[::-1]

        if len(game_stats) < 3:
            return pace

        # Pitching per game
        pp_cands = sorted([n for n in pitching_pbp_df['teamName'].dropna().unique()
                           if isinstance(n, str) and n.startswith(team_name)], key=len)
        p_name = pp_cands[0] if pp_cands else team_name if not pitching_pbp_df.empty else team_name
        p = pitching_pbp_df[pitching_pbp_df['teamName'] == p_name].copy() if not pitching_pbp_df.empty else pd.DataFrame()
        if not p.empty:
            p['date'] = pd.to_datetime(p['date'], errors='coerce', format='mixed')
            for c in ['ip','h','bb','so','hrA','hb','er']:
                if c in p.columns: p[c] = pd.to_numeric(p[c], errors='coerce').fillna(0)
            p_games = p.groupby('gameId').agg(
                date=('date','first'), ip=('ip','sum'), p_h=('h','sum'),
                p_bb=('bb','sum'), p_so=('so','sum'), p_hr=('hrA','sum'),
                p_hb=('hb','sum'), p_er=('er','sum')
            ).sort_values('date', ascending=False).head(n_games).iloc[::-1]
        else:
            p_games = pd.DataFrame()

        # Rolling 7-game windows
        for i in range(len(game_stats)):
            window = game_stats.iloc[max(0, i-6):i+1]
            ab = window['ab'].sum()
            hits_w = window['hits'].sum()
            tb = window['tb'].sum()
            bb = window['bb'].sum()
            hbp = window['hbp'].sum()
            pa = ab + bb + hbp
            ops = ((hits_w+bb+hbp)/pa + tb/ab) if ab > 0 and pa > 0 else 0
            pace['ops'].append(round(ops, 3))
            pace['wrc'].append(round(ops * 100 / 0.8, 0) if ops > 0 else 100)  # rough wRC+ proxy

            if not p_games.empty and i < len(p_games):
                pw = p_games.iloc[max(0, i-6):i+1]
                ip = pw['ip'].sum()
                whip = (pw['p_h'].sum() + pw['p_bb'].sum()) / ip if ip > 0 else 0
                hr_p = pw['p_hr'].sum()
                bb_p = pw['p_bb'].sum()
                hb_p = pw['p_hb'].sum()
                so_p = pw['p_so'].sum()
                fip = ((13*hr_p + 3*(bb_p+hb_p) - 2*so_p) / ip + 3.10) if ip > 0 else 0
                pace['fip'].append(round(fip, 2))
                pace['whip'].append(round(whip, 2))
            else:
                pace['fip'].append(4.5)
                pace['whip'].append(1.4)
        return pace

    pace_a = compute_pace(team_a, hitting_pbp, pitching_pbp)
    pace_b = compute_pace(team_b, hitting_pbp, pitching_pbp)

    pace_js = f"""
    window.PACE = {{
      a: {{ ops: {json.dumps(pace_a['ops'])}, xfip: {json.dumps(pace_a['fip'])},
           wrc: {json.dumps(pace_a['wrc'])}, siera: {json.dumps(pace_a['whip'])} }},
      b: {{ ops: {json.dumps(pace_b['ops'])}, xfip: {json.dumps(pace_b['fip'])},
           wrc: {json.dumps(pace_b['wrc'])}, siera: {json.dumps(pace_b['whip'])} }},
      meta: {{
        ops:   {{ divAvg: 0.750, min: 0.55, max: 1.05, format: function(v) {{ return '.' + Math.max(0, Math.round(v*1000)).toString().padStart(3,'0').slice(-3); }} }},
        xfip:  {{ divAvg: 4.50, min: 2.0, max: 7.0, format: function(v) {{ return v.toFixed(2); }}, lowerBetter: true }},
        wrc:   {{ divAvg: 100, min: 70, max: 160, format: function(v) {{ return Math.round(v).toString(); }} }},
        siera: {{ divAvg: 1.40, min: 0.9, max: 2.0, format: function(v) {{ return v.toFixed(2); }}, lowerBetter: true }}
      }}
    }};
    """
    html = html.replace('window.PACE = {', f'{pace_js}\nwindow._ORIG_PACE = {{')

    # Team logo watermarks — replace hardcoded team logos with selected team's logo
    import base64 as b64mod
    logo_dir = _APP_DIR / 'team_logos_512'
    for team_name_key, zone_id in [(team_a, 'zoneA'), (team_b, 'zoneB')]:
        tid = get_team_id(team_name_key, teams_df, sel_sport)
        if tid:
            for ext in ('png', 'webp'):
                logo_path = logo_dir / f'{tid}.{ext}'
                if logo_path.exists():
                    logo_b64 = b64mod.b64encode(logo_path.read_bytes()).decode()
                    mime = 'image/png' if ext == 'png' else 'image/webp'
                    # Inject CSS to override the pace-grid watermark for this zone
                    css_inject = f"""
                    #{zone_id} .pace-grid-4::after {{
                        background-image: url('data:{mime};base64,{logo_b64}') !important;
                    }}
                    """
                    html = html.replace('</style>', f'{css_inject}</style>')
                    break

    # Inject all dynamic JS right before </head>
    all_js = f'<script>{data_js}\n{pace_js}</script>'
    html = html.replace('</head>', f'{all_js}\n</head>')

    # Inject a PNG download button directly into the card HTML
    png_btn = f"""
    <div style="text-align:center;padding:8px 0 0;">
      <button onclick="window.downloadCardPNG()" style="
        background:#C41230;color:#fff;border:none;padding:8px 24px;
        font-size:13px;font-weight:700;border-radius:6px;cursor:pointer;
        letter-spacing:0.06em;font-family:Inter,sans-serif;">
        Download PNG
      </button>
    </div>
    """
    html = html.replace('</body>', f'{png_btn}</body>')

    components.html(html, height=960, scrolling=False)

else:
    st.warning('Series preview template not found. Run the design build first.')


# ── Game-by-Game Predictions ─────────────────────────────────────────────────
st.markdown('---')
st.markdown('### Game-by-Game Predictions')

starters_a = get_starters(team_a_id, pr, players, n=3)
starters_b = get_starters(team_b_id, pr, players, n=3)
bp_a = get_bullpen(team_a_id, pr, players)
bp_b = get_bullpen(team_b_id, pr, players)

for game_num in [1, 2, 3]:
    g1, g2, g3 = st.columns([2, 1, 2])
    sp_a = starters_a[game_num - 1] if len(starters_a) >= game_num else None
    sp_b = starters_b[game_num - 1] if len(starters_b) >= game_num else None

    # Compute WP for this game
    a_static = rank_pct_map.get(team_a, 0.5)
    b_static = rank_pct_map.get(team_b, 0.5)
    a_player = adjusted_team_pct(team_a, profiles, 'Weekend', game_num, a_static, sport=sel_sport)
    b_player = adjusted_team_pct(team_b, profiles, 'Weekend', game_num, b_static, sport=sel_sport)
    a_p = blend_with_static(a_player, a_static)
    b_p = blend_with_static(b_player, b_static)
    # team_b is home
    wp_home = pre_game_wp(b_p, a_p)
    wp_away = 1 - wp_home

    with g1:
        st.markdown(f'**Game {game_num} — {team_a} ({wp_away*100:.1f}%)**')
        if sp_a:
            st.write(f'SP: {sp_a["name"]} ({sp_a["position"]})')
            st.caption(f'Pitcher percentile: {sp_a["pit_pct"]:.1%}')
        else:
            st.write('SP: TBD')
    with g2:
        # Visual WP bar
        st.markdown(f'<div style="text-align:center; font-size:1.5em; font-weight:bold; margin-top:10px;">'
                    f'{wp_away*100:.0f}% — {wp_home*100:.0f}%</div>',
                    unsafe_allow_html=True)
    with g3:
        st.markdown(f'**Game {game_num} — {team_b} ({wp_home*100:.1f}%)**')
        if sp_b:
            st.write(f'SP: {sp_b["name"]} ({sp_b["position"]})')
            st.caption(f'Pitcher percentile: {sp_b["pit_pct"]:.1%}')
        else:
            st.write('SP: TBD')

# Bullpen comparison
st.markdown('---')
st.markdown('### Bullpen Comparison')
bp1, bp2 = st.columns(2)
with bp1:
    st.markdown(f'**{team_a} Bullpen** (arms 4-10)')
    if bp_a['count']:
        st.metric('Mean Percentile', f"{bp_a['mean_pct']:.1%}")
        st.caption(', '.join(bp_a['names'][:5]))
    else:
        st.write('Insufficient data')
with bp2:
    st.markdown(f'**{team_b} Bullpen** (arms 4-10)')
    if bp_b['count']:
        st.metric('Mean Percentile', f"{bp_b['mean_pct']:.1%}")
        st.caption(', '.join(bp_b['names'][:5]))
    else:
        st.write('Insufficient data')


# ── Who's Hot (last 14 days) ─────────────────────────────────────────────────
st.markdown('---')
st.markdown("### Who's Hot (last 14 days)")

# Need to map team names to PBP team names (PBP has mascot suffix)
# Build reverse map: short name -> PBP full name
if not hitting_pbp.empty:
    pbp_names = set(hitting_pbp['teamName'].dropna())
    def find_pbp_name(short):
        cands = sorted([n for n in pbp_names if isinstance(n, str) and n.startswith(short)], key=len)
        return cands[0] if cands else short
    pbp_a = find_pbp_name(team_a)
    pbp_b = find_pbp_name(team_b)
else:
    pbp_a = team_a
    pbp_b = team_b

hot1, hot2 = st.columns(2)
with hot1:
    st.markdown(f'**{team_a} — Hot Hitters**')
    hot_h_a = get_hot_hitters(pbp_a, hitting_pbp)
    if hot_h_a:
        for h in hot_h_a:
            st.write(f"**{h['name']}** — {h['avg']} AVG, {h['hr']} HR, {h['rbi']} RBI ({h['games']}G)")
    else:
        st.caption('No recent data')

    st.markdown(f'**{team_a} — Hot Pitcher**')
    hot_p_a = get_hot_pitcher(pbp_a, pitching_pbp)
    if hot_p_a:
        st.write(f"**{hot_p_a['name']}** — {hot_p_a['era']} ERA, {hot_p_a['so']} K, {hot_p_a['bb']} BB ({hot_p_a['ip']} IP, {hot_p_a['games']}G)")
    else:
        st.caption('No recent data')

with hot2:
    st.markdown(f'**{team_b} — Hot Hitters**')
    hot_h_b = get_hot_hitters(pbp_b, hitting_pbp)
    if hot_h_b:
        for h in hot_h_b:
            st.write(f"**{h['name']}** — {h['avg']} AVG, {h['hr']} HR, {h['rbi']} RBI ({h['games']}G)")
    else:
        st.caption('No recent data')

    st.markdown(f'**{team_b} — Hot Pitcher**')
    hot_p_b = get_hot_pitcher(pbp_b, pitching_pbp)
    if hot_p_b:
        st.write(f"**{hot_p_b['name']}** — {hot_p_b['era']} ERA, {hot_p_b['so']} K, {hot_p_b['bb']} BB ({hot_p_b['ip']} IP, {hot_p_b['games']}G)")
    else:
        st.caption('No recent data')


# ── Series Win Probability ───────────────────────────────────────────────────
st.markdown('---')
st.markdown('### Series Outcome Probabilities')
# Monte Carlo: probability of sweeping, winning 2-1, etc.
n_sims = 5000
game_wps = []
for gn in [1, 2, 3]:
    a_pl = adjusted_team_pct(team_a, profiles, 'Weekend', gn, a_static, sport=sel_sport)
    b_pl = adjusted_team_pct(team_b, profiles, 'Weekend', gn, b_static, sport=sel_sport)
    a_bl = blend_with_static(a_pl, a_static)
    b_bl = blend_with_static(b_pl, b_static)
    game_wps.append(pre_game_wp(b_bl, a_bl))  # P(home=team_b wins)

rng = np.random.default_rng(42)
a_series_wins = 0
b_series_wins = 0
outcomes = {'3-0 away': 0, '2-1 away': 0, '2-1 home': 0, '3-0 home': 0}
for _ in range(n_sims):
    a_w = sum(1 for gw in game_wps if rng.random() > gw)  # away wins when home loses
    b_w = 3 - a_w
    if a_w >= 2:
        a_series_wins += 1
    else:
        b_series_wins += 1
    if a_w == 3:
        outcomes['3-0 away'] += 1
    elif a_w == 2:
        outcomes['2-1 away'] += 1
    elif b_w == 2:
        outcomes['2-1 home'] += 1
    else:
        outcomes['3-0 home'] += 1

s1, s2, s3, s4 = st.columns(4)
s1.metric(f'{team_a} sweep', f"{outcomes['3-0 away']/n_sims:.1%}")
s2.metric(f'{team_a} 2-1', f"{outcomes['2-1 away']/n_sims:.1%}")
s3.metric(f'{team_b} 2-1', f"{outcomes['2-1 home']/n_sims:.1%}")
s4.metric(f'{team_b} sweep', f"{outcomes['3-0 home']/n_sims:.1%}")

st.caption(f'Series win: **{team_a} {a_series_wins/n_sims:.1%}** — **{team_b} {b_series_wins/n_sims:.1%}** ({n_sims:,} simulations)')
