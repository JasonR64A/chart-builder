"""
Spray-chart data layer.

PBP rows have a `hitLocation` column with a baseball position code:
  1=P, 2=C, 3=1B, 4=2B, 5=3B, 6=SS, 7=LF, 8=CF, 9=RF
  10=LCF gap, 11=RCF gap, 12-14 = various foul / extra zones (rare)

This module provides:
  - compute_spray_distribution(sport, division, team=None, player_id=None)
    → per-zone counts + percentages for the selected scope
  - compute_field_side_buckets(spray_df)
    → Left side (5/6/7) / Middle (4/8) / Right side (3/9) splits
  - HIT_RESULT_BUCKETS — categorize playResult into 1B/2B/3B/HR/Out/Other
  - ZONE_NAMES, ZONE_COORDS — labels + (x,y) for the field-diagram SVG
"""
from pathlib import Path
import pandas as pd
import streamlit as st

PBP_DIR = Path(__file__).resolve().parent.parent / 'pbp_data' / 'play_by_play'
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'


@st.cache_data(show_spinner=False)
def _load_csv(name: str) -> pd.DataFrame:
    """Load chart-builder data CSV with permissive encoding."""
    p = DATA_DIR / name
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, low_memory=False, encoding='utf-8-sig')
    except UnicodeDecodeError:
        return pd.read_csv(p, low_memory=False, encoding='latin-1')

ZONE_NAMES = {
    1: 'P', 2: 'C', 3: '1B', 4: '2B', 5: '3B', 6: 'SS',
    7: 'LF', 8: 'CF', 9: 'RF',
    10: 'LCF', 11: 'RCF', 12: 'LF-Foul', 13: 'RF-Foul', 14: 'CF-Deep',
}

# (x, y) center positions for each zone on a 100x120 baseball field SVG.
# Origin (0,0) is top-left. Home plate at (50, 110). Outfield arc top.
# Tuned to look like a realistic baseball diamond layout.
ZONE_COORDS = {
    1:  (50, 80),   # Pitcher's mound
    2:  (50, 108),  # Catcher (just behind home)
    3:  (74, 88),   # 1B
    4:  (66, 70),   # 2B side of bag
    5:  (26, 88),   # 3B
    6:  (34, 70),   # SS side of bag
    7:  (20, 38),   # LF
    8:  (50, 28),   # CF
    9:  (80, 38),   # RF
    10: (32, 28),   # LCF gap
    11: (68, 28),   # RCF gap
    12: (10, 70),   # LF foul
    13: (90, 70),   # RF foul
    14: (50, 12),   # CF deep
}

# Hit result categorization — playResult column uses NCAA scorebook codes
HIT_RESULT_BUCKETS = {
    '1B': ['1B'],
    '2B': ['2B'],
    '3B': ['3B'],
    'HR': ['HR'],
    'Out': ['GO', 'FO', 'PO', 'LO', 'DP', 'TP', 'SF', 'SH'],
    'FC/Err': ['FC', 'E'],
}


def categorize_result(r: str) -> str:
    if not isinstance(r, str):
        return 'Other'
    for bucket, codes in HIT_RESULT_BUCKETS.items():
        if r in codes:
            return bucket
    return 'Other'


# Linear weights for wOBA on contact (BIP-only). Standard FanGraphs values
# rounded to three decimals — walks/HBP excluded since they have no
# hitLocation. This is wOBAcon, the contact-quality complement to xwOBA.
WOBA_WEIGHTS = {'1B': 0.888, '2B': 1.271, '3B': 1.616, 'HR': 2.101}


def add_zone_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-zone rate stats to a spray distribution DataFrame.

    Columns added: AVG, SLG, wOBA (BIP-denominator versions), TB (raw count).
    Denominator is the per-zone total BIP. AVG/SLG/wOBA are NOT the standard
    PA-denominator versions — they're conditional on putting the ball in
    the zone, which is the right framing for a spray chart.
    """
    if df.empty:
        return df
    out = df.copy()
    for col in ['1B', '2B', '3B', 'HR']:
        if col not in out.columns:
            out[col] = 0
    hits = out['1B'] + out['2B'] + out['3B'] + out['HR']
    tb = out['1B'] + 2*out['2B'] + 3*out['3B'] + 4*out['HR']
    woba_num = (WOBA_WEIGHTS['1B']*out['1B'] + WOBA_WEIGHTS['2B']*out['2B']
                + WOBA_WEIGHTS['3B']*out['3B'] + WOBA_WEIGHTS['HR']*out['HR'])
    n = out['total'].where(out['total'] > 0, other=1)
    out['AVG']  = (hits / n).round(3)
    out['SLG']  = (tb / n).round(3)
    out['wOBA'] = (woba_num / n).round(3)
    out['TB']   = tb.astype(int)
    out.loc[out['total'] == 0, ['AVG', 'SLG', 'wOBA']] = 0.0
    return out


@st.cache_data(show_spinner=False)
def _load_pbp(sport: str, division: str) -> pd.DataFrame:
    """Load events PBP for sport+division. Schema is the chart-builder
    naming: {sport}_play_by_play_{division}.csv (or .csv.gz on Render —
    raw .csv is gitignored at ~330MB; only .gz reaches Streamlit Cloud)."""
    cols = ['gameId', 'date', 'battingTeam', 'fieldingTeam',
            'player', 'playerId', 'playResult', 'hitLocation', 'inning']
    # Prefer raw .csv (faster), fall back to .gz which is what Render has
    csv_p = PBP_DIR / f'{sport}_play_by_play_{division}.csv'
    gz_p  = PBP_DIR / f'{sport}_play_by_play_{division}.csv.gz'
    if csv_p.exists():
        return pd.read_csv(csv_p, usecols=lambda c: c in cols, low_memory=False)
    if gz_p.exists():
        return pd.read_csv(gz_p, usecols=lambda c: c in cols, low_memory=False, compression='gzip')
    return pd.DataFrame()


def compute_spray_distribution(
    sport: str,
    division: str,
    team_name: str | None = None,
    player_id: str | None = None,
) -> pd.DataFrame:
    """Return a per-zone count table for the selected scope.

    Filters:
      team_name (optional): restrict to plays where battingTeam matches
      player_id (optional): restrict to plays by this batter (overrides team)

    Output columns: hitLocation, zone_name, total, plus one column per
    HIT_RESULT_BUCKETS bucket (1B, 2B, 3B, HR, Out, FC/Err, Other) and a
    pct column = total / sum(total).
    """
    df = _load_pbp(sport, division)
    if df.empty:
        return pd.DataFrame()

    # Filter scope. Use IDs / exact equality only — never substring on
    # team names (would match Texas/Texas A&M/Texas State/Texas Tech).
    if player_id:
        target = str(int(float(player_id))) if player_id else ''
        df_pid = pd.to_numeric(df['playerId'], errors='coerce')
        df_pid_str = df_pid.where(df_pid.notna(), other=pd.NA).astype('Int64').astype(str)
        df = df[df_pid_str == target]
    elif team_name:
        # Exact equality — team_name is the canonical NCAA name from list_teams.
        df = df[df['battingTeam'].astype(str) == str(team_name)]

    # Restrict to balls in play (hitLocation populated)
    df = df.dropna(subset=['hitLocation']).copy()
    if df.empty:
        return pd.DataFrame()

    df['hitLocation'] = df['hitLocation'].astype(int)
    df['result_bucket'] = df['playResult'].apply(categorize_result)

    # Pivot: rows = zone, columns = result bucket, values = count
    pv = df.pivot_table(index='hitLocation', columns='result_bucket',
                        values='gameId', aggfunc='count', fill_value=0)
    pv['total'] = pv.sum(axis=1)
    pv = pv.reset_index()
    pv['zone_name'] = pv['hitLocation'].map(ZONE_NAMES)

    grand_total = pv['total'].sum()
    pv['pct'] = (100 * pv['total'] / grand_total).round(1) if grand_total > 0 else 0.0

    # Order columns predictably
    bucket_cols = [c for c in ['1B', '2B', '3B', 'HR', 'Out', 'FC/Err', 'Other'] if c in pv.columns]
    pv = pv[['hitLocation', 'zone_name', 'total', 'pct'] + bucket_cols]
    return pv.sort_values('hitLocation').reset_index(drop=True)


def compute_field_side_buckets(spray_df: pd.DataFrame) -> dict:
    """Roll up zone-level spray data into Left / Middle / Right side splits.
    Without batter handedness we can't call this Pull/Center/Oppo; field-side
    is the unbiased label.

    Left side  (3B/SS/LF + LCF):  zones 5, 6, 7, 10
    Middle     (P/2B/CF + deep CF): zones 1, 4, 8, 14
    Right side (1B/RF + RCF):       zones 3, 9, 11
    Other / foul:                   zones 2, 12, 13
    """
    if spray_df.empty:
        return {'left': 0, 'middle': 0, 'right': 0, 'other': 0,
                'left_pct': 0, 'middle_pct': 0, 'right_pct': 0, 'other_pct': 0,
                'total': 0}
    LEFT  = {5, 6, 7, 10}
    MID   = {1, 4, 8, 14}
    RIGHT = {3, 9, 11}
    OTHER = {2, 12, 13}
    s = spray_df.set_index('hitLocation')['total']
    left = int(s[s.index.isin(LEFT)].sum())
    middle = int(s[s.index.isin(MID)].sum())
    right = int(s[s.index.isin(RIGHT)].sum())
    other = int(s[s.index.isin(OTHER)].sum())
    total = left + middle + right + other
    f = lambda v: round(100 * v / total, 1) if total > 0 else 0.0
    return {
        'left': left, 'middle': middle, 'right': right, 'other': other,
        'left_pct': f(left), 'middle_pct': f(middle),
        'right_pct': f(right), 'other_pct': f(other), 'total': total,
    }


@st.cache_data(show_spinner=False)
def list_teams(sport: str, division: str) -> pd.DataFrame:
    """Return DataFrame of (ncaa_team, team_id, bip) — every team that
    appears in the PBP file, with its chart-builder team_id resolved
    when possible. Filtering uses ncaa_team for exact equality (the PBP
    file's canonical name); team_id is shown in the dropdown so the user
    sees an unambiguous handle."""
    df = _load_pbp(sport, division)
    if df.empty:
        return pd.DataFrame(columns=['ncaa_team','team_id','bip'])
    df = df.dropna(subset=['hitLocation', 'battingTeam']).copy()
    bip = df.groupby('battingTeam').size().reset_index(name='bip')
    bip = bip.rename(columns={'battingTeam': 'ncaa_team'})

    # Best-effort team_id lookup via chart-builder teams.csv. NCAA names in
    # PBP can include mascots ("Texas Longhorns") while teams.csv uses short
    # names ("Texas"). Match by short-name-prefix.
    teams_csv = _load_csv('teams.csv')
    sport_label = 'Baseball' if sport.lower() == 'baseball' else 'Softball'
    cb = teams_csv[teams_csv['sport'] == sport_label][['id','name']].copy() if not teams_csv.empty else pd.DataFrame()
    name_to_id = {}
    if not cb.empty:
        # Sort by name length DESC so "Texas A&M" matches before "Texas"
        cb_sorted = cb.assign(_len=cb['name'].str.len()).sort_values('_len', ascending=False)
        for _, r in cb_sorted.iterrows():
            name_to_id[str(r['name'])] = int(r['id'])

    def find_id(ncaa: str) -> int | None:
        # Exact prefix match — pick the longest short-name that prefixes the NCAA full name
        for short, tid in name_to_id.items():
            if ncaa.startswith(short):
                return tid
        return None

    bip['team_id'] = bip['ncaa_team'].apply(find_id)
    return bip.sort_values('ncaa_team').reset_index(drop=True)


@st.cache_data(show_spinner=False)
def list_players(sport: str, division: str, team_name: str | None = None) -> pd.DataFrame:
    """Distinct (playerId, player, balls_in_play) per team. Sorted by BIP desc.

    Dedup by playerId only — NCAA play descriptions sometimes spell the same
    player two ways ('Kozeal' vs 'Kozeal,cam'). Pick the most-common spelling
    for the dropdown label and sum balls_in_play across all spellings."""
    df = _load_pbp(sport, division)
    if df.empty:
        return pd.DataFrame()
    if team_name:
        df = df[df['battingTeam'].astype(str).str.contains(team_name, case=False, na=False)]
    df = df.dropna(subset=['hitLocation', 'playerId']).copy()
    if df.empty:
        return pd.DataFrame()
    df['playerId'] = pd.to_numeric(df['playerId'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['playerId'])
    df['playerId'] = df['playerId'].astype(int).astype(str)

    # For each playerId, pick the most-frequent player-name string
    name_counts = df.groupby(['playerId','player']).size().reset_index(name='n')
    best_name = name_counts.sort_values('n', ascending=False).drop_duplicates('playerId')[['playerId','player']]
    bip = df.groupby('playerId').size().reset_index(name='balls_in_play')
    out = bip.merge(best_name, on='playerId', how='left')
    return out.sort_values('balls_in_play', ascending=False)[['playerId','player','balls_in_play']].reset_index(drop=True)
