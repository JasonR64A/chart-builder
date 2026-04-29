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


@st.cache_data(show_spinner=False)
def _load_pbp(sport: str, division: str) -> pd.DataFrame:
    """Load events PBP for sport+division. Schema is the chart-builder
    naming: {sport}_play_by_play_{division}.csv."""
    p = PBP_DIR / f'{sport}_play_by_play_{division}.csv'
    if not p.exists():
        return pd.DataFrame()
    cols = ['gameId', 'date', 'battingTeam', 'fieldingTeam',
            'player', 'playerId', 'playResult', 'hitLocation', 'inning']
    df = pd.read_csv(p, usecols=lambda c: c in cols, low_memory=False)
    return df


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

    # Filter scope
    if player_id:
        df = df[df['playerId'].astype(str) == str(player_id)]
    elif team_name:
        df = df[df['battingTeam'].astype(str).str.contains(team_name, case=False, na=False)]

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
def list_teams(sport: str, division: str) -> list[str]:
    """All team names that appear as battingTeam in the PBP file."""
    df = _load_pbp(sport, division)
    if df.empty:
        return []
    return sorted(df['battingTeam'].dropna().astype(str).unique().tolist())


@st.cache_data(show_spinner=False)
def list_players(sport: str, division: str, team_name: str | None = None) -> pd.DataFrame:
    """Distinct (player, playerId, balls_in_play) per team. Sorted by BIP desc."""
    df = _load_pbp(sport, division)
    if df.empty:
        return pd.DataFrame()
    if team_name:
        df = df[df['battingTeam'].astype(str).str.contains(team_name, case=False, na=False)]
    df = df.dropna(subset=['hitLocation', 'playerId']).copy()
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(['playerId', 'player']).size().reset_index(name='balls_in_play')
    g['playerId'] = g['playerId'].astype(int).astype(str)
    return g.sort_values('balls_in_play', ascending=False).reset_index(drop=True)
