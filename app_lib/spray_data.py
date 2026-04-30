"""
Spray-chart data layer.

PBP rows have a `hitLocation` column with a baseball position code:
  1=P, 2=C, 3=1B, 4=2B, 5=3B, 6=SS, 7=LF, 8=CF, 9=RF
  10=LCF gap, 11=RCF gap
  12=down the LF line (FAIR — not foul; NCAA scorebook quirk)
  13=down the RF line (FAIR)
  14=deep CF / over-the-fence CF (FAIR)
The names "L-Foul" / "R-Foul" some scorebooks use for 12/13 are misleading
— they're balls that stayed fair but landed close to the line, often XBH.

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
    10: 'LCF', 11: 'RCF',
    12: 'L Line', 13: 'R Line', 14: 'Deep CF',
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
    raw .csv is gitignored at ~330MB; only .gz reaches Streamlit Cloud).

    Loads enough columns to support the filter set: pitcher handedness
    join key (pitcherId), 2-out (outs), RISP (runner2B/3B), and 2-strike
    counts (strikes1..15)."""
    base_cols = ['gameId', 'date', 'battingTeam', 'fieldingTeam',
                 'player', 'playerId', 'pitcherId', 'playResult',
                 'hitLocation', 'inning', 'outs', 'runner2B', 'runner3B']
    strikes_cols = [f'strikes{i}' for i in range(1, 16)]
    cols = set(base_cols + strikes_cols)
    csv_p = PBP_DIR / f'{sport}_play_by_play_{division}.csv'
    gz_p  = PBP_DIR / f'{sport}_play_by_play_{division}.csv.gz'
    if csv_p.exists():
        return pd.read_csv(csv_p, usecols=lambda c: c in cols, low_memory=False)
    if gz_p.exists():
        return pd.read_csv(gz_p, usecols=lambda c: c in cols, low_memory=False, compression='gzip')
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _ncaa_pid_to_cb_player() -> pd.DataFrame:
    """Bridge from PBP playerId (NCAA season-specific) to chart-builder
    player record. PBP uses NCAA's season ID (8-9 digit), players.csv
    uses CB's own integer PK; rosters.csv carries both columns. Returns
    a DataFrame with [ncaa_pid, cb_id, position, classification, throw,
    bat, player_name] for the latest roster year."""
    rosters = _load_csv('rosters.csv')
    players = _load_csv('players.csv')
    if rosters.empty or players.empty:
        return pd.DataFrame(columns=['ncaa_pid', 'cb_id', 'position',
                                     'classification', 'throw', 'bat', 'player_name'])
    rosters['Year'] = pd.to_numeric(rosters['Year'], errors='coerce')
    current_year = int(rosters['Year'].max())
    r = rosters[rosters['Year'] == current_year][['player_id', 'player_ncaa_season_id']].copy()
    r['ncaa_pid'] = pd.to_numeric(r['player_ncaa_season_id'], errors='coerce').astype('Int64')
    r = r.dropna(subset=['ncaa_pid', 'player_id'])
    keep = ['id'] + [c for c in ['player_name', 'position', 'classification',
                                  'throw', 'bat'] if c in players.columns]
    j = r.merge(players[keep], left_on='player_id', right_on='id', how='left')
    j = j.rename(columns={'id': 'cb_id'})
    return j[['ncaa_pid', 'cb_id'] + keep[1:]]


@st.cache_data(show_spinner=False)
def _pitcher_throw_lookup(sport: str) -> dict:
    """{ncaa_pid (int): 'L'|'R'|'B'} for vs-LHP / vs-RHP filtering."""
    bridge = _ncaa_pid_to_cb_player()
    if bridge.empty or 'throw' not in bridge.columns:
        return {}
    sub = bridge[bridge['throw'].isin(['L', 'R', 'B'])].dropna(subset=['ncaa_pid'])
    return dict(zip(sub['ncaa_pid'].astype(int), sub['throw']))


def _strikes_at_contact(df: pd.DataFrame) -> pd.Series:
    """For each row, the strikes count immediately before the contact pitch
    (i.e., the strikes value at the highest non-null pitch index). Vectorized
    by walking the strikesN columns from 15 down to 1."""
    out = pd.Series(pd.NA, index=df.index, dtype='Int64')
    for n in range(15, 0, -1):
        col = f'strikes{n}'
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors='coerce')
        mask = out.isna() & vals.notna()
        if mask.any():
            out.loc[mask] = vals.loc[mask].astype('Int64')
    return out


def compute_spray_distribution(
    sport: str,
    division: str,
    team_name: str | None = None,
    player_id: str | None = None,
    *,
    vs_hand: str | None = None,    # 'L', 'R', or None
    two_strikes: bool = False,
    two_outs: bool = False,
    risp: bool = False,
) -> pd.DataFrame:
    """Return a per-zone count table for the selected scope.

    Scope filters (mutually exclusive):
      team_name: restrict to plays where battingTeam matches
      player_id: restrict to plays by this batter (overrides team)

    Situational filters (compose):
      vs_hand     'L' or 'R' to keep only plays vs that pitcher hand
      two_strikes True keeps only plays where the contact pitch was at 2 strikes
      two_outs    True keeps only plays where the play started with 2 outs
      risp        True keeps only plays with a runner on 2B and/or 3B

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

    # Situational filters — applied AFTER scope to keep the BIP universe stable
    if vs_hand in ('L', 'R') and 'pitcherId' in df.columns:
        throws = _pitcher_throw_lookup(sport)
        if throws:
            pid = pd.to_numeric(df['pitcherId'], errors='coerce').astype('Int64')
            mapped = pid.map(throws)
            df = df[mapped == vs_hand]
    if two_outs and 'outs' in df.columns:
        df = df[pd.to_numeric(df['outs'], errors='coerce') == 2]
    if risp:
        # runner2B / runner3B are 0.0 / 1.0 floats — empty bases are 0, not NaN.
        on_2b = pd.to_numeric(df['runner2B'], errors='coerce') > 0 if 'runner2B' in df.columns else False
        on_3b = pd.to_numeric(df['runner3B'], errors='coerce') > 0 if 'runner3B' in df.columns else False
        df = df[on_2b | on_3b]
    if two_strikes:
        df = df.assign(_strk=_strikes_at_contact(df))
        df = df[df['_strk'] == 2].drop(columns=['_strk'])
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

    Left side  (3B/SS/LF/LCF/L Line):   zones 5, 6, 7, 10, 12
    Middle     (P/C/2B/CF/Deep CF):     zones 1, 2, 4, 8, 14
    Right side (1B/RF/RCF/R Line):      zones 3, 9, 11, 13
    Catcher (zone 2) is folded into Middle so the L/M/R splits add up
    to 100%. Other stays empty as a reserved bucket.
    """
    if spray_df.empty:
        return {'left': 0, 'middle': 0, 'right': 0, 'other': 0,
                'left_pct': 0, 'middle_pct': 0, 'right_pct': 0, 'other_pct': 0,
                'total': 0}
    LEFT  = {5, 6, 7, 10, 12}
    MID   = {1, 2, 4, 8, 14}
    RIGHT = {3, 9, 11, 13}
    OTHER = set()
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
    id_to_short = {}
    if not cb.empty:
        # Sort by name length DESC so "Texas A&M" matches before "Texas"
        cb_sorted = cb.assign(_len=cb['name'].str.len()).sort_values('_len', ascending=False)
        for _, r in cb_sorted.iterrows():
            name_to_id[str(r['name'])] = int(r['id'])
            id_to_short[int(r['id'])] = str(r['name'])

    def find_id(ncaa: str) -> int | None:
        # Exact prefix match — pick the longest short-name that prefixes the NCAA full name
        for short, tid in name_to_id.items():
            if ncaa.startswith(short):
                return tid
        return None

    bip['team_id'] = bip['ncaa_team'].apply(find_id)
    bip['short_name'] = bip['team_id'].apply(
        lambda tid: id_to_short.get(int(tid)) if pd.notna(tid) else None
    )
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

    # Join through rosters.csv (PBP playerId -> NCAA season ID -> CB id ->
    # players.csv position/classification). PBP and players.csv use
    # different ID systems — rosters.csv is the bridge.
    bridge = _ncaa_pid_to_cb_player()
    if not bridge.empty:
        b = bridge.dropna(subset=['ncaa_pid']).copy()
        b['playerId'] = b['ncaa_pid'].astype(int).astype(str)
        keep = ['playerId', 'cb_id'] + [c for c in ['player_name', 'position', 'classification']
                                         if c in b.columns]
        b = b[keep].drop_duplicates('playerId')
        out = out.merge(b, on='playerId', how='left')
    for col in ('cb_id', 'player_name', 'position', 'classification'):
        if col not in out.columns:
            out[col] = pd.NA

    return out.sort_values('balls_in_play', ascending=False)[
        ['playerId','cb_id','player','player_name','balls_in_play','position','classification']
    ].reset_index(drop=True)
