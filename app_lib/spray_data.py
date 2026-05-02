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

    Loads enough columns to support the full state-filter set: handedness
    bridges (pitcherId), runners (runner1B/2B/3B), inning + outs, balls
    and strikes per pitch (for count-at-contact), and away/home score +
    awayTeam (for run-differential from the batter's POV)."""
    base_cols = ['gameId', 'date', 'battingTeam', 'fieldingTeam',
                 'awayTeam', 'homeTeam', 'awayScore', 'homeScore',
                 'player', 'playerId', 'pitcherId', 'playResult',
                 'hitLocation', 'inning', 'outs',
                 'runner1B', 'runner2B', 'runner3B']
    strikes_cols = [f'strikes{i}' for i in range(1, 16)]
    balls_cols   = [f'balls{i}'   for i in range(1, 16)]
    cols = set(base_cols + strikes_cols + balls_cols)
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


@st.cache_data(show_spinner=False)
def _verified_pitcher_ids(sport: str, division: str) -> set:
    """Set of NCAA player_ids that appear in the box-score pitching PBP file
    (pbp_data/{sport}/pitching_pbp_{division}.csv) — i.e., players whose
    team-stats scrape attributed actual innings pitched. Used to filter out
    misattributed pitchers in the event-level play_by_play file (NCAA
    scorebook operators occasionally type the wrong name; a position player
    can show up as the listed pitcher for an entire game even though they
    have zero box-score IP). Cross-referencing event-level pitcherId against
    this set keeps phantom pitchers like a DH "throwing 8 games" out of the
    spray-chart pitcher list."""
    p = Path(__file__).resolve().parent.parent / 'pbp_data' / sport / f'pitching_pbp_{division}.csv'
    if not p.exists():
        return set()
    try:
        df = pd.read_csv(p, low_memory=False, usecols=['playerId'])
    except Exception:
        return set()
    pids = pd.to_numeric(df['playerId'], errors='coerce').dropna().astype(int)
    return set(pids.unique().tolist())


@st.cache_data(show_spinner=False)
def _batter_bat_lookup(sport: str) -> dict:
    """{ncaa_pid (int): 'L'|'R'|'B'} for vs-LHB / vs-RHB filtering on the
    pitching-perspective spray chart."""
    bridge = _ncaa_pid_to_cb_player()
    if bridge.empty or 'bat' not in bridge.columns:
        return {}
    sub = bridge[bridge['bat'].isin(['L', 'R', 'B'])].dropna(subset=['ncaa_pid'])
    return dict(zip(sub['ncaa_pid'].astype(int), sub['bat']))


def _count_at_contact(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Generic count-at-contact extractor. For each row, returns the value
    at the highest-index non-null `{prefix}N` column. Vectorized by walking
    columns from 15 down to 1. Used for both balls and strikes counts."""
    out = pd.Series(pd.NA, index=df.index, dtype='Int64')
    for n in range(15, 0, -1):
        col = f'{prefix}{n}'
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors='coerce')
        mask = out.isna() & vals.notna()
        if mask.any():
            out.loc[mask] = vals.loc[mask].astype('Int64')
    return out


def _strikes_at_contact(df: pd.DataFrame) -> pd.Series:
    return _count_at_contact(df, 'strikes')


def _balls_at_contact(df: pd.DataFrame) -> pd.Series:
    return _count_at_contact(df, 'balls')


@st.cache_data(show_spinner=False)
def _opp_rank_lookup(sport: str, role: str) -> dict:
    """{ncaa_pid (int): integer_rank (1..N)} for filtering by the opponent
    player's quality. role='pitcher' uses the pitcher percentile column
    (percentile_rank_weighted_run_allowed_efficiency); role='batter' uses
    the hitter percentile column (percentile_rank_weighted_run_created_efficiency).
    Players with percentile == 0 are treated as unranked and omitted from
    the map — so any specific bucket filter (1-100, etc.) excludes plays
    against unranked opponents.

    Integer rank is computed within (sport, role, year=2026) by sorting
    percentile DESC and numbering 1..N. So rank 1 is the best hitter / best
    pitcher in that sport-year."""
    pr = _load_csv('player_rank.csv')
    if pr.empty:
        return {}
    pr['year'] = pd.to_numeric(pr['year'], errors='coerce')
    pr = pr[pr['year'] == 2026].copy()
    if pr.empty:
        return {}

    teams_csv = _load_csv('teams.csv')
    sport_label = 'Baseball' if sport.lower() == 'baseball' else 'Softball'
    sport_team_ids = set(pd.to_numeric(
        teams_csv[teams_csv['sport'] == sport_label]['id'], errors='coerce'
    ).dropna().astype(int)) if not teams_csv.empty else set()
    pr['team_id_int'] = pd.to_numeric(pr['team_id'], errors='coerce').astype('Int64')
    pr = pr[pr['team_id_int'].isin(sport_team_ids)]
    if pr.empty:
        return {}

    col = ('percentile_rank_weighted_run_allowed_efficiency' if role == 'pitcher'
           else 'percentile_rank_weighted_run_created_efficiency')
    if col not in pr.columns:
        return {}
    pr['_pct'] = pd.to_numeric(pr[col], errors='coerce')
    pr = pr.dropna(subset=['_pct', 'player_id'])
    pr = pr[pr['_pct'] > 0]
    if pr.empty:
        return {}
    # Sort DESC: highest percentile = best = rank 1
    pr = pr.sort_values('_pct', ascending=False).reset_index(drop=True)
    pr['_rank'] = pr.index + 1
    cb_to_rank = dict(zip(pd.to_numeric(pr['player_id'], errors='coerce').astype(int),
                          pr['_rank'].astype(int)))

    bridge = _ncaa_pid_to_cb_player()
    if bridge.empty:
        return {}
    bb = bridge.dropna(subset=['cb_id', 'ncaa_pid']).copy()
    bb['cb_id'] = bb['cb_id'].astype(int)
    bb['ncaa_pid'] = bb['ncaa_pid'].astype(int)
    return {int(r.ncaa_pid): cb_to_rank[int(r.cb_id)]
            for r in bb.itertuples()
            if int(r.cb_id) in cb_to_rank}


# Constants used by both the data layer and the UI. Centralized so the page
# enum-style controls and the filter logic can't drift.
RUNNER_OPTIONS = ('Any', 'Empty', 'Occupied')
RISP_OPTIONS   = ('Any', 'Yes', 'No')
RUN_DIFF_OPTIONS = ('Any', 'Losing', 'Tied', 'Winning')
RANK_BUCKET_OPTIONS = ('Any', '1-100', '101-200', '201-500', '500+')
INNING_VALUES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 'Extras')


def compute_spray_distribution(
    sport: str,
    division: str,
    team_name: str | None = None,
    player_id: str | None = None,
    *,
    perspective: str = 'hitting',  # 'hitting' or 'pitching'
    vs_hand: str | None = None,    # 'L', 'R', or None
    runner1b: str = 'Any',         # 'Any' | 'Empty' | 'Occupied'
    runner2b: str = 'Any',
    runner3b: str = 'Any',
    risp: str = 'Any',             # 'Any' | 'Yes' | 'No' (2B or 3B occupied)
    balls: str | int = 'Any',      # 'Any' | 0 | 1 | 2 | 3
    strikes: str | int = 'Any',    # 'Any' | 0 | 1 | 2
    innings: tuple | list | None = None,  # subset of INNING_VALUES, None = all
    run_diff: str = 'Any',         # 'Any' | 'Losing' | 'Tied' | 'Winning'
    opp_rank_bucket: str = 'Any',  # one of RANK_BUCKET_OPTIONS
) -> pd.DataFrame:
    """Return a per-zone count table for the selected scope.

    Perspective:
      'hitting'  — "where does this batter / batting team put the ball?"
                   Scope filters target battingTeam + batter playerId; the
                   vs_hand filter targets the OPPONENT pitcher's throw side.
      'pitching' — "where do hitters put the ball against this pitcher /
                   pitching staff?" Scope filters target fieldingTeam +
                   pitcherId; vs_hand targets the OPPONENT batter's bat side.

    Scope filters (mutually exclusive):
      team_name: restrict to plays where the relevant team (battingTeam in
                 hitting, fieldingTeam in pitching) matches.
      player_id: restrict to plays by this player (batter in hitting,
                 pitcher in pitching). Overrides team_name.

    State filters (all default to 'Any' / no-op; compose):
      vs_hand          'L'/'R' — opp pitcher throw (hitting) or opp batter
                       bat (pitching).
      runner1b/2b/3b   'Empty' or 'Occupied' on that base.
      risp             'Yes' = runner on 2B and/or 3B; 'No' = neither.
      balls / strikes  Pre-contact count from the per-pitch count columns.
      innings          Iterable of inning numbers (1..9 ints) and/or the
                       string 'Extras' for innings >= 10.
      run_diff         'Losing'/'Tied'/'Winning' from the BATTING team's
                       point of view (sign computed from awayTeam check).
      opp_rank_bucket  Bucket of the opponent player's integer rank in
                       chart-builder/data/player_rank.csv year=2026 within
                       sport. 'pitching' POV ranks opposing batters by
                       hitter percentile; 'hitting' POV ranks opposing
                       pitchers by pitcher percentile.

    Output columns: hitLocation, zone_name, total, plus one column per
    HIT_RESULT_BUCKETS bucket (1B, 2B, 3B, HR, Out, FC/Err, Other) and a
    pct column = total / sum(total).
    """
    df = _load_pbp(sport, division)
    if df.empty:
        return pd.DataFrame()

    # Resolve perspective-dependent column names so the rest of the function
    # treats them uniformly. Hitting POV: scope = battingTeam / playerId,
    # opponent-hand lookup = pitcher throw. Pitching POV: scope =
    # fieldingTeam / pitcherId, opponent-hand lookup = batter bat side.
    if perspective == 'pitching':
        team_col = 'fieldingTeam'
        player_col = 'pitcherId'
        opp_player_col = 'playerId'
        hand_lookup = _batter_bat_lookup(sport)
    else:
        team_col = 'battingTeam'
        player_col = 'playerId'
        opp_player_col = 'pitcherId'
        hand_lookup = _pitcher_throw_lookup(sport)

    # Filter scope. Use IDs / exact equality only — never substring on
    # team names (would match Texas/Texas A&M/Texas State/Texas Tech).
    if player_id:
        target = str(int(float(player_id))) if player_id else ''
        df_pid = pd.to_numeric(df[player_col], errors='coerce')
        df_pid_str = df_pid.where(df_pid.notna(), other=pd.NA).astype('Int64').astype(str)
        df = df[df_pid_str == target]
    elif team_name:
        # Exact equality — team_name is the canonical NCAA name from list_teams.
        df = df[df[team_col].astype(str) == str(team_name)]

    # Restrict to balls in play (hitLocation populated)
    df = df.dropna(subset=['hitLocation']).copy()
    if df.empty:
        return pd.DataFrame()

    # Pitching POV: drop rows where the listed pitcherId doesn't appear in
    # the box-score pitching file. Otherwise NCAA scorebook misattributions
    # (DH listed as the pitcher for an entire game) leak into team-level
    # spray totals.
    if perspective == 'pitching' and 'pitcherId' in df.columns:
        verified = _verified_pitcher_ids(sport, division)
        if verified:
            pid_int = pd.to_numeric(df['pitcherId'], errors='coerce')
            df = df[pid_int.isin(verified)]
            if df.empty:
                return pd.DataFrame()

    # State filters — applied AFTER scope to keep the BIP universe stable
    if vs_hand in ('L', 'R') and opp_player_col in df.columns:
        if hand_lookup:
            pid = pd.to_numeric(df[opp_player_col], errors='coerce').astype('Int64')
            mapped = pid.map(hand_lookup)
            df = df[mapped == vs_hand]

    # Runner on bases — runnerXB columns are 0.0/1.0 floats (empty bases = 0,
    # not NaN, so a missing column means the base never matters).
    def _runner_mask(col, state):
        if state == 'Any' or col not in df.columns:
            return None
        on_base = pd.to_numeric(df[col], errors='coerce') > 0
        return on_base if state == 'Occupied' else ~on_base
    for col, state in [('runner1B', runner1b), ('runner2B', runner2b), ('runner3B', runner3b)]:
        m = _runner_mask(col, state)
        if m is not None:
            df = df[m]

    # RISP — runner on 2B and/or 3B
    if risp in ('Yes', 'No'):
        on_2b = pd.to_numeric(df['runner2B'], errors='coerce') > 0 if 'runner2B' in df.columns else False
        on_3b = pd.to_numeric(df['runner3B'], errors='coerce') > 0 if 'runner3B' in df.columns else False
        in_scoring = on_2b | on_3b
        df = df[in_scoring] if risp == 'Yes' else df[~in_scoring]

    # Balls / strikes count at contact
    if balls != 'Any':
        df = df.assign(_b=_balls_at_contact(df))
        df = df[df['_b'] == int(balls)].drop(columns=['_b'])
    if strikes != 'Any':
        df = df.assign(_s=_strikes_at_contact(df))
        df = df[df['_s'] == int(strikes)].drop(columns=['_s'])

    # Innings — set of allowed inning ints + optional 'Extras' (>=10)
    if innings:
        inn_int = pd.to_numeric(df['inning'], errors='coerce') if 'inning' in df.columns else None
        if inn_int is not None:
            num_set = {int(v) for v in innings if isinstance(v, int) or (isinstance(v, str) and v.isdigit())}
            extras = any(isinstance(v, str) and v.lower() == 'extras' for v in innings)
            mask = inn_int.isin(num_set) if num_set else pd.Series(False, index=df.index)
            if extras:
                mask = mask | (inn_int >= 10)
            df = df[mask]

    # Run differential from batting team's POV
    if run_diff != 'Any' and all(c in df.columns for c in ['awayScore','homeScore','battingTeam','awayTeam']):
        away = pd.to_numeric(df['awayScore'], errors='coerce')
        home = pd.to_numeric(df['homeScore'], errors='coerce')
        is_away = df['battingTeam'].astype(str) == df['awayTeam'].astype(str)
        diff = away.where(is_away, other=home) - home.where(is_away, other=away)
        if run_diff == 'Losing':
            df = df[diff < 0]
        elif run_diff == 'Tied':
            df = df[diff == 0]
        elif run_diff == 'Winning':
            df = df[diff > 0]

    # Opponent rank bucket
    if opp_rank_bucket != 'Any' and opp_player_col in df.columns:
        opp_role = 'pitcher' if perspective == 'hitting' else 'batter'
        rank_map = _opp_rank_lookup(sport, opp_role)
        if rank_map:
            opp_pid = pd.to_numeric(df[opp_player_col], errors='coerce').astype('Int64')
            ranks = opp_pid.map(rank_map)
            if opp_rank_bucket == '1-100':
                df = df[(ranks >= 1) & (ranks <= 100)]
            elif opp_rank_bucket == '101-200':
                df = df[(ranks >= 101) & (ranks <= 200)]
            elif opp_rank_bucket == '201-500':
                df = df[(ranks >= 201) & (ranks <= 500)]
            elif opp_rank_bucket == '500+':
                df = df[ranks > 500]
        else:
            # No rank lookup available → no plays match a specific bucket
            df = df.iloc[0:0]

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


@st.cache_data(show_spinner=False)
def list_pitching_teams(sport: str, division: str) -> pd.DataFrame:
    """Pitching-perspective mirror of list_teams. Counts BIP allowed by
    each fieldingTeam (i.e., balls put in play against their pitching staff)
    and resolves chart-builder team_id the same way.

    Applies the same verified-pitcher gate as list_pitchers and
    compute_spray_distribution so the dropdown count matches the chart's
    actual BIP-allowed total. Without the gate, NCAA scorebook
    misattributions (a SS typed into the pitcher slot for ~6 games) inflate
    the sidebar count vs the chart — Minnesota showed 1,031 BIP in the
    dropdown but 665 in the chart, the 366-row gap was Jack Spanier (SS)
    listed as pitcher in 346 events that aren't in the box-score file."""
    df = _load_pbp(sport, division)
    if df.empty:
        return pd.DataFrame(columns=['ncaa_team', 'team_id', 'bip', 'short_name'])
    df = df.dropna(subset=['hitLocation', 'fieldingTeam']).copy()
    verified = _verified_pitcher_ids(sport, division)
    if verified and 'pitcherId' in df.columns:
        pid_int = pd.to_numeric(df['pitcherId'], errors='coerce')
        df = df[pid_int.isin(verified)]
    bip = df.groupby('fieldingTeam').size().reset_index(name='bip')
    bip = bip.rename(columns={'fieldingTeam': 'ncaa_team'})

    teams_csv = _load_csv('teams.csv')
    sport_label = 'Baseball' if sport.lower() == 'baseball' else 'Softball'
    cb = teams_csv[teams_csv['sport'] == sport_label][['id', 'name']].copy() if not teams_csv.empty else pd.DataFrame()
    name_to_id, id_to_short = {}, {}
    if not cb.empty:
        cb_sorted = cb.assign(_len=cb['name'].str.len()).sort_values('_len', ascending=False)
        for _, r in cb_sorted.iterrows():
            name_to_id[str(r['name'])] = int(r['id'])
            id_to_short[int(r['id'])] = str(r['name'])

    def find_id(ncaa: str):
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
def list_pitchers(sport: str, division: str, team_name: str | None = None) -> pd.DataFrame:
    """Pitching-perspective mirror of list_players. Distinct
    (pitcherId, pitcher, balls_in_play_allowed) per team, sorted by
    balls-in-play allowed descending. Joins to chart-builder rosters
    bridge for cb_id / position / classification.

    Filters out pitcherIds that don't have any rows in the box-score
    pitching PBP file. The event-level play-by-play feed occasionally
    misattributes pitcher = some position player (NCAA scorebook
    operator error); without this gate a DH like Carter Rutenbar shows
    up at the top of Arkansas's pitcher list with 120 phantom BIP
    allowed. The box-score file is built from a separate team-stats
    scrape and only contains real pitchers."""
    df = _load_pbp(sport, division)
    if df.empty:
        return pd.DataFrame()
    if team_name:
        df = df[df['fieldingTeam'].astype(str) == str(team_name)]
    df = df.dropna(subset=['hitLocation', 'pitcherId']).copy()
    if df.empty:
        return pd.DataFrame()
    df['pitcherId'] = pd.to_numeric(df['pitcherId'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['pitcherId'])

    # Drop misattributed pitchers — keep only NCAA pids that appear in the
    # box-score pitching file.
    verified = _verified_pitcher_ids(sport, division)
    if verified:
        df = df[df['pitcherId'].astype(int).isin(verified)]
        if df.empty:
            return pd.DataFrame()

    df['pitcherId'] = df['pitcherId'].astype(int).astype(str)

    # The pitcher's name string isn't carried as a separate column on PBP
    # rows (the `player` field is the BATTER). The bridge to players.csv
    # gives us the pitcher's full name; if the bridge misses, fall back to
    # the pitcherId for the dropdown label.
    bip = df.groupby('pitcherId').size().reset_index(name='balls_in_play')

    bridge = _ncaa_pid_to_cb_player()
    if not bridge.empty:
        b = bridge.dropna(subset=['ncaa_pid']).copy()
        b['pitcherId'] = b['ncaa_pid'].astype(int).astype(str)
        keep = ['pitcherId', 'cb_id'] + [c for c in ['player_name', 'position', 'classification', 'throw']
                                          if c in b.columns]
        b = b[keep].drop_duplicates('pitcherId')
        bip = bip.merge(b, on='pitcherId', how='left')
    for col in ('cb_id', 'player_name', 'position', 'classification', 'throw'):
        if col not in bip.columns:
            bip[col] = pd.NA

    # `player` column kept for parity with list_players callers — show the
    # bridge name when available, else the bare pitcherId.
    bip['player'] = bip['player_name'].where(bip['player_name'].notna(),
                                              other=bip['pitcherId'])
    return bip.sort_values('balls_in_play', ascending=False)[
        ['pitcherId', 'cb_id', 'player', 'player_name', 'balls_in_play',
         'position', 'classification', 'throw']
    ].reset_index(drop=True)
