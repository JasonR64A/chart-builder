"""
Derive team-level hitting / pitching stats from PBP data with optional
game-context filters (Conference/Non-Conference, Weekend/Midweek).

Used by chart_builder.py when a filter is active. The pre-aggregated
{hitting,pitching}_team.csv files are season totals — they can't be sliced
by game context. This module re-aggregates from per-game PBP data.

Limitations:
- Only current year (2026) is derivable; historical years stay from the
  pre-aggregated CSV.
- Subset of columns: the basic raw + rate stats. Percentile ranks and
  advanced sabermetrics that depend on league context are not re-computed
  (left blank — chart-builder users plotting those should not enable the
  filter, or treat the filter as 'all' for those columns).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
PBP_DIR = Path(__file__).resolve().parent.parent / 'pbp_data'

# Filter values — keep in sync with chart_builder.py sidebar
GAME_TYPE_OPTIONS = ['All', 'Conference', 'Non-Conference']
DAY_OPTIONS = ['All', 'Weekend (Fri-Sun)', 'Midweek (Mon-Thu)']


def _filter_label(game_type: str, day_filter: str) -> str:
    """Compact label for display."""
    parts = []
    if game_type != 'All':
        parts.append(game_type)
    if day_filter != 'All':
        parts.append('Weekend' if day_filter.startswith('Weekend') else 'Midweek')
    return ' · '.join(parts) if parts else 'All games'


def is_active(game_type: str, day_filter: str) -> bool:
    return game_type != 'All' or day_filter != 'All'


@st.cache_data(show_spinner=False)
def _load_team_conf_map() -> dict[str, str]:
    """teamName -> conference_id (str). Used to determine same-conference games."""
    teams = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False)
    return {
        str(r['name']): str(int(r['conference_id']))
        for _, r in teams.iterrows()
        if pd.notna(r.get('conference_id'))
    }


@st.cache_data(show_spinner=False)
def _load_team_id_map() -> dict[tuple[str, str], int]:
    """(sport_lower, teamName) -> team_id. Used to attach team_id for chart-builder joins."""
    teams = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False)
    out = {}
    for _, r in teams.iterrows():
        sport_l = str(r.get('sport', '')).lower()
        out[(sport_l, str(r['name']))] = int(r['id'])
    return out


@st.cache_data(show_spinner=False)
def _load_team_division_map() -> dict[tuple[str, str], str]:
    """(sport_lower, teamName) -> division string (e.g. 'D-I'). Used to compute
    division-scoped percentile ranks so 'best WHIP team in D1' = 1.0."""
    teams = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False)
    confs = pd.read_csv(DATA_DIR / 'conferences.csv', low_memory=False)
    conf_to_div = dict(zip(confs['id'].astype(int), confs['division']))
    out = {}
    for _, r in teams.iterrows():
        if pd.notna(r.get('conference_id')):
            div = conf_to_div.get(int(r['conference_id']))
            if div:
                out[(str(r.get('sport', '')).lower(), str(r['name']))] = div
    return out


def _add_division_percentile_ranks(df: pd.DataFrame, sport: str, stat_directions: dict[str, str]) -> pd.DataFrame:
    """For each (stat_col, direction) in stat_directions, add a column
    percentile_rank_<stat_col> with values in [0.0, 1.0] computed WITHIN
    each division. direction='high' → higher value = 1.0; 'low' → lower = 1.0.

    Teams without a division (uncatalogued) get NaN. Single-team divisions
    get 0.5 (no spread).
    """
    div_map = _load_team_division_map()
    sport_l = sport.lower()
    df = df.copy()
    df['_division'] = df['teamName'].apply(lambda n: div_map.get((sport_l, n)))

    for col, direction in stat_directions.items():
        if col not in df.columns:
            continue
        ranks = pd.Series(index=df.index, dtype=float)
        for div, group in df.groupby('_division'):
            if pd.isna(div) or len(group) < 2:
                ranks.loc[group.index] = 0.5 if not pd.isna(div) else float('nan')
                continue
            vals = pd.to_numeric(group[col], errors='coerce')
            mn, mx = vals.min(), vals.max()
            if mn == mx:
                ranks.loc[group.index] = 0.5
                continue
            if direction == 'high':
                ranks.loc[group.index] = (vals - mn) / (mx - mn)
            else:  # 'low'
                ranks.loc[group.index] = (mx - vals) / (mx - mn)
        df[f'percentile_rank_{col}'] = ranks.round(3)

    df = df.drop(columns=['_division'])
    return df


def _select_pbp_paths(sport: str, stat_kind: str) -> list[Path]:
    """All 3 division PBP files for a sport+stat_kind."""
    return [PBP_DIR / sport / f'{stat_kind}_pbp_{div}.csv' for div in ('D1', 'D2', 'D3')]


@st.cache_data(show_spinner=False)
def _load_concat_pbp(sport: str, stat_kind: str) -> pd.DataFrame:
    parts = []
    for p in _select_pbp_paths(sport, stat_kind):
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _apply_filters(df: pd.DataFrame, sport: str, game_type: str, day_filter: str) -> pd.DataFrame:
    """Filter the concatenated PBP frame by game context."""
    if df.empty:
        return df
    df = df.copy()

    # Day-of-week filter (cheap; do first to shrink the set)
    if day_filter != 'All':
        df['_dow'] = pd.to_datetime(df['date'], errors='coerce', format='mixed').dt.weekday
        weekend = {4, 5, 6}  # Fri, Sat, Sun
        if day_filter.startswith('Weekend'):
            df = df[df['_dow'].isin(weekend)]
        else:
            df = df[~df['_dow'].isin(weekend) & df['_dow'].notna()]
        df = df.drop(columns=['_dow'])

    # Game-type filter (per-gameId: same-conf vs cross-conf)
    if game_type != 'All' and not df.empty:
        conf_map = _load_team_conf_map()
        # For each gameId, collect the two teams and check conferences
        game_teams = df.groupby('gameId')['teamName'].apply(lambda s: set(s.dropna().unique())).to_dict()
        conf_games, nonconf_games = set(), set()
        for gid, teams in game_teams.items():
            if len(teams) != 2:
                # Incomplete coverage — skip from BOTH buckets so it doesn't pollute either
                continue
            t1, t2 = list(teams)
            c1, c2 = conf_map.get(t1, ''), conf_map.get(t2, '')
            if c1 and c2 and c1 == c2:
                conf_games.add(gid)
            elif c1 and c2:  # Both teams have conferences but different
                nonconf_games.add(gid)
        keep = conf_games if game_type == 'Conference' else nonconf_games
        df = df[df['gameId'].isin(keep)]

    return df


def _aggregate_hitting(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """Roll up per-game player rows to per-team season totals.
    Returns columns matching the basic schema of hitting_team.csv."""
    if df.empty:
        return pd.DataFrame()
    sum_cols = ['ab', 'r', 'h', 'doubles', 'triples', 'tb', 'hr', 'rbi',
                'bb', 'hbp', 'sf', 'sh', 'k', 'sb', 'cs', 'ibb']
    sum_cols = [c for c in sum_cols if c in df.columns]
    grp = df.groupby('teamName', as_index=False)[sum_cols].sum()
    grp['games_played'] = df.groupby('teamName')['gameId'].nunique().reindex(grp['teamName']).values

    # Rate stats
    grp['plate_appearances'] = grp[['ab', 'bb', 'hbp', 'sf', 'sh']].sum(axis=1)
    grp['batting_average'] = (grp['h'] / grp['ab']).where(grp['ab'] > 0, 0).round(3)
    obp_denom = grp['ab'] + grp['bb'] + grp['hbp'] + grp['sf']
    grp['on_base_percentage'] = ((grp['h'] + grp['bb'] + grp['hbp']) / obp_denom).where(obp_denom > 0, 0).round(3)
    grp['slugging_percentage'] = (grp['tb'] / grp['ab']).where(grp['ab'] > 0, 0).round(3)
    grp['on_base_plus_slugging'] = (grp['on_base_percentage'] + grp['slugging_percentage']).round(3)
    grp['isolated_power'] = (grp['slugging_percentage'] - grp['batting_average']).round(3)
    babip_denom = grp['ab'] - grp['k'] - grp['hr'] + grp['sf']
    grp['batting_average_on_balls_in_play'] = ((grp['h'] - grp['hr']) / babip_denom).where(babip_denom > 0, 0).round(3)
    grp['strikeout_percentage'] = (grp['k'] / grp['plate_appearances'] * 100).where(grp['plate_appearances'] > 0, 0).round(2)
    grp['walk_percentage'] = (grp['bb'] / grp['plate_appearances'] * 100).where(grp['plate_appearances'] > 0, 0).round(2)
    grp['strikeout_to_walk_ratio'] = (grp['k'] / grp['bb']).where(grp['bb'] > 0, 0).round(2)
    grp['runs_plate_appearance'] = (grp['r'] / grp['plate_appearances']).where(grp['plate_appearances'] > 0, 0).round(3)

    # ── wOBA, wRC, wRAA (FanGraphs-style college constants) ──
    # College linear-weight wOBA constants, calibrated to historical NCAA seasons.
    WOBA_BB = 0.690; WOBA_HBP = 0.722; WOBA_1B = 0.888
    WOBA_2B = 1.271; WOBA_3B = 1.616; WOBA_HR = 2.101
    WOBA_SCALE = 1.157  # OBP/wOBA conversion factor
    grp['_singles'] = grp['h'] - grp['doubles'] - grp['triples'] - grp['hr']
    woba_denom = grp['ab'] + grp['bb'] - grp['ibb'] + grp['sf'] + grp['hbp']  # exclude IBB
    grp['weighted_on_base_average'] = (
        (WOBA_BB * (grp['bb'] - grp['ibb']) + WOBA_HBP * grp['hbp']
         + WOBA_1B * grp['_singles'] + WOBA_2B * grp['doubles']
         + WOBA_3B * grp['triples'] + WOBA_HR * grp['hr'])
        / woba_denom
    ).where(woba_denom > 0, 0).round(3)

    # League aggregates (across the FILTERED dataset — same population the user is viewing)
    league_woba_num = (
        WOBA_BB * (grp['bb'] - grp['ibb']).sum() + WOBA_HBP * grp['hbp'].sum()
        + WOBA_1B * grp['_singles'].sum() + WOBA_2B * grp['doubles'].sum()
        + WOBA_3B * grp['triples'].sum() + WOBA_HR * grp['hr'].sum()
    )
    league_woba_denom = (grp['ab'].sum() + grp['bb'].sum() - grp['ibb'].sum()
                        + grp['sf'].sum() + grp['hbp'].sum())
    league_woba = league_woba_num / league_woba_denom if league_woba_denom > 0 else 0
    league_pa = grp['plate_appearances'].sum()
    league_runs = grp['r'].sum()
    league_r_pa = league_runs / league_pa if league_pa > 0 else 0

    # wRAA = ((wOBA - league_wOBA) / wOBA_scale) * PA
    grp['weighted_runs_above_average'] = (
        (grp['weighted_on_base_average'] - league_woba) / WOBA_SCALE
    ) * grp['plate_appearances']
    grp['weighted_runs_above_average'] = grp['weighted_runs_above_average'].round(1)

    # wRC = wRAA + (league_R/PA * PA)  — i.e., team's runs above avg plus what an average team would have produced
    grp['weighted_runs_created'] = (
        grp['weighted_runs_above_average'] + league_r_pa * grp['plate_appearances']
    ).round(1)

    grp = grp.drop(columns=['_singles'])

    # Rename to canonical hitting_team.csv columns
    grp = grp.rename(columns={
        'r': 'runs_scored', 'ab': 'at_bats', 'h': 'hits',
        'tb': 'total_bases', 'hr': 'home_runs', 'rbi': 'runs_batted_in',
        'bb': 'walks', 'hbp': 'hit_by_pitch', 'sf': 'sac_fly', 'sh': 'sac_hit',
        'k': 'strikeouts', 'cs': 'caught_stealing', 'sb': 'stolen_bases',
        'ibb': 'intentional_walk',
    })
    if 'picked_off' not in grp.columns:
        grp['picked_off'] = 0  # not tracked at team-totals level

    # Division-scoped percentile ranks (1.0 = best in division for that stat)
    grp = _add_division_percentile_ranks(grp, sport, {
        'on_base_plus_slugging': 'high',
        'weighted_on_base_average': 'high',
        'weighted_runs_created': 'high',
        'weighted_runs_above_average': 'high',
        'isolated_power': 'high',
        'batting_average_on_balls_in_play': 'high',
        'strikeout_percentage': 'low',  # lower K% = better hitter
        'walk_percentage': 'high',       # higher BB% = better hitter
        'strikeout_to_walk_ratio': 'low',
        'runs_plate_appearance': 'high',
    })

    # Attach team_id for chart-builder joins
    tid_map = _load_team_id_map()
    sport_l = sport.lower()
    grp['team_id'] = grp['teamName'].apply(lambda n: tid_map.get((sport_l, n), 0)).astype(str)
    grp['year'] = '2026'
    return grp


def _aggregate_pitching(df: pd.DataFrame, sport: str, hitting_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Roll up per-game pitcher rows to per-team season totals.
    Subset of pitching_team.csv schema. If hitting_df is supplied, A-OPS
    (opponent OPS) is derived by joining each team's gameIds with the
    opponent's batting line in the hitting PBP frame."""
    if df.empty:
        return pd.DataFrame()
    # Pitching PBP column names: so (Ks), hrA (HRs allowed), sfa/sha (SF/SH allowed),
    # doublesA/triplesA (doubles/triples allowed), bf (batters faced), kl (looking),
    # inhRun (inherited runners). NOT the same names as hitting PBP.
    sum_cols = ['h', 'r', 'er', 'bb', 'so', 'hb', 'wp', 'bk',
                'hrA', 'sfa', 'sha', 'kl', 'tuer', 'pickoffs', 'ibb',
                'doublesA', 'triplesA', 'bf', 'inhRun', 'inhRunScore']
    sum_cols = [c for c in sum_cols if c in df.columns]
    # IP needs special outs accumulation: 6.2 IP = 6 + 2/3 outs; sum outs first.
    if 'ip' in df.columns:
        whole = df['ip'].astype(float).fillna(0).astype(int)
        frac = ((df['ip'].astype(float).fillna(0) - whole) * 10).round().astype(int)
        df = df.copy()
        df['_outs'] = whole * 3 + frac
    grp = df.groupby('teamName', as_index=False)[[c for c in sum_cols if c != 'ip']].sum()
    if '_outs' in df.columns:
        outs = df.groupby('teamName')['_outs'].sum().reindex(grp['teamName']).values
        whole_ip = (outs // 3).astype(int)
        leftover = (outs % 3).astype(int)
        grp['innings_pitched'] = whole_ip + leftover / 10.0
        grp['innings_pitched'] = grp['innings_pitched'].round(1)
        grp['_outs'] = outs
    grp['games'] = df.groupby('teamName')['gameId'].nunique().reindex(grp['teamName']).values

    # Rate stats (use outs-based IP for accuracy)
    actual_ip = grp['_outs'] / 3.0 if '_outs' in grp.columns else grp.get('innings_pitched', 0)
    grp['earned_run_average'] = (grp.get('er', 0) * 9 / actual_ip).where(actual_ip > 0, 0).round(3) if 'er' in grp.columns else 0
    grp['walks_plus_hits_per_inning_pitched'] = ((grp.get('bb', 0) + grp.get('h', 0)) / actual_ip).where(actual_ip > 0, 0).round(3) if 'bb' in grp.columns and 'h' in grp.columns else 0
    grp['strikeouts_per_9_innings'] = (grp.get('so', 0) * 9 / actual_ip).where(actual_ip > 0, 0).round(2) if 'so' in grp.columns else 0

    # ── FIP: ((13*HR + 3*(BB+HBP) - 2*K) / IP) + cFIP_constant ──
    # cFIP balances league FIP to league ERA. Re-derive from filtered data:
    # cFIP = league_ERA - league_raw_FIP
    if 'er' in grp.columns and 'hrA' in grp.columns and 'bb' in grp.columns and 'so' in grp.columns:
        league_er = grp['er'].sum()
        league_outs = grp['_outs'].sum() if '_outs' in grp.columns else 0
        league_ip = league_outs / 3.0 if league_outs > 0 else 0
        league_era = (league_er * 9 / league_ip) if league_ip > 0 else 0
        league_hbp = grp['hb'].sum() if 'hb' in grp.columns else 0
        league_raw_fip = (
            (13 * grp['hrA'].sum() + 3 * (grp['bb'].sum() + league_hbp) - 2 * grp['so'].sum())
            / league_ip
        ) if league_ip > 0 else 0
        cfip = league_era - league_raw_fip
        per_team_hbp = grp['hb'] if 'hb' in grp.columns else 0
        per_team_raw_fip = (
            (13 * grp['hrA'] + 3 * (grp['bb'] + per_team_hbp) - 2 * grp['so'])
            / actual_ip
        ).where(actual_ip > 0, 0)
        grp['fielding_independent_pitching'] = (per_team_raw_fip + cfip).round(3)
    else:
        grp['fielding_independent_pitching'] = 0

    # ── A-OPS (opponent OPS): join with hitting PBP — for each pitching team's gameIds,
    # the opponent batting line is the OTHER team's hitting rows in those games.
    if hitting_df is not None and not hitting_df.empty:
        # Build per-team gameId set
        team_games = df.groupby('teamName')['gameId'].apply(set).to_dict()
        a_ab = {}
        a_h = {}
        a_tb = {}
        a_bb = {}
        a_hbp = {}
        a_sf = {}
        for team, gids in team_games.items():
            opp_rows = hitting_df[(hitting_df['gameId'].isin(gids)) & (hitting_df['teamName'] != team)]
            a_ab[team] = opp_rows['ab'].sum() if 'ab' in opp_rows.columns else 0
            a_h[team]  = opp_rows['h'].sum()  if 'h'  in opp_rows.columns else 0
            a_tb[team] = opp_rows['tb'].sum() if 'tb' in opp_rows.columns else 0
            a_bb[team] = opp_rows['bb'].sum() if 'bb' in opp_rows.columns else 0
            a_hbp[team]= opp_rows['hbp'].sum() if 'hbp' in opp_rows.columns else 0
            a_sf[team] = opp_rows['sf'].sum() if 'sf' in opp_rows.columns else 0
        grp['opponent_at_bats'] = grp['teamName'].map(a_ab).fillna(0).astype(int)
        opp_h  = grp['teamName'].map(a_h ).fillna(0)
        opp_tb = grp['teamName'].map(a_tb).fillna(0)
        opp_bb = grp['teamName'].map(a_bb).fillna(0)
        opp_hbp= grp['teamName'].map(a_hbp).fillna(0)
        opp_sf = grp['teamName'].map(a_sf).fillna(0)
        grp['batting_average_against'] = (opp_h / grp['opponent_at_bats']).where(grp['opponent_at_bats'] > 0, 0).round(3)
        opp_obp_denom = grp['opponent_at_bats'] + opp_bb + opp_hbp + opp_sf
        grp['on_base_percentage_against'] = ((opp_h + opp_bb + opp_hbp) / opp_obp_denom).where(opp_obp_denom > 0, 0).round(3)
        grp['slugging_percentage_against'] = (opp_tb / grp['opponent_at_bats']).where(grp['opponent_at_bats'] > 0, 0).round(3)
        grp['on_base_plus_slugging_against'] = (grp['on_base_percentage_against'] + grp['slugging_percentage_against']).round(3)

    if '_outs' in grp.columns:
        grp = grp.drop(columns=['_outs'])

    grp = grp.rename(columns={
        'h': 'hits_allowed', 'r': 'runs_allowed', 'er': 'earned_runs',
        'bb': 'walks_issued', 'so': 'strikeouts', 'hb': 'hit_batter',
        'wp': 'wild_pitch', 'bk': 'balk', 'hrA': 'homeruns_allowed',
        'sfa': 'sac_fly_allowed', 'sha': 'sac_hit_allowed',
        'ibb': 'intentional_walk_allowed', 'doublesA': 'doubles_allowed',
        'triplesA': 'triples_allowed', 'bf': 'batters_faced',
        'inhRun': 'inherited_runner', 'inhRunScore': 'inherited_runner_scored',
        'kl': 'strikeout_looking',
    })

    # Division-scoped percentile ranks (1.0 = best pitcher in div for that stat;
    # lower-is-better stats inverted)
    grp = _add_division_percentile_ranks(grp, sport, {
        'earned_run_average': 'low',
        'walks_plus_hits_per_inning_pitched': 'low',
        'fielding_independent_pitching': 'low',
        'strikeouts_per_9_innings': 'high',
        'batting_average_against': 'low',
        'on_base_percentage_against': 'low',
        'slugging_percentage_against': 'low',
        'on_base_plus_slugging_against': 'low',
    })

    tid_map = _load_team_id_map()
    sport_l = sport.lower()
    grp['team_id'] = grp['teamName'].apply(lambda n: tid_map.get((sport_l, n), 0)).astype(str)
    grp['year'] = '2026'
    return grp


@st.cache_data(show_spinner=True, max_entries=18)  # 2 sports × 3 game_type × 3 day = 18
def derive_team_stats(sport: str, stat_kind: str, game_type: str = 'All', day_filter: str = 'All') -> pd.DataFrame:
    """Public entry point: returns a 2026-only DataFrame matching the
    canonical {hitting,pitching}_team.csv schema (basic columns).
    Caller is responsible for splicing this into the historical CSV."""
    df = _load_concat_pbp(sport, stat_kind)
    df = _apply_filters(df, sport, game_type, day_filter)
    if stat_kind == 'hitting':
        return _aggregate_hitting(df, sport)
    elif stat_kind == 'pitching':
        # Load + filter hitting PBP too so we can derive opponent OPS (A-OPS)
        hit = _load_concat_pbp(sport, 'hitting')
        hit = _apply_filters(hit, sport, game_type, day_filter)
        return _aggregate_pitching(df, sport, hitting_df=hit)
    return pd.DataFrame()


def merge_with_historical(historical_df: pd.DataFrame, derived_2026: pd.DataFrame) -> pd.DataFrame:
    """Replace year-2026 rows in historical_df with derived rows; preserve all other years.
    Column union: keep historical_df's columns AND any new derived columns
    (wOBA, wRC, wRAA, FIP, A-OPS, percentile_rank_*) so chart-builder dropdowns
    can offer the new stats when filter is active.
    """
    if derived_2026.empty:
        return historical_df
    historical_df = historical_df.copy()
    if 'year' in historical_df.columns:
        kept = historical_df[historical_df['year'].astype(str) != '2026']
    else:
        kept = historical_df
    # Union of columns. Any column in historical_df missing from derived gets NaN
    # in derived rows. Any new column in derived gets NaN in historical rows.
    all_cols = list(historical_df.columns) + [c for c in derived_2026.columns if c not in historical_df.columns]
    for col in historical_df.columns:
        if col not in derived_2026.columns:
            derived_2026[col] = pd.NA
    kept = kept.copy()
    for col in derived_2026.columns:
        if col not in kept.columns:
            kept[col] = pd.NA
    derived_2026 = derived_2026[all_cols]
    kept = kept[all_cols]
    return pd.concat([kept, derived_2026], ignore_index=True)
