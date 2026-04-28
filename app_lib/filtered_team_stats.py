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

    # Attach team_id for chart-builder joins
    tid_map = _load_team_id_map()
    sport_l = sport.lower()
    grp['team_id'] = grp['teamName'].apply(lambda n: tid_map.get((sport_l, n), 0)).astype(str)
    grp['year'] = '2026'
    return grp


def _aggregate_pitching(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """Roll up per-game pitcher rows to per-team season totals.
    Subset of pitching_team.csv schema."""
    if df.empty:
        return pd.DataFrame()
    sum_cols = ['ip', 'h', 'r', 'er', 'bb', 'k', 'hb', 'wp', 'bk',
                'hr', 'sf', 'sh', 'cg', 'sho', 'so', 'ibb', 'pickoffs']
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
    grp['strikeouts_per_9_innings'] = (grp.get('k', 0) * 9 / actual_ip).where(actual_ip > 0, 0).round(2) if 'k' in grp.columns else 0
    if '_outs' in grp.columns:
        grp = grp.drop(columns=['_outs'])

    grp = grp.rename(columns={
        'h': 'hits_allowed', 'r': 'runs_allowed', 'er': 'earned_runs',
        'bb': 'walks_issued', 'k': 'strikeouts', 'hb': 'hit_batter',
        'wp': 'wild_pitch', 'bk': 'balk', 'hr': 'homeruns_allowed',
        'sf': 'sac_fly_allowed', 'sh': 'sac_hit_allowed',
        'cg': 'complete_games', 'sho': 'shutouts',
        'ibb': 'intentional_walk_allowed',
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
        return _aggregate_pitching(df, sport)
    return pd.DataFrame()


def merge_with_historical(historical_df: pd.DataFrame, derived_2026: pd.DataFrame) -> pd.DataFrame:
    """Replace year-2026 rows in historical_df with derived rows; preserve all other years."""
    if derived_2026.empty:
        return historical_df
    historical_df = historical_df.copy()
    if 'year' in historical_df.columns:
        kept = historical_df[historical_df['year'].astype(str) != '2026']
    else:
        kept = historical_df
    # Align columns: keep everything historical_df expects; fill missing with 0/NaN
    for col in historical_df.columns:
        if col not in derived_2026.columns:
            derived_2026[col] = pd.NA
    derived_2026 = derived_2026[historical_df.columns]
    return pd.concat([kept, derived_2026], ignore_index=True)
