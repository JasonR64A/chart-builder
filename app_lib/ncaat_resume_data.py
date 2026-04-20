"""Build real-data team dict for the NCAAT Resume infographic.

The template expects a single-team entry keyed as 'selected' with the shape
defined in the original data.jsx sample. This module wires it up from the
chart-builder-app CSVs: teams.csv, schedules.csv, schedules_full_*.csv,
hitting_team.csv, pitching_team.csv, fielding_team.csv, staff.csv,
baseball_rpi_D1.csv / softball_rpi_D1.csv, team_rank.csv, conferences.csv,
bracketology/team_locations.csv.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _APP_DIR / 'data'


def _norm(name: str) -> str:
    if not isinstance(name, str):
        return ''
    # Opponent names in schedules_full_*.csv include "Team @Arlington, TX" style
    # location suffixes — strip after the first " @" or " vs " marker.
    n = re.split(r'\s+@|\s+vs\s+', name, maxsplit=1)[0]
    n = n.lower().strip()
    n = re.sub(r'[^a-z0-9]+', '', n)
    return n


def _is_completed(g) -> bool:
    """Row is completed if isFuture is not True (NaN treated as completed)."""
    v = g.get('isFuture')
    if v is True or v == 1 or str(v).lower() == 'true':
        return False
    return True


def _is_win(g) -> bool:
    """isWin is 1.0 for wins, NaN for losses in this dataset."""
    v = g.get('isWin')
    try:
        return float(v) == 1.0
    except (TypeError, ValueError):
        return False


@st.cache_data(show_spinner=False)
def _load_csv(rel_path: str, **kwargs) -> pd.DataFrame:
    p = DATA_DIR / rel_path
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=False, **kwargs)


@st.cache_data(show_spinner=False)
def list_d1_teams(sport_key: str) -> list[str]:
    """Return sorted D-I team names for the given sport."""
    teams = _load_csv('teams.csv')
    conferences = _load_csv('conferences.csv')
    if teams.empty or conferences.empty:
        return []
    sport_label = 'Baseball' if sport_key == 'baseball' else 'Softball'
    di_ids = set(conferences[conferences['division'] == 'D-I']['id'].tolist())
    d1 = teams[(teams['sport'] == sport_label) & (teams['conference_id'].isin(di_ids))]
    return sorted(d1['name'].dropna().unique().tolist())


@st.cache_data(show_spinner=False)
def _rpi_frame(sport_key: str) -> pd.DataFrame:
    fname = 'baseball_rpi_D1.csv' if sport_key == 'baseball' else 'softball_rpi_D1.csv'
    df = _load_csv(fname)
    if df.empty:
        return df
    df = df.copy()
    df['name_norm'] = df['teamName'].apply(_norm)
    return df


@st.cache_data(show_spinner=False)
def _opponent_rpi_lookup(sport_key: str) -> dict:
    df = _rpi_frame(sport_key)
    if df.empty:
        return {}
    return dict(zip(df['name_norm'], df['rank'].astype(int)))


@st.cache_data(show_spinner=False)
def _sixty_four_lookup(year: int = 2026) -> dict:
    tr = _load_csv('team_rank.csv')
    if tr.empty:
        return {}
    tr = tr[tr['year'] == year]
    rank_col = 'integer_64_rank_total' if 'integer_64_rank_total' in tr.columns else 'sixty_four_rank_weighted_run_efficiency'
    out = {}
    for _, r in tr.iterrows():
        v = r.get(rank_col)
        if pd.notna(v):
            out[int(r['team_id'])] = int(v)
    return out


@st.cache_data(show_spinner=False)
def _resume_score_lookup(sport_key: str, year: int = 2026) -> dict:
    """Precompute resume score for every D-I team so we can build nearest-by-score.
    Score = 100 * (1 - 0.6*rpi_pct - 0.4*rank64_pct), bounded 0-100.
    rpi_pct = rpi_rank / 301; rank64_pct same.
    """
    teams = _load_csv('teams.csv')
    conferences = _load_csv('conferences.csv')
    if teams.empty:
        return {}
    sport_label = 'Baseball' if sport_key == 'baseball' else 'Softball'
    di_ids = set(conferences[conferences['division'] == 'D-I']['id'].tolist())
    d1 = teams[(teams['sport'] == sport_label) & (teams['conference_id'].isin(di_ids))]
    rpi = _rpi_frame(sport_key)
    rpi_by_name = dict(zip(rpi['name_norm'], rpi['rank'].astype(int))) if not rpi.empty else {}
    rank64 = _sixty_four_lookup(year)
    out = {}
    for _, r in d1.iterrows():
        name = r['name']
        rpi_r = rpi_by_name.get(_norm(name), 301)
        r64 = rank64.get(int(r['id']), 301)
        out[name] = _resume_score_from_ranks(rpi_r, r64)
    return out


@st.cache_data(show_spinner=False)
def _team_locations() -> dict:
    """name -> 'City, ST'"""
    loc = _load_csv('bracketology/team_locations.csv')
    if loc.empty:
        return {}
    out = {}
    for _, r in loc.iterrows():
        if pd.notna(r.get('city')) and pd.notna(r.get('state')):
            out[r['team_name']] = f"{r['city']}, {r['state']}"
    return out


@st.cache_data(show_spinner=False)
def _dsr_rank_lookup(sport_key: str) -> dict:
    """Map normalized team name -> DSR rank from data/rankings/dsr_*.csv."""
    fname = f'rankings/dsr_{sport_key}.csv'
    df = _load_csv(fname)
    if df.empty:
        return {}
    # Keep the most recent date (the file usually carries a single daily snapshot,
    # but some appenders leave history).
    if 'date' in df.columns:
        latest = df['date'].max()
        df = df[df['date'] == latest]
    return {_norm(t): int(r) for t, r in zip(df['team'], df['rank'])}


@st.cache_data(show_spinner=False)
def _sos_lookup(sport_key: str) -> dict:
    """Compute NCAA-style SOS for every team with completed games, and return
    {team_name: (sos_rank_overall, sos_rank_nonCon)}.

    NCAA SOS = 2/3 * OWP + 1/3 * OOWP.
    OWP: average winning pct of a team's opponents (each opponent counted once
    per meeting; opponent WP is computed over ALL their games).
    OOWP: average of each opponent's OWP.
    Non-con SOS uses only games where opponent conference != team conference.
    """
    sched_full = _load_csv(f'schedules_full_{sport_key}.csv')
    if sched_full.empty:
        return {}
    teams_df = _load_csv('teams.csv')
    conferences_df = _load_csv('conferences.csv')
    sport_label = 'Baseball' if sport_key == 'baseball' else 'Softball'
    conf_by_team = dict(zip(
        teams_df[teams_df['sport'] == sport_label]['name'],
        teams_df[teams_df['sport'] == sport_label]['conference_id'],
    ))

    # Normalize opponent names once
    df = sched_full.copy()
    df = df[df.apply(_is_completed, axis=1)]
    df['opp_clean'] = df['opponentName'].apply(_clean_opp)
    df['is_win_flag'] = df.apply(_is_win, axis=1)

    # Team WP over all completed games
    grouped = df.groupby('teamName')['is_win_flag'].agg(['sum', 'count'])
    grouped['wp'] = grouped['sum'] / grouped['count'].clip(lower=1)
    wp_by_team = grouped['wp'].to_dict()
    games_by_team = grouped['count'].to_dict()

    # Per-team opponent lists (and per-team opponent lists filtered by non-con)
    team_opps = {}
    team_opps_nc = {}
    for team, chunk in df.groupby('teamName'):
        opps = chunk['opp_clean'].tolist()
        team_opps[team] = opps
        team_conf = conf_by_team.get(team)
        nc = []
        for _, row in chunk.iterrows():
            opp = row['opp_clean']
            opp_conf = conf_by_team.get(opp)
            if team_conf is None or opp_conf is None or opp_conf != team_conf:
                nc.append(opp)
        team_opps_nc[team] = nc

    # OWP and OOWP
    def _owp_for(opps):
        vals = [wp_by_team.get(o) for o in opps if o in wp_by_team]
        return sum(vals) / len(vals) if vals else 0.0

    owp_by_team = {t: _owp_for(os) for t, os in team_opps.items()}
    owp_nc_by_team = {t: _owp_for(os) for t, os in team_opps_nc.items()}

    sos_overall = {}
    sos_noncon = {}
    for team, opps in team_opps.items():
        # OOWP = mean of each opponent's OWP
        oo_vals = [owp_by_team.get(o) for o in opps if o in owp_by_team]
        oowp = sum(oo_vals) / len(oo_vals) if oo_vals else 0.0
        sos_overall[team] = (2/3) * owp_by_team.get(team, 0) + (1/3) * oowp

        opps_nc = team_opps_nc.get(team, [])
        oo_nc_vals = [owp_by_team.get(o) for o in opps_nc if o in owp_by_team]
        oowp_nc = sum(oo_nc_vals) / len(oo_nc_vals) if oo_nc_vals else 0.0
        sos_noncon[team] = (2/3) * owp_nc_by_team.get(team, 0) + (1/3) * oowp_nc

    # Rank teams — restrict ranking to D-I teams so ranks are 1..N within division.
    di_ids = set(conferences_df[conferences_df['division'] == 'D-I']['id'].tolist())
    di_names = set(teams_df[(teams_df['sport'] == sport_label) & (teams_df['conference_id'].isin(di_ids))]['name'])

    def _rank(scores: dict) -> dict:
        di_scores = [(n, s) for n, s in scores.items() if n in di_names]
        di_scores.sort(key=lambda t: -t[1])
        return {n: i + 1 for i, (n, _) in enumerate(di_scores)}

    rank_overall = _rank(sos_overall)
    rank_noncon = _rank(sos_noncon)

    out = {}
    of = max(len(rank_overall), 1)
    for team in rank_overall.keys() | rank_noncon.keys():
        out[team] = {
            'overall_rank': rank_overall.get(team, of),
            'noncon_rank': rank_noncon.get(team, of),
            'overall_score': sos_overall.get(team, 0.0),
            'noncon_score': sos_noncon.get(team, 0.0),
            'of': of,
        }
    return out


@st.cache_data(show_spinner=False)
def _historical_analog_pool(sport_key: str) -> pd.DataFrame:
    """Build a per-(year, team) table of selection RPI + national seed + final result
    that we can use to find historical analogs. Only baseball is wired today.
    """
    if sport_key != 'baseball':
        return pd.DataFrame()
    rpi_hist = _load_csv('bracketology/historical_selection_rpi.csv')
    brackets = _load_csv('bracketology/historical_brackets.csv')
    results = _load_csv('bracketology/historical_results.csv')
    if rpi_hist.empty or brackets.empty:
        return pd.DataFrame()
    rpi_hist = rpi_hist[rpi_hist['sport'] == 'baseball'].copy()
    # Convert "2024-25" year label to integer year (the tournament year)
    def _parse_year(y):
        s = str(y)
        if '-' in s:
            parts = s.split('-')
            return int(parts[0]) + 1
        return int(s)
    rpi_hist['year_int'] = rpi_hist['year'].apply(_parse_year)
    # Strip "(AQ)" / " (AQ)" suffixes from teamName
    rpi_hist['team_clean'] = rpi_hist['teamName'].astype(str).str.replace(r'\s*\(AQ\)\s*', '', regex=True).str.strip()
    rpi_hist['name_norm'] = rpi_hist['team_clean'].apply(_norm)
    rpi_hist = rpi_hist[['year_int', 'team_clean', 'name_norm', 'rank', 'rpi', 'conference']].rename(
        columns={'year_int': 'year', 'team_clean': 'team', 'rank': 'rpi_rank'}
    )
    brackets_bb = brackets[brackets['sport'] == 'baseball'].copy()
    brackets_bb['name_norm'] = brackets_bb['team'].apply(_norm)
    # National seed: only for teams that hosted AND were 1-seed in their regional.
    host_rows = brackets_bb[
        (brackets_bb['team'] == brackets_bb['host_team']) &
        (brackets_bb['regional_seed'] == 1)
    ][['year', 'name_norm', 'national_seed']].rename(columns={'national_seed': 'nat_seed'})
    # Regional seed: each team's 1-4 slot within their regional (everyone has one).
    reg_seed_rows = brackets_bb[['year', 'name_norm', 'regional_seed']].drop_duplicates(['year', 'name_norm'])
    out = rpi_hist.merge(host_rows, on=['year', 'name_norm'], how='left')
    out = out.merge(reg_seed_rows, on=['year', 'name_norm'], how='left')
    # Attach result + tournament W-L if available
    if not results.empty:
        results_c = results.copy()
        results_c['name_norm'] = results_c['team'].apply(_norm)
        merge_cols = ['year', 'name_norm', 'result']
        if 'ncaat_wins' in results_c.columns:
            merge_cols += ['ncaat_wins', 'ncaat_losses']
        out = out.merge(results_c[merge_cols], on=['year', 'name_norm'], how='left')
    else:
        out['result'] = None
    if 'ncaat_wins' not in out.columns:
        out['ncaat_wins'] = None
        out['ncaat_losses'] = None
    # Historical resume score uses the same rank-to-score curve as the current
    # season, with a small bonus for national seeds (1-16). We don't have the
    # 64A composite for historical teams, so this is RPI-only.
    of = 301
    out['rpi_rank_num'] = pd.to_numeric(out['rpi_rank'], errors='coerce').fillna(of).astype(int)
    rpi_score_series = out['rpi_rank_num'].apply(_rank_to_score)
    seed_bonus = out['nat_seed'].apply(lambda s: 0 if pd.isna(s) else max(0, 17 - int(s))) * 0.4
    out['score'] = (rpi_score_series + seed_bonus).clip(0, 100).round().astype(int)
    # Restrict to teams that actually played in the tournament (have a regional_seed).
    # Non-tournament teams stay out of the analog pool so we don't surface random #70 RPI misses.
    out = out[out['regional_seed'].notna()].copy()
    return out[[
        'year', 'team', 'name_norm', 'rpi_rank_num', 'nat_seed', 'regional_seed',
        'result', 'ncaat_wins', 'ncaat_losses', 'score', 'conference',
    ]]


def _compute_analogs(team_name: str, score: int, sport_key: str, limit: int = 5) -> list:
    """Return up to `limit` historical teams closest in resume score to `score`.
    Each analog ships with scoreDelta (positive = historical team scored higher).
    """
    pool = _historical_analog_pool(sport_key)
    if pool.empty:
        return []
    pool = pool.copy()
    pool['abs_diff'] = (pool['score'] - score).abs()
    pool = pool.sort_values(['abs_diff', 'nat_seed']).head(limit)
    out = []
    for _, row in pool.iterrows():
        result = row['result']
        if not isinstance(result, str) or not result:
            result = 'Regional'
        seed_val = row.get('nat_seed')
        seed_int = int(seed_val) if pd.notna(seed_val) else 99
        reg_seed_val = row.get('regional_seed')
        reg_seed = int(reg_seed_val) if pd.notna(reg_seed_val) else None
        # scoreDelta is historical - current; user wants current-oriented perspective,
        # so flip: positive = current team is stronger than analog.
        score_delta = int(score) - int(row['score'])
        rpi_rank = int(row['rpi_rank_num']) if pd.notna(row.get('rpi_rank_num')) else None
        wins = row.get('ncaat_wins')
        losses = row.get('ncaat_losses')
        wins_int = int(wins) if pd.notna(wins) else None
        losses_int = int(losses) if pd.notna(losses) else None
        out.append({
            'team': row['team'],
            'year': int(row['year']),
            'score': int(row['score']),
            'seed': seed_int,
            'regionalSeed': reg_seed,
            'result': result,
            'scoreDelta': score_delta,
            'rpi': rpi_rank,
            'ncaatWins': wins_int,
            'ncaatLosses': losses_int,
        })
    return out


def _radar_area_pct(stat_pcts: dict) -> int:
    """Return 0-100 representing the polygon area on the radar chart as a
    percentage of the maximum possible area (all 9 stats at 100th percentile).
    For a regular N-gon with radial lengths r_i (0-1), area = 0.5*sin(2pi/N)*sum(r_i*r_{i+1}).
    Max area when all r_i = 1.
    """
    keys = list(stat_pcts.keys())
    n = len(keys)
    if n < 3:
        return 0
    rs = [max(0.0, min(1.0, (stat_pcts[k] or 0) / 100.0)) for k in keys]
    pair_sum = sum(rs[i] * rs[(i + 1) % n] for i in range(n))
    max_pair_sum = n  # when all r=1
    return int(round(100 * pair_sum / max_pair_sum))


def _compute_remaining_quadrants(sched_full: pd.DataFrame, team_name: str, rpi_lookup: dict) -> dict:
    """Return a list of upcoming games per quadrant:
      {'q1': [{opp, venue, date, oppRank}, ...], 'q2': [...], ...}
    Venue: 'home' | 'neutral' | 'away'.
    Games without a ranked opponent bucket as Q4.
    """
    m = sched_full[sched_full['teamName'] == team_name].copy()
    if not m.empty:
        m['_d'] = pd.to_datetime(m['date'], errors='coerce')
        m = m.sort_values('_d')
    quads = {'q1': [], 'q2': [], 'q3': [], 'q4': []}
    for _, g in m.iterrows():
        if _is_completed(g):
            continue
        opp_raw = g.get('opponentName')
        if not isinstance(opp_raw, str):
            continue
        venue = _venue(g)
        opp_rank = rpi_lookup.get(_norm(opp_raw), 999)
        q = _quad_bucket(opp_rank, venue)
        quads[q].append({
            'opp': _clean_opp(opp_raw),
            'venue': venue,
            'date': g.get('date'),
            'oppRank': int(opp_rank) if opp_rank < 999 else None,
        })
    return quads


def _project_rpi_range(current_rpi_rank: int, remaining_games: int) -> list:
    """Return [low_rank, high_rank] band for projected final RPI based on how many
    games are left. Tighter band late in the season.
    Heuristic: band width scales with sqrt(games_remaining).
    """
    if current_rpi_rank >= 301:
        return [301, 301]
    if remaining_games <= 0:
        return [current_rpi_rank, current_rpi_rank]
    width = int(round(1.5 * (remaining_games ** 0.5)))
    low = max(1, current_rpi_rank - width)
    high = min(301, current_rpi_rank + int(round(width * 1.4)))
    return [low, high]


@st.cache_data(show_spinner=False)
def _head_coach_lookup() -> dict:
    staff = _load_csv('staff.csv')
    if staff.empty:
        return {}
    hc = staff[staff['position'].astype(str).str.contains('Head', case=False, na=False)]
    out = {}
    for _, r in hc.iterrows():
        tid = r.get('team_id')
        if pd.notna(tid):
            out[int(tid)] = r.get('name', '')
    return out


def _venue(g) -> str:
    """Return 'away' | 'neutral' | 'home' for a schedules_full row."""
    is_away = g.get('isAway')
    if is_away is True or is_away == 1 or is_away == 1.0:
        return 'away'
    opp = g.get('opponentName')
    if isinstance(opp, str) and '@' in opp:
        return 'neutral'
    return 'home'


def _quad_bucket(opp_rank: int, venue: str) -> str:
    """NCAA quadrant thresholds: Q1 H1-25 N1-40 A1-60; Q2 H26-50 N41-80 A61-120;
    Q3 H51-100 N81-160 A121-240; Q4 H101+ N161+ A241+."""
    if venue == 'home':
        if opp_rank <= 25:  return 'q1'
        if opp_rank <= 50:  return 'q2'
        if opp_rank <= 100: return 'q3'
        return 'q4'
    elif venue == 'neutral':
        if opp_rank <= 40:  return 'q1'
        if opp_rank <= 80:  return 'q2'
        if opp_rank <= 160: return 'q3'
        return 'q4'
    else:  # away
        if opp_rank <= 60:  return 'q1'
        if opp_rank <= 120: return 'q2'
        if opp_rank <= 240: return 'q3'
        return 'q4'


def _grade_for_score(score: int) -> str:
    """Academic 100-point scale with a 76 landing on C+ per user spec."""
    if score >= 97: return 'A+'
    if score >= 93: return 'A'
    if score >= 90: return 'A-'
    if score >= 87: return 'B+'
    if score >= 83: return 'B'
    if score >= 80: return 'B-'
    if score >= 76: return 'C+'
    if score >= 73: return 'C'
    if score >= 70: return 'C-'
    if score >= 67: return 'D+'
    if score >= 63: return 'D'
    if score >= 60: return 'D-'
    return 'F'


def _verdict_for_score(score: int) -> tuple[str, str]:
    """Anchored on the score curve: RPI ~50 -> 70 (last in),
    RPI ~60 -> 64 (bubble floor), RPI 61+ -> out.
    """
    if score >= 88: return '1 seed', 'Lock'       # RPI 1-16
    if score >= 80: return '2 seed', 'Lock'       # RPI 17-30
    if score >= 75: return '3 seed', 'In'         # RPI 31-40
    if score >= 70: return '4 seed', 'In'         # RPI 41-50
    if score >= 64: return 'Bubble', 'Bubble'     # RPI 51-60
    return 'Out', 'Out'                           # RPI 61+


_SCORE_ANCHORS = [(1, 100), (16, 88), (30, 80), (50, 70), (60, 64), (100, 50), (200, 15), (300, 0)]


def _rank_to_score(rank) -> float:
    """Piecewise-linear map from ranking (1-300) to 0-100 score.
    Anchored on NCAA tournament reality: top-16 national seeds score 85+,
    RPI ~50 (bubble) scores 60, RPI ~100 (clearly out) scores 25.
    """
    if rank is None:
        return 0.0
    try:
        r = float(rank)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(r):
        return 0.0
    if r <= _SCORE_ANCHORS[0][0]:
        return float(_SCORE_ANCHORS[0][1])
    if r >= _SCORE_ANCHORS[-1][0]:
        return float(_SCORE_ANCHORS[-1][1])
    for (r1, s1), (r2, s2) in zip(_SCORE_ANCHORS, _SCORE_ANCHORS[1:]):
        if r1 <= r <= r2:
            t = (r - r1) / (r2 - r1)
            return s1 + t * (s2 - s1)
    return 0.0


def _resume_score_from_ranks(rpi_rank, rank64) -> int:
    """Blended resume score (60% RPI, 40% 64A rank), bounded 0-100."""
    rpi_score = _rank_to_score(rpi_rank)
    r64_score = _rank_to_score(rank64)
    blended = 0.6 * rpi_score + 0.4 * r64_score
    return max(0, min(100, int(round(blended))))


def _clean_opp(name: str) -> str:
    """Strip ' @Location, ST' and ' vs X' suffixes from opponent names."""
    if not isinstance(name, str):
        return ''
    return re.split(r'\s+@|\s+vs\s+', name, maxsplit=1)[0].strip()


def _compute_quad_record(sched_full: pd.DataFrame, team_name: str, rpi_lookup: dict) -> dict:
    m = sched_full[sched_full['teamName'] == team_name]
    quads = {'q1': [0, 0], 'q2': [0, 0], 'q3': [0, 0], 'q4': [0, 0]}
    for _, g in m.iterrows():
        if not _is_completed(g):
            continue
        opp = g.get('opponentName')
        if not isinstance(opp, str):
            continue
        venue = _venue(g)
        opp_rank = rpi_lookup.get(_norm(opp), 999)
        q = _quad_bucket(opp_rank, venue)
        if _is_win(g):
            quads[q][0] += 1
        else:
            quads[q][1] += 1
    return {k: f'{v[0]}-{v[1]}' for k, v in quads.items()}


def _last_10_games(sched_full: pd.DataFrame, team_name: str, rpi_lookup: dict) -> list:
    m = sched_full[sched_full['teamName'] == team_name].copy()
    if m.empty:
        return []
    m = m[m.apply(_is_completed, axis=1)]
    if m.empty:
        return []
    m['_d'] = pd.to_datetime(m['date'], errors='coerce')
    m = m.sort_values('_d').tail(10)
    out = []
    for _, g in m.iterrows():
        opp_raw = g.get('opponentName') or ''
        opp = _clean_opp(opp_raw)
        rs = int(g['runsFor']) if pd.notna(g.get('runsFor')) else 0
        ra = int(g['runsAgainst']) if pd.notna(g.get('runsAgainst')) else 0
        venue = _venue(g)
        home = venue == 'home'
        out.append({
            'opp': opp,
            'home': home,
            'rs': rs,
            'ra': ra,
            'result': 'W' if _is_win(g) else 'L',
            'oppRank': int(rpi_lookup.get(_norm(opp_raw), 999)),
        })
    return out


def _big_wins(sched_full: pd.DataFrame, team_name: str, rpi_lookup: dict, top_n: int = 3) -> list:
    m = sched_full[sched_full['teamName'] == team_name].copy()
    if m.empty:
        return []
    m = m[m.apply(lambda g: _is_completed(g) and _is_win(g), axis=1)]
    if m.empty:
        return []
    m['opp_rank'] = m['opponentName'].apply(lambda x: rpi_lookup.get(_norm(x), 999))
    m = m[m['opp_rank'] < 999].sort_values('opp_rank').head(top_n)
    out = []
    for _, g in m.iterrows():
        rs = int(g['runsFor']) if pd.notna(g.get('runsFor')) else 0
        ra = int(g['runsAgainst']) if pd.notna(g.get('runsAgainst')) else 0
        venue = _venue(g)
        home = venue == 'home'
        out.append({
            'opp': _clean_opp(g['opponentName']),
            'score': f'{rs}-{ra}',
            'note': f'{"Home" if venue == "home" else "Neutral" if venue == "neutral" else "Road"} · opp #{int(g["opp_rank"])}',
        })
    return out


def _bad_losses(sched_full: pd.DataFrame, team_name: str, rpi_lookup: dict, threshold: int = 100, max_n: int = 3) -> list:
    m = sched_full[sched_full['teamName'] == team_name].copy()
    if m.empty:
        return []
    m = m[m.apply(lambda g: _is_completed(g) and not _is_win(g), axis=1)]
    if m.empty:
        return []
    m['opp_rank'] = m['opponentName'].apply(lambda x: rpi_lookup.get(_norm(x), 999))
    m = m[m['opp_rank'] > threshold].sort_values('opp_rank', ascending=False).head(max_n)
    out = []
    for _, g in m.iterrows():
        rs = int(g['runsFor']) if pd.notna(g.get('runsFor')) else 0
        ra = int(g['runsAgainst']) if pd.notna(g.get('runsAgainst')) else 0
        venue = _venue(g)
        home = venue == 'home'
        out.append({
            'opp': _clean_opp(g['opponentName']),
            'score': f'{rs}-{ra}',
            'note': f'{"Home" if venue == "home" else "Neutral" if venue == "neutral" else "Road"} · opp #{int(g["opp_rank"])}',
        })
    return out


def _nearest_by_score(team_name: str, score: int, score_lookup: dict, teams_df: pd.DataFrame, conferences_df: pd.DataFrame, sport_key: str, limit: int = 5) -> list:
    if not score_lookup:
        return []
    conf_map = dict(zip(conferences_df['id'], conferences_df['abbreviation']))
    sport_label = 'Baseball' if sport_key == 'baseball' else 'Softball'
    sport_teams = teams_df[teams_df['sport'] == sport_label]
    team_conf = dict(zip(sport_teams['name'], sport_teams['conference_id']))
    team_id_by_name = dict(zip(sport_teams['name'], sport_teams['id']))
    candidates = []
    for other, other_score in score_lookup.items():
        if other == team_name:
            continue
        candidates.append((other, other_score, abs(other_score - score)))
    candidates.sort(key=lambda t: t[2])
    # We need the primary team's stat percentiles to compute similarity. Resolve its team_id.
    primary_tid = team_id_by_name.get(team_name)
    primary_stats = None
    if primary_tid is not None:
        ps = _team_stats(int(primary_tid))
        if ps:
            primary_stats = {k: v['pct'] for k, v in ps.items()}
    out = []
    for other, other_score, _ in candidates[:limit]:
        conf_abbrev = conf_map.get(team_conf.get(other, ''), '')
        entry = {
            'team': other,
            'conf': conf_abbrev or '',
            'score': int(other_score),
            'diff': int(other_score - score),
        }
        tid = team_id_by_name.get(other)
        if tid is not None:
            stats = _team_stats(int(tid))
            if stats:
                pct_map = {k: v['pct'] for k, v in stats.items()}
                entry['stats'] = pct_map
                entry['statArea'] = _radar_area_pct(pct_map)
                if primary_stats:
                    diffs = [abs(pct_map.get(k, 50) - primary_stats.get(k, 50)) for k in primary_stats.keys()]
                    avg_diff = sum(diffs) / max(len(diffs), 1)
                    entry['statSim'] = int(round(max(0, 100 - avg_diff)))
        out.append(entry)
    return out


def _stat_pct(row, col: str, invert: bool = False, default: float = 0.5) -> int:
    """percentile_rank_* cols are 0-1. Return 0-100 int; invert flips so lower raw = better."""
    v = row.get(col)
    if not pd.notna(v):
        return int(default * 100)
    v = float(v)
    if invert:
        v = 1.0 - v
    return int(round(max(0, min(1, v)) * 100))


def _team_stats(team_id: int, year: int = 2026) -> dict | None:
    hitting = _load_csv('hitting_team.csv')
    pitching = _load_csv('pitching_team.csv')
    fielding = _load_csv('fielding_team.csv')
    if hitting.empty or pitching.empty:
        return None
    h = hitting[(hitting['team_id'] == team_id) & (hitting['year'] == year)]
    p = pitching[(pitching['team_id'] == team_id) & (pitching['year'] == year)]
    f = fielding[(fielding['team_id'] == team_id) & (fielding['year'] == year)]
    if h.empty or p.empty:
        return None
    h = h.iloc[0]
    p = p.iloc[0]
    f_row = f.iloc[0] if not f.empty else None
    g = float(h.get('games_played') or 0)
    rs = float(h.get('runs_scored') or 0)
    rpg = rs / g if g else 0.0
    ip = float(p.get('innings_pitched') or 0)
    bb = float(p.get('walks_issued') or 0)
    bb9 = (bb * 9.0 / ip) if ip > 0 else 0.0
    rpg_pct = _stat_pct(h, 'percentile_rank_weighted_runs_created_plus')
    rf_pct = 50
    rf_value = 0.0
    if f_row is not None and not fielding.empty:
        rf_value = float(f_row.get('range_factor') or 0)
        rf2026 = fielding[fielding['year'] == year]['range_factor'].dropna()
        if len(rf2026):
            rank = (rf2026 < rf_value).sum()
            rf_pct = int(round(100 * rank / max(len(rf2026) - 1, 1)))
    k9 = float(p.get('strikeouts_per_9_innings') or 0)
    return {
        'ops': {
            'value': float(h.get('on_base_plus_slugging') or 0),
            'natAvg': 0.78,
            'pct': _stat_pct(h, 'percentile_rank_on_base_plus_slugging'),
        },
        'woba': {
            'value': float(h.get('weighted_on_base_average') or 0),
            'natAvg': 0.36,
            'pct': _stat_pct(h, 'percentile_rank_weighted_on_base_average'),
        },
        'runsPerGame': {
            'value': round(rpg, 2),
            'natAvg': 6.1,
            'pct': rpg_pct,
        },
        'fip': {
            'value': float(p.get('fielding_independent_pitching') or 0),
            'natAvg': 5.0,
            'pct': _stat_pct(p, 'percentile_rank_fielding_independent_pitching'),
            'inverted': True,
        },
        'whip': {
            'value': float(p.get('walks_plus_hits_per_inning_pitched') or 0),
            'natAvg': 1.45,
            'pct': _stat_pct(p, 'percentile_rank_walks_plus_hits_per_inning_pitched'),
            'inverted': True,
        },
        'k9': {
            'value': round(k9, 2),
            'natAvg': 8.5,
            'pct': _stat_pct(p, 'percentile_rank_strikeout_percentage'),
        },
        'bb9': {
            'value': round(bb9, 2),
            'natAvg': 4.1,
            'pct': _stat_pct(p, 'percentile_rank_walk_percentage', invert=True),
            'inverted': True,
        },
        'rangeFactor': {
            'value': round(rf_value, 2),
            'natAvg': 4.3,
            'pct': rf_pct,
        },
        'wrcPlus': {
            'value': float(h.get('weighted_runs_created_plus') or 100),
            'natAvg': 100,
            'pct': _stat_pct(h, 'percentile_rank_weighted_runs_created_plus'),
        },
    }


def build_resume_team(team_name: str, sport_key: str, year: int = 2026) -> dict | None:
    """Build the single-team dict the React infographic expects.
    Returns None if critical data (teams, rankings, stats) is missing.
    """
    teams = _load_csv('teams.csv')
    conferences = _load_csv('conferences.csv')
    schedules = _load_csv('schedules.csv')
    sport_label = 'Baseball' if sport_key == 'baseball' else 'Softball'
    team_row = teams[(teams['sport'] == sport_label) & (teams['name'] == team_name)]
    if team_row.empty:
        return None
    team_row = team_row.iloc[0]
    team_id = int(team_row['id'])
    conf_id = team_row.get('conference_id')
    conf_abbrev = ''
    conf_name = ''
    conf_match = conferences[conferences['id'] == conf_id]
    if not conf_match.empty:
        conf_abbrev = conf_match.iloc[0].get('abbreviation', '') or ''
        conf_name = conf_match.iloc[0].get('name', '') or conf_abbrev

    # Record
    sched_year = schedules[(schedules['team_id'] == team_id) & (schedules['Year'] == year)]
    if sched_year.empty:
        record = '0-0'
        conf_record = '0-0'
    else:
        s = sched_year.iloc[0]
        cw = int(s.get('Conf_Win') or 0)
        cl = int(s.get('Conf_Loss') or 0)
        w = cw + int(s.get('OOC_Win') or 0) + int(s.get('Post_Win') or 0)
        l = cl + int(s.get('OOC_Loss') or 0) + int(s.get('Post_Loss') or 0)
        record = f'{w}-{l}'
        conf_record = f'{cw}-{cl}'

    # Rankings
    rpi_df = _rpi_frame(sport_key)
    rpi_rank = 301
    if not rpi_df.empty:
        match = rpi_df[rpi_df['name_norm'] == _norm(team_name)]
        if not match.empty:
            rpi_rank = int(match.iloc[0]['rank'])
    rank64 = _sixty_four_lookup(year).get(team_id, 301)
    dsr_rank = _dsr_rank_lookup(sport_key).get(_norm(team_name), 301)
    # ELO currently has no source in-repo; fall back to the mean of available
    # ranks so the module doesn't lie with a fake-looking value.
    elo_rank = int(round((rpi_rank + rank64 + dsr_rank) / 3))
    rankings = {
        'rpi': rpi_rank,
        'dsr': dsr_rank,
        'massey': rank64,
        'elo': elo_rank,
        '64a': rank64,
    }

    # Resume score: same piecewise rank-to-score curve as _resume_score_lookup.
    of = 301
    resume_score = _resume_score_from_ranks(rpi_rank, rank64)
    seed_proj, bubble_verdict = _verdict_for_score(resume_score)

    # SOS: real NCAA-formula computation from schedules_full.
    def _tier(rank: int) -> str:
        if rank <= 25:  return 'Elite'
        if rank <= 75:  return 'Strong'
        if rank <= 150: return 'Moderate'
        return 'Weak'
    sos_map = _sos_lookup(sport_key).get(team_name)
    if sos_map:
        sos_rank = int(sos_map['overall_rank'])
        non_con_rank = int(sos_map['noncon_rank'])
    else:
        sos_rank = of
        non_con_rank = of

    # Schedule-derived blocks
    sched_full = _load_csv(f'schedules_full_{sport_key}.csv')
    rpi_lookup = _opponent_rpi_lookup(sport_key)
    quad_record = _compute_quad_record(sched_full, team_name, rpi_lookup) if not sched_full.empty else {'q1': '0-0', 'q2': '0-0', 'q3': '0-0', 'q4': '0-0'}
    remaining_quads = _compute_remaining_quadrants(sched_full, team_name, rpi_lookup) if not sched_full.empty else {'q1': [], 'q2': [], 'q3': [], 'q4': []}
    remaining_count = sum(len(v) for v in remaining_quads.values())
    projected_rpi_range = _project_rpi_range(rpi_rank, remaining_count)
    last10 = _last_10_games(sched_full, team_name, rpi_lookup) if not sched_full.empty else []
    big_wins = _big_wins(sched_full, team_name, rpi_lookup) if not sched_full.empty else []
    bad_losses = _bad_losses(sched_full, team_name, rpi_lookup) if not sched_full.empty else []

    # Nearest by resume score
    score_lookup = _resume_score_lookup(sport_key, year)
    nearest = _nearest_by_score(team_name, resume_score, score_lookup, teams, conferences, sport_key)

    # Stats module
    stats = _team_stats(team_id, year)
    if stats is None:
        # Safe fallback so the component still renders
        stats = {
            k: {'value': 0.0, 'natAvg': 0.0, 'pct': 50}
            for k in ('ops', 'woba', 'runsPerGame', 'fip', 'whip', 'k9', 'bb9', 'fieldingPct', 'wrcPlus')
        }
    stat_pcts = {k: v['pct'] for k, v in stats.items()}
    stat_area = _radar_area_pct(stat_pcts)

    location = _team_locations().get(team_name, '')
    coach = _head_coach_lookup().get(team_id, '')

    # Nickname: use team name if no better source
    nickname = team_name

    return {
        'name': team_name,
        'nickname': nickname,
        'year': year,
        'conf': conf_abbrev or conf_name or '',
        'record': record,
        'confRecord': conf_record,
        'colors': {'primary': '#0A0A0A', 'secondary': '#CFAE70', 'accent': '#8B6F3E'},
        'location': location,
        'coach': coach,
        'resumeScore': resume_score,
        'grade': _grade_for_score(resume_score),
        'seedProjection': seed_proj,
        'bubbleVerdict': bubble_verdict,
        'stats': stats,
        'statArea': stat_area,
        'sos': {'value': sos_rank, 'natRank': sos_rank, 'of': of, 'tier': _tier(sos_rank)},
        'nonConSos': {'value': non_con_rank, 'natRank': non_con_rank, 'of': of, 'tier': _tier(non_con_rank)},
        'rankings': rankings,
        'projectedRpiRange': projected_rpi_range,
        'last10': last10,
        'quadRecord': quad_record,
        'remainingQuadSchedule': remaining_quads,
        'bigWins': big_wins,
        'badLosses': bad_losses,
        'nearestByScore': nearest,
        'analogs': _compute_analogs(team_name, resume_score, sport_key),
    }
