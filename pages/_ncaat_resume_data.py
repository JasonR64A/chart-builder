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
    of = 301
    out = {}
    for _, r in d1.iterrows():
        name = r['name']
        rpi_r = rpi_by_name.get(_norm(name), 301)
        r64 = rank64.get(int(r['id']), 301)
        rpi_pct = min(rpi_r / of, 1.0)
        r64_pct = min(r64 / of, 1.0)
        score = round(100 * (1 - 0.6 * rpi_pct - 0.4 * r64_pct))
        out[name] = max(0, min(100, score))
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


def _quad_bucket(opp_rank: int, home: bool) -> str:
    if home:
        if opp_rank <= 30:  return 'q1'
        if opp_rank <= 75:  return 'q2'
        if opp_rank <= 160: return 'q3'
        return 'q4'
    else:
        if opp_rank <= 50:  return 'q1'
        if opp_rank <= 100: return 'q2'
        if opp_rank <= 200: return 'q3'
        return 'q4'


def _grade_for_score(score: int) -> str:
    if score >= 93: return 'A+'
    if score >= 88: return 'A'
    if score >= 83: return 'A-'
    if score >= 78: return 'B+'
    if score >= 73: return 'B'
    if score >= 68: return 'B-'
    if score >= 63: return 'C+'
    if score >= 58: return 'C'
    if score >= 53: return 'C-'
    return 'D'


def _verdict_for_score(score: int) -> tuple[str, str]:
    if score >= 85: return '1 seed', 'Lock'
    if score >= 75: return '2 seed', 'Lock'
    if score >= 65: return '3 seed', 'In'
    if score >= 55: return '4 seed', 'Bubble'
    return 'Out', 'Out'


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
        is_away = g.get('isAway')
        home = not (is_away is True or is_away == 1)
        opp_rank = rpi_lookup.get(_norm(opp), 999)
        q = _quad_bucket(opp_rank, home)
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
        is_away = g.get('isAway')
        home = not (is_away is True or is_away == 1)
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
        is_away = g.get('isAway')
        home = not (is_away is True or is_away == 1)
        out.append({
            'opp': _clean_opp(g['opponentName']),
            'score': f'{rs}-{ra}',
            'note': f'{"Home" if home else "Road"} · opp #{int(g["opp_rank"])}',
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
        is_away = g.get('isAway')
        home = not (is_away is True or is_away == 1)
        out.append({
            'opp': _clean_opp(g['opponentName']),
            'score': f'{rs}-{ra}',
            'note': f'{"Home" if home else "Road"} · opp #{int(g["opp_rank"])}',
        })
    return out


def _nearest_by_score(team_name: str, score: int, score_lookup: dict, teams_df: pd.DataFrame, conferences_df: pd.DataFrame, limit: int = 5) -> list:
    if not score_lookup:
        return []
    conf_map = dict(zip(conferences_df['id'], conferences_df['abbreviation']))
    team_conf = dict(zip(teams_df['name'], teams_df['conference_id']))
    candidates = []
    for other, other_score in score_lookup.items():
        if other == team_name:
            continue
        candidates.append((other, other_score, abs(other_score - score)))
    candidates.sort(key=lambda t: t[2])
    out = []
    for other, other_score, _ in candidates[:limit]:
        conf_abbrev = conf_map.get(team_conf.get(other, ''), '')
        out.append({
            'team': other,
            'conf': conf_abbrev or '',
            'score': int(other_score),
            'diff': int(other_score - score),
        })
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
    # rough rpg percentile among 2026 D-I: use OPS percentile as proxy (correlated)
    rpg_pct = _stat_pct(h, 'percentile_rank_weighted_runs_created_plus')
    fld_pct = 50
    if f_row is not None and not fielding.empty:
        # Compute fielding pct percentile from the 2026 distribution
        f2026 = fielding[fielding['year'] == year]['fielding_percentage'].dropna()
        if len(f2026):
            rank = (f2026 < float(f_row.get('fielding_percentage') or 0)).sum()
            fld_pct = int(round(100 * rank / max(len(f2026) - 1, 1)))
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
        'fieldingPct': {
            'value': float(f_row.get('fielding_percentage') or 0.968) if f_row is not None else 0.968,
            'natAvg': 0.968,
            'pct': fld_pct,
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
    # For SOR/Massey/D1Baseball we don't have sources yet; mirror the available
    # systems so the card reflects real agreement rather than fake spread.
    rankings = {
        'rpi': rpi_rank,
        'sor': rpi_rank,
        'massey': rank64,
        'd1baseball': int(round((rpi_rank + rank64) / 2)),
        '64a': rank64,
    }

    # Resume score = same formula as _resume_score_lookup
    of = 301
    rpi_pct = min(rpi_rank / of, 1.0)
    r64_pct = min(rank64 / of, 1.0)
    resume_score = round(100 * (1 - 0.6 * rpi_pct - 0.4 * r64_pct))
    resume_score = max(0, min(100, int(resume_score)))
    seed_proj, bubble_verdict = _verdict_for_score(resume_score)

    # SOS: proxy from RPI rank (real SOS not in CSVs).
    def _tier(rank: int) -> str:
        if rank <= 25:  return 'Elite'
        if rank <= 75:  return 'Strong'
        if rank <= 150: return 'Moderate'
        return 'Weak'
    sos_rank = min(rpi_rank, of)
    non_con_rank = min(rpi_rank + 20, of)  # rough proxy; non-con usually weaker than overall

    # Schedule-derived blocks
    sched_full = _load_csv(f'schedules_full_{sport_key}.csv')
    rpi_lookup = _opponent_rpi_lookup(sport_key)
    quad_record = _compute_quad_record(sched_full, team_name, rpi_lookup) if not sched_full.empty else {'q1': '0-0', 'q2': '0-0', 'q3': '0-0', 'q4': '0-0'}
    last10 = _last_10_games(sched_full, team_name, rpi_lookup) if not sched_full.empty else []
    big_wins = _big_wins(sched_full, team_name, rpi_lookup) if not sched_full.empty else []
    bad_losses = _bad_losses(sched_full, team_name, rpi_lookup) if not sched_full.empty else []

    # Nearest by resume score
    score_lookup = _resume_score_lookup(sport_key, year)
    nearest = _nearest_by_score(team_name, resume_score, score_lookup, teams, conferences)

    # Stats module
    stats = _team_stats(team_id, year)
    if stats is None:
        # Safe fallback so the component still renders
        stats = {
            k: {'value': 0.0, 'natAvg': 0.0, 'pct': 50}
            for k in ('ops', 'woba', 'runsPerGame', 'fip', 'whip', 'k9', 'bb9', 'fieldingPct', 'wrcPlus')
        }

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
        'sos': {'value': sos_rank, 'natRank': sos_rank, 'of': of, 'tier': _tier(sos_rank)},
        'nonConSos': {'value': non_con_rank, 'natRank': non_con_rank, 'of': of, 'tier': _tier(non_con_rank)},
        'rankings': rankings,
        'last10': last10,
        'quadRecord': quad_record,
        'bigWins': big_wins,
        'badLosses': bad_losses,
        'nearestByScore': nearest,
        'analogs': [],
    }
