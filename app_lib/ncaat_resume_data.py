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
import base64
import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

try:
    from PIL import Image  # for dominant-color extraction
except ImportError:
    Image = None

_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _APP_DIR / 'data'


def _strip_conf_tourney_tags(s: str) -> str:
    """Strip conference-tournament tags the schedule scraper adds to opponent
    names so they resolve to the plain team. Two forms:
      - leading seed prefix:  "#3 UC San Diego"  -> "UC San Diego"
      - trailing event suffix: "Adrian 2026 MIAA Baseball Tournament" -> "Adrian"
        (the "@City, ST (...)" form is already removed by the @/vs split).
    Deliberately does NOT strip a trailing "(XX)" — that would break legit
    disambiguators in teams.csv ("LMU (CA)", "Miami (FL)" vs "Miami (OH)").
    """
    if not isinstance(s, str):
        return ''
    s = re.sub(r'^\s*#\d+\s+', '', s)               # "#3 " seed prefix
    s = re.sub(r'\s+(?:19|20)\d{2}\b.*$', '', s)     # " 2026 ... Championship/Tournament"
    s = re.sub(r'\s+\d{2}-\d{2}\b.*$', '', s)        # " 25-26 ... Tournament"
    return s.strip()


def _norm(name: str) -> str:
    """Normalize a team name for cross-file matching.
    - Strip " @Location, ST" / " vs X" suffixes (schedules_full convention)
    - Lowercase
    - Collapse "State" and "Saint" to "st" so "Mississippi St." (teams.csv),
      "Mississippi State" (DSR), and "Saint Mary's" / "St. Mary's" variants
      all share one normalized form.
    - Drop "University" / "Univ." suffixes so "Lamar" (teams.csv / RPI) and
      "Lamar University" (schedules_full / PBP) collapse to one form.
    - Preserve "&" as a literal "and" token so "Missouri S&T" (D-II GLVC)
      and "Missouri St." (D-I CUSA) don't both collapse to "missourist"
      and overwrite each other in name->conference lookups. Same for the
      A&M family.
    - Strip non-alphanumeric.
    """
    if not isinstance(name, str):
        return ''
    n = re.split(r'\s+@|\s+vs\s+', name, maxsplit=1)[0]
    n = _strip_conf_tourney_tags(n)
    n = n.lower().strip()
    n = n.replace('&', 'and')
    n = re.sub(r'\bsaint\b', 'st', n)
    n = re.sub(r'\bstate\b', 'st', n)
    n = re.sub(r'\buniversity\b|\buniv\.?\b', '', n)
    n = re.sub(r'[^a-z0-9]+', '', n)
    return n


def _is_completed(g) -> bool:
    """Row represents a game that's actually been played.

    isFuture=1/True flags a future-scheduled game and must be excluded.
    BUT some schedule rows have isFuture=NaN AND runsFor=NaN — those are
    games that have been scheduled but haven't been played yet (the
    schedule scraper hasn't restamped them yet either way). They were
    silently counted as losses (since isWin=NaN→False), inflating each
    team's loss count by every still-pending game on its slate.

    True only if isFuture is NOT a positive flag AND we have actual run
    totals on both sides."""
    v = g.get('isFuture')
    if v is True or v == 1 or v == 1.0 or str(v).lower() == 'true':
        return False
    rf, ra = g.get('runsFor'), g.get('runsAgainst')
    try:
        # NaN comparisons return False; this guards against unplayed games.
        if rf is None or ra is None:
            return False
        return not (pd.isna(rf) or pd.isna(ra))
    except Exception:
        return False


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
    # Route through _SOURCE_TEAM_ALIASES so RPI's spelling (e.g. "Loyola
    # Marymount") also resolves under the teams.csv canonical ("LMU (CA)").
    return _register_ranks_with_aliases(df, 'teamName', 'rank')


@st.cache_data(show_spinner=False)
def _actual_field_lookup(sport_key: str, year: int) -> dict:
    """team_db_id -> {'seed': int, 'regional': str} from the ACTUAL committee
    field file (data/bracketology/tournament_field_{sport}_{year}.csv). Empty
    dict if no actual-field file exists (e.g. softball / future years), in
    which case the resume Verdict stays the projection."""
    p = DATA_DIR / 'bracketology' / f'tournament_field_{sport_key}_{year}.csv'
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out = {}
    for _, r in df.iterrows():
        tid = pd.to_numeric(r.get('team_db_id'), errors='coerce')
        if pd.isna(tid):
            continue
        seed = pd.to_numeric(r.get('seed'), errors='coerce')
        out[int(tid)] = {
            'seed': int(seed) if pd.notna(seed) else None,
            'regional': str(r.get('regional', '') or '').strip(),
        }
    return out


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
    rpi_by_name = _opponent_rpi_lookup(sport_key)
    rank64 = _sixty_four_lookup(year)
    out = {}
    for _, r in d1.iterrows():
        name = r['name']
        rpi_r = rpi_by_name.get(_norm(name), 301)
        r64 = rank64.get(int(r['id']), 301)
        out[name] = _resume_score_from_ranks(rpi_r, r64)
    return out


_LOGO_DIR = _APP_DIR / 'team_logos_512'


@st.cache_data(show_spinner=False)
def _team_logo_map(sport_key: str) -> dict:
    """team_name -> logo_id. Logos in team_logos_512/ are named by BASEBALL
    team_id; softball entries don't have their own logo file. So for softball
    teams, fall back to the baseball ID for the same school name. (Same
    pattern chart_builder.py uses, originally missed in this module — that's
    why softball Team Resume cards had no logos. Fixed 2026-04-29.)"""
    teams = _load_csv('teams.csv')
    if teams.empty:
        return {}
    bb_map = {n: int(i) for n, i in zip(
        teams[teams['sport']=='Baseball']['name'],
        teams[teams['sport']=='Baseball']['id'],
    ) if pd.notna(i)}
    if sport_key == 'baseball':
        return bb_map
    # Softball: prefer SB team_id only if that file exists; otherwise fall back
    # to the baseball ID for the same school name.
    sb_rows = teams[teams['sport']=='Softball']
    out = {}
    for n, i in zip(sb_rows['name'], sb_rows['id']):
        if pd.isna(i):
            continue
        sb_id = int(i)
        # Prefer SB-specific logo if file exists
        if (_LOGO_DIR / f'{sb_id}.png').exists() or (_LOGO_DIR / f'{sb_id}.webp').exists():
            out[n] = sb_id
        elif n in bb_map:
            out[n] = bb_map[n]
    return out


def _team_logo_data_uri(team_name: str, sport_key: str) -> str:
    """Base64 data URI for team_logos_512/{id}.png; '' if missing."""
    logo_id = _team_logo_map(sport_key).get(team_name)
    if not logo_id:
        return ''
    for ext in ('png', 'webp'):
        p = _LOGO_DIR / f'{logo_id}.{ext}'
        if p.exists():
            try:
                data = p.read_bytes()
                mime = 'image/png' if ext == 'png' else 'image/webp'
                return f'data:{mime};base64,{base64.b64encode(data).decode()}'
            except Exception:
                return ''
    return ''


def _lum(hex_color: str) -> float:
    """Relative luminance 0-1 from a '#rrggbb' string."""
    h = hex_color.lstrip('#')
    if len(h) != 6:
        return 0.5
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return 0.5
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


def _contrast_on(hex_color: str) -> str:
    """Return a readable text color (#000 or #fff) for text placed over hex_color."""
    return '#ffffff' if _lum(hex_color) < 0.55 else '#0a0a0a'


@st.cache_data(show_spinner=False)
def _team_primary_color(team_name: str, sport_key: str) -> str:
    """Extract a dominant color from the team logo. Returns a hex string.
    Falls back to 64A brand red if the logo is missing or unusable."""
    fallback = '#C41230'
    if Image is None:
        return fallback
    logo_id = _team_logo_map(sport_key).get(team_name)
    if not logo_id:
        return fallback
    for ext in ('png', 'webp'):
        p = _LOGO_DIR / f'{logo_id}.{ext}'
        if not p.exists():
            continue
        try:
            img = Image.open(p).convert('RGBA')
            img.thumbnail((64, 64))
            pixels = np.array(img)
            mask = pixels[:, :, 3] > 128
            rgb = pixels[mask][:, :3]
            if not len(rgb):
                return fallback
            filtered = []
            for r, g, b in rgb:
                brightness = (int(r) + int(g) + int(b)) / 3
                if brightness > 220 or brightness < 35:
                    continue
                filtered.append((int(r), int(g), int(b)))
            if not filtered:
                return fallback
            quantized = [(r // 16 * 16, g // 16 * 16, b // 16 * 16) for r, g, b in filtered]
            most_common = Counter(quantized).most_common(1)[0][0]
            return f'#{most_common[0]:02x}{most_common[1]:02x}{most_common[2]:02x}'
        except Exception:
            return fallback
    return fallback


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


# Sources use their own team-name conventions that simple suffix normalization
# can't reconcile (e.g. DSR/Massey use acronyms like "USC" for schools that
# teams.csv spells out as "Southern California"). Each entry maps a
# SOURCE-specific team name to the canonical teams.csv name so both forms
# resolve to the same rank in every ranking lookup.
_SOURCE_TEAM_ALIASES = {
    'USC': 'Southern California',                   # DSR, Massey, ELO all use "USC"
    'South Carolina Upstate': 'USC Upstate',        # ELO spells it out; teams.csv = "USC Upstate"
    'Loyola Marymount': 'LMU (CA)',                 # RPI/DSR use full name; teams.csv = "LMU (CA)"
    'Loyola-Marymount': 'LMU (CA)',                 # ELO hyphenates
    # Ranking sources spell these out; teams.csv (and schedule opponents) use
    # the short form, so without the alias they showed rank #999 both on their
    # own resume and as opponents (e.g. Binghamton's losses to "DBU").
    'Dallas Baptist': 'DBU',
    'UMass': 'Massachusetts',                       # NOT "UMass Lowell" (separate school)
    'North Alabama': 'North Ala.',
    'Incarnate Word': 'UIW',
    'UNCG': 'UNC Greensboro',
}


def _register_ranks_with_aliases(df, team_col, rank_col) -> dict:
    """Build {_norm(name): rank} from a ranking df, registering both the
    source's spelling AND any canonical teams.csv alias so either form
    resolves to the same rank.
    """
    out = {}
    for t, r in zip(df[team_col], df[rank_col]):
        try:
            rank = int(r)
        except (TypeError, ValueError):
            continue
        out[_norm(t)] = rank
        canonical = _SOURCE_TEAM_ALIASES.get(t)
        if canonical:
            out[_norm(canonical)] = rank
    return out


@st.cache_data(show_spinner=False)
def _dsr_rank_lookup(sport_key: str) -> dict:
    """Map normalized team name -> DSR rank from data/rankings/dsr_*.csv.
    Handles the USC/Southern California mismatch via _SOURCE_TEAM_ALIASES.
    """
    fname = f'rankings/dsr_{sport_key}.csv'
    df = _load_csv(fname)
    if df.empty:
        return {}
    # Keep the most recent date (the file usually carries a single daily snapshot,
    # but some appenders leave history).
    if 'date' in df.columns:
        latest = df['date'].max()
        df = df[df['date'] == latest]
    return _register_ranks_with_aliases(df, 'team', 'rank')


@st.cache_data(show_spinner=False)
def _massey_rank_lookup(sport_key: str) -> dict:
    """Map normalized team name -> Massey rank from data/rankings/massey_*.csv.
    Routes through _SOURCE_TEAM_ALIASES so e.g. Massey's "USC" resolves for
    the teams.csv "Southern California" lookup key.
    """
    fname = f'rankings/massey_{sport_key}.csv'
    df = _load_csv(fname)
    if df.empty:
        return {}
    if 'date' in df.columns:
        latest = df['date'].max()
        df = df[df['date'] == latest]
    return _register_ranks_with_aliases(df, 'team', 'rank')


@st.cache_data(show_spinner=False)
def _elo_rank_lookup(sport_key: str) -> dict:
    """Map normalized team name -> ELO rank from data/rankings/elo_*.csv (Warren Nolan).
    Aliases via _SOURCE_TEAM_ALIASES so "USC" -> "Southern California" and
    "South Carolina Upstate" -> "USC Upstate" resolve correctly.
    """
    fname = f'rankings/elo_{sport_key}.csv'
    df = _load_csv(fname)
    if df.empty:
        return {}
    if 'date' in df.columns:
        latest = df['date'].max()
        df = df[df['date'] == latest]
    return _register_ranks_with_aliases(df, 'team', 'rank')


@st.cache_data(show_spinner=False)
def _team_year_id_lookup(sport_key: str) -> dict:
    """Map 64A team name (teams.csv) -> NCAA teamYearId used by
    schedules_full_*.csv. Built once so every downstream schedule filter can
    use the ID instead of re-matching on names (which breaks for Lamar and
    similar suffix drift).
    """
    sched = _load_csv(f'schedules_full_{sport_key}.csv')
    teams = _load_csv('teams.csv')
    if sched.empty or teams.empty:
        return {}
    sport_label = 'Baseball' if sport_key == 'baseball' else 'Softball'
    # First teamYearId per schedule teamName (a single value per team-season)
    sched_id_by_name = sched.groupby('teamName')['teamYearId'].first().to_dict()
    # Normalize schedule names once
    sched_by_norm = {_norm(k): int(v) for k, v in sched_id_by_name.items() if pd.notna(v)}
    # Map 64A team name -> teamYearId via the normalized bridge
    out = {}
    for _, r in teams[teams['sport'] == sport_label].iterrows():
        name = r.get('name')
        if not isinstance(name, str):
            continue
        tid = sched_by_norm.get(_norm(name))
        if tid is not None:
            out[name] = tid
    return out


@st.cache_data(show_spinner=False)
def _sos_lookup(sport_key: str) -> dict:
    """Compute NCAA-style SOS for every team with completed games and return
    {teamYearId: {overall_rank, noncon_rank, overall_score, noncon_score, of}}.
    Keying by NCAA teamYearId (not team name) eliminates name-drift bugs
    like "Lamar" vs "Lamar University" between teams.csv and schedules_full.

    NCAA SOS = 2/3 * OWP + 1/3 * OOWP.
    """
    sched_full = _load_csv(f'schedules_full_{sport_key}.csv')
    if sched_full.empty:
        return {}
    teams_df = _load_csv('teams.csv')
    conferences_df = _load_csv('conferences.csv')
    sport_label = 'Baseball' if sport_key == 'baseball' else 'Softball'

    # teamYearId <-> conference_id. Built via the schedule teamName -> 64A name
    # bridge using the normalized form so suffix variants collapse.
    conf_by_64a_name = dict(zip(
        teams_df[teams_df['sport'] == sport_label]['name'],
        teams_df[teams_df['sport'] == sport_label]['conference_id'],
    ))
    norm_to_conf = {_norm(n): c for n, c in conf_by_64a_name.items()}
    sched_first = sched_full.groupby('teamYearId')['teamName'].first().to_dict()
    conf_by_year_id = {int(tyid): norm_to_conf.get(_norm(n))
                       for tyid, n in sched_first.items() if pd.notna(tyid)}

    # Build the D-I team-year-id set for ranking restriction.
    di_confs = set(conferences_df[conferences_df['division'] == 'D-I']['id'].tolist())
    di_year_ids = {tyid for tyid, c in conf_by_year_id.items() if c in di_confs}

    df = sched_full.copy()
    df = df[df.apply(_is_completed, axis=1)]
    df['is_win_flag'] = df.apply(_is_win, axis=1)
    # Drop rows without a usable opponentYearId — those games can't join.
    df = df[df['teamYearId'].notna() & df['opponentYearId'].notna()].copy()
    df['teamYearId'] = df['teamYearId'].astype(int)
    df['opponentYearId'] = df['opponentYearId'].astype(int)

    # Team WP over all completed games (keyed on teamYearId)
    grouped = df.groupby('teamYearId')['is_win_flag'].agg(['sum', 'count'])
    grouped['wp'] = grouped['sum'] / grouped['count'].clip(lower=1)
    wp_by_tid = grouped['wp'].to_dict()

    # Per-team opponent lists (overall + non-con), keyed on teamYearId
    team_opps = {}
    team_opps_nc = {}
    for tid, chunk in df.groupby('teamYearId'):
        opps = chunk['opponentYearId'].tolist()
        team_opps[tid] = opps
        team_conf = conf_by_year_id.get(int(tid))
        nc = []
        for opp_tid in opps:
            opp_conf = conf_by_year_id.get(int(opp_tid))
            if team_conf is None or opp_conf is None or opp_conf != team_conf:
                nc.append(opp_tid)
        team_opps_nc[tid] = nc

    def _owp_for(opps):
        vals = [wp_by_tid.get(o) for o in opps if o in wp_by_tid]
        return sum(vals) / len(vals) if vals else 0.0

    owp_by_tid = {t: _owp_for(os) for t, os in team_opps.items()}
    owp_nc_by_tid = {t: _owp_for(os) for t, os in team_opps_nc.items()}

    sos_overall = {}
    sos_noncon = {}
    for tid, opps in team_opps.items():
        oo_vals = [owp_by_tid.get(o) for o in opps if o in owp_by_tid]
        oowp = sum(oo_vals) / len(oo_vals) if oo_vals else 0.0
        sos_overall[tid] = (2/3) * owp_by_tid.get(tid, 0) + (1/3) * oowp

        opps_nc = team_opps_nc.get(tid, [])
        oo_nc_vals = [owp_by_tid.get(o) for o in opps_nc if o in owp_by_tid]
        oowp_nc = sum(oo_nc_vals) / len(oo_nc_vals) if oo_nc_vals else 0.0
        sos_noncon[tid] = (2/3) * owp_nc_by_tid.get(tid, 0) + (1/3) * oowp_nc

    def _rank(scores: dict) -> dict:
        di_scores = [(t, s) for t, s in scores.items() if int(t) in di_year_ids]
        di_scores.sort(key=lambda t: -t[1])
        return {int(t): i + 1 for i, (t, _) in enumerate(di_scores)}

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
    that we can use to find historical analogs. Supports baseball and softball.
    """
    rpi_hist = _load_csv('bracketology/historical_selection_rpi.csv')
    brackets = _load_csv('bracketology/historical_brackets.csv')
    results = _load_csv('bracketology/historical_results.csv')
    if rpi_hist.empty or brackets.empty:
        return pd.DataFrame()
    rpi_hist = rpi_hist[rpi_hist['sport'] == sport_key].copy()
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
    brackets_sport = brackets[brackets['sport'] == sport_key].copy()
    brackets_sport['name_norm'] = brackets_sport['team'].apply(_norm)
    # National seed: only for teams that hosted AND were 1-seed in their regional.
    host_rows = brackets_sport[
        (brackets_sport['team'] == brackets_sport['host_team']) &
        (brackets_sport['regional_seed'] == 1)
    ][['year', 'name_norm', 'national_seed']].rename(columns={'national_seed': 'nat_seed'})
    # Regional seed: each team's 1-4 slot within their regional (everyone has one).
    reg_seed_rows = brackets_sport[['year', 'name_norm', 'regional_seed']].drop_duplicates(['year', 'name_norm'])
    out = rpi_hist.merge(host_rows, on=['year', 'name_norm'], how='left')
    out = out.merge(reg_seed_rows, on=['year', 'name_norm'], how='left')
    # Attach result + tournament W-L if available, filtered by sport when the column exists
    if not results.empty:
        results_c = results.copy()
        if 'sport' in results_c.columns:
            results_c = results_c[results_c['sport'] == sport_key]
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


def _compute_remaining_quadrants(sched_full: pd.DataFrame, team_year_id, rpi_lookup: dict, sport_key: str = 'baseball') -> dict:
    """Return a list of upcoming games per quadrant:
      {'q1': [{opp, venue, date, oppRank}, ...], 'q2': [...], ...}
    Venue: 'home' | 'neutral' | 'away'.
    Games without a ranked opponent bucket as Q4.
    """
    if team_year_id is None:
        return {'q1': [], 'q2': [], 'q3': [], 'q4': []}
    m = sched_full[sched_full['teamYearId'] == team_year_id].copy()
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
        q = _quad_bucket(opp_rank, venue, sport_key)
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


def _quad_bucket(opp_rank: int, venue: str, sport_key: str = 'baseball') -> str:
    """NCAA quadrant thresholds — different per sport.
    Baseball: Q1 H1-25 N1-40 A1-60; Q2 H26-50 N41-80 A61-120;
              Q3 H51-100 N81-160 A121-240; Q4 H101+ N161+ A241+.
    Softball: location-independent — Q1 1-25, Q2 26-50, Q3 51-100, Q4 101+.
    """
    if sport_key.lower() == 'softball':
        # Venue ignored — softball uses pure RPI rank thresholds.
        if opp_rank <= 25:  return 'q1'
        if opp_rank <= 50:  return 'q2'
        if opp_rank <= 100: return 'q3'
        return 'q4'
    # baseball (default)
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
    """Legacy score-based verdict. Kept for back-compat callers; new code
    should use _verdict_for_consensus(avg_rank) below."""
    if score >= 88: return '1 seed', 'Lock'
    if score >= 80: return '2 seed', 'Lock'
    if score >= 75: return '3 seed', 'In'
    if score >= 70: return '4 seed', 'In'
    if score >= 64: return 'Bubble', 'Bubble'
    return 'Out', 'Out'


def _verdict_for_consensus(avg_rank: int) -> tuple[str, str]:
    """Verdict + seed projection driven by computer-consensus average rank
    (RPI/DSR/Massey/ELO/64A). Bands the user owns:
      Verdict — 1-16 Lock, 17-34 In, 35-50 Bubble, 51+ Out
      Seed   — 1-12 "1 seed", 13-25 "2 seed", 26-35 "2-3 seed",
               36-50 "3 seed", 51+ "Out"
    Verdict and seed-projection bands intentionally don't line up — Verdict
    asks "are they in?" and seed asks "if seeded, where?".
    """
    r = int(avg_rank)
    if r <= 16:   verdict = 'Lock'
    elif r <= 34: verdict = 'In'
    elif r <= 50: verdict = 'Bubble'
    else:         verdict = 'Out'
    if r <= 12:   seed = '1 seed'
    elif r <= 25: seed = '2 seed'
    elif r <= 35: seed = '2-3 seed'
    elif r <= 50: seed = '3 seed'
    else:         seed = 'Out'
    return seed, verdict


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
    """Strip ' @Location, ST' / ' vs X' suffixes AND conference-tournament tags
    ('#3 UC San Diego', 'Adrian 2026 MIAA Baseball Tournament') from opponent
    names so the displayed/looked-up opponent is the plain team."""
    if not isinstance(name, str):
        return ''
    base = re.split(r'\s+@|\s+vs\s+', name, maxsplit=1)[0]
    return _strip_conf_tourney_tags(base)


def _compute_quad_record(sched_full: pd.DataFrame, team_year_id, rpi_lookup: dict, sport_key: str = 'baseball') -> dict:
    m = sched_full[sched_full['teamYearId'] == team_year_id] if team_year_id is not None else sched_full.iloc[0:0]
    quads = {'q1': [0, 0], 'q2': [0, 0], 'q3': [0, 0], 'q4': [0, 0]}
    for _, g in m.iterrows():
        if not _is_completed(g):
            continue
        opp = g.get('opponentName')
        if not isinstance(opp, str):
            continue
        venue = _venue(g)
        opp_rank = rpi_lookup.get(_norm(opp), 999)
        q = _quad_bucket(opp_rank, venue, sport_key)
        if _is_win(g):
            quads[q][0] += 1
        else:
            quads[q][1] += 1
    return {k: f'{v[0]}-{v[1]}' for k, v in quads.items()}


def _last_10_games(sched_full: pd.DataFrame, team_year_id, rpi_lookup: dict) -> list:
    if team_year_id is None:
        return []
    m = sched_full[sched_full['teamYearId'] == team_year_id].copy()
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


def _big_wins(sched_full: pd.DataFrame, team_year_id, rpi_lookup: dict, top_n: int = 3) -> list:
    if team_year_id is None:
        return []
    m = sched_full[sched_full['teamYearId'] == team_year_id].copy()
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


def _bad_losses(sched_full: pd.DataFrame, team_year_id, rpi_lookup: dict, threshold: int = 100, max_n: int = 3) -> list:
    if team_year_id is None:
        return []
    m = sched_full[sched_full['teamYearId'] == team_year_id].copy()
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
        ps = _team_stats(int(primary_tid), sport_key=sport_key)
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
            stats = _team_stats(int(tid), sport_key=sport_key)
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


_STAT_NAT_AVG = {
    'baseball': {'ops': 0.82, 'woba': 0.37, 'rpg': 6.8, 'fip': 4.95, 'whip': 1.69,
                 'k_rate': 7.6, 'bb_rate': 4.9, 'defense': 3.73, 'wrcPlus': 100},
    'softball': {'ops': 0.79, 'woba': 0.35, 'rpg': 5.0, 'fip': 4.59, 'whip': 1.65,
                 'k_rate': 4.2, 'bb_rate': 3.1, 'defense': 0.957, 'wrcPlus': 100},
}


def _team_stats(team_id: int, year: int = 2026, sport_key: str = 'baseball') -> dict | None:
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
    # Softball plays 7-inning games, baseball plays 9. Rate per game length.
    inn_len = 7 if sport_key == 'softball' else 9
    bb_rate = (bb * inn_len / ip) if ip > 0 else 0.0
    rpg_pct = _stat_pct(h, 'percentile_rank_weighted_runs_created_plus')
    # Defensive metric: baseball uses range_factor, softball uses fielding_percentage.
    # Range factor distributions are noisy for softball (different positional splits
    # and scoring conventions), so we show fielding percentage there instead.
    defense_col = 'fielding_percentage' if sport_key == 'softball' else 'range_factor'
    defense_label = 'FPCT' if sport_key == 'softball' else 'RF'
    defense_fmt = 'pct3' if sport_key == 'softball' else 'num2'
    defense_pct = 50
    defense_value = 0.0
    if f_row is not None and not fielding.empty:
        defense_value = float(f_row.get(defense_col) or 0)
        # Restrict percentile pool to teams of the same sport — the two sports have
        # structurally different distributions, so a mixed pool would skew results.
        teams_df = _load_csv('teams.csv')
        sport_label = 'Baseball' if sport_key == 'baseball' else 'Softball'
        sport_team_ids = set(teams_df[teams_df['sport'] == sport_label]['id']) if not teams_df.empty else set()
        f_sport = fielding[(fielding['year'] == year) & (fielding['team_id'].isin(sport_team_ids))]
        pool = f_sport[defense_col].dropna()
        if len(pool):
            rank = (pool < defense_value).sum()
            defense_pct = int(round(100 * rank / max(len(pool) - 1, 1)))
    # K rate: compute from raw for softball (strikeouts_per_7_innings is NaN for many teams);
    # baseball uses the precomputed strikeouts_per_9_innings.
    if sport_key == 'softball':
        strikeouts_raw = float(p.get('strikeouts') or 0)
        k_rate = (strikeouts_raw * 7.0 / ip) if ip > 0 else 0.0
    else:
        k_rate = float(p.get('strikeouts_per_9_innings') or 0)
    nat = _STAT_NAT_AVG.get(sport_key, _STAT_NAT_AVG['baseball'])
    k_label = 'K/7' if sport_key == 'softball' else 'K/9'
    bb_label = 'BB/7' if sport_key == 'softball' else 'BB/9'
    return {
        'ops': {
            'value': float(h.get('on_base_plus_slugging') or 0),
            'natAvg': nat['ops'],
            'pct': _stat_pct(h, 'percentile_rank_on_base_plus_slugging'),
        },
        'woba': {
            'value': float(h.get('weighted_on_base_average') or 0),
            'natAvg': nat['woba'],
            'pct': _stat_pct(h, 'percentile_rank_weighted_on_base_average'),
        },
        'runsPerGame': {
            'value': round(rpg, 2),
            'natAvg': nat['rpg'],
            'pct': rpg_pct,
        },
        'fip': {
            'value': float(p.get('fielding_independent_pitching') or 0),
            'natAvg': nat['fip'],
            'pct': _stat_pct(p, 'percentile_rank_fielding_independent_pitching'),
            'inverted': True,
        },
        'whip': {
            'value': float(p.get('walks_plus_hits_per_inning_pitched') or 0),
            'natAvg': nat['whip'],
            # percentile_rank_* columns in pitching_team.csv are already "higher = better"
            # (goodness percentile), so don't invert even though lower raw WHIP is better.
            'pct': _stat_pct(p, 'percentile_rank_walks_plus_hits_per_inning_pitched'),
            'inverted': True,
        },
        'k9': {
            'value': round(k_rate, 2),
            'natAvg': nat['k_rate'],
            'pct': _stat_pct(p, 'percentile_rank_strikeout_percentage'),
            'label': k_label,
        },
        'bb9': {
            'value': round(bb_rate, 2),
            'natAvg': nat['bb_rate'],
            # Same as WHIP: percentile_rank_walk_percentage is already goodness-oriented
            # (low BB% -> high percentile). Previously I had invert=True, which flipped it
            # so elite control teams like Ole Miss showed up as worst-in-class. Fixed.
            'pct': _stat_pct(p, 'percentile_rank_walk_percentage'),
            'inverted': True,
            'label': bb_label,
        },
        'rangeFactor': {
            'value': round(defense_value, 3),
            'natAvg': nat['defense'],
            'pct': defense_pct,
            'label': defense_label,
            'fmt': defense_fmt,
        },
        'wrcPlus': {
            'value': float(h.get('weighted_runs_created_plus') or 100),
            'natAvg': nat['wrcPlus'],
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

    # Rankings — RPI via the alias-aware lookup so canonical names like
    # "LMU (CA)" resolve to RPI's "Loyola Marymount".
    rpi_rank = _opponent_rpi_lookup(sport_key).get(_norm(team_name), 301)
    rank64 = _sixty_four_lookup(year).get(team_id, 301)
    dsr_rank = _dsr_rank_lookup(sport_key).get(_norm(team_name), 301)
    massey_rank = _massey_rank_lookup(sport_key).get(_norm(team_name), 301)
    elo_rank = _elo_rank_lookup(sport_key).get(_norm(team_name))
    # Warren Nolan only ranks teams that have played games / sit in the top N.
    # If the team is missing, fall back to the mean of the other four so the
    # card doesn't misleadingly show 301.
    if elo_rank is None:
        elo_rank = int(round((rpi_rank + rank64 + dsr_rank + massey_rank) / 4))
    rankings = {
        'rpi': rpi_rank,
        'dsr': dsr_rank,
        'massey': massey_rank,
        'elo': elo_rank,
        '64a': rank64,
    }

    # Resume score: same piecewise rank-to-score curve as _resume_score_lookup.
    resume_score = _resume_score_from_ranks(rpi_rank, rank64)
    # Verdict is driven by the 5-system computer-consensus average (the same
    # number rendered in the "Computer Consensus" module) rather than the
    # resume-score curve, so a team's Verdict matches what the card shows.
    consensus_avg = int(round((rpi_rank + dsr_rank + massey_rank + elo_rank + rank64) / 5))
    seed_proj, bubble_verdict = _verdict_for_consensus(consensus_avg)

    # Postseason: if the team is in the ACTUAL committee field, the Verdict
    # shows their real seed + regional instead of the projection (e.g. Cal Poly
    # -> "3 seed - Los Angeles Regional"). fieldPlacement also drives the
    # remaining-schedule block (their season is over; show the regional).
    _field_lookup = _actual_field_lookup(sport_key, year)
    field_placement = _field_lookup.get(team_id)
    if field_placement and field_placement.get('seed') and field_placement.get('regional'):
        _sd = field_placement['seed']; _reg = field_placement['regional']
        seed_proj = f"{_sd} seed"
        bubble_verdict = f"{_sd} seed - {_reg} Regional"

    # SOS: real NCAA-formula computation from schedules_full.
    def _tier(rank: int) -> str:
        if rank <= 25:  return 'Elite'
        if rank <= 75:  return 'Strong'
        if rank <= 150: return 'Moderate'
        return 'Weak'
    # Resolve team_year_id up front so all schedule-based helpers key on the ID.
    team_year_id = _team_year_id_lookup(sport_key).get(team_name)
    sos_pool = _sos_lookup(sport_key)
    sos_map = sos_pool.get(team_year_id) if team_year_id is not None else None
    # Pool size for the 'of N' display — comes from the SOS rank pool itself
    # (the count of D-I teams with completed games), not a hardcoded number.
    # Falls back to the count of teams in the pool, or 308 as a last resort.
    if sos_map and sos_map.get('of'):
        of = int(sos_map['of'])
    elif sos_pool:
        of = max(len(sos_pool), 1)
    else:
        of = 308
    if sos_map:
        sos_rank = int(sos_map['overall_rank'])
        non_con_rank = int(sos_map['noncon_rank'])
    else:
        sos_rank = of
        non_con_rank = of

    # Schedule-derived blocks — all keyed on NCAA teamYearId rather than the
    # string team name, so "Lamar" vs "Lamar University" (and similar suffix
    # drift) stops silently returning empty schedules. team_year_id resolved above.
    sched_full = _load_csv(f'schedules_full_{sport_key}.csv')
    rpi_lookup = _opponent_rpi_lookup(sport_key)
    quad_record = _compute_quad_record(sched_full, team_year_id, rpi_lookup, sport_key) if not sched_full.empty else {'q1': '0-0', 'q2': '0-0', 'q3': '0-0', 'q4': '0-0'}
    remaining_quads = _compute_remaining_quadrants(sched_full, team_year_id, rpi_lookup, sport_key) if not sched_full.empty else {'q1': [], 'q2': [], 'q3': [], 'q4': []}
    # Postseason: a team in the committee field has no remaining regular-season
    # games — any "remaining" rows are stale. Blank them so the resume shows
    # the regional (via fieldPlacement) instead of leftover schedule noise.
    if field_placement:
        remaining_quads = {'q1': [], 'q2': [], 'q3': [], 'q4': []}
    remaining_count = sum(len(v) for v in remaining_quads.values())
    projected_rpi_range = _project_rpi_range(rpi_rank, remaining_count)
    last10 = _last_10_games(sched_full, team_year_id, rpi_lookup) if not sched_full.empty else []
    big_wins = _big_wins(sched_full, team_year_id, rpi_lookup) if not sched_full.empty else []
    bad_losses = _bad_losses(sched_full, team_year_id, rpi_lookup) if not sched_full.empty else []

    # Nearest by resume score
    score_lookup = _resume_score_lookup(sport_key, year)
    nearest = _nearest_by_score(team_name, resume_score, score_lookup, teams, conferences, sport_key)

    # Stats module
    stats = _team_stats(team_id, year, sport_key=sport_key)
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

    primary = _team_primary_color(team_name, sport_key)
    secondary = _contrast_on(primary)
    logo_uri = _team_logo_data_uri(team_name, sport_key)

    return {
        'name': team_name,
        'nickname': nickname,
        'year': year,
        'conf': conf_abbrev or conf_name or '',
        'record': record,
        'confRecord': conf_record,
        'colors': {'primary': primary, 'secondary': secondary, 'accent': '#8B6F3E'},
        'logoDataUri': logo_uri,
        'location': location,
        'coach': coach,
        'resumeScore': resume_score,
        # Postseason: the field is set, so the letter grade is moot — show
        # IN / OUT instead. Only when an actual field file exists for this
        # sport/year (baseball 2026); otherwise keep the resume letter grade.
        'grade': (('IN' if field_placement else 'OUT') if _field_lookup
                  else _grade_for_score(resume_score)),
        'seedProjection': seed_proj,
        'bubbleVerdict': bubble_verdict,
        'fieldPlacement': field_placement,
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
