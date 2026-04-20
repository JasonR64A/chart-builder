"""
Shared Predicted RPI computation used by both Win Generator and Bracketology.

Uses the same pre-game WP model as the Win Probability page and the Win
Generator's Team Projection: compressed log5 + HFA + 50/50 team-rank/player
blend + auto-detected Weekend/Midweek series context. Single source of truth
lives in pages/_win_prob_model.py.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from app_lib.win_prob_model import (
    build_team_profiles, detect_game_context, compute_matchup_wp,
    build_rank_pct_map,
)


def compute_predicted_rpi_for_bracketology(sport: str, DATA_DIR: Path) -> pd.DataFrame:
    """
    Compute Predicted RPI for all D1 teams using the unified WP pipeline.
    Returns DataFrame with: team, pred_rpi, pred_rpi_rank, proj_wins, proj_losses.

    Remaining-game wins are expected values (sum of per-game WP) using the
    full pipeline. Played games come straight from schedules_full result
    column. Game-count source = schedule for both played and remaining, so
    results line up with Predicted RPI display / Bracketology.
    """
    # Load True Rank components
    teams = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False)
    tr = pd.read_csv(DATA_DIR / 'team_rank.csv', low_memory=False)
    rpi_df = pd.read_csv(DATA_DIR / f'{sport}_rpi_D1.csv', low_memory=False)
    confs = pd.read_csv(DATA_DIR / 'conferences.csv', low_memory=False)

    sport_label = 'Baseball' if sport == 'baseball' else 'Softball'
    sport_teams = teams[teams['sport'] == sport_label].copy()

    tr_2026 = tr[tr['year'] == 2026][['team_id', 'integer_64_rank_total']].copy()
    tr_2026.columns = ['team_id', 'rank_64a']
    ranked = sport_teams.merge(tr_2026, left_on='id', right_on='team_id', how='left')
    ranked['rank_rpi'] = ranked['name'].map(dict(zip(rpi_df['teamName'], rpi_df['rank'])))

    nm_file = DATA_DIR / 'rankings' / 'name_map.csv'
    ext_map: dict = {}
    if nm_file.exists():
        nm = pd.read_csv(nm_file)
        ext_map = dict(zip(nm['external_name'], nm['our_name']))

    for src, col in [('massey', 'rank_massey'), ('dsr', 'rank_dsr')]:
        f = DATA_DIR / 'rankings' / f'{src}_{sport}.csv'
        if f.exists():
            d = pd.read_csv(f, low_memory=False)
            ranked[col] = ranked['name'].map({ext_map.get(t, t): r for t, r in zip(d['team'], d['rank'])})
        else:
            ranked[col] = np.nan

    rcols = ['rank_64a', 'rank_rpi', 'rank_massey', 'rank_dsr']
    ranked['true_rank'] = ranked[rcols].mean(axis=1).fillna(ranked['rank_64a'])
    # Median fallback for teams missing every ranking
    median_rank = float(ranked['true_rank'].median()) if ranked['true_rank'].notna().any() else 150.0
    ranked['true_rank'] = ranked['true_rank'].fillna(median_rank)

    # Build rank_pct map for D1 (so percentiles are computed within-division)
    d1_ids = set(confs[(confs['division'] == 'D-I') & (confs['name'] != 'Big Sky Conference')]['id'])
    d1_team_names = set(teams[(teams['sport'] == sport_label) & (teams['conference_id'].isin(d1_ids))]['name'])

    ranked_d1 = ranked[ranked['name'].isin(d1_team_names)][['name', 'true_rank']].copy()
    ranked_d1.columns = ['team_name', 'rank']
    rank_pct_map = build_rank_pct_map(ranked_d1, 'rank')

    # Load player-driven profiles (D1 for this sport)
    profiles = build_team_profiles(sport_label, 'D1')

    # Schedule is the single source of truth for both played and remaining
    sched_file = DATA_DIR / f'schedules_full_{sport}.csv'
    if not sched_file.exists():
        return pd.DataFrame()

    sched = pd.read_csv(sched_file, low_memory=False)

    team_wins: dict = {}
    team_games: dict = {}
    team_opponents: dict = {}

    # Played games: actual W/L from the result column
    played = sched[sched['result'].notna() & (sched['result'] != '')]
    for tn, grp in played.groupby('teamName'):
        team_wins[tn] = float(grp['result'].str.startswith('W').sum())
        team_games[tn] = float(len(grp))
        team_opponents[tn] = grp['opponentName'].apply(
            lambda x: str(x).split('@')[0].strip()).tolist()

    # Remaining games: expected wins via the full pipeline
    remaining = sched[(sched['result'].isna()) | (sched['result'] == '')]
    # Group by team so we can feed detect_game_context the team's own schedule
    for tn, grp in remaining.groupby('teamName'):
        full_team_sched = sched[sched['teamName'] == tn]  # both played + remaining for series detection
        for _, g in grp.iterrows():
            opp = str(g.get('opponentName', '')).split('@')[0].strip()
            is_home = not (pd.notna(g.get('isAway')) and g.get('isAway') == 1.0)
            home = tn if is_home else opp
            away = opp if is_home else tn
            gtype, gnum = detect_game_context(g['date'], opp, full_team_sched)
            if home in rank_pct_map and away in rank_pct_map:
                wp = compute_matchup_wp(home, away, is_home, gtype, gnum,
                                         rank_pct_map, profiles, sport=sport_label)
            else:
                wp = 0.5
            team_wins[tn] = team_wins.get(tn, 0) + wp
            team_games[tn] = team_games.get(tn, 0) + 1
            team_opponents.setdefault(tn, []).append(opp)

    # Location-weighted WP (kept unchanged — still the Bracketology convention)
    team_loc: dict = {'home': {}, 'away': {}, 'neutral': {}}
    for tn, grp in sched.groupby('teamName'):
        for _, g in grp.iterrows():
            if pd.notna(g.get('isAway')) and g['isAway'] == 1.0:
                team_loc['away'][tn] = team_loc['away'].get(tn, 0) + 1
            elif '@' in str(g.get('opponentName', '')):
                team_loc['neutral'][tn] = team_loc['neutral'].get(tn, 0) + 1
            else:
                team_loc['home'][tn] = team_loc['home'].get(tn, 0) + 1

    wp_lookup: dict = {}
    for t in team_games:
        tw = team_wins.get(t, 0)
        tg = team_games.get(t, 0)
        if tg == 0:
            wp_lookup[t] = 0.5
            continue
        wp_raw = tw / tg
        hg = team_loc['home'].get(t, 0)
        ag = team_loc['away'].get(t, 0)
        ng = team_loc['neutral'].get(t, 0)
        if hg + ag + ng > 0:
            wc = wp_raw * (hg * 0.7 + ag * 1.3 + ng * 1.0)
            lc = (1 - wp_raw) * (hg * 1.3 + ag * 0.7 + ng * 1.0)
            wp_lookup[t] = wc / (wc + lc) if (wc + lc) > 0 else 0.5
        else:
            wp_lookup[t] = wp_raw

    owp_lookup: dict = {}
    for t in wp_lookup:
        opps = team_opponents.get(t, [])
        owps = [wp_lookup.get(o, 0.5) for o in opps if o in wp_lookup]
        owp_lookup[t] = float(np.mean(owps)) if owps else 0.5

    true_rank_lookup = dict(zip(ranked['name'], ranked['true_rank']))

    results = []
    for t in wp_lookup:
        opps = team_opponents.get(t, [])
        oowps = [owp_lookup.get(o, 0.5) for o in opps if o in owp_lookup]
        oowp = float(np.mean(oowps)) if oowps else 0.5
        results.append({
            'team': t,
            'pred_rpi': 0.25 * wp_lookup[t] + 0.50 * owp_lookup.get(t, 0.5) + 0.25 * oowp,
            'proj_wins': round(team_wins.get(t, 0)),
            'proj_losses': round(team_games.get(t, 0) - team_wins.get(t, 0)),
        })

    df = pd.DataFrame(results).sort_values('pred_rpi', ascending=False)
    df = df[df['team'].isin(d1_team_names)].reset_index(drop=True)
    df['pred_rpi_rank'] = range(1, len(df) + 1)
    df['true_rank'] = df['team'].map(true_rank_lookup)
    df['true_rank_d1'] = df['true_rank'].rank(method='min').fillna(len(df)).astype(int)
    df['final_rank_score'] = 0.70 * df['true_rank_d1'] + 0.30 * df['pred_rpi_rank']
    df = df.sort_values(['final_rank_score', 'pred_rpi'], ascending=[True, False]).reset_index(drop=True)
    df['final_rank'] = range(1, len(df) + 1)
    return df
