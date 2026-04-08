"""
Shared Predicted RPI computation used by both Win Generator and Bracketology.
Uses True Rank (64A + RPI + Massey + DSR) Log5 model.
"""

import pandas as pd
import numpy as np
from pathlib import Path

LOGISTIC_A = 0.006355505043163222
LOGISTIC_B = 151.44616903215248


def _rank_to_win_pct(rank):
    return 1.0 / (1.0 + np.exp(LOGISTIC_A * (rank - LOGISTIC_B)))


def _log5(rank_a, rank_b):
    pA = np.clip(_rank_to_win_pct(rank_a), 0.01, 0.99)
    pB = np.clip(_rank_to_win_pct(rank_b), 0.01, 0.99)
    return (pA * (1 - pB)) / (pA * (1 - pB) + pB * (1 - pA))


def compute_predicted_rpi_for_bracketology(sport: str, DATA_DIR: Path) -> pd.DataFrame:
    """
    Compute Predicted RPI for all D1 teams using True Rank Log5.
    Returns DataFrame with: team, pred_rpi, pred_rpi_rank, proj_wins, proj_losses
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

    # Name mapping for external rankings
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
    ranked['true_rank'] = ranked[rcols].mean(axis=1).fillna(ranked['rank_64a']).fillna(LOGISTIC_B)
    true_rank_lookup = dict(zip(ranked['name'], ranked['true_rank']))

    # Load schedule, predict games, compute RPI
    sched_file = DATA_DIR / f'schedules_full_{sport}.csv'
    if not sched_file.exists():
        return pd.DataFrame()

    sched = pd.read_csv(sched_file, low_memory=False)
    played = sched[sched['result'].notna() & (sched['result'] != '')]
    remaining = sched[(sched['result'].isna()) | (sched['result'] == '')]

    team_wins: dict = {}
    team_games: dict = {}
    team_opponents: dict = {}

    for tn, grp in played.groupby('teamName'):
        team_wins[tn] = float(grp['result'].str.startswith('W').sum())
        team_games[tn] = float(len(grp))
        team_opponents[tn] = grp['opponentName'].apply(lambda x: str(x).split('@')[0].strip()).tolist()

    for tn, grp in remaining.groupby('teamName'):
        tr_val = true_rank_lookup.get(tn, LOGISTIC_B)
        for _, g in grp.iterrows():
            opp = str(g.get('opponentName', '')).split('@')[0].strip()
            team_wins[tn] = team_wins.get(tn, 0) + _log5(tr_val, true_rank_lookup.get(opp, LOGISTIC_B))
            team_games[tn] = team_games.get(tn, 0) + 1
            team_opponents.setdefault(tn, []).append(opp)

    # Location-weighted WP
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

    # OWP
    owp_lookup: dict = {}
    for t in wp_lookup:
        opps = team_opponents.get(t, [])
        owps = [wp_lookup.get(o, 0.5) for o in opps if o in wp_lookup]
        owp_lookup[t] = float(np.mean(owps)) if owps else 0.5

    # RPI
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

    # Filter to D1 (excl Big Sky catch-all)
    d1_ids = set(confs[(confs['division'] == 'D-I') & (confs['name'] != 'Big Sky Conference')]['id'])
    d1_names = set(teams[(teams['sport'] == sport_label) & (teams['conference_id'].isin(d1_ids))]['name'])
    df = df[df['team'].isin(d1_names)].reset_index(drop=True)
    df['pred_rpi_rank'] = range(1, len(df) + 1)

    # Add True Rank and Final Rank (70% True Rank + 30% Predicted RPI Rank)
    df['true_rank'] = df['team'].map(true_rank_lookup)
    # Re-rank True Rank within D1 teams
    df['true_rank_d1'] = df['true_rank'].rank(method='min').fillna(len(df)).astype(int)
    df['final_rank_score'] = 0.70 * df['true_rank_d1'] + 0.30 * df['pred_rpi_rank']
    # Break ties by pred_rpi (higher = better) so no two teams share the same final rank.
    df = df.sort_values(['final_rank_score', 'pred_rpi'], ascending=[True, False]).reset_index(drop=True)
    df['final_rank'] = range(1, len(df) + 1)

    return df
