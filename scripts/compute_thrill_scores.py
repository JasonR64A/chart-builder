"""
Pre-compute Thrill Scores for all games across all sport/division combos.
Output: data/thrill_scores.csv

Run manually:   python scripts/compute_thrill_scores.py
Nightly: called from update_data.sh after PBP data is refreshed.

Processes all 6 combos: Baseball D1/D2/D3 + Softball D1/D2/D3.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from pages._win_prob_model import (
    build_team_profiles, adjusted_team_pct, pre_game_wp,
    TEAM_RANK_BLEND,
)

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / 'data'
PBP_DIR = APP_DIR / 'pbp_data'
LOOKUP_FILE = PBP_DIR / 'wp_state_lookup_bb_d1.pkl'
OUTPUT = DATA_DIR / 'thrill_scores.csv'

DIVISION_LABELS = {'D1': 'D-I', 'D2': 'D-II', 'D3': 'D-III'}


def load_lookup():
    if LOOKUP_FILE.exists():
        with open(LOOKUP_FILE, 'rb') as f:
            return pickle.load(f)
    return {}


def build_rank_pct(sport_label, div_code):
    teams = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False)
    tr = pd.read_csv(DATA_DIR / 'team_rank.csv', low_memory=False)
    confs = pd.read_csv(DATA_DIR / 'conferences.csv', low_memory=False)
    tr26 = tr[tr['year'] == 2026].copy()
    tr26['rank'] = pd.to_numeric(tr26['integer_64_rank_total'], errors='coerce')
    div_label = DIVISION_LABELS.get(div_code, div_code)
    div_ids = set(confs[confs['division'] == div_label]['id'])
    bucket = teams[(teams['sport'] == sport_label) & (teams['conference_id'].isin(div_ids))]
    tr_b = tr26[tr26['team_id'].isin(bucket['id'])].dropna(subset=['rank']).copy()
    N = len(tr_b)
    if N < 2:
        return {}, {}
    tr_b['rank_pct'] = 1 - (tr_b['rank'] - 1) / (N - 1)
    id_to_name = dict(zip(bucket['id'], bucket['name']))
    name_pct = {id_to_name[tid]: pct for tid, pct in zip(tr_b['team_id'], tr_b['rank_pct']) if tid in id_to_name}
    name_rank = {id_to_name[tid]: int(r) for tid, r in zip(tr_b['team_id'], tr_b['rank']) if tid in id_to_name}
    return name_pct, name_rank


def state_key(row):
    inning = min(int(row['inning']), 10)
    half = 1 if row.get('halfInning') == 'bottom' else 0
    outs = min(int(row['outs']), 2)
    bases = (int(pd.notna(row.get('runner1B'))) +
             int(pd.notna(row.get('runner2B'))) * 2 +
             int(pd.notna(row.get('runner3B'))) * 4)
    sd = max(-10, min(10, int(row['homeScore']) - int(row['awayScore'])))
    return (inning, half, outs, bases, sd)


def compute_for_division(sport_label, div_code, lookup, name_pct, profiles):
    sport_lower = sport_label.lower()
    # Try .csv first, then .csv.gz
    pbp_csv = PBP_DIR / 'play_by_play' / f'{sport_lower}_play_by_play_{div_code}.csv'
    pbp_gz = pbp_csv.with_suffix('.csv.gz')
    if pbp_csv.exists():
        pbp = pd.read_csv(pbp_csv, low_memory=False,
                          usecols=['gameId', 'date', 'awayTeam', 'homeTeam', 'inning',
                                   'halfInning', 'battingTeam', 'outs',
                                   'runner1B', 'runner2B', 'runner3B',
                                   'awayScore', 'homeScore'])
    elif pbp_gz.exists():
        pbp = pd.read_csv(pbp_gz, low_memory=False, compression='gzip',
                          usecols=['gameId', 'date', 'awayTeam', 'homeTeam', 'inning',
                                   'halfInning', 'battingTeam', 'outs',
                                   'runner1B', 'runner2B', 'runner3B',
                                   'awayScore', 'homeScore'])
    else:
        print(f'  No PBP file for {sport_label} {div_code}')
        return []

    pbp['date'] = pd.to_datetime(pbp['date'], errors='coerce', format='mixed')
    for c in ['awayScore', 'homeScore', 'inning', 'outs']:
        pbp[c] = pd.to_numeric(pbp[c], errors='coerce').fillna(0).astype(int)
    pbp['_ho'] = np.where(pbp['battingTeam'] == pbp['awayTeam'], 0, 1)

    # Build PBP full name -> short name map
    pbp_names = set(pbp['homeTeam'].dropna()) | set(pbp['awayTeam'].dropna())
    p2s = {}
    for full in pbp_names:
        if not isinstance(full, str):
            continue
        for short in name_pct:
            if full.startswith(short):
                p2s[full] = short
                break

    results = []
    game_ids = pbp['gameId'].unique()
    total = len(game_ids)

    for idx, gid in enumerate(game_ids):
        if idx % 500 == 0 and idx > 0:
            print(f'    {idx}/{total} games processed...')

        g = pbp[pbp['gameId'] == gid]
        home_full = g['homeTeam'].iloc[0]
        away_full = g['awayTeam'].iloc[0]
        if not isinstance(home_full, str) or not isinstance(away_full, str):
            continue
        hs = p2s.get(home_full)
        aS = p2s.get(away_full)
        if not hs or not aS or hs not in name_pct or aS not in name_pct:
            continue

        # Pre-game WP
        h_static = name_pct[hs]
        a_static = name_pct[aS]
        h_player = adjusted_team_pct(hs, profiles, 'Weekend', 1, h_static)
        a_player = adjusted_team_pct(aS, profiles, 'Weekend', 1, a_static)
        hp = TEAM_RANK_BLEND * h_static + (1 - TEAM_RANK_BLEND) * h_player
        ap = TEAM_RANK_BLEND * a_static + (1 - TEAM_RANK_BLEND) * a_player
        pg = pre_game_wp(hp, ap)

        # Compute WP curve
        g = g.sort_values(['inning', '_ho'], kind='stable').reset_index(drop=True)
        n_plays = len(g)
        wpc = [pg]
        for i in range(n_plays):
            row = g.iloc[i]
            key = state_key(row)
            info = lookup.get(key)
            w = min(1.0, (i + 1) / n_plays)
            if info and info[1] >= 10:
                wp = max(0.01, min(0.99, w * info[0] + (1 - w) * pg))
            else:
                wp = pg
            wpc.append(wp)

        # Lock final
        fa = int(g['awayScore'].max())
        fh = int(g['homeScore'].max())
        if fa != fh:
            wpc[-1] = 1.0 if fh > fa else 0.0

        # Thrill components
        wp_arr = np.array(wpc[:-1])
        dev = np.abs(wp_arr - 0.5)
        h_area = float(np.sum(np.maximum(wp_arr - 0.5, 0)))
        a_area = float(np.sum(np.maximum(0.5 - wp_arr, 0)))
        mx = max(h_area, a_area)
        mn = min(h_area, a_area)
        balance = mn / mx if mx > 0 else 0
        closeness = max(0.0, float(1 - np.mean(dev) / 0.5))
        crosses = sum(1 for j in range(1, len(wp_arr))
                      if (wp_arr[j-1] < 0.5) != (wp_arr[j] < 0.5))
        lcs = min(1.0, crosses / 6.0)
        quality = (h_static + a_static) / 2.0
        thrill = (0.30 * balance + 0.30 * closeness + 0.15 * lcs + 0.25 * quality) * 100

        game_date = g['date'].iloc[0]
        results.append({
            'sport': sport_label, 'division': div_code,
            'gameId': gid,
            'date': game_date.strftime('%Y-%m-%d') if pd.notna(game_date) else '',
            'home': hs, 'away': aS,
            'final_home': fh, 'final_away': fa,
            'thrill_score': round(thrill, 1),
            'balance': round(balance, 3),
            'closeness': round(closeness, 3),
            'lead_changes': crosses,
            'team_quality': round(quality, 3),
            'plays': n_plays,
        })

    return results


def main():
    print('Loading state lookup...')
    lookup = load_lookup()
    print(f'  {len(lookup)} state entries loaded')

    all_results = []
    combos = [
        ('Baseball', 'D1'), ('Baseball', 'D2'), ('Baseball', 'D3'),
        ('Softball', 'D1'), ('Softball', 'D2'), ('Softball', 'D3'),
    ]
    for sport_label, div_code in combos:
        print(f'\n{sport_label} {div_code}:')
        name_pct, name_rank = build_rank_pct(sport_label, div_code)
        if not name_pct:
            print(f'  No ranked teams — skipping')
            continue
        print(f'  {len(name_pct)} ranked teams')
        profiles = build_team_profiles.__wrapped__(sport_label, div_code)  # bypass st.cache
        print(f'  {len(profiles)} team profiles built')
        results = compute_for_division(sport_label, div_code, lookup, name_pct, profiles)
        print(f'  {len(results)} games scored')
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    df = df.sort_values(['sport', 'division', 'thrill_score'], ascending=[True, True, False])
    df.to_csv(OUTPUT, index=False)
    print(f'\nWrote {len(df)} thrill scores to {OUTPUT}')

    # Top 5 per combo
    for (s, d), grp in df.groupby(['sport', 'division']):
        top = grp.head(3)
        print(f'\n  {s} {d} top 3:')
        for _, r in top.iterrows():
            print(f"    {r['thrill_score']:5.1f} | {r['date']} {r['away']} @ {r['home']} | {r['final_away']}-{r['final_home']}")


if __name__ == '__main__':
    main()
