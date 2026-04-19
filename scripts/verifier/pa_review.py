"""PA (Player Alignment) review — box-score vs season-stat cross-check for D1 baseball.

Uses **plate appearances** as the primary alignment metric. PA is measured every
game for a regular hitter, so missing box-score games show up as clean PA gaps.
HR gaps are too sparse (a player homers every 4-10 games) to reliably reveal
missing data — a team could drop 3 games from the box scores without losing a
single HR from the box-score total.

PA definition (consistent across both sources):
  PA = AB + BB + HBP + SF + SH
hitting.csv has a native `plate_appearances` column we use for the season side.
"""
import sys, json, re, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\Users\sixty\OneDrive\Desktop\chart-builder-app')
from pathlib import Path

from scripts.verifier.name_resolver import keys_for, build_name_index, match as nm_match

DATA = Path(r'C:\Users\sixty\OneDrive\Desktop\chart-builder-app\data')
PBP  = Path(r'C:\Users\sixty\OneDrive\Desktop\chart-builder-app\pbp_data')

rpi = pd.read_csv(DATA / 'baseball_rpi_D1.csv', low_memory=False)
d1_rpi_teams = set(rpi['teamName'].dropna())

h_pbp = pd.read_csv(PBP / 'baseball' / 'hitting_pbp_D1.csv', low_memory=False)
# Build box-side PA = AB + BB + HBP + SF + SH
for _c in ['ab','bb','hbp','sf','sh','hr']:
    h_pbp[_c] = pd.to_numeric(h_pbp[_c], errors='coerce').fillna(0) if _c in h_pbp.columns else 0
h_pbp['pa'] = h_pbp['ab'] + h_pbp['bb'] + h_pbp['hbp'] + h_pbp['sf'] + h_pbp['sh']
h_pbp['date'] = pd.to_datetime(h_pbp['date'], errors='coerce', format='mixed')

hit = pd.read_csv(DATA / 'hitting.csv', low_memory=False)
hit = hit[hit['year']==2026].copy()
hit['plate_appearances'] = pd.to_numeric(hit['plate_appearances'], errors='coerce').fillna(0).astype(int)
hit['home_runs'] = pd.to_numeric(hit['home_runs'], errors='coerce').fillna(0).astype(int)
hit['player_id'] = pd.to_numeric(hit['player_id'], errors='coerce').astype('Int64')

pl = pd.read_csv(DATA / 'players.csv', low_memory=False, encoding='latin-1', dtype=str).fillna('')
pl['id_i'] = pd.to_numeric(pl['id'], errors='coerce').astype('Int64')
pl['team_id_i'] = pd.to_numeric(pl['team_id'], errors='coerce').astype('Int64')

teams = pd.read_csv(DATA / 'teams.csv', low_memory=False)
teams_bb = teams[teams['sport']=='Baseball'][['id','name']].copy()
teams_bb['id_n'] = pd.to_numeric(teams_bb['id'], errors='coerce').astype('Int64')

hit_full = hit.merge(pl[['id_i','player_name','team_id_i']], left_on='player_id', right_on='id_i', how='left')
hit_full = hit_full.merge(teams_bb[['id_n','name']].rename(columns={'name':'team_64a'}), left_on='team_id_i', right_on='id_n', how='left')

# LONGEST-prefix match for PBP -> 64A direction (the fix)
def resolve_pbp_to_64a(pbp_name):
    cands = sorted([t for t in d1_rpi_teams if pbp_name.startswith(t)], key=len, reverse=True)
    return cands[0] if cands else None

d1_box_mask = h_pbp['teamName'].apply(lambda t: resolve_pbp_to_64a(t) is not None if isinstance(t, str) else False)
d1_box = h_pbp[d1_box_mask].copy()
d1_box['team_64a'] = d1_box['teamName'].apply(resolve_pbp_to_64a)

# Group box-score data by playerId (the ONLY reliable identity within a source).
# Same player can appear under multiple spellings ("Nate" / "Nathan", "Longo" / "Longo II",
# "Pena" / "Pena-Edwards") — grouping by name splits them and produces false Bucket C gaps.
# After grouping, pick the LONGEST name variant seen for each playerId as canonical.
_box_by_id = d1_box.groupby(['playerId','team_64a']).agg(
    box_pa=('pa','sum'),
    box_hr=('hr','sum'),
    box_games=('gameId','nunique'),
    _names=('playerName', lambda s: sorted(set(x for x in s.dropna().tolist() if isinstance(x, str)), key=len, reverse=True)),
).reset_index()
_box_by_id['box_pa'] = _box_by_id['box_pa'].astype(int)
_box_by_id['box_hr'] = _box_by_id['box_hr'].astype(int)
_box_by_id['box_games'] = _box_by_id['box_games'].astype(int)
_box_by_id['playerName'] = _box_by_id['_names'].apply(lambda xs: xs[0] if xs else '')
_box_by_id['name_variants'] = _box_by_id['_names'].apply(lambda xs: '; '.join(xs) if len(xs) > 1 else '')
box = _box_by_id[['playerName','team_64a','box_pa','box_hr','box_games','playerId','name_variants']].copy()

hit_full_d1 = hit_full[hit_full['team_64a'].isin(d1_rpi_teams)].copy()
season = hit_full_d1.groupby(['player_name','team_64a']).agg(
    season_pa=('plate_appearances','sum'),
    season_hr=('home_runs','sum'),
).reset_index()

# Build per-team name index for season data (carry PA + HR in the index entries)
season_idx_by_team: dict[str, dict] = {}
for team in d1_rpi_teams:
    entries = season[season['team_64a']==team][['player_name','season_pa','season_hr']].to_records(index=False).tolist()
    season_idx_by_team[team] = build_name_index(entries)

print(f'[PA] box players (D1): {len(box):,}')

# Attempt match: name_resolver fuzz within the player's team. Box side carries both
# PA and HR; season side carries the same. Compare primarily on PA (every game;
# missing games are obvious). HR is kept as a secondary sanity signal.
matched_rows = []
unmatched_rows = []
for _, r in box.iterrows():
    name, team, box_pa, box_hr, box_games = r['playerName'], r['team_64a'], int(r['box_pa']), int(r['box_hr']), int(r['box_games'])
    idx = season_idx_by_team.get(team, {})
    hits = nm_match(name, idx)
    if hits:
        # Multiple candidates: pick largest season_pa (favor the real match over namesakes)
        hits_sorted = sorted(hits, key=lambda h: -int(h[1]) if len(h) > 1 else 0)
        best = hits_sorted[0]
        season_pa = int(best[1]) if len(best) > 1 else 0
        season_hr = int(best[2]) if len(best) > 2 else 0
        matched_rows.append({'playerName': name, 'team_64a': team,
                             'box_pa': box_pa, 'season_pa': season_pa,
                             'box_hr': box_hr, 'season_hr': season_hr,
                             'box_games': box_games,
                             'season_name': best[0], 'variant_match': best[0] != name})
    else:
        unmatched_rows.append({'playerName': name, 'team_64a': team,
                               'box_pa': box_pa, 'box_hr': box_hr, 'box_games': box_games})

m = pd.DataFrame(matched_rows)
u = pd.DataFrame(unmatched_rows)

print(f'[PA] matched with resolver: {len(m):,}  ({100*len(m)/len(box):.1f}%)')
print(f'      of which via name variants (not exact): {m["variant_match"].sum() if len(m) else 0}')
print(f'[PA] unmatched: {len(u):,}  ({100*len(u)/len(box):.1f}%)')

# Bucket breakdown — using PA as the primary metric.
# Tolerances calibrated for a typical regular (~4 PA/game, ~40 games = ~160 PA):
#   ±5 PA: within noise (bench variation)
#   ±10 PA: could be 2-3 missed games
#   15+ PA season > box: likely 3+ box-score games missing
m['pa_diff'] = m['season_pa'] - m['box_pa']  # positive = box missing games
m['hr_diff'] = m['season_hr'] - m['box_hr']
TOL = 5   # noise tolerance
C_THRESH = 10  # season > box by this = Bucket C

A = m[m['pa_diff'].abs() <= TOL]
B = m[m['pa_diff'] < -TOL]                 # box > season (season-stat undercount)
C = m[m['pa_diff'] >= C_THRESH]            # season > box (missing box games)
middle = m[(m['pa_diff'] > TOL) & (m['pa_diff'] < C_THRESH)]  # minor gap, noise/bench

print(f'\n===== BUCKET REPORT (PA-based) =====')
print(f'A aligned (|pa_diff| <= {TOL}):              {len(A):,}  ({100*len(A)/len(box):.1f}%)')
print(f'B box > season + {TOL} (season undercount):   {len(B):,}  ({100*len(B)/len(box):.1f}%)')
print(f'C season > box + {C_THRESH} (missing box games): {len(C):,}  ({100*len(C)/len(box):.1f}%)')
print(f'  (minor PA gap, likely noise):            {len(middle):,}  ({100*len(middle)/len(box):.1f}%)')
print(f'D unmatched:                              {len(u):,}  ({100*len(u)/len(box):.1f}%)')

# Bucket C — the rescrape targets
print(f'\n===== BUCKET C (rescrape candidates) — sorted by PA gap =====')
C_sorted = C.sort_values('pa_diff', ascending=False)
for _, r in C_sorted.head(40).iterrows():
    marker = '[v]' if r['variant_match'] else '   '
    print(f"  {marker} {r['playerName']:28s} ({r['team_64a']:25s})  "
          f"box_pa={r['box_pa']:4d}  season_pa={r['season_pa']:4d}  pa_gap={r['pa_diff']:4d}  "
          f"box_games={r['box_games']:3d}  hr_gap={r['hr_diff']:3d}")

# Save Bucket C team list + identify missing gameIds for rescrape
BUCKET_C_TEAMS = set(C['team_64a'].unique())
print(f'\n[RESCRAPE] Unique teams needing rescrape: {len(BUCKET_C_TEAMS)}')
for t in sorted(BUCKET_C_TEAMS): print(f'   - {t}')

# Find specific gameIds: for each Bucket C team, list schedule-played-games that are missing from hitting_pbp
sched = pd.read_csv(DATA / 'schedules_full_baseball.csv', low_memory=False)
sched['played'] = sched['result'].notna() & (sched['result']!='')
sched_played = sched[sched['played']]

missing_games = []
for team in BUCKET_C_TEAMS:
    team_sched = sched_played[sched_played['teamName']==team]
    sched_gids = set(pd.to_numeric(team_sched['gameId'], errors='coerce').dropna().astype(int).tolist()) if 'gameId' in team_sched.columns else set()

    # Resolve PBP team name (team_64a might not exactly match PBP teamName)
    pbp_candidates = [t for t in d1_box['teamName'].unique() if isinstance(t, str) and t.startswith(team)]
    # pick the shortest full PBP name (mirror to 64A short name logic)
    pbp_candidates.sort(key=len)
    pbp_tn = pbp_candidates[0] if pbp_candidates else team
    pbp_gids = set(d1_box[d1_box['teamName']==pbp_tn]['gameId'].unique().tolist())

    missing = sched_gids - pbp_gids
    for gid in sorted(missing):
        row = team_sched[pd.to_numeric(team_sched['gameId'], errors='coerce') == gid].iloc[0]
        missing_games.append({
            'gameId': int(gid),
            'date': str(row.get('date','')),
            'teamName': team,
            'opponent': str(row.get('opponentName',''))[:40]
        })
    print(f'  {team}: sched gids={len(sched_gids)}  pbp gids={len(pbp_gids)}  missing={len(missing)}')

# Write gameIds file for scrape-pbp-puppeteer.ts --game-ids-file
out = Path(r'C:\Users\sixty\AppData\Local\Temp\bucket_c_missing_games.json')
out.write_text(json.dumps(missing_games, indent=2))
print(f'\n[RESCRAPE] {len(missing_games)} missing games written to {out}')
EOF