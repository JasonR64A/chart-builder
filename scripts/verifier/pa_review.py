"""PA review v2 — uses name_resolver for Bucket D recovery + builds rescrape gameIds."""
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
h_pbp['hr'] = pd.to_numeric(h_pbp['hr'], errors='coerce').fillna(0)
h_pbp['date'] = pd.to_datetime(h_pbp['date'], errors='coerce', format='mixed')

hit = pd.read_csv(DATA / 'hitting.csv', low_memory=False)
hit = hit[hit['year']==2026].copy()
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

box = d1_box.groupby(['playerName','team_64a'])['hr'].sum().reset_index(name='box_hr').astype({'box_hr':int})

hit_full_d1 = hit_full[hit_full['team_64a'].isin(d1_rpi_teams)].copy()
season = hit_full_d1.groupby(['player_name','team_64a'])['home_runs'].sum().reset_index()
season.columns = ['player_name','team_64a','season_hr']

# Build per-team name index for season data
season_idx_by_team: dict[str, dict] = {}
for team in d1_rpi_teams:
    entries = season[season['team_64a']==team][['player_name','season_hr']].to_records(index=False).tolist()
    season_idx_by_team[team] = build_name_index(entries)

print(f'[PA v2] box players (D1): {len(box):,}')

# Attempt match: first-pass exact name + team; fallback with name_resolver
matched_rows = []
unmatched_rows = []
for _, r in box.iterrows():
    name, team, box_hr = r['playerName'], r['team_64a'], int(r['box_hr'])
    idx = season_idx_by_team.get(team, {})
    hits = nm_match(name, idx)
    if hits:
        # If multiple, pick largest season_hr (favor the real match over incidental namesakes)
        hits_sorted = sorted(hits, key=lambda h: -int(h[1]) if len(h) > 1 else 0)
        best = hits_sorted[0]
        season_hr = int(best[1]) if len(best) > 1 else 0
        matched_rows.append({'playerName': name, 'team_64a': team, 'box_hr': box_hr,
                             'season_hr': season_hr, 'season_name': best[0],
                             'variant_match': best[0] != name})
    else:
        unmatched_rows.append({'playerName': name, 'team_64a': team, 'box_hr': box_hr})

m = pd.DataFrame(matched_rows)
u = pd.DataFrame(unmatched_rows)

print(f'[PA v2] matched with resolver: {len(m):,}  ({100*len(m)/len(box):.1f}%)')
print(f'         of which via name variants (not exact): {m["variant_match"].sum() if len(m) else 0}')
print(f'[PA v2] unmatched: {len(u):,}  ({100*len(u)/len(box):.1f}%)')

# Bucket breakdown
m['diff'] = m['season_hr'] - m['box_hr']
A = m[m['season_hr'] >= m['box_hr'] - 1]
B = m[m['box_hr'] > m['season_hr'] + 1]
C = m[m['season_hr'] >= m['box_hr'] + 3]
print(f'\n===== BUCKET REPORT v2 =====')
print(f'A aligned:                                    {len(A):,}  ({100*len(A)/len(box):.1f}%)')
print(f'B box > season +1 (season undercount):        {len(B):,}  ({100*len(B)/len(box):.1f}%)')
print(f'C season > box +2 (possible missing box):     {len(C):,}  ({100*len(C)/len(box):.1f}%)')
print(f'D unmatched:                                  {len(u):,}  ({100*len(u)/len(box):.1f}%)')

# Bucket C — the rescrape targets
print(f'\n===== BUCKET C (rescrape candidates) =====')
C_sorted = C.sort_values('diff', ascending=False)
for _, r in C_sorted.iterrows():
    marker = '[v]' if r['variant_match'] else '   '
    print(f"  {marker} {r['playerName']:28s} ({r['team_64a']:25s})  box={r['box_hr']:3d}  season={r['season_hr']:3d}  missing={r['diff']:3d}")

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