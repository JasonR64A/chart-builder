"""Player Alignment review — box-score vs season-stats cross-check for D1 baseball.

Pure ID-based join. No name matching.

Join path:
  hitting_pbp.playerId  ==  rosters.player_ncaa_season_id   (NCAA id)
  rosters.player_id     ==  hitting.csv.player_id           (64A internal id)

Primary metric: plate appearances (PA). HR kept as a secondary sanity signal.
PA is every-game, so missing box-score games show up as clean PA gaps;
HR is sparse (every 4-10 games) and hides the signal.
"""
import sys, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

DATA = Path(r'C:\Users\sixty\OneDrive\Desktop\chart-builder-app\data')
PBP  = Path(r'C:\Users\sixty\OneDrive\Desktop\chart-builder-app\pbp_data')

# ── D1 whitelist (for team-name filtering of box scores) ─────────────────────
rpi = pd.read_csv(DATA / 'baseball_rpi_D1.csv', low_memory=False)
d1_rpi_teams = set(rpi['teamName'].dropna())

# ── Box-score: per-player PA + HR + games (grouped by playerId only) ─────────
h_pbp = pd.read_csv(PBP / 'baseball' / 'hitting_pbp_D1.csv', low_memory=False)
for _c in ['ab','bb','hbp','sf','sh','hr']:
    h_pbp[_c] = pd.to_numeric(h_pbp[_c], errors='coerce').fillna(0) if _c in h_pbp.columns else 0
h_pbp['pa'] = h_pbp['ab'] + h_pbp['bb'] + h_pbp['hbp'] + h_pbp['sf'] + h_pbp['sh']
h_pbp['playerId'] = pd.to_numeric(h_pbp['playerId'], errors='coerce').astype('Int64')

# Filter to D1 rows via team-name prefix (longest match = most specific)
def resolve_pbp_to_64a(pbp_name):
    if not isinstance(pbp_name, str):
        return None
    cands = sorted([t for t in d1_rpi_teams if pbp_name.startswith(t)], key=len, reverse=True)
    return cands[0] if cands else None

h_pbp['team_64a'] = h_pbp['teamName'].apply(resolve_pbp_to_64a)
d1_box = h_pbp[h_pbp['team_64a'].notna()].copy()

box = d1_box.groupby(['playerId','team_64a']).agg(
    box_pa=('pa','sum'),
    box_hr=('hr','sum'),
    box_games=('gameId','nunique'),
    _names=('playerName', lambda s: sorted(set(x for x in s.dropna().tolist() if isinstance(x, str)), key=len, reverse=True)),
).reset_index()
box['box_pa']   = box['box_pa'].astype(int)
box['box_hr']   = box['box_hr'].astype(int)
box['box_games']= box['box_games'].astype(int)
box['box_name'] = box['_names'].apply(lambda xs: xs[0] if xs else '')
box['name_variants'] = box['_names'].apply(lambda xs: '; '.join(xs) if len(xs) > 1 else '')
box = box.drop(columns=['_names'])
print(f'[box] D1 unique box-score players (by playerId): {len(box):,}')

# ── Bridge: rosters.csv maps box-score NCAA id -> 64A internal player_id ─────
rosters = pd.read_csv(DATA / 'rosters.csv', low_memory=False, encoding='latin-1', dtype=str).fillna('')
rosters['year_i']     = pd.to_numeric(rosters['Year'], errors='coerce').astype('Int64')
rosters['pid_64a']    = pd.to_numeric(rosters['player_id'], errors='coerce').astype('Int64')
rosters['pid_ncaa']   = pd.to_numeric(rosters['player_ncaa_season_id'], errors='coerce').astype('Int64')
rosters['team_64a_id']= pd.to_numeric(rosters['team_id'], errors='coerce').astype('Int64')
r2026 = rosters[rosters['year_i']==2026]
bridge = r2026[['pid_ncaa','pid_64a','team_64a_id']].dropna(subset=['pid_ncaa']).drop_duplicates(subset=['pid_ncaa'])
print(f'[bridge] rosters 2026 rows with NCAA id: {len(bridge):,}')

# ── Season: hitting.csv keyed by 64A player_id ───────────────────────────────
hit = pd.read_csv(DATA / 'hitting.csv', low_memory=False)
hit = hit[hit['year']==2026].copy()
hit['plate_appearances'] = pd.to_numeric(hit['plate_appearances'], errors='coerce').fillna(0).astype(int)
hit['home_runs']         = pd.to_numeric(hit['home_runs'], errors='coerce').fillna(0).astype(int)
hit['player_id']         = pd.to_numeric(hit['player_id'], errors='coerce').astype('Int64')
hit_season = hit[['player_id','plate_appearances','home_runs']].rename(
    columns={'player_id':'pid_64a','plate_appearances':'season_pa','home_runs':'season_hr'})
# Collapse any duplicate rows per player
hit_season = hit_season.groupby('pid_64a', as_index=False).agg(season_pa=('season_pa','sum'), season_hr=('season_hr','sum'))

# Add player name for output
players = pd.read_csv(DATA / 'players.csv', low_memory=False, encoding='latin-1', dtype=str).fillna('')
players['id_i'] = pd.to_numeric(players['id'], errors='coerce').astype('Int64')

# ── Join: box -> bridge -> season ────────────────────────────────────────────
m = box.merge(bridge, left_on='playerId', right_on='pid_ncaa', how='left')
m = m.merge(hit_season, on='pid_64a', how='left')
m = m.merge(players[['id_i','player_name']], left_on='pid_64a', right_on='id_i', how='left')

# Matched = has both pid_64a (bridge) and season_pa (hitting.csv row)
matched   = m[m['pid_64a'].notna() & m['season_pa'].notna()].copy()
no_bridge = m[m['pid_64a'].isna()].copy()            # box playerId not in rosters
no_season = m[m['pid_64a'].notna() & m['season_pa'].isna()].copy()  # bridged but no hitting.csv row

matched['season_pa'] = matched['season_pa'].astype(int)
matched['season_hr'] = matched['season_hr'].astype(int)
matched['pa_diff']   = matched['season_pa'] - matched['box_pa']
matched['hr_diff']   = matched['season_hr'] - matched['box_hr']

print(f'[join] box -> bridge -> season:')
print(f'   matched:            {len(matched):,}  ({100*len(matched)/len(box):.1f}%)')
print(f'   no bridge (roster): {len(no_bridge):,}  ({100*len(no_bridge)/len(box):.1f}%)')
print(f'   bridge ok, no hit.csv row: {len(no_season):,}  ({100*len(no_season)/len(box):.1f}%)')

# ── Buckets (PA-based tolerances) ────────────────────────────────────────────
TOL = 5        # within noise
C_TH = 10      # season > box by this = missing games
A = matched[matched['pa_diff'].abs() <= TOL]
B = matched[matched['pa_diff'] < -TOL]
C = matched[matched['pa_diff'] >= C_TH]
mid = matched[(matched['pa_diff'] > TOL) & (matched['pa_diff'] < C_TH)]

total = len(box)
print(f'\n===== BUCKET REPORT (PA-based, ID-joined) =====')
print(f'A aligned |pa_diff| <= {TOL:>2}:          {len(A):>5,}  ({100*len(A)/total:5.1f}%)')
print(f'B box > season + {TOL} (season under):   {len(B):>5,}  ({100*len(B)/total:5.1f}%)')
print(f'C season > box + {C_TH} (missing box):    {len(C):>5,}  ({100*len(C)/total:5.1f}%)')
print(f'  (minor PA gap, likely noise):       {len(mid):>5,}  ({100*len(mid)/total:5.1f}%)')
print(f'D no bridge (box pid not in rosters): {len(no_bridge):>5,}  ({100*len(no_bridge)/total:5.1f}%)')
print(f'E bridge ok, no hitting.csv row:      {len(no_season):>5,}  ({100*len(no_season)/total:5.1f}%)')

# ── Bucket C detail ──────────────────────────────────────────────────────────
print(f'\n===== BUCKET C (top 40 by PA gap) =====')
C_sorted = C.sort_values('pa_diff', ascending=False).head(40)
for _, r in C_sorted.iterrows():
    nm = r.get('player_name') or r['box_name']
    print(f"  {str(nm):28s} ({r['team_64a']:25s})  "
          f"box_pa={r['box_pa']:4d}  season_pa={int(r['season_pa']):4d}  pa_gap={r['pa_diff']:4d}  "
          f"box_games={r['box_games']:3d}  hr_gap={r['hr_diff']:3d}")

# Team-level summary for rescrape planning
print(f'\n===== BUCKET C by team =====')
team_summary = C.groupby('team_64a').agg(
    players_affected=('pid_64a','count'),
    total_pa_gap=('pa_diff','sum'),
    avg_pa_gap=('pa_diff','mean'),
).reset_index().sort_values('total_pa_gap', ascending=False).head(25)
for _, r in team_summary.iterrows():
    print(f"  {r['team_64a']:25s}  players={r['players_affected']:3d}  total_gap={int(r['total_pa_gap']):5d}  avg={r['avg_pa_gap']:5.1f}")
