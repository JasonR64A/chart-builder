"""Player Alignment review — pure-ID cross-check. No names used anywhere.

Every identity flows through numeric IDs:
  box-score row                          hitting_pbp.playerId
       ↓  (rosters.player_ncaa_season_id ↔ player_id)
  rosters.csv (year=2026)                rosters.player_id, rosters.team_id
       ↓
  hitting.csv (year=2026)                hitting.player_id, hitting.team_id
       ↓
  teams.csv + conferences.csv            team_id → D1 whitelist

Primary metric: plate appearances.
"""
import sys, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

DATA = Path(r'C:\Dev\chart-builder-app\data')
PBP  = Path(r'C:\Dev\chart-builder-app\pbp_data')

# ── Build D1 team_id whitelist from teams + conferences (NO team names) ──────
teams = pd.read_csv(DATA / 'teams.csv', low_memory=False)
confs = pd.read_csv(DATA / 'conferences.csv', low_memory=False)
teams['id']            = pd.to_numeric(teams['id'], errors='coerce').astype('Int64')
teams['conference_id'] = pd.to_numeric(teams['conference_id'], errors='coerce').astype('Int64')
confs['id']            = pd.to_numeric(confs['id'], errors='coerce').astype('Int64')
d1_confs = set(confs[confs['division']=='D-I']['id'].dropna().astype(int).tolist())
d1_team_ids = set(teams[(teams['sport']=='Baseball') & (teams['conference_id'].isin(d1_confs))]['id']
                  .dropna().astype(int).tolist())
print(f'[D1] team_ids: {len(d1_team_ids):,}')

# ── Rosters 2026: the ID bridge (ncaa_id ↔ 64a_id ↔ team_id) ─────────────────
rosters = pd.read_csv(DATA / 'rosters.csv', low_memory=False, encoding='latin-1', dtype=str).fillna('')
rosters['year_i']    = pd.to_numeric(rosters['Year'], errors='coerce')
rosters['pid_64a']   = pd.to_numeric(rosters['player_id'], errors='coerce').astype('Int64')
rosters['pid_ncaa']  = pd.to_numeric(rosters['player_ncaa_season_id'], errors='coerce').astype('Int64')
rosters['team_id']   = pd.to_numeric(rosters['team_id'], errors='coerce').astype('Int64')
bridge = rosters[(rosters['year_i']==2026) & rosters['pid_ncaa'].notna()][['pid_ncaa','pid_64a','team_id']].drop_duplicates(subset='pid_ncaa')
print(f'[bridge] rosters-2026 with NCAA id: {len(bridge):,}')

# ── Box-score data (NO team-name filtering) ──────────────────────────────────
h_pbp = pd.read_csv(PBP / 'baseball' / 'hitting_pbp_D1.csv', low_memory=False)
for _c in ['ab','bb','hbp','sf','sh','hr']:
    h_pbp[_c] = pd.to_numeric(h_pbp[_c], errors='coerce').fillna(0) if _c in h_pbp.columns else 0
h_pbp['pa']       = h_pbp['ab'] + h_pbp['bb'] + h_pbp['hbp'] + h_pbp['sf'] + h_pbp['sh']
h_pbp['playerId'] = pd.to_numeric(h_pbp['playerId'], errors='coerce').astype('Int64')
h_pbp = h_pbp[h_pbp['playerId'].notna()]
print(f'[box] total hitting_pbp_D1 rows (all teams): {len(h_pbp):,}')

# Attach team_id from bridge (player's actual team per rosters 2026)
pbp = h_pbp.merge(bridge, left_on='playerId', right_on='pid_ncaa', how='left')
print(f'[box] rows after bridge: matched={pbp["pid_64a"].notna().sum():,}  unmatched={pbp["pid_64a"].isna().sum():,}')

# Keep only D1-team players (filter purely by team_id)
pbp_d1 = pbp[pbp['team_id'].isin(d1_team_ids)]
print(f'[box] rows after D1 team_id filter: {len(pbp_d1):,}')

# Group per (player 64a id, team_id)
box = pbp_d1.groupby(['pid_64a','team_id']).agg(
    box_pa=('pa','sum'),
    box_hr=('hr','sum'),
    box_games=('gameId','nunique'),
).reset_index()
box['box_pa']    = box['box_pa'].astype(int)
box['box_hr']    = box['box_hr'].astype(int)
box['box_games'] = box['box_games'].astype(int)
print(f'[box] unique D1 players in box: {len(box):,}')

# ── Season stats by ID ───────────────────────────────────────────────────────
hit = pd.read_csv(DATA / 'hitting.csv', low_memory=False)
hit = hit[hit['year']==2026].copy()
hit['plate_appearances'] = pd.to_numeric(hit['plate_appearances'], errors='coerce').fillna(0).astype(int)
hit['home_runs']         = pd.to_numeric(hit['home_runs'], errors='coerce').fillna(0).astype(int)
hit['player_id']         = pd.to_numeric(hit['player_id'], errors='coerce').astype('Int64')
hit['team_id']           = pd.to_numeric(hit['team_id'], errors='coerce').astype('Int64')
# Only D1 players
hit_d1 = hit[hit['team_id'].isin(d1_team_ids)][['player_id','team_id','plate_appearances','home_runs']].rename(
    columns={'player_id':'pid_64a','plate_appearances':'season_pa','home_runs':'season_hr'})
hit_d1 = hit_d1.groupby(['pid_64a','team_id'], as_index=False).agg(season_pa=('season_pa','sum'), season_hr=('season_hr','sum'))
print(f'[season] D1 hitting.csv rows: {len(hit_d1):,}')

# ── Join on (pid_64a, team_id) ───────────────────────────────────────────────
m = box.merge(hit_d1, on=['pid_64a','team_id'], how='outer', indicator=True)
for c in ['box_pa','box_hr','box_games','season_pa','season_hr']:
    m[c] = m[c].fillna(0).astype(int)

matched   = m[m['_merge']=='both'].copy()
box_only  = m[m['_merge']=='left_only']
season_only = m[m['_merge']=='right_only']
print(f'\n[join] (pid_64a, team_id):  matched={len(matched):,}  box_only={len(box_only):,}  season_only={len(season_only):,}')

matched['pa_diff'] = matched['season_pa'] - matched['box_pa']
matched['hr_diff'] = matched['season_hr'] - matched['box_hr']

TOL = 5
C_TH = 10
A   = matched[matched['pa_diff'].abs() <= TOL]
B   = matched[matched['pa_diff'] < -TOL]
C   = matched[matched['pa_diff'] >= C_TH]
mid = matched[(matched['pa_diff'] > TOL) & (matched['pa_diff'] < C_TH)]

# Attach player + team names only for display (not used in matching)
pl = pd.read_csv(DATA / 'players.csv', low_memory=False, encoding='latin-1', dtype=str).fillna('')
pl['id_i'] = pd.to_numeric(pl['id'], errors='coerce').astype('Int64')
team_name_by_id = dict(zip(teams['id'].dropna().astype(int), teams[teams['sport']=='Baseball'].set_index('id')['name']))
player_name_by_id = dict(zip(pl['id_i'].dropna().astype(int), pl['player_name']))

def display(df):
    df = df.copy()
    df['team']   = df['team_id'].astype(int).map(team_name_by_id).fillna('')
    df['player'] = df['pid_64a'].astype(int).map(player_name_by_id).fillna('')
    return df

total = len(box) + len(season_only)  # full player universe across both sources
print(f'\n===== BUCKET REPORT (pure ID, PA-based) =====')
print(f'A aligned |pa_diff| <= {TOL}:              {len(A):>5,}')
print(f'B box > season + {TOL} (season under):     {len(B):>5,}')
print(f'C season > box + {C_TH} (missing box):     {len(C):>5,}')
print(f'  minor gap ({TOL}-{C_TH} PA):                {len(mid):>5,}')
print(f'Box-only (player in box, no hitting.csv row): {len(box_only):>5,}')
print(f'Season-only (hitting.csv but never in box):   {len(season_only):>5,}')

print(f'\n===== BUCKET C by team =====')
Cn = display(C)
team_summary = Cn.groupby(['team_id','team']).agg(
    players_affected=('pid_64a','count'),
    total_pa_gap=('pa_diff','sum'),
    avg_pa_gap=('pa_diff','mean'),
).reset_index().sort_values('total_pa_gap', ascending=False).head(20)
for _, r in team_summary.iterrows():
    print(f"  {r['team']:28s}  players={r['players_affected']:3d}  total_gap={int(r['total_pa_gap']):5d}  avg={r['avg_pa_gap']:5.1f}")

print(f'\n===== Top 30 Bucket C players =====')
for _, r in Cn.sort_values('pa_diff', ascending=False).head(30).iterrows():
    print(f"  {r['player']:28s} ({r['team']:25s})  box_pa={r['box_pa']:4d}  season_pa={r['season_pa']:4d}  pa_gap={r['pa_diff']:4d}  box_games={r['box_games']:3d}  hr_gap={r['hr_diff']:3d}")
