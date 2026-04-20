"""Pinpoint dates with missing box-score data for specific D1 teams.

For each target team:
  1. Get played-dates from schedules_full_baseball (via teamYearId)
  2. Get dates present in hitting_pbp_D1 (via playerId -> rosters.team_id)
  3. Diff = dates the team played but box scores lack any row

Pure ID join — no team-name matching.
"""
import sys, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

DATA = Path(r'C:\Users\sixty\OneDrive\Desktop\chart-builder-app\data')
PBP  = Path(r'C:\Users\sixty\OneDrive\Desktop\chart-builder-app\pbp_data')

# Targets: teams flagged in Bucket C we want to inspect
TARGETS = ['Southern U.', 'Georgia Tech', 'Bradley', 'LIU', 'VCU',
           'Delaware St.', 'Wagner', 'Texas Southern', 'Le Moyne', 'La Salle']

teams = pd.read_csv(DATA / 'teams.csv', low_memory=False)
teams['id'] = pd.to_numeric(teams['id'], errors='coerce').astype('Int64')
# team_id lookup from 64A name (baseball)
tid_by_name = dict(zip(teams[teams['sport']=='Baseball']['name'], teams[teams['sport']=='Baseball']['id']))

# Schedule has teamYearId — rosters has team_ncaa_season_id which matches teamYearId
# But also has teamName. The cleanest path: use teamYearId.
# rosters.csv has 'team_ncaa_season_id' — let's use it to get teamYearId per team_id.
rosters = pd.read_csv(DATA / 'rosters.csv', low_memory=False, encoding='latin-1', dtype=str).fillna('')
rosters['year_i']      = pd.to_numeric(rosters['Year'], errors='coerce')
rosters['team_id']     = pd.to_numeric(rosters['team_id'], errors='coerce').astype('Int64')
rosters['team_yid']    = pd.to_numeric(rosters['team_ncaa_season_id'], errors='coerce').astype('Int64')
rosters['pid_ncaa']    = pd.to_numeric(rosters['player_ncaa_season_id'], errors='coerce').astype('Int64')
r2026 = rosters[rosters['year_i']==2026]

# For each team_id, get its teamYearId (first non-null value per team)
tid_to_yid = r2026.dropna(subset=['team_yid']).groupby('team_id')['team_yid'].first().to_dict()

# Bridge: pid_ncaa -> team_id
bridge = r2026[r2026['pid_ncaa'].notna()][['pid_ncaa','team_id']].drop_duplicates(subset='pid_ncaa')

# Schedule, indexed by teamYearId
sched = pd.read_csv(DATA / 'schedules_full_baseball.csv', low_memory=False)
sched['played']        = sched['result'].notna() & (sched['result']!='')
sched['date_iso']      = pd.to_datetime(sched['date'], errors='coerce', format='mixed').dt.strftime('%Y-%m-%d')
sched['teamYearId']    = pd.to_numeric(sched['teamYearId'], errors='coerce').astype('Int64')

# Box-score data, attach team_id via bridge
h = pd.read_csv(PBP / 'baseball' / 'hitting_pbp_D1.csv', low_memory=False, usecols=['playerId','teamName','gameId','date'])
h['playerId'] = pd.to_numeric(h['playerId'], errors='coerce').astype('Int64')
h['date_iso'] = pd.to_datetime(h['date'], errors='coerce', format='mixed').dt.strftime('%Y-%m-%d')
h = h.merge(bridge, left_on='playerId', right_on='pid_ncaa', how='inner')

# ── Per-team diagnostic ──────────────────────────────────────────────────────
for name in TARGETS:
    if name not in tid_by_name:
        print(f'\n### {name} — NOT in teams.csv as Baseball team'); continue
    tid  = int(tid_by_name[name])
    yid  = tid_to_yid.get(tid)
    if yid is None:
        print(f'\n### {name} (team_id={tid}) — NO teamYearId in 2026 rosters'); continue

    # Schedule played dates for this team
    tsched = sched[(sched['teamYearId']==int(yid)) & sched['played']]
    sched_dates = set(tsched['date_iso'].dropna())

    # Box dates for this team (any box row where player's team_id == tid)
    tbox = h[h['team_id']==tid]
    box_dates = set(tbox['date_iso'].dropna())

    missing = sorted(sched_dates - box_dates)
    extra   = sorted(box_dates - sched_dates)

    # Split missing into real-missing (legit game played) vs expected-empty (canceled/ppd)
    real_missing = []
    expected_empty = []
    for d in missing:
        row = tsched[tsched['date_iso']==d].iloc[0]
        res = str(row.get('result','')).lower()
        if any(k in res for k in ['cancel','ppd','postpone']):
            expected_empty.append((d, row))
        else:
            real_missing.append((d, row))

    print(f'\n### {name}  team_id={tid}  teamYearId={int(yid)}')
    print(f'   played dates (sched):  {len(sched_dates)}')
    print(f'   box dates:             {len(box_dates)}')
    print(f'   REAL missing (need rescrape):  {len(real_missing)}')
    print(f'   canceled/ppd (expected empty): {len(expected_empty)}')
    print(f'   extra in box (schedule behind): {len(extra)}')
    if real_missing:
        print(f'   REAL MISSING DATES:')
        for d, row in real_missing:
            opp = str(row.get('opponentName',''))[:45]
            venue = '@' if pd.notna(row.get('isAway')) and row.get('isAway')==1.0 else 'vs'
            print(f"     {d}  {venue} {opp}  result={row.get('result','')}")
