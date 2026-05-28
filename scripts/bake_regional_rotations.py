"""Bake frozen rotation + lineup MEMBERSHIP for the Regional Preview.

Run on the machine that has the fresh scrape_final PBP:
    python scripts/bake_regional_rotations.py

Writes data/regional_rotations_baseball_2026.json — a per-team frozen list of WHO
is in the staff/lineup. Stats are NOT baked; the Regional Preview page pulls those
live from data/pitching.csv and data/hitting.csv (so they stay current).

  rotation: aces SP1/2/3 = top-3 weekend starters over the last 6 weeks, ordered by
            player_rank pitching pctile; bullpen RP = top-8-IP last 4 weeks minus aces.
  lineup:   top-9 hitters by PA over the last 4 weeks.
Each member stored as {id (64A player_id), name[, role]} so the page joins by id.

Re-run this when the rotations/lineups change (e.g., once regionals are set), then
commit the JSON. Baseball D1 only.
"""
import pandas as pd, numpy as np, json, re, unicodedata
from pathlib import Path
from difflib import SequenceMatcher

PBP = r'C:/Dev/scrape_final/output/2026/baseball/pbp'
D = str(Path(__file__).resolve().parent.parent / 'data')


def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'[^a-z ]', ' ', s); s = re.sub(r'\b(jr|sr|ii|iii|iv)\b', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def outs(v):
    try:
        v = float(v); w = int(v); return w * 3 + round((v - w) * 10)
    except (TypeError, ValueError):
        return 0


def parse(p):
    df = pd.read_csv(p, low_memory=False)
    d = pd.to_datetime(df['date'], format='%m/%d/%y', errors='coerce').fillna(
        pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce'))
    df['d'] = d
    return df, d.max()


teams = pd.read_csv(f'{D}/teams.csv', low_memory=False)
tb = teams[teams['sport'] == 'Baseball']
tname2id = {norm(n): int(i) for n, i in zip(tb['name'], tb['id'])}
pl = pd.read_csv(f'{D}/players.csv', low_memory=False, encoding='latin-1')
pl['id'] = pd.to_numeric(pl['id'], errors='coerce'); pl['team_id'] = pd.to_numeric(pl['team_id'], errors='coerce')
team_names = {}
for tid, g in pl.groupby('team_id'):
    if pd.isna(tid):
        continue
    team_names[int(tid)] = {norm(n): (int(i), n) for n, i in zip(g['player_name'], g['id']) if pd.notna(i)}


def lookup(tid, pbp_name):
    dd = team_names.get(tid, {}); n = norm(pbp_name)
    if n in dd:
        return dd[n]
    last = n.split()[-1] if n else ''
    cand = [v for k, v in dd.items() if k.split()[-1:] == [last]]
    if len(cand) == 1:
        return cand[0]
    best = None; br = 0.0
    for k, v in dd.items():
        r = SequenceMatcher(None, n, k).ratio()
        if r > br:
            br = r; best = v
    return best if br >= 0.85 else (None, pbp_name)


# pitchers
pf, pmx = parse(f'{PBP}/pitching_pbp_D1.csv'); pf['o'] = pf['ip'].apply(outs)
w6 = pf[(pf['d'] >= pmx - pd.Timedelta(days=42)) & (pf['d'].dt.dayofweek.isin([4, 5, 6]))]
st6 = w6.groupby(['gameId', 'teamName'], sort=False).head(1)
top_st = {tm: g['playerName'].value_counts().index.tolist() for tm, g in st6.groupby('teamName')}
pw4 = pf[pf['d'] >= pmx - pd.Timedelta(days=28)]
pip4 = pw4.groupby(['teamName', 'playerName'])['o'].sum().div(3).rename('ip4').reset_index()
ptop8 = {tm: g.sort_values('ip4', ascending=False)['playerName'].tolist()[:8] for tm, g in pip4.groupby('teamName')}
pip4map = {(r['teamName'], r['playerName']): r['ip4'] for _, r in pip4.iterrows()}
pr = pd.read_csv(f'{D}/player_rank.csv', low_memory=False); pr = pr[pr['year'] == 2026]
pr['id'] = pd.to_numeric(pr['player_id'], errors='coerce')
pr['pit'] = pd.to_numeric(pr['percentile_rank_weighted_run_allowed_efficiency'], errors='coerce')
pctile_by_id = dict(zip(pr['id'], pr['pit']))

# hitters
hf, hmx = parse(f'{PBP}/hitting_pbp_D1.csv')
for c in ['ab', 'bb', 'hbp', 'sf', 'sh']:
    hf[c] = pd.to_numeric(hf[c], errors='coerce').fillna(0)
hf['pa'] = hf['ab'] + hf['bb'] + hf['hbp'] + hf['sf'] + hf['sh']
hw4 = hf[hf['d'] >= hmx - pd.Timedelta(days=28)]
hpa = hw4.groupby(['teamName', 'playerName'])['pa'].sum().rename('pa4').reset_index()
htop9 = {tm: g.sort_values('pa4', ascending=False)['playerName'].tolist()[:9] for tm, g in hpa.groupby('teamName')}

rot = {}
for tm in set(top_st) | set(ptop8) | set(htop9):
    tid = tname2id.get(norm(tm))
    if tid is None:
        continue
    starters = top_st.get(tm, [])[:3]

    def pct(nm, _tid=tid):
        i, _ = lookup(_tid, nm); v = pctile_by_id.get(i, 0.5) if i else 0.5
        return v if v == v else 0.5
    aces = sorted(starters, key=lambda nm: -pct(nm))
    staff = list(ptop8.get(tm, []))
    for a in aces:
        if a not in staff:
            staff.append(a)
    bull = sorted([nm for nm in staff if nm not in aces], key=lambda nm: -pip4map.get((tm, nm), 0))
    members = []
    for i, nm in enumerate(aces):
        pid, canon = lookup(tid, nm); members.append({'id': pid, 'name': canon, 'role': f'SP{i + 1}'})
    for nm in bull:
        pid, canon = lookup(tid, nm); members.append({'id': pid, 'name': canon, 'role': 'RP'})
    lineup = []
    for nm in htop9.get(tm, []):
        pid, canon = lookup(tid, nm); lineup.append({'id': pid, 'name': canon})
    rot[tid] = {'rotation': members[:8], 'lineup': lineup[:9]}

out = f'{D}/regional_rotations_baseball_2026.json'
json.dump(rot, open(out, 'w'), indent=0)
print(f'baked {len(rot)} teams -> {out}')
