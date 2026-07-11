"""Auto-sync the live MLB draft into Supabase draft_picks.

Polls MLB StatsAPI (the public API behind mlb.com/draft/tracker) every
POLL_SECONDS and INSERTs any drafted pick whose pick number isn't in the
draft_picks table yet. Never updates existing rows — manual entries and
recorded signing bonuses are untouched.

Usage:
  py scripts/draft_autosync.py            # loop until the draft goes quiet
  py scripts/draft_autosync.py --once     # single catch-up pass, then exit
"""
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
YEAR = 2026
POLL_SECONDS = 30
QUIET_EXIT_MIN = 90          # stop after this long with no new picks (draft over)
MLB_URL = f'https://statsapi.mlb.com/api/v1/draft/{YEAR}'

SUPABASE_URL = 'https://vfzoroabzmbvwkcyozes.supabase.co'
_src = (ROOT / 'pages' / '21_Draft_Assistant.py').read_text(encoding='utf-8')
KEY = ''.join(re.findall(r"'([^']*)'",
                         re.search(r"SUPABASE_ANON_KEY = \((.*?)\)", _src, re.S).group(1)))
H = {'apikey': KEY, 'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json'}

# pick -> numeric round (fill-forward over comp rounds, same as the page)
_slots = pd.read_csv(ROOT / 'data' / 'draft' / 'draft_slots_2026.csv', dtype=str)
round_by_pick, _cur = {}, 1
for _, r in _slots.sort_values(by='pick', key=lambda s: pd.to_numeric(s, errors='coerce')).iterrows():
    if str(r['round']).replace('.', '').isdigit():
        _cur = int(float(r['round']))
    round_by_pick[int(float(r['pick']))] = _cur

# board-name matcher: use the master's spelling so tweets/enrichment link up
_master = pd.read_csv(ROOT / 'data' / 'draft' / 'draft_master.csv', dtype=str, keep_default_na=False)
def _norm(s):
    s = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b\.?', '', str(s).lower())
    return re.sub(r'[^a-z]', '', s)
MASTER_BY_KEY = {}
for n in _master['name']:
    MASTER_BY_KEY.setdefault(_norm(n), n)


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def fetch_mlb_drafted():
    r = requests.get(MLB_URL, timeout=20)
    r.raise_for_status()
    picks = [p for rd in r.json()['drafts']['rounds'] for p in rd.get('picks', [])]
    return [p for p in picks if p.get('isDrafted') and p.get('person', {}).get('fullName')]


def fetch_recorded_picknums():
    r = requests.get(f'{SUPABASE_URL}/rest/v1/draft_picks', headers=H,
                     params={'select': 'pick', 'year': f'eq.{YEAR}'}, timeout=20)
    r.raise_for_status()
    return {int(row['pick']) for row in r.json()}


def sync_once():
    drafted = fetch_mlb_drafted()
    have = fetch_recorded_picknums()
    new = [p for p in drafted if int(p['pickNumber']) not in have]
    for p in sorted(new, key=lambda x: int(x['pickNumber'])):
        num = int(p['pickNumber'])
        mlb_name = p['person']['fullName']
        name = MASTER_BY_KEY.get(_norm(mlb_name), mlb_name)
        rnd = round_by_pick.get(num) or (int(p['pickRound']) if str(p['pickRound']).isdigit() else 20)
        slot = int(p['pickValue']) if str(p.get('pickValue', '')).isdigit() else None
        payload = {'year': YEAR, 'round': rnd, 'pick': num,
                   'team': p.get('team', {}).get('name', ''),
                   'player_name': name, 'slot_value': slot,
                   'signing_bonus': None, 'entered_by': 'mlb-autosync'}
        resp = requests.post(f'{SUPABASE_URL}/rest/v1/draft_picks',
                             headers={**H, 'Prefer': 'return=minimal'}, json=payload, timeout=20)
        if resp.status_code in (200, 201):
            tag = '' if name == mlb_name else f'  (board name for "{mlb_name}")'
            log(f"+ pick {num}: {name} -> {payload['team']}{tag}")
        else:
            log(f"! pick {num} FAILED {resp.status_code}: {resp.text[:120]}")
    return len(new), len(drafted)


def main():
    once = '--once' in sys.argv
    last_new = time.time()
    log(f"draft autosync start (poll {POLL_SECONDS}s, quiet-exit {QUIET_EXIT_MIN}m)")
    while True:
        try:
            n_new, n_total = sync_once()
            if n_new:
                last_new = time.time()
                log(f"synced {n_new} new (MLB total drafted: {n_total})")
        except Exception as e:
            log(f"cycle error (will retry): {e}")
        if once:
            log("single pass done."); break
        if time.time() - last_new > QUIET_EXIT_MIN * 60:
            log("no new picks for a while — draft looks done. exiting."); break
        time.sleep(POLL_SECONDS)


if __name__ == '__main__':
    main()
