"""
Build per-team Q1+Q2 records for every D1 baseball tournament entrant 2013-2025
(no 2020). Reads:
  - tournament_master_2013_2025.csv (entrants + is_host)
  - weekly_rpi_baseball_YYYY_D1.csv (is_selection=True snapshot for RPI rank + yearId)
  - team_schedules_D1.csv (per-game venue + result + opponent name)
  - selection_dates.csv (cutoff date per year)

Q1/Q2 thresholds (matches snubs/reaches chart footnote):
  Q1: Home <= 30, Neutral <= 50, Away <= 75
  Q2: Home <= 75, Neutral <= 100, Away <= 135

Output: data/bracketology/historical_q1q2_baseball.csv
"""
import csv
import re
from pathlib import Path
from collections import defaultdict

REPO = Path(r'C:\Dev\chart-builder-app')
SCRAPE = Path(r'C:\Dev\scrape_final\output\historical')
BRACK = REPO / 'data' / 'bracketology'
WEEKLY = BRACK / 'weekly_rpi'

YEARS = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

Q1 = {'home': 30, 'neutral': 50, 'away': 75}
Q2 = {'home': 75, 'neutral': 100, 'away': 135}


def norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


# Schedule-name -> RPI-name aliases for cases the normalizer can't bridge.
# Keep in sync with build_current_q1q2.py.
NAME_ALIASES = {
    'uncgreensboro': 'uncg',
}


def normalize_opp(s):
    n = norm(s)
    return NAME_ALIASES.get(n, n)


def load_selection_dates():
    out = {}
    with open(BRACK / 'selection_dates.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['sport'] == 'baseball' and r['selection_date']:
                out[int(r['year'])] = r['selection_date']
    return out


def load_tournament_master():
    """year -> list of entrant dicts."""
    by_year = defaultdict(list)
    with open(BRACK / 'tournament_master_2013_2025.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            yr = int(r['year'])
            by_year[yr].append({
                'team_rpi_name': r['team_rpi_name'],
                'team_canonical': r['team_canonical'],
                'rpi_rank': int(float(r['rpi_rank'])) if r['rpi_rank'] else None,
                'w': int(r['w']) if r['w'] else 0,
                'l': int(r['l']) if r['l'] else 0,
                'in_tournament': r['in_tournament'] == 'True',
                'is_host': r['is_host'] == 'True',
                'host_seed': r['host_seed'],
            })
    return by_year


def load_selection_rpi(year):
    """For a year, return (norm_name -> yearId, norm_name -> rank, yearId -> conference)."""
    path = WEEKLY / f'weekly_rpi_baseball_{year}_D1.csv'
    name2yid = {}
    name2rank = {}
    yid2name = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['is_selection'] != 'True':
                continue
            if not r['rank'] or not r['yearId']:
                continue
            n = norm(r['name'])
            name2yid[n] = int(float(r['yearId']))
            name2rank[n] = int(float(r['rank']))
            yid2name[int(float(r['yearId']))] = r['name']
    return name2yid, name2rank, yid2name


def load_schedule_for_year(year):
    """yearId -> list of games (date, loc, opp_norm, result)."""
    path = SCRAPE / str(year) / 'baseball' / 'team_schedules_D1.csv'
    by_team = defaultdict(list)
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            try:
                yid = int(r['yearId'])
            except (ValueError, KeyError):
                continue
            by_team[yid].append({
                'date': r['date'],  # MM/DD/YYYY
                'loc': r['loc'],
                'opp_norm': normalize_opp(r['opponent_clean']),
                'result': r['result'],
            })
    return by_team


def date_le(game_date_mdy, cutoff_iso):
    """Return True if game date (MM/DD/YYYY) is on or before cutoff (YYYY-MM-DD)."""
    try:
        m, d, y = game_date_mdy.split('/')
        game_iso = f'{y}-{int(m):02d}-{int(d):02d}'
        return game_iso <= cutoff_iso
    except Exception:
        return False


def parse_win(result):
    """Result string starts with W/L; (T are rare draws — count as neither)."""
    if not result:
        return None
    c = result.strip()[:1].upper()
    if c == 'W':
        return True
    if c == 'L':
        return False
    return None


def quad_bucket(opp_rank, venue):
    v = venue if venue in Q1 else 'home'
    if opp_rank <= Q1[v]:
        return 'q1'
    if opp_rank <= Q2[v]:
        return 'q2'
    return 'q34'


def main():
    sel_dates = load_selection_dates()
    tournaments = load_tournament_master()

    out_rows = []

    for year in YEARS:
        cutoff = sel_dates[year]
        name2yid, name2rank, yid2name = load_selection_rpi(year)
        sched = load_schedule_for_year(year)
        entrants = tournaments[year]

        # Build per-year conference lookup from historical_selection_rpi if needed later.
        # For now we don't have conference here — that's a separate join.

        for e in entrants:
            n = norm(e['team_rpi_name'])
            yid = name2yid.get(n)
            if yid is None:
                # Try the canonical name
                n2 = norm(e['team_canonical'])
                yid = name2yid.get(n2)
                if yid is None:
                    out_rows.append({
                        'year': year, 'team': e['team_rpi_name'],
                        'rpi_rank': e['rpi_rank'],
                        'in_tournament': e['in_tournament'],
                        'is_host': e['is_host'],
                        'w': e['w'], 'l': e['l'],
                        'q1_w': '', 'q1_l': '', 'q2_w': '', 'q2_l': '',
                        'q1q2_w': '', 'q1q2_g': '', 'q1q2_pct': '',
                        'note': 'no_yearid_match',
                    })
                    continue
                n = n2

            games = sched.get(yid, [])
            q1_w = q1_l = q2_w = q2_l = 0
            for g in games:
                if not date_le(g['date'], cutoff):
                    continue
                won = parse_win(g['result'])
                if won is None:
                    continue
                opp_rank = name2rank.get(g['opp_norm'])
                if opp_rank is None:
                    # opponent not in selection-week RPI → non-D1 or unknown
                    # Treat as Q4 (no Q1/Q2 contribution); skip for counts.
                    continue
                bucket = quad_bucket(opp_rank, g['loc'])
                if bucket == 'q1':
                    q1_w += won
                    q1_l += (not won)
                elif bucket == 'q2':
                    q2_w += won
                    q2_l += (not won)

            q1q2_w = q1_w + q2_w
            q1q2_g = q1_w + q1_l + q2_w + q2_l
            q1q2_pct = (q1q2_w / q1q2_g) if q1q2_g > 0 else 0.0

            out_rows.append({
                'year': year, 'team': e['team_rpi_name'],
                'rpi_rank': e['rpi_rank'],
                'in_tournament': e['in_tournament'],
                'is_host': e['is_host'],
                'w': e['w'], 'l': e['l'],
                'q1_w': q1_w, 'q1_l': q1_l, 'q2_w': q2_w, 'q2_l': q2_l,
                'q1q2_w': q1q2_w, 'q1q2_g': q1q2_g,
                'q1q2_pct': round(q1q2_pct, 4),
                'note': '',
            })

    out_path = BRACK / 'historical_q1q2_baseball.csv'
    fields = ['year', 'team', 'rpi_rank', 'in_tournament', 'is_host', 'w', 'l',
              'q1_w', 'q1_l', 'q2_w', 'q2_l',
              'q1q2_w', 'q1q2_g', 'q1q2_pct', 'note']
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    # Quick audit
    print(f'Wrote {len(out_rows)} rows to {out_path}')
    unmatched = [r for r in out_rows if r['note']]
    print(f'Unmatched yearId: {len(unmatched)}')
    for r in unmatched[:10]:
        print(f"  {r['year']} {r['team']}")

    # Sanity: hosts per year
    hosts_per_year = defaultdict(int)
    for r in out_rows:
        if r['is_host']:
            hosts_per_year[r['year']] += 1
    print('Hosts per year:', dict(sorted(hosts_per_year.items())))


if __name__ == '__main__':
    main()
