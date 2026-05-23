"""
Compute current-season Q1+Q2 records for a list of teams (USC + Mercer for 2026).
Uses:
  - data/schedules_full_baseball.csv (current schedule with isAway + opponentName)
  - data/baseball_rpi_D1.csv (current RPI ranks + names)

Venue rule: neutral if opponentName contains ' @' (NCAA neutral-site convention),
else away if isAway == 1, else home. Matches the historical scrape convention.

Q1/Q2 thresholds match historical: H<=30/<=75, N<=50/<=100, A<=75/<=135.

Output: data/bracketology/current_q1q2_baseball.csv (one row per team in TEAMS).
"""
import csv
import re
from pathlib import Path

REPO = Path(r'C:\Dev\chart-builder-app')
BRACK = REPO / 'data' / 'bracketology'

TEAMS = ['Southern California', 'Mercer']

# Manual overrides applied AFTER computation. Use this for cases where the
# 64A internal RPI calc diverges from the d1baseball.com / NCAA published
# numbers and we want the chart label to match what readers see externally.
# Each override is keyed by team name. Only fields listed are replaced.
OVERRIDES = {
    'Mercer': {
        # Per d1baseball.com Selection Monday display 2026-05-23
        'rpi_rank': 28,
        'q1_w': 0, 'q1_l': 7,   # 9-13 with Q2 totals below = full override
        'q2_w': 9, 'q2_l': 6,
        # Recomputed: q1q2_w = q1_w + q2_w; q1q2_g = sum of w+l; pct = w/g
        '_note': 'Override: d1baseball.com published RPI + Q1/Q2 split (2026-05-23)',
    },
}

Q1 = {'home': 30, 'neutral': 50, 'away': 75}
Q2 = {'home': 75, 'neutral': 100, 'away': 135}


def norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


# Schedule-name -> RPI-name aliases for cases the normalizer can't bridge
NAME_ALIASES = {
    'uncgreensboro': 'uncg',
    # Add more here when audit surfaces them
}


def normalize_opp(opp_name):
    """Apply opponent-name aliases on top of base normalization."""
    n = norm(opp_name)
    return NAME_ALIASES.get(n, n)


def strip_neutral_suffix(opp_name):
    """opponentName like 'Mount Union @Canton, OH' -> ('Mount Union', True).
    Also strip leading '#N ' conference-tournament seed prefix (e.g. '#5 The Citadel').
    """
    s = opp_name or ''
    is_neutral = False
    if ' @' in s:
        s = s.split(' @', 1)[0].strip()
        is_neutral = True
    # Strip leading '#<digits> ' that NCAA uses for conf-tourney seeds
    s = re.sub(r'^#\d+\s+', '', s).strip()
    return s, is_neutral


def main():
    # Load RPI rank lookup by normalized teamName
    rpi = {}
    name_to_yid = {}
    teams_meta = {}
    with open(REPO / 'data' / 'baseball_rpi_D1.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if not r.get('rank') or not r.get('teamName'):
                continue
            n = norm(r['teamName'])
            try:
                rpi[n] = int(float(r['rank']))
            except ValueError:
                continue
            if r.get('teamSeasonId'):
                try:
                    name_to_yid[n] = int(float(r['teamSeasonId']))
                except ValueError:
                    pass
            teams_meta[n] = {
                'teamName': r['teamName'],
                'conference': r['conference'],
                'record': r['record'],
                'rpi_rank': rpi[n],
            }

    # Load schedules indexed by teamYearId
    games_by_team = {}
    with open(REPO / 'data' / 'schedules_full_baseball.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            try:
                tid = int(float(r['teamYearId']))
            except (ValueError, KeyError):
                continue
            games_by_team.setdefault(tid, []).append(r)

    out_rows = []
    for team_name in TEAMS:
        n = norm(team_name)
        meta = teams_meta.get(n)
        if meta is None:
            print(f'NOT FOUND in RPI: {team_name}')
            continue
        # Find teamYearId via name->yid OR via schedules teamName scan
        tid = None
        # Schedules has teamName; find the yid by scanning
        for cand_tid, glist in games_by_team.items():
            if glist and norm(glist[0]['teamName']) == n:
                tid = cand_tid
                break
        if tid is None:
            print(f'NO teamYearId for {team_name}')
            continue

        q1_w = q1_l = q2_w = q2_l = 0
        w = l = 0
        for g in games_by_team[tid]:
            if g.get('isFuture') == 'True':
                continue
            # isWin='1' for wins, blank for losses (NOT '0'). Use result string.
            result = (g.get('result') or '').strip()
            if result.startswith('W'):
                won = True
                w += 1
            elif result.startswith('L'):
                won = False
                l += 1
            else:
                continue
            opp_raw = g.get('opponentName', '')
            opp_clean, is_neutral = strip_neutral_suffix(opp_raw)
            is_away = (g.get('isAway') == '1')
            venue = 'neutral' if is_neutral else ('away' if is_away else 'home')
            opp_rank = rpi.get(normalize_opp(opp_clean))
            if opp_rank is None:
                continue
            if opp_rank <= Q1[venue]:
                if won: q1_w += 1
                else: q1_l += 1
            elif opp_rank <= Q2[venue]:
                if won: q2_w += 1
                else: q2_l += 1

        q1q2_w = q1_w + q2_w
        q1q2_g = q1_w + q1_l + q2_w + q2_l
        q1q2_pct = q1q2_w / q1q2_g if q1q2_g else 0.0

        row = {
            'year': 2026,
            'team': meta['teamName'],
            'conference': meta['conference'],
            'rpi_rank': meta['rpi_rank'],
            'record': meta['record'],
            'w': w, 'l': l,
            'q1_w': q1_w, 'q1_l': q1_l,
            'q2_w': q2_w, 'q2_l': q2_l,
            'q1q2_w': q1q2_w, 'q1q2_g': q1q2_g,
            'q1q2_pct': round(q1q2_pct, 4),
            'note': '',
        }
        # Apply override if present
        ovr = OVERRIDES.get(meta['teamName'])
        if ovr:
            for k, v in ovr.items():
                if k.startswith('_'):
                    row['note'] = v.replace('_note', '').lstrip(': ').strip() \
                        if k == '_note' else v
                    continue
                row[k] = v
            # Recompute derived fields from overridden q1/q2
            row['q1q2_w'] = row['q1_w'] + row['q2_w']
            row['q1q2_g'] = row['q1_w'] + row['q1_l'] + row['q2_w'] + row['q2_l']
            row['q1q2_pct'] = round(row['q1q2_w'] / row['q1q2_g'], 4) \
                              if row['q1q2_g'] else 0.0
        out_rows.append(row)

    fields = ['year', 'team', 'conference', 'rpi_rank', 'record', 'w', 'l',
              'q1_w', 'q1_l', 'q2_w', 'q2_l',
              'q1q2_w', 'q1q2_g', 'q1q2_pct', 'note']
    out_path = BRACK / 'current_q1q2_baseball.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f'Wrote {len(out_rows)} rows to {out_path}')
    for r in out_rows:
        print(f"  {r['team']:25s} RPI #{r['rpi_rank']:>3}  {r['record']}  "
              f"Q1 {r['q1_w']}-{r['q1_l']}  Q2 {r['q2_w']}-{r['q2_l']}  "
              f"Q1+Q2 {r['q1q2_w']}/{r['q1q2_g']} ({r['q1q2_pct']:.3f})")


if __name__ == '__main__':
    main()
