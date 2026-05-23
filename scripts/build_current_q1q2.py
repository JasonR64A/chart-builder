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

Q1 = {'home': 30, 'neutral': 50, 'away': 75}
Q2 = {'home': 75, 'neutral': 100, 'away': 135}


def norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def strip_neutral_suffix(opp_name):
    """opponentName like 'Mount Union @Canton, OH' -> ('Mount Union', True)."""
    if isinstance(opp_name, str) and ' @' in opp_name:
        base = opp_name.split(' @', 1)[0].strip()
        return base, True
    return opp_name, False


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
            opp_rank = rpi.get(norm(opp_clean))
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

        out_rows.append({
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
        })

    fields = ['year', 'team', 'conference', 'rpi_rank', 'record', 'w', 'l',
              'q1_w', 'q1_l', 'q2_w', 'q2_l',
              'q1q2_w', 'q1q2_g', 'q1q2_pct']
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
