"""Offline builder for the Draft Assistant page.

The live site only has SEASON-aggregate hitting/pitching. This builder runs LOCALLY
(where the per-game PBP box scores live in scrape_final) and precomputes each
player's single best "weekend series" into a small committed CSV the page reads:
data/draft_best_series.csv.

Best weekend series = best 3 games (>=2) vs the SAME opponent on consecutive days
(a true college series). Falls back to best 3 consecutive games if no clean
same-opponent series exists. Scored to capture "he went insane" at the plate.

Join: PBP box scores key on statsPlayerSeq (a dead namespace vs our players.csv),
so we bridge by (playerName, teamName->team_id) with a name-unique fallback (~94%).
Opponent is derived from the other team sharing each gameId.

Run:  py scripts/build_draft_assistant.py
Reads PBP from C:/Dev/scrape_final/output/{year}/baseball/pbp/hitting_pbp_D*.csv
(2025 + 2026 are the only years with PBP). Writes data/draft_best_series.csv.
"""
import csv, glob, datetime as dt
from collections import defaultdict
from pathlib import Path

CB = Path("C:/Dev/chart-builder-app/data")
SF = Path("C:/Dev/scrape_final/output")
OUT = CB / "draft_best_series.csv"
YEARS = ["2025", "2026"]


def norm(s):
    s = str(s).strip()
    return s[:-2] if s.endswith(".0") else s


def i(x):
    try:
        return int(float(x))
    except Exception:
        return 0


def pdate(s):
    s = str(s).strip()
    for fmt in ("%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except Exception:
            pass
    return None


# ---- id bridge ----
players = list(csv.DictReader(open(CB / "players.csv", encoding="latin-1")))
teams = {r["name"].strip(): norm(r["id"])
         for r in csv.DictReader(open(CB / "teams.csv", encoding="latin-1"))
         if r["sport"] == "Baseball"}
by_name_team, name_to_ids = {}, defaultdict(list)
for r in players:
    by_name_team[(r["player_name"].strip().lower(), norm(r["team_id"]))] = norm(r["id"])
    name_to_ids[r["player_name"].strip().lower()].append(norm(r["id"]))


def resolve_pid(name, team):
    tid = teams.get(team.strip())
    key = (name.strip().lower(), tid)
    if tid and key in by_name_team:
        return by_name_team[key]
    ids = name_to_ids.get(name.strip().lower(), [])
    return ids[0] if len(ids) == 1 else None


# ---- load per-game box scores; derive opponent from gameId ----
# games[pid] = list of dicts {date, opp, ab,h,2b,3b,hr,rbi,bb,sb,r}
games = defaultdict(list)
files = []
for y in YEARS:
    files += glob.glob(str(SF / y / "baseball" / "pbp" / f"hitting_pbp_D*.csv"))
files = [f for f in files if ".bak" not in f and "_TEST" not in f]
print(f"PBP box-score files: {len(files)}")

for f in files:
    rows = list(csv.DictReader(open(f, encoding="latin-1")))
    # gameId -> set of teams (to find opponent)
    gteams = defaultdict(set)
    for r in rows:
        gteams[r["gameId"]].add(r["teamName"].strip())
    for r in rows:
        pid = resolve_pid(r["playerName"], r["teamName"])
        if not pid:
            continue
        d = pdate(r["date"])
        if not d:
            continue
        others = [t for t in gteams[r["gameId"]] if t != r["teamName"].strip()]
        opp = others[0] if others else "?"
        games[pid].append({
            "date": d, "opp": opp, "gid": r["gameId"],
            "ab": i(r["ab"]), "h": i(r["h"]), "2b": i(r["doubles"]), "3b": i(r["triples"]),
            "hr": i(r["hr"]), "rbi": i(r["rbi"]), "bb": i(r["bb"]), "sb": i(r["sb"]), "r": i(r["r"]),
        })
print(f"players with PBP games: {len(games)}")


def line_score(gs):
    """A 'he went insane' score for a set of game lines."""
    tb = sum(g["h"] - g["2b"] - g["3b"] - g["hr"] + 2 * g["2b"] + 3 * g["3b"] + 4 * g["hr"] for g in gs)
    return tb + sum(g["rbi"] + g["bb"] + g["sb"] + g["r"] for g in gs)


def agg(gs):
    a = {k: sum(g[k] for g in gs) for k in ("ab", "h", "2b", "3b", "hr", "rbi", "bb", "sb", "r")}
    a["g"] = len(gs)
    a["tb"] = a["h"] - a["2b"] - a["3b"] - a["hr"] + 2 * a["2b"] + 3 * a["3b"] + 4 * a["hr"]
    pa = a["ab"] + a["bb"]
    a["avg"] = round(a["h"] / a["ab"], 3) if a["ab"] else 0
    a["obp"] = round((a["h"] + a["bb"]) / pa, 3) if pa else 0
    a["slg"] = round(a["tb"] / a["ab"], 3) if a["ab"] else 0
    return a


def best_series(gs):
    """Best same-opponent consecutive-day window of up to 3 games; fallback best 3 consecutive."""
    gs = sorted(gs, key=lambda g: g["date"])
    best, best_s = None, -1
    # same-opponent series: contiguous runs vs one opp within a 4-day span
    n = len(gs)
    for a in range(n):
        for b in range(a + 1, min(a + 2, n - 1) + 1):  # windows of 2 or 3 games (cap 3)
            win = gs[a:b + 1]
            span = (win[-1]["date"] - win[0]["date"]).days
            same_opp = len({g["opp"] for g in win}) == 1
            if span <= 4 and same_opp and len(win) >= 2:
                s = line_score(win)
                if s > best_s:
                    best_s, best = s, win
    if best:
        return best, "series"
    # fallback: best any 3 (or 2) consecutive games by date gap <=4
    for w in (3, 2):
        for a in range(n - w + 1):
            win = gs[a:a + w]
            if (win[-1]["date"] - win[0]["date"]).days <= 6:
                s = line_score(win)
                if s > best_s:
                    best_s, best = s, win
    return best, "stretch"


# ---- pick each player's best series and write ----
hdr = ["player_id", "kind", "opponent", "date_start", "date_end", "g",
       "ab", "h", "2b", "3b", "hr", "rbi", "bb", "sb", "r", "tb", "avg", "obp", "slg", "summary"]
out_rows = []
for pid, gs in games.items():
    if len(gs) < 2:
        continue
    win, kind = best_series(gs)
    if not win or len(win) < 2:
        continue
    a = agg(win)
    if a["h"] < 3:  # not worth highlighting
        continue
    opp = win[0]["opp"] if kind == "series" else "multiple"
    summ = (f"{a['g']} games vs {opp}: {a['h']}-for-{a['ab']}, "
            f"{a['hr']} HR, {a['rbi']} RBI, {a['bb']} BB"
            + (f", {a['sb']} SB" if a['sb'] else "")
            + f" ({a['avg']:.3f}/{a['obp']:.3f}/{a['slg']:.3f})")
    out_rows.append({
        "player_id": pid, "kind": kind, "opponent": opp,
        "date_start": win[0]["date"].isoformat(), "date_end": win[-1]["date"].isoformat(),
        **{k: a[k] for k in ("g", "ab", "h", "2b", "3b", "hr", "rbi", "bb", "sb", "r", "tb", "avg", "obp", "slg")},
        "summary": summ,
    })

with open(OUT, "w", newline="", encoding="utf-8") as fo:
    w = csv.DictWriter(fo, fieldnames=hdr)
    w.writeheader()
    w.writerows(out_rows)
print(f"wrote {OUT} | {len(out_rows)} players with a best series")
