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


# ---- id bridge + team/conference metadata ----
players = list(csv.DictReader(open(CB / "players.csv", encoding="latin-1")))
conf_rows = {norm(r["id"]): r for r in csv.DictReader(open(CB / "conferences.csv", encoding="latin-1"))}
POWER = {"SEC", "ACC", "Big 12", "Big Ten", "Pac-12", "Big 12 Conference",
         "Southeastern Conference", "Atlantic Coast Conference"}  # 'vs quality' proxy
teams = {}            # teamName -> 64A team_id
team_conf = {}        # teamName -> conference_id
team_power = {}       # teamName -> bool (opponent counts as 'quality')
for r in csv.DictReader(open(CB / "teams.csv", encoding="latin-1")):
    if r["sport"] != "Baseball":
        continue
    nm = r["name"].strip()
    teams[nm] = norm(r["id"])
    cid = norm(r["conference_id"])
    team_conf[nm] = cid
    c = conf_rows.get(cid, {})
    team_power[nm] = (c.get("abbreviation", "").strip() in POWER) or (c.get("name", "").strip() in POWER)
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
        own_team = r["teamName"].strip()
        own_c = team_conf.get(own_team)
        opp_c = team_conf.get(opp)
        ih = str(r.get("isHome", "")).strip() in ("1", "1.0")
        games[pid].append({
            "date": d, "opp": opp, "gid": r["gameId"], "year": d.year,
            "is_conf": (own_c is not None and own_c == opp_c),
            "opp_known": opp_c is not None, "is_home": ih, "month": d.month,
            "opp_power": team_power.get(opp, False),
            "ab": i(r["ab"]), "h": i(r["h"]), "2b": i(r["doubles"]), "3b": i(r["triples"]),
            "hr": i(r["hr"]), "rbi": i(r["rbi"]), "bb": i(r["bb"]), "sb": i(r["sb"]), "r": i(r["r"]),
            "hbp": i(r.get("hbp", 0)), "sf": i(r.get("sf", 0)),
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


# ============================================================
# Splits engine — per-player 2026 splits + peer percentiles
# ============================================================
SPLIT_YEAR = 2026
PA_GATE = 40
SPLITS = {
    "conference": lambda g: g["is_conf"] and g["opp_known"],
    "road": lambda g: not g["is_home"],
    "home": lambda g: g["is_home"],
    "late": lambda g: g["month"] in (5, 6),
    "quality": lambda g: g["opp_power"],
}


def line(gs):
    a = {k: sum(g[k] for g in gs) for k in ("ab", "h", "2b", "3b", "hr", "rbi", "bb", "sb", "hbp", "sf")}
    a["tb"] = a["h"] - a["2b"] - a["3b"] - a["hr"] + 2 * a["2b"] + 3 * a["3b"] + 4 * a["hr"]
    a["pa"] = a["ab"] + a["bb"] + a["hbp"] + a["sf"]
    a["g"] = len(gs)
    a["avg"] = a["h"] / a["ab"] if a["ab"] else 0.0
    obp = (a["h"] + a["bb"] + a["hbp"]) / a["pa"] if a["pa"] else 0.0
    slg = a["tb"] / a["ab"] if a["ab"] else 0.0
    a["ops"] = round(obp + slg, 3)
    a["iso"] = round(slg - a["avg"], 3)
    a["avg"] = round(a["avg"], 3)
    return a

# per player: 2026 games, overall + each split (gate applied)
player_splits = {}      # pid -> {split: line, '_overall': line}
peer = defaultdict(list)      # split -> [ops] over gated players
peer_iso = defaultdict(list)  # split -> [iso] over gated players
for pid, gs in games.items():
    g26 = [g for g in gs if g["year"] == SPLIT_YEAR]
    if len(g26) < 5:
        continue
    overall = line(g26)
    if overall["pa"] < 60:
        continue
    rec = {"_overall": overall}
    for name, filt in SPLITS.items():
        sub = [g for g in g26 if filt(g)]
        if not sub:
            continue
        L = line(sub)
        if L["pa"] >= PA_GATE:
            rec[name] = L
            peer[name].append(L["ops"])
            peer_iso[name].append(L["iso"])
    player_splits[pid] = rec

peer_sorted = {k: sorted(v) for k, v in peer.items()}
peer_iso_sorted = {k: sorted(v) for k, v in peer_iso.items()}


def _pct(pool, split, val):
    import bisect
    arr = pool.get(split, [])
    return round(100 * bisect.bisect_left(arr, val) / len(arr)) if arr else None

shdr = ["player_id", "split", "g", "pa", "ab", "h", "hr", "tb", "avg", "iso", "ops",
        "ops_pct", "iso_pct", "overall_ops"]
srows = []
for pid, rec in player_splits.items():
    ov = rec["_overall"]["ops"]
    for name in SPLITS:
        if name not in rec:
            continue
        L = rec[name]
        srows.append({
            "player_id": pid, "split": name, "g": L["g"], "pa": L["pa"], "ab": L["ab"],
            "h": L["h"], "hr": L["hr"], "tb": L["tb"], "avg": f'{L["avg"]:.3f}',
            "iso": f'{L["iso"]:.3f}', "ops": f'{L["ops"]:.3f}',
            "ops_pct": _pct(peer_sorted, name, L["ops"]),
            "iso_pct": _pct(peer_iso_sorted, name, L["iso"]),
            "overall_ops": f"{ov:.3f}",
        })
SPLIT_OUT = CB / "draft_splits.csv"
with open(SPLIT_OUT, "w", newline="", encoding="utf-8") as fo:
    w = csv.DictWriter(fo, fieldnames=shdr)
    w.writeheader()
    w.writerows(srows)
print(f"wrote {SPLIT_OUT} | {len(srows)} split-rows across {len(player_splits)} players "
      f"| peer pools: {{{', '.join(f'{k}:{len(v)}' for k,v in peer_sorted.items())}}}")
