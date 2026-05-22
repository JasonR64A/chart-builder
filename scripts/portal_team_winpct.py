"""
Team-level portal reliance vs. 2026 win percentage.

For every 2026 NCAA baseball team with >=20 games played:
  - share of team AB from "portal players" (any player_id in portal history)
  - share of team IP from portal players
  - 2026 overall win %
  - correlations + quartile breakdown, overall and by division

Usage: python scripts/portal_team_winpct.py
"""

from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

CB = Path("C:/Dev/chart-builder-app/data")
SC = Path("C:/Dev/scrape_final")
TEAM_RECORDS = SC / "output/2026/04/21/baseball/stats/team_records.csv"


def load():
    portal = pd.read_csv(CB / "portal_rank_player.csv", low_memory=False)
    hit = pd.read_csv(CB / "hitting.csv", low_memory=False)
    pit = pd.read_csv(CB / "pitching.csv", low_memory=False)
    tr = pd.read_csv(TEAM_RECORDS, low_memory=False)
    st = pd.read_csv(SC / "output/2026/baseball/teams.csv", low_memory=False)
    teams_cb = pd.read_csv(CB / "teams.csv", low_memory=False)
    confs = pd.read_csv(CB / "conferences.csv", low_memory=False)
    return portal, hit, pit, tr, st, teams_cb, confs


def build_team_reliance(hit, pit, portal_pids):
    h = hit[hit["year"] == 2026].copy()
    p = pit[pit["year"] == 2026].copy()
    h["is_portal"] = h["player_id"].isin(portal_pids)
    p["is_portal"] = p["player_id"].isin(portal_pids)

    team_hit = (
        h.groupby("team_id")
        .apply(
            lambda g: pd.Series({
                "total_ab": g["at_bats"].sum(),
                "portal_ab": g.loc[g["is_portal"], "at_bats"].sum(),
            }),
            include_groups=False,
        )
        .reset_index()
    )
    team_hit["portal_ab_pct"] = team_hit["portal_ab"] / team_hit["total_ab"] * 100

    team_pit = (
        p.groupby("team_id")
        .apply(
            lambda g: pd.Series({
                "total_ip": g["innings_pitched"].sum(),
                "portal_ip": g.loc[g["is_portal"], "innings_pitched"].sum(),
            }),
            include_groups=False,
        )
        .reset_index()
    )
    team_pit["portal_ip_pct"] = team_pit["portal_ip"] / team_pit["total_ip"] * 100

    return team_hit, team_pit


def build_winpct(tr, st, teams_cb):
    yid_to_org = dict(
        zip(st["yearId"].astype("Int64"), st["orgId"].astype("Int64"))
    )
    org_to_64a = defaultdict(list)
    for _, r in teams_cb.iterrows():
        if pd.notna(r["team_id_ncaa"]) and r["sport"] == "Baseball":
            org_to_64a[int(r["team_id_ncaa"])].append(int(r["id"]))

    def yid_to_64a(y):
        org = yid_to_org.get(y)
        if pd.isna(org) or org is None:
            return None
        lst = org_to_64a.get(int(org), [])
        return lst[0] if lst else None

    tr = tr.copy()
    tr["team_id_64a"] = tr["Team ID"].apply(
        lambda x: yid_to_64a(int(x)) if pd.notna(x) else None
    )
    tr["games"] = tr["Overall Wins"] + tr["Overall Losses"] + tr["Overall Ties"]
    tr["win_pct"] = tr["Overall Wins"] / tr["games"].replace(0, np.nan)
    return tr


def division_sets(teams_cb, confs):
    out = {}
    for label, tag in [("D1", "D-I"), ("D2", "D-II"), ("D3", "D-III")]:
        cs = set(confs[confs["division"] == tag]["id"])
        out[label] = set(
            teams_cb[
                (teams_cb["sport"] == "Baseball")
                & (teams_cb["conference_id"].isin(cs))
            ]["id"].astype(int)
        )
    return out


def safe_q(sub, col, nq=4):
    try:
        q = pd.qcut(sub[col], nq, duplicates="drop")
    except Exception:
        return None
    n = len(q.cat.categories)
    return pd.qcut(
        sub[col], nq, labels=[f"Q{i+1}" for i in range(n)], duplicates="drop"
    )


def main():
    portal, hit, pit, tr, st, teams_cb, confs = load()
    portal_pids = set(portal["player_id"].dropna().astype(int))
    divs = division_sets(teams_cb, confs)

    team_hit, team_pit = build_team_reliance(hit, pit, portal_pids)
    tr = build_winpct(tr, st, teams_cb)

    df = tr.merge(team_hit, left_on="team_id_64a", right_on="team_id", how="inner")
    df = df.merge(
        team_pit,
        left_on="team_id_64a",
        right_on="team_id",
        how="inner",
        suffixes=("_h", "_p"),
    )
    df = df[df["games"] >= 20].copy()
    df["name"] = df["team_id_64a"].map(dict(zip(teams_cb["id"], teams_cb["name"])))
    df["div"] = df["team_id_64a"].apply(
        lambda t: "D1" if t in divs["D1"]
        else "D2" if t in divs["D2"]
        else "D3" if t in divs["D3"]
        else "?"
    )

    print(f"Sample sizes by division: {df['div'].value_counts().to_dict()}")

    print("\n=== Pearson correlations (Win% vs portal reliance) ===")
    for d in ["ALL", "D1", "D2", "D3"]:
        sub = df if d == "ALL" else df[df["div"] == d]
        ca = sub["win_pct"].corr(sub["portal_ab_pct"])
        ci = sub["win_pct"].corr(sub["portal_ip_pct"])
        print(
            f"  {d:4s} (n={len(sub):3d}):  "
            f"Win% vs Portal AB% = {ca:+.3f}    "
            f"Win% vs Portal IP% = {ci:+.3f}"
        )

    for stat, col in [("AB", "portal_ab_pct"), ("IP", "portal_ip_pct")]:
        print(f"\n=== Win% by portal-{stat}% quartile ===")
        for d in ["ALL", "D1", "D2", "D3"]:
            sub = (df if d == "ALL" else df[df["div"] == d]).copy()
            q = safe_q(sub, col, 4)
            if q is None:
                print(f"-- {d} -- skipped (too few unique values)")
                continue
            sub["q"] = q
            g = (
                sub.groupby("q", observed=True)
                .agg(
                    n=("team_id_64a", "size"),
                    portal_pct=(col, "mean"),
                    win_pct=("win_pct", "mean"),
                )
                .round(3)
            )
            print(f"\n-- {d} --")
            print(g.to_string())

    print("\n=== D1 TOP 10 portal-AB teams ===")
    top = df[df["div"] == "D1"].nlargest(10, "portal_ab_pct")
    print(
        top[["name", "portal_ab_pct", "portal_ip_pct", "win_pct"]]
        .round(3)
        .to_string(index=False)
    )

    print(
        "\nCAVEAT: 2026 win pcts are season-to-date (~75-80% through schedule). "
        "\"Portal player\" = any player_id ever in portal table (2023-2025), "
        "not just recent transfers."
    )


if __name__ == "__main__":
    main()
