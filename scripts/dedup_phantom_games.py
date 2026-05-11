"""Remove phantom duplicate gameIds from box-score PBP files.

NCAA's scoreboard occasionally re-lists the same game under a second gameId
(suspended/resumed, rescheduled, or system glitch), producing byte-identical
stat lines in our box-score CSVs under two different gameIds. The standard
dedup in update_data.sh (subset=gameId/playerId/teamName) only catches
duplicates WITHIN a single gameId. This script catches the cross-gameId case.

Algorithm: for each (date, set of teamNames) with 2+ gameIds, compare their
player stat lines. If byte-identical, the second gameId is a phantom; drop it.
Real doubleheaders (different stat lines) are preserved.

Run: python scripts/dedup_phantom_games.py
"""

from pathlib import Path
import pandas as pd

PBP_ROOT = Path(__file__).resolve().parent.parent / "pbp_data"
SPORTS = ["baseball", "softball"]
DIVISIONS = ["D1", "D2", "D3"]
STAT_TYPES = ["hitting", "pitching", "fielding"]

# Columns used to decide "identical stat lines" (hitting has the richest signal).
DETECT_COLS = [
    "teamName", "playerName",
    "ab", "r", "h", "doubles", "triples", "tb", "hr", "rbi",
    "bb", "hbp", "k",
]


def find_phantom_gids(hitting_df):
    """Return set of gameIds that are byte-identical phantoms of another gameId.

    Groups candidates by `teams` ALONE (not date). NCAA's phantom duplicates can
    have a date shift (suspended/resumed, rescheduled, etc.), so the original
    (date, teams) grouping missed them. Byte-identical 24-player stat lines
    across two games is essentially impossible by chance even when two real
    games are played against the same opponent on different days — different
    games yield different stat lines.
    """
    df = hitting_df.copy()

    game_meta = df.groupby("gameId").agg(
        teams=("teamName", lambda s: frozenset(s.dropna().unique())),
    ).reset_index()

    groups = game_meta.groupby("teams")["gameId"].apply(list).reset_index()
    multi = groups[groups["gameId"].apply(len) > 1]

    use_cols = [c for c in DETECT_COLS if c in df.columns]
    phantoms = set()
    for _, row in multi.iterrows():
        gids = sorted(row["gameId"])
        # Compare every pair; keep the lowest-id as canonical, drop any byte-match
        canonical_signatures = {}
        for gid in gids:
            sig = (
                df[df["gameId"] == gid][use_cols]
                .sort_values(["teamName", "playerName"])
                .reset_index(drop=True)
            )
            sig_key = tuple(map(tuple, sig.values))
            if sig_key in canonical_signatures and len(sig) > 0:
                phantoms.add(gid)
            else:
                canonical_signatures[sig_key] = gid
    return phantoms


def run_one_sport_div(sport, div):
    hit_path = PBP_ROOT / sport / f"hitting_pbp_{div}.csv"
    if not hit_path.exists():
        return {"sport": sport, "div": div, "skipped": True}

    hit_df = pd.read_csv(hit_path, low_memory=False)
    phantoms = find_phantom_gids(hit_df)
    if not phantoms:
        return {"sport": sport, "div": div, "phantoms": 0, "rows_removed": 0}

    total_removed = 0
    for stat in STAT_TYPES:
        p = PBP_ROOT / sport / f"{stat}_pbp_{div}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, low_memory=False)
        before = len(df)
        df = df[~df["gameId"].isin(phantoms)]
        removed = before - len(df)
        if removed:
            df.to_csv(p, index=False)
            total_removed += removed
    return {
        "sport": sport,
        "div": div,
        "phantoms": len(phantoms),
        "rows_removed": total_removed,
    }


def main():
    total_phantoms = 0
    total_rows = 0
    for sport in SPORTS:
        for div in DIVISIONS:
            r = run_one_sport_div(sport, div)
            if r.get("skipped"):
                continue
            if r["phantoms"]:
                print(
                    f"  {sport} {div}: {r['phantoms']} phantom gameIds, "
                    f"{r['rows_removed']:,} rows removed"
                )
                total_phantoms += r["phantoms"]
                total_rows += r["rows_removed"]
    if total_phantoms == 0:
        print("  No phantom duplicates found.")
    else:
        print(f"  TOTAL: {total_phantoms} phantom gameIds, {total_rows:,} rows removed")


if __name__ == "__main__":
    main()
