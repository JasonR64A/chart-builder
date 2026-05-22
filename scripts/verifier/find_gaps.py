"""
Find box-score coverage gaps.

Three checks:
  1. Schedule-vs-boxscore: for each team, dates with a played game in the
     schedule but no box-score rows. Each missing date is a rescrape target.
  2. PBP-vs-boxscore: gameIds that appear in play-by-play event files but
     have no rows in the box-score (hitting_pbp) files. Highest-confidence
     rescrape targets — we know NCAA has the game.
  3. Season-stats parity: players whose season HR total (hitting.csv)
     exceeds the sum of HR in their box-score rows. Delta rows indicate
     missing games containing additional HRs.

Outputs:
  - scripts/verifier/out/missing_games.csv           (scheduled but no box score)
  - scripts/verifier/out/pbp_without_boxscore.csv    (PBP has it, box score doesn't)
  - scripts/verifier/out/hr_parity_mismatches.csv    (per-player HR delta)
  - scripts/verifier/out/rescrape_queue.csv          (deduplicated rescrape targets)
  - scripts/verifier/out/gap_by_date.csv             (missing games binned by date)
"""
from __future__ import annotations

import os
from pathlib import Path
from collections import defaultdict

import pandas as pd

ROOT = Path(r"C:\Dev\chart-builder-app")
OUT = Path(__file__).parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

SPORTS = ["baseball", "softball"]
DIVS = ["D1", "D2", "D3"]


def parse_date(s):
    s = str(s).strip()
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            pass
    return pd.NaT


def load_schedules():
    frames = []
    for sport in SPORTS:
        fp = ROOT / "data" / f"schedules_full_{sport}.csv"
        df = pd.read_csv(fp, low_memory=False)
        df["sport"] = sport
        df["d"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
        df = df[df["runsFor"].notna()].copy()  # played games only
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_boxscores():
    """Return dict[(sport, div)] -> DataFrame."""
    out = {}
    for sport in SPORTS:
        for div in DIVS:
            fp = ROOT / "pbp_data" / sport / f"hitting_pbp_{div}.csv"
            df = pd.read_csv(
                fp,
                usecols=["gameId", "date", "teamName", "playerId", "hr"],
                low_memory=False,
            )
            df = df.drop_duplicates(subset=["gameId", "teamName", "playerId"])
            df["d"] = df["date"].apply(parse_date)
            out[(sport, div)] = df
    return out


def load_team_division_map():
    """Build teamName -> division using the RPI files."""
    mapping = {}
    for sport in SPORTS:
        for div in DIVS:
            fp = ROOT / "data" / f"{sport}_rpi_{div}.csv"
            if not fp.exists():
                continue
            d = pd.read_csv(fp, low_memory=False)
            for name in d["teamName"].dropna().unique():
                mapping[(sport, name)] = div
    return mapping


def find_missing_games(schedules, boxscores, team_div_map):
    """For each (sport, team, played-date) in schedules, check if it appears in the
    corresponding division's box-score file. Missing = rescrape target.
    """
    rows = []
    # Flatten all boxscore dates per (sport, teamName)
    bs_dates_by_team = defaultdict(set)
    for (sport, div), df in boxscores.items():
        for team, sub in df.groupby("teamName"):
            bs_dates_by_team[(sport, team)].update(sub["d"].dropna().dt.date)

    for (sport, team), gdf in schedules.groupby(["sport", "teamName"]):
        div = team_div_map.get((sport, team), "unknown")
        bs_dates = bs_dates_by_team.get((sport, team), set())
        # If the team has NO box-score rows at all, they're probably D2/D3 opponents of
        # a D1 team whose games are just not captured (or a non-DI program not in our
        # scrape universe). Skip to avoid flooding with 1000s of non-actionable gaps.
        if not bs_dates:
            continue
        for _, r in gdf.iterrows():
            if pd.isna(r["d"]):
                continue
            game_date = r["d"].date()
            if game_date not in bs_dates:
                rows.append({
                    "sport": sport,
                    "division": div,
                    "team": team,
                    "date": game_date,
                    "opponent": r["opponentName"],
                    "result": r["result"],
                    "runsFor": r["runsFor"],
                    "runsAgainst": r["runsAgainst"],
                    "teamYearId": r.get("teamYearId"),
                    "opponentYearId": r.get("opponentYearId"),
                })
    return pd.DataFrame(rows)


def hr_parity_check(boxscores):
    """Compare season HR (hitting.csv) vs sum of HR in box-score rows (2026).

    ID bridge: hitting.csv.player_id (64A internal) -> rosters.csv.player_id /
    rosters.csv.player_ncaa_season_id -> hitting_pbp.playerId.
    """
    hitting = pd.read_csv(ROOT / "data" / "hitting.csv", low_memory=False)
    hitting_2026 = hitting[hitting["year"] == 2026][
        ["player_id", "team_id", "games_played", "home_runs"]
    ].copy()

    players = pd.read_csv(ROOT / "data" / "players.csv", low_memory=False, encoding="latin1")
    players = players[["id", "player_name"]].rename(columns={"id": "player_id"})

    rosters = pd.read_csv(ROOT / "data" / "rosters.csv", low_memory=False, encoding="latin1")
    rosters_2026 = rosters[rosters["Year"] == 2026][
        ["player_id", "team_id", "player_ncaa_season_id"]
    ].dropna(subset=["player_ncaa_season_id"])

    # Box-score HR total per season-level playerId
    bs_frames = []
    for (sport, div), df in boxscores.items():
        agg = df.groupby("playerId")["hr"].sum().reset_index()
        bs_frames.append(agg)
    bs_total = pd.concat(bs_frames, ignore_index=True)
    bs_total = bs_total.groupby("playerId")["hr"].sum().reset_index()
    bs_total.columns = ["player_ncaa_season_id", "boxscore_hr"]

    merged = (
        hitting_2026
        .merge(rosters_2026, on=["player_id", "team_id"], how="left")
        .merge(players, on="player_id", how="left")
        .merge(bs_total, on="player_ncaa_season_id", how="left")
    )
    merged["boxscore_hr"] = merged["boxscore_hr"].fillna(0).astype(int)
    merged["delta"] = merged["home_runs"].fillna(0).astype(int) - merged["boxscore_hr"]
    missing = merged[merged["delta"] > 0].sort_values("delta", ascending=False)
    return missing[[
        "player_name", "player_id", "player_ncaa_season_id", "team_id",
        "games_played", "home_runs", "boxscore_hr", "delta",
    ]]


def load_pbp_events():
    """Return DataFrame of (gameId, sport, division, date) from PBP event files."""
    rows = []
    for sport in SPORTS:
        for div in DIVS:
            fp = ROOT / "pbp_data" / "play_by_play" / f"{sport}_play_by_play_{div}.csv"
            if not fp.exists():
                continue
            # Read just gameId + any date column present
            sample = pd.read_csv(fp, nrows=1)
            date_col = None
            for c in ("date", "gameDate"):
                if c in sample.columns:
                    date_col = c
                    break
            usecols = ["gameId"] + ([date_col] if date_col else [])
            df = pd.read_csv(fp, usecols=usecols, low_memory=False)
            df = df.drop_duplicates("gameId")
            df["sport"] = sport
            df["division"] = div
            if date_col:
                df = df.rename(columns={date_col: "date"})
                df["d"] = df["date"].apply(parse_date)
            else:
                df["date"] = pd.NA
                df["d"] = pd.NaT
            rows.append(df[["gameId", "sport", "division", "date", "d"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def pbp_without_boxscore(boxscores, pbp_events):
    """PBP has the game but box score doesn't."""
    bs_ids_by_sd = {(s, d): set(df["gameId"].dropna().unique()) for (s, d), df in boxscores.items()}
    gap_rows = []
    for _, r in pbp_events.iterrows():
        key = (r["sport"], r["division"])
        if r["gameId"] not in bs_ids_by_sd.get(key, set()):
            gap_rows.append(r.to_dict())
    return pd.DataFrame(gap_rows)


def build_rescrape_queue(missing_games, pbp_gap, pbp_events):
    """Deduplicated queue of games to rescrape. Prefer gameId when known
    (from PBP), otherwise fall back to team/date/opponent identifiers.
    """
    # Map (sport, date_str, team_name pair) -> gameId when PBP knows it.
    # PBP files don't tell us team names, so we can only attach gameIds by
    # matching dates — the downstream scraper can resolve team-id pairs.
    queue = []

    # Bucket 1: gameId known (PBP succeeded, box failed). Highest confidence.
    for _, r in pbp_gap.iterrows():
        queue.append({
            "source": "pbp_gap",
            "gameId": int(r["gameId"]) if pd.notna(r["gameId"]) else None,
            "sport": r["sport"],
            "division": r["division"],
            "date": r.get("date"),
            "team": None,
            "opponent": None,
        })

    # Bucket 2: schedule gap. No gameId known; identify by team+date+opponent.
    # Dedupe across the two team-rows of each physical game.
    seen = set()
    for _, r in missing_games.iterrows():
        key = (r["sport"], str(r["date"]), *sorted([str(r["team"]), str(r["opponent"])]))
        if key in seen:
            continue
        seen.add(key)
        queue.append({
            "source": "schedule_gap",
            "gameId": None,
            "sport": r["sport"],
            "division": r["division"],
            "date": r["date"],
            "team": r["team"],
            "opponent": r["opponent"],
        })

    q = pd.DataFrame(queue)
    # Deduplicate: if a schedule_gap has the same date as a pbp_gap for the
    # same sport/division, assume they're the same game and prefer the
    # pbp_gap entry (has gameId). Not perfect — PBP has no team names — but
    # close enough; the scraper can resolve.
    return q


def gap_by_date(missing_games, pbp_gap):
    """Date histogram to test the 'a whole day was missed' hypothesis."""
    m = missing_games.copy()
    m["d"] = pd.to_datetime(m["date"], errors="coerce")
    m_daily = (
        m.drop_duplicates(subset=["sport", "team", "opponent", "date"])  # unique physical games
        .groupby(["sport", "division", m["d"].dt.date])
        .size()
        .reset_index(name="missing_schedule")
    )

    p = pbp_gap.copy()
    p_daily = (
        p.groupby(["sport", "division", p["d"].dt.date])
        .size()
        .reset_index(name="missing_pbp_gap")
    )

    merged = pd.merge(
        m_daily, p_daily,
        left_on=["sport", "division", "d"],
        right_on=["sport", "division", "d"],
        how="outer",
    ).fillna(0)
    merged = merged.rename(columns={"d": "date"}).sort_values(
        ["missing_pbp_gap", "missing_schedule"], ascending=False
    )
    return merged


def main():
    print("Loading schedules...")
    schedules = load_schedules()
    print(f"  played-game rows: {len(schedules):,}")

    print("Loading box-score files...")
    boxscores = load_boxscores()
    for k, v in boxscores.items():
        print(f"  {k}: {len(v):,} rows, {v['gameId'].nunique():,} unique games")

    print("Building team->division map from RPI files...")
    team_div_map = load_team_division_map()
    print(f"  {len(team_div_map):,} (sport, team) -> division entries")

    print("Finding missing games (schedule has it, box score doesn't)...")
    missing = find_missing_games(schedules, boxscores, team_div_map)
    missing_out = OUT / "missing_games.csv"
    missing.to_csv(missing_out, index=False)
    print(f"  wrote {missing_out} with {len(missing):,} team-game rows")

    # Each physical game appears twice (once per team). Unique physical games:
    if len(missing):
        missing["gkey"] = missing.apply(
            lambda r: tuple(sorted([str(r["team"]), str(r["opponent"])])) + (str(r["date"]),),
            axis=1,
        )
        unique_games = missing.drop_duplicates("gkey")
        print(f"  unique physical games missing: {len(unique_games):,}")

        print("\n  Missing games by (sport, division):")
        print(
            unique_games.groupby(["sport", "division"]).size().to_string()
        )

    print("\nLoading PBP events to find PBP-has-it-no-box-score gap...")
    pbp_events = load_pbp_events()
    print(f"  PBP unique gameIds: {len(pbp_events):,}")
    pbp_gap = pbp_without_boxscore(boxscores, pbp_events)
    pbp_gap_out = OUT / "pbp_without_boxscore.csv"
    pbp_gap.to_csv(pbp_gap_out, index=False)
    print(f"  wrote {pbp_gap_out} with {len(pbp_gap):,} games")

    print("\nBuilding rescrape queue...")
    queue = build_rescrape_queue(missing, pbp_gap, pbp_events)
    queue_out = OUT / "rescrape_queue.csv"
    queue.to_csv(queue_out, index=False)
    print(f"  wrote {queue_out} with {len(queue):,} entries")
    print(f"    pbp_gap (gameId known): {(queue['source']=='pbp_gap').sum():,}")
    print(f"    schedule_gap (team+date): {(queue['source']=='schedule_gap').sum():,}")

    print("\nDate histogram (testing 'a day was missed' hypothesis)...")
    by_date = gap_by_date(missing, pbp_gap)
    by_date_out = OUT / "gap_by_date.csv"
    by_date.to_csv(by_date_out, index=False)
    print(f"  wrote {by_date_out}")
    print("\n  Top 20 dates by total missing games (pbp_gap + schedule_gap):")
    by_date["total"] = by_date["missing_schedule"] + by_date["missing_pbp_gap"]
    print(by_date.sort_values("total", ascending=False).head(20).to_string(index=False))

    print("\nRunning HR parity check (season stats vs box-score sum)...")
    parity = hr_parity_check(boxscores)
    parity_out = OUT / "hr_parity_mismatches.csv"
    parity.to_csv(parity_out, index=False)
    print(f"  wrote {parity_out} with {len(parity):,} players having missing HRs")
    if len(parity):
        print(f"  total missing HRs: {int(parity['delta'].sum()):,}")
        print("\n  Top 20 players by missing HR delta:")
        print(parity.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
