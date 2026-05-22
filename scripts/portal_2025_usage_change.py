"""
Portal 2025 entrants: usage change (AB / IP) from 2025 to 2026.

For players who entered the portal during the 2025 cycle and have 2026 playing
data (i.e., landed somewhere), compare their at_bats (hitters) and
innings_pitched (pitchers) between seasons.

Reports TWO views:
  1. Raw totals  — 2025 full season vs 2026 season-to-date (partial)
  2. Pace-normalized — AB/game for hitters, IP/appearance for pitchers.
     This normalizes for the 2026 season being ~75-80% complete.

Usage: python scripts/portal_2025_usage_change.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"


def load_data():
    portal = pd.read_csv(DATA / "portal_rank_player.csv", low_memory=False)
    hit = pd.read_csv(DATA / "hitting.csv", low_memory=False)
    pit = pd.read_csv(DATA / "pitching.csv", low_memory=False)
    return portal, hit, pit


# ---------- Raw-totals analysis ----------

def build_totals(df, value_col, player_ids, out_a, out_b):
    def agg(y, col):
        return (
            df[(df["year"] == y) & (df["player_id"].isin(player_ids))]
            .groupby("player_id", as_index=False)[value_col]
            .sum()
            .rename(columns={value_col: col})
        )
    merged = agg(2025, out_a).merge(agg(2026, out_b), on="player_id", how="inner")
    merged = merged[merged[out_a] > 0]
    merged["pct_change"] = (merged[out_b] - merged[out_a]) / merged[out_a] * 100
    return merged


# ---------- Pace-normalized analysis ----------

def build_rates(df, value_col, games_col, player_ids,
                min_games_25, min_games_26, rate_name):
    sub = df[df["player_id"].isin(player_ids) & df["year"].isin([2025, 2026])]
    agg = sub.groupby(["player_id", "year"], as_index=False).agg(
        v=(value_col, "sum"), g=(games_col, "sum")
    )
    y25 = agg[agg["year"] == 2025].rename(columns={"v": "v25", "g": "g25"})[
        ["player_id", "v25", "g25"]
    ]
    y26 = agg[agg["year"] == 2026].rename(columns={"v": "v26", "g": "g26"})[
        ["player_id", "v26", "g26"]
    ]
    m = y25.merge(y26, on="player_id", how="inner")
    m = m[(m["g25"] >= min_games_25) & (m["g26"] >= min_games_26) & (m["v25"] > 0)]
    m[f"{rate_name}_25"] = m["v25"] / m["g25"]
    m[f"{rate_name}_26"] = m["v26"] / m["g26"]
    m["pct_change"] = (
        (m[f"{rate_name}_26"] - m[f"{rate_name}_25"]) / m[f"{rate_name}_25"] * 100
    )
    return m


# ---------- Reporting helpers ----------

def report_buckets(df):
    buckets = pd.cut(
        df["pct_change"],
        bins=[-np.inf, -50, -25, -0.001, 25, 50, np.inf],
        labels=["<-50%", "-50..-25%", "-25..0%", "0..25%", "25..50%", ">50%"],
    )
    for lbl, n in buckets.value_counts().sort_index().items():
        print(f"    {str(lbl):>10}: {n:>5,d}  ({n/len(df)*100:.1f}%)")


def report_totals(label, df, col_a, col_b, unit):
    total_a = df[col_a].sum()
    total_b = df[col_b].sum()
    agg_pct = (total_b / total_a - 1) * 100 if total_a else float("nan")
    q = df["pct_change"].quantile([0.10, 0.25, 0.50, 0.75, 0.90])

    print(f"\n{label} (n = {len(df):,})")
    print(f"  total 2025 {unit:>3} = {total_a:>12,.1f}")
    print(f"  total 2026 {unit:>3} = {total_b:>12,.1f}")
    print(f"  aggregate change  = {agg_pct:+.1f}%")
    print(f"  median per-player = {df['pct_change'].median():+.1f}%")
    print(f"  mean per-player   = {df['pct_change'].mean():+.1f}%   "
          f"(distorted by small denominators; prefer median)")
    print(f"  p10/p25/p50/p75/p90 = "
          f"{q[0.10]:+.0f}% / {q[0.25]:+.0f}% / {q[0.50]:+.0f}% / "
          f"{q[0.75]:+.0f}% / {q[0.90]:+.0f}%")
    print("  bucket distribution:")
    report_buckets(df)


def report_rates(label, df, rate_name, unit_label):
    pooled_25 = df["v25"].sum() / df["g25"].sum()
    pooled_26 = df["v26"].sum() / df["g26"].sum()
    q = df["pct_change"].quantile([0.10, 0.25, 0.50, 0.75, 0.90])

    print(f"\n{label} (n = {len(df):,})")
    print(f"  pooled 2025 {unit_label} = {pooled_25:.3f}")
    print(f"  pooled 2026 {unit_label} = {pooled_26:.3f}")
    print(f"  aggregate rate change = {(pooled_26/pooled_25-1)*100:+.1f}%")
    print(f"  median per-player     = {df['pct_change'].median():+.1f}%")
    print(f"  p10/p25/p50/p75/p90   = "
          f"{q[0.10]:+.0f}% / {q[0.25]:+.0f}% / {q[0.50]:+.0f}% / "
          f"{q[0.75]:+.0f}% / {q[0.90]:+.0f}%")
    print("  bucket distribution:")
    report_buckets(df)


# ---------- Main ----------

def main():
    portal, hit, pit = load_data()
    p2025 = set(portal[portal["year"] == 2025]["player_id"].dropna().astype(int))
    print(f"2025 portal entrants: {len(p2025):,}")

    print("\n" + "=" * 60)
    print("PART 1 - RAW TOTALS  (2025 full season vs 2026 YTD)")
    print("=" * 60)
    hit_totals = build_totals(hit, "at_bats", p2025, "ab_2025", "ab_2026")
    pit_totals = build_totals(pit, "innings_pitched", p2025, "ip_2025", "ip_2026")
    report_totals("HITTERS - AB 2025 vs 2026",
                  hit_totals, "ab_2025", "ab_2026", "AB")
    report_totals("PITCHERS - IP 2025 vs 2026",
                  pit_totals, "ip_2025", "ip_2026", "IP")

    print("\n" + "=" * 60)
    print("PART 2 - PACE-NORMALIZED  (rate per game)")
    print("=" * 60)
    hit_rates = build_rates(
        hit, "at_bats", "games_played", p2025,
        min_games_25=5, min_games_26=5, rate_name="ab_per_g",
    )
    pit_rates = build_rates(
        pit, "innings_pitched", "games_appeared", p2025,
        min_games_25=3, min_games_26=3, rate_name="ip_per_app",
    )
    report_rates("HITTERS - AB per game (>=5 GP both years)",
                 hit_rates, "ab_per_g", "AB/G   ")
    report_rates("PITCHERS - IP per appearance (>=3 GA both years)",
                 pit_rates, "ip_per_app", "IP/App ")

    print(
        "\nCAVEAT: 2026 is a PARTIAL season (~75-80% through as of late April). "
        "Part 1 totals under-represent final-season usage; Part 2 pace rates "
        "are the apples-to-apples view."
    )


if __name__ == "__main__":
    main()
