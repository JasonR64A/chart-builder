"""
Bubble Meter — bubble-grade prediction for NCAAT field

Two scoring methods exposed side-by-side so the user can see how a naive
average-of-marginals compares to a properly calibrated logistic regression.

  - NAIVE (user's proposal): for each metric, compute the historical
    "% of teams that got in given they meet the threshold" and average
    those probabilities. Intuitive, easy to narrate ("if you check 3 of
    these 4 boxes, you're in"), but the metrics are correlated so the
    composite probability is inflated.

  - LOGISTIC REGRESSION: fits coefficients via gradient descent on the
    log-likelihood. Honest about correlation between features. We use
    only numpy/scipy (no sklearn dependency).

Both methods are trained on tournament_master_2013_2025.csv with conference
info joined from historical_selection_rpi.csv (2021-2025 coverage; older
years default to non-power-conf which is the dominant class anyway).

The naive method intentionally mirrors the user's "RPI <40, SOS <50,
Q1+Q2 >=15, conf wins >=12" pattern — we use the metrics we have right
now (rpi_rank, win_pct, power_conf) and document where SOS / Q1/Q2 /
conf-wins plug in once the schedule-derived features are wired up.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


POWER_CONFS = {'SEC', 'ACC', 'Big Ten', 'Big 12', 'Pac-12'}


def _canonical_team(name: str) -> str:
    """Normalise team names for cross-file joins."""
    import re
    if not isinstance(name, str):
        return ''
    n = re.sub(r'\(AQ\)\s*$', '', name).strip()
    n = re.sub(r'\bState\b', 'St.', n)
    return n.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Historical data load (training corpus)
# ─────────────────────────────────────────────────────────────────────────────

def load_historical(bracket_dir: Path) -> pd.DataFrame:
    """
    Merge tournament_master_2013_2025.csv (full coverage, no conference)
    with historical_selection_rpi.csv (2021-2025 only, has conference)
    on (year, canonical team). Returns rows with the features and label
    needed to train both bubble-meter methods.
    """
    tm = pd.read_csv(bracket_dir / 'tournament_master_2013_2025.csv')
    sr = pd.read_csv(bracket_dir / 'historical_selection_rpi.csv')

    # Pre-clean tm
    tm = tm.dropna(subset=['rpi_rank', 'w', 'l']).copy()
    tm['win_pct'] = tm['w'] / (tm['w'] + tm['l']).replace(0, np.nan)
    tm['team_key'] = tm['team_canonical'].apply(_canonical_team)

    # Pre-clean sr (baseball only, has conference)
    sr_bb = sr[sr['sport'] == 'baseball'].copy()
    sr_bb['year_int'] = sr_bb['year'].apply(lambda s: int(str(s).split('-')[1]) + 2000)
    sr_bb['team_key'] = sr_bb['teamName'].apply(_canonical_team)
    conf_map = sr_bb.set_index(['year_int', 'team_key'])['conference'].to_dict()

    tm['conference'] = tm.apply(lambda r: conf_map.get((int(r['year']), r['team_key']), ''), axis=1)
    tm['power_conf'] = tm['conference'].isin(POWER_CONFS).astype(int)

    return tm[['year', 'team_canonical', 'rpi_rank', 'rpi_value', 'w', 'l',
               'win_pct', 'conference', 'power_conf', 'in_tournament']].copy()


# ─────────────────────────────────────────────────────────────────────────────
# NAIVE method — user's proposal: per-metric thresholds, average % in
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NaiveRule:
    """One per-metric rule for the naive bubble grade."""
    name: str
    label: str           # human-readable, e.g. "RPI rank ≤ 40"
    column: str          # column in the dataframe
    direction: str       # 'le' (lower is better, e.g. rpi_rank) or 'ge'
    threshold: float
    historical_p_in: float = 0.0  # filled at fit time

    def passes(self, value) -> bool:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return False
        return value <= self.threshold if self.direction == 'le' else value >= self.threshold


def fit_naive_thresholds(df: pd.DataFrame) -> list[NaiveRule]:
    """
    For each rule we care about, compute the historical % of teams that
    met the threshold AND got in. That's the user's "9/10" number.

    Thresholds are pre-set to match the kind of boxes a committee actually
    discusses. We could grid-search for the optimal threshold per metric
    (highest precision while keeping recall up), but the explanation
    suffers — coaches want round numbers.
    """
    rules = [
        NaiveRule('rpi_top40', 'RPI rank ≤ 40', 'rpi_rank', 'le', 40),
        NaiveRule('rpi_top60', 'RPI rank ≤ 60', 'rpi_rank', 'le', 60),
        NaiveRule('winpct_65', 'Win pct ≥ .650', 'win_pct', 'ge', 0.650),
        NaiveRule('power_conf', 'Power conference', 'power_conf', 'ge', 1),
    ]
    for r in rules:
        mask = df[r.column].apply(lambda v: r.passes(v))
        sub = df[mask]
        r.historical_p_in = float(sub['in_tournament'].mean()) if len(sub) else 0.0
    return rules


def naive_grade(features: dict, rules: list[NaiveRule]) -> tuple[float, list[tuple[NaiveRule, bool]]]:
    """
    Apply each rule to the team's features. Returns:
      - composite grade (avg of historical_p_in for rules that passed)
      - per-rule pass/fail breakdown for the UI

    Rules that DON'T pass contribute 0 to the average (you can't claim
    the upside of a box you didn't check).
    """
    breakdown = []
    contribs = []
    for r in rules:
        passes = r.passes(features.get(r.column))
        breakdown.append((r, passes))
        contribs.append(r.historical_p_in if passes else 0.0)
    grade = float(np.mean(contribs)) if contribs else 0.0
    return grade, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# LOGISTIC REGRESSION — properly calibrated probability
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LogisticModel:
    feature_names: list[str]
    mean: np.ndarray         # for standardising
    std: np.ndarray
    coef: np.ndarray         # one per feature
    intercept: float
    accuracy: float          # on training set
    log_loss: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """x: (n_samples, n_features) raw values. Returns probabilities."""
        x_std = (x - self.mean) / np.where(self.std == 0, 1, self.std)
        z = x_std @ self.coef + self.intercept
        # numerically stable sigmoid
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def _logistic_fit(X: np.ndarray, y: np.ndarray, lr: float = 0.05,
                  n_iter: int = 4000, l2: float = 0.01) -> tuple[np.ndarray, float]:
    """Plain gradient-descent logistic regression. Returns (coef, intercept)."""
    n, d = X.shape
    coef = np.zeros(d)
    intercept = 0.0
    for _ in range(n_iter):
        z = X @ coef + intercept
        # stable sigmoid
        p = np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
        err = p - y
        grad_coef = (X.T @ err) / n + l2 * coef
        grad_intercept = float(err.mean())
        coef -= lr * grad_coef
        intercept -= lr * grad_intercept
    return coef, intercept


def fit_logistic(df: pd.DataFrame, feature_names: Optional[list[str]] = None) -> LogisticModel:
    """Fit logistic regression on the historical training corpus."""
    if feature_names is None:
        feature_names = ['rpi_rank', 'win_pct', 'power_conf']
    sub = df.dropna(subset=feature_names + ['in_tournament']).copy()
    X_raw = sub[feature_names].to_numpy(dtype=float)
    y = sub['in_tournament'].astype(int).to_numpy()

    mean = X_raw.mean(axis=0)
    std = X_raw.std(axis=0)
    X = (X_raw - mean) / np.where(std == 0, 1, std)

    coef, intercept = _logistic_fit(X, y)

    # Training metrics
    z = X @ coef + intercept
    p = np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
    pred = (p >= 0.5).astype(int)
    accuracy = float((pred == y).mean())
    eps = 1e-12
    log_loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    return LogisticModel(
        feature_names=feature_names,
        mean=mean, std=std,
        coef=coef, intercept=intercept,
        accuracy=accuracy, log_loss=log_loss,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Current-year ingest
# ─────────────────────────────────────────────────────────────────────────────

def load_current_year_teams(data_dir: Path, sport_key: str) -> pd.DataFrame:
    """
    Load the live RPI file for the current season and produce a row per
    team with the features needed to compute the bubble grade.
    """
    rpi_path = data_dir / f'{sport_key}_rpi_D1.csv'
    if not rpi_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(rpi_path)

    # Parse W-L from "48-6" or "48-6-1" record string
    def _parse_record(s: str) -> tuple[int, int]:
        if not isinstance(s, str):
            return 0, 0
        parts = s.split('-')
        try:
            return int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            return 0, 0

    rec = df['record'].apply(_parse_record)
    df['w'] = [t[0] for t in rec]
    df['l'] = [t[1] for t in rec]
    df['win_pct'] = df['w'] / (df['w'] + df['l']).replace(0, np.nan)
    df['rpi_rank'] = df['rank']
    df['power_conf'] = df['conference'].isin(POWER_CONFS).astype(int)
    df['team_canonical'] = df['teamName'].apply(_canonical_team)

    return df[['team_canonical', 'teamName', 'conference', 'record',
               'rpi_rank', 'rpi', 'w', 'l', 'win_pct', 'power_conf']].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Calibration check (how well does each method match reality?)
# ─────────────────────────────────────────────────────────────────────────────

def calibration_summary(df: pd.DataFrame, rules: list[NaiveRule],
                        model: LogisticModel) -> dict:
    """
    Validate both methods on the historical training set.

    For both: bucket teams by predicted probability, then compute actual
    selection rate per bucket. A well-calibrated model has actual rate ≈
    predicted rate in each bucket.

    Returns a dict with summary stats + buckets.
    """
    feat_cols = model.feature_names
    sub = df.dropna(subset=feat_cols + ['in_tournament']).copy()

    # Naive grades
    sub['naive'] = sub.apply(
        lambda r: naive_grade({c: r[c] for c in [x.column for x in rules]}, rules)[0],
        axis=1,
    )
    # Logistic grades
    X = sub[feat_cols].to_numpy(dtype=float)
    sub['logreg'] = model.predict_proba(X)

    # Bucket and compare actual vs predicted
    def _buckets(col: str) -> pd.DataFrame:
        edges = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.01]
        sub['_bucket'] = pd.cut(sub[col], edges, include_lowest=True, right=False)
        b = sub.groupby('_bucket', observed=True).agg(
            n=('in_tournament', 'size'),
            actual=('in_tournament', 'mean'),
            pred=(col, 'mean'),
        ).reset_index()
        b['method'] = col
        return b

    naive_buckets = _buckets('naive')
    logreg_buckets = _buckets('logreg')

    return {
        'naive_buckets': naive_buckets,
        'logreg_buckets': logreg_buckets,
        'n_train': len(sub),
        'pos_rate': float(sub['in_tournament'].mean()),
        'logreg_accuracy': model.accuracy,
        'logreg_log_loss': model.log_loss,
    }
