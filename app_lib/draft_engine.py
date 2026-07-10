"""Draft Assistant compute engine — replicates the MLB Draft Database formulas.

Validated against the 615 picks of the 2025 draft in draft_history.csv
(Excel's own computed columns are the ground truth; see scripts/validate_draft_engine.py).

Model (from the workbook):
  - round code : pick 1-5 'S', 6-10 'T', 11-15 'U', 16-20 'V'; later picks bucket
                 by round via the Trends round ranges ('A'..'F').
  - age code   : 'G' < 18, then one letter per year of age up to 'N'.
  - dist code  : Overall-board-rank minus pick number, bucketed ('O'..'V' tail letters).
  - expected diff-from-slot = composite lookup (dist+round+age, e.g. 'QSK') from
    historical signings; fallback = sum of the three individual code rates.
  - expected signing = slot * (1 + expected diff)
  - pool impact: signed rounds 1-10 count in full; rounds 11+ count only the
    amount above $150K; unsigned = no impact.
"""
import json
from datetime import date
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / 'data' / 'draft'


def load_trends():
    return json.loads((DATA / 'draft_trends.json').read_text())


def age_years(dob, on=None):
    """Age in years like Excel's (date - dob)/365."""
    if dob is None:
        return None
    on = on or date.today()
    if isinstance(dob, str):
        try:
            dob = date.fromisoformat(dob[:10])
        except ValueError:
            return None
    return (on - dob).days / 365


def code_round(pick, rnd):
    """Empirically validated on 2021-2025 history: S/T/U/V = picks 1-5/6-10/11-15/16-20,
    A = rest of round 1, then B rounds 2-3, C 4-6, D 7-10, E 11-15, F 16+."""
    if pick is None:
        return ''
    if pick <= 5: return 'S'
    if pick <= 10: return 'T'
    if pick <= 15: return 'U'
    if pick <= 20: return 'V'
    r = rnd or 1
    if r <= 1: return 'A'
    if r <= 3: return 'B'
    if r <= 6: return 'C'
    if r <= 10: return 'D'
    if r <= 15: return 'E'
    return 'F'


def code_age(age):
    if age is None:
        return ''
    if age < 18: return 'G'
    for i, letter in enumerate('HIJKLMN'):
        if age < 19 + i:
            return letter
    return 'N'


def code_dist(dist):
    """dist = overall board rank minus pick number (negative = drafted above rank).
    Empirical: O <= -50, P -49..0, Q 1..50, R 51+."""
    if dist is None:
        return ''
    if dist <= -50: return 'O'
    if dist <= 0: return 'P'
    if dist <= 50: return 'Q'
    return 'R'


def expected_diff(dist_c, round_c, age_c, trends):
    comp = trends['composite'].get(f'{dist_c}{round_c}{age_c}')
    if comp is not None:
        return comp
    pct = trends['code_pct']
    parts = []
    for c in (dist_c, round_c, age_c):
        v = pct.get(c)
        if v is None and c in ('L', 'M'):
            v = pct.get('L&M')
        if v is not None:
            parts.append(v)
    return sum(parts) if parts else 0.0


def expected_signing(slot, dist_c, round_c, age_c, trends):
    if not slot:
        return None
    return slot * (1 + expected_diff(dist_c, round_c, age_c, trends))


def pool_impact(signed, pool_round, bonus):
    """Rounds 1-10 ('Yes'): full bonus counts. Rounds 11+: only the over-$150K excess."""
    if not signed or bonus in (None, ''):
        return None
    if pool_round:
        return bonus
    return max(bonus - 150000, 0)


def diff_from_slot(bonus, slot, pool_round):
    if bonus in (None, '') or not slot:
        return None
    if not pool_round:
        return (bonus - 130000) / 130000 if bonus > 130000 else None
    return (bonus - slot) / slot
