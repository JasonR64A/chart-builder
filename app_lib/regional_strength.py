"""Blended Bradley-Terry team strength for the Regional Preview page.

strength(team) = exp(k * Z)
    Z        = CR_WEIGHT * z_CR  +  (1-CR_WEIGHT) * z_roster
    z_CR     = league z-score of Computer Rank  (mean of 64A / RPI / Massey / DSR
               ranks; lower rank = stronger -> sign flipped so higher z = better)
    z_roster = league z-score of (z_offense + z_pitching)
        offense  = sum of the team's top-9 hitters' wRAA
                   (weighted_runs_above_average, from hitting.csv)
        pitching = role-weighted FIP of the top-12 pitchers by IP
                   (from pitching.csv; lower FIP = better -> sign flipped)
    k        = calibrated so the league spread of log-strengths matches the legacy
               RPI-rank model, i.e. the new model is no more/less decisive overall.

Standardized across every D1 team that has all three inputs, so it works for any
4-team selection. Home-field advantage is applied by the caller (host +HFA/game).

Bradley-Terry win prob:  P(a beats b) = s_a / (s_a + s_b).
"""
from __future__ import annotations
import functools
import json
import math
from pathlib import Path
from statistics import NormalDist
import numpy as np
import pandas as pd

_ND = NormalDist()
_SQRT2 = math.sqrt(2.0)
# Calibrated 2026-05-27 so avg host regional win ~55% (real-world national-seed rate).
K_DECISIVENESS = 0.78
HFA = 0.04          # host single-game win-prob bump
_SQRT = _SQRT2

DATA = Path(__file__).resolve().parent.parent / 'data'
CURRENT_YEAR = 2026
CR_WEIGHT = 0.65
LEAGUE_AVG_FIP = 4.5
PA_GATE = 50          # min PA for a hitter to count toward the top-9 lineup wRAA
MIN_IP = 1.0


def _ip_to_outs(v):
    try:
        v = float(v); w = int(v); return w * 3 + round((v - w) * 10)
    except (TypeError, ValueError):
        return 0


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd else s * 0.0


@functools.lru_cache(maxsize=8)
def _components(sport: str, division: str) -> pd.DataFrame:
    """Per-team CR / top-9 wRAA / weighted FIP + legacy RPI rank. Indexed by name."""
    teams = pd.read_csv(DATA / 'teams.csv', low_memory=False)
    teams['id'] = teams['id'].astype(str)
    tb = teams[teams['sport'].str.lower() == sport.lower()][['id', 'name']]

    # 64A rank
    tr = pd.read_csv(DATA / 'team_rank.csv', low_memory=False)
    tr = tr[tr['year'] == CURRENT_YEAR].copy()
    tr['team_id'] = tr['team_id'].astype(str)
    tr['rank_64a'] = pd.to_numeric(tr['integer_64_rank_total'], errors='coerce')
    m = tb.merge(tr[['team_id', 'rank_64a']], left_on='id', right_on='team_id', how='left')

    # RPI rank
    try:
        rpi = pd.read_csv(DATA / f'{sport}_rpi_{division}.csv')
        rcol = next((c for c in rpi.columns if 'rpi' in c.lower() and 'rank' in c.lower()), None) \
            or next((c for c in rpi.columns if c.lower() == 'rank'), None)
        ncol = next((c for c in rpi.columns if c.lower() in ('teamname', 'team', 'name')), None)
        rpimap = dict(zip(rpi[ncol], pd.to_numeric(rpi[rcol], errors='coerce')))
        m['rank_rpi'] = m['name'].map(rpimap)
    except Exception:
        m['rank_rpi'] = np.nan

    # Massey / DSR rank (via optional external name map)
    ext = {}
    nm_path = DATA / 'rankings' / 'name_map.csv'
    if nm_path.exists():
        nm = pd.read_csv(nm_path); ext = dict(zip(nm['external_name'], nm['our_name']))
    mpf = lambda x: ext.get(x, x)
    for src, col in [('massey', 'rank_massey'), ('dsr', 'rank_dsr')]:
        p = DATA / 'rankings' / f'{src}_{sport}.csv'
        if p.exists():
            d = pd.read_csv(p)
            m[col] = m['name'].map({mpf(t): r for t, r in zip(d['team'], d['rank'])})
        else:
            m[col] = np.nan

    rank_cols = ['rank_64a', 'rank_rpi', 'rank_massey', 'rank_dsr']
    m['CR'] = m[rank_cols].mean(axis=1)
    # normalized int-string team id for joining stat tables (their ids vary float/int)
    m['_id'] = pd.to_numeric(m['id'], errors='coerce').astype('Int64').astype(str)
    _norm = lambda idx: pd.Index(pd.to_numeric(idx, errors='coerce')).astype('Int64').astype(str)

    # Offense: top-9 wRAA from hitting.csv
    hit = pd.read_csv(DATA / 'hitting.csv', low_memory=False)
    hit = hit[hit['year'] == CURRENT_YEAR].copy()
    hit['pa'] = pd.to_numeric(hit['plate_appearances'], errors='coerce')
    hit['wraa'] = pd.to_numeric(hit['weighted_runs_above_average'], errors='coerce')
    hit = hit[(hit['pa'] >= PA_GATE)].dropna(subset=['wraa'])
    off9 = (hit.sort_values('wraa', ascending=False)
              .groupby('team_id')['wraa'].apply(lambda s: s.head(9).sum()))
    off9.index = _norm(off9.index)
    m['off9'] = m['_id'].map(off9)

    # Pitching: role-weighted FIP from pitching.csv
    pit = pd.read_csv(DATA / 'pitching.csv', low_memory=False)
    pit = pit[pit['year'] == CURRENT_YEAR].copy()
    for c in ['homeruns_allowed', 'walks_issued', 'hit_batter', 'strikeouts']:
        pit[c] = pd.to_numeric(pit[c], errors='coerce').fillna(0)
    pit['_outs'] = pit['innings_pitched'].apply(_ip_to_outs)
    gp = pit.groupby(['team_id', 'player_id']).agg(
        outs=('_outs', 'sum'), HR=('homeruns_allowed', 'sum'),
        BB=('walks_issued', 'sum'), HBP=('hit_batter', 'sum'),
        SO=('strikeouts', 'sum')).reset_index()
    gp['IP'] = gp['outs'] / 3.0
    gp = gp[gp['IP'] >= MIN_IP]
    gp['FIP'] = ((13 * gp['HR'] + 3 * (gp['BB'] + gp['HBP']) - 2 * gp['SO']) / gp['IP']) + 3.0
    sw = [0.15, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10]

    def _team_fip(g):
        fips = g.sort_values('IP', ascending=False)['FIP'].tolist()
        f7 = (fips + [LEAGUE_AVG_FIP] * 7)[:7]
        wsum = sum(w * f for w, f in zip(sw, f7))
        rest = fips[7:12]
        ra = sum(rest) / len(rest) if rest else LEAGUE_AVG_FIP
        return wsum + 0.15 * ra

    fip = gp.groupby('team_id').apply(_team_fip, include_groups=False)
    fip.index = _norm(fip.index)
    m['fip'] = m['_id'].map(fip)
    return m.set_index('name')


@functools.lru_cache(maxsize=8)
def team_strengths(sport: str = 'baseball', division: str = 'D1',
                   cr_weight: float = CR_WEIGHT) -> dict:
    """Return {team_name: bradley_terry_strength} for all teams.

    Strength is positive; only the ratio matters for win prob. Teams missing
    roster data fall back to the Computer-Rank component alone; teams missing
    everything get a small floor strength.
    """
    m = _components(sport, division).copy()
    full = m.dropna(subset=['CR', 'off9', 'fip'])          # league standardization pool
    if full.empty:
        return {}
    z_cr = _zscore(-full['CR'])                            # lower CR = better
    z_off = _zscore(full['off9'])
    z_pit = _zscore(-full['fip'])                          # lower FIP = better
    z_roster = _zscore(z_off + z_pit)
    Z = cr_weight * z_cr + (1 - cr_weight) * z_roster

    # Calibrate k so the log-strength spread matches the legacy RPI-rank x FIP
    # model AMONG tournament-caliber teams (top ~64 by CR). Calibrating across all
    # of D1 would leave the strong teams bunched and the bracket far too flat.
    ref = full.nsmallest(min(64, len(full)), 'CR')
    rpir = ref['rank_rpi'].clip(lower=1.0)
    fipmult = (1.0 + (LEAGUE_AVG_FIP - ref['fip']) * 0.30).clip(0.30, 2.50)
    old_log = np.log((1.0 / np.sqrt(rpir)) * fipmult)
    z_ref = Z.loc[ref.index]
    k = old_log.std(ddof=0) / z_ref.std(ddof=0) if z_ref.std(ddof=0) else 1.0

    # League params for fallback teams (CR present but no roster data)
    cr_mean, cr_sd = full['CR'].mean(), full['CR'].std(ddof=0) or 1.0

    out = {}
    for name, row in m.iterrows():
        if name in Z.index:
            out[name] = float(np.exp(k * Z.loc[name]))
        elif pd.notna(row['CR']):
            z = cr_weight * (-(row['CR'] - cr_mean) / cr_sd)   # CR-only fallback
            out[name] = float(np.exp(k * z))
        else:
            out[name] = float(np.exp(k * (-2.0)))              # floor: ~2 sd below
    return out


# ── Per-game weekend-rotation model ────────────────────────────────────────
# strength(team, game n) = exp(K * Z), Z = CR_WEIGHT*z_cr + (1-CR_WEIGHT)*z_roster
#   z_roster = (z_off + z_pit_n) / sqrt(2);  z_pit_n = inv_normal(per-game pctile)
# Per-game pitching pctiles come from the baked weekend-rotation lookup.
_ROT = None


def _rotations():
    global _ROT
    if _ROT is None:
        p = DATA / 'regional_rotations_baseball_2026.json'
        _ROT = {int(k): v for k, v in json.load(open(p)).items()} if p.exists() else {}
    return _ROT


@functools.lru_cache(maxsize=8)
def _zparams(sport, division):
    m = _components(sport, division)
    full = m.dropna(subset=['CR', 'off9'])
    return (m, float(full['CR'].mean()), float(full['CR'].std(ddof=0)),
            float(full['off9'].mean()), float(full['off9'].std(ddof=0)))


def _cr_off(m, crm, crs, om, osd, name):
    cr = (crm - m.loc[name, 'CR']) / crs if name in m.index and pd.notna(m.loc[name, 'CR']) else 0.0
    of = (m.loc[name, 'off9'] - om) / osd if name in m.index and pd.notna(m.loc[name, 'off9']) else 0.0
    return float(cr), float(of)


def _tid(m, name):
    if name in m.index and pd.notna(m.loc[name, 'id']):
        try: return int(m.loc[name, 'id'])
        except Exception: return None
    return None


def _strength_array(cr, off, tid, model, rots, cr_weight=CR_WEIGHT):
    """Bradley-Terry strength for a team's 1st..5th game of the regional."""
    r = rots.get(int(tid)) if tid is not None else None
    out = []
    for n in range(1, 6):
        pit = r[model][f'g{min(n, 4)}'] if r else 0.5
        zp = _ND.inv_cdf(min(0.98, max(0.02, pit)))
        zr = (off + zp) / _SQRT2
        out.append(math.exp(K_DECISIVENESS * (cr_weight * cr + (1 - cr_weight) * zr)))
    return out


def team_game_strengths(name, sport='baseball', division='D1', model='model1'):
    m, crm, crs, om, osd = _zparams(sport, division)
    cr, off = _cr_off(m, crm, crs, om, osd, name)
    return _strength_array(cr, off, _tid(m, name), model, _rotations())


def rotation_starters(name, sport='baseball', division='D1'):
    """[(sp1_name, pctile), (sp2_name, pctile), (sp3_name, pctile)] ace-first, or None."""
    m, *_ = _zparams(sport, division)
    r = _rotations().get(_tid(m, name) or -1)
    return [(s['name'], s['pit']) for s in r['sp']] if r else None


def simulate_regional_milestones(team_names, sport='baseball', division='D1',
                                 host_model='model1', host_idx=0, n=20000, seed=42):
    """Per-game double-elim sim. teams ordered [1seed,2seed,3seed,4seed].
    Returns {name: {'r1','r2','r3','r4'}} (won opener / won W-bracket / reached
    final / champ). Host (host_idx) gets HFA and may use 'model2' (save ace)."""
    m, crm, crs, om, osd = _zparams(sport, division)
    rots = _rotations()
    S = []
    for i, nm in enumerate(team_names):
        cr, off = _cr_off(m, crm, crs, om, osd, nm)
        mdl = host_model if i == host_idx else 'model1'
        S.append(_strength_array(cr, off, _tid(m, nm), mdl, rots))
    rng = np.random.default_rng(seed); rnd = rng.random
    cnt = [{'r1': 0, 'r2': 0, 'r3': 0, 'r4': 0} for _ in team_names]
    for _ in range(n):
        g = [0, 0, 0, 0]
        def play(a, b):
            sa, sb = S[a][g[a]], S[b][g[b]]; p = sa / (sa + sb)
            if a == host_idx: p = min(.97, max(.03, p + HFA))
            elif b == host_idx: p = min(.97, max(.03, p - HFA))
            g[a] += 1; g[b] += 1
            return (a, b) if rnd() < p else (b, a)
        w1, l1 = play(0, 3); w2, l2 = play(1, 2)
        cnt[w1]['r1'] += 1; cnt[w2]['r1'] += 1
        w3, _ = play(l1, l2); w4, l4 = play(w1, w2); cnt[w4]['r2'] += 1
        w5, _ = play(l4, w3); cnt[w4]['r3'] += 1; cnt[w5]['r3'] += 1
        g6, _ = play(w4, w5)
        champ = w4 if g6 == w4 else play(w4, w5)[0]
        cnt[champ]['r4'] += 1
    return {team_names[i]: {k: cnt[i][k] / n for k in cnt[i]} for i in range(len(team_names))}


def g1_ace_vs_n2(team_names, sport='baseball', division='D1', host_idx=0):
    """Host's Game-1 (vs the 4-seed) win prob throwing the ace vs throwing #2.
    Returns {'ace':p, 'n2':p, 'ace_name':, 'n2_name':, 'opp':, 'opp_ace':}."""
    m, crm, crs, om, osd = _zparams(sport, division); rots = _rotations()
    host = team_names[host_idx]; opp = team_names[3 if host_idx == 0 else 0]
    hc, ho = _cr_off(m, crm, crs, om, osd, host); oc, oo = _cr_off(m, crm, crs, om, osd, opp)
    h1 = _strength_array(hc, ho, _tid(m, host), 'model1', rots)[0]
    h2 = _strength_array(hc, ho, _tid(m, host), 'model2', rots)[0]
    o1 = _strength_array(oc, oo, _tid(m, opp), 'model1', rots)[0]
    bt = lambda a, b: min(.97, max(.03, a / (a + b) + HFA))   # host at home
    sp = rotation_starters(host, sport, division) or [('SP1', 0), ('SP2', 0)]
    osp = rotation_starters(opp, sport, division)
    return {'ace': bt(h1, o1), 'n2': bt(h2, o1),
            'ace_name': sp[0][0], 'n2_name': sp[1][0] if len(sp) > 1 else sp[0][0],
            'opp': opp, 'opp_ace': osp[0][0] if osp else 'SP1'}
