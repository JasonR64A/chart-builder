"""Regionals Top Pitchers — editorial layout for the Top Pitchers tab in
Regional Preview. Mirrors `regionals_top_hitters.py` with pitcher-specific
data + metrics:

  Stats grid: ERA / WHIP / K% / BB% / IP / K / Opp OPS / FIP
  Splits:    Opp AVG and Opp OPS vs LHB, vs RHB, w/ RISP, 1st PA, Late
  Pace:      Cumulative ERA over the last 30 days
  Spray:     batted-ball zones AGAINST the pitcher (perspective='pitching')
  Scatter 1: K% × BB% — best is upper-right (high K%, low BB%) via invert_y
  Scatter 2: FIP × WHIP — both lower=better, invert_x + invert_y so
             best lands upper-right
  Hits-allowed donut: 1B / 2B / 3B / HR allowed

K%/BB% replaces the older K/9 + BB/9 so the metric is innings-length-agnostic
(softball is 7-inning games, baseball is 9 — rate-per-batter is uniform).

Top-4 pitcher selection uses player_rank.csv weighted_run_allowed_efficiency
(lower = better; pool restricted to current-year pitchers with IP >= 30).
"""
from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path

import pandas as pd

# Reuse the formatters / SVG renderers from the hitters module — the only
# stat-specific bits live in this file.
from app_lib.regionals_top_hitters import (
    _xe, _initials, _team_short, _format_height, _suffix,
    _rank_badge, _pct_bar, _donut_svg, _render_pace_svg,
    _render_scatter_svg, _STYLES,
)

CURRENT_YEAR = 2026
MIN_IP = 30  # qualifier gate

STAT_KEYS = ['ERA', 'WHIP', 'K%', 'BB%', 'IP', 'K', 'Opp OPS', 'FIP']
# Map design-stat-name → chart-builder pitching.csv column. Special cases
# noted inline.
_STAT_COL = {
    'ERA':     'earned_run_average',
    'WHIP':    'walks_plus_hits_per_inning_pitched',
    'K%':      'strikeout_percentage',
    'BB%':     'walk_percentage',
    'IP':      'innings_pitched',
    'K':       'strikeouts',
    'Opp OPS': 'on_base_plus_slugging_against',
    'FIP':     'fielding_independent_pitching',
}
# Lower-is-better stats — rank ASC for D1 ranks
_LOWER_IS_BETTER = {'ERA', 'WHIP', 'BB%', 'FIP', 'Opp OPS'}

# NCAA scorebook codes — same reference set as the hitters module so the
# hits-allowed donut + spray work cleanly.
_HIT_CODES_TB = {'1B': 1, '2B': 2, '3B': 3, 'HR': 4}


# ── Data: pitcher selection + ranks ─────────────────────────────────────────
def select_top_pitchers(pitching_df: pd.DataFrame, players_df: pd.DataFrame,
                         team_ids: list[int], top_n: int = 4,
                         player_rank_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pick top-N pitchers across the given team_ids by wRAE (asc, lower=better).
    Falls back to ERA asc if player_rank_df not provided. PA gate replaced by
    IP gate (>= MIN_IP)."""
    p_pool = pitching_df[(pitching_df['year'] == CURRENT_YEAR) &
                          (pitching_df['team_id'].isin(team_ids)) &
                          (pitching_df['innings_pitched'] >= MIN_IP)].copy()
    if p_pool.empty:
        return p_pool

    if player_rank_df is not None and not player_rank_df.empty:
        pr = player_rank_df[(player_rank_df['year'] == CURRENT_YEAR) &
                             (player_rank_df['team_id'].isin(team_ids))].copy()
        pr['_wrae'] = pd.to_numeric(pr['weighted_run_allowed_efficiency'], errors='coerce')
        # Real pitchers only — wRAE percentile > 0
        pr['_pct'] = pd.to_numeric(pr['percentile_rank_weighted_run_allowed_efficiency'], errors='coerce')
        pr = pr[(pr['_pct'] > 0) & pr['_wrae'].notna()]
        pool_pids = set(p_pool['player_id'].astype(int))
        pr = pr[pr['player_id'].astype(int).isin(pool_pids)]
        if not pr.empty:
            top_pr = pr.sort_values('_wrae', ascending=True).head(top_n)
            picked_pids = top_pr['player_id'].astype(int).tolist()
            h = p_pool[p_pool['player_id'].astype(int).isin(picked_pids)].copy()
            h = h.merge(top_pr[['player_id', '_wrae',
                                 'percentile_rank_weighted_run_allowed_efficiency',
                                 'integer_rank_weighted_run_allowed_efficiency']],
                        on='player_id', how='left')
            h = h.set_index('player_id').loc[picked_pids].reset_index()
            p = players_df[['id', 'player_name', 'position', 'classification',
                            'height', 'bat', 'throw']].rename(columns={'id': 'player_id'})
            h = h.merge(p, on='player_id', how='left')
            return h.reset_index(drop=True)

    # Fallback to ERA
    h = p_pool.sort_values('earned_run_average', ascending=True).head(top_n)
    p = players_df[['id', 'player_name', 'position', 'classification',
                    'height', 'bat', 'throw']].rename(columns={'id': 'player_id'})
    h = h.merge(p, on='player_id', how='left')
    return h.reset_index(drop=True)


def build_division_pitcher_pool(pitching_df: pd.DataFrame, teams_df: pd.DataFrame,
                                 sport: str, division: str,
                                 conferences_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """All current-year qualified pitchers in the same sport/division."""
    sport_label = 'Baseball' if sport == 'baseball' else 'Softball'
    div_full = {'D1': 'D-I', 'D2': 'D-II', 'D3': 'D-III'}[division]
    sport_teams = teams_df[teams_df['sport'] == sport_label]
    if conferences_df is not None:
        div_conf_ids = set(conferences_df[conferences_df['division'] == div_full]['id']
                            .astype(int).tolist())
        conf_id_num = pd.to_numeric(sport_teams['conference_id'], errors='coerce')
        sport_teams = sport_teams[conf_id_num.isin(div_conf_ids)]
    pool = pitching_df[(pitching_df['year'] == CURRENT_YEAR) &
                        (pitching_df['team_id'].isin(sport_teams['id'])) &
                        (pitching_df['innings_pitched'] >= MIN_IP)].copy()
    return pool


def compute_pitcher_ranks(player_row: pd.Series, pool: pd.DataFrame) -> dict:
    """{stat_key: (value, rank, percentile)}. Lower-is-better stats rank ASC."""
    out = {}

    for k in STAT_KEYS:
        col = _STAT_COL[k]
        v = pd.to_numeric(player_row.get(col), errors='coerce')
        col_vals = pd.to_numeric(pool[col], errors='coerce') if col in pool.columns else None

        if v is None or pd.isna(v) or col_vals is None:
            out[k] = (None, None, None)
            continue

        valid = col_vals.dropna()
        if valid.empty:
            out[k] = (v, None, None)
            continue
        if k in _LOWER_IS_BETTER:
            rank = int((valid < v).sum()) + 1
        else:
            rank = int((valid > v).sum()) + 1
        n_valid = len(valid)
        pct = 100 * (1 - (rank - 1) / max(n_valid - 1, 1))
        out[k] = (v, rank, pct)
    return out


def hits_allowed(player_row: pd.Series) -> dict:
    """{1B, 2B, 3B, HR} given up — derived from pitching.csv columns."""
    h = int(player_row.get('hits_allowed', 0) or 0)
    d = int(player_row.get('doubles_allowed', 0) or 0)
    t = int(player_row.get('triples_allowed', 0) or 0)
    hr = int(player_row.get('homeruns_allowed', 0) or 0)
    singles = max(0, h - d - t - hr)
    return {'1B': singles, '2B': d, '3B': t, 'HR': hr}


def build_division_pitcher_scatter(pool: pd.DataFrame) -> list[tuple[float, float]]:
    """(K%, BB%) tuples for every qualifier in the division pool. Uniform
    across baseball (9-inning) and softball (7-inning) — rate-per-batter."""
    if pool.empty:
        return []
    kp = pd.to_numeric(pool['strikeout_percentage'], errors='coerce')
    bbp = pd.to_numeric(pool['walk_percentage'], errors='coerce')
    valid = kp.notna() & bbp.notna()
    return list(zip(kp[valid].tolist(), bbp[valid].tolist()))


def build_division_pitcher_scatter_fw(pool: pd.DataFrame) -> list[tuple[float, float]]:
    """(FIP, WHIP) tuples for every qualifier — both lower-is-better."""
    if pool.empty:
        return []
    fip = pd.to_numeric(pool['fielding_independent_pitching'], errors='coerce')
    whip = pd.to_numeric(pool['walks_plus_hits_per_inning_pitched'], errors='coerce')
    valid = fip.notna() & whip.notna()
    return list(zip(fip[valid].tolist(), whip[valid].tolist()))


# ── Splits + pace from events PBP (perspective: pitching) ───────────────────
_AB_OUT_CODES = {'GO', 'FO', 'PO', 'LO', 'DP', 'TP', 'FC', 'E',
                 'K', 'KL', 'KS'}


_AB_NON_CODES = {'BB', 'IBB', 'HBP', 'SF', 'SH'}


def _opp_split_from_subset(subset: pd.DataFrame) -> dict:
    """Return {'avg', 'ops', 'ab', 'pa'} for the pitcher's split.
    OPS allowed = OBP allowed + SLG allowed. AVG = H/AB; SLG = TB/AB; OBP =
    (H+BB+HBP)/(AB+BB+HBP+SF) — same definitions as the hitter side, applied
    to events the pitcher was on the mound for."""
    empty = {'avg': 0.0, 'ops': 0.0, 'ab': 0, 'pa': 0}
    if subset.empty:
        return empty
    counts = subset['playResult'].value_counts()
    hits = sum(counts.get(c, 0) for c in _HIT_CODES_TB.keys())
    tb = sum(counts.get(c, 0) * w for c, w in _HIT_CODES_TB.items())
    outs = sum(counts.get(c, 0) for c in _AB_OUT_CODES)
    bb = counts.get('BB', 0) + counts.get('IBB', 0)
    hbp = counts.get('HBP', 0)
    sf = counts.get('SF', 0)
    sh = counts.get('SH', 0)
    ab = hits + outs
    obp_denom = ab + bb + hbp + sf
    pa = ab + bb + hbp + sf + sh
    if ab == 0:
        return empty
    avg = hits / ab
    slg = tb / ab
    obp = (hits + bb + hbp) / obp_denom if obp_denom else 0.0
    return {
        'avg': round(avg, 3),
        'ops': round(obp + slg, 3),
        'ab':  int(ab),
        'pa':  int(pa),
    }


def compute_pitcher_splits(events_df: pd.DataFrame, ncaa_pitcher_id: int,
                            batter_bat_lookup: dict) -> dict:
    """For one pitcher, compute opp-AVG + opp-OPS per split.
    Keys: 'vs LHB', 'vs RHB', 'w/ RISP', '1st PA', 'Late'. Each value is
    {'avg', 'ops', 'ab', 'pa'}.
    'Late' = inning >= 7. '1st PA' = first time the batter faces this pitcher
    in the game (heuristic: first occurrence of (gameId, batter playerId)).
    """
    empty = {'avg': 0.0, 'ops': 0.0, 'ab': 0, 'pa': 0}
    keys = ('vs LHB', 'vs RHB', 'w/ RISP', '1st PA', 'Late')
    if events_df.empty or pd.isna(ncaa_pitcher_id):
        return {k: dict(empty) for k in keys}
    df = events_df[events_df['pitcherId'] == ncaa_pitcher_id].copy()
    if df.empty:
        return {k: dict(empty) for k in keys}

    bat_pid = pd.to_numeric(df['playerId'], errors='coerce').astype('Int64')
    df['batter_bat'] = bat_pid.map(batter_bat_lookup)

    # 1st PA per game per batter — first row in event order
    df = df.sort_values(['gameId', 'inning'])
    df['_pa_idx'] = df.groupby(['gameId', 'playerId']).cumcount()

    inning_num = pd.to_numeric(df['inning'], errors='coerce')

    return {
        'vs LHB':  _opp_split_from_subset(df[df['batter_bat'] == 'L']),
        'vs RHB':  _opp_split_from_subset(df[df['batter_bat'] == 'R']),
        'w/ RISP': _opp_split_from_subset(df[(df['runner2B'] == 1) | (df['runner3B'] == 1)]),
        '1st PA':  _opp_split_from_subset(df[df['_pa_idx'] == 0]),
        'Late':    _opp_split_from_subset(df[inning_num >= 7]),
    }


def compute_pitcher_pace(events_df: pd.DataFrame, ncaa_pitcher_id: int,
                          window_days: int = 30) -> list[dict]:
    """Cumulative WHIP over the last `window_days`. Returns list of
    {'game', 'date', 'whip'}. WHIP = (hits + walks) / IP, where IP =
    outs / 3 (each AB-out + K). Walks counted from playResult ∈ {BB, IBB}."""
    if events_df.empty or pd.isna(ncaa_pitcher_id):
        return []
    df = events_df[events_df['pitcherId'] == ncaa_pitcher_id].copy()
    if df.empty:
        return []

    df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date_dt']).sort_values(['date_dt', 'gameId'])
    if df.empty:
        return []

    pr = df['playResult']
    df['is_out']  = pr.isin(_AB_OUT_CODES).astype(int)
    df['is_hit']  = pr.isin(_HIT_CODES_TB.keys()).astype(int)
    df['is_walk'] = pr.isin({'BB', 'IBB'}).astype(int)

    by_game = df.groupby(['date_dt', 'gameId'], sort=True, as_index=False).agg(
        outs=('is_out', 'sum'), hits=('is_hit', 'sum'), walks=('is_walk', 'sum'),
    )
    by_game['c_outs']  = by_game['outs'].cumsum()
    by_game['c_hits']  = by_game['hits'].cumsum()
    by_game['c_walks'] = by_game['walks'].cumsum()

    if window_days and window_days > 0:
        cutoff = by_game['date_dt'].max() - pd.Timedelta(days=window_days)
    else:
        cutoff = None

    rows = []
    for i, r in by_game.iterrows():
        ip = r['c_outs'] / 3
        whip = ((r['c_hits'] + r['c_walks']) / ip) if ip > 0 else 0.0
        if cutoff is None or r['date_dt'] >= cutoff:
            rows.append({
                'game': i + 1,
                'date': r['date_dt'].strftime('%Y-%m-%d'),
                'whip': float(whip),
                # Reuse the hitters pace renderer which keys on 'ops' — alias
                # so we don't have to duplicate the SVG path here.
                'ops':  float(whip),
                'pa':   int(r['c_outs']),
            })
    return rows


# ── Format helpers ──────────────────────────────────────────────────────────
def _fmt_pitcher_stat(k: str, v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return '—'
    if k in ('ERA', 'WHIP', 'FIP'):
        return f'{float(v):.2f}'
    if k == 'Opp OPS':
        return f'{float(v):.3f}'
    if k in ('K%', 'BB%'):
        return f'{float(v) * 100:.1f}%'
    if k == 'IP':
        return f'{float(v):.1f}'
    if k == 'K':
        return f'{int(v)}'
    return str(v)


# ── Row HTML ────────────────────────────────────────────────────────────────
def _row_html(idx: int, p: dict, accent: str, total_qualifiers: int) -> str:
    rail = (
        f'<aside class="rth-rail">'
        f'<div class="rth-rail__rank" style="color:{accent};">{idx + 1:02d}</div>'
        f'<div class="rth-rail__lbl">RANK</div>'
        f'<div class="rth-rail__div"></div>'
        f'<div class="rth-rail__tag">TOP 4</div>'
        f'</aside>'
    )

    initials = _initials(p['name'])
    bio_parts = [
        f'<span><strong>{_xe(p["pos"])}</strong></span>',
        f'<span>{_xe(p.get("role", "Starter"))}</span>',
        f'<span>{_xe(p["yr"])}</span>',
        f'<span>B/T {_xe(p["bats"])}/{_xe(p["throws"])}</span>',
    ]
    ht_fmt = _format_height(p.get('ht'))
    if ht_fmt:
        bio_parts.append(f'<span>{_xe(ht_fmt)}</span>')

    if p.get('photo_b64'):
        pos_y = p.get('photo_pos_y', 20)
        headshot = (
            f'<div class="rth-headshot" style="--rth-accent:{accent};">'
            f'<div style="position:absolute;inset:0;width:100%;height:100%;'
            f'background-image:url(data:image/{p.get("photo_mime","jpeg")};base64,{p["photo_b64"]});'
            f'background-size:cover;background-position:50% {pos_y}%;"></div>'
            f'</div>'
        )
    else:
        headshot = (
            f'<div class="rth-headshot" style="--rth-accent:{accent};">'
            f'<svg viewBox="0 0 100 100" class="rth-headshot__bg" preserveAspectRatio="none">'
            f'<defs><pattern id="rth-stripe-p{idx}" width="6" height="6" '
            f'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<line x1="0" y1="0" x2="0" y2="6" stroke="{accent}" stroke-width="2.2" stroke-opacity="0.18"/>'
            f'</pattern></defs>'
            f'<rect width="100" height="100" fill="#f4efe7"/>'
            f'<rect width="100" height="100" fill="url(#rth-stripe-p{idx})"/>'
            f'</svg>'
            f'<div class="rth-headshot__init">{initials}</div>'
            f'</div>'
        )

    # Splits — opp AVG and opp OPS, both lower=better. Two stacked blocks
    # mirror the hitter view's SLG / OBP layout. Bars inverted: longer = lower
    # opponent production = better pitcher.
    splits = p.get('splits') or {}

    def _opp_block(metric_key: str, head_label: str,
                   max_floor: float, max_ceil: float) -> str:
        if not splits:
            return ''
        rows = []
        for label in ('vs LHB', 'vs RHB', 'w/ RISP', '1st PA', 'Late'):
            entry = splits.get(label) or {}
            ab = entry.get('ab', 0)
            v = entry.get(metric_key, 0.0)
            if ab == 0:
                rows.append(
                    f'<div class="rth-splits__row">'
                    f'<div class="rth-splits__lbl">{label}</div>'
                    f'<div class="rth-splits__track"></div>'
                    f'<div class="rth-splits__val rth-splits__val--dim">—</div>'
                    f'</div>'
                )
                continue
            pct = max(8, min(100, ((max_ceil - v) / (max_ceil - max_floor)) * 70 + 30))
            v_str = f'{v:.3f}'.lstrip('0') if v < 1 else f'{v:.3f}'
            rows.append(
                f'<div class="rth-splits__row">'
                f'<div class="rth-splits__lbl">{label}</div>'
                f'<div class="rth-splits__track">'
                f'<div class="rth-splits__fill" style="width:{pct:.1f}%; background:{accent};"></div>'
                f'</div>'
                f'<div class="rth-splits__val">{v_str}</div>'
                f'</div>'
            )
        return (
            f'<div class="rth-splits">'
            f'<div class="rth-splits__head">{head_label}</div>'
            f'<div class="rth-splits__list">{"".join(rows)}</div>'
            f'</div>'
        )

    splits_block = (
        _opp_block('avg', 'SPLITS · OPP AVG', 0.300, 0.380)
        + _opp_block('ops', 'SPLITS · OPP OPS', 0.700, 1.000)
    )

    identity = (
        f'<header class="rth-id">'
        f'{headshot}'
        f'<div>'
        f'<div class="rth-id__meta">'
        f'<span class="rth-id__school">{_xe(p["school"])}</span>'
        f'<span style="font-size:6px;opacity:0.6;">●</span>'
        f'<span>#{p["seed"]} seed · {_xe(p["region"])}</span>'
        f'</div>'
        f'<h2 class="rth-id__name">{_xe(p["name"])}</h2>'
        f'<div class="rth-id__bio">{"".join(bio_parts)}</div>'
        f'</div>'
        f'{splits_block}'
        f'</header>'
    )

    # Stat cells
    stat_cells = []
    for k in STAT_KEYS:
        v, rank, pct = p['ranks'].get(k, (None, None, None))
        cell = (
            f'<div class="rth-stat">'
            f'<div class="rth-stat__top">'
            f'<div class="rth-stat__key">{k}</div>'
            f'{_rank_badge(rank, accent)}'
            f'</div>'
            f'<div class="rth-stat__val">{_fmt_pitcher_stat(k, v)}</div>'
            f'{_pct_bar(pct, accent)}'
            f'</div>'
        )
        stat_cells.append(cell)

    # Spray-against
    spray_svg = p.get('spray_svg') or ''
    if spray_svg:
        import re as _re
        spray_svg_flex = _re.sub(r'\swidth="[^"]+"\s+height="[^"]+"', '', spray_svg, count=1)
        if 'preserveAspectRatio' not in spray_svg_flex:
            spray_svg_flex = spray_svg_flex.replace('<svg ', '<svg preserveAspectRatio="xMidYMax meet" ', 1)
        spray_block = (
            f'<div class="rth-stats__spray">'
            f'<div class="rth-block-head">'
            f'<div>'
            f'<div class="rth-eyebrow">SPRAY AGAINST · BATTED-BALL ZONES</div>'
            f'<div class="rth-block-title">Hit distribution allowed</div>'
            f'</div></div>'
            f'<div class="rth-stats__spray-figure">{spray_svg_flex}</div>'
            f'</div>'
        )
    else:
        spray_block = ''

    stats = (
        f'<section class="rth-stats">'
        f'<div class="rth-stats__head">'
        f'<span>RATE & COUNTING STATS</span>'
        f'<span>D1 RANK · {total_qualifiers:,} qualifiers (≥{MIN_IP} IP)</span>'
        f'</div>'
        f'<div class="rth-stats__grid">{"".join(stat_cells)}</div>'
        f'{spray_block}'
        f'</section>'
    )

    # Right column — pace (rolling WHIP), K% × BB% scatter, hits-allowed donut
    pace_data = p.get('pace') or []
    whip_v, _, _ = p['ranks'].get('WHIP', (None, None, None))
    final_whip = f'{float(whip_v):.2f}' if whip_v is not None and not pd.isna(whip_v) else '—'
    pace_svg = _render_pace_svg(pace_data, accent)
    pace_block = (
        f'<div>'
        f'<div class="rth-block-head">'
        f'<div>'
        f'<div class="rth-eyebrow">PACE · LAST 30 DAYS · {len(pace_data) if pace_data else 0} APPEARANCES</div>'
        f'<div class="rth-block-title">Recent WHIP trajectory</div>'
        f'</div>'
        f'<div style="display:flex;flex-direction:column;align-items:flex-end;">'
        f'<span style="font-family:\'Inter\',sans-serif;font-weight:700;font-size:28px;line-height:1;letter-spacing:-0.02em;color:{accent};font-variant-numeric:tabular-nums;">{final_whip}</span>'
        f'<span class="rth-iso__lbl">final WHIP</span>'
        f'</div>'
        f'</div>'
        f'{pace_svg}'
        f'</div>'
    )

    # Scatter 1 — K% × BB%. Best is upper-right (high K%, low BB%) via
    # invert_y. Innings-length-agnostic so baseball (9-inning) and softball
    # (7-inning) qualifiers share the same scale.
    scatter_cloud = p.get('scatter_cloud') or []
    k_pct_v, _, _ = p['ranks'].get('K%', (None, None, None))
    bb_pct_v, _, _ = p['ranks'].get('BB%', (None, None, None))
    last_name = p['name'].split()[-1] if p.get('name') else ''

    def _scatter_legend():
        return (
            f'<div style="font-size:9px;letter-spacing:0.06em;color:var(--rth-muted);">'
            f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:rgba(0,0,0,0.25);margin-right:4px;vertical-align:middle;"></span>'
            f' qualifier '
            f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{accent};margin:0 4px 0 8px;vertical-align:middle;"></span>'
            f' {_xe(last_name)}</div>'
        )

    scatter_svg = _render_scatter_svg(
        scatter_cloud, k_pct_v, bb_pct_v, last_name, accent,
        x_label='K% →', y_label='← BB%',
        x_fmt='.1%', y_fmt='.1%',
        x_ticks=tuple(t / 100 for t in (10, 15, 20, 25, 30, 35, 40)),
        y_ticks=tuple(t / 100 for t in (2, 5, 8, 12, 16, 20)),
        x_pad=0.01, y_pad=0.01,
        x_clip=(0.0, 0.55), y_clip=(0.0, 0.30),
        invert_y=True,
    )
    scatter_block = (
        f'<div style="padding-top:14px;border-top:1px dashed rgba(0,0,0,0.18);">'
        f'<div class="rth-block-head">'
        f'<div>'
        f'<div class="rth-eyebrow">DIVISION LANDSCAPE · ALL QUALIFIERS</div>'
        f'<div class="rth-block-title">K% × BB%</div>'
        f'</div>{_scatter_legend()}'
        f'</div>'
        f'{scatter_svg}'
        f'</div>'
    )

    # Scatter 2 — FIP × WHIP. Both lower=better, so invert BOTH axes so the
    # best pitchers still land upper-right (matching the K%/BB% chart's
    # convention).
    scatter_cloud_fw = p.get('scatter_cloud_fw') or []
    fip_v, _, _ = p['ranks'].get('FIP', (None, None, None))
    whip_v, _, _ = p['ranks'].get('WHIP', (None, None, None))
    scatter_svg_fw = _render_scatter_svg(
        scatter_cloud_fw, fip_v, whip_v, last_name, accent,
        x_label='← FIP', y_label='← WHIP',
        x_fmt='.2f', y_fmt='.2f',
        x_ticks=(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
        y_ticks=(0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2),
        x_pad=0.2, y_pad=0.05,
        x_clip=(0.0, 12.0), y_clip=(0.4, 3.5),
        invert_x=True, invert_y=True,
    )
    scatter_block_fw = (
        f'<div style="padding-top:14px;border-top:1px dashed rgba(0,0,0,0.18);">'
        f'<div class="rth-block-head">'
        f'<div>'
        f'<div class="rth-eyebrow">RUN-PREVENTION LANDSCAPE</div>'
        f'<div class="rth-block-title">FIP × WHIP</div>'
        f'</div>{_scatter_legend()}'
        f'</div>'
        f'{scatter_svg_fw}'
        f'</div>'
    )

    # Hits-allowed donut
    hits_total = sum(p['hits_allowed'].values()) if p.get('hits_allowed') else 0
    mix_block = (
        f'<div class="rth-mix">'
        f'<div class="rth-block-head">'
        f'<div>'
        f'<div class="rth-eyebrow">HITS ALLOWED · BY TYPE</div>'
        f'<div class="rth-block-title">Damage profile</div>'
        f'</div>'
        f'<div class="rth-iso">'
        f'<span class="rth-iso__num" style="color:{accent};">{hits_total}</span>'
        f'<span class="rth-iso__lbl">HITS</span>'
        f'</div>'
        f'</div>'
        f'{_donut_svg(p.get("hits_allowed", {}), accent)}'
        f'</div>'
    )
    right = f'<section class="rth-right">{pace_block}{scatter_block}{scatter_block_fw}{mix_block}</section>'

    # Team logo watermark
    logo_b64 = p.get('team_logo_b64')
    bg_layer = ''
    row_style = f'--rth-accent:{accent};'
    if logo_b64:
        row_style += f' --rth-team-logo: url(data:image/png;base64,{logo_b64});'
        bg_layer = '<div class="rth-row__bg-logo"></div>'

    return (
        f'<article class="rth-row" style="{row_style}">'
        f'{bg_layer}{rail}{identity}{stats}{right}'
        f'</article>'
    )


# ── Top-level render ────────────────────────────────────────────────────────
def render_top_pitchers_html(players: list[dict], regional_name: str, sport: str,
                              division: str, total_qualifiers: int,
                              as_of_date: str | None = None) -> str:
    if as_of_date is None:
        as_of_date = datetime.now().strftime('%b %d, %Y')

    div_label = {'D1': 'Division I', 'D2': 'Division II', 'D3': 'Division III'}[division]
    sport_label = sport.title()

    rows_html = ''.join(
        _row_html(i, p, p['accent'], total_qualifiers) for i, p in enumerate(players)
    )

    masthead = (
        f'<header class="rth-mast">'
        f'<div class="rth-mast__kicker">'
        f'NCAA {div_label} {sport_label}<br/>'
        f'<strong>2026 Regionals · Pre-Tournament Brief</strong>'
        f'</div>'
        f'<h1 class="rth-mast__title">Regionals <em>Top Pitchers</em></h1>'
        f'<div class="rth-mast__meta">'
        f'{_xe(regional_name).upper()}<br/>'
        f'<strong>STATS THROUGH {as_of_date.upper()}</strong>'
        f'</div>'
        f'</header>'
    )
    foot = (
        f'<footer class="rth-foot">'
        f'<span>64 Analytics</span>'
        f'<span>Compiled {_xe(as_of_date)}</span>'
        f'</footer>'
    )
    # html2canvas + Download PNG button (same pattern as Top Hitters)
    png_filename = f'top_pitchers_{_xe(regional_name).lower().replace(" ", "_")}_{division.lower()}.png'.replace('/', '_')
    # html2canvas-pro fork — supports modern CSS color() / oklch() / lab() that
    # html2canvas@1.4.1 rejects. Same window.html2canvas function surface.
    png_script = f'''
<script src="https://cdn.jsdelivr.net/npm/html2canvas-pro@1.5.8/dist/html2canvas-pro.min.js"></script>
<script>
window.downloadTopPitchersPNG = async function() {{
  var node = document.querySelector('.rth-root');
  if (!node) return;
  var btn = document.getElementById('rtp-png-btn');
  var origText = btn ? btn.textContent : '';
  if (btn) {{ btn.disabled = true; btn.textContent = 'Rendering…'; }}
  try {{
    if (document.fonts && document.fonts.ready) await document.fonts.ready;
    var imgs = Array.from(node.querySelectorAll('img'));
    await Promise.all(imgs.map(function(img) {{
      if (img.complete && img.naturalWidth > 0) return Promise.resolve();
      return new Promise(function(res) {{
        img.addEventListener('load', res, {{ once: true }});
        img.addEventListener('error', res, {{ once: true }});
      }});
    }}));
    var canvas = await html2canvas(node, {{
      scale: 2, backgroundColor: '#FAF8F2', useCORS: true, allowTaint: true,
      logging: false, imageTimeout: 15000
    }});
    var a = document.createElement('a');
    a.download = {png_filename!r};
    a.href = canvas.toDataURL('image/png');
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
  }} catch (err) {{
    console.error('Top Pitchers PNG export failed:', err);
    alert('Download failed: ' + (err && err.message ? err.message : err) +
          '\\n\\nA common cause is a team-logo image blocked by CORS. ' +
          'Open the browser console (F12) for details.');
  }} finally {{
    if (btn) {{ btn.disabled = false; btn.textContent = origText || 'Download Top Pitchers PNG'; }}
  }}
}};
</script>
'''
    download_btn = (
        '<div style="text-align:center;margin:18px 0 8px;">'
        '<button id="rtp-png-btn" onclick="window.downloadTopPitchersPNG()" style="'
        'padding:10px 24px;background:#1a1a1a;color:#fff;border:none;border-radius:4px;'
        "font-family:'Inter',sans-serif;font-weight:700;font-size:12px;letter-spacing:.18em;"
        'text-transform:uppercase;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.25);">'
        'Download Top Pitchers PNG'
        '</button>'
        '</div>'
    )
    return f'{_STYLES}{png_script}{download_btn}<div class="rth-root">{masthead}{rows_html}{foot}</div>'


# ── Streamlit entry point ───────────────────────────────────────────────────
def render_tab(teams: list[str], seeds: list[int], team_ids: dict, sport: str,
                division: str, regional_name: str, pitching_df: pd.DataFrame,
                players_df: pd.DataFrame, teams_df: pd.DataFrame,
                accent_for: callable | None = None,
                conferences_df: pd.DataFrame | None = None,
                player_rank_df: pd.DataFrame | None = None,
                logo_bytes_for: callable | None = None):
    """Streamlit-side wrapper. Builds player dicts and renders HTML."""
    import streamlit as st

    valid_team_ids = [team_ids[t] for t in teams if team_ids.get(t) is not None]
    if not valid_team_ids:
        st.warning('No valid team IDs for this regional; cannot pull pitcher data.')
        return

    pool = build_division_pitcher_pool(pitching_df, teams_df, sport, division, conferences_df)
    scatter_cloud = build_division_pitcher_scatter(pool)
    scatter_cloud_fw = build_division_pitcher_scatter_fw(pool)
    top = select_top_pitchers(pitching_df, players_df, valid_team_ids, top_n=4,
                                player_rank_df=player_rank_df)
    if top.empty:
        st.info(f'No qualified pitchers (≥{MIN_IP} IP) found across the 4 selected teams in 2026 yet.')
        return

    # Events + bridges (lazy, only if available)
    try:
        from app_lib.spray_data import _load_pbp, _ncaa_pid_to_cb_player, _batter_bat_lookup
        events = _load_pbp(sport, division)
        # For pitcher splits we need the BATTER's hand — use _batter_bat_lookup
        batter_bat = _batter_bat_lookup(sport)
        bridge = _ncaa_pid_to_cb_player()
        cb_to_ncaa = {}
        if not bridge.empty:
            valid = bridge.dropna(subset=['cb_id', 'ncaa_pid'])
            cb_to_ncaa = dict(zip(valid['cb_id'].astype(int),
                                    valid['ncaa_pid'].astype(int)))
    except Exception:
        events = pd.DataFrame()
        batter_bat = {}
        cb_to_ncaa = {}

    # Optional photo upload — mirror the hitter renderer: normalize MIME + resize
    # large uploads so the embedded data URL stays small enough for html2canvas.
    import base64 as _b64
    from pathlib import Path as _Path

    def _normalize_and_resize(up) -> tuple[str, str]:
        """Unconditionally downscale uploaded photos to 800×1000 max + JPEG q=85
        (mirror of the hitter renderer). Keeps the iframe + html2canvas-pro
        capture from OOMing on full-res phone photos."""
        raw = up.read()
        mime = (up.type or '').strip()
        if not mime or not mime.startswith('image/'):
            ext = (_Path(up.name or '').suffix.lower().lstrip('.')) if up.name else ''
            mime = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
                    'png': 'image/png',  'webp': 'image/webp'}.get(ext, 'image/jpeg')
        if raw:
            try:
                from PIL import Image
                import io as _io
                im = Image.open(_io.BytesIO(raw))
                im.thumbnail((800, 1000), Image.LANCZOS)
                buf = _io.BytesIO()
                im.convert('RGB').save(buf, format='JPEG', quality=85, optimize=True)
                raw = buf.getvalue()
                mime = 'image/jpeg'
            except Exception:
                pass
        return _b64.b64encode(raw).decode('ascii'), mime.split('/')[-1]

    with st.expander('Add pitcher photos (replaces the striped placeholder)', expanded=False):
        st.caption('Move the slider until the head sits where you want in the preview. '
                   '0 = top of photo at top of frame, 100 = bottom of photo at top of frame. '
                   'The preview matches the actual headshot crop in the graphic.')
        upload_cols = st.columns(min(4, len(top)))
        for i, (_, row) in enumerate(top.iterrows()):
            cb_id = int(row['player_id']) if pd.notna(row.get('player_id')) else None
            with upload_cols[i % len(upload_cols)]:
                st.caption(str(row.get('player_name', '—')))
                up = st.file_uploader(
                    f'photo {i+1}',
                    type=['png', 'jpg', 'jpeg', 'webp'],
                    key=f'rtp_photo_{cb_id}',
                    label_visibility='collapsed',
                )
                if up is not None:
                    b64, mime_short = _normalize_and_resize(up)
                    st.session_state[f'rtp_photo_b64_{cb_id}'] = b64
                    st.session_state[f'rtp_photo_mime_{cb_id}'] = mime_short
                pos_y = st.slider(
                    f'crop Y · {i+1}', min_value=0, max_value=100, value=20, step=1,
                    key=f'rtp_photo_pos_y_{cb_id}',
                    label_visibility='collapsed',
                )
                preview_b64 = st.session_state.get(f'rtp_photo_b64_{cb_id}')
                preview_mime = st.session_state.get(f'rtp_photo_mime_{cb_id}', 'jpeg')
                if preview_b64:
                    st.markdown(
                        f'<div style="width:100%;aspect-ratio:4/3;'
                        f'background-image:url(data:image/{preview_mime};base64,{preview_b64});'
                        f'background-size:cover;background-position:50% {pos_y}%;'
                        f'border:1px solid #d4cfc4;border-radius:4px;'
                        f'margin-top:4px;"></div>'
                        f'<div style="font-size:10px;color:#888;text-align:right;'
                        f'margin-top:2px;">crop Y = {pos_y}</div>',
                        unsafe_allow_html=True,
                    )

    # Spray (perspective='pitching')
    try:
        from app_lib.spray_render import build_player_spray_svg
    except Exception:
        build_player_spray_svg = None

    team_to_seed = dict(zip(teams, seeds))
    id_to_team = {team_ids[t]: t for t in teams if team_ids.get(t) is not None}

    players_payload = []
    for _, row in top.iterrows():
        team_name = id_to_team.get(int(row['team_id']))
        if team_name is None:
            continue
        accent = accent_for(team_ids[team_name], team_to_seed[team_name]) if accent_for else '#1a1a1a'

        cb_id = int(row['player_id']) if pd.notna(row.get('player_id')) else None
        ncaa_pid = cb_to_ncaa.get(cb_id) if cb_id is not None else None

        splits = {}
        pace = []
        spray_svg = ''
        if ncaa_pid is not None and not events.empty:
            splits = compute_pitcher_splits(events, ncaa_pid, batter_bat)
            pace = compute_pitcher_pace(events, ncaa_pid)
            if build_player_spray_svg is not None:
                try:
                    spray_svg = build_player_spray_svg(
                        sport, division, ncaa_pid,
                        player_name=str(row.get('player_name', '')),
                        perspective='pitching',
                    )
                except Exception:
                    spray_svg = ''

        photo_b64 = st.session_state.get(f'rtp_photo_b64_{cb_id}') if cb_id is not None else None
        photo_mime = st.session_state.get(f'rtp_photo_mime_{cb_id}', 'jpeg') if cb_id is not None else 'jpeg'
        photo_pos_y = st.session_state.get(f'rtp_photo_pos_y_{cb_id}', 20) if cb_id is not None else 20

        # Team logo for row watermark
        team_logo_b64 = None
        tid = team_ids.get(team_name)
        if tid is not None:
            data = None
            if logo_bytes_for is not None:
                try:
                    data = logo_bytes_for(tid)
                except Exception:
                    data = None
            if data is None:
                logo_path = Path(__file__).resolve().parent.parent / 'team_logos_512' / f'{int(tid)}.png'
                if logo_path.exists():
                    try:
                        data = logo_path.read_bytes()
                    except Exception:
                        data = None
            if data is not None:
                try:
                    team_logo_b64 = base64.b64encode(data).decode('ascii')
                except Exception:
                    team_logo_b64 = None

        players_payload.append({
            'name': row.get('player_name', '—') or '—',
            'pos': (row.get('throw') or 'R') + 'HP',  # RHP/LHP from throw side
            'role': 'Starter' if (row.get('games_started', 0) or 0) >= 5 else 'Reliever',
            'yr': row.get('classification', '') or '',
            'bats': row.get('bat', '') or '—',
            'throws': row.get('throw', '') or '—',
            'ht': row.get('height', '') or '',
            'school': team_name,
            'seed': team_to_seed[team_name],
            'region': regional_name,
            'accent': accent,
            'ranks': compute_pitcher_ranks(row, pool),
            'hits_allowed': hits_allowed(row),
            'splits': splits,
            'pace': pace,
            'spray_svg': spray_svg,
            'scatter_cloud': scatter_cloud,
            'scatter_cloud_fw': scatter_cloud_fw,
            'photo_b64': photo_b64,
            'photo_mime': photo_mime,
            'photo_pos_y': photo_pos_y,
            'team_logo_b64': team_logo_b64,
        })

    if not players_payload:
        st.info('Could not resolve any pitchers back to the regional teams.')
        return

    html_doc = render_top_pitchers_html(
        players_payload, regional_name, sport, division,
        total_qualifiers=len(pool),
    )
    # components.html so the html2canvas <script> tag executes (markdown
    # strips/disables scripts). Generous height for 4-pitcher vertical stack.
    import streamlit.components.v1 as components
    n = len(players_payload)
    iframe_height = 320 + (n * 740) + 80
    components.html(html_doc, height=iframe_height, scrolling=True)
