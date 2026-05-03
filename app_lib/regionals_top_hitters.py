"""Regionals Top Hitters — editorial sports-magazine layout for Regional Preview.

Renders the top N hitters across the 4 selected regional teams as an HTML/CSS
document mirroring the Claude Design prototype: masthead → sub-bar → 4 player
rows (rail | identity+splits | stats+spray | pace+scatter+hit-mix).

Data comes from chart-builder/data/hitting.csv (already keyed by player_id, with
percentile ranks pre-computed). D1 ranks are computed within today's qualified
hitter pool (year=CURRENT_YEAR, division match, PA >= MIN_PA).

V1 scope: real slash line + counting stats + D1 ranks + hit-mix donut. Pace
chart, splits bars, and D1 scatter render with placeholder data — wire to PBP
events once the per-game / handedness pipeline lands.
"""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import pandas as pd

CURRENT_YEAR = 2026
MIN_PA = 100  # ~half a 56-game season; matches the 2.7 PA/team-game qualifier roughly

STAT_KEYS = ['AVG', 'OBP', 'SLG', 'OPS', 'HR', 'RBI', 'H', 'wRC+']
STAT_COLS = {
    'AVG': 'batting_average',
    'OBP': 'on_base_percentage',
    'SLG': 'slugging_percentage',
    'OPS': 'on_base_plus_slugging',
    'HR':  'home_runs',
    'RBI': 'runs_batted_in',
    'H':   'hits',
    'wRC+': 'weighted_runs_created_plus',
}
# Lower = better for none of these (all are higher-is-better)


# ── Data: hitter selection ──────────────────────────────────────────────────
def select_top_hitters(hitting_df: pd.DataFrame, players_df: pd.DataFrame,
                        team_ids: list[int], top_n: int = 4,
                        player_rank_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pick top-N hitters across the given team_ids, ranked by wRCE
    (weighted_run_created_efficiency, from player_rank.csv) when available,
    falling back to OPS within the min-PA gate. Returns the row joined to
    hitting.csv slash line + players.csv bio fields.
    """
    if player_rank_df is not None and not player_rank_df.empty:
        pr = player_rank_df[(player_rank_df['year'] == CURRENT_YEAR) &
                             (player_rank_df['team_id'].isin(team_ids))].copy()
        pr['_wrce'] = pd.to_numeric(pr['weighted_run_created_efficiency'], errors='coerce')
        pr = pr.dropna(subset=['_wrce'])
        # Gate by hitting PA so we don't pick a player with 0 box-score AB
        h_eligible = hitting_df[(hitting_df['year'] == CURRENT_YEAR) &
                                 (hitting_df['team_id'].isin(team_ids)) &
                                 (hitting_df['plate_appearances'] >= MIN_PA)]
        eligible_pids = set(h_eligible['player_id'].astype(int))
        pr = pr[pr['player_id'].astype(int).isin(eligible_pids)]
        if pr.empty:
            top_pr = pd.DataFrame()
        else:
            top_pr = pr.sort_values('_wrce', ascending=False).head(top_n)
            picked_pids = top_pr['player_id'].astype(int).tolist()
            h = h_eligible[h_eligible['player_id'].astype(int).isin(picked_pids)].copy()
            h = h.merge(top_pr[['player_id', '_wrce',
                                 'percentile_rank_weighted_run_created_efficiency',
                                 'integer_rank_weighted_run_created_efficiency']],
                        on='player_id', how='left')
            # Re-order to match the wRCE descending order
            h = h.set_index('player_id').loc[picked_pids].reset_index()
            p = players_df[['id', 'player_name', 'position', 'classification',
                            'height', 'bat', 'throw']].rename(columns={'id': 'player_id'})
            h = h.merge(p, on='player_id', how='left')
            return h.reset_index(drop=True)

    # Fallback to OPS
    h = hitting_df[(hitting_df['year'] == CURRENT_YEAR) &
                   (hitting_df['team_id'].isin(team_ids)) &
                   (hitting_df['plate_appearances'] >= MIN_PA)].copy()
    if h.empty:
        return h
    h = h.sort_values('on_base_plus_slugging', ascending=False).head(top_n)
    p = players_df[['id', 'player_name', 'position', 'classification',
                    'height', 'bat', 'throw']].rename(columns={'id': 'player_id'})
    h = h.merge(p, on='player_id', how='left')
    return h.reset_index(drop=True)


def build_d1_pool(hitting_df: pd.DataFrame, teams_df: pd.DataFrame, sport: str,
                   division: str, conferences_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """All current-year qualified hitters in the same sport/division as the regional.
    Used to compute D1 ranks + percentile + scatter cloud."""
    sport_label = 'Baseball' if sport == 'baseball' else 'Softball'
    div_full = {'D1': 'D-I', 'D2': 'D-II', 'D3': 'D-III'}[division]
    sport_teams = teams_df[teams_df['sport'] == sport_label]

    if conferences_df is not None:
        div_conf_ids = set(conferences_df[conferences_df['division'] == div_full]['id']
                            .astype(int).tolist())
        conf_id_num = pd.to_numeric(sport_teams['conference_id'], errors='coerce')
        sport_teams = sport_teams[conf_id_num.isin(div_conf_ids)]

    pool = hitting_df[(hitting_df['year'] == CURRENT_YEAR) &
                       (hitting_df['team_id'].isin(sport_teams['id'])) &
                       (hitting_df['plate_appearances'] >= MIN_PA)].copy()
    return pool


def compute_ranks(player_row: pd.Series, pool: pd.DataFrame) -> dict:
    """Return {stat_key: (value, rank, percentile)} for each stat in STAT_KEYS."""
    out = {}
    n = len(pool)
    for k in STAT_KEYS:
        col = STAT_COLS[k]
        if col not in pool.columns:
            out[k] = (None, None, None)
            continue
        v = player_row.get(col)
        if pd.isna(v):
            out[k] = (None, None, None)
            continue
        # Higher = better for all current keys
        rank = int((pool[col] > v).sum()) + 1
        pct = 100 * (1 - (rank - 1) / max(n - 1, 1))
        out[k] = (v, rank, pct)
    return out


def hit_mix(player_row: pd.Series) -> dict:
    """{1B, 2B, 3B, HR} counts."""
    h = player_row.get('hits', 0) or 0
    d = player_row.get('doubles', 0) or 0
    t = player_row.get('triples', 0) or 0
    hr = player_row.get('home_runs', 0) or 0
    singles = max(0, int(h) - int(d) - int(t) - int(hr))
    return {'1B': singles, '2B': int(d), '3B': int(t), 'HR': int(hr)}


# ── Splits — vs LHP / vs RHP / Home / Away / RISP ───────────────────────────
# Walks/HBP/Sac don't count as AB. Hits drive total bases. SLG = TB / AB.
# Codes here track the NCAA scorebook strings in events 'playResult'.
_AB_OUT_CODES = {'GO', 'FO', 'PO', 'LO', 'DP', 'TP', 'FC', 'E',
                 'K', 'KL', 'KS'}  # all count as AB
_HIT_CODES_TB = {'1B': 1, '2B': 2, '3B': 3, 'HR': 4}
_NON_AB_CODES = {'BB', 'IBB', 'HBP', 'SF', 'SH'}  # exclude from AB


def _slg_from_subset(subset: pd.DataFrame) -> tuple[float, int]:
    """Return (SLG, AB) for the subset. SLG = TB/AB; 0.0 if AB == 0."""
    if subset.empty:
        return (0.0, 0)
    counts = subset['playResult'].value_counts()
    tb = sum(counts.get(code, 0) * w for code, w in _HIT_CODES_TB.items())
    hits = sum(counts.get(code, 0) for code in _HIT_CODES_TB.keys())
    outs = sum(counts.get(code, 0) for code in _AB_OUT_CODES)
    ab = hits + outs
    if ab == 0:
        return (0.0, 0)
    return (round(tb / ab, 3), int(ab))


def compute_player_pace(events_df: pd.DataFrame, ncaa_player_id: int) -> list[dict]:
    """Per-game cumulative OPS trajectory from G1 to last game. Returns a list
    of {'game': int, 'date': str, 'ops': float, 'pa': int} ordered by date.
    OPS = OBP + SLG; both denominators are CUMULATIVE (PA for OBP, AB for SLG)
    so the line smooths out over the season rather than spiking on tiny samples.
    """
    if events_df.empty or pd.isna(ncaa_player_id):
        return []
    df = events_df[events_df['playerId'] == ncaa_player_id].copy()
    if df.empty:
        return []

    df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date_dt']).sort_values(['date_dt', 'gameId'])

    # Per-event flags
    pr = df['playResult']
    is_hit = pr.isin(_HIT_CODES_TB.keys())
    is_ab_out = pr.isin(_AB_OUT_CODES)
    is_walk = pr.isin({'BB', 'IBB'})
    is_hbp = pr == 'HBP'
    is_sf = pr == 'SF'
    is_sh = pr == 'SH'
    # AB = hits + AB-counted outs (excludes BB/HBP/SF/SH/IBB)
    df['is_ab'] = (is_hit | is_ab_out).astype(int)
    df['is_h'] = is_hit.astype(int)
    df['tb'] = pr.map(_HIT_CODES_TB).fillna(0).astype(int)
    df['is_bb'] = (is_walk).astype(int)
    df['is_hbp'] = is_hbp.astype(int)
    df['is_sf'] = is_sf.astype(int)
    df['is_sh'] = is_sh.astype(int)
    # PA = AB + BB + HBP + SF + SH (intentional walks roll into BB count)
    df['is_pa'] = df['is_ab'] + df['is_bb'] + df['is_hbp'] + df['is_sf'] + df['is_sh']

    by_game = df.groupby(['date_dt', 'gameId'], sort=True, as_index=False).agg(
        ab=('is_ab', 'sum'), h=('is_h', 'sum'), tb=('tb', 'sum'),
        bb=('is_bb', 'sum'), hbp=('is_hbp', 'sum'),
        sf=('is_sf', 'sum'), sh=('is_sh', 'sum'),
        pa=('is_pa', 'sum'),
    ).reset_index(drop=True)

    # Cumulative
    by_game['c_ab'] = by_game['ab'].cumsum()
    by_game['c_h'] = by_game['h'].cumsum()
    by_game['c_tb'] = by_game['tb'].cumsum()
    by_game['c_bb'] = by_game['bb'].cumsum()
    by_game['c_hbp'] = by_game['hbp'].cumsum()
    by_game['c_sf'] = by_game['sf'].cumsum()
    by_game['c_pa'] = by_game['pa'].cumsum()

    out = []
    for i, r in by_game.iterrows():
        c_ab = r['c_ab']; c_h = r['c_h']; c_tb = r['c_tb']
        c_bb = r['c_bb']; c_hbp = r['c_hbp']; c_sf = r['c_sf']; c_pa = r['c_pa']
        # OBP denominator excludes SH (and is AB+BB+HBP+SF). We approximate with c_pa - c_sh
        obp_denom = c_pa - r.get('sh', 0) - 0  # exclude SH
        obp = (c_h + c_bb + c_hbp) / obp_denom if obp_denom > 0 else 0.0
        slg = c_tb / c_ab if c_ab > 0 else 0.0
        out.append({
            'game': i + 1,
            'date': r['date_dt'].strftime('%Y-%m-%d'),
            'ops': float(obp + slg),
            'pa': int(c_pa),
        })
    return out


def compute_player_splits(events_df: pd.DataFrame, ncaa_player_id: int,
                            pitcher_throw_lookup: dict) -> dict:
    """For one batter (by NCAA season pid), compute SLG splits.
    Returns {'vs LHP', 'vs RHP', 'Home', 'Away', 'RISP'} -> (slg, ab).
    Empty-data splits return (0.0, 0)."""
    if events_df.empty or pd.isna(ncaa_player_id):
        return {k: (0.0, 0) for k in ('vs LHP', 'vs RHP', 'Home', 'Away', 'RISP')}
    df = events_df[events_df['playerId'] == ncaa_player_id].copy()
    if df.empty:
        return {k: (0.0, 0) for k in ('vs LHP', 'vs RHP', 'Home', 'Away', 'RISP')}

    # Pitcher hand bridge
    pitcher_pid = pd.to_numeric(df['pitcherId'], errors='coerce').astype('Int64')
    df['pitcher_throw'] = pitcher_pid.map(pitcher_throw_lookup)

    out = {
        'vs LHP': _slg_from_subset(df[df['pitcher_throw'] == 'L']),
        'vs RHP': _slg_from_subset(df[df['pitcher_throw'] == 'R']),
        'Home':   _slg_from_subset(df[df['battingTeam'] == df['homeTeam']]),
        'Away':   _slg_from_subset(df[df['battingTeam'] == df['awayTeam']]),
        'RISP':   _slg_from_subset(df[(df['runner2B'] == 1) | (df['runner3B'] == 1)]),
    }
    return out


# ── Rendering helpers ───────────────────────────────────────────────────────
def _fmt_stat(k: str, v) -> str:
    if v is None or pd.isna(v):
        return '—'
    if k in ('AVG', 'OBP', 'SLG'):
        return f'{float(v):.3f}'.lstrip('0') if 0 <= float(v) < 1 else f'{float(v):.3f}'
    if k == 'OPS':
        return f'{float(v):.3f}'
    if k == 'wRC+':
        return f'{int(round(float(v)))}'
    return f'{int(v)}'


def _suffix(n: int) -> str:
    v = n % 100
    if 11 <= v <= 13:
        return 'th'
    return {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')


def _initials(name: str) -> str:
    parts = (name or '').split()
    return ''.join(p[0] for p in parts if p)[:2].upper()


def _xe(s):
    return html.escape(str(s) if s is not None else '')


def _team_short(team_name: str) -> str:
    """Crude short-name fallback. Replace later with team_meta abbreviation column."""
    if not team_name:
        return '—'
    s = ''.join(c for c in team_name if c.isalpha() or c == ' ').strip()
    parts = s.split()
    if len(parts) == 1:
        return parts[0][:4].upper()
    if len(parts[0]) <= 4:
        return parts[0].upper()
    return ''.join(p[0] for p in parts[:3]).upper()


# ── HTML/CSS — top-level styles (scoped to .rth- root so they don't leak) ──
_STYLES = """
<style>
.rth-root {
  --rth-bg: #f6f1e8; --rth-paper: #fbf7ef; --rth-ink: #16130d; --rth-ink2: #3a342a;
  --rth-muted: #756d5e; --rth-rule: #1a1a1a; --rth-brand: #B22234;
  --rth-serif: "Source Serif 4", "Source Serif Pro", Georgia, serif;
  --rth-sans: "Inter", "Helvetica Neue", Helvetica, Arial, sans-serif;
  --rth-mono: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;
  background: var(--rth-bg);
  color: var(--rth-ink);
  font-family: var(--rth-sans);
  padding: 32px 36px 56px;
  border-radius: 6px;
  -webkit-font-smoothing: antialiased;
}
.rth-root * { box-sizing: border-box; }

.rth-mast {
  border-top: 6px solid var(--rth-brand);
  border-bottom: 1px solid var(--rth-ink);
  padding: 18px 0 22px;
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 24px;
  align-items: end;
}
.rth-mast__kicker {
  font-family: var(--rth-mono); font-size: 11px; letter-spacing: 0.18em;
  text-transform: uppercase; color: var(--rth-muted);
  display: flex; flex-direction: column; gap: 4px; margin-bottom: 8px;
}
.rth-mast__kicker strong { color: var(--rth-ink); font-weight: 600; }
.rth-mast__title {
  font-family: var(--rth-serif); font-weight: 600;
  font-size: clamp(40px, 5.4vw, 72px); line-height: 0.94;
  letter-spacing: -0.025em; margin: 0;
}
.rth-mast__title em { font-style: italic; font-weight: 400; color: var(--rth-brand); }
.rth-mast__meta {
  text-align: right; font-family: var(--rth-mono); font-size: 11px;
  letter-spacing: 0.06em; color: var(--rth-muted); line-height: 1.7;
}
.rth-mast__meta strong { color: var(--rth-ink); font-weight: 600; }

.rth-sub {
  display: grid; grid-template-columns: repeat(4, 1fr);
  border-bottom: 1px solid var(--rth-ink);
  padding: 12px 0; gap: 24px; margin-bottom: 24px;
}
.rth-sub__cell { display: flex; flex-direction: column; gap: 3px; }
.rth-sub__lbl {
  font-family: var(--rth-mono); font-size: 10px; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--rth-muted);
}
.rth-sub__val {
  font-family: var(--rth-serif); font-size: 22px; font-weight: 600;
  letter-spacing: -0.01em;
}
.rth-sub__sub { font-family: var(--rth-mono); font-size: 10px; color: var(--rth-muted); }

.rth-row {
  display: grid;
  grid-template-columns: 56px minmax(240px, 0.95fr) minmax(300px, 1.25fr) minmax(300px, 1fr);
  gap: 0; border-bottom: 1px solid var(--rth-ink);
  padding: 28px 0; position: relative;
}
.rth-row::before {
  content: ""; position: absolute; inset: 0 auto 0 0; width: 64px;
  background: var(--rth-accent, #1a1a1a); opacity: 0.06; pointer-events: none;
}
.rth-rail {
  width: 56px; display: flex; flex-direction: column; align-items: center;
  padding-top: 4px; border-right: 1px solid rgba(0,0,0,0.08); position: relative;
}
.rth-rail__rank {
  font-family: var(--rth-serif); font-weight: 600; font-size: 44px; line-height: 1;
  letter-spacing: -0.04em; color: var(--rth-accent, #1a1a1a);
}
.rth-rail__lbl {
  font-family: var(--rth-mono); font-size: 9px; letter-spacing: 0.2em;
  color: var(--rth-muted); margin-top: 6px;
}
.rth-rail__div { width: 16px; height: 1px; background: var(--rth-ink); margin: 18px 0; opacity: 0.4; }
.rth-rail__tag {
  font-family: var(--rth-mono); font-size: 9px; letter-spacing: 0.18em;
  color: var(--rth-muted); writing-mode: vertical-rl; transform: rotate(180deg); margin-top: 8px;
}

.rth-id {
  padding: 4px 24px 8px 20px; display: flex; flex-direction: column; gap: 12px;
  border-right: 1px solid rgba(0,0,0,0.08); position: relative; z-index: 1;
}
.rth-headshot {
  width: 100%; aspect-ratio: 4 / 3; position: relative;
  border: 1px solid var(--rth-ink); overflow: hidden; background: #f4efe7;
}
.rth-headshot__bg { position: absolute; inset: 0; width: 100%; height: 100%; }
.rth-headshot__init {
  position: absolute; inset: 0; display: grid; place-items: center;
  font-family: var(--rth-serif); font-weight: 700; font-size: 64px;
  letter-spacing: -0.04em; color: var(--rth-accent, #1a1a1a); opacity: 0.85;
}
.rth-id__meta {
  font-family: var(--rth-mono); font-size: 10px; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--rth-muted);
  display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
}
.rth-id__school { color: var(--rth-accent, #1a1a1a); font-weight: 600; }
.rth-id__name {
  font-family: var(--rth-serif); font-weight: 600; font-size: clamp(24px, 2.2vw, 32px);
  line-height: 1.02; letter-spacing: -0.02em; margin: 2px 0 4px;
}
.rth-id__bio {
  display: flex; gap: 14px; flex-wrap: wrap;
  font-family: var(--rth-mono); font-size: 11px; color: var(--rth-ink2);
  padding: 8px 0;
  border-top: 1px solid rgba(0,0,0,0.1);
  border-bottom: 1px solid rgba(0,0,0,0.1);
}
.rth-id__bio strong { font-family: var(--rth-sans); font-weight: 600; font-size: 13px; }

.rth-stats {
  padding: 4px 24px 0 24px; display: flex; flex-direction: column; gap: 12px;
  border-right: 1px solid rgba(0,0,0,0.08); position: relative; z-index: 1;
}
.rth-stats__head {
  display: flex; justify-content: space-between; align-items: baseline;
  font-family: var(--rth-mono); font-size: 10px; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--rth-muted);
  border-bottom: 1px solid var(--rth-ink); padding-bottom: 8px;
}
.rth-stats__grid {
  display: grid; grid-template-columns: repeat(2, 1fr);
  border-left: 1px solid rgba(0,0,0,0.08);
}
.rth-stat {
  padding: 6px 9px; display: flex; flex-direction: column; gap: 3px;
  border-right: 1px solid rgba(0,0,0,0.08);
  border-bottom: 1px solid rgba(0,0,0,0.08);
  background: var(--rth-paper);
}
.rth-stat__top { display: flex; justify-content: space-between; align-items: center; }
.rth-stat__key {
  font-family: var(--rth-mono); font-size: 10px; letter-spacing: 0.14em; color: var(--rth-muted);
}
.rth-stat__val {
  font-family: var(--rth-serif); font-weight: 600; font-size: 18px; line-height: 1;
  letter-spacing: -0.02em; color: var(--rth-ink); font-variant-numeric: tabular-nums;
}
.rth-rb {
  display: inline-flex; align-items: baseline; gap: 1px;
  font-family: var(--rth-mono); font-size: 10px; color: var(--rth-accent, #1a1a1a);
  font-weight: 600; padding: 2px 5px 1px;
  background: color-mix(in srgb, var(--rth-accent, #1a1a1a) 8%, transparent);
  border: 1px solid color-mix(in srgb, var(--rth-accent, #1a1a1a) 30%, transparent);
}
.rth-rb__hash { font-size: 8px; opacity: 0.7; }
.rth-rb__num  { font-size: 11px; }
.rth-rb__suf  { font-size: 7px; opacity: 0.7; margin-left: 1px; }

.rth-pct { display: flex; flex-direction: column; gap: 2px; }
.rth-pct__track {
  position: relative; height: 4px; background: rgba(0,0,0,0.06); overflow: visible;
}
.rth-pct__fill { position: absolute; top: 0; left: 0; bottom: 0; }
.rth-pct__tick {
  position: absolute; top: 0; bottom: 0; width: 1px;
  background: rgba(0,0,0,0.18); transform: translateX(-0.5px);
}
.rth-pct__tick--major { background: rgba(0,0,0,0.32); }
.rth-pct__marker {
  position: absolute; top: -2px; bottom: -2px; width: 2px; transform: translateX(-1px);
}
.rth-pct__lbls {
  display: flex; justify-content: space-between;
  font-family: var(--rth-mono); font-size: 8px; color: var(--rth-muted); margin-top: 1px;
}
.rth-pct__lbls span:nth-child(2) { margin-left: 35%; }
.rth-pct__lbls span:nth-child(3) { margin-left: 22%; }

.rth-right {
  padding: 4px 0 0 24px; display: flex; flex-direction: column; gap: 18px;
  position: relative; z-index: 1;
}
.rth-block-head { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; }
.rth-eyebrow {
  font-family: var(--rth-mono); font-size: 10px; letter-spacing: 0.14em; color: var(--rth-muted);
}
.rth-block-title {
  font-family: var(--rth-serif); font-size: 16px; font-weight: 600;
  letter-spacing: -0.01em; margin-top: 2px;
}

.rth-mix { padding-top: 14px; border-top: 1px dashed rgba(0,0,0,0.18); display: flex; flex-direction: column; gap: 8px; }
.rth-iso { display: flex; flex-direction: column; align-items: flex-end; }
.rth-iso__num {
  font-family: var(--rth-serif); font-weight: 700; font-size: 22px; line-height: 1;
  letter-spacing: -0.02em; color: var(--rth-accent, #1a1a1a); font-variant-numeric: tabular-nums;
}
.rth-iso__lbl {
  font-family: var(--rth-mono); font-size: 9px; letter-spacing: 0.14em;
  color: var(--rth-muted); margin-top: 4px;
}
.rth-donut { display: grid; grid-template-columns: auto 1fr; gap: 14px; align-items: center; }
.rth-donut__legend { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 4px; }
.rth-donut__legend li {
  display: grid; grid-template-columns: 10px 22px 1fr auto;
  align-items: center; gap: 8px;
  font-family: var(--rth-mono); font-size: 11px; font-variant-numeric: tabular-nums;
  border-bottom: 1px dotted rgba(0,0,0,0.10); padding: 3px 0;
}
.rth-donut__sw { width: 9px; height: 9px; }
.rth-donut__k { color: var(--rth-ink); font-weight: 600; }
.rth-donut__v { color: var(--rth-ink2); }
.rth-donut__pct { color: var(--rth-muted); font-size: 10px; }

.rth-placeholder {
  font-family: var(--rth-mono); font-size: 10px; color: var(--rth-muted);
  letter-spacing: 0.08em; padding: 14px 0; border-top: 1px dashed rgba(0,0,0,0.18);
}

.rth-splits {
  margin-top: 6px; padding-top: 12px; border-top: 1px dashed rgba(0,0,0,0.18);
}
.rth-splits__head {
  font-family: var(--rth-mono); font-size: 10px; letter-spacing: 0.14em;
  color: var(--rth-muted); margin-bottom: 10px;
}
.rth-splits__list { display: flex; flex-direction: column; gap: 6px; }
.rth-splits__row {
  display: grid; grid-template-columns: 56px 1fr 56px;
  align-items: center; gap: 10px;
}
.rth-splits__lbl {
  font-family: var(--rth-mono); font-size: 10px; letter-spacing: 0.08em;
  color: var(--rth-muted);
}
.rth-splits__track {
  height: 8px; background: rgba(0,0,0,0.06); position: relative;
}
.rth-splits__fill { height: 100%; }
.rth-splits__val {
  font-family: var(--rth-mono); font-size: 11px; text-align: right;
  font-variant-numeric: tabular-nums; color: var(--rth-ink);
}
.rth-splits__val--dim { color: var(--rth-muted); }

.rth-foot {
  margin-top: 28px; padding-top: 16px; border-top: 1px solid var(--rth-ink);
  display: flex; justify-content: space-between;
  font-family: var(--rth-mono); font-size: 10px; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--rth-muted);
}
</style>
"""


def _rank_badge(rank, accent):
    if rank is None:
        return ''
    return (
        f'<span class="rth-rb" style="--rth-accent: {accent};">'
        f'<span class="rth-rb__hash">#</span>'
        f'<span class="rth-rb__num">{rank}</span>'
        f'<span class="rth-rb__suf">{_suffix(rank)}</span>'
        f'</span>'
    )


def _pct_bar(pct, accent):
    if pct is None:
        return '<div class="rth-pct"><div class="rth-pct__track"></div></div>'
    pct = max(0.0, min(100.0, float(pct)))
    return (
        f'<div class="rth-pct">'
        f'<div class="rth-pct__track">'
        f'<div class="rth-pct__tick" style="left:50%;"></div>'
        f'<div class="rth-pct__tick rth-pct__tick--major" style="left:90%;"></div>'
        f'<div class="rth-pct__fill" style="width:{pct:.1f}%; background:{accent};"></div>'
        f'<div class="rth-pct__marker" style="left:{pct:.1f}%; background:{accent};"></div>'
        f'</div>'
        f'<div class="rth-pct__lbls"><span>0</span><span>50</span><span>90</span><span>100</span></div>'
        f'</div>'
    )


def _donut_svg(mix: dict, accent: str, size: int = 130) -> str:
    """Render the hit-mix donut as inline SVG."""
    order = ['1B', '2B', '3B', 'HR']
    total = sum(mix.get(k, 0) for k in order)
    if total == 0:
        return ''
    # Light → full accent across 1B → HR
    def shade(t):
        return f'color-mix(in srgb, {accent} {100 - t}%, #ffffff)'
    colors = {'1B': shade(60), '2B': shade(35), '3B': shade(15), 'HR': accent}
    cx = cy = size / 2
    r = size / 2 - 6
    r2 = r * 0.62
    import math
    cum = 0
    arc_paths = []
    legend_rows = []
    for k in order:
        v = mix.get(k, 0)
        if v == 0:
            legend_rows.append((k, v, 0, colors[k]))
            continue
        start = (cum / total) * math.pi * 2 - math.pi / 2
        cum += v
        end = (cum / total) * math.pi * 2 - math.pi / 2
        large = 1 if (end - start) > math.pi else 0
        x1 = cx + math.cos(start) * r; y1 = cy + math.sin(start) * r
        x2 = cx + math.cos(end) * r;   y2 = cy + math.sin(end) * r
        x3 = cx + math.cos(end) * r2;  y3 = cy + math.sin(end) * r2
        x4 = cx + math.cos(start) * r2; y4 = cy + math.sin(start) * r2
        d = (f'M {x1:.2f} {y1:.2f} A {r:.2f} {r:.2f} 0 {large} 1 {x2:.2f} {y2:.2f} '
             f'L {x3:.2f} {y3:.2f} A {r2:.2f} {r2:.2f} 0 {large} 0 {x4:.2f} {y4:.2f} Z')
        arc_paths.append(f'<path d="{d}" fill="{colors[k]}" stroke="#fbf7ef" stroke-width="1.2"/>')
        legend_rows.append((k, v, round(100 * v / total), colors[k]))
    arcs_svg = ''.join(arc_paths)
    legend_html = ''.join(
        f'<li><span class="rth-donut__sw" style="background:{c};"></span>'
        f'<span class="rth-donut__k">{k}</span>'
        f'<span class="rth-donut__v">{v}</span>'
        f'<span class="rth-donut__pct">{p}%</span></li>'
        for k, v, p, c in legend_rows
    )
    return (
        f'<div class="rth-donut">'
        f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}">'
        f'{arcs_svg}'
        f'<text x="{cx}" y="{cy - 2}" text-anchor="middle" font-family="serif" '
        f'font-size="22" font-weight="600" fill="#16130d">{total}</text>'
        f'<text x="{cx}" y="{cy + 12}" text-anchor="middle" font-family="ui-monospace,Menlo,monospace" '
        f'font-size="8" fill="#756d5e" letter-spacing="1.5">HITS</text>'
        f'</svg>'
        f'<ul class="rth-donut__legend">{legend_html}</ul>'
        f'</div>'
    )


def build_division_scatter_cloud(pool: pd.DataFrame) -> list[tuple[float, float]]:
    """Return (OBP, SLG) tuples for every qualified hitter in the division pool."""
    if pool.empty:
        return []
    obp = pd.to_numeric(pool['on_base_percentage'], errors='coerce')
    slg = pd.to_numeric(pool['slugging_percentage'], errors='coerce')
    valid = obp.notna() & slg.notna()
    return list(zip(obp[valid].tolist(), slg[valid].tolist()))


def _render_pace_svg(pace: list[dict], accent: str, height: int = 110, width: int = 520) -> str:
    """Cumulative OPS line chart, G1 → last game."""
    if not pace or len(pace) < 2:
        return ('<svg viewBox="0 0 100 75" preserveAspectRatio="none" '
                'style="width:100%;height:110px;display:block;">'
                '<text x="50" y="40" text-anchor="middle" '
                'font-family="ui-monospace,Menlo,monospace" font-size="3" fill="#999">'
                'No game-level data yet</text></svg>')
    pad = {'t': 12, 'r': 14, 'b': 22, 'l': 36}
    w = width - pad['l'] - pad['r']
    h = height - pad['t'] - pad['b']
    ops_vals = [pt['ops'] for pt in pace]
    ymin = max(0.0, min(ops_vals) * 0.92)
    ymax = max(ops_vals) * 1.05 or 1.0
    n = len(pace)
    def x_(i): return pad['l'] + (i / max(n - 1, 1)) * w
    def y_(v): return pad['t'] + (1 - (v - ymin) / max(ymax - ymin, 1e-6)) * h
    line = ' '.join(
        f'{"M" if i == 0 else "L"}{x_(i):.1f},{y_(pt["ops"]):.1f}'
        for i, pt in enumerate(pace)
    )
    # Y-axis ticks at 4 evenly spaced OPS levels
    yticks = [ymin + (ymax - ymin) * t for t in (0.0, 0.33, 0.66, 1.0)]
    grid = ''.join(
        f'<line x1="{pad["l"]}" x2="{pad["l"]+w}" y1="{y_(t):.1f}" y2="{y_(t):.1f}" '
        f'stroke="#1a1a1a" stroke-opacity="0.07"/>'
        f'<text x="{pad["l"]-6}" y="{y_(t)+3:.1f}" text-anchor="end" '
        f'font-size="9" fill="#666" font-family="ui-monospace,Menlo,monospace">'
        f'{t:.3f}</text>'
        for t in yticks
    )
    final = pace[-1]
    end_marker = (f'<circle cx="{x_(n-1):.1f}" cy="{y_(final["ops"]):.1f}" '
                  f'r="3.2" fill="{accent}" stroke="#fff" stroke-width="1.4"/>')
    g1_label = (f'<text x="{pad["l"]}" y="{height-6}" font-size="9" fill="#666" '
                f'font-family="ui-monospace,Menlo,monospace">G1</text>')
    final_label = (f'<text x="{pad["l"]+w}" y="{height-6}" font-size="9" '
                   f'fill="#666" text-anchor="end" '
                   f'font-family="ui-monospace,Menlo,monospace">G{n}</text>')
    axis = (f'<line x1="{pad["l"]}" x2="{pad["l"]+w}" '
            f'y1="{pad["t"]+h}" y2="{pad["t"]+h}" stroke="#222" stroke-opacity="0.4"/>')
    return (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" '
        f'style="width:100%;height:{height}px;display:block;">'
        f'{grid}'
        f'<path d="{line}" fill="none" stroke="{accent}" stroke-width="1.6" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'{end_marker}{g1_label}{final_label}{axis}'
        f'</svg>'
    )


def _render_scatter_svg(cloud: list[tuple[float, float]],
                          player_obp: float | None, player_slg: float | None,
                          player_last_name: str, accent: str,
                          width: int = 320, height: int = 220) -> str:
    """OBP × SLG scatter for the division. Player highlighted with halo + label."""
    if not cloud:
        return ('<svg viewBox="0 0 100 75" preserveAspectRatio="none" '
                'style="width:100%;height:220px;display:block;"><text x="50" y="40" '
                'text-anchor="middle" font-size="3" fill="#999">No qualifier data</text></svg>')
    pad = {'t': 18, 'r': 16, 'b': 30, 'l': 40}
    w = width - pad['l'] - pad['r']
    h = height - pad['t'] - pad['b']
    obps = [pt[0] for pt in cloud]
    slgs = [pt[1] for pt in cloud]
    xmin, xmax = max(0.18, min(obps) - 0.02), min(0.55, max(obps) + 0.02)
    ymin, ymax = max(0.20, min(slgs) - 0.02), min(1.10, max(slgs) + 0.02)
    def sx(v): return pad['l'] + ((v - xmin) / max(xmax - xmin, 1e-6)) * w
    def sy(v): return pad['t'] + (1 - (v - ymin) / max(ymax - ymin, 1e-6)) * h
    xticks = [t for t in (0.250, 0.300, 0.350, 0.400, 0.450, 0.500) if xmin <= t <= xmax]
    yticks = [t for t in (0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900) if ymin <= t <= ymax]
    grid = ''
    for t in xticks:
        grid += (f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{pad["t"]}" '
                 f'y2="{pad["t"]+h}" stroke="#000" stroke-opacity="0.05"/>'
                 f'<text x="{sx(t):.1f}" y="{height-10}" font-size="9" fill="#666" '
                 f'text-anchor="middle" font-family="ui-monospace,Menlo,monospace">{t:.3f}</text>')
    for t in yticks:
        grid += (f'<line x1="{pad["l"]}" x2="{pad["l"]+w}" y1="{sy(t):.1f}" '
                 f'y2="{sy(t):.1f}" stroke="#000" stroke-opacity="0.05"/>'
                 f'<text x="{pad["l"]-6}" y="{sy(t)+3:.1f}" font-size="9" fill="#666" '
                 f'text-anchor="end" font-family="ui-monospace,Menlo,monospace">{t:.3f}</text>')
    # Cloud
    dots = ''.join(
        f'<circle cx="{sx(o):.1f}" cy="{sy(s):.1f}" r="1.4" fill="#1a1a1a" fill-opacity="0.18"/>'
        for o, s in cloud
    )
    # Division mean crosshair
    mean_obp = sum(obps) / len(obps); mean_slg = sum(slgs) / len(slgs)
    cross = (f'<line x1="{sx(mean_obp):.1f}" x2="{sx(mean_obp):.1f}" y1="{pad["t"]}" '
             f'y2="{pad["t"]+h}" stroke="#999" stroke-dasharray="2 3" stroke-opacity="0.5"/>'
             f'<line x1="{pad["l"]}" x2="{pad["l"]+w}" y1="{sy(mean_slg):.1f}" '
             f'y2="{sy(mean_slg):.1f}" stroke="#999" stroke-dasharray="2 3" stroke-opacity="0.5"/>')
    # Player highlight
    highlight = ''
    if player_obp is not None and player_slg is not None and not pd.isna(player_obp) and not pd.isna(player_slg):
        px, py = sx(float(player_obp)), sy(float(player_slg))
        highlight = (
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="14" fill="{accent}" fill-opacity="0.12"/>'
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="{accent}" fill-opacity="0.28"/>'
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.6" fill="{accent}" stroke="#fff" stroke-width="1.5"/>'
            f'<g transform="translate({px+12:.1f}, {py-8:.1f})">'
            f'<text font-size="10" font-family="ui-monospace,Menlo,monospace" '
            f'fill="{accent}" font-weight="600">{_xe(player_last_name).upper()}</text>'
            f'<text y="11" font-size="9" font-family="ui-monospace,Menlo,monospace" fill="#444">'
            f'{float(player_obp):.3f} / {float(player_slg):.3f}</text></g>'
        )
    axes = (f'<line x1="{pad["l"]}" x2="{pad["l"]+w}" y1="{pad["t"]+h}" y2="{pad["t"]+h}" '
            f'stroke="#222" stroke-opacity="0.4"/>'
            f'<line x1="{pad["l"]}" x2="{pad["l"]}" y1="{pad["t"]}" y2="{pad["t"]+h}" '
            f'stroke="#222" stroke-opacity="0.4"/>'
            f'<text x="{pad["l"]+w/2}" y="{height}" font-size="9" fill="#666" '
            f'text-anchor="middle" font-family="ui-monospace,Menlo,monospace" '
            f'letter-spacing="1">OBP →</text>'
            f'<text x="6" y="{pad["t"]+h/2}" font-size="9" fill="#666" '
            f'font-family="ui-monospace,Menlo,monospace" letter-spacing="1" '
            f'transform="rotate(-90, 10, {pad["t"]+h/2})">SLG →</text>')
    return (f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" '
            f'style="width:100%;height:{height}px;display:block;">'
            f'{grid}{dots}{cross}{axes}{highlight}</svg>')


def _row_html(idx: int, p: dict, accent: str, total_qualifiers: int) -> str:
    """Render one player row."""
    # Rail
    rail = (
        f'<aside class="rth-rail">'
        f'<div class="rth-rail__rank" style="color:{accent};">{idx + 1:02d}</div>'
        f'<div class="rth-rail__lbl">RANK</div>'
        f'<div class="rth-rail__div"></div>'
        f'<div class="rth-rail__tag">TOP 4</div>'
        f'</aside>'
    )

    # Identity
    initials = _initials(p['name'])
    school_short = _team_short(p['school'])
    bio_parts = [
        f'<span><strong>{_xe(p["pos"])}</strong></span>',
        f'<span>{_xe(p["yr"])}</span>',
        f'<span>B/T {_xe(p["bats"])}/{_xe(p["throws"])}</span>',
    ]
    if p.get('ht') and str(p['ht']).strip():
        bio_parts.append(f'<span>{_xe(p["ht"])}</span>')
    if p.get('photo_b64'):
        # User-uploaded headshot — full-bleed, cropped to the 4:3 frame
        headshot = (
            f'<div class="rth-headshot" style="--rth-accent:{accent};">'
            f'<div style="position:absolute;inset:0;width:100%;height:100%;'
            f'background-image:url(data:image/{p.get("photo_mime","jpeg")};base64,{p["photo_b64"]});'
            f'background-size:cover;background-position:50% 35%;"></div>'
            f'</div>'
        )
    else:
        headshot = (
            f'<div class="rth-headshot" style="--rth-accent:{accent};">'
            f'<svg viewBox="0 0 100 100" class="rth-headshot__bg" preserveAspectRatio="none">'
            f'<defs><pattern id="rth-stripe-{idx}" width="6" height="6" '
            f'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<line x1="0" y1="0" x2="0" y2="6" stroke="{accent}" stroke-width="2.2" stroke-opacity="0.18"/>'
            f'</pattern></defs>'
            f'<rect width="100" height="100" fill="#f4efe7"/>'
            f'<rect width="100" height="100" fill="url(#rth-stripe-{idx})"/>'
            f'</svg>'
            f'<div class="rth-headshot__init">{initials}</div>'
            f'</div>'
        )
    # Splits block — SLG vs LHP/RHP/Home/Away/RISP
    splits = p.get('splits') or {}
    if splits:
        # Max scale dynamically — clip floor at .500 so a quiet split doesn't
        # blow up the bar visually, ceiling at 1.200 (well above realistic SLG)
        all_slg = [v[0] for v in splits.values() if v[1] > 0]
        max_scale = max(0.500, min(1.200, (max(all_slg) if all_slg else 0.500) * 1.05))
        rows = []
        for label in ('vs LHP', 'vs RHP', 'Home', 'Away', 'RISP'):
            slg, ab = splits.get(label, (0.0, 0))
            if ab == 0:
                rows.append(
                    f'<div class="rth-splits__row">'
                    f'<div class="rth-splits__lbl">{label}</div>'
                    f'<div class="rth-splits__track"></div>'
                    f'<div class="rth-splits__val rth-splits__val--dim">—</div>'
                    f'</div>'
                )
                continue
            pct = max(0, min(100, 100 * slg / max_scale))
            slg_str = f'{slg:.3f}'.lstrip('0') if slg < 1 else f'{slg:.3f}'
            rows.append(
                f'<div class="rth-splits__row">'
                f'<div class="rth-splits__lbl">{label}</div>'
                f'<div class="rth-splits__track">'
                f'<div class="rth-splits__fill" style="width:{pct:.1f}%; background:{accent};"></div>'
                f'</div>'
                f'<div class="rth-splits__val">{slg_str}</div>'
                f'</div>'
            )
        splits_block = (
            f'<div class="rth-splits">'
            f'<div class="rth-splits__head">SPLITS · SLG</div>'
            f'<div class="rth-splits__list">{"".join(rows)}</div>'
            f'</div>'
        )
    else:
        splits_block = ''

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

    # Stats grid
    stat_cells = []
    for k in STAT_KEYS:
        v, rank, pct = p['ranks'].get(k, (None, None, None))
        cell = (
            f'<div class="rth-stat">'
            f'<div class="rth-stat__top">'
            f'<div class="rth-stat__key">{k}</div>'
            f'{_rank_badge(rank, accent)}'
            f'</div>'
            f'<div class="rth-stat__val">{_fmt_stat(k, v)}</div>'
            f'{_pct_bar(pct, accent)}'
            f'</div>'
        )
        stat_cells.append(cell)
    spray_svg = p.get('spray_svg') or ''
    if spray_svg:
        spray_block = (
            f'<div style="margin-top:14px;padding-top:14px;'
            f'border-top:1px dashed rgba(0,0,0,0.18);">'
            f'<div class="rth-block-head">'
            f'<div>'
            f'<div class="rth-eyebrow">SPRAY · BATTED-BALL ZONES</div>'
            f'<div class="rth-block-title">Hit distribution</div>'
            f'</div></div>'
            f'<div style="width:100%;display:block;">{spray_svg}</div>'
            f'</div>'
        )
    else:
        spray_block = ''

    stats = (
        f'<section class="rth-stats">'
        f'<div class="rth-stats__head">'
        f'<span>SLASH LINE & COUNTING STATS</span>'
        f'<span>D1 RANK · {total_qualifiers:,} qualifiers</span>'
        f'</div>'
        f'<div class="rth-stats__grid">{"".join(stat_cells)}</div>'
        f'{spray_block}'
        f'</section>'
    )

    # Right column — pace placeholder + hit-mix donut
    avg_v, _, _ = p['ranks'].get('AVG', (None, None, None))
    slg_v, _, _ = p['ranks'].get('SLG', (None, None, None))
    iso_int = None
    if avg_v is not None and slg_v is not None:
        try:
            iso_int = int(round((float(slg_v) - float(avg_v)) * 1000))
        except (TypeError, ValueError):
            iso_int = None
    iso_str = f'.{iso_int:03d}' if iso_int is not None and iso_int >= 0 else (str(iso_int) if iso_int is not None else '—')

    pace_data = p.get('pace') or []
    ops_v, _, _ = p['ranks'].get('OPS', (None, None, None))
    final_ops = f'{float(ops_v):.3f}' if ops_v is not None and not pd.isna(ops_v) else '—'
    pace_svg = _render_pace_svg(pace_data, accent)
    pace_block = (
        f'<div>'
        f'<div class="rth-block-head">'
        f'<div>'
        f'<div class="rth-eyebrow">PACE · CUMULATIVE OPS · G1 → G{len(pace_data) if pace_data else "?"}</div>'
        f'<div class="rth-block-title">Season trajectory</div>'
        f'</div>'
        f'<div style="display:flex;flex-direction:column;align-items:flex-end;">'
        f'<span style="font-family:var(--rth-serif);font-weight:700;font-size:28px;line-height:1;letter-spacing:-0.02em;color:{accent};font-variant-numeric:tabular-nums;">{final_ops}</span>'
        f'<span class="rth-iso__lbl">final OPS</span>'
        f'</div>'
        f'</div>'
        f'{pace_svg}'
        f'</div>'
    )

    # Scatter — division OBP × SLG cloud, player highlighted
    scatter_cloud = p.get('scatter_cloud') or []
    obp_v, _, _ = p['ranks'].get('OBP', (None, None, None))
    slg_v, _, _ = p['ranks'].get('SLG', (None, None, None))
    last_name = p['name'].split()[-1] if p.get('name') else ''
    scatter_svg = _render_scatter_svg(scatter_cloud, obp_v, slg_v, last_name, accent)
    scatter_block = (
        f'<div style="padding-top:14px;border-top:1px dashed rgba(0,0,0,0.18);">'
        f'<div class="rth-block-head">'
        f'<div>'
        f'<div class="rth-eyebrow">DIVISION LANDSCAPE · ALL QUALIFIERS</div>'
        f'<div class="rth-block-title">OBP × SLG</div>'
        f'</div>'
        f'<div style="font-family:var(--rth-mono);font-size:9px;letter-spacing:0.06em;color:var(--rth-muted);">'
        f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:rgba(0,0,0,0.25);margin-right:4px;vertical-align:middle;"></span>'
        f' qualifier '
        f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{accent};margin:0 4px 0 8px;vertical-align:middle;"></span>'
        f' {_xe(last_name)}'
        f'</div>'
        f'</div>'
        f'{scatter_svg}'
        f'</div>'
    )
    mix_block = (
        f'<div class="rth-mix">'
        f'<div class="rth-block-head">'
        f'<div>'
        f'<div class="rth-eyebrow">HIT MIX · BY TYPE</div>'
        f'<div class="rth-block-title">Singles → Home runs</div>'
        f'</div>'
        f'<div class="rth-iso">'
        f'<span class="rth-iso__num" style="color:{accent};">{iso_str}</span>'
        f'<span class="rth-iso__lbl">ISO</span>'
        f'</div>'
        f'</div>'
        f'{_donut_svg(p["hit_mix"], accent)}'
        f'</div>'
    )
    right = f'<section class="rth-right">{pace_block}{scatter_block}{mix_block}</section>'

    return f'<article class="rth-row" style="--rth-accent:{accent};">{rail}{identity}{stats}{right}</article>'


def render_top_hitters_html(players: list[dict], regional_name: str, sport: str,
                             division: str, total_qualifiers: int,
                             as_of_date: str | None = None) -> str:
    """Build the full HTML doc.

    `players`: list of dicts with keys
        name, pos, yr, bats, throws, ht, school, seed, region,
        accent, ranks ({stat_key: (value, rank, pct)}), hit_mix
    """
    if as_of_date is None:
        as_of_date = datetime.now().strftime('%b %d, %Y')

    div_label = {'D1': 'Division I', 'D2': 'Division II', 'D3': 'Division III'}[division]
    sport_label = sport.title()

    combined_hr = sum(int(p['ranks'].get('HR', (0, None, None))[0] or 0) for p in players)
    combined_ops_vals = [float(p['ranks'].get('OPS', (0, None, None))[0] or 0) for p in players]
    combined_ops = sum(combined_ops_vals) / len(combined_ops_vals) if combined_ops_vals else 0

    rows_html = ''.join(
        _row_html(i, p, p['accent'], total_qualifiers) for i, p in enumerate(players)
    )

    masthead = (
        f'<header class="rth-mast">'
        f'<div>'
        f'<div class="rth-mast__kicker">'
        f'<span>NCAA {div_label} {sport_label}</span>'
        f'<span><strong>2026 Regionals · Pre-Tournament Brief</strong></span>'
        f'</div>'
        f'<h1 class="rth-mast__title">Regionals <em>Top Hitters</em></h1>'
        f'</div>'
        f'<div class="rth-mast__meta">'
        f'<div>{_xe(regional_name).upper()}</div>'
        f'<div><strong>STATS THROUGH {as_of_date.upper()}</strong></div>'
        f'<div>56-GAME REGULAR SEASON</div>'
        f'</div>'
        f'</header>'
    )

    sub = (
        f'<div class="rth-sub">'
        f'<div class="rth-sub__cell">'
        f'<span class="rth-sub__lbl">Players Profiled</span>'
        f'<span class="rth-sub__val">{len(players):02d}</span>'
        f'<span class="rth-sub__sub">Top hitters across the 4 regional teams</span>'
        f'</div>'
        f'<div class="rth-sub__cell">'
        f'<span class="rth-sub__lbl">Qualifying Pool</span>'
        f'<span class="rth-sub__val">{total_qualifiers:,}</span>'
        f'<span class="rth-sub__sub">{div_label} hitters with ≥ {MIN_PA} PA</span>'
        f'</div>'
        f'<div class="rth-sub__cell">'
        f'<span class="rth-sub__lbl">Combined HR</span>'
        f'<span class="rth-sub__val">{combined_hr}</span>'
        f'<span class="rth-sub__sub">Across the four players this season</span>'
        f'</div>'
        f'<div class="rth-sub__cell">'
        f'<span class="rth-sub__lbl">Combined OPS</span>'
        f'<span class="rth-sub__val">{combined_ops:.3f}</span>'
        f'<span class="rth-sub__sub">Group average</span>'
        f'</div>'
        f'</div>'
    )

    foot = (
        f'<footer class="rth-foot">'
        f'<span>64 Analytics</span>'
        f'<span>Sources · NCAA box scores</span>'
        f'<span>Compiled {_xe(as_of_date)}</span>'
        f'</footer>'
    )

    return f'{_STYLES}<div class="rth-root">{masthead}{sub}{rows_html}{foot}</div>'


# ── Streamlit entry point ───────────────────────────────────────────────────
def render_tab(teams: list[str], seeds: list[int], team_ids: dict, sport: str,
                division: str, regional_name: str, hitting_df: pd.DataFrame,
                players_df: pd.DataFrame, teams_df: pd.DataFrame,
                accent_for: callable | None = None,
                conferences_df: pd.DataFrame | None = None,
                player_rank_df: pd.DataFrame | None = None):
    """Streamlit-side wrapper. Builds player dicts from real data, renders HTML."""
    import base64
    import streamlit as st

    valid_team_ids = [team_ids[t] for t in teams if team_ids.get(t) is not None]
    if not valid_team_ids:
        st.warning('No valid team IDs for this regional; cannot pull hitter data.')
        return

    pool = build_d1_pool(hitting_df, teams_df, sport, division, conferences_df)
    scatter_cloud = build_division_scatter_cloud(pool)
    top = select_top_hitters(hitting_df, players_df, valid_team_ids, top_n=4,
                              player_rank_df=player_rank_df)
    if top.empty:
        st.info(f'No qualified hitters (≥{MIN_PA} PA) found across the 4 selected teams in 2026 yet.')
        return

    # Splits — load events PBP + pitcher-hand bridge once for all 4 players.
    # Use spray_data's helpers to avoid duplicating the load/cache logic.
    try:
        from app_lib.spray_data import _load_pbp, _pitcher_throw_lookup, _ncaa_pid_to_cb_player
        events = _load_pbp(sport, division)
        pitcher_throw = _pitcher_throw_lookup(sport)
        # cb_id (64A player_id) -> ncaa_pid bridge so we can find each top hitter's
        # NCAA season pid (the playerId column in events).
        bridge = _ncaa_pid_to_cb_player()
        cb_to_ncaa = {}
        if not bridge.empty:
            valid = bridge.dropna(subset=['cb_id', 'ncaa_pid'])
            cb_to_ncaa = dict(zip(valid['cb_id'].astype(int),
                                    valid['ncaa_pid'].astype(int)))
    except Exception as _e:
        events = pd.DataFrame()
        pitcher_throw = {}
        cb_to_ncaa = {}

    # Build per-player render dict
    players_payload = []
    team_to_seed = dict(zip(teams, seeds))
    id_to_team = {team_ids[t]: t for t in teams if team_ids.get(t) is not None}

    # Image upload UI — one expander with 4 file uploaders, persists in session
    with st.expander('Add player photos (replaces the striped placeholder)', expanded=False):
        upload_cols = st.columns(min(4, len(top)))
        for i, (_, row) in enumerate(top.iterrows()):
            cb_id = int(row['player_id']) if pd.notna(row.get('player_id')) else None
            with upload_cols[i % len(upload_cols)]:
                st.caption(str(row.get('player_name', '—')))
                up = st.file_uploader(
                    f'photo {i+1}',
                    type=['png', 'jpg', 'jpeg', 'webp'],
                    key=f'rth_photo_{cb_id}',
                    label_visibility='collapsed',
                )
                if up is not None:
                    photo_bytes = up.read()
                    st.session_state[f'rth_photo_b64_{cb_id}'] = base64.b64encode(photo_bytes).decode('ascii')
                    mime = up.type.split('/')[-1] if up.type else 'jpeg'
                    st.session_state[f'rth_photo_mime_{cb_id}'] = mime

    # Lazy import for spray rendering — only when we have an NCAA pid
    try:
        from app_lib.spray_render import build_player_spray_svg
    except Exception:
        build_player_spray_svg = None

    for _, row in top.iterrows():
        team_name = id_to_team.get(int(row['team_id']))
        if team_name is None:
            continue
        accent = accent_for(team_ids[team_name], team_to_seed[team_name]) if accent_for else '#1a1a1a'

        cb_id = int(row['player_id']) if pd.notna(row.get('player_id')) else None
        ncaa_pid = cb_to_ncaa.get(cb_id) if cb_id is not None else None

        # Splits + pace + spray via NCAA pid
        splits = {}
        pace = []
        spray_svg = ''
        if ncaa_pid is not None and not events.empty:
            splits = compute_player_splits(events, ncaa_pid, pitcher_throw)
            pace = compute_player_pace(events, ncaa_pid)
            if build_player_spray_svg is not None:
                try:
                    spray_svg = build_player_spray_svg(
                        sport, division, ncaa_pid,
                        player_name=str(row.get('player_name', '')),
                        perspective='hitting',
                    )
                except Exception:
                    spray_svg = ''

        photo_b64 = st.session_state.get(f'rth_photo_b64_{cb_id}') if cb_id is not None else None
        photo_mime = st.session_state.get(f'rth_photo_mime_{cb_id}', 'jpeg') if cb_id is not None else 'jpeg'

        players_payload.append({
            'name': row.get('player_name', '—') or '—',
            'pos': row.get('position', '') or '',
            'yr': row.get('classification', '') or '',
            'bats': row.get('bat', '') or '—',
            'throws': row.get('throw', '') or '—',
            'ht': row.get('height', '') or '',
            'school': team_name,
            'seed': team_to_seed[team_name],
            'region': regional_name,
            'accent': accent,
            'ranks': compute_ranks(row, pool),
            'hit_mix': hit_mix(row),
            'splits': splits,
            'pace': pace,
            'spray_svg': spray_svg,
            'scatter_cloud': scatter_cloud,
            'photo_b64': photo_b64,
            'photo_mime': photo_mime,
        })

    if not players_payload:
        st.info('Could not resolve any hitters back to the regional teams.')
        return

    html_doc = render_top_hitters_html(
        players_payload, regional_name, sport, division,
        total_qualifiers=len(pool),
    )
    st.markdown(html_doc, unsafe_allow_html=True)
