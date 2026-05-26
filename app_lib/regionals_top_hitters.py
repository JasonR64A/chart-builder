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

STAT_KEYS = ['AVG', 'OBP', 'SLG', 'OPS', 'HR', 'RBI', 'H', 'wRAA']
STAT_COLS = {
    'AVG': 'batting_average',
    'OBP': 'on_base_percentage',
    'SLG': 'slugging_percentage',
    'OPS': 'on_base_plus_slugging',
    'HR':  'home_runs',
    'RBI': 'runs_batted_in',
    'H':   'hits',
    'wRAA': 'weighted_runs_above_average',
}
# Lower = better for none of these (all are higher-is-better)


# ── Data: hitter selection ──────────────────────────────────────────────────
def select_top_hitters(hitting_df: pd.DataFrame, players_df: pd.DataFrame,
                        team_ids: list[int], top_n: int = 4,
                        player_rank_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pick top-N hitters across the given team_ids, ranked by wRAA
    (weighted_runs_above_average, from hitting.csv) within the min-PA gate.
    wRAA rewards both efficiency and playing time (raw above-average runs),
    so a top-end hitter with full PA volume ranks above an efficiency-leader
    with fewer PAs — matches the user's expected "top hitters in regional"
    view (changed 2026-05-12, was wRCE before).
    Returns the row joined to hitting.csv slash line + players.csv bio fields.
    """
    h = hitting_df[(hitting_df['year'] == CURRENT_YEAR) &
                   (hitting_df['team_id'].isin(team_ids)) &
                   (hitting_df['plate_appearances'] >= MIN_PA)].copy()
    if h.empty:
        return h
    h['_wraa'] = pd.to_numeric(h['weighted_runs_above_average'], errors='coerce')
    h = h.dropna(subset=['_wraa']).sort_values('_wraa', ascending=False).head(top_n)
    p = players_df[['id', 'player_name', 'position', 'classification',
                    'height', 'bat', 'throw']].rename(columns={'id': 'player_id'})
    h = h.merge(p, on='player_id', how='left')

    # If player_rank.csv is provided, merge the wRCE percentile/rank columns
    # so the per-row card can still display "D-I wRCE rank" if any downstream
    # code references them. Optional.
    if player_rank_df is not None and not player_rank_df.empty:
        pr = player_rank_df[(player_rank_df['year'] == CURRENT_YEAR) &
                             (player_rank_df['team_id'].isin(team_ids))][
            ['player_id', 'weighted_run_created_efficiency',
             'percentile_rank_weighted_run_created_efficiency',
             'integer_rank_weighted_run_created_efficiency']
        ].copy()
        pr = pr.rename(columns={'weighted_run_created_efficiency': '_wrce'})
        # Convert player_id types to match
        pr['player_id'] = pd.to_numeric(pr['player_id'], errors='coerce').astype('Int64')
        h['player_id'] = pd.to_numeric(h['player_id'], errors='coerce').astype('Int64')
        h = h.merge(pr, on='player_id', how='left')

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


def _splits_from_subset(subset: pd.DataFrame) -> dict:
    """Return {'slg', 'obp', 'ab', 'pa'} for the subset.
    SLG = TB/AB; OBP = (H+BB+HBP)/(AB+BB+HBP+SF); 0.0 if denominator == 0."""
    if subset.empty:
        return {'slg': 0.0, 'obp': 0.0, 'ab': 0, 'pa': 0}
    counts = subset['playResult'].value_counts()
    tb = sum(counts.get(code, 0) * w for code, w in _HIT_CODES_TB.items())
    hits = sum(counts.get(code, 0) for code in _HIT_CODES_TB.keys())
    outs = sum(counts.get(code, 0) for code in _AB_OUT_CODES)
    bb = counts.get('BB', 0) + counts.get('IBB', 0)
    hbp = counts.get('HBP', 0)
    sf = counts.get('SF', 0)
    sh = counts.get('SH', 0)
    ab = hits + outs
    obp_denom = ab + bb + hbp + sf  # excludes SH
    pa = ab + bb + hbp + sf + sh
    return {
        'slg': round(tb / ab, 3) if ab else 0.0,
        'obp': round((hits + bb + hbp) / obp_denom, 3) if obp_denom else 0.0,
        'ab':  int(ab),
        'pa':  int(pa),
    }


def compute_player_pace(events_df: pd.DataFrame, ncaa_player_id: int,
                          window_days: int = 30) -> list[dict]:
    """Per-game cumulative OPS trajectory over the most recent `window_days`
    days. Returns {'game', 'date', 'ops', 'pa'} ordered by date.
    OPS = OBP + SLG; cumulative WITHIN the window so a noisy early-season
    sample doesn't blow up the line.
    """
    if events_df.empty or pd.isna(ncaa_player_id):
        return []
    df = events_df[events_df['playerId'] == ncaa_player_id].copy()
    if df.empty:
        return []

    df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date_dt']).sort_values(['date_dt', 'gameId'])
    if df.empty:
        return []
    # Window cutoff (relative to player's most recent game). We do NOT drop
    # earlier rows yet — cumulative aggregates need the full season so the
    # OPS at the start of the window already reflects 100+ PA, not 4.
    window_cutoff = None
    if window_days and window_days > 0:
        window_cutoff = df['date_dt'].max() - pd.Timedelta(days=window_days)

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

    rows = []
    for i, r in by_game.iterrows():
        c_ab = r['c_ab']; c_h = r['c_h']; c_tb = r['c_tb']
        c_bb = r['c_bb']; c_hbp = r['c_hbp']; c_sf = r['c_sf']; c_pa = r['c_pa']
        # OBP denom = AB + BB + HBP + SF; SH excluded
        # Cumulative SH count
        sh_cum = by_game['sh'].iloc[:i+1].sum()
        obp_denom = c_pa - sh_cum
        obp = (c_h + c_bb + c_hbp) / obp_denom if obp_denom > 0 else 0.0
        slg = c_tb / c_ab if c_ab > 0 else 0.0
        rows.append({
            'date_dt': r['date_dt'],
            'game': i + 1,
            'date': r['date_dt'].strftime('%Y-%m-%d'),
            'ops': float(obp + slg),
            'pa': int(c_pa),
        })
    # Trim to the window for plotting; OPS values still reflect the full
    # season cumulative state at each game.
    if window_cutoff is not None:
        rows = [r for r in rows if r['date_dt'] >= window_cutoff]
    return [{'game': r['game'], 'date': r['date'], 'ops': r['ops'], 'pa': r['pa']}
            for r in rows]


def compute_player_splits(events_df: pd.DataFrame, ncaa_player_id: int,
                            pitcher_throw_lookup: dict) -> dict:
    """For one batter (by NCAA season pid), compute SLG splits.
    Returns {'vs LHP', 'vs RHP', 'Home', 'Away', 'RISP'} -> (slg, ab).
    Empty-data splits return (0.0, 0)."""
    empty = {'slg': 0.0, 'obp': 0.0, 'ab': 0, 'pa': 0}
    if events_df.empty or pd.isna(ncaa_player_id):
        return {k: dict(empty) for k in ('vs LHP', 'vs RHP', 'Home', 'Away', 'RISP')}
    df = events_df[events_df['playerId'] == ncaa_player_id].copy()
    if df.empty:
        return {k: dict(empty) for k in ('vs LHP', 'vs RHP', 'Home', 'Away', 'RISP')}

    pitcher_pid = pd.to_numeric(df['pitcherId'], errors='coerce').astype('Int64')
    df['pitcher_throw'] = pitcher_pid.map(pitcher_throw_lookup)

    out = {
        'vs LHP': _splits_from_subset(df[df['pitcher_throw'] == 'L']),
        'vs RHP': _splits_from_subset(df[df['pitcher_throw'] == 'R']),
        'Home':   _splits_from_subset(df[df['battingTeam'] == df['homeTeam']]),
        'Away':   _splits_from_subset(df[df['battingTeam'] == df['awayTeam']]),
        'RISP':   _splits_from_subset(df[(df['runner2B'] == 1) | (df['runner3B'] == 1)]),
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
    if k == 'wRAA':
        # Signed 1-decimal (wRAA is centered on 0; sign carries meaning)
        f = float(v)
        return f'{f:+.1f}'
    return f'{int(v)}'


def _suffix(n: int) -> str:
    v = n % 100
    if 11 <= v <= 13:
        return 'th'
    return {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')


def _initials(name: str) -> str:
    parts = (name or '').split()
    return ''.join(p[0] for p in parts if p)[:2].upper()


def _format_height(raw) -> str:
    """Convert height to feet'inches" format. Accepts:
      - 70 / "70" / 70.0 → 5'10"
      - "5-10" / "5'10" / "5'10\"" → 5'10" (already-formatted, normalized)
      - "" / NaN → "" """
    if raw is None or raw == '' or (isinstance(raw, float) and pd.isna(raw)):
        return ''
    s = str(raw).strip()
    # Already formatted as feet'inches
    if "'" in s or '"' in s:
        return s.replace('"', '"').strip()
    # "5-10" feet-dash-inches
    if '-' in s and not s.startswith('-'):
        parts = s.split('-')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return f'{parts[0]}\'{parts[1]}"'
    # Pure number — interpret as inches
    try:
        total_in = int(round(float(s)))
        if total_in <= 0:
            return ''
        ft, inch = divmod(total_in, 12)
        return f'{ft}\'{inch}"'
    except (ValueError, TypeError):
        return s


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
  --rth-sans: "Inter", "Helvetica Neue", Helvetica, Arial, sans-serif;
  --rth-serif: var(--rth-sans);   /* alias — Inter throughout per user pref */
  --rth-mono: var(--rth-sans);    /* alias — Inter throughout per user pref */
  background: var(--rth-bg);
  color: var(--rth-ink);
  font-family: var(--rth-sans);
  padding: 32px 36px 56px;
  border-radius: 6px;
  -webkit-font-smoothing: antialiased;
}
.rth-root, .rth-root * { font-family: var(--rth-sans) !important; }
.rth-root * { box-sizing: border-box; }

.rth-mast {
  border-top: 6px solid var(--rth-brand);
  border-bottom: 1px solid var(--rth-ink);
  padding: 14px 0 16px;
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 24px;
}
.rth-mast__kicker {
  font-family: var(--rth-sans); font-size: 11px; letter-spacing: 0.18em;
  text-transform: uppercase; color: var(--rth-muted);
  text-align: left;
  line-height: 1.4;
}
.rth-mast__kicker strong { color: var(--rth-ink); font-weight: 600; }
.rth-mast__title {
  font-family: var(--rth-sans); font-weight: 800;
  font-size: clamp(28px, 3.4vw, 44px); line-height: 1.0;
  letter-spacing: -0.025em; margin: 0; text-align: center;
  white-space: nowrap;
}
.rth-mast__title em { font-style: italic; font-weight: 600; color: var(--rth-brand); }
.rth-mast__meta {
  font-family: var(--rth-sans); font-size: 11px;
  letter-spacing: 0.12em; color: var(--rth-muted); text-transform: uppercase;
  text-align: right; line-height: 1.4;
}
.rth-mast__meta strong { color: var(--rth-ink); font-weight: 600; }

.rth-sub {
  display: grid; grid-template-columns: repeat(4, 1fr);
  border-bottom: 1px solid var(--rth-ink);
  padding: 12px 0; gap: 24px; margin-bottom: 24px;
}
.rth-sub__cell { display: flex; flex-direction: column; gap: 3px; }
.rth-sub__lbl {
  font-family: 'Inter',sans-serif; font-size: 10px; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--rth-muted);
}
.rth-sub__val {
  font-family: 'Inter',sans-serif; font-size: 22px; font-weight: 600;
  letter-spacing: -0.01em;
}
.rth-sub__sub { font-family: 'Inter',sans-serif; font-size: 10px; color: var(--rth-muted); }

.rth-row {
  display: grid;
  grid-template-columns: 56px minmax(240px, 0.95fr) minmax(300px, 1.25fr) minmax(300px, 1fr);
  gap: 0; border-bottom: 1px solid var(--rth-ink);
  padding: 28px 0; position: relative;
  overflow: hidden;
}
.rth-row::before {
  content: ""; position: absolute; inset: 0 auto 0 0; width: 64px;
  background: var(--rth-accent, #1a1a1a); opacity: 0.06; pointer-events: none;
}
/* Team logo watermark — sits behind everything in the row at 20% opacity */
.rth-row__bg-logo {
  position: absolute;
  inset: 0;
  background-image: var(--rth-team-logo, none);
  background-position: center center;
  background-repeat: no-repeat;
  background-size: 60% auto;
  opacity: 0.20;
  pointer-events: none;
  z-index: 0;
}
.rth-rail {
  width: 56px; display: flex; flex-direction: column; align-items: center;
  padding-top: 4px; border-right: 1px solid rgba(0,0,0,0.08); position: relative;
}
.rth-rail__rank {
  font-family: 'Inter',sans-serif; font-weight: 600; font-size: 44px; line-height: 1;
  letter-spacing: -0.04em; color: var(--rth-accent, #1a1a1a);
}
.rth-rail__lbl {
  font-family: 'Inter',sans-serif; font-size: 9px; letter-spacing: 0.2em;
  color: var(--rth-muted); margin-top: 6px;
}
.rth-rail__div { width: 16px; height: 1px; background: var(--rth-ink); margin: 18px 0; opacity: 0.4; }
.rth-rail__tag {
  font-family: 'Inter',sans-serif; font-size: 9px; letter-spacing: 0.18em;
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
  font-family: 'Inter',sans-serif; font-weight: 700; font-size: 64px;
  letter-spacing: -0.04em; color: var(--rth-accent, #1a1a1a); opacity: 0.85;
}
.rth-id__meta {
  font-family: 'Inter',sans-serif; font-size: 10px; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--rth-muted);
  display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
}
.rth-id__school { color: var(--rth-accent, #1a1a1a); font-weight: 600; }
.rth-id__name {
  font-family: 'Inter',sans-serif; font-weight: 600; font-size: clamp(24px, 2.2vw, 32px);
  line-height: 1.02; letter-spacing: -0.02em; margin: 2px 0 4px;
}
.rth-id__bio {
  display: flex; gap: 14px; flex-wrap: wrap;
  font-family: 'Inter',sans-serif; font-size: 11px; color: var(--rth-ink2);
  padding: 8px 0;
  border-top: 1px solid rgba(0,0,0,0.1);
  border-bottom: 1px solid rgba(0,0,0,0.1);
}
.rth-id__bio strong { font-family: var(--rth-sans); font-weight: 600; font-size: 13px; }

.rth-stats {
  padding: 4px 24px 0 24px; display: flex; flex-direction: column; gap: 12px;
  border-right: 1px solid rgba(0,0,0,0.08); position: relative; z-index: 1;
  justify-content: flex-start;
}
.rth-stats__spray {
  margin-top: 14px; padding-top: 14px;
  border-top: 1px dashed rgba(0,0,0,0.18);
  display: flex; flex-direction: column; gap: 6px;
  flex: 1 1 auto;
  min-height: 280px;
}
/* SVG anchored to the top of the section so it sits right under the
   "Hit distribution" header rather than centered with a gap above. */
.rth-stats__spray-figure {
  flex: 0 0 auto;
  display: block;
  margin-top: 4px;
}
.rth-stats__spray svg {
  /* Use max-width (not forced width:100%) so the SVG keeps its intrinsic
     aspect ratio AND html2canvas-pro can resolve layout. The browser will
     scale it down to fit while preserving its intrinsic height attribute,
     which is what html2canvas needs to capture the chart. */
  max-width: 100%;
  height: auto;
  display: block;
}
.rth-stats__head {
  display: flex; justify-content: space-between; align-items: baseline;
  font-family: 'Inter',sans-serif; font-size: 10px; letter-spacing: 0.14em;
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
  background: transparent;   /* let the row's team-logo watermark bleed through */
}
.rth-stat__top { display: flex; justify-content: space-between; align-items: center; }
.rth-stat__key {
  font-family: 'Inter',sans-serif; font-size: 10px; letter-spacing: 0.14em; color: var(--rth-muted);
}
.rth-stat__val {
  font-family: 'Inter',sans-serif; font-weight: 600; font-size: 18px; line-height: 1;
  letter-spacing: -0.02em; color: var(--rth-ink); font-variant-numeric: tabular-nums;
}
.rth-rb {
  display: inline-flex; align-items: baseline; gap: 1px;
  font-family: 'Inter',sans-serif; font-size: 10px; color: var(--rth-accent, #1a1a1a);
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
  font-family: 'Inter',sans-serif; font-size: 8px; color: var(--rth-muted); margin-top: 1px;
}
.rth-pct__lbls span:nth-child(2) { margin-left: 35%; }
.rth-pct__lbls span:nth-child(3) { margin-left: 22%; }

.rth-right {
  padding: 4px 0 0 24px; display: flex; flex-direction: column; gap: 18px;
  position: relative; z-index: 1;
}
.rth-block-head { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; }
.rth-eyebrow {
  font-family: 'Inter',sans-serif; font-size: 10px; letter-spacing: 0.14em; color: var(--rth-muted);
}
.rth-block-title {
  font-family: 'Inter',sans-serif; font-size: 16px; font-weight: 600;
  letter-spacing: -0.01em; margin-top: 2px;
}

.rth-mix { padding-top: 14px; border-top: 1px dashed rgba(0,0,0,0.18); display: flex; flex-direction: column; gap: 8px; }
.rth-iso { display: flex; flex-direction: column; align-items: flex-end; }
.rth-iso__num {
  font-family: 'Inter',sans-serif; font-weight: 700; font-size: 22px; line-height: 1;
  letter-spacing: -0.02em; color: var(--rth-accent, #1a1a1a); font-variant-numeric: tabular-nums;
}
.rth-iso__lbl {
  font-family: 'Inter',sans-serif; font-size: 9px; letter-spacing: 0.14em;
  color: var(--rth-muted); margin-top: 4px;
}
.rth-donut { display: grid; grid-template-columns: auto 1fr; gap: 14px; align-items: center; }
.rth-donut__legend { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 4px; }
.rth-donut__legend li {
  display: grid; grid-template-columns: 10px 22px 1fr auto;
  align-items: center; gap: 8px;
  font-family: 'Inter',sans-serif; font-size: 11px; font-variant-numeric: tabular-nums;
  border-bottom: 1px dotted rgba(0,0,0,0.10); padding: 3px 0;
}
.rth-donut__sw { width: 9px; height: 9px; }
.rth-donut__k { color: var(--rth-ink); font-weight: 600; }
.rth-donut__v { color: var(--rth-ink2); }
.rth-donut__pct { color: var(--rth-muted); font-size: 10px; }

.rth-placeholder {
  font-family: 'Inter',sans-serif; font-size: 10px; color: var(--rth-muted);
  letter-spacing: 0.08em; padding: 14px 0; border-top: 1px dashed rgba(0,0,0,0.18);
}

.rth-splits {
  margin-top: 6px; padding-top: 12px; border-top: 1px dashed rgba(0,0,0,0.18);
}
.rth-splits__head {
  font-family: 'Inter',sans-serif; font-size: 10px; letter-spacing: 0.14em;
  color: var(--rth-muted); margin-bottom: 10px;
}
.rth-splits__list { display: flex; flex-direction: column; gap: 6px; }
.rth-splits__row {
  display: grid; grid-template-columns: 56px 1fr 56px;
  align-items: center; gap: 10px;
}
.rth-splits__lbl {
  font-family: 'Inter',sans-serif; font-size: 10px; letter-spacing: 0.08em;
  color: var(--rth-muted);
}
.rth-splits__track {
  height: 8px; background: rgba(0,0,0,0.06); position: relative;
}
.rth-splits__fill { height: 100%; }
.rth-splits__val {
  font-family: 'Inter',sans-serif; font-size: 11px; text-align: right;
  font-variant-numeric: tabular-nums; color: var(--rth-ink);
}
.rth-splits__val--dim { color: var(--rth-muted); }

.rth-foot {
  margin-top: 28px; padding-top: 16px; border-top: 1px solid var(--rth-ink);
  display: flex; justify-content: space-between;
  font-family: 'Inter',sans-serif; font-size: 10px; letter-spacing: 0.12em;
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
        f'<text x="{cx}" y="{cy - 2}" text-anchor="middle" font-family="Inter,sans-serif" '
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


def build_division_scatter_cloud_disc(pool: pd.DataFrame) -> list[tuple[float, float]]:
    """Return (BB%, K%) tuples for every qualified hitter — plate discipline."""
    if pool.empty:
        return []
    pa = pd.to_numeric(pool['plate_appearances'], errors='coerce')
    bb = pd.to_numeric(pool['walks'], errors='coerce')
    k = pd.to_numeric(pool['strikeouts'], errors='coerce')
    valid = pa.notna() & bb.notna() & k.notna() & (pa > 0)
    bb_pct = (bb[valid] / pa[valid]).round(4)
    k_pct = (k[valid] / pa[valid]).round(4)
    return list(zip(bb_pct.tolist(), k_pct.tolist()))


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
    # X-axis labels — first and last game date in the window
    first_date = pace[0].get('date', '')
    last_date = pace[-1].get('date', '')
    g1_label = (f'<text x="{pad["l"]}" y="{height-6}" font-size="9" fill="#666" '
                f'font-family="ui-monospace,Menlo,monospace">{first_date}</text>')
    final_label = (f'<text x="{pad["l"]+w}" y="{height-6}" font-size="9" '
                   f'fill="#666" text-anchor="end" '
                   f'font-family="ui-monospace,Menlo,monospace">{last_date}</text>')
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
                          player_x: float | None, player_y: float | None,
                          player_last_name: str, accent: str,
                          x_label: str = 'OBP →', y_label: str = 'SLG →',
                          x_fmt: str = '.3f', y_fmt: str = '.3f',
                          x_ticks: tuple = (0.250, 0.300, 0.350, 0.400, 0.450, 0.500),
                          y_ticks: tuple = (0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900),
                          x_pad: float = 0.02, y_pad: float = 0.02,
                          x_clip: tuple = (0.0, 1.0), y_clip: tuple = (0.0, 1.5),
                          invert_y: bool = False, invert_x: bool = False,
                          width: int = 320, height: int = 220) -> str:
    """Generic scatter for the division pool with the player highlighted.
    `cloud` is a list of (x, y) tuples; `player_x` / `player_y` are the
    player's coordinates."""
    if not cloud:
        return ('<svg viewBox="0 0 100 75" preserveAspectRatio="none" '
                'style="width:100%;height:220px;display:block;"><text x="50" y="40" '
                'text-anchor="middle" font-size="3" fill="#999">No qualifier data</text></svg>')
    pad = {'t': 18, 'r': 16, 'b': 30, 'l': 40}
    w = width - pad['l'] - pad['r']
    h = height - pad['t'] - pad['b']
    xs = [pt[0] for pt in cloud]
    ys = [pt[1] for pt in cloud]
    xmin = max(x_clip[0], min(xs) - x_pad)
    xmax = min(x_clip[1], max(xs) + x_pad)
    ymin = max(y_clip[0], min(ys) - y_pad)
    ymax = min(y_clip[1], max(ys) + y_pad)
    if invert_x:
        # Mirror x so LOW values land on the right (good for "lower is better"
        # x-axis stats like FIP). Best players land upper-right alongside the
        # upper-right convention shared with other scatters.
        def sx(v): return pad['l'] + (1 - (v - xmin) / max(xmax - xmin, 1e-6)) * w
    else:
        def sx(v): return pad['l'] + ((v - xmin) / max(xmax - xmin, 1e-6)) * w
    if invert_y:
        # Flip so that the LOW end of y is at the top (good for "lower is
        # better" stats like K%). Best players land upper-right.
        def sy(v): return pad['t'] + ((v - ymin) / max(ymax - ymin, 1e-6)) * h
    else:
        def sy(v): return pad['t'] + (1 - (v - ymin) / max(ymax - ymin, 1e-6)) * h
    xticks_in = [t for t in x_ticks if xmin <= t <= xmax]
    yticks_in = [t for t in y_ticks if ymin <= t <= ymax]
    grid = ''
    for t in xticks_in:
        grid += (f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{pad["t"]}" '
                 f'y2="{pad["t"]+h}" stroke="#000" stroke-opacity="0.05"/>'
                 f'<text x="{sx(t):.1f}" y="{height-10}" font-size="9" fill="#666" '
                 f'text-anchor="middle">{format(t, x_fmt)}</text>')
    for t in yticks_in:
        grid += (f'<line x1="{pad["l"]}" x2="{pad["l"]+w}" y1="{sy(t):.1f}" '
                 f'y2="{sy(t):.1f}" stroke="#000" stroke-opacity="0.05"/>'
                 f'<text x="{pad["l"]-6}" y="{sy(t)+3:.1f}" font-size="9" fill="#666" '
                 f'text-anchor="end">{format(t, y_fmt)}</text>')
    dots = ''.join(
        f'<circle cx="{sx(o):.1f}" cy="{sy(s):.1f}" r="1.4" fill="#1a1a1a" fill-opacity="0.18"/>'
        for o, s in cloud
    )
    mean_x = sum(xs) / len(xs); mean_y = sum(ys) / len(ys)
    cross = (f'<line x1="{sx(mean_x):.1f}" x2="{sx(mean_x):.1f}" y1="{pad["t"]}" '
             f'y2="{pad["t"]+h}" stroke="#999" stroke-dasharray="2 3" stroke-opacity="0.5"/>'
             f'<line x1="{pad["l"]}" x2="{pad["l"]+w}" y1="{sy(mean_y):.1f}" '
             f'y2="{sy(mean_y):.1f}" stroke="#999" stroke-dasharray="2 3" stroke-opacity="0.5"/>')
    highlight = ''
    if player_x is not None and player_y is not None and not pd.isna(player_x) and not pd.isna(player_y):
        px, py = sx(float(player_x)), sy(float(player_y))
        highlight = (
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="14" fill="{accent}" fill-opacity="0.12"/>'
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="{accent}" fill-opacity="0.28"/>'
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.6" fill="{accent}" stroke="#fff" stroke-width="1.5"/>'
            f'<g transform="translate({px+12:.1f}, {py-8:.1f})">'
            f'<text font-size="10" fill="{accent}" font-weight="700">{_xe(player_last_name).upper()}</text>'
            f'<text y="11" font-size="9" fill="#444">'
            f'{format(float(player_x), x_fmt)} / {format(float(player_y), y_fmt)}</text></g>'
        )
    axes = (f'<line x1="{pad["l"]}" x2="{pad["l"]+w}" y1="{pad["t"]+h}" y2="{pad["t"]+h}" '
            f'stroke="#222" stroke-opacity="0.4"/>'
            f'<line x1="{pad["l"]}" x2="{pad["l"]}" y1="{pad["t"]}" y2="{pad["t"]+h}" '
            f'stroke="#222" stroke-opacity="0.4"/>'
            f'<text x="{pad["l"]+w/2}" y="{height}" font-size="9" fill="#666" '
            f'text-anchor="middle" letter-spacing="1">{x_label}</text>'
            f'<text x="6" y="{pad["t"]+h/2}" font-size="9" fill="#666" '
            f'letter-spacing="1" '
            f'transform="rotate(-90, 10, {pad["t"]+h/2})">{y_label}</text>')
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
    ht_fmt = _format_height(p.get('ht'))
    if ht_fmt:
        bio_parts.append(f'<span>{_xe(ht_fmt)}</span>')
    if p.get('photo_b64'):
        # User-uploaded headshot — full-bleed, cropped to the 4:3 frame.
        # Vertical focal point comes from the per-photo slider; default 20 keeps
        # the head visible on tall portraits (lower number pulls top of photo
        # into view; 50 = centered).
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
    # Splits — render SLG and OBP as separate stacked blocks. Both share the
    # 5 conditions (vs LHP/RHP/Home/Away/RISP) with the same accent color.
    splits = p.get('splits') or {}

    def _splits_block_for(metric_key: str, head_label: str, max_floor: float, max_ceiling: float) -> str:
        if not splits:
            return ''
        vals = [v.get(metric_key, 0.0) for v in splits.values() if v.get('ab', 0) > 0]
        max_scale = max(max_floor, min(max_ceiling, (max(vals) if vals else max_floor) * 1.05))
        rows = []
        for label in ('vs LHP', 'vs RHP', 'Home', 'Away', 'RISP'):
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
            pct = max(0, min(100, 100 * v / max_scale))
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
        _splits_block_for('slg', 'SPLITS · SLG', 0.500, 1.200)
        + _splits_block_for('obp', 'SPLITS · OBP', 0.400, 0.700)
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
        # Replace the SVG's intrinsic 100×75 dimensions with 400×300 (same 4:3
        # aspect, same viewBox so content scales). Reasons:
        #   1. html2canvas-pro needs concrete pixel dimensions to capture an
        #      SVG; if width/height are stripped or set via CSS only, the SVG
        #      vanishes from the PNG.
        #   2. The previous attempt KEPT the intrinsic 100×75 + used
        #      max-width:100% — that rendered at exactly 100 CSS px in browser
        #      (super tiny diamonds).
        # 400×300 is large enough to fill the stats column at a reasonable size
        # but max-width:100% on the parent still lets it scale down responsively.
        import re as _re
        spray_svg_flex = _re.sub(r'\swidth="[^"]+"\s+height="[^"]+"',
                                  ' width="400" height="300"', spray_svg, count=1)
        # Force preserveAspectRatio so the wedge stays centered as it scales
        if 'preserveAspectRatio' not in spray_svg_flex:
            spray_svg_flex = spray_svg_flex.replace('<svg ', '<svg preserveAspectRatio="xMidYMax meet" ', 1)
        spray_block = (
            f'<div class="rth-stats__spray">'
            f'<div class="rth-block-head">'
            f'<div>'
            f'<div class="rth-eyebrow">SPRAY · BATTED-BALL ZONES</div>'
            f'<div class="rth-block-title">Hit distribution</div>'
            f'</div></div>'
            f'<div class="rth-stats__spray-figure">{spray_svg_flex}</div>'
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
        f'<div class="rth-eyebrow">PACE · LAST 30 DAYS · {len(pace_data) if pace_data else 0} GAMES</div>'
        f'<div class="rth-block-title">Recent OPS trajectory</div>'
        f'</div>'
        f'<div style="display:flex;flex-direction:column;align-items:flex-end;">'
        f'<span style="font-family:\'Inter\',sans-serif;font-weight:700;font-size:28px;line-height:1;letter-spacing:-0.02em;color:{accent};font-variant-numeric:tabular-nums;">{final_ops}</span>'
        f'<span class="rth-iso__lbl">final OPS</span>'
        f'</div>'
        f'</div>'
        f'{pace_svg}'
        f'</div>'
    )

    # Scatter 1 — division OBP × SLG cloud
    scatter_cloud = p.get('scatter_cloud') or []
    obp_v, _, _ = p['ranks'].get('OBP', (None, None, None))
    slg_v, _, _ = p['ranks'].get('SLG', (None, None, None))
    last_name = p['name'].split()[-1] if p.get('name') else ''

    def _scatter_legend(label_left='qualifier', label_right=last_name):
        return (
            f'<div style="font-size:9px;letter-spacing:0.06em;color:var(--rth-muted);">'
            f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:rgba(0,0,0,0.25);margin-right:4px;vertical-align:middle;"></span>'
            f' {label_left} '
            f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{accent};margin:0 4px 0 8px;vertical-align:middle;"></span>'
            f' {_xe(label_right)}</div>'
        )

    scatter_svg_obp = _render_scatter_svg(scatter_cloud, obp_v, slg_v, last_name, accent)
    scatter_block = (
        f'<div style="padding-top:14px;border-top:1px dashed rgba(0,0,0,0.18);">'
        f'<div class="rth-block-head">'
        f'<div>'
        f'<div class="rth-eyebrow">DIVISION LANDSCAPE · ALL QUALIFIERS</div>'
        f'<div class="rth-block-title">OBP × SLG</div>'
        f'</div>{_scatter_legend()}'
        f'</div>'
        f'{scatter_svg_obp}'
        f'</div>'
    )

    # Scatter 2 — plate discipline (BB% × K%)
    scatter_cloud_disc = p.get('scatter_cloud_disc') or []
    bb_pct_v = p.get('bb_pct'); k_pct_v = p.get('k_pct')
    scatter_svg_disc = _render_scatter_svg(
        scatter_cloud_disc, bb_pct_v, k_pct_v, last_name, accent,
        x_label='BB% →', y_label='← K%',  # arrow flipped — low K% is up
        x_fmt='.1%', y_fmt='.1%',
        x_ticks=tuple(t / 100 for t in (5, 10, 15, 20, 25)),
        y_ticks=tuple(t / 100 for t in (10, 15, 20, 25, 30, 35)),
        x_pad=0.01, y_pad=0.01,
        x_clip=(0.0, 0.40), y_clip=(0.0, 0.50),
        invert_y=True,  # K% bad — best players upper-right (low K%, high BB%)
    )
    scatter_block_disc = (
        f'<div style="padding-top:14px;border-top:1px dashed rgba(0,0,0,0.18);">'
        f'<div class="rth-block-head">'
        f'<div>'
        f'<div class="rth-eyebrow">PLATE DISCIPLINE · ALL QUALIFIERS</div>'
        f'<div class="rth-block-title">BB% × K%</div>'
        f'</div>{_scatter_legend()}'
        f'</div>'
        f'{scatter_svg_disc}'
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
    right = f'<section class="rth-right">{pace_block}{scatter_block}{scatter_block_disc}{mix_block}</section>'

    # Team logo watermark behind the row at 20% opacity
    logo_b64 = p.get('team_logo_b64')
    bg_layer = ''
    row_style = f'--rth-accent:{accent};'
    if logo_b64:
        row_style += f' --rth-team-logo: url(data:image/png;base64,{logo_b64});'
        bg_layer = '<div class="rth-row__bg-logo"></div>'

    return (
        f'<article class="rth-row" style="{row_style}">'
        f'{bg_layer}'
        f'{rail}{identity}{stats}{right}'
        f'</article>'
    )


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
        f'<div class="rth-mast__kicker">'
        f'NCAA {div_label} {sport_label}<br/>'
        f'<strong>2026 Regionals · Pre-Tournament Brief</strong>'
        f'</div>'
        f'<h1 class="rth-mast__title">Regionals <em>Top Hitters</em></h1>'
        f'<div class="rth-mast__meta">'
        f'{_xe(regional_name).upper()}<br/>'
        f'<strong>STATS THROUGH {as_of_date.upper()}</strong>'
        f'</div>'
        f'</header>'
    )

    sub = ''  # sub-bar removed per user feedback

    foot = (
        f'<footer class="rth-foot">'
        f'<span>64 Analytics</span>'
        f'<span>Compiled {_xe(as_of_date)}</span>'
        f'</footer>'
    )

    # html2canvas script + Download PNG button at the top. Captures the
    # whole .rth-root element including all player rows.
    png_filename = f'top_hitters_{_xe(regional_name).lower().replace(" ", "_")}_{division.lower()}.png'.replace('/', '_')
    # html2canvas-pro is a maintained fork that supports modern CSS color
    # functions (color(), color-mix(), oklch(), lab() etc.) which the original
    # html2canvas@1.4.1 rejects with "Attempting to parse an unsupported color
    # function 'color'". Same window.html2canvas function surface — no other
    # code changes needed.
    png_script = f'''
<script src="https://cdn.jsdelivr.net/npm/html2canvas-pro@1.5.8/dist/html2canvas-pro.min.js"></script>
<script>
window.downloadTopHittersPNG = async function() {{
  var node = document.querySelector('.rth-root');
  if (!node) return;
  var btn = document.getElementById('rth-png-btn');
  var origText = btn ? btn.textContent : '';
  if (btn) {{ btn.disabled = true; btn.textContent = 'Rendering…'; }}
  try {{
    if (document.fonts && document.fonts.ready) await document.fonts.ready;
    // Force-load all <img> tags inside the card so html2canvas doesn't snapshot
    // before logos/photos resolve. Failures resolve (not reject) so one bad
    // image can't kill the whole capture.
    var imgs = Array.from(node.querySelectorAll('img'));
    await Promise.all(imgs.map(function(img) {{
      if (img.complete && img.naturalWidth > 0) return Promise.resolve();
      return new Promise(function(res) {{
        img.addEventListener('load', res, {{ once: true }});
        img.addEventListener('error', res, {{ once: true }});
      }});
    }}));
    // Cap the canvas height (~6000px) so a tall 4-player card doesn't build a
    // huge canvas that hangs the tab or hits the browser's canvas-size ceiling.
    var scale = Math.min(2, 6000 / Math.max(1, node.scrollHeight));
    var canvas = await html2canvas(node, {{
      scale: scale, backgroundColor: '#FAF8F2', useCORS: true, allowTaint: true,
      logging: false, imageTimeout: 15000
    }});
    var fname = {png_filename!r};
    var triggerDownload = function(href, revoke) {{
      var a = document.createElement('a');
      a.download = fname; a.href = href;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      if (revoke) setTimeout(function() {{ URL.revokeObjectURL(href); }}, 60000);
    }};
    // toBlob encodes the PNG off the main thread (toDataURL is synchronous and
    // freezes the tab on large canvases). Fall back to toDataURL if unavailable.
    if (canvas.toBlob) {{
      await new Promise(function(res) {{
        canvas.toBlob(function(blob) {{
          if (blob) {{ triggerDownload(URL.createObjectURL(blob), true); }}
          else {{ triggerDownload(canvas.toDataURL('image/png'), false); }}
          res();
        }}, 'image/png');
      }});
    }} else {{
      triggerDownload(canvas.toDataURL('image/png'), false);
    }}
  }} catch (err) {{
    console.error('Top Hitters PNG export failed:', err);
    alert('Download failed: ' + (err && err.message ? err.message : err) +
          '\\n\\nA common cause is a team-logo image blocked by CORS. ' +
          'Open the browser console (F12) for details.');
  }} finally {{
    if (btn) {{ btn.disabled = false; btn.textContent = origText || 'Download Top Hitters PNG'; }}
  }}
}};
</script>
'''
    download_btn = (
        '<div style="text-align:center;margin:18px 0 8px;">'
        '<button id="rth-png-btn" onclick="window.downloadTopHittersPNG()" style="'
        'padding:10px 24px;background:#1a1a1a;color:#fff;border:none;border-radius:4px;'
        "font-family:'Inter',sans-serif;font-weight:700;font-size:12px;letter-spacing:.18em;"
        'text-transform:uppercase;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.25);">'
        'Download Top Hitters PNG'
        '</button>'
        '</div>'
    )
    return f'{_STYLES}{png_script}{download_btn}<div class="rth-root">{masthead}{sub}{rows_html}{foot}</div>'


# ── Streamlit entry point ───────────────────────────────────────────────────
def render_tab(teams: list[str], seeds: list[int], team_ids: dict, sport: str,
                division: str, regional_name: str, hitting_df: pd.DataFrame,
                players_df: pd.DataFrame, teams_df: pd.DataFrame,
                accent_for: callable | None = None,
                conferences_df: pd.DataFrame | None = None,
                player_rank_df: pd.DataFrame | None = None,
                logo_bytes_for: callable | None = None):
    """Streamlit-side wrapper. Builds player dicts from real data, renders HTML."""
    import base64
    import streamlit as st

    valid_team_ids = [team_ids[t] for t in teams if team_ids.get(t) is not None]
    if not valid_team_ids:
        st.warning('No valid team IDs for this regional; cannot pull hitter data.')
        return

    pool = build_d1_pool(hitting_df, teams_df, sport, division, conferences_df)
    scatter_cloud = build_division_scatter_cloud(pool)
    scatter_cloud_disc = build_division_scatter_cloud_disc(pool)
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

    # Image upload UI — one expander with 4 file uploaders, persists in session.
    # Resize big uploads (≥600KB) so the embedded data URL stays small enough
    # for html2canvas to capture without OOM; normalize the MIME so the data URL
    # actually renders (some uploads arrive as application/octet-stream).
    def _normalize_and_resize(up) -> tuple[str, str]:
        """Unconditionally downscale uploaded photos to 800×1000 max + JPEG q=85.
        The headshot frame renders at ~200px wide; anything larger is wasted
        bytes that bloat the Streamlit iframe AND the html2canvas-pro capture
        (browser tab was crashing on the full-res versions)."""
        raw = up.read()
        mime = (up.type or '').strip()
        if not mime or not mime.startswith('image/'):
            ext = (Path(up.name or '').suffix.lower().lstrip('.')) if up.name else ''
            mime = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
                    'png': 'image/png',  'webp': 'image/webp'}.get(ext, 'image/jpeg')
        if raw:
            try:
                from PIL import Image
                import io as _io
                im = Image.open(_io.BytesIO(raw))
                im.thumbnail((800, 1000), Image.LANCZOS)
                buf = _io.BytesIO()
                # Always re-encode as JPEG (smaller; the card crops out
                # transparency anyway via background-size:cover).
                im.convert('RGB').save(buf, format='JPEG', quality=85, optimize=True)
                raw = buf.getvalue()
                mime = 'image/jpeg'
            except Exception:
                pass
        return base64.b64encode(raw).decode('ascii'), mime.split('/')[-1]

    with st.expander('Add player photos (replaces the striped placeholder)', expanded=False):
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
                    key=f'rth_photo_{cb_id}',
                    label_visibility='collapsed',
                )
                if up is not None:
                    b64, mime_short = _normalize_and_resize(up)
                    st.session_state[f'rth_photo_b64_{cb_id}'] = b64
                    st.session_state[f'rth_photo_mime_{cb_id}'] = mime_short
                pos_y = st.slider(
                    f'crop Y · {i+1}', min_value=0, max_value=100, value=20, step=1,
                    key=f'rth_photo_pos_y_{cb_id}',
                    label_visibility='collapsed',
                )
                # Live preview — same background-image/size/position as the
                # rendered card so the user can dial in pos_y without having
                # to download the PNG and check.
                preview_b64 = st.session_state.get(f'rth_photo_b64_{cb_id}')
                preview_mime = st.session_state.get(f'rth_photo_mime_{cb_id}', 'jpeg')
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
        photo_pos_y = st.session_state.get(f'rth_photo_pos_y_{cb_id}', 20) if cb_id is not None else 20

        # BB% / K% for the discipline scatter
        pa_v = row.get('plate_appearances', 0) or 0
        bb_pct = (row.get('walks', 0) / pa_v) if pa_v else None
        k_pct = (row.get('strikeouts', 0) / pa_v) if pa_v else None

        # Team logo for the row's background watermark
        team_logo_b64 = None
        tid_for_logo = team_ids.get(team_name)
        if tid_for_logo is not None:
            data = None
            if logo_bytes_for is not None:
                try:
                    data = logo_bytes_for(tid_for_logo)
                except Exception:
                    data = None
            if data is None:
                from pathlib import Path
                logo_path = Path(__file__).resolve().parent.parent / 'team_logos_512' / f'{int(tid_for_logo)}.png'
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
            'scatter_cloud_disc': scatter_cloud_disc,
            'bb_pct': bb_pct,
            'k_pct': k_pct,
            'photo_b64': photo_b64,
            'photo_mime': photo_mime,
            'photo_pos_y': photo_pos_y,
            'team_logo_b64': team_logo_b64,
        })

    if not players_payload:
        st.info('Could not resolve any hitters back to the regional teams.')
        return

    html_doc = render_top_hitters_html(
        players_payload, regional_name, sport, division,
        total_qualifiers=len(pool),
    )
    # Use components.html so the html2canvas <script> tag executes; markdown
    # strips/disables scripts. Set generous height for 4-player vertical stack
    # (~720px per row + masthead/footer) and enable scrolling so nothing
    # gets clipped while still letting html2canvas snapshot the full DOM.
    import streamlit.components.v1 as components
    n = len(players_payload)
    iframe_height = 320 + (n * 740) + 80  # masthead + rows + footer/button
    components.html(html_doc, height=iframe_height, scrolling=True)
