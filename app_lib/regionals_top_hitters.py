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
                        team_ids: list[int], top_n: int = 4) -> pd.DataFrame:
    """Pick top-N hitters across the given team_ids, ranked by OPS, min PA gate."""
    h = hitting_df[(hitting_df['year'] == CURRENT_YEAR) &
                   (hitting_df['team_id'].isin(team_ids)) &
                   (hitting_df['plate_appearances'] >= MIN_PA)].copy()
    if h.empty:
        return h
    h = h.sort_values('on_base_plus_slugging', ascending=False).head(top_n)

    # Join player names + position
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
    stats = (
        f'<section class="rth-stats">'
        f'<div class="rth-stats__head">'
        f'<span>SLASH LINE & COUNTING STATS</span>'
        f'<span>D1 RANK · {total_qualifiers:,} qualifiers</span>'
        f'</div>'
        f'<div class="rth-stats__grid">{"".join(stat_cells)}</div>'
        f'<div class="rth-placeholder">SPRAY · WIRE TO PBP EVENTS NEXT</div>'
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

    final_avg = _fmt_stat('AVG', avg_v)
    pace_block = (
        f'<div>'
        f'<div class="rth-block-head">'
        f'<div>'
        f'<div class="rth-eyebrow">PACE · 10-GAME ROLLING AVG</div>'
        f'<div class="rth-block-title">Season trajectory</div>'
        f'</div>'
        f'<div style="display:flex;flex-direction:column;align-items:flex-end;">'
        f'<span style="font-family:var(--rth-serif);font-weight:700;font-size:28px;line-height:1;letter-spacing:-0.02em;color:{accent};font-variant-numeric:tabular-nums;">{final_avg}</span>'
        f'<span class="rth-iso__lbl">final AVG</span>'
        f'</div>'
        f'</div>'
        f'<div class="rth-placeholder">PACE CHART · WIRE TO PER-GAME PBP NEXT</div>'
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
    right = f'<section class="rth-right">{pace_block}{mix_block}</section>'

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
                conferences_df: pd.DataFrame | None = None):
    """Streamlit-side wrapper. Builds player dicts from real data, renders HTML."""
    import streamlit as st

    valid_team_ids = [team_ids[t] for t in teams if team_ids.get(t) is not None]
    if not valid_team_ids:
        st.warning('No valid team IDs for this regional; cannot pull hitter data.')
        return

    pool = build_d1_pool(hitting_df, teams_df, sport, division, conferences_df)
    top = select_top_hitters(hitting_df, players_df, valid_team_ids, top_n=4)
    if top.empty:
        st.info(f'No qualified hitters (≥{MIN_PA} PA) found across the 4 selected teams in 2026 yet.')
        return

    # Build per-player render dict
    players_payload = []
    team_to_seed = dict(zip(teams, seeds))
    team_to_name = dict(zip(teams, teams))  # already names
    # team_id → team_name reverse lookup
    id_to_team = {team_ids[t]: t for t in teams if team_ids.get(t) is not None}

    for _, row in top.iterrows():
        team_name = id_to_team.get(int(row['team_id']))
        if team_name is None:
            continue
        accent = accent_for(team_ids[team_name], team_to_seed[team_name]) if accent_for else '#1a1a1a'
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
        })

    if not players_payload:
        st.info('Could not resolve any hitters back to the regional teams.')
        return

    html_doc = render_top_hitters_html(
        players_payload, regional_name, sport, division,
        total_qualifiers=len(pool),
    )
    st.markdown(html_doc, unsafe_allow_html=True)
