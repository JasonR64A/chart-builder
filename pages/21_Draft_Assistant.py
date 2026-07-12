"""Draft Assistant — live draft tracker + prospect board + scouting card.

Four tabs:
  1. Live Draft Tracker — enter picks on draft night (Supabase draft_picks table);
     every formula from the MLB Draft Database workbook fires instantly, and a
     ready-to-post tweet is generated for the latest pick.
  2. Tweet Feed — one copy-ready tweet per recorded pick, newest first
     (2026 stat line, rank-vs-pick value angle, expected bonus from the trends
     model, portal history 2025+2026, HS commitment).
  3. Prospect Board — ranks from MLB / ESPN / BA / Over-Slot / PG, drafted flags.
  4. Scouting Card — the original per-player broadcast card (unchanged).

Formula engine: app_lib/draft_engine.py — validated against all 615 picks of the
2025 draft (expected signing 315/315 exact on slotted picks; codes 100%).
Reference data: data/draft/*.csv|json exported from the workbook.
"""
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app_lib import draft_engine as E

DATA = Path(__file__).resolve().parent.parent / "data"
DRAFT_DAY = date(2026, 7, 11)   # 2026 draft begins Jul 11 (per ESPN); update if it slips
YEAR = 2026

SUPABASE_URL = 'https://vfzoroabzmbvwkcyozes.supabase.co'
SUPABASE_ANON_KEY = (
    'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.'
    'eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZmem9yb2Fiem1idndrY3lvemVzIiwi'
    'cm9sZSI6ImFub24iLCJpYXQiOjE2OTQwNDU5NTgsImV4cCI6MjAwOTYyMTk1OH0.'
    'MpzhpgI2fVDC5ucrECl2AuQ9VfT_8aaTmFunthyJAPA'
)
HEADERS = {
    'apikey': SUPABASE_ANON_KEY,
    'Authorization': f'Bearer {SUPABASE_ANON_KEY}',
    'Content-Type': 'application/json',
}
PICKS_TABLE = 'draft_picks'

st.set_page_config(page_title="Draft Assistant", layout="wide")


def norm(s):
    s = str(s).strip()
    return s[:-2] if s.endswith(".0") else s


def fnum(x):
    try:
        return float(x)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load(name, **kw):
    return pd.read_csv(DATA / name, dtype=str, encoding="latin-1", keep_default_na=False, **kw)


@st.cache_data(show_spinner=False)
def load_draft_refs():
    master = pd.read_csv(DATA / 'draft' / 'draft_master.csv', dtype=str, keep_default_na=False)
    history = pd.read_csv(DATA / 'draft' / 'draft_history.csv', dtype=str, keep_default_na=False)
    trends = json.loads((DATA / 'draft' / 'draft_trends.json').read_text())

    # Cumulative board rank: AVERAGE of the source ranks, where a player missing
    # from a source counts as that source's list length + 1 (BA top 500 -> 501,
    # ESPN top 250 -> 251, ...). Sources with no data at all (e.g. PG until
    # provided) are excluded. Board # = 1..N in cumulative order; players ranked
    # by nobody stay unnumbered.
    rank_cols = ['rank_ba', 'rank_mlb', 'rank_espn', 'rank_fss', 'pg_rank']
    active = []
    for c in rank_cols:
        v = pd.to_numeric(master[c], errors='coerce')
        if v.notna().any():
            active.append((v, int(v.max()) + 1))
    if active:
        total = sum(v.fillna(pen) for v, pen in active)
        avg = (total / len(active)).round(1)
        ranked_any = pd.concat([v for v, _ in active], axis=1).notna().any(axis=1)
        master['overall_avg'] = avg.where(ranked_any, other=pd.NA)
        master = master.sort_values('overall_avg', na_position='last').reset_index(drop=True)
        nums = pd.Series(master.index + 1, index=master.index, dtype='Int64')
        nums = nums.where(master['overall_avg'].notna())
        master['number'] = nums.astype(str).replace('<NA>', '')
        master['overall_avg'] = master['overall_avg'].astype(str).replace('<NA>', '')
    return master, history, trends


master, history, trends = load_draft_refs()

# Unranked convention: n players carry a cumulative board number; anyone NOT
# ranked by any of our sources (blank number, incl. off-board MLB-feed picks)
# is treated as rank n+1 — same penalty logic the cumulative average uses
# per-source (missing from BA's top 500 counts as 501).
import pandas as _pd
N_RANKED = int(_pd.to_numeric(master['number'], errors='coerce').max() or 0)
UNRANKED_NUM = N_RANKED + 1

# Official 2026 slot values + pick->team assignments (from MLB StatsAPI)
try:
    _slots = pd.read_csv(DATA / 'draft' / 'draft_slots_2026.csv', dtype=str, keep_default_na=False)
    slot_by_pick = {int(float(r['pick'])): fnum(r['slot_value']) or 0 for _, r in _slots.iterrows()}
    team_by_pick = {int(float(r['pick'])): r['team'] for _, r in _slots.iterrows()}
    # Complete pick -> round map: comp/PPI picks (round labels like 'CB-A',
    # 'PPI', '4C') inherit the last numeric round before them (fill-forward
    # in pick order — matches draft_history convention, e.g. CB-A rows = rd 1).
    round_by_pick = {}
    _cur_rnd = 1
    for _, r in _slots.sort_values(by='pick', key=lambda s: pd.to_numeric(s, errors='coerce')).iterrows():
        _lbl = str(r['round'])
        if _lbl.replace('.', '').isdigit():
            _cur_rnd = int(float(_lbl))
        round_by_pick[int(float(r['pick']))] = _cur_rnd
    MLB_TEAMS = sorted(_slots['team'].dropna().unique())
except Exception:
    slot_by_pick, team_by_pick, round_by_pick = {}, {}, {}
    MLB_TEAMS = sorted(history[history['year'] == '2025']['team'].dropna().unique())


def fetch_picks():
    try:
        r = requests.get(f'{SUPABASE_URL}/rest/v1/{PICKS_TABLE}', headers=HEADERS,
                         params={'select': '*', 'year': f'eq.{YEAR}', 'order': 'pick.asc'}, timeout=12)
        if r.status_code == 200:
            return pd.DataFrame(r.json()), None
        return pd.DataFrame(), f'{r.status_code}: {r.text[:200]}'
    except Exception as ex:
        return pd.DataFrame(), str(ex)


# ── shared slim stat + portal data (numeric loads; string loads OOM'd Render) ──
HIT_COLS = ['player_id', 'team_id', 'year', 'plate_appearances', 'at_bats', 'hits', 'doubles',
            'triples', 'home_runs', 'walks', 'hit_by_pitch', 'sac_fly', 'runs_batted_in',
            'stolen_bases', 'games_played', 'on_base_percentage', 'slugging_percentage',
            'isolated_power', 'weighted_on_base_average', 'on_base_plus_slugging', 'batting_average',
            'percentile_rank_on_base_plus_slugging', 'percentile_rank_weighted_on_base_average',
            'percentile_rank_isolated_power', 'percentile_rank_weighted_runs_created',
            'percentile_rank_runs_plate_appearance']
PIT_COLS = ['player_id', 'team_id', 'year', 'innings_pitched', 'batters_faced', 'strikeouts',
            'earned_run_average', 'strikeout_to_walk_ratio', 'walks_plus_hits_per_inning_pitched',
            'fielding_independent_pitching', 'on_base_plus_slugging_against',
            'percentile_rank_on_base_plus_slugging_against', 'percentile_rank_strikeout_to_walk_ratio',
            'percentile_rank_walks_plus_hits_per_inning_pitched',
            'percentile_rank_fielding_independent_pitching',
            'percentile_rank_skill_interactive_earned_run_average']


@st.cache_data(show_spinner=False)
def load_stats(name, cols):
    return pd.read_csv(DATA / name, low_memory=False, encoding="latin-1",
                       usecols=lambda c: c in cols)


hit_all = load_stats("hitting.csv", HIT_COLS)
pit_all = load_stats("pitching.csv", PIT_COLS)


@st.cache_data(show_spinner=False)
def load_portal():
    """player_id -> portal info for the 2025 and 2026 cycles + team names."""
    prp = pd.read_csv(DATA / 'portal_rank_player.csv', dtype=str, keep_default_na=False,
                      usecols=['player_id', 'team_id', 'new_team_id', 'year'])
    teams = pd.read_csv(DATA / 'teams.csv', dtype=str, keep_default_na=False, usecols=['id', 'name'])
    tname = dict(zip(teams['id'].map(norm), teams['name']))
    portal = {}
    for _, r in prp.iterrows():
        yr = norm(r['year'])
        if yr in ('2025', '2026'):
            portal.setdefault(norm(r['player_id']), {})[yr] = {
                'from': tname.get(norm(r['team_id']), ''),
                'to': tname.get(norm(r['new_team_id']), '') if r['new_team_id'].strip() else ''}
    return portal


PORTAL = load_portal()


def _money(v):
    if v is None:
        return ''
    return f'${v/1e6:.1f}M' if v >= 950_000 else f'${v/1e3:.0f}K'


def stat_line_2026(pid):
    """One tweet-ready 2026 stat line for a linked college player ('' if none)."""
    if not pid:
        return ''
    pidn = pd.to_numeric(pid, errors='coerce')
    h = hit_all[(hit_all['year'] == 2026) & (hit_all['player_id'] == pidn)]
    p = pit_all[(pit_all['year'] == 2026) & (pit_all['player_id'] == pidn)]
    pa = float(h['plate_appearances'].sum()) if len(h) else 0
    bf = float(p['batters_faced'].sum()) if len(p) else 0
    if bf > pa and len(p):
        s = p.iloc[0]
        return (f"{s['earned_run_average']:.2f} ERA, {s['walks_plus_hits_per_inning_pitched']:.2f} WHIP, "
                f"{int(s['strikeouts'])} K in {s['innings_pitched']} IP")
    if len(h) and pa >= 40:
        s = h.iloc[0]
        avg = f"{s['batting_average']:.3f}".lstrip('0')
        ops = f"{s['on_base_plus_slugging']:.3f}".lstrip('0')
        line = f"{avg}/{ops}, {int(s['home_runs'])} HR, {int(s['runs_batted_in'])} RBI"
        if s['stolen_bases'] >= 15:
            line += f", {int(s['stolen_bases'])} SB"
        return line
    return ''


# ── expected-$ explanation + comps (the model's own age/rank/round/pick/conference codes) ──
ROUND_DESC = {'S': 'top-5 pick', 'T': 'pick 6-10', 'U': 'pick 11-15', 'V': 'pick 16-20',
              'A': 'back of Rd 1', 'B': 'Rd 2-3', 'C': 'Rd 4-6', 'D': 'Rd 7-10',
              'E': 'Rd 11-15', 'F': 'Rd 16+'}
AGE_DESC = {'G': 'under-18', 'H': '18yo', 'I': '19yo', 'J': '20yo', 'K': '21yo',
            'L': '22yo', 'M': '23yo', 'N': '24+'}
DIST_DESC = {'O': 'slid 50+ past his rank', 'P': 'went near his rank',
             'Q': 'reach of 1-50 vs rank', 'R': 'reach of 50+ vs rank'}
SCHOOL2CONF = {s: c for s, c in zip(history['school'], history['conference']) if s and c}


def _code_rate(c):
    pct = trends['code_pct']
    v = pct.get(c)
    if v is None and c in ('L', 'M'):
        v = pct.get('L&M')
    return v


def why_expected(e):
    """One tweet line explaining the expected $ from the model's own codes."""
    c_d, c_r, c_a = e.get('_codes', ('', '', ''))
    if not (e['_slot'] and e['_exp'] and c_d and c_r and c_a):
        return ''
    diff = e['_exp'] / e['_slot'] - 1
    profile = f"{AGE_DESC.get(c_a, '?')}, {ROUND_DESC.get(c_r, '?')}, {DIST_DESC.get(c_d, '?')}"
    if trends['composite'].get(f'{c_d}{c_r}{c_a}') is not None:
        return (f"🧮 Why ~{_money(e['_exp'])}: {profile} — that exact profile signed "
                f"{diff:+.0%} vs slot across the '21-'25 drafts")
    parts = []
    for c, desc in ((c_d, DIST_DESC.get(c_d, '?')), (c_r, ROUND_DESC.get(c_r, '?')),
                    (c_a, AGE_DESC.get(c_a, '?'))):
        v = _code_rate(c)
        if v is not None:
            parts.append(f"{desc} {v:+.0%}")
    return (f"🧮 Why ~{_money(e['_exp'])}: " + ' · '.join(parts) +
            " — summed '21-'25 signing rates") if parts else ''


def find_comps(e, mrow, n=3):
    """Historical picks with the SAME age/rank/round/pick codes; same conference first."""
    c_d, c_r, c_a = e.get('_codes', ('', '', ''))
    if not (c_d and c_r and c_a):
        return []
    h = history[(history['code_dist'] == c_d) & (history['code_round'] == c_r) &
                (history['code_age'] == c_a) & (history['signed'] == 'Yes')].copy()
    if not len(h):
        return []
    h['bonus_n'] = pd.to_numeric(h['signing_bonus'], errors='coerce')
    h['slot_n'] = pd.to_numeric(h['slot_value'], errors='coerce')
    h['pick_n'] = pd.to_numeric(h['pick'], errors='coerce')
    h = h[(h['bonus_n'] > 0) & (h['slot_n'] > 0) & h['pick_n'].notna()]
    h = h[h['player'].str.lower() != str(e['Player']).lower()]
    if not len(h):
        return []
    conf = SCHOOL2CONF.get((mrow['school'] if mrow is not None else '') or '', '')
    h['same_conf'] = (h['conference'] == conf) & (conf != '')
    h['pickdiff'] = (h['pick_n'] - (e['Pick'] or 0)).abs()
    h = h.sort_values(['same_conf', 'pickdiff', 'year'], ascending=[False, True, False])
    out = []
    for _, r in h.head(n).iterrows():
        d = (r['bonus_n'] - r['slot_n']) / r['slot_n']
        conf_tag = f" {r['conference']}" if r['conference'] else ''
        out.append(f"{r['player']} ('{str(r['year'])[2:]}{conf_tag}, pick {int(r['pick_n'])}): "
                   f"{_money(r['bonus_n'])} vs {_money(r['slot_n'])} slot ({d:+.0%})")
    return out


def make_tweet(e, mrow):
    """Punchy draft tweet from an enriched pick + its board row."""
    name = e['Player']
    head = f"⚾ Pick {e['Pick']} — {e['Team']} select {name}"
    if e['Pos']:
        head += f", {e['Pos']}"
    if e['School']:
        head += f", {e['School']}"
    body = []
    pid = norm(mrow['player_id_64a']) if (mrow is not None and mrow['player_id_64a']) else ''
    stat = stat_line_2026(pid)
    if stat:
        body.append(stat)
    # rank-vs-pick value angle — OUR cumulative board rank (avg of all sources,
    # same number the expected-$ model uses); fall back to a single source only
    # if the player is unnumbered on our board.
    if mrow is not None:
        if mrow['number']:
            rk, unranked = int(float(mrow['number'])), False
        else:
            rk, unranked = UNRANKED_NUM, True   # not ranked by any source -> n+1
        lab = 'Our board'
        if rk:
            # d = rank - pick (engine convention): POSITIVE = ranked worse than
            # the pick = team reached; NEGATIVE = ranked better = player slid.
            d = rk - e['Pick']
            tag = f"  (reach of {d})" if d >= 15 else (f"  (slides {-d})" if d <= -15 else '')
            body.append(f"{lab}: #{rk}{' (unranked)' if unranked else ''} · went {e['Pick']}{tag}")
    if e['_slot']:
        m = f"💰 slot {_money(e['_slot'])}"
        if e['_exp']:
            m += f" · model says ~{_money(e['_exp'])}"
        body.append(m)
    why = why_expected(e)
    if why:
        body.append(why)
    comps = find_comps(e, mrow)
    if comps:
        body.append("📊 Model comps (same age/rank/pick profile):")
        body.extend(f"  • {c}" for c in comps)
    # portal / commitment note
    note = ''
    if pid and pid in PORTAL:
        if '2026' in PORTAL[pid]:
            pi = PORTAL[pid]['2026']
            note = f"🔁 In the 2026 portal from {pi['from']}" + (f" — was committed to {pi['to']}" if pi['to'] else " (uncommitted)")
        elif '2025' in PORTAL[pid]:
            pi = PORTAL[pid]['2025']
            note = f"🔁 Transferred via the 2025 portal" + (f" ({pi['from']} → {pi['to']})" if pi['to'] else f" from {pi['from']}")
    elif mrow is not None and mrow['committed']:
        note = f"🎓 Committed to {mrow['committed']}"
    if note:
        body.append(note)
    return head + "\n\n" + "\n".join(body)


def master_row(name):
    m = master[master['name'] == name]
    return m.iloc[0] if len(m) else None


# ── fallback enrichment for picks NOT on our board: bio from MLB StatsAPI +
#    a strict players.csv link for 4-year players (last name + school + first-
#    name prefix must ALL agree — never fuzzy-guess an identity) ──
_MROW_KEYS = ['name', 'number', 'overall_avg', 'pos', 'bt', 'classification', 'school',
              'committed', 'dob', 'player_id_64a', 'rank_ba', 'rank_mlb', 'rank_espn',
              'rank_fss', 'pg_rank', 'ht', 'wt', 'hs']


@st.cache_data(ttl=120, show_spinner=False)
def mlb_pick_info():
    """pick number -> bio dict from the live MLB draft feed."""
    try:
        r = requests.get(f'https://statsapi.mlb.com/api/v1/draft/{YEAR}', timeout=15)
        r.raise_for_status()
        out = {}
        for rd in r.json()['drafts']['rounds']:
            for p in rd.get('picks', []):
                if not (p.get('isDrafted') and p.get('person', {}).get('fullName')):
                    continue
                per, sch = p['person'], p.get('school', {}) or {}
                out[int(p['pickNumber'])] = {
                    'name': per['fullName'],
                    'pos': (per.get('primaryPosition') or {}).get('abbreviation', ''),
                    'bt': f"{(per.get('batSide') or {}).get('code', '')}/{(per.get('pitchHand') or {}).get('code', '')}".strip('/'),
                    'dob': per.get('birthDate', ''),
                    'school': sch.get('name', ''),
                    'classification': sch.get('schoolClass', ''),
                }
        return out
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _players_slim():
    pl = pd.read_csv(DATA / 'players.csv', dtype=str, encoding='latin-1', keep_default_na=False,
                     usecols=['id', 'player_name', 'position', 'team_id'])
    tm = pd.read_csv(DATA / 'teams.csv', dtype=str, keep_default_na=False, usecols=['id', 'name', 'sport'])
    tm = tm[tm['sport'] == 'Baseball']
    pl['team_name'] = pl['team_id'].map(dict(zip(tm['id'].map(norm), tm['name'])))
    return pl


def _match_pid(mlb_name, school):
    """players.csv id for a 4YR draftee — exactly one candidate must pass ALL of:
    same last name, same school (normalized), first-name prefix agreement
    (handles Ty/Tyner, Greg/Gregory). Ambiguous or no match -> ''."""
    az = lambda s: re.sub(r'[^a-z]', '', str(s).lower())
    parts = str(mlb_name).split()
    if len(parts) < 2 or not school:
        return ''
    first, last = az(parts[0]), az(parts[-1])
    pl = _players_slim()
    cand = pl[pl['player_name'].map(lambda n: az(str(n).split()[-1]) == last if str(n).split() else False)]
    cand = cand[cand['team_name'].fillna('').map(az) == az(school)]
    cand = cand[cand['player_name'].map(
        lambda n: az(str(n).split()[0]).startswith(first[:2]) or first.startswith(az(str(n).split()[0])[:2]))]
    return norm(cand['id'].iloc[0]) if len(cand) == 1 else ''


def resolve_row(name, pick_no):
    """Board row if the player is on it; otherwise a pseudo-row built from the
    MLB feed (bio) + players.csv (64A identity for 4-year players)."""
    mrow = master_row(name)
    if mrow is not None:
        return mrow
    info = mlb_pick_info().get(int(pick_no or 0))
    if not info or info['name'] != name and norm(info['name']) != norm(name):
        return None
    row = {k: '' for k in _MROW_KEYS}
    row.update(info)
    if str(info.get('classification', '')).startswith('4YR'):
        row['player_id_64a'] = _match_pid(info['name'], info['school'])
    return pd.Series(row)


def enrich_pick(p):
    """One Supabase pick row -> full computed dict (the Excel formula columns)."""
    mrow = resolve_row(p.get('player_name', ''), p.get('pick'))
    slot = fnum(p.get('slot_value'))
    bonus = fnum(p.get('signing_bonus'))
    pick_no = int(p.get('pick') or 0)
    rnd = int(p.get('round') or 1)
    out = {
        'Pick': pick_no, 'Rd': rnd, 'Team': p.get('team', ''), 'Player': p.get('player_name', ''),
        'Board rank': '', 'Pos': '', 'B/T': '', 'Class': '', 'School': '', 'Committed': '',
    }
    board_rank = None
    age = None
    if mrow is not None:
        board_rank = fnum(mrow['number']) if str(mrow['number']).strip() else UNRANKED_NUM
        out.update({'Board rank': int(board_rank) if board_rank else '',
                    'Pos': mrow['pos'], 'B/T': mrow['bt'], 'Class': mrow['classification'],
                    'School': mrow['school'], 'Committed': mrow['committed']})
        age = E.age_years(mrow['dob'], DRAFT_DAY) if mrow['dob'] else None
    dist = (board_rank - pick_no) if (board_rank and pick_no) else None
    c_r = E.code_round(pick_no, rnd)
    c_a = E.code_age(age) if age is not None else ''
    c_d = E.code_dist(dist) if dist is not None else ''
    exp = E.expected_signing(slot, c_d, c_r, c_a, trends) if (slot and c_d and c_a) else None
    pool_round = rnd <= 10
    impact = E.pool_impact(bonus is not None, pool_round, bonus)
    diff = E.diff_from_slot(bonus, slot, pool_round)
    out.update({
        'Age': f'{age:.1f}' if age is not None else '',
        'Slot $': f'{slot:,.0f}' if slot else '',
        'Expected $': f'{exp:,.0f}' if exp else '',
        'Bonus $': f'{bonus:,.0f}' if bonus is not None else '—',
        'Pool impact': f'{impact:,.0f}' if impact is not None else '',
        'vs slot': f'{diff:+.1%}' if diff is not None else '',
        '_slot': slot or 0, '_exp': exp or 0, '_impact': impact or 0, '_bonus': bonus,
        '_codes': (c_d, c_r, c_a), '_id': p.get('id'),
    })
    return out


tab_live, tab_feed, tab_board, tab_super, tab_card = st.tabs(
    ["🎙 Live Draft Tracker", "🐦 Tweet Feed", "📋 Prospect Board", "🏆 Superlatives", "🔍 Scouting Card"])

# ═══════════════════════ TAB 1: LIVE DRAFT TRACKER ═══════════════════════
with tab_live:
    st.title(f"{YEAR} Draft — Live Tracker")
    picks, err = fetch_picks()

    if err is not None:
        st.error("The draft_picks table isn't reachable yet. Run the setup SQL in Supabase "
                 "(SQL Editor) — see DRAFT_SETUP.md in the repo — then reload.")
        st.code(err)
    else:
        # ---- entry form ----
        with st.expander("➕ Enter a pick", expanded=True):
            c1, c2, c3 = st.columns([1, 1, 2])
            pick_in = c2.number_input("Pick #", 1, 700, int(picks['pick'].max()) + 1 if len(picks) else 1)
            _auto_rnd = round_by_pick.get(int(pick_in))
            if _auto_rnd is not None:
                rnd_in = _auto_rnd
                c1.metric("Round (auto)", rnd_in)
            else:
                rnd_in = c1.number_input("Round", 1, 20, 20)
            team_default = team_by_pick.get(int(pick_in), '')
            team_idx = MLB_TEAMS.index(team_default) if team_default in MLB_TEAMS else 0
            team_in = c3.selectbox("MLB team (auto-set from the pick's slot)", MLB_TEAMS, index=team_idx)
            q = st.text_input("Player (search the board)", key='pick_search')
            cands = master[master['name'].str.lower().str.contains(q.strip().lower(), na=False)] if q.strip() else master.head(0)
            options = [f"{r['name']}  ·  #{r['number']} {r['pos']} — {r['school']}" for _, r in cands.head(30).iterrows()]
            options.append('— not on the board (type name below) —')
            sel = st.selectbox("Match", options, key='pick_match') if options else None
            manual_name = ''
            if sel == '— not on the board (type name below) —':
                manual_name = st.text_input("Player name (verbatim)", key='pick_manual')
            c4, c5, c6 = st.columns(3)
            slot_default = slot_by_pick.get(int(pick_in), 0)
            slot_in = c4.number_input("Slot value $ (official 2026)",
                                      0, 20_000_000, int(slot_default), step=100)
            bonus_in = c5.number_input("Signing bonus $ (0 = unsigned)", 0, 20_000_000, 0, step=100)
            entered_by = c6.text_input("Your name", key='pick_by')
            if st.button("Record pick", type='primary'):
                name = manual_name.strip() if manual_name.strip() else (sel or '').split('  ·  ')[0]
                if not name:
                    st.warning("Pick a player first.")
                else:
                    payload = {'year': YEAR, 'round': int(rnd_in), 'pick': int(pick_in),
                               'team': team_in, 'player_name': name,
                               'slot_value': int(slot_in) or None,
                               'signing_bonus': int(bonus_in) or None,
                               'entered_by': entered_by}
                    r = requests.post(f'{SUPABASE_URL}/rest/v1/{PICKS_TABLE}',
                                      headers={**HEADERS, 'Prefer': 'return=minimal'},
                                      json=payload, timeout=12)
                    if r.status_code in (200, 201):
                        st.success(f"Pick {pick_in}: {name} → {team_in}")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(f'{r.status_code}: {r.text[:200]}')

        # ---- record a signing for an existing pick ----
        if len(picks):
            with st.expander("✍️ Record a signing bonus"):
                unsigned = picks[picks['signing_bonus'].isna()] if 'signing_bonus' in picks.columns else picks
                lab = {f"#{int(r['pick'])} {r['player_name']} ({r['team']})": r['id'] for _, r in unsigned.iterrows()}
                if lab:
                    pick_sel = st.selectbox("Pick", list(lab))
                    b = st.number_input("Bonus $", 0, 20_000_000, 0, step=100, key='sign_b')
                    if st.button("Save bonus"):
                        r = requests.patch(f'{SUPABASE_URL}/rest/v1/{PICKS_TABLE}',
                                           headers={**HEADERS, 'Prefer': 'return=minimal'},
                                           params={'id': f'eq.{lab[pick_sel]}'},
                                           json={'signing_bonus': int(b)}, timeout=12)
                        if r.status_code in (200, 204):
                            st.success("Saved."); st.rerun()
                        else:
                            st.error(f'{r.status_code}: {r.text[:200]}')
                else:
                    st.caption("All recorded picks have bonuses.")

        # ---- live table + pool ----
        if len(picks):
            rows = [enrich_pick(p) for _, p in picks.iterrows()]
            df = pd.DataFrame(rows)
            show = ['Pick', 'Rd', 'Team', 'Player', 'Board rank', 'Pos', 'B/T', 'Age', 'Class',
                    'School', 'Committed', 'Slot $', 'Expected $', 'Bonus $', 'Pool impact', 'vs slot']
            st.markdown(f"### Picks so far ({len(df)})")
            st.dataframe(df[show], use_container_width=True, height=420, hide_index=True)

            st.markdown("### Bonus-pool usage by team")
            pool = df.groupby('Team').agg(picks=('Pick', 'count'), slot=('_slot', 'sum'),
                                          expected=('_exp', 'sum'), impact=('_impact', 'sum')).reset_index()
            # theoretical money available = Σ slot - Σ expected (positive = the
            # model expects under-slot signings -> pool money freed to spend)
            pool['avail'] = pool['slot'] - pool['expected']
            # keep money columns NUMERIC (in $M) so header-click sorting works —
            # comma-formatted strings sort alphabetically ("9,7.." > "10,5..")
            for c in ('slot', 'expected', 'impact', 'avail'):
                pool[c] = (pool[c] / 1e6).round(3)
            pool = pool.sort_values('avail', ascending=False)
            pool.columns = ['Team', 'Picks', 'Σ slot $M', 'Σ expected $M',
                            'Σ pool impact $M', 'Theoretical $M available']
            st.dataframe(pool, use_container_width=True, hide_index=True)
            st.caption("Theoretical available = Σ slot − Σ expected: what the signing-trends "
                       "model expects each class to free up vs full slot. Columns sort numerically.")

            # ---- tweet for the latest pick (copy button lives on the code block) ----
            st.markdown("### 🐦 Latest pick, tweet-ready")
            latest = max(rows, key=lambda r: r['Pick'])
            st.code(make_tweet(latest, resolve_row(latest['Player'], latest['Pick'])), language=None)
            st.caption("Copy straight into X. Every pick's tweet lives in the Tweet Feed tab.")
        else:
            st.info("No picks recorded yet — the board is live and waiting.")

# ═══════════════════════ TAB 2: TWEET FEED ═══════════════════════
with tab_feed:
    st.title("Tweet Feed")
    st.caption("One ready-to-post tweet per recorded pick, newest first — catch up any time.")
    picks_f, err_f = fetch_picks()
    if err_f is not None or not len(picks_f):
        st.info("Tweets appear here as picks are recorded.")
    else:
        rows_f = [enrich_pick(p) for _, p in picks_f.iterrows()]
        for e in sorted(rows_f, key=lambda r: -r['Pick']):
            st.code(make_tweet(e, resolve_row(e['Player'], e['Pick'])), language=None)

# ═══════════════════════ TAB 2: PROSPECT BOARD ═══════════════════════
with tab_board:
    st.title("Prospect Board")
    st.caption("The Master Document, live: computed age, drafted flags, ranks from BA / MLB / FSS / ESPN.")
    picks_b, _ = fetch_picks()
    drafted_names = set(picks_b['player_name']) if len(picks_b) else set()

    b1, b2, b3 = st.columns([2, 1, 1])
    q = b1.text_input("Search name / school / committed", key='board_q').strip().lower()
    pos_f = b2.multiselect("Pos", sorted({p for p in master['pos'].unique() if p}))
    hide_drafted = b3.checkbox("Hide drafted")

    board = master.copy()
    board['Age'] = board['dob'].map(lambda d: f'{E.age_years(d, DRAFT_DAY):.1f}' if d else '')
    board['Drafted'] = board['name'].map(lambda n: '✓' if n in drafted_names else '')
    if q:
        mask = (board['name'].str.lower().str.contains(q, na=False)
                | board['school'].str.lower().str.contains(q, na=False)
                | board['committed'].str.lower().str.contains(q, na=False))
        board = board[mask]
    if pos_f:
        board = board[board['pos'].isin(pos_f)]
    if hide_drafted:
        board = board[board['Drafted'] == '']
    board['num'] = pd.to_numeric(board['number'], errors='coerce')
    board = board.sort_values('num')
    cols = {'number': '#', 'overall_avg': 'Cume avg', 'name': 'Player', 'pos': 'Pos', 'bt': 'B/T', 'Age': 'Age',
            'classification': 'Class', 'school': 'School', 'committed': 'Committed',
            'rank_ba': 'BA', 'rank_mlb': 'MLB', 'rank_fss': 'Over-Slot', 'rank_espn': 'ESPN', 'pg_rank': 'PG',
            'Drafted': 'Drafted'}
    st.dataframe(board[list(cols)].rename(columns=cols), use_container_width=True,
                 height=560, hide_index=True)
    st.caption(f"{len(board):,} prospects shown of {len(master):,} on the board.")

# ═══════════════════════ TAB 4: SUPERLATIVES ═══════════════════════
with tab_super:
    st.title("Draft Superlatives")
    st.caption("Quick splits on the picks so far: source (HS / 4-year / JC), portal movement, "
               "broken commitments, and the best names still on the board.")
    picks_s, err_s = fetch_picks()
    if err_s is not None or not len(picks_s):
        st.info("Superlatives appear once picks are recorded.")
    else:
        srecs = []
        for _, p in picks_s.iterrows():
            mrow = resolve_row(p.get('player_name', ''), p.get('pick'))
            cls = mrow['classification'] if mrow is not None else ''
            pid = norm(mrow['player_id_64a']) if mrow is not None else ''
            src = ('HS' if cls.startswith('HS') else '4YR' if cls.startswith('4YR')
                   else 'JC' if cls.startswith('JC') else '—')
            cyc, pinfo = '', None
            if pid and pid in PORTAL:
                cyc = '2026' if '2026' in PORTAL[pid] else '2025'
                pinfo = PORTAL[pid][cyc]
            srecs.append({'Pick': int(p.get('pick') or 0), 'MLB team': p.get('team', ''),
                          'Player': p.get('player_name', ''), 'Class': cls, 'src': src,
                          'Pos': mrow['pos'] if mrow is not None else '',
                          'School': mrow['school'] if mrow is not None else '',
                          'Committed': mrow['committed'] if mrow is not None else '',
                          'cycle': cyc,
                          'p_from': pinfo['from'] if pinfo else '',
                          'p_to': pinfo['to'] if pinfo else ''})
        sdf = pd.DataFrame(srecs).sort_values('Pick')
        n_hs = int((sdf['src'] == 'HS').sum())
        n_4yr = int((sdf['src'] == '4YR').sum())
        n_jc = int((sdf['src'] == 'JC').sum())
        n_unk = int((sdf['src'] == '—').sum())
        n_portal = int((sdf['cycle'] != '').sum())

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Picks so far", len(sdf))
        m2.metric("High school", n_hs)
        m3.metric("4-year college", n_4yr)
        m4.metric("Juco", n_jc)
        m5.metric("Portal players", n_portal)
        if n_unk:
            st.caption(f"{n_unk} pick(s) aren't matched to the board (manual names) and are "
                       "excluded from the splits.")

        # ── vs previous drafts: same splits through the SAME pick number ──
        st.markdown(f"### 📅 vs previous drafts (through pick {int(sdf['Pick'].max())})")
        _maxp = int(sdf['Pick'].max())
        _h = history.drop_duplicates(subset=['year', 'pick', 'player']).copy()
        _h['pick_n'] = pd.to_numeric(_h['pick'], errors='coerce')
        _h = _h[_h['pick_n'] <= _maxp]
        yr_rows = []
        for yr, g in _h.groupby('year'):
            cls = g['classification'].fillna('')
            hs_n = int((g['hs'] == 'True').sum())
            c4_n = int(cls.str.startswith('4YR').sum())
            jc_n = int((g['juco'] == 'Yes').sum())
            yr_rows.append({'Draft': yr, 'Picks': len(g), 'HS': hs_n,
                            '4-year': c4_n, 'JC': jc_n,
                            'HS %': f"{hs_n / len(g):.0%}" if len(g) else '',
                            '4-year %': f"{c4_n / len(g):.0%}" if len(g) else ''})
        n_matched = len(sdf) - n_unk
        yr_rows.append({'Draft': f'{YEAR} (live)', 'Picks': len(sdf), 'HS': n_hs,
                        '4-year': n_4yr, 'JC': n_jc,
                        'HS %': f"{n_hs / n_matched:.0%}" if n_matched else '',
                        '4-year %': f"{n_4yr / n_matched:.0%}" if n_matched else ''})
        st.dataframe(pd.DataFrame(yr_rows), use_container_width=True, hide_index=True)
        _avg_hs = sum(r['HS'] for r in yr_rows[:-1]) / max(len(yr_rows) - 1, 1)
        _d = n_hs - _avg_hs
        st.caption(f"Through pick {_maxp}, the {YEAR} draft has {n_hs} HS picks vs a "
                   f"{_avg_hs:.1f} average for 2021-2025 at the same point "
                   f"({'+' if _d >= 0 else ''}{_d:.1f} vs trend).")

        st.markdown("### 🔁 Portal players drafted — and where they were headed")
        pdf = sdf[sdf['cycle'] != ''].copy()
        if len(pdf):
            pdf['Portal'] = pdf.apply(
                lambda r: f"{r['cycle']}: {r['p_from']} → " +
                          (r['p_to'] if r['p_to'] else ('uncommitted' if r['cycle'] == '2026' else '?')),
                axis=1)
            st.dataframe(pdf[['Pick', 'MLB team', 'Player', 'Pos', 'Class', 'Portal']],
                         use_container_width=True, hide_index=True)
            going = pdf[(pdf['cycle'] == '2026') & (pdf['p_to'] != '')]['p_to'].value_counts()
            if len(going):
                st.caption("2026-portal destinations losing a commit to the draft: "
                           + " · ".join(f"{k} ({v})" for k, v in going.items()))
        else:
            st.caption("No portal players drafted yet.")

        st.markdown("### 🎓 HS picks — where they were supposed to go")
        hdf = sdf[sdf['src'] == 'HS'].copy()
        if len(hdf):
            hdf['Committed to'] = hdf['Committed'].replace('', '—')
            st.dataframe(hdf[['Pick', 'MLB team', 'Player', 'Pos', 'School', 'Committed to']],
                         use_container_width=True, hide_index=True)
            lost = hdf[hdf['Committed'] != '']['Committed'].value_counts()
            if len(lost):
                st.caption("Programs losing the most HS commits: "
                           + " · ".join(f"{k} ({v})" for k, v in lost.head(12).items()))
        else:
            st.caption("No high-school picks yet.")

        st.markdown("### 💎 Top players still on the board")
        drafted_s = set(sdf['Player'])
        left = master[~master['name'].isin(drafted_s)].copy()
        left['num'] = pd.to_numeric(left['number'], errors='coerce')
        left = left[left['num'].notna()].sort_values('num').head(25)
        left_cols = {'number': '#', 'name': 'Player', 'pos': 'Pos', 'classification': 'Class',
                     'school': 'School', 'committed': 'Committed'}
        st.dataframe(left[list(left_cols)].rename(columns=left_cols),
                     use_container_width=True, hide_index=True)
        st.caption("Best 25 undrafted by cumulative board rank — the full list lives on the Prospect Board tab.")

# ═══════════════════════ TAB 3: SCOUTING CARD (original, verbatim) ═══════════════════════
with tab_card:

    players = load("players.csv")
    teams = load("teams.csv")
    conf = load("conferences.csv")
    hit = hit_all.copy()
    pit = pit_all.copy()
    try:
        series = load("draft_best_series.csv")
    except Exception:
        series = pd.DataFrame()
    try:
        splits = load("draft_splits.csv")
    except Exception:
        splits = pd.DataFrame()
    for df in (hit, pit):
        df["pid"] = df["player_id"].map(norm)
    splits_by_pid = {}
    if len(splits):
        for r in splits.itertuples():
            splits_by_pid.setdefault(norm(r.player_id), []).append(r)
    tname = {norm(r["id"]): r["name"] for _, r in teams.iterrows()}
    cdiv = {norm(r["id"]): r["division"] for _, r in conf.iterrows()}
    tconf = {norm(r["id"]): norm(r["conference_id"]) for _, r in teams.iterrows()}
    series_by_pid = {norm(r.player_id): r for r in series.itertuples()
                     if r.kind != 'outing'} if len(series) else {}
    outing_by_pid = {norm(r.player_id): r for r in series.itertuples()
                     if r.kind == 'outing'} if len(series) else {}

    def f(x):
        try:
            return float(x)
        except Exception:
            return None

    def num(df, col):
        return pd.to_numeric(df[col], errors="coerce")

    # draft-board tie-in appears after the player is chosen (inserted below)

    st.title("Draft Day Assistant")
    st.caption("Live-stream scouting card for any baseball player. Search by name.")

    q = st.text_input("Player search", placeholder="Type a player name…").strip().lower()
    if not q:
        st.info("Start typing a player name above.")
        st.stop()

    matches = players[players["player_name"].str.lower().str.contains(q, na=False)]
    # prefer players who actually have stats (draftable), and de-dupe by id
    matches = matches.drop_duplicates("id")
    if matches.empty:
        st.warning("No players match that name.")
        st.stop()


    def _label(r):
        t = tname.get(norm(r["team_id"]), "")
        pos = r.get("position", "")
        return f'{r["player_name"]} — {t} ({pos})' if t else f'{r["player_name"]} ({pos})'


    opts = {_label(r): norm(r["id"]) for _, r in matches.head(60).iterrows()}
    choice = st.selectbox(f"{len(matches)} match(es)", list(opts))
    pid = opts[choice]
    prow = players[players["id"].map(norm) == pid].iloc[0]

    # ---------------- gather this player's stat rows ----------------
    ph = hit[hit["pid"] == pid].copy()
    pp = pit[pit["pid"] == pid].copy()
    ph["yr"] = num(ph, "year")
    pp["yr"] = num(pp, "year")

    bat = (prow.get("bat") or "").strip().upper()
    throw = (prow.get("throw") or "").strip().upper()
    BAT_LABEL = {"L": "left-handed", "R": "right-handed", "S": "switch"}
    THROW_LABEL = {"L": "left-handed", "R": "right-handed"}

    # career aggregates (hitting)
    def csum(df, col):
        return int(num(df, col).fillna(0).sum()) if len(df) else 0

    car_ab = csum(ph, "at_bats"); car_h = csum(ph, "hits")
    car_2b = csum(ph, "doubles"); car_3b = csum(ph, "triples"); car_hr = csum(ph, "home_runs")
    car_bb = csum(ph, "walks"); car_hbp = csum(ph, "hit_by_pitch"); car_sf = csum(ph, "sac_fly")
    car_rbi = csum(ph, "runs_batted_in"); car_sb = csum(ph, "stolen_bases"); car_gp = csum(ph, "games_played")
    car_tb = car_h - car_2b - car_3b - car_hr + 2 * car_2b + 3 * car_3b + 4 * car_hr
    car_xbh = car_2b + car_3b + car_hr
    car_pa = car_ab + car_bb + car_hbp + car_sf
    car_obp = (car_h + car_bb + car_hbp) / car_pa if car_pa else 0
    car_slg = car_tb / car_ab if car_ab else 0
    car_ops = round(car_obp + car_slg, 3)
    car_avg = round(car_h / car_ab, 3) if car_ab else 0

    # career pitching
    car_ip = round(num(pp, "innings_pitched").fillna(0).sum(), 1) if len(pp) else 0
    car_bf = csum(pp, "batters_faced")
    car_k = csum(pp, "strikeouts")

    predominant = "pitching" if car_bf > car_pa else "hitting"

    # schools played (distinct teams across stat rows, in year order)
    school_years = {}
    for _, r in pd.concat([ph, pp]).iterrows():
        t = tname.get(norm(r["team_id"]))
        if t:
            school_years.setdefault(t, set()).add(norm(r["year"]))
    schools = sorted(school_years, key=lambda t: min(school_years[t]))
    transferred = (str(prow.get("transferred", "")).strip().upper() in ("TRUE", "1", "YES")) or len(schools) > 1

    # ---------------- 1) OVERVIEW ----------------
    team_now = tname.get(norm(prow["team_id"]), "")
    hand = f"{BAT_LABEL.get(bat, bat or '?')}/{THROW_LABEL.get(throw, throw or '?')}"
    st.markdown(f"## {prow['player_name']}")
    mline = master[master["player_id_64a"].map(norm) == pid]
    if len(mline):
        m = mline.iloc[0]
        st.success(f"On the draft board: **#{m['number']}** overall - BA {m['rank_ba'] or '-'} / "
                   f"MLB {m['rank_mlb'] or '-'} / FSS {m['rank_fss'] or '-'} / ESPN {m['rank_espn'] or '-'}"
                   + (f" - committed to **{m['committed']}**" if m['committed'] else ""))

    # portal status — broadcasters want to know this the moment the card opens
    if pid in PORTAL:
        if '2026' in PORTAL[pid]:
            pi = PORTAL[pid]['2026']
            dest = f"committed to **{pi['to']}**" if pi['to'] else "**uncommitted**"
            st.warning(f"🔁 **IN THE 2026 PORTAL** — from {pi['from']}, {dest}")
        elif '2025' in PORTAL[pid]:
            pi = PORTAL[pid]['2025']
            st.info(f"🔁 2025 portal alum — {pi['from']} → {pi['to'] or '?'} (not in the 2026 portal)")
    else:
        st.caption("Not in the transfer portal (2025-26 cycles).")

    st.markdown(
        f"**{prow.get('position','')}** · bats {BAT_LABEL.get(bat, bat or '?')}, throws {THROW_LABEL.get(throw, throw or '?')}"
        + (f" · {prow.get('height','')}" if prow.get('height') else "")
        + (f" · {prow.get('hometown','')}" if prow.get('hometown') else "")
    )

    c = st.columns(5)
    c[0].metric("Career OPS", f"{car_ops:.3f}")
    c[1].metric("Games", car_gp or "—")
    c[2].metric("Career HR", car_hr)
    c[3].metric("Extra-base hits", car_xbh)
    c[4].metric("Career AVG", f"{car_avg:.3f}")

    over = []
    yrs = sorted({norm(y) for y in pd.concat([ph, pp])["year"]})
    over.append(f"**{len(schools)} school{'s' if len(schools)!=1 else ''}:** " +
                " → ".join(schools) + (f"  ·  _transfer_" if transferred else "  ·  _no transfers on record_"))
    over.append(f"**Seasons in data:** {', '.join(yrs)}")
    # superlatives
    sup = []
    if len(ph):
        best = ph.loc[num(ph, "on_base_plus_slugging").idxmax()] if num(ph, "on_base_plus_slugging").notna().any() else None
        if best is not None:
            sup.append(f"best season OPS **{f(best['on_base_plus_slugging']):.3f}** ({norm(best['year'])} at {tname.get(norm(best['team_id']),'?')})")
        if car_hr >= 30:
            sup.append(f"**{car_hr}** career home runs")
        if car_sb >= 30:
            sup.append(f"**{car_sb}** career steals")
        if car_xbh >= 60:
            sup.append(f"**{car_xbh}** career extra-base hits")
    if car_ip >= 20:
        sup.append(f"**{car_ip}** career IP, {car_k} K on the mound")
    if sup:
        over.append("**Superlatives:** " + "; ".join(sup) + ".")
    st.markdown("\n\n".join(over))

    st.divider()

    # ---------------- 2) BIG WEEKEND / BIG PERFORMANCE ----------------
    # Pitchers throw once a series — a 3-game weekend line doesn't fit them.
    # They get their best single OUTING instead (kind='outing' in the CSV).
    if predominant == "pitching" and outing_by_pid.get(pid) is not None:
        st.markdown("### 🔥 Big performance")
        o = outing_by_pid[pid]
        st.success(o.summary)
        st.caption(f"{o.date_start} · biggest single outing on record (from per-game PBP)")
    elif predominant == "pitching":
        st.markdown("### 🔥 Big performance")
        st.caption("No qualifying outing on record for this pitcher (PBP covers 2025–2026; "
                   "3+ IP required; small-school arms may not have one).")
    else:
        st.markdown("### 🔥 Big weekend")
        s = series_by_pid.get(pid)
        if s is not None:
            kind = "series" if s.kind == "series" else "hot stretch"
            st.success(s.summary)
            st.caption(f"{s.date_start} to {s.date_end} · best 3-game {kind} on record (from per-game PBP)")
        else:
            st.caption("No multi-game PBP series on record for this player (PBP covers 2025–2026; "
                       "small-school or limited-sample players may not have one).")

    st.divider()

    # ---------------- 3) SOMETHING TO WATCH (weakness in predominant role) ----------------
    st.markdown("### 👀 Something to watch")
    HIT_PCT = {
        "percentile_rank_on_base_plus_slugging": "OPS",
        "percentile_rank_weighted_on_base_average": "wOBA",
        "percentile_rank_isolated_power": "ISO (power)",
        "percentile_rank_weighted_runs_created": "wRC",
        "percentile_rank_runs_plate_appearance": "runs/PA",
    }
    PIT_PCT = {
        "percentile_rank_on_base_plus_slugging_against": "OPS allowed",
        "percentile_rank_strikeout_to_walk_ratio": "K/BB",
        "percentile_rank_walks_plus_hits_per_inning_pitched": "WHIP",
        "percentile_rank_fielding_independent_pitching": "FIP",
        "percentile_rank_skill_interactive_earned_run_average": "SIERA",
    }


    def latest_with_volume(df, vol_col, min_vol):
        d = df[num(df, vol_col).fillna(0) >= min_vol]
        if not len(d):
            return None
        return d.loc[d["yr"].idxmax()]

    weak = []
    if predominant == "hitting":
        row = latest_with_volume(ph, "plate_appearances", 50)
        if row is not None:
            for col, lab in HIT_PCT.items():
                v = f(row.get(col))
                if v is not None and v < 0.60:
                    weak.append((lab, v))
            ctx = f"{norm(row['year'])} · {int(f(row['plate_appearances']))} PA"
        else:
            ctx = None
    else:
        row = latest_with_volume(pp, "innings_pitched", 15)
        if row is not None:
            for col, lab in PIT_PCT.items():
                v = f(row.get(col))
                if v is not None and v < 0.60:
                    weak.append((lab, v))
            ctx = f"{norm(row['year'])} · {f(row['innings_pitched']):.0f} IP"
        else:
            ctx = None

    if ctx is None:
        st.caption(f"Not enough {predominant} volume to flag a weakness.")
    elif weak:
        weak.sort(key=lambda x: x[1])
        st.markdown(f"Below the 60th percentile in his **{predominant}** ({ctx}):")
        st.markdown("\n".join(f"- **{lab}** — {int(round(v*100))}th percentile" for lab, v in weak[:4]))
    else:
        st.markdown(f"No clear weakness — every tracked {predominant} metric is at or above the 60th percentile ({ctx}). "
                    "That's a positive note for the broadcast.")

    # --- 2026 split notes (peer-relative, volume-gated; both directions) ---
    SPLIT_LABEL = {"conference": "vs conference", "road": "on the road", "home": "at home",
                   "late": "late-season (May/Jun)", "quality": "vs power-conference foes"}
    LO, HI = 15, 85   # extreme tails


    def _i(x):
        try:
            return int(float(x))
        except Exception:
            return None

    # this player's overall 2026 power rate, for self-baseline contrasts
    ph26 = ph[ph["yr"] == 2026]
    ab26 = int(num(ph26, "at_bats").fillna(0).sum())
    hr26 = int(num(ph26, "home_runs").fillna(0).sum())
    ovr_hr_rate = hr26 / ab26 if ab26 else 0.0

    split_watch, split_good, seen = [], [], set()
    for r in splits_by_pid.get(pid, []):
        lab = SPLIT_LABEL.get(r.split, r.split)
        opc, isc = _i(r.ops_pct), _i(r.iso_pct)
        ops, iso = r.ops, r.iso
        pa, ab, hr = _i(r.pa), _i(r.ab), _i(r.hr)
        # 1) peer-outlier weakness: overall (OPS) first, else power (ISO)
        if opc is not None and opc <= LO:
            split_watch.append((opc, f"**{lab}**: {ops} OPS — {opc}th pct ({pa} PA)")); seen.add(r.split)
        elif isc is not None and isc <= LO:
            split_watch.append((isc, f"**{lab}** power: {hr} HR in {ab} AB ({iso} ISO, {isc}th pct)")); seen.add(r.split)
        # 2) self-baseline power contrast (e.g. 'only 2 HR vs conference' vs his own norm).
        #    Skip if this split is actually a strength (don't nitpick HR on an elite split).
        elif (ab and ab >= 60 and ovr_hr_rate >= 0.018 and ab26 >= 80 and hr is not None
              and not (opc is not None and opc >= HI)
              and (hr / ab) <= 0.70 * ovr_hr_rate):
            per_o = round(ab26 / hr26) if hr26 else None
            if hr == 0:
                txt = f"**{lab}**: 0 HR in {ab} AB (he's 1 per {per_o} overall)"
            else:
                txt = f"**{lab}**: just {hr} of his {hr26} HR — 1 per {round(ab/hr)} AB vs 1 per {per_o} overall"
            split_watch.append((20, txt)); seen.add(r.split)
        # 3) strength
        if opc is not None and opc >= HI:
            split_good.append((-opc, f"**{lab}**: {ops} OPS — {opc}th pct ({pa} PA)"))
        elif isc is not None and isc >= HI:
            split_good.append((-isc, f"**{lab}** power: {hr} HR in {ab} AB ({iso} ISO, {isc}th pct)"))

    if split_watch:
        split_watch.sort()
        st.markdown("**2026 splits worth flagging:**")
        st.markdown("\n".join(f"- {t}" for _, t in split_watch[:3]))

    st.divider()

    # ---------------- 3b) NOTABLE STRENGTHS ----------------
    st.markdown("### ✅ Notable strengths")
    strengths = []
    # top overall percentile metrics in predominant role
    if predominant == "hitting" and row is not None and ctx is not None:
        for col, lab in HIT_PCT.items():
            v = f(row.get(col))
            if v is not None and v >= 0.85:
                strengths.append((-v, f"**{lab}** — {int(round(v*100))}th percentile ({ctx})"))
    elif predominant == "pitching" and row is not None and ctx is not None:
        for col, lab in PIT_PCT.items():
            v = f(row.get(col))
            if v is not None and v >= 0.85:
                strengths.append((-v, f"**{lab}** — {int(round(v*100))}th percentile ({ctx})"))
    strengths.sort()
    lines = [t for _, t in strengths[:3]] + [t for _, t in sorted(split_good)[:3]]
    if lines:
        st.markdown("\n".join(f"- {t}" for t in lines))
    else:
        st.caption("No top-15% standout metrics or splits flagged for the latest season.")

    st.divider()

    # ---------------- 4) HANDEDNESS ----------------
    st.markdown("### 🤚 Handedness angle")


    def handed_pct(df, peer_df, metric, higher_better=True, label=""):
        """Percentile of this player's metric within the same-handedness peer group."""
        vals = pd.to_numeric(peer_df[metric], errors="coerce").dropna()
        mine = f(df.iloc[-1][metric]) if len(df) else None
        if mine is None or not len(vals):
            return None
        pct = (vals < mine).mean() if higher_better else (vals > mine).mean()
        return round(pct * 100)


    notes = []
    LATEST = max([norm(y) for y in pd.concat([ph, pp])["year"]] or ["0"])
    if predominant == "hitting" and bat in ("L", "R", "S"):
        row = ph[ph["year"].map(norm) == LATEST]
        peers = hit[(hit["year"].map(norm) == LATEST) &
                    (pd.to_numeric(hit["plate_appearances"], errors="coerce").fillna(0) >= 50)].copy()
        # restrict peer group to same handedness via players.csv bat
        bat_map = {norm(r["id"]): (r.get("bat") or "").strip().upper() for _, r in players.iterrows()}
        peers = peers[peers["pid"].map(bat_map) == bat]
        for metric, lab in (("on_base_percentage", "OBP"), ("slugging_percentage", "SLG"),
                            ("isolated_power", "ISO"), ("weighted_on_base_average", "wOBA")):
            p = handed_pct(row, peers, metric, True)
            if p is not None:
                notes.append(f"**{p}th percentile in {lab}** among {BAT_LABEL.get(bat,bat)} hitters ({LATEST})")
    elif predominant == "pitching" and throw in ("L", "R"):
        row = pp[pp["year"].map(norm) == LATEST]
        peers = pit[(pit["year"].map(norm) == LATEST) &
                    (pd.to_numeric(pit["innings_pitched"], errors="coerce").fillna(0) >= 15)].copy()
        thr_map = {norm(r["id"]): (r.get("throw") or "").strip().upper() for _, r in players.iterrows()}
        peers = peers[peers["pid"].map(thr_map) == throw]
        for metric, lab, hb in (("strikeout_to_walk_ratio", "K/BB", True),
                                ("fielding_independent_pitching", "FIP", False),
                                ("walks_plus_hits_per_inning_pitched", "WHIP", False),
                                ("on_base_plus_slugging_against", "OPS allowed", False)):
            p = handed_pct(row, peers, metric, hb)
            if p is not None:
                notes.append(f"**{p}th percentile in {lab}** among {THROW_LABEL.get(throw,throw)} pitchers ({LATEST})")

    if notes:
        st.markdown("\n".join(f"- {n}" for n in notes[:4]))
    else:
        st.caption("Not enough same-handedness peer data to compute a split for the latest season.")

    st.divider()
    st.caption("Overview = career totals across all seasons in our data. Big weekend = best 3-game (pitchers: Big performance = best single outing, 3+ IP) "
               "series from per-game PBP (2025–2026). Percentiles are within all qualified players; "
               "handedness percentiles are within the player's own bat/throw group.")
