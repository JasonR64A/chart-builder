"""
Portal Entrant Graphic Builder — produce a print-style graphic card for a
transfer portal entrant. Auto-fills from portal_archive + hitting.csv, all
fields editable, renders the design template, and offers a PNG download.
"""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _APP_DIR / 'data'
ASSETS_DIR = _APP_DIR / 'assets' / 'portal-entrant'
# Template choices keyed by output format (used by the format radio below).
# Landscape is the original 1500x1000 design. The two Instagram options are
# 1080x1350 (4:5 portrait) cards intended as a two-image carousel: card 1
# leads with the photo + identity, card 2 carries the full stat grid +
# career table. Trying to fit both on a single portrait clipped the table
# (the failed combined attempt is intentionally not in this list).
TEMPLATES = {
    'Landscape (1500x1000)':            {'path': ASSETS_DIR / 'template.html',          'w': 1500, 'h': 1000},
    'Instagram — Photo (1080x1350)':    {'path': ASSETS_DIR / 'template_ig_photo.html', 'w': 1080, 'h': 1350},
    'Instagram — Stats (1080x1350)':    {'path': ASSETS_DIR / 'template_ig_stats.html', 'w': 1080, 'h': 1350},
}

CLASS_SHORT = {
    'Freshman': 'FR.', 'Sophomore': 'SO.', 'Junior': 'JR.', 'Senior': 'SR.',
    'FR': 'FR.', 'SO': 'SO.', 'JR': 'JR.', 'SR': 'SR.',
    'Graduate': 'GR.', 'Grad': 'GR.', 'GR': 'GR.', 'Redshirt Senior': 'SR.',
}
CLASS_LONG = {
    'FR.': 'FRESHMAN', 'SO.': 'SOPHOMORE', 'JR.': 'JUNIOR', 'SR.': 'SENIOR',
    'GR.': 'GRADUATE',
}

st.set_page_config(page_title='Portal Entrant Graphic', layout='wide')
st.title('Portal Entrant Graphic')
st.caption('Build a shareable transfer-portal graphic for a single player.')

format_choice = st.radio(
    'Format',
    list(TEMPLATES.keys()),
    horizontal=True,
    help=(
        'Landscape (3:2) — Twitter / X / web. '
        'Instagram cards are a 2-image carousel: render Photo, download, then switch to Stats and download again.'
    ),
)
TEMPLATE_PATH = TEMPLATES[format_choice]['path']
TEMPLATE_W = TEMPLATES[format_choice]['w']
TEMPLATE_H = TEMPLATES[format_choice]['h']


# ── Data loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_players() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / 'players.csv', encoding='latin1', dtype=str).fillna('')


@st.cache_data
def load_teams() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / 'teams.csv', dtype=str).fillna('')


@st.cache_data
def load_hitting() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / 'hitting.csv', low_memory=False)


@st.cache_data
def load_pitching() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / 'pitching.csv', low_memory=False)


players = load_players()
teams = load_teams()
hitting = load_hitting()
pitching = load_pitching()


# Hitter season grid + career table (default labels).
HITTER_LABELS = {
    'LAB_1': 'AVG', 'LAB_2': 'OBP', 'LAB_3': 'SLG', 'LAB_4': 'OPS',
    'LAB_5': 'HR',  'LAB_6': 'RBI', 'LAB_7': 'R',   'LAB_8': 'SB',
    'LAB_9': 'H',   'LAB_10':'AB',  'LAB_11':'2B',  'LAB_12':'3B',
    'LAB_13':'TB',  'LAB_14':'BB',  'LAB_15':'HBP', 'LAB_16':'SO',
    'C_TH1': 'YEAR','C_TH2': 'G',   'C_TH3': 'AVG','C_TH4': 'OBP',
    'C_TH5': 'SLG', 'C_TH6': 'OPS', 'C_TH7': 'HR', 'C_TH8': 'RBI',
    'C_TH9': 'H/AB','C_TH10':'BB/K',
}
# Pitcher season grid + career table — mirrors PBP Analytics pitcher card.
# Slash row = ERA / FIP / WHIP / K%. Cell row = BB% / K-BB% / K/9 / OPS-A.
# Last row = IP / GS / W / L / SV / H / ER / BB.
PITCHER_LABELS = {
    'LAB_1': 'ERA', 'LAB_2': 'FIP', 'LAB_3': 'WHIP','LAB_4': 'K%',
    'LAB_5': 'BB%', 'LAB_6': 'K-BB%','LAB_7':'K/9','LAB_8': 'OPS-A',
    'LAB_9': 'IP',  'LAB_10':'GS',  'LAB_11':'W',   'LAB_12':'L',
    'LAB_13':'SV',  'LAB_14':'H',   'LAB_15':'ER',  'LAB_16':'BB',
    'C_TH1': 'YEAR','C_TH2': 'G',   'C_TH3': 'ERA', 'C_TH4': 'FIP',
    'C_TH5': 'WHIP','C_TH6': 'K%',  'C_TH7': 'BB%', 'C_TH8': 'IP',
    'C_TH9': 'K/BB','C_TH10':'W-L',
}


def _is_pitcher(player_row) -> bool:
    """Pitcher detection — position contains 'P' (NCAA position codes use P,
    SP, RP, RHP, LHP) and not just 'PH'/'PR' (pinch-hitter / pinch-runner).
    Falls back to "has rows in pitching.csv" when position is blank."""
    if player_row is None:
        return False
    pos = str(player_row.get('position') or '').upper().strip()
    if pos in ('PH', 'PR'):
        return False
    if 'P' in pos and pos != '':
        return True
    try:
        pid = int(player_row.get('id') or 0)
    except Exception:
        return False
    return pid != 0 and not pitching[pitching['player_id'] == pid].empty

# Join team name + sport onto players for display
team_lookup = teams[['id', 'name', 'sport']].rename(
    columns={'id': 'team_id', 'name': 'team_name', 'sport': 'sport'}
)
players_ext = players.merge(team_lookup, on='team_id', how='left').fillna('')


# ── Player picker ───────────────────────────────────────────────────────────
st.subheader('1. Pick a player')
query = st.text_input('Search players by name',
                        placeholder='Start typing a name...',
                        key='player_search')

prec = None
school_name = ''
sport_filter = ''
if query:
    q = query.lower().strip()
    hits = players_ext[players_ext['player_name'].str.lower().str.contains(q, na=False)]
    hits = hits.head(50)
    if hits.empty:
        st.info('No matches — type more of the name.')
    else:
        opts = hits.apply(
            lambda r: f"{r['player_name']} — {r['team_name']} ({r['position'] or '—'}, {r['classification'] or '—'})",
            axis=1
        ).tolist()
        pick = st.selectbox(f'Matches ({len(hits)})', opts, key='player_pick')
        prec = hits.iloc[opts.index(pick)]
        school_name = prec.get('team_name', '')
        sport_filter = (prec.get('sport') or '').lower()


def career_stats(player_id: str):
    if not player_id or hitting.empty:
        return pd.DataFrame()
    df = hitting[hitting['player_id'] == int(player_id)].copy() if str(player_id).isdigit() else pd.DataFrame()
    return df.sort_values('year', ascending=False)


def pitcher_career_stats(player_id: str):
    if not player_id or pitching.empty:
        return pd.DataFrame()
    df = pitching[pitching['player_id'] == int(player_id)].copy() if str(player_id).isdigit() else pd.DataFrame()
    return df.sort_values('year', ascending=False)


def fmt_rate(v, digits=2):
    """Format a rate stat (ERA, FIP, WHIP) to fixed decimals or em-dash."""
    try:
        f = float(v)
        if pd.isna(f):
            return '—'
        return f'{f:.{digits}f}'
    except Exception:
        return '—'


def fmt_pct(v):
    """Format K%/BB% — pitching.csv stores them as decimals (0.18 = 18%)."""
    try:
        f = float(v)
        if pd.isna(f):
            return '—'
        return f'{f * 100:.1f}'
    except Exception:
        return '—'


def fmt_slash(v, digits=3):
    try:
        f = float(v)
        s = f'{f:.{digits}f}'
        # .301 style (drop leading 0)
        if s.startswith('0.'):
            return s[1:]
        if s.startswith('-0.'):
            return '-' + s[2:]
        return s
    except Exception:
        return '—'


def season_stat(row, key, default=0):
    if row is None or key not in row:
        return default
    v = row[key]
    if pd.isna(v):
        return default
    try:
        return int(v)
    except Exception:
        try:
            return float(v)
        except Exception:
            return default


def _career_sum(df: pd.DataFrame, col: str) -> float:
    """Sum a numeric career column, tolerant of strings/NaN."""
    if df is None or df.empty or col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors='coerce').fillna(0).sum())


def _two_way_sides(player_row):
    """Return (has_hitting, has_pitching) for the player, gated on meaningful
    volume so an incidental mop-up inning or a pitcher's stray AB doesn't
    falsely flag a two-way. Career AB >= 10 and IP >= 5."""
    if player_row is None:
        return False, False
    try:
        pid = int(player_row.get('id') or 0)
    except Exception:
        return False, False
    if not pid:
        return False, False
    has_hit = _career_sum(career_stats(pid), 'at_bats') >= 10
    has_pit = _career_sum(pitcher_career_stats(pid), 'innings_pitched') >= 5
    return has_hit, has_pit


# Two-way players (e.g. Ryan Kroepel: hits AND pitches) get a toggle so the
# user picks which stat set the card features. Non-two-way players keep the
# automatic hitter/pitcher detection.
if prec is None:
    is_pitcher_player = False
else:
    _has_hit, _has_pit = _two_way_sides(prec)
    if _has_hit and _has_pit:
        _default_idx = 1 if _is_pitcher(prec) else 0  # default to primary side
        _stat_mode = st.radio(
            'Two-way player — feature which stat?',
            ['Hitting', 'Pitching'],
            index=_default_idx,
            horizontal=True,
            key='two_way_stat_mode',
            help=('This player has both hitting and pitching data. Choose which '
                  'stat set drives the season line + career table on the card.'),
        )
        is_pitcher_player = (_stat_mode == 'Pitching')
    else:
        is_pitcher_player = _is_pitcher(prec)

# Build defaults from the selected row. Stat keys s1..s16 map positionally to
# the template's 16 season cells (LAB_1..LAB_16 supply the labels). Hitters
# fill them with AVG/OBP/SLG/OPS/HR/RBI/R/SB/H/AB/2B/3B/TB/BB/HBP/SO; pitchers
# with ERA/FIP/WHIP/K%/BB%/K-BB%/K/9/OPS-A/IP/GS/W/L/SV/H/ER/BB.
defaults = {
    'name': '', 'school': '', 'school_abbr': '', 'pos': '', 'class_short': '',
    'class_long': '', 'bat': '', 'throw': '',
    'season_year': 2026, 'season_gp': 0, 'season_pa': 0,
    's_avg': '.000', 's_obp': '.000', 's_slg': '.000', 's_ops': '.000',
    's_hr': 0, 's_rbi': 0, 's_r': 0, 's_sb': 0,
    's_h': 0, 's_ab': 0, 's_2b': 0, 's_3b': 0, 's_tb': 0,
    's_bb': 0, 's_hbp': 0, 's_so': 0,
    'career_slash': '.000 / .000 / .000',
    'career_rows': [],
    'career_total': None,
    'career_subtitle': '',
    'photo_src_override': None,
    # Pitcher-side: editable values shown when the player is a pitcher.
    'p_era': '0.00', 'p_fip': '0.00', 'p_whip': '0.00',
    'p_k_pct': '0.0', 'p_bb_pct': '0.0', 'p_kbb_pct': '0.0',
    'p_k9': '0.0', 'p_oppa': '.000',
    'p_ip': '0.0', 'p_gs': 0, 'p_w': 0, 'p_l': 0,
    'p_sv': 0, 'p_h': 0, 'p_er': 0, 'p_bb': 0,
    'p_career_subtitle': '',
    'p_career_slash': '0.00 / 0.00 / 0.00',
    'p_career_rows': [],
    'p_career_total': None,
}

if prec is not None and not is_pitcher_player:
    defaults['name'] = prec.get('player_name', '')
    defaults['school'] = school_name
    defaults['school_abbr'] = ''.join(w[0] for w in school_name.split()[:3]).upper() if school_name else ''
    defaults['pos'] = (prec.get('position') or '').upper()
    cls = prec.get('classification') or ''
    defaults['class_short'] = CLASS_SHORT.get(cls, cls.upper()[:3]+'.' if cls else '')
    defaults['class_long'] = CLASS_LONG.get(defaults['class_short'], cls.upper())
    defaults['bat'] = (prec.get('bat') or '').upper()
    defaults['throw'] = (prec.get('throw') or '').upper()

    # Career stats
    try:
        pid = int(prec.get('id') or 0)
    except Exception:
        pid = 0
    hist = career_stats(pid) if pid else pd.DataFrame()
    if not hist.empty:
        cur = hist[hist['year'] == 2026]
        cur_row = cur.iloc[0] if not cur.empty else None
        if cur_row is not None:
            defaults['season_gp'] = season_stat(cur_row, 'games_played', 0)
            defaults['season_pa'] = season_stat(cur_row, 'plate_appearances', 0)
            defaults['s_avg'] = fmt_slash(cur_row.get('batting_average'))
            defaults['s_obp'] = fmt_slash(cur_row.get('on_base_percentage'))
            defaults['s_slg'] = fmt_slash(cur_row.get('slugging_percentage'))
            defaults['s_ops'] = fmt_slash(cur_row.get('on_base_plus_slugging'))
            defaults['s_hr'] = season_stat(cur_row, 'home_runs')
            defaults['s_rbi'] = season_stat(cur_row, 'runs_batted_in')
            defaults['s_r'] = season_stat(cur_row, 'runs_scored')
            defaults['s_sb'] = season_stat(cur_row, 'stolen_bases')
            defaults['s_h'] = season_stat(cur_row, 'hits')
            defaults['s_ab'] = season_stat(cur_row, 'at_bats')
            defaults['s_2b'] = season_stat(cur_row, 'doubles')
            defaults['s_3b'] = season_stat(cur_row, 'triples')
            defaults['s_tb'] = season_stat(cur_row, 'total_bases')
            defaults['s_bb'] = season_stat(cur_row, 'walks')
            defaults['s_hbp'] = season_stat(cur_row, 'hit_by_pitch')
            defaults['s_so'] = season_stat(cur_row, 'strikeouts')

        # Career rows + totals
        tot = {'ab': 0, 'h': 0, 'bb': 0, 'k': 0, 'hr': 0, 'rbi': 0, 'g': 0,
                 'hbp': 0, 'sf': 0, 'tb': 0, 'pa': 0}
        rows = []
        for _, yr in hist.iterrows():
            g = season_stat(yr, 'games_played')
            ab = season_stat(yr, 'at_bats')
            h  = season_stat(yr, 'hits')
            bb = season_stat(yr, 'walks')
            k  = season_stat(yr, 'strikeouts')
            hr = season_stat(yr, 'home_runs')
            rbi = season_stat(yr, 'runs_batted_in')
            tb = season_stat(yr, 'total_bases')
            hbp = season_stat(yr, 'hit_by_pitch')
            sf = season_stat(yr, 'sac_fly')
            rows.append({
                'year': int(yr['year']) if not pd.isna(yr['year']) else '',
                'g': g,
                'avg': fmt_slash(yr.get('batting_average')),
                'obp': fmt_slash(yr.get('on_base_percentage')),
                'slg': fmt_slash(yr.get('slugging_percentage')),
                'ops': fmt_slash(yr.get('on_base_plus_slugging')),
                'hr': hr, 'rbi': rbi,
                'hAb': f'{h}/{ab}',
                'bbK': f'{bb}/{k}',
                # Hidden raw fields used to recompute OBP/SLG/OPS in the totals row:
                '_tb': tb, '_hbp': hbp, '_sf': sf,
            })
            for key, val in zip(('g','ab','h','bb','k','hr','rbi','tb','hbp','sf'),
                                  (g, ab, h, bb, k, hr, rbi, tb, hbp, sf)):
                tot[key] += val
        defaults['career_rows'] = rows
        # Totals
        if tot['ab'] > 0:
            avg = tot['h'] / tot['ab'] if tot['ab'] else 0
            pa_est = tot['ab'] + tot['bb'] + tot['hbp'] + tot['sf']
            obp = (tot['h'] + tot['bb'] + tot['hbp']) / pa_est if pa_est else 0
            slg = tot['tb'] / tot['ab'] if tot['ab'] else 0
            ops = obp + slg
            defaults['career_total'] = {
                'year': 'TOTAL', 'g': tot['g'],
                'avg': fmt_slash(avg), 'obp': fmt_slash(obp),
                'slg': fmt_slash(slg), 'ops': fmt_slash(ops),
                'hr': tot['hr'], 'rbi': tot['rbi'],
                'hAb': f"{tot['h']}/{tot['ab']}",
                'bbK': f"{tot['bb']}/{tot['k']}",
            }
            defaults['career_slash'] = f"{defaults['career_total']['avg']} / {defaults['career_total']['obp']} / {defaults['career_total']['slg']}"
            defaults['career_subtitle'] = f"{len(rows)} {'SEASON' if len(rows)==1 else 'SEASONS'} AT {defaults['school'].upper()}"


if prec is not None and is_pitcher_player:
    defaults['name'] = prec.get('player_name', '')
    defaults['school'] = school_name
    defaults['school_abbr'] = ''.join(w[0] for w in school_name.split()[:3]).upper() if school_name else ''
    defaults['pos'] = (prec.get('position') or '').upper()
    cls = prec.get('classification') or ''
    defaults['class_short'] = CLASS_SHORT.get(cls, cls.upper()[:3]+'.' if cls else '')
    defaults['class_long'] = CLASS_LONG.get(defaults['class_short'], cls.upper())
    defaults['bat'] = (prec.get('bat') or '').upper()
    defaults['throw'] = (prec.get('throw') or '').upper()

    try:
        pid = int(prec.get('id') or 0)
    except Exception:
        pid = 0
    hist = pitcher_career_stats(pid) if pid else pd.DataFrame()
    if not hist.empty:
        cur = hist[hist['year'] == 2026]
        cur_row = cur.iloc[0] if not cur.empty else None
        if cur_row is not None:
            defaults['season_gp'] = season_stat(cur_row, 'games_appeared') or season_stat(cur_row, 'appearances') or 0
            defaults['season_pa'] = season_stat(cur_row, 'batters_faced')
            defaults['p_era']  = fmt_rate(cur_row.get('earned_run_average'), 2)
            defaults['p_fip']  = fmt_rate(cur_row.get('fielding_independent_pitching'), 2)
            defaults['p_whip'] = fmt_rate(cur_row.get('walks_plus_hits_per_inning_pitched'), 2)
            defaults['p_k_pct']  = fmt_pct(cur_row.get('strikeout_percentage'))
            defaults['p_bb_pct'] = fmt_pct(cur_row.get('walk_percentage'))
            defaults['p_kbb_pct'] = fmt_pct(cur_row.get('strikeout_minus_walk_percentage'))
            defaults['p_k9'] = fmt_rate(cur_row.get('strikeouts_per_9_innings'), 1)
            # OPS-A: not in pitching.csv directly — leave editable, derive
            # from on_base_percentage_against + slugging_percentage_against if
            # present, else blank.
            obpa = pd.to_numeric(cur_row.get('on_base_percentage_against'), errors='coerce')
            slga = pd.to_numeric(cur_row.get('slugging_percentage_against'), errors='coerce')
            opsa = (obpa + slga) if (pd.notna(obpa) and pd.notna(slga)) else None
            defaults['p_oppa'] = fmt_slash(opsa) if opsa is not None else '—'
            defaults['p_ip'] = fmt_rate(cur_row.get('innings_pitched'), 1)
            defaults['p_gs'] = season_stat(cur_row, 'games_started')
            defaults['p_w']  = season_stat(cur_row, 'wins')
            defaults['p_l']  = season_stat(cur_row, 'losses')
            defaults['p_sv'] = season_stat(cur_row, 'saves')
            defaults['p_h']  = season_stat(cur_row, 'hits_allowed')
            defaults['p_er'] = season_stat(cur_row, 'earned_runs_allowed') or season_stat(cur_row, 'earned_runs')
            defaults['p_bb'] = season_stat(cur_row, 'walks_issued')

        # Career rows + totals — accumulate raw counts so we can recompute
        # ERA/FIP/WHIP/K%/BB% on a true totals line rather than averaging rates.
        tot = {'g': 0, 'gs': 0, 'w': 0, 'l': 0, 'ip': 0.0, 'er': 0,
               'h': 0, 'bb': 0, 'k': 0, 'hr': 0, 'hbp': 0, 'bf': 0}
        rows = []
        for _, yr in hist.iterrows():
            g = season_stat(yr, 'games_appeared') or season_stat(yr, 'appearances')
            gs = season_stat(yr, 'games_started')
            w = season_stat(yr, 'wins'); l = season_stat(yr, 'losses')
            ip = float(season_stat(yr, 'innings_pitched') or 0)
            er = season_stat(yr, 'earned_runs_allowed') or season_stat(yr, 'earned_runs')
            h  = season_stat(yr, 'hits_allowed')
            bb = season_stat(yr, 'walks_issued')
            k  = season_stat(yr, 'strikeouts')
            hr = season_stat(yr, 'homeruns_allowed') or season_stat(yr, 'home_runs_allowed')
            hbp = season_stat(yr, 'hit_batter')
            bf = season_stat(yr, 'batters_faced')
            rows.append({
                'year': int(yr['year']) if not pd.isna(yr['year']) else '',
                'g': g,
                'era':  fmt_rate(yr.get('earned_run_average'), 2),
                'fip':  fmt_rate(yr.get('fielding_independent_pitching'), 2),
                'whip': fmt_rate(yr.get('walks_plus_hits_per_inning_pitched'), 2),
                'kPct': fmt_pct(yr.get('strikeout_percentage')),
                'bbPct': fmt_pct(yr.get('walk_percentage')),
                'ip':  fmt_rate(yr.get('innings_pitched'), 1),
                'kBB': f'{k}/{bb}',
                'wL':  f'{w}-{l}',
                # Hidden raw fields used for the totals line
                '_er': er, '_h': h, '_hr': hr, '_hbp': hbp, '_bf': bf,
                '_ip_num': ip, '_k': k, '_bb': bb, '_w': w, '_l': l,
                '_gs': gs,
            })
            tot['g'] += g; tot['gs'] += gs; tot['w'] += w; tot['l'] += l
            tot['ip'] += ip; tot['er'] += er; tot['h'] += h
            tot['bb'] += bb; tot['k'] += k; tot['hr'] += hr
            tot['hbp'] += hbp; tot['bf'] += bf
        defaults['p_career_rows'] = rows
        if tot['ip'] > 0:
            era = 9 * tot['er'] / tot['ip']
            fip_const = 3.0
            fip = ((13 * tot['hr']) + (3 * (tot['bb'] + tot['hbp'])) - (2 * tot['k'])) / tot['ip'] + fip_const
            whip = (tot['h'] + tot['bb']) / tot['ip']
            k_pct  = (tot['k'] / tot['bf']) if tot['bf'] else 0
            bb_pct = (tot['bb'] / tot['bf']) if tot['bf'] else 0
            defaults['p_career_total'] = {
                'year': 'TOTAL', 'g': tot['g'],
                'era': fmt_rate(era, 2), 'fip': fmt_rate(fip, 2), 'whip': fmt_rate(whip, 2),
                'kPct': f'{k_pct*100:.1f}', 'bbPct': f'{bb_pct*100:.1f}',
                'ip': fmt_rate(tot['ip'], 1),
                'kBB': f"{tot['k']}/{tot['bb']}",
                'wL':  f"{tot['w']}-{tot['l']}",
            }
            defaults['p_career_slash'] = f"{fmt_rate(era,2)} / {fmt_rate(fip,2)} / {fmt_rate(whip,2)}"
            defaults['p_career_subtitle'] = f"{len(rows)} {'SEASON' if len(rows)==1 else 'SEASONS'} AT {defaults['school'].upper()}"


# ── Editable form ───────────────────────────────────────────────────────────
st.subheader('2. Review & edit fields')

with st.expander('Identity', expanded=True):
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    name = c1.text_input('Full name', value=defaults['name'])
    school = c2.text_input('School (full)', value=defaults['school'])
    school_abbr = c3.text_input('School abbrev.', value=defaults['school_abbr'])
    pos = c4.text_input('Position', value=defaults['pos'])

    c5, c6, c7, c8 = st.columns(4)
    class_short = c5.text_input('Class (short, e.g. JR.)', value=defaults['class_short'])
    class_long = c6.text_input('Class (long, e.g. JUNIOR)', value=defaults['class_long'])
    bat = c7.text_input('Bats', value=defaults['bat'])
    throw = c8.text_input('Throws', value=defaults['throw'])

    photo_upload = st.file_uploader('Player photo (optional — leave blank to use sample)',
                                     type=['jpg', 'jpeg', 'png', 'webp'])
    if photo_upload is not None:
        # Inline preview so we can verify the upload actually reached Python
        # (this is what gets data-URL'd into the card). If it shows here but
        # the card stays black, the issue is in template rendering, not upload.
        st.image(photo_upload, caption=f'Uploaded · {photo_upload.size:,} bytes · {photo_upload.type}', width=180)

if is_pitcher_player:
    section_title = 'Season line — pitcher'
else:
    section_title = 'Season line'

with st.expander(section_title, expanded=True):
    c1, c2, c3 = st.columns(3)
    season_year = c1.number_input('Season year', min_value=2000, max_value=2100,
                                    value=int(defaults['season_year']))
    season_gp = c2.number_input(
        'Appearances' if is_pitcher_player else 'Games played',
        min_value=0, value=int(defaults['season_gp']))
    season_pa = c3.number_input(
        'Batters faced' if is_pitcher_player else 'Plate appearances',
        min_value=0, value=int(defaults['season_pa']))

    if not is_pitcher_player:
        c1, c2, c3, c4 = st.columns(4)
        s_avg = c1.text_input('AVG', value=defaults['s_avg'])
        s_obp = c2.text_input('OBP', value=defaults['s_obp'])
        s_slg = c3.text_input('SLG', value=defaults['s_slg'])
        s_ops = c4.text_input('OPS', value=defaults['s_ops'])

        c1, c2, c3, c4 = st.columns(4)
        s_hr = c1.number_input('HR', value=int(defaults['s_hr']))
        s_rbi = c2.number_input('RBI', value=int(defaults['s_rbi']))
        s_r = c3.number_input('R', value=int(defaults['s_r']))
        s_sb = c4.number_input('SB', value=int(defaults['s_sb']))

        c1, c2, c3, c4 = st.columns(4)
        s_h = c1.number_input('H', value=int(defaults['s_h']))
        s_ab = c2.number_input('AB', value=int(defaults['s_ab']))
        s_2b = c3.number_input('2B', value=int(defaults['s_2b']))
        s_3b = c4.number_input('3B', value=int(defaults['s_3b']))

        c1, c2, c3, c4 = st.columns(4)
        s_tb = c1.number_input('TB', value=int(defaults['s_tb']))
        s_bb = c2.number_input('BB', value=int(defaults['s_bb']))
        s_hbp = c3.number_input('HBP', value=int(defaults['s_hbp']))
        s_so = c4.number_input('SO', value=int(defaults['s_so']))
    else:
        # Pitcher season cells — labels mirror PBP Analytics pitcher card.
        c1, c2, c3, c4 = st.columns(4)
        p_era  = c1.text_input('ERA',  value=defaults['p_era'])
        p_fip  = c2.text_input('FIP',  value=defaults['p_fip'])
        p_whip = c3.text_input('WHIP', value=defaults['p_whip'])
        p_k_pct = c4.text_input('K%',  value=defaults['p_k_pct'])

        c1, c2, c3, c4 = st.columns(4)
        p_bb_pct  = c1.text_input('BB%',   value=defaults['p_bb_pct'])
        p_kbb_pct = c2.text_input('K-BB%', value=defaults['p_kbb_pct'])
        p_k9      = c3.text_input('K/9',   value=defaults['p_k9'])
        p_oppa    = c4.text_input('OPS-A', value=defaults['p_oppa'])

        c1, c2, c3, c4 = st.columns(4)
        p_ip = c1.text_input('IP',  value=defaults['p_ip'])
        p_gs = c2.number_input('GS', value=int(defaults['p_gs']))
        p_w  = c3.number_input('W',  value=int(defaults['p_w']))
        p_l  = c4.number_input('L',  value=int(defaults['p_l']))

        c1, c2, c3, c4 = st.columns(4)
        p_sv = c1.number_input('SV', value=int(defaults['p_sv']))
        p_h  = c2.number_input('H',  value=int(defaults['p_h']))
        p_er = c3.number_input('ER', value=int(defaults['p_er']))
        p_bb = c4.number_input('BB', value=int(defaults['p_bb']))

with st.expander('Career by year (editable)', expanded=False):
    st.caption('Edit, add, or remove rows. Totals row is recomputed automatically.')
    if not is_pitcher_player:
        default_df = pd.DataFrame(defaults['career_rows']) if defaults['career_rows'] else pd.DataFrame(
            [{'year': season_year, 'g': season_gp,
              'avg': s_avg, 'obp': s_obp, 'slg': s_slg, 'ops': s_ops,
              'hr': s_hr, 'rbi': s_rbi,
              'hAb': f'{s_h}/{s_ab}', 'bbK': f'{s_bb}/{s_so}',
              '_tb': s_tb, '_hbp': s_hbp, '_sf': 0}]
        )
        edited = st.data_editor(
            default_df, num_rows='dynamic', use_container_width=True,
            key='career_editor',
            column_config={
                '_tb': st.column_config.NumberColumn('TB', help='Total bases — used for SLG/OPS totals'),
                '_hbp': st.column_config.NumberColumn('HBP', help='Hit by pitch — used for OBP totals'),
                '_sf': st.column_config.NumberColumn('SF', help='Sac fly — used for OBP totals'),
            },
        )
        career_subtitle = st.text_input('Career section subtitle',
                                        value=defaults['career_subtitle'] or f"{len(edited)} SEASON{'S' if len(edited) != 1 else ''} AT {school.upper()}")
        career_slash_override = st.text_input('Career slash (top-right header)',
                                              value=defaults['career_slash'])
    else:
        # Pitcher career editor — visible columns are display strings, hidden
        # _-prefixed columns hold the raw counts used to recompute the totals
        # row (rates can't simply be summed/averaged).
        default_df = pd.DataFrame(defaults['p_career_rows']) if defaults['p_career_rows'] else pd.DataFrame(
            [{'year': season_year, 'g': season_gp,
              'era': p_era, 'fip': p_fip, 'whip': p_whip,
              'kPct': p_k_pct, 'bbPct': p_bb_pct, 'ip': p_ip,
              'kBB': f'{int(season_gp) if False else 0}/{p_bb}',
              'wL':  f'{p_w}-{p_l}',
              '_er': p_er, '_h': p_h, '_hr': 0, '_hbp': 0,
              '_bf': int(season_pa), '_ip_num': float(p_ip or 0),
              '_k': 0, '_bb': int(p_bb), '_w': int(p_w), '_l': int(p_l),
              '_gs': int(p_gs)}]
        )
        edited = st.data_editor(
            default_df, num_rows='dynamic', use_container_width=True,
            key='career_editor_pitcher',
            column_config={
                '_er':     st.column_config.NumberColumn('ER', help='Earned runs — totals ERA'),
                '_h':      st.column_config.NumberColumn('H',  help='Hits allowed — totals WHIP'),
                '_hr':     st.column_config.NumberColumn('HR', help='HR allowed — totals FIP'),
                '_hbp':    st.column_config.NumberColumn('HBP',help='HBP — totals FIP'),
                '_bf':     st.column_config.NumberColumn('BF', help='Batters faced — totals K%/BB%'),
                '_ip_num': st.column_config.NumberColumn('IP#',help='IP as a number — totals ERA/WHIP/FIP'),
                '_k':      st.column_config.NumberColumn('K',  help='Ks — totals K%'),
                '_bb':     st.column_config.NumberColumn('BB', help='BBs — totals BB%'),
                '_w':      st.column_config.NumberColumn('W',  help='Wins — totals W-L'),
                '_l':      st.column_config.NumberColumn('L',  help='Losses — totals W-L'),
                '_gs':     st.column_config.NumberColumn('GS', help='Games started'),
            },
        )
        career_subtitle = st.text_input('Career section subtitle',
                                        value=defaults['p_career_subtitle'] or f"{len(edited)} SEASON{'S' if len(edited) != 1 else ''} AT {school.upper()}")
        career_slash_override = st.text_input('Career slash (top-right header)',
                                              value=defaults['p_career_slash'])

with st.expander('Headline/eyebrow', expanded=False):
    eyebrow = st.text_input('Eyebrow', value=f'TRANSFER PORTAL · {season_year}')
    headline = st.text_input('Headline', value='ENTERED THE PORTAL')


# ── Render ──────────────────────────────────────────────────────────────────
def data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    b = path.read_bytes()
    return f'data:{mime or "image/png"};base64,' + base64.b64encode(b).decode()


def build_pitcher_career_rows_html(rows_df: pd.DataFrame) -> str:
    """Pitcher table: YEAR / G / ERA / FIP / WHIP / K% / BB% / IP / K/BB / W-L.
    Totals row recomputed from per-year raw counts (rates can't be summed)."""
    if rows_df is None or rows_df.empty:
        return ''
    html = []
    def _int(v):
        try: return int(v)
        except Exception:
            try: return int(float(v))
            except Exception: return 0
    def _flt(v):
        try: return float(v)
        except Exception: return 0.0
    tot = {'g': 0, 'gs': 0, 'w': 0, 'l': 0, 'ip': 0.0, 'er': 0,
           'h': 0, 'bb': 0, 'k': 0, 'hr': 0, 'hbp': 0, 'bf': 0}
    for _, r in rows_df.iterrows():
        html.append(
            f'<tr><td>{r.get("year","")}</td><td>{r.get("g","")}</td>'
            f'<td>{r.get("era","")}</td><td>{r.get("fip","")}</td>'
            f'<td>{r.get("whip","")}</td><td>{r.get("kPct","")}</td>'
            f'<td>{r.get("bbPct","")}</td><td>{r.get("ip","")}</td>'
            f'<td>{r.get("kBB","")}</td><td>{r.get("wL","")}</td></tr>'
        )
        tot['g'] += _int(r.get('g'))
        tot['gs'] += _int(r.get('_gs'))
        tot['w']  += _int(r.get('_w'));  tot['l'] += _int(r.get('_l'))
        tot['ip'] += _flt(r.get('_ip_num'))
        tot['er'] += _int(r.get('_er'));  tot['h']  += _int(r.get('_h'))
        tot['bb'] += _int(r.get('_bb'));  tot['k']  += _int(r.get('_k'))
        tot['hr'] += _int(r.get('_hr'));  tot['hbp'] += _int(r.get('_hbp'))
        tot['bf'] += _int(r.get('_bf'))
    if tot['ip'] > 0:
        era  = 9 * tot['er'] / tot['ip']
        fip  = ((13 * tot['hr']) + (3 * (tot['bb'] + tot['hbp'])) - (2 * tot['k'])) / tot['ip'] + 3.0
        whip = (tot['h'] + tot['bb']) / tot['ip']
        kp   = (tot['k']  / tot['bf']) if tot['bf'] else 0
        bbp  = (tot['bb'] / tot['bf']) if tot['bf'] else 0
        html.append(
            f'<tr class="totals"><td>TOTAL</td><td>{tot["g"]}</td>'
            f'<td>{era:.2f}</td><td>{fip:.2f}</td><td>{whip:.2f}</td>'
            f'<td>{kp*100:.1f}</td><td>{bbp*100:.1f}</td>'
            f'<td>{tot["ip"]:.1f}</td>'
            f'<td>{tot["k"]}/{tot["bb"]}</td>'
            f'<td>{tot["w"]}-{tot["l"]}</td></tr>'
        )
    return '\n'.join(html)


def build_career_rows_html(rows_df: pd.DataFrame) -> str:
    if rows_df is None or rows_df.empty:
        return ''
    html = []
    for _, r in rows_df.iterrows():
        html.append(
            f'<tr><td>{r.get("year","")}</td><td>{r.get("g","")}</td>'
            f'<td>{r.get("avg","")}</td><td>{r.get("obp","")}</td>'
            f'<td>{r.get("slg","")}</td><td>{r.get("ops","")}</td>'
            f'<td>{r.get("hr","")}</td><td>{r.get("rbi","")}</td>'
            f'<td>{r.get("hAb","")}</td><td>{r.get("bbK","")}</td></tr>'
        )
    # Totals: recompute from per-year raw fields
    tot = {'g': 0, 'hr': 0, 'rbi': 0, 'h': 0, 'ab': 0, 'bb': 0, 'k': 0,
           'tb': 0, 'hbp': 0, 'sf': 0}

    def _int(v):
        try: return int(v)
        except Exception:
            try: return int(float(v))
            except Exception: return 0

    for _, r in rows_df.iterrows():
        tot['g']   += _int(r.get('g'))
        tot['hr']  += _int(r.get('hr'))
        tot['rbi'] += _int(r.get('rbi'))
        hAb = str(r.get('hAb') or '0/0')
        bbK = str(r.get('bbK') or '0/0')
        try:
            h, ab = [int(x) for x in hAb.split('/')]
            tot['h'] += h; tot['ab'] += ab
        except Exception: pass
        try:
            bb, k = [int(x) for x in bbK.split('/')]
            tot['bb'] += bb; tot['k'] += k
        except Exception: pass
        tot['tb']  += _int(r.get('_tb'))
        tot['hbp'] += _int(r.get('_hbp'))
        tot['sf']  += _int(r.get('_sf'))

    if tot['ab'] > 0:
        avg = tot['h'] / tot['ab']
        pa_est = tot['ab'] + tot['bb'] + tot['hbp'] + tot['sf']
        obp = (tot['h'] + tot['bb'] + tot['hbp']) / pa_est if pa_est else 0
        slg = tot['tb'] / tot['ab'] if tot['tb'] else 0
        ops = obp + slg
        avg_s = fmt_slash(avg)
        obp_s = fmt_slash(obp) if tot['tb'] or tot['hbp'] else '—'
        slg_s = fmt_slash(slg) if tot['tb'] else '—'
        ops_s = fmt_slash(ops) if tot['tb'] else '—'
        html.append(
            f'<tr class="totals"><td>TOTAL</td><td>{tot["g"]}</td>'
            f'<td>{avg_s}</td><td>{obp_s}</td><td>{slg_s}</td><td>{ops_s}</td>'
            f'<td>{tot["hr"]}</td><td>{tot["rbi"]}</td>'
            f'<td>{tot["h"]}/{tot["ab"]}</td><td>{tot["bb"]}/{tot["k"]}</td></tr>'
        )
    return '\n'.join(html)


# Photo. The browser sometimes hands Streamlit an empty `type` (or even
# 'application/octet-stream') for re-encoded PNGs / screenshots; that would
# produce a malformed `data:;base64,...` URL and the photo-frame would render
# as just its #222 background ("black box"). Fall back to filename extension
# and finally to image/png since 99% of uploads are images.
if photo_upload is not None:
    photo_bytes = photo_upload.getvalue()
    mime = (photo_upload.type or '').strip()
    if not mime or not mime.startswith('image/'):
        ext = (Path(photo_upload.name or '').suffix.lower().lstrip('.')) if photo_upload.name else ''
        mime = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
                'png': 'image/png',  'webp': 'image/webp'}.get(ext, 'image/png')
    # Resize big uploads — phone-shot PNGs are commonly 4K+ and produce 8MB+
    # data URLs that can render as a blank iframe in components.html. The
    # photo-frame is only 430px wide on the card, so 1200px on the long side
    # is plenty.
    if photo_bytes and len(photo_bytes) > 600_000:
        try:
            from PIL import Image
            import io
            im = Image.open(io.BytesIO(photo_bytes))
            im.thumbnail((1200, 1500), Image.LANCZOS)
            buf = io.BytesIO()
            if mime == 'image/png':
                im.save(buf, format='PNG', optimize=True)
            elif mime == 'image/webp':
                im.save(buf, format='WEBP', quality=88)
            else:
                im.convert('RGB').save(buf, format='JPEG', quality=88, optimize=True)
                mime = 'image/jpeg'
            photo_bytes = buf.getvalue()
        except Exception as e:
            st.warning(f'Could not resize image ({e}); using original.')
    if photo_bytes:
        photo_src = f'data:{mime};base64,' + base64.b64encode(photo_bytes).decode()
        st.caption(f'Photo data URL: {len(photo_bytes):,} bytes ({mime})')
    else:
        st.warning('Photo upload returned 0 bytes — falling back to sample.')
        photo_src = data_url(ASSETS_DIR / 'sample-photo.jpg') if (ASSETS_DIR / 'sample-photo.jpg').exists() else ''
else:
    photo_src = data_url(ASSETS_DIR / 'sample-photo.jpg') if (ASSETS_DIR / 'sample-photo.jpg').exists() else ''

# Other assets → data URLs
bg_url = data_url(ASSETS_DIR / 'foul-line-dirt.png')
circle_url = data_url(ASSETS_DIR / '64-circlelogo-red-white.png')
mark_url = data_url(ASSETS_DIR / '64-logo-red-white.png')
# Triple-logo top-left brand mark (replaces the single 64A circle logo)
logo_cbs_url = data_url(ASSETS_DIR / 'logo-college-baseball-show.png')
logo_jb_url = data_url(ASSETS_DIR / 'logo-jb.png')
logo_64a_url = data_url(ASSETS_DIR / 'logo-64-analytics.png')

meta_line = f'{pos} &nbsp;·&nbsp; {school_abbr or school.upper()[:3]} &nbsp;·&nbsp; {class_long or class_short} &nbsp;·&nbsp; {bat}/{throw}'

if is_pitcher_player:
    season_slash = f'{p_era} / {p_fip} / {p_whip}'
    season_subtitle = f"{school.upper()} · {int(season_gp)} APP · {p_ip} IP"
    if edited is not None and not edited.empty and defaults['p_career_total']:
        career_slash = career_slash_override or defaults['p_career_slash']
    else:
        career_slash = career_slash_override
    season_values = {
        '{{S_AVG}}': p_era, '{{S_OBP}}': p_fip, '{{S_SLG}}': p_whip, '{{S_OPS}}': p_k_pct,
        '{{S_HR}}': p_bb_pct, '{{S_RBI}}': p_kbb_pct,
        '{{S_R}}':  p_k9,    '{{S_SB}}':  p_oppa,
        '{{S_H}}':  p_ip,    '{{S_AB}}':  str(int(p_gs)),
        '{{S_2B}}': str(int(p_w)),  '{{S_3B}}': str(int(p_l)),
        '{{S_TB}}': str(int(p_sv)), '{{S_BB}}': str(int(p_h)),
        '{{S_HBP}}': str(int(p_er)), '{{S_SO}}': str(int(p_bb)),
    }
    label_map = {f'{{{{{k}}}}}': v for k, v in PITCHER_LABELS.items()}
    career_rows_html = build_pitcher_career_rows_html(edited)
else:
    season_slash = f'{s_avg} / {s_obp} / {s_slg}'
    season_subtitle = f"{school.upper()} · {int(season_gp)} GP · {int(season_pa)} PA"
    if edited is not None and not edited.empty and defaults['career_total']:
        career_slash = career_slash_override or defaults['career_slash']
    else:
        career_slash = career_slash_override
    season_values = {
        '{{S_AVG}}': s_avg, '{{S_OBP}}': s_obp, '{{S_SLG}}': s_slg, '{{S_OPS}}': s_ops,
        '{{S_HR}}': str(int(s_hr)), '{{S_RBI}}': str(int(s_rbi)),
        '{{S_R}}': str(int(s_r)), '{{S_SB}}': str(int(s_sb)),
        '{{S_H}}': str(int(s_h)), '{{S_AB}}': str(int(s_ab)),
        '{{S_2B}}': str(int(s_2b)), '{{S_3B}}': str(int(s_3b)),
        '{{S_TB}}': str(int(s_tb)), '{{S_BB}}': str(int(s_bb)),
        '{{S_HBP}}': str(int(s_hbp)), '{{S_SO}}': str(int(s_so)),
    }
    label_map = {f'{{{{{k}}}}}': v for k, v in HITTER_LABELS.items()}
    career_rows_html = build_career_rows_html(edited)

template = TEMPLATE_PATH.read_text(encoding='utf-8')
replacements = {
    '{{NAME}}': name.upper(),
    '{{SCHOOL}}': school.upper(),
    '{{POS}}': pos,
    '{{CLASS_SHORT}}': class_short,
    '{{CLASS_LONG}}': (class_long or class_short).upper(),
    '{{META_LINE}}': meta_line,
    '{{SEASON_SLASH}}': season_slash,
    '{{SEASON_YEAR}}': str(int(season_year)),
    '{{SEASON_SUBTITLE}}': season_subtitle,
    **season_values,
    **label_map,
    '{{CAREER_SLASH}}': career_slash or '—',
    '{{CAREER_SUBTITLE}}': career_subtitle,
    '{{CAREER_ROWS_HTML}}': career_rows_html,
    '{{EYEBROW}}': eyebrow,
    '{{HEADLINE}}': headline,
    '{{PHOTO_SRC}}': photo_src,
    'assets/foul-line-dirt.png': bg_url,
    'assets/64-circlelogo-red-white.png': circle_url,
    'assets/64-logo-red-white.png': mark_url,
    'assets/logo-college-baseball-show.png': logo_cbs_url,
    'assets/logo-jb.png': logo_jb_url,
    'assets/logo-64-analytics.png': logo_64a_url,
}
rendered = template
for k, v in replacements.items():
    rendered = rendered.replace(k, str(v))

# Inject html2canvas + PNG download button. Await document.fonts.ready and
# pre-decode all <img> tags before snapshotting, otherwise html2canvas
# captures the frame before the Google Fonts CSS resolves and the PNG
# falls back to system fonts. Button stays disabled while a render is in
# flight to prevent double-clicks producing duplicate downloads.
png_script = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<script>
window.downloadCardPNG = async function(btn) {
  var stage = document.querySelector('.stage');
  if (!stage) return;
  if (btn) { btn.disabled = true; btn.dataset.label = btn.textContent; btn.textContent = 'Rendering…'; }
  try {
    if (document.fonts && document.fonts.ready) {
      await document.fonts.ready;
    }
    var imgs = Array.from(stage.querySelectorAll('img'));
    await Promise.all(imgs.map(function(img) {
      if (img.complete && img.naturalWidth > 0) return Promise.resolve();
      return new Promise(function(res) {
        img.addEventListener('load', res, { once: true });
        img.addEventListener('error', res, { once: true });
      });
    }));
    var canvas = await html2canvas(stage, { scale: 2, useCORS: true, allowTaint: true });
    var a = document.createElement('a');
    a.download = 'portal_entrant.png';
    a.href = canvas.toDataURL('image/png');
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = btn.dataset.label || 'Download PNG'; }
  }
};
</script>
"""
png_btn = """
<div style="text-align:center;margin:16px 0;">
  <button onclick="window.downloadCardPNG(this)" style="
    padding:10px 24px;background:#b11f2c;color:#fff;border:none;border-radius:4px;
    font-family:'Inter','Inter',sans-serif;font-weight:700;font-size:14px;
    letter-spacing:.2em;text-transform:uppercase;cursor:pointer;
    box-shadow:0 4px 12px rgba(0,0,0,.3);">
    Download PNG
  </button>
</div>
"""
rendered = rendered.replace('</body>', f'{png_script}{png_btn}</body>')

st.subheader('3. Preview')
# Iframe height tracks the template's aspect so the preview doesn't crop or
# leave dead space. Add ~70px for the download-PNG button below.
_iframe_h = int(min(1080, max(640, TEMPLATE_H * 0.6))) + 70
components.html(rendered, height=_iframe_h, scrolling=False)
