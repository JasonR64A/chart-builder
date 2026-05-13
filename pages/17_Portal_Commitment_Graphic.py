"""
Portal Commitment Graphic Builder — produce a 1500x1000 commitment graphic
for a player who has committed to a new school. Auto-fills from
portal_rank_player + players.csv + hitting.csv/pitching.csv. All fields
editable; renders the design template; offers a PNG download.

Filter to committed players: portal_rank_player rows where year=2026 AND
new_team_id is not null. New-school color drives the crimson palette
(falls back to the design default #9D2235 when unknown).
"""
from __future__ import annotations

import base64
import mimetypes
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _APP_DIR / 'data'
ASSETS_DIR = _APP_DIR / 'assets' / 'portal-commitment'
ENTRANT_ASSETS = _APP_DIR / 'assets' / 'portal-entrant'  # share the 64A logo

TEMPLATE_PATH = ASSETS_DIR / 'template.html'
TEMPLATE_W, TEMPLATE_H = 1500, 1000

CLASS_LONG = {
    'Freshman': 'FRESHMAN', 'FR': 'FRESHMAN', 'FR.': 'FRESHMAN',
    'Sophomore': 'SOPHOMORE', 'SO': 'SOPHOMORE', 'SO.': 'SOPHOMORE',
    'Junior': 'JUNIOR', 'JR': 'JUNIOR', 'JR.': 'JUNIOR',
    'Senior': 'SENIOR', 'SR': 'SENIOR', 'SR.': 'SENIOR', 'Redshirt Senior': 'SENIOR',
    'Graduate': 'GRADUATE', 'Grad': 'GRADUATE', 'GR': 'GRADUATE', 'GR.': 'GRADUATE',
}

st.set_page_config(page_title='Portal Commitment Graphic', layout='wide')
st.title('Portal Commitment Graphic')
st.caption('Build a shareable commitment graphic for a player who has committed to a new school.')


# ── Data loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_players() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / 'players.csv', encoding='latin1', dtype=str).fillna('')


@st.cache_data
def load_teams() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False)
    return df


@st.cache_data
def load_conferences() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / 'conferences.csv', low_memory=False)


@st.cache_data
def load_hitting() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / 'hitting.csv', low_memory=False)


@st.cache_data
def load_pitching() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / 'pitching.csv', low_memory=False)


@st.cache_data
def load_portal_rank_player() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / 'portal_rank_player.csv', low_memory=False)


players = load_players()
teams = load_teams()
confs = load_conferences()
hitting = load_hitting()
pitching = load_pitching()
prp = load_portal_rank_player()


# ── Build the "committed players" pool ──────────────────────────────────────
# Filter to current-year 2026 rows where new_team_id is populated. Join the
# player display name from players.csv and the from/to team names.
prp_26 = prp[(prp['year'] == 2026) & prp['new_team_id'].notna()].copy()
prp_26['player_id'] = prp_26['player_id'].astype('Int64')
prp_26['team_id'] = prp_26['team_id'].astype('Int64')
prp_26['new_team_id'] = prp_26['new_team_id'].astype('Int64')

# Join player name (players.csv 'id' is string in our cached load)
players_idx = players.set_index('id')
team_idx = teams.set_index('id')
conf_idx = confs.set_index('id')


def _team_name(team_id) -> str:
    if pd.isna(team_id):
        return ''
    try:
        return str(team_idx.loc[int(team_id), 'name'])
    except Exception:
        return ''


def _team_conf_abbr(team_id) -> str:
    if pd.isna(team_id):
        return ''
    try:
        cid = team_idx.loc[int(team_id), 'conference_id']
        if pd.isna(cid):
            return ''
        return str(conf_idx.loc[int(cid), 'abbreviation'])
    except Exception:
        return ''


def _team_logo_url(team_id) -> str:
    if pd.isna(team_id):
        return ''
    try:
        u = team_idx.loc[int(team_id), 'logo_url']
        return str(u) if pd.notna(u) else ''
    except Exception:
        return ''


# Picker rows
def _pick_label(row) -> str:
    pid = int(row['player_id']) if pd.notna(row['player_id']) else 0
    name = str(players_idx.loc[str(pid), 'player_name']) if str(pid) in players_idx.index else f'player_id {pid}'
    from_team = _team_name(row['team_id'])
    to_team = _team_name(row['new_team_id'])
    rank = row.get('sixty_four_rating_portal_player')
    rank_str = f'#{int(rank)}' if pd.notna(rank) else 'unranked'
    return f'{name} — {from_team} → {to_team} ({rank_str})'


# Sort committed pool by portal rank (Int64 -> NA last)
prp_26_sorted = prp_26.sort_values(
    'sixty_four_rating_portal_player', na_position='last', kind='stable'
)
prp_26_sorted = prp_26_sorted.reset_index(drop=True)
prp_26_sorted['_label'] = prp_26_sorted.apply(_pick_label, axis=1)


# ── Player picker ───────────────────────────────────────────────────────────
st.subheader('1. Pick a committed player')
search = st.text_input('Search committed players by name / school',
                       placeholder='Start typing...', key='commit_search')
pool = prp_26_sorted
if search.strip():
    q = search.lower().strip()
    pool = prp_26_sorted[prp_26_sorted['_label'].str.lower().str.contains(q, na=False)]

if pool.empty:
    st.info('No committed players match — try a different search or clear the box.')
    st.stop()

pick = st.selectbox(f'Committed players ({len(pool):,})', pool['_label'].tolist(), key='commit_pick')
crow = pool.iloc[pool['_label'].tolist().index(pick)]

player_id = int(crow['player_id'])
from_team_id = int(crow['team_id']) if pd.notna(crow['team_id']) else None
to_team_id = int(crow['new_team_id']) if pd.notna(crow['new_team_id']) else None
portal_rank_default = int(crow['sixty_four_rating_portal_player']) if pd.notna(crow['sixty_four_rating_portal_player']) else 0

p_row = players[players['id'] == str(player_id)]
if p_row.empty:
    st.error(f'player_id {player_id} not found in players.csv')
    st.stop()
prec = p_row.iloc[0]


def _is_pitcher(player_row, pid_int: int) -> bool:
    pos = (player_row.get('position') or '').upper().strip()
    if pos in ('PH', 'PR'):
        return False
    if 'P' in pos and pos != '':
        return True
    return not pitching[pitching['player_id'] == pid_int].empty


is_pitcher_player = _is_pitcher(prec, player_id)


# ── Auto-fill defaults from career data ─────────────────────────────────────
def fmt_slash(v, digits=3):
    try:
        f = float(v)
        s = f'{f:.{digits}f}'
        if s.startswith('0.'):
            return s[1:]
        if s.startswith('-0.'):
            return '-' + s[2:]
        return s
    except Exception:
        return '—'


def fmt_rate(v, digits=2):
    try:
        f = float(v)
        if pd.isna(f):
            return '—'
        return f'{f:.{digits}f}'
    except Exception:
        return '—'


def season_int(row, key, default=0):
    if row is None or key not in row:
        return default
    v = row[key]
    if pd.isna(v):
        return default
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return default


def hitter_seasons(pid: int) -> pd.DataFrame:
    if hitting.empty:
        return pd.DataFrame()
    return hitting[hitting['player_id'] == pid].sort_values('year', ascending=False)


def pitcher_seasons(pid: int) -> pd.DataFrame:
    if pitching.empty:
        return pd.DataFrame()
    return pitching[pitching['player_id'] == pid].sort_values('year', ascending=False)


# Defaults
default_name = (prec.get('player_name') or '').strip()
default_pos = (prec.get('position') or '').strip().upper()
default_cls_raw = (prec.get('classification') or '').strip()
default_cls_long = CLASS_LONG.get(default_cls_raw, default_cls_raw.upper())
default_bat = (prec.get('bat') or '').strip().upper()
default_throw = (prec.get('throw') or '').strip().upper()

from_school = _team_name(from_team_id).upper() if from_team_id else ''
to_school = _team_name(to_team_id).upper() if to_team_id else ''
from_conf = _team_conf_abbr(from_team_id) if from_team_id else ''
to_conf = _team_conf_abbr(to_team_id) if to_team_id else ''
to_logo_url = _team_logo_url(to_team_id) if to_team_id else ''

# How many seasons at the FROM school (count distinct years with stats at team_id == from_team_id)
seasons_at_from = 0
if from_team_id:
    h_all = hitter_seasons(player_id)
    p_all = pitcher_seasons(player_id)
    yrs = set()
    if not h_all.empty:
        yrs.update(int(y) for y in h_all.loc[h_all['team_id'] == from_team_id, 'year'].dropna().unique())
    if not p_all.empty:
        yrs.update(int(y) for y in p_all.loc[p_all['team_id'] == from_team_id, 'year'].dropna().unique())
    seasons_at_from = len(yrs)

# 2026 season line
hitter_2026 = None
if not is_pitcher_player:
    h_all = hitter_seasons(player_id)
    if not h_all.empty:
        cur = h_all[h_all['year'] == 2026]
        hitter_2026 = cur.iloc[0] if not cur.empty else None

pitcher_2026 = None
if is_pitcher_player:
    p_all = pitcher_seasons(player_id)
    if not p_all.empty:
        cur = p_all[p_all['year'] == 2026]
        pitcher_2026 = cur.iloc[0] if not cur.empty else None


# ── Editable form ───────────────────────────────────────────────────────────
st.subheader('2. Review & edit fields')

with st.expander('Identity', expanded=True):
    c1, c2, c3 = st.columns([3, 1, 1])
    name = c1.text_input('Player name (use line break "\\n" to wrap)',
                         value=default_name.upper().replace(' ', '\n', 1))
    pos = c2.text_input('Position', value=default_pos or 'OF')
    class_long = c3.text_input('Class', value=default_cls_long or 'JUNIOR')

    c4, c5, c6 = st.columns([1, 1, 2])
    bat = c4.text_input('Bats', value=default_bat or 'R')
    thr = c5.text_input('Throws', value=default_throw or 'R')
    hand_badge = c6.text_input('Class · Hand badge',
                               value=f'{class_long} · {bat}/{thr}' if class_long else f'{bat}/{thr}')

    photo_upload = st.file_uploader('Player photo (upload to replace placeholder)',
                                    type=['jpg', 'jpeg', 'png', 'webp'])
    if photo_upload is not None:
        st.image(photo_upload, caption=f'Uploaded · {photo_upload.size:,} bytes', width=180)

with st.expander('Schools (FROM → TO)', expanded=True):
    c1, c2, c3 = st.columns([2, 1, 1])
    from_school_in = c1.text_input('FROM school', value=from_school)
    from_conf_in = c2.text_input('FROM conference (abbr)', value=from_conf or '')
    from_sub_seasons = c3.text_input('FROM subtitle',
                                     value=f'{from_conf or ""} · {seasons_at_from} {"SEASON" if seasons_at_from==1 else "SEASONS"}' if from_conf else f'{seasons_at_from} {"SEASON" if seasons_at_from==1 else "SEASONS"}')

    c4, c5, c6 = st.columns([2, 1, 1])
    to_school_in = c4.text_input('TO school', value=to_school)
    to_conf_in = c5.text_input('TO conference (abbr)', value=to_conf or '')
    next_year = date.today().year + 1
    to_sub = c6.text_input('TO subtitle', value=f'{to_conf or ""} · {next_year}' if to_conf else f'{next_year}')

    to_logo_upload = st.file_uploader(
        'TO school logo override (optional — leaves the S3 URL as default)',
        type=['png', 'jpg', 'jpeg', 'webp', 'svg'],
        key='to_logo_upload',
    )

with st.expander('Rank · Date · Eyebrow', expanded=True):
    c1, c2, c3 = st.columns(3)
    portal_rank = c1.number_input('Portal rank (No.)', min_value=0, max_value=99999,
                                  value=int(portal_rank_default))
    commit_date_dt = c2.date_input('Commit date', value=date.today())
    rank_src = c3.text_input('Rank source', value='64 ANALYTICS')

    c4, c5 = st.columns(2)
    eyebrow = c4.text_input('Eyebrow (above COMMITTED)', value='OFFICIAL COMMITMENT')
    kicker = c5.text_input('Top-right kicker', value='TRANSFER PORTAL · COMMITMENT')

with st.expander('Stats strip (5 cells)', expanded=True):
    # Auto-derive defaults from the player's 2026 hitter row (or pitcher row)
    if not is_pitcher_player and hitter_2026 is not None:
        avg = fmt_slash(hitter_2026.get('batting_average'))
        obp = fmt_slash(hitter_2026.get('on_base_percentage'))
        slg = fmt_slash(hitter_2026.get('slugging_percentage'))
        ops = fmt_slash(hitter_2026.get('on_base_plus_slugging'))
        gp = season_int(hitter_2026, 'games_played')

        # wRAA from hitting.csv. wRAA can be negative; show one decimal and
        # surface the national percentile as the sub-line.
        wraa_raw = pd.to_numeric(hitter_2026.get('weighted_runs_above_average'), errors='coerce')
        wraa_str = f'{float(wraa_raw):+.1f}' if pd.notna(wraa_raw) else '—'
        wraa_pct = pd.to_numeric(hitter_2026.get('percentile_rank_weighted_runs_above_average'), errors='coerce')
        wraa_sub = (f'TOP {100 - int(round(float(wraa_pct)*100))}% NTL'
                    if pd.notna(wraa_pct) else '')

        # wRC+ from hitting.csv. Display as integer; reference line is 100=avg.
        wrcp_raw = pd.to_numeric(hitter_2026.get('weighted_runs_created_plus'), errors='coerce')
        wrcp_str = f'{int(round(float(wrcp_raw)))}' if pd.notna(wrcp_raw) else '—'
        wrcp_sub = 'VS 100 LG AVG'

        # wRCE lives in player_rank.csv, joined by (player_id, year=2026). Show
        # the 64A national rank as the sub-line when available.
        try:
            pr_row = prp[(prp['player_id'] == player_id) & (prp['year'] == 2026)]
            pr_2026 = pr_row.iloc[0] if not pr_row.empty else None
        except Exception:
            pr_2026 = None
        # Note: pr is the local portal_rank_player frame; we actually want
        # player_rank.csv (full hitter ranking) here. Load it lazily so the
        # page doesn't pay the cost when not needed.
        @st.cache_data
        def _load_player_rank():
            return pd.read_csv(DATA_DIR / 'player_rank.csv', low_memory=False,
                               usecols=['player_id', 'year',
                                        'weighted_run_created_efficiency',
                                        'sixty_four_rank_weighted_run_created_efficiency',
                                        'percentile_rank_weighted_run_created_efficiency'])
        pr_full = _load_player_rank()
        pr_match = pr_full[(pr_full['player_id'] == player_id) & (pr_full['year'] == 2026)]
        wrce_raw = pd.to_numeric(pr_match['weighted_run_created_efficiency'].iloc[0], errors='coerce') if not pr_match.empty else None
        wrce_str = f'{float(wrce_raw):.2f}' if pd.notna(wrce_raw) else '—'
        # Use the percentile column directly — the sixty_four_rank field is 0
        # for players outside the ranked pool, which renders as a misleading
        # "RANK #0" in the sub-line.
        wrce_pct = pd.to_numeric(pr_match['percentile_rank_weighted_run_created_efficiency'].iloc[0], errors='coerce') if not pr_match.empty else None
        wrce_sub = (f'TOP {max(1, 100 - int(round(float(wrce_pct)*100)))}% NTL'
                    if pd.notna(wrce_pct) else '')

        d_hero_lab = '2026 SLASH LINE'
        d_hero_val = f'{avg} / {obp} / {slg}'
        d_hero_sub = f'AVG · OBP · SLG · {gp} GP'
        d_s1_lab, d_s1_val, d_s1_sub = 'OPS', ops, ''
        d_s2_lab, d_s2_val, d_s2_sub = 'wRAA', wraa_str, wraa_sub
        d_s3_lab, d_s3_val_html, d_s3_sub = 'wRC+', wrcp_str, wrcp_sub
        d_s4_lab, d_s4_val, d_s4_sub = 'wRCE', wrce_str, wrce_sub
    elif is_pitcher_player and pitcher_2026 is not None:
        era = fmt_rate(pitcher_2026.get('earned_run_average'), 2)
        whip = fmt_rate(pitcher_2026.get('walks_plus_hits_per_inning_pitched'), 2)
        ip = fmt_rate(pitcher_2026.get('innings_pitched'), 1)
        kp = pitcher_2026.get('strikeout_percentage')
        kp_str = f'{float(kp)*100:.1f}%' if pd.notna(kp) else '—'
        k = season_int(pitcher_2026, 'strikeouts')
        bb = season_int(pitcher_2026, 'walks_issued')
        w = season_int(pitcher_2026, 'wins')
        l = season_int(pitcher_2026, 'losses')
        g = season_int(pitcher_2026, 'games_appeared') or season_int(pitcher_2026, 'appearances')

        d_hero_lab = '2026 PITCHING LINE'
        d_hero_val = f'{era} / {whip} / {kp_str}'
        d_hero_sub = f'ERA · WHIP · K% · {g} G'
        d_s1_lab, d_s1_val, d_s1_sub = 'IP', ip, ''
        d_s2_lab, d_s2_val, d_s2_sub = 'K', str(k), f'{kp_str} K%'
        d_s3_lab, d_s3_val_html, d_s3_sub = ('K / BB',
            f'{k}<span style="color:var(--crimson-hot); font-size:32px; padding: 0 4px;">/</span>{bb}',
            f'{(k/bb if bb else 0):.2f} RATIO')
        d_s4_lab, d_s4_val, d_s4_sub = 'W-L', f'{w}-{l}', ''
    else:
        d_hero_lab = '2026 SLASH LINE'
        d_hero_val = '.000 / .000 / .000'
        d_hero_sub = 'AVG · OBP · SLG · 0 GP'
        d_s1_lab, d_s1_val, d_s1_sub = 'OPS', '.000', ''
        d_s2_lab, d_s2_val, d_s2_sub = 'HR', '0', '0 RBI'
        d_s3_lab, d_s3_val_html, d_s3_sub = 'BB / K', '0/0', '0.00 RATIO'
        d_s4_lab, d_s4_val, d_s4_sub = 'SB', '0', '0 CS'

    c1, c2, c3 = st.columns([2, 2, 2])
    hero_lab = c1.text_input('Hero · label', value=d_hero_lab)
    hero_val = c2.text_input('Hero · value', value=d_hero_val)
    hero_sub = c3.text_input('Hero · sub', value=d_hero_sub)

    c1, c2, c3 = st.columns(3)
    s1_lab = c1.text_input('Cell 1 · label', value=d_s1_lab)
    s1_val = c2.text_input('Cell 1 · value', value=d_s1_val)
    s1_sub = c3.text_input('Cell 1 · sub', value=d_s1_sub)

    c1, c2, c3 = st.columns(3)
    s2_lab = c1.text_input('Cell 2 · label', value=d_s2_lab)
    s2_val = c2.text_input('Cell 2 · value', value=d_s2_val)
    s2_sub = c3.text_input('Cell 2 · sub', value=d_s2_sub)

    c1, c2, c3 = st.columns(3)
    s3_lab = c1.text_input('Cell 3 · label', value=d_s3_lab)
    s3_val_html = c2.text_input('Cell 3 · value (HTML ok)', value=d_s3_val_html)
    s3_sub = c3.text_input('Cell 3 · sub', value=d_s3_sub)

    c1, c2, c3 = st.columns(3)
    s4_lab = c1.text_input('Cell 4 · label', value=d_s4_lab)
    s4_val = c2.text_input('Cell 4 · value', value=d_s4_val)
    s4_sub = c3.text_input('Cell 4 · sub', value=d_s4_sub)


# ── Render ──────────────────────────────────────────────────────────────────
def data_url_from_bytes(b: bytes, mime: str) -> str:
    return f'data:{mime};base64,' + base64.b64encode(b).decode()


def data_url_from_path(p: Path) -> str:
    if not p.exists():
        return ''
    mime, _ = mimetypes.guess_type(str(p))
    return data_url_from_bytes(p.read_bytes(), mime or 'image/png')


def _resize_image_bytes(raw: bytes, mime: str, max_long_side: int = 1600) -> tuple[bytes, str]:
    if not raw or len(raw) <= 600_000:
        return raw, mime
    try:
        from PIL import Image
        import io as _io
        im = Image.open(_io.BytesIO(raw))
        im.thumbnail((max_long_side, max_long_side), Image.LANCZOS)
        buf = _io.BytesIO()
        if mime == 'image/png':
            im.save(buf, format='PNG', optimize=True)
        elif mime == 'image/webp':
            im.save(buf, format='WEBP', quality=88)
        else:
            im.convert('RGB').save(buf, format='JPEG', quality=88, optimize=True)
            mime = 'image/jpeg'
        return buf.getvalue(), mime
    except Exception as e:
        st.warning(f'Could not resize image ({e}); using original.')
        return raw, mime


# Player photo
photo_src = ''
if photo_upload is not None:
    photo_bytes = photo_upload.getvalue()
    mime = (photo_upload.type or '').strip()
    if not mime or not mime.startswith('image/'):
        ext = (Path(photo_upload.name or '').suffix.lower().lstrip('.')) if photo_upload.name else ''
        mime = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
                'png': 'image/png',  'webp': 'image/webp'}.get(ext, 'image/png')
    photo_bytes, mime = _resize_image_bytes(photo_bytes, mime)
    if photo_bytes:
        photo_src = data_url_from_bytes(photo_bytes, mime)
        st.caption(f'Photo data URL: {len(photo_bytes):,} bytes ({mime})')

# Fall back to sample photo from entrant assets if no upload
if not photo_src:
    sample = ENTRANT_ASSETS / 'sample-photo.jpg'
    if sample.exists():
        photo_src = data_url_from_path(sample)

# TO logo
to_logo_src = ''
if to_logo_upload is not None:
    raw = to_logo_upload.getvalue()
    mime = (to_logo_upload.type or '').strip() or 'image/png'
    raw, mime = _resize_image_bytes(raw, mime, max_long_side=400)
    if raw:
        to_logo_src = data_url_from_bytes(raw, mime)
elif to_logo_url:
    # Fetch the S3 logo and embed (html2canvas needs same-origin / data URLs).
    try:
        import urllib.request
        with urllib.request.urlopen(to_logo_url, timeout=8) as resp:
            raw = resp.read()
        mime_guess, _ = mimetypes.guess_type(to_logo_url)
        to_logo_src = data_url_from_bytes(raw, mime_guess or 'image/png')
    except Exception as e:
        st.warning(f'Could not fetch TO logo from S3 ({e}). Upload one manually if needed.')

# 64 Analytics logo
logo_src = data_url_from_path(ENTRANT_ASSETS / '64-logo-red-white.png')

# Colors — default to the design's crimson; the design's effects depend on
# rgba() forms of these values, so when the user overrides the new-school
# color we still need to derive the alpha variants from the hex.
def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip('#')
    if len(h) == 3:
        h = ''.join(c*2 for c in h)
    try:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except Exception:
        return 0x9D, 0x22, 0x35  # design default


crimson_hex = '#9D2235'           # design default — keeps the cinematic palette
crimson_deep = '#6e1624'
crimson_hot = '#c32a41'
r, g, b = _hex_to_rgb(crimson_hex)
crimson_fade = f'rgba({r},{g},{b},0.35)'
crimson_glow = f'rgba({r},{g},{b},0.45)'
crimson_bg_hero_a = f'rgba({r},{g},{b},0.96)'
r2, g2, b2 = _hex_to_rgb(crimson_deep)
crimson_bg_hero_b = f'rgba({r2},{g2},{b2},0.96)'

# Date formatting
commit_date_str = commit_date_dt.strftime('%b %d, %Y').upper()

# NAME html — let the user break on \n
name_html = name.replace('\n', '<br/>')

template = TEMPLATE_PATH.read_text(encoding='utf-8')
replacements = {
    '{{NAME}}': name.replace('\n', ' '),
    '{{NAME_HTML}}': name_html,
    '{{POS}}': pos,
    '{{HAND_BADGE}}': hand_badge.upper(),
    '{{KICKER}}': kicker.upper(),
    '{{COMMIT_DATE}}': commit_date_str,
    '{{EYEBROW}}': eyebrow.upper(),
    '{{FROM_SCHOOL}}': from_school_in.upper(),
    '{{FROM_SUB}}': from_sub_seasons.upper(),
    '{{TO_SCHOOL}}': to_school_in.upper(),
    '{{TO_SUB}}': to_sub.upper(),
    '{{PORTAL_RANK}}': str(int(portal_rank)),
    '{{RANK_SRC}}': rank_src.upper(),
    '{{HERO_LAB}}': hero_lab.upper(),
    '{{HERO_VAL}}': hero_val,
    '{{HERO_SUB}}': hero_sub.upper(),
    '{{S1_LAB}}': s1_lab.upper(),
    '{{S1_VAL}}': s1_val,
    '{{S1_SUB}}': s1_sub.upper(),
    '{{S2_LAB}}': s2_lab.upper(),
    '{{S2_VAL}}': s2_val,
    '{{S2_SUB}}': s2_sub.upper(),
    '{{S3_LAB}}': s3_lab.upper(),
    '{{S3_VAL_HTML}}': s3_val_html,
    '{{S3_SUB}}': s3_sub.upper(),
    '{{S4_LAB}}': s4_lab.upper(),
    '{{S4_VAL}}': s4_val,
    '{{S4_SUB}}': s4_sub.upper(),
    '{{FOOTER_TAG}}': '64 ANALYTICS · @64ANALYTICS',
    '{{PHOTO_SRC}}': photo_src,
    '{{TO_LOGO_SRC}}': to_logo_src,
    '{{LOGO_SRC}}': logo_src,
    '{{CRIMSON}}': crimson_hex,
    '{{CRIMSON_DEEP}}': crimson_deep,
    '{{CRIMSON_HOT}}': crimson_hot,
    '{{CRIMSON_FADE}}': crimson_fade,
    '{{CRIMSON_GLOW}}': crimson_glow,
    '{{CRIMSON_BG_HERO_A}}': crimson_bg_hero_a,
    '{{CRIMSON_BG_HERO_B}}': crimson_bg_hero_b,
}
rendered = template
for k, v in replacements.items():
    rendered = rendered.replace(k, str(v))

# html2canvas PNG download — same pattern as the entrant page.
png_script = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<script>
window.downloadCardPNG = async function(btn) {
  var stage = document.querySelector('.stage');
  if (!stage) return;
  if (btn) { btn.disabled = true; btn.dataset.label = btn.textContent; btn.textContent = 'Rendering…'; }
  try {
    if (document.fonts && document.fonts.ready) await document.fonts.ready;
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
    a.download = 'portal_commitment.png';
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
    padding:10px 24px;background:#9D2235;color:#fff;border:none;border-radius:4px;
    font-family:'Inter',sans-serif;font-weight:700;font-size:14px;
    letter-spacing:.2em;text-transform:uppercase;cursor:pointer;
    box-shadow:0 4px 12px rgba(0,0,0,.3);">
    Download PNG
  </button>
</div>
"""
rendered = rendered.replace('</body>', f'{png_script}{png_btn}</body>')

st.subheader('3. Preview')
_iframe_h = int(min(1080, max(640, TEMPLATE_H * 0.6))) + 70
components.html(rendered, height=_iframe_h, scrolling=False)
