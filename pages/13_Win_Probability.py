"""
Win Probability Pace Chart — play-by-play WP curve for any game.

Starting WP uses log5 of 64 integer ranks ± 4% home field advantage.
In-game WP is an empirical state lookup (inning, half, outs, runners,
score_diff) blended with the pre-game anchor: early-game weight on
pre-game, late-game weight on state.

Built 2026-04-15. Data: 2026 D1 play-by-play (~5,200 games, 520K plays).
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
from pathlib import Path
from io import BytesIO
import plotly.graph_objects as go
from PIL import Image
from collections import Counter

_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _APP_DIR / 'data'
PBP_DIR = _APP_DIR / 'pbp_data' / 'play_by_play'
LOOKUP_FILE = _APP_DIR / 'pbp_data' / 'wp_state_lookup_bb_d1.pkl'  # shared fallback — baseball D1 model
LOGO_DIR = _APP_DIR / 'team_logos_512'
BRAND_LOGO = _APP_DIR / 'assets' / 'brand_logo_wide.png'

# Division labels: short codes used in file names, readable labels for UI
DIVISION_LABELS = {'D1': 'D-I', 'D2': 'D-II', 'D3': 'D-III'}
SPORT_OPTIONS = ['Baseball', 'Softball']
DIVISION_OPTIONS = ['D1', 'D2', 'D3']


def pbp_paths(sport, division):
    sport_l = sport.lower()
    base = PBP_DIR / f'{sport_l}_play_by_play_{division}'
    return base.with_suffix('.csv'), base.with_suffix('.csv.gz')


def hitting_pbp_path(sport, division):
    return _APP_DIR / 'pbp_data' / sport.lower() / f'hitting_pbp_{division}.csv'

# Pre-game WP constants + helpers live in the shared model module so every
# page that projects games (Win Probability, Win Generator, Predicted RPI)
# uses the exact same math.
from pages._win_prob_model import (
    HOME_FIELD_ADVANTAGE, PREGAME_EDGE_SCALE,
    PREGAME_CLAMP_MIN, PREGAME_CLAMP_MAX,
    TEAM_RANK_BLEND, PITCHING_WEIGHT,
    log5 as _shared_log5,
    pre_game_wp as _shared_pre_game_wp,
    build_team_profiles as _shared_build_team_profiles,
    adjusted_team_pct as _shared_adjusted_team_pct,
    blend_with_static as _shared_blend_with_static,
)

# In-game clamps — different from pre-game (blowouts should be able to read near-certain)
CLAMP_MIN = 0.01
CLAMP_MAX = 0.99

# 64Analytics brand colors (matches other chart pages)
BG_COLOR = '#FAF8F2'       # warm off-white
TEXT_COLOR = '#2D2926'
TEXT_MUTED = '#4A4540'
GRID_COLOR = '#D8D2C4'
HOME_COLOR = '#2E5C8A'     # navy blue
AWAY_COLOR = '#C41230'     # cardinal red


# ── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_pbp(sport='Baseball', division='D1'):
    cols = ['gameId', 'date', 'awayTeam', 'homeTeam', 'inning', 'halfInning',
            'battingTeam',
            'outs', 'runner1B', 'runner2B', 'runner3B',
            'awayScore', 'homeScore', 'player', 'playDescription']
    pbp_csv, pbp_gz = pbp_paths(sport, division)
    if pbp_csv.exists():
        df = pd.read_csv(pbp_csv, low_memory=False, usecols=cols)
    elif pbp_gz.exists():
        df = pd.read_csv(pbp_gz, low_memory=False, usecols=cols, compression='gzip')
    else:
        return None
    df['inning'] = pd.to_numeric(df['inning'], errors='coerce').fillna(0).astype(int)
    df['outs'] = pd.to_numeric(df['outs'], errors='coerce').fillna(0).astype(int)
    df['awayScore'] = pd.to_numeric(df['awayScore'], errors='coerce').fillna(0).astype(int)
    df['homeScore'] = pd.to_numeric(df['homeScore'], errors='coerce').fillna(0).astype(int)
    # Derive the correct half from battingTeam. The scraper's halfInning
    # label is swapped ("bottom" is recorded for top-of-inning plays where
    # away team bats). Use battingTeam as ground truth.
    # true_half = 'top' if awayTeam is batting, else 'bottom'
    df['half_true'] = np.where(df['battingTeam'] == df['awayTeam'], 'top', 'bottom')
    return df


@st.cache_data
def load_lookup():
    with open(LOOKUP_FILE, 'rb') as f:
        return pickle.load(f)


@st.cache_data
def load_teams(sport='Baseball', division='D1'):
    t = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False)
    t['id'] = pd.to_numeric(t['id'], errors='coerce').fillna(0).astype(int)
    tr = pd.read_csv(DATA_DIR / 'team_rank.csv', low_memory=False)
    tr = tr[tr['year'] == 2026].copy()
    tr['rank'] = pd.to_numeric(tr['integer_64_rank_total'], errors='coerce')
    confs = pd.read_csv(DATA_DIR / 'conferences.csv', low_memory=False)
    div_label = DIVISION_LABELS.get(division, division)
    div_ids = set(confs[confs['division'] == div_label]['id'])
    bucket = t[(t['sport'] == sport) & (t['conference_id'].isin(div_ids))]
    tr_b = tr[tr['team_id'].isin(bucket['id'])]
    N = len(tr_b)
    tr_b = tr_b.dropna(subset=['rank']).copy()
    if N < 2:
        return {}, {}
    tr_b['rank_pct'] = 1 - (tr_b['rank'] - 1) / (N - 1)
    name_to_pct = dict(zip(bucket.set_index('id').loc[tr_b['team_id'].values, 'name'],
                            tr_b['rank_pct']))
    name_to_rank = dict(zip(bucket.set_index('id').loc[tr_b['team_id'].values, 'name'],
                             tr_b['rank'].astype(int)))
    return name_to_pct, name_to_rank


def build_team_profiles(sport='Baseball', division='D1'):
    """Thin wrapper — real implementation in pages/_win_prob_model.py."""
    return _shared_build_team_profiles(sport, division)


def _legacy_build_team_profiles_inline(sport='Baseball', division='D1'):
    """Retained for reference only; not called."""
    try:
        pr = pd.read_csv(DATA_DIR / 'player_rank.csv', low_memory=False,
                         usecols=['player_id','team_id','year',
                                  'percentile_rank_weighted_run_created_efficiency',
                                  'percentile_rank_weighted_run_allowed_efficiency'])
        pr['year'] = pd.to_numeric(pr['year'], errors='coerce')
        pr = pr.sort_values('year').drop_duplicates('player_id', keep='last')
        pr['team_id'] = pd.to_numeric(pr['team_id'], errors='coerce').astype('Int64')
        pr['hit'] = pd.to_numeric(pr['percentile_rank_weighted_run_created_efficiency'], errors='coerce')
        pr['pit'] = pd.to_numeric(pr['percentile_rank_weighted_run_allowed_efficiency'], errors='coerce')

        teams = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False, dtype=str).fillna('')
        teams = teams[teams['sport'] == sport]
        teams['id_int'] = pd.to_numeric(teams['id'], errors='coerce').astype('Int64')
        id_to_name = dict(zip(teams['id_int'], teams['name']))

        # PA usage proxy: games-played in last 30 days from the matching sport+division hitting_pbp
        hit_pbp_file = hitting_pbp_path(sport, division)
        pa_counts = {}
        if hit_pbp_file.exists():
            hp = pd.read_csv(hit_pbp_file, low_memory=False, usecols=['date','playerId'])
            hp['date'] = pd.to_datetime(hp['date'], errors='coerce', format='mixed')
            cutoff = hp['date'].max() - pd.Timedelta(days=30)
            hp = hp[hp['date'] >= cutoff]
            pa_counts = hp.groupby('playerId').size().to_dict()

        profiles = {}
        for tid, grp in pr.groupby('team_id'):
            name = id_to_name.get(tid)
            if not name: continue
            # Pitchers: those with a pitching percentile, sorted desc
            pit_df = grp.dropna(subset=['pit'])
            pit_df = pit_df[pit_df['pit'] > 0]
            pit_list = pit_df.sort_values('pit', ascending=False)['pit'].tolist()
            # Hitters: those with a hitting percentile, sorted by last-30d usage
            hit_df = grp.dropna(subset=['hit']).copy()
            hit_df = hit_df[hit_df['hit'] > 0]
            hit_df['usage'] = hit_df['player_id'].map(pa_counts).fillna(0)
            hit_list = hit_df.sort_values(['usage','hit'], ascending=[False, False])['hit'].tolist()
            profiles[name] = {'pitchers': pit_list, 'hitters_by_pa': hit_list}
        return profiles
    except Exception as e:
        print(f'build_team_profiles failed: {e}')
        return {}


def adjusted_team_pct(name, profiles, game_type, game_num, static_pct):
    """Thin wrapper — real implementation in pages/_win_prob_model.py."""
    return _shared_adjusted_team_pct(name, profiles, game_type, game_num, static_pct)


@st.cache_data
def brand_logo_b64():
    if BRAND_LOGO.exists():
        with open(BRAND_LOGO, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    return None


@st.cache_data
def team_logo_map():
    """Build team name → 64A team id for locating logo files.
    Covers both Baseball and Softball (logo files are shared by id)."""
    t = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False)
    t['id'] = pd.to_numeric(t['id'], errors='coerce').fillna(0).astype(int)
    both = t[t['sport'].isin(['Baseball','Softball'])][['name', 'id']].drop_duplicates('name')
    return dict(zip(both['name'], both['id']))


def team_logo_path(team_full_name):
    """Find the logo file for a team (accounting for trailing mascot names).
    Returns None if team_full_name isn't a usable string (e.g. NaN from a
    game row where team data was missing)."""
    if not isinstance(team_full_name, str) or not team_full_name:
        return None
    logo_map = team_logo_map()
    for cand in [team_full_name, short_team(team_full_name)]:
        if cand in logo_map:
            for ext in ('png', 'webp'):
                p = LOGO_DIR / f'{logo_map[cand]}.{ext}'
                if p.exists():
                    return p
    # Fuzzy: try substring match
    for name, tid in logo_map.items():
        if not isinstance(name, str):
            continue
        if name in team_full_name or team_full_name in name:
            for ext in ('png', 'webp'):
                p = LOGO_DIR / f'{tid}.{ext}'
                if p.exists():
                    return p
    return None


def team_color(team_full_name, fallback='#C41230'):
    """Extract dominant non-black-non-white color from team logo."""
    p = team_logo_path(team_full_name)
    if not p:
        return fallback
    try:
        img = Image.open(p).convert('RGBA')
        img.thumbnail((64, 64))
        px = np.array(img)
        mask = px[:, :, 3] > 128
        rgb = px[mask][:, :3]
        if len(rgb) == 0:
            return fallback
        filtered = []
        for r, g, b in rgb:
            bright = (int(r) + int(g) + int(b)) / 3
            if bright > 220 or bright < 35:
                continue
            filtered.append((r, g, b))
        if not filtered:
            return fallback
        quant = [(r // 16 * 16, g // 16 * 16, b // 16 * 16) for r, g, b in filtered]
        top = Counter(quant).most_common(1)[0][0]
        return f'#{top[0]:02x}{top[1]:02x}{top[2]:02x}'
    except Exception:
        return fallback


def rgba_from_hex(hex_color, alpha=0.22):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


@st.cache_data
def faded_logo_b64(team_full_name, size=160, alpha=0.16):
    """Load a team's logo at given size with reduced opacity. Not rotated —
    used as a wallpaper-style tile across the chart."""
    p = team_logo_path(team_full_name)
    if not p:
        return None
    try:
        img = Image.open(p).convert('RGBA')
        img.thumbnail((size, size), Image.LANCZOS)
        alpha_layer = img.split()[3]
        alpha_layer = alpha_layer.point(lambda a: int(a * alpha))
        img.putalpha(alpha_layer)
        buf = BytesIO()
        img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def find_all_segments(home_wps, want_home_ahead, min_len=5):
    """Return list of (start_idx, end_idx) stretches where:
       - want_home_ahead=True  → WP > 50
       - want_home_ahead=False → WP < 50
    Only returns segments of at least `min_len` plays."""
    segments = []
    cur_start = None
    for i, wp in enumerate(home_wps):
        ahead = wp > 50 if want_home_ahead else wp < 50
        if ahead:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None and (i - cur_start) >= min_len:
                segments.append((cur_start, i - 1))
            cur_start = None
    if cur_start is not None and (len(home_wps) - cur_start) >= min_len:
        segments.append((cur_start, len(home_wps) - 1))
    return segments


# Thin wrappers so existing call sites keep working
def log5(wa, wb):
    return _shared_log5(wa, wb)


def pre_game_wp(home_pct, away_pct):
    return _shared_pre_game_wp(home_pct, away_pct)


def state_key(row):
    # Lookup uses the raw halfInning (consistent with how the table was
    # built). The raw label is swapped but applied consistently across
    # all games, so the probabilities still resolve correctly.
    inning = min(int(row['inning']), 10) if pd.notna(row['inning']) else 1
    half = 1 if row.get('halfInning') == 'bottom' else 0
    outs = min(int(row['outs']), 2) if pd.notna(row['outs']) else 0
    bases = int(pd.notna(row.get('runner1B'))) + \
            int(pd.notna(row.get('runner2B'))) * 2 + \
            int(pd.notna(row.get('runner3B'))) * 4
    sd = int(row['homeScore']) - int(row['awayScore'])
    sd = max(-10, min(10, sd))
    return (inning, half, outs, bases, sd)


def short_team(name):
    for sfx in [' Eagles', ' Tigers', ' Bulldogs', ' Wildcats', ' Hokies', ' Crimson Tide',
                ' Volunteers', ' Razorbacks', ' Commodores', ' Rebels', ' Gators', ' Longhorns',
                ' Aggies', ' Bears', ' Hawks', ' Owls', ' Ducks', ' Beavers', ' Huskies',
                ' Jayhawks', ' Cornhuskers', ' Sooners', ' Cowboys', ' Cyclones']:
        if name.endswith(sfx):
            return name[:-len(sfx)]
    parts = name.rsplit(' ', 1)
    return parts[0] if len(parts) == 2 else name


def find_team(name, team_pct, team_rank):
    for cand in [name, short_team(name)]:
        if cand in team_pct:
            return cand, team_pct[cand], team_rank.get(cand)
    # Longest prefix match to avoid "Oklahoma St." -> "Oklahoma" errors
    best_k, best_len = None, 0
    for k in team_pct.keys():
        if name.startswith(k) and len(k) > best_len:
            best_k, best_len = k, len(k)
        elif k.startswith(short_team(name)) and len(short_team(name)) > best_len:
            best_k, best_len = k, len(short_team(name))
    if best_k:
        return best_k, team_pct[best_k], team_rank.get(best_k)
    return name, None, None


# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='Win Probability', layout='wide')
st.title('Win Probability — Pace Chart')
st.caption('Play-by-play WP curve. Starting point = log5 of 64 integer ranks ± 4% home-field bump. '
           'In-game WP = empirical state lookup blended with pre-game anchor (weight shifts to state as game progresses).')

# Sport + division selectors drive every data load below.
st.sidebar.markdown('### Sport / Division')
sel_sport = st.sidebar.selectbox('Sport', SPORT_OPTIONS, index=0, key='wp_sport')
sel_division = st.sidebar.selectbox('Division', DIVISION_OPTIONS, index=0, key='wp_division')

pbp = load_pbp(sel_sport, sel_division)
if pbp is None:
    st.warning(f'Play-by-play data for {sel_sport} {sel_division} not available.')
    st.stop()

lookup = load_lookup()
if sel_sport != 'Baseball' or sel_division != 'D1':
    st.sidebar.caption('ℹ️ In-game WP uses the baseball D1 state model. Sport/division-specific models coming later.')

team_pct, team_rank = load_teams(sel_sport, sel_division)

# Game picker
st.sidebar.markdown('### Game selection')
teams_in_data = sorted(set(pbp['awayTeam'].dropna()) | set(pbp['homeTeam'].dropna()))
default_team = 'Boston College Eagles' if 'Boston College Eagles' in teams_in_data else teams_in_data[0]
sel_team = st.sidebar.selectbox('Team', teams_in_data,
                                 index=teams_in_data.index(default_team))

team_games = pbp[(pbp['awayTeam'] == sel_team) | (pbp['homeTeam'] == sel_team)]
game_list = team_games.groupby('gameId').first().reset_index()
game_list['date_parsed'] = pd.to_datetime(game_list['date'], format='mixed', errors='coerce')
game_list = game_list.sort_values('date_parsed', ascending=False)
# Label each game; add "G1/G2" suffix for doubleheaders (same date + same matchup)
game_list['base_label'] = game_list.apply(
    lambda r: f"{r['date']}  {r['awayTeam']} @ {r['homeTeam']}", axis=1)
dup_mask = game_list.duplicated(subset='base_label', keep=False)
game_list['_dh_num'] = game_list.groupby('base_label').cumcount() + 1
game_list['label'] = game_list.apply(
    lambda r: f"{r['base_label']} (G{r['_dh_num']})" if dup_mask[r.name] else r['base_label'],
    axis=1)
sel_label = st.sidebar.selectbox('Game', game_list['label'].tolist())
sel_gid = int(game_list[game_list['label'] == sel_label]['gameId'].iloc[0])

game = pbp[pbp['gameId'] == sel_gid].copy()
# The scraper writes all bottom-of-inning plays BEFORE the top-of-inning
# plays within each inning. That breaks chronological order. Sort so that
# top precedes bottom within every inning. Stable sort preserves the
# correct within-half play sequence (outs already increase monotonically).
game['_half_order'] = (game['half_true'] == 'bottom').astype(int)
game = (game.sort_values(['inning', '_half_order'], kind='stable')
             .drop(columns=['_half_order'])
             .reset_index(drop=True))
home = game['homeTeam'].iloc[0]
away = game['awayTeam'].iloc[0]
if not isinstance(home, str) or not isinstance(away, str):
    st.warning(f'Selected game has missing team data (home={home!r}, away={away!r}). Pick a different game.')
    st.stop()

home_key, home_p, home_r = find_team(home, team_pct, team_rank)
away_key, away_p, away_r = find_team(away, team_pct, team_rank)

# Auto-detect Weekend vs Midweek + series game # from the PBP data.
# A game is part of a 3-game series if the same two teams play >=2 times
# within a 4-day window around the selected date. Otherwise it's midweek.
sel_date = pd.to_datetime(game['date'].iloc[0], format='mixed', errors='coerce')
series_games = pbp[((pbp['homeTeam'] == home) & (pbp['awayTeam'] == away)) |
                   ((pbp['homeTeam'] == away) & (pbp['awayTeam'] == home))].copy()
series_games['_d'] = pd.to_datetime(series_games['date'], format='mixed', errors='coerce')
if pd.notna(sel_date):
    series_games = series_games[(series_games['_d'] >= sel_date - pd.Timedelta(days=2)) &
                                 (series_games['_d'] <= sel_date + pd.Timedelta(days=2))]
# One row per gameId in the window, sorted by (date, gameId) so
# doubleheaders on the same date get distinct positions.
series_ids = (series_games.drop_duplicates('gameId')
                          .sort_values(['_d', 'gameId'])['gameId'].tolist())
if len(series_ids) >= 2:
    game_type = 'Weekend'
    try:
        game_num = max(1, min(3, series_ids.index(sel_gid) + 1))
    except ValueError:
        game_num = 1
    detected_label = f'Weekend — Game {game_num} of {len(series_ids)}'
else:
    game_type = 'Midweek'
    game_num = 0
    detected_label = 'Midweek (single game)'

profiles = build_team_profiles(sel_sport, sel_division)
home_p_player = adjusted_team_pct(home_key, profiles, game_type, game_num, home_p) if home_p else home_p
away_p_player = adjusted_team_pct(away_key, profiles, game_type, game_num, away_p) if away_p else away_p

home_p_adj = _shared_blend_with_static(home_p_player, home_p)
away_p_adj = _shared_blend_with_static(away_p_player, away_p)

# Sidebar transparency: what we detected + resolved values
st.sidebar.markdown('### Talent model')
st.sidebar.info(f'Detected: **{detected_label}**')
with st.sidebar.expander('Talent breakdown', expanded=False):
    st.caption(f'Team-rank blend: {TEAM_RANK_BLEND:.2f}')
    st.caption(f'**{home}** (home)')
    st.caption(f'  team-rank pct: {home_p:.3f}' if home_p else '  no team pct')
    st.caption(f'  player-adj pct: {home_p_player:.3f}' if home_p_player else '  fallback')
    st.caption(f'  blended: {home_p_adj:.3f}' if home_p_adj else '  fallback')
    st.caption(f'**{away}** (away)')
    st.caption(f'  team-rank pct: {away_p:.3f}' if away_p else '  no team pct')
    st.caption(f'  player-adj pct: {away_p_player:.3f}' if away_p_player else '  fallback')
    st.caption(f'  blended: {away_p_adj:.3f}' if away_p_adj else '  fallback')

# Pre-game WP — use the blended pct
pg_home = pre_game_wp(home_p_adj, away_p_adj) if home_p_adj and away_p_adj else 0.5

# Header cards (Thrill Score added after WP curve is computed below)
_hdr_cols = st.columns(5)
_hdr_cols[0].metric('Pre-game WP (home)', f"{pg_home*100:.1f}%")
_hdr_cols[1].metric('Pre-game WP (away)', f"{(1-pg_home)*100:.1f}%")
_hdr_cols[2].metric('Final score', f"{away} {int(game['awayScore'].max())} — {int(game['homeScore'].max())} {home}")
_hdr_cols[3].metric('Plays', f"{len(game)}")

# Compute WP curve per play
total_plays = len(game)
wp_curve = [pg_home]  # anchor at index 0 (pre-game)
state_wp_curve = [None]
for i, row in game.iterrows():
    key = state_key(row)
    info = lookup.get(key)
    w = min(1.0, (i + 1) / total_plays)
    if info is None or info[1] < 10:
        wp = pg_home
        sw = None
    else:
        sw = info[0]
        wp = w * sw + (1 - w) * pg_home
    wp_curve.append(max(CLAMP_MIN, min(CLAMP_MAX, wp)))
    state_wp_curve.append(sw)

# Lock the final WP to the actual game result. After the last out is
# recorded, the outcome is determined — not a probability anymore.
final_home_score = int(game['homeScore'].max())
final_away_score = int(game['awayScore'].max())
if final_home_score != final_away_score:
    home_won = final_home_score > final_away_score
    wp_curve[-1] = 1.0 if home_won else 0.0

# ── Thrill Score ─────────────────────────────────────────────────────────────
# Measures how exciting the game was based on WP curve shape + team quality.
# Components:
#   Balance  (30%) — did both teams hold the lead at some point? (0-1)
#   Closeness (30%) — how tight was the WP to 50% throughout? (0-1)
#   Lead changes (15%) — how many times did the WP cross 50%? (0-1 normalized)
#   Team quality (25%) — mean of both teams' rank percentile (0-1, higher = elite matchup)
wp_arr = np.array(wp_curve[:-1])  # exclude the locked final 0/100 endpoint
deviations = np.abs(wp_arr - 0.5)
home_lead_area = float(np.sum(np.maximum(wp_arr - 0.5, 0)))
away_lead_area = float(np.sum(np.maximum(0.5 - wp_arr, 0)))
max_area = max(home_lead_area, away_lead_area)
min_area = min(home_lead_area, away_lead_area)
balance = (min_area / max_area) if max_area > 0 else 0.0
closeness = float(1.0 - np.mean(deviations) / 0.5)
closeness = max(0.0, closeness)
crosses = 0
for j in range(1, len(wp_arr)):
    if (wp_arr[j-1] < 0.5 and wp_arr[j] >= 0.5) or (wp_arr[j-1] >= 0.5 and wp_arr[j] < 0.5):
        crosses += 1
lead_change_score = min(1.0, crosses / 6.0)
# Team quality: average rank_pct of both teams (0-1, higher = better matchup)
home_rpct = home_p if home_p else 0.5
away_rpct = away_p if away_p else 0.5
team_quality = (home_rpct + away_rpct) / 2.0
thrill_score = (0.30 * balance + 0.30 * closeness + 0.15 * lead_change_score + 0.25 * team_quality) * 100
_hdr_cols[4].metric('Thrill Score', f"{thrill_score:.0f}",
                     help='0-100 excitement rating. Balance (both teams led) + Closeness (WP near 50%) + Lead changes.')

# Build hover data
hover_texts = []
hover_texts.append(f"<b>Pre-game</b><br>{home} starting WP: <b>{pg_home*100:.1f}%</b>")
for i, row in game.iterrows():
    wp_before = wp_curve[i]
    wp_after = wp_curve[i + 1]
    delta = (wp_after - wp_before) * 100
    arrow = '▲' if delta > 0 else ('▼' if delta < 0 else '—')
    delta_str = f"{arrow} {abs(delta):.1f}%"
    half = 'Bot' if row['half_true'] == 'bottom' else 'Top'
    score = f"{int(row['awayScore'])}-{int(row['homeScore'])}"
    player = row.get('player', '') or ''
    desc = str(row.get('playDescription', '') or '')[:90]
    hover_texts.append(
        f"<b>{half} {int(row['inning'])}</b>  {score}  ·  {int(row['outs'])} out<br>"
        f"<i>{player}</i>: {desc}<br>"
        f"<b>{home} WP: {wp_after*100:.1f}%</b>  ({delta_str})"
    )

# Build inning-based x-axis positions
# The x coord is play index (0..N), but we label ticks at inning boundaries.
inning_labels = []
inning_positions = []
last_inning = None
for i, row in game.iterrows():
    ih = (int(row['inning']), row['half_true'])
    if ih != last_inning:
        inning_positions.append(i + 1)  # +1 because pre-game is at 0
        half_abbr = 'T' if row['half_true'] == 'top' else 'B'
        inning_labels.append(f"{half_abbr}{int(row['inning'])}")
        last_inning = ih

x_indices = list(range(len(wp_curve)))
home_wps = [w * 100 for w in wp_curve]

# Team-specific colors from logos
home_col = team_color(home, fallback=HOME_COLOR)
away_col = team_color(away, fallback=AWAY_COLOR)

# Plotly figure
fig = go.Figure()

# Baseline at 50 (used as a fill anchor — needs mode='lines' or
# tonexty doesn't register the previous trace).
INVISIBLE = 'rgba(0,0,0,0)'
fig.add_trace(go.Scatter(
    x=x_indices, y=[50] * len(x_indices), mode='lines',
    line=dict(color=INVISIBLE, width=0),
    showlegend=False, hoverinfo='skip',
))

# HOME ahead region: fill between 50 and curve where curve > 50.
# Everywhere else, y equals 50 (zero-height fill).
home_ahead_y = [w if w > 50 else 50 for w in home_wps]
fig.add_trace(go.Scatter(
    x=x_indices, y=home_ahead_y, fill='tonexty',
    fillcolor=rgba_from_hex(home_col, 0.55),
    mode='lines', line=dict(color=INVISIBLE, width=0),
    name=f'{home}', showlegend=True, hoverinfo='skip',
))

# Reset baseline for away fill
fig.add_trace(go.Scatter(
    x=x_indices, y=[50] * len(x_indices), mode='lines',
    line=dict(color=INVISIBLE, width=0),
    showlegend=False, hoverinfo='skip',
))

# AWAY ahead region: fill between curve and 50 where curve < 50.
away_ahead_y = [w if w < 50 else 50 for w in home_wps]
fig.add_trace(go.Scatter(
    x=x_indices, y=away_ahead_y, fill='tonexty',
    fillcolor=rgba_from_hex(away_col, 0.55),
    mode='lines', line=dict(color=INVISIBLE, width=0),
    name=f'{away}', showlegend=True, hoverinfo='skip',
))

# 50% reference line on top of fills
fig.add_hline(y=50, line_dash='dash', line_color='#888', line_width=1, opacity=0.6)

# Main WP line with rich hover
fig.add_trace(go.Scatter(
    x=x_indices, y=home_wps, mode='lines+markers',
    line=dict(color=TEXT_COLOR, width=2.4),
    marker=dict(size=4, color=TEXT_COLOR),
    hovertext=hover_texts, hoverinfo='text',
    name='Win Probability', showlegend=False,
))

# Inning divider lines
for pos in inning_positions[1:]:
    fig.add_vline(x=pos, line_dash='dot', line_color='#bbb', line_width=0.6, opacity=0.4)

# Layout with 64Analytics branding
annotations = [
    dict(x=0.5, y=1.08, xref='paper', yref='paper', showarrow=False,
         text=f'<b>{away}</b> @ <b>{home}</b>  ·  {game["date"].iloc[0]}',
         font=dict(size=18, color=TEXT_COLOR, family='Arial Black')),
    dict(x=0.5, y=1.03, xref='paper', yref='paper', showarrow=False,
         text=f'Pre-game: {pg_home*100:.1f}%  →  Final: {home_wps[-1]:.1f}%',
         font=dict(size=11, color=TEXT_MUTED)),
]

# Team logo wallpaper: one large centered watermark per team half.
# Home logo in the upper zone (50-100%), away logo in the lower zone (0-50%).
# Fixed position — no more tile-grid math, no floating/gap issues.
images = []
home_logo_b64 = faded_logo_b64(home, size=300, alpha=0.18)
away_logo_b64 = faded_logo_b64(away, size=300, alpha=0.18)

if home_logo_b64:
    images.append(dict(
        source=f'data:image/png;base64,{home_logo_b64}',
        xref='paper', yref='paper',
        x=0.5, y=0.75,
        sizex=0.30, sizey=0.40,
        xanchor='center', yanchor='middle', layer='below',
    ))
if away_logo_b64:
    images.append(dict(
        source=f'data:image/png;base64,{away_logo_b64}',
        xref='paper', yref='paper',
        x=0.5, y=0.25,
        sizex=0.30, sizey=0.40,
        xanchor='center', yanchor='middle', layer='below',
    ))

# 64Analytics brand logo bottom-right
brand_b64 = brand_logo_b64()
if brand_b64:
    images.append(dict(
        source=f'data:image/png;base64,{brand_b64}',
        xref='paper', yref='paper',
        x=0.99, y=-0.16, sizex=0.16, sizey=0.10,
        xanchor='right', yanchor='bottom', opacity=0.85, layer='above',
    ))

fig.update_layout(
    plot_bgcolor=BG_COLOR,
    paper_bgcolor=BG_COLOR,
    height=560,
    margin=dict(t=90, b=80, l=70, r=30),
    xaxis=dict(
        title=dict(text='Inning', font=dict(size=14, color=TEXT_COLOR)),
        tickmode='array', tickvals=inning_positions, ticktext=inning_labels,
        tickangle=0, color=TEXT_COLOR,
        tickfont=dict(size=13, color=TEXT_COLOR, family='Arial Black'),
        gridcolor=GRID_COLOR, gridwidth=0.5,
        showgrid=False, zeroline=False,
    ),
    yaxis=dict(
        title=dict(text=f'{home} Win Probability', font=dict(size=13, color=TEXT_COLOR)),
        range=[0, 100], tickvals=[0, 25, 50, 75, 100],
        ticktext=['0%', '25%', '50%', '75%', '100%'],
        color=TEXT_MUTED, gridcolor=GRID_COLOR, gridwidth=0.5, zeroline=False,
    ),
    annotations=annotations,
    images=images,
    legend=dict(orientation='h', y=-0.12, x=0, font=dict(color=TEXT_COLOR, size=11),
                bgcolor='rgba(0,0,0,0)'),
    hoverlabel=dict(bgcolor=BG_COLOR, bordercolor=TEXT_COLOR,
                    font=dict(color=TEXT_COLOR, size=12)),
)

# ── Play highlighter: select a play to pin an annotation on the chart ────────
# Build play options (biggest WP swings first, then full list)
play_options = ['(none)']
play_data = []  # parallel list of (play_index, label, wp_before, wp_after, hover)
for idx in range(len(game)):
    wp_b = wp_curve[idx]
    wp_a = wp_curve[idx + 1]
    delta = (wp_a - wp_b) * 100
    row = game.iloc[idx]
    half = 'Bot' if row['half_true'] == 'bottom' else 'Top'
    player = str(row.get('player', '') or '')[:20]
    desc = str(row.get('playDescription', '') or '')[:60]
    label = f"{half} {int(row['inning'])}, {int(row['outs'])} out | {player}: {desc} ({delta:+.1f}%)"
    play_data.append((idx + 1, label, wp_b, wp_a, hover_texts[idx + 1]))

# Sort by absolute delta for the "Top swings" section
top_swings = sorted(play_data, key=lambda x: abs(x[3] - x[2]), reverse=True)[:8]
play_options += [f"★ {p[1]}" for p in top_swings]
play_options += ['───── All plays ─────']
play_options += [p[1] for p in play_data]

st.markdown('**Pin a play** — select to add a visible callout on the chart (stays in PNG export):')
pc1, pc2, pc3 = st.columns([3, 1, 1])
selected_play = pc1.selectbox('Highlight play', play_options, index=0, key='highlight_play',
                               label_visibility='collapsed')
ann_x_offset = pc2.slider('H offset', -200, 200, 0, step=10, key='ann_x',
                           help='Horizontal offset of the callout box (negative = left, positive = right)')
ann_y_offset = pc3.slider('V offset', -200, 200, -70, step=10, key='ann_y',
                           help='Vertical offset of the callout box (negative = up, positive = down)')

# If a play is selected, add annotation to the figure
if selected_play and selected_play not in ('(none)', '───── All plays ─────'):
    clean_label = selected_play.lstrip('★ ').strip()
    match = next((p for p in play_data if p[1] == clean_label), None)
    if match:
        pidx, plabel, wp_b, wp_a, hover_html = match
        # Build a compact annotation that fits in a fixed-width box.
        # Same styling as the hover tooltip but truncated description
        # so text doesn't overflow.
        row = game.iloc[pidx - 1]
        half = 'Bot' if row['half_true'] == 'bottom' else 'Top'
        score = f"{int(row['awayScore'])}-{int(row['homeScore'])}"
        outs = int(row['outs'])
        player = str(row.get('player', '') or '')
        desc = str(row.get('playDescription', '') or '')
        delta = (wp_a - wp_b) * 100
        arrow = '▲' if delta > 0 else ('▼' if delta < 0 else '—')
        delta_str = f"{arrow} {abs(delta):.1f}%"
        short_home = home.split(' ')[0] if len(home) > 15 else home
        # Word-wrap the description by inserting <br> at ~50 char intervals
        # Wrap ALL lines to max 28 chars (conservative for variable-width font).
        # Plotly annotation width is set to match so box and text agree.
        WRAP = 28
        def _wrap_line(text):
            words = text.split(' ')
            lines, cur = [], ''
            for w in words:
                if cur and len(cur) + 1 + len(w) > WRAP:
                    lines.append(cur)
                    cur = w
                else:
                    cur = f'{cur} {w}'.strip()
            if cur:
                lines.append(cur)
            return '<br>'.join(lines)
        line1 = f"<b>{half} {int(row['inning'])}</b>  {score}  ·  {outs} out"
        line2 = _wrap_line(f'{player}: {desc}')
        line3 = f"<b>{short_home} WP: {wp_a*100:.1f}%</b>  ({delta_str})"
        ann_text = f"{line1}<br><i>{line2}</i><br>{line3}"
        fig.add_annotation(
            x=pidx, y=wp_a * 100,
            text=ann_text,
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
            arrowcolor=TEXT_COLOR,
            ax=ann_x_offset, ay=ann_y_offset,
            bordercolor=TEXT_COLOR, borderwidth=1, borderpad=6,
            bgcolor=BG_COLOR, opacity=0.95,
            font=dict(size=10, color=TEXT_COLOR, family='Courier New'),
            align='left',
        )
        fig.add_trace(go.Scatter(
            x=[pidx], y=[wp_a * 100], mode='markers',
            marker=dict(size=10, color=TEXT_COLOR, symbol='circle'),
            showlegend=False, hoverinfo='skip',
        ))

st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {
    'format': 'png', 'filename': f'WP_{home}_vs_{away}', 'scale': 2}})

# Play-by-play log expander
with st.expander('Play-by-play log with WP'):
    gdisp = game.copy()
    gdisp['Home WP'] = [f'{w*100:.1f}%' for w in wp_curve[1:]]
    gdisp['WP Δ'] = [f'{(wp_curve[i+1]-wp_curve[i])*100:+.1f}%' for i in range(len(game))]
    gdisp['half'] = gdisp['half_true']
    st.dataframe(gdisp[['inning', 'half', 'outs', 'awayScore', 'homeScore',
                         'player', 'playDescription', 'Home WP', 'WP Δ']],
                 use_container_width=True, hide_index=True)

# ── Thrill Score Leaderboard ─────────────────────────────────────────────────
st.markdown('---')
st.markdown('### Thrill Score Leaderboard')

thrill_csv = DATA_DIR / 'thrill_scores.csv'
if thrill_csv.exists():
    @st.cache_data
    def load_thrill_scores():
        return pd.read_csv(thrill_csv, low_memory=False)

    tdf = load_thrill_scores()
    # Filter to selected sport + division
    tdf_filt = tdf[(tdf['sport'] == sel_sport) & (tdf['division'] == sel_division)].copy()

    if tdf_filt.empty:
        st.info(f'No thrill scores available for {sel_sport} {sel_division}.')
    else:
        tdf_filt['date'] = pd.to_datetime(tdf_filt['date'], errors='coerce')
        # Date range filter
        tc1, tc2, tc3 = st.columns([1, 1, 2])
        min_date = tdf_filt['date'].min()
        max_date = tdf_filt['date'].max()
        if pd.notna(min_date) and pd.notna(max_date):
            date_start = tc1.date_input('From', value=max_date - pd.Timedelta(days=14),
                                         min_value=min_date, max_value=max_date, key='thrill_from')
            date_end = tc2.date_input('To', value=max_date,
                                       min_value=min_date, max_value=max_date, key='thrill_to')
            tdf_filt = tdf_filt[(tdf_filt['date'] >= pd.Timestamp(date_start)) &
                                 (tdf_filt['date'] <= pd.Timestamp(date_end))]

        search = tc3.text_input('Search team', '', placeholder='Filter by team name...', key='thrill_search')
        if search:
            q = search.lower()
            tdf_filt = tdf_filt[tdf_filt['home'].str.lower().str.contains(q, na=False) |
                                 tdf_filt['away'].str.lower().str.contains(q, na=False)]

        tdf_filt = tdf_filt.sort_values('thrill_score', ascending=False).reset_index(drop=True)
        tdf_filt.index = tdf_filt.index + 1
        tdf_filt.index.name = '#'

        display_cols = ['date', 'away', 'home', 'final_away', 'final_home',
                        'thrill_score', 'balance', 'closeness', 'lead_changes', 'team_quality', 'plays']
        display_cols = [c for c in display_cols if c in tdf_filt.columns]
        tdf_filt['date'] = tdf_filt['date'].dt.strftime('%m/%d')
        tdf_filt = tdf_filt.rename(columns={
            'final_away': 'Away', 'final_home': 'Home',
            'thrill_score': 'Thrill', 'lead_changes': 'Leads',
            'team_quality': 'Quality',
        })
        display_cols = ['date', 'away', 'home', 'Away', 'Home', 'Thrill',
                        'balance', 'closeness', 'Leads', 'Quality', 'plays']
        display_cols = [c for c in display_cols if c in tdf_filt.columns]
        st.dataframe(tdf_filt[display_cols], use_container_width=True, height=500)
        st.caption(f'{len(tdf_filt)} games shown')
else:
    st.info('Thrill scores not yet computed. Run `python scripts/compute_thrill_scores.py` to generate.')
