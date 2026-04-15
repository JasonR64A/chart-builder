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
PBP_FILE = PBP_DIR / 'baseball_play_by_play_D1.csv'
PBP_FILE_GZ = PBP_DIR / 'baseball_play_by_play_D1.csv.gz'
LOOKUP_FILE = _APP_DIR / 'pbp_data' / 'wp_state_lookup_bb_d1.pkl'
LOGO_DIR = _APP_DIR / 'team_logos_512'
BRAND_LOGO = _APP_DIR / 'assets' / 'brand_logo_wide.png'

HOME_FIELD_ADVANTAGE = 0.04
# In-game clamps — allow real blowouts to read near-certain
CLAMP_MIN = 0.01
CLAMP_MAX = 0.99
# Pre-game clamps — tighter so talent gap never implies a near-lock.
# A 90% ceiling keeps the best-vs-worst D1 matchups (Arkansas vs UAPB etc.)
# from pinning at 99%, which felt unrealistic. Tune from here.
PREGAME_CLAMP_MIN = 0.10
PREGAME_CLAMP_MAX = 0.90

# 64Analytics brand colors (matches other chart pages)
BG_COLOR = '#FAF8F2'       # warm off-white
TEXT_COLOR = '#2D2926'
TEXT_MUTED = '#4A4540'
GRID_COLOR = '#D8D2C4'
HOME_COLOR = '#2E5C8A'     # navy blue
AWAY_COLOR = '#C41230'     # cardinal red


# ── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_pbp():
    cols = ['gameId', 'date', 'awayTeam', 'homeTeam', 'inning', 'halfInning',
            'battingTeam',
            'outs', 'runner1B', 'runner2B', 'runner3B',
            'awayScore', 'homeScore', 'player', 'playDescription']
    if PBP_FILE.exists():
        df = pd.read_csv(PBP_FILE, low_memory=False, usecols=cols)
    elif PBP_FILE_GZ.exists():
        df = pd.read_csv(PBP_FILE_GZ, low_memory=False, usecols=cols, compression='gzip')
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
def load_teams():
    t = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False)
    t['id'] = pd.to_numeric(t['id'], errors='coerce').fillna(0).astype(int)
    tr = pd.read_csv(DATA_DIR / 'team_rank.csv', low_memory=False)
    tr = tr[tr['year'] == 2026].copy()
    tr['rank'] = pd.to_numeric(tr['integer_64_rank_total'], errors='coerce')
    confs = pd.read_csv(DATA_DIR / 'conferences.csv', low_memory=False)
    di_ids = set(confs[confs['division'] == 'D-I']['id'])
    bb_d1 = t[(t['sport'] == 'Baseball') & (t['conference_id'].isin(di_ids))]
    tr_d1 = tr[tr['team_id'].isin(bb_d1['id'])]
    N = len(tr_d1)
    tr_d1 = tr_d1.dropna(subset=['rank']).copy()
    tr_d1['rank_pct'] = 1 - (tr_d1['rank'] - 1) / (N - 1)
    name_to_pct = dict(zip(bb_d1.set_index('id').loc[tr_d1['team_id'].values, 'name'],
                            tr_d1['rank_pct']))
    name_to_rank = dict(zip(bb_d1.set_index('id').loc[tr_d1['team_id'].values, 'name'],
                             tr_d1['rank'].astype(int)))
    return name_to_pct, name_to_rank


@st.cache_data
def brand_logo_b64():
    if BRAND_LOGO.exists():
        with open(BRAND_LOGO, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    return None


@st.cache_data
def team_logo_map():
    """Build team name → 64A team id for locating logo files."""
    t = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False)
    t['id'] = pd.to_numeric(t['id'], errors='coerce').fillna(0).astype(int)
    bb = t[t['sport'] == 'Baseball'][['name', 'id']].drop_duplicates('name')
    m = dict(zip(bb['name'], bb['id']))
    return m


def team_logo_path(team_full_name):
    """Find the logo file for a team (accounting for trailing mascot names)."""
    logo_map = team_logo_map()
    for cand in [team_full_name, short_team(team_full_name)]:
        if cand in logo_map:
            for ext in ('png', 'webp'):
                p = LOGO_DIR / f'{logo_map[cand]}.{ext}'
                if p.exists():
                    return p
    # Fuzzy: try substring match
    for name, tid in logo_map.items():
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


def log5(wa, wb):
    denom = wa + wb - 2 * wa * wb
    return (wa - wa * wb) / denom if denom else 0.5


def pre_game_wp(home_pct, away_pct):
    if pd.isna(home_pct) or pd.isna(away_pct):
        return 0.5
    base = log5(home_pct, away_pct)
    return max(PREGAME_CLAMP_MIN, min(PREGAME_CLAMP_MAX, base + HOME_FIELD_ADVANTAGE))


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
    for k in team_pct.keys():
        if name.startswith(k) or k.startswith(short_team(name)):
            return k, team_pct[k], team_rank.get(k)
    return name, None, None


# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='Win Probability', layout='wide')
st.title('Win Probability — Pace Chart')
st.caption('Play-by-play WP curve. Starting point = log5 of 64 integer ranks ± 4% home-field bump. '
           'In-game WP = empirical state lookup blended with pre-game anchor (weight shifts to state as game progresses).')

pbp = load_pbp()
if pbp is None:
    st.warning('Play-by-play data not available on Render. Redeploy should include the gzipped PBP file.')
    st.stop()

lookup = load_lookup()
team_pct, team_rank = load_teams()

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
game_list['label'] = game_list.apply(
    lambda r: f"{r['date']}  {r['awayTeam']} @ {r['homeTeam']}", axis=1)
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

home_key, home_p, home_r = find_team(home, team_pct, team_rank)
away_key, away_p, away_r = find_team(away, team_pct, team_rank)

# Pre-game WP
pg_home = pre_game_wp(home_p, away_p) if home_p and away_p else 0.5

# Header cards
c1, c2, c3, c4 = st.columns(4)
c1.metric('Pre-game WP (home)', f"{pg_home*100:.1f}%")
c2.metric('Pre-game WP (away)', f"{(1-pg_home)*100:.1f}%")
c3.metric('Final score', f"{away} {int(game['awayScore'].max())} — {int(game['homeScore'].max())} {home}")
c4.metric('Plays', f"{len(game)}")

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
    name=f'{home} ahead', showlegend=True, hoverinfo='skip',
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
    name=f'{away} ahead', showlegend=True, hoverinfo='skip',
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

# Brand logo (bottom-right) + team-logo wallpaper tile pattern
# (whichever team is leading at a tile's x-position gets that tile)
images = []

home_logo_b64 = faded_logo_b64(home, size=200, alpha=0.16)
away_logo_b64 = faded_logo_b64(away, size=200, alpha=0.16)

# Tile grid CONFINED to the lead band (between 50% and the curve).
# Two interleaved column sets so the pattern doesn't look uniform:
#   - "main"   columns every X_STEP_MAIN plays, rows starting at Y_BASE
#   - "offset" columns shifted half-step right, rows shifted half-step down
# Half as many rows as before (Y_STEP doubled).
tile_size_x = 5
tile_size_y = 7
X_STEP_MAIN = 12
Y_STEP = 16
Y_BASE = 6
n_x = max(1, len(x_indices))

def add_wallpaper_column(cx, y_start):
    """Drop tiles down a single x-column, but only inside the lead band."""
    if cx < 0 or cx >= n_x:
        return
    w = home_wps[cx]
    if w == 50:
        return
    home_leading = w > 50
    logo_b64 = home_logo_b64 if home_leading else away_logo_b64
    if not logo_b64:
        return
    band_low, band_high = (50, w) if home_leading else (w, 50)
    for cy in range(y_start, 100, Y_STEP):
        # Tile must fit entirely inside the lead band so it never bleeds
        # into the unfilled half of the chart.
        if (cy - tile_size_y / 2) < band_low or (cy + tile_size_y / 2) > band_high:
            continue
        images.append(dict(
            source=f'data:image/png;base64,{logo_b64}',
            xref='x', yref='y',
            x=cx, y=cy,
            sizex=tile_size_x, sizey=tile_size_y,
            xanchor='center', yanchor='middle', layer='below',
        ))

# Main columns
for cx in range(0, n_x, X_STEP_MAIN):
    add_wallpaper_column(cx, y_start=Y_BASE)
# In-between offset columns (shifted half-step horizontally + vertically)
for cx in range(X_STEP_MAIN // 2, n_x, X_STEP_MAIN):
    add_wallpaper_column(cx, y_start=Y_BASE + Y_STEP // 2)

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

st.plotly_chart(fig, use_container_width=True)

# Play-by-play log expander
with st.expander('Play-by-play log with WP'):
    gdisp = game.copy()
    gdisp['Home WP'] = [f'{w*100:.1f}%' for w in wp_curve[1:]]
    gdisp['WP Δ'] = [f'{(wp_curve[i+1]-wp_curve[i])*100:+.1f}%' for i in range(len(game))]
    gdisp['half'] = gdisp['half_true']
    st.dataframe(gdisp[['inning', 'half', 'outs', 'awayScore', 'homeScore',
                         'player', 'playDescription', 'Home WP', 'WP Δ']],
                 use_container_width=True, hide_index=True)
