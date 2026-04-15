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
def load_pbp():
    cols = ['gameId', 'date', 'awayTeam', 'homeTeam', 'inning', 'halfInning',
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
def rotated_logo_b64(team_full_name, angle=-45, size=200, alpha=0.18):
    """Load a team's logo, rotate, reduce opacity, return base64 PNG."""
    p = team_logo_path(team_full_name)
    if not p:
        return None
    try:
        img = Image.open(p).convert('RGBA')
        img.thumbnail((size, size), Image.LANCZOS)
        img = img.rotate(angle, expand=True, resample=Image.BICUBIC)
        # Apply alpha
        alpha_layer = img.split()[3]
        alpha_layer = alpha_layer.point(lambda a: int(a * alpha))
        img.putalpha(alpha_layer)
        buf = BytesIO()
        img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def find_biggest_segment(home_wps, want_home_ahead):
    """Return (start_idx, end_idx) of the longest stretch where:
       - want_home_ahead=True  → WP > 50
       - want_home_ahead=False → WP < 50
    Returns None if no such stretch exists."""
    best = None
    cur_start = None
    for i, wp in enumerate(home_wps):
        ahead = wp > 50 if want_home_ahead else wp < 50
        if ahead:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                if best is None or (i - cur_start) > (best[1] - best[0]):
                    best = (cur_start, i - 1)
                cur_start = None
    if cur_start is not None:
        end = len(home_wps) - 1
        if best is None or (end - cur_start) > (best[1] - best[0]):
            best = (cur_start, end)
    return best


def log5(wa, wb):
    denom = wa + wb - 2 * wa * wb
    return (wa - wa * wb) / denom if denom else 0.5


def pre_game_wp(home_pct, away_pct):
    if pd.isna(home_pct) or pd.isna(away_pct):
        return 0.5
    base = log5(home_pct, away_pct)
    return max(CLAMP_MIN, min(CLAMP_MAX, base + HOME_FIELD_ADVANTAGE))


def state_key(row):
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

game = pbp[pbp['gameId'] == sel_gid].copy().reset_index(drop=True)
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

# Build hover data
hover_texts = []
hover_texts.append(f"<b>Pre-game</b><br>{home} starting WP: <b>{pg_home*100:.1f}%</b>")
for i, row in game.iterrows():
    wp_before = wp_curve[i]
    wp_after = wp_curve[i + 1]
    delta = (wp_after - wp_before) * 100
    arrow = '▲' if delta > 0 else ('▼' if delta < 0 else '—')
    delta_str = f"{arrow} {abs(delta):.1f}%"
    half = 'Bot' if row['halfInning'] == 'bottom' else 'Top'
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
    ih = (int(row['inning']), row['halfInning'])
    if ih != last_inning:
        inning_positions.append(i + 1)  # +1 because pre-game is at 0
        half_abbr = 'T' if row['halfInning'] == 'top' else 'B'
        inning_labels.append(f"{half_abbr}{int(row['inning'])}")
        last_inning = ih

x_indices = list(range(len(wp_curve)))
home_wps = [w * 100 for w in wp_curve]

# Team-specific colors from logos
home_col = team_color(home, fallback=HOME_COLOR)
away_col = team_color(away, fallback=AWAY_COLOR)

# Plotly figure
fig = go.Figure()

# 50% reference
fig.add_hline(y=50, line_dash='dash', line_color='#888', line_width=1, opacity=0.5)

# Shaded area uses each team's own color (home above 50%, away below)
above = [max(w, 50) for w in home_wps]
below = [min(w, 50) for w in home_wps]

fig.add_trace(go.Scatter(
    x=x_indices, y=above, fill='tonexty',
    fillcolor=rgba_from_hex(home_col, 0.22),
    mode='none', name=f'{home} ahead', showlegend=True, hoverinfo='skip',
))
fig.add_trace(go.Scatter(
    x=x_indices, y=[50] * len(x_indices), mode='none',
    showlegend=False, hoverinfo='skip',
))
fig.add_trace(go.Scatter(
    x=x_indices, y=below, fill='tonexty',
    fillcolor=rgba_from_hex(away_col, 0.22),
    mode='none', name=f'{away} ahead', showlegend=True, hoverinfo='skip',
))

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

# Brand logo (bottom-right) + team logos rotated 45° in each team's shaded region
images = []

def add_team_logo_in_segment(team_name, want_home_ahead):
    """Place the team's rotated logo in the middle of their largest lead segment."""
    seg = find_biggest_segment(home_wps, want_home_ahead=want_home_ahead)
    if seg is None:
        return
    start, end = seg
    mid_x = (start + end) / 2
    # Compute a representative midpoint y value within the segment
    seg_wps = home_wps[start:end + 1]
    if want_home_ahead:
        # WPs above 50 — logo centered between 50 and the segment's average WP
        avg_wp = sum(seg_wps) / len(seg_wps)
        mid_y = (50 + avg_wp) / 2
    else:
        avg_wp = sum(seg_wps) / len(seg_wps)
        mid_y = (50 + avg_wp) / 2  # still centered in the shaded band
    logo_b64_rot = rotated_logo_b64(team_name, angle=-45, size=240, alpha=0.22)
    if not logo_b64_rot:
        return
    # Size logo proportional to segment width (clamped)
    seg_len = end - start + 1
    sizex = max(6, min(18, seg_len / 2.5))
    sizey = 18  # WP units
    images.append(dict(
        source=f'data:image/png;base64,{logo_b64_rot}',
        xref='x', yref='y',
        x=mid_x, y=mid_y, sizex=sizex, sizey=sizey,
        xanchor='center', yanchor='middle', layer='below',
    ))

add_team_logo_in_segment(home, want_home_ahead=True)
add_team_logo_in_segment(away, want_home_ahead=False)

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
        title=dict(text='Inning', font=dict(size=13, color=TEXT_COLOR)),
        tickmode='array', tickvals=inning_positions, ticktext=inning_labels,
        tickangle=0, color=TEXT_MUTED, gridcolor=GRID_COLOR, gridwidth=0.5,
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
    st.dataframe(gdisp[['inning', 'halfInning', 'outs', 'awayScore', 'homeScore',
                         'player', 'playDescription', 'Home WP', 'WP Δ']],
                 use_container_width=True, hide_index=True)
