"""
64 Analytics — Stock Up / Stock Down
Weekly mover report: compare old vs. current rankings.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw
from pathlib import Path
from io import BytesIO
import os
import matplotlib.font_manager as fm

# ── Path setup (works locally and on Streamlit Cloud) ─────────────────────────
_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _APP_DIR / 'data'
OLD_DIR  = DATA_DIR / 'Graphcis'
LOGO_DIR = _APP_DIR / 'team_logos_512'
BRAND_LOGO_DARK  = _APP_DIR / 'assets' / 'brand_logo_dark.png'
BRAND_LOGO_LIGHT = _APP_DIR / 'assets' / 'brand_logo_light.png'

RED      = '#C41230'
RED_DK   = '#8B1A2A'
RED_LT   = '#E8455E'

def _has_font(name):
    return any(name.lower() in f.name.lower() for f in fm.fontManager.ttflist)

TITLE_FONT    = 'Franklin Gothic Heavy'  if _has_font('Franklin Gothic') else 'DejaVu Sans'
SUBTITLE_FONT = 'Franklin Gothic Medium' if _has_font('Franklin Gothic') else 'DejaVu Sans'
BODY_FONT     = 'Calibri' if _has_font('Calibri') else 'DejaVu Sans'
AXIS_FONT     = 'Calibri' if _has_font('Calibri') else 'DejaVu Sans'

THEMES = {
    'Dark': {
        'bg': '#1a1a1a', 'plot_bg': '#222222',
        'text': '#FFFFFF', 'text_sub': '#C8C8C8',
        'text_md': '#888888', 'text_dk': '#555555',
        'grid': '#2e2e2e', 'spine': '#3a3a3a',
        'up': RED, 'up_light': RED_LT,
        'down': '#4A7FB5', 'down_dark': '#1E3A5F',
        'start_marker': '#888888',
        'brand_logo': BRAND_LOGO_DARK,
        'streamlit_bg': '#1a1a1a',
    },
    'Light': {
        'bg': '#FFFFFF', 'plot_bg': '#F5F5F5',
        'text': '#1a1a1a', 'text_sub': '#333333',
        'text_md': '#666666', 'text_dk': '#999999',
        'grid': '#E0E0E0', 'spine': '#CCCCCC',
        'up': RED_DK, 'up_light': RED,
        'down': '#1E3A5F', 'down_dark': '#4A7FB5',
        'start_marker': '#999999',
        'brand_logo': BRAND_LOGO_LIGHT,
        'streamlit_bg': '#FFFFFF',
    },
}

# ── Rank column options ───────────────────────────────────────────────────────
TEAM_RANK_COLS = {
    '64 Rank (Percentile)': 'sixty_four_rank_weighted_run_efficiency',
    '64 Rank (Integer)': 'integer_64_rank_total',
    'Run Created Rank (Percentile)': 'rank_weighted_run_created_per_35_at_bat',
    'Run Allowed Rank (Percentile)': 'rank_weighted_run_allowed_per_35_at_bat',
    'Run Created Rank (Integer)': 'integer_weighted_run_created_per_35_at_bat',
    'Run Allowed Rank (Integer)': 'integer_weighted_run_allowed_per_35_at_bat',
}

PLAYER_RANK_COLS = {
    '64 Hitting Rank (#)': 'integer_rank_weighted_run_created_efficiency',
    '64 Pitching Rank (#)': 'integer_rank_weighted_run_allowed_efficiency',
    '64 Hitting Rank (Percentile)': 'percentile_rank_weighted_run_created_efficiency',
    '64 Pitching Rank (Percentile)': 'percentile_rank_weighted_run_allowed_efficiency',
    'Runs Created AB (Percentile)': 'percentile_rank_runs_created_ab',
    'Runs Allowed AB (Percentile)': 'percentile_rank_runs_allowed_ab',
    '64 Hitting Rank (Raw)': 'sixty_four_rank_weighted_run_created_efficiency',
    '64 Pitching Rank (Raw)': 'sixty_four_rank_weighted_run_allowed_efficiency',
}

# Map rank columns to hitting or pitching for PA/IP filtering
HITTING_RANK_COLS = {
    'integer_rank_weighted_run_created_efficiency',
    'percentile_rank_weighted_run_created_efficiency',
    'percentile_rank_runs_created_ab',
    'sixty_four_rank_weighted_run_created_efficiency',
}
PITCHING_RANK_COLS = {
    'integer_rank_weighted_run_allowed_efficiency',
    'percentile_rank_weighted_run_allowed_efficiency',
    'percentile_rank_runs_allowed_ab',
    'sixty_four_rank_weighted_run_allowed_efficiency',
}

PERCENTILE_COLS = {
    'sixty_four_rank_weighted_run_efficiency',
    'rank_weighted_run_created_per_35_at_bat',
    'rank_weighted_run_allowed_per_35_at_bat',
    'percentile_rank_weighted_run_created_efficiency',
    'percentile_rank_weighted_run_allowed_efficiency',
    'percentile_rank_runs_created_ab',
    'percentile_rank_runs_allowed_ab',
    'sixty_four_rank_weighted_run_created_efficiency',
    'sixty_four_rank_weighted_run_allowed_efficiency',
}

# Columns where lower value = better rank (integer ranks, integer team ranks)
# For these, a negative delta means IMPROVEMENT, so we invert the color/direction
LOWER_IS_BETTER_COLS = {
    'integer_rank_weighted_run_created_efficiency',
    'integer_rank_weighted_run_allowed_efficiency',
    'integer_64_rank_total',
    'integer_weighted_run_created_per_35_at_bat',
    'integer_weighted_run_allowed_per_35_at_bat',
}


# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_rank_csv(path):
    df = pd.read_csv(path, low_memory=False, encoding='latin-1')
    for col in ['team_id', 'player_id']:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors='coerce')
            df[col] = np.where(numeric.notna(), numeric.fillna(0).astype(int).astype(str), df[col])
    if 'year' in df.columns:
        df['year'] = df['year'].astype(str)
    return df


@st.cache_data
def load_hitting_pa(year):
    """Load plate appearances per player for a given year."""
    df = pd.read_csv(DATA_DIR / 'hitting.csv', low_memory=False, encoding='latin-1',
                     usecols=['player_id', 'year', 'plate_appearances'])
    df['year'] = df['year'].astype(str)
    df['player_id'] = pd.to_numeric(df['player_id'], errors='coerce').fillna(0).astype(int).astype(str)
    df = df[df['year'] == year]
    df['plate_appearances'] = pd.to_numeric(df['plate_appearances'], errors='coerce').fillna(0)
    return dict(zip(df['player_id'], df['plate_appearances']))


@st.cache_data
def load_pitching_ip(year):
    """Load innings pitched per player for a given year."""
    df = pd.read_csv(DATA_DIR / 'pitching.csv', low_memory=False, encoding='latin-1',
                     usecols=['player_id', 'year', 'innings_pitched'])
    df['year'] = df['year'].astype(str)
    df['player_id'] = pd.to_numeric(df['player_id'], errors='coerce').fillna(0).astype(int).astype(str)
    df = df[df['year'] == year]
    df['innings_pitched'] = pd.to_numeric(df['innings_pitched'], errors='coerce').fillna(0)
    return dict(zip(df['player_id'], df['innings_pitched']))


@st.cache_data
def load_teams():
    teams = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False)
    confs = pd.read_csv(DATA_DIR / 'conferences.csv', low_memory=False)
    teams = teams.merge(confs[['id', 'name', 'abbreviation', 'division', 'classification']],
                        left_on='conference_id', right_on='id', suffixes=('', '_conf'))
    teams = teams.rename(columns={'id': 'team_db_id', 'name': 'team_name',
                                  'name_conf': 'conference_name'})
    teams['team_db_id'] = pd.to_numeric(teams['team_db_id'], errors='coerce').fillna(0).astype(int).astype(str)
    bb = teams[teams['sport'] == 'Baseball'][['team_name', 'team_db_id']].drop_duplicates('team_name')
    bb_name_to_id = dict(zip(bb['team_name'], bb['team_db_id']))
    teams['logo_id'] = teams.apply(
        lambda r: r['team_db_id'] if r['sport'] == 'Baseball'
        else bb_name_to_id.get(r['team_name'], r['team_db_id']), axis=1)
    return teams


@st.cache_data
def load_players():
    df = pd.read_csv(DATA_DIR / 'players.csv', low_memory=False, encoding='latin-1')
    df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype(int).astype(str)
    df['team_id'] = pd.to_numeric(df['team_id'], errors='coerce').fillna(0).astype(int).astype(str)
    return df


@st.cache_data
def load_brand_logo(path):
    return np.array(Image.open(path).convert('RGBA'))


def abbreviate(name, max_len=0):
    """Shorten team/player names for x-axis labels. No truncation by default."""
    replacements = [
        ('University of ', ''), ('University', ''), (' State', ' St.'),
        ('Northern ', 'N. '), ('Southern ', 'S. '), ('Eastern ', 'E. '),
        ('Western ', 'W. '), ('Central ', 'C. '),
    ]
    s = name
    for old, new in replacements:
        s = s.replace(old, new)
    return s.upper()


# ── Build Comparison Data ─────────────────────────────────────────────────────
def build_team_data(rank_col, year, sport, division, conference, teams_df):
    cur = load_rank_csv(DATA_DIR / 'team_rank.csv')
    old = load_rank_csv(OLD_DIR / 'team_rank_old.csv')
    cur = cur[cur['year'] == year]
    old = old[old['year'] == year]
    if rank_col not in cur.columns:
        st.error(f'Column {rank_col} not found in team_rank.csv')
        st.stop()

    sport_teams = teams_df[teams_df['sport'] == sport]
    if division != 'All':
        sport_teams = sport_teams[sport_teams['division'] == division]
    if conference != 'All':
        sport_teams = sport_teams[sport_teams['conference_name'] == conference]
    valid_ids = set(sport_teams['team_db_id'].values)

    cur = cur[cur['team_id'].isin(valid_ids)][['team_id', rank_col]].copy()
    old = old[old['team_id'].isin(valid_ids)][['team_id', rank_col]].copy()
    cur = cur.rename(columns={rank_col: 'end'})
    old = old.rename(columns={rank_col: 'start'})

    merged = cur.merge(old, on='team_id', how='inner')
    merged['end'] = pd.to_numeric(merged['end'], errors='coerce')
    merged['start'] = pd.to_numeric(merged['start'], errors='coerce')
    merged = merged.dropna(subset=['start', 'end'])
    merged['delta'] = merged['end'] - merged['start']

    name_map = dict(zip(sport_teams['team_db_id'], sport_teams['team_name']))
    logo_map = dict(zip(sport_teams['team_db_id'], sport_teams['logo_id']))
    merged['name'] = merged['team_id'].map(name_map)
    merged['logo_id'] = merged['team_id'].map(logo_map)
    merged = merged.dropna(subset=['name'])
    return merged


def build_player_data(rank_col, year, sport, division, conference, teams_df):
    cur = load_rank_csv(DATA_DIR / 'player_rank.csv')
    old = load_rank_csv(OLD_DIR / 'player_rank_old.csv')
    cur = cur[cur['year'] == year]
    old = old[old['year'] == year]
    if rank_col not in cur.columns:
        st.error(f'Column {rank_col} not found in player_rank.csv')
        st.stop()

    sport_teams = teams_df[teams_df['sport'] == sport]
    if division != 'All':
        sport_teams = sport_teams[sport_teams['division'] == division]
    if conference != 'All':
        sport_teams = sport_teams[sport_teams['conference_name'] == conference]
    valid_ids = set(sport_teams['team_db_id'].values)

    cur = cur[cur['team_id'].isin(valid_ids)][['player_id', 'team_id', rank_col]].copy()
    old = old[old['team_id'].isin(valid_ids)][['player_id', 'team_id', rank_col]].copy()
    cur = cur.rename(columns={rank_col: 'end'})
    old = old.rename(columns={rank_col: 'start'})

    merged = cur.merge(old, on=['player_id', 'team_id'], how='inner')
    merged['end'] = pd.to_numeric(merged['end'], errors='coerce')
    merged['start'] = pd.to_numeric(merged['start'], errors='coerce')
    merged = merged.dropna(subset=['start', 'end'])
    merged['delta'] = merged['end'] - merged['start']

    players_df = load_players()
    player_names = dict(zip(players_df['id'], players_df['player_name']))
    merged['name'] = merged['player_id'].map(player_names)

    name_map = dict(zip(sport_teams['team_db_id'], sport_teams['team_name']))
    logo_map = dict(zip(sport_teams['team_db_id'], sport_teams['logo_id']))
    merged['team_name'] = merged['team_id'].map(name_map)
    merged['logo_id'] = merged['team_id'].map(logo_map)
    merged = merged.dropna(subset=['name'])
    return merged


# ── Chart Rendering ───────────────────────────────────────────────────────────
def render_stock_chart(data, cfg):
    t = THEMES[cfg['theme']]
    n = len(data)
    is_pct = cfg['is_percentile']

    # Match chart_builder layout: large figure, manual positioning
    fig_w = max(18, n * 1.1)
    fig = plt.figure(figsize=(fig_w, 14))
    fig.patch.set_facecolor(t['bg'])

    # ── Header (matches chart_builder exactly) ──
    # Top-left: sport / division / year
    div_label = cfg['division'] if cfg['division'] != 'All' else 'ALL DIVISIONS'
    fig.text(0.03, 0.965,
             f'{div_label.upper()} {cfg["sport"].upper()} {cfg["year"]}',
             fontsize=13, color=t['text_md'], fontfamily=SUBTITLE_FONT, fontweight='bold')

    # Center: main title
    title_text = (cfg.get('title') or 'STOCK UP / STOCK DOWN').strip().upper()
    fig.text(0.50, 0.905, title_text,
             fontsize=40, color=t['text'], fontfamily=TITLE_FONT, fontweight='bold',
             ha='center')

    # Top-right: count + season
    mode_word = 'PROGRAMS' if cfg['mode'] == 'Team' else 'PLAYERS'
    fig.text(0.97, 0.965, f'{n} {mode_word}  \u00b7  {cfg["year"]} SEASON',
             fontsize=12, color=t['text_dk'], fontfamily=BODY_FONT, ha='right')

    # Brand logo (top-right, same position as chart_builder)
    brand_arr = load_brand_logo(t['brand_logo'])
    logo_ax = fig.add_axes([0.90, 0.89, 0.08, 0.08])
    logo_ax.imshow(brand_arr)
    logo_ax.axis('off')

    # Subtitle under main title
    metric_label = cfg['metric_label'].upper()
    subtitle_text = cfg.get('subtitle', '').strip()
    if not subtitle_text:
        subtitle_text = f'{metric_label} \u2014 START OF WEEK VS. END OF WEEK'
    else:
        subtitle_text = subtitle_text.upper()
    fig.text(0.50, 0.875, subtitle_text,
             fontsize=12, color=t['text_md'], fontfamily=SUBTITLE_FONT, ha='center')

    # ── Legend ──
    lower_better = cfg.get('lower_is_better', False)
    up_label = 'Improved (rank decreased)' if lower_better else 'End of week (higher)'
    down_label = 'Declined (rank increased)' if lower_better else 'End of week (lower)'
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=t['up'],
               markeredgecolor=t['up'], markersize=9, label=up_label),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=t['down'],
               markeredgecolor=t['down'], markersize=9, label=down_label),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor=t['start_marker'], markeredgewidth=1.5, markersize=9,
               label='Start of week'),
    ]
    leg_ax = fig.add_axes([0.25, 0.845, 0.50, 0.03])
    leg_ax.set_xlim(0, 1)
    leg_ax.set_ylim(0, 1)
    leg_ax.axis('off')
    leg = leg_ax.legend(handles=legend_elements, loc='center', ncol=3,
                        frameon=False, fontsize=10, labelcolor=t['text_sub'],
                        handletextpad=0.5, columnspacing=2.5)

    # ── Main axes ──
    ax = fig.add_axes([0.06, 0.08, 0.90, 0.74])
    ax.set_facecolor(t['plot_bg'])

    xs = np.arange(n)

    # Prepare display values
    starts = data['start'].values.copy()
    ends = data['end'].values.copy()
    deltas = data['delta'].values.copy()
    if is_pct:
        starts = starts * 100
        ends = ends * 100
        deltas = deltas * 100

    # Resolve logo paths and headshots
    logo_ids = data['logo_id'].values if 'logo_id' in data.columns else [None] * n
    player_ids = data['player_id'].values if 'player_id' in data.columns else [None] * n
    logo_zoom = cfg.get('logo_zoom', 0.06)
    headshots = cfg.get('headshots', {})
    lower_better = cfg.get('lower_is_better', False)
    is_player_mode = cfg['mode'] == 'Player'
    headshot_size = int(logo_zoom * 1200)  # Scale with logo zoom

    def is_improved(delta):
        """Return True if this delta represents improvement."""
        if lower_better:
            return delta < 0  # rank went down numerically = improved
        return delta > 0      # value went up = improved

    def make_circular_headshot(img_data, size):
        """Create a circular headshot with red border, matching chart_builder style."""
        pil_img = Image.open(BytesIO(img_data)).convert('RGBA')
        w, h = pil_img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        pil_img = pil_img.crop((left, top, left + side, top + side))
        pil_img = pil_img.resize((size, size), Image.LANCZOS)
        # Circular mask
        mask = Image.new('L', (size, size), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
        # Border ring
        border = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        bd = ImageDraw.Draw(border)
        bd.ellipse((0, 0, size - 1, size - 1),
                   outline=(196, 18, 48, 255), width=3)
        # Composite
        result = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        result.paste(pil_img, mask=mask)
        result = Image.alpha_composite(result, border)
        return np.array(result)

    # Draw lollipops
    for i in range(n):
        s, e, d = starts[i], ends[i], deltas[i]
        color = t['up'] if is_improved(d) else t['down'] if d != 0 else t['start_marker']

        # Vertical connecting line
        ax.plot([i, i], [s, e], color=color, linewidth=3, solid_capstyle='round', zorder=2)

        # Start circle (open/hollow)
        ax.scatter(i, s, s=80, facecolors='none', edgecolors=t['start_marker'],
                   linewidths=2, zorder=3)

        # End point: headshot (player mode) > team logo > filled circle
        pid = str(player_ids[i]) if player_ids[i] is not None else ''
        hs_data = headshots.get(pid) if is_player_mode else None

        if hs_data is not None:
            try:
                hs_img = make_circular_headshot(hs_data, headshot_size)
                ab = AnnotationBbox(OffsetImage(hs_img, zoom=1.0, alpha=0.95),
                                    (i, e), frameon=False, zorder=5)
                ax.add_artist(ab)
            except Exception:
                ax.scatter(i, e, s=80, facecolors=color, edgecolors=color,
                           linewidths=1.5, zorder=4)
        else:
            logo_path = LOGO_DIR / f'{logo_ids[i]}.png' if logo_ids[i] else None
            if logo_path and logo_path.exists():
                img = mpimg.imread(str(logo_path))
                ab = AnnotationBbox(OffsetImage(img, zoom=logo_zoom, alpha=0.95),
                                    (i, e), frameon=False, zorder=5)
                ax.add_artist(ab)
            else:
                ax.scatter(i, e, s=80, facecolors=color, edgecolors=color,
                           linewidths=1.5, zorder=4)

        # Value label at the end point
        above = e >= s
        va = 'bottom' if above else 'top'
        fmt = '.0f' if lower_better else '.1f'
        ax.annotate(f'{e:{fmt}}', (i, e), textcoords='offset points',
                    xytext=(0, 14 if above else -14), ha='center', va=va,
                    fontsize=8.5, fontfamily=BODY_FONT, fontweight='bold',
                    color=t['text_sub'])

    # X-axis tick labels: name + team (for players) + delta
    # For "lower is better", flip the display delta so improvement shows as positive
    x_labels = []
    x_colors = []
    is_player = cfg['mode'] == 'Player'
    fmt = '.0f' if lower_better else '.1f'
    for i, (_, row) in enumerate(data.iterrows()):
        abbr = abbreviate(row['name'])
        d = deltas[i]
        display_d = -d if lower_better else d
        sign = '+' if display_d >= 0 else ''
        if is_player and 'team_name' in row and pd.notna(row.get('team_name')):
            team_abbr = abbreviate(row['team_name'])
            x_labels.append(f'{abbr}\n{team_abbr}\n{sign}{display_d:{fmt}}')
        else:
            x_labels.append(f'{abbr}\n{sign}{display_d:{fmt}}')
        x_colors.append(t['up'] if is_improved(d) else t['down'] if d != 0 else t['start_marker'])

    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, fontsize=8, fontfamily=AXIS_FONT, fontweight='bold')
    for tick_label, color in zip(ax.get_xticklabels(), x_colors):
        tick_label.set_color(color)

    # Y-axis
    ax.tick_params(axis='y', labelsize=10, labelcolor=t['text_sub'], colors=t['spine'])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}'))

    # Grid — horizontal only
    ax.grid(axis='y', color=t['grid'], linewidth=0.5, alpha=0.7)
    ax.grid(axis='x', visible=False)

    # Spines
    for spine in ax.spines.values():
        spine.set_color(t['spine'])
        spine.set_linewidth(0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Y padding
    all_vals = np.concatenate([starts, ends])
    if len(all_vals) == 0:
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.5, 0.5)
    else:
        y_range = all_vals.max() - all_vals.min()
        pad = max(y_range * 0.12, 2)
        ax.set_ylim(all_vals.min() - pad, all_vals.max() + pad)
        ax.set_xlim(-0.6, n - 0.4)

    # Footer
    view = cfg.get('view', 'Biggest Movers')
    view_labels = {
        'Top Gainers': 'top gainers',
        'Top Losers': 'top losers',
        'Biggest Movers': 'largest absolute change',
        'All': 'largest absolute change',
    }
    sort_desc = view_labels.get(view, 'net change')
    fig.text(0.50, 0.025, f'Sorted by {sort_desc}  \u00b7  64analytics.com',
             fontsize=9, color=t['text_dk'], fontfamily=BODY_FONT, ha='center')

    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar_config():
    st.sidebar.image(str(BRAND_LOGO_DARK), width=80)
    st.sidebar.markdown('## Stock Chart')
    if st.sidebar.button('Reload data'):
        st.cache_data.clear()
        st.rerun()

    cfg = {}
    teams_df = load_teams()

    cfg['mode'] = st.sidebar.radio('Mode', ['Team', 'Player'], horizontal=True)

    st.sidebar.markdown('---')
    st.sidebar.markdown('### Filters')
    cfg['sport'] = st.sidebar.selectbox('Sport', ['Baseball', 'Softball'])
    sport_teams = teams_df[teams_df['sport'] == cfg['sport']]

    divisions = ['All'] + sorted(sport_teams['division'].dropna().unique().tolist())
    cfg['division'] = st.sidebar.selectbox('Division', divisions)
    if cfg['division'] != 'All':
        sport_teams = sport_teams[sport_teams['division'] == cfg['division']]

    conferences = ['All'] + sorted(sport_teams['conference_name'].dropna().unique().tolist())
    cfg['conference'] = st.sidebar.selectbox('Conference', conferences)

    years = ['2026', '2025', '2024', '2023', '2022']
    cfg['year'] = st.sidebar.selectbox('Year', years)

    st.sidebar.markdown('---')
    st.sidebar.markdown('### Metric')
    if cfg['mode'] == 'Team':
        rank_options = list(TEAM_RANK_COLS.keys())
        selected = st.sidebar.selectbox('Rank Column', rank_options)
        cfg['rank_col'] = TEAM_RANK_COLS[selected]
        cfg['metric_label'] = selected
    else:
        rank_options = list(PLAYER_RANK_COLS.keys())
        selected = st.sidebar.selectbox('Rank Column', rank_options)
        cfg['rank_col'] = PLAYER_RANK_COLS[selected]
        cfg['metric_label'] = selected

    cfg['is_percentile'] = cfg['rank_col'] in PERCENTILE_COLS
    cfg['lower_is_better'] = cfg['rank_col'] in LOWER_IS_BETTER_COLS

    # PA / IP filters (player mode only)
    cfg['min_pa'] = 0
    cfg['min_ip'] = 0.0
    if cfg['mode'] == 'Player':
        is_hitting = cfg['rank_col'] in HITTING_RANK_COLS
        is_pitching = cfg['rank_col'] in PITCHING_RANK_COLS
        if is_hitting:
            cfg['min_pa'] = st.sidebar.number_input('Min. Plate Appearances', value=50,
                                                     min_value=0, step=10,
                                                     help='Exclude hitters below this PA threshold')
        elif is_pitching:
            cfg['min_ip'] = st.sidebar.number_input('Min. Innings Pitched', value=10.0,
                                                     min_value=0.0, step=5.0,
                                                     help='Exclude pitchers below this IP threshold')

    # Rank filter — e.g. "biggest movers in the top 250"
    cfg['max_rank'] = 0
    cfg['min_percentile'] = 0.0
    if cfg['lower_is_better']:
        cfg['max_rank'] = st.sidebar.number_input('Max current rank to include', value=0,
                                                    min_value=0, step=50,
                                                    help='Only show players/teams currently ranked at or above this. '
                                                         '0 = no filter. E.g. 250 = top 250 only.')
    elif cfg['is_percentile']:
        cfg['min_percentile'] = st.sidebar.number_input('Min current percentile', value=0.0,
                                                         min_value=0.0, max_value=1.0, step=0.05,
                                                         help='Only show players/teams at or above this percentile. '
                                                              '0 = no filter. E.g. 0.50 = top half only.')

    st.sidebar.markdown('---')
    st.sidebar.markdown('### Chart')
    cfg['title'] = st.sidebar.text_input('Title', value='STOCK UP / STOCK DOWN')
    cfg['subtitle'] = st.sidebar.text_input('Subtitle', value='',
                                             help='Leave blank for auto-generated subtitle from metric name')

    st.sidebar.markdown('---')
    st.sidebar.markdown('### Display')
    cfg['view'] = st.sidebar.radio('View',
                                    ['Top Gainers', 'Top Losers', 'Biggest Movers', 'All'],
                                    horizontal=False,
                                    help='Pareto: show only gainers, losers, or biggest absolute movers')
    cfg['top_n'] = st.sidebar.slider('Max to show', 3, 50, 10)
    cfg['min_delta'] = st.sidebar.number_input('Min. change to include', value=0.0, step=0.01,
                                                help='Exclude small movers (in raw units)')
    cfg['logo_zoom'] = st.sidebar.slider('Logo size', 0.02, 0.15, 0.06, 0.01)

    cfg['teams_df'] = teams_df
    cfg['sport_teams'] = sport_teams
    return cfg


# ── Main ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='64 Analytics — Stock Chart', layout='wide',
                   initial_sidebar_state='expanded')

cfg = sidebar_config()

# Theme toggle + exclusion filter (above chart, like chart_builder)
col1, col2 = st.columns([4, 1])
_use_light = col2.checkbox('Light theme', value=False, key='stock_light')
cfg['theme'] = 'Light' if _use_light else 'Dark'
theme = THEMES[cfg['theme']]

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, .stApp, .stApp * {{ font-family: 'Inter', sans-serif !important; }}
.stApp {{ background-color: {theme['streamlit_bg']}; }}
h1, h2, h3, p, label, .stMarkdown {{ color: {theme['text_sub']} !important; }}
</style>
""", unsafe_allow_html=True)

# Build data
teams_df = cfg['teams_df']
if cfg['mode'] == 'Team':
    data = build_team_data(cfg['rank_col'], cfg['year'], cfg['sport'],
                           cfg['division'], cfg['conference'], teams_df)
else:
    data = build_player_data(cfg['rank_col'], cfg['year'], cfg['sport'],
                             cfg['division'], cfg['conference'], teams_df)

if len(data) == 0:
    st.warning('No data. Check that both current and old rank CSVs exist and have matching records.')
    st.stop()

# ── Apply PA / IP filter (player mode) ──
if cfg['mode'] == 'Player' and 'player_id' in data.columns:
    rank_col = cfg['rank_col']
    if rank_col in HITTING_RANK_COLS and cfg['min_pa'] > 0:
        pa_map = load_hitting_pa(cfg['year'])
        data['_pa'] = data['player_id'].map(pa_map).fillna(0)
        before = len(data)
        data = data[data['_pa'] >= cfg['min_pa']]
        filtered = before - len(data)
        if filtered > 0:
            st.caption(f'Filtered {filtered:,} players below {cfg["min_pa"]} PA')
        data = data.drop(columns=['_pa'])
    elif rank_col in PITCHING_RANK_COLS and cfg['min_ip'] > 0:
        ip_map = load_pitching_ip(cfg['year'])
        data['_ip'] = data['player_id'].map(ip_map).fillna(0)
        before = len(data)
        data = data[data['_ip'] >= cfg['min_ip']]
        filtered = before - len(data)
        if filtered > 0:
            st.caption(f'Filtered {filtered:,} players below {cfg["min_ip"]:.0f} IP')
        data = data.drop(columns=['_ip'])

if len(data) == 0:
    st.warning('No players meet the minimum PA/IP threshold. Try lowering it.')
    st.stop()

# ── Rank range filter ──
if cfg.get('max_rank', 0) > 0 and cfg['lower_is_better']:
    # For integer ranks: keep only players/teams whose CURRENT rank <= max_rank
    before = len(data)
    data = data[data['end'] <= cfg['max_rank']]
    filtered = before - len(data)
    if filtered > 0:
        st.caption(f'Filtered to top {cfg["max_rank"]} (removed {filtered:,})')
elif cfg.get('min_percentile', 0) > 0 and cfg['is_percentile']:
    before = len(data)
    data = data[data['end'] >= cfg['min_percentile']]
    filtered = before - len(data)
    if filtered > 0:
        st.caption(f'Filtered to percentile >= {cfg["min_percentile"]:.0%} (removed {filtered:,})')

if len(data) == 0:
    st.warning('No data after rank filter. Try increasing the max rank.')
    st.stop()

# ── Exclude specific teams/players ──
with st.expander('Exclude from chart'):
    all_names = sorted(data['name'].dropna().unique().tolist())
    entity = 'players' if cfg['mode'] == 'Player' else 'teams'
    excluded = st.multiselect(f'Select {entity} to exclude', all_names,
                              help=f'Type to search. Excluded {entity} will not appear on the chart.')
    if excluded:
        data = data[~data['name'].isin(excluded)]

if len(data) == 0:
    st.warning(f'All {entity} excluded.')
    st.stop()

# Filter tiny movers
if cfg['min_delta'] > 0:
    data = data[data['delta'].abs() >= cfg['min_delta']]

if len(data) == 0:
    st.warning('All movers filtered out. Try lowering the minimum change threshold.')
    st.stop()

# ── Pareto: select top movers by view ──
top_n = cfg['top_n']
view = cfg['view']

lower_better = cfg.get('lower_is_better', False)

if view == 'Top Gainers':
    if lower_better:
        # Improved = delta < 0 (rank number decreased)
        data = data[data['delta'] < 0].nsmallest(top_n, 'delta')
    else:
        data = data[data['delta'] > 0].nlargest(top_n, 'delta')
elif view == 'Top Losers':
    if lower_better:
        # Declined = delta > 0 (rank number increased)
        data = data[data['delta'] > 0].nlargest(top_n, 'delta')
    else:
        data = data[data['delta'] < 0].nsmallest(top_n, 'delta')
else:  # Biggest Movers / All
    data = data[data['delta'] != 0]

# Always sort by absolute delta for pareto (biggest movers on left)
data = data.reindex(data['delta'].abs().sort_values(ascending=False).index).head(top_n)

data = data.reset_index(drop=True)

if len(data) == 0:
    st.info('No movers found — the old and current rank files may be identical. '
            'Data will diverge after tonight\'s update.')
    st.stop()

col1.metric('Movers shown', len(data))

# ── Player headshot uploads (player mode only) ──
cfg['headshots'] = {}
if cfg['mode'] == 'Player' and 'player_id' in data.columns:
    with st.expander('Upload player headshots'):
        st.caption('Upload headshots to display at the end-point instead of team logos.')
        # Build name list from current data
        player_opts = []
        pid_map = {}
        for _, row in data.iterrows():
            name = row.get('name', 'Unknown')
            team = row.get('team_name', '')
            pid = str(row['player_id'])
            label = f'{name} — {team}' if team else name
            player_opts.append(label)
            pid_map[label] = pid

        selected_players = st.multiselect('Select players', sorted(set(player_opts)),
                                          help='Upload a headshot for each selected player')
        for label in selected_players:
            pid = pid_map.get(label, '')
            uploaded = st.file_uploader(f'{label}', type=['png', 'jpg', 'jpeg'],
                                        key=f'hs_{pid}')
            if uploaded:
                cfg['headshots'][pid] = uploaded.read()

# Render
with st.spinner('Rendering chart...'):
    fig = render_stock_chart(data, cfg)

buf = BytesIO()
fig.savefig(buf, format='png', dpi=180, bbox_inches='tight',
            facecolor=theme['bg'], edgecolor='none')
buf.seek(0)
plt.close(fig)

st.image(buf, use_container_width=True)

buf.seek(0)
st.download_button('Download PNG', data=buf, file_name='64analytics_stock_chart.png',
                   mime='image/png')

with st.expander('View data table'):
    display = data.copy()
    if cfg['is_percentile']:
        display['start'] = display['start'] * 100
        display['end'] = display['end'] * 100
        display['delta'] = display['delta'] * 100
    if cfg['lower_is_better']:
        display['delta'] = -display['delta']
    show_cols = ['name']
    if 'team_name' in display.columns and cfg['mode'] == 'Player':
        show_cols.append('team_name')
    show_cols += ['start', 'end', 'delta']
    display = display[show_cols].rename(columns={
        'name': 'Player' if cfg['mode'] == 'Player' else 'Team',
        'team_name': 'Team', 'start': 'Start', 'end': 'End', 'delta': 'Change',
    })
    st.dataframe(display, use_container_width=True)
