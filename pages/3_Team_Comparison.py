"""
64 Analytics — Team Comparison Matrix
Side-by-side comparison of multiple teams across selected metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from pathlib import Path
from io import BytesIO
import matplotlib.font_manager as fm

# ── Path setup (works locally and on Streamlit Cloud) ─────────────────────────
_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _APP_DIR / 'data'
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
        'cell_high': '#1a3322', 'cell_low': '#331a1a', 'cell_mid': '#2a2a2a',
        'brand_logo': BRAND_LOGO_DARK,
        'streamlit_bg': '#1a1a1a',
    },
    'Light': {
        'bg': '#FFFFFF', 'plot_bg': '#F5F5F5',
        'text': '#1a1a1a', 'text_sub': '#333333',
        'text_md': '#666666', 'text_dk': '#999999',
        'grid': '#E0E0E0', 'spine': '#CCCCCC',
        'cell_high': '#D8F0E0', 'cell_low': '#F0D8D8', 'cell_mid': '#F0F0F0',
        'brand_logo': BRAND_LOGO_LIGHT,
        'streamlit_bg': '#FFFFFF',
    },
}

# ── CSV sources for team data ─────────────────────────────────────────────────
TEAM_CSVS = {
    'Hitting': 'hitting_team.csv',
    'Pitching': 'pitching_team.csv',
    'Fielding': 'fielding_team.csv',
    '64 Rank': 'team_rank.csv',
    'Schedule': 'schedules.csv',
}

# Columns to exclude from metric selection (identifiers, not plottable)
ID_COLS = {'id', 'team_id', 'year', 'Year', 'player_id'}

# Friendly display names for common columns
FRIENDLY_NAMES = {
    # Hitting
    'on_base_plus_slugging': 'OPS',
    'batting_average': 'Batting Avg',
    'on_base_percentage': 'OBP',
    'slugging_percentage': 'SLG',
    'isolated_power': 'ISO',
    'weighted_on_base_average': 'wOBA',
    'weighted_runs_created': 'wRC',
    'weighted_runs_above_average': 'wRAA',
    'weighted_runs_created_plus': 'wRC+',
    'runs_plate_appearance': 'R/PA',
    'strikeout_percentage': 'K%',
    'walk_percentage': 'BB%',
    'strikeout_to_walk_ratio': 'K/BB',
    'batting_average_on_balls_in_play': 'BABIP',
    'home_runs': 'HR',
    'stolen_bases': 'SB',
    'runs_scored': 'Runs',
    'plate_appearances': 'PA',
    'games_played': 'GP',
    'at_bats': 'AB',
    'hits': 'H',
    'doubles': '2B',
    'triples': '3B',
    'total_bases': 'TB',
    'runs_batted_in': 'RBI',
    'walks': 'BB',
    'hit_by_pitch': 'HBP',
    'sac_fly': 'SF',
    'sac_hit': 'SH',
    'strikeouts': 'K',
    'caught_stealing': 'CS',
    'picked_off': 'PO',
    'intentional_walk': 'IBB',
    # Pitching
    'earned_run_average': 'ERA',
    'fielding_independent_pitching': 'FIP',
    'expected_fielding_independent_pitching': 'xFIP',
    'skill_interactive_earned_run_average': 'SIERA',
    'walks_plus_hits_per_inning_pitched': 'WHIP',
    'batting_average_against': 'BAA',
    'on_base_percentage_against': 'OBA Against',
    'slugging_percentage_against': 'SLG Against',
    'on_base_plus_slugging_against': 'OPS Against',
    'left_on_base_percentage': 'LOB%',
    'ground_out_per_air_out': 'GO/AO',
    'home_run_to_fly_ball_percentage': 'HR/FB%',
    'strikeout_minus_walk_percentage': 'K-BB%',
    'innings_pitched': 'IP',
    'complete_games': 'CG',
    'hits_allowed': 'H Allowed',
    'runs_allowed': 'R Allowed',
    'earned_runs': 'ER',
    'walks_issued': 'BB Issued',
    'shutouts': 'SHO',
    'batters_faced': 'BF',
    'opponent_at_bats': 'Opp AB',
    'doubles_allowed': '2B Allowed',
    'triples_allowed': '3B Allowed',
    'balk': 'BK',
    'homeruns_allowed': 'HR Allowed',
    'wild_pitch': 'WP',
    'hit_batter': 'HBP (P)',
    'intentional_walk_allowed': 'IBB (P)',
    'inherited_runner': 'IR',
    'inherited_runner_scored': 'IRS',
    'sac_hit_allowed': 'SH Allowed',
    'sac_fly_allowed': 'SF Allowed',
    'pitches': 'Pitches',
    'ground_out': 'GO',
    'fly_out': 'FO',
    'wins': 'W',
    'losses': 'L',
    'saves': 'SV',
    'strikeout_looking': 'KL',
    'pickoffs': 'Pickoffs',
    'strikeout_looking_percentage': 'KL%',
    'strikeout_swinging_percentage': 'KS%',
    'pitches_per_inning': 'P/Inn',
    'strikeouts_per_9_innings': 'K/9',
    'strikeouts_per_7_innings': 'K/7',
    'games': 'GP',
    # Fielding
    'fielding_percentage': 'Fld%',
    'errors': 'Errors',
    'double_plays_turned': 'DP',
    'range_factor': 'RF',
    'passed_balls': 'PB',
    'caught_stealing_by': 'CS By',
    'putouts': 'PO',
    'assists': 'A',
    'total_chances': 'TC',
    'catchers_interference': 'CI',
    'stolen_bases_allowed': 'SB Allowed',
    # 64 Rank
    'integer_64_rank_total': '64 Rank (#)',
    'sixty_four_rank_weighted_run_efficiency': '64 Rank (Pctl)',
    'integer_weighted_run_created_per_35_at_bat': 'RC Rank (#)',
    'integer_weighted_run_allowed_per_35_at_bat': 'RA Rank (#)',
    'rank_weighted_run_created_per_35_at_bat': 'RC Rank (Pctl)',
    'rank_weighted_run_allowed_per_35_at_bat': 'RA Rank (Pctl)',
    'runs_created_ab': 'RC/AB',
    'runs_allowed_ab': 'RA/AB',
    'weighted_run_created_efficiency': 'WRC Efficiency',
    'weighted_run_allowed_efficiency': 'WRA Efficiency',
    'weighted_run_created_per_35_at_bats': 'WRC/35 AB',
    'weighted_run_allowed_per_35_at_bats': 'WRA/35 AB',
    'sixty_four_metric_weighted_run_efficiency': '64 Metric',
    # Schedule
    'Conf_Win': 'Conf W', 'Conf_Loss': 'Conf L', 'Conf_Tie': 'Conf T',
    'OOC_Win': 'OOC W', 'OOC_Loss': 'OOC L', 'OOC_Tie': 'OOC T',
    'Post_Win': 'Post W', 'Post_Loss': 'Post L', 'Post_Tie': 'Post T',
}

# Metrics where lower is better (for color coding)
LOWER_IS_BETTER = {
    'earned_run_average', 'fielding_independent_pitching',
    'expected_fielding_independent_pitching', 'skill_interactive_earned_run_average',
    'walks_plus_hits_per_inning_pitched', 'batting_average_against',
    'on_base_percentage_against', 'slugging_percentage_against',
    'on_base_plus_slugging_against',
    'errors', 'passed_balls', 'stolen_bases_allowed',
    'home_run_to_fly_ball_percentage',
    'runs_allowed', 'earned_runs', 'hits_allowed', 'homeruns_allowed',
    'walks_issued', 'wild_pitch', 'balk', 'hit_batter',
    'inherited_runner_scored',
    'integer_64_rank_total', 'integer_weighted_run_created_per_35_at_bat',
    'integer_weighted_run_allowed_per_35_at_bat',
    'Conf_Loss', 'OOC_Loss', 'Post_Loss',
    'caught_stealing', 'picked_off',
}

# Category-specific overrides: (category, col) -> higher is better even if in LOWER_IS_BETTER
HIGHER_IS_BETTER_OVERRIDE = {
    ('Pitching', 'strikeout_percentage'),
    ('Pitching', 'strikeout_minus_walk_percentage'),
    ('Pitching', 'strikeouts'),
    ('Pitching', 'strikeout_looking'),
    ('Pitching', 'strikeout_to_walk_ratio'),
}

# Category-specific overrides: (category, col) -> lower is better even if not in LOWER_IS_BETTER
LOWER_IS_BETTER_OVERRIDE = {
    ('Hitting', 'strikeout_percentage'),
    ('Hitting', 'strikeout_to_walk_ratio'),
    ('Hitting', 'strikeouts'),
    ('Pitching', 'walk_percentage'),
}


@st.cache_data
def build_metric_catalog():
    """Auto-discover all numeric columns from each team CSV."""
    catalog = {}
    for category, csv_file in TEAM_CSVS.items():
        try:
            df = pd.read_csv(DATA_DIR / csv_file, nrows=5, low_memory=False, encoding='latin-1')
        except Exception:
            continue
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        metrics = {}
        for col in numeric_cols:
            if col.lower() in {c.lower() for c in ID_COLS}:
                continue
            display = FRIENDLY_NAMES.get(col, col.replace('_', ' ').title())
            # Disambiguate duplicate display names across categories
            if category == 'Pitching' and col in ('strikeout_percentage', 'walk_percentage',
                                                    'strikeout_to_walk_ratio',
                                                    'batting_average_on_balls_in_play',
                                                    'strikeouts', 'games_played'):
                display = f'{display} (P)'
            metrics[display] = col
        catalog[category] = {'csv': csv_file, 'metrics': metrics}
    return catalog


# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(filename):
    df = pd.read_csv(DATA_DIR / filename, low_memory=False, encoding='latin-1')
    if 'Team_Id' in df.columns:
        df = df.rename(columns={'Team_Id': 'team_id'})
    if 'Year' in df.columns and 'year' not in df.columns:
        df = df.rename(columns={'Year': 'year'})
    if 'year' in df.columns:
        df['year'] = df['year'].astype(str)
    for col in ['team_id']:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors='coerce')
            df[col] = np.where(numeric.notna(), numeric.fillna(0).astype(int).astype(str), df[col])
    return df


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
def load_brand_logo(path):
    return np.array(Image.open(path).convert('RGBA'))


# ── Build comparison matrix ───────────────────────────────────────────────────
def build_matrix(team_ids, metrics_selected, year, catalog):
    """
    Returns a DataFrame: rows = metrics, columns = team_ids, values = metric values.
    Also returns metric_meta: list of (display_name, category, col_name, lower_is_better).
    """
    rows = {}
    metric_meta = []

    for category, display_name, col_name in metrics_selected:
        csv_file = catalog[category]['csv']
        df = load_csv(csv_file)
        df = df[df['year'] == year] if 'year' in df.columns else df

        if col_name not in df.columns:
            continue

        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        val_map = dict(zip(df['team_id'].astype(str), df[col_name]))

        row_vals = {}
        for tid in team_ids:
            row_vals[tid] = val_map.get(tid, np.nan)

        rows[display_name] = row_vals

        lib = col_name in LOWER_IS_BETTER
        if (category, col_name) in HIGHER_IS_BETTER_OVERRIDE:
            lib = False
        if (category, col_name) in LOWER_IS_BETTER_OVERRIDE:
            lib = True
        metric_meta.append((display_name, category, col_name, lib))

    matrix = pd.DataFrame(rows).T
    matrix.columns = team_ids
    return matrix, metric_meta


# ── Render comparison chart ───────────────────────────────────────────────────
def render_comparison(matrix, metric_meta, team_names, team_logo_ids, cfg):
    t = THEMES[cfg['theme']]
    n_teams = len(matrix.columns)
    n_metrics = len(matrix.index)

    if n_teams == 0 or n_metrics == 0:
        return None

    # 16:9 presentation slide aspect ratio
    fig_w = 22
    fig_h = max(10, 3.5 + n_metrics * 0.55)  # scale height with rows

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(t['bg'])

    # ── Header — compact, leaves room for the grid ──
    header_top = 1.0 - 0.3 / fig_h  # small top margin

    div_label = cfg.get('division', 'ALL')
    if div_label == 'All':
        div_label = 'ALL DIVISIONS'
    fig.text(0.03, header_top,
             f'{div_label.upper()} {cfg["sport"].upper()} {cfg["year"]}',
             fontsize=13, color=t['text_md'], fontfamily=SUBTITLE_FONT, fontweight='bold',
             va='top')

    title = (cfg.get('title') or 'TEAM COMPARISON').strip().upper()
    fig.text(0.50, header_top - 0.01, title,
             fontsize=36, color=t['text'], fontfamily=TITLE_FONT, fontweight='bold',
             ha='center', va='top')

    fig.text(0.97, header_top,
             f'{n_teams} TEAMS  \u00b7  {cfg["year"]} SEASON',
             fontsize=12, color=t['text_dk'], fontfamily=BODY_FONT, ha='right', va='top')

    brand_arr = load_brand_logo(t['brand_logo'])
    brand_size = 0.55 / fig_h  # keep logo proportional
    logo_ax = fig.add_axes([0.91, header_top - brand_size - 0.02, brand_size * fig_h / fig_w, brand_size])
    logo_ax.imshow(brand_arr)
    logo_ax.axis('off')

    # ── Main axes — fill nearly the entire figure ──
    grid_top = header_top - 1.8 / fig_h   # below header
    grid_bottom = 0.6 / fig_h              # small footer margin
    grid_left = 0.12                        # room for row labels
    grid_right = 0.97                       # near right edge

    ax = fig.add_axes([grid_left, grid_bottom, grid_right - grid_left, grid_top - grid_bottom])
    ax.set_facecolor(t['plot_bg'])

    # Layout: logo row at top (1.5 units), then metric rows (1 unit each)
    logo_rows = 1.5
    ax.set_xlim(0, n_teams)
    ax.set_ylim(-n_metrics, logo_rows)
    ax.axis('off')

    # ── Team logos at top ──
    logo_zoom = cfg.get('logo_zoom', 0.07)
    for j, tid in enumerate(matrix.columns):
        lid = team_logo_ids.get(tid, tid)
        logo_path = LOGO_DIR / f'{lid}.png'
        cx = j + 0.5
        cy = logo_rows * 0.5

        if logo_path.exists():
            img = mpimg.imread(str(logo_path))
            ab = AnnotationBbox(OffsetImage(img, zoom=logo_zoom, alpha=0.95),
                                (cx, cy), frameon=False, zorder=5)
            ax.add_artist(ab)
        else:
            ax.text(cx, cy, team_names.get(tid, tid)[:6],
                    ha='center', va='center', fontsize=9,
                    color=t['text'], fontfamily=BODY_FONT, fontweight='bold')

    # ── Divider line below logos ──
    ax.axhline(0, color=t['spine'], linewidth=1.5, zorder=3)

    # ── Draw cells ──
    for i, (metric_name, row_data) in enumerate(matrix.iterrows()):
        y_top = -i
        y_bot = -i - 1
        y_mid = -i - 0.5

        # Find meta for this metric
        meta = None
        for m in metric_meta:
            if m[0] == metric_name:
                meta = m
                break
        lower_better = meta[3] if meta else False

        # Row label (left side)
        ax.text(-0.2, y_mid, metric_name,
                ha='right', va='center', fontsize=11,
                color=t['text_sub'], fontfamily=BODY_FONT, fontweight='bold',
                transform=ax.transData)

        # Horizontal gridline
        ax.axhline(y_bot, color=t['grid'], linewidth=0.5, alpha=0.5)

        # Get values for color scaling
        vals = row_data.values.astype(float)
        valid = vals[~np.isnan(vals)]
        if len(valid) > 1:
            v_min, v_max = valid.min(), valid.max()
        else:
            v_min = v_max = valid[0] if len(valid) == 1 else 0

        for j, tid in enumerate(matrix.columns):
            val = row_data[tid]
            cx = j + 0.5

            # Cell background color based on rank among displayed teams
            if not np.isnan(val) and v_max > v_min:
                norm = (val - v_min) / (v_max - v_min)
                if lower_better:
                    norm = 1.0 - norm  # flip so low = good = green

                # Interpolate: low(red) -> mid(neutral) -> high(green)
                r_low = np.array(matplotlib.colors.to_rgb(t['cell_low']))
                r_mid = np.array(matplotlib.colors.to_rgb(t['cell_mid']))
                r_high = np.array(matplotlib.colors.to_rgb(t['cell_high']))

                if norm < 0.5:
                    frac = norm / 0.5
                    cell_color = r_low + (r_mid - r_low) * frac
                else:
                    frac = (norm - 0.5) / 0.5
                    cell_color = r_mid + (r_high - r_mid) * frac

                ax.fill_between([j + 0.02, j + 0.98], y_bot + 0.02, y_top - 0.02,
                                color=cell_color, alpha=0.6, zorder=1)

            # Value text
            if np.isnan(val):
                display = '\u2014'
            elif abs(val) >= 100:
                display = f'{val:.0f}'
            elif abs(val) >= 10:
                display = f'{val:.1f}'
            else:
                display = f'{val:.3f}'

            ax.text(cx, y_mid, display,
                    ha='center', va='center', fontsize=11,
                    color=t['text'], fontfamily=BODY_FONT, fontweight='bold',
                    zorder=4)

        # Vertical gridlines
        for j in range(n_teams + 1):
            ax.axvline(j, color=t['grid'], linewidth=0.5, alpha=0.3)

    # Footer
    fig.text(0.50, 0.01, '64analytics.com',
             fontsize=8, color=t['text_dk'], fontfamily=BODY_FONT, ha='center')

    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar_config():
    st.sidebar.image(str(BRAND_LOGO_DARK), width=80)
    st.sidebar.markdown('## Team Comparison')
    if st.sidebar.button('Reload data'):
        st.cache_data.clear()
        st.rerun()

    cfg = {}
    teams_df = load_teams()

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
    if cfg['conference'] != 'All':
        sport_teams = sport_teams[sport_teams['conference_name'] == cfg['conference']]

    years = ['2026', '2025', '2024', '2023', '2022']
    cfg['year'] = st.sidebar.selectbox('Year', years)

    cfg['teams_df'] = teams_df
    cfg['sport_teams'] = sport_teams

    # Chart settings
    st.sidebar.markdown('---')
    st.sidebar.markdown('### Chart')
    cfg['title'] = st.sidebar.text_input('Title', value='TEAM COMPARISON')
    cfg['logo_zoom'] = st.sidebar.slider('Logo size', 0.03, 0.15, 0.07, 0.01)

    return cfg


# ── Main ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='64 Analytics — Team Comparison', layout='wide',
                   initial_sidebar_state='expanded')

cfg = sidebar_config()

col1, col2 = st.columns([4, 1])
_use_light = col2.checkbox('Light theme', value=False, key='comp_light')
cfg['theme'] = 'Light' if _use_light else 'Dark'
theme = THEMES[cfg['theme']]

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, .stApp, .stApp *:not([class*="icon"]):not([data-testid*="icon"]):not(.material-icons):not(.material-symbols):not([data-testid="stDataFrameSortIcon"]) {{ font-family: 'Inter', sans-serif !important; }}
.stApp {{ background-color: {theme['streamlit_bg']}; }}
h1, h2, h3, p, label, .stMarkdown {{ color: {theme['text_sub']} !important; }}
</style>
""", unsafe_allow_html=True)

# ── Team selection ──
sport_teams = cfg['sport_teams']
team_options = sorted(sport_teams['team_name'].dropna().unique().tolist())
team_name_to_id = dict(zip(sport_teams['team_name'], sport_teams['team_db_id']))
team_id_to_name = dict(zip(sport_teams['team_db_id'], sport_teams['team_name']))
team_id_to_logo = dict(zip(sport_teams['team_db_id'], sport_teams['logo_id']))

st.markdown('### Select Teams')
selected_names = st.multiselect('Teams to compare', team_options,
                                help='Type to search. Select 2-10 teams.',
                                max_selections=10)
selected_ids = [team_name_to_id[n] for n in selected_names if n in team_name_to_id]

if len(selected_ids) < 2:
    st.info('Select at least 2 teams to compare.')
    st.stop()

# ── Metric selection ──
st.markdown('### Select Metrics')
catalog = build_metric_catalog()

# Quick presets
presets = {
    'Offense Overview': [
        ('Hitting', 'OPS', 'on_base_plus_slugging'),
        ('Hitting', 'wOBA', 'weighted_on_base_average'),
        ('Hitting', 'wRAA', 'weighted_runs_above_average'),
        ('Hitting', 'wRC+', 'weighted_runs_created_plus'),
        ('Hitting', 'ISO', 'isolated_power'),
        ('Hitting', 'K%', 'strikeout_percentage'),
        ('Hitting', 'BB%', 'walk_percentage'),
        ('Hitting', 'BABIP', 'batting_average_on_balls_in_play'),
        ('Hitting', 'R/PA', 'runs_plate_appearance'),
    ],
    'Pitching Overview': [
        ('Pitching', 'ERA', 'earned_run_average'),
        ('Pitching', 'FIP', 'fielding_independent_pitching'),
        ('Pitching', 'xFIP', 'expected_fielding_independent_pitching'),
        ('Pitching', 'SIERA', 'skill_interactive_earned_run_average'),
        ('Pitching', 'WHIP', 'walks_plus_hits_per_inning_pitched'),
        ('Pitching', 'K% (P)', 'strikeout_percentage'),
        ('Pitching', 'BB% (P)', 'walk_percentage'),
        ('Pitching', 'BAA', 'batting_average_against'),
        ('Pitching', 'HR/FB%', 'home_run_to_fly_ball_percentage'),
    ],
    'Full Overview': [
        ('64 Rank', '64 Rank (#)', 'integer_64_rank_total'),
        ('Hitting', 'OPS', 'on_base_plus_slugging'),
        ('Hitting', 'wOBA', 'weighted_on_base_average'),
        ('Hitting', 'wRAA', 'weighted_runs_above_average'),
        ('Hitting', 'ISO', 'isolated_power'),
        ('Pitching', 'ERA', 'earned_run_average'),
        ('Pitching', 'FIP', 'fielding_independent_pitching'),
        ('Pitching', 'WHIP', 'walks_plus_hits_per_inning_pitched'),
        ('Fielding', 'Fld%', 'fielding_percentage'),
    ],
}

preset_choice = st.radio('Quick preset', ['Custom'] + list(presets.keys()), horizontal=True)

if preset_choice != 'Custom':
    metrics_selected = presets[preset_choice]
else:
    metrics_selected = []
    for category, info in catalog.items():
        with st.expander(f'{category} ({len(info["metrics"])} metrics)'):
            metric_names = sorted(info['metrics'].keys())
            chosen = st.multiselect(f'{category} metrics', metric_names, key=f'cat_{category}')
            for name in chosen:
                col_name = info['metrics'][name]
                metrics_selected.append((category, name, col_name))

if len(metrics_selected) == 0:
    st.info('Select at least one metric or choose a preset.')
    st.stop()

# ── Build and render ──
matrix, metric_meta = build_matrix(selected_ids, metrics_selected, cfg['year'], catalog)

if matrix.empty:
    st.warning('No data found for the selected teams/metrics/year.')
    st.stop()

col1.metric('Teams', len(selected_ids))

with st.spinner('Rendering chart...'):
    fig = render_comparison(matrix, metric_meta, team_id_to_name, team_id_to_logo, cfg)

if fig is None:
    st.warning('Could not render chart.')
    st.stop()

buf = BytesIO()
fig.savefig(buf, format='png', dpi=180, bbox_inches='tight',
            facecolor=theme['bg'], edgecolor='none')
buf.seek(0)
plt.close(fig)

st.image(buf, use_container_width=True)

buf.seek(0)
st.download_button('Download PNG', data=buf, file_name='64analytics_comparison.png',
                   mime='image/png')

with st.expander('View data table'):
    display = matrix.copy()
    display.columns = [team_id_to_name.get(tid, tid) for tid in display.columns]
    st.dataframe(display, use_container_width=True)
