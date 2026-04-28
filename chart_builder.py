"""
64 Analytics — Chart Builder
Run:  streamlit run chart_builder.py
"""

import streamlit as st
import traceback

try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from PIL import Image, ImageDraw
    from pathlib import Path
    from io import BytesIO
    import os
    import base64
    import matplotlib.font_manager as fm
    from app_lib.filtered_team_stats import (
        derive_team_stats,
        merge_with_historical,
        is_active as _filter_is_active,
        GAME_TYPE_OPTIONS,
        DAY_OPTIONS,
    )
except Exception as e:
    st.error(f"Import failed: {e}")
    st.code(traceback.format_exc())
    st.stop()

# Files that support PBP-derived game-context filtering. The filter only
# overrides the 2026 rows; pre-2026 history stays from the on-disk CSV.
_FILTERABLE_CSVS = {
    'hitting_team.csv': 'hitting',
    'pitching_team.csv': 'pitching',
}

# ── Path setup (works locally and on Streamlit Cloud) ─────────────────────────
_APP_DIR = Path(__file__).resolve().parent
DATA_DIR   = _APP_DIR / 'data'
LOGO_DIR   = _APP_DIR / 'team_logos_512'
BRAND_LOGO_DARK  = _APP_DIR / 'assets' / 'brand_logo_dark.png'
BRAND_LOGO_LIGHT = _APP_DIR / 'assets' / 'brand_logo_light.png'

RED      = '#C41230'
RED_DK   = '#8B1A2A'
RED_LT   = '#E8455E'

# All 8 CWS (Omaha) teams by year (team names match teams.csv)
CWS_FINAL_FOUR = {
    '2021': ['Mississippi St.', 'Vanderbilt', 'NC State', 'Stanford', 'Texas', 'Tennessee', 'Arizona', 'Virginia'],
    '2022': ['Ole Miss', 'Oklahoma', 'Arkansas', 'Stanford', 'Texas A&M', 'Notre Dame', 'Auburn', 'Texas'],
    '2023': ['LSU', 'Wake Forest', 'Florida', 'Stanford', 'Virginia', 'TCU', 'Oral Roberts', 'Tennessee'],
    '2024': ['Tennessee', 'Texas A&M', 'Kentucky', 'North Carolina', 'Florida St.', 'Virginia', 'NC State', 'Florida'],
    '2025': ['Arkansas', 'LSU', 'Coastal Carolina', 'Louisville', 'Oregon St.', 'Arizona', 'UCLA', 'Murray St.'],
}

CWS_CHAMPIONS = {
    '2021': 'Mississippi St.',
    '2022': 'Ole Miss',
    '2023': 'LSU',
    '2024': 'Tennessee',
    '2025': 'LSU',
}

CWS_YEAR_COLORS = {
    '2021': {'dark': '#FF6B6B', 'light': '#CC3333'},   # coral / darker red
    '2022': {'dark': '#4ECDC4', 'light': '#2A9D8F'},   # teal / darker teal
    '2023': {'dark': '#FFD93D', 'light': '#CC8800'},   # gold / darker gold
    '2024': {'dark': '#6BCB77', 'light': '#2D8A4E'},   # green / darker green
    '2025': {'dark': '#4D96FF', 'light': '#2563EB'},   # blue / darker blue
}

THEMES = {
    'Dark': {
        'bg': '#1a1a1a', 'plot_bg': '#222222',
        'text': '#FFFFFF', 'text_sub': '#C8C8C8',
        'text_md': '#888888', 'text_dk': '#555555',
        'grid': '#2e2e2e', 'spine': '#3a3a3a',
        'avg_bg': '#1a1a1a',
        'q_tl': '#2a2233', 'q_tr': '#1a2a22', 'q_bl': '#2a2222', 'q_br': '#2a2a1a',
        'q_tl_text': '#9988bb', 'q_tr_text': '#66bb88',
        'q_bl_text': '#bb6666', 'q_br_text': '#bbbb66',
        'corr_bg': '#0e0e0e', 'callout_bg': '#1a1a1a',
        'brand_logo': BRAND_LOGO_DARK,
        'streamlit_bg': '#1a1a1a', 'streamlit_sidebar': '#111111',
        'streamlit_text': '#C8C8C8',
    },
    'Light': {
        'bg': '#FAF8F2', 'plot_bg': '#F2EFE5',
        'text': '#2D2926', 'text_sub': '#3F3A35',
        'text_md': '#4A4540', 'text_dk': '#6E6660',
        'grid': '#E2DCCC', 'spine': '#D6D0C0',
        'avg_bg': '#FAF8F2',
        'q_tl': '#E8E0F0', 'q_tr': '#D8F0E0', 'q_bl': '#F0D8D8', 'q_br': '#F0F0D8',
        'q_tl_text': '#6B5B95', 'q_tr_text': '#2E8B57',
        'q_bl_text': '#B22222', 'q_br_text': '#8B8B00',
        'corr_bg': '#F0EDE3', 'callout_bg': '#FAF8F2',
        'brand_logo': BRAND_LOGO_LIGHT,
        'streamlit_bg': '#FAF8F2', 'streamlit_sidebar': '#F2EFE5',
        'streamlit_text': '#2D2926',
    },
}

# Title font: bold condensed for impact (fall back on Linux/Cloud)
def _has_font(name):
    return any(name.lower() in f.name.lower() for f in fm.fontManager.ttflist)

TITLE_FONT    = 'Franklin Gothic Heavy'  if _has_font('Franklin Gothic') else 'DejaVu Sans'
SUBTITLE_FONT = 'Franklin Gothic Medium' if _has_font('Franklin Gothic') else 'DejaVu Sans'
BODY_FONT     = 'Calibri' if _has_font('Calibri') else 'DejaVu Sans'
AXIS_FONT     = 'Calibri' if _has_font('Calibri') else 'DejaVu Sans'

_EXTRA_SUBDIRS = ['Graphcis', 'rankings']
ALL_CSVS = sorted(
    [f.name for f in DATA_DIR.glob('*.csv')]
    + [str(f.relative_to(DATA_DIR)).replace('\\', '/')
       for sd in _EXTRA_SUBDIRS
       for f in (DATA_DIR / sd).glob('*.csv') if f.is_file()]
)

# Columns to exclude from axis dropdowns (identifiers, not plottable)
ID_COLS = {'id', 'key_id', 'team_id', 'Team_Id', 'player_id', 'year', 'Year',
           'sport', 'name', 'logo_url', 'conference_id', 'team_id_ncaa',
           'class_ranking', 'academics', 'in_portal', 'initiated_date',
           'portal_commit', 'current_rpi'}

# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
@st.cache_data
def _load_team_name_to_id():
    """Build sport-specific team_name -> team_id lookups.
    Returns (baseball_map, softball_map), each applying rankings/name_map.csv
    aliases so external sources like 'Miami FL' resolve to 'Miami (FL)'.
    Falls back across sports: if a name exists for baseball but not softball
    (or vice versa), the missing-sport map borrows the other sport's ID
    (better than nothing, but sport-correct IDs take precedence)."""
    teams = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False)
    teams['id'] = pd.to_numeric(teams['id'], errors='coerce').fillna(0).astype(int).astype(str)
    bb = teams[teams['sport'] == 'Baseball'][['name', 'id']].drop_duplicates('name')
    sb = teams[teams['sport'] == 'Softball'][['name', 'id']].drop_duplicates('name')
    bb_map = dict(zip(bb['name'], bb['id']))
    sb_map = dict(zip(sb['name'], sb['id']))

    # Apply external name aliases (rankings/name_map.csv) to both maps
    name_map_path = DATA_DIR / 'rankings' / 'name_map.csv'
    if name_map_path.exists():
        try:
            nm = pd.read_csv(name_map_path, low_memory=False)
            for _, row in nm.iterrows():
                ext = row.get('external_name')
                ours = row.get('our_name')
                if not ext or not ours:
                    continue
                if ours in bb_map:
                    bb_map[ext] = bb_map[ours]
                if ours in sb_map:
                    sb_map[ext] = sb_map[ours]
        except Exception:
            pass

    # Cross-fill: if a name only exists in one sport, let the other sport
    # borrow the ID (keeps non-NaN even for single-sport programs).
    for name, tid in bb_map.items():
        sb_map.setdefault(name, tid)
    for name, tid in sb_map.items():
        bb_map.setdefault(name, tid)

    return bb_map, sb_map


def _infer_sport_from_filename(filename):
    """Infer sport from the CSV filename. Returns 'baseball', 'softball', or None."""
    fn = str(filename).lower()
    if 'softball' in fn or '_sb_' in fn or '/sb/' in fn:
        return 'softball'
    if 'baseball' in fn or '_bb_' in fn or '/bb/' in fn:
        return 'baseball'
    return None


def _inject_team_id(df, filename=None):
    """If df has a team name column but no team_id, look up team_id from teams.csv.
    Uses the sport-specific lookup when the filename indicates a sport;
    otherwise defaults to baseball (preserves existing behavior for
    sport-agnostic files)."""
    if 'team_id' in df.columns:
        return df
    # Find a team-name-like column
    name_col = None
    for candidate in ['team_name', 'teamName', 'team', 'Team']:
        if candidate in df.columns:
            name_col = candidate
            break
    if name_col is None:
        return df
    try:
        bb_map, sb_map = _load_team_name_to_id()
    except Exception:
        return df
    sport = _infer_sport_from_filename(filename)
    name_to_id = sb_map if sport == 'softball' else bb_map
    df = df.copy()
    df['team_id'] = df[name_col].map(name_to_id)
    return df


def load_csv(filename):
    """Load a CSV and normalize column names.

    For team-level stat files (hitting_team.csv, pitching_team.csv): if the
    user has set a game-context filter (Conference/Non-Conf, Weekend/Midweek)
    via the sidebar, the 2026 rows are re-derived from PBP data with that
    filter applied; pre-2026 rows are preserved from disk.
    """
    df = pd.read_csv(DATA_DIR / filename, low_memory=False, encoding='utf-8-sig')
    # Strip BOM and whitespace from column names
    df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]
    if 'Team_Id' in df.columns:
        df = df.rename(columns={'Team_Id': 'team_id'})
    if 'Year' in df.columns and 'year' not in df.columns:
        df = df.rename(columns={'Year': 'year'})
    if 'year' in df.columns:
        df['year'] = df['year'].astype(str)
    # Inject team_id for CSVs that only have team names (rankings, RPI, etc.)
    df = _inject_team_id(df, filename)
    for col in ['team_id', 'player_id']:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors='coerce')
            df[col] = np.where(numeric.notna(), numeric.fillna(0).astype(int).astype(str), df[col])

    # ── Game-context filter override (2026 only, team-level stat files) ──
    if filename in _FILTERABLE_CSVS:
        game_type = st.session_state.get('_chart_game_type', 'All')
        day_filter = st.session_state.get('_chart_day_filter', 'All')
        sport = st.session_state.get('_chart_sport', 'Baseball')
        if _filter_is_active(game_type, day_filter):
            stat_kind = _FILTERABLE_CSVS[filename]
            derived = derive_team_stats(sport.lower(), stat_kind, game_type, day_filter)
            if not derived.empty:
                # Re-run team_id injection on derived rows so they match
                derived['year'] = derived['year'].astype(str)
                for col in ['team_id', 'player_id']:
                    if col in derived.columns:
                        numeric = pd.to_numeric(derived[col], errors='coerce')
                        derived[col] = np.where(numeric.notna(), numeric.fillna(0).astype(int).astype(str), derived[col])
                df = merge_with_historical(df, derived)
    return df


@st.cache_data
def load_teams():
    """Load teams + conferences for filtering."""
    teams = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False)
    confs = pd.read_csv(DATA_DIR / 'conferences.csv', low_memory=False)
    teams = teams.merge(confs[['id', 'name', 'abbreviation', 'division', 'classification']],
                        left_on='conference_id', right_on='id',
                        suffixes=('', '_conf'))
    teams = teams.rename(columns={'id': 'team_db_id', 'name': 'team_name',
                                  'name_conf': 'conference_name'})
    teams['team_db_id'] = pd.to_numeric(teams['team_db_id'], errors='coerce').fillna(0).astype(int).astype(str)

    # Build logo_id: logos are named after baseball team IDs, so map softball teams
    # to their baseball counterpart by school name
    bb = teams[teams['sport'] == 'Baseball'][['team_name', 'team_db_id']].drop_duplicates('team_name')
    bb_name_to_id = dict(zip(bb['team_name'], bb['team_db_id']))
    teams['logo_id'] = teams.apply(
        lambda r: r['team_db_id'] if r['sport'] == 'Baseball'
        else bb_name_to_id.get(r['team_name'], r['team_db_id']), axis=1)

    return teams


@st.cache_data
def load_players():
    """Load player name lookup: players.csv id -> player_name."""
    df = pd.read_csv(DATA_DIR / 'players.csv', low_memory=False, encoding='latin-1')
    df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype(int).astype(str)
    return df.set_index('id')['player_name'].to_dict()


@st.cache_data
def load_brand_logo(path=None):
    p = path or BRAND_LOGO_DARK
    return np.array(Image.open(p).convert('RGBA'))


@st.cache_data
def load_logo_thumbnail(logo_path, size=128):
    """Load a team logo resized to a thumbnail to save memory."""
    img = Image.open(logo_path).convert('RGBA')
    img.thumbnail((size, size), Image.LANCZOS)
    return (np.array(img) / 255.0).astype(np.float32)


@st.cache_data
def _get_plottable_columns_keyed(csv_name, game_type, day_filter, sport):
    """Cached lookup keyed on csv_name AND filter state. The filter args
    are part of the cache key so flipping Game Type / Day of Week busts
    the cache and returns the correct column list including derived stats
    (wOBA, wRC, wRAA, FIP, A-OPS, division percentile ranks)."""
    df = load_csv(csv_name)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in ID_COLS and c.lower() not in
            {x.lower() for x in ID_COLS}]


def get_plottable_columns(csv_name):
    """Return numeric columns suitable for plotting (filter-aware)."""
    game_type = st.session_state.get('_chart_game_type', 'All')
    day_filter = st.session_state.get('_chart_day_filter', 'All')
    sport = st.session_state.get('_chart_sport', 'Baseball')
    return _get_plottable_columns_keyed(csv_name, game_type, day_filter, sport)


def add_schedule_derived(df):
    """Add win_percentage to schedules data."""
    if 'Conf_Win' in df.columns:
        df = df.copy()
        wins = df['Conf_Win'].fillna(0) + df['OOC_Win'].fillna(0)
        total = (df['Conf_Win'].fillna(0) + df['Conf_Loss'].fillna(0) +
                 df['Conf_Tie'].fillna(0) + df['OOC_Win'].fillna(0) +
                 df['OOC_Loss'].fillna(0) + df['OOC_Tie'].fillna(0))
        df['win_percentage'] = np.where(total > 0, wins / total, np.nan)
        df['total_wins'] = wins
        df['total_losses'] = df['Conf_Loss'].fillna(0) + df['OOC_Loss'].fillna(0)
        df['total_games'] = total
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar():
    st.sidebar.image(str(BRAND_LOGO_DARK), width=80)
    st.sidebar.markdown('## Chart Builder')
    if st.sidebar.button('Reload data'):
        st.cache_data.clear()
        st.rerun()

    cfg = {}

    # ── Mode ──
    cfg['mode'] = st.sidebar.radio('Chart Mode', ['Team', 'Player'], horizontal=True)
    cfg['view_mode'] = st.sidebar.radio(
        'View',
        ['Static', 'Interactive'],
        horizontal=True,
        help='Static = matplotlib PNG (downloadable, Twitter-ready). '
             'Interactive = Plotly with hover tooltips on every dot. '
             'Interactive view is only available in Player mode.',
    )
    available_csvs = ALL_CSVS

    # ── Filters ──
    st.sidebar.markdown('---')
    st.sidebar.markdown('### Filters')
    teams_df = load_teams()

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

    strength_options = ['All'] + sorted(sport_teams['classification'].dropna().loc[lambda s: s != ''].unique().tolist())
    cfg['strength'] = st.sidebar.selectbox('Conference Strength', strength_options)

    if cfg['strength'] != 'All':
        sport_teams = sport_teams[sport_teams['classification'] == cfg['strength']]

    cfg['team_ids'] = set(sport_teams['team_db_id'].astype(str).tolist())

    year_list = ['2026', '2025', '2024', '2023', '2022', '2021']
    cfg['year'] = st.sidebar.selectbox('Year', year_list)

    # ── Game-context filter (PBP-derived; only applies to current year + team-level stat CSVs) ──
    cfg['game_type'] = st.sidebar.selectbox(
        'Game Type', GAME_TYPE_OPTIONS, key='_chart_game_type',
        help='Filters team stats to conference-only or non-conference-only games. '
             'Only applies to current-year team stats (hitting_team.csv / pitching_team.csv); '
             'historical years stay from the season aggregate.',
    )
    cfg['day_filter'] = st.sidebar.selectbox(
        'Day of Week', DAY_OPTIONS, key='_chart_day_filter',
        help='Weekend = Fri-Sun (typically conference series). Midweek = Mon-Thu (typically non-con singles). '
             'Same scope as Game Type filter.',
    )
    # Mirror sport into session_state so load_csv can read it
    st.session_state['_chart_sport'] = cfg['sport']
    if _filter_is_active(cfg['game_type'], cfg['day_filter']):
        st.sidebar.caption(f"⚠️ {cfg['year']} team stats are PBP-derived ({cfg['game_type']} · {cfg['day_filter']}). "
                          "Advanced metrics (wOBA percentiles, etc.) are blank under this filter.")

    cfg['portal_only'] = st.sidebar.checkbox('Portal transfers only', key='portal_only')
    if cfg['portal_only']:
        portal_year = str(int(cfg['year']) - 1)
        cfg['_portal_year_label'] = portal_year
        portal_df = load_csv('portal_rank_player.csv')
        portal_df = portal_df[portal_df['year'] == portal_year]
        cfg['portal_player_ids'] = set(portal_df['player_id'].astype(str).tolist())
        st.sidebar.caption(f'{len(cfg["portal_player_ids"])} players transferred in {portal_year}')
    else:
        cfg['portal_player_ids'] = None

    # ── X Axis ──
    st.sidebar.markdown('---')
    st.sidebar.markdown('### X Axis')
    x_default = available_csvs.index('hitting_team.csv') if 'hitting_team.csv' in available_csvs else 0
    cfg['x_csv'] = st.sidebar.selectbox('Data source (X)', available_csvs, index=x_default, key='x_csv')
    x_is_portal = 'portal' in cfg['x_csv'].lower()
    if x_is_portal:
        x_yr_default = str(int(cfg['year']) - 1)
        cfg['x_year'] = st.sidebar.selectbox('Year (X)', year_list, index=year_list.index(x_yr_default) if x_yr_default in year_list else 0, key='x_year')
        st.sidebar.caption(f'Portal data: showing {cfg["x_year"]} transfer class')
    else:
        cfg['x_year'] = cfg['year']
    x_cols = get_plottable_columns(cfg['x_csv'])
    if cfg['x_csv'] == 'schedules.csv':
        x_cols = ['win_percentage', 'total_wins', 'total_losses', 'total_games'] + x_cols
    if not x_cols:
        st.sidebar.warning(f'No numeric columns in {cfg["x_csv"]}')
        x_cols = ['(none)']
    cfg['x_cols'] = st.sidebar.multiselect('Columns (X)', x_cols, default=[x_cols[0]], key='x_cols')
    if not cfg['x_cols']:
        st.sidebar.warning('Select at least one column for X axis')
        cfg['x_cols'] = [x_cols[0]]

    # Cross-CSV: add columns from additional CSVs to this axis
    cfg['x_extra_sources'] = []
    x_cross = st.sidebar.checkbox('Add columns from another CSV (X)', key='x_cross')
    if x_cross:
        x_extra_csv = st.sidebar.selectbox('Additional CSV (X)', available_csvs, key='x_extra_csv')
        x_extra_cols_avail = get_plottable_columns(x_extra_csv)
        if x_extra_csv == 'schedules.csv':
            x_extra_cols_avail = ['win_percentage', 'total_wins', 'total_losses', 'total_games'] + x_extra_cols_avail
        x_extra_cols = st.sidebar.multiselect('Additional columns (X)', x_extra_cols_avail, key='x_extra_cols')
        if x_extra_cols:
            cfg['x_extra_sources'].append({'csv': x_extra_csv, 'cols': x_extra_cols})

    all_x_col_names = list(cfg['x_cols']) + [c for src in cfg['x_extra_sources'] for c in src['cols']]

    if len(all_x_col_names) > 1:
        cfg['x_combine'] = st.sidebar.selectbox('Combine method (X)', ['Sum', 'Average'], key='x_combine')
        cfg['x_col'] = '_x_combined'
        if cfg['x_combine'] == 'Sum':
            x_label_default = ' + '.join(c.replace('_', ' ').title() for c in all_x_col_names)
        else:
            x_label_default = f"Avg({', '.join(c.replace('_', ' ').title() for c in all_x_col_names)})"
    else:
        cfg['x_combine'] = None
        cfg['x_col'] = cfg['x_cols'][0]
        x_label_default = cfg['x_cols'][0].replace('_', ' ').title()
    cfg['x_label'] = st.sidebar.text_input('Label (X)', value=x_label_default, key='x_label')
    cfg['x_direction'] = st.sidebar.radio('X direction', ['Higher is better', 'Lower is better'],
                                          horizontal=True, key='x_dir')

    if cfg['mode'] == 'Team':
        with st.sidebar.expander('X Subset / Aggregation'):
            cfg['x_subset'] = st.sidebar.checkbox('Enable subset (X)', key='x_sub')
            if cfg['x_subset']:
                # For team CSVs, use the corresponding player CSV for ranking
                team_to_player = {
                    'hitting_team.csv': 'hitting.csv',
                    'pitching_team.csv': 'pitching.csv',
                    'fielding_team.csv': 'fielding.csv',
                }
                x_subset_csv = team_to_player.get(cfg['x_csv'], cfg['x_csv'])
                cfg['x_subset_csv'] = x_subset_csv
                if x_subset_csv != cfg['x_csv']:
                    st.sidebar.caption(f'Ranking players from {x_subset_csv}')
                x_all_cols = get_plottable_columns(x_subset_csv)
                if x_subset_csv == 'schedules.csv':
                    x_all_cols = ['win_percentage', 'total_wins', 'total_losses', 'total_games'] + x_all_cols
                cfg['x_rank_col'] = st.sidebar.selectbox('Rank players by (X)',
                                                         x_all_cols, key='x_rank')
                # Plot column: pick from the player CSV too
                x_plot_cols = get_plottable_columns(x_subset_csv)
                x_plot_default = x_plot_cols.index(cfg['x_cols'][0]) if cfg['x_cols'][0] in x_plot_cols else 0
                cfg['x_subset_plot_col'] = st.sidebar.selectbox('Aggregate column (X)',
                                                                 x_plot_cols, index=x_plot_default,
                                                                 key='x_sub_plot')
                c1, c2 = st.sidebar.columns(2)
                cfg['x_range_start'] = c1.number_input('From rank', 1, 50, 1, key='x_rs')
                cfg['x_range_end'] = c2.number_input('To rank', 1, 50, 5, key='x_re')
                cfg['x_agg'] = st.sidebar.selectbox('Aggregate', ['mean', 'sum', 'median'],
                                                    key='x_agg')
            else:
                cfg['x_subset_csv'] = cfg['x_csv']
                cfg['x_subset_plot_col'] = None
                cfg['x_rank_col'] = cfg['x_range_start'] = cfg['x_range_end'] = cfg['x_agg'] = None
    else:
        cfg['x_subset'] = False
        cfg['x_subset_csv'] = cfg['x_csv']
        cfg['x_subset_plot_col'] = None
        cfg['x_rank_col'] = cfg['x_range_start'] = cfg['x_range_end'] = cfg['x_agg'] = None

    # ── Y Axis ──
    st.sidebar.markdown('---')
    st.sidebar.markdown('### Y Axis')
    y_default = available_csvs.index('schedules.csv') if 'schedules.csv' in available_csvs else 0
    cfg['y_csv'] = st.sidebar.selectbox('Data source (Y)', available_csvs, index=y_default, key='y_csv')
    y_is_portal = 'portal' in cfg['y_csv'].lower()
    if y_is_portal:
        y_yr_default = str(int(cfg['year']) - 1)
        cfg['y_year'] = st.sidebar.selectbox('Year (Y)', year_list, index=year_list.index(y_yr_default) if y_yr_default in year_list else 0, key='y_year')
        st.sidebar.caption(f'Portal data: showing {cfg["y_year"]} transfer class')
    else:
        cfg['y_year'] = cfg['year']
    y_cols = get_plottable_columns(cfg['y_csv'])
    if cfg['y_csv'] == 'schedules.csv':
        y_cols = ['win_percentage', 'total_wins', 'total_losses', 'total_games'] + y_cols
    if not y_cols:
        st.sidebar.warning(f'No numeric columns in {cfg["y_csv"]}')
        y_cols = ['(none)']
    cfg['y_cols'] = st.sidebar.multiselect('Columns (Y)', y_cols, default=[y_cols[0]], key='y_cols')
    if not cfg['y_cols']:
        st.sidebar.warning('Select at least one column for Y axis')
        cfg['y_cols'] = [y_cols[0]]

    # Cross-CSV: add columns from additional CSVs to this axis
    cfg['y_extra_sources'] = []
    y_cross = st.sidebar.checkbox('Add columns from another CSV (Y)', key='y_cross')
    if y_cross:
        y_extra_csv = st.sidebar.selectbox('Additional CSV (Y)', available_csvs, key='y_extra_csv')
        y_extra_cols_avail = get_plottable_columns(y_extra_csv)
        if y_extra_csv == 'schedules.csv':
            y_extra_cols_avail = ['win_percentage', 'total_wins', 'total_losses', 'total_games'] + y_extra_cols_avail
        y_extra_cols = st.sidebar.multiselect('Additional columns (Y)', y_extra_cols_avail, key='y_extra_cols')
        if y_extra_cols:
            cfg['y_extra_sources'].append({'csv': y_extra_csv, 'cols': y_extra_cols})

    all_y_col_names = list(cfg['y_cols']) + [c for src in cfg['y_extra_sources'] for c in src['cols']]

    if len(all_y_col_names) > 1:
        cfg['y_combine'] = st.sidebar.selectbox('Combine method (Y)', ['Sum', 'Average'], key='y_combine')
        cfg['y_col'] = '_y_combined'
        if cfg['y_combine'] == 'Sum':
            y_label_default = ' + '.join(c.replace('_', ' ').title() for c in all_y_col_names)
        else:
            y_label_default = f"Avg({', '.join(c.replace('_', ' ').title() for c in all_y_col_names)})"
    else:
        cfg['y_combine'] = None
        cfg['y_col'] = cfg['y_cols'][0]
        y_label_default = cfg['y_cols'][0].replace('_', ' ').title()
    cfg['y_label'] = st.sidebar.text_input('Label (Y)', value=y_label_default, key='y_label')
    cfg['y_direction'] = st.sidebar.radio('Y direction', ['Higher is better', 'Lower is better'],
                                          horizontal=True, key='y_dir')

    if cfg['mode'] == 'Team':
        with st.sidebar.expander('Y Subset / Aggregation'):
            cfg['y_subset'] = st.sidebar.checkbox('Enable subset (Y)', key='y_sub')
            if cfg['y_subset']:
                team_to_player = {
                    'hitting_team.csv': 'hitting.csv',
                    'pitching_team.csv': 'pitching.csv',
                    'fielding_team.csv': 'fielding.csv',
                }
                y_subset_csv = team_to_player.get(cfg['y_csv'], cfg['y_csv'])
                cfg['y_subset_csv'] = y_subset_csv
                if y_subset_csv != cfg['y_csv']:
                    st.sidebar.caption(f'Ranking players from {y_subset_csv}')
                y_all_cols = get_plottable_columns(y_subset_csv)
                if y_subset_csv == 'schedules.csv':
                    y_all_cols = ['win_percentage', 'total_wins', 'total_losses', 'total_games'] + y_all_cols
                cfg['y_rank_col'] = st.sidebar.selectbox('Rank players by (Y)',
                                                         y_all_cols, key='y_rank')
                y_plot_cols = get_plottable_columns(y_subset_csv)
                y_plot_default = y_plot_cols.index(cfg['y_cols'][0]) if cfg['y_cols'][0] in y_plot_cols else 0
                cfg['y_subset_plot_col'] = st.sidebar.selectbox('Aggregate column (Y)',
                                                                 y_plot_cols, index=y_plot_default,
                                                                 key='y_sub_plot')
                c1, c2 = st.sidebar.columns(2)
                cfg['y_range_start'] = c1.number_input('From rank', 1, 50, 1, key='y_rs')
                cfg['y_range_end'] = c2.number_input('To rank', 1, 50, 5, key='y_re')
                cfg['y_agg'] = st.sidebar.selectbox('Aggregate', ['mean', 'sum', 'median'],
                                                    key='y_agg')
            else:
                cfg['y_subset_csv'] = cfg['y_csv']
                cfg['y_subset_plot_col'] = None
                cfg['y_rank_col'] = cfg['y_range_start'] = cfg['y_range_end'] = cfg['y_agg'] = None
    else:
        cfg['y_subset'] = False
        cfg['y_subset_csv'] = cfg['y_csv']
        cfg['y_subset_plot_col'] = None
        cfg['y_rank_col'] = cfg['y_range_start'] = cfg['y_range_end'] = cfg['y_agg'] = None

    # ── Data Filters ──
    st.sidebar.markdown('---')
    st.sidebar.markdown('### Data Filters')
    if cfg['mode'] == 'Player':
        st.sidebar.caption('Stack multiple filters (e.g. PA >= 50 AND K% <= 0.30)')
    else:
        st.sidebar.caption('Filter teams by stat thresholds (e.g. HR >= 20)')
    num_filters = st.sidebar.number_input('Number of filters', 0, 5, 0, key='pf_count')
    cfg['player_filters'] = []
    for i in range(int(num_filters)):
        with st.sidebar.expander(f'Filter {i+1}', expanded=True):
            pf_csv = st.selectbox('CSV', available_csvs, key=f'pf_csv_{i}')
            pf_cols = get_plottable_columns(pf_csv)
            if pf_csv == 'schedules.csv':
                pf_cols = ['win_percentage', 'total_wins', 'total_losses', 'total_games'] + pf_cols
            pf_col = st.selectbox('Column', pf_cols, key=f'pf_col_{i}')
            if cfg['mode'] == 'Player':
                pf_modes = ['Minimum', 'Maximum', 'Top N per team']
            else:
                pf_modes = ['Minimum', 'Maximum']
            pf_mode = st.radio('Type', pf_modes, horizontal=True, key=f'pf_mode_{i}')
            pf_entry = {'csv': pf_csv, 'col': pf_col, 'mode': pf_mode}
            if pf_mode == 'Top N per team':
                pf_entry['top_n'] = st.number_input('Top N', 1, 100, 9, key=f'pf_topn_{i}')
            elif pf_mode == 'Minimum':
                pf_entry['min'] = st.number_input('Min value', value=0.0, step=1.0, key=f'pf_min_{i}')
            else:
                pf_entry['max'] = st.number_input('Max value', value=100.0, step=1.0, key=f'pf_max_{i}')
            cfg['player_filters'].append(pf_entry)

    # ── Rank Weighting ──
    st.sidebar.markdown('---')
    st.sidebar.markdown('### Rank Weighting (Logo Size)')
    st.sidebar.caption('Higher-ranked teams/players get bigger logos')
    cfg['rank_enabled'] = st.sidebar.checkbox('Enable rank weighting')
    if cfg['rank_enabled']:
        # Source selection
        if cfg['mode'] == 'Team':
            rank_sources = {'64 Rank': 'team_rank.csv', 'SOS': 'Graphcis/SOS.csv'}
            rank_source = st.sidebar.radio('Rank source', list(rank_sources.keys()),
                                            horizontal=True, key='rank_source')
            cfg['rank_csv'] = rank_sources[rank_source]
        else:
            rank_source = '64 Rank'
            cfg['rank_csv'] = 'player_rank.csv'

        rank_cols = get_plottable_columns(cfg['rank_csv'])

        if rank_source == 'SOS':
            # SOS.csv columns: rank, SOS (percentile), sosRating
            default_rank = 'SOS'
            rank_idx = rank_cols.index(default_rank) if default_rank in rank_cols else 0
        elif cfg['mode'] == 'Team':
            default_rank = '64_rank_total'
            rank_idx = rank_cols.index(default_rank) if default_rank in rank_cols else 0
        else:
            default_rank = 'percentile_rank_weighted_run_created_efficiency'
            rank_idx = rank_cols.index(default_rank) if default_rank in rank_cols else 0

        cfg['rank_col'] = st.sidebar.selectbox('Rank by', rank_cols, index=rank_idx, key='rank_col',
                                               help='Column used to size logos. Higher value = bigger logo.')
        cfg['rank_invert'] = st.sidebar.checkbox('Invert (lower value = better rank)',
                                                 help='Check this if the rank column uses 1=best (like integer ranks). '
                                                      'Leave unchecked for percentiles where higher=better.',
                                                 key='rank_inv')
        cfg['rank_intensity'] = st.sidebar.slider('Size contrast', 1, 10, 3,
                                                  help='1 = subtle, 5 = noticeable, 10 = extreme',
                                                  key='rank_int')
        # Wider zoom ranges for more dramatic size differences
        base_zoom = 0.055  # matches default logo size slider
        spread = cfg['rank_intensity'] * 0.008
        cfg['zoom_min'] = max(0.01, base_zoom - spread)
        cfg['zoom_max'] = base_zoom + spread
    else:
        cfg['rank_csv'] = cfg['rank_col'] = None
        cfg['rank_invert'] = False
        cfg['zoom_min'] = cfg['zoom_max'] = None

    # ── Bubble Scale (size by third stat) ──
    st.sidebar.markdown('---')
    st.sidebar.markdown('### Bubble Scale (Third Stat)')
    st.sidebar.caption('Size dots / logos by a third stat. Useful for spotting '
                       'context — e.g. ERA vs WHIP scatter, sized by opponent ISO.')
    cfg['bubble_enabled'] = st.sidebar.checkbox('Enable bubble scale', key='bubble_en')
    if cfg['bubble_enabled']:
        bubble_csv_default = available_csvs.index(cfg['x_csv']) if cfg['x_csv'] in available_csvs else 0
        cfg['bubble_csv'] = st.sidebar.selectbox('Source CSV (bubble)', available_csvs,
                                                  index=bubble_csv_default, key='bubble_csv')
        bubble_cols = get_plottable_columns(cfg['bubble_csv'])
        if not bubble_cols:
            st.sidebar.warning(f'No numeric columns in {cfg["bubble_csv"]}')
            cfg['bubble_col'] = None
        else:
            cfg['bubble_col'] = st.sidebar.selectbox('Bubble stat', bubble_cols, key='bubble_col_sel')
        cfg['bubble_invert'] = st.sidebar.checkbox(
            'Lower value = bigger bubble',
            help='Check this if a LOW value is "worse" and you want it shown as a BIG bubble '
                 '(e.g. opponent ISO — high means hit hard, so leave UNCHECKED to make hit-hard '
                 'pitchers stand out as big bubbles).',
            key='bubble_inv')
        cfg['bubble_intensity'] = st.sidebar.slider(
            'Size contrast', 1, 10, 4,
            help='1 = subtle, 10 = extreme', key='bubble_int')
        if cfg['rank_enabled']:
            st.sidebar.caption('\u26A0\uFE0F Rank weighting and bubble scale both control size '
                               '— bubble scale takes priority when enabled.')
    else:
        cfg['bubble_csv'] = cfg['bubble_col'] = None
        cfg['bubble_invert'] = False
        cfg['bubble_intensity'] = 4

    # ── Correlation ──
    st.sidebar.markdown('---')
    st.sidebar.markdown('### Correlation')
    cfg['show_correlation'] = st.sidebar.checkbox('Show trend line & Pearson r')
    if cfg['show_correlation']:
        cfg['show_prediction_band'] = st.sidebar.checkbox('Show prediction band', value=True, key='pred_band')
        cfg['prediction_band_sd'] = st.sidebar.slider('Band width (SD)', 1, 3, 1, key='pred_sd',
                                                       help='Number of standard deviations from the trend line')
    else:
        cfg['show_prediction_band'] = False
        cfg['prediction_band_sd'] = 1

    # ── Chart Style ──
    st.sidebar.markdown('---')
    st.sidebar.markdown('### Chart Style')
    cfg['theme'] = 'Dark'  # default, overridden in main()
    cfg['logo_zoom'] = st.sidebar.slider('Default logo size', 0.02, 0.10, 0.055, 0.005)
    cfg['show_quadrants'] = st.sidebar.checkbox('Show quadrants', value=True)
    if cfg['show_quadrants']:
        c1, c2 = st.sidebar.columns(2)
        cfg['q_tl'] = c1.text_input('Top-Left label', 'DEPTH-CARRIED', key='qtl')
        cfg['q_tr'] = c2.text_input('Top-Right label', 'ELITE', key='qtr')
        cfg['q_bl'] = c1.text_input('Bottom-Left label', 'STRUGGLING', key='qbl')
        cfg['q_br'] = c2.text_input('Bottom-Right label', 'TOP-HEAVY', key='qbr')
        cfg['custom_q_colors'] = st.sidebar.checkbox('Custom quadrant colors', key='custom_qc')
        if cfg['custom_q_colors']:
            c1, c2 = st.sidebar.columns(2)
            cfg['q_tl_color'] = c1.color_picker('Top-Left', '#2a2233', key='qc_tl')
            cfg['q_tr_color'] = c2.color_picker('Top-Right', '#1a2a22', key='qc_tr')
            cfg['q_bl_color'] = c1.color_picker('Bottom-Left', '#2a2222', key='qc_bl')
            cfg['q_br_color'] = c2.color_picker('Bottom-Right', '#2a2a1a', key='qc_br')

    # ── CWS Overlay ──
    if cfg['mode'] == 'Team' and cfg['sport'] == 'Baseball':
        st.sidebar.markdown('---')
        st.sidebar.markdown('### CWS Teams Overlay')
        cfg['cws_enabled'] = st.sidebar.checkbox('Show CWS Teams', key='cws_on')
        if cfg['cws_enabled']:
            cws_years = sorted(CWS_FINAL_FOUR.keys(), reverse=True)
            cfg['cws_years'] = st.sidebar.multiselect('Years', cws_years, default=cws_years, key='cws_yrs')
            cfg['cws_ring_size'] = st.sidebar.slider('Ring size', 20, 80, 45, 5, key='cws_ring')
            cfg['cws_ring_width'] = st.sidebar.slider('Ring thickness', 1.0, 5.0, 2.5, 0.5, key='cws_rw')
        else:
            cfg['cws_years'] = []
            cfg['cws_ring_size'] = 45
            cfg['cws_ring_width'] = 2.5
    else:
        cfg['cws_enabled'] = False
        cfg['cws_years'] = []

    # ── Player Marker Style ──
    if cfg['mode'] == 'Player':
        st.sidebar.markdown('---')
        st.sidebar.markdown('### Player Markers')
        marker_shapes = {'Circle': 'o', 'Diamond': 'D', 'Square': 's',
                         'Triangle Up': '^', 'Triangle Down': 'v',
                         'Star': '*', 'Pentagon': 'p', 'Hexagon': 'h', 'Plus': 'P', 'X': 'X'}
        shape_name = st.sidebar.selectbox('Shape', list(marker_shapes.keys()), key='p_shape')
        cfg['player_marker'] = marker_shapes[shape_name]
        cfg['player_size'] = st.sidebar.slider('Size', 2, 20, 6, key='p_size')
        cfg['player_alpha'] = st.sidebar.slider('Opacity', 0.1, 1.0, 0.5, 0.05, key='p_alpha')

        color_presets = {'64 Red': '#C41230', 'White': '#FFFFFF', 'Sky Blue': '#4DA6FF',
                         'Gold': '#FFD700', 'Lime': '#32CD32', 'Orange': '#FF8C00',
                         'Hot Pink': '#FF69B4', 'Cyan': '#00CED1', 'Custom': 'custom'}
        color_name = st.sidebar.selectbox('Color', list(color_presets.keys()), key='p_color')
        if color_name == 'Custom':
            cfg['player_color'] = st.sidebar.color_picker('Pick color', '#C41230', key='p_cpick')
        else:
            cfg['player_color'] = color_presets[color_name]

        cfg['player_edge'] = st.sidebar.checkbox('Show edge', value=False, key='p_edge')
        if cfg['player_edge']:
            edge_color_name = st.sidebar.selectbox('Edge color',
                                                    list(color_presets.keys()), index=1, key='p_ecolor')
            if edge_color_name == 'Custom':
                cfg['player_edge_color'] = st.sidebar.color_picker('Pick edge color', '#FFFFFF', key='p_ecpick')
            else:
                cfg['player_edge_color'] = color_presets[edge_color_name]
            cfg['player_edge_width'] = st.sidebar.slider('Edge width', 0.5, 3.0, 1.0, 0.5, key='p_ew')
        else:
            cfg['player_edge_color'] = 'none'
            cfg['player_edge_width'] = 0
    else:
        cfg['player_marker'] = 'o'
        cfg['player_size'] = 6
        cfg['player_alpha'] = 0.5
        cfg['player_color'] = RED
        cfg['player_edge_color'] = 'none'
        cfg['player_edge_width'] = 0
        cfg['player_edge'] = False

    cfg['title'] = st.sidebar.text_input('Chart title',
        f'{cfg["division"] if cfg["division"] != "All" else ""} {cfg["sport"].upper()} {cfg["year"]}')
    cfg['subtitle'] = st.sidebar.text_input('Subtitle',
        f'{cfg["x_label"].upper()} vs {cfg["y_label"].upper()}')

    # ── Annotations ──
    st.sidebar.markdown('---')
    st.sidebar.markdown('### Annotations')
    st.sidebar.caption('Add custom callout labels with arrows on the chart')
    num_annotations = st.sidebar.number_input('Number of annotations', 0, 10, 0, key='ann_count')
    cfg['annotations'] = []
    for i in range(int(num_annotations)):
        with st.sidebar.expander(f'Annotation {i+1}', expanded=True):
            text = st.text_input('Text', key=f'ann_text_{i}')
            c1, c2 = st.columns(2)
            ax_val = c1.number_input('Point X', value=0.0, step=0.01, format='%.3f', key=f'ann_x_{i}')
            ay_val = c2.number_input('Point Y', value=0.0, step=0.01, format='%.3f', key=f'ann_y_{i}')
            c3, c4 = st.columns(2)
            off_x = c3.number_input('Label offset X', value=40, step=5, key=f'ann_ox_{i}')
            off_y = c4.number_input('Label offset Y', value=40, step=5, key=f'ann_oy_{i}')
            if text:
                cfg['annotations'].append({
                    'text': text, 'x': ax_val, 'y': ay_val,
                    'off_x': off_x, 'off_y': off_y,
                })

    return cfg


# ── Data Pipeline ─────────────────────────────────────────────────────────────
def load_and_filter(csv_name, year, team_ids):
    """Load CSV, filter by year and team_ids."""
    df = load_csv(csv_name)
    if csv_name == 'schedules.csv':
        df = add_schedule_derived(df)
    if 'year' in df.columns:
        df = df[df['year'] == str(year)]
    # Portal CSVs: use new_team_id (destination team) for filtering/joining
    if 'new_team_id' in df.columns:
        df = df.rename(columns={'team_id': '_from_team_id', 'new_team_id': 'team_id'})
        df['team_id'] = pd.to_numeric(df['team_id'], errors='coerce')
        df['team_id'] = np.where(df['team_id'].notna(),
                                  df['team_id'].fillna(0).astype(int).astype(str), df['team_id'])
    if 'team_id' in df.columns:
        df = df[df['team_id'].astype(str).isin(team_ids)]
    return df


def apply_subset(df, plot_col, rank_col, range_start, range_end, agg_func):
    """Group by team, rank within team, slice, aggregate."""
    if 'team_id' not in df.columns:
        return df

    df = df.copy()
    df[plot_col] = pd.to_numeric(df[plot_col], errors='coerce')
    df[rank_col] = pd.to_numeric(df[rank_col], errors='coerce')

    results = []
    for tid, group in df.groupby('team_id'):
        sorted_g = group.sort_values(rank_col, ascending=False)
        sliced = sorted_g.iloc[range_start - 1 : range_end]
        if len(sliced) < (range_end - range_start + 1):
            continue
        val = sliced[plot_col].agg(agg_func)
        results.append({'team_id': tid, plot_col: val})

    return pd.DataFrame(results)


def build_cws_overlay(cfg):
    """Build CWS Teams data points from their respective years."""
    teams_df = load_teams()
    bb_teams = teams_df[teams_df['sport'] == 'Baseball']
    name_to_id = dict(zip(bb_teams['team_name'], bb_teams['team_db_id']))

    all_rows = []
    for cws_year in cfg.get('cws_years', []):
        cws_team_names = CWS_FINAL_FOUR.get(cws_year, [])
        champion = CWS_CHAMPIONS.get(cws_year, '')
        cws_team_ids = {name_to_id[n] for n in cws_team_names if n in name_to_id}
        if not cws_team_ids:
            continue

        # Build a modified cfg for this CWS year
        cws_cfg = dict(cfg)
        cws_cfg['year'] = cws_year
        cws_cfg['x_year'] = cws_year
        cws_cfg['y_year'] = cws_year
        cws_cfg['team_ids'] = cws_team_ids
        cws_cfg['portal_player_ids'] = None
        cws_cfg['portal_only'] = False
        cws_cfg['player_filters'] = []
        # Disable subsets for CWS — just use raw team-level data
        cws_cfg['x_subset'] = False
        cws_cfg['y_subset'] = False
        cws_cfg['x_rank_col'] = None
        cws_cfg['y_rank_col'] = None
        cws_cfg['x_subset_csv'] = cws_cfg['x_csv']
        cws_cfg['y_subset_csv'] = cws_cfg['y_csv']
        cws_cfg['x_subset_plot_col'] = None
        cws_cfg['y_subset_plot_col'] = None
        # Disable CWS overlay recursion
        cws_cfg['cws_enabled'] = False

        try:
            cws_data = build_data(cws_cfg)
            import streamlit as _st
            expected = len(cws_team_names)
            got = len(cws_data)
            if got < expected:
                found_names = set(cws_data['team_name'].tolist()) if 'team_name' in cws_data.columns else set()
                all_names = set(cws_team_names)
                missing = all_names - found_names
                _st.caption(f'CWS {cws_year}: {got}/{expected} teams loaded. Missing: {missing}')
            if len(cws_data) == 0:
                continue
            cws_data['_cws_year'] = cws_year
            cws_data['_is_champion'] = cws_data['team_name'].apply(lambda n: n == champion)
            all_rows.append(cws_data)
        except Exception as e:
            import streamlit as _st
            _st.warning(f'CWS {cws_year} overlay error: {e}')
            continue

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    return pd.DataFrame()


def build_data(cfg):
    """Build the final dataset for plotting."""
    teams_df = load_teams()
    team_ids = cfg['team_ids']

    # ── X axis data ──
    x_df = load_and_filter(cfg['x_csv'], cfg.get('x_year', cfg['year']), team_ids)

    # Merge in cross-CSV columns for X axis
    for extra in cfg.get('x_extra_sources', []):
        extra_df = load_and_filter(extra['csv'], cfg.get('x_year', cfg['year']), team_ids)
        for col in extra['cols']:
            if col in extra_df.columns:
                extra_df[col] = pd.to_numeric(extra_df[col], errors='coerce')
        merge_cols = ['team_id'] + [c for c in extra['cols'] if c in extra_df.columns]
        extra_key = extra_df[merge_cols].drop_duplicates('team_id')
        x_df = x_df.merge(extra_key, on='team_id', how='left', suffixes=('', '_extra'))

    all_x_col_names = list(cfg.get('x_cols', [])) + [c for src in cfg.get('x_extra_sources', []) for c in src['cols']]

    if len(all_x_col_names) > 1:
        for col in all_x_col_names:
            if col in x_df.columns:
                x_df[col] = pd.to_numeric(x_df[col], errors='coerce')
        valid = [c for c in all_x_col_names if c in x_df.columns]
        if cfg.get('x_combine') == 'Average':
            x_df['_x_combined'] = x_df[valid].mean(axis=1)
        else:
            x_df['_x_combined'] = x_df[valid].sum(axis=1)
    elif len(cfg.get('x_cols', [])) > 1:
        for col in cfg['x_cols']:
            if col in x_df.columns:
                x_df[col] = pd.to_numeric(x_df[col], errors='coerce')
        valid = [c for c in cfg['x_cols'] if c in x_df.columns]
        if cfg.get('x_combine') == 'Average':
            x_df['_x_combined'] = x_df[valid].mean(axis=1)
        else:
            x_df['_x_combined'] = x_df[valid].sum(axis=1)
    if cfg['x_subset'] and cfg['x_rank_col']:
        # Load player-level CSV for subsetting (may differ from team-level axis CSV)
        x_subset_csv = cfg.get('x_subset_csv', cfg['x_csv'])
        x_subset_df = load_and_filter(x_subset_csv, cfg.get('x_year', cfg['year']), team_ids)
        x_plot_col = cfg.get('x_subset_plot_col') or cfg['x_col']
        x_df = apply_subset(x_subset_df, x_plot_col, cfg['x_rank_col'],
                            cfg['x_range_start'], cfg['x_range_end'], cfg['x_agg'])
        # Rename the aggregated column to match expected x_col
        if x_plot_col != cfg['x_col'] and x_plot_col in x_df.columns:
            x_df = x_df.rename(columns={x_plot_col: cfg['x_col']})
    else:
        if cfg['x_col'] in x_df.columns:
            x_df[cfg['x_col']] = pd.to_numeric(x_df[cfg['x_col']], errors='coerce')

    # ── Y axis data ──
    y_df = load_and_filter(cfg['y_csv'], cfg.get('y_year', cfg['year']), team_ids)

    # Merge in cross-CSV columns for Y axis
    for extra in cfg.get('y_extra_sources', []):
        extra_df = load_and_filter(extra['csv'], cfg.get('y_year', cfg['year']), team_ids)
        for col in extra['cols']:
            if col in extra_df.columns:
                extra_df[col] = pd.to_numeric(extra_df[col], errors='coerce')
        merge_cols = ['team_id'] + [c for c in extra['cols'] if c in extra_df.columns]
        extra_key = extra_df[merge_cols].drop_duplicates('team_id')
        y_df = y_df.merge(extra_key, on='team_id', how='left', suffixes=('', '_extra'))

    all_y_col_names = list(cfg.get('y_cols', [])) + [c for src in cfg.get('y_extra_sources', []) for c in src['cols']]

    if len(all_y_col_names) > 1:
        for col in all_y_col_names:
            if col in y_df.columns:
                y_df[col] = pd.to_numeric(y_df[col], errors='coerce')
        valid = [c for c in all_y_col_names if c in y_df.columns]
        if cfg.get('y_combine') == 'Average':
            y_df['_y_combined'] = y_df[valid].mean(axis=1)
        else:
            y_df['_y_combined'] = y_df[valid].sum(axis=1)
    elif len(cfg.get('y_cols', [])) > 1:
        for col in cfg['y_cols']:
            if col in y_df.columns:
                y_df[col] = pd.to_numeric(y_df[col], errors='coerce')
        valid = [c for c in cfg['y_cols'] if c in y_df.columns]
        if cfg.get('y_combine') == 'Average':
            y_df['_y_combined'] = y_df[valid].mean(axis=1)
        else:
            y_df['_y_combined'] = y_df[valid].sum(axis=1)
    if cfg['y_subset'] and cfg['y_rank_col']:
        y_subset_csv = cfg.get('y_subset_csv', cfg['y_csv'])
        y_subset_df = load_and_filter(y_subset_csv, cfg.get('y_year', cfg['year']), team_ids)
        y_plot_col = cfg.get('y_subset_plot_col') or cfg['y_col']
        y_df = apply_subset(y_subset_df, y_plot_col, cfg['y_rank_col'],
                            cfg['y_range_start'], cfg['y_range_end'], cfg['y_agg'])
        if y_plot_col != cfg['y_col'] and y_plot_col in y_df.columns:
            y_df = y_df.rename(columns={y_plot_col: cfg['y_col']})
    else:
        if cfg['y_col'] in y_df.columns:
            y_df[cfg['y_col']] = pd.to_numeric(y_df[cfg['y_col']], errors='coerce')

    # ── Data filters (can reference any CSV) ──
    for pf in cfg.get('player_filters', []):
        pf_csv = pf['csv']
        pf_col = pf['col']

        # Load filter data from the source CSV
        if pf_csv == cfg['x_csv']:
            filt_df = x_df
        elif pf_csv == cfg['y_csv']:
            filt_df = y_df
        else:
            filt_df = load_and_filter(pf_csv, cfg['year'], team_ids)

        if pf_col not in filt_df.columns:
            continue

        filt_df = filt_df.copy()
        filt_df[pf_col] = pd.to_numeric(filt_df[pf_col], errors='coerce')

        if pf['mode'] == 'Minimum':
            filt_df = filt_df[filt_df[pf_col] >= pf['min']]
        elif pf['mode'] == 'Maximum':
            filt_df = filt_df[filt_df[pf_col] <= pf['max']]
        elif pf['mode'] == 'Top N per team' and 'team_id' in filt_df.columns:
            filt_df = (filt_df.sort_values(pf_col, ascending=False)
                        .groupby('team_id', group_keys=False)
                        .head(pf['top_n']))

        # Apply: if same CSV as an axis, replace directly; otherwise filter by ID
        if pf_csv == cfg['x_csv']:
            x_df = filt_df
        elif pf_csv == cfg['y_csv']:
            y_df = filt_df
        else:
            # External CSV — filter by player_id or team_id
            if 'player_id' in filt_df.columns and 'player_id' in x_df.columns:
                keep_ids = set(filt_df['player_id'].astype(str))
                x_df = x_df[x_df['player_id'].astype(str).isin(keep_ids)]
                if 'player_id' in y_df.columns:
                    y_df = y_df[y_df['player_id'].astype(str).isin(keep_ids)]
            elif 'team_id' in filt_df.columns:
                keep_ids = set(filt_df['team_id'].astype(str))
                x_df = x_df[x_df['team_id'].astype(str).isin(keep_ids)]
                y_df = y_df[y_df['team_id'].astype(str).isin(keep_ids)]
                y_df = y_df[y_df['team_id'].astype(str).isin(keep_ids)]

    x_df = x_df.rename(columns={cfg['x_col']: '_x_val'})
    y_df = y_df.rename(columns={cfg['y_col']: '_y_val'})

    # ── Validate required columns exist ──
    for label, df, val_col in [('X', x_df, '_x_val'), ('Y', y_df, '_y_val')]:
        if val_col not in df.columns:
            raise ValueError(f'{label} column not found in {cfg[label.lower() + "_csv"]}. '
                             f'Pick a different data source or column.')
        if 'team_id' not in df.columns:
            raise ValueError(f'{cfg[label.lower() + "_csv"]} has no team_id column. '
                             f'Pick a data source that contains team_id.')

    # ── Join ──
    is_team_level = (cfg['mode'] == 'Team' or cfg.get('x_subset') or cfg.get('y_subset'))

    if is_team_level:
        x_key = x_df[['team_id', '_x_val']].dropna().drop_duplicates('team_id')
        y_key = y_df[['team_id', '_y_val']].dropna().drop_duplicates('team_id')
        merged = x_key.merge(y_key, on='team_id')
    else:
        # Both raw player-level
        if 'player_id' in x_df.columns and 'player_id' in y_df.columns:
            if cfg['x_csv'] == cfg['y_csv']:
                # Same CSV: x_df has _x_val, y_df has _y_val — merge on player_id
                x_key = x_df[['player_id', 'team_id', '_x_val']].dropna()
                y_key = y_df[['player_id', '_y_val']].dropna()
                merged = x_key.merge(y_key, on='player_id')
            else:
                x_key = x_df[['player_id', 'team_id', '_x_val']].dropna()
                y_key = y_df[['player_id', '_y_val']].dropna()
                merged = x_key.merge(y_key, on='player_id')
        else:
            x_key = x_df[['team_id', '_x_val']].dropna().drop_duplicates('team_id')
            y_key = y_df[['team_id', '_y_val']].dropna().drop_duplicates('team_id')
            merged = x_key.merge(y_key, on='team_id')

    # ── Add team metadata ──
    merged['team_id'] = merged['team_id'].astype(str)
    teams_meta = teams_df[['team_db_id', 'team_name', 'conference_name', 'division', 'logo_id']].copy()
    teams_meta['team_db_id'] = teams_meta['team_db_id'].astype(str)
    merged = merged.merge(teams_meta, left_on='team_id', right_on='team_db_id', how='left')

    # ── Filter to portal transfers only ──
    if cfg.get('portal_player_ids') and 'player_id' in merged.columns:
        merged = merged[merged['player_id'].astype(str).isin(cfg['portal_player_ids'])]

    # ── Add player names (for player mode) ──
    if 'player_id' in merged.columns:
        player_names = load_players()
        merged['player_name'] = merged['player_id'].map(player_names).fillna('Unknown')

    # ── Add rank weighting ──
    if cfg['rank_enabled'] and cfg['rank_col']:
        rank_df = load_and_filter(cfg['rank_csv'], cfg['year'], team_ids)
        rank_df[cfg['rank_col']] = pd.to_numeric(rank_df[cfg['rank_col']], errors='coerce')
        rank_key = rank_df[['team_id', cfg['rank_col']]].dropna().drop_duplicates('team_id')
        rank_key['team_id'] = rank_key['team_id'].astype(str)
        merged = merged.merge(rank_key, on='team_id', how='left', suffixes=('', '_rank'))

    # ── Add bubble scale column ──
    # Pulls a third stat (e.g. opponent ISO) into the merged frame so render_chart
    # can use it to size player dots / team logos. Joins on player_id when in
    # player mode (and the CSV has player_id), else falls back to team_id.
    if cfg.get('bubble_enabled') and cfg.get('bubble_col'):
        bcol = cfg['bubble_col']
        if bcol not in merged.columns:
            bdf = load_and_filter(cfg['bubble_csv'], cfg['year'], team_ids)
            if bcol in bdf.columns:
                bdf[bcol] = pd.to_numeric(bdf[bcol], errors='coerce')
                if 'player_id' in merged.columns and 'player_id' in bdf.columns:
                    bkey = bdf[['player_id', bcol]].dropna().drop_duplicates('player_id')
                    bkey['player_id'] = bkey['player_id'].astype(str)
                    merged['player_id'] = merged['player_id'].astype(str)
                    merged = merged.merge(bkey, on='player_id', how='left', suffixes=('', '_bubble'))
                elif 'team_id' in bdf.columns:
                    bkey = bdf[['team_id', bcol]].dropna().drop_duplicates('team_id')
                    bkey['team_id'] = bkey['team_id'].astype(str)
                    merged = merged.merge(bkey, on='team_id', how='left', suffixes=('', '_bubble'))

    # ── Add logo paths (use logo_id which maps softball→baseball IDs) ──
    def _logo_path(lid):
        if pd.isna(lid):
            return None
        # Normalize numeric ids — float 815.0 would produce '815.0.png' which doesn't exist
        try:
            as_num = pd.to_numeric(lid)
            if pd.notna(as_num):
                lid = int(as_num)
        except (ValueError, TypeError):
            pass
        p = LOGO_DIR / f'{lid}.png'
        return str(p) if p.exists() else None
    logo_col = 'logo_id' if 'logo_id' in merged.columns else 'team_id'
    merged['logo_path'] = merged[logo_col].apply(_logo_path)

    return merged


# ── Chart Rendering ───────────────────────────────────────────────────────────
def render_chart(data, cfg):
    """Render matplotlib chart and return figure."""
    x_col = '_x_val'
    y_col = '_y_val'
    is_team_markers = cfg['mode'] == 'Team' or cfg.get('x_subset') or cfg.get('y_subset')

    x_vals = data[x_col].values.astype(float)
    y_vals = data[y_col].values.astype(float)
    avg_x = np.nanmean(x_vals)
    avg_y = np.nanmean(y_vals)

    # Include CWS overlay points in axis range calculation
    all_x = x_vals.copy()
    all_y = y_vals.copy()
    _cws_data_cache = None
    if cfg.get('cws_enabled') and is_team_markers:
        _cws_data_cache = build_cws_overlay(cfg)
        if len(_cws_data_cache) > 0:
            cws_x = _cws_data_cache['_x_val'].values.astype(float)
            cws_y = _cws_data_cache['_y_val'].values.astype(float)
            all_x = np.concatenate([all_x, cws_x])
            all_y = np.concatenate([all_y, cws_y])

    pad_x = (np.nanmax(all_x) - np.nanmin(all_x)) * 0.10
    pad_y = (np.nanmax(all_y) - np.nanmin(all_y)) * 0.10
    xl = (np.nanmin(all_x) - pad_x, np.nanmax(all_x) + pad_x)
    yl = (np.nanmin(all_y) - pad_y, np.nanmax(all_y) + pad_y)

    # ── Compute zoom per team (rank weighting) ──
    if cfg['rank_enabled'] and cfg['rank_col'] and cfg['rank_col'] in data.columns:
        rank_vals = pd.to_numeric(data[cfg['rank_col']], errors='coerce')
        r_min, r_max = rank_vals.min(), rank_vals.max()
        if r_max > r_min:
            norm = (rank_vals - r_min) / (r_max - r_min)
        else:
            norm = pd.Series(0.5, index=rank_vals.index)
        if cfg.get('rank_invert', False):
            norm = 1.0 - norm
        zooms = cfg['zoom_min'] + norm * (cfg['zoom_max'] - cfg['zoom_min'])
    else:
        zooms = pd.Series(cfg['logo_zoom'], index=data.index)

    # ── Bubble scale (third-stat sizing) ──
    # Player mode: per-row marker size. Team mode: overrides `zooms` above.
    # Default = no per-row sizing (bubble_sizes is None → players use cfg['player_size']).
    bubble_sizes = None
    if (cfg.get('bubble_enabled') and cfg.get('bubble_col')
            and cfg['bubble_col'] in data.columns):
        bvals = pd.to_numeric(data[cfg['bubble_col']], errors='coerce')
        bmin, bmax = bvals.min(), bvals.max()
        if pd.notna(bmin) and pd.notna(bmax) and bmax > bmin:
            bnorm = (bvals - bmin) / (bmax - bmin)
        else:
            bnorm = pd.Series(0.5, index=bvals.index)
        if cfg.get('bubble_invert', False):
            bnorm = 1.0 - bnorm
        # Fill missing rows at the midpoint so they don't disappear
        bnorm = bnorm.fillna(0.5)
        # Top-end emphasis curve: bnorm**1.7 keeps mid-pack players modest in
        # size and lets the top ~10-20% really stand out. (Linear scaling made
        # the 'do it well' players blend in with the mid-pack.)
        bnorm = bnorm ** 1.7
        intensity = cfg.get('bubble_intensity', 4)
        if is_team_markers:
            # Team logos: asymmetric spread — much bigger jump at the top
            base_zoom = cfg.get('logo_zoom', 0.055)
            zoom_lo = max(0.01, base_zoom - intensity * 0.003)
            zoom_hi = base_zoom + intensity * 0.020
            zooms = zoom_lo + bnorm * (zoom_hi - zoom_lo)
        else:
            # Player dots: small bubbles barely shrink, big bubbles grow a LOT.
            # At intensity=4 the top dot is ~26pt; at intensity=10 it's ~56pt.
            base_size = cfg.get('player_size', 6)
            size_lo = max(2.0, base_size - intensity * 0.4)
            size_hi = base_size + intensity * 5.0
            bubble_sizes = size_lo + bnorm * (size_hi - size_lo)

    # ── Highlighted players ──
    callout_ids = set(cfg.get('callout_ids', []))
    callout_images = cfg.get('callout_images', {})

    # ── Theme colors ──
    t = THEMES[cfg.get('theme', 'Dark')]

    # ── Figure ──
    fig = plt.figure(figsize=(22, 17))
    fig.patch.set_facecolor(t['bg'])

    # Title block — bold condensed headline
    fig.text(0.03, 0.965, (cfg['title'] or '').strip().upper(),
             fontsize=13, color=t['text_md'], fontfamily=SUBTITLE_FONT, fontweight='bold',
             fontstyle='normal')
    title_main = cfg['subtitle'] if cfg['subtitle'] else f"{cfg['x_label']} vs {cfg['y_label']}"
    words = title_main.upper().split()
    if len(words) > 6:
        mid = len(words) // 2
        title_main = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
    else:
        title_main = title_main.upper()
    fig.text(0.50, 0.905, title_main, fontsize=40, color=t['text'],
             fontfamily=TITLE_FONT, fontweight='bold', linespacing=1.05, ha='center')

    count_label = f'{len(data)} PROGRAMS' if is_team_markers else f'{len(data)} PLAYERS'
    fig.text(0.97, 0.965, f'{count_label}  \u00b7  {cfg["year"]} SEASON',
             fontsize=12, color=t['text_dk'], fontfamily=BODY_FONT, ha='right')

    # Brand logo
    brand_arr = load_brand_logo(t['brand_logo'])
    logo_ax = fig.add_axes([0.90, 0.89, 0.08, 0.08])
    logo_ax.imshow(brand_arr)
    logo_ax.axis('off')

    # ── Main axes ──
    ax = fig.add_axes([0.07, 0.06, 0.90, 0.80])
    ax.set_facecolor(t['plot_bg'])
    # Flip axes so "best" is always top-right
    if cfg.get('x_direction') == 'Lower is better':
        ax.set_xlim(xl[1], xl[0])  # invert X
    else:
        ax.set_xlim(*xl)
    if cfg.get('y_direction') == 'Lower is better':
        ax.set_ylim(yl[1], yl[0])  # invert Y
    else:
        ax.set_ylim(*yl)

    # ── Quadrants ──
    if cfg['show_quadrants']:
        if cfg.get('custom_q_colors'):
            q_alpha = 0.25
        else:
            q_alpha = 0.6 if cfg['theme'] == 'Dark' else 0.35
        qc_tl = cfg.get('q_tl_color', t['q_tl']) if cfg.get('custom_q_colors') else t['q_tl']
        qc_tr = cfg.get('q_tr_color', t['q_tr']) if cfg.get('custom_q_colors') else t['q_tr']
        qc_bl = cfg.get('q_bl_color', t['q_bl']) if cfg.get('custom_q_colors') else t['q_bl']
        qc_br = cfg.get('q_br_color', t['q_br']) if cfg.get('custom_q_colors') else t['q_br']
        y_frac = (avg_y - yl[0]) / (yl[1] - yl[0])
        ax.axvspan(xl[0], avg_x, ymin=y_frac, ymax=1.0,
                   facecolor=qc_tl, alpha=q_alpha, zorder=0)
        ax.axvspan(avg_x, xl[1], ymin=y_frac, ymax=1.0,
                   facecolor=qc_tr, alpha=q_alpha, zorder=0)
        ax.axvspan(xl[0], avg_x, ymin=0, ymax=y_frac,
                   facecolor=qc_bl, alpha=q_alpha, zorder=0)
        ax.axvspan(avg_x, xl[1], ymin=0, ymax=y_frac,
                   facecolor=qc_br, alpha=q_alpha, zorder=0)

        ax.axvline(avg_x, color=t['text_dk'], linewidth=1.2, linestyle='--', alpha=0.7, zorder=1)
        ax.axhline(avg_y, color=t['text_dk'], linewidth=1.2, linestyle='--', alpha=0.7, zorder=1)

        ax.text(avg_x, yl[0] + (yl[1] - yl[0]) * 0.005, f'AVG {avg_x:.3f}',
                fontsize=9, color=t['text_md'], ha='center', va='bottom', fontfamily=BODY_FONT,
                bbox=dict(facecolor=t['avg_bg'], edgecolor='none', alpha=0.8, pad=2))
        ax.text(xl[0] + (xl[1] - xl[0]) * 0.005, avg_y, f'AVG {avg_y:.3f}',
                fontsize=9, color=t['text_md'], ha='left', va='center', fontfamily=BODY_FONT,
                bbox=dict(facecolor=t['avg_bg'], edgecolor='none', alpha=0.8, pad=2))

        lp = dict(fontsize=11, fontfamily=SUBTITLE_FONT, alpha=0.7, fontweight='bold')
        margin_x = (xl[1] - xl[0]) * 0.02
        margin_y = (yl[1] - yl[0]) * 0.02
        ax.text(xl[0] + margin_x, yl[1] - margin_y, cfg.get('q_tl', ''),
                color=t['q_tl_text'], ha='left', va='top', **lp)
        ax.text(xl[1] - margin_x, yl[1] - margin_y, cfg.get('q_tr', ''),
                color=t['q_tr_text'], ha='right', va='top', **lp)
        ax.text(xl[0] + margin_x, yl[0] + margin_y, cfg.get('q_bl', ''),
                color=t['q_bl_text'], ha='left', va='bottom', **lp)
        ax.text(xl[1] - margin_x, yl[0] + margin_y, cfg.get('q_br', ''),
                color=t['q_br_text'], ha='right', va='bottom', **lp)

    # ── Correlation / Trend line ──
    if cfg.get('show_correlation'):
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        if mask.sum() > 2:
            xf, yf = x_vals[mask], y_vals[mask]
            m, b = np.polyfit(xf, yf, 1)
            corr = np.corrcoef(xf, yf)[0, 1]
            r_sq = corr ** 2
            x_line = np.linspace(xl[0], xl[1], 200)
            y_line = m * x_line + b
            ax.plot(x_line, y_line, color=RED, linewidth=2.0, alpha=0.55, zorder=2,
                    linestyle='-')

            # ── Prediction band ──
            if cfg.get('show_prediction_band'):
                residuals = yf - (m * xf + b)
                resid_sd = np.std(residuals, ddof=2)
                n_sd = cfg.get('prediction_band_sd', 1)
                band = resid_sd * n_sd
                band_alpha = 0.12 if cfg.get('theme') == 'Dark' else 0.10
                ax.fill_between(x_line, y_line - band, y_line + band,
                                color=RED, alpha=band_alpha, zorder=1,
                                label=f'\u00b1{n_sd} SD')
                ax.plot(x_line, y_line - band, color=RED, linewidth=0.7,
                        alpha=0.25, linestyle='--', zorder=1)
                ax.plot(x_line, y_line + band, color=RED, linewidth=0.7,
                        alpha=0.25, linestyle='--', zorder=1)

            direction = 'positive' if corr > 0 else 'negative'
            band_label = f'\u00b1{cfg.get("prediction_band_sd", 1)} SD band' if cfg.get('show_prediction_band') else ''
            corr_box = dict(boxstyle='round,pad=0.5', facecolor=t['corr_bg'],
                            edgecolor=RED_DK, alpha=0.90, linewidth=1.2)
            ax.text(0.98, 0.04,
                    f'Pearson r = {corr:+.3f}\n'
                    f'R\u00b2 = {r_sq:.3f}\n'
                    f'{abs(corr):.0%} {direction} correlation'
                    + (f'\n{band_label}' if band_label else ''),
                    fontsize=10, color=t['text_sub'], va='bottom', ha='right',
                    transform=ax.transAxes, bbox=corr_box, fontfamily=BODY_FONT)

    # Grid
    ax.set_axisbelow(True)
    ax.grid(color=t['grid'], linewidth=0.5, linestyle='-', alpha=0.5)

    # ── Plot markers ──
    if is_team_markers:
        for idx, row in data.iterrows():
            logo_path = row.get('logo_path')
            zoom = zooms.loc[idx] if idx in zooms.index else cfg['logo_zoom']
            # Guard against NaN / non-string logo_path values (unmapped teams)
            if isinstance(logo_path, str) and logo_path and os.path.exists(logo_path):
                img = load_logo_thumbnail(str(logo_path))
                ab = AnnotationBbox(OffsetImage(img, zoom=zoom * 4, alpha=0.93),
                                    (row[x_col], row[y_col]),
                                    frameon=False, zorder=3)
                ax.add_artist(ab)
            else:
                ax.plot(row[x_col], row[y_col], 'o', color=RED, markersize=6,
                        alpha=0.7, zorder=3)
    else:
        # Player mode — scatter all players
        # Non-highlighted: small gray dots
        # Highlighted: larger markers with name labels
        has_pid = 'player_id' in data.columns
        for idx, row in data.iterrows():
            px, py = row[x_col], row[y_col]
            pid = str(row.get('player_id', '')) if has_pid else ''
            is_callout = pid in callout_ids

            if is_callout:
                # Check for uploaded image
                img_data = callout_images.get(pid)
                img_size = cfg.get('callout_sizes', {}).get(pid, 80)
                if img_data is not None:
                    try:
                        pil_img = Image.open(BytesIO(img_data)).convert('RGBA')
                        # Center-crop to square then resize
                        w, h = pil_img.size
                        side = min(w, h)
                        left = (w - side) // 2
                        top = (h - side) // 2
                        pil_img = pil_img.crop((left, top, left + side, top + side))
                        pil_img = pil_img.resize((img_size, img_size), Image.LANCZOS)
                        # Apply circular mask with border
                        mask = Image.new('L', (img_size, img_size), 0)
                        ImageDraw.Draw(mask).ellipse((0, 0, img_size - 1, img_size - 1), fill=255)
                        # Draw border ring
                        border = Image.new('RGBA', (img_size, img_size), (0, 0, 0, 0))
                        bd = ImageDraw.Draw(border)
                        bd.ellipse((0, 0, img_size - 1, img_size - 1),
                                   outline=(196, 18, 48, 255), width=3)
                        # Composite: circular photo + border
                        result = Image.new('RGBA', (img_size, img_size), (0, 0, 0, 0))
                        result.paste(pil_img, mask=mask)
                        result = Image.alpha_composite(result, border)
                        ab = AnnotationBbox(OffsetImage(np.array(result), zoom=1.0, alpha=0.95),
                                            (px, py), frameon=False, zorder=7)
                        ax.add_artist(ab)
                    except Exception:
                        ax.plot(px, py, 'o', color='#FFFFFF', markersize=18,
                                markeredgecolor=RED, markeredgewidth=3, zorder=5)
                else:
                    ax.plot(px, py, 'o', color='#FFFFFF', markersize=18,
                            markeredgecolor=RED, markeredgewidth=3, zorder=5)

                # Name label
                name = row.get('player_name', f'ID:{pid}')
                team = row.get('team_name', '')
                label_text = f'{name}\n{team}' if team else name
                co = cfg.get('callout_offsets', {}).get(pid, (20, 20))
                ax.annotate(label_text, (px, py),
                            textcoords='offset points', xytext=co,
                            fontsize=14, color=t['text'], fontfamily=BODY_FONT, fontweight='bold',
                            bbox=dict(facecolor=t['callout_bg'], edgecolor=RED, alpha=0.92,
                                      pad=5, boxstyle='round,pad=0.5'),
                            arrowprops=dict(arrowstyle='->', color=RED, lw=2.5,
                                           connectionstyle='arc3,rad=0.15'),
                            zorder=6)
            else:
                ms = (float(bubble_sizes.loc[idx])
                      if bubble_sizes is not None and idx in bubble_sizes.index
                      else cfg['player_size'])
                ax.plot(px, py, cfg['player_marker'],
                        color=cfg['player_color'],
                        markersize=ms,
                        alpha=cfg['player_alpha'],
                        markeredgecolor=cfg['player_edge_color'],
                        markeredgewidth=cfg['player_edge_width'],
                        zorder=3)

    # ── Axes styling ──
    ax.tick_params(colors=t['text_sub'], labelsize=11, labelcolor=t['text_md'])
    for spine in ax.spines.values():
        spine.set_edgecolor(t['spine'])

    x_range = np.nanmax(x_vals) - np.nanmin(x_vals)
    y_range = np.nanmax(y_vals) - np.nanmin(y_vals)
    if x_range < 5:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.3f}'))
    if y_range < 5:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.3f}'))

    x_arrow = '\u2192' if cfg['x_direction'] == 'Higher is better' else '\u2190'
    y_arrow = '\u2192' if cfg['y_direction'] == 'Higher is better' else '\u2190'
    ax.set_xlabel(f'{cfg["x_label"].upper()}  {x_arrow}  {cfg["x_direction"].upper()}',
                  fontsize=12, color=t['text_md'], labelpad=14, fontfamily=AXIS_FONT, fontweight='bold')
    ax.set_ylabel(f'{cfg["y_label"].upper()}  {y_arrow}  {cfg["y_direction"].upper()}',
                  fontsize=12, color=t['text_md'], labelpad=14, fontfamily=AXIS_FONT, fontweight='bold')

    # ── Custom annotations ──
    for ann in cfg.get('annotations', []):
        ax.annotate(ann['text'], (ann['x'], ann['y']),
                    textcoords='offset points', xytext=(ann['off_x'], ann['off_y']),
                    fontsize=11, color=t['text'], fontfamily=BODY_FONT, fontweight='bold',
                    bbox=dict(facecolor=t['callout_bg'], edgecolor=t['spine'], alpha=0.90,
                              pad=4, boxstyle='round,pad=0.4'),
                    arrowprops=dict(arrowstyle='->', color=t['text_md'], lw=1.5,
                                   connectionstyle='arc3,rad=0.2'),
                    zorder=7)

    # ── CWS Teams Overlay ──
    # Plots CWS teams using their stats from their actual CWS year
    if cfg.get('cws_enabled') and is_team_markers:
        from matplotlib.lines import Line2D
        theme_mode = 'light' if cfg.get('theme') == 'Light' else 'dark'
        champ_color = '#FFD700' if theme_mode == 'dark' else '#B8860B'

        cws_data = _cws_data_cache if _cws_data_cache is not None else build_cws_overlay(cfg)
        legend_entries = []
        years_seen = set()

        if len(cws_data) > 0:
            # Fade older CWS years — most recent is opaque, oldest is transparent
            selected_years = sorted(cfg.get('cws_years', []))
            if selected_years:
                most_recent = int(selected_years[-1])
                oldest = int(selected_years[0])
                yr_span = max(most_recent - oldest, 1)

            for _, row in cws_data.iterrows():
                px, py = row['_x_val'], row['_y_val']
                cws_year = row['_cws_year']
                is_champ = row.get('_is_champion', False)

                yr_color_set = CWS_YEAR_COLORS.get(cws_year, {'dark': '#FFFFFF', 'light': '#333333'})
                yr_color = yr_color_set[theme_mode]
                years_seen.add(cws_year)

                # Low opacity for all CWS teams — champions get gold ring to stand out
                logo_alpha = 0.35
                label_alpha = 0.55

                # Draw team logo
                logo_path_str = row.get('logo_path')
                logo_z = cfg.get('logo_zoom', 0.055)
                if isinstance(logo_path_str, str) and logo_path_str and os.path.exists(logo_path_str):
                    img = load_logo_thumbnail(str(logo_path_str))
                    ab = AnnotationBbox(OffsetImage(img, zoom=logo_z * 4, alpha=logo_alpha),
                                        (px, py), frameon=False, zorder=8)
                    ax.add_artist(ab)
                else:
                    ax.plot(px, py, 'o', color=yr_color, markersize=8, alpha=logo_alpha, zorder=8)

                # Champion gets gold ring
                if is_champ:
                    ax.scatter(px, py, s=55**2, facecolors='none', edgecolors=champ_color,
                              linewidths=3, zorder=9, alpha=0.95)
                    label_text = f"\U0001F3C6 '{cws_year[2:]}"
                    label_bg = champ_color
                    label_fg = '#1a1a1a'
                    label_fs = 8
                    label_edge = champ_color
                else:
                    label_text = f"CWS '{cws_year[2:]}"
                    label_bg = t['plot_bg']
                    label_fg = yr_color
                    label_fs = 7.5
                    label_edge = yr_color

                ax.annotate(label_text, (px, py),
                            textcoords='offset points', xytext=(0, -18),
                            ha='center', va='top', fontsize=label_fs, fontweight='bold',
                            fontfamily=BODY_FONT, color=label_fg,
                            bbox=dict(facecolor=label_bg, edgecolor=label_edge,
                                      alpha=label_alpha, pad=1.5, boxstyle='round,pad=0.2'),
                            zorder=10)

        # Legend
        for yr in sorted(years_seen):
            yr_color_set = CWS_YEAR_COLORS.get(yr, {'dark': '#FFFFFF', 'light': '#333333'})
            yr_color = yr_color_set[theme_mode]
            legend_entries.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                       markeredgecolor=yr_color, markeredgewidth=2,
                       markersize=10, label=f"'{yr[2:]} CWS Teams")
            )
        legend_entries.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor=champ_color, markeredgewidth=3,
                   markersize=12, label='National Champion')
        )

        if legend_entries:
            leg = ax.legend(handles=legend_entries, loc='upper left',
                           frameon=True, fontsize=9, labelcolor=t['text_sub'],
                           facecolor=t['plot_bg'], edgecolor=t['spine'],
                           framealpha=0.9)
            leg.set_zorder(11)

    n_label = f'{len(data)} teams' if is_team_markers else f'{len(data)} players'
    ax.text(0.01, -0.05, f'n = {n_label}  \u00b7  64analytics.com',
            fontsize=10, color=t['text_dk'], transform=ax.transAxes, fontfamily=BODY_FONT)

    return fig


def render_plotly_scatter(data, cfg, static_png_b64):
    """Interactive view = the static matplotlib chart + invisible hover overlay.

    Strategy: render the matplotlib chart in Player mode (the *exact* static
    visual the user already likes — quadrants, trend line, logo, callouts,
    everything), embed it as a Plotly background image, and overlay an
    invisible hover-only scatter trace at the player coordinates. The user
    sees the static chart pixel-for-pixel and gains tooltips on every dot.

    The matplotlib axes live at figure-fraction [0.07, 0.06, 0.97, 0.86]
    (left, bottom, right, top) — Plotly's axis domain is set to match so
    data-coord overlays land on the same pixels as the matplotlib dots.
    """
    import plotly.graph_objects as go

    x_col, y_col = '_x_val', '_y_val'

    df = data.copy()
    for col in ('player_name', 'team_name', 'conference_name'):
        if col not in df.columns:
            df[col] = ''
    if 'player_id' not in df.columns:
        df['player_id'] = ''

    # Axis range MUST match the matplotlib computation in render_chart so dots
    # overlay correctly on the static PNG.
    x_vals = df[x_col].values.astype(float)
    y_vals = df[y_col].values.astype(float)
    pad_x = (np.nanmax(x_vals) - np.nanmin(x_vals)) * 0.10
    pad_y = (np.nanmax(y_vals) - np.nanmin(y_vals)) * 0.10
    xl = (float(np.nanmin(x_vals) - pad_x), float(np.nanmax(x_vals) + pad_x))
    yl = (float(np.nanmin(y_vals) - pad_y), float(np.nanmax(y_vals) + pad_y))

    # Apply axis inversion to match render_chart's "best is top-right" rule
    x_range = [xl[1], xl[0]] if cfg.get('x_direction') == 'Lower is better' else list(xl)
    y_range = [yl[1], yl[0]] if cfg.get('y_direction') == 'Lower is better' else list(yl)

    customdata = np.column_stack([
        df['player_name'].fillna('Unknown').astype(str),
        df['team_name'].fillna('').astype(str),
        df['conference_name'].fillna('').astype(str),
        df['player_id'].astype(str),
    ])

    x_label = cfg['x_label']
    y_label = cfg['y_label']
    hover_tmpl = (
        '<b>%{customdata[0]}</b><br>'
        '%{customdata[1]} \u00b7 %{customdata[2]}<br>'
        f'{x_label}: ' + '%{x:.3f}<br>'
        f'{y_label}: ' + '%{y:.3f}'
        '<extra></extra>'
    )

    fig = go.Figure()

    # The static PNG as the entire visual layer
    fig.add_layout_image(
        source=f'data:image/png;base64,{static_png_b64}',
        xref='paper', yref='paper',
        x=0, y=1, sizex=1, sizey=1,
        xanchor='left', yanchor='top',
        sizing='stretch', layer='below',
    )

    # Invisible hover dots — opacity is 0.01 so the static dots remain pristine
    # but Plotly still triggers hover/click events when the cursor hits them.
    fig.add_trace(go.Scatter(
        x=df[x_col], y=df[y_col],
        mode='markers',
        marker=dict(
            size=18,                       # bigger than the static dot for easier hover targeting
            color='rgba(0,0,0,0.01)',      # essentially invisible
            line=dict(width=0),
        ),
        customdata=customdata,
        hovertemplate=hover_tmpl,
        name='', showlegend=False,
    ))

    fig.update_layout(
        # Match the matplotlib figsize=(22,17) aspect (~1.294)
        width=1100, height=850,
        autosize=False,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        # Hidden axes — the static PNG already has its own axis labels and ticks
        xaxis=dict(
            visible=False,
            range=x_range,
            domain=[0.07, 0.97],           # matches matplotlib ax.add_axes([0.07, 0.06, 0.90, 0.80])
            fixedrange=True,
        ),
        yaxis=dict(
            visible=False,
            range=y_range,
            domain=[0.06, 0.86],
            fixedrange=True,
        ),
        hovermode='closest',
        clickmode='event+select',
        dragmode=False,
        hoverlabel=dict(
            bgcolor='rgba(20,20,20,0.95)',
            bordercolor=RED,
            font=dict(color='#FFFFFF', family='Inter', size=13),
        ),
    )

    return fig


# ── Player Callout UI ─────────────────────────────────────────────────────────
def player_callout_ui(data, cfg):
    """Show player search/select UI when in player mode. Returns callout config."""
    cfg['callout_ids'] = set()
    cfg['callout_images'] = {}

    if cfg['mode'] != 'Player' or 'player_id' not in data.columns:
        return

    st.markdown('### Highlight Players')
    st.caption('Search and select players to callout on the chart with name labels. '
               'Optionally upload a headshot image for each.')

    # Build searchable list: "Name (Team) — id"
    options = []
    pid_map = {}
    for _, row in data.iterrows():
        name = row.get('player_name', 'Unknown')
        team = row.get('team_name', '')
        pid = str(row['player_id'])
        label = f'{name} — {team}' if team else name
        options.append(label)
        pid_map[label] = pid

    selected = st.multiselect('Select players to highlight', sorted(set(options)),
                              help='Type to search by name or team')

    callout_ids = {pid_map[s] for s in selected if s in pid_map}
    cfg['callout_ids'] = callout_ids

    # Per-callout controls
    cfg['callout_offsets'] = {}
    cfg['callout_sizes'] = {}
    if callout_ids:
        for label in selected:
            pid = pid_map.get(label, '')
            with st.expander(f'{label}', expanded=False):
                c1, c2 = st.columns(2)
                off_x = c1.number_input('Label X offset', value=20, step=5, key=f'co_x_{pid}')
                off_y = c2.number_input('Label Y offset', value=20, step=5, key=f'co_y_{pid}')
                cfg['callout_offsets'][pid] = (off_x, off_y)
                cfg['callout_sizes'][pid] = st.slider('Headshot size', 40, 200, 80, 10, key=f'co_sz_{pid}')
                uploaded = st.file_uploader('Headshot (optional)', type=['png', 'jpg', 'jpeg'],
                                            key=f'img_{pid}')
                if uploaded:
                    cfg['callout_images'][pid] = uploaded.read()


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title='64 Analytics Chart Builder', layout='wide',
                       initial_sidebar_state='expanded')

    cfg = sidebar()

    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
    html, body, .stApp, .stApp *:not([class*="icon"]):not([class*="Icon"]):not([data-testid*="icon"]):not([data-testid*="Icon"]):not([data-testid*="arrow"]):not(.material-icons):not(.material-symbols):not(.material-symbols-rounded){ font-family: 'Inter', sans-serif !important; }
    .stApp { background-color: #1a1a1a; }
    .stSidebar { background-color: #111111; }
    h1, h2, h3, p, label, .stMarkdown { color: #C8C8C8 !important; }
    </style>
    """, unsafe_allow_html=True)

    # Build data
    try:
        data = build_data(cfg)
    except Exception as e:
        st.error(f'Error building data: {e}')
        st.stop()

    if len(data) == 0:
        st.warning('No data matches your filters. Try adjusting year, division, or conference.')
        st.stop()

    # Info bar + theme toggle
    is_team = cfg['mode'] == 'Team' or cfg.get('x_subset') or cfg.get('y_subset')
    col1, col2, col3, col4 = st.columns([1, 1, 1, 0.5])
    col1.metric('Teams' if is_team else 'Players', len(data))
    col2.metric(f'Avg {cfg["x_label"]}', f'{data["_x_val"].mean():.3f}')
    col3.metric(f'Avg {cfg["y_label"]}', f'{data["_y_val"].mean():.3f}')
    _use_light = col4.checkbox('Light theme', value=False, key='use_light_chart')
    cfg['theme'] = 'Light' if _use_light else 'Dark'
    chart_theme = THEMES[cfg['theme']]

    # Player callout UI (shown above chart in player mode)
    player_callout_ui(data, cfg)

    # Branch on view mode: Interactive (Plotly w/ hover) only works in Player mode;
    # Static (matplotlib PNG) is the default and the only option for Team mode.
    use_interactive = cfg.get('view_mode') == 'Interactive' and not is_team
    if cfg.get('view_mode') == 'Interactive' and is_team:
        st.info('Interactive view is only available in Player mode. '
                'Showing static chart for team mode.')

    if use_interactive:
        try:
            with st.spinner('Rendering interactive chart...'):
                # Render the matplotlib chart once and use it as both the
                # downloadable PNG AND the visual layer of the interactive view.
                fig = render_chart(data, cfg)
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=180,
                            facecolor=chart_theme['bg'], edgecolor='none')
                buf.seek(0)
                png_bytes = buf.getvalue()
                plt.close(fig)
                static_b64 = base64.b64encode(png_bytes).decode('ascii')
                pfig = render_plotly_scatter(data, cfg, static_b64)

            event = st.plotly_chart(
                pfig, use_container_width=False, theme=None,
                key='interactive_chart',
                on_select='rerun',
                selection_mode='points',
                config={'displayModeBar': False, 'doubleClick': False},
            )
            st.caption('Hover any dot for player details. **Click** a dot to '
                       'pin it below. Switch to Static view in the sidebar to '
                       'download a PNG.')

            # Click-to-freeze panel: show the most recently clicked player
            sel_points = []
            try:
                sel_points = event.selection.points if event and event.selection else []
            except Exception:
                sel_points = []
            if sel_points:
                pt = sel_points[-1]
                cd = pt.get('customdata') or []
                pname = cd[0] if len(cd) > 0 else 'Unknown'
                pteam = cd[1] if len(cd) > 1 else ''
                pconf = cd[2] if len(cd) > 2 else ''
                px = pt.get('x')
                py = pt.get('y')
                with st.container(border=True):
                    c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
                    c1.markdown(f'**{pname}**  \n{pteam} \u00b7 {pconf}')
                    c2.metric(cfg['x_label'], f'{px:.3f}' if px is not None else '\u2014')
                    c3.metric(cfg['y_label'], f'{py:.3f}' if py is not None else '\u2014')
                    if c4.button('Clear', key='clear_pin'):
                        st.session_state.pop('interactive_chart', None)
                        st.rerun()

            # Download button still available in interactive view
            st.download_button('Download PNG', data=png_bytes,
                               file_name='64analytics_chart.png',
                               mime='image/png')
        except Exception as e:
            st.error(f"Interactive chart rendering failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            return
    else:
        try:
            with st.spinner('Rendering chart...'):
                fig = render_chart(data, cfg)

            # Render to image buffer with explicit facecolor
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=180, bbox_inches='tight',
                        facecolor=chart_theme['bg'], edgecolor='none')
            buf.seek(0)
            plt.close(fig)
        except Exception as e:
            st.error(f"Chart rendering failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

        st.image(buf, use_container_width=True)

        # Download button (re-seek buffer)
        buf.seek(0)
        st.download_button('Download PNG', data=buf, file_name='64analytics_chart.png',
                           mime='image/png')

    # Data preview
    with st.expander('View data table'):
        display = data.copy()
        display = display.rename(columns={'_x_val': cfg['x_label'], '_y_val': cfg['y_label']})
        show_cols = ['team_name', cfg['x_label'], cfg['y_label']]
        if 'player_name' in display.columns:
            show_cols.insert(0, 'player_name')
        if cfg['rank_enabled'] and cfg['rank_col'] and cfg['rank_col'] in display.columns:
            show_cols.append(cfg['rank_col'])
        available = [c for c in show_cols if c in display.columns]
        st.dataframe(display[available].sort_values(cfg['x_label'], ascending=False),
                     use_container_width=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"App crashed: {e}")
        st.code(traceback.format_exc())
