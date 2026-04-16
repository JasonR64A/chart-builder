"""
64 Analytics — Similar Entities Finder
Find players or teams with similar performance profiles within the same division.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from io import BytesIO
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── Path setup ────────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _APP_DIR / 'data'
LOGO_DIR = _APP_DIR / 'team_logos_512'

# ── Fonts ─────────────────────────────────────────────────────────────────────
def _has_font(name):
    return any(name.lower() in f.name.lower() for f in fm.fontManager.ttflist)

TITLE_FONT = 'Franklin Gothic Heavy' if _has_font('Franklin Gothic') else 'DejaVu Sans'
SUBTITLE_FONT = 'Franklin Gothic Medium' if _has_font('Franklin Gothic') else 'DejaVu Sans'
BODY_FONT = 'Calibri' if _has_font('Calibri') else 'DejaVu Sans'

# ── Metric definitions ────────────────────────────────────────────────────────
# (raw_column, display_name, lower_is_better)
HITTING_METRICS = [
    ('on_base_plus_slugging',     'OPS',  False),
    ('weighted_on_base_average',  'wOBA', False),
    ('isolated_power',            'ISO',  False),
    ('weighted_runs_created',     'wRC',  False),
    ('weighted_runs_above_average','wRAA', False),
    ('runs_plate_appearance',     'R/PA', False),
]

PITCHING_METRICS = [
    ('on_base_plus_slugging_against',         'A-OPS', True),
    ('walk_percentage',                        'BB%',   True),
    ('strikeout_percentage',                   'K%',    False),
    ('strikeout_to_walk_ratio',                'K/BB',  False),
    ('walks_plus_hits_per_inning_pitched',     'WHIP',  True),
    ('fielding_independent_pitching',          'FIP',   True),
]

VOLUME_WEIGHT = 2.0  # how strongly to weight PA/IP similarity vs metric similarity

# All 8 CWS (Omaha) teams by year (team names match teams.csv)
CWS_TEAMS = {
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

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_teams_division(sport):
    """Returns DataFrame with team_id, team_name, division, sport for the given sport."""
    teams = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False)
    confs = pd.read_csv(DATA_DIR / 'conferences.csv', low_memory=False)
    sport_label = sport.title() if sport != 'softball' else 'Softball'
    teams = teams[teams['sport'] == sport_label][['id', 'name', 'conference_id']].copy()
    teams = teams.rename(columns={'id': 'team_id', 'name': 'team_name'})
    div_map = dict(zip(confs['id'], confs['division']))
    teams['division'] = teams['conference_id'].map(div_map)
    div_normalize = {'D-I': 'D1', 'D-II': 'D2', 'D-III': 'D3'}
    teams['division'] = teams['division'].map(div_normalize)
    return teams[['team_id', 'team_name', 'division']]

@st.cache_data
def load_players():
    df = pd.read_csv(DATA_DIR / 'players.csv', low_memory=False, encoding='latin-1')
    cols = ['id', 'player_name', 'position']
    cols = [c for c in cols if c in df.columns]
    out = df[cols].rename(columns={'id': 'player_id'})
    if 'position' in out.columns:
        # Normalize case (some rows have 'p' vs 'P', 'lf' vs 'LF', etc.)
        out['position'] = out['position'].astype(str).str.upper().str.strip()
    return out


# Position compatibility rules for similarity matching.
# Each key is the target player's position; value is the set of positions
# that the target can be meaningfully compared against.
POSITION_GROUPS = {
    # OF: any outfielder (and generic OF)
    'OF': {'OF', 'LF', 'CF', 'RF'},
    'LF': {'OF', 'LF', 'CF', 'RF'},
    'CF': {'OF', 'LF', 'CF', 'RF'},
    'RF': {'OF', 'LF', 'CF', 'RF'},
    # Corners: 1B↔3B↔INF
    '1B': {'1B', '3B', 'INF'},
    '3B': {'3B', '1B', 'INF'},
    # Middle IF: 2B↔SS↔INF
    '2B': {'2B', 'SS', 'INF'},
    'SS': {'SS', '2B', 'INF'},
    # Generic INF matches any infielder
    'INF': {'1B', '2B', '3B', 'SS', 'INF'},
    # C only with C
    'C': {'C'},
    # DH matches anyone (generic bat)
    'DH': {'OF', 'LF', 'CF', 'RF', '1B', '2B', '3B', 'SS', 'INF', 'C', 'DH', 'UT'},
    # UT (utility) — treat like DH, any
    'UT': {'OF', 'LF', 'CF', 'RF', '1B', '2B', '3B', 'SS', 'INF', 'C', 'DH', 'UT'},
    # Pitchers (pitcher similarity already filters to pitching stats)
    'P': {'P', 'SP', 'RP'},
    'SP': {'P', 'SP', 'RP'},
    'RP': {'P', 'SP', 'RP'},
}


def compatible_positions(target_pos):
    """Return the set of positions compatible with target_pos, per POSITION_GROUPS.
    Unknown/missing positions fall back to "all positions allowed" so we never
    silently exclude everyone."""
    if not isinstance(target_pos, str) or not target_pos:
        return None  # None => no filter
    pos = target_pos.upper().strip()
    return POSITION_GROUPS.get(pos)  # None if not mapped => no filter

@st.cache_data
def load_player_hitting(sport, division):
    """Load hitting for ALL years for the given sport/division."""
    hit = pd.read_csv(DATA_DIR / 'hitting.csv', low_memory=False)
    teams = load_teams_division(sport)
    teams = teams[teams['division'] == division]
    df = hit.merge(teams, on='team_id', how='inner')
    players = load_players()
    df = df.merge(players, on='player_id', how='left')
    df = df[df['plate_appearances'] > 0].copy()
    return df

@st.cache_data
def load_player_pitching(sport, division):
    pit = pd.read_csv(DATA_DIR / 'pitching.csv', low_memory=False)
    teams = load_teams_division(sport)
    teams = teams[teams['division'] == division]
    df = pit.merge(teams, on='team_id', how='inner')
    players = load_players()
    df = df.merge(players, on='player_id', how='left')
    df = df[df['batters_faced'] > 0].copy()
    return df

@st.cache_data
def load_team_hitting(sport, division):
    th = pd.read_csv(DATA_DIR / 'hitting_team.csv', low_memory=False)
    teams = load_teams_division(sport)
    teams = teams[teams['division'] == division]
    return th.merge(teams, on='team_id', how='inner')

@st.cache_data
def load_team_pitching(sport, division):
    tp = pd.read_csv(DATA_DIR / 'pitching_team.csv', low_memory=False)
    teams = load_teams_division(sport)
    teams = teams[teams['division'] == division]
    return tp.merge(teams, on='team_id', how='inner')

# ── Percentile + Similarity ───────────────────────────────────────────────────
def compute_percentiles(df, metrics, cohort_col='year'):
    """Compute 0-100 percentile rank for each metric WITHIN each cohort
    (default: per year). Inverts lower-is-better stats so higher = better."""
    out = pd.DataFrame(index=df.index)
    for col, _, lower_better in metrics:
        if col not in df.columns:
            continue
        # Rank within each cohort (year)
        ranks = df.groupby(cohort_col)[col].rank(pct=True, method='min')
        if lower_better:
            out[col] = (1 - ranks) * 100
        else:
            out[col] = ranks * 100
    return out

def find_similar(target_idx, df, metrics, volume_col=None, top_n=5, position_filter=None):
    """Return the top_n most similar entities to the target by Euclidean distance
    in percentile space, with optional volume weighting. Percentiles are
    computed per-year cohort so cross-year comparisons are normalized.

    position_filter: optional set of allowed positions. If provided and df has
    a 'position' column, candidates are restricted to these positions."""
    pct = compute_percentiles(df, metrics, cohort_col='year')
    if target_idx not in pct.index:
        return None, None
    target_vec = pct.loc[target_idx].values
    diffs = pct.values - target_vec  # (n, m)
    eucl = np.sqrt((diffs ** 2).sum(axis=1))

    if volume_col and volume_col in df.columns:
        # Volume percentile also computed per-year cohort
        vol_pct = df.groupby('year')[volume_col].rank(pct=True, method='min') * 100
        target_vol = vol_pct.loc[target_idx]
        vol_penalty = np.abs(vol_pct.values - target_vol) * VOLUME_WEIGHT
        eucl = eucl + vol_penalty

    df_out = df.copy()
    df_out['_distance'] = eucl
    df_out = df_out[df_out.index != target_idx]

    # Apply position filter (if provided and column exists)
    if position_filter is not None and 'position' in df_out.columns:
        df_out = df_out[df_out['position'].isin(position_filter)]

    df_out = df_out.sort_values('_distance').head(top_n)
    return df_out, pct


def find_closest_omaha(target_idx, df, metrics):
    """Find the single closest CWS (Omaha) team to the target.
    Uses the same percentile-space distance as find_similar but filters
    candidates to only teams that appeared in the CWS (2021-2025)."""
    # Build set of (team_name, year) for all CWS teams
    cws_set = set()
    for yr_str, teams in CWS_TEAMS.items():
        yr = int(yr_str)
        for t in teams:
            cws_set.add((t, yr))

    # Find which rows in df are CWS teams
    cws_mask = df.apply(lambda r: (r.get('team_name'), int(r.get('year', 0))) in cws_set, axis=1)
    if not cws_mask.any():
        return None, None

    pct = compute_percentiles(df, metrics, cohort_col='year')
    if target_idx not in pct.index:
        return None, None

    target_vec = pct.loc[target_idx].values
    diffs = pct.values - target_vec
    eucl = np.sqrt((diffs ** 2).sum(axis=1))

    df_out = df.copy()
    df_out['_distance'] = eucl
    # Exclude the target itself and filter to CWS teams only
    df_out = df_out[(df_out.index != target_idx) & cws_mask]
    if len(df_out) == 0:
        return None, None
    closest = df_out.sort_values('_distance').head(1)
    return closest, pct

# ── Logo helpers ──────────────────────────────────────────────────────────────
@st.cache_data
def load_team_logo_map():
    teams = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False)
    teams['id'] = pd.to_numeric(teams['id'], errors='coerce').fillna(0).astype(int)
    bb = teams[teams['sport'] == 'Baseball'][['name', 'id']].drop_duplicates('name')
    name_to_id = dict(zip(bb['name'], bb['id']))
    sb = teams[teams['sport'] == 'Softball'][['name', 'id']].drop_duplicates('name')
    for _, row in sb.iterrows():
        if row['name'] not in name_to_id:
            name_to_id[row['name']] = row['id']
    return name_to_id

def get_team_color(team_name):
    """Extract dominant color from a team's logo."""
    from collections import Counter
    team_map = load_team_logo_map()
    logo_id = team_map.get(team_name)
    if not logo_id:
        return '#C41230'
    for ext in ['png', 'webp']:
        p = LOGO_DIR / f'{logo_id}.{ext}'
        if p.exists():
            try:
                img = Image.open(p).convert('RGBA')
                img.thumbnail((64, 64))
                pixels = np.array(img)
                mask = pixels[:, :, 3] > 128
                rgb = pixels[mask][:, :3]
                if len(rgb) == 0:
                    return '#C41230'
                filtered = []
                for r, g, b in rgb:
                    brightness = (int(r) + int(g) + int(b)) / 3
                    if brightness > 220 or brightness < 35:
                        continue
                    filtered.append((r, g, b))
                if not filtered:
                    return '#C41230'
                quantized = [(r // 16 * 16, g // 16 * 16, b // 16 * 16) for r, g, b in filtered]
                most_common = Counter(quantized).most_common(1)[0][0]
                return f'#{most_common[0]:02x}{most_common[1]:02x}{most_common[2]:02x}'
            except Exception:
                return '#C41230'
    return '#C41230'

# ── Bar chart ─────────────────────────────────────────────────────────────────
def _make_grainy_bg(color_rgb, size=400, noise_scale=0.03, seed=42):
    """Generate a grainy noise background array around a base color."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, (size, size, 3)) * noise_scale
    base = np.array(color_rgb)
    return np.clip(base + noise, 0, 1)

def render_similarity_chart(target_label, target_team, target_pct, matches, match_pcts, metrics, theme='Light', show_year=False, entity_type='player', omaha_comp=None, omaha_pcts=None):
    """Spider/radar chart: each axis is a metric, each entity is a polygon."""
    if theme == 'Dark':
        bg_solid = '#1a1a1a'; text_color = '#e2e8f0'; text_md = '#a0aec0'
        grid_color = '#3a3a3a'; spine_color = '#2d3748'
        bg_rgb = (0.10, 0.10, 0.10)
        noise_scale = 0.012
    else:
        bg_solid = '#FAF8F2'; text_color = '#2D2926'; text_md = '#4A4540'
        grid_color = '#D8D2C4'; spine_color = '#E2DCCC'
        bg_rgb = (0.980, 0.972, 0.949)  # near-white with hint of warmth
        noise_scale = 0.012

    # Use the target team's actual color (falls back to brand red)
    target_color = get_team_color(target_team) or '#C41230'

    metric_labels = [m[1] for m in metrics]
    metric_cols = [m[0] for m in metrics]
    n_metrics = len(metric_labels)

    # Compute angle per axis
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    fig = plt.figure(figsize=(11, 11), facecolor=bg_solid)

    # Grainy textured background that fills the whole figure
    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
    bg_ax.set_axis_off()
    bg_img = _make_grainy_bg(bg_rgb, size=500, noise_scale=noise_scale)
    bg_ax.imshow(bg_img, aspect='auto', extent=[0, 1, 0, 1], interpolation='bilinear')

    ax = fig.add_subplot(111, polar=True, facecolor='none')

    # Match polygons FIRST (behind target)
    for i, (_, m_row) in enumerate(matches.iterrows()):
        m_name = m_row.get('player_name') or m_row.get('team_name', '?')
        m_team = m_row.get('team_name', '?')
        m_year = int(m_row['year']) if show_year and 'year' in m_row.index else None
        if show_year and m_year is not None:
            if m_row.get('player_name'):
                legend_label = f'{m_name} — {m_team} ({m_year})'
            else:
                legend_label = f'{m_team} ({m_year})'
        else:
            legend_label = f'{m_name} ({m_team})'
        m_color = get_team_color(m_team)
        m_vals = [float(match_pcts.loc[m_row.name, c]) if c in match_pcts.columns else 0.0 for c in metric_cols]
        m_vals_closed = m_vals + [m_vals[0]]
        ax.plot(angles_closed, m_vals_closed, color=m_color, linewidth=1.8,
                alpha=0.75, zorder=3, label=legend_label, linestyle='--')
        ax.fill(angles_closed, m_vals_closed, color=m_color, alpha=0.06, zorder=2)

    # Omaha comp polygon — drawn BETWEEN matches and target, gold solid line
    if omaha_comp is not None and omaha_pcts is not None and len(omaha_comp) > 0:
        om_row = omaha_comp.iloc[0]
        om_team = om_row.get('team_name', '?')
        om_year = int(om_row['year'])
        is_champ = CWS_CHAMPIONS.get(str(om_year)) == om_team
        om_badge = '\U0001F3C6 ' if is_champ else ''
        om_label = f'{om_badge}CWS: {om_team} ({om_year})'
        om_color = '#D4A017'  # gold
        om_vals = [float(omaha_pcts.loc[om_row.name, c]) if c in omaha_pcts.columns else 0.0 for c in metric_cols]
        om_vals_closed = om_vals + [om_vals[0]]
        ax.plot(angles_closed, om_vals_closed, color=om_color, linewidth=3.0,
                alpha=0.9, zorder=6, label=om_label, linestyle='-')
        ax.fill(angles_closed, om_vals_closed, color=om_color, alpha=0.10, zorder=5)
        ax.scatter(angles, om_vals, color=om_color, s=60, zorder=7,
                   edgecolors=bg_solid, linewidths=1.5, marker='D')

    # Target polygon — draw LAST and on TOP, with markers and bold styling
    target_vals = [float(target_pct[c]) if c in target_pct.index else 0.0 for c in metric_cols]
    target_vals_closed = target_vals + [target_vals[0]]
    # Solid heavy line with fill
    ax.fill(angles_closed, target_vals_closed, color=target_color, alpha=0.30, zorder=9)
    ax.plot(angles_closed, target_vals_closed, color=target_color, linewidth=4.5,
            zorder=10, label=f'{target_label} ({target_team})', solid_capstyle='round')
    # Big markers at each metric vertex
    ax.scatter(angles, target_vals, color=target_color, s=120, zorder=11,
               edgecolors=bg_solid, linewidths=2)

    # Axis styling
    ax.set_theta_offset(np.pi / 2)  # start at top
    ax.set_theta_direction(-1)       # clockwise
    ax.set_xticks(angles)
    ax.set_xticklabels(metric_labels, fontsize=12, fontfamily=BODY_FONT,
                       fontweight='bold', color=text_color)
    # Pad tick labels away from circle
    ax.tick_params(axis='x', pad=15)

    # Radial gridlines. Compute a dynamic floor for BOTH players and teams so
    # elite entities clustered in the 90-100 range don't all collapse onto the
    # same polygon. The floor is set to 5 below the lowest plotted value,
    # rounded down to the nearest 5, and is only applied when the result is
    # at or above 50 (below that, the full 0-100 scale is still informative).
    all_plotted_vals = list(target_vals)
    for _, m_row in matches.iterrows():
        m_vals = [float(match_pcts.loc[m_row.name, c]) if c in match_pcts.columns else 0.0 for c in metric_cols]
        all_plotted_vals.extend(m_vals)
    if omaha_comp is not None and omaha_pcts is not None and len(omaha_comp) > 0:
        om_r = omaha_comp.iloc[0]
        om_v = [float(omaha_pcts.loc[om_r.name, c]) if c in omaha_pcts.columns else 0.0 for c in metric_cols]
        all_plotted_vals.extend(om_v)

    min_val = min(all_plotted_vals) if all_plotted_vals else 0
    if min_val >= 50:
        r_floor = max(0, int(min_val // 5) * 5 - 5)
    else:
        r_floor = 0

    ax.set_ylim(r_floor, 100)

    # Choose tick spacing based on the visible range so we get round numbers.
    tick_range = 100 - r_floor
    if tick_range <= 10:
        step = 2
    elif tick_range <= 25:
        step = 5
    elif tick_range <= 50:
        step = 10
    else:
        step = 25
    yticks = list(range(int(r_floor), 101, step))
    if yticks[-1] != 100:
        yticks.append(100)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(t) for t in yticks], fontsize=8,
                       fontfamily=BODY_FONT, color=text_md)
    ax.set_rlabel_position(180 / n_metrics)  # offset radial labels between axes

    ax.grid(color=grid_color, alpha=0.5, linewidth=0.8)
    ax.spines['polar'].set_color(spine_color)
    ax.spines['polar'].set_linewidth(1.0)

    # Title + subtitle (clear space between title and subtitle, then space before chart)
    fig.text(0.5, 0.985, 'Similarity Profile', fontsize=20,
             fontfamily=TITLE_FONT, fontweight='bold',
             color=text_color, ha='center', va='top')
    fig.text(0.5, 0.945, 'Percentile within division (per year)',
             fontsize=11, fontfamily=SUBTITLE_FONT, color=text_md,
             ha='center', va='top')

    # Legend below chart
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
                       ncol=2, frameon=False, fontsize=10, labelcolor=text_color)
    for text in legend.get_texts():
        text.set_fontfamily(SUBTITLE_FONT)

    # Push the polar chart down so the top metric label doesn't crash into the subtitle
    fig.subplots_adjust(top=0.86, bottom=0.18)

    # Target team logo at the center of the radar
    team_map = load_team_logo_map()
    logo_id = team_map.get(target_team)
    if logo_id:
        for ext in ('png', 'webp'):
            center_logo_path = LOGO_DIR / f'{logo_id}.{ext}'
            if center_logo_path.exists():
                try:
                    center_img = Image.open(center_logo_path).convert('RGBA')
                    center_img.thumbnail((260, 260), Image.LANCZOS)
                    center_arr = np.array(center_img)
                    center_im = OffsetImage(center_arr, zoom=0.42, alpha=0.95)
                    # Place at center of the polar axes (axes fraction 0.5, 0.5)
                    center_ab = AnnotationBbox(center_im, (0.5, 0.5),
                                                xycoords=ax.transAxes,
                                                frameon=False, zorder=15,
                                                box_alignment=(0.5, 0.5))
                    ax.add_artist(center_ab)
                except Exception:
                    pass
                break

    # Brand logo bottom-right
    wide_logo_path = _APP_DIR / 'assets' / 'brand_logo_wide.png'
    if wide_logo_path.exists():
        try:
            logo_img = Image.open(wide_logo_path).convert('RGBA')
            logo_img.thumbnail((220, 60), Image.LANCZOS)
            logo_arr = np.array(logo_img)
            logo_im = OffsetImage(logo_arr, zoom=0.42, alpha=0.85)
            logo_ab = AnnotationBbox(logo_im, (0.97, 0.04), xycoords='figure fraction',
                                      frameon=False, zorder=20, box_alignment=(1, 0))
            fig.add_artist(logo_ab)
        except Exception:
            pass

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=180, facecolor=bg_solid, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='Similar Entities', layout='wide')
st.title('Similar Entities Finder')
st.markdown('Find players or teams with similar performance profiles within the same division.')

st.sidebar.markdown('### Setup')
sport = st.sidebar.selectbox('Sport', ['baseball', 'softball'])
division = st.sidebar.selectbox('Division', ['D1', 'D2', 'D3'])
mode = st.sidebar.radio('Mode', ['Player', 'Team'], horizontal=True)
theme = st.sidebar.radio('Theme', ['Light', 'Dark'], horizontal=True)
top_n = st.sidebar.number_input('Number of matches', value=5, min_value=3, max_value=15, step=1)

if mode == 'Player':
    hit_df = load_player_hitting(sport, division)
    pit_df = load_player_pitching(sport, division)

    # Identify two-way players (player_id appears in both)
    hitter_ids = set(hit_df['player_id'])
    pitcher_ids = set(pit_df['player_id'])
    two_way_ids = hitter_ids & pitcher_ids

    type_choice = st.sidebar.radio('Player Type', ['Hitter', 'Pitcher', 'Two-Way'], horizontal=True)

    if type_choice == 'Hitter':
        df = hit_df.copy()
        df['__display'] = df['player_name'].fillna('Unknown') + ' — ' + df['team_name'] + ' (' + df['year'].astype(str) + ')'
        df = df.sort_values('__display').reset_index(drop=True)
        metrics = HITTING_METRICS
        volume_col = 'plate_appearances'
        volume_label = 'PA'
    elif type_choice == 'Pitcher':
        df = pit_df.copy()
        df['__display'] = df['player_name'].fillna('Unknown') + ' — ' + df['team_name'] + ' (' + df['year'].astype(str) + ')'
        df = df.sort_values('__display').reset_index(drop=True)
        metrics = PITCHING_METRICS
        volume_col = 'batters_faced'
        volume_label = 'BF'
    else:  # Two-Way
        # Two-way detection: a (player_id, year) appearing in BOTH hitting and pitching
        h_keys = set(zip(hit_df['player_id'], hit_df['year']))
        p_keys = set(zip(pit_df['player_id'], pit_df['year']))
        two_way_keys = h_keys & p_keys
        if not two_way_keys:
            st.warning(f'No two-way players found in {sport.title()} {division}.')
            st.stop()
        h2 = hit_df[hit_df.set_index(['player_id', 'year']).index.isin(two_way_keys)].copy()
        p2 = pit_df[pit_df.set_index(['player_id', 'year']).index.isin(two_way_keys)].copy()
        # Merge on (player_id, year) so we don't cross years
        df = h2.merge(p2[['player_id', 'year'] + [m[0] for m in PITCHING_METRICS] + ['batters_faced']],
                      on=['player_id', 'year'], how='inner', suffixes=('', '_p'))
        # Overlapping columns (walk_percentage, strikeout_percentage, strikeout_to_walk_ratio)
        # got '_p' suffix on the pitching side. Replace the hitting versions with the
        # pitching versions since HITTING_METRICS doesn't reference these columns.
        for ov_col in ['walk_percentage', 'strikeout_percentage', 'strikeout_to_walk_ratio']:
            if ov_col + '_p' in df.columns:
                df[ov_col] = df[ov_col + '_p']
                df = df.drop(columns=[ov_col + '_p'])
        df['__display'] = df['player_name'].fillna('Unknown') + ' — ' + df['team_name'] + ' (' + df['year'].astype(str) + ')'
        df = df.sort_values('__display').reset_index(drop=True)
        metrics = HITTING_METRICS + PITCHING_METRICS
        volume_col = 'plate_appearances'  # use PA for volume weighting on two-way
        volume_label = 'PA'

    if len(df) == 0:
        st.warning('No players found.')
        st.stop()

    # Target selection
    target_display = st.selectbox(f'Select {type_choice}', df['__display'].tolist())
    target_row = df[df['__display'] == target_display].iloc[0]
    target_idx = target_row.name

    # Position-aware matching (hitters and two-way; pitcher position is uniform)
    pos_filter = None
    if type_choice in ('Hitter', 'Two-Way') and 'position' in df.columns:
        target_pos = target_row.get('position')
        match_by_position = st.sidebar.checkbox(
            'Match by position',
            value=True,
            help='Restrict matches to positionally-compatible players '
                 '(e.g. 1B ↔ 3B ↔ INF; SS ↔ 2B ↔ INF; OF within itself; '
                 'C only with C; DH with anyone).',
            key='match_by_pos')
        if match_by_position:
            pos_filter = compatible_positions(target_pos)
            if pos_filter is not None:
                st.sidebar.caption(
                    f"Target position: **{target_pos}** · "
                    f"Matching against: {', '.join(sorted(pos_filter))}"
                )
            else:
                st.sidebar.caption(
                    f"Target position: **{target_pos or 'unknown'}** · no filter applied"
                )

    # Find similar
    matches, all_pcts = find_similar(target_idx, df, metrics, volume_col=volume_col,
                                      top_n=top_n, position_filter=pos_filter)
    if matches is None or len(matches) == 0:
        st.warning('Could not compute similarity.')
        st.stop()

    target_pct = all_pcts.loc[target_idx]

    # Header card
    st.markdown(f"### {target_row['player_name']} — {target_row['team_name']} ({int(target_row['year'])})")
    cols = st.columns(len(metrics) + 1)
    cols[0].metric(volume_label, int(target_row[volume_col]))
    for i, (col, label, _) in enumerate(metrics):
        if col in target_row.index:
            raw = target_row[col]
            pct = target_pct.get(col, 0)
            cols[i + 1].metric(label, f'{raw:.3f}' if abs(raw) < 100 else f'{raw:.1f}', f'{pct:.0f} pct')

    st.markdown('---')
    st.markdown('### Top Matches')

    # Match table
    show_cols = ['year', 'player_name', 'team_name', volume_col] + [m[0] for m in metrics] + ['_distance']
    show_cols = [c for c in show_cols if c in matches.columns]
    rename = {'year': 'Year', 'player_name': 'Player', 'team_name': 'School',
              volume_col: volume_label, '_distance': 'Distance'}
    for col, label, _ in metrics:
        rename[col] = label
    display_df = matches[show_cols].rename(columns=rename).reset_index(drop=True)
    display_df.index = display_df.index + 1
    st.dataframe(display_df, use_container_width=True)

    # Bar chart
    target_label = f"{target_row['player_name']} {int(target_row['year'])}"
    target_team = target_row['team_name']
    chart = render_similarity_chart(target_label, target_team, target_pct, matches, all_pcts, metrics, theme=theme, show_year=True, entity_type='player')
    st.image(chart, use_container_width=True)
    st.download_button('Download Chart PNG', data=chart,
                       file_name=f'similar_{target_row["player_name"]}_{int(target_row["year"])}_{sport}_{division}.png', mime='image/png')

else:  # Team
    th = load_team_hitting(sport, division)
    tp = load_team_pitching(sport, division)
    # Merge on (team_id, year) so each team-season is its own row
    df = th.merge(tp[['team_id', 'year'] + [m[0] for m in PITCHING_METRICS]],
                  on=['team_id', 'year'], how='inner', suffixes=('', '_p'))
    # Overlapping columns (walk_percentage, strikeout_percentage, strikeout_to_walk_ratio)
    # got '_p' suffix on the pitching side. Replace the hitting versions with the
    # pitching versions since HITTING_METRICS doesn't reference these columns.
    for ov_col in ['walk_percentage', 'strikeout_percentage', 'strikeout_to_walk_ratio']:
        if ov_col + '_p' in df.columns:
            df[ov_col] = df[ov_col + '_p']
            df = df.drop(columns=[ov_col + '_p'])
    df['__display'] = df['team_name'] + ' (' + df['year'].astype(str) + ')'
    # Sort: most recent year first, then alphabetical
    df = df.sort_values(['year', 'team_name'], ascending=[False, True]).reset_index(drop=True)
    metrics = HITTING_METRICS + PITCHING_METRICS

    if len(df) == 0:
        st.warning('No teams found.')
        st.stop()

    team_mode = st.radio('Mode', ['Find Similar', 'Compare Teams'], horizontal=True, key='team_mode')

    if team_mode == 'Compare Teams':
        tc1, tc2 = st.columns(2)
        team_a_display = tc1.selectbox('Team A', df['__display'].tolist(), index=0, key='comp_a')
        team_b_display = tc2.selectbox('Team B', df['__display'].tolist(),
                                        index=min(1, len(df) - 1), key='comp_b')
        if team_a_display == team_b_display:
            st.warning('Pick two different team-seasons.')
            st.stop()

        row_a = df[df['__display'] == team_a_display].iloc[0]
        row_b = df[df['__display'] == team_b_display].iloc[0]
        all_pcts = compute_percentiles(df, metrics)

        pct_a = all_pcts.loc[row_a.name]
        pct_b = all_pcts.loc[row_b.name]

        st.markdown(f"### {row_a['team_name']} ({int(row_a['year'])}) vs {row_b['team_name']} ({int(row_b['year'])})")

        # Side-by-side metrics
        header_cols = st.columns([2] + [1] * min(len(metrics), 6) + [2])
        header_cols[0].markdown(f"**{row_a['team_name']} {int(row_a['year'])}**")
        for i, (col, label, _) in enumerate(metrics[:6]):
            header_cols[i + 1].caption(label)
        header_cols[-1].markdown(f"**{row_b['team_name']} {int(row_b['year'])}**")

        val_cols = st.columns([2] + [1] * min(len(metrics), 6) + [2])
        val_cols[0].write('')
        for i, (col, label, _) in enumerate(metrics[:6]):
            a_val = row_a.get(col, 0)
            b_val = row_b.get(col, 0)
            a_str = f'{a_val:.3f}' if abs(a_val) < 100 else f'{a_val:.1f}'
            b_str = f'{b_val:.3f}' if abs(b_val) < 100 else f'{b_val:.1f}'
            val_cols[i + 1].write(f'{a_str} / {b_str}')
        val_cols[-1].write('')

        st.markdown('---')

        # Build a "matches" df with just team B so render_similarity_chart works
        comp_matches = df.loc[[row_b.name]].copy()
        comp_matches['_distance'] = 0.0

        chart_label = f"{row_a['team_name']} {int(row_a['year'])}"
        chart = render_similarity_chart(chart_label, row_a['team_name'],
                                         pct_a, comp_matches, all_pcts, metrics,
                                         theme=theme, show_year=True, entity_type='team')
        st.image(chart, use_container_width=True)
        st.download_button('Download Comparison PNG', data=chart,
                           file_name=f'compare_{row_a["team_name"]}_{int(row_a["year"])}_vs_{row_b["team_name"]}_{int(row_b["year"])}.png',
                           mime='image/png')
        st.stop()

    target_display = st.selectbox('Select Team', df['__display'].tolist())
    target_row = df[df['__display'] == target_display].iloc[0]
    target_idx = target_row.name

    matches, all_pcts = find_similar(target_idx, df, metrics, volume_col=None, top_n=top_n)
    if matches is None or len(matches) == 0:
        st.warning('Could not compute similarity.')
        st.stop()

    target_pct = all_pcts.loc[target_idx]

    st.markdown(f"### {target_row['team_name']} ({int(target_row['year'])})")
    cols = st.columns(min(len(metrics), 6))
    for i, (col, label, _) in enumerate(metrics[:6]):
        if col in target_row.index:
            raw = target_row[col]
            pct = target_pct.get(col, 0)
            cols[i].metric(label, f'{raw:.3f}' if abs(raw) < 100 else f'{raw:.1f}', f'{pct:.0f} pct')

    st.markdown('---')
    st.markdown('### Top Matches')

    show_cols = ['year', 'team_name'] + [m[0] for m in metrics] + ['_distance']
    show_cols = [c for c in show_cols if c in matches.columns]
    rename = {'year': 'Year', 'team_name': 'Team', '_distance': 'Distance'}
    for col, label, _ in metrics:
        rename[col] = label
    display_df = matches[show_cols].rename(columns=rename).reset_index(drop=True)
    display_df.index = display_df.index + 1
    st.dataframe(display_df, use_container_width=True)

    # ── Closest Omaha Comp (baseball D1 only) ──
    omaha_match = None
    omaha_pcts = None
    if sport == 'baseball' and division == 'D1':
        omaha_match, omaha_pcts = find_closest_omaha(target_idx, df, metrics)
        if omaha_match is not None and len(omaha_match) > 0:
            om = omaha_match.iloc[0]
            om_name = om['team_name']
            om_year = int(om['year'])
            om_dist = om['_distance']
            is_champ = CWS_CHAMPIONS.get(str(om_year)) == om_name
            champ_badge = ' \U0001F3C6' if is_champ else ''

            st.markdown('---')
            st.markdown(f"### Closest Omaha Comp: {om_name} ({om_year}){champ_badge}")
            st.caption(f'Similarity distance: {om_dist:.1f} — based on hitting + pitching percentile profile')

            om_cols = st.columns(min(len(metrics), 6))
            om_pct = omaha_pcts.loc[om.name]
            for i, (col, label, _) in enumerate(metrics[:6]):
                if col in om.index:
                    raw = om[col]
                    pct_val = om_pct.get(col, 0)
                    om_cols[i].metric(label, f'{raw:.3f}' if abs(raw) < 100 else f'{raw:.1f}', f'{pct_val:.0f} pct')

    chart_label = f"{target_row['team_name']} {int(target_row['year'])}"
    chart = render_similarity_chart(chart_label, target_row['team_name'],
                                     target_pct, matches, all_pcts, metrics, theme=theme,
                                     show_year=True, entity_type='team',
                                     omaha_comp=omaha_match, omaha_pcts=omaha_pcts)
    st.image(chart, use_container_width=True)
    st.download_button('Download Chart PNG', data=chart,
                       file_name=f'similar_team_{target_row["team_name"]}_{int(target_row["year"])}_{sport}_{division}.png',
                       mime='image/png')
