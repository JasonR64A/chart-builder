"""
64 Analytics — Transfer Portal Impact
Dumbbell / dot-plot showing before vs after stats for players who transferred.
"""

import streamlit as st
from app_lib.safe_render import safe_savefig, safe_pyplot
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from io import BytesIO

# ── Path setup ───────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _APP_DIR / 'data'
LOGO_DIR = _APP_DIR / 'team_logos_512'
BRAND_LOGO_DARK = _APP_DIR / 'assets' / 'brand_logo_dark.png'

# ── Page config & styling ────────────────────────────────────────────────────
st.set_page_config(
    page_title='64 Analytics — Transfer Portal Impact',
    layout='wide',
    initial_sidebar_state='expanded',
)
st.markdown('''
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, .stApp, .stApp *:not([class*="icon"]):not([class*="Icon"]):not([data-testid*="icon"]):not([data-testid*="Icon"]):not([data-testid*="arrow"]):not(.material-icons):not(.material-symbols):not(.material-symbols-rounded):not([data-testid="stDataFrameSortIcon"]) {
    font-family: 'Inter', sans-serif !important;
}
.stApp { background-color: #1a1a1a; }
h1, h2, h3, p, label, .stMarkdown { color: #C8C8C8 !important; }
</style>
''', unsafe_allow_html=True)

# ── Fonts ────────────────────────────────────────────────────────────────────
def _has_font(name):
    return any(name.lower() in f.name.lower() for f in fm.fontManager.ttflist)

TITLE_FONT    = 'Franklin Gothic Heavy'  if _has_font('Franklin Gothic') else 'DejaVu Sans'
SUBTITLE_FONT = 'Franklin Gothic Medium' if _has_font('Franklin Gothic') else 'DejaVu Sans'
BODY_FONT     = 'Calibri' if _has_font('Calibri') else 'DejaVu Sans'

# ── Metric definitions ───────────────────────────────────────────────────────
HITTING_METRICS = {
    'OPS':  'on_base_plus_slugging',
    'wOBA': 'weighted_on_base_average',
    'BA':   'batting_average',
    'SLG':  'slugging_percentage',
    'ISO':  'isolated_power',
}

PITCHING_METRICS = {
    'ERA':         'earned_run_average',
    'FIP':         'fielding_independent_pitching',
    'WHIP':        'walks_plus_hits_per_inning_pitched',
    'K%':          'strikeout_percentage',
    'OPS Against': 'on_base_plus_slugging_against',
}

# For these pitching metrics, a *decrease* is an improvement
LOWER_IS_BETTER = {
    'earned_run_average',
    'fielding_independent_pitching',
    'walks_plus_hits_per_inning_pitched',
    'on_base_plus_slugging_against',
}

# ── Data loading (cached) ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_players():
    return pd.read_csv(DATA_DIR / 'players.csv', encoding='latin-1')

@st.cache_data(show_spinner=False)
def load_hitting():
    return pd.read_csv(DATA_DIR / 'hitting.csv')

@st.cache_data(show_spinner=False)
def load_pitching():
    return pd.read_csv(DATA_DIR / 'pitching.csv')

@st.cache_data(show_spinner=False)
def load_teams():
    return pd.read_csv(DATA_DIR / 'teams.csv')

@st.cache_data(show_spinner=False)
def load_conferences():
    return pd.read_csv(DATA_DIR / 'conferences.csv')


# ── Build transfer comparison dataframe ──────────────────────────────────────
@st.cache_data(show_spinner='Crunching transfer data …')
def build_transfer_data(sport, division, stat_type, metric_col, min_threshold):
    """
    Returns a DataFrame with one row per transfer player who has stats at both
    their old and new school, including before/after values and the delta.
    """
    players = load_players()
    teams   = load_teams()
    confs   = load_conferences()

    # Filter teams by sport
    sport_teams = teams[teams['sport'] == sport]
    sport_team_ids = set(sport_teams['id'].tolist())

    # Filter by division if needed
    if division != 'All':
        div_conf_ids = set(confs[confs['division'] == division]['id'].tolist())
        sport_teams = sport_teams[sport_teams['conference_id'].isin(div_conf_ids)]
        sport_team_ids = set(sport_teams['id'].tolist())

    # Get transfer players whose current team is in the sport
    transfers = players[
        (players['transferred'].astype(str).str.upper() == 'TRUE') &
        (players['team_id'].isin(sport_team_ids)) &
        (players['previous_team'].notna())
    ].copy()

    if transfers.empty:
        return pd.DataFrame()

    transfers['previous_team'] = transfers['previous_team'].astype(int)

    # Load stats
    if stat_type == 'Hitting':
        stats = load_hitting()
        threshold_col = 'plate_appearances'
    else:
        stats = load_pitching()
        threshold_col = 'innings_pitched'

    # Ensure numeric types for join columns
    stats['player_id'] = pd.to_numeric(stats['player_id'], errors='coerce')
    stats['team_id']   = pd.to_numeric(stats['team_id'],   errors='coerce')
    stats['year']      = pd.to_numeric(stats['year'],       errors='coerce')
    stats.dropna(subset=['player_id', 'team_id', 'year'], inplace=True)
    stats['player_id'] = stats['player_id'].astype(int)
    stats['team_id']   = stats['team_id'].astype(int)
    stats['year']      = stats['year'].astype(int)

    # Make sure the metric column is numeric
    if metric_col not in stats.columns:
        return pd.DataFrame()
    stats[metric_col] = pd.to_numeric(stats[metric_col], errors='coerce')
    if threshold_col in stats.columns:
        stats[threshold_col] = pd.to_numeric(stats[threshold_col], errors='coerce')

    # Build team name lookup
    team_name_map = dict(zip(sport_teams['id'], sport_teams['name']))
    # Also add teams from ALL sports so we can label the previous school
    all_team_name_map = dict(zip(teams['id'], teams['name']))

    rows = []
    for _, p in transfers.iterrows():
        pid = int(p['id'])
        prev_tid = int(p['previous_team'])
        curr_tid = int(p['team_id'])

        # Before stats: at previous team, most recent year
        before = stats[(stats['player_id'] == pid) & (stats['team_id'] == prev_tid)]
        if before.empty:
            continue
        before = before.sort_values('year', ascending=False).iloc[0]

        # Apply minimum threshold on "before" stats
        if threshold_col in stats.columns:
            thr_val = before.get(threshold_col, 0)
            if pd.isna(thr_val) or thr_val < min_threshold:
                continue

        # After stats: at current team, most recent year
        after = stats[(stats['player_id'] == pid) & (stats['team_id'] == curr_tid)]
        if after.empty:
            continue
        after = after.sort_values('year', ascending=False).iloc[0]

        before_val = before[metric_col]
        after_val  = after[metric_col]

        if pd.isna(before_val) or pd.isna(after_val):
            continue

        delta = after_val - before_val

        # Determine if this is an improvement
        if metric_col in LOWER_IS_BETTER:
            improved = delta < 0
        else:
            improved = delta > 0

        rows.append({
            'player_name':  p['player_name'],
            'position':     p.get('position', ''),
            'old_team_id':  prev_tid,
            'new_team_id':  curr_tid,
            'old_school':   all_team_name_map.get(prev_tid, f'Team {prev_tid}'),
            'new_school':   team_name_map.get(curr_tid, f'Team {curr_tid}'),
            'before':       round(before_val, 3),
            'after':        round(after_val, 3),
            'delta':        round(delta, 3),
            'abs_delta':    round(abs(delta), 3),
            'improved':     improved,
            'before_year':  int(before['year']),
            'after_year':   int(after['year']),
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ── Dumbbell chart ───────────────────────────────────────────────────────────
def make_dumbbell_chart(df, metric_label, metric_col, n_players=25):
    """
    Create a horizontal dumbbell/dot plot.
    Returns a matplotlib Figure.
    """
    n = min(n_players, len(df))
    plot_df = df.head(n).iloc[::-1].reset_index(drop=True)  # reverse for bottom-to-top

    fig_height = max(5, 0.45 * n + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    for i, row in plot_df.iterrows():
        color = '#4CAF50' if row['improved'] else '#E74C3C'
        # Connecting line
        ax.plot(
            [row['before'], row['after']], [i, i],
            color=color, linewidth=1.5, alpha=0.6, zorder=1,
        )
        # Before dot (muted gray)
        ax.scatter(
            row['before'], i, color='#888888', s=60, zorder=2,
            edgecolors='#AAAAAA', linewidths=0.5,
        )
        # After dot (green/red)
        ax.scatter(
            row['after'], i, color=color, s=80, zorder=3,
            edgecolors='white', linewidths=0.5,
        )

    # Y-axis labels: player name + school transition
    labels = []
    for _, row in plot_df.iterrows():
        sign = '+' if row['delta'] > 0 else ''
        labels.append(f"{row['player_name']}  ({sign}{row['delta']:.3f})")

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(labels, fontsize=8, fontfamily=BODY_FONT, color='#C8C8C8')

    # X-axis
    ax.tick_params(axis='x', colors='#888888', labelsize=9)
    ax.set_xlabel(metric_label, fontsize=10, fontfamily=BODY_FONT, color='#C8C8C8', labelpad=8)

    # Spines & grid
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, color='#2e2e2e', linewidth=0.5, alpha=0.7)
    ax.yaxis.grid(False)

    # Title
    direction_word = 'Movers' if df.iloc[0]['improved'] or df.iloc[0]['delta'] != 0 else 'Transfers'
    ax.set_title(
        f'Transfer Portal — {metric_label}',
        fontsize=16, fontfamily=TITLE_FONT, fontweight='bold',
        color='#FFFFFF', pad=14, loc='left',
    )

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='#888888', label='Before (old school)',
                   markerfacecolor='#888888', markersize=7, linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='#4CAF50', label='After (new school, improved)',
                   markerfacecolor='#4CAF50', markersize=7, linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='#E74C3C', label='After (new school, declined)',
                   markerfacecolor='#E74C3C', markersize=7, linestyle='None'),
    ]
    ax.legend(
        handles=legend_elements, loc='lower right', frameon=False,
        fontsize=8, labelcolor='#C8C8C8',
    )

    # Brand watermark
    if BRAND_LOGO_DARK.exists():
        try:
            from matplotlib.offsetbox import OffsetImage, AnnotationBbox
            from PIL import Image
            logo = Image.open(BRAND_LOGO_DARK).convert('RGBA')
            logo.thumbnail((120, 120))
            imagebox = OffsetImage(np.array(logo), zoom=0.35, alpha=0.5)
            ab = AnnotationBbox(
                imagebox, (0.98, 0.02), xycoords='axes fraction',
                frameon=False, box_alignment=(1, 0),
            )
            ax.add_artist(ab)
        except Exception:
            pass

    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig):
    buf = BytesIO()
    safe_savefig(fig, buf, format='png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Brand logo ───────────────────────────────────────────────────────────
    if BRAND_LOGO_DARK.exists():
        st.sidebar.image(str(BRAND_LOGO_DARK), use_container_width=True)

    st.sidebar.markdown('## Transfer Portal Impact')
    st.sidebar.markdown('---')

    # ── Sidebar controls ─────────────────────────────────────────────────────
    sport = st.sidebar.selectbox('Sport', ['Baseball', 'Softball'], index=0)

    division_options = ['All', 'D-I', 'D-II', 'D-III']
    division = st.sidebar.selectbox('Division', division_options, index=0)

    stat_type = st.sidebar.radio('Stat Type', ['Hitting', 'Pitching'], index=0)

    if stat_type == 'Hitting':
        metric_options = list(HITTING_METRICS.keys())
        metric_label = st.sidebar.selectbox('Key Metric', metric_options, index=0)
        metric_col = HITTING_METRICS[metric_label]
        min_val = st.sidebar.slider('Min PA (before)', 0, 200, 30, step=5)
    else:
        metric_options = list(PITCHING_METRICS.keys())
        metric_label = st.sidebar.selectbox('Key Metric', metric_options, index=0)
        metric_col = PITCHING_METRICS[metric_label]
        min_val = st.sidebar.slider('Min IP (before)', 0, 100, 10, step=5)

    sort_options = ['Biggest Improvement', 'Biggest Decline', 'Alphabetical']
    sort_by = st.sidebar.selectbox('Sort By', sort_options, index=0)

    n_display = st.sidebar.slider('Players to show in chart', 5, 50, 25, step=5)

    # ── Build data ───────────────────────────────────────────────────────────
    with st.spinner('Loading transfer data …'):
        df = build_transfer_data(sport, division, stat_type, metric_col, min_val)

    if df.empty:
        st.warning('No transfer data found for the selected filters. Try adjusting the sport, division, stat type, or minimum threshold.')
        return

    # ── Sort ─────────────────────────────────────────────────────────────────
    if sort_by == 'Biggest Improvement':
        if metric_col in LOWER_IS_BETTER:
            # Improvement = biggest negative delta
            df = df.sort_values('delta', ascending=True).reset_index(drop=True)
        else:
            # Improvement = biggest positive delta
            df = df.sort_values('delta', ascending=False).reset_index(drop=True)
    elif sort_by == 'Biggest Decline':
        if metric_col in LOWER_IS_BETTER:
            df = df.sort_values('delta', ascending=False).reset_index(drop=True)
        else:
            df = df.sort_values('delta', ascending=True).reset_index(drop=True)
    else:  # Alphabetical
        df = df.sort_values('player_name').reset_index(drop=True)

    # ── Summary stats ────────────────────────────────────────────────────────
    st.markdown(f'### Transfer Portal — {metric_label} ({sport}, {division})')

    total = len(df)
    avg_delta = df['delta'].mean()
    pct_improved = (df['improved'].sum() / total * 100) if total > 0 else 0
    pct_declined = 100 - pct_improved

    # Top improver: best delta in "improvement" direction
    if metric_col in LOWER_IS_BETTER:
        top_idx = df['delta'].idxmin()
    else:
        top_idx = df['delta'].idxmax()
    top_row = df.loc[top_idx]
    top_sign = '+' if top_row['delta'] > 0 else ''

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Transfers Analyzed', f'{total:,}')
    c2.metric(f'Avg Δ {metric_label}', f'{avg_delta:+.3f}')
    c3.metric('Improved / Declined', f'{pct_improved:.0f}% / {pct_declined:.0f}%')
    c4.metric(
        'Top Improver',
        top_row['player_name'],
        delta=f'{top_sign}{top_row["delta"]:.3f}',
        delta_color='normal' if top_row['improved'] else 'inverse',
    )

    st.markdown('---')

    # ── Dumbbell chart ───────────────────────────────────────────────────────
    fig = make_dumbbell_chart(df, metric_label, metric_col, n_players=n_display)
    safe_pyplot(fig, use_container_width=True)

    # ── Download buttons ─────────────────────────────────────────────────────
    dl_col1, dl_col2, _ = st.columns([1, 1, 3])

    png_bytes = fig_to_png_bytes(fig)
    dl_col1.download_button(
        label='📥 Download Chart (PNG)',
        data=png_bytes,
        file_name=f'transfer_portal_{metric_label.lower().replace(" ", "_")}_{sport.lower()}.png',
        mime='image/png',
    )

    csv_data = df.to_csv(index=False).encode('utf-8')
    dl_col2.download_button(
        label='📥 Download Data (CSV)',
        data=csv_data,
        file_name=f'transfer_portal_{metric_label.lower().replace(" ", "_")}_{sport.lower()}.csv',
        mime='text/csv',
    )

    plt.close(fig)

    # ── Data table ───────────────────────────────────────────────────────────
    st.markdown('---')
    st.markdown('### Full Transfer Data')

    display_df = df[[
        'player_name', 'position', 'old_school', 'new_school',
        'before', 'after', 'delta', 'improved', 'before_year', 'after_year',
    ]].copy()
    display_df.columns = [
        'Player', 'Pos', 'Old School', 'New School',
        f'{metric_label} (Before)', f'{metric_label} (After)',
        'Delta', 'Improved?', 'Before Year', 'After Year',
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        height=min(600, 35 * len(display_df) + 38),
    )


if __name__ == '__page__' or __name__ == '__main__':
    main()
