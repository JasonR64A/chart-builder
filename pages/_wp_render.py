"""Reusable Win Probability render helpers.

Lets any Streamlit page compute + render a static WP chart for a specific
gameId using the same model the Win Probability page uses. Designed to be
imported (not a Streamlit page itself).

Usage:
    from pages._wp_render import (
        load_wp_pbp, load_wp_lookup, load_wp_teams,
        compute_wp_for_game, build_wp_figure,
    )
    from pages._win_prob_model import build_team_profiles

    pbp   = load_wp_pbp('Baseball', 'D1')
    lookup= load_wp_lookup()
    tp,tr = load_wp_teams('Baseball', 'D1')
    prof  = build_team_profiles('Baseball', 'D1')

    data  = compute_wp_for_game(pbp, gid, tp, tr, prof, lookup, sport='Baseball')
    fig   = build_wp_figure(data)
    st.plotly_chart(fig, use_container_width=True,
                    config={'staticPlot': True, 'displayModeBar': False})
"""
import re
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

from pages._win_prob_model import (
    pre_game_wp as _shared_pre_game_wp,
    adjusted_team_pct as _shared_adjusted_team_pct,
    blend_with_static as _shared_blend_with_static,
)

_APP_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = _APP_DIR / 'data'
PBP_DIR     = _APP_DIR / 'pbp_data' / 'play_by_play'
LOOKUP_FILE = _APP_DIR / 'pbp_data' / 'wp_state_lookup_bb_d1.pkl'

CLAMP_MIN = 0.01
CLAMP_MAX = 0.99

BG_COLOR   = '#FAF8F2'
TEXT_COLOR = '#2D2926'
TEXT_MUTED = '#4A4540'
GRID_COLOR = '#D8D2C4'
HOME_COLOR = '#2E5C8A'
AWAY_COLOR = '#C41230'

DIVISION_LABELS = {'D1': 'D-I', 'D2': 'D-II', 'D3': 'D-III'}


def _pbp_paths(sport, division):
    base = PBP_DIR / f'{sport.lower()}_play_by_play_{division}'
    return base.with_suffix('.csv'), base.with_suffix('.csv.gz')


@st.cache_data
def load_wp_pbp(sport='Baseball', division='D1'):
    cols = ['gameId', 'date', 'awayTeam', 'homeTeam', 'inning', 'halfInning',
            'battingTeam', 'outs', 'runner1B', 'runner2B', 'runner3B',
            'awayScore', 'homeScore', 'player', 'playDescription']
    csv, gz = _pbp_paths(sport, division)
    if csv.exists():
        df = pd.read_csv(csv, low_memory=False, usecols=cols)
    elif gz.exists():
        df = pd.read_csv(gz, low_memory=False, usecols=cols, compression='gzip')
    else:
        return None
    df['inning']    = pd.to_numeric(df['inning'], errors='coerce').fillna(0).astype(int)
    df['outs']      = pd.to_numeric(df['outs'], errors='coerce').fillna(0).astype(int)
    df['awayScore'] = pd.to_numeric(df['awayScore'], errors='coerce').fillna(0).astype(int)
    df['homeScore'] = pd.to_numeric(df['homeScore'], errors='coerce').fillna(0).astype(int)
    df['half_true'] = np.where(df['battingTeam'] == df['awayTeam'], 'top', 'bottom')
    return df


@st.cache_data
def load_wp_lookup():
    if not LOOKUP_FILE.exists():
        return {}
    with open(LOOKUP_FILE, 'rb') as f:
        return pickle.load(f)


@st.cache_data
def load_wp_teams(sport='Baseball', division='D1'):
    t  = pd.read_csv(DATA_DIR / 'teams.csv', low_memory=False)
    t['id'] = pd.to_numeric(t['id'], errors='coerce').fillna(0).astype(int)
    tr = pd.read_csv(DATA_DIR / 'team_rank.csv', low_memory=False)
    tr = tr[tr['year'] == 2026].copy()
    tr['rank']  = pd.to_numeric(tr['integer_64_rank_total'], errors='coerce')
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
    name_to_pct  = dict(zip(bucket.set_index('id').loc[tr_b['team_id'].values, 'name'],
                             tr_b['rank_pct']))
    name_to_rank = dict(zip(bucket.set_index('id').loc[tr_b['team_id'].values, 'name'],
                              tr_b['rank'].astype(int)))
    return name_to_pct, name_to_rank


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
    best_k, best_len = None, 0
    for k in team_pct.keys():
        if name.startswith(k) and len(k) > best_len:
            best_k, best_len = k, len(k)
        elif k.startswith(short_team(name)) and len(short_team(name)) > best_len:
            best_k, best_len = k, len(short_team(name))
    if best_k:
        return best_k, team_pct[best_k], team_rank.get(best_k)
    return name, None, None


def state_key(row, is_softball=False):
    raw_inning = min(int(row['inning']), 10) if pd.notna(row['inning']) else 1
    if is_softball and raw_inning <= 7:
        inning = min(10, round(raw_inning * 9 / 7))
    else:
        inning = raw_inning
    half = 1 if row.get('halfInning') == 'bottom' else 0
    outs = min(int(row['outs']), 2) if pd.notna(row['outs']) else 0
    bases = int(pd.notna(row.get('runner1B'))) + \
            int(pd.notna(row.get('runner2B'))) * 2 + \
            int(pd.notna(row.get('runner3B'))) * 4
    sd = int(row['homeScore']) - int(row['awayScore'])
    sd = max(-10, min(10, sd))
    return (inning, half, outs, bases, sd)


_SUB_RE = re.compile(r'\bto\b.+\bfor\b|\bpinch (ran|hit) for\b', re.IGNORECASE)


def _is_substitution(row, prev):
    desc = str(row.get('playDescription', '') or '')
    if not _SUB_RE.search(desc):
        return False
    if prev is None:
        return False
    return (int(row['outs']) == int(prev['outs']) and
            int(row['awayScore']) == int(prev['awayScore']) and
            int(row['homeScore']) == int(prev['homeScore']))


def compute_wp_for_game(pbp_df, game_id, team_pct, team_rank, profiles,
                         lookup, sport='Baseball'):
    """Compute WP curve for a single game. Returns a dict of plot-ready fields, or None."""
    game = pbp_df[pbp_df['gameId'] == game_id].copy()
    if game.empty:
        return None

    # Fix top/bottom ordering (scraper writes bottom before top within inning)
    game['_half_order'] = (game['half_true'] == 'bottom').astype(int)
    game = (game.sort_values(['inning', '_half_order'], kind='stable')
                 .drop(columns=['_half_order']).reset_index(drop=True))

    home = game['homeTeam'].iloc[0]
    away = game['awayTeam'].iloc[0]
    if not isinstance(home, str) or not isinstance(away, str):
        return None
    date = game['date'].iloc[0]

    home_key, home_p, _ = find_team(home, team_pct, team_rank)
    away_key, away_p, _ = find_team(away, team_pct, team_rank)

    # Detect Weekend vs Midweek + game number within the 4-day matchup window
    sel_date = pd.to_datetime(date, format='mixed', errors='coerce')
    series_games = pbp_df[((pbp_df['homeTeam'] == home) & (pbp_df['awayTeam'] == away)) |
                           ((pbp_df['homeTeam'] == away) & (pbp_df['awayTeam'] == home))].copy()
    series_games['_d'] = pd.to_datetime(series_games['date'], format='mixed', errors='coerce')
    if pd.notna(sel_date):
        series_games = series_games[(series_games['_d'] >= sel_date - pd.Timedelta(days=2)) &
                                     (series_games['_d'] <= sel_date + pd.Timedelta(days=2))]
    series_ids = (series_games.drop_duplicates('gameId')
                              .sort_values(['_d', 'gameId'])['gameId'].tolist())
    if len(series_ids) >= 2:
        game_type = 'Weekend'
        try:
            game_num = max(1, min(3, series_ids.index(game_id) + 1))
        except ValueError:
            game_num = 1
    else:
        game_type = 'Midweek'
        game_num  = 0

    # Pre-game WP from blended talent
    home_p_player = _shared_adjusted_team_pct(home_key, profiles, game_type, game_num, home_p, sport=sport) if home_p else home_p
    away_p_player = _shared_adjusted_team_pct(away_key, profiles, game_type, game_num, away_p, sport=sport) if away_p else away_p
    home_p_adj    = _shared_blend_with_static(home_p_player, home_p)
    away_p_adj    = _shared_blend_with_static(away_p_player, away_p)
    pg_home       = _shared_pre_game_wp(home_p_adj, away_p_adj) if home_p_adj and away_p_adj else 0.5

    # Real plays for blend weight
    rows = list(game.iterrows())
    prev = None
    is_real = []
    real_count = 0
    for _, row in rows:
        if _is_substitution(row, prev):
            is_real.append(False)
        else:
            is_real.append(True)
            real_count += 1
        prev = row
    total = max(1, real_count)

    is_softball = (sport == 'Softball')
    wp_curve = [pg_home]
    real_idx = 0
    for i, (_, row) in enumerate(rows):
        k = state_key(row, is_softball=is_softball)
        info = lookup.get(k)
        if is_real[i]:
            real_idx += 1
        w = min(1.0, real_idx / total)
        if info is None or info[1] < 10:
            wp = pg_home
        else:
            sw = info[0]
            wp = w * sw + (1 - w) * pg_home
        wp_curve.append(max(CLAMP_MIN, min(CLAMP_MAX, wp)))

    # Lock final to actual result
    fh = int(game['homeScore'].max())
    fa = int(game['awayScore'].max())
    if fh != fa:
        wp_curve[-1] = 1.0 if fh > fa else 0.0

    return {
        'game': game,
        'home': home,
        'away': away,
        'home_key': home_key,
        'away_key': away_key,
        'date': date,
        'wp_curve': wp_curve,
        'pg_home': pg_home,
        'final_home_score': fh,
        'final_away_score': fa,
    }


def _rgba(hex_color, alpha):
    h = hex_color.lstrip('#')
    return f'rgba({int(h[:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})'


def build_wp_figure(wp_data, home_color=None, away_color=None, height=320):
    """Build a minimal static-friendly WP figure. No play-picker, no logos."""
    if wp_data is None:
        return None
    game     = wp_data['game']
    home     = wp_data['home']
    away     = wp_data['away']
    wp_curve = wp_data['wp_curve']
    pg_home  = wp_data['pg_home']
    date     = wp_data['date']

    # Inning tick positions
    inning_labels    = []
    inning_positions = []
    last = None
    for i, row in game.iterrows():
        ih = (int(row['inning']), row['half_true'])
        if ih != last:
            inning_positions.append(i + 1)
            half_abbr = 'T' if row['half_true'] == 'top' else 'B'
            inning_labels.append(f"{half_abbr}{int(row['inning'])}")
            last = ih

    x_indices = list(range(len(wp_curve)))
    home_wps  = [w * 100 for w in wp_curve]

    hc = home_color or HOME_COLOR
    ac = away_color or AWAY_COLOR
    INVIS = 'rgba(0,0,0,0)'

    fig = go.Figure()

    # Home-ahead fill
    fig.add_trace(go.Scatter(x=x_indices, y=[50] * len(x_indices), mode='lines',
                              line=dict(color=INVIS, width=0),
                              showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x_indices,
                              y=[w if w > 50 else 50 for w in home_wps],
                              fill='tonexty', fillcolor=_rgba(hc, 0.55),
                              mode='lines', line=dict(color=INVIS, width=0),
                              name=home, showlegend=True, hoverinfo='skip'))

    # Away-ahead fill
    fig.add_trace(go.Scatter(x=x_indices, y=[50] * len(x_indices), mode='lines',
                              line=dict(color=INVIS, width=0),
                              showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x_indices,
                              y=[w if w < 50 else 50 for w in home_wps],
                              fill='tonexty', fillcolor=_rgba(ac, 0.55),
                              mode='lines', line=dict(color=INVIS, width=0),
                              name=away, showlegend=True, hoverinfo='skip'))

    # 50% baseline
    fig.add_hline(y=50, line_dash='dash', line_color='#888', line_width=1, opacity=0.6)

    # WP curve
    fig.add_trace(go.Scatter(x=x_indices, y=home_wps, mode='lines',
                              line=dict(color=TEXT_COLOR, width=2.2),
                              name='Win Probability', showlegend=False,
                              hoverinfo='skip'))

    # Inning dividers
    for pos in inning_positions[1:]:
        fig.add_vline(x=pos, line_dash='dot', line_color='#bbb',
                      line_width=0.5, opacity=0.4)

    fig.update_layout(
        plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR,
        height=height, margin=dict(t=65, b=40, l=55, r=20),
        xaxis=dict(
            title=None, tickmode='array',
            tickvals=inning_positions, ticktext=inning_labels,
            tickangle=0, color=TEXT_COLOR,
            tickfont=dict(size=10, color=TEXT_COLOR),
            gridcolor=GRID_COLOR, gridwidth=0.3,
            showgrid=False, zeroline=False,
        ),
        yaxis=dict(
            title=None, range=[0, 100],
            tickvals=[0, 25, 50, 75, 100],
            ticktext=['0%', '25%', '50%', '75%', '100%'],
            color=TEXT_MUTED, gridcolor=GRID_COLOR,
            gridwidth=0.3, zeroline=False,
        ),
        annotations=[
            dict(x=0.5, y=1.18, xref='paper', yref='paper', showarrow=False,
                 text=f'<b>{away}</b> @ <b>{home}</b>  ·  {date}',
                 font=dict(size=13, color=TEXT_COLOR, family='Arial Black')),
            dict(x=0.5, y=1.06, xref='paper', yref='paper', showarrow=False,
                 text=f'Pre-game: {pg_home*100:.1f}%  →  Final: {home_wps[-1]:.1f}%',
                 font=dict(size=10, color=TEXT_MUTED)),
        ],
        legend=dict(orientation='h', y=-0.18, x=0.5, xanchor='center',
                    font=dict(color=TEXT_COLOR, size=10),
                    bgcolor='rgba(0,0,0,0)'),
    )
    return fig
