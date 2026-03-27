"""
64 Analytics — Player of the Period Lineup Card
Auto-selects the best players by position from PBP data.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

# ── Path setup ────────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent.parent
PBP_DIR = _APP_DIR / 'pbp_data'
BRAND_LOGO = _APP_DIR / 'assets' / 'brand_logo_dark.png'

# wOBA weights
WOBA_BB, WOBA_HBP, WOBA_1B, WOBA_2B, WOBA_3B, WOBA_HR = 0.690, 0.722, 0.888, 1.271, 1.616, 2.101
WOBA_SCALE = 1.0
FIP_CONSTANT = 3.0


# ── Position cleaning ────────────────────────────────────────────────────────
def _clean_position(pos):
    if not pos or (isinstance(pos, float) and np.isnan(pos)):
        return ''
    pos = str(pos).strip()
    if pos.startswith('/'):
        parts = pos.lstrip('/').split('/')
        return parts[0] if parts else pos
    return pos.split('/')[0]


# ── IP helpers ────────────────────────────────────────────────────────────────
def _ip_to_outs(ip_col):
    whole = np.floor(ip_col)
    frac = np.round((ip_col - whole) * 10).astype(int)
    return (whole * 3 + frac).astype(int)


def _outs_to_ip_display(total_outs):
    return f"{int(total_outs // 3)}.{int(total_outs % 3)}"


# ── Stat computation ─────────────────────────────────────────────────────────
def compute_hitting(df):
    ab = df['ab'].sum(); h = df['h'].sum(); bb = df['bb'].sum()
    hbp = df['hbp'].sum(); sf = df['sf'].sum(); sh = df['sh'].sum()
    tb = df['tb'].sum(); hr = df['hr'].sum()
    doubles = df['doubles'].sum(); triples = df['triples'].sum()
    k = df['k'].sum(); r = df['r'].sum(); rbi = df['rbi'].sum()
    sb = df['sb'].sum()
    singles = h - doubles - triples - hr
    pa = ab + bb + hbp + sf + sh
    obp_d = ab + bb + hbp + sf
    obp = (h + bb + hbp) / obp_d if obp_d > 0 else 0
    slg = tb / ab if ab > 0 else 0
    ba = h / ab if ab > 0 else 0
    woba_d = ab + bb + sf + hbp
    woba = (WOBA_BB*bb + WOBA_HBP*hbp + WOBA_1B*singles +
            WOBA_2B*doubles + WOBA_3B*triples + WOBA_HR*hr) / woba_d if woba_d > 0 else 0
    return {'PA': int(pa), 'AB': int(ab), 'H': int(h), 'HR': int(hr),
            'R': int(r), 'RBI': int(rbi), 'BB': int(bb), 'K': int(k), 'SB': int(sb),
            'BA': round(ba, 3), 'OBP': round(obp, 3), 'SLG': round(slg, 3),
            'OPS': round(obp + slg, 3), 'wOBA': round(woba, 3)}


def compute_pitching(df):
    total_outs = _ip_to_outs(df['ip']).sum()
    ip = total_outs / 3
    h = df['h'].sum(); bb = df['bb'].sum(); hb = df['hb'].sum()
    so = df['so'].sum(); hr = df['hrA'].sum(); bf = df['bf'].sum()
    er = df['er'].sum()
    games = df['gameId'].nunique() if 'gameId' in df.columns else len(df)
    fip = ((13*hr) + (3*(bb+hb)) - (2*so)) / ip + FIP_CONSTANT if ip > 0 else 99
    era = (er / ip) * 9 if ip > 0 else 99
    k_pct = so / bf if bf > 0 else 0
    return {'IP': _outs_to_ip_display(total_outs), 'IP_actual': ip,
            'BF': int(bf), 'H': int(h), 'ER': int(er), 'BB': int(bb),
            'SO': int(so), 'HR': int(hr), 'Games': int(games),
            'ERA': round(era, 2), 'FIP': round(fip, 2), 'K%': round(k_pct, 3)}


def compute_wraa(woba, league_woba, pa):
    return round(((woba - league_woba) / WOBA_SCALE) * pa, 1)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_pbp(sport, division, stat_type):
    filepath = PBP_DIR / sport / f'{stat_type}_pbp_{division}.csv'
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath, low_memory=False)
    if 'playerId' in df.columns:
        df['playerId'] = pd.to_numeric(df['playerId'], errors='coerce').fillna(0).astype(int).astype(str)
    if 'playerPosition' in df.columns:
        df['playerPosition'] = df['playerPosition'].apply(_clean_position)
    if 'date' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    return df


# ── Best player selection ─────────────────────────────────────────────────────
FIELD_POSITIONS = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH']


def get_best_hitters(hitting_df, league_woba, min_pa=10):
    """Get best hitter at each position by OPS."""
    best = {}
    for pos in FIELD_POSITIONS:
        pos_df = hitting_df[hitting_df['playerPosition'] == pos]
        if len(pos_df) == 0:
            continue
        rows = []
        for name, group in pos_df.groupby('playerName'):
            stats = compute_hitting(group)
            if stats['PA'] >= min_pa:
                stats['playerName'] = name
                stats['teamName'] = group['teamName'].mode().iloc[0] if len(group['teamName'].mode()) > 0 else ''
                stats['wRAA'] = compute_wraa(stats['wOBA'], league_woba, stats['PA'])
                rows.append(stats)
        if rows:
            df = pd.DataFrame(rows).sort_values('OPS', ascending=False)
            best[pos] = df.iloc[0].to_dict()
    return best


def get_best_pitchers(pitching_df, min_bf=15, n_starters=3, n_relievers=3):
    """Get best starters and relievers by FIP."""
    rows = []
    for name, group in pitching_df.groupby('playerName'):
        stats = compute_pitching(group)
        if stats['BF'] >= min_bf:
            stats['playerName'] = name
            stats['teamName'] = group['teamName'].mode().iloc[0] if len(group['teamName'].mode()) > 0 else ''
            # Classify: avg IP/game >= 3.0 = starter
            stats['is_starter'] = stats['IP_actual'] / max(stats['Games'], 1) >= 3.0
            rows.append(stats)
    if not rows:
        return [], []
    df = pd.DataFrame(rows)
    starters = df[df['is_starter']].sort_values('FIP').head(n_starters).to_dict('records')
    relievers = df[~df['is_starter']].sort_values('FIP').head(n_relievers).to_dict('records')
    return starters, relievers


# ── SVG rendering ─────────────────────────────────────────────────────────────
def _initials(name):
    parts = name.split()
    if len(parts) >= 2:
        return parts[0][0] + parts[-1][0]
    return name[:2].upper()


def _last_name(name):
    parts = name.split()
    return parts[-1] if parts else name


def _player_node(x, y, name, pos, ring_color='#C41230', r=22, r_inner=19,
                 text_color='#e8d0b0', name_color='#c8a880', pos_color='#9a8060', font_size=10):
    ini = _initials(name)
    last = _last_name(name)
    return f'''<g transform="translate({x},{y})">
    <circle r="{r}" fill="{ring_color}"/>
    <circle r="{r_inner}" fill="#1c2a38"/>
    <text font-size="{font_size}" font-weight="500" fill="{text_color}" text-anchor="middle"
          dominant-baseline="central" font-family="sans-serif">{ini}</text>
    <text font-size="8" fill="{name_color}" text-anchor="middle" y="{r+6}" font-family="sans-serif">{last}</text>
    <text font-size="7" fill="{pos_color}" text-anchor="middle" y="{r+15}" font-family="sans-serif">{pos}</text>
  </g>'''


def _pitcher_node(x, y, name, ring_color='#C41230', r=20, r_inner=17):
    ini = _initials(name)
    last = _last_name(name)
    return f'''<g transform="translate({x},{y})">
    <circle r="{r}" fill="{ring_color}"/>
    <circle r="{r_inner}" fill="#1c2a38"/>
    <text font-size="9" font-weight="500" fill="#e8d0b0" text-anchor="middle"
          dominant-baseline="central" font-family="sans-serif">{ini}</text>
    <text font-size="7.5" fill="#c8a880" text-anchor="middle" y="25" font-family="sans-serif">{last}</text>
  </g>'''


def _empty_node(x, y, pos, r=22, r_inner=19):
    return f'''<g transform="translate({x},{y})">
    <circle r="{r}" fill="#555"/>
    <circle r="{r_inner}" fill="#1c2a38"/>
    <text font-size="9" fill="#666" text-anchor="middle" dominant-baseline="central" font-family="sans-serif">—</text>
    <text font-size="7" fill="#666" text-anchor="middle" y="{r+15}" font-family="sans-serif">{pos}</text>
  </g>'''


# Position coordinates on the field SVG
POS_COORDS = {
    'CF': (230, 54), 'LF': (108, 127), 'RF': (352, 127),
    'SS': (196, 192), '2B': (264, 192),
    '3B': (152, 252), '1B': (308, 252),
    'C': (230, 307), 'DH': (230, 375),
}

RED = '#C41230'
DH_COLOR = '#22d3a0'
RELIEVER_COLOR = '#a855f7'


def render_lineup_svg(best_hitters, starters, relievers, title, subtitle):
    nodes = []

    # Field players
    for pos, (x, y) in POS_COORDS.items():
        if pos in best_hitters:
            p = best_hitters[pos]
            color = DH_COLOR if pos == 'DH' else RED
            nodes.append(_player_node(x, y, p['playerName'], pos, ring_color=color))
        else:
            nodes.append(_empty_node(x, y, pos))

    # Pitcher sidebar
    nodes.append('<line x1="416" y1="10" x2="416" y2="390" stroke="#3a3a3a" stroke-width="1"/>')
    nodes.append('<text x="438" y="26" font-size="8" fill="#a89880" text-anchor="middle" letter-spacing="0.08em" font-family="sans-serif">STARTERS</text>')

    for i, sp in enumerate(starters[:3]):
        nodes.append(_pitcher_node(438, 58 + i * 56, sp['playerName'], ring_color=RED))
    for i in range(len(starters), 3):
        nodes.append(f'<g transform="translate(438,{58 + i * 56})"><circle r="20" fill="#555"/><circle r="17" fill="#1c2a38"/><text font-size="9" fill="#666" text-anchor="middle" dominant-baseline="central" font-family="sans-serif">—</text></g>')

    nodes.append('<text x="438" y="218" font-size="8" fill="#a89880" text-anchor="middle" letter-spacing="0.08em" font-family="sans-serif">RELIEVERS</text>')

    for i, rp in enumerate(relievers[:3]):
        nodes.append(_pitcher_node(438, 246 + i * 56, rp['playerName'], ring_color=RELIEVER_COLOR))
    for i in range(len(relievers), 3):
        nodes.append(f'<g transform="translate(438,{246 + i * 56})"><circle r="20" fill="#555"/><circle r="17" fill="#1c2a38"/><text font-size="9" fill="#666" text-anchor="middle" dominant-baseline="central" font-family="sans-serif">—</text></g>')

    # Title
    title_node = f'''<text x="230" y="-10" font-size="14" font-weight="600" fill="#C8C8C8"
        text-anchor="middle" font-family="sans-serif">{title}</text>
    <text x="230" y="6" font-size="9" fill="#888" text-anchor="middle" font-family="sans-serif">{subtitle}</text>'''

    svg = f'''<svg width="100%" viewBox="0 -20 460 420" xmlns="http://www.w3.org/2000/svg">
  {title_node}
  <!-- Outfield grass -->
  <path d="M230,340 L50,90 Q230,0 410,90 Z" fill="#2d8a45"/>
  <path d="M50,90 Q230,0 410,90" fill="none" stroke="{RED}" stroke-width="10"/>
  <line x1="230" y1="340" x2="50" y2="90" stroke="{RED}" stroke-width="8"/>
  <line x1="230" y1="340" x2="410" y2="90" stroke="{RED}" stroke-width="8"/>
  <!-- Infield -->
  <path d="M230,340 L128,220 Q230,150 332,220 Z" fill="#c8883a"/>
  <path d="M230,340 L144,230 Q230,166 316,230 Z" fill="#2d8a45"/>
  <!-- Bases -->
  <rect x="222" y="168" width="16" height="16" rx="2" fill="#f5efe0" transform="rotate(45 230 176)"/>
  <rect x="308" y="228" width="14" height="14" rx="2" fill="#f5efe0" transform="rotate(45 315 235)"/>
  <rect x="148" y="228" width="14" height="14" rx="2" fill="#f5efe0" transform="rotate(45 155 235)"/>
  <polygon points="230,326 220,316 220,306 240,306 240,316" fill="#f5efe0"/>
  <!-- Mound -->
  <circle cx="230" cy="250" r="9" fill="#b87830" opacity="0.9"/>
  <circle cx="230" cy="250" r="4" fill="#a06820"/>
  {chr(10).join(nodes)}
</svg>'''
    return svg


def render_legend():
    return '''<div style="display:flex;gap:16px;justify-content:center;padding:6px 0;flex-wrap:wrap;">
  <div style="display:flex;align-items:center;gap:5px;font-size:11px;color:#888;">
    <div style="width:10px;height:10px;border-radius:50%;background:#C41230;"></div>Fielder / Starter</div>
  <div style="display:flex;align-items:center;gap:5px;font-size:11px;color:#888;">
    <div style="width:10px;height:10px;border-radius:50%;background:#22d3a0;"></div>DH</div>
  <div style="display:flex;align-items:center;gap:5px;font-size:11px;color:#888;">
    <div style="width:10px;height:10px;border-radius:50%;background:#a855f7;"></div>Reliever</div>
</div>'''


# ── Player detail card ────────────────────────────────────────────────────────
def render_hitter_card(player_dict):
    p = player_dict
    ini = _initials(p['playerName'])
    return f'''<div style="background:#222;border-radius:16px;border:1px solid #3a3a3a;padding:18px 16px;max-width:360px;margin:8px auto;font-family:sans-serif;">
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px;padding-bottom:14px;border-bottom:1px solid #3a3a3a;">
    <div style="width:52px;height:52px;border-radius:50%;background:#1c2a38;border:2.5px solid #C41230;display:flex;align-items:center;justify-content:center;font-size:15px;font-weight:500;color:#e8d0b0;flex-shrink:0;">{ini}</div>
    <div>
      <div style="font-size:15px;font-weight:500;color:#C8C8C8;">{p['playerName']}</div>
      <div style="font-size:11px;color:#888;margin-top:2px;">{p.get('teamName', '')} · {p.get('pos', '')}</div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:6px;">
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:17px;font-weight:500;color:#C41230;">{p['OPS']:.3f}</div><div style="font-size:9px;color:#888;margin-top:1px;">OPS</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:17px;font-weight:500;color:#C41230;">{p['wOBA']:.3f}</div><div style="font-size:9px;color:#888;margin-top:1px;">wOBA</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:17px;font-weight:500;color:#C41230;">{p['wRAA']:.1f}</div><div style="font-size:9px;color:#888;margin-top:1px;">wRAA</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:17px;font-weight:500;color:#C8C8C8;">{p['BA']:.3f}</div><div style="font-size:9px;color:#888;margin-top:1px;">AVG</div></div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;">
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:15px;font-weight:500;color:#C8C8C8;">{p['HR']}</div><div style="font-size:9px;color:#888;margin-top:1px;">HR</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:15px;font-weight:500;color:#C8C8C8;">{p['RBI']}</div><div style="font-size:9px;color:#888;margin-top:1px;">RBI</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:15px;font-weight:500;color:#C8C8C8;">{p['R']}</div><div style="font-size:9px;color:#888;margin-top:1px;">R</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:15px;font-weight:500;color:#C8C8C8;">{p['BB']}</div><div style="font-size:9px;color:#888;margin-top:1px;">BB</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:15px;font-weight:500;color:#C8C8C8;">{p['SB']}</div><div style="font-size:9px;color:#888;margin-top:1px;">SB</div></div>
  </div>
</div>'''


def render_pitcher_card(player_dict, role='Starter'):
    p = player_dict
    ini = _initials(p['playerName'])
    ring = RED if role == 'Starter' else RELIEVER_COLOR
    return f'''<div style="background:#222;border-radius:16px;border:1px solid #3a3a3a;padding:18px 16px;max-width:360px;margin:8px auto;font-family:sans-serif;">
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px;padding-bottom:14px;border-bottom:1px solid #3a3a3a;">
    <div style="width:52px;height:52px;border-radius:50%;background:#1c2a38;border:2.5px solid {ring};display:flex;align-items:center;justify-content:center;font-size:15px;font-weight:500;color:#e8d0b0;flex-shrink:0;">{ini}</div>
    <div>
      <div style="font-size:15px;font-weight:500;color:#C8C8C8;">{p['playerName']}</div>
      <div style="font-size:11px;color:#888;margin-top:2px;">{p.get('teamName', '')} · {role}</div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:6px;">
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:17px;font-weight:500;color:#C41230;">{p['ERA']:.2f}</div><div style="font-size:9px;color:#888;margin-top:1px;">ERA</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:17px;font-weight:500;color:#C41230;">{p['FIP']:.2f}</div><div style="font-size:9px;color:#888;margin-top:1px;">FIP</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:17px;font-weight:500;color:#C8C8C8;">{p['IP']}</div><div style="font-size:9px;color:#888;margin-top:1px;">IP</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:17px;font-weight:500;color:#C8C8C8;">{p['K%']:.3f}</div><div style="font-size:9px;color:#888;margin-top:1px;">K%</div></div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;">
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:15px;font-weight:500;color:#C8C8C8;">{p['SO']}</div><div style="font-size:9px;color:#888;margin-top:1px;">SO</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:15px;font-weight:500;color:#C8C8C8;">{p['BB']}</div><div style="font-size:9px;color:#888;margin-top:1px;">BB</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:15px;font-weight:500;color:#C8C8C8;">{p['H']}</div><div style="font-size:9px;color:#888;margin-top:1px;">H</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:15px;font-weight:500;color:#C8C8C8;">{p['HR']}</div><div style="font-size:9px;color:#888;margin-top:1px;">HR</div></div>
    <div style="background:#2a2a2a;border-radius:8px;padding:8px 4px;text-align:center;"><div style="font-size:15px;font-weight:500;color:#C8C8C8;">{p['Games']}</div><div style="font-size:9px;color:#888;margin-top:1px;">G</div></div>
  </div>
</div>'''


# ── Main ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='64 Analytics — Lineup Card', layout='wide',
                   initial_sidebar_state='expanded')

st.markdown("""
<style>
.stApp { background-color: #1a1a1a; }
h1, h2, h3, p, label, .stMarkdown { color: #C8C8C8 !important; }
</style>
""", unsafe_allow_html=True)

st.sidebar.image(str(BRAND_LOGO), width=80)
st.sidebar.markdown('## Lineup Card')

# Controls
st.sidebar.markdown('---')
sport = st.sidebar.selectbox('Sport', ['baseball', 'softball'])
division = st.sidebar.selectbox('Division', ['D1', 'D2', 'D3'])

period = st.sidebar.selectbox('Time Period', [
    'Last 7 Days', 'Last 14 Days', 'Last 30 Days', 'Season', 'Custom'
])

# Load data
hitting = load_pbp(sport, division, 'hitting')
pitching = load_pbp(sport, division, 'pitching')

if hitting is None or pitching is None:
    st.error(f'PBP data not found for {sport} {division}')
    st.stop()

# Apply time filter
if 'date_parsed' in hitting.columns and period != 'Season':
    max_date = hitting['date_parsed'].max()
    if period == 'Custom':
        min_date = hitting['date_parsed'].min()
        date_range = st.sidebar.date_input('Date range',
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(), max_value=max_date.date())
        if len(date_range) == 2:
            start, end = date_range
            hitting = hitting[(hitting['date_parsed'].dt.date >= start) & (hitting['date_parsed'].dt.date <= end)]
            pitching = pitching[(pitching['date_parsed'].dt.date >= start) & (pitching['date_parsed'].dt.date <= end)]
            period_label = f"{start.strftime('%b %d')} – {end.strftime('%b %d, %Y')}"
        else:
            period_label = 'Custom'
    else:
        days = {'Last 7 Days': 7, 'Last 14 Days': 14, 'Last 30 Days': 30}[period]
        cutoff = max_date - timedelta(days=days)
        hitting = hitting[hitting['date_parsed'] >= cutoff]
        pitching = pitching[pitching['date_parsed'] >= cutoff]
        period_label = f"{cutoff.strftime('%b %d')} – {max_date.strftime('%b %d, %Y')}"
else:
    period_label = 'Full Season'

min_pa = st.sidebar.number_input('Min PA (hitters)', value=10, min_value=1, step=5)
min_bf = st.sidebar.number_input('Min BF (pitchers)', value=15, min_value=1, step=5)

st.sidebar.markdown(f'**{len(hitting):,}** hitting lines · **{len(pitching):,}** pitching lines')

if len(hitting) == 0 or len(pitching) == 0:
    st.warning('No data for selected period.')
    st.stop()

# Compute
league_stats = compute_hitting(hitting)
league_woba = league_stats['wOBA']
best_hitters = get_best_hitters(hitting, league_woba, min_pa=min_pa)
starters, relievers = get_best_pitchers(pitching, min_bf=min_bf)

# Render lineup card
title = f"Player of the {period.replace('Last ', '').replace('Season', 'Season')}"
subtitle = f"{sport.title()} {division} · {period_label}"

svg = render_lineup_svg(best_hitters, starters, relievers, title, subtitle)
st.markdown(svg + render_legend(), unsafe_allow_html=True)

# Detail cards below
st.markdown('---')
st.markdown('### Position Players')

cols = st.columns(3)
for i, pos in enumerate(FIELD_POSITIONS):
    if pos in best_hitters:
        p = best_hitters[pos]
        p['pos'] = pos
        with cols[i % 3]:
            st.markdown(render_hitter_card(p), unsafe_allow_html=True)

st.markdown('### Pitchers')
cols2 = st.columns(3)
for i, sp in enumerate(starters[:3]):
    with cols2[i]:
        st.markdown(render_pitcher_card(sp, 'Starter'), unsafe_allow_html=True)

if relievers:
    cols3 = st.columns(3)
    for i, rp in enumerate(relievers[:3]):
        with cols3[i]:
            st.markdown(render_pitcher_card(rp, 'Reliever'), unsafe_allow_html=True)
