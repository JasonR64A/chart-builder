"""
Spray Charts — visualize where batted balls go for a team or player.

Uses the play_by_play data's hitLocation column (NCAA position codes 1-14)
to render a baseball-field heatmap, a bar chart, a result-by-zone matrix,
and a Left/Middle/Right side-split summary.
"""
import streamlit as st
import pandas as pd
import base64
import io
from pathlib import Path

import sys
_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR))
from app_lib.spray_data import (
    compute_spray_distribution,
    compute_field_side_buckets,
    add_zone_metrics,
    list_teams,
    list_players,
    ZONE_COORDS,
    ZONE_NAMES,
)

BRAND_LOGO     = _APP_DIR / 'assets' / 'logo-circle-black.png'
BRAND_64A_WIDE = _APP_DIR / 'assets' / 'logo-64a-wide.png'

st.set_page_config(page_title='Spray Charts — 64 Analytics', layout='wide')

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    if BRAND_LOGO.exists():
        st.image(str(BRAND_LOGO), width=180)
    st.markdown('## Spray Charts')

    sport = st.selectbox('Sport', ['baseball', 'softball'], format_func=str.title)
    division = st.selectbox('Division', ['D1', 'D2', 'D3'])

    view_mode = st.radio('View', ['Team', 'Player', 'Team Grid'], horizontal=True,
                         help='Team / Player render a single chart. '
                              'Team Grid renders a 3x3 array of the team\'s nine '
                              'highest-BIP batters with the same single 64A logo on top.')

    teams_df = list_teams(sport, division)
    if teams_df.empty:
        st.error(f'No PBP data found for {sport} {division}. '
                 f'Expected pbp_data/play_by_play/{sport}_play_by_play_{division}.csv '
                 f'(or .csv.gz on Render).')
        st.stop()

    # Build dropdown labels with team_id when known: "Texas Longhorns (id 856 · 5,510 BIP)"
    def team_label(row):
        tid = f"id {int(row['team_id'])}" if pd.notna(row['team_id']) else 'no id'
        return f"{row['ncaa_team']}  ({tid} · {int(row['bip']):,} BIP)"
    teams_df = teams_df.copy()
    teams_df['_label'] = teams_df.apply(team_label, axis=1)
    options = ['(all teams)'] + teams_df['_label'].tolist()
    name_by_label = dict(zip(teams_df['_label'], teams_df['ncaa_team']))
    selected_label = st.selectbox('Team', options)
    team_filter = None if selected_label == '(all teams)' else name_by_label.get(selected_label)

    # Resolve team_id and the short teams.csv name for the selected team
    selected_team_id = None
    selected_team_short = team_filter   # fallback to PBP name if no match
    if team_filter is not None:
        match = teams_df[teams_df['ncaa_team'] == team_filter]
        if not match.empty:
            row = match.iloc[0]
            if pd.notna(row['team_id']):
                selected_team_id = int(row['team_id'])
            if 'short_name' in match.columns and pd.notna(row.get('short_name')):
                selected_team_short = str(row['short_name'])

    # Team Grid mode requires a team selection
    if view_mode == 'Team Grid' and team_filter is None:
        st.warning('Pick a team to see the 3x3 grid of its top 9 batters.')
        st.stop()

    selected_player_id = None
    selected_player_name = None         # display name (full from players.csv if available)
    selected_player_position = None
    selected_player_class = None
    if view_mode == 'Player':
        plist = list_players(sport, division, team_filter)
        if plist.empty:
            st.warning('No players with batted balls found for this scope.')
            st.stop()
        head = plist.head(80).reset_index(drop=True)
        # Prefer the full name from players.csv; fall back to PBP last-name spelling
        def _disp_name(r):
            full = r.get('player_name')
            if isinstance(full, str) and full.strip():
                return full
            return str(r['player'])
        labels = [f"{_disp_name(r)}  ({r['balls_in_play']} BIP)"
                  for _, r in head.iterrows()]
        choice = st.selectbox('Player', range(len(labels)), format_func=lambda i: labels[i])
        selected_player_id = head.loc[choice, 'playerId']
        selected_player_name = _disp_name(head.loc[choice])
        selected_player_position = head.loc[choice, 'position'] if 'position' in head.columns else None
        selected_player_class = head.loc[choice, 'classification'] if 'classification' in head.columns else None

    st.markdown('---')
    st.markdown('### Filters')
    f_hand = st.radio('Pitcher hand', ['Any', 'vs LHP', 'vs RHP'], horizontal=True)
    f_two_strikes = st.checkbox('2-strike counts only')
    f_two_outs = st.checkbox('2-out situations only')
    f_risp = st.checkbox('Runners in scoring position')

vs_hand = {'vs LHP': 'L', 'vs RHP': 'R'}.get(f_hand)

# ── Compute ──────────────────────────────────────────────────────────────────
spray = compute_spray_distribution(
    sport, division,
    team_name=team_filter if view_mode == 'Team' else None,
    player_id=selected_player_id if view_mode == 'Player' else None,
    vs_hand=vs_hand,
    two_strikes=f_two_strikes,
    two_outs=f_two_outs,
    risp=f_risp,
)
spray = add_zone_metrics(spray)
buckets = compute_field_side_buckets(spray)

# ── Header ───────────────────────────────────────────────────────────────────
title_scope = (
    f"Player: {selected_player_name}" if view_mode == 'Player'
    else (f"Team: {selected_team_short}" if selected_team_short else f"All {division} {sport}")
)
st.markdown(f"## {sport.title()} {division} — Spray Chart")
st.caption(title_scope + f" · {buckets['total']:,} balls in play")

if spray.empty:
    st.warning('No batted-ball data for this selection.')
    st.stop()

# ── Metric toggle ───────────────────────────────────────────────────────────
METRIC_OPTIONS = ['% of BIP', 'AVG', 'SLG', 'wOBA', 'TB']
metric_choice = st.radio(
    'Color & label by:', METRIC_OPTIONS, horizontal=True,
    help=('% of BIP = share of balls in play landing in that zone. '
          'AVG/SLG/wOBA are conditional on the ball reaching that zone '
          '(denominator = zone BIP). TB = raw total bases.'),
)

def _metric_value(row, choice: str) -> float:
    if choice == '% of BIP': return float(row['pct'])
    if choice == 'TB':       return float(row['TB'])
    return float(row[choice])  # AVG / SLG / wOBA

def _metric_fmt(v: float, choice: str) -> str:
    if choice == '% of BIP': return f"{v:.0f}%"
    if choice == 'TB':       return f"{int(v)}"
    # AVG/SLG/wOBA — drop the leading zero baseball-style (.350 not 0.350)
    if 0 <= v < 1:           return f".{int(round(v*1000)):03d}"
    return f"{v:.3f}"


# ── Team Grid mode ──────────────────────────────────────────────────────────
if view_mode == 'Team Grid':
    import math, base64
    HEADSHOT_DIR = _APP_DIR / 'assets' / 'player_headshots'

    plist = list_players(sport, division, team_filter)
    if plist.empty:
        st.error(f'No players with batted balls found for {selected_team_short}.')
        st.stop()
    top9 = plist.head(9).reset_index(drop=True)

    def _disp_name(r):
        full = r.get('player_name')
        if isinstance(full, str) and full.strip():
            return full
        return str(r['player'])

    # Compute each player's spray with the same filter set
    cells = []
    for _, p in top9.iterrows():
        sp = compute_spray_distribution(
            sport, division,
            team_name=team_filter, player_id=p['playerId'],
            vs_hand=vs_hand, two_strikes=f_two_strikes,
            two_outs=f_two_outs, risp=f_risp,
        )
        sp = add_zone_metrics(sp)
        from app_lib.spray_data import compute_field_side_buckets as _cfsb
        b = _cfsb(sp)
        cells.append({
            'name': _disp_name(p),
            'pos':  p.get('position') if isinstance(p.get('position'), str) else '',
            'cls':  p.get('classification') if isinstance(p.get('classification'), str) else '',
            'cb_id': int(p['cb_id']) if pd.notna(p.get('cb_id')) else None,
            'spray': sp,
            'buckets': b,
        })

    # ── Cell renderer ────────────────────────────────────────────────────────
    # Cell viewBox 0..100 × 0..90, with all geometry in cell-local coords. The
    # parent SVG translates each cell to its grid position.
    C_HOME    = (50, 76)
    C_RINNER  = 7
    C_RMID    = 22
    C_ROUTER  = 42
    C_LINE_HALF = 15
    C_OF = [
        ((7, 10),  -45, -9),
        ((8, 14),   -9,  9),
        ((9, 11),    9, 45),
    ]
    C_IF = [
        ((5,), -45,   -22.5),
        ((6,), -22.5,   0),
        ((4,),   0,    22.5),
        ((3,),  22.5,  45),
    ]
    C_LN = [
        ((12,), -45 - C_LINE_HALF, -45),
        ((13,),  45,  45 + C_LINE_HALF),
    ]

    def _cpolar(a, r):
        rad = math.radians(a)
        return (C_HOME[0] + r * math.sin(rad), C_HOME[1] - r * math.cos(rad))

    def _cwedge(a1, a2, ri, ro):
        x1i, y1i = _cpolar(a1, ri); x1o, y1o = _cpolar(a1, ro)
        x2i, y2i = _cpolar(a2, ri); x2o, y2o = _cpolar(a2, ro)
        return (f"M {x1i:.2f},{y1i:.2f} L {x1o:.2f},{y1o:.2f} "
                f"A {ro},{ro} 0 0 1 {x2o:.2f},{y2o:.2f} "
                f"L {x2i:.2f},{y2i:.2f} A {ri},{ri} 0 0 0 {x1i:.2f},{y1i:.2f} Z")

    def _cpie(a1, a2, ro):
        x1, y1 = _cpolar(a1, ro); x2, y2 = _cpolar(a2, ro)
        return (f"M {C_HOME[0]:.2f},{C_HOME[1]:.2f} L {x1:.2f},{y1:.2f} "
                f"A {ro},{ro} 0 0 1 {x2:.2f},{y2:.2f} Z")

    def _clabelpos(a1, a2, ri, ro):
        ang = (a1 + a2) / 2; r = (ri + ro) / 2
        return _cpolar(ang, r)

    def _ro(intensity):
        i = max(0.05, min(1.0, intensity))
        r = int(255 - (255 - 198) * i)
        g = int(224 - (224 - 40)  * i)
        b = int(204 - (204 - 40)  * i)
        return f'#{r:02X}{g:02X}{b:02X}'
    def _tc(intensity):
        return '#FFFFFF' if intensity > 0.55 else '#0F2A4D'

    def _ccombined(spray, codes):
        sub = spray[spray['hitLocation'].isin(codes)]
        if sub.empty: return None
        n = int(sub['total'].sum())
        if n == 0: return None
        get = lambda c: int(sub[c].sum()) if c in sub.columns else 0
        c1, c2, c3, ch = get('1B'), get('2B'), get('3B'), get('HR')
        hits = c1 + c2 + c3 + ch
        tb = c1 + 2*c2 + 3*c3 + 4*ch
        woba = 0.888*c1 + 1.271*c2 + 1.616*c3 + 2.101*ch
        return {'BIP': n, 'pct': float(sub['pct'].sum()),
                '1B': c1, '2B': c2, '3B': c3, 'HR': ch,
                'AVG': hits/n, 'SLG': tb/n, 'wOBA': woba/n, 'TB': tb}

    def _compact(n):
        n = int(n)
        if abs(n) >= 1_000_000: return f'{n/1_000_000:.1f}M'
        if abs(n) >= 10_000:    return f'{n/1_000:.0f}K'
        return f'{n:,}'

    def _initials(name: str) -> str:
        parts = [w for w in str(name).split() if w]
        if not parts: return '?'
        if len(parts) == 1: return parts[0][:2].upper()
        return (parts[0][0] + parts[-1][0]).upper()

    def _embed_headshot_circle(cb_id, cx, cy, r):
        """Profile bubble: PNG from assets/player_headshots/{cb_id}.png if it
        exists, otherwise a styled circle with the player's initials."""
        if cb_id is not None:
            path = HEADSHOT_DIR / f'{cb_id}.png'
            if path.exists():
                try:
                    b64 = base64.b64encode(path.read_bytes()).decode('ascii')
                    href = f'data:image/png;base64,{b64}'
                    clip_id = f'clip_p_{cb_id}'
                    return (
                        f'<defs><clipPath id="{clip_id}">'
                        f'<circle cx="{cx}" cy="{cy}" r="{r}"/></clipPath></defs>'
                        f'<image href="{href}" xlink:href="{href}" '
                        f'x="{cx - r}" y="{cy - r}" width="{r*2}" height="{r*2}" '
                        f'preserveAspectRatio="xMidYMid slice" '
                        f'clip-path="url(#{clip_id})"/>'
                        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" '
                        f'stroke="#0F2A4D" stroke-width="0.5"/>'
                    )
                except Exception:
                    pass
        return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#F0EAD6" '
                f'stroke="#0F2A4D" stroke-width="0.5"/>'
                f'<text x="{cx}" y="{cy + r*0.3:.2f}" text-anchor="middle" '
                f'font-family="Inter,sans-serif" font-size="{r*0.9:.1f}" '
                f'font-weight="800" fill="#0F2A4D" opacity="0.5"></text>')

    def _build_cell(c) -> str:
        spray = c['spray']
        b = c['buckets']
        parts_c = []

        # Cell frame — eggshell background + border so each cell reads as
        # its own card. The 2-unit inset on each side leaves a thin gutter
        # between cells when the parent grid lays them out flush.
        parts_c.append(
            '<rect x="2" y="2" width="96" height="91" rx="3" ry="3" '
            'fill="#F0EAD6" stroke="#0F2A4D" stroke-width="0.4"/>'
        )

        # Header strip — name on left, pos · class on right
        full_label = c['name']
        sub_bits = ' · '.join(x for x in [c['pos'], c['cls']] if x)
        parts_c.append(
            f'<text x="5" y="8" font-family="Inter,sans-serif" font-size="4.2" '
            f'font-weight="800" fill="#0F2A4D">{full_label}</text>'
        )
        if sub_bits:
            parts_c.append(
                f'<text x="95" y="8" text-anchor="end" font-family="Inter,sans-serif" '
                f'font-size="2.8" font-weight="600" fill="#0F2A4D">{sub_bits}</text>'
            )

        if spray is None or spray.empty:
            parts_c.append('<text x="50" y="50" text-anchor="middle" '
                           'font-family="Inter,sans-serif" font-size="3" '
                           'font-weight="600" fill="#0F2A4D">No batted-ball data</text>')
            return ''.join(parts_c)

        # Per-zone metric values
        val_by, fmt_by = {}, {}
        for zc, *_ in C_OF + C_IF + C_LN + [((1,),0,0), ((2,),0,0)]:
            bd = _ccombined(spray, zc)
            primary = zc[0]
            if bd is None:
                val_by[primary] = 0.0; fmt_by[primary] = '–'
            else:
                v = bd['pct'] if metric_choice == '% of BIP' else (
                    float(bd['TB']) if metric_choice == 'TB' else bd[metric_choice])
                val_by[primary] = v
                fmt_by[primary] = _metric_fmt(v, metric_choice)
        fair_max = max((val_by.get(zc[0], 0) for zc, *_ in C_OF + C_IF + C_LN),
                       default=1.0) or 1.0

        # Top-left mini-diamond (L/M/R splits) — sized 2.5x larger so the
        # split percentages are readable even on the team-grid page.
        m_home = (22, 33); m_r = 21.0
        side_pcts = {'L': float(b['left_pct']),
                     'C': float(b['middle_pct']),
                     'R': float(b['right_pct'])}
        side_max = max(side_pcts.values()) or 1.0
        SIDE = [('L', -45, -15), ('C', -15, 15), ('R', 15, 45)]
        def _mp(a1, a2, r):
            r1, r2 = math.radians(a1), math.radians(a2)
            x1 = m_home[0] + r * math.sin(r1); y1 = m_home[1] - r * math.cos(r1)
            x2 = m_home[0] + r * math.sin(r2); y2 = m_home[1] - r * math.cos(r2)
            return (f"M {m_home[0]},{m_home[1]} L {x1:.2f},{y1:.2f} "
                    f"A {r},{r} 0 0 1 {x2:.2f},{y2:.2f} Z")
        for k, a1, a2 in SIDE:
            pct = side_pcts[k]
            inten = pct / side_max
            parts_c.append(f'<path d="{_mp(a1, a2, m_r)}" fill="{_ro(inten)}" '
                           f'stroke="#FFFFFF" stroke-width="0.5"/>')
            mid = math.radians((a1 + a2) / 2)
            lx = m_home[0] + (m_r * 0.72) * math.sin(mid)
            ly = m_home[1] - (m_r * 0.72) * math.cos(mid)
            parts_c.append(
                f'<text x="{lx:.1f}" y="{ly+0.6:.1f}" text-anchor="middle" '
                f'font-family="Inter,sans-serif" font-size="2.4" font-weight="800" '
                f'fill="{_tc(inten)}">{pct:.0f}%</text>'
            )
        parts_c.append(f'<path d="{_mp(-45, 45, m_r)}" fill="none" '
                       f'stroke="#0F2A4D" stroke-width="0.5"/>')

        # Top-right hit-type 2x2 grid (1B/2B top, 3B/HR bottom)
        get = lambda col: int(spray[col].sum()) if col in spray.columns else 0
        ht = [[('1B', get('1B')), ('2B', get('2B'))],
              [('3B', get('3B')), ('HR', get('HR'))]]
        tr_x0, tr_y0 = 75, 11; cw, ch = 11.5, 5.5
        for ri, row in enumerate(ht):
            for ci, (lab, val) in enumerate(row):
                cx = tr_x0 + cw/2 + ci * cw
                cy = tr_y0 + ri * ch
                parts_c.append(
                    f'<text x="{cx:.1f}" y="{cy+1.5:.1f}" text-anchor="middle" '
                    f'font-family="Inter,sans-serif" font-size="1.3" font-weight="600" '
                    f'fill="#0F2A4D">{lab}</text>'
                )
                parts_c.append(
                    f'<text x="{cx:.1f}" y="{cy+4.7:.1f}" text-anchor="middle" '
                    f'font-family="Inter,sans-serif" font-size="2.4" font-weight="800" '
                    f'fill="#0F2A4D">{_compact(val)}</text>'
                )

        # Main diamond wedges
        def _draw(zc, a1, a2, ri, ro, lf, *, pie=False):
            primary = zc[0]
            v = val_by.get(primary, 0)
            inten = v / fair_max if fair_max else 0
            d = _cpie(a1, a2, ro) if pie else _cwedge(a1, a2, ri, ro)
            parts_c.append(f'<path d="{d}" fill="{_ro(inten)}" '
                           f'stroke="#FFFFFF" stroke-width="0.4"/>')
            lx, ly = _clabelpos(a1, a2, ri, ro)
            parts_c.append(
                f'<text x="{lx:.1f}" y="{ly+0.7:.1f}" text-anchor="middle" '
                f'font-family="Inter,sans-serif" font-size="{lf}" font-weight="800" '
                f'fill="{_tc(inten)}">{fmt_by.get(primary, "")}</text>'
            )
        for zc, a1, a2 in C_OF: _draw(zc, a1, a2, C_RMID, C_ROUTER, 3.5)
        for zc, a1, a2 in C_LN: _draw(zc, a1, a2, C_RINNER, C_ROUTER, 2.6, pie=True)
        for zc, a1, a2 in C_IF: _draw(zc, a1, a2, C_RINNER, C_RMID, 2.0)

        # Pitcher circle + label
        p_pct = fmt_by.get(1, '')
        px, py = C_HOME[0], C_HOME[1] - 4.5
        parts_c.append(f'<circle cx="{px}" cy="{py}" r="2.6" fill="#F8E8E2" '
                       f'stroke="#0F2A4D" stroke-width="0.4"/>')
        parts_c.append(f'<text x="{px}" y="{py+0.8:.1f}" text-anchor="middle" '
                       f'font-family="Inter,sans-serif" font-size="2.0" font-weight="800" '
                       f'fill="#0F2A4D">{p_pct}</text>')

        # Catcher pentagon at apex
        c_pct = fmt_by.get(2, '')
        pw, ph = 2.5, 2.3
        cx, cy = C_HOME[0], C_HOME[1] + ph
        plate = [(cx-pw, cy-ph), (cx+pw, cy-ph),
                 (cx+pw, cy+ph*0.10), (cx, cy+ph), (cx-pw, cy+ph*0.10)]
        parts_c.append('<polygon points="' + ' '.join(f'{x:.2f},{y:.2f}' for x,y in plate)
                       + '" fill="#F8E8E2" stroke="#0F2A4D" stroke-width="0.3"/>')
        parts_c.append(f'<text x="{cx}" y="{cy+0.4:.1f}" text-anchor="middle" '
                       f'font-family="Inter,sans-serif" font-size="1.4" font-weight="800" '
                       f'fill="#0F2A4D">{c_pct}</text>')

        # Field outline
        fo = 45 + C_LINE_HALF
        fxL, fyL = _cpolar(-fo, C_ROUTER); fxR, fyR = _cpolar(fo, C_ROUTER)
        parts_c.append(f'<path d="M {C_HOME[0]},{C_HOME[1]} L {fxL:.2f},{fyL:.2f} '
                       f'A {C_ROUTER},{C_ROUTER} 0 0 1 {fxR:.2f},{fyR:.2f} Z" '
                       f'fill="none" stroke="#0F2A4D" stroke-width="0.5"/>')

        # Bottom-left aggregate stats — 2x2 grid mirroring the upper-right
        # hit-type layout. AVG/SLG on the top row, wOBA/TB on the bottom.
        n_bip = int(spray['total'].sum())
        c1 = get('1B'); c2 = get('2B'); c3 = get('3B'); chh = get('HR')
        hits = c1 + c2 + c3 + chh
        tb_t = c1 + 2*c2 + 3*c3 + 4*chh
        woba_n = 0.888*c1 + 1.271*c2 + 1.616*c3 + 2.101*chh
        avg = hits/n_bip if n_bip else 0
        slg = tb_t/n_bip if n_bip else 0
        woba = woba_n/n_bip if n_bip else 0
        overall = [('AVG', _metric_fmt(avg, 'AVG')),
                   ('SLG', _metric_fmt(slg, 'SLG')),
                   ('wOBA', _metric_fmt(woba, 'wOBA')),
                   ('TB', _compact(tb_t))]
        bl_x0 = 4
        bl_cell_w, bl_cell_h = 22, 6
        bl_y0 = 80
        for i, (lab, val) in enumerate(overall):
            row, col = divmod(i, 2)
            cxv = bl_x0 + bl_cell_w/2 + col * bl_cell_w
            cyv = bl_y0 + row * bl_cell_h
            parts_c.append(
                f'<text x="{cxv:.1f}" y="{cyv + 1.6:.1f}" text-anchor="middle" '
                f'font-family="Inter,sans-serif" font-size="2.8" font-weight="600" '
                f'fill="#0F2A4D">{lab}</text>'
            )
            parts_c.append(
                f'<text x="{cxv:.1f}" y="{cyv + 5.1:.1f}" text-anchor="middle" '
                f'font-family="Inter,sans-serif" font-size="4.2" font-weight="800" '
                f'fill="#0F2A4D">{val}</text>'
            )

        # Bottom-right profile bubble
        parts_c.append(_embed_headshot_circle(c['cb_id'], cx=85, cy=85, r=7.0))

        return ''.join(parts_c)

    # ── Build parent SVG ─────────────────────────────────────────────────────
    HDR_H = 20
    CELL_W, CELL_H = 100, 95
    COLS, ROWS = 3, 3
    GRID_W = COLS * CELL_W
    GRID_H = ROWS * CELL_H
    VB_W = GRID_W + 4
    VB_H = HDR_H + GRID_H + 4

    parts_g = [
        f'<svg viewBox="0 0 {VB_W} {VB_H}" width="{VB_W}" height="{VB_H}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink">'
        f'<rect x="0" y="0" width="{VB_W}" height="{VB_H}" fill="#FFFFFF"/>'
    ]

    def _embed(path, x, y, w, h, opacity=1.0):
        if not path.exists(): return ''
        try:
            b64 = base64.b64encode(path.read_bytes()).decode('ascii')
        except Exception:
            return ''
        href = f'data:image/png;base64,{b64}'
        return (f'<image href="{href}" xlink:href="{href}" '
                f'x="{x}" y="{y}" width="{w}" height="{h}" opacity="{opacity}" '
                f'preserveAspectRatio="xMidYMid meet"/>')

    # 64A logo top-center
    parts_g.append(_embed(BRAND_64A_WIDE, x=VB_W/2 - 18, y=2, w=36, h=8.5))

    # Team name + scope below logo
    sport_label = 'Baseball' if sport.lower() == 'baseball' else 'Softball'
    fb = []
    if vs_hand: fb.append('vs LHP' if vs_hand == 'L' else 'vs RHP')
    if f_two_strikes: fb.append('2 strikes')
    if f_two_outs:    fb.append('2 outs')
    if f_risp:        fb.append('RISP')
    fbtxt = ' · ' + ', '.join(fb) if fb else ''
    parts_g.append(
        f'<text x="{VB_W/2}" y="16" text-anchor="middle" font-family="Inter,sans-serif" '
        f'font-size="3.3" font-weight="700" fill="#0F2A4D">'
        f'{selected_team_short} · {sport_label} {division}{fbtxt} · '
        f'Top 9 by BIP · Coloring by {metric_choice}</text>'
    )

    # 3x3 grid
    GRID_X0 = (VB_W - GRID_W) / 2
    for i, c in enumerate(cells):
        row, col = divmod(i, COLS)
        ox = GRID_X0 + col * CELL_W
        oy = HDR_H + row * CELL_H
        parts_g.append(f'<g transform="translate({ox},{oy})">')
        parts_g.append(_build_cell(c))
        parts_g.append('</g>')
    # pad empty cells
    for i in range(len(cells), 9):
        row, col = divmod(i, COLS)
        ox = GRID_X0 + col * CELL_W
        oy = HDR_H + row * CELL_H
        parts_g.append(
            f'<rect x="{ox+2}" y="{oy+2}" width="96" height="91" '
            f'rx="3" ry="3" fill="#F8F8F8" stroke="#D0C9B0" stroke-width="0.3"/>'
        )

    parts_g.append('</svg>')
    grid_svg = ''.join(parts_g)

    display_grid = grid_svg.replace(
        '<svg ',
        '<svg style="width:100%;max-width:1200px;height:auto;display:block;'
        'margin:0 auto;border-radius:8px;" ', 1,
    )
    st.markdown(display_grid, unsafe_allow_html=True)

    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(bytestring=grid_svg.encode('utf-8'), output_width=2200)
        safe_team = ''.join(ch if ch.isalnum() else '_' for ch in str(selected_team_short))[:40]
        st.download_button('Download PNG', data=png_bytes,
                           file_name=f'spray_team_{sport}_{division}_{safe_team}.png',
                           mime='image/png', use_container_width=False)
    except Exception as e:
        st.caption(f'PNG export unavailable in this environment ({type(e).__name__}).')

    st.markdown('---')
    st.markdown('### Top 9 batters by balls in play')
    rows_t = []
    for c in cells:
        sp = c['spray']
        if sp is None or sp.empty:
            rows_t.append({'Player': c['name'], 'Pos': c['pos'], 'Class': c['cls'],
                           'BIP': 0, 'AVG': '-', 'SLG': '-', 'wOBA': '-', 'TB': 0})
            continue
        get = lambda col: int(sp[col].sum()) if col in sp.columns else 0
        n = int(sp['total'].sum())
        c1 = get('1B'); c2 = get('2B'); c3 = get('3B'); ch = get('HR')
        hits = c1 + c2 + c3 + ch; tb = c1 + 2*c2 + 3*c3 + 4*ch
        wn = 0.888*c1 + 1.271*c2 + 1.616*c3 + 2.101*ch
        rows_t.append({
            'Player': c['name'], 'Pos': c['pos'], 'Class': c['cls'],
            'BIP': n,
            'AVG': _metric_fmt(hits/n if n else 0, 'AVG'),
            'SLG': _metric_fmt(tb/n if n else 0, 'SLG'),
            'wOBA': _metric_fmt(wn/n if n else 0, 'wOBA'),
            'TB': tb,
        })
    st.dataframe(pd.DataFrame(rows_t), hide_index=True, use_container_width=True)
    st.caption(
        f'Headshots: drop a PNG named `{{cb_id}}.png` into '
        f'`assets/player_headshots/` (size ~256×256, square crop). '
        f'When the PNG exists for a cb_id, it replaces the placeholder circle.'
    )
    st.stop()


# ── Field-diagram SVG (heatmap) + summary side-by-side ──────────────────────
col_field, col_summary = st.columns([3, 2])

with col_field:
    # ── Wedge-based spray-chart heatmap ────────────────────────────────────
    # Layout: home plate at apex (bottom). Fair territory = 90° wedge from
    # -45° to +45°. Outer ring = 5 outfield wedges (CF merges 8+14). Inner
    # ring = 4 infield wedges (3B/SS/2B/1B). Slim down-the-line wedges sit
    # just outside ±45°. Pitcher = centerline circle, catcher = home-plate
    # pentagon at the apex. Single red/orange ramp across all fair zones.
    import math

    def _combined(zone_codes):
        """Sum raw counts across one or more zones and recompute rate stats
        from the totals. Returns None if no BIP."""
        sub = spray[spray['hitLocation'].isin(zone_codes)]
        if sub.empty:
            return None
        n = int(sub['total'].sum())
        if n == 0:
            return None
        get = lambda c: int(sub[c].sum()) if c in sub.columns else 0
        c1, c2, c3, ch = get('1B'), get('2B'), get('3B'), get('HR')
        hits = c1 + c2 + c3 + ch
        tb = c1 + 2*c2 + 3*c3 + 4*ch
        woba = 0.888*c1 + 1.271*c2 + 1.616*c3 + 2.101*ch
        return {
            'BIP': n, 'pct': float(sub['pct'].sum()),
            '1B': c1, '2B': c2, '3B': c3, 'HR': ch,
            'Out': get('Out'), 'FC/Err': get('FC/Err'),
            'AVG': hits / n, 'SLG': tb / n, 'wOBA': woba / n, 'TB': tb,
        }

    def _value_for(bd, choice):
        if choice == '% of BIP': return bd['pct']
        if choice == 'TB':       return float(bd['TB'])
        return bd[choice]

    def _label_for(zone_codes):
        if zone_codes == (8, 14): return 'CF + Deep CF'
        if zone_codes == (7, 10): return 'LF + LCF'
        if zone_codes == (9, 11): return 'RF + RCF'
        return ZONE_NAMES.get(zone_codes[0], str(zone_codes[0]))

    def _tooltip_for(zone_codes) -> str:
        bd = _combined(zone_codes)
        if bd is None:
            return f"{_label_for(zone_codes)}: no BIP"
        zlist = '+'.join(str(z) for z in zone_codes)
        bits = [f"{_label_for(zone_codes)} (zone {zlist})",
                f"BIP: {bd['BIP']}  ({bd['pct']:.1f}%)"]
        for c in ['1B','2B','3B','HR','Out','FC/Err']:
            if bd.get(c, 0) > 0:
                bits.append(f"{c}: {bd[c]}")
        bits.append(f"AVG {_metric_fmt(bd['AVG'],'AVG')} · "
                    f"SLG {_metric_fmt(bd['SLG'],'SLG')} · "
                    f"wOBA {_metric_fmt(bd['wOBA'],'wOBA')} · "
                    f"TB {bd['TB']}")
        return '\n'.join(bits)

    # SVG geometry — viewBox 0 0 100 72 leaves the line wedges' outer
    # corners (at polar(-65°, 48)) just inside the left/right edges.
    HOME = (50, 65)
    R_INNER = 8    # pitcher / inner apex
    R_MID   = 25   # infield / outfield boundary
    R_OUTER = 48   # field edge
    # Angles are in degrees from straight up (toward CF). Negative = left.
    # Zones 10 (LCF) and 11 (RCF) are infrequently scored independently
    # of LF/RF, so they're merged into the corresponding flank wedges:
    # LF wedge = zones 7+10, RF wedge = zones 9+11. CF wedge already
    # combines 8 (CF) + 14 (Deep CF). The three merged outfield wedges
    # give a more honest read on the chart at the cost of one less zone
    # of granularity — granular zone detail still lives in the tables.
    OUTFIELD = [   # (zone_codes, ang_start, ang_end)
        ((7, 10),  -45, -9),    # LF + LCF
        ((8, 14),   -9,  9),    # CF + Deep CF
        ((9, 11),    9, 45),    # RF + RCF
    ]
    INFIELD = [
        ((5,), -45,   -22.5),
        ((6,), -22.5,   0),
        ((4,),   0,    22.5),
        ((3,),  22.5,  45),
    ]
    # "Down the line" wedges — NCAA codes 12/13 are FAIR balls that landed
    # close to the foul line (often XBH), not actual foul balls. Slim 20°
    # slivers just outside the foul line; their outer edges align with the
    # field-outline foul lines.
    LINE_HALF = 15
    LINE = [
        ((12,), -45 - LINE_HALF, -45),
        ((13,),  45,  45 + LINE_HALF),
    ]

    # Build value/format lookups for every wedge group + pitcher + catcher.
    # Must come AFTER the geometry (OUTFIELD/INFIELD/LINE) is defined.
    breakdowns = {}
    val_by_zone = {}
    fmt_by_zone = {}
    for zone_codes, *_ in OUTFIELD + INFIELD + LINE + [((1,), 0, 0), ((2,), 0, 0)]:
        bd = _combined(zone_codes)
        primary = zone_codes[0]
        breakdowns[primary] = (zone_codes, bd)
        if bd is None:
            val_by_zone[primary] = 0.0
            fmt_by_zone[primary] = '–'
        else:
            v = _value_for(bd, metric_choice)
            val_by_zone[primary] = v
            fmt_by_zone[primary] = _metric_fmt(v, metric_choice)

    def polar(angle_deg, radius):
        # angle 0 = straight up; positive = clockwise (right)
        rad = math.radians(angle_deg)
        return (HOME[0] + radius * math.sin(rad), HOME[1] - radius * math.cos(rad))

    def wedge_path(a1, a2, r_in, r_out):
        x1i, y1i = polar(a1, r_in)
        x1o, y1o = polar(a1, r_out)
        x2i, y2i = polar(a2, r_in)
        x2o, y2o = polar(a2, r_out)
        large = 1 if (a2 - a1) > 180 else 0
        return (f"M {x1i:.2f},{y1i:.2f} L {x1o:.2f},{y1o:.2f} "
                f"A {r_out},{r_out} 0 {large} 1 {x2o:.2f},{y2o:.2f} "
                f"L {x2i:.2f},{y2i:.2f} A {r_in},{r_in} 0 {large} 0 {x1i:.2f},{y1i:.2f} Z")

    def pie_path(a1, a2, r_out):
        """Pie wedge from the apex (HOME) outward — used for the foul zones
        so they form proper diamond corners instead of annular sectors with
        a curved gap near the apex."""
        x1, y1 = polar(a1, r_out)
        x2, y2 = polar(a2, r_out)
        large = 1 if (a2 - a1) > 180 else 0
        return (f"M {HOME[0]:.2f},{HOME[1]:.2f} L {x1:.2f},{y1:.2f} "
                f"A {r_out},{r_out} 0 {large} 1 {x2:.2f},{y2:.2f} Z")

    def label_pos(a1, a2, r_in, r_out):
        ang = (a1 + a2) / 2
        radius = (r_in + r_out) / 2
        return polar(ang, radius)

    # Single fair-territory color ramp: light orange → deep red.
    def red_orange(intensity):
        i = max(0.05, min(1.0, intensity))
        r = int(255 - (255 - 198) * i)
        g = int(224 - (224 - 40)  * i)
        b = int(204 - (204 - 40)  * i)
        return f'#{r:02X}{g:02X}{b:02X}'
    def text_color(intensity):
        return '#FFFFFF' if intensity > 0.55 else '#0F2A4D'

    # Single intensity scale across ALL fair-territory zones (infield,
    # outfield, down-the-line) so identical metric values render identical
    # shades.
    def max_val(zones):
        return max((val_by_zone.get(zc[0], 0) for zc, *_ in zones), default=1.0) or 1.0
    fair_max = max_val(OUTFIELD + INFIELD + LINE)

    # Helper: embed a PNG file into the SVG as an <image> element. Uses
    # BOTH href and xlink:href for cross-renderer reliability (cairosvg
    # and some browsers require xlink:href; modern browsers prefer href).
    def _embed_image(path: Path, x: float, y: float, w: float, h: float,
                     opacity: float = 1.0) -> str:
        if not path.exists():
            return ''
        try:
            b64 = base64.b64encode(path.read_bytes()).decode('ascii')
        except Exception:
            return ''
        href = f'data:image/png;base64,{b64}'
        return (f'<image href="{href}" xlink:href="{href}" '
                f'x="{x}" y="{y}" width="{w}" height="{h}" '
                f'opacity="{opacity}" preserveAspectRatio="xMidYMid meet" '
                f'pointer-events="none"/>')

    # Numeric width/height + viewBox give cairosvg a clean intrinsic size.
    # NO inline style on the <svg> — cairosvg can mis-parse `width:100%`
    # in the style and compute a zero-sized canvas (blank PNG). Browser
    # responsiveness is handled by a wrapper <div> at render time.
    # Background is a full-viewport <rect> since cairosvg ignores CSS
    # `background:`.
    parts = ['<svg viewBox="0 0 100 72" width="100" height="72" '
             'xmlns="http://www.w3.org/2000/svg" '
             'xmlns:xlink="http://www.w3.org/1999/xlink">'
             '<rect x="0" y="0" width="100" height="72" fill="#F0EAD6"/>']

    # 64 Analytics wide brand logo, centered above the diamond. Native
    # ratio ≈ 4.3:1, so 38×9 fits the top band cleanly.
    parts.append(_embed_image(BRAND_64A_WIDE, x=31, y=0.5, w=38, h=9))

    # Player identity (when in Player mode) is already shown in the
    # bottom-right scope caption — no separate header needed.

    # Compact integer format — keeps wide blocks readable for "All teams"
    # views where totals run into the hundreds of thousands.
    def _compact(n):
        n = int(n)
        if abs(n) >= 1_000_000: return f'{n/1_000_000:.1f}M'
        if abs(n) >= 10_000:    return f'{n/1_000:.0f}K'
        return f'{n:,}'

    # Top-left corner: mini-diamond echoing the main field, split into
    # Left / Center / Right wedges with the same red/orange heatmap ramp.
    # Intensity is per-side percent normalized to the dominant side so
    # the pull/oppo skew reads at a glance.
    MINI_HOME = (14, 13)
    MINI_R    = 11.5      # bigger so labels don't crowd
    LBL_FRAC  = 0.72      # label radius as fraction of MINI_R (further out = more separation between adjacent labels)
    side_pcts = {
        'L': float(buckets['left_pct']),
        'C': float(buckets['middle_pct']),
        'R': float(buckets['right_pct']),
    }
    side_max_pct = max(side_pcts.values()) or 1.0
    SIDE_WEDGES = [
        ('L', -45, -15),
        ('C', -15,  15),
        ('R',  15,  45),
    ]
    def _mini_pie_path(a1, a2, r):
        rad1, rad2 = math.radians(a1), math.radians(a2)
        x1 = MINI_HOME[0] + r * math.sin(rad1)
        y1 = MINI_HOME[1] - r * math.cos(rad1)
        x2 = MINI_HOME[0] + r * math.sin(rad2)
        y2 = MINI_HOME[1] - r * math.cos(rad2)
        large = 1 if (a2 - a1) > 180 else 0
        return (f"M {MINI_HOME[0]},{MINI_HOME[1]} L {x1:.2f},{y1:.2f} "
                f"A {r},{r} 0 {large} 1 {x2:.2f},{y2:.2f} Z")
    for key, a1, a2 in SIDE_WEDGES:
        pct = side_pcts[key]
        intensity = pct / side_max_pct
        fill = red_orange(intensity)
        parts.append(
            f'<path d="{_mini_pie_path(a1, a2, MINI_R)}" fill="{fill}" '
            f'stroke="#FFFFFF" stroke-width="0.5"/>'
        )
        ang_mid = math.radians((a1 + a2) / 2)
        lx = MINI_HOME[0] + (MINI_R * LBL_FRAC) * math.sin(ang_mid)
        ly = MINI_HOME[1] - (MINI_R * LBL_FRAC) * math.cos(ang_mid)
        tc = text_color(intensity)
        parts.append(
            f'<text x="{lx:.1f}" y="{ly+0.5:.1f}" text-anchor="middle" '
            f'font-family="Inter,sans-serif" font-size="1.6" font-weight="800" '
            f'fill="{tc}">{pct:.0f}%</text>'
        )
    parts.append(
        f'<path d="{_mini_pie_path(-45, 45, MINI_R)}" '
        f'fill="none" stroke="#0F2A4D" stroke-width="0.5"/>'
    )

    # Top-right corner: hit-type 2x2 grid (1B/2B on top, 3B/HR on bottom).
    # Sized to match the mini-diamond's vertical bounds (y=1.5 to y=13)
    # for visual symmetry across the top of the chart.
    def _all_top(c): return int(spray[c].sum()) if c in spray.columns else 0
    hit_grid = [
        [('1B', _all_top('1B')), ('2B', _all_top('2B'))],
        [('3B', _all_top('3B')), ('HR', _all_top('HR'))],
    ]
    tr_x0, tr_y0 = 72, 1.5
    cell_w, cell_h = 13, 5.75      # 2 rows × 5.75 = 11.5, matches MINI_R
    for ri, row in enumerate(hit_grid):
        for ci, (lab, val) in enumerate(row):
            cx = tr_x0 + cell_w/2 + ci * cell_w
            cy = tr_y0 + ri * cell_h
            parts.append(
                f'<text x="{cx:.1f}" y="{cy+1.7:.1f}" text-anchor="middle" '
                f'font-family="Inter,sans-serif" font-size="1.7" font-weight="600" '
                f'fill="#0F2A4D" letter-spacing="0.3">{lab}</text>'
            )
            parts.append(
                f'<text x="{cx:.1f}" y="{cy+5.0:.1f}" text-anchor="middle" '
                f'font-family="Inter,sans-serif" font-size="3.0" font-weight="800" '
                f'fill="#0F2A4D">{_compact(val)}</text>'
            )

    def _draw_wedge(zone_codes, a1, a2, r_in, r_out, label_font, *, pie=False):
        primary = zone_codes[0]
        v = val_by_zone.get(primary, 0)
        intensity = v / fair_max if fair_max else 0
        fill = red_orange(intensity)
        path_d = pie_path(a1, a2, r_out) if pie else wedge_path(a1, a2, r_in, r_out)
        parts.append(f'<path d="{path_d}" fill="{fill}" stroke="#FFFFFF" stroke-width="0.6"><title>{_tooltip_for(zone_codes)}</title></path>')
        lx, ly = label_pos(a1, a2, r_in, r_out)
        tc = text_color(intensity)
        parts.append(f'<text x="{lx:.1f}" y="{ly+1.0:.1f}" text-anchor="middle" font-family="Inter,sans-serif" font-size="{label_font}" font-weight="800" fill="{tc}" pointer-events="none">{fmt_by_zone.get(primary, "")}</text>')

    # Outfield (CF wedge spans 8+14)
    for zc, a1, a2 in OUTFIELD:
        _draw_wedge(zc, a1, a2, R_MID, R_OUTER, 4.5)
    # Down-the-line wedges (zones 12/13) — fair territory, pie wedges from apex
    for zc, a1, a2 in LINE:
        _draw_wedge(zc, a1, a2, R_INNER, R_OUTER, 3.5, pie=True)
    # Infield
    for zc, a1, a2 in INFIELD:
        _draw_wedge(zc, a1, a2, R_INNER, R_MID, 2.3)

    # Pitcher (zone 1) — circle on the centerline, sized so its bottom edge
    # meets the catcher's top edge for a clean stacked diamond apex.
    r_p = 3.0
    px, py = HOME[0], HOME[1] - 5   # center 5 units above HOME → bottom at y=63
    parts.append(f'<circle cx="{px}" cy="{py}" r="{r_p}" fill="#F8E8E2" stroke="#0F2A4D" stroke-width="0.5"><title>{_tooltip_for((1,))}</title></circle>')
    parts.append(f'<text x="{px}" y="{py+0.9:.1f}" text-anchor="middle" font-family="Inter,sans-serif" font-size="2.4" font-weight="800" fill="#0F2A4D" pointer-events="none">{fmt_by_zone.get(1,"")}</text>')

    # Catcher (zone 2) — home-plate pentagon, positioned ENTIRELY BELOW the
    # foul-line apex so its number stays clear of the V where the foul
    # wedges meet. Front (flat) edge sits at HOME, point extends downward.
    pw, ph = 2.8, 2.6   # half-width, half-height
    cx, cy = HOME[0], HOME[1] + ph   # top of plate aligns with HOME y
    plate_pts = [
        (cx - pw, cy - ph),         # front-left (flat edge facing pitcher)
        (cx + pw, cy - ph),         # front-right
        (cx + pw, cy + ph * 0.10),  # right shoulder
        (cx,      cy + ph),         # back point (toward catcher)
        (cx - pw, cy + ph * 0.10),  # left shoulder
    ]
    pts_str = ' '.join(f'{x:.2f},{y:.2f}' for x, y in plate_pts)
    parts.append(f'<polygon points="{pts_str}" fill="#F8E8E2" stroke="#0F2A4D" stroke-width="0.4"><title>{_tooltip_for((2,))}</title></polygon>')
    parts.append(f'<text x="{cx}" y="{cy+0.4:.1f}" text-anchor="middle" font-family="Inter,sans-serif" font-size="2.2" font-weight="800" fill="#0F2A4D" pointer-events="none">{fmt_by_zone.get(2,"")}</text>')

    # Single continuous field outline — foul line L, outfield arc, foul line R,
    # all sharing the same endpoints as the foul-wedge outer corners. This is
    # the merged diamond/semi-circle outline the design calls for.
    foul_outer = 45 + LINE_HALF
    fxL, fyL = polar(-foul_outer, R_OUTER)
    fxR, fyR = polar( foul_outer, R_OUTER)
    parts.append(
        f'<path d="M {HOME[0]},{HOME[1]} L {fxL:.2f},{fyL:.2f} '
        f'A {R_OUTER},{R_OUTER} 0 0 1 {fxR:.2f},{fyR:.2f} Z" '
        f'fill="none" stroke="#0F2A4D" stroke-width="0.7"/>'
    )

    # Team logo at 50% opacity overlaid on the diamond (Team mode only).
    # Drawn AFTER the wedges so it sits on top — otherwise the opaque
    # wedge fills cover it completely.
    if view_mode == 'Team' and selected_team_id is not None:
        logo_path = _APP_DIR / 'team_logos_512' / f'{selected_team_id}.png'
        logo_size = 28
        parts.append(_embed_image(logo_path,
                                  x=50 - logo_size/2, y=36 - logo_size/2,
                                  w=logo_size, h=logo_size, opacity=0.4))

    # Bottom-left overall stats block — aggregate across the entire selection.
    # AVG/SLG/wOBA are BIP-conditional (denominator = total BIP); strikeouts
    # never enter the universe because SOs have no hitLocation.
    def _all(c): return int(spray[c].sum()) if c in spray.columns else 0
    n_bip = int(spray['total'].sum())
    a1, a2, a3, ahr = _all('1B'), _all('2B'), _all('3B'), _all('HR')
    a_hits = a1 + a2 + a3 + ahr
    a_tb = a1 + 2*a2 + 3*a3 + 4*ahr
    a_woba_num = 0.888*a1 + 1.271*a2 + 1.616*a3 + 2.101*ahr
    if n_bip > 0:
        agg_avg, agg_slg, agg_woba = a_hits / n_bip, a_tb / n_bip, a_woba_num / n_bip
    else:
        agg_avg = agg_slg = agg_woba = 0.0
    overall = [
        ('AVG',  _metric_fmt(agg_avg,  'AVG')),
        ('SLG',  _metric_fmt(agg_slg,  'SLG')),
        ('wOBA', _metric_fmt(agg_woba, 'wOBA')),
        ('TB',   _compact(a_tb)),
    ]
    block_w = 9.5
    x0 = 1.5
    # Font sizes match the bottom-right caption rows (label 2.0 / value 2.4)
    # at identical baselines so the two bottom blocks sit on the same plane.
    for i, (lab, val) in enumerate(overall):
        cx = x0 + block_w/2 + i * block_w
        parts.append(
            f'<text x="{cx:.1f}" y="66.5" text-anchor="middle" '
            f'font-family="Inter,sans-serif" font-size="2.0" font-weight="600" '
            f'fill="#0F2A4D" letter-spacing="0.3">{lab}</text>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="70" text-anchor="middle" '
            f'font-family="Inter,sans-serif" font-size="2.4" font-weight="800" '
            f'fill="#0F2A4D">{val}</text>'
        )

    # Bottom-right caption — what the chart depicts.
    sport_label = 'Baseball' if sport.lower() == 'baseball' else 'Softball'
    if view_mode == 'Player' and selected_player_name:
        scope_txt = str(selected_player_name)
    elif selected_team_short:
        scope_txt = str(selected_team_short)
    elif team_filter:
        scope_txt = str(team_filter)
    else:
        scope_txt = 'All teams'
    filter_bits = []
    if vs_hand: filter_bits.append('vs LHP' if vs_hand == 'L' else 'vs RHP')
    if f_two_strikes: filter_bits.append('2 strikes')
    if f_two_outs: filter_bits.append('2 outs')
    if f_risp: filter_bits.append('RISP')
    filter_txt = (' · ' + ', '.join(filter_bits)) if filter_bits else ''
    caption_line1 = f'{scope_txt} · {sport_label} {division}{filter_txt}'
    caption_line2 = f'{metric_choice} · {buckets["total"]:,} balls in play'
    parts.append(
        f'<text x="98" y="66.5" text-anchor="end" font-family="Inter,sans-serif" '
        f'font-size="2.0" font-weight="600" fill="#0F2A4D">{caption_line1}</text>'
    )
    parts.append(
        f'<text x="98" y="70" text-anchor="end" font-family="Inter,sans-serif" '
        f'font-size="2.4" font-weight="800" fill="#0F2A4D">{caption_line2}</text>'
    )

    parts.append('</svg>')
    svg_str = ''.join(parts)
    # For browser display: inject a style on the <svg> so it scales to
    # the container. For PNG export below: use svg_str unchanged so
    # cairosvg sees clean numeric width/height with no CSS overrides.
    display_svg = svg_str.replace(
        '<svg ', '<svg style="width:100%;height:auto;display:block;border-radius:8px;" ', 1
    )
    st.markdown(display_svg, unsafe_allow_html=True)

    # PNG download — convert the SVG to a 1600px-wide PNG via cairosvg.
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'), output_width=1600)
        fname_scope = (selected_player_name if view_mode == 'Player' and selected_player_name
                       else (team_filter or 'all'))
        fname_scope = ''.join(c if c.isalnum() else '_' for c in str(fname_scope))[:40]
        fname = f'spray_{sport}_{division}_{fname_scope}.png'
        st.download_button('Download PNG', data=png_bytes, file_name=fname,
                           mime='image/png', use_container_width=False)
    except Exception as e:
        st.caption(f'PNG export unavailable in this environment ({type(e).__name__}).')

with col_summary:
    st.markdown('### Field-side splits')
    side_df = pd.DataFrame([
        ['Left side',  '3B/SS/LF/L Line',  buckets['left'],   f"{buckets['left_pct']}%"],
        ['Middle',     'P/C/2B/CF',        buckets['middle'], f"{buckets['middle_pct']}%"],
        ['Right side', '1B/RF/R Line',     buckets['right'],  f"{buckets['right_pct']}%"],
    ], columns=['Side', 'Zones', 'Count', 'Share'])
    st.dataframe(side_df, hide_index=True, use_container_width=True)

    st.markdown('### Top zones')
    top = spray.sort_values('total', ascending=False).head(5)[
        ['zone_name', 'total', 'pct']
    ].rename(columns={'zone_name': 'Zone', 'total': 'Count', 'pct': '%'})
    st.dataframe(top, hide_index=True, use_container_width=True)

# ── Zone drill-down card ────────────────────────────────────────────────────
st.markdown('### Zone detail')
zone_options = spray.sort_values('total', ascending=False)
zone_labels = [f"{r['zone_name']} (zone {int(r['hitLocation'])}) — {int(r['total'])} BIP"
               for _, r in zone_options.iterrows()]
zone_ids = zone_options['hitLocation'].tolist()
if zone_labels:
    pick = st.selectbox('Show details for:', range(len(zone_labels)),
                        format_func=lambda i: zone_labels[i], key='zone_drill')
    z = int(zone_ids[pick])
    row = spray[spray['hitLocation'] == z].iloc[0]
    cols = st.columns(8)
    metrics = [
        ('BIP',  f"{int(row['total'])}", f"{row['pct']:.1f}% of all"),
        ('1B',   f"{int(row.get('1B', 0))}", ''),
        ('2B',   f"{int(row.get('2B', 0))}", ''),
        ('3B',   f"{int(row.get('3B', 0))}", ''),
        ('HR',   f"{int(row.get('HR', 0))}", ''),
        ('AVG',  _metric_fmt(float(row['AVG']),  'AVG'),  ''),
        ('SLG',  _metric_fmt(float(row['SLG']),  'SLG'),  ''),
        ('wOBA', _metric_fmt(float(row['wOBA']), 'wOBA'), f"TB {int(row['TB'])}"),
    ]
    for col, (label, val, sub) in zip(cols, metrics):
        col.metric(label, val, sub if sub else None, delta_color='off')

# ── Result-by-zone matrix ────────────────────────────────────────────────────
st.markdown('### Hit results by zone')
matrix_cols = [c for c in ['1B', '2B', '3B', 'HR', 'Out', 'FC/Err', 'Other'] if c in spray.columns]
if matrix_cols:
    display = spray[['zone_name', 'total'] + matrix_cols + ['AVG', 'SLG', 'wOBA', 'TB']].copy()
    display = display.rename(columns={'zone_name': 'Zone', 'total': 'BIP'})
    for c in ['AVG', 'SLG', 'wOBA']:
        display[c] = display[c].apply(lambda v: _metric_fmt(float(v), c))
    st.dataframe(display, hide_index=True, use_container_width=True)

# ── Bar chart of zone distribution ──────────────────────────────────────────
st.markdown('### Distribution by zone')
chart_df = spray[['zone_name', 'total']].rename(columns={'zone_name': 'Zone', 'total': 'Count'})
st.bar_chart(chart_df, x='Zone', y='Count', height=240)

# ── Footer note ─────────────────────────────────────────────────────────────
st.caption(
    'Zone codes: 1=P, 2=C, 3=1B, 4=2B, 5=3B, 6=SS, 7=LF, 8=CF, 9=RF, '
    '10=LCF gap, 11=RCF gap, 12=L Line, 13=R Line, 14=Deep CF. '
    'Zones 12 / 13 are FAIR (down-the-line hits), not foul; zone 14 is '
    'over-the-fence / deep CF. Without batter handedness we can\'t '
    'classify Pull/Oppo, so Left/Middle/Right is the unbiased split.'
)
