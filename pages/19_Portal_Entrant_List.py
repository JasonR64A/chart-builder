"""
Portal Entrant List — a multi-entrant, print-ready table graphic.

Companion to the single-player Portal Entrant Graphic. Implements the
"Portal Entrant List" design (Claude Design handoff, 2026-06-02): a brand-matched
64 Analytics table of portal entrants with summary counts, search/conference/
position/status filters, three layouts (Ledger / Board / Index), grouping, and
a print/PDF + PNG export.

Streamlit widgets drive the filters; the table itself is rendered as static HTML
(the design's exact CSS, inlined) inside an iframe so it prints clean and exports
to PNG via html2canvas.
"""
from __future__ import annotations

import html as _html
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _APP_DIR / 'data'

# Ordering used for conference / position group sort, from the design.
CONF_ORDER = ['ACC', 'SEC', 'Big Ten', 'Big 12', 'Pac-12', 'CAA', 'Sun Belt',
              'AAC', 'C-USA', 'WAC', 'WCC', 'Big West', 'Mountain West']
POS_ORDER = ['RHP', 'LHP', 'P', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'OF', 'DH', 'UT']
CLASS_SHORT = {'Freshman': 'Fr', 'Sophomore': 'So', 'Junior': 'Jr', 'Senior': 'Sr',
               'Graduate': 'Gr', 'Grad': 'Gr', 'Redshirt Senior': 'Sr',
               'FR': 'Fr', 'SO': 'So', 'JR': 'Jr', 'SR': 'Sr', 'GR': 'Gr'}


# ── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def _csv(name, **kw):
    return pd.read_csv(DATA_DIR / name, low_memory=False, **kw)


@st.cache_data
def build_entrants() -> pd.DataFrame:
    """Join the live portal list into the design's per-entrant schema."""
    prp = _csv('portal_rank_player.csv')
    prp['player_id'] = pd.to_numeric(prp['player_id'], errors='coerce').astype('Int64')
    if 'year' in prp.columns:
        prp['year'] = pd.to_numeric(prp['year'], errors='coerce')
        prp = prp[prp['year'] == prp['year'].max()]
    prp = prp.dropna(subset=['player_id']).drop_duplicates('player_id', keep='first')

    players = _csv('players.csv', encoding='latin-1', dtype=str).fillna('')
    players['id'] = pd.to_numeric(players['id'], errors='coerce').astype('Int64')
    pmap = players.dropna(subset=['id']).drop_duplicates('id', keep='first').set_index('id')

    teams = _csv('teams.csv', dtype=str).fillna('')
    teams['id'] = pd.to_numeric(teams['id'], errors='coerce').astype('Int64')
    # teams.csv carries legacy duplicate ids — dedup so .loc returns one row.
    tmap = teams.dropna(subset=['id']).drop_duplicates('id', keep='first').set_index('id')

    confs = _csv('conferences.csv', dtype=str).fillna('')
    confs['id'] = pd.to_numeric(confs['id'], errors='coerce').astype('Int64')
    cmap = confs.dropna(subset=['id']).drop_duplicates('id', keep='first').set_index('id')

    pr = _csv('player_rank.csv')
    pr['player_id'] = pd.to_numeric(pr['player_id'], errors='coerce').astype('Int64')
    if 'year' in pr.columns:
        pr = pr[pd.to_numeric(pr['year'], errors='coerce') == pd.to_numeric(pr['year'], errors='coerce').max()]
    pr = pr.drop_duplicates('player_id', keep='first').set_index('player_id')

    # 2026 stat lines, one row per player (max volume row).
    hit = _csv('hitting.csv')
    hit = hit[pd.to_numeric(hit['year'], errors='coerce') == 2026].copy()
    hit['player_id'] = pd.to_numeric(hit['player_id'], errors='coerce').astype('Int64')
    if 'plate_appearances' in hit.columns:
        hit = hit.sort_values('plate_appearances', ascending=False)
    hmap = hit.drop_duplicates('player_id', keep='first').set_index('player_id')

    pit = _csv('pitching.csv')
    pit = pit[pd.to_numeric(pit['year'], errors='coerce') == 2026].copy()
    pit['player_id'] = pd.to_numeric(pit['player_id'], errors='coerce').astype('Int64')
    if 'innings_pitched' in pit.columns:
        pit = pit.sort_values('innings_pitched', ascending=False)
    pmap_pit = pit.drop_duplicates('player_id', keep='first').set_index('player_id')

    def tname(tid):
        try:
            return tmap.loc[int(tid), 'name']
        except Exception:
            return ''

    def tconf(tid):
        try:
            cid = int(tmap.loc[int(tid), 'conference_id'])
            abbr = cmap.loc[cid, 'abbreviation']
            return abbr or cmap.loc[cid, 'name']
        except Exception:
            return ''

    def tsport(tid):
        try:
            return tmap.loc[int(tid), 'sport']
        except Exception:
            return ''

    rows = []
    for r in prp.itertuples():
        pid = int(r.player_id)
        prev_tid = r.team_id
        new_tid = getattr(r, 'new_team_id', None)
        prow = pmap.loc[pid] if pid in pmap.index else None
        name = (prow['player_name'] if prow is not None else '') or str(getattr(r, 'name', '')) or f'#{pid}'
        pos = (prow['position'].upper() if prow is not None else '') or ''
        cls_raw = (prow['classification'] if prow is not None else '') or ''
        cls = CLASS_SHORT.get(cls_raw, cls_raw[:2].title() if cls_raw else '')
        hometown = (prow['hometown'] if prow is not None else '') or ''
        rank = pd.to_numeric(getattr(r, 'sixty_four_rating_portal_player', None), errors='coerce')

        committed = pd.notna(new_tid) and str(new_tid) not in ('', '0', 'nan')
        to_school = tname(new_tid) if committed else ''
        status = 'Committed' if committed else 'Uncommitted'

        entered = ''
        if pid in pr.index:
            entered = str(pr.loc[pid].get('initiated_date', '') or '')[:10]

        is_pitcher = ('P' in pos and pos not in ('PH', 'PR')) or (pid not in hmap.index and pid in pmap_pit.index)
        if is_pitcher and pid in pmap_pit.index:
            s = pmap_pit.loc[pid]
            stat_type = 'pitch'
            stat = {
                'era': _rate(s.get('earned_run_average'), 2),
                'ip': _rate(s.get('innings_pitched'), 1),
                'so': _int(s.get('strikeouts')),
                'rec': f"{_int(s.get('wins'))}-{_int(s.get('losses'))}",
            }
        else:
            s = hmap.loc[pid] if pid in hmap.index else None
            stat_type = 'hit'
            stat = {
                'avg': _slash(s.get('batting_average') if s is not None else None),
                'obp': _slash(s.get('on_base_percentage') if s is not None else None),
                'slg': _slash(s.get('slugging_percentage') if s is not None else None),
                'hr': _int(s.get('home_runs') if s is not None else None),
                'sb': _int(s.get('stolen_bases') if s is not None else None),
            }

        rows.append({
            'player_id': pid, 'name': name, 'sport': tsport(prev_tid) or '',
            'pos': pos or '—', 'prevSchool': tname(prev_tid), 'conf': tconf(prev_tid),
            'cls': cls, 'rank': rank, 'status': status, 'to': to_school,
            'hometown': hometown, 'entered': entered,
            'stat_type': stat_type, **{f'st_{k}': v for k, v in stat.items()},
        })
    df = pd.DataFrame(rows)
    df = df[df['name'].astype(str).str.strip() != '']
    return df


def _int(v):
    try:
        f = float(v)
        return '' if pd.isna(f) else str(int(round(f)))
    except Exception:
        return ''


def _rate(v, d):
    try:
        f = float(v)
        return '' if pd.isna(f) else f'{f:.{d}f}'
    except Exception:
        return ''


def _slash(v):
    try:
        f = float(v)
        if pd.isna(f):
            return ''
        s = f'{f:.3f}'
        return s[1:] if s.startswith('0.') else s
    except Exception:
        return ''


# Design CSS — inlined verbatim from the handoff styles.css (the print rules and
# all three layout classes are preserved).
DESIGN_CSS = """
:root{--blush:#f3e6e8;--blush-deep:#ecd9dc;--paper:#fff;--ink:#14202e;--ink-2:#3a4654;
--ink-soft:#7c8694;--maroon:#9a303e;--maroon-deep:#7c2531;--blue:#005ca6;--orange:#c6783a;
--green:#2f7d5b;--hair:#15212e;--dot:#cbb6ba;--zebra:#faf4f5;--radius:3px;
--mono:'Spline Sans Mono',ui-monospace,Menlo,monospace;--sans:'Archivo',system-ui,sans-serif;}
*{box-sizing:border-box;}html,body{margin:0;padding:0;}
body{font-family:var(--sans);background:var(--blush);color:var(--ink);-webkit-font-smoothing:antialiased;}
.wrap{max-width:1240px;margin:0 auto;padding:34px 28px 28px;}
.mast{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;flex-wrap:wrap;margin-bottom:22px;}
.brand{display:flex;align-items:baseline;gap:0;line-height:1;}
.brand .b64{font-weight:800;font-size:40px;letter-spacing:-1.5px;color:var(--maroon);}
.brand .bword{font-weight:800;font-size:30px;letter-spacing:-0.5px;color:var(--ink);text-transform:uppercase;}
.mast-meta{text-align:right;}
.mast h1{font-size:15px;font-weight:700;text-transform:uppercase;letter-spacing:2.5px;color:var(--ink);margin:0 0 4px;}
.mast .sub{font-family:var(--mono);font-size:11.5px;color:var(--ink-soft);letter-spacing:.2px;}
.summary{display:flex;gap:0;border:1.5px solid var(--hair);background:var(--paper);margin-bottom:18px;}
.stat{flex:1;padding:13px 18px;border-right:1px solid var(--dot);position:relative;}
.stat:last-child{border-right:none;}
.stat.active{background:var(--ink);}
.stat.active .stat-num,.stat.active .stat-lab{color:var(--paper);}
.stat-num{font-family:var(--mono);font-size:25px;font-weight:600;color:var(--ink);line-height:1;letter-spacing:-1px;}
.stat-num .accent{color:var(--maroon);}
.stat-lab{font-size:10.5px;text-transform:uppercase;letter-spacing:1.3px;color:var(--ink-soft);margin-top:6px;font-weight:600;}
.resultline{font-family:var(--mono);font-size:11px;color:var(--ink-soft);margin:0 2px 8px;letter-spacing:.2px;text-transform:uppercase;}
.tablecard{background:var(--paper);border:1.5px solid var(--hair);}
table.portal{width:100%;border-collapse:collapse;}
table.portal thead th{font-size:10.5px;text-transform:uppercase;letter-spacing:1.2px;font-weight:700;color:var(--ink);
text-align:left;padding:11px 14px;border-bottom:1.5px solid var(--hair);white-space:nowrap;background:var(--blush);}
table.portal thead th.num{text-align:center;}
table.portal tbody td{padding:11px 14px;font-size:13.5px;vertical-align:middle;}
.colrank{text-align:center;width:64px;}.colpos{text-align:center;width:64px;}
.rank-fig{font-family:var(--mono);font-weight:600;font-size:15px;color:var(--orange);letter-spacing:-.5px;}
.pos-tag{font-family:var(--mono);font-weight:600;font-size:13px;color:var(--blue);}
.pname{font-weight:700;color:var(--ink);letter-spacing:-.2px;font-size:14.5px;white-space:nowrap;}
.pschool{font-weight:600;color:var(--ink);}
.pconf{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--ink-2);white-space:nowrap;}
.pclass{font-family:var(--mono);font-size:11.5px;color:var(--ink-soft);}
.pdate{font-family:var(--mono);font-size:11.5px;color:var(--ink-soft);white-space:nowrap;}
.pill{display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:700;letter-spacing:.3px;padding:3px 9px;border-radius:20px;white-space:nowrap;}
.pill .dot{width:6px;height:6px;border-radius:50%;}
.pill.committed{background:rgba(47,125,91,.12);color:var(--green);}.pill.committed .dot{background:var(--green);}
.pill.uncommitted{background:rgba(0,92,166,.10);color:var(--blue);}.pill.uncommitted .dot{background:var(--blue);}
.pill.withdrawn{background:rgba(124,37,49,.10);color:var(--maroon);}.pill.withdrawn .dot{background:var(--maroon);}
.commit-to{font-weight:700;color:var(--ink);font-size:13px;}
.commit-arrow{color:var(--green);margin:0 5px;font-weight:700;}
.lay-ledger tbody tr{border-bottom:1px dotted var(--dot);}
.lay-ledger tbody tr:last-child{border-bottom:none;}
.lay-board tbody tr{border-bottom:1px solid #eee3e5;}
.lay-board tbody tr:nth-child(even){background:var(--zebra);}
.lay-board .colrank{width:74px;}
.lay-board .rank-badge{display:inline-flex;align-items:center;justify-content:center;width:40px;height:40px;background:var(--ink);color:var(--paper);font-family:var(--mono);font-weight:600;font-size:16px;border-radius:var(--radius);}
.lay-board .rank-badge.top{background:var(--maroon);}
.lay-board .pname{font-size:15.5px;}
.lay-board .submeta{font-size:11.5px;color:var(--ink-soft);font-weight:600;margin-top:3px;display:flex;gap:7px;align-items:center;}
.lay-board .submeta .sep{color:var(--dot);}
.lay-board tbody td{padding:13px 14px;}
.statline{font-family:var(--mono);font-size:12px;color:var(--ink-2);white-space:nowrap;}
.statline b{color:var(--ink);font-weight:600;}
.lay-index{border:none;background:transparent;}
.lay-index table.portal thead th{background:transparent;border-bottom:2px solid var(--hair);border-top:2px solid var(--hair);font-size:10px;}
.lay-index tbody tr{border-bottom:1px solid #e4d6d9;}
.lay-index tbody td{padding:8px 14px;font-size:13px;}
.lay-index .rank-fig{color:var(--ink);}.lay-index .pos-tag{color:var(--ink-2);}
.grouphd td{background:var(--ink)!important;color:var(--paper);padding:9px 14px!important;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1.4px;}
.grouphd .gcount{float:right;font-family:var(--mono);color:var(--blush);font-weight:600;letter-spacing:0;}
.lay-index .grouphd td{background:transparent!important;color:var(--maroon);border-bottom:1.5px solid var(--maroon);}
.lay-index .grouphd .gcount{color:var(--ink-soft);}
.empty{padding:60px 20px;text-align:center;color:var(--ink-soft);font-family:var(--mono);font-size:13px;}
.foot{margin-top:18px;font-family:var(--mono);font-size:10.5px;color:var(--ink-soft);display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;border-top:1px dotted var(--dot);padding-top:14px;}
.btn-row{text-align:center;margin:16px 0;}
.btn-row button{padding:10px 22px;background:#9a303e;color:#fff;border:none;border-radius:3px;font-family:var(--sans);font-weight:700;font-size:13px;letter-spacing:.15em;text-transform:uppercase;cursor:pointer;margin:0 5px;box-shadow:0 4px 12px rgba(0,0,0,.25);}
@media print{@page{margin:12mm;}body{background:#fff;}.wrap{max-width:none;padding:0;}.btn-row{display:none!important;}
table.portal thead th{background:#fff!important;-webkit-print-color-adjust:exact;}
.grouphd td{background:#14202e!important;-webkit-print-color-adjust:exact;print-color-adjust:exact;}
tbody tr{break-inside:avoid;}.pill,.rank-badge,.stat-num .accent{-webkit-print-color-adjust:exact;print-color-adjust:exact;}}
"""


def _e(v):
    return _html.escape(str(v if v is not None else ''))


def status_cell(p) -> str:
    if p['status'] == 'Committed':
        return (f'<span class="pill committed"><span class="dot"></span>COMMIT</span>'
                f'<span class="commit-to"><span class="commit-arrow">→</span>{_e(p["to"])}</span>')
    if p['status'] == 'Withdrawn':
        return '<span class="pill withdrawn"><span class="dot"></span>WITHDREW</span>'
    return '<span class="pill uncommitted"><span class="dot"></span>AVAILABLE</span>'


def key_stat(p):
    if p['stat_type'] == 'hit':
        return (p.get('st_avg') or '—', 'BA', p.get('st_hr') or '0', 'HR')
    return (p.get('st_era') or '—', 'ERA', p.get('st_so') or '0', 'K')


def rank_disp(p):
    r = p['rank']
    return '—' if pd.isna(r) else str(int(r))


def fmt_date(iso):
    try:
        d = datetime.strptime(str(iso)[:10], '%Y-%m-%d')
        return f'{d.strftime("%b")} {d.day}'
    except Exception:
        return iso or '—'


def render_row(p, layout, show_stat, highlight_top) -> str:
    a, al, b, bl = key_stat(p)
    top = highlight_top and (not pd.isna(p['rank'])) and int(p['rank']) <= 10
    if layout == 'board':
        cells = [
            f'<td class="colrank"><span class="rank-badge {"top" if top else ""}">{rank_disp(p)}</span></td>',
            (f'<td><div class="pname">{_e(p["name"])}</div>'
             f'<div class="submeta"><span>{_e(p["prevSchool"])}</span><span class="sep">·</span>'
             f'<span>{_e(p["conf"])}</span><span class="sep">·</span><span>{_e(p["cls"])}</span></div></td>'),
            f'<td class="colpos"><span class="pos-tag">{_e(p["pos"])}</span></td>',
        ]
        if show_stat:
            cells.append(f'<td><span class="statline"><b>{_e(a)}</b> {al} · <b>{_e(b)}</b> {bl}</span></td>')
        cells.append(f'<td>{status_cell(p)}</td>')
        cells.append(f'<td class="pdate">{_e(fmt_date(p["entered"]))}</td>')
        return '<tr>' + ''.join(cells) + '</tr>'
    # ledger + index
    rstyle = ' style="color:var(--maroon)"' if (top and layout == 'ledger') else ''
    cells = [
        f'<td class="colrank"><span class="rank-fig"{rstyle}>{rank_disp(p)}</span></td>',
        f'<td class="pname">{_e(p["name"])}</td>',
        f'<td class="colpos"><span class="pos-tag">{_e(p["pos"])}</span></td>',
        f'<td class="pschool">{_e(p["prevSchool"])}</td>',
        f'<td class="pconf">{_e(p["conf"])}</td>',
        f'<td class="pclass">{_e(p["cls"])}</td>',
    ]
    if show_stat:
        cells.append(f'<td class="statline"><b>{_e(a)}</b> {al}<span style="color:var(--dot)"> · </span><b>{_e(b)}</b> {bl}</td>')
    cells.append(f'<td>{status_cell(p)}</td>')
    return '<tr>' + ''.join(cells) + '</tr>'


def header_cells(layout, show_stat) -> str:
    if layout == 'board':
        th = ['<th class="num">Rank</th>', '<th>Player</th>', '<th class="num">Pos</th>']
        if show_stat:
            th.append('<th>Key Stat</th>')
        th += ['<th>Status</th>', '<th>Entered</th>']
        return '<tr>' + ''.join(th) + '</tr>'
    th = ['<th class="num">Rank</th>', '<th>Player</th>', '<th class="num">Pos</th>',
          '<th>Prev School</th>', '<th>Conf</th>', '<th>Class</th>']
    if show_stat:
        th.append('<th>Key Stat</th>')
    th.append('<th>Status</th>')
    return '<tr>' + ''.join(th) + '</tr>'


# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='Portal Entrant List', layout='wide')
st.title('Portal Entrant List')
st.caption('Multi-entrant, print-ready table graphic. Filter on the left; the board '
           'below prints to PDF and exports to PNG. Companion to the single-player card.')

df = build_entrants()

with st.sidebar:
    st.markdown('### Filters')
    q = st.text_input('Search', placeholder='player, school, conference…').strip().lower()
    sport = st.radio('Sport', ['All', 'Baseball', 'Softball'], horizontal=True)
    conf_opts = ['All'] + sorted([c for c in df['conf'].dropna().unique() if c],
                                  key=lambda c: (CONF_ORDER.index(c) if c in CONF_ORDER else 99, c))
    conf = st.selectbox('Conference', conf_opts)
    pos_opts = ['All'] + sorted([p for p in df['pos'].dropna().unique() if p and p != '—'],
                                key=lambda p: (POS_ORDER.index(p) if p in POS_ORDER else 99, p))
    pos = st.selectbox('Position', pos_opts)
    status_f = st.radio('Status', ['All', 'Committed', 'Available'], horizontal=True)
    st.markdown('### Layout')
    layout = {'Ledger': 'ledger', 'Board': 'board', 'Index': 'index'}[
        st.radio('Direction', ['Ledger', 'Board', 'Index'],
                 help='Ledger = reference look · Board = two-line scouting board · Index = print sheet')]
    group_by = {'None': 'none', 'Conference': 'conf', 'Position': 'pos', 'Prev school': 'school'}[
        st.selectbox('Group by', ['None', 'Conference', 'Position', 'Prev school'])]
    show_stat = st.toggle('Key-stat column', value=True)
    highlight_top = st.toggle('Highlight top 10', value=True)
    limit = st.number_input('Max rows', min_value=5, max_value=500, value=50, step=5,
                            help='Caps the table; counts above reflect the full filtered pool.')

# Pool = search + conference + position (drives summary counts).
pool = df.copy()
if q:
    pool = pool[pool.apply(lambda p: q in str(p['name']).lower() or q in str(p['prevSchool']).lower()
                           or q in str(p['to']).lower() or q in str(p['conf']).lower(), axis=1)]
if conf != 'All':
    pool = pool[pool['conf'] == conf]
if pos != 'All':
    pool = pool[pool['pos'] == pos]

counts = {
    'total': len(pool),
    'committed': int((pool['status'] == 'Committed').sum()),
    'available': int((pool['status'] == 'Uncommitted').sum()),
    'baseball': int((pool['sport'] == 'Baseball').sum()),
    'softball': int((pool['sport'] == 'Softball').sum()),
}

# Table set = pool + sport + status, sorted by rank, limited.
view = pool.copy()
if sport != 'All':
    view = view[view['sport'] == sport]
if status_f == 'Committed':
    view = view[view['status'] == 'Committed']
elif status_f == 'Available':
    view = view[view['status'] == 'Uncommitted']
view = view.sort_values('rank', ascending=True, na_position='last').head(int(limit))

# ── Build the rows (with optional grouping) ──────────────────────────────────
col_count = (5 + int(show_stat)) if layout == 'board' else (7 + int(show_stat))


def group_blocks(rows_df):
    if group_by == 'none':
        return [(None, rows_df)]
    key_col = {'conf': 'conf', 'pos': 'pos', 'school': 'prevSchool'}[group_by]
    keys = list(rows_df[key_col].fillna('—').unique())
    if group_by == 'conf':
        keys.sort(key=lambda k: (CONF_ORDER.index(k) if k in CONF_ORDER else 99, k))
    elif group_by == 'pos':
        keys.sort(key=lambda k: (POS_ORDER.index(k) if k in POS_ORDER else 99, k))
    else:
        keys.sort(key=lambda k: (-int((rows_df[key_col].fillna('—') == k).sum()), k))
    return [(k, rows_df[rows_df[key_col].fillna('—') == k]) for k in keys]


body = []
if view.empty:
    body.append(f'<tr><td class="empty" colspan="{col_count}">No entrants match these filters.</td></tr>')
else:
    for gkey, grows in group_blocks(view):
        if gkey is not None:
            label = (f'Position · {gkey}' if group_by == 'pos' else gkey)
            n = len(grows)
            body.append(f'<tr class="grouphd"><td colspan="{col_count}">{_e(label)}'
                        f'<span class="gcount">{n} {"PLAYER" if n == 1 else "PLAYERS"}</span></td></tr>')
        for _, p in grows.iterrows():
            body.append(render_row(p, layout, show_stat, highlight_top))

# Masthead / summary / result line
updated = ''
if not df.empty:
    d = pd.to_datetime(df['entered'], errors='coerce').max()
    d = d if pd.notna(d) else pd.Timestamp(datetime.now())
    updated = f'{d.strftime("%b")} {d.day}, {d.year}'
sport_lbl = 'BASEBALL + SOFTBALL' if sport == 'All' else sport.upper()

summary_html = (
    f'<div class="stat"><div class="stat-num"><span class="accent">{counts["total"]:,}</span></div>'
    f'<div class="stat-lab">Total Entrants</div></div>'
    f'<div class="stat {"active" if status_f == "Committed" else ""}"><div class="stat-num">{counts["committed"]:,}</div>'
    f'<div class="stat-lab">Committed</div></div>'
    f'<div class="stat {"active" if status_f == "Available" else ""}"><div class="stat-num">{counts["available"]:,}</div>'
    f'<div class="stat-lab">Available</div></div>'
    f'<div class="stat {"active" if sport == "Baseball" else ""}"><div class="stat-num">{counts["baseball"]:,}</div>'
    f'<div class="stat-lab">Baseball</div></div>'
    f'<div class="stat {"active" if sport == "Softball" else ""}"><div class="stat-num">{counts["softball"]:,}</div>'
    f'<div class="stat-lab">Softball</div></div>'
)

filt_bits = []
if conf != 'All':
    filt_bits.append(conf)
if pos != 'All':
    filt_bits.append(pos)
if sport != 'All':
    filt_bits.append(sport)
if status_f != 'All':
    filt_bits.append(status_f)
result_line = f'SHOWING {len(view)} OF {counts["total"]:,} ENTRANTS' + (' · ' + ' · '.join(filt_bits) if filt_bits else '')

page_html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"/>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700;800&family=Spline+Sans+Mono:wght@400;500;600&display=swap" rel="stylesheet"/>
<style>{DESIGN_CSS}</style></head><body>
<div class="wrap" id="capture">
  <header class="mast">
    <div><div class="brand"><span class="b64">64</span><span class="bword">nalytics</span></div></div>
    <div class="mast-meta"><h1>Transfer Portal — Entrant List</h1>
      <div class="sub">{_e(sport_lbl)} · UPDATED {_e(updated.upper())}</div></div>
  </header>
  <div class="summary">{summary_html}</div>
  <div class="resultline">{_e(result_line)}</div>
  <div class="tablecard lay-{layout}">
    <table class="portal"><thead>{header_cells(layout, show_stat)}</thead>
    <tbody>{''.join(body)}</tbody></table>
  </div>
  <div class="foot"><span>64 ANALYTICS · TRANSFER PORTAL TRACKER</span><span>{_e(updated.upper())}</span></div>
</div>
<div class="btn-row">
  <button onclick="window.print()">Print / PDF</button>
  <button onclick="window.dlPNG(this)">Download PNG</button>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<script>
window.dlPNG = async function(btn){{
  var el = document.getElementById('capture'); if(!el) return;
  if(btn){{ btn.disabled = true; var t = btn.textContent; btn.textContent='Rendering…'; }}
  try{{
    if(document.fonts && document.fonts.ready) await document.fonts.ready;
    var canvas = await html2canvas(el, {{scale:2, useCORS:true, backgroundColor:'#f3e6e8'}});
    var a = document.createElement('a'); a.download='portal_entrant_list.png';
    a.href = canvas.toDataURL('image/png'); document.body.appendChild(a); a.click(); document.body.removeChild(a);
  }} finally {{ if(btn){{ btn.disabled=false; btn.textContent = t || 'Download PNG'; }} }}
}};
</script></body></html>"""

st.markdown(f'**{len(view)}** shown · **{counts["total"]:,}** in filtered pool · '
            f'layout: {layout.title()}' + (f' · grouped by {group_by}' if group_by != 'none' else ''))
_h = min(2400, 360 + len(body) * 46)
components.html(page_html, height=_h, scrolling=True)
