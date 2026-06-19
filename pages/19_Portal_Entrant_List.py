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
ASSETS = _APP_DIR / 'assets' / 'portal-entrant'


@st.cache_data
def poster_data_url():
    """Base64 the rankings poster once (cached) for use as the graphic background."""
    import base64
    p = ASSETS / 'rankings-poster.jpg'
    return 'data:image/jpeg;base64,' + base64.b64encode(p.read_bytes()).decode()

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
            'toConf': (tconf(new_tid) if committed else ''),
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


# Graphic CSS — dark "Portal 2026 Rankings" poster theme. All Inter. The table
# is sized in em off a Python-computed --fs so any row count fills the poster's
# middle zone. __POSTER__ is replaced with the base64 background at assembly.
DESIGN_CSS = """
:root{--red:#d6232f;--red-deep:#9a303e;--text:#ededed;--muted:#b9bfc7;--soft:#838b95;
--green:#5bd39a;--blue:#82b9ff;--line:rgba(255,255,255,.10);--zebra:rgba(255,255,255,.04);
--sans:'Inter',system-ui,-apple-system,sans-serif;}
*{box-sizing:border-box;}html,body{margin:0;padding:0;}
body{font-family:var(--sans);background:#0b0c0f;-webkit-font-smoothing:antialiased;}
.poster{position:relative;width:1054px;height:1492px;margin:0 auto;
background-image:url('__POSTER__');background-size:100% 100%;background-repeat:no-repeat;font-family:var(--sans);}
.content{position:absolute;top:30%;left:9%;right:9%;bottom:6%;display:flex;flex-direction:column;overflow:hidden;
background:rgba(9,10,13,0.66);border:1px solid rgba(255,255,255,.07);border-radius:6px;padding:14px 18px;}
.resultline{font-size:11px;font-weight:700;letter-spacing:1.6px;text-transform:uppercase;color:var(--muted);margin:0 2px 10px;}
table.portal{width:100%;border-collapse:collapse;font-size:var(--fs);color:var(--text);text-shadow:0 1px 2px rgba(0,0,0,.65);line-height:1.2;}
table.portal thead th{font-size:.6em;text-transform:uppercase;letter-spacing:1.1px;font-weight:800;color:#fff;
text-align:left;padding:.5em .7em;border-bottom:2px solid var(--red);white-space:nowrap;}
table.portal thead th.num{text-align:center;}
table.portal tbody td{padding:var(--pady) .7em;font-size:1em;vertical-align:middle;font-weight:500;white-space:nowrap;}
.colrank{text-align:center;width:9%;}.colpos{text-align:center;width:8%;}
.rank-fig{font-weight:800;font-size:1.08em;color:var(--red);letter-spacing:-.5px;}
.pos-tag{font-weight:700;font-size:.92em;color:var(--blue);}
.pname{font-weight:800;color:#fff;font-size:1.02em;white-space:nowrap;letter-spacing:-.2px;}
.pschool{font-weight:600;color:var(--text);}
.pconf{font-size:.74em;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);white-space:nowrap;}
.pclass{font-size:.78em;color:var(--soft);font-weight:600;}
.pdate{font-size:.78em;color:var(--soft);white-space:nowrap;font-weight:600;}
.pill{display:inline-flex;align-items:center;gap:5px;font-size:.72em;font-weight:800;letter-spacing:.3px;padding:.18em .6em;border-radius:20px;white-space:nowrap;text-shadow:none;}
.pill .dot{width:6px;height:6px;border-radius:50%;}
.pill.committed{background:rgba(91,211,154,.18);color:var(--green);}.pill.committed .dot{background:var(--green);}
.pill.uncommitted{background:rgba(130,185,255,.16);color:var(--blue);}.pill.uncommitted .dot{background:var(--blue);}
.pill.withdrawn{background:rgba(214,35,47,.20);color:#ff8a90;}.pill.withdrawn .dot{background:var(--red);}
.commit-to{font-weight:700;color:#fff;font-size:.92em;}
.commit-arrow{color:var(--green);margin:0 4px;font-weight:800;}
.statline{font-size:.84em;color:var(--muted);white-space:nowrap;}
.statline b{color:#fff;font-weight:700;}
.lay-ledger tbody tr{border-bottom:1px solid var(--line);}
.lay-ledger tbody tr:last-child{border-bottom:none;}
.lay-board tbody tr{border-bottom:1px solid var(--line);}
.lay-board tbody tr:nth-child(even){background:var(--zebra);}
.lay-board .rank-badge{display:inline-flex;align-items:center;justify-content:center;width:2.3em;height:2.3em;background:#1b1e25;color:#fff;font-weight:800;font-size:1em;border-radius:3px;text-shadow:none;}
.lay-board .rank-badge.top{background:var(--red);}
.lay-board .submeta{font-size:.72em;color:var(--soft);font-weight:600;margin-top:2px;display:flex;gap:6px;align-items:center;}
.lay-board .submeta .sep{color:#555;}
.lay-index tbody tr{border-bottom:1px solid var(--line);}
.lay-index .rank-fig{color:#fff;}
.grouphd td{background:var(--red)!important;color:#fff;padding:.4em .7em!important;font-size:.7em;font-weight:800;text-transform:uppercase;letter-spacing:1.4px;text-shadow:none;}
.grouphd .gcount{float:right;font-weight:700;letter-spacing:0;color:rgba(255,255,255,.88);}
.empty{padding:40px 20px;text-align:center;color:var(--soft);font-size:14px;}
.btn-row{text-align:center;margin:16px 0;}
.btn-row button{padding:10px 22px;background:var(--red);color:#fff;border:none;border-radius:3px;font-family:var(--sans);font-weight:700;font-size:13px;letter-spacing:.15em;text-transform:uppercase;cursor:pointer;margin:0 5px;box-shadow:0 4px 12px rgba(0,0,0,.4);}
@media print{@page{margin:0;}.btn-row{display:none!important;}.poster{-webkit-print-color-adjust:exact;print-color-adjust:exact;}}
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
    conf_basis = st.radio('Conference basis', ['Entered', 'Committed to'], horizontal=True,
                          help="Entered = the conference a player came FROM (where they entered the "
                               "portal). Committed to = the conference of the school they committed TO "
                               "(only committed players have one). Pick 'Committed to' for boards like "
                               "'top SEC commits'.")
    conf_col = 'toConf' if conf_basis == 'Committed to' else 'conf'
    conf_opts = ['All'] + sorted([c for c in df[conf_col].dropna().unique() if c],
                                  key=lambda c: (CONF_ORDER.index(c) if c in CONF_ORDER else 99, c))
    conf = st.selectbox('Committed-to conference' if conf_col == 'toConf' else 'Entered conference',
                        conf_opts)
    pos_opts = ['All'] + sorted([p for p in df['pos'].dropna().unique() if p and p != '—'],
                                key=lambda p: (POS_ORDER.index(p) if p in POS_ORDER else 99, p))
    pos = st.selectbox('Position', pos_opts)
    status_f = st.radio('Status', ['All', 'Committed', 'Available'], horizontal=True)
    st.markdown('### Sort')
    sort_by = st.selectbox('Sort by', ['Rank', 'Team', 'Name', 'Conference', 'Position'])
    sort_dir = st.radio('Order', ['Asc', 'Desc'], horizontal=True,
                        help='Team sort groups by school, then orders by rank within each.')
    st.markdown('### Layout')
    layout = {'Ledger': 'ledger', 'Board': 'board', 'Index': 'index'}[
        st.radio('Direction', ['Ledger', 'Board', 'Index'],
                 help='Ledger = reference look · Board = two-line scouting board · Index = print sheet')]
    group_by = {'None': 'none', 'Conference': 'conf', 'Position': 'pos', 'Prev school': 'school'}[
        st.selectbox('Group by', ['None', 'Conference', 'Position', 'Prev school'])]
    show_stat = st.toggle('Key-stat column', value=True)
    highlight_top = st.toggle('Highlight top 10', value=True)
    st.markdown('### Pagination')
    per_page = st.number_input('Rows per page', min_value=5, max_value=30, value=20, step=1,
                               help='Rows per graphic. For a long list, build one page at a time: '
                                    'set this, pick the Page below, download, then step Page +1.')

# Pool = search + conference + position (drives summary counts).
pool = df.copy()
if q:
    pool = pool[pool.apply(lambda p: q in str(p['name']).lower() or q in str(p['prevSchool']).lower()
                           or q in str(p['to']).lower() or q in str(p['conf']).lower()
                           or q in str(p['toConf']).lower(), axis=1)]
if conf != 'All':
    pool = pool[pool[conf_col] == conf]
if pos != 'All':
    pool = pool[pool['pos'] == pos]

counts = {
    'total': len(pool),
    'committed': int((pool['status'] == 'Committed').sum()),
    'available': int((pool['status'] == 'Uncommitted').sum()),
    'baseball': int((pool['sport'] == 'Baseball').sum()),
    'softball': int((pool['sport'] == 'Softball').sum()),
}

# Table set = pool + sport + status, sorted, limited.
view = pool.copy()
if sport != 'All':
    view = view[view['sport'] == sport]
if status_f == 'Committed':
    view = view[view['status'] == 'Committed']
elif status_f == 'Available':
    view = view[view['status'] == 'Uncommitted']
_asc = (sort_dir == 'Asc')
_sortcol = {'Rank': 'rank', 'Team': 'prevSchool', 'Name': 'name',
            'Conference': conf_col, 'Position': 'pos'}[sort_by]
if _sortcol == 'rank':
    view = view.sort_values('rank', ascending=_asc, na_position='last')
else:
    # Group by the chosen field, then rank within it (so "by team" reads cleanly).
    view = view.assign(_k=view[_sortcol].astype(str).str.lower())
    view = view.sort_values(['_k', 'rank'], ascending=[_asc, True], na_position='last').drop(columns='_k')
view_all = view                       # full sorted, filtered set
total_rows = len(view_all)
per_page_i = max(1, int(per_page))
n_pages = max(1, (total_rows + per_page_i - 1) // per_page_i)
page = int(st.sidebar.number_input(
    'Page', min_value=1, max_value=n_pages, value=1, step=1,
    help=f'{n_pages} page(s) of {per_page_i} for the current {total_rows} filtered entrants. '
         'Download each page, then step to the next.'))
_start = (page - 1) * per_page_i
view = view_all.iloc[_start:_start + per_page_i]

# ── Build the rows (with optional grouping) ──────────────────────────────────
col_count = (5 + int(show_stat)) if layout == 'board' else (7 + int(show_stat))


def group_blocks(rows_df):
    if group_by == 'none':
        return [(None, rows_df)]
    key_col = {'conf': conf_col, 'pos': 'pos', 'school': 'prevSchool'}[group_by]
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

# Result line (the poster supplies the masthead/branding; table only).
filt_bits = []
if conf != 'All':
    filt_bits.append(f'{conf} commits' if conf_col == 'toConf' else conf)
if pos != 'All':
    filt_bits.append(pos)
if sport != 'All':
    filt_bits.append(sport)
if status_f != 'All':
    filt_bits.append(status_f)
_sortlbl = {'Rank': 'BY RANK', 'Team': 'BY TEAM', 'Name': 'A–Z',
            'Conference': 'BY CONF', 'Position': 'BY POS'}[sort_by]
_rng = f'{_start + 1}–{_start + len(view)} OF {total_rows:,}'
_pg = f' · PAGE {page}/{n_pages}' if n_pages > 1 else ''
result_line = (f'{_rng}{_pg} · {_sortlbl}'
               + (' · ' + ' · '.join(b.upper() for b in filt_bits) if filt_bits else ''))

# Auto-fit: size rows so the chosen count fits the middle zone WITHOUT clipping.
# Row height ≈ 2·pady + fs·line-height; line-height is pinned to 1.2 in CSS, so
# solving fs from the zone budget is accurate (no more bottom-row cutoff).
ZONE_H = 905.0        # usable zone height (px) inside the scrim padding; runs near full page
RESULT_H = 28.0       # result line above the table
HEADER_W = 2.2        # header-row height in fs units (1 line + 2·pady; was under-counted at 1.5)
# data-row height in fs units. Board = 2 text lines + a 2.3em rank badge + 2·pady,
# which is ~3.4·fs — the old 3.1 under-counted it, so the 20th row clipped off the
# bottom. Ledger/Index are single-line (~2.2·fs).
ROW_W = 3.45 if layout == 'board' else 2.2
rows_total = max(1, len(body))
fs = (ZONE_H - RESULT_H) / (HEADER_W + ROW_W * rows_total)
fs = max(8.5, min(18.0, fs))
pady = max(1.5, fs * 0.5)

css = DESIGN_CSS.replace('__POSTER__', poster_data_url())
page_html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"/>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet"/>
<style>{css}</style></head><body>
<div class="poster" id="capture">
  <div class="content" style="--fs:{fs:.1f}px;--pady:{pady:.1f}px">
    <div class="resultline">{_e(result_line)}</div>
    <div class="lay-{layout}">
      <table class="portal"><thead>{header_cells(layout, show_stat)}</thead>
      <tbody>{''.join(body)}</tbody></table>
    </div>
  </div>
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
    var canvas = await html2canvas(el, {{scale:2, useCORS:true, backgroundColor:'#0b0c0f'}});
    var a = document.createElement('a'); a.download='portal_2026_rankings_p{page}of{n_pages}.png';
    a.href = canvas.toDataURL('image/png'); document.body.appendChild(a); a.click(); document.body.removeChild(a);
  }} finally {{ if(btn){{ btn.disabled=false; btn.textContent = t || 'Download PNG'; }} }}
}};
</script></body></html>"""

st.markdown(f'**Page {page}/{n_pages}** · rows **{_start + 1}–{_start + len(view)}** of '
            f'**{total_rows:,}** filtered · sort: {sort_by} {sort_dir} · layout: {layout.title()}'
            + (f' · grouped by {group_by}' if group_by != 'none' else '')
            + ('' if n_pages == 1 else f' · download p{page}, then step Page → for the rest'))
components.html(page_html, height=1620, scrolling=True)
