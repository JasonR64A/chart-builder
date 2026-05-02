"""
Portal Entrant Graphic Builder — produce a print-style graphic card for a
transfer portal entrant. Auto-fills from portal_archive + hitting.csv, all
fields editable, renders the design template, and offers a PNG download.
"""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _APP_DIR / 'data'
ASSETS_DIR = _APP_DIR / 'assets' / 'portal-entrant'
TEMPLATE_PATH = ASSETS_DIR / 'template.html'

CLASS_SHORT = {
    'Freshman': 'FR.', 'Sophomore': 'SO.', 'Junior': 'JR.', 'Senior': 'SR.',
    'FR': 'FR.', 'SO': 'SO.', 'JR': 'JR.', 'SR': 'SR.',
    'Graduate': 'GR.', 'Grad': 'GR.', 'GR': 'GR.', 'Redshirt Senior': 'SR.',
}
CLASS_LONG = {
    'FR.': 'FRESHMAN', 'SO.': 'SOPHOMORE', 'JR.': 'JUNIOR', 'SR.': 'SENIOR',
    'GR.': 'GRADUATE',
}

st.set_page_config(page_title='Portal Entrant Graphic', layout='wide')
st.title('Portal Entrant Graphic')
st.caption('Build a shareable transfer-portal graphic for a single player.')


# ── Data loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_players() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / 'players.csv', encoding='latin1', dtype=str).fillna('')


@st.cache_data
def load_teams() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / 'teams.csv', dtype=str).fillna('')


@st.cache_data
def load_hitting() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / 'hitting.csv', low_memory=False)


players = load_players()
teams = load_teams()
hitting = load_hitting()

# Join team name + sport onto players for display
team_lookup = teams[['id', 'name', 'sport']].rename(
    columns={'id': 'team_id', 'name': 'team_name', 'sport': 'sport'}
)
players_ext = players.merge(team_lookup, on='team_id', how='left').fillna('')


# ── Player picker ───────────────────────────────────────────────────────────
st.subheader('1. Pick a player')
query = st.text_input('Search players by name',
                        placeholder='Start typing a name...',
                        key='player_search')

prec = None
school_name = ''
sport_filter = ''
if query:
    q = query.lower().strip()
    hits = players_ext[players_ext['player_name'].str.lower().str.contains(q, na=False)]
    hits = hits.head(50)
    if hits.empty:
        st.info('No matches — type more of the name.')
    else:
        opts = hits.apply(
            lambda r: f"{r['player_name']} — {r['team_name']} ({r['position'] or '—'}, {r['classification'] or '—'})",
            axis=1
        ).tolist()
        pick = st.selectbox(f'Matches ({len(hits)})', opts, key='player_pick')
        prec = hits.iloc[opts.index(pick)]
        school_name = prec.get('team_name', '')
        sport_filter = (prec.get('sport') or '').lower()


def career_stats(player_id: str):
    if not player_id or hitting.empty:
        return pd.DataFrame()
    df = hitting[hitting['player_id'] == int(player_id)].copy() if str(player_id).isdigit() else pd.DataFrame()
    return df.sort_values('year', ascending=False)


def fmt_slash(v, digits=3):
    try:
        f = float(v)
        s = f'{f:.{digits}f}'
        # .301 style (drop leading 0)
        if s.startswith('0.'):
            return s[1:]
        if s.startswith('-0.'):
            return '-' + s[2:]
        return s
    except Exception:
        return '—'


def season_stat(row, key, default=0):
    if row is None or key not in row:
        return default
    v = row[key]
    if pd.isna(v):
        return default
    try:
        return int(v)
    except Exception:
        try:
            return float(v)
        except Exception:
            return default


# Build defaults from the selected row
defaults = {
    'name': '', 'school': '', 'school_abbr': '', 'pos': '', 'class_short': '',
    'class_long': '', 'bat': '', 'throw': '',
    'season_year': 2026, 'season_gp': 0, 'season_pa': 0,
    's_avg': '.000', 's_obp': '.000', 's_slg': '.000', 's_ops': '.000',
    's_hr': 0, 's_rbi': 0, 's_r': 0, 's_sb': 0,
    's_h': 0, 's_ab': 0, 's_2b': 0, 's_3b': 0, 's_tb': 0,
    's_bb': 0, 's_hbp': 0, 's_so': 0,
    'career_slash': '.000 / .000 / .000',
    'career_rows': [],           # list of dicts: year, g, avg, obp, slg, ops, hr, rbi, hAb, bbK
    'career_total': None,        # dict same shape
    'career_subtitle': '',
    'photo_src_override': None,  # bytes for uploaded photo
}

if prec is not None:
    defaults['name'] = prec.get('player_name', '')
    defaults['school'] = school_name
    defaults['school_abbr'] = ''.join(w[0] for w in school_name.split()[:3]).upper() if school_name else ''
    defaults['pos'] = (prec.get('position') or '').upper()
    cls = prec.get('classification') or ''
    defaults['class_short'] = CLASS_SHORT.get(cls, cls.upper()[:3]+'.' if cls else '')
    defaults['class_long'] = CLASS_LONG.get(defaults['class_short'], cls.upper())
    defaults['bat'] = (prec.get('bat') or '').upper()
    defaults['throw'] = (prec.get('throw') or '').upper()

    # Career stats
    try:
        pid = int(prec.get('id') or 0)
    except Exception:
        pid = 0
    hist = career_stats(pid) if pid else pd.DataFrame()
    if not hist.empty:
        cur = hist[hist['year'] == 2026]
        cur_row = cur.iloc[0] if not cur.empty else None
        if cur_row is not None:
            defaults['season_gp'] = season_stat(cur_row, 'games_played', 0)
            defaults['season_pa'] = season_stat(cur_row, 'plate_appearances', 0)
            defaults['s_avg'] = fmt_slash(cur_row.get('batting_average'))
            defaults['s_obp'] = fmt_slash(cur_row.get('on_base_percentage'))
            defaults['s_slg'] = fmt_slash(cur_row.get('slugging_percentage'))
            defaults['s_ops'] = fmt_slash(cur_row.get('on_base_plus_slugging'))
            defaults['s_hr'] = season_stat(cur_row, 'home_runs')
            defaults['s_rbi'] = season_stat(cur_row, 'runs_batted_in')
            defaults['s_r'] = season_stat(cur_row, 'runs_scored')
            defaults['s_sb'] = season_stat(cur_row, 'stolen_bases')
            defaults['s_h'] = season_stat(cur_row, 'hits')
            defaults['s_ab'] = season_stat(cur_row, 'at_bats')
            defaults['s_2b'] = season_stat(cur_row, 'doubles')
            defaults['s_3b'] = season_stat(cur_row, 'triples')
            defaults['s_tb'] = season_stat(cur_row, 'total_bases')
            defaults['s_bb'] = season_stat(cur_row, 'walks')
            defaults['s_hbp'] = season_stat(cur_row, 'hit_by_pitch')
            defaults['s_so'] = season_stat(cur_row, 'strikeouts')

        # Career rows + totals
        tot = {'ab': 0, 'h': 0, 'bb': 0, 'k': 0, 'hr': 0, 'rbi': 0, 'g': 0,
                 'hbp': 0, 'sf': 0, 'tb': 0, 'pa': 0}
        rows = []
        for _, yr in hist.iterrows():
            g = season_stat(yr, 'games_played')
            ab = season_stat(yr, 'at_bats')
            h  = season_stat(yr, 'hits')
            bb = season_stat(yr, 'walks')
            k  = season_stat(yr, 'strikeouts')
            hr = season_stat(yr, 'home_runs')
            rbi = season_stat(yr, 'runs_batted_in')
            tb = season_stat(yr, 'total_bases')
            hbp = season_stat(yr, 'hit_by_pitch')
            sf = season_stat(yr, 'sac_fly')
            rows.append({
                'year': int(yr['year']) if not pd.isna(yr['year']) else '',
                'g': g,
                'avg': fmt_slash(yr.get('batting_average')),
                'obp': fmt_slash(yr.get('on_base_percentage')),
                'slg': fmt_slash(yr.get('slugging_percentage')),
                'ops': fmt_slash(yr.get('on_base_plus_slugging')),
                'hr': hr, 'rbi': rbi,
                'hAb': f'{h}/{ab}',
                'bbK': f'{bb}/{k}',
                # Hidden raw fields used to recompute OBP/SLG/OPS in the totals row:
                '_tb': tb, '_hbp': hbp, '_sf': sf,
            })
            for key, val in zip(('g','ab','h','bb','k','hr','rbi','tb','hbp','sf'),
                                  (g, ab, h, bb, k, hr, rbi, tb, hbp, sf)):
                tot[key] += val
        defaults['career_rows'] = rows
        # Totals
        if tot['ab'] > 0:
            avg = tot['h'] / tot['ab'] if tot['ab'] else 0
            pa_est = tot['ab'] + tot['bb'] + tot['hbp'] + tot['sf']
            obp = (tot['h'] + tot['bb'] + tot['hbp']) / pa_est if pa_est else 0
            slg = tot['tb'] / tot['ab'] if tot['ab'] else 0
            ops = obp + slg
            defaults['career_total'] = {
                'year': 'TOTAL', 'g': tot['g'],
                'avg': fmt_slash(avg), 'obp': fmt_slash(obp),
                'slg': fmt_slash(slg), 'ops': fmt_slash(ops),
                'hr': tot['hr'], 'rbi': tot['rbi'],
                'hAb': f"{tot['h']}/{tot['ab']}",
                'bbK': f"{tot['bb']}/{tot['k']}",
            }
            defaults['career_slash'] = f"{defaults['career_total']['avg']} / {defaults['career_total']['obp']} / {defaults['career_total']['slg']}"
            defaults['career_subtitle'] = f"{len(rows)} {'SEASON' if len(rows)==1 else 'SEASONS'} AT {defaults['school'].upper()}"


# ── Editable form ───────────────────────────────────────────────────────────
st.subheader('2. Review & edit fields')

with st.expander('Identity', expanded=True):
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    name = c1.text_input('Full name', value=defaults['name'])
    school = c2.text_input('School (full)', value=defaults['school'])
    school_abbr = c3.text_input('School abbrev.', value=defaults['school_abbr'])
    pos = c4.text_input('Position', value=defaults['pos'])

    c5, c6, c7, c8 = st.columns(4)
    class_short = c5.text_input('Class (short, e.g. JR.)', value=defaults['class_short'])
    class_long = c6.text_input('Class (long, e.g. JUNIOR)', value=defaults['class_long'])
    bat = c7.text_input('Bats', value=defaults['bat'])
    throw = c8.text_input('Throws', value=defaults['throw'])

    photo_upload = st.file_uploader('Player photo (optional — leave blank to use sample)',
                                     type=['jpg', 'jpeg', 'png', 'webp'])

with st.expander('Season line', expanded=True):
    c1, c2, c3 = st.columns(3)
    season_year = c1.number_input('Season year', min_value=2000, max_value=2100,
                                    value=int(defaults['season_year']))
    season_gp = c2.number_input('Games played', min_value=0, value=int(defaults['season_gp']))
    season_pa = c3.number_input('Plate appearances', min_value=0, value=int(defaults['season_pa']))

    c1, c2, c3, c4 = st.columns(4)
    s_avg = c1.text_input('AVG', value=defaults['s_avg'])
    s_obp = c2.text_input('OBP', value=defaults['s_obp'])
    s_slg = c3.text_input('SLG', value=defaults['s_slg'])
    s_ops = c4.text_input('OPS', value=defaults['s_ops'])

    c1, c2, c3, c4 = st.columns(4)
    s_hr = c1.number_input('HR', value=int(defaults['s_hr']))
    s_rbi = c2.number_input('RBI', value=int(defaults['s_rbi']))
    s_r = c3.number_input('R', value=int(defaults['s_r']))
    s_sb = c4.number_input('SB', value=int(defaults['s_sb']))

    c1, c2, c3, c4 = st.columns(4)
    s_h = c1.number_input('H', value=int(defaults['s_h']))
    s_ab = c2.number_input('AB', value=int(defaults['s_ab']))
    s_2b = c3.number_input('2B', value=int(defaults['s_2b']))
    s_3b = c4.number_input('3B', value=int(defaults['s_3b']))

    c1, c2, c3, c4 = st.columns(4)
    s_tb = c1.number_input('TB', value=int(defaults['s_tb']))
    s_bb = c2.number_input('BB', value=int(defaults['s_bb']))
    s_hbp = c3.number_input('HBP', value=int(defaults['s_hbp']))
    s_so = c4.number_input('SO', value=int(defaults['s_so']))

with st.expander('Career by year (editable)', expanded=False):
    st.caption('Edit, add, or remove rows. Totals row is recomputed automatically.')
    default_df = pd.DataFrame(defaults['career_rows']) if defaults['career_rows'] else pd.DataFrame(
        [{'year': season_year, 'g': season_gp,
          'avg': s_avg, 'obp': s_obp, 'slg': s_slg, 'ops': s_ops,
          'hr': s_hr, 'rbi': s_rbi,
          'hAb': f'{s_h}/{s_ab}', 'bbK': f'{s_bb}/{s_so}',
          '_tb': s_tb, '_hbp': s_hbp, '_sf': 0}]
    )
    edited = st.data_editor(
        default_df, num_rows='dynamic', use_container_width=True,
        key='career_editor',
        column_config={
            # Hidden raw fields used to recompute OBP/SLG/OPS totals
            '_tb': st.column_config.NumberColumn('TB', help='Total bases — used for SLG/OPS totals'),
            '_hbp': st.column_config.NumberColumn('HBP', help='Hit by pitch — used for OBP totals'),
            '_sf': st.column_config.NumberColumn('SF', help='Sac fly — used for OBP totals'),
        },
    )
    career_subtitle = st.text_input('Career section subtitle',
                                      value=defaults['career_subtitle'] or f"{len(edited)} SEASON{'S' if len(edited) != 1 else ''} AT {school.upper()}")
    career_slash_override = st.text_input('Career slash (top-right header)',
                                            value=defaults['career_slash'])

with st.expander('Headline/eyebrow', expanded=False):
    eyebrow = st.text_input('Eyebrow', value=f'TRANSFER PORTAL · {season_year}')
    headline = st.text_input('Headline', value='ENTERED THE PORTAL')


# ── Render ──────────────────────────────────────────────────────────────────
def data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    b = path.read_bytes()
    return f'data:{mime or "image/png"};base64,' + base64.b64encode(b).decode()


def build_career_rows_html(rows_df: pd.DataFrame) -> str:
    if rows_df is None or rows_df.empty:
        return ''
    html = []
    for _, r in rows_df.iterrows():
        html.append(
            f'<tr><td>{r.get("year","")}</td><td>{r.get("g","")}</td>'
            f'<td>{r.get("avg","")}</td><td>{r.get("obp","")}</td>'
            f'<td>{r.get("slg","")}</td><td>{r.get("ops","")}</td>'
            f'<td>{r.get("hr","")}</td><td>{r.get("rbi","")}</td>'
            f'<td>{r.get("hAb","")}</td><td>{r.get("bbK","")}</td></tr>'
        )
    # Totals: recompute from per-year raw fields
    tot = {'g': 0, 'hr': 0, 'rbi': 0, 'h': 0, 'ab': 0, 'bb': 0, 'k': 0,
           'tb': 0, 'hbp': 0, 'sf': 0}

    def _int(v):
        try: return int(v)
        except Exception:
            try: return int(float(v))
            except Exception: return 0

    for _, r in rows_df.iterrows():
        tot['g']   += _int(r.get('g'))
        tot['hr']  += _int(r.get('hr'))
        tot['rbi'] += _int(r.get('rbi'))
        hAb = str(r.get('hAb') or '0/0')
        bbK = str(r.get('bbK') or '0/0')
        try:
            h, ab = [int(x) for x in hAb.split('/')]
            tot['h'] += h; tot['ab'] += ab
        except Exception: pass
        try:
            bb, k = [int(x) for x in bbK.split('/')]
            tot['bb'] += bb; tot['k'] += k
        except Exception: pass
        tot['tb']  += _int(r.get('_tb'))
        tot['hbp'] += _int(r.get('_hbp'))
        tot['sf']  += _int(r.get('_sf'))

    if tot['ab'] > 0:
        avg = tot['h'] / tot['ab']
        pa_est = tot['ab'] + tot['bb'] + tot['hbp'] + tot['sf']
        obp = (tot['h'] + tot['bb'] + tot['hbp']) / pa_est if pa_est else 0
        slg = tot['tb'] / tot['ab'] if tot['tb'] else 0
        ops = obp + slg
        avg_s = fmt_slash(avg)
        obp_s = fmt_slash(obp) if tot['tb'] or tot['hbp'] else '—'
        slg_s = fmt_slash(slg) if tot['tb'] else '—'
        ops_s = fmt_slash(ops) if tot['tb'] else '—'
        html.append(
            f'<tr class="totals"><td>TOTAL</td><td>{tot["g"]}</td>'
            f'<td>{avg_s}</td><td>{obp_s}</td><td>{slg_s}</td><td>{ops_s}</td>'
            f'<td>{tot["hr"]}</td><td>{tot["rbi"]}</td>'
            f'<td>{tot["h"]}/{tot["ab"]}</td><td>{tot["bb"]}/{tot["k"]}</td></tr>'
        )
    return '\n'.join(html)


# Photo
if photo_upload is not None:
    photo_src = 'data:' + (photo_upload.type or 'image/jpeg') + ';base64,' \
                + base64.b64encode(photo_upload.getvalue()).decode()
else:
    photo_src = data_url(ASSETS_DIR / 'sample-photo.jpg') if (ASSETS_DIR / 'sample-photo.jpg').exists() else ''

# Other assets → data URLs
bg_url = data_url(ASSETS_DIR / 'foul-line-dirt.png')
circle_url = data_url(ASSETS_DIR / '64-circlelogo-red-white.png')
mark_url = data_url(ASSETS_DIR / '64-logo-red-white.png')

season_slash = f'{s_avg} / {s_obp} / {s_slg}'
meta_line = f'{pos} &nbsp;·&nbsp; {school_abbr or school.upper()[:3]} &nbsp;·&nbsp; {class_long or class_short} &nbsp;·&nbsp; {bat}/{throw}'
season_subtitle = f"{school.upper()} · {int(season_gp)} GP · {int(season_pa)} PA"

# Use recomputed career totals for header career slash if user didn't override
if edited is not None and not edited.empty and defaults['career_total']:
    career_slash = career_slash_override or defaults['career_slash']
else:
    career_slash = career_slash_override

template = TEMPLATE_PATH.read_text(encoding='utf-8')
replacements = {
    '{{NAME}}': name.upper(),
    '{{SCHOOL}}': school.upper(),
    '{{POS}}': pos,
    '{{CLASS_SHORT}}': class_short,
    '{{CLASS_LONG}}': (class_long or class_short).upper(),
    '{{META_LINE}}': meta_line,
    '{{SEASON_SLASH}}': season_slash,
    '{{SEASON_YEAR}}': str(int(season_year)),
    '{{SEASON_SUBTITLE}}': season_subtitle,
    '{{S_AVG}}': s_avg, '{{S_OBP}}': s_obp, '{{S_SLG}}': s_slg, '{{S_OPS}}': s_ops,
    '{{S_HR}}': str(int(s_hr)), '{{S_RBI}}': str(int(s_rbi)),
    '{{S_R}}': str(int(s_r)), '{{S_SB}}': str(int(s_sb)),
    '{{S_H}}': str(int(s_h)), '{{S_AB}}': str(int(s_ab)),
    '{{S_2B}}': str(int(s_2b)), '{{S_3B}}': str(int(s_3b)),
    '{{S_TB}}': str(int(s_tb)), '{{S_BB}}': str(int(s_bb)),
    '{{S_HBP}}': str(int(s_hbp)), '{{S_SO}}': str(int(s_so)),
    '{{CAREER_SLASH}}': career_slash or '—',
    '{{CAREER_SUBTITLE}}': career_subtitle,
    '{{CAREER_ROWS_HTML}}': build_career_rows_html(edited),
    '{{EYEBROW}}': eyebrow,
    '{{HEADLINE}}': headline,
    '{{PHOTO_SRC}}': photo_src,
    'assets/foul-line-dirt.png': bg_url,
    'assets/64-circlelogo-red-white.png': circle_url,
    'assets/64-logo-red-white.png': mark_url,
}
rendered = template
for k, v in replacements.items():
    rendered = rendered.replace(k, str(v))

# Inject html2canvas + PNG download button. Await document.fonts.ready and
# pre-decode all <img> tags before snapshotting, otherwise html2canvas
# captures the frame before the Google Fonts CSS resolves and the PNG
# falls back to system fonts. Button stays disabled while a render is in
# flight to prevent double-clicks producing duplicate downloads.
png_script = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<script>
window.downloadCardPNG = async function(btn) {
  var stage = document.querySelector('.stage');
  if (!stage) return;
  if (btn) { btn.disabled = true; btn.dataset.label = btn.textContent; btn.textContent = 'Rendering…'; }
  try {
    if (document.fonts && document.fonts.ready) {
      await document.fonts.ready;
    }
    var imgs = Array.from(stage.querySelectorAll('img'));
    await Promise.all(imgs.map(function(img) {
      if (img.complete && img.naturalWidth > 0) return Promise.resolve();
      return new Promise(function(res) {
        img.addEventListener('load', res, { once: true });
        img.addEventListener('error', res, { once: true });
      });
    }));
    var canvas = await html2canvas(stage, { scale: 2, useCORS: true, allowTaint: true });
    var a = document.createElement('a');
    a.download = 'portal_entrant.png';
    a.href = canvas.toDataURL('image/png');
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = btn.dataset.label || 'Download PNG'; }
  }
};
</script>
"""
png_btn = """
<div style="text-align:center;margin:16px 0;">
  <button onclick="window.downloadCardPNG(this)" style="
    padding:10px 24px;background:#b11f2c;color:#fff;border:none;border-radius:4px;
    font-family:'Barlow Condensed','Inter',sans-serif;font-weight:700;font-size:14px;
    letter-spacing:.2em;text-transform:uppercase;cursor:pointer;
    box-shadow:0 4px 12px rgba(0,0,0,.3);">
    Download PNG
  </button>
</div>
"""
rendered = rendered.replace('</body>', f'{png_script}{png_btn}</body>')

st.subheader('3. Preview')
components.html(rendered, height=1040, scrolling=False)
