"""Portal Team Rankings — social/print graphic (Instagram 1080x1350, Twitter 1600x900).

Leaderboard of the top-N transfer-portal recruiting classes by
sixty_four_rating_portal_team (from portal_rank_team.csv, per sport). Same dark
64A aesthetic as the Top 10 Portal Players / PBP Analytics boards: team logos,
embedded condensed fonts (so html2canvas exports cleanly), gold top-3, per-team
accent tint from the logo palette. JUCO / NAIA / Drafted-Pro buckets are excluded
(not real programs). Click Download PNG to export.
"""
import base64
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

APP_DIR = Path(__file__).resolve().parent.parent
DATA = APP_DIR / 'data'
LOGO_DIR = APP_DIR / 'team_logos_512'
BRAND = APP_DIR / 'assets' / 'portal-entrant'
FONT_DIR = APP_DIR / 'assets' / 'fonts'
BUCKETS = {'960', '961', '2269'}   # JUCO, NAIA, Drafted/Pro — not real programs

st.set_page_config(page_title='Portal Team Rankings', layout='wide')

_FONT_WEIGHTS = {500: 'BarlowCondensed-Medium.ttf', 600: 'BarlowCondensed-SemiBold.ttf',
                 700: 'BarlowCondensed-Bold.ttf', 800: 'BarlowCondensed-ExtraBold.ttf'}


def data_url(path, mime='image/png'):
    p = Path(path)
    return f'data:{mime};base64,' + base64.b64encode(p.read_bytes()).decode() if p.exists() else ''


def norm(s):
    s = str(s).strip()
    return s[:-2] if s.endswith('.0') else s


def _esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def rgba(hex_, a):
    n = int(hex_.lstrip('#'), 16)
    return f'rgba({(n>>16)&255},{(n>>8)&255},{n&255},{a})'


@st.cache_data(show_spinner=False)
def font_face_css():
    out = []
    for w, fn in _FONT_WEIGHTS.items():
        p = FONT_DIR / fn
        if not p.exists():
            continue
        b64 = base64.b64encode(p.read_bytes()).decode()
        out.append("@font-face{font-family:'Saira Condensed';font-style:normal;font-weight:%d;"
                   "src:url(data:font/ttf;base64,%s) format('truetype');}" % (w, b64))
    return '\n'.join(out)


@st.cache_data(show_spinner=False)
def _csv(name):
    return pd.read_csv(DATA / name, dtype=str, encoding='latin-1', keep_default_na=False)


@st.cache_data(show_spinner=False)
def _logo_id_resolver():
    """Map every team id -> an id whose local logo file exists (softball ids often have
    no own file; fall back to the same-NCAA-org baseball sibling)."""
    teams = _csv('teams.csv')
    by_ncaa, nc_of = {}, {}
    for _, r in teams.iterrows():
        i = norm(r['id']); n = norm(r['team_id_ncaa'])
        nc_of[i] = n; by_ncaa.setdefault(n, []).append(i)
    have = lambda i: (LOGO_DIR / f'{i}.png').exists()
    out = {}
    for i in nc_of:
        out[i] = i if have(i) else next((s for s in by_ncaa.get(nc_of[i], []) if have(s)), i)
    return out


def _logo_id(tid):
    return _logo_id_resolver().get(norm(tid), norm(tid)) if tid else tid


@st.cache_data(show_spinner=False)
def logo_palette(team_id):
    """Dominant hex color from the team logo (for the row accent). Brand-red fallback."""
    fallback = '#b23a48'
    p = LOGO_DIR / f'{_logo_id(team_id)}.png'
    if not p.exists():
        return fallback
    try:
        img = Image.open(p).convert('RGBA'); img.thumbnail((96, 96))
        px = np.array(img); rgb = px[px[:, :, 3] > 128][:, :3]
        buckets = Counter()
        for r, g, b in rgb:
            br = (int(r) + int(g) + int(b)) / 3
            if br > 225 or br < 28:
                continue
            buckets[(int(r) // 24 * 24, int(g) // 24 * 24, int(b) // 24 * 24)] += 1
        if not buckets:
            return fallback
        c = buckets.most_common(1)[0][0]
        return '#%02x%02x%02x' % c
    except Exception:
        return fallback


@st.cache_data(show_spinner=False)
def load_rankings(sport, top_n):
    teams = _csv('teams.csv'); conf = _csv('conferences.csv')
    cabbr, cdiv = {}, {}
    for _, r in conf.iterrows():
        i = norm(r['id']); cabbr[i] = (r.get('abbreviation') or r.get('name') or ''); cdiv[i] = r.get('division', '')
    tinfo = {}
    for _, r in teams.iterrows():
        i = norm(r['id']); cid = norm(r['conference_id'])
        tinfo[i] = {'name': r['name'], 'sport': r['sport'], 'conf': cabbr.get(cid, ''), 'div': cdiv.get(cid, '')}
    prt = _csv('portal_rank_team.csv'); prt = prt[prt['year'].map(norm) == '2026'].copy()
    prt['rating'] = pd.to_numeric(prt['sixty_four_rating_portal_team'], errors='coerce').fillna(0)
    prt['sport'] = prt['team_id'].map(norm).map(lambda t: tinfo.get(t, {}).get('sport', ''))
    prt = prt[(prt['sport'] == sport) & (~prt['team_id'].map(norm).isin(BUCKETS)) & (prt['rating'] > 0)]
    prt = prt.sort_values('rating', ascending=False).head(top_n).reset_index(drop=True)
    out = []
    for i, r in prt.iterrows():
        tid = norm(r['team_id']); t = tinfo.get(tid, {})
        out.append({'rank': i + 1, 'tid': tid, 'name': t.get('name', ''), 'conf': t.get('conf', ''),
                    'rating': float(r['rating']),
                    'committed': int(float(r['players_committed'] or 0)),
                    'top_add': int(float(r['top_players_add'] or 0))})
    return out


# ---------------- CSS ----------------
DESIGN_CSS = r"""
:root{--card:#171c23;--card-edge:#262d36;--ink:#fff;--ink-soft:#8b97a6;
--maroon:#b23a48;--maroon-deep:#9a303e;--gold:#d9a94a;
--cond:'Saira Condensed','Barlow Condensed',sans-serif;
--mono:ui-monospace,'Consolas','SFMono-Regular',monospace;}
*{box-sizing:border-box;}
.board{position:relative;color:var(--ink);font-family:var(--mono);
background:radial-gradient(120% 80% at 50% -10%, #1b2129 0%, rgba(27,33,41,0) 55%),repeating-linear-gradient(125deg,#0d1116 0 22px,#0b0e12 22px 44px);overflow:hidden;}
.board::after{content:"";position:absolute;inset:0;pointer-events:none;box-shadow:inset 0 0 0 2px rgba(255,255,255,.04),inset 0 0 160px rgba(0,0,0,.65);}
.bhead{display:flex;align-items:center;justify-content:space-between;}
.logo64{display:block;width:auto;}
.bhead .tag{color:var(--ink-soft);text-transform:uppercase;letter-spacing:3px;text-align:right;}
.btitle{font-family:var(--cond);font-weight:800;text-transform:uppercase;color:#fff;line-height:.9;letter-spacing:-.5px;display:flex;align-items:center;gap:.32em;white-space:nowrap;}
.btitle .accent{color:var(--maroon);}
.btitle .rule{flex:1;height:3px;background:linear-gradient(90deg,var(--maroon),transparent);margin-left:.4em;}
.rows{display:flex;flex-direction:column;}
.cols{display:grid;grid-template-columns:1fr 1fr;}
.trow{position:relative;display:flex;align-items:center;background:var(--card);border:1px solid var(--card-edge);border-radius:9px;overflow:hidden;box-shadow:0 6px 16px rgba(0,0,0,.4);}
.trow .tint{position:absolute;inset:0;z-index:0;pointer-events:none;}
.trow.top3{border-color:rgba(217,169,74,.45);}
.trank{position:relative;z-index:1;font-family:var(--cond);font-weight:800;color:#fff;text-align:center;line-height:1;flex:0 0 auto;}
.trow.top3 .trank{color:var(--gold);}
.trank .hash{opacity:.5;font-weight:700;}
.tlogo{position:relative;z-index:1;object-fit:contain;flex:0 0 auto;filter:drop-shadow(0 2px 3px rgba(0,0,0,.5));}
.tinfo{position:relative;z-index:1;flex:1;min-width:0;display:flex;flex-direction:column;}
.tname{font-family:var(--cond);font-weight:700;color:#fff;text-transform:uppercase;line-height:.92;letter-spacing:-.3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.tmeta{color:var(--ink-soft);text-transform:uppercase;letter-spacing:.6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.tmeta .conf{color:var(--gold);}
.tratewrap{position:relative;z-index:1;flex:0 0 auto;display:flex;flex-direction:column;align-items:flex-end;}
.trate{font-family:var(--cond);font-weight:800;color:#fff;line-height:1;}
.tbar{background:rgba(255,255,255,.09);border-radius:99px;overflow:hidden;}
.tbarfill{height:100%;border-radius:99px;}
.bfoot{display:flex;align-items:center;justify-content:space-between;color:var(--ink-soft);text-transform:uppercase;letter-spacing:1px;}
.bfoot .pip{width:7px;height:7px;border-radius:50%;background:var(--maroon);display:inline-block;margin-right:.5em;vertical-align:middle;}

.board[data-format="ig"]{width:1080px;height:1350px;padding:44px 44px 30px;}
.board[data-format="ig"] .tag{font-size:12px;}
.board[data-format="ig"] .logo64{height:56px;}
.board[data-format="ig"] .btitle{font-size:62px;margin:16px 0 18px;}
.board[data-format="ig"] .rows{gap:8px;}
.board[data-format="ig"] .trow{height:41px;padding:0 16px 0 6px;gap:12px;}
.board[data-format="ig"] .trank{width:46px;font-size:26px;}
.board[data-format="ig"] .trank .hash{font-size:.6em;}
.board[data-format="ig"] .tlogo{width:30px;height:30px;}
.board[data-format="ig"] .tname{font-size:25px;}
.board[data-format="ig"] .tmeta{font-size:10px;margin-top:2px;}
.board[data-format="ig"] .trate{font-size:27px;}
.board[data-format="ig"] .tbar{width:96px;height:5px;margin-top:4px;}
.board[data-format="ig"] .bfoot{font-size:12px;margin-top:14px;}

.board[data-format="tw"]{width:1600px;height:900px;padding:36px 44px 26px;}
.board[data-format="tw"] .tag{font-size:11px;}
.board[data-format="tw"] .logo64{height:48px;}
.board[data-format="tw"] .btitle{font-size:52px;margin:12px 0 16px;}
.board[data-format="tw"] .cols{gap:10px 30px;}
.board[data-format="tw"] .trow{height:44px;padding:0 16px 0 6px;gap:12px;}
.board[data-format="tw"] .trank{width:44px;font-size:26px;}
.board[data-format="tw"] .trank .hash{font-size:.6em;}
.board[data-format="tw"] .tlogo{width:32px;height:32px;}
.board[data-format="tw"] .tname{font-size:26px;}
.board[data-format="tw"] .tmeta{font-size:10px;margin-top:2px;}
.board[data-format="tw"] .trate{font-size:28px;}
.board[data-format="tw"] .tbar{width:90px;height:5px;margin-top:4px;}
.board[data-format="tw"] .bfoot{font-size:11px;margin-top:12px;}
"""


def row_html(t, rmin, rmax, rowH):
    accent = logo_palette(t['tid'])
    pct = 12 + 88 * (t['rating'] - rmin) / (rmax - rmin) if rmax > rmin else 100
    logo = data_url(LOGO_DIR / f"{_logo_id(t['tid'])}.png")
    lg = round(rowH - 11)
    name_fs = round(min(40, max(20, rowH * 0.60)))
    rate_fs = round(min(42, max(21, rowH * 0.64)))
    rank_fs = round(min(38, max(20, rowH * 0.58)))
    barw = 110 if rowH >= 46 else 92
    logo_html = f'<img class="tlogo" style="width:{lg}px;height:{lg}px" src="{logo}"/>' if logo else f'<div class="tlogo" style="width:{lg}px;height:{lg}px"></div>'
    top3 = 'top3' if t['rank'] <= 3 else ''
    tint = f'linear-gradient(90deg,{rgba(accent,0.22)} 0%,{rgba(accent,0.05)} 34%,rgba(0,0,0,0) 60%)'
    return (
        f'<div class="trow {top3}" style="height:{rowH}px">'
        f'<div class="tint" style="background:{tint}"></div>'
        f'<div class="trank" style="font-size:{rank_fs}px"><span class="hash">#</span>{t["rank"]}</div>'
        f'{logo_html}'
        f'<div class="tinfo"><div class="tname" style="font-size:{name_fs}px">{_esc(t["name"])}</div>'
        f'<div class="tmeta"><span class="conf">{_esc(t["conf"])}</span> · {t["committed"]} commits · {t["top_add"]} top adds</div></div>'
        f'<div class="tratewrap"><div class="trate" style="font-size:{rate_fs}px">{t["rating"]:.2f}</div>'
        f'<div class="tbar" style="width:{barw}px"><div class="tbarfill" style="width:{pct:.0f}%;background:{accent}"></div></div></div>'
        f'</div>'
    )


# ---------------- UI ----------------
st.title('Portal Team Rankings')
st.caption('Top transfer-portal recruiting classes by 64A team rating. Pick sport / size / format, then Download PNG.')

c1, c2, c3 = st.columns(3)
sport = c1.radio('Sport', ['Baseball', 'Softball'], horizontal=True)
top_n = c2.radio('Top N', [10, 25, 50], index=1, horizontal=True)
fmt = 'ig' if c3.radio('Format', ['Instagram (1080×1350)', 'Twitter/X (1600×900)'], horizontal=True).startswith('Instagram') else 'tw'

teams = load_rankings(sport, top_n)
if not teams:
    st.warning(f'No {sport} portal team rankings found.'); st.stop()

rmin = min(t['rating'] for t in teams); rmax = max(t['rating'] for t in teams)
st.markdown('**' + ' · '.join(f'#{t["rank"]} {t["name"]} ({t["rating"]:.2f})' for t in teams[:5]) + ' …**')

if not st.button('🎨 Generate image', type='primary'):
    st.caption('Set options above, then click **Generate image** to render and download the PNG.')
    st.stop()

# ---------------- build board ----------------
import math
two_col = (fmt == 'tw') or (len(teams) > 25)     # 2 columns for Twitter, or any 50-deep board
n_col = 2 if two_col else 1
n_per = math.ceil(len(teams) / n_col)
usable = 1104 if fmt == 'ig' else 706            # column height left after header/title/footer
gap = 7
rowH = round(max(34, min(70, (usable - (n_per - 1) * gap) / n_per)))
if two_col:
    left = ''.join(row_html(t, rmin, rmax, rowH) for t in teams[:n_per])
    right = ''.join(row_html(t, rmin, rmax, rowH) for t in teams[n_per:])
    body = (f'<div class="cols" style="gap:{gap}px 28px">'
            f'<div class="rows" style="gap:{gap}px">{left}</div>'
            f'<div class="rows" style="gap:{gap}px">{right}</div></div>')
else:
    body = f'<div class="rows" style="gap:{gap}px">{"".join(row_html(t, rmin, rmax, rowH) for t in teams)}</div>'
title = f'TOP {len(teams)}' if len(teams) != 25 else 'TOP 25'
brand_html = f'<img class="logo64" src="{data_url(BRAND / "logo-64-analytics.png")}" alt="64A"/>'
board = (
    f'<div class="board" data-format="{fmt}" id="capture">'
    f'<div class="bhead">{brand_html}<div class="tag">Transfer Portal Classes · {sport} · &rsquo;26</div></div>'
    f'<div class="btitle"><span class="accent">{title}</span><span>PORTAL TEAM RANKINGS</span><span class="rule"></span></div>'
    f'{body}'
    f'<div class="bfoot"><span><span class="pip"></span>64A Proprietary Team Rating</span>'
    f'<span>64analytics.com</span></div>'
    f'</div>'
)

W, H = (1080, 1350) if fmt == 'ig' else (1600, 900)
scale = round(680 / W, 4)
css = font_face_css() + '\n' + DESIGN_CSS
fname = f'portal_team_rankings_{sport.lower()}_top{len(teams)}_{fmt}.png'

html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"/>
<style>{css}
body{{margin:0;background:#0b0c0f;}}
#stagewrap{{width:{int(W*scale)}px;height:{int(H*scale)}px;overflow:hidden;}}
#stageinner{{transform:scale({scale});transform-origin:top left;}}
.btnrow{{margin:14px 0 4px;}}
.btnrow button{{font-family:system-ui,sans-serif;font-size:14px;font-weight:700;color:#fff;background:#b23a48;border:none;border-radius:7px;padding:9px 16px;cursor:pointer;}}
.btnrow button:disabled{{opacity:.6;}}
</style></head><body>
<div id="stagewrap"><div id="stageinner">{board}</div></div>
<div class="btnrow"><button onclick="window.dlPNG(this)">Download PNG</button></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<script>
window.dlPNG = async function(btn){{
  var el = document.getElementById('capture'); if(!el) return;
  var inner = document.getElementById('stageinner'), wrap = document.getElementById('stagewrap');
  var pT = inner ? inner.style.transform : '', pW = wrap ? wrap.style.cssText : '';
  var t = btn.textContent; btn.disabled = true; btn.textContent = 'Rendering…';
  if(inner) inner.style.transform = 'none';
  if(wrap){{ wrap.style.width='{W}px'; wrap.style.height='{H}px'; wrap.style.overflow='visible'; }}
  try{{
    if(document.fonts){{ try{{ await document.fonts.load("800 62px 'Saira Condensed'"); await document.fonts.load("700 26px 'Saira Condensed'"); }}catch(e){{}} if(document.fonts.ready) await document.fonts.ready; }}
    var canvas = await html2canvas(el, {{scale:2, useCORS:true, backgroundColor:'#0b0e12'}});
    var a = document.createElement('a'); a.download='{fname}';
    a.href = canvas.toDataURL('image/png'); document.body.appendChild(a); a.click(); document.body.removeChild(a);
  }} finally {{ if(inner) inner.style.transform=pT; if(wrap) wrap.style.cssText=pW; btn.disabled=false; btn.textContent = t || 'Download PNG'; }}
}};
</script></body></html>"""

components.html(html, height=int(H * scale) + 90, scrolling=False)
