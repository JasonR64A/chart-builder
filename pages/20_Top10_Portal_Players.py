"""Top 10 Portal Players — social/print graphic (Instagram 1080x1350, Twitter 1600x900).

Ported from the Claude Design "Top 10 Portal Players" board. Live top 10 by
sixty_four_rating_portal_player (1 = best) per sport. Each school "ballcap" is tinted
with a logo-derived SECONDARY color and wears the team logo on the front (so the logo
contrasts the hat). Per-player headshots can be uploaded and are remembered by player_id.
Mirrors the asset->data-url + html2canvas pattern used by 16_Portal_Entrant_Graphic.
"""
import base64
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

APP_DIR = Path(__file__).resolve().parent.parent
DATA = APP_DIR / 'data'
LOGO_DIR = APP_DIR / 'team_logos_512'
BRAND = APP_DIR / 'assets' / 'portal-entrant'
CAP_PNG = APP_DIR / 'assets' / 'top10' / 'cap-blank.png'
HEADSHOTS = DATA / 'top10_headshots'
HEADSHOTS.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title='Top 10 Portal Players', layout='wide')


# ---------------- helpers ----------------
def data_url(path, mime='image/png'):
    p = Path(path)
    if not p.exists():
        return ''
    return f'data:{mime};base64,' + base64.b64encode(p.read_bytes()).decode()


def _b64_bytes(b, mime='image/png'):
    return f'data:{mime};base64,' + base64.b64encode(b).decode()


@st.cache_data(show_spinner=False)
def _csv(name):
    return pd.read_csv(DATA / name, dtype=str, encoding='latin-1', keep_default_na=False)


def norm(s):
    s = str(s).strip()
    return s[:-2] if s.endswith('.0') else s


def _esc(s):
    return (str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def darken(hex_, amt):
    h = hex_.lstrip('#')
    n = int(h, 16)
    r, g, b = (n >> 16) & 255, (n >> 8) & 255, n & 255
    r, g, b = int(r * (1 - amt)), int(g * (1 - amt)), int(b * (1 - amt))
    return f'rgb({r},{g},{b})'


def rgba(hex_, a):
    h = hex_.lstrip('#')
    n = int(h, 16)
    return f'rgba({(n >> 16) & 255},{(n >> 8) & 255},{n & 255},{a})'


def cap_text_color(hex_):
    h = hex_.lstrip('#')
    n = int(h, 16)
    lum = 0.299 * ((n >> 16) & 255) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255)
    return '#16202b' if lum > 150 else '#ffffff'


@st.cache_data(show_spinner=False)
def logo_palette(team_id):
    """(primary, secondary) hex from the team logo. primary = dominant mid-tone color,
    secondary = next distinct color cluster (used to tint the cap so it contrasts the
    logo). Falls back to brand red / dark slate when the logo is missing or monochrome."""
    fallback = ('#b23a48', '#2a313b')
    p = LOGO_DIR / f'{team_id}.png'
    if not p.exists():
        return fallback
    try:
        img = Image.open(p).convert('RGBA')
        img.thumbnail((96, 96))
        px = np.array(img)
        rgb = px[px[:, :, 3] > 128][:, :3]
        buckets = Counter()
        for r, g, b in rgb:
            br = (int(r) + int(g) + int(b)) / 3
            if br > 225 or br < 28:
                continue
            buckets[(int(r) // 24 * 24, int(g) // 24 * 24, int(b) // 24 * 24)] += 1
        if not buckets:
            return fallback
        ranked = [c for c, _ in buckets.most_common()]
        prim = ranked[0]
        sec = None
        for c in ranked[1:]:
            if abs(c[0] - prim[0]) + abs(c[1] - prim[1]) + abs(c[2] - prim[2]) > 90:
                sec = c
                break
        hx = lambda c: '#%02x%02x%02x' % c
        return (hx(prim), hx(sec) if sec else '#2a313b')
    except Exception:
        return fallback


def abbr_of(name):
    name = (name or '??').strip()
    drop = {'university', 'univ', 'of', 'the', 'college', 'at', 'a&m'}
    words = [w for w in name.replace('.', '').split() if w.lower() not in drop]
    caps = ''.join(w[0] for w in words if w[:1].isupper())[:4]
    return (caps or name[:4]).upper()


CLASS_SHORT = {'Freshman': 'Fr', 'Sophomore': 'So', 'Junior': 'Jr', 'Senior': 'Sr'}


@st.cache_data(show_spinner=False)
def build_top10(sport):
    prp = _csv('portal_rank_player.csv')
    prp = prp[prp['year'].map(norm) == '2026'].copy()
    prp['rk'] = pd.to_numeric(prp['sixty_four_rating_portal_player'], errors='coerce')

    teams = _csv('teams.csv')
    tname, tsport, tconf = {}, {}, {}
    for _, r in teams.iterrows():
        i = norm(r['id'])
        tname[i] = r['name']; tsport[i] = r['sport']; tconf[i] = norm(r['conference_id'])
    conf = _csv('conferences.csv')
    cabbr = {norm(r['id']): (r.get('abbreviation') or r.get('name') or '') for _, r in conf.iterrows()}
    players = _csv('players.csv')
    pname, ppos = {}, {}
    for _, r in players.iterrows():
        i = norm(r['id']); pname[i] = r['player_name']; ppos[i] = r['position']

    prp['sport'] = prp['team_id'].map(norm).map(tsport)
    sub = prp[(prp['sport'] == sport) & prp['rk'].notna()].nsmallest(10, 'rk')

    out = []
    for _, r in sub.iterrows():
        pid = norm(r['player_id']); ftid = norm(r['team_id']); ntid = norm(r['new_team_id'])
        committed = ntid not in ('', '0', 'nan')
        out.append({
            'pid': pid,
            'name': pname.get(pid) or r['name'] or f'#{pid}',
            'pos': (ppos.get(pid) or '').upper(),
            'conf': cabbr.get(tconf.get(ftid, ''), ''),
            'rank': int(r['rk']),
            'from_tid': ftid, 'from_name': tname.get(ftid, ''),
            'committed': committed,
            'to_tid': ntid if committed else '', 'to_name': tname.get(ntid, '') if committed else '',
        })
    return out


def split_name(full):
    parts = full.strip().split()
    if len(parts) <= 1:
        return ('', full.strip())
    suffixes = {'II', 'III', 'IV', 'Jr.', 'Sr.', 'Jr', 'Sr'}
    last = parts.pop()
    if last in suffixes and len(parts) > 1:
        last = parts.pop() + ' ' + last
    return (' '.join(parts), last)


# ---------------- design CSS (ported from top10.css) ----------------
DESIGN_CSS = r"""
:root{--bg:#0b0e12;--card:#171c23;--card-edge:#262d36;--ink:#fff;--ink-soft:#8b97a6;
--maroon:#b23a48;--maroon-deep:#9a303e;--orange:#d98443;
--cond:'Saira Condensed',sans-serif;--sans:'Archivo',system-ui,sans-serif;--mono:'Spline Sans Mono',ui-monospace,monospace;}
*{box-sizing:border-box;}
.board{position:relative;color:var(--ink);font-family:var(--sans);-webkit-font-smoothing:antialiased;
background:radial-gradient(120% 80% at 50% -10%, #1b2129 0%, rgba(27,33,41,0) 55%),repeating-linear-gradient(125deg,#0d1116 0 22px,#0b0e12 22px 44px);overflow:hidden;}
.board::after{content:"";position:absolute;inset:0;pointer-events:none;box-shadow:inset 0 0 0 2px rgba(255,255,255,.04),inset 0 0 160px rgba(0,0,0,.65);}
.bhead{display:flex;align-items:center;justify-content:space-between;}
.bhead .brand{display:flex;align-items:center;line-height:1;gap:14px;}
.logo64{display:block;width:auto;}
.brand-div{width:1.5px;height:38px;background:rgba(255,255,255,.2);}
.partner{display:block;width:auto;}
.bhead .tag{font-family:var(--mono);color:var(--ink-soft);text-transform:uppercase;letter-spacing:3px;text-align:right;}
.btitle{font-family:var(--cond);font-weight:800;text-transform:uppercase;color:#fff;line-height:.9;letter-spacing:-.5px;display:flex;align-items:baseline;gap:.32em;white-space:nowrap;}
.btitle span{white-space:nowrap;}
.btitle .t10{color:var(--maroon);}
.btitle .rule{flex:1;height:3px;background:linear-gradient(90deg,var(--maroon),transparent);align-self:center;margin-left:.4em;}
.grid{display:grid;grid-template-columns:1fr 1fr;}
.pcard{position:relative;overflow:hidden;border-radius:10px;display:flex;align-items:stretch;background:var(--card);border:1px solid var(--card-edge);box-shadow:0 10px 26px rgba(0,0,0,.45);isolation:isolate;}
.pcard .tint{position:absolute;inset:0;z-index:0;pointer-events:none;}
.pcard .watermark{position:absolute;z-index:0;right:-.08em;bottom:-.30em;font-family:var(--cond);font-weight:800;color:rgba(255,255,255,.06);line-height:.7;pointer-events:none;letter-spacing:-3px;}
.pphoto{position:relative;z-index:1;flex:0 0 auto;height:100%;aspect-ratio:1/1;display:flex;align-items:flex-end;justify-content:center;overflow:hidden;}
.pphoto .ring{position:absolute;inset:0;z-index:0;}
.pphoto .phead{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;z-index:1;}
.pphoto .sil{position:absolute;z-index:0;bottom:0;left:50%;transform:translateX(-50%);color:rgba(255,255,255,.18);}
.pcontent{position:relative;z-index:2;flex:1;min-width:0;display:flex;flex-direction:column;justify-content:center;}
.pname{font-family:var(--cond);text-transform:uppercase;line-height:.84;}
.pname .fn{display:block;font-weight:500;color:rgba(255,255,255,.72);}
.pname .ln{display:block;font-weight:800;color:#fff;letter-spacing:-.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.pmeta{display:flex;align-items:center;gap:.5em;}
.ppos{font-family:var(--mono);font-weight:600;color:var(--orange);}
.psport{font-family:var(--mono);color:var(--ink-soft);text-transform:uppercase;letter-spacing:1px;}
.transfer{display:flex;align-items:center;}
.capwrap{display:flex;flex-direction:column;align-items:center;}
.caphat{position:relative;display:block;isolation:isolate;}
.cap-base{position:absolute;inset:0;-webkit-mask:url(__CAP__) center/contain no-repeat;mask:url(__CAP__) center/contain no-repeat;}
.cap-shade{position:absolute;inset:0;background:url(__CAP__) center/contain no-repeat;mix-blend-mode:multiply;}
.cap-abbr{position:absolute;left:0;right:0;top:36%;text-align:center;font-family:var(--cond);font-weight:800;line-height:1;letter-spacing:.3px;z-index:2;}
.caphat .cap-logo{position:absolute;left:24%;top:27%;width:52%;height:32%;object-fit:contain;z-index:3;filter:drop-shadow(0 1px 1px rgba(0,0,0,.35));}
.cap-q{position:absolute;left:0;right:0;top:33%;text-align:center;font-family:var(--cond);font-weight:800;color:#8b97a6;line-height:1;z-index:2;}
.caplbl{font-family:var(--mono);text-transform:uppercase;color:var(--ink-soft);letter-spacing:1.5px;line-height:1;}
.t-arrow{color:var(--maroon);display:flex;align-items:center;}
.cap-open{opacity:.55;}
.prank{position:absolute;z-index:3;top:0;right:0;display:flex;align-items:flex-start;justify-content:flex-end;}
.prank .num{font-family:var(--cond);font-weight:800;color:#fff;line-height:1;background:linear-gradient(160deg,var(--maroon),var(--maroon-deep));display:flex;align-items:center;justify-content:center;border-bottom-left-radius:13px;box-shadow:-4px 4px 14px rgba(0,0,0,.4);}
.prank .num .hash{opacity:.62;font-weight:700;font-size:.66em;margin-right:.05em;transform:translateY(-.04em);}
.prank.top3 .num{background:linear-gradient(160deg,#d9a94a,#b9863a);color:#1a1205;}
.prank.top3 .num .hash{opacity:.5;}
.bfoot{display:flex;align-items:center;justify-content:space-between;font-family:var(--mono);color:var(--ink-soft);text-transform:uppercase;}
.bfoot .dotrow{display:flex;gap:.5em;align-items:center;}
.bfoot .pip{width:7px;height:7px;border-radius:50%;background:var(--maroon);}
.board[data-format="ig"]{width:1080px;height:1350px;padding:52px 48px 44px;}
.board[data-format="ig"] .bhead{margin-bottom:18px;}
.board[data-format="ig"] .tag{font-size:12px;}
.board[data-format="ig"] .btitle{font-size:76px;margin-bottom:24px;}
.board[data-format="ig"] .grid{gap:16px;grid-auto-rows:1fr;height:1004px;}
.board[data-format="ig"] .pname{font-size:35px;}
.board[data-format="ig"] .pname .fn{font-size:25px;}
.board[data-format="ig"] .pmeta{font-size:16px;margin-top:8px;}
.board[data-format="ig"] .logo64{height:60px;}
.board[data-format="ig"] .partner{height:58px;}
.board[data-format="ig"] .pcontent{padding:14px 90px 14px 18px;gap:12px;}
.board[data-format="ig"] .transfer{gap:10px;}
.board[data-format="ig"] .caplbl{font-size:10px;margin-bottom:5px;}
.board[data-format="ig"] .prank .num{height:48px;min-width:48px;padding:0 13px;font-size:29px;}
.board[data-format="ig"] .watermark{font-size:230px;}
.board[data-format="ig"] .bfoot{font-size:12px;margin-top:18px;}
.board[data-format="tw"]{width:1600px;height:900px;padding:40px 46px 34px;}
.board[data-format="tw"] .bhead{margin-bottom:14px;}
.board[data-format="tw"] .tag{font-size:11px;}
.board[data-format="tw"] .btitle{font-size:60px;margin-bottom:18px;}
.board[data-format="tw"] .grid{gap:14px 22px;grid-auto-rows:1fr;height:642px;}
.board[data-format="tw"] .pname{font-size:35px;}
.board[data-format="tw"] .pname .fn{font-size:21px;}
.board[data-format="tw"] .pmeta{font-size:15px;margin-top:6px;}
.board[data-format="tw"] .logo64{height:48px;}
.board[data-format="tw"] .partner{height:48px;}
.board[data-format="tw"] .pcontent{padding:10px 96px 10px 18px;gap:16px;flex-direction:row;align-items:center;}
.board[data-format="tw"] .pcontent>.nameblock{flex:1;min-width:0;}
.board[data-format="tw"] .transfer{flex:0 0 auto;gap:9px;}
.board[data-format="tw"] .caplbl{font-size:9px;margin-bottom:4px;}
.board[data-format="tw"] .prank .num{height:44px;min-width:44px;padding:0 12px;font-size:27px;}
.board[data-format="tw"] .watermark{font-size:180px;}
.board[data-format="tw"] .bfoot{font-size:11px;margin-top:14px;}
"""

ARROW = ('<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
         'stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round">'
         '<path d="M4 12h15"/><path d="m13 6 7 6-7 6"/></svg>')
SIL = ('<svg class="sil" width="78%" height="78%" viewBox="0 0 100 100" fill="currentColor">'
       '<circle cx="50" cy="36" r="20"/><path d="M12 100 C12 70 30 60 50 60 C70 60 88 70 88 100 Z"/></svg>')


def cap_html(tid, capw, open_=False):
    h = round(capw * 1.25)
    if open_ or not tid:
        return (f'<div class="caphat cap-open" style="width:{capw}px;height:{h}px">'
                f'<div class="cap-base" style="background:#3a434e"></div><div class="cap-shade"></div>'
                f'<span class="cap-q" style="font-size:{round(capw*0.34)}px">?</span></div>')
    _, secondary = logo_palette(tid)
    logo = data_url(LOGO_DIR / f'{tid}.png')
    inner = (f'<img class="cap-logo" src="{logo}"/>' if logo
             else f'<span class="cap-abbr" style="color:{cap_text_color(secondary)};font-size:{round(capw*0.23)}px">{abbr_of("")}</span>')
    return (f'<div class="caphat" style="width:{capw}px;height:{h}px">'
            f'<div class="cap-base" style="background:{secondary}"></div><div class="cap-shade"></div>{inner}</div>')


def card_html(p, pos, capw, headshot_url):
    fn, ln = split_name(p['name'])
    fprim, _ = logo_palette(p['from_tid']) if p['from_tid'] else ('#b23a48', '')
    accent = (logo_palette(p['to_tid'])[0] if p['committed'] and p['to_tid'] else fprim)
    photo_bg = f'linear-gradient(120deg,{rgba(fprim,0.95)},{darken(fprim,0.45)})'
    tint_bg = f'linear-gradient(100deg,{rgba(accent,0.0)} 30%,{rgba(accent,0.16)} 75%,{rgba(accent,0.30)} 100%)'
    wm = (abbr_of(p['to_name']) if p['committed'] else abbr_of(p['from_name']))[:1]
    head = f'<img class="phead" src="{headshot_url}"/>' if headshot_url else ''
    fn_html = f'<span class="fn">{_esc(fn)}</span>' if fn else ''
    to_lbl = 'To' if p['committed'] else 'To'
    return (
        f'<div class="pcard {"is-top" if pos<=3 else ""}">'
        f'<div class="tint" style="background:{tint_bg}"></div>'
        f'<div class="watermark">{_esc(wm)}</div>'
        f'<div class="pphoto"><div class="ring" style="background:{photo_bg}"></div>{SIL}{head}</div>'
        f'<div class="pcontent"><div class="nameblock">'
        f'<div class="pname">{fn_html}<span class="ln">{_esc(ln)}</span></div>'
        f'<div class="pmeta"><span class="ppos">{_esc(p["pos"])}</span><span class="psport">{_esc(p["conf"])}</span></div>'
        f'</div>'
        f'<div class="transfer">'
        f'<div class="capwrap"><span class="caplbl">From</span>{cap_html(p["from_tid"], capw)}</div>'
        f'<span class="t-arrow">{ARROW.format(s=round(capw*0.42))}</span>'
        f'<div class="capwrap"><span class="caplbl">{to_lbl}</span>{cap_html(p["to_tid"], capw, open_=not p["committed"])}</div>'
        f'</div></div>'
        f'<div class="prank {"top3" if pos<=3 else ""}"><div class="num"><span class="hash">#</span>{p["rank"]:,}</div></div>'
        f'</div>'
    )


# ---------------- UI ----------------
st.title('Top 10 Portal Players')
st.caption('Live top 10 by 64A portal rank (1 = best). Tweak format/sport, drop headshots, then Download PNG.')

c1, c2 = st.columns(2)
sport = c1.radio('Sport', ['Baseball', 'Softball'], horizontal=True)
fmt_label = c2.radio('Format', ['Instagram (1080×1350)', 'Twitter/X (1600×900)'], horizontal=True)
fmt = 'ig' if fmt_label.startswith('Instagram') else 'tw'
capw = 60 if fmt == 'ig' else 54

players = build_top10(sport)
if not players:
    st.warning(f'No {sport} portal players found in portal_rank_player.csv.')
    st.stop()

# headshots: upload + remember (by player_id)
with st.expander('Headshots — upload (remembered per player by id)', expanded=False):
    for p in players:
        cached = HEADSHOTS / f'{p["pid"]}.png'
        cols = st.columns([3, 2])
        cols[0].markdown(f'**#{p["rank"]:,} · {p["name"]}** ({p["from_name"]})'
                         + ('  ✅ on file' if cached.exists() else ''))
        up = cols[1].file_uploader('headshot', type=['png', 'jpg', 'jpeg', 'webp'],
                                   key=f'hs_{p["pid"]}', label_visibility='collapsed')
        if up is not None:
            try:
                img = Image.open(up).convert('RGBA')
                img.thumbnail((600, 600))
                cached.parent.mkdir(parents=True, exist_ok=True)
                img.save(cached, 'PNG')
                st.toast(f'Saved headshot for {p["name"]}')
            except Exception as e:
                st.warning(f'Could not read that image: {e}')


def headshot_for(pid):
    c = HEADSHOTS / f'{pid}.png'
    return data_url(c) if c.exists() else ''


# ---------------- build board ----------------
season = 'Baseball' if sport == 'Baseball' else 'Softball'
cards = ''.join(card_html(p, i + 1, capw, headshot_for(p['pid'])) for i, p in enumerate(players))
brand_html = (
    f'<img class="logo64" src="{data_url(BRAND / "logo-64-analytics.png")}" alt="64A"/>'
    f'<span class="brand-div"></span>'
    f'<img class="partner" src="{data_url(BRAND / "logo-college-baseball-show.png")}" alt="CBS"/>'
    f'<img class="partner" src="{data_url(BRAND / "logo-jb.png")}" alt="JB"/>'
)
board = (
    f'<div class="board" data-format="{fmt}" id="capture">'
    f'<div class="bhead"><div class="brand">{brand_html}</div>'
    f'<div class="tag">Transfer Portal · {season} · &rsquo;26</div></div>'
    f'<div class="btitle"><span class="t10">TOP 10</span><span>PORTAL PLAYERS</span><span class="rule"></span></div>'
    f'<div class="grid">{cards}</div>'
    f'<div class="bfoot"><span>64A Proprietary Rankings</span>'
    f'<span class="dotrow"><span class="pip"></span> 64analytics.com</span></div>'
    f'</div>'
)

css = DESIGN_CSS.replace('__CAP__', data_url(CAP_PNG))
W, H = (1080, 1350) if fmt == 'ig' else (1600, 900)
scale = round(660 / W, 4)
fname = f'top10_portal_{sport.lower()}_{fmt}.png'

html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"/>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700;800&family=Saira+Condensed:wght@500;600;700;800&family=Spline+Sans+Mono:wght@400;500;600&display=swap" rel="stylesheet"/>
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
  var t = btn.textContent; btn.disabled = true; btn.textContent = 'Rendering…';
  try{{
    if(document.fonts && document.fonts.ready) await document.fonts.ready;
    var canvas = await html2canvas(el, {{scale:2, useCORS:true, backgroundColor:'#0b0e12', width:{W}, height:{H}, windowWidth:{W}, windowHeight:{H}}});
    var a = document.createElement('a'); a.download='{fname}';
    a.href = canvas.toDataURL('image/png'); document.body.appendChild(a); a.click(); document.body.removeChild(a);
  }} finally {{ btn.disabled=false; btn.textContent = t || 'Download PNG'; }}
}};
</script></body></html>"""

components.html(html, height=int(H * scale) + 80, scrolling=False)

st.markdown('**Top 10 · ' + sport + '**: ' + ' · '.join(f'#{p["rank"]:,} {p["name"]}' for p in players[:10]))
