"""Top 10 Portal Players — social/print graphic (Instagram 1080x1350, Twitter 1600x900).

Ported from the Claude Design "Top 10 Portal Players" board. Live top 10 by
sixty_four_rating_portal_player (1 = best) within the filtered pool, per sport.
Filters: conference, commitment status, division, position. Each school "ballcap" is
tinted with a logo-derived SECONDARY color (overridable per-school via color picker) and
wears the team logo. Per-player headshots upload + remember. Fonts are embedded as base64
@font-face so html2canvas exports cleanly (Google-hosted fonts don't render in capture).
"""
import base64
import json
from io import BytesIO
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parent.parent
DATA = APP_DIR / 'data'
LOGO_DIR = APP_DIR / 'team_logos_512'
CAP_LOGO_DIR = APP_DIR / 'cap_logos'   # cleaner cap-specific marks (override); falls back to LOGO_DIR
BRAND = APP_DIR / 'assets' / 'portal-entrant'
FONT_DIR = APP_DIR / 'assets' / 'fonts'
CAP_PNG = APP_DIR / 'assets' / 'top10' / 'cap-blank.png'
HEADSHOTS = DATA / 'top10_headshots'
HAT_JSON = DATA / 'top10_hat_overrides.json'
HEADSHOTS.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title='Top 10 Portal Players', layout='wide')

# condensed display font embedded as 'Saira Condensed' (substitute = local BarlowCondensed)
_FONT_WEIGHTS = {500: 'BarlowCondensed-Medium.ttf', 600: 'BarlowCondensed-SemiBold.ttf',
                 700: 'BarlowCondensed-Bold.ttf', 800: 'BarlowCondensed-ExtraBold.ttf'}


# ---------------- helpers ----------------
def data_url(path, mime='image/png'):
    p = Path(path)
    if not p.exists():
        return ''
    return f'data:{mime};base64,' + base64.b64encode(p.read_bytes()).decode()


HEAD_SIZE = 600  # photo cells are 1:1; store/serve square so html2canvas can't stretch


def square_headshot(im):
    """Cover-crop a headshot to a square that fills the 1:1 photo cell with NO distortion.
    Necessary because html2canvas IGNORES CSS object-fit:cover on export and stretches the
    <img> to the cell box — a tall photo would come out squished. Baking the crop in fixes
    it for any uploaded aspect. Bias slightly toward the top so ballcaps aren't clipped."""
    return ImageOps.fit(im.convert('RGBA'), (HEAD_SIZE, HEAD_SIZE),
                        method=Image.LANCZOS, centering=(0.5, 0.34))


@st.cache_data(show_spinner=False)
def tinted_cap(color):
    """Colorize the blank-cap PNG to `color` in Python (color x shading), keeping the cap
    silhouette + alpha. Done server-side because html2canvas can't render CSS masks, which
    otherwise left a square color block behind each hat."""
    base = Image.open(CAP_PNG).convert('RGBA')
    arr = np.array(base).astype('float32')
    n = int(color.lstrip('#'), 16)
    cr, cg, cb = (n >> 16) & 255, (n >> 8) & 255, n & 255
    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]) / 255.0
    f = np.clip(gray * 1.12, 0, 1)  # keep colors vivid, retain photo shading
    arr[:, :, 0] = cr * f; arr[:, :, 1] = cg * f; arr[:, :, 2] = cb * f
    buf = BytesIO()
    Image.fromarray(arr.astype('uint8'), 'RGBA').save(buf, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


@st.cache_data(show_spinner=False)
def face_shadow(pid):
    """Dark, cool-tinted 'shadow' version of a player's headshot for the corner watermark.
    Baked in Python (alpha preserved) so html2canvas exports it — CSS filters don't render."""
    c = HEADSHOTS / f'{pid}.png'
    if not c.exists():
        return ''
    arr = np.array(Image.open(c).convert('RGBA')).astype('float32')
    g = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]) * 0.30
    arr[:, :, 0] = g * 0.85; arr[:, :, 1] = g * 0.95; arr[:, :, 2] = g * 1.15
    buf = BytesIO()
    Image.fromarray(arr.clip(0, 255).astype('uint8'), 'RGBA').save(buf, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


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


def norm(s):
    s = str(s).strip()
    return s[:-2] if s.endswith('.0') else s


def _esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def darken(hex_, amt):
    n = int(hex_.lstrip('#'), 16)
    r, g, b = (n >> 16) & 255, (n >> 8) & 255, n & 255
    return f'rgb({int(r*(1-amt))},{int(g*(1-amt))},{int(b*(1-amt))})'


def rgba(hex_, a):
    n = int(hex_.lstrip('#'), 16)
    return f'rgba({(n>>16)&255},{(n>>8)&255},{n&255},{a})'


def cap_text_color(hex_):
    n = int(hex_.lstrip('#'), 16)
    lum = 0.299 * ((n >> 16) & 255) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255)
    return '#16202b' if lum > 150 else '#ffffff'


@st.cache_data(show_spinner=False)
def logo_palette(team_id):
    """(primary, secondary) hex from the team logo. secondary = 2nd distinct color cluster,
    used to tint the cap so the logo contrasts the hat. Brand-red / dark-slate fallback."""
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
        sec = next((c for c in ranked[1:]
                    if abs(c[0]-prim[0]) + abs(c[1]-prim[1]) + abs(c[2]-prim[2]) > 90), None)
        hx = lambda c: '#%02x%02x%02x' % c
        return (hx(prim), hx(sec) if sec else '#2a313b')
    except Exception:
        return fallback


def load_overrides():
    try:
        return json.loads(HAT_JSON.read_text()) if HAT_JSON.exists() else {}
    except Exception:
        return {}


def save_overrides(d):
    try:
        HAT_JSON.write_text(json.dumps(d))
    except Exception:
        pass


@st.cache_data(show_spinner=False)
def load_pool(sport):
    prp = _csv('portal_rank_player.csv')
    prp = prp[prp['year'].map(norm) == '2026'].copy()
    prp['rk'] = pd.to_numeric(prp['sixty_four_rating_portal_player'], errors='coerce')

    teams = _csv('teams.csv')
    tname, tsport, tconfid = {}, {}, {}
    for _, r in teams.iterrows():
        i = norm(r['id']); tname[i] = r['name']; tsport[i] = r['sport']; tconfid[i] = norm(r['conference_id'])
    conf = _csv('conferences.csv')
    cabbr, cdiv = {}, {}
    for _, r in conf.iterrows():
        i = norm(r['id']); cabbr[i] = (r.get('abbreviation') or r.get('name') or ''); cdiv[i] = r.get('division', '')
    players = _csv('players.csv')
    pname, ppos = {}, {}
    for _, r in players.iterrows():
        i = norm(r['id']); pname[i] = r['player_name']; ppos[i] = r['position']

    # per-player stats for the stat-threshold filter (2026, best-sample row)
    def _statmap(fname, ipcol, cols):
        df = _csv(fname); df = df[df['year'].map(norm) == '2026'].copy()
        df['pid'] = df['player_id'].map(norm)
        df['_s'] = pd.to_numeric(df[ipcol], errors='coerce')
        df = df.sort_values('_s').groupby('pid').tail(1).set_index('pid')
        m = {}
        for pid_, row in df.iterrows():
            d = {}
            for k, c in cols.items():
                v = pd.to_numeric(row.get(c), errors='coerce')
                if pd.notna(v):
                    d[k] = float(v)
            m[pid_] = d
        return m
    hmap = _statmap('hitting.csv', 'plate_appearances',
                    {'OPS': 'on_base_plus_slugging', 'OBP': 'on_base_percentage', 'ISO': 'isolated_power',
                     'AVG': 'batting_average', 'HR': 'home_runs', 'wRC': 'weighted_runs_created',
                     'wRAA': 'weighted_runs_above_average', 'PA': 'plate_appearances'})
    pmap_s = _statmap('pitching.csv', 'innings_pitched',
                      {'ERA': 'earned_run_average', 'FIP': 'fielding_independent_pitching',
                       'WHIP': 'walks_plus_hits_per_inning_pitched', 'K9': 'strikeouts_per_9_innings',
                       'IP': 'innings_pitched'})

    # portal entry date (player_rank initiated_date, 2026)
    pr2 = _csv('player_rank.csv'); pr2 = pr2[pr2['year'].map(norm) == '2026']
    entered = {}
    for _, r in pr2.iterrows():
        d = str(r.get('initiated_date', '') or '')[:10]
        if d:
            entered.setdefault(norm(r['player_id']), d)

    prp['sport'] = prp['team_id'].map(norm).map(tsport)
    sub = prp[(prp['sport'] == sport) & prp['rk'].notna()].sort_values('rk')

    out = []
    for _, r in sub.iterrows():
        pid = norm(r['player_id']); ftid = norm(r['team_id']); ntid = norm(r['new_team_id'])
        committed = ntid not in ('', '0', 'nan')
        fcid = tconfid.get(ftid, '')
        tcid = tconfid.get(ntid, '') if committed else ''
        out.append({
            'pid': pid, 'name': pname.get(pid) or r['name'] or f'#{pid}',
            'pos': (ppos.get(pid) or '').upper(), 'conf': cabbr.get(fcid, ''),
            'to_conf': cabbr.get(tcid, '') if committed else '',
            'division': cdiv.get(fcid, ''), 'rank': int(r['rk']),
            'from_tid': ftid, 'from_name': tname.get(ftid, ''),
            'committed': committed, 'to_tid': ntid if committed else '',
            'to_name': tname.get(ntid, '') if committed else '',
            'entered': entered.get(pid, ''),
            'stats': {**hmap.get(pid, {}), **pmap_s.get(pid, {})},
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


# ---------------- design CSS (ported) ----------------
DESIGN_CSS = r"""
:root{--card:#171c23;--card-edge:#262d36;--ink:#fff;--ink-soft:#8b97a6;
--maroon:#b23a48;--maroon-deep:#9a303e;--orange:#d98443;
--cond:'Saira Condensed','Barlow Condensed',sans-serif;
--sans:system-ui,'Segoe UI',Arial,sans-serif;
--mono:ui-monospace,'Consolas','SFMono-Regular',monospace;}
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
.pcard .watermark-photo{position:absolute;z-index:0;bottom:-2%;right:-2%;height:92%;width:auto;opacity:.18;pointer-events:none;}
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
.caphat .cap-base-img{position:absolute;inset:0;width:100%;height:100%;z-index:1;}
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


def cap_html(tid, color, capw, open_=False):
    h = round(capw * 1.25)
    if open_ or not tid:
        return (f'<div class="caphat cap-open" style="width:{capw}px;height:{h}px">'
                f'<img class="cap-base-img" src="{tinted_cap("#3a434e")}"/>'
                f'<span class="cap-q" style="font-size:{round(capw*0.34)}px">?</span></div>')
    cap_override = CAP_LOGO_DIR / f'{tid}.png'
    logo = data_url(cap_override if cap_override.exists() else LOGO_DIR / f'{tid}.png')
    inner = f'<img class="cap-logo" src="{logo}"/>' if logo else ''
    return (f'<div class="caphat" style="width:{capw}px;height:{h}px">'
            f'<img class="cap-base-img" src="{tinted_cap(color)}"/>{inner}</div>')


def card_html(p, pos, capw, headshot_url, from_color, to_color, badge, stat_disp=''):
    fn, ln = split_name(p['name'])
    fprim = logo_palette(p['from_tid'])[0] if p['from_tid'] else '#b23a48'
    accent = logo_palette(p['to_tid'])[0] if (p['committed'] and p['to_tid']) else fprim
    photo_bg = f'linear-gradient(120deg,{rgba(fprim,0.95)},{darken(fprim,0.45)})'
    tint_bg = f'linear-gradient(100deg,{rgba(accent,0.0)} 30%,{rgba(accent,0.16)} 75%,{rgba(accent,0.30)} 100%)'
    wm = (p['to_name'] if p['committed'] else p['from_name'] or '?')[:1].upper()
    _sh = face_shadow(p['pid'])
    wm_html = f'<img class="watermark-photo" src="{_sh}"/>' if _sh else f'<div class="watermark">{_esc(wm)}</div>'
    head = f'<img class="phead" src="{headshot_url}"/>' if headshot_url else ''
    fn_html = f'<span class="fn">{_esc(fn)}</span>' if fn else ''
    meta_extra = f'<span class="psport" style="color:#d9a94a">{_esc(stat_disp)}</span>' if stat_disp else ''
    return (
        f'<div class="pcard {"is-top" if pos<=3 else ""}">'
        f'<div class="tint" style="background:{tint_bg}"></div>'
        f'{wm_html}'
        f'<div class="pphoto"><div class="ring" style="background:{photo_bg}"></div>{SIL}{head}</div>'
        f'<div class="pcontent"><div class="nameblock">'
        f'<div class="pname">{fn_html}<span class="ln">{_esc(ln)}</span></div>'
        f'<div class="pmeta"><span class="ppos">{_esc(p["pos"])}</span><span class="psport">{_esc(p["conf"])}</span>{meta_extra}</div>'
        f'</div>'
        f'<div class="transfer">'
        f'<div class="capwrap"><span class="caplbl">From</span>{cap_html(p["from_tid"], from_color, capw)}</div>'
        f'<span class="t-arrow">{ARROW.format(s=round(capw*0.42))}</span>'
        f'<div class="capwrap"><span class="caplbl">To</span>{cap_html(p["to_tid"], to_color, capw, open_=not p["committed"])}</div>'
        f'</div></div>'
        f'<div class="prank {"top3" if pos<=3 else ""}"><div class="num"><span class="hash">#</span>{badge:,}</div></div>'
        f'</div>'
    )


# ---------------- UI ----------------
st.title('Top 10 Portal Players')
st.caption('Top 10 by 64A portal rank (1 = best) within your filters. Tweak filters, drop headshots / hat colors, then Download PNG.')

c1, c2 = st.columns(2)
sport = c1.radio('Sport', ['Baseball', 'Softball'], horizontal=True)
fmt = 'ig' if c2.radio('Format', ['Instagram (1080×1350)', 'Twitter/X (1600×900)'], horizontal=True).startswith('Instagram') else 'tw'
capw = 60 if fmt == 'ig' else 54

pool = load_pool(sport)
if not pool:
    st.warning(f'No {sport} portal players found.'); st.stop()

# ---- filters ----
st.markdown('**Filters**')
f1, f2, f3, f4 = st.columns([2, 2, 1.4, 1.4])
conf_opts = sorted({p['conf'] for p in pool if p['conf']} | {p['to_conf'] for p in pool if p.get('to_conf')})
pos_opts = sorted({p['pos'] for p in pool if p['pos']})
div_opts = [d for d in ['D-I', 'D-II', 'D-III'] if any(p['division'] == d for p in pool)]
sel_conf = f1.multiselect('Conference', conf_opts, default=[])
conf_scope = f1.radio('Conference applies to', ['From', 'To', 'Either'], index=0, horizontal=True,
                      help="Match the conference filter on the player's FROM school, their committed TO school, "
                           "or EITHER. e.g. To + SEC surfaces McKenzie Pickens (Louisville→LSU).")
sel_pos = f2.multiselect('Position', pos_opts, default=[])
sel_div = f3.multiselect('Division', div_opts, default=[])
commit = f4.radio('Commitment', ['All', 'Committed', 'Available'], index=0)

# date-entered filter + stat SORTER (order the board by a stat instead of portal rank)
import datetime as _dt
def _pdate(s):
    try: return pd.to_datetime(s).date()
    except: return None
HIT_KEYS = {'OPS', 'OBP', 'ISO', 'AVG', 'HR', 'wRC', 'wRAA'}
SORTS = {'Portal rank': None,
         'OPS': ('OPS', 'hi'), 'OBP': ('OBP', 'hi'), 'ISO': ('ISO', 'hi'), 'AVG': ('AVG', 'hi'),
         'HR': ('HR', 'hi'), 'wRC': ('wRC', 'hi'), 'wRAA': ('wRAA', 'hi'),
         'ERA': ('ERA', 'lo'), 'FIP': ('FIP', 'lo'), 'WHIP': ('WHIP', 'lo'), 'K/9': ('K9', 'hi')}
_edates = [d for d in (_pdate(p['entered']) for p in pool) if d]
dmin, dmax = (min(_edates), max(_edates)) if _edates else (_dt.date(2025, 10, 1), _dt.date.today())
s1, s2, s3 = st.columns([2, 2, 2])
ent_from = s1.date_input('Entered from', value=dmin, min_value=dmin, max_value=dmax)
ent_to = s2.date_input('Entered to', value=dmax, min_value=dmin, max_value=dmax)
sort_label = s3.selectbox('Sort board by', list(SORTS),
                          help='Orders the board by this stat (top N) instead of 64A portal rank — e.g. pos=3B + Available + OPS = top 3B hitters available. A hitting stat keeps hitters; a pitching stat keeps pitchers.')
date_active = (ent_from != dmin or ent_to != dmax)

def _conf_match(p):
    if not sel_conf:
        return True
    if conf_scope == 'From':
        return p['conf'] in sel_conf
    if conf_scope == 'To':
        return p['to_conf'] in sel_conf
    return p['conf'] in sel_conf or p['to_conf'] in sel_conf   # Either

filtered = [p for p in pool
            if _conf_match(p)
            and (not sel_pos or p['pos'] in sel_pos)
            and (not sel_div or p['division'] in sel_div)
            and (commit == 'All' or (commit == 'Committed' and p['committed']) or (commit == 'Available' and not p['committed']))
            and (not date_active or (_pdate(p['entered']) is not None and ent_from <= _pdate(p['entered']) <= ent_to))]
sort_key = SORTS[sort_label]
if sort_key is None:
    filtered.sort(key=lambda p: p['rank'])                       # 64A portal rank (1 = best)
else:
    skey, hilo = sort_key
    samp, floor = ('PA', 30) if skey in HIT_KEYS else ('IP', 15)  # min sample so a fluke can't top it
    filtered = [p for p in filtered if p['stats'].get(skey) is not None and (p['stats'].get(samp) or 0) >= floor]
    filtered.sort(key=lambda p: p['stats'][skey], reverse=(hilo == 'hi'))
if not filtered:
    st.warning('No players match those filters.'); st.stop()
# paginate into boards of 10 (the grid is fixed at 10 cards) so a "Top 40" = 4 boards
PER_BOARD = 10
n_boards = max(1, (len(filtered) + PER_BOARD - 1) // PER_BOARD)
def _plabel(pp):
    lo_ = (pp - 1) * PER_BOARD + 1; hi_ = min(pp * PER_BOARD, len(filtered))
    return f'Board {pp}  (#{lo_}–{hi_})'
page = st.selectbox(f'Board — {len(filtered)} players match → {n_boards} board(s) of {PER_BOARD}',
                    list(range(1, n_boards + 1)), format_func=_plabel)
_lo = (page - 1) * PER_BOARD
players = filtered[_lo:_lo + PER_BOARD]
rank_lo, rank_hi = _lo + 1, _lo + len(players)
title_main = 'TOP 10' if page == 1 else f'#{rank_lo}–{rank_hi}'

overrides = load_overrides()


def cap_color(tid):
    return overrides.get(tid) or (logo_palette(tid)[1] if tid else '#2a313b')


# ---- headshots + hat colors (remembered) ----
with st.expander('Headshots & hat colors (remembered)', expanded=False):
    st.caption('Drop a headshot per player; adjust a cap color where the logo blends in. Both persist.')
    changed = False
    for p in players:
        cached = HEADSHOTS / f'{p["pid"]}.png'
        cc = st.columns([3, 2, 1.3, 1.3])
        cc[0].markdown(f'**#{p["rank"]:,} · {p["name"]}**  \n{p["from_name"]}'
                       + (f' → {p["to_name"]}' if p['committed'] else '') + ('  ·  📷' if cached.exists() else ''))
        up = cc[1].file_uploader('headshot', type=['png', 'jpg', 'jpeg', 'webp'],
                                 key=f'hs_{p["pid"]}', label_visibility='collapsed')
        if up is not None:
            try:
                square_headshot(Image.open(up)).save(cached, 'PNG')
                st.toast(f'Saved headshot · {p["name"]}')
            except Exception as e:
                st.warning(f'Bad image: {e}')
        fc = cc[2].color_picker(f'From hat', cap_color(p['from_tid']), key=f'hf_{p["pid"]}')
        if p['from_tid'] and fc.lower() != cap_color(p['from_tid']).lower():
            overrides[p['from_tid']] = fc; changed = True
        if p['committed'] and p['to_tid']:
            tc = cc[3].color_picker(f'To hat', cap_color(p['to_tid']), key=f'ht_{p["pid"]}')
            if tc.lower() != cap_color(p['to_tid']).lower():
                overrides[p['to_tid']] = tc; changed = True
    if changed:
        save_overrides(overrides)


def headshot_for(pid):
    c = HEADSHOTS / f'{pid}.png'
    if not c.exists():
        return ''
    buf = BytesIO()
    square_headshot(Image.open(c)).save(buf, 'PNG')   # square at render too (fixes legacy non-square uploads)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


# ---------------- summary + generate ----------------
W, H = (1080, 1350) if fmt == 'ig' else (1600, 900)
scale = round(660 / W, 4)
fname = f'top10_portal_{sport.lower()}_{fmt}_b{page}.png'
st.markdown(f'**Board {page}/{n_boards} · {sport} · #{rank_lo}–{rank_hi}**: ' + ' · '.join(f'#{p["rank"]:,} {p["name"]}' for p in players))
gen = st.button('🎨 Generate image', type='primary')
if not gen:
    st.caption('Set filters / headshots / hat colors above, then click **Generate image** to render and download. '
               '(Kept off auto-refresh so changes are instant.)')
    st.stop()

# ---------------- build board (only on Generate) ----------------
def _statval(p):
    sk = SORTS[sort_label]
    if sk is None:
        return ''
    k = sk[0]; v = p['stats'].get(k)
    if v is None:
        return ''
    if k == 'HR': s = f'{int(v)}'
    elif k in ('ERA', 'FIP', 'WHIP'): s = f'{v:.2f}'
    elif k in ('wRC', 'wRAA', 'K9'): s = f'{v:.1f}'
    else: s = f'{v:.3f}'
    return f'{sort_label} {s}'
cards = ''.join(card_html(p, i + 1, capw, headshot_for(p['pid']), cap_color(p['from_tid']), cap_color(p['to_tid']),
                          (p['rank'] if SORTS[sort_label] is None else rank_lo + i), _statval(p))
                for i, p in enumerate(players))
brand_html = (
    f'<img class="logo64" src="{data_url(BRAND / "logo-64-analytics.png")}" alt="64A"/>'
    f'<span class="brand-div"></span>'
    f'<img class="partner" src="{data_url(BRAND / "logo-college-baseball-show.png")}" alt="CBS"/>'
    f'<img class="partner" src="{data_url(BRAND / "logo-jb.png")}" alt="JB"/>'
)
board = (
    f'<div class="board" data-format="{fmt}" id="capture">'
    f'<div class="bhead"><div class="brand">{brand_html}</div>'
    f'<div class="tag">Transfer Portal · {sport} · &rsquo;26</div></div>'
    f'<div class="btitle"><span class="t10">{title_main}</span><span>PORTAL PLAYERS</span><span class="rule"></span></div>'
    f'<div class="grid">{cards}</div>'
    f'<div class="bfoot"><span>64A Proprietary Rankings</span>'
    f'<span class="dotrow"><span class="pip"></span> 64analytics.com</span></div>'
    f'</div>'
)
css = font_face_css() + '\n' + DESIGN_CSS.replace('__CAP__', data_url(CAP_PNG))

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
  if(inner) inner.style.transform = 'none';                 // capture at true 1:1, not the preview scale
  if(wrap){{ wrap.style.width='{W}px'; wrap.style.height='{H}px'; wrap.style.overflow='visible'; }}
  try{{
    if(document.fonts){{ try{{ await document.fonts.load("800 76px 'Saira Condensed'"); await document.fonts.load("500 35px 'Saira Condensed'"); }}catch(e){{}} if(document.fonts.ready) await document.fonts.ready; }}
    var canvas = await html2canvas(el, {{scale:2, useCORS:true, backgroundColor:'#0b0e12'}});
    var a = document.createElement('a'); a.download='{fname}';
    a.href = canvas.toDataURL('image/png'); document.body.appendChild(a); a.click(); document.body.removeChild(a);
  }} finally {{ if(inner) inner.style.transform=pT; if(wrap) wrap.style.cssText=pW; btn.disabled=false; btn.textContent = t || 'Download PNG'; }}
}};
</script></body></html>"""

components.html(html, height=int(H * scale) + 80, scrolling=False)
