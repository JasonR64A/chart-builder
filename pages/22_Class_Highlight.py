"""Class Highlight — one team's 2026 transfer class as a social graphic (IG 1080x1350).

Ported from the Claude Design "Class Highlight.dc.html" board: cutout player collage
(hero + floating supporting players, grayscale sticker treatment) over a paper-tone
canvas with geometric accents, and a dark quarter-oval superlatives panel — class
ranking, headliners, value-added tiles, portal-share bars, and a US map of where the
class is coming from (player hometowns).

html2canvas-safe porting choices (lessons from Top10/Weekly Awards):
  - The design's SVG feColorMatrix/feMorphology cutout filters are baked into the
    headshot PNGs with PIL (grayscale + contrast + dilated color ring) — html2canvas
    cannot render SVG filter references.
  - The map's 64A watermark is an HTML <img> behind the map SVG, never an <image>
    inside it (raster-in-SVG exports blank).
  - Paper-grain is a PIL noise tile, not a feTurbulence data-URI background.
  - Inter is embedded as a base64 @font-face (Google-hosted fonts don't capture).
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
from PIL import Image, ImageDraw, ImageFilter

APP_DIR = Path(__file__).resolve().parent.parent
DATA = APP_DIR / 'data'
LOGO_DIR = APP_DIR / 'team_logos_512'
FONT_DIR = APP_DIR / 'assets' / 'fonts'
EMBLEM = APP_DIR / 'assets' / 'portal-entrant' / '64-emblem.png'
MAP_JS = APP_DIR / 'assets' / 'class-highlight' / 'commit-map.js'
HEADSHOTS = DATA / 'class_highlight_headshots'
COLOR_JSON = DATA / 'class_highlight_colors.json'
LAYOUT_JSON = DATA / 'class_highlight_layout.json'
HEADSHOTS.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title='Class Highlight', layout='wide')

INK = '#141414'
PANEL_TXT = '#fdf6f1'

# state_id in teams.csv = 1-based alphabetical index of the 50 states (verified:
# AL=1, CA=5, CO=6, FL=9, KS=16, OH=35, TX=43)
STATES_ALPHA = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID',
                'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS',
                'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
                'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV',
                'WI', 'WY']
STATE_CENTROID = {
    'AL': (32.8, -86.8), 'AK': (61.4, -152.3), 'AZ': (34.3, -111.7), 'AR': (34.9, -92.4),
    'CA': (37.2, -119.3), 'CO': (39.0, -105.5), 'CT': (41.6, -72.7), 'DE': (39.0, -75.5),
    'FL': (28.6, -82.4), 'GA': (32.6, -83.4), 'HI': (20.3, -156.4), 'ID': (44.4, -114.6),
    'IL': (40.0, -89.2), 'IN': (39.9, -86.3), 'IA': (42.1, -93.5), 'KS': (38.5, -98.4),
    'KY': (37.5, -85.3), 'LA': (31.1, -92.0), 'ME': (45.4, -69.2), 'MD': (39.0, -76.8),
    'MA': (42.3, -71.8), 'MI': (44.3, -85.4), 'MN': (46.3, -94.3), 'MS': (32.7, -89.7),
    'MO': (38.4, -92.5), 'MT': (47.0, -109.6), 'NE': (41.5, -99.8), 'NV': (39.3, -116.6),
    'NH': (43.7, -71.6), 'NJ': (40.2, -74.7), 'NM': (34.4, -106.1), 'NY': (42.9, -75.5),
    'NC': (35.6, -79.4), 'ND': (47.4, -100.5), 'OH': (40.3, -82.8), 'OK': (35.6, -97.5),
    'OR': (43.9, -120.6), 'PA': (40.9, -77.8), 'RI': (41.7, -71.6), 'SC': (33.9, -80.9),
    'SD': (44.4, -100.2), 'TN': (35.8, -86.4), 'TX': (31.5, -99.3), 'UT': (39.3, -111.7),
    'VT': (44.1, -72.7), 'VA': (37.5, -78.9), 'WA': (47.4, -120.4), 'WV': (38.6, -80.6),
    'WI': (44.6, -89.9), 'WY': (43.0, -107.6), 'DC': (38.9, -77.0),
}

# collage slots from the design's DCLogic (left %, bottom px, height px, z, rotate deg)
# — max 5 players total (hero + 4 supporting), per user 2026-07-14
SUPP_SLOTS = [
    (17, 560, 560, 20, -5), (55, 585, 545, 18, 6), (25, 220, 660, 26, -2),
    (41, 720, 470, 10, 0),
]

SILHOUETTE = ('<svg viewBox="0 0 240 480" style="height:100%;width:auto;display:block;" '
              'preserveAspectRatio="xMidYMax meet">'
              '<defs><linearGradient id="silFill{uid}" x1="0" y1="0" x2="0.35" y2="1">'
              '<stop offset="0" stop-color="#d0d0d0"/><stop offset="0.5" stop-color="#8f8f8f"/>'
              '<stop offset="1" stop-color="#454545"/></linearGradient></defs>'
              '<circle cx="120" cy="58" r="40" fill="url(#silFill{uid})"/>'
              '<path fill="url(#silFill{uid})" d="M120 96 C150 96 158 110 172 128 C196 150 206 176 '
              '210 210 L206 300 C206 332 196 342 182 342 L178 480 L62 480 L58 342 C44 342 34 332 '
              '34 300 L30 210 C34 176 44 150 68 128 C82 110 90 96 120 96 Z"/></svg>')


# ---------------- generic helpers (house pattern, per Top10 page) ----------------
def norm(s):
    s = str(s).strip()
    return s[:-2] if s.endswith('.0') else s


def _esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def data_url(path, mime='image/png'):
    p = Path(path)
    if not p.exists():
        return ''
    return f'data:{mime};base64,' + base64.b64encode(p.read_bytes()).decode()


@st.cache_data(show_spinner=False)
def _csv(name, **kw):
    return pd.read_csv(DATA / name, dtype=str, encoding='latin-1', keep_default_na=False, **kw)


@st.cache_data(show_spinner=False)
def font_face_css():
    p = FONT_DIR / 'Inter-Variable.ttf'
    if not p.exists():
        return ''
    b64 = base64.b64encode(p.read_bytes()).decode()
    return ("@font-face{font-family:'Inter';font-style:normal;font-weight:100 900;"
            "src:url(data:font/ttf;base64,%s) format('truetype');}" % b64)


@st.cache_data(show_spinner=False)
def _logo_id_resolver():
    """team id -> id whose logo file exists (softball ids share the baseball logo)."""
    teams = _csv('teams.csv')
    by_ncaa, nc_of = {}, {}
    for _, r in teams.iterrows():
        i = norm(r['id']); nc = norm(r['team_id_ncaa'])
        nc_of[i] = nc
        by_ncaa.setdefault(nc, []).append(i)
    have = lambda i: (LOGO_DIR / f'{i}.png').exists()
    return {i: (i if have(i) else next((s for s in by_ncaa.get(nc_of[i], []) if have(s)), i))
            for i in nc_of}


@st.cache_data(show_spinner=False)
def logo_primary(team_id):
    """Dominant saturated logo color -> default team color. Brand-red fallback."""
    fallback = '#BA0C2F'
    p = LOGO_DIR / f"{_logo_id_resolver().get(norm(team_id), norm(team_id))}.png"
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
        c = buckets.most_common(1)[0][0]
        return '#%02x%02x%02x' % c
    except Exception:
        return fallback


def load_colors():
    try:
        return json.loads(COLOR_JSON.read_text()) if COLOR_JSON.exists() else {}
    except Exception:
        return {}


def save_colors(d):
    try:
        COLOR_JSON.write_text(json.dumps(d))
    except Exception:
        pass


# ---------------- sticker cutout (replaces the design's SVG filters) ----------------
def sticker(pid, ring_hex, radius, boost):
    """Grayscale + contrast + dilated ring 'sticker' baked into the PNG.
    Mirrors the design's feColorMatrix(saturate 0) + feComponentTransfer + feMorphology
    dilate + team-color flood — done in PIL so the export matches the preview."""
    src = HEADSHOTS / f'{pid}.png'
    if not src.exists():
        return ''
    im = Image.open(src).convert('RGBA')
    im.thumbnail((900, 900), Image.LANCZOS)
    arr = np.array(im).astype('float32')
    g = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    g = np.clip(g * boost - 255 * 0.05, 0, 255)
    alpha = im.getchannel('A')
    pad = radius + 2
    big = Image.new('L', (im.width + 2 * pad, im.height + 2 * pad), 0)
    big.paste(alpha, (pad, pad))
    dil = big.filter(ImageFilter.MaxFilter(2 * radius + 1)) if radius > 0 else big
    n = int(ring_hex.lstrip('#'), 16)
    ring = Image.new('RGBA', big.size, ((n >> 16) & 255, (n >> 8) & 255, n & 255, 0))
    ring.putalpha(dil)
    gray = np.zeros((im.height, im.width, 4), dtype='uint8')
    gray[:, :, 0] = gray[:, :, 1] = gray[:, :, 2] = g.astype('uint8')
    gray[:, :, 3] = np.array(alpha)
    out = ring.copy()
    out.alpha_composite(Image.fromarray(gray, 'RGBA'), (pad, pad))
    buf = BytesIO()
    out.save(buf, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def _b64png(im):
    buf = BytesIO()
    im.save(buf, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


@st.cache_data(show_spinner=False)
def eggshell_tile():
    """Paper texture BAKED into the canvas color (no blend modes — html2canvas
    ignores mix-blend-mode, which is why the old grain overlay exported wrong)."""
    rng = np.random.default_rng(64)
    S = 320
    base = np.full((S, S, 3), (236, 231, 221), dtype='float32')
    base += rng.normal(0, 11, (S, S, 1))                        # grain (visible, not subtle)
    n = 320                                                     # paper flecks
    ys, xs = rng.integers(0, S, n), rng.integers(0, S, n)
    for y, x in zip(ys, xs):
        base[max(0, y-1):y+2, max(0, x-1):x+2] -= rng.uniform(14, 40)
    return _b64png(Image.fromarray(base.clip(0, 255).astype('uint8'), 'RGB'))


def _panel_pts(scale=1.0):
    """Sample the quarter-oval outline (two cubic beziers from the design)."""
    def cubic(p0, p1, p2, p3, n=90):
        return [((1-t)**3*p0[0] + 3*(1-t)**2*t*p1[0] + 3*(1-t)*t*t*p2[0] + t**3*p3[0],
                 (1-t)**3*p0[1] + 3*(1-t)**2*t*p1[1] + 3*(1-t)*t*t*p2[1] + t**3*p3[1])
                for t in (i/n for i in range(1, n+1))]
    pts = [(620, 0), (360, 0)]
    pts += cubic((360, 0), (232, 360), (150, 620), (96, 1000))
    pts += cubic((96, 1000), (74, 1150), (66, 1260), (62, 1350))
    pts += [(620, 1350)]
    return [(x*scale, y*scale) for x, y in pts]


@st.cache_data(show_spinner=False)
def panel_png(stipple):
    """The charcoal quarter-oval, rendered server-side as a PNG: soft left shadow,
    optional stipple texture, hairline curve highlight. Replaces the design's
    preserveAspectRatio="none" SVG + CSS drop-shadow, both of which html2canvas
    mangles — an <img> exports 1:1."""
    S = 2
    W, H = 620*S, 1350*S
    poly = _panel_pts(S)
    mask = Image.new('L', (W, H), 0)
    ImageDraw.Draw(mask).polygon(poly, fill=255)

    img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    # shadow: the panel silhouette nudged left and blurred
    sh = Image.new('L', (W, H), 0)
    sh.paste(mask, (-14*S, 0))
    sh = sh.filter(ImageFilter.GaussianBlur(13*S))
    shadow = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    shadow.putalpha(sh.point(lambda v: int(v * 0.30)))
    img.alpha_composite(shadow)

    body = np.zeros((H, W, 4), dtype='uint8')
    body[:, :, 0], body[:, :, 1], body[:, :, 2] = 38, 38, 43
    body[:, :, 3] = np.array(mask)
    if stipple:
        rng = np.random.default_rng(26)
        m = np.array(mask) > 0
        # stipple dots, denser + brighter toward the curved edge for a print feel
        n_dots = 42000
        ys = rng.integers(0, H, n_dots)
        xs = rng.integers(0, W, n_dots)
        keep = m[ys, xs]
        ys, xs = ys[keep], xs[keep]
        edge_x = np.array([min((p[0] for p in poly if abs(p[1]-y) < 30*S), default=0) for y in range(0, H, 40*S)])
        for y, x in zip(ys, xs):
            e = edge_x[min(y // (40*S), len(edge_x)-1)]
            fall = max(0.30, 1.0 - (x - e) / (260.0*S))     # brighter near the curve
            a = int(rng.uniform(22, 60) * fall)
            lum = int(rng.uniform(190, 255))
            r2 = 2 if rng.random() < 0.25 else 1            # a few chunkier dots
            body[y:y+r2, x:x+r2, 0] = np.minimum(255, body[y:y+r2, x:x+r2, 0].astype(int) + lum*a//255)
            body[y:y+r2, x:x+r2, 1] = np.minimum(255, body[y:y+r2, x:x+r2, 1].astype(int) + lum*a//255)
            body[y:y+r2, x:x+r2, 2] = np.minimum(255, body[y:y+r2, x:x+r2, 2].astype(int) + (lum+8)*a//255)
    img.alpha_composite(Image.fromarray(body, 'RGBA'))
    edge = [(x, y) for x, y in poly[1:-1]]
    ImageDraw.Draw(img).line(edge, fill=(255, 255, 255, 36), width=2*S)
    return _b64png(img.resize((620, 1350), Image.LANCZOS))


def load_layout():
    try:
        return json.loads(LAYOUT_JSON.read_text()) if LAYOUT_JSON.exists() else {}
    except Exception:
        return {}


def save_layout(d):
    try:
        LAYOUT_JSON.write_text(json.dumps(d))
    except Exception:
        pass


# ---------------- data ----------------
@st.cache_data(show_spinner=False)
def load_world(sport):
    teams = _csv('teams.csv')
    tinfo = {norm(r['id']): r for _, r in teams.iterrows()}
    prp = _csv('portal_rank_player.csv')
    prp = prp[prp['year'].map(norm) == '2026'].copy()
    prp['pid'] = prp['player_id'].map(norm)
    prp['tid'] = prp['team_id'].map(norm)
    prp['ntid'] = prp['new_team_id'].map(lambda v: norm(v) if str(v).strip() else '')
    prp['rk'] = pd.to_numeric(prp['sixty_four_rating_portal_player'], errors='coerce')
    sport_of = lambda tid: tinfo.get(tid, {}).get('sport', '') if tid in tinfo else ''
    prp = prp[[sport_of(t) == sport for t in prp['tid']]]

    players = _csv('players.csv')
    pname, ppos, phome = {}, {}, {}
    for _, r in players.iterrows():
        i = norm(r['id'])
        pname[i] = r['player_name']; ppos[i] = r['position']; phome[i] = r['hometown']

    prt = _csv('portal_rank_team.csv')
    prt = prt[prt['year'].map(norm) == '2026']
    class_rank = {norm(r['team_id']): norm(r['rank']) for _, r in prt.iterrows()
                  if sport_of(norm(r['team_id'])) == sport}

    hit = pd.read_csv(DATA / 'hitting.csv', encoding='latin-1', low_memory=False,
                      usecols=['player_id', 'year', 'plate_appearances', 'doubles', 'triples',
                               'home_runs', 'on_base_plus_slugging', 'strikeout_to_walk_ratio',
                               'weighted_runs_above_average'])
    hit = hit[hit['year'].astype(str).str.startswith('2026')]
    hstat = {}
    for _, r in hit.iterrows():
        d = hstat.setdefault(norm(r['player_id']),
                             dict(pa=0, xbh=0, _big=0, ops=None, kbb=None, wraa=None))
        pa = float(r['plate_appearances'] or 0)
        d['pa'] += pa
        d['xbh'] += float(r['doubles'] or 0) + float(r['triples'] or 0) + float(r['home_runs'] or 0)
        if pa >= d['_big']:   # rate stats from the biggest-PA stint
            d['_big'] = pa
            d['ops'] = r['on_base_plus_slugging']; d['kbb'] = r['strikeout_to_walk_ratio']
            d['wraa'] = r['weighted_runs_above_average']
    pit = pd.read_csv(DATA / 'pitching.csv', encoding='latin-1', low_memory=False,
                      usecols=['player_id', 'year', 'innings_pitched', 'strikeouts',
                               'earned_run_average', 'strikeout_to_walk_ratio',
                               'fielding_independent_pitching'])
    pit = pit[pit['year'].astype(str).str.startswith('2026')]
    pstat = {}
    for _, r in pit.iterrows():
        d = pstat.setdefault(norm(r['player_id']),
                             dict(ip=0.0, k=0, _big=0, era=None, kbb=None, fip=None))
        ip = float(r['innings_pitched'] or 0)
        d['ip'] += ip; d['k'] += float(r['strikeouts'] or 0)
        if ip >= d['_big']:
            d['_big'] = ip
            d['era'] = r['earned_run_average']; d['kbb'] = r['strikeout_to_walk_ratio']
            d['fip'] = r['fielding_independent_pitching']

    # raw 64A ratings + hitter/pitcher side from the ranked file (the values the
    # team rankings are built from; TWP counts on both sides)
    PITCH_POS = {'P', 'RHP', 'LHP', 'SP', 'RP'}
    val = {}
    rankedf = DATA / 'portal' / f'{sport.lower()}_ranked.csv'
    if rankedf.exists():
        rk = pd.read_csv(rankedf, dtype=str, keep_default_na=False,
                         usecols=['player_id_64a', 'position', 'twp',
                                  'sixty_four_rating_portal_player'])
        for _, r in rk.iterrows():
            pid = norm(r['player_id_64a'])
            if not pid:
                continue
            try:
                rating = float(r['sixty_four_rating_portal_player'])
            except ValueError:
                continue
            twp = False
            try:
                twp = float(r['twp'] or 0) > 0
            except ValueError:
                pass
            pos = (r['position'] or ppos.get(pid, '')).upper()
            is_pit = twp or pos in PITCH_POS
            is_hit = twp or pos not in PITCH_POS
            val[pid] = (rating, is_hit, is_pit)

    return prp, tinfo, pname, ppos, phome, class_rank, hstat, pstat, val


def class_sums(pids, hstat, pstat, val):
    s = dict(ip=0.0, k=0, pa=0, xbh=0, hit_val=0.0, pit_val=0.0)
    for pid in pids:
        p = pstat.get(pid); h = hstat.get(pid)
        if p: s['ip'] += p['ip']; s['k'] += p['k']
        if h: s['pa'] += h['pa']; s['xbh'] += h['xbh']
        v = val.get(pid)
        if v:
            rating, is_hit, is_pit = v
            if is_hit: s['hit_val'] += rating
            if is_pit: s['pit_val'] += rating
    return s


# ---------------- UI ----------------
st.title('Class Highlight')
st.caption('One team’s 2026 transfer class — collage + superlatives panel (from Claude Design). '
           'IG 1080×1350. Upload transparent-background player cutouts for the collage.')

c1, c2, c3 = st.columns([1, 2, 1])
sport = c1.selectbox('Sport', ['Baseball', 'Softball'])
prp, tinfo, pname, ppos, phome, class_rank, hstat, pstat, val = load_world(sport)

DRAFTED_PRO = '2269'
commits = prp[(prp['ntid'] != '') & (prp['ntid'] != DRAFTED_PRO)]
by_team = commits.groupby('ntid')
team_ids = [t for t in by_team.groups if t in tinfo]
team_ids.sort(key=lambda t: (int(class_rank.get(t, 9999) or 9999), tinfo[t]['name']))
if not team_ids:
    st.warning('No committed classes found.')
    st.stop()
tlabel = {t: f"{tinfo[t]['name']}  (class #{class_rank.get(t, '—')}, {len(by_team.groups[t])} commits)"
          for t in team_ids}
team = c2.selectbox('Team', team_ids, format_func=lambda t: tlabel[t])

colors = load_colors()
team_color = c3.color_picker('Team color', colors.get(team) or logo_primary(team))
if (colors.get(team) or logo_primary(team)).lower() != team_color.lower():
    colors[team] = team_color
    save_colors(colors)

o1, o2, o3, o4, o5 = st.columns(5)
accents_team = o1.toggle('Accents in team color', value=False)
texture = o2.toggle('Paper grain', value=True)
stipple = o3.toggle('Panel stipple', value=True)
hero_r = o4.slider('Hero outline', 0, 16, 7)
supp_r = o5.slider('Supporting outline', 0, 14, 5)

cls = commits[commits['ntid'] == team].copy().sort_values('rk')
cls_pids = list(cls['pid'])
show = cls.head(5)
show_pids = list(show['pid'])

names = {p: (pname.get(p) or cls[cls['pid'] == p].iloc[0]['name']) for p in cls_pids}
hero_pid = st.selectbox('Hero (front & center)', show_pids,
                        format_func=lambda p: names.get(p, p))

layout = load_layout()
layout_changed = False
with st.expander('🧩 Collage layout — nudge / resize each player', expanded=False):
    st.caption('Offsets from each player’s default spot. Saved per team, so they stick.')
    for pid in show_pids:
        key = f'{team}:{pid}'
        cur = layout.get(key, {})
        cc = st.columns([2, 2, 2, 2, 2])
        cc[0].markdown(f"**{_esc(names[pid])}**" + (' · hero' if pid == hero_pid else ''))
        dx = cc[1].slider('← left · right →', -30, 30, int(cur.get('dx', 0)), key=f'dx_{key}',
                          help='% of canvas width')
        dy = cc[2].slider('↓ lower · higher ↑', -250, 250, int(cur.get('dy', 0)), key=f'dy_{key}')
        sc = cc[3].slider('size %', 55, 145, int(cur.get('sc', 100)), key=f'sc_{key}')
        rot = cc[4].slider('↺ rotate ↻ (°)', -25, 25, int(cur.get('rot', 0)), key=f'rot_{key}')
        if (dx, dy, sc, rot) != (cur.get('dx', 0), cur.get('dy', 0), cur.get('sc', 100), cur.get('rot', 0)):
            layout[key] = {'dx': dx, 'dy': dy, 'sc': sc, 'rot': rot}
            layout_changed = True
if layout_changed:
    save_layout(layout)


def _adj(pid):
    d = layout.get(f'{team}:{pid}', {})
    return (int(d.get('dx', 0)), int(d.get('dy', 0)), int(d.get('sc', 100)) / 100.0,
            int(d.get('rot', 0)))

def _strip_bg(im):
    """Best-effort background removal. rembg is intentionally NOT installed on Render
    (its ~300MB inference spike would OOM the instance) — there, opaque uploads are
    saved as-is with a warning; run tools/remove_bg.py locally for clean cutouts."""
    try:
        from rembg import remove
        return remove(im), True
    except Exception:
        return im, False


with st.expander('📸 Player cutouts (PNG/WebP/JPG — remembered per player)', expanded=False):
    st.caption('Transparent PNGs drop straight in. Opaque images (JPGs) get automatic '
               'background removal when available; otherwise upload a pre-cut PNG '
               '(or batch-strip locally with tools/remove_bg.py).')
    for pid in show_pids:
        cc = st.columns([2, 3, 1])
        cc[0].markdown(f"**{_esc(names[pid])}** · {_esc(ppos.get(pid, ''))}")
        up = cc[1].file_uploader('cutout', type=['png', 'webp', 'jpg', 'jpeg'], key=f'ch_{pid}',
                                 label_visibility='collapsed')
        if up is not None:
            try:
                im = Image.open(up).convert('RGBA')
                alpha = np.array(im.getchannel('A'))
                if (alpha < 250).mean() < 0.02:   # effectively opaque -> needs stripping
                    im, ok = _strip_bg(im)
                    if not ok:
                        st.warning(f'{names[pid]}: no transparency and no local remover — '
                                   'saved as-is; the collage will show the photo background.')
                im.convert('RGBA').save(HEADSHOTS / f'{pid}.png', 'PNG')
                st.toast(f'Saved cutout · {names[pid]}')
            except Exception as e:
                st.warning(f'Bad image: {e}')
        cc[2].markdown('✅' if (HEADSHOTS / f'{pid}.png').exists() else '—')

gen = st.button('🎨 Generate image', type='primary')
if not gen:
    st.caption('Pick the team, drop in cutouts, then **Generate image**.')
    st.stop()

# ---------------- computed panel values ----------------
sums = class_sums(cls_pids, hstat, pstat, val)
team_sums = {t: class_sums(list(g['pid']), hstat, pstat, val) for t, g in by_team}
mx = {k: max((s[k] for s in team_sums.values()), default=1) or 1 for k in ('ip', 'k', 'pa', 'xbh')}
bar = lambda v, k: min(100, 100 * v / mx[k])
# national rank of THIS team's summed portal value, per side (1 = most value added)
val_rank = lambda k: 1 + sum(1 for s in team_sums.values() if s[k] > sums[k])
hit_rank, pit_rank = val_rank('hit_val'), val_rank('pit_val')

# single headliner: the class's best-rated commit, name in 64A red + 2026 stat line
hl = cls_pids[0]
hl_v = val.get(hl)
hl_is_pitcher = bool(hl_v and hl_v[2] and not hl_v[1])
hl_rk = prp[prp['pid'] == hl]['rk'].iloc[0] if (prp['pid'] == hl).any() else None
def _f(v, nd=3):
    try:
        s = f'{float(v):.{nd}f}'
        return s.lstrip('0') if nd == 3 and s.startswith('0.') else s
    except (TypeError, ValueError):
        return '—'
if hl_is_pitcher:
    hp = pstat.get(hl, {})
    hl_stats = [('ERA', _f(hp.get('era'), 2)), ('K/BB', _f(hp.get('kbb'), 2)),
                ('FIP', _f(hp.get('fip'), 2))]
else:
    hh = hstat.get(hl, {})
    hl_stats = [('OPS', _f(hh.get('ops'))), ('K/BB', _f(hh.get('kbb'), 2)),
                ('wRAA', _f(hh.get('wraa'), 1))]
hl_stats.append(('PORTAL', f'#{int(hl_rk)}' if pd.notna(hl_rk) else '—'))
stat_chips = ''.join(
    f'<div style="text-align:left;"><div style="font-size:12px;font-weight:800;letter-spacing:0.1em;'
    f'color:rgba(255,255,255,0.6);">{k}</div>'
    f'<div style="font-size:24px;font-weight:900;letter-spacing:-0.02em;">{v}</div></div>'
    for k, v in hl_stats)
headliners = (
    f'<div style="display:flex;align-items:baseline;gap:10px;">'
    f'<span style="font-size:14px;font-weight:800;background:rgba(255,255,255,0.16);'
    f'border-radius:4px;padding:3px 8px;">{_esc(ppos.get(hl, "") or "ATH")}</span>'
    f'<span style="font-size:34px;font-weight:900;color:#e94f60;letter-spacing:-0.01em;'
    f'text-shadow:0 2px 8px rgba(0,0,0,0.5);">{_esc(names[hl])}</span></div>'
    f'<div style="display:flex;gap:22px;margin-top:10px;">{stat_chips}</div>')

def bar_row(label, sub, val_txt, width):
    subhtml = f' <span style="color:rgba(255,255,255,0.55);">· {sub}</span>' if sub else ''
    return (f'<div><div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;">'
            f'<span style="font-size:15px;font-weight:800;letter-spacing:0.07em;text-transform:uppercase;'
            f'color:rgba(255,255,255,0.85);white-space:nowrap;">{label}{subhtml}</span>'
            f'<span style="font-size:30px;font-weight:900;letter-spacing:-0.02em;">{val_txt}</span></div>'
            f'<div style="height:12px;border-radius:7px;background:rgba(0,0,0,0.24);overflow:hidden;">'
            f'<div style="height:100%;width:{width:.0f}%;background:{PANEL_TXT};border-radius:7px;"></div></div></div>')

# bar width = where this class sits vs the nation's best class (full bar = #1 overall)
bars = (bar_row('IP Added', 'Pitching', f'{sums["ip"]:.0f}', bar(sums['ip'], 'ip'))
        + bar_row('K Added', 'Pitching', f'+{sums["k"]:.0f}', bar(sums['k'], 'k'))
        + bar_row('PA Added', 'Hitting', f'{sums["pa"]:.0f}', bar(sums['pa'], 'pa'))
        + bar_row('XBH Added', 'Hitting', f'+{sums["xbh"]:.0f}', bar(sums['xbh'], 'xbh')))

# map dots: every commit's hometown state (fallback: origin school's state)
dots = []
for _, r in cls.iterrows():
    ab = ''
    home = phome.get(r['pid'], '')
    if ',' in home:
        ab = home.rsplit(',', 1)[1].strip().upper()[:2]
    if ab not in STATE_CENTROID:
        sid = norm(tinfo.get(r['tid'], {}).get('state_id', '')) if r['tid'] in tinfo else ''
        try:
            ab = STATES_ALPHA[int(sid) - 1] if sid else ''
        except Exception:
            ab = ''
    if ab in STATE_CENTROID:
        lat, lng = STATE_CENTROID[ab]
        dots.append({'lat': lat, 'lng': lng, 'name': names[r['pid']]})
map_players = json.dumps(dots).replace('"', '&quot;')

# ---------------- collage ----------------
tri = team_color if accents_team else INK
xc = team_color if accents_team else INK
supp_pids = [p for p in show_pids if p != hero_pid]

def player_div(pid, wrap, ring, radius):
    img = sticker(pid, ring, radius, 1.12)
    inner = (f'<img src="{img}" alt="" style="height:100%;width:auto;display:block;"/>' if img
             else SILHOUETTE.format(uid=pid))
    return f'<div style="{wrap}">{inner}</div>'

hdx, hdy, hsc, hrot = _adj(hero_pid)
hero_wrap = (f'position:absolute;bottom:{hdy}px;left:{43+hdx}%;'
             f'transform:translateX(-50%) rotate({hrot}deg);transform-origin:bottom center;'
             f'height:{int(1000*hsc)}px;z-index:40;display:flex;align-items:flex-end;justify-content:center;')
collage = player_div(hero_pid, hero_wrap, team_color, hero_r)
for i, pid in enumerate(supp_pids):
    left, bottom, h, z, rot = SUPP_SLOTS[i % len(SUPP_SLOTS)]
    dx, dy, sc, drot = _adj(pid)
    wrap = (f'position:absolute;bottom:{bottom+dy}px;left:{left+dx}%;'
            f'transform:translateX(-50%) rotate({rot+drot}deg);transform-origin:bottom center;'
            f'height:{int(h*sc)}px;z-index:{z};display:flex;align-items:flex-end;justify-content:center;')
    collage += player_div(pid, wrap, INK, supp_r)

accents_svg = f'''<svg viewBox="0 0 1080 1350" width="1080" height="1350" style="position:absolute;inset:0;pointer-events:none;z-index:1;">
<polyline points="150,360 150,232 330,232" fill="none" stroke="{INK}" stroke-width="30"/>
<polyline points="205,360 205,286 300,286" fill="none" stroke="{INK}" stroke-width="12"/>
<polygon points="470,300 690,300 580,470" fill="none" stroke="{INK}" stroke-width="4" stroke-dasharray="3 13" stroke-linecap="round"/>
<polygon points="720,470 860,470 790,600" fill="none" stroke="{tri}" stroke-width="4" stroke-dasharray="3 13" stroke-linecap="round"/>
<path d="M110,560 A150,150 0 0 1 260,410" fill="none" stroke="{tri}" stroke-width="5" stroke-dasharray="2 15" stroke-linecap="round"/>
<path d="M110,600 A190,190 0 0 1 300,410" fill="none" stroke="{tri}" stroke-width="5" stroke-dasharray="2 15" stroke-linecap="round"/>
<g stroke="{xc}" stroke-width="14" stroke-linecap="round">
<line x1="150" y1="1120" x2="210" y2="1180"/><line x1="210" y1="1120" x2="150" y2="1180"/>
<line x1="640" y1="360" x2="690" y2="410"/><line x1="690" y1="360" x2="640" y2="410"/>
<line x1="705" y1="415" x2="748" y2="458"/><line x1="748" y1="415" x2="705" y2="458"/></g>
<g stroke="{team_color}" stroke-width="26" stroke-linecap="butt" opacity="0.92">
<line x1="560" y1="1120" x2="710" y2="900"/>
<line x1="620" y1="1170" x2="770" y2="950"/>
<line x1="680" y1="1215" x2="820" y2="1005"/></g></svg>'''

rank_txt = class_rank.get(team, '—')
emblem = data_url(EMBLEM)

# map pinned to the panel's wide bottom: pulled left into the curve, ~2.5x the old area
panel = f'''
<img src="{panel_png(stipple)}" alt="" style="position:absolute;top:0;right:0;width:620px;height:1350px;z-index:3;"/>
<div style="position:absolute;top:0;right:0;width:620px;height:1350px;z-index:4;color:{PANEL_TXT};padding:196px 52px 30px 312px;display:flex;flex-direction:column;gap:14px;">
  <div>
    <div style="font-size:15px;font-weight:800;letter-spacing:0.22em;text-transform:uppercase;color:rgba(255,255,255,0.86);">2026 Class Ranking</div>
    <div style="display:flex;align-items:baseline;gap:10px;margin-top:2px;margin-left:-64px;">
      <span style="font-size:110px;font-weight:900;line-height:0.82;letter-spacing:-0.04em;text-shadow:0 4px 18px rgba(0,0,0,0.85),0 0 3px rgba(0,0,0,0.6);-webkit-text-stroke:1.5px rgba(0,0,0,0.35);">#{_esc(rank_txt)}</span>
      <span style="font-size:17px;font-weight:700;color:rgba(255,255,255,0.82);padding-bottom:12px;">NATIONAL<br/>(64A Portal)</span>
    </div>
  </div>
  <div style="height:1px;background:rgba(255,255,255,0.2);"></div>
  <div>
    <div style="font-size:15px;font-weight:800;letter-spacing:0.22em;text-transform:uppercase;color:rgba(255,255,255,0.72);margin-bottom:10px;">Headliner</div>
    {headliners}
  </div>
  <div style="height:1px;background:rgba(255,255,255,0.2);"></div>
  <div>
    <div style="font-size:16px;font-weight:800;letter-spacing:0.22em;text-transform:uppercase;color:rgba(255,255,255,0.72);margin-bottom:12px;margin-left:-110px;text-align:center;">Portal Value Added</div>
    <div style="display:flex;gap:16px;margin-left:-110px;">
      <div style="flex:1;background:rgba(0,0,0,0.22);border-radius:12px;padding:20px 22px;text-align:center;">
        <div style="font-size:58px;font-weight:900;line-height:1;letter-spacing:-0.03em;">#{pit_rank}</div>
        <div style="font-size:14px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:rgba(255,255,255,0.78);margin-top:8px;">Portal Value · Pitching</div>
      </div>
      <div style="flex:1;background:rgba(0,0,0,0.22);border-radius:12px;padding:20px 22px;text-align:center;">
        <div style="font-size:58px;font-weight:900;line-height:1;letter-spacing:-0.03em;">#{hit_rank}</div>
        <div style="font-size:14px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:rgba(255,255,255,0.78);margin-top:8px;">Portal Value · Hitting</div>
      </div>
    </div>
  </div>
  <div style="display:flex;flex-direction:column;gap:12px;margin-left:-110px;">{bars}</div>
  <div style="margin-top:auto;margin-bottom:26px;">
    <div style="font-size:15px;font-weight:800;letter-spacing:0.22em;text-transform:uppercase;color:rgba(255,255,255,0.72);margin-bottom:6px;margin-left:-150px;">Where They're Coming From</div>
    <div style="width:472px;height:400px;position:relative;margin-left:-216px;">
      <img src="{emblem}" alt="" style="position:absolute;left:50%;top:50%;width:150px;height:150px;transform:translate(-50%,-50%);opacity:0.20;"/>
      <us-commit-map players="{map_players}" accent="#ffffff" land="rgba(255,255,255,0.14)" border="rgba(255,255,255,0.38)" style="position:absolute;inset:0;"></us-commit-map>
    </div>
  </div>
</div>'''

team_logo = data_url(LOGO_DIR / f"{_logo_id_resolver().get(norm(team), norm(team))}.png")
team_logo_html = (f'<img src="{team_logo}" alt="" style="position:absolute;top:40px;right:48px;'
                  f'width:84px;height:84px;object-fit:contain;z-index:6;"/>') if team_logo else ''
header = f'''
{team_logo_html}
<div style="position:absolute;top:0;left:0;right:0;z-index:6;display:flex;align-items:center;gap:22px;padding:40px 48px 0 48px;">
  <img src="{emblem}" alt="64 Analytics" style="width:82px;height:82px;display:block;flex:none;"/>
  <div style="line-height:1;">
    <div style="font-size:20px;font-weight:700;letter-spacing:0.34em;color:#BA0C2F;text-transform:uppercase;">64 Analytics</div>
    <div style="font-size:52px;font-weight:900;letter-spacing:-0.02em;text-transform:uppercase;color:{INK};margin-top:4px;">Class Highlight</div>
  </div>
</div>
<div style="position:absolute;top:150px;left:48px;z-index:6;display:flex;align-items:center;gap:14px;">
  <div style="width:52px;height:6px;background:{team_color};"></div>
  <div style="font-size:22px;font-weight:800;letter-spacing:0.16em;text-transform:uppercase;color:{INK};">{_esc(tinfo[team]['name'])} <span style="color:#6b6b6b;font-weight:700;">· 2026 Transfer Class</span></div>
  <div style="font-size:14px;font-weight:800;letter-spacing:0.08em;color:#fff;background:{team_color};border-radius:5px;padding:4px 10px;">{len(cls_pids)} COMMITS</div>
</div>
<div style="position:absolute;left:48px;bottom:24px;z-index:6;font-size:15px;font-weight:700;letter-spacing:0.18em;color:#6b6b6b;">64ANALYTICS.COM</div>'''

bg_style = (f'background:#ece7dd url({eggshell_tile()}) repeat;' if texture else 'background:#ece7dd;')
board = (f'<div id="capture" style="position:relative;width:1080px;height:1350px;{bg_style}'
         f"font-family:Inter,system-ui,sans-serif;overflow:hidden;color:{INK};\">"
         f'{accents_svg}<div style="position:absolute;inset:0;z-index:2;">{collage}</div>'
         f'{panel}{header}</div>')

W, H = 1080, 1350
scale = round(660 / W, 4)
map_js = MAP_JS.read_text(encoding='utf-8') if MAP_JS.exists() else ''
fname = f"class_highlight_{tinfo[team]['name'].lower().replace(' ', '_')}_26.png"

html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"/>
<style>{font_face_css()}
*{{box-sizing:border-box;}}
body{{margin:0;background:#0b0c0f;}}
#stagewrap{{width:{int(W*scale)}px;height:{int(H*scale)}px;overflow:hidden;}}
#stageinner{{transform:scale({scale});transform-origin:top left;}}
.btnrow{{margin:14px 0 4px;}}
.btnrow button{{font-family:system-ui,sans-serif;font-size:14px;font-weight:700;color:#fff;background:#b23a48;border:none;border-radius:7px;padding:9px 16px;cursor:pointer;}}
.btnrow button:disabled{{opacity:.6;}}
</style></head><body>
<script>{map_js}</script>
<div id="stagewrap"><div id="stageinner">{board}</div></div>
<div class="btnrow"><button onclick="window.dlPNG(this)">Download PNG</button></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<script>
// html2canvas mangles complex SVG (JS-built map, dashed accents, gradients), so we
// rasterize every <svg> in the board to a same-size <img> right before capture and
// restore afterward — the browser's own SVG renderer does the drawing, not html2canvas.
async function freezeSVGs(root){{
  var svgs = Array.prototype.slice.call(root.querySelectorAll('svg'));
  var restores = [];
  for (var i = 0; i < svgs.length; i++){{
    var s = svgs[i];
    var w = s.clientWidth || s.getBoundingClientRect().width;
    var h = s.clientHeight || s.getBoundingClientRect().height;
    if (!w || !h) continue;
    var clone = s.cloneNode(true);
    clone.setAttribute('width', w); clone.setAttribute('height', h);
    var xml = new XMLSerializer().serializeToString(clone);
    var img = document.createElement('img');
    img.width = w; img.height = h;
    img.style.cssText = s.getAttribute('style') || '';
    await new Promise(function(res){{ img.onload = res; img.onerror = res;
      img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(xml); }});
    s.parentNode.insertBefore(img, s);
    s.style.display = 'none';
    restores.push([s, img]);
  }}
  return function(){{ restores.forEach(function(r){{ r[1].remove(); r[0].style.display = ''; }}); }};
}}
window.dlPNG = async function(btn){{
  var el = document.getElementById('capture'); if(!el) return;
  var inner = document.getElementById('stageinner'), wrap = document.getElementById('stagewrap');
  var pT = inner ? inner.style.transform : '', pW = wrap ? wrap.style.cssText : '';
  var t = btn.textContent; btn.disabled = true; btn.textContent = 'Rendering…';
  if(inner) inner.style.transform = 'none';
  if(wrap){{ wrap.style.width='{W}px'; wrap.style.height='{H}px'; wrap.style.overflow='visible'; }}
  var thaw = null;
  try{{
    if(document.fonts){{ try{{ await document.fonts.load("900 110px 'Inter'"); await document.fonts.load("700 20px 'Inter'"); }}catch(e){{}} if(document.fonts.ready) await document.fonts.ready; }}
    thaw = await freezeSVGs(el);
    var canvas = await html2canvas(el, {{scale:2, useCORS:true, backgroundColor:'#ece7dd'}});
    var a = document.createElement('a'); a.download='{fname}';
    a.href = canvas.toDataURL('image/png'); document.body.appendChild(a); a.click(); document.body.removeChild(a);
  }} finally {{ if(thaw) thaw(); if(inner) inner.style.transform=pT; if(wrap) wrap.style.cssText=pW; btn.disabled=false; btn.textContent = t || 'Download PNG'; }}
}};
</script></body></html>"""

components.html(html, height=int(H * scale) + 80, scrolling=False)
