"""Four IG-format (1080x1350) draft infographics -> Desktop/draft_infographics/."""
import re, json, html, requests, subprocess, tempfile
import pandas as pd
from pathlib import Path

EDGE = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'

CB = Path('C:/Dev/chart-builder-app')
OUT = Path('C:/Users/sixty/OneDrive/Desktop/draft_infographics')
OUT.mkdir(exist_ok=True)
W, H = 1080, 1350

# ---- palette (validated, dark mode) ----
SURF, PLANE = '#1a1a19', '#0d0d0d'
INK, INK2, MUTED = '#ffffff', '#c3c2b7', '#898781'
GRID, BASE = '#2c2c2a', '#383835'
ACCENT = '#C41230'                       # 64A brand chrome (not a series color)
C_4YR, C_HS, C_JC = '#3987e5', '#199e70', '#c98500'   # categorical slots 1-3
D_POS, D_NEG = '#3987e5', '#e66767'                    # diverging pair
FONT = 'Segoe UI, Arial, sans-serif'
esc = html.escape

def header(title, sub):
    return (f'<rect width="{W}" height="{H}" fill="{SURF}"/>'
            f'<rect x="60" y="64" width="8" height="76" rx="3" fill="{ACCENT}"/>'
            f'<text x="92" y="98" font-family="{FONT}" font-size="44" font-weight="700" fill="{INK}">{esc(title)}</text>'
            f'<text x="92" y="136" font-family="{FONT}" font-size="23" fill="{INK2}">{esc(sub)}</text>')

def footer(note):
    return (f'<line x1="60" y1="{H-84}" x2="{W-60}" y2="{H-84}" stroke="{GRID}" stroke-width="1"/>'
            f'<text x="60" y="{H-50}" font-family="{FONT}" font-size="19" fill="{MUTED}">{esc(note)}</text>')

    # ---- mini renderer: draws the rect/text/line subset of SVG via PIL ----
from PIL import Image, ImageDraw, ImageFont

FONTS = {400: 'C:/Windows/Fonts/segoeui.ttf', 600: 'C:/Windows/Fonts/seguisb.ttf',
         700: 'C:/Windows/Fonts/segoeuib.ttf'}
_fcache = {}
def _font(size, weight):
    w = 700 if weight >= 700 else (600 if weight >= 600 else 400)
    k = (w, size)
    if k not in _fcache:
        _fcache[k] = ImageFont.truetype(FONTS[w], size)
    return _fcache[k]

def _color(s, default=(0, 0, 0, 255)):
    s = (s or '').strip()
    if not s or s == 'none': return None
    m = re.match(r'rgba\(([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)', s)
    if m:
        r, g, bl, a = m.groups()
        return (int(float(r)), int(float(g)), int(float(bl)), int(float(a) * 255))
    if s.startswith('#'):
        s = s[1:]
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), 255)
    return default

TAG = re.compile(r'<(rect|line|text)\b([^>]*?)(?:/>|>(.*?)</text>)', re.S)
ATTR = re.compile(r'([\w-]+)="([^"]*)"')

def save(name, body):
    img = Image.new('RGB', (W, H), SURF)
    dr = ImageDraw.Draw(img, 'RGBA')
    for m in TAG.finditer(body):
        tag, attrs_s, inner = m.group(1), m.group(2), m.group(3)
        a = dict(ATTR.findall(attrs_s))
        f = lambda k, d=0.0: float(a.get(k, d))
        if tag == 'rect':
            x, y, w_, h_ = f('x'), f('y'), f('width'), f('height')
            fill, stroke = _color(a.get('fill', '#000')), _color(a.get('stroke'))
            r = f('rx')
            dr.rounded_rectangle([x, y, x + w_, y + h_], radius=r, fill=fill,
                                 outline=stroke, width=int(f('stroke-width', 1)) if stroke else 0)
        elif tag == 'line':
            dr.line([f('x1'), f('y1'), f('x2'), f('y2')], fill=_color(a.get('stroke', '#000')),
                    width=max(1, int(f('stroke-width', 1))))
        elif tag == 'text':
            txt = html.unescape(re.sub(r'<[^>]+>', '', inner or ''))
            size = int(f('font-size', 16)); weight = int(f('font-weight', 400))
            fnt = _font(size, weight)
            x, y = f('x'), f('y')
            anchor = a.get('text-anchor', 'start')
            tw = dr.textlength(txt, font=fnt)
            if anchor == 'middle': x -= tw / 2
            elif anchor == 'end': x -= tw
            dr.text((x, y - size * 0.85), txt, font=fnt, fill=_color(a.get('fill', '#000')))
    img.save(OUT / name)
    print('wrote', OUT / name)

# ════════ data ════════
src = (CB / 'pages/21_Draft_Assistant.py').read_text(encoding='utf-8')
key = ''.join(re.findall(r"'([^']*)'", re.search(r"SUPABASE_ANON_KEY = \((.*?)\)", src, re.S).group(1)))
HDR = {'apikey': key, 'Authorization': f'Bearer {key}'}
picks = pd.DataFrame(requests.get('https://vfzoroabzmbvwkcyozes.supabase.co/rest/v1/draft_picks', headers=HDR,
                     params={'select': '*', 'year': 'eq.2026', 'order': 'pick.asc'}, timeout=20).json())
MAXP = int(picks['pick'].max())
feed = requests.get('https://statsapi.mlb.com/api/v1/draft/2026', timeout=20).json()
mlb = {int(p['pickNumber']): p for rd in feed['drafts']['rounds'] for p in rd.get('picks', []) if p.get('isDrafted')}
slots = pd.read_csv(CB / 'data/draft/draft_slots_2026.csv', dtype=str)
rbp, cur = {}, 1
for _, r in slots.sort_values(by='pick', key=lambda s: pd.to_numeric(s, errors='coerce')).iterrows():
    if str(r['round']).replace('.', '').isdigit():
        cur = int(float(r['round']))
    rbp[int(float(r['pick']))] = cur
R10_MAX = max(p for p, r in rbp.items() if r <= 10)
history = pd.read_csv(CB / 'data/draft/draft_history.csv', dtype=str, keep_default_na=False)\
            .drop_duplicates(subset=['year', 'pick', 'player'])
history['pick_n'] = pd.to_numeric(history['pick'], errors='coerce')

# ════════ 1) classification breakdown vs prior drafts ════════
hw = history[history.pick_n <= MAXP]
rows = []
for yr, g in hw.groupby('year'):
    cls = g['classification'].fillna('')
    rows.append({'yr': yr, 'HS': int((g['hs'] == 'True').sum()),
                 '4YR': int(cls.str.startswith('4YR').sum()), 'JC': int((g['juco'] == 'Yes').sum())})
live = {'yr': '2026', 'HS': 0, '4YR': 0, 'JC': 0}
for p in mlb.values():
    c = (p.get('school', {}) or {}).get('schoolClass', '')
    if c.startswith('HS'): live['HS'] += 1
    elif c.startswith('4YR'): live['4YR'] += 1
    elif c.startswith('JC'): live['JC'] += 1
rows.append(live)

b = header('Where the picks come from', f'HS vs college vs JuCo through pick {MAXP} - 2026 draft vs the last five')
b += ''.join(f'<rect x="{x}" y="176" width="26" height="26" rx="5" fill="{c}"/>'
             f'<text x="{x+36}" y="197" font-family="{FONT}" font-size="24" fill="{INK2}">{t}</text>'
             for x, c, t in [(92, C_4YR, '4-year college'), (330, C_HS, 'High school'), (540, C_JC, 'JuCo')])
y0, bh, gap, x0, bw = 250, 96, 52, 200, W - 200 - 80
total_max = max(r['HS'] + r['4YR'] + r['JC'] for r in rows)
for i, r in enumerate(rows):
    y = y0 + i * (bh + gap)
    is_live = r['yr'] == '2026'
    b += (f'<text x="{x0-24}" y="{y+bh/2+10}" text-anchor="end" font-family="{FONT}" font-size="30" '
          f'font-weight="{700 if is_live else 400}" fill="{INK if is_live else INK2}">{r["yr"]}{" LIVE" if is_live else ""}</text>')
    x = x0
    for key_, col in (('4YR', C_4YR), ('HS', C_HS), ('JC', C_JC)):
        wpx = bw * r[key_] / total_max
        if wpx <= 0: continue
        b += f'<rect x="{x:.1f}" y="{y}" width="{max(wpx-2,1):.1f}" height="{bh}" rx="4" fill="{col}"/>'
        if wpx > 54:
            b += (f'<text x="{x+wpx/2-1:.1f}" y="{y+bh/2+10}" text-anchor="middle" font-family="{FONT}" '
                  f'font-size="27" font-weight="600" fill="{SURF}">{r[key_]}</text>')
        x += wpx
    if is_live:
        b += f'<rect x="{x0-2}" y="{y-2}" width="{bw*(r["HS"]+r["4YR"]+r["JC"])/total_max+4:.1f}" height="{bh+4}" rx="6" fill="none" stroke="rgba(255,255,255,0.35)" stroke-width="2"/>'
b += footer(f'Counts through pick {MAXP} of each draft. 2026 is live and still moving.')
save('1_classification_vs_history.png', b)

# ════════ 2) bonus pool usage by team ════════
master = pd.read_csv(CB / 'data/draft/draft_master.csv', dtype=str, keep_default_na=False)
trends = json.loads((CB / 'data/draft/draft_trends.json').read_text())
rank_cols = ['rank_ba', 'rank_mlb', 'rank_espn', 'rank_fss', 'pg_rank']
active = [(pd.to_numeric(master[c], errors='coerce'), int(pd.to_numeric(master[c], errors='coerce').max()) + 1)
          for c in rank_cols if pd.to_numeric(master[c], errors='coerce').notna().any()]
tot = sum(v.fillna(pn) for v, pn in active)
ranked_any = pd.concat([v for v, _ in active], axis=1).notna().any(axis=1)
master['ovr'] = (tot / len(active)).where(ranked_any)
master = master.sort_values('ovr', na_position='last').reset_index(drop=True)
master['num'] = pd.Series(master.index + 1, dtype='Int64').where(master['ovr'].notna())
N = int(master['num'].max()); UNRANKED = N + 1
minfo = {n: (int(r) if pd.notna(r) else None, d) for n, r, d in zip(master['name'], master['num'], master['dob'])}

import sys
sys.path.insert(0, str(CB))
from app_lib import draft_engine as E
from datetime import date
prow = []
for _, p in picks.iterrows():
    slot = float(p['slot_value'] or 0)
    if not slot: continue
    rk, dob = minfo.get(p['player_name'], (None, None))
    if rk is None: rk = UNRANKED
    if not dob: dob = (mlb.get(int(p['pick']), {}).get('person', {}) or {}).get('birthDate')
    age = E.age_years(dob, date(2026, 7, 11)) if dob else None
    c_r = E.code_round(int(p['pick']), int(p['round'] or 1))
    c_a = E.code_age(age) if age is not None else ''
    c_d = E.code_dist(rk - int(p['pick']))
    exp = E.expected_signing(slot, c_d, c_r, c_a, trends) if (c_d and c_a) else slot
    prow.append({'team': p['team'], 'slot': slot, 'exp': exp})
pool = pd.DataFrame(prow).groupby('team').agg(slot=('slot', 'sum'), exp=('exp', 'sum')).reset_index()
pool['avail'] = (pool['slot'] - pool['exp']) / 1e6
pool = pool.sort_values('avail', ascending=False).reset_index(drop=True)
SHORT = {'Arizona Diamondbacks': 'D-backs', 'San Francisco Giants': 'Giants', 'Philadelphia Phillies': 'Phillies',
         'Los Angeles Dodgers': 'Dodgers', 'Los Angeles Angels': 'Angels', 'Chicago White Sox': 'White Sox',
         'Kansas City Royals': 'Royals', 'St. Louis Cardinals': 'Cardinals', 'Pittsburgh Pirates': 'Pirates',
         'Washington Nationals': 'Nationals', 'Baltimore Orioles': 'Orioles', 'Cleveland Guardians': 'Guardians',
         'Milwaukee Brewers': 'Brewers', 'Minnesota Twins': 'Twins', 'Toronto Blue Jays': 'Blue Jays',
         'Colorado Rockies': 'Rockies', 'Cincinnati Reds': 'Reds', 'Seattle Mariners': 'Mariners',
         'Houston Astros': 'Astros', 'Detroit Tigers': 'Tigers', 'Boston Red Sox': 'Red Sox',
         'New York Yankees': 'Yankees', 'New York Mets': 'Mets', 'Chicago Cubs': 'Cubs',
         'Tampa Bay Rays': 'Rays', 'San Diego Padres': 'Padres', 'Atlanta Braves': 'Braves',
         'Texas Rangers': 'Rangers', 'Miami Marlins': 'Marlins', 'Athletics': 'Athletics'}
b = header('Theoretical pool money', f'Sum of slot values minus model-expected signings, through pick {MAXP}')
y0, rh = 208, 32
vmax = max(abs(pool['avail'].min()), abs(pool['avail'].max()))
cx = 560; span = 330
b += f'<line x1="{cx}" y1="{y0-14}" x2="{cx}" y2="{y0+30*rh+6}" stroke="{BASE}" stroke-width="2"/>'
for i, r in pool.iterrows():
    y = y0 + i * rh
    wpx = abs(r['avail']) / vmax * span
    col = D_POS if r['avail'] >= 0 else D_NEG
    x = cx if r['avail'] >= 0 else cx - wpx
    b += (f'<text x="{cx-span-24}" y="{y+16}" text-anchor="end" font-family="{FONT}" font-size="21" fill="{INK2}">{esc(SHORT.get(r["team"], r["team"]))}</text>'
          f'<rect x="{x:.1f}" y="{y}" width="{max(wpx,2):.1f}" height="{rh-11}" rx="4" fill="{col}"/>')
    if r['avail'] < 0 and wpx >= 110:
        # long negative bar: label inside so it can't collide with the team name
        b += (f'<text x="{cx-wpx+10:.1f}" y="{y+16}" font-family="{FONT}" font-size="20" '
              f'font-weight="600" fill="{SURF}">{r["avail"]:+.2f}M</text>')
    else:
        lx = cx + wpx + 12 if r['avail'] >= 0 else cx - wpx - 12
        anch = 'start' if r['avail'] >= 0 else 'end'
        b += (f'<text x="{lx:.1f}" y="{y+16}" text-anchor="{anch}" font-family="{FONT}" font-size="20" '
              f'font-weight="600" fill="{INK}">{r["avail"]:+.2f}M</text>')
b += (f'<text x="{cx+span}" y="{y0-24}" text-anchor="end" font-family="{FONT}" font-size="20" fill="{MUTED}">money freed &#8594;</text>'
      f'<text x="{cx-span}" y="{y0-24}" font-family="{FONT}" font-size="20" fill="{MUTED}">&#8592; over-committed</text>')
b += footer('Expected signings from 2021-25 signing trends by age, pick and rank-vs-pick profile.')
save('2_pool_money_by_team.png', b)

# ════════ 3) portal commits still in the MLB draft pool ════════
pros, off = [], 0
while True:
    d = requests.get(f'https://statsapi.mlb.com/api/v1/draft/prospects/2026?limit=1000&offset={off}', timeout=30).json()
    batch = d.get('prospects', [])
    pros += batch
    if len(batch) < 1000: break
    off += 1000
azs = lambda s: re.sub(r'[^a-z]', '', str(s).lower())
pool_names = {azs(p.get('person', {}).get('fullName', '')) for p in pros}
drafted_norm = {azs(n) for n in picks['player_name']}

norm = lambda s: str(s).strip()[:-2] if str(s).strip().endswith('.0') else str(s).strip()
prp = pd.read_csv('C:/Users/sixty/Dev/portal-pipeline/output/04_upload/portal_rank_player.csv')
ranked = pd.read_csv('C:/Users/sixty/Dev/portal-pipeline/output/03_ranked/baseball_ranked.csv')
prank = dict(zip(ranked.player_id_64a, ranked.portal_rank))
players_all = pd.read_csv(CB / 'data/players.csv', encoding='latin-1', low_memory=False)
pname = dict(zip(players_all.id, players_all.player_name))
ppos = dict(zip(players_all.id, players_all.position))
teams_df = pd.read_csv(CB / 'data/teams.csv', dtype=str, keep_default_na=False, usecols=['id', 'name'])
tname = dict(zip(teams_df['id'].map(norm), teams_df['name']))
p26c = prp[(prp.year == 2026) & prp.new_team_id.notna() & (prp.new_team_id != 2269)]
pool_rows = []
for r in p26c.itertuples():
    pid = int(r.player_id)
    nm = pname.get(pid, '')
    if azs(nm) in drafted_norm or azs(nm) not in pool_names: continue
    rk = prank.get(pid)
    pool_rows.append({'rank': rk if rk and rk > 0 else None, 'name': nm, 'pos': ppos.get(pid, ''),
                      'fr': tname.get(norm(r.team_id), ''), 'to': tname.get(norm(r.new_team_id), '')})
n_total = len(pool_rows)
top = sorted([r for r in pool_rows if r['rank']], key=lambda r: r['rank'])[:20]
b = header('Commits MLB could still take', f'{n_total} committed 2026 portal players sit in the MLB.com draft pool - top 20')
y0, rh = 210, 52
for i, r in enumerate(top):
    y = y0 + i * rh
    b += (f'<text x="118" y="{y+32}" text-anchor="middle" font-family="{FONT}" font-size="27" font-weight="700" fill="{ACCENT}">{int(r["rank"])}</text>'
          f'<text x="170" y="{y+32}" font-family="{FONT}" font-size="26" font-weight="600" fill="{INK}">{esc(r["name"])}</text>')
    move = esc(r["fr"]) + ' &#8594; ' + esc(r["to"])
    b += f'<text x="{W-72}" y="{y+32}" text-anchor="end" font-family="{FONT}" font-size="21" fill="{INK2}">{esc(r["pos"])} &#183; {move}</text>'
    if i < len(top) - 1:
        b += f'<line x1="72" y1="{y+rh-8}" x2="{W-72}" y2="{y+rh-8}" stroke="{GRID}" stroke-width="1"/>'
b += footer(f'Red number = 64A portal rank. Undrafted through pick {MAXP}. All have signed college commitments.')
save('3_portal_commits_in_draft_pool.png', b)

# ════════ 4) 2025 portal players in the first 10 rounds ════════
norm = lambda s: str(s).strip()[:-2] if str(s).strip().endswith('.0') else str(s).strip()
name2pid = {r['name']: norm(r['player_id_64a']) for _, r in master.iterrows() if r['player_id_64a']}
prp = pd.read_csv(CB / 'data/portal_rank_player.csv', dtype=str, keep_default_na=False,
                  usecols=['player_id', 'team_id', 'new_team_id', 'year'])
teams = pd.read_csv(CB / 'data/teams.csv', dtype=str, keep_default_na=False, usecols=['id', 'name'])
tname = dict(zip(teams['id'].map(norm), teams['name']))
p25 = {}
for _, r in prp.iterrows():
    if norm(r['year']) == '2025':
        p25[norm(r['player_id'])] = (tname.get(norm(r['team_id']), ''),
                                     tname.get(norm(r['new_team_id']), '') if r['new_team_id'].strip() else '')
players_sl = pd.read_csv(CB / 'data/players.csv', dtype=str, encoding='latin-1', keep_default_na=False,
                         usecols=['id', 'player_name', 'team_id'])
players_sl['team_name'] = players_sl['team_id'].map(tname)
def fb_pid(pk, nm):
    p = mlb.get(int(pk))
    if not p: return ''
    sch = (p.get('school') or {}).get('name', ''); cls = (p.get('school') or {}).get('schoolClass', '')
    if not cls.startswith('4YR') or not sch: return ''
    parts = nm.split()
    if len(parts) < 2: return ''
    fs, ls = azs(parts[0]), azs(parts[-1])
    c = players_sl[players_sl.player_name.map(lambda n: azs(str(n).split()[-1]) == ls if str(n).split() else False)]
    c = c[c.team_name.fillna('').map(azs) == azs(sch)]
    c = c[c.player_name.map(lambda n: azs(str(n).split()[0]).startswith(fs[:2]) or fs.startswith(azs(str(n).split()[0])[:2]))]
    return norm(c['id'].iloc[0]) if len(c) == 1 else ''
plist = []
for _, p in picks.iterrows():
    if int(p['pick']) > R10_MAX: continue
    pid = norm(name2pid.get(p['player_name'], '')) or fb_pid(p['pick'], p['player_name'])
    if pid and pid in p25:
        fr, to = p25[pid]
        plist.append({'pick': int(p['pick']), 'name': p['player_name'], 'fr': fr, 'to': to})
plist.sort(key=lambda r: r['pick'])
b = header('The portal owns the draft', f'{len(plist)} players from the 2025 transfer portal drafted in rounds 1-10')
import math
col_x = [72, 560]; y0 = 210
percol = math.ceil(len(plist) / 2)
rh = min(38, int((H - 84 - 30 - y0) / percol))
for i, r in enumerate(plist):
    cx0 = col_x[i // percol]; y = y0 + (i % percol) * rh
    b += (f'<text x="{cx0}" y="{y}" font-family="{FONT}" font-size="20" font-weight="700" fill="{ACCENT}">{r["pick"]}</text>'
          f'<text x="{cx0+58}" y="{y}" font-family="{FONT}" font-size="21" font-weight="600" fill="{INK}">{esc(r["name"])}</text>'
          f'<text x="{cx0+58}" y="{y+17}" font-family="{FONT}" font-size="15" fill="{MUTED}">{esc(r["fr"])} &#8594; {esc(r["to"] or "?")}</text>')
b += footer('School change happened via the 2025 transfer portal cycle. Picks 1-' + str(R10_MAX) + '.')
save('4_portal_players_r1_10.png', b)
print('ALL DONE')
