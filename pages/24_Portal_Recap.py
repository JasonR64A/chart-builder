"""Portal Recap — 1080×1350 share graphic (draft-recap style).

Three panels off the top-250 transfer-portal players (portal_rank <= 250):
  1. Teams with the most top-250 commits (count tiers + logos)
  2. Donut of where the top 250 went (SEC/ACC/Big 12/Big Ten + Other + Drafted/Pro + Uncommitted)
  3. % committing to a Power 4 by portal-rank tier

Renderer lives in app_lib/portal_recap_render.py; PNG via safe_svg2png.
"""
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

from app_lib.portal_recap_render import build_portal_recap_svg, logo_b64, CONF_COLOR
from app_lib.safe_render import safe_svg2png

APP_DIR = Path(__file__).resolve().parent.parent
DATA = APP_DIR / 'data'
# portal rank = portal_rank_player.csv 'sixty_four_rating_portal_player' (integer 1..N per
# sport, 1=best) — verified identical to the portal-pipeline output/03_ranked portal_rank.
BUCKET_TEAMS = {'2269', '960', '961'}   # Drafted/Pro, JUCO, NAIA — not real programs
P4 = {'SEC', 'ACC', 'Big 12', 'Big Ten'}
PIE_ORDER = ['SEC', 'ACC', 'Big 12', 'Big Ten', 'Other', 'Drafted/Pro', 'Uncommitted']

st.set_page_config(page_title='Portal Recap', layout='wide')


def norm(s):
    s = str(s).strip()
    return s[:-2] if s.endswith('.0') else s


def iv(x):
    try:
        return int(float(x))
    except Exception:
        return 0


@st.cache_data(show_spinner=False)
def _csv(path):
    return pd.read_csv(path, dtype=str, encoding='latin-1', keep_default_na=False)


@st.cache_data(show_spinner=False)
def build_payload(sport, top_n):
    conf = _csv(DATA / 'conferences.csv'); tm = _csv(DATA / 'teams.csv')
    cabbr = {norm(r['id']): (r.get('abbreviation') or r.get('name') or '') for _, r in conf.iterrows()}
    tconf = {norm(r['id']): norm(r['conference_id']) for _, r in tm.iterrows()}
    tsport = {norm(r['id']): r['sport'] for _, r in tm.iterrows()}
    # baseball logo id (softball falls back to same-name baseball id)
    bb = {r['name']: norm(r['id']) for _, r in tm.iterrows() if r['sport'] == 'Baseball'}
    name_of = {norm(r['id']): r['name'] for _, r in tm.iterrows()}

    def logo_id(tid):
        p = APP_DIR / 'team_logos_512' / f'{tid}.png'
        if p.exists():
            return tid
        return bb.get(name_of.get(tid), tid)

    def conf_of(tid):
        return cabbr.get(tconf.get(norm(tid), ''), '')

    # portal rank straight from PRP (per-sport integer rank; 1 = best)
    prp = _csv(DATA / 'portal_rank_player.csv'); prp = prp[prp['year'].map(norm) == '2026'].copy()
    prp['pr'] = prp['sixty_four_rating_portal_player'].map(iv)
    prp = prp[(prp['team_id'].map(lambda t: tsport.get(norm(t), '')) == sport) & (prp['pr'] >= 1)]
    rows = [(norm(r['new_team_id']), r['pr']) for _, r in prp.iterrows()]

    top = [(nt, pr) for nt, pr in rows if pr <= top_n]

    # panel 1 — most commits (count tiers)
    cnt = Counter(nt for nt, _ in top if nt not in ('', *BUCKET_TEAMS))
    bycount = defaultdict(list)
    for tid, c in cnt.items():
        bycount[c].append(tid)
    tiers = [(c, [logo_b64(logo_id(t)) for t in sorted(bycount[c])[:6]])
             for c in sorted(bycount, reverse=True)[:6]]

    # panel 2 — destination donut
    pie_c = Counter()
    for nt, _ in top:
        if nt == '2269':
            pie_c['Drafted/Pro'] += 1
        elif nt == '':
            pie_c['Uncommitted'] += 1
        else:
            cf = conf_of(nt); pie_c[cf if cf in P4 else 'Other'] += 1
    pie = [(k, pie_c[k], CONF_COLOR[k]) for k in PIE_ORDER if pie_c.get(k, 0) > 0]

    # panel 3 — % to P4 by rank tier (committed-only denominator)
    buckets = [(f'Top {top_n}', 1, top_n), ('251-500', 251, 500), ('501-1000', 501, 1000),
               ('1001-2500', 1001, 2500), ('2500+', 2501, 10**9)]
    bars = []
    for lbl, lo, hi in buckets:
        comm = p4 = 0
        for nt, pr in rows:
            if lo <= pr <= hi and nt not in ('', *BUCKET_TEAMS):
                comm += 1
                if conf_of(nt) in P4:
                    p4 += 1
        bars.append((lbl, 100 * p4 / comm if comm else 0))

    return {'subtitle': f'TOP {top_n} TRANSFER RECAP', 'tiers': tiers, 'pie': pie,
            'bars': bars, 'total_top250': sum(1 for _, pr in rows if pr <= top_n)}


# ---------------- UI ----------------
st.title('Portal Recap')
st.caption('Draft-recap-style graphic off the top-250 portal players. Generate → Download PNG.')
sport = st.sidebar.radio('Sport', ['Baseball', 'Softball'], horizontal=True)

payload = build_payload(sport, 250)
if not payload['tiers']:
    st.warning(f'No {sport} portal data found.'); st.stop()

svg = build_portal_recap_svg(payload)
display_svg = svg.replace('<svg ', '<svg style="width:100%;max-width:540px;height:auto;display:block;'
                          'margin:0 auto;border-radius:8px;box-shadow:0 8px 24px rgba(0,0,0,.35);" ', 1)
st.markdown(display_svg, unsafe_allow_html=True)

try:
    png = safe_svg2png(bytestring=svg.encode('utf-8'), output_width=1080)
    st.download_button('⬇️ Download PNG (1080×1350)', data=png,
                       file_name=f'portal_recap_{sport.lower()}.png', mime='image/png', type='primary')
except Exception as e:
    st.caption(f'PNG export unavailable here ({type(e).__name__}: {str(e)[:80]}).')
