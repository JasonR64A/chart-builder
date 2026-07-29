"""Portal Team Rankings — 1080×1350 share graphic.

Same look as the PBP Analytics "Top 25 Rankings" graphic: it reuses
app_lib.top25_render.build_top25_svg (branded background + 5×5 paper-tile grid
with red rank tabs and team logos). The only difference is the data source —
the transfer-portal team rating (portal_rank_team.csv) instead of the company
team rank — and the tile stat shows the 64A portal rating in place of a W-L record.
JUCO / NAIA / Drafted-Pro buckets are excluded. PNG export via safe_svg2png.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from app_lib.top25_render import (build_top25_svg, fetch_team_logo_b64,
                                  _build_logo_id_map)
from app_lib.safe_render import safe_svg2png

APP_DIR = Path(__file__).resolve().parent.parent
DATA = APP_DIR / 'data'
BUCKETS = {'960', '961', '2269'}   # JUCO, NAIA, Drafted/Pro — not real programs

st.set_page_config(page_title='Portal Team Rankings', layout='wide')


def norm(s):
    s = str(s).strip()
    return s[:-2] if s.endswith('.0') else s


@st.cache_data(show_spinner=False)
def _csv(name):
    return pd.read_csv(DATA / name, dtype=str, encoding='latin-1', keep_default_na=False)


@st.cache_data(show_spinner=False)
def load_top25(sport):
    teams = _csv('teams.csv'); conf = _csv('conferences.csv')
    cdiv = {norm(r['id']): r.get('division', '') for _, r in conf.iterrows()}
    tinfo = {norm(r['id']): {'name': r['name'], 'sport': r['sport'],
                             'div': cdiv.get(norm(r['conference_id']), '')} for _, r in teams.iterrows()}
    prt = _csv('portal_rank_team.csv'); prt = prt[prt['year'].map(norm) == '2026'].copy()
    prt['rating'] = pd.to_numeric(prt['sixty_four_rating_portal_team'], errors='coerce').fillna(0)
    prt['sport'] = prt['team_id'].map(norm).map(lambda t: tinfo.get(t, {}).get('sport', ''))
    prt = prt[(prt['sport'] == sport) & (~prt['team_id'].map(norm).isin(BUCKETS)) & (prt['rating'] > 0)]
    prt = prt.sort_values('rating', ascending=False).head(25).reset_index(drop=True)
    out = []
    for i, r in prt.iterrows():
        tid = norm(r['team_id'])
        out.append({'rank': i + 1, 'tid': tid, 'name': tinfo.get(tid, {}).get('name', ''),
                    'rating': float(r['rating']),
                    'committed': int(float(r['players_committed'] or 0))})
    return out


# ---------------- UI ----------------
st.title('Portal Team Rankings')
st.caption('1080×1350 graphic — same design as the Top 25 Rankings board, sourced from the '
           'transfer-portal team rating (portal_rank_team.csv). Tile stat = 64A portal rating.')

sport = st.sidebar.radio('Sport', ['Baseball', 'Softball'], horizontal=True)
sub_label = st.sidebar.text_input('Subtitle', value='Transfer Portal Recruiting Classes 2026')
div_label = st.sidebar.text_input('Header pill', value='D-I')
tile_stat = st.sidebar.radio('Tile stat', ['Rating', 'Commits', 'None'], index=0, horizontal=True)

teams = load_top25(sport)
if not teams:
    st.warning(f'No {sport} portal team rankings found.'); st.stop()

# Build the payload build_top25_svg expects: rank, name, record, logo_b64.
# The tile's "record" line carries our chosen stat (rating by default).
teams_df = pd.read_csv(DATA / 'teams.csv', low_memory=False)
name_to_logo = _build_logo_id_map(teams_df, sport)
payload = []
for t in teams:
    if tile_stat == 'Rating':
        rec = f"{t['rating']:.2f}"
    elif tile_stat == 'Commits':
        rec = f"{t['committed']} COMMITS"
    else:
        rec = ''
    payload.append({'rank': t['rank'], 'name': t['name'], 'record': rec,
                    'logo_b64': fetch_team_logo_b64(name_to_logo.get(t['name']))})

svg = build_top25_svg(payload, sport.title(), div_label, week_label=sub_label)

# On-page preview (scaled, intrinsic viewBox kept)
display_svg = svg.replace(
    '<svg ',
    '<svg style="width:100%;max-width:540px;height:auto;display:block;margin:0 auto;'
    'border-radius:8px;box-shadow:0 8px 24px rgba(0,0,0,.35);" ', 1)
st.markdown(display_svg, unsafe_allow_html=True)

# PNG export
try:
    png_bytes = safe_svg2png(bytestring=svg.encode('utf-8'), output_width=1080)
    fname = f'portal_team_rankings_{sport.lower()}_{pd.Timestamp.now().strftime("%Y%m%d")}.png'
    st.download_button('⬇️ Download PNG (1080×1350)', data=png_bytes, file_name=fname,
                       mime='image/png', type='primary')
except Exception as e:
    st.caption(f'PNG export unavailable in this environment ({type(e).__name__}: {str(e)[:80]}).')
