"""
64 Analytics — Portal Player Review
Review and confirm player ID matches from the NCAA transfer portal scrape.
Decisions persist to Supabase (portal_review_decisions table) so multiple
reviewers can collaborate and data survives Render redeploys.
"""
import streamlit as st
import pandas as pd
import requests
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _APP_DIR / 'data'
PORTAL_DIR = DATA_DIR / 'portal'

# ── Supabase connection (anon key is public/client-safe) ─────────────────────
SUPABASE_URL = 'https://vfzoroabzmbvwkcyozes.supabase.co'
SUPABASE_ANON_KEY = (
    'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.'
    'eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZmem9yb2Fiem1idndrY3lvemVzIiwi'
    'cm9sZSI6ImFub24iLCJpYXQiOjE2OTQwNDU5NTgsImV4cCI6MjAwOTYyMTk1OH0.'
    'MpzhpgI2fVDC5ucrECl2AuQ9VfT_8aaTmFunthyJAPA'
)
DECISIONS_TABLE = 'portal_review_decisions'

HEADERS = {
    'apikey': SUPABASE_ANON_KEY,
    'Authorization': f'Bearer {SUPABASE_ANON_KEY}',
    'Content-Type': 'application/json',
    'Prefer': 'return=minimal',
}


def sb_url(table: str) -> str:
    return f'{SUPABASE_URL}/rest/v1/{table}'


def load_decisions_from_supabase() -> dict:
    """Load all saved decisions from Supabase."""
    try:
        resp = requests.get(
            sb_url(DECISIONS_TABLE),
            headers={**HEADERS, 'Prefer': ''},
            params={'select': 'ncaa_id,action,override_id'},
            timeout=10,
        )
        if resp.status_code == 200:
            decisions = {}
            for r in resp.json():
                decisions[r['ncaa_id']] = {
                    'action': r.get('action', ''),
                    'override_id': r.get('override_id', ''),
                }
            return decisions
        else:
            st.warning(f'Could not load decisions from Supabase (HTTP {resp.status_code}). Using local session only.')
            return {}
    except Exception as e:
        st.warning(f'Supabase connection failed: {e}. Using local session only.')
        return {}


def save_decisions_to_supabase(decisions: dict) -> bool:
    """Upsert all decisions to Supabase."""
    if not decisions:
        return True
    rows = [
        {
            'ncaa_id': ncaa_id,
            'action': d.get('action', ''),
            'override_id': d.get('override_id', ''),
        }
        for ncaa_id, d in decisions.items()
        if d.get('action') or d.get('override_id')
    ]
    if not rows:
        return True
    try:
        resp = requests.post(
            sb_url(DECISIONS_TABLE),
            headers={
                **HEADERS,
                'Prefer': 'resolution=merge-duplicates,return=minimal',
            },
            json=rows,
            timeout=15,
        )
        return resp.status_code in (200, 201, 204)
    except Exception:
        return False


# ── Page config & styling ────────────────────────────────────────────────────
st.set_page_config(
    page_title='64 Analytics — Portal Review',
    layout='wide',
    initial_sidebar_state='collapsed',
)
st.markdown('''
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, .stApp, .stApp *:not([class*="icon"]):not([class*="Icon"]):not([data-testid*="icon"]):not([data-testid*="Icon"]):not([data-testid*="arrow"]):not(.material-icons):not(.material-symbols):not(.material-symbols-rounded):not([data-testid="stDataFrameSortIcon"]) {
    font-family: 'Inter', sans-serif !important;
}
.stApp { background-color: #1a1a1a; }
h1, h2, h3, p, label, .stMarkdown { color: #C8C8C8 !important; }
</style>
''', unsafe_allow_html=True)

st.title('Portal Player Review')


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_all_portal_players():
    rows = []
    for sport in ['baseball', 'softball']:
        for source, filename in [
            ('matched', f'{sport}_matched.csv'),
            ('needs_review', f'needs_review_{sport}.csv'),
            ('unmatched', f'unmatched_{sport}.csv'),
        ]:
            f = PORTAL_DIR / filename
            if not f.exists():
                continue
            df = pd.read_csv(f, dtype=str).fillna('')
            for _, r in df.iterrows():
                rows.append({
                    'sport': sport,
                    'portal_name': r.get('portal_name', ''),
                    'institution': r.get('institution', ''),
                    'division': r.get('division', ''),
                    'ncaa_id': r.get('ncaa_id', ''),
                    'status': r.get('status', ''),
                    'predicted_name': r.get('matched_name', ''),
                    'predicted_64a_id': r.get('player_id_64a', ''),
                    'match_score': r.get('match_score', ''),
                    'match_method': r.get('match_method', 'none'),
                    'confidence': r.get('confidence', 'NONE'),
                    'source': source,
                })
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────
all_players = load_all_portal_players()

if all_players.empty:
    st.warning('No portal data found. Run the portal pipeline first.')
    st.stop()

# Load decisions from Supabase
if 'decisions' not in st.session_state:
    st.session_state.decisions = load_decisions_from_supabase()

decisions_map = st.session_state.decisions

# ── Filters ──────────────────────────────────────────────────────────────────
fcol1, fcol2, fcol3, fcol4 = st.columns(4)
with fcol1:
    sport_filter = st.selectbox('Sport', ['All', 'baseball', 'softball'])
with fcol2:
    source_filter = st.selectbox('Match Status', ['All', 'needs_review', 'unmatched', 'matched'])
with fcol3:
    div_filter = st.selectbox('Division', ['All', 'I', 'II', 'III'])
with fcol4:
    search = st.text_input('Search', '', placeholder='Name or school...')

df = all_players.copy()
if sport_filter != 'All':
    df = df[df['sport'] == sport_filter]
if source_filter != 'All':
    df = df[df['source'] == source_filter]
if div_filter != 'All':
    df = df[df['division'] == div_filter]
if search:
    q = search.lower()
    df = df[
        df['portal_name'].str.lower().str.contains(q, na=False) |
        df['predicted_name'].str.lower().str.contains(q, na=False) |
        df['institution'].str.lower().str.contains(q, na=False)
    ]

# ── Summary ──────────────────────────────────────────────────────────────────
total = len(all_players)
n_matched = len(all_players[all_players['source'] == 'matched'])
n_review = len(all_players[all_players['source'] == 'needs_review'])
n_unmatched = len(all_players[all_players['source'] == 'unmatched'])
n_decided = sum(1 for d in decisions_map.values() if d.get('action') or d.get('override_id'))

st.markdown(
    f'**{total} total** — '
    f':green[{n_matched} matched] · '
    f':orange[{n_review} needs review] · '
    f':red[{n_unmatched} unmatched] · '
    f'**{n_decided} decisions saved** · '
    f'Showing {len(df)}'
)

# ── Paginated review ─────────────────────────────────────────────────────────
PAGE_SIZE = 20
total_pages = max(1, (len(df) + PAGE_SIZE - 1) // PAGE_SIZE)
page = st.number_input('Page', min_value=1, max_value=total_pages, value=1)
start = (page - 1) * PAGE_SIZE
page_df = df.iloc[start:start + PAGE_SIZE].reset_index(drop=True)

st.caption(f'Rows {start + 1}–{min(start + PAGE_SIZE, len(df))} of {len(df)}')

with st.form('review_form'):
    hdr = st.columns([2.5, 2, 3, 1, 1.2, 1.5, 1.5])
    hdr[0].markdown('**Portal Name**')
    hdr[1].markdown('**Predicted Match**')
    hdr[2].markdown('**School (Division)**')
    hdr[3].markdown('**Score**')
    hdr[4].markdown('**Status**')
    hdr[5].markdown('**Action**')
    hdr[6].markdown('**Override 64A ID**')

    form_data = []

    for _, row in page_df.iterrows():
        ncaa_id = row['ncaa_id']
        has_pred = bool(row['predicted_64a_id'])
        existing = decisions_map.get(ncaa_id, {})

        cols = st.columns([2.5, 2, 3, 1, 1.2, 1.5, 1.5])

        cols[0].write(row['portal_name'])

        if has_pred:
            cols[1].write(f"{row['predicted_name']}  \n`{row['predicted_64a_id']}`")
        else:
            cols[1].write('—')

        cols[2].write(f"{row['institution']}  \n(D-{row['division']})")

        cols[3].write(row['match_score'] if row['match_score'] else '—')

        src = row['source']
        if src == 'matched':
            cols[4].markdown(':green[matched]')
        elif src == 'needs_review':
            cols[4].markdown(':orange[review]')
        else:
            cols[4].markdown(':red[unmatched]')

        if has_pred:
            options = ['', 'confirm', 'adjust']
            cur = existing.get('action', '')
            idx = options.index(cur) if cur in options else 0
            action = cols[5].selectbox(
                'act', options, index=idx,
                key=f'act_{ncaa_id}_{page}',
                label_visibility='collapsed',
            )
        else:
            action = ''
            cols[5].write('—')

        cur_override = existing.get('override_id', '')
        override = cols[6].text_input(
            'ovr', value=cur_override,
            key=f'ovr_{ncaa_id}_{page}',
            label_visibility='collapsed',
            placeholder='64A ID',
        )

        form_data.append({
            'ncaa_id': ncaa_id,
            'action': action,
            'override': override,
            'has_pred': has_pred,
        })

    submitted = st.form_submit_button('Save Decisions', type='primary', use_container_width=True)

    if submitted:
        for item in form_data:
            ncaa_id = item['ncaa_id']
            action = item['action']
            override = item['override'].strip()

            if override and not action:
                action = 'adjust'

            if action or override:
                decisions_map[ncaa_id] = {
                    'action': action,
                    'override_id': override,
                }

        st.session_state.decisions = decisions_map
        ok = save_decisions_to_supabase(decisions_map)
        if ok:
            st.success(f'Saved {n_decided} decisions to Supabase')
        else:
            st.error('Failed to save to Supabase. Decisions are in your session only.')

# ── Download ─────────────────────────────────────────────────────────────────
st.divider()
dcol1, dcol2 = st.columns(2)
with dcol1:
    st.download_button(
        'Download All Portal Players (CSV)',
        data=all_players.to_csv(index=False),
        file_name='portal_players_all.csv',
        mime='text/csv',
    )
with dcol2:
    if decisions_map:
        dec_rows = [
            {'ncaa_id': k, 'action': v.get('action', ''), 'override_id': v.get('override_id', '')}
            for k, v in decisions_map.items()
            if v.get('action') or v.get('override_id')
        ]
        st.download_button(
            'Download Decisions (CSV)',
            data=pd.DataFrame(dec_rows).to_csv(index=False),
            file_name='review_decisions.csv',
            mime='text/csv',
        )
