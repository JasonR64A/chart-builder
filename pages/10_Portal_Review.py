"""
64 Analytics — Portal Player Review
Review and confirm player ID matches from the NCAA transfer portal scrape.
"""
import streamlit as st
import pandas as pd
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _APP_DIR / 'data'
PORTAL_DIR = DATA_DIR / 'portal'
DECISIONS_PATH = PORTAL_DIR / 'review_decisions.csv'

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
div[data-testid="stForm"] { border: 1px solid #333; padding: 1rem; border-radius: 8px; }
</style>
''', unsafe_allow_html=True)

st.title('Portal Player Review')


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_all_portal_players():
    """Combine matched, needs_review, and unmatched into one unified table."""
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


def load_decisions() -> pd.DataFrame:
    if not DECISIONS_PATH.exists():
        return pd.DataFrame(columns=['ncaa_id', 'action', 'override_id'])
    return pd.read_csv(DECISIONS_PATH, dtype=str).fillna('')


def save_decisions(df: pd.DataFrame):
    PORTAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DECISIONS_PATH, index=False)


# ── Main ─────────────────────────────────────────────────────────────────────
all_players = load_all_portal_players()

if all_players.empty:
    st.warning('No portal data found. Run the portal pipeline first.')
    st.stop()

# Load existing decisions
decisions_df = load_decisions()
decisions_map = {}
for _, r in decisions_df.iterrows():
    decisions_map[r['ncaa_id']] = {'action': r['action'], 'override_id': r['override_id']}

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
n_decided = len(decisions_map)

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

# Build editable form
with st.form('review_form'):
    # Column headers
    hdr = st.columns([2.5, 2, 3, 1, 1.2, 1.5, 1.5])
    hdr[0].markdown('**Portal Name**')
    hdr[1].markdown('**Predicted Match**')
    hdr[2].markdown('**School (Division)**')
    hdr[3].markdown('**Score**')
    hdr[4].markdown('**Status**')
    hdr[5].markdown('**Action**')
    hdr[6].markdown('**Override 64A ID**')

    form_data = []

    for i, (_, row) in enumerate(page_df.iterrows()):
        ncaa_id = row['ncaa_id']
        has_pred = bool(row['predicted_64a_id'])
        existing = decisions_map.get(ncaa_id, {})

        cols = st.columns([2.5, 2, 3, 1, 1.2, 1.5, 1.5])

        # Portal name
        cols[0].write(row['portal_name'])

        # Predicted match
        if has_pred:
            cols[1].write(f"{row['predicted_name']}  \n`{row['predicted_64a_id']}`")
        else:
            cols[1].write('—')

        # School + division
        cols[2].write(f"{row['institution']}  \n(D-{row['division']})")

        # Score
        score = row['match_score']
        cols[3].write(score if score else '—')

        # Source status
        src = row['source']
        if src == 'matched':
            cols[4].markdown(':green[matched]')
        elif src == 'needs_review':
            cols[4].markdown(':orange[review]')
        else:
            cols[4].markdown(':red[unmatched]')

        # Action dropdown
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

        # Override ID
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

            # Infer action from override
            if override and not action:
                action = 'adjust'
            elif action == 'confirm' and not override:
                action = 'confirm'

            if action or override:
                decisions_map[ncaa_id] = {
                    'action': action,
                    'override_id': override,
                }

        # Save to CSV
        rows = [
            {'ncaa_id': k, 'action': v.get('action', ''), 'override_id': v.get('override_id', '')}
            for k, v in decisions_map.items()
        ]
        save_decisions(pd.DataFrame(rows))
        st.success(f'Saved {len(decisions_map)} decisions')
        st.cache_data.clear()

# ── Download ─────────────────────────────────────────────────────────────────
st.divider()
dcol1, dcol2 = st.columns(2)
with dcol1:
    st.download_button(
        '📥 Download All Portal Players (CSV)',
        data=all_players.to_csv(index=False),
        file_name='portal_players_all.csv',
        mime='text/csv',
    )
with dcol2:
    if decisions_map:
        dec_df = pd.DataFrame([
            {'ncaa_id': k, 'action': v.get('action', ''), 'override_id': v.get('override_id', '')}
            for k, v in decisions_map.items()
        ])
        st.download_button(
            '📥 Download Decisions (CSV)',
            data=dec_df.to_csv(index=False),
            file_name='review_decisions.csv',
            mime='text/csv',
        )
