"""
Portal Player Review — Standalone App
Run: streamlit run portal_review.py

Only shows players that need human review (not 100% confident matches).
Decisions persist to Supabase so multiple reviewers see the same state.
"""
import streamlit as st
import pandas as pd
import requests
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
PORTAL_DIR = _APP_DIR / 'data' / 'portal'
MATCHED_DIR = PORTAL_DIR

# ── Supabase ─────────────────────────────────────────────────────────────────
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
}


def sb_url(table):
    return f'{SUPABASE_URL}/rest/v1/{table}'


def load_decisions_from_supabase():
    try:
        resp = requests.get(
            sb_url(DECISIONS_TABLE),
            headers=HEADERS,
            params={'select': 'ncaa_id,action,override_id'},
            timeout=10,
        )
        if resp.status_code == 200:
            return {r['ncaa_id']: {'action': r.get('action', ''), 'override_id': r.get('override_id', '')} for r in resp.json()}
        return {}
    except Exception:
        return {}


def save_decisions_to_supabase(decisions):
    rows = [
        {'ncaa_id': k, 'action': v.get('action', ''), 'override_id': v.get('override_id', '')}
        for k, v in decisions.items()
        if v.get('action') or v.get('override_id')
    ]
    if not rows:
        return True
    try:
        resp = requests.post(
            sb_url(DECISIONS_TABLE),
            headers={**HEADERS, 'Prefer': 'resolution=merge-duplicates,return=minimal'},
            json=rows,
            timeout=15,
        )
        return resp.status_code in (200, 201, 204)
    except Exception:
        return False


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Portal Player Review", layout="wide")
st.title("Portal Player Review")
st.caption("Only showing players that need review (100% confident matches are excluded)")


# ── Load players ─────────────────────────────────────────────────────────────
@st.cache_data
def load_review_players():
    """Load needs_review + unmatched players. Skip 100% confident matches."""
    rows = []

    for sport in ['baseball', 'softball']:
        # Needs review (fuzzy matches below auto-accept threshold)
        for source, filename in [
            ('needs_review', f'needs_review_{sport}.csv'),
            ('unmatched', f'unmatched_{sport}.csv'),
        ]:
            f = MATCHED_DIR / filename
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
                    'confidence': r.get('confidence', 'NONE'),
                    'source': source,
                })

        # From matched: only include those that are NOT 100% score
        f = MATCHED_DIR / f'{sport}_matched.csv'
        if f.exists():
            df = pd.read_csv(f, dtype=str).fillna('')
            for _, r in df.iterrows():
                try:
                    score = float(r.get('match_score', '0'))
                except (ValueError, TypeError):
                    score = 0
                if score < 100:
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
                        'confidence': r.get('confidence', ''),
                        'source': 'matched_low',
                    })

    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────
all_players = load_review_players()

if all_players.empty:
    st.info('No players need review. Run the portal pipeline first, or all matches are 100% confident.')
    st.stop()

if 'decisions' not in st.session_state:
    st.session_state.decisions = load_decisions_from_supabase()

decisions_map = st.session_state.decisions

# ── Filters ──────────────────────────────────────────────────────────────────
fcol1, fcol2, fcol3, fcol4 = st.columns(4)
with fcol1:
    sport_filter = st.selectbox('Sport', ['All', 'baseball', 'softball'])
with fcol2:
    source_filter = st.selectbox('Match Status', ['All', 'needs_review', 'unmatched', 'matched_low'])
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
n_review = len(all_players[all_players['source'] == 'needs_review'])
n_unmatched = len(all_players[all_players['source'] == 'unmatched'])
n_matched_low = len(all_players[all_players['source'] == 'matched_low'])
n_decided = sum(1 for d in decisions_map.values() if d.get('action') or d.get('override_id'))

remaining = len(df)
st.markdown(
    f'**{remaining} remaining** of {len(all_players)} total — '
    f'**{n_decided} decided** · '
    f':orange[{n_review} needs review] · '
    f':red[{n_unmatched} unmatched] · '
    f':blue[{n_matched_low} matched <100%]'
)
if remaining == 0:
    st.success('All players have been reviewed! Re-run the pipeline to generate updated rankings and upload documents.')
    st.stop()

# ── Review table ─────────────────────────────────────────────────────────────
# Filter out already-decided players so reviewers only see what's left
decided_ids = set(k for k, v in decisions_map.items() if v.get('action') or v.get('override_id'))
df = df[~df['ncaa_id'].isin(decided_ids)]

page_df = df.reset_index(drop=True)
page = 1  # no pagination — all players on one page

with st.form('review_form'):
    hdr = st.columns([2.5, 1.2, 2, 3, 1, 1.2, 1.5, 1.5])
    hdr[0].markdown('**Portal Name**')
    hdr[1].markdown('**NCAA ID**')
    hdr[2].markdown('**Predicted Match**')
    hdr[3].markdown('**School (Division)**')
    hdr[4].markdown('**Score**')
    hdr[5].markdown('**Status**')
    hdr[6].markdown('**Action**')
    hdr[7].markdown('**Override 64A ID**')

    form_data = []

    for row_idx, (_, row) in enumerate(page_df.iterrows()):
        ncaa_id = row['ncaa_id']
        has_pred = bool(row['predicted_64a_id'])
        existing = decisions_map.get(ncaa_id, {})

        cols = st.columns([2.5, 1.2, 2, 3, 1, 1.2, 1.5, 1.5])

        cols[0].write(row['portal_name'])

        cols[1].write(f"`{row['ncaa_id']}`")

        if has_pred:
            cols[2].write(f"{row['predicted_name']}  \n`{row['predicted_64a_id']}`")
        else:
            cols[2].write('--')

        cols[3].write(f"{row['institution']}  \n(D-{row['division']})")

        cols[4].write(row['match_score'] if row['match_score'] else '--')

        src = row['source']
        if src == 'matched_low':
            cols[5].markdown(':blue[matched]')
        elif src == 'needs_review':
            cols[5].markdown(':orange[review]')
        else:
            cols[5].markdown(':red[unmatched]')

        # Use row_idx for unique keys (ncaa_id can duplicate across status changes)
        if has_pred:
            options = ['', 'confirm', 'adjust']
            cur = existing.get('action', '')
            idx = options.index(cur) if cur in options else 0
            action = cols[6].selectbox(
                'act', options, index=idx,
                key=f'act_{row_idx}',
                label_visibility='collapsed',
            )
        else:
            action = ''
            cols[6].write('--')

        cur_override = existing.get('override_id', '')
        override = cols[7].text_input(
            'ovr', value=cur_override,
            key=f'ovr_{row_idx}',
            label_visibility='collapsed',
            placeholder='64A ID',
        )

        form_data.append({'ncaa_id': ncaa_id, 'action': action, 'override': override})

    submitted = st.form_submit_button('Save Decisions', type='primary', use_container_width=True)

    if submitted:
        for item in form_data:
            ncaa_id = item['ncaa_id']
            action = item['action']
            override = item['override'].strip()
            if override and not action:
                action = 'adjust'
            if action or override:
                decisions_map[ncaa_id] = {'action': action, 'override_id': override}

        st.session_state.decisions = decisions_map
        if save_decisions_to_supabase(decisions_map):
            st.success(f'Saved {len([d for d in decisions_map.values() if d.get("action") or d.get("override_id")])} decisions to Supabase')
        else:
            st.error('Failed to save to Supabase')

# ── Downloads ────────────────────────────────────────────────────────────────
st.divider()
dcol1, dcol2 = st.columns(2)
with dcol1:
    st.download_button(
        'Download Review Players (CSV)',
        data=all_players.to_csv(index=False),
        file_name='portal_review_players.csv',
        mime='text/csv',
    )
with dcol2:
    if decisions_map:
        dec_rows = [
            {'ncaa_id': k, 'action': v.get('action', ''), 'override_id': v.get('override_id', '')}
            for k, v in decisions_map.items()
            if v.get('action') or v.get('override_id')
        ]
        if dec_rows:
            st.download_button(
                'Download Decisions (CSV)',
                data=pd.DataFrame(dec_rows).to_csv(index=False),
                file_name='review_decisions.csv',
                mime='text/csv',
            )
