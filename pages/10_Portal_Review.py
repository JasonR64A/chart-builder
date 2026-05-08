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


@st.cache_data
def load_all_known_players():
    """Build ncaa_id -> context dict from matched + needs_review + unmatched
    files. Used to render past decisions with full player context (portal_name,
    institution, predicted match, etc.) since decisions only store ncaa_id."""
    known = {}
    for sport in ['baseball', 'softball']:
        for fname in [f'{sport}_matched.csv', f'needs_review_{sport}.csv', f'unmatched_{sport}.csv']:
            f = MATCHED_DIR / fname
            if not f.exists():
                continue
            df = pd.read_csv(f, dtype=str).fillna('')
            for _, r in df.iterrows():
                nid = r.get('ncaa_id', '')
                if not nid:
                    continue
                # Don't overwrite a richer matched row with a sparser unmatched row
                if nid in known and known[nid].get('predicted_name'):
                    continue
                known[nid] = {
                    'sport': sport,
                    'portal_name': r.get('portal_name', ''),
                    'institution': r.get('institution', ''),
                    'division': r.get('division', ''),
                    'predicted_name': r.get('matched_name', ''),
                    'predicted_64a_id': r.get('player_id_64a', ''),
                    'match_score': r.get('match_score', ''),
                    'status': r.get('status', ''),
                }
    return known


@st.cache_data
def load_in_production_pids():
    """Set of player_ids that exist in portal_rank_player.csv year=2026.
    Used for the 'In Production' indicator on past decisions."""
    p = _APP_DIR / 'data' / 'portal_rank_player.csv'
    if not p.exists():
        return set()
    df = pd.read_csv(p, low_memory=False)
    df_2026 = df[df['year'] == 2026]
    return set(pd.to_numeric(df_2026['player_id'], errors='coerce').dropna().astype(int))


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
            options = ['', 'confirm', 'adjust', 'unmatch']
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

# ── Past Decisions (override) ────────────────────────────────────────────────
st.divider()

decided_items = [
    (k, v) for k, v in decisions_map.items()
    if v.get('action') or v.get('override_id')
]

show_past = st.checkbox(
    f'Show past decisions ({len(decided_items)})',
    value=False,
    help='Review or override decisions previously saved to Supabase. '
         'Edits update the decision but do NOT retroactively rewrite '
         'portal_rank_player.csv / player_rank.csv — re-run the portal '
         'pipeline to apply changes.',
)

if show_past and decided_items:
    known = load_all_known_players()
    in_prod = load_in_production_pids()

    st.warning(
        'Edits here update the decision in Supabase. To flow changes through '
        'to **portal_rank_player.csv** and **player_rank.csv**, re-run the '
        'portal pipeline and re-stage the 4 upload CSVs.'
    )

    # Build the past-decisions dataframe
    past_rows = []
    for ncaa_id, dec in decided_items:
        info = known.get(ncaa_id, {})
        action = dec.get('action', '')
        override = dec.get('override_id', '')

        # Resolve the chosen 64A id for the in-production check
        chosen_id_str = override.strip() if override.strip() else info.get('predicted_64a_id', '')
        try:
            chosen_id = int(float(chosen_id_str)) if chosen_id_str else None
        except (ValueError, TypeError):
            chosen_id = None
        if action == 'unmatch':
            in_prod_flag = '—'  # not expected in production
        elif chosen_id is None:
            in_prod_flag = '?'
        else:
            in_prod_flag = '✓' if chosen_id in in_prod else '✗'

        past_rows.append({
            'ncaa_id': ncaa_id,
            'sport': info.get('sport', ''),
            'portal_name': info.get('portal_name', '?'),
            'institution': info.get('institution', '?'),
            'division': info.get('division', ''),
            'predicted_name': info.get('predicted_name', ''),
            'predicted_64a_id': info.get('predicted_64a_id', ''),
            'match_score': info.get('match_score', ''),
            'action': action,
            'override_id': override,
            'in_prod': in_prod_flag,
        })

    past_df = pd.DataFrame(past_rows).sort_values(
        ['in_prod', 'sport', 'portal_name'],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    # Filter row for past decisions
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        past_sport = st.selectbox(
            'Sport (past)', ['All', 'baseball', 'softball'], key='past_sport',
        )
    with pcol2:
        past_action = st.selectbox(
            'Action (past)', ['All', 'confirm', 'adjust', 'unmatch'], key='past_action',
        )
    with pcol3:
        past_search = st.text_input(
            'Search (past)', '', key='past_search',
            placeholder='Name or school...',
        )

    fdf = past_df.copy()
    if past_sport != 'All':
        fdf = fdf[fdf['sport'] == past_sport]
    if past_action != 'All':
        fdf = fdf[fdf['action'] == past_action]
    if past_search:
        q = past_search.lower()
        fdf = fdf[
            fdf['portal_name'].str.lower().str.contains(q, na=False) |
            fdf['predicted_name'].str.lower().str.contains(q, na=False) |
            fdf['institution'].str.lower().str.contains(q, na=False)
        ]
    fdf = fdf.reset_index(drop=True)

    st.markdown(
        f'**{len(fdf)} of {len(past_df)} past decisions** '
        f'· In production: ✓ = chosen 64A id is in `portal_rank_player.csv` 2026 '
        f'· ✗ = decision exists but not in production yet '
        f'· — = unmatched (not expected in production)'
    )

    if not fdf.empty:
        with st.form('past_decisions_form'):
            hdr = st.columns([2.2, 1.2, 2.5, 1, 1.3, 1.3, 0.8])
            hdr[0].markdown('**Portal Name / School**')
            hdr[1].markdown('**NCAA ID**')
            hdr[2].markdown('**Predicted**')
            hdr[3].markdown('**Score**')
            hdr[4].markdown('**Action**')
            hdr[5].markdown('**Override 64A ID**')
            hdr[6].markdown('**Prod**')

            past_form = []
            for row_idx, (_, row) in enumerate(fdf.iterrows()):
                cols = st.columns([2.2, 1.2, 2.5, 1, 1.3, 1.3, 0.8])
                cols[0].write(
                    f"{row['portal_name']}  \n"
                    f"_{row['institution']} (D-{row['division']})_"
                )
                cols[1].write(f"`{row['ncaa_id']}`")
                if row['predicted_64a_id']:
                    cols[2].write(f"{row['predicted_name']}  \n`{row['predicted_64a_id']}`")
                else:
                    cols[2].write('--')
                cols[3].write(row['match_score'] if row['match_score'] else '--')

                opts = ['', 'confirm', 'adjust', 'unmatch']
                cur = row['action'] if row['action'] in opts else ''
                action_new = cols[4].selectbox(
                    'past_act', opts, index=opts.index(cur),
                    key=f'past_act_{row_idx}',
                    label_visibility='collapsed',
                )
                override_new = cols[5].text_input(
                    'past_ovr', value=row['override_id'],
                    key=f'past_ovr_{row_idx}',
                    label_visibility='collapsed',
                    placeholder='64A ID',
                )
                if row['in_prod'] == '✓':
                    cols[6].markdown(':green[✓]')
                elif row['in_prod'] == '✗':
                    cols[6].markdown(':red[✗]')
                else:
                    cols[6].write(row['in_prod'])

                past_form.append({
                    'ncaa_id': row['ncaa_id'],
                    'old_action': row['action'],
                    'old_override': row['override_id'],
                    'new_action': action_new,
                    'new_override': override_new,
                })

            past_submitted = st.form_submit_button(
                'Update Decisions',
                type='primary',
                use_container_width=True,
            )
            if past_submitted:
                changed = 0
                for item in past_form:
                    new_action = item['new_action']
                    new_override = item['new_override'].strip()
                    if new_override and not new_action:
                        new_action = 'adjust'
                    if (new_action != item['old_action']) or (new_override != item['old_override']):
                        decisions_map[item['ncaa_id']] = {
                            'action': new_action,
                            'override_id': new_override,
                        }
                        changed += 1
                if changed:
                    st.session_state.decisions = decisions_map
                    if save_decisions_to_supabase(decisions_map):
                        st.success(
                            f'Updated {changed} decision(s) in Supabase. '
                            f'Re-run the portal pipeline to apply changes to '
                            f'`portal_rank_player.csv` / `player_rank.csv`.'
                        )
                    else:
                        st.error('Failed to save updates to Supabase')
                else:
                    st.info('No changes detected.')

# ── Downloads ────────────────────────────────────────────────────────────────
st.divider()
dcol1, dcol2, dcol3 = st.columns(3)
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
with dcol3:
    # Unmatched export: players the reviewer flagged as false-positive matches.
    # Output columns are what's needed to seed a fresh player upload.
    unmatched_ids = {k for k, v in decisions_map.items() if v.get('action') == 'unmatch'}
    if unmatched_ids:
        um_rows = all_players[all_players['ncaa_id'].isin(unmatched_ids)][
            ['portal_name', 'institution', 'division', 'sport', 'ncaa_id']
        ].drop_duplicates()
        st.download_button(
            f'Download Unmatched ({len(um_rows)}) for Player Upload',
            data=um_rows.to_csv(index=False),
            file_name='portal_unmatched_for_upload.csv',
            mime='text/csv',
        )
