"""
Portal Commitment — record where a portal player committed.

Built for team members to log commitments: search a player by id (or name),
confirm they're in the current portal, pick the school they committed to, and
save. Entries persist in Supabase (`portal_commitment_queue`) so they survive
refreshes and are visible to every reviewer — never writes to
portal_commitments.csv directly.

Apply flow (mirrors Player Editor): save commitments here → download the pending
CSV → merge into portal-pipeline/data/portal_commitments.csv locally (resolving
player_id -> ncaa_id) → run the pipeline → click "Mark all as applied".

Guardrail: only players present in portal_rank_player.csv (the live portal list)
can have a commitment recorded — the Save button is blocked otherwise.
"""
import streamlit as st
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
from io import StringIO

# ── Supabase ─────────────────────────────────────────────────────────────────
SUPABASE_URL = 'https://vfzoroabzmbvwkcyozes.supabase.co'
SUPABASE_ANON_KEY = (
    'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.'
    'eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZmem9yb2Fiem1idndrY3lvemVzIiwi'
    'cm9sZSI6ImFub24iLCJpYXQiOjE2OTQwNDU5NTgsImV4cCI6MjAwOTYyMTk1OH0.'
    'MpzhpgI2fVDC5ucrECl2AuQ9VfT_8aaTmFunthyJAPA'
)
QUEUE_TABLE = 'portal_commitment_queue'

HEADERS = {
    'apikey': SUPABASE_ANON_KEY,
    'Authorization': f'Bearer {SUPABASE_ANON_KEY}',
    'Content-Type': 'application/json',
}

# One-time setup SQL, shown in-page if the table doesn't exist yet.
SETUP_SQL = """create table public.portal_commitment_queue (
  id          bigint generated always as identity primary key,
  player_id   bigint not null,
  action      text default 'commit',
  payload     jsonb,
  created_at  timestamptz default now(),
  applied_at  timestamptz
);
alter table public.portal_commitment_queue enable row level security;
create policy "anon all" on public.portal_commitment_queue
  for all using (true) with check (true);"""


def sb_url(table):
    return f'{SUPABASE_URL}/rest/v1/{table}'


def load_pending_queue():
    """Fetch all unapplied commitment entries. Returns (rows, table_missing)."""
    try:
        resp = requests.get(
            sb_url(QUEUE_TABLE),
            headers=HEADERS,
            params={'select': 'id,player_id,payload,created_at',
                    'applied_at': 'is.null', 'order': 'created_at.desc'},
            timeout=10,
        )
        if resp.status_code == 200:
            flat = []
            for r in resp.json():
                payload = r.get('payload') or {}
                flat.append({'queue_id': r['id'], 'player_id': r['player_id'],
                             'created_at': r.get('created_at', '')[:19], **payload})
            return flat, False
        if resp.status_code == 404:
            return [], True
        st.error(f'Failed to load queue: {resp.status_code} {resp.text[:200]}')
        return [], False
    except Exception as e:
        st.error(f'Supabase load error: {e}')
        return [], False


def load_applied_history(limit=300):
    try:
        resp = requests.get(
            sb_url(QUEUE_TABLE),
            headers=HEADERS,
            params={'select': 'id,player_id,payload,created_at,applied_at',
                    'applied_at': 'not.is.null', 'order': 'applied_at.desc',
                    'limit': str(limit)},
            timeout=10,
        )
        if resp.status_code == 200:
            flat = []
            for r in resp.json():
                payload = r.get('payload') or {}
                flat.append({'queue_id': r['id'], 'player_id': r['player_id'],
                             'created_at': r.get('created_at', '')[:19],
                             'applied_at': r.get('applied_at', '')[:19], **payload})
            return flat
        return []
    except Exception:
        return []


def save_entry(player_id, payload):
    row = {'player_id': int(player_id), 'action': 'commit', 'payload': payload}
    try:
        resp = requests.post(
            sb_url(QUEUE_TABLE),
            headers={**HEADERS, 'Prefer': 'return=minimal'},
            json=[row], timeout=10,
        )
        return resp.status_code in (200, 201, 204)
    except Exception as e:
        st.error(f'Supabase save error: {e}')
        return False


def delete_pending_for_player(player_id):
    """One pending commitment per player — drop any existing before re-saving."""
    try:
        resp = requests.delete(
            sb_url(QUEUE_TABLE),
            headers={**HEADERS, 'Prefer': 'return=minimal'},
            params={'player_id': f'eq.{int(player_id)}', 'applied_at': 'is.null'},
            timeout=10,
        )
        return resp.status_code in (200, 204)
    except Exception as e:
        st.error(f'Supabase delete-pending error: {e}')
        return False


def delete_entry(queue_id):
    try:
        resp = requests.delete(
            sb_url(QUEUE_TABLE),
            headers={**HEADERS, 'Prefer': 'return=minimal'},
            params={'id': f'eq.{queue_id}'}, timeout=10,
        )
        return resp.status_code in (200, 204)
    except Exception as e:
        st.error(f'Supabase delete error: {e}')
        return False


def mark_all_applied(ids):
    if not ids:
        return True
    try:
        resp = requests.patch(
            sb_url(QUEUE_TABLE),
            headers={**HEADERS, 'Prefer': 'return=minimal'},
            params={'id': f'in.({",".join(str(i) for i in ids)})'},
            json={'applied_at': datetime.utcnow().isoformat() + 'Z'}, timeout=15,
        )
        return resp.status_code in (200, 204)
    except Exception as e:
        st.error(f'Supabase mark-applied error: {e}')
        return False


# ── Data ─────────────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _APP_DIR / 'data'
PRP_PATH = DATA_DIR / 'portal_rank_player.csv'      # the live portal list
TEAMS_PATH = DATA_DIR / 'teams.csv'
PLAYERS_PATH = DATA_DIR / 'players.csv'


@st.cache_data
def load_prp():
    df = pd.read_csv(PRP_PATH, low_memory=False)
    df['player_id'] = pd.to_numeric(df['player_id'], errors='coerce').astype('Int64')
    # Most recent year only — the live portal snapshot
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df[df['year'] == df['year'].max()]
    return df


@st.cache_data
def load_teams():
    t = pd.read_csv(TEAMS_PATH, low_memory=False)
    t['id'] = pd.to_numeric(t['id'], errors='coerce').fillna(0).astype(int)
    return t


@st.cache_data
def load_player_names():
    """player_id -> name fallback (PRP already carries a name, this is backup)."""
    try:
        p = pd.read_csv(PLAYERS_PATH, low_memory=False, encoding='latin-1')
        p['id'] = pd.to_numeric(p['id'], errors='coerce').astype('Int64')
        return dict(zip(p['id'], p['player_name']))
    except Exception:
        return {}


def team_name_of(team_id, teams_df):
    try:
        tid = int(team_id)
    except (ValueError, TypeError):
        return ''
    m = teams_df[teams_df['id'] == tid]
    return m.iloc[0]['name'] if len(m) else f'id={tid}'


def sport_of(team_id, teams_df):
    try:
        tid = int(team_id)
    except (ValueError, TypeError):
        return None
    m = teams_df[teams_df['id'] == tid]
    return m.iloc[0].get('sport') if len(m) else None


# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='Portal Commitment', layout='wide')
st.title('Portal Commitment Entry')
st.caption('Record where a portal player committed. Search by player id (or name), '
           'confirm they are in the current portal, choose the school, and save. '
           'Entries queue in Supabase and are visible to every reviewer.')

prp = load_prp()
teams_df = load_teams()
name_fallback = load_player_names()

pending_queue, table_missing = load_pending_queue()

if table_missing:
    st.error('The `portal_commitment_queue` table does not exist yet. Paste this once '
             'into the Supabase SQL editor, then refresh:')
    st.code(SETUP_SQL, language='sql')
    st.stop()

# ── Search ───────────────────────────────────────────────────────────────────
st.markdown('### 1 · Find the player')
sc1, sc2 = st.columns([1, 2])
with sc1:
    pid_input = st.number_input('Player ID', min_value=0, step=1, value=0, key='commit_pid',
                                help='The 64A player_id shown on the portal pages.')
with sc2:
    name_q = st.text_input('…or search by name', key='commit_name_q',
                           placeholder='start typing a name to find the player_id').strip()

# Name search helper — surfaces matching in-portal players + their ids
if name_q and len(name_q) >= 2:
    hits = prp[prp['name'].fillna('').str.contains(name_q, case=False, na=False)]
    if len(hits):
        opts = hits.head(25).apply(
            lambda r: f"{r['name']} — {team_name_of(r['team_id'], teams_df)} "
                      f"(player_id {int(r['player_id'])})", axis=1).tolist()
        st.caption(f'{len(hits)} in-portal match(es) — copy the player_id into the box above:')
        st.write('  \n'.join(f'• {o}' for o in opts[:25]))
    else:
        st.caption('No in-portal players match that name.')

# ── Resolve + guardrail ──────────────────────────────────────────────────────
target = None
if pid_input > 0:
    match = prp[prp['player_id'] == int(pid_input)]
    if len(match) == 0:
        st.error(f'Player {pid_input} is **not in the current portal** '
                 f'(portal_rank_player.csv). A commitment can only be recorded for a '
                 f'player who is in the portal. Double-check the id, or search by name above.')
    else:
        target = match.iloc[0]

if target is not None:
    pname = target.get('name') or name_fallback.get(int(pid_input), '') or f'player {pid_input}'
    cur_team_id = target.get('team_id')
    cur_team = team_name_of(cur_team_id, teams_df)
    psport = sport_of(cur_team_id, teams_df)
    rank = target.get('sixty_four_rating_portal_player')
    already = target.get('new_team_id')

    st.markdown('### 2 · Confirm & record the commitment')
    info = f"**{pname}** — from {cur_team or 'unknown'}"
    if psport:
        info += f"  ·  {psport}"
    if pd.notna(rank):
        info += f"  ·  portal rank #{int(rank)}"
    st.success(info)

    if pd.notna(already) and str(already) not in ('', '0'):
        st.warning(f"⚠️ This player already has a committed team on file "
                   f"(**{team_name_of(already, teams_df)}**). Saving will queue an update.")

    existing = next((e for e in pending_queue
                     if int(e.get('player_id', -1)) == int(pid_input)), None)
    if existing:
        st.info(f"A pending commitment for this player is already queued "
                f"(**{existing.get('committed_team_name','?')}**, by "
                f"{existing.get('entered_by','?')}). Saving again replaces it.")

    # Team picker — same sport as the player when resolvable
    tpick = teams_df.copy()
    if psport:
        tpick = tpick[tpick['sport'] == psport]
    tpick = tpick.sort_values('name')
    team_options = [''] + [f"{r['name']} (id={r['id']})" for _, r in tpick.iterrows()]

    fc1, fc2 = st.columns(2)
    with fc1:
        committed_label = st.selectbox('Committed to (school)', team_options, key='commit_team',
                                       help='The school the player committed to.')
    with fc2:
        entered_by = st.text_input('Your name / initials (required)', key='commit_by',
                                   help='So we know who logged this commitment.').strip()

    notes = st.text_input('Notes (optional)', key='commit_notes',
                          placeholder='source, date heard, etc.').strip()

    committed_team_id, committed_team_name = None, None
    if committed_label and 'id=' in committed_label:
        committed_team_id = int(committed_label.rsplit('id=', 1)[1].rstrip(')').strip())
        committed_team_name = committed_label.rsplit(' (id=', 1)[0]

    # Self-commit guard — committing to their current team is almost always a mistake
    self_commit = (committed_team_id is not None and pd.notna(cur_team_id)
                   and int(committed_team_id) == int(cur_team_id))
    if self_commit:
        st.error('That is the player’s current school — pick the school they '
                 'transferred TO.')

    can_save = bool(committed_team_id) and bool(entered_by) and not self_commit

    if st.button('💾  Save commitment', type='primary', disabled=not can_save, key='commit_save'):
        payload = {
            'player_name': pname,
            'current_team_id': int(cur_team_id) if pd.notna(cur_team_id) else None,
            'current_team_name': cur_team,
            'committed_team_id': committed_team_id,
            'committed_team_name': committed_team_name,
            'portal_rank': int(rank) if pd.notna(rank) else None,
            'entered_by': entered_by,
            'notes': notes or None,
        }
        delete_pending_for_player(int(pid_input))   # enforce one pending per player
        if save_entry(int(pid_input), payload):
            st.success(f'Saved: {pname} → {committed_team_name} (by {entered_by}).')
            st.rerun()
        else:
            st.error('Save failed — see error above.')

# ── Pending queue ────────────────────────────────────────────────────────────
st.markdown('---')
st.markdown(f'### Pending commitments ({len(pending_queue)})')

if not pending_queue:
    st.caption('No pending commitments yet.')
else:
    cols = ['player_id', 'player_name', 'current_team_name', 'committed_team_name',
            'portal_rank', 'entered_by', 'notes', 'created_at', 'queue_id']
    q = pd.DataFrame(pending_queue)
    q = q[[c for c in cols if c in q.columns]]
    st.dataframe(q, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        buf = StringIO(); q.to_csv(buf, index=False)
        st.download_button('Download pending CSV', data=buf.getvalue(),
                           file_name=f'portal_commitments_pending_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                           mime='text/csv', type='primary')
    with c2:
        if st.button('Mark all as applied', type='secondary',
                     help='Click after merging the downloaded CSV into '
                          'portal_commitments.csv and running the pipeline.'):
            if mark_all_applied([e['queue_id'] for e in pending_queue]):
                st.success('Marked applied.'); st.rerun()
            else:
                st.error('Failed to mark applied.')
    with c3:
        del_id = st.number_input('queue_id to remove', min_value=0, step=1, value=0, key='commit_del')
        if del_id and st.button('Remove entry', key='commit_del_btn'):
            if delete_entry(int(del_id)):
                st.success(f'Removed {del_id}.'); st.rerun()

st.markdown('---')
with st.expander('Applied history', expanded=False):
    hist = load_applied_history()
    if not hist:
        st.caption('No applied commitments yet.')
    else:
        hdf = pd.DataFrame(hist)
        st.caption(f'{len(hdf)} applied (most recent first)')
        st.dataframe(hdf, use_container_width=True, hide_index=True)

st.caption('Workflow: save here → download pending CSV → merge into '
           'portal-pipeline/data/portal_commitments.csv (player_id → ncaa_id) → run '
           'pipeline → "Mark all as applied". Only in-portal players can be recorded.')
