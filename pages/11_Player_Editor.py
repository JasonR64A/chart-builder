"""
Player Editor — Edit or create player records for cross-referencing with players.csv.

Edit mode: Search by player_id, see current values, edit position/classification/
height/bat/throw/hometown. Changes are tracked in a session queue, never written
to players.csv directly.

Create mode: Enter a new player's details. Gets the next available ID; also
tracked in the session queue.

Exports: Download the queue as a CSV with original values (for edits) + new
values, so a human can cross-reference and manually merge into players.csv.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from io import StringIO

_APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _APP_DIR / 'data'
PLAYERS_PATH = DATA_DIR / 'players.csv'
TEAMS_PATH = DATA_DIR / 'teams.csv'

# Columns that are editable in this tool
EDITABLE_COLS = ['player_name', 'position', 'classification', 'height', 'bat', 'throw', 'hometown']

POSITION_OPTIONS = ['', 'P', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'OF', 'INF', 'DH', 'UT']
CLASS_OPTIONS = ['', 'Fr', 'So', 'Jr', 'Sr', 'Gr', 'HS']
BAT_OPTIONS = ['', 'L', 'R', 'S']
THROW_OPTIONS = ['', 'L', 'R']


@st.cache_data
def load_players():
    df = pd.read_csv(PLAYERS_PATH, low_memory=False, encoding='latin-1')
    # Normalize id to int for lookups
    df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype(int)
    return df


@st.cache_data
def load_teams():
    t = pd.read_csv(TEAMS_PATH, low_memory=False)
    t['id'] = pd.to_numeric(t['id'], errors='coerce').fillna(0).astype(int)
    return t


def team_label(team_id, teams_df):
    if pd.isna(team_id):
        return ''
    try:
        tid = int(team_id)
    except (ValueError, TypeError):
        return str(team_id)
    match = teams_df[teams_df['id'] == tid]
    if len(match) == 0:
        return f'id={tid} (unknown)'
    row = match.iloc[0]
    return f"{row['name']} ({row.get('sport','?')}) — id={tid}"


def next_player_id(players_df):
    return int(players_df['id'].max()) + 1


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='Player Editor', layout='wide')
st.title('Player Editor')
st.caption('Edit or create player records for cross-referencing with players.csv. '
           'Never modifies players.csv directly — download the queue and merge manually.')

players_df = load_players()
teams_df = load_teams()

# Session queue — list of dicts
if 'editor_queue' not in st.session_state:
    st.session_state.editor_queue = []

mode = st.radio('Mode', ['Edit', 'Create'], horizontal=True, key='editor_mode')
st.markdown('---')

# ── EDIT MODE ────────────────────────────────────────────────────────────────
if mode == 'Edit':
    st.markdown('### Edit existing player')

    col_search, col_info = st.columns([1, 2])
    with col_search:
        pid_input = st.number_input('Player ID', min_value=0, step=1, value=0,
                                     key='edit_pid',
                                     help='Search for a player by their id from players.csv')

    target = None
    if pid_input > 0:
        match = players_df[players_df['id'] == int(pid_input)]
        if len(match) == 0:
            with col_info:
                st.error(f'No player found with id {pid_input}')
        else:
            target = match.iloc[0]

    if target is not None:
        with col_info:
            team = team_label(target.get('team_id'), teams_df)
            st.success(f"**{target['player_name']}** — {team}")

        # Show current values + editable inputs side by side
        st.markdown('#### Current values → New values')
        c1, c2 = st.columns(2)
        new_vals = {}
        for col in EDITABLE_COLS:
            cur = target.get(col)
            cur_display = '' if pd.isna(cur) else str(cur)
            with c1:
                st.text_input(f'Current {col}', value=cur_display, disabled=True, key=f'cur_{col}')
            with c2:
                if col == 'position':
                    new_vals[col] = st.selectbox(f'New {col}', POSITION_OPTIONS,
                        index=POSITION_OPTIONS.index(cur_display) if cur_display in POSITION_OPTIONS else 0,
                        key=f'new_{col}')
                elif col == 'classification':
                    new_vals[col] = st.selectbox(f'New {col}', CLASS_OPTIONS,
                        index=CLASS_OPTIONS.index(cur_display) if cur_display in CLASS_OPTIONS else 0,
                        key=f'new_{col}')
                elif col == 'bat':
                    new_vals[col] = st.selectbox(f'New {col}', BAT_OPTIONS,
                        index=BAT_OPTIONS.index(cur_display) if cur_display in BAT_OPTIONS else 0,
                        key=f'new_{col}')
                elif col == 'throw':
                    new_vals[col] = st.selectbox(f'New {col}', THROW_OPTIONS,
                        index=THROW_OPTIONS.index(cur_display) if cur_display in THROW_OPTIONS else 0,
                        key=f'new_{col}')
                elif col == 'height':
                    try:
                        cur_h = float(cur_display) if cur_display else 0.0
                    except ValueError:
                        cur_h = 0.0
                    h = st.number_input(f'New {col} (inches)', min_value=0.0, max_value=100.0,
                                         step=0.5, value=cur_h, key=f'new_{col}')
                    new_vals[col] = h if h > 0 else None
                else:
                    new_vals[col] = st.text_input(f'New {col}', value=cur_display, key=f'new_{col}')

        if st.button('Add to export queue', type='primary', key='edit_add'):
            entry = {
                'action': 'edit',
                'id': int(target['id']),
                'timestamp': datetime.now().isoformat(timespec='seconds'),
            }
            # Capture original + new for each editable col
            for col in EDITABLE_COLS:
                entry[f'original_{col}'] = target.get(col)
                entry[f'new_{col}'] = new_vals.get(col)
            st.session_state.editor_queue.append(entry)
            st.success(f"Added edit for {target['player_name']} (id={target['id']}) to queue.")

# ── CREATE MODE ──────────────────────────────────────────────────────────────
else:
    st.markdown('### Create new player')

    suggested_id = next_player_id(players_df)
    st.info(f'Suggested new player id: **{suggested_id}** (next available after max in players.csv). '
            f'You can override below if you prefer a specific id.')

    c1, c2 = st.columns(2)
    with c1:
        new_id = st.number_input('Player ID', min_value=1, step=1, value=suggested_id,
                                  key='create_pid')
        new_name = st.text_input('Player name (required)', key='create_name')
        new_pos = st.selectbox('Position', POSITION_OPTIONS, key='create_pos')
        new_class = st.selectbox('Classification', CLASS_OPTIONS, key='create_class')
    with c2:
        # Team picker — optional, helps downstream
        team_list = [''] + [team_label(tid, teams_df) for tid in sorted(teams_df['id'].unique())]
        new_team_label = st.selectbox('Team (optional)', team_list, key='create_team')
        new_height = st.number_input('Height (inches)', min_value=0.0, max_value=100.0,
                                      step=0.5, value=0.0, key='create_height')
        new_bat = st.selectbox('Bat', BAT_OPTIONS, key='create_bat')
        new_throw = st.selectbox('Throw', THROW_OPTIONS, key='create_throw')
        new_home = st.text_input('Hometown', key='create_home')

    # Check for ID conflict
    id_conflict = (players_df['id'] == int(new_id)).any()
    if id_conflict:
        existing_name = players_df[players_df['id'] == int(new_id)].iloc[0]['player_name']
        st.warning(f'ID {new_id} already exists in players.csv as "{existing_name}". '
                   'Choose a different ID or use Edit mode.')

    can_add = bool(new_name.strip()) and not id_conflict

    if st.button('Add to export queue', type='primary', disabled=not can_add, key='create_add'):
        # Parse team_id from the label (format: "Name (Sport) — id=123")
        team_id = None
        if new_team_label and 'id=' in new_team_label:
            try:
                team_id = int(new_team_label.rsplit('id=', 1)[1].strip().rstrip(')').strip())
            except (ValueError, IndexError):
                team_id = None
        entry = {
            'action': 'create',
            'id': int(new_id),
            'player_name': new_name.strip(),
            'position': new_pos or None,
            'classification': new_class or None,
            'team_id': team_id,
            'height': new_height if new_height > 0 else None,
            'bat': new_bat or None,
            'throw': new_throw or None,
            'hometown': new_home.strip() or None,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }
        st.session_state.editor_queue.append(entry)
        st.success(f"Added new player {new_name.strip()} (id={new_id}) to queue.")

# ── Queue display + export ───────────────────────────────────────────────────
st.markdown('---')
st.markdown(f'### Export queue ({len(st.session_state.editor_queue)} entries)')

if len(st.session_state.editor_queue) == 0:
    st.caption('No entries yet. Use Edit or Create mode above to add some.')
else:
    q = pd.DataFrame(st.session_state.editor_queue)
    st.dataframe(q, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        # Export as CSV
        csv_buf = StringIO()
        q.to_csv(csv_buf, index=False)
        st.download_button(
            'Download queue CSV',
            data=csv_buf.getvalue(),
            file_name=f'player_editor_queue_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv',
            type='primary',
        )
    with c2:
        if st.button('Clear queue', type='secondary'):
            st.session_state.editor_queue = []
            st.rerun()

st.markdown('---')
st.caption(
    'Queue lives in your browser session only — refresh the page and it\'s gone. '
    'Download before closing. Nothing here writes to players.csv; the CSV is for '
    'manual cross-reference and merge.'
)
