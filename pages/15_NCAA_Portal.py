"""
NCAA Transfer Portal — Browse all portal entries across years.
Formatted to match the NCAA portal display: searchable, filterable table
with Year, Name, Institution, Division, Conference, Status, Transfer Date.
"""
import streamlit as st
import pandas as pd
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = _APP_DIR / 'data' / 'portal_archive'
PORTAL_DIR = _APP_DIR / 'data' / 'portal'
PLAYERS_CSV = _APP_DIR / 'data' / 'players.csv'

st.set_page_config(page_title='NCAA Transfer Portal', layout='wide')

# ── Header ───────────────────────────────────────────────────────────────────
st.title('NCAA Transfer Portal')
st.caption('Search and filter all transfer portal entries across divisions and years.')


@st.cache_data
def load_portal_data():
    frames = []
    for sport, f in [('Baseball', 'baseball_full_archive.csv'),
                      ('Softball', 'softball_full_archive.csv')]:
        path = ARCHIVE_DIR / f
        if path.exists():
            df = pd.read_csv(path, dtype=str).fillna('')
            df['sport_label'] = sport
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data
def load_enrichment():
    """Build ncaa_id -> 64A player info from matched.csv + players.csv.

    Joins current-year matched files (the only ones that carry the
    ncaa_id <-> player_id_64a bridge) with the canonical players.csv so we
    can display position / classification / height / bat / throw / hometown
    for portal entries that have been matched to a 64A player record.
    """
    # Load matched bridges (current year only — historical years aren't preserved)
    matched_frames = []
    for f in ['baseball_matched.csv', 'softball_matched.csv']:
        p = PORTAL_DIR / f
        if p.exists():
            mdf = pd.read_csv(p, dtype=str).fillna('')
            matched_frames.append(mdf[['ncaa_id', 'player_id_64a']])
    if not matched_frames:
        return pd.DataFrame(columns=['ncaa_id', 'player_id_64a', 'position',
                                      'classification', 'height', 'bat',
                                      'throw', 'hometown'])
    matched = pd.concat(matched_frames, ignore_index=True)
    matched = matched[matched['player_id_64a'] != ''].drop_duplicates('ncaa_id')

    # Load players.csv (cp1252 because some hometown entries have extended chars)
    if not PLAYERS_CSV.exists():
        return matched
    try:
        players = pd.read_csv(PLAYERS_CSV, dtype=str, encoding='utf-8').fillna('')
    except UnicodeDecodeError:
        players = pd.read_csv(PLAYERS_CSV, dtype=str, encoding='cp1252').fillna('')
    keep = ['id', 'position', 'classification', 'height', 'bat', 'throw', 'hometown']
    players = players[[c for c in keep if c in players.columns]]
    players = players.rename(columns={'id': 'player_id_64a'})

    return matched.merge(players, on='player_id_64a', how='left').fillna('')


df = load_portal_data()
enrichment = load_enrichment()
if not enrichment.empty:
    df = df.merge(enrichment, on='ncaa_id', how='left').fillna('')
if df.empty:
    st.warning('No portal data found.')
    st.stop()

# ── Filters (NCAA-style top bar) ─────────────────────────────────────────────
f1, f2, f3, f4, f5, f6 = st.columns(6)

with f1:
    sport_opts = ['All'] + sorted(df['sport_label'].dropna().unique().tolist())
    sel_sport = st.selectbox('Sport', sport_opts, key='portal_sport')

with f2:
    div_opts = ['All'] + sorted(df['division'].dropna().unique().tolist())
    sel_div = st.selectbox('Division', div_opts, key='portal_div')

with f3:
    year_opts = ['All'] + sorted(df['year'].dropna().unique().tolist(), reverse=True)
    sel_year = st.selectbox('Academic Year', year_opts, key='portal_year')

with f4:
    status_opts = ['All'] + sorted(df['status'].dropna().unique().tolist())
    sel_status = st.selectbox('Status', status_opts, key='portal_status')

with f5:
    match_opts = ['All', 'Matched to 64A', 'Unmatched']
    sel_match = st.selectbox('64A Match', match_opts, key='portal_match')

with f6:
    search = st.text_input('Search', '', placeholder='Name, school, conference...',
                            key='portal_search')

# Apply filters
filtered = df.copy()
if sel_sport != 'All':
    filtered = filtered[filtered['sport_label'] == sel_sport]
if sel_div != 'All':
    filtered = filtered[filtered['division'] == sel_div]
if sel_year != 'All':
    filtered = filtered[filtered['year'] == sel_year]
if sel_status != 'All':
    filtered = filtered[filtered['status'] == sel_status]
if sel_match != 'All' and 'player_id_64a' in filtered.columns:
    has_pid = filtered['player_id_64a'].astype(str).str.strip() != ''
    if sel_match == 'Matched to 64A':
        filtered = filtered[has_pid]
    else:  # 'Unmatched'
        filtered = filtered[~has_pid]
if search:
    q = search.lower()
    filtered = filtered[
        filtered['first_name'].str.lower().str.contains(q, na=False) |
        filtered['last_name'].str.lower().str.contains(q, na=False) |
        filtered['institution'].str.lower().str.contains(q, na=False) |
        filtered['conference'].str.lower().str.contains(q, na=False)
    ]

# ── Summary bar ──────────────────────────────────────────────────────────────
s1, s2, s3, s4, s5 = st.columns(5)
s1.metric('Total Entries', f'{len(filtered):,}')
s2.metric('Active', f'{(filtered["status"] == "Active").sum():,}')
s3.metric('Signed', f'{(filtered["status"] == "Signed").sum():,}')
s4.metric('Withdrawn', f'{(filtered["status"] == "Withdrawn").sum():,}')
if 'player_id_64a' in filtered.columns:
    matched_n = (filtered['player_id_64a'].astype(str).str.strip() != '').sum()
    pct = (matched_n / len(filtered) * 100) if len(filtered) else 0
    s5.metric('Matched to 64A', f'{matched_n:,}', f'{pct:.1f}%')
else:
    s5.metric('Matched to 64A', '0', '—')

st.markdown('---')

# ── Table (NCAA format) ──────────────────────────────────────────────────────
# Build display columns matching NCAA portal layout
display = filtered.copy()
display['Name'] = display['first_name'].str.title() + ' ' + display['last_name'].str.title()

# Division display
div_map = {'I': 'D-I', 'II': 'D-II', 'III': 'D-III'}
display['Div'] = display['division'].map(div_map).fillna(display['division'])

# Rename for display
display = display.rename(columns={
    'year': 'Year',
    'institution': 'Institution',
    'conference': 'Conference',
    'status': 'Status',
    'transfer_date': 'Transfer Date',
    'update_date': 'Updated',
    'sport_label': 'Sport',
    'player_id_64a': '64A ID',
    'position': 'Pos',
    'classification': 'Class',
    'height': 'Ht',
    'bat': 'B',
    'throw': 'T',
    'hometown': 'Hometown',
})

show_cols = ['Year', 'Name', 'Div', 'Institution', 'Conference', 'Status',
             'Transfer Date', 'Updated', 'Sport',
             '64A ID', 'Pos', 'Class', 'Ht', 'B', 'T', 'Hometown']
show_cols = [c for c in show_cols if c in display.columns]

# Sort: most recent transfer date first
display['_sort_date'] = pd.to_datetime(display['Transfer Date'], errors='coerce', format='mixed')
display = display.sort_values('_sort_date', ascending=False).reset_index(drop=True)
display.index = display.index + 1
display.index.name = '#'

st.dataframe(display[show_cols], use_container_width=True, height=700)

# ── Download ─────────────────────────────────────────────────────────────────
st.download_button(
    f'Download filtered data ({len(filtered):,} entries)',
    data=display[show_cols].to_csv(index=False),
    file_name='ncaa_transfer_portal.csv',
    mime='text/csv',
)
