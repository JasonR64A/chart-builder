"""
NCAA Transfer Portal — Browse all portal entries across years.

Shows the full portal archive enriched with the 64A player database:
  - 64A player ID, position, classification, height, bat, throw, hometown
  - Portal rating (so the user can prioritize highest-rated players first)
  - Live Supabase decisions (decisions made since the last pipeline run
    are reflected on the page immediately, without waiting for a rerun)

Filters expose every enrichment column with an explicit "Blank" option
so the user can find players where 64A is missing a field that should
be filled in.
"""
import streamlit as st
import pandas as pd
import requests
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = _APP_DIR / 'data' / 'portal_archive'
PORTAL_DIR = _APP_DIR / 'data' / 'portal'
PLAYERS_CSV = _APP_DIR / 'data' / 'players.csv'
PRP_CSV = _APP_DIR / 'data' / 'portal_rank_player.csv'

SUPABASE_URL = 'https://vfzoroabzmbvwkcyozes.supabase.co'
SUPABASE_KEY = (
    'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.'
    'eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZmem9yb2Fiem1idndrY3lvemVzIiwi'
    'cm9sZSI6ImFub24iLCJpYXQiOjE2OTQwNDU5NTgsImV4cCI6MjAwOTYyMTk1OH0.'
    'MpzhpgI2fVDC5ucrECl2AuQ9VfT_8aaTmFunthyJAPA'
)

st.set_page_config(page_title='NCAA Transfer Portal', layout='wide')
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
def load_players():
    """The canonical 64A player records (keyed by id)."""
    if not PLAYERS_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(PLAYERS_CSV, dtype=str, encoding='utf-8').fillna('')
    except UnicodeDecodeError:
        df = pd.read_csv(PLAYERS_CSV, dtype=str, encoding='cp1252').fillna('')
    keep = ['id', 'position', 'classification', 'height', 'bat', 'throw', 'hometown']
    df = df[[c for c in keep if c in df.columns]]
    return df.rename(columns={'id': 'player_id_64a'})


@st.cache_data
def load_matched_bridge():
    """ncaa_id -> player_id_64a from current pipeline run."""
    frames = []
    for f in ['baseball_matched.csv', 'softball_matched.csv']:
        p = PORTAL_DIR / f
        if p.exists():
            df = pd.read_csv(p, dtype=str).fillna('')
            frames.append(df[['ncaa_id', 'player_id_64a']])
    if not frames:
        return pd.DataFrame(columns=['ncaa_id', 'player_id_64a'])
    out = pd.concat(frames, ignore_index=True)
    out = out[out['player_id_64a'] != ''].drop_duplicates('ncaa_id')
    return out


@st.cache_data(ttl=60)
def load_supabase_decisions():
    """Pull portal_review_decisions live (1-min TTL). Returns a dict of
    ncaa_id -> {action, override_id}. Lets the page reflect decisions made
    after the last pipeline run without waiting for a rerun."""
    try:
        r = requests.get(
            f'{SUPABASE_URL}/rest/v1/portal_review_decisions',
            headers={'apikey': SUPABASE_KEY, 'Authorization': f'Bearer {SUPABASE_KEY}'},
            params={'select': 'ncaa_id,action,override_id', 'limit': '5000'},
            timeout=10,
        )
        if r.status_code == 200:
            return {d['ncaa_id']: d for d in r.json()
                    if d.get('ncaa_id') and (d.get('action') or d.get('override_id'))}
    except Exception:
        pass
    return {}


@st.cache_data
def load_rating_bridge():
    """player_id_64a -> 64A portal rating (2026 only) so the table can sort by rank."""
    if not PRP_CSV.exists():
        return pd.DataFrame(columns=['player_id_64a', 'rating'])
    df = pd.read_csv(PRP_CSV, dtype=str).fillna('')
    df = df[df['year'] == '2026']
    if 'sixty_four_rating_portal_player' not in df.columns:
        return pd.DataFrame(columns=['player_id_64a', 'rating'])
    df = df[['player_id', 'sixty_four_rating_portal_player']].rename(
        columns={'player_id': 'player_id_64a', 'sixty_four_rating_portal_player': 'rating'}
    )
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])
    df = df[df['rating'] > 0].drop_duplicates('player_id_64a')
    return df


# ── Build the merged dataset ─────────────────────────────────────────────────
df = load_portal_data()
if df.empty:
    st.warning('No portal data found.')
    st.stop()

bridge = load_matched_bridge()
decisions = load_supabase_decisions()
players = load_players()
ratings = load_rating_bridge()

# 1. Resolve effective player_id_64a:
#    - first from matched.csv
#    - then fall back to Supabase 'adjust' decisions with an override_id
df = df.merge(bridge, on='ncaa_id', how='left').fillna('')


def effective_pid(row):
    if row['player_id_64a']:
        return row['player_id_64a'], 'matched'
    dec = decisions.get(row['ncaa_id'])
    if not dec:
        return '', 'none'
    if dec.get('action') == 'unmatch':
        return '', 'unmatch'
    ov = (dec.get('override_id') or '').strip()
    if ov:
        return ov, 'pending'
    # 'confirm' without override_id: matcher's default match would apply
    # on next pipeline run, but we don't have that resolved here. Show as
    # pending without an enrichment join.
    return '', 'pending_confirm'


resolved = df.apply(effective_pid, axis=1, result_type='expand')
resolved.columns = ['effective_pid', 'match_state']
df['effective_pid'] = resolved['effective_pid']
df['match_state'] = resolved['match_state']

# 2. Enrichment join — by effective_pid (covers both matched + pending adjust)
if not players.empty:
    enrich_keyed = players.rename(columns={'player_id_64a': 'effective_pid'})
    df = df.merge(enrich_keyed, on='effective_pid', how='left').fillna('')

# 3. Portal rating + per-sport rank
if not ratings.empty:
    rating_keyed = ratings.rename(columns={'player_id_64a': 'effective_pid'})
    df = df.merge(rating_keyed, on='effective_pid', how='left')
df['rating'] = pd.to_numeric(df.get('rating'), errors='coerce')

# Rank within sport for 2026 — highest rating = rank 1
rank_2026 = df[df['year'] == '2025-26'].copy()
rank_2026['portal_rank'] = (
    rank_2026.groupby('sport_label')['rating']
    .rank(method='min', ascending=False)
)
df = df.merge(
    rank_2026[['ncaa_id', 'portal_rank']],
    on='ncaa_id', how='left'
)

# ── Filters (top row + enrichment row) ───────────────────────────────────────
st.subheader('Filters')
r1c1, r1c2, r1c3, r1c4, r1c5, r1c6 = st.columns(6)
with r1c1:
    sport_opts = ['All'] + sorted(df['sport_label'].dropna().unique().tolist())
    sel_sport = st.selectbox('Sport', sport_opts, key='portal_sport')
with r1c2:
    div_opts = ['All'] + sorted(df['division'].dropna().unique().tolist())
    sel_div = st.selectbox('Division', div_opts, key='portal_div')
with r1c3:
    year_opts = ['All'] + sorted(df['year'].dropna().unique().tolist(), reverse=True)
    sel_year = st.selectbox('Academic Year', year_opts, key='portal_year', index=1 if len(year_opts) > 1 else 0)
with r1c4:
    status_opts = ['All'] + sorted(df['status'].dropna().unique().tolist())
    sel_status = st.selectbox('Status', status_opts, key='portal_status')
with r1c5:
    match_opts = ['All', 'Matched (incl. pending)', 'Pending decision only', 'Truly unmatched']
    sel_match = st.selectbox('64A Match', match_opts, key='portal_match')
with r1c6:
    search = st.text_input('Search', '', placeholder='Name, school, conf…', key='portal_search')


def field_filter_options(col):
    """Return ['All', 'Blank (incomplete)', ...sorted unique values] for a field."""
    vals = sorted([v for v in df.get(col, pd.Series()).dropna().unique() if v != ''])
    return ['All', 'Blank (incomplete)'] + vals


r2c1, r2c2, r2c3, r2c4, r2c5, r2c6 = st.columns(6)
with r2c1:
    sel_pos = st.selectbox('Position', field_filter_options('position'), key='portal_pos')
with r2c2:
    sel_class = st.selectbox('Classification', field_filter_options('classification'), key='portal_class')
with r2c3:
    sel_ht = st.selectbox('Height', field_filter_options('height'), key='portal_ht')
with r2c4:
    sel_bat = st.selectbox('Bat', field_filter_options('bat'), key='portal_bat')
with r2c5:
    sel_throw = st.selectbox('Throw', field_filter_options('throw'), key='portal_throw')
with r2c6:
    # Hometown has many distinct values — provide blank-only or All
    sel_home = st.selectbox('Hometown', ['All', 'Blank (incomplete)'], key='portal_home')

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
if sel_match == 'Matched (incl. pending)':
    filtered = filtered[filtered['match_state'].isin(['matched', 'pending', 'pending_confirm'])]
elif sel_match == 'Pending decision only':
    filtered = filtered[filtered['match_state'].isin(['pending', 'pending_confirm'])]
elif sel_match == 'Truly unmatched':
    filtered = filtered[~filtered['match_state'].isin(['matched', 'pending', 'pending_confirm'])]


def apply_field_filter(df_, col, sel):
    if sel == 'All':
        return df_
    if sel == 'Blank (incomplete)':
        return df_[df_[col].astype(str).str.strip() == '']
    return df_[df_[col] == sel]


filtered = apply_field_filter(filtered, 'position', sel_pos)
filtered = apply_field_filter(filtered, 'classification', sel_class)
filtered = apply_field_filter(filtered, 'height', sel_ht)
filtered = apply_field_filter(filtered, 'bat', sel_bat)
filtered = apply_field_filter(filtered, 'throw', sel_throw)
filtered = apply_field_filter(filtered, 'hometown', sel_home)

if search:
    q = search.lower()
    filtered = filtered[
        filtered['first_name'].str.lower().str.contains(q, na=False) |
        filtered['last_name'].str.lower().str.contains(q, na=False) |
        filtered['institution'].str.lower().str.contains(q, na=False) |
        filtered['conference'].str.lower().str.contains(q, na=False)
    ]

# ── Summary bar ──────────────────────────────────────────────────────────────
s1, s2, s3, s4, s5, s6 = st.columns(6)
s1.metric('Total Entries', f'{len(filtered):,}')
s2.metric('Active', f'{(filtered["status"] == "Active").sum():,}')
s3.metric('Signed', f'{(filtered["status"] == "Signed").sum():,}')
s4.metric('Withdrawn', f'{(filtered["status"] == "Withdrawn").sum():,}')
matched_n = (filtered['match_state'] == 'matched').sum()
pending_n = filtered['match_state'].isin(['pending', 'pending_confirm']).sum()
pct = (matched_n / len(filtered) * 100) if len(filtered) else 0
s5.metric('Matched (live)', f'{matched_n:,}', f'{pct:.1f}%')
s6.metric('Pending decisions', f'{pending_n:,}')

if pending_n:
    st.info(
        f'{pending_n:,} entries have a Supabase decision made since the last pipeline run. '
        f'They show as matched here but won\'t appear in `portal_rank_player.csv` until the '
        f'next portal pipeline run rebuilds matched.csv.'
    )

st.markdown('---')

# ── Table ────────────────────────────────────────────────────────────────────
display = filtered.copy()
display['Name'] = display['first_name'].str.title() + ' ' + display['last_name'].str.title()
div_map = {'I': 'D-I', 'II': 'D-II', 'III': 'D-III'}
display['Div'] = display['division'].map(div_map).fillna(display['division'])
state_map = {'matched': '✓', 'pending': '⏳', 'pending_confirm': '⏳', 'unmatch': '—', 'none': '✗'}
display['Match'] = display['match_state'].map(state_map).fillna('?')
display['Rank'] = display['portal_rank'].apply(
    lambda v: int(v) if pd.notna(v) and v != '' else None
)
display['Rating'] = display['rating'].apply(
    lambda v: round(v, 1) if pd.notna(v) else None
)

display = display.rename(columns={
    'year': 'Year',
    'institution': 'Institution',
    'conference': 'Conference',
    'status': 'Status',
    'transfer_date': 'Transfer Date',
    'update_date': 'Updated',
    'sport_label': 'Sport',
    'effective_pid': '64A ID',
    'position': 'Pos',
    'classification': 'Class',
    'height': 'Ht',
    'bat': 'B',
    'throw': 'T',
    'hometown': 'Hometown',
})

show_cols = ['Rank', 'Rating', 'Year', 'Sport', 'Name', 'Div', 'Institution',
             'Conference', 'Status', 'Transfer Date', 'Updated', 'Match',
             '64A ID', 'Pos', 'Class', 'Ht', 'B', 'T', 'Hometown']
show_cols = [c for c in show_cols if c in display.columns]

# Sort: rank ascending if present, else most-recent transfer date
display['_sort_date'] = pd.to_datetime(display['Transfer Date'], errors='coerce', format='mixed')
if 'Rank' in display.columns:
    # rank nulls go to bottom; among ranked rows, lower rank = higher importance
    display = display.sort_values(
        ['Rank', '_sort_date'],
        ascending=[True, False],
        na_position='last',
    ).reset_index(drop=True)
else:
    display = display.sort_values('_sort_date', ascending=False).reset_index(drop=True)
display.index = display.index + 1
display.index.name = '#'

st.dataframe(display[show_cols], use_container_width=True, height=700)

st.download_button(
    f'Download filtered data ({len(filtered):,} entries)',
    data=display[show_cols].to_csv(index=False),
    file_name='ncaa_transfer_portal.csv',
    mime='text/csv',
)

st.caption(
    '· Match column: ✓ matched in matched.csv · ⏳ pending Supabase decision '
    '(not yet in matched.csv — runs through on next pipeline rebuild) · '
    '✗ no decision (truly unmatched) · — explicit unmatch decision. '
    'Supabase decisions are refetched every 60 seconds; force a refresh by '
    'reloading the page.'
)
