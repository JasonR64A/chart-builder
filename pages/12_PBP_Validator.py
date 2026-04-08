"""
PBP Validator — Review failures, correct plays, and build classification rules
that auto-fix similar events across the dataset.

Three sections:
  1. Failure Browser — look at specific games and correct individual plays
  2. Rules Library — view/edit/disable classification rules that apply globally
  3. Suggested Rules — auto-proposed rules based on common failure patterns
"""
import streamlit as st
import pandas as pd
import json
import re
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

_APP_DIR = Path(__file__).resolve().parent.parent
VALIDATED_DIR = _APP_DIR / 'pbp_data' / 'validated'
RULES_PATH = _APP_DIR / 'data' / 'pbp_classification_rules.csv'
CORRECTIONS_PATH = _APP_DIR / 'data' / 'pbp_corrections.csv'
EXCEPTIONS_PATH = _APP_DIR / 'data' / 'pbp_validation_exceptions.csv'
REVIEWED_PATTERNS_PATH = _APP_DIR / 'data' / 'pbp_reviewed_patterns.csv'

# ── Classification options ───────────────────────────────────────────────────
OUTCOME_OPTIONS = [
    'keep as is',
    '1B', '2B', '3B', 'HR',
    'BB', 'HBP',
    'K', 'KL',
    'GO', 'FO', 'LO', 'PO', 'FC', 'DP', 'TP', 'E',
    'SF', 'SAC',
    'NONE (not a PA — baserunning/pickoff/etc.)',
]
def outcome_to_code(label):
    if label == 'keep as is':
        return None
    if label.startswith('NONE'):
        return 'NONE'
    return label.split()[0]

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_diff(sport, division):
    path = VALIDATED_DIR / f'validation_diff_{sport}_{division}.json'
    if not path.exists():
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def load_rules():
    if not RULES_PATH.exists():
        return pd.DataFrame(columns=[
            'rule_id', 'priority', 'enabled', 'name', 'match_playResult',
            'match_contains', 'match_excludes', 'match_regex', 'corrected_result',
            'notes', 'added_at',
        ])
    return pd.read_csv(RULES_PATH, keep_default_na=False, dtype=str)

def save_rules(df):
    RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RULES_PATH, index=False)

def add_rule(name, match_playResult, match_contains, match_excludes, match_regex, corrected_result, notes):
    df = load_rules()
    next_id = 1
    if len(df) > 0:
        try:
            next_id = max(int(x) for x in df['rule_id'].astype(str) if str(x).isdigit()) + 1
        except Exception:
            next_id = len(df) + 1
    next_priority = 100 + next_id  # user-added rules after seed rules
    new_row = pd.DataFrame([{
        'rule_id': str(next_id),
        'priority': str(next_priority),
        'enabled': 'true',
        'name': name,
        'match_playResult': match_playResult or '',
        'match_contains': match_contains or '',
        'match_excludes': match_excludes or '',
        'match_regex': match_regex or '',
        'corrected_result': corrected_result,
        'notes': notes or '',
        'added_at': datetime.now().strftime('%Y-%m-%d'),
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    save_rules(df)
    return next_id

def load_corrections():
    if not CORRECTIONS_PATH.exists():
        return pd.DataFrame(columns=[
            'gameId', 'playerId', 'eventIndex', 'original_result',
            'corrected_result', 'playAction', 'correctedBy', 'correctedAt',
        ])
    return pd.read_csv(CORRECTIONS_PATH, low_memory=False)

def save_correction(gameId, playerId, eventIndex, original_result, corrected_result, playAction):
    df = load_corrections()
    mask = (
        (df['gameId'] == gameId) &
        (df['playerId'] == playerId) &
        (df['eventIndex'] == eventIndex)
    )
    df = df[~mask].copy()
    new_row = pd.DataFrame([{
        'gameId': gameId,
        'playerId': playerId,
        'eventIndex': eventIndex,
        'original_result': original_result,
        'corrected_result': corrected_result,
        'playAction': playAction,
        'correctedBy': 'user',
        'correctedAt': datetime.now().isoformat(timespec='seconds'),
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    CORRECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CORRECTIONS_PATH, index=False)

def remove_correction(gameId, playerId, eventIndex):
    df = load_corrections()
    mask = (
        (df['gameId'] == gameId) &
        (df['playerId'] == playerId) &
        (df['eventIndex'] == eventIndex)
    )
    if mask.any():
        df = df[~mask].copy()
        df.to_csv(CORRECTIONS_PATH, index=False)

def load_exceptions():
    """Game-players manually verified as correct (counts mismatch is a data quality
    issue outside of the parser's control — box score error, missing event, etc.).
    The validator treats these as validated."""
    if not EXCEPTIONS_PATH.exists():
        return pd.DataFrame(columns=[
            'gameId', 'playerId', 'playerName', 'teamName', 'date',
            'reason', 'verifiedAt',
        ])
    return pd.read_csv(EXCEPTIONS_PATH, low_memory=False)

def save_exception(gameId, playerId, playerName, teamName, date, reason=''):
    df = load_exceptions()
    mask = (df['gameId'] == gameId) & (df['playerId'] == playerId)
    df = df[~mask].copy()
    new_row = pd.DataFrame([{
        'gameId': gameId,
        'playerId': playerId,
        'playerName': playerName,
        'teamName': teamName,
        'date': date,
        'reason': reason,
        'verifiedAt': datetime.now().isoformat(timespec='seconds'),
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    EXCEPTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXCEPTIONS_PATH, index=False)

def remove_exception(gameId, playerId):
    df = load_exceptions()
    mask = (df['gameId'] == gameId) & (df['playerId'] == playerId)
    if mask.any():
        df = df[~mask].copy()
        df.to_csv(EXCEPTIONS_PATH, index=False)

def load_reviewed_patterns():
    """Patterns the user has reviewed and chose to keep as-is. Stored so they
    don't reappear in the Pattern Review tab."""
    if not REVIEWED_PATTERNS_PATH.exists():
        return pd.DataFrame(columns=['playResult', 'signal', 'decision', 'reviewedAt'])
    return pd.read_csv(REVIEWED_PATTERNS_PATH, low_memory=False, keep_default_na=False)

def save_reviewed_pattern(playResult, signal, decision):
    df = load_reviewed_patterns()
    mask = (df['playResult'] == playResult) & (df['signal'] == signal)
    df = df[~mask].copy()
    new_row = pd.DataFrame([{
        'playResult': playResult,
        'signal': signal,
        'decision': decision,
        'reviewedAt': datetime.now().isoformat(timespec='seconds'),
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    REVIEWED_PATTERNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REVIEWED_PATTERNS_PATH, index=False)

# ── Rule matching helpers (for UI preview + suggester) ───────────────────────
def strip_parens(text):
    return re.sub(r'\([^)]*\)', '', str(text))

def rule_matches(rule, playResult, playAction):
    """Return True if the given event would match this rule."""
    action_clean = strip_parens(playAction).lower()
    result = (playResult or '').upper()
    pr_filter = str(rule.get('match_playResult', '')).strip()
    if pr_filter:
        codes = {c.strip().upper() for c in pr_filter.split(',') if c.strip()}
        if result not in codes:
            return False
    contains_spec = str(rule.get('match_contains', '')).strip()
    if contains_spec:
        for sub in [s.strip().lower() for s in contains_spec.split(',') if s.strip()]:
            if sub not in action_clean:
                return False
    excludes_spec = str(rule.get('match_excludes', '')).strip()
    if excludes_spec:
        for sub in [s.strip().lower() for s in excludes_spec.split(',') if s.strip()]:
            if sub in action_clean:
                return False
    regex_spec = str(rule.get('match_regex', '')).strip()
    if regex_spec:
        try:
            if not re.search(regex_spec, action_clean, flags=re.IGNORECASE):
                return False
        except re.error:
            return False
    return True

@st.cache_data
def load_all_failed_events(sport, division):
    """Flatten all events from failed game-players into a single DataFrame
    for rule matching/preview."""
    data = load_diff(sport, division)
    if data is None:
        return pd.DataFrame()
    rows = []
    for f in data['failures']:
        if f.get('failure_type') != 'parser_mismatch':
            continue
        for i, ev in enumerate(f.get('events', [])):
            rows.append({
                'gameId': f.get('gameId'),
                'playerId': f.get('playerId'),
                'playerName': f.get('playerName', ''),
                'teamName': f.get('teamName', ''),
                'date': f.get('date', ''),
                'eventIndex': i,
                'playResult': ev.get('playResult') or '',
                'playAction': ev.get('playAction') or '',
                'inning': ev.get('inning', ''),
                'half': ev.get('half', ''),
            })
    return pd.DataFrame(rows)

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='PBP Validator', layout='wide')
st.title('PBP Validator')

st.sidebar.markdown('### Data Source')
sport = st.sidebar.selectbox('Sport', ['baseball', 'softball'])
division = st.sidebar.selectbox('Division', ['D1', 'D2', 'D3'])

data = load_diff(sport, division)
if data is None:
    st.error(f'No validation data for {sport} {division}. Run validate-pbp-events.py first.')
    st.stop()

# Summary
total = data['total_game_players']
valid = data['validated_count']
gaps = data.get('scrape_gap_count', 0)
parser_miss = data.get('parser_mismatch_count', 0)
rate = data['validation_rate']

c1, c2, c3, c4 = st.columns(4)
c1.metric('Total game-players', f'{total:,}')
c2.metric('Validated', f'{valid:,}', f'{rate*100:.1f}%')
c3.metric('Scrape gaps', f'{gaps:,}', help='Box has records but no events scraped — not a parser issue')
c4.metric('Parser mismatches', f'{parser_miss:,}', help='Fixable via rules or point corrections')

tab_suggest, tab_browser, tab_rules = st.tabs(['🎯 Pattern Review', '🔍 Failure Browser', '📚 Rules Library'])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: FAILURE BROWSER
# ─────────────────────────────────────────────────────────────────────────────
with tab_browser:
    failure_type = st.radio(
        'Failure type', ['Parser mismatches', 'Scrape gaps', 'All failures'],
        horizontal=True, key='fb_type',
    )
    failures = data['failures']
    if failure_type == 'Parser mismatches':
        failures = [f for f in failures if f.get('failure_type') == 'parser_mismatch']
    elif failure_type == 'Scrape gaps':
        failures = [f for f in failures if f.get('failure_type') == 'scrape_gap']

    # Hide game-players the user has manually verified (unless the "show verified"
    # toggle is on so you can review/unverify them).
    show_verified = st.checkbox('Show already-verified failures', value=False, key='fb_show_verified')
    exc_df = load_exceptions()
    if not show_verified and len(exc_df) > 0:
        verified_keys = {(int(r['gameId']), int(r['playerId'])) for _, r in exc_df.iterrows()}
        before_count = len(failures)
        failures = [f for f in failures if (int(f.get('gameId', 0)), int(f.get('playerId', 0))) not in verified_keys]
        hidden = before_count - len(failures)
        if hidden > 0:
            st.caption(f'{hidden:,} failures hidden (marked as verified).')

    col_filters = st.columns(3)
    teams = sorted({f.get('teamName', '') for f in failures if f.get('teamName')})
    team_filter = col_filters[0].selectbox('Team', ['(all)'] + teams, key='fb_team')
    if team_filter != '(all)':
        failures = [f for f in failures if f.get('teamName') == team_filter]
    player_q = col_filters[1].text_input('Player name contains', '', key='fb_player')
    if player_q:
        q = player_q.lower()
        failures = [f for f in failures if q in str(f.get('playerName', '')).lower()]
    sort_by = col_filters[2].selectbox('Sort by',
        ['Biggest AB diff', 'Player name', 'Team', 'Date (recent first)'], key='fb_sort')
    if sort_by == 'Biggest AB diff':
        failures = sorted(failures, key=lambda f: abs(f.get('diff', {}).get('ab', {}).get('delta', 0)), reverse=True)
    elif sort_by == 'Player name':
        failures = sorted(failures, key=lambda f: str(f.get('playerName', '')))
    elif sort_by == 'Team':
        failures = sorted(failures, key=lambda f: str(f.get('teamName', '')))
    else:
        failures = sorted(failures, key=lambda f: str(f.get('date', '')), reverse=True)

    st.markdown(f'**{len(failures):,} failures match your filters**')

    if not failures:
        st.info('No failures to display.')
    else:
        options = [
            f"{i+1}. {f.get('playerName','?')} — {f.get('teamName','?')} ({f.get('date','')}) — game {f.get('gameId')}"
            for i, f in enumerate(failures[:500])
        ]
        if len(failures) > 500:
            st.caption(f'Showing first 500 of {len(failures):,} — narrow filters to see more.')
        selected_idx = st.selectbox('Select a failure', range(len(options)), format_func=lambda i: options[i], key='fb_select')
        failure = failures[selected_idx]

        st.markdown('---')
        hcol = st.columns([3, 2])
        gameId = int(failure['gameId'])
        playerId = int(failure['playerId'])
        with hcol[0]:
            st.markdown(f"#### {failure.get('playerName', '?')} — {failure.get('teamName', '?')}")
            st.markdown(f"**Game:** {failure.get('gameId')} &nbsp;&nbsp; **Date:** {failure.get('date','')} &nbsp;&nbsp; **Type:** `{failure.get('failure_type','?')}`")
        with hcol[1]:
            if failure.get('failure_type') == 'scrape_gap':
                st.warning('Scrape gap — no events captured. Needs re-scrape, not manual correction.')

            # Mark-as-verified controls
            exc_df_detail = load_exceptions()
            is_verified = (
                (exc_df_detail['gameId'] == gameId) & (exc_df_detail['playerId'] == playerId)
            ).any() if len(exc_df_detail) > 0 else False
            if is_verified:
                st.success('✓ Marked as verified')
                if st.button('Undo verification', key=f'unverify_{gameId}_{playerId}'):
                    remove_exception(gameId, playerId)
                    st.rerun()
            else:
                reason = st.text_input(
                    'Verification reason (optional)',
                    placeholder='e.g. box score error, missing event, data quality issue',
                    key=f'vreason_{gameId}_{playerId}',
                )
                if st.button('✓ Mark as verified (keep all plays as-is)',
                             key=f'verify_{gameId}_{playerId}', type='primary'):
                    save_exception(
                        gameId=gameId,
                        playerId=playerId,
                        playerName=failure.get('playerName', ''),
                        teamName=failure.get('teamName', ''),
                        date=failure.get('date', ''),
                        reason=reason,
                    )
                    st.toast('Marked as verified — will no longer show as a failure.')
                    st.rerun()

        # Expected vs derived
        exp = failure.get('expected', {})
        der = failure.get('derived', {})
        fields = ['ab', 'h', 'doubles', 'triples', 'hr', 'bb', 'hbp', 'k', 'sf', 'sh']
        summary_rows = [
            {'Stat': f.upper(), 'Box score': exp.get(f, 0), 'Derived': der.get(f, 0),
             'Delta': der.get(f, 0) - exp.get(f, 0),
             'OK': '✓' if der.get(f, 0) == exp.get(f, 0) else '✗'}
            for f in fields
        ]
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=False)

        if failure.get('failure_type') == 'scrape_gap':
            pass  # nothing more to do
        else:
            st.markdown('#### Events')
            st.caption('Pick the correct classification for any event the parser got wrong. Save as a one-off point correction OR promote to a rule that applies everywhere.')

            existing_corrections = load_corrections()
            corr_lookup = {
                (int(row['gameId']), int(row['playerId']), int(row['eventIndex'])): row['corrected_result']
                for _, row in existing_corrections.iterrows()
            }

            events = failure.get('events', [])

            for i, e in enumerate(events):
                result = e.get('playResult') or '-'
                action = e.get('playAction') or '(no description)'
                inning = e.get('inning', '?')
                half = e.get('half', '?')
                existing = corr_lookup.get((gameId, playerId, i))

                with st.container(border=True):
                    ecol = st.columns([0.7, 0.7, 4, 2, 1.5])
                    ecol[0].markdown(f"**{inning} {half}**")
                    ecol[1].markdown(f"`{result}`")
                    ecol[2].markdown(f"{action}")

                    # Determine default dropdown value
                    if existing:
                        default_code = existing if existing != 'NONE' else 'NONE (not a PA — baserunning/pickoff/etc.)'
                        if default_code in OUTCOME_OPTIONS:
                            default_idx = OUTCOME_OPTIONS.index(default_code)
                        else:
                            default_idx = 0
                    else:
                        default_idx = 0

                    choice = ecol[3].selectbox(
                        'correct to', OUTCOME_OPTIONS,
                        index=default_idx, key=f'ocur_{gameId}_{playerId}_{i}',
                        label_visibility='collapsed',
                    )

                    # Action buttons
                    with ecol[4]:
                        if choice != 'keep as is':
                            new_code = outcome_to_code(choice)
                            if st.button('💾 Save', key=f'save_{gameId}_{playerId}_{i}', use_container_width=True):
                                save_correction(gameId, playerId, i, result, new_code, action)
                                st.toast(f"Saved: {result} → {new_code}")
                                st.rerun()
                            # Offer to promote to rule (pre-fill suggestion)
                            if st.button('📐 Rule…', key=f'rule_{gameId}_{playerId}_{i}', use_container_width=True):
                                st.session_state['promote_source'] = {
                                    'playResult': result,
                                    'playAction': action,
                                    'corrected_result': new_code,
                                }
                                st.session_state['active_tab'] = 'rules'

                    if existing:
                        ecol[4].caption(f'Corrected → {existing}')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: RULES LIBRARY
# ─────────────────────────────────────────────────────────────────────────────
with tab_rules:
    st.markdown('### Classification Rules')
    st.caption('Rules evaluate in priority order (lower = first). First match wins. Each rule can match on playResult code, substring text, exclusion text, and regex.')

    rules_df = load_rules()

    # Show the current rules table
    if len(rules_df) > 0:
        display = rules_df.copy()
        display['enabled'] = display['enabled'].astype(str).str.lower().isin(['true', '1', 'yes', 't', 'y'])
        edited = st.data_editor(
            display,
            hide_index=True,
            use_container_width=True,
            num_rows='fixed',
            column_config={
                'rule_id': st.column_config.TextColumn('ID', disabled=True, width='small'),
                'priority': st.column_config.NumberColumn('Priority', width='small'),
                'enabled': st.column_config.CheckboxColumn('Enabled'),
                'name': st.column_config.TextColumn('Name'),
                'match_playResult': st.column_config.TextColumn('playResult in'),
                'match_contains': st.column_config.TextColumn('Contains (all)'),
                'match_excludes': st.column_config.TextColumn('Excludes (none)'),
                'match_regex': st.column_config.TextColumn('Regex'),
                'corrected_result': st.column_config.TextColumn('→'),
                'notes': st.column_config.TextColumn('Notes'),
                'added_at': st.column_config.TextColumn('Added', disabled=True, width='small'),
            },
            key='rules_editor',
        )
        if st.button('Save rule changes', type='primary'):
            edited['enabled'] = edited['enabled'].map(lambda v: 'true' if v else 'false')
            save_rules(edited)
            st.success('Rules saved. Run validate-pbp-events.py to apply them.')
            st.cache_data.clear()
    else:
        st.info('No rules yet. Add one below or promote from a point correction.')

    st.markdown('---')
    st.markdown('### Add Rule')

    promote = st.session_state.get('promote_source')
    with st.form('add_rule_form'):
        default_name = ''
        default_pr = ''
        default_contains = ''
        default_excludes = ''
        default_regex = ''
        default_corr = 'SAC'
        if promote:
            default_name = f"Auto from {promote['playResult']}: {promote['playAction'][:50]}"
            default_pr = promote['playResult']
            default_corr = promote['corrected_result'] or 'NONE'
            # Suggest the playAction keywords (strip parens + keep main words)
            clean = strip_parens(promote['playAction']).lower().strip().rstrip('.').strip()
            # Pull first 2-3 distinctive words
            words = [w for w in re.split(r'[\s,]+', clean) if len(w) > 2][:3]
            default_contains = ', '.join(words)

        name = st.text_input('Rule name', value=default_name)
        colf = st.columns(3)
        with colf[0]:
            match_pr = st.text_input('playResult codes (comma-separated, e.g. FO,LO)', value=default_pr)
        with colf[1]:
            match_contains = st.text_input('Contains (ALL, comma-separated)', value=default_contains)
        with colf[2]:
            match_excludes = st.text_input('Excludes (NONE, comma-separated)', value=default_excludes)
        colg = st.columns(2)
        with colg[0]:
            match_regex = st.text_input('Regex (optional, case-insensitive)', value=default_regex)
        with colg[1]:
            outcome = st.selectbox('Corrected result', [
                '1B', '2B', '3B', 'HR', 'BB', 'HBP', 'K', 'KL',
                'GO', 'FO', 'LO', 'PO', 'FC', 'DP', 'E', 'SF', 'SAC', 'NONE'
            ], index=['1B', '2B', '3B', 'HR', 'BB', 'HBP', 'K', 'KL',
                      'GO', 'FO', 'LO', 'PO', 'FC', 'DP', 'E', 'SF', 'SAC', 'NONE'].index(default_corr) if default_corr in ['1B', '2B', '3B', 'HR', 'BB', 'HBP', 'K', 'KL', 'GO', 'FO', 'LO', 'PO', 'FC', 'DP', 'E', 'SF', 'SAC', 'NONE'] else 0)
        notes = st.text_input('Notes (optional)', value='')

        # Preview impact before saving
        preview_clicked = st.form_submit_button('Preview impact')
        submit_clicked = st.form_submit_button('Add rule', type='primary')

    if preview_clicked or submit_clicked:
        tentative = {
            'match_playResult': match_pr,
            'match_contains': match_contains,
            'match_excludes': match_excludes,
            'match_regex': match_regex,
            'corrected_result': outcome,
        }
        failed_events = load_all_failed_events(sport, division)
        if len(failed_events) > 0:
            match_count = sum(
                1 for _, r in failed_events.iterrows()
                if rule_matches(tentative, r['playResult'], r['playAction'])
            )
            st.info(f'This rule would match **{match_count:,}** failed events in {sport} {division}.')
        if submit_clicked:
            if not name.strip():
                st.error('Please provide a rule name.')
            else:
                new_id = add_rule(name, match_pr, match_contains, match_excludes, match_regex, outcome, notes)
                st.success(f'Rule #{new_id} added. Run validate-pbp-events.py to apply.')
                st.session_state['promote_source'] = None
                st.cache_data.clear()

# ─────────────────────────────────────────────────────────────────────────────
# TAB: PATTERN REVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_suggest:
    st.markdown('### Pattern Review')
    st.caption("Default for every pattern is 'keep as is'. Only touch the dropdown for patterns you want to correct, then hit 'Submit all' at the top or bottom. Corrections become rules that apply everywhere; keeps just hide the pattern from this queue.")

    failed_events = load_all_failed_events(sport, division)
    if len(failed_events) == 0:
        st.info('No parser mismatches to analyze.')
        st.stop()

    # Build patterns: (playResult, signal) → list of example actions
    STOP = {'the', 'a', 'to', 'of', 'on', 'in', 'at', 'out', 'advanced', 'base',
            'second', 'third', 'first', 'home', 'left', 'right', 'center',
            'field', 'for', 'from', 'by', 'an', 'and', 'or', 'his', 'her'}
    patterns = defaultdict(list)
    for _, r in failed_events.iterrows():
        result = (r['playResult'] or '').upper()
        action = strip_parens(r['playAction']).lower().strip().rstrip('.')
        if not action:
            continue
        words = [w for w in re.findall(r"[a-z]+", action) if w not in STOP and len(w) > 2]
        signals = set()
        for phrase in ['picked off', 'sacrifice fly', 'sacrifice bunt',
                       'reached on', 'scored', 'advanced on', 'bunt', 'hit by pitch',
                       'sac', 'sf', 'rbi', 'unearned', 'foul', 'lineup', 'pinch']:
            if phrase in action:
                signals.add(phrase)
        if len(words) >= 2:
            signals.add(' '.join(words[:2]))
        for sig in signals:
            patterns[(result, sig)].append(r['playAction'])

    # Filter out patterns the user has already reviewed
    reviewed_df = load_reviewed_patterns()
    reviewed_keys: set = set()
    if len(reviewed_df) > 0:
        reviewed_keys = {(str(row['playResult']), str(row['signal']))
                         for _, row in reviewed_df.iterrows()}

    top_patterns = sorted(
        [((pr, sig), examples) for (pr, sig), examples in patterns.items()
         if len(examples) >= 5 and (pr, sig) not in reviewed_keys],
        key=lambda x: -len(x[1]),
    )[:100]

    header = st.columns([3, 1])
    header[0].markdown(
        f'Analyzing **{len(failed_events):,}** failed events. '
        f'Showing top **{len(top_patterns)}** unreviewed patterns (≥5 occurrences).'
    )
    if len(reviewed_keys) > 0:
        header[1].caption(f'{len(reviewed_keys)} patterns already reviewed')

    if not top_patterns:
        st.success('All common patterns have been reviewed. Nothing left in the queue.')
    else:
        outcome_choices = [
            'keep as is',
            '1B', '2B', '3B', 'HR', 'BB', 'HBP', 'K', 'KL',
            'GO', 'FO', 'LO', 'PO', 'FC', 'DP', 'E', 'SF', 'SAC', 'NONE',
        ]

        def _submit_all_patterns():
            keeps = 0
            rules_added = 0
            for (pr, sig), examples in top_patterns:
                key = f'pr_choice_{pr}_{sig}'
                choice = st.session_state.get(key, 'keep as is')
                if choice == 'keep as is':
                    save_reviewed_pattern(pr, sig, 'keep')
                    keeps += 1
                else:
                    rule_name = f'Pattern Review: {pr or "any"} + "{sig}" → {choice}'
                    add_rule(
                        name=rule_name,
                        match_playResult=pr,
                        match_contains=sig,
                        match_excludes='',
                        match_regex='',
                        corrected_result=choice,
                        notes=f'Created from Pattern Review ({len(examples)} failed events)',
                    )
                    save_reviewed_pattern(pr, sig, choice)
                    rules_added += 1
            st.cache_data.clear()
            st.toast(f'Submitted: {keeps} kept, {rules_added} new rules')

        if st.button(f'✅ Submit all {len(top_patterns)} decisions',
                     type='primary', use_container_width=True, key='pr_submit_all_top'):
            _submit_all_patterns()
            st.rerun()

        for (pr, sig), examples in top_patterns:
            with st.container(border=True):
                top = st.columns([1, 3, 1])
                top[0].markdown(f"**{pr or '(any)'}**")
                top[1].markdown(f"contains `{sig}`")
                top[2].metric('Count', len(examples))

                with st.expander(f'{min(5, len(examples))} example plays'):
                    for ex in examples[:5]:
                        st.caption(f'• {ex}')

                st.selectbox(
                    'Decision',
                    outcome_choices,
                    key=f'pr_choice_{pr}_{sig}',
                    label_visibility='collapsed',
                )

        if st.button(f'✅ Submit all {len(top_patterns)} decisions',
                     type='primary', use_container_width=True, key='pr_submit_all_bottom'):
            _submit_all_patterns()
            st.rerun()
