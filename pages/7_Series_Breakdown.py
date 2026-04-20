"""
64 Analytics — Series Breakdown

Consolidates Series Preview (upcoming series matchup breakdown) and Series
Review (weekend recap with AI summary) into one sidebar entry. A radio
toggle at the top selects the active view; only that body executes on each
rerun (st.tabs() would render both bodies every time, doubling compute).
"""
from pathlib import Path
import streamlit as st

st.set_page_config(page_title='64 Analytics — Series Breakdown', layout='wide',
                   initial_sidebar_state='expanded')

_ROOT = Path(__file__).resolve().parent.parent
_PREVIEW_BODY = _ROOT / 'app_lib' / '_series_preview_body.py'
_REVIEW_BODY = _ROOT / 'app_lib' / '_series_review_body.py'


def _run_body(path: Path):
    """Execute a page body in a fresh globals dict so module-level names
    don't leak between Preview and Review on view-switches. __file__ is set
    to the body path so Path(__file__).parent.parent inside each body
    resolves to the project root.
    """
    src = path.read_text(encoding='utf-8')
    code = compile(src, str(path), 'exec')
    body_globals = {
        '__file__': str(path),
        '__name__': '__main__',
    }
    exec(code, body_globals)


_view = st.radio(
    'View',
    ['Series Preview', 'Series Review'],
    horizontal=True,
    label_visibility='collapsed',
    key='series_breakdown_view',
)

if _view == 'Series Preview':
    _run_body(_PREVIEW_BODY)
else:
    _run_body(_REVIEW_BODY)
