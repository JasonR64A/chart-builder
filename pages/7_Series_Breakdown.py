"""
64 Analytics — Series Breakdown

Consolidates Series Preview (upcoming series matchup breakdown) and Series
Review (weekend recap with AI summary) into one page with tabs. The two
bodies live in app_lib/ and are executed inside tabs here.
"""
from pathlib import Path
import streamlit as st

st.set_page_config(page_title='64 Analytics — Series Breakdown', layout='wide',
                   initial_sidebar_state='expanded')

_ROOT = Path(__file__).resolve().parent.parent
_PREVIEW_BODY = _ROOT / 'app_lib' / '_series_preview_body.py'
_REVIEW_BODY = _ROOT / 'app_lib' / '_series_review_body.py'


def _run_body(path: Path):
    """Execute a page body in this module's globals so Streamlit renders
    inside the current tab. Uses compile() with the real path so tracebacks
    point at the original file.
    """
    src = path.read_text(encoding='utf-8')
    code = compile(src, str(path), 'exec')
    # Each tab gets a fresh globals dict so module-level names from Preview don't
    # leak into Review. __file__ is set to the body so Path(__file__).parent.parent
    # inside each body resolves to the project root (app_lib/x.py -> root).
    body_globals = {
        '__file__': str(path),
        '__name__': '__main__',
    }
    exec(code, body_globals)


tab_preview, tab_review = st.tabs(['Series Preview', 'Series Review'])

with tab_preview:
    _run_body(_PREVIEW_BODY)

with tab_review:
    _run_body(_REVIEW_BODY)
