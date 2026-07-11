"""Serialized cairosvg rendering.

libcairo is not safe under concurrent use — two Streamlit sessions
rasterizing SVGs at the same time can segfault the whole process
(Render 'Exited with status 139'). Every svg2png call in the app goes
through this module-level lock so renders queue instead of colliding.
"""
import threading

import cairosvg

_CAIRO_LOCK = threading.Lock()


def safe_svg2png(*args, **kwargs):
    with _CAIRO_LOCK:
        return cairosvg.svg2png(*args, **kwargs)
