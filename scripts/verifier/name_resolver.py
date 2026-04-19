"""
Name resolver for reconciling player names across sources.

Reason: hitting_pbp (box scores) and hitting.csv (season stats) often spell the
same player differently — "Matt Priest" vs "Matthew Priest", "Pena" vs
"Pena-Edwards", "O'Conor" vs "OConor", "Nunez" vs "Nunez" (accented), etc.
The exact-name join drops ~8% of D1 box-score players, which breaks cross-checks.

This module produces a set of normalized keys for each name; two names match if
they share any key. Use `build_name_index(names)` to build a lookup, then call
`match(name)` to find candidate matches.

Directions of attack:
- Case folding + punctuation strip
- Accent strip (ASCII-fold)
- Common nickname expansion (Matt <-> Matthew, Nate <-> Nathan, ...)
- Suffix strip (Jr / Sr / II / III / IV / V)
- Last-name truncation tolerance (for hyphenated names)
"""

import re
import unicodedata
from typing import Iterable


# Common nickname <-> given name map. Each entry maps a short form to a set of
# equivalent long forms (and vice versa). Keep the short form as the dictionary
# key; long forms as the value set.
_NICKNAMES: dict[str, set[str]] = {
    'matt':    {'matthew'},
    'mike':    {'michael'},
    'nate':    {'nathan', 'nathaniel'},
    'nick':    {'nicholas', 'nicolas'},
    'chris':   {'christopher', 'christian'},
    'tom':     {'thomas'},
    'tommy':   {'thomas'},
    'dave':    {'david'},
    'dan':     {'daniel'},
    'danny':   {'daniel'},
    'jim':     {'james'},
    'jimmy':   {'james'},
    'jon':     {'jonathan'},
    'johnny':  {'jonathan', 'john'},
    'rob':     {'robert'},
    'robbie':  {'robert'},
    'bobby':   {'robert'},
    'will':    {'william'},
    'billy':   {'william'},
    'bill':    {'william'},
    'alex':    {'alexander'},
    'xander':  {'alexander'},
    'ben':     {'benjamin'},
    'benny':   {'benjamin'},
    'tony':    {'anthony'},
    'joe':     {'joseph'},
    'joey':    {'joseph'},
    'sam':     {'samuel'},
    'sammy':   {'samuel'},
    'ken':     {'kenneth'},
    'kenny':   {'kenneth'},
    'steve':   {'steven', 'stephen'},
    'andy':    {'andrew'},
    'drew':    {'andrew'},
    'rick':    {'richard'},
    'ricky':   {'richard'},
    'rich':    {'richard'},
    'dick':    {'richard'},
    'greg':    {'gregory'},
    'ed':      {'edward', 'edgar', 'eddie'},
    'eddie':   {'edward', 'edgar'},
    'pat':     {'patrick'},
    'patty':   {'patrick'},
    'fred':    {'frederick', 'alfred'},
    'charlie': {'charles'},
    'chuck':   {'charles'},
    'phil':    {'philip', 'phillip'},
    'zach':    {'zachary'},
    'zack':    {'zachary'},
    'zeke':    {'ezekiel'},
    'tim':     {'timothy'},
    'vince':   {'vincent'},
    'gabe':    {'gabriel'},
    'raf':     {'rafael', 'raphael'},
    'cam':     {'cameron'},
    'jack':    {'john', 'jackson'},
    'jake':    {'jacob'},
    'josh':    {'joshua'},
    'ty':      {'tyler', 'tyrone'},
    'ron':     {'ronald'},
    'ronnie':  {'ronald'},
    'al':      {'albert', 'alan'},
    'tyler':   set(),
    'brad':    {'bradley', 'bradford'},
}

# Build reverse map: long_form -> short_form
_LONG_TO_SHORT: dict[str, str] = {}
for short, longs in _NICKNAMES.items():
    for long in longs:
        _LONG_TO_SHORT[long] = short

_SUFFIX_RE = re.compile(r'\s+(jr|sr|ii|iii|iv|v)\b\.?', re.IGNORECASE)


def strip_accents(s: str) -> str:
    """Convert accented characters to ASCII equivalents."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if not unicodedata.combining(c)
    )


def normalize(name: str) -> str:
    """Lowercase, strip accents/punctuation/suffixes, collapse whitespace."""
    if not isinstance(name, str):
        return ''
    s = strip_accents(name).lower()
    # Strip suffixes
    s = _SUFFIX_RE.sub('', s)
    # Remove punctuation (apostrophes, periods, commas)
    s = re.sub(r"[.,']", '', s)
    # Normalize hyphens to spaces (so "Pena-Edwards" -> "pena edwards")
    s = s.replace('-', ' ')
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def keys_for(name: str) -> set[str]:
    """Generate the set of equivalence keys for a name.

    A name's keys include:
    - The normalized full form
    - Variants with first name expanded from nickname (Matt -> Matthew)
    - Variants with first name contracted to nickname (Matthew -> Matt)
    - Last-name-only (for hyphenated last-name tolerance)
    - First-initial + last-name (e.g., "m priest")
    """
    keys: set[str] = set()
    n = normalize(name)
    if not n:
        return keys
    keys.add(n)

    parts = n.split(' ')
    if len(parts) >= 2:
        first = parts[0]
        rest = parts[1:]
        last = parts[-1]

        # First-initial + last name (specific enough to avoid same-surname collisions)
        keys.add(f'{first[0]} {last}')

        # Hyphenated last-name tolerance: "Ethan Pena-Edwards" (normalized "ethan pena edwards")
        # should still match "Ethan Pena" via key "e pena" (first-initial + first chunk of last).
        # We DO NOT add last-name-only as a key — that causes false positives across
        # unrelated players who share a surname (e.g., Zach Davis vs Tague Davis).
        if len(rest) >= 2:
            for part in rest:
                keys.add(f'{first[0]} {part}')

        # Nickname expansion: Matt -> Matthew
        if first in _NICKNAMES:
            for long_form in _NICKNAMES[first]:
                keys.add(' '.join([long_form] + rest))
                keys.add(f'{long_form[0]} {last}')

        # Contraction: Matthew -> Matt
        if first in _LONG_TO_SHORT:
            short_form = _LONG_TO_SHORT[first]
            keys.add(' '.join([short_form] + rest))
            keys.add(f'{short_form[0]} {last}')

    return keys


def build_name_index(names: Iterable[tuple[str, ...]]):
    """Build a key -> list-of-tuples index for fast lookups.

    Pass iterable of (name, *extra_tags) tuples. Returns dict[key -> list].
    """
    idx: dict[str, list] = {}
    for item in names:
        if isinstance(item, str):
            name, extra = item, ()
        else:
            name, extra = item[0], item[1:]
        for k in keys_for(name):
            idx.setdefault(k, []).append((name,) + tuple(extra))
    return idx


def match(name: str, index: dict) -> list:
    """Look up a name against a prebuilt index. Returns matching entries."""
    hits = []
    seen = set()
    for k in keys_for(name):
        for entry in index.get(k, []):
            key = tuple(entry)
            if key in seen:
                continue
            seen.add(key)
            hits.append(entry)
    return hits


if __name__ == '__main__':
    # Quick sanity tests
    cases = [
        ('Matt Priest', 'Matthew Priest'),
        ('Nate Conner', 'Nathan Conner'),
        ('Ethan Pena', 'Ethan Pena-Edwards'),
        ("Mike O'Conor", 'Michael OConor'),
        ('Jose Nunez', 'José Núñez'),
        ('Dan Smith', 'Daniel Smith'),
        ('Matt Priest Jr.', 'Matthew Priest'),
    ]
    for a, b in cases:
        ka = keys_for(a)
        kb = keys_for(b)
        overlap = ka & kb
        print(f"{a!r:30s} ~ {b!r:30s}  overlap? {bool(overlap):5}  common_keys={sorted(overlap)[:3]}")
