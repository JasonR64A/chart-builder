"""One-time script: expand rankings/name_map.csv to cover every Massey/DSR/RPI team.

Reads all external ranking sources, finds names not in teams.csv or name_map.csv,
maps them via explicit rules + transformations, and appends new rows to name_map.
"""
import pandas as pd
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / 'data'
NAME_MAP_PATH = DATA / 'rankings' / 'name_map.csv'

teams = pd.read_csv(DATA / 'teams.csv', low_memory=False)
our_bb = set(n for n in teams[teams['sport'] == 'Baseball']['name'].dropna().unique())
our_sb = set(n for n in teams[teams['sport'] == 'Softball']['name'].dropna().unique())
our_names = our_bb | our_sb

nm = pd.read_csv(NAME_MAP_PATH, low_memory=False)
mapped_ext = set(nm['external_name'].dropna().tolist())

# Explicit mappings — every external-name variant we've seen that doesn't
# match teams.csv directly. Keys are the external-source name, values are
# the name in teams.csv.
EXPLICIT = {
    # --- Massey / DSR abbreviations → teams.csv canonical ---
    'Mississippi': 'Ole Miss',
    'USC': 'Southern California',
    'USF': 'South Fla.',
    'Miami': 'Miami (FL)',
    'Miami FL': 'Miami (FL)',
    'Miami OH': 'Miami (OH)',
    'UConn': 'UConn',
    'Seattle': 'Seattle U',
    'Southern': 'Southern U.',
    'Citadel': 'The Citadel',
    "Saint Joseph's": "Saint Joseph's",

    # State abbreviations (Massey uses "St", DSR uses "State")
    'Alabama St': 'Alabama St.',
    'Alabama State': 'Alabama St.',
    'Alcorn St': 'Alcorn',
    'Alcorn State': 'Alcorn',
    'Arizona St': 'Arizona St.',
    'Arizona State': 'Arizona St.',
    'Arkansas St': 'Arkansas St.',
    'Arkansas State': 'Arkansas St.',
    'Ball St': 'Ball St.',
    'Ball State': 'Ball St.',
    'Coppin St': 'Coppin St.',
    'Coppin State': 'Coppin St.',
    'Delaware St': 'Delaware St.',
    'Delaware State': 'Delaware St.',
    'Florida St': 'Florida St.',
    'Florida State': 'Florida St.',
    'Fresno St': 'Fresno St.',
    'Fresno State': 'Fresno St.',
    'Georgia St': 'Georgia St.',
    'Georgia State': 'Georgia St.',
    'Illinois St': 'Illinois St.',
    'Illinois State': 'Illinois St.',
    'Indiana St': 'Indiana St.',
    'Indiana State': 'Indiana St.',
    'Jackson St': 'Jackson St.',
    'Jackson State': 'Jackson St.',
    'Jacksonville St': 'Jacksonville St.',
    'Jacksonville State': 'Jacksonville St.',
    'Kansas St': 'Kansas St.',
    'Kansas State': 'Kansas St.',
    'Kennesaw': 'Kennesaw St.',
    'Kennesaw St': 'Kennesaw St.',
    'Kennesaw State': 'Kennesaw St.',
    'Kent': 'Kent St.',
    'Kent St': 'Kent St.',
    'Kent State': 'Kent St.',
    'Michigan St': 'Michigan St.',
    'Michigan State': 'Michigan St.',
    'Morehead St': 'Morehead St.',
    'Morehead State': 'Morehead St.',
    'Murray St': 'Murray St.',
    'Murray State': 'Murray St.',
    'New Mexico St': 'New Mexico St.',
    'New Mexico State': 'New Mexico St.',
    'Nicholls St': 'Nicholls',
    'Norfolk St': 'Norfolk St.',
    'Norfolk State': 'Norfolk St.',
    'North Dakota St': 'North Dakota St.',
    'North Dakota State': 'North Dakota St.',
    'Ohio St': 'Ohio St.',
    'Ohio State': 'Ohio St.',
    'Penn St': 'Penn St.',
    'Penn State': 'Penn St.',
    'Sacramento St': 'Sacramento St.',
    'Sacramento State': 'Sacramento St.',
    'Sam Houston St': 'Sam Houston',
    'South Dakota St': 'South Dakota St.',
    'South Dakota State': 'South Dakota St.',
    'Tarleton St': 'Tarleton St.',
    'Tarleton State': 'Tarleton St.',
    'Washington St': 'Washington',
    'Washington State': 'Washington',
    'Wichita St': 'Wichita St.',
    'Wichita State': 'Wichita St.',
    'Wright St': 'Wright St.',
    'Wright State': 'Wright St.',
    'Youngstown St': 'Youngstown St.',
    'Youngstown State': 'Youngstown St.',

    # Direction abbreviations
    'Cal Baptist': 'California Baptist',
    'Cal State Fullerton': 'Cal St. Fullerton',
    'Cal State Northridge': 'CSUN',
    'Cent Arkansas': 'Central Ark.',
    'Central Arkansas': 'Central Ark.',
    'Central Conn': 'Central Conn. St.',
    'Central Connecticut': 'Central Conn. St.',
    'Central Michigan': 'Central Mich.',
    'C Michigan': 'Central Mich.',
    'Charleston So': 'Charleston So.',
    'Coastal Car': 'Coastal Carolina',
    'Col Charleston': 'Col. of Charleston',
    'College of Charleston': 'Col. of Charleston',
    'Dallas Bap': 'DBU',
    'Dallas Baptist': 'DBU',
    'East Tennessee State': 'ETSU',
    'Eastern Illinois': 'Eastern Ill.',
    'Eastern Ill': 'Eastern Ill.',
    'Eastern Kentucky': 'Eastern Ky.',
    'Eastern Michigan': 'Eastern Mich.',
    'F Dickinson': 'FDU',
    'Fairleigh Dickinson': 'FDU',
    'Florida Intl': 'FIU',
    'Gardner Webb': 'Gardner-Webb',
    'Georgia Southern': 'Ga. Southern',
    'Houston Chr': 'Houston Christian',
    'Illinois-Chicago': 'UIC',
    'Long Beach State': 'Long Beach St.',
    'Loy Marymount': 'LMU (CA)',
    'Loyola Marymount': 'LMU (CA)',
    'MA Lowell': 'UMass Lowell',
    'MD E Shore': 'UMES',
    'Maryland Eastern Shore': 'UMES',
    'MS Valley St': 'Mississippi Val.',
    'Mississippi Valley': 'Mississippi Val.',
    'MTSU': 'Middle Tenn.',
    'Middle Tennessee': 'Middle Tenn.',
    'Monmouth NJ': 'Monmouth',
    "Mt St Mary's": 'Mt. St. Marys',
    'N Colorado': 'Northern Colo.',
    'Northern Colorado': 'Northern Colo.',
    'NC A&T': 'N.C. A&T',
    'North Carolina A&T': 'N.C. A&T',
    'NE Omaha': 'Omaha',
    'North Alabama': 'North Ala.',
    'Northern Illinois': 'NIU',
    'Northern Kentucky': 'Northern Ky.',
    'Northwestern LA': 'Northwestern St.',
    'Northwestern State': 'Northwestern St.',
    'Queens NC': 'Queens (NC)',
    'SC Upstate': 'USC Upstate',
    'SF Austin': 'SFA',
    'Stephen F. Austin': 'SFA',
    'SUNY Albany': 'UAlbany',
    'Southeastern Louisiana': 'Southeastern La.',
    'Southern Illinois': 'Southern Ill.',
    'Southern Indiana': 'Southern Ind.',
    'Southern Miss': 'Southern Miss.',
    'St Bonaventure': 'St. Bonaventure',
    "St John's": "St. John's (NY)",
    "St. John's": "St. John's (NY)",
    "St Joseph's PA": "Saint Joseph's",
    "St. Joseph's PA": "Saint Joseph's",
    'St Louis': 'Saint Louis',
    "St Mary's CA": "Saint Mary's (CA)",
    "St Peter's": "Saint Peter's",
    'St Thomas MN': 'St. Thomas (MN)',
    'TAM C. Christi': 'A&M-Corpus Christi',
    'TN Martin': 'UT Martin',
    'TX Southern': 'Texas Southern',
    'UNC Wilmington': 'UNCW',
    'UNCG': 'UNCG',
    'UT San Antonio': 'UTSA',
    'UMass': 'Massachusetts',
    'UL Monroe': 'ULM',
    'WI Milwaukee': 'Milwaukee',
    'WKU': 'Western Ky.',
    'Western Kentucky': 'Western Ky.',
    'West Georgia': 'West Ga.',
    'Western Carolina': 'Western Caro.',
    'Western Illinois': 'Western Ill.',
    'Western Michigan': 'Western Mich.',
    'Incarnate Word': 'Incarnate Word',
    'Northern Ill.': 'NIU',

    # Second-pass additions
    'Boston Univ': 'Boston U.',
    'Boston University': 'Boston U.',
    'Detroit': 'Detroit Mercy',
    'IUPUI': 'IU Indy',
    'Lamar University': 'Lamar',
    'Loyola-Chicago': 'Loyola Chicago',
    'Missouri KC': 'Kansas City',
    "Mt St Mary's": "Mount St. Mary's",
    'NC Central': 'N.C. Central',
    'North Carolina Central': 'N.C. Central',
    'Northern Iowa': 'UNI',
    'S Carolina St': 'South Carolina St.',
    'Southern Univ': 'Southern U.',
    'UNCG': 'UNC Greensboro',
    'WI Green Bay': 'Green Bay',
}


def try_pattern_match(ext_name):
    """Try pattern-based matching when EXPLICIT doesn't cover it."""
    # 'Foo State' or 'Foo St' → 'Foo St.'
    if re.search(r'\sState$', ext_name):
        candidate = re.sub(r'\sState$', ' St.', ext_name)
        if candidate in our_names:
            return candidate
    if re.search(r'\sSt$', ext_name):
        candidate = ext_name + '.'
        if candidate in our_names:
            return candidate
    # 'St Foo' → 'St. Foo'
    if re.search(r'^St\s', ext_name):
        candidate = re.sub(r'^St\s', 'St. ', ext_name)
        if candidate in our_names:
            return candidate
    return None


# Gather all external names across RPI, Massey, DSR (baseball + softball)
sources = [
    DATA / 'rankings' / 'massey_baseball.csv',
    DATA / 'rankings' / 'massey_softball.csv',
    DATA / 'rankings' / 'dsr_baseball.csv',
    DATA / 'rankings' / 'dsr_softball.csv',
    DATA / 'baseball_rpi_D1.csv',
    DATA / 'softball_rpi_D1.csv',
]

all_ext_names = set()
for src in sources:
    if not src.exists():
        continue
    df = pd.read_csv(src, low_memory=False)
    name_col = next((c for c in ['team', 'teamName', 'team_name'] if c in df.columns), None)
    if not name_col:
        continue
    for n in df[name_col].dropna().unique():
        if isinstance(n, str):
            all_ext_names.add(n.strip())

print(f'Total external names across all sources: {len(all_ext_names)}')

# Build new mappings
new_rows = []
still_missing = []

for ext in sorted(all_ext_names):
    if ext in our_names:
        continue  # matches teams.csv directly, no mapping needed
    if ext in mapped_ext:
        continue  # already mapped
    # Try explicit
    if ext in EXPLICIT:
        ours = EXPLICIT[ext]
        if ours in our_names:
            new_rows.append({'external_name': ext, 'our_name': ours})
            continue
        else:
            print(f'  WARN: EXPLICIT mapping {ext} -> {ours} but {ours} not in teams.csv')
    # Try patterns
    match = try_pattern_match(ext)
    if match:
        new_rows.append({'external_name': ext, 'our_name': match})
        continue
    still_missing.append(ext)

print(f'\nNew mappings to add: {len(new_rows)}')
print(f'Still missing: {len(still_missing)}')
if still_missing:
    print('\nStill missing teams:')
    for m in still_missing:
        print(f'  {m}')

if new_rows:
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([nm, new_df], ignore_index=True).drop_duplicates('external_name')
    combined = combined.sort_values('external_name').reset_index(drop=True)
    combined.to_csv(NAME_MAP_PATH, index=False)
    print(f'\nWrote {NAME_MAP_PATH}: {len(combined)} total mappings')
