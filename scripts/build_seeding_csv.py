"""Build seeding_softball_2026.csv — committee seed + RPI/Massey/DSR/64A ranks
per team for use as an X/Y axis source on the chart_builder.py main page.

Schema:
  team_id, team_name, year, sport, conference, multi_bid_conf,
  committee_seed (1-64), regional_role (1/2/3/4),
  rpi_rank, massey_rank, dsr_rank, sixty_four_rank,
  dev_vs_rpi, dev_vs_massey, dev_vs_dsr, dev_vs_64a
"""
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / 'data'
OUT = DATA / 'seeding_softball_2026.csv'

teams = pd.read_csv(DATA/'teams.csv', encoding='latin-1', dtype=str).fillna('')
sb = teams[teams['sport']=='Softball'].copy()
confs = pd.read_csv(DATA/'conferences.csv', encoding='latin-1', dtype=str).fillna('')
rpi = pd.read_csv(DATA/'softball_rpi_D1.csv', dtype=str, low_memory=False).fillna('')
rpi = rpi[rpi['date']==rpi['date'].max()]
rpi['rank_n'] = pd.to_numeric(rpi['rank'], errors='coerce')
massey = pd.read_csv(DATA/'rankings/massey_softball.csv', dtype=str).fillna('')
massey['rank_n'] = pd.to_numeric(massey['rank'], errors='coerce')
dsr = pd.read_csv(DATA/'rankings/dsr_softball.csv', dtype=str).fillna('')
dsr['rank_n'] = pd.to_numeric(dsr['rank'], errors='coerce')
elo = pd.read_csv(DATA/'rankings/elo_custom_all.csv', dtype=str).fillna('')
elo_sd = elo[(elo['sport']=='softball') & (elo['division']=='D1')].copy()
elo_sd['rank_n'] = pd.to_numeric(elo_sd['rank'], errors='coerce')

# Bracket name → canonical chart-builder name
NAME = {
    'SE LA':'Southeastern La.', 'JAX ST':'Jacksonville St.', 'CSFULL':'Cal St. Fullerton',
    'CALBAP':'California Baptist', 'UNC G':'UNC Greensboro', 'C OF C':'Col. of Charleston',
    "St. Mary's":"St. Mary's (CA)",
}

HOST_SEED = {
    'Alabama':1, 'Texas':2, 'Oklahoma':3, 'Nebraska':4, 'Arkansas':5, 'Florida':6,
    'Tennessee':7, 'UCLA':8, 'Florida St.':9, 'Georgia':10, 'Texas Tech':11, 'Duke':12,
    'Oklahoma St.':13, 'Oregon':14, 'Texas A&M':15, 'LSU':16,
}

REGIONALS = [
    ('Tuscaloosa','Alabama','SE LA','USC Upstate','Belmont'),
    ('Baton Rouge','LSU','Virginia Tech','South Alabama','Akron'),
    ('Tallahassee','Florida St.','UCF','Stetson','JAX ST'),
    ('Los Angeles','UCLA','South Carolina','CSFULL','CALBAP'),
    ('Fayetteville','Arkansas','Washington','Fordham','South Fla.'),
    ('Durham','Duke','Arizona','Marshall','Howard'),
    ('Stillwater','Oklahoma St.','Stanford','Eastern Ill.','Princeton'),
    ('Lincoln','Nebraska','Louisville','Grand Canyon','South Dakota'),
    ('Norman','Oklahoma','Kansas','Binghamton','Michigan'),
    ('Eugene','Oregon','Mississippi St.',"St. Mary's",'Idaho St.'),
    ('Lubbock','Texas Tech','Ole Miss','Marist','Boston U.'),
    ('Gainesville','Florida','Texas St.','Georgia Tech','Florida A&M'),
    ('Knoxville','Tennessee','Virginia','Northern Ky.','Indiana'),
    ('Athens','Georgia','Clemson','UNC G','C OF C'),
    ('College Station','Texas A&M','Arizona St.','UConn','McNeese'),
    ('Austin','Texas','Wisconsin','Baylor','Wagner'),
]


def canon(bracket_name):
    return NAME.get(bracket_name, bracket_name)


def team_id(bracket_name):
    n = canon(bracket_name)
    m = sb[sb['name'] == n]
    if m.empty:
        m = sb[sb['name'].str.contains(n, case=False, na=False, regex=False)]
    return (m.iloc[0]['id'], m.iloc[0]['name'], m.iloc[0]['conference_id']) if not m.empty else (None, n, None)


def rank_from(df, name_col, query, source=None):
    """Look up a team's rank by name, trying multiple variants since each
    ranking source uses a different naming convention (Massey: 'Florida St',
    DSR: 'Florida State', RPI: 'Florida St.')."""
    base = canon(query)
    # Build a list of name variants to try, in priority order
    variants = [base]
    # Replace "St." with various forms
    if 'St.' in base:
        variants.append(base.replace('St.', 'St'))     # Massey: "Florida St"
        variants.append(base.replace('St.', 'State'))  # DSR: "Florida State"
    # Specific overrides where the school's name is fundamentally different
    SPECIAL = {
        'Ole Miss': ['Ole Miss', 'Mississippi'],
        'Mississippi': ['Mississippi', 'Ole Miss'],
        'Texas St.': ['Texas St.', 'Texas St', 'Texas State'],
        'Southeastern La.': ['Southeastern La.', 'Southeastern Louisiana'],
        'SE LA': ['Southeastern La.', 'Southeastern Louisiana'],
        'South Carolina': ['South Carolina', 'S Carolina'],
        'South Florida': ['South Florida', 'South Fla', 'South Fla.'],
        'Cal St. Fullerton': ['Cal St. Fullerton', 'CS Fullerton', 'Cal State Fullerton'],
    }
    if base in SPECIAL:
        for alt in SPECIAL[base]:
            if alt not in variants: variants.append(alt)

    for v in variants:
        m = df[df[name_col].str.replace('(AQ)', '', regex=False).str.strip() == v]
        if not m.empty and pd.notna(m.iloc[0]['rank_n']):
            return int(m.iloc[0]['rank_n'])
    # Last resort: substring on the unique first word + last-name portion
    for v in variants:
        m = df[df[name_col].str.contains(v, case=False, na=False, regex=False)]
        if len(m) == 1 and pd.notna(m.iloc[0]['rank_n']):
            return int(m.iloc[0]['rank_n'])
    return None


def elo_rank(query):
    n = canon(query)
    m = elo_sd[elo_sd['name'] == n]
    if m.empty:
        m = elo_sd[elo_sd['name'].str.contains(n, case=False, na=False, regex=False)]
    if not m.empty and pd.notna(m.iloc[0]['rank_n']):
        return int(m.iloc[0]['rank_n'])
    return None


# Determine 3/4 seeds per regional (higher RPI = 3-seed)
bracket = []
for city, host, two, ua, ub in REGIONALS:
    rA, rB = rank_from(rpi, 'teamName', ua), rank_from(rpi, 'teamName', ub)
    if (rA or 9999) <= (rB or 9999):
        three, four = ua, ub
    else:
        three, four = ub, ua
    bracket.append({'reg': city, 'host': host, 'host_seed': HOST_SEED[host],
                    'two': two, 'three': three, 'four': four})

# Snake assignments: best 3-seed (by RPI) → nat #33; worst → #48. Same for 4-seeds.
three_sorted = sorted([(b['host_seed'], b['three']) for b in bracket],
                       key=lambda x: rank_from(rpi, 'teamName', x[1]) or 9999)
three_nat = {team: 33 + i for i, (hs, team) in enumerate(three_sorted)}
four_sorted = sorted([(b['host_seed'], b['four']) for b in bracket],
                      key=lambda x: rank_from(rpi, 'teamName', x[1]) or 9999)
four_nat = {team: 49 + i for i, (hs, team) in enumerate(four_sorted)}

# Build records
records = []
all_picks = []  # (bracket_name, nat_seed, role)
for b in bracket:
    all_picks.append((b['host'], b['host_seed'], 1))
    all_picks.append((b['two'], 33 - b['host_seed'], 2))
    all_picks.append((b['three'], three_nat[b['three']], 3))
    all_picks.append((b['four'], four_nat[b['four']], 4))

# Multi-bid conference detection
conf_counts = {}
for bn, _, _ in all_picks:
    _, _, cid = team_id(bn)
    if cid: conf_counts[cid] = conf_counts.get(cid, 0) + 1
multi_bid_conf_ids = {cid for cid, n in conf_counts.items() if n >= 2}

for bn, seed, role in all_picks:
    tid, name, cid = team_id(bn)
    conf_name = ''
    if cid:
        c = confs[confs['id'] == cid]
        conf_name = c.iloc[0]['name'] if not c.empty else ''
    rpi_r = rank_from(rpi, 'teamName', bn)
    mas_r = rank_from(massey, 'team', bn)
    dsr_r = rank_from(dsr, 'team', bn)
    elo_r = elo_rank(bn)
    rec = {
        'team_id': tid or '',
        'team_name': name,
        'sport': 'softball',
        'year': '2026',
        'conference': conf_name,
        'multi_bid_conf': 'TRUE' if cid in multi_bid_conf_ids else 'FALSE',
        'committee_seed': seed,
        'regional_role': role,
        'rpi_rank': rpi_r if rpi_r else '',
        'massey_rank': mas_r if mas_r else '',
        'dsr_rank': dsr_r if dsr_r else '',
        'sixty_four_rank': elo_r if elo_r else '',
        'dev_vs_rpi': (seed - rpi_r) if rpi_r else '',
        'dev_vs_massey': (seed - mas_r) if mas_r else '',
        'dev_vs_dsr': (seed - dsr_r) if dsr_r else '',
        'dev_vs_64a': (seed - elo_r) if elo_r else '',
    }
    records.append(rec)

# Filter to multi-bid conferences only — single-bid auto-qualifiers from low confs
# are outliers in ranking-vs-seed analysis and dilute the chart.
df = pd.DataFrame(records)
df = df[df['multi_bid_conf'] == 'TRUE'].copy()
df = df.sort_values('committee_seed').reset_index(drop=True)
df.to_csv(OUT, index=False, encoding='utf-8')
print(f'Wrote {OUT.name}: {len(df)} rows, {len(df.columns)} cols  (multi-bid only)')
print(df.head(8).to_string(index=False))
