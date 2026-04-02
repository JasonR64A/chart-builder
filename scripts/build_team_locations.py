#!/usr/bin/env python3
"""
Build team_locations.csv for NCAA D1 baseball and softball teams.
Matches team names from teams.csv to IPEDS institution data for lat/lon coordinates.
"""
import csv
import re
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEAMS_CSV = os.path.join(BASE_DIR, "data", "teams.csv")
CONFERENCES_CSV = os.path.join(BASE_DIR, "data", "conferences.csv")
IPEDS_CSV = os.path.join(BASE_DIR, "data", "bracketology", "ipeds_locations_raw.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "bracketology", "team_locations.csv")

# ── Explicit mappings: NCAA team name -> IPEDS NAME (exact or substring) ──────
EXPLICIT_MAP = {
    "A&M-Corpus Christi": "Texas A & M University-Corpus Christi",
    "Abilene Christian": "Abilene Christian University",
    "Air Force": "United States Air Force Academy",
    "Akron": "University of Akron Main Campus",
    "Alabama": "The University of Alabama",
    "Alabama A&M": "Alabama A & M University",
    "Alabama St.": "Alabama State University",
    "Albany": "SUNY at Albany",
    "Alcorn": "Alcorn State University",
    "App State": "Appalachian State University",
    "Arizona": "University of Arizona",
    "Arizona St.": "Arizona State University",
    "Arkansas": "University of Arkansas",
    "Arkansas St.": "Arkansas State University",
    "Ark.-Pine Bluff": "University of Arkansas at Pine Bluff",
    "Arkansas-Pine Bluff": "University of Arkansas at Pine Bluff",
    "Army West Point": "United States Military Academy",
    "Auburn": "Auburn University",
    "Austin Peay": "Austin Peay State University",
    "BYU": "Brigham Young University-Provo",
    "Ball St.": "Ball State University",
    "Baylor": "Baylor University",
    "Bellarmine": "Bellarmine University",
    "Belmont": "Belmont University",
    "Bethune-Cookman": "Bethune-Cookman University",
    "Binghamton": "Binghamton University",
    "Boise St.": "Boise State University",
    "Boston College": "Boston College",
    "Boston U.": "Boston University",
    "Bowling Green": "Bowling Green State University",
    "Bradley": "Bradley University",
    "Brown": "Brown University",
    "Bryant": "Bryant University",
    "Bucknell": "Bucknell University",
    "Buffalo": "University at Buffalo",
    "Butler": "Butler University",
    "Cal Baptist": "California Baptist University",
    "Cal Poly": "California Polytechnic State University-San Luis Obispo",
    "Cal St. Bakersfield": "California State University-Bakersfield",
    "Cal St. Fullerton": "California State University-Fullerton",
    "Cal St. Northridge": "California State University-Northridge",
    "CSU Bakersfield": "California State University-Bakersfield",
    "CSU Fullerton": "California State University-Fullerton",
    "CSU Northridge": "California State University-Northridge",
    "Campbell": "Campbell University",
    "Canisius": "Canisius College",
    "Central Ark.": "University of Central Arkansas",
    "Central Conn. St.": "Central Connecticut State University",
    "Central Conn.": "Central Connecticut State University",
    "Central Mich.": "Central Michigan University",
    "Charleston": "College of Charleston",
    "Charleston So.": "Charleston Southern University",
    "Charlotte": "University of North Carolina at Charlotte",
    "Chattanooga": "The University of Tennessee-Chattanooga",
    "Chicago St.": "Chicago State University",
    "Cincinnati": "University of Cincinnati",
    "Citadel": "Citadel Military College of South Carolina",
    "The Citadel": "Citadel Military College of South Carolina",
    "Clemson": "Clemson University",
    "Cleveland St.": "Cleveland State University",
    "Coastal Carolina": "Coastal Carolina University",
    "Colgate": "Colgate University",
    "Colorado": "University of Colorado Boulder",
    "Colorado St.": "Colorado State University",
    "Columbia": "Columbia University in the City of New York",
    "Connecticut": "University of Connecticut",
    "Coppin St.": "Coppin State University",
    "Cornell": "Cornell University",
    "Creighton": "Creighton University",
    "Dallas Baptist": "Dallas Baptist University",
    "Dartmouth": "Dartmouth College",
    "Davidson": "Davidson College",
    "Dayton": "University of Dayton",
    "Delaware": "University of Delaware",
    "Delaware St.": "Delaware State University",
    "DePaul": "DePaul University",
    "Drake": "Drake University",
    "Drexel": "Drexel University",
    "Duke": "Duke University",
    "Duquesne": "Duquesne University",
    "East Carolina": "East Carolina University",
    "Eastern Ill.": "Eastern Illinois University",
    "Eastern Ky.": "Eastern Kentucky University",
    "Eastern Mich.": "Eastern Michigan University",
    "Eastern Wash.": "Eastern Washington University",
    "Elon": "Elon University",
    "Evansville": "University of Evansville",
    "ETSU": "East Tennessee State University",
    "Fairfield": "Fairfield University",
    "Fairleigh Dickinson": "Fairleigh Dickinson University-Metropolitan Campus",
    "FDU": "Fairleigh Dickinson University-Metropolitan Campus",
    "FGCU": "Florida Gulf Coast University",
    "FIU": "Florida International University",
    "Fla. Atlantic": "Florida Atlantic University",
    "Florida": "University of Florida",
    "Florida A&M": "Florida Agricultural and Mechanical University",
    "Florida St.": "Florida State University",
    "Fordham": "Fordham University",
    "Fresno St.": "California State University-Fresno",
    "Furman": "Furman University",
    "Ga. Southern": "Georgia Southern University",
    "Gardner-Webb": "Gardner-Webb University",
    "George Mason": "George Mason University",
    "George Washington": "George Washington University",
    "Georgetown": "Georgetown University",
    "Georgia": "University of Georgia",
    "Georgia St.": "Georgia State University",
    "Georgia Tech": "Georgia Institute of Technology",
    "Gonzaga": "Gonzaga University",
    "Grambling": "Grambling State University",
    "Grand Canyon": "Grand Canyon University",
    "Hartford": "University of Hartford",
    "Harvard": "Harvard University",
    "Hawai'i": "University of Hawaii at Manoa",
    "Hawaii": "University of Hawaii at Manoa",
    "High Point": "High Point University",
    "Hofstra": "Hofstra University",
    "Holy Cross": "College of the Holy Cross",
    "Houston": "University of Houston",
    "Houston Christian": "Houston Baptist University",
    "Idaho": "University of Idaho",
    "Idaho St.": "Idaho State University",
    "Illinois": "University of Illinois at Urbana-Champaign",
    "Illinois St.": "Illinois State University",
    "Incarnate Word": "University of the Incarnate Word",
    "Indiana": "Indiana University-Bloomington",
    "Indiana St.": "Indiana State University",
    "Iona": "Iona University",
    "Iowa": "University of Iowa",
    "Iowa St.": "Iowa State University",
    "IUPUI": "Indiana University-Purdue University-Indianapolis",
    "IU Indy": "Indiana University-Purdue University-Indianapolis",
    "Jackson St.": "Jackson State University",
    "Jacksonville": "Jacksonville University",
    "Jacksonville St.": "Jacksonville State University",
    "James Madison": "James Madison University",
    "Kansas": "University of Kansas",
    "Kansas City": "University of Missouri-Kansas City",
    "Kansas St.": "Kansas State University",
    "Kennesaw St.": "Kennesaw State University",
    "Kent St.": "Kent State University at Kent",
    "Kentucky": "University of Kentucky",
    "La Salle": "La Salle University",
    "Lafayette": "Lafayette College",
    "Lamar": "Lamar University",
    "Le Moyne": "Le Moyne College",
    "Lehigh": "Lehigh University",
    "Liberty": "Liberty University",
    "Lindenwood": "Lindenwood University",
    "Lipscomb": "Lipscomb University",
    "Little Rock": "University of Arkansas at Little Rock",
    "LIU": "Long Island University",
    "LMU (CA)": "Loyola Marymount University",
    "Long Beach St.": "California State University-Long Beach",
    "Longwood": "Longwood University",
    "Louisiana": "University of Louisiana at Lafayette",
    "Louisiana Tech": "Louisiana Tech University",
    "Louisville": "University of Louisville",
    "Loyola Chicago": "Loyola University Chicago",
    "Loyola Maryland": "Loyola University Maryland",
    "Loyola Marymount": "Loyola Marymount University",
    "LSU": "Louisiana State University and Agricultural & Mechanical College",
    "Maine": "University of Maine",
    "Manhattan": "Manhattan College",
    "Marist": "Marist College",
    "Marquette": "Marquette University",
    "Marshall": "Marshall University",
    "Maryland": "University of Maryland-College Park",
    "McNeese": "McNeese State University",
    "Memphis": "University of Memphis",
    "Mercer": "Mercer University",
    "Merrimack": "Merrimack College",
    "Miami": "University of Miami",
    "Miami (OH)": "Miami University-Oxford",
    "Michigan": "University of Michigan-Ann Arbor",
    "Michigan St.": "Michigan State University",
    "Middle Tenn.": "Middle Tennessee State University",
    "Milwaukee": "University of Wisconsin-Milwaukee",
    "Minnesota": "University of Minnesota-Twin Cities",
    "Mississippi St.": "Mississippi State University",
    "Mississippi Val.": "Mississippi Valley State University",
    "Missouri": "University of Missouri-Columbia",
    "Missouri St.": "Missouri State University",
    "Monmouth": "Monmouth University",
    "Montana": "University of Montana",
    "Montana St.": "Montana State University",
    "Morehead St.": "Morehead State University",
    "Morgan St.": "Morgan State University",
    "Mount St. Mary's": "Mount St Mary's University",
    "Mt. St. Mary's": "Mount St Mary's University",
    "Murray St.": "Murray State University",
    "N.C. A&T": "North Carolina A & T State University",
    "N.C. Central": "North Carolina Central University",
    "Navy": "United States Naval Academy",
    "NC State": "North Carolina State University at Raleigh",
    "Nebraska": "University of Nebraska-Lincoln",
    "Nevada": "University of Nevada-Reno",
    "New Hampshire": "University of New Hampshire-Main Campus",
    "New Mexico": "University of New Mexico",
    "New Mexico St.": "New Mexico State University",
    "New Orleans": "University of New Orleans",
    "Niagara": "Niagara University",
    "Nicholls": "Nicholls State University",
    "NJIT": "New Jersey Institute of Technology",
    "Norfolk St.": "Norfolk State University",
    "North Alabama": "University of North Alabama",
    "North Carolina": "University of North Carolina at Chapel Hill",
    "North Dakota": "University of North Dakota",
    "North Dakota St.": "North Dakota State University",
    "North Florida": "University of North Florida",
    "Northern Colo.": "University of Northern Colorado",
    "Northern Ill.": "Northern Illinois University",
    "Northern Ky.": "Northern Kentucky University",
    "Northwestern": "Northwestern University",
    "Northwestern St.": "Northwestern State University of Louisiana",
    "Notre Dame": "University of Notre Dame",
    "Oakland": "Oakland University",
    "Ohio": "Ohio University-Main Campus",
    "Ohio St.": "Ohio State University-Main Campus",
    "Oklahoma": "University of Oklahoma-Norman Campus",
    "Oklahoma St.": "Oklahoma State University",
    "Old Dominion": "Old Dominion University",
    "Ole Miss": "University of Mississippi",
    "Omaha": "University of Nebraska at Omaha",
    "Oral Roberts": "Oral Roberts University",
    "Oregon": "University of Oregon",
    "Oregon St.": "Oregon State University",
    "Pacific": "University of the Pacific",
    "Penn": "University of Pennsylvania",
    "Penn St.": "Pennsylvania State University-Main Campus",
    "Pepperdine": "Pepperdine University",
    "Pittsburgh": "University of Pittsburgh-Pittsburgh Campus",
    "Portland": "University of Portland",
    "Portland St.": "Portland State University",
    "Prairie View": "Prairie View A & M University",
    "Prairie View A&M": "Prairie View A & M University",
    "Presbyterian": "Presbyterian College",
    "Princeton": "Princeton University",
    "Providence": "Providence College",
    "Purdue": "Purdue University-Main Campus",
    "Purdue Fort Wayne": "Purdue University Fort Wayne",
    "Queens (NC)": "Queens University of Charlotte",
    "Quinnipiac": "Quinnipiac University",
    "Radford": "Radford University",
    "Rhode Island": "University of Rhode Island",
    "Rice": "Rice University",
    "Richmond": "University of Richmond",
    "Rider": "Rider University",
    "Robert Morris": "Robert Morris University",
    "Rutgers": "Rutgers University-New Brunswick",
    "SIU Edwardsville": "Southern Illinois University-Edwardsville",
    "SIUE": "Southern Illinois University-Edwardsville",
    "Sacramento St.": "California State University-Sacramento",
    "Sacred Heart": "Sacred Heart University",
    "Saint Joseph's": "Saint Joseph's University",
    "Saint Louis": "Saint Louis University",
    "Sam Houston": "Sam Houston State University",
    "Samford": "Samford University",
    "San Diego": "University of San Diego",
    "San Diego St.": "San Diego State University",
    "San Francisco": "University of San Francisco",
    "San Jose St.": "San Jose State University",
    "Santa Clara": "Santa Clara University",
    "Seattle U": "Seattle University",
    "Seton Hall": "Seton Hall University",
    "Siena": "Siena College",
    "South Alabama": "University of South Alabama",
    "South Carolina": "University of South Carolina-Columbia",
    "South Carolina St.": "South Carolina State University",
    "South Carolina Upstate": "University of South Carolina-Upstate",
    "South Dakota": "University of South Dakota",
    "South Dakota St.": "South Dakota State University",
    "South Fla.": "University of South Florida",
    "SE Missouri St.": "Southeast Missouri State University",
    "Southeast Mo. St.": "Southeast Missouri State University",
    "SE Louisiana": "Southeastern Louisiana University",
    "Southeastern La.": "Southeastern Louisiana University",
    "Southern": "Southern University and A & M College",
    "Southern Ill.": "Southern Illinois University-Carbondale",
    "Southern Miss": "University of Southern Mississippi",
    "Southern U.": "Southern University and A & M College",
    "Southern Utah": "Southern Utah University",
    "SFA": "Stephen F Austin State University",
    "Stephen F. Austin": "Stephen F Austin State University",
    "St. Bonaventure": "St Bonaventure University",
    "St. John's (NY)": "St John's University-New York",
    "St. Thomas (MN)": "University of St Thomas",
    "Stanford": "Stanford University",
    "Stetson": "Stetson University",
    "Stonehill": "Stonehill College",
    "Stony Brook": "Stony Brook University",
    "Syracuse": "Syracuse University",
    "Tarleton St.": "Tarleton State University",
    "TCU": "Texas Christian University",
    "Temple": "Temple University",
    "Tennessee": "The University of Tennessee-Knoxville",
    "Tennessee St.": "Tennessee State University",
    "Tennessee Tech": "Tennessee Technological University",
    "Texas": "The University of Texas at Austin",
    "Texas A&M": "Texas A & M University-College Station",
    "Texas St.": "Texas State University",
    "Texas Southern": "Texas Southern University",
    "Texas Tech": "Texas Tech University",
    "Toledo": "University of Toledo",
    "Towson": "Towson University",
    "Troy": "Troy University",
    "Tulane": "Tulane University of Louisiana",
    "Tulsa": "University of Tulsa",
    "UAB": "University of Alabama at Birmingham",
    "UC Davis": "University of California-Davis",
    "UC Irvine": "University of California-Irvine",
    "UC Riverside": "University of California-Riverside",
    "UC San Diego": "University of California-San Diego",
    "UC Santa Barbara": "University of California-Santa Barbara",
    "UCF": "University of Central Florida",
    "UCLA": "University of California-Los Angeles",
    "UConn": "University of Connecticut",
    "UIC": "University of Illinois at Chicago",
    "UMBC": "University of Maryland-Baltimore County",
    "UMKC": "University of Missouri-Kansas City",
    "UMass": "University of Massachusetts-Amherst",
    "UMass Lowell": "University of Massachusetts-Lowell",
    "UNC Asheville": "University of North Carolina at Asheville",
    "UNC Greensboro": "University of North Carolina at Greensboro",
    "UNC Wilmington": "University of North Carolina Wilmington",
    "UNCW": "University of North Carolina Wilmington",
    "UNLV": "University of Nevada-Las Vegas",
    "UNI": "University of Northern Iowa",
    "UNF": "University of North Florida",
    "USC": "University of Southern California",
    "USC Upstate": "University of South Carolina-Upstate",
    "UT Arlington": "The University of Texas at Arlington",
    "UT Martin": "The University of Tennessee-Martin",
    "UT Rio Grande Valley": "The University of Texas Rio Grande Valley",
    "UTEP": "The University of Texas at El Paso",
    "UTSA": "The University of Texas at San Antonio",
    "Utah": "University of Utah",
    "Utah St.": "Utah State University",
    "Utah Tech": "Dixie State University",
    "Utah Valley": "Utah Valley University",
    "Valparaiso": "Valparaiso University",
    "Vanderbilt": "Vanderbilt University",
    "VCU": "Virginia Commonwealth University",
    "Vermont": "University of Vermont",
    "Villanova": "Villanova University",
    "Virginia": "University of Virginia-Main Campus",
    "Virginia Tech": "Virginia Polytechnic Institute and State University",
    "VMI": "Virginia Military Institute",
    "Wagner": "Wagner College",
    "Wake Forest": "Wake Forest University",
    "Washington": "University of Washington-Seattle Campus",
    "Washington St.": "Washington State University",
    "Weber St.": "Weber State University",
    "West Virginia": "West Virginia University",
    "Western Carolina": "Western Carolina University",
    "Western Ill.": "Western Illinois University",
    "Western Ky.": "Western Kentucky University",
    "Western Mich.": "Western Michigan University",
    "Wichita St.": "Wichita State University",
    "William & Mary": "William & Mary",
    "Winthrop": "Winthrop University",
    "Wisconsin": "University of Wisconsin-Madison",
    "Wofford": "Wofford College",
    "Wright St.": "Wright State University-Main Campus",
    "Wyoming": "University of Wyoming",
    "Xavier": "Xavier University",
    "Yale": "Yale University",
    "Youngstown St.": "Youngstown State University",
    # Additional common abbreviation patterns
    "Central Fla.": "University of Central Florida",
    "Fla. Gulf Coast": "Florida Gulf Coast University",
    "Fla. Int'l": "Florida International University",
    "South Fla. Bulls": "University of South Florida",
    "UTA": "The University of Texas at Arlington",
    "UTRGV": "The University of Texas Rio Grande Valley",
    "Wis.-Milwaukee": "University of Wisconsin-Milwaukee",
    "Green Bay": "University of Wisconsin-Green Bay",
    "SIU": "Southern Illinois University-Carbondale",
    "UCSB": "University of California-Santa Barbara",
    "N.C. State": "North Carolina State University at Raleigh",
    "Tex. A&M-CC": "Texas A & M University-Corpus Christi",
    "UMES": "University of Maryland Eastern Shore",
    "Md.-Eastern Shore": "University of Maryland Eastern Shore",
    "St. Peter's": "Saint Peter's University",
    "Stony Brook": "Stony Brook University",
    "Binghamton": "Binghamton University",
    "UAlbany": "SUNY at Albany",
    "CCSU": "Central Connecticut State University",
    "SEMO": "Southeast Missouri State University",
    "SLU": "Saint Louis University",
    "SMU": "Southern Methodist University",
    "Neb. Omaha": "University of Nebraska at Omaha",
    "NIU": "Northern Illinois University",
    "North Texas": "University of North Texas",
    "UTSA": "The University of Texas at San Antonio",
    "Wis.-Green Bay": "University of Wisconsin-Green Bay",
    "Texas-Arlington": "The University of Texas at Arlington",
    "Nicholls St.": "Nicholls State University",
    "McNeese St.": "McNeese State University",
    # ── Additional mappings from unmatched analysis ──
    "Col. of Charleston": "College of Charleston",
    "CSUN": "California State University-Northridge",
    "DBU": "Dallas Baptist University",
    "East Texas A&M": "Texas A & M University-Commerce",
    "Fla. Memorial": "Florida Memorial University",
    "Kent St.-Tuscarawas": "Kent State University at Tuscarawas",
    "Loyola (LA)": "Loyola University New Orleans",
    "LSU-Alexandria": "Louisiana State University-Alexandria",
    "Miami (FL)": "University of Miami",
    "Miami-Hamilton": "Miami University-Hamilton",
    "Middle Ga. St.": "Middle Georgia State University",
    "Northwestern (IA)": "Northwestern College",
    "Oregon Tech": "Oregon Institute of Technology",
    "Penn St.-Beaver": "Pennsylvania State University-Penn State Beaver",
    "Penn St.-Brandywine": "Pennsylvania State University-Penn State Brandywine",
    "Penn St.-Du Bois": "Pennsylvania State University-Penn State DuBois",
    "Penn St.-Fayette": "Pennsylvania State University-Penn State Fayette- Eberly",
    "Penn St.-Gr Allegheny": "Pennsylvania State University-Penn State Greater Allegheny",
    "Penn St.-Hazleton": "Pennsylvania State University-Penn State Hazleton",
    "Penn St.-New Kens.": "Pennsylvania State University-Penn State New Kensington",
    "Penn St.-Schuylkill": "Pennsylvania State University-Penn State Schuylkill",
    "Penn St.-Scranton": "Pennsylvania State University-Penn State Scranton",
    "Penn St.-Shenango": "Pennsylvania State University-Penn State Shenango",
    "Penn St.-Wilkes Barre": "Pennsylvania State University-Penn State Wilkes-Barre",
    "Penn St.-York": "Pennsylvania State University-Penn State York",
    "Saint Mary's (CA)": "Saint Mary's College of California",
    "Southeastern (FL)": "Southeastern University",
    "Southern-N.O.": "Southern University at New Orleans",
    "St. Francis (IL)": "University of St Francis",
    "St. Xavier": "Saint Xavier University",
    "St. Thomas (FL)": "St. Thomas University",
    "Tex. A&M-Texarkana": "Texas A&M University-Texarkana",
    "Thomas (GA)": "Thomas University",
    "UC Clermont": "University of Cincinnati-Clermont College",
    "UIW": "University of the Incarnate Word",
    "ULM": "University of Louisiana at Monroe",
    "Union (KY)": "Union College",
    "West Ga.": "University of West Georgia",
    "Wright St.-Lake": "Wright State University-Lake Campus",
    "Xavier (LA)": "Xavier University of Louisiana",
    "Okla. Panhandle": "Oklahoma Panhandle State University",
    "Concordia (MI)": "Concordia University-Ann Arbor",
    "Benedictine (KS)": "Benedictine College",
    "Bethel (TN)": "SKIP",  # Bethel University, McKenzie, TN
    "Rochester Univ. (MI)": "Rochester University",
    "Southwestern (KS)": "SKIP",  # Southwestern College, Winfield, KS
    "La. Christian": "Louisiana College",
    "Hannibal-La Grange": "Hannibal-LaGrange University",
    "Science & Arts Okla.": "University of Science and Arts of Oklahoma",
    "Mid-Amer. Christian": "Mid-America Christian University",
    "Indiana Tech": "Indiana Institute of Technology",
    "Judson (IL)": "Judson University",
    "UHSP": "University of Health Sciences and Pharmacy in St. Louis",
    "West Va. Tech": "West Virginia University Institute of Technology",
    "New College (FL)": "New College of Florida",
    "Ark. Baptist": "Arkansas Baptist College",
    "Grace Christian (MI)": "Grace Christian University",
    "Blue Mountain (MS)": "Blue Mountain College",
    "Hope Int'l": "Hope International University",
    "Simpson (CA)": "Simpson University",
    "Bethesda (CA)": "Bethesda University",
    "Bry. & Strat. (Alb.)": "Bryant & Stratton College-Albany",
    "Bry. & Strat. (Syr.)": "Bryant & Stratton College-Syracuse North",
    "SAGU American Indian": "Southwestern Assemblies of God University",
    "St. Katherine": "SKIP",  # Small school, no clear IPEDS match
    "CarolinaU": "SKIP",  # Very small school
    "IUPUC": "SKIP",  # Regional campus, use Columbus IN coords
    "Col. of Idaho": "The College of Idaho",
    "Tenn. Southern": "SKIP",  # New school, not in IPEDS
    "British Colum. (CAN)": "SKIP",  # Canadian school
    "Apprentice": "SKIP",  # Apprentice School, Newport News VA - trade school
    "TBA": "SKIP",  # Placeholder
}

# ── Manual fallback coordinates for teams not in IPEDS ──────
MANUAL_COORDS = {
    "St. Katherine": {"city": "San Marcos", "state": "CA", "lat": "33.1434", "lon": "-117.1661"},
    "CarolinaU": {"city": "Winston-Salem", "state": "NC", "lat": "36.0999", "lon": "-80.2442"},
    "IUPUC": {"city": "Columbus", "state": "IN", "lat": "39.2014", "lon": "-85.9214"},
    "Bethel (TN)": {"city": "McKenzie", "state": "TN", "lat": "36.137848", "lon": "-88.516762"},
    "Southwestern (KS)": {"city": "Winfield", "state": "KS", "lat": "37.248608", "lon": "-96.975405"},
    "Tenn. Southern": {"city": "Pulaski", "state": "TN", "lat": "35.1998", "lon": "-87.0306"},
    "British Colum. (CAN)": {"city": "Vancouver", "state": "BC", "lat": "49.2606", "lon": "-123.2460"},
    "Apprentice": {"city": "Newport News", "state": "VA", "lat": "36.9768", "lon": "-76.4300"},
    "TBA": {"city": "TBA", "state": "TBA", "lat": "0", "lon": "0"},
}


def load_d1_conference_ids(path):
    """Return set of conference IDs where division == 'D-I'."""
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["division"] == "D-I":
                ids.add(row["id"])
    return ids


def load_d1_team_names(path, d1_conf_ids):
    """Return sorted list of unique D-I Baseball/Softball team names."""
    names = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["conference_id"] in d1_conf_ids and row["sport"] in ("Baseball", "Softball"):
                names.add(row["name"])
    return sorted(names)


def load_ipeds(path):
    """Return list of dicts with NAME, CITY, STATE, LAT, LON from IPEDS."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "name": row["NAME"].strip(),
                "city": row["CITY"].strip(),
                "state": row["STATE"].strip(),
                "lat": row["LAT"].strip(),
                "lon": row["LON"].strip(),
            })
    return records


def normalize(s):
    """Normalize string for fuzzy comparison."""
    s = s.lower().strip()
    s = re.sub(r"[''`]", "'", s)
    s = re.sub(r"[^a-z0-9\s&\-']", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def try_match(team_name, ipeds_records, ipeds_norm_map):
    """Try to match a team name to an IPEDS record. Returns dict, 'SKIP', or None."""

    # 0. Check for manual coordinates
    if team_name in MANUAL_COORDS:
        return MANUAL_COORDS[team_name]

    # 1. Explicit mapping
    if team_name in EXPLICIT_MAP:
        mapped = EXPLICIT_MAP[team_name]
        if mapped == "SKIP":
            return "SKIP"
        target = normalize(mapped)
        if target in ipeds_norm_map:
            return ipeds_norm_map[target]
        # Try partial match on the mapped name
        for norm_name, record in ipeds_norm_map.items():
            if target in norm_name:
                return record

    # 2. Direct name match (normalized)
    tn = normalize(team_name)
    if tn in ipeds_norm_map:
        return ipeds_norm_map[tn]

    # 3. Try "University of X" pattern
    for prefix in ["university of ", "the university of "]:
        test = normalize(prefix + team_name)
        if test in ipeds_norm_map:
            return ipeds_norm_map[test]

    # 4. Try "X University" pattern
    test = normalize(team_name + " University")
    if test in ipeds_norm_map:
        return ipeds_norm_map[test]

    # 5. Try "X State University" pattern
    test = normalize(team_name + " State University")
    if test in ipeds_norm_map:
        return ipeds_norm_map[test]

    # 6. Handle "St." -> "State" transformations for team names ending in "St."
    if team_name.endswith(" St."):
        base = team_name[:-4]
        for suffix in [" State University", " State University at " + base]:
            test = normalize(base + suffix)
            if test in ipeds_norm_map:
                return ipeds_norm_map[test]

    # 7. Substring matching - find IPEDS names that contain the team name
    best_match = None
    best_score = 0
    for norm_name, record in ipeds_norm_map.items():
        # Check if all significant words of team name appear in IPEDS name
        if len(tn) > 3 and tn in norm_name:
            score = len(tn) / len(norm_name)  # Prefer shorter IPEDS names (more specific)
            if score > best_score:
                best_score = score
                best_match = record

    if best_match and best_score > 0.3:
        return best_match

    return None


def main():
    d1_conf_ids = load_d1_conference_ids(CONFERENCES_CSV)
    team_names = load_d1_team_names(TEAMS_CSV, d1_conf_ids)
    ipeds = load_ipeds(IPEDS_CSV)

    # Build normalized lookup
    ipeds_norm_map = {}
    for rec in ipeds:
        key = normalize(rec["name"])
        if key not in ipeds_norm_map:
            ipeds_norm_map[key] = rec

    print(f"D-I conferences: {len(d1_conf_ids)}")
    print(f"Unique D-I team names: {len(team_names)}")
    print(f"IPEDS records: {len(ipeds)}")

    matched = []
    unmatched = []

    for tn in team_names:
        result = try_match(tn, ipeds, ipeds_norm_map)
        if result == "SKIP":
            # Check if there are manual coordinates for this team
            if tn in MANUAL_COORDS:
                mc = MANUAL_COORDS[tn]
                matched.append({
                    "team_name": tn,
                    "city": mc["city"],
                    "state": mc["state"],
                    "lat": mc["lat"],
                    "lon": mc["lon"],
                })
            else:
                unmatched.append(tn)
        elif result:
            matched.append({
                "team_name": tn,
                "city": result["city"],
                "state": result["state"],
                "lat": result["lat"],
                "lon": result["lon"],
            })
        else:
            unmatched.append(tn)

    print(f"\nMatched: {len(matched)}")
    print(f"Unmatched: {len(unmatched)}")
    if unmatched:
        print("Unmatched teams:")
        for u in unmatched:
            print(f"  - {u}")

    # Write output
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["team_name", "city", "state", "lat", "lon"])
        writer.writeheader()
        for row in matched:
            writer.writerow(row)

    print(f"\nWrote {len(matched)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
