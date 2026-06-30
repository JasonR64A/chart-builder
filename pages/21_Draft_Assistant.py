"""Draft Day Live-Stream Assistant — per-player scouting card for baseball.

Search a player; get a broadcast-ready blurb in four parts:
  1. Overview      — career OPS, games, schools, transfer status, XBH, superlatives
  2. Big weekend   — his best 3-game series (precomputed from PBP -> draft_best_series.csv)
  3. Something to watch — a metric below the 60th percentile in his PREDOMINANT role
                      (hitting vs pitching decided by volume, so we don't ding a hitter
                       for 2 IP of mop-up)
  4. Handedness    — where he ranks within his own bat/throw peer group

Everything reads committed data so it runs on Render. The big-weekend series is
precomputed locally by scripts/build_draft_assistant.py (Render has no per-game PBP).
"""
import csv
from pathlib import Path
import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parent.parent / "data"
st.set_page_config(page_title="Draft Assistant", layout="wide")


def norm(s):
    s = str(s).strip()
    return s[:-2] if s.endswith(".0") else s


@st.cache_data(show_spinner=False)
def load(name, **kw):
    return pd.read_csv(DATA / name, dtype=str, encoding="latin-1", keep_default_na=False, **kw)


@st.cache_data(show_spinner=False)
def load_all():
    players = load("players.csv")
    teams = load("teams.csv")
    conf = load("conferences.csv")
    hit = load("hitting.csv")
    pit = load("pitching.csv")
    try:
        series = load("draft_best_series.csv")
    except Exception:
        series = pd.DataFrame()
    try:
        splits = load("draft_splits.csv")
    except Exception:
        splits = pd.DataFrame()
    for df in (hit, pit):
        df["pid"] = df["player_id"].map(norm)
    return players, teams, conf, hit, pit, series, splits


players, teams, conf, hit, pit, series, splits = load_all()
splits_by_pid = {}
if len(splits):
    for r in splits.itertuples():
        splits_by_pid.setdefault(norm(r.player_id), []).append(r)

tname = {norm(r["id"]): r["name"] for _, r in teams.iterrows()}
cdiv = {norm(r["id"]): r["division"] for _, r in conf.iterrows()}
tconf = {norm(r["id"]): norm(r["conference_id"]) for _, r in teams.iterrows()}
series_by_pid = {norm(r.player_id): r for r in series.itertuples()} if len(series) else {}


def f(x):
    try:
        return float(x)
    except Exception:
        return None


def num(df, col):
    return pd.to_numeric(df[col], errors="coerce")


# ---------------- player picker ----------------
st.title("Draft Day Assistant")
st.caption("Live-stream scouting card for any baseball player. Search by name.")

q = st.text_input("Player search", placeholder="Type a player name…").strip().lower()
if not q:
    st.info("Start typing a player name above.")
    st.stop()

matches = players[players["player_name"].str.lower().str.contains(q, na=False)]
# prefer players who actually have stats (draftable), and de-dupe by id
matches = matches.drop_duplicates("id")
if matches.empty:
    st.warning("No players match that name.")
    st.stop()


def _label(r):
    t = tname.get(norm(r["team_id"]), "")
    pos = r.get("position", "")
    return f'{r["player_name"]} — {t} ({pos})' if t else f'{r["player_name"]} ({pos})'


opts = {_label(r): norm(r["id"]) for _, r in matches.head(60).iterrows()}
choice = st.selectbox(f"{len(matches)} match(es)", list(opts))
pid = opts[choice]
prow = players[players["id"].map(norm) == pid].iloc[0]

# ---------------- gather this player's stat rows ----------------
ph = hit[hit["pid"] == pid].copy()
pp = pit[pit["pid"] == pid].copy()
ph["yr"] = num(ph, "year")
pp["yr"] = num(pp, "year")

bat = (prow.get("bat") or "").strip().upper()
throw = (prow.get("throw") or "").strip().upper()
BAT_LABEL = {"L": "left-handed", "R": "right-handed", "S": "switch"}
THROW_LABEL = {"L": "left-handed", "R": "right-handed"}

# career aggregates (hitting)
def csum(df, col):
    return int(num(df, col).fillna(0).sum()) if len(df) else 0

car_ab = csum(ph, "at_bats"); car_h = csum(ph, "hits")
car_2b = csum(ph, "doubles"); car_3b = csum(ph, "triples"); car_hr = csum(ph, "home_runs")
car_bb = csum(ph, "walks"); car_hbp = csum(ph, "hit_by_pitch"); car_sf = csum(ph, "sac_fly")
car_rbi = csum(ph, "runs_batted_in"); car_sb = csum(ph, "stolen_bases"); car_gp = csum(ph, "games_played")
car_tb = car_h - car_2b - car_3b - car_hr + 2 * car_2b + 3 * car_3b + 4 * car_hr
car_xbh = car_2b + car_3b + car_hr
car_pa = car_ab + car_bb + car_hbp + car_sf
car_obp = (car_h + car_bb + car_hbp) / car_pa if car_pa else 0
car_slg = car_tb / car_ab if car_ab else 0
car_ops = round(car_obp + car_slg, 3)
car_avg = round(car_h / car_ab, 3) if car_ab else 0

# career pitching
car_ip = round(num(pp, "innings_pitched").fillna(0).sum(), 1) if len(pp) else 0
car_bf = csum(pp, "batters_faced")
car_k = csum(pp, "strikeouts")

predominant = "pitching" if car_bf > car_pa else "hitting"

# schools played (distinct teams across stat rows, in year order)
school_years = {}
for _, r in pd.concat([ph, pp]).iterrows():
    t = tname.get(norm(r["team_id"]))
    if t:
        school_years.setdefault(t, set()).add(norm(r["year"]))
schools = sorted(school_years, key=lambda t: min(school_years[t]))
transferred = (str(prow.get("transferred", "")).strip().upper() in ("TRUE", "1", "YES")) or len(schools) > 1

# ---------------- 1) OVERVIEW ----------------
team_now = tname.get(norm(prow["team_id"]), "")
hand = f"{BAT_LABEL.get(bat, bat or '?')}/{THROW_LABEL.get(throw, throw or '?')}"
st.markdown(f"## {prow['player_name']}")
st.markdown(
    f"**{prow.get('position','')}** · bats {BAT_LABEL.get(bat, bat or '?')}, throws {THROW_LABEL.get(throw, throw or '?')}"
    + (f" · {prow.get('height','')}" if prow.get('height') else "")
    + (f" · {prow.get('hometown','')}" if prow.get('hometown') else "")
)

c = st.columns(5)
c[0].metric("Career OPS", f"{car_ops:.3f}")
c[1].metric("Games", car_gp or "—")
c[2].metric("Career HR", car_hr)
c[3].metric("Extra-base hits", car_xbh)
c[4].metric("Career AVG", f"{car_avg:.3f}")

over = []
yrs = sorted({norm(y) for y in pd.concat([ph, pp])["year"]})
over.append(f"**{len(schools)} school{'s' if len(schools)!=1 else ''}:** " +
            " → ".join(schools) + (f"  ·  _transfer_" if transferred else "  ·  _no transfers on record_"))
over.append(f"**Seasons in data:** {', '.join(yrs)}")
# superlatives
sup = []
if len(ph):
    best = ph.loc[num(ph, "on_base_plus_slugging").idxmax()] if num(ph, "on_base_plus_slugging").notna().any() else None
    if best is not None:
        sup.append(f"best season OPS **{f(best['on_base_plus_slugging']):.3f}** ({norm(best['year'])} at {tname.get(norm(best['team_id']),'?')})")
    if car_hr >= 30:
        sup.append(f"**{car_hr}** career home runs")
    if car_sb >= 30:
        sup.append(f"**{car_sb}** career steals")
    if car_xbh >= 60:
        sup.append(f"**{car_xbh}** career extra-base hits")
if car_ip >= 20:
    sup.append(f"**{car_ip}** career IP, {car_k} K on the mound")
if sup:
    over.append("**Superlatives:** " + "; ".join(sup) + ".")
st.markdown("\n\n".join(over))

st.divider()

# ---------------- 2) BIG WEEKEND ----------------
st.markdown("### 🔥 Big weekend")
s = series_by_pid.get(pid)
if s is not None:
    kind = "series" if s.kind == "series" else "hot stretch"
    st.success(s.summary)
    st.caption(f"{s.date_start} to {s.date_end} · best 3-game {kind} on record (from per-game PBP)")
else:
    st.caption("No multi-game PBP series on record for this player (PBP covers 2025–2026; "
               "small-school or limited-sample players may not have one).")

st.divider()

# ---------------- 3) SOMETHING TO WATCH (weakness in predominant role) ----------------
st.markdown("### 👀 Something to watch")
HIT_PCT = {
    "percentile_rank_on_base_plus_slugging": "OPS",
    "percentile_rank_weighted_on_base_average": "wOBA",
    "percentile_rank_isolated_power": "ISO (power)",
    "percentile_rank_weighted_runs_created": "wRC",
    "percentile_rank_runs_plate_appearance": "runs/PA",
}
PIT_PCT = {
    "percentile_rank_on_base_plus_slugging_against": "OPS allowed",
    "percentile_rank_strikeout_to_walk_ratio": "K/BB",
    "percentile_rank_walks_plus_hits_per_inning_pitched": "WHIP",
    "percentile_rank_fielding_independent_pitching": "FIP",
    "percentile_rank_skill_interactive_earned_run_average": "SIERA",
}


def latest_with_volume(df, vol_col, min_vol):
    d = df[num(df, vol_col).fillna(0) >= min_vol]
    if not len(d):
        return None
    return d.loc[d["yr"].idxmax()]

weak = []
if predominant == "hitting":
    row = latest_with_volume(ph, "plate_appearances", 50)
    if row is not None:
        for col, lab in HIT_PCT.items():
            v = f(row.get(col))
            if v is not None and v < 0.60:
                weak.append((lab, v))
        ctx = f"{norm(row['year'])} · {int(f(row['plate_appearances']))} PA"
    else:
        ctx = None
else:
    row = latest_with_volume(pp, "innings_pitched", 15)
    if row is not None:
        for col, lab in PIT_PCT.items():
            v = f(row.get(col))
            if v is not None and v < 0.60:
                weak.append((lab, v))
        ctx = f"{norm(row['year'])} · {f(row['innings_pitched']):.0f} IP"
    else:
        ctx = None

if ctx is None:
    st.caption(f"Not enough {predominant} volume to flag a weakness.")
elif weak:
    weak.sort(key=lambda x: x[1])
    st.markdown(f"Below the 60th percentile in his **{predominant}** ({ctx}):")
    st.markdown("\n".join(f"- **{lab}** — {int(round(v*100))}th percentile" for lab, v in weak[:4]))
else:
    st.markdown(f"No clear weakness — every tracked {predominant} metric is at or above the 60th percentile ({ctx}). "
                "That's a positive note for the broadcast.")

# --- 2026 split notes (peer-relative, volume-gated; both directions) ---
SPLIT_LABEL = {"conference": "vs conference", "road": "on the road", "home": "at home",
               "late": "late-season (May/Jun)", "quality": "vs power-conference foes"}
LO, HI = 15, 85   # extreme tails


def _i(x):
    try:
        return int(float(x))
    except Exception:
        return None

# this player's overall 2026 power rate, for self-baseline contrasts
ph26 = ph[ph["yr"] == 2026]
ab26 = int(num(ph26, "at_bats").fillna(0).sum())
hr26 = int(num(ph26, "home_runs").fillna(0).sum())
ovr_hr_rate = hr26 / ab26 if ab26 else 0.0

split_watch, split_good, seen = [], [], set()
for r in splits_by_pid.get(pid, []):
    lab = SPLIT_LABEL.get(r.split, r.split)
    opc, isc = _i(r.ops_pct), _i(r.iso_pct)
    ops, iso = r.ops, r.iso
    pa, ab, hr = _i(r.pa), _i(r.ab), _i(r.hr)
    # 1) peer-outlier weakness: overall (OPS) first, else power (ISO)
    if opc is not None and opc <= LO:
        split_watch.append((opc, f"**{lab}**: {ops} OPS — {opc}th pct ({pa} PA)")); seen.add(r.split)
    elif isc is not None and isc <= LO:
        split_watch.append((isc, f"**{lab}** power: {hr} HR in {ab} AB ({iso} ISO, {isc}th pct)")); seen.add(r.split)
    # 2) self-baseline power contrast (e.g. 'only 2 HR vs conference' vs his own norm).
    #    Skip if this split is actually a strength (don't nitpick HR on an elite split).
    elif (ab and ab >= 60 and ovr_hr_rate >= 0.018 and ab26 >= 80 and hr is not None
          and not (opc is not None and opc >= HI)
          and (hr / ab) <= 0.70 * ovr_hr_rate):
        per_o = round(ab26 / hr26) if hr26 else None
        if hr == 0:
            txt = f"**{lab}**: 0 HR in {ab} AB (he's 1 per {per_o} overall)"
        else:
            txt = f"**{lab}**: just {hr} of his {hr26} HR — 1 per {round(ab/hr)} AB vs 1 per {per_o} overall"
        split_watch.append((20, txt)); seen.add(r.split)
    # 3) strength
    if opc is not None and opc >= HI:
        split_good.append((-opc, f"**{lab}**: {ops} OPS — {opc}th pct ({pa} PA)"))
    elif isc is not None and isc >= HI:
        split_good.append((-isc, f"**{lab}** power: {hr} HR in {ab} AB ({iso} ISO, {isc}th pct)"))

if split_watch:
    split_watch.sort()
    st.markdown("**2026 splits worth flagging:**")
    st.markdown("\n".join(f"- {t}" for _, t in split_watch[:3]))

st.divider()

# ---------------- 3b) NOTABLE STRENGTHS ----------------
st.markdown("### ✅ Notable strengths")
strengths = []
# top overall percentile metrics in predominant role
if predominant == "hitting" and row is not None and ctx is not None:
    for col, lab in HIT_PCT.items():
        v = f(row.get(col))
        if v is not None and v >= 0.85:
            strengths.append((-v, f"**{lab}** — {int(round(v*100))}th percentile ({ctx})"))
elif predominant == "pitching" and row is not None and ctx is not None:
    for col, lab in PIT_PCT.items():
        v = f(row.get(col))
        if v is not None and v >= 0.85:
            strengths.append((-v, f"**{lab}** — {int(round(v*100))}th percentile ({ctx})"))
strengths.sort()
lines = [t for _, t in strengths[:3]] + [t for _, t in sorted(split_good)[:3]]
if lines:
    st.markdown("\n".join(f"- {t}" for t in lines))
else:
    st.caption("No top-15% standout metrics or splits flagged for the latest season.")

st.divider()

# ---------------- 4) HANDEDNESS ----------------
st.markdown("### 🤚 Handedness angle")


def handed_pct(df, peer_df, metric, higher_better=True, label=""):
    """Percentile of this player's metric within the same-handedness peer group."""
    vals = pd.to_numeric(peer_df[metric], errors="coerce").dropna()
    mine = f(df.iloc[-1][metric]) if len(df) else None
    if mine is None or not len(vals):
        return None
    pct = (vals < mine).mean() if higher_better else (vals > mine).mean()
    return round(pct * 100)


notes = []
LATEST = max([norm(y) for y in pd.concat([ph, pp])["year"]] or ["0"])
if predominant == "hitting" and bat in ("L", "R", "S"):
    row = ph[ph["year"].map(norm) == LATEST]
    peers = hit[(hit["year"].map(norm) == LATEST) &
                (pd.to_numeric(hit["plate_appearances"], errors="coerce").fillna(0) >= 50)].copy()
    # restrict peer group to same handedness via players.csv bat
    bat_map = {norm(r["id"]): (r.get("bat") or "").strip().upper() for _, r in players.iterrows()}
    peers = peers[peers["pid"].map(bat_map) == bat]
    for metric, lab in (("on_base_percentage", "OBP"), ("slugging_percentage", "SLG"),
                        ("isolated_power", "ISO"), ("weighted_on_base_average", "wOBA")):
        p = handed_pct(row, peers, metric, True)
        if p is not None:
            notes.append(f"**{p}th percentile in {lab}** among {BAT_LABEL.get(bat,bat)} hitters ({LATEST})")
elif predominant == "pitching" and throw in ("L", "R"):
    row = pp[pp["year"].map(norm) == LATEST]
    peers = pit[(pit["year"].map(norm) == LATEST) &
                (pd.to_numeric(pit["innings_pitched"], errors="coerce").fillna(0) >= 15)].copy()
    thr_map = {norm(r["id"]): (r.get("throw") or "").strip().upper() for _, r in players.iterrows()}
    peers = peers[peers["pid"].map(thr_map) == throw]
    for metric, lab, hb in (("strikeout_to_walk_ratio", "K/BB", True),
                            ("fielding_independent_pitching", "FIP", False),
                            ("walks_plus_hits_per_inning_pitched", "WHIP", False),
                            ("on_base_plus_slugging_against", "OPS allowed", False)):
        p = handed_pct(row, peers, metric, hb)
        if p is not None:
            notes.append(f"**{p}th percentile in {lab}** among {THROW_LABEL.get(throw,throw)} pitchers ({LATEST})")

if notes:
    st.markdown("\n".join(f"- {n}" for n in notes[:4]))
else:
    st.caption("Not enough same-handedness peer data to compute a split for the latest season.")

st.divider()
st.caption("Overview = career totals across all seasons in our data. Big weekend = best 3-game "
           "series from per-game PBP (2025–2026). Percentiles are within all qualified players; "
           "handedness percentiles are within the player's own bat/throw group.")
