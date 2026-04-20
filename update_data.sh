#!/bin/bash
# Update chart builder data and push to Streamlit Cloud
# Run: bash update_data.sh

# Ensure Unix tools are on PATH even when invoked from Task Scheduler's
# stripped-down environment (which otherwise gives us only bash.exe and
# no mkdir/gzip/tail/date/cp/git). 2026-04-19: the nightly was silently
# failing on all cp/mkdir calls because of this.
export PATH="/usr/bin:/mingw64/bin:/c/Program Files/Git/usr/bin:/c/Program Files/Git/mingw64/bin:/c/Program Files/Git/cmd:$PATH"

SOURCE="C:/Users/sixty/OneDrive/Desktop/64Analytics/Website_Data/Zip Uploads/Zips"
DEST="C:/Users/sixty/OneDrive/Desktop/chart-builder-app/data"

PBP_SOURCE="C:/Users/sixty/OneDrive/Desktop/scrape_final/output/2026"
PBP_DEST="C:/Users/sixty/OneDrive/Desktop/chart-builder-app/pbp_data"

echo "Copying latest CSVs..."
cp "$SOURCE"/*.csv "$DEST/"

echo "Copying RPI data..."
for sport in baseball softball; do
    cp "$PBP_SOURCE/$sport/rpi/${sport}_rpi_D1.csv" "$DEST/" 2>/dev/null
done

echo "Copying external rankings (Massey + DSR + ELO)..."
mkdir -p "$DEST/rankings"
for sport in baseball softball; do
    cp "$PBP_SOURCE/$sport/rankings/massey_${sport}.csv" "$DEST/rankings/" 2>/dev/null
    cp "$PBP_SOURCE/$sport/rankings/dsr_${sport}.csv" "$DEST/rankings/" 2>/dev/null
    cp "$PBP_SOURCE/$sport/rankings/elo_${sport}.csv" "$DEST/rankings/" 2>/dev/null
done

echo "Copying bracketology snapshots..."
mkdir -p "$DEST/bracketology/snapshots"
for sport in baseball softball; do
    cp "$PBP_SOURCE/$sport/bracketology/${sport}_bracketology_"*.csv "$DEST/bracketology/snapshots/" 2>/dev/null
    cp "$PBP_SOURCE/$sport/bracketology/${sport}_bracketology_history.csv" "$DEST/bracketology/snapshots/" 2>/dev/null
done

echo "Copying play-by-play event data..."
mkdir -p "$PBP_DEST/play_by_play"
for sport in baseball softball; do
    for div in D1 D2 D3; do
        cp "$PBP_SOURCE/$sport/pbp/play_by_play_${div}.csv" "$PBP_DEST/play_by_play/${sport}_play_by_play_${div}.csv" 2>/dev/null
    done
done

# Regenerate gzipped copies so Render can access PBP events.
# The raw .csv files are gitignored (~260MB each), but .csv.gz compress to ~15MB.
echo "Gzipping play-by-play event data for Render..."
for f in "$PBP_DEST/play_by_play/"*.csv; do
    if [ -f "$f" ]; then
        gzip -9 -c "$f" > "$f.gz"
    fi
done

echo "Copying PBP data..."
for sport in baseball softball; do
    cp "$PBP_SOURCE/$sport/pbp/hitting_pbp_"*.csv "$PBP_DEST/$sport/" 2>/dev/null
    cp "$PBP_SOURCE/$sport/pbp/pitching_pbp_"*.csv "$PBP_DEST/$sport/" 2>/dev/null
    cp "$PBP_SOURCE/$sport/pbp/fielding_pbp_"*.csv "$PBP_DEST/$sport/" 2>/dev/null
    cp "$PBP_SOURCE/$sport/schedules_full.csv" "$DEST/schedules_full_${sport}.csv" 2>/dev/null
done

echo "Copying PBP validation output (diff JSONs for the Validator page)..."
mkdir -p "$PBP_DEST/validated"
for sport in baseball softball; do
    for div in D1 D2 D3; do
        cp "$PBP_SOURCE/$sport/pbp/validated/validation_diff_${sport}_${div}.json" "$PBP_DEST/validated/" 2>/dev/null
        cp "$PBP_SOURCE/$sport/pbp/validated/validated_events_${sport}_${div}.parquet" "$PBP_DEST/validated/" 2>/dev/null
    done
done

cd "C:/Users/sixty/OneDrive/Desktop/chart-builder-app"

# Fix player names with commas (e.g. "Bocachica, Jr.") that break CSV parsing
echo "Fixing unquoted commas in player names..."
python3 -c "
import os
def fix_csv(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    header = lines[0]
    expected = header.count(',') + 1
    fixed = 0
    out = [header]
    for line in lines[1:]:
        field_count = 0
        in_quote = False
        for ch in line:
            if ch == '\"': in_quote = not in_quote
            elif ch == ',' and not in_quote: field_count += 1
        field_count += 1
        if field_count > expected:
            parts = []
            current = ''
            in_q = False
            for ch in line:
                if ch == '\"':
                    in_q = not in_q
                    continue
                elif ch == ',' and not in_q:
                    parts.append(current)
                    current = ''
                    continue
                current += ch
            parts.append(current.rstrip('\n').rstrip('\r'))
            if len(parts) == expected + 1:
                parts[5] = '\"' + parts[5] + ',' + parts[6] + '\"'
                del parts[6]
                out.append(','.join(parts) + '\n')
                fixed += 1
            else:
                out.append(line)
        else:
            out.append(line)
    if fixed > 0:
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            f.writelines(out)
        print(f'  {os.path.basename(filepath)}: fixed {fixed} rows')

for sport in ['baseball', 'softball']:
    for stat in ['hitting', 'pitching', 'fielding']:
        for div in ['D1', 'D2', 'D3']:
            path = f'C:/Users/sixty/OneDrive/Desktop/chart-builder-app/pbp_data/{sport}/{stat}_pbp_{div}.csv'
            if os.path.exists(path):
                fix_csv(path)
"

# Deduplicate PBP files (cross-division scrapes can produce duplicate game rows)
echo "Deduplicating PBP files..."
python3 -c "
import pandas as pd
for sport in ['baseball', 'softball']:
    for stat in ['hitting', 'pitching', 'fielding']:
        for div in ['D1', 'D2', 'D3']:
            path = f'C:/Users/sixty/OneDrive/Desktop/chart-builder-app/pbp_data/{sport}/{stat}_pbp_{div}.csv'
            try:
                df = pd.read_csv(path, low_memory=False)
                before = len(df)
                df = df.drop_duplicates(subset=['gameId', 'playerId', 'teamName'], keep='first')
                removed = before - len(df)
                if removed > 0:
                    df.to_csv(path, index=False)
                    print(f'  {sport}/{stat}_{div}: removed {removed} dupes')
            except Exception:
                pass
"

# Refresh thrill scores (uses current PBP data to compute per-game excitement ratings)
echo "Computing thrill scores..."
python scripts/compute_thrill_scores.py 2>&1 | tail -3

# Check if anything changed
if git diff --quiet data/ pbp_data/ 2>/dev/null; then
    echo "No data changes detected."
    exit 0
fi

echo "Pushing updated data..."
git add data/ pbp_data/
git commit -m "Daily data update $(date +%Y-%m-%d)"
git push

echo "Done! Streamlit Cloud will redeploy automatically."
