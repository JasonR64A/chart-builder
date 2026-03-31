#!/bin/bash
# Update chart builder data and push to Streamlit Cloud
# Run: bash update_data.sh

SOURCE="C:/Users/sixty/OneDrive/Desktop/64Analytics/Website_Data/Zip Uploads/Zips"
DEST="C:/Users/sixty/OneDrive/Desktop/chart-builder-app/data"

PBP_SOURCE="C:/Users/sixty/OneDrive/Desktop/scrape_final/output/2026"
PBP_DEST="C:/Users/sixty/OneDrive/Desktop/chart-builder-app/pbp_data"

echo "Copying latest CSVs..."
cp "$SOURCE"/*.csv "$DEST/"

echo "Copying PBP data..."
for sport in baseball softball; do
    cp "$PBP_SOURCE/$sport/pbp/hitting_pbp_"*.csv "$PBP_DEST/$sport/" 2>/dev/null
    cp "$PBP_SOURCE/$sport/pbp/pitching_pbp_"*.csv "$PBP_DEST/$sport/" 2>/dev/null
    cp "$PBP_SOURCE/$sport/pbp/fielding_pbp_"*.csv "$PBP_DEST/$sport/" 2>/dev/null
    cp "$PBP_SOURCE/$sport/schedules_full.csv" "$DEST/schedules_full_${sport}.csv" 2>/dev/null
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

# Check if anything changed
if git diff --quiet data/ pbp_data/; then
    echo "No data changes detected."
    exit 0
fi

echo "Pushing updated data..."
git add data/ pbp_data/
git commit -m "Daily data update $(date +%Y-%m-%d)"
git push

echo "Done! Streamlit Cloud will redeploy automatically."
