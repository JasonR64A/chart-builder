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
