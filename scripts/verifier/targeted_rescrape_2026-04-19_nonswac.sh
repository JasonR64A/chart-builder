#!/bin/bash
# Non-SWAC targeted rescrape (GT + Bradley + LIU + VCU + Delaware St. + Wagner + Le Moyne + La Salle)
# Excludes Southern U. + Texas Southern (known bad NCAA data source).

cd "C:/Users/sixty/OneDrive/Desktop/scrape_final" || exit 1

# 2026-02-20  (Georgia Tech, Delaware St., La Salle)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-02-20 --division D1 --no-cache

# 2026-02-21  (Georgia Tech)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-02-21 --division D1 --no-cache

# 2026-02-22  (VCU)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-02-22 --division D1 --no-cache

# 2026-02-28  (VCU)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-02-28 --division D1 --no-cache

# 2026-03-07  (Bradley)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-07 --division D1 --no-cache

# 2026-03-10  (LIU)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-10 --division D1 --no-cache

# 2026-03-14  (LIU, VCU, Wagner, La Salle)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-14 --division D1 --no-cache

# 2026-03-15  (LIU, Wagner)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-15 --division D1 --no-cache

# 2026-03-20  (LIU, VCU, Delaware St., Wagner, Le Moyne, La Salle)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-20 --division D1 --no-cache

# 2026-03-22  (Bradley)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-22 --division D1 --no-cache

# 2026-03-28  (Bradley)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-28 --division D1 --no-cache

# 2026-03-29  (Bradley)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-29 --division D1 --no-cache

# 2026-03-31  (Bradley, Delaware St.)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-31 --division D1 --no-cache

# 2026-04-02  (LIU, Le Moyne)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-04-02 --division D1 --no-cache

# 2026-04-03  (LIU, Le Moyne)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-04-03 --division D1 --no-cache

# 2026-04-04  (Bradley)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-04-04 --division D1 --no-cache

# 2026-04-07  (Delaware St.)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-04-07 --division D1 --no-cache

# 2026-04-11  (Wagner)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-04-11 --division D1 --no-cache

# 2026-04-14  (Bradley, LIU, VCU)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-04-14 --division D1 --no-cache

echo "Done. Rerun pa_review.py + missing_dates.py to verify."
