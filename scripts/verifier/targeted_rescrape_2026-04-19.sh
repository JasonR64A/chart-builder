#!/bin/bash
# Targeted rescrape -- generated 2026-04-19
# Each entry below shows the teams with missing data on that date.
# Each date = one full D1 scoreboard pass (~3-5 min).

cd "C:/Dev/scrape_final" || exit 1

# 2026-02-20  (missing for: Georgia Tech, Delaware St., La Salle)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-02-20 --division D1 --no-cache

# 2026-02-21  (missing for: Southern U., Georgia Tech, Texas Southern)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-02-21 --division D1 --no-cache

# 2026-02-22  (missing for: VCU)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-02-22 --division D1 --no-cache

# 2026-02-24  (missing for: Southern U.)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-02-24 --division D1 --no-cache

# 2026-02-27  (missing for: Southern U.)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-02-27 --division D1 --no-cache

# 2026-02-28  (missing for: Southern U., VCU)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-02-28 --division D1 --no-cache

# 2026-03-01  (missing for: Southern U.)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-01 --division D1 --no-cache

# 2026-03-03  (missing for: Southern U.)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-03 --division D1 --no-cache

# 2026-03-07  (missing for: Bradley)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-07 --division D1 --no-cache

# 2026-03-10  (missing for: Southern U., LIU)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-10 --division D1 --no-cache

# 2026-03-14  (missing for: Southern U., LIU, VCU, Wagner, La Salle)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-14 --division D1 --no-cache

# 2026-03-15  (missing for: LIU, Wagner)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-15 --division D1 --no-cache

# 2026-03-17  (missing for: Texas Southern)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-17 --division D1 --no-cache

# 2026-03-20  (missing for: LIU, VCU, Delaware St., Wagner, Texas Southern, Le Moyne, La Salle)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-20 --division D1 --no-cache

# 2026-03-22  (missing for: Bradley, Texas Southern)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-22 --division D1 --no-cache

# 2026-03-27  (missing for: Southern U.)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-27 --division D1 --no-cache

# 2026-03-28  (missing for: Southern U., Bradley)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-28 --division D1 --no-cache

# 2026-03-29  (missing for: Southern U., Bradley)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-29 --division D1 --no-cache

# 2026-03-31  (missing for: Bradley, Delaware St.)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-31 --division D1 --no-cache

# 2026-04-02  (missing for: LIU, Le Moyne)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-04-02 --division D1 --no-cache

# 2026-04-03  (missing for: Southern U., LIU, Le Moyne)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-04-03 --division D1 --no-cache

# 2026-04-04  (missing for: Southern U., Bradley)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-04-04 --division D1 --no-cache

# 2026-04-07  (missing for: Delaware St.)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-04-07 --division D1 --no-cache

# 2026-04-11  (missing for: Wagner)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-04-11 --division D1 --no-cache

# 2026-04-14  (missing for: Bradley, LIU, VCU)
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-04-14 --division D1 --no-cache

echo "Done. Rerun pa_review.py + missing_dates.py to verify."
