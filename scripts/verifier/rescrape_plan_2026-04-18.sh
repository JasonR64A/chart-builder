#!/bin/bash
# Targeted rescrape for Bucket C teams' missing box-score dates.
# Run from scrape_final dir. Each date = one full NCAA scoreboard pass for D1.
cd "C:/Users/sixty/OneDrive/Desktop/scrape_final" || exit 1

npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-02-15 --division D1 --no-cache
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-02-21 --division D1 --no-cache
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-02-22 --division D1 --no-cache
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-02-24 --division D1 --no-cache
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-02-26 --division D1 --no-cache
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-02-27 --division D1 --no-cache
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-02-28 --division D1 --no-cache
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-01 --division D1 --no-cache
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-03 --division D1 --no-cache
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-10 --division D1 --no-cache
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-14 --division D1 --no-cache
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-17 --division D1 --no-cache
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-20 --division D1 --no-cache
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-22 --division D1 --no-cache
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-27 --division D1 --no-cache
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-28 --division D1 --no-cache
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-03-29 --division D1 --no-cache
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-04-03 --division D1 --no-cache
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-04-04 --division D1 --no-cache
npx ts-node scripts/scrape-pbp-puppeteer.ts baseball 2026 --date 2026-04-14 --division D1 --no-cache

echo "Done. Rerun PA review to verify."
