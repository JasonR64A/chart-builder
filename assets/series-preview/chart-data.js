/* ===== DATA ===== */
window.DATA = {
  teamA: {
    name: "NORTH GEORGIA",
    hitting: [
      { label: "OPS",  value: ".914",  pct: 82.2 },
      { label: "wOBA", value: ".402",  pct: 82.2 },
      { label: "ISO",  value: ".173",  pct: 80.3 },
      { label: "wRC",  value: "202.11",pct: 87.7 },
      { label: "wRAA", value: "22.59", pct: 88.1 },
      { label: "R/PA", value: "0.20",  pct: 76.7 },
      { label: "K/BB", value: "0.94",  pct: 97.3 }
    ],
    pitching: [
      { label: "OPS",    value: ".725",  pct: 84.7 },
      { label: "BB%",    value: "7.6%",  pct: 95.3 },
      { label: "K%",     value: "16.5%", pct: 36.2 },
      { label: "WHIP",   value: "1.36",  pct: 90.2 },
      { label: "xFIP",   value: "4.61",  pct: 68.2 },
      { label: "SIERA",  value: "4.45",  pct: 61.1 },
      { label: "HR/FB%", value: "8.0%",  pct: 65.4 }
    ]
  },
  teamB: {
    name: "CATAWBA",
    hitting: [
      { label: "OPS",  value: ".894",  pct: 77.5 },
      { label: "wOBA", value: ".395",  pct: 77.5 },
      { label: "ISO",  value: ".166",  pct: 74.8 },
      { label: "wRC",  value: "207.88",pct: 90.9 },
      { label: "wRAA", value: "18.81", pct: null },
      { label: "R/PA", value: "0.20",  pct: 78.7 },
      { label: "K/BB", value: "1.14",  pct: 88.6 }
    ],
    pitching: [
      { label: "OPS",    value: ".741",  pct: 80.8 },
      { label: "BB%",    value: "9.6%",  pct: 78.8 },
      { label: "K%",     value: "16.3%", pct: 33.4 },
      { label: "WHIP",   value: "1.45",  pct: 83.5 },
      { label: "xFIP",   value: "5.06",  pct: 43.8 },
      { label: "SIERA",  value: "4.72",  pct: 48.1 },
      { label: "HR/FB%", value: "7.2%",  pct: 76.8 }
    ]
  }
};

/* Head-to-head bullet bars: 5 headline stats */
window.BULLETS = [
  { label: "TEAM wRAE35", a: { v: "3.31", pct: 92.2 }, b: { v: "2.87", pct: 89.8 } },
  { label: "wRCE35",      a: { v: "1.53", pct: 62.9 }, b: { v: "1.62", pct: 69.9 } },
  { label: "ROTATION",    a: { v: "67.78", pct: 76.9 }, b: { v: "48.54", pct: 87.1 } },
  { label: "BULLPEN",     a: { v: "16.03", pct: 86.7 }, b: { v: "9.55",  pct: 80.0 } },
  { label: "wRE",         a: { v: "4.84", pct: 92.2 }, b: { v: "4.49",  pct: 89.8 } }
];

/* Radar axes — hitting (7) + pitching (7) = 14 axes, in percentiles */
window.RADAR_AXES = [
  // Hitting axes
  { key: "OPS",    group: "H" },
  { key: "wOBA",   group: "H" },
  { key: "ISO",    group: "H" },
  { key: "wRC",    group: "H" },
  { key: "R/PA",   group: "H" },
  { key: "K/BB",   group: "H" },
  { key: "wRAA",   group: "H" },
  // Pitching axes
  { key: "OPS-P",  label: "OPS-A", group: "P", source: "pitching", lookup: "OPS" },
  { key: "BB%",    group: "P" },
  { key: "K%",     group: "P" },
  { key: "WHIP",   group: "P" },
  { key: "xFIP",   group: "P" },
  { key: "SIERA",  group: "P" },
  { key: "HR/FB%", group: "P" }
];

/* Mock 14-day pace data — per game. Teams played roughly 14 games.
   Values are "rolling 7-game" snapshots so curves look smoothed. */
function mockSeries(seed, mean, variance, trend) {
  // seeded-ish pseudo-random using sin
  const pts = [];
  for (let i = 0; i < 14; i++) {
    const n = Math.sin((seed + i) * 12.9898) * 43758.5453;
    const noise = (n - Math.floor(n) - 0.5) * 2 * variance;
    pts.push(mean + noise + (i / 13) * trend);
  }
  return pts;
}

window.LAST5 = {
  a: [
    { wl: 'W', score: '9-2',  opp: 'vs LU'  },
    { wl: 'W', score: '6-3',  opp: '@ AU'   },
    { wl: 'L', score: '4-7',  opp: '@ AU'   },
    { wl: 'W', score: '11-4', opp: 'vs FMU' },
    { wl: 'W', score: '8-5',  opp: 'vs FMU' }
  ],
  b: [
    { wl: 'W', score: '7-4',  opp: 'vs WIN' },
    { wl: 'L', score: '3-5',  opp: 'vs WIN' },
    { wl: 'W', score: '10-6', opp: '@ LMU'  },
    { wl: 'W', score: '5-2',  opp: '@ LMU'  },
    { wl: 'L', score: '2-8',  opp: '@ LMU'  }
  ]
};

window.PACE = {
  // Team A series
  a: {
    ops:   mockSeries(7.2, 0.900, 0.05, 0.030),
    xfip:  mockSeries(4.9, 4.55, 0.30, -0.25),
    wrc:   mockSeries(6.4, 128,   14,   8),
    siera: mockSeries(8.1, 4.45, 0.30, -0.22)
  },
  // Team B series
  b: {
    ops:   mockSeries(3.1, 0.880, 0.06, -0.010),
    xfip:  mockSeries(2.3, 5.05, 0.35, 0.15),
    wrc:   mockSeries(5.7, 118,   15,  -4),
    siera: mockSeries(9.8, 4.72, 0.35, 0.12)
  },
  meta: {
    ops:   { divAvg: 0.800, min: 0.70, max: 1.02, format: v => '.' + Math.max(0, Math.round(v*1000)).toString().padStart(3,'0').slice(-3) },
    xfip:  { divAvg: 4.80,  min: 3.6,  max: 6.2,  format: v => v.toFixed(2), lowerBetter: true },
    wrc:   { divAvg: 100,   min: 85,   max: 155,  format: v => Math.round(v).toString() },
    siera: { divAvg: 4.70,  min: 3.6,  max: 6.0,  format: v => v.toFixed(2), lowerBetter: true }
  }
};
