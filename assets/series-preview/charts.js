/* ==========================================================
   CHARTS: radar, bullets, pace
   ========================================================== */

const $ = s => document.querySelector(s);

function pctClass(p) {
  if (p == null) return 'na';
  if (p >= 75) return 'pct-green';
  if (p >= 50) return 'pct-gold';
  if (p >= 25) return 'pct-orange';
  return 'pct-red';
}
function pctText(p) { return p == null ? '—' : p.toFixed(1) + '%'; }

function axisPct(team, axis) {
  if (axis.group === 'H') {
    const row = team.hitting.find(r => r.label === axis.key);
    return row ? row.pct : null;
  }
  const key = axis.lookup || axis.key;
  const row = team.pitching.find(r => r.label === key);
  return row ? row.pct : null;
}

/* ====== RADAR (hero, in center) ====== */
function renderRadar(el, axes, teamA, teamB, opts = {}) {
  if (!el) return;
  const W = 520, H = 440;
  const cx = W / 2, cy = H / 2;
  const R = 150;
  const n = axes.length;
  const style = opts.style || 'rings';
  const angle = i => -Math.PI / 2 + (i / n) * Math.PI * 2;
  const pt = (i, r) => [cx + Math.cos(angle(i)) * r, cy + Math.sin(angle(i)) * r];
  const rings = [25, 50, 75, 100];

  let gridSVG = '';
  if (style === 'rings') {
    rings.forEach(v => {
      const r = (v / 100) * R;
      gridSVG += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="#E2DED2" stroke-width="${v === 100 ? 1 : 0.6}" ${v < 100 ? 'stroke-dasharray="2 2"' : ''}/>`;
    });
  } else if (style === 'polygon') {
    rings.forEach(v => {
      const r = (v / 100) * R;
      const poly = axes.map((_, i) => pt(i, r).join(',')).join(' ');
      gridSVG += `<polygon points="${poly}" fill="none" stroke="#E2DED2" stroke-width="${v === 100 ? 1 : 0.6}" ${v < 100 ? 'stroke-dasharray="2 2"' : ''}/>`;
    });
  }

  let spokes = '';
  axes.forEach((_, i) => {
    const [x, y] = pt(i, R);
    spokes += `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="#CFCABA" stroke-width="0.5" stroke-dasharray="1.5 2"/>`;
  });

  const makePoly = (team, fill, stroke) => {
    const pts = axes.map((a, i) => {
      const p = axisPct(team, a) ?? 0;
      return pt(i, (p / 100) * R).join(',');
    }).join(' ');
    return `<polygon points="${pts}" fill="${fill}" stroke="${stroke}" stroke-width="1.6" stroke-linejoin="round"/>`;
  };
  const polyA = makePoly(teamA, 'rgba(196,18,48,0.16)', '#C41230');
  const polyB = makePoly(teamB, 'rgba(41,51,92,0.16)', '#29335C');

  let dotsA = '', dotsB = '';
  if (opts.dots !== false) {
    axes.forEach((a, i) => {
      const pa = axisPct(teamA, a);
      const pb = axisPct(teamB, a);
      if (pa != null) {
        const [x, y] = pt(i, (pa / 100) * R);
        dotsA += `<circle cx="${x}" cy="${y}" r="2.8" fill="#C41230"/>`;
      }
      if (pb != null) {
        const [x, y] = pt(i, (pb / 100) * R);
        dotsB += `<circle cx="${x}" cy="${y}" r="2.8" fill="#29335C"/>`;
      }
    });
  }

  let labels = '';
  axes.forEach((a, i) => {
    const [x, y] = pt(i, R + 18);
    const ang = angle(i);
    let anchor = 'middle';
    if (Math.cos(ang) > 0.25) anchor = 'start';
    else if (Math.cos(ang) < -0.25) anchor = 'end';
    const label = a.label || a.key;
    const color = a.group === 'H' ? '#2D2926' : '#888888';
    labels += `<text x="${x}" y="${y}" text-anchor="${anchor}" dominant-baseline="middle"
      font-family="Inter, sans-serif" font-size="11" font-weight="700"
      letter-spacing="0.04em" fill="${color}">${label}</text>`;
  });

  let ringLabels = '';
  [25, 50, 75].forEach(v => {
    const r = (v / 100) * R;
    ringLabels += `<text x="${cx + 3}" y="${cy - r + 3}" font-family="JetBrains Mono, monospace" font-size="8" fill="#B5B0A3" font-weight="500">${v}</text>`;
  });

  const hIdx = axes.map((a,i)=>a.group==='H'?i:null).filter(x=>x!==null);
  const pIdx = axes.map((a,i)=>a.group==='P'?i:null).filter(x=>x!==null);
  function arcBracket(indices, color, label) {
    if (!indices.length) return '';
    const rr = R + 46;
    const a0 = angle(indices[0]) - (Math.PI/n)*0.9;
    const a1 = angle(indices[indices.length-1]) + (Math.PI/n)*0.9;
    const x0 = cx + Math.cos(a0)*rr, y0 = cy + Math.sin(a0)*rr;
    const x1 = cx + Math.cos(a1)*rr, y1 = cy + Math.sin(a1)*rr;
    const large = (a1 - a0) > Math.PI ? 1 : 0;
    const mid = (a0 + a1)/2;
    const tx = cx + Math.cos(mid)*(rr + 14);
    const ty = cy + Math.sin(mid)*(rr + 14);
    return `
      <path d="M ${x0} ${y0} A ${rr} ${rr} 0 ${large} 1 ${x1} ${y1}" fill="none" stroke="${color}" stroke-width="1.2" stroke-linecap="round" opacity="0.6"/>
      <text x="${tx}" y="${ty}" text-anchor="middle" dominant-baseline="middle"
        font-family="Inter, sans-serif" font-size="10" font-weight="800" letter-spacing="0.22em" fill="${color}">${label}</text>`;
  }
  const brackets = arcBracket(hIdx, '#2D2926', 'HITTING') + arcBracket(pIdx, '#888888', 'PITCHING');

  el.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
      ${gridSVG}${spokes}${ringLabels}${polyA}${polyB}${dotsA}${dotsB}${labels}${brackets}
    </svg>`;
}

/* ====== MICRO STRIP ====== */
function renderMicro(el, stats) {
  if (!el) return;
  el.innerHTML = stats.map(s => `
    <div class="micro-cell">
      <div class="micro-label">${s.label}</div>
      <div class="micro-value">${s.value}</div>
      <div class="micro-pct ${pctClass(s.pct)}">${pctText(s.pct)}</div>
    </div>`).join('');
}

/* ====== BULLETS ====== */
function renderBullets(el, rows) {
  if (!el) return;
  el.innerHTML = rows.map(r => {
    const aW = Math.max(2, (r.a.pct || 0) / 100 * 50);
    const bW = Math.max(2, (r.b.pct || 0) / 100 * 50);
    return `
      <div class="bullet-row">
        <div class="bullet-val a">${r.a.v}</div>
        <div class="bullet-track">
          <div class="bullet-mid"></div>
          <div class="bullet-bar a" style="width:${aW}%"></div>
          <div class="bullet-bar b" style="width:${bW}%"></div>
          <div class="bullet-label">${r.label}</div>
        </div>
        <div class="bullet-val b">${r.b.v}</div>
      </div>`;
  }).join('');
}

/* ====== PACE (small sparkline) ======
   One team's series, one metric. Subtle band, single line + area. */
function renderPaceSmall(el, values, meta, teamColor, opts = {}) {
  if (!el) return;
  const W = 260, H = 110;
  const padL = 28, padR = 10, padT = 10, padB = 16;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;
  const n = values.length;
  const x = i => padL + (i / (n - 1)) * innerW;
  const yScale = v => {
    const t = (v - meta.min) / (meta.max - meta.min);
    const tt = meta.lowerBetter ? t : (1 - t);
    return padT + tt * innerH;
  };

  // Y ticks
  const ticks = [meta.min, (meta.min + meta.max)/2, meta.max];
  let yGrid = '';
  ticks.forEach(v => {
    const y = yScale(v);
    yGrid += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="#E2DED2" stroke-width="0.5" stroke-dasharray="2 2"/>`;
    yGrid += `<text x="${padL - 5}" y="${y}" text-anchor="end" dominant-baseline="middle" font-family="JetBrains Mono, monospace" font-size="7.5" fill="#B5B0A3" font-weight="500">${meta.format(v)}</text>`;
  });

  // X labels
  let xAxis = '';
  [0, n-1].forEach((i, k) => {
    const xx = x(i);
    const lbl = i === 0 ? 'G‑14' : 'NOW';
    xAxis += `<text x="${xx}" y="${H - 4}" text-anchor="${k===0?'start':'end'}" font-family="JetBrains Mono, monospace" font-size="7.5" fill="#B5B0A3" font-weight="500" letter-spacing="0.06em">${lbl}</text>`;
  });

  // Band
  let band = '';
  if (opts.band !== false) {
    const avgY = yScale(meta.divAvg);
    band += `<line x1="${padL}" y1="${avgY}" x2="${W - padR}" y2="${avgY}" stroke="#888888" stroke-width="0.5" stroke-dasharray="3 3" opacity="0.6"/>`;
  }

  // Smooth path
  function path(pts) {
    if (!pts.length) return '';
    let d = `M ${pts[0][0]} ${pts[0][1]}`;
    for (let i = 1; i < pts.length; i++) {
      const p0 = pts[i-1], p1 = pts[i];
      const cpx = (p0[0] + p1[0]) / 2;
      d += ` C ${cpx} ${p0[1]}, ${cpx} ${p1[1]}, ${p1[0]} ${p1[1]}`;
    }
    return d;
  }

  const pts = values.map((v, i) => [x(i), yScale(v)]);
  const baselineY = padT + innerH;
  const areaD = `M ${pts[0][0]} ${baselineY} ` + pts.map(p => `L ${p[0]} ${p[1]}`).join(' ') + ` L ${pts[pts.length-1][0]} ${baselineY} Z`;

  const last = pts[pts.length-1];
  const lastVal = values[values.length-1];
  const fillColor = teamColor === 'red' ? 'rgba(196,18,48,0.12)' : 'rgba(41,51,92,0.12)';
  const strokeColor = teamColor === 'red' ? '#C41230' : '#29335C';

  el.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      ${yGrid}${band}${xAxis}
      <path d="${areaD}" fill="${fillColor}"/>
      <path d="${path(pts)}" fill="none" stroke="${strokeColor}" stroke-width="2" stroke-linejoin="round"/>
      ${pts.slice(0, -1).map(p => `<circle cx="${p[0]}" cy="${p[1]}" r="1.3" fill="${strokeColor}" opacity="0.5"/>`).join('')}
      <circle cx="${last[0]}" cy="${last[1]}" r="3.5" fill="${strokeColor}" stroke="#FAF8F2" stroke-width="1.5"/>
      <text x="${last[0] - 5}" y="${last[1] - 7}" text-anchor="end" font-family="Inter, sans-serif" font-size="10" font-weight="800" fill="${strokeColor}">${meta.format(lastVal)}</text>
    </svg>`;
}

function setDelta(idNow, idDelta, values, meta) {
  const first = values[0], last = values[values.length - 1];
  const delta = last - first;
  const improving = meta.lowerBetter ? delta < 0 : delta > 0;
  const nowEl = document.getElementById(idNow);
  const dEl = document.getElementById(idDelta);
  if (nowEl) nowEl.textContent = 'NOW ' + meta.format(last);
  if (dEl) {
    const sign = delta > 0 ? '+' : '';
    dEl.textContent = sign + (meta.lowerBetter ? delta.toFixed(2) : (Math.abs(delta) < 1 ? delta.toFixed(3) : delta.toFixed(1)));
    dEl.classList.toggle('up', improving);
    dEl.classList.toggle('down', !improving);
  }
}

function renderLast5(el, games) {
  if (!el) return;
  el.innerHTML = games.map(g => `
    <div class="last5-item">
      <span class="last5-wl ${g.wl.toLowerCase()}">${g.wl}</span>
      <span class="last5-score">${g.score}</span>
      <span class="last5-opp">${g.opp}</span>
    </div>`).join('');
}

/* ====== TWEAKS ====== */
const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "radarStyle": "rings",
  "showBand": true,
  "showDots": true,
  "theme": "warm",
  "seriesTitle": "Peach Belt Conference"
}/*EDITMODE-END*/;

let tweaks = { ...TWEAK_DEFAULTS };

function applyTheme(t) {
  const root = document.documentElement;
  if (t === 'paper') {
    root.style.setProperty('--bg', '#FFFFFF');
    root.style.setProperty('--card', '#FBFAF5');
    root.style.setProperty('--card-border', '#E5E2DA');
    root.style.setProperty('--line', '#EDEAE1');
  } else {
    root.style.setProperty('--bg', '#FAF8F2');
    root.style.setProperty('--card', '#FFFFFF');
    root.style.setProperty('--card-border', '#D9D5CC');
    root.style.setProperty('--line', '#E2DED2');
  }
}

/* ====== RENDER ALL ====== */
function renderAll() {
  renderRadar($('#radarCenter'), RADAR_AXES, DATA.teamA, DATA.teamB, {
    style: tweaks.radarStyle, dots: tweaks.showDots
  });

  renderMicro($('#microAH'), DATA.teamA.hitting);
  renderMicro($('#microAP'), DATA.teamA.pitching);
  renderMicro($('#microBH'), DATA.teamB.hitting);
  renderMicro($('#microBP'), DATA.teamB.pitching);

  renderLast5($('#last5A'), LAST5.a);
  renderLast5($('#last5B'), LAST5.b);

  renderBullets($('#bullets'), BULLETS);

  // Team A pace charts
  renderPaceSmall($('#paceAOPS'),   PACE.a.ops,   PACE.meta.ops,   'red',  { band: tweaks.showBand });
  renderPaceSmall($('#paceAXFIP'),  PACE.a.xfip,  PACE.meta.xfip,  'red',  { band: tweaks.showBand });
  renderPaceSmall($('#paceAWRC'),   PACE.a.wrc,   PACE.meta.wrc,   'red',  { band: tweaks.showBand });
  renderPaceSmall($('#paceASIERA'), PACE.a.siera, PACE.meta.siera, 'red',  { band: tweaks.showBand });

  setDelta('paceAOPSnow',   'paceAOPSdelta',   PACE.a.ops,   PACE.meta.ops);
  setDelta('paceAXFIPnow',  'paceAXFIPdelta',  PACE.a.xfip,  PACE.meta.xfip);
  setDelta('paceAWRCnow',   'paceAWRCdelta',   PACE.a.wrc,   PACE.meta.wrc);
  setDelta('paceASIERAnow', 'paceASIERAdelta', PACE.a.siera, PACE.meta.siera);

  // Team B pace charts
  renderPaceSmall($('#paceBOPS'),   PACE.b.ops,   PACE.meta.ops,   'navy', { band: tweaks.showBand });
  renderPaceSmall($('#paceBXFIP'),  PACE.b.xfip,  PACE.meta.xfip,  'navy', { band: tweaks.showBand });
  renderPaceSmall($('#paceBWRC'),   PACE.b.wrc,   PACE.meta.wrc,   'navy', { band: tweaks.showBand });
  renderPaceSmall($('#paceBSIERA'), PACE.b.siera, PACE.meta.siera, 'navy', { band: tweaks.showBand });

  setDelta('paceBOPSnow',   'paceBOPSdelta',   PACE.b.ops,   PACE.meta.ops);
  setDelta('paceBXFIPnow',  'paceBXFIPdelta',  PACE.b.xfip,  PACE.meta.xfip);
  setDelta('paceBWRCnow',   'paceBWRCdelta',   PACE.b.wrc,   PACE.meta.wrc);
  setDelta('paceBSIERAnow', 'paceBSIERAdelta', PACE.b.siera, PACE.meta.siera);

  applyTheme(tweaks.theme);
  document.getElementById('seriesTitle').textContent = tweaks.seriesTitle;
}
renderAll();

/* ====== SCALE ====== */
function fit() {
  const wrap = document.getElementById('cardWrap');
  const s = Math.min(window.innerWidth / 1600, window.innerHeight / 900);
  wrap.style.transform = `scale(${s})`;
}
fit();
window.addEventListener('resize', fit);

/* ====== TWEAK WIRING ====== */
document.getElementById('tweakRadar').value = tweaks.radarStyle;
document.getElementById('tweakTheme').value = tweaks.theme;
document.getElementById('tweakTitle').value = tweaks.seriesTitle;
document.getElementById('tweakBand').classList.toggle('on', !!tweaks.showBand);
document.getElementById('tweakDots').classList.toggle('on', !!tweaks.showDots);

function postEdit(patch) {
  window.parent.postMessage({ type: '__edit_mode_set_keys', edits: patch }, '*');
}
document.getElementById('tweakRadar').addEventListener('change', e => {
  tweaks.radarStyle = e.target.value; renderAll(); postEdit({ radarStyle: tweaks.radarStyle });
});
document.getElementById('tweakTheme').addEventListener('change', e => {
  tweaks.theme = e.target.value; applyTheme(tweaks.theme); postEdit({ theme: tweaks.theme });
});
document.getElementById('tweakBand').addEventListener('click', () => {
  tweaks.showBand = !tweaks.showBand;
  document.getElementById('tweakBand').classList.toggle('on', tweaks.showBand);
  renderAll(); postEdit({ showBand: tweaks.showBand });
});
document.getElementById('tweakDots').addEventListener('click', () => {
  tweaks.showDots = !tweaks.showDots;
  document.getElementById('tweakDots').classList.toggle('on', tweaks.showDots);
  renderAll(); postEdit({ showDots: tweaks.showDots });
});
document.getElementById('tweakTitle').addEventListener('input', e => {
  tweaks.seriesTitle = e.target.value;
  document.getElementById('seriesTitle').textContent = tweaks.seriesTitle;
  postEdit({ seriesTitle: tweaks.seriesTitle });
});

window.addEventListener('message', (ev) => {
  const m = ev.data;
  if (!m || !m.type) return;
  if (m.type === '__activate_edit_mode') document.getElementById('tweaksPanel').classList.add('on');
  else if (m.type === '__deactivate_edit_mode') document.getElementById('tweaksPanel').classList.remove('on');
});
window.parent.postMessage({ type: '__edit_mode_available' }, '*');
