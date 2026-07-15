/* <us-commit-map> - self-contained US map with lat/lng commit dots.
   Loads d3 + topojson + us-atlas (states-10m) from CDN, projects with geoAlbersUsa.
   Degrades gracefully (no CDN -> empty; the 64A watermark is an HTML <img> behind
   this element, kept OUT of the SVG so html2canvas exports it). */
(function () {
  const D3 = 'https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js';
  const TOPO = 'https://cdn.jsdelivr.net/npm/topojson-client@3/dist/topojson-client.min.js';
  const ATLAS = 'https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json';
  const loaded = {};
  function loadScript(src) {
    if (loaded[src]) return loaded[src];
    loaded[src] = new Promise((res, rej) => {
      const s = document.createElement('script');
      s.src = src; s.onload = res; s.onerror = rej;
      document.head.appendChild(s);
    });
    return loaded[src];
  }
  let atlasPromise = null;
  function loadAtlas() {
    if (window.__usAtlas) return Promise.resolve(window.__usAtlas);
    if (atlasPromise) return atlasPromise;
    atlasPromise = fetch(ATLAS).then(r => r.json()).then(j => (window.__usAtlas = j));
    return atlasPromise;
  }
  const SVGNS = 'http://www.w3.org/2000/svg';
  const el = (n, a) => { const e = document.createElementNS(SVGNS, n); for (const k in a) e.setAttribute(k, a[k]); return e; };
  class USCommitMap extends HTMLElement {
    static get observedAttributes() { return ['players', 'accent', 'land', 'border']; }
    connectedCallback() { this.style.display = 'block'; this._ensure(); this._boot(); }
    attributeChangedCallback() { if (this._svg) this._draw(); }
    _ensure() {
      if (this._svg) return;
      const svg = el('svg', { viewBox: '0 0 960 600', preserveAspectRatio: 'xMidYMid meet' });
      svg.style.width = '100%'; svg.style.height = '100%'; svg.style.display = 'block';
      this._svg = svg; this.appendChild(svg);
    }
    _players() { try { return JSON.parse(this.getAttribute('players') || '[]'); } catch (e) { return []; } }
    _boot() {
      Promise.all([loadScript(D3), loadScript(TOPO)]).then(loadAtlas)
        .then(() => { this._ready = true; this._draw(); }).catch(() => {});
    }
    _draw() {
      if (!this._ready || !window.d3 || !window.topojson || !window.__usAtlas) return;
      const d3 = window.d3, topojson = window.topojson, us = window.__usAtlas;
      const svg = this._svg; svg.innerHTML = '';
      const accent = this.getAttribute('accent') || '#BA0C2F';
      const land = this.getAttribute('land') || 'rgba(255,255,255,0.16)';
      const border = this.getAttribute('border') || 'rgba(255,255,255,0.34)';
      const nation = topojson.feature(us, us.objects.nation);
      const statesMesh = topojson.mesh(us, us.objects.states, (a, b) => a !== b);
      const proj = d3.geoAlbersUsa().fitExtent([[24, 20], [936, 580]], nation);
      const path = d3.geoPath(proj);
      svg.appendChild(el('path', { d: path(nation), fill: land, stroke: 'none' }));
      svg.appendChild(el('path', { d: path(statesMesh), fill: 'none', stroke: border, 'stroke-width': 1, 'stroke-linejoin': 'round' }));
      svg.appendChild(el('path', { d: path(nation), fill: 'none', stroke: border, 'stroke-width': 1.6, 'stroke-linejoin': 'round' }));
      this._players().forEach(p => {
        const lng = p.lng != null ? p.lng : p.lon;
        const c = proj([lng, p.lat]);
        if (!c) return;
        svg.appendChild(el('circle', { cx: c[0], cy: c[1], r: 10, fill: 'rgba(0,0,0,0.35)' }));
        svg.appendChild(el('circle', { cx: c[0], cy: c[1], r: 8.5, fill: '#fff' }));
        svg.appendChild(el('circle', { cx: c[0], cy: c[1], r: 5, fill: accent }));
      });
    }
  }
  if (!customElements.get('us-commit-map')) customElements.define('us-commit-map', USCommitMap);
})();
