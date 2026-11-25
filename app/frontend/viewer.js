/**
 * Interactive figure viewer of ECT Crack Chat (plain JavaScript, no libraries, works offline).
 *
 * Input: the JSON of app/backend/ect/interactive.py - the real grids behind a figure. Everything
 * shown here (hover, pins, profile, 3D) is read from those grids; nothing is simulated and no
 * value is read back from a colour.
 *
 *   const v = new EctViewer({bar, card, onImage});   // onImage(true): show the PNG instead
 *   await v.load(id, url);  v.hide();  v.snapshot(); // snapshot = the current view as a PNG data: URL
 *
 * 2D: hover a point, click to pin it, drag to move, wheel to zoom.
 * 3D: drag to rotate, Shift+drag to move, wheel to zoom. Drawn by a small software renderer with
 * a depth buffer (a scan has ~15 000 points), so it needs no GPU and picks exactly.
 *
 * Coordinates: X to the right, Y down, in mm of the ORIGINAL grid. Scan point (col, row) covers
 * [col, col + 1) x [row, row + 1) times the pitch; its position is the centre of that cell.
 */
(function (root) {
  'use strict';

  const INK = '#1b1f24', MUTED = '#5f6b76', LINE = '#c9d1d9', GRID = '#e3e8ed', COPPER = '#b86a3d', CUT = '#0891b2';
  const FONT = '"Segoe UI", system-ui, sans-serif';
  const HEIGHT = 0.7;                 // 3D height of the colour range, in units of half the long side
  const CAMERA = 4.5;                 // camera distance, same units
  const HOME = { az: -28, el: 42, zoom: 1, px: 0, py: 0 };
  const MAX_PINS = 30, CACHE = 12;

  // ------------------------------------------------------------------ numbers

  function bytes(b64) {
    const s = atob(b64), u = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
    return u;
  }
  const f32 = (b64) => new Float32Array(bytes(b64).buffer);
  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

  /** A value with the decimals its map's range needs (1/1000 of the range or finer): neighbours stay distinct. */
  function fmt(v, span) {
    if (!Number.isFinite(v)) return '–';
    const d = span > 0 ? clamp(Math.ceil(-Math.log10(span)) + 3, 0, 12) : 3;
    return (d > 9 ? v.toExponential(4) : v.toFixed(d)).replace('-', '−');
  }
  const mm = (v) => (Number.isFinite(v) ? v.toFixed(1).replace('-', '−') : '–');

  function tickStep(span, n) {
    const raw = span / Math.max(1, n), mag = 10 ** Math.floor(Math.log10(raw)), f = raw / mag;
    return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * mag;
  }
  function ticks(lo, hi, n) {
    if (!(hi > lo)) return { step: 1, values: [lo] };
    const step = tickStep(hi - lo, n), out = [];
    for (let k = Math.ceil(lo / step - 1e-6); k * step <= hi + step * 1e-6; k++) out.push(k * step);
    return { step, values: out };
  }
  function tickText(v, step) {
    if (Math.abs(v) < step * 1e-6) v = 0;
    const d = clamp(-Math.floor(Math.log10(step) + 1e-9), 0, 8);
    return (d > 6 ? v.toExponential(1) : v.toFixed(d)).replace('-', '−');
  }

  // ------------------------------------------------------------------ grids

  function decodeGrid(g) {
    const v = f32(g.f32);
    let lo = Infinity, hi = -Infinity;
    for (const x of v) { if (x < lo) lo = x; if (x > hi) hi = x; }
    return { w: g.w, h: g.h, ox: g.ox, oy: g.oy, v, span: hi - lo };
  }
  /** Value at an absolute scan point; NaN outside the grid. */
  function valueAt(g, col, row) {
    const c = col - g.ox, r = row - g.oy;
    return c >= 0 && r >= 0 && c < g.w && r < g.h ? g.v[r * g.w + c] : NaN;
  }
  function decodeLut(b64) {
    const u = bytes(b64), packed = new Uint32Array(256);
    for (let i = 0; i < 256; i++) packed[i] = (255 << 24) | (u[3 * i + 2] << 16) | (u[3 * i + 1] << 8) | u[3 * i];
    return { rgb: u, packed };
  }

  // ------------------------------------------------------------------ 2D painting

  /** Colour a w x h pixel buffer: bilinear between scan points (like the PNG) or one flat cell each. */
  function paintMap(out, w, h, grid, lo, hi, lut, x0mm, y0mm, mmPerPx, pitch, smooth) {
    const { w: W, h: H, v } = grid, scale = 255 / ((hi - lo) || 1), empty = 0xffe0e0e0;
    const c0 = new Int32Array(w), c1 = new Int32Array(w), ca = new Float32Array(w);
    for (let i = 0; i < w; i++) {
      const u = (x0mm + (i + 0.5) * mmPerPx) / pitch - grid.ox;
      if (smooth) {
        const f = clamp(u - 0.5, 0, W - 1);
        c0[i] = Math.floor(f); c1[i] = Math.min(c0[i] + 1, W - 1); ca[i] = f - c0[i];
      } else { c0[i] = c1[i] = clamp(Math.floor(u), 0, W - 1); }
    }
    for (let j = 0; j < h; j++) {
      const u = (y0mm + (j + 0.5) * mmPerPx) / pitch - grid.oy;
      let r0, r1, b = 0;
      if (smooth) {
        const f = clamp(u - 0.5, 0, H - 1);
        r0 = Math.floor(f); r1 = Math.min(r0 + 1, H - 1); b = f - r0;
      } else { r0 = r1 = clamp(Math.floor(u), 0, H - 1); }
      const R0 = r0 * W, R1 = r1 * W;
      for (let i = 0; i < w; i++) {
        const a = ca[i], l = c0[i], r = c1[i];
        let x = smooth ? (v[R0 + l] * (1 - a) + v[R0 + r] * a) * (1 - b) + (v[R1 + l] * (1 - a) + v[R1 + r] * a) * b
                       : v[R0 + l];
        if (x !== x) x = v[(b < 0.5 ? R0 : R1) + (a < 0.5 ? l : r)];        // a neighbour is empty: nearest point
        if (x !== x) { out[j * w + i] = empty; continue; }
        const k = (x - lo) * scale;
        out[j * w + i] = lut[k <= 0 ? 0 : k >= 255 ? 255 : k | 0];
      }
    }
  }

  // ------------------------------------------------------------------ 3D

  /** Camera basis: X east, Y north (up on the 2D map), Z = value. depth grows away from the viewer.
   *  Looking straight down the perspective fades out, so the Top view is the flat map. */
  function basis(cam) {
    const a = cam.az * Math.PI / 180, e = cam.el * Math.PI / 180;
    const ca = Math.cos(a), sa = Math.sin(a), ce = Math.cos(e), se = Math.sin(e);
    const t = clamp((Math.abs(cam.el) - 60) / 30, 0, 1), far = CAMERA / Math.max(0.01, 1 - t * t * (3 - 2 * t));
    return { right: [ca, -sa, 0], up: [sa * se, ca * se, ce], depth: [sa * ce, ca * ce, -se], far };
  }
  function projector(cam, cx, cy, size) {
    const b = basis(cam), s = size * cam.zoom;
    return (x, y, z) => {
      const d = x * b.depth[0] + y * b.depth[1] + z * b.depth[2], f = b.far / (b.far + d);
      return [cx + s * f * (x * b.right[0] + y * b.right[1]),
              cy - s * f * (x * b.up[0] + y * b.up[1] + z * b.up[2]), d];
    };
  }

  /** The surface of one map: vertices in model units, rasterised with a depth buffer. */
  class Surface {
    constructor(p) {
      const g = p.grid, n = g.w * g.h;
      this.p = p;
      this.x0 = g.ox * p.pitch; this.x1 = (g.ox + g.w) * p.pitch;
      this.y0 = g.oy * p.pitch; this.y1 = (g.oy + g.h) * p.pitch;
      this.xm = (this.x0 + this.x1) / 2; this.ym = (this.y0 + this.y1) / 2;
      this.L = Math.max(this.x1 - this.x0, this.y1 - this.y0) / 2;
      this.t = new Float32Array(n);                                  // 0..1 in the colour range, NaN = empty
      const span = (p.hi - p.lo) || 1;
      for (let i = 0; i < n; i++) { const v = g.v[i]; this.t[i] = v !== v ? NaN : clamp((v - p.lo) / span, 0, 1); }
      this.sx = new Float32Array(n); this.sy = new Float32Array(n); this.sd = new Float32Array(n);
      this.shade = new Float32Array(n);
      this.w = this.h = 0;
    }
    mx(xmm) { return (xmm - this.xm) / this.L; }
    my(ymm) { return -(ymm - this.ym) / this.L; }
    mz(v, height) {
      const span = (this.p.hi - this.p.lo) || 1;
      return (clamp((v - this.p.lo) / span, 0, 1) - 0.5) * HEIGHT * height;
    }
    radius(height) {
      const hx = (this.x1 - this.x0) / 2 / this.L, hy = (this.y1 - this.y0) / 2 / this.L;
      return Math.hypot(hx, hy, HEIGHT * height / 2);
    }
    /** Project and draw into a w x h RGBA buffer (Uint32). Keeps the depth and cell of every pixel. */
    render(out, w, h, cam, cx, cy, size, height) {
      const g = this.p.grid, W = g.w, H = g.h, n = W * H, { t, sx, sy, sd, shade } = this;
      const b = basis(cam), s = size * cam.zoom, zs = HEIGHT * height, cell = this.p.pitch / this.L;
      const lx = -0.45 * b.right[0] + 0.6 * b.up[0] - 0.66 * b.depth[0];       // light over the viewer's left shoulder
      const ly = -0.45 * b.right[1] + 0.6 * b.up[1] - 0.66 * b.depth[1];
      const lz = 0.6 * b.up[2] - 0.66 * b.depth[2];
      for (let r = 0; r < H; r++) {
        const y = this.my((r + g.oy + 0.5) * this.p.pitch);
        for (let c = 0; c < W; c++) {
          const i = r * W + c, x = this.mx((c + g.ox + 0.5) * this.p.pitch), z = (t[i] - 0.5) * zs;
          const d = x * b.depth[0] + y * b.depth[1] + z * b.depth[2], f = b.far / (b.far + d);
          sx[i] = cx + s * f * (x * b.right[0] + y * b.right[1]);
          sy[i] = cy - s * f * (x * b.up[0] + y * b.up[1] + z * b.up[2]);
          sd[i] = d;
          const l = t[c > 0 ? i - 1 : i], rr = t[c < W - 1 ? i + 1 : i], u = t[r > 0 ? i - W : i], dn = t[r < H - 1 ? i + W : i];
          const nx = -(rr - l) * zs / (2 * cell), ny = (dn - u) * zs / (2 * cell), len = Math.hypot(nx, ny, 1);
          const k = (nx * lx + ny * ly + lz) / len;
          shade[i] = k === k ? 0.74 + 0.3 * Math.max(0, k) : 0.9;
        }
      }
      if (this.w !== w || this.h !== h) {
        this.depth = new Float32Array(w * h); this.cell = new Int32Array(w * h); this.w = w; this.h = h;
      }
      this.depth.fill(Infinity); this.cell.fill(-1); out.fill(0);
      for (let r = 0; r < H - 1; r++) {
        for (let c = 0; c < W - 1; c++) {
          const i = r * W + c;
          this._tri(out, i, i + 1, i + W + 1, i);
          this._tri(out, i, i + W + 1, i + W, i);
        }
      }
      return n;
    }
    _tri(out, a, b, c, cellId) {
      const { sx, sy, sd, t, shade, w, h, depth, cell } = this;
      const ta = t[a], tb = t[b], tc = t[c];
      if (ta !== ta || tb !== tb || tc !== tc) return;
      const xa = sx[a], ya = sy[a], xb = sx[b], yb = sy[b], xc = sx[c], yc = sy[c];
      const area = (xb - xa) * (yc - ya) - (xc - xa) * (yb - ya);
      if (area > -1e-7 && area < 1e-7) return;
      const minx = Math.max(0, Math.ceil(Math.min(xa, xb, xc) - 0.5)), maxx = Math.min(w - 1, Math.floor(Math.max(xa, xb, xc) - 0.5));
      const miny = Math.max(0, Math.ceil(Math.min(ya, yb, yc) - 0.5)), maxy = Math.min(h - 1, Math.floor(Math.max(ya, yb, yc) - 0.5));
      if (maxx < minx || maxy < miny) return;
      const inv = 1 / area, rgb = this.p.lut.rgb, eps = -1e-5;
      const w0dx = (yb - yc) * inv, w1dx = (yc - ya) * inv;                   // barycentric steps along x
      for (let y = miny; y <= maxy; y++) {
        const py = y + 0.5, px = minx + 0.5;
        let wa = ((xb - px) * (yc - py) - (xc - px) * (yb - py)) * inv;
        let wb = ((xc - px) * (ya - py) - (xa - px) * (yc - py)) * inv;
        let k = y * w + minx;
        for (let x = minx; x <= maxx; x++, k++, wa += w0dx, wb += w1dx) {
          const wc = 1 - wa - wb;
          if (wa < eps || wb < eps || wc < eps) continue;
          const d = wa * sd[a] + wb * sd[b] + wc * sd[c];
          if (d >= depth[k]) continue;
          depth[k] = d; cell[k] = cellId;
          const q = clamp((wa * ta + wb * tb + wc * tc) * 255, 0, 255) | 0;
          const l = wa * shade[a] + wb * shade[b] + wc * shade[c];
          out[k] = (255 << 24) | (Math.min(255, rgb[3 * q + 2] * l) << 16) | (Math.min(255, rgb[3 * q + 1] * l) << 8) | Math.min(255, rgb[3 * q] * l);
        }
      }
    }
    /** Nearest visible scan point to a pixel of the buffer, or null. */
    pick(x, y) {
      const W = this.p.grid.w;
      let id = -1;
      for (let rad = 0; rad <= 8 && id < 0; rad++) {                          // forgive a few pixels off the edge
        for (let j = -rad; j <= rad && id < 0; j++) {
          for (let i = -rad; i <= rad && id < 0; i++) {
            if (Math.max(Math.abs(i), Math.abs(j)) !== rad) continue;
            const px = Math.floor(x) + i, py = Math.floor(y) + j;
            if (px >= 0 && py >= 0 && px < this.w && py < this.h) id = this.cell[py * this.w + px];
          }
        }
      }
      if (id < 0) return null;
      let best = id, bd = Infinity;
      for (const k of [id, id + 1, id + W, id + W + 1]) {
        const d = (this.sx[k] - x) ** 2 + (this.sy[k] - y) ** 2;
        if (d < bd) { bd = d; best = k; }
      }
      return { col: (best % W) + this.p.grid.ox, row: Math.floor(best / W) + this.p.grid.oy };
    }
    /** Is a projected point (buffer pixels, depth) in front of the surface? */
    visible(x, y, d) {
      const px = Math.floor(x), py = Math.floor(y);
      if (px < 0 || py < 0 || px >= this.w || py >= this.h) return true;
      return d <= this.depth[py * this.w + px] + 0.03;
    }
  }

  // ------------------------------------------------------------------ small DOM helpers

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function button(text, title, onClick) {
    const b = el('button', 'vw-btn', text);
    b.type = 'button';
    b.title = title;
    b.addEventListener('click', onClick);
    return b;
  }
  function halo(ctx, text, x, y, color, back = 'rgba(0,0,0,.6)', width = 3) {
    ctx.lineJoin = 'round';
    ctx.strokeStyle = back; ctx.lineWidth = width; ctx.strokeText(text, x, y);
    ctx.fillStyle = color; ctx.fillText(text, x, y);
  }
  function fitText(ctx, text, width) {
    if (ctx.measureText(text).width <= width) return text;
    let lo = 0, hi = text.length;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (ctx.measureText(text.slice(0, m) + '…').width <= width) lo = m; else hi = m - 1; }
    return text.slice(0, lo) + '…';
  }

  // ------------------------------------------------------------------ the viewer

  class EctViewer {
    constructor({ bar, card, onImage }) {
      this.bar = bar; this.card = card; this.onImage = onImage || (() => {});
      this.cache = new Map();             // id -> decoded figure
      this.saved = new Map();             // id -> view state (mode, camera, zoom, pins)
      this.fig = null; this.id = null; this.s = null;
      this.hover = null; this.cells = []; this.drag = null;
      this.buildDom();
      this.frame = 0; this.needBase = false; this.idle = 0;
      new ResizeObserver(() => { if (this.fig && !this.card.hidden) this.redraw(); }).observe(this.grid);
    }

    buildDom() {
      const b = this.bar, c = this.card;
      b.textContent = ''; c.textContent = '';
      this.seg = el('div', 'vw-seg');
      this.modeBtn = {};
      for (const [mode, text, title] of [['2d', '2D', 'Flat map: hover any point'], ['3d', '3D', 'Surface: drag to rotate'],
                                         ['image', 'Image', 'The figure as a still picture']]) {
        const x = button(text, title, () => this.setMode(mode));
        this.modeBtn[mode] = x; this.seg.append(x);
      }
      this.btnBoxes = button('Boxes', 'Show or hide the crack boxes', () => { this.s.boxes = !this.s.boxes; this.sync(); });
      this.btnSmooth = button('Smooth', 'Smooth between scan points, or one flat cell per measured point', () => { this.s.smooth = !this.s.smooth; this.sync(); });
      this.btnProfile = button('Profile', 'Values along a cut through the map', () => { this.s.profile.on = !this.s.profile.on; this.sync(true); });
      this.presets = el('div', 'vw-group');
      for (const [name, az, elv] of [['Top', 0, 90], ['Front', 0, 18], ['Side', 90, 18]])
        this.presets.append(button(name, `Look from the ${name.toLowerCase()}`, () => { Object.assign(this.s.cam, HOME, { az, el: elv }); this.redraw(); }));
      const hl = el('label', 'vw-range', 'Height');
      this.height = el('input');
      Object.assign(this.height, { type: 'range', min: '0.2', max: '3', step: '0.1', title: 'Height of the surface' });
      this.height.addEventListener('input', () => { this.s.height = Number(this.height.value); this.redraw(true); });
      hl.append(this.height);
      this.presets.append(hl);
      this.btnReset = button('Reset view', 'Back to the first view', () => this.resetView());
      this.btnAll = button('All maps', 'Show every map of this figure again', () => this.setFocus(null));
      b.append(this.seg, this.btnBoxes, this.btnSmooth, this.btnProfile, this.presets, this.btnReset, this.btnAll);

      this.sub = el('div', 'vw-sub');
      this.grid = el('div', 'vw-grid');
      this.prof = el('div', 'vw-profile');
      const ph = el('div', 'vw-profile-head');
      this.profAlong = el('div', 'vw-seg');
      this.profBtn = {};
      for (const a of ['X', 'Y']) {
        const x = button('Along ' + a, `Values along ${a}`, () => this.setCut(a, null));
        this.profBtn[a] = x; this.profAlong.append(x);
      }
      this.profPos = el('input');
      Object.assign(this.profPos, { type: 'range', title: 'Position of the cut' });
      this.profPos.addEventListener('input', () => this.setCutIndex(Number(this.profPos.value)));
      this.profText = el('span', 'vw-profile-text');
      ph.append(this.profAlong, this.profPos, this.profText);
      this.profCanvas = el('canvas');
      this.profCanvas.addEventListener('pointermove', (e) => this.onProfileMove(e));
      this.profCanvas.addEventListener('pointerleave', () => this.setHover(null));
      this.profCanvas.addEventListener('click', () => this.pin());
      this.prof.append(ph, this.profCanvas);
      this.pinBox = el('div', 'vw-pins');
      this.hint = el('div', 'vw-hint');
      this.tip = el('div', 'vw-tip');
      this.tip.hidden = true;
      c.append(this.sub, this.grid, this.prof, this.hint, this.pinBox, this.tip);
    }

    // ---------------------------------------------------------------- loading

    async load(id, url) {
      this.store();
      this.id = id;
      this.hover = null; this.drag = null; this.tip.hidden = true;
      let fig = this.cache.get(id);
      if (!fig) {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        fig = this.decode(await res.json());
        this.cache.set(id, fig);
        if (this.cache.size > CACHE) this.cache.delete(this.cache.keys().next().value);
      }
      if (this.id !== id) return false;                // another figure was chosen meanwhile
      this.fig = fig;
      this.s = this.saved.get(id) || this.freshState(fig);
      this.sync(true);
      return true;
    }

    decode(j) {
      const luts = {};
      for (const k of Object.keys(j.cmaps)) luts[k] = decodeLut(j.cmaps[k]);
      const panels = j.panels.map((p, index) => {
        const q = { ...p, index };
        if (p.type === 'map') {
          q.grid = decodeGrid(p); q.lut = luts[p.cmap];
          q.readouts = p.readouts.map((r) => ({ name: r.name, grid: decodeGrid(r) }));
        } else if (p.type === 'rgb') q.rgb = bytes(p.u8);
        else { q.y = f32(p.f32); const [lo, hi] = EctViewer.range(q.y); q.span = hi - lo; }
        delete q.f32; delete q.u8;
        return q;
      });
      for (const p of panels) {
        if (p.type === 'line') continue;
        p.ext = { x0: p.ox * p.pitch, x1: (p.ox + p.w) * p.pitch, y0: p.oy * p.pitch, y1: (p.oy + p.h) * p.pitch };
      }
      for (const p of panels) {                        // linked maps share one frame, so places line up
        if (p.type === 'line') continue;
        const peers = p.link ? panels.filter((q) => q.link === p.link) : [p];
        p.peers = peers;
        p.frame = { x0: Math.min(...peers.map((q) => q.ext.x0)), x1: Math.max(...peers.map((q) => q.ext.x1)),
                    y0: Math.min(...peers.map((q) => q.ext.y0)), y1: Math.max(...peers.map((q) => q.ext.y1)) };
      }
      return { ...j, panels };
    }

    freshState(fig) {
      const maps = fig.panels.filter((p) => p.type === 'map');
      const first = maps[0], pr = fig.profile;
      const s = { mode: fig.mode === '3d' && first ? '3d' : '2d', focus: null, boxes: true, smooth: true, cam: { ...HOME },
                  height: 1, views: {}, pins: [],
                  profile: { on: !!pr, along: pr ? pr.along : 'X', pos: pr ? pr.pos : null } };
      if (first && s.profile.pos === null) s.profile.pos = (first.ext.y0 + first.ext.y1) / 2;
      return s;
    }

    store() { if (this.id && this.s && this.fig) this.saved.set(this.id, this.s); }

    hide() {
      this.store();
      this.fig = null; this.id = null; this.s = null;
      this.bar.hidden = true; this.card.hidden = true; this.tip.hidden = true;
    }

    forget() { this.hide(); this.saved.clear(); this.cache.clear(); }

    get active() { return !!this.fig && this.s.mode !== 'image'; }

    setMode(mode) { this.s.mode = mode; this.hover = null; this.tip.hidden = true; this.sync(true); }

    setFocus(index) { this.s.focus = index; this.hover = null; this.tip.hidden = true; this.sync(true); }

    resetView() {
      const s = this.s;
      if (s.mode === '3d') { s.cam = { ...HOME }; s.height = 1; } else s.views = {};
      this.sync();
    }

    // ---------------------------------------------------------------- state -> screen

    visiblePanels() {
      const all = this.fig.panels;
      return this.s.focus === null ? all : [all[this.s.focus]];
    }
    /** The one map a profile can belong to (a single map on screen), or null. */
    soloMap() {
      const v = this.visiblePanels();
      return v.length === 1 && v[0].type === 'map' ? v[0] : null;
    }

    sync(relayout = false) {
      const s = this.s, fig = this.fig, vis = this.visiblePanels();
      const maps = vis.filter((p) => p.type === 'map'), d3 = s.mode === '3d', image = s.mode === 'image';
      for (const m of ['2d', '3d', 'image']) {
        this.modeBtn[m].classList.toggle('on', s.mode === m);
        this.modeBtn[m].setAttribute('aria-pressed', String(s.mode === m));
      }
      this.modeBtn['3d'].hidden = !fig.panels.some((p) => p.type === 'map');
      const toggle = (b, show, on) => { b.hidden = !show || image; b.classList.toggle('on', !!on); b.setAttribute('aria-pressed', String(!!on)); };
      toggle(this.btnBoxes, vis.some((p) => p.boxes.length), s.boxes);
      toggle(this.btnSmooth, maps.length && !d3, s.smooth);
      toggle(this.btnProfile, !!this.soloMap(), s.profile.on);
      this.presets.hidden = image || !d3 || !maps.length;
      this.height.value = String(s.height);
      this.btnReset.hidden = image;
      this.btnAll.hidden = image || s.focus === null;
      this.hint.textContent = image ? '' : d3 ? 'Drag: rotate · Shift+drag: move · Wheel: zoom · Click: pin a point'
                                              : 'Hover: values · Click: pin a point · Drag: move · Wheel: zoom';
      this.bar.hidden = false;
      this.card.hidden = image;
      this.onImage(image);
      if (image) { this.tip.hidden = true; return; }
      this.renderSub(vis);
      this.prof.hidden = !(this.soloMap() && s.profile.on);
      if (relayout) this.layout();
      this.renderPins();
      this.redraw();
    }

    renderSub(vis) {
      this.sub.textContent = '';
      this.sub.append(el('span', 'vw-sub-text', this.fig.subtitle || ''));
      const kinds = new Set(vis.flatMap((p) => p.boxes.map((b) => b.kind)));
      if (!this.s.boxes || ![...kinds].some((k) => k !== 'plain')) return;
      for (const [k, text] of [['found', 'matches a label'], ['false', 'no matching label'], ['missed', 'missed label']]) {
        if (!kinds.has(k)) continue;
        const item = el('span', 'vw-key'), sw = el('i', k === 'missed' ? 'dotted' : '');
        sw.style.borderColor = this.fig.box_colors[k];
        item.append(sw, el('span', '', text));
        this.sub.append(item);
      }
    }

    layout() {
      const vis = this.visiblePanels(), n = vis.length;
      this.grid.textContent = '';
      this.cols = 0;
      this.arrange(vis);
      this.cells = vis.map((p) => {
        const box = el('div', 'vw-panel'), base = el('canvas'), over = el('canvas', 'vw-over');
        over.tabIndex = 0;
        over.setAttribute('role', 'img');
        over.setAttribute('aria-label', `${p.title}. Arrow keys move the readout, Enter pins the point.`);
        box.append(base, over);
        if (this.fig.panels.length > 1) {
          const max = button(n > 1 ? '⤢' : '⤡', n > 1 ? 'Show only this one' : 'Show all', () => this.setFocus(n > 1 ? p.index : null));
          max.classList.add('vw-max');
          box.append(max);
        }
        this.grid.append(box);
        const cell = { p, box, base, over, surface: null, off: null };
        this.bind(cell);
        return cell;
      });
    }

    /** The number of columns that shows the maps largest in the space there is. */
    arrange(vis) {
      const n = vis.length, w = this.grid.clientWidth, h = this.grid.clientHeight, p = vis.find((q) => q.frame) || vis[0], f = p.frame;
      const ratio = f ? (f.x1 - f.x0) / (f.y1 - f.y0) : 1.5;
      let best = Math.min(this.fig.ncols || 2, n), top = -1;
      for (let cols = 1; cols <= n && w > 0 && h > 0; cols++) {
        const rows = Math.ceil(n / cols), size = Math.min((w / cols - 110) / ratio, h / rows - 48);
        if (size > top + 1) { top = size; best = cols; }
      }
      if (best === this.cols) return;
      this.cols = best;
      this.grid.style.gridTemplateColumns = `repeat(${best}, minmax(0, 1fr))`;
      this.grid.style.gridTemplateRows = `repeat(${Math.ceil(n / best)}, minmax(0, 1fr))`;
    }

    redraw(fast = false) {
      this.needBase = true;
      this.fast = fast;
      if (!this.frame) this.frame = requestAnimationFrame(() => { this.frame = 0; this.paint(); });
      clearTimeout(this.idle);
      if (fast) this.idle = setTimeout(() => this.redraw(false), 140);      // sharp picture once the mouse rests
    }
    refresh() {
      if (!this.frame) this.frame = requestAnimationFrame(() => { this.frame = 0; this.paint(); });
    }

    paint() {
      if (!this.active) return;
      if (this.needBase) {
        this.arrange(this.visiblePanels());
        for (const c of this.cells) this.paintBase(c);
        this.needBase = false;
      }
      for (const c of this.cells) this.paintOverlay(c);
      if (!this.prof.hidden) this.paintProfile();
    }

    sizeCanvas(cv, w, h) {
      const dpr = window.devicePixelRatio || 1, W = Math.max(1, Math.round(w * dpr)), H = Math.max(1, Math.round(h * dpr));
      if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
      const ctx = cv.getContext('2d');
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, cv.width, cv.height);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return ctx;
    }

    /** Margins and plot area of a panel, in CSS px. */
    areaOf(c) {
      const w = c.box.clientWidth, h = c.box.clientHeight, tight = w < 330 || h < 240, p = c.p;
      const L = p.type === 'line' ? 56 : tight ? 34 : 50, T = 24, B = tight ? 18 : 38;
      const R = p.type === 'map' ? 28 + this.barWidth(p, h) + (tight ? 2 : 22) : 14;
      return { w, h, tight, L, T, R, B, pw: Math.max(10, w - L - R), ph: Math.max(10, h - T - B) };
    }

    /** Width of the widest number on the colour bar. */
    barWidth(p, h) {
      const ctx = this.measure || (this.measure = el('canvas').getContext('2d')), t = ticks(p.lo, p.hi, Math.max(40, h * 0.8) / 38);
      ctx.font = `10.5px ${FONT}`;
      return Math.max(...t.values.map((v) => ctx.measureText(tickText(v, t.step)).width));
    }

    title(ctx, c, x, y) {
      ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left';
      ctx.fillStyle = INK; ctx.font = `600 12px ${FONT}`;
      ctx.fillText(fitText(ctx, c.p.title, c.area.w - x - (this.fig.panels.length > 1 ? 40 : 10)), x, y);
    }

    paintBase(c) {
      const a = c.area = this.areaOf(c), ctx = this.sizeCanvas(c.base, a.w, a.h), p = c.p;
      if (p.type === 'line') this.paintLine(c, ctx);
      else if (this.s.mode === '3d' && p.type === 'map') this.paint3d(c, ctx);
      else this.paint2d(c, ctx);
    }

    // ---------------------------------------------------------------- 2D map

    viewOf(p) {
      const key = p.link ? 'L:' + p.link : 'P:' + p.index, f = p.frame;
      let v = this.s.views[key];
      if (!v) {
        v = this.s.views[key] = { cx: (f.x0 + f.x1) / 2, cy: (f.y0 + f.y1) / 2, z: 1 };
        if (p.window) {
          const [x0, y0, x1, y1] = p.window;
          v.cx = (x0 + x1) / 2; v.cy = (y0 + y1) / 2; v.window = p.window;
        }
      }
      return v;
    }
    /** mm <-> CSS px of one panel. */
    frameOf(c) {
      const p = c.p, a = c.area, f = p.frame, v = this.viewOf(p);
      const fit = Math.min(a.pw / (f.x1 - f.x0), a.ph / (f.y1 - f.y0));
      if (v.window) {                                     // first showing of a zoomed figure
        const [x0, y0, x1, y1] = v.window;
        v.z = Math.min(a.pw / (x1 - x0), a.ph / (y1 - y0)) / fit;
        delete v.window;
      }
      const s = fit * v.z;
      const keep = (mid, lo, hi, px) => { const half = px / s / 2; return hi - lo <= 2 * half ? (lo + hi) / 2 : clamp(mid, lo + half, hi - half); };
      v.cx = keep(v.cx, f.x0, f.x1, a.pw); v.cy = keep(v.cy, f.y0, f.y1, a.ph);
      const X = (x) => a.L + a.pw / 2 + (x - v.cx) * s, Y = (y) => a.T + a.ph / 2 + (y - v.cy) * s;
      const e = p.ext;
      const clip = { x0: Math.max(a.L, X(e.x0)), x1: Math.min(a.L + a.pw, X(e.x1)), y0: Math.max(a.T, Y(e.y0)), y1: Math.min(a.T + a.ph, Y(e.y1)) };
      return { s, fit, X, Y, clip, ix: (px) => v.cx + (px - a.L - a.pw / 2) / s, iy: (py) => v.cy + (py - a.T - a.ph / 2) / s };
    }

    paint2d(c, ctx) {
      const p = c.p, a = c.area, fr = c.fr = this.frameOf(c), k = fr.clip, dpr = window.devicePixelRatio || 1;
      const w = Math.round((k.x1 - k.x0) * dpr), h = Math.round((k.y1 - k.y0) * dpr);
      if (w < 1 || h < 1) return;
      const img = new ImageData(w, h), px = new Uint32Array(img.data.buffer), per = 1 / (fr.s * dpr);
      if (p.type === 'rgb') this.paintRgb(px, w, h, p, fr.ix(k.x0), fr.iy(k.y0), per);
      else paintMap(px, w, h, p.grid, p.lo, p.hi, p.lut.packed, fr.ix(k.x0), fr.iy(k.y0), per, p.pitch, this.s.smooth);
      const x0 = Math.round(k.x0 * dpr), y0 = Math.round(k.y0 * dpr);
      ctx.putImageData(img, x0, y0);

      ctx.save();
      ctx.beginPath(); ctx.rect(k.x0, k.y0, k.x1 - k.x0, k.y1 - k.y0); ctx.clip();
      for (const [x, y, sw, sh] of p.shade || []) { ctx.fillStyle = 'rgba(255,77,79,.28)'; ctx.fillRect(fr.X(x), fr.Y(y), sw * fr.s, sh * fr.s); }
      if (p.rect) {
        ctx.strokeStyle = '#22d3ee'; ctx.lineWidth = 1.8; ctx.setLineDash([6, 4]);
        ctx.strokeRect(fr.X(p.rect[0]), fr.Y(p.rect[1]), (p.rect[2] - p.rect[0]) * fr.s, (p.rect[3] - p.rect[1]) * fr.s);
        ctx.setLineDash([]);
      }
      if (this.s.boxes) {
        ctx.font = `600 11px ${FONT}`; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
        for (const b of p.boxes) {
          const x = fr.X(b.x0 * p.pitch), y = fr.Y(b.y0 * p.pitch), bw = (b.x1 - b.x0) * p.pitch * fr.s, bh = (b.y1 - b.y0) * p.pitch * fr.s;
          const col = this.fig.box_colors[b.kind];
          ctx.lineWidth = 1.6; ctx.strokeStyle = col; ctx.setLineDash(b.kind === 'missed' ? [2, 3] : []);
          ctx.strokeRect(x, y, bw, bh); ctx.setLineDash([]);
          if (b.n !== null) {
            const tag = bw > 34 && b.conf !== undefined ? `${b.n} ${b.conf.toFixed(2)}` : String(b.n);
            halo(ctx, tag, Math.max(x, k.x0 + 2), Math.max(y - 1, k.y0 + 13), col);
          }
        }
      }
      for (const ln of p.lines || []) {
        const [[ax, ay], [bx, by]] = ln.pts;
        ctx.lineWidth = 2.4; ctx.strokeStyle = ln.color;
        ctx.beginPath(); ctx.moveTo(fr.X(ax), fr.Y(ay)); ctx.lineTo(fr.X(bx), fr.Y(by)); ctx.stroke();
        ctx.font = `600 11px ${FONT}`; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
        const [lx, ly] = ax + ay > bx + by ? [ax, ay] : [bx, by];
        halo(ctx, ln.label, fr.X(lx) + 8, fr.Y(ly), ln.color);
      }
      ctx.restore();
      this.title(ctx, c, k.x0, Math.max(15, k.y0 - 8));
      this.axes2d(ctx, a, fr);
      if (p.type === 'map') this.colorBar(ctx, c, { x: k.x1 + 10, y0: k.y0, y1: k.y1 });
    }

    paintRgb(px, w, h, p, x0, y0, per) {
      for (let j = 0; j < h; j++) {
        const r = clamp(Math.floor((y0 + (j + 0.5) * per) / p.pitch) - p.oy, 0, p.h - 1);
        for (let i = 0; i < w; i++) {
          const q = 3 * (r * p.w + clamp(Math.floor((x0 + (i + 0.5) * per) / p.pitch) - p.ox, 0, p.w - 1));
          px[j * w + i] = (255 << 24) | (p.rgb[q + 2] << 16) | (p.rgb[q + 1] << 8) | p.rgb[q];
        }
      }
    }

    axes2d(ctx, a, fr) {
      const k = fr.clip;
      ctx.strokeStyle = LINE; ctx.lineWidth = 1;
      ctx.strokeRect(k.x0 - 0.5, k.y0 - 0.5, k.x1 - k.x0 + 1, k.y1 - k.y0 + 1);
      ctx.fillStyle = MUTED; ctx.font = `11px ${FONT}`;
      const xt = ticks(fr.ix(k.x0), fr.ix(k.x1), (k.x1 - k.x0) / 70), yt = ticks(fr.iy(k.y0), fr.iy(k.y1), (k.y1 - k.y0) / 45);
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const v of xt.values) { const x = fr.X(v); ctx.beginPath(); ctx.moveTo(x, k.y1); ctx.lineTo(x, k.y1 + 4); ctx.stroke(); ctx.fillText(tickText(v, xt.step), x, k.y1 + 5); }
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of yt.values) { const y = fr.Y(v); ctx.beginPath(); ctx.moveTo(k.x0 - 4, y); ctx.lineTo(k.x0, y); ctx.stroke(); ctx.fillText(tickText(v, yt.step), k.x0 - 6, y); }
      if (a.tight) return;
      ctx.fillStyle = INK; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText('X (mm)', (k.x0 + k.x1) / 2, k.y1 + 20);
      ctx.save(); ctx.translate(k.x0 - 36, (k.y0 + k.y1) / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'middle';
      ctx.fillText('Y (mm)', 0, 0); ctx.restore();
    }

    colorBar(ctx, c, at) {
      const p = c.p, w = 11, tip = 6, y0 = at.y0 + tip, y1 = Math.max(at.y1 - tip, y0 + 20), x = at.x, rgb = p.lut.rgb;
      const css = (i) => `rgb(${rgb[3 * i]},${rgb[3 * i + 1]},${rgb[3 * i + 2]})`;
      const grad = ctx.createLinearGradient(0, y1, 0, y0);
      for (let i = 0; i <= 32; i++) grad.addColorStop(i / 32, css(Math.round(i * 255 / 32)));
      ctx.fillStyle = grad; ctx.fillRect(x, y0, w, y1 - y0);
      const tri = (ya, yb, col) => { ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(x, ya); ctx.lineTo(x + w, ya); ctx.lineTo(x + w / 2, yb); ctx.fill(); };
      tri(y0, y0 - tip, css(255)); tri(y1, y1 + tip, css(0));                 // values beyond the colour range
      ctx.strokeStyle = LINE; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y0, w - 1, y1 - y0);
      const t = ticks(p.lo, p.hi, (y1 - y0) / 38), Yv = (v) => y1 - (clamp(v, p.lo, p.hi) - p.lo) / ((p.hi - p.lo) || 1) * (y1 - y0);
      ctx.font = `10.5px ${FONT}`; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = MUTED;
      let widest = 0;
      for (const v of t.values) { const s = tickText(v, t.step); widest = Math.max(widest, ctx.measureText(s).width); ctx.fillText(s, x + w + 5, Yv(v)); }
      c.bar = { x, w, Yv };
      const tx = x + w + 5 + widest + 10;
      if (p.label && !c.area.tight && tx < c.area.w - 4 && ctx.measureText(p.label).width < y1 - y0) {
        ctx.save(); ctx.translate(tx, (y0 + y1) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.fillStyle = MUTED;
        ctx.fillText(p.label, 0, 0); ctx.restore();
      }
    }

    // ---------------------------------------------------------------- 3D surface

    paint3d(c, ctx) {
      const p = c.p, a = c.area, s = this.s, dpr = window.devicePixelRatio || 1;
      const sf = c.surface || (c.surface = new Surface(p));
      const R = sf.radius(s.height), edge = a.tight ? 14 : 34;
      const size = Math.max(10, Math.min((a.pw - edge) / (2 * R), (a.ph - edge) / (2 * R * 0.8)));
      const cx = a.L + a.pw / 2 + s.cam.px, cy = a.T + a.ph * 0.47 + s.cam.py;
      const proj = c.proj = projector(s.cam, cx, cy, size);
      c.sf = sf;
      let q = dpr * (this.fast ? 1 : 2);                                     // supersample when nothing moves
      const limit = 3.2e6 / this.cells.length;
      q = Math.min(q, Math.sqrt(limit / (a.w * a.h)));
      const w = Math.max(1, Math.round(a.w * q)), h = Math.max(1, Math.round(a.h * q));
      if (!c.off || c.off.width !== w || c.off.height !== h) {
        c.off = document.createElement('canvas'); c.off.width = w; c.off.height = h;
        c.img = c.off.getContext('2d').createImageData(w, h);
        c.px = new Uint32Array(c.img.data.buffer);
      }
      c.q = w / a.w;
      sf.render(c.px, w, h, s.cam, cx * c.q, cy * c.q, size * c.q, s.height);
      c.off.getContext('2d').putImageData(c.img, 0, 0);

      const draw = () => { ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high'; ctx.drawImage(c.off, 0, 0, a.w, a.h); };
      const f = this.frame3d(c);
      this.walls3d(ctx, c, f);
      if (s.cam.el >= 0) { this.floor3d(ctx, c, f); draw(); } else { draw(); this.floor3d(ctx, c, f); }
      if (s.boxes) this.rings3d(ctx, c);
      for (const ln of p.lines || []) this.path3d(ctx, c, this.sampleLine(p, ln.pts), ln.color, 2.4);
      if (this.soloMap() === p && s.profile.on) this.path3d(ctx, c, this.cutPoints(p), CUT, 2.2);
      const barTop = a.T + a.ph * 0.12, barBottom = a.T + a.ph * 0.88;
      this.colorBar(ctx, c, { x: a.w - a.R + 8, y0: barTop, y1: barBottom });
      this.title(ctx, c, a.L, 15);
    }

    pos3d(c, col, row, v) {                                                  // a scan point on the surface
      const sf = c.sf, p = c.p;
      return c.proj(sf.mx((col + 0.5) * p.pitch), sf.my((row + 0.5) * p.pitch), sf.mz(v, this.s.height));
    }

    /** Corners, ticks and label edges of the 3D frame. Labels go on the two floor edges lowest on screen. */
    frame3d(c) {
      const sf = c.sf, p = c.p, P = c.proj, h = HEIGHT * this.s.height, z0 = -h / 2, small = c.area.tight;
      const X0 = sf.mx(sf.x0), X1 = sf.mx(sf.x1), Y0 = sf.my(sf.y0), Y1 = sf.my(sf.y1);
      const lower = (a, b) => P(...a)[1] > P(...b)[1];
      return { X0, X1, Y0, Y1, z0, z1: h / 2, small, zOf: (v) => ((v - p.lo) / ((p.hi - p.lo) || 1) - 0.5) * h,
               xt: ticks(sf.x0, sf.x1, small ? 4 : 7), yt: ticks(sf.y0, sf.y1, small ? 3 : 5), zt: ticks(p.lo, p.hi, small ? 3 : 5),
               edgeY: lower([0, Y0, z0], [0, Y1, z0]) ? Y0 : Y1, edgeX: lower([X0, 0, z0], [X1, 0, z0]) ? X0 : X1 };
    }
    static line(ctx, p0, p1, col, w = 1) { ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]); ctx.stroke(); }

    walls3d(ctx, c, f) {
      if (Math.cos(this.s.cam.el * Math.PI / 180) <= 0.2) return;              // edge-on from the top
      const P = c.proj, line = EctViewer.line, { X0, X1, Y0, Y1, z0, z1, zOf, zt } = f;
      const backY = P(0, Y0, 0)[2] > P(0, Y1, 0)[2] ? Y0 : Y1, backX = P(X0, 0, 0)[2] > P(X1, 0, 0)[2] ? X0 : X1;
      for (const v of zt.values) { line(ctx, P(X0, backY, zOf(v)), P(X1, backY, zOf(v)), GRID); line(ctx, P(backX, Y0, zOf(v)), P(backX, Y1, zOf(v)), GRID); }
      // value axis on the leftmost upright edge
      const up = [[X0, Y0], [X1, Y0], [X1, Y1], [X0, Y1]].reduce((b, q) => (P(q[0], q[1], 0)[0] < P(b[0], b[1], 0)[0] ? q : b));
      line(ctx, P(up[0], up[1], z0), P(up[0], up[1], z1), LINE);
      ctx.font = `10.5px ${FONT}`; ctx.fillStyle = MUTED; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of zt.values) { const q = P(up[0], up[1], zOf(v)); line(ctx, q, [q[0] - 4, q[1]], LINE); ctx.fillText(tickText(v, zt.step), q[0] - 7, q[1]); }
    }

    floor3d(ctx, c, f) {
      const sf = c.sf, P = c.proj, line = EctViewer.line, { X0, X1, Y0, Y1, z0, xt, yt } = f;
      if (this.s.cam.el >= 0) {                                                // from below the grid would cross the surface
        for (const v of xt.values) line(ctx, P(sf.mx(v), Y0, z0), P(sf.mx(v), Y1, z0), GRID);
        for (const v of yt.values) line(ctx, P(X0, sf.my(v), z0), P(X1, sf.my(v), z0), GRID);
      }
      for (const [a, b] of [[[X0, Y0], [X1, Y0]], [[X1, Y0], [X1, Y1]], [[X1, Y1], [X0, Y1]], [[X0, Y1], [X0, Y0]]])
        line(ctx, P(a[0], a[1], z0), P(b[0], b[1], z0), LINE);
      const mid = P(0, 0, z0), label = (e0, e1, marks, name) => {              // numbers pushed away from the middle
        const cx = (e0[0] + e1[0]) / 2, cy = (e0[1] + e1[1]) / 2, l = Math.hypot(cx - mid[0], cy - mid[1]) || 1, ux = (cx - mid[0]) / l, uy = (cy - mid[1]) / l;
        ctx.font = `10.5px ${FONT}`; ctx.fillStyle = MUTED; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        for (const [q, text] of marks) { line(ctx, q, [q[0] + ux * 5, q[1] + uy * 5], LINE); ctx.fillText(text, q[0] + ux * 15, q[1] + uy * 13); }
        if (!f.small) { ctx.fillStyle = INK; ctx.font = `11px ${FONT}`; ctx.fillText(name, cx + ux * 50, cy + uy * 34); }
      };
      label(P(X0, f.edgeY, z0), P(X1, f.edgeY, z0), xt.values.map((v) => [P(sf.mx(v), f.edgeY, z0), tickText(v, xt.step)]), 'X (mm)');
      label(P(f.edgeX, Y0, z0), P(f.edgeX, Y1, z0), yt.values.map((v) => [P(f.edgeX, sf.my(v), z0), tickText(v, yt.step)]), 'Y (mm)');
    }

    /** A polyline on the surface ([col, row] absolute points); hidden stretches are drawn faint. */
    path3d(ctx, c, pts, color, width) {
      let prev = null;
      ctx.lineWidth = width; ctx.strokeStyle = color; ctx.lineCap = 'round';
      for (const [col, row] of pts) {
        const v = valueAt(c.p.grid, col, row);
        if (!Number.isFinite(v)) { prev = null; continue; }
        const q = this.pos3d(c, col, row, v), seen = c.sf.visible(q[0] * c.q, q[1] * c.q, q[2]);
        if (prev) {
          ctx.globalAlpha = seen && prev.seen ? 1 : 0.22;
          ctx.beginPath(); ctx.moveTo(prev.q[0], prev.q[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
        }
        prev = { q, seen };
      }
      ctx.globalAlpha = 1; ctx.lineCap = 'butt';
    }

    rings3d(ctx, c) {
      const g = c.p.grid;
      for (const b of c.p.boxes) {
        const x0 = Math.max(b.x0, g.ox), x1 = Math.min(b.x1, g.ox + g.w) - 1, y0 = Math.max(b.y0, g.oy), y1 = Math.min(b.y1, g.oy + g.h) - 1;
        if (x1 <= x0 || y1 <= y0) continue;
        const ring = [];
        for (let x = x0; x <= x1; x++) ring.push([x, y0]);
        for (let y = y0; y <= y1; y++) ring.push([x1, y]);
        for (let x = x1; x >= x0; x--) ring.push([x, y1]);
        for (let y = y1; y >= y0; y--) ring.push([x0, y]);
        const col = this.fig.box_colors[b.kind];
        if (b.kind === 'missed') ctx.setLineDash([2, 3]);
        this.path3d(ctx, c, ring, col, 1.6);
        ctx.setLineDash([]);
        if (b.n !== null) {
          const q = this.pos3d(c, x0, y0, valueAt(g, x0, y0));
          ctx.font = `600 11px ${FONT}`; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
          halo(ctx, String(b.n), q[0], q[1] - 3, col);
        }
      }
    }

    sampleLine(p, [[x0, y0], [x1, y1]]) {
      const n = Math.max(2, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / p.pitch * 2)), out = [];
      for (let i = 0; i <= n; i++) out.push([Math.floor((x0 + (x1 - x0) * i / n) / p.pitch), Math.floor((y0 + (y1 - y0) * i / n) / p.pitch)]);
      return out;
    }

    // ---------------------------------------------------------------- line charts

    chartFrame(a, x0, x1, lo, hi) {
      const pad = (hi - lo || Math.abs(hi) || 1) * 0.06;
      lo -= pad; hi += pad;
      return { lo, hi, X: (x) => a.L + (x - x0) / ((x1 - x0) || 1) * a.pw, Y: (y) => a.T + a.ph - (y - lo) / (hi - lo) * a.ph,
               ix: (px) => x0 + (px - a.L) / a.pw * (x1 - x0) };
    }
    chartAxes(ctx, a, f, x0, x1, xlabel, ylabel) {
      const xt = ticks(x0, x1, a.pw / 70), yt = ticks(f.lo, f.hi, a.ph / 32);
      ctx.font = `10.5px ${FONT}`; ctx.lineWidth = 1; ctx.strokeStyle = GRID; ctx.fillStyle = MUTED;
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of yt.values) { const y = f.Y(v); ctx.beginPath(); ctx.moveTo(a.L, y); ctx.lineTo(a.L + a.pw, y); ctx.stroke(); ctx.fillText(tickText(v, yt.step), a.L - 5, y); }
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const v of xt.values) ctx.fillText(tickText(v, xt.step), f.X(v), a.T + a.ph + 5);
      ctx.strokeStyle = LINE; ctx.beginPath(); ctx.moveTo(a.L, a.T + a.ph); ctx.lineTo(a.L + a.pw, a.T + a.ph); ctx.stroke();
      ctx.fillStyle = INK; ctx.font = `11px ${FONT}`;
      if (!a.tight) ctx.fillText(xlabel, a.L + a.pw / 2, a.T + a.ph + 19);
      if (ylabel) { ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillStyle = MUTED; ctx.fillText(ylabel, a.L, a.T - 3); }
    }
    static range(y) {
      let lo = Infinity, hi = -Infinity;
      for (const v of y) if (v === v) { if (v < lo) lo = v; if (v > hi) hi = v; }
      return [lo, hi];
    }
    static stroke(ctx, f, xs, ys) {
      ctx.strokeStyle = INK; ctx.lineWidth = 1.5; ctx.lineJoin = 'round'; ctx.beginPath();
      let pen = false;
      for (let i = 0; i < ys.length; i++) {
        if (ys[i] !== ys[i]) { pen = false; continue; }
        if (pen) ctx.lineTo(f.X(xs(i)), f.Y(ys[i])); else ctx.moveTo(f.X(xs(i)), f.Y(ys[i]));
        pen = true;
      }
      ctx.stroke();
    }

    paintLine(c, ctx) {
      const p = c.p, a = { ...c.area, T: 38, ph: Math.max(10, c.area.ph - 14) }, [lo, hi] = EctViewer.range(p.y);
      c.area = a;
      this.title(ctx, c, a.L, 15);
      const f = c.chart = this.chartFrame(a, 0, p.n - 1, lo, hi);
      this.chartAxes(ctx, a, f, 0, p.n - 1, p.xlabel, '');
      ctx.strokeStyle = COPPER; ctx.setLineDash([5, 4]); ctx.lineWidth = 1;
      for (const v of p.vlines) { ctx.beginPath(); ctx.moveTo(f.X(v), a.T); ctx.lineTo(f.X(v), a.T + a.ph); ctx.stroke(); }
      ctx.setLineDash([]);
      ctx.fillStyle = COPPER; ctx.font = `11px ${FONT}`; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      for (const [x, t] of p.notes) ctx.fillText(t, f.X(x), a.T - 2);
      EctViewer.stroke(ctx, f, (i) => i, p.y);
    }

    // ---------------------------------------------------------------- profile

    cutIndex(p) {                                                            // absolute row (along X) or column (along Y)
      const pr = this.s.profile, i = Math.floor(pr.pos / p.pitch + 1e-9);
      return pr.along === 'X' ? clamp(i, p.oy, p.oy + p.h - 1) : clamp(i, p.ox, p.ox + p.w - 1);
    }
    cutPoints(p) {
      const i = this.cutIndex(p), out = [];
      if (this.s.profile.along === 'X') for (let c = 0; c < p.w; c++) out.push([c + p.ox, i]);
      else for (let r = 0; r < p.h; r++) out.push([i, r + p.oy]);
      return out;
    }
    setCut(along, pos) {
      const pr = this.s.profile, p = this.soloMap();
      if (along && along !== pr.along) { pr.along = along; pr.pos = along === 'X' ? (p.ext.y0 + p.ext.y1) / 2 : (p.ext.x0 + p.ext.x1) / 2; }
      if (pos !== null) pr.pos = pos;
      this.s.mode === '3d' ? this.redraw(true) : this.refresh();
    }
    setCutIndex(i) { const p = this.soloMap(); if (p) this.setCut(null, (i + 0.5) * p.pitch); }

    paintProfile() {
      const p = this.soloMap();
      if (!p) return;
      const pr = this.s.profile, X = pr.along === 'X', i = this.cutIndex(p), pts = this.cutPoints(p);
      const y = pts.map(([c, r]) => valueAt(p.grid, c, r)), at = (k) => ((X ? pts[k][0] : pts[k][1]) + 0.5) * p.pitch;
      for (const k of ['X', 'Y']) this.profBtn[k].classList.toggle('on', pr.along === k);
      Object.assign(this.profPos, X ? { min: p.oy, max: p.oy + p.h - 1 } : { min: p.ox, max: p.ox + p.w - 1 });
      this.profPos.value = String(i);
      this.profText.textContent = `at ${X ? 'Y' : 'X'} = ${mm((i + 0.5) * p.pitch)} mm (${X ? 'row' : 'col'} ${i})`;
      const w = this.profCanvas.clientWidth, h = this.profCanvas.clientHeight, ctx = this.sizeCanvas(this.profCanvas, w, h);
      const a = { L: 56, T: 20, pw: Math.max(10, w - 56 - 16), ph: Math.max(10, h - 20 - 34), tight: false };
      const x0 = at(0), x1 = at(pts.length - 1), [lo, hi] = EctViewer.range(y), f = this.chartFrame(a, x0, x1, lo, hi);
      if (this.s.boxes) {                                                     // where the cut crosses a crack box
        for (const b of p.boxes) {
          const [c0, c1, b0, b1] = X ? [b.y0, b.y1, b.x0, b.x1] : [b.x0, b.x1, b.y0, b.y1];
          if (i < c0 || i >= c1) continue;
          const xa = Math.max(a.L, f.X(b0 * p.pitch)), xb = Math.min(a.L + a.pw, f.X(b1 * p.pitch));
          ctx.fillStyle = 'rgba(184,106,61,.12)'; ctx.fillRect(xa, a.T, xb - xa, a.ph);
          if (b.n !== null) { ctx.fillStyle = COPPER; ctx.font = `10.5px ${FONT}`; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText('#' + b.n, (xa + xb) / 2, a.T + 2); }
        }
      }
      this.chartAxes(ctx, a, f, x0, x1, X ? 'X (mm)' : 'Y (mm)', p.label);
      EctViewer.stroke(ctx, f, at, y);
      this.profChart = { f, a, pts, at };
      const hv = this.hover;
      if (hv && hv.panel === p.index && (X ? hv.row : hv.col) === i) {
        const k = X ? hv.col - p.ox : hv.row - p.oy;
        if (y[k] === y[k]) this.dot(ctx, f.X(at(k)), f.Y(y[k]), a);
      }
    }

    onProfileMove(e) {
      const p = this.soloMap(), k = this.profChart;
      if (!p || !k) return;
      const x = k.f.ix(e.offsetX), first = k.at(0), idx = clamp(Math.round((x - first) / p.pitch), 0, k.pts.length - 1);
      const [col, row] = k.pts[idx];
      this.setHover({ panel: p.index, col, row }, e.clientX, e.clientY);
    }

    dot(ctx, x, y, a) {
      ctx.strokeStyle = 'rgba(27,31,36,.35)'; ctx.lineWidth = 1;
      if (a) { ctx.beginPath(); ctx.moveTo(Math.round(x) + 0.5, a.T); ctx.lineTo(Math.round(x) + 0.5, a.T + a.ph); ctx.stroke(); }
      ctx.fillStyle = COPPER; ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 4.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    }

    // ---------------------------------------------------------------- overlay: crosshair, pins, cut

    paintOverlay(c) {
      const a = c.area;
      if (!a) return;
      const ctx = this.sizeCanvas(c.over, a.w, a.h), p = c.p, hv = this.hover;
      if (p.type === 'line') {
        if (hv && hv.panel === p.index && hv.k !== undefined) this.dot(ctx, c.chart.X(hv.k), c.chart.Y(p.y[hv.k]), a);
        return;
      }
      const d3 = this.s.mode === '3d' && p.type === 'map';
      if (!d3 && this.soloMap() === p && this.s.profile.on) this.cut2d(ctx, c);
      this.s.pins.forEach((pin, i) => { if (this.shares(pin.panel, p)) this.marker(ctx, c, pin, String(i + 1), d3); });
      if (!hv || hv.k !== undefined || !this.shares(hv.panel, p)) return;
      const v = p.type === 'map' ? valueAt(p.grid, hv.col, hv.row) : 0;
      const inside = hv.col >= p.ox && hv.row >= p.oy && hv.col < p.ox + p.w && hv.row < p.oy + p.h;
      if (d3) { if (Number.isFinite(v)) this.cross3d(ctx, c, hv, v); } else if (inside) this.cross2d(ctx, c, hv);
      if (p.type === 'map' && Number.isFinite(v) && c.bar) {                  // where the value sits on the colour bar
        const y = c.bar.Yv(v), x = c.bar.x;
        ctx.strokeStyle = INK; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x - 4, y); ctx.lineTo(x + c.bar.w + 3, y); ctx.stroke();
      }
    }

    shares(panelIndex, p) {
      if (panelIndex === p.index) return true;
      const q = this.fig.panels[panelIndex];
      return !!(q && q.link && q.link === p.link);
    }

    cross2d(ctx, c, hv) {
      const fr = c.fr, k = fr.clip, p = c.p, x = fr.X((hv.col + 0.5) * p.pitch), y = fr.Y((hv.row + 0.5) * p.pitch);
      if (x < k.x0 || x > k.x1 || y < k.y0 || y > k.y1) return;
      const cell = p.pitch * fr.s, gap = Math.max(5, cell / 2);
      const path = () => {
        ctx.beginPath();
        ctx.moveTo(k.x0, y); ctx.lineTo(x - gap, y); ctx.moveTo(x + gap, y); ctx.lineTo(k.x1, y);
        ctx.moveTo(x, k.y0); ctx.lineTo(x, y - gap); ctx.moveTo(x, y + gap); ctx.lineTo(x, k.y1);
        if (cell >= 5) ctx.rect(x - cell / 2, y - cell / 2, cell, cell);
        else { ctx.moveTo(x + 4, y); ctx.arc(x, y, 4, 0, 2 * Math.PI); }
        ctx.stroke();
      };
      ctx.strokeStyle = 'rgba(0,0,0,.45)'; ctx.lineWidth = 3; path();
      ctx.strokeStyle = 'rgba(255,255,255,.95)'; ctx.lineWidth = 1; path();
      this.axisTag(ctx, mm((hv.col + 0.5) * p.pitch), x, k.y1 + 4, 'center');
      this.axisTag(ctx, mm((hv.row + 0.5) * p.pitch), k.x0 - 4, y, 'right');
    }
    axisTag(ctx, text, x, y, align) {
      ctx.font = `600 10.5px ${FONT}`;
      const w = ctx.measureText(text).width + 8, h = 15, x0 = align === 'center' ? x - w / 2 : x - w, y0 = align === 'center' ? y : y - h / 2;
      ctx.fillStyle = INK; ctx.beginPath(); ctx.roundRect(x0, y0, w, h, 3); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(text, x0 + w / 2, y0 + h / 2 + 0.5);
    }
    cross3d(ctx, c, hv, v) {
      const sf = c.sf, p = c.p, P = c.proj, x = (hv.col + 0.5) * p.pitch, y = (hv.row + 0.5) * p.pitch;
      const f = this.frame3d(c), top = this.pos3d(c, hv.col, hv.row, v), foot = P(sf.mx(x), sf.my(y), f.z0);
      const ex = P(sf.mx(x), f.edgeY, f.z0), ey = P(f.edgeX, sf.my(y), f.z0);
      ctx.strokeStyle = 'rgba(27,31,36,.65)'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(top[0], top[1]); ctx.lineTo(foot[0], foot[1]);
      ctx.moveTo(foot[0], foot[1]); ctx.lineTo(ex[0], ex[1]); ctx.moveTo(foot[0], foot[1]); ctx.lineTo(ey[0], ey[1]); ctx.stroke();
      ctx.setLineDash([]);
      this.axisTag(ctx, 'X ' + mm(x), ex[0], ex[1] + 3, 'center');
      this.axisTag(ctx, 'Y ' + mm(y), ey[0], ey[1] + 3, 'center');
      ctx.fillStyle = COPPER; ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(top[0], top[1], 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    }
    marker(ctx, c, pin, text, d3) {
      const p = c.p;
      let x, y;
      if (d3) {
        const v = valueAt(p.grid, pin.col, pin.row);
        if (!Number.isFinite(v)) return;
        const q = this.pos3d(c, pin.col, pin.row, v);
        ctx.globalAlpha = c.sf.visible(q[0] * c.q, q[1] * c.q, q[2]) ? 1 : 0.35;
        [x, y] = q;
      } else {
        const k = c.fr.clip;
        x = c.fr.X((pin.col + 0.5) * p.pitch); y = c.fr.Y((pin.row + 0.5) * p.pitch);
        if (x < k.x0 || x > k.x1 || y < k.y0 || y > k.y1) return;
      }
      ctx.fillStyle = COPPER; ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.font = `700 11px ${FONT}`; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      halo(ctx, text, x + 6, y - 3, '#fff', 'rgba(27,31,36,.9)', 3.5);
      ctx.globalAlpha = 1;
    }
    cut2d(ctx, c) {
      const fr = c.fr, k = fr.clip, p = c.p, X = this.s.profile.along === 'X', at = (this.cutIndex(p) + 0.5) * p.pitch;
      ctx.beginPath();
      if (X) { ctx.moveTo(k.x0, fr.Y(at)); ctx.lineTo(k.x1, fr.Y(at)); } else { ctx.moveTo(fr.X(at), k.y0); ctx.lineTo(fr.X(at), k.y1); }
      ctx.strokeStyle = 'rgba(0,0,0,.4)'; ctx.lineWidth = 4; ctx.stroke();
      ctx.strokeStyle = '#22d3ee'; ctx.lineWidth = 1.8; ctx.stroke();
    }
    nearCut(c, x, y) {
      if (this.s.mode === '3d' || this.soloMap() !== c.p || !this.s.profile.on) return false;
      const at = (this.cutIndex(c.p) + 0.5) * c.p.pitch;
      return Math.abs(this.s.profile.along === 'X' ? y - c.fr.Y(at) : x - c.fr.X(at)) < 7;
    }

    // ---------------------------------------------------------------- pointer and keyboard

    bind(c) {
      const o = c.over;
      o.addEventListener('pointerdown', (e) => {
        o.focus({ preventScroll: true });
        o.classList.remove('keys');
        if (c.p.type === 'line') return;
        const d3 = this.s.mode === '3d' && c.p.type === 'map';
        o.setPointerCapture(e.pointerId);
        this.drag = { c, x: e.offsetX, y: e.offsetY, x0: e.offsetX, y0: e.offsetY, moved: false,
                      kind: d3 ? (e.shiftKey || e.button === 2 || e.button === 1 ? 'shift' : 'rotate')
                               : this.nearCut(c, e.offsetX, e.offsetY) ? 'cut' : 'pan' };
        e.preventDefault();
      });
      o.addEventListener('pointermove', (e) => {
        const d = this.drag;
        if (d && d.c === c) {
          if (Math.hypot(e.offsetX - d.x0, e.offsetY - d.y0) > 4) d.moved = true;
          if (d.moved) { this.dragMove(d, e.offsetX - d.x, e.offsetY - d.y, e); this.setHover(null); }
          d.x = e.offsetX; d.y = e.offsetY;
          if (d.moved) return;
        }
        o.style.cursor = this.nearCut(c, e.offsetX, e.offsetY) ? (this.s.profile.along === 'X' ? 'ns-resize' : 'ew-resize') : '';
        this.setHover(this.pointAt(c, e.offsetX, e.offsetY), e.clientX, e.clientY);
      });
      const up = (e) => {
        const d = this.drag;
        if (!d || d.c !== c) return;
        this.drag = null;
        if (!d.moved && e.type === 'pointerup' && e.button === 0) { this.hover = this.pointAt(c, e.offsetX, e.offsetY); this.pin(); }
      };
      o.addEventListener('pointerup', up);
      o.addEventListener('pointercancel', up);
      o.addEventListener('pointerleave', () => { if (!this.drag) this.setHover(null); });
      o.addEventListener('contextmenu', (e) => e.preventDefault());
      o.addEventListener('dblclick', (e) => { e.preventDefault(); });
      o.addEventListener('wheel', (e) => { e.preventDefault(); this.wheel(c, e); }, { passive: false });
      o.addEventListener('keydown', (e) => { o.classList.add('keys'); this.key(c, e); });
      o.addEventListener('keyup', (e) => { if (e.key === 'Tab') o.classList.add('keys'); });
      o.addEventListener('blur', () => { o.classList.remove('keys'); if (this.hover && this.hover.keyboard) this.setHover(null); });
    }

    dragMove(d, dx, dy, e) {
      const s = this.s, c = d.c;
      if (d.kind === 'rotate') { s.cam.az = ((s.cam.az + dx * 0.45 + 540) % 360) - 180; s.cam.el = clamp(s.cam.el + dy * 0.45, -90, 90); this.redraw(true); }
      else if (d.kind === 'shift') { s.cam.px += dx; s.cam.py += dy; this.redraw(true); }
      else if (d.kind === 'cut') this.setCut(null, s.profile.along === 'X' ? c.fr.iy(e.offsetY) : c.fr.ix(e.offsetX));
      else { const v = this.viewOf(c.p); v.cx -= dx / c.fr.s; v.cy -= dy / c.fr.s; this.redraw(); }
    }

    wheel(c, e) {
      if (c.p.type === 'line') return;
      const k = Math.exp(-clamp(e.deltaY, -200, 200) * 0.0016);
      if (this.s.mode === '3d' && c.p.type === 'map') { this.s.cam.zoom = clamp(this.s.cam.zoom * k, 0.4, 12); this.redraw(true); return; }
      this.zoomAt(c, e.offsetX, e.offsetY, k);
    }
    zoomAt(c, x, y, k) {
      const v = this.viewOf(c.p), X = c.fr.ix(x), Y = c.fr.iy(y), z = clamp(v.z * k, 1, 40), r = v.z / z;
      v.cx = X - (X - v.cx) * r; v.cy = Y - (Y - v.cy) * r; v.z = z;           // the point under the mouse stays put
      this.redraw();
      this.setHover(null);
    }

    key(c, e) {
      const p = c.p, move = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
      if (p.type === 'line') return;
      if (move && e.shiftKey && this.s.mode === '3d') {
        this.s.cam.az += move[0] * 6; this.s.cam.el = clamp(this.s.cam.el + move[1] * 6, -90, 90); this.redraw(true);
      } else if (move) {
        const n = e.shiftKey ? 10 : 1, h = this.hover && this.hover.panel === p.index ? this.hover
          : { col: p.ox + (p.w >> 1), row: p.oy + (p.h >> 1) };
        const hv = { panel: p.index, col: clamp(h.col + move[0] * n, p.ox, p.ox + p.w - 1), row: clamp(h.row + move[1] * n, p.oy, p.oy + p.h - 1), keyboard: true };
        const box = c.over.getBoundingClientRect(), v = p.type === 'map' ? valueAt(p.grid, hv.col, hv.row) : 0;
        const q = this.s.mode === '3d' && p.type === 'map' ? this.pos3d(c, hv.col, hv.row, v) : [c.fr.X((hv.col + 0.5) * p.pitch), c.fr.Y((hv.row + 0.5) * p.pitch)];
        this.setHover(hv, box.left + q[0], box.top + q[1]);
      } else if (e.key === 'Enter' || e.key === ' ') this.pin();
      else if (e.key === 'Escape') this.setHover(null);
      else if (e.key === '+' || e.key === '=' || e.key === '-') {
        const k = e.key === '-' ? 0.8 : 1.25;
        if (this.s.mode === '3d') { this.s.cam.zoom = clamp(this.s.cam.zoom * k, 0.4, 12); this.redraw(true); }
        else this.zoomAt(c, c.area.L + c.area.pw / 2, c.area.T + c.area.ph / 2, k);
      } else if (e.key === '0') this.resetView();
      else return;
      e.preventDefault();
    }

    /** The scan point under a pixel of a panel, or null. */
    pointAt(c, x, y) {
      const p = c.p;
      if (!c.area) return null;
      if (p.type === 'line') {
        const a = c.area;
        if (x < a.L - 4 || x > a.L + a.pw + 4 || y < a.T || y > a.T + a.ph) return null;
        return { panel: p.index, k: clamp(Math.round(c.chart.ix(x)), 0, p.n - 1) };
      }
      if (this.s.mode === '3d' && p.type === 'map') {
        const hit = c.sf && c.sf.pick(x * c.q, y * c.q);
        return hit && { panel: p.index, ...hit };
      }
      const k = c.fr && c.fr.clip;
      if (!k || x < k.x0 || x > k.x1 || y < k.y0 || y > k.y1) return null;
      return { panel: p.index, col: clamp(Math.floor(c.fr.ix(x) / p.pitch), p.ox, p.ox + p.w - 1),
               row: clamp(Math.floor(c.fr.iy(y) / p.pitch), p.oy, p.oy + p.h - 1) };
    }

    // ---------------------------------------------------------------- readout

    /** Every number known at a scan point: the map's own value first, then raw / filtered / z or the linked maps. */
    samples(h) {
      const p = this.fig.panels[h.panel], out = [];
      if (p.type === 'line') return [{ name: p.ylabel || 'value', value: p.y[h.k], span: p.span }];
      if (p.type === 'rgb') {
        const q = 3 * ((h.row - p.oy) * p.w + h.col - p.ox);
        return p.channels.map((name, i) => ({ name, value: p.rgb[q + i], span: 1000 }));
      }
      out.push({ name: this.fig.panels.length > 1 ? p.name : p.label || p.name, value: valueAt(p.grid, h.col, h.row), span: p.hi - p.lo });
      for (const r of p.readouts) out.push({ name: r.name, value: valueAt(r.grid, h.col, h.row), span: r.grid.span });
      for (const q of p.peers) if (q !== p && q.type === 'map') out.push({ name: q.name, value: valueAt(q.grid, h.col, h.row), span: q.hi - q.lo });
      return out;
    }
    boxAt(h) {
      const p = this.fig.panels[h.panel];
      if (!this.s.boxes || !p.boxes.length) return null;
      const hit = p.boxes.filter((b) => h.col >= b.x0 && h.col < b.x1 && h.row >= b.y0 && h.row < b.y1);
      hit.sort((a, b) => (a.x1 - a.x0) * (a.y1 - a.y0) - (b.x1 - b.x0) * (b.y1 - b.y0));
      return hit[0] || null;
    }
    boxText(p, b) {
      const name = b.kind === 'missed' ? 'Missed label' : b.conf === undefined && b.score === undefined ? `Box #${b.n}` : `Crack #${b.n}`;
      const score = b.conf !== undefined ? `, confidence ${b.conf.toFixed(2)}` : b.score !== undefined ? `, |z| ${b.score.toFixed(1)}` : '';
      const at = `centre (${mm((b.x0 + b.x1) / 2 * p.pitch)}, ${mm((b.y0 + b.y1) / 2 * p.pitch)}) mm, box ${mm((b.x1 - b.x0) * p.pitch)} × ${mm((b.y1 - b.y0) * p.pitch)} mm`;
      return [name + score, at, b.kind === 'found' ? 'matches a label' : b.kind === 'false' ? 'no matching label' : ''];
    }

    setHover(h, cx, cy) {
      const same = (a, b) => a === b || (a && b && a.panel === b.panel && a.col === b.col && a.row === b.row && a.k === b.k);
      const changed = !same(h, this.hover);
      this.hover = h;
      if (!h) { this.tip.hidden = true; if (changed) this.refresh(); return; }
      if (changed) { this.fillTip(h); this.refresh(); }
      this.placeTip(cx, cy);
    }

    fillTip(h) {
      const p = this.fig.panels[h.panel], t = this.tip;
      t.textContent = '';
      const [main, ...rest] = this.samples(h);
      const top = el('div', 'vw-tip-main');
      top.append(el('b', '', fmt(main.value, main.span)), el('span', '', main.name));
      t.append(top);
      if (p.type === 'line') { t.append(el('div', 'vw-tip-row', `sample ${h.k}`)); return; }
      const pos = el('div', 'vw-tip-pos');
      pos.append(el('b', '', `X ${mm((h.col + 0.5) * p.pitch)}`), el('b', '', `Y ${mm((h.row + 0.5) * p.pitch)}`), el('span', '', `mm · col ${h.col}, row ${h.row}`));
      t.append(pos);
      if (rest.length) {
        const g = el('div', 'vw-tip-grid');
        for (const r of rest) g.append(el('b', '', fmt(r.value, r.span)), el('span', '', r.name));
        t.append(g);
      }
      const b = this.boxAt(h);
      if (b) {
        const bx = el('div', 'vw-tip-box'), [a1, a2, a3] = this.boxText(p, b);
        bx.style.borderColor = this.fig.box_colors[b.kind];
        bx.append(el('b', '', a1), el('span', '', a2));
        if (a3) bx.append(el('span', '', a3));
        t.append(bx);
      }
    }
    placeTip(cx, cy) {
      const t = this.tip;
      t.hidden = false;
      const w = t.offsetWidth, h = t.offsetHeight, x = cx + 18 + w > window.innerWidth - 8 ? cx - 18 - w : cx + 18;
      t.style.left = Math.max(8, x) + 'px';
      t.style.top = clamp(cy + 18 + h > window.innerHeight - 8 ? cy - 18 - h : cy + 18, 8, Math.max(8, window.innerHeight - h - 8)) + 'px';
    }

    // ---------------------------------------------------------------- pins

    pin() {
      const h = this.hover;
      if (!h || h.k !== undefined) return;
      const i = this.s.pins.findIndex((q) => q.col === h.col && q.row === h.row && q.panel === h.panel);
      if (i >= 0) this.s.pins.splice(i, 1);                                   // click a pinned point again: unpin
      else {
        if (this.s.pins.length >= MAX_PINS) this.s.pins.shift();
        this.s.pins.push({ panel: h.panel, col: h.col, row: h.row });
      }
      this.renderPins();
      this.refresh();
    }

    pinRows() {
      let prev = null;
      return this.s.pins.map((pin, i) => {
        const p = this.fig.panels[pin.panel], x = (pin.col + 0.5) * p.pitch, y = (pin.row + 0.5) * p.pitch, b = this.boxAt(pin);
        const row = { n: i + 1, map: p.name, x, y, col: pin.col, row: pin.row, samples: this.samples(pin),
                      crack: b && b.n !== null ? '#' + b.n : b ? 'missed' : '', dist: prev && this.shares(prev.panel, p) ? Math.hypot(x - prev.x, y - prev.y) : null };
        prev = { x, y, panel: pin.panel };
        return row;
      });
    }

    renderPins() {
      const rows = this.pinRows(), box = this.pinBox;
      box.textContent = '';
      box.hidden = !rows.length;
      if (!rows.length) return;
      const multi = new Set(this.s.pins.map((q) => q.panel)).size > 1, cracks = rows.some((r) => r.crack), dist = rows.some((r) => r.dist !== null);
      const head = el('div', 'vw-pins-head');
      head.append(el('span', '', `${rows.length} pinned point${rows.length > 1 ? 's' : ''}`),
                  button('Copy', 'Copy the table (paste it into Excel)', () => this.copyPins()),
                  button('Clear', 'Remove every pin', () => { this.s.pins = []; this.renderPins(); this.refresh(); }));
      const table = el('table'), tr = el('tr');
      for (const t of ['#', 'X (mm)', 'Y (mm)', 'Col, row', ...(multi ? ['Map'] : []), 'Values', ...(cracks ? ['Crack'] : []), ...(dist ? ['To previous (mm)'] : []), '']) tr.append(el('th', '', t));
      table.append(tr);
      rows.forEach((r, i) => {
        const line = el('tr');
        const cell = (t, cls) => { const d = el('td', cls || '', t); line.append(d); return d; };
        cell(String(r.n), 'n'); cell(mm(r.x), 'num'); cell(mm(r.y), 'num'); cell(`${r.col}, ${r.row}`, 'num');
        if (multi) cell(r.map);
        const vals = cell('', 'vals');
        r.samples.forEach((q) => { const s = el('span', 'vw-val'); s.append(el('b', '', fmt(q.value, q.span)), el('span', '', ' ' + q.name)); vals.append(s); });
        if (cracks) cell(r.crack);
        if (dist) cell(r.dist === null ? '' : mm(r.dist), 'num');
        const x = button('×', 'Remove this pin', () => { this.s.pins.splice(i, 1); this.renderPins(); this.refresh(); });
        cell('').append(x);
        table.append(line);
      });
      box.append(head, table);
      box.scrollTop = box.scrollHeight;
    }

    pinsText() {
      const rows = this.pinRows(), names = [...new Set(rows.flatMap((r) => r.samples.map((q) => q.name)))];
      const out = [['#', 'X (mm)', 'Y (mm)', 'col', 'row', 'map', ...names, 'crack', 'to previous (mm)'].join('\t')];
      for (const r of rows) {
        const by = Object.fromEntries(r.samples.map((q) => [q.name, q.value]));
        out.push([r.n, r.x.toFixed(2), r.y.toFixed(2), r.col, r.row, r.map, ...names.map((n) => (Number.isFinite(by[n]) ? Number(by[n].toPrecision(7)) : '')),
                  r.crack, r.dist === null ? '' : r.dist.toFixed(2)].join('\t'));
      }
      return out.join('\n');
    }
    async copyPins() {
      const text = this.pinsText(), label = this.pinBox.querySelector('.vw-pins-head .vw-btn');
      try { await navigator.clipboard.writeText(text); } catch (e) {          // no clipboard permission in this window
        const t = el('textarea'); t.value = text; document.body.append(t); t.select(); document.execCommand('copy'); t.remove();
      }
      if (label) { label.textContent = 'Copied'; setTimeout(() => { label.textContent = 'Copy'; }, 1200); }
    }

    // ---------------------------------------------------------------- saving the view

    /** The maps exactly as shown (turned, zoomed, with pins) as a PNG data: URL. null in Image mode. */
    snapshot() {
      if (!this.active) return null;
      const dpr = window.devicePixelRatio || 1, g = this.grid.getBoundingClientRect(), prof = this.prof.hidden ? 0 : this.profCanvas.clientHeight + 8;
      const rows = this.pinRows(), head = this.fig.subtitle ? 26 : 8, tail = rows.length ? 14 + rows.length * 17 : 0;
      const cv = el('canvas');
      cv.width = Math.round(g.width * dpr); cv.height = Math.round((head + g.height + prof + tail) * dpr);
      const ctx = cv.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, g.width, cv.height);
      ctx.fillStyle = MUTED; ctx.font = `11px ${FONT}`; ctx.textBaseline = 'middle';
      ctx.fillText(fitText(ctx, this.fig.subtitle || '', g.width - 16), 8, 13);
      const hover = this.hover;
      this.hover = null;                                                      // no crosshair in the saved picture
      for (const c of this.cells) {
        this.paintOverlay(c);
        const r = c.box.getBoundingClientRect();
        for (const l of [c.base, c.over]) ctx.drawImage(l, r.left - g.left, head + r.top - g.top, r.width, r.height);
      }
      if (prof) { this.paintProfile(); ctx.drawImage(this.profCanvas, 0, head + g.height + 4, this.profCanvas.clientWidth, this.profCanvas.clientHeight); }
      this.hover = hover;
      ctx.fillStyle = INK; ctx.font = `11px ${FONT}`; ctx.textAlign = 'left';
      rows.forEach((r, i) => ctx.fillText(fitText(ctx, `${r.n}.  X ${mm(r.x)} mm, Y ${mm(r.y)} mm:  ${r.samples.map((q) => `${q.name} ${fmt(q.value, q.span)}`).join(',  ')}${r.crack ? ',  crack ' + r.crack : ''}`, g.width - 16),
                                          8, head + g.height + prof + 14 + i * 17));
      this.refresh();
      return cv.toDataURL('image/png');
    }
  }

  const api = { EctViewer, Surface, paintMap, projector, basis, ticks, tickText, fmt, valueAt, decodeGrid, decodeLut };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.EctViewer = EctViewer;
})(typeof window !== 'undefined' ? window : globalThis);
