# "Explore" style: the frozen hydrant technique (hydrant_style.py, the video that did well), with two additions:
#   joins go AROUND dense areas: for every hop several routes are tried (walk pieces of different widths, curves
#   bowed left/right by different amounts) and the one crossing the least existing ink wins, length counting only a
#   little, so the line prefers a wide arc to somewhere else over cutting through a dense patch
#   line weight from the photo's darkness (widths_d*.npy + meta_d*.json, the files the video reads)
# Stroke order: route.py (route optimiser, on by default).
# Usage: python3 explore.py <photo> <outdir> [key=value ...]   (hydrant_style keys, plus E keys below)
import sys, os, math, json, glob, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import linephoto as LP, walkjoin as WJ
import hydrant_style as HS
import route as RT                                       # route-optimised stroke order (route=0 turns it off)
import depth as DP                                       # depth-aware selection + weight (depth=1 turns it on)

E = dict(avoid=1, bows='0.25,0.5,0.8', walk_tries=5, len_cost=0.15, dens_sigma=10.0, margin=0.03)
_DENS = {}

def density(ink):
    key = id(ink)
    if key not in _DENS:
        d = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), E['dens_sigma']); _DENS.clear(); _DENS[key] = d / (d.max() + 1e-6)
    return _DENS[key]

def bowed(a, b, din, dout, bow):
    # cubic curve that swings out sideways by bow x distance (sign = side), leaving/arriving along the stroke directions
    d = float(np.hypot(*(b - a))); u = (b - a) / max(d, 1e-6); n = np.array([-u[1], u[0]])
    c1 = a + 0.35 * d * np.array([math.cos(din), math.sin(din)]) + n * bow * d
    c2 = b - 0.35 * d * np.array([math.cos(dout), math.sin(dout)]) + n * bow * d
    t = np.linspace(0, 1, max(8, int(d * (1 + abs(bow)) / 1.5)))[:, None]
    return (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * b

# memoise the line model: the frozen style calls LP._contour directly (uncached), so every option on the same photo
# re-ran the same ~11 s inference. Same input -> saved result (disk cache keyed by the exact pixels and size).
_raw_contour = LP._contour
LP._contour = lambda img: LP._cache('idcontour', img, _raw_contour)

_orig_travel = HS.travel
def travel_around(a, b, din, dout, W, H, rng, ink=None):
    if not E['avoid'] or ink is None: return _orig_travel(a, b, din, dout, W, H, rng, ink)
    d = float(np.hypot(*(b - a)))
    if d < HS.A['hop'] * W: return _orig_travel(a, b, din, dout, W, H, rng, ink)
    dens = density(ink); mg = E['margin'] * min(W, H)
    cands = [_orig_travel(a, b, din, dout, W, H, rng, ink)[0]]
    for lo, hi in ((1.05, 1.5), (1.3, 2.0), (1.6, 2.6)):
        for _ in range(E['walk_tries']):
            wj = WJ.join(a, b, din, dout, (W, H), None, rng, top=10, ratio=(lo, hi))
            if wj is not None: cands.append(wj[0])
    for bw in [float(x) for x in E['bows'].split(',')]:
        cands += [bowed(a, b, din, dout, bw), bowed(a, b, din, dout, -bw)]
    best, bc = None, 1e18
    for c in cands:
        if c is None or len(c) < 2: continue
        if (c[:, 0] < mg).any() or (c[:, 0] > W - mg).any() or (c[:, 1] < mg).any() or (c[:, 1] > H - mg).any(): continue
        ix = np.clip(np.round(c).astype(int), 0, [W - 1, H - 1])
        L = float(np.hypot(*np.diff(c, axis=0).T).sum())
        cost = float(dens[ix[:, 1], ix[:, 0]].mean()) + E['len_cost'] * (L / max(d, 1.0) - 1)
        if cost < bc: best, bc = c, cost
    return (best, None) if best is not None else _orig_travel(a, b, din, dout, W, H, rng, ink)
HS.travel = travel_around

# Full silhouette (outline=1): the outer edge of the subject mask is always drawn as complete strokes, so shapes like
# a helmet get a closed contour even where the line model breaks their edge into short pieces (which then fall under
# the minimum length). Trace strokes lying along that silhouette are dropped so nothing is drawn twice. The silhouette
# counts toward the line budget, so the total amount of line stays the same.
E.update(outline=0, ol_flow=0.5, ol_near=7.0, ol_min=0.01)
_OL = {}
def silhouette(W, H, level):
    key = (W, H, level)
    if key in _OL: return _OL[key]
    m = (cv2.resize(LP.SRC['matte'], (W, H), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    tot = m.sum(); out = []; e = 4
    for c in cs:
        if cv2.contourArea(c) < E['ol_min'] * tot: continue
        c = c[:, 0, :].astype(np.float64); c = np.vstack([c, c[:1]])
        on = (c[:, 0] > e) & (c[:, 0] < W - 1 - e) & (c[:, 1] > e) & (c[:, 1] < H - 1 - e)   # photo-edge parts are not outline
        if on.all(): segs = [c]
        else:
            k = int(np.argmin(on)); c = np.roll(c[:-1], -k, axis=0); on = np.roll(on[:-1], -k)   # start on an edge point
            segs, cur = [], []
            for p, o in zip(c, on):
                if o: cur.append(p)
                elif cur: segs.append(np.array(cur)); cur = []
            if cur: segs.append(np.array(cur))
        for s in segs:
            if len(s) > 30: out.append(HS.flow(s, E['ol_flow'] * HS.A['flow'] * level))
    ink = np.zeros((H, W), np.uint8)
    for s in out: cv2.polylines(ink, [np.round(s).astype(np.int32)], False, 1, 1)
    dist = cv2.distanceTransform(1 - ink, cv2.DIST_L2, 3)
    _OL[key] = (out, dist); return _OL[key]

_sel_prev = HS.select
def select_outline(strokes, level, U, W, H, inside, face, labels, lscale=1.0):
    if not E['outline']: return _sel_prev(strokes, level, U, W, H, inside, face, labels, lscale)
    ol, dist = silhouette(W, H, level)
    def along(s):
        ix = np.clip(np.round(s).astype(int), 0, [W - 1, H - 1]); return (dist[ix[:, 1], ix[:, 0]] < E['ol_near']).mean() > 0.6
    rest = [s for s in strokes if not along(s)]
    keep = _sel_prev(rest, level, U, W, H, inside, face, labels, lscale)
    obj = keep[0][1] if keep else 0
    return [(s, obj) for s in ol] + keep
HS.select = select_outline

# ---------- depth (depth=1): depth-edge strokes kept shorter, flat-surface strokes need to be longer ----------
_sel_before_depth = HS.select
def select_depth(strokes, level, U, W, H, inside, face, labels, lscale=1.0):
    if not DP.D['depth']: return _sel_before_depth(strokes, level, U, W, H, inside, face, labels, lscale)
    RT.CTX['W'], RT.CTX['H'] = W, H
    if 'edges' not in _DEPTH or _DEPTH['edges'].shape != (H, W):
        d = cv2.resize(_DEPTH['d'], (W, H), interpolation=cv2.INTER_AREA); _DEPTH['edges'] = DP.depth_edges(d)
    ed = _DEPTH['edges']; A = HS.A
    Lmin = A['Lmin'] * U * level * lscale; Lface = A['face_frac'] * A['Lmin'] * U * level; f = U / W; keep = []
    for s_ in strokes:
        ix = np.clip(np.round(s_).astype(int), 0, [W - 1, H - 1])
        if inside[ix[:, 1], ix[:, 0]].mean() < A['keep_in']: continue
        L = LP.arclen(s_); mid = ix[len(ix) // 2]; inface = bool(face[mid[1], mid[0]])
        cov = float(ed[ix[:, 1], ix[:, 0]].mean())                 # share of the stroke lying on a depth edge
        need = Lmin * (DP.D['depth_keep'] if cov > 0.5 else DP.D['depth_flat'] if cov < 0.05 else 1.0)
        if L >= need or (inface and L >= Lface):
            obj = int(labels[mid[1], mid[0]]) if labels is not None else 0
            fl = (A['flow_face'] if inface else A['flow']) * (DP.D['depth_flow'] if cov > 0.5 else 1.0)
            keep.append((HS.flow(s_, fl * level * f), obj))
    return keep
HS.select = select_depth
_DEPTH = {}

def depth_weight(photo, outdir):
    W = 900; d = cv2.resize(_DEPTH['d'], (W, int(_DEPTH['d'].shape[0] * W / _DEPTH['d'].shape[1])), interpolation=cv2.INTER_AREA)
    for f in sorted(glob.glob(os.path.join(outdir, 'widths_d*.npy'))):
        dd = f[-5]; P = np.load(os.path.join(outdir, f'path_d{dd}.npy')) * W / 500.0
        ix = np.clip(np.round(P).astype(int), 0, [d.shape[1] - 1, d.shape[0] - 1]); near = d[ix[:, 1], ix[:, 0]]
        near = cv2.GaussianBlur(near.reshape(-1, 1).astype(np.float32), (1, 0), 6).ravel()
        r = np.load(f) * (1 - DP.D['depth_w'] / 2 + DP.D['depth_w'] * near)
        np.save(f, r.astype(np.float32))

def darkness_widths(photo, outdir):
    W = 900; ph = cv2.imread(photo); H = int(ph.shape[0] * W / ph.shape[1])
    g = cv2.cvtColor(cv2.resize(ph, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    dark = cv2.GaussianBlur(1 - cv2.createCLAHE(2.0, (8, 8)).apply(g).astype(np.float32) / 255, (0, 0), 4)
    for f in sorted(glob.glob(os.path.join(outdir, 'path_d*.npy'))):
        dd = f[-5]; P = np.load(f) * W / 500.0; kinds = json.load(open(os.path.join(outdir, f'kinds_d{dd}.json')))
        r = LP.widths(P, kinds); ix = np.clip(np.round(P).astype(int), 0, [W - 1, H - 1]); dk = dark[ix[:, 1], ix[:, 0]]
        lo, hi = np.percentile(dk, 5), np.percentile(dk, 95); dk = np.clip((dk - lo) / (hi - lo + 1e-6), 0, 1)
        f_ = 0.35 + 1.9 * dk ** 1.3; wk = np.array(kinds) == 'walk'; f_[wk] = np.clip(f_[wk], 0.55, 0.95)
        r = (r * cv2.GaussianBlur(f_.reshape(-1, 1).astype(np.float32), (1, 0), 8).ravel()).astype(np.float32)
        np.save(os.path.join(outdir, f'widths_d{dd}.npy'), r)
        json.dump(dict(nib_scale=LP.P['nib_scale'], px_per_unit=W / 500.0), open(os.path.join(outdir, f'meta_d{dd}.json'), 'w'))

if __name__ == '__main__':
    rest = []
    for kv in sys.argv[3:]:
        k, v = kv.split('=', 1)
        if k in E: E[k] = v if isinstance(E[k], str) else type(E[k])(float(v))
        elif k in DP.D: DP.D[k] = float(v)
        elif k in RT.R: RT.R[k] = float(v)
        else: rest.append(kv)
    sys.argv = sys.argv[:3] + rest
    if DP.D['depth']: _DEPTH['d'] = DP.depth_map(cv2.imread(sys.argv[1]))
    HS.main(); darkness_widths(sys.argv[1], sys.argv[2])
    if DP.D['depth']: depth_weight(sys.argv[1], sys.argv[2])
