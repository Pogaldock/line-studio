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

E = dict(avoid=1, bows='0.25,0.5,0.8', walk_tries=5, len_cost=0.15, dens_sigma=10.0, margin=0.03, quiet_joins=0.0, merge=0.0, merge_turn=70.0, wobble=0.0, wobble_face=1, wobble_long=0.0, avoid_face=0.0, fuse=0.0, fuse_flow=1.0)
_DENS = {}
_FACE = {}                                    # face map of the current photo (set by select, read by travel)

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

# ---------- relative subject: background removal assumes a subject in front of a background. On close-ups (a face
# filling the frame) it fails and marks almost nothing, so the face was treated as background: blurred before the line
# model saw it, and only given sparse scene lines. The person detector is checked against it: when the matte covers
# far less than the person (or almost nothing at all), the person is the subject, and when that fills most of the
# frame the whole frame is. A normal matte is left exactly as it was (the locked look is unchanged).
E.update(rel_subject=1)
def relative_matte(full, m):
    if not E['rel_subject']: return m
    frac = float((m > 0.5).mean())
    small = cv2.resize(full, (900, max(1, int(full.shape[0] * 900 / full.shape[1]))), interpolation=cv2.INTER_AREA)
    keep_img = LP.SRC.get('img'); LP.SRC['img'] = small
    try: pb = cv2.resize(LP.person_mask(small).astype(np.uint8), (full.shape[1], full.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    finally:
        if keep_img is None: LP.SRC.pop('img', None)
        else: LP.SRC['img'] = keep_img
    pf = float(pb.mean())
    if frac >= 0.03 and not (pf > 0.5 and frac < 0.35 * pf): return m
    if pf > 0.6 or (frac < 0.03 and pf == 0):
        # the subject fills the frame: split it from the room behind by depth, at the level that separates this photo
        # best (Otsu on the depth map); only if that gives no sensible split, the whole frame
        d = DP.depth_map(full); d8 = (cv2.GaussianBlur(d, (0, 0), 3) * 255).astype(np.uint8)
        t, near = cv2.threshold(d8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        n_, lab_, st_, _ = cv2.connectedComponentsWithStats(near)
        if n_ > 1:
            big = 1 + int(np.argmax(st_[1:, cv2.CC_STAT_AREA])); near = (lab_ == big).astype(np.uint8)
            near = cv2.morphologyEx(near, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        cover = float(near.mean())
        if 0.2 <= cover <= 0.95: out = cv2.GaussianBlur(near.astype(np.float32), (0, 0), 2); why = f'near part by depth ({100 * cover:.0f}% of the frame)'
        else: out = np.ones_like(m); why = 'whole frame'
    else: out = np.maximum(m, pb.astype(np.float32)); why = 'person area'
    print(f'[relative subject] background removal kept {100 * frac:.0f}% of the photo, person covers {100 * pf:.0f}%: subject = {why}', flush=True)
    return out
LP.MATTE_FIX = relative_matte

_orig_travel = HS.travel
def travel_around(a, b, din, dout, W, H, rng, ink=None):
    if not E['avoid'] or ink is None: return _orig_travel(a, b, din, dout, W, H, rng, ink)
    d = float(np.hypot(*(b - a)))
    if d < HS.A['hop'] * W and not E['avoid_face']: return _orig_travel(a, b, din, dout, W, H, rng, ink)
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
        if E['avoid_face'] and 'm' in _FACE and _FACE['m'].shape == (H, W):     # joins pay for every point on the face
            cost += E['avoid_face'] * float(_FACE['m'][ix[:, 1], ix[:, 0]].mean())
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
        if DP.D['depth_relief']:                     # relative relief counts as outline too, and adds its own strokes
            rl = DP.relief_lines(d); _DEPTH['edges'] = _DEPTH['edges'] | (cv2.dilate(rl.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
            have = np.zeros((H, W), np.uint8)
            for s0 in strokes: cv2.polylines(have, [np.round(s0).astype(np.int32)], False, 1, 9)
            rs = HS.chain_strokes(HS.strokes_from_contour(rl * (1 - have), 0.5))
            _DEPTH['relief_strokes'] = [r for r in rs if len(r) > 6]
            log_ = getattr(HS, 'log', print); log_(len(_DEPTH['relief_strokes']), 'relief strokes added from the depth map')
    if DP.D['depth_relief']: strokes = list(strokes) + _DEPTH.get('relief_strokes', [])
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
            fl = (A['flow_face'] if inface else A['flow']) * (DP.D['depth_flow'] if cov > DP.D['depth_shape_cov'] else 1.0)
            keep.append((HS.flow(s_, fl * level * f) if fl > 0 else s_.astype(float), obj))
    if DP.D['depth_gap']:                       # depth outlines nothing covers yet: added, lightly smoothed
        if 'gaps' not in _DEPTH or _DEPTH['gaps'][0] != (H, W):
            dc = DP.depth_curves(cv2.resize(_DEPTH['d'], (W, H), interpolation=cv2.INTER_AREA), DP.D['depth_gap_canny'])
            _DEPTH['gaps'] = ((H, W), HS.chain_strokes(HS.strokes_from_contour(dc, 0.5)))
        cover = np.zeros((H, W), np.uint8)
        for k_, _ in keep: cv2.polylines(cover, [np.round(k_).astype(np.int32)], False, 1, 11)
        gmin = DP.D['depth_gap_min'] * Lmin
        for c_ in _DEPTH['gaps'][1]:
            ix = np.clip(np.round(c_).astype(int), 0, [W - 1, H - 1])
            free = (cover[ix[:, 1], ix[:, 0]] == 0) & inside[ix[:, 1], ix[:, 0]]
            k0 = None
            for i_ in range(len(free) + 1):                     # runs of uncovered points become strokes
                if i_ < len(free) and free[i_]:
                    if k0 is None: k0 = i_
                elif k0 is not None:
                    run = c_[k0:i_]
                    if len(run) > 3 and LP.arclen(run) >= gmin:
                        mid = ix[(k0 + i_) // 2]; obj = int(labels[mid[1], mid[0]]) if labels is not None else 0
                        keep.append((HS.flow(run, max(DP.D['depth_flow'], 0.05) * A['flow'] * level * f), obj))
                        cv2.polylines(cover, [np.round(run).astype(np.int32)], False, 1, 11)
                    k0 = None
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

# ---------- merge (merge=<fraction of width>): kept strokes whose ends are within that distance and roughly continue
# each other become one stroke (short straight bridge), so there are fewer separate strokes and so fewer joins.
# Closest pairs first; a pair merges only if leaving one end and entering the other turns less than merge_turn degrees.
def _end_dir(s, at_end):
    q = s[-1] - s[max(0, len(s) - 8)] if at_end else s[0] - s[min(len(s) - 1, 7)]
    return q / (np.hypot(*q) + 1e-9)                      # direction pointing OUT of the stroke at that end
_sel_before_merge = HS.select
def select_merge(strokes, level, U, W, H, inside, face, labels, lscale=1.0):
    keep = _sel_before_merge(strokes, level, U, W, H, inside, face, labels, lscale)
    if not E['merge'] or len(keep) < 2: return keep
    S = [np.asarray(k_[0], float) for k_ in keep]; O = [k_[1] for k_ in keep]; alive = [True] * len(S)
    dmax = E['merge'] * W; cmin = math.cos(math.radians(E['merge_turn']))
    while True:
        best = None
        for i in range(len(S)):
            if not alive[i]: continue
            for j in range(i + 1, len(S)):
                if not alive[j]: continue
                for ei in (0, 1):
                    for ej in (0, 1):
                        a = S[i][-1] if ei else S[i][0]; b = S[j][-1] if ej else S[j][0]; v = b - a; d = float(np.hypot(*v))
                        if d > dmax: continue
                        u = v / (d + 1e-9)
                        if d > 2 and (_end_dir(S[i], ei) @ u < cmin or -_end_dir(S[j], ej) @ u < cmin): continue
                        if best is None or d < best[0]: best = (d, i, j, ei, ej)
        if best is None: break
        _, i, j, ei, ej = best
        A_ = S[i] if ei else S[i][::-1]; B_ = S[j] if not ej else S[j][::-1]          # A ends at the joint, B starts there
        n = max(2, int(np.hypot(*(B_[0] - A_[-1])) / 1.5))
        bridge = np.linspace(A_[-1], B_[0], n)[1:-1]
        S[i] = np.vstack([A_, bridge, B_]) if len(bridge) else np.vstack([A_, B_]); alive[j] = False
    out = [(S[i], O[i]) for i in range(len(S)) if alive[i]]
    return out
HS.select = select_merge

# ---------- wobble (wobble=<max turning, radians per 10 px>): scribbly strokes dropped. Turning is measured on the kept
# (already smoothed) stroke resampled every 3 px; a clean contour (a glasses rim, a jaw) turns ~0.1-0.3 rad per 10 px,
# beard and skin-texture pieces far more. wobble_face=1: only strokes in the face (face/hand map, or the whole subject
# when the photo is a close-up and no face box covers it); wobble_long=<fraction of width>: strokes at least that long
# always stay. The style's budget search still runs over the result.
def turning(s):
    s = np.asarray(s, float); L = LP.arclen(s)
    if L < 6: return 99.0
    seg = np.hypot(*np.diff(s, axis=0).T); c = np.r_[0, np.cumsum(seg)]; t = np.arange(0, c[-1], 3.0)
    q = np.c_[np.interp(t, c, s[:, 0]), np.interp(t, c, s[:, 1])]
    if len(q) < 4: return 99.0
    a = np.arctan2(*np.diff(q, axis=0).T[::-1]); da = np.abs((np.diff(a) + np.pi) % (2 * np.pi) - np.pi)
    return float(da.sum() / (L / 10.0))
# ---------- fuse (fuse=<band width as a fraction of the frame width>): the line model often traces one real contour (a
# glasses rim) as several overlapping, parallel or end-to-end fragments, each then drawn as its own stroke with joins
# between them. Stroke consolidation (as in StrokeAggregator, Liu et al. 2018, in a simple raster form): all kept
# strokes are painted as a band fuse x W wide, so fragments lying on the same curve melt into one shape; its centre
# line is traced, chained through junctions where directions agree, and smoothed (fuse_flow x the style's smoothing)
# into one flowing stroke. Lines further apart than the band (two lips, two rims) stay separate.
def fuse(keep, level, U, W, H, labels):
    if not keep: return keep
    w = max(3, int(round(E['fuse'] * W)))
    band = np.zeros((H, W), np.uint8)
    for k_ in keep: cv2.polylines(band, [np.round(k_[0]).astype(np.int32)], False, 1, w, cv2.LINE_8)
    n_, lab_, st_, _ = cv2.connectedComponentsWithStats((1 - band).astype(np.uint8), connectivity=4)
    small = np.nonzero(st_[:, cv2.CC_STAT_AREA] < (2 * w) ** 2)[0]      # pinholes inside the band -> no tiny rings
    band[np.isin(lab_, small[small > 0])] = 1
    tr = HS.chain_strokes(HS.strokes_from_contour(band.astype(np.float32), 0.5), gap_max=max(6, w))
    f = U / W; sig = E['fuse_flow'] * HS.A['flow'] * level * f; out = []
    for t in tr:
        if LP.arclen(t) < 2 * w: continue                       # knots left at junctions
        mid = np.clip(np.round(t[len(t) // 2]).astype(int), 0, [W - 1, H - 1])
        out.append((HS.flow(t, sig) if sig > 0 else t, int(labels[mid[1], mid[0]]) if labels is not None else 0))
    return out

_sel_before_wobble = HS.select
def select_wobble(strokes, level, U, W, H, inside, face, labels, lscale=1.0):
    _FACE['m'] = face.astype(np.float32) if face is not None else np.zeros((H, W), np.float32)
    keep = _sel_before_wobble(strokes, level, U, W, H, inside, face, labels, lscale)
    if E['fuse']: keep = fuse(keep, level, U, W, H, labels)
    if not E['wobble']: return keep
    out = []
    for k_ in keep:
        ix = np.clip(np.round(k_[0]).astype(int), 0, [W - 1, H - 1])
        onface = (not E['wobble_face']) or face is None or not face.any() or face[ix[:, 1], ix[:, 0]].mean() > 0.5
        long_ = E['wobble_long'] and LP.arclen(k_[0]) >= E['wobble_long'] * W     # long strokes (a glasses rim) always stay
        if onface and not long_ and turning(k_[0]) > E['wobble']: continue
        out.append(k_)
    return out
HS.select = select_wobble

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
        if E['quiet_joins'] and wk.any() and (~wk).any():
            # joins as a hairline thread: quiet_joins x the median line width, applied AFTER the width smoothing (which
            # otherwise hands short joins most of their neighbours' weight); a short taper where a join meets a line
            hair = E['quiet_joins'] * float(np.median(r[~wk]))
            # taper by drawn length (6 px), not by point count: joins are stored with sparse points
            cum = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
            dist = np.full(len(r), 1e9); last = -1e9
            for i in range(len(r)):
                if not wk[i]: last = cum[i]
                dist[i] = cum[i] - last
            last = 1e9
            for i in range(len(r) - 1, -1, -1):
                if not wk[i]: last = cum[i]
                dist[i] = min(dist[i], last - cum[i])
            t = np.clip(dist / 6.0, 0, 1)
            r = np.where(wk, r * (1 - t) + np.minimum(r, hair) * t, r).astype(np.float32)
        np.save(os.path.join(outdir, f'widths_d{dd}.npy'), r)
        meta = dict(nib_scale=LP.P['nib_scale'], px_per_unit=W / 500.0)
        if E['quiet_joins'] and wk.any() and (~wk).any(): meta['quiet'] = round(hair, 4)   # humanize keeps joins at this width
        json.dump(meta, open(os.path.join(outdir, f'meta_d{dd}.json'), 'w'))

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
