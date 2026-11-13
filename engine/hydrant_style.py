# FROZEN: the exact tracing technique of the hydrant video that did well on TikTok (13 Nov 2026). Do not edit; copy instead.
# Abstract free-flowing single line (contour-first method), full pipeline up to the files video.py reads.
# photo -> background removed (subject matte) -> contour-style lines -> long chained strokes inside the subject
# -> smoothed flowing curves -> grouped object by object -> joined with walk pieces -> eye signature -> broad nib.
# Usage: python3 abstract.py <photo> <outdir> [key=value ...]
# Writes line_d{1,2,3}.png (d1 most abstract), path_dN.npy (500-unit space), kinds_dN.json, lazy_dN.npy, secs_dN.npy
import sys, os, math, json, numpy as np, cv2
from skimage.morphology import skeletonize, remove_small_objects
import linephoto as LP, walkjoin as WJ, eye as EY

A = dict(width=900, thr=0.35, keep_in=0.6, matte_pad=0.02, Lmin=0.035, face_frac=0.3, flow=6.0, flow_face=3.0,
         levels='1.8,1.5,1.2', sign=1, seed=0,
         target=12.5, hop=0.03, arc_max=0.4, eye_margin=0.04,
         subj_ref=0.8, scene=1, scene_L=0.12, scene_share=0.25)

def log(*a): print('[abstract]', *a, flush=True)

def strokes_from_contour(a, thr):
    m = remove_small_objects(a > thr, max_size=20)
    segs, _ = LP.segments(skeletonize(m))
    return [p.astype(float) for _, _, p in segs if len(p) >= 3]

def chain_strokes(strokes, gap_max=6):
    # join strokes end to end where the ends meet and directions agree, so long contours stay one stroke.
    # Same greedy rule as before, but only ends within gap_max are compared (spatial index), not every pair.
    from scipy.spatial import cKDTree
    S = [s for s in strokes]; alive = [True] * len(S)
    def ends_of(i):
        s = S[i]; return [(s[0], i, 0), (s[-1], i, 1)]
    changed = True
    while changed:
        changed = False
        E = [e for i in range(len(S)) if alive[i] for e in ends_of(i)]
        if len(E) < 2: break
        tree = cKDTree(np.array([e[0] for e in E])); used = set()
        for k, (p, i, side) in enumerate(E):
            if i in used or not alive[i]: continue
            A_ = S[i] if side == 1 else S[i][::-1]                         # A_ ends at this end point
            da = A_[-1] - A_[max(0, len(A_) - 6)]; best = None
            for k2 in tree.query_ball_point(p, gap_max):
                q, j, side2 = E[k2]
                if j == i or j in used or not alive[j]: continue
                B_ = S[j] if side2 == 0 else S[j][::-1]                    # B_ starts at that end point
                db = B_[min(len(B_) - 1, 5)] - B_[0]
                c = np.dot(da, db) / (np.hypot(*da) * np.hypot(*db) + 1e-6)
                gap = float(np.hypot(*(B_[0] - A_[-1])))
                if c > 0.5 and (best is None or gap < best[0]): best = (gap, j, B_)
            if best:
                _, j, B_ = best; S[i] = np.vstack([A_, B_]); alive[j] = False
                used.update((i, j)); changed = True
    return [S[i] for i in range(len(S)) if alive[i]]

def flow(p, sigma):
    if len(p) < 6: return p
    q = p.astype(np.float32).copy()
    for c in (0, 1): q[:, c] = cv2.GaussianBlur(p[:, c].reshape(-1, 1).astype(np.float32), (1, 0), sigma, borderType=cv2.BORDER_REPLICATE).ravel()
    return q.astype(float)

def soft_join(a, b, din, dout):
    d = float(np.hypot(*(b - a))); k = 0.4 * d
    c1 = a + k * np.array([math.cos(din), math.sin(din)]); c2 = b - k * np.array([math.cos(dout), math.sin(dout)])
    t = np.linspace(0, 1, max(6, int(d / 2)))[:, None]
    return (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * b

def head(p, end):
    q = p[-1] - p[max(0, len(p) - 8)] if end else p[min(len(p) - 1, 7)] - p[0]
    return math.atan2(q[1], q[0])

def select(strokes, level, U, W, H, inside, face, labels, lscale=1.0):
    # long strokes on the subject (min length raised by lscale on busy photos), shorter ones in faces at the base
    # threshold; smoothed more as the level rises. U = the subject's size in pixels, so small figures get the same
    # treatment as big portraits.
    Lmin = A['Lmin'] * U * level * lscale; Lface = A['face_frac'] * A['Lmin'] * U * level; f = U / W; keep = []
    for s in strokes:
        ix = np.clip(np.round(s).astype(int), 0, [W - 1, H - 1])
        if inside[ix[:, 1], ix[:, 0]].mean() < A['keep_in']: continue          # mostly background: drop
        L = LP.arclen(s); mid = ix[len(ix) // 2]
        inface = bool(face[mid[1], mid[0]])
        if L >= Lmin or (inface and L >= Lface):
            obj = int(labels[mid[1], mid[0]]) if labels is not None else 0
            keep.append((flow(s, (A['flow_face'] if inface else A['flow']) * level * f), obj))
    return keep

def order(keep):
    # finish one object before the next: inside an object always take the nearest stroke end,
    # then move to the object whose nearest stroke end is closest
    groups = {}
    for s, o in keep: groups.setdefault(o, []).append(s)
    cur = np.array([0.0, 0.0]); out = []; sec = 0
    def nearest(rem):
        c = [(np.hypot(*(s[0] - cur)), i, False) for i, s in enumerate(rem)] + [(np.hypot(*(s[-1] - cur)), i, True) for i, s in enumerate(rem)]
        return min(c, key=lambda t: t[0])
    while groups:
        o = min(groups, key=lambda g: nearest(groups[g])[0]); rem = groups.pop(o)
        while rem:
            _, i, rev = nearest(rem); s = rem.pop(i); s = s[::-1] if rev else s
            out.append((s, sec)); cur = s[-1]
        sec += 1
    return out

def travel(a, b, din, dout, W, H, rng, ink=None):
    # walk pieces for every hop: swooping arcs (path no more than ~1.8x the distance, so no curls), steering around
    # existing ink, kept within arc_max of the straight path; tiny hops are a plain soft curve
    d = float(np.hypot(*(b - a)))
    if d >= A['hop'] * W:
        ratio = lambda dist: (1.06, 1.8) if dist < 0.3 * W else (1.03, 1.5)
        u = (b - a) / d; n = np.array([-u[1], u[0]])
        for _ in range(4):
            wj = WJ.join(a, b, din, dout, (W, H), ink, rng, top=8, ratio=ratio)
            if wj is None: break
            j = wj[0]
            if np.abs((j - a) @ n).max() <= A['arc_max'] * d: return j, 'walk'
    return soft_join(a, b, din, dout), 'walk'

def drawn_len(seq, U): return sum(LP.arclen(s) for s, _ in seq) / U

def scene_lines(strokes, inside, U, W, H, level, budget):
    # the scene behind the subject, kept sparse: only its longest lines (horizon, hydrant, bowl rim, shadow),
    # longest first, up to a share of the drawing; texture (graffiti, wall grain) is short and never makes it
    cand = []
    for s in strokes:
        ix = np.clip(np.round(s).astype(int), 0, [W - 1, H - 1])
        if inside[ix[:, 1], ix[:, 0]].mean() > 0.3: continue
        L = LP.arclen(s)
        if L >= A['scene_L'] * W: cand.append((L, s))
    out, tot = [], 0.0
    for L, s in sorted(cand, key=lambda t: -t[0]):
        if tot + L > budget * U: break
        out.append((flow(s, A['flow'] * level * 1.5), -1)); tot += L
    return out

def assemble(seq, W, H, rng):
    ink = np.zeros((H, W), np.uint8)                       # where the drawing's own strokes are (joins avoid crossing them)
    for st, _ in seq: cv2.polylines(ink, [np.round(st).astype(np.int32)], False, 1, 3)
    ink = ink > 0
    pts, kinds, secs = [], [], []
    for k, (s, sec) in enumerate(seq):
        if pts:
            a = pts[-1][-1]; din = head(pts[-1], True); dout = head(s, False)
            j, _ = travel(a, s[0], din, dout, W, H, rng, ink)
            pts.append(j); kinds += ['walk'] * len(j); secs += [sec] * len(j)
        pts.append(s); kinds += ['L1'] * len(s); secs += [sec] * len(s)
    return pts, kinds, secs

def main():
    src, outdir = sys.argv[1], sys.argv[2]
    for kv in sys.argv[3:]:
        k, v = kv.split('=', 1)
        if k in A: A[k] = v if isinstance(A[k], str) else type(A[k])(float(v))
        else: LP.P[k] = type(LP.P[k])(float(v)) if not isinstance(LP.P[k], str) else v
    os.makedirs(outdir, exist_ok=True)
    full = cv2.imread(src)
    if LP.P['upscale'] and max(full.shape[:2]) < LP.P['upscale_below']: full = LP.upscale(full)
    LP.SRC['full'] = full
    Wd = A['width']; img = cv2.resize(full, (Wd, int(full.shape[0] * Wd / full.shape[1])), interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]; LP.SRC['img'] = img
    labels = LP.segment_objects(img)
    fh = LP._cache('facehand2', img, LP._focus_map); face = fh[0] > 0
    LP.SRC['full'] = LP.prep_for_lines(full)                       # background removed for the line model
    m = cv2.resize(LP.SRC['matte'], (W, H), interpolation=cv2.INTER_AREA) > 0.5
    pad = max(3, int(A['matte_pad'] * W))
    inside = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))) > 0
    prepped = LP.SRC['full']; Hf, Wf = full.shape[:2]; k = W / Wf
    x0, y0, x1, y1 = LP.SRC.get('subj_box', (0, 0, Wf, Hf))
    U = float(np.clip(max(x1 - x0, y1 - y0) * k / A['subj_ref'], 0.3 * W, W))       # subject size in drawing pixels
    # subject: the line model runs on a crop around the subject, so a small figure gets the model's full resolution
    LP.SRC['full'] = prepped[y0:y1, x0:x1]
    bx0, by0, bx1, by1 = int(x0 * k), int(y0 * k), max(int(x0 * k) + 2, int(x1 * k)), max(int(y0 * k) + 2, int(y1 * k))
    sub = np.zeros((H, W), np.float32)
    sub[by0:by1, bx0:bx1] = LP._contour(np.zeros((by1 - by0, bx1 - bx0, 3), np.uint8))
    strokes = chain_strokes(strokes_from_contour(sub, A['thr']))
    LP.SRC['full'] = full; scene = chain_strokes(strokes_from_contour(LP._contour(img), A['thr'])) if A['scene'] else []
    LP.SRC['full'] = prepped
    log(len(strokes), 'subject strokes,', len(scene), 'scene strokes; subject covers', round(100 * m.mean()), '% of the photo, size', round(U / W, 2), 'of the width')
    persons = cv2.resize(LP.person_mask(img).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    av = (m | persons | ((labels > 0) if labels is not None else False)).astype(np.uint8)
    e = max(3, int(A['eye_margin'] * W)); avoid = cv2.dilate(av, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1))) > 0
    for d, level in enumerate([float(x) for x in A['levels'].split(',')], 1):
        rng = np.random.default_rng(A['seed'] + d)
        target = A['target'] * (level / 1.8) ** -0.5 * float(np.clip(math.sqrt(m.mean() / 0.5), 0.8, 1.2) if U >= 0.95 * W else 1.0)
        lo, hi, ls = 0.4, 4.0, 1.0; sel = select(strokes, level, U, W, H, inside, face, labels, 1.0)
        if drawn_len(sel, U) > target: lo = 1.0
        else: hi = 1.0
        for _ in range(8):
            ls = (lo + hi) / 2; sel = select(strokes, level, U, W, H, inside, face, labels, ls)
            if drawn_len(sel, U) > target: lo = ls
            else: hi = ls
        ls = hi; sel = select(strokes, level, U, W, H, inside, face, labels, ls)
        if A['scene']: sel += scene_lines(scene, inside, U, W, H, level, A['scene_share'] * target)
        log(f'  level {level}: target {target:.1f} widths, min stroke length x{ls:.2f}')
        seq = order(sel)
        pts, kinds, secs = assemble(seq, W, H, rng)
        if A['sign']:                                              # the eye mark in free space, off the subject
            sig = EY.eye_at(W, H, pts, 'A', avoid=avoid)
            q = pts[-1]; din = head(q, True); dout = math.atan2(*(sig[3] - sig[0])[::-1])
            j, _ = travel(q[-1], sig[0], din, dout, W, H, rng)
            s_ = secs[-1] + 1
            pts += [j, sig]; kinds += ['walk'] * len(j) + ['sig'] * len(sig); secs += [s_] * (len(j) + len(sig))
        P_ = np.vstack(pts)
        LP.render(P_, kinds, H, W, f'{outdir}/line_d{d}.png', scale=1.5)
        np.save(f'{outdir}/path_d{d}.npy', P_ * (500.0 / W)); json.dump(kinds, open(f'{outdir}/kinds_d{d}.json', 'w'))
        np.save(f'{outdir}/lazy_d{d}.npy', np.zeros(len(P_), np.float32)); np.save(f'{outdir}/secs_d{d}.npy', np.array(secs, np.int32))
        drawn = sum(LP.arclen(s) for s, _ in seq) / U
        log(f'detail {d} (level {level}): {len(seq)} strokes in {secs[-1] + 1} sections, {drawn:.1f} subject sizes of line')

if __name__ == '__main__': main()
