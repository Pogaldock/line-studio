# One-curve method (Nov 2026): the idea of SLDgen (Magne et al., CGF 2026: one smooth curve initialised from a
# point tour, then optimised) without its GPU image model. Here the curve is judged against what we can compute on a
# CPU: the line model's lines on the subject and the depth map's outlines.
#
#   target    points on the subject's lines (line model, relative subject) + depth edges; faces/hands weigh more
#   start     K points spread along the target by weight (farthest-point sampling), joined into one short tour
#             (greedy + 2-opt), the longest hop removed -> open curve
#   curve     cubic B-spline with K control points (K = how much detail it can afford)
#   optimise  Adam on the control points, ~600 steps, minimising
#               cover  * how far each target point is from the curve (weighted): everything important gets drawn
#               fit    * how far the curve is from the target: it stays on real lines
#               length * total length: no wandering
#               repel  * near-touching of distant parts: no scribbling over itself
#               smooth * bending energy: confident, smooth line
#   output    path_d1 / kinds / secs / lazy / widths / meta in the pipeline's format, so stills and videos work as usual
#
# Usage: python3 curve.py <photo> <outdir> [K=260] [steps=600] [cover=2] [fit=1] [length=0.6] [repel=4] [smooth=0.3]
#        [depth=1] [scene=0.25]
import sys, os, json, time, math, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import explore as EX                                    # relative subject (close-ups), depth model hookup
import hydrant_style as HS, linephoto as LP, depth as DP

C = dict(K=260, steps=600, cover=2.0, fit=1.0, length=0.6, repel=4.0, smooth=0.3, depth=1, scene=0.25, face=2.0,
         width=900, thr=0.35, n_target=4000, seed=0)

def log(*a): print(f'[curve {time.time() - T0:6.1f}s]', *a, flush=True)
T0 = time.time()

def target_points(src):
    # the lines worth drawing: same line model and subject handling as the main pipeline, plus depth edges
    full = cv2.imread(src)
    if LP.P['upscale'] and max(full.shape[:2]) < LP.P['upscale_below']: full = LP.upscale(full)
    W = int(C['width']); LP.SRC['full'] = full
    img = cv2.resize(full, (W, int(full.shape[0] * W / full.shape[1])), interpolation=cv2.INTER_AREA); H = img.shape[0]; LP.SRC['img'] = img
    fh = LP._cache('facehand2', img, LP._focus_map); face = cv2.dilate((fh[0] > 0).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    prepped = LP.prep_for_lines(full); LP.SRC['full'] = prepped
    m = cv2.resize(LP.SRC['matte'], (W, H), interpolation=cv2.INTER_AREA) > 0.5
    Hf, Wf = full.shape[:2]; k = W / Wf; x0, y0, x1, y1 = LP.SRC.get('subj_box', (0, 0, Wf, Hf))
    LP.SRC['full'] = prepped[y0:y1, x0:x1]
    bx0, by0, bx1, by1 = int(x0 * k), int(y0 * k), max(int(x0 * k) + 2, int(x1 * k)), max(int(y0 * k) + 2, int(y1 * k))
    sub = np.zeros((H, W), np.float32); sub[by0:by1, bx0:bx1] = LP._contour(np.zeros((by1 - by0, bx1 - bx0, 3), np.uint8))
    LP.SRC['full'] = full; scene = LP._contour(img)
    inside = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    from skimage.morphology import skeletonize
    lines = skeletonize((sub > C['thr']) & inside)
    sc = skeletonize((scene > C['thr']) & ~inside) if C['scene'] > 0 else np.zeros_like(lines)
    wmap = np.zeros((H, W), np.float32); wmap[lines] = 1.0; wmap[sc] = C['scene']
    if C['depth']:
        d = DP.depth_map(full); d = cv2.resize(d, (W, H), interpolation=cv2.INTER_AREA)
        de = skeletonize(DP.depth_edges(d) & inside); wmap[de] = np.maximum(wmap[de], 1.5)
    wmap[face & (wmap > 0)] *= C['face']
    ys, xs = np.nonzero(wmap); w = wmap[ys, xs]; pts = np.stack([xs, ys], 1).astype(np.float32)
    rng = np.random.default_rng(int(C['seed']))
    if len(pts) > C['n_target']:                        # keep a weighted subsample so each step stays fast
        idx = rng.choice(len(pts), int(C['n_target']), replace=False, p=w / w.sum()); pts, w = pts[idx], w[idx]
    log(f'target: {len(pts)} points (subject {100 * m.mean():.0f}% of the frame)')
    return pts, w, (W, H), full

def farthest(pts, w, K, rng):
    # spread K points along the target, favouring heavy points (weighted farthest-point sampling)
    i = [int(rng.choice(len(pts), p=w / w.sum()))]; dmin = np.full(len(pts), np.inf)
    for _ in range(K - 1):
        dmin = np.minimum(dmin, np.hypot(*(pts - pts[i[-1]]).T)); i.append(int(np.argmax(dmin * np.sqrt(w))))
    return pts[i]

def tour(P):
    # greedy nearest-neighbour tour, then 2-opt until nothing improves; open it at the longest hop
    n = len(P); D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    seq = [0]; left = set(range(1, n))
    while left:
        j = min(left, key=lambda q: D[seq[-1], q]); seq.append(j); left.discard(j)
    seq = np.array(seq); improved = True
    while improved:
        improved = False
        for a in range(1, n - 2):
            A, B = seq[a - 1], seq[a]; Cc = seq[a + 1:]; Dd = np.r_[seq[a + 2:], seq[0]]
            gain = D[A, B] + D[Cc, Dd] - D[A, Cc] - D[B, Dd]
            b = int(np.argmax(gain))
            if gain[b] > 1e-6: seq[a:a + b + 2] = seq[a:a + b + 2][::-1]; improved = True
    hop = np.hypot(*(P[seq] - P[np.roll(seq, -1)]).T); cut = int(np.argmax(hop))
    return P[np.roll(seq, -(cut + 1))]

def bspline_basis(K, S):
    # clamped uniform cubic B-spline: S samples x K control points
    deg = 3; n_kn = K + deg + 1; t = np.r_[np.zeros(deg), np.linspace(0, 1, n_kn - 2 * deg), np.ones(deg)]
    u = np.linspace(0, 1 - 1e-9, S); B = np.zeros((S, K))
    N = np.array([[1.0 if t[i] <= x < t[i + 1] else 0.0 for i in range(n_kn - 1)] for x in u])
    for p in range(1, deg + 1):
        Nn = np.zeros((S, n_kn - 1 - p))
        for i in range(n_kn - 1 - p):
            a = (u - t[i]) / (t[i + p] - t[i]) if t[i + p] > t[i] else 0
            b = (t[i + p + 1] - u) / (t[i + p + 1] - t[i + 1]) if t[i + p + 1] > t[i + 1] else 0
            Nn[:, i] = a * N[:, i] + b * N[:, i + 1]
        N = Nn
    return N

def optimise(ctrl, pts, w, size):
    import torch
    torch.manual_seed(int(C['seed'])); W, H = size
    K = len(ctrl); S = K * 10
    Bm = torch.tensor(bspline_basis(K, S), dtype=torch.float32)
    Cp = torch.tensor(ctrl, dtype=torch.float32, requires_grad=True)
    T = torch.tensor(pts); Wt = torch.tensor(w / w.mean(), dtype=torch.float32)
    opt = torch.optim.Adam([Cp], lr=1.5); scale = float(W)
    ii = torch.arange(S); far = (ii[:, None] - ii[None, :]).abs() > 25          # sample pairs far apart along the curve
    for step in range(int(C['steps'])):
        Sp = Bm @ Cp
        Dm = torch.cdist(Sp, T)                                                     # S x N
        cover = (Dm.min(0).values * Wt).mean() / scale * 100
        fit = Dm.min(1).values.mean() / scale * 100
        seg = (Sp[1:] - Sp[:-1]).norm(dim=1); length = seg.sum() / scale
        sub = Sp[::4]; Dd = torch.cdist(sub, sub); fr = far[::4, ::4]
        repel = (torch.relu(6.0 - Dd)[fr] ** 2).mean()
        smooth = ((Sp[2:] - 2 * Sp[1:-1] + Sp[:-2]) ** 2).sum(1).mean()
        loss = C['cover'] * cover + C['fit'] * fit + C['length'] * length + C['repel'] * repel + C['smooth'] * smooth
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): Cp[:, 0].clamp_(2, W - 3); Cp[:, 1].clamp_(2, H - 3)
        if step % 100 == 0 or step == C['steps'] - 1:
            log(f'step {step}: cover {cover.item():.2f} fit {fit.item():.2f} length {length.item():.1f} repel {repel.item():.3f} smooth {smooth.item():.2f}')
    with torch.no_grad(): return (Bm @ Cp).numpy()

def write(path_px, size, outdir, src):
    W, H = size; os.makedirs(outdir, exist_ok=True)
    # resample to ~1 px steps (the format the renderer expects)
    seg = np.hypot(*np.diff(path_px, axis=0).T); s = np.r_[0, np.cumsum(seg)]; n = max(2, int(s[-1]))
    q = np.stack([np.interp(np.linspace(0, s[-1], n), s, path_px[:, i]) for i in (0, 1)], 1)
    P = q * 500.0 / W; kinds = ['L1'] * len(P)
    np.save(os.path.join(outdir, 'path_d1.npy'), P.astype(np.float32)); json.dump(kinds, open(os.path.join(outdir, 'kinds_d1.json'), 'w'))
    np.save(os.path.join(outdir, 'secs_d1.npy'), np.zeros(len(P), np.int32)); np.save(os.path.join(outdir, 'lazy_d1.npy'), np.zeros(len(P), np.float32))
    np.save(os.path.join(outdir, 'widths_d1.npy'), LP.widths(P * W / 500.0, kinds).astype(np.float32))
    json.dump(dict(nib_scale=LP.P['nib_scale'], px_per_unit=W / 500.0), open(os.path.join(outdir, 'meta_d1.json'), 'w'))
    EX.darkness_widths(src, outdir)                       # same line weight from the photo's darkness as the main style
    log(f'wrote {len(P)} points, {s[-1] / W:.1f} frame-widths of line')

if __name__ == '__main__':
    src, outdir = sys.argv[1], sys.argv[2]
    for kv in sys.argv[3:]:
        k, v = kv.split('=', 1); C[k] = type(C[k])(float(v))
    rng = np.random.default_rng(int(C['seed']))
    pts, w, size, full = target_points(src)
    ctrl = tour(farthest(pts, w, int(C['K']), rng)); log(f'start tour through {len(ctrl)} points')
    path = optimise(ctrl, pts, w, size)
    write(path, size, outdir, src)
