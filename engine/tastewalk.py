# Taste walk (Nov 2026): one line that visits points around the subject, drawn by the trained walk and chosen by the
# Taste judge (Rowan's good/bad labels), instead of SLDgen's diffusion model.
#
#   waypoints  points spread over the subject's lines and depth outlines (curve.py's target, relative subject), more
#              where lines are heavy (faces, outlines); `n` sets how many
#   order      one short tour through them (greedy + 2-opt), opened at its longest hop
#   each hop   the walk bank (walkjoin.py: real pieces of the trained Endless Line walk, rotated/scaled, never bent)
#              offers candidates for A -> B; each is drawn into the judge's window (the last stretch of line, older ink
#              lighter, exactly as raster.js draws it for the judge) and scored:
#                 taste (judge.onnx, DINOv2-small + Rowan's good/bad fit)  - fid x distance from the subject's lines
#                 - busy x ink already under it
#              the best one is kept
#   output     the pipeline's path files, so stills and videos work as usual
#
# Usage: python3 tastewalk.py <photo> <outdir> [n=160] [cands=10] [fid=1.0] [busy=0.5] [win=0.7] [seed=0]
import sys, os, json, time, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import curve as CV, walkjoin as WJ, linephoto as LP

T = dict(n=160, cands=10, fid=1.0, busy=0.5, win=0.7, seed=0, judge=os.path.join(HERE, 'models', 'judge.onnx'))
T0 = time.time()
def log(*a): print(f'[tastewalk {time.time() - T0:6.1f}s]', *a, flush=True)

_SESS = {}
def judge(imgs):
    # imgs: (B, 224, 224) ink 0..1 -> taste score per picture (higher = more like the lines Rowan keeps)
    import onnxruntime as ort
    if 's' not in _SESS: _SESS['s'] = ort.InferenceSession(T['judge'], providers=['CPUExecutionProvider'])
    return _SESS['s'].run(None, {'ink': imgs[:, None].astype(np.float32)})[0]

def raster(P, S=224):
    # port of taste-data distill/raster.js: window fitted into an S x S square (86 %), older ink lighter, width from pressure
    img = np.zeros((S, S), np.float32)
    if len(P) < 2: return img
    a, b, c, d = P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()
    s = S * 0.86 / max(b - a, d - c, 1e-6); mx, my = (a + b) / 2, (c + d) / 2; t0, t1 = P[0, 3], P[-1, 3]; h = S / 2; wsc = S / 224
    xy = np.c_[h + (P[:, 0] - mx) * s, h + (P[:, 1] - my) * s]
    for k in range(1, len(P)):                                   # in time order: later (darker) ink lands on top
        ink = 1 - (200 * (1 - (0.3 + 0.7 * (P[k, 3] - t0) / ((t1 - t0) or 1)))) / 255
        hw = max(0.5, 3.2 * (0.25 + P[k, 2]) * wsc / 2)
        lay = np.zeros_like(img)
        cv2.line(lay, tuple(np.round(xy[k - 1] * 16).astype(int)), tuple(np.round(xy[k] * 16).astype(int)), float(ink), max(1, int(round(2 * hw))), cv2.LINE_AA, 4)
        np.maximum(img, lay, out=img)
    return img

def main(src, outdir):
    rng = np.random.default_rng(int(T['seed']))
    pts, w, (W, H), full = CV.target_points(src)
    wmap = np.zeros((H, W), np.uint8); ix = np.clip(np.round(pts).astype(int), 0, [W - 1, H - 1]); wmap[ix[:, 1], ix[:, 0]] = 1
    dist = cv2.distanceTransform(1 - cv2.dilate(wmap, np.ones((3, 3), np.uint8)), cv2.DIST_L2, 3) / W   # distance to the subject's lines
    way = CV.tour(CV.farthest(pts, w, int(T['n']), rng)); log(f'{len(way)} waypoints in one tour')
    # the line so far: x, y, pressure, t (t = arc length, the judge's "time")
    path = [np.array([[way[0][0], way[0][1], 0.5, 0.0]])]; ink = np.zeros((H, W), np.uint8); L = 0.0; win = T['win'] * W
    din = np.arctan2(*(way[1] - way[0])[::-1])
    for i in range(len(way) - 1):
        A, B = way[i].astype(float), way[i + 1].astype(float)
        dout = np.arctan2(*((way[i + 2] - B) if i + 2 < len(way) else (B - A))[::-1])
        cands = []
        for _ in range(int(T['cands'])):
            j = WJ.join(A, B, din, dout, (W, H), None, rng, top=12)
            if j is not None and len(j[0]) > 1: cands.append(j)
        if not cands:                                                   # no walk piece fits: straight hop
            q = np.linspace(A, B, max(2, int(np.hypot(*(B - A)) / 2))); cands = [(q, np.full(len(q), 0.4))]
        prev = np.concatenate(path); keep = prev[prev[:, 3] >= L - win]
        imgs, extra = [], []
        for q, pr in cands:
            seg = np.hypot(*np.diff(q, axis=0).T); t = L + np.r_[0, np.cumsum(seg)]
            Q = np.c_[q, np.clip(pr[:len(q)] if len(pr) >= len(q) else np.full(len(q), 0.4), 0, 1), t]
            Wn = np.concatenate([keep, Q[1:]]); Wn = Wn[Wn[:, 3] >= Wn[-1, 3] - win][::2]
            imgs.append(raster(Wn))
            qi = np.clip(np.round(q).astype(int), 0, [W - 1, H - 1])
            extra.append((T['fid'] * float(dist[qi[:, 1], qi[:, 0]].mean()) * 20 + T['busy'] * float(ink[qi[:, 1], qi[:, 0]].mean()), Q))
        taste = judge(np.stack(imgs))
        score = taste - np.array([e for e, _ in extra]); b = int(np.argmax(score)); Q = extra[b][1]
        path.append(Q[1:]); L = Q[-1, 3]
        cv2.polylines(ink, [np.round(Q[:, :2]).astype(np.int32)], False, 1, 5)
        din = np.arctan2(*(Q[-1, :2] - Q[-2, :2])[::-1])
        if i % 20 == 0: log(f'hop {i}/{len(way) - 1}: taste {taste[b]:.2f} (best of {len(cands)}, mean {taste.mean():.2f})')
    P = np.concatenate(path)
    CV.write(P[:, :2], (W, H), outdir, src)

if __name__ == '__main__':
    src, outdir = sys.argv[1], sys.argv[2]
    for kv in sys.argv[3:]:
        k, v = kv.split('=', 1)
        if k in T: T[k] = type(T[k])(float(v)) if not isinstance(T[k], str) else v
        elif k in CV.C: CV.C[k] = type(CV.C[k])(float(v))
    main(src, outdir)
