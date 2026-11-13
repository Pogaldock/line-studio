# Joins between objects drawn by the real Endless Line walk (line-lab distill/write/natural.js: the trained walk with the
# taste judge steering it). The walk is cut into many pieces; for each gap A->B the piece whose start and end directions
# continue the strokes on both sides, that stays in the picture and avoids busy ink, is rotated and scaled onto A->B.
# The walk's own shape is never bent, only chosen, so it keeps its energy. Choice is random among the best few.
import glob, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N = 48                                      # points per stored piece
_BANK = {}

def _resample(p, n):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p[:, :2], axis=0).T))]
    t = np.linspace(0, d[-1], n)
    return np.c_[np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1]), np.interp(t, d, p[:, 2])], d[-1]

def bank():
    if 'C' in _BANK: return _BANK
    srcs = sorted(glob.glob(os.path.join(HERE, 'models', 'walk_s*.json')))
    cf = os.path.join(HERE, 'models', 'cache', 'walkbank_%d_%d.npz' % (N, int(sum(os.path.getmtime(f) for f in srcs))))
    if os.path.exists(cf):                                                          # built once, then loaded
        z = np.load(cf); _BANK.update({k: z[k] for k in z.files}); return _BANK
    shapes, a0s, a1s, ratios, press, wigs = [], [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(HERE, 'models', 'walk_s*.json'))):
        w = np.array(json.load(open(f)), float)
        w = w[np.r_[True, np.hypot(*np.diff(w[:, :2], axis=0).T) > 1e-6]]          # drop standstill samples
        arc = np.r_[0, np.cumsum(np.hypot(*np.diff(w[:, :2], axis=0).T))]
        for L in (1.5, 2.5, 4, 6, 9):                                               # piece lengths in walk units
            step = max(1, int(np.searchsorted(arc, L * 0.25)))
            for i0 in range(0, len(w) - 2, step):
                i1 = int(np.searchsorted(arc, arc[i0] + L))
                if i1 >= len(w): break
                q, _ = _resample(w[i0:i1 + 1], N)
                ch = q[-1, :2] - q[0, :2]; cl = float(np.hypot(*ch))
                if cl < 1e-3: continue
                ang = math.atan2(ch[1], ch[0]); c, s = math.cos(-ang), math.sin(-ang)
                xy = (q[:, :2] - q[0, :2]) @ np.array([[c, s], [-s, c]]) / cl          # chord on +x, length 1
                xy = _smooth(xy)
                h = np.unwrap(np.arctan2(*np.diff(xy, axis=0)[:, ::-1].T)); wig = float(np.abs(np.diff(h, 2)).sum())
                t0 = xy[2] - xy[0]; t1 = xy[-1] - xy[-3]
                for mir in (1, -1):                                                 # and its mirror image
                    m = xy * [1, mir]
                    shapes.append(m); a0s.append(math.atan2(mir * t0[1], t0[0])); a1s.append(math.atan2(mir * t1[1], t1[0]))
                    ratios.append(L / cl); press.append(q[:, 2]); wigs.append(wig)
    _BANK.update(C=np.array(shapes, np.float32), a0=np.array(a0s), a1=np.array(a1s), ratio=np.array(ratios), p=np.array(press, np.float32), wig=np.array(wigs))
    os.makedirs(os.path.dirname(cf), exist_ok=True); np.savez(cf, **_BANK)
    return _BANK

def _smooth(xy, passes=3):
    # keep the large arc, remove fine wobble: repeated end-anchored smoothing over ~1/8 of the piece
    out = xy.copy(); k = np.array([1, 2, 3, 4, 3, 2, 1], float); k /= k.sum()
    for _ in range(passes):
        pad = np.vstack([np.repeat(out[:1], 3, 0), out, np.repeat(out[-1:], 3, 0)])
        sm = np.c_[np.convolve(pad[:, 0], k, 'valid'), np.convolve(pad[:, 1], k, 'valid')]
        w_ = np.minimum(np.arange(len(out)), np.arange(len(out))[::-1]) / 4.0; w_ = np.clip(w_, 0, 1)[:, None]
        out = out * (1 - w_) + sm * w_
    return out

def _wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi

def join(A, B, din, dout, size, ink=None, rng=None, top=6, ratio=None):
    """Walk-drawn stroke from A to B. din = direction arriving at A, dout = direction leaving B (radians).
    size = (W, H) of the picture; ink = boolean map of drawn lines (to avoid busy crossings)."""
    bk = bank(); rng = rng or np.random.default_rng()
    AB = np.asarray(B, float) - np.asarray(A, float); dist = float(np.hypot(*AB))
    if dist < 2: return None
    phi = math.atan2(AB[1], AB[0])
    # short gaps may curl, long gaps should travel
    lo, hi = (1.1, 3.2) if dist < 40 else ((1.05, 2.2) if dist < 150 else (1.02, 1.6))
    if ratio is not None: lo, hi = ratio(dist) if callable(ratio) else ratio      # caller's own path-length/distance range
    ok = (bk['ratio'] >= lo) & (bk['ratio'] <= hi)
    if not ok.any(): return None
    idx = np.nonzero(ok)[0]
    e0 = _wrap(bk['a0'][idx] - (din - phi)); e1 = _wrap(bk['a1'][idx] - (dout - phi))
    score = e0 ** 2 + 0.7 * e1 ** 2 + 0.15 * (bk['ratio'][idx] - lo) ** 2 + 0.6 * bk['wig'][idx]
    cand = idx[np.argsort(score)[:300]]
    c, s = math.cos(phi), math.sin(phi); R = np.array([[c, s], [-s, c]], np.float32)
    P = bk['C'][cand] @ R * dist + np.asarray(A, np.float32)                    # (k, N, 2) in picture space
    Wd, Ht = size
    mg = 0.02 * min(Wd, Ht)
    out = ((P[..., 0] < mg) | (P[..., 0] > Wd - mg) | (P[..., 1] < mg) | (P[..., 1] > Ht - mg)).mean(1)
    busy = 0
    if ink is not None:
        q = np.clip(np.round(P).astype(int), 0, [ink.shape[1] - 1, ink.shape[0] - 1])
        busy = ink[q[..., 1], q[..., 0]].mean(1)
    sc = np.sort(score)[:300] + 2.5 * busy
    sc = np.where(out > 0, np.inf, sc)                                        # never leave the picture
    if not np.isfinite(sc).any(): return None
    best = np.argsort(sc)[:top]
    pick = best[int(rng.integers(0, len(best)))]
    pts = P[pick].astype(float); pts[0] = A; pts[-1] = B
    return pts, bk['p'][cand[pick]].astype(float)
