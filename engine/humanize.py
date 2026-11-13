# Finishing pass on a traced single line, before the video: removes glitches and makes it read hand-made.
#   evenly spaced points (no jumps: a long join stored as a few far-apart points drew as one sudden wedge)
#   tiny back-and-forth reversals removed (they drew as spiky stars)
#   gentle smoothing so corners turn in a flowing curve instead of a kink
#   slow hand drift (a fraction of a pixel to ~1 px, long wavelength) so lines are not machine-perfect
#   pressure: strokes start and end thinner, with slow width variation along each stroke
# Usage: python3 humanize.py <linedir> <outdir> [smooth=2.5] [drift=0.9] [taper=18] [seed=1]
#   reads/writes path_d1.npy, kinds_d1.json, secs_d1.npy, lazy_d1.npy, widths_d1.npy, meta_d1.json; writes line_d1.png
import sys, os, json, shutil, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import linephoto as LP

H_ = dict(smooth=2.5, drift=0.9, drift_wave=70.0, taper=18.0, var=0.18, seed=1, step=1.0, spike=4.0)

def gsmooth(x, s):
    if s <= 0 or len(x) < 5: return x
    return cv2.GaussianBlur(x.reshape(-1, 1).astype(np.float32), (1, 0), s, borderType=cv2.BORDER_REPLICATE).ravel()

def humanize(P, r, kinds, secs, lazy, px_per_unit):
    rng = np.random.default_rng(int(H_['seed']))
    Q = P * px_per_unit                                                     # work in drawing pixels
    # 1. drop repeated points, then remove tiny loops/spikes: a point is skipped when the path comes back within
    #    `spike` px of it after a detour shorter than 3x that (a pinhead knot, not a real loop)
    keep = np.r_[True, np.hypot(*np.diff(Q, axis=0).T) > 0.05]
    Q, r, kinds, secs, lazy = Q[keep], r[keep], [k for k, f in zip(kinds, keep) if f], secs[keep], lazy[keep]
    sp = H_['spike']; idx = [0]; i = 1; n = len(Q)
    while i < n:
        j_end = min(n, i + int(3 * sp / H_['step']) + 4)
        d = np.hypot(*(Q[i + 1:j_end] - Q[idx[-1]]).T) if i + 1 < j_end else np.array([])
        back = np.nonzero(d < 0.6)[0]
        if len(back): i = i + 1 + back[-1]; continue                       # skip the knot
        idx.append(i); i += 1
    idx = np.array(idx); Q, r, secs, lazy = Q[idx], r[idx], secs[idx], lazy[idx]; kinds = [kinds[i] for i in idx]
    # 2. resample to even spacing along the path (interpolates across the big jumps)
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(Q, axis=0).T))]
    t = np.arange(0, s[-1], H_['step'])
    Qn = np.c_[np.interp(t, s, Q[:, 0]), np.interp(t, s, Q[:, 1])]
    near = np.clip(np.searchsorted(s, t), 0, len(s) - 1)
    rn = np.interp(t, s, r); kn = [kinds[i] for i in near]; secn = secs[near]; lazn = lazy[near]
    # 3. flowing turns: smooth the path (strokes and joins alike, so a join leaves a stroke without a kink)
    for c in (0, 1): Qn[:, c] = gsmooth(Qn[:, c], H_['smooth'] / H_['step'])
    # 4. hand drift: slow sideways wander, a fraction of a pixel to ~1 px
    tg = np.gradient(Qn, axis=0); nrm = np.c_[-tg[:, 1], tg[:, 0]] / (np.hypot(*tg.T)[:, None] + 1e-9)
    wav = gsmooth(rng.normal(0, 1, len(Qn)), H_['drift_wave'] / H_['step']); wav /= np.abs(wav).max() + 1e-9
    Qn = Qn + nrm * (wav * H_['drift'])[:, None]
    # 5. pressure: each stroke (run of non-join points) tapers in and out; slow width variation along the line
    isj = np.array([k == 'walk' for k in kn]); run_id = np.cumsum(np.r_[True, isj[1:] != isj[:-1]])
    taper = np.ones(len(Qn)); L = H_['taper'] / H_['step']
    for rid in np.unique(run_id):
        ii = np.nonzero(run_id == rid)[0]
        if isj[ii[0]] or len(ii) < 4: continue
        a = np.arange(len(ii)); f = np.minimum(1, np.minimum(a + 1, len(ii) - a) / max(1.0, min(L, len(ii) / 2)))
        taper[ii] = 0.35 + 0.65 * np.sin(f * np.pi / 2)
    vary = 1 + H_['var'] * gsmooth(rng.normal(0, 1, len(Qn)), 40 / H_['step']) / 0.12
    rn = rn * gsmooth(taper, 2) * np.clip(vary, 1 - 2 * H_['var'], 1 + 2 * H_['var'])
    return Qn / px_per_unit, rn.astype(np.float32), kn, secn, lazn

def main():
    src, dst = sys.argv[1], sys.argv[2]
    for kv in sys.argv[3:]:
        k, v = kv.split('=', 1); H_[k] = float(v)
    os.makedirs(dst, exist_ok=True)
    meta = json.load(open(f'{src}/meta_d1.json')); ppu = meta['px_per_unit']
    P = np.load(f'{src}/path_d1.npy'); kinds = json.load(open(f'{src}/kinds_d1.json'))
    r = np.load(f'{src}/widths_d1.npy'); secs = np.load(f'{src}/secs_d1.npy'); lazy = np.load(f'{src}/lazy_d1.npy')
    P2, r2, k2, s2, l2 = humanize(P, r, kinds, secs, lazy, ppu)
    np.save(f'{dst}/path_d1.npy', P2); np.save(f'{dst}/widths_d1.npy', r2); json.dump(k2, open(f'{dst}/kinds_d1.json', 'w'))
    np.save(f'{dst}/secs_d1.npy', s2); np.save(f'{dst}/lazy_d1.npy', l2); json.dump(meta, open(f'{dst}/meta_d1.json', 'w'))
    st = np.hypot(*np.diff(P2 * ppu, axis=0).T)
    print(f'[humanize] {len(P)} -> {len(P2)} points, max step {st.max():.1f}px', flush=True)
    # still at the same scale as the other stills (900 px wide drawing)
    Wd = 500 * ppu; Hd = Wd * 16 / 9 if 'photo' not in meta else meta['photo'][1]
    return P2, r2, ppu

if __name__ == '__main__': main()
