# Timing test for a TikTok LIVE version (Nov 2026): how long does one profile picture take from photo to line path,
# with the models kept loaded between pictures (as a live program would)? Prints one line per picture and a summary.
#
# Usage: python3 pfpbench.py <outdir> <img1> [img2 ...]
#   pictures are cycled through the options in OPTS (one option per picture, like one viewer = one drawing);
#   the first picture also pays for loading the models (reported separately as "cold").
import sys, os, json, time, shutil, copy, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); CWD = os.getcwd(); os.chdir(HERE)
OPTS = ['e', 'a', 'q']                                   # Minimal (free tier), Detailed, Close-up clean (gift tiers)
T0 = time.time()
import explore as EX, hydrant_style as HS, linephoto as LP, depth as DP, route as RT
t_import = time.time() - T0
styles = {o['id']: o for o in json.load(open('options.json'))['options']}
base = {n: copy.deepcopy(d) for n, d in (('E', EX.E), ('D', DP.D), ('R', RT.R), ('A', HS.A), ('P', LP.P))}

def reset():
    for n, d in (('E', EX.E), ('D', DP.D), ('R', RT.R), ('A', HS.A), ('P', LP.P)): d.clear(); d.update(copy.deepcopy(base[n]))
    for k in list(EX._DEPTH): del EX._DEPTH[k]
    LP.SRC.clear()

def trace(img, out, style):
    reset(); rest = []
    for kv in style.split() + ['levels=1.8']:
        k, v = kv.split('=', 1)
        if k in EX.E: EX.E[k] = v if isinstance(EX.E[k], str) else type(EX.E[k])(float(v))
        elif k in DP.D: DP.D[k] = float(v)
        elif k in RT.R: RT.R[k] = float(v)
        else: rest.append(kv)
    sys.argv = ['explore.py', img, out] + rest
    if DP.D['depth']: EX._DEPTH['d'] = DP.depth_map(cv2.imread(img))
    HS.main(); EX.darkness_widths(img, out)
    if DP.D['depth']: EX.depth_weight(img, out)

if __name__ == '__main__':
    outdir = os.path.join(CWD, sys.argv[1]); os.makedirs(outdir, exist_ok=True); imgs = [os.path.join(CWD, x) for x in sys.argv[2:]]
    shutil.rmtree(os.path.join(HERE, 'models', 'cache'), ignore_errors=True)       # no help from earlier runs
    print(f'[bench] imports {t_import:.1f}s, {os.cpu_count()} cores', flush=True)
    rows = []
    for i, img in enumerate(imgs):
        oid = OPTS[i % len(OPTS)]; out = os.path.join(outdir, f'{i:02d}_{oid}')
        t = time.time(); trace(img, out, styles[oid]['style']); dt = time.time() - t
        P = np.load(os.path.join(out, 'path_d1.npy')); L = float(np.hypot(*np.diff(P, axis=0).T).sum()) / 500
        rows.append((i, oid, dt)); print(f'[bench] picture {i:2d} option {oid} ({styles[oid]["name"]}): {dt:5.1f}s  {"(cold: models load)" if i == 0 else ""}  line {L:.1f} widths', flush=True)
    warm = [r for r in rows if r[0] > 0]
    for oid in OPTS:
        ts = [r[2] for r in warm if r[1] == oid]
        if ts: print(f'[bench] warm {oid} ({styles[oid]["name"]}): median {np.median(ts):.1f}s, max {max(ts):.1f}s over {len(ts)}', flush=True)
    print(f'[bench] all warm pictures: median {np.median([r[2] for r in warm]):.1f}s', flush=True)
