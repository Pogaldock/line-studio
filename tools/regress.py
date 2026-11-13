# Regression check: trace the reference photos (tools/ref/<id>/photo.jpg) with the locked style and compare the line
# point for point with the reference drawing made on the original machine. Prints IDENTICAL or how far off it is.
# Usage: python3 tools/regress.py   (run from the repo root)
import sys, os, subprocess, time, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ENG = os.path.join(ROOT, 'engine'); REF = os.path.join(ROOT, 'tools', 'ref')
STYLE = 'scene=1 flow=10 Lmin=0.035 target=16 arc_max=0.8 avoid=0 route=1 levels=1.8'.split()
bad = 0
for i in sorted(os.listdir(REF)):
    r = os.path.join(REF, i); out = f'/tmp/_reg/{i}'; os.makedirs(out, exist_ok=True); t = time.time()
    subprocess.run([sys.executable, 'explore.py', os.path.join(r, 'photo.jpg'), out] + STYLE, check=True, capture_output=True, cwd=ENG)
    a, b = np.load(f'{r}/path_d1.npy'), np.load(f'{out}/path_d1.npy')
    same = a.shape == b.shape and np.array_equal(a, b) and json.load(open(f'{r}/kinds_d1.json')) == json.load(open(f'{out}/kinds_d1.json'))
    w = os.path.exists(f'{out}/widths_d1.npy') and np.array_equal(np.load(f'{r}/widths_d1.npy'), np.load(f'{out}/widths_d1.npy'))
    note = '' if same else (f'max point offset {np.abs(a - b).max() * 900 / 500:.2f}px' if a.shape == b.shape else f'{len(a)} vs {len(b)} points')
    print(f'{i}: {time.time() - t:5.1f} s  path {"IDENTICAL" if same else "DIFFERENT " + note}  widths {"same" if w else "diff"}', flush=True)
    bad += not (same and w)
print('ALL IDENTICAL' if not bad else f'{bad} DIFFER'); sys.exit(1 if bad else 0)
