# Regression check: trace photos with the current code and compare to the frozen reference drawings (ref/c_<id>).
# Usage: python3 regress.py <id> [<id> ...]   (extra key=value args go to explore.py)
import sys, os, subprocess, time, json, numpy as np
ids = [a for a in sys.argv[1:] if '=' not in a]; extra = [a for a in sys.argv[1:] if '=' in a]
STYLE = 'scene=1 flow=10 Lmin=0.035 target=16 arc_max=0.8 avoid=0 route=1 levels=1.8'.split()
bad = 0
for i in ids:
    out = f'/tmp/_reg/{i}'; os.makedirs(out, exist_ok=True); t = time.time()
    subprocess.run([sys.executable, 'explore.py', f'cand3/{i}.jpg', out] + STYLE + extra, check=True, capture_output=True)
    dt = time.time() - t
    a, b = np.load(f'ref/c_{i}/path_d1.npy'), np.load(f'{out}/path_d1.npy')
    same = a.shape == b.shape and np.array_equal(a, b) and json.load(open(f'ref/c_{i}/kinds_d1.json')) == json.load(open(f'{out}/kinds_d1.json'))
    w = np.array_equal(np.load(f'ref/c_{i}/widths_d1.npy'), np.load(f'{out}/widths_d1.npy'))
    print(f'{i}: {dt:5.1f} s  path {"IDENTICAL" if same else "DIFFERENT"} {a.shape} vs {b.shape}  widths {"same" if w else "diff"}', flush=True)
    bad += not (same and w)
print('ALL IDENTICAL' if not bad else f'{bad} DIFFER')
