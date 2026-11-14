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
    wa, wb = np.load(f'{r}/widths_d1.npy'), np.load(f'{out}/widths_d1.npy') if os.path.exists(f'{out}/widths_d1.npy') else None
    note = '' if same else (f'max point offset {np.abs(a - b).max() * 900 / 500:.6f}px' if a.shape == b.shape else f'{len(a)} vs {len(b)} points')
    if not w and wb is not None and wa.shape == wb.shape: note += f', max width change {100 * np.abs(wa - wb).max() / max(wa.max(), 1e-9):.4f}%'
    try: note += ' | cpu: ' + [l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')][0]
    except Exception: pass
    # different CPU models round floating point differently in the last digits: under 0.001 px counts as the same drawing
    tiny = (not same) and a.shape == b.shape and np.abs(a - b).max() * 900 / 500 < 1e-3 and (w or (wb is not None and wa.shape == wb.shape and np.abs(wa - wb).max() <= 1e-4 * max(wa.max(), 1e-9)))
    verdict = 'IDENTICAL' if same and w else 'SAME within 0.001 px (CPU rounding)' if tiny else 'DIFFERENT ' + note
    print(f'{i}: {time.time() - t:5.1f} s  {verdict}', flush=True)
    bad += not (same and w) and not tiny
print('ALL SAME' if not bad else f'{bad} DIFFER'); sys.exit(1 if bad else 0)
