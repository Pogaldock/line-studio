# Side-by-side stills for testing a style change on real jobs (compare.yml). For one job: the photo, then each base
# option and its variant (the same style plus extra keys), traced in one go. Writes compare_<job>.jpg (labelled).
# Usage: python tools/compare.py <photo> <workdir> <out.jpg> <base ids, e.g. k,t> "<extra keys, e.g. budget=1>"
import sys, os, json, subprocess, cv2, numpy as np
photo, work, out, bases, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(','), sys.argv[5]
HERE = os.path.dirname(os.path.abspath(__file__)); ENG = os.path.join(HERE, '..', 'engine')
p = os.path.join(ENG, 'options.json'); o = json.load(open(p)); by = {x['id']: x for x in o['options']}
ids = []
for b in bases:                                   # variant id = the base id in capitals (only on this runner, never committed)
    v = dict(by[b], id=b.upper(), name=by[b]['name'] + ' +' + extra, style=by[b]['style'] + ' ' + extra)
    o['options'].append(v); ids += [b, b.upper()]
json.dump(o, open(p, 'w'))
r = subprocess.run([sys.executable, os.path.join(ENG, 'studio.py'), 'trace', photo, work, '--only', ','.join(ids)])
tiles = []
def tile(path, label):
    a = cv2.imread(path) if path and os.path.exists(path) else None
    a = np.full((800, 450, 3), 255, np.uint8) if a is None else cv2.resize(a, (450, int(a.shape[0] * 450 / a.shape[1])))
    a = a[:800] if a.shape[0] >= 800 else np.vstack([a, np.full((800 - a.shape[0], 450, 3), 255, np.uint8)])
    cv2.rectangle(a, (0, 0), (450, 34), (0, 0, 0), -1); cv2.putText(a, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return a
tiles.append(tile(os.path.join(work, 'photo.jpg'), 'photo'))
for b in bases:
    tiles.append(tile(os.path.join(work, f'still_{b}.jpg'), f'{by[b]["name"]} now'))
    tiles.append(tile(os.path.join(work, f'still_{b.upper()}.jpg'), f'{by[b]["name"]} + budget'))
cv2.imwrite(out, np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85]); print('wrote', out, flush=True)
