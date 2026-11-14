# Profile-picture checks for the live (Nov 2026): decide, without a human, whether a picture gets drawn.
#
#   before tracing (instant, ~0.3 s):
#     blank       default avatar or a flat colour (almost no contrast)
#     blurry      too little sharp detail to find lines in (Laplacian variance on a 360 px copy)
#     nudity      NudeNet (MIT, 320n ONNX): exposed nudity (score >= 0.6) or, with suggestive=1, covered breasts/buttocks
#                 (>= 0.55) -> never drawn
#   after tracing (free: tracing runs while the previous drawing plays):
#     messy       too much of the line is joins between pieces (the drawing would be a tangle)
#     busy        lots of line *and* lots of small edges (collages, scenery, busy art)
#
# Thresholds come from 40 real profile pictures of the accounts' commenters (Nov 2026): they skip 7 of the 9 that came
# out unreadable and none of the 18 good ones. Usage: python3 pfpcheck.py <img> [<traced dir>]
import os, sys, json, numpy as np, cv2

C = dict(blank_std=18.0, blur_min=25.0, nsfw_score=0.6, suggestive=1, suggestive_score=0.55, join_max=0.27, busy_line=28.0, busy_edge=10.0, scene_edge=15.0, scene_line=20.0)
NSFW = {'FEMALE_BREAST_EXPOSED', 'FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED', 'ANUS_EXPOSED', 'BUTTOCKS_EXPOSED'}
SUGGESTIVE = {'FEMALE_BREAST_COVERED', 'BUTTOCKS_COVERED', 'FEMALE_GENITALIA_COVERED'}   # suggestive=1: also kept off an unwatched stream
_DET = {}

def features(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); s = cv2.resize(g, (360, 360), interpolation=cv2.INTER_AREA)
    return dict(std=float(g.std()), blur=float(cv2.Laplacian(s, cv2.CV_64F).var()),
                edge=100.0 * float((cv2.Canny(cv2.GaussianBlur(s, (3, 3), 0), 60, 150) > 0).mean()))

def nudity(img):
    if 'd' not in _DET:
        from nudenet import NudeDetector
        _DET['d'] = NudeDetector()
    # NudeNet also fires weakly on non-people (a cartoon dog 0.36, a toy duck 0.58 in the test set), hence 0.6
    hits = [d for d in _DET['d'].detect(img) if (d['class'] in NSFW and d['score'] >= C['nsfw_score'])
            or (C['suggestive'] and d['class'] in SUGGESTIVE and d['score'] >= C['suggestive_score'])]
    return max((d['score'] for d in hits), default=0.0)

def before(img):
    # -> (ok, reason, features)
    f = features(img)
    if f['std'] < C['blank_std']: return False, 'blank or default picture', f
    if f['blur'] < C['blur_min']: return False, 'too blurry to find lines', f
    if nudity(img) > 0: return False, 'not allowed on stream', f
    return True, '', f

def after(traced, f):
    # traced = the trace's output folder (path_d1.npy, kinds_d1.json); f = before()'s features
    P = np.load(os.path.join(traced, 'path_d1.npy')); line = float(np.hypot(*np.diff(P, axis=0).T).sum()) / 500.0
    k = json.load(open(os.path.join(traced, 'kinds_d1.json'))); join = sum(1 for x in k if x == 'walk') / max(1, len(k))
    if join >= C['join_max']: return False, 'too messy to draw cleanly', dict(line=line, join=join)
    if (line >= C['busy_line'] and f['edge'] >= C['busy_edge']) or (f['edge'] >= C['scene_edge'] and line >= C['scene_line']):
        return False, 'too busy to draw cleanly', dict(line=line, join=join)
    return True, '', dict(line=line, join=join)

if __name__ == '__main__':
    im = cv2.imread(sys.argv[1]); ok, why, f = before(im); print('before:', 'ok' if ok else 'skip: ' + why, {k: round(v, 1) for k, v in f.items()})
    if ok and len(sys.argv) > 2: ok, why, g = after(sys.argv[2], f); print('after:', 'ok' if ok else 'skip: ' + why, {k: round(v, 3) for k, v in g.items()})
