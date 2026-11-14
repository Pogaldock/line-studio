# Photo -> one continuous line, after Wong & Takahashi (2011),
# "A Graph-based Approach to Continuous Line Illustrations with Variable Levels of Detail".
# Steps: multi-level edges -> segments with detail levels -> graph -> connect components ->
# fix dead ends -> semi-Eulerize -> smooth traversal -> level of detail -> trim, curves, curls -> render.
# Usage: python3 linephoto.py <image> <outdir> [width]
import sys, os, json, math, random, itertools, time
import numpy as np, cv2, networkx as nx
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import approximate_polygon
from scipy.spatial import Delaunay
from scipy.ndimage import convolve, label

P = dict(n=3, r0=1, rdec=4, s0=10, sdec=25, epsmul=0.6, smooth=4.5, spur=12, stroke_max=7, stroke_C=22, stroke_dark=70,          # bilateral sizes per level (paper)
         sig_e=1.0, tau=0.985, eps=0.009, min_area=25, # DoG edge detector (stand-in for FDoG)
         cover=0.8, merge=3.0, corner=80, h=25, alpha=4, beta=20, beta_ang=30,
         kappa=3, epsil=100, lam=30, gamma=0.2, cutoff=350,
         R=1.0, ss=3, fit=2.8, regions=10, obj_min=0.0015, obj_max=0.15, min_piece=36, paint=False, fine_rest=0.0, target=9.0, target_iters=5, lazy=True, lazy_win=31, lazy_d0=0.10, lazy_sigma=9.0, lazy_face_shrink=0.06, lazy_k=4.0, dense_cell=24, dense_max=0.12, face_r=0.8, prep=True, prep_clahe=2.0, bg_keep=0.2, blend=1.0, blend_rest=0.35, var=0.45, closeup_ref=0.25, face_pass=False, upscale=True, upscale_below=600, heq=False, detail_prop=True, detail_pow=0.5, enhance=False, clahe=1.6, clahe_local=3.0, lift_to=0.5, face_ink=8.0, hand_ink=3.5, structure='coarse', major=4.0, major_floor=0.15, major_res=416, focus=True, face_w=5.0, hand_w=1.5, part_w=1.5, part_min=0.0004, part_max=0.02, sil_bonus=3.0, budget=17.0, budget_grid=10, cell_cap=1.1, bg_weight=0.55, hop=0.04, sign='eye', eye_style='A', sign_font='fonts/MrsSaintDelafield-Regular.ttf', sign_h=0.07, walk=True, seed=1, nib_ang=70.0, nib_min=0.08, nib_scale=2.4, edges='lineart', la_res=1024, la_thr=(0.55, 0.32, 0.16), la_q=(0.70, 0.40), dens_win=31, dens_0=0.15, dens_k=8.0, bg_damp=0.85, bg_min_area=70)
T0 = time.time()
QUIET = [False]
def log(*a):
    if QUIET[0]: return
    print(f'[{time.time()-T0:6.1f}s]', *a, flush=True)

# ---------- 1. edges at n levels ----------
STROKE = {}
def stroke_mask(img):
    g = cv2.cvtColor(cv2.bilateralFilter(img, 0, 30, 4), cv2.COLOR_BGR2GRAY)
    dark = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 61, P['stroke_C']) > 0
    dark &= g < P['stroke_dark']   # properly dark paint, not grey gaps
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    dt = cv2.distanceTransform(dark.astype(np.uint8), cv2.DIST_L2, 5)
    lab_, n_ = label(dark, structure=np.ones((3, 3)))
    keep = np.zeros_like(dark)
    for l in range(1, n_ + 1):
        m = lab_ == l; a = m.sum()
        if a < 120: continue
        if np.percentile(dt[m], 95) <= P['stroke_max'] and dt[m].max() >= 2.0: keep |= m   # long thin dark mark, not a shadow
    STROKE['mask'] = keep; STROKE['dt'] = dt * keep
    log('painted marks', int(keep.sum()), 'px')
    return keep

SRC = {}
def _cache(name, img, fn):
    # model outputs are cached per picture (content hash), so re-tracing the same photo with new settings is quick
    import hashlib
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'cache'); os.makedirs(d, exist_ok=True)
    key = hashlib.md5(SRC.get('full', img).tobytes() + str(img.shape).encode() + str(P['la_res']).encode() + str(SRC.get('subj_box')).encode()).hexdigest()[:16]
    f = os.path.join(d, f'{name}_{key}.npy')
    if os.path.exists(f): log(name, 'from cache'); return np.load(f, allow_pickle=True)[()]
    out = fn(img); np.save(f, np.array(out, dtype=object) if out is None else out); return out

def face_pass(img, a):
    # close-up pass: each face cropped and re-drawn by the line-art model at full resolution (the model draws cleanest
    # when the subject fills its input), then used inside that face
    fh = _cache('facehand2', img, _focus_map); n_, ids = cv2.connectedComponents((fh[0] > 0).astype(np.uint8))
    full = SRC['full']; sx, sy = full.shape[1] / img.shape[1], full.shape[0] / img.shape[0]
    from controlnet_aux.lineart import LineartDetector
    from PIL import Image
    det = LineartDetector.from_pretrained('lllyasviel/Annotators')
    for z in range(1, n_):
        ys, xs = np.nonzero(ids == z); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        pad = int(0.15 * max(x1 - x0, y1 - y0)); x0, y0 = max(0, x0 - pad), max(0, y0 - pad); x1, y1 = min(img.shape[1], x1 + pad), min(img.shape[0], y1 + pad)
        crop = full[int(y0 * sy):int(y1 * sy), int(x0 * sx):int(x1 * sx)]
        if crop.size == 0: continue
        out = det(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), detect_resolution=512, image_resolution=512)
        fl = cv2.resize(np.array(out)[:, :, 0], (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        w = cv2.GaussianBlur((ids[y0:y1, x0:x1] == z).astype(np.float32), (0, 0), 3)
        a[y0:y1, x0:x1] = a[y0:y1, x0:x1] * (1 - w) + fl * w
    log('face close-up pass on', n_ - 1, 'faces')
    return a

MATTE_FIX = None   # set by explore.py: relative_matte (subject = whole frame on close-ups where background removal fails)

def prep_for_lines(full):
    # what the line models see: background removed (subject matte), subject tone-normalised and lightly smoothed.
    # The video still shows the untouched photo.
    import hashlib
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'cache'); os.makedirs(d, exist_ok=True)
    f = os.path.join(d, 'matte_' + hashlib.md5(full.tobytes()).hexdigest()[:16] + '.png')
    if os.path.exists(f): m = cv2.imread(f, 0).astype(np.float32) / 255
    else:
        from rembg import remove, new_session
        from PIL import Image
        m = np.array(remove(Image.fromarray(cv2.cvtColor(full, cv2.COLOR_BGR2RGB)), session=new_session('isnet-general-use'), only_mask=True)).astype(np.float32) / 255
        cv2.imwrite(f, (m * 255).astype(np.uint8))
    if MATTE_FIX is not None: m = MATTE_FIX(full, m)          # optional correction when background removal fails
    SRC['matte'] = m
    ys, xs = np.nonzero(m > 0.5)
    if len(xs) > 100:
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max(); pad = int(0.06 * max(x1 - x0, y1 - y0))
        SRC['subj_box'] = (max(0, x0 - pad), max(0, y0 - pad), min(full.shape[1], x1 + pad + 1), min(full.shape[0], y1 + pad + 1))
        SRC['subj_frac'] = float((m > 0.5).mean())
    lab = cv2.cvtColor(full, cv2.COLOR_BGR2LAB); L = cv2.createCLAHE(P['prep_clahe'], (8, 8)).apply(lab[:, :, 0])
    lab[:, :, 0] = (lab[:, :, 0] * (1 - m) + L * m).astype(np.uint8); t = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    t = cv2.bilateralFilter(t, 0, 25, 3)
    bg = cv2.GaussianBlur(full, (0, 0), 0.02 * max(full.shape[:2]) + 1).astype(np.float32)
    bg = bg * P['bg_keep'] + np.median(full.reshape(-1, 3), 0) * (1 - P['bg_keep'])
    mm = cv2.GaussianBlur(m, (0, 0), 3)[..., None]
    log('preprocess: subject covers', round(float((m > 0.5).mean()) * 100), '% of the photo; background removed')
    return (t * mm + bg * (1 - mm)).astype(np.uint8)

def upscale(img):
    # Real-ESRGAN x4 for small photos, so faces have enough pixels for clean lines (cached per photo)
    import hashlib
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'cache'); os.makedirs(d, exist_ok=True)
    f = os.path.join(d, 'up_' + hashlib.md5(img.tobytes()).hexdigest()[:16] + '.png')
    if os.path.exists(f): log('upscaled photo from cache'); return cv2.imread(f)
    import torch; from spandrel import ModelLoader
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    m = ModelLoader().load_from_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'RealESRGAN_x4plus.pth')).eval()
    x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None] / 255
    with torch.no_grad(): y = m(x)
    out = cv2.cvtColor((y[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f, out); log('upscaled', img.shape[1], 'x', img.shape[0], '->', out.shape[1], 'x', out.shape[0]); return out

def contour(img): return _cache('contour', img, _contour)
def _contour(img):
    import onnxruntime as ort
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'id_contour_1024.onnx')
    s = ort.InferenceSession(f, providers=['CPUExecutionProvider'])
    src = SRC.get('full', img); h, w = src.shape[:2]; side = max(h, w)
    sq = np.full((side, side, 3), np.median(src.reshape(-1, 3), axis=0), np.uint8)   # pad to a square (no stretching)
    oy, ox = (side - h) // 2, (side - w) // 2; sq[oy:oy + h, ox:ox + w] = src
    x = cv2.resize(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB), (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32).transpose(2, 0, 1)[None] / 255
    y = s.run(None, {s.get_inputs()[0].name: x})[0][0, 0]
    y = cv2.resize(np.clip(y, 0, 1), (side, side), interpolation=cv2.INTER_CUBIC)[oy:oy + h, ox:ox + w]
    return cv2.resize(1 - y, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)   # 1 = line

def lineart(img): return _cache('lineart', img, _lineart)
def lineart_major(img):
    r0 = P['la_res']; P['la_res'] = P['major_res']
    try: return _cache('lineart_major', img, _lineart)
    finally: P['la_res'] = r0
def _lineart(img):
    # learned line drawing (Chan et al. 2022, 'informative drawings'), the ControlNet line-art annotator
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    from controlnet_aux.lineart import LineartDetector
    from PIL import Image
    det = LineartDetector.from_pretrained('lllyasviel/Annotators')
    src = SRC.get('full', img); Hf, Wf = src.shape[:2]
    x0, y0, x1, y1 = SRC.get('subj_box', (0, 0, Wf, Hf))     # subject-normalised: the model sees the subject at a fixed size
    crop = src[y0:y1, x0:x1]
    out = det(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), detect_resolution=int(P['la_res']), image_resolution=int(P['la_res']))
    full_a = np.zeros((Hf, Wf), np.float32)
    full_a[y0:y1, x0:x1] = cv2.resize(np.array(out)[:, :, 0], (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return cv2.resize(full_a, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)   # 1 = line, 0 = paper

def edge_maps(img):
    n = P['n']; maps = []
    if P['edges'] == 'contour':
        # contour-style drawing (Chan et al. 2022, trained on artists' contour drawings): already selective
        a = contour(img)
        vals = a[a > 0.15]
        thr = [float(np.quantile(vals, q)) for q in (0.6, 0.3)] + [0.15] if len(vals) else [0.5, 0.3, 0.15]
        for t_ in thr: maps.append(remove_small_objects(a > t_, max_size=P['min_area']))
        SRC['la'] = a
        log('contour drawing: line pixels per level', [int(m.sum()) for m in maps])
        return maps
    if P['edges'] == 'lineart':
        a = lineart(img)
        if P['structure'] == 'coarse':
            # readable structure comes from the coarse-scale drawing (only major lines survive there);
            # the fine drawing is used only where detail matters: faces, hands, standout small parts
            co = cv2.resize(lineart_major(img), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
            fm = np.zeros(img.shape[:2], np.float32)
            if OBJECTS.get('facehand') is not None: fm = np.maximum(fm, OBJECTS['facehand'].astype(np.float32))
            if OBJECTS.get('parts') is not None: fm = np.maximum(fm, 0.7 * OBJECTS['parts'].astype(np.float32))
            fm = cv2.GaussianBlur(fm, (0, 0), 0.01 * img.shape[1] + 1)
            a = np.maximum(co, a * np.maximum(fm, P['fine_rest']))   # fine detail: full in focus areas, fine_rest elsewhere
            if P['blend'] > 0:                                # important details from the contour-style drawing
                cm = contour(img).copy(); bm = int(0.025 * min(cm.shape)); cm[:bm] = 0; cm[-bm:] = 0; cm[:, :bm] = 0; cm[:, -bm:] = 0   # no frame along the photo edge
                wgt = P['blend'] * (P['blend_rest'] + (1 - P['blend_rest']) * np.clip(fm * 1.5, 0, 1))
                a = np.maximum(a, cm * wgt)
            if P['face_pass']: a = face_pass(img, a)
            log('structure: coarse drawing + fine detail in', round(float((fm > 0.3).mean()) * 100, 1), '% of the picture (faces, hands, details)')
        # texture control: outside detected objects, weaken lines by how crowded the area is (foliage, branches,
        # cobbles turn into light hints); objects (people, things) keep full detail
        dens = cv2.blur((a > P['la_thr'][-1]).astype(np.float32), (P['dens_win'], P['dens_win']))
        damp = P['bg_damp'] / (1.0 + P['dens_k'] * np.maximum(dens - P['dens_0'], 0) / (1 - P['dens_0']))
        if OBJECTS.get('labels') is not None: damp = np.where(OBJECTS['labels'] > 0, 1.0, damp)   # objects keep full detail
        if OBJECTS.get('facehand') is not None: damp = np.where(OBJECTS['facehand'], 1.0, damp)   # faces and hands always do
        a_raw = a; a = a * damp; SRC['la'] = a_raw
        if P['major'] > 0:                                    # lines that survive at a coarse scale = the major lines
            mj = lineart_major(img); mj = cv2.dilate((mj > 0.12).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
            SRC['major'] = cv2.GaussianBlur(mj, (0, 0), 2)
            log('major lines (coarse scale) cover', round(float((mj > 0).mean()) * 100, 1), '% of the picture')
        log('texture control: background lines kept', round(float(((a > P['la_thr'][-1]) & (damp < 1)).sum() / max(1, ((a_raw > P['la_thr'][-1]) & (damp < 1)).sum())) * 100), '%')
        vals = a_raw[a_raw > P['la_thr'][-1]]                 # thresholds from this picture's own line strengths
        thr = [float(np.quantile(vals, q)) for q in P['la_q']] + [P['la_thr'][-1]] if len(vals) else P['la_thr']
        for t_ in thr:                                        # strong lines = level 1 ... faint lines = level n
            m_ = a > t_
            if OBJECTS.get('labels') is not None:             # stray specks in the background go
                m_ = remove_small_objects(m_, max_size=P['min_area']) & ~((OBJECTS['labels'] == 0) & ~remove_small_objects(m_, max_size=P['bg_min_area']))
            maps.append(remove_small_objects(m_, max_size=P['min_area']))
        log('line-art pixels per level', [int(m.sum()) for m in maps])
        return maps
    sm = stroke_mask(img) if P['paint'] else np.zeros(img.shape[:2], bool)   # painted marks (graffiti etc.) only on request
    sdil = cv2.dilate(sm.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    for i in range(1, n + 1):
        f = img.copy(); ss = P['r0'] + (n - i) * P['rdec']; sr = P['s0'] + (n - i) * P['sdec']
        for _ in range(5): f = cv2.bilateralFilter(f, 0, sr, ss)
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        g1 = cv2.GaussianBlur(g, (0, 0), P['sig_e']); g2 = cv2.GaussianBlur(g, (0, 0), P['sig_e'] * 1.6)
        m = (g1 - P['tau'] * g2) < -P['eps'] * (1 + (n - i) * P['epsmul'])
        m = remove_small_objects(m, max_size=P['min_area'] * (1 + 2 * (n - i)))
        maps.append((m & ~sdil) | sm)
    return maps   # maps[0] = level 1 (coarsest), maps[-1] = finest

# ---------- 2. skeleton -> segments ----------
OFF = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
def segments(sk):
    H, W = sk.shape
    nb = convolve(sk.astype(int), np.ones((3, 3), int), mode='constant') - 1; nb[~sk] = 0
    node = sk & (nb != 2)
    lab, nl = label(node, structure=np.ones((3, 3)))
    def nbrs(p):
        y, x = p; return [(y+dy, x+dx) for dy, dx in OFF if 0 <= y+dy < H and 0 <= x+dx < W and sk[y+dy, x+dx]]
    used = np.zeros_like(sk); segs = []; pairs = set()
    def walk(start, first):
        path = [start, first]; prev, cur = start, first
        while True:
            if node[cur]: return path
            used[cur] = True
            nx_ = [r for r in nbrs(cur) if r != prev and not (used[r] and not node[r])]
            if not nx_: return path
            # prefer a node neighbour so we stop at junctions
            nxt = next((r for r in nx_ if node[r]), nx_[0])
            if nxt == start and len(path) > 3: path.append(nxt); return path
            path.append(nxt); prev, cur = cur, nxt
    for y, x in zip(*np.nonzero(node)):
        p = (y, x)
        for q in nbrs(p):
            if node[q]:
                a, b = lab[p], lab[q]
                if a != b and (min(a, b), max(a, b)) not in pairs: pairs.add((min(a, b), max(a, b))); segs.append([p, q])
                continue
            if used[q]: continue
            segs.append(walk(p, q))
    # closed loops with no junction
    for y, x in zip(*np.nonzero(sk & ~node & ~used)):
        if used[y, x]: continue
        p = (y, x); ns = nbrs(p)
        if not ns: continue
        node[p] = True; lab[p] = nl = nl + 1
        segs.append(walk(p, ns[0])); node[p] = False
    # cluster centres
    cents = {}
    for l in range(1, nl + 1):
        ys, xs = np.nonzero(lab == l) if False else (None, None)
    ys, xs = np.nonzero(lab); ls = lab[ys, xs]
    sums = {}
    for l, yy, xx in zip(ls, ys, xs):
        s = sums.setdefault(l, [0, 0, 0]); s[0] += xx; s[1] += yy; s[2] += 1
    cents = {l: (s[0] / s[2], s[1] / s[2]) for l, s in sums.items()}
    out = []
    for s in segs:
        a, b = s[0], s[-1]
        if not (node[a] or lab[a]) or not (node[b] or lab[b]): continue
        la, lb = lab[a], lab[b]
        if la == 0 or lb == 0: continue
        pts = np.array([(x, y) for y, x in s], float)
        pts[0] = cents[la]; pts[-1] = cents[lb]
        out.append((la, lb, pts))
    return out, cents

def arclen(p): return float(np.sum(np.hypot(*np.diff(p, axis=0).T))) if len(p) > 1 else 0.0

def resample(p, step):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    if d[-1] < 1e-6: return p[[0, -1]]
    t = np.linspace(0, d[-1], max(2, int(math.ceil(d[-1] / step)) + 1))
    return np.c_[np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])]

# ---------- object segmentation (FastSAM) ----------
OBJECTS = {}
def segment_objects(img):
    both = _cache('objects2', img, _segment_objects)
    if both is None: return None
    OBJECTS['parts'] = both[1] > 0; return both[0]
def _segment_objects(img):
    # label map: each pixel -> the largest mid-sized object outline covering it (whole person, not just the vest)
    try:
        from ultralytics import FastSAM
    except ImportError:
        log('no ultralytics: falling back to position-only sections'); return None
    here = os.path.dirname(os.path.abspath(__file__))
    m = FastSAM(os.path.join(here, 'models', 'FastSAM-s.pt'))
    r = m(img, imgsz=640, conf=0.4, iou=0.9, retina_masks=True, verbose=False)[0]
    if r.masks is None: return None
    mk = r.masks.data.numpy().astype(bool); H_, W_ = img.shape[:2]
    if mk.shape[1:] != (H_, W_): mk = np.array([cv2.resize(x.astype(np.uint8), (W_, H_), interpolation=cv2.INTER_NEAREST) > 0 for x in mk])
    area = mk.reshape(len(mk), -1).sum(1) / (H_ * W_)
    keep = np.nonzero((area > P['obj_min']) & (area < P['obj_max']))[0]
    lab = np.zeros((H_, W_), np.int32)
    for n_, i in enumerate(keep[np.argsort(area[keep])], 1): lab[mk[i]] = n_      # larger written last: whole objects win
    log('objects found', len(keep), 'of', len(mk), 'outlines')
    small = np.nonzero((area > P['part_min']) & (area < P['part_max']))[0]       # distinct small items: lettering, logos, chains
    parts = np.zeros((H_, W_), np.int32)
    for i in small: parts[mk[i]] = 1
    return np.stack([lab, parts])

# ---------- focus: where detail should go (faces, hands, standout small parts) ----------
def person_mask(img): return _cache('persons', img, _person_mask)
def _person_mask(img):
    m = np.zeros(img.shape[:2], bool)
    try:
        from ultralytics import YOLO
        r = YOLO(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'yolov8n-pose.pt'))(img, verbose=False)[0]
        for b in (r.boxes.xyxy.numpy() if r.boxes is not None else []):
            x0, y0, x1, y1 = [int(v) for v in b]; m[max(0, y0):y1, max(0, x0):x1] = True
    except Exception as e: log('person boxes unavailable:', e)
    return m

def focus_map(img):
    fh = _cache('facehand2', img, _focus_map)               # cached: face and hand areas only; weights applied here
    OBJECTS['facehand'] = (fh[0] > 0) | (fh[1] > 0)
    OBJECTS['face_ids'] = cv2.connectedComponents((fh[0] > 0).astype(np.uint8))[1]; OBJECTS['hand_ids'] = cv2.connectedComponents(((fh[1] > 0) & ~(fh[0] > 0)).astype(np.uint8))[1]
    H_, W_ = img.shape[:2]
    F = 1 + P['face_w'] * fh[0].astype(np.float32) + P['hand_w'] * fh[1].astype(np.float32)
    if OBJECTS.get('parts') is not None: F += P['part_w'] * OBJECTS['parts']
    return cv2.GaussianBlur(F.astype(np.float32), (0, 0), 0.01 * W_ + 1)
def _focus_map(img):
    H_, W_ = img.shape[:2]; F = np.ones((H_, W_), np.float32)
    try:
        from ultralytics import YOLO
        r = YOLO(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'yolov8n-pose.pt'))(img, verbose=False)[0]
        kp = r.keypoints.xy.numpy() if r.keypoints is not None else np.zeros((0, 17, 2)); cf = r.keypoints.conf.numpy() if r.keypoints is not None and r.keypoints.conf is not None else np.ones(kp.shape[:2])
    except Exception as e:
        log('pose model unavailable:', e); kp = np.zeros((0, 17, 2)); cf = np.zeros((0, 17))
    face = np.zeros((H_, W_), np.uint8); hand = np.zeros((H_, W_), np.uint8)
    for k, c in zip(kp, cf):
        fk = [i for i in range(5) if c[i] > 0.3]                      # nose, eyes, ears
        if len(fk) >= 2:
            pts_ = k[fk]; ctr = pts_.mean(0); span = np.ptp(pts_, axis=0).max()
            rad = max(P['face_r'] * span, 0.03 * W_); cv2.ellipse(face, (int(ctr[0]), int(ctr[1] + 0.15 * rad)), (int(0.85 * rad), int(rad)), 0, 0, 360, 1, -1)   # eyes-nose-mouth only, no hair
            for w_, e_ in ((9, 7), (10, 8)):                           # wrists, pushed a little towards the hand
                if c[w_] > 0.3:
                    hp = k[w_] + (0.35 * (k[w_] - k[e_]) if c[e_] > 0.3 else 0)
                    cv2.circle(hand, (int(hp[0]), int(hp[1])), int(0.8 * rad), 1, -1)
    log('focus: people', len(kp), '| face area', round(face.mean() * 100, 1), '% | hands', round(hand.mean() * 100, 1), '%')
    return np.stack([face, hand])

# ---------- local contrast for the line drawing: lift dark faces and hands ----------
def enhance(full, face_small, hand_small):
    lab = cv2.cvtColor(full, cv2.COLOR_BGR2LAB); L = lab[:, :, 0]
    Lg = cv2.createCLAHE(clipLimit=P['clahe'], tileGridSize=(8, 8)).apply(L)            # mild, whole picture
    out = Lg.astype(np.float32)
    H_, W_ = L.shape
    for small in (face_small, hand_small):
        if small is None or not small.any(): continue
        n_, ids = cv2.connectedComponents(cv2.resize(small.astype(np.uint8), (W_, H_), interpolation=cv2.INTER_NEAREST))
        for z in range(1, n_):
            m = ids == z; mean = float(L[m].mean()) / 255
            g = float(np.clip(math.log(P['lift_to']) / math.log(max(mean, 1e-3)), 0.45, 1.0))   # gamma that brings it to mid-tone
            lifted = (255 * (L.astype(np.float32) / 255) ** g).astype(np.uint8)
            loc = cv2.createCLAHE(clipLimit=P['clahe_local'], tileGridSize=(4, 4)).apply(lifted).astype(np.float32)
            w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.01 * W_ + 2)
            out = out * (1 - w) + loc * w
            log(f'contrast: area {z} mean brightness {mean:.2f} -> gamma {g:.2f}')
    lab[:, :, 0] = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ---------- 3. graph ----------
class Graph:
    def __init__(self): self.G = nx.MultiGraph(); self.nid = 0
    def node(self, pos, orig=True):
        self.nid += 1; self.G.add_node(self.nid, pos=np.array(pos, float), orig=orig); return self.nid
    def edge(self, a, b, pts, level, kind, base=None):
        k = self.G.add_edge(a, b, pts=pts, a=a, level=level, kind=kind, length=max(arclen(pts), 1e-3), base=base)
        return k
    def pos(self, n): return self.G.nodes[n]['pos']

def oriented(e, a):
    return e['pts'] if e['a'] == a else e['pts'][::-1]

def out_dir(e, a, k=6):
    p = oriented(e, a); d = 0; i = 1
    while i < len(p) - 1 and d < k: d += np.hypot(*(p[i] - p[i - 1])); i += 1
    v = p[min(i, len(p) - 1)] - p[0]; n_ = np.hypot(*v)
    return v / n_ if n_ > 1e-9 else np.array([1.0, 0.0])

def build(img):
    maps = edge_maps(img); n = P['n']
    log('edge pixels per level', [int(m.sum()) for m in maps])
    sk = skeletonize(maps[-1])
    segs, cents = segments(sk)
    log('segments', len(segs))
    # merge close cluster centres
    labs = list(cents); pts = np.array([cents[l] for l in labs])
    parent = {l: l for l in labs}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    if len(pts) > 3:
        tri = Delaunay(pts + np.random.default_rng(0).normal(0, 1e-3, pts.shape))
        for s in tri.simplices:
            for i, j in ((0, 1), (1, 2), (0, 2)):
                if np.hypot(*(pts[s[i]] - pts[s[j]])) < P['merge']: parent[find(labs[s[i]])] = find(labs[s[j]])
    gr = Graph(); vid = {}
    groups = {}
    for l in labs: groups.setdefault(find(l), []).append(cents[l])
    for r, ps in groups.items(): vid[r] = gr.node(np.mean(ps, 0))
    dil = [cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0 for m in maps]
    H, W = maps[0].shape
    raw = []
    for la, lb, p in segs:
        a, b = vid[find(la)], vid[find(lb)]
        p = p.copy(); p[0] = gr.pos(a); p[-1] = gr.pos(b)
        L = arclen(p)
        if a == b and L < 12: continue
        if len(p) > 5:
            q = p.copy(); sg = P['smooth']
            q[:, 0] = cv2.GaussianBlur(p[:, 0].reshape(-1, 1).astype(np.float32), (1, 0), sg, borderType=cv2.BORDER_REPLICATE).ravel()
            q[:, 1] = cv2.GaussianBlur(p[:, 1].reshape(-1, 1).astype(np.float32), (1, 0), sg, borderType=cv2.BORDER_REPLICATE).ravel()
            w_ = np.minimum(np.arange(len(p)), np.arange(len(p))[::-1]); w_ = np.clip(w_ / (2 * sg), 0, 1)[:, None]
            p = p * (1 - w_) + q * w_
            ap = approximate_polygon(p, tolerance=P['fit'])
            if len(ap) >= 3:
                p = sweep(ap) if 'sweep' in globals() else p
                p[0] = ap[0]; p[-1] = ap[-1]
        if L < 1: continue
        ix = np.clip(np.round(p).astype(int), 0, [W - 1, H - 1])
        lev = n
        for i in range(n):
            if dil[i][ix[:, 1], ix[:, 0]].mean() >= P['cover']: lev = i + 1; break
        raw.append([a, b, p, lev])
    # prune short spurs (dead ends shorter than 6 px)
    deg = {}
    for a, b, p, l in raw: deg[a] = deg.get(a, 0) + 1; deg[b] = deg.get(b, 0) + 1
    for _ in range(2):
        deg = {}
        for a, b, p, l in raw: deg[a] = deg.get(a, 0) + 1; deg[b] = deg.get(b, 0) + 1
        raw = [r for r in raw if not (arclen(r[2]) < P['spur'] and (deg[r[0]] == 1 or deg[r[1]] == 1))]
    # length promotion: long edges go up a level
    for lv in range(n, 1, -1):
        Ls = [arclen(r[2]) for r in raw if r[3] == lv]
        if len(Ls) < 2: continue
        eta = np.mean(Ls) + np.std(Ls)
        for r in raw:
            if r[3] == lv and arclen(r[2]) > eta: r[3] = lv - 1
    log('levels', {lv: sum(1 for r in raw if r[3] == lv) for lv in range(1, n + 1)})
    # split at sharp corners, then subdivide every h px
    for a, b, p, lev in raw:
        cuts = [0]
        if len(p) > 4:
            ap = approximate_polygon(p, tolerance=1.5)
            for k in range(1, len(ap) - 1):
                v1 = ap[k] - ap[k - 1]; v2 = ap[k + 1] - ap[k]
                c = np.dot(v1, v2) / (np.hypot(*v1) * np.hypot(*v2) + 1e-9)
                if math.degrees(math.acos(np.clip(c, -1, 1))) > P['corner']:
                    idx = int(np.argmin(np.hypot(*(p - ap[k]).T)))
                    if cuts[-1] < idx < len(p) - 1: cuts.append(idx)
        cuts.append(len(p) - 1)
        prev = a
        for ci in range(len(cuts) - 1):
            piece = p[cuts[ci]:cuts[ci + 1] + 1]
            end = b if ci == len(cuts) - 2 else gr.node(piece[-1], orig=True)
            L = arclen(piece); k = max(1, int(L // P['h']))
            d = np.r_[0, np.cumsum(np.hypot(*np.diff(piece, axis=0).T))]
            bounds = [int(np.searchsorted(d, L * j / k)) for j in range(k)] + [len(piece) - 1]
            q = prev
            for j in range(k):
                sub = piece[bounds[j]:max(bounds[j + 1], bounds[j] + 1) + 1]
                r_ = end if j == k - 1 else gr.node(sub[-1], orig=False)
                sub = sub.copy(); sub[0] = gr.pos(q); sub[-1] = gr.pos(r_)
                gr.edge(q, r_, sub, lev, 'line'); q = r_
            prev = end
    G = gr.G
    log('graph', G.number_of_nodes(), 'nodes', G.number_of_edges(), 'edges', nx.number_connected_components(G), 'components')
    return gr

def straight(a, b): return resample(np.array([a, b], float), 2.0)

def connect(gr, join=True):
    G = gr.G
    nodes = list(G.nodes); pos = np.array([gr.pos(v) for v in nodes])
    orig = [v for v in nodes if G.nodes[v]['orig']]; opos = np.array([gr.pos(v) for v in orig])
    cand = []
    def tri_edges(ids, ps, mult):
        if len(ids) < 4: return
        t = Delaunay(ps + np.random.default_rng(1).normal(0, 1e-3, ps.shape))
        seen = set()
        for s in t.simplices:
            for i, j in ((0, 1), (1, 2), (0, 2)):
                u, v = ids[s[i]], ids[s[j]]
                if (u, v) in seen or (v, u) in seen: continue
                seen.add((u, v)); cand.append((mult * np.hypot(*(ps[s[i]] - ps[s[j]])), u, v))
        return t
    tri_edges(orig, opos, 1.0)
    tV = tri_edges(nodes, pos, P['alpha'])
    comp = {}
    for ci, c in enumerate(nx.connected_components(G)):
        for v in c: comp[v] = ci
    parent = {c: c for c in set(comp.values())}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    added = 0
    for w, u, v in (sorted(cand) if join else []):
        a, b = find(comp[u]), find(comp[v])
        if a != b:
            parent[a] = b; gr.edge(u, v, straight(gr.pos(u), gr.pos(v)), P['n'], 'aux'); added += 1
    log('connector edges', added, 'components now', nx.number_connected_components(G))
    # dead ends: extend to a close vertex in roughly the same direction
    nbr = {}
    if tV is not None:
        ind, ptr = tV.vertex_neighbor_vertices
        for i, v in enumerate(nodes): nbr[v] = [nodes[j] for j in ptr[ind[i]:ind[i + 1]]]
    ext = 0
    for u in list(G.nodes):
        if G.degree(u) != 1: continue
        (_, w, k, e), = G.edges(u, keys=True, data=True)
        din = -out_dir(e, u)
        best = None
        for v in nbr.get(u, []):
            if v == w or G.has_edge(u, v): continue
            dv = gr.pos(v) - gr.pos(u); d = np.hypot(*dv)
            if d < P['beta'] and np.dot(dv / d, din) >= math.cos(math.radians(P['beta_ang'])):
                if best is None or d < best[0]: best = (d, v)
        if best: gr.edge(u, best[1], straight(gr.pos(u), gr.pos(best[1])), e['level'], 'line'); ext += 1
    log('dead-end extensions', ext)
    # merge degree-two subdivision vertices of the same level
    merged = 0
    for v in list(G.nodes):
        if G.nodes[v]['orig'] or G.degree(v) != 2: continue
        es = list(G.edges(v, keys=True, data=True))
        if len(es) != 2 or es[0][1] == v or es[1][1] == v: continue
        (_, x, k1, e1), (_, y, k2, e2) = es
        if e1['level'] != e2['level'] or e1['kind'] != e2['kind'] or x == y: continue
        p = np.vstack([oriented(e1, x), oriented(e2, v)[1:]])
        G.remove_node(v); gr.edge(x, y, p, e1['level'], e1['kind']); merged += 1
    log('merged', merged, '->', G.number_of_nodes(), 'nodes', G.number_of_edges(), 'edges')

# ---------- 4. semi-Eulerize ----------
def eulerize(gr):
    G = gr.G
    S = nx.Graph()
    for a, b, k, e in G.edges(keys=True, data=True):
        if a == b: continue
        if not S.has_edge(a, b) or S[a][b]['length'] > e['length']: S.add_edge(a, b, length=e['length'], key=k)
    odd = [v for v in G.nodes if G.degree(v) % 2]
    log('odd vertices', len(odd))
    oddset = set(odd); pairs = []
    for u in odd:
        dist, paths = nx.single_source_dijkstra(S, u, cutoff=P['cutoff'], weight='length')
        for v, l in dist.items():
            if v in oddset and v > u: pairs.append((l, u, v, paths[v]))
    pairs.sort(key=lambda t: t[0])
    done = set(); dups = {}; nshort = ndup = 0
    remaining = len(odd)
    for l, u, v, path in pairs:
        if remaining <= 2: break
        if u in done or v in done: continue
        d = np.hypot(*(gr.pos(u) - gr.pos(v)))
        if l > P['kappa'] * d and d < P['epsil']:
            gr.edge(u, v, straight(gr.pos(u), gr.pos(v)), P['n'], 'aux'); nshort += 1
        else:
            for a, b in zip(path[:-1], path[1:]):
                k = S[a][b]['key']; key = (min(a, b), max(a, b), k)
                if key in dups: G.remove_edge(a, b, dups.pop(key))
                else:
                    e = G[a][b][k]; dups[key] = gr.edge(a, b, e['pts'], e['level'], 'dup', base=key) if e['a'] == a else gr.edge(b, a, e['pts'], e['level'], 'dup', base=key)
                    ndup += 1
        done.add(u); done.add(v); remaining -= 2
    left = [v for v in G.nodes if G.degree(v) % 2]
    # leftovers far apart: pair greedily by distance with direct connectors
    while len(left) > 2:
        u = left.pop(); ps = np.array([gr.pos(v) for v in left]); j = int(np.argmin(np.hypot(*(ps - gr.pos(u)).T)))
        v = left.pop(j); gr.edge(u, v, straight(gr.pos(u), gr.pos(v)), P['n'], 'aux')
    log('shortcuts', nshort, 'retraced edges', ndup, 'odd left', len([v for v in G.nodes if G.degree(v) % 2]))

# ---------- 5. traversal: Fleury, with straightest pairing at each vertex ----------
def min_match(vecs, idx):
    # min total cost of pairing idx (cost = dot of outgoing directions; straight-through = -1); one may stay single
    best = [0.0]
    @lru(maxsize=None)
    def rec(fs):
        if len(fs) <= 1: return 0.0
        f = fs[0]; rest = fs[1:]; b = 1e9
        if len(fs) % 2: b = rec(rest)
        for j, g in enumerate(rest):
            b = min(b, float(np.dot(vecs[f], vecs[g])) + rec(rest[:j] + rest[j + 1:]))
        return b
    return rec(tuple(idx))
from functools import lru_cache as lru

def traverse(gr, start_pt=None, sections=True):
    G = gr.G
    odd = [v for v in G.nodes if G.degree(v) % 2]
    # sections: every edge belongs to an object (from the segmentation) or, outside objects, to a background area
    # by position; the line finishes the section it is in before moving on
    es = list(G.edges(keys=True, data=True)) if sections else []; LBL = OBJECTS.get('labels')
    bg = []
    for idx_, (a_, b_, k_, e) in enumerate(es):
        lab = 0
        if LBL is not None:
            q = np.clip(np.round(e['pts'][::3]).astype(int), 0, [LBL.shape[1] - 1, LBL.shape[0] - 1])
            ls = LBL[q[:, 1], q[:, 0]]; ls = ls[ls > 0]
            if len(ls) * 2 >= max(1, len(q)): lab = int(np.bincount(ls).argmax())   # mostly inside one object
        e['reg'] = lab
        if lab == 0: bg.append(idx_)
    nobj = len({e['reg'] for _, _, _, e in es} - {0}) if es else 0
    nbg = int(np.clip(len(bg) / 80, 1, P['regions'])) if bg else 0
    if nbg > 1:
        mids = np.array([es[i][3]['pts'][len(es[i][3]['pts']) // 2] for i in bg], np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, labels, _ = cv2.kmeans(mids, nbg, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        for i, l in zip(bg, labels.ravel()): es[i][3]['reg'] = 100000 + int(l)
    nreg = nobj + nbg
    cands = odd if odd else list(G.nodes)
    start = (min(cands, key=lambda v: gr.pos(v)[0] + gr.pos(v)[1]) if start_pt is None     # top-left, like the reference
             else min(cands, key=lambda v: float(np.hypot(*(gr.pos(v) - start_pt)))))
    cur_reg = None; switches = 0; started = set()
    v = start; path = []; din = None; total = G.number_of_edges(); t = time.time()
    while G.degree(v) > 0:
        inc = list(G.edges(v, keys=True, data=True))
        vecs = [out_dir(e, v) for _, _, _, e in inc]
        if din is None or len(inc) > 9:
            if din is None: order = list(range(len(inc)))
            else: order = sorted(range(len(inc)), key=lambda i: float(np.dot(din, vecs[i])))
        else:
            allv = vecs + [din]; me = len(vecs)
            cost = []
            for i in range(len(inc)):
                rest = tuple(j for j in range(len(inc)) if j != i)
                cost.append(float(np.dot(din, vecs[i])) + min_match(allv, rest))
            order = sorted(range(len(inc)), key=lambda i: cost[i])
        if cur_reg is not None:                   # stay in the current section; else go back to one already started
            def rank(i):
                r_ = inc[i][3].get('reg')
                return 0 if r_ == cur_reg else (1 if r_ in started else 2)
            order = sorted(order, key=rank)
        chosen = None
        for i in order:
            _, w, k, e = inc[i]
            if len(order) == 1: chosen = i; break
            if w == v: chosen = i; break            # self loop never disconnects
            data = dict(e); G.remove_edge(v, w, k)
            ok = nx.has_path(G, v, w)
            G.add_edge(v, w, key=k, **data)
            if ok: chosen = i; break
        if chosen is None: chosen = order[0]
        _, w, k, e = inc[chosen]
        if e.get('reg') != cur_reg: switches += 1; cur_reg = e.get('reg'); started.add(cur_reg)
        p = oriented(e, v)
        path.append(dict(u=v, v=w, pts=p, level=e['level'], kind=e['kind'], base=e['base'] or (min(e['a'], w if e['a'] == v else v), max(e['a'], w if e['a'] == v else v), k)))
        G.remove_edge(v, w, k)
        q = p[::-1]; din = out_dir({'pts': q, 'a': 0}, 0)   # direction we arrived along, pointing back
        din = -din if False else din
        v = w
        if len(path) % 500 == 0: log(f'  traversed {len(path)}/{total}')
    log('path edges', len(path), 'of', total, '| objects', nobj, 'background areas', nbg, 'section changes', switches)
    return path

# ---------- 5a. line budget: clear outlines first, ink spread evenly ----------
def budget_pieces(pieces):
    # rank pieces by length x line strength x (object or background); accept them best-first, but each area of the
    # picture (grid cell) can only take so much ink, so busy spots stop growing and sparse spots keep their outlines.
    # Stop at the budget (in picture widths of line), which matches the reference videos' density.
    if not pieces: return pieces
    la = SRC.get('la'); allp = np.vstack([p['pts'] for p in pieces]); Wd = max(allp[:, 0].max(), 1); Hd = max(allp[:, 1].max(), 1)
    budget = P['budget'] * Wd; cell = max(Wd, Hd) / P['budget_grid']
    def cells(p): return np.unique((np.floor(p[:, 1] / cell) * 1000 + np.floor(p[:, 0] / cell)).astype(int))
    for p in pieces:
        q = np.clip(np.round(p['pts'][::4]).astype(int), 0, [la.shape[1] - 1, la.shape[0] - 1]) if la is not None else None
        strength = float(la[q[:, 1], q[:, 0]].mean()) if q is not None else 1.0
        p['imp'] = p['L'] * (0.3 + strength) * (1.0 if p['reg'] < 100000 else P['bg_weight'])
        p['cells'] = cells(p['pts'])
    used = {}; cap = P['cell_cap'] * budget / max(1, len({c for p in pieces for c in p['cells']}))
    keep = []; total = 0.0
    for p in sorted(pieces, key=lambda p: -p['imp']):
        load = np.mean([used.get(c, 0.0) for c in p['cells']])
        if load > cap and p['L'] < 0.25 * Wd: continue           # this area is already full (long silhouettes still pass)
        keep.append(p); total += p['L']
        share = p['L'] / max(1, len(p['cells']))
        for c in p['cells']: used[c] = used.get(c, 0.0) + share
        if total >= budget: break
    log('line budget: kept', len(keep), 'of', len(pieces), 'pieces,', round(total / Wd, 1), 'picture widths of line')
    return keep

def budget_edges(G):
    # same rule at the grain of single outline runs (junction to junction): strongest, longest runs first,
    # each area of the picture capped, stop at the budget; removed runs simply are not drawn
    la = SRC.get('la'); es = list(G.edges(keys=True, data=True))
    if not es: return
    allp = np.vstack([e['pts'] for *_, e in es]); Wd = max(allp[:, 0].max(), 1); Hd = max(allp[:, 1].max(), 1)
    budget = P['budget'] * Wd * float(np.clip(math.sqrt(SRC.get('subj_frac', 0.5) / 0.5), 0.6, 1.25)); cell = max(Wd, Hd) / P['budget_grid']
    LBL = OBJECTS.get('labels'); edge = None
    if LBL is not None:                                       # object silhouettes: where the label changes
        lb = LBL.astype(np.float32); edge = (cv2.dilate(lb, np.ones((3, 3))) != cv2.erode(lb, np.ones((3, 3))))
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    FM = OBJECTS.get('focus'); rows = []
    # laziness at selection time: candidate-line density, excluding the inner face/hands; dense filling gets fewer lines
    cand = np.zeros((int(Hd) + 2, int(Wd) + 2), np.float32)
    for *_, e in es: cv2.polylines(cand, [np.round(e['pts']).astype(np.int32)], False, 1.0, 2)
    cdens = cv2.blur(cand, (P['lazy_win'], P['lazy_win']))
    inner = np.zeros_like(cand)
    if OBJECTS.get('facehand') is not None:
        fh_ = OBJECTS['facehand'].astype(np.uint8); k_ = int(P['lazy_face_shrink'] * fh_.shape[1]) | 1
        fh_ = cv2.erode(fh_, np.ones((k_, k_), np.uint8)); hh, ww = min(fh_.shape[0], inner.shape[0]), min(fh_.shape[1], inner.shape[1])
        inner[:hh, :ww] = fh_[:hh, :ww]
    # every face and every hand gets its own detail allowance, proportional to its size (so faces match each other)
    zones = []
    for key_, mult in (('face_ids', P['face_ink']), ('hand_ids', P['hand_ink'])):
        ids = OBJECTS.get(key_)
        if ids is None: continue
        for z in range(1, int(ids.max()) + 1):
            area = float((ids == z).sum())
            frac = area / float(ids.size)
            if area > 0: zones.append((key_, z, mult * math.sqrt(area) * max(1.0, math.sqrt(frac) / P['closeup_ref'])))
    zone_used = {}
    for a, b, k, e in es:
        p = e['pts']; L = e['length']
        q = np.clip(np.round(p[::3]).astype(int), 0, [la.shape[1] - 1, la.shape[0] - 1]) if la is not None else None
        st = float(la[q[:, 1], q[:, 0]].mean()) if q is not None else 1.0
        imp = L ** 1.3 * (0.3 + st) * (1.0 if e.get('reg', 0) < 100000 else P['bg_weight'])   # long runs count more than their length
        if edge is not None:
            qq = np.clip(np.round(p[::3]).astype(int), 0, [edge.shape[1] - 1, edge.shape[0] - 1])
            imp *= 1 + P['sil_bonus'] * float(edge[qq[:, 1], qq[:, 0]].mean())                    # silhouettes first
        if SRC.get('major') is not None:                      # persistence across scales: major lines first
            MJ = SRC['major']; qm = np.clip(np.round(p[::3]).astype(int), 0, [MJ.shape[1] - 1, MJ.shape[0] - 1])
            imp *= P['major_floor'] + P['major'] * float(MJ[qm[:, 1], qm[:, 0]].mean())
        if P['lazy']:
            qd = np.clip(np.round(p[::3]).astype(int), 0, [cdens.shape[1] - 1, cdens.shape[0] - 1])
            dd = float(cdens[qd[:, 1], qd[:, 0]].mean()); inn = float(inner[qd[:, 1], qd[:, 0]].mean())
            imp *= 1.0 / (1.0 + P['lazy_k'] * max(dd - P['lazy_d0'], 0) / P['lazy_d0'] * (1 - inn))
        fo = 1.0
        if FM is not None:
            qf = np.clip(np.round(p[::3]).astype(int), 0, [FM.shape[1] - 1, FM.shape[0] - 1]); fo = float(FM[qf[:, 1], qf[:, 0]].mean())
        imp *= fo
        cs = np.unique((np.floor(p[:, 1] / cell) * 1000 + np.floor(p[:, 0] / cell)).astype(int))
        zk = None
        for key_, z, cap_ in zones:
            ids = OBJECTS[key_]; qz = np.clip(np.round(p[::3]).astype(int), 0, [ids.shape[1] - 1, ids.shape[0] - 1])
            if (ids[qz[:, 1], qz[:, 0]] == z).mean() > 0.5: zk = (key_, z, cap_); break
        qc = np.clip(np.round(p[::2]).astype(int), 0, [cdens.shape[1] - 1, cdens.shape[0] - 1])
        hc = (qc[:, 1] // P['dense_cell']) * 100000 + qc[:, 0] // P['dense_cell']
        hcells, hcount = np.unique(hc, return_counts=True); inn_ = float(inner[qc[:, 1], qc[:, 0]].mean())
        rows.append((imp, L, cs, (a, b, k), fo, zk, hcells, hcount * (L / max(1, len(qc))), inn_))
    ncell = len({c for r in rows for c in r[2]}); cap = P['cell_cap'] * budget / max(1, ncell)
    # detail-proportional allowance: each area's share of the budget follows how much line the photo has there
    # (compressed by a power < 1 so busy areas get more, but not everything)
    det = {}
    for r in rows:
        for c in r[2]: det[c] = det.get(c, 0.0) + r[1] / len(r[2])
    wsum = sum(v ** P['detail_pow'] for v in det.values()) or 1.0
    capc = {c: P['cell_cap'] * budget * v ** P['detail_pow'] / wsum for c, v in det.items()} if P['detail_prop'] else {}
    used = {}; total = 0.0; keep = set()
    dense_used = {}; dense_cap = P['dense_max'] * P['dense_cell'] ** 2 / 2.0; nskip = 0
    for imp, L, cs, key, fo, zk, hcells, hlen, inn_ in sorted(rows, key=lambda r: -r[0]):
        if total >= budget: break
        if inn_ < 0.5 and any(dense_used.get(c_, 0.0) + l_ > dense_cap for c_, l_ in zip(hcells, hlen)):   # filling patches are cut (eyes-nose-mouth exempt)
            nskip += 1; continue                                 # hard rule: this patch is already full
        if inn_ < 0.5:
            for c_, l_ in zip(hcells, hlen): dense_used[c_] = dense_used.get(c_, 0.0) + l_
        if zk is not None and zone_used.get(zk[:2], 0.0) + L > zk[2]: continue          # this face/hand has its share
        if zk is not None: zone_used[zk[:2]] = zone_used.get(zk[:2], 0.0) + L
        cmax = np.mean([capc.get(c, cap) for c in cs]) if capc else cap
        if np.mean([used.get(c, 0.0) for c in cs]) > cmax * fo and L < 0.2 * Wd: continue      # focus areas may hold more ink
        keep.add(key); total += L
        for c in cs: used[c] = used.get(c, 0.0) + L / len(cs)
    G.remove_edges_from([r[3] for r in rows if r[3] not in keep])
    G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
    log('density rule skipped', nskip, 'runs'); log('line budget: kept', len(keep), 'of', len(rows), 'outline runs,', round(total / Wd, 1), 'picture widths of line')

# ---------- 5b. route object by object ----------
def assign_sections(G):
    es = list(G.edges(keys=True, data=True)); LBL = OBJECTS.get('labels'); bg = []
    for i_, (_, _, _, e) in enumerate(es):
        lab = 0
        if LBL is not None:
            q = np.clip(np.round(e['pts'][::3]).astype(int), 0, [LBL.shape[1] - 1, LBL.shape[0] - 1])
            ls = LBL[q[:, 1], q[:, 0]]; ls = ls[ls > 0]
            if len(ls) * 2 >= max(1, len(q)): lab = int(np.bincount(ls).argmax())
        e['reg'] = lab
        if lab == 0: bg.append(i_)
    nbg = int(np.clip(len(bg) / 80, 1, P['regions'])) if bg else 0
    if nbg > 1:
        mids = np.array([es[i][3]['pts'][len(es[i][3]['pts']) // 2] for i in bg], np.float32)
        _, labels, _ = cv2.kmeans(mids, nbg, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5), 3, cv2.KMEANS_PP_CENTERS)
        for i, l in zip(bg, labels.ravel()): es[i][3]['reg'] = 100000 + int(l)
    elif nbg == 1:
        for i in bg: es[i][3]['reg'] = 100000

def route_by_object(gr):
    # each object (and background area) gets its own one-stroke route; routes are chained by thin travel lines,
    # always to the nearest unfinished piece, finishing an object's pieces before moving to the next object
    G = gr.G; assign_sections(G); budget_edges(G)
    pieces = []
    for reg in sorted({e['reg'] for _, _, _, e in G.edges(keys=True, data=True)}):
        H = G.edge_subgraph([(a, b, k) for a, b, k, e in G.edges(keys=True, data=True) if e['reg'] == reg]).copy()
        for comp in nx.connected_components(H):
            sub = H.subgraph(comp).copy()
            L = sum(e['length'] for _, _, _, e in sub.edges(keys=True, data=True))
            FM = OBJECTS.get('focus'); fo = 1.0
            if FM is not None:
                pp = np.vstack([e['pts'] for _, _, _, e in sub.edges(keys=True, data=True)])
                qf = np.clip(np.round(pp[::3]).astype(int), 0, [FM.shape[1] - 1, FM.shape[0] - 1]); fo = float(FM[qf[:, 1], qf[:, 0]].mean())
            if L < P['min_piece'] / fo: continue                    # small pieces survive in faces, hands and details
            pieces.append(dict(reg=reg, G=sub, L=L, pos=np.array([sub.nodes[n]['pos'] for n in sub.nodes]),
                               pts=np.vstack([e['pts'] for _, _, _, e in sub.edges(keys=True, data=True)])))
    nreg = len({p['reg'] for p in pieces})
    import walkjoin as WJ
    allpts = np.vstack([e['pts'] for _, _, _, e in G.edges(keys=True, data=True)])
    if OBJECTS.get('labels') is not None: Hh, Ww = OBJECTS['labels'].shape
    else: Ww, Hh = int(allpts[:, 0].max()) + 2, int(allpts[:, 1].max()) + 2
    inkmap = np.zeros((Hh, Ww), np.uint8)
    for _, _, _, e in G.edges(keys=True, data=True): cv2.polylines(inkmap, [np.round(e['pts']).astype(np.int32)], False, 1, 3)
    inkmap = inkmap > 0; rng_ = np.random.default_rng(P['seed']); nwalk = 0
    def heading(p, end):
        q = p[-1] - p[max(0, len(p) - 6)] if end else p[min(len(p) - 1, 5)] - p[0]
        return math.atan2(q[1], q[0])
    cur = np.array([0.0, 0.0]); cur_reg = None; path = []; prev_v = None; QUIET[0] = True; nsec = 0
    while pieces:
        same = [p for p in pieces if p['reg'] == cur_reg]
        p = min(same or pieces, key=lambda p: float(np.min(np.hypot(*(p['pos'] - cur).T))))
        pieces.remove(p); cur_reg = p['reg']
        sub = Graph(); sub.G = p['G']; sub.nid = gr.nid
        eulerize(sub); gr.nid = sub.nid
        seg = traverse(sub, start_pt=cur, sections=False)
        if not seg: continue
        for e_ in seg: e_['sec'] = nsec
        nsec += 1
        if prev_v is not None:
            a_, b_ = cur, sub.G.nodes[seg[0]['u']]['pos']
            near = float(np.hypot(*(np.asarray(b_) - np.asarray(a_)))) < P['hop'] * Ww    # small gaps: a short direct hop
            wj = WJ.join(a_, b_, heading(path[-1]['pts'], True), heading(seg[0]['pts'], False), (Ww, Hh), inkmap, rng_) if (P['walk'] and not near) else None
            if wj is not None:
                path.append(dict(u=prev_v, v=seg[0]['u'], pts=wj[0], level=1, kind='walk', base=None, sec=nsec - 1)); nwalk += 1
            else:
                path.append(dict(u=prev_v, v=seg[0]['u'], pts=soft_curve(a_, b_, heading(path[-1]['pts'], True), heading(seg[0]['pts'], False), Ww, Hh), level=P['n'], kind='aux', base=None, sec=nsec - 1))
        path += seg; prev_v = seg[-1]['v']; cur = sub.G.nodes[prev_v]['pos']
    QUIET[0] = False
    log('route: sections', nreg, '| joins drawn by the walk', nwalk, 'straight', sum(1 for e in path if e['kind'] == 'aux' and e['base'] is None))
    return path

def lazy_fill(pts, H, W):
    # be lazy on dense filling: where the line piles up (hair, fabric, texture) and it is not a face, a hand or the
    # outer outline, loosen it into a quick scribble (heavier smoothing); returns per-point laziness for the pen speed
    if not P['lazy'] or len(pts) < 50: return pts, np.zeros(len(pts))
    ink = np.zeros((H, W), np.float32)
    cv2.polylines(ink, [np.round(pts).astype(np.int32)], False, 1.0, 2)
    dens = cv2.blur(ink, (P['lazy_win'], P['lazy_win']))
    prot = np.zeros((H, W), np.float32)
    if OBJECTS.get('facehand') is not None:                   # only the inner face and hand (hair around the face may be lazy)
        k_ = int(P['lazy_face_shrink'] * W) | 1
        prot = np.maximum(prot, cv2.erode(OBJECTS['facehand'].astype(np.uint8), np.ones((k_, k_), np.uint8)).astype(np.float32))
    if SRC.get('matte') is not None:                          # outer outline of the subject
        m = cv2.resize((SRC['matte'] > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        prot = np.maximum(prot, cv2.dilate(edge, np.ones((int(0.02 * W) | 1,) * 2, np.uint8)).astype(np.float32))
    prot = cv2.GaussianBlur(prot, (0, 0), 4)
    q = np.clip(np.round(pts).astype(int), 0, [W - 1, H - 1])
    lz = np.clip((dens[q[:, 1], q[:, 0]] - P['lazy_d0']) / P['lazy_d0'], 0, 1) * (1 - np.clip(prot[q[:, 1], q[:, 0]], 0, 1))
    lz = cv2.GaussianBlur(lz.reshape(-1, 1).astype(np.float32), (1, 0), 15).ravel()
    sm = pts.copy()
    for c in (0, 1): sm[:, c] = cv2.GaussianBlur(pts[:, c].reshape(-1, 1).astype(np.float32), (1, 0), P['lazy_sigma'], borderType=cv2.BORDER_REPLICATE).ravel()
    out = pts * (1 - lz[:, None]) + sm * lz[:, None]
    log('lazy filling: loosened', round(float((lz > 0.3).mean()) * 100), '% of the line')
    return out, lz

def soft_curve(a, b, din, dout, W, H):
    # gentle curved hop: leaves along the incoming direction, arrives along the outgoing one, kept inside the picture
    a, b = np.asarray(a, float), np.asarray(b, float); d = float(np.hypot(*(b - a)))
    k = 0.35 * d
    c1 = a + k * np.array([math.cos(din), math.sin(din)]); c2 = b - k * np.array([math.cos(dout), math.sin(dout)])
    t = np.linspace(0, 1, max(8, int(d / 2)))[:, None]
    p = (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * b
    m = 0.02 * min(W, H); p[:, 0] = np.clip(p[:, 0], m, W - m); p[:, 1] = np.clip(p[:, 1], m, H - m)
    return p

def add_signature(path, W, H):
    # the line swoops down (walk arc) and writes the signature bottom-right, as its own last section
    import walkjoin as WJ
    last = path[-1]; sec = max(e.get('sec', 0) for e in path) + 1
    if P['sign'] == 'eye':                                   # the mark: one-stroke eye, bottom-right
        import eye as EY
        sig = EY.eye_at(W, H, [e['pts'] for e in path], P['eye_style'], avoid=((OBJECTS['labels'] > 0) if OBJECTS.get('labels') is not None else np.zeros((int(H), int(W)), bool)) | cv2.resize(person_mask(SRC['img']).astype(np.uint8), (int(W), int(H)), interpolation=cv2.INTER_NEAREST).astype(bool)); spath = [dict(u=-1, v=-2, pts=sig, level=1, kind='sig', base=None)]
    elif P['sign_font'].endswith('.ttf'):                    # handwriting font: rendered bold, centre line traced
        import sigtrace as STR
        font = P['sign_font'] if os.path.isabs(P['sign_font']) else os.path.join(os.path.dirname(os.path.abspath(__file__)), P['sign_font'])
        spath = STR.signature(P['sign'], font, W, H, P['sign_h'])
        sig = spath[0]['pts']
    else:                                                    # single-stroke vector script
        import signature as SG
        sig = SG.signature(P['sign'], W, H, P['sign_font']); spath = [dict(u=-1, v=-2, pts=sig, level=1, kind='sig', base=None)]
    q = last['pts']; din = math.atan2(*(q[-1] - q[max(0, len(q) - 6)])[::-1]); dout = math.atan2(*(sig[3] - sig[0])[::-1])
    wj = WJ.join(q[-1], sig[0], din, dout, (W, H), None, np.random.default_rng(P['seed'] + 7))
    jp = wj[0] if wj is not None else soft_curve(q[-1], sig[0], din, dout, W, H)
    path.append(dict(u=last['v'], v=-1, pts=jp, level=1, kind='walk', base=None, sec=sec))
    for e in spath:
        e = dict(e); e['kind'] = 'aux' if e['kind'] == 'aux' else 'sig'; e['level'] = 1; e['sec'] = sec
        if e['kind'] == 'aux': e['kind'] = 'sig'                 # hops between letters are part of the signature
        path.append(e)
    log('signature', repr(P['sign']), 'added,', len(spath), 'strokes')

# ---------- 6. level of detail ----------
def sweep(pts):
    # connector drawn as a smooth curve through its simplified corners, not a zigzag
    if len(pts) < 3: return resample(pts, 2.0)
    q = np.vstack([pts[0], pts, pts[-1]]); seg = []
    for k in range(1, len(q) - 2):
        n_ = max(3, int(np.hypot(*(q[k + 1] - q[k])) / 2)); seg.append(catmull(q[k - 1], q[k], q[k + 1], q[k + 2], n_)[:-1])
    seg.append(pts[-1:]); return np.vstack(seg)

def lod(path, G0, d):
    out = []; i = 0
    while i < len(path):
        e = path[i]
        hidden = e['kind'] == 'aux' or e['level'] > d
        if not hidden: out.append(dict(e)); i += 1; continue
        j = i
        while j < len(path) and (path[j]['kind'] == 'aux' or path[j]['level'] > d): j += 1
        run = path[i:j]; a = run[0]['u']; b = run[-1]['v']
        verts = set([a, b]) | {r['v'] for r in run}
        sub = G0.subgraph(verts)
        try:
            sp = nx.shortest_path(sub, a, b, weight='length')
            pts = [G0.nodes[a]['pos']]
            for x, y in zip(sp[:-1], sp[1:]):
                es = min(G0[x][y].values(), key=lambda e: e['length']); pts += list(oriented(es, x)[1:])
            pts = np.array(pts)
        except nx.NetworkXNoPath:
            pts = np.vstack([r['pts'] for r in run])
        if a == b and arclen(pts) < 1: i = j; continue
        pts = approximate_polygon(pts, tolerance=18) if len(pts) > 2 else pts
        out.append(dict(u=a, v=b, pts=sweep(pts), level=0, kind='aux', base=None, sec=run[0].get('sec', 0)))
        i = j
    return out

# ---------- 7. trim, join with Catmull-Rom, curls, render ----------
def catmull(p0, p1, p2, p3, n):
    t = np.linspace(0, 1, n)[:, None]
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)

def trim(p, a, b):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]; L = d[-1]
    if L - a - b < 2: m = p[np.searchsorted(d, L / 2)] if len(p) else p; return np.array([m, m])
    t = np.linspace(a, L - b, max(2, int((L - a - b) / 1.0)))
    return np.c_[np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])]

def jitter(p, rng, amp=2.0):
    if len(p) < 3: return p
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]; L = d[-1] or 1; s = d / L
    tg = np.gradient(p, axis=0); nrm = np.c_[-tg[:, 1], tg[:, 0]]; nrm /= np.hypot(*nrm.T)[:, None] + 1e-9
    a1, a2, ph = rng.uniform(-amp, amp), rng.uniform(-amp / 2, amp / 2), rng.uniform(0, 6.28)
    off = np.sin(np.pi * s) * (a1 + a2 * np.sin(2 * np.pi * s * 2 + ph))
    return p + nrm * off[:, None]

ASSEMBLED = {}
def assemble(path, seed=0):
    rng = np.random.default_rng(seed)
    items = []
    for e in path:
        p = e['pts']; L = arclen(p)
        dt = 10.0 if e['kind'] == 'aux' else min(P['lam'], P['gamma'] * L)
        items.append(dict(p=jitter(p, rng) if e['kind'] == 'dup' else p, dt=dt, kind=e['kind'], base=e['base'], L=L, lev=e['level'], sec=e.get('sec', 0)))
    n = len(items); out = []; kinds = []; secs = []
    for i, it in enumerate(items):
        ta = 0 if i == 0 else min(items[i - 1]['dt'], it['dt'])
        tb = 0 if i == n - 1 else min(it['dt'], items[i + 1]['dt'])
        if it['kind'] == 'aux': ta = tb = 0
        it['q'] = trim(it['p'], min(ta, it['L'] * 0.45), min(tb, it['L'] * 0.45))
    for i, it in enumerate(items):
        q = it['q']; out.append(q); kinds += [it['kind'] if it['kind'] in ('aux', 'walk', 'sig') else 'L%d' % it['lev']] * len(q); secs += [it['sec']] * len(q)
        if i == n - 1: break
        nx_ = items[i + 1]['q']
        A, B = q[-1], nx_[0]
        p0 = q[-2] if len(q) > 1 else A; p3 = nx_[1] if len(nx_) > 1 else B
        retrace = it['base'] is not None and it['base'] == items[i + 1]['base']
        if retrace:
            prv = items[i - 1]['q'] if i > 0 else q; nxt = items[i + 2]['q'] if i + 2 < n else nx_
            o1 = prv[-1] - prv[0]; o2 = nxt[-1] - nxt[0]
            th = math.acos(min(1, abs(np.dot(o1, o2)) / (np.hypot(*o1) * np.hypot(*o2) + 1e-9)))
            tip = it['p'][-1]; tdir = tip - A; tl = np.hypot(*tdir) or 1; tdir /= tl; nrm = np.array([-tdir[1], tdir[0]])
            if th > math.pi / 4:   # curl: a small loop past the tip
                r = max(6.0, min(10, tl * 1.2))
                ctrl = [p0, A, tip + nrm * r * 0.6, tip + tdir * r * 1.1, tip - nrm * r * 0.6, B, p3]
            else:                  # cusp: run into the tip and back
                ctrl = [p0, A, tip, B, p3]
            seg = []
            for k in range(1, len(ctrl) - 2): seg.append(catmull(ctrl[k - 1], ctrl[k], ctrl[k + 1], ctrl[k + 2], 12))
            seg = np.vstack(seg)
        else:
            gap = np.hypot(*(B - A)); seg = catmull(p0, A, B, p3, max(3, int(gap)))
        out.append(seg[1:-1]); kinds += ['join'] * max(0, len(seg) - 2); secs += [items[i + 1]['sec']] * max(0, len(seg) - 2)
    pts = np.vstack(out)
    if len(pts) > 9:
        for c in (0, 1): pts[:, c] = cv2.GaussianBlur(pts[:, c].reshape(-1, 1).astype(np.float32), (1, 0), 2.4, borderType=cv2.BORDER_REPLICATE).ravel()
    ASSEMBLED['secs'] = np.array(secs, np.int32)
    return pts, kinds

def widths(pts, kinds):
    LW = {'L1': 1.25, 'L2': 0.95, 'L3': 0.7, 'aux': 0.4, 'walk': 0.38, 'sig': 0.9, 'join': None, 'dup': 0.8}
    pr = np.array([LW.get(k) if LW.get(k) is not None else np.nan for k in kinds], float)
    # joins take the weight of their neighbours
    idx = np.arange(len(pr)); ok = ~np.isnan(pr)
    pr = np.interp(idx, idx[ok], pr[ok]) if ok.any() else np.ones(len(pr))
    if 'dt' in STROKE:
        ix = np.clip(np.round(pts).astype(int), 0, [STROKE['dt'].shape[1] - 1, STROKE['dt'].shape[0] - 1])
        dtv = cv2.GaussianBlur(STROKE['dt'], (0, 0), 1.5)[ix[:, 1], ix[:, 0]]
        pr = np.maximum(pr, np.where(dtv > 0.8, np.minimum(1.0 + 0.32 * dtv, 2.6), 0))
    # speed from curvature: slow (thick) in tight bends, fast (thin) on straights
    tg = np.gradient(pts, axis=0); ang = np.unwrap(np.arctan2(tg[:, 1], tg[:, 0]))
    ds = np.hypot(*tg.T) + 1e-6; curv = np.abs(np.gradient(ang)) / ds
    curv = cv2.GaussianBlur(curv.reshape(-1, 1).astype(np.float32), (1, 0), 6).ravel() if len(curv) > 5 else curv
    slow = np.clip(curv * 6, 0, 1)
    r = P['R'] * pr * (0.88 + 0.22 * slow)
    r = cv2.GaussianBlur(r.reshape(-1, 1).astype(np.float32), (1, 0), 16).ravel() if len(r) > 5 else r
    if P['var'] > 0 and len(r) > 20:                          # hand pressure: slow, smooth swells and thinnings along the line
        rng_ = np.random.default_rng(P['seed'] + 11); nz = rng_.normal(0, 1, len(r)).astype(np.float32)
        nz = cv2.GaussianBlur(nz.reshape(-1, 1), (1, 0), 45).ravel(); nz /= (np.abs(nz).max() + 1e-6)
        r = r * (1 + P['var'] * nz)
    nT = min(40, len(r) // 4)
    if nT > 1: r[:nT] *= np.linspace(0.2, 1, nT); r[-nT:] *= np.linspace(1, 0.2, nT)
    return np.maximum(r, 0.05)

def draw_nib(img, pts_px, half_px, ang_deg, min_frac):
    # broad flat nib held at a fixed angle: each step sweeps the nib edge, so width follows stroke direction
    # (strokes across the nib are fat, strokes along it are hairlines) and ends are cut at the nib angle
    a = math.radians(ang_deg); nv = np.array([math.cos(a), math.sin(a)])
    P16 = np.round(pts_px * 16).astype(np.int32); H16 = (half_px[:, None] * nv[None, :] * 16)
    for k in range(1, len(pts_px)):
        p0, p1 = pts_px[k - 1] * 16, pts_px[k] * 16
        quad = np.round(np.array([p0 + H16[k - 1], p1 + H16[k], p1 - H16[k], p0 - H16[k - 1]])).astype(np.int32)
        cv2.fillConvexPoly(img, quad, 0, cv2.LINE_AA, 4)
        cv2.line(img, tuple(P16[k - 1]), tuple(P16[k]), 0, max(1, int(round(2 * half_px[k] * min_frac))), cv2.LINE_AA, 4)

def render(pts, kinds, H, W, path_png, bg=None, scale=1.0):
    ss = P['ss']; S = scale * ss
    canvas = np.full((int(H * S), int(W * S)), 255, np.uint8) if bg is None else None
    r = widths(pts, kinds)
    img = np.full((int(H * S), int(W * S)), 255, np.uint8)
    P2 = (pts * S * 16).astype(np.int32)
    draw_nib(img, pts * S, r * S * P['nib_scale'], P['nib_ang'], P['nib_min'])
    img = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    if bg is not None:
        g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY); g = cv2.resize(g, (img.shape[1], img.shape[0]))
        faded = (255 - (255 - g.astype(np.float32)) * 0.35)
        img = (faded * img.astype(np.float32) / 255).astype(np.uint8)
    cv2.imwrite(path_png, img)
    return img

if __name__ == '__main__':
    src, outdir = sys.argv[1], sys.argv[2]; Wd = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    for kv in sys.argv[4:]:
        k_, v_ = kv.split('=', 1)
        P[k_] = v_ if isinstance(P[k_], str) else (v_ in ('1', 'true', 'True') if isinstance(P[k_], bool) else type(P[k_])(float(v_)))
    os.makedirs(outdir, exist_ok=True)
    img = cv2.imread(src)
    if P['upscale'] and max(img.shape[:2]) < P['upscale_below']: img = upscale(img)
    if P['heq']:                                              # histogram-equalise brightness (colour kept)
        lab_ = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); lab_[:, :, 0] = cv2.equalizeHist(lab_[:, :, 0]); img = cv2.cvtColor(lab_, cv2.COLOR_LAB2BGR)
    SRC['full'] = img; h, w = img.shape[:2]; img = cv2.resize(img, (Wd, int(h * Wd / w)), interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    OBJECTS['labels'] = segment_objects(img); SRC['img'] = img
    if P['prep']: SRC['full'] = prep_for_lines(SRC['full'])   # line models see the cleaned photo; objects/pose used the original
    OBJECTS['focus'] = focus_map(img) if P['focus'] else None
    if P['enhance'] and P['focus']:
        fh = _cache('facehand2', img, _focus_map)
        SRC['full'] = enhance(SRC['full'], fh[0] > 0, fh[1] > 0); cv2.imwrite(os.path.join(outdir, 'enhanced.jpg'), SRC['full'])
    def trace_once():
        gr_ = build(img); connect(gr_, join=False); G0_ = gr_.G.copy(); path_ = route_by_object(gr_)
        drawn = sum(arclen(e['pts']) for e in path_ if e['kind'] not in ('aux', 'walk')) / W
        return gr_, G0_, path_, drawn
    # equal detail for every photo: let in fine detail outside faces until the drawing reaches the target amount of line
    target = P['target'] * float(np.clip(math.sqrt(SRC.get('subj_frac', 0.5) / 0.5), 0.7, 1.2))
    lo, hi = 0.0, 1.0; P['fine_rest'] = 0.0; QUIET[0] = True
    gr, G0, path, drawn = trace_once(); best = (abs(drawn - target), 0.0, gr, G0, path, drawn)
    if drawn < target:
        for _ in range(P['target_iters']):
            P['fine_rest'] = (lo + hi) / 2; r_ = trace_once()
            if abs(r_[3] - target) < best[0]: best = (abs(r_[3] - target), P['fine_rest']) + r_
            if r_[3] < target: lo = P['fine_rest']
            else: hi = P['fine_rest']
    QUIET[0] = False
    _, P['fine_rest'], gr, G0, path, drawn = best
    log(f'detail target {target:.1f} picture widths of drawn line -> {drawn:.1f} (fine detail outside faces {P["fine_rest"]:.2f})')
    if P['sign']: add_signature(path, W, H)
    res = {}
    for d in range(1, P['n'] + 1):
        pl = lod(path, G0, d)
        pts, kinds = assemble(pl, seed=d)
        pts, lazy = lazy_fill(pts, H, W)
        render(pts, kinds, H, W, f'{outdir}/line_d{d}.png', scale=1.5)
        res[d] = dict(points=len(pts), length=round(arclen(pts)))
        np.save(f'{outdir}/path_d{d}.npy', pts * (500.0 / W)); json.dump(kinds, open(f'{outdir}/kinds_d{d}.json', 'w')); np.save(f'{outdir}/lazy_d{d}.npy', lazy.astype(np.float32)); np.save(f'{outdir}/secs_d{d}.npy', ASSEMBLED['secs'])
        log('detail', d, res[d])
    pts, kinds = assemble(lod(path, G0, 2), seed=2)
    render(pts, kinds, H, W, f'{outdir}/overlay_d2.png', bg=img, scale=1.5)
    log('done', res)
