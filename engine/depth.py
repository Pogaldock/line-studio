# Depth-aware line selection and weight (experiment, Nov 2026). Turned on with depth=1 in an option's style.
#
# Depth Anything V2 Small (Apache-2.0) gives a relative depth map of the photo (larger = nearer). From it:
#   depth edges   where depth jumps (object against background, arm in front of body): the lines that carry the shape
#   selection     strokes lying on depth edges are kept down to depth_keep x the usual minimum length; strokes on
#                 flat, unbroken surfaces (prints, texture) need depth_flat x the length. The total amount of line stays
#                 the same (the style's own budget search still runs over this selection).
#   shape         strokes with more than depth_shape_cov of their length on depth edges get depth_flow x the style's smoothing, so small shapes on an object's
#                 outline (a far wheel, a fingertip) survive the loose flowing look instead of being rounded away
#   gaps          (depth_gap=1) outlines traced straight from the depth map wherever no kept line already covers them
#                 (a far wheel sticking out of a board's silhouette): runs at least depth_gap_min x the minimum length
#   relief        (depth_relief=1) depth change measured against the strongest change in its own area (~1/12 of the
#                 frame) instead of the whole photo: soft 3D form (side of a nose, cheekbone, lips) becomes lines, and
#                 those count as outlines in the selection; traced strokes are added to the line model's candidates
#   weight        line width x (1 - depth_w/2 + depth_w x nearness): nearer is thicker, further is thinner
# Model file: models/depth_anything_v2_small.onnx (onnx-community export, sha256 in models/models.txt).
import os, numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
D = dict(depth=0, depth_keep=0.4, depth_flat=1.4, depth_w=0.8, depth_edge=0.08, depth_flow=0.35, depth_shape_cov=0.5, depth_gap=0, depth_gap_min=0.15, depth_gap_canny=12, depth_relief=0, relief_hi=0.4, relief_lo=0.2)
_CACHE = {}

def depth_map(img):
    # relative inverse depth, 0 (far) .. 1 (near), same size as img
    key = (img.shape, hash(img[::17, ::17].tobytes()))
    if key in _CACHE: return _CACHE[key]
    if 'sess' not in _CACHE:                                  # model loaded once per process (a live keeps it warm)
        import onnxruntime as ort
        f = os.path.join(HERE, 'models', 'dl', 'depth_anything_v2_small.onnx')
        if not os.path.exists(f): f = os.path.join(HERE, 'models', 'depth_anything_v2_small.onnx')
        _CACHE['sess'] = ort.InferenceSession(f, providers=['CPUExecutionProvider'])
    s = _CACHE['sess']
    h, w = img.shape[:2]; sc = 518 / min(h, w)
    H14, W14 = max(14, int(round(h * sc / 14)) * 14), max(14, int(round(w * sc / 14)) * 14)
    x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (W14, H14), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
    x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).astype(np.float32).transpose(2, 0, 1)[None]
    d = s.run(None, {s.get_inputs()[0].name: x})[0][0]
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
    lo, hi = np.percentile(d, 1), np.percentile(d, 99); d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
    _CACHE[key] = d; return d

def depth_edges(d):
    # relative depth jumps: gradient of the (lightly smoothed) depth, divided by how near it is, top share kept
    g = cv2.GaussianBlur(d, (0, 0), 1.5)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = np.hypot(gx, gy) / (g + 0.15)
    t = np.quantile(m, 1 - D['depth_edge'])
    e = (m >= t).astype(np.uint8)
    return cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0

def depth_curves(d, canny):
    # thin outline curves of the depth map itself (Canny on the depth image), as a 0/1 map for the stroke tracer
    g = (cv2.GaussianBlur(d, (0, 0), 1.0) * 255).astype(np.uint8)
    e = cv2.Canny(g, canny, canny * 2.5)                     # 3 px wide for the shared tracer (see relief_lines)
    return (cv2.dilate(e, np.ones((3, 3), np.uint8)) > 0).astype(np.float32)

def relief(d):
    # relative depth change: each point against the strongest change around it, so a nose counts inside a face even
    # though the jaw against the background is a far bigger jump. Returns the 0..1 map.
    h, w = d.shape; gd = cv2.GaussianBlur(d, (0, 0), 2)
    dg = np.hypot(cv2.Sobel(gd, cv2.CV_32F, 1, 0), cv2.Sobel(gd, cv2.CV_32F, 0, 1))
    k = int(max(h, w) / 12) | 1
    locmax = cv2.dilate(dg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return np.clip(dg / (cv2.GaussianBlur(locmax, (0, 0), k / 4) + 0.02 * dg.max()), 0, 1.5)

def relief_lines(d):
    # Canny-style on the relative relief: keep only the crest of each ridge (a pixel stays if it is stronger than its
    # two neighbours across the edge, i.e. along the depth gradient), then hysteresis (weak crest kept where connected to
    # strong crest). Thin, single lines by construction: no blobs, no junction noise. 0/1 float map.
    gd = cv2.GaussianBlur(d, (0, 0), 2); gx, gy = cv2.Sobel(gd, cv2.CV_32F, 1, 0), cv2.Sobel(gd, cv2.CV_32F, 0, 1)
    r = relief(d); h, w = r.shape
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180; q = np.zeros(r.shape, np.int8)
    q[(ang >= 22.5) & (ang < 67.5)] = 1; q[(ang >= 67.5) & (ang < 112.5)] = 2; q[(ang >= 112.5) & (ang < 157.5)] = 3
    p = np.pad(r, 1, mode='edge'); c = p[1:-1, 1:-1]
    nbr = {0: (p[1:-1, 2:], p[1:-1, :-2]), 1: (p[2:, 2:], p[:-2, :-2]), 2: (p[2:, 1:-1], p[:-2, 1:-1]), 3: (p[2:, :-2], p[:-2, 2:])}
    crest = np.zeros(r.shape, bool)
    for k, (a1, a2) in nbr.items(): crest |= (q == k) & (c >= a1) & (c >= a2)
    weak = crest & (r > D['relief_lo']); strong = crest & (r > D['relief_hi'])
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool); keep[np.unique(lab[strong])] = True; keep[0] = False
    from skimage.morphology import thin
    # returned 3 px wide: the shared stroke tracer drops 4-connected specks before thinning, which would cut a 1 px
    # diagonal line into pieces; at 3 px it stays one piece and the tracer thins it back to its centre
    return cv2.dilate(thin(keep[lab]).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
