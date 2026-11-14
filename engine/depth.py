# Depth-aware line selection and weight (experiment, Nov 2026). Turned on with depth=1 in an option's style.
#
# Depth Anything V2 Small (Apache-2.0) gives a relative depth map of the photo (larger = nearer). From it:
#   depth edges   where depth jumps (object against background, arm in front of body): the lines that carry the shape
#   selection     strokes lying on depth edges are kept down to depth_keep x the usual minimum length; strokes on
#                 flat, unbroken surfaces (prints, texture) need depth_flat x the length. The total amount of line stays
#                 the same (the style's own budget search still runs over this selection).
#   weight        line width x (1 - depth_w/2 + depth_w x nearness): nearer is thicker, further is thinner
# Model file: models/depth_anything_v2_small.onnx (onnx-community export, sha256 in models/models.txt).
import os, numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
D = dict(depth=0, depth_keep=0.4, depth_flat=1.4, depth_w=0.8, depth_edge=0.08)
_CACHE = {}

def depth_map(img):
    # relative inverse depth, 0 (far) .. 1 (near), same size as img
    key = (img.shape, hash(img[::17, ::17].tobytes()))
    if key in _CACHE: return _CACHE[key]
    import onnxruntime as ort
    f = os.path.join(HERE, 'models', 'dl', 'depth_anything_v2_small.onnx')
    if not os.path.exists(f): f = os.path.join(HERE, 'models', 'depth_anything_v2_small.onnx')
    s = ort.InferenceSession(f, providers=['CPUExecutionProvider'])
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
