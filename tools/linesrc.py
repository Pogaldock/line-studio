# Round 1 of the line-source test: the same pictures through many line extractors, raw line maps side by side
# (black lines on white). Shows which sources see the structure (outlines, eyes, mouth) and which see the shading.
# Usage: python tools/linesrc.py <out.jpg> <photo.jpg> [<photo.jpg> ...]
import sys, os, time, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'engine'))
import linephoto as LP
from PIL import Image
def log(*a): print('[linesrc]', *a, flush=True)
R = 640
def pil(img): return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
def gray(out, white_lines):
    a = np.array(out.convert('L') if hasattr(out, 'convert') else out).astype(np.float32) / 255
    return 1 - a if white_lines else a                      # -> 1 = paper, 0 = line
_M = {}
def cna(name):
    if name in _M: return _M[name]
    import controlnet_aux as C
    rep = 'lllyasviel/Annotators'
    m = {'lineart': lambda: C.LineartDetector.from_pretrained(rep), 'anime': lambda: C.LineartAnimeDetector.from_pretrained(rep),
         'hed': lambda: C.HEDdetector.from_pretrained(rep), 'pidi': lambda: C.PidiNetDetector.from_pretrained(rep),
         'teed': lambda: C.TEEDdetector.from_pretrained('fal-ai/teed', filename='5_model.pth'),
         'anyline': lambda: C.AnylineDetector.from_pretrained('TheMistoAI/MistoLine', filename='MTEED.pth', subfolder='Anyline')}[name]()
    _M[name] = m; return m
def xdog(img, sigma=0.8, k=1.6, p=20, eps=0.01, phi=10):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    a = cv2.GaussianBlur(g, (0, 0), sigma); b = cv2.GaussianBlur(g, (0, 0), sigma * k); d = (1 + p) * a - p * b
    return np.where(d >= eps, 1.0, 1 + np.tanh(phi * (d - eps))).clip(0, 1)
def smooth_canny(img):
    s = img.copy()
    for _ in range(6): s = cv2.bilateralFilter(s, 9, 60, 9)            # flatten texture, keep big edges
    e = cv2.Canny(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY), 60, 140)
    return 1 - cv2.dilate(e, np.ones((2, 2), np.uint8)).astype(np.float32) / 255
SOURCES = [
    ('now: contour model', lambda im: 1 - LP._contour(im)),
    ('lineart (photo)', lambda im: gray(cna('lineart')(pil(im), detect_resolution=R, image_resolution=R), True)),
    ('lineart coarse', lambda im: gray(cna('lineart')(pil(im), coarse=True, detect_resolution=R, image_resolution=R), True)),
    ('lineart anime', lambda im: gray(cna('anime')(pil(im), detect_resolution=R, image_resolution=R), True)),
    ('HED soft edges', lambda im: gray(cna('hed')(pil(im), detect_resolution=R, image_resolution=R), True)),
    ('PiDiNet', lambda im: gray(cna('pidi')(pil(im), detect_resolution=R, image_resolution=R), True)),
    ('TEED', lambda im: gray(cna('teed')(pil(im), detect_resolution=R), True)),
    ('Anyline', lambda im: gray(cna('anyline')(pil(im), detect_resolution=R), True)),
    ('XDoG (no model)', xdog),
    ('flattened + Canny', smooth_canny),
]
rows = []
for f in sys.argv[2:]:
    im = cv2.imread(f); h, w = im.shape[:2]; s = 512 / max(h, w); im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    LP.SRC.clear(); LP.SRC['full'] = im
    tiles = [im]
    for name, fn in SOURCES:
        t0 = time.time()
        try:
            m = fn(im); m = cv2.resize(np.asarray(m, np.float32), (im.shape[1], im.shape[0]))
            t = cv2.cvtColor((m * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        except Exception as e:
            log(name, 'failed:', type(e).__name__, str(e)[:200]); t = np.full_like(im, 230)
        log(os.path.basename(f), name, f'{time.time() - t0:.1f}s')
        tiles.append(t)
    rows.append(tiles)
W = 300
def lab(a, text):
    a = cv2.resize(a, (W, int(a.shape[0] * W / a.shape[1]))); a = a[:W] if a.shape[0] >= W else np.vstack([a, np.full((W - a.shape[0], W, 3), 255, np.uint8)])
    if text: cv2.rectangle(a, (0, 0), (W, 22), (0, 0, 0), -1); cv2.putText(a, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return a
names = ['photo'] + [n for n, _ in SOURCES]
sheet = np.vstack([np.hstack([lab(t, names[i] if r == 0 else '') for i, t in enumerate(row)]) for r, row in enumerate(rows)])
cv2.imwrite(sys.argv[1], sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]); log('wrote', sys.argv[1], sheet.shape)
