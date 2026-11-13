# Signature mark: a simple eye drawn as one continuous stroke (lids, then the iris, ending in a small spiral pupil).
import numpy as np

def _bez(p0, p1, p2, p3, n=60):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3

def eye_unit(style='A'):
    """Eye in a unit box (corners at x=-1 and x=+1), one continuous polyline."""
    L, R = np.array([-1.0, 0.0]), np.array([1.0, 0.0])
    if style == 'A':      # almond, iris circle, spiral pupil
        up = _bez(L, [-0.45, -0.85], [0.45, -0.85], R)
        lo = _bez(R, [0.45, 0.55], [-0.45, 0.55], L + [0.08, 0.0])
        a = np.linspace(np.pi * 0.95, np.pi * 0.95 + 2 * np.pi * 1.05, 120); ir = np.c_[0.36 * np.cos(a), -0.06 + 0.36 * np.sin(a)]
        b = np.linspace(a[-1], a[-1] + 2 * np.pi * 1.6, 90); rr = np.linspace(0.36, 0.05, len(b)); pu = np.c_[rr * np.cos(b), -0.06 + rr * np.sin(b)]
        link = _bez(lo[-1], lo[-1] + [0.25, -0.05], ir[0] + [-0.1, 0.1], ir[0], 20)
        return np.vstack([up, lo, link, ir, pu])
    if style == 'B':      # overshooting corners, iris tucked under the upper lid, dot pupil curl
        up = _bez(L + [-0.18, 0.06], [-0.4, -0.95], [0.5, -0.8], R + [0.2, 0.08])
        lo = _bez(R + [0.2, 0.08], [0.4, 0.62], [-0.5, 0.5], L + [-0.05, -0.04])
        a = np.linspace(np.pi * 1.15, np.pi * 1.15 + 2 * np.pi * 0.85, 100); ir = np.c_[0.4 * np.cos(a), -0.12 + 0.4 * np.sin(a)]
        b = np.linspace(a[-1], a[-1] + 2 * np.pi * 2.2, 100); rr = np.linspace(0.13, 0.02, len(b)); pu = np.c_[rr * np.cos(b), -0.12 + rr * np.sin(b)]
        link = _bez(lo[-1], lo[-1] + [0.3, -0.1], ir[0] + [-0.1, 0.15], ir[0], 20)
        link2 = _bez(ir[-1], ir[-1] + [0.05, 0.1], pu[0] + [0.1, 0.1], pu[0], 16)
        return np.vstack([up, lo, link, ir, link2, pu])
    # 'C': minimal: almond whose lower lid curls straight into a spiral iris
    up = _bez(L, [-0.4, -0.8], [0.4, -0.8], R)
    lo = _bez(R, [0.5, 0.5], [-0.1, 0.5], [-0.3, 0.05])
    b = np.linspace(np.pi, np.pi + 2 * np.pi * 2.3, 200); rr = np.linspace(0.33, 0.04, len(b)); sp = np.c_[0.02 + rr * np.cos(b), -0.05 + rr * np.sin(b)]
    link = _bez(lo[-1], lo[-1] + [0.0, -0.1], sp[0] + [-0.05, 0.1], sp[0], 14)
    return np.vstack([up, lo, link, sp])

def eye(W, H, style='A', width_frac=0.16, margin=0.05):
    """Eye polyline in picture coordinates, bottom-right corner."""
    p = eye_unit(style); lo, hi = p.min(0), p.max(0)
    s = width_frac * W / (hi[0] - lo[0]); p = (p - lo) * s
    return p + [W * (1 - margin) - (hi[0] - lo[0]) * s, H * (1 - margin) - (hi[1] - lo[1]) * s]

def eye_at(W, H, path_pts, style='A', max_frac=0.09, min_frac=0.05, avoid=None):
    """Place the eye in the most open spot: largest clearance from existing ink (distance transform), mild preference
    for the lower part of the picture, kept inside a margin. Width shrinks to fit the space, within [min, max] of W."""
    import cv2
    ink = np.zeros((int(H) + 1, int(W) + 1), np.uint8)
    for p in path_pts: cv2.polylines(ink, [np.round(p).astype(np.int32)], False, 1, 2)
    free = cv2.distanceTransform((1 - ink).astype(np.uint8), cv2.DIST_L2, 5)
    if avoid is not None:                                 # never inside a person or object, even where it is blank
        av = cv2.resize(avoid.astype(np.uint8), (free.shape[1], free.shape[0]), interpolation=cv2.INTER_NEAREST)
        free = np.minimum(free, cv2.distanceTransform((1 - av).astype(np.uint8), cv2.DIST_L2, 5))
    m = int(0.06 * min(W, H)); free[:m] = 0; free[-m:] = 0; free[:, :m] = 0; free[:, -m:] = 0
    low = free.copy(); low[: int(0.5 * H)] = 0           # lower half preferred (signature territory)
    if low.max() >= min_frac * W / 1.6: free = low
    end = np.asarray(path_pts[-1])[-1]; yy, xx = np.mgrid[0:free.shape[0], 0:free.shape[1]]
    near = 1 - 0.6 * np.clip(np.hypot(xx - end[0], yy - end[1]) / np.hypot(W, H), 0, 1)
    score = np.minimum(free, max_frac * W / 1.6) * near   # enough room counts; beyond that, closer to where the line ends wins
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    clear = float(free[y, x])
    w = float(np.clip(1.6 * clear, min_frac * W, max_frac * W))
    p = eye_unit(style); lo, hi = p.min(0), p.max(0); s = w / (hi[0] - lo[0])
    q = (p - (lo + hi) / 2) * s + [x, y]
    return q
