# Reveal video: the line draws itself over the faded photo at hand speed, a cut-out hand holds the pencil,
# the camera follows like the reference (close, ~2x, iPad edge visible near borders), then a hard cut to the clean drawing.
# Usage: python3 video.py <photo 9:16> <line dir (path_dN.npy, kinds_dN.json)> <detail N> <hand_sprite.png> <out.mp4> [draw seconds]
import sys, json, math, subprocess, time
import numpy as np, cv2
HERE = __import__('os').path.dirname(__import__('os').path.abspath(__file__)); sys.path.insert(0, HERE)
import linephoto as LP

photo_p, linedir, det, hand_p, out_p = sys.argv[1:6]
FPS, W, H = 30, 1080, 1920
DRAW_S = float(sys.argv[6]) if len(sys.argv) > 6 else 48.0
UNIT = W / 500.0                 # screen px per path unit at zoom 1 (the line was traced at 500 px wide)
K = UNIT * 2.0                   # world raster px per unit (1:1 with the screen at zoom 2)
M = 70                           # margin in units around the photo: iPad bezel + desk
T0 = time.time()
def log(*a): print(f'[{time.time()-T0:6.1f}s]', *a, flush=True)

pts = np.load(f'{linedir}/path_d{det}.npy'); kinds = json.load(open(f'{linedir}/kinds_d{det}.json'))
lazy = np.load(f'{linedir}/lazy_d{det}.npy') if __import__('os').path.exists(f'{linedir}/lazy_d{det}.npy') else np.zeros(len(pts))
secs = np.load(f'{linedir}/secs_d{det}.npy') if __import__('os').path.exists(f'{linedir}/secs_d{det}.npy') else np.zeros(len(pts), np.int32)
r_units = LP.widths(pts, kinds); NIB = LP.P['nib_scale']
INK = float(__import__('os').environ.get('INK', '1.8'))       # heavy ink in videos (the hydrant look); 1.0 = same as the still
if __import__('os').path.exists(f'{linedir}/widths_d{det}.npy'):          # exact widths and brush of the approved still
    _m = json.load(open(f'{linedir}/meta_d{det}.json')); r_units = np.load(f'{linedir}/widths_d{det}.npy') / _m['px_per_unit']; NIB = _m['nib_scale']
NIB *= INK
img = cv2.imread(photo_p); ph, pw = img.shape[:2]
PW, PH = 500.0, 500.0 * ph / pw  # photo size in units

# ---------- world raster: desk, iPad, faded photo ----------
RW, RH = int((PW + 2 * M) * K), int((PH + 2 * M) * K)
NOIPAD = __import__('os').environ.get('NOIPAD') == '1'       # no iPad: the photo alone on white
base = np.empty((RH, RW, 3), np.uint8); base[:] = (255, 255, 255) if NOIPAD else (24, 25, 27)
rng = np.random.default_rng(0)
if not NOIPAD: base = np.clip(base.astype(np.int16) + rng.normal(0, 3, (RH, RW, 1)).astype(np.int16), 0, 255).astype(np.uint8)
bz = 16                                                     # bezel in units
x0, y0 = int((M - bz) * K), int((M - bz) * K); x1, y1 = int((M + PW + bz) * K), int((M + PH + bz) * K)
if not NOIPAD: cv2.rectangle(base, (x0, y0), (x1, y1), (12, 12, 13), -1, cv2.LINE_AA)
for cx_, cy_ in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)): pass
sx0, sy0 = int(M * K), int(M * K); sw, sh = int(PW * K), int(PH * K)
g = cv2.cvtColor(cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
g = 255 - (255 - g) * 0.48                                   # washed out like the reference screen
scr = np.dstack([g + 6, g + 2, g + 5]).clip(0, 255).astype(np.uint8)   # slight cool/lilac screen tint
if NOIPAD:                                                   # outside the photo: blurred, faded, enlarged copy of it
    sc_ = max(RW / sw, RH / sh) * 1.08
    bgi = cv2.resize(cv2.cvtColor(cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY), (int(sw * sc_), int(sh * sc_)))
    oy, ox = (bgi.shape[0] - RH) // 2, (bgi.shape[1] - RW) // 2; bgi = bgi[oy:oy + RH, ox:ox + RW].astype(np.float32)
    bgi = cv2.GaussianBlur(bgi, (0, 0), 0.03 * RW); bgi = 255 - (255 - bgi) * 0.30
    base[:] = np.dstack([bgi + 6, bgi + 2, bgi + 5]).clip(0, 255).astype(np.uint8)
base[sy0:sy0 + sh, sx0:sx0 + sw] = scr
base_col = base.copy(); base_col[sy0:sy0 + sh, sx0:sx0 + sw] = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
ink = np.full((RH, RW), 255, np.uint8)
log('world raster', RW, 'x', RH)

# ---------- timing: one continuous, smooth speed ----------
# Speed eases down a little in tight bends (never below ~55% of cruising speed), travel sweeps are a bit quicker,
# and the speed is then smoothed in time (~0.35 s) so it can never change abruptly. No stops, no pulses.
d = np.r_[0, np.hypot(*np.diff(pts, axis=0).T)]; s_ = np.cumsum(d)
tg = np.gradient(pts, axis=0); head = np.unwrap(np.arctan2(tg[:, 1], tg[:, 0]))
curv = np.abs(np.gradient(head)) / (np.hypot(*tg.T) + 1e-6)
curv = cv2.GaussianBlur(curv.reshape(-1, 1).astype(np.float32), (1, 0), 8).ravel().astype(np.float64)
aux = np.array([k in ('aux', 'walk') for k in kinds])   # travel strokes
vr = 0.20 + 0.80 / (1 + (curv * 12) ** 1.4)          # hands slow clearly in bends, speed up on straights
vr *= 1 + 1.8 * lazy                                          # dense filling is scribbled fast
vr[aux] *= 2.6                                                # joins and travel lines are quick bursts
vr[np.array([k == 'sig' for k in kinds])] *= 0.75                # the signature is written calmly
vr = cv2.GaussianBlur(vr.reshape(-1, 1).astype(np.float32), (1, 0), 16).ravel().astype(np.float64)
t0 = np.cumsum(d / vr); t0 *= DRAW_S / t0[-1]
# smooth the speed in time: resample distance on a fine clock, blur the speed, re-integrate
FS = 120.0; tg_ = np.arange(0, DRAW_S + 1 / FS, 1 / FS); sg = np.interp(tg_, t0, s_)
spd = np.gradient(sg, 1 / FS); spd = cv2.GaussianBlur(spd.reshape(-1, 1).astype(np.float32), (1, 0), 0.22 * FS).ravel().astype(np.float64)
ramp = np.clip(tg_ / 0.6, 0, 1); ramp = ramp * ramp * (3 - 2 * ramp); spd *= np.maximum(ramp, 0.15)   # ease off the start
spd = np.maximum(spd, 1e-3); sg = np.r_[0, np.cumsum((spd[1:] + spd[:-1]) / 2 / FS)]; sg *= s_[-1] / sg[-1]
t = np.interp(s_, sg, tg_)
v = np.gradient(s_) / np.maximum(np.gradient(t), 1e-6)
vn = cv2.GaussianBlur((v / np.median(v)).reshape(-1, 1).astype(np.float32), (1, 0), 10).ravel()
r_units = r_units * np.clip(0.85 + 0.25 * (vn - 1), 0.75, 1.35)      # ink swells on fast sweeps
log('path', len(pts), 'points, mean speed', round(float(s_[-1] / DRAW_S), 1), 'units/s, speed range', np.percentile(spd, [5, 95]).round(0))

def pen_at(tt):
    i = int(np.searchsorted(t, tt)); i = min(max(i, 1), len(pts) - 1)
    a = (tt - t[i - 1]) / max(t[i] - t[i - 1], 1e-6); a = min(max(a, 0), 1)
    return pts[i - 1] * (1 - a) + pts[i] * a, i

# ---------- hand sprite (alpha and soft shadow precomputed once) ----------
hand = cv2.imread(hand_p, cv2.IMREAD_UNCHANGED)
TIP = tuple(map(int, open(hand_p.rsplit('.', 1)[0] + '_tip.txt').read().split()))
HS = 1.35                                                    # sprite scale at zoom 2 (big hand: little forearm in frame)
HAND_C = np.ascontiguousarray(hand[:, :, :3]); HAND_A = hand[:, :, 3].astype(np.float32) / 255
HAND_SH = cv2.GaussianBlur(HAND_A, (0, 0), 18) * 0.30
hh, hw = HAND_A.shape
_ys, _xs = np.nonzero(HAND_A > 0.5); _cen = np.array([_xs.mean(), _ys.mean()])
PIV = np.array(TIP, float) + 0.6 * (_cen - np.array(TIP, float))   # wrist-ish pivot on the hand
BASEV = PIV - np.array(TIP, float); BASE_ANG = math.degrees(math.atan2(BASEV[1], BASEV[0])); BASE_LEN = float(np.hypot(*BASEV))
CORNERS = np.array([[0, 0, 1], [hw, 0, 1], [0, hh, 1], [hw, hh, 1]], float).T

PEN = __import__('os').environ.get('PEN', 'dot')            # 'dot' (soft glowing pen point) or 'hand'
def dot_draw(frame, sp, scale, fade=1.0, trail=()):
    x, y = float(sp[0]), float(sp[1]); R = 34 * scale
    pts_ = [(x, y)] + list(trail)
    xs = [p[0] for p in pts_]; ys = [p[1] for p in pts_]; pad = 3 * R
    x0, y0, x1, y1 = int(max(0, min(xs) - pad)), int(max(0, min(ys) - pad)), int(min(W, max(xs) + pad + 1)), int(min(H, max(ys) + pad + 1))
    if x0 >= x1 or y0 >= y1: return frame
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32); f = frame[y0:y1, x0:x1].astype(np.float32)
    for k, (tx, ty) in enumerate(trail):                       # fading comet tail
        a = (1 - (k + 1) / (len(trail) + 1)) ** 1.5 * 0.35 * fade; rr = R * (0.55 - 0.35 * (k + 1) / (len(trail) + 1))
        d_ = np.hypot(xx - tx, yy - ty); g = np.clip((rr - d_) / 2, 0, 1) * a
        f = f * (1 - g[..., None]) + 255 * g[..., None]
    d = np.hypot(xx - x, yy - y)
    sh = np.exp(-(np.hypot(xx - x - 0.25 * R, yy - y - 0.35 * R) / (0.9 * R)) ** 2) * 0.35 * fade      # soft drop shadow
    f = f * (1 - sh[..., None])
    glow = np.exp(-(d / (1.1 * R)) ** 2) * 0.45 * fade
    f = f * (1 - glow[..., None]) + 255 * glow[..., None]
    rim = np.clip((0.62 * R - d) / 1.5, 0, 1) * fade                # bright disc
    f = f * (1 - rim[..., None]) + 250 * rim[..., None]
    core = np.clip((0.30 * R - d) / 1.3, 0, 1) * fade              # dark ink point
    f = f * (1 - core[..., None]) + 18 * core[..., None]
    frame[y0:y1, x0:x1] = np.clip(f, 0, 255).astype(np.uint8)
    return frame

def hand_warp(frame, sp, scale, rot):
    Mh = cv2.getRotationMatrix2D((float(TIP[0]), float(TIP[1])), rot, scale); Mh[:, 2] += [sp[0] - TIP[0], sp[1] - TIP[1]]
    sx, sy = 14 * scale, 26 * scale
    c = Mh @ CORNERS
    bx0, by0 = int(max(math.floor(c[0].min()), 0)), int(max(math.floor(c[1].min()), 0))
    bx1, by1 = int(min(math.ceil(c[0].max() + sx), W)), int(min(math.ceil(c[1].max() + sy), H))
    if bx0 >= bx1 or by0 >= by1: return frame
    Mb = Mh.copy(); Mb[:, 2] -= [bx0, by0]; Ms = Mb.copy(); Ms[:, 2] += [sx, sy]
    size = (bx1 - bx0, by1 - by0)
    sh = cv2.warpAffine(HAND_SH, Ms, size, flags=cv2.INTER_LINEAR, borderValue=0)
    a = cv2.warpAffine(HAND_A, Mb, size, flags=cv2.INTER_LINEAR, borderValue=0)
    col = cv2.warpAffine(HAND_C, Mb, size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    f = frame[by0:by1, bx0:bx1].astype(np.float32)
    keep = (1 - sh) * (1 - a)
    f = f * keep[:, :, None] + col.astype(np.float32) * a[:, :, None]
    frame[by0:by1, bx0:bx1] = f.astype(np.uint8)
    return frame

# ---------- camera helpers ----------
class Spring:
    def __init__(s, x, tau): s.x = np.array(x, float); s.v = np.zeros_like(s.x); s.w = 2.0 / tau
    def step(s, target, dt=1 / FPS):
        a = s.w * s.w * (np.asarray(target, float) - s.x) - 2 * s.w * s.v; s.v += a * dt; s.x += s.v * dt; return s.x
def smooth(cur, target, tau): return cur + (target - cur) * (1 - math.exp(-1 / (FPS * tau)))
def ease(x): x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)
def ease_in(x): x = min(max(x, 0.0), 1.0); return x ** 3

Z_FULL = min(W / (PW + 2 * bz + 40), H / (PH + 2 * bz + 40)) / UNIT        # whole iPad with a little desk
if NOIPAD: Z_FULL = min(W / PW, H / PH) / UNIT                              # the photo fills the screen's width or height
C_PHOTO = np.array([PW / 2, PH / 2])
Z_DRAW = 2.2
# camera frames the section being drawn: small, detailed sections get a deep zoom, big ones a wide view
SEC_Z = {}; SEC_C = {}
for sid in np.unique(secs):
    q = pts[secs == sid]
    if len(q) < 2: continue
    lo_, hi_ = np.percentile(q, 3, axis=0), np.percentile(q, 97, axis=0); bw, bh = np.maximum(hi_ - lo_, 8)
    SEC_Z[int(sid)] = float(np.clip(min(0.62 * W / (bw * UNIT), 0.62 * H / (bh * UNIT)), 1.2, 2.7))      # a little more dynamic (Nov 2026); SEC_C[int(sid)] = (lo_ + hi_) / 2
def sec_at(i): return int(secs[min(max(i, 0), len(secs) - 1)])
PHASES = [('hand_in', 0.35 if PEN != 'dot' else 0.0), ('draw', DRAW_S), ('zoom_out', 1.3), ('ipad_hold', 0.25), ('to_white', 0.7), ('end_hold', 1.4)]   # dot pen: ink moves from frame 0
OFF = np.array([950.0, 1300.0])                               # where the hand goes when it lifts away (off screen)

def clamp_cam(cam, z):
    S = UNIT * z; cw, ch = W / S * K, H / S * K
    cxr = RW / 2 if cw >= RW else min(max((cam[0] + M) * K, cw / 2), RW - cw / 2)
    cyr = RH / 2 if ch >= RH else min(max((cam[1] + M) * K, ch / 2), RH - ch / 2)
    return np.array([cxr / K - M, cyr / K - M]), S

def follow_target(pen_ahead, S):
    want = np.array([W * 0.46, H * 0.44])
    t_ = pen_ahead - (want - [W / 2, H / 2]) / S
    vw, vh = W / S, H / S
    lx, hx, ly, hy = vw / 2, PW - vw / 2, vh / 2, PH - vh / 2           # view stays inside the photo while drawing
    if hx < lx: lx = hx = PW / 2
    if hy < ly: ly = hy = PH / 2
    return np.array([PW / 2 if lx > hx else min(max(t_[0], lx), hx), PH / 2 if ly > hy else min(max(t_[1], ly), hy)])
def wrist(pen): return -5.0 * (pen[0] / PW - 0.5) - 3.0 * (pen[1] / PH - 0.5)

# ---------- pass 1: plan every frame (cheap, sequential because the springs carry state) ----------
DRAW_F = []        # indices of drawing frames (camera smoothed afterwards)
plan = []          # (cam, z, col, white, ink_idx, hand or None) ; hand = (anchor_world, offset_px, scale, rot)
vmean = float(np.mean(v)); vs_hist = 0.0
def hand_pose(pen, wrist_w, c, S):
    # rotate/scale the sprite about the pen tip so its wrist sits where the (lagging) wrist is
    sp = (pen - c) * S + [W / 2, H / 2]; wp = (wrist_w - c) * S + [W / 2, H / 2]
    vec = wp - sp; ang = math.degrees(math.atan2(vec[1], vec[0]))
    sc = np.clip(np.hypot(*vec) / BASE_LEN, HS * (S / UNIT) / 2 * 0.93, HS * (S / UNIT) / 2 * 1.07)
    return -(ang - BASE_ANG), float(sc)
def wrist_rest(pen, S, extra_rot=0.0):
    a = math.radians(BASE_ANG - extra_rot); ln = BASE_LEN * HS * (S / UNIT) / 2 / S
    return pen + ln * np.array([math.cos(a), math.sin(a)])
_lo = np.percentile(pts, 0.5, axis=0) - 6; _hi = np.percentile(pts, 99.5, axis=0) + 6   # ink extent (ignores stray tails)
C_INK = (_lo + _hi) / 2; Z_INK = min(0.84 * W / (_hi[0] - _lo[0]), 0.80 * H / (_hi[1] - _lo[1])) / UNIT
# opening frame: about OPEN_SHOW of the subject in view (enough to tell what it is, not all of it), centred between
# where the line starts and the middle of the drawing; the line is already moving on the first frame
OPEN_SHOW = float(__import__('os').environ.get('OPEN_SHOW', '0.85'))
OPEN_HOLD = float(__import__('os').environ.get('OPEN_HOLD', '1.8'))   # seconds the wide opening view is held before zooming into sections
_sub = pts[np.array(kinds) == 'L1'] if (np.array(kinds) == 'L1').sum() > 50 else pts
_bw, _bh = np.percentile(_sub, 97, axis=0) - np.percentile(_sub, 3, axis=0)      # the whole subject (its strokes, not the joins)
Z_DRAW = float(np.clip(min(W / (OPEN_SHOW * _bw * UNIT), H / (OPEN_SHOW * _bh * UNIT)), Z_FULL * 1.05, 1.6))
C_START, _ = clamp_cam(0.35 * pts[0] + 0.65 * C_INK, Z_DRAW)
for name, dur in PHASES:
    n = int(round(dur * FPS))
    if name == 'draw':
        zsp = Spring([Z_DRAW], 3.0); csp = Spring(C_START, 2.0); rsp = Spring([wrist(pts[0])], 2.5)
        wsp = Spring(wrist_rest(pts[0], UNIT * Z_DRAW, wrist(pts[0])), 0.45)
    for k in range(n):
        x = (k + 1) / n
        if name == 'show':
            c, _ = clamp_cam(C_PHOTO, Z_FULL); plan.append((c, Z_FULL, 0.0, 0.0, 0, None))
        elif name == 'zoom_in':
            e = ease(x); z = Z_FULL * (Z_DRAW / Z_FULL) ** e
            c, _ = clamp_cam(C_PHOTO + (C_START - C_PHOTO) * e, z); plan.append((c, z, 0.0, 0.0, 0, None))
        elif name == 'hand_in':
            off = OFF * (1 - x) ** 3                                   # glides in and settles on the start point
            plan.append((C_START, Z_DRAW, 0.0, 0.0, 0, (pts[0].copy(), off, HS * Z_DRAW / 2, wrist(pts[0]))))
        elif name == 'draw':
            tt = k / FPS
            pen, idx = pen_at(tt); ahead, _ = pen_at(min(tt + 0.9, DRAW_S))
            _, idx_a = pen_at(min(tt + 0.6, DRAW_S)); sid = sec_at(idx_a)
            zt = SEC_Z.get(sid, Z_DRAW)
            if tt < OPEN_HOLD: zt = Z_DRAW                        # hold the wide opening: viewers see what it is first
            z = float(zsp.step([zt])[0]); S = UNIT * z
            cam = csp.step(0.55 * follow_target(ahead, S) + 0.45 * SEC_C.get(sid, ahead)).copy()
            sp = (pen - cam) * S + [W / 2, H / 2]
            for lo, hi, soft in ((np.array([0.22 * W, 0.18 * H]), np.array([0.68 * W, 0.74 * H]), 0.08),
                                 (np.array([0.10 * W, 0.08 * H]), np.array([0.82 * W, 0.86 * H]), 1.0)):
                over = np.maximum(sp - hi, 0) - np.maximum(lo - sp, 0)
                cam += over * soft / S; sp = (pen - cam) * S + [W / 2, H / 2]
            _vw, _vh = W / S, H / S                                    # the view never leaves the photo while drawing
            cam[0] = PW / 2 if _vw >= PW else min(max(cam[0], _vw / 2), PW - _vw / 2)
            cam[1] = PH / 2 if _vh >= PH else min(max(cam[1], _vh / 2), PH - _vh / 2)
            csp.x = cam.copy()
            c, _ = clamp_cam(cam, z)
            bias = float(rsp.step([wrist(pen)])[0])
            ww = wsp.step(wrist_rest(pen, S, bias)).copy()            # wrist trails the pen tip
            rot, scl = hand_pose(pen, ww, c, S)
            plan.append((c, z, 0.0, 0.0, idx, (pen.copy(), np.zeros(2), scl, rot))); DRAW_F.append(len(plan) - 1)
            cam_end, z_end, rot_end = c.copy(), z, rot
        elif name == 'zoom_out':
            e = ease(x); z = z_end * (Z_FULL / z_end) ** e
            c, _ = clamp_cam(cam_end + (C_PHOTO - cam_end) * e, z)
            hand = (pts[-1], OFF * ease_in(x / 0.6), HS * z / 2, rot_end) if x < 0.6 else None
            plan.append((c, z, 0.0, 0.0, len(pts) - 1, hand))
        elif name == 'ipad_hold':
            c, _ = clamp_cam(C_PHOTO, Z_FULL); plan.append((c, Z_FULL, 0.0, 0.0, len(pts) - 1, None))
        else:                                                         # fade to white while centring the drawing itself
            e = ease(x) if name == 'to_white' else 1.0
            c0, _ = clamp_cam(C_PHOTO, Z_FULL)
            plan.append((c0 + (C_INK - c0) * e, Z_FULL * (Z_INK / Z_FULL) ** e, 0.0, e, len(pts) - 1, None))
log('planned', len(plan), 'frames', round(len(plan) / FPS, 1), 's')

# ---------- camera glide: smooth the planned drawing-phase camera as a whole (no lurches against the photo edge) ----------
CAM_SMOOTH = float(__import__('os').environ.get('CAM_SMOOTH', '0.45'))      # seconds
if DRAW_F and CAM_SMOOTH > 0:
    i0, i1 = DRAW_F[0], DRAW_F[-1] + 1
    C_ = np.array([p[0] for p in plan[i0:i1]], float); Z_ = np.array([p[1] for p in plan[i0:i1]], float)
    sg = CAM_SMOOTH * FPS
    def gs(x): return cv2.GaussianBlur(x.reshape(-1, 1).astype(np.float32), (1, 0), sg, borderType=cv2.BORDER_REPLICATE).ravel()
    Zs = np.exp(gs(np.log(Z_)))
    def edge(cx, half, size):                   # soft limit: keeps the view inside the photo without a hard stop
        lo, hi = half, size - half
        return np.where(hi < lo, size / 2, np.clip(cx, lo, np.maximum(hi, lo)))
    for _ in range(2):
        Cs = np.c_[gs(C_[:, 0]), gs(C_[:, 1])]
        vw, vh = W / (UNIT * Zs), H / (UNIT * Zs)
        C_ = np.c_[edge(Cs[:, 0], vw / 2, PW), edge(Cs[:, 1], vh / 2, PH)]
    Cs = np.c_[gs(C_[:, 0]), gs(C_[:, 1])]; Cs = np.c_[edge(Cs[:, 0], vw / 2, PW), edge(Cs[:, 1], vh / 2, PH)]
    # keep the pen inside a safe box: where it would get within 10% of the frame edge, ease the camera over in time
    PEN_ = np.array([plan[f][5][0] if plan[f][5] is not None else plan[f][0] for f in range(i0, i1)], float)
    for _ in range(6):
        S_ = UNIT * Zs; half = np.c_[0.40 * W / S_, 0.40 * H / S_]
        d = PEN_ - Cs; need = np.sign(d) * np.maximum(np.abs(d) - half, 0)
        if not np.abs(need).max() > 1e-3: break
        Cs = Cs + np.c_[gs(need[:, 0]), gs(need[:, 1])] * 2.2
        Cs = np.c_[edge(Cs[:, 0], vw / 2, PW), edge(Cs[:, 1], vh / 2, PH)]
    # blend the first/last half second back into the unsmoothed camera so the opening and outro join seamlessly
    nb = int(0.5 * FPS); wgt = np.ones(len(Cs))
    wgt[:nb] = np.linspace(0, 1, nb) if len(Cs) > 2 * nb else 1; wgt[-nb:] = np.minimum(wgt[-nb:], np.linspace(1, 0, nb)) if len(Cs) > 2 * nb else 1
    for k, f in enumerate(range(i0, i1)):
        c, z, col, white, idx, hand_ = plan[f]; w_ = wgt[k]
        plan[f] = (c * (1 - w_) + Cs[k] * w_, z * (1 - w_) + Zs[k] * w_, col, white, idx, hand_)
    out_ = 0
    for f in range(i0, i1):
        c, z, _, _, idx, hand_ = plan[f]
        if hand_ is None: continue
        sp_ = (hand_[0] - c) * UNIT * z
        out_ += abs(sp_[0]) > 0.5 * W or abs(sp_[1]) > 0.5 * H
    log(f'camera glide: drawing camera smoothed over {CAM_SMOOTH:.2f} s; pen outside the frame in {out_} of {i1 - i0} frames')

# ---------- caption overlay (optional): CAPTION_PNG = RGBA image the size of the frame, BGR order ----------
CAP, CAP_Y = None, (0, 0)
if __import__('os').environ.get('CAPTION_PNG'):
    CAP = cv2.imread(__import__('os').environ['CAPTION_PNG'], cv2.IMREAD_UNCHANGED)
    _rows = np.nonzero(CAP[:, :, 3].max(axis=1))[0]
    CAP_Y = (int(_rows.min()), int(_rows.max()) + 1) if len(_rows) else (0, 0)

# ---------- pass 2: render chunks in parallel ----------
def render_chunk(job):
    ci, f0, f1, path = job
    ink = np.full((RH, RW), 255, np.uint8)
    comp = base.copy()                                   # photo x ink, updated only where ink lands
    drawn = 0
    def ink_to(idx):
        nonlocal drawn
        if idx <= drawn: return
        seg = pts[max(drawn - 1, 0):idx + 1]; rr = r_units[max(drawn - 1, 0):idx + 1]
        P2 = ((seg + M) * K * 16).astype(np.int32)
        LP.draw_nib(ink, (seg + M) * K, rr * K * NIB, LP.P['nib_ang'], LP.P['nib_min'])
        pad = int(rr.max() * K * NIB) + 3; q = (seg + M) * K
        x0, y0 = max(int(q[:, 0].min()) - pad, 0), max(int(q[:, 1].min()) - pad, 0)
        x1, y1 = min(int(q[:, 0].max()) + pad + 1, RW), min(int(q[:, 1].max()) + pad + 1, RH)
        comp[y0:y1, x0:x1] = (base[y0:y1, x0:x1].astype(np.uint16) * ink[y0:y1, x0:x1, None] // 255).astype(np.uint8)
        drawn = idx
    ink_to(plan[f0][4])                                  # catch up to where this chunk starts
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                           '-c:v', 'libx264', '-preset', 'veryfast', '-crf', __import__('os').environ.get('CRF', '19'), '-pix_fmt', 'yuv420p', '-threads', '1', path], stdin=subprocess.PIPE)
    for f in range(f0, f1):
        c, z, col, white, idx, hand_ = plan[f]
        ink_to(idx)
        S = UNIT * z; cw, ch = W / S * K, H / S * K
        fx0, fy0 = (c[0] + M) * K - cw / 2, (c[1] + M) * K - ch / 2
        X0, Y0 = max(int(fx0) - 2, 0), max(int(fy0) - 2, 0); X1, Y1 = min(int(fx0 + cw) + 3, RW), min(int(fy0 + ch) + 3, RH)
        crop = comp[Y0:Y1, X0:X1]
        if white > 0:
            crop = (cv2.addWeighted(base[Y0:Y1, X0:X1], 1 - white, np.full_like(crop, 255), white, 0).astype(np.uint16) * ink[Y0:Y1, X0:X1, None] // 255).astype(np.uint8)
        sc = W / cw; k2 = 1.0
        while sc / k2 < 0.6: crop = cv2.resize(crop, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA); k2 *= 0.5
        Mb = np.array([[sc / k2, 0, -(fx0 - X0) * sc], [0, sc / k2, -(fy0 - Y0) * sc]], np.float64)
        fr = cv2.warpAffine(crop, Mb, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if hand_ is not None:
            anchor, off, hs, rot = hand_
            if PEN == 'dot':
                lift = float(np.hypot(*off)); fade = max(0.0, 1.0 - lift / 400.0)          # sliding in/out becomes a fade
                if fade > 0:
                    trail = []
                    for back in (1, 2, 3, 4, 5, 6):
                        if f - back < f0 - 0 and f - back < 0: break
                        pb = plan[max(f - back, 0)][5]
                        if pb is not None: trail.append(tuple((pb[0] - c) * S + [W / 2, H / 2]))
                    fr = dot_draw(fr, (anchor - c) * S + [W / 2, H / 2], S / UNIT / 2, fade, trail)
            else:
                fr = hand_warp(fr, (anchor - c) * S + [W / 2, H / 2] + off, hs, rot)
        if CAP is not None:                                   # caption burned in while drawing (no second encode)
            y0c, y1c = CAP_Y; a_ = CAP[y0c:y1c, :, 3:4].astype(np.float32) / 255
            fr[y0c:y1c] = (fr[y0c:y1c] * (1 - a_) + CAP[y0c:y1c, :, :3] * a_).astype(np.uint8)
        ff.stdin.write(fr.tobytes())
        if (f - f0) % 300 == 0: log(f'  chunk {ci}: frame {f - f0}/{f1 - f0}')
    ff.stdin.close(); ff.wait()
    return path

if __name__ == '__main__':
    import multiprocessing as mp, os, tempfile
    nproc = int(os.environ.get('JOBS', 0)) or max(1, os.cpu_count())
    nchunk = nproc
    bounds = np.linspace(0, len(plan), nchunk + 1).astype(int)
    tmpd = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(out_p)))
    jobs = [(i, bounds[i], bounds[i + 1], os.path.join(tmpd, f'part{i:02d}.mp4')) for i in range(nchunk)]
    with mp.get_context('fork').Pool(nproc) as pool: parts = pool.map(render_chunk, jobs)
    lst = os.path.join(tmpd, 'parts.txt'); open(lst, 'w').write(''.join(f"file '{p}'\n" for p in parts))
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', '-movflags', '+faststart', out_p], check=True)
    for p in parts: os.remove(p)
    os.remove(lst); os.rmdir(tmpd)
    log('wrote', out_p, round(len(plan) / FPS, 1), 's with', nproc, 'processes')
