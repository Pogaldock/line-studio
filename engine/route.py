# Route optimiser for the single line (the "delivery driver" view): the strokes are stops that must all be visited,
# each can be driven in either direction, and the order + directions are chosen so the whole tour flows.
# Join cost = distance x (1 + turning penalty) + ink-crossing penalty + small penalty for switching objects.
#   turning: how far the line must turn leaving one stroke and entering the next (carrying on straight is cheap)
#   ink: how much already-drawn line the straight hop would cut through (dense patches are avoided)
# Start: greedy (cheapest next stop). Then 2-opt with stroke flips: reverse a stretch of the tour (its order and each
# stroke's direction) whenever that lowers the total, until no reversal helps. Plugs into hydrant_style.order.
# Usage: python3 route.py <photo> <outdir> [key=value ...]   (runs hydrant_style with this ordering)
import sys, os, math, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hydrant_style as HS

R = dict(fast=1, route=1, w_turn=0.8, w_ink=2.5, obj_pen=0.08, passes=8, ink_sigma=8.0)
CTX = {}

def _dir_end(s):
    q = s[-1] - s[max(0, len(s) - 8)]; return q / (np.hypot(*q) + 1e-9)
def _dir_start(s):
    q = s[min(len(s) - 1, 7)] - s[0]; return q / (np.hypot(*q) + 1e-9)

def build(keep, W, H):
    S = [s for s, _ in keep]; O = [o for _, o in keep]; n = len(S)
    ink = np.zeros((H, W), np.uint8)
    for s in S: cv2.polylines(ink, [np.round(s).astype(np.int32)], False, 1, 3)
    dens = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), R['ink_sigma']); dens /= dens.max() + 1e-6
    # ends: index 2*i = start of stroke i, 2*i+1 = end; leaving point / direction for each orientation
    P = np.array([[s[0], s[-1]] for s in S]).reshape(-1, 2)                  # 2n points
    Dout = np.array([[-_dir_start(s), _dir_end(s)] for s in S]).reshape(-1, 2)   # direction when LEAVING at that end
    Din = -Dout                                                                    # direction when ENTERING at that end
    def cost(e_from, e_to):
        a, b = P[e_from], P[e_to]; v = b - a; d = float(np.hypot(*v))
        if d < 1e-6: return 0.0
        u = v / d
        t1 = math.acos(np.clip(Dout[e_from] @ u, -1, 1)); t2 = math.acos(np.clip(u @ Din[e_to], -1, 1))
        k = max(4, int(d / 6)); t = np.linspace(0, 1, k)[1:-1, None]
        pts = np.clip(np.round(a + v * t).astype(int), 0, [W - 1, H - 1])
        ink_c = float(dens[pts[:, 1], pts[:, 0]].mean()) if len(pts) else 0.0
        same = O[e_from // 2] == O[e_to // 2]
        return d * (1 + R['w_turn'] * (t1 + t2) / math.pi) + R['w_ink'] * ink_c * d + (0 if same else R['obj_pen'] * W)
    return S, O, P, cost

def cost_matrix(keep, W, H):
    # every join cost at once (same formula and same arithmetic as cost() above, vectorised): M[e_from, e_to]
    S = [s for s, _ in keep]; O = np.array([o for _, o in keep]); n = len(S)
    ink = np.zeros((H, W), np.uint8)
    for s in S: cv2.polylines(ink, [np.round(s).astype(np.int32)], False, 1, 3)
    dens = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), R['ink_sigma']); dens /= dens.max() + 1e-6
    P = np.array([[s[0], s[-1]] for s in S]).reshape(-1, 2)
    Dout = np.array([[-_dir_start(s), _dir_end(s)] for s in S]).reshape(-1, 2); Din = -Dout
    m = 2 * n; A = np.repeat(P, m, axis=0); B = np.tile(P, (m, 1)); V = B - A
    D = np.hypot(V[:, 0], V[:, 1]); ok = D >= 1e-6; Dsafe = np.where(ok, D, 1.0); U_ = V / Dsafe[:, None]
    DO = np.repeat(Dout, m, axis=0); DI = np.tile(Din, (m, 1))
    c1 = np.clip(DO[:, 0] * U_[:, 0] + DO[:, 1] * U_[:, 1], -1, 1); c2 = np.clip(U_[:, 0] * DI[:, 0] + U_[:, 1] * DI[:, 1], -1, 1)
    T1 = np.arccos(c1); T2 = np.arccos(c2)
    K_ = np.maximum(4, (D / 6).astype(int)); INK = np.zeros(m * m)
    lim = np.array([W - 1, H - 1])
    for k in np.unique(K_[ok]):
        idx = np.nonzero(ok & (K_ == k))[0]; t = np.linspace(0, 1, k)[1:-1, None]
        for c0 in range(0, len(idx), max(1, 2000000 // k)):
            ii = idx[c0:c0 + max(1, 2000000 // k)]
            pts = np.clip(np.round(A[ii][:, None, :] + V[ii][:, None, :] * t[None]).astype(int), 0, lim)
            INK[ii] = dens[pts[..., 1], pts[..., 0]].mean(axis=1)
    same = np.repeat(O[np.arange(m) // 2], m) == np.tile(O[np.arange(m) // 2], m)
    Mx = D * (1 + R['w_turn'] * (T1 + T2) / math.pi) + R['w_ink'] * INK * D + np.where(same, 0, R['obj_pen'] * W)
    Mx[~ok] = 0.0
    return Mx.reshape(m, m)

def tour_cost(seq, cost):
    # seq: list of (stroke i, reversed?) ; leaving end of x -> entering end of y
    tot = 0.0
    for (i, ri), (j, rj) in zip(seq[:-1], seq[1:]):
        tot += cost(2 * i + (0 if ri else 1), 2 * j + (1 if rj else 0))
    return tot

def order_route(keep):
    W, H = CTX['W'], CTX['H']
    S, O, P, cost = build(keep, W, H); n = len(S)
    if n < 3: return HS._order_greedy(keep)
    if R.get('fast', 1):                                   # table lookup instead of computing each join on demand
        Mx = cost_matrix(keep, W, H); cost = lambda a, b: float(Mx[a, b])
    # greedy start from the stroke end nearest the top-left (same start as before)
    left = set(range(n)); seq = []
    e0 = int(np.argmin(np.hypot(*P.T)))
    i0 = e0 // 2; seq.append((i0, e0 % 2 == 1)); left.discard(i0)
    while left:
        i, ri = seq[-1]; ef = 2 * i + (0 if ri else 1)
        best = min(((cost(ef, 2 * j + r), j, r == 1) for j in left for r in (0, 1)), key=lambda t: t[0])
        seq.append((best[1], best[2])); left.discard(best[1])
    c0 = tour_cost(seq, cost)
    # 2-opt with flips: reversing seq[a..b] reverses order and flips each stroke
    memo = {}
    def C(ef, et):
        k = (ef, et)
        if k not in memo: memo[k] = cost(ef, et)
        return memo[k]
    def lv(x): i, r = x; return 2 * i + (0 if r else 1)        # leaving end
    def en(x): i, r = x; return 2 * i + (1 if r else 0)        # entering end
    for _ in range(int(R['passes'])):
        improved = False
        for a in range(1, n - 1):
            for b in range(a + 1, n):
                A_, B_ = seq[a - 1], seq[a]; Y, Z = seq[b], seq[b + 1] if b + 1 < n else None
                old = C(lv(A_), en(B_)) + (C(lv(Y), en(Z)) if Z else 0)
                Bf, Yf = (B_[0], not B_[1]), (Y[0], not Y[1])      # after reversal Y comes first (flipped), B last (flipped)
                new = C(lv(A_), en(Yf)) + (C(lv(Bf), en(Z)) if Z else 0)
                if new < old - 1e-6:
                    seq[a:b + 1] = [(i, not r) for i, r in reversed(seq[a:b + 1])]; improved = True
        if not improved: break
    c1 = tour_cost(seq, cost)
    print(f'[route] {n} strokes: greedy tour cost {c0:.0f} -> optimised {c1:.0f} ({100 * (1 - c1 / max(c0, 1)):.0f}% lower)', flush=True)
    out = []; sec = 0
    for k, (i, r) in enumerate(seq):
        if k and O[i] != O[seq[k - 1][0]]: sec += 1
        out.append((S[i][::-1] if r else S[i], sec))
    return out

HS._order_greedy = HS.order
def order_switch(keep): return order_route(keep) if R['route'] else HS._order_greedy(keep)
HS.order = order_switch

_orig_select = HS.select
def select_ctx(strokes, level, U, W, H, *a, **k):
    CTX['W'], CTX['H'] = W, H; return _orig_select(strokes, level, U, W, H, *a, **k)
HS.select = select_ctx

if __name__ == '__main__':
    rest = []
    for kv in sys.argv[3:]:
        k, v = kv.split('=', 1)
        if k in R: R[k] = float(v)
        else: rest.append(kv)
    sys.argv = sys.argv[:3] + rest
    HS.main()
