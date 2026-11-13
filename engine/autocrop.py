# Crop a photo around its subject so the subject fills about `fill` of the frame (what made the hydrant and the
# motocross band work), keeping some scene around it; small results are AI-upscaled so the trace has real pixels.
# Usage: python3 autocrop.py <photo> <out.jpg> [fill=0.33] [aspect_min=0.6] [aspect_max=1.4]
import sys, os, math, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import linephoto as LP

def main():
    src, out = sys.argv[1], sys.argv[2]; kv = dict(fill=0.33, aspect_min=0.6, aspect_max=1.4, min_side=1300)
    for a in sys.argv[3:]:
        k, v = a.split('=', 1); kv[k] = float(v)
    img = cv2.imread(src); H, W = img.shape[:2]
    LP.SRC['full'] = img; LP.prep_for_lines(img); m = LP.SRC['matte'] > 0.5
    small = cv2.resize(img, (900, int(H * 900 / W)), interpolation=cv2.INTER_AREA); LP.SRC['img'] = small
    m |= cv2.resize(LP.person_mask(small).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    if n > 1:
        big = st[1:, cv2.CC_STAT_AREA].max(); keep = np.zeros_like(m)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= 0.15 * big: keep |= lab == i
        m = keep
    ys, xs = np.nonzero(m)
    if not len(xs): cv2.imwrite(out, img); print('no subject found, photo unchanged'); return
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max(); bw, bh = x1 - x0 + 1, y1 - y0 + 1; area = m.sum()
    # crop area so the subject covers `fill` of it, never tighter than the subject box plus a margin
    ca = area / kv['fill']; asp = float(np.clip(bw / bh, kv['aspect_min'], kv['aspect_max']))
    cw = max(math.sqrt(ca * asp), bw * 1.2); ch = max(cw / asp, bh * 1.2); cw = max(cw, ch * kv['aspect_min'])
    cw, ch = min(cw, W), min(ch, H)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    X0 = int(np.clip(cx - cw / 2, 0, W - cw)); Y0 = int(np.clip(cy - ch / 2, 0, H - ch))
    crop = img[Y0:Y0 + int(ch), X0:X0 + int(cw)]
    if max(crop.shape[:2]) < 650: crop = LP.upscale(crop)                       # tiny: AI upscale (x4)
    if max(crop.shape[:2]) < kv['min_side']:                                    # otherwise a plain high-quality resize
        f_ = kv['min_side'] / max(crop.shape[:2]); crop = cv2.resize(crop, None, fx=f_, fy=f_, interpolation=cv2.INTER_LANCZOS4)
    if max(crop.shape[:2]) > 2400:
        f_ = 2400 / max(crop.shape[:2]); crop = cv2.resize(crop, None, fx=f_, fy=f_, interpolation=cv2.INTER_AREA)
    cv2.imwrite(out, crop, [cv2.IMWRITE_JPEG_QUALITY, 97])
    print(f'subject {100 * area / (W * H):.0f}% of the photo -> crop {int(cw)}x{int(ch)} at ({X0},{Y0}), subject now {100 * area / (cw * ch):.0f}% of the frame; saved {crop.shape[1]}x{crop.shape[0]}')

if __name__ == '__main__': main()
