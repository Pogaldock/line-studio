# Crop a traced photo to 9:16 around its drawing and move the line data with it, then render the video.
# Usage: python3 make916.py <photo> <linedir> <out.mp4> [draw_seconds=26]
import sys, os, json, shutil, subprocess, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
photo, linedir, out = sys.argv[1], sys.argv[2], sys.argv[3]; draw_s = sys.argv[4] if len(sys.argv) > 4 else '26'
img = cv2.imread(photo); Hf, Wf = img.shape[:2]
if not os.path.exists(f'{linedir}/widths_d1.npy'):              # widths fallback: plain hydrant-style weights
    import linephoto as LP0
    _P = np.load(f'{linedir}/path_d1.npy') * 900 / 500.0; _k = json.load(open(f'{linedir}/kinds_d1.json'))
    np.save(f'{linedir}/widths_d1.npy', LP0.widths(_P, _k).astype(np.float32))
    json.dump(dict(nib_scale=LP0.P['nib_scale'], px_per_unit=900 / 500.0), open(f'{linedir}/meta_d1.json', 'w'))
P = np.load(f'{linedir}/path_d1.npy'); kinds = np.array(json.load(open(f'{linedir}/kinds_d1.json')))
px = P * Wf / 500.0
core = px[kinds == 'L1'] if (kinds == 'L1').any() else px
cx, cy = np.percentile(core, 50, axis=0)
if os.environ.get('NO916') == '1': ch, cw = Hf, Wf          # keep the photo's own shape
elif Wf / Hf > 9 / 16: ch, cw = Hf, int(round(Hf * 9 / 16))
else: cw, ch = Wf, int(round(Wf * 16 / 9))
x0 = int(np.clip(cx - cw / 2, 0, Wf - cw)); y0 = int(np.clip(cy - ch / 2, 0, Hf - ch))
crop = img[y0:y0 + ch, x0:x0 + cw]
work = os.path.splitext(out)[0] + '_916'; ld = os.path.join(work, 'line'); os.makedirs(ld, exist_ok=True)
cphoto = os.path.join(work, 'photo916.jpg'); cv2.imwrite(cphoto, crop, [cv2.IMWRITE_JPEG_QUALITY, 97])
mg = 0.02 * cw
# start on the subject: skip the lead-in from the corner (travel and scene strokes before the first real stroke
# on the middle of the drawing), so the very first frames already draw the subject
lo_, hi_ = np.percentile(core, 25, axis=0), np.percentile(core, 75, axis=0)
inmid = (kinds == 'L1') & (px[:, 0] >= lo_[0]) & (px[:, 0] <= hi_[0]) & (px[:, 1] >= lo_[1]) & (px[:, 1] <= hi_[1])
# OFF by default: with the route order the line already starts on a stroke, and everything before the middle is
# real drawing (on the motorbike it was the whole rider). SKIP_LEAD=1 restores the old skip (greedy order only).
i0 = int(np.argmax(inmid)) if (inmid.any() and os.environ.get('SKIP_LEAD') == '1') else 0
while i0 > 0 and kinds[i0 - 1] == 'L1': i0 -= 1                 # back up to the start of that stroke
print(f'line starts at point {i0} of {len(px)} (skipped {100 * i0 / len(px):.0f}%)', flush=True)
px = px[i0:]; q = px - [x0, y0]
# parts of the line outside the crop are dropped (never squashed onto the edge); each gap is bridged by a straight
# link between where the line left the frame and where it came back (inside, since the frame is a rectangle)
ok = (q[:, 0] >= mg) & (q[:, 0] <= cw - mg) & (q[:, 1] >= mg) & (q[:, 1] <= ch - mg)
kk = np.array(kinds[i0:]); extra = {f: np.load(f'{linedir}/{f}')[i0:] for f in ('secs_d1.npy', 'lazy_d1.npy', 'widths_d1.npy')}
first = int(np.argmax(ok)); last = len(ok) - int(np.argmax(ok[::-1]))
idx_out, P_out, K_out = [], [], []
i = first
while i < last:
    if ok[i]: idx_out.append(i); P_out.append(q[i]); K_out.append(kk[i]); i += 1; continue
    j = i
    while j < last and not ok[j]: j += 1
    if j >= last: break
    a_, b_ = q[i - 1], q[j]; n = max(2, int(np.hypot(*(b_ - a_)) / 1.5))
    ctr = np.array([cw / 2, ch / 2]); c_ = (a_ + b_) / 2; c_ = c_ + 0.45 * (ctr - c_)   # bow the link inward, away from the edge
    for t in np.linspace(0, 1, n + 1)[1:-1]:
        idx_out.append(i - 1); P_out.append((1 - t) ** 2 * a_ + 2 * (1 - t) * t * c_ + t ** 2 * b_); K_out.append('walk')
    i = j
idx_out = np.array(idx_out); q2 = np.array(P_out)
print(f'{(~ok).sum()} points outside the crop dropped, {len(q2)} kept', flush=True)
np.save(f'{ld}/path_d1.npy', q2 * 500.0 / cw)
json.dump(K_out, open(f'{ld}/kinds_d1.json', 'w'))
for f, arr in extra.items(): np.save(f'{ld}/{f}', arr[idx_out])
m = json.load(open(f'{linedir}/meta_d1.json')); m['px_per_unit'] = m['px_per_unit'] * cw / Wf   # same thickness on the photo
json.dump(m, open(f'{ld}/meta_d1.json', 'w'))
print(f'crop {cw}x{ch} at ({x0},{y0}) of {Wf}x{Hf}', flush=True)
# finishing pass: no jumps or spikes, flowing turns, hand drift, pressure taper
if os.environ.get('HUMANIZE', '1') == '1':
    import humanize as HU
    for kv in os.environ.get('HU_ARGS', '').split():
        k_, v_ = kv.split('=', 1); HU.H_[k_] = float(v_)
    mt = json.load(open(f'{ld}/meta_d1.json'))
    P2, r2, k2, s2, l2 = HU.humanize(np.load(f'{ld}/path_d1.npy'), np.load(f'{ld}/widths_d1.npy'), json.load(open(f'{ld}/kinds_d1.json')),
                                     np.load(f'{ld}/secs_d1.npy'), np.load(f'{ld}/lazy_d1.npy'), mt['px_per_unit'])
    np.save(f'{ld}/path_d1.npy', P2); np.save(f'{ld}/widths_d1.npy', r2); json.dump(k2, open(f'{ld}/kinds_d1.json', 'w'))
    np.save(f'{ld}/secs_d1.npy', s2); np.save(f'{ld}/lazy_d1.npy', l2)
    st_ = np.hypot(*np.diff(P2 * mt['px_per_unit'], axis=0).T); print(f'humanized: {len(P2)} points, max step {st_.max():.1f}px', flush=True)
if os.environ.get('STILL') == '1':                                # still only (for checking before a render)
    import linephoto as LP
    mt = json.load(open(f'{ld}/meta_d1.json')); ppu = mt['px_per_unit']; Wd = 500 * ppu; Hd = Wd * ch / cw; Sx = 1.5 * LP.P['ss']
    can = np.full((int(Hd * Sx), int(Wd * Sx)), 255, np.uint8)
    LP.draw_nib(can, np.load(f'{ld}/path_d1.npy') * ppu * Sx, np.load(f'{ld}/widths_d1.npy') * Sx * mt['nib_scale'], LP.P['nib_ang'], LP.P['nib_min'])
    cv2.imwrite(os.path.splitext(out)[0] + '_still.png', cv2.resize(can, (int(Wd * 1.5), int(Hd * 1.5)), interpolation=cv2.INTER_AREA)); sys.exit()
subprocess.run([sys.executable, os.path.join(HERE, 'video.py'), cphoto, ld, '1', os.path.join(HERE, 'hand4_sprite.png'), os.path.abspath(out), draw_s], check=True, cwd=HERE)
