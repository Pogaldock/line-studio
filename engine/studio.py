# Line Studio orchestrator: one photo in -> several drawing options (stills) -> the chosen one as a captioned video.
#
#   python3 studio.py trace  <photo> <workdir> [--tag job-xxx]
#   python3 studio.py render <workdir> <option_id> <caption text | variant index> [--seconds 26] [--tag job-xxx]
#   python3 studio.py prerender <workdir> <option_id> [--seconds 26] [--tag job-xxx]
#
# Speed: right after the trace, every option is pre-rendered without a caption (raw_<option>_<seconds>.mp4, a
# near-lossless master). `render` then only burns the caption onto that master (one quick encode); without a master
# (other speed, or tapped before it was ready) it renders in full. Same drawing, same frames either way.
#
# Options come from options.json next to this file (name + crop on/off + style settings), so new looks are a JSON edit.
# With --tag, results go to that GitHub release as they finish (stills one by one, progress_*.json after every step),
# which is what the phone page reads. Without --tag everything just stays in <workdir>.
# Privacy (the repo is public): with LS_KEY set (base64, 32 bytes; a repo secret), the photo arrives encrypted
# (photo.jpg.enc) and every still, drawing bundle and video is encrypted before upload: AES-256-GCM, file = 12-byte
# nonce + ciphertext + tag. Only the phone page holds the same key to decrypt. progress.json stays plain (no image data).
import sys, os, json, time, shutil, subprocess, tarfile, argparse, threading, base64
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
HU_ARGS = 'smooth=1.2 drift=0.5 spike=2.5 taper=10 var=0.12'          # light humanize (the locked look)
MAX_SIDE = 3000                                                        # phone photos (50 MP) are cut down to this first

class Progress:
    # progress file for one run (progress_trace.json, progress_render_<option>.json, so runs never clash): overall
    # stage + % + one entry per option. Written locally on every change; a heartbeat thread uploads it every 3 s when
    # it changed, so the phone sees real movement without every update waiting on an upload.
    def __init__(s, work, tag, kind, fname):
        s.work, s.tag, s.lock, s.fname = work, tag, threading.Lock(), fname; s.t0 = time.time(); s.dirty = False
        s.d = dict(kind=kind, stage='starting', pct=0, options=[], started=int(s.t0), updated=int(s.t0), error=None, done=False)
        s.path = os.path.join(work, fname); s.stop = threading.Event()
        if tag: threading.Thread(target=s._beat, daemon=True).start()
    def _beat(s):
        while not s.stop.wait(3):
            if s.dirty: s.push()
    def push(s):
        with s.lock: s.dirty = False; json.dump(s.d, open(s.path, 'w'))
        upload(s.tag, [s.path])
    def set(s, push=False, **kw):
        with s.lock:
            if 'pct' in kw: kw['pct'] = max(s.d.get('pct', 0), round(float(kw['pct']), 1))     # never goes backwards
            s.d.update(kw); s.d['updated'] = int(time.time()); s.d['elapsed'] = round(time.time() - s.t0, 1)
            json.dump(s.d, open(s.path, 'w')); s.dirty = True
        if push and s.tag: s.push()
        print(f"[studio {s.d['elapsed']:6.1f}s] {s.d['stage']} {s.d['pct']}%", flush=True)
    def opt(s, oid, **kw):
        with s.lock:
            for o in s.d['options']:
                if o['id'] == oid:
                    if 'pct' in kw: kw['pct'] = max(o.get('pct', 0), round(float(kw['pct']), 1))
                    o.update(kw)
    def finish(s): s.stop.set()

# where a trace is, read from the pipeline's own log lines (fractions measured on an uncached photo: the AI models are
# ~60 % of an option, strokes + route most of the rest). Cached lines count the same: they just arrive quickly.
MILESTONES = [('objects found', .33), ('objects2 from cache', .33), ('focus:', .38), ('facehand2 from cache', .38),
              ('preprocess:', .60), ('subject strokes', .90), ('[route]', .95), ('detail 1', 1.0)]
# measured seconds from each milestone to the next (uncached): between two real milestones the bar eases towards the
# next one over that time, up to 90 % of the gap, and never past it
FRACS = [0, .33, .38, .60, .90, .95, 1.0]; SEGS = [23, 1, 15, 21, 1, 2.5]

KEY = base64.b64decode(os.environ['LS_KEY']) if os.environ.get('LS_KEY') else None

def seal(path):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    n = os.urandom(12); open(path + '.enc', 'wb').write(n + AESGCM(KEY).encrypt(n, open(path, 'rb').read(), None)); return path + '.enc'

def unseal(src, dst):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    b = open(src, 'rb').read(); open(dst, 'wb').write(AESGCM(KEY).decrypt(b[:12], b[12:], None)); return dst

def upload_private(tag, files):
    # image data is only ever uploaded encrypted: with no key, nothing leaves the machine
    if not tag: return
    if not KEY: raise RuntimeError('LS_KEY is not set: refusing to upload image data unencrypted')
    upload(tag, [seal(f) for f in files])

def upload(tag, files):
    if not tag: return
    r = subprocess.run(['gh', 'release', 'upload', tag, *files, '--clobber'], capture_output=True, text=True)
    if r.returncode: print('[upload failed]', r.stderr.strip()[:300], flush=True)

def run(cmd, env=None, log=None, on_line=None):
    e = dict(os.environ, **(env or {}))
    p = subprocess.Popen(cmd, cwd=HERE, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    out = []
    for line in p.stdout:
        out.append(line)
        if log: log.write(line)
        if on_line: on_line(line)
    if p.wait(): raise RuntimeError(f'{os.path.basename(cmd[1])} failed:\n' + ''.join(out[-25:]))
    return ''.join(out)

def prepare(photo, work):
    img = cv2.imread(photo)                                  # imread applies the phone's EXIF rotation
    if img is None: raise RuntimeError('could not read the photo')
    h, w = img.shape[:2]; f = MAX_SIDE / max(h, w)
    if f < 1: img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    p = os.path.join(work, 'photo.jpg'); cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 97]); return p

def preview(still_png, out_jpg, width=900):
    a = cv2.imread(still_png); h, w = a.shape[:2]
    cv2.imwrite(out_jpg, cv2.resize(a, (width, int(h * width / w)), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])

def trace(args):
    work = os.path.abspath(args.workdir); os.makedirs(work, exist_ok=True)
    opts = json.load(open(os.path.join(HERE, 'options.json')))['options']
    pr = Progress(work, args.tag, 'trace', 'progress_trace.json')
    pr.d['options'] = [dict(id=o['id'], name=o['name'], note=o.get('note', ''), state='queued', pct=0) for o in opts]
    log = open(os.path.join(work, 'trace.log'), 'a')
    crops = [o for o in opts if o.get('crop', True)]; wholes = [o for o in opts if not o.get('crop', True)]
    leads = [g[0]['id'] for g in (crops, wholes) if g]; rest = [o['id'] for o in crops[1:] + wholes[1:]]
    # overall bar, split by measured time: crop 0-12 %, the first option of each image 12-80 %, the others 80-100 %
    def overall(stage=None):
        pc = {o['id']: o.get('pct', 0) for o in pr.d['options']}
        lead = sum(pc[i] for i in leads) / max(len(leads), 1); rst = sum(pc[i] for i in rest) / max(len(rest), 1) if rest else 100
        n_done = sum(o['state'] == 'done' for o in pr.d['options'])
        pct = 12 + 68 * lead / 100 + 20 * rst / 100 if rest else 12 + 88 * lead / 100
        pr.set(stage=stage or (f'{n_done} of {len(opts)} options ready' if n_done else 'tracing the options'), pct=pct)
    try:
        pr.set(push=True, stage='reading the photo', pct=1)
        src_photo = unseal(args.photo, os.path.join(work, 'in.jpg')) if args.photo.endswith('.enc') else args.photo
        photo = prepare(src_photo, work)
        pr.set(stage='finding the subject and cropping around it', pct=2)
        crop = os.path.join(work, 'crop.jpg')
        t_crop = time.time(); fill = threading.Event()
        def creep():                                         # the crop step logs nothing until it ends: ~9 s measured
            while not fill.wait(1): pr.set(pct=min(11, 2 + 9 * (time.time() - t_crop) / 9))
        threading.Thread(target=creep, daemon=True).start()
        run([PY, 'autocrop.py', photo, crop], log=log); fill.set()
        srcs = {True: crop, False: photo}
        creeping = {}
        def creep_opts():
            while not done_ev.wait(1):
                for oid, st in list(creeping.items()):
                    i = FRACS.index(st['f']) if st['f'] in FRACS else 0
                    if i + 1 >= len(FRACS): continue
                    g = (FRACS[i + 1] - st['f']) * 0.9 * min(1.0, (time.time() - st['t']) / SEGS[i])
                    pr.opt(oid, pct=3 + 82 * (st['f'] + g))
                if creeping: overall()
        done_ev = threading.Event(); threading.Thread(target=creep_opts, daemon=True).start()

        def one(o):
            oid = o['id']; d = os.path.join(work, f'opt_{oid}'); os.makedirs(d, exist_ok=True)
            src = srcs[bool(o.get('crop', True))]; shutil.copy(src, os.path.join(d, 'photo.jpg'))
            pr.opt(oid, state='tracing', pct=3); overall()
            st = creeping[oid] = {'f': 0.0, 't': time.time()}
            def on_line(line):
                for key, frac in MILESTONES:
                    if key in line and frac > st['f']: st['f'], st['t'] = frac, time.time(); pr.opt(oid, pct=3 + 82 * frac); overall(); break
            run([PY, 'explore.py', os.path.join(d, 'photo.jpg'), os.path.join(d, 'line'), *o['style'].split(), 'levels=1.8'], log=log, on_line=on_line)
            creeping.pop(oid, None); pr.opt(oid, state='drawing the still', pct=88); overall()
            run([PY, 'make916.py', os.path.join(d, 'photo.jpg'), os.path.join(d, 'line'), os.path.join(work, f'opt_{oid}.mp4'), '26'],
                env=dict(NO916='1', HUMANIZE='1', STILL='1', HU_ARGS=HU_ARGS), log=log)
            jpg = os.path.join(work, f'still_{oid}.jpg'); preview(os.path.join(work, f'opt_{oid}_still.png'), jpg)
            tgz = os.path.join(work, f'opt_{oid}.tgz')                          # what the render job needs later
            with tarfile.open(tgz, 'w:gz') as t:
                t.add(os.path.join(d, 'photo.jpg'), 'photo.jpg')
                for f in os.listdir(os.path.join(d, 'line')):
                    if f.endswith(('_d1.npy', '_d1.json')): t.add(os.path.join(d, 'line', f), 'line/' + f)
            pr.opt(oid, state='uploading', pct=95); overall()
            upload_private(args.tag, [jpg, tgz])
            pr.opt(oid, state='done', pct=100); overall(); pr.push() if args.tag else None

        # the first option alone: it fills the shared model cache (background removal, line model, objects, pose)
        # that the other options on the same crop then reuse; after that, the rest run side by side
        # (options on the whole photo share nothing with the crop, so the first of those starts at the same time)
        pr.set(stage='reading the photo with the AI models', pct=12)
        by_id = {o['id']: o for o in opts}
        with ThreadPoolExecutor(max(1, os.cpu_count() or 2)) as ex:
            for f in [ex.submit(one, by_id[i]) for i in leads]: f.result()
            for f in [ex.submit(one, by_id[i]) for i in rest]: f.result()
        done_ev.set(); pr.set(push=True, stage='all options ready: pick one', pct=100, done=True)
    except Exception as e:
        pr.set(push=True, stage='failed', error=str(e)[-1500:], done=True); raise
    finally: pr.finish()

def unpack(work, oid, d):
    tgz = os.path.join(work, f'opt_{oid}.tgz')
    if not os.path.exists(tgz) and os.path.exists(tgz + '.enc'): unseal(tgz + '.enc', tgz)
    with tarfile.open(tgz) as t: t.extractall(d, filter='data')

def frames_progress(pr, lo, hi, label):
    state = {}
    def on_line(line):                                       # video.py logs "chunk i: frame f/n" -> overall %
        if 'chunk' in line and 'frame' in line:
            try:
                c = int(line.split('chunk')[1].split(':')[0]); f, n = line.split('frame')[1].split()[0].split('/')
                state[c] = (int(f), int(n)); done = sum(v[0] for v in state.values()); tot = max(sum(v[1] for v in state.values()), 1)
                pr.set(stage=label, pct=int(lo + (hi - lo) * done / tot))
            except Exception: pass
    return on_line

def draw_video(d, out, seconds, pr, lo, hi, caption_png=None, crf='19', ink=1.0):
    env = dict(NO916='1', HUMANIZE='1', HU_ARGS=HU_ARGS, NOIPAD='1', CRF=crf, INK=f'{float(ink):g}')   # INK: line weight (1.0 = the still's)
    if caption_png: env['CAPTION_PNG'] = caption_png
    run([PY, 'make916.py', os.path.join(d, 'photo.jpg'), os.path.join(d, 'line'), out, str(seconds)], env=env,
        log=open(os.path.join(os.path.dirname(d), 'render.log'), 'a'), on_line=frames_progress(pr, lo, hi, 'drawing the video frames'))

def caption_png(text, path):
    sys.path.insert(0, HERE); import caption as CP
    img = np.array(CP.caption_rgba(CP.pick('x', text if text.strip() else '0'), 1080, 1920))
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)); return path

# ---------- comment overlay (optional): a cropped comment screenshot floating over the video ----------
# style = "<pos>-<size>-<fade>[-<align>]": pos top|mid|bottom (card centre at 30 / 50 / 70 % of the frame height, clear of
# the caption and of TikTok's bottom bar), size s|m|l (a thumbnail: 30 / 40 / 52 % of the frame width), fade stay|out
# (out: only at the start, gone by COMMENT_GONE seconds, so it sets up the video and then leaves the drawing alone), align left|center|right (5 % margin at the sides).
# The card gets rounded corners and a soft shadow, fades in at 0.3 s and bobs 8 px up and down every 3.2 s.
COMMENT_GONE = 4.0
CPOS = dict(top=0.30, mid=0.50, bottom=0.70); CSIZE = dict(s=0.30, m=0.40, l=0.52); CALIGN = ('left', 'center', 'right')
def parse_cstyle(st):
    p = (st or 'bottom-s-out-left').split('-') + ['', '', '', '']
    return (p[0] if p[0] in CPOS else 'bottom', p[1] if p[1] in CSIZE else 's', p[2] if p[2] in ('stay', 'out') else 'out',
            p[3] if p[3] in CALIGN else 'center')

def comment_card(src, out, style, W=1080, H=1920):
    pos, size, _, align = parse_cstyle(style)
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    if im is None: raise RuntimeError('comment image could not be read')
    cw = int(W * CSIZE[size]); ch = max(1, int(im.shape[0] * cw / im.shape[1]))
    if ch > 0.30 * H: ch = int(0.30 * H); cw = int(im.shape[1] * ch / im.shape[0])     # very tall crops: limit height
    im = cv2.resize(im, (cw, ch), interpolation=cv2.INTER_AREA)
    pad = 24; r = max(10, int(cw * 0.045))
    card = np.zeros((ch + 2 * pad, cw + 2 * pad, 4), np.uint8)
    mask = np.zeros((ch, cw), np.uint8)
    cv2.rectangle(mask, (r, 0), (cw - r, ch), 255, -1); cv2.rectangle(mask, (0, r), (cw, ch - r), 255, -1)
    for cx, cy in ((r, r), (cw - r - 1, r), (r, ch - r - 1), (cw - r - 1, ch - r - 1)): cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    sh = np.zeros(card.shape[:2], np.float32); sh[pad + 6:pad + 6 + ch, pad:pad + cw] = mask / 255.0
    sh = cv2.GaussianBlur(sh, (0, 0), 9) * 0.5                                         # soft drop shadow
    card[:, :, 3] = (sh * 255).astype(np.uint8)
    a = mask[:, :, None] / 255.0; reg = card[pad:pad + ch, pad:pad + cw]
    reg[:, :, :3] = (im * a).astype(np.uint8); reg[:, :, 3] = np.maximum(reg[:, :, 3], mask)
    cv2.imwrite(out, card)
    y = int(np.clip(int(CPOS[pos] * H - card.shape[0] / 2), int(0.16 * H), int(0.82 * H) - card.shape[0]))
    m = int(0.05 * W) - pad
    x = m if align == 'left' else W - card.shape[1] - m if align == 'right' else (W - card.shape[1]) // 2
    return out, x, y

def overlay_pass(raw, capf, final, seconds, pr, comment=None, cstyle=''):
    # one encode: caption (static) + optional comment card (fade in, gentle bob, optional fade out)
    ins = ['-i', raw, '-i', capf]; fc = '[0][1]overlay=0:0'
    if comment:
        card, x, y = comment_card(comment, os.path.splitext(capf)[0] + '_comment.png', cstyle)
        dur = float(seconds) + 4; fade = 'fade=t=in:st=0.3:d=0.4:alpha=1'
        if parse_cstyle(cstyle)[2] == 'out': fade += f',fade=t=out:st={COMMENT_GONE - 0.6:.2f}:d=0.6:alpha=1'
        ins += ['-loop', '1', '-t', f'{dur:.2f}', '-i', card]
        fc += f'[v];[2]format=rgba,{fade}[c];[v][c]overlay=x={x}:y={y}+8*sin(2*PI*t/3.2):shortest=1'
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', *ins, '-filter_complex', fc, '-c:v', 'libx264', '-preset', 'veryfast',
                          '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-progress', 'pipe:1', final], stdout=subprocess.PIPE, text=True)
    total_us = float(seconds + 4) * 1e6
    for line in p.stdout:                                  # ffmpeg reports out_time_us as it encodes
        if line.startswith('out_time_us='):
            try: pr.set(stage='adding the caption to the video', pct=int(20 + 70 * min(1, float(line.split('=')[1]) / total_us)))
            except ValueError: pass
    if p.wait(): raise RuntimeError('caption pass failed')

def secs_tag(x): return f'{float(x):g}'

def master_name(work, oid, secs, ink): return os.path.join(work, f'raw_{oid}_{secs_tag(secs)}_{float(ink):g}.mp4')

def prerender(args):
    # several options per runner, one after the other (each uses all the runner's cores)
    work = os.path.abspath(args.workdir)
    for oid in args.option.split(','):
        d = os.path.join(work, f'pre_{oid}'); os.makedirs(d, exist_ok=True)
        unpack(work, oid, d)
        raw = master_name(work, oid, args.seconds, args.ink)
        class Quiet:                                           # no progress file: nobody is waiting on this one
            def set(self, **kw): pass
        draw_video(d, raw, args.seconds, Quiet(), 0, 100, crf='12', ink=args.ink)
        upload_private(args.tag, [raw])
        print(f'[studio] master ready: {os.path.basename(raw)}', flush=True)

def render(args):
    work = os.path.abspath(args.workdir); oid = args.option; pr = Progress(work, args.tag, 'render', f'progress_render_{oid}.json')
    pr.d['option'] = oid
    d = os.path.join(work, f'render_{oid}'); os.makedirs(d, exist_ok=True)
    try:
        final = os.path.join(work, f'video_{oid}.mp4'); capf = caption_png(args.caption, os.path.join(work, f'caption_{oid}.png'))
        raw = master_name(work, oid, args.seconds, args.ink)
        if not os.path.exists(raw) and os.path.exists(raw + '.enc'): unseal(raw + '.enc', raw)
        comment = None
        if args.comment:                                       # encrypted comment crop from the phone
            enc = os.path.join(work, args.comment)
            if not os.path.exists(enc): raise RuntimeError(f'comment image {args.comment} missing')
            comment = enc[:-4]; unseal(enc, comment)
        if not os.path.exists(raw):                            # no master for this speed: draw one first (no caption)
            pr.set(push=True, stage='unpacking the chosen drawing', pct=3)
            unpack(work, oid, d)
            pr.set(stage='drawing the video frames', pct=5)
            draw_video(d, raw, args.seconds, pr, 5, 80, crf='12', ink=args.ink)
            pr.set(push=True, stage='adding the caption to the video', pct=80)
            class Tail:                                        # the overlay pass reports 20..90 %; here it is 80..92 %
                def set(self, **kw): pr.set(**{**kw, 'pct': 80 + (kw.get('pct', 20) - 20) * 12 // 70})
            overlay_pass(raw, capf, final, args.seconds, Tail(), comment, args.cstyle)
        else:                                                  # pre-rendered master: only the caption pass
            pr.set(push=True, stage='adding the caption to the pre-rendered video', pct=20)
            overlay_pass(raw, capf, final, args.seconds, pr, comment, args.cstyle)
        pr.set(stage='uploading the video', pct=94)
        upload_private(args.tag, [final])
        pr.set(push=True, stage='video ready', pct=100, done=True, video=os.path.basename(final))
    except Exception as e:
        pr.set(push=True, stage='failed', error=str(e)[-1500:], done=True); raise
    finally: pr.finish()

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    a = sp.add_parser('trace'); a.add_argument('photo'); a.add_argument('workdir'); a.add_argument('--tag')
    b = sp.add_parser('render'); b.add_argument('workdir'); b.add_argument('option'); b.add_argument('caption')
    b.add_argument('--seconds', type=float, default=26); b.add_argument('--ink', type=float, default=1.0); b.add_argument('--tag')
    b.add_argument('--comment', default='', help='encrypted comment crop asset (comment_*.png.enc) in the workdir'); b.add_argument('--cstyle', default='')
    c = sp.add_parser('prerender'); c.add_argument('workdir'); c.add_argument('option', help='one id or several: a,e'); c.add_argument('--seconds', type=float, default=26); c.add_argument('--ink', type=float, default=1.0); c.add_argument('--tag')
    args = ap.parse_args(); dict(trace=trace, render=render, prerender=prerender)[args.cmd](args)
