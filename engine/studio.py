# Line Studio orchestrator: one photo in -> several drawing options (stills) -> the chosen one as a captioned video.
#
#   python3 studio.py trace  <photo> <workdir> [--tag job-xxx]
#   python3 studio.py render <workdir> <option_id> <caption text | variant index> [--seconds 26] [--tag job-xxx]
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
    # progress.json: overall stage + one entry per option; written locally and (rate limited) to the release
    # one file per run so several runs on the same photo never overwrite each other:
    # progress_trace.json, progress_render_<option>.json
    def __init__(s, work, tag, kind, fname):
        s.work, s.tag, s.lock, s.fname = work, tag, threading.Lock(), fname; s.last_push = 0.0; s.t0 = time.time()
        s.d = dict(kind=kind, stage='starting', pct=0, options=[], started=int(s.t0), updated=int(s.t0), error=None, done=False)
    def set(s, push=False, **kw):
        with s.lock:
            s.d.update(kw); s.d['updated'] = int(time.time()); s.d['elapsed'] = round(time.time() - s.t0, 1)
            p = os.path.join(s.work, s.fname); json.dump(s.d, open(p, 'w'))
            if s.tag and (push or time.time() - s.last_push > 4): s.last_push = time.time(); upload(s.tag, [p])
        print(f"[studio {s.d['elapsed']:6.1f}s] {s.d['stage']} {s.d['pct']}%", flush=True)
    def opt(s, oid, **kw):
        for o in s.d['options']:
            if o['id'] == oid: o.update(kw)

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
    try:
        pr.set(push=True, stage='reading the photo', pct=2)
        src_photo = unseal(args.photo, os.path.join(work, 'in.jpg')) if args.photo.endswith('.enc') else args.photo
        photo = prepare(src_photo, work)
        pr.set(stage='finding the subject and cropping around it', pct=5)
        crop = os.path.join(work, 'crop.jpg')
        run([PY, 'autocrop.py', photo, crop], log=log)
        srcs = {True: crop, False: photo}

        def one(o):
            oid = o['id']; d = os.path.join(work, f'opt_{oid}'); os.makedirs(d, exist_ok=True)
            src = srcs[bool(o.get('crop', True))]; shutil.copy(src, os.path.join(d, 'photo.jpg'))
            pr.opt(oid, state='tracing', pct=10); pr.set()
            run([PY, 'explore.py', os.path.join(d, 'photo.jpg'), os.path.join(d, 'line'), *o['style'].split(), 'levels=1.8'], log=log)
            pr.opt(oid, state='drawing the still', pct=80); pr.set()
            run([PY, 'make916.py', os.path.join(d, 'photo.jpg'), os.path.join(d, 'line'), os.path.join(work, f'opt_{oid}.mp4'), '26'],
                env=dict(NO916='1', HUMANIZE='1', STILL='1', HU_ARGS=HU_ARGS), log=log)
            jpg = os.path.join(work, f'still_{oid}.jpg'); preview(os.path.join(work, f'opt_{oid}_still.png'), jpg)
            tgz = os.path.join(work, f'opt_{oid}.tgz')                          # what the render job needs later
            with tarfile.open(tgz, 'w:gz') as t:
                t.add(os.path.join(d, 'photo.jpg'), 'photo.jpg')
                for f in os.listdir(os.path.join(d, 'line')):
                    if f.endswith(('_d1.npy', '_d1.json')): t.add(os.path.join(d, 'line', f), 'line/' + f)
            upload_private(args.tag, [jpg, tgz])
            pr.opt(oid, state='done', pct=100)
            n_done = sum(x['state'] == 'done' for x in pr.d['options'])
            pr.set(push=True, stage=f'{n_done} of {len(opts)} options ready', pct=int(15 + 85 * n_done / len(opts)))

        # the first option alone: it fills the shared model cache (background removal, line model, objects, pose)
        # that the other options on the same crop then reuse; after that, the rest run side by side
        # (options on the whole photo share nothing with the crop, so the first of those starts at the same time)
        crops = [o for o in opts if o.get('crop', True)]; wholes = [o for o in opts if not o.get('crop', True)]
        pr.set(stage='reading the photo with the AI models', pct=10)
        with ThreadPoolExecutor(max(1, os.cpu_count() or 2)) as ex:
            leads = [ex.submit(one, g[0]) for g in (crops, wholes) if g]
            for f in leads: f.result()
            for f in [ex.submit(one, o) for o in crops[1:] + wholes[1:]]: f.result()
        pr.set(push=True, stage='all options ready: pick one', pct=100, done=True)
    except Exception as e:
        pr.set(push=True, stage='failed', error=str(e)[-1500:], done=True); raise

def render(args):
    work = os.path.abspath(args.workdir); oid = args.option; pr = Progress(work, args.tag, 'render', f'progress_render_{oid}.json')
    pr.d['option'] = oid
    d = os.path.join(work, f'render_{oid}'); os.makedirs(d, exist_ok=True); log = open(os.path.join(work, 'render.log'), 'a')
    try:
        pr.set(push=True, stage='unpacking the chosen drawing', pct=3)
        tgz = os.path.join(work, f'opt_{oid}.tgz')
        if not os.path.exists(tgz) and os.path.exists(tgz + '.enc'): unseal(tgz + '.enc', tgz)
        with tarfile.open(tgz) as t: t.extractall(d, filter='data')
        final = os.path.join(work, f'video_{oid}.mp4')
        pr.set(stage='drawing the video frames', pct=8)
        state = {}
        def on_line(line):                                   # "chunk i: frame f/n" -> overall %
            if 'chunk' in line and 'frame' in line:
                try:
                    c = int(line.split('chunk')[1].split(':')[0]); f, n = line.split('frame')[1].split()[0].split('/')
                    state[c] = (int(f), int(n)); done = sum(v[0] for v in state.values()); tot = max(sum(v[1] for v in state.values()), 1)
                    pr.set(stage='drawing the video frames', pct=int(8 + 80 * done / tot))
                except Exception: pass
        sys.path.insert(0, HERE); import caption as CP
        text = CP.pick('x', args.caption if args.caption.strip() else '0')
        capf = os.path.join(d, 'caption.png'); img = np.array(CP.caption_rgba(text, 1080, 1920))
        cv2.imwrite(capf, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
        run([PY, 'make916.py', os.path.join(d, 'photo.jpg'), os.path.join(d, 'line'), final, str(args.seconds)],
            env=dict(NO916='1', HUMANIZE='1', HU_ARGS=HU_ARGS, NOIPAD='1', CAPTION_PNG=capf), log=log, on_line=on_line)
        pr.set(stage='uploading the video', pct=96)
        upload_private(args.tag, [final])
        pr.set(push=True, stage='video ready', pct=100, done=True, video=os.path.basename(final))
    except Exception as e:
        pr.set(push=True, stage='failed', error=str(e)[-1500:], done=True); raise

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    a = sp.add_parser('trace'); a.add_argument('photo'); a.add_argument('workdir'); a.add_argument('--tag')
    b = sp.add_parser('render'); b.add_argument('workdir'); b.add_argument('option'); b.add_argument('caption')
    b.add_argument('--seconds', type=float, default=26); b.add_argument('--tag')
    args = ap.parse_args(); (trace if args.cmd == 'trace' else render)(args)
