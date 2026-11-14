# Automatic request list (Nov 2026): reads the comments on the account's videos, finds the ones asking to be drawn,
# and gets each one ready to post: profile picture -> automatic check (pfpcheck.py) -> trace -> video with the pfp
# caption and a card of their comment at the start. Runs on GitHub Actions every 2 hours (requests.yml).
#
# State lives in release "requests" of this (public) repo, encrypted with LS_KEY like every other image:
#   requests.json.enc   written here: one entry per request comment
#                       {cid, vid, handle, nick, text, ts, thumb (64 px jpeg, base64), status, reason, job, rid, found}
#                       status: skipped (reason; job set when the trace ran) | failed (reason) | ready (job + rid -> video_t_<rid>.mp4)
#   posted.json.enc     written only by the phone page: {"posted": [cid, ...]}
#   each request's job is an ordinary release job-<id> (named "request", nothing personal in clear text), so the phone's
#   job view, other options and re-renders work as for any photo.
# Usage: python3 requests_scan.py [--dry] [--test <account> "<text>"]
#   --dry: list the requests found, process nothing; --test: add that account as a request (end-to-end test)
import os, sys, re, json, time, base64, subprocess, unicodedata, uuid, datetime, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import studio as ST, pfpcheck as K
CFG = json.load(open(os.path.join(HERE, 'requests_config.json')))
REL = 'requests'; WORK = os.path.abspath('reqwork'); os.makedirs(WORK, exist_ok=True)
T0 = time.time()
def log(*a): print(f'[requests {time.time() - T0:6.1f}s]', *a, flush=True)

# ---------- TikTok reading (public pages and the public comment list, as a browser would) ----------
def http(url, **kw):
    from curl_cffi import requests
    for i in range(3):
        try:
            r = requests.get(url, impersonate='chrome', timeout=30, headers={'Accept-Language': 'en-GB,en'}, **kw)
            if r.status_code == 200: return r
        except Exception as e: log('retry', i + 1, type(e).__name__)
        time.sleep(2 + 3 * i)
    return None

def videos(acc):
    ytdlp = os.path.join(os.path.dirname(sys.executable), 'yt-dlp'); ytdlp = ytdlp if os.path.exists(ytdlp) else 'yt-dlp'   # the venv's copy
    out = subprocess.run([ytdlp, '--impersonate', 'chrome', '--flat-playlist', '--playlist-end', '200', '--print', '%(id)s %(comment_count)s',
                          f'https://www.tiktok.com/@{acc}'], capture_output=True, text=True).stdout.split('\n')
    return {a: (int(b) if b.isdigit() else 0) for a, b in (l.split()[:2] for l in out if len(l.split()) >= 2)}

def comments(vid):
    cur, out = 0, []
    while len(out) < 500:
        r = http('https://www.tiktok.com/api/comment/list/', params=dict(aid=1988, aweme_id=vid, count=50, cursor=cur))
        if not r: break
        try: d = r.json()
        except Exception: break
        out += d.get('comments') or []
        if not d.get('has_more'): break
        cur = d.get('cursor', cur + 50); time.sleep(0.7)
    return out

def big_avatar(unique_id):
    r = http(f'https://www.tiktok.com/@{unique_id}')
    m = r and re.search(r'<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__"[^>]*>(.*?)</script>', r.text, re.S)
    if not m: return None
    u = json.loads(m.group(1))['__DEFAULT_SCOPE__'].get('webapp.user-detail', {}).get('userInfo', {}).get('user', {})
    return u.get('avatarLarger') or u.get('avatarMedium')

def plain(t): return ''.join(c for c in unicodedata.normalize('NFKD', t.lower()) if not unicodedata.combining(c))
PHRASES = [re.compile(r'(?<!\w)' + re.escape(plain(p)) + r'(?!\w)') for p in CFG['phrases']]
def is_request(text): t = plain(text); return any(p.search(t) for p in PHRASES)

# ---------- state (encrypted release assets) ----------
def load(name, default):
    p = os.path.join(WORK, name)
    if subprocess.run(['gh', 'release', 'download', REL, '-p', name, '-D', WORK, '--clobber'], capture_output=True).returncode: return default
    return json.loads(open(ST.unseal(p, p[:-4]), 'rb').read())
def save(state):
    p = os.path.join(WORK, 'requests.json'); json.dump(state, open(p, 'w'))
    ST.upload(REL, [ST.seal(p)])

# ---------- the comment card shown at the start of the video ----------
def card(avatar, handle, text, path):
    from PIL import Image, ImageDraw, ImageFont
    def font(bold, size):
        f = '/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf' % ('-Bold' if bold else '')
        return ImageFont.truetype(f, size) if os.path.exists(f) else ImageFont.load_default(size)
    text = ''.join(c for c in text if ord(c) < 0x2190 or 0x3000 <= ord(c) < 0xFFF0).strip() or '...'   # emoji the font lacks
    W, pad, av = 1000, 34, 104; fh, ft = font(True, 34), font(False, 42)
    words, lines, cur = text.split(), [], ''
    for w in words:
        t = (cur + ' ' + w).strip()
        if ImageDraw.Draw(Image.new('RGB', (1, 1))).textlength(t, font=ft) > W - 3 * pad - av and cur: lines.append(cur); cur = w
        else: cur = t
    lines = (lines + [cur])[:3]
    H = max(av + 2 * pad, pad + 48 + 54 * len(lines) + pad)
    im = Image.new('RGB', (W, H), (31, 41, 46)); d = ImageDraw.Draw(im)
    a = Image.fromarray(cv2.cvtColor(cv2.resize(avatar, (av, av), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    m = Image.new('L', (av, av), 0); ImageDraw.Draw(m).ellipse((0, 0, av - 1, av - 1), fill=255); im.paste(a, (pad, pad), m)
    x = pad * 2 + av; d.text((x, pad), handle, font=fh, fill=(160, 166, 172))
    for i, l in enumerate(lines): d.text((x, pad + 48 + 54 * i), l, font=ft, fill=(240, 240, 240))
    im.save(path); return path

# ---------- one request -> ready video ----------
def process(e, avatar, f):
    # -> (status, reason, job, rid). The job release is made first, so even a skipped trace can be opened on the phone.
    jid = datetime.datetime.utcnow().strftime('%Y%m%d-%H%M-') + uuid.uuid4().hex[:4]; tag = 'job-' + jid
    w = os.path.join(WORK, jid, 'work'); os.makedirs(w, exist_ok=True)
    photo = os.path.join(WORK, jid, 'photo.jpg'); cv2.imwrite(photo, avatar, [cv2.IMWRITE_JPEG_QUALITY, 95]); ST.seal(photo)
    subprocess.run(['gh', 'release', 'create', tag, photo + '.enc', '--title', tag, '--prerelease',
                    '--notes', json.dumps(dict(created=datetime.datetime.utcnow().isoformat() + 'Z', name='request'))], check=True, capture_output=True)
    py = sys.executable; v = CFG['video']
    subprocess.run([py, os.path.join(HERE, 'studio.py'), 'trace', photo + '.enc', w, '--tag', tag, '--only', CFG['options']], check=True)
    ok, why, _ = K.after(os.path.join(w, 'opt_' + v['option'], 'line'), f)        # the drawing would be a tangle: no video
    if not ok: return 'skipped', why, jid, ''
    cname = 'comment_' + uuid.uuid4().hex[:8] + '.png'; card(avatar, '@' + e['handle'], e['text'], os.path.join(w, cname))
    ST.upload_private(tag, [os.path.join(w, cname)]); rid = uuid.uuid4().hex[:6]
    subprocess.run([py, os.path.join(HERE, 'studio.py'), 'render', w, v['option'], v['caption'], '--seconds', str(v['seconds']),
                    '--comment', cname + '.enc', '--cstyle', v['comment_style'], '--rid', rid, '--tag', tag], check=True)
    return 'ready', '', jid, rid

def main(dry=False):
    acc = CFG['account']; state = load('requests.json.enc', {'requests': [], 'videos': {}})
    known = {e['cid'] for e in state['requests']}; vids = videos(acc); log(f'{len(vids)} videos on @{acc}')
    new = []
    for vid, n in vids.items():
        if n == 0 or state['videos'].get(vid) == n: continue                # nothing new under this video since last run
        for c in comments(vid):
            u = c.get('user') or {}; cid = c.get('cid'); text = c.get('text') or ''
            if not cid or cid in known or u.get('unique_id') == acc or not is_request(text): continue
            new.append(dict(cid=cid, vid=vid, handle=u.get('unique_id', ''), nick=u.get('nickname', ''), text=text[:300],
                            ts=c.get('create_time', 0), status='new', found=int(time.time()))); known.add(cid)
        state['videos'][vid] = n; time.sleep(0.8)
    if 'TEST' in globals(): new.append(TEST)
    log(f'{len(new)} new requests')
    if dry:
        for e in new: log('request:', e['text'][:80])
        return
    state['requests'] += new; save(state)
    todo = [e for e in state['requests'] if e['status'] == 'new'][:int(CFG['max_per_run'])]
    for e in todo:
        try:
            url = big_avatar(e['handle']); r = url and http(url)
            im = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR) if r else None
            if im is None: e.update(status='skipped', reason='profile picture could not be loaded'); continue
            h, w_ = im.shape[:2]; s = min(h, w_); im = im[(h - s) // 2:(h + s) // 2, (w_ - s) // 2:(w_ + s) // 2]
            im = cv2.resize(im, (720, 720), interpolation=cv2.INTER_AREA if s > 720 else cv2.INTER_CUBIC)
            e['thumb'] = base64.b64encode(cv2.imencode('.jpg', cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).decode()
            if min(h, w_) < 200: e.update(status='skipped', reason='profile picture too small'); continue
            ok, why, f = K.before(im)
            if not ok: e.update(status='skipped', reason=why); continue
            st, why, jid, rid = process(e, im, f)
            e.update(status=st, reason=why, job=jid, rid=rid); log(st, jid, why)
        except Exception as ex:
            e.update(status='failed', reason=str(ex)[-200:]); log('failed:', ex)
        finally: save(state)

if __name__ == '__main__':
    # --test <account> "<text>": also treat that account as having commented <text> (an end-to-end test on a known picture)
    if '--test' in sys.argv:
        i = sys.argv.index('--test'); TEST = dict(cid='test-' + uuid.uuid4().hex[:8], vid='', handle=sys.argv[i + 1], nick='', text=sys.argv[i + 2],
                                                   ts=int(time.time()), status='new', found=int(time.time()), test=True)
    main(dry='--dry' in sys.argv)
