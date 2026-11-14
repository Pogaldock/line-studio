# Automatic request list (Nov 2026): reads the comments on the account's videos, finds the ones asking to be drawn,
# and gets each one ready to post: profile picture -> automatic check (pfpcheck.py) -> trace -> video with the pfp
# caption and a card of their comment at the start. Runs on GitHub Actions every 2 hours (requests.yml).
#
# State lives in release "requests" of this (public) repo, encrypted with LS_KEY like every other image:
#   requests.json.enc   written here: one entry per request comment
#                       {cid, vid, handle, nick, text, ts, thumb (64 px jpeg, base64), status, reason, job, rid, found}
#                       status: skipped (reason; job set when the trace ran) | failed (reason) | ready (job + rid -> video_t_<rid>.mp4)
#   posted.json.enc     written only by the phone page: {"posted": [cid, ...]}
#   shot_*/add_*.enc    the phone's inbox (screenshots, typed @names), read and removed by the scan (see inbox())
#   status needs-name: a screenshot comment whose @handle could not be confirmed; the page asks for it
#   each request's job is an ordinary release job-<id> (named "request", nothing personal in clear text), so the phone's
#   job view, other options and re-renders work as for any photo.
# Usage: python3 requests_scan.py [--dry] [--redo] [--test <account> "<text>"]
#   --redo: process every request not marked posted again (after a change to the look), removing test entries
#   --scan-only / --process-only: the two halves, for the 10-minute workflow (light scan, heavy job only when needed)
#   --dry: list the requests found, process nothing; --test: add that account as a request (end-to-end test)
import os, sys, re, json, time, base64, subprocess, unicodedata, uuid, datetime, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import studio as ST, pfpcheck as K, shotread as SR   # numpy/cv2 at the top only (OCR loads on first use): fine in the light environment
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

# ---------- inbox: comment screenshots and typed @names from the phone page (assets of release "requests") ----------
#   shot_<id>.png.enc / .jpg.enc   a screenshot of comments: every comment in it becomes a request (except the account's own)
#   add_<id>.json.enc               {"handles": ["name", ...], "assign": {"<request id>": "name"}, "redo": ["<request id>"]}:
#                                   typed @names, @names given to screenshot requests the matcher could not place, and
#                                   requests to remake from scratch (new job)
# each file is removed once read.
def profile(h):
    r = http(f'https://www.tiktok.com/@{h}')
    m = r and re.search(r'<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__"[^>]*>(.*?)</script>', r.text, re.S)
    if not m: return None
    u = json.loads(m.group(1))['__DEFAULT_SCOPE__'].get('webapp.user-detail', {}).get('userInfo', {}).get('user', {})
    if not u.get('uniqueId'): return None
    a = http(u.get('avatarThumb') or u.get('avatarMedium') or '')
    return dict(nickname=u.get('nickname', ''), avatar=cv2.imdecode(np.frombuffer(a.content, np.uint8), cv2.IMREAD_COLOR) if a else None, uid=u.get('id', ''))

def inbox(state, known, people):
    r = subprocess.run(['gh', 'release', 'view', REL, '--json', 'assets', '-q', '.assets[].name'], capture_output=True, text=True)
    files = [n for n in r.stdout.split() if re.fullmatch(r'(shot_[a-z0-9]{6,12}\.(png|jpg)|add_[a-z0-9]{6,12}\.json)\.enc', n)]
    new, byid = [], {e['cid']: e for e in state['requests']}
    for n in files:
        p = os.path.join(WORK, n)
        if subprocess.run(['gh', 'release', 'download', REL, '-p', n, '-D', WORK, '--clobber'], capture_output=True).returncode: continue
        try:
            raw = ST.unseal(p, p[:-4])
            if n.startswith('shot_'):
                img = cv2.imread(raw); found = SR.read(img) if img is not None else []
                log(f'{n}: {len(found)} comments read')
                for c in found:
                    cid = 'shot-' + uuid.uuid5(uuid.NAMESPACE_OID, SR._plain(c['nick']) + '|' + c['text']).hex[:12]
                    if cid in known: continue
                    h, why = SR.resolve(c['nick'], c['avatar'], profile, state.get('names'))
                    if h and h == CFG['account']: continue
                    if h and h in people: log(f'  {c["nick"]}: already in the list as @{h}'); continue
                    thumb = base64.b64encode(cv2.imencode('.jpg', cv2.resize(c['avatar'], (64, 64)), [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).decode() if c['avatar'] is not None else ''
                    e = dict(cid=cid, vid='', handle=h or '', nick=c['nick'], text=c['text'][:300], ts=int(time.time()), found=int(time.time()), source='follower' if c.get('follower') else 'screenshot', test=n.startswith('shot_test'),
                             status='new' if h else 'needs-name', reason='' if h else why, thumb=thumb)
                    log(f'  {c["nick"]}: {"@" + h if h else "needs a @name"} ({why})')
                    new.append(e); known.add(cid); people.add(h) if h else None
            else:
                d = json.load(open(raw))
                for h in d.get('handles', []):
                    h = h.strip().lstrip('@').lower()
                    if not re.fullmatch(r'[a-z0-9._]{2,24}', h) or h in people: continue
                    new.append(dict(cid='name-' + h, vid='', handle=h, nick='', text='', ts=int(time.time()), found=int(time.time()), source='typed', status='new', test=n.startswith('add_test')))
                    known.add('name-' + h); people.add(h); log(f'typed: @{h}')
                for cid in d.get('redo') or []:                # remake from scratch (new job: video + all the options)
                    e = byid.get(cid)
                    if e and e.get('handle'): e.update(status='new', reason='', job='', rid='', more=False, more_tries=0, gone=False); log(f'redo: @{e["handle"]}')
                for cid, h in (d.get('assign') or {}).items():
                    h = h.strip().lstrip('@').lower(); e = byid.get(cid)
                    if e and re.fullmatch(r'[a-z0-9._]{2,24}', h): e.update(handle=h, status='new', reason=''); people.add(h); log(f'@name given: @{h}')
        except Exception as ex: log(f'{n}: could not be read ({ex})')
        subprocess.run(['gh', 'release', 'delete-asset', REL, n, '-y'], capture_output=True)
    return new

# ---------- the comment card shown at the start of the video ----------
def card(avatar, handle, text, path):
    from PIL import Image, ImageDraw, ImageFont
    def font(bold, size):
        f = '/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf' % ('-Bold' if bold else '')
        return ImageFont.truetype(f, size) if os.path.exists(f) else ImageFont.load_default(size)
    text = ''.join(c for c in text if ord(c) < 0x2190 or 0x3000 <= ord(c) < 0xFFF0).strip()   # emoji the font lacks
    W, pad, av = 860, 30, 112; fh, ft = font(True, 40), font(False, 58)          # big text: the card is shown small
    words, lines, cur = text.split(), [], ''
    for w in words:
        t = (cur + ' ' + w).strip()
        if ImageDraw.Draw(Image.new('RGB', (1, 1))).textlength(t, font=ft) > W - 3 * pad - av and cur: lines.append(cur); cur = w
        else: cur = t
    lines = [l for l in (lines + [cur])[:3] if l]
    if not lines: fh = font(True, 52)                                              # a typed @name only: just the name, bigger
    meas = ImageDraw.Draw(Image.new('RGB', (1, 1)))           # card as wide as its text: a short comment stays big on screen
    W = int(min(W, max(420, 3 * pad + av + max([meas.textlength(handle, font=fh)] + [meas.textlength(l, font=ft) for l in lines]))))
    H = max(av + 2 * pad, pad + 54 + 70 * len(lines) + pad) if lines else av + 2 * pad
    im = Image.new('RGB', (W, H), (31, 41, 46)); d = ImageDraw.Draw(im)
    a = Image.fromarray(cv2.cvtColor(cv2.resize(avatar, (av, av), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    m = Image.new('L', (av, av), 0); ImageDraw.Draw(m).ellipse((0, 0, av - 1, av - 1), fill=255); im.paste(a, (pad, pad), m)
    x = pad * 2 + av; d.text((x, pad if lines else pad + (av - 52) // 2), handle, font=fh, fill=(160, 166, 172) if lines else (240, 240, 240))
    for i, l in enumerate(lines): d.text((x, pad + 54 + 70 * i), l, font=ft, fill=(240, 240, 240))
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
    e.update(comment=cname + '.enc', cstyle=v['comment_style'], caption=v['caption'])
    subprocess.run([py, os.path.join(HERE, 'studio.py'), 'render', w, v['option'], v['caption'], '--seconds', str(v['seconds']),
                    '--comment', cname + '.enc', '--cstyle', v['comment_style'], '--rid', rid, '--tag', tag], check=True)
    return 'ready', '', jid, rid

def main(dry=False):
    acc = CFG['account']; state = load('requests.json.enc', {'requests': [], 'videos': {}})
    state['account'] = acc; state['scanned'] = int(time.time()); state['post_caption'] = CFG.get('post_caption', '')
    names = state.setdefault('names', {}); hidden = state.setdefault('hidden', {})   # hidden: {video: comments TikTok kept from the scan}
    known = {e['cid'] for e in state['requests']}; people = {e.get('uid') or e['handle'] for e in state['requests']}; vids = {} if PROCESS_ONLY else videos(acc); log(f'{len(vids)} videos on @{acc}')
    new = []
    for vid, n in vids.items():
        if n == 0 or (state['videos'].get(vid) == n and state.get('hidden_counted')): continue   # nothing new under this video
        cs = comments(vid)
        for c in cs:
            u = c.get('user') or {}; cid = c.get('cid'); text = c.get('text') or ''
            if u.get('nickname') and u.get('unique_id'):          # display name -> @handle, for reading screenshots later
                hs = names.setdefault(SR._plain(u['nickname']), [])
                if u['unique_id'] not in hs: hs.append(u['unique_id'])
            uid = u.get('uid') or u.get('unique_id')
            if not cid or cid in known or uid in people or u.get('unique_id') == acc or not is_request(text): continue
            people.add(uid)
            new.append(dict(cid=cid, vid=vid, handle=u.get('unique_id', ''), nick=u.get('nickname', ''), text=text[:300],
                            ts=c.get('create_time', 0), status='new', found=int(time.time()), uid=uid)); known.add(cid)
        got = len(cs) + sum(int(c.get('reply_comment_total') or 0) for c in cs)
        if n > got: hidden[vid] = n - got
        else: hidden.pop(vid, None)
        state['videos'][vid] = n; time.sleep(0.8)
    state['hidden_counted'] = True                     # first run with hidden counts reads every video once
    if 'TEST' in globals(): new.append(TEST)
    if '--clean-tests' in sys.argv:                     # remove every entry made by a test run
        n0 = len(state['requests']); state['requests'] = [e for e in state['requests'] if not e.get('test')]; log(f'removed {n0 - len(state["requests"])} test entries')
    if TEST_INBOX:                                      # end-to-end test: a mock screenshot (stock faces, made-up names) + a typed @name
        shot = os.path.join(WORK, 'shot_test' + uuid.uuid4().hex[:4] + '.png'); import shutil; shutil.copy(os.path.join(HERE, '..', 'tools', 'test', 'comments_mock.png'), shot)
        add = os.path.join(WORK, 'add_test' + uuid.uuid4().hex[:4] + '.json'); json.dump({'handles': [TEST_INBOX]}, open(add, 'w'))
        ST.upload(REL, [ST.seal(shot), ST.seal(add)])
    if not PROCESS_ONLY: new += inbox(state, known, people)
    log(f'{len(new)} new requests')
    if dry:
        for e in new: log('request:', e['text'][:80])
        return
    if REDO:                                            # --redo: drop test entries, process every unposted request again
        posted = set((load('posted.json.enc', {}) or {}).get('posted', []))
        state['requests'] = [e for e in state['requests'] if not e.get('test')]
        for e in state['requests']:
            if e['cid'] not in posted: e.update(status='new', reason='', job='', rid='')
    state['requests'] += new; save(state)
    todo = [e for e in state['requests'] if e['status'] == 'new'][:int(CFG['max_per_run'])]
    extra = [e for e in state['requests'] if e.get('job') and e['status'] in ('ready', 'skipped') and not e.get('more') and e.get('more_tries', 0) < 2] if CFG.get('more_options') else []
    if SCAN_ONLY:                                       # light job: tell the workflow whether the drawing job is needed
        log(f'{len(todo)} waiting to be drawn, {len(extra)} waiting for more options')
        if os.environ.get('GITHUB_OUTPUT'): open(os.environ['GITHUB_OUTPUT'], 'a').write(f'pending={len(todo) + len(extra)}\n')
        return
    for e in todo:
        try:
            pf = profile(e['handle'])                     # learn their display name: later screenshots of them match at once
            if pf and pf.get('nickname'):
                hs = state.setdefault('names', {}).setdefault(SR._plain(pf['nickname']), [])
                if e['handle'] not in hs: hs.append(e['handle'])
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
            if jid and CFG.get('more_options'):              # video first; then more drawings to choose from, in the same job
                save(state); w = os.path.join(WORK, jid, 'work')
                r = subprocess.run([sys.executable, os.path.join(HERE, 'studio.py'), 'trace', os.path.join(WORK, jid, 'photo.jpg.enc'), w, '--tag', 'job-' + jid,
                                    '--only', CFG['more_options'], '--append'])
                log('more options', 'ready' if r.returncode == 0 else 'failed', jid); e['more'] = r.returncode == 0
        except Exception as ex:
            e.update(status='failed', reason=str(ex)[-200:]); log('failed:', ex)
        finally: save(state)
    for e in extra[:int(CFG['max_per_run'])]:          # earlier requests made before more_options existed (or whose extra trace failed)
        jid = e['job']; w = os.path.join(WORK, jid, 'work'); os.makedirs(w, exist_ok=True)
        for n in ('photo.jpg.enc', 'progress_trace.json'):
            subprocess.run(['gh', 'release', 'download', 'job-' + jid, '-p', n, '-D', w if n.endswith('.json') else os.path.join(WORK, jid), '--clobber'], capture_output=True)
        if not os.path.exists(os.path.join(WORK, jid, 'photo.jpg.enc')):    # the job was deleted from the phone: the page offers Redo
            e.update(gone=True, more_tries=2); log('job deleted:', jid); save(state); continue
        r = subprocess.run([sys.executable, os.path.join(HERE, 'studio.py'), 'trace', os.path.join(WORK, jid, 'photo.jpg.enc'), w, '--tag', 'job-' + jid,
                            '--only', CFG['more_options'], '--append'])
        e['more'] = r.returncode == 0; e['more_tries'] = e.get('more_tries', 0) + 1; log('more options for an earlier request:', 'ready' if e['more'] else 'failed', jid); save(state)

REDO = '--redo' in sys.argv
SCAN_ONLY = '--scan-only' in sys.argv          # read comments and update the list only (no models needed)
PROCESS_ONLY = '--process-only' in sys.argv    # draw what is waiting, without reading comments again
TEST_INBOX = sys.argv[sys.argv.index('--test-inbox') + 1] if '--test-inbox' in sys.argv else ''   # --test-inbox <account>
if __name__ == '__main__':
    # --test <account> "<text>": also treat that account as having commented <text> (an end-to-end test on a known picture)
    if '--test' in sys.argv:
        i = sys.argv.index('--test'); TEST = dict(cid='test-' + uuid.uuid4().hex[:8], vid='', handle=sys.argv[i + 1], nick='', text=sys.argv[i + 2],
                                                   ts=int(time.time()), status='new', found=int(time.time()), test=True)
    main(dry='--dry' in sys.argv)
