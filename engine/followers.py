# The account's newest followers, read as the account itself (TikTok shows a follower list only to the logged-in owner).
#
# The login: the phone page saves the TikTok cookies (a Cookie-Editor JSON export from Firefox on Android), encrypted with
# LS_KEY, as tt_session.json.enc in release "requests". It is never written anywhere else and never printed.
# A real browser (Playwright Chromium) opens the profile with those cookies and opens the Followers list; TikTok's own page
# fetches the list (it signs its requests itself), and the answers are read as they arrive.
#
# Output: followers.json.enc in release "requests":
#   {"status": "ok" | "no-login" | "login-expired" | "error", "checked": unix time, "msg": "...",
#    "list": [{"uid", "unique_id", "nickname"}, ...] newest first}
# engine/requests_scan.py reads it: new followers become requests, their display names are learned for screenshots.
import os, sys, json, time, base64, subprocess, tempfile

REL = 'requests'
ACC = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requests_config.json')))['account']
KEY = base64.b64decode(os.environ['LS_KEY'])
WORK = tempfile.mkdtemp()
MAX = 200
T0 = time.time()
def log(*a): print(f'[followers {time.time() - T0:5.1f}s]', *a, flush=True)

def aes(b, enc):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    if enc: n = os.urandom(12); return n + AESGCM(KEY).encrypt(n, b, None)
    return AESGCM(KEY).decrypt(b[:12], b[12:], None)

def download(name):
    if subprocess.run(['gh', 'release', 'download', REL, '-p', name, '-D', WORK, '--clobber'], capture_output=True).returncode: return None
    return aes(open(os.path.join(WORK, name), 'rb').read(), False)

def upload(name, data):
    p = os.path.join(WORK, name); open(p, 'wb').write(aes(data, True))
    r = subprocess.run(['gh', 'release', 'upload', REL, p, '--clobber'], capture_output=True, text=True)
    if r.returncode: log('upload failed', r.stderr[:200])

def cookies(raw):
    # Cookie-Editor JSON export (a list), or a "name=value; name2=value2" header string
    raw = raw.decode('utf-8', 'replace').strip(); out = []
    if raw.startswith('['):
        same = {'no_restriction': 'None', 'none': 'None', 'lax': 'Lax', 'strict': 'Strict'}
        for c in json.loads(raw):
            if 'tiktok.com' not in (c.get('domain') or ''): continue
            k = dict(name=c['name'], value=c['value'], domain=c['domain'], path=c.get('path') or '/', secure=bool(c.get('secure')), httpOnly=bool(c.get('httpOnly')))
            if c.get('expirationDate'): k['expires'] = float(c['expirationDate'])
            if (c.get('sameSite') or '').lower() in same: k['sameSite'] = same[c['sameSite'].lower()]
            if k.get('sameSite') == 'None': k['secure'] = True
            out.append(k)
    else:
        for part in raw.split(';'):
            if '=' in part:
                n, v = part.strip().split('=', 1); out.append(dict(name=n, value=v, domain='.tiktok.com', path='/', secure=True))
    return out

def read(cs):
    from playwright.sync_api import sync_playwright
    got, seen = [], set()
    def on_response(r):
        if '/api/user/list/' not in r.url: return
        try: d = r.json()
        except Exception: return
        for it in d.get('userList') or []:
            u = it.get('user') or {}
            if u.get('uniqueId') and u['uniqueId'] not in seen:
                seen.add(u['uniqueId']); got.append(dict(uid=str(u.get('id', '')), unique_id=u['uniqueId'], nickname=u.get('nickname', '')))
    with sync_playwright() as p:
        b = p.chromium.launch(args=['--disable-blink-features=AutomationControlled'])
        ctx = b.new_context(viewport={'width': 1280, 'height': 900}, locale='en-GB',
                            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36')
        ctx.add_cookies(cs); pg = ctx.new_page(); pg.on('response', on_response)
        try:
            pg.goto(f'https://www.tiktok.com/@{ACC}', wait_until='domcontentloaded', timeout=60000); pg.wait_for_timeout(6000)
            if pg.locator('[data-e2e="top-login-button"]').count(): return 'login-expired', 'TikTok shows the page as logged out', got, pg.screenshot()
            pg.locator('[data-e2e="followers-count"]').first.click(timeout=15000); pg.wait_for_timeout(4000)
            tab = pg.locator('[role="dialog"] >> text=/^Followers/').first           # the list opens on the tab clicked; make sure
            if tab.count(): tab.click(); pg.wait_for_timeout(3000)
            n, still = 0, 0
            while len(got) < MAX and still < 4:                                    # scroll the list: each page of 30 loads as it is reached
                box = pg.locator('[role="dialog"]').first.bounding_box()
                if box: pg.mouse.move(box['x'] + box['width'] / 2, box['y'] + box['height'] * 0.7)
                pg.mouse.wheel(0, 3000); pg.wait_for_timeout(1500)
                still = still + 1 if len(got) == n else 0; n = len(got)
            if not got: return 'error', 'the follower list did not load', got, pg.screenshot()
            return 'ok', '', got, None
        except Exception as e:
            return 'error', f'{type(e).__name__}: {str(e)[:160]}', got, pg.screenshot()
        finally: b.close()

def main():
    raw = b'sessionid=smoke-test-not-a-login' if os.environ.get('SMOKE') == 'true' else download('tt_session.json.enc')   # smoke: a fake login, the real one untouched
    if not raw:
        upload('followers.json.enc', json.dumps(dict(status='no-login', checked=int(time.time()), msg='no TikTok login saved', list=[])).encode()); log('no login saved'); return
    cs = cookies(raw); log(f'{len(cs)} cookies, session {"present" if any(c["name"] == "sessionid" for c in cs) else "MISSING"}')
    st, msg, got, shot = read(cs)
    log(st, msg, f'{len(got)} followers read')
    if os.environ.get('SMOKE') == 'true':
        if shot: open('followers_debug.png', 'wb').write(shot)       # logged-out page only: kept as a run artifact
        return
    if shot: upload('followers_debug.png.enc', shot)                     # what the browser saw, encrypted (for fixing)
    upload('followers.json.enc', json.dumps(dict(status=st, checked=int(time.time()), msg=msg, list=got)).encode())

if __name__ == '__main__': main()
