# Read TikTok comment screenshots (Nov 2026): every comment in the picture -> display name, text and the commenter's
# avatar cut out of the screenshot (used to confirm their @name, never drawn: it is too small).
#
# How a comment looks in TikTok's comment sheet (any language, light or dark): a round avatar at the left, the display name
# just right of it, the comment text in the lines below at the same indent, then a line with the age ("7h", "2d",
# "il y a 3 h") and "Reply"/"Répondre". Replies use the same layout, indented further. RapidOCR (ONNX, Apache-2.0) finds the
# text lines; a comment is a name line, the lines below it at the same
# indent until the age/reply line (the avatar is cut out when it stands out from the sheet). The post's own caption
# ("Name · Creator") and TikTok's interface text are skipped.
# Usage: python3 shotread.py <screenshot> [...]
import re, sys, numpy as np, cv2

UI = re.compile(r'^(comments?|commentaires?|likes?|j\'aime|views?|vues?|replying to|réponse à|en réponse à|add comment|ajouter un commentaire|'
                r'view \d+ repl|voir \d+ réponse|hide|masquer|reply|répondre|responder|\d+[,.]?\d*\s*[km]?)\b', re.I)
TAIL = re.compile(r'(^|\s)(\d+\s?(s|m|min|h|d|j|w|sem|mo|y)|\d{1,2}[-/]\d{1,2}|il y a|ago|hier|yesterday|reply|répondre|responder)(\s|$)', re.I)
CREATOR = re.compile(r'(creator|créateur|créatrice|creador|autor)', re.I)
_OCR = {}

def ocr(img):
    if 'o' not in _OCR:
        from rapidocr_onnxruntime import RapidOCR
        _OCR['o'] = RapidOCR()
    res, _ = _OCR['o'](img)
    out = []
    for box, text, conf in (res or []):
        b = np.array(box); out.append(dict(x0=float(b[:, 0].min()), x1=float(b[:, 0].max()), y0=float(b[:, 1].min()), y1=float(b[:, 1].max()), t=text.strip(), c=float(conf)))
    return sorted(out, key=lambda r: (r['y0'], r['x0']))

def avatar_left_of(img, line):
    # the round avatar left of a name line: a compact blob, roughly square, in the strip left of the text
    H, W = img.shape[:2]; h = line['y1'] - line['y0']
    x1 = int(line['x0'] - 0.012 * W); x0 = max(0, int(line['x0'] - 0.16 * W)); y0 = max(0, int(line['y0'] - 0.6 * h)); y1 = min(H, int(line['y0'] + 3.5 * h))
    if x1 - x0 < 20 or y1 - y0 < 20: return None
    reg = img[y0:y1, x0:x1]; bg = np.median(np.concatenate([reg[:, :3].reshape(-1, 3), reg[:, -3:].reshape(-1, 3)]), 0)
    m = (np.abs(reg.astype(int) - bg).max(2) > 22).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)), 8)
    best = None
    for i in range(1, n):
        x, y, w, hh, a = st[i]
        if w < 0.035 * W or hh < 0.035 * W or not (0.75 < w / hh < 1.33) or a < 0.5 * w * hh: continue
        if best is None or a > best[4]: best = (x, y, w, hh, a)
    if best is None: return None
    x, y, w, hh, _ = best; s = max(w, hh)
    return img[y0 + y:y0 + y + s, x0 + x:x0 + x + s].copy()

def read(img):
    # -> list of {nick, text, avatar (BGR crop or None), y}. A comment = a name line, text lines at the same indent, then
    # the age / reply line at that indent. The avatar is optional (a dark picture on a dark sheet may not stand out).
    H, W = img.shape[:2]; lines = [l for l in ocr(img) if l['y0'] > 0.04 * H and l['c'] > 0.5]
    out, i = [], 0
    while i < len(lines):
        l = lines[i]
        if UI.match(l['t']) or (TAIL.search(l['t']) and len(l['t']) < 16): i += 1; continue
        body, j = [], i + 1
        while j < len(lines):
            m = lines[j]
            if abs(m['x0'] - l['x0']) > 0.05 * W or m['y0'] - lines[j - 1]['y1'] > 0.08 * H: break
            if TAIL.search(m['t']) or UI.match(m['t']): break
            body.append(m['t']); j += 1
        # the age / reply line: within the next few lines (an attached photo leaves a gap), at or a little right of the indent
        last = lines[j - 1]['y1'] if body else l['y1']
        ends = any(TAIL.search(m['t']) and m['y0'] - last < 0.25 * H and l['x0'] - 0.05 * W <= m['x0'] <= l['x0'] + 0.3 * W for m in lines[j:j + 3])
        if not (body and ends): i += 1; continue
        nick = l['t']; creator = bool(CREATOR.search(nick)); nick = re.split(r'\s*[·•]\s*', nick)[0].strip()
        if not creator: out.append(dict(nick=nick, text=' '.join(body), avatar=avatar_left_of(img, l), y=int(l['y0'])))
        i = j + 1
    return out

if __name__ == '__main__':
    for f in sys.argv[1:]:
        for c in read(cv2.imread(f)): print(f'{c["nick"]!r}: {c["text"]!r}  avatar {None if c["avatar"] is None else c["avatar"].shape[:2]}  at y={c["y"]}')

# ---------- display name -> @handle ----------
# TikTok screenshots show the display name, not the @handle. Candidates: handles already seen with that display name
# (known), then handles built from the name itself (accents and symbols dropped; spaces removed, or as _ or .). Each
# candidate's public profile is fetched; it is accepted when its display name matches and its profile picture looks like
# the avatar cut from the screenshot (or, with no avatar cut, on the name alone), or on a very close picture alone.
import unicodedata
def _plain(t): return re.sub(r'[^a-z0-9._ ]', '', ''.join(c for c in unicodedata.normalize('NFKD', t.lower()) if not unicodedata.combining(c))).strip()

def candidates(nick, known=None):
    out = list((known or {}).get(_plain(nick), []))
    b = _plain(nick)
    if b:
        for c in (b.replace(' ', ''), b.replace(' ', '_'), b.replace(' ', '.')):
            if c and c not in out and re.fullmatch(r'[a-z0-9._]{2,24}', c): out.append(c)
    return out[:6]

def similarity(a, b):
    # normalised correlation of two avatars (colour, 40 px, inside the circle)
    a = cv2.resize(a, (40, 40), interpolation=cv2.INTER_AREA).astype(np.float32); b = cv2.resize(b, (40, 40), interpolation=cv2.INTER_AREA).astype(np.float32)
    yy, xx = np.mgrid[:40, :40]; m = (yy - 19.5) ** 2 + (xx - 19.5) ** 2 < 17 ** 2
    a, b = a[m].ravel(), b[m].ravel(); a -= a.mean(); b -= b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))

def resolve(nick, avatar, profile, known=None):
    # profile(handle) -> dict(nickname, avatar BGR) or None (fetching is left to the caller). -> (handle or None, note)
    for h in candidates(nick, known):
        p = profile(h)
        if not p: continue
        same_name = _plain(p['nickname']) == _plain(nick)
        sim = similarity(avatar, p['avatar']) if avatar is not None and p.get('avatar') is not None else None
        if same_name and (sim is None or sim > 0.5): return h, f'name match{"" if sim is None else f", picture {sim:.2f}"}'
        if sim is not None and sim > 0.85: return h, f'picture {sim:.2f}'
    return None, 'no account found for this name'
