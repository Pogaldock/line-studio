# Burn a TikTok-style caption (white bold text, black outline, top of frame) into a finished video.
# Every video gets one of the "addicted to doing lines" variants; pick by index or let the video name choose.
# Usage: python3 caption.py <in.mp4> <out.mp4> [variant_index | "custom text"]
#        python3 caption.py --mock <frame.png> <out.png> [variant_index]   (still mockup, no video)
import sys, os, json, hashlib, subprocess, textwrap, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

VARIANTS = [
    "dr help im addicted to doing lines",
    "day 47 of being addicted to doing lines",
    "i cant stop doing lines",
    "my therapist says i need to stop doing lines",
    "one more line and ill stop i promise",
    "mum found out im doing lines",
    "pov: you've been doing lines since 6am",
    "doing lines again sorry",
    "its not a problem if its art",
    "i only do lines on weekends (lie)",
]
FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'Inter-ExtraBold.otf')

def pick(name, sel=None):
    if sel is None: return VARIANTS[int(hashlib.md5(name.encode()).hexdigest(), 16) % len(VARIANTS)]
    return VARIANTS[int(sel) % len(VARIANTS)] if str(sel).lstrip('-').isdigit() else sel

def caption_rgba(text, W, H):
    # text block ~78% of the frame width, top 9% of the frame (clear of TikTok's own top bar). A second part after a
    # line break ("caption\ncall to action") is set smaller under it, the whole block a size smaller and starting at 12%
    # (under TikTok's header); it still ends above 20% of the frame, where the drawing's safe area begins (video.py SAFE).
    main, _, sub = text.partition('\n')
    size = int(W * (0.044 if sub.strip() else 0.052)); font = ImageFont.truetype(FONT, size)   # two-part captions: a size smaller
    img = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    y = int(H * (0.12 if sub.strip() else 0.09))      # two-part: below TikTok's Following / For You header
    for txt, sz, wrap in ((main, size, 36), (sub.strip(), int(size * 0.66), 40)):
        if not txt: continue
        f = font if sz == size else ImageFont.truetype(FONT, sz); stroke = max(3, sz // 9)
        for ln in textwrap.wrap(txt, width=wrap):
            w = d.textlength(ln, font=f); d.text(((W - w) / 2, y), ln, font=f, fill=(255, 255, 255, 255), stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
            y += int(sz * 1.22)
        y += int(size * 0.15)
    return img

def main():
    if sys.argv[1] == '--mock':
        fr = cv2.imread(sys.argv[2]); H, W = fr.shape[:2]
        cap = caption_rgba(pick(sys.argv[2], sys.argv[4] if len(sys.argv) > 4 else None), W, H)
        base = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).convert('RGBA'); base.alpha_composite(cap)
        cv2.imwrite(sys.argv[3], cv2.cvtColor(np.array(base.convert('RGB')), cv2.COLOR_RGB2BGR)); return
    src, out = sys.argv[1], sys.argv[2]; text = pick(os.path.basename(src), sys.argv[3] if len(sys.argv) > 3 else None)
    pr = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'json', src], capture_output=True, text=True)
    st = json.loads(pr.stdout)['streams'][0]; W, H = st['width'], st['height']
    png = os.path.splitext(out)[0] + '_cap.png'; caption_rgba(text, W, H).save(png)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', src, '-i', png, '-filter_complex', 'overlay=0:0',
                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '19', '-pix_fmt', 'yuv420p', '-c:a', 'copy', out], check=True)
    os.remove(png); print(f'[caption] "{text}" -> {out}', flush=True)

if __name__ == '__main__': main()
