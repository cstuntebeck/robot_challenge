"""Abgabe-Video mit Titeltafeln und Bildunterschriften (Claas, 23.11.).
  python abgabe_video2.py <schnitt.json> <ziel.mp4>
schnitt.json: Liste von {"title": "...", "sub": "...", "s": 3} (Tafel) oder
{"clip": "pfad.mp4", "caption": "...", "max_s": 40} (Clip, auf 1280x720 eingepasst, 25 fps).
Fehlende Clips werden uebersprungen und gemeldet."""
import json, os, sys
import numpy as np
import imageio.v2 as iio
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 25


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    try:
        return ImageFont.load_default(size=sz)
    except TypeError:
        return ImageFont.load_default()


F_T, F_S, F_C = font(46), font(28), font(24)


def card(title, sub):
    im = Image.new("RGB", (W, H), (18, 32, 58))
    d = ImageDraw.Draw(im)
    d.text((80, 270), title, font=F_T, fill=(255, 255, 255))
    y = 350
    for line in sub.split("\n"):
        d.text((80, y), line, font=F_S, fill=(190, 205, 230))
        y += 40
    d.text((80, 640), "RobotChallenge IMES - Gruppe A - Unitree G1 (23 DoF)", font=F_C, fill=(120, 140, 170))
    return np.asarray(im)


def fit(frame, caption):
    im = Image.fromarray(frame[..., :3])
    im.thumbnail((W, H))
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    canvas.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
    if caption:
        d = ImageDraw.Draw(canvas, "RGBA")
        d.rectangle([0, H - 56, W, H], fill=(0, 0, 0, 170))
        d.text((24, H - 44), caption, font=F_C, fill=(255, 255, 255))
    return np.asarray(canvas)


spec = json.load(open(sys.argv[1]))
out = sys.argv[2]
wr = iio.get_writer(out, fps=FPS, codec="libx264", quality=7, macro_block_size=8)
total = 0
for seg in spec:
    if "clip" not in seg:
        img = card(seg.get("title", ""), seg.get("sub", ""))
        for _ in range(int(seg.get("s", 3) * FPS)):
            wr.append_data(img)
        total += seg.get("s", 3)
        continue
    p = seg["clip"]
    if not os.path.exists(p):
        print("FEHLT", p)
        continue
    r = iio.get_reader(p)
    src_fps = r.get_meta_data().get("fps", 25) or 25
    step = max(1, round(src_fps / FPS))
    n = 0
    for i, fr in enumerate(r):
        if i % step:
            continue
        wr.append_data(fit(fr, seg.get("caption", "")))
        n += 1
        if n >= seg.get("max_s", 1e9) * FPS:
            break
    total += n / FPS
    print("CLIP", p, round(n / FPS, 1), "s (Quelle", src_fps, "fps)")
wr.close()
print("ABGABE_OK", out, round(total, 1), "s")
