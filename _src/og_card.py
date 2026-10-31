"""Draw site/og.png (1200 x 630): the landing poster's six figures, read from the built site/index.html.

Needs Pillow. Run after _src/build.py: python _src/og_card.py
"""
import html
import pathlib
import re

from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parent.parent
page = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
tiles = [(html.unescape(t), html.unescape(f), html.unescape(c)) for t, f, c in
         re.findall(r'<span class="t-topic">(.*?)</span><span class="t-fig">(.*?)</span>.*?<span class="t-claim">(.*?)</span>', page)]
PAPER, INK, RED = (243, 238, 227), (21, 21, 21), (200, 16, 46)


def font(size, bold=True):
    for n in (["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default()


def wrap(d, text, f, width):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = t
    return lines + [cur]


im = Image.new("RGB", (1200, 630), INK)
d = ImageDraw.Draw(im)
d.rectangle([0, 0, 1200, 92], fill=PAPER)
d.rectangle([40, 30, 66, 56], fill=RED)
d.text((80, 24), "THE ECONOMY IN SIX NUMBERS", font=font(40), fill=INK)
cw, ch, x0, y0, gap = 392, 262, 0, 98, 12
for i, (topic, fig, claim) in enumerate(tiles):
    x = 6 + (i % 3) * (cw + 6)
    y = y0 + (i // 3) * (ch + 6)
    d.rectangle([x, y, x + cw - 1, y + ch - 1], fill=PAPER)
    d.text((x + 20, y + 16), topic.upper(), font=font(20), fill=RED)
    d.text((x + 18, y + 42), fig, font=font(78), fill=INK)
    for j, line in enumerate(wrap(d, claim, font(20, False), cw - 40)[:3]):
        d.text((x + 20, y + 150 + j * 26), line, font=font(20, False), fill=INK)
im.save(ROOT / "site" / "og.png", optimize=True)
print("site/og.png", len(tiles), "tiles")
