"""Draw site/og.png (1200 x 630): the landing's reserves wedge, from _src/data/oil.json.

Needs Pillow. Run after the data changes: python _src/og_card.py
"""
import json
import pathlib

from PIL import Image, ImageDraw, ImageFont

SRC = pathlib.Path(__file__).resolve().parent
rows = json.loads((SRC / "data" / "oil.json").read_text(encoding="utf-8"))["reserves"]["rows"][:10]
PAPER, INK, RED = (243, 238, 227), (21, 21, 21), (200, 16, 46)


def font(size, bold=True):
    for n in (["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default()


im = Image.new("RGB", (1200, 630), PAPER)
d = ImageDraw.Draw(im)
d.rectangle([0, 0, 1200, 14], fill=INK)
d.text((48, 34), "WHO HOLDS THE WORLD'S OIL?", font=font(52), fill=INK)
d.text((48, 96), "It depends who is counting. Bars: US EIA. Ticks: OPEC (above), BP (below).", font=font(24, False), fill=INK)
vmax = max(max(v for v in (r["eia"], r["opec"], r["bp"]) if v) for r in rows)
x0, x1, y0, rh = 300, 1040, 150, 46
for i, r in enumerate(rows):
    y = y0 + i * rh
    w = (x1 - x0) * r["eia"] / vmax
    d.text((x0 - 16 - d.textlength(r["country"], font=font(24)), y + 6), r["country"], font=font(24), fill=INK)
    d.rectangle([x0, y + 8, x0 + w, y + 34], fill=RED if r["opec_member"] else INK)
    for src, a, b in (("opec", y, y + 21), ("bp", y + 21, y + 42)):
        if r[src]:
            tx = x0 + (x1 - x0) * r[src] / vmax
            d.line([tx, a, tx, b], fill=PAPER, width=8)
            d.line([tx, a, tx, b], fill=INK, width=3)
    label = f"{r['eia'] / 1000:,.0f} bn"
    if r["opec"] and r["eia"] / r["opec"] > 3:
        label += f"  OPEC {r['opec'] / 1000:.1f} bn"
    end = max([x0 + w] + [x0 + (x1 - x0) * r[s] / vmax for s in ("opec", "bp") if r[s]])
    d.text((end + 14, y + 8), label, font=font(22), fill=RED if "OPEC" in label else INK)
im.save(SRC.parent / "site" / "og.png", optimize=True)
print("site/og.png")
