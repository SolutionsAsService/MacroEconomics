"""The wider-economy pages: economies, people, prices, markets, money, jobs, climate, the country explorer.

Every number a page prints is computed here from _src/data/macro_*.json (built by clean_macro.py), so
nothing can drift from the data; check_figures.py recomputes the headline figures independently.
Imported by build.py.
"""
import json
import math
import pathlib
from collections import defaultdict

SRC = pathlib.Path(__file__).resolve().parent
_ = lambda n: json.loads((SRC / "data" / f"{n}.json").read_text(encoding="utf-8"))
CD = _("macro_countries")
MK = _("macro_markets")
MREP = _("macro_repairs")
COUNTRIES, AGGS = CD["countries"], CD["aggregates"]
BY = {c["code"]: c for c in COUNTRIES}
WORLD = next(a for a in AGGS if a["code"] == "WLD")

import html as _html
e = _html.escape


# ---------- reading the packed series

def at(c, key, y):
    s = c["s"].get(key)
    if not s:
        return None
    i = y - s["y0"]
    return s["v"][i] if 0 <= i < len(s["v"]) else None


def last(c, key, upto=None):
    s = c["s"].get(key)
    if not s:
        return None
    for i in range(len(s["v"]) - 1, -1, -1):
        y = s["y0"] + i
        if upto is not None and y > upto:
            continue
        if s["v"][i] is not None:
            return y, s["v"][i]
    return None


def span(c, key):
    s = c["s"].get(key)
    return (s["y0"], s["y0"] + len(s["v"]) - 1) if s else None


def latest_year(key, min_n=180):
    n = defaultdict(int)
    for c in COUNTRIES:
        s = c["s"].get(key)
        if s:
            for i, v in enumerate(s["v"]):
                if v is not None:
                    n[s["y0"] + i] += 1
    return max(y for y, k in n.items() if k >= min_n)


GDP_Y, POP_Y, INF_Y, CO2_Y = latest_year("gdp"), latest_year("pop"), latest_year("infl"), latest_year("co2pc")


def ranked(key, y, pred=lambda c: True):
    out = [(c, at(c, key, y)) for c in COUNTRIES if at(c, key, y) is not None and pred(c)]
    return sorted(out, key=lambda t: -t[1])


def gdp_pc(c, y):
    g, p = at(c, "gdp", y), at(c, "pop", y)
    return g / p if g and p else None


# ---------- number formats

def usd(v):
    a = abs(v)
    if a >= 1e12:
        return f"${v / 1e12:,.1f} tn"
    if a >= 1e9:
        return f"${v / 1e9:,.0f} bn" if a >= 1e10 else f"${v / 1e9:,.1f} bn"
    return f"${v / 1e6:,.0f} m"


def people(v):
    if v >= 1e9:
        return f"{v / 1e9:,.2f} bn"
    if v >= 1e6:
        return f"{v / 1e6:,.1f} m" if v < 1e8 else f"{v / 1e6:,.0f} m"
    return f"{v / 1e3:,.0f} k"


def pct(v, d=1, sign=False):
    return f"{v:+,.{d}f}%" if sign else f"{v:,.{d}f}%"


def rate(v):
    return f"{v:,.2f}" if v < 1000 else f"{v:,.0f}"


def mo(k):
    """'2026-09' to a decimal year."""
    return int(k[:4]) + (int(k[5:7]) - 1) / 12


# ---------- charts. Every chart has a wide version and a tall one for phones (CSS swaps them).

def bars(rows, aria, fmt=lambda v: f"{v:,.0f}", signed=False, pid="b"):
    """rows: dicts {label, value, red?, note?}. Draws a wide and a tall horizontal bar chart."""
    out = []
    for tall in (False, True):
        W = 400 if tall else 1000
        x0 = 150 if tall else 230
        rh = 38 if tall else 34
        bh = 18 if tall else 22
        right = 70 if tall else 150
        lo = min([0] + [r["value"] for r in rows]) if signed else 0
        hi = max(r["value"] for r in rows)
        scale = (W - x0 - right) / ((hi - lo) or 1)
        zero = x0 - lo * scale
        parts = [f'<line class="axis" x1="{zero:.1f}" x2="{zero:.1f}" y1="0" y2="{len(rows) * rh}"/>'] if signed else []
        for i, r in enumerate(rows):
            y = i * rh
            v = r["value"]
            w = abs(v) * scale
            x = zero if v >= 0 else zero - w
            cls = "bar opec" if r.get("red") else "bar non"
            style = f' style="fill:url(#{pid}-pat-{"t" if tall else "w"})"' if r.get("red") else ""
            tx = zero + w + 8 if v >= 0 else zero + 8
            anchor = "start"
            lab_x = (x0 - 10) if not signed else x0 - 10
            parts.append(f'<text class="lbl{" sm" if tall else ""}" x="{lab_x}" y="{y + bh - 3}" text-anchor="end">{e(r["label"])}</text>'
                         f'<rect class="{cls}" x="{x:.1f}" y="{y + 4}" width="{max(w, 1.5):.1f}" height="{bh}"{style}/>'
                         f'<text class="val{" sm" if tall else ""}" x="{tx:.1f}" y="{y + bh - 3}" text-anchor="{anchor}">{e(fmt(v))}{e(r.get("note", ""))}</text>')
        defs = (f'<defs><pattern id="{pid}-pat-{"t" if tall else "w"}" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
                f'<rect class="hatch-bg" width="7" height="7"/><rect class="hatch-ln" width="2.2" height="7"/></pattern></defs>')
        out.append(f'<svg class="mbars {"tall" if tall else "wide"}" viewBox="0 0 {W} {len(rows) * rh + 4}" role="img" aria-label="{e(aria)}" '
                   f'>{defs}{"".join(parts)}</svg>')
    return f'<div class="dual">{"".join(out)}</div>'


DASH = ["", "12 6", "3 5", "16 5 3 5", "6 4"]


def nice_ticks(lo, hi, n=5):
    span_ = hi - lo
    step = 10 ** math.floor(math.log10(span_ / n))
    for m in (1, 2, 2.5, 5, 10):
        if span_ / (step * m) <= n:
            step *= m
            break
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-9:
        out.append(round(t, 10))
        t += step
    return out


def lines(series, aria, *, lfmt=None, yfmt=lambda v: f"{v:,.0f}", xfmt=lambda x: f"{x:.0f}", log=False, ylo=None, yhi=None, gap=2.0,
          xticks=None, yticks=None, hline=None):
    """series: dicts {name, pts[(x, y)], red?}. Wide chart with end labels, tall chart for phones; HTML legend follows."""
    xs = [p[0] for s in series for p in s["pts"]]
    ys = [p[1] for s in series for p in s["pts"] if p[1] is not None and (p[1] > 0 or not log)]
    xmin, xmax = min(xs), max(xs)
    f = (lambda v: math.log10(v)) if log else (lambda v: v)
    lo = f(ylo if ylo is not None else min(ys))
    hi = f(yhi if yhi is not None else max(ys))
    if ylo is None and not log:
        lo = min(lo, 0)
    if log:
        yticks = yticks or [10 ** k for k in range(math.floor(lo), math.ceil(hi) + 1)]
        lo, hi = min(lo, f(min(yticks))), max(hi, f(max(yticks)))
    else:
        yticks = yticks or nice_ticks(lo, hi)
        hi = max(hi, f(yticks[-1])) if yticks[-1] - hi < (hi - lo) * .05 else hi
    xticks = xticks or nice_ticks(xmin, xmax, 6)
    out = []
    for tall in (False, True):
        W, H = (400, 380) if tall else (1000, 440)
        L, R, T, B = (58, 14, 14, 34) if tall else (74, 210, 16, 36)
        X = lambda x: L + (W - L - R) * (x - xmin) / ((xmax - xmin) or 1)
        Y = lambda v: H - B - (H - T - B) * (f(v) - lo) / ((hi - lo) or 1)
        parts = [f'<line class="axis" x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}"/>']
        for t in yticks:
            parts.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/><text class="tick-l{" sm" if tall else ""}" x="{L - 8}" y="{Y(t) + 5:.1f}" text-anchor="end">{e(yfmt(t))}</text>')
        for t in xticks:
            parts.append(f'<text class="tick-l{" sm" if tall else ""}" x="{X(t):.1f}" y="{H - B + 22}" text-anchor="middle">{e(xfmt(t))}</text>')
        if hline is not None:
            parts.append(f'<line class="refline" x1="{L}" x2="{W - R}" y1="{Y(hline):.1f}" y2="{Y(hline):.1f}"/>')
        label_y = {}
        prev = -99.0
        order = sorted(range(len(series)), key=lambda i: Y(series[i]["pts"][-1][1]))
        for i in order:
            label_y[i] = max(Y(series[i]["pts"][-1][1]) + 5, prev + 20)
            prev = label_y[i]
        for i, s in enumerate(series):
            segs, cur = [], []
            for x, y in s["pts"]:
                if y is None or (log and y <= 0):
                    continue
                if cur and x - cur[-1][0] > gap:
                    segs.append(cur)
                    cur = []
                cur.append((x, y))
            if cur:
                segs.append(cur)
            cls = "ln red" if s.get("red") else "ln"
            for seg in segs:
                if len(seg) == 1:
                    parts.append(f'<circle class="ln-dot" cx="{X(seg[0][0]):.1f}" cy="{Y(seg[0][1]):.1f}" r="3"/>')
                else:
                    d = "M" + " L".join(f"{X(x):.1f} {Y(y):.1f}" for x, y in seg)
                    parts.append(f'<path class="{cls}" stroke-dasharray="{DASH[i % len(DASH)]}" d="{d}"/>')
            if not tall:
                lx, ly = s["pts"][-1]
                parts.append(f'<text class="lbl small{" redtxt" if s.get("red") else ""}" x="{W - R + 10}" y="{label_y[i]:.1f}">{e(s["name"])} {e((lfmt or yfmt)(ly))}</text>')
        out.append(f'<svg class="mlines {"tall" if tall else "wide"}" viewBox="0 0 {W} {H}" role="img" aria-label="{e(aria)}">{"".join(parts)}</svg>')
    legend = ""
    if len(series) > 1:
        legend = '<ul class="legend lines-legend" aria-label="Lines">' + "".join(
            f'<li><svg viewBox="0 0 36 10" width="36" height="10" aria-hidden="true"><line class="ln{" red" if s.get("red") else ""}" stroke-dasharray="{DASH[i % len(DASH)]}" x1="0" x2="36" y1="5" y2="5"/></svg>{e(s["name"])}</li>'
            for i, s in enumerate(series)) + "</ul>"
    return f'<div class="dual">{"".join(out)}</div>{legend}'


def scatter(points, aria, xlabel, ylabel, label_codes=()):
    """points: dicts {x, y, code, name}. Log x. One wide chart that scales down."""
    W, H, L, R, T, B = 1000, 520, 80, 20, 16, 56
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    lx0, lx1 = math.log10(min(xs)), math.log10(max(xs))
    ymax = max(ys)
    yt = nice_ticks(0, ymax, 5)
    ymax = max(ymax, yt[-1])
    X = lambda v: L + (W - L - R) * (math.log10(v) - lx0) / (lx1 - lx0)
    Y = lambda v: H - B - (H - T - B) * v / ymax
    parts = [f'<line class="axis" x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}"/>']
    for t in yt:
        parts.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/><text class="tick-l" x="{L - 8}" y="{Y(t) + 5:.1f}" text-anchor="end">{t:g}</text>')
    for k in range(math.ceil(lx0), math.floor(lx1) + 1):
        parts.append(f'<line class="grid" x1="{X(10 ** k):.1f}" x2="{X(10 ** k):.1f}" y1="{T}" y2="{H - B}"/><text class="tick-l" x="{X(10 ** k):.1f}" y="{H - B + 22}" text-anchor="middle">${10 ** k:,.0f}</text>')
    parts.append(f'<text class="axis-t" x="{(L + W - R) / 2}" y="{H - 8}" text-anchor="middle">{e(xlabel)}</text>')
    parts.append(f'<text class="axis-t" transform="translate(18 {(T + H - B) / 2}) rotate(-90)" text-anchor="middle">{e(ylabel)}</text>')
    for p in points:
        hl = p["code"] in label_codes
        parts.append(f'<circle class="dot{" hl" if hl else ""}" cx="{X(p["x"]):.1f}" cy="{Y(p["y"]):.1f}" r="{5.5 if hl else 3.6}"><title>{e(p["name"])}</title></circle>')
    for p in points:
        if p["code"] in label_codes:
            parts.append(f'<text class="lbl small" x="{X(p["x"]) + 9:.1f}" y="{Y(p["y"]) - 8:.1f}">{e(p["name"])}</text>')
    return f'<div class="scroll-x" tabindex="0" role="region" aria-label="Scatter chart, scrolls sideways on a phone"><svg class="scatter" viewBox="0 0 {W} {H}" role="img" aria-label="{e(aria)}">{"".join(parts)}</svg></div>'


def histogram(buckets, aria):
    """buckets: [(label, count, red?)]. Vertical columns."""
    W, H, L, B, T = 1000, 320, 40, 52, 30
    top = max(c for _, c, _ in buckets)
    bw = (W - L) / len(buckets)
    parts = [f'<line class="axis" x1="{L}" x2="{W}" y1="{H - B}" y2="{H - B}"/>']
    for i, (lab, c, red) in enumerate(buckets):
        h = (H - B - T) * c / top
        x = L + i * bw + 8
        parts.append(f'<rect class="bar {"opec" if red else "non"}" x="{x:.1f}" y="{H - B - h:.1f}" width="{bw - 16:.1f}" height="{h:.1f}"{" style=\"fill:var(--red)\"" if red else ""}/>'
                     f'<text class="val" x="{x + (bw - 16) / 2:.1f}" y="{H - B - h - 8:.1f}" text-anchor="middle">{c}</text>'
                     f'<text class="tick-l" x="{x + (bw - 16) / 2:.1f}" y="{H - B + 24}" text-anchor="middle">{e(lab)}</text>')
    return f'<svg class="hist" viewBox="0 0 {W} {H}" role="img" aria-label="{e(aria)}">{"".join(parts)}</svg>'


# ---------- page assembly (helpers come from build.py)

def make(B):
    layout, frame, table = B.layout, B.frame, B.table

    def head(kicker, h1, plain):
        return f'<header class="page-head wrap"><p class="kicker">{kicker}</p><h1>{h1}</h1><p class="plain"><strong>In plain words:</strong> {plain}</p></header>'

    def lr(rows):  # label, rows for tables
        return rows

    # ----- economies
    def page_economies():
        top = ranked("gdp", GDP_Y)[:15]
        world = at(WORLD, "gdp", GDP_Y)
        tot_all = sum(v for _, v in ranked("gdp", GDP_Y))
        top3 = sum(v for _, v in top[:3])
        top10 = sum(v for _, v in top[:10])
        n_gdp = len(ranked("gdp", GDP_Y))
        t_top = table(["Economy", "GDP, current US$", "Share of world GDP"], [[e(c["name"]), usd(v), pct(v / world * 100)] for c, v in top], num_cols=(1, 2))
        c1 = bars([{"label": c["name"], "value": v / 1e12, "red": i == 0} for i, (c, v) in enumerate(top)],
                  f"The 15 largest economies by GDP in {GDP_Y}, trillions of current US dollars: " + "; ".join(f"{c['name']} {v / 1e12:.1f}" for c, v in top) + ".",
                  fmt=lambda v: f"${v:,.1f} tn", pid="gdp")
        # share of world GDP since 1960 for the six largest
        six = [c for c, _ in top[:6]]
        years = range(1960, GDP_Y + 1)
        ser = []
        for c in six:
            pts = [(y, at(c, "gdp", y) / at(WORLD, "gdp", y) * 100) for y in years if at(c, "gdp", y) and at(WORLD, "gdp", y)]
            if pts:
                ser.append({"name": c["name"], "pts": pts, "red": c["code"] == "CHN"})
        china = next(s for s in ser if s["name"] == "China")
        us = next(s for s in ser if s["name"] == "United States")
        usmax = max(us["pts"], key=lambda p: p[1])
        c2 = lines(ser, f"Share of world GDP, 1960 to {GDP_Y}, for the six largest economies: " + "; ".join(f"{s['name']} {s['pts'][0][1]:.1f}% in {s['pts'][0][0]}, {s['pts'][-1][1]:.1f}% in {GDP_Y}" for s in ser) + ".",
                   yfmt=lambda v: f"{v:g}%", lfmt=lambda v: f"{v:.1f}%", xticks=[1960, 1980, 2000, GDP_Y], xfmt=lambda x: f"{x:.0f}")
        t2 = table(["Year"] + [e(s["name"]) for s in ser], [[str(y)] + [f"{dict(s['pts']).get(y, float('nan')):.1f}%" for s in ser] for y in (1960, 1970, 1980, 1990, 2000, 2010, 2020, GDP_Y)], num_cols=tuple(range(1, len(ser) + 1)))
        # rank movers 2000 to GDP_Y
        r0 = {c["code"]: i + 1 for i, (c, _) in enumerate(ranked("gdp", 2000))}
        r1 = {c["code"]: i + 1 for i, (c, _) in enumerate(ranked("gdp", GDP_Y))}
        movers = sorted([(c, r0[c["code"]], r1[c["code"]]) for c, _ in ranked("gdp", GDP_Y)[:30] if c["code"] in r0], key=lambda t: t[2])[:20]
        up = max(movers, key=lambda t: t[1] - t[2])
        down = min(movers, key=lambda t: t[1] - t[2])
        t3 = table(["Economy", f"Rank in 2000", f"Rank in {GDP_Y}", "Places gained"], [[e(c["name"]), str(a), str(b), f"{a - b:+d}"] for c, a, b in movers], num_cols=(1, 2, 3))
        c3 = bars([{"label": c["name"], "value": a - b, "red": c["code"] == up[0]["code"]} for c, a, b in sorted(movers, key=lambda t: t[2] - t[1])],
                  f"Places gained or lost in the GDP ranking from 2000 to {GDP_Y} for the 20 largest economies; {up[0]['name']} gained {up[1] - up[2]}, {down[0]['name']} lost {down[2] - down[1]}.",
                  fmt=lambda v: f"{v:+d}", signed=True, pid="mv")
        # GDP per person
        pcs = [(c, gdp_pc(c, GDP_Y)) for c in COUNTRIES if gdp_pc(c, GDP_Y) and (at(c, "pop", GDP_Y) or 0) >= 1e6]
        pcs.sort(key=lambda t: -t[1])
        rich, poor = pcs[:12], pcs[-12:][::-1]
        ratio = pcs[0][1] / pcs[-1][1]
        c4 = bars([{"label": c["name"], "value": v, "red": False} for c, v in rich] + [{"label": c["name"], "value": v, "red": True} for c, v in reversed(poor)],
                  f"GDP per person in {GDP_Y}, US dollars: the 12 highest, led by {rich[0][0]['name']} at {rich[0][1]:,.0f}, and the 12 lowest, ending with {pcs[-1][0]['name']} at {pcs[-1][1]:,.0f}.",
                  fmt=lambda v: f"${v:,.0f}", pid="pc")
        t4 = table(["Economy", "GDP per person, US$", "Population"], [[e(c["name"]), f"{v:,.0f}", people(at(c, "pop", GDP_Y))] for c, v in rich + poor[::-1]], num_cols=(1, 2))
        all_rows = [[e(c["name"]), e(c["region"]), usd(g), f"{gdp_pc(c, GDP_Y):,.0f}" if gdp_pc(c, GDP_Y) else "", people(at(c, "pop", GDP_Y)) if at(c, "pop", GDP_Y) else ""]
                    for c, g in ranked("gdp", GDP_Y)]
        miss = [c["name"] for c in COUNTRIES if at(c, "gdp", GDP_Y) is None]
        body = f"""{head("Economies", "How big, how fast, how rich", "GDP, gross domestic product, is the value of everything a country produces in a year. Here it is in US dollars at that year's prices and exchange rates, so it grows when output grows and also when prices rise or the dollar weakens.")}
<section class="section wrap">
  {frame("top", f"{e(top[0][0]['name'])}'s economy is {top[0][1] / top[1][1]:.1f} times {e(top[1][0]['name'])}'s", f"World Bank, GDP (current US$), NY.GDP.MKTP.CD, {GDP_Y}: the 15 largest of {n_gdp} economies with a figure. Share of world uses the World Bank's world total.", c1, t_top,
         f"The three largest hold {top3 / world * 100:.0f}% of world GDP, the ten largest {top10 / world * 100:.0f}%. The other {n_gdp - 10} economies share the remaining {100 - top10 / world * 100:.0f}%.")}
  {frame("share", f"China's share of world GDP went from {china['pts'][0][1]:.1f}% to {china['pts'][-1][1]:.1f}%", f"Each economy's GDP in current US$ divided by the world total, 1960 to {GDP_Y}. Current dollars, so currency swings show up as share changes (Japan's rise and fall after 1985 is partly the yen).", c2, t2,
         f"The United States peaked at {usmax[1]:.1f}% in {usmax[0]:.0f} and stood at {us['pts'][-1][1]:.1f}% in {GDP_Y}.")}
  {frame("move", f"{e(up[0]['name'])} gained {up[1] - up[2]} places in the ranking since 2000", f"Rank by GDP in current US$, 2000 against {GDP_Y}, for the {len(movers)} largest economies in {GDP_Y} that have both years. Positive: moved up.", c3, t3,
         f"{e(down[0]['name'])} fell {down[2] - down[1]} places, the most of the group.")}
  {frame("pc", f"GDP per person ranges {ratio:,.0f} to 1 between the top and bottom", f"GDP divided by population, {GDP_Y}, current US$, for the 12 highest and 12 lowest of {len(pcs)} economies with at least 1 million people. Red: lowest 12.", c4, t4,
         f"{e(pcs[0][0]['name'])} is at ${pcs[0][1]:,.0f} a person and {e(pcs[-1][0]['name'])} at ${pcs[-1][1]:,.0f}. This is output, not income, and the dollars are at market exchange rates, not adjusted for what money buys locally.")}
  <section aria-labelledby="all-h"><h2 id="all-h">All {n_gdp} economies</h2>
    <p>Ordered by GDP in {GDP_Y}. Population is for {GDP_Y}. Region is the UN's. {len(miss)} economies in the data have no {GDP_Y} GDP figure: {e(", ".join(miss[:12]))}{"…" if len(miss) > 12 else ""}.</p>
    <div class="table-wrap tall" role="region" aria-label="GDP of all economies" tabindex="0">{table(["Economy", "Region", "GDP, current US$", "GDP per person", "Population"], all_rows, num_cols=(2, 3, 4))}</div>
    <p class="source">World Bank, via the Open Knowledge Foundation's <code>datasets/gdp</code> and <code>datasets/population</code> (CC-BY-4.0, PDDL). <a href="data.html#countries">What was repaired</a></p>
  </section>
</section>"""
        return layout("economies", f"The World's Economies by Size, Share and GDP per Person ({GDP_Y}) · MacroEconomics",
                      f"The 15 largest economies in {GDP_Y}, each big economy's share of world GDP since 1960, who moved up the ranking since 2000, and GDP per person for the richest and poorest, with all {n_gdp} economies in a table.", body, crumb="Economies")

    # ----- people
    def page_people():
        P = ranked("pop", POP_Y)
        n = len(P)
        world = at(WORLD, "pop", POP_Y)
        top = P[:15]
        c1 = bars([{"label": c["name"], "value": v / 1e6, "red": c["code"] in ("IND", "CHN")} for c, v in top],
                  f"The 15 most populous countries in {POP_Y}, millions: " + "; ".join(f"{c['name']} {v / 1e6:,.0f}" for c, v in top) + ".",
                  fmt=lambda v: f"{v:,.0f} m", pid="pop")
        t1 = table(["Country", f"Population {POP_Y}", "Share of world"], [[e(c["name"]), f"{v:,.0f}", pct(v / world * 100)] for c, v in top], num_cols=(1, 2))
        # when India passed China
        ind, chn = BY["IND"], BY["CHN"]
        cross = next((y for y in range(1960, POP_Y + 1) if at(ind, "pop", y) and at(ind, "pop", y) > at(chn, "pop", y)), None)
        # regions as share of world
        regs = defaultdict(lambda: defaultdict(float))
        for c in COUNTRIES:
            if c["region"]:
                for y in range(1960, POP_Y + 1):
                    v = at(c, "pop", y)
                    if v:
                        regs[c["region"]][y] += v
        tot_by_y = {y: sum(r[y] for r in regs.values()) for y in range(1960, POP_Y + 1)}
        ser = [{"name": r, "pts": [(y, regs[r][y] / tot_by_y[y] * 100) for y in range(1960, POP_Y + 1)], "red": r == "Africa"} for r in sorted(regs, key=lambda r: -regs[r][POP_Y])]
        afr = next(s for s in ser if s["name"] == "Africa")
        c2 = lines(ser, f"Share of the world's people by UN region, 1960 to {POP_Y}: " + "; ".join(f"{s['name']} {s['pts'][0][1]:.0f}% to {s['pts'][-1][1]:.0f}%" for s in ser) + ".",
                   yfmt=lambda v: f"{v:g}%", lfmt=lambda v: f"{v:.1f}%", xticks=[1960, 1980, 2000, POP_Y], xfmt=lambda x: f"{x:.0f}")
        t2 = table(["Year"] + [e(s["name"]) for s in ser], [[str(y)] + [f"{dict(s['pts'])[y]:.1f}%" for s in ser] for y in (1960, 1980, 2000, 2020, POP_Y)], num_cols=tuple(range(1, len(ser) + 1)))
        # shrinking and growing since peak / since 2000
        shr = []
        for c, v in P:
            if v >= 1e6:
                s = c["s"]["pop"]
                peak = max(x for x in s["v"] if x is not None)
                if peak > v * 1.01:
                    shr.append((c, (v / peak - 1) * 100, peak))
        shr.sort(key=lambda t: t[1])
        shr = shr[:12]
        c3 = bars([{"label": c["name"], "value": ch, "red": True} for c, ch, _ in shr],
                  f"Countries of at least a million people whose {POP_Y} population is furthest below their own peak: " + "; ".join(f"{c['name']} {ch:.0f}%" for c, ch, _ in shr) + ".",
                  fmt=lambda v: f"{v:.0f}%", signed=True, pid="shr")
        t3 = table(["Country", f"Population {POP_Y}", "Peak population", "Change from peak"], [[e(c["name"]), f"{at(c, 'pop', POP_Y):,.0f}", f"{pk:,.0f}", pct(ch, 1, True)] for c, ch, pk in shr], num_cols=(1, 2, 3))
        n_shrink = sum(1 for c, v in P if v >= 1e6 and max(x for x in c["s"]["pop"]["v"] if x is not None) > v * 1.01)
        gro = sorted([(c, (v / at(c, "pop", 2000) - 1) * 100) for c, v in P if at(c, "pop", 2000) and v >= 1e6], key=lambda t: -t[1])[:12]
        c4 = bars([{"label": c["name"], "value": g, "red": False} for c, g in gro],
                  f"Fastest population growth from 2000 to {POP_Y}, countries over a million people: " + "; ".join(f"{c['name']} {g:.0f}%" for c, g in gro) + ".",
                  fmt=lambda v: f"+{v:.0f}%", pid="gro")
        t4 = table(["Country", "Population 2000", f"Population {POP_Y}", "Growth"], [[e(c["name"]), f"{at(c, 'pop', 2000):,.0f}", f"{at(c, 'pop', POP_Y):,.0f}", pct(g, 0, True)] for c, g in gro], num_cols=(1, 2, 3))
        body = f"""{head("People", "Who lives where, and who is growing or shrinking", "Population is the number of people living in a country at the middle of the year. The World Bank compiles it from national counts and the UN's estimates, and the latest years are estimates.")}
<section class="section wrap">
  {frame("pop", f"{e(top[0][0]['name'])} has {top[0][1] / top[1][1]:.2f} times {e(top[1][0]['name'])}'s people" if cross is None else f"India passed China in {cross} and now has {at(ind, 'pop', POP_Y) / 1e6:,.0f} million people", f"World Bank, SP.POP.TOTL, {POP_Y}: the 15 largest of {n} countries. India and China in red. World total {people(world)}.", c1, t1,
         f"The ten most populous countries hold {sum(v for _, v in P[:10]) / world * 100:.0f}% of the world's people. {n - 10} countries share the rest." )}
  {frame("reg", f"Africa's share of the world's people rose from {afr['pts'][0][1]:.0f}% to {afr['pts'][-1][1]:.0f}%", f"Each UN region's people as a share of the sum of {n} countries, 1960 to {POP_Y}. Regions are the ISO/UN M49 regions.", c2, t2,
         f"Asia is still the largest region at {next(s for s in ser if s['name'] == 'Asia')['pts'][-1][1]:.0f}%; Europe's share fell from {next(s for s in ser if s['name'] == 'Europe')['pts'][0][1]:.0f}% to {next(s for s in ser if s['name'] == 'Europe')['pts'][-1][1]:.0f}%.")}
  {frame("shr", f"{n_shrink} countries have fewer people now than at their peak", f"{POP_Y} population against each country's own highest year since 1960, countries of at least a million people; the 12 furthest below peak.", c3, t3,
         f"{e(shr[0][0]['name'])} is {abs(shr[0][1]):.0f}% below its peak of {people(shr[0][2])}.")}
  {frame("gro", f"{e(gro[0][0]['name'])}'s population grew {gro[0][1]:.0f}% since 2000", f"Change in population from 2000 to {POP_Y}, the 12 fastest of the countries with a million people or more.", c4, t4,
         f"The world grew {(world / at(WORLD, 'pop', 2000) - 1) * 100:.0f}% over the same years.")}
  <p class="source">World Bank via the Open Knowledge Foundation's <code>datasets/population</code> (PDDL). The latest year is an estimate. <a href="data.html#countries">About the data</a></p>
</section>"""
        return layout("people", f"Population: Who Lives Where and Who Is Growing or Shrinking ({POP_Y}) · MacroEconomics",
                      f"The 15 most populous countries, each UN region's share of the world's people since 1960, the countries with fewer people than at their peak, and the fastest growers since 2000.", body, crumb="People")

    # ----- prices
    def page_prices():
        I = ranked("infl", INF_Y)
        n = len(I)
        bounds = [(-100, 0, "below 0"), (0, 2, "0 to 2"), (2, 5, "2 to 5"), (5, 10, "5 to 10"), (10, 20, "10 to 20"), (20, 50, "20 to 50"), (50, 1e9, "50 and up")]
        counts = [(lab, sum(1 for _, v in I if lo <= v < hi), lo >= 20) for lo, hi, lab in bounds]
        h = histogram(counts, f"Countries by consumer price inflation in {INF_Y}, percent a year: " + "; ".join(f"{l}: {c}" for l, c, _ in counts) + ".")
        over10 = sum(c for l, c, _ in counts[4:])
        under5 = sum(c for l, c, _ in counts[:3])
        t1 = table(["Inflation, % a year", "Countries"], [[l, str(c)] for l, c, _ in counts], num_cols=(1,))
        high = I[:15]
        c2 = bars([{"label": c["name"], "value": v, "red": v >= 50} for c, v in high], f"The 15 highest inflation rates in {INF_Y}, percent: " + "; ".join(f"{c['name']} {v:.0f}" for c, v in high) + ".", fmt=lambda v: f"{v:,.0f}%", pid="hi")
        t2 = table(["Country", f"Inflation {INF_Y}"], [[e(c["name"]), pct(v)] for c, v in high], num_cols=(1,))
        # peaks ever
        peaks = []
        for c in COUNTRIES:
            s = c["s"].get("infl")
            if s:
                vals = [(s["y0"] + i, v) for i, v in enumerate(s["v"]) if v is not None]
                if vals:
                    y, v = max(vals, key=lambda t: t[1])
                    peaks.append((c, y, v, vals[0][0]))
        peaks.sort(key=lambda t: -t[2])
        pk = peaks[:12]
        c3 = bars([{"label": f"{c['name']} {y}", "value": v, "red": True} for c, y, v, _ in pk], "The 12 highest yearly inflation rates on record in the file, by country and year: " + "; ".join(f"{c['name']} {y} {v:,.0f}%" for c, y, v, _ in pk) + ".", fmt=lambda v: f"{v:,.0f}%", pid="pk")
        t3 = table(["Country", "Peak year", "Peak inflation", "Series starts"], [[e(c["name"]), str(y), pct(v, 0), str(f)] for c, y, v, f in pk], num_cols=(1, 2, 3))
        # world average line
        w = WORLD["s"]["infl"]
        wpts = [(w["y0"] + i, v) for i, v in enumerate(w["v"]) if v is not None]
        c4 = lines([{"name": "World", "pts": wpts, "red": True}], f"World consumer price inflation, percent a year, {wpts[0][0]} to {wpts[-1][0]}: {wpts[0][1]:.1f}% in {wpts[0][0]}, peak {max(wpts, key=lambda p: p[1])[1]:.1f}% in {max(wpts, key=lambda p: p[1])[0]}, {wpts[-1][1]:.1f}% in {wpts[-1][0]}.",
                   yfmt=lambda v: f"{v:g}%", lfmt=lambda v: f"{v:.1f}%", xticks=[1960, 1980, 2000, wpts[-1][0]], hline=0)
        wpk = max(wpts, key=lambda p: p[1])
        t4 = table(["Year", "World inflation"], [[str(y), pct(v)] for y, v in wpts if y % 5 == 0 or y == wpts[-1][0]], num_cols=(1,))
        # US CPI since 1913
        us = MK["us_cpi"]
        keys = sorted(us)
        first, lastk = keys[0], keys[-1]
        pts = [(mo(k), us[k]) for k in keys]
        c5 = lines([{"name": "US CPI", "pts": pts, "red": True}], f"US consumer price index, {first} to {lastk}: {us[first]:.1f} to {us[lastk]:.1f}, a rise of {us[lastk] / us[first]:.1f} times.",
                   yfmt=lambda v: f"{v:g}", lfmt=lambda v: f"{v:.1f}", xticks=[1920, 1950, 1980, 2010], gap=1.0)
        cents = us[first] / us[lastk] * 100
        t5 = table(["Year", "US CPI (all items)"], [[k[:4], f"{us[k]:.1f}"] for k in keys if k.endswith("-01") and int(k[:4]) % 10 == 3 or k == lastk], num_cols=(1,))
        body = f"""{head("Prices", "Where prices are rising, and how fast", "Inflation is the yearly rise in what a typical basket of goods costs. A rate of 2% is gentle; at 50% a year, prices double in under two years. The data here is consumer price inflation as the World Bank collects it.")}
<section class="section wrap">
  {frame("dist", f"In {INF_Y}, {under5} of {n} countries had inflation under 5% and {over10} had 10% or more", f"World Bank, FP.CPI.TOTL.ZG, {INF_Y}: countries by yearly consumer price inflation. Red: 20% or more.", h, t1,
         f"The median country had {sorted(v for _, v in I)[n // 2]:.1f}% inflation. The highest was {e(I[0][0]['name'])} at {I[0][1]:,.0f}%.")}
  {frame("high", f"{e(I[0][0]['name'])} had {I[0][1]:,.0f}% inflation in {INF_Y}", f"The 15 highest rates of {n} countries in {INF_Y}. Red: 50% or more, where prices double in under two years.", c2, t2)}
  {frame("peak", f"The worst year on record in the file: {e(pk[0][0]['name'])}, {pk[0][1]}, at {pk[0][2]:,.0f}%", "Each country's single highest yearly rate in the file, the 12 highest. Series start in different years (most in the late 1980s), so older episodes may be missing.", c3, t3,
         f"The file starts in different years by country; the list shows the peak within each country's own range, not a world ranking of all history.")}
  {frame("world", f"World inflation peaked at {wpk[1]:.1f}% in {wpk[0]}", f"World Bank world aggregate, consumer prices, {wpts[0][0]} to {wpts[-1][0]}. The dotted line is zero.", c4, t4)}
  {frame("us", f"A 1913 US dollar buys {cents:.0f} cents of 1913 goods today", f"US Bureau of Labor Statistics, CPI-U all items, monthly, {first} to {lastk}. The index is {us[first]:.1f} in {first} and {us[lastk]:.1f} in {lastk}.", c5, t5,
         f"US prices are {us[lastk] / us[first]:.1f} times their {first[:4]} level.")}
  <p class="source">World Bank via <code>datasets/inflation</code>, and BLS via <code>datasets/cpi-us</code> (PDDL). The older <a href="inflation.html">inflation page</a> follows 100 units of money from 1960. <a href="data.html#markets">About the data</a></p>
</section>"""
        return layout("prices", f"Inflation and Prices Around the World ({INF_Y}) · MacroEconomics",
                      f"Inflation in {n} countries in {INF_Y}, the highest rates and worst years on record, world inflation since 1960, and a century of US prices.", body, crumb="Prices")

    # ----- markets
    def series_pts(d, gap=None):
        return [(mo(k), v) for k, v in sorted(d.items())]

    def page_markets():
        uk = MK["us_cpi"]
        base_k = max(uk)

        def real(d):
            return {k: v * uk[base_k] / uk[k] for k, v in d.items() if k in uk}
        brent, wti = MK["brent"], MK["wti"]
        bl, bk = max(brent), min(brent)
        brent_real = real(brent)
        peak_n = max(brent, key=lambda k: brent[k])
        peak_r = max(brent_real, key=lambda k: brent_real[k])
        c1 = lines([{"name": "Brent, today's dollars", "pts": series_pts(brent_real), "red": True}, {"name": "Brent", "pts": series_pts(brent)}, {"name": "WTI", "pts": series_pts(wti)}],
                   f"Oil prices per barrel, monthly, {bk} to {bl}: Brent at {brent[bl]:.0f} dollars in {bl}; its highest month was {peak_n} at {brent[peak_n]:.0f}; in {base_k} dollars its highest was {peak_r} at {brent_real[peak_r]:.0f}.",
                   yfmt=lambda v: f"${v:g}", lfmt=lambda v: f"${v:.0f}", xticks=[1990, 2000, 2010, 2020], gap=1.0)
        t1 = table(["Year", "Brent, US$", f"Brent, {base_k[:4]} dollars", "WTI, US$"],
                   [[str(y), f"{sum(v for k, v in brent.items() if k.startswith(str(y))) / max(1, sum(1 for k in brent if k.startswith(str(y)))):.1f}",
                     f"{sum(v for k, v in brent_real.items() if k.startswith(str(y))) / max(1, sum(1 for k in brent_real if k.startswith(str(y)))):.1f}",
                     f"{sum(v for k, v in wti.items() if k.startswith(str(y))) / max(1, sum(1 for k in wti if k.startswith(str(y)))):.1f}"] for y in (1990, 2000, 2008, 2010, 2015, 2020, 2022, 2024, int(bl[:4]))], num_cols=(1, 2, 3))
        gold = MK["gold"]
        gl = max(gold)
        gold_real = real(gold)
        c2 = lines([{"name": "Gold, nominal", "pts": series_pts(gold), "red": True}, {"name": f"Gold, {base_k[:4]} dollars", "pts": series_pts(gold_real)}],
                   f"Gold price per troy ounce, monthly, 1833 to {gl}, on a log scale: {gold[min(gold)]:.2f} dollars in {min(gold)} and {gold[gl]:,.0f} in {gl}.", yfmt=lambda v: f"${v:,.0f}", log=True,
                   yticks=[10, 100, 1000, 10000], xticks=[1850, 1900, 1950, 2000], gap=1.0)
        gpk = max(gold_real, key=lambda k: gold_real[k])
        gk0 = min(gold)
        t2 = table(["Year", "Gold, US$", f"Gold, {base_k[:4]} dollars"], [[y, f"{gold[f'{y}-01']:,.0f}", f"{gold_real[f'{y}-01']:,.0f}"] for y in ("1850", "1900", "1933", "1950", "1971", "1980", "2000", "2011", "2020", "2025") if f"{y}-01" in gold and f"{y}-01" in gold_real], num_cols=(1, 2))
        sp = MK["sp500"]
        sl = max(sp)
        sp_real = real(sp)
        c3 = lines([{"name": "S&P 500, nominal", "pts": series_pts(sp), "red": True}, {"name": f"S&P 500, {base_k[:4]} dollars", "pts": series_pts(sp_real)}],
                   f"S&P 500 index, monthly, 1871 to {sl}, log scale: {sp[min(sp)]:.2f} in {min(sp)} and {sp[sl]:,.0f} in {sl}.", yfmt=lambda v: f"{v:,.0f}", log=True, yticks=[3, 10, 100, 1000, 10000], xticks=[1880, 1920, 1960, 2000], gap=1.0)
        # worst drawdown on real index
        peak, worst = 0, (0, None, None)
        pk_k = None
        for k in sorted(sp_real):
            if sp_real[k] > peak:
                peak, pk_k = sp_real[k], k
            dd = sp_real[k] / peak - 1
            if dd < worst[0]:
                worst = (dd, pk_k, k)
        t3 = table(["Year (January)", "S&P 500", f"{base_k[:4]} dollars"], [[y, f"{sp[f'{y}-01']:,.1f}", f"{sp_real[f'{y}-01']:,.0f}"] for y in ("1880", "1900", "1929", "1950", "1980", "2000", "2009", "2020", "2025") if f"{y}-01" in sp and f"{y}-01" in sp_real], num_cols=(1, 2))
        pe = MK["pe10"]
        pel = max(pe)
        pe_mean = sum(pe.values()) / len(pe)
        c4 = lines([{"name": "P/E10", "pts": series_pts(pe), "red": True}], f"Shiller cyclically adjusted price-earnings ratio, monthly, {min(pe)} to {pel}: {pe[pel]:.1f} in {pel}, average {pe_mean:.1f}, highest {max(pe.values()):.1f} in {max(pe, key=lambda k: pe[k])}.",
                   yfmt=lambda v: f"{v:g}", lfmt=lambda v: f"{v:.1f}", xticks=[1900, 1940, 1980, 2020], hline=pe_mean, gap=1.0)
        t4 = table(["Year (January)", "P/E10"], [[y, f"{pe[f'{y}-01']:.1f}"] for y in ("1890", "1920", "1929", "1950", "1966", "1982", "2000", "2010", "2020") if f"{y}-01" in pe], num_cols=(1,))
        ng = MK["natgas"]
        ngl = max(ng)
        c5 = lines([{"name": "Henry Hub", "pts": series_pts(ng), "red": True}], f"US natural gas spot price, Henry Hub, dollars per million BTU, monthly, {min(ng)} to {ngl}: highest {max(ng.values()):.2f} in {max(ng, key=lambda k: ng[k])}.",
                   yfmt=lambda v: f"${v:g}", lfmt=lambda v: f"${v:.2f}", xticks=[2000, 2010, 2020], gap=1.0)
        t5 = table(["Year (January)", "Henry Hub, US$ per million BTU"], [[y, f"{ng[f'{y}-01']:.2f}"] for y in ("1998", "2003", "2008", "2012", "2016", "2020", "2022", "2024") if f"{y}-01" in ng], num_cols=(1,))
        body = f"""{head("Markets", "Oil, gold, shares and gas, in the dollars of the day and in today's", "A price in dollars from 1980 is not the same dollar as today's. Where a chart says &ldquo;today's dollars&rdquo;, the price is scaled by US consumer prices so old and new prices can be compared.")}
<section class="section wrap">
  {frame("oil", f"Brent peaked at ${brent[peak_n]:.0f} in {peak_n}; in today's dollars the peak is {peak_r}", f"US EIA Europe Brent and Cushing WTI spot prices, daily averaged by month, {bk} to {bl}. Today's dollars use BLS CPI-U ({base_k}).", c1, t1,
         f"Brent was ${brent[bl]:.0f} a barrel in {bl}. In today's dollars the {peak_r} peak was ${brent_real[peak_r]:.0f}; the highest month in nominal terms was ${brent[peak_n]:.0f} in {peak_n}.")}
  {frame("gold", f"An ounce of gold cost ${gold[gk0]:.2f} in {gk0[:4]} and ${gold[gl]:,.0f} in {gl}", f"World Bank Pink Sheet and Timothy Green's table (National Mining Association), US$ per troy ounce, monthly. Log scale: each gridline is ten times the one below.", c2, t2,
         f"In {base_k[:4]} dollars, gold's highest month was {gpk} at ${gold_real[gpk]:,.0f}.")}
  {frame("sp", f"The S&P 500 index is up {sp[sl] / sp[min(sp)]:,.0f} times since {min(sp)[:4]}", f"Robert Shiller's long-run S&P series spliced with the S&P 500 (FRED) to {sl}. Index level only: dividends and earnings in the file end in mid-2023. Log scale.", c3, t3,
         f"Adjusted for prices, the worst fall from a peak was {abs(worst[0]) * 100:.0f}%, from {worst[1]} to {worst[2]}.")}
  {frame("pe", f"Shiller's P/E10 stood at {pe[pel]:.1f} in {pel}, against an average of {pe_mean:.1f}", f"Price divided by ten years of average inflation-adjusted earnings, {min(pe)} to {pel} (the file's earnings end there). Line: the average over the whole series.", c4, t4,
         f"The highest reading was {max(pe.values()):.1f} in {max(pe, key=lambda k: pe[k])}. A high ratio has said little about next month's prices; it is a description, not a forecast.")}
  {frame("gas", f"US natural gas reached ${max(ng.values()):.2f} per million BTU in {max(ng, key=lambda k: ng[k])}", f"US EIA Henry Hub spot price, monthly, {min(ng)} to {ngl}. Nominal dollars.", c5, t5)}
  <p class="source">Sources: US EIA; World Bank Pink Sheet; Robert Shiller; FRED; BLS, via the Open Knowledge Foundation's <code>datasets/oil-prices</code>, <code>gold-prices</code>, <code>s-and-p-500</code>, <code>natural-gas</code> and <code>cpi-us</code> (PDDL). <a href="data.html#markets">What was repaired</a></p>
</section>"""
        return layout("markets", f"Oil, Gold, Shares and Gas: Prices in the Dollars of the Day and Today's · MacroEconomics",
                      f"Brent and WTI oil since 1987, gold since 1833, the S&P 500 since 1871 with its price-earnings ratio, and US natural gas, each in nominal dollars and adjusted for US consumer prices.", body, crumb="Markets")

    # ----- money
    def page_money():
        fx, days = MK["fx"], MK["fx_days"]
        full = max(int(y) for y, n in days["Euro"].items() if n >= 200)
        base = 2000
        ch = []
        for cur, ys in fx.items():
            a, b = ys.get(str(base)), ys.get(str(full))
            if a and b and cur != "Venezuela":
                ch.append((cur, (a / b - 1) * 100))
        ch.sort(key=lambda t: -t[1])
        worst = ch[-1]
        c1 = bars([{"label": c, "value": v, "red": v < 0} for c, v in ch], f"Change in the US dollar value of each currency, {base} to {full}, percent; red: fell. " + "; ".join(f"{c} {v:+.0f}%" for c, v in ch) + ".",
                  fmt=lambda v: f"{v:+.0f}%", signed=True, pid="fx")
        t1 = table(["Currency (country)", f"Local units per US$ in {base}", f"in {full}", "Change in value against the dollar"], [[e(c), rate(fx[c][str(base)]), rate(fx[c][str(full)]), pct(v, 0, True)] for c, v in ch], num_cols=(1, 2, 3))
        pick = ["Euro", "Japan", "United Kingdom", "Switzerland", "China"]
        ser = []
        for cur in pick:
            a = fx[cur][str(base)]
            ser.append({"name": cur, "pts": [(int(y), a / v * 100) for y, v in fx[cur].items() if int(y) >= base and days[cur][y] >= 200], "red": cur == "Euro"})
        c2 = lines(ser, f"Value of five currencies against the US dollar, {base} = 100, annual averages to {full}: " + "; ".join(f"{s['name']} {s['pts'][-1][1]:.0f}" for s in ser) + ".", yfmt=lambda v: f"{v:g}", lfmt=lambda v: f"{v:.0f}", xticks=[2000, 2008, 2016, full], xfmt=lambda x: f"{x:.0f}")
        t2 = table(["Year"] + [e(s["name"]) for s in ser], [[str(y)] + [f"{dict(s['pts'])[y]:.0f}" for s in ser] for y in (2000, 2005, 2010, 2015, 2020, full)], num_cols=tuple(range(1, len(ser) + 1)))
        years = [1980, 1990, 2000, 2010, 2020, full]
        t3 = table(["Currency (country)"] + [str(y) for y in years], [[e(c)] + [rate(fx[c][str(y)]) if str(y) in fx[c] else "n/a" for y in years] for c in fx], num_cols=tuple(range(1, len(years) + 1)))
        usd_ = f"{full}"
        body = f"""{head("Money", "What the dollar buys in other currencies", "An exchange rate says how many units of another currency one US dollar buys. When the number rises, the other currency has lost value against the dollar. The charts below are inverted to show each currency's own value, so up means stronger.")}
<section class="section wrap">
  {frame("chg", f"{e(worst[0])}'s currency lost {abs(worst[1]):.0f}% of its value against the dollar from {base} to {full}", f"Federal Reserve rates, annual averages, local currency per US dollar; change in the dollar value of {len(ch)} currencies with both years. Red: fell. Euro means the euro area. Venezuela is left out: its currency was redenominated several times and the series is not on one scale.", c1, t1,
         f"{sum(1 for _, v in ch if v < 0)} of {len(ch)} currencies lost value against the dollar; {sum(1 for _, v in ch if v > 0)} gained. {e(ch[0][0])} gained the most, at {ch[0][1]:+.0f}%.")}
  {frame("five", f"Five currencies against the dollar since {base}", f"Value of one unit against the US dollar, annual average, {base} = 100. {full} is an average of the days published so far, not a full year.", c2, t2,
         f"By {full}, {e(ser[0]['name'])} stands at {ser[0]['pts'][-1][1]:.0f}, {e(ser[1]['name'])} at {ser[1]['pts'][-1][1]:.0f}, {e(ser[2]['name'])} at {ser[2]['pts'][-1][1]:.0f}.")}
  <section aria-labelledby="fx-all"><h2 id="fx-all">All {len(fx)} currencies</h2>
    <p>Local currency units per US dollar, annual average. n/a: the Federal Reserve series had not begun. The first year of each series varies; Venezuela's figures are in bolívares of different vintages (the currency was redenominated), so they are not comparable across years and the chart above leaves it out.</p>
    <div class="table-wrap" role="region" aria-label="Exchange rates of all currencies" tabindex="0">{t3}</div>
    <p class="source">Federal Reserve Bank of St. Louis via <code>datasets/exchange-rates</code>. The file's readme gives the wrong quoting for five currencies; see <a href="data.html#money">About the data</a>.</p>
  </section>
</section>"""
        return layout("money", f"Exchange Rates: What the Dollar Buys, {base} to {full} · MacroEconomics",
                      f"How 22 currencies moved against the US dollar from {base} to {full}, five major currencies indexed, and every rate in a table.", body, crumb="Money")

    # ----- jobs
    def page_jobs():
        emp = MK["us_employment"]
        u = [(r["year"], r["unemployed_pct"]) for r in emp if r["unemployed_pct"] is not None]
        up = max(u, key=lambda p: p[1])
        lo = min(u, key=lambda p: p[1])
        y1 = u[-1]
        c1 = lines([{"name": "Unemployment rate", "pts": u, "red": True}], f"US unemployment rate, percent of the labour force, {u[0][0]} to {y1[0]}: highest {up[1]}% in {up[0]}, lowest {lo[1]}% in {lo[0]}, {y1[1]}% in {y1[0]}.", yfmt=lambda v: f"{v:g}%", lfmt=lambda v: f"{v:.1f}%", xticks=[1950, 1970, 1990, 2010], gap=1.5)
        t1 = table(["Year", "Unemployment rate", "Unemployed (thousands)"], [[str(r["year"]), f"{r['unemployed_pct']:.1f}%", f"{r['unemployed']:,.0f}"] for r in emp if r["year"] % 5 == 0 or r["year"] == y1[0]], num_cols=(1, 2))
        part = [(r["year"], r["participation"]) for r in emp if r["participation"] is not None]
        pp = max(part, key=lambda p: p[1])
        c2 = lines([{"name": "Participation", "pts": part, "red": True}], f"US labour force participation rate, percent of people 16 and over, {part[0][0]} to {part[-1][0]}: {part[0][1]}% in {part[0][0]}, highest {pp[1]}% in {pp[0]}, {part[-1][1]}% in {part[-1][0]}.", yfmt=lambda v: f"{v:g}%", lfmt=lambda v: f"{v:.1f}%", xticks=[1950, 1970, 1990, 2010], ylo=50, yhi=70, gap=1.5)
        t2 = table(["Year", "Participation rate", "Labour force (thousands)", "Employed (thousands)"], [[str(r["year"]), f"{r['participation']:.1f}%", f"{r['labor']:,.0f}", f"{r['employed']:,.0f}"] for r in emp if r["year"] % 5 == 0 or r["year"] == part[-1][0]], num_cols=(1, 2, 3))
        body = f"""{head("Jobs", "Work in the United States since 1941", "The unemployment rate is the share of people who want a job and are looking but have none. The participation rate is the share of everyone aged 16 and over who is working or looking. A fall in participation can hide unemployment, so the two belong together.")}
<section class="section wrap">
  {frame("un", f"Unemployment reached {up[1]}% in {up[0]} and {lo[1]}% in {lo[0]}", f"US Bureau of Labor Statistics, Current Population Survey table A-1 annual averages, {u[0][0]} to {y1[0]}. People aged 16 and over, not in institutions.", c1, t1,
         f"It was {y1[1]}% in {y1[0]}. Survey redesigns (1994 and others) mean some year-to-year steps are partly method; the file does not mark which.")}
  {frame("part", f"Participation peaked at {pp[1]}% in {pp[0]}", f"Share of the civilian non-institutional population aged 16 and over that is in the labour force, {part[0][0]} to {part[-1][0]}. The axis starts at 50%.", c2, t2,
         f"It was {part[-1][1]}% in {part[-1][0]}, {pp[1] - part[-1][1]:.1f} points below the peak.")}
  <p class="source">US BLS via the Open Knowledge Foundation's <code>datasets/employment-us</code> (PDDL). Only the United States is covered. <a href="data.html#markets">About the data</a></p>
</section>"""
        return layout("jobs", f"US Unemployment and Participation Since 1941 · MacroEconomics",
                      f"The US unemployment rate and labour force participation rate from {u[0][0]} to {y1[0]}, with the figures in tables.", body, crumb="Jobs")

    # ----- climate
    def page_climate():
        fo = {int(k): v for k, v in MK["co2_fossil"].items()}
        yrs = sorted(fo)
        ly = yrs[-1]
        k = 3.664
        tot = [(y, fo[y]["total"] * k / 1000) for y in yrs if fo[y]["total"]]
        ser = [{"name": "Total", "pts": tot, "red": True}]
        for key, nm in (("solid", "Coal"), ("liquid", "Oil"), ("gas", "Gas")):
            ser.append({"name": nm, "pts": [(y, fo[y][key] * k / 1000) for y in yrs if fo[y][key] is not None]})
        c1 = lines(ser, f"Global CO2 from fossil fuels and cement, billions of tonnes of CO2, 1750 to {ly}: {tot[-1][1]:.1f} billion in {ly}; coal {ser[1]['pts'][-1][1]:.1f}, oil {ser[2]['pts'][-1][1]:.1f}, gas {ser[3]['pts'][-1][1]:.1f}.",
                   yfmt=lambda v: f"{v:g} bn", lfmt=lambda v: f"{v:.1f} bn", xticks=[1800, 1850, 1900, 1950, 2000], gap=1.5)
        cell = lambda y, key: f"{fo[y][key] * k / 1000:.2f}" if fo[y][key] is not None else "n/a"
        t1 = table(["Year", "Total", "Coal", "Oil", "Gas"], [[str(y), cell(y, "total"), cell(y, "solid"), cell(y, "liquid"), cell(y, "gas")] for y in (1850, 1900, 1950, 1975, 2000, 2010, 2020, ly)], num_cols=(1, 2, 3, 4))
        y1950 = fo[1950]["total"]
        ppm = MK["co2_ppm"]
        pk = sorted(ppm)
        c2 = lines([{"name": "Mauna Loa", "pts": [(mo(k_), ppm[k_]) for k_ in pk], "red": True}], f"Atmospheric CO2 at Mauna Loa, parts per million, monthly, {pk[0]} to {pk[-1]}: {ppm[pk[0]]:.1f} to {ppm[pk[-1]]:.1f}.", yfmt=lambda v: f"{v:g}", lfmt=lambda v: f"{v:.0f}", xticks=[1960, 1980, 2000, 2020], ylo=300, yhi=440, gap=0.5)
        t2 = table(["Month", "CO2, parts per million"], [[k_, f"{ppm[k_]:.2f}"] for k_ in pk if k_.endswith("-03") and int(k_[:4]) % 10 == 8 or k_ == pk[-1]], num_cols=(1,))
        # country scatter
        pts = []
        for c in COUNTRIES:
            g, p, co = gdp_pc(c, 2022), at(c, "pop", 2022), at(c, "co2pc", 2022)
            if g and co and p and p >= 1e6:
                pts.append({"x": g, "y": co, "code": c["code"], "name": c["name"], "pop": p})
        hi = ("USA", "CHN", "IND", "QAT", "NGA", "DEU")
        sc = scatter(pts, f"Each of {len(pts)} countries in 2022: GDP per person in dollars on a log scale against CO2 per person in tonnes. Richer countries emit more per person, with large exceptions.", "GDP per person, US$ (log scale)", "Tonnes of CO2 per person", hi)
        import statistics
        corr = statistics.correlation([math.log10(p["x"]) for p in pts], [p["y"] for p in pts])
        t3 = table(["Country", "GDP per person, US$", "CO2 per person, tonnes"], [[e(p["name"]), f"{p['x']:,.0f}", f"{p['y']:.2f}"] for p in sorted(pts, key=lambda p: -p["y"])], num_cols=(1, 2))
        # top emitters
        E = ranked("co2", CO2_Y)
        world_co2 = sum(v for _, v in E)
        te = E[:12]
        c4 = bars([{"label": c["name"], "value": v / 1000, "red": c["code"] == te[0][0]["code"]} for c, v in te], f"The 12 largest national CO2 emitters in {CO2_Y}, billions of tonnes: " + "; ".join(f"{c['name']} {v / 1000:.2f}" for c, v in te) + ".", fmt=lambda v: f"{v:.2f} bn", pid="em")
        t4 = table(["Country", f"CO2, million tonnes {CO2_Y}", "Share of countries' total"], [[e(c["name"]), f"{v:,.0f}", pct(v / world_co2 * 100)] for c, v in te], num_cols=(1, 2))
        pcs = ranked("co2pc", CO2_Y, lambda c: (at(c, "pop", CO2_Y) or 0) >= 1e6)
        c5 = bars([{"label": c["name"], "value": v, "red": False} for c, v in pcs[:12]], f"Highest CO2 per person in {CO2_Y}, tonnes, countries with at least a million people: " + "; ".join(f"{c['name']} {v:.1f}" for c, v in pcs[:12]) + ".", fmt=lambda v: f"{v:.1f} t", pid="epc")
        t5 = table(["Country", "Tonnes CO2 per person"], [[e(c["name"]), f"{v:.2f}"] for c, v in pcs[:12]], num_cols=(1,))
        body = f"""{head("Climate", "Energy, carbon and the economy", "Burning coal, oil and gas releases carbon dioxide. The more a country produces and the richer it is, the more it has tended to emit. The charts show how far that holds, and where it does not.")}
<section class="section wrap">
  {frame("gl", f"Global CO2 from fossil fuels is {tot[-1][1] / (y1950 * k / 1000):.1f} times its 1950 level", f"Global Carbon Project via <code>datasets/co2-fossil-global</code>: CO2 from fossil fuels and cement, 1750 to {ly}, in billions of tonnes of CO2 (converted from the file's tonnes of carbon at 3.664).", c1, t1,
         f"{tot[-1][1]:.1f} billion tonnes in {ly}. Coal supplied {ser[1]['pts'][-1][1] / tot[-1][1] * 100:.0f}% of it, oil {ser[2]['pts'][-1][1] / tot[-1][1] * 100:.0f}% and gas {ser[3]['pts'][-1][1] / tot[-1][1] * 100:.0f}%; cement makes up the rest.")}
  {frame("ppm", f"Atmospheric CO2 rose from {ppm[pk[0]]:.0f} to {ppm[pk[-1]]:.0f} parts per million", f"NOAA, Mauna Loa Observatory, monthly mean, {pk[0]} to {pk[-1]}. The yearly zigzag is the Northern Hemisphere's plants breathing. The axis starts at 300.", c2, t2)}
  {frame("sc", "Richer countries emit more per person, but the spread is wide", f"{len(pts)} countries with at least a million people, 2022: GDP per person (log scale) against CO2 per person. Some highlighted.", sc, t3,
         f"The correlation between the logarithm of GDP per person and CO2 per person is {corr:.2f}. At similar incomes, emissions differ several times over, depending on energy sources and industry.")}
  {frame("em", f"{e(te[0][0]['name'])} emitted {te[0][1] / world_co2 * 100:.0f}% of the countries' total CO2 in {CO2_Y}", f"Our World in Data, CO2 from fossil fuels and cement, {CO2_Y}, million tonnes: the 12 largest of {len(E)} countries; total {world_co2 / 1000:.1f} bn tonnes.", c4, t4,
         f"The top three emit {sum(v for _, v in E[:3]) / world_co2 * 100:.0f}% of the total.")}
  {frame("epc", f"{e(pcs[0][0]['name'])} emits {pcs[0][1]:.0f} tonnes of CO2 per person", f"Tonnes of CO2 per person, {CO2_Y}, the 12 highest of {len(pcs)} countries with a million people or more.", c5, t5,
         f"The median country emits {sorted(v for _, v in pcs)[len(pcs) // 2]:.1f} tonnes per person.")}
  <p class="source">Global Carbon Project; NOAA; Our World in Data's <code>co2-data</code> (CC-BY-4.0); World Bank GDP and population. Emissions are territorial: they leave out CO2 embedded in imports. <a href="data.html#climate">What was repaired</a></p>
</section>"""
        return layout("climate", f"Energy, CO2 and the Economy: Emissions per Person and per Dollar · MacroEconomics",
                      f"Global fossil CO2 since 1750, atmospheric CO2 since 1958, emissions against income for {len(pts)} countries, the largest emitters and who emits most per person.", body, crumb="Climate")

    # ----- explorer
    def page_explorer():
        body = f"""{head("Explorer", "Any country, side by side", "Pick a country to see its economy over time: output, output per person, population, inflation and carbon. Pick a second to draw it on the same charts. Everything comes from the datasets on this site.")}
<section class="section wrap" id="explorer" data-src="data/macro_countries.json">
  <noscript><p class="warn">The explorer draws its charts in the browser, so it needs JavaScript. Without it, every figure is on the <a href="economies.html">economies</a>, <a href="people.html">people</a> and <a href="prices.html">prices</a> pages and in the <a href="data/macro_countries.json">country data file</a>.</p></noscript>
  <form class="ex-form" onsubmit="return false">
    <div class="field"><label for="ex-a">Country</label><input id="ex-a" list="ex-list" autocomplete="off" placeholder="Start typing, e.g. Kenya" aria-describedby="ex-help"></div>
    <div class="field"><label for="ex-b">Compare with (optional)</label><input id="ex-b" list="ex-list" autocomplete="off" placeholder="Another country"></div>
    <datalist id="ex-list"></datalist>
  </form>
  <p class="hint" id="ex-help">{len(COUNTRIES)} countries. The address bar keeps your choice, so you can share it.</p>
  <p id="ex-status" role="status" aria-live="polite" class="hint">Loading the data…</p>
  <ul class="ex-tiles" id="ex-tiles" aria-label="Key figures"></ul>
  <div class="ex-grid" id="ex-grid"></div>
  <p class="source">World Bank (GDP, population, consumer price inflation), Our World in Data (CO2). GDP is in current US dollars, not adjusted for prices. A gap in a line is a year with no figure; nothing is filled in. <a href="data.html#countries">About the data</a></p>
</section>"""
        return layout("explorer", "Country Explorer: GDP, People, Prices and Carbon for Every Country · MacroEconomics",
                      f"Choose any of {len(COUNTRIES)} countries and compare two: GDP, GDP per person, population, inflation and CO2 per person over time, drawn in your browser from open data.", body, crumb="Explorer", scripts='<script src="js/explorer.js" defer></script>')

    mp = {"economies": page_economies, "people": page_people, "prices": page_prices, "markets": page_markets,
          "money": page_money, "jobs": page_jobs, "climate": page_climate, "explorer": page_explorer}

    def wider():
        gdp = ranked("gdp", GDP_Y)
        pop = ranked("pop", POP_Y)
        infl = ranked("infl", INF_Y)
        co = ranked("co2", CO2_Y)
        brent = MK["brent"]
        bl = max(brent)
        gold = MK["gold"]
        fx = MK["fx"]
        emp = [r for r in MK["us_employment"] if r["unemployed_pct"] is not None][-1]
        ppm = MK["co2_ppm"]
        world = at(WORLD, "gdp", GDP_Y)
        cards = [
            ("explorer.html", "Explorer", f"{len(COUNTRIES)}", f"countries, each with GDP, people, prices and carbon; compare any two."),
            ("economies.html", "Economies", f"{gdp[0][1] / world * 100:.0f}%", f"of world GDP in {GDP_Y} came from {e(gdp[0][0]['name'])}; {sum(v for _, v in gdp[:10]) / world * 100:.0f}% from the ten largest."),
            ("people.html", "People", f"{pop[0][1] / 1e9:.2f} bn", f"people in {e(pop[0][0]['name'])} in {POP_Y}, the most of {len(pop)} countries."),
            ("prices.html", "Prices", f"{infl[0][1]:,.0f}%", f"inflation in {e(infl[0][0]['name'])} in {INF_Y}, the highest of {len(infl)} countries."),
            ("markets.html", "Markets", f"${brent[bl]:.0f}", f"for a barrel of Brent in {bl}; gold is ${gold[max(gold)]:,.0f} an ounce."),
            ("money.html", "Money", f"{len(fx)}", "currencies against the dollar since 2000, and which lost the most."),
            ("jobs.html", "Jobs", f"{emp['unemployed_pct']:.1f}%", f"US unemployment in {emp['year']}, from a series that starts in 1941."),
            ("climate.html", "Climate", f"{ppm[max(ppm)]:.0f}", f"parts per million of CO2 in the air at Mauna Loa in {max(ppm)}; {e(co[0][0]['name'])} emits the most."),
        ]
        items = "".join(f'<li><a class="tile" href="{h}"><span class="t-topic">{t}</span><span class="t-fig">{f}</span><span class="t-claim">{c}</span></a></li>' for h, t, f, c in cards)
        return f"""<section class="section wrap" aria-labelledby="wider-h">
  <h2 id="wider-h">The wider economy: {len(COUNTRIES)} countries, markets, money, jobs and climate</h2>
  <p class="take">Beyond the six topics, open data on every country's output, population, prices and carbon, a century and more of market prices, and the explorer to put any two countries side by side.</p>
  <ul class="poster wide-poster">{items}</ul>
</section>"""

    def desc():
        ny = len([c for c in COUNTRIES if at(c, "gdp", GDP_Y)])
        return {
            "countries": f"{len(COUNTRIES)} countries and {len(AGGS)} regions and groups: GDP in current US dollars (World Bank, 1960 to {GDP_Y}), population (1960 to {POP_Y}), consumer price inflation and CPI index (World Bank), and CO2, CO2 per person and energy per person (Our World in Data, 1950 to {CO2_Y}). Licences: CC-BY-4.0 and PDDL.",
            "markets": "Monthly Brent and WTI oil (US EIA), gold since 1833 (World Bank Pink Sheet, National Mining Association), the S&P 500 and P/E10 since 1871 (Robert Shiller, FRED), Henry Hub natural gas (US EIA), US CPI since 1913 and US employment since 1941 (BLS), global fossil CO2 since 1750 (Global Carbon Project) and Mauna Loa CO2 (NOAA); exchange rates for 22 currencies (Federal Reserve), as annual means. Licence: PDDL.",
            "money": "Annual average exchange rates, local currency units per US dollar, for 22 currencies, from the Federal Reserve's daily H.10 and G.5 series via the Open Knowledge Foundation. Included in the markets file.",
            "climate": "Global fossil CO2 (Global Carbon Project), atmospheric CO2 at Mauna Loa (NOAA) and country CO2 (Our World in Data). Included in the markets and countries files.",
        }

    mp.update({"_wider": wider, "_desc": desc})
    return mp


def long_csv():
    keys = ["gdp", "pop", "infl", "cpi", "co2", "co2pc", "energypc"]
    head = ["code", "country", "region", "year", "gdp_usd", "population", "inflation_pct", "cpi_index", "co2_mt", "co2_t_per_person", "energy_kwh_per_person"]
    rows = [",".join(head)]
    for c in COUNTRIES:
        ys = set()
        for k in keys:
            s = c["s"].get(k)
            if s:
                ys.update(s["y0"] + i for i, v in enumerate(s["v"]) if v is not None)
        for y in sorted(ys):
            vals = [at(c, k, y) for k in keys]
            rows.append(",".join([c["code"], '"' + c["name"].replace('"', '""') + '"', '"' + c["region"] + '"', str(y)] + ["" if v is None else repr(v) for v in vals]))
    return "\n".join(rows) + "\n"


