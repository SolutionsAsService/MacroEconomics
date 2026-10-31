"""Build the MacroEconomics site from _src/data/ into site/.

Run _src/clean.py first if the raw files changed. Every chart is drawn here as inline SVG from
the data, and every figure the pages print is computed here, so nothing can drift from the data.
The build fails on a broken link, a missing #anchor, a duplicate id or invalid JSON-LD.

    python _src/build.py
"""
import html
import json
import math
import pathlib
import re
import shutil
import sys

SRC = pathlib.Path(__file__).resolve().parent
ROOT = SRC.parent
OUT = ROOT / "site"
DOMAIN = "https://macroeconomics.shadw.app/"
NAME = "MacroEconomics"
TAGLINE = "Oil, prices, ports and money: the data, with its seams showing"
e = html.escape


def load(name):
    return json.loads((SRC / "data" / f"{name}.json").read_text(encoding="utf-8"))


OIL, INFL, PORTS, CFA, FUND, REPAIRS = (load(n) for n in ("oil", "inflation", "ports", "cfa_franc", "funding", "repairs"))
NAV = [("oil", "Oil"), ("inflation", "Inflation"), ("ports", "Ports"), ("cfa-franc", "CFA franc"),
       ("funding", "Funding public goods"), ("crypto-strategies", "Crypto strategies"), ("data", "About the data")]


def bn(million):
    """Million barrels to a short 'bn' label."""
    v = million / 1000
    return f"{v:,.0f} bn" if v >= 10 else f"{v:,.1f} bn"


def page_url(name):
    return DOMAIN if name == "index" else DOMAIN + name


def split_camel(s):
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s).replace("_", " ")


# ---------- layout

def layout(name, title, description, body, crumb=None, extra_ld=None, scripts=""):
    nav = "\n".join(
        f'        <li><a href="{slug}.html"{" aria-current=\"page\"" if slug == name else ""}>{label}</a></li>' for slug, label in NAV)
    graph = [{"@type": "WebSite", "@id": DOMAIN + "#site", "url": DOMAIN, "name": NAME, "description": TAGLINE, "inLanguage": "en"},
             {"@type": "WebPage", "@id": page_url(name) + "#page", "url": page_url(name), "name": title,
              "description": description, "isPartOf": {"@id": DOMAIN + "#site"}}]
    if crumb:
        graph.append({"@type": "BreadcrumbList", "itemListElement": [
            {"@type": "ListItem", "position": 1, "name": "Home", "item": DOMAIN},
            {"@type": "ListItem", "position": 2, "name": crumb, "item": page_url(name)}]})
    graph += extra_ld or []
    ld = json.dumps({"@context": "https://schema.org", "@graph": graph}, ensure_ascii=False)
    meta = "" if name == "404" else f"""<link rel="canonical" href="{page_url(name)}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="{NAME}">
<meta property="og:title" content="{e(title)}">
<meta property="og:description" content="{e(description)}">
<meta property="og:url" content="{page_url(name)}">
<meta property="og:image" content="{DOMAIN}og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="Proved oil reserves of the ten largest holders as heavy bars, with OPEC's and BP's figures marked as ticks; Canada's OPEC figure sits near zero.">
<meta name="twitter:card" content="summary_large_image">
<script type="application/ld+json">{ld}</script>"""
    robots = "noindex" if name == "404" else "index, follow, max-image-preview:large"
    foot_nav = "".join(f'<li><a href="{s}.html">{l}</a></li>' for s, l in NAV)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title>
<meta name="description" content="{e(description)}">
<meta name="robots" content="{robots}">
{meta}
<link rel="icon" href="favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..900&display=swap">
<link rel="stylesheet" href="css/site.css">
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="site-head">
  <div class="wrap head-row">
    <a class="brand" href="index.html"><span class="brand-name">{NAME}</span><span class="brand-tag">{e(TAGLINE)}</span></a>
    <nav aria-label="Main">
      <ul class="site-nav">
{nav}
      </ul>
    </nav>
  </div>
</header>
<main id="main">
{body}
</main>
<footer class="site-foot">
  <div class="wrap">
    <p class="foot-name">{NAME}</p>
    <p>{e(TAGLINE)}. Open data and code under the MIT license; every repair to the inherited data is listed on <a href="data.html">About the data</a>.</p>
    <nav aria-label="Footer"><ul class="foot-nav"><li><a href="index.html">Home</a></li>{foot_nav}</ul></nav>
  </div>
</footer>
<script src="js/site.js" defer></script>
{scripts}
</body>
</html>
"""


def frame(id_, title, source, chart, table, takeaway=None):
    """The chart frame every topic page repeats: claim title, takeaway, chart, source line, table twin."""
    return f"""<figure class="chart" aria-labelledby="{id_}-t">
  <figcaption>
    <h2 id="{id_}-t">{title}</h2>
    {f'<p class="take">{takeaway}</p>' if takeaway else ""}
  </figcaption>
  {chart}
  <p class="source">{source}</p>
  <details class="twin"><summary>Show the numbers as a table</summary>
    <div class="table-wrap" role="region" aria-label="{e(re.sub('<[^>]+>', '', title))}: table" tabindex="0">{table}</div>
  </details>
</figure>"""


def table(head, rows, num_cols=()):
    th = "".join(f'<th scope="col"{" class=num" if i in num_cols else ""}>{h}</th>' for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(
        (f'<th scope="row">{c}</th>' if i == 0 else f'<td{" class=num" if i in num_cols else ""}>{c}</td>')
        for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


# ---------- charts

WORDS = "No One Two Three Four Five Six Seven Eight Nine".split()


def spread(r):
    """How much the highest count exceeds the lowest, as a fraction; None with fewer than two counts."""
    vals = [v for v in (r["eia"], r["opec"], r["bp"]) if v]
    return max(vals) / min(vals) - 1 if len(vals) >= 2 else None


def hatch(pid):
    """OPEC bars: red plus a diagonal hatch, so membership does not rest on colour alone."""
    return (f'<defs><pattern id="{pid}" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<rect class="hatch-bg" width="7" height="7"/><rect class="hatch-ln" width="2.2" height="7"/></pattern></defs>')


def opec_names(rows):
    return ", ".join(r["country"] for r in rows if r["opec_member"])


def reserves_rows():
    return OIL["reserves"]["rows"][:10]


def fill(cls, pid):
    return f' style="fill:url(#{pid})"' if cls == "opec" else ""


def reserves_svg(tall=False):
    rows = reserves_rows()
    vmax = max(max(v for v in (r["eia"], r["opec"], r["bp"]) if v) for r in rows)
    pid = "hatch-tall" if tall else "hatch-wide"
    n_diff = sum(1 for r in rows if spread(r) and spread(r) > .25)
    can = next(r for r in rows if r["country"] == "Canada")
    canada_ratio = can["eia"] / can["opec"]
    finding = ("The ten largest proved oil reserves by the US EIA's count, with OPEC's and BP's counts marked. "
               f"Canada is the extreme: EIA 170 bn barrels, BP 172 bn, OPEC 4.4 bn, so the EIA's count is {canada_ratio:.0f} times OPEC's (the two define oil differently); {n_diff} of the ten differ by more than a quarter. "
               f"OPEC members, drawn hatched: {opec_names(rows)}.")
    out = [hatch(pid)]
    if tall:
        W, x0, x1, rh, top = 360, 6, 300, 44, 6
        for i, r in enumerate(rows):
            y = top + i * rh
            w = (x1 - x0) * r["eia"] / vmax
            cls = "opec" if r["opec_member"] else "non"
            note = f' · OPEC {bn(r["opec"])} ({r["eia"] / r["opec"]:.0f}×)' if r["opec"] and r["eia"] / r["opec"] > 3 else ""
            out.append(f'<text class="lbl" x="{x0}" y="{y + 13}">{e(r["country"])}</text>'
                       f'<text class="val{" dispute-all" if note else ""}" x="{W - 4}" y="{y + 13}" text-anchor="end">{bn(r["eia"])}{e(note)}</text>'
                       f'<rect class="bar {cls}" x="{x0}" y="{y + 19}" width="{w:.1f}" height="17"{fill(cls, pid)}/>')
            for src, y1, y2 in (("opec", y + 15, y + 28), ("bp", y + 28, y + 41)):
                if r[src]:
                    tx = x0 + (x1 - x0) * r[src] / vmax
                    out.append(f'<line class="tick-bg" x1="{tx:.1f}" x2="{tx:.1f}" y1="{y1}" y2="{y2}"/><line class="tick {src}" x1="{tx:.1f}" x2="{tx:.1f}" y1="{y1}" y2="{y2}"/>')
        H = top + len(rows) * rh + 4
    else:
        W, x0, x1, rh, top = 1000, 200, 820, 41, 10
        big = []
        for i, r in enumerate(rows):
            y = top + i * rh
            w = (x1 - x0) * r["eia"] / vmax
            cls = "opec" if r["opec_member"] else "non"
            end = max(x0 + w, *(x0 + (x1 - x0) * r[s] / vmax for s in ("opec", "bp") if r[s]))
            note = ""
            if r["opec"] and r["eia"] / r["opec"] > 3:
                note = f'<tspan class="dispute" dx="10">OPEC counts {bn(r["opec"])}</tspan>'
                big.append(f'<text class="big-ratio" x="{W - 6}" y="{y + 40}" text-anchor="end">{r["eia"] / r["opec"]:.0f}×</text>'
                           f'<text class="big-note" x="{W - 6}" y="{y + 62}" text-anchor="end">EIA count ÷ OPEC count; definitions differ</text>')
            out.append(f'<text class="lbl" x="{x0 - 14}" y="{y + 20}" text-anchor="end">{e(r["country"])}</text>'
                       f'<rect class="bar {cls}" x="{x0}" y="{y + 4}" width="{w:.1f}" height="24"{fill(cls, pid)}/>'
                       f'<text class="val" x="{end + 12:.1f}" y="{y + 22}">{bn(r["eia"])}{note}</text>')
            for src, y1, y2 in (("opec", y - 2, y + 16), ("bp", y + 16, y + 34)):
                if r[src]:
                    tx = x0 + (x1 - x0) * r[src] / vmax
                    out.append(f'<line class="tick-bg" x1="{tx:.1f}" x2="{tx:.1f}" y1="{y1}" y2="{y2}"/><line class="tick {src}" x1="{tx:.1f}" x2="{tx:.1f}" y1="{y1}" y2="{y2}"/>')
        H = top + len(rows) * rh
        out += big
    return (f'<svg class="reserves {"tall" if tall else "wide"}" viewBox="0 0 {W} {H}" role="img" aria-label="{e(finding)}" '
            f'preserveAspectRatio="xMinYMin meet">{"".join(out)}</svg>')


def reserves_legend():
    return """<ul class="legend" aria-label="How to read the chart">
  <li><span class="key bar-opec" aria-hidden="true"></span>OPEC member (red, hatched)</li>
  <li><span class="key bar-non" aria-hidden="true"></span>Not in OPEC</li>
  <li><span class="key tick-up" aria-hidden="true"></span>OPEC's count (tick above)</li>
  <li><span class="key tick-down" aria-hidden="true"></span>BP's count (tick below)</li>
</ul>"""


def hbars(rows, label_key, value_key, fmt, red_key=None, aria=""):
    W, x0, x1, rh = 1000, 200, 860, 34
    vmax = max(r[value_key] for r in rows)
    out = [hatch("hatch-prod")] if red_key else []
    for i, r in enumerate(rows):
        y = i * rh
        w = (x1 - x0) * r[value_key] / vmax
        cls = "opec" if red_key and r[red_key] else "non"
        out.append(f'<text class="lbl" x="{x0 - 14}" y="{y + 20}" text-anchor="end">{e(r[label_key])}</text>'
                   f'<rect class="bar {cls}" x="{x0}" y="{y + 5}" width="{w:.1f}" height="22"{fill(cls, "hatch-prod")}/>'
                   f'<text class="val" x="{x0 + w + 10:.1f}" y="{y + 21}">{fmt(r[value_key])}</text>')
    return f'<svg class="hbars" viewBox="0 0 {W} {len(rows) * rh}" role="img" aria-label="{e(aria)}">{"".join(out)}</svg>'


def purchasing_power():
    """100 units of money at the start of 1960, after each year's inflation, for rows complete from 1960."""
    series = []
    for r in INFL["rows"]:
        if r["first"] == 1960 and not r["missing_years_inside_range"]:
            v, pts = 100.0, []
            for y in range(1960, r["last"] + 1):
                v = v / (1 + r["values"][str(y)] / 100)
                pts.append((y, v))
            series.append((r["name"], pts))
    return series


def pp_svg(series):
    W, H, x0, x1, y0, y1 = 1000, 420, 60, 800, 20, 380
    X = lambda y: x0 + (x1 - x0) * (y - 1960) / (2022 - 1960)
    Y = lambda v: y1 - (y1 - y0) * v / 100
    dashes = ["", "10 6", "3 5", "14 4 3 4"]
    out = [f'<line class="axis" x1="{x0}" x2="{x1}" y1="{y1}" y2="{y1}"/>']
    for v in (0, 25, 50, 75, 100):
        out.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/><text class="tick-l" x="{x0 - 8}" y="{Y(v) + 5:.1f}" text-anchor="end">{v}</text>')
    for yr in (1960, 1980, 2000, 2022):
        out.append(f'<text class="tick-l" x="{X(yr):.1f}" y="{y1 + 24}" text-anchor="middle">{yr}</text>')
    label_y, last = {}, -99.0
    for name, pts in sorted(series, key=lambda s: -s[1][-1][1]):  # top label first, each at least 20 below the last
        label_y[name] = max(Y(pts[-1][1]) + 5, last + 20)
        last = label_y[name]
    for i, (name, pts) in enumerate(series):
        d = "M" + " L".join(f"{X(y):.1f} {Y(v):.1f}" for y, v in pts)
        out.append(f'<path class="pp" stroke-dasharray="{dashes[i % 4]}" d="{d}"/>')
        out.append(f'<text class="lbl small" x="{x1 + 10}" y="{label_y[name]:.1f}">{e(name)} {pts[-1][1]:.1f}</text>')
    finding = "What 100 units of money from the start of 1960 buys in each later year: " + "; ".join(
        f"{n} {p[-1][1]:.1f} by {p[-1][0]}" for n, p in series) + "."
    return f'<svg class="pp-chart" viewBox="0 0 {W} {H}" role="img" aria-label="{e(finding)}">{"".join(out)}</svg>'


def multiples():
    cells = []
    for r in INFL["rows"]:
        W, H, base, cap = 200, 70, 50, 50
        X = lambda y: 4 + (W - 8) * (y - 1960) / 63
        bars = []
        peak = max(r["values"].items(), key=lambda kv: kv[1]) if r["values"] else None
        for y in range(1960, 2023):
            v = r["values"].get(str(y))
            if v is None:
                if r["first"] and r["first"] <= y <= r["last"]:
                    bars.append(f'<circle class="gap" cx="{X(y) + 1.2:.1f}" cy="{base + 4}" r="1.2"/>')
                continue
            h = min(v, cap) * (base - 4) / cap if v >= 0 else max(v, -10) * (base - 4) / cap
            cls = "off" if v > cap else ("neg" if v < 0 else "pos")
            y_top = base - h if h >= 0 else base
            bars.append(f'<rect class="m {cls}" x="{X(y):.1f}" y="{y_top:.1f}" width="2.4" height="{abs(h):.1f}"/>')
        over = f'<p class="peak">Off the scale: {peak[1]:,.0f}% in {peak[0]}</p>' if peak and peak[1] > cap else ""
        aria = f"{r['name']}: annual inflation {r['first']} to {r['last']}" + (f"; peak {peak[1]:,.0f}% in {peak[0]}" if peak else "")
        cells.append(f"""<li><h3>{e(r["name"])}{' <span class="agg">region</span>' if r["kind"] == "aggregate" else ""}</h3>
  <svg viewBox="0 0 {W} {H}" role="img" aria-label="{e(aria)}"><line class="base" x1="0" x2="{W}" y1="{base}" y2="{base}"/>{"".join(bars)}</svg>
  <p class="span">{r["first"]} to {r["last"]}</p>{over}</li>""")
    return f'<ul class="multiples">{"".join(cells)}</ul>'


def dumbbells(rows):
    rows = sorted([r for r in rows if r["stat_rank"]], key=lambda r: r["admin_rank"])[:30]
    rmax = max(max(r["admin_rank"], r["stat_rank"]) for r in rows)
    W, x0, x1, rh = 1000, 260, 960, 26
    X = lambda k: x0 + (x1 - x0) * (k - 1) / (rmax - 1)
    out = [f'<text class="tick-l" x="{X(1):.1f}" y="14" text-anchor="middle">rank 1</text>',
           f'<text class="tick-l" x="{X(rmax):.1f}" y="14" text-anchor="end">rank {rmax}</text>']
    for i, r in enumerate(rows):
        y = 34 + i * rh
        moved = abs(r["admin_rank"] - r["stat_rank"]) >= 10
        out.append(f'<text class="lbl small" x="{x0 - 12}" y="{y + 5}" text-anchor="end">{e(r["port"])} <tspan class="dim">{e(r["country"])}</tspan></text>'
                   f'<line class="db{" moved" if moved else ""}" x1="{X(r["admin_rank"]):.1f}" x2="{X(r["stat_rank"]):.1f}" y1="{y}" y2="{y}"/>'
                   f'<circle class="sta" cx="{X(r["stat_rank"]):.1f}" cy="{y}" r="7.5"/><circle class="adm" cx="{X(r["admin_rank"]):.1f}" cy="{y}" r="4.5"/>')
    aria = "The 30 best-ranked container ports by the administrative method, each joined to its rank by the statistical method; lines in red move 10 or more places."
    return f'<svg class="dumbbell" viewBox="0 0 {W} {34 + len(rows) * rh}" role="img" aria-label="{e(aria)}">{"".join(out)}</svg>', rows


# ---------- quadratic funding (same rule as js/qf.js)

def qf(projects, pool):
    raw = {k: sum(math.sqrt(c) for c in cs) ** 2 - sum(cs) for k, cs in projects.items()}
    tot = sum(raw.values())
    return {k: (pool * v / tot if tot > 0 else 0.0) for k, v in raw.items()}, raw


QF_EXAMPLE = {"Project A": [100], "Project B": [10] * 10, "Project C": [25] * 4}
QF_POOL = 1000


# ---------- pages

def page_index():
    rows = reserves_rows()
    canada = next(r for r in rows if r["country"] == "Canada")
    n_diff = sum(1 for r in rows if spread(r) and spread(r) > .25)
    body = f"""<section class="hero" aria-labelledby="hero-h">
  <div class="wrap">
    <h1 id="hero-h">Who holds the world's oil? It depends who is counting.</h1>
    <p class="hero-take">Three official counts of proved reserves, ten countries. Canada is the extreme: {bn(canada['eia'])} by the US EIA, {bn(canada['opec'])} by OPEC. {WORDS[n_diff - 1]} others here differ by a quarter or more.</p>
    <div class="hero-chart">
      {reserves_svg()}
      {reserves_svg(tall=True)}
    </div>
    {reserves_legend()}
    <p class="source">Proved oil reserves, billions of barrels. Bars: US Energy Information Administration; ticks: OPEC and BP. Ten largest by the EIA count, of {len(OIL["reserves"]["rows"])} countries in the data. The year of each count is not stated in the inherited file. <a href="oil.html">More on oil</a></p>
  </div>
</section>
<section class="section wrap" aria-labelledby="topics-h">
  <h2 id="topics-h">What is here</h2>
  <p class="lede">Six small datasets about the economy, repaired and charted, each with its sources and its gaps shown.</p>
  <div class="groups">
    <section aria-labelledby="g1"><h3 id="g1" class="group-h">Energy and trade</h3><ul class="cards">
      <li><a href="oil.html"><strong>Oil</strong><span>Who has the oil, who pumps it, and why the counts disagree.</span></a></li>
      <li><a href="ports.html"><strong>Ports</strong><span>Which container ports work best, by two methods that do not always agree.</span></a></li></ul></section>
    <section aria-labelledby="g2"><h3 id="g2" class="group-h">Money and prices</h3><ul class="cards">
      <li><a href="inflation.html"><strong>Inflation</strong><span>What 100 units of money from 1960 buys today, in four countries.</span></a></li>
      <li><a href="cfa-franc.html"><strong>CFA franc</strong><span>Why fourteen African countries' money is fixed to the euro.</span></a></li></ul></section>
    <section aria-labelledby="g3"><h3 id="g3" class="group-h">Funding and investing</h3><ul class="cards">
      <li><a href="funding.html"><strong>Funding public goods</strong><span>How quadratic funding splits a matching pool. Try the calculator.</span></a></li>
      <li><a href="crypto-strategies.html"><strong>Crypto strategies</strong><span>Common ways people invest in crypto, each beside how it loses.</span></a></li></ul></section>
  </div>
  <p><a href="data.html">About the data: every source, gap and repair →</a></p>
</section>"""
    return layout("index", "Who Holds the World's Oil? Macroeconomics Data, Charted · MacroEconomics",
                  "Proved oil reserves counted three ways, inflation as purchasing power, port rankings, the CFA franc and quadratic funding: small economics datasets, repaired and charted with their sources and gaps shown.", body)


def page_oil():
    res = OIL["reserves"]["rows"]
    prod = OIL["production"]["rows"]
    multi = [r for r in res if spread(r) is not None]
    wide = [r for r in multi if spread(r) > .25]
    t_res = table(["Country", "US EIA", "OPEC", "BP", "Spread", "Production 2021, barrels a day", "Years of reserves"],
                  [[e(r["country"]) + (" <span class=agg>OPEC</span>" if r["opec_member"] else ""),
                    f'{r["eia"]:,.0f}' if r["eia"] else "not given", f'{r["opec"]:,.0f}' if r["opec"] else "not given",
                    f'{r["bp"]:,.0f}' if r["bp"] else "not given", f"{spread(r) * 100:,.0f}%" if spread(r) is not None else "",
                    f'{r["production_2021_bpd"]:,.0f}' if r["production_2021_bpd"] else "not given",
                    f'{r["years_of_reserves"]:,.0f}' if r["years_of_reserves"] else ""] for r in res], num_cols=(1, 2, 3, 4, 5, 6))
    biggest = max(multi, key=spread)
    top_prod = prod[0]
    opec_share = sum(p["bpd"] for p in prod if p["opec"]) / sum(p["bpd"] for p in prod) * 100
    t_prod = table(["Country", "Barrels a day", "OPEC"], [[e(p["country"]), f'{p["bpd"]:,}', "yes" if p["opec"] else "no"] for p in prod], num_cols=(1,))
    t_top = table(["Country", "US EIA", "OPEC", "BP"], [[e(r["country"]), f'{r["eia"]:,.0f}', f'{r["opec"]:,.0f}', f'{r["bp"]:,.0f}'] for r in reserves_rows()], num_cols=(1, 2, 3))
    body = f"""<header class="page-head wrap">
  <p class="kicker">Oil</p>
  <h1>Who has the oil, and who is counting?</h1>
  <p class="plain"><strong>In plain words:</strong> "proved reserves" is oil that can be pumped at today's prices with today's technology. Three bodies publish their own count, and on most countries they agree. Where they do not, it is usually a difference in what counts as oil.</p>
</header>
<section class="section wrap">
  {frame("res", f"The counts disagree, most of all on {e(biggest['country'])}", f"Proved reserves, billions of barrels. Bars: US EIA; ticks above: OPEC; ticks below: BP. Ten largest by the EIA count, of {len(res)} countries. Year of each count not stated in the inherited file.", '<div class="hero-chart">' + reserves_svg() + reserves_svg(tall=True) + "</div>" + reserves_legend(), t_top,
         f"Of the {len(multi)} countries with two or more counts, the highest count is more than 25% above the lowest for {len(wide)}. {e(biggest['country'])} is the extreme, at {spread(biggest) * 100:,.0f}%: OPEC's figure appears to leave out most of its oil sands, which the EIA and BP include.")}
  {frame("prod", f"In September 2023 the {e(top_prod['country'])} pumped the most oil", f"Crude oil production, barrels per day, September 2023, the ten largest producers in the inherited file. OPEC members red and hatched: {opec_share:.0f}% of these ten countries' output.", hbars(prod, "country", "bpd", lambda v: f"{v / 1e6:.1f} m", "opec", "Crude oil production in September 2023, barrels per day: " + "; ".join(f"{p['country']} {p['bpd'] / 1e6:.1f} million" for p in prod) + ". OPEC members, drawn hatched: " + ", ".join(p["country"] for p in prod if p["opec"]) + "."), t_prod,
         "Holding the most oil and pumping the most are different lists: Venezuela holds the most proved reserves and is not among the ten largest producers.")}
  <section aria-labelledby="all-h">
    <h2 id="all-h">All {len(res)} countries</h2>
    <p>Millions of barrels of proved reserves by each count. <strong>Spread</strong> is how much the highest count exceeds the lowest. <strong>Years of reserves</strong> is the EIA count divided by 2021 production: how long the reserves would last at that rate, if nothing changed. Mexico's entry is cut off in the inherited file; only its EIA figure survives.</p>
    <div class="table-wrap" role="region" aria-label="Proved reserves of all countries in the data" tabindex="0">{t_res}</div>
  </section>
</section>"""
    return layout("oil", "Oil Reserves and Production: Who Has the Oil and Who Is Counting · MacroEconomics",
                  "Proved oil reserves by the US EIA, OPEC and BP side by side for 41 countries, where they disagree and why, and the ten largest producers in September 2023.", body, crumb="Oil")


def page_inflation():
    series = purchasing_power()
    pp_table = table(["Year"] + [e(n) for n, _ in series],
                     [[str(y)] + [f"{dict(p)[y]:.1f}" for _, p in series] for y in range(1960, 2023, 5)] + [["2022"] + [f"{p[-1][1]:.1f}" for _, p in series]],
                     num_cols=tuple(range(1, len(series) + 1)))
    worst = min(series, key=lambda s: s[1][-1][1])
    best = max(series, key=lambda s: s[1][-1][1])
    body = f"""<header class="page-head wrap">
  <p class="kicker">Inflation</p>
  <h1>What does 100 from 1960 buy today?</h1>
  <p class="plain"><strong>In plain words:</strong> inflation is how fast prices rise. Instead of a line of yearly percentages, this page follows 100 units of money from the start of 1960 and shows how much of 1960's shopping it would still buy each year.</p>
</header>
<section class="section wrap">
  {frame("pp", f"In {e(worst[0])}, 100 from 1960 buys {worst[1][-1][1]:.1f} by {worst[1][-1][0]}", f"World Bank, consumer price inflation (annual %), compounded from 1960. Only the {len(series)} rows in the inherited file with every year from 1960 can be followed this way.", pp_svg(series), pp_table,
         f"Even in {e(best[0])}, where prices rose slowest of the four, 100 from 1960 buys {best[1][-1][1]:.1f} by {best[1][-1][0]}.")}
  <section aria-labelledby="mult-h">
    <h2 id="mult-h">Every row in the file</h2>
    <p>Each bar is one year's inflation, 1960 to 2022, on the same scale. Bars stop at 50%; anything higher is drawn red and its peak named. A dot on the line is a year with no figure inside the row's range; nothing is drawn where the series has not started. Regions are marked: they are averages, not countries.</p>
    {multiples()}
    <p class="source">World Bank, Inflation, consumer prices (annual %), indicator FP.CPI.TOTL.ZG. The inherited file holds {len(INFL["rows"])} rows, the World Bank's from Aruba to Bangladesh only; the other economies are not in it. <a href="data.html#inflation">What was repaired</a></p>
  </section>
</section>"""
    return layout("inflation", "Inflation as Purchasing Power: What 100 from 1960 Buys Today · MacroEconomics",
                  "Inflation shown as purchasing power: what 100 units of money from 1960 buys each year in Australia, Austria, Belgium and Burkina Faso, plus yearly inflation for 18 World Bank rows with gaps shown.", body, crumb="Inflation")


def page_ports():
    rows = PORTS["rows"]
    svg, top = dumbbells(rows)
    both = [r for r in rows if r["stat_rank"]]
    shown = {r["admin_rank"] for r in top}
    skipped = [k for k in range(1, top[-1]["admin_rank"] + 1) if k not in shown]
    no_stat = {r["admin_rank"] for r in rows if not r["stat_rank"]}
    movers = sorted([r for r in rows if r["stat_rank"] and abs(r["admin_rank"] - r["stat_rank"]) >= 10],
                    key=lambda r: -abs(r["admin_rank"] - r["stat_rank"]))
    t_top = table(["Port", "Country", "Administrative rank", "Statistical rank"], [[e(r["port"]), e(r["country"]), r["admin_rank"], r["stat_rank"]] for r in top], num_cols=(2, 3))
    t_all = table(["Port", "Country", "Administrative rank", "Points", "Statistical rank", "Index"],
                  [[e(r["port"]), e(r["country"]), r["admin_rank"], f'{r["admin_points"]:.1f}', r["stat_rank"] or "not given", f'{r["stat_index"]:.1f}' if r["stat_index"] is not None else "not given"]
                   for r in sorted(rows, key=lambda r: r["admin_rank"])], num_cols=(2, 3, 4, 5))
    big = movers[0]
    body = f"""<header class="page-head wrap">
  <p class="kicker">Ports</p>
  <h1>Which container ports work best? Two methods, two answers.</h1>
  <p class="plain"><strong>In plain words:</strong> the Container Port Performance Index ranks ports by how long ships spend in them. It ranks every port twice, by two different methods. Near the top they mostly agree; further down some ports jump hundreds of places.</p>
</header>
<section class="section wrap">
  {frame("ports", "Near the top the two methods mostly agree", f"Container Port Performance Index 2021, World Bank and S&amp;P Global Market Intelligence. The {len(top)} best-ranked ports by the administrative method that have both ranks{' (administrative rank ' + ', '.join(f'{k}' + (' has no statistical rank' if k in no_stat else ' is not in the inherited file') for k in skipped) + ', so it is left out)' if skipped else ''}. Filled dot: administrative rank; ring: statistical rank; red line: 10 or more places apart. {len(rows)} ports in the inherited file, of an index that ranks {max(r['admin_rank'] for r in rows)}.", svg, t_top,
         f"Of the {len(both)} ports with both ranks, {len(movers)} move 10 or more places between the two methods. The largest, off this chart, is {e(big['port'])} ({e(big['country'])}): ranked {big['admin_rank']} by one method and {big['stat_rank']} by the other.")}
  <section aria-labelledby="all-ports">
    <h2 id="all-ports">All {len(rows)} ports in the file</h2>
    <div class="table-wrap tall" role="region" aria-label="All ports with both ranks" tabindex="0">{t_all}</div>
  </section>
</section>"""
    return layout("ports", "Container Port Rankings 2021: Two Methods, Two Answers · MacroEconomics",
                  "The Container Port Performance Index 2021 ranks each port by two methods. See where they agree, which ports move 10 or more places, and all 287 ports in the data.", body, crumb="Ports")


FR = {
    "Restricts sovereignty of African member states": "Restreint la souveraineté des États africains membres",
    "Monetary policy in hands of European Central Bank": "La politique monétaire est entre les mains de la Banque centrale européenne",
    "Facilitates exports and imports between France and member countries": "Facilite les exportations et les importations entre la France et les pays membres",
}


def bilingual(c):
    fr = FR.get(c)
    return f'<li>{e(c)}{f" <span class=fr-line lang=fr>{e(fr)}</span>" if fr else ""}</li>'


def page_cfa():
    o, h, inst = CFA["Overview"], CFA["History"], CFA["Institutions"]
    zones = "".join(f"""<section class="zone" aria-labelledby="z-{k}"><h3 id="z-{k}">{e(split_camel(k).replace(" C F A", " CFA"))} <span class="code">{v['ISO_Code']}</span></h3>
      <p>Central bank: {e(v['Central_Bank'])}</p><ul>{"".join(f"<li>{e(c)}</li>" for c in v['Countries'])}</ul></section>""" for k, v in inst.items())
    n = sum(len(v["Countries"]) for v in inst.values())
    steps = "".join(f"<li><strong>{e(k)}</strong> {e(v)}</li>" for k, v in h["ExchangeRateHistory"].items())
    changes = "".join(f"<li>{e(c)}</li>" for c in h["ChangesInCountriesUsingFranc"])
    body = f"""<header class="page-head wrap">
  <p class="kicker">CFA franc</p>
  <h1>Why is the money of {n} African countries fixed to the euro?</h1>
  <p class="plain"><strong>In plain words:</strong> the CFA franc is two currencies shared by {n} countries in West and Central Africa. Both are pegged to the euro at a fixed rate, a link that began with France's colonies in 1945.</p>
  <p class="plain fr" lang="fr"><strong>En bref :</strong> le franc CFA désigne deux monnaies partagées par {n} pays d'Afrique de l'Ouest et d'Afrique centrale. Toutes deux sont arrimées à l'euro à un taux fixe, un lien qui remonte aux colonies françaises en 1945.</p>
</header>
<section class="peg" aria-labelledby="peg-h">
  <div class="wrap">
    <h2 id="peg-h" class="sr-only">The fixed rate</h2>
    <p class="peg-rate"><span>€1</span> <span class="eq">=</span> <span>655.957</span> <span class="unit">F.CFA</span></p>
    <p class="peg-note">The same rate since 1999, for about {o['Usage']['Population_2021']:.0f} million people (2021).</p>
  </div>
</section>
<section class="section wrap">
  <div class="zones">{zones}</div>
  <section aria-labelledby="debate-h" class="debate">
    <h2 id="debate-h">The case against, and the case made for it</h2>
    <div class="two">
      <div><h3>Criticisms <span class="fr-h" lang="fr">Critiques</span></h3><ul>{"".join(bilingual(c) for c in o["Criticism"])}</ul></div>
      <div><h3>Claimed benefits <span class="fr-h" lang="fr">Avantages avancés</span></h3><ul>{"".join(bilingual(c) for c in o["ClaimedBenefits"])}</ul></div>
    </div>
    <p class="hint">As listed in the inherited data, which cites no sources for either side. The benefit was filed under "Criticism" there; it is moved here so each claim stands on the side it argues.</p>
  </section>
  <section aria-labelledby="hist-h">
    <h2 id="hist-h">How the rate has moved</h2>
    <ol class="steps">{steps}</ol>
    <h3>Who joined and who left</h3>
    <ul class="changes">{changes}</ul>
  </section>
  <section aria-labelledby="reform-h">
    <h2 id="reform-h">Reform</h2>
    <p>On {e(CFA["CriticismAndReplacementInWestAfrica"]["Replacement"]["InitiativeDate"])}, West African members announced a replacement currency, the {e(CFA["CriticismAndReplacementInWestAfrica"]["Replacement"]["NewCurrencyName"])}. In {e(o["Reform"]["Date"])} France's National Assembly ended its formal engagement, including the requirement to hold half of members' foreign reserves at the French Treasury. On {e(CFA["DebateOnEndingCentralAfricanCFA"]["Date"])} France said it was open to proposals to reform the Central African franc.</p>
    <p class="fr" lang="fr">Le 21 décembre 2019, les pays membres d'Afrique de l'Ouest ont annoncé une nouvelle monnaie, l'Eco. En mai 2020, l'Assemblée nationale française a mis fin à l'engagement formel de la France, y compris l'obligation de déposer la moitié des réserves de change des membres au Trésor français. Le 25 avril 2023, la France s'est dite ouverte à des propositions de réforme du franc CFA d'Afrique centrale.</p>
  </section>
</section>"""
    return layout("cfa-franc", "The CFA Franc: Why 14 African Currencies Are Fixed to the Euro · MacroEconomics",
                  "The CFA franc explained: two currencies, 14 countries, a fixed rate of 655.957 francs to the euro since 1999, its history, the criticisms and the case made for it, with a summary in French.", body, crumb="CFA franc")


def page_funding():
    q = FUND["quadratic_funding"]
    match, raw = qf(QF_EXAMPLE, QF_POOL)
    ex_rows = [[k, f"{len(v)} × ${v[0]:,}", f"${sum(v):,}", f"${raw[k]:,.0f}", f"${match[k]:,.0f}", f"${sum(v) + match[k]:,.0f}"] for k, v in QF_EXAMPLE.items()]
    ex = table(["Project", "Contributions", "Total given", "Before scaling: (Σ√c)² − Σc", "Match", "Receives"], ex_rows, num_cols=(2, 3, 4, 5))
    inputs = "".join(f"""<div class="field"><label for="qf-{i}">{k}: contributions in dollars, separated by commas</label>
      <input id="qf-{i}" name="{k}" inputmode="decimal" value="{', '.join(str(c) for c in v)}" aria-describedby="qf-help"></div>""" for i, (k, v) in enumerate(QF_EXAMPLE.items()))
    mech = "".join(f"""<section class="mech" aria-labelledby="m-{k}"><h3 id="m-{k}">{e(split_camel(k))}</h3><p>{e(v["Description"])}</p>
      <p class="mini"><strong>For:</strong> {e("; ".join(v.get("Objectives", [])))}</p><p class="mini"><strong>Examples:</strong> {e("; ".join(v.get("Examples", [])))}</p></section>"""
                   for k, v in FUND["mechanisms"].items())
    body = f"""<header class="page-head wrap">
  <p class="kicker">Funding public goods</p>
  <h1>Many small gifts beat one large one: try it.</h1>
  <p class="plain"><strong>In plain words:</strong> quadratic funding is a way to share out a pot of matching money. Each project gets more of the pot the more <em>people</em> back it, not the more money they give. Change the numbers below and watch the pot move.</p>
</header>
<section class="section wrap">
  <form class="qf" id="qf" novalidate>
    <fieldset>
      <legend>Three projects and a matching pool</legend>
      <p class="hint" id="qf-help">Each number is one person's contribution. Example: "10, 10, 10" is three people giving $10 each.</p>
      {inputs}
      <div class="field"><label for="qf-pool">Matching pool in dollars</label><input id="qf-pool" name="pool" inputmode="decimal" value="{QF_POOL}"></div>
    </fieldset>
    <p class="err" id="qf-err" role="alert" hidden></p>
  </form>
  <div class="qf-out" id="qf-out" aria-live="polite">
    <h2>What each project receives</h2>
    <div class="table-wrap" role="region" aria-label="Quadratic funding result" tabindex="0">{ex}</div>
  </div>
  <p class="take">With the starting numbers, Project A's one $100 gift earns no match at all, while Project B's ten $10 gifts, the same $100 in total, take ${match['Project B']:,.0f} of the ${QF_POOL:,} pool.</p>
  <section aria-labelledby="rule-h">
    <h2 id="rule-h">The rule</h2>
    <p>For each project, add up the square roots of its contributions, square that, and subtract what was given: (Σ√c)² − Σc. Then scale every project's result by the same factor so that all the matches add up to the pool. A project receives its contributions plus its match. With one contributor the result is zero; with many small ones it grows fast.</p>
    <p class="was">The inherited data stated it as <del>"Direct contributions + (Square of the sum of the square roots of individual contributions) * Matching pool"</del>, which multiplies by the whole pool and would pay out many times the pot.</p>
    <p class="source">The rule as set out by Vitalik Buterin, Zoë Hitzig and E. Glen Weyl, "A Flexible Design for Funding Public Goods" (2019). Used by Gitcoin Grants and clr.fund.</p>
  </section>
  <section aria-labelledby="qf-two">
    <h2 id="qf-two">What it is good at, and where it breaks</h2>
    <div class="two"><div><h3>For</h3><ul>{"".join(f"<li>{e(x)}</li>" for x in q["Benefits"])}</ul></div>
    <div><h3>Against</h3><ul>{"".join(f"<li>{e(x)}</li>" for x in q["Challenges"])}</ul></div></div>
  </section>
  <section aria-labelledby="mech-h">
    <h2 id="mech-h">Other ways projects get funded</h2>
    <div class="mechs">{mech}</div>
  </section>
</section>"""
    return layout("funding", "Quadratic Funding Calculator: How Matching Pools Favor Many Small Gifts · MacroEconomics",
                  "Try quadratic funding: change the contributions to three projects and see how a matching pool is split. The rule explained, the inherited formula corrected, and nine other ways projects get funded.",
                  body, crumb="Funding public goods", scripts='<script src="js/qf.js" defer></script>')


def page_crypto():
    rows = "".join(f"""<section class="strat" aria-labelledby="s-{k}"><h2 id="s-{k}">{e(split_camel(k))}</h2><p>{e(v["Description"])}</p>
      <div class="two"><div><h3>The case made for it</h3><ul>{"".join(f"<li>{e(x)}</li>" for x in v.get("Benefits", []))}</ul></div>
      <div><h3>How it loses</h3><ul>{"".join(f"<li>{e(x)}</li>" for x in v.get("Risks", []))}</ul></div></div></section>"""
                   for k, v in FUND["allocation_strategies"].items())
    body = f"""<header class="page-head wrap">
  <p class="kicker">Crypto strategies</p>
  <h1>Ways people invest in crypto, and how each one loses</h1>
  <p class="plain"><strong>In plain words:</strong> these are common plans for buying and holding cryptocurrency. Each is set beside the way it goes wrong, at the same weight.</p>
  <p class="warn"><strong>This is not investment advice.</strong> Cryptocurrencies can lose most or all of their value. Nothing here is a forecast or a recommendation, and no returns are shown because the inherited data holds none.</p>
</header>
<section class="section wrap strats">
{rows}
</section>"""
    return layout("crypto-strategies", "Crypto Investment Strategies and How Each One Loses · MacroEconomics",
                  "Eight common crypto investment strategies, from dollar-cost averaging to hedging, each set beside the way it goes wrong. Not investment advice.", body, crumb="Crypto strategies")


def page_data():
    by = {}
    for r in REPAIRS:
        by.setdefault(r["dataset"], []).append(r["repair"])
    sets = [("oil", "Oil", "oil.json", f"US EIA, OPEC and BP proved reserves for {len(OIL['reserves']['rows'])} countries (million barrels); production for 10 countries, September 2023 (barrels per day)."),
            ("inflation", "Inflation", "inflation.json", f"World Bank consumer price inflation, annual %, {len(INFL['rows'])} rows (Aruba to Bangladesh), 1960 to 2022."),
            ("ports", "Ports", "ports.json", f"Container Port Performance Index 2021, {len(PORTS['rows'])} ports, two ranks each."),
            ("cfa_franc", "CFA franc", "cfa_franc.json", "History, institutions, members, rate history and reform of the CFA franc."),
            ("funding", "Funding and crypto strategies", "funding.json", "Nine funding mechanisms, quadratic funding, and eight crypto allocation strategies, as written in the inherited files."),
            ("price_prediction", "Price prediction", None, "A script, not data.")]
    items = "".join(f"""<section aria-labelledby="{k.replace('_', '-')}"><h2 id="{k.replace('_', '-')}">{t}</h2><p>{d}</p>
      {f'<p><a href="data/{f}">Download {f}</a></p>' if f else ""}
      {'<h3>Repairs</h3><ul>' + "".join(f"<li>{e(x)}</li>" for x in by.get(k, [])) + "</ul>" if by.get(k) else ""}</section>""" for k, t, f, d in sets)
    ds = [{"@type": "Dataset", "name": f"MacroEconomics: {t}", "description": d, "license": "https://opensource.org/licenses/MIT",
           "distribution": {"@type": "DataDownload", "encodingFormat": "application/json", "contentUrl": DOMAIN + "data/" + f}}
          for k, t, f, d in sets if f]
    body = f"""<header class="page-head wrap">
  <p class="kicker">About the data</p>
  <h1>Where every number comes from, and what was fixed</h1>
  <p class="plain"><strong>In plain words:</strong> this site began as a folder of data files, three of which could not be opened by a program. They were repaired, not replaced: the originals are kept unchanged, and every repair is listed below.</p>
</header>
<section class="section wrap prose">
{items}
  <section aria-labelledby="raw-h"><h2 id="raw-h">The originals</h2>
  <p>The inherited files are kept unchanged in <code>_src/raw/</code> in the <a href="https://github.com/SolutionsAsService/MacroEconomics">source repository</a>, and the repository as it stood before this redesign is on the branch <code>legacy-2026-10-30</code>. The repairs are made by <code>_src/clean.py</code>; anyone can rerun them.</p></section>
  <section aria-labelledby="fix-h"><h2 id="fix-h">Found a mistake?</h2>
  <p>Open an issue on the repository with the figure, the page and, ideally, a source.</p></section>
</section>"""
    return layout("data", "About the Data: Sources, Gaps and Repairs · MacroEconomics",
                  "Where every dataset on MacroEconomics comes from, what it covers, what is missing, and each repair made to the inherited files, with JSON downloads.",
                  body, crumb="About the data", extra_ld=ds)


def page_404():
    body = """<header class="page-head wrap"><p class="kicker">Not found</p><h1>That page is not in the data.</h1>
  <p class="plain">Try <a href="index.html">the home page</a> or <a href="data.html">about the data</a>.</p></header>"""
    return layout("404", "Page not found · MacroEconomics", "This page does not exist.", body)


# ---------- assets and checks

FAVICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
<style>.p{fill:#f3eee3}.i{fill:#151515}.r{fill:#c8102e}@media (prefers-color-scheme: dark){.p{fill:#151515}.i{fill:#f3eee3}.r{fill:#ff5a5f}}</style>
<rect class="p" width="32" height="32" rx="4"/>
<rect class="r" x="4" y="5" width="24" height="5"/>
<rect class="i" x="4" y="13.5" width="17" height="5"/>
<rect class="i" x="4" y="22" width="10" height="5"/>
</svg>
"""


def check():
    bad = []
    for p in OUT.glob("*.html"):
        s = p.read_text(encoding="utf-8")
        for href in re.findall(r'href="([^"#:?]+)(?:#[^"]*)?"', s):
            if not href.startswith(("http", "//")) and not (OUT / href).exists():
                bad.append(f"{p.name} -> {href}")
        for target, frag in re.findall(r'href="([a-z0-9-]+\.html)#([^"]+)"', s):
            if f'id="{frag}"' not in (OUT / target).read_text(encoding="utf-8"):
                bad.append(f"{p.name} -> {target}#{frag}")
        for frag in re.findall(r'href="#([^"]+)"', s):
            if f'id="{frag}"' not in s:
                bad.append(f"{p.name} -> #{frag}")
        ids = re.findall(r' id="([^"]+)"', s)
        bad += [f"{p.name}: duplicate id {i}" for i in sorted(set(ids)) if ids.count(i) > 1]
        for block in re.findall(r'<script type="application/ld\+json">(.*?)</script>', s, re.S):
            json.loads(block)
    if bad:
        sys.exit("problems:\n  " + "\n  ".join(bad))


def main():
    pages = {"index": page_index(), "oil": page_oil(), "inflation": page_inflation(), "ports": page_ports(),
             "cfa-franc": page_cfa(), "funding": page_funding(), "crypto-strategies": page_crypto(), "data": page_data(), "404": page_404()}
    for n, s in pages.items():
        (OUT / f"{n}.html").write_text(s, encoding="utf-8")
    (OUT / "data").mkdir(exist_ok=True)
    for f in ("oil", "inflation", "ports", "cfa_franc", "funding", "repairs"):
        shutil.copy(SRC / "data" / f"{f}.json", OUT / "data" / f"{f}.json")
    (OUT / "favicon.svg").write_text(FAVICON, encoding="utf-8")
    listed = [n for n in pages if n != "404"]
    (OUT / "sitemap.xml").write_text('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
                                     + "\n".join(f"  <url><loc>{page_url(n)}</loc></url>" for n in listed) + "\n</urlset>\n", encoding="utf-8")
    (OUT / "robots.txt").write_text(f"User-agent: *\nAllow: /\n\nSitemap: {DOMAIN}sitemap.xml\n", encoding="utf-8")
    llms = [f"# {NAME}", "", f"> {TAGLINE}. Small economics datasets, repaired and charted. Data: {DOMAIN}data/ (MIT).", ""]
    for n in listed:
        t = html.unescape(re.search(r"<title>(.*?)</title>", pages[n]).group(1))
        d = html.unescape(re.search(r'<meta name="description" content="(.*?)">', pages[n]).group(1))
        llms.append(f"- [{t}]({page_url(n)}): {d}")
    (OUT / "llms.txt").write_text("\n".join(llms) + "\n", encoding="utf-8")
    check()
    print(f"built {len(pages)} pages into site/, links checked")


if __name__ == "__main__":
    main()
