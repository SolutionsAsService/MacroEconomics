"""Recompute every figure the site prints from _src/data/ and compare it with site/*.html.

The design run's final-state check (DESIGN-LOG.md, 2g). Rerun it after any change:

    python _src/check_figures.py
"""
import html
import json
import math
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parent
SITE = SRC.parent / "site"
load = lambda n: json.loads((SRC / "data" / f"{n}.json").read_text(encoding="utf-8"))
text = lambda p: html.unescape(re.sub(r"<[^>]+>", " ", (SITE / p).read_text(encoding="utf-8")))
OIL, INFL, PORTS, CFA = load("oil"), load("inflation"), load("ports"), load("cfa_franc")
rows = []


def chk(name, printed, actual):
    rows.append((name, str(printed), str(actual), "match" if str(printed) == str(actual) else "MISMATCH"))


def spread(r):
    v = [x for x in (r["eia"], r["opec"], r["bp"]) if x]
    return max(v) / min(v) - 1 if len(v) >= 2 else None


res = OIL["reserves"]["rows"]
top = res[:10]
can = next(r for r in res if r["country"] == "Canada")
home, oil, ports, cfa, infl = text("index.html"), text("oil.html"), text("ports.html"), text("cfa-franc.html"), text("inflation.html")
raw_home = (SITE / "index.html").read_text(encoding="utf-8")

m = re.search(r"Canada is the extreme: (\d+) bn by the US EIA, ([\d.]+) bn by OPEC\. (\w+) others", home)
chk("home: Canada EIA bn", m.group(1), f"{can['eia'] / 1000:.0f}")
chk("home: Canada OPEC bn", m.group(2), f"{can['opec'] / 1000:.1f}")
words = "No One Two Three Four Five Six Seven Eight Nine".split()
chk("home: others over 25%", m.group(3), words[sum(1 for r in top if spread(r) > .25) - 1])
chk("home: big ratio", re.search(r'class="big-ratio"[^>]*>(\d+)×', raw_home).group(1), f"{can['eia'] / can['opec']:.0f}")
chk("home: n countries in data", re.search(r"of (\d+) countries in the data", home).group(1), len(res))
for r in top:
    chk(f"home bar: {r['country']}", re.search(re.escape(r["country"]) + r"\s+(\d+) bn", home).group(1), f"{r['eia'] / 1000:.0f}")

multi = [r for r in res if spread(r) is not None]
m = re.search(r"Of the (\d+) countries with two or more counts, the highest count is more than 25% above the lowest for (\d+)", oil)
chk("oil: countries with 2+ counts", m.group(1), len(multi))
chk("oil: over 25%", m.group(2), sum(1 for r in multi if spread(r) > .25))
chk("oil: Canada spread %", re.search(r"Canada is the extreme, at ([\d,]+)%", oil).group(1), f"{spread(can) * 100:,.0f}")
prod = OIL["production"]["rows"]
chk("oil: OPEC share of production %", re.search(r"red and hatched: (\d+)%", oil).group(1), f"{sum(p['bpd'] for p in prod if p['opec']) / sum(p['bpd'] for p in prod) * 100:.0f}")

for r in INFL["rows"]:
    if r["first"] == 1960 and not r["missing_years_inside_range"]:
        v = 100.0
        for y in range(1960, r["last"] + 1):
            v /= 1 + r["values"][str(y)] / 100
        chk(f"inflation: {r['name']} 100 from 1960", re.search(re.escape(r["name"]) + r" ([\d.]+)", infl).group(1), f"{v:.1f}")
for r in INFL["rows"]:
    if r["values"] and max(r["values"].values()) > 50:
        y, v = max(r["values"].items(), key=lambda kv: kv[1])
        chk(f"inflation peak: {r['name']}", re.search(r"Off the scale: ([\d,]+)% in " + y, infl).group(1), f"{v:,.0f}")

both = [r for r in PORTS["rows"] if r["stat_rank"]]
movers = [r for r in both if abs(r["admin_rank"] - r["stat_rank"]) >= 10]
m = re.search(r"Of the (\d+) ports with both ranks, (\d+) move 10 or more places", ports)
chk("ports: with both ranks", m.group(1), len(both))
chk("ports: movers", m.group(2), len(movers))
big = max(movers, key=lambda r: abs(r["admin_rank"] - r["stat_rank"]))
chk("ports: biggest mover", re.search(r"off this chart, is ([^(]+?) \(", ports).group(1), big["port"])
chk("ports: in file", re.search(r"(\d+) ports in the inherited file", ports).group(1), len(PORTS["rows"]))

chk("cfa: countries", re.search(r"money of (\d+) African countries", cfa).group(1), sum(len(v["Countries"]) for v in CFA["Institutions"].values()))
chk("cfa: population m", re.search(r"for about (\d+) million people", cfa).group(1), f"{CFA['Overview']['Usage']['Population_2021']:.0f}")


def qf(projects, pool):
    raw = {k: sum(math.sqrt(c) for c in cs) ** 2 - sum(cs) for k, cs in projects.items()}
    t = sum(raw.values())
    return {k: pool * v / t for k, v in raw.items()}


ex = qf({"Project A": [100], "Project B": [10] * 10, "Project C": [25] * 4}, 1000)
fund = text("funding.html")
for k, v in ex.items():
    chk(f"funding example: {k} match", re.search(re.escape(k) + r" +\d+ × \$[\d,]+ +\$[\d,]+ +\$[\d,]+ +\$([\d,]+)", fund).group(1), f"{v:,.0f}")

for r in rows:
    print(" | ".join(r))
bad = sum(r[3] != "match" for r in rows)
print(f"{len(rows)} checks, {bad} mismatches")
raise SystemExit(1 if bad else 0)
