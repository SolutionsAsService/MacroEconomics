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

tiles = dict(re.findall(r'<span class="t-topic">(.*?)</span><span class="t-fig">(.*?)</span>', raw_home))
pp_min = min((r for r in INFL["rows"] if r["first"] == 1960 and not r["missing_years_inside_range"]),
             key=lambda r: math.prod(1 / (1 + r["values"][str(y)] / 100) for y in range(1960, r["last"] + 1)))
pp_val = 100 * math.prod(1 / (1 + pp_min["values"][str(y)] / 100) for y in range(1960, pp_min["last"] + 1))
chk("poster: Oil", tiles["Oil"], f"{can['eia'] / can['opec']:.0f}×")
chk("poster: Inflation", tiles["Inflation"], f"{pp_val:.1f}")
chk("poster: Ports", tiles["Ports"], len(movers))
chk("poster: CFA franc", tiles["CFA franc"], CFA["Overview"]["ExchangeRate"]["FixedToEuro"].split()[-1])
chk("poster: Funding public goods", html.unescape(tiles["Funding public goods"]), f"${ex['Project B']:,.0f}")
chk("poster: Crypto strategies", tiles["Crypto strategies"], len(load("funding")["allocation_strategies"]))

# ---- wider-economy pages, recomputed from the raw CSVs (not from _src/data/macro_*.json)
import csv
RAWM = SRC / "raw" / "macro"
rc = lambda n: list(csv.DictReader(open(RAWM / n, encoding="utf-8")))
iso = {r["ISO3166-1-Alpha-3"] for r in rc("country_codes.csv")} | {"XKX"}
gdp = {}
for r in rc("gdp.csv"):
    if r["Country Code"] in iso and r["Value"]:
        gdp.setdefault(int(r["Year"]), {})[r["Country Name"]] = float(r["Value"])
gy = max(y for y, d in gdp.items() if len(d) >= 180)
world_gdp = next(float(r["Value"]) for r in rc("gdp.csv") if r["Country Code"] == "WLD" and int(r["Year"]) == gy)
top = sorted(gdp[gy].items(), key=lambda kv: -kv[1])
eco = text("economies.html")
chk("economies: year", re.search(r"World Bank, GDP \(current US\$\), NY\.GDP\.MKTP\.CD, (\d{4})", eco).group(1), gy)
chk("economies: economies with a figure", re.search(r"15 largest of (\d+) economies", eco).group(1), len(gdp[gy]))
chk("economies: top-3 share %", re.search(r"The three largest hold (\d+)% of world GDP", eco).group(1), f"{sum(v for _, v in top[:3]) / world_gdp * 100:.0f}")
chk("economies: top-10 share %", re.search(r"the ten largest (\d+)%", eco).group(1), f"{sum(v for _, v in top[:10]) / world_gdp * 100:.0f}")
chk("economies: top ÷ second", re.search(r"is ([\d.]+) times", eco).group(1), f"{top[0][1] / top[1][1]:.1f}")

pop = {}
for r in rc("population.csv"):
    if r["Country Code"] in iso and r["Value"]:
        pop.setdefault(int(r["Year"]), {})[r["Country Name"]] = float(r["Value"])
py = max(y for y, d in pop.items() if len(d) >= 180)
ppl = text("people.html")
big = sorted(pop[py].items(), key=lambda kv: -kv[1])
chk("people: countries", re.search(r"15 largest of (\d+) countries", ppl).group(1), len(pop[py]))
chk("people: ten most populous share %", re.search(r"ten most populous countries hold (\d+)%", ppl).group(1),
    f"{sum(v for _, v in big[:10]) / next(float(r['Value']) for r in rc('population.csv') if r['Country Code'] == 'WLD' and int(r['Year']) == py) * 100:.0f}")

infl = {}
for r in rc("inflation_consumer.csv"):
    if r["Country Code"] in iso and r["Inflation"]:
        infl.setdefault(int(r["Year"]), {})[r["Country"]] = float(r["Inflation"])
iy = max(y for y, d in infl.items() if len(d) >= 180)
pr = text("prices.html")
vals = sorted(infl[iy].values())
chk("prices: countries under 5%", re.search(r"had inflation under 5% and", pr) and re.search(r"(\d+) of \d+ countries had inflation under 5%", pr).group(1), sum(1 for v in vals if v < 5))
chk("prices: 10% or more", re.search(r"and (\d+) had 10% or more", pr).group(1), sum(1 for v in vals if v >= 10))
chk("prices: median", re.search(r"median country had ([\d.]+)% inflation", pr).group(1), f"{vals[len(vals) // 2]:.1f}")
hi = max(infl[iy].items(), key=lambda kv: kv[1])
chk("prices: highest", re.search(r"(\w[\w .,'-]*?) had ([\d,]+)% inflation in " + str(iy), pr).group(2), f"{hi[1]:,.0f}")

mk = text("markets.html")
brent = {}
for r in rc("brent_daily.csv"):
    brent.setdefault(r["Date"][:7], []).append(float(r["Price"]))
bm = {k: sum(v) / len(v) for k, v in brent.items()}
pk = max(bm, key=lambda k: bm[k])
chk("markets: Brent peak month", re.search(r"Brent peaked at \$(\d+) in (\d{4}-\d{2})", mk).group(2), pk)
chk("markets: Brent peak $", re.search(r"Brent peaked at \$(\d+)", mk).group(1), f"{bm[pk]:.0f}")
gold = {r["Date"]: float(r["Price"]) for r in rc("gold_monthly.csv") if r["Price"]}
chk("markets: gold latest $", re.search(r"and \$([\d,]+) in \d{4}-\d{2}", mk).group(1), f"{gold[max(gold)]:,.0f}")

jb = text("jobs.html")
emp = [r for r in rc("us_employment.csv") if r["unemployed_percent"]]
mx = max(emp, key=lambda r: float(r["unemployed_percent"]))
chk("jobs: unemployment peak", re.search(r"Unemployment reached ([\d.]+)% in (\d{4})", jb).groups(), (mx["unemployed_percent"], mx["year"]))

cl = text("climate.html")
fo = {int(r["Year"]): float(r["Total"]) for r in rc("co2_fossil_global.csv") if r["Total"]}
ly = max(fo)
chk("climate: fossil CO2 bn t, latest", re.search(r"([\d.]+) billion tonnes in " + str(ly), cl).group(1), f"{fo[ly] * 3.664 / 1000:.1f}")
chk("climate: times 1950", re.search(r"is ([\d.]+) times its 1950 level", cl).group(1), f"{fo[ly] / fo[1950]:.1f}")

for r in rows:
    print(" | ".join(r))
bad = sum(r[3] != "match" for r in rows)
print(f"{len(rows)} checks, {bad} mismatches")
raise SystemExit(1 if bad else 0)
