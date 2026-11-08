"""Turn the raw macro files (_src/raw/macro/, from fetch_macro.py) into _src/data/macro_*.json.

    python _src/clean_macro.py

Rules, as in clean.py: never fill a gap, never guess a unit, list every repair. Each repair is
appended to _src/data/macro_repairs.json and printed on the About the data page.
"""
import csv
import json
import pathlib
from collections import defaultdict

SRC = pathlib.Path(__file__).resolve().parent
RAW = SRC / "raw" / "macro"
OUT = SRC / "data"
repairs = []


def rd(name):
    with open(RAW / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def rep(dataset, text):
    repairs.append({"dataset": dataset, "repair": text})


def dump(name, obj):
    (OUT / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def pack(points, digits):
    """{year: value} to {"y0": first year, "v": [...]}, with null for a missing year inside the range."""
    pts = {y: v for y, v in points.items() if v is not None}
    if not pts:
        return None
    y0, y1 = min(pts), max(pts)
    return {"y0": y0, "v": [None if y not in pts else (round(pts[y], digits) if digits else int(round(pts[y]))) for y in range(y0, y1 + 1)]}


# ---------- countries

def countries():
    codes = {r["ISO3166-1-Alpha-3"]: r for r in rd("country_codes.csv") if r["ISO3166-1-Alpha-3"]}
    names = {}
    series = defaultdict(lambda: defaultdict(dict))
    agg = defaultdict(lambda: defaultdict(dict))
    spec = [("gdp.csv", "Country Name", "Value", "gdp", 0), ("population.csv", "Country Name", "Value", "pop", 0),
            ("inflation_consumer.csv", "Country", "Inflation", "infl", 2), ("cpi_index.csv", "Country", "CPI", "cpi", 2)]
    for fn, nk, vk, key, _ in spec:
        for r in rd(fn):
            code, v = r["Country Code"], num(r[vk])
            names.setdefault(code, r[nk])
            target = series if (code in codes or code == "XKX") else agg
            target[code][key][int(r["Year"])] = v
    for r in rd("owid_co2_trimmed.csv"):
        code = r["iso_code"]
        if code in codes or code == "XKX":
            y = int(r["year"])
            series[code]["co2"][y] = num(r["co2"])
            series[code]["co2pc"][y] = num(r["co2_per_capita"])
            series[code]["energypc"][y] = num(r["energy_per_capita"])
    digits = {"gdp": 0, "pop": 0, "infl": 2, "cpi": 2, "co2": 1, "co2pc": 2, "energypc": 0}
    out = []
    for code in sorted(series, key=lambda c: names.get(c, c)):
        info = codes.get(code, {})
        packed = {k: pack(v, digits[k]) for k, v in series[code].items()}
        packed = {k: v for k, v in packed.items() if v}
        if "gdp" not in packed and "pop" not in packed:
            continue
        out.append({"code": code, "name": names.get(code, code), "region": info.get("Region Name") or "",
                    "sub": info.get("Sub-region Name") or "", "s": packed})
    aggs = []
    for code in sorted(agg, key=lambda c: names.get(c, c)):
        packed = {k: pack(v, digits[k]) for k, v in agg[code].items()}
        aggs.append({"code": code, "name": names.get(code, code), "s": {k: v for k, v in packed.items() if v}})
    rep("countries", f"World Bank rows that are not countries (regions, income groups, the world: {len(aggs)} codes) are kept apart from the {len(out)} countries, so no ranking mixes them; the test is whether the code is an ISO 3166-1 alpha-3 country code (Kosovo, XKX, is added by hand: it has no ISO code but is a World Bank economy)")
    rep("countries", "the country's region comes from the ISO country-codes file (UN M49 regions), not from the World Bank, so 'Europe' here follows the UN, not the World Bank's 'Europe & Central Asia'")
    rep("countries", "GDP is in current US dollars (not adjusted for prices), so growth in it mixes real growth, inflation and exchange-rate moves; the pages say so wherever it is compared across years")
    rep("countries", "the World Bank CPI index series is kept in the data file but not charted: its base year is not stated and some countries' values are not comparable between 2000 and 2023 (Albania's index rises 95 times, others fall to zero or below), which looks like changes of base or currency rather than prices")
    rep("countries", "CO2 per capita, CO2 and energy per capita are taken from Our World in Data's CO2 file (columns trimmed, years from 1950 only); its population and GDP columns are not used")
    dump("macro_countries", {"countries": out, "aggregates": aggs})
    return out


# ---------- markets and money

def monthly_mean(rows, dk, vk):
    acc = defaultdict(list)
    for r in rows:
        v = num(r[vk])
        if v is not None and r[dk]:
            acc[r[dk][:7]].append(v)
    return {k: sum(v) / len(v) for k, v in sorted(acc.items())}


def markets():
    out = {}
    out["brent"] = {k: round(v, 2) for k, v in monthly_mean(rd("brent_daily.csv"), "Date", "Price").items()}
    out["wti"] = {k: round(v, 2) for k, v in monthly_mean(rd("wti_daily.csv"), "Date", "Price").items()}
    out["gold"] = {r["Date"]: num(r["Price"]) for r in rd("gold_monthly.csv") if num(r["Price"]) is not None}
    out["natgas"] = {r["Month"]: num(r["Price"]) for r in rd("natgas_monthly.csv") if num(r["Price"]) is not None}
    sp, pe, dropped = {}, {}, 0
    for r in rd("sp500_monthly.csv"):
        k = r["Date"][:7]
        sp[k] = num(r["SP500"])
        rp = num(r["Real Price"])
        pe10 = num(r["PE10"])
        if pe10 and pe10 > 0:
            pe[k] = pe10
        else:
            dropped += 1
    out["sp500"] = sp
    out["pe10"] = pe
    rep("markets", f"S&P 500: the file's PE10 column holds 0.0 for 1871 to 1880 (a ten-year average does not exist yet) and again from October 2023, where Shiller's earnings data ends; the dividend, earnings and real-price columns also go to 0.0 from mid-2023. The {dropped} months with PE10 = 0 are treated as missing, not as a ratio of zero, and only the index level is used to the latest month")
    us_cpi = {}
    for r in rd("us_cpi.csv"):
        v = num(r["Index"])
        if v is not None:
            us_cpi[r["Date"][:7]] = v
    out["us_cpi"] = us_cpi
    emp = []
    for r in rd("us_employment.csv"):
        emp.append({"year": int(r["year"]), "pop": num(r["population"]), "labor": num(r["labor_force"]), "employed": num(r["employed_total"]),
                    "unemployed": num(r["unemployed"]), "unemployed_pct": num(r["unemployed_percent"]), "participation": num(r["population_percent"])})
    out["us_employment"] = emp
    rep("markets", "US employment: figures are thousands of people aged 16 and over (the BLS table 'cpsaat01'); the file's footnotes column marks years in which the survey was redesigned, and those breaks are not smoothed")
    ppm = {}
    for r in rd("co2_ppm_monthly.csv"):
        v = num(r["Average"])
        if v is not None and v > 0:
            ppm[r["Date"]] = v
    out["co2_ppm"] = ppm
    rep("climate", "CO2 parts per million (Mauna Loa): the inherited file's rows carry more fields than its header names, so only the third column, 'Average', is used (March 1958 = 315.71, matching NOAA); the interpolated and trend columns are dropped because they cannot be told apart in the file")
    out["co2_fossil"] = {int(r["Year"]): {"total": num(r["Total"]), "gas": num(r["Gas Fuel"]), "liquid": num(r["Liquid Fuel"]),
                                          "solid": num(r["Solid Fuel"]), "cement": num(r["Cement"]), "per_capita": num(r["Per Capita"])} for r in rd("co2_fossil_global.csv")}
    rep("climate", "Global fossil CO2: million tonnes of carbon (not of CO2; multiply by 3.664 for CO2), as in the Global Carbon Project's file; per-capita is tonnes of carbon per person")
    # exchange rates: annual means of daily rates, local currency units per US dollar
    acc = defaultdict(lambda: defaultdict(list))
    for r in rd("fx_daily.csv"):
        v = num(r["Exchange rate"])
        if v:
            acc[r["Country"]][int(r["Date"][:4])].append(v)
    fx = {c: {y: round(sum(v) / len(v), 4) for y, v in sorted(ys.items())} for c, ys in sorted(acc.items())}
    out["fx"] = fx
    rep("money", "Exchange rates: the file's readme says Australia, the euro, Ireland, New Zealand and the United Kingdom are quoted as US dollars per local unit, but the values (Australia 0.90 in 1971, 1.42 in 2026) show they are local units per US dollar like all the rest; all 22 are used as local units per US dollar")
    rep("money", "Venezuela's series is kept in the data but left out of the currency charts: the bolívar was redenominated (2008, 2018, 2021) and the file does not say which unit each year is in")
    rep("money", "Exchange rates are annual averages of the daily rates, where the Federal Reserve published one that day; a year with few days (the first and the current year) is a part-year average and is marked so on the page")
    counts = {c: {y: len(v) for y, v in ys.items()} for c, ys in acc.items()}
    out["fx_days"] = counts
    dump("macro_markets", out)


def main():
    OUT.mkdir(exist_ok=True)
    cs = countries()
    markets()
    dump("macro_repairs", repairs)
    print(f"{len(cs)} countries; {len(repairs)} repairs written")


if __name__ == "__main__":
    main()
