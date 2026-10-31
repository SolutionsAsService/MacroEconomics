"""Repair the inherited raw files in _src/raw/ into clean JSON in _src/data/.

The raw files are kept exactly as inherited. Three of them do not parse (OilReserves.json,
WBInflationDataHistoric.json, and the empty HistoricInflationDataSchema.json), so this script
reads them leniently and records every repair it makes in _src/data/repairs.json. It never
fills a gap: a missing year stays missing, a value whose unit cannot be determined is dropped
and listed.

    python _src/clean.py
"""
import json
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parent
RAW, OUT = SRC / "raw", SRC / "data"
repairs = []


def note(dataset, what):
    repairs.append({"dataset": dataset, "repair": what})


def num(s):
    return float(str(s).replace(",", "")) if s not in (None, "") else None


# ---------- inflation (World Bank, annual CPI inflation %)

def inflation():
    text = (RAW / "WBInflationDataHistoric.json").read_text(encoding="utf-8")
    lines = [l for l in text.splitlines() if not l.strip().startswith("//")]
    dropped = text.count("//")
    text = "\n".join(lines)
    text, n = re.subn(r"\}\s*\n(\s*)\{", r"},\n\1{", text)
    data = json.loads(text)["countries"]
    note("inflation", f"removed {dropped} comment line(s) of the form '// ... continue with null for years without data ...' and added {n} missing comma(s) between rows")
    aggregates = {"Africa Eastern and Southern", "Africa Western and Central", "Arab World"}
    out = []
    for row in data:
        years = {int(y): v for y, v in row["inflationRates"].items() if v is not None}
        present = sorted(years)
        gaps = [y for y in range(present[0], present[-1] + 1) if y not in years] if present else []
        out.append({"name": row["name"], "kind": "aggregate" if row["name"] in aggregates else "country",
                    "values": {str(y): round(years[y], 3) for y in present},
                    "first": present[0] if present else None, "last": present[-1] if present else None,
                    "missing_years_inside_range": gaps})
    note("inflation", f"{len(out)} rows: the World Bank file's rows from Aruba to Bangladesh only; the rest of the alphabet is not in the inherited file")
    return {"source": "World Bank, Inflation, consumer prices (annual %), indicator FP.CPI.TOTL.ZG, as inherited (partial)",
            "unit": "percent per year", "rows": out}


# ---------- oil reserves and production

def oil():
    text = (RAW / "OilReserves.json").read_text(encoding="utf-8")
    head, _, tail = text.partition('"OilData"')
    rows = {}
    for name, body in re.findall(r'"([A-Z][A-Za-z ]+)":\s*\{([^{}]*)\}', head):
        f = dict(re.findall(r'"([^"]+)":\s*"([^"]*)"', body))
        get = lambda pre: next((v for k, v in f.items() if k.startswith(pre)), None)
        rows[name] = {"country": name,
                      "eia": num((get("US EIA Reserves") or "").replace(" million barrels", "")),
                      "opec": num((get("OPEC Reserves") or "").replace(" million barrels", "")),
                      "bp": num((get("BP Reserves") or "").replace(" million barrels", "")),
                      "production_2021_bpd": num(get("Oil Production 2021")), "block": "Countries"}
    m = re.search(r'"Mexico":\s*\{\s*"US EIA Reserves \(bbl\)":\s*"([\d,]+)"', head)
    if m and "Mexico" not in rows:
        rows["Mexico"] = {"country": "Mexico", "eia": num(m.group(1)), "opec": None, "bp": None,
                          "production_2021_bpd": None, "block": "Countries"}
        note("oil", "Mexico's entry is cut off mid-line in the inherited file; only its US EIA figure (5,800 million barrels) survives")
    note("oil", "fields labelled '(bbl)' in the second half of the 'Countries' block hold the same values as 'million barrels' (Nigeria 37,000 = 37 billion barrels) and are read as millions of barrels")
    for body in re.findall(r"\{([^{}]*\"Country\"[^{}]*)\}", tail):
        f = {k: v for k, v in re.findall(r'"([A-Za-z_0-9]+)":\s*("?[^",\n]*"?)', body)}
        name = f["Country"].strip('"').replace(" (OPEC)", "")
        rows[name] = {"country": name, "eia": num(f.get("US_EIA_Reserves")), "opec": num(f.get("OPEC_Reserves")),
                      "bp": num(f.get("BP_Reserves")), "production_2021_bpd": None, "block": "OilData"}
    note("oil", "the 'OilData' block's 'Other_Reserves' (e.g. Oman 980000) and 'Production_2021' (e.g. Oman 15) have no stated unit and do not match any of the other figures; both are dropped")
    note("oil", "'Years of Production in Reserve' is not copied; it is recomputed from US EIA reserves and 2021 production where both exist")
    opec = {"Algeria", "Congo", "Equatorial Guinea", "Gabon", "Iran", "Iraq", "Kuwait", "Libya", "Nigeria",
            "Saudi Arabia", "United Arab Emirates", "Venezuela"}
    note("oil", "OPEC membership added (twelve members as of 2024; Angola left in January 2024): the inherited reserves file marked only Gabon and Congo, and the production file flags members only for its ten countries")
    for r in rows.values():
        r["opec_member"] = r["country"] in opec
        r["years_of_reserves"] = round(r["eia"] * 1e6 / (r["production_2021_bpd"] * 365), 1) if r["eia"] and r["production_2021_bpd"] else None
    prod = json.loads((RAW / "OilProductionData1.json").read_text(encoding="utf-8"))["oil_production_data"]
    production = [{"country": p["country"], "bpd": p["production_bbl_per_day"], "opec": p["is_OPEC_member"]} for p in prod]
    reserves = sorted(rows.values(), key=lambda r: -(r["eia"] or 0))
    return {"reserves": {"unit": "million barrels of proved reserves", "sources": "US EIA, OPEC and BP figures as inherited; the year of each figure is not stated in the inherited file", "rows": reserves},
            "production": {"unit": "barrels per day", "month": "September 2023", "rows": production}}


# ---------- ports (Container Port Performance Index 2021)

def ports():
    rows = json.loads((RAW / "CPPI2021GlobalRankings.json").read_text(encoding="utf-8"))["cppi2021GlobalRanking"]
    out = [{"port": r["portName"], "country": r["country"], "admin_rank": r["administrativeApproach"]["rank"],
            "admin_points": r["administrativeApproach"]["totalPoints"], "stat_rank": r["statisticalApproach"]["rank"],
            "stat_index": r["statisticalApproach"]["indexValue"]} for r in rows]
    seen, unique = set(), []
    for r in out:
        key = json.dumps(r, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    top = max(r["admin_rank"] for r in unique)
    note("ports", f"removed {len(out) - len(unique)} exact duplicate rows; {len(unique)} ports remain, of an index that ranks {top}: the inherited file is partial")
    missing = sum(r["stat_rank"] is None for r in unique)
    if missing:
        note("ports", f"{missing} ports have no statistical-approach rank in the inherited file; they are shown with the administrative rank only")
    out = unique
    return {"source": "World Bank and S&P Global Market Intelligence, Container Port Performance Index 2021, as inherited",
            "note": "Two ranking methods: an administrative approach (points) and a statistical approach (index). They usually agree near the top and diverge lower down.",
            "rows": out}


# ---------- small YAML reader for the three inherited YAML files (maps of maps and string lists)

def yaml_file(name):
    lines = (RAW / name).read_text(encoding="utf-8").splitlines()
    out, path = {}, []
    for i, raw in enumerate(lines):
        if not raw.strip():
            continue
        ind = len(raw) - len(raw.lstrip())
        line = raw.strip()
        while path and path[-1][0] >= ind and not line.startswith("- "):
            path.pop()
        node = out
        for _, k in path[:-1]:
            node = node[k]
        if line.startswith("- "):
            parent = out
            for _, k in path[:-1]:
                parent = parent[k]
            key = path[-1][1]
            if not isinstance(parent[key], list):
                parent[key] = []
            parent[key].append(line[2:].strip().strip('"'))
            continue
        key, _, val = line.partition(":")
        val = val.strip()
        target = out
        for _, k in path:
            target = target[k]
        target[key] = val.strip('"') if val else {}
        if not val:
            path.append((ind, key))
    return out


def funding():
    mech = yaml_file("FundingMechanisms.yaml")["FundingMechanisms"]
    qf = yaml_file("QuadraticFunding.yaml")["QuadraticFunding"]
    strat = yaml_file("AllocationStrategies.yaml")["CryptoAllocationStrategies"]
    wrong = qf["KeyComponents"]["ContributionFormula"]
    qf["KeyComponents"]["ContributionFormula"] = (
        "A project's match is proportional to (the sum of the square roots of its contributions) squared, minus the sum of its contributions; "
        "every project's match is then scaled by the same factor so that all matches add up to the matching pool. "
        "The project receives its direct contributions plus its match.")
    note("funding", f"QuadraticFunding.yaml stated the formula as '{wrong}', which multiplies by the whole matching pool; replaced with the standard rule (Buterin, Hitzig and Weyl, 'A Flexible Design for Funding Public Goods', 2019)")
    return {"mechanisms": mech, "quadratic_funding": qf, "allocation_strategies": strat}


def cfa():
    d = json.loads((RAW / "CFAFranc.json").read_text(encoding="utf-8"))["CFAFranc"]
    crit = d["Overview"]["Criticism"]
    moved = [c for c in crit if c.startswith("Facilitates")]
    d["Overview"]["Criticism"] = [c for c in crit if not c.startswith("Facilitates")]
    d["Overview"]["ClaimedBenefits"] = moved
    if moved:
        note("cfa_franc", "moved 'Facilitates exports and imports between France and member countries' from Criticism to a new ClaimedBenefits list: it states an advantage, not a criticism")
    return d


def main():
    OUT.mkdir(exist_ok=True)
    data = {"inflation": inflation(), "oil": oil(), "ports": ports(), "cfa_franc": cfa(), "funding": funding()}
    if not (RAW / "HistoricInflationDataSchema.json").read_text(encoding="utf-8").strip():
        note("inflation", "HistoricInflationDataSchema.json is empty in the inherited repo; nothing to carry over")
    note("price_prediction", "PricePrediction.py fits a straight line through prices from historical_prices.csv, a file not in the repo; it is kept in _src/raw/ and not published as a forecast")
    for k, v in data.items():
        (OUT / f"{k}.json").write_text(json.dumps(v, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "repairs.json").write_text(json.dumps(repairs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(data)} datasets and {len(repairs)} repairs to _src/data/")


if __name__ == "__main__":
    main()
