"""Download the open macro datasets into _src/raw/macro/ (unchanged), for _src/clean_macro.py to read.

    python _src/fetch_macro.py

Sources are the Open Knowledge Foundation's "datasets" repositories (World Bank, US EIA, BLS, NOAA,
Federal Reserve, Robert Shiller, Global Carbon Project) and Our World in Data's CO2 file. Nothing is
edited here; large files are trimmed by column only (OWID) and listed in .gitignore if not committed.
"""
import csv
import io
import pathlib
import ssl
import urllib.request

RAW = pathlib.Path(__file__).resolve().parent / "raw" / "macro"
D = "https://raw.githubusercontent.com/datasets/{}/main/data/{}"
FILES = {
    "gdp.csv": D.format("gdp", "gdp.csv"),
    "inflation_consumer.csv": D.format("inflation", "inflation-consumer.csv"),
    "cpi_index.csv": D.format("cpi", "cpi.csv"),
    "population.csv": D.format("population", "population.csv"),
    "brent_daily.csv": D.format("oil-prices", "brent-daily.csv"),
    "wti_daily.csv": D.format("oil-prices", "wti-daily.csv"),
    "gold_monthly.csv": D.format("gold-prices", "monthly.csv"),
    "sp500_monthly.csv": D.format("s-and-p-500", "data.csv"),
    "natgas_monthly.csv": D.format("natural-gas", "monthly.csv"),
    "us_employment.csv": D.format("employment-us", "aat1.csv"),
    "us_cpi.csv": D.format("cpi-us", "cpiai.csv"),
    "co2_fossil_global.csv": D.format("co2-fossil-global", "global.csv"),
    "co2_ppm_monthly.csv": D.format("co2-ppm", "co2-mm-mlo.csv"),
    "fx_daily.csv": D.format("exchange-rates", "daily.csv"),
    "country_codes.csv": D.format("country-codes", "country-codes.csv"),
}
OWID = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
OWID_COLS = ["country", "year", "iso_code", "population", "gdp", "co2", "co2_per_capita", "co2_per_gdp",
             "cumulative_co2", "share_global_co2", "primary_energy_consumption", "energy_per_capita", "methane"]


def get(url):
    ctx = ssl.create_default_context(cafile="/root/.ccr/ca-bundle.crt") if pathlib.Path("/root/.ccr/ca-bundle.crt").exists() else None
    with urllib.request.urlopen(url, timeout=120, context=ctx) as r:
        return r.read()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        (RAW / name).write_bytes(get(url))
        print("fetched", name)
    rows = csv.DictReader(io.StringIO(get(OWID).decode("utf-8")))
    with open(RAW / "owid_co2_trimmed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(OWID_COLS)
        for r in rows:
            if int(r["year"]) >= 1950:
                w.writerow([r[c] for c in OWID_COLS])
    print("fetched owid_co2_trimmed.csv (columns trimmed, years from 1950)")


if __name__ == "__main__":
    main()
