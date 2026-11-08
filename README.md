# MacroEconomics

Oil, prices, ports and money: small economics datasets, repaired and charted, with their sources and gaps shown.

Live at https://macroeconomics.shadw.app/ once deployed.

## What is here

| Page | Data | Shows |
|---|---|---|
| Home | oil reserves | the ten largest proved reserves by the US EIA, with OPEC's and BP's counts as ticks; they disagree on Canada by a factor of 38 |
| Oil | `oil.json` | reserves for 41 countries by three counts, and production for the ten largest producers (September 2023) |
| Inflation | `inflation.json` | what 100 units of money from 1960 buys each year in four countries, and yearly inflation for 18 World Bank rows |
| Ports | `ports.json` | the Container Port Performance Index 2021, two ranking methods side by side, 287 ports |
| CFA franc | `cfa_franc.json` | the euro peg, the two zones and 14 countries, history, criticisms and the case made for it, with a French summary |
| Funding public goods | `funding.json` | a quadratic funding calculator, the rule (corrected), and nine other funding mechanisms |
| Crypto strategies | `funding.json` | eight strategies, each beside how it loses; not advice |
| About the data | `repairs.json`, `macro_repairs.json` | every source, gap and repair |
| Explorer | `macro_countries.json` | any of 216 countries, or two side by side: GDP, GDP per person, population, inflation, CO2 per person |
| Economies | `macro_countries.json` | the 15 largest economies, shares of world GDP since 1960, rank movers, GDP per person |
| People | `macro_countries.json` | the most populous countries, regional shares since 1960, shrinking and fast-growing countries |
| Prices | `macro_countries.json`, `macro_markets.json` | inflation in every country, worst years on record, world inflation, US prices since 1913 |
| Markets | `macro_markets.json` | Brent and WTI since 1987, gold since 1833, the S&P 500 and P/E10 since 1871, Henry Hub gas, each in today's dollars too |
| Money | `macro_markets.json` | 22 currencies against the US dollar since 2000 |
| Jobs | `macro_markets.json` | US unemployment and participation since 1941 |
| Climate | `macro_countries.json`, `macro_markets.json` | global fossil CO2 since 1750, Mauna Loa CO2, emissions against income, largest and per-person emitters |

## The data

The inherited files are kept unchanged in `_src/raw/`. Three of them could not be parsed: `OilReserves.json` and `WBInflationDataHistoric.json` were malformed, and `HistoricInflationDataSchema.json` is empty. `_src/clean.py` repairs them into `_src/data/*.json` and writes every repair to `_src/data/repairs.json`. It never fills a gap: missing years stay missing, unitless values are dropped and listed. The inherited formula for quadratic funding was wrong and is corrected (Buterin, Hitzig and Weyl, 2019). `PricePrediction.py` fits a straight line to a CSV that is not in the repo; it is kept in `_src/raw/` and not published.

The inflation file holds only the World Bank's rows from Aruba to Bangladesh, and the ports file 287 of the 370 ranked ports. Completing them needs a fresh download from the World Bank.

The repository as it stood before this redesign is on the branch `legacy-2026-10-30`.

The wider-economy pages use open datasets from the World Bank, US EIA, BLS, NOAA, the Federal Reserve, Robert Shiller, the Global Carbon Project and Our World in Data (via the Open Knowledge Foundation's `datasets` repositories). `_src/fetch_macro.py` downloads them unchanged into `_src/raw/macro/` (the 7.6 MB daily exchange-rate file is not committed; rerun the fetch to get it), and `_src/clean_macro.py` writes `_src/data/macro_countries.json` and `macro_markets.json`. It never fills a gap; every repair is on the About the data page. Each source keeps its own licence (CC-BY-4.0 or PDDL).

## Building

```bash
python _src/fetch_macro.py   # only to refresh the macro data; needs network
python _src/clean.py
python _src/clean_macro.py
python _src/build.py
python _src/check_figures.py
```

`build.py` (Python 3, standard library only) writes the nine pages, sitemap, `robots.txt`, `llms.txt` and favicon into `site/`, draws every chart as inline SVG from the data, and fails on a broken link, a missing `#anchor`, a duplicate id or invalid JSON-LD. `_src/check_figures.py` recomputes every figure the pages print and compares it with the built HTML. `_src/og_card.py` (needs Pillow) draws the social card.

## Deploying

Static. Root `vercel.json` (`outputDirectory: site`) and `site/vercel.json` (`outputDirectory: .`) deploy it with the Root Directory blank or set to `site`, on Vercel or Shadw. There are no framework markers at the root.

`DESIGN-LOG.md` and `panel/` record the design run that produced the site.

## License

MIT. See [LICENSE](LICENSE) and [CONTRIBUTING.md](CONTRIBUTING.md).
