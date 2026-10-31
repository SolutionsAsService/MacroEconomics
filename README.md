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
| About the data | `repairs.json` | every source, gap and repair |

## The data

The inherited files are kept unchanged in `_src/raw/`. Three of them could not be parsed: `OilReserves.json` and `WBInflationDataHistoric.json` were malformed, and `HistoricInflationDataSchema.json` is empty. `_src/clean.py` repairs them into `_src/data/*.json` and writes every repair to `_src/data/repairs.json`. It never fills a gap: missing years stay missing, unitless values are dropped and listed. The inherited formula for quadratic funding was wrong and is corrected (Buterin, Hitzig and Weyl, 2019). `PricePrediction.py` fits a straight line to a CSV that is not in the repo; it is kept in `_src/raw/` and not published.

The inflation file holds only the World Bank's rows from Aruba to Bangladesh, and the ports file 287 of the 370 ranked ports. Completing them needs a fresh download from the World Bank.

The repository as it stood before this redesign is on the branch `legacy-2026-10-30`.

## Building

```bash
python _src/clean.py
python _src/build.py
python _src/check_figures.py
```

`build.py` (Python 3, standard library only) writes the nine pages, sitemap, `robots.txt`, `llms.txt` and favicon into `site/`, draws every chart as inline SVG from the data, and fails on a broken link, a missing `#anchor`, a duplicate id or invalid JSON-LD. `_src/check_figures.py` recomputes every figure the pages print and compares it with the built HTML. `_src/og_card.py` (needs Pillow) draws the social card.

## Deploying

Static. Root `vercel.json` (`outputDirectory: site`) and `site/vercel.json` (`outputDirectory: .`) deploy it with the Root Directory blank or set to `site`, on Vercel or Shadw. There are no framework markers at the root.

`DESIGN-LOG.md` and `panel/` record the design run that produced the site.

## License

MIT. See [LICENSE](LICENSE) and [CONTRIBUTING.md](CONTRIBUTING.md).
