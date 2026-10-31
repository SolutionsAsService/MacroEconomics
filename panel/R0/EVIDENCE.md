# R0 premortem evidence: MacroEconomics as inherited (commit d4339fb, saved on branch legacy-2026-10-30)

The repo holds 12 files, 176 KB, no README and no page. CONTRIBUTING.md says "See the README for how to run or use what is here"; there is no README. The raw files are in the repo root next to this folder: D:\New folder\PrometheusIntelligence\MacroEconomics\

Measured (Python json.load, file reads):
- WBInflationDataHistoric.json (38 KB): INVALID JSON (missing commas, `// ... continue with null for years without data ...` comments). 18 rows only: World Bank rows alphabetically from Aruba to Bangladesh, mixing countries with regional aggregates ("Arab World", "Africa Eastern and Southern"). Some rows skip years (Benin jumps 1996 to 2008). Values are annual CPI inflation, 1960 to 2022.
- HistoricInflationDataSchema.json: EMPTY (0 bytes).
- OilReserves.json (8.8 KB): INVALID JSON. Two structures in one file: "Countries" (24 countries, string values like "304,000 million barrels" and, later, "(bbl)" fields that are also millions of barrels) and "OilData" (17 smaller countries, numbers in millions of barrels, plus "Other_Reserves" values like 980000 and "Production_2021" values like 15 whose units cannot be determined). Reserves given three ways per country: US EIA, OPEC, BP; they disagree (Venezuela 304,000 / 302,809 / 300,900 million barrels).
- OilProductionData1.json: valid; one month (September 2023), production in barrels per day for the top producers, with an OPEC flag.
- CPPI2021GlobalRankings.json (87 KB): valid; the World Bank / S&P Container Port Performance Index 2021, each port ranked by two methods (administrative points, statistical index).
- CFAFranc.json: valid; history, peg (€1 = F.CFA 655.957), member countries in two zones, reform. Under "Criticism" it lists "Facilitates exports and imports between France and member countries", which reads as a claimed benefit.
- FundingMechanisms.yaml, QuadraticFunding.yaml: descriptions of ways to fund projects. QuadraticFunding.yaml states the formula as "Direct contributions + (Square of the sum of the square roots of individual contributions) * Matching pool", which is wrong: the standard rule gives each project (sum of sqrt(c))^2 minus the sum of c, scaled so all matches add up to the pool.
- AllocationStrategies.yaml: crypto investment strategies (dollar-cost averaging, buy and hold, hedging...) with benefits such as "Steadier returns over time", no sources, no risk disclaimer.
- PricePrediction.py: fits a straight line to a CSV that is not in the repo (historical_prices.csv) and plots 30 days ahead as "Predicted Future Prices".
- No sources are cited anywhere except dataset names (EIA, OPEC, BP, World Bank) inside field names.

Planned redesign: a multi-page static site (a landing page with one bold graphic complete on the first screen, then a page per topic: oil, inflation, ports, the CFA franc, funding public goods with a quadratic funding calculator, allocation strategies, about the data), built from repaired data files.
