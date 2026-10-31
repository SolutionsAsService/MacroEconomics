# Critic: Evidence (Cairo), R1

**Anchor used:** Cairo, How Charts Lie / The Truthful Art: a chart is an argument; state units, source, date, n, exclusions; a headline must not claim more than the data.

**Measured (Python on _src/data, raw in _src/raw):**
- Landing/oil: Canada EIA 170,000 vs OPEC 4,421 -> spread 3,795%: match. Years of reserves (EIA/2021 output): Venezuela 1,190, Saudi 65, US 10.7 (prints 11), Brazil 11.7 (12): match. OPEC share of the ten-country Sept 2023 output 22.38M/57.23M = 39.1%: match.
- Inflation: 100 from 1960 by 2022: Australia 5.8, Austria 13.1, Belgium 11.7, Burkina Faso 7.6: match (compounded through end-2022). Peaks Angola 4,145% (1996), Armenia 3,374% (1995), Azerbaijan 1,662% (1994), Albania 226% (1993): match; no other row exceeds 50%. Repaired values equal raw to 3 decimals (710 cells, 0 real differences).
- Ports: 55 move 10+ places: match; Vung Tau 37 vs 271: match. But 12 ports lack a statistical rank, so 55 is of 275, not "all 287": denominator misstated. 287 unique, 13 exact duplicates removed, last rank 370: match raw.
- CFA: 193.1 million (2021), 14 countries: match.
- Calculator: $0/$750/$250; "50, 50" gives $77/$692/$231: match.
- repairs.json vs raw: Mexico truncation, "(bbl)" fields, Other_Reserves/Production_2021 (Oman 980000/15), two comment lines, 13 duplicates, 12 null ranks, CFA criticism line, QF formula: all confirmed in raw.
- Chart labelling: units, source, n present. Date missing on reserves (admitted). Ports "top 30" chart has 29 ports (admin rank 24 absent); its biggest mover, Vung Tau, is not drawn.

**Verdict: OBJECT** (two claims, small fixes).

**Requested change:**
1. Landing hero and oil take say the counts "agree almost everywhere / except where they define oil differently". Measured: 15 of 34 countries with two or more counts differ by over 25% (Libya 63%, US 68%, Iran 34%, Russia 28% among the top ten). Reword: "Canada is the extreme; many others differ by a quarter or more."
2. Ports: "55 of the 275 ports with both ranks"; caption "29 ports shown (rank 24 not in file)"; say Vung Tau lies outside the chart.

**Statement of meaning:** The numbers are right; two sentences still say the sources agree more, and cover more ports, than the data shows.
