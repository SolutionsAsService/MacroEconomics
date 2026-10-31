# Critic: Evidence (Cairo), R0

**Anchor used:** Cairo, How Charts Lie and The Truthful Art: a chart is an argument; show uncertainty, state the denominator, source and units, never chart what you cannot vouch for. (Raw files now live in _src/raw.)

**Measured (Python, regex salvage and json.load):**
- Inflation: invalid JSON (line 1019). 18 rows of 190+ economies, A-to-B alphabetical. Non-null years per row: 14 (UAE) to 63; only 4 rows full (Australia, Austria, Belgium, Burkina Faso). Maxima 4,145% (Angola 1996), 3,374% (Armenia 1995), 1,662% (Azerbaijan 1994): a linear axis is crushed by three points. Two regional aggregates mixed in with countries.
- Oil reserves: invalid JSON, two schemas. 24+17 countries. EIA/OPEC/BP spread: Venezuela 1%, Iran 34%, US 68%, Canada 3,795% (OPEC 4,421 vs EIA 170,000: definition gap, oil sands). "Other_Reserves" (980,000 for Oman) and Production_2021 (15) have no units. Stated "years in reserve" is a static ratio: US 6.3 stated vs 10.7 computed; Venezuela 1,000 vs 1,190.
- Production: valid, one month (Sep 2023), 10 countries, 57.2M bbl/d total; a snapshot, not a trend, and not comparable with 2021 reserve-file production.
- Ports (CPPI 2021): valid, 300 rows, 287 unique names (13 repeats); 12 null statistical ranks; 13 duplicate admin ranks, 35 duplicate stat ranks. Of 288 ports ranked by both methods, rank correlation 0.985, median shift 3, but 56 move 10 or more places (Vung Tau 37 vs 271). Last row Los Angeles ranks 370 and 369 of 300 rows, so the ranks and the row count disagree.

**Verdict: OBJECT.**

**Requested change:**
1. Chart only: oil production (add the date and "top 10 only" in the title); the three-source reserves spread as a range, not one bar; CPPI as a slope or dumbbell chart showing both methods, with the 56 large movers marked; inflation only for the 4 complete rows plus labelled gaps.
2. Inflation: use a log or symlog axis, or clip with a visible "off-scale" marker; never connect across null years; label "18 of ~190 economies, A-B only".
3. The landing chart must carry: title as claim, units (barrels per day, million barrels), source and date, n, and what is excluded. Do not use a lead chart that implies a global picture from a sample.
4. Do not chart: PricePrediction (no data, linear extrapolation), "Other_Reserves"/"Production_2021", the CFA "benefit" line as fact, or any funding or strategy "returns".
5. Repair data into documented files with a page stating each fix.

**Statement of meaning:** Honest charts here show a snapshot with named gaps and disagreeing sources, not a world.
