# Critic: Evidence (Cairo), R3

**Anchor used:** Cairo, The Truthful Art / How Charts Lie: a headline figure must state its procedure and not imply more precision or like-for-like comparison than the data supports.

**Measured (Python on _src/data/oil.json; site/*.html):**
- 38x: 170000 / 4421 = 38.45, shown as 38x; build note "EIA count / OPEC count" states the procedure. Match. (oil.html's 3,795% is BP 172,200 / 4,421 - 1 = 37.95x, a different, labelled base; consistent.)
- "Four others": top ten by EIA, max/min-1 >= 0.25: Iran .34, Russia .28, Libya .63, US .68 plus Canada = 5 of ten; 4 others. Match.
- Oil "highest more than 25% above lowest for 15": 34 countries with 2+ counts, 15 pass. Match.
- Production chart OPEC members: Saudi Arabia, Iraq, UAE, Iran, Kuwait in file = label; share 39.1% = "39%". Match. Reserves chart: 7 named members = data. Match.
- Spot-check: Canada 170 bn / 4.4 bn, 41 countries, 5 of ten: match.

**Verdict: PASS**

**Requested change:** none required. Optional: the 38x is a ratio of counts that are defined differently (OPEC appears to omit oil sands, said only on oil.html). Add "definitions differ" under the numeral so it reads as a definitional gap, not a measurement error.

**Statement of meaning:** Every changed figure recomputes exactly, and 38x is honest because its procedure is printed, though the reason for the gap sits one page away.
