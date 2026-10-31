# Critic: Evidence (Cairo), R2

**Anchor used:** Cairo, The Truthful Art / How Charts Lie: a headline may not claim more than the data; state n, exclusions, and the base of any percentage.

**Measured (Python on _src/data):**
- Landing "4 others differ by a quarter or more": top ten by EIA, (max/min - 1): Iran 34, Russia 28, Libya 63, US 68, plus Canada = 5 of ten; 4 others. Printed 4, recomputed 4: match.
- Oil "34 countries with two or more counts, 15 differ by more than 25%": 34 and 15 recomputed: match. Base is the lowest count; against the highest, only 11 pass.
- Ports "275 ports with both ranks, 55 move 10+": 287 rows, 275 with both, 55: match. Vung Tau (Vietnam) 37 vs 271, the largest gap (234): match.
- Dumbbell: 30 ports, last admin rank 31, rank 24 absent from file: match. 
- Spot-check: Canada 170 bn vs 4.4 bn, ten-country OPEC-member list, 287 total: unchanged, match.

**Verdict: PASS**

**Requested change:** none required. Optional: the oil title's "differ by more than 25%" should say "highest more than 25% above lowest", since the other base gives 11 of 34 and 3 others on the landing.

**Statement of meaning:** Every changed sentence now equals what the data computes, and the remaining ambiguity is only the unstated base of "differ by".
