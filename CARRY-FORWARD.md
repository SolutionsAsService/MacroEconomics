# MacroEconomics carry-forward ledger

Facts this run established by measurement. Read before any later design run on this project.

| Fact | How it was measured | Move it changed |
|---|---|---|
| A seven-link nav wraps to three lines at 375 px and pushes the landing graphic below the fold: the header alone was 235 px and the chart ended at y = 970 of 812. Folding the links behind a Menu button (full list without JS) brought the header to 78 px and the chart's bottom to 681. | in-app browser, `getBoundingClientRect()` at 375 x 812 | M3 (phone layout) |
| An SVG drawn with `preserveAspectRatio="xMinYMin meet"` and a CSS height of `min(60vh, 450px)` stays complete at any width: it shrinks to fit rather than overflowing. | chart bottom 732 of 800 at 1280 x 800 | M3 |
| One more line in the hero lede costs about 30 px; at 1280 x 800 a three-line lede pushed the legend to 793 of 800. Widening the lede's measure to 84ch kept it at two lines. | `hero-take` height ÷ line height; legend bottom | M12 |
| Text inside an SVG with `role="img"` is not read by screen readers; only the `aria-label` is. A finding drawn as SVG text (the 38× numeral) must also be in the label. | 0 of the landing's aria-labels contained "38" before M13; 2 after | M13 |
| Two inline SVGs on one page that each define a pattern must use different ids (`hatch-wide`, `hatch-tall`), or the second reference resolves to the first. The build's duplicate-id check catches it. | build check; computed `fill: url("#hatch-wide")` | M10 |
| Several chart labels pinned to line ends must be stacked from the top down, each at least 20 units below the last; pushing down from the lowest value overlapped three labels. | label y positions 338, 358, 378, 398 after the fix | inflation chart |
| "Differ by more than 25%" is ambiguous: against the lowest count 15 of 34 countries pass, against the highest only 11. State the base. | Cairo R2, recomputed | M11 |
| The inherited oil file's "(bbl)" fields are millions of barrels (Nigeria 37,000 = 37 bn); its OilData block's "Other_Reserves" and "Production_2021" have no recoverable unit. | comparison with the three named counts | M1 |
| The inherited port file ranks to 370 but holds 287 unique ports (13 exact duplicate rows; rank 24 absent; 12 ports without a statistical rank). | `_src/clean.py` and Cairo R0, R1 | M1, M10 |
