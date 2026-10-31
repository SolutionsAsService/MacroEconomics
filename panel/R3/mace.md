# Critic: Inclusion (Mace), round R3

**Anchor used:** Principle 4, Perceptible Information (redundant presentation), with Principle 3, Simple and Intuitive Use.

**Observed:** Fixed since R2: the production aria-label now ends "OPEC members, drawn hatched: Saudi Arabia, Iraq, United Arab Emirates, Iran, Kuwait", and its source line says "red and hatched". French text is ink at 1em (.fr-line), heading 0.85em at weight 600, with a 6px red rule; no muted italic remains. Gap: the "38×" and "EIA count ÷ OPEC count" are SVG <text> inside a role="img" chart, so a screen reader skips them. Neither the wide aria-label nor the page's visible text states the ratio ("38" occurs nowhere in accessible text). The reader gets 170 bn and 4.4 bn but must divide to reach the page's headline.

**Focus visibility:** Check: global :focus-visible, 3px solid var(--red), offset 2px (site.css line 32), not overridden. Result: pass.

**Verdict:** OBJECT (small, build-level).

**Requested change:** Add "EIA's count is 38 times OPEC's for Canada" to the wide and phone chart aria-labels (or one sentence in the lede or takeaway).

**Statement of meaning:** The 38× finding must reach every reader as words, not only as a giant red numeral.
