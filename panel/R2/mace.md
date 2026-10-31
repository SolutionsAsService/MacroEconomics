# Critic: Inclusion (Mace), round R2

**Anchor used:** Principle 4, Perceptible Information (redundant presentation), with Principle 2, Flexibility in Use.

**Observed:** R1's objection is answered. OPEC bars are filled with an SVG pattern (hatch-wide, hatch-tall, hatch-prod; style fill:url(#...)), so membership no longer rests on red versus black. In hero-hatch-1280.png the hatch reads clearly against the solid near-black non-OPEC bars (Canada, Russia, US). The legend key is hatched, reads "OPEC member (red, hatched)", and sits as text beside the key. The reserves aria-labels (index and oil, wide and tall) end "OPEC members, drawn hatched: Venezuela ... Libya". Escape handling exists in site.js (aria-expanded true, set(false), btn.focus()). No regression seen. One gap: the evidence says every production chart label names OPEC members, but the production chart's aria-label (oil.html) lists values only. Its source line says "OPEC members in red", which is colour only. The table twin has an OPEC yes/no column, so the information is reachable.

**Focus visibility:** Named check: global :focus-visible, 3px solid var(--red), offset 2px, still present in site.css line 32 and not overridden. Result: pass.

**Verdict:** PASS.

**Requested change:** none required. Optional: add "OPEC members, drawn hatched: Saudi Arabia, Iraq, United Arab Emirates, Iran, Kuwait" to the production aria-label, and "hatched" to its source line.

**Statement of meaning:** The OPEC finding now reaches readers by pattern, text and table, not by colour alone.
