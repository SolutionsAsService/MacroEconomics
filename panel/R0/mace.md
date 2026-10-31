# Critic: Inclusion (Mace), round R0

**Anchor used:** Principle 4, Perceptible Information (with Principle 2, Flexibility in Use, and Principle 7, Size and Space for Approach and Use).

**Observed:** The inherited repo has no page, so nothing is perceivable by anyone. Its only "interface" is data: two JSON files that do not parse, an empty schema, and units that cannot be determined (Production_2021 "15"). Meaning is carried by field names and unlabelled numbers, so even a reader of the raw files cannot recover what a value measures. The planned redesign adds the usual exclusion risks: a bold first-screen graphic, charts, and a slider-style calculator.

**Verdict:** OBJECT.

**Requested change:**
1. Every page: lang attribute, one h1, landmarks (header, nav, main, footer), a skip link, a unique title, and semantic headings.
2. Focus: a visible 3 px focus ring with at least 3:1 contrast on all controls, never removed. Tab order follows reading order. No keyboard traps.
3. Charts: each has a text title and a one-sentence takeaway. Each has a real data table (caption, th scope) beside it, or behind a visible "Show data" toggle. Series are never told apart by colour alone: use direct labels, patterns or markers. Tooltips and hover detail must also open on focus and keypress. Chart SVG gets role="img" with an aria-label that states the finding. Motion respects prefers-reduced-motion.
4. Calculator: native labelled inputs with units and ranges stated in text, number fields beside any slider, and no drag-only input. Output sits in an aria-live="polite" region, with the formula and a worked example in plain words. Errors are announced and name the fix. It works with no pointer, and a no-script fallback table is provided.
5. Low vision and phone: body text of at least 16 px, contrast of at least 4.5:1, reflow at 320 px and 400% zoom with no horizontal scroll, relative units, light and dark themes. Touch targets of at least 44 px. Pinch-zoom stays enabled. Data tables scroll inside their own region, with that region focusable.
6. Plain language, with units and sources stated on every figure.

**Statement of meaning:** A page is finished only when a person using a keyboard, a screen reader, magnification or a thumb gets the same finding and the same calculator result as everyone else.
