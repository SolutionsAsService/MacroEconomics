# Critic: Inclusion (Mace), round R1

**Anchor used:** Principle 4, Perceptible Information, with Principle 1, Equitable Use, and Principle 2, Flexibility in Use.

**Observed:** Most R0 requests are met. Every page has lang, a skip link, header/nav/main/footer landmarks and a labelled nav. Charts carry role="img" and finding-stating aria-labels, with a "Show the numbers as a table" details twin; the tables sit in focusable scroll regions (tabindex 0, aria-label). The calculator has labelled inputs with units, a help text tied by aria-describedby, aria-invalid, a role="alert" error that names the fix, a polite live region and a no-script table. The Menu button sets aria-expanded and aria-controls, is 44 px, and the full list shows without JS. Nav, card and footer targets are 44 px or more, the 16 px+ body text reflows to 375 px, and there are dark theme tokens. The one gap: OPEC membership in the oil bars (hero, production chart) is carried by red versus near-black fill alone. The legend keys are aria-hidden, and the chart aria-labels do not say which bars are OPEC. Red against ink is a weak difference for protan and deuteranopic readers, and in dark theme #ff5a5f against the cream bars is weaker still. Smaller: tables have no caption element, though the region labels cover it; Menu does not close on Escape.

**Focus visibility:** Named check: :focus-visible { outline 3px solid var(--red); offset 2px } is global and never removed. Red on paper is about 5:1 in light and the light red on dark paper is above 3:1. Result: pass, with the Menu button and inputs showing the ring.

**Verdict:** OBJECT (one item).

**Requested change:** Add a non-colour cue for OPEC in the oil bars: an "OPEC" text tag, or a hatch pattern, on the bar or its value label, mirrored in the legend key. Also name the OPEC members in the chart aria-labels. Optional: Escape closes the Menu.

**Statement of meaning:** Everything works for keyboard, screen reader and thumb; the OPEC-versus-not finding must also reach a reader who cannot tell red from black.
