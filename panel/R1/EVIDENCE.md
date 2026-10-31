# R1 WHOLE round evidence: the redesigned MacroEconomics site

Artifact: D:\New folder\PrometheusIntelligence\MacroEconomics\site\ (9 pages: index, oil, inflation, ports, cfa-franc, funding, crypto-strategies, data, 404; css\site.css; js\site.js, js\qf.js). Repaired data: ..\_src\data\ (repairs.json lists every repair); raw files unchanged in ..\_src\raw\. Design log with moves M1 to M9: ..\DESIGN-LOG.md. Screenshots here: home-1280.jpg, home-375.jpg (the landing's first screen), cfa-1280.jpg, ports-1280-before-dot-fix.jpg (since fixed: where both ranks are equal the dot now shows as a filled dot inside a ring).

Measured in the in-app browser after the last edit (2026-10-31):
- First screen rule. 1280 x 800: landing chart top y = 282, bottom y = 732, legend bottom 764 (inside 800). 375 x 812: chart top 252, bottom 681, legend bottom 767 (inside 812); header 78 px (the 7 nav links fold behind a 44 px Menu button on phones; without JS the full list shows). Horizontal overflow 0 at both.
- Horizontal overflow at 375 px: 0 on all 8 content pages. No console errors.
- Calculator: starting example splits a $1,000 pool $0 / $750 / $250 (one $100 gift; ten $10 gifts; four $25 gifts). Changing Project A to "50, 50" gives $77 / $692 / $231. Invalid input ("50, abc") shows "Project A: enter amounts as numbers separated by commas, for example 10, 10, 10.", sets aria-invalid, and clears when fixed. Results sit in an aria-live="polite" region; without JS the same example shows as a static table.
- Inflation chart end labels were overlapping; fixed (labels 20 units apart).
- Build: 9 pages; every link, #anchor, id and JSON-LD block checked.
