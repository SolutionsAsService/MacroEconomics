// Country explorer: loads data/macro_countries.json and draws five small SVG charts for one or two countries.
// No library. A gap in a line is a year with no figure; nothing is interpolated.
(function () {
  var root = document.getElementById("explorer");
  if (!root) return;
  var inA = document.getElementById("ex-a"), inB = document.getElementById("ex-b");
  var list = document.getElementById("ex-list"), status = document.getElementById("ex-status");
  var tiles = document.getElementById("ex-tiles"), grid = document.getElementById("ex-grid");
  var data, byName = {}, byCode = {};
  var NS = "http://www.w3.org/2000/svg";

  function val(c, key, y) {
    var s = c.s[key]; if (!s) return null;
    var i = y - s.y0; return i >= 0 && i < s.v.length ? s.v[i] : null;
  }
  function lastPt(c, key) {
    var s = c.s[key]; if (!s) return null;
    for (var i = s.v.length - 1; i >= 0; i--) if (s.v[i] != null) return { y: s.y0 + i, v: s.v[i] };
    return null;
  }
  function pts(c, key, fn) {
    var s = c.s[key], out = []; if (!s) return out;
    for (var i = 0; i < s.v.length; i++) {
      var v = s.v[i], y = s.y0 + i;
      if (fn) v = fn(c, y, v);
      out.push([y, v]);
    }
    return out;
  }
  function usd(v) {
    var a = Math.abs(v);
    if (a >= 1e12) return "$" + (v / 1e12).toFixed(1) + " tn";
    if (a >= 1e9) return "$" + (v / 1e9).toFixed(a >= 1e10 ? 0 : 1) + " bn";
    return "$" + (v / 1e6).toFixed(0) + " m";
  }
  function ppl(v) { return v >= 1e9 ? (v / 1e9).toFixed(2) + " bn" : v >= 1e6 ? (v / 1e6).toFixed(1) + " m" : Math.round(v / 1e3) + " k"; }
  function num(v, d) { if (Math.abs(v) < Math.pow(10, -(d || 0)) / 2) v = 0; return v.toLocaleString("en-US", { maximumFractionDigits: d == null ? 0 : d }); }

  function rank(key, c, fn) {
    var l = lastPt(c, key); if (!l) return null;
    var y = l.y, mine = fn ? fn(c, y) : l.v, n = 0, better = 0;
    data.countries.forEach(function (o) {
      var v = fn ? fn(o, y) : val(o, key, y);
      if (v != null) { n++; if (v > mine) better++; }
    });
    return { rank: better + 1, of: n, y: y };
  }
  var gdppc = function (c, y) { var g = val(c, "gdp", y), p = val(c, "pop", y); return g != null && p ? g / p : null; };

  function svg(el, attrs, text) {
    var n = document.createElementNS(NS, el);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    return n;
  }
  function ticks(lo, hi, n) {
    var span = hi - lo || 1, step = Math.pow(10, Math.floor(Math.log10(span / n)));
    [1, 2, 2.5, 5, 10].some(function (m) { if (span / (step * m) <= n) { step *= m; return true; } });
    var out = [], t = Math.ceil(lo / step) * step;
    while (t <= hi + 1e-9) { out.push(t); t += step; }
    return out;
  }

  // series: [{name, pts:[[year, value|null]], cls}], fmt for the y axis
  function chart(series, fmt, zero) {
    var W = 420, H = 230, L = 52, R = 12, T = 10, B = 26;
    var xs = [], ys = [];
    series.forEach(function (s) { s.pts.forEach(function (p) { xs.push(p[0]); if (p[1] != null) ys.push(p[1]); }); });
    var el = svg("svg", { viewBox: "0 0 " + W + " " + H, role: "img" });
    if (!ys.length) {
      el.appendChild(svg("text", { x: W / 2, y: H / 2, "text-anchor": "middle", class: "tick-l" }, "No figures for this country"));
      return el;
    }
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys);
    if (zero) lo = Math.min(lo, 0);
    if (hi === lo) hi = lo + 1;
    var X = function (x) { return L + (W - L - R) * (x - x0) / ((x1 - x0) || 1); };
    var Y = function (v) { return H - B - (H - T - B) * (v - lo) / (hi - lo); };
    el.appendChild(svg("line", { class: "axis", x1: L, x2: W - R, y1: H - B, y2: H - B }));
    ticks(lo, hi, 4).forEach(function (t) {
      el.appendChild(svg("line", { class: "grid", x1: L, x2: W - R, y1: Y(t), y2: Y(t) }));
      el.appendChild(svg("text", { class: "tick-l", x: L - 6, y: Y(t) + 4, "text-anchor": "end", style: "font-size:12px" }, fmt(t)));
    });
    ticks(x0, x1, 4).forEach(function (t) {
      el.appendChild(svg("text", { class: "tick-l", x: X(t), y: H - 8, "text-anchor": "middle", style: "font-size:12px" }, String(Math.round(t))));
    });
    series.forEach(function (s, i) {
      var d = "", pen = false, only = null;
      s.pts.forEach(function (p, j) {
        if (p[1] == null) { pen = false; return; }
        d += (pen ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1);
        pen = true;
      });
      var path = svg("path", { class: "ln" + (i === 0 ? " red" : ""), d: d });
      if (i === 1) path.setAttribute("stroke-dasharray", "10 5");
      el.appendChild(path);
    });
    return el;
  }

  function figure(title, series, fmt, zero, note) {
    var f = document.createElement("figure");
    var h = document.createElement("h3"); h.textContent = title; f.appendChild(h);
    var c = chart(series, fmt, zero);
    var names = series.map(function (s) { return s.name; }).join(" and ");
    var desc = title + ", " + names + ": " + series.map(function (s) {
      var p = s.pts.filter(function (q) { return q[1] != null; });
      return p.length ? s.name + " from " + fmt(p[0][1]) + " in " + p[0][0] + " to " + fmt(p[p.length - 1][1]) + " in " + p[p.length - 1][0] : s.name + " no figures";
    }).join("; ") + ".";
    c.setAttribute("aria-label", desc);
    f.appendChild(c);
    if (series.length > 1) {
      var k = document.createElement("p"); k.className = "ex-key";
      k.textContent = "Solid red: " + series[0].name + ". Dashed: " + series[1].name + ".";
      f.appendChild(k);
    }
    if (note) { var p = document.createElement("p"); p.className = "cap"; p.textContent = note; f.appendChild(p); }
    return f;
  }

  function tile(k, n, s) {
    var li = document.createElement("li");
    li.innerHTML = '<span class="k"></span><span class="n"></span><span class="s"></span>';
    li.children[0].textContent = k; li.children[1].textContent = n; li.children[2].textContent = s;
    return li;
  }

  function render() {
    var a = byName[inA.value.trim().toLowerCase()], b = byName[inB.value.trim().toLowerCase()];
    tiles.innerHTML = ""; grid.innerHTML = "";
    if (!a) { status.textContent = inA.value ? "No country by that name. Pick one from the list." : "Choose a country to begin."; return; }
    status.textContent = a.name + (b ? " compared with " + b.name : "") + ": " + (a.region || "") + (a.sub ? ", " + a.sub : "") + ".";
    history.replaceState(null, "", "#" + a.code + (b ? "," + b.code : ""));
    var g = lastPt(a, "gdp"), p = lastPt(a, "pop"), i = lastPt(a, "infl"), c = lastPt(a, "co2pc");
    var rg = rank("gdp", a), rp = rank("pop", a), rpc = rank("gdp", a, gdppc);
    if (g) tiles.appendChild(tile("GDP " + g.y, usd(g.v), rg ? "No. " + rg.rank + " of " + rg.of : ""));
    if (g && p) { var pc = gdppc(a, g.y); if (pc != null) tiles.appendChild(tile("GDP per person", "$" + num(pc), rpc ? "No. " + rpc.rank + " of " + rpc.of : "")); }
    if (p) tiles.appendChild(tile("People " + p.y, ppl(p.v), rp ? "No. " + rp.rank + " of " + rp.of : ""));
    if (i) tiles.appendChild(tile("Inflation " + i.y, num(i.v, 1) + "%", "consumer prices"));
    if (c) tiles.appendChild(tile("CO2 per person " + c.y, num(c.v, 1) + " t", "tonnes a year"));
    if (!tiles.children.length) tiles.appendChild(tile("No figures", "", "for this country"));
    var both = function (key, fn) {
      var out = [{ name: a.name, pts: pts(a, key, fn) }];
      if (b) out.push({ name: b.name, pts: pts(b, key, fn) });
      return out;
    };
    grid.appendChild(figure("GDP, current US$", both("gdp"), usd, true, "Not adjusted for prices."));
    grid.appendChild(figure("GDP per person, current US$", [a].concat(b ? [b] : []).map(function (c) {
      var y0 = c.s.gdp ? c.s.gdp.y0 : 1960, out = [];
      var ys = c.s.gdp ? c.s.gdp.v.length : 0;
      for (var k = 0; k < ys; k++) out.push([y0 + k, gdppc(c, y0 + k)]);
      return { name: c.name, pts: out };
    }), function (v) { return "$" + num(v); }, true));
    grid.appendChild(figure("Population", both("pop"), ppl, true));
    grid.appendChild(figure("Inflation, % a year", both("infl"), function (v) { return num(v, 1) + "%"; }, false, "Consumer prices. Very high rates squash the rest of the line."));
    grid.appendChild(figure("CO2 per person, tonnes", both("co2pc"), function (v) { return num(v, 1); }, true, "From fossil fuels and cement, within the country's borders."));
  }

  function pick() { render(); }

  fetch(root.getAttribute("data-src")).then(function (r) { return r.json(); }).then(function (d) {
    data = d;
    d.countries.forEach(function (c) {
      byName[c.name.toLowerCase()] = c; byCode[c.code] = c;
      var o = document.createElement("option"); o.value = c.name; list.appendChild(o);
    });
    var h = (location.hash || "").slice(1).split(",");
    inA.value = (byCode[h[0]] || byCode.KEN || d.countries[0]).name;
    if (h[1] && byCode[h[1]]) inB.value = byCode[h[1]].name;
    inA.addEventListener("input", pick); inB.addEventListener("input", pick);
    inA.addEventListener("focus", function () { inA.select(); }); inB.addEventListener("focus", function () { inB.select(); });
    render();
  }).catch(function () { status.textContent = "The data could not be loaded. It is also at data/macro_countries.json."; });
})();
