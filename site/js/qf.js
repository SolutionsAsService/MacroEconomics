// Quadratic funding calculator. The same rule as qf() in _src/build.py: each project's match is
// (sum of square roots of its contributions)^2 minus the sum, scaled so matches total the pool.
// Without this script the page shows the same worked example as a static table.
(function () {
  var form = document.getElementById("qf");
  if (!form) return;
  var out = document.getElementById("qf-out");
  var err = document.getElementById("qf-err");
  var money = function (v) { return "$" + Math.round(v).toLocaleString("en-US"); };

  function parse(input) {
    var parts = input.value.split(",").map(function (s) { return s.trim(); }).filter(Boolean);
    var nums = parts.map(Number);
    if (!parts.length || nums.some(function (n) { return !isFinite(n) || n < 0; })) return null;
    return nums;
  }

  function render() {
    var inputs = form.querySelectorAll("input[id^='qf-']:not(#qf-pool)");
    var projects = [], bad = [];
    inputs.forEach(function (i) {
      var c = parse(i);
      i.setAttribute("aria-invalid", c ? "false" : "true");
      if (c) projects.push({ name: i.name, c: c }); else bad.push(i.name);
    });
    var pool = Number(form.pool.value);
    var poolOk = form.pool.value.trim() !== "" && isFinite(pool) && pool >= 0;
    form.pool.setAttribute("aria-invalid", poolOk ? "false" : "true");
    if (bad.length || !poolOk) {
      err.textContent = (bad.length ? bad.join(" and ") + ": enter amounts as numbers separated by commas, for example 10, 10, 10. " : "") +
        (poolOk ? "" : "Matching pool: enter a number of dollars, 0 or more.");
      err.hidden = false;
      return;
    }
    err.hidden = true;
    var total = 0;
    projects.forEach(function (p) {
      var roots = p.c.reduce(function (s, v) { return s + Math.sqrt(v); }, 0);
      p.given = p.c.reduce(function (s, v) { return s + v; }, 0);
      p.raw = roots * roots - p.given;
      total += p.raw;
    });
    var maxGot = 0;
    projects.forEach(function (p) {
      p.match = total > 0 ? pool * p.raw / total : 0;
      maxGot = Math.max(maxGot, p.given + p.match);
    });
    var rows = projects.map(function (p) {
      var w = maxGot > 0 ? (100 * (p.given + p.match) / maxGot).toFixed(1) : 0;
      return "<tr><th scope='row'>" + p.name + "</th><td>" + p.c.length + (p.c.length === 1 ? " gift" : " gifts") +
        "</td><td class='num'>" + money(p.given) + "</td><td class='num'>" + money(p.raw) + "</td><td class='num'>" + money(p.match) +
        "</td><td class='num'>" + money(p.given + p.match) + "<span class='qf-bar' style='width:" + w + "%' aria-hidden='true'></span></td></tr>";
    }).join("");
    out.querySelector("tbody").innerHTML = rows;
    out.querySelector("thead tr th:nth-child(2)").textContent = "Contributions";
  }

  form.addEventListener("input", render);
  form.addEventListener("submit", function (ev) { ev.preventDefault(); render(); });
  render();
})();
