// Phone menu: on narrow screens the seven nav links fold behind one Menu button, so the landing
// chart fits the first screen. Without this script the full list shows, as on desktop.
(function () {
  var nav = document.querySelector(".site-head nav");
  var list = nav && nav.querySelector(".site-nav");
  if (!list) return;
  document.documentElement.classList.add("js");
  var btn = document.createElement("button");
  btn.type = "button";
  btn.className = "menu-btn";
  btn.textContent = "Menu";
  btn.setAttribute("aria-expanded", "false");
  btn.setAttribute("aria-controls", "site-nav");
  list.id = "site-nav";
  nav.insertBefore(btn, list);
  function set(open) {
    btn.setAttribute("aria-expanded", String(open));
    list.classList.toggle("open", open);
  }
  btn.addEventListener("click", function () { set(btn.getAttribute("aria-expanded") !== "true"); });
  document.addEventListener("keydown", function (ev) {
    if (ev.key === "Escape" && btn.getAttribute("aria-expanded") === "true") { set(false); btn.focus(); }
  });
})();
