(function () {
  // The theme itself is set before first paint by a script in <head>; this
  // only wires up the button and tells the charts when to restyle.
  var root = document.documentElement;
  var button = document.getElementById("themeToggle");
  var label = document.getElementById("themeLabel");
  var KEY = "tempobench-theme";

  function stored() {
    try {
      return localStorage.getItem(KEY);
    } catch (e) {
      return null; // storage blocked (private window, file:// policy)
    }
  }

  function apply(theme) {
    if (root.getAttribute("data-theme") !== theme) {
      root.setAttribute("data-theme", theme);
      document.dispatchEvent(new CustomEvent("tempobench:themechange"));
    }
    var dark = theme === "dark";
    if (button) button.setAttribute("aria-pressed", dark ? "true" : "false");
    if (label) label.textContent = dark ? "Light" : "Dark";
  }

  apply(root.getAttribute("data-theme") === "dark" ? "dark" : "light");

  if (button) {
    button.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      try {
        localStorage.setItem(KEY, next);
      } catch (e) {
        /* the choice just won't outlive the page */
      }
      apply(next);
    });
  }

  // Follow the operating system until the reader picks a theme themselves.
  if (window.matchMedia) {
    var query = window.matchMedia("(prefers-color-scheme: dark)");
    var follow = function (event) {
      if (!stored()) apply(event.matches ? "dark" : "light");
    };
    if (query.addEventListener) query.addEventListener("change", follow);
  }
})();
