(function () {
  // Renders every chart on the page, and again in the other palette whenever
  // the theme changes: Vega bakes colours into the drawing, so a CSS switch
  // alone would leave dark text on a dark page.
  var specsEl = document.getElementById("tb-chart-specs");
  var darkEl = document.getElementById("tb-chart-dark");
  if (!specsEl) return;
  var specs = JSON.parse(specsEl.textContent);
  var dark = darkEl ? JSON.parse(darkEl.textContent) : {};
  var views = [];

  function merge(base, extra) {
    var out = {};
    var key;
    for (key in base) out[key] = base[key];
    for (key in extra) {
      var a = out[key];
      var b = extra[key];
      var nested = a && b && typeof a === "object" && typeof b === "object" &&
        !Array.isArray(a) && !Array.isArray(b);
      out[key] = nested ? merge(a, b) : b;
    }
    return out;
  }

  function themed(spec) {
    if (document.documentElement.getAttribute("data-theme") !== "dark") return spec;
    return merge(spec, { config: merge(spec.config || {}, dark) });
  }

  function fail(el, err) {
    var pre = document.createElement("pre");
    pre.className = "chart-error";
    pre.textContent = "Error rendering chart: " + err;
    el.replaceChildren(pre);
  }

  function render() {
    views.forEach(function (view) { view.finalize(); });
    views = [];
    specs.forEach(function (spec, i) {
      var el = document.getElementById("chart-" + i);
      if (!el) return;
      if (typeof vegaEmbed === "undefined") {
        fail(el, "the Vega libraries could not be loaded (offline?)");
        return;
      }
      vegaEmbed(el, themed(spec), {
        mode: "vega-lite",
        renderer: "svg",
        actions: { export: true, source: false, compiled: false, editor: false },
      })
        .then(function (result) { views.push(result); })
        .catch(function (err) { fail(el, err); });
    });
  }

  render();
  document.addEventListener("tempobench:themechange", render);
})();
