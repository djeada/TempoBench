(function () {
  const toggle = document.getElementById("themeToggle");
  const icon = document.getElementById("themeIcon");
  const label = document.getElementById("themeLabel");
  const html = document.documentElement;

  const currentTheme = localStorage.getItem("theme") || "light";
  html.setAttribute("data-theme", currentTheme);

  if (currentTheme === "dark") {
    icon.textContent = "☀️";
    label.textContent = "Light";
  }

  toggle.addEventListener("click", function () {
    const theme = html.getAttribute("data-theme");
    const newTheme = theme === "light" ? "dark" : "light";

    html.setAttribute("data-theme", newTheme);
    localStorage.setItem("theme", newTheme);

    if (newTheme === "dark") {
      icon.textContent = "☀️";
      label.textContent = "Light";
    } else {
      icon.textContent = "🌙";
      label.textContent = "Dark";
    }
  });
})();
