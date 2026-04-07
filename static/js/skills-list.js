(function () {
  "use strict";

  function getCsrf() {
    var meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute("content") : "";
  }

  var toastEl = document.getElementById("skills-toast");
  var toastTimer = null;

  function showToast(message) {
    if (!toastEl) return;
    toastEl.textContent = message;
    toastEl.classList.remove("hidden");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(function () {
      toastEl.classList.add("hidden");
    }, 4000);
  }

  function levelLabel(level) {
    if (level === "system") return "built-in";
    if (level === "org") return "organization";
    if (level === "user") return "your";
    return level;
  }

  function applyToggleResult(row, slug, enabled, replaced) {
    // Update this row's checkbox + opacity + dropdown button label.
    var checkbox = row.querySelector(".skill-toggle");
    if (checkbox) checkbox.checked = enabled;
    if (enabled) {
      row.classList.remove("opacity-50");
    } else {
      row.classList.add("opacity-50");
    }
    var btn = row.querySelector(".skill-toggle-btn");
    if (btn) btn.textContent = enabled ? "Disable" : "Enable";

    // If we just enabled a skill that replaced a previously-active sibling,
    // mirror the change in any visible row that matches the replaced id.
    if (enabled && replaced && replaced.id) {
      var others = document.querySelectorAll(
        '.skill-row[data-skill-slug="' + slug + '"]'
      );
      others.forEach(function (other) {
        if (other === row) return;
        var cb = other.querySelector(".skill-toggle");
        if (cb) cb.checked = false;
        other.classList.add("opacity-50");
        var ob = other.querySelector(".skill-toggle-btn");
        if (ob) ob.textContent = "Enable";
      });
    }
  }

  function postToggle(row, enabled) {
    var url = row.getAttribute("data-toggle-url");
    if (!url) return;
    var slug = row.getAttribute("data-skill-slug");
    var body = new URLSearchParams();
    body.set("enabled", enabled ? "1" : "0");
    body.set("csrfmiddlewaretoken", getCsrf());

    fetch(url, {
      method: "POST",
      credentials: "same-origin",
      headers: {
        "X-CSRFToken": getCsrf(),
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: body.toString(),
    })
      .then(function (resp) {
        return resp.json();
      })
      .then(function (data) {
        if (!data || !data.ok) {
          showToast("Could not update this skill. Please try again.");
          // Revert checkbox state.
          var checkbox = row.querySelector(".skill-toggle");
          if (checkbox) checkbox.checked = !enabled;
          return;
        }
        applyToggleResult(row, slug, data.now_active, data.replaced);
        if (data.replaced) {
          showToast(
            "Disabled the " +
              levelLabel(data.replaced.level) +
              " version of " +
              data.replaced.name +
              " because your version is now active."
          );
        }
      })
      .catch(function () {
        showToast("Could not update this skill. Please try again.");
        var checkbox = row.querySelector(".skill-toggle");
        if (checkbox) checkbox.checked = !enabled;
      });
  }

  document.addEventListener("change", function (e) {
    var target = e.target;
    if (!target || !target.classList || !target.classList.contains("skill-toggle")) return;
    var row = target.closest(".skill-row");
    if (!row) return;
    postToggle(row, target.checked);
  });

  document.addEventListener("click", function (e) {
    var btn = e.target.closest(".skill-toggle-btn");
    if (!btn) return;
    var row = btn.closest(".skill-row");
    if (!row) return;
    var checkbox = row.querySelector(".skill-toggle");
    if (!checkbox) return;
    checkbox.checked = !checkbox.checked;
    postToggle(row, checkbox.checked);
  });
})();
