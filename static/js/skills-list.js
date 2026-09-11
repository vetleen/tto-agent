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
    if (level === "system") return "system";
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

  var SCAN_SPINNER =
    '<svg class="w-4 h-4 animate-spin text-body" viewBox="0 0 24 24" fill="none"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path></svg>';

  // Enabling a not-yet-approved skill runs a safety scan server-side, which can
  // take a few seconds — show a spinner by the toggle and lock it until it lands.
  function setToggleBusy(row, busy) {
    var checkbox = row.querySelector(".skill-toggle");
    if (checkbox) checkbox.disabled = busy;
    var btn = row.querySelector(".skill-toggle-btn");
    if (btn) btn.disabled = busy;
    var existing = row.querySelector(".skill-scan-spinner");
    if (busy && !existing) {
      var label = checkbox ? checkbox.closest("label") : null;
      var anchor = label || checkbox;
      if (anchor && anchor.parentNode) {
        var sp = document.createElement("span");
        sp.className = "skill-scan-spinner inline-flex items-center ms-2 align-middle";
        sp.setAttribute("title", "Scanning…");
        sp.innerHTML = SCAN_SPINNER;
        anchor.parentNode.insertBefore(sp, anchor.nextSibling);
      }
    } else if (!busy && existing) {
      existing.remove();
    }
  }

  function postToggle(row, enabled) {
    var url = row.getAttribute("data-toggle-url");
    if (!url) return;
    var slug = row.getAttribute("data-skill-slug");
    var body = new URLSearchParams();
    body.set("enabled", enabled ? "1" : "0");
    body.set("csrfmiddlewaretoken", getCsrf());

    setToggleBusy(row, true);

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
        setToggleBusy(row, false);
        if (!data || !data.ok) {
          if (data && data.error === "blocked") {
            showToast(
              data.detail ||
                "This skill couldn't be enabled — its content was flagged by the safety scan."
            );
          } else {
            showToast("Could not update this skill. Please try again.");
          }
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
        setToggleBusy(row, false);
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

  // Import skill: each audience tab has its own button proxying to a hidden
  // file input inside its form, which auto-submits once a file is picked.
  document.querySelectorAll(".import-skill-btn").forEach(function (importBtn) {
    var form = importBtn.closest("form");
    if (!form) return;
    var importInput = form.querySelector(".import-skill-input");
    if (!importInput) return;
    importBtn.addEventListener("click", function () {
      importInput.click();
    });
    importInput.addEventListener("change", function () {
      if (importInput.files && importInput.files.length > 0) {
        form.submit();
      }
    });
  });

  // Deep-link to a tab via ?tab=subagent (e.g. "Back to skills" from a
  // sub-agent skill's detail page). Flowbite initializes its Tabs instance
  // after this script runs, so poll for it, then drive its own show() API
  // (which keeps Flowbite's internal active-tab state consistent — clicking
  // before init, or after a later re-init, would silently revert to main).
  if (new URLSearchParams(window.location.search).get("tab") === "subagent") {
    var tabTries = 0;
    var tabTimer = setInterval(function () {
      var inst = window.FlowbiteInstances &&
        window.FlowbiteInstances.getInstance("Tabs", "skills-tabs");
      if (inst && typeof inst.show === "function") {
        inst.show("#panel-subagent");
        clearInterval(tabTimer);
      } else if (++tabTries > 50) {
        clearInterval(tabTimer);
      }
    }, 100);
  }
})();
