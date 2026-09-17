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

  // Opacity lives on the row's [data-dim] content elements, never on the row
  // itself, so the ⋯ dropdown stays fully opaque when the skill is off.
  function setRowDimmed(row, enabled) {
    row.classList.remove("opacity-50");
    row.querySelectorAll("[data-dim]").forEach(function (el) {
      el.classList.toggle("opacity-50", !enabled);
    });
  }

  // Which status pill a row shows — the same rule the server uses
  // (views._scan_pill): usable → none; blocked → blocked; pending → pending;
  // otherwise "scan needed" only when the user has the skill switched on.
  function pillFor(selected, approved, scanState) {
    if (approved) return "";
    if (scanState === "blocked") return "blocked";
    if (scanState === "pending") return "pending";
    return selected ? "needed" : "";
  }

  // Render a row from its safety-scan verdict. `info` is a toggle or
  // scan-status response ({approved, scan_state, detail, now_active?});
  // missing fields fall back to the row's data-* attributes.
  function applyScanState(row, info) {
    info = info || {};
    var selected = typeof info.now_active === "boolean"
      ? info.now_active
      : row.getAttribute("data-selected") === "1";
    var approved = typeof info.approved === "boolean"
      ? info.approved
      : row.getAttribute("data-approved") === "1";
    var scanState = typeof info.scan_state === "string"
      ? info.scan_state
      : (row.getAttribute("data-scan-state") || "");
    row.setAttribute("data-selected", selected ? "1" : "0");
    row.setAttribute("data-approved", approved ? "1" : "0");
    row.setAttribute("data-scan-state", scanState);

    var pending = scanState === "pending" && !approved;
    var enabled = selected && approved; // effective availability

    var checkbox = row.querySelector(".skill-toggle");
    if (checkbox) checkbox.checked = selected && (approved || scanState === "pending");
    setRowDimmed(row, enabled);
    var btn = row.querySelector(".skill-toggle-btn");
    if (btn) btn.textContent = enabled ? "Disable" : "Enable";

    var pill = pillFor(selected, approved, scanState);
    row.querySelectorAll(".scan-pill").forEach(function (el) {
      var kind = el.getAttribute("data-pill");
      // The `hidden` attribute, not the class: preflight's [hidden] rule is
      // !important, whereas the class loses to the pill's inline-flex.
      el.hidden = kind !== pill;
      if (kind === "blocked" && typeof info.detail === "string" && info.detail) {
        el.setAttribute("title", info.detail);
      }
    });
    // Busy (spinner + locked controls) exactly while the scan is running.
    setToggleBusy(row, pending);
  }

  function applyToggleResult(row, slug, enabled, replaced) {
    // Update this row's checkbox + opacity + dropdown button label.
    var checkbox = row.querySelector(".skill-toggle");
    if (checkbox) checkbox.checked = enabled;
    setRowDimmed(row, enabled);
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
        applyScanState(other, { now_active: false });
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
        // The selection is saved immediately; the safety scan (if one is
        // needed) runs on the worker. The row shows the effective state and
        // polls while the scan is pending.
        applyToggleResult(row, slug, !!(data.now_active && data.approved), data.replaced);
        applyScanState(row, data);
        if (data.scan_state === "pending" && !data.approved) {
          pollScanStatus();
        } else if (data.scan_state === "blocked" && !data.approved && data.now_active) {
          showToast(data.detail || "This skill was blocked by the safety scan.");
        }
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

  // ----- Safety-scan status polling -----
  // While any row is "Scanning…" (pending and not yet approved), ask the
  // server for those rows' verdicts every 2.5 s and re-render them; stops by
  // itself once nothing is pending. Same shape as the resource-status poll.
  var configEl = document.getElementById("skills-list-config");
  var scanStatusUrl = configEl ? configEl.getAttribute("data-scan-status-url") : "";
  var scanPolling = false;

  function pendingScanIds() {
    var ids = [];
    document
      .querySelectorAll('.skill-row[data-scan-state="pending"][data-approved="0"]')
      .forEach(function (row) {
        var id = row.getAttribute("data-skill-id");
        if (id && ids.indexOf(id) === -1) ids.push(id);
      });
    return ids;
  }

  function pollScanStatus() {
    if (scanPolling || !scanStatusUrl || !pendingScanIds().length) return;
    scanPolling = true;
    setTimeout(function () {
      var ids = pendingScanIds();
      if (!ids.length) {
        scanPolling = false;
        return;
      }
      fetch(scanStatusUrl + "?ids=" + encodeURIComponent(ids.join(",")), {
        credentials: "same-origin",
        headers: { "X-Requested-With": "XMLHttpRequest", "Accept": "application/json" },
      })
        .then(function (resp) {
          return resp.json();
        })
        .then(function (data) {
          scanPolling = false;
          if (data && data.ok && data.skills) {
            Object.keys(data.skills).forEach(function (id) {
              document
                .querySelectorAll('.skill-row[data-skill-id="' + id + '"]')
                .forEach(function (row) {
                  applyScanState(row, data.skills[id]);
                });
            });
          }
          pollScanStatus();
        })
        .catch(function () {
          scanPolling = false;
        });
    }, 2500);
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

  // Rows the server rendered as "Scanning…" (e.g. right after a save
  // redirected here) resolve without a reload.
  pollScanStatus();

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
