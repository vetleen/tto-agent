(function () {
  "use strict";

  var form = document.getElementById("skill-detail-form");
  if (!form) return;

  var config = document.getElementById("skill-detail-config");
  var editable = config && config.getAttribute("data-editable") === "1";
  var skillName = config ? config.getAttribute("data-skill-name") : "";
  var skillLevel = config ? config.getAttribute("data-skill-level") : "user";
  var colleagueCount = parseInt(
    (config && config.getAttribute("data-colleague-count")) || "0",
    10
  );

  var toolNamesInput = document.getElementById("tool-names-json");
  var actionInput = document.getElementById("skill-form-action");

  var toolChipsEl = document.getElementById("tool-chips");

  // ----- State -----
  var toolNames = [];
  try {
    toolNames = JSON.parse(toolNamesInput.value || "[]");
  } catch (e) {
    toolNames = [];
  }

  // ----- Markdown preview helpers -----
  var EYE_ICON = '<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/></svg>';
  var PEN_ICON = '<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"/></svg>';

  function renderMarkdown(text) {
    if (!text) return "";
    try {
      var html = marked.parse(text);
      return DOMPurify.sanitize(html, {
        ALLOWED_TAGS: ["p","br","strong","em","u","s","del","code","pre","ul","ol","li",
                       "h1","h2","h3","h4","h5","h6","blockquote","a","hr",
                       "table","thead","tbody","tr","th","td","div","span","sup","section"],
        ALLOWED_ATTR: ["href","title","target","class","id"]
      });
    } catch (e) {
      return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/\n/g, "<br>");
    }
  }

  // Mounts a CodeMirror markdown editor over a textarea, keeping the (hidden)
  // textarea synced so the Django form POST still submits its value. Returns the
  // editor adapter (or null if the bundle failed to load — in which case the
  // plain textarea is left visible as a graceful fallback). An optional preview
  // button toggles between the editor and a rendered-markdown div.
  //
  // On read-only skills (o.readOnly) pass o.defaultPreview to open in the rendered
  // view: the button then toggles to the raw markdown source instead of an editor,
  // so system skills read as formatted text by default but the source stays a
  // click away.
  function mountMarkdownEditor(o) {
    var textarea = o.textarea;
    if (!textarea) return null;
    if (!window.WilfredEditor) return null; // fallback: leave the textarea visible

    var preview = o.preview;
    var wrapper = document.createElement("div");
    // White-inset field: .wf-input supplies the white background, forest hairline and
    // inset shadow so the editor reads as a clean field cut into the section card.
    wrapper.className = "cm-field wf-input rounded-base overflow-hidden";
    textarea.classList.add("hidden");
    textarea.parentNode.insertBefore(wrapper, textarea.nextSibling);

    var ed = window.WilfredEditor.create(wrapper, {
      value: textarea.value,
      readOnly: !!o.readOnly,
      toolbar: !o.readOnly,
      maxLength: o.maxLength || null,
      minHeight: o.minHeight,
      maxHeight: o.maxHeight,
      placeholder: textarea.getAttribute("placeholder") || "",
      onChange: function () {
        var val = ed.getValue();
        textarea.value = val; // keep the named form field in sync
        if (o.onChange) o.onChange(val);
      },
    });

    if (o.previewBtn && preview) {
      var inPreview = !!o.defaultPreview;
      var applyPreviewState = function () {
        if (inPreview) {
          preview.innerHTML = renderMarkdown(ed.getValue());
          preview.classList.remove("hidden");
          wrapper.classList.add("hidden");
          // On read-only skills (system, or org skills for non-admins) the
          // raw view isn't editable, so "Edit" would mislead — call it
          // "View markdown" instead.
          var rawLabel = o.readOnly ? "View markdown" : "Edit";
          o.previewBtn.innerHTML = "<span>" + rawLabel + "</span>" + PEN_ICON;
          o.previewBtn.title = o.readOnly ? "Toggle markdown source" : "Toggle editing";
        } else {
          preview.classList.add("hidden");
          wrapper.classList.remove("hidden");
          o.previewBtn.innerHTML = "<span>Preview</span>" + EYE_ICON;
          o.previewBtn.title = "Toggle preview";
        }
      };
      o.previewBtn.addEventListener("click", function () {
        inPreview = !inPreview;
        applyPreviewState();
      });
      // Render the initial preview when a skill opens in preview-by-default mode.
      if (inPreview) applyPreviewState();
    }
    return ed;
  }

  // ----- Tools UI -----
  function renderToolChips() {
    toolChipsEl.innerHTML = "";
    if (!toolNames.length) {
      var empty = document.createElement("p");
      empty.className = "text-xs text-body italic";
      empty.textContent = "No tools attached.";
      toolChipsEl.appendChild(empty);
      return;
    }
    var tmpl = document.getElementById("tool-chip-template");
    toolNames.forEach(function (name) {
      var node = tmpl.content.firstElementChild.cloneNode(true);
      node.querySelector(".tool-chip-name").textContent = name;
      // Look up description from the picker list if available.
      var pickerItem = document.querySelector(
        '.tool-picker-checkbox[data-tool-name="' + cssEscape(name) + '"]'
      );
      if (pickerItem) {
        var desc = pickerItem.parentElement.querySelector("p");
        if (desc) node.setAttribute("title", desc.textContent.trim());
      }
      var removeBtn = node.querySelector(".remove-tool-btn");
      if (!editable) {
        removeBtn.remove();
      } else {
        removeBtn.addEventListener("click", function () {
          toolNames = toolNames.filter(function (n) {
            return n !== name;
          });
          syncToolNamesInput();
          renderToolChips();
        });
      }
      toolChipsEl.appendChild(node);
    });
  }

  function syncToolNamesInput() {
    toolNamesInput.value = JSON.stringify(toolNames);
  }

  // CSS.escape polyfill for older browsers in attribute selectors.
  function cssEscape(s) {
    if (window.CSS && window.CSS.escape) return window.CSS.escape(s);
    return String(s).replace(/[^a-zA-Z0-9_-]/g, "\\$&");
  }

  // ----- Tool picker modal -----
  var applyToolPickerBtn = document.getElementById("apply-tool-picker");
  var openToolPickerBtn = document.getElementById("open-tool-picker");

  var toolPickerSearch = document.getElementById("tool-picker-search");
  var toolPickerCount = document.getElementById("tool-picker-count");

  function updateToolPickerCount() {
    if (!toolPickerCount) return;
    var n = document.querySelectorAll(".tool-picker-checkbox:checked").length;
    toolPickerCount.textContent =
      n === 0 ? "No tools selected" : n === 1 ? "1 tool selected" : n + " tools selected";
  }

  function filterToolCards() {
    var q = toolPickerSearch ? toolPickerSearch.value.trim().toLowerCase() : "";
    var anyVisible = false;
    document.querySelectorAll("#tool-picker-list [data-tool-card]").forEach(function (card) {
      var cb = card.querySelector(".tool-picker-checkbox");
      var name = cb ? (cb.getAttribute("data-tool-name") || "").toLowerCase() : "";
      var descEl = card.querySelector("p");
      var desc = descEl ? descEl.textContent.toLowerCase() : "";
      var match = name.indexOf(q) !== -1 || desc.indexOf(q) !== -1;
      // "flex", not "": the card's layout is its inline display:flex.
      card.style.display = match ? "flex" : "none";
      if (match) anyVisible = true;
    });
    var nr = document.getElementById("tool-picker-noresults");
    if (nr) nr.style.display = (q !== "" && !anyVisible) ? "" : "none";
  }

  if (toolPickerSearch) {
    toolPickerSearch.addEventListener("input", filterToolCards);
  }
  document.querySelectorAll(".tool-picker-checkbox").forEach(function (cb) {
    cb.addEventListener("change", updateToolPickerCount);
  });

  // A main playbook hides the Tools box until it has individual tools; the
  // "Add individual tool" link under Requires opens the same picker.
  var addIndividualToolBtn = document.getElementById("add-individual-tool");
  [openToolPickerBtn, addIndividualToolBtn].forEach(function (btn) {
    if (!btn) return;
    btn.addEventListener("click", function () {
      // Pre-check existing tools.
      document.querySelectorAll(".tool-picker-checkbox").forEach(function (cb) {
        cb.checked = toolNames.indexOf(cb.getAttribute("data-tool-name")) !== -1;
      });
      if (toolPickerSearch) toolPickerSearch.value = "";
      filterToolCards();
      updateToolPickerCount();
    });
  });

  if (applyToolPickerBtn) {
    applyToolPickerBtn.addEventListener("click", function () {
      var picked = [];
      document.querySelectorAll(".tool-picker-checkbox:checked").forEach(function (cb) {
        picked.push(cb.getAttribute("data-tool-name"));
      });
      toolNames = picked;
      syncToolNamesInput();
      renderToolChips();
      // Picked tools on a playbook: show the Tools box (it stays until reload).
      var toolsSection = document.getElementById("tools-section");
      if (toolNames.length && toolsSection && toolsSection.hidden) {
        toolsSection.hidden = false;
        if (addIndividualToolBtn) addIndividualToolBtn.hidden = true;
      }
      // Hide modal via Flowbite if present.
      var modal = document.getElementById("tool-picker-modal");
      if (modal && window.FlowbiteInstances) {
        var instance = window.FlowbiteInstances.getInstance(
          "Modal",
          "tool-picker-modal"
        );
        if (instance) instance.hide();
      } else if (modal) {
        modal.classList.add("hidden");
        modal.setAttribute("aria-hidden", "true");
        document.body.classList.remove("overflow-hidden");
        var backdrop = document.querySelector("[modal-backdrop]");
        if (backdrop) backdrop.remove();
      }
    });
  }

  // ----- Requires (toolkits attached automatically with this skill) -----
  // Same anatomy as the tools picker: a hidden JSON input of slugs, chips,
  // and a checkbox modal listing the available toolkits. The section is only
  // rendered for playbooks of the main audience, so every element is optional.
  var requiresInput = document.getElementById("requires-json");
  var requiresChipsEl = document.getElementById("requires-chips");
  var requires = [];
  if (requiresInput) {
    try {
      requires = JSON.parse(requiresInput.value || "[]");
    } catch (e) {
      requires = [];
    }
  }

  function syncRequiresInput() {
    if (requiresInput) requiresInput.value = JSON.stringify(requires);
  }

  // Requires are saved straight away (Apply, a click outside the picker, or a
  // chip's ×) rather than waiting for the page's Save. The hidden input stays
  // in sync, so a later full Save writes the same list.
  var requiresSection = document.getElementById("requires-section");
  var requiresSaveUrl = requiresSection ? requiresSection.getAttribute("data-save-url") : "";
  var savedRequires = JSON.stringify(requires);

  function saveRequires() {
    var payload = JSON.stringify(requires);
    if (!editable || !requiresSaveUrl || payload === savedRequires) return;
    var errorEl = document.getElementById("requires-save-error");
    var csrfEl = form.querySelector("[name=csrfmiddlewaretoken]");
    var body = new FormData();
    body.append("requires_json", payload);
    fetch(requiresSaveUrl, {
      method: "POST",
      body: body,
      credentials: "same-origin",
      headers: {
        "X-CSRFToken": csrfEl ? csrfEl.value : "",
        "X-Requested-With": "XMLHttpRequest",
      },
    })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (data) {
        // The server drops unknown slugs: show what was actually stored.
        requires = data.requires || [];
        savedRequires = JSON.stringify(requires);
        syncRequiresInput();
        renderRequiresChips();
        if (errorEl) errorEl.classList.add("hidden");
      })
      .catch(function () {
        if (errorEl) errorEl.classList.remove("hidden");
      });
  }

  function requiresPickerCard(slug) {
    return document.querySelector(
      '.requires-picker-checkbox[data-skill-slug="' + cssEscape(slug) + '"]'
    );
  }

  function renderRequiresChips() {
    if (!requiresChipsEl) return;
    requiresChipsEl.innerHTML = "";
    if (!requires.length) {
      var empty = document.createElement("p");
      empty.className = "text-xs text-body italic";
      empty.textContent = "No toolkits required.";
      requiresChipsEl.appendChild(empty);
      return;
    }
    var tmpl = document.getElementById("requires-chip-template");
    requires.forEach(function (slug) {
      var node = tmpl.content.firstElementChild.cloneNode(true);
      var cb = requiresPickerCard(slug);
      var label = cb ? cb.getAttribute("data-skill-name") || slug : slug;
      node.querySelector(".tool-chip-name").textContent = label;
      node.setAttribute("title", slug);
      var removeBtn = node.querySelector(".remove-tool-btn");
      if (!editable) {
        removeBtn.remove();
      } else {
        removeBtn.addEventListener("click", function () {
          requires = requires.filter(function (s) { return s !== slug; });
          syncRequiresInput();
          renderRequiresChips();
          saveRequires();
        });
      }
      requiresChipsEl.appendChild(node);
    });
  }

  var applyRequiresPickerBtn = document.getElementById("apply-requires-picker");
  var openRequiresPickerBtn = document.getElementById("open-requires-picker");
  var requiresPickerSearch = document.getElementById("requires-picker-search");
  var requiresPickerCount = document.getElementById("requires-picker-count");

  function updateRequiresPickerCount() {
    if (!requiresPickerCount) return;
    var n = document.querySelectorAll(".requires-picker-checkbox:checked").length;
    requiresPickerCount.textContent =
      n === 0 ? "No toolkits selected" : n === 1 ? "1 toolkit selected" : n + " toolkits selected";
  }

  function filterRequiresCards() {
    var q = requiresPickerSearch ? requiresPickerSearch.value.trim().toLowerCase() : "";
    var anyVisible = false;
    document.querySelectorAll("#requires-picker-list [data-requires-card]").forEach(function (card) {
      var cb = card.querySelector(".requires-picker-checkbox");
      var hay = cb
        ? ((cb.getAttribute("data-skill-slug") || "") + " " + (cb.getAttribute("data-skill-name") || "")).toLowerCase()
        : "";
      var descEl = card.querySelector("p");
      if (descEl) hay += " " + descEl.textContent.toLowerCase();
      var match = hay.indexOf(q) !== -1;
      card.style.display = match ? "flex" : "none";
      if (match) anyVisible = true;
    });
    var nr = document.getElementById("requires-picker-noresults");
    if (nr) nr.style.display = (q !== "" && !anyVisible) ? "" : "none";
  }

  if (requiresPickerSearch) {
    requiresPickerSearch.addEventListener("input", filterRequiresCards);
  }
  document.querySelectorAll(".requires-picker-checkbox").forEach(function (cb) {
    cb.addEventListener("change", updateRequiresPickerCount);
  });

  if (openRequiresPickerBtn) {
    openRequiresPickerBtn.addEventListener("click", function () {
      document.querySelectorAll(".requires-picker-checkbox").forEach(function (cb) {
        cb.checked = requires.indexOf(cb.getAttribute("data-skill-slug")) !== -1;
      });
      if (requiresPickerSearch) requiresPickerSearch.value = "";
      filterRequiresCards();
      updateRequiresPickerCount();
    });
  }

  function applyRequiresPicker() {
    var picked = [];
    document.querySelectorAll(".requires-picker-checkbox:checked").forEach(function (cb) {
      picked.push(cb.getAttribute("data-skill-slug"));
    });
    requires = picked;
    syncRequiresInput();
    renderRequiresChips();
    saveRequires();
  }

  // A click outside the picker (on the modal's backdrop area) keeps the
  // selection, like Apply; Flowbite closes the modal itself. Cancel, × and
  // Escape still discard it.
  var requiresModalEl = document.getElementById("requires-picker-modal");
  if (requiresModalEl) {
    requiresModalEl.addEventListener("click", function (e) {
      if (e.target === requiresModalEl) applyRequiresPicker();
    });
  }

  if (applyRequiresPickerBtn) {
    applyRequiresPickerBtn.addEventListener("click", function () {
      applyRequiresPicker();
      var modal = document.getElementById("requires-picker-modal");
      if (modal && window.FlowbiteInstances) {
        var instance = window.FlowbiteInstances.getInstance("Modal", "requires-picker-modal");
        if (instance) instance.hide();
      } else if (modal) {
        modal.classList.add("hidden");
        modal.setAttribute("aria-hidden", "true");
        document.body.classList.remove("overflow-hidden");
        var backdrop = document.querySelector("[modal-backdrop]");
        if (backdrop) backdrop.remove();
      }
    });
  }

  // Resources (formerly "templates") are managed by skills-resources.js via
  // dedicated endpoints, not this form.

  // ----- Save buttons -----
  function levelLabel(level) {
    if (level === "system") return "system";
    if (level === "org") return "organization";
    return level;
  }

  function saveWarning() {
    if (skillLevel === "org") {
      var who =
        colleagueCount === 1
          ? "1 colleague"
          : colleagueCount + " of your colleagues";
      return (
        "This will replace the org-wide instructions for " +
        skillName +
        ", used by " +
        who +
        ". There is no undo. Continue?"
      );
    }
    return "This will overwrite your version of " + skillName + ". There is no undo. Continue?";
  }

  var saveBtn = document.getElementById("save-btn");

  // Busy state while the save POST navigates (the safety scan itself runs on
  // the worker after the redirect; the list row then shows "Scanning…").
  function setSaveBusy(on) {
    if (!saveBtn) return;
    saveBtn.disabled = on;
    var spinner = document.getElementById("save-btn-spinner");
    var label = document.getElementById("save-btn-label");
    if (spinner) spinner.classList.toggle("hidden", !on);
    if (label) label.textContent = on ? "Saving…" : "Save";
  }

  if (saveBtn) {
    saveBtn.addEventListener("click", function () {
      if (!confirm(saveWarning())) return;
      actionInput.value = "save";
      setSaveBusy(true);
      form.submit();
    });
  }
  // A back-navigation restores the page from bfcache with the button still
  // busy — reset it.
  window.addEventListener("pageshow", function (e) {
    if (e.persisted) setSaveBusy(false);
  });

  var saveAsUserBtn = document.getElementById("save-as-user-btn");
  if (saveAsUserBtn) {
    saveAsUserBtn.addEventListener("click", function () {
      actionInput.value = "save_as_user";
      form.submit();
    });
  }

  var saveAsOrgBtn = document.getElementById("save-as-org-btn");
  if (saveAsOrgBtn) {
    saveAsOrgBtn.addEventListener("click", function () {
      if (!confirm("Save this skill to your organization? Members will see it immediately.")) return;
      actionInput.value = "save_as_org";
      form.submit();
    });
  }

  // Promote: move this personal skill up to the organization (changes its type).
  var promoteBtn = document.getElementById("promote-btn");
  if (promoteBtn) {
    promoteBtn.addEventListener("click", function () {
      if (!confirm("Promote this to an organization skill? Your whole organization will be able to use it, and it will no longer be one of your personal skills.")) return;
      actionInput.value = "promote";
      form.submit();
    });
  }

  // Demote: move this org skill down to your personal skills (changes its type).
  var demoteBtn = document.getElementById("demote-btn");
  if (demoteBtn) {
    demoteBtn.addEventListener("click", function () {
      if (!confirm("Demote this to a personal skill? It will be removed from your organization and only you will have it. There is no undo.")) return;
      actionInput.value = "demote";
      form.submit();
    });
  }

  // Block <Enter> on text inputs from triggering an accidental submit.
  form.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && e.target.tagName === "INPUT") {
      e.preventDefault();
    }
  });

  // ----- Slug override warning -----
  var slugInput = document.getElementById("skill-slug");
  var slugWarning = document.getElementById("slug-override-warning");
  var slugWarningText = document.getElementById("slug-override-text");
  var overrideSlugMap = {};
  try {
    overrideSlugMap = JSON.parse(
      (config && config.getAttribute("data-override-slug-map")) || "{}"
    );
  } catch (e) {
    overrideSlugMap = {};
  }

  function slugify(s) {
    return s.toLowerCase().trim().replace(/[^\w\s-]/g, "").replace(/[\s_]+/g, "-").replace(/^-+|-+$/g, "");
  }

  function checkSlugOverride() {
    if (!slugInput || !slugWarning) return;
    var slug = slugify(slugInput.value);
    var matchedName = overrideSlugMap[slug];
    if (matchedName) {
      slugWarningText.textContent =
        "This slug matches the system skill \u2018" +
        matchedName +
        "\u2019 \u2014 your version will override it by default (for you only, and can be changed).";
      slugWarning.classList.remove("hidden");
    } else {
      slugWarning.classList.add("hidden");
    }
  }

  if (slugInput) {
    slugInput.addEventListener("input", checkSlugOverride);
    checkSlugOverride();
  }

  // ----- Advanced slug editing -----
  // The slug auto-follows the name on save; revealing this input marks the slug
  // as user-customized (via the slug_customize hidden field) so it stops syncing
  // with the name on future renames.
  var slugEditBtn = document.getElementById("skill-slug-edit-btn");
  var slugDisplay = document.getElementById("skill-slug-display");
  var slugHint = document.getElementById("skill-slug-hint");
  var slugCustomize = document.getElementById("skill-slug-customize");

  function markSlugCustomized() {
    if (slugCustomize) slugCustomize.value = "1";
  }

  if (slugEditBtn && slugInput) {
    slugEditBtn.addEventListener("click", function () {
      slugInput.classList.remove("hidden");
      if (slugDisplay) slugDisplay.classList.add("hidden");
      slugEditBtn.classList.add("hidden");
      if (slugHint) slugHint.classList.remove("hidden");
      markSlugCustomized();
      slugInput.focus();
    });
    slugInput.addEventListener("input", markSlugCustomized);
  }

  // ----- Safety-scan status (header pills) -----
  // The three pills are pre-rendered by the template; this only toggles
  // `hidden` and polls the scan-status endpoint while the scan is running.
  // Exposed as window.WilfredSkillScan so skills-resources.js can refresh the
  // pill after a resource write re-queues the scan.
  var scanPillsEl = document.getElementById("skill-scan-pills");
  var scanStatusUrl = scanPillsEl ? scanPillsEl.getAttribute("data-scan-status-url") : "";
  var scanSkillId = scanPillsEl ? scanPillsEl.getAttribute("data-skill-id") : "";
  var scanPolling = false;

  // Same rule as views._scan_pill with is_selected=True (this is the author's
  // own page): usable → none; blocked; pending; otherwise "scan needed".
  function scanPillFor(approved, scanState) {
    if (approved) return "";
    if (scanState === "blocked") return "blocked";
    if (scanState === "pending") return "pending";
    return "needed";
  }

  function applyScanPills(info) {
    if (!scanPillsEl || !info) return;
    var approved = !!info.approved;
    var scanState = typeof info.scan_state === "string" ? info.scan_state : "";
    scanPillsEl.setAttribute("data-scan-state", scanState);
    scanPillsEl.setAttribute("data-approved", approved ? "1" : "0");
    var pill = scanPillFor(approved, scanState);
    scanPillsEl.querySelectorAll(".scan-pill").forEach(function (el) {
      var kind = el.getAttribute("data-pill");
      // The `hidden` attribute, not the class (see skills-list.js).
      el.hidden = kind !== pill;
      if (kind === "blocked" && typeof info.detail === "string" && info.detail) {
        el.setAttribute("title", info.detail);
      }
    });
  }

  function scanIsPending() {
    return (
      !!scanPillsEl &&
      scanPillsEl.getAttribute("data-scan-state") === "pending" &&
      scanPillsEl.getAttribute("data-approved") !== "1"
    );
  }

  function fetchScanStatus() {
    if (!scanStatusUrl || !scanSkillId) return Promise.reject();
    return fetch(scanStatusUrl + "?ids=" + encodeURIComponent(scanSkillId), {
      credentials: "same-origin",
      headers: { "X-Requested-With": "XMLHttpRequest", "Accept": "application/json" },
    })
      .then(function (resp) {
        return resp.json();
      })
      .then(function (data) {
        if (data && data.ok && data.skills && data.skills[scanSkillId]) {
          applyScanPills(data.skills[scanSkillId]);
        }
      });
  }

  function pollScanStatus() {
    if (scanPolling || !scanIsPending()) return;
    scanPolling = true;
    setTimeout(function () {
      fetchScanStatus()
        .then(function () {
          scanPolling = false;
          pollScanStatus();
        })
        .catch(function () {
          scanPolling = false;
        });
    }, 2500);
  }

  // Fetch the verdict right away (a resource write just re-queued the scan),
  // then keep polling while it is pending.
  function refreshScanStatus() {
    fetchScanStatus()
      .then(pollScanStatus)
      .catch(function () {});
  }

  window.WilfredSkillScan = {
    apply: applyScanPills,
    poll: pollScanStatus,
    refresh: refreshScanStatus,
  };

  // ----- Initial render -----
  renderToolChips();
  renderRequiresChips();
  pollScanStatus();

  // Mount CM editors over the instructions/description fields. Template rows are
  // mounted inside renderTemplates so they re-attach on add/remove. These two
  // textareas are named form fields, so mountMarkdownEditor keeps them synced.
  mountMarkdownEditor({
    textarea: document.getElementById("skill-instructions"),
    preview: document.getElementById("instructions-preview"),
    previewBtn: document.getElementById("instructions-preview-btn"),
    readOnly: !editable,
    defaultPreview: !editable,
    minHeight: "14rem",
    maxHeight: "32rem",
  });
  mountMarkdownEditor({
    textarea: document.getElementById("skill-description"),
    preview: document.getElementById("description-preview"),
    previewBtn: document.getElementById("description-preview-btn"),
    readOnly: !editable,
    defaultPreview: !editable,
    maxLength: 1024,
    minHeight: "5rem",
    maxHeight: "20rem",
  });
})();
