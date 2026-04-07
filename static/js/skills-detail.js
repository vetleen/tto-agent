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
  var templatesInput = document.getElementById("templates-json");
  var actionInput = document.getElementById("skill-form-action");

  var toolChipsEl = document.getElementById("tool-chips");
  var templateListEl = document.getElementById("template-list");

  // ----- State -----
  var toolNames = [];
  try {
    toolNames = JSON.parse(toolNamesInput.value || "[]");
  } catch (e) {
    toolNames = [];
  }

  var templates = [];
  try {
    templates = JSON.parse(templatesInput.value || "[]");
  } catch (e) {
    templates = [];
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

  // Wires a toggle button (or none, for read-only) to swap between a textarea
  // and a rendered markdown preview div. If `btn` is null, the field is locked
  // to preview mode (textarea hidden, no toggle).
  function setupMarkdownToggle(btn, textarea, preview, startInPreview) {
    if (!textarea || !preview) return;
    var mode = !!startInPreview;
    function apply() {
      if (mode) {
        preview.innerHTML = renderMarkdown(textarea.value);
        preview.classList.remove("hidden");
        textarea.classList.add("hidden");
        if (btn) {
          btn.innerHTML = "<span>Edit</span>" + PEN_ICON;
          btn.title = "Toggle editing";
        }
      } else {
        preview.classList.add("hidden");
        textarea.classList.remove("hidden");
        if (btn) {
          btn.innerHTML = "<span>Preview</span>" + EYE_ICON;
          btn.title = "Toggle preview";
        }
      }
    }
    if (btn) {
      btn.addEventListener("click", function () {
        mode = !mode;
        apply();
      });
    }
    apply();
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

  if (openToolPickerBtn) {
    openToolPickerBtn.addEventListener("click", function () {
      // Pre-check existing tools.
      document.querySelectorAll(".tool-picker-checkbox").forEach(function (cb) {
        cb.checked = toolNames.indexOf(cb.getAttribute("data-tool-name")) !== -1;
      });
    });
  }

  if (applyToolPickerBtn) {
    applyToolPickerBtn.addEventListener("click", function () {
      var picked = [];
      document.querySelectorAll(".tool-picker-checkbox:checked").forEach(function (cb) {
        picked.push(cb.getAttribute("data-tool-name"));
      });
      toolNames = picked;
      syncToolNamesInput();
      renderToolChips();
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

  // ----- Templates UI -----
  function renderTemplates() {
    templateListEl.innerHTML = "";
    if (!templates.length) {
      var empty = document.createElement("p");
      empty.className = "text-xs text-body italic";
      empty.textContent = "No templates yet.";
      templateListEl.appendChild(empty);
      return;
    }
    var tmpl = document.getElementById("template-row-template");
    templates.forEach(function (entry, idx) {
      var node = tmpl.content.firstElementChild.cloneNode(true);
      node.setAttribute("data-template-id", entry.id || "");
      var nameInput = node.querySelector(".template-name");
      var contentInput = node.querySelector(".template-content");
      var contentPreview = node.querySelector(".template-content-preview");
      var previewToggleBtn = node.querySelector(".template-preview-btn");
      var removeBtn = node.querySelector(".remove-template-btn");
      nameInput.value = entry.name || "";
      contentInput.value = entry.content || "";
      if (!editable) {
        nameInput.setAttribute("readonly", "");
        contentInput.setAttribute("readonly", "");
        removeBtn.remove();
        if (previewToggleBtn) previewToggleBtn.remove();
      } else {
        nameInput.addEventListener("input", function () {
          templates[idx].name = nameInput.value;
          syncTemplatesInput();
        });
        contentInput.addEventListener("input", function () {
          templates[idx].content = contentInput.value;
          syncTemplatesInput();
        });
        removeBtn.addEventListener("click", function () {
          templates.splice(idx, 1);
          syncTemplatesInput();
          renderTemplates();
        });
      }
      templateListEl.appendChild(node);
      setupMarkdownToggle(
        editable ? previewToggleBtn : null,
        contentInput,
        contentPreview,
        !editable
      );
    });
  }

  function syncTemplatesInput() {
    templatesInput.value = JSON.stringify(templates);
  }

  var addTemplateBtn = document.getElementById("add-template-btn");
  if (addTemplateBtn) {
    addTemplateBtn.addEventListener("click", function () {
      templates.push({ id: null, name: "", content: "" });
      syncTemplatesInput();
      renderTemplates();
    });
  }

  // ----- Save buttons -----
  function levelLabel(level) {
    if (level === "system") return "built-in";
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
  if (saveBtn) {
    saveBtn.addEventListener("click", function () {
      if (!confirm(saveWarning())) return;
      actionInput.value = "save";
      form.submit();
    });
  }

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

  // Block <Enter> on text inputs from triggering an accidental submit.
  form.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && e.target.tagName === "INPUT") {
      e.preventDefault();
    }
  });

  // ----- Initial render -----
  renderToolChips();
  renderTemplates();

  // Set up markdown toggles for the static instructions/description fields
  // (Template rows are wired up inside renderTemplates so they re-attach on add/remove.)
  setupMarkdownToggle(
    document.getElementById("instructions-preview-btn"),
    document.getElementById("skill-instructions"),
    document.getElementById("instructions-preview"),
    !editable
  );
  setupMarkdownToggle(
    document.getElementById("description-preview-btn"),
    document.getElementById("skill-description"),
    document.getElementById("description-preview"),
    !editable
  );
})();
