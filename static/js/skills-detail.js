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
        removeBtn.classList.add("hidden");
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
      var removeBtn = node.querySelector(".remove-template-btn");
      nameInput.value = entry.name || "";
      contentInput.value = entry.content || "";
      if (!editable) {
        nameInput.setAttribute("readonly", "");
        contentInput.setAttribute("readonly", "");
        removeBtn.classList.add("hidden");
      }
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
      templateListEl.appendChild(node);
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
})();
