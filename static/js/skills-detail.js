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

  // Mounts a CodeMirror markdown editor over a textarea, keeping the (hidden)
  // textarea synced so the Django form POST still submits its value. Returns the
  // editor adapter (or null if the bundle failed to load — in which case the
  // plain textarea is left visible as a graceful fallback). An optional preview
  // button toggles between the editor and a rendered-markdown div.
  function mountMarkdownEditor(o) {
    var textarea = o.textarea;
    if (!textarea) return null;
    if (!window.WilfredEditor) return null; // fallback: leave the textarea visible

    var preview = o.preview;
    var wrapper = document.createElement("div");
    // Slim wrapper: the editor component renders its own toolbar/accent chrome, so
    // we drop the redundant border/shadow and just provide the field background and
    // rounded clipping. (The section card around it already supplies a border.)
    wrapper.className = "cm-field bg-neutral-secondary-medium rounded-base overflow-hidden";
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
      var inPreview = false;
      o.previewBtn.addEventListener("click", function () {
        inPreview = !inPreview;
        if (inPreview) {
          preview.innerHTML = renderMarkdown(ed.getValue());
          preview.classList.remove("hidden");
          wrapper.classList.add("hidden");
          o.previewBtn.innerHTML = "<span>Edit</span>" + PEN_ICON;
          o.previewBtn.title = "Toggle editing";
        } else {
          preview.classList.add("hidden");
          wrapper.classList.remove("hidden");
          o.previewBtn.innerHTML = "<span>Preview</span>" + EYE_ICON;
          o.previewBtn.title = "Toggle preview";
        }
      });
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
  var templateEditors = [];

  function renderTemplates() {
    // Tear down CM instances from the previous render before clearing the DOM.
    templateEditors.forEach(function (ed) {
      if (ed) ed.destroy();
    });
    templateEditors = [];
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
        removeBtn.remove();
        if (previewToggleBtn) previewToggleBtn.remove();
      } else {
        nameInput.addEventListener("input", function () {
          templates[idx].name = nameInput.value;
          syncTemplatesInput();
        });
        removeBtn.addEventListener("click", function () {
          templates.splice(idx, 1);
          syncTemplatesInput();
          renderTemplates();
        });
      }
      templateListEl.appendChild(node);
      var ed = mountMarkdownEditor({
        textarea: contentInput,
        preview: contentPreview,
        previewBtn: editable ? previewToggleBtn : null,
        readOnly: !editable,
        minHeight: "9rem",
        maxHeight: "24rem",
        onChange: function (val) {
          templates[idx].content = val;
          syncTemplatesInput();
        },
      });
      if (!ed && editable) {
        // Fallback (no editor bundle): keep the plain textarea wired up.
        contentInput.addEventListener("input", function () {
          templates[idx].content = contentInput.value;
          syncTemplatesInput();
        });
      }
      templateEditors.push(ed);
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

  // ----- Initial render -----
  renderToolChips();
  renderTemplates();

  // Mount CM editors over the instructions/description fields. Template rows are
  // mounted inside renderTemplates so they re-attach on add/remove. These two
  // textareas are named form fields, so mountMarkdownEditor keeps them synced.
  mountMarkdownEditor({
    textarea: document.getElementById("skill-instructions"),
    preview: document.getElementById("instructions-preview"),
    previewBtn: document.getElementById("instructions-preview-btn"),
    readOnly: !editable,
    minHeight: "14rem",
    maxHeight: "32rem",
  });
  mountMarkdownEditor({
    textarea: document.getElementById("skill-description"),
    preview: document.getElementById("description-preview"),
    previewBtn: document.getElementById("description-preview-btn"),
    readOnly: !editable,
    maxLength: 1024,
    minHeight: "5rem",
    maxHeight: "20rem",
  });
})();
