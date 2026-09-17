/* Skill resources UI: drag-drop upload, list rows, create/edit/view modal,
 * replace-file, and bulk delete. Talks to the /skills/<id>/resources/* endpoints.
 * Resources are managed independently of the main skill-detail form.
 *
 * The modal is one shared shell that adapts to the resource kind: a WilfredEditor
 * (Write/Preview) for text, an <img> preview for images, a same-origin <iframe>
 * for PDFs (the CSP's object-src 'none' rules out <embed>). On a skill the user
 * can't edit it opens read-only (View + Download only).
 */
(function () {
  "use strict";

  var section = document.getElementById("resources-section");
  if (!section) return;

  var canEdit = section.getAttribute("data-editable") === "1";
  var cap = parseInt(section.getAttribute("data-cap") || "50", 10);
  var uploadUrl = section.getAttribute("data-upload-url");
  var createUrl = section.getAttribute("data-create-url");
  var statusUrl = section.getAttribute("data-status-url");
  var updateTpl = section.getAttribute("data-update-url-tpl");
  var replaceTpl = section.getAttribute("data-replace-url-tpl");
  var deleteTpl = section.getAttribute("data-delete-url-tpl");
  var ID_PLACEHOLDER = "00000000-0000-0000-0000-000000000000";

  var listEl = document.getElementById("resource-list");
  var listHead = document.getElementById("resource-list-head");
  var countEl = document.getElementById("resource-count");
  var selectAll = document.getElementById("resource-select-all");
  var rowTemplate = document.getElementById("resource-row-template");

  var resources = [];
  try {
    var dataEl = document.getElementById("resources-data");
    resources = dataEl ? JSON.parse(dataEl.textContent) : [];
  } catch (e) {
    resources = [];
  }

  function csrf() {
    var el = document.querySelector("#skill-detail-form [name=csrfmiddlewaretoken]");
    return el ? el.value : "";
  }

  function post(url, formData) {
    // Include the token in the body too — this also guarantees a non-empty
    // multipart payload (an empty FormData body trips Django's parser -> 400).
    formData.append("csrfmiddlewaretoken", csrf());
    return fetch(url, {
      method: "POST",
      headers: { "X-CSRFToken": csrf(), "X-Requested-With": "XMLHttpRequest" },
      body: formData,
    }).then(function (r) {
      return r.text().then(function (t) {
        var data = {};
        try {
          data = t ? JSON.parse(t) : {};
        } catch (e) {
          data = {};
        }
        return { ok: r.ok, status: r.status, data: data };
      });
    });
  }

  function urlFor(tpl, id) {
    return tpl.replace(ID_PLACEHOLDER, id);
  }

  // A resource write changes the skill's content hash and re-queues its safety
  // scan; refresh the header pill (skills-detail.js owns it) so the page shows
  // "Scanning…" and then the verdict without a reload.
  function notifySkillScan() {
    if (window.WilfredSkillScan && typeof window.WilfredSkillScan.refresh === "function") {
      window.WilfredSkillScan.refresh();
    }
  }

  function triggerDownload(url) {
    if (!url) return;
    var a = document.createElement("a");
    a.href = url;
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  // ----- Icons & markers -----
  var ICON_TEXT =
    '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.6" d="M9 12h6m-6 4h6m-7 5h8a2 2 0 002-2V7l-5-5H8a2 2 0 00-2 2v15a2 2 0 002 2z"/></svg>';
  var ICON_PDF =
    '<svg class="w-5 h-5 text-fg-danger" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.6" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.4" d="M8.5 14.5h.5a1 1 0 000-2h-.5v3m3-3v3h.6a1 1 0 001-1v-1a1 1 0 00-1-1H11.5m4 0H15v3m0-1.5h.8"/></svg>';
  var ICON_IMAGE =
    '<svg class="w-5 h-5 text-fg-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.6" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M4.5 19.5h15a2.25 2.25 0 002.25-2.25V6.75A2.25 2.25 0 0019.5 4.5h-15A2.25 2.25 0 002.25 6.75v10.5A2.25 2.25 0 004.5 19.5z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.6" d="M8.25 9a.75.75 0 100-1.5.75.75 0 000 1.5z"/></svg>';

  function iconFor(fileType) {
    if (fileType === "pdf") return ICON_PDF;
    if (fileType === "image") return ICON_IMAGE;
    return ICON_TEXT;
  }

  function iconChipClass(fileType) {
    // Copper chip for text/image; danger tint for PDF (matches the design).
    if (fileType === "pdf") return "bg-danger-soft border border-danger-subtle text-fg-danger";
    return "bg-accent-soft border border-accent-subtle text-fg-accent";
  }

  var SPINNER =
    '<svg class="w-4 h-4 animate-spin text-body" viewBox="0 0 24 24" fill="none"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path></svg>';
  var CHECK =
    '<svg class="w-4 h-4 text-fg-success" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>';
  var WARN =
    '<svg class="w-4 h-4 text-fg-warning" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';
  var DANGER =
    '<svg class="w-4 h-4 text-fg-danger" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2 4 5v6c0 5 3.4 8.5 8 10 4.6-1.5 8-5 8-10V5l-8-3z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>';

  function statusMarker(status) {
    if (status === "ready") return { html: CHECK, title: "Ready" };
    if (status === "quarantined")
      return { html: DANGER, title: "Quarantined — not readable by the assistant" };
    if (status === "scan_failed")
      return { html: WARN, title: "Processing failed" };
    return { html: SPINNER, title: "Processing…" };
  }

  function pill(text, tone) {
    var cls =
      tone === "danger"
        ? "bg-danger-soft text-fg-danger-strong border-danger-subtle"
        : "bg-neutral-secondary-soft text-body border-default";
    return (
      '<span class="inline-flex items-center px-1.5 py-0.5 rounded-full border text-[10px] font-medium ' +
      cls +
      '">' +
      text +
      "</span>"
    );
  }

  function piiPills(r) {
    if (r.is_quarantined) return pill("Quarantined", "danger");
    var p = r.pii || {};
    if (p.special || p.criminal) return pill("Sensitive data", "danger");
    if (p.ordinary) return pill("Personal data", "neutral");
    return "";
  }

  function isTemplate(r) {
    return r.kind === "template";
  }

  // ----- Row menu (contents depend on can-edit) -----
  var MENU_ITEM_CLS =
    "resource-menu-item block w-full text-left px-3 py-1.5 text-sm text-body hover:bg-neutral-tertiary";

  function buildMenu(r, menu) {
    menu.innerHTML = "";
    function item(action, label, danger) {
      var b = document.createElement("button");
      b.type = "button";
      b.setAttribute("data-action", action);
      b.className = danger
        ? "resource-menu-item block w-full text-left px-3 py-1.5 text-sm text-fg-danger hover:bg-danger-soft"
        : MENU_ITEM_CLS;
      b.textContent = label;
      menu.appendChild(b);
    }
    if (canEdit) {
      item("edit", "View / edit");
      item("rename", "Rename");
      if (r.original_filename) item("replace", "Replace file");
      if (r.download_url) item("download", "Download");
      item("delete", "Delete", true);
    } else {
      item("edit", "View");
      if (r.download_url) item("download", "Download");
    }
  }

  // ----- Rendering -----
  function renderRow(r) {
    var node = rowTemplate.content.firstElementChild.cloneNode(true);
    node.setAttribute("data-resource-id", r.id);
    var mk = statusMarker(r.status);
    var statusEl = node.querySelector(".resource-status");
    statusEl.innerHTML = mk.html;
    statusEl.title = mk.title;
    node.querySelector(".resource-icon").innerHTML = iconFor(r.file_type);
    node.querySelector(".resource-name").textContent = r.name;

    var subEl = node.querySelector(".resource-subline");
    if (r.original_filename && r.original_filename !== r.name) {
      subEl.textContent = r.original_filename;
      subEl.classList.remove("hidden");
    } else {
      subEl.remove();
    }

    var errEl = node.querySelector(".resource-error");
    if (r.error) errEl.textContent = r.error;
    else errEl.remove();

    var tpl = node.querySelector(".resource-template-pill");
    if (isTemplate(r)) tpl.classList.remove("hidden");
    else tpl.remove();

    node.querySelector(".resource-pii").innerHTML = piiPills(r);
    node.querySelector(".resource-time").textContent = r.updated_display || "";

    var checkbox = node.querySelector(".resource-checkbox");
    if (!canEdit) checkbox.remove();
    else checkbox.addEventListener("change", refreshBulkBar);

    // Whole row opens the modal (view or edit); clicks on the checkbox/menu
    // are handled separately and must not also open it.
    node.addEventListener("click", function (e) {
      if (e.target.closest(".resource-menu-wrap") || e.target.closest(".resource-checkbox"))
        return;
      openModal(r);
    });

    var menuBtn = node.querySelector(".resource-menu-btn");
    var menu = node.querySelector(".resource-menu");
    buildMenu(r, menu);
    menuBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      var willOpen = menu.classList.contains("hidden");
      closeAllMenus(menu);
      if (willOpen) {
        // Flip upward when there isn't room below (last rows sit near the
        // action bar / viewport bottom).
        var rect = menuBtn.getBoundingClientRect();
        if (window.innerHeight - rect.bottom < 200) {
          menu.style.top = "auto";
          menu.style.bottom = "100%";
          menu.style.marginTop = "0";
          menu.style.marginBottom = "4px";
        } else {
          menu.style.bottom = "auto";
          menu.style.top = "100%";
          menu.style.marginBottom = "0";
          menu.style.marginTop = "4px";
        }
      }
      menu.classList.toggle("hidden");
    });
    menu.addEventListener("click", function (e) {
      var item = e.target.closest("[data-action]");
      if (!item) return;
      e.stopPropagation();
      menu.classList.add("hidden");
      var action = item.getAttribute("data-action");
      if (action === "delete") deleteResource(r);
      else if (action === "download") triggerDownload(r.download_url);
      else if (action === "replace") startReplace(r);
      else openModal(r); // edit / rename / view
    });
    return node;
  }

  function renderAll() {
    listEl.innerHTML = "";
    if (!resources.length) {
      if (listHead) listHead.classList.add("hidden");
      var empty = document.createElement("p");
      empty.className = "px-4 py-6 text-sm text-body italic";
      empty.textContent = canEdit
        ? "No resources yet. Upload a file or create one."
        : "No resources.";
      listEl.appendChild(empty);
      refreshBulkBar();
      return;
    }
    var sorted = resources.slice().sort(function (a, b) {
      return a.name.localeCompare(b.name);
    });
    sorted.forEach(function (r) {
      listEl.appendChild(renderRow(r));
    });
    if (listHead) {
      listHead.classList.remove("hidden");
      listHead.classList.add("flex");
    }
    if (countEl) {
      var n = resources.length;
      var m = resources.filter(isTemplate).length;
      var txt = n === 1 ? "1 resource" : n + " resources";
      if (m > 0) txt += " · " + m + (m === 1 ? " template" : " templates");
      countEl.textContent = txt;
    }
    if (selectAll) selectAll.checked = false;
    refreshBulkBar();
  }

  function upsert(r) {
    var i = resources.findIndex(function (x) {
      return x.id === r.id;
    });
    if (i === -1) resources.push(r);
    else resources[i] = r;
    renderAll();
  }

  function removeLocal(id) {
    resources = resources.filter(function (x) {
      return x.id !== id;
    });
    renderAll();
  }

  // ----- Status polling (uploads/replacements process on the worker) -----
  var TERMINAL = ["ready", "quarantined", "scan_failed"];
  var polling = false;

  function hasPending() {
    return resources.some(function (r) {
      return TERMINAL.indexOf(r.status) === -1;
    });
  }

  function pollStatus() {
    if (polling || !statusUrl || !hasPending()) return;
    polling = true;
    setTimeout(function () {
      fetch(statusUrl, { headers: { "X-Requested-With": "XMLHttpRequest" } })
        .then(function (r) {
          return r.json();
        })
        .then(function (data) {
          polling = false;
          if (data && data.ok && Array.isArray(data.resources)) {
            resources = data.resources;
            renderAll();
            // The poll also carries the skill's own scan verdict (uploads
            // re-run the approval gate on the worker once they land).
            if (data.skill && window.WilfredSkillScan) {
              window.WilfredSkillScan.apply(data.skill);
              window.WilfredSkillScan.poll();
            }
            pollStatus();
          }
        })
        .catch(function () {
          polling = false;
        });
    }, 2500);
  }

  function closeAllMenus(except) {
    document.querySelectorAll(".resource-menu").forEach(function (m) {
      if (m !== except) m.classList.add("hidden");
    });
  }
  document.addEventListener("click", function () {
    closeAllMenus(null);
  });

  // ----- Bulk selection -----
  var bulkBar = document.getElementById("resource-bulk-bar");
  var bulkCount = document.getElementById("resource-bulk-count");
  var bulkDelete = document.getElementById("resource-bulk-delete");

  function selectedIds() {
    return Array.prototype.map.call(
      document.querySelectorAll(".resource-checkbox:checked"),
      function (cb) {
        return cb.closest(".resource-row").getAttribute("data-resource-id");
      }
    );
  }

  function refreshBulkBar() {
    if (!bulkBar) return;
    var n = selectedIds().length;
    if (n === 0) {
      bulkBar.classList.add("hidden");
      bulkBar.classList.remove("flex");
    } else {
      bulkBar.classList.remove("hidden");
      bulkBar.classList.add("flex");
      bulkCount.textContent = n === 1 ? "1 selected" : n + " selected";
    }
  }

  if (selectAll) {
    selectAll.addEventListener("change", function () {
      document.querySelectorAll(".resource-checkbox").forEach(function (cb) {
        cb.checked = selectAll.checked;
      });
      refreshBulkBar();
    });
  }

  if (bulkDelete) {
    bulkDelete.addEventListener("click", function () {
      var ids = selectedIds();
      if (!ids.length) return;
      if (!confirm("Delete " + ids.length + " resource(s)? This can't be undone."))
        return;
      Promise.all(
        ids.map(function (id) {
          return post(urlFor(deleteTpl, id), new FormData());
        })
      ).then(function () {
        ids.forEach(removeLocal);
        notifySkillScan();
      });
    });
  }

  function deleteResource(r) {
    if (!confirm("Delete “" + r.name + "”? This can't be undone.")) return;
    post(urlFor(deleteTpl, r.id), new FormData()).then(function (res) {
      if (res.data && res.data.ok) {
        removeLocal(r.id);
        notifySkillScan();
      }
    });
  }

  // ----- Upload -----
  var dropzone = document.getElementById("resource-dropzone");
  var fileInput = document.getElementById("resource-file-input");

  function uploadFiles(files) {
    if (!files || !files.length) return;
    if (resources.length + files.length > cap) {
      alert("This skill can hold at most " + cap + " resources.");
      return;
    }
    var fd = new FormData();
    Array.prototype.forEach.call(files, function (f) {
      fd.append("file", f);
    });
    if (dropzone) dropzone.classList.add("opacity-60", "pointer-events-none");
    post(uploadUrl, fd)
      .then(function (res) {
        if (dropzone) dropzone.classList.remove("opacity-60", "pointer-events-none");
        if (!res.data) return;
        (res.data.resources || []).forEach(upsert);
        pollStatus();
        notifySkillScan();
        if (res.data.errors && res.data.errors.length) {
          alert(res.data.errors.join("\n"));
        }
      })
      .catch(function () {
        if (dropzone) dropzone.classList.remove("opacity-60", "pointer-events-none");
        alert("Upload failed. Please try again.");
      });
  }

  if (fileInput) {
    fileInput.addEventListener("change", function () {
      uploadFiles(fileInput.files);
      fileInput.value = "";
    });
  }
  if (dropzone) {
    ["dragenter", "dragover"].forEach(function (ev) {
      dropzone.addEventListener(ev, function (e) {
        e.preventDefault();
        dropzone.classList.add("border-brand", "bg-brand-softer");
      });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      dropzone.addEventListener(ev, function (e) {
        e.preventDefault();
        dropzone.classList.remove("border-brand", "bg-brand-softer");
      });
    });
    dropzone.addEventListener("drop", function (e) {
      if (e.dataTransfer && e.dataTransfer.files) uploadFiles(e.dataTransfer.files);
    });
  }

  // ----- Create / edit / view modal -----
  var modal = document.getElementById("resource-modal");
  var modalIcon = document.getElementById("resource-modal-icon");
  var modalTitle = document.getElementById("resource-modal-title");
  var modalNameRow = document.getElementById("resource-modal-name-row");
  var modalName = document.getElementById("resource-modal-name");
  var modalContentWrap = document.getElementById("resource-modal-content-wrap");
  var modalEditorMount = document.getElementById("resource-modal-editor");
  var modalImageWrap = document.getElementById("resource-modal-image-wrap");
  var modalImage = document.getElementById("resource-modal-image");
  var modalPdfWrap = document.getElementById("resource-modal-pdf-wrap");
  var modalPdf = document.getElementById("resource-modal-pdf");
  var modalTemplateRow = document.getElementById("resource-modal-template-row");
  var modalTemplate = document.getElementById("resource-modal-template");
  var modalDownload = document.getElementById("resource-modal-download");
  var modalReplace = document.getElementById("resource-modal-replace");
  var modalReplaceInput = document.getElementById("resource-modal-replace-input");
  var modalError = document.getElementById("resource-modal-error");
  var modalSave = document.getElementById("resource-modal-save");
  var modalSaveLabel = modal ? modal.querySelector(".resource-modal-save-label") : null;
  var modalSaveSpinner = modal ? modal.querySelector(".resource-modal-save-spinner") : null;

  var editingId = null; // null => create
  var editingType = "text";
  var modalEditor = null;

  function showModal() {
    modal.classList.remove("hidden");
    modal.classList.add("flex");
    modal.setAttribute("aria-hidden", "false");
  }
  function hideModal() {
    modal.classList.add("hidden");
    modal.classList.remove("flex");
    modal.setAttribute("aria-hidden", "true");
    if (modalEditor) {
      modalEditor.destroy();
      modalEditor = null;
    }
    // Release any large preview source.
    if (modalImage) modalImage.removeAttribute("src");
    if (modalPdf) modalPdf.removeAttribute("src");
  }

  function setModalError(msg) {
    if (!modalError) return;
    if (msg) {
      modalError.textContent = msg;
      modalError.classList.remove("hidden");
    } else {
      modalError.classList.add("hidden");
    }
  }

  function show(el, on) {
    if (!el) return;
    el.classList.toggle("hidden", !on);
  }

  function openModal(r) {
    if (!modal) return;
    setModalError("");
    editingId = r ? r.id : null;
    editingType = r ? r.file_type : "text";
    var creating = !r;
    var viewing = !!r && !canEdit;
    var editing = !!r && canEdit;

    // Header icon + title
    modalIcon.className =
      "inline-flex items-center justify-center w-8 h-8 rounded-base shrink-0 " +
      iconChipClass(editingType);
    modalIcon.innerHTML = iconFor(editingType);
    modalTitle.textContent = creating
      ? "Create resource"
      : viewing
      ? r.name
      : "Edit resource";

    // Reset body sections
    show(modalNameRow, !viewing); // view mode: the title stands in for the name
    show(modalContentWrap, false);
    show(modalImageWrap, false);
    show(modalPdfWrap, false);
    if (modalEditor) {
      modalEditor.destroy();
      modalEditor = null;
    }
    modalName.value = r ? r.name : "";

    // Footer defaults
    show(modalTemplateRow, !viewing);
    if (modalTemplate) modalTemplate.checked = !!r && isTemplate(r);
    show(modalSave, !viewing);
    show(modalReplace, editing && !!r && !!r.original_filename);
    if (modalDownload) {
      if (r && r.download_url) {
        modalDownload.href = r.download_url;
        show(modalDownload, true);
      } else {
        show(modalDownload, false);
      }
    }

    // Body per file type
    if (editingType === "image") {
      show(modalImageWrap, true);
      if (r && r.file_url) modalImage.src = r.file_url;
    } else if (editingType === "pdf") {
      show(modalPdfWrap, true);
      if (r && r.file_url) modalPdf.src = r.file_url;
    } else {
      // text — typed text is editable; an uploaded text file (has original_filename)
      // has no inline content, so offer download instead of an empty editor.
      var hasInlineText = creating || (r && r.editable_content);
      if (hasInlineText) {
        show(modalContentWrap, true);
        mountEditor(r ? r.content || "" : "", viewing);
      } else {
        // Uploaded text file: nothing editable here.
        show(modalContentWrap, false);
        show(modalTemplateRow, editing);
        setModalError("");
      }
    }

    showModal();
    if (!viewing) modalName.focus();
  }

  function mountEditor(value, readOnly) {
    modalEditorMount.innerHTML = "";
    if (window.WilfredEditor) {
      modalEditor = window.WilfredEditor.create(modalEditorMount, {
        value: value,
        toolbar: true,
        preview: true,
        readOnly: !!readOnly,
        minHeight: "14rem",
        maxHeight: "26rem",
        placeholder: "Resource content (markdown supported)",
      });
    } else {
      var ta = document.createElement("textarea");
      ta.className = "wf-input text-heading text-sm rounded-base block w-full px-3 py-2.5 font-mono";
      ta.rows = 10;
      ta.value = value;
      ta.readOnly = !!readOnly;
      modalEditorMount.appendChild(ta);
      modalEditor = {
        getValue: function () {
          return ta.value;
        },
        destroy: function () {},
      };
    }
  }

  var createBtn = document.getElementById("create-resource-btn");
  if (createBtn) {
    createBtn.addEventListener("click", function () {
      if (resources.length >= cap) {
        alert("This skill can hold at most " + cap + " resources.");
        return;
      }
      openModal(null);
    });
  }

  if (modal) {
    modal.querySelectorAll(".resource-modal-close").forEach(function (el) {
      el.addEventListener("click", hideModal);
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && !modal.classList.contains("hidden")) hideModal();
    });
  }

  function setSaving(on) {
    if (!modalSave) return;
    modalSave.disabled = on;
    if (modalSaveSpinner) modalSaveSpinner.classList.toggle("hidden", !on);
    if (modalSaveLabel) modalSaveLabel.textContent = on ? "Saving…" : "Save";
  }

  if (modalSave) {
    modalSave.addEventListener("click", function () {
      var name = (modalName.value || "").trim();
      if (!name) {
        setModalError("Please give the resource a name.");
        return;
      }
      var fd = new FormData();
      fd.append("name", name);
      fd.append("is_template", modalTemplate && modalTemplate.checked ? "1" : "0");
      // Only text resources carry editable content.
      if (editingType === "text" && modalEditor && !modalContentWrap.classList.contains("hidden")) {
        fd.append("content", modalEditor.getValue());
      }
      setSaving(true);
      setModalError("");
      var url = editingId ? urlFor(updateTpl, editingId) : createUrl;
      post(url, fd)
        .then(function (res) {
          setSaving(false);
          if (res.data && res.data.ok) {
            upsert(res.data.resource);
            hideModal();
            notifySkillScan();
          } else {
            var err = res.data && res.data.error;
            setModalError(
              err === "duplicate_name"
                ? "A resource with that name already exists in this skill."
                : err === "limit"
                ? "This skill has reached its resource limit."
                : "Couldn't save the resource."
            );
          }
        })
        .catch(function () {
          setSaving(false);
          setModalError("Couldn't save the resource.");
        });
    });
  }

  // ----- Replace file -----
  var replacingId = null;

  function startReplace(r) {
    if (!modalReplaceInput) return;
    replacingId = r.id;
    modalReplaceInput.value = "";
    modalReplaceInput.click();
  }

  if (modalReplace) {
    modalReplace.addEventListener("click", function () {
      if (editingId) startReplace({ id: editingId });
    });
  }

  if (modalReplaceInput) {
    modalReplaceInput.addEventListener("change", function () {
      var f = modalReplaceInput.files && modalReplaceInput.files[0];
      if (!f || !replacingId) return;
      var id = replacingId;
      var fd = new FormData();
      fd.append("file", f);
      setSaving(true);
      post(urlFor(replaceTpl, id), fd)
        .then(function (res) {
          setSaving(false);
          if (res.data && res.data.ok) {
            upsert(res.data.resource);
            pollStatus();
            hideModal();
            notifySkillScan();
          } else {
            var err = res.data && res.data.error;
            setModalError(
              err === "too_large"
                ? "That file is too large."
                : err === "unsupported_type"
                ? "That file type isn't supported."
                : "Couldn't replace the file."
            );
          }
        })
        .catch(function () {
          setSaving(false);
          setModalError("Couldn't replace the file.");
        });
    });
  }

  // ----- Init -----
  renderAll();
  pollStatus(); // in case a resource is still processing when the page loads
})();
