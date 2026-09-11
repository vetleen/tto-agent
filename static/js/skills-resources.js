/* Skill resources UI: drag-drop upload, list rows, create/edit modal, and
 * bulk delete. Talks to the /skills/<id>/resources/* endpoints. Resources are
 * managed independently of the main skill-detail form. */
(function () {
  "use strict";

  var section = document.getElementById("resources-section");
  if (!section) return;

  var editable = section.getAttribute("data-editable") === "1";
  var cap = parseInt(section.getAttribute("data-cap") || "50", 10);
  var uploadUrl = section.getAttribute("data-upload-url");
  var createUrl = section.getAttribute("data-create-url");
  var statusUrl = section.getAttribute("data-status-url");
  var updateTpl = section.getAttribute("data-update-url-tpl");
  var deleteTpl = section.getAttribute("data-delete-url-tpl");
  var ID_PLACEHOLDER = "00000000-0000-0000-0000-000000000000";

  var listEl = document.getElementById("resource-list");
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
    return fetch(url, {
      method: "POST",
      headers: { "X-CSRFToken": csrf(), "X-Requested-With": "XMLHttpRequest" },
      body: formData,
    }).then(function (r) {
      return r.json().then(function (data) {
        return { ok: r.ok, status: r.status, data: data };
      });
    });
  }

  function urlFor(tpl, id) {
    return tpl.replace(ID_PLACEHOLDER, id);
  }

  // ----- Icons & markers -----
  var ICON_TEXT =
    '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.6" d="M9 12h6m-6 4h6m-7 5h8a2 2 0 002-2V7l-5-5H8a2 2 0 00-2 2v15a2 2 0 002 2z"/></svg>';
  // Reused from the canvas export dropdown (chat.html) so the PDF glyph matches.
  var ICON_PDF =
    '<svg class="w-5 h-5 text-fg-danger" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.6" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.4" d="M8.5 14.5h.5a1 1 0 000-2h-.5v3m3-3v3h.6a1 1 0 001-1v-1a1 1 0 00-1-1H11.5m4 0H15v3m0-1.5h.8"/></svg>';
  var ICON_IMAGE =
    '<svg class="w-5 h-5 text-fg-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.6" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M4.5 19.5h15a2.25 2.25 0 002.25-2.25V6.75A2.25 2.25 0 0019.5 4.5h-15A2.25 2.25 0 002.25 6.75v10.5A2.25 2.25 0 004.5 19.5z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.6" d="M8.25 9a.75.75 0 100-1.5.75.75 0 000 1.5z"/></svg>';

  function iconFor(fileType) {
    if (fileType === "pdf") return ICON_PDF;
    if (fileType === "image") return ICON_IMAGE;
    return ICON_TEXT;
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
    var errEl = node.querySelector(".resource-error");
    if (r.error) errEl.textContent = r.error;
    else errEl.remove();
    node.querySelector(".resource-pii").innerHTML = piiPills(r);
    node.querySelector(".resource-time").textContent = r.updated_display || "";

    var checkbox = node.querySelector(".resource-checkbox");
    var menuWrap = node.querySelector(".resource-menu-wrap");
    if (!editable) {
      checkbox.remove();
      menuWrap.remove();
      return node;
    }
    checkbox.addEventListener("change", refreshBulkBar);

    var menuBtn = node.querySelector(".resource-menu-btn");
    var menu = node.querySelector(".resource-menu");
    var editItem = menu.querySelector('[data-action="edit"]');
    // Only typed text resources are content-editable; others get rename only.
    if (!r.editable_content) editItem.textContent = "Rename";
    menuBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      closeAllMenus(menu);
      menu.classList.toggle("hidden");
    });
    menu.querySelectorAll("[data-action]").forEach(function (item) {
      item.addEventListener("click", function () {
        menu.classList.add("hidden");
        var action = item.getAttribute("data-action");
        if (action === "delete") deleteResource(r);
        else openModal(r); // edit or rename both open the modal
      });
    });
    return node;
  }

  function renderAll() {
    listEl.innerHTML = "";
    if (!resources.length) {
      var empty = document.createElement("p");
      empty.className = "px-4 py-6 text-sm text-body italic";
      empty.textContent = editable
        ? "No resources yet. Upload a file or create one."
        : "No resources.";
      listEl.appendChild(empty);
      refreshBulkBar();
      return;
    }
    resources
      .slice()
      .sort(function (a, b) {
        return a.name.localeCompare(b.name);
      })
      .forEach(function (r) {
        listEl.appendChild(renderRow(r));
      });
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

  // ----- Status polling (uploads process on the worker) -----
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
    return Array.prototype.map
      .call(
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
      });
    });
  }

  function deleteResource(r) {
    if (!confirm("Delete “" + r.name + "”? This can't be undone.")) return;
    post(urlFor(deleteTpl, r.id), new FormData()).then(function (res) {
      if (res.data && res.data.ok) removeLocal(r.id);
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

  // ----- Create / edit modal -----
  var modal = document.getElementById("resource-modal");
  var modalTitle = document.getElementById("resource-modal-title");
  var modalName = document.getElementById("resource-modal-name");
  var modalContentWrap = document.getElementById("resource-modal-content-wrap");
  var modalEditorMount = document.getElementById("resource-modal-editor");
  var modalFileNote = document.getElementById("resource-modal-file-note");
  var modalFileIcon = document.getElementById("resource-modal-file-icon");
  var modalFileType = document.getElementById("resource-modal-file-type");
  var modalError = document.getElementById("resource-modal-error");
  var modalSave = document.getElementById("resource-modal-save");
  var modalSaveLabel = modal ? modal.querySelector(".resource-modal-save-label") : null;
  var modalSaveSpinner = modal ? modal.querySelector(".resource-modal-save-spinner") : null;

  var editingId = null; // null => create
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

  function openModal(r) {
    if (!modal) return;
    setModalError("");
    editingId = r ? r.id : null;
    var showEditor = !r || r.editable_content;
    modalTitle.textContent = r ? (showEditor ? "Edit resource" : "Rename resource") : "Create resource";
    modalName.value = r ? r.name : "";

    if (showEditor) {
      modalContentWrap.classList.remove("hidden");
      modalFileNote.classList.add("hidden");
      modalFileNote.classList.remove("flex");
      modalEditorMount.innerHTML = "";
      if (window.WilfredEditor) {
        modalEditor = window.WilfredEditor.create(modalEditorMount, {
          value: r ? r.content || "" : "",
          toolbar: true,
          minHeight: "12rem",
          maxHeight: "26rem",
          placeholder: "Resource content (markdown supported)",
        });
      } else {
        var ta = document.createElement("textarea");
        ta.className = "wf-input text-heading text-sm rounded-base block w-full px-3 py-2.5 font-mono";
        ta.rows = 10;
        ta.value = r ? r.content || "" : "";
        modalEditorMount.appendChild(ta);
        modalEditor = { getValue: function () { return ta.value; }, destroy: function () {} };
      }
    } else {
      modalContentWrap.classList.add("hidden");
      modalFileNote.classList.remove("hidden");
      modalFileNote.classList.add("flex");
      modalFileIcon.innerHTML = iconFor(r.file_type);
      modalFileType.textContent = r.file_type === "pdf" ? "PDF" : r.file_type;
    }
    showModal();
    modalName.focus();
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
      if (modalEditor && modalContentWrap && !modalContentWrap.classList.contains("hidden")) {
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

  // ----- Init -----
  renderAll();
  pollStatus(); // in case a resource is still processing when the page loads
})();
