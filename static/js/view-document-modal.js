/**
 * View / edit document modal. One shell that adapts to the document's file kind:
 *   - text  : Edit/Read toggle. Editable text docs load their working markdown,
 *             edit in WilfredEditor, and Save re-indexes synchronously.
 *   - image : the picture leads; the assistant's description (the indexed text)
 *             sits beneath in a collapsible section.
 *   - pdf   : Pages (inline browser viewer) | Extracted text (indexed chunks).
 *   - other : reconstructed extracted text with a reading-view toggle (legacy).
 */
(function () {
  var trigger = document.getElementById("view-document-modal-trigger");
  var modal = document.getElementById("view-document-modal");
  if (!trigger || !modal) return;

  var cfgEl = document.getElementById("doc-list-config");
  var csrf = cfgEl ? cfgEl.getAttribute("data-csrf-token") : "";

  var titleEl = modal.querySelector("#view-document-title");
  var iconEl = modal.querySelector("#view-document-icon");
  var loadingEl = modal.querySelector("#view-document-loading");
  var errorEl = modal.querySelector("#view-document-error");
  var retryBtn = modal.querySelector("#view-document-retry");
  var textWrap = modal.querySelector("#view-document-textwrap");
  var textEl = modal.querySelector("#view-document-text");
  var previewEl = modal.querySelector("#view-document-preview");
  var editorMount = modal.querySelector("#view-document-editor");
  var imagePane = modal.querySelector("#view-document-image-pane");
  var imageEl = modal.querySelector("#view-document-image");
  var descEl = modal.querySelector("#view-document-desc");
  var descToggle = modal.querySelector("#view-document-desc-toggle");
  var pdfPane = modal.querySelector("#view-document-pdf-pane");
  var pdfEl = modal.querySelector("#view-document-pdf");
  var segText = modal.querySelector("#view-document-seg-text");
  var segPdf = modal.querySelector("#view-document-seg-pdf");
  var previewBtn = modal.querySelector("#view-document-preview-btn");
  var piiEl = modal.querySelector("#view-document-pii");
  var downloadBtn = modal.querySelector("#view-document-download");
  var saveBtn = modal.querySelector("#view-document-save");
  var saveSpinner = modal.querySelector("#view-document-save-spinner");
  var saveLabel = modal.querySelector("#view-document-save-label");
  var saveStatus = modal.querySelector("#view-document-savestatus");

  var ICON = {
    text: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg>',
    image: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8.5" cy="9.5" r="1.5"/><path d="m21 15-4.5-4.5L7 20"/></svg>',
    pdf: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"/></svg>',
  };
  var ALLOWED = {
    ALLOWED_TAGS: ["p","br","strong","em","u","s","del","code","pre","ul","ol","li","h1","h2","h3","h4","h5","h6","blockquote","a","hr","table","thead","tbody","tr","th","td","div","span","sup","section"],
    ALLOWED_ATTR: ["href","title","target","class","id"],
  };

  // Per-open state
  var cur = {};
  var editor = null;
  var mode = ""; // edit|read (text) · pages|extracted (pdf)
  var chunksCache = null; // reconstructed extracted text (lazy)

  function hidePanes() {
    [loadingEl, errorEl, textEl, previewEl, editorMount, imagePane, pdfPane].forEach(function (e) {
      if (e) e.classList.add("hidden");
    });
    if (textWrap) textWrap.classList.remove("hidden");
  }

  function showLoading() {
    hidePanes();
    loadingEl.classList.remove("hidden");
  }
  function showError() {
    hidePanes();
    errorEl.classList.remove("hidden");
  }

  function renderMd(el, md) {
    try {
      el.innerHTML = DOMPurify.sanitize(marked.parse(md || ""), ALLOWED);
    } catch (e) {
      el.textContent = md || "";
    }
  }

  function destroyEditor() {
    if (editor) {
      try { editor.destroy(); } catch (e) {}
      editor = null;
    }
  }

  function setHeader(kind) {
    var k = kind === "image" || kind === "pdf" ? kind : "text";
    iconEl.innerHTML = ICON[k];
    // Copper chip for text/image; danger tint for PDF.
    if (k === "pdf") {
      iconEl.style.background = "var(--color-danger-soft)";
      iconEl.style.borderColor = "var(--color-danger-subtle)";
      iconEl.style.color = "var(--color-fg-danger)";
    } else {
      iconEl.style.background = "var(--color-accent-soft)";
      iconEl.style.borderColor = "var(--color-accent-subtle)";
      iconEl.style.color = "var(--color-fg-accent)";
    }
  }

  function fetchChunksText() {
    // Returns a promise of the reconstructed extracted text (cached per open).
    if (chunksCache !== null) return Promise.resolve(chunksCache);
    return fetch(cur.chunksUrl, { credentials: "same-origin" })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (data) {
        chunksCache = (data.chunks || [])
          .map(function (c) { return c.text; })
          .join("\n\n");
        return chunksCache;
      });
  }

  // ---- Segmented toggles ----
  function setSeg(segEl, active) {
    if (!segEl) return;
    segEl.querySelectorAll(".wf-seg-btn").forEach(function (b) {
      b.classList.toggle("is-active", b.getAttribute("data-mode") === active);
    });
  }

  // ---- Text (editable): Edit / Read ----
  function enterTextEdit() {
    mode = "edit";
    setSeg(segText, "edit");
    hidePanes();
    editorMount.classList.remove("hidden");
    saveBtn.classList.remove("hidden");
    if (!editor && window.WilfredEditor) {
      editor = window.WilfredEditor.create(editorMount, {
        value: cur.content || "",
        toolbar: true,
        minHeight: "20rem",
        maxHeight: "52vh",
        placeholder: "Document content (markdown)",
      });
    } else if (!editor) {
      var ta = document.createElement("textarea");
      ta.className = "wf-input text-heading text-sm rounded-base block w-full px-3 py-2.5 font-mono";
      ta.rows = 16;
      ta.value = cur.content || "";
      editorMount.appendChild(ta);
      editor = { getValue: function () { return ta.value; }, destroy: function () {} };
    }
  }
  function enterTextRead() {
    mode = "read";
    setSeg(segText, "read");
    hidePanes();
    var md = editor ? editor.getValue() : cur.content || "";
    renderMd(previewEl, md);
    previewEl.classList.remove("hidden");
  }

  // ---- Read-only text / office docs: raw / reading-view ----
  var roPreview = false;
  function renderReadOnly() {
    hidePanes();
    if (roPreview) {
      renderMd(previewEl, chunksCache || "");
      previewEl.classList.remove("hidden");
    } else {
      textEl.textContent = chunksCache || "";
      textEl.classList.remove("hidden");
    }
    previewBtn.classList.toggle("is-active", roPreview);
  }

  // ---- PDF: Pages / Extracted text ----
  function enterPdfPages() {
    mode = "pages";
    setSeg(segPdf, "pages");
    hidePanes();
    if (textWrap) textWrap.classList.add("hidden");
    if (!pdfEl.getAttribute("src")) pdfEl.setAttribute("src", cur.fileUrl);
    pdfPane.classList.remove("hidden");
  }
  function enterPdfExtracted() {
    mode = "extracted";
    setSeg(segPdf, "extracted");
    showLoading();
    fetchChunksText()
      .then(function (txt) {
        hidePanes();
        textEl.textContent = txt;
        textEl.classList.remove("hidden");
      })
      .catch(showError);
  }

  // ---- Open a document ----
  function openDoc(cfg) {
    cur = cfg;
    editor && destroyEditor();
    chunksCache = null;
    roPreview = false;
    mode = "";
    setStatus("");
    // Reset chrome
    [segText, segPdf, previewBtn, saveBtn].forEach(function (e) { if (e) e.classList.add("hidden"); });
    titleEl.textContent = cfg.name || "Document";
    setHeader(cfg.fileKind);
    renderPiiPills(cfg.row);
    if (imageEl) imageEl.removeAttribute("src");
    if (pdfEl) pdfEl.removeAttribute("src");

    if (cfg.fileKind === "image") {
      openImage();
    } else if (cfg.fileKind === "pdf") {
      openPdf();
    } else if (cfg.fileKind === "text" && cfg.editable) {
      openEditableText();
    } else {
      openReadOnly();
    }
    setTimeout(function () { trigger.click(); }, 0);
  }

  function openImage() {
    imageEl.src = cur.fileUrl;
    hidePanes();
    if (textWrap) textWrap.classList.add("hidden");
    imagePane.classList.remove("hidden");
    // Description = the indexed chunk text.
    if (descEl) {
      descEl.textContent = "";
      fetchChunksText()
        .then(function (txt) {
          if (txt && txt.trim()) {
            renderMd(descEl, txt);
          } else {
            descEl.textContent = "No description was indexed for this image.";
          }
        })
        .catch(function () {});
    }
  }

  function openPdf() {
    segPdf.classList.remove("hidden");
    enterPdfPages();
  }

  function openEditableText() {
    segText.classList.remove("hidden");
    showLoading();
    fetch(cur.editSourceUrl, { credentials: "same-origin" })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (data) {
        if (data.editable === false) {
          // Over DOCUMENT_INLINE_EDIT_MAX_CHARS: a save would re-index inline on
          // the web dyno, so the server refuses to offer the editor. Read-only.
          segText.classList.add("hidden");
          setStatus(
            "This document is too large to edit here (over " +
              Number(data.max_chars || 0).toLocaleString() +
              " characters). Showing it read-only — upload a revised file instead.",
            "warn"
          );
          openReadOnly();
          return;
        }
        cur.content = data.content || "";
        if (data.warning) setStatus(data.warning, "warn");
        enterTextEdit();
      })
      .catch(showError);
  }

  function openReadOnly() {
    previewBtn.classList.remove("hidden");
    showLoading();
    fetchChunksText()
      .then(function () { renderReadOnly(); })
      .catch(showError);
  }

  // ---- Save (editable text) ----
  function setSaving(on) {
    saveBtn.disabled = on;
    if (saveSpinner) saveSpinner.classList.toggle("hidden", !on);
    if (saveLabel) saveLabel.textContent = on ? "Saving…" : "Save";
  }
  function setStatus(msg, tone) {
    if (!saveStatus) return;
    if (!msg) {
      saveStatus.classList.add("hidden");
      saveStatus.textContent = "";
      return;
    }
    saveStatus.textContent = msg;
    saveStatus.style.color =
      tone === "danger" ? "var(--color-fg-danger)"
      : tone === "warn" ? "var(--color-fg-warning)"
      : tone === "success" ? "var(--color-fg-success)"
      : "var(--color-body-subtle)";
    saveStatus.classList.remove("hidden");
  }

  if (saveBtn) {
    saveBtn.addEventListener("click", function () {
      if (!editor) return;
      var fd = new FormData();
      fd.append("content", editor.getValue());
      fd.append("csrfmiddlewaretoken", csrf);
      setSaving(true);
      setStatus("Saving and re-indexing…");
      fetch(cur.saveUrl, {
        method: "POST",
        headers: { "X-CSRFToken": csrf, "X-Requested-With": "XMLHttpRequest" },
        body: fd,
        credentials: "same-origin",
      })
        .then(function (r) { return r.json().catch(function () { return {}; }); })
        .then(function (data) {
          setSaving(false);
          if (data.error === "too_large") {
            setStatus(
              "Too large to save here (max " +
                Number(data.max_chars || 0).toLocaleString() +
                " characters). Trim the text or upload it as a file.",
              "danger"
            );
            return;
          }
          if (data.unchanged) {
            setStatus("No changes to save.");
            return;
          }
          if (data.verdict === "clean") {
            setStatus("Saved and re-indexed.", "success");
            reloadSoon();
          } else if (data.verdict === "warn") {
            setStatus("Saved — some sections were excluded as sensitive.", "warn");
            reloadSoon();
          } else if (data.verdict === "blocked") {
            setStatus((data.reason || "Content was rejected as sensitive.") + " Edit and save again.", "danger");
          } else {
            setStatus("The safety scan could not complete. Try saving again.", "danger");
          }
        })
        .catch(function () {
          setSaving(false);
          setStatus("Couldn't save. Please try again.", "danger");
        });
    });
  }

  var reloadTimer = null;
  function reloadSoon() {
    if (reloadTimer) return;
    reloadTimer = setTimeout(function () { window.location.reload(); }, 950);
  }

  // ---- Download ----
  if (downloadBtn) {
    downloadBtn.addEventListener("click", function () {
      if (cur.fileKind === "text") {
        // Download what the user sees (reflects unsaved edits) as .txt.
        var md = editor ? editor.getValue() : chunksCache || "";
        var blob = new Blob([md], { type: "text/plain;charset=utf-8" });
        var url = URL.createObjectURL(blob);
        var a = document.createElement("a");
        a.href = url;
        a.download = (cur.name || "document").replace(/\.[^.]+$/, "") + ".txt";
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(function () { URL.revokeObjectURL(url); }, 0);
      } else if (cur.fileUrl) {
        var link = document.createElement("a");
        link.href = cur.fileUrl + (cur.fileUrl.indexOf("?") === -1 ? "?" : "&") + "download=1";
        link.rel = "noopener";
        document.body.appendChild(link);
        link.click();
        link.remove();
      }
    });
  }

  // ---- Toggle wiring ----
  if (segText) {
    segText.addEventListener("click", function (e) {
      var b = e.target.closest(".wf-seg-btn");
      if (!b) return;
      if (b.getAttribute("data-mode") === "edit") enterTextEdit();
      else enterTextRead();
    });
  }
  if (segPdf) {
    segPdf.addEventListener("click", function (e) {
      var b = e.target.closest(".wf-seg-btn");
      if (!b) return;
      if (b.getAttribute("data-mode") === "pages") enterPdfPages();
      else enterPdfExtracted();
    });
  }
  if (previewBtn) {
    previewBtn.addEventListener("click", function () {
      roPreview = !roPreview;
      renderReadOnly();
    });
  }
  if (descToggle && descEl) {
    descToggle.addEventListener("click", function () {
      var hidden = descEl.classList.toggle("hidden");
      descToggle.textContent = hidden ? "Show" : "Hide";
    });
  }
  if (retryBtn) {
    retryBtn.addEventListener("click", function () {
      if (cur && cur.chunksUrl) openDoc(cur);
    });
  }

  // Destroy the editor when the modal is dismissed (Flowbite close controls).
  modal.querySelectorAll("[data-modal-hide]").forEach(function (el) {
    el.addEventListener("click", function () { destroyEditor(); });
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && !modal.classList.contains("hidden")) destroyEditor();
  });

  // ---- PII pills: clone from the originating row into the footer ----
  function renderPiiPills(row) {
    if (!piiEl) return;
    piiEl.innerHTML = "";
    var src = row && row.querySelector(".pii-pills");
    if (!src) return;
    var clone = src.cloneNode(true);
    clone.classList.remove("hidden", "ms-3");
    clone.classList.add("flex");
    clone.querySelectorAll("[data-tooltip-target]").forEach(function (trg) {
      var oldId = trg.getAttribute("data-tooltip-target");
      var target = clone.querySelector("#" + (window.CSS && CSS.escape ? CSS.escape(oldId) : oldId));
      var newId = "vd-" + oldId;
      trg.setAttribute("data-tooltip-target", newId);
      if (!target) return;
      target.id = newId;
      if (window.Tooltip) {
        new window.Tooltip(target, trg, { placement: "top", triggerType: "hover" });
      } else {
        trg.setAttribute("title", target.textContent.trim());
      }
    });
    piiEl.appendChild(clone);
  }

  // ---- Bind row buttons ----
  document.querySelectorAll(".view-document-btn").forEach(function (btn) {
    btn.addEventListener("click", function (e) {
      e.preventDefault();
      e.stopPropagation();
      openDoc({
        docId: btn.getAttribute("data-doc-id"),
        fileKind: btn.getAttribute("data-file-kind") || "",
        name: btn.getAttribute("data-doc-name") || "Document",
        chunksUrl: btn.getAttribute("data-chunks-url"),
        fileUrl: btn.getAttribute("data-file-url"),
        editSourceUrl: btn.getAttribute("data-edit-source-url"),
        saveUrl: btn.getAttribute("data-save-url"),
        editable: btn.getAttribute("data-editable") === "1",
        quarantined: btn.getAttribute("data-quarantined") === "1",
        row: btn.closest("[data-doc-id]"),
      });
    });
  });
})();
