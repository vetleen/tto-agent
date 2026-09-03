/* Shared slide-theme editor modal (org settings + in-deck picker).
 *
 * A theme is: {id?, label, colors{13}, fonts{4}, typography{3 sizes + bullet_char},
 * tables{4}, footer{bg_color, text, sections[3]{colspan,align,content}}, logo_ext}.
 * The modal builds its own form, previews live, and saves via the accounts
 * slide-theme endpoints (URLs + CSRF read from data-* on #ste-modal). It handles
 * only create/edit + logo upload; the host page owns the list, delete, set-default
 * and (in chat) applying a theme to the deck.
 *
 * Usage: window.SlideThemeEditor.open(themeOrNull, {scope:'org'|'user', onSaved:fn});
 */
(function () {
  "use strict";

  var COLOR_FIELDS = [
    ["lt1", "Background", "Core"], ["dk1", "Body text", "Core"],
    ["dk2", "Headings", "Core"], ["lt2", "Band / soft fill", "Core"],
    ["accent1", "Accent 1", "Accents"], ["accent2", "Accent 2", "Accents"],
    ["accent3", "Accent 3", "Accents"], ["accent4", "Accent 4", "Accents"],
    ["accent5", "Accent 5", "Accents"], ["accent6", "Accent 6", "Accents"],
    ["success", "Success", "Semantic"], ["warning", "Warning", "Semantic"],
    ["danger", "Danger", "Semantic"]
  ];
  var FONT_FIELDS = [["headline", "Headline"], ["subhead", "Subheads"], ["body", "Body"], ["data", "Data / tables"]];
  var FONT_FAMILIES = ["Caladea", "Tinos", "Gelasio", "EBGaramond", "Carlito", "Arimo", "Cousine"];
  var SIZE_FIELDS = [["headline_size", "Headline", 18, 60], ["subhead_size", "Subhead", 10, 40], ["body_size", "Body", 8, 28]];
  var TABLE_FIELDS = [["header_fill", "Header fill"], ["header_color", "Header text"], ["band_fill", "Banded row"], ["grid_color", "Gridlines"]];
  var CONTENTS = [["none", "Empty"], ["logo", "Logo"], ["text", "Disclaimer"], ["page", "Page number"]];
  var ALIGNS = [["left", "Left"], ["center", "Center"], ["right", "Right"]];

  var FOREST = {
    label: "", logo_ext: "",
    colors: {
      lt1: "#FBFAF6", dk1: "#12241B", dk2: "#1F3D30", lt2: "#ECEFE9",
      accent1: "#B87333", accent2: "#2E6B52", accent3: "#7FA891", accent4: "#1C4A42",
      accent5: "#D9A441", accent6: "#A9552F", success: "#3E7D5A", warning: "#D9A441", danger: "#B23A2E"
    },
    fonts: { headline: "Caladea", subhead: "Caladea", body: "Carlito", data: "Carlito" },
    typography: { headline_size: 34, subhead_size: 20, body_size: 14, bullet_char: "‣" },
    tables: { header_fill: "#1F3D30", header_color: "#FBFAF6", band_fill: "#ECEFE9", grid_color: "#7FA891" },
    footer: {
      bg_color: "", text: "", sections: [
        { colspan: 4, align: "left", content: "none" },
        { colspan: 4, align: "center", content: "none" },
        { colspan: 4, align: "right", content: "page" }
      ]
    }
  };

  var modal, body, titleEl, statusEl, cfg = {}, scope = "user", onSaved = null;
  var currentId = "", logoFile = null, existingLogoExt = "";

  function el(id) { return document.getElementById(id); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function clone(o) { return JSON.parse(JSON.stringify(o)); }

  function buildForm() {
    var colorRows = COLOR_FIELDS.map(function (f) {
      return '<label class="flex items-center justify-between gap-2 text-xs text-body">' +
        '<span>' + esc(f[1]) + '</span>' +
        '<input type="color" data-color="' + f[0] + '" class="h-7 w-10 rounded border border-default bg-transparent p-0"></label>';
    }).join("");
    var fontRows = FONT_FIELDS.map(function (f) {
      var opts = FONT_FAMILIES.map(function (fam) { return '<option value="' + fam + '">' + fam + '</option>'; }).join("");
      return '<label class="flex items-center justify-between gap-2 text-xs text-body"><span>' + esc(f[1]) + '</span>' +
        '<select data-font="' + f[0] + '" class="rounded border border-default bg-neutral-primary text-xs px-2 py-1">' + opts + '</select></label>';
    }).join("");
    var sizeRows = SIZE_FIELDS.map(function (f) {
      return '<label class="flex items-center justify-between gap-2 text-xs text-body"><span>' + esc(f[1]) + '</span>' +
        '<input type="number" data-size="' + f[0] + '" min="' + f[2] + '" max="' + f[3] + '" class="w-16 rounded border border-default bg-neutral-primary text-xs px-2 py-1"></label>';
    }).join("");
    var tableRows = TABLE_FIELDS.map(function (f) {
      return '<label class="flex items-center justify-between gap-2 text-xs text-body"><span>' + esc(f[1]) + '</span>' +
        '<input type="color" data-table="' + f[0] + '" class="h-7 w-10 rounded border border-default bg-transparent p-0"></label>';
    }).join("");
    var sectionRows = [0, 1, 2].map(function (i) {
      var cOpts = CONTENTS.map(function (c) { return '<option value="' + c[0] + '">' + c[1] + '</option>'; }).join("");
      var aOpts = ALIGNS.map(function (a) { return '<option value="' + a[0] + '">' + a[1] + '</option>'; }).join("");
      var label = ["Left", "Middle", "Right"][i];
      return '<div class="grid grid-cols-[4rem_1fr_1fr_4rem] items-center gap-2 text-xs">' +
        '<span class="text-body">' + label + '</span>' +
        '<select data-sec="' + i + '" data-k="content" class="rounded border border-default bg-neutral-primary px-2 py-1">' + cOpts + '</select>' +
        '<select data-sec="' + i + '" data-k="align" class="rounded border border-default bg-neutral-primary px-2 py-1">' + aOpts + '</select>' +
        '<input type="number" data-sec="' + i + '" data-k="colspan" min="1" max="12" class="w-full rounded border border-default bg-neutral-primary px-2 py-1" title="Columns (1-12)"></div>';
    }).join("");

    body.innerHTML =
      '<div><label class="block text-sm font-medium text-heading mb-1">Name</label>' +
      '<input id="ste-label" type="text" maxlength="40" placeholder="e.g. Acme Brand" class="w-full rounded-base border border-default bg-neutral-primary px-3 py-2 text-sm"></div>' +

      '<div id="ste-preview" class="rounded-base border border-default overflow-hidden"></div>' +

      '<details open><summary class="cursor-pointer text-sm font-medium text-heading">Colours</summary>' +
      '<div class="mt-2 grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-1.5">' + colorRows + '</div></details>' +

      '<details><summary class="cursor-pointer text-sm font-medium text-heading">Fonts &amp; type</summary>' +
      '<div class="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-1.5">' + fontRows + sizeRows +
      '<label class="flex items-center justify-between gap-2 text-xs text-body"><span>Bullet</span>' +
      '<input id="ste-bullet" type="text" maxlength="4" class="w-16 rounded border border-default bg-neutral-primary text-xs px-2 py-1"></label>' +
      '</div></details>' +

      '<details><summary class="cursor-pointer text-sm font-medium text-heading">Tables</summary>' +
      '<div class="mt-2 grid grid-cols-2 gap-x-4 gap-y-1.5">' + tableRows + '</div></details>' +

      '<details open><summary class="cursor-pointer text-sm font-medium text-heading">Footer</summary>' +
      '<div class="mt-2 space-y-2">' +
      '<div class="grid grid-cols-[4rem_1fr_1fr_4rem] gap-2 text-[11px] uppercase tracking-wide text-body"><span></span><span>Content</span><span>Align</span><span>Cols</span></div>' +
      sectionRows +
      '<label class="block text-xs text-body mt-1">Disclaimer text' +
      '<input id="ste-footer-text" type="text" maxlength="200" class="mt-1 w-full rounded border border-default bg-neutral-primary px-2 py-1 text-sm"></label>' +
      '<div class="flex items-center gap-3 mt-1">' +
      '<label class="flex items-center gap-2 text-xs text-body"><input id="ste-footer-bg-on" type="checkbox" class="rounded border-default text-brand"> Band background</label>' +
      '<input id="ste-footer-bg" type="color" class="h-7 w-10 rounded border border-default bg-transparent p-0"></div>' +
      '<div class="flex items-center gap-3 mt-2">' +
      '<span id="ste-logo-prev" class="inline-flex h-10 min-w-10 max-w-32 items-center justify-center overflow-hidden rounded border border-default bg-neutral-primary-soft px-2 text-xs text-body">None</span>' +
      '<button type="button" id="ste-logo-btn" class="rounded-base border border-default px-3 py-1.5 text-xs font-medium text-heading hover:bg-neutral-secondary-soft">Upload logo</button>' +
      '<button type="button" id="ste-logo-rm" class="text-xs text-fg-danger hover:underline hidden">Remove</button>' +
      '<input id="ste-logo-file" type="file" accept="image/png,image/jpeg,image/webp" class="hidden"></div>' +
      '<p class="text-[11px] text-body">A section set to &ldquo;Logo&rdquo; shows this. Also placeable on slides via the token.</p>' +
      '</div></details>';

    body.addEventListener("input", schedulePreview);
    body.addEventListener("change", schedulePreview);
    el("ste-footer-bg-on").addEventListener("change", schedulePreview);
    el("ste-logo-btn").addEventListener("click", function () { el("ste-logo-file").click(); });
    el("ste-logo-file").addEventListener("change", function () {
      logoFile = this.files && this.files[0] ? this.files[0] : null;
      if (logoFile) { showLogoPreview(URL.createObjectURL(logoFile)); }
    });
    el("ste-logo-rm").addEventListener("click", function () {
      logoFile = null; existingLogoExt = ""; el("ste-logo-file").value = "";
      showLogoPreview(null);
    });
  }

  function showLogoPreview(url) {
    var box = el("ste-logo-prev"), rm = el("ste-logo-rm");
    if (url) {
      box.innerHTML = '<img src="' + url + '" alt="logo" class="max-h-10 w-auto object-contain">';
      rm.classList.remove("hidden");
    } else {
      box.textContent = "None"; rm.classList.add("hidden");
    }
  }

  function populate(theme) {
    var t = theme ? clone(theme) : clone(FOREST);
    if (!t.colors) t = clone(FOREST);
    currentId = theme && theme.id ? theme.id : "";
    existingLogoExt = t.logo_ext || "";
    logoFile = null;
    el("ste-label").value = t.label || "";
    COLOR_FIELDS.forEach(function (f) { var i = body.querySelector('[data-color="' + f[0] + '"]'); if (i) i.value = t.colors[f[0]] || "#000000"; });
    FONT_FIELDS.forEach(function (f) { var i = body.querySelector('[data-font="' + f[0] + '"]'); if (i) i.value = (t.fonts && t.fonts[f[0]]) || "Carlito"; });
    SIZE_FIELDS.forEach(function (f) { var i = body.querySelector('[data-size="' + f[0] + '"]'); if (i) i.value = (t.typography && t.typography[f[0]]) || f[2]; });
    el("ste-bullet").value = (t.typography && t.typography.bullet_char) || "‣";
    TABLE_FIELDS.forEach(function (f) { var i = body.querySelector('[data-table="' + f[0] + '"]'); if (i) i.value = (t.tables && t.tables[f[0]]) || "#000000"; });
    var footer = t.footer || clone(FOREST.footer);
    (footer.sections || []).forEach(function (s, i) {
      var cs = body.querySelector('[data-sec="' + i + '"][data-k="content"]'); if (cs) cs.value = s.content || "none";
      var al = body.querySelector('[data-sec="' + i + '"][data-k="align"]'); if (al) al.value = s.align || "left";
      var col = body.querySelector('[data-sec="' + i + '"][data-k="colspan"]'); if (col) col.value = s.colspan || 4;
    });
    el("ste-footer-text").value = footer.text || "";
    var bg = footer.bg_color || "";
    el("ste-footer-bg-on").checked = !!bg;
    el("ste-footer-bg").value = (bg && bg[0] === "#") ? bg : "#1F3D30";
    if (existingLogoExt && currentId && cfg.logoServe) {
      showLogoPreview(cfg.logoServe + "?scope=" + encodeURIComponent(scope) + "&id=" + encodeURIComponent(currentId) + "&t=" + Date.now());
    } else {
      showLogoPreview(null);
    }
    schedulePreview();
  }

  function collect() {
    var t = { label: el("ste-label").value.trim(), colors: {}, fonts: {}, typography: {}, tables: {}, footer: { sections: [] } };
    if (currentId) t.id = currentId;
    COLOR_FIELDS.forEach(function (f) { t.colors[f[0]] = body.querySelector('[data-color="' + f[0] + '"]').value; });
    FONT_FIELDS.forEach(function (f) { t.fonts[f[0]] = body.querySelector('[data-font="' + f[0] + '"]').value; });
    SIZE_FIELDS.forEach(function (f) { t.typography[f[0]] = parseInt(body.querySelector('[data-size="' + f[0] + '"]').value, 10) || f[2]; });
    t.typography.bullet_char = el("ste-bullet").value || "‣";
    TABLE_FIELDS.forEach(function (f) { t.tables[f[0]] = body.querySelector('[data-table="' + f[0] + '"]').value; });
    [0, 1, 2].forEach(function (i) {
      t.footer.sections.push({
        content: body.querySelector('[data-sec="' + i + '"][data-k="content"]').value,
        align: body.querySelector('[data-sec="' + i + '"][data-k="align"]').value,
        colspan: parseInt(body.querySelector('[data-sec="' + i + '"][data-k="colspan"]').value, 10) || 4
      });
    });
    t.footer.text = el("ste-footer-text").value;
    t.footer.bg_color = el("ste-footer-bg-on").checked ? el("ste-footer-bg").value : "";
    return t;
  }

  var previewTimer = null;
  function schedulePreview() { clearTimeout(previewTimer); previewTimer = setTimeout(renderPreview, 60); }
  function renderPreview() {
    var t = collect();
    var band = t.footer.bg_color ? t.footer.bg_color : t.colors.lt1;
    var logoUrl = null;
    if (logoFile) logoUrl = URL.createObjectURL(logoFile);
    else if (existingLogoExt && currentId && cfg.logoServe) logoUrl = cfg.logoServe + "?scope=" + encodeURIComponent(scope) + "&id=" + encodeURIComponent(currentId) + "&t=" + Date.now();
    function cell(s) {
      var a = s.align === "center" ? "center" : (s.align === "right" ? "flex-end" : "flex-start");
      var inner = "";
      if (s.content === "text") inner = '<span style="font-size:9px;color:' + t.colors.dk2 + '">' + esc(t.footer.text || "Disclaimer") + '</span>';
      else if (s.content === "page") inner = '<span style="font-size:9px;color:' + t.colors.dk2 + '">1</span>';
      else if (s.content === "logo") inner = logoUrl ? '<img src="' + logoUrl + '" style="height:14px;width:auto">' : '<span style="font-size:8px;color:' + t.colors.accent3 + '">[logo]</span>';
      return '<div style="flex:' + (s.colspan || 4) + ';display:flex;justify-content:' + a + ';align-items:center;padding:0 6px">' + inner + '</div>';
    }
    el("ste-preview").innerHTML =
      '<div style="background:' + t.colors.lt1 + ';padding:14px 16px 0">' +
      '<div style="font-weight:600;color:' + t.colors.dk2 + '">Quarterly overview</div>' +
      '<div style="font-size:12px;color:' + t.colors.dk1 + ';margin-top:2px">Highlights and results for the period.</div>' +
      '<div style="display:flex;gap:6px;margin:10px 0">' +
      ["accent1", "accent2", "accent3", "accent4"].map(function (k) { return '<span style="width:26px;height:8px;border-radius:9999px;background:' + t.colors[k] + '"></span>'; }).join("") +
      '</div><div style="display:flex;height:24px;background:' + band + '">' + t.footer.sections.map(cell).join("") + '</div></div>';
  }

  function setStatus(msg, isErr) { statusEl.textContent = msg || ""; statusEl.className = "text-xs mr-auto " + (isErr ? "text-fg-danger" : "text-body"); }

  function postJSON(url, obj) {
    return fetch(url, { method: "POST", headers: { "X-CSRFToken": cfg.csrf, "Content-Type": "application/json" }, credentials: "same-origin", body: JSON.stringify(obj) })
      .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); });
  }

  function save() {
    var theme = collect();
    if (!theme.label) { setStatus("Give the theme a name.", true); return; }
    setStatus("Saving…", false);
    postJSON(cfg.save, { scope: scope, theme: theme }).then(function (res) {
      if (!res.ok || res.d.error) { setStatus(res.d.error || "Could not save.", true); return; }
      var saved = res.d.theme;
      var done = function () { hide(); if (onSaved) onSaved(saved); };
      if (logoFile) {
        var fd = new FormData(); fd.append("scope", scope); fd.append("theme_id", saved.id); fd.append("logo", logoFile);
        fetch(cfg.logoUpload, { method: "POST", headers: { "X-CSRFToken": cfg.csrf }, credentials: "same-origin", body: fd })
          .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); })
          .then(function (lr) { if (lr.ok && lr.d.logo_ext) saved.logo_ext = lr.d.logo_ext; else setStatus(lr.d.error || "Logo upload failed.", true); done(); })
          .catch(function () { done(); });
      } else if (existingLogoExt === "" && currentId) {
        // logo was removed in the editor
        postJSON(cfg.logoDelete, { scope: scope, id: currentId }).then(done).catch(done);
      } else { done(); }
    }).catch(function () { setStatus("Network error.", true); });
  }

  function show() { modal.classList.remove("hidden"); modal.classList.add("flex"); }
  function hide() { modal.classList.add("hidden"); modal.classList.remove("flex"); }

  function open(theme, opts) {
    opts = opts || {};
    scope = opts.scope === "org" ? "org" : "user";
    onSaved = opts.onSaved || null;
    titleEl.textContent = theme && theme.id ? "Edit theme" : "New theme";
    setStatus("", false);
    populate(theme);
    show();
  }

  document.addEventListener("DOMContentLoaded", function () {
    modal = el("ste-modal");
    if (!modal) return;  // page didn't include the editor partial
    body = el("ste-body");
    titleEl = el("ste-title");
    statusEl = el("ste-status");
    cfg = {
      csrf: modal.dataset.csrf, save: modal.dataset.saveUrl, logoUpload: modal.dataset.logoUploadUrl,
      logoDelete: modal.dataset.logoDeleteUrl, logoServe: modal.dataset.logoServeUrl
    };
    buildForm();
    el("ste-close").addEventListener("click", hide);
    el("ste-cancel").addEventListener("click", hide);
    el("ste-save").addEventListener("click", save);
    modal.addEventListener("click", function (e) { if (e.target === modal) hide(); });
    window.SlideThemeEditor = { open: open };
  });
})();
