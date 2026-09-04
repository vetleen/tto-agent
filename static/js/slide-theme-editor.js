/* Shared slide-theme editor modal (org settings + in-deck picker).
 *
 * A theme is: {id?, label, colors{13}, fonts{4}, typography{3 sizes + bullet_char},
 * tables{4}, footer{bg_color, text, sections[3]{colspan,align,content}}, logo_ext}.
 * The modal is a two-pane "studio": a section rail on the left, one section's
 * controls in the middle, and a live preview pinned on the right. It builds its
 * own form, previews live, and saves via the accounts slide-theme endpoints
 * (URLs + CSRF read from data-* on #ste-modal). It handles only create/edit +
 * logo upload; the host page owns the list, delete, set-default and (in chat)
 * applying a theme to the deck.
 *
 * Usage:  window.SlideThemeEditor.open(themeOrNull, {scope:'org'|'user', onSaved:fn});
 * Shared: window.SlideThemeEditor.miniSlide(theme, {height, logoUrl, showBullet, showFooter})
 *         → HTML string for a miniature title slide (also used by the picker and
 *         the org theme list so all three surfaces render identical previews).
 *         The title is the theme's own name, set in its headline font — the
 *         accounts:slide_fonts_css stylesheet (linked by the modal partial)
 *         provides @font-face for bundled + org-uploaded families. The footer
 *         band renders only with showFooter (the editor, where it's configured).
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
  var SECTIONS = [["general", "General"], ["colours", "Colours"], ["fonts", "Fonts &amp; type"], ["tables", "Tables"], ["footer", "Footer"], ["logo", "Logo"]];

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
      bg_color: "", text: "", logo_height: 20, sections: [
        { colspan: 4, align: "left", content: "none" },
        { colspan: 4, align: "center", content: "none" },
        { colspan: 4, align: "right", content: "page" }
      ]
    }
  };

  var modal, body, titleEl, scopeEl, statusEl, cfg = {}, scope = "user", onSaved = null;
  var currentId = "", logoFile = null, existingLogoExt = "", logoObjectUrl = null;
  var uploadedFonts = [];  // org-uploaded families: [{family, family_norm}, …]

  function el(id) { return document.getElementById(id); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function clone(o) { return JSON.parse(JSON.stringify(o)); }

  // Font <select> options: bundled families + the org's uploaded "Your fonts".
  function fontOptionsHtml() {
    var opt = function (fam) { return '<option value="' + esc(fam) + '">' + esc(fam) + '</option>'; };
    var html = '<optgroup label="Bundled">' + FONT_FAMILIES.map(opt).join("") + '</optgroup>';
    if (uploadedFonts.length) {
      html += '<optgroup label="Your fonts">' + uploadedFonts.map(function (f) { return opt(f.family); }).join("") + '</optgroup>';
    }
    return html;
  }
  function ensureFontOption(select, fam) {
    if (!fam || !select) return;
    if (![].some.call(select.options, function (o) { return o.value === fam; })) {
      var o = document.createElement("option"); o.value = fam; o.textContent = fam; select.appendChild(o);
    }
  }
  function rebuildFontSelects() {
    body.querySelectorAll('select[data-font]').forEach(function (s) {
      var cur = s.value; s.innerHTML = fontOptionsHtml(); ensureFontOption(s, cur); s.value = cur;
    });
  }

  /* ---- Shared miniature title slide -------------------------------------- */
  function miniSlide(theme, opts) {
    opts = opts || {};
    var t = theme || {};
    var c = t.colors || {};
    var lt1 = c.lt1 || "#FBFAF6", dk1 = c.dk1 || "#12241B", dk2 = c.dk2 || "#1F3D30", lt2 = c.lt2 || "#ECEFE9";
    var footer = t.footer || {};
    var band = footer.bg_color ? footer.bg_color : lt1;
    var logoUrl = opts.logoUrl || null;
    var h = opts.height || 96;
    var big = h >= 130;
    var pillW = big ? 26 : 20, pillH = big ? 8 : 6, titleSize = big ? 19 : 15, subSize = big ? 11 : 9;
    var pills = ["accent1", "accent2", "accent3", "accent4"].map(function (k) {
      return '<span style="width:' + pillW + 'px;height:' + pillH + 'px;border-radius:9999px;background:' + (c[k] || "#8FA891") + '"></span>';
    }).join("");

    var bandPx = big ? 24 : 18;
    var logoH = Math.max(4, Math.round((footer.logo_height || 20) * bandPx / 28));  // 28 = FOOTER_BAND_H (pt)
    var secs = (footer.sections && footer.sections.length) ? footer.sections : [{ colspan: 4, align: "right", content: "page" }];
    function cell(s) {
      var a = s.align === "center" ? "center" : (s.align === "right" ? "flex-end" : "flex-start");
      var inner = "";
      if (s.content === "text") inner = '<span style="font-size:' + (big ? 9 : 8) + 'px;color:' + dk2 + '">' + esc(footer.text || "Disclaimer") + '</span>';
      else if (s.content === "page") inner = '<span style="font-size:' + (big ? 9 : 8) + 'px;color:' + dk2 + '">1</span>';
      else if (s.content === "logo") inner = logoUrl ? '<img src="' + logoUrl + '" style="height:' + logoH + 'px;width:auto;display:block">' : "";
      return '<div style="flex:' + (s.colspan || 4) + ';display:flex;justify-content:' + a + ';align-items:center;padding:0 6px;min-width:0;overflow:hidden">' + inner + '</div>';
    }

    var pad = big ? 18 : 14;
    var bullet = "";
    if (opts.showBullet) {
      var bch = (t.typography && t.typography.bullet_char) || "‣";
      bullet = '<div style="margin:8px 0 0;display:flex;align-items:center;gap:6px">' +
        '<span style="color:' + dk2 + ';font-size:11px">' + esc(bch) + '</span>' +
        '<span style="height:5px;flex:1;max-width:120px;border-radius:9999px;background:' + lt2 + '"></span></div>';
    }
    // Title = the theme's own name in its headline font (the eyes should land on
    // the name, not sample copy); subtitle stays sample text in the subhead font.
    var fonts = t.fonts || {};
    var headStack = cssFontStack(fonts.headline, "var(--font-serif)");
    var subStack = cssFontStack(fonts.subhead || fonts.body, "var(--font-serif)");
    var title = (t.label || "").trim() || "Theme name";
    var footerBand = "";
    if (opts.showFooter) {
      footerBand = '<div style="margin-top:auto;margin-left:-' + pad + 'px;margin-right:-' + pad + 'px;display:flex;align-items:center;height:' + (big ? 24 : 18) + 'px;background:' + band + '">' +
        secs.map(cell).join("") + '</div>';
    }
    return '<div style="background:' + lt1 + ';padding:' + pad + 'px ' + pad + 'px 0;height:' + h + 'px;display:flex;flex-direction:column">' +
      '<div style="font-family:' + headStack + ';font-weight:600;font-size:' + titleSize + 'px;line-height:1.15;color:' + dk2 + ';white-space:nowrap;overflow:hidden;text-overflow:ellipsis">' + esc(title) + '</div>' +
      '<div style="font-family:' + subStack + ';font-size:' + subSize + 'px;color:' + dk1 + ';margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">Highlights and results for the period.</div>' +
      '<div style="display:flex;gap:' + (big ? 5 : 4) + 'px;margin-top:' + (big ? 12 : 8) + 'px">' + pills + '</div>' +
      bullet + footerBand + '</div>';
  }

  // A safe CSS font-family stack: the family name stripped of characters that
  // could break out of the inline style, quoted, with the given fallback.
  function cssFontStack(family, fallback) {
    var fam = family ? String(family).replace(/['"<>\\;{}()&]/g, "").trim() : "";
    return fam ? "'" + fam + "', " + fallback : fallback;
  }

  /* ---- Form construction -------------------------------------------------- */
  function normHex(s) {
    if (s == null) return null;
    s = String(s).trim().replace(/^#/, "");
    if (/^[0-9a-fA-F]{3}$/.test(s)) s = s[0] + s[0] + s[1] + s[1] + s[2] + s[2];
    return /^[0-9a-fA-F]{6}$/.test(s) ? "#" + s.toUpperCase() : null;
  }
  // A swatch "well": a clickable colour chip (over a native colour input) plus an
  // editable hex text field. `attr` is data-color or data-table so collect()/populate()
  // keep reading the colour input as the source of truth.
  function swatchWell(attr, key, label) {
    return '<div class="ste-well flex flex-col gap-1.5 rounded-base border border-default-subtle p-2 hover:border-default-strong">' +
      '<div class="relative h-7">' +
      '<span data-chip class="block h-7 w-full rounded-sm border border-default"></span>' +
      '<input type="color" ' + attr + '="' + key + '" aria-label="' + esc(label) + '" class="absolute inset-0 h-full w-full cursor-pointer opacity-0">' +
      '</div>' +
      '<span class="text-[11px] leading-tight text-body">' + esc(label) + '</span>' +
      '<input type="text" data-hex maxlength="7" spellcheck="false" aria-label="' + esc(label) + ' hex" ' +
      'class="w-full rounded-sm border border-default-subtle bg-neutral-secondary-soft px-1 py-0.5 text-center font-mono text-[9px] uppercase tracking-tight text-body focus:border-default-strong focus:outline-none">' +
      '</div>';
  }
  function colorGroup(title, cols) {
    var wells = COLOR_FIELDS.filter(function (f) { return f[2] === title; })
      .map(function (f) { return swatchWell("data-color", f[0], f[1]); }).join("");
    return '<div><div class="mb-2.5 text-[11px] font-semibold uppercase tracking-wide text-fg-accent">' + title + '</div>' +
      '<div class="grid ' + cols + ' gap-2.5">' + wells + '</div></div>';
  }

  function buildForm() {
    // Nav rail
    var navBtns = SECTIONS.map(function (s) {
      return '<button type="button" class="ste-nav shrink-0 whitespace-nowrap rounded-base border-l-2 border-transparent px-3 py-2 text-left text-[13px] text-body hover:bg-neutral-tertiary hover:text-heading lg:rounded-none lg:px-5" data-section="' + s[0] + '">' + s[1] + '</button>';
    }).join("");
    var nav = '<nav class="flex gap-1 overflow-x-auto border-b border-default-subtle px-3 py-2 lg:flex-col lg:gap-0.5 lg:overflow-visible lg:border-b-0 lg:border-r lg:px-0 lg:py-4">' +
      '<div class="hidden px-5 pb-2 text-[11px] font-semibold uppercase tracking-wide text-body-subtle lg:block">Sections</div>' + navBtns + '</nav>';

    // Middle — one panel per section (General holds the name)
    var pGeneral = '<div class="ste-panel space-y-1.5" data-panel="general">' +
      '<label class="block max-w-md"><span class="mb-1.5 block text-xs font-semibold text-heading">Name</span>' +
      '<input id="ste-label" type="text" maxlength="40" placeholder="e.g. Acme Brand" class="h-9 w-full rounded-base border border-default bg-neutral-secondary-soft px-3 text-sm text-heading"></label>' +
      '<p class="text-[11px] text-body-subtle">Shown in the deck theme picker.</p></div>';

    var pColours = '<div class="ste-panel hidden space-y-5" data-panel="colours">' +
      colorGroup("Core", "grid-cols-2 sm:grid-cols-4") +
      colorGroup("Accents", "grid-cols-2 sm:grid-cols-3") +
      colorGroup("Semantic", "grid-cols-3") + '</div>';

    var fontRows = FONT_FIELDS.map(function (f) {
      return '<label class="flex items-center justify-between gap-3 text-xs text-body"><span>' + esc(f[1]) + '</span>' +
        '<select data-font="' + f[0] + '" class="h-8 rounded-base border border-default bg-neutral-secondary-soft px-2 text-xs text-heading">' + fontOptionsHtml() + '</select></label>';
    }).join("");
    var sizeRows = SIZE_FIELDS.map(function (f) {
      return '<label class="flex items-center justify-between gap-2 text-xs text-body"><span>' + esc(f[1]) + '</span>' +
        '<input type="number" data-size="' + f[0] + '" min="' + f[2] + '" max="' + f[3] + '" class="h-8 w-20 rounded-base border border-default bg-neutral-secondary-soft px-2 text-xs text-heading"></label>';
    }).join("");
    var uploadRow = cfg.isOrgAdmin && cfg.fontsUpload
      ? '<div class="flex flex-wrap items-center gap-3 border-t border-default-subtle pt-3">' +
        '<button type="button" id="ste-font-btn" class="rounded-base border border-default px-3 py-1.5 text-xs font-medium text-heading hover:bg-neutral-tertiary">Upload font&hellip;</button>' +
        '<input id="ste-font-file" type="file" accept=".ttf,.otf,.woff,.woff2" class="hidden">' +
        '<span id="ste-font-status" class="text-xs text-body-subtle"></span>' +
        '<p class="w-full text-[11px] text-body-subtle">Uploaded fonts are shared with your organization (canvas &amp; slides). TrueType/OpenType fonts embed in the exported .pptx; WOFF renders in the preview and is referenced by name.</p>' +
        '</div>'
      : "";
    var pFonts = '<div class="ste-panel hidden space-y-4" data-panel="fonts">' +
      '<div class="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2">' + fontRows + '</div>' +
      '<div class="grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-3">' + sizeRows +
      '<label class="flex items-center justify-between gap-2 text-xs text-body"><span>Bullet</span>' +
      '<input id="ste-bullet" type="text" maxlength="4" class="h-8 w-20 rounded-base border border-default bg-neutral-secondary-soft px-2 text-xs text-heading"></label>' +
      '</div>' + uploadRow + '</div>';

    var tableWells = TABLE_FIELDS.map(function (f) { return swatchWell("data-table", f[0], f[1]); }).join("");
    var pTables = '<div class="ste-panel hidden" data-panel="tables">' +
      '<div class="grid grid-cols-2 gap-2.5 sm:grid-cols-4">' + tableWells + '</div></div>';

    var sectionRows = [0, 1, 2].map(function (i) {
      var cOpts = CONTENTS.map(function (c) { return '<option value="' + c[0] + '">' + c[1] + '</option>'; }).join("");
      var aOpts = ALIGNS.map(function (a) { return '<option value="' + a[0] + '">' + a[1] + '</option>'; }).join("");
      var label = ["Left", "Middle", "Right"][i];
      return '<div class="grid grid-cols-[3.5rem_1fr_1fr_3.5rem] items-center gap-2 text-xs">' +
        '<span class="text-body">' + label + '</span>' +
        '<select data-sec="' + i + '" data-k="content" class="h-8 rounded-base border border-default bg-neutral-secondary-soft px-2 text-xs text-heading">' + cOpts + '</select>' +
        '<select data-sec="' + i + '" data-k="align" class="h-8 rounded-base border border-default bg-neutral-secondary-soft px-2 text-xs text-heading">' + aOpts + '</select>' +
        '<input type="number" data-sec="' + i + '" data-k="colspan" min="1" max="12" class="h-8 w-full rounded-base border border-default bg-neutral-secondary-soft px-2 text-xs text-heading" title="Columns (1-12)"></div>';
    }).join("");
    var pFooter = '<div class="ste-panel hidden space-y-3" data-panel="footer">' +
      '<div class="grid grid-cols-[3.5rem_1fr_1fr_3.5rem] gap-2 text-[10px] font-semibold uppercase tracking-wide text-body-subtle"><span></span><span>Content</span><span>Align</span><span>Cols</span></div>' +
      sectionRows +
      '<label class="block text-xs text-body">Disclaimer text' +
      '<input id="ste-footer-text" type="text" maxlength="200" class="mt-1 h-9 w-full rounded-base border border-default bg-neutral-secondary-soft px-2 text-sm text-heading"></label>' +
      '<div class="flex items-center gap-3">' +
      '<label class="flex items-center gap-2 text-xs text-body"><input id="ste-footer-bg-on" type="checkbox" class="rounded border-default text-brand"> Band background</label>' +
      '<input id="ste-footer-bg" type="color" class="h-7 w-10 rounded border border-default bg-transparent p-0"></div>' +
      '</div>';

    var pLogo = '<div class="ste-panel hidden space-y-3" data-panel="logo">' +
      '<div class="flex flex-wrap items-center gap-3">' +
      '<span id="ste-logo-prev" class="inline-flex h-12 min-w-12 max-w-40 items-center justify-center overflow-hidden rounded-base border border-default bg-neutral-secondary-soft px-2 text-xs text-body-subtle">None</span>' +
      '<button type="button" id="ste-logo-btn" class="rounded-base border border-default px-3 py-1.5 text-xs font-medium text-heading hover:bg-neutral-tertiary">Upload logo</button>' +
      '<button type="button" id="ste-logo-rm" class="hidden text-xs text-fg-danger hover:underline">Remove</button>' +
      '<input id="ste-logo-file" type="file" accept="image/png,image/jpeg,image/webp" class="hidden"></div>' +
      '<label class="flex max-w-xs items-center justify-between gap-3 text-xs text-body"><span>Footer logo height (pt)</span>' +
      '<input id="ste-logo-height" type="number" min="8" max="48" class="h-8 w-20 rounded-base border border-default bg-neutral-secondary-soft px-2 text-xs text-heading"></label>' +
      '<p class="text-[11px] text-body-subtle">A footer section set to &ldquo;Logo&rdquo; shows this, vertically centred in the footer band. Also placeable on slides via the token.</p>' +
      '</div>';

    var middle = '<div class="min-w-0 px-6 py-5 lg:min-h-0 lg:overflow-y-auto">' + pGeneral + pColours + pFonts + pTables + pFooter + pLogo + '</div>';

    // Right — pinned live preview
    var aside = '<aside class="border-t border-default-subtle bg-neutral-secondary-soft p-5 lg:min-h-0 lg:overflow-y-auto lg:border-l lg:border-t-0">' +
      '<div class="mb-2.5 text-[11px] font-semibold uppercase tracking-wide text-body-subtle">Live preview</div>' +
      '<div id="ste-preview" class="overflow-hidden rounded-base border border-default shadow-md"></div>' +
      '<div id="ste-summary" class="mt-3.5 flex flex-col gap-1.5"></div>' +
      '<p class="mt-3.5 text-[11px] leading-relaxed text-body-subtle">The preview updates as you edit. It shows the title block, accent order, and footer band.</p>' +
      '</aside>';

    body.innerHTML = '<div class="grid grid-cols-1 lg:h-full lg:min-h-0 lg:grid-cols-[172px_1fr_320px]">' + nav + middle + aside + '</div>';

    body.addEventListener("input", onBodyInput);
    body.addEventListener("change", onBodyInput);
    body.querySelectorAll(".ste-nav").forEach(function (b) {
      b.addEventListener("click", function () { showSection(b.getAttribute("data-section")); });
    });
    el("ste-logo-btn").addEventListener("click", function () { el("ste-logo-file").click(); });
    el("ste-logo-file").addEventListener("change", function () {
      logoFile = this.files && this.files[0] ? this.files[0] : null;
      if (logoObjectUrl) { try { URL.revokeObjectURL(logoObjectUrl); } catch (e) {} logoObjectUrl = null; }
      if (logoFile) { logoObjectUrl = URL.createObjectURL(logoFile); showLogoPreview(logoObjectUrl); schedulePreview(); }
    });
    el("ste-logo-rm").addEventListener("click", function () {
      logoFile = null; existingLogoExt = ""; el("ste-logo-file").value = "";
      if (logoObjectUrl) { try { URL.revokeObjectURL(logoObjectUrl); } catch (e) {} logoObjectUrl = null; }
      showLogoPreview(null); schedulePreview();
    });
    el("ste-label").addEventListener("input", function () { this.style.borderColor = ""; });
    var fontBtn = el("ste-font-btn");
    if (fontBtn) {
      fontBtn.addEventListener("click", function () { el("ste-font-file").click(); });
      el("ste-font-file").addEventListener("change", function () {
        var file = this.files && this.files[0]; this.value = "";
        if (file) uploadFont(file);
      });
    }
    showSection("general");
  }

  // Re-fetch the @font-face stylesheet (cache-busted) so a just-uploaded font
  // can render in the live preview without a page reload.
  function refreshFontsCss() {
    var link = el("ste-fonts-css");
    if (!link) return;
    var href = link.getAttribute("href").split("?")[0];
    link.setAttribute("href", href + "?t=" + Date.now());
  }

  function uploadFont(file) {
    var st = el("ste-font-status");
    var setSt = function (msg, err) { if (st) { st.textContent = msg; st.className = "text-xs " + (err ? "text-fg-danger" : "text-body-subtle"); } };
    setSt("Uploading…", false);
    var fd = new FormData(); fd.append("file", file);
    fetch(cfg.fontsUpload, { method: "POST", headers: { "X-CSRFToken": cfg.csrf }, credentials: "same-origin", body: fd })
      .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); })
      .then(function (res) {
        if (!res.ok || res.d.error) { setSt(res.d.error || "Upload failed.", true); return; }
        var before = uploadedFonts.map(function (f) { return f.family; });
        uploadedFonts = res.d.fonts || [];
        rebuildFontSelects();
        refreshFontsCss();
        var added = uploadedFonts.map(function (f) { return f.family; }).filter(function (f) { return before.indexOf(f) < 0; });
        setSt(added.length ? ("Added “" + added[0] + "” — pick it above.") : "Fonts updated.", false);
      })
      .catch(function () { setSt("Network error.", true); });
  }

  function onBodyInput(e) {
    var t = e.target;
    if (t && t.matches && t.matches('input[type="color"]')) {
      var wc = t.closest(".ste-well"); if (wc) syncSwatch(wc);
      schedulePreview();
      return;
    }
    if (t && t.matches && t.matches("input[data-hex]")) {
      var wh = t.closest(".ste-well");
      var ci = wh && wh.querySelector('input[type="color"]');
      var norm = normHex(t.value);
      if (ci && norm) {
        ci.value = norm;
        var chip = wh.querySelector("[data-chip]"); if (chip) chip.style.background = norm;
        if (e.type === "change") t.value = norm;  // canonicalise on commit
        schedulePreview();
      } else if (e.type === "change" && ci) {
        t.value = (ci.value || "").toUpperCase();  // reject invalid, restore last good
      }
      return;
    }
    schedulePreview();
  }

  function syncSwatch(w) {
    var inp = w.querySelector('input[type="color"]');
    if (!inp) return;
    var v = (inp.value || "").toUpperCase();
    var chip = w.querySelector("[data-chip]"); if (chip) chip.style.background = v;
    var hex = w.querySelector("[data-hex]"); if (hex) hex.value = v;
  }
  function syncAllSwatches() { body.querySelectorAll(".ste-well").forEach(syncSwatch); }

  function showSection(name) {
    body.querySelectorAll(".ste-panel").forEach(function (p) {
      p.classList.toggle("hidden", p.getAttribute("data-panel") !== name);
    });
    body.querySelectorAll(".ste-nav").forEach(function (b) {
      var on = b.getAttribute("data-section") === name;
      b.classList.toggle("text-fg-accent", on);
      b.classList.toggle("bg-accent-soft", on);
      b.classList.toggle("lg:border-accent", on);
      b.classList.toggle("text-body", !on);
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

  function currentLogoUrl() {
    if (logoFile && logoObjectUrl) return logoObjectUrl;
    if (existingLogoExt && currentId && cfg.logoServe) {
      return cfg.logoServe + "?scope=" + encodeURIComponent(scope) + "&id=" + encodeURIComponent(currentId) + "&t=" + logoCacheBust;
    }
    return null;
  }
  var logoCacheBust = Date.now();

  function populate(theme) {
    var t = theme ? clone(theme) : clone(FOREST);
    if (!t.colors) t = clone(FOREST);
    currentId = theme && theme.id ? theme.id : "";
    existingLogoExt = t.logo_ext || "";
    logoFile = null;
    if (logoObjectUrl) { try { URL.revokeObjectURL(logoObjectUrl); } catch (e) {} logoObjectUrl = null; }
    logoCacheBust = Date.now();
    el("ste-label").value = t.label || "";
    el("ste-label").style.borderColor = "";
    COLOR_FIELDS.forEach(function (f) { var i = body.querySelector('[data-color="' + f[0] + '"]'); if (i) i.value = t.colors[f[0]] || "#000000"; });
    FONT_FIELDS.forEach(function (f) {
      var i = body.querySelector('[data-font="' + f[0] + '"]');
      if (i) { var fam = (t.fonts && t.fonts[f[0]]) || "Carlito"; ensureFontOption(i, fam); i.value = fam; }
    });
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
    el("ste-logo-height").value = footer.logo_height || 20;
    var bg = footer.bg_color || "";
    el("ste-footer-bg-on").checked = !!bg;
    el("ste-footer-bg").value = (bg && bg[0] === "#") ? bg : "#1F3D30";
    syncAllSwatches();
    if (existingLogoExt && currentId && cfg.logoServe) {
      showLogoPreview(currentLogoUrl());
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
    t.footer.logo_height = Math.max(8, Math.min(48, parseInt(el("ste-logo-height").value, 10) || 20));
    return t;
  }

  var previewTimer = null;
  function schedulePreview() { clearTimeout(previewTimer); previewTimer = setTimeout(renderPreview, 60); }
  function renderPreview() {
    var t = collect();
    el("ste-preview").innerHTML = miniSlide(t, { height: 150, logoUrl: currentLogoUrl(), showBullet: true, showFooter: true });
    renderSummary(t);
  }
  function summaryRow(label, val) {
    return '<div class="flex justify-between text-xs"><span class="text-body-subtle">' + label + '</span>' +
      '<span class="font-mono text-[11px] text-body">' + esc(val) + '</span></div>';
  }
  function footerSummary(secs) {
    var pick = null;
    (secs || []).forEach(function (s) { if (s.content && s.content !== "none") pick = s; });
    if (!pick) return "—";
    var contentLabel = { logo: "logo", text: "text", page: "page" }[pick.content] || pick.content;
    return contentLabel + " · " + (pick.align || "left");
  }
  function renderSummary(t) {
    var ty = t.typography || {}, fo = t.fonts || {};
    el("ste-summary").innerHTML =
      summaryRow("Headline", (fo.headline || "—") + " · " + (ty.headline_size || "")) +
      summaryRow("Body", (fo.body || "—") + " · " + (ty.body_size || "")) +
      summaryRow("Footer", footerSummary(t.footer && t.footer.sections));
  }

  function setStatus(msg, isErr) { statusEl.textContent = msg || ""; statusEl.className = "text-xs mr-auto " + (isErr ? "text-fg-danger" : "text-body-subtle"); }

  function postJSON(url, obj) {
    return fetch(url, { method: "POST", headers: { "X-CSRFToken": cfg.csrf, "Content-Type": "application/json" }, credentials: "same-origin", body: JSON.stringify(obj) })
      .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); });
  }

  function save() {
    var theme = collect();
    if (!theme.label) {
      setStatus("Give the theme a name.", true);
      showSection("general");
      var li = el("ste-label");
      if (li) { li.style.borderColor = "#B23A2E"; try { li.scrollIntoView({ block: "center" }); } catch (e) {} li.focus(); }
      return;
    }
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
    if (scopeEl) scopeEl.textContent = scope === "org" ? "Organization" : "Yours";
    setStatus("", false);
    showSection("general");
    populate(theme);
    show();
  }

  document.addEventListener("DOMContentLoaded", function () {
    modal = el("ste-modal");
    if (!modal) return;  // page didn't include the editor partial
    body = el("ste-body");
    titleEl = el("ste-title");
    scopeEl = el("ste-scope");
    statusEl = el("ste-status");
    cfg = {
      csrf: modal.dataset.csrf, save: modal.dataset.saveUrl, logoUpload: modal.dataset.logoUploadUrl,
      logoDelete: modal.dataset.logoDeleteUrl, logoServe: modal.dataset.logoServeUrl,
      fontsUpload: modal.dataset.fontsUploadUrl, isOrgAdmin: modal.dataset.isOrgAdmin === "1"
    };
    try { uploadedFonts = JSON.parse(modal.dataset.fonts || "[]") || []; } catch (e) { uploadedFonts = []; }
    buildForm();
    el("ste-close").addEventListener("click", hide);
    el("ste-cancel").addEventListener("click", hide);
    el("ste-save").addEventListener("click", save);
    modal.addEventListener("click", function (e) { if (e.target === modal) hide(); });
    window.SlideThemeEditor = { open: open, miniSlide: miniSlide };
  });
})();
