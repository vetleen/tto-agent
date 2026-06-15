// Reusable CodeMirror 6 markdown editor for Wilfred.
//
// Bundled by esbuild into static/js/editor.bundle.js and exposed as
// window.WilfredEditor. Used by the chat Canvas (templates/chat/chat.html) and
// the skill / skill-template editors (static/js/skills-detail.js).
//
// Design notes:
//   * Markdown-source mode — the editor value is always a plain markdown string,
//     so the existing AI-write-back / diff / checkpoint pipeline is untouched.
//   * setValue() is "silent": it mirrors how `textarea.value = x` does NOT fire
//     `input`, so programmatic writes never trigger autosave loops or undo churn.
//   * maxLength is enforced by a transactionFilter; setValue() pre-truncates.
//   * Optional formatting toolbar (opts.toolbar) + ⌘/Ctrl-B/I/K shortcuts, link
//     smart-paste, and hover previews for footnotes/links are shared by every
//     instance, so the editor looks and behaves the same everywhere.
//
// Styling note: DOM the component builds itself is styled with INLINE styles for
// layout and with Tailwind classes ONLY where those exact classes already appear
// in templates (so they're guaranteed present in output.css regardless of whether
// this source file is scanned).

import { EditorState, Compartment, Annotation, Transaction } from "@codemirror/state";
import {
  EditorView,
  keymap,
  drawSelection,
  hoverTooltip,
  tooltips,
  placeholder as placeholderExt,
} from "@codemirror/view";
import { history, historyKeymap, defaultKeymap, indentWithTab } from "@codemirror/commands";
import { bracketMatching } from "@codemirror/language";
import {
  markdown,
  markdownLanguage,
  insertNewlineContinueMarkup,
  deleteMarkupBackward,
} from "@codemirror/lang-markdown";
import { unifiedMergeView } from "@codemirror/merge";

// Marks a transaction as a programmatic (non-user) document change.
const External = Annotation.define();

// The edit view shows plain, unstyled markdown *source* (uniform monospace, no
// rendered formatting) — rendered output lives only in the preview.
const MONO = "'Maple Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";

const IS_MAC =
  typeof navigator !== "undefined" &&
  /Mac|iPhone|iPod|iPad/.test(navigator.platform || navigator.userAgent || "");
function shortcutLabel(modKey) {
  return (IS_MAC ? "⌘" : "Ctrl+") + modKey.toUpperCase();
}

// ---------------------------------------------------------------------------
// Formatting commands (operate on the main selection, dispatch user edits).
// ---------------------------------------------------------------------------
function wrapCmd(before, after) {
  return (view) => {
    const { state } = view;
    const sel = state.selection.main;
    const from = sel.from, to = sel.to;
    const bl = before.length, al = after.length;
    const inner = state.sliceDoc(from, to);

    // Toggle off: markers immediately outside the selection.
    if (
      from - bl >= 0 && to + al <= state.doc.length &&
      state.sliceDoc(from - bl, from) === before &&
      state.sliceDoc(to, to + al) === after
    ) {
      view.dispatch({
        changes: [
          { from: from - bl, to: from, insert: "" },
          { from: to, to: to + al, insert: "" },
        ],
        selection: { anchor: from - bl, head: to - bl },
      });
      return true;
    }
    // Toggle off: markers inside the selection.
    if (inner.length >= bl + al && inner.slice(0, bl) === before && inner.slice(inner.length - al) === after) {
      const stripped = inner.slice(bl, inner.length - al);
      view.dispatch({
        changes: { from, to, insert: stripped },
        selection: { anchor: from, head: from + stripped.length },
      });
      return true;
    }
    // Wrap — but hug the trimmed core so any leading/trailing whitespace in the
    // selection stays OUTSIDE the markers. Otherwise `** text **` is invalid
    // markdown and the literal stars leak into the rendered output.
    const lead = inner.length - inner.trimStart().length;
    const trail = inner.length - inner.trimEnd().length;
    if (to > from && inner.length - lead - trail > 0) {
      const coreFrom = from + lead;
      const coreTo = to - trail;
      view.dispatch({
        changes: [
          { from: coreFrom, insert: before },
          { from: coreTo, insert: after },
        ],
        selection: { anchor: coreFrom + bl, head: coreTo + bl },
      });
      return true;
    }
    // Empty (or all-whitespace) selection: insert markers, cursor between.
    view.dispatch({
      changes: { from, insert: before + after },
      selection: { anchor: from + bl },
    });
    return true;
  };
}

function linkCmd(view) {
  const { state } = view;
  const sel = state.selection.main;
  const text = state.sliceDoc(sel.from, sel.to);
  if (text) {
    const insert = "[" + text + "](url)";
    const urlStart = sel.from + 1 + text.length + 2; // after "[text]("
    view.dispatch({
      changes: { from: sel.from, to: sel.to, insert },
      selection: { anchor: urlStart, head: urlStart + 3 }, // select "url"
    });
  } else {
    view.dispatch({
      changes: { from: sel.from, insert: "[text](url)" },
      selection: { anchor: sel.from + 1, head: sel.from + 5 }, // select "text"
    });
  }
  return true;
}

function prefixCmd(prefix) {
  return (view) => {
    const { state } = view;
    const sel = state.selection.main;
    const first = state.doc.lineAt(sel.from).number;
    const last = state.doc.lineAt(sel.to).number;
    let allPrefixed = true;
    for (let n = first; n <= last; n++) {
      if (!state.doc.line(n).text.startsWith(prefix)) { allPrefixed = false; break; }
    }
    const changes = [];
    for (let n = first; n <= last; n++) {
      const line = state.doc.line(n);
      if (allPrefixed) {
        changes.push({ from: line.from, to: line.from + prefix.length, insert: "" });
      } else if (!line.text.startsWith(prefix)) {
        changes.push({ from: line.from, insert: prefix });
      }
    }
    if (changes.length) view.dispatch({ changes });
    return true;
  };
}

// Heading toggles to "## " — strips any existing leading #'s first, and removes
// the heading entirely if every selected line is already a level-2 heading.
function headingCmd(view) {
  const { state } = view;
  const sel = state.selection.main;
  const first = state.doc.lineAt(sel.from).number;
  const last = state.doc.lineAt(sel.to).number;
  const lead = /^#{1,6}\s+/;
  let allH2 = true;
  for (let n = first; n <= last; n++) {
    if (!/^##\s/.test(state.doc.line(n).text)) { allH2 = false; break; }
  }
  const changes = [];
  for (let n = first; n <= last; n++) {
    const line = state.doc.line(n);
    const m = line.text.match(lead);
    const existing = m ? m[0].length : 0;
    changes.push({ from: line.from, to: line.from + existing, insert: allH2 ? "" : "## " });
  }
  view.dispatch({ changes });
  return true;
}

// Fenced code block ("Plain block"): wraps the selection in ``` fences on their own
// lines, or — with no selection — drops an empty fenced block with the cursor inside.
function cmdPlainBlock(view) {
  const { state } = view;
  const sel = state.selection.main;
  if (sel.empty) {
    const pos = sel.from;
    const line = state.doc.lineAt(pos);
    const lead = pos === line.from ? "" : "\n";
    const tail = pos === line.to ? "" : "\n";
    view.dispatch({
      changes: { from: pos, insert: lead + "```\n\n```" + tail },
      selection: { anchor: pos + lead.length + 4 }, // the empty line between the fences
    });
    return true;
  }
  const from = sel.from, to = sel.to;
  const selected = state.sliceDoc(from, to);
  const lead = from === state.doc.lineAt(from).from ? "" : "\n";
  const tail = to === state.doc.lineAt(to).to ? "" : "\n";
  const innerStart = from + lead.length + 4; // after lead + "```\n"
  view.dispatch({
    changes: { from, to, insert: lead + "```\n" + selected + "\n```" + tail },
    selection: { anchor: innerStart, head: innerStart + selected.length },
  });
  return true;
}

const cmdBold = wrapCmd("**", "**");
const cmdItalic = wrapCmd("*", "*");
const cmdStrike = wrapCmd("~~", "~~");
const cmdLink = linkCmd;
const cmdHeading = headingCmd;
const cmdUl = prefixCmd("- ");
const cmdOl = prefixCmd("1. ");
const cmdQuote = prefixCmd("> ");

// ---------------------------------------------------------------------------
// Toolbar.
// ---------------------------------------------------------------------------
function glyph(content, style) {
  return (
    '<span style="display:inline-flex;align-items:center;justify-content:center;width:1rem;height:1rem;' +
    (style || "") +
    '">' + content + "</span>"
  );
}
const SVG_LINK =
  '<svg style="width:1rem;height:1rem" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.658 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"/></svg>';
const SVG_UL =
  '<svg style="width:1rem;height:1rem" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/></svg>';
const SVG_PLAINBLOCK =
  '<svg style="width:1rem;height:1rem" fill="none" stroke="currentColor" viewBox="0 0 24 24"><rect x="3.5" y="5" width="17" height="14" rx="2" stroke-width="2"/><path stroke-linecap="round" stroke-width="2" d="M7 9.5h7M7 12.5h10M7 15.5h5"/></svg>';

// Right-side action icons. The save button swaps between four states by toggling
// `.hidden` on these siblings — the exact floppy/spinner/check/cross set the chat
// canvas uses, so the affordance reads identically across the app.
const SVG_SAVE =
  '<svg class="cm-btn-icon" style="width:1rem;height:1rem" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 21H7a2 2 0 01-2-2V5a2 2 0 012-2h7l5 5v11a2 2 0 01-2 2z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 21v-8H7v8M7 3v5h8"/></svg>';
const SVG_SPINNER =
  '<svg class="cm-btn-spinner animate-spin hidden" style="width:1rem;height:1rem" fill="none" viewBox="0 0 24 24"><circle style="opacity:0.25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path style="opacity:0.75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path></svg>';
const SVG_CHECK =
  '<svg class="cm-btn-check hidden text-fg-success" style="width:1rem;height:1rem" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7"/></svg>';
const SVG_ERROR =
  '<svg class="cm-btn-error hidden text-fg-danger" style="width:1rem;height:1rem" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M6 18L18 6M6 6l12 12"/></svg>';
const SVG_EYE =
  '<svg style="width:1rem;height:1rem" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/></svg>';

const TB = [
  { title: "Bold", mod: "b", run: cmdBold, html: glyph("B", "font-weight:700") },
  { title: "Italic", mod: "i", run: cmdItalic, html: glyph("I", "font-style:italic;font-weight:600") },
  { title: "Strikethrough", run: cmdStrike, html: glyph("S", "text-decoration:line-through;font-weight:600") },
  { sep: true },
  { title: "Link", mod: "k", run: cmdLink, html: SVG_LINK },
  { title: "Heading", run: cmdHeading, html: glyph("H", "font-weight:700") },
  { title: "Bulleted list", run: cmdUl, html: SVG_UL },
  { title: "Numbered list", run: cmdOl, html: glyph("1.", "font-weight:600;font-size:11px") },
  { title: "Quote", run: cmdQuote, html: glyph("”", "font-weight:700;font-size:16px") },
  { title: "Plain block", run: cmdPlainBlock, html: SVG_PLAINBLOCK },
];

function populateToolbar(toolbar, view) {
  TB.forEach((b) => {
    if (b.sep) {
      const sep = document.createElement("span");
      sep.style.cssText = "width:1px;height:18px;margin:0 4px;background:currentColor;opacity:0.16;";
      toolbar.appendChild(sep);
      return;
    }
    const btn = document.createElement("button");
    btn.type = "button"; // never submit the surrounding skills <form>
    // Themed classes below all already appear in templates → present in output.css.
    btn.className = "rounded-base text-body hover:text-heading hover:bg-neutral-tertiary";
    btn.style.cssText =
      "cursor:pointer;border:none;background:transparent;padding:5px;display:inline-flex;align-items:center;justify-content:center;";
    btn.title = b.mod ? b.title + " (" + shortcutLabel(b.mod) + ")" : b.title;
    btn.innerHTML = b.html;
    btn.addEventListener("mousedown", (e) => {
      e.preventDefault(); // keep focus + selection in the editor
      b.run(view);
      view.focus();
    });
    toolbar.appendChild(btn);
  });
}

// ---------------------------------------------------------------------------
// Right-side toolbar actions (save / preview) — shared so every editor with the
// internal toolbar frame gets the same affordances as the chat canvas.
// ---------------------------------------------------------------------------

// Square icon button matching the canvas document-action buttons. Themed via
// template-proven classes; layout via inline styles.
function makeActionButton(title, innerHTML) {
  const btn = document.createElement("button");
  btn.type = "button"; // never submit a surrounding <form>
  btn.className = "rounded-base text-body hover:text-heading hover:bg-neutral-tertiary disabled:opacity-80";
  btn.style.cssText =
    "cursor:pointer;border:none;background:transparent;padding:6px;display:inline-flex;align-items:center;justify-content:center;";
  btn.title = title;
  btn.innerHTML = innerHTML;
  return btn;
}

// Save-button state machine: idle floppy → spinner → check/cross → idle, by
// toggling `.hidden` on the four icon siblings (mirrors the canvas).
function setBtnState(btn, state) {
  ["cm-btn-icon", "cm-btn-spinner", "cm-btn-check", "cm-btn-error"].forEach((c) => {
    const el = btn.querySelector("." + c);
    if (el) el.classList.add("hidden");
  });
  const cls =
    state === "loading" ? "cm-btn-spinner" :
    state === "success" ? "cm-btn-check" :
    state === "error" ? "cm-btn-error" : "cm-btn-icon";
  const target = btn.querySelector("." + cls);
  if (target) target.classList.remove("hidden");
  btn.disabled = state === "loading";
}

// Markdown → sanitized HTML for the preview pane. Uses the page's global marked +
// DOMPurify when present (chat / skills load them); otherwise falls back to
// escaped plain text with <br>, which is exactly right for non-markdown content
// like meeting transcripts.
function renderPreviewHtml(text) {
  if (!text) return "";
  try {
    if (window.marked && window.DOMPurify) {
      return window.DOMPurify.sanitize(window.marked.parse(text), {
        ALLOWED_TAGS: ["p", "br", "strong", "em", "u", "s", "del", "code", "pre", "ul", "ol", "li",
                       "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "a", "hr",
                       "table", "thead", "tbody", "tr", "th", "td", "div", "span", "sup", "section"],
        ALLOWED_ATTR: ["href", "title", "target", "class", "id"],
      });
    }
  } catch (e) {
    /* fall through to plain-text rendering */
  }
  return String(text)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/\n/g, "<br>");
}

// ---------------------------------------------------------------------------
// Link smart-paste: pasting a bare URL over a selection makes a markdown link.
// ---------------------------------------------------------------------------
const URL_RE = /^(https?:\/\/|mailto:)[^\s]+$/;
const smartPaste = EditorView.domEventHandlers({
  paste(event, view) {
    const cd = event.clipboardData;
    const text = cd && cd.getData("text/plain");
    if (!text) return false;
    const url = text.trim();
    const sel = view.state.selection.main;
    if (sel.empty || !URL_RE.test(url)) return false;
    const selected = view.state.sliceDoc(sel.from, sel.to);
    view.dispatch({
      changes: { from: sel.from, to: sel.to, insert: "[" + selected + "](" + url + ")" },
      selection: { anchor: sel.from + 1 + selected.length + 2 + url.length + 1 },
    });
    event.preventDefault();
    return true;
  },
});

// ---------------------------------------------------------------------------
// Hover previews: footnote references show their definition; links show the URL
// with a clickable "Open".
// ---------------------------------------------------------------------------
function matchAround(text, off, re) {
  let m;
  re.lastIndex = 0;
  while ((m = re.exec(text)) !== null) {
    if (off >= m.index && off <= m.index + m[0].length) return m;
    if (m.index > off) break;
  }
  return null;
}
const footnoteRefHover = hoverTooltip((view, pos) => {
  const { state } = view;
  const line = state.doc.lineAt(pos);
  const off = pos - line.from;

  // Footnote reference [^id] (not the [^id]: definition).
  const fn = matchAround(line.text, off, /\[\^([^\]\s]+)\]/g);
  if (fn && line.text[fn.index + fn[0].length] !== ":") {
    const id = fn[1].replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const def = new RegExp("^\\[\\^" + id + "\\]:\\s?(.*)$", "m").exec(state.doc.toString());
    if (def) {
      return makeTip(line.from + fn.index, line.from + fn.index + fn[0].length, (dom) => {
        dom.textContent = def[1];
      });
    }
  }

  // Link [text](url).
  const link = matchAround(line.text, off, /\[[^\]]*\]\(([^)\s]+)\)/g);
  if (link) {
    return makeTip(line.from + link.index, line.from + link.index + link[0].length, (dom) => {
      const span = document.createElement("span");
      span.textContent = link[1] + " ";
      const a = document.createElement("a");
      a.href = link[1];
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = "Open ↗";
      a.className = "wf-tip-link";
      dom.appendChild(span);
      dom.appendChild(a);
    });
  }
  return null;
});
function makeTip(from, to, build) {
  return {
    pos: from,
    end: to,
    above: true,
    create() {
      const dom = document.createElement("div");
      dom.className = "wf-hover-tip";
      build(dom);
      return { dom };
    },
  };
}

// ---------------------------------------------------------------------------
// Themes (plain text + selection/caret/merge/tooltip styling, light & dark).
// ---------------------------------------------------------------------------
function buildTheme(v, dark) {
  return EditorView.theme(
    {
      "&": {
        backgroundColor: "transparent",
        color: v.text,
        fontSize: "0.875rem",
      },
      "&.cm-focused": { outline: "none" },
      ".cm-scroller": { fontFamily: MONO, lineHeight: "1.6", overflow: "auto" },
      ".cm-content": { padding: "12px 16px", caretColor: v.caret },
      ".cm-cursor, .cm-dropCursor": { borderLeftColor: v.caret },
      ".cm-selectionBackground, .cm-content ::selection": { backgroundColor: v.selection },
      "&.cm-focused .cm-selectionBackground": { backgroundColor: v.selection },
      ".cm-activeLine": { backgroundColor: "transparent" },
      ".cm-gutters": { backgroundColor: "transparent", border: "none", color: v.muted },
      ".cm-placeholder": { color: v.muted },
      // Inline merge view: subtle tints, controls above the change, struck deletions.
      ".cm-changedLine": { backgroundColor: v.addBg },
      ".cm-deletedChunk": { backgroundColor: v.delBg },
      ".cm-deletedChunk .cm-deletedLine del": { textDecoration: "line-through" },
      ".cm-deletedChunk .cm-chunkButtons": {
        position: "static",
        display: "flex",
        gap: "6px",
        justifyContent: "flex-end",
        margin: "2px 0 6px",
      },
      ".cm-deletedChunk .cm-chunkButtons button": {
        fontSize: "0.75rem",
        lineHeight: "1.3",
        padding: "1px 10px",
        borderRadius: "4px",
      },
      // Hover preview tooltips.
      ".cm-tooltip.cm-tooltip-hover": {
        border: "1px solid " + v.tipBorder,
        backgroundColor: v.tipBg,
        borderRadius: "6px",
        boxShadow: "0 4px 14px rgba(0,0,0,0.14)",
      },
      ".wf-hover-tip": {
        padding: "6px 10px",
        maxWidth: "360px",
        fontSize: "0.8125rem",
        lineHeight: "1.4",
        fontFamily: "inherit",
        color: v.text,
      },
      ".wf-tip-link": { color: v.link, textDecoration: "underline", whiteSpace: "nowrap" },
    },
    { dark: dark }
  );
}

const LIGHT = buildTheme(
  {
    text: "#283044",                       // slate-800
    caret: "#0B2418",                      // forest-900
    selection: "rgba(190, 130, 66, 0.20)", // copper
    muted: "#939BAE",                      // slate-400
    addBg: "rgba(26, 146, 85, 0.12)",      // success
    delBg: "rgba(178, 59, 54, 0.12)",      // danger
    tipBg: "#FFFFFF",                      // paper-0
    tipBorder: "#D5CEBE",                  // paper-300
    link: "#16432C",                       // forest-700
  },
  false
);

const DARK = buildTheme(
  {
    text: "#C4D6CB",
    caret: "#EFF5F1",
    selection: "rgba(210, 156, 99, 0.28)", // copper-400
    muted: "#8AA395",
    addBg: "rgba(70, 199, 126, 0.18)",     // success (dark)
    delBg: "rgba(224, 129, 123, 0.18)",    // danger (dark)
    tipBg: "#0E2719",                      // surface-card (dark)
    tipBorder: "rgba(255, 255, 255, 0.12)",
    link: "#E2BC93",                       // copper-300
  },
  true
);

// ---------------------------------------------------------------------------
// Instance registry, for app-wide theme switching.
// ---------------------------------------------------------------------------
const instances = new Set();
let currentTheme = null;

function detectTheme() {
  return document.documentElement.classList.contains("dark") ? "dark" : "light";
}
function themeExt(name) {
  return name === "dark" ? DARK : LIGHT;
}
function setTheme(name) {
  currentTheme = name === "dark" ? "dark" : "light";
  instances.forEach((inst) => {
    inst.view.dispatch({ effects: inst.themeCompartment.reconfigure(themeExt(currentTheme)) });
  });
}

// ---------------------------------------------------------------------------
// Factory.
// ---------------------------------------------------------------------------
function create(parent, opts) {
  opts = opts || {};
  const maxLength = opts.maxLength != null ? opts.maxLength : null;
  const onChange = typeof opts.onChange === "function" ? opts.onChange : null;
  const lineWrapping = opts.lineWrapping !== false;

  if (currentTheme === null) currentTheme = detectTheme();

  const themeCompartment = new Compartment();
  const readOnlyCompartment = new Compartment();
  const mergeCompartment = new Compartment();

  const clamp = (str) => {
    let doc = str == null ? "" : String(str);
    if (maxLength != null && doc.length > maxLength) doc = doc.slice(0, maxLength);
    return doc;
  };

  // Wired to the toolbar save icon below (if any) so Mod-S triggers the same save.
  let saveAction = null;
  const keyBindings = [
    { key: "Mod-b", run: cmdBold },
    { key: "Mod-i", run: cmdItalic },
    { key: "Mod-k", run: cmdLink },
    { key: "Enter", run: insertNewlineContinueMarkup },
    { key: "Backspace", run: deleteMarkupBackward },
  ];
  if (opts.onSave) {
    // Returning true preventDefaults the browser's native save dialog.
    keyBindings.push({
      key: "Mod-s",
      run: () => { if (saveAction) { saveAction(); return true; } return false; },
    });
  }
  keyBindings.push(indentWithTab, ...defaultKeymap, ...historyKeymap);

  const extensions = [
    history(),
    keymap.of(keyBindings),
    markdown({ base: markdownLanguage }),
    bracketMatching(),
    drawSelection(),
    // Fixed positioning so hover tooltips escape the canvas's overflow:hidden frame.
    tooltips({ position: "fixed" }),
    smartPaste,
    footnoteRefHover,
    themeCompartment.of(themeExt(currentTheme)),
    readOnlyCompartment.of(EditorState.readOnly.of(!!opts.readOnly)),
    mergeCompartment.of([]),
    EditorView.updateListener.of((update) => {
      if (!update.docChanged || !onChange) return;
      const isExternal = update.transactions.some((tr) => tr.annotation(External));
      if (!isExternal) onChange();
    }),
  ];

  if (lineWrapping) extensions.push(EditorView.lineWrapping);
  if (opts.placeholder) extensions.push(placeholderExt(opts.placeholder));

  // Height: fill an absolutely-sized parent (canvas) or grow with content (skills).
  if (opts.fillParent) {
    extensions.push(EditorView.theme({ "&": { height: "100%" } }));
  }
  if (opts.minHeight || opts.maxHeight) {
    const sizing = {};
    if (opts.minHeight) sizing[".cm-content"] = { minHeight: opts.minHeight };
    if (opts.maxHeight) sizing[".cm-scroller"] = { maxHeight: opts.maxHeight, overflow: "auto" };
    extensions.push(EditorView.theme(sizing));
  }
  if (maxLength != null) {
    extensions.push(
      EditorState.transactionFilter.of((tr) =>
        tr.docChanged && tr.newDoc.length > maxLength ? [] : tr
      )
    );
  }

  // Optional toolbar frame: a strip of formatting buttons above the editor, so
  // every instance shares the same recognizable chrome. Inline layout styles +
  // template-proven Tailwind classes for theming.
  let mountTarget = parent;
  let toolbarEl = null;
  let frameEl = null;
  let hostEl = null;
  if (opts.toolbarTarget) {
    // Render the formatting buttons into a caller-provided element (e.g. the canvas
    // header row, alongside its document-action icons); the editor mounts directly
    // into `parent`, with no internal frame.
    toolbarEl = opts.toolbarTarget;
  } else if (opts.toolbar) {
    frameEl = document.createElement("div");
    frameEl.className = "border-t-[3px] border-t-blue-500"; // recognizable accent
    frameEl.style.cssText =
      "display:flex;flex-direction:column;overflow:hidden;" +
      (opts.fillParent ? "height:100%;width:100%;" : "");
    toolbarEl = document.createElement("div");
    toolbarEl.className = "border-b border-default bg-neutral-primary-soft";
    toolbarEl.style.cssText =
      "display:flex;align-items:center;flex-wrap:wrap;gap:2px;padding:4px 6px;flex-shrink:0;";
    hostEl = document.createElement("div");
    hostEl.style.cssText = opts.fillParent ? "flex:1 1 0;min-height:0;position:relative;" : "";
    frameEl.appendChild(toolbarEl);
    frameEl.appendChild(hostEl);
    parent.appendChild(frameEl);
    mountTarget = hostEl;
  }

  const view = new EditorView({
    state: EditorState.create({ doc: clamp(opts.value), extensions }),
    parent: mountTarget,
  });
  if (toolbarEl) {
    if (frameEl) {
      // Internal frame: keep the format buttons in their own group so the preview
      // toggle can hide just them, then add right-aligned document actions
      // (save / preview) like the chat canvas. `margin-left:auto` pushes the
      // actions to the right edge; the toolbar's flex-wrap lets the group drop to
      // its own line on narrow widths.
      const formatGroup = document.createElement("div");
      formatGroup.style.cssText = "display:flex;align-items:center;flex-wrap:wrap;gap:2px;";
      populateToolbar(formatGroup, view);
      toolbarEl.appendChild(formatGroup);

      if (opts.onSave || opts.preview) {
        const actions = document.createElement("div");
        actions.style.cssText = "margin-left:auto;display:flex;align-items:center;gap:2px;";

        if (opts.onSave) {
          const saveBtn = makeActionButton(
            "Save" + (IS_MAC ? " (⌘S)" : " (Ctrl+S)"),
            SVG_SAVE + SVG_SPINNER + SVG_CHECK + SVG_ERROR
          );
          const doSave = () => {
            if (saveBtn.disabled) return;
            setBtnState(saveBtn, "loading");
            let result;
            try {
              result = opts.onSave(view.state.doc.toString());
            } catch (e) {
              result = Promise.reject(e);
            }
            Promise.resolve(result).then(
              () => { setBtnState(saveBtn, "success"); setTimeout(() => setBtnState(saveBtn, "idle"), 1500); },
              () => { setBtnState(saveBtn, "error"); setTimeout(() => setBtnState(saveBtn, "idle"), 2000); }
            );
          };
          saveBtn.addEventListener("click", doSave);
          saveAction = doSave; // wired to Mod-S above
          actions.appendChild(saveBtn);
        }

        if (opts.preview) {
          const previewEl = document.createElement("div");
          previewEl.className = "markdown-content hidden";
          previewEl.style.cssText =
            "overflow:auto;padding:12px 16px;font-size:0.875rem;" +
            (opts.fillParent ? "flex:1 1 0;min-height:0;" : "") +
            (opts.minHeight ? "min-height:" + opts.minHeight + ";" : "") +
            (opts.maxHeight ? "max-height:" + opts.maxHeight + ";" : "");
          frameEl.appendChild(previewEl);

          const previewBtn = makeActionButton("Toggle preview", SVG_EYE);
          let previewing = false;
          previewBtn.addEventListener("click", () => {
            previewing = !previewing;
            if (previewing) {
              previewEl.innerHTML = renderPreviewHtml(view.state.doc.toString());
              previewEl.classList.remove("hidden");
              hostEl.classList.add("hidden");
              // Set inline display, not `.hidden` — the class can't beat the
              // group's inline `display:flex`. Can't format while previewing.
              formatGroup.style.display = "none";
              previewBtn.classList.add("text-heading", "bg-neutral-tertiary");
              previewBtn.title = "Back to editor";
            } else {
              previewEl.classList.add("hidden");
              hostEl.classList.remove("hidden");
              formatGroup.style.display = "flex";
              previewBtn.classList.remove("text-heading", "bg-neutral-tertiary");
              previewBtn.title = "Toggle preview";
              view.focus();
            }
          });
          actions.appendChild(previewBtn);
        }

        toolbarEl.appendChild(actions);
      }
    } else {
      populateToolbar(toolbarEl, view);
    }
  }
  if (opts.readOnly) view.dom.classList.add("cm-readonly");

  const inst = { view, themeCompartment };
  instances.add(inst);

  return {
    view,
    dom: view.dom,

    getValue() {
      return view.state.doc.toString();
    },

    // Silent: no onChange, excluded from undo history, selection reset to start,
    // pre-truncated so the maxLength filter never rejects it.
    setValue(str) {
      const doc = clamp(str);
      view.dispatch({
        changes: { from: 0, to: view.state.doc.length, insert: doc },
        selection: { anchor: 0 },
        annotations: [External.of(true), Transaction.addToHistory.of(false)],
      });
    },

    setReadOnly(ro) {
      view.dispatch({ effects: readOnlyCompartment.reconfigure(EditorState.readOnly.of(!!ro)) });
      view.dom.classList.toggle("cm-readonly", !!ro);
    },

    enableMerge(original) {
      view.dispatch({
        effects: mergeCompartment.reconfigure(
          unifiedMergeView({ original: original == null ? "" : String(original) })
        ),
      });
    },

    disableMerge() {
      view.dispatch({ effects: mergeCompartment.reconfigure([]) });
    },

    focus() {
      view.focus();
    },

    destroy() {
      instances.delete(inst);
      view.destroy();
    },
  };
}

window.WilfredEditor = { create, setTheme, detectTheme };
