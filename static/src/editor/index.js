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
const MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";

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
    // Wrap.
    view.dispatch({
      changes: { from, to, insert: before + inner + after },
      selection: to > from ? { anchor: from + bl, head: to + bl } : { anchor: from + bl },
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

const cmdBold = wrapCmd("**", "**");
const cmdItalic = wrapCmd("*", "*");
const cmdStrike = wrapCmd("~~", "~~");
const cmdCode = wrapCmd("`", "`");
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
  { title: "Inline code", run: cmdCode, html: glyph("&lt;/&gt;", "font-family:" + MONO + ";font-size:10px") },
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
    text: "#1f2937",
    caret: "#111827",
    selection: "rgba(37, 99, 235, 0.15)",
    muted: "#9ca3af",
    addBg: "rgba(34, 197, 94, 0.12)",
    delBg: "rgba(239, 68, 68, 0.12)",
    tipBg: "#ffffff",
    tipBorder: "#e5e7eb",
    link: "#2563eb",
  },
  false
);

const DARK = buildTheme(
  {
    text: "#e5e7eb",
    caret: "#f9fafb",
    selection: "rgba(96, 165, 250, 0.28)",
    muted: "#6b7280",
    addBg: "rgba(34, 197, 94, 0.18)",
    delBg: "rgba(239, 68, 68, 0.18)",
    tipBg: "#1f2937",
    tipBorder: "#374151",
    link: "#60a5fa",
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

  const extensions = [
    history(),
    keymap.of([
      { key: "Mod-b", run: cmdBold },
      { key: "Mod-i", run: cmdItalic },
      { key: "Mod-k", run: cmdLink },
      { key: "Enter", run: insertNewlineContinueMarkup },
      { key: "Backspace", run: deleteMarkupBackward },
      indentWithTab,
      ...defaultKeymap,
      ...historyKeymap,
    ]),
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
  if (opts.toolbar) {
    const frame = document.createElement("div");
    frame.className = "border-t-[3px] border-t-blue-500"; // recognizable accent
    frame.style.cssText =
      "display:flex;flex-direction:column;overflow:hidden;" +
      (opts.fillParent ? "height:100%;width:100%;" : "");
    toolbarEl = document.createElement("div");
    toolbarEl.className = "border-b border-default bg-neutral-primary-soft";
    toolbarEl.style.cssText =
      "display:flex;align-items:center;flex-wrap:wrap;gap:2px;padding:4px 6px;flex-shrink:0;";
    const host = document.createElement("div");
    host.style.cssText = opts.fillParent ? "flex:1 1 0;min-height:0;position:relative;" : "";
    frame.appendChild(toolbarEl);
    frame.appendChild(host);
    parent.appendChild(frame);
    mountTarget = host;
  }

  const view = new EditorView({
    state: EditorState.create({ doc: clamp(opts.value), extensions }),
    parent: mountTarget,
  });
  if (toolbarEl) populateToolbar(toolbarEl, view);
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
