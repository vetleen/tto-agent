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
//     `input`, so programmatic writes (AI updates, revert, restore, import,
//     canvas switch) never trigger autosave loops or pollute undo history.
//   * maxLength is enforced by a transactionFilter; setValue() pre-truncates so
//     programmatic over-cap content (legacy >cap docs, imports) is never rejected.

import { EditorState, Compartment, Annotation, Transaction } from "@codemirror/state";
import {
  EditorView,
  keymap,
  drawSelection,
  placeholder as placeholderExt,
} from "@codemirror/view";
import { history, historyKeymap, defaultKeymap, indentWithTab } from "@codemirror/commands";
import { syntaxHighlighting, HighlightStyle, bracketMatching } from "@codemirror/language";
import {
  markdown,
  markdownLanguage,
  insertNewlineContinueMarkup,
  deleteMarkupBackward,
} from "@codemirror/lang-markdown";
import { unifiedMergeView } from "@codemirror/merge";
import { tags as t } from "@lezer/highlight";

// Marks a transaction as a programmatic (non-user) document change.
const External = Annotation.define();

// ---------------------------------------------------------------------------
// Live-styled markdown highlighting. Structural styling (size/weight/family)
// is fixed; colors reference CSS vars set per-theme below, so one HighlightStyle
// works for both light and dark.
// ---------------------------------------------------------------------------
const MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";

const wilfredHighlight = HighlightStyle.define([
  { tag: t.heading1, fontSize: "1.5em", fontWeight: "700", color: "var(--md-head)" },
  { tag: t.heading2, fontSize: "1.3em", fontWeight: "700", color: "var(--md-head)" },
  { tag: t.heading3, fontSize: "1.15em", fontWeight: "700", color: "var(--md-head)" },
  { tag: [t.heading4, t.heading5, t.heading6], fontWeight: "700", color: "var(--md-head)" },
  { tag: t.strong, fontWeight: "700" },
  { tag: t.emphasis, fontStyle: "italic" },
  { tag: t.strikethrough, textDecoration: "line-through" },
  { tag: t.monospace, fontFamily: MONO, color: "var(--md-code)" },
  { tag: [t.link, t.url], color: "var(--md-link)", textDecoration: "underline" },
  { tag: t.quote, color: "var(--md-quote)", fontStyle: "italic" },
  { tag: t.list, color: "var(--md-head)" },
  { tag: [t.processingInstruction, t.contentSeparator], color: "var(--md-muted)" },
]);

// ---------------------------------------------------------------------------
// Themes. Colors live in CSS vars on the editor root so the HighlightStyle can
// reference them; switching theme just reconfigures these.
// ---------------------------------------------------------------------------
function buildTheme(v, dark) {
  return EditorView.theme(
    {
      "&": {
        height: "100%",
        backgroundColor: "transparent",
        color: v.text,
        fontSize: "0.875rem",
        "--md-head": v.head,
        "--md-link": v.link,
        "--md-code": v.code,
        "--md-quote": v.quote,
        "--md-muted": v.muted,
      },
      "&.cm-focused": { outline: "none" },
      ".cm-scroller": { fontFamily: "inherit", lineHeight: "1.6", overflow: "auto" },
      ".cm-content": { padding: "12px 16px", caretColor: v.caret },
      ".cm-cursor, .cm-dropCursor": { borderLeftColor: v.caret },
      ".cm-selectionBackground, .cm-content ::selection": { backgroundColor: v.selection },
      "&.cm-focused .cm-selectionBackground": { backgroundColor: v.selection },
      ".cm-activeLine": { backgroundColor: "transparent" },
      ".cm-gutters": { backgroundColor: "transparent", border: "none", color: v.muted },
      ".cm-placeholder": { color: v.muted },
      // Merge view (Phase 2) tints.
      ".cm-deletedChunk": { backgroundColor: v.delBg },
      ".cm-changedLine": { backgroundColor: v.addBg },
    },
    { dark: dark }
  );
}

const LIGHT = buildTheme(
  {
    text: "#1f2937",
    caret: "#111827",
    selection: "rgba(37, 99, 235, 0.15)",
    head: "#111827",
    link: "#2563eb",
    code: "#be185d",
    quote: "#6b7280",
    muted: "#9ca3af",
    addBg: "rgba(34, 197, 94, 0.12)",
    delBg: "rgba(239, 68, 68, 0.12)",
  },
  false
);

const DARK = buildTheme(
  {
    text: "#e5e7eb",
    caret: "#f9fafb",
    selection: "rgba(96, 165, 250, 0.28)",
    head: "#f3f4f6",
    link: "#60a5fa",
    code: "#f0abfc",
    quote: "#9ca3af",
    muted: "#6b7280",
    addBg: "rgba(34, 197, 94, 0.18)",
    delBg: "rgba(239, 68, 68, 0.18)",
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
      { key: "Enter", run: insertNewlineContinueMarkup },
      { key: "Backspace", run: deleteMarkupBackward },
      indentWithTab,
      ...defaultKeymap,
      ...historyKeymap,
    ]),
    markdown({ base: markdownLanguage }),
    syntaxHighlighting(wilfredHighlight),
    bracketMatching(),
    drawSelection(),
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
  if (maxLength != null) {
    extensions.push(
      // Reject any (non-programmatic) transaction that would push past the cap.
      EditorState.transactionFilter.of((tr) =>
        tr.docChanged && tr.newDoc.length > maxLength ? [] : tr
      )
    );
  }

  const view = new EditorView({
    state: EditorState.create({ doc: clamp(opts.value), extensions }),
    parent: parent,
  });

  if (opts.readOnly) view.dom.classList.add("cm-readonly");

  const inst = { view, themeCompartment };
  instances.add(inst);

  return {
    view,
    dom: view.dom,

    getValue() {
      return view.state.doc.toString();
    },

    // Silent: does not fire onChange, is excluded from undo history, resets the
    // selection to the start (avoids a RangeError from a stale offset past the
    // new doc), and pre-truncates so the maxLength filter never rejects it.
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

    // Phase 2: show an inline per-hunk diff against `original`.
    enableMerge(original) {
      view.dispatch({
        effects: mergeCompartment.reconfigure(unifiedMergeView({ original: original == null ? "" : String(original) })),
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
