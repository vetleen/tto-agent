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
// rendered formatting) — rendered output lives only in the preview. So there is
// deliberately no HighlightStyle; the markdown() language is kept solely for
// editing affordances (list/quote continuation on Enter).
const MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";

// ---------------------------------------------------------------------------
// Themes (plain text + selection/caret/merge tints for light and dark).
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
      // Inline merge view (Phase 2): subtle tints, per-hunk controls on their own
      // line above the change (not overlapping the text), struck-through deletions.
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

  // Height: either fill an absolutely-sized parent (canvas) or grow with content
  // between min/max (skill fields).
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
