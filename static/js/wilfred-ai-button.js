(function () {
  'use strict';

  // Wilfred AI-trigger buttons that navigate away on submit (e.g. "Summarize
  // meeting in chat", "Edit with Wilfred"). On submit we swap the twinkling
  // constellation for a spinner and show a present-tense label until the new
  // page loads. The async "Ask Wilfred" description button reverts instead of
  // navigating, so it has its own handler in data-room-description.js.
  //
  // Markup contract on the trigger button:
  //   data-wf-ai-trigger            marks the button
  //   data-busy-label="Summarizing…" present-tense label shown while waiting
  //   <span data-wf-ai-icon>…stars…</span>      idle icon
  //   <span data-wf-ai-spinner class="hidden">  waiting spinner
  //   <span data-wf-ai-label>Summarize…</span>  the label to swap
  // Any sibling control in the same form marked data-wf-ai-sibling (e.g. a
  // split-button caret) is dimmed alongside it.

  function enterWaiting(btn) {
    var icon = btn.querySelector('[data-wf-ai-icon]');
    var spinner = btn.querySelector('[data-wf-ai-spinner]');
    var label = btn.querySelector('[data-wf-ai-label]');
    if (icon) icon.classList.add('hidden');
    if (spinner) spinner.classList.remove('hidden');
    if (label && btn.dataset.busyLabel) label.textContent = btn.dataset.busyLabel;
    // Use the pointer-events:none modifier rather than `disabled` so the native
    // form submission isn't cancelled and the button's value still posts.
    btn.classList.add('wf-ai-waiting');
    btn.setAttribute('aria-busy', 'true');
  }

  document.querySelectorAll('form [data-wf-ai-trigger]').forEach(function (btn) {
    var form = btn.closest('form');
    if (!form) return;
    form.addEventListener('submit', function () {
      enterWaiting(btn);
      form.querySelectorAll('[data-wf-ai-sibling]').forEach(function (el) {
        el.classList.add('wf-ai-waiting');
        el.setAttribute('aria-busy', 'true');
      });
    });
  });
})();
