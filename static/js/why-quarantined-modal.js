/**
 * "Why was this quarantined?" modal behaviour: buttons with class
 * .why-quarantined-btn carry the explanation as data attributes; on click we fill
 * the shared #why-quarantined-modal and trigger it via the hidden Flowbite trigger
 * (same pattern as rename-modal.js).
 *
 * data-intro  : generic sentence shown to every quarantined doc.
 * data-detail : the PII reviewer's specific finding (present for full/PII
 *               quarantine only; absent for guardrails/partial quarantine).
 *
 * Text is set with textContent (never innerHTML) so reviewer-authored content
 * cannot inject markup.
 */
(function () {
  var modal = document.getElementById("why-quarantined-modal");
  if (!modal) return;
  var trigger = document.querySelector('[data-modal-target="why-quarantined-modal"]');
  var introEl = modal.querySelector("#why-quarantined-intro");
  var detailWrap = modal.querySelector("#why-quarantined-detail-wrap");
  var detailEl = modal.querySelector("#why-quarantined-detail");

  document.querySelectorAll(".why-quarantined-btn").forEach(function (btn) {
    btn.addEventListener("click", function (e) {
      e.preventDefault();
      e.stopPropagation();
      if (introEl) introEl.textContent = btn.getAttribute("data-intro") || "";
      var detail = btn.getAttribute("data-detail") || "";
      if (detail && detailEl && detailWrap) {
        detailEl.textContent = detail;
        detailWrap.classList.remove("hidden");
      } else if (detailWrap) {
        if (detailEl) detailEl.textContent = "";
        detailWrap.classList.add("hidden");
      }
      if (trigger) {
        setTimeout(function () { trigger.click(); }, 0);
      }
    });
  });
})();
