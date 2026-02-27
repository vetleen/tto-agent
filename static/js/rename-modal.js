/**
 * Shared rename modal behaviour: buttons with data-rename-modal open the matching modal,
 * set the form action and input from data-action and data-name, then trigger the modal.
 */
(function() {
  document.querySelectorAll('[data-rename-modal]').forEach(function(btn) {
    var modalId = btn.getAttribute('data-rename-modal');
    var modal = document.getElementById(modalId);
    if (!modal) return;
    var trigger = document.querySelector('[data-modal-target="' + modalId + '"]');
    var form = modal.querySelector('form');
    var input = modal.querySelector('input[type="text"]');
    if (!form || !input) return;
    btn.addEventListener('click', function(e) {
      e.preventDefault();
      e.stopPropagation();
      form.action = btn.getAttribute('data-action') || '';
      input.value = btn.getAttribute('data-name') || '';
      input.focus();
      if (trigger) {
        setTimeout(function() { trigger.click(); }, 0);
      }
    });
  });
})();
