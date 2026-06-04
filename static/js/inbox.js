(function () {
  'use strict';

  var config = document.getElementById('inbox-config');
  if (!config) return;

  var renewUrl = config.dataset.renewUrl;
  var csrf = config.dataset.csrfToken || document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');

  // ── Show-archived toggle ──────────────────────────────────────────
  var archivedToggle = document.getElementById('show-archived-toggle');
  if (archivedToggle) {
    archivedToggle.addEventListener('change', function () {
      var url = new URL(window.location.href);
      if (archivedToggle.checked) {
        url.searchParams.set('show_archived', '1');
      } else {
        url.searchParams.delete('show_archived');
      }
      url.searchParams.delete('page'); // reset to first page when toggling
      window.location.href = url.toString();
    });
  }

  // ── Selection + bulk actions ──────────────────────────────────────
  var selectAll = document.getElementById('select-all-inbox');
  var bulkActions = document.getElementById('inbox-bulk-actions');
  var countEl = document.getElementById('inbox-selection-count');
  var checkboxes = Array.prototype.slice.call(
    document.querySelectorAll('[data-list="inbox"] .inbox-checkbox')
  );

  function getSelectedKeys() {
    var keys = [];
    checkboxes.forEach(function (cb) {
      if (cb.checked) {
        var row = cb.closest('[data-item-key]');
        if (row) keys.push(row.dataset.itemKey);
      }
    });
    return keys;
  }

  function updateUI() {
    var checked = checkboxes.filter(function (cb) { return cb.checked; }).length;
    var total = checkboxes.length;
    if (selectAll) {
      selectAll.checked = total > 0 && checked === total;
      selectAll.indeterminate = checked > 0 && checked < total;
    }
    if (!bulkActions || !countEl) return;
    if (checked > 0) {
      bulkActions.classList.remove('hidden');
      bulkActions.classList.add('flex');
      countEl.classList.remove('hidden');
      countEl.querySelector('.count').textContent = checked;
    } else {
      bulkActions.classList.add('hidden');
      bulkActions.classList.remove('flex');
      countEl.classList.add('hidden');
    }
  }

  if (selectAll) {
    selectAll.addEventListener('change', function () {
      checkboxes.forEach(function (cb) { cb.checked = selectAll.checked; });
      updateUI();
    });
  }
  checkboxes.forEach(function (cb) { cb.addEventListener('change', updateUI); });

  // ── Renew requests ────────────────────────────────────────────────
  function setButtonLoading(btn, loading) {
    if (!btn) return;
    if (loading) {
      btn.disabled = true;
      btn.dataset.origHtml = btn.innerHTML;
      btn.innerHTML = '<svg class="w-3.5 h-3.5 animate-spin inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/></svg> Renewing…';
    } else {
      btn.disabled = false;
      if (btn.dataset.origHtml) btn.innerHTML = btn.dataset.origHtml;
    }
  }

  function doFetch(url, body) {
    return fetch(url, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrf },
      body: JSON.stringify(body)
    });
  }

  function renew(keys, btn) {
    if (!keys.length) return;
    setButtonLoading(btn, true);
    doFetch(renewUrl, { items: keys })
      .then(function (r) {
        if (!r.ok) throw new Error('Renew failed (' + r.status + ')');
        return r.json();
      })
      .then(function () {
        // Renewed items leave the 30-day window — reload to reflect the new list.
        window.location.reload();
      })
      .catch(function () {
        setButtonLoading(btn, false);
        alert('Could not renew right now. Please try again.');
      });
  }

  // Per-row renew (event delegation so it survives any future re-render).
  document.addEventListener('click', function (e) {
    if (!e.target.closest) return;
    var btn = e.target.closest('.inbox-renew-btn');
    if (!btn) return;
    var key = btn.dataset.itemKey;
    if (key) renew([key], btn);
  });

  // Bulk renew.
  var bulkRenew = document.getElementById('inbox-bulk-renew');
  if (bulkRenew) {
    bulkRenew.addEventListener('click', function () {
      renew(getSelectedKeys(), bulkRenew);
    });
  }
})();
