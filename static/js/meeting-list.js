// Checkbox selection + bulk archive/delete for the meetings list.
// Mirrors the selection slice of document-list.js, but keyed on meeting UUIDs
// (strings) rather than integer document ids, and without the doc-only
// upload / status-poll / delete-confirm-modal layers.
(function () {
  'use strict';

  var config = document.getElementById('mtg-list-config');
  if (!config) return;

  var bulkArchiveUrl = config.dataset.bulkArchiveUrl;
  var bulkDeleteUrl = config.dataset.bulkDeleteUrl;
  var csrf = config.dataset.csrfToken ||
    (document.querySelector('meta[name="csrf-token"]') || {}).content;

  // ── Selection factory ──────────────────────────────────────────────
  function setupListSelection(listName, selectAllId, bulkActionsId, countId) {
    var selectAll = document.getElementById(selectAllId);
    var bulkActions = document.getElementById(bulkActionsId);
    var countEl = document.getElementById(countId);
    if (!selectAll || !bulkActions) return null;

    var rows = document.querySelectorAll('[data-list="' + listName + '"]');
    var checkboxes = [];
    rows.forEach(function (row) {
      var cb = row.querySelector('.meeting-checkbox');
      if (cb) checkboxes.push(cb);
    });
    if (!checkboxes.length) return null;

    function getSelectedUuids() {
      var uuids = [];
      checkboxes.forEach(function (cb) {
        if (cb.checked) {
          var row = cb.closest('[data-meeting-uuid]');
          if (row) uuids.push(row.dataset.meetingUuid);
        }
      });
      return uuids;
    }

    function updateUI() {
      var checked = checkboxes.filter(function (cb) { return cb.checked; }).length;
      var total = checkboxes.length;
      selectAll.checked = checked === total;
      selectAll.indeterminate = checked > 0 && checked < total;
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

    selectAll.addEventListener('change', function () {
      checkboxes.forEach(function (cb) { cb.checked = selectAll.checked; });
      updateUI();
    });

    checkboxes.forEach(function (cb) {
      cb.addEventListener('change', updateUI);
    });

    return { getSelectedUuids: getSelectedUuids };
  }

  var activeList = setupListSelection('active', 'select-all-active', 'active-bulk-actions', 'active-selection-count');
  var archivedList = setupListSelection('archived', 'select-all-archived', 'archived-bulk-actions', 'archived-selection-count');

  // ── POST helper ────────────────────────────────────────────────────
  function doFetch(url, body) {
    return fetch(url, {
      method: 'POST',
      credentials: 'same-origin',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrf
      },
      body: JSON.stringify(body)
    });
  }

  function setButtonLoading(btn, loading) {
    if (!btn) return;
    if (loading) {
      btn.disabled = true;
      btn.dataset.origHtml = btn.innerHTML;
      btn.innerHTML = '<svg class="w-4 h-4 animate-spin inline-block mr-1" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/></svg>Working…';
    } else {
      btn.disabled = false;
      if (btn.dataset.origHtml) btn.innerHTML = btn.dataset.origHtml;
    }
  }

  // ── Wiring ─────────────────────────────────────────────────────────
  function wireArchive(btnId, list, action) {
    var btn = document.getElementById(btnId);
    if (!btn || !list) return;
    btn.addEventListener('click', function () {
      var uuids = list.getSelectedUuids();
      if (!uuids.length) return;
      setButtonLoading(btn, true);
      doFetch(bulkArchiveUrl, { meeting_uuids: uuids, action: action })
        .then(function (r) { if (!r.ok) throw new Error('Request failed'); location.reload(); })
        .catch(function () { alert('Something went wrong. Please try again.'); setButtonLoading(btn, false); });
    });
  }

  function wireDelete(btnId, list) {
    var btn = document.getElementById(btnId);
    if (!btn || !list) return;
    btn.addEventListener('click', function () {
      var uuids = list.getSelectedUuids();
      if (!uuids.length) return;
      var n = uuids.length;
      var msg = 'Delete ' + n + ' meeting' + (n === 1 ? '' : 's') + "? This can't be undone.";
      if (!window.confirm(msg)) return;
      setButtonLoading(btn, true);
      doFetch(bulkDeleteUrl, { meeting_uuids: uuids })
        .then(function (r) { if (!r.ok) throw new Error('Delete failed'); location.reload(); })
        .catch(function () { alert('Failed to delete meetings. Please try again.'); setButtonLoading(btn, false); });
    });
  }

  wireArchive('active-bulk-archive', activeList, 'archive');
  wireDelete('active-bulk-delete', activeList);
  wireArchive('archived-bulk-restore', archivedList, 'restore');
  wireDelete('archived-bulk-delete', archivedList);
})();
