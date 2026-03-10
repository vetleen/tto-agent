(function () {
  'use strict';

  var config = document.getElementById('doc-list-config');
  if (!config) return;

  var bulkDeleteUrl = config.dataset.bulkDeleteUrl;
  var bulkArchiveUrl = config.dataset.bulkArchiveUrl;
  var statusUrl = config.dataset.statusUrl;
  var csrf = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');

  // ── Status icons ──────────────────────────────────────────────────
  var ICONS = {
    uploaded: '<svg class="w-4 h-4 animate-spin text-body" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/></svg>',
    processing: '<svg class="w-4 h-4 animate-spin text-body" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/></svg>',
    ready: '<svg class="w-4 h-4 text-green-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"/></svg>',
    failed: '<svg class="w-4 h-4 text-red-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="m9.75 9.75 4.5 4.5m0-4.5-4.5 4.5M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"/></svg>'
  };

  function renderStatusIcons() {
    document.querySelectorAll('[data-doc-id]').forEach(function (row) {
      var status = row.dataset.status;
      var iconEl = row.querySelector('.doc-status-icon');
      if (iconEl) iconEl.innerHTML = ICONS[status] || '';
    });
  }

  renderStatusIcons();

  // ── Polling ────────────────────────────────────────────────────────
  var TERMINAL = { ready: true, failed: true };
  var pollInterval = null;

  function hasNonTerminal() {
    var rows = document.querySelectorAll('[data-doc-id]');
    for (var i = 0; i < rows.length; i++) {
      if (!TERMINAL[rows[i].dataset.status]) return true;
    }
    return false;
  }

  function pollStatuses() {
    fetch(statusUrl, { credentials: 'same-origin' })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        var statuses = data.statuses || {};
        var changed = false;
        document.querySelectorAll('[data-doc-id]').forEach(function (row) {
          var newStatus = statuses[row.dataset.docId];
          if (newStatus && newStatus !== row.dataset.status) {
            row.dataset.status = newStatus;
            changed = true;
          }
        });
        if (changed) renderStatusIcons();
        if (!hasNonTerminal() && pollInterval) {
          clearInterval(pollInterval);
          pollInterval = null;
        }
      })
      .catch(function () { /* silently retry on next interval */ });
  }

  if (hasNonTerminal()) {
    pollInterval = setInterval(pollStatuses, 2000);
  }

  // ── Checkbox selection + bulk actions ──────────────────────────────
  function setupListSelection(listName, selectAllId, bulkActionsId, countId, labelId) {
    var selectAll = document.getElementById(selectAllId);
    var bulkActions = document.getElementById(bulkActionsId);
    var countEl = document.getElementById(countId);
    var labelEl = document.getElementById(labelId);
    if (!selectAll || !bulkActions) return;

    var rows = document.querySelectorAll('[data-list="' + listName + '"]');
    var checkboxes = [];
    rows.forEach(function (row) {
      var cb = row.querySelector('.doc-checkbox');
      if (cb) checkboxes.push(cb);
    });
    if (!checkboxes.length) return;

    function getSelectedIds() {
      var ids = [];
      checkboxes.forEach(function (cb) {
        if (cb.checked) {
          var row = cb.closest('[data-doc-id]');
          if (row) ids.push(Number(row.dataset.docId));
        }
      });
      return ids;
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

    return { getSelectedIds: getSelectedIds };
  }

  var activeList = setupListSelection('active', 'select-all-active', 'active-bulk-actions', 'active-selection-count', 'active-selection-label');
  var archivedList = setupListSelection('archived', 'select-all-archived', 'archived-bulk-actions', 'archived-selection-count', 'archived-selection-label');

  function setButtonLoading(btn, loading) {
    if (!btn) return;
    if (loading) {
      btn.disabled = true;
      btn.dataset.origHtml = btn.innerHTML;
      btn.innerHTML = '<svg class="w-4 h-4 animate-spin inline-block mr-1" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/></svg>Deleting\u2026';
    } else {
      btn.disabled = false;
      if (btn.dataset.origHtml) btn.innerHTML = btn.dataset.origHtml;
    }
  }

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

  // Active list buttons
  var activeBulkDelete = document.getElementById('active-bulk-delete');
  var activeBulkArchive = document.getElementById('active-bulk-archive');

  if (activeBulkDelete && activeList) {
    activeBulkDelete.addEventListener('click', function () {
      var ids = activeList.getSelectedIds();
      if (!ids.length) return;
      if (!confirm('Are you sure you want to delete ' + ids.length + ' document(s)?')) return;
      setButtonLoading(activeBulkDelete, true);
      doFetch(bulkDeleteUrl, { document_ids: ids })
        .then(function (r) { if (!r.ok) throw new Error('Delete failed'); location.reload(); })
        .catch(function () { alert('Failed to delete documents. Please try again.'); setButtonLoading(activeBulkDelete, false); });
    });
  }

  if (activeBulkArchive && activeList) {
    activeBulkArchive.addEventListener('click', function () {
      var ids = activeList.getSelectedIds();
      if (!ids.length) return;
      doFetch(bulkArchiveUrl, { document_ids: ids, action: 'archive' }).then(function () { location.reload(); });
    });
  }

  // Archived list buttons
  var archivedBulkDelete = document.getElementById('archived-bulk-delete');
  var archivedBulkRestore = document.getElementById('archived-bulk-restore');

  if (archivedBulkDelete && archivedList) {
    archivedBulkDelete.addEventListener('click', function () {
      var ids = archivedList.getSelectedIds();
      if (!ids.length) return;
      if (!confirm('Are you sure you want to delete ' + ids.length + ' document(s)?')) return;
      setButtonLoading(archivedBulkDelete, true);
      doFetch(bulkDeleteUrl, { document_ids: ids })
        .then(function (r) { if (!r.ok) throw new Error('Delete failed'); location.reload(); })
        .catch(function () { alert('Failed to delete documents. Please try again.'); setButtonLoading(archivedBulkDelete, false); });
    });
  }

  if (archivedBulkRestore && archivedList) {
    archivedBulkRestore.addEventListener('click', function () {
      var ids = archivedList.getSelectedIds();
      if (!ids.length) return;
      doFetch(bulkArchiveUrl, { document_ids: ids, action: 'restore' }).then(function () { location.reload(); });
    });
  }
})();
