/**
 * View document modal: fetches document chunks and displays full text
 * with an optional markdown preview toggle.
 */
(function() {
  var trigger = document.getElementById('view-document-modal-trigger');
  var modal = document.getElementById('view-document-modal');
  if (!trigger || !modal) return;

  var titleEl = modal.querySelector('#view-document-title');
  var loadingEl = modal.querySelector('#view-document-loading');
  var errorEl = modal.querySelector('#view-document-error');
  var retryBtn = modal.querySelector('#view-document-retry');
  var textEl = modal.querySelector('#view-document-text');
  var previewEl = modal.querySelector('#view-document-preview');
  var previewBtn = modal.querySelector('#view-document-preview-btn');
  var downloadBtn = modal.querySelector('#view-document-download');

  var currentUrl = '';
  var currentName = 'Document';
  var rawText = '';
  var previewMode = false;

  function showState(state) {
    loadingEl.classList.toggle('hidden', state !== 'loading');
    errorEl.classList.toggle('hidden', state !== 'error');
    textEl.classList.toggle('hidden', state !== 'text');
    previewEl.classList.toggle('hidden', state !== 'preview');
  }

  function updatePreviewButton() {
    previewBtn.classList.toggle('is-active', previewMode);
  }

  function fetchChunks(url) {
    currentUrl = url;
    showState('loading');
    fetch(url, { credentials: 'same-origin' })
      .then(function(resp) {
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        return resp.json();
      })
      .then(function(data) {
        rawText = (data.chunks || []).map(function(c) { return c.text; }).join('\n\n');
        if (previewMode) {
          renderPreview();
        } else {
          textEl.textContent = rawText;
          showState('text');
        }
      })
      .catch(function() {
        showState('error');
      });
  }

  function renderPreview() {
    try {
      var html = marked.parse(rawText);
      previewEl.innerHTML = DOMPurify.sanitize(html, {
        ALLOWED_TAGS: ['p','br','strong','em','u','s','del','code','pre','ul','ol','li','h1','h2','h3','h4','h5','h6','blockquote','a','hr','table','thead','tbody','tr','th','td','div','span','sup','section'],
        ALLOWED_ATTR: ['href','title','target','class','id']
      });
      showState('preview');
    } catch (e) {
      textEl.textContent = rawText;
      showState('text');
    }
  }

  // Preview toggle
  previewBtn.addEventListener('click', function() {
    previewMode = !previewMode;
    updatePreviewButton();
    if (!rawText) return;
    if (previewMode) {
      renderPreview();
    } else {
      textEl.textContent = rawText;
      showState('text');
    }
  });

  // Retry on error
  retryBtn.addEventListener('click', function() {
    if (currentUrl) fetchChunks(currentUrl);
  });

  // Download the reconstructed document text as a .txt file.
  if (downloadBtn) {
    downloadBtn.addEventListener('click', function() {
      if (!rawText) return;
      var blob = new Blob([rawText], { type: 'text/plain;charset=utf-8' });
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url;
      a.download = (currentName || 'document').replace(/\.[^.]+$/, '') + '.txt';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(function() { URL.revokeObjectURL(url); }, 0);
    });
  }

  var piiEl = modal.querySelector('#view-document-pii');

  // Mirror the document's PII status pills (and their hover tooltips) from the
  // list row into the modal footer. Clones the row's pills, re-scopes their
  // tooltip IDs so they don't collide with the originals, and binds Flowbite.
  function renderPiiPills(row) {
    if (!piiEl) return;
    piiEl.innerHTML = '';
    var src = row && row.querySelector('.pii-pills');
    if (!src) return;
    var clone = src.cloneNode(true);
    clone.classList.remove('hidden', 'ms-3');
    clone.classList.add('flex');
    clone.querySelectorAll('[data-tooltip-target]').forEach(function(trigger) {
      var oldId = trigger.getAttribute('data-tooltip-target');
      var target = clone.querySelector('#' + (window.CSS && CSS.escape ? CSS.escape(oldId) : oldId));
      var newId = 'vd-' + oldId;
      trigger.setAttribute('data-tooltip-target', newId);
      if (!target) return;
      target.id = newId;
      if (window.Tooltip) {
        new window.Tooltip(target, trigger, { placement: 'top', triggerType: 'hover' });
      } else {
        trigger.setAttribute('title', target.textContent.trim());
      }
    });
    piiEl.appendChild(clone);
  }

  // Click handler for all view-document buttons
  document.querySelectorAll('.view-document-btn').forEach(function(btn) {
    btn.addEventListener('click', function(e) {
      e.preventDefault();
      e.stopPropagation();
      var url = btn.getAttribute('data-chunks-url');
      var name = btn.getAttribute('data-doc-name') || 'Document';

      // Reset state
      previewMode = false;
      rawText = '';
      currentName = name;
      updatePreviewButton();
      titleEl.textContent = name;
      textEl.textContent = '';
      previewEl.innerHTML = '';
      renderPiiPills(btn.closest('[data-doc-id]'));

      fetchChunks(url);
      setTimeout(function() { trigger.click(); }, 0);
    });
  });
})();
