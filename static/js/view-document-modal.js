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

  var currentUrl = '';
  var rawText = '';
  var previewMode = false;

  function showState(state) {
    loadingEl.classList.toggle('hidden', state !== 'loading');
    errorEl.classList.toggle('hidden', state !== 'error');
    textEl.classList.toggle('hidden', state !== 'text');
    previewEl.classList.toggle('hidden', state !== 'preview');
  }

  function updatePreviewButton() {
    if (previewMode) {
      previewBtn.classList.add('bg-neutral-tertiary', 'text-heading');
      previewBtn.classList.remove('text-body');
    } else {
      previewBtn.classList.remove('bg-neutral-tertiary', 'text-heading');
      previewBtn.classList.add('text-body');
    }
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
      updatePreviewButton();
      titleEl.textContent = name;
      textEl.textContent = '';
      previewEl.innerHTML = '';

      fetchChunks(url);
      setTimeout(function() { trigger.click(); }, 0);
    });
  });
})();
