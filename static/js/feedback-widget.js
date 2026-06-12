(function () {
  'use strict';

  // --- Console error buffer (installed immediately) ---
  var consoleErrors = [];
  var MAX_ERRORS = 50;
  var originalOnError = window.onerror;

  window.onerror = function (message, source, lineno, colno, error) {
    if (consoleErrors.length < MAX_ERRORS) {
      consoleErrors.push({
        message: String(message || ''),
        source: String(source || ''),
        lineno: lineno,
        colno: colno,
        stack: error && error.stack ? error.stack.substring(0, 500) : '',
        timestamp: new Date().toISOString(),
      });
    }
    if (originalOnError) return originalOnError.apply(this, arguments);
  };

  window.addEventListener('unhandledrejection', function (event) {
    if (consoleErrors.length < MAX_ERRORS) {
      consoleErrors.push({
        message: 'Unhandled rejection: ' + String(event.reason || ''),
        source: '',
        lineno: 0,
        colno: 0,
        stack:
          event.reason && event.reason.stack
            ? event.reason.stack.substring(0, 500)
            : '',
        timestamp: new Date().toISOString(),
      });
    }
  });

  // --- State ---
  var screenshotDataUrl = null;
  var screenshotBlob = null;
  var isSubmitting = false;

  // --- DOM refs ---
  var triggerDesktop, triggerMobile, modal;
  var textArea, metaCheckbox, screenshotCheckbox;
  var screenshotPreview, screenshotThumb;
  var submitBtn, cancelBtn, closeBtn;

  function init() {
    triggerDesktop = document.getElementById('feedback-trigger-desktop');
    triggerMobile = document.getElementById('feedback-trigger-mobile');
    modal = document.getElementById('feedback-modal');
    textArea = document.getElementById('feedback-text');
    metaCheckbox = document.getElementById('feedback-include-meta');
    screenshotCheckbox = document.getElementById('feedback-include-screenshot');
    screenshotPreview = document.getElementById('feedback-screenshot-preview');
    screenshotThumb = document.getElementById('feedback-screenshot-thumb');
    submitBtn = document.getElementById('feedback-submit-btn');
    cancelBtn = document.getElementById('feedback-cancel-btn');
    closeBtn = document.getElementById('feedback-modal-close');

    if (!modal) return;

    if (triggerDesktop)
      triggerDesktop.addEventListener('click', onTriggerClick);
    if (triggerMobile)
      triggerMobile.addEventListener('click', onTriggerClick);

    closeBtn.addEventListener('click', closeModal);
    cancelBtn.addEventListener('click', closeModal);
    submitBtn.addEventListener('click', onSubmit);

    screenshotCheckbox.addEventListener('change', function () {
      screenshotPreview.classList.toggle(
        'hidden',
        !this.checked || !screenshotDataUrl
      );
    });

    modal.addEventListener('click', function (e) {
      if (e.target === modal) closeModal();
    });

    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
        closeModal();
      }
    });
  }

  function onTriggerClick() {
    screenshotDataUrl = null;
    screenshotBlob = null;
    captureScreenshot()
      .then(function () {
        openModal();
      })
      .catch(function (err) {
        console.warn('Feedback: screenshot capture failed:', err);
        screenshotDataUrl = null;
        screenshotBlob = null;
        openModal();
      });
  }

  function captureScreenshot() {
    if (typeof html2canvas !== 'function') {
      return Promise.reject(new Error('html2canvas-pro not loaded'));
    }
    var maxHeight = Math.min(
      document.documentElement.scrollHeight,
      window.innerHeight * 3
    );
    return html2canvas(document.body, {
      useCORS: true,
      scale: 0.5,
      logging: false,
      height: maxHeight,
      windowWidth: document.documentElement.clientWidth,
      windowHeight: window.innerHeight,
    }).then(function (canvas) {
      if (!canvas || canvas.width === 0 || canvas.height === 0) {
        return Promise.reject(new Error('html2canvas returned empty canvas'));
      }
      screenshotDataUrl = canvas.toDataURL('image/jpeg', 0.6);
      return new Promise(function (resolve, reject) {
        canvas.toBlob(
          function (blob) {
            if (!blob) {
              reject(new Error('toBlob returned null'));
              return;
            }
            screenshotBlob = blob;
            resolve();
          },
          'image/jpeg',
          0.6
        );
      });
    });
  }

  function openModal() {
    modal.classList.remove('hidden');
    modal.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';

    if (screenshotDataUrl) {
      screenshotThumb.src = screenshotDataUrl;
      screenshotPreview.classList.remove('hidden');
      screenshotCheckbox.checked = true;
      screenshotCheckbox.disabled = false;
    } else {
      screenshotPreview.classList.add('hidden');
      screenshotCheckbox.checked = false;
      screenshotCheckbox.disabled = true;
    }

    textArea.value = '';
    textArea.focus();
  }

  function closeModal() {
    modal.classList.add('hidden');
    modal.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
    screenshotDataUrl = null;
    screenshotBlob = null;
    // Deliberately does NOT reset the submit state: closing the modal while a
    // request is in flight must not re-arm the submit button (duplicate
    // submission). Only the fetch's terminal handlers reset it.
  }

  function resetSubmitState() {
    isSubmitting = false;
    submitBtn.disabled = false;
    submitBtn.textContent = 'Send feedback';
  }

  // Resolve to the parsed JSON body, or null when the response is not JSON
  // (e.g. an HTML error or login page). Never rejects on a parse failure.
  function parseJsonSafe(res) {
    var contentType = res.headers.get('content-type') || '';
    if (contentType.indexOf('application/json') === -1) {
      return Promise.resolve(null);
    }
    return res.json().catch(function () {
      return null;
    });
  }

  function onSubmit() {
    var text = textArea.value.trim();
    if (!text) {
      textArea.focus();
      return;
    }
    if (isSubmitting) return;
    isSubmitting = true;
    submitBtn.disabled = true;
    submitBtn.textContent = 'Sending...';

    var csrf = document
      .querySelector('meta[name="csrf-token"]')
      ?.getAttribute('content');
    var fd = new FormData();
    fd.append('text', text);

    // Console errors contain script URLs and stack traces, so they fall under
    // the same opt-out as the page URL and browser info.
    var includeMeta = metaCheckbox.checked;
    if (includeMeta) {
      fd.append('url', window.location.href);
      fd.append('user_agent', navigator.userAgent);
      fd.append('viewport', window.innerWidth + 'x' + window.innerHeight);
      fd.append('console_errors', JSON.stringify(consoleErrors));
    }

    if (screenshotCheckbox.checked && screenshotBlob) {
      fd.append('screenshot', screenshotBlob, 'screenshot.jpg');
    }

    fetch('/api/feedback/submit/', {
      method: 'POST',
      headers: { 'X-CSRFToken': csrf },
      body: fd,
      credentials: 'same-origin',
    })
      .then(function (res) {
        // A lost session makes @login_required redirect the POST to the login
        // page; fetch follows it and we arrive at a 200 HTML page that is not
        // JSON. Detect the redirect rather than trying to parse the page.
        if (res.redirected) {
          throw new Error(
            'Your session has expired. Please refresh the page and sign in again.'
          );
        }
        if (res.ok) return null;
        if (res.status === 502 || res.status === 503 || res.status === 504) {
          throw new Error(
            'The server is temporarily unavailable. Please try again in a moment.'
          );
        }
        // Non-OK: the body may be our JSON error payload or an HTML error page.
        // Only treat it as JSON when the server says so.
        return parseJsonSafe(res).then(function (data) {
          throw new Error(
            (data && data.error) ||
              'Could not send feedback (error ' + res.status + ').'
          );
        });
      })
      .then(function () {
        // Clear the buffer only when it was actually sent, so an opted-out
        // submission doesn't silently discard errors the user might choose to
        // include next time.
        if (includeMeta) consoleErrors.length = 0;
        resetSubmitState();
        closeModal();
        showToast('Thank you for your feedback!');
      })
      .catch(function (err) {
        resetSubmitState();
        var msg = (err && err.message) || 'Please try again.';
        // Network failures (and any stray parse error) read as technical noise;
        // show something actionable instead of leaking it to the user.
        if (
          /JSON|Unexpected token|Failed to fetch|NetworkError|Load failed/i.test(
            msg
          )
        ) {
          msg =
            'Could not reach the server. Please check your connection and try again.';
        }
        showToast('Failed to send feedback: ' + msg);
      });
  }

  function showToast(msg) {
    var el = document.createElement('div');
    el.textContent = msg;
    el.className =
      'fixed bottom-4 left-1/2 -translate-x-1/2 z-50 px-4 py-2 rounded-lg ' +
      'bg-neutral-800 text-white text-sm shadow-lg transition-opacity duration-300';
    document.body.appendChild(el);
    setTimeout(function () {
      el.style.opacity = '0';
      setTimeout(function () {
        el.remove();
      }, 300);
    }, 4000);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
