(function () {
  'use strict';

  var section = document.getElementById('description-section');
  if (!section) return;

  var generateUrl = section.dataset.generateUrl;
  var updateUrl = section.dataset.updateUrl;
  var textarea = document.getElementById('data-room-description');
  var generateBtn = document.getElementById('generate-description-btn');
  var generateText = document.getElementById('generate-description-text');
  var generateIcon = generateBtn ? generateBtn.querySelector('[data-wf-ai-icon]') : null;
  var generateSpinner = generateBtn ? generateBtn.querySelector('[data-wf-ai-spinner]') : null;
  var generateIdleLabel = generateText ? generateText.textContent : '';
  var generateBusyLabel = (generateBtn && generateBtn.dataset.busyLabel) || 'Working…';
  var saveBtn = document.getElementById('save-description-btn');
  var charCount = document.getElementById('description-char-count');
  var csrf = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');
  var originalValue = textarea ? textarea.value : '';
  var maxLen = 1000;

  if (!textarea || !generateBtn || !saveBtn) return;

  function updateCharCount() {
    if (!charCount) return;
    charCount.textContent = textarea.value.length + ' / ' + maxLen;
  }

  function saveDescription() {
    return fetch(updateUrl, {
      method: 'POST',
      headers: {
        'X-CSRFToken': csrf,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ description: textarea.value }),
    })
      .then(function (res) { return res.json(); })
      .then(function (data) {
        if (data.status === 'ok') {
          originalValue = textarea.value;
          syncSaveBtn();
        } else if (data.error) {
          alert(data.error);
        }
      })
      .catch(function () {
        alert('Failed to save description. Please try again.');
      });
  }

  // Init counter
  updateCharCount();

  function syncSaveBtn() {
    saveBtn.disabled = textarea.value === originalValue;
  }

  // Enable save button when text changes
  textarea.addEventListener('input', function () {
    updateCharCount();
    syncSaveBtn();
  });

  // Generate description via LLM
  generateBtn.addEventListener('click', function () {
    generateBtn.disabled = true;
    generateBtn.classList.add('wf-ai-waiting');
    generateBtn.setAttribute('aria-busy', 'true');
    if (generateIcon) generateIcon.classList.add('hidden');
    if (generateSpinner) generateSpinner.classList.remove('hidden');
    generateText.textContent = generateBusyLabel;

    fetch(generateUrl, {
      method: 'POST',
      headers: {
        'X-CSRFToken': csrf,
        'Content-Type': 'application/json',
      },
      body: '{}',
    })
      .then(function (res) { return res.json(); })
      .then(function (data) {
        if (data.description) {
          textarea.value = data.description;
          updateCharCount();
          // Auto-save the generated description
          return saveDescription();
        } else if (data.error) {
          alert(data.error);
        }
      })
      .catch(function () {
        alert('Failed to generate description. Please try again.');
      })
      .finally(function () {
        generateBtn.disabled = false;
        generateBtn.classList.remove('wf-ai-waiting');
        generateBtn.removeAttribute('aria-busy');
        if (generateSpinner) generateSpinner.classList.add('hidden');
        if (generateIcon) generateIcon.classList.remove('hidden');
        generateText.textContent = generateIdleLabel;
      });
  });

  // Save description
  saveBtn.addEventListener('click', function () {
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving...';

    saveDescription().finally(function () {
      saveBtn.disabled = false;
      saveBtn.textContent = 'Save';
    });
  });
})();
