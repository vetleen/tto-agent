(function () {
  'use strict';

  var section = document.getElementById('description-section');
  if (!section) return;

  var generateUrl = section.dataset.generateUrl;
  var updateUrl = section.dataset.updateUrl;
  var textarea = document.getElementById('data-room-description');
  var generateBtn = document.getElementById('generate-description-btn');
  var generateText = document.getElementById('generate-description-text');
  var saveBtn = document.getElementById('save-description-btn');
  var csrf = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');
  var originalValue = textarea ? textarea.value : '';

  if (!textarea || !generateBtn || !saveBtn) return;

  // Show save button when text changes
  textarea.addEventListener('input', function () {
    if (textarea.value !== originalValue) {
      saveBtn.classList.remove('hidden');
    } else {
      saveBtn.classList.add('hidden');
    }
  });

  // Generate description via LLM
  generateBtn.addEventListener('click', function () {
    generateBtn.disabled = true;
    generateText.textContent = 'Generating...';

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
          saveBtn.classList.remove('hidden');
        } else if (data.error) {
          alert(data.error);
        }
      })
      .catch(function () {
        alert('Failed to generate description. Please try again.');
      })
      .finally(function () {
        generateBtn.disabled = false;
        generateText.textContent = 'Ask Wilfred';
      });
  });

  // Save description
  saveBtn.addEventListener('click', function () {
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving...';

    fetch(updateUrl, {
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
          saveBtn.classList.add('hidden');
        } else if (data.error) {
          alert(data.error);
        }
      })
      .catch(function () {
        alert('Failed to save description. Please try again.');
      })
      .finally(function () {
        saveBtn.disabled = false;
        saveBtn.textContent = 'Save';
      });
  });
})();
