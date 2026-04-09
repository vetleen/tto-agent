/* meetings/transcribe.js — live in-browser transcription for meeting detail page.
 *
 * Architecture:
 *   - getUserMedia({audio:{deviceId}}) -> MediaStream
 *   - AudioContext + AnalyserNode -> mic level meter (rAF loop)
 *   - MediaRecorder restart-every-30s pattern: each cycle produces a fully-
 *     formed container file we can transcribe independently. Small (<200ms)
 *     gap at the boundary is acceptable for minutes.
 *   - WebSocket binary frames carry the audio bytes. Text JSON frames carry
 *     control messages and incoming segment transcripts.
 *
 * State machine:
 *   idle -> requesting -> running -> stopping -> idle
 */
(function () {
  'use strict';

  const SEGMENT_DURATION_MS = 30000;
  const RECONNECT_BACKOFF_MS = 2000;

  const root = document.querySelector('[data-meeting-uuid]');
  if (!root) return;

  const meetingUuid = root.dataset.meetingUuid;
  const initialStatus = root.dataset.meetingStatus;
  const autoStopDefault = parseInt(root.dataset.autoStopDefault || '3600', 10);
  const autoStopMax = parseInt(root.dataset.autoStopMax || '14400', 10);
  const autoStartTranscription = root.dataset.autoStartTranscription === '1';

  const transcribeBtn = document.getElementById('transcribe-btn');
  const transcribeBtnLabel = document.getElementById('transcribe-btn-label');
  const stopBtn = document.getElementById('stop-btn');
  const controlsEl = document.getElementById('transcribe-controls');
  const elapsedEl = document.getElementById('elapsed-counter');
  const micSelect = document.getElementById('mic-select');
  const autoStopInput = document.getElementById('auto-stop-input');
  const levelBar = document.getElementById('mic-level-bar');
  const transcriptPane = document.getElementById('transcript-pane');
  const transcribingIndicator = document.getElementById('transcribing-indicator');
  const transcribingIndicatorLabel = document.getElementById('transcribing-indicator-label');
  const unsupportedBanner = document.getElementById('transcribe-unsupported');
  const uploadForm = document.getElementById('upload-form');
  const uploadBanner = document.getElementById('upload-transcribing-banner');
  const uploadBannerLabel = document.getElementById('upload-transcribing-banner-label');
  const uploadBannerProgressWrap = document.getElementById('upload-transcribing-banner-progress-wrap');
  const uploadBannerProgressBar = document.getElementById('upload-transcribing-banner-progress-bar');

  const metadataForm = document.getElementById('meeting-metadata-form');
  const metadataSavedEl = document.getElementById('meeting-metadata-saved');
  const nameInput = document.getElementById('meeting-name-input');

  const meetingHasTranscript = root.dataset.meetingHasTranscript === '1';
  const startLabel = meetingHasTranscript ? 'Continue transcription' : 'Start transcription';

  function setTranscribeBtnLabel(text) {
    if (transcribeBtnLabel) {
      transcribeBtnLabel.textContent = text;
    } else if (transcribeBtn) {
      transcribeBtn.textContent = text;
    }
  }

  // ---------------- feature detection ----------------
  if (!('MediaRecorder' in window) || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    if (unsupportedBanner) unsupportedBanner.classList.remove('hidden');
    if (transcribeBtn) transcribeBtn.disabled = true;
  }

  // ---------------- metadata save (fetch, no reload) ----------------
  function postMetadata(fields, onSuccess) {
    if (!metadataForm) return;
    const data = new FormData();
    const csrf = metadataForm.querySelector('input[name="csrfmiddlewaretoken"]');
    if (csrf) data.append('csrfmiddlewaretoken', csrf.value);
    Object.keys(fields).forEach(function (k) { data.append(k, fields[k]); });
    fetch(metadataForm.action, {
      method: 'POST',
      body: data,
      credentials: 'same-origin',
      headers: { 'X-Requested-With': 'XMLHttpRequest' },
    }).then(function (r) {
      if (!r.ok) throw new Error('save failed');
      if (typeof onSuccess === 'function') onSuccess();
    }).catch(function () {
      alert('Could not save meeting metadata.');
    });
  }

  if (metadataForm) {
    // Block accidental form submission (e.g. Enter key in a field) — we
    // auto-save on input changes instead.
    metadataForm.addEventListener('submit', function (ev) {
      ev.preventDefault();
    });

    const autoSaveFields = ['agenda', 'participants', 'description'];
    const lastSaved = {};
    const debounceTimers = {};
    let savedHideTimer = null;

    function flashSaved() {
      if (!metadataSavedEl) return;
      metadataSavedEl.textContent = 'Saved.';
      metadataSavedEl.classList.remove('hidden');
      if (savedHideTimer) clearTimeout(savedHideTimer);
      savedHideTimer = setTimeout(function () {
        metadataSavedEl.classList.add('hidden');
      }, 2000);
    }

    autoSaveFields.forEach(function (field) {
      const el = metadataForm.querySelector('[name="' + field + '"]');
      if (!el) return;
      lastSaved[field] = el.value;
      el.addEventListener('input', function () {
        if (debounceTimers[field]) clearTimeout(debounceTimers[field]);
        debounceTimers[field] = setTimeout(function () {
          const value = el.value;
          if (value === lastSaved[field]) return;
          const payload = {};
          payload[field] = value;
          postMetadata(payload, function () {
            lastSaved[field] = value;
            flashSaved();
          });
        }, 600);
      });
      el.addEventListener('blur', function () {
        // Flush any pending debounced save immediately on blur.
        if (debounceTimers[field]) {
          clearTimeout(debounceTimers[field]);
          debounceTimers[field] = null;
        }
        const value = el.value;
        if (value === lastSaved[field]) return;
        const payload = {};
        payload[field] = value;
        postMetadata(payload, function () {
          lastSaved[field] = value;
          flashSaved();
        });
      });
    });
  }

  // Save the meeting name when the user blurs the title input or hits Enter.
  if (nameInput) {
    let lastSavedName = nameInput.value;
    function saveName() {
      const value = nameInput.value.trim();
      if (!value) {
        nameInput.value = lastSavedName;
        return;
      }
      if (value === lastSavedName) return;
      postMetadata({ name: value }, function () {
        lastSavedName = value;
        document.title = value + ' — Wilfred';
      });
    }
    nameInput.addEventListener('blur', saveName);
    nameInput.addEventListener('keydown', function (ev) {
      if (ev.key === 'Enter') {
        ev.preventDefault();
        nameInput.blur();
      }
    });
  }

  // ---------------- live transcription state ----------------
  let mediaStream = null;
  let mediaRecorder = null;
  let segmentTimer = null;
  let elapsedTimer = null;
  let autoStopTimer = null;
  let segmentIndex = 0;          // local index since this session started
  let segmentIndexBase = 0;      // server tells us where to start (resume support)
  let startOffsetSec = 0;
  let elapsedSec = 0;
  let ws = null;
  let transcribing = false;
  let stoppingPendingDrain = false;  // user clicked Stop, waiting for in-flight chunks
  let pendingSegments = {};      // index -> placeholder DOM node
  const inFlightSegments = new Set();  // segment indices queued but not yet ready/failed
  let preferredMime = '';
  let audioContext = null;
  let analyserNode = null;
  let levelLoop = null;
  let beforeUnloadHandler = null;

  // Choose the best supported audio mime type for MediaRecorder.
  function pickMime() {
    const candidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/ogg',
      'audio/mp4',
      '',
    ];
    for (const c of candidates) {
      if (c === '' || (window.MediaRecorder && MediaRecorder.isTypeSupported(c))) {
        return c;
      }
    }
    return '';
  }

  async function populateMics() {
    if (!micSelect) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      micSelect.innerHTML = '';
      const mics = devices.filter(function (d) { return d.kind === 'audioinput'; });
      if (mics.length === 0) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = 'No microphone detected';
        micSelect.appendChild(opt);
        return;
      }
      mics.forEach(function (d, i) {
        const opt = document.createElement('option');
        opt.value = d.deviceId;
        opt.textContent = d.label || ('Microphone ' + (i + 1));
        micSelect.appendChild(opt);
      });
    } catch (err) {
      console.warn('enumerateDevices failed', err);
    }
  }

  function startLevelMeter(stream) {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      audioContext = new AudioCtx();
      const source = audioContext.createMediaStreamSource(stream);
      analyserNode = audioContext.createAnalyser();
      analyserNode.fftSize = 1024;
      source.connect(analyserNode);
      const buf = new Uint8Array(analyserNode.fftSize);
      const tick = function () {
        if (!analyserNode || !levelBar) return;
        analyserNode.getByteTimeDomainData(buf);
        // RMS
        let sumSq = 0;
        for (let i = 0; i < buf.length; i++) {
          const v = (buf[i] - 128) / 128;
          sumSq += v * v;
        }
        const rms = Math.sqrt(sumSq / buf.length);
        const pct = Math.min(100, Math.round(rms * 200));
        levelBar.style.width = pct + '%';
        levelLoop = requestAnimationFrame(tick);
      };
      levelLoop = requestAnimationFrame(tick);
    } catch (err) {
      console.warn('level meter failed', err);
    }
  }

  function stopLevelMeter() {
    if (levelLoop) {
      cancelAnimationFrame(levelLoop);
      levelLoop = null;
    }
    analyserNode = null;
    if (audioContext) {
      try { audioContext.close(); } catch (e) {}
      audioContext = null;
    }
    if (levelBar) levelBar.style.width = '0%';
  }

  function fmtElapsed(sec) {
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    if (h > 0) return h + ':' + String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
    return String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
  }

  function startElapsedTimer() {
    elapsedSec = 0;
    if (elapsedEl) elapsedEl.textContent = fmtElapsed(0);
    elapsedTimer = setInterval(function () {
      elapsedSec += 1;
      if (elapsedEl) elapsedEl.textContent = fmtElapsed(elapsedSec);
      // Auto-stop check.
      const limit = Math.max(60, Math.min(autoStopMax, (parseInt(autoStopInput.value, 10) || 60) * 60));
      if (elapsedSec >= limit) {
        stopTranscription();
      }
    }, 1000);
  }

  function stopElapsedTimer() {
    if (elapsedTimer) {
      clearInterval(elapsedTimer);
      elapsedTimer = null;
    }
  }

  function installBeforeUnloadGuard() {
    beforeUnloadHandler = function (e) {
      if (transcribing || stoppingPendingDrain) {
        e.preventDefault();
        e.returnValue = '';
        return '';
      }
    };
    window.addEventListener('beforeunload', beforeUnloadHandler);
  }

  function removeBeforeUnloadGuard() {
    if (beforeUnloadHandler) {
      window.removeEventListener('beforeunload', beforeUnloadHandler);
      beforeUnloadHandler = null;
    }
  }

  // ---------------- transcribing indicator + upload visibility ----------------

  function updateIndicator() {
    if (!transcribingIndicator) return;
    const pendingCount = inFlightSegments.size;
    const shouldShow = transcribing || stoppingPendingDrain || pendingCount > 0;
    if (shouldShow) {
      transcribingIndicator.classList.remove('hidden');
      let label;
      if (stoppingPendingDrain && pendingCount > 0) {
        label = 'Finalizing transcription… (' + pendingCount + ' segment' + (pendingCount === 1 ? '' : 's') + ' left)';
      } else if (pendingCount > 0) {
        label = 'Transcribing… (' + pendingCount + ' segment' + (pendingCount === 1 ? '' : 's') + ' in flight)';
      } else {
        label = 'Transcribing…';
      }
      if (transcribingIndicatorLabel) transcribingIndicatorLabel.textContent = label;
    } else {
      transcribingIndicator.classList.add('hidden');
    }
  }

  function setUploadFormVisible(visible) {
    if (!uploadForm) return;
    uploadForm.style.display = visible ? '' : 'none';
  }

  // Toggle disabled + visually-disabled styling on every element marked with
  // `data-disable-while-busy`. Used while transcription is in flight (both
  // the live-recording WS path and the audio-upload poll path) so the user
  // can't e.g. click "Summarize in chat" or "Save to data room" on a meeting
  // whose transcript is still being built. Original disabled state is stashed
  // so non-busy restores it correctly (e.g. Summarize button disabled when
  // there's no transcript yet).
  function setActionsBusy(busy) {
    const elements = document.querySelectorAll('[data-disable-while-busy]');
    elements.forEach(function (el) {
      if (busy) {
        if (el.dataset.busyOrigDisabled === undefined) {
          el.dataset.busyOrigDisabled = ('disabled' in el && el.disabled) ? '1' : '0';
        }
        if ('disabled' in el) el.disabled = true;
        el.setAttribute('aria-disabled', 'true');
        el.classList.add('opacity-50', 'pointer-events-none');
        el.querySelectorAll('input, button').forEach(function (inner) {
          if (inner.dataset.busyOrigDisabled === undefined) {
            inner.dataset.busyOrigDisabled = inner.disabled ? '1' : '0';
          }
          inner.disabled = true;
        });
      } else {
        if ('disabled' in el) el.disabled = el.dataset.busyOrigDisabled === '1';
        delete el.dataset.busyOrigDisabled;
        el.removeAttribute('aria-disabled');
        el.classList.remove('opacity-50', 'pointer-events-none');
        el.querySelectorAll('input, button').forEach(function (inner) {
          inner.disabled = inner.dataset.busyOrigDisabled === '1';
          delete inner.dataset.busyOrigDisabled;
        });
      }
    });
  }

  function appendOrUpdateSegmentNode(idx, text, kind) {
    // Replace the "no transcript yet" placeholder if present.
    if (transcriptPane.querySelector('.italic')) {
      transcriptPane.innerHTML = '';
    }
    let node = pendingSegments[idx];
    if (!node) {
      node = document.createElement('span');
      node.dataset.segmentIndex = String(idx);
      pendingSegments[idx] = node;
      // Insert in order based on dataset.segmentIndex.
      const existing = Array.prototype.slice.call(transcriptPane.children);
      let inserted = false;
      for (const ch of existing) {
        const chIdx = parseInt(ch.dataset && ch.dataset.segmentIndex, 10);
        if (!isNaN(chIdx) && chIdx > idx) {
          transcriptPane.insertBefore(node, ch);
          inserted = true;
          break;
        }
      }
      if (!inserted) transcriptPane.appendChild(node);
    }
    if (kind === 'failed') {
      node.textContent = '[transcription failed: ' + (text || 'unknown error') + '] ';
      node.className = 'text-fg-danger';
    } else {
      node.textContent = (text || '') + ' ';
      node.className = '';
    }
  }

  function openWebSocket() {
    return new Promise(function (resolve, reject) {
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const url = proto + '://' + window.location.host + '/ws/meetings/' + meetingUuid + '/transcribe/';
      ws = new WebSocket(url);
      ws.binaryType = 'arraybuffer';
      let resolved = false;
      ws.onopen = function () {};
      ws.onmessage = function (ev) {
        let msg;
        try { msg = JSON.parse(ev.data); } catch (e) { return; }
        if (msg.type === 'started') {
          segmentIndexBase = parseInt(msg.segment_index_base || 0, 10);
          segmentIndex = segmentIndexBase;
          if (!resolved) { resolved = true; resolve(); }
        } else if (msg.type === 'segment.queued') {
          inFlightSegments.add(msg.segment_index);
          updateIndicator();
        } else if (msg.type === 'segment.ready') {
          appendOrUpdateSegmentNode(msg.segment_index, msg.text, 'ready');
          inFlightSegments.delete(msg.segment_index);
          updateIndicator();
          maybeFlushStop();
        } else if (msg.type === 'segment.failed') {
          appendOrUpdateSegmentNode(msg.segment_index, msg.error, 'failed');
          inFlightSegments.delete(msg.segment_index);
          updateIndicator();
          maybeFlushStop();
        } else if (msg.type === 'stopped') {
          // server confirmed stop; reload to show final transcript + status
          window.setTimeout(function () { window.location.reload(); }, 500);
        } else if (msg.type === 'error') {
          console.warn('meeting WS error:', msg.message);
        }
      };
      ws.onclose = function () {
        if (transcribing) {
          // Connection dropped mid-stream. Tear down local recording so the
          // user sees an interrupted state instead of a silent failure.
          shutdownLocal();
          alert('The transcription connection was lost. Reload the page to resume.');
        }
      };
      ws.onerror = function () {
        if (!resolved) reject(new Error('WebSocket connection failed'));
      };
    });
  }

  function startSegmentRecorder() {
    if (!mediaStream) return;
    let chunks = [];
    const opts = preferredMime ? { mimeType: preferredMime } : undefined;
    let recorder;
    try {
      recorder = new MediaRecorder(mediaStream, opts);
    } catch (err) {
      console.error('MediaRecorder constructor failed', err);
      return;
    }
    mediaRecorder = recorder;
    const thisIndex = segmentIndex;
    const thisOffset = startOffsetSec;
    segmentIndex += 1;
    startOffsetSec += SEGMENT_DURATION_MS / 1000;

    recorder.ondataavailable = function (ev) {
      if (ev.data && ev.data.size > 0) chunks.push(ev.data);
    };
    recorder.onstop = function () {
      const blob = new Blob(chunks, { type: preferredMime || 'audio/webm' });
      chunks = [];
      uploadChunkBlob(thisIndex, thisOffset, blob);
      if (transcribing && mediaStream && mediaStream.active) {
        // Immediately start the next segment.
        startSegmentRecorder();
      }
    };
    try {
      recorder.start();
    } catch (err) {
      console.error('recorder.start failed', err);
      return;
    }
    // Schedule the stop after SEGMENT_DURATION_MS so we get a complete file.
    segmentTimer = setTimeout(function () {
      if (recorder && recorder.state === 'recording') {
        try { recorder.stop(); } catch (e) {}
      }
    }, SEGMENT_DURATION_MS);
  }

  function uploadChunkBlob(idx, offsetSec, blob) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!blob || blob.size === 0) return;
    blob.arrayBuffer().then(function (ab) {
      const meta = {
        type: 'chunk_meta',
        segment_index: idx,
        byte_length: ab.byteLength,
        mime: preferredMime || 'audio/webm',
        start_offset_seconds: offsetSec,
      };
      try {
        ws.send(JSON.stringify(meta));
        ws.send(ab);
      } catch (err) {
        console.warn('failed to send chunk', err);
      }
    });
  }

  async function startTranscription() {
    if (transcribing) return;
    transcribing = true;
    transcribeBtn.disabled = true;
    setTranscribeBtnLabel('Connecting…');
    try {
      preferredMime = pickMime();
      const deviceId = micSelect && micSelect.value ? { exact: micSelect.value } : undefined;
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: deviceId ? { deviceId: deviceId } : true,
      });
      // Repopulate mic labels now that permission is granted.
      await populateMics();
    } catch (err) {
      transcribing = false;
      transcribeBtn.disabled = false;
      setTranscribeBtnLabel(startLabel);
      alert('Could not access microphone: ' + err.message);
      return;
    }

    try {
      await openWebSocket();
    } catch (err) {
      transcribing = false;
      shutdownLocal();
      transcribeBtn.disabled = false;
      setTranscribeBtnLabel(startLabel);
      alert('Could not open transcription connection.');
      return;
    }

    autoStopInput.value = String(Math.round(autoStopDefault / 60));
    controlsEl.classList.remove('hidden');
    // Use inline style instead of the `hidden` class: the button has
    // `inline-flex`, and Tailwind v4 emits `.inline-flex` *after* `.hidden`,
    // so the class would lose the cascade and the button would stay visible.
    transcribeBtn.style.display = 'none';
    setUploadFormVisible(false);
    setActionsBusy(true);
    updateIndicator();
    startLevelMeter(mediaStream);
    startElapsedTimer();
    installBeforeUnloadGuard();
    startSegmentRecorder();
  }

  function shutdownLocal(opts) {
    opts = opts || {};
    transcribing = false;
    stopElapsedTimer();
    stopLevelMeter();
    if (segmentTimer) { clearTimeout(segmentTimer); segmentTimer = null; }
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      try { mediaRecorder.stop(); } catch (e) {}
    }
    mediaRecorder = null;
    if (mediaStream) {
      mediaStream.getTracks().forEach(function (t) { try { t.stop(); } catch (e) {} });
      mediaStream = null;
    }
    if (!opts.keepBeforeUnload) removeBeforeUnloadGuard();
  }

  function stopTranscription() {
    if (!transcribing && !stoppingPendingDrain) return;
    transcribing = false;
    stoppingPendingDrain = true;
    if (stopBtn) stopBtn.disabled = true;
    if (segmentTimer) { clearTimeout(segmentTimer); segmentTimer = null; }
    // Force the current recorder to stop so the final chunk gets uploaded.
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      try { mediaRecorder.stop(); } catch (e) {}
    }
    updateIndicator();
    // Tear down local recording state immediately, but keep the WS open
    // so we can keep receiving segment.ready/segment.failed events for any
    // in-flight chunks. We only send the actual `stop` message once all
    // in-flight chunks have come back; otherwise the server finalizes the
    // meeting too early and the reload shows an incomplete transcript.
    shutdownLocal({ keepBeforeUnload: true });
    // The recorder.onstop handler runs asynchronously and uploads the final
    // chunk, which then becomes "in flight". Give it a tick before checking.
    setTimeout(maybeFlushStop, 300);
  }

  function maybeFlushStop() {
    if (!stoppingPendingDrain) return;
    if (inFlightSegments.size > 0) return;
    stoppingPendingDrain = false;
    removeBeforeUnloadGuard();
    if (ws && ws.readyState === WebSocket.OPEN) {
      try { ws.send(JSON.stringify({ type: 'stop' })); } catch (e) {}
    }
    updateIndicator();
  }

  // ---------------- transcription model picker ----------------

  const modelPicker = (function () {
    const btn = document.getElementById('transcription-model-btn');
    const label = document.getElementById('transcription-model-label');
    const dropdown = document.getElementById('transcription-model-dropdown');
    const optionsContainer = document.getElementById('transcription-model-options');
    const choicesEl = document.getElementById('transcription-model-choices');
    const selectedEl = document.getElementById('transcription-model-selected');
    if (!btn || !dropdown || !optionsContainer || !choicesEl || !selectedEl) {
      return null;
    }

    let choices = [];
    let selected = '';
    try {
      choices = JSON.parse(choicesEl.textContent || '[]') || [];
      selected = JSON.parse(selectedEl.textContent || '""') || '';
    } catch (e) {
      console.warn('transcription model picker: failed to parse choices', e);
      return null;
    }

    function renderOptions() {
      optionsContainer.innerHTML = '';
      choices.forEach(function (m) {
        const opt = document.createElement('button');
        opt.type = 'button';
        opt.className = 'flex items-center justify-between w-full px-3 py-1.5 text-sm text-body hover:bg-neutral-tertiary hover:text-heading';
        opt.setAttribute('data-model-id', m.id);
        const checkHidden = m.id === selected ? '' : 'hidden';
        opt.innerHTML = '<span></span><svg class="w-4 h-4 ' + checkHidden + '" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>';
        opt.firstChild.textContent = m.display_name;
        opt.addEventListener('click', function () {
          selectModel(m.id);
          dropdown.classList.add('hidden');
        });
        optionsContainer.appendChild(opt);
      });
    }

    function selectModel(modelId) {
      if (!modelId || modelId === selected) return;
      selected = modelId;
      const found = choices.find(function (c) { return c.id === modelId; });
      if (label && found) label.textContent = found.display_name;
      renderOptions();
      // Persist on the meeting record so reload + future WS connects use it.
      postMetadata({ transcription_model: modelId });
      // Hot-update the active session if any.
      if (ws && ws.readyState === WebSocket.OPEN) {
        try { ws.send(JSON.stringify({ type: 'set_model', model_id: modelId })); } catch (e) {}
      }
    }

    btn.addEventListener('click', function (e) {
      e.stopPropagation();
      dropdown.classList.toggle('hidden');
    });
    document.addEventListener('click', function (e) {
      if (!dropdown.contains(e.target) && e.target !== btn) {
        dropdown.classList.add('hidden');
      }
    });

    renderOptions();
    return { selectModel: selectModel };
  })();

  // ---------------- wire-up ----------------
  if (transcribeBtn) {
    transcribeBtn.addEventListener('click', function () { startTranscription().catch(function (err) { console.error(err); }); });
  }
  if (stopBtn) {
    stopBtn.addEventListener('click', stopTranscription);
  }

  // Pre-populate the mic dropdown with whatever labels are available before
  // permission is granted (most browsers return generic labels).
  populateMics();

  // If the meeting was opened with ?transcribe=1, auto-start transcription on
  // load. Strip the query param so a refresh doesn't re-trigger it.
  if (autoStartTranscription && transcribeBtn && !transcribeBtn.disabled) {
    if (window.history && window.history.replaceState) {
      const cleanUrl = window.location.pathname + window.location.hash;
      window.history.replaceState({}, document.title, cleanUrl);
    }
    startTranscription().catch(function (err) { console.error(err); });
  }

  // ---------------- upload transcription progress poll ----------------
  // The upload path runs in a Celery worker (the meetings/services/audio_transcription
  // orchestrator). While it works, the meeting sits in LIVE_TRANSCRIBING with
  // transcript_source=audio_upload. We poll a small JSON endpoint to surface
  // chunk-level progress to the user via the same #transcribing-indicator.
  (function uploadProgressPoll() {
    const transcriptSource = root.dataset.meetingTranscriptSource || '';
    if (initialStatus !== 'live_transcribing') return;
    if (transcriptSource !== 'audio_upload') return;
    // Don't poll if a live recording session is active in this tab — the WS
    // path drives the indicator on its own and we'd race with it.
    if (transcribing || stoppingPendingDrain) return;

    const progressWrap = document.getElementById('transcribing-progress-wrap');
    const progressBar = document.getElementById('transcribing-progress-bar');
    const url = '/meetings/' + meetingUuid + '/transcription-progress/';
    let stopped = false;

    // While upload transcription is in flight, disable every action button
    // on the page so the user can't navigate the meeting into an inconsistent
    // state (e.g. summarize in chat before the transcript is finalized).
    setActionsBusy(true);

    function showLabel(text) {
      if (transcribingIndicator) transcribingIndicator.classList.remove('hidden');
      if (transcribingIndicatorLabel) transcribingIndicatorLabel.textContent = text;
      if (uploadBannerLabel) uploadBannerLabel.textContent = text;
    }

    function setProgress(done, total) {
      const pct = total > 0
        ? Math.min(100, Math.max(0, (done / total) * 100)).toFixed(1) + '%'
        : null;
      if (progressWrap && progressBar) {
        if (pct !== null) {
          progressWrap.classList.remove('hidden');
          progressBar.style.width = pct;
        } else {
          progressWrap.classList.add('hidden');
        }
      }
      if (uploadBannerProgressWrap && uploadBannerProgressBar) {
        if (pct !== null) {
          uploadBannerProgressWrap.classList.remove('hidden');
          uploadBannerProgressBar.style.width = pct;
        } else {
          uploadBannerProgressWrap.classList.add('hidden');
        }
      }
    }

    function poll() {
      if (stopped) return;
      fetch(url, { credentials: 'same-origin', headers: { 'X-Requested-With': 'XMLHttpRequest' } })
        .then(function (r) { if (!r.ok) throw new Error('progress ' + r.status); return r.json(); })
        .then(function (data) {
          if (stopped) return;
          const status = data.status;
          const chunksTotal = parseInt(data.chunks_total || 0, 10);
          const chunksDone = parseInt(data.chunks_done || 0, 10);
          if (status === 'ready' || status === 'failed') {
            stopped = true;
            window.location.reload();
            return;
          }
          if (chunksTotal > 0) {
            showLabel('Transcribing chunk ' + chunksDone + ' of ' + chunksTotal + '…');
            setProgress(chunksDone, chunksTotal);
          } else {
            showLabel('Transcribing…');
            setProgress(0, 0);
          }
          setTimeout(poll, 3000);
        })
        .catch(function (err) {
          console.warn('upload progress poll error:', err);
          if (!stopped) setTimeout(poll, 5000);
        });
    }

    // Show indicator immediately so the user sees something while we wait
    // for the first poll response.
    showLabel('Transcribing…');
    setProgress(0, 0);
    poll();
  })();
})();
