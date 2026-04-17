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
  const RECONNECT_BACKOFF_MS_BASE = 2000;
  const RECONNECT_BACKOFF_MS_MAX = 30000;
  const RECONNECT_MAX_ATTEMPTS = 6;
  // Watchdog fires if we haven't uploaded a chunk in 2× the segment duration,
  // indicating MediaRecorder has stalled (common when the tab is backgrounded
  // or the OS briefly suspended the process).
  const WATCHDOG_INTERVAL_MS = 5000;
  const WATCHDOG_CHUNK_STALL_MS = SEGMENT_DURATION_MS * 2;
  const WATCHDOG_CHUNK_GIVEUP_MS = SEGMENT_DURATION_MS * 3;

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
  const stopBtnIcon = document.getElementById('stop-btn-icon');
  const stopBtnSpinner = document.getElementById('stop-btn-spinner');
  const stopBtnLabel = document.getElementById('stop-btn-label');
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
    if (transcribeBtn) {
      transcribeBtn.disabled = true;
      transcribeBtn.dataset.unsupported = '1';
    }
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

    // Language select saves immediately on change — no debounce because it's
    // a discrete choice, not a typing stream.
    const langEl = metadataForm.querySelector('[name="forced_language"]');
    if (langEl) {
      let lastSavedLang = langEl.value;
      langEl.addEventListener('change', function () {
        const value = langEl.value;
        if (value === lastSavedLang) return;
        postMetadata({ forced_language: value }, function () {
          lastSavedLang = value;
          flashSaved();
        });
      });
    }
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
  let visibilityHandler = null;
  let trackEndedHandler = null;
  // Reconnect + resume state. When the socket drops mid-session, we keep the
  // recorder running, buffer finished chunks here, and flush them once a new
  // socket is up. The original session is only abandoned once attempts are
  // exhausted — unlike the previous "one drop = reload" behavior.
  let reconnectAttempts = 0;
  let reconnecting = false;
  let reconnectTimer = null;
  const pendingUploads = [];
  // Watchdog state — detects a stalled MediaRecorder.
  let lastChunkSentAt = 0;
  let watchdogTimer = null;
  let watchdogRestartAttempted = false;
  // Active live-transcription mode — server announces this in the "started"
  // event. "chunked" (legacy) stops and restarts MediaRecorder every 30s to
  // get self-contained WebM files. "realtime" / "realtime_with_fallback"
  // use a single continuous MediaRecorder with a small timeslice so the
  // server-side ffmpeg pipe sees one unbroken Matroska stream.
  let liveMode = 'chunked';
  const STREAMING_TIMESLICE_MS = 250;

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

  // ---------------- watchdog + visibility handling ----------------

  function startWatchdog() {
    if (watchdogTimer) return;
    lastChunkSentAt = Date.now();
    watchdogRestartAttempted = false;
    watchdogTimer = setInterval(function () {
      if (!transcribing) return;
      const silenceMs = Date.now() - lastChunkSentAt;
      if (silenceMs < WATCHDOG_CHUNK_STALL_MS) return;

      if (silenceMs >= WATCHDOG_CHUNK_GIVEUP_MS) {
        // Recorder has been silent for 3× the segment duration and our one
        // restart attempt didn't produce a chunk. Treat as a terminal local
        // failure and surface the same reconnect/alert path.
        console.warn('watchdog: recorder silent for ' + silenceMs + 'ms, giving up');
        handleUnexpectedClose();
        return;
      }

      if (!watchdogRestartAttempted) {
        watchdogRestartAttempted = true;
        console.warn('watchdog: recorder appears stalled (' + silenceMs + 'ms), restarting');
        try {
          if (mediaRecorder && mediaRecorder.state === 'recording') {
            try { mediaRecorder.stop(); } catch (e) {}
          }
          if (segmentTimer) { clearTimeout(segmentTimer); segmentTimer = null; }
          if (mediaStream && mediaStream.active) {
            startRecorder();
          }
        } catch (err) {
          console.warn('watchdog: restart failed', err);
        }
      }
    }, WATCHDOG_INTERVAL_MS);
  }

  function stopWatchdog() {
    if (watchdogTimer) {
      clearInterval(watchdogTimer);
      watchdogTimer = null;
    }
  }

  function installVisibilityHandler() {
    if (visibilityHandler) return;
    visibilityHandler = function () {
      if (document.visibilityState !== 'visible') return;
      if (!transcribing) return;
      // Tab is back in the foreground. If the socket died while we were
      // hidden, kick off reconnect immediately rather than waiting for the
      // watchdog interval.
      if (!ws || ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING) {
        if (!reconnecting) {
          handleUnexpectedClose();
        }
      }
    };
    document.addEventListener('visibilitychange', visibilityHandler);
  }

  function removeVisibilityHandler() {
    if (visibilityHandler) {
      document.removeEventListener('visibilitychange', visibilityHandler);
      visibilityHandler = null;
    }
  }

  function installTrackEndedHandler(stream) {
    if (!stream) return;
    trackEndedHandler = function () {
      if (!transcribing) return;
      console.warn('microphone track ended');
      shutdownLocal();
      alert('The microphone was disconnected. Reload the page to resume transcription.');
    };
    stream.getAudioTracks().forEach(function (track) {
      try { track.addEventListener('ended', trackEndedHandler); } catch (e) {}
    });
  }

  function removeTrackEndedHandler() {
    if (!mediaStream || !trackEndedHandler) {
      trackEndedHandler = null;
      return;
    }
    mediaStream.getAudioTracks().forEach(function (track) {
      try { track.removeEventListener('ended', trackEndedHandler); } catch (e) {}
    });
    trackEndedHandler = null;
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
    const shouldShow = transcribing || stoppingPendingDrain || pendingCount > 0 || reconnecting;
    if (shouldShow) {
      transcribingIndicator.classList.remove('hidden');
      let label;
      if (reconnecting) {
        label = 'Reconnecting… (attempt ' + reconnectAttempts + ' of ' + RECONNECT_MAX_ATTEMPTS + ')';
      } else if (stoppingPendingDrain && pendingCount > 0) {
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
    // Replace the "no transcript yet" placeholder if present. The placeholder
    // is the only node in the pane with data-placeholder; don't match the
    // interim delta node (also italic) or we'd nuke live text on every seg.
    const placeholder = transcriptPane.querySelector('[data-placeholder="1"]');
    if (placeholder) {
      placeholder.remove();
    }
    // Clear any interim delta node — the canonical segment text supersedes it.
    clearInterimDelta();
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

  // ---------------- realtime interim delta rendering ----------------
  let interimNode = null;
  function renderInterimDelta(textChunk) {
    if (!transcriptPane) return;
    // Evict only the "No transcript yet" placeholder (not the interim node
    // which is also italic). Previous version ran innerHTML='' here and
    // destroyed every already-committed segment on each delta.
    const placeholder = transcriptPane.querySelector('[data-placeholder="1"]');
    if (placeholder) placeholder.remove();
    if (!interimNode) {
      interimNode = document.createElement('span');
      interimNode.className = 'text-body italic opacity-70';
      interimNode.dataset.interim = '1';
      transcriptPane.appendChild(interimNode);
    }
    interimNode.textContent = (interimNode.textContent || '') + textChunk;
  }
  function clearInterimDelta() {
    if (interimNode && interimNode.parentNode) {
      interimNode.parentNode.removeChild(interimNode);
    }
    interimNode = null;
  }
  function updateSessionStatus(state) {
    // Surface upstream-provider state next to the transcribe indicator so a
    // silent OpenAI dropout is visible to the user rather than looking like
    // the mic stopped working.
    if (!transcribingIndicatorLabel) return;
    if (state === 'reconnecting') {
      transcribingIndicatorLabel.textContent = 'Reconnecting to transcription…';
    } else if (state === 'connected') {
      transcribingIndicatorLabel.textContent = 'Transcribing…';
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
        if (msg.type === 'ping') {
          return;
        }
        if (msg.type === 'started') {
          segmentIndexBase = parseInt(msg.segment_index_base || 0, 10);
          liveMode = msg.live_mode || 'chunked';
          // On a fresh session we reset segmentIndex; on a reconnect we keep
          // it (the recorder kept advancing while disconnected) and just
          // rely on the server's base for correctness on the next clean start.
          if (!reconnecting) {
            segmentIndex = segmentIndexBase;
          }
          if (reconnecting) {
            reconnecting = false;
            reconnectAttempts = 0;
            flushPendingUploads();
            updateIndicator();
          }
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
        } else if (msg.type === 'transcript.delta') {
          // Realtime path: interim text appears between segment.ready boundaries.
          // Render as a single grey/italic node that we update in place, then
          // clear when the next segment.ready lands.
          renderInterimDelta(msg.text || '');
        } else if (msg.type === 'session.status') {
          // Realtime only — reflects connection state with the upstream provider.
          // "reconnecting" shows a warning; "connected" / "disconnected" clear it.
          updateSessionStatus(msg.state);
        } else if (msg.type === 'stopped') {
          // server confirmed stop; reload to show final transcript + status
          window.setTimeout(function () { window.location.reload(); }, 500);
        } else if (msg.type === 'error') {
          console.warn('meeting WS error:', msg.message);
        }
      };
      ws.onclose = function () {
        if (!resolved) {
          // Socket closed before we received `started`. Either the initial
          // connect failed (onerror already rejected, this is idempotent) or
          // a reconnect attempt failed — reject so the .catch in
          // handleUnexpectedClose schedules the next retry.
          reject(new Error('WebSocket closed before ready'));
          return;
        }
        if (transcribing) {
          handleUnexpectedClose();
        } else if (stoppingPendingDrain) {
          // User is stopping and the socket died mid-drain. Can't recover
          // server-side `segment_ready` events that haven't arrived yet;
          // clean up and let the server's disconnect() mark it INTERRUPTED.
          stoppingPendingDrain = false;
          shutdownLocal();
          alert('The transcription connection was lost while finalizing. Reload the page to see the partial transcript.');
        }
      };
      ws.onerror = function () {
        if (!resolved) reject(new Error('WebSocket connection failed'));
      };
    });
  }

  function handleUnexpectedClose() {
    // The transport dropped mid-session. Keep MediaRecorder running so we
    // don't lose audio, and try to reopen the socket. Only give up — and
    // alert the user — after exhausting the retry budget.
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    if (reconnectAttempts >= RECONNECT_MAX_ATTEMPTS) {
      reconnecting = false;
      shutdownLocal();
      alert('Transcription disconnected and could not reconnect. Reload the page to resume.');
      return;
    }
    reconnectAttempts += 1;
    reconnecting = true;
    const backoff = Math.min(
      RECONNECT_BACKOFF_MS_MAX,
      RECONNECT_BACKOFF_MS_BASE * Math.pow(2, reconnectAttempts - 1),
    );
    updateIndicator();
    reconnectTimer = setTimeout(function () {
      reconnectTimer = null;
      openWebSocket().catch(function () {
        // openWebSocket's internal onclose already re-enters this function
        // via the reject path — fall back to an explicit retry schedule in
        // case it didn't (e.g. `onerror` before open).
        if (reconnecting && transcribing) {
          handleUnexpectedClose();
        }
      });
    }, backoff);
  }

  function flushPendingUploads() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    while (pendingUploads.length > 0) {
      const item = pendingUploads.shift();
      try {
        ws.send(JSON.stringify({
          type: 'chunk_meta',
          segment_index: item.idx,
          byte_length: item.ab.byteLength,
          mime: item.mime,
          start_offset_seconds: item.offsetSec,
        }));
        ws.send(item.ab);
      } catch (err) {
        console.warn('flushPendingUploads: send failed, re-queuing', err);
        pendingUploads.unshift(item);
        return;
      }
    }
  }

  function startRecorder() {
    // Dispatch on the effective live-transcription mode announced by the
    // server. Realtime modes want a single continuous MediaRecorder; the
    // legacy chunked mode stops and restarts every 30s.
    console.log('[meetings] startRecorder: liveMode=' + liveMode);
    if (liveMode === 'realtime' || liveMode === 'realtime_with_fallback') {
      startStreamingRecorder();
    } else {
      startSegmentRecorder();
    }
  }

  function startStreamingRecorder() {
    if (!mediaStream) return;
    const opts = preferredMime ? { mimeType: preferredMime } : undefined;
    let recorder;
    try {
      recorder = new MediaRecorder(mediaStream, opts);
    } catch (err) {
      console.error('MediaRecorder constructor failed', err);
      return;
    }
    mediaRecorder = recorder;
    // Single continuous recording. The first ondataavailable burst contains
    // the Matroska init segment; every subsequent burst is a cluster that
    // extends the same WebM stream. Server-side ffmpeg decodes them as one
    // continuous input — no timestamp resets, no DTS warnings.
    recorder.ondataavailable = function (ev) {
      if (!ev.data || ev.data.size === 0) return;
      ev.data.arrayBuffer().then(function (ab) { pushStreamBurst(ab); });
    };
    recorder.onstop = function () {
      // If we're still transcribing when this fires (user paused mic, then
      // resumed via visibility handler) restart a fresh continuous session.
      if (transcribing && mediaStream && mediaStream.active) {
        startStreamingRecorder();
      }
    };
    try {
      recorder.start(STREAMING_TIMESLICE_MS);
    } catch (err) {
      console.error('recorder.start failed', err);
      return;
    }
  }

  function pushStreamBurst(ab) {
    lastChunkSentAt = Date.now();
    watchdogRestartAttempted = false;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      // Socket is down — drop the burst. Unlike chunked mode we can't meaningfully
      // buffer: ffmpeg expects a continuous stream, and replaying mid-stream
      // bytes out of order would corrupt it. The reconnect code will start a
      // fresh MediaRecorder on re-open.
      return;
    }
    const mime = preferredMime || 'audio/webm';
    const meta = {
      type: 'chunk_meta',
      segment_index: segmentIndex,
      byte_length: ab.byteLength,
      mime: mime,
      start_offset_seconds: 0,
    };
    segmentIndex += 1;
    try {
      ws.send(JSON.stringify(meta));
      ws.send(ab);
    } catch (err) {
      console.warn('failed to send stream burst', err);
    }
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
    if (!blob || blob.size === 0) return;
    const mime = preferredMime || 'audio/webm';
    blob.arrayBuffer().then(function (ab) {
      lastChunkSentAt = Date.now();
      watchdogRestartAttempted = false;
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        // Socket is down (reconnect in progress or initial open races). Buffer
        // the chunk so we can flush it once the next `started` message lands.
        pendingUploads.push({ idx: idx, offsetSec: offsetSec, ab: ab, mime: mime });
        return;
      }
      const meta = {
        type: 'chunk_meta',
        segment_index: idx,
        byte_length: ab.byteLength,
        mime: mime,
        start_offset_seconds: offsetSec,
      };
      try {
        ws.send(JSON.stringify(meta));
        ws.send(ab);
      } catch (err) {
        console.warn('failed to send chunk, buffering for retry', err);
        pendingUploads.push({ idx: idx, offsetSec: offsetSec, ab: ab, mime: mime });
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
    installVisibilityHandler();
    installTrackEndedHandler(mediaStream);
    startWatchdog();
    startRecorder();
  }

  function shutdownLocal(opts) {
    opts = opts || {};
    transcribing = false;
    stopElapsedTimer();
    stopLevelMeter();
    stopWatchdog();
    removeVisibilityHandler();
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
    reconnecting = false;
    if (segmentTimer) { clearTimeout(segmentTimer); segmentTimer = null; }
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      try { mediaRecorder.stop(); } catch (e) {}
    }
    mediaRecorder = null;
    removeTrackEndedHandler();
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
    if (stopBtn) {
      stopBtn.disabled = true;
      if (stopBtnLabel) stopBtnLabel.textContent = 'Stopping…';
      if (stopBtnIcon) stopBtnIcon.classList.add('hidden');
      if (stopBtnSpinner) stopBtnSpinner.classList.remove('hidden');
    }
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

    function applyCapabilityGating(modelChoice) {
      // Disable the live Start button and show the diarize banner when the
      // selected model doesn't support live streaming. The server re-enforces
      // this — the gating is a UX affordance, not a security boundary.
      const supportsLive = !!(modelChoice && modelChoice.supports_live_streaming);
      const diarizeBanner = document.getElementById('diarize-live-banner');
      if (transcribeBtn) {
        transcribeBtn.disabled = !supportsLive || transcribeBtn.dataset.unsupported === '1';
        if (!supportsLive) {
          transcribeBtn.classList.add('opacity-50', 'cursor-not-allowed');
          transcribeBtn.title = 'Live transcription is not available for this model. Upload an audio file instead.';
        } else {
          transcribeBtn.classList.remove('opacity-50', 'cursor-not-allowed');
          transcribeBtn.removeAttribute('title');
        }
      }
      if (diarizeBanner) {
        if (supportsLive) diarizeBanner.classList.add('hidden');
        else diarizeBanner.classList.remove('hidden');
      }
    }

    function selectModel(modelId) {
      if (!modelId || modelId === selected) return;
      selected = modelId;
      const found = choices.find(function (c) { return c.id === modelId; });
      if (label && found) label.textContent = found.display_name;
      renderOptions();
      applyCapabilityGating(found);
      // Persist on the meeting record so reload + future WS connects use it.
      postMetadata({ transcription_model: modelId });
      // Hot-update the active session if any.
      if (ws && ws.readyState === WebSocket.OPEN) {
        try { ws.send(JSON.stringify({ type: 'set_model', model_id: modelId })); } catch (e) {}
      }
    }

    // Initial gating: whatever was selected at page render.
    applyCapabilityGating(choices.find(function (c) { return c.id === selected; }));

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
    const cancelBtn = document.getElementById('cancel-upload-transcription-btn');
    let stopped = false;

    if (cancelBtn) {
      cancelBtn.addEventListener('click', function () {
        if (!confirm('Stop transcription? Any chunks already transcribed will be kept as a partial transcript.')) return;
        cancelBtn.disabled = true;
        cancelBtn.classList.add('opacity-50');
        const csrf = document.querySelector('meta[name="csrf-token"]');
        fetch('/meetings/' + meetingUuid + '/cancel-transcription/', {
          method: 'POST',
          credentials: 'same-origin',
          headers: {
            'X-CSRFToken': csrf ? csrf.getAttribute('content') : '',
            'X-Requested-With': 'XMLHttpRequest',
          },
        })
          .then(function (r) {
            if (!r.ok) throw new Error('cancel ' + r.status);
            // Let the next poll observe the FAILED status and reload the page.
            showLabel('Stopping — waiting for current chunk to finish…');
          })
          .catch(function (err) {
            console.warn('cancel transcription error:', err);
            cancelBtn.disabled = false;
            cancelBtn.classList.remove('opacity-50');
          });
      });
    }

    // While upload transcription is in flight, disable every action button
    // on the page so the user can't navigate the meeting into an inconsistent
    // state (e.g. summarize in chat before the transcript is finalized).
    setActionsBusy(true);

    function showLabel(text) {
      if (transcribingIndicator) transcribingIndicator.classList.remove('hidden');
      if (transcribingIndicatorLabel) transcribingIndicatorLabel.textContent = text;
      if (uploadBannerLabel) uploadBannerLabel.textContent = text;
    }

    // Compute the percentage to display. We start at half a tick (so 4 chunks
    // begins at 12.5%, not 0%) — this gives the user immediate visual feedback
    // that work has begun. Once at least one chunk has finished, we switch to
    // the true done/total ratio so the percentages land on familiar fractions
    // (25%, 50%, 75%, 100%) rather than being permanently offset.
    function computePct(done, total) {
      if (total <= 0) return null;
      const ratio = done > 0 ? (done / total) : (0.5 / total);
      return Math.min(100, Math.max(0, ratio * 100));
    }

    function setProgress(pct) {
      const width = pct !== null ? pct.toFixed(1) + '%' : null;
      if (progressWrap && progressBar) {
        if (width !== null) {
          progressWrap.classList.remove('hidden');
          progressBar.style.width = width;
        } else {
          progressWrap.classList.add('hidden');
        }
      }
      if (uploadBannerProgressWrap && uploadBannerProgressBar) {
        if (width !== null) {
          uploadBannerProgressWrap.classList.remove('hidden');
          uploadBannerProgressBar.style.width = width;
        } else {
          uploadBannerProgressWrap.classList.add('hidden');
        }
      }
    }

    let lastTranscriptAt = null;

    function renderTranscript(text) {
      if (!transcriptPane) return;
      // Replace the pane's innerText (not innerHTML) so a transcript
      // containing angle brackets can't inject markup.
      transcriptPane.textContent = text || '';
    }

    function poll() {
      if (stopped) return;
      // Ask for the transcript body only when it's changed since our last
      // snapshot. First poll unconditionally pulls it so the user sees
      // whatever partial text the server has already flushed.
      const needTranscript = lastTranscriptAt === null;
      const qs = needTranscript ? '?include_transcript=1' : '';
      fetch(url + qs, { credentials: 'same-origin', headers: { 'X-Requested-With': 'XMLHttpRequest' } })
        .then(function (r) { if (!r.ok) throw new Error('progress ' + r.status); return r.json(); })
        .then(function (data) {
          if (stopped) return;
          const status = data.status;
          const chunksTotal = parseInt(data.chunks_total || 0, 10);
          const chunksDone = parseInt(data.chunks_done || 0, 10);
          const updatedAt = data.transcript_updated_at || null;

          if (typeof data.transcript === 'string') {
            renderTranscript(data.transcript);
            lastTranscriptAt = updatedAt;
          } else if (updatedAt && updatedAt !== lastTranscriptAt) {
            // Transcript changed since last time — fetch it on the next poll
            // rather than a second round-trip right now. Trigger immediately
            // so the user doesn't wait the full poll interval for fresh text.
            fetch(url + '?include_transcript=1', {
              credentials: 'same-origin',
              headers: { 'X-Requested-With': 'XMLHttpRequest' },
            })
              .then(function (r) { return r.ok ? r.json() : null; })
              .then(function (d) {
                if (!stopped && d && typeof d.transcript === 'string') {
                  renderTranscript(d.transcript);
                  lastTranscriptAt = d.transcript_updated_at || null;
                }
              })
              .catch(function () { /* ignore; next poll will retry */ });
          }

          if (status === 'ready' || status === 'failed') {
            stopped = true;
            window.location.reload();
            return;
          }
          if (chunksTotal > 0) {
            const pct = computePct(chunksDone, chunksTotal);
            showLabel('Transcribing — ' + Math.round(pct) + '% done…');
            setProgress(pct);
          } else {
            showLabel('Transcribing…');
            setProgress(null);
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
    setProgress(null);
    poll();
  })();
})();
