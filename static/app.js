/* static/app.js */

(() => {
  "use strict";

  /* ================= configuration ================= */
  const COUNTDOWN_SECONDS = 10;
  const RECORD_START_AT   = 5;
  const DWELL_MS          = 2000;

  // 11 dots arranged as a rectangular grid across the whole viewport.
  // Layout:  4 on top  ·  3 in middle  ·  4 on bottom
  // Values are percentages of the viewport (x = left, y = top).
  // Order = gaze sequence:  centre first, then the 4 corners, then the
  // 4 near-edges, then the two side mid-points.
  const DOT_LAYOUT = [
    { id: 0,  label: "center",           x: 50, y: 50 },
    { id: 1,  label: "top-left",         x:  6, y:  8 },
    { id: 2,  label: "top-right",        x: 94, y:  8 },
    { id: 3,  label: "bottom-right",     x: 94, y: 92 },
    { id: 4,  label: "bottom-left",      x:  6, y: 92 },
    { id: 5,  label: "top-mid-left",     x: 33, y:  8 },
    { id: 6,  label: "top-mid-right",    x: 66, y:  8 },
    { id: 7,  label: "bottom-mid-right", x: 66, y: 92 },
    { id: 8,  label: "bottom-mid-left",  x: 33, y: 92 },
    { id: 9,  label: "mid-left",         x:  6, y: 50 },
    { id: 10, label: "mid-right",        x: 94, y: 50 },
  ];

  /* ================= helpers ================= */
  const $ = (id) => document.getElementById(id);

  const screens = {
    start:       $("screen-start"),
    permission:  $("screen-permission"),
    countdown:   $("screen-countdown"),
    calibration: $("screen-calibration"),
    done:        $("screen-done"),
  };

  function showScreen(name) {
    Object.values(screens).forEach((s) => s.classList.remove("active"));
    screens[name].classList.add("active");
  }

  /* ================= state ================= */
  let stream = null;
  let recorder = null;
  let chunks = [];
  let sessionId = null;
  let screenInfo = null;
  let recordingStartedAt = null;
  let dotNodes = [];
  let fakeTimerHandle = null;

  /* ================= screen dimensions ================= */
  function collectScreenInfo() {
    return {
      screen: {
        width:       window.screen.width,
        height:      window.screen.height,
        availWidth:  window.screen.availWidth,
        availHeight: window.screen.availHeight,
        colorDepth:  window.screen.colorDepth,
        pixelDepth:  window.screen.pixelDepth,
        orientation: (window.screen.orientation && window.screen.orientation.type) || null,
      },
      window: {
        innerWidth:  window.innerWidth,
        innerHeight: window.innerHeight,
        outerWidth:  window.outerWidth,
        outerHeight: window.outerHeight,
      },
      devicePixelRatio: window.devicePixelRatio || 1,
      userAgent: navigator.userAgent,
      capturedAt: new Date().toISOString(),
    };
  }

  /* ================= camera ================= */
  async function ensureCamera() {
    if (stream && stream.active) return true;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      $("permMsg").textContent =
        "This browser does not support camera access. Please use Chrome, Edge or Firefox on localhost/HTTPS.";
      return false;
    }

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
      return true;
    } catch (err) {
      $("permMsg").textContent =
        "Camera access was not given. Please allow camera access to continue. (" + err.name + ")";
      return false;
    }
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
  }

  /* ================= session ================= */
  async function createSession(info) {
    try {
      const res = await fetch("/api/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ screen: info }),
      });
      const data = await res.json();
      return data.session_id || "session_" + Date.now();
    } catch (e) {
      return "session_" + Date.now();
    }
  }

  /* ================= countdown ================= */
  function runCountdown() {
    return new Promise((resolve) => {
      let remaining = COUNTDOWN_SECONDS;

      $("countNumber").textContent = remaining;
      $("countHint").textContent = "Get ready…";

      const timer = setInterval(() => {
        remaining -= 1;

        if (remaining === RECORD_START_AT) {
          startRecording();
          $("countHint").textContent = "● Recording…";
        }

        $("countNumber").textContent = Math.max(remaining, 0);

        if (remaining <= 0) {
          clearInterval(timer);
          resolve();
        }
      }, 1000);
    });
  }

  /* ================= recording ================= */
  function pickMimeType() {
    const options = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
    for (const o of options) {
      if (window.MediaRecorder && MediaRecorder.isTypeSupported(o)) return o;
    }
    return "";
  }

  function startRecording() {
    if (!stream || recorder) return;

    chunks = [];
    const mime = pickMimeType();

    try {
      recorder = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
    } catch (e) {
      recorder = new MediaRecorder(stream);
    }

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    };

    recorder.start(1000);
    recordingStartedAt = new Date().toISOString();
  }

  function stopRecording() {
    return new Promise((resolve) => {
      if (!recorder || recorder.state === "inactive") {
        resolve(null);
        return;
      }
      recorder.onstop = () => {
        const type = recorder.mimeType || "video/webm";
        resolve(chunks.length ? new Blob(chunks, { type }) : null);
      };
      recorder.stop();
    });
  }

  /* ================= fake exam timer ================= */
  function startFakeTimer() {
    let total = 45 * 60; // 45:00
    const el = $("examTimer");
    const render = () => {
      const h = String(Math.floor(total / 3600)).padStart(2, "0");
      const m = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
      const s = String(total % 60).padStart(2, "0");
      el.textContent = `${h}:${m}:${s}`;
    };
    render();
    fakeTimerHandle = setInterval(() => {
      if (total > 0) total -= 1;
      render();
    }, 1000);
  }

  function stopFakeTimer() {
    if (fakeTimerHandle) {
      clearInterval(fakeTimerHandle);
      fakeTimerHandle = null;
    }
  }

  /* ================= dot placement (viewport corners) ================= */
  function placeDotNode(node, desc) {
    const W = window.innerWidth;
    const H = window.innerHeight;
    node.style.left = (W * desc.x / 100) + "px";
    node.style.top  = (H * desc.y / 100) + "px";
  }

  function buildDots() {
    const layer = $("dotLayer");
    layer.innerHTML = "";
    dotNodes = [];

    DOT_LAYOUT.forEach((desc) => {
      const node = document.createElement("div");
      node.className = "dot";
      node.dataset.id = String(desc.id);

      const inner = document.createElement("span");
      inner.className = "dot-label";
      inner.textContent = String(desc.id);
      node.appendChild(inner);

      placeDotNode(node, desc);
      layer.appendChild(node);
      dotNodes.push(node);
    });
  }

  function repositionDots() {
    if (!dotNodes.length) return;
    DOT_LAYOUT.forEach((desc, i) => {
      const node = dotNodes[i];
      if (node) placeDotNode(node, desc);
    });
  }

  window.addEventListener("resize", repositionDots);

  function highlightDot(index) {
    dotNodes.forEach((n) => n.classList.remove("active"));
    if (index >= 0 && index < dotNodes.length) {
      dotNodes[index].classList.add("active");
      const d = DOT_LAYOUT[index];
      $("promptText").textContent =
        "Look at dot " + d.id + "  (" + d.label + ")";
    }
  }

  function runCalibration() {
    return new Promise((resolve) => {
      showScreen("calibration");
      startFakeTimer();

      // one frame so layout settles, then build dots
      requestAnimationFrame(() => {
        buildDots();

        let i = 0;
        const step = () => {
          if (i >= dotNodes.length) {
            highlightDot(-1);
            $("promptText").textContent = "Calibration finished.";
            stopFakeTimer();
            resolve();
            return;
          }
          highlightDot(i);
          i += 1;
          setTimeout(step, DWELL_MS);
        };

        setTimeout(step, 800);
      });
    });
  }

  /* ================= upload ================= */
  async function uploadResult(blob) {
    const fd = new FormData();
    fd.append("session_id", sessionId);
    fd.append(
      "metadata",
      JSON.stringify({
        screen: screenInfo,
        dots: DOT_LAYOUT,
        countdownSeconds: COUNTDOWN_SECONDS,
        recordStartAt: RECORD_START_AT,
        dwellMs: DWELL_MS,
        recordingStartedAt,
        finishedAt: new Date().toISOString(),
      })
    );
    fd.append("video", blob, sessionId + ".webm");

    const res = await fetch("/api/upload", { method: "POST", body: fd });
    if (!res.ok) throw new Error("Upload failed with status " + res.status);
    return res.json();
  }

  /* ================= full session ================= */
  async function runSession() {
    screenInfo = collectScreenInfo();
    sessionId = await createSession(screenInfo);

    showScreen("countdown");
    await runCountdown();

    await runCalibration();

    const blob = await stopRecording();
    stopCamera();

    showScreen("done");

    if (!blob) {
      $("doneMsg").textContent = "No video was recorded.";
      return;
    }

    $("doneMsg").textContent = "Saving video to the computer…";

    try {
      const result = await uploadResult(blob);
      $("doneMsg").textContent =
        "Saved as  " + result.video_file + "   (" + Math.round(result.bytes / 1024) + " KB)";
    } catch (e) {
      $("doneMsg").textContent = "Could not save to the server: " + e.message;
    }

    const url = URL.createObjectURL(blob);
    $("preview").src = url;
    $("downloadLink").href = url;
  }

  /* ================= button handlers ================= */
  $("startBtn").addEventListener("click", async () => {
    $("startBtn").disabled = true;
    const ok = await ensureCamera();
    $("startBtn").disabled = false;

    if (!ok) {
      showScreen("permission");
      return;
    }
    await runSession();
  });

  $("retryBtn").addEventListener("click", async () => {
    $("retryBtn").disabled = true;
    const ok = await ensureCamera();
    $("retryBtn").disabled = false;

    if (!ok) return;
    await runSession();
  });

  $("restartBtn").addEventListener("click", () => {
    const preview = $("preview");
    if (preview.src) URL.revokeObjectURL(preview.src);
    preview.removeAttribute("src");
    $("doneMsg").textContent = "";
    showScreen("start");
  });
})();