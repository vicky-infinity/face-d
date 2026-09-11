# app.py
# Python web server for the gaze-calibration task.
# Run with:  python app.py

import json
import os
import threading
import time
import webbrowser
from datetime import datetime

from flask import Flask, jsonify, render_template, request

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 5000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # allow up to 512 MB upload


def _stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_name(name):
    keep = "".join(c for c in (name or "") if c.isalnum() or c in "_-")
    return keep or f"session_{_stamp()}"


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/session", methods=["POST"])
def create_session():
    """Called right when the countdown starts. Saves the screen dimensions."""
    payload = request.get_json(silent=True) or {}
    session_id = f"session_{_stamp()}_{int(time.time() * 1000) % 1000:03d}"

    screen_path = os.path.join(RECORDINGS_DIR, f"{session_id}_screen.json")
    with open(screen_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "screen": payload.get("screen", {}),
            },
            fh,
            indent=2,
        )

    print(f"[session] {session_id}  screen info -> {os.path.basename(screen_path)}")
    return jsonify({"session_id": session_id, "screen_file": os.path.basename(screen_path)})


@app.route("/api/upload", methods=["POST"])
def upload():
    """Receives the recorded video + metadata and writes them to disk."""
    session_id = _safe_name(request.form.get("session_id"))

    video_file = request.files.get("video")
    if video_file is None:
        return jsonify({"ok": False, "error": "no video file received"}), 400

    meta_raw = request.form.get("metadata") or "{}"
    try:
        metadata = json.loads(meta_raw)
    except json.JSONDecodeError:
        metadata = {"raw": meta_raw}

    # ---- save the video -------------------------------------------------
    video_name = f"{session_id}.webm"
    video_path = os.path.join(RECORDINGS_DIR, video_name)
    video_file.save(video_path)
    size = os.path.getsize(video_path)

    # ---- save the metadata ---------------------------------------------
    meta_name = f"{session_id}_meta.json"
    meta_path = os.path.join(RECORDINGS_DIR, meta_name)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "session_id": session_id,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "video_file": video_name,
                "video_bytes": size,
                "metadata": metadata,
            },
            fh,
            indent=2,
        )

    print(f"[saved] {video_name}  ({size/1024:.0f} KB)  +  {meta_name}")
    return jsonify(
        {"ok": True, "video_file": video_name, "meta_file": meta_name, "bytes": size}
    )


@app.route("/api/recordings")
def recordings():
    """Optional: list everything saved so far."""
    files = sorted(
        f for f in os.listdir(RECORDINGS_DIR) if f.endswith((".webm", ".json"))
    )
    return jsonify({"folder": RECORDINGS_DIR, "files": files})


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    url = f"http://{HOST}:{PORT}/"
    print("=" * 60)
    print(f"  Gaze calibration server running at {url}")
    print(f"  Recordings folder: {RECORDINGS_DIR}")
    print("  Press CTRL+C to stop.")
    print("=" * 60)

    # open the browser one second after the server starts
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host=HOST, port=PORT, debug=False, use_reloader=False, threaded=True)