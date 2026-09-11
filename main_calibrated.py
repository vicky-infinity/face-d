import cv2
import time
import math
import json
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

# ------------------------------------------------------------
# Landmark indices
# ------------------------------------------------------------
# LEFT eye on SCREEN (person's right eye)
L_EYE_INNER = 133      # inner corner (near nose)
L_EYE_OUTER = 33       # outer corner
L_EYE_TOP   = 159
L_EYE_BOTTOM = 145
L_IRIS      = 468

# RIGHT eye on SCREEN (person's left eye)
R_EYE_INNER = 362
R_EYE_OUTER = 263
R_EYE_TOP   = 386
R_EYE_BOTTOM = 374
R_IRIS      = 473

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def iris_offset_h(face, iris_idx, inner_idx, outer_idx):
    """Signed horizontal offset of the iris from the eye's center, in image x
    (not distance-from-corner) so both eyes agree in sign for the same gaze
    direction — the two eyes move together (conjugate gaze), and averaging a
    per-eye "distance from inner corner" ratio cancels the signal out."""
    iris  = face[iris_idx]
    inner = face[inner_idx]
    outer = face[outer_idx]

    cx = (inner.x + outer.x) / 2.0
    dx, dy = outer.x - inner.x, outer.y - inner.y
    eye_width = math.sqrt(dx*dx + dy*dy)
    if eye_width == 0:
        return 0.0

    return (iris.x - cx) / (eye_width / 2.0)

def iris_offset_v(face, iris_idx, top_idx, bottom_idx):
    """Signed vertical offset of the iris from the eye's center (negative=up, positive=down)."""
    iris   = face[iris_idx]
    top    = face[top_idx]
    bottom = face[bottom_idx]

    cy = (top.y + bottom.y) / 2.0
    eye_height = bottom.y - top.y
    if eye_height == 0:
        return 0.0

    return (iris.y - cy) / (eye_height / 2.0)

def gaze_from_face(face):
    gaze_h = (iris_offset_h(face, L_IRIS, L_EYE_INNER, L_EYE_OUTER) +
              iris_offset_h(face, R_IRIS, R_EYE_INNER, R_EYE_OUTER)) / 2.0
    gaze_v = (iris_offset_v(face, L_IRIS, L_EYE_TOP, L_EYE_BOTTOM) +
              iris_offset_v(face, R_IRIS, R_EYE_TOP, R_EYE_BOTTOM)) / 2.0
    return gaze_h, gaze_v

def dot(frame, lm, color, r=3):
    h, w = frame.shape[:2]
    cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), r, color, -1)

# ------------------------------------------------------------
# Video-based calibration settings
# ------------------------------------------------------------
CALIB_VIDEO_DIR = "calibration_video"
VIDEO_EXTS = (".mp4", ".webm", ".mov", ".avi", ".mkv")

# static/app.js's runCalibration(): after the countdown reaches RECORD_START_AT
# and recording starts, there's one requestAnimationFrame + a fixed 800ms pause
# before dot 0 lights up. This mirrors that constant so precise-mode slicing
# lines up with the actual recorded video.
DOT_STEP_LEAD_MS = 800

# Fallback mode (no _meta.json found): assume the first CENTER_FALLBACK_SEC
# seconds are the user looking at the screen center, and everything after
# that is the user scanning the screen's corners/edges.
CENTER_FALLBACK_SEC = 2.0

# Ignore the start/end of each dot's dwell window — the eye is still
# saccading toward the new dot right after it appears.
TRIM_START_MS = 400
TRIM_END_MS   = 100

# The learned "how far the eyes normally move" range is NOT a violation.
# We pad it by this fraction before treating anything further as a violation,
# so a genuine glance to the edge you calibrated against doesn't false-alarm.
MARGIN_RATIO = 0.15

# Safety floor so a degenerate/too-tight calibration (e.g. very few samples,
# or a very still calibration video) doesn't produce a hair-trigger threshold.
MIN_THRESH = 0.10

MIN_SAMPLES_PER_POINT = 3

# Debounce: require sustained deviation before declaring an alert, so a
# single blink/jitter frame doesn't flip the status
ALERT_FRAMES = 15   # ~0.5s at 30fps

# ------------------------------------------------------------
# Locate the calibration video (+ optional metadata) to use
# ------------------------------------------------------------
def find_calibration_video(folder):
    if not os.path.isdir(folder):
        return None
    candidates = [f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXTS)]
    if not candidates:
        return None
    candidates.sort(key=lambda f: os.path.getmtime(os.path.join(folder, f)), reverse=True)
    return os.path.join(folder, candidates[0])

def load_meta_for_video(video_path):
    base, _ = os.path.splitext(video_path)
    meta_path = base + "_meta.json"
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None

# ------------------------------------------------------------
# Calibration from a recorded video
# ------------------------------------------------------------
def calibrate_from_video(video_path, meta, detector):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open calibration video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 240:
        print(f"  (warning: video reported an unusable fps={fps!r}, assuming 30fps)")
        fps = 30.0
    frame_period_ms = 1000.0 / fps

    dots_meta = None
    lead_in_ms = None
    dwell_ms = None
    if meta and isinstance(meta.get("metadata"), dict) and meta["metadata"].get("dots"):
        m = meta["metadata"]
        dots_meta = sorted(m["dots"], key=lambda d: d["id"])
        dwell_ms = float(m.get("dwellMs", 2000))
        record_start_at = float(m.get("recordStartAt", 5))
        lead_in_ms = record_start_at * 1000.0 + DOT_STEP_LEAD_MS
        samples_by_dot = {d["id"]: [] for d in dots_meta}
    else:
        center_samples = []
        scan_samples = []

    frame_idx = 0
    ts_prev = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_ms = frame_idx * frame_period_ms
        frame_idx += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(t_ms)
        if ts <= ts_prev:
            ts = ts_prev + 1
        ts_prev = ts

        result = detector.detect_for_video(mp_image, ts)
        if not result.face_landmarks:
            continue
        face = result.face_landmarks[0]
        if len(face) < 478:
            continue

        gaze_h, gaze_v = gaze_from_face(face)

        if dots_meta is not None:
            if t_ms < lead_in_ms:
                continue
            idx = int((t_ms - lead_in_ms) // dwell_ms)
            if idx < 0 or idx >= len(dots_meta):
                continue
            window_pos = (t_ms - lead_in_ms) - idx * dwell_ms
            if window_pos < TRIM_START_MS or window_pos > (dwell_ms - TRIM_END_MS):
                continue
            dot_id = dots_meta[idx]["id"]
            samples_by_dot[dot_id].append((gaze_h, gaze_v))
        else:
            if t_ms < CENTER_FALLBACK_SEC * 1000.0:
                center_samples.append((gaze_h, gaze_v))
            else:
                scan_samples.append((gaze_h, gaze_v))

    cap.release()

    def mean(samples):
        n = len(samples)
        return (sum(s[0] for s in samples) / n, sum(s[1] for s in samples) / n)

    if dots_meta is not None:
        center_entry = next((d for d in dots_meta if d.get("label") == "center"), dots_meta[0])
        center_samples = samples_by_dot.get(center_entry["id"], [])
        if len(center_samples) < MIN_SAMPLES_PER_POINT:
            raise RuntimeError(
                "Calibration failed: too few usable frames for the center dot "
                f"({len(center_samples)}). Re-record with better lighting/face visibility."
            )
        baseline_h, baseline_v = mean(center_samples)

        max_h_pos = max_h_neg = max_v_pos = max_v_neg = 0.0
        print(f"Calibration points ({len(dots_meta)} dots):")
        for d in dots_meta:
            pts = samples_by_dot.get(d["id"], [])
            label = d.get("label", str(d["id"]))
            if len(pts) < MIN_SAMPLES_PER_POINT:
                print(f"  {label:<18} SKIPPED (only {len(pts)} usable frames)")
                continue
            gh, gv = mean(pts)
            dh, dv = gh - baseline_h, gv - baseline_v
            print(f"  {label:<18} gaze=({gh:+.3f},{gv:+.3f})  delta=({dh:+.3f},{dv:+.3f})  n={len(pts)}")
            if d["id"] == center_entry["id"]:
                continue
            max_h_pos = max(max_h_pos, dh)
            max_h_neg = max(max_h_neg, -dh)
            max_v_pos = max(max_v_pos, dv)
            max_v_neg = max(max_v_neg, -dv)
    else:
        if len(center_samples) < MIN_SAMPLES_PER_POINT:
            raise RuntimeError(
                "Calibration failed: too few usable frames in the first "
                f"{CENTER_FALLBACK_SEC:.1f}s (center segment) of the video."
            )
        if len(scan_samples) < MIN_SAMPLES_PER_POINT:
            raise RuntimeError(
                "Calibration failed: too few usable frames after the first "
                f"{CENTER_FALLBACK_SEC:.1f}s (corner-scan segment) of the video."
            )
        baseline_h, baseline_v = mean(center_samples)
        max_h_pos = max_h_neg = max_v_pos = max_v_neg = 0.0
        for gh, gv in scan_samples:
            dh, dv = gh - baseline_h, gv - baseline_v
            max_h_pos = max(max_h_pos, dh)
            max_h_neg = max(max_h_neg, -dh)
            max_v_pos = max(max_v_pos, dv)
            max_v_neg = max(max_v_neg, -dv)
        print(f"Fallback calibration: center n={len(center_samples)}, scan n={len(scan_samples)}")

    baseline = {
        "gaze_h": baseline_h,
        "gaze_v": baseline_v,
        "thresh_h_pos": max(max_h_pos * (1.0 + MARGIN_RATIO), MIN_THRESH),
        "thresh_h_neg": max(max_h_neg * (1.0 + MARGIN_RATIO), MIN_THRESH),
        "thresh_v_pos": max(max_v_pos * (1.0 + MARGIN_RATIO), MIN_THRESH),
        "thresh_v_neg": max(max_v_neg * (1.0 + MARGIN_RATIO), MIN_THRESH),
    }
    return baseline

# ------------------------------------------------------------
# Run calibration from the video before touching the webcam
# ------------------------------------------------------------
video_path = find_calibration_video(CALIB_VIDEO_DIR)
if video_path is None:
    print(f"No calibration video found in '{CALIB_VIDEO_DIR}/'.")
    print("See calibration_video/README.md for how to record one.")
    exit()

meta = load_meta_for_video(video_path)
mode = "precise (dot metadata found)" if (meta and meta.get("metadata", {}).get("dots")) else "fallback (no metadata — center/corner-scan split)"
print(f"Using calibration video: {video_path}  [{mode}]")

calib_detector = vision.FaceLandmarker.create_from_options(options)
try:
    baseline = calibrate_from_video(video_path, meta, calib_detector)
finally:
    calib_detector.close()

print("Calibration done:")
print(f"  baseline gaze = ({baseline['gaze_h']:+.3f}, {baseline['gaze_v']:+.3f})")
print(f"  thresholds (right/left/down/up) = "
      f"{baseline['thresh_h_pos']:.3f} / {baseline['thresh_h_neg']:.3f} / "
      f"{baseline['thresh_v_pos']:.3f} / {baseline['thresh_v_neg']:.3f}")

# ------------------------------------------------------------
# Live detection (fresh detector — VIDEO mode needs its own
# monotonic timestamp sequence, separate from the calibration video's)
# ------------------------------------------------------------
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

away_streak = 0
alerting = False

start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts = int((time.time() - start_time) * 1000)

    result = detector.detect_for_video(mp_image, ts)
    frame_count += 1

    if result.face_landmarks:
        face = result.face_landmarks[0]

        if len(face) < 478:
            cv2.putText(frame, "NO IRIS DATA", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Cheating Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        gaze_h, gaze_v = gaze_from_face(face)

        # --- Deviation from the video-calibrated baseline ---
        d_gaze_h = gaze_h - baseline["gaze_h"]
        d_gaze_v = gaze_v - baseline["gaze_v"]

        looking_away = (
            d_gaze_h > baseline["thresh_h_pos"] or
            d_gaze_h < -baseline["thresh_h_neg"] or
            d_gaze_v > baseline["thresh_v_pos"] or
            d_gaze_v < -baseline["thresh_v_neg"]
        )

        if looking_away:
            away_streak += 1
        else:
            away_streak = 0

        alerting = away_streak >= ALERT_FRAMES

        if alerting:
            label = "LOOKING_AWAY - ALERT"
            color = (0, 0, 255)
        elif looking_away:
            label = "LOOKING_AWAY"
            color = (0, 165, 255)
        else:
            label = "FACING_SCREEN"
            color = (0, 255, 0)

        if frame_count % 30 == 0:
            print(f"gaze_h={d_gaze_h:+.3f} gaze_v={d_gaze_v:+.3f} streak={away_streak}")

        # --- Draw landmarks ---
        dot(frame, face[L_IRIS], (255, 0, 0), 4)
        dot(frame, face[R_IRIS], (255, 0, 0), 4)
        dot(frame, face[L_EYE_INNER], (0, 255, 255), 2)
        dot(frame, face[L_EYE_OUTER], (0, 255, 255), 2)
        dot(frame, face[R_EYE_INNER], (0, 255, 255), 2)
        dot(frame, face[R_EYE_OUTER], (0, 255, 255), 2)

        # --- Display ---
        cv2.putText(frame, label, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"gazeH={d_gaze_h:+.2f} gazeV={d_gaze_v:+.2f}",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

    else:
        away_streak += 1
        alerting = away_streak >= ALERT_FRAMES
        label = "NO FACE DETECTED" + (" - ALERT" if alerting else "")
        cv2.putText(frame, label, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Cheating Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
