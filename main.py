import cv2
import time
import math
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

detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

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

def dot(frame, lm, color, r=3):
    h, w = frame.shape[:2]
    cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), r, color, -1)

# ------------------------------------------------------------
# Calibration + detection settings
# ------------------------------------------------------------
CALIBRATION_FRAMES = 45   # ~1.5s at 30fps: user should be facing the screen

# Deviation-from-baseline thresholds (tune if too strict/loose for your setup).
# gaze_h/gaze_v are signed offsets from eye-center in units of half the eye's
# width/height (0 = centered, ~1 = iris at the corner/lid), so these are
# fractions of that range — kept tight so small eye movements still trigger.
GAZE_H_THRESH  = 0.13
GAZE_V_THRESH  = 0.13

# Debounce: require sustained deviation before declaring an alert, so a
# single blink/jitter frame doesn't flip the status
ALERT_FRAMES = 15   # ~0.5s at 30fps

baseline = None            # dict of calibrated center values, set after calibration
calib_samples = []
away_streak = 0
alerting = False

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
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

        gaze_h = (iris_offset_h(face, L_IRIS, L_EYE_INNER, L_EYE_OUTER) +
                  iris_offset_h(face, R_IRIS, R_EYE_INNER, R_EYE_OUTER)) / 2.0
        gaze_v = (iris_offset_v(face, L_IRIS, L_EYE_TOP, L_EYE_BOTTOM) +
                  iris_offset_v(face, R_IRIS, R_EYE_TOP, R_EYE_BOTTOM)) / 2.0

        # --- Calibration phase: learn this person's "facing screen" baseline ---
        if baseline is None:
            calib_samples.append((gaze_h, gaze_v))
            remaining = CALIBRATION_FRAMES - len(calib_samples)
            cv2.putText(frame, f"Calibrating... look at screen ({remaining})",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("Cheating Detector", frame)

            if len(calib_samples) >= CALIBRATION_FRAMES:
                n = len(calib_samples)
                baseline = {
                    "gaze_h": sum(s[0] for s in calib_samples) / n,
                    "gaze_v": sum(s[1] for s in calib_samples) / n,
                }
                print(f"Calibration done: {baseline}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # --- Deviation from calibrated baseline ---
        d_gaze_h = gaze_h - baseline["gaze_h"]
        d_gaze_v = gaze_v - baseline["gaze_v"]

        looking_away = (
            abs(d_gaze_h) > GAZE_H_THRESH or
            abs(d_gaze_v) > GAZE_V_THRESH
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
