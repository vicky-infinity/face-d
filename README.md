# face-d

Webcam-based gaze tracker that flags when a person looks away from the
screen — e.g. for proctoring/cheating detection during an online exam.

It uses MediaPipe's Face Landmarker (with iris tracking) to work out where
the eyes are pointing, compares that to a "facing the screen" baseline
learned from a short calibration recording, and raises an alert if the
eyes stay off that baseline for too long.

## Project structure

```
main_calibrated.py     The detector — calibrates from a pre-recorded
                        video, then runs live detection.
app.py                 Local web tool to record that calibration video —
                        shows dots around the screen, records your webcam
                        while you look at each one, saves it to recordings/.
templates/, static/    Front-end for app.py.
calibration_video/     Drop the video from recordings/ here for
                        main_calibrated.py to use.
recordings/            Where app.py saves recordings.
models/                MediaPipe face_landmarker.task model file.
requirements.txt       Python dependencies.
```

## Setup

```
pip install -r requirements.txt
```

`models/face_landmarker.task` must already be present (it is, in this repo).

## How to run

### 1. Record a calibration video

```
python app.py
```

This opens a browser page.(if not open local host 5000) Click **Start**, allow camera access, and just
follow the 11 dots it shows you — center, then the corners and edges of
the screen — about 30 seconds total. When it finishes, it saves a matching
pair of files into `recordings/`:

```
session_<timestamp>.webm
session_<timestamp>_meta.json
```

You can close `app.py` (Ctrl+C) once that's saved.

### 2. Copy that pair into `calibration_video/`

Copy **both** files (same `session_<timestamp>` name) from `recordings/`
into `calibration_video/`. That's the video the detector calibrates from.
(If there's an old pair already in there from a previous run, remove it
first — only the most recently modified video is used.)

### 3. Run the detector

```
python main_calibrated.py
```

It reads the video from `calibration_video/` and prints the calibration
results to the terminal — check that each of the 11 points shows a
reasonable sample count (`n=...`) and that corner readings look
directionally sensible. Then it opens your webcam and starts live
detection:

- Green **FACING_SCREEN** — normal.
- Orange **LOOKING_AWAY** — currently outside the learned range, not yet
  sustained long enough to alert.
- Red **LOOKING_AWAY - ALERT** — sustained for ~0.5s, counts as a
  violation.

Press `q` to quit.

## How it works

**Gaze signal** — for each eye, the iris landmark's position is compared
to the eye's own center (midpoint of inner/outer corner), normalized by
eye width/height. This gives a signed `gaze_h`/`gaze_v` value per eye
(`0` = iris centered, `~1` = iris at the lid/corner). Both eyes are
averaged together since they move in sync (conjugate gaze).

**Calibration** — raw gaze values aren't `0` when someone looks at the
screen; it depends on face shape and camera angle, and varies person to
person. So a personalized baseline is learned first from the recording:

- The video's metadata says exactly which frames correspond to which of
  the 11 on-screen points (center, corners, edges), so each point's
  average gaze reading can be measured precisely.
- The **center** point's reading becomes the baseline ("facing the
  screen").
- How far the gaze moved toward each of the other 10 points, relative to
  that baseline, becomes the *real* range of normal eye movement — per
  direction (left/right/up/down independently, not one fixed number).
- A small margin (15%) is added on top of that learned range before
  anything counts as a violation, so a normal glance toward a screen edge
  doesn't false-alarm — only something clearly beyond it does.

**Detection** — every live frame, the current gaze is compared to the
baseline. If the deviation exceeds the (per-direction) threshold for
`ALERT_FRAMES` (~0.5s) *consecutively*, the status escalates to an alert.
A single blink or jitter frame doesn't trigger anything — only a
sustained deviation does. No face detected at all also counts toward the
alert streak.

## Future enhancements

- Log alert events (timestamp + duration) to a file instead of just
  printing to console, for a reviewable session report.
- Auto-pick / validate the calibration video instead of relying on the
  user to manually copy files into `calibration_video/`.
- Head-pose (not just eye/iris) as a second signal, to catch someone
  turning their whole head away rather than just their eyes.
- Multi-face handling — currently assumes exactly one face; could flag
  "second person detected" as its own event.
- Package the live detector behind `app.py` (single web UI: record
  calibration → run live monitoring → see results) instead of two
  separate scripts.
