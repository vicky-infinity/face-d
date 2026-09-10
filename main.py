import cv2
import time
import math
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# 1. MediaPipe Face Landmarker setup
# ============================================================
# model path 

MODEL_PATH = "models/face_landmarker.task"



base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)


# ============================================================
# 2. Open laptop camera
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


# ============================================================
# 3. Helper function
# ============================================================

def calculate_distance(point1, point2):
    """
    Calculate 2D distance between two landmarks.
    """
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )


# ============================================================
# 4. Main camera loop
# ============================================================

start_time = time.time()

while True:

    # --------------------------------------------------------
    # Capture frame
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break


    # --------------------------------------------------------
    # Convert BGR → RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # --------------------------------------------------------
    # Convert NumPy image → MediaPipe Image
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # Create timestamp
    # MediaPipe VIDEO mode requires timestamps
    # --------------------------------------------------------

    timestamp_ms = int((time.time() - start_time) * 1000)


    # --------------------------------------------------------
    # Run Face Landmarker
    # --------------------------------------------------------

    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ========================================================
    # 5. If a face is detected
    # ========================================================

    if result.face_landmarks:

        face = result.face_landmarks[0]


        # ----------------------------------------------------
        # Important landmarks
        #
        # These are approximate points used for a simple
        # left/right face-orientation estimate.
        # ----------------------------------------------------

        nose = face[1]

        left_face = face[234]

        right_face = face[454]


        # ----------------------------------------------------
        # Calculate face width
        # ----------------------------------------------------

        face_width = calculate_distance(
            left_face,
            right_face
        )


        # ----------------------------------------------------
        # Calculate nose's relative horizontal position
        # ----------------------------------------------------

        if face_width > 0:

            nose_position = (
                nose.x - left_face.x
            ) / face_width

        else:

            nose_position = 0.5


        # ----------------------------------------------------
        # Simple classification
        # ----------------------------------------------------

        # When facing the camera, the nose is approximately
        # centered between the left and right sides of the face.
        #
        # When the head turns significantly, the relative
        # position changes.
        # ----------------------------------------------------

        if 0.35 <= nose_position <= 0.65:

            label = "FACING_SCREEN"

        else:

            label = "LOOKING_AWAY"


        # ----------------------------------------------------
        # Draw selected landmarks
        # ----------------------------------------------------

        height, width = frame.shape[:2]

        for landmark in [nose, left_face, right_face]:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


        # ----------------------------------------------------
        # Display classification
        # ----------------------------------------------------

        cv2.putText(
            frame,
            label,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:

        # ----------------------------------------------------
        # No face detected
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # ========================================================
    # 6. Display camera
    # ========================================================

    cv2.imshow(
        "Face Direction Detector",
        frame
    )


    # ========================================================
    # 7. Quit when 'q' is pressed
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# 8. Cleanup
# ============================================================

cap.release()
cv2.destroyAllWindows()
detector.close()