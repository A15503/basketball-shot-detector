import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# ─── Angle Function ───────────────────────────────────────────
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return round(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))), 1)

# ─── Setup MediaPipe ──────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO  # ← key difference from before
)
detector = vision.PoseLandmarker.create_from_options(options)

# ─── Open Video ───────────────────────────────────────────────
cap = cv2.VideoCapture("test_shot.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

print(f"Video FPS: {fps}")
print("Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w = frame.shape[:2]

    # Convert frame to MediaPipe format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Calculate timestamp in milliseconds
    timestamp_ms = int((frame_count / fps) * 1000)

    # Detect pose
    results = detector.detect_for_video(mp_image, timestamp_ms)

    if results.pose_landmarks:
        lm = results.pose_landmarks[0]

        def get_point(landmark):
            return [int(landmark.x * w), int(landmark.y * h)]

        # Get key joints
        r_shoulder = get_point(lm[12])
        r_elbow    = get_point(lm[14])
        r_wrist    = get_point(lm[16])
        r_hip      = get_point(lm[24])
        r_knee     = get_point(lm[26])
        r_ankle    = get_point(lm[28])

        # Calculate angles
        elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        knee_angle  = calculate_angle(r_hip, r_knee, r_ankle)

        # Draw skeleton
        for point in [r_shoulder, r_elbow, r_wrist, r_hip, r_knee, r_ankle]:
            cv2.circle(frame, tuple(point), 8, (0, 255, 0), -1)

        cv2.line(frame, tuple(r_shoulder), tuple(r_elbow), (255, 255, 0), 3)
        cv2.line(frame, tuple(r_elbow), tuple(r_wrist), (255, 255, 0), 3)
        cv2.line(frame, tuple(r_hip), tuple(r_knee), (255, 0, 255), 3)
        cv2.line(frame, tuple(r_knee), tuple(r_ankle), (255, 0, 255), 3)

        # Draw angles on frame
        cv2.putText(frame, f"Elbow: {elbow_angle} deg", tuple(r_elbow),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Knee: {knee_angle} deg", tuple(r_knee),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Draw frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Shot Analysis", frame)

    # Press Q to quit, SPACE to pause
    key = cv2.waitKey(60) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        cv2.waitKey(0)  # pause until any key pressed

cap.release()
cv2.destroyAllWindows()
print(f"Done! Processed {frame_count} frames.")