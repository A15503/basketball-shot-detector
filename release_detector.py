import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# ─── Angle Function ───────────────────────────────────────────
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return round(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))), 1)

# ─── Setup MediaPipe ──────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.PoseLandmarker.create_from_options(options)

# ─── First Pass — Track Wrist Y ───────────────────────────────
print("Pass 1: Finding release frame...")

cap = cv2.VideoCapture("test_shot.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

wrist_y_per_frame = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((frame_count / fps) * 1000)

    results = detector.detect_for_video(mp_image, timestamp_ms)

    if results.pose_landmarks:
        lm = results.pose_landmarks[0]
        wrist_y = lm[16].y
        wrist_y_per_frame.append((frame_count, wrist_y))
    else:
        wrist_y_per_frame.append((frame_count, 1.0))

cap.release()

# ─── Find Release Frame — Biggest Drop in Y ───────────────────
biggest_drop = 0
release_frame = 0

for i in range(1, len(wrist_y_per_frame)):
    prev_y    = wrist_y_per_frame[i-1][1]
    current_y = wrist_y_per_frame[i][1]
    drop      = prev_y - current_y

    if drop > biggest_drop:
        biggest_drop  = drop
        release_frame = wrist_y_per_frame[i][0]

print(f"Release frame detected: Frame {release_frame} (biggest drop = {biggest_drop:.3f})")

print(f"Release frame detected: Frame {release_frame} (biggest drop = {biggest_drop:.3f})")

# Temporary debug - print top 5 biggest drops
drops = []
for i in range(1, len(wrist_y_per_frame)):
    prev_y    = wrist_y_per_frame[i-1][1]
    current_y = wrist_y_per_frame[i][1]
    drop      = prev_y - current_y
    drops.append((wrist_y_per_frame[i][0], drop))

top5 = sorted(drops, key=lambda x: x[1], reverse=True)[:5]
print("Top 5 frames with biggest wrist drop:")
for frame_num, drop in top5:
    print(f"  Frame {frame_num}: drop = {drop:.3f}")

# ─── Second Pass — Analyze Release Frame ──────────────────────
print("\nPass 2: Analyzing release frame...")

detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture("test_shot.mp4")
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((frame_count / fps) * 1000)
    results = detector.detect_for_video(mp_image, timestamp_ms)

    if frame_count != release_frame:
        continue

    if results.pose_landmarks:
        lm = results.pose_landmarks[0]

        def get_point(lm_point):
            return [int(lm_point.x * w), int(lm_point.y * h)]

        r_shoulder = get_point(lm[12])
        r_elbow    = get_point(lm[14])
        r_wrist    = get_point(lm[16])
        r_hip      = get_point(lm[24])
        r_knee     = get_point(lm[26])
        r_ankle    = get_point(lm[28])

        elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        knee_angle  = calculate_angle(r_hip, r_knee, r_ankle)

        print("\n====== SHOT ANALYSIS ======")
        print(f"Release Frame: {release_frame}")
        print(f"Elbow angle: {elbow_angle} deg")
        print(f"Knee angle:  {knee_angle} deg")
        print("\n--- Feedback ---")

        if 80 <= elbow_angle <= 110:
            print("[OK] Elbow angle: Good!")
        elif elbow_angle < 80:
            print("[!] Elbow too bent - aim for 90 deg at release")
        else:
            print("[!] Elbow too straight - bend it more")

        if 100 <= knee_angle <= 140:
            print("[OK] Knee bend: Good!")
        elif knee_angle > 140:
            print("[!] Bend your knees more for better power")
        else:
            print("[!] Knees too bent at release")

        for point in [r_shoulder, r_elbow, r_wrist, r_hip, r_knee, r_ankle]:
            cv2.circle(frame, tuple(point), 8, (0, 255, 0), -1)

        cv2.line(frame, tuple(r_shoulder), tuple(r_elbow), (255, 255, 0), 3)
        cv2.line(frame, tuple(r_elbow), tuple(r_wrist), (255, 255, 0), 3)
        cv2.line(frame, tuple(r_hip), tuple(r_knee), (255, 0, 255), 3)
        cv2.line(frame, tuple(r_knee), tuple(r_ankle), (255, 0, 255), 3)

        cv2.putText(frame, f"RELEASE FRAME {release_frame}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Elbow: {elbow_angle} deg", tuple(r_elbow),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Knee: {knee_angle} deg", tuple(r_knee),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Release Frame Analysis", frame)
        cv2.imwrite("release_frame.jpg", frame)
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
print("Done!")