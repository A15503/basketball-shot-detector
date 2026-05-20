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

# ─── Feedback Function ────────────────────────────────────────
def generate_feedback(elbow_angle, knee_angle, wrist_angle, elbow_alignment, hip_shoulder_diff):
    feedback = []
    score = 0
    total = 5

    # 1. Elbow angle
    if 80 <= elbow_angle <= 110:
        feedback.append(("[OK] Elbow angle", f"{elbow_angle} deg — Good L-shape at release!"))
        score += 1
    elif elbow_angle < 80:
        feedback.append(("[!!] Elbow angle", f"{elbow_angle} deg — Too bent, aim for 80-110 deg"))
    else:
        feedback.append(("[!!] Elbow angle", f"{elbow_angle} deg — Too straight, keep elbow bent at release"))

    # 2. Knee bend
    if 100 <= knee_angle <= 140:
        feedback.append(("[OK] Knee bend", f"{knee_angle} deg — Good power from legs!"))
        score += 1
    elif knee_angle > 140:
        feedback.append(("[!!] Knee bend", f"{knee_angle} deg — Bend knees more for better power"))
    else:
        feedback.append(("[!!] Knee bend", f"{knee_angle} deg — Knees too bent at release"))

    # 3. Wrist follow through
    if wrist_angle < 40:
        feedback.append(("[OK] Wrist follow through", f"{wrist_angle} deg — Good snap!"))
        score += 1
    else:
        feedback.append(("[!!] Wrist follow through", f"{wrist_angle} deg — Snap wrist down more (goose neck)"))

    # 4. Elbow alignment (elbow X vs wrist X)
    if elbow_alignment < 0.05:
        feedback.append(("[OK] Elbow alignment", f"{elbow_alignment:.3f} — Elbow nicely under the ball!"))
        score += 1
    else:
        feedback.append(("[!!] Elbow alignment", f"{elbow_alignment:.3f} — Elbow is flaring out, tuck it in"))

    # 5. Hip-shoulder alignment
    if hip_shoulder_diff < 0.1:
        feedback.append(("[OK] Body alignment", f"{hip_shoulder_diff:.3f} — Good straight form!"))
        score += 1
    else:
        feedback.append(("[!!] Body alignment", f"{hip_shoulder_diff:.3f} — Body is twisted, square up to the basket"))

    return feedback, score, total

# ─── Setup MediaPipe ──────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.PoseLandmarker.create_from_options(options)

# ─── First Pass — Find Release Frame ──────────────────────────
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
        wrist_y_per_frame.append((frame_count, lm[16].y))
    else:
        wrist_y_per_frame.append((frame_count, 1.0))

cap.release()

# Find biggest drop
biggest_drop = 0
release_frame = 0
for i in range(1, len(wrist_y_per_frame)):
    prev_y    = wrist_y_per_frame[i-1][1]
    current_y = wrist_y_per_frame[i][1]
    drop      = prev_y - current_y
    if drop > biggest_drop:
        biggest_drop  = drop
        release_frame = wrist_y_per_frame[i][0]

print(f"Release frame: {release_frame}")

# ─── Second Pass — Full Analysis ──────────────────────────────
print("Pass 2: Analyzing...")

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

        # Key joints
        r_shoulder    = get_point(lm[12])
        r_elbow       = get_point(lm[14])
        r_wrist       = get_point(lm[16])
        r_index       = get_point(lm[20])
        r_hip         = get_point(lm[24])
        r_knee        = get_point(lm[26])
        r_ankle       = get_point(lm[28])

        # Calculate angles
        elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        knee_angle  = calculate_angle(r_hip, r_knee, r_ankle)
        wrist_angle = calculate_angle(r_elbow, r_wrist, r_index)

        # Elbow alignment (normalized coordinates)
        elbow_alignment   = abs(lm[14].x - lm[16].x)

        # Hip-shoulder alignment
        hip_shoulder_diff = abs(lm[12].x - lm[24].x)

        # Generate feedback
        feedback, score, total = generate_feedback(
            elbow_angle, knee_angle, wrist_angle,
            elbow_alignment, hip_shoulder_diff
        )

        # Print report
        print("\n" + "="*40)
        print("   BASKETBALL SHOT ANALYSIS REPORT")
        print("="*40)
        print(f"Release Frame: {release_frame}")
        print(f"Overall Score: {score}/{total}\n")
        for category, message in feedback:
            print(f"{category}: {message}")
        print("="*40)

        # Save report to text file
        with open("shot_report.txt", "w") as f:
            f.write("BASKETBALL SHOT ANALYSIS REPORT\n")
            f.write("="*40 + "\n")
            f.write(f"Release Frame: {release_frame}\n")
            f.write(f"Overall Score: {score}/{total}\n\n")
            for category, message in feedback:
                f.write(f"{category}: {message}\n")
            f.write("="*40 + "\n")
        print("\nReport saved to shot_report.txt")

        # Draw on frame
        for point in [r_shoulder, r_elbow, r_wrist, r_hip, r_knee, r_ankle, r_index]:
            cv2.circle(frame, tuple(point), 8, (0, 255, 0), -1)

        cv2.line(frame, tuple(r_shoulder), tuple(r_elbow), (255, 255, 0), 3)
        cv2.line(frame, tuple(r_elbow), tuple(r_wrist), (255, 255, 0), 3)
        cv2.line(frame, tuple(r_wrist), tuple(r_index), (255, 165, 0), 3)
        cv2.line(frame, tuple(r_hip), tuple(r_knee), (255, 0, 255), 3)
        cv2.line(frame, tuple(r_knee), tuple(r_ankle), (255, 0, 255), 3)

        cv2.putText(frame, f"Score: {score}/{total}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Elbow: {elbow_angle} deg", tuple(r_elbow),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Knee: {knee_angle} deg", tuple(r_knee),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Wrist: {wrist_angle} deg", tuple(r_wrist),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Shot Analysis", frame)
        cv2.imwrite("release_frame.jpg", frame)
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
print("Done!")