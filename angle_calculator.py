import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# ─── Angle Function ───────────────────────────────────────────
def calculate_angle(a, b, c):
    """
    Calculate angle at point B, formed by A-B-C
    a, b, c = [x, y] coordinates
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b  # vector from B to A
    bc = c - b  # vector from B to C

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return round(angle, 1)

# ─── Setup MediaPipe ──────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(base_options=base_options)
detector = vision.PoseLandmarker.create_from_options(options)

# ─── Load Image ───────────────────────────────────────────────
img_cv = cv2.imread("test.jpg")
h, w = img_cv.shape[:2]

image = mp.Image.create_from_file("test.jpg")
results = detector.detect(image)

if results.pose_landmarks:
    lm = results.pose_landmarks[0]

    # ─── Get coordinates (convert from 0-1 to pixels) ─────────
    def get_point(landmark):
        return [int(landmark.x * w), int(landmark.y * h)]

    # Right arm
    r_shoulder = get_point(lm[12])
    r_elbow    = get_point(lm[14])
    r_wrist    = get_point(lm[16])

    # Right leg
    r_hip      = get_point(lm[24])
    r_knee     = get_point(lm[26])
    r_ankle    = get_point(lm[28])

    # ─── Calculate Angles ─────────────────────────────────────
    elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
    knee_angle  = calculate_angle(r_hip, r_knee, r_ankle)

    print(f"Elbow angle: {elbow_angle}°")
    print(f"Knee angle:  {knee_angle}°")

    # ─── Basic Feedback ───────────────────────────────────────
    print("\n--- Feedback ---")

    if 80 <= elbow_angle <= 110:
        print("✅ Elbow angle looks good!")
    elif elbow_angle < 80:
        print("⚠️  Elbow too bent — try to keep it around 90°")
    else:
        print("⚠️  Elbow too straight — bend it more at release")

    if 100 <= knee_angle <= 140:
        print("✅ Knee bend looks good!")
    elif knee_angle < 100:
        print("⚠️  Knees too bent")
    else:
        print("⚠️  Bend your knees more for better power")

    # ─── Draw on image ────────────────────────────────────────
    # Draw dots
    for point in [r_shoulder, r_elbow, r_wrist, r_hip, r_knee, r_ankle]:
        cv2.circle(img_cv, tuple(point), 8, (0, 255, 0), -1)

    # Draw lines
    cv2.line(img_cv, tuple(r_shoulder), tuple(r_elbow), (255, 255, 0), 3)
    cv2.line(img_cv, tuple(r_elbow), tuple(r_wrist), (255, 255, 0), 3)
    cv2.line(img_cv, tuple(r_hip), tuple(r_knee), (255, 0, 255), 3)
    cv2.line(img_cv, tuple(r_knee), tuple(r_ankle), (255, 0, 255), 3)

    # Draw angle text on image
    cv2.putText(img_cv, f"Elbow: {elbow_angle} deg", tuple(r_elbow),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img_cv, f"Knee: {knee_angle} deg", tuple(r_knee),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show and save
    cv2.imshow("Angle Analysis", img_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("result_angles.jpg", img_cv)
    print("\nSaved as result_angles.jpg")

else:
    print("❌ No pose detected.")