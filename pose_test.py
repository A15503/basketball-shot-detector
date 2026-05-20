import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os

# Download the pose model if not already downloaded
model_path = "pose_landmarker_full.task"
if not os.path.exists(model_path):
    print("Downloading pose model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        model_path
    )
    print("Downloaded!")

# Setup
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

# Load your photo
image = mp.Image.create_from_file("test.jpg")

# Detect pose
results = detector.detect(image)

if results.pose_landmarks:
    print("✅ Pose detected! Landmarks found.")

    # Convert to OpenCV format for drawing
    img_cv = cv2.imread("test.jpg")
    h, w = img_cv.shape[:2]

    # Draw landmarks manually
    for landmark in results.pose_landmarks[0]:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(img_cv, (x, y), 5, (0, 255, 0), -1)

    # Print key joints
    lm = results.pose_landmarks[0]
    print(f"Right shoulder: x={lm[12].x:.2f}, y={lm[12].y:.2f}")
    print(f"Right elbow:    x={lm[14].x:.2f}, y={lm[14].y:.2f}")
    print(f"Right wrist:    x={lm[16].x:.2f}, y={lm[16].y:.2f}")
    print(f"Right knee:     x={lm[26].x:.2f}, y={lm[26].y:.2f}")

    # Show and save
    cv2.imshow("Pose Detection", img_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("result.jpg", img_cv)
    print("Saved as result.jpg")

else:
    print("❌ No pose detected. Try a clearer photo.")