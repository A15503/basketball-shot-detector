import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.angles import calculate_angle

def detect_shooting_hand(lm):
    if lm[16].y < lm[15].y:
        return "right"
    return "left"

def get_landmark_ids(shooting_hand):
    if shooting_hand == "right":
        return 12, 14, 16, 20, 24, 26, 28
    return 11, 13, 15, 19, 23, 25, 27

def analyze_video(video_path, progress_bar):
    base_options = python.BaseOptions(model_asset_path="models/pose_landmarker_full.task")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )

    # ── Pass 1 — Track wrist ──────────────────────────────────
    detector = vision.PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(video_path)
    fps        = cap.get(cv2.CAP_PROP_FPS)
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = 0
    wrist_y_per_frame = []

    progress_bar.progress(0, text="Pass 1: Scanning video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((frame_count / fps) * 1000)
        results = detector.detect_for_video(mp_image, ts)

        if results.pose_landmarks:
            lm = results.pose_landmarks[0]
            wrist_id = 16 if detect_shooting_hand(lm) == "right" else 15
            wrist_y_per_frame.append((frame_count, lm[wrist_id].y))
        else:
            wrist_y_per_frame.append((frame_count, 1.0))

        progress_bar.progress(int(frame_count / total * 40), text="Pass 1: Scanning video...")

    cap.release()

    # ── Smooth wrist tracking ─────────────────────────────────
    smoothed = []
    for i in range(len(wrist_y_per_frame)):
        if i < 2:
            smoothed.append(wrist_y_per_frame[i])
        else:
            frame_num = wrist_y_per_frame[i][0]
            avg_y = (wrist_y_per_frame[i-2][1] +
                     wrist_y_per_frame[i-1][1] +
                     wrist_y_per_frame[i][1]) / 3
            smoothed.append((frame_num, avg_y))
    wrist_y_per_frame = smoothed

    # ── Find release frame ────────────────────────────────────
    biggest_drop, release_frame = 0, 0
    for i in range(1, len(wrist_y_per_frame)):
        drop = wrist_y_per_frame[i-1][1] - wrist_y_per_frame[i][1]
        if drop > biggest_drop:
            biggest_drop  = drop
            release_frame = wrist_y_per_frame[i][0]

    # ── Pass 2 — Analyze + annotate ───────────────────────────
    progress_bar.progress(40, text="Pass 2: Analyzing and saving video...")

    detector = vision.PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    output_path = video_path.replace(".mp4", "_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    result_frame = None
    elbow_angle = knee_angle = wrist_angle = elbow_alignment = hip_shoulder_diff = None
    shooting_hand = "right"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((frame_count / fps) * 1000)
        results = detector.detect_for_video(mp_image, ts)

        if results.pose_landmarks:
            lm = results.pose_landmarks[0]
            shooting_hand = detect_shooting_hand(lm)
            shoulder_id, elbow_id, wrist_id, index_id, hip_id, knee_id, ankle_id = get_landmark_ids(shooting_hand)

            def get_point(p):
                return [int(p.x * w), int(p.y * h)]

            r_shoulder = get_point(lm[shoulder_id])
            r_elbow    = get_point(lm[elbow_id])
            r_wrist    = get_point(lm[wrist_id])
            r_index    = get_point(lm[index_id])
            r_hip      = get_point(lm[hip_id])
            r_knee     = get_point(lm[knee_id])
            r_ankle    = get_point(lm[ankle_id])

            for point in [r_shoulder, r_elbow, r_wrist, r_hip, r_knee, r_ankle, r_index]:
                cv2.circle(frame, tuple(point), 8, (0, 255, 0), -1)

            cv2.line(frame, tuple(r_shoulder), tuple(r_elbow), (255, 255, 0), 3)
            cv2.line(frame, tuple(r_elbow), tuple(r_wrist), (255, 255, 0), 3)
            cv2.line(frame, tuple(r_wrist), tuple(r_index), (255, 165, 0), 3)
            cv2.line(frame, tuple(r_hip), tuple(r_knee), (255, 0, 255), 3)
            cv2.line(frame, tuple(r_knee), tuple(r_ankle), (255, 0, 255), 3)

            cv2.putText(frame, f"{shooting_hand.upper()} HAND", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if frame_count == release_frame:
                elbow_angle       = calculate_angle(r_shoulder, r_elbow, r_wrist)
                knee_angle        = calculate_angle(r_hip, r_knee, r_ankle)
                wrist_angle       = calculate_angle(r_elbow, r_wrist, r_index)
                elbow_alignment   = abs(lm[elbow_id].x - lm[wrist_id].x)
                hip_shoulder_diff = abs(lm[shoulder_id].x - lm[hip_id].x)

                cv2.putText(frame, "*** RELEASE FRAME ***", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Elbow: {elbow_angle} deg", tuple(r_elbow),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Knee: {knee_angle} deg", tuple(r_knee),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Wrist: {wrist_angle} deg", tuple(r_wrist),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                result_frame = frame.copy()

        if abs(frame_count - release_frame) <= 20:
            out.write(frame)
            out.write(frame)
            out.write(frame)
        else:
            out.write(frame)

        progress_bar.progress(
            40 + int(frame_count / total * 60),
            text="Pass 2: Analyzing and saving video..."
        )

    cap.release()
    out.release()
    progress_bar.progress(100, text="Done!")

    return result_frame, release_frame, elbow_angle, knee_angle, wrist_angle, elbow_alignment, hip_shoulder_diff, output_path, shooting_hand, wrist_y_per_frame