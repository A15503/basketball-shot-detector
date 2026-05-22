import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
import io
from PIL import Image

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Basketball Shot Analyzer",
    page_icon="🏀",
    layout="centered"
)

st.title("🏀 Basketball Shot Analyzer")
st.write("Upload a side-view basketball shooting video and get instant feedback on your form!")

# ─── Angle Function ───────────────────────────────────────────
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return round(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))), 1)

# ─── Detect Shooting Hand ─────────────────────────────────────
def detect_shooting_hand(lm):
    right_wrist_y = lm[16].y
    left_wrist_y  = lm[15].y
    if right_wrist_y < left_wrist_y:
        return "right"
    else:
        return "left"

# ─── Get Landmark IDs ─────────────────────────────────────────
def get_landmark_ids(shooting_hand):
    if shooting_hand == "right":
        return 12, 14, 16, 20, 24, 26, 28
    else:
        return 11, 13, 15, 19, 23, 25, 27

# ─── Feedback Function ────────────────────────────────────────
def generate_feedback(elbow_angle, knee_angle, wrist_angle, elbow_alignment, hip_shoulder_diff):
    feedback = []
    score = 0

    if 80 <= elbow_angle <= 110:
        feedback.append(("✅", "Elbow Angle", f"{elbow_angle}° — Good L-shape at release!"))
        score += 1
    elif elbow_angle < 80:
        feedback.append(("⚠️", "Elbow Angle", f"{elbow_angle}° — Too bent, aim for 80-110°"))
    else:
        feedback.append(("⚠️", "Elbow Angle", f"{elbow_angle}° — Too straight, keep elbow bent"))

    if 100 <= knee_angle <= 140:
        feedback.append(("✅", "Knee Bend", f"{knee_angle}° — Good power from legs!"))
        score += 1
    elif knee_angle > 140:
        feedback.append(("⚠️", "Knee Bend", f"{knee_angle}° — Bend knees more for power"))
    else:
        feedback.append(("⚠️", "Knee Bend", f"{knee_angle}° — Knees too bent at release"))

    if wrist_angle < 40:
        feedback.append(("✅", "Wrist Follow Through", f"{wrist_angle}° — Great snap!"))
        score += 1
    else:
        feedback.append(("⚠️", "Wrist Follow Through", f"{wrist_angle}° — Snap wrist down more (goose neck)"))

    if elbow_alignment < 0.05:
        feedback.append(("✅", "Elbow Alignment", f"{elbow_alignment:.3f} — Elbow nicely under the ball!"))
        score += 1
    else:
        feedback.append(("⚠️", "Elbow Alignment", f"{elbow_alignment:.3f} — Elbow flaring out, tuck it in"))

    if hip_shoulder_diff < 0.1:
        feedback.append(("✅", "Body Alignment", f"{hip_shoulder_diff:.3f} — Good straight form!"))
        score += 1
    else:
        feedback.append(("⚠️", "Body Alignment", f"{hip_shoulder_diff:.3f} — Body twisted, square up"))

    return feedback, score

# ─── Wrist Graph Function ─────────────────────────────────────
def plot_wrist_graph(wrist_y_per_frame, release_frame):
    frames = [f[0] for f in wrist_y_per_frame]
    y_vals = [f[1] for f in wrist_y_per_frame]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    ax.plot(frames, y_vals, color='#00ff88', linewidth=2, label='Wrist Height')

    release_y = next(y for f, y in wrist_y_per_frame if f == release_frame)
    ax.axvline(x=release_frame, color='red', linestyle='--', linewidth=2, label=f'Release Frame {release_frame}')
    ax.plot(release_frame, release_y, 'ro', markersize=10)

    ax.set_xlabel('Frame', color='white')
    ax.set_ylabel('Wrist Y Position', color='white')
    ax.set_title('Wrist Arc During Shot', color='white', fontsize=14)
    ax.tick_params(colors='white')
    ax.invert_yaxis()
    ax.legend(facecolor='#262730', labelcolor='white')

    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close()
    return Image.open(buf)

# ─── Analysis Function ────────────────────────────────────────
def analyze_video(video_path, progress_bar):
    base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )

    # Pass 1 — Find release frame
    detector = vision.PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    wrist_y_per_frame = []
    frame_count = 0

    progress_bar.progress(0, text="Pass 1: Scanning video...")

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
            shooting_hand = detect_shooting_hand(lm)
            wrist_id = 16 if shooting_hand == "right" else 15
            wrist_y_per_frame.append((frame_count, lm[wrist_id].y))
        else:
            wrist_y_per_frame.append((frame_count, 1.0))

        progress_bar.progress(int(frame_count / total_frames * 40), text="Pass 1: Scanning video...")

    cap.release()

    # Find release frame
    biggest_drop = 0
    release_frame = 0
    for i in range(1, len(wrist_y_per_frame)):
        drop = wrist_y_per_frame[i-1][1] - wrist_y_per_frame[i][1]
        if drop > biggest_drop:
            biggest_drop  = drop
            release_frame = wrist_y_per_frame[i][0]

    # Pass 2 — Analyze + save annotated video
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

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((frame_count / fps) * 1000)
        results = detector.detect_for_video(mp_image, timestamp_ms)

        if results.pose_landmarks:
            lm = results.pose_landmarks[0]
            shooting_hand = detect_shooting_hand(lm)
            shoulder_id, elbow_id, wrist_id, index_id, hip_id, knee_id, ankle_id = get_landmark_ids(shooting_hand)

            def get_point(lm_point):
                return [int(lm_point.x * w), int(lm_point.y * h)]

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

        # Slow motion near release, normal speed elsewhere
        if abs(frame_count - release_frame) <= 20:
            out.write(frame)
            out.write(frame)
            out.write(frame)
        else:
            out.write(frame)

        progress_bar.progress(
            40 + int(frame_count / total_frames * 60),
            text="Pass 2: Analyzing and saving video..."
        )

    cap.release()
    out.release()
    progress_bar.progress(100, text="Done!")

    return result_frame, release_frame, elbow_angle, knee_angle, wrist_angle, elbow_alignment, hip_shoulder_diff, output_path, shooting_hand, wrist_y_per_frame

# ─── UI ───────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your shooting video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.video(uploaded_file)

    if st.button("🔍 Analyze My Shot"):
        progress_bar = st.progress(0, text="Starting analysis...")

        result_frame, release_frame, elbow_angle, knee_angle, wrist_angle, elbow_alignment, hip_shoulder_diff, output_path, shooting_hand, wrist_y_per_frame = analyze_video(tmp_path, progress_bar)

        if result_frame is not None:
            feedback, score = generate_feedback(
                elbow_angle, knee_angle, wrist_angle,
                elbow_alignment, hip_shoulder_diff
            )

            st.markdown("---")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Score", f"{score}/5")
            with col2:
                st.metric("Release Frame", release_frame)
            with col3:
                st.metric("Shooting Hand", shooting_hand.capitalize())

            st.markdown("### Release Frame")
            frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, use_container_width=True)

            st.markdown("### Wrist Arc Graph")
            graph_img = plot_wrist_graph(wrist_y_per_frame, release_frame)
            st.image(graph_img, use_container_width=True)

            st.markdown("### Annotated Video")
            with open(output_path, "rb") as f:
                st.download_button(
                    "📥 Download Annotated Video",
                    f,
                    file_name="shot_annotated.mp4",
                    mime="video/mp4"
                )

            st.markdown("### Detailed Feedback")
            for icon, category, message in feedback:
                if icon == "✅":
                    st.success(f"**{category}:** {message}")
                else:
                    st.warning(f"**{category}:** {message}")

            report = f"BASKETBALL SHOT ANALYSIS REPORT\n{'='*40}\n"
            report += f"Shooting Hand: {shooting_hand.capitalize()}\n"
            report += f"Release Frame: {release_frame}\n"
            report += f"Overall Score: {score}/5\n\n"
            for icon, category, message in feedback:
                report += f"{icon} {category}: {message}\n"

            st.download_button("📥 Download Text Report", report, "shot_report.txt")

        else:
            st.error("Could not detect a pose. Try a clearer side-view video.")

        os.unlink(tmp_path)