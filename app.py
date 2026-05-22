import streamlit as st
import cv2
import os
import tempfile
from core.analyzer import analyze_video
from core.feedback import generate_feedback
from core.graph import plot_wrist_graph

st.set_page_config(
    page_title="Basketball Shot Analyzer",
    page_icon="🏀",
    layout="centered"
)

st.title("🏀 Basketball Shot Analyzer")
st.write("Upload a side-view basketball shooting video and get instant feedback on your form!")

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