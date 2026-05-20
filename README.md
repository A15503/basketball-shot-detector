# 🏀 Basketball Shot Analyzer

A computer vision tool that analyzes basketball shooting form from video.

## What it does
- Detects body pose using MediaPipe
- Automatically finds the release frame
- Analyzes 5 key form checkpoints
- Gives feedback and a downloadable report

## How to run
1. Download the MediaPipe model:
   pose_landmarker_full.task from Google MediaPipe
2. Install dependencies:
   pip install -r requirements.txt
3. Run the app:
   streamlit run app.py

## Tech stack
- MediaPipe — pose detection
- OpenCV — video processing
- NumPy — angle math
- Streamlit — web interface