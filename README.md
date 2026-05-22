# 🏀 Basketball Shot Analyzer

A computer vision web app that analyzes basketball shooting form from video using pose detection and angle mathematics.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.35-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-red)

---

## 📸 What it does

Upload a side-view basketball shooting video and get:
- Automatic release frame detection
- 5-point shooting form score
- Joint angle analysis with visual skeleton overlay
- Wrist arc graph showing your full shot trajectory
- Slow motion annotated video download
- Downloadable feedback report

---

## 🧠 How it works

### Step 1 — Pose Detection
Uses Google's **MediaPipe Pose Landmarker** to detect 33 body landmarks (joints) on every frame. Each landmark has normalized x, y coordinates between 0 and 1.

```
0 = top of screen
1 = bottom of screen

So wrist going UP = Y value getting SMALLER
```

Key landmarks used:

| ID | Joint |
|---|---|
| 12 / 11 | Right / Left shoulder |
| 14 / 13 | Right / Left elbow |
| 16 / 15 | Right / Left wrist |
| 20 / 19 | Right / Left index finger |
| 24 / 23 | Right / Left hip |
| 26 / 25 | Right / Left knee |
| 28 / 27 | Right / Left ankle |

---

### Step 2 — Shooting Hand Detection
At the release frame, the shooting hand wrist is higher than the guide hand. Since higher = smaller Y value:

```python
if right_wrist_y < left_wrist_y:
    shooting_hand = "right"
else:
    shooting_hand = "left"
```

---

### Step 3 — Release Frame Detection
Tracks the shooting wrist Y position across all frames. First applies **rolling average smoothing** to reduce MediaPipe noise:

```python
avg_y = (wrist_y_per_frame[i-2][1] +
         wrist_y_per_frame[i-1][1] +
         wrist_y_per_frame[i][1]) / 3
```

Then finds the frame with the biggest upward velocity (biggest drop in Y):

```
Wrist Y over time:

high |        *
     |      *   *   ← release detected here
     |    *
low  |  *
     |________________
          frames
```

---

### Step 4 — Angle Mathematics
At the release frame, calculates joint angles using **vector dot product**:

```python
def calculate_angle(a, b, c):
    ba = a - b  # vector from B to A
    bc = c - b  # vector from B to C
    cosine = dot(ba, bc) / (norm(ba) * norm(bc))
    return degrees(arccos(cosine))
```

The angle is always measured at the **middle point B**. For example:
- Elbow angle → `calculate_angle(shoulder, elbow, wrist)`
- Knee angle → `calculate_angle(hip, knee, ankle)`
- Wrist snap → `calculate_angle(elbow, wrist, index_finger)`

---

### Step 5 — Feedback Engine
Compares calculated angles against good form benchmarks:

| Checkpoint | Landmarks | Good Range |
|---|---|---|
| Elbow angle | shoulder→elbow→wrist | 80–110° |
| Knee bend | hip→knee→ankle | 100–140° |
| Wrist follow through | elbow→wrist→index | < 40° |
| Elbow alignment | elbow X vs wrist X | < 0.05 |
| Body alignment | shoulder X vs hip X | < 0.1 |

---

## 🚀 How to run

### 1. Clone the repo
```bash
git clone https://github.com/A15503/basketball-shot-detector.git
cd basketball-shot-detector
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the MediaPipe model
Download `pose_landmarker_full.task` from [Google MediaPipe](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) and place it inside the `models/` folder.

### 5. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project structure

```
basketball_shot_detector/
├── core/
│   ├── __init__.py
│   ├── analyzer.py      # video processing + MediaPipe logic
│   ├── feedback.py      # form rules and scoring
│   └── graph.py         # wrist arc visualization
├── utils/
│   ├── __init__.py
│   └── angles.py        # vector angle math
├── models/
│   └── pose_landmarker_full.task
├── app.py               # Streamlit UI
└── requirements.txt
```

---

## 🛠️ Tech stack

| Tool | Purpose |
|---|---|
| [MediaPipe](https://mediapipe.dev) | Pose landmark detection |
| [OpenCV](https://opencv.org) | Video frame processing |
| [NumPy](https://numpy.org) | Vector math and angle calculation |
| [Streamlit](https://streamlit.io) | Web interface |
| [Matplotlib](https://matplotlib.org) | Wrist arc visualization |

---

## 📊 Features

- ✅ Automatic release frame detection
- ✅ Left and right hand support
- ✅ Wrist arc graph
- ✅ Slow motion around release in annotated video
- ✅ 5-point form score with detailed feedback
- ✅ Downloadable annotated video and text report

---

## 🔮 Future improvements

- [ ] Jump shot vs free throw threshold modes
- [ ] Multiple shot detection per video
- [ ] Side-by-side comparison with pro player
- [ ] Webcam recording support
- [ ] Mobile app version

---

## 📚 What you will learn building this

- Computer vision basics with OpenCV
- Pose detection with MediaPipe
- Vector math for angle calculation
- Video frame processing
- Building web apps with Streamlit
- Python project structure and modules