# PingPongVision 🏓

**Ball Tracking & Table Bounce Analytics** — Full Computer Vision Pipeline for Table Tennis.

## Project Architecture

```
[Video Stream / Dataset]
   │
   ├──> 1. Data Pipeline & EDA          ← Step 1 (Completed)
   │       (TTNet dataset loaders, trajectory & visibility EDA)
   │
   ├──> 2. Table Detection & Homography ← Step 2 (Completed)
   │       (ITTF dimensions, auto-corner extraction, Bird's Eye View)
   │
   ├──> 3. Ball Detection & Tracking    ← Step 3 (Completed)
   │       (Kalman Filter, motion blur filling, speed in km/h, YOLO/CV)
   │
   └──> 4. Analytics & Web Dashboard    ← Step 4 (Completed)
           (Interactive Streamlit UI, 2D Heatmaps, Match Reports)
```

## Dataset

**Source:** [lab.osai.ai](https://lab.osai.ai/) — TTNet academic dataset.

Each game clip contains:
| File | Description |
|------|-------------|
| `ball_markup.json` | `{frame_id: {x, y}}` — ball center in pixels; `(-1,-1)` = not visible |
| `events_markup.json` | `{frame_id: event}` — `"bounce"`, `"net"`, `"empty_event"` |
| `segmentation_masks/{frame_id}.png` | RGB 320×128 semantic segmentation (Red = table surface) |

Ball coordinates are relative to **1920×1080** full-HD frames.  
Segmentation masks are downscaled to **320×128**.

## Setup & Installation

```bash
# 1. Clone & create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS

# 2. Install project with development dependencies
pip install -e ".[dev]"

# Optional: Install Ultralytics for custom YOLO training
pip install -e ".[models]"
```

## Project Structure

```
table-tennis-cv/
├── app.py                    # Streamlit Dashboard launcher
├── data/
│   ├── raw/                  # Raw dataset files (from lab.osai.ai)
│   │   ├── ball_markup.json
│   │   ├── events_markup.json
│   │   └── segmentation_masks/
│   └── processed/            # Intermediate artefacts / YOLO dataset
├── src/
│   └── ttcv/
│       ├── __init__.py
│       ├── app.py            # Main Streamlit web application
│       ├── analytics/        # Step 4: Game metrics & match reports
│       │   ├── __init__.py
│       │   ├── metrics.py    # RallyAnalyzer, RallyMetrics, ShotInfo
│       │   └── report.py     # Markdown & JSON match report generators
│       ├── data/             # Step 1: Data loaders & EDA
│       │   ├── __init__.py
│       │   ├── dataset.py    # TTNetDataset — core data loader
│       │   └── eda.py        # Exploratory analysis helpers
│       ├── geometry/         # Step 2: ITTF standards & homography
│       │   ├── __init__.py
│       │   ├── table.py      # ITTF dimensions, zones, and coordinates
│       │   ├── homography.py # Table corner detection & TableHomography engine
│       │   └── visualizer.py # 2D BEV table drawing & bounce heatmaps
│       └── tracking/         # Step 3: Ball tracking & detectors
│           ├── __init__.py
│           ├── kalman.py     # 6-state constant acceleration Kalman filter
│           ├── tracker.py    # BallTracker (interpolation, speed, comet tail)
│           ├── detector.py   # YOLOv8, ColorMotion, and Mock detectors
│           └── dataset_generator.py # YOLO converter & synthetic rally generator
├── scripts/
│   ├── run_eda.py            # Run EDA on dataset and save plots
│   ├── run_homography.py     # Run table detection, BEV projection & heatmap
│   ├── infer.py              # Run live tracking & 2D mini-map inference
│   ├── prepare_yolo_dataset.py # Convert ball_markup.json to YOLO format
│   └── train_yolo.py         # Launch YOLOv8 training with Ultralytics
├── tests/
│   ├── test_dataset.py       # 25 unit tests for data structures
│   ├── test_eda.py           # 6 smoke tests for EDA visualizations
│   ├── test_geometry.py      # 29 tests for table geometry & homography
│   ├── test_tracking.py      # 16 tests for Kalman filter, tracker & detectors
│   └── test_analytics.py     # 6 tests for rally analytics & match reporting
├── outputs/                  # Generated plots, videos and reports (gitignored)
├── pyproject.toml
└── README.md
```

## Quick Start & Usage

### 1. Interactive Streamlit Dashboard (Step 4)
Launch the interactive web application in your browser:
```bash
streamlit run app.py
```
**Features:**
- Interactive court analytics with **Bird's Eye View bounce density heatmaps**.
- Instant toggle between **Real TTNet match data**, **Synthetic rally simulation**, and **Custom video upload**.
- Speed velocity profile (km/h) and vertical trajectory curves.
- Video tracking with **comet tail** and live **2D mini-map** overlay in the corner.
- Automated executive match report export (Markdown & JSON).

---

### 2. Live Video Tracking & Inference (Step 3)
```bash
# A. Synthetic rally simulation (generates video with live BEV mini-map and comet tail):
python scripts/infer.py --synthetic --max-frames 120 --output outputs/tracking/synthetic_tracked.mp4

# B. Replay real dataset ground truth with Kalman tracking & speed calculation:
python scripts/infer.py --replay-markup data/raw/ball_markup.json --start-frame 1298 --max-frames 120 --output outputs/tracking/real_markup_tracked.mp4

# C. Custom video with trained YOLOv8 weights:
python scripts/infer.py --video input.mp4 --weights best.pt --mask data/raw/segmentation_masks/1278.png --output outputs/tracking/match_tracked.mp4
```

---

### 3. Table Detection & Bird's Eye View Homography (Step 2)
```bash
python scripts/run_homography.py --mask data/raw/segmentation_masks/1278.png --output-dir outputs/homography
```
Extracts corners from segmentation mask, computes homography matrix, and projects all bounce events onto 2D ITTF table coordinates.

---

### 4. Exploratory Data Analysis (Step 1)
```bash
python scripts/run_eda.py data/raw --output-dir outputs/eda
```

---

### 5. Training YOLOv8 Ball Detector
```bash
# 1. Convert dataset to YOLO format:
python scripts/prepare_yolo_dataset.py data/raw/ball_markup.json --output-dir data/processed/yolo_dataset

# 2. Train YOLOv8-nano on ball dataset:
python scripts/train_yolo.py --data data/processed/yolo_dataset/dataset.yaml --epochs 50 --imgsz 1280
```

---

### 6. Running Tests
```bash
python -m pytest tests/ -v
# 82 passed in ~5s
```

## Roadmap

- [x] **Step 1** — Data Pipeline & EDA
- [x] **Step 2** — Table Detection & Homography (BEV)
- [x] **Step 3** — Ball Detection & Tracking (Kalman Filter + YOLO / Classical Detectors)
- [x] **Step 4** — Analytics Dashboard (Streamlit interactive demo)
