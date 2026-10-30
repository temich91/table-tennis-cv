# PingPongVision 🏓

**Ball Tracking & Table Bounce Analytics** — Junior-level Computer Vision project.

## Project Architecture

```
[Video Stream]
   │
   ├──> 1. Data Pipeline & EDA          ← Step 1 (Completed)
   │
   ├──> 2. Table Detection & Homography ← Step 2 (Completed)
   │       (Bird's Eye View transform)
   │
   ├──> 3. Ball Detection & Tracking    ← Step 3 (Completed)
   │       (Kalman Filter + YOLO / Classical Detectors)
   │
   └──> 4. Analytics & Dashboard        ← Step 4 (Next)
           (Bounce Heatmap, Speed, Rally Stats, Streamlit)
```

## Dataset

**Source:** [lab.osai.ai](https://lab.osai.ai/) — TTNet dataset.

Each game clip contains:
| File | Description |
|------|-------------|
| `ball_markup.json` | `{frame_id: {x, y}}` — ball center in pixels; `(-1,-1)` = not visible |
| `events_markup.json` | `{frame_id: event}` — `"bounce"`, `"net"`, `"empty_event"` |
| `segmentation_masks/{frame_id}.png` | RGB 320×128 semantic segmentation (Red = table surface) |

Ball coordinates are relative to **1920×1080** full-HD frames.  
Segmentation masks are downscaled to **320×128**.

## Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS

# Install project + dev dependencies
pip install -e ".[dev]"
```

## Project Structure

```
table-tennis-cv/
├── data/
│   ├── raw/                  # Raw dataset files (from lab.osai.ai)
│   │   ├── ball_markup.json
│   │   ├── events_markup.json
│   │   └── segmentation_masks/
│   └── processed/            # Intermediate artefacts / YOLO dataset
├── src/
│   └── ttcv/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py    # TTNetDataset — core data loader
│       │   └── eda.py        # Exploratory analysis helpers
│       ├── geometry/
│       │   ├── __init__.py
│       │   ├── table.py      # ITTF table dimensions, zones, and coordinates
│       │   ├── homography.py # Table corner detection & TableHomography engine
│       │   └── visualizer.py # 2D BEV table drawing & bounce heatmaps
│       └── tracking/
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
│   └── test_tracking.py      # 16 tests for Kalman filter, tracker & detectors
├── outputs/                  # Generated plots, videos and artefacts (gitignored)
├── pyproject.toml
└── README.md
```

## Quick Start & Usage

### 1. Exploratory Data Analysis (Step 1)
```bash
python scripts/run_eda.py data/raw --output-dir outputs/eda
```

### 2. Table Detection & Bird's Eye View Homography (Step 2)
```bash
python scripts/run_homography.py --mask data/raw/segmentation_masks/1278.png --output-dir outputs/homography
```

### 3. Ball Tracking & Inference (Step 3)
```bash
# A. Run inference on a realistic synthetic rally clip (with live 2D mini-map overlay):
python scripts/infer.py --synthetic --max-frames 120 --output outputs/tracking/synthetic_tracked.mp4

# B. Replay real dataset ground truth coordinates with Kalman tracking and speed estimation:
python scripts/infer.py --replay-markup data/raw/ball_markup.json --start-frame 1298 --max-frames 120 --output outputs/tracking/real_markup_tracked.mp4

# C. Run on custom video file with trained YOLOv8 weights:
python scripts/infer.py --video input_match.mp4 --weights best.pt --mask data/raw/segmentation_masks/1278.png --output outputs/tracking/match_tracked.mp4
```

### 4. Training YOLOv8 Ball Detector
```bash
# 1. Convert dataset to YOLO format (creates data/processed/yolo_dataset/dataset.yaml)
python scripts/prepare_yolo_dataset.py data/raw/ball_markup.json --output-dir data/processed/yolo_dataset

# 2. Launch training (requires: pip install ultralytics)
python scripts/train_yolo.py --data data/processed/yolo_dataset/dataset.yaml --epochs 50 --imgsz 1280
```

### 5. Running the Test Suite
```bash
python -m pytest tests/ -v
# 76 passed in ~4s
```

## Roadmap

- [x] **Step 1** — Data Pipeline & EDA
- [x] **Step 2** — Table Detection & Homography (BEV)
- [x] **Step 3** — Ball Detection & Tracking (Kalman Filter + YOLO / Classical Detectors)
- [ ] **Step 4** — Analytics Dashboard (Streamlit interactive demo)
