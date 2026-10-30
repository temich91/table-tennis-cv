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
   ├──> 3. Ball Detection & Tracking    ← Step 3 (Next)
   │       (TrackNet / YOLOv8 + Kalman)
   │
   └──> 4. Analytics & Dashboard        ← Step 4
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
│   └── processed/            # Intermediate artefacts (gitignored)
├── src/
│   └── ttcv/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py    # TTNetDataset — core data loader
│       │   └── eda.py        # Exploratory analysis helpers
│       └── geometry/
│           ├── __init__.py
│           ├── table.py      # ITTF table dimensions, zones, and coordinates
│           ├── homography.py # Table corner detection & TableHomography engine
│           └── visualizer.py # 2D BEV table drawing & bounce heatmaps
├── scripts/
│   ├── run_eda.py            # Run EDA on dataset and save plots
│   └── run_homography.py     # Run table detection, BEV projection & heatmap
├── tests/
│   ├── test_dataset.py       # 25 unit tests for data structures
│   ├── test_eda.py           # 6 smoke tests for EDA visualizations
│   └── test_geometry.py      # 29 tests for table geometry & homography
├── outputs/                  # Generated plots and artefacts (gitignored)
├── pyproject.toml
└── README.md
```

## Quick Start & Usage

### 1. Exploratory Data Analysis (Step 1)
```bash
python scripts/run_eda.py data/raw --output-dir outputs/eda
```
Generates trajectory plots, event distributions, visibility timelines, and gap analyses.

### 2. Table Detection & Bird's Eye View Homography (Step 2)
```bash
python scripts/run_homography.py --mask data/raw/segmentation_masks/1278.png --output-dir outputs/homography
```
- Automatically extracts 4 table corners from segmentation mask.
- Computes perspective transform to canonical ITTF table dimensions (2.74 m × 1.525 m).
- Projects all ball bounces into metric 2D table space.
- Generates 4 visual artefacts:
  * `outputs/homography/bev_table_diagram.png`: clean 2D ITTF table diagram
  * `outputs/homography/bev_bounce_heatmap.png`: 2D density heatmap of bounces with Player A / Player B breakdown
  * `outputs/homography/bev_trajectories.png`: full ball trajectory projected onto table
  * `outputs/homography/camera_table_overlay.png`: camera frame with table grid and corner labels

### 3. Running Tests
```bash
python -m pytest tests/ -v
# 60 passed in ~4s
```

## Step 3 Preview: Ball Detection & Tracking (Training & Inference)

In Step 3, the model detects and tracks the ball across video frames:

```bash
# 1. Prepare YOLO dataset from video frames and ball_markup.json
python scripts/prepare_yolo_dataset.py --video data/raw/game.mp4 --markup data/raw/ball_markup.json

# 2. Train YOLOv8-nano detector
python scripts/train.py --data configs/ball.yaml --epochs 50 --imgsz 1280

# 3. Run Inference + Tracking + BEV projection on custom video
python scripts/infer.py --video input.mp4 --weights best.pt --mask mask.png --output output_tracked.mp4
```

## Roadmap

- [x] **Step 1** — Data Pipeline & EDA
- [x] **Step 2** — Table Detection & Homography (BEV)
- [ ] **Step 3** — Ball Detection & Tracking (YOLOv8 / TrackNet + Kalman Filter)
- [ ] **Step 4** — Analytics Dashboard (Streamlit interactive demo)
