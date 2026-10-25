# PingPongVision 🏓

**Ball Tracking & Table Bounce Analytics** — Junior-level Computer Vision project.

## Project Architecture

```
[Video Stream]
   │
   ├──> 1. Data Pipeline & EDA          ← Step 1 (current)
   │
   ├──> 2. Table Detection & Homography ← Step 2
   │       (Bird's Eye View transform)
   │
   ├──> 3. Ball Detection & Tracking    ← Step 3
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
| `segmentation_masks/{frame_id}.png` | RGB 320×128 semantic segmentation (table, players, background) |

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
│       └── data/
│           ├── __init__.py
│           ├── dataset.py    # TTNetDataset — core data loader
│           └── eda.py        # Exploratory analysis helpers
├── scripts/
│   └── run_eda.py            # Run EDA and save visualizations
├── tests/
│   └── test_dataset.py
├── outputs/                  # EDA plots, reports (gitignored)
├── pyproject.toml
└── README.md
```

## Roadmap

- [x] **Step 1** — Data Pipeline & EDA
- [ ] **Step 2** — Table Detection & Homography
- [ ] **Step 3** — Ball Detection & Tracking
- [ ] **Step 4** — Analytics Dashboard
