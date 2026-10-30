"""Prepare YOLO format training dataset from ball_markup.json.

Usage::

    python scripts/prepare_yolo_dataset.py [MARKUP_PATH] [--output-dir DIR] [--val-ratio 0.2]

Example::

    python scripts/prepare_yolo_dataset.py data/raw/ball_markup.json --output-dir data/yolo_dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow importing ttcv without installing in editable mode
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ttcv.tracking.dataset_generator import convert_markup_to_yolo_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ball_markup.json to YOLO format.")
    parser.add_argument(
        "markup",
        nargs="?",
        default="data/raw/ball_markup.json",
        help="Path to ball_markup.json (default: data/raw/ball_markup.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/yolo_dataset",
        help="Directory to store YOLO dataset (default: data/yolo_dataset)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2 = 20%%)",
    )
    parser.add_argument(
        "--bbox-size",
        type=int,
        default=24,
        help="Bounding box size in pixels (default: 24 px)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 65)
    print(" PingPongVision — YOLO Dataset Exporter")
    print("=" * 65)
    print(f" Source markup: {args.markup}")
    print(f" Target dir   : {args.output_dir}")
    print(f" Val ratio    : {args.val_ratio:.1%}")

    res = convert_markup_to_yolo_format(
        ball_markup_path=args.markup,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        bbox_size_px=args.bbox_size,
    )

    print("\nDataset preparation successful!")
    print(f"  • Total frames converted : {res['total_frames']:,}")
    print(f"  • Train labels           : {res['train_frames']:,}")
    print(f"  • Val labels             : {res['val_frames']:,}")
    print(f"  • Visible balls          : {res['visible_balls']:,}")
    print(f"  • Config saved to        : {res['dataset_yaml']}")
    print("\nTo train YOLOv8 on this dataset:")
    print(f"  yolo detect train data={res['dataset_yaml']} model=yolov8n.pt epochs=50 imgsz=1280")
    print("=" * 65)


if __name__ == "__main__":
    main()
