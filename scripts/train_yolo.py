"""Train YOLOv8 on table tennis ball dataset.

Usage::

    python scripts/train_yolo.py [--data data/processed/yolo_dataset/dataset.yaml] [--epochs 50] [--imgsz 1280]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on table tennis ball dataset.")
    parser.add_argument(
        "--data",
        default="data/processed/yolo_dataset/dataset.yaml",
        help="Path to dataset.yaml (default: data/processed/yolo_dataset/dataset.yaml)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base pretrained model (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Image size for training (1280 recommended for small balls)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to train on ('0', 'cpu', or None for auto)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)

    print("=" * 65)
    print(" PingPongVision — YOLOv8 Ball Detector Training")
    print("=" * 65)
    print(f" Dataset config : {data_path}")
    print(f" Base model     : {args.model}")
    print(f" Epochs         : {args.epochs}")
    print(f" Image size     : {args.imgsz}")
    print(f" Batch size     : {args.batch}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[NOTE] 'ultralytics' is not installed in the current environment.")
        print("To install and run training:")
        print("    pip install ultralytics")
        print("\nThen run:")
        print(f"    yolo detect train data={data_path.as_posix()} model={args.model} epochs={args.epochs} imgsz={args.imgsz} batch={args.batch}")
        print("=" * 65)
        sys.exit(0)

    if not data_path.exists():
        print(f"\n[ERROR] Dataset config '{data_path}' not found.")
        print("Run first: python scripts/prepare_yolo_dataset.py")
        sys.exit(1)

    model = YOLO(args.model)
    print("\nStarting training...")
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name="tt_ball_detector",
        save=True,
    )
    print("\nTraining complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
