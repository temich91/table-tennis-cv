"""Обучение модели YOLOv8 на датасете мячей для настольного тенниса.

Использование::

    python scripts/train_yolo.py [--data data/processed/yolo_dataset/dataset.yaml] [--epochs 50] [--imgsz 1280]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение YOLOv8 на датасете мячей для настольного тенниса.")
    parser.add_argument(
        "--data",
        default="data/processed/yolo_dataset/dataset.yaml",
        help="Путь к dataset.yaml (по умолчанию: data/processed/yolo_dataset/dataset.yaml)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Базовая предобученная модель (по умолчанию: yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Количество эпох обучения (по умолчанию: 50)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Размер изображения для обучения (рекомендуется 1280 для маленьких мячей)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Размер батча (по умолчанию: 8)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Устройство для обучения ('0', 'cpu' или None для автовыбора)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)

    print("=" * 65)
    print(" PingPongVision — Обучение детектора мяча YOLOv8")
    print("=" * 65)
    print(f" Конфигурация датасета : {data_path}")
    print(f" Базовая модель        : {args.model}")
    print(f" Эпох                  : {args.epochs}")
    print(f" Размер изображения    : {args.imgsz}")
    print(f" Размер батча          : {args.batch}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[ПРИМЕЧАНИЕ] 'ultralytics' не установлен в текущем окружении.")
        print("Для установки и запуска обучения выполните:")
        print("    pip install ultralytics")
        print("\nЗатем запустите:")
        print(f"    yolo detect train data={data_path.as_posix()} model={args.model} epochs={args.epochs} imgsz={args.imgsz} batch={args.batch}")
        print("=" * 65)
        sys.exit(0)

    if not data_path.exists():
        print(f"\n[ОШИБКА] Конфигурация датасета '{data_path}' не найдена.")
        print("Сначала выполните: python scripts/prepare_yolo_dataset.py")
        sys.exit(1)

    model = YOLO(args.model)
    print("\nЗапуск обучения...")
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name="tt_ball_detector",
        save=True,
    )
    print("\nОбучение завершено!")
    print("=" * 65)


if __name__ == "__main__":
    main()
