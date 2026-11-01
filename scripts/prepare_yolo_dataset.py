"""Подготовка обучающего датасета в формате YOLO из файла ball_markup.json.

Использование::

    python scripts/prepare_yolo_dataset.py [MARKUP_PATH] [--output-dir DIR] [--val-ratio 0.2]

Пример::

    python scripts/prepare_yolo_dataset.py data/raw/ball_markup.json --output-dir data/yolo_dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Разрешаем импорт ttcv без предварительной установки в режиме editable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ttcv.tracking.dataset_generator import convert_markup_to_yolo_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Конвертация ball_markup.json в формат YOLO.")
    parser.add_argument(
        "markup",
        nargs="?",
        default="data/raw/ball_markup.json",
        help="Путь к ball_markup.json (по умолчанию: data/raw/ball_markup.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/yolo_dataset",
        help="Директория для сохранения датасета YOLO (по умолчанию: data/yolo_dataset)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Доля выборки валидации (по умолчанию: 0.2 = 20%%)",
    )
    parser.add_argument(
        "--bbox-size",
        type=int,
        default=24,
        help="Размер ограничивающей рамки в пикселях (по умолчанию: 24 px)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 65)
    print(" PingPongVision — Экспортер датасета YOLO")
    print("=" * 65)
    print(f" Исходная разметка: {args.markup}")
    print(f" Целевая папка    : {args.output_dir}")
    print(f" Доля валидации   : {args.val_ratio:.1%}")

    res = convert_markup_to_yolo_format(
        ball_markup_path=args.markup,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        bbox_size_px=args.bbox_size,
    )

    print("\nПодготовка датасета успешно завершена!")
    print(f"  • Всего сконвертировано кадров : {res['total_frames']:,}")
    print(f"  • Обучающих меток (train)      : {res['train_frames']:,}")
    print(f"  • Валидационных меток (val)    : {res['val_frames']:,}")
    print(f"  • Видимых мячей                : {res['visible_balls']:,}")
    print(f"  • Конфигурация сохранена в     : {res['dataset_yaml']}")
    print("\nДля запуска обучения YOLOv8 на этом датасете:")
    print(f"  yolo detect train data={res['dataset_yaml']} model=yolov8n.pt epochs=50 imgsz=1280")
    print("=" * 65)


if __name__ == "__main__":
    main()
