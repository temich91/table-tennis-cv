"""Скрипт EDA для Шага 1 — запускается из корня проекта.

Использование::

    python scripts/run_eda.py [DATA_ROOT] [--output-dir OUTPUT_DIR]

По умолчанию DATA_ROOT равен ``data/raw``.
Графики сохраняются в OUTPUT_DIR (по умолчанию: ``outputs/eda``).

Пример::

    python scripts/run_eda.py data/raw --output-dir outputs/eda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Разрешаем запуск из корня проекта без установки пакета
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")  # Неинтерактивный бэкенд для фонового выполнения

import matplotlib.pyplot as plt

from ttcv.data.dataset import load_dataset
from ttcv.data.eda import (
    plot_ball_heatmap,
    plot_event_distribution,
    plot_gap_distribution,
    plot_trajectory,
    plot_visibility,
    print_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Запуск EDA по датасету TTNet.")
    parser.add_argument(
        "data_root",
        nargs="?",
        default="data/raw",
        help="Корневая директория датасета (по умолчанию: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/eda",
        help="Директория для сохранения графиков EDA (по умолчанию: outputs/eda)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Загрузка датасета из: {data_root.resolve()}")
    clips = load_dataset(data_root)

    if not clips:
        print(f"[ERROR] В каталоге '{data_root}' не найдено клипов. Проверьте путь.")
        sys.exit(1)

    print(f"Обнаружено клипов: {len(clips)}.\n")

    for i, clip in enumerate(clips):
        tag = f"clip_{i:02d}"
        print_summary(clip)

        # ── Траектория ──────────────────────────────────────────────────────
        fig = plot_trajectory(clip, title=f"Ball Trajectory — {tag}")
        path = output_dir / f"{tag}_trajectory.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Сохранено: {path}")

        # ── Распределение событий ───────────────────────────────────────────
        fig = plot_event_distribution(clip, title=f"Event Distribution — {tag}")
        path = output_dir / f"{tag}_events.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Сохранено: {path}")

        # ── Временная шкала видимости ───────────────────────────────────────
        fig = plot_visibility(clip, title=f"Ball Visibility — {tag}")
        path = output_dir / f"{tag}_visibility.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Сохранено: {path}")

        # ── Тепловая карта положений мяча ───────────────────────────────────
        fig = plot_ball_heatmap(clip, title=f"Ball Heatmap — {tag}")
        path = output_dir / f"{tag}_heatmap.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Сохранено: {path}")

        # ── Распределение длительности пропусков видимости ──────────────────
        fig = plot_gap_distribution(clip, title=f"Invisibility Gaps — {tag}")
        path = output_dir / f"{tag}_gaps.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Сохранено: {path}")

        print()

    print(f"EDA успешно завершен. Все графики сохранены в: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
