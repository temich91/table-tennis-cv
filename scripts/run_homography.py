"""Скрипт CLI для Шага 2: детекция стола, гомография и аналитика вида сверху (BEV).

Использование::

    python scripts/run_homography.py [--mask MASK_PATH] [--output-dir OUTPUT_DIR]

По умолчанию:
  - Маска: ``data/raw/segmentation_masks/1278.png``
  - Папка вывода: ``outputs/homography``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Разрешаем импорт ttcv без предварительной установки в режиме editable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ttcv.data.dataset import load_game_clip
from ttcv.geometry import (
    TABLE_LENGTH_MM,
    TABLE_WIDTH_MM,
    TableHomography,
    detect_table_corners_from_mask,
)
from ttcv.geometry.visualizer import (
    draw_camera_table_overlay,
    draw_table_bev,
    plot_bev_heatmap,
    plot_bev_trajectories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Запуск анализа детекции стола и гомографии (вид сверху / BEV)."
    )
    parser.add_argument(
        "--mask",
        default="data/raw/segmentation_masks/1278.png",
        help="Путь к маске сегментации (по умолчанию: data/raw/segmentation_masks/1278.png)",
    )
    parser.add_argument(
        "--data-root",
        default="data/raw",
        help="Корневой каталог с ball_markup.json и events_markup.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/homography",
        help="Директория для сохранения графиков (по умолчанию: outputs/homography)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_path = Path(args.mask)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" PingPongVision — Шаг 2: Детекция стола и гомография (BEV)")
    print("=" * 65)

    # ── 1. Детекция углов стола по маске ────────────────────────────────────
    print(f"\n[1/5] Детекция углов стола из: {mask_path}...")
    corners = detect_table_corners_from_mask(mask_path, target_resolution=(1920, 1080))
    print(f"      Размеры маски      : {corners.mask_size}")
    print(f"      Целевой кадр       : {corners.target_size}")
    print(f"      Дальний левый (TL) : ({corners.top_left[0]:.1f}, {corners.top_left[1]:.1f}) px")
    print(f"      Дальний правый(TR) : ({corners.top_right[0]:.1f}, {corners.top_right[1]:.1f}) px")
    print(f"      Ближний правый(BR) : ({corners.bottom_right[0]:.1f}, {corners.bottom_right[1]:.1f}) px")
    print(f"      Ближний левый (BL) : ({corners.bottom_left[0]:.1f}, {corners.bottom_left[1]:.1f}) px")

    # ── 2. Вычисление матрицы гомографии ────────────────────────────────────
    print("\n[2/5] Вычисление матрицы гомографии...")
    homography = TableHomography(corners)
    print("      Гомография (Камера -> мм стола):\n", homography.H_table)

    # ── 3. Загрузка данных мяча и событий ───────────────────────────────────
    print(f"\n[3/5] Загрузка аннотаций из: {data_root}...")
    clip = load_game_clip(data_root)
    print(f"      Загружено всего размеченных кадров: {clip.n_frames:,}.")

    # Извлечение точек отскоков
    bounce_fids = clip.bounce_frames
    bounce_cam: list[list[float]] = []
    for fid in bounce_fids:
        if fid in clip.ball_data and clip.ball_data[fid].visible:
            bp = clip.ball_data[fid]
            bounce_cam.append([float(bp.x), float(bp.y)])

    bounce_cam_arr = np.array(bounce_cam, dtype=np.float32)
    print(f"      Найдено {len(bounce_cam_arr)} событий отскока с видимым мячом.")

    # ── 4. Проекция и анализ отскоков ───────────────────────────────────────
    print("\n[4/5] Проекция отскоков в 2D метрическое пространство стола (вид сверху)...")
    bounces_table = homography.project_to_table(bounce_cam_arr)

    # Анализ разбиения по зонам
    inside_count = 0
    side_counts = {"player_a": 0, "player_b": 0, "net": 0}
    depth_counts = {"deep": 0, "medium": 0, "short": 0, "outside": 0}
    lateral_counts = {"left": 0, "right": 0, "outside": 0}

    for pt in bounces_table:
        tp = homography.classify_camera_point(
            float(homography.project_table_to_camera(pt.reshape(1, 2))[0, 0]),
            float(homography.project_table_to_camera(pt.reshape(1, 2))[0, 1]),
        )
        if tp.is_on_table(margin_mm=50.0):
            inside_count += 1
            side_counts[tp.side.value] += 1
            depth_counts[tp.depth_zone.value] += 1
            lateral_counts[tp.lateral_zone.value] += 1

    total = len(bounces_table)
    print(f"      Отскоков на столе : {inside_count} / {total} ({inside_count / total * 100:.1f}%)")
    print(f"      • Половина Игрока A : {side_counts['player_a']} ({side_counts['player_a'] / inside_count * 100:.1f}%)")
    print(f"      • Половина Игрока B : {side_counts['player_b']} ({side_counts['player_b'] / inside_count * 100:.1f}%)")
    print(f"      • По глубине        : Короткие={depth_counts['short']} | Средние={depth_counts['medium']} | Глубокие={depth_counts['deep']}")
    print(f"      • По ширине         : Левая (дальняя)={lateral_counts['left']} | Правая (ближняя)={lateral_counts['right']}")

    # ── 5. Генерация и сохранение визуализаций ──────────────────────────────
    print("\n[5/5] Генерация визуализаций...")

    # A. 2D-схема стола
    table_img = draw_table_bev()
    table_path = output_dir / "bev_table_diagram.png"
    cv2.imwrite(str(table_path), cv2.cvtColor(table_img, cv2.COLOR_RGB2BGR))
    print(f"      [OK] Сохранено: {table_path}")

    # B. Тепловая карта отскоков (вид сверху / BEV)
    fig_heat = plot_bev_heatmap(homography, bounce_cam_arr)
    heat_path = output_dir / "bev_bounce_heatmap.png"
    fig_heat.savefig(heat_path, dpi=150, bbox_inches="tight")
    plt.close(fig_heat)
    print(f"      [OK] Сохранено: {heat_path}")

    # C. Траектория и отскоки на столе (BEV)
    traj_arr = clip.trajectory_array
    fig_traj = plot_bev_trajectories(homography, traj_arr, bounce_cam_arr)
    traj_path = output_dir / "bev_trajectories.png"
    fig_traj.savefig(traj_path, dpi=150, bbox_inches="tight")
    plt.close(fig_traj)
    print(f"      [OK] Сохранено: {traj_path}")

    # D. Наложение стола на кадр камеры
    mask_bgr = cv2.imread(str(mask_path))
    # Масштабирование маски под целевое разрешение для визуализации наложения
    mask_full = cv2.resize(mask_bgr, (1920, 1080), interpolation=cv2.INTER_NEAREST)
    overlay = draw_camera_table_overlay(mask_full, homography)
    overlay_path = output_dir / "camera_table_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    print(f"      [OK] Сохранено: {overlay_path}")

    print("\n" + "=" * 65)
    print(f" Шаг 2 успешно завершен! Все артефакты сохранены в: {output_dir.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
