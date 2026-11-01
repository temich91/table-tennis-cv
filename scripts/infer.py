"""Скрипт инференса для Шага 3: детекция мяча, трекинг и наложение мини-карты вида сверху (BEV).

Использование::

    # Запуск на синтетическом розыгрыше (не требует сторонних видеофайлов!):
    python scripts/infer.py --synthetic --output outputs/tracking/synthetic_tracked.mp4

    # Запуск на пользовательском видеофайле:
    python scripts/infer.py --video path/to/game.mp4 --weights best.pt --output outputs/tracking/tracked.mp4

    # Запуск на примере разметки с использованием MockBallDetector:
    python scripts/infer.py --replay-markup data/raw/ball_markup.json --output outputs/tracking/replay.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Разрешаем импорт ttcv без предварительной установки в режиме editable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from ttcv.geometry import TableHomography
from ttcv.geometry.visualizer import draw_table_bev
from ttcv.tracking import (
    BallTracker,
    BaseBallDetector,
    ColorMotionBallDetector,
    MockBallDetector,
    YOLOv8BallDetector,
    generate_synthetic_rally,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Запуск инференса: детекция мяча, трекинг и наложение мини-карты вида сверху (BEV)."
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Путь к входному видеофайлу.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Сгенерировать и обработать реалистичный синтетический видеоклип розыгрыша.",
    )
    parser.add_argument(
        "--replay-markup",
        default=None,
        help="Воспроизвести эталонную разметку ball_markup.json на синтетическом или чистом фоне.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Путь к обученным весам модели YOLOv8 (.pt). Если не указан, используется классический детектор.",
    )
    parser.add_argument(
        "--mask",
        default="data/raw/segmentation_masks/1278.png",
        help="Путь к маске сегментации для калибровки гомографии стола.",
    )
    parser.add_argument(
        "--output",
        default="outputs/tracking/tracked_rally.mp4",
        help="Путь для сохранения результирующего аннотированного видео.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Частота кадров видео (по умолчанию: 60 fps).",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Начальный индекс кадра (удобно для воспроизведения разметки с нужного момента).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Максимальное количество кадров для обработки.",
    )
    return parser.parse_args()


def render_hud_and_minimap(
    frame: np.ndarray,
    tracker: BallTracker,
    current_pt,
    bev_base_img: np.ndarray,
    homography: TableHomography,
    frame_idx: int,
    bounces_count: int,
) -> np.ndarray:
    """Отрисовка кометного шлейфа трекинга, информационного HUD и 2D мини-карты вида сверху на кадре."""
    h_frame, w_frame = frame.shape[:2]
    annotated = frame.copy()

    # 1. Отрисовка кометного шлейфа (затухающая линия траектории)
    tail = tracker.get_recent_tail()
    if len(tail) >= 2:
        for i in range(1, len(tail)):
            alpha = float(i) / len(tail)
            thickness = max(1, int(1 + 3 * alpha))
            # Градиент цвета: желто-оранжевый хвост переходит в яркий циан
            color = (
                int(255 * (1 - alpha)),
                int(200 * alpha + 50),
                int(255 * alpha),
            )
            pt1 = (int(tail[i - 1][0]), int(tail[i - 1][1]))
            pt2 = (int(tail[i][0]), int(tail[i][1]))
            cv2.line(annotated, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

    # 2. Отрисовка текущего маркера мяча
    if current_pt is not None:
        cx, cy = int(current_pt.x), int(current_pt.y)
        marker_color = (0, 165, 255) if current_pt.is_predicted else (0, 255, 255)

        # Внешнее прицельное кольцо
        cv2.circle(annotated, (cx, cy), 12, marker_color, 2, lineType=cv2.LINE_AA)
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        # Метка скорости рядом с мячом
        speed_text = f"{current_pt.speed_kmh:.1f} km/h" if current_pt.speed_kmh else f"{current_pt.speed_px:.1f} px/f"
        cv2.putText(
            annotated,
            speed_text,
            (cx + 16, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 3. Отрисовка 2D мини-карты вида сверху в правом верхнем углу
    minimap_w = int(w_frame * 0.28)
    minimap_h = int(minimap_w * (650.0 / 1100.0))
    bev_resized = cv2.resize(bev_base_img, (minimap_w, minimap_h))

    # Проекция текущего положения мяча на мини-карту
    if current_pt is not None:
        pt_bev = homography.project_to_bev(np.array([current_pt.x, current_pt.y]))
        bx_norm = pt_bev[0] / homography.bev_size[0]
        by_norm = pt_bev[1] / homography.bev_size[1]
        map_bx = int(bx_norm * minimap_w)
        map_by = int(by_norm * minimap_h)

        if 0 <= map_bx < minimap_w and 0 <= map_by < minimap_h:
            cv2.circle(bev_resized, (map_bx, map_by), 5, (0, 0, 255), -1)
            cv2.circle(bev_resized, (map_bx, map_by), 7, (255, 255, 255), 1)

    # Наложение мини-карты в правом верхнем углу с рамкой
    x_offset = w_frame - minimap_w - 20
    y_offset = 20
    annotated[y_offset : y_offset + minimap_h, x_offset : x_offset + minimap_w] = bev_resized
    cv2.rectangle(
        annotated,
        (x_offset - 2, y_offset - 2),
        (x_offset + minimap_w + 2, y_offset + minimap_h + 2),
        (255, 255, 255),
        2,
    )
    cv2.putText(
        annotated,
        "2D Court (BEV)",
        (x_offset + 5, y_offset - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # 4. Отрисовка информационной панели HUD (в левом верхнем углу)
    hud_bg = (20, 25, 30)
    cv2.rectangle(annotated, (15, 15), (280, 110), hud_bg, -1)
    cv2.rectangle(annotated, (15, 15), (280, 110), (100, 110, 120), 1)

    status_str = "INTERPOLATING" if (current_pt and current_pt.is_predicted) else ("TRACKING" if current_pt else "SEARCHING")
    status_col = (0, 165, 255) if (current_pt and current_pt.is_predicted) else ((0, 255, 0) if current_pt else (0, 0, 255))

    cv2.putText(annotated, f"Frame: {frame_idx:04d}", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(annotated, f"Status: {status_str}", (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_col, 2)
    cv2.putText(annotated, f"Bounces: {bounces_count}", (25, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if current_pt and current_pt.speed_kmh:
        cv2.putText(annotated, f"Speed: {current_pt.speed_kmh:.1f} km/h", (25, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return annotated


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" PingPongVision — Шаг 3: Инференс детекции и трекинга мяча")
    print("=" * 65)

    # 1. Инициализация гомографии для метрической проекции и мини-карты
    mask_path = Path(args.mask)
    if mask_path.exists():
        print(f"\n[1/4] Загрузка гомографии стола из: {mask_path}...")
        homography = TableHomography.from_mask(mask_path)
    else:
        print("\n[1/4] Использование канонической гомографии стола...")
        corners = np.array([[528, 607.5], [1260, 607.5], [1416, 776.25], [396, 784.7]], dtype=np.float32)
        homography = TableHomography(corners)

    bev_base_rgb = draw_table_bev()
    bev_base_bgr = cv2.cvtColor(bev_base_rgb, cv2.COLOR_RGB2BGR)

    # 2. Выбор детектора
    detector: BaseBallDetector
    if args.weights is not None:
        print(f"\n[2/4] Инициализация детектора YOLOv8 из: {args.weights}...")
        detector = YOLOv8BallDetector(args.weights)
    elif args.replay_markup is not None:
        print(f"\n[2/4] Инициализация MockBallDetector (воспроизведение) из: {args.replay_markup}...")
        detector = MockBallDetector(args.replay_markup)
    else:
        print("\n[2/4] Инициализация ColorMotionBallDetector (классическое CV)...")
        detector = ColorMotionBallDetector()

    # 3. Источник кадров / видео
    video_source = args.video
    if args.synthetic or (video_source is None and args.replay_markup is None):
        print("\n[3/4] Генерация синтетического видео розыгрыша...")
        synth_out = output_path.parent / "temp_synthetic_input.mp4"
        synth_data = generate_synthetic_rally(
            output_video_path=synth_out,
            num_frames=args.max_frames,
            img_size=(1280, 720),
            fps=args.fps,
        )
        video_source = str(synth_out)
        if args.replay_markup is None and args.weights is None:
            # Воспроизведение синтетического эталона для демонстрации безупречного трекинга
            detector = MockBallDetector(synth_data["ball_markup"])
    elif video_source is None and args.replay_markup is not None:
        # Воспроизведение эталонной разметки на холсте со столом
        print("\n[3/4] Генерация фонового холста для воспроизведения эталона...")
        synth_out = output_path.parent / "temp_replay_canvas.mp4"
        generate_synthetic_rally(
            output_video_path=synth_out,
            num_frames=args.max_frames,
            img_size=(1920, 1080),
            fps=args.fps,
        )
        video_source = str(synth_out)

    # 4. Запуск трекера и запись видео
    print(f"\n[4/4] Обработка кадров из: {video_source}...")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Не удалось открыть источник видео: {video_source}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    tracker = BallTracker(homography=homography, fps=fps)
    frame_idx = args.start_frame
    frames_processed = 0
    bounces_count = 0

    while frames_processed < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Детекция
        detections = detector.detect(frame, frame_id=frame_idx)
        best_det = detections[0] if detections else None
        det_xy = (best_det.x, best_det.y) if best_det else None
        conf = best_det.confidence if best_det else 1.0

        # Обновление трекинга (предсказание + коррекция Калмана)
        track_pt = tracker.update(frame_id=frame_idx, detection_xy=det_xy, confidence=conf)

        # Детекция отскоков
        detected_bounces = tracker.detect_bounces()
        bounces_count = len(detected_bounces)

        # Отрисовка HUD и 2D мини-карты BEV
        annotated_frame = render_hud_and_minimap(
            frame,
            tracker,
            track_pt,
            bev_base_bgr,
            homography,
            frame_idx,
            bounces_count,
        )

        writer.write(annotated_frame)
        frame_idx += 1
        frames_processed += 1

        if frames_processed % 30 == 0:
            stats = tracker.stats
            print(f"      Кадр {frame_idx:04d} | Отслежено: {stats['total_tracked_frames']} | Предсказано: {stats['predicted_frames']} ({stats['interpolation_rate']:.1f}%)")

    cap.release()
    writer.release()

    final_stats = tracker.stats
    print("\n" + "=" * 65)
    print(" Инференс трекинга успешно завершен!")
    print(f" Выходное видео        : {output_path.resolve()}")
    print(f" Всего обработано      : {frame_idx} кадров")
    print(f" Обнаружено на кадрах  : {final_stats['detected_frames']}")
    print(f" Заполненные пропуски  : {final_stats['predicted_frames']} ({final_stats['interpolation_rate']:.1f}%)")
    print(f" Всего отскоков        : {bounces_count}")
    print(f" Макс. скорость мяча   : {final_stats['max_speed_kmh']:.1f} км/ч")
    print(f" Средняя скорость мяча : {final_stats['mean_speed_kmh']:.1f} км/ч")
    print("=" * 65)


if __name__ == "__main__":
    main()
