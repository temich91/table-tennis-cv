"""Утилиты подготовки датасета для обучения YOLO и генерации синтетических розыгрышей.

Функции:
  - convert_markup_to_yolo_format: Преобразует ball_markup.json в файлы разметки формата YOLO.
  - generate_synthetic_rally: Генерирует реалистичные синтетические видеокадры настольного тенниса
    с баллистической физикой полета, размытием в движении (motion blur) и эталонной разметкой.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def convert_markup_to_yolo_format(
    ball_markup_path: str | Path,
    output_dir: str | Path,
    val_ratio: float = 0.2,
    bbox_size_px: int = 24,
    img_size: tuple[int, int] = (1920, 1080),
) -> dict:
    """Конвертация ball_markup.json в стандартные файлы аннотаций YOLO.

    Для каждого размеченного кадра:
      - Видимый мяч: записывает строку "0 x_center y_center width height" (нормализованные [0, 1]).
      - Невидимый мяч: создает пустой файл разметки (отрицательный фоновый образец).

    Аргументы:
        ball_markup_path: Путь к ball_markup.json.
        output_dir: Директория, где будут созданы папки labels/ и файл dataset.yaml.
        val_ratio: Доля кадров, выделяемая под валидацию (например, 0.2 = 20%).
        bbox_size_px: Длина стороны квадратной ограничивающей рамки в пикселях вокруг центра мяча.
        img_size: (ширина, высота) в пикселях.

    Возвращает:
        Словарь со статистикой количества сгенерированных меток train и val.
    """
    markup_path = Path(ball_markup_path)
    out_path = Path(output_dir)

    raw: dict[str, dict[str, int]] = json.loads(markup_path.read_text(encoding="utf-8"))
    fids = sorted(int(k) for k in raw.keys())

    train_labels_dir = out_path / "labels" / "train"
    val_labels_dir = out_path / "labels" / "val"
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    w_img, h_img = img_size
    norm_w = bbox_size_px / float(w_img)
    norm_h = bbox_size_px / float(h_img)

    n_total = len(fids)
    n_val = int(n_total * val_ratio)
    val_indices = set(np.linspace(0, n_total - 1, n_val, dtype=int))

    train_count = 0
    val_count = 0
    visible_count = 0

    for idx, fid in enumerate(fids):
        pos = raw[str(fid)]
        x, y = pos["x"], pos["y"]

        is_val = idx in val_indices
        target_dir = val_labels_dir if is_val else train_labels_dir
        label_file = target_dir / f"frame_{fid:06d}.txt"

        if x >= 0 and y >= 0:
            xc = x / float(w_img)
            yc = y / float(h_img)
            line = f"0 {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}\n"
            label_file.write_text(line, encoding="utf-8")
            visible_count += 1
        else:
            # Пустой файл = фоновое изображение без мяча
            label_file.write_text("", encoding="utf-8")

        if is_val:
            val_count += 1
        else:
            train_count += 1

    # Генерация dataset.yaml
    yaml_content = f"""# Датасет мячей для настольного тенниса Ultralytics YOLOv8
path: {out_path.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: ball
"""
    yaml_path = out_path / "dataset.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    return {
        "dataset_yaml": str(yaml_path),
        "total_frames": n_total,
        "train_frames": train_count,
        "val_frames": val_count,
        "visible_balls": visible_count,
    }


def generate_synthetic_rally(
    output_video_path: str | Path | None = None,
    output_frames_dir: str | Path | None = None,
    num_frames: int = 150,
    img_size: tuple[int, int] = (1280, 720),
    fps: float = 60.0,
) -> dict:
    """Генерация реалистичного синтетического видеоклипа розыгрыша в настольном теннисе.

    Симулирует:
      - Стол для настольного тенниса в перспективе сбоку с белой окантовкой и сеткой.
      - Параболическую траекторию полета мяча с учетом гравитации.
      - Отскоки на обеих половинах игроков.
      - Размытие в движении (motion blur) вдоль вектора скорости.

    Аргументы:
        output_video_path: Опциональный путь для сохранения видео в MP4.
        output_frames_dir: Опциональный путь для сохранения отдельных кадров PNG/JPEG.
        num_frames: Количество кадров для синтеза.
        img_size: (ширина, высота) видео.
        fps: Частота кадров в секунду.

    Возвращает:
        Словарь с эталонными положениями мяча и путями к файлам.
    """
    w, h = img_size
    scale_x = w / 1920.0
    scale_y = h / 1080.0

    # Углы стола в синтетическом разрешении (соответствуют перспективе TTNet)
    c_tl = (int(528 * scale_x), int(608 * scale_y))
    c_tr = (int(1260 * scale_x), int(608 * scale_y))
    c_br = (int(1416 * scale_x), int(776 * scale_y))
    c_bl = (int(396 * scale_x), int(785 * scale_y))
    table_poly = np.array([c_tl, c_tr, c_br, c_bl], dtype=np.int32)

    net_top = (int(906 * scale_x), int(580 * scale_y))
    net_bot = (int(906 * scale_x), int(780 * scale_y))
    table_surface_y = int(680 * scale_y)

    writer = None
    if output_video_path is not None:
        out_p = Path(output_video_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_p), fourcc, fps, (w, h))

    if output_frames_dir is not None:
        Path(output_frames_dir).mkdir(parents=True, exist_ok=True)

    # Параметры физической траектории мяча
    # Мяч перелетает туда и обратно через сетку
    ball_markup: dict[str, dict[str, int]] = {}

    # Начальное состояние розыгрыша
    ball_x = float(c_bl[0] + 50)
    ball_y = float(table_surface_y - 120)
    vx = 7.5 * (w / 1280.0)
    vy = -4.0 * (h / 720.0)
    gravity = 0.35 * (h / 720.0)

    for fid in range(num_frames):
        # Обновление физики
        ball_x += vx
        ball_y += vy
        vy += gravity

        # Отскок от поверхности стола
        if ball_y >= table_surface_y:
            ball_y = float(table_surface_y)
            vy = -abs(vy) * 0.88  # Коэффициент восстановления ~0.88
            # При ударе возле правого края возвращаем влево
            if ball_x > c_tr[0]:
                vx = -abs(vx)
            elif ball_x < c_tl[0]:
                vx = abs(vx)

        # Разворот при вылете за границы площадки
        if ball_x > w - 100:
            vx = -abs(vx)
        elif ball_x < 100:
            vx = abs(vx)

        # ── Отрисовка кадра ──────────────────────────────────────────────────
        frame = np.full((h, w, 3), (45, 50, 55), dtype=np.uint8)

        # Пол зала
        cv2.rectangle(frame, (0, int(h * 0.55)), (w, h), (35, 38, 42), -1)

        # Поверхность стола
        cv2.fillPoly(frame, [table_poly], (115, 65, 25))
        # Белая линия периметра
        cv2.polylines(frame, [table_poly], isClosed=True, color=(240, 240, 240), thickness=2)
        # Сетка
        cv2.line(frame, net_top, net_bot, (230, 230, 230), 2)

        # Отрисовка мяча со шлейфом размытия в движении
        speed = np.hypot(vx, vy)
        bx, by = int(ball_x), int(ball_y)
        blur_len = int(min(20, speed * 1.5))
        if blur_len > 2:
            blur_start = (int(bx - vx * 0.8), int(by - vy * 0.8))
            cv2.line(frame, blur_start, (bx, by), (220, 220, 220), 4)

        # Четкий круг мяча
        cv2.circle(frame, (bx, by), 5, (255, 255, 255), -1)

        # Запись эталонных данных
        ball_markup[str(fid)] = {"x": bx, "y": by}

        if writer is not None:
            writer.write(frame)

        if output_frames_dir is not None:
            frame_path = Path(output_frames_dir) / f"frame_{fid:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)

    if writer is not None:
        writer.release()

    return {
        "video_path": str(output_video_path) if output_video_path else None,
        "frames_count": num_frames,
        "ball_markup": ball_markup,
    }
