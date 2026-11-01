"""BallTracker: Высокоуровневый трекер мяча для настольного тенниса.

Возможности:
  - Интеграция BallKalmanFilter для сглаживания траектории и заполнения пропущенных кадров.
  - Расчет метрической скорости в реальном времени в км/ч с использованием TableHomography и FPS камеры.
  - Ведение очереди последних точек траектории для отрисовки кометного шлейфа мяча.
  - Детекция отскоков мяча на основе инверсии знака вертикальной скорости траектории.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ttcv.tracking.kalman import BallKalmanFilter

if TYPE_CHECKING:
    from ttcv.geometry.homography import TableHomography


@dataclass
class TrackPoint:
    """Отслеженное положение мяча на конкретном кадре.

    Атрибуты:
        frame_id: Номер кадра.
        x: Оцененная горизонтальная координата в пикселях камеры.
        y: Оцененная вертикальная координата в пикселях камеры.
        vx: Горизонтальная скорость (пикс/кадр).
        vy: Вертикальная скорость (пикс/кадр).
        speed_px: Евклидова скорость (пикс/кадр).
        speed_kmh: Метрическая скорость в км/ч (при наличии гомографии и FPS).
        is_predicted: True, если положение восстановлено предсказанием Калмана при перекрытии/размытии.
        confidence: Уверенность детектора (1.0 для ручной разметки / эталона).
        table_x_mm: Спроецированная координата X на столе в мм (при наличии гомографии).
        table_y_mm: Спроецированная координата Y на столе в мм (при наличии гомографии).
    """

    frame_id: int
    x: float
    y: float
    vx: float
    vy: float
    speed_px: float
    speed_kmh: float | None = None
    is_predicted: bool = False
    confidence: float = 1.0
    table_x_mm: float | None = None
    table_y_mm: float | None = None

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def table_xy_m(self) -> tuple[float, float] | None:
        if self.table_x_mm is not None and self.table_y_mm is not None:
            return (self.table_x_mm / 1000.0, self.table_y_mm / 1000.0)
        return None


class BallTracker:
    """Отслеживание мяча для настольного тенниса на видеокадрах с фильтрацией Калмана.

    Заполняет короткие промежутки отсутствия детекций (до `max_lost_frames`),
    вычисляет физическую скорость через гомографию и обнаруживает отскоки от стола.
    """

    def __init__(
        self,
        homography: TableHomography | None = None,
        fps: float = 120.0,
        max_lost_frames: int = 15,
        max_tail_length: int = 30,
    ) -> None:
        """Инициализация BallTracker.

        Аргументы:
            homography: Опциональный экземпляр TableHomography для метрических проекций.
            fps: Частота кадров видео (в TTNet используется 120 fps; стандартное видео 30/60).
            max_lost_frames: Максимальное число подряд пропущенных кадров для предсказания перед сбросом.
            max_tail_length: Количество последних точек в памяти для кометного шлейфа.
        """
        self.homography = homography
        self.fps = fps
        self.max_lost_frames = max_lost_frames
        self.max_tail_length = max_tail_length

        self.kalman = BallKalmanFilter()
        self.lost_count = 0
        self.history: list[TrackPoint] = []
        self.tail: deque[tuple[float, float]] = deque(maxlen=max_tail_length)

        self._detected_count = 0
        self._predicted_count = 0

    def reset(self) -> None:
        """Сброс состояния трекера."""
        self.kalman.reset()
        self.lost_count = 0
        self.history.clear()
        self.tail.clear()
        self._detected_count = 0
        self._predicted_count = 0

    def update(
        self,
        frame_id: int,
        detection_xy: tuple[float, float] | None = None,
        confidence: float = 1.0,
    ) -> TrackPoint | None:
        """Обработка нового кадра с опциональной детекцией мяча.

        Аргументы:
            frame_id: Текущий индекс кадра видео.
            detection_xy: Обнаруженные координаты (x, y) или None / (-1, -1), если мяч не обнаружен.
            confidence: Оценка уверенности детектора [0.0, 1.0].

        Возвращает:
            Экземпляр :class:`TrackPoint` со сглаженным положением и скоростью, либо None при потере трека.
        """
        has_detection = (
            detection_xy is not None
            and detection_xy[0] >= 0
            and detection_xy[1] >= 0
        )

        # 1. Шаг предсказания
        pred_x, pred_y = self.kalman.predict()

        if has_detection:
            # 2. Обновление по измерениям
            est_x, est_y = self.kalman.update(detection_xy)
            self.lost_count = 0
            is_predicted = False
            self._detected_count += 1
        else:
            # Пропуск детекции: проверяем, укладываемся ли в допустимый горизонт предсказания
            if self.kalman.initialized and self.lost_count < self.max_lost_frames:
                est_x, est_y = pred_x, pred_y
                self.lost_count += 1
                is_predicted = True
                self._predicted_count += 1
            else:
                self.lost_count += 1
                return None

        # 3. Извлечение скоростей
        vx, vy = self.kalman.velocity
        speed_px = self.kalman.speed_px

        # 4. Метрическая проекция и реальная скорость в км/ч
        speed_kmh: float | None = None
        table_x_mm: float | None = None
        table_y_mm: float | None = None

        if self.homography is not None:
            pt_table = self.homography.project_to_table(np.array([est_x, est_y]))
            table_x_mm = float(pt_table[0])
            table_y_mm = float(pt_table[1])

            # Расчет метрической скорости по недавней истории
            if len(self.history) >= 1:
                prev = self.history[-1]
                if prev.table_x_mm is not None and prev.table_y_mm is not None:
                    dt_sec = max(1, frame_id - prev.frame_id) / self.fps
                    dist_m = float(
                        np.hypot(
                            (table_x_mm - prev.table_x_mm) / 1000.0,
                            (table_y_mm - prev.table_y_mm) / 1000.0,
                        )
                    )
                    v_mps = dist_m / dt_sec
                    # Фильтрация нефизичных скачков (> 150 км/ч)
                    speed_kmh = min(150.0, v_mps * 3.6)

        point = TrackPoint(
            frame_id=frame_id,
            x=est_x,
            y=est_y,
            vx=vx,
            vy=vy,
            speed_px=speed_px,
            speed_kmh=speed_kmh,
            is_predicted=is_predicted,
            confidence=confidence if not is_predicted else 0.5,
            table_x_mm=table_x_mm,
            table_y_mm=table_y_mm,
        )

        self.history.append(point)
        self.tail.append((est_x, est_y))
        return point

    # ── Аналитика траектории и отскоков ───────────────────────────────────────

    def get_trajectory_array(self) -> np.ndarray:
        """Возвращает массив формы (N, 3) с [frame_id, x, y] для всех отслеженных кадров."""
        if not self.history:
            return np.empty((0, 3), dtype=np.float32)
        return np.array([[p.frame_id, p.x, p.y] for p in self.history], dtype=np.float32)

    def get_recent_tail(self) -> list[tuple[float, float]]:
        """Возвращает список недавних точек (x, y) кометного шлейфа."""
        return list(self.tail)

    def detect_bounces(
        self,
        min_vy_change: float = 4.0,
        min_frame_gap: int = 15,
    ) -> list[int]:
        """Детекция кадров отскока, где вертикальная скорость меняет знак с движения вниз на движение вверх.

        В координатах камеры ось Y направлена вниз.
        До отскока: мяч падает вниз -> vy > 0.
        После отскока: мяч поднимается вверх -> vy < 0.

        Аргументы:
            min_vy_change: Минимальный скачок вертикальной скорости (vy_prev - vy_next).
            min_frame_gap: Минимальное количество кадров между последовательными отскоками.

        Возвращает:
            Список frame_id, на которых произошли отскоки.
        """
        if len(self.history) < 5:
            return []

        bounce_frames: list[int] = []
        last_bounce_fid = -1000

        for i in range(1, len(self.history) - 1):
            prev_pt = self.history[i - 1]
            curr_pt = self.history[i]
            next_pt = self.history[i + 1]

            # Движение вниз до точки, движение вверх после
            if prev_pt.vy > 0 and next_pt.vy < 0:
                vy_jump = prev_pt.vy - next_pt.vy
                if vy_jump >= min_vy_change:
                    if curr_pt.frame_id - last_bounce_fid >= min_frame_gap:
                        bounce_frames.append(curr_pt.frame_id)
                        last_bounce_fid = curr_pt.frame_id

        return bounce_frames

    @property
    def stats(self) -> dict:
        """Возвращает сводку эффективности трекинга."""
        total = len(self.history)
        speeds = [p.speed_kmh for p in self.history if p.speed_kmh is not None]
        return {
            "total_tracked_frames": total,
            "detected_frames": self._detected_count,
            "predicted_frames": self._predicted_count,
            "interpolation_rate": (self._predicted_count / total * 100.0) if total else 0.0,
            "max_speed_kmh": max(speeds) if speeds else 0.0,
            "mean_speed_kmh": float(np.mean(speeds)) if speeds else 0.0,
        }
