"""Аналитика розыгрышей и извлечение физических метрик.

Извлекает:
  - Общую длительность розыгрыша и количество ударов.
  - Количество отскоков и их пространственное распределение (Игрок A против Игрока B, глубокие против коротких).
  - Профили скорости мяча и распределение скорости в км/ч.
  - Сегментацию ударов (интервал, скорость и целевая зона для каждого удара).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ttcv.geometry.table import (
    TABLE_LENGTH_MM,
    DepthZone,
    LateralZone,
    TableDimensions,
    TablePoint,
    TableSide,
)

if TYPE_CHECKING:
    from ttcv.geometry.homography import TableHomography
    from ttcv.tracking.tracker import TrackPoint


@dataclass
class ShotInfo:
    """Информация об отдельном ударе в ходе розыгрыша.

    Атрибуты:
        shot_id: Индекс удара в розыгрыше (0, 1, 2...).
        start_frame: Начальный кадр удара.
        end_frame: Конечный кадр удара (в момент последующего отскока / удара).
        duration_sec: Время полета в секундах.
        bounce_frame: Кадр, в котором мяч отскочил на половине стола принимающего.
        hitter: Сторона стола бьющего игрока (TableSide: Игрок A или Игрок B).
        receiver: Сторона стола принимающего игрока (TableSide).
        bounce_xy_mm: Координаты (x, y) на столе в мм при отскоке.
        bounce_zone: Детальное описание зоны (например, 'player_b_deep_left').
        max_speed_kmh: Пиковая скорость, достигнутая во время удара.
        mean_speed_kmh: Средняя скорость полета в км/ч.
    """

    shot_id: int
    start_frame: int
    end_frame: int
    duration_sec: float
    bounce_frame: int | None = None
    hitter: TableSide = TableSide.PLAYER_A
    receiver: TableSide = TableSide.PLAYER_B
    bounce_xy_mm: tuple[float, float] | None = None
    bounce_zone: str = "unknown"
    max_speed_kmh: float = 0.0
    mean_speed_kmh: float = 0.0


@dataclass
class RallyMetrics:
    """Агрегированная аналитика матча и розыгрыша.

    Атрибуты:
        total_frames: Общее количество обработанных кадров.
        duration_sec: Длительность в секундах.
        fps: Частота кадров видео.
        total_bounces: Количество обнаруженных отскоков.
        bounces_player_a: Количество отскоков на половине Игрока A.
        bounces_player_b: Количество отскоков на половине Игрока B.
        depth_distribution: Распределение по глубине ({'deep': N, 'medium': N, 'short': N}).
        lateral_distribution: Распределение по сторонам ({'left': N, 'right': N}).
        max_speed_kmh: Зафиксированная максимальная скорость мяча.
        mean_speed_kmh: Средняя скорость мяча.
        speed_series: Список кортежей (time_sec, speed_kmh) для построения графиков.
        shots: Сегментированные удары.
        interpolation_rate: Доля кадров (в процентах), восстановленных трекером.
    """

    total_frames: int
    duration_sec: float
    fps: float
    total_bounces: int = 0
    bounces_player_a: int = 0
    bounces_player_b: int = 0
    depth_distribution: dict[str, int] = field(
        default_factory=lambda: {"deep": 0, "medium": 0, "short": 0, "outside": 0}
    )
    lateral_distribution: dict[str, int] = field(
        default_factory=lambda: {"left": 0, "right": 0, "outside": 0}
    )
    max_speed_kmh: float = 0.0
    mean_speed_kmh: float = 0.0
    speed_series: list[tuple[float, float]] = field(default_factory=list)
    shots: list[ShotInfo] = field(default_factory=list)
    interpolation_rate: float = 0.0

    def summary_dict(self) -> dict:
        """Возвращает структурированный словарь для JSON-отчетов и дашборда."""
        return {
            "duration_sec": round(self.duration_sec, 2),
            "total_frames": self.total_frames,
            "fps": self.fps,
            "total_bounces": self.total_bounces,
            "bounces_player_a": self.bounces_player_a,
            "bounces_player_b": self.bounces_player_b,
            "depth_distribution": self.depth_distribution,
            "lateral_distribution": self.lateral_distribution,
            "max_speed_kmh": round(self.max_speed_kmh, 1),
            "mean_speed_kmh": round(self.mean_speed_kmh, 1),
            "total_shots": len(self.shots),
            "interpolation_rate_pct": round(self.interpolation_rate, 1),
        }


class RallyAnalyzer:
    """Извлечение игровой аналитики и физических параметров из отслеженных траекторий мяча."""

    def __init__(
        self,
        homography: TableHomography | None = None,
        fps: float = 120.0,
    ) -> None:
        """Инициализация RallyAnalyzer.

        Аргументы:
            homography: Опциональный экземпляр TableHomography.
            fps: Частота кадров видеопоследовательности.
        """
        self.homography = homography
        self.fps = fps

    def analyze_trajectory(
        self,
        track_points: list[TrackPoint],
        annotated_bounce_frames: list[int] | None = None,
    ) -> RallyMetrics:
        """Анализ полной последовательности точек TrackPoint.

        Аргументы:
            track_points: История трекинга от BallTracker.
            annotated_bounce_frames: Опциональный список известных индексов кадров с отскоками.

        Возвращает:
            Вычисленный объект :class:`RallyMetrics`.
        """
        if not track_points:
            return RallyMetrics(total_frames=0, duration_sec=0.0, fps=self.fps)

        total_frames = len(track_points)
        min_fid = track_points[0].frame_id
        max_fid = track_points[-1].frame_id
        duration_sec = (max_fid - min_fid + 1) / self.fps

        # 1. Скорости
        speeds: list[float] = []
        speed_series: list[tuple[float, float]] = []
        predicted_count = 0

        for pt in track_points:
            t_sec = (pt.frame_id - min_fid) / self.fps
            if pt.is_predicted:
                predicted_count += 1

            if pt.speed_kmh is not None and pt.speed_kmh > 0:
                speeds.append(pt.speed_kmh)
                speed_series.append((round(t_sec, 3), round(pt.speed_kmh, 1)))

        max_speed = max(speeds) if speeds else 0.0
        mean_speed = float(np.mean(speeds)) if speeds else 0.0
        interp_rate = (predicted_count / total_frames * 100.0) if total_frames else 0.0

        # 2. Отскоки: используем предоставленные или определяем по вертикальной скорости
        if annotated_bounce_frames is not None:
            bounce_fids = set(annotated_bounce_frames)
        else:
            bounce_fids = set(self._detect_bounces(track_points))

        # 3. Анализ отскоков на столе
        bounces_a = 0
        bounces_b = 0
        depth_dist = {"deep": 0, "medium": 0, "short": 0, "outside": 0}
        lateral_dist = {"left": 0, "right": 0, "outside": 0}

        pt_map = {p.frame_id: p for p in track_points}
        sorted_bounce_fids = sorted(f for f in bounce_fids if f in pt_map)

        for b_fid in sorted_bounce_fids:
            pt = pt_map[b_fid]
            if pt.table_x_mm is not None and pt.table_y_mm is not None:
                tp = TableDimensions.classify_point(pt.table_x_mm, pt.table_y_mm)
                if tp.is_on_table(margin_mm=50.0):
                    if tp.side == TableSide.PLAYER_A:
                        bounces_a += 1
                    else:
                        bounces_b += 1
                    depth_dist[tp.depth_zone.value] += 1
                    lateral_dist[tp.lateral_zone.value] += 1
                else:
                    depth_dist["outside"] += 1
                    lateral_dist["outside"] += 1

        # 4. Сегментация ударов
        shots = self._segment_shots(track_points, sorted_bounce_fids)

        return RallyMetrics(
            total_frames=total_frames,
            duration_sec=duration_sec,
            fps=self.fps,
            total_bounces=len(sorted_bounce_fids),
            bounces_player_a=bounces_a,
            bounces_player_b=bounces_b,
            depth_distribution=depth_dist,
            lateral_distribution=lateral_dist,
            max_speed_kmh=max_speed,
            mean_speed_kmh=mean_speed,
            speed_series=speed_series,
            shots=shots,
            interpolation_rate=interp_rate,
        )

    def _detect_bounces(
        self,
        track_points: list[TrackPoint],
        min_vy_change: float = 3.5,
    ) -> list[int]:
        """Определение кадров отскоков, где вертикальная скорость меняет знак (движение вниз сменяется движением вверх)."""
        bounce_frames: list[int] = []
        last_bounce_fid = -1000

        for i in range(1, len(track_points) - 1):
            prev_pt = track_points[i - 1]
            curr_pt = track_points[i]
            next_pt = track_points[i + 1]

            if prev_pt.vy > 0 and next_pt.vy < 0:
                vy_jump = prev_pt.vy - next_pt.vy
                if vy_jump >= min_vy_change and (curr_pt.frame_id - last_bounce_fid >= 12):
                    bounce_frames.append(curr_pt.frame_id)
                    last_bounce_fid = curr_pt.frame_id

        return bounce_frames

    def _segment_shots(
        self,
        track_points: list[TrackPoint],
        bounce_fids: list[int],
    ) -> list[ShotInfo]:
        """Группировка точек траектории в отдельные удары, разделенные отскоками."""
        shots: list[ShotInfo] = []
        if len(bounce_fids) < 1:
            return shots

        pt_map = {p.frame_id: p for p in track_points}
        min_fid = track_points[0].frame_id

        prev_bound_fid = min_fid
        for shot_idx, b_fid in enumerate(bounce_fids):
            segment_pts = [
                pt_map[f]
                for f in range(prev_bound_fid, b_fid + 1)
                if f in pt_map
            ]
            if not segment_pts:
                prev_bound_fid = b_fid
                continue

            seg_speeds = [p.speed_kmh for p in segment_pts if p.speed_kmh is not None]
            b_pt = pt_map[b_fid]

            bounce_xy = (
                (b_pt.table_x_mm, b_pt.table_y_mm)
                if (b_pt.table_x_mm and b_pt.table_y_mm)
                else None
            )

            zone_str = "unknown"
            if bounce_xy:
                tp = TableDimensions.classify_point(bounce_xy[0], bounce_xy[1])
                zone_str = tp.zone_name
                receiver = tp.side
                hitter = TableSide.PLAYER_B if receiver == TableSide.PLAYER_A else TableSide.PLAYER_A
            else:
                hitter = TableSide.PLAYER_A
                receiver = TableSide.PLAYER_B

            dur = (b_fid - prev_bound_fid) / self.fps
            shots.append(
                ShotInfo(
                    shot_id=shot_idx,
                    start_frame=prev_bound_fid,
                    end_frame=b_fid,
                    duration_sec=round(dur, 3),
                    bounce_frame=b_fid,
                    hitter=hitter,
                    receiver=receiver,
                    bounce_xy_mm=bounce_xy,
                    bounce_zone=zone_str,
                    max_speed_kmh=round(max(seg_speeds), 1) if seg_speeds else 0.0,
                    mean_speed_kmh=round(float(np.mean(seg_speeds)), 1) if seg_speeds else 0.0,
                )
            )
            prev_bound_fid = b_fid

        return shots
