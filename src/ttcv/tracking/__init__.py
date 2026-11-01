"""Подпакет трекинга: детекция мяча и оценка траектории в настольном теннисе.

Содержит:
  - BallKalmanFilter: 6-параметрический кинематический фильтр Калмана постоянного ускорения.
  - BallTracker: Высокоуровневый трекер с кометным шлейфом, оценкой скорости и детекцией отскоков.
  - Детекторы: BaseBallDetector, BallDetection, MockBallDetector, ColorMotionBallDetector, YOLOv8BallDetector.
  - Генератор датасета: convert_markup_to_yolo_format, generate_synthetic_rally.
"""

from ttcv.tracking.dataset_generator import (
    convert_markup_to_yolo_format,
    generate_synthetic_rally,
)
from ttcv.tracking.detector import (
    BallDetection,
    BaseBallDetector,
    ColorMotionBallDetector,
    MockBallDetector,
    YOLOv8BallDetector,
)
from ttcv.tracking.kalman import BallKalmanFilter
from ttcv.tracking.tracker import BallTracker, TrackPoint

__all__ = [
    "BallKalmanFilter",
    "BallTracker",
    "TrackPoint",
    "BaseBallDetector",
    "BallDetection",
    "MockBallDetector",
    "ColorMotionBallDetector",
    "YOLOv8BallDetector",
    "convert_markup_to_yolo_format",
    "generate_synthetic_rally",
]
