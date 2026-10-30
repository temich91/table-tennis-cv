"""Tracking sub-package for table tennis ball detection and trajectory estimation.

Contains:
  - BallKalmanFilter: 6-state constant acceleration kinematic filter.
  - BallTracker: High-level tracker with comet tail, speed estimation, and bounce detection.
  - Detectors: BaseBallDetector, BallDetection, MockBallDetector, ColorMotionBallDetector, YOLOv8BallDetector.
  - Dataset Generator: convert_markup_to_yolo_format, generate_synthetic_rally.
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
