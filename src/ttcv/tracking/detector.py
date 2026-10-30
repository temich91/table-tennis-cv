"""Ball detector interfaces and implementations.

Detectors:
  - BaseBallDetector: Abstract base class for ball detectors.
  - BallDetection: Dataclass storing bounding box, center, and confidence.
  - MockBallDetector: Replays ground-truth ball markup (with optional noise / dropouts).
  - ColorMotionBallDetector: Fast classical CV detector using HSV color & circularity filtering.
  - YOLOv8BallDetector: Ultralytics YOLOv8 / YOLOv11 wrapper for trained weights.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class BallDetection:
    """A detected ball candidate in a frame.

    Attributes:
        x: Center horizontal pixel coordinate.
        y: Center vertical pixel coordinate.
        bbox: Bounding box tuple (x1, y1, x2, y2).
        confidence: Detection confidence score [0.0, 1.0].
        class_name: Label name (default: "ball").
    """

    x: float
    y: float
    bbox: tuple[float, float, float, float]
    confidence: float = 1.0
    class_name: str = "ball"

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class BaseBallDetector(ABC):
    """Abstract base class for all ball detection models."""

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_id: int | None = None) -> list[BallDetection]:
        """Detect the ball in a single image/video frame.

        Args:
            frame: (H, W, 3) BGR or RGB image.
            frame_id: Optional frame index for sequence/stateful detectors.

        Returns:
            List of detected :class:`BallDetection` instances, sorted by confidence desc.
        """
        pass


class MockBallDetector(BaseBallDetector):
    """Detector that replays ground-truth annotations with optional simulated noise.

    Enables testing tracking and analytics without requiring raw video or GPU.
    """

    def __init__(
        self,
        markup_path_or_dict: str | Path | dict[str, dict[str, int]],
        noise_std: float = 0.0,
        dropout_prob: float = 0.0,
        seed: int = 42,
    ) -> None:
        """Initialize MockBallDetector.

        Args:
            markup_path_or_dict: Path to ball_markup.json or loaded dictionary.
            noise_std: Standard deviation of Gaussian noise added to coordinates in px.
            dropout_prob: Probability [0.0, 1.0] of simulating missing detection.
            seed: Random seed for reproducibility.
        """
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.rng = np.random.default_rng(seed)

        if isinstance(markup_path_or_dict, (str, Path)):
            raw = json.loads(Path(markup_path_or_dict).read_text(encoding="utf-8"))
        else:
            raw = markup_path_or_dict

        self.markup: dict[int, tuple[int, int]] = {}
        for fid_str, pos in raw.items():
            self.markup[int(fid_str)] = (int(pos["x"]), int(pos["y"]))

    def detect(self, frame: np.ndarray, frame_id: int | None = None) -> list[BallDetection]:
        if frame_id is None or frame_id not in self.markup:
            return []

        x, y = self.markup[frame_id]
        if x < 0 or y < 0:
            return []  # Invisible in ground truth

        # Simulated dropout (motion blur / occlusion)
        if self.dropout_prob > 0.0 and self.rng.random() < self.dropout_prob:
            return []

        # Simulated detector noise
        if self.noise_std > 0.0:
            x = float(x + self.rng.normal(0.0, self.noise_std))
            y = float(y + self.rng.normal(0.0, self.noise_std))

        # 20x20 bounding box
        bbox = (float(x - 10), float(y - 10), float(x + 10), float(y + 10))
        return [BallDetection(x=float(x), y=float(y), bbox=bbox, confidence=0.98)]


class ColorMotionBallDetector(BaseBallDetector):
    """Fast classical computer vision ball detector.

    Filters by HSV color (white / orange ball) and circular contour shape.
    Works on CPU without pretrained neural network weights.
    """

    def __init__(
        self,
        ball_color: str = "white",
        min_radius: int = 2,
        max_radius: int = 15,
        min_circularity: float = 0.45,
        roi_bbox: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Initialize ColorMotionBallDetector.

        Args:
            ball_color: "white" or "orange".
            min_radius: Minimum ball radius in pixels.
            max_radius: Maximum ball radius in pixels.
            min_circularity: Minimum circularity = 4*pi*area / (perimeter^2).
            roi_bbox: Optional (x1, y1, x2, y2) to restrict search region.
        """
        self.ball_color = ball_color.lower()
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_circularity = min_circularity
        self.roi_bbox = roi_bbox

    def detect(self, frame: np.ndarray, frame_id: int | None = None) -> list[BallDetection]:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Color mask
        if self.ball_color == "white":
            # Low saturation, high brightness
            mask = cv2.inRange(hsv, (0, 0, 195), (180, 70, 255))
        elif self.ball_color == "orange":
            # Orange hue [5, 25]
            mask = cv2.inRange(hsv, (5, 100, 120), (25, 255, 255))
        else:
            raise ValueError(f"Unknown ball color: {self.ball_color}")

        # Crop to ROI if specified
        if self.roi_bbox is not None:
            x1, y1, x2, y2 = self.roi_bbox
            roi_mask = np.zeros_like(mask)
            roi_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            mask = roi_mask

        # Morphological opening to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[BallDetection] = []

        min_area = np.pi * (self.min_radius**2)
        max_area = np.pi * (self.max_radius**2)

        for c in cnts:
            area = cv2.contourArea(c)
            if not (min_area <= area <= max_area):
                continue

            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue

            circularity = 4.0 * np.pi * (area / (peri**2))
            if circularity >= self.min_circularity:
                (cx, cy), radius = cv2.minEnclosingCircle(c)
                bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
                conf = min(1.0, float(circularity))
                detections.append(
                    BallDetection(
                        x=float(cx),
                        y=float(cy),
                        bbox=bbox,
                        confidence=conf,
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


class YOLOv8BallDetector(BaseBallDetector):
    """Ultralytics YOLO detector wrapper for trained weights."""

    def __init__(
        self,
        weights_path: str | Path,
        conf_thresh: float = 0.25,
        imgsz: int = 1280,
        device: str | None = None,
    ) -> None:
        """Initialize YOLO detector.

        Args:
            weights_path: Path to best.pt or yolov8n.pt.
            conf_thresh: Confidence threshold for candidate detections.
            imgsz: Inference image size (1280 recommended for small balls).
            device: 'cuda', 'cpu', or None for auto.
        """
        try:
            from ultralytics import YOLO
        except ImportError as err:
            raise ImportError(
                "Ultralytics is required to use YOLOv8BallDetector. "
                "Install with: pip install ultralytics"
            ) from err

        self.model = YOLO(str(weights_path))
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.device = device

    def detect(self, frame: np.ndarray, frame_id: int | None = None) -> list[BallDetection]:
        results = self.model.predict(
            frame,
            conf=self.conf_thresh,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections: list[BallDetection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            detections.append(
                BallDetection(
                    x=cx,
                    y=cy,
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=conf,
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
