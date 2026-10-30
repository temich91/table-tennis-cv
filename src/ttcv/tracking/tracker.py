"""BallTracker: High-level table tennis ball tracker.

Features:
  - Integrates BallKalmanFilter for trajectory smoothing and missing frame filling.
  - Computes real-time metric speed in km/h using TableHomography and camera FPS.
  - Maintains trailing trajectory queue for drawing motion comet tails.
  - Performs trajectory-based bounce detection (vertical velocity sign inversion).
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
    """A tracked ball position at a specific frame.

    Attributes:
        frame_id: Frame number.
        x: Estimated horizontal coordinate in camera pixels.
        y: Estimated vertical coordinate in camera pixels.
        vx: Horizontal velocity (px/frame).
        vy: Vertical velocity (px/frame).
        speed_px: Euclidean speed (px/frame).
        speed_kmh: Metric speed in km/h (if homography & fps available).
        is_predicted: True if filled by Kalman prediction during occlusion/motion blur.
        confidence: Detection confidence (1.0 for manual markup / ground truth).
        table_x_mm: Projected X coordinate on table in mm (if homography available).
        table_y_mm: Projected Y coordinate on table in mm (if homography available).
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
    """Tracks a table tennis ball across video frames with Kalman filtering.

    Fills short occlusion / motion-blur gaps (up to `max_lost_frames`),
    computes physical speed via homography, and detects table bounces.
    """

    def __init__(
        self,
        homography: TableHomography | None = None,
        fps: float = 120.0,
        max_lost_frames: int = 15,
        max_tail_length: int = 30,
    ) -> None:
        """Initialize the BallTracker.

        Args:
            homography: Optional TableHomography instance for metric projections.
            fps: Video frames per second (TTNet dataset uses 120 fps; standard video 30/60).
            max_lost_frames: Max consecutive missing frames to predict before resetting.
            max_tail_length: Number of recent points to keep in memory for the comet tail.
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
        """Reset the tracker state."""
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
        """Process a new frame with an optional ball detection.

        Args:
            frame_id: Current video frame index.
            detection_xy: Detected (x, y) coordinates, or None / (-1, -1) if not detected.
            confidence: Detector confidence score [0.0, 1.0].

        Returns:
            :class:`TrackPoint` with smoothed position and speed, or None if lost.
        """
        has_detection = (
            detection_xy is not None
            and detection_xy[0] >= 0
            and detection_xy[1] >= 0
        )

        # 1. Prediction step
        pred_x, pred_y = self.kalman.predict()

        if has_detection:
            # 2. Measurement update
            est_x, est_y = self.kalman.update(detection_xy)
            self.lost_count = 0
            is_predicted = False
            self._detected_count += 1
        else:
            # Missing detection: check if within allowable prediction horizon
            if self.kalman.initialized and self.lost_count < self.max_lost_frames:
                est_x, est_y = pred_x, pred_y
                self.lost_count += 1
                is_predicted = True
                self._predicted_count += 1
            else:
                self.lost_count += 1
                return None

        # 3. Extract velocity and speed
        vx, vy = self.kalman.velocity
        speed_px = self.kalman.speed_px

        # 4. Metric projection & real speed in km/h
        speed_kmh: float | None = None
        table_x_mm: float | None = None
        table_y_mm: float | None = None

        if self.homography is not None:
            pt_table = self.homography.project_to_table(np.array([est_x, est_y]))
            table_x_mm = float(pt_table[0])
            table_y_mm = float(pt_table[1])

            # Calculate metric speed from recent history
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
                    # Filter out non-physical teleports (> 150 km/h)
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

    # ── Trajectory & Bounce Analytics ─────────────────────────────────────────

    def get_trajectory_array(self) -> np.ndarray:
        """Return array of shape (N, 3) with [frame_id, x, y] for all tracked frames."""
        if not self.history:
            return np.empty((0, 3), dtype=np.float32)
        return np.array([[p.frame_id, p.x, p.y] for p in self.history], dtype=np.float32)

    def get_recent_tail(self) -> list[tuple[float, float]]:
        """Return the recent (x, y) points in the comet tail."""
        return list(self.tail)

    def detect_bounces(
        self,
        min_vy_change: float = 4.0,
        min_frame_gap: int = 15,
    ) -> list[int]:
        """Detect bounce frames where vertical velocity flips from downward to upward.

        In camera coordinates, Y increases downward.
        Before bounce: ball falls downwards -> vy > 0.
        After bounce: ball rises upwards -> vy < 0.

        Args:
            min_vy_change: Minimum jump in vertical velocity (vy_prev - vy_next).
            min_frame_gap: Minimum frames between consecutive bounces.

        Returns:
            List of frame_ids where bounces occurred.
        """
        if len(self.history) < 5:
            return []

        bounce_frames: list[int] = []
        last_bounce_fid = -1000

        for i in range(1, len(self.history) - 1):
            prev_pt = self.history[i - 1]
            curr_pt = self.history[i]
            next_pt = self.history[i + 1]

            # Downward before, upward after
            if prev_pt.vy > 0 and next_pt.vy < 0:
                vy_jump = prev_pt.vy - next_pt.vy
                if vy_jump >= min_vy_change:
                    if curr_pt.frame_id - last_bounce_fid >= min_frame_gap:
                        bounce_frames.append(curr_pt.frame_id)
                        last_bounce_fid = curr_pt.frame_id

        return bounce_frames

    @property
    def stats(self) -> dict:
        """Return tracking performance summary."""
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
