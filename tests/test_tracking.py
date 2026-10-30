"""Tests for ttcv.tracking (Kalman filter, tracker, detectors, dataset generator)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ttcv.geometry import TableHomography
from ttcv.tracking import (
    BallDetection,
    BallKalmanFilter,
    BallTracker,
    ColorMotionBallDetector,
    MockBallDetector,
    TrackPoint,
    YOLOv8BallDetector,
    convert_markup_to_yolo_format,
    generate_synthetic_rally,
)


# ── BallKalmanFilter ──────────────────────────────────────────────────────────


class TestBallKalmanFilter:
    def test_initial_state(self):
        kf = BallKalmanFilter()
        assert not kf.initialized
        assert kf.position == (0.0, 0.0)
        assert kf.velocity == (0.0, 0.0)

    def test_first_update_initializes_position(self):
        kf = BallKalmanFilter()
        pos = kf.update((100.0, 200.0))
        assert kf.initialized
        assert pos == (100.0, 200.0)
        assert kf.position == (100.0, 200.0)

    def test_prediction_moves_with_velocity(self):
        kf = BallKalmanFilter()
        kf.update((100.0, 200.0))
        # Provide second measurement to establish velocity (dx=10, dy=-5)
        kf.predict()
        kf.update((110.0, 195.0))

        # Next predict should move roughly along the same velocity
        pred_x, pred_y = kf.predict()
        assert pred_x > 115.0
        assert pred_y < 195.0

    def test_reset(self):
        kf = BallKalmanFilter()
        kf.update((50.0, 50.0))
        assert kf.initialized
        kf.reset()
        assert not kf.initialized
        assert kf.position == (0.0, 0.0)

    def test_reset_with_position(self):
        kf = BallKalmanFilter()
        kf.reset(initial_xy=(300.0, 400.0))
        assert kf.initialized
        assert kf.position == (300.0, 400.0)


# ── BallTracker ───────────────────────────────────────────────────────────────


class TestBallTracker:
    def test_tracking_continuous_detections(self):
        tracker = BallTracker(fps=60.0)
        points: list[TrackPoint] = []
        for fid in range(10):
            pt = tracker.update(frame_id=fid, detection_xy=(100.0 + fid * 10, 200.0 - fid * 2))
            assert pt is not None
            assert not pt.is_predicted
            points.append(pt)

        assert len(points) == 10
        assert tracker.stats["detected_frames"] == 10
        assert tracker.stats["predicted_frames"] == 0

    def test_gap_filling_interpolation(self):
        tracker = BallTracker(max_lost_frames=5)
        # Establish trajectory
        tracker.update(frame_id=0, detection_xy=(100.0, 200.0))
        tracker.update(frame_id=1, detection_xy=(110.0, 200.0))
        tracker.update(frame_id=2, detection_xy=(120.0, 200.0))

        # Dropouts for 2 frames
        pt3 = tracker.update(frame_id=3, detection_xy=None)
        pt4 = tracker.update(frame_id=4, detection_xy=(-1.0, -1.0))

        assert pt3 is not None
        assert pt3.is_predicted
        assert pt3.x > 120.0  # Continued along trajectory

        assert pt4 is not None
        assert pt4.is_predicted
        assert pt4.x > pt3.x

        assert tracker.stats["predicted_frames"] == 2

    def test_loss_after_max_lost_frames(self):
        tracker = BallTracker(max_lost_frames=3)
        tracker.update(frame_id=0, detection_xy=(100.0, 100.0))

        # 3 missing frames allowed
        for fid in range(1, 4):
            assert tracker.update(frame_id=fid, detection_xy=None) is not None

        # 4th missing frame exceeds max_lost_frames -> returns None
        lost_pt = tracker.update(frame_id=4, detection_xy=None)
        assert lost_pt is None

    def test_speed_kmh_with_homography(self):
        corners = np.array([[528, 607.5], [1260, 607.5], [1416, 776.25], [396, 784.7]], dtype=np.float32)
        homography = TableHomography(corners)
        tracker = BallTracker(homography=homography, fps=120.0)

        # Move horizontally across table
        pt1 = tracker.update(frame_id=0, detection_xy=(600.0, 650.0))
        pt2 = tracker.update(frame_id=1, detection_xy=(620.0, 650.0))

        assert pt2 is not None
        assert pt2.speed_kmh is not None
        assert pt2.speed_kmh > 0.0

    def test_comet_tail_queue(self):
        tracker = BallTracker(max_tail_length=5)
        for fid in range(10):
            tracker.update(frame_id=fid, detection_xy=(float(fid * 10), 100.0))

        tail = tracker.get_recent_tail()
        assert len(tail) == 5
        assert tail[-1][0] == pytest.approx(90.0, abs=0.1)

    def test_bounce_detection(self):
        tracker = BallTracker()
        # Simulate downward fall then upward bounce (in camera px, Y increases downwards)
        # Frames 0-4: falling down (Y increases from 500 to 700)
        # Frame 5: bounce at Y=700
        # Frames 6-10: rising up (Y decreases from 700 to 500)
        y_coords = [500, 550, 610, 660, 700, 650, 600, 550, 510, 480]
        for fid, y in enumerate(y_coords):
            tracker.update(frame_id=fid, detection_xy=(float(500 + fid * 10), float(y)))

        bounces = tracker.detect_bounces(min_vy_change=3.0)
        assert len(bounces) >= 1


# ── Detectors ─────────────────────────────────────────────────────────────────


class TestDetectors:
    def test_mock_detector(self):
        markup = {
            "10": {"x": 500, "y": 600},
            "11": {"x": -1, "y": -1},
        }
        detector = MockBallDetector(markup)
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Visible frame
        dets10 = detector.detect(dummy_frame, frame_id=10)
        assert len(dets10) == 1
        assert dets10[0].x == 500.0
        assert dets10[0].y == 600.0

        # Invisible frame
        dets11 = detector.detect(dummy_frame, frame_id=11)
        assert len(dets11) == 0

        # Unindexed frame
        dets99 = detector.detect(dummy_frame, frame_id=99)
        assert len(dets99) == 0

    def test_color_motion_detector(self):
        detector = ColorMotionBallDetector(ball_color="white", min_radius=3, max_radius=10)
        frame = np.full((300, 300, 3), 30, dtype=np.uint8)
        # Draw a bright white circular ball at (150, 150)
        import cv2
        cv2.circle(frame, (150, 150), 6, (255, 255, 255), -1)

        dets = detector.detect(frame)
        assert len(dets) >= 1
        assert abs(dets[0].x - 150.0) <= 2.0
        assert abs(dets[0].y - 150.0) <= 2.0

    def test_yolo_detector_import_error(self, monkeypatch):
        # Force ImportError for ultralytics if tested
        import sys
        monkeypatch.setitem(sys.modules, "ultralytics", None)

        with pytest.raises(ImportError):
            YOLOv8BallDetector("non_existent.pt")


# ── Dataset Generator ─────────────────────────────────────────────────────────


class TestDatasetGenerator:
    def test_convert_markup_to_yolo(self, tmp_path: Path):
        markup = {
            "1": {"x": 960, "y": 540},
            "2": {"x": -1, "y": -1},
            "3": {"x": 1000, "y": 600},
            "4": {"x": 1100, "y": 650},
            "5": {"x": 1200, "y": 700},
        }
        markup_file = tmp_path / "markup.json"
        markup_file.write_text(json.dumps(markup), encoding="utf-8")

        out_dir = tmp_path / "yolo"
        res = convert_markup_to_yolo_format(
            markup_file,
            out_dir,
            val_ratio=0.2,
            bbox_size_px=24,
            img_size=(1920, 1080),
        )

        assert res["total_frames"] == 5
        assert res["visible_balls"] == 4
        assert Path(res["dataset_yaml"]).exists()

        # Check a visible label file
        label_file = out_dir / "labels" / "train" / "frame_000001.txt"
        if not label_file.exists():
            label_file = out_dir / "labels" / "val" / "frame_000001.txt"
        assert label_file.exists()
        line = label_file.read_text().strip().split()
        assert line[0] == "0"
        assert float(line[1]) == pytest.approx(0.5, abs=1e-3)  # 960 / 1920
        assert float(line[2]) == pytest.approx(0.5, abs=1e-3)  # 540 / 1080

    def test_generate_synthetic_rally(self, tmp_path: Path):
        video_path = tmp_path / "synth.mp4"
        data = generate_synthetic_rally(
            output_video_path=video_path,
            num_frames=20,
            img_size=(640, 360),
            fps=30.0,
        )
        assert video_path.exists()
        assert data["frames_count"] == 20
        assert len(data["ball_markup"]) == 20
