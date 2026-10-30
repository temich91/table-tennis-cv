"""Tests for ttcv.geometry (table dimensions, corner detection, homography, and visualizers)."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ttcv.geometry import (
    BORDER_LINE_WIDTH_MM,
    CENTER_LINE_WIDTH_MM,
    NET_HEIGHT_MM,
    NET_OVERHANG_MM,
    NET_TOTAL_LENGTH_MM,
    TABLE_HEIGHT_MM,
    TABLE_LENGTH_MM,
    TABLE_WIDTH_MM,
    DepthZone,
    LateralZone,
    TableCorners,
    TableDimensions,
    TableHomography,
    TablePoint,
    TableSide,
    detect_table_corners_from_mask,
    order_quadrilateral_points,
)
from ttcv.geometry.visualizer import (
    draw_camera_table_overlay,
    draw_table_bev,
    plot_bev_heatmap,
    plot_bev_trajectories,
)

SAMPLE_MASK_PATH = Path("data/raw/segmentation_masks/1278.png")


# ── Table Specifications ──────────────────────────────────────────────────────


class TestTableDimensions:
    def test_dimensions_ittf_standard(self):
        assert TABLE_LENGTH_MM == 2740.0
        assert TABLE_WIDTH_MM == 1525.0
        assert TABLE_HEIGHT_MM == 760.0
        assert NET_HEIGHT_MM == 152.5
        assert NET_OVERHANG_MM == 152.5
        assert NET_TOTAL_LENGTH_MM == 1830.0
        assert BORDER_LINE_WIDTH_MM == 20.0
        assert CENTER_LINE_WIDTH_MM == 3.0

    def test_canonical_corners_shape(self):
        corners = TableDimensions.canonical_corners_mm()
        assert corners.shape == (4, 2)
        assert corners.dtype == np.float32
        # Check coordinates in mm
        np.testing.assert_array_equal(corners[0], [0.0, 0.0])
        np.testing.assert_array_equal(corners[1], [2740.0, 0.0])
        np.testing.assert_array_equal(corners[2], [2740.0, 1525.0])
        np.testing.assert_array_equal(corners[3], [0.0, 1525.0])

    def test_canonical_corners_meters(self):
        corners_m = TableDimensions.canonical_corners_m()
        assert corners_m.shape == (4, 2)
        np.testing.assert_allclose(corners_m[1], [2.74, 0.0])
        np.testing.assert_allclose(corners_m[2], [2.74, 1.525])

    def test_net_segment(self):
        start, end = TableDimensions.net_segment_mm()
        assert start[0] == 1370.0
        assert end[0] == 1370.0
        assert start[1] == -152.5
        assert end[1] == 1525.0 + 152.5


# ── TablePoint Classification ─────────────────────────────────────────────────


class TestTablePoint:
    def test_point_on_table(self):
        pt = TablePoint(x_mm=500.0, y_mm=600.0)
        assert pt.is_on_table()
        assert pt.x_m == 0.5
        assert pt.y_m == 0.6
        np.testing.assert_array_equal(pt.as_array(), [500.0, 600.0])

    def test_point_outside_table(self):
        pt = TablePoint(x_mm=-10.0, y_mm=2000.0)
        assert not pt.is_on_table()
        assert pt.is_on_table(margin_mm=500.0)

    def test_table_side_player_a(self):
        pt = TablePoint(x_mm=500.0, y_mm=700.0)
        assert pt.side == TableSide.PLAYER_A

    def test_table_side_player_b(self):
        pt = TablePoint(x_mm=2000.0, y_mm=700.0)
        assert pt.side == TableSide.PLAYER_B

    def test_table_side_net(self):
        pt = TablePoint(x_mm=1370.0, y_mm=700.0)
        assert pt.side == TableSide.NET

    def test_lateral_zones(self):
        left_pt = TablePoint(x_mm=500.0, y_mm=300.0)
        right_pt = TablePoint(x_mm=500.0, y_mm=1000.0)
        assert left_pt.lateral_zone == LateralZone.LEFT
        assert right_pt.lateral_zone == LateralZone.RIGHT

    def test_depth_zones(self):
        # Baseline is X=0 for Player A: deep is < 456.7 mm, short is near net (> 913.3 mm)
        deep_a = TablePoint(x_mm=200.0, y_mm=500.0)
        med_a = TablePoint(x_mm=600.0, y_mm=500.0)
        short_a = TablePoint(x_mm=1200.0, y_mm=500.0)
        assert deep_a.depth_zone == DepthZone.DEEP
        assert med_a.depth_zone == DepthZone.MEDIUM
        assert short_a.depth_zone == DepthZone.SHORT

    def test_zone_name(self):
        pt = TablePoint(x_mm=200.0, y_mm=300.0)
        assert pt.zone_name == "player_a_deep_left"


# ── Corner Ordering ───────────────────────────────────────────────────────────


class TestOrderQuadrilateralPoints:
    def test_already_ordered(self):
        pts = np.array([[10, 10], [100, 10], [100, 80], [10, 80]], dtype=np.float32)
        ordered = order_quadrilateral_points(pts)
        np.testing.assert_array_equal(ordered, pts)

    def test_shuffled_ordering(self):
        pts = np.array([[100, 80], [10, 10], [10, 80], [100, 10]], dtype=np.float32)
        ordered = order_quadrilateral_points(pts)
        expected = np.array([[10, 10], [100, 10], [100, 80], [10, 80]], dtype=np.float32)
        np.testing.assert_array_equal(ordered, expected)

    def test_perspective_trapezoid(self):
        # Mask points: Far edge (smaller y) narrower than near edge (larger y)
        pts = np.array([[66, 93], [236, 92], [210, 72], [88, 72]], dtype=np.float32)
        ordered = order_quadrilateral_points(pts)
        assert ordered[0, 1] <= ordered[2, 1]  # TL y <= BR y
        assert ordered[1, 1] <= ordered[3, 1]  # TR y <= BL y
        assert ordered[0, 0] < ordered[1, 0]   # TL x < TR x
        assert ordered[3, 0] < ordered[2, 0]   # BL x < BR x


# ── Corner Detection ──────────────────────────────────────────────────────────


class TestDetectTableCorners:
    @pytest.mark.skipif(not SAMPLE_MASK_PATH.exists(), reason="Sample mask not available")
    def test_detect_from_sample_mask(self):
        corners = detect_table_corners_from_mask(SAMPLE_MASK_PATH, target_resolution=(1920, 1080))
        assert isinstance(corners, TableCorners)
        assert corners.corners_camera.shape == (4, 2)
        assert corners.target_size == (1920, 1080)
        assert corners.mask_size == (320, 128)
        # Check expected corners in 1920x1080
        assert 500 < corners.top_left[0] < 550
        assert 1200 < corners.top_right[0] < 1300
        assert 1350 < corners.bottom_right[0] < 1450
        assert 350 < corners.bottom_left[0] < 450

    def test_detect_from_synthetic_binary_mask(self):
        # Create a 200x200 binary image with a trapezoid table
        mask = np.zeros((200, 200), dtype=np.uint8)
        poly = np.array([[40, 120], [160, 120], [140, 60], [60, 60]], dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)

        corners = detect_table_corners_from_mask(mask, target_resolution=(1000, 1000))
        assert corners.corners_camera.shape == (4, 2)
        # Scaled by 5x (1000 / 200)
        assert corners.corners_camera[0, 0] == pytest.approx(300, abs=15)
        assert corners.corners_camera[0, 1] == pytest.approx(300, abs=15)

    def test_missing_file_raises_error(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            detect_table_corners_from_mask(tmp_path / "non_existent.png")

    def test_empty_mask_raises_error(self):
        empty_mask = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            detect_table_corners_from_mask(empty_mask)


# ── Homography & Projections ──────────────────────────────────────────────────


class TestTableHomography:
    @pytest.fixture
    def homography(self) -> TableHomography:
        if SAMPLE_MASK_PATH.exists():
            return TableHomography.from_mask(SAMPLE_MASK_PATH)
        # Synthetic corners
        corners = np.array([[528, 607.5], [1260, 607.5], [1416, 776.25], [396, 784.7]], dtype=np.float32)
        return TableHomography(corners)

    def test_corner_projection_to_table(self, homography: TableHomography):
        corners_cam = homography.corners.corners_camera
        pts_table = homography.project_to_table(corners_cam)

        # Corners should map to (0,0), (2740,0), (2740,1525), (0,1525)
        expected = TableDimensions.canonical_corners_mm()
        np.testing.assert_allclose(pts_table, expected, atol=1e-2)

    def test_roundtrip_camera_to_table_and_back(self, homography: TableHomography):
        orig_cam = np.array([[900.0, 700.0], [600.0, 650.0], [1200.0, 720.0]], dtype=np.float32)
        table_pts = homography.project_to_table(orig_cam)
        recovered_cam = homography.project_table_to_camera(table_pts)
        np.testing.assert_allclose(recovered_cam, orig_cam, atol=1e-3)

    def test_bev_canvas_projection(self, homography: TableHomography):
        corners_cam = homography.corners.corners_camera
        bev_pts = homography.project_to_bev(corners_cam)

        w_canv, h_canv = homography.bev_size
        m = homography.bev_margin
        expected_bev = np.array(
            [[m, m], [w_canv - m, m], [w_canv - m, h_canv - m], [m, h_canv - m]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(bev_pts, expected_bev, atol=1e-2)

    def test_roundtrip_bev_and_back(self, homography: TableHomography):
        orig_cam = np.array([[900.0, 700.0], [1100.0, 680.0]], dtype=np.float32)
        bev_pts = homography.project_to_bev(orig_cam)
        recovered = homography.project_bev_to_camera(bev_pts)
        np.testing.assert_allclose(recovered, orig_cam, atol=1e-3)

    def test_warp_to_bev(self, homography: TableHomography):
        dummy_frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        warped = homography.warp_to_bev(dummy_frame)
        assert warped.shape == (homography.bev_size[1], homography.bev_size[0], 3)

    def test_classify_camera_point(self, homography: TableHomography):
        # Point on Player A half, right lateral zone
        pt_table_mm = np.array([[500.0, 1000.0]], dtype=np.float32)
        pt_cam = homography.project_table_to_camera(pt_table_mm)[0]

        tp = homography.classify_camera_point(float(pt_cam[0]), float(pt_cam[1]))
        assert tp.is_on_table()
        assert tp.side == TableSide.PLAYER_A
        assert tp.lateral_zone == LateralZone.RIGHT


# ── Visualizations ────────────────────────────────────────────────────────────


class TestVisualizers:
    @pytest.fixture
    def homography(self) -> TableHomography:
        if SAMPLE_MASK_PATH.exists():
            return TableHomography.from_mask(SAMPLE_MASK_PATH)
        corners = np.array([[528, 607.5], [1260, 607.5], [1416, 776.25], [396, 784.7]], dtype=np.float32)
        return TableHomography(corners)

    def test_draw_table_bev(self):
        img = draw_table_bev(canvas_size=(800, 500))
        assert img.shape == (500, 800, 3)
        assert img.dtype == np.uint8

    def test_plot_bev_heatmap(self, homography: TableHomography):
        # Generate some synthetic bounce points in camera space
        pts_mm = np.array([[500, 700], [800, 500], [2000, 800], [2200, 600]], dtype=np.float32)
        pts_cam = homography.project_table_to_camera(pts_mm)

        fig = plot_bev_heatmap(homography, pts_cam)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_bev_trajectories(self, homography: TableHomography):
        pts_mm = np.array([[500, 700], [1000, 650], [1370, 600], [2000, 700]], dtype=np.float32)
        pts_cam = homography.project_table_to_camera(pts_mm)

        fig = plot_bev_trajectories(homography, pts_cam)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_draw_camera_table_overlay(self, homography: TableHomography):
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        annotated = draw_camera_table_overlay(dummy_frame, homography)
        assert annotated.shape == (1080, 1920, 3)
