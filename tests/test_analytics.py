"""Тесты для ttcv.analytics (RallyAnalyzer, RallyMetrics, ShotInfo, форматирование отчетов)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ttcv.analytics import (
    RallyAnalyzer,
    RallyMetrics,
    ShotInfo,
    format_markdown_report,
    save_report_json,
)
from ttcv.geometry import TableHomography
from ttcv.geometry.table import TableSide
from ttcv.tracking.tracker import TrackPoint


@pytest.fixture
def sample_homography() -> TableHomography:
    mask_path = Path("data/raw/segmentation_masks/1278.png")
    if mask_path.exists():
        return TableHomography.from_mask(mask_path)
    corners = np.array([[528, 607.5], [1260, 607.5], [1416, 776.25], [396, 784.7]], dtype=np.float32)
    return TableHomography(corners)


@pytest.fixture
def synthetic_track_points(sample_homography: TableHomography) -> list[TrackPoint]:
    """Создание синтетической последовательности из 30 точек трека с одним явным отскоком."""
    points: list[TrackPoint] = []
    # Кадры 0-14: движение слева направо и падение вниз (vy > 0)
    # Кадр 15: отскок от поверхности стола (table_x_mm=800, table_y_mm=500)
    # Кадры 16-29: движение слева направо и подъем вверх (vy < 0)
    for fid in range(30):
        x = 500.0 + fid * 15.0
        if fid <= 15:
            y = 550.0 + fid * 10.0  # падение
            vy = 10.0
        else:
            y = 700.0 - (fid - 15) * 8.0  # подъем
            vy = -8.0

        pt_table = sample_homography.project_to_table(np.array([x, y]))
        points.append(
            TrackPoint(
                frame_id=fid,
                x=x,
                y=y,
                vx=15.0,
                vy=vy,
                speed_px=18.0,
                speed_kmh=45.0 + (fid % 5),
                is_predicted=False,
                confidence=0.95,
                table_x_mm=float(pt_table[0]),
                table_y_mm=float(pt_table[1]),
            )
        )
    return points


# ── RallyMetrics ──────────────────────────────────────────────────────────────


class TestRallyMetrics:
    def test_summary_dict_keys(self):
        m = RallyMetrics(
            total_frames=100,
            duration_sec=1.67,
            fps=60.0,
            total_bounces=3,
            bounces_player_a=1,
            bounces_player_b=2,
            max_speed_kmh=85.4,
            mean_speed_kmh=52.1,
            interpolation_rate=4.5,
        )
        d = m.summary_dict()
        assert d["total_frames"] == 100
        assert d["duration_sec"] == 1.67
        assert d["total_bounces"] == 3
        assert d["bounces_player_a"] == 1
        assert d["bounces_player_b"] == 2
        assert d["max_speed_kmh"] == 85.4
        assert d["mean_speed_kmh"] == 52.1
        assert d["interpolation_rate_pct"] == 4.5


# ── RallyAnalyzer ─────────────────────────────────────────────────────────────


class TestRallyAnalyzer:
    def test_empty_trajectory(self):
        analyzer = RallyAnalyzer()
        metrics = analyzer.analyze_trajectory([])
        assert metrics.total_frames == 0
        assert metrics.duration_sec == 0.0

    def test_analyzes_synthetic_trajectory(
        self,
        synthetic_track_points: list[TrackPoint],
        sample_homography: TableHomography,
    ):
        analyzer = RallyAnalyzer(homography=sample_homography, fps=60.0)
        metrics = analyzer.analyze_trajectory(synthetic_track_points)

        assert metrics.total_frames == 30
        assert metrics.duration_sec == pytest.approx(30 / 60.0, abs=1e-2)
        assert metrics.max_speed_kmh >= 45.0
        assert metrics.mean_speed_kmh >= 40.0
        assert metrics.total_bounces >= 1

    def test_shot_segmentation(
        self,
        synthetic_track_points: list[TrackPoint],
        sample_homography: TableHomography,
    ):
        analyzer = RallyAnalyzer(homography=sample_homography, fps=60.0)
        metrics = analyzer.analyze_trajectory(
            synthetic_track_points,
            annotated_bounce_frames=[15],
        )

        assert len(metrics.shots) == 1
        shot = metrics.shots[0]
        assert shot.bounce_frame == 15
        assert shot.duration_sec > 0.0
        assert shot.max_speed_kmh > 0.0


# ── Отчеты (Reports) ──────────────────────────────────────────────────────────


class TestReports:
    def test_format_markdown_report(self):
        m = RallyMetrics(
            total_frames=120,
            duration_sec=1.0,
            fps=120.0,
            total_bounces=2,
            bounces_player_a=1,
            bounces_player_b=1,
            max_speed_kmh=78.5,
            mean_speed_kmh=45.2,
        )
        md = format_markdown_report(m, title="Test Report")
        assert "Test Report" in md
        assert "78.5 km/h" in md
        assert "Player A half:" in md

    def test_save_report_json(self, tmp_path: Path):
        m = RallyMetrics(
            total_frames=60,
            duration_sec=0.5,
            fps=120.0,
            total_bounces=1,
            bounces_player_a=1,
            bounces_player_b=0,
        )
        json_file = tmp_path / "report.json"
        save_report_json(m, json_file)

        assert json_file.exists()
        data = json.loads(json_file.read_text(encoding="utf-8"))
        assert data["total_frames"] == 60
        assert data["total_bounces"] == 1
