"""Rally analytics and physical metrics extraction.

Extracts:
  - Total rally duration and shot count.
  - Bounce counts and spatial distribution (Player A vs Player B, deep vs short).
  - Ball velocity profiles and speed distribution in km/h.
  - Shot segmentation (interval, speed, and target zone per stroke).
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
    """Information about a single stroke / shot during a rally.

    Attributes:
        shot_id: Index of the shot in the rally (0, 1, 2...).
        start_frame: Beginning frame of the shot.
        end_frame: Ending frame of the shot (at subsequent bounce / hit).
        duration_sec: Flight time in seconds.
        bounce_frame: Frame where ball bounced on receiver's table half.
        hitter: TableSide (Player A or Player B).
        receiver: TableSide.
        bounce_xy_mm: (x, y) coordinates on table in mm at bounce.
        bounce_zone: Detailed zone descriptor (e.g. 'player_b_deep_left').
        max_speed_kmh: Peak velocity reached during the shot.
        mean_speed_kmh: Average flight velocity in km/h.
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
    """Aggregated match and rally analytics.

    Attributes:
        total_frames: Total processed frames.
        duration_sec: Duration in seconds.
        fps: Video frame rate.
        total_bounces: Number of detected bounces.
        bounces_player_a: Bounces on Player A's half.
        bounces_player_b: Bounces on Player B's half.
        depth_distribution: Counts per depth zone ({'deep': N, 'medium': N, 'short': N}).
        lateral_distribution: Counts per lateral zone ({'left': N, 'right': N}).
        max_speed_kmh: Peak ball velocity recorded.
        mean_speed_kmh: Average ball velocity.
        speed_series: List of (time_sec, speed_kmh) tuples for plotting.
        shots: Segmented shots.
        interpolation_rate: Percentage of frames reconstructed by tracker.
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
        """Return a structured dictionary for JSON / dashboard reporting."""
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
    """Extracts game analytics and physics from tracked ball trajectories."""

    def __init__(
        self,
        homography: TableHomography | None = None,
        fps: float = 120.0,
    ) -> None:
        """Initialize RallyAnalyzer.

        Args:
            homography: Optional TableHomography instance.
            fps: Frame rate of the video sequence.
        """
        self.homography = homography
        self.fps = fps

    def analyze_trajectory(
        self,
        track_points: list[TrackPoint],
        annotated_bounce_frames: list[int] | None = None,
    ) -> RallyMetrics:
        """Analyze a complete sequence of TrackPoints.

        Args:
            track_points: Output history from BallTracker.
            annotated_bounce_frames: Optional list of known bounce frame indices.

        Returns:
            Computed :class:`RallyMetrics`.
        """
        if not track_points:
            return RallyMetrics(total_frames=0, duration_sec=0.0, fps=self.fps)

        total_frames = len(track_points)
        min_fid = track_points[0].frame_id
        max_fid = track_points[-1].frame_id
        duration_sec = (max_fid - min_fid + 1) / self.fps

        # 1. Speeds
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

        # 2. Bounces: use provided or detect from vertical velocity
        if annotated_bounce_frames is not None:
            bounce_fids = set(annotated_bounce_frames)
        else:
            bounce_fids = set(self._detect_bounces(track_points))

        # 3. Analyze Bounces on Table
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

        # 4. Segment Shots
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
        """Detect bounce frames where vertical velocity flips from downward to upward."""
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
        """Group trajectory points into individual shots divided by bounces."""
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
