"""Table tennis table geometry and ITTF specifications.

Official ITTF (International Table Tennis Federation) rules:
  - Table length: 2.74 m (2740 mm)
  - Table width: 1.525 m (1525 mm)
  - Table surface height above floor: 760 mm
  - Net height: 152.5 mm
  - Net overhang (protrusion past table edges): 152.5 mm
  - Total net length: 1525 + 2 * 152.5 = 1830 mm
  - Perimeter white line: 20 mm wide
  - Center service line (for doubles): 3 mm wide

Coordinate System:
  - X axis: along the table length, 0 mm (Player A baseline) to 2740 mm (Player B baseline).
    Net is at X = 1370 mm.
  - Y axis: across table width, 0 mm (Far sideline as seen in side-view camera)
    to 1525 mm (Near sideline).
    Center line is at Y = 762.5 mm.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# ── ITTF Standard Dimensions (millimetres) ────────────────────────────────────

TABLE_LENGTH_MM: float = 2740.0
TABLE_WIDTH_MM: float = 1525.0
TABLE_HEIGHT_MM: float = 760.0
NET_HEIGHT_MM: float = 152.5
NET_OVERHANG_MM: float = 152.5
NET_TOTAL_LENGTH_MM: float = TABLE_WIDTH_MM + 2 * NET_OVERHANG_MM  # 1830.0 mm
BORDER_LINE_WIDTH_MM: float = 20.0
CENTER_LINE_WIDTH_MM: float = 3.0

NET_X_MM: float = TABLE_LENGTH_MM / 2.0  # 1370.0 mm
CENTER_LINE_Y_MM: float = TABLE_WIDTH_MM / 2.0  # 762.5 mm


class TableSide(str, Enum):
    """Side of the table divided by the net."""

    PLAYER_A = "player_a"  # Left half: 0 <= X < 1370 mm
    PLAYER_B = "player_b"  # Right half: 1370 < X <= 2740 mm
    NET = "net"            # At net line: X ≈ 1370 mm


class DepthZone(str, Enum):
    """Depth zone relative to baseline and net.

    Divided into 3 equal thirds per half (each third ≈ 456.7 mm):
      - DEEP: closest to player baseline (defensive / heavy shots)
      - MEDIUM: mid-table
      - SHORT: closest to net (drop shots / short returns)
    """

    DEEP = "deep"
    MEDIUM = "medium"
    SHORT = "short"
    OUTSIDE = "outside"


class LateralZone(str, Enum):
    """Lateral zone (divided by center service line)."""

    LEFT = "left"        # Y < 762.5 mm (far sideline in side-view)
    RIGHT = "right"      # Y >= 762.5 mm (near sideline in side-view)
    OUTSIDE = "outside"


@dataclass(frozen=True)
class TablePoint:
    """A point on the 2D table surface expressed in millimetres.

    Attributes:
        x_mm: Coordinate along length (0 = Player A baseline, 2740 = Player B baseline).
        y_mm: Coordinate along width (0 = far sideline, 1525 = near sideline).
    """

    x_mm: float
    y_mm: float

    @property
    def x_m(self) -> float:
        """Coordinate along length in meters."""
        return self.x_mm / 1000.0

    @property
    def y_m(self) -> float:
        """Coordinate along width in meters."""
        return self.y_mm / 1000.0

    def as_array(self) -> np.ndarray:
        """Return [x_mm, y_mm] as a float32 array."""
        return np.array([self.x_mm, self.y_mm], dtype=np.float32)

    def is_on_table(self, margin_mm: float = 0.0) -> bool:
        """Return True if point is within the table boundaries plus optional margin."""
        return (
            -margin_mm <= self.x_mm <= TABLE_LENGTH_MM + margin_mm
            and -margin_mm <= self.y_mm <= TABLE_WIDTH_MM + margin_mm
        )

    @property
    def side(self) -> TableSide:
        """Return which half of the table this point belongs to."""
        if abs(self.x_mm - NET_X_MM) < 20.0:
            return TableSide.NET
        return TableSide.PLAYER_A if self.x_mm < NET_X_MM else TableSide.PLAYER_B

    @property
    def lateral_zone(self) -> LateralZone:
        """Return left or right lateral zone."""
        if not self.is_on_table(margin_mm=50.0):
            return LateralZone.OUTSIDE
        return LateralZone.LEFT if self.y_mm < CENTER_LINE_Y_MM else LateralZone.RIGHT

    @property
    def depth_zone(self) -> DepthZone:
        """Return deep, medium, or short depth zone."""
        if not self.is_on_table(margin_mm=50.0):
            return DepthZone.OUTSIDE

        half_length = TABLE_LENGTH_MM / 2.0  # 1370 mm
        third = half_length / 3.0            # ~456.7 mm

        # Distance from nearest baseline
        if self.x_mm < NET_X_MM:
            dist_from_baseline = self.x_mm
        else:
            dist_from_baseline = TABLE_LENGTH_MM - self.x_mm

        if dist_from_baseline < third:
            return DepthZone.DEEP
        if dist_from_baseline < 2 * third:
            return DepthZone.MEDIUM
        return DepthZone.SHORT

    @property
    def zone_name(self) -> str:
        """Full descriptor string, e.g. 'player_a_deep_left'."""
        if not self.is_on_table(margin_mm=50.0):
            return "outside"
        return f"{self.side.value}_{self.depth_zone.value}_{self.lateral_zone.value}"


class TableDimensions:
    """Helper for table geometry and canonical 2D coordinates."""

    length_mm: float = TABLE_LENGTH_MM
    width_mm: float = TABLE_WIDTH_MM
    net_x_mm: float = NET_X_MM
    center_line_y_mm: float = CENTER_LINE_Y_MM

    @classmethod
    def canonical_corners_mm(cls) -> np.ndarray:
        """Return the 4 canonical table corners in millimetres.

        Order: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
        (Far-Left, Far-Right, Near-Right, Near-Left in side view)
        Shape: (4, 2)
        """
        return np.array(
            [
                [0.0, 0.0],
                [cls.length_mm, 0.0],
                [cls.length_mm, cls.width_mm],
                [0.0, cls.width_mm],
            ],
            dtype=np.float32,
        )

    @classmethod
    def canonical_corners_m(cls) -> np.ndarray:
        """Return canonical corners in meters."""
        return cls.canonical_corners_mm() / 1000.0

    @classmethod
    def net_segment_mm(cls) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return endpoints of the net line across the table, including overhangs."""
        return (
            (cls.net_x_mm, -NET_OVERHANG_MM),
            (cls.net_x_mm, cls.width_mm + NET_OVERHANG_MM),
        )

    @classmethod
    def center_line_segment_mm(cls) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return endpoints of the center line dividing left and right halves."""
        return ((0.0, cls.center_line_y_mm), (cls.length_mm, cls.center_line_y_mm))

    @classmethod
    def classify_point(cls, x_mm: float, y_mm: float) -> TablePoint:
        """Create and classify a TablePoint."""
        return TablePoint(x_mm=float(x_mm), y_mm=float(y_mm))
