"""Geometry sub-package for table tennis computer vision.

Contains table dimensions (ITTF standards), corner detection from segmentation masks,
perspective transformation (homography) to Bird's Eye View (BEV), and 2D court visualization.
"""

from ttcv.geometry.homography import (
    TableCorners,
    TableHomography,
    detect_table_corners_from_mask,
    order_quadrilateral_points,
)
from ttcv.geometry.table import (
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
    TableDimensions,
    TablePoint,
    TableSide,
)

__all__ = [
    "TABLE_LENGTH_MM",
    "TABLE_WIDTH_MM",
    "NET_HEIGHT_MM",
    "TABLE_HEIGHT_MM",
    "NET_OVERHANG_MM",
    "NET_TOTAL_LENGTH_MM",
    "BORDER_LINE_WIDTH_MM",
    "CENTER_LINE_WIDTH_MM",
    "TableSide",
    "LateralZone",
    "DepthZone",
    "TablePoint",
    "TableDimensions",
    "TableCorners",
    "TableHomography",
    "detect_table_corners_from_mask",
    "order_quadrilateral_points",
]
