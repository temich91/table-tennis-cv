"""Подпакет геометрии для компьютерного зрения в настольном теннисе.

Содержит спецификации размеров стола (стандарты ITTF), детекцию углов по маскам
сегментации, перспективное преобразование (гомографию) в вид сверху (Bird's Eye View / BEV)
и модуль визуализации 2D стола.
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
