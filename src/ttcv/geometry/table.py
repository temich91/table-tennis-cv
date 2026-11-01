"""Геометрия стола для настольного тенниса и спецификации ITTF.

Официальные правила ITTF (Международная федерация настольного тенниса):
  - Длина стола: 2,74 м (2740 мм)
  - Ширина стола: 1,525 м (1525 мм)
  - Высота игровой поверхности над полом: 760 мм
  - Высота сетки: 152,5 мм
  - Выступ сетки за пределы стола: 152,5 мм
  - Полная длина сетки: 1525 + 2 * 152,5 = 1830 мм
  - Белая линия разметки по периметру: ширина 20 мм
  - Центральная линия подачи (для парной игры): ширина 3 мм

Система координат:
  - Ось X: вдоль длины стола, от 0 мм (задняя линия Игрока A) до 2740 мм (задняя линия Игрока B).
    Сетка расположена на X = 1370 мм.
  - Ось Y: поперек ширины стола, от 0 мм (дальняя боковая линия при виде сбоку)
    до 1525 мм (ближняя боковая линия).
    Центральная линия расположена на Y = 762,5 мм.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# ── Стандартные размеры ITTF (в миллиметрах) ──────────────────────────────────

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
    """Половина стола, разделенная сеткой."""

    PLAYER_A = "player_a"  # Левая половина: 0 <= X < 1370 мм
    PLAYER_B = "player_b"  # Правая половина: 1370 < X <= 2740 мм
    NET = "net"            # Линия сетки: X ≈ 1370 мм


class DepthZone(str, Enum):
    """Зона глубины относительно задней линии и сетки.

    Делится на 3 равные трети для каждой половины (каждая треть ≈ 456,7 мм):
      - DEEP: ближе к задней линии игрока (защитные / мощные удары)
      - MEDIUM: средняя зона
      - SHORT: ближе к сетке (укороченные удары / короткие приемы)
    """

    DEEP = "deep"
    MEDIUM = "medium"
    SHORT = "short"
    OUTSIDE = "outside"


class LateralZone(str, Enum):
    """Зона стола по ширине (разделенная центральной линией)."""

    LEFT = "left"        # Y < 762,5 мм (дальняя боковая линия при виде сбоку)
    RIGHT = "right"      # Y >= 762,5 мм (ближняя боковая линия при виде сбоку)
    OUTSIDE = "outside"


@dataclass(frozen=True)
class TablePoint:
    """Точка на 2D поверхности стола, выраженная в миллиметрах.

    Атрибуты:
        x_mm: Координата вдоль длины стола (0 = задняя линия Игрока A, 2740 = задняя линия Игрока B).
        y_mm: Координата вдоль ширины стола (0 = дальняя боковая линия, 1525 = ближняя боковая линия).
    """

    x_mm: float
    y_mm: float

    @property
    def x_m(self) -> float:
        """Координата вдоль длины стола в метрах."""
        return self.x_mm / 1000.0

    @property
    def y_m(self) -> float:
        """Координата вдоль ширины стола в метрах."""
        return self.y_mm / 1000.0

    def as_array(self) -> np.ndarray:
        """Возвращает [x_mm, y_mm] как массив float32."""
        return np.array([self.x_mm, self.y_mm], dtype=np.float32)

    def is_on_table(self, margin_mm: float = 0.0) -> bool:
        """Возвращает True, если точка лежит в пределах границ стола с учетом погрешности."""
        return (
            -margin_mm <= self.x_mm <= TABLE_LENGTH_MM + margin_mm
            and -margin_mm <= self.y_mm <= TABLE_WIDTH_MM + margin_mm
        )

    @property
    def side(self) -> TableSide:
        """Возвращает половину стола, к которой относится данная точка."""
        if abs(self.x_mm - NET_X_MM) < 20.0:
            return TableSide.NET
        return TableSide.PLAYER_A if self.x_mm < NET_X_MM else TableSide.PLAYER_B

    @property
    def lateral_zone(self) -> LateralZone:
        """Возвращает левую или правую боковую зону."""
        if not self.is_on_table(margin_mm=50.0):
            return LateralZone.OUTSIDE
        return LateralZone.LEFT if self.y_mm < CENTER_LINE_Y_MM else LateralZone.RIGHT

    @property
    def depth_zone(self) -> DepthZone:
        """Возвращает зону глубины: deep (глубокая), medium (средняя) или short (короткая)."""
        if not self.is_on_table(margin_mm=50.0):
            return DepthZone.OUTSIDE

        half_length = TABLE_LENGTH_MM / 2.0  # 1370 мм
        third = half_length / 3.0            # ~456.7 мм

        # Расстояние от ближайшей задней линии
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
        """Полная строка-дескриптор зоны, например 'player_a_deep_left'."""
        if not self.is_on_table(margin_mm=50.0):
            return "outside"
        return f"{self.side.value}_{self.depth_zone.value}_{self.lateral_zone.value}"


class TableDimensions:
    """Вспомогательный класс геометрии стола и канонических 2D-координат."""

    length_mm: float = TABLE_LENGTH_MM
    width_mm: float = TABLE_WIDTH_MM
    net_x_mm: float = NET_X_MM
    center_line_y_mm: float = CENTER_LINE_Y_MM

    @classmethod
    def canonical_corners_mm(cls) -> np.ndarray:
        """Возвращает 4 канонических угла стола в миллиметрах.

        Порядок: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
        (Дальний левый, дальний правый, ближний правый, ближний левый при виде сбоку)
        Форма: (4, 2)
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
        """Возвращает канонические углы стола в метрах."""
        return cls.canonical_corners_mm() / 1000.0

    @classmethod
    def net_segment_mm(cls) -> tuple[tuple[float, float], tuple[float, float]]:
        """Возвращает конечные точки отрезка сетки поперек стола, включая свесы."""
        return (
            (cls.net_x_mm, -NET_OVERHANG_MM),
            (cls.net_x_mm, cls.width_mm + NET_OVERHANG_MM),
        )

    @classmethod
    def center_line_segment_mm(cls) -> tuple[tuple[float, float], tuple[float, float]]:
        """Возвращает конечные точки центральной линии, делящей левую и правую половины."""
        return ((0.0, cls.center_line_y_mm), (cls.length_mm, cls.center_line_y_mm))

    @classmethod
    def classify_point(cls, x_mm: float, y_mm: float) -> TablePoint:
        """Создание и классификация объекта TablePoint."""
        return TablePoint(x_mm=float(x_mm), y_mm=float(y_mm))
