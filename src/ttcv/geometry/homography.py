"""Вычисление гомографии и перспективное преобразование для настольного тенниса.

Предоставляет:
  - Каноническое упорядочивание 4 угловых точек четырехугольника перспективы.
  - Автоматическую детекцию углов стола по маскам семантической сегментации.
  - Класс TableHomography для двунаправленного проецирования между координатами камеры,
    метрическими координатами стола (в мм и метрах) и пикселями холста вида сверху (BEV).
  - Перспективное выравнивание (warp) кадров камеры в проекцию сверху (BEV).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ttcv.geometry.table import (
    TABLE_LENGTH_MM,
    TABLE_WIDTH_MM,
    TableDimensions,
    TablePoint,
)


def order_quadrilateral_points(pts: np.ndarray) -> np.ndarray:
    """Упорядочивание 4 точек четырехугольника в каноническом порядке по часовой стрелке.

    Канонический порядок:
      0: Top-Left     (Дальний левый при виде сбоку,   X=0,    Y=0)
      1: Top-Right    (Дальний правый при виде сбоку,  X=2740, Y=0)
      2: Bottom-Right (Ближний правый при виде сбоку,  X=2740, Y=1525)
      3: Bottom-Left  (Ближний левый при виде сбоку,   X=0,    Y=1525)

    Аргументы:
        pts: Массив формы (4, 2) или (4, 1, 2), содержащий 4 координаты (x, y).

    Возвращает:
        Массив float32 формы (4, 2) в каноническом порядке [TL, TR, BR, BL].
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)

    # Сортировка всех точек по оси Y на 2 верхние и 2 нижние
    by_y = pts[np.argsort(pts[:, 1])]
    top = by_y[:2]
    bottom = by_y[2:]

    # Верхние две точки сортируются по X -> top-left, top-right
    top = top[np.argsort(top[:, 0])]
    tl, tr = top[0], top[1]

    # Нижние две точки сортируются по X -> bottom-left, bottom-right
    bottom = bottom[np.argsort(bottom[:, 0])]
    bl, br = bottom[0], bottom[1]

    return np.array([tl, tr, br, bl], dtype=np.float32)


@dataclass
class TableCorners:
    """Обнаруженные или заданные 4 угла стола в координатах камеры.

    Атрибуты:
        corners_camera: Массив формы (4, 2) [TL, TR, BR, BL] в пикселях камеры.
        corners_mask:   Массив формы (4, 2) [TL, TR, BR, BL] в координатах маски (если обнаружены).
        target_size:    (width, height) целевого разрешения камеры.
        mask_size:      (width, height) входной маски (если обнаружена).
        contour_area:   Площадь контура в пространстве маски.
    """

    corners_camera: np.ndarray
    corners_mask: np.ndarray | None = None
    target_size: tuple[int, int] = (1920, 1080)
    mask_size: tuple[int, int] | None = None
    contour_area: float | None = None

    @property
    def top_left(self) -> tuple[float, float]:
        """Дальний левый угол (задняя линия Игрока A, дальняя боковая линия)."""
        return (float(self.corners_camera[0, 0]), float(self.corners_camera[0, 1]))

    @property
    def top_right(self) -> tuple[float, float]:
        """Дальний правый угол (задняя линия Игрока B, дальняя боковая линия)."""
        return (float(self.corners_camera[1, 0]), float(self.corners_camera[1, 1]))

    @property
    def bottom_right(self) -> tuple[float, float]:
        """Ближний правый угол (задняя линия Игрока B, ближняя боковая линия)."""
        return (float(self.corners_camera[2, 0]), float(self.corners_camera[2, 1]))

    @property
    def bottom_left(self) -> tuple[float, float]:
        """Ближний левый угол (задняя линия Игрока A, ближняя боковая линия)."""
        return (float(self.corners_camera[3, 0]), float(self.corners_camera[3, 1]))

    def as_array(self) -> np.ndarray:
        """Возвращает углы камеры как массив float32 формы (4, 2)."""
        return np.asarray(self.corners_camera, dtype=np.float32)

    def to_dict(self) -> dict:
        """Возвращает представление в виде словаря."""
        return {
            "top_left": list(self.top_left),
            "top_right": list(self.top_right),
            "bottom_right": list(self.bottom_right),
            "bottom_left": list(self.bottom_left),
            "target_size": list(self.target_size),
        }


def detect_table_corners_from_mask(
    mask: str | Path | np.ndarray,
    target_resolution: tuple[int, int] = (1920, 1080),
    table_color_bgr: tuple[int, int, int] = (0, 0, 255),
) -> TableCorners:
    """Детекция 4 углов стола по маске семантической сегментации.

    В датасете TTNet поверхность стола размечена чистым красным цветом (BGR [0, 0, 255]).

    Аргументы:
        mask: Путь к PNG-файлу маски или загруженный массив NumPy (BGR или бинарный).
        target_resolution: (ширина, высота) видеокадра/камеры (по умолчанию: 1920x1080).
        table_color_bgr: Цвет класса стола в BGR (по умолчанию: [0, 0, 255] = красный).

    Возвращает:
        :class:`TableCorners`, содержащий канонические 4 угла, масштабированные под `target_resolution`.

    Исключения:
        FileNotFoundError: Если файл маски не найден.
        ValueError: Если в маске не обнаружен контур стола.
    """
    if isinstance(mask, (str, Path)):
        p = Path(mask)
        if not p.exists():
            raise FileNotFoundError(f"Mask file not found: {p}")
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Failed to read image from {p}")
    else:
        img = np.asarray(mask)

    if img.ndim == 2:
        # Бинарная маска / маска в градациях серого
        table_mask = (img > 0).astype(np.uint8)
    elif img.shape[2] == 3:
        # 3-канальная маска BGR
        table_mask = np.all(img == table_color_bgr, axis=-1).astype(np.uint8)
        if table_mask.sum() == 0:
            # Запасной вариант: проверяем доминирование красного канала
            table_mask = (
                (img[:, :, 2] > 200) & (img[:, :, 0] < 50) & (img[:, :, 1] < 50)
            ).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported mask shape: {img.shape}")

    if table_mask.sum() == 0:
        raise ValueError("No table pixels matching the table class were found in the mask.")

    mask_h, mask_w = img.shape[:2]
    cnts, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("No contours found in the table mask.")

    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))

    # Адаптивная аппроксимация полигона для поиска 4 вершин
    poly: np.ndarray | None = None
    peri = cv2.arcLength(c, True)
    for eps in np.linspace(0.01, 0.12, 60):
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4:
            poly = approx.reshape(4, 2)
            break

    # Запасной вариант, если approxPolyDP не вернул ровно 4 вершины
    if poly is None:
        hull = cv2.convexHull(c).reshape(-1, 2)
        if len(hull) >= 4:
            # Выбор 4 экстремальных точек
            s = hull.sum(axis=1)
            diff = np.diff(hull, axis=1).squeeze()
            tl = hull[np.argmin(s)]
            br = hull[np.argmax(s)]
            tr = hull[np.argmin(diff)]
            bl = hull[np.argmax(diff)]
            poly = np.array([tl, tr, br, bl], dtype=np.float32)
        else:
            rect = cv2.minAreaRect(c)
            poly = cv2.boxPoints(rect)

    ordered_mask = order_quadrilateral_points(poly)

    # Масштабирование под целевое разрешение камеры
    sx = target_resolution[0] / mask_w
    sy = target_resolution[1] / mask_h

    ordered_cam = ordered_mask.copy()
    ordered_cam[:, 0] *= sx
    ordered_cam[:, 1] *= sy

    return TableCorners(
        corners_camera=ordered_cam,
        corners_mask=ordered_mask,
        target_size=target_resolution,
        mask_size=(mask_w, mask_h),
        contour_area=area,
    )


class TableHomography:
    """Двунаправленная гомография между пикселями камеры, метрическим столом (мм) и холстом BEV.

    Позволяет:
      - Проецировать 2D-положения мяча из кадра камеры в реальные координаты стола (мм/м).
      - Проецировать 2D-координаты стола обратно в кадр камеры.
      - Трансформировать кадры камеры в проекцию сверху вниз (Bird's Eye View / BEV).
      - Классифицировать отскоки от стола (попадание/аут, сторона игрока, зоны глубины и ширины).
    """

    def __init__(
        self,
        corners: TableCorners | np.ndarray,
        bev_canvas_size: tuple[int, int] = (1100, 650),
        bev_margin: int = 50,
    ) -> None:
        """Инициализация TableHomography.

        Аргументы:
            corners: Экземпляр :class:`TableCorners` или массив (4, 2) углов камеры.
            bev_canvas_size: (ширина, высота) в пикселях холста вида сверху (BEV).
            bev_margin: Отступ в пикселях вокруг стола на холсте BEV.
        """
        if isinstance(corners, TableCorners):
            self.corners = corners
        else:
            arr = order_quadrilateral_points(corners)
            self.corners = TableCorners(corners_camera=arr)

        self.bev_size = bev_canvas_size
        self.bev_margin = bev_margin

        src_pts = self.corners.corners_camera.astype(np.float32)

        # ── 1. Метрическая гомография (пиксели камеры -> мм стола) ───────────
        # Канонические углы: [TL, TR, BR, BL]
        dst_mm = TableDimensions.canonical_corners_mm().astype(np.float32)
        self.H_table: np.ndarray = cv2.getPerspectiveTransform(src_pts, dst_mm)
        self.H_table_inv: np.ndarray = np.linalg.inv(self.H_table)

        # ── 2. Гомография холста (пиксели камеры -> пиксели холста BEV) ──────
        w_canv, h_canv = self.bev_size
        m = self.bev_margin
        dst_bev = np.array(
            [
                [m, m],                    # TL
                [w_canv - m, m],           # TR
                [w_canv - m, h_canv - m],  # BR
                [m, h_canv - m],           # BL
            ],
            dtype=np.float32,
        )
        self.H_bev: np.ndarray = cv2.getPerspectiveTransform(src_pts, dst_bev)
        self.H_bev_inv: np.ndarray = np.linalg.inv(self.H_bev)

    @classmethod
    def from_mask(
        cls,
        mask: str | Path | np.ndarray,
        target_resolution: tuple[int, int] = (1920, 1080),
        bev_canvas_size: tuple[int, int] = (1100, 650),
        bev_margin: int = 50,
    ) -> TableHomography:
        """Фабричный метод: создание TableHomography напрямую по маске сегментации."""
        corners = detect_table_corners_from_mask(mask, target_resolution=target_resolution)
        return cls(corners, bev_canvas_size=bev_canvas_size, bev_margin=bev_margin)

    # ── Проекция точек ────────────────────────────────────────────────────────

    def project_to_table(self, points_cam: np.ndarray) -> np.ndarray:
        """Проекция координат пикселей камеры в координаты поверхности стола (мм).

        Аргументы:
            points_cam: Массив формы (N, 2) или (2,) с [x, y] в пикселях камеры.

        Возвращает:
            Массив формы (N, 2) с [x_mm, y_mm] на поверхности стола.
        """
        pts = np.asarray(points_cam, dtype=np.float32)
        orig_shape = pts.shape
        pts = pts.reshape(-1, 1, 2)
        pts_table = cv2.perspectiveTransform(pts, self.H_table)
        return pts_table.reshape(orig_shape)

    def project_to_bev(self, points_cam: np.ndarray) -> np.ndarray:
        """Проекция координат пикселей камеры на холст вида сверху (BEV).

        Аргументы:
            points_cam: Массив формы (N, 2) или (2,) с [x, y] в пикселях камеры.

        Возвращает:
            Массив формы (N, 2) с [x_bev, y_bev] в пикселях холста.
        """
        pts = np.asarray(points_cam, dtype=np.float32)
        orig_shape = pts.shape
        pts = pts.reshape(-1, 1, 2)
        pts_bev = cv2.perspectiveTransform(pts, self.H_bev)
        return pts_bev.reshape(orig_shape)

    def project_table_to_camera(self, points_table: np.ndarray) -> np.ndarray:
        """Обратная проекция координат стола (мм) в пиксели камеры.

        Аргументы:
            points_table: Массив формы (N, 2) или (2,) с [x_mm, y_mm].

        Возвращает:
            Массив формы (N, 2) с [x_cam, y_cam] в пикселях камеры.
        """
        pts = np.asarray(points_table, dtype=np.float32)
        orig_shape = pts.shape
        pts = pts.reshape(-1, 1, 2)
        pts_cam = cv2.perspectiveTransform(pts, self.H_table_inv)
        return pts_cam.reshape(orig_shape)

    def project_bev_to_camera(self, points_bev: np.ndarray) -> np.ndarray:
        """Обратная проекция пикселей холста BEV в пиксели камеры."""
        pts = np.asarray(points_bev, dtype=np.float32)
        orig_shape = pts.shape
        pts = pts.reshape(-1, 1, 2)
        pts_cam = cv2.perspectiveTransform(pts, self.H_bev_inv)
        return pts_cam.reshape(orig_shape)

    # ── Трансформация изображений (Warping) ───────────────────────────────────

    def warp_to_bev(self, camera_frame: np.ndarray) -> np.ndarray:
        """Перспективное преобразование кадра камеры в вид сверху (Bird's Eye View).

        Аргументы:
            camera_frame: Изображение (H, W, C) или (H, W), соответствующее target_resolution.

        Возвращает:
            Трансформированное изображение формы (bev_size[1], bev_size[0], C).
        """
        w, h = self.bev_size
        return cv2.warpPerspective(camera_frame, self.H_bev, (w, h))

    # ── Анализ точек ──────────────────────────────────────────────────────────

    def classify_camera_point(self, x_cam: float, y_cam: float) -> TablePoint:
        """Классификация координаты камеры путем проецирования в пространство стола."""
        pt_table = self.project_to_table(np.array([x_cam, y_cam]))
        return TablePoint(x_mm=float(pt_table[0]), y_mm=float(pt_table[1]))

    def is_on_table(
        self, x_cam: float, y_cam: float, margin_mm: float = 0.0
    ) -> bool:
        """Проверка, попадает ли пиксель камеры на поверхность стола."""
        tp = self.classify_camera_point(x_cam, y_cam)
        return tp.is_on_table(margin_mm=margin_mm)
