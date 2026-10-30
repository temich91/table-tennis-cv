"""Homography computation and perspective transformation for table tennis.

Provides:
  - Canonical 4-corner ordering for perspective quadrilaterals.
  - Automatic table corner detection from segmentation masks.
  - TableHomography class for bi-directional projection between camera coordinates,
    metric table coordinates (mm / meters), and Bird's Eye View (BEV) canvas pixels.
  - BEV warping of camera frames.
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
    """Order 4 points of a quadrilateral into canonical clockwise order.

    Canonical order:
      0: Top-Left     (Far-Left in side view,   X=0,    Y=0)
      1: Top-Right    (Far-Right in side view,  X=2740, Y=0)
      2: Bottom-Right (Near-Right in side view, X=2740, Y=1525)
      3: Bottom-Left  (Near-Left in side view,  X=0,    Y=1525)

    Args:
        pts: Array of shape (4, 2) or (4, 1, 2) containing 4 (x, y) coordinates.

    Returns:
        float32 array of shape (4, 2) in canonical order [TL, TR, BR, BL].
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)

    # Sort all points by Y coordinate into top 2 and bottom 2
    by_y = pts[np.argsort(pts[:, 1])]
    top = by_y[:2]
    bottom = by_y[2:]

    # Top two points sorted by X -> top-left, top-right
    top = top[np.argsort(top[:, 0])]
    tl, tr = top[0], top[1]

    # Bottom two points sorted by X -> bottom-left, bottom-right
    bottom = bottom[np.argsort(bottom[:, 0])]
    bl, br = bottom[0], bottom[1]

    return np.array([tl, tr, br, bl], dtype=np.float32)


@dataclass
class TableCorners:
    """Detected or specified 4 corners of the table in camera view.

    Attributes:
        corners_camera: Shape (4, 2) array [TL, TR, BR, BL] in camera pixel coordinates.
        corners_mask:   Shape (4, 2) array [TL, TR, BR, BL] in mask coordinates (if detected).
        target_size:    (width, height) of the target camera resolution.
        mask_size:      (width, height) of the input mask (if detected).
        contour_area:   Contour area in mask space.
    """

    corners_camera: np.ndarray
    corners_mask: np.ndarray | None = None
    target_size: tuple[int, int] = (1920, 1080)
    mask_size: tuple[int, int] | None = None
    contour_area: float | None = None

    @property
    def top_left(self) -> tuple[float, float]:
        """Far-Left corner (Player A baseline, far sideline)."""
        return (float(self.corners_camera[0, 0]), float(self.corners_camera[0, 1]))

    @property
    def top_right(self) -> tuple[float, float]:
        """Far-Right corner (Player B baseline, far sideline)."""
        return (float(self.corners_camera[1, 0]), float(self.corners_camera[1, 1]))

    @property
    def bottom_right(self) -> tuple[float, float]:
        """Near-Right corner (Player B baseline, near sideline)."""
        return (float(self.corners_camera[2, 0]), float(self.corners_camera[2, 1]))

    @property
    def bottom_left(self) -> tuple[float, float]:
        """Near-Left corner (Player A baseline, near sideline)."""
        return (float(self.corners_camera[3, 0]), float(self.corners_camera[3, 1]))

    def as_array(self) -> np.ndarray:
        """Return camera corners as float32 array of shape (4, 2)."""
        return np.asarray(self.corners_camera, dtype=np.float32)

    def to_dict(self) -> dict:
        """Return dict representation."""
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
    """Detect the 4 corners of the table from a semantic segmentation mask.

    In the TTNet dataset, the table surface is annotated as pure Red (BGR [0, 0, 255]).

    Args:
        mask: Path to the mask PNG file or a loaded NumPy array (BGR or binary).
        target_resolution: (width, height) of the video / camera frame (default: 1920x1080).
        table_color_bgr: BGR color of the table class (default: [0, 0, 255] = Red).

    Returns:
        :class:`TableCorners` containing the canonical 4 corners scaled to `target_resolution`.

    Raises:
        FileNotFoundError: If the mask file does not exist.
        ValueError: If no table contour is found in the mask.
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
        # Grayscale / binary mask
        table_mask = (img > 0).astype(np.uint8)
    elif img.shape[2] == 3:
        # 3-channel BGR mask
        table_mask = np.all(img == table_color_bgr, axis=-1).astype(np.uint8)
        if table_mask.sum() == 0:
            # Fallback: check red channel dominant
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

    # Adaptive polygon approximation to find 4 corners
    poly: np.ndarray | None = None
    peri = cv2.arcLength(c, True)
    for eps in np.linspace(0.01, 0.12, 60):
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4:
            poly = approx.reshape(4, 2)
            break

    # Fallback if approxPolyDP didn't yield exactly 4 vertices
    if poly is None:
        hull = cv2.convexHull(c).reshape(-1, 2)
        if len(hull) >= 4:
            # Pick 4 extreme points
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

    # Scale to target camera resolution
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
    """Bi-directional homography between camera pixels, metric table (mm), and BEV canvas.

    Enables:
      - Projecting 2D ball positions from camera frame to real-world table coordinates (mm/m).
      - Projecting 2D table coordinates back onto the camera frame.
      - Warping camera frames to a top-down Bird's Eye View (BEV) image.
      - Classifying table hits (in/out, player side, depth and lateral zones).
    """

    def __init__(
        self,
        corners: TableCorners | np.ndarray,
        bev_canvas_size: tuple[int, int] = (1100, 650),
        bev_margin: int = 50,
    ) -> None:
        """Initialize TableHomography.

        Args:
            corners: Either a :class:`TableCorners` instance or a (4, 2) array of camera corners.
            bev_canvas_size: (width, height) in pixels of the Bird's Eye View canvas.
            bev_margin: Margin in pixels around the table on the BEV canvas.
        """
        if isinstance(corners, TableCorners):
            self.corners = corners
        else:
            arr = order_quadrilateral_points(corners)
            self.corners = TableCorners(corners_camera=arr)

        self.bev_size = bev_canvas_size
        self.bev_margin = bev_margin

        src_pts = self.corners.corners_camera.astype(np.float32)

        # ── 1. Metric Homography (Camera px -> Table mm) ─────────────────────
        # Canonical corners: [TL, TR, BR, BL]
        dst_mm = TableDimensions.canonical_corners_mm().astype(np.float32)
        self.H_table: np.ndarray = cv2.getPerspectiveTransform(src_pts, dst_mm)
        self.H_table_inv: np.ndarray = np.linalg.inv(self.H_table)

        # ── 2. Canvas Homography (Camera px -> BEV Canvas px) ────────────────
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
        """Factory method: create TableHomography directly from a segmentation mask."""
        corners = detect_table_corners_from_mask(mask, target_resolution=target_resolution)
        return cls(corners, bev_canvas_size=bev_canvas_size, bev_margin=bev_margin)

    # ── Point Projections ─────────────────────────────────────────────────────

    def project_to_table(self, points_cam: np.ndarray) -> np.ndarray:
        """Project camera pixel coordinates to table surface coordinates (mm).

        Args:
            points_cam: Array of shape (N, 2) or (2,) with [x, y] in camera pixels.

        Returns:
            Array of shape (N, 2) with [x_mm, y_mm] on table surface.
        """
        pts = np.asarray(points_cam, dtype=np.float32)
        orig_shape = pts.shape
        pts = pts.reshape(-1, 1, 2)
        pts_table = cv2.perspectiveTransform(pts, self.H_table)
        return pts_table.reshape(orig_shape)

    def project_to_bev(self, points_cam: np.ndarray) -> np.ndarray:
        """Project camera pixel coordinates to Bird's Eye View canvas pixels.

        Args:
            points_cam: Array of shape (N, 2) or (2,) with [x, y] in camera pixels.

        Returns:
            Array of shape (N, 2) with [x_bev, y_bev] canvas pixels.
        """
        pts = np.asarray(points_cam, dtype=np.float32)
        orig_shape = pts.shape
        pts = pts.reshape(-1, 1, 2)
        pts_bev = cv2.perspectiveTransform(pts, self.H_bev)
        return pts_bev.reshape(orig_shape)

    def project_table_to_camera(self, points_table: np.ndarray) -> np.ndarray:
        """Project table coordinates (mm) back to camera pixel coordinates.

        Args:
            points_table: Array of shape (N, 2) or (2,) with [x_mm, y_mm].

        Returns:
            Array of shape (N, 2) with [x_cam, y_cam] in camera pixels.
        """
        pts = np.asarray(points_table, dtype=np.float32)
        orig_shape = pts.shape
        pts = pts.reshape(-1, 1, 2)
        pts_cam = cv2.perspectiveTransform(pts, self.H_table_inv)
        return pts_cam.reshape(orig_shape)

    def project_bev_to_camera(self, points_bev: np.ndarray) -> np.ndarray:
        """Project BEV canvas pixel coordinates back to camera pixel coordinates."""
        pts = np.asarray(points_bev, dtype=np.float32)
        orig_shape = pts.shape
        pts = pts.reshape(-1, 1, 2)
        pts_cam = cv2.perspectiveTransform(pts, self.H_bev_inv)
        return pts_cam.reshape(orig_shape)

    # ── Image Warping ─────────────────────────────────────────────────────────

    def warp_to_bev(self, camera_frame: np.ndarray) -> np.ndarray:
        """Warp a full camera frame to Bird's Eye View.

        Args:
            camera_frame: (H, W, C) or (H, W) image array matching target_resolution.

        Returns:
            Warped image array of shape (bev_size[1], bev_size[0], C).
        """
        w, h = self.bev_size
        return cv2.warpPerspective(camera_frame, self.H_bev, (w, h))

    # ── Point Analysis ────────────────────────────────────────────────────────

    def classify_camera_point(self, x_cam: float, y_cam: float) -> TablePoint:
        """Classify a camera coordinate by projecting it to table space."""
        pt_table = self.project_to_table(np.array([x_cam, y_cam]))
        return TablePoint(x_mm=float(pt_table[0]), y_mm=float(pt_table[1]))

    def is_on_table(
        self, x_cam: float, y_cam: float, margin_mm: float = 0.0
    ) -> bool:
        """Check whether a camera pixel lies on the table surface."""
        tp = self.classify_camera_point(x_cam, y_cam)
        return tp.is_on_table(margin_mm=margin_mm)
