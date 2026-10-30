"""Visualizations for table geometry and Bird's Eye View (BEV).

Functions:
  - draw_table_bev: Draw a clean, accurate 2D ITTF table diagram as an RGB image.
  - plot_bev_heatmap: Plot a 2D density heatmap of bounces on the table surface.
  - plot_bev_trajectories: Plot projected ball trajectory and bounce points on the table.
  - draw_camera_table_overlay: Draw the table polygon, net, and corners on a camera frame.
"""

from __future__ import annotations

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ttcv.geometry.homography import TableHomography
from ttcv.geometry.table import (
    CENTER_LINE_Y_MM,
    NET_X_MM,
    TABLE_LENGTH_MM,
    TABLE_WIDTH_MM,
    TableDimensions,
    TableSide,
)

# ── Color Palette ─────────────────────────────────────────────────────────────

TABLE_BLUE = (25, 60, 110)       # Tournament table surface (RGB)
TABLE_BORDER_WHITE = (250, 250, 250)
NET_COLOR = (230, 230, 230)
NET_POST_COLOR = (80, 80, 80)
CANVAS_BG = (240, 243, 246)


def draw_table_bev(
    canvas_size: tuple[int, int] = (1100, 650),
    margin: int = 50,
) -> np.ndarray:
    """Render a standard 2D table tennis court (Bird's Eye View) as an RGB image.

    Args:
        canvas_size: (width, height) of the canvas image.
        margin: Margin in pixels around the table.

    Returns:
        RGB uint8 array of shape (height, width, 3).
    """
    w, h = canvas_size
    img = np.full((h, w, 3), CANVAS_BG, dtype=np.uint8)

    # Table rectangle in canvas space
    x0, y0 = margin, margin
    x1, y1 = w - margin, h - margin
    table_w = x1 - x0
    table_h = y1 - y0

    # 1. Table surface
    cv2.rectangle(img, (x0, y0), (x1, y1), TABLE_BLUE, -1)

    # 2. White perimeter line (20 mm)
    line_thick = max(2, int(table_w * 20.0 / TABLE_LENGTH_MM))
    cv2.rectangle(img, (x0, y0), (x1, y1), TABLE_BORDER_WHITE, line_thick)

    # 3. Center service line (for doubles, 3 mm)
    cy = int(y0 + table_h / 2.0)
    cv2.line(img, (x0, cy), (x1, cy), TABLE_BORDER_WHITE, max(1, line_thick // 3))

    # 4. Net line across the middle (with overhangs)
    net_x = int(x0 + table_w / 2.0)
    overhang_px = int(table_h * 152.5 / TABLE_WIDTH_MM)
    net_y0 = max(5, y0 - overhang_px)
    net_y1 = min(h - 5, y1 + overhang_px)

    # Net mesh / line
    cv2.line(img, (net_x, net_y0), (net_x, net_y1), NET_COLOR, 3)

    # Net posts
    cv2.circle(img, (net_x, net_y0), 5, NET_POST_COLOR, -1)
    cv2.circle(img, (net_x, net_y1), 5, NET_POST_COLOR, -1)

    # 5. Side labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "PLAYER A", (x0 + 40, y0 - 15), font, 0.6, (80, 90, 105), 2)
    cv2.putText(img, "PLAYER B", (x1 - 160, y0 - 15), font, 0.6, (80, 90, 105), 2)
    cv2.putText(img, "NET", (net_x - 18, net_y1 + 22), font, 0.5, (100, 110, 125), 1)

    return img


def plot_bev_heatmap(
    homography: TableHomography,
    bounce_points_cam: np.ndarray,
    *,
    figsize: tuple[float, float] = (12, 7),
    title: str = "Bounce Heatmap on Table (Bird's Eye View)",
) -> plt.Figure:
    """Plot a 2D density heatmap of ball bounces on the table surface.

    Args:
        homography: Fitted :class:`TableHomography`.
        bounce_points_cam: Array of shape (N, 2) of bounce coordinates in camera pixels.
        figsize: Matplotlib figure size.
        title: Figure title.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    pts = np.asarray(bounce_points_cam, dtype=np.float32).reshape(-1, 2)
    pts_table = homography.project_to_table(pts)  # in mm

    # Filter points near or on table
    valid = (
        (pts_table[:, 0] >= -100)
        & (pts_table[:, 0] <= TABLE_LENGTH_MM + 100)
        & (pts_table[:, 1] >= -100)
        & (pts_table[:, 1] <= TABLE_WIDTH_MM + 100)
    )
    pts_table = pts_table[valid]

    # Convert to meters for clearer axes
    xs_m = pts_table[:, 0] / 1000.0
    ys_m = pts_table[:, 1] / 1000.0

    fig, ax = plt.subplots(figsize=figsize)

    # Table background
    table_rect = mpatches.Rectangle(
        (0.0, 0.0),
        TABLE_LENGTH_MM / 1000.0,
        TABLE_WIDTH_MM / 1000.0,
        linewidth=2.5,
        edgecolor="white",
        facecolor="#1e3d59",
        zorder=1,
    )
    ax.add_patch(table_rect)

    # Net line
    net_m = NET_X_MM / 1000.0
    ax.axvline(net_m, color="white", linewidth=2.0, linestyle="--", alpha=0.9, zorder=2)

    # Center service line
    cy_m = CENTER_LINE_Y_MM / 1000.0
    ax.axhline(cy_m, color="white", linewidth=1.0, alpha=0.6, zorder=2)

    # 2D KDE density contour / heatmap
    if len(xs_m) >= 5:
        sns.kdeplot(
            x=xs_m,
            y=ys_m,
            ax=ax,
            cmap="hot",
            fill=True,
            alpha=0.6,
            levels=12,
            thresh=0.05,
            zorder=3,
        )

    # Scatter points for individual bounces
    player_a_mask = xs_m < net_m
    ax.scatter(
        xs_m[player_a_mask],
        ys_m[player_a_mask],
        s=25,
        color="#00ffff",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
        label=f"Player A half ({player_a_mask.sum()})",
        zorder=4,
    )
    ax.scatter(
        xs_m[~player_a_mask],
        ys_m[~player_a_mask],
        s=25,
        color="#ff7f50",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
        label=f"Player B half ({(~player_a_mask).sum()})",
        zorder=4,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Table Length (meters)", fontsize=11)
    ax.set_ylabel("Table Width (meters)", fontsize=11)
    ax.set_xlim(-0.2, (TABLE_LENGTH_MM / 1000.0) + 0.2)
    ax.set_ylim((TABLE_WIDTH_MM / 1000.0) + 0.2, -0.2)  # image convention: y downwards
    ax.set_facecolor("#f0f3f6")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_aspect("equal")

    # Annotate sides
    ax.text(
        0.35,
        -0.08,
        "← PLAYER A SIDE",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color="#1e3d59",
    )
    ax.text(
        0.65,
        -0.08,
        "PLAYER B SIDE →",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color="#1e3d59",
    )

    fig.tight_layout()
    return fig


def plot_bev_trajectories(
    homography: TableHomography,
    trajectory_cam: np.ndarray,
    bounce_points_cam: np.ndarray | None = None,
    *,
    figsize: tuple[float, float] = (12, 7),
    title: str = "Ball Trajectory on Table (Bird's Eye View)",
) -> plt.Figure:
    """Plot ball trajectory projected onto the 2D table surface.

    Args:
        homography: Fitted :class:`TableHomography`.
        trajectory_cam: Shape (N, 3) array [frame_id, x_cam, y_cam] or (N, 2) [x_cam, y_cam].
        bounce_points_cam: Optional shape (M, 2) array of bounce points in camera space.
        figsize: Figure size.
        title: Figure title.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    traj = np.asarray(trajectory_cam, dtype=np.float32)
    if traj.shape[1] == 3:
        pts_cam = traj[:, 1:3]
    else:
        pts_cam = traj

    pts_table = homography.project_to_table(pts_cam)  # in mm
    xs_m = pts_table[:, 0] / 1000.0
    ys_m = pts_table[:, 1] / 1000.0

    fig, ax = plt.subplots(figsize=figsize)

    # Table background
    table_rect = mpatches.Rectangle(
        (0.0, 0.0),
        TABLE_LENGTH_MM / 1000.0,
        TABLE_WIDTH_MM / 1000.0,
        linewidth=2.5,
        edgecolor="white",
        facecolor="#1e3d59",
        zorder=1,
    )
    ax.add_patch(table_rect)

    # Net and center lines
    ax.axvline(NET_X_MM / 1000.0, color="white", linewidth=2.0, linestyle="--", alpha=0.9, zorder=2)
    ax.axhline(CENTER_LINE_Y_MM / 1000.0, color="white", linewidth=1.0, alpha=0.6, zorder=2)

    # Draw trajectory line (filter out extreme jumps)
    valid = (xs_m >= -0.5) & (xs_m <= 3.24) & (ys_m >= -0.5) & (ys_m <= 2.0)
    ax.plot(
        xs_m[valid],
        ys_m[valid],
        color="#f1c40f",
        linewidth=0.7,
        alpha=0.6,
        label="Projected Trajectory",
        zorder=3,
    )

    # Draw bounces if provided
    if bounce_points_cam is not None and len(bounce_points_cam) > 0:
        bpts = homography.project_to_table(bounce_points_cam) / 1000.0
        ax.scatter(
            bpts[:, 0],
            bpts[:, 1],
            s=40,
            color="#e74c3c",
            edgecolor="white",
            linewidth=1.0,
            label=f"Bounces ({len(bpts)})",
            zorder=4,
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Table Length (meters)", fontsize=11)
    ax.set_ylabel("Table Width (meters)", fontsize=11)
    ax.set_xlim(-0.2, (TABLE_LENGTH_MM / 1000.0) + 0.2)
    ax.set_ylim((TABLE_WIDTH_MM / 1000.0) + 0.2, -0.2)
    ax.set_facecolor("#f0f3f6")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_aspect("equal")

    fig.tight_layout()
    return fig


def draw_camera_table_overlay(
    image: np.ndarray,
    homography: TableHomography,
    color_bgr: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    """Draw table perimeter, net line, and corners directly on a camera image.

    Args:
        image: Camera frame or mask as a NumPy array (BGR or RGB).
        homography: Fitted TableHomography.
        color_bgr: Color for the table outline.
        thickness: Line thickness.

    Returns:
        Annotated copy of the image.
    """
    annotated = image.copy()
    corners = homography.corners.corners_camera.astype(np.int32)

    # 1. Table perimeter polygon
    cv2.polylines(annotated, [corners], isClosed=True, color=color_bgr, thickness=thickness)

    # 2. Net line (endpoints in table mm projected to camera)
    net_start_mm = np.array([[NET_X_MM, 0.0]], dtype=np.float32)
    net_end_mm = np.array([[NET_X_MM, TABLE_WIDTH_MM]], dtype=np.float32)
    net_cam = homography.project_table_to_camera(
        np.vstack([net_start_mm, net_end_mm])
    ).astype(np.int32)
    cv2.line(
        annotated,
        tuple(net_cam[0]),
        tuple(net_cam[1]),
        (255, 255, 255),
        thickness + 1,
    )

    # 3. Center service line
    center_start_mm = np.array([[0.0, CENTER_LINE_Y_MM]], dtype=np.float32)
    center_end_mm = np.array([[TABLE_LENGTH_MM, CENTER_LINE_Y_MM]], dtype=np.float32)
    center_cam = homography.project_table_to_camera(
        np.vstack([center_start_mm, center_end_mm])
    ).astype(np.int32)
    cv2.line(
        annotated,
        tuple(center_cam[0]),
        tuple(center_cam[1]),
        (255, 255, 255),
        max(1, thickness // 2),
    )

    # 4. Corner circles with labels
    corner_labels = ["TL (Far-L)", "TR (Far-R)", "BR (Near-R)", "BL (Near-L)"]
    for i, (pt, label) in enumerate(zip(corners, corner_labels)):
        cv2.circle(annotated, tuple(pt), 7, (0, 0, 255), -1)
        cv2.circle(annotated, tuple(pt), 9, (255, 255, 255), 2)
        cv2.putText(
            annotated,
            label,
            (pt[0] - 30, pt[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

    return annotated
