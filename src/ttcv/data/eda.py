"""Exploratory Data Analysis helpers for the TTNet dataset.

Provides functions to:
  - Print a textual summary of a GameClip.
  - Plot the ball trajectory (pixel space).
  - Plot the event distribution.
  - Plot ball visibility gaps (frames where ball is invisible).
  - Load and display a segmentation mask.
  - Plot heatmap of ball positions over a schematic table silhouette.

All plotting functions return a ``matplotlib.figure.Figure`` so they can be
saved programmatically or displayed in a notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from ttcv.data.dataset import GameClip

from ttcv.data.dataset import (
    EVENT_BOUNCE,
    EVENT_EMPTY,
    EVENT_NET,
    FRAME_HEIGHT,
    FRAME_WIDTH,
)

# ── Color palette ─────────────────────────────────────────────────────────────

EVENT_COLORS: dict[str, str] = {
    EVENT_BOUNCE: "#e74c3c",
    EVENT_NET: "#3498db",
    EVENT_EMPTY: "#95a5a6",
}

# ── Text summary ──────────────────────────────────────────────────────────────


def print_summary(clip: "GameClip") -> None:
    """Print a formatted summary of the clip statistics to stdout."""
    s = clip.summary()
    print("=" * 60)
    print(f"Clip: {s['clip_root']}")
    print("=" * 60)
    print(f"  Annotated frames     : {s['total_annotated_frames']:,}")
    print(f"  Ball annotations     : {s['ball_annotated_frames']:,}")
    print(f"    • visible          : {s['visible_ball_frames']:,}")
    print(f"    • invisible (-1,-1): {s['invisible_ball_frames']:,}")
    print(f"  Event annotations    : {s['event_annotated_frames']:,}")
    print(f"    • bounce           : {s['events'][EVENT_BOUNCE]:,}")
    print(f"    • net              : {s['events'][EVENT_NET]:,}")
    print(f"    • empty_event      : {s['events'][EVENT_EMPTY]:,}")
    print(f"  Segmentation masks   : {s['mask_frames']:,}")
    print(f"  Ball X range (px)    : {s['ball_x_range']}")
    print(f"  Ball Y range (px)    : {s['ball_y_range']}")
    print("=" * 60)


# ── Trajectory plot ───────────────────────────────────────────────────────────


def plot_trajectory(
    clip: "GameClip",
    *,
    figsize: tuple[float, float] = (14, 6),
    title: str = "Ball Trajectory (pixel space)",
) -> plt.Figure:
    """Plot X and Y pixel coordinates of the ball over time.

    Visible frames are drawn as a continuous line; invisible frames are shown
    as grey shaded regions on the axes.

    Args:
        clip:    The game clip to visualise.
        figsize: Figure size in inches.
        title:   Figure super-title.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    traj = clip.trajectory_array  # shape (N, 3): [frame_id, x, y]
    all_fids = np.array(clip.frame_ids)

    # Identify invisible frames
    invisible_fids = sorted(
        fid for fid, bp in clip.ball_data.items() if not bp.visible
    )

    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, dim, label, color in [
        (ax_x, 1, "X (px)", "#2ecc71"),
        (ax_y, 2, "Y (px)", "#9b59b6"),
    ]:
        if traj.shape[0] > 0:
            ax.plot(traj[:, 0], traj[:, dim], color=color, linewidth=0.8, alpha=0.85)
            ax.scatter(traj[:, 0], traj[:, dim], s=3, color=color, alpha=0.5, zorder=3)

        # Shade invisible regions
        for fid in invisible_fids:
            ax.axvline(fid, color="grey", linewidth=0.4, alpha=0.3)

        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, FRAME_WIDTH if dim == 1 else FRAME_HEIGHT)

    ax_y.set_xlabel("Frame ID")

    # Add event markers on X axis
    _add_event_markers(ax_x, clip)
    _add_event_markers(ax_y, clip)

    fig.tight_layout()
    return fig


def _add_event_markers(ax: plt.Axes, clip: "GameClip") -> None:
    """Draw vertical lines for bounce/net events on *ax*."""
    for fid, ev in clip.event_data.items():
        if ev.event in (EVENT_BOUNCE, EVENT_NET):
            ax.axvline(
                fid,
                color=EVENT_COLORS[ev.event],
                linewidth=1.0,
                alpha=0.6,
                linestyle="--",
            )


# ── Event distribution ────────────────────────────────────────────────────────


def plot_event_distribution(
    clip: "GameClip",
    *,
    figsize: tuple[float, float] = (6, 4),
    title: str = "Event Distribution",
) -> plt.Figure:
    """Bar chart showing count of each event type.

    Args:
        clip:    The game clip to visualise.
        figsize: Figure size.
        title:   Plot title.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    s = clip.summary()["events"]
    labels = [EVENT_BOUNCE, EVENT_NET, EVENT_EMPTY]
    counts = [s.get(lbl, 0) for lbl in labels]
    colors = [EVENT_COLORS[lbl] for lbl in labels]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, padding=3, fontsize=11)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(counts) * 1.2 if counts else 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ── Visibility gap analysis ───────────────────────────────────────────────────


def plot_visibility(
    clip: "GameClip",
    *,
    figsize: tuple[float, float] = (14, 2.5),
    title: str = "Ball Visibility Timeline",
) -> plt.Figure:
    """Horizontal timeline showing when the ball is visible vs invisible.

    Args:
        clip:    The game clip.
        figsize: Figure size.
        title:   Plot title.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    fids = np.array(clip.frame_ids)
    visibility = np.array(
        [1 if (fid in clip.ball_data and clip.ball_data[fid].visible) else 0 for fid in fids]
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(fids, visibility, step="post", color="#2ecc71", alpha=0.7, label="Visible")
    ax.fill_between(
        fids, 1 - visibility, step="post", color="#e74c3c", alpha=0.4, label="Invisible"
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Frame ID")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Invisible", "Visible"])
    ax.legend(loc="upper right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# ── Segmentation mask display ─────────────────────────────────────────────────


def show_mask(
    clip: "GameClip",
    frame_id: int,
    *,
    figsize: tuple[float, float] = (8, 4),
) -> plt.Figure:
    """Display the segmentation mask for a specific frame.

    Args:
        clip:     The game clip.
        frame_id: Frame to display.
        figsize:  Figure size.

    Returns:
        A :class:`matplotlib.figure.Figure`.

    Raises:
        FileNotFoundError: If no mask exists for *frame_id*.
        ValueError:        If *clip* has no mask directory.
    """
    import cv2  # local import — only needed when actually called

    fa = clip.get_frame(frame_id)
    if fa.mask_path is None:
        raise FileNotFoundError(
            f"No segmentation mask found for frame {frame_id} in '{clip.mask_dir}'"
        )

    img_bgr = cv2.imread(str(fa.mask_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_rgb)
    ax.set_title(
        f"Segmentation mask — frame {frame_id}"
        + (f"  [{fa.event.event}]" if fa.has_event else ""),
        fontweight="bold",
    )
    ax.axis("off")
    fig.tight_layout()
    return fig


# ── Ball position heatmap ─────────────────────────────────────────────────────


def plot_ball_heatmap(
    clip: "GameClip",
    *,
    figsize: tuple[float, float] = (10, 5),
    bins: int = 40,
    title: str = "Ball Position Heatmap (pixel space)",
) -> plt.Figure:
    """2D heatmap of all visible ball positions in full-frame pixel space.

    Args:
        clip:    The game clip.
        figsize: Figure size.
        bins:    Number of histogram bins in each dimension.
        title:   Plot title.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    visible = clip.visible_ball_positions
    if not visible:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No visible ball positions", ha="center", va="center")
        return fig

    xs = [p.x for p in visible]
    ys = [p.y for p in visible]

    fig, ax = plt.subplots(figsize=figsize)
    h = ax.hist2d(
        xs,
        ys,
        bins=bins,
        range=[[0, FRAME_WIDTH], [0, FRAME_HEIGHT]],
        cmap="hot",
        density=True,
    )
    fig.colorbar(h[3], ax=ax, label="Density")

    # Mark bounce positions with dots
    bounce_fids = set(clip.bounce_frames)
    bx = [clip.ball_data[f].x for f in bounce_fids if f in clip.ball_data and clip.ball_data[f].visible]
    by = [clip.ball_data[f].y for f in bounce_fids if f in clip.ball_data and clip.ball_data[f].visible]
    if bx:
        ax.scatter(bx, by, s=20, color=EVENT_COLORS[EVENT_BOUNCE], alpha=0.8, label="Bounce", zorder=5)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_xlim(0, FRAME_WIDTH)
    ax.set_ylim(FRAME_HEIGHT, 0)  # Image convention: y increases downward
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# ── Consecutive gap analysis ──────────────────────────────────────────────────


def plot_gap_distribution(
    clip: "GameClip",
    *,
    figsize: tuple[float, float] = (8, 4),
    title: str = "Invisibility Gap Length Distribution",
) -> plt.Figure:
    """Histogram of consecutive invisible-ball segment lengths (in frames).

    Useful for understanding how long the ball is typically occluded/missing,
    which informs the choice of interpolation / Kalman prediction horizon.

    Args:
        clip:    The game clip.
        figsize: Figure size.
        title:   Plot title.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    fids = np.array(sorted(clip.ball_data.keys()))
    is_invisible = np.array([not clip.ball_data[f].visible for f in fids])

    # Find contiguous invisible segments
    gap_lengths: list[int] = []
    current_gap = 0
    for flag in is_invisible:
        if flag:
            current_gap += 1
        else:
            if current_gap > 0:
                gap_lengths.append(current_gap)
                current_gap = 0
    if current_gap > 0:
        gap_lengths.append(current_gap)

    fig, ax = plt.subplots(figsize=figsize)
    if gap_lengths:
        ax.hist(gap_lengths, bins=max(gap_lengths), color="#e67e22", edgecolor="white")
        ax.set_xlabel("Gap length (frames)")
        ax.set_ylabel("Count")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        # Statistics text
        ax.text(
            0.98,
            0.95,
            f"mean={np.mean(gap_lengths):.1f}  median={np.median(gap_lengths):.0f}  max={max(gap_lengths)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )
    else:
        ax.text(0.5, 0.5, "No invisible segments", ha="center", va="center")
    fig.tight_layout()
    return fig
