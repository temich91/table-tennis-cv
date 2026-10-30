"""Step 2 CLI script: Table Detection, Homography, and Bird's Eye View (BEV) Analytics.

Usage::

    python scripts/run_homography.py [--mask MASK_PATH] [--output-dir OUTPUT_DIR]

Default:
  - Mask: ``data/raw/segmentation_masks/1278.png``
  - Output dir: ``outputs/homography``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow importing ttcv without installing in editable mode
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ttcv.data.dataset import load_game_clip
from ttcv.geometry import (
    TABLE_LENGTH_MM,
    TABLE_WIDTH_MM,
    TableHomography,
    detect_table_corners_from_mask,
)
from ttcv.geometry.visualizer import (
    draw_camera_table_overlay,
    draw_table_bev,
    plot_bev_heatmap,
    plot_bev_trajectories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Table Detection & Homography (Bird's Eye View) analysis."
    )
    parser.add_argument(
        "--mask",
        default="data/raw/segmentation_masks/1278.png",
        help="Path to segmentation mask (default: data/raw/segmentation_masks/1278.png)",
    )
    parser.add_argument(
        "--data-root",
        default="data/raw",
        help="Root directory with ball_markup.json and events_markup.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/homography",
        help="Directory to save output plots (default: outputs/homography)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_path = Path(args.mask)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" PingPongVision — Step 2: Table Detection & Homography (BEV)")
    print("=" * 65)

    # ── 1. Detect Table Corners from Mask ────────────────────────────────────
    print(f"\n[1/5] Detecting table corners from: {mask_path}...")
    corners = detect_table_corners_from_mask(mask_path, target_resolution=(1920, 1080))
    print(f"      Mask dimensions : {corners.mask_size}")
    print(f"      Target frame    : {corners.target_size}")
    print(f"      Far-Left  (TL)  : ({corners.top_left[0]:.1f}, {corners.top_left[1]:.1f}) px")
    print(f"      Far-Right (TR)  : ({corners.top_right[0]:.1f}, {corners.top_right[1]:.1f}) px")
    print(f"      Near-Right (BR) : ({corners.bottom_right[0]:.1f}, {corners.bottom_right[1]:.1f}) px")
    print(f"      Near-Left (BL)  : ({corners.bottom_left[0]:.1f}, {corners.bottom_left[1]:.1f}) px")

    # ── 2. Compute Homography ────────────────────────────────────────────────
    print("\n[2/5] Computing Homography matrix...")
    homography = TableHomography(corners)
    print("      Homography (Camera -> Table mm):\n", homography.H_table)

    # ── 3. Load Ball & Event Data ────────────────────────────────────────────
    print(f"\n[3/5] Loading annotations from {data_root}...")
    clip = load_game_clip(data_root)
    print(f"      Loaded {clip.n_frames:,} total annotated frames.")

    # Extract bounce points
    bounce_fids = clip.bounce_frames
    bounce_cam: list[list[float]] = []
    for fid in bounce_fids:
        if fid in clip.ball_data and clip.ball_data[fid].visible:
            bp = clip.ball_data[fid]
            bounce_cam.append([float(bp.x), float(bp.y)])

    bounce_cam_arr = np.array(bounce_cam, dtype=np.float32)
    print(f"      Found {len(bounce_cam_arr)} bounce events with visible ball coordinates.")

    # ── 4. Project & Analyze Bounces ─────────────────────────────────────────
    print("\n[4/5] Projecting bounces to 2D Table metric space (Bird's Eye View)...")
    bounces_table = homography.project_to_table(bounce_cam_arr)

    # Classification breakdown
    inside_count = 0
    side_counts = {"player_a": 0, "player_b": 0, "net": 0}
    depth_counts = {"deep": 0, "medium": 0, "short": 0, "outside": 0}
    lateral_counts = {"left": 0, "right": 0, "outside": 0}

    for pt in bounces_table:
        tp = homography.classify_camera_point(
            float(homography.project_table_to_camera(pt.reshape(1, 2))[0, 0]),
            float(homography.project_table_to_camera(pt.reshape(1, 2))[0, 1]),
        )
        if tp.is_on_table(margin_mm=50.0):
            inside_count += 1
            side_counts[tp.side.value] += 1
            depth_counts[tp.depth_zone.value] += 1
            lateral_counts[tp.lateral_zone.value] += 1

    total = len(bounces_table)
    print(f"      Bounces on table : {inside_count} / {total} ({inside_count / total * 100:.1f}%)")
    print(f"      • Player A half  : {side_counts['player_a']} ({side_counts['player_a'] / inside_count * 100:.1f}%)")
    print(f"      • Player B half  : {side_counts['player_b']} ({side_counts['player_b'] / inside_count * 100:.1f}%)")
    print(f"      • Depth split    : Short={depth_counts['short']} | Medium={depth_counts['medium']} | Deep={depth_counts['deep']}")
    print(f"      • Lateral split  : Left (Far)={lateral_counts['left']} | Right (Near)={lateral_counts['right']}")

    # ── 5. Generate & Save Visualizations ────────────────────────────────────
    print("\n[5/5] Generating visualizations...")

    # A. 2D Table Diagram
    table_img = draw_table_bev()
    table_path = output_dir / "bev_table_diagram.png"
    cv2.imwrite(str(table_path), cv2.cvtColor(table_img, cv2.COLOR_RGB2BGR))
    print(f"      [OK] Saved: {table_path}")

    # B. Bird's Eye View Bounce Heatmap
    fig_heat = plot_bev_heatmap(homography, bounce_cam_arr)
    heat_path = output_dir / "bev_bounce_heatmap.png"
    fig_heat.savefig(heat_path, dpi=150, bbox_inches="tight")
    plt.close(fig_heat)
    print(f"      [OK] Saved: {heat_path}")

    # C. Bird's Eye View Trajectory + Bounces
    traj_arr = clip.trajectory_array
    fig_traj = plot_bev_trajectories(homography, traj_arr, bounce_cam_arr)
    traj_path = output_dir / "bev_trajectories.png"
    fig_traj.savefig(traj_path, dpi=150, bbox_inches="tight")
    plt.close(fig_traj)
    print(f"      [OK] Saved: {traj_path}")

    # D. Camera View Table Overlay
    mask_bgr = cv2.imread(str(mask_path))
    # Resize mask to target resolution for camera-view overlay visualization
    mask_full = cv2.resize(mask_bgr, (1920, 1080), interpolation=cv2.INTER_NEAREST)
    overlay = draw_camera_table_overlay(mask_full, homography)
    overlay_path = output_dir / "camera_table_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    print(f"      [OK] Saved: {overlay_path}")

    print("\n" + "=" * 65)
    print(f" Step 2 Complete! All artefacts saved to: {output_dir.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
