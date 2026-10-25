"""Step 1 EDA script — run from the project root.

Usage::

    python scripts/run_eda.py [DATA_ROOT] [--output-dir OUTPUT_DIR]

Default DATA_ROOT is ``data/raw``.
Plots are saved to OUTPUT_DIR (default: ``outputs/eda``).

Example::

    python scripts/run_eda.py data/raw --output-dir outputs/eda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless execution

import matplotlib.pyplot as plt

from ttcv.data.dataset import load_dataset
from ttcv.data.eda import (
    plot_ball_heatmap,
    plot_event_distribution,
    plot_gap_distribution,
    plot_trajectory,
    plot_visibility,
    print_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA on the TTNet dataset.")
    parser.add_argument(
        "data_root",
        nargs="?",
        default="data/raw",
        help="Root directory of the dataset (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/eda",
        help="Directory to save EDA plots (default: outputs/eda)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {data_root.resolve()}")
    clips = load_dataset(data_root)

    if not clips:
        print(f"[ERROR] No game clips found under '{data_root}'. Check the path.")
        sys.exit(1)

    print(f"Found {len(clips)} clip(s).\n")

    for i, clip in enumerate(clips):
        tag = f"clip_{i:02d}"
        print_summary(clip)

        # ── Trajectory ──────────────────────────────────────────────────────
        fig = plot_trajectory(clip, title=f"Ball Trajectory — {tag}")
        path = output_dir / f"{tag}_trajectory.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        # ── Event distribution ───────────────────────────────────────────────
        fig = plot_event_distribution(clip, title=f"Event Distribution — {tag}")
        path = output_dir / f"{tag}_events.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        # ── Visibility timeline ──────────────────────────────────────────────
        fig = plot_visibility(clip, title=f"Ball Visibility — {tag}")
        path = output_dir / f"{tag}_visibility.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        # ── Ball position heatmap ────────────────────────────────────────────
        fig = plot_ball_heatmap(clip, title=f"Ball Heatmap — {tag}")
        path = output_dir / f"{tag}_heatmap.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        # ── Invisibility gap distribution ────────────────────────────────────
        fig = plot_gap_distribution(clip, title=f"Invisibility Gaps — {tag}")
        path = output_dir / f"{tag}_gaps.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        print()

    print(f"EDA complete. All plots saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
