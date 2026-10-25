"""TTNet dataset loader.

Reads annotation files from lab.osai.ai dataset:
  - ball_markup.json     : {frame_id: {x, y}} — ball center in pixels
                           (-1, -1) means the ball is not visible in that frame.
  - events_markup.json   : {frame_id: event}  — "bounce" | "net" | "empty_event"
  - segmentation_masks/  : RGB PNG images (320×128) per annotated frame

Full-resolution frames (1920×1080) are NOT required for Step 1.
They will be used in Step 3 (ball detection / tracking).

Typical dataset layout (one game clip)::

    <game_root>/
        ball_markup.json
        events_markup.json
        segmentation_masks/
            <frame_id>.png
            ...
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

#: Standard table tennis table dimensions in millimetres (ITF rules).
TABLE_WIDTH_MM: int = 1525   # across the net
TABLE_LENGTH_MM: int = 2740  # along the net

#: Full-HD frame resolution used by the TTNet dataset cameras.
FRAME_WIDTH: int = 1920
FRAME_HEIGHT: int = 1080

#: Resolution of the segmentation mask images.
MASK_WIDTH: int = 320
MASK_HEIGHT: int = 128

#: Event label constants.
EVENT_BOUNCE = "bounce"
EVENT_NET = "net"
EVENT_EMPTY = "empty_event"
ALL_EVENTS = frozenset({EVENT_BOUNCE, EVENT_NET, EVENT_EMPTY})

# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class BallPosition:
    """Ball position for a single frame.

    Attributes:
        frame_id: Frame index (0-based integer from the dataset).
        x:        Horizontal pixel coordinate (in FRAME_WIDTH × FRAME_HEIGHT space).
                  -1 means the ball is not visible (occluded / out of frame).
        y:        Vertical pixel coordinate. -1 means not visible.
    """

    frame_id: int
    x: int
    y: int

    @property
    def visible(self) -> bool:
        """Return True when the ball is visible (coordinates ≥ 0)."""
        return self.x >= 0 and self.y >= 0

    @property
    def xy(self) -> tuple[int, int]:
        """Return (x, y) tuple."""
        return (self.x, self.y)

    def as_array(self) -> np.ndarray:
        """Return ball position as a float32 NumPy array [x, y]."""
        return np.array([self.x, self.y], dtype=np.float32)


@dataclass
class EventAnnotation:
    """Game event annotation for a single frame.

    Attributes:
        frame_id: Frame index.
        event:    One of "bounce", "net", or "empty_event".
    """

    frame_id: int
    event: str

    @property
    def is_bounce(self) -> bool:
        return self.event == EVENT_BOUNCE

    @property
    def is_net(self) -> bool:
        return self.event == EVENT_NET

    @property
    def is_empty(self) -> bool:
        return self.event == EVENT_EMPTY


@dataclass
class FrameAnnotation:
    """Combined annotation for a single frame (ball position + optional event).

    Attributes:
        frame_id:    Frame index.
        ball:        Ball position (may indicate invisible ball).
        event:       Event annotation if this frame is annotated; else None.
        mask_path:   Path to the segmentation mask PNG, if it exists.
    """

    frame_id: int
    ball: BallPosition | None = None
    event: EventAnnotation | None = None
    mask_path: Path | None = None

    @property
    def has_ball(self) -> bool:
        return self.ball is not None and self.ball.visible

    @property
    def has_event(self) -> bool:
        return self.event is not None

    @property
    def has_mask(self) -> bool:
        return self.mask_path is not None


@dataclass
class GameClip:
    """In-memory representation of one game clip annotation.

    Attributes:
        root:          Root directory of the clip.
        ball_data:     Mapping from frame_id → BallPosition.
        event_data:    Mapping from frame_id → EventAnnotation.
        mask_dir:      Directory containing segmentation mask images.
    """

    root: Path
    ball_data: dict[int, BallPosition] = field(default_factory=dict)
    event_data: dict[int, EventAnnotation] = field(default_factory=dict)
    mask_dir: Path | None = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def frame_ids(self) -> list[int]:
        """Sorted list of all frame IDs that have at least one annotation."""
        return sorted(set(self.ball_data) | set(self.event_data))

    @property
    def n_frames(self) -> int:
        return len(self.frame_ids)

    @property
    def bounce_frames(self) -> list[int]:
        """Sorted frame IDs where a bounce was annotated."""
        return sorted(fid for fid, ev in self.event_data.items() if ev.is_bounce)

    @property
    def net_frames(self) -> list[int]:
        """Sorted frame IDs where a net touch was annotated."""
        return sorted(fid for fid, ev in self.event_data.items() if ev.is_net)

    @property
    def visible_ball_positions(self) -> list[BallPosition]:
        """All BallPosition instances where the ball is visible."""
        return [bp for bp in self.ball_data.values() if bp.visible]

    @property
    def trajectory_array(self) -> np.ndarray:
        """Shape (N, 3) array of [frame_id, x, y] for visible ball frames.

        Rows are sorted by frame_id.
        """
        positions = sorted(self.visible_ball_positions, key=lambda bp: bp.frame_id)
        if not positions:
            return np.empty((0, 3), dtype=np.float32)
        return np.array([[p.frame_id, p.x, p.y] for p in positions], dtype=np.float32)

    # ── Frame-level access ────────────────────────────────────────────────────

    def get_frame(self, frame_id: int) -> FrameAnnotation:
        """Return a combined FrameAnnotation for the given frame ID."""
        mask_path: Path | None = None
        if self.mask_dir is not None:
            candidate = self.mask_dir / f"{frame_id}.png"
            if candidate.exists():
                mask_path = candidate
        return FrameAnnotation(
            frame_id=frame_id,
            ball=self.ball_data.get(frame_id),
            event=self.event_data.get(frame_id),
            mask_path=mask_path,
        )

    def iter_frames(self) -> Iterator[FrameAnnotation]:
        """Iterate over all annotated frames in order."""
        for fid in self.frame_ids:
            yield self.get_frame(fid)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a concise statistics dictionary."""
        event_counts: dict[str, int] = {EVENT_BOUNCE: 0, EVENT_NET: 0, EVENT_EMPTY: 0}
        for ev in self.event_data.values():
            event_counts[ev.event] = event_counts.get(ev.event, 0) + 1

        visible = self.visible_ball_positions
        xs = [p.x for p in visible]
        ys = [p.y for p in visible]

        mask_count = sum(
            1
            for fid in self.frame_ids
            if self.mask_dir and (self.mask_dir / f"{fid}.png").exists()
        )

        return {
            "clip_root": str(self.root),
            "total_annotated_frames": self.n_frames,
            "ball_annotated_frames": len(self.ball_data),
            "visible_ball_frames": len(visible),
            "invisible_ball_frames": len(self.ball_data) - len(visible),
            "event_annotated_frames": len(self.event_data),
            "events": event_counts,
            "mask_frames": mask_count,
            "ball_x_range": (min(xs), max(xs)) if xs else (None, None),
            "ball_y_range": (min(ys), max(ys)) if ys else (None, None),
        }


# ── Loader ────────────────────────────────────────────────────────────────────


def load_game_clip(root: str | Path) -> GameClip:
    """Load all annotation files from a single game clip directory.

    Args:
        root: Path to the clip directory containing:
              - ``ball_markup.json``
              - ``events_markup.json``
              - ``segmentation_masks/`` (optional)

    Returns:
        A :class:`GameClip` instance with all annotations loaded into memory.

    Raises:
        FileNotFoundError: If neither ball_markup.json nor events_markup.json
                           exists in *root*.
    """
    root = Path(root)
    ball_path = root / "ball_markup.json"
    events_path = root / "events_markup.json"
    mask_dir = root / "segmentation_masks"

    if not ball_path.exists() and not events_path.exists():
        raise FileNotFoundError(
            f"No annotation files found in '{root}'. "
            "Expected ball_markup.json and/or events_markup.json."
        )

    # ── Load ball positions ────────────────────────────────────────────────
    ball_data: dict[int, BallPosition] = {}
    if ball_path.exists():
        raw: dict[str, dict] = json.loads(ball_path.read_text(encoding="utf-8"))
        for fid_str, pos in raw.items():
            fid = int(fid_str)
            ball_data[fid] = BallPosition(frame_id=fid, x=int(pos["x"]), y=int(pos["y"]))

    # ── Load events ────────────────────────────────────────────────────────
    event_data: dict[int, EventAnnotation] = {}
    if events_path.exists():
        raw_events: dict[str, str] = json.loads(events_path.read_text(encoding="utf-8"))
        for fid_str, event in raw_events.items():
            fid = int(fid_str)
            event_data[fid] = EventAnnotation(frame_id=fid, event=event)

    return GameClip(
        root=root,
        ball_data=ball_data,
        event_data=event_data,
        mask_dir=mask_dir if mask_dir.exists() else None,
    )


def load_dataset(data_root: str | Path) -> list[GameClip]:
    """Load all game clips found under *data_root*.

    Recursively searches *data_root* for directories that contain
    ``ball_markup.json`` or ``events_markup.json``.

    Args:
        data_root: Root of the dataset (e.g. ``data/raw``).

    Returns:
        List of :class:`GameClip` instances, one per discovered clip directory.
    """
    data_root = Path(data_root)
    clips: list[GameClip] = []

    # A clip directory is one that directly contains ball_markup.json or events_markup.json.
    candidate_dirs: set[Path] = set()
    for marker in ("ball_markup.json", "events_markup.json"):
        for p in data_root.rglob(marker):
            candidate_dirs.add(p.parent)

    for clip_dir in sorted(candidate_dirs):
        try:
            clips.append(load_game_clip(clip_dir))
        except FileNotFoundError:
            pass  # Should not happen given the discovery logic above

    return clips
