"""Tests for ttcv.data.dataset."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ttcv.data.dataset import (
    BallPosition,
    EventAnnotation,
    GameClip,
    load_dataset,
    load_game_clip,
    EVENT_BOUNCE,
    EVENT_NET,
    EVENT_EMPTY,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_clip_dir(tmp_path: Path) -> Path:
    """Create a minimal temporary clip directory with sample annotations."""
    ball_data = {
        "10": {"x": 100, "y": 200},
        "11": {"x": 110, "y": 195},
        "12": {"x": -1, "y": -1},  # invisible
        "13": {"x": 130, "y": 185},
    }
    events_data = {
        "10": EVENT_EMPTY,
        "11": EVENT_BOUNCE,
        "13": EVENT_NET,
    }
    (tmp_path / "ball_markup.json").write_text(json.dumps(ball_data), encoding="utf-8")
    (tmp_path / "events_markup.json").write_text(json.dumps(events_data), encoding="utf-8")

    # Add a dummy segmentation mask (1px PNG)
    mask_dir = tmp_path / "segmentation_masks"
    mask_dir.mkdir()
    _write_minimal_png(mask_dir / "11.png")

    return tmp_path


def _write_minimal_png(path: Path) -> None:
    """Write a 1×1 black PNG to *path* without external dependencies."""
    import zlib
    import struct

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = b"\x00\x00\x00\x00"  # filter byte + RGB pixel (black)
    idat = png_chunk(b"IDAT", zlib.compress(raw))
    iend = png_chunk(b"IEND", b"")
    path.write_bytes(sig + ihdr + idat + iend)


# ── BallPosition ──────────────────────────────────────────────────────────────


class TestBallPosition:
    def test_visible_true(self):
        bp = BallPosition(frame_id=1, x=100, y=200)
        assert bp.visible is True

    def test_visible_false_when_minus_one(self):
        bp = BallPosition(frame_id=2, x=-1, y=-1)
        assert bp.visible is False

    def test_xy_property(self):
        bp = BallPosition(frame_id=3, x=50, y=75)
        assert bp.xy == (50, 75)

    def test_as_array(self):
        bp = BallPosition(frame_id=4, x=10, y=20)
        arr = bp.as_array()
        assert arr.shape == (2,)
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(arr, [10.0, 20.0])


# ── EventAnnotation ───────────────────────────────────────────────────────────


class TestEventAnnotation:
    def test_is_bounce(self):
        ev = EventAnnotation(frame_id=1, event=EVENT_BOUNCE)
        assert ev.is_bounce
        assert not ev.is_net
        assert not ev.is_empty

    def test_is_net(self):
        ev = EventAnnotation(frame_id=2, event=EVENT_NET)
        assert ev.is_net

    def test_is_empty(self):
        ev = EventAnnotation(frame_id=3, event=EVENT_EMPTY)
        assert ev.is_empty


# ── load_game_clip ────────────────────────────────────────────────────────────


class TestLoadGameClip:
    def test_loads_ball_data(self, sample_clip_dir: Path):
        clip = load_game_clip(sample_clip_dir)
        assert 10 in clip.ball_data
        assert clip.ball_data[10].x == 100
        assert clip.ball_data[10].y == 200

    def test_loads_invisible_frame(self, sample_clip_dir: Path):
        clip = load_game_clip(sample_clip_dir)
        assert clip.ball_data[12].visible is False

    def test_loads_event_data(self, sample_clip_dir: Path):
        clip = load_game_clip(sample_clip_dir)
        assert clip.event_data[11].event == EVENT_BOUNCE
        assert clip.event_data[13].event == EVENT_NET

    def test_mask_dir_detected(self, sample_clip_dir: Path):
        clip = load_game_clip(sample_clip_dir)
        assert clip.mask_dir is not None
        assert clip.mask_dir.exists()

    def test_raises_when_no_files(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_game_clip(tmp_path)


# ── GameClip properties ───────────────────────────────────────────────────────


class TestGameClip:
    def test_frame_ids_sorted(self, sample_clip_dir: Path):
        clip = load_game_clip(sample_clip_dir)
        fids = clip.frame_ids
        assert fids == sorted(fids)

    def test_n_frames(self, sample_clip_dir: Path):
        clip = load_game_clip(sample_clip_dir)
        # ball frames: 10, 11, 12, 13 | event frames: 10, 11, 13 → union = {10, 11, 12, 13}
        assert clip.n_frames == 4

    def test_bounce_frames(self, sample_clip_dir: Path):
        clip = load_game_clip(sample_clip_dir)
        assert clip.bounce_frames == [11]

    def test_net_frames(self, sample_clip_dir: Path):
        clip = load_game_clip(sample_clip_dir)
        assert clip.net_frames == [13]

    def test_visible_ball_positions(self, sample_clip_dir: Path):
        clip = load_game_clip(sample_clip_dir)
        visible = clip.visible_ball_positions
        assert len(visible) == 3  # frames 10, 11, 13

    def test_trajectory_array_shape(self, sample_clip_dir: Path):
        clip = load_game_clip(sample_clip_dir)
        traj = clip.trajectory_array
        assert traj.shape == (3, 3)  # 3 visible frames × [frame_id, x, y]

    def test_trajectory_array_sorted(self, sample_clip_dir: Path):
        clip = load_game_clip(sample_clip_dir)
        traj = clip.trajectory_array
        assert list(traj[:, 0]) == sorted(traj[:, 0])

    def test_get_frame_has_mask(self, sample_clip_dir: Path):
        clip = load_game_clip(sample_clip_dir)
        fa = clip.get_frame(11)
        assert fa.has_mask
        assert fa.mask_path.name == "11.png"

    def test_get_frame_no_mask_for_unlisted(self, sample_clip_dir: Path):
        clip = load_game_clip(sample_clip_dir)
        fa = clip.get_frame(10)
        assert not fa.has_mask  # only frame 11 has a mask file

    def test_summary_structure(self, sample_clip_dir: Path):
        clip = load_game_clip(sample_clip_dir)
        s = clip.summary()
        assert "total_annotated_frames" in s
        assert "events" in s
        assert s["events"][EVENT_BOUNCE] == 1
        assert s["events"][EVENT_NET] == 1
        assert s["visible_ball_frames"] == 3
        assert s["invisible_ball_frames"] == 1


# ── load_dataset ──────────────────────────────────────────────────────────────


class TestLoadDataset:
    def test_discovers_clips(self, sample_clip_dir: Path):
        # sample_clip_dir itself contains the annotation files
        clips = load_dataset(sample_clip_dir)
        assert len(clips) == 1

    def test_discovers_multiple_clips(self, tmp_path: Path):
        for name in ("game_01", "game_02"):
            d = tmp_path / name
            d.mkdir()
            (d / "ball_markup.json").write_text(
                json.dumps({"1": {"x": 10, "y": 20}}), encoding="utf-8"
            )
            (d / "events_markup.json").write_text(
                json.dumps({"1": EVENT_EMPTY}), encoding="utf-8"
            )
        clips = load_dataset(tmp_path)
        assert len(clips) == 2

    def test_empty_dir_returns_empty_list(self, tmp_path: Path):
        clips = load_dataset(tmp_path)
        assert clips == []
