"""Тесты для ttcv.data.eda — смоук-тесты (проверка вызова функций без ошибок)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ttcv.data.dataset import load_game_clip, EVENT_BOUNCE, EVENT_NET, EVENT_EMPTY
from ttcv.data.eda import (
    plot_ball_heatmap,
    plot_event_distribution,
    plot_gap_distribution,
    plot_trajectory,
    plot_visibility,
    print_summary,
)


@pytest.fixture
def clip(tmp_path: Path):
    ball_data = {str(i): {"x": i * 10, "y": 500 + i} for i in range(1, 30)}
    ball_data["15"] = {"x": -1, "y": -1}  # пропуск видимости
    ball_data["16"] = {"x": -1, "y": -1}
    events_data = {
        "5": EVENT_EMPTY,
        "10": EVENT_BOUNCE,
        "20": EVENT_NET,
        "25": EVENT_BOUNCE,
    }
    (tmp_path / "ball_markup.json").write_text(json.dumps(ball_data), encoding="utf-8")
    (tmp_path / "events_markup.json").write_text(json.dumps(events_data), encoding="utf-8")
    return load_game_clip(tmp_path)


def test_print_summary(clip, capsys):
    print_summary(clip)
    out = capsys.readouterr().out
    assert "bounce" in out.lower()


def test_plot_trajectory_returns_figure(clip):
    fig = plot_trajectory(clip)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_event_distribution_returns_figure(clip):
    fig = plot_event_distribution(clip)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_visibility_returns_figure(clip):
    fig = plot_visibility(clip)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_ball_heatmap_returns_figure(clip):
    fig = plot_ball_heatmap(clip)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_gap_distribution_returns_figure(clip):
    fig = plot_gap_distribution(clip)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
