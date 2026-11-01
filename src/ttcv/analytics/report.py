"""Генератор отчетов о матче и розыгрышах."""

from __future__ import annotations

import json
from pathlib import Path

from ttcv.analytics.metrics import RallyMetrics


def format_markdown_report(metrics: RallyMetrics, title: str = "PingPongVision Match Report") -> str:
    """Форматирование RallyMetrics в структурированный Markdown-отчет о матче."""
    d = metrics.summary_dict()

    md = f"""# 🏓 {title}

## 📊 Executive Summary / Сводка

| Metric / Метрика | Value / Значение |
|---|---|
| **Rally Duration** | `{d['duration_sec']} s` ({d['total_frames']} frames @ {d['fps']} FPS) |
| **Total Strokes / Shots** | `{d['total_shots']}` |
| **Total Bounces** | `{d['total_bounces']}` |
| **Max Ball Speed** | `{d['max_speed_kmh']} km/h` |
| **Average Ball Speed** | `{d['mean_speed_kmh']} km/h` |
| **Tracker Interpolation** | `{d['interpolation_rate_pct']}%` (Kalman reconstructed frames) |

---

## 🎯 Spatial Distribution (Table Bounces)

* **Player A half:** `{d['bounces_player_a']}` bounces
* **Player B half:** `{d['bounces_player_b']}` bounces

### Depth Zones
* **Deep (Baseline):** `{d['depth_distribution'].get('deep', 0)}`
* **Medium (Midcourt):** `{d['depth_distribution'].get('medium', 0)}`
* **Short (Net):** `{d['depth_distribution'].get('short', 0)}`

### Lateral Zones
* **Left (Far sideline):** `{d['lateral_distribution'].get('left', 0)}`
* **Right (Near sideline):** `{d['lateral_distribution'].get('right', 0)}`

---

## ⚡ Shot-by-Shot Breakdown

| Shot # | Hitter → Receiver | Duration | Max Speed | Mean Speed | Target Zone |
|---|---|---|---|---|---|
"""
    for s in metrics.shots:
        md += f"| **#{s.shot_id + 1}** | {s.hitter.value.upper()} → {s.receiver.value.upper()} | `{s.duration_sec}s` | `{s.max_speed_kmh} km/h` | `{s.mean_speed_kmh} km/h` | `{s.bounce_zone}` |\n"

    return md


def save_report_json(metrics: RallyMetrics, path: str | Path) -> None:
    """Сохранение полных метрик в файл формата JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics.summary_dict(), indent=2), encoding="utf-8")
