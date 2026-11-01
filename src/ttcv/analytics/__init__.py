"""Подпакет аналитики: вычисление игровых метрик и формирование отчетов."""

from ttcv.analytics.metrics import RallyAnalyzer, RallyMetrics, ShotInfo
from ttcv.analytics.report import format_markdown_report, save_report_json

__all__ = [
    "RallyAnalyzer",
    "RallyMetrics",
    "ShotInfo",
    "format_markdown_report",
    "save_report_json",
]
