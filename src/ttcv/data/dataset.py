"""Загрузчик датасета TTNet.

Считывает файлы аннотаций из датасета lab.osai.ai:
  - ball_markup.json     : {frame_id: {x, y}} — центр мяча в пикселях.
                           (-1, -1) означает, что мяч не виден на данном кадре.
  - events_markup.json   : {frame_id: event}  — "bounce" | "net" | "empty_event"
  - segmentation_masks/  : RGB PNG изображения (320×128) для каждого размеченного кадра.

Полноразмерные кадры (1920×1080) НЕ требуются для Шага 1.
Они используются на Шаге 3 (детекция и отслеживание мяча).

Типичная структура клипа::

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

# ── Константы ────────────────────────────────────────────────────────────────

#: Стандартные размеры теннисного стола в миллиметрах (правила ITTF).
TABLE_WIDTH_MM: int = 1525   # ширина поперек сетки
TABLE_LENGTH_MM: int = 2740  # длина вдоль стола

#: Full-HD разрешение кадров камер в датасете TTNet.
FRAME_WIDTH: int = 1920
FRAME_HEIGHT: int = 1080

#: Разрешение изображений масок сегментации.
MASK_WIDTH: int = 320
MASK_HEIGHT: int = 128

#: Константы меток событий.
EVENT_BOUNCE = "bounce"
EVENT_NET = "net"
EVENT_EMPTY = "empty_event"
ALL_EVENTS = frozenset({EVENT_BOUNCE, EVENT_NET, EVENT_EMPTY})

# ── Структуры данных ──────────────────────────────────────────────────────────


@dataclass
class BallPosition:
    """Положение мяча на отдельном кадре.

    Атрибуты:
        frame_id: Индекс кадра (целое число от 0 из датасета).
        x:        Горизонтальная координата пикселя (в пространстве FRAME_WIDTH × FRAME_HEIGHT).
                  -1 означает, что мяч не виден (перекрыт / за пределами кадра).
        y:        Вертикальная координата пикселя. -1 означает, что не виден.
    """

    frame_id: int
    x: int
    y: int

    @property
    def visible(self) -> bool:
        """Возвращает True, если мяч виден (координаты >= 0)."""
        return self.x >= 0 and self.y >= 0

    @property
    def xy(self) -> tuple[int, int]:
        """Возвращает кортеж (x, y)."""
        return (self.x, self.y)

    def as_array(self) -> np.ndarray:
        """Возвращает положение мяча как массив NumPy float32 [x, y]."""
        return np.array([self.x, self.y], dtype=np.float32)


@dataclass
class EventAnnotation:
    """Аннотация игрового события для отдельного кадра.

    Атрибуты:
        frame_id: Индекс кадра.
        event:    Одно из значений: "bounce", "net" или "empty_event".
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
    """Объединенная аннотация кадра (положение мяча + опциональное событие).

    Атрибуты:
        frame_id:    Индекс кадра.
        ball:        Положение мяча (может указывать на невидимый мяч).
        event:       Аннотация события (если кадр содержит событие; иначе None).
        mask_path:   Путь к PNG-файлу маски сегментации, если он существует.
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
    """Представление аннотаций одного игрового клипа в оперативной памяти.

    Атрибуты:
        root:          Корневая директория клипа.
        ball_data:     Отображение frame_id → BallPosition.
        event_data:    Отображение frame_id → EventAnnotation.
        mask_dir:      Директория с изображениями масок сегментации.
    """

    root: Path
    ball_data: dict[int, BallPosition] = field(default_factory=dict)
    event_data: dict[int, EventAnnotation] = field(default_factory=dict)
    mask_dir: Path | None = None

    # ── Свойства ──────────────────────────────────────────────────────────────

    @property
    def frame_ids(self) -> list[int]:
        """Отсортированный список всех ID кадров, имеющих хотя бы одну аннотацию."""
        return sorted(set(self.ball_data) | set(self.event_data))

    @property
    def n_frames(self) -> int:
        return len(self.frame_ids)

    @property
    def bounce_frames(self) -> list[int]:
        """Отсортированные ID кадров, где зафиксирован отскок (bounce)."""
        return sorted(fid for fid, ev in self.event_data.items() if ev.is_bounce)

    @property
    def net_frames(self) -> list[int]:
        """Отсортированные ID кадров, где зафиксировано касание сетки (net)."""
        return sorted(fid for fid, ev in self.event_data.items() if ev.is_net)

    @property
    def visible_ball_positions(self) -> list[BallPosition]:
        """Все экземпляры BallPosition, на которых мяч виден."""
        return [bp for bp in self.ball_data.values() if bp.visible]

    @property
    def trajectory_array(self) -> np.ndarray:
        """Массив формы (N, 3) вида [frame_id, x, y] для кадров с видимым мячом.

        Строки отсортированы по frame_id.
        """
        positions = sorted(self.visible_ball_positions, key=lambda bp: bp.frame_id)
        if not positions:
            return np.empty((0, 3), dtype=np.float32)
        return np.array([[p.frame_id, p.x, p.y] for p in positions], dtype=np.float32)

    # ── Доступ на уровне кадров ───────────────────────────────────────────────

    def get_frame(self, frame_id: int) -> FrameAnnotation:
        """Возвращает объединенный FrameAnnotation для указанного кадра."""
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
        """Итератор по всем размеченным кадрам по порядку."""
        for fid in self.frame_ids:
            yield self.get_frame(fid)

    # ── Сводка ────────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Возвращает краткий словарь со статистикой клипа."""
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


# ── Загрузчик ─────────────────────────────────────────────────────────────────


def load_game_clip(root: str | Path) -> GameClip:
    """Загрузка всех файлов аннотаций из директории одного игрового клипа.

    Аргументы:
        root: Путь к директории клипа, содержащей:
              - ``ball_markup.json``
              - ``events_markup.json``
              - ``segmentation_masks/`` (опционально)

    Возвращает:
        Экземпляр :class:`GameClip` со всеми загруженными в память аннотациями.

    Исключения:
        FileNotFoundError: Если ни ball_markup.json, ни events_markup.json
                           не найдены в *root*.
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

    # ── Загрузка положений мяча ────────────────────────────────────────────
    ball_data: dict[int, BallPosition] = {}
    if ball_path.exists():
        raw: dict[str, dict] = json.loads(ball_path.read_text(encoding="utf-8"))
        for fid_str, pos in raw.items():
            fid = int(fid_str)
            ball_data[fid] = BallPosition(frame_id=fid, x=int(pos["x"]), y=int(pos["y"]))

    # ── Загрузка событий ───────────────────────────────────────────────────
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
    """Загрузка всех игровых клипов, найденных в каталоге *data_root*.

    Рекурсивно сканирует *data_root* на наличие папок, содержащих
    ``ball_markup.json`` или ``events_markup.json``.

    Аргументы:
        data_root: Корневой каталог датасета (например, ``data/raw``).

    Возвращает:
        Список экземпляров :class:`GameClip` для каждой обнаруженной папки клипа.
    """
    data_root = Path(data_root)
    clips: list[GameClip] = []

    # Папка клипа — это папка, непосредственно содержащая ball_markup.json или events_markup.json.
    candidate_dirs: set[Path] = set()
    for marker in ("ball_markup.json", "events_markup.json"):
        for p in data_root.rglob(marker):
            candidate_dirs.add(p.parent)

    for clip_dir in sorted(candidate_dirs):
        try:
            clips.append(load_game_clip(clip_dir))
        except FileNotFoundError:
            pass  # Не должно происходить при логике поиска выше

    return clips
