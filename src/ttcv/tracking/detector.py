"""Интерфейсы и реализации детекторов мяча.

Детекторы:
  - BaseBallDetector: Абстрактный базовый класс для детекторов мяча.
  - BallDetection: Датакласс для хранения ограничивающей рамки (bounding box), центра и уверенности.
  - MockBallDetector: Воспроизводит эталонную разметку мяча (с опциональным шумом / пропусками).
  - ColorMotionBallDetector: Быстрый классический CV-детектор по цвету HSV и фильтрации округлости.
  - YOLOv8BallDetector: Обертка для моделей Ultralytics YOLOv8 / YOLOv11 с обученными весами.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class BallDetection:
    """Обнаруженный кандидат на мяч в кадре.

    Атрибуты:
        x: Горизонтальная координата центра в пикселях.
        y: Вертикальная координата центра в пикселях.
        bbox: Кортеж ограничивающей рамки (x1, y1, x2, y2).
        confidence: Оценка уверенности детекции [0.0, 1.0].
        class_name: Имя класса (по умолчанию: "ball").
    """

    x: float
    y: float
    bbox: tuple[float, float, float, float]
    confidence: float = 1.0
    class_name: str = "ball"

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class BaseBallDetector(ABC):
    """Абстрактный базовый класс для всех моделей детекции мяча."""

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_id: int | None = None) -> list[BallDetection]:
        """Детекция мяча на отдельном изображении / кадре видео.

        Аргументы:
            frame: Изображение BGR или RGB формы (H, W, 3).
            frame_id: Опциональный индекс кадра для последовательных детекторов с состоянием.

        Возвращает:
            Список обнаруженных экземпляров :class:`BallDetection`, отсортированный по убыванию уверенности.
        """
        pass


class MockBallDetector(BaseBallDetector):
    """Детектор, воспроизводящий эталонную разметку с опциональной симуляцией шума.

    Позволяет тестировать трекинг и аналитику без необходимости работы с сырым видео или GPU.
    """

    def __init__(
        self,
        markup_path_or_dict: str | Path | dict[str, dict[str, int]],
        noise_std: float = 0.0,
        dropout_prob: float = 0.0,
        seed: int = 42,
    ) -> None:
        """Инициализация MockBallDetector.

        Аргументы:
            markup_path_or_dict: Путь к ball_markup.json или загруженный словарь.
            noise_std: Стандартное отклонение гауссовского шума, добавляемого к координатам (в пикс).
            dropout_prob: Вероятность [0.0, 1.0] симуляции пропуска детекции.
            seed: Зерно генератора случайных чисел для воспроизводимости.
        """
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.rng = np.random.default_rng(seed)

        if isinstance(markup_path_or_dict, (str, Path)):
            raw = json.loads(Path(markup_path_or_dict).read_text(encoding="utf-8"))
        else:
            raw = markup_path_or_dict

        self.markup: dict[int, tuple[int, int]] = {}
        for fid_str, pos in raw.items():
            self.markup[int(fid_str)] = (int(pos["x"]), int(pos["y"]))

    def detect(self, frame: np.ndarray, frame_id: int | None = None) -> list[BallDetection]:
        if frame_id is None or frame_id not in self.markup:
            return []

        x, y = self.markup[frame_id]
        if x < 0 or y < 0:
            return []  # Не виден в эталонной разметке

        # Симулированный пропуск (размытие в движении / перекрытие)
        if self.dropout_prob > 0.0 and self.rng.random() < self.dropout_prob:
            return []

        # Симулированный шум детектора
        if self.noise_std > 0.0:
            x = float(x + self.rng.normal(0.0, self.noise_std))
            y = float(y + self.rng.normal(0.0, self.noise_std))

        # Ограничивающая рамка 20x20
        bbox = (float(x - 10), float(y - 10), float(x + 10), float(y + 10))
        return [BallDetection(x=float(x), y=float(y), bbox=bbox, confidence=0.98)]


class ColorMotionBallDetector(BaseBallDetector):
    """Быстрый классический детектор мяча методами компьютерного зрения.

    Фильтрует по цвету HSV (белый / оранжевый мяч) и круглой форме контура.
    Работает на CPU без предобученных весов нейросетей.
    """

    def __init__(
        self,
        ball_color: str = "white",
        min_radius: int = 2,
        max_radius: int = 15,
        min_circularity: float = 0.45,
        roi_bbox: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Инициализация ColorMotionBallDetector.

        Аргументы:
            ball_color: "white" (белый) или "orange" (оранжевый).
            min_radius: Минимальный радиус мяча в пикселях.
            max_radius: Максимальный радиус мяча в пикселях.
            min_circularity: Минимальная округлость = 4*pi*area / (perimeter^2).
            roi_bbox: Опциональный (x1, y1, x2, y2) для ограничения области поиска (ROI).
        """
        self.ball_color = ball_color.lower()
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_circularity = min_circularity
        self.roi_bbox = roi_bbox

    def detect(self, frame: np.ndarray, frame_id: int | None = None) -> list[BallDetection]:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Цветовая маска
        if self.ball_color == "white":
            # Низкая насыщенность, высокая яркость
            mask = cv2.inRange(hsv, (0, 0, 195), (180, 70, 255))
        elif self.ball_color == "orange":
            # Оранжевый оттенок [5, 25]
            mask = cv2.inRange(hsv, (5, 100, 120), (25, 255, 255))
        else:
            raise ValueError(f"Unknown ball color: {self.ball_color}")

        # Обрезка по области интереса (ROI), если задана
        if self.roi_bbox is not None:
            x1, y1, x2, y2 = self.roi_bbox
            roi_mask = np.zeros_like(mask)
            roi_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            mask = roi_mask

        # Морфологическое размыкание (opening) для подавления шума
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[BallDetection] = []

        min_area = np.pi * (self.min_radius**2)
        max_area = np.pi * (self.max_radius**2)

        for c in cnts:
            area = cv2.contourArea(c)
            if not (min_area <= area <= max_area):
                continue

            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue

            circularity = 4.0 * np.pi * (area / (peri**2))
            if circularity >= self.min_circularity:
                (cx, cy), radius = cv2.minEnclosingCircle(c)
                bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
                conf = min(1.0, float(circularity))
                detections.append(
                    BallDetection(
                        x=float(cx),
                        y=float(cy),
                        bbox=bbox,
                        confidence=conf,
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


class YOLOv8BallDetector(BaseBallDetector):
    """Обертка детектора Ultralytics YOLO для обученных весов."""

    def __init__(
        self,
        weights_path: str | Path,
        conf_thresh: float = 0.25,
        imgsz: int = 1280,
        device: str | None = None,
    ) -> None:
        """Инициализация детектора YOLO.

        Аргументы:
            weights_path: Путь к best.pt или yolov8n.pt.
            conf_thresh: Порог уверенности для кандидатов в детекции.
            imgsz: Размер изображения для инференса (1280 рекомендуется для маленьких мячей).
            device: 'cuda', 'cpu' или None для автовыбора.
        """
        try:
            from ultralytics import YOLO
        except ImportError as err:
            raise ImportError(
                "Ultralytics is required to use YOLOv8BallDetector. "
                "Install with: pip install ultralytics"
            ) from err

        self.model = YOLO(str(weights_path))
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.device = device

    def detect(self, frame: np.ndarray, frame_id: int | None = None) -> list[BallDetection]:
        results = self.model.predict(
            frame,
            conf=self.conf_thresh,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections: list[BallDetection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            detections.append(
                BallDetection(
                    x=cx,
                    y=cy,
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=conf,
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
