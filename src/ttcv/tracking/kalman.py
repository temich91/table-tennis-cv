"""Фильтр Калмана для отслеживания мяча в настольном теннисе.

Использует 6-параметрическую кинематическую модель постоянного ускорения (CA):
    Вектор состояния X = [x, y, v_x, v_y, a_x, a_y]^T
    Вектор измерений Z = [x, y]^T

Физика:
  - Поддерживает неравномерные или единичные шаги по времени (dt = 1.0 кадр).
  - Моделирование ускорения учитывает гравитацию по вертикальной оси (+Y направлена вниз).
  - Обрабатывает пропуски измерений (перекрытия / размытие в движении) через шаги только предсказания.
  - Невязка измерений (инновация) позволяет определять события отскоков и ударов.
"""

from __future__ import annotations

import numpy as np


class BallKalmanFilter:
    """Фильтр Калмана, настроенный под динамику быстро движущегося теннисного мяча.

    Вектор состояния (6x1):
        [x, y, v_x, v_y, a_x, a_y]
        где x, y — координаты в пикселях, v_x, v_y — скорости (пикс/кадр),
        а a_x, a_y — ускорения (пикс/кадр^2).
    """

    def __init__(
        self,
        dt: float = 1.0,
        process_noise_pos: float = 0.5,
        process_noise_vel: float = 2.0,
        process_noise_acc: float = 5.0,
        measurement_noise: float = 4.0,
    ) -> None:
        """Инициализация матриц фильтра Калмана.

        Аргументы:
            dt: Шаг по времени (обычно 1.0 кадр).
            process_noise_pos: Дисперсия шума процесса для положения.
            process_noise_vel: Дисперсия шума процесса для скорости.
            process_noise_acc: Дисперсия шума процесса для ускорения.
            measurement_noise: Дисперсия шума измерений для детекций [x, y].
        """
        self.dt = dt
        self.initialized = False

        # Матрица перехода состояния F (6x6)
        # x_k = x_{k-1} + v*dt + 0.5*a*dt^2
        # v_k = v_{k-1} + a*dt
        # a_k = a_{k-1}
        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0, 0.5 * dt**2, 0.0],
                [0.0, 1.0, 0.0, dt, 0.0, 0.5 * dt**2],
                [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # Матрица измерений H (2x6): наблюдаем [x, y]
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        # Ковариация шума процесса Q (6x6)
        self.Q = np.diag(
            [
                process_noise_pos,
                process_noise_pos,
                process_noise_vel,
                process_noise_vel,
                process_noise_acc,
                process_noise_acc,
            ]
        ).astype(np.float32)

        # Ковариация шума измерений R (2x2)
        self.R = np.eye(2, dtype=np.float32) * measurement_noise

        # Вектор оценки состояния x (6x1) и ковариация ошибки P (6x6)
        self.x = np.zeros((6, 1), dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 100.0

        # Последняя невязка инновации для детекции отскоков
        self.last_residual: np.ndarray = np.zeros((2, 1), dtype=np.float32)

    def reset(self, initial_xy: tuple[float, float] | None = None) -> None:
        """Сброс фильтра Калмана в неинициализированное или заданное начальное положение."""
        self.x = np.zeros((6, 1), dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 100.0
        self.last_residual = np.zeros((2, 1), dtype=np.float32)
        if initial_xy is not None:
            self.x[0, 0] = initial_xy[0]
            self.x[1, 0] = initial_xy[1]
            self.initialized = True
        else:
            self.initialized = False

    def predict(self) -> tuple[float, float]:
        """Выполнение шага предсказания Калмана (априорная оценка).

        Возвращает:
            (x_pred, y_pred): предсказанное 2D-положение.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return (float(self.x[0, 0]), float(self.x[1, 0]))

    def update(self, measurement_xy: tuple[float, float]) -> tuple[float, float]:
        """Выполнение шага коррекции по измерениям Калмана (апостериорная оценка).

        Аргументы:
            measurement_xy: Обнаруженные координаты (x, y) в пикселях.

        Возвращает:
            (x_est, y_est): скорректированное 2D-положение состояния.
        """
        z = np.array([[measurement_xy[0]], [measurement_xy[1]]], dtype=np.float32)

        if not self.initialized:
            self.x[0, 0] = z[0, 0]
            self.x[1, 0] = z[1, 0]
            self.initialized = True
            return (float(self.x[0, 0]), float(self.x[1, 0]))

        # Невязка инновации: y = z - H * x_pred
        y = z - self.H @ self.x
        self.last_residual = y

        # Ковариация инновации: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Коэффициент Калмана (Kalman gain): K = P * H^T * S^{-1}
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Обновление состояния: x = x_pred + K * y
        self.x = self.x + K @ y

        # Обновление ковариации: P = (I - K * H) * P
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        return (float(self.x[0, 0]), float(self.x[1, 0]))

    # ── Свойства состояния ───────────────────────────────────────────────────

    @property
    def position(self) -> tuple[float, float]:
        """Текущее оцененное положение (x, y) в пикселях."""
        return (float(self.x[0, 0]), float(self.x[1, 0]))

    @property
    def velocity(self) -> tuple[float, float]:
        """Текущая оцененная скорость (v_x, v_y) в пикселях/кадр."""
        return (float(self.x[2, 0]), float(self.x[3, 0]))

    @property
    def speed_px(self) -> float:
        """Текущая евклидова скорость в пикселях/кадр."""
        vx, vy = self.velocity
        return float(np.hypot(vx, vy))

    @property
    def acceleration(self) -> tuple[float, float]:
        """Текущее оцененное ускорение (a_x, a_y) в пикселях/кадр^2."""
        return (float(self.x[4, 0]), float(self.x[5, 0]))
