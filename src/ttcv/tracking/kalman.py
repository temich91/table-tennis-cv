"""Kalman Filter for table tennis ball tracking.

Uses a 6-state constant acceleration (CA) kinematic model:
    State X = [x, y, v_x, v_y, a_x, a_y]^T
    Measurements Z = [x, y]^T

Physics:
  - Supports non-uniform or unit time steps (dt = 1.0 frame).
  - Acceleration modeling accounts for gravity in vertical axis (+Y downwards).
  - Handles missing measurements (occlusion / motion blur) via prediction-only steps.
  - Innovation residual allows bounce / collision event detection.
"""

from __future__ import annotations

import numpy as np


class BallKalmanFilter:
    """Kalman Filter tuned for fast-moving table tennis ball dynamics.

    State vector (6x1):
        [x, y, v_x, v_y, a_x, a_y]
        where x, y are pixel positions, v_x, v_y are velocities (px/frame),
        and a_x, a_y are accelerations (px/frame^2).
    """

    def __init__(
        self,
        dt: float = 1.0,
        process_noise_pos: float = 0.5,
        process_noise_vel: float = 2.0,
        process_noise_acc: float = 5.0,
        measurement_noise: float = 4.0,
    ) -> None:
        """Initialize the Kalman filter matrices.

        Args:
            dt: Time step (typically 1.0 frame).
            process_noise_pos: Process noise variance for position.
            process_noise_vel: Process noise variance for velocity.
            process_noise_acc: Process noise variance for acceleration.
            measurement_noise: Measurement noise variance for [x, y] detections.
        """
        self.dt = dt
        self.initialized = False

        # State transition matrix F (6x6)
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

        # Measurement matrix H (2x6): we observe [x, y]
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        # Process noise covariance Q (6x6)
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

        # Measurement noise covariance R (2x2)
        self.R = np.eye(2, dtype=np.float32) * measurement_noise

        # State estimate vector x (6x1) and covariance P (6x6)
        self.x = np.zeros((6, 1), dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 100.0

        # Last innovation residual for bounce detection
        self.last_residual: np.ndarray = np.zeros((2, 1), dtype=np.float32)

    def reset(self, initial_xy: tuple[float, float] | None = None) -> None:
        """Reset the Kalman filter to uninitialized or specified initial position."""
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
        """Execute Kalman prediction step (a priori estimate).

        Returns:
            (x_pred, y_pred): predicted 2D position.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return (float(self.x[0, 0]), float(self.x[1, 0]))

    def update(self, measurement_xy: tuple[float, float]) -> tuple[float, float]:
        """Execute Kalman measurement update step (a posteriori estimate).

        Args:
            measurement_xy: Detected (x, y) coordinates in pixels.

        Returns:
            (x_est, y_est): updated 2D state position.
        """
        z = np.array([[measurement_xy[0]], [measurement_xy[1]]], dtype=np.float32)

        if not self.initialized:
            self.x[0, 0] = z[0, 0]
            self.x[1, 0] = z[1, 0]
            self.initialized = True
            return (float(self.x[0, 0]), float(self.x[1, 0]))

        # Innovation residual: y = z - H * x_pred
        y = z - self.H @ self.x
        self.last_residual = y

        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: K = P * H^T * S^{-1}
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update: x = x_pred + K * y
        self.x = self.x + K @ y

        # Covariance update: P = (I - K * H) * P
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        return (float(self.x[0, 0]), float(self.x[1, 0]))

    # ── State Properties ──────────────────────────────────────────────────────

    @property
    def position(self) -> tuple[float, float]:
        """Current estimated (x, y) position in pixels."""
        return (float(self.x[0, 0]), float(self.x[1, 0]))

    @property
    def velocity(self) -> tuple[float, float]:
        """Current estimated (v_x, v_y) velocity in pixels/frame."""
        return (float(self.x[2, 0]), float(self.x[3, 0]))

    @property
    def speed_px(self) -> float:
        """Current Euclidean speed in pixels/frame."""
        vx, vy = self.velocity
        return float(np.hypot(vx, vy))

    @property
    def acceleration(self) -> tuple[float, float]:
        """Current estimated (a_x, a_y) acceleration in pixels/frame^2."""
        return (float(self.x[4, 0]), float(self.x[5, 0]))
