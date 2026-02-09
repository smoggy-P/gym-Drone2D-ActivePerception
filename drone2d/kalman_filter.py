"""Linear Kalman filter for tracking dynamic obstacle agents.

Each :class:`KalmanFilter` instance tracks a single agent using a
constant-velocity motion model with 2-D position and velocity state.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from drone2d.config import SimConfig


class KalmanFilter:
    """4-D Kalman filter (x, y, vx, vy) for obstacle tracking.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration (used for noise variance and map bounds).
    mu : np.ndarray
        Initial state mean, shape ``(4, 1)``.
    sigma : np.ndarray
        Initial state covariance, shape ``(4, 4)``.
    """

    def __init__(
        self,
        config: SimConfig,
        mu: np.ndarray | None = None,
        sigma: np.ndarray | None = None,
    ) -> None:
        if mu is None:
            mu = np.zeros((4, 1))
        if sigma is None:
            sigma = np.diag([1.0, 1.0, 10.0, 10.0])

        self.active: bool = False
        self.config = config
        self.radius: float = config.agent_radius

        dx = mu.shape[0]
        assert mu.shape == (dx, 1)
        assert sigma.shape == (dx, dx)

        self.mu_upds: list[np.ndarray] = [mu]
        self.Sigma_upds: list[np.ndarray] = [sigma]
        self.ts: list[float] = [0.0]
        self._Dx = dx

        # --- Noise parameters -------------------------------------------
        has_cam_noise = config.var_cam != 0
        noise_pos = 0.1 if has_cam_noise else 0.001
        noise_vel = 0.1 if has_cam_noise else 0.001
        noise_z = config.var_cam

        dt = config.dt
        # State transition  (constant-velocity model)
        self.F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        # Observation matrix  (position only)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        # Process noise covariance
        self.Sigma_x = np.diag(
            [noise_pos, noise_pos, noise_vel, noise_vel]
        ).astype(np.float64)
        # Measurement noise covariance
        self.Sigma_z = np.diag([noise_z, noise_z]).astype(np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def copy(self) -> KalmanFilter:
        """Return a deep copy of this filter instance."""
        new_kf = KalmanFilter(self.config)
        new_kf.mu_upds = deepcopy(self.mu_upds)
        new_kf.Sigma_upds = deepcopy(self.Sigma_upds)
        new_kf.ts = list(self.ts)
        new_kf.active = self.active
        new_kf.radius = self.radius
        return new_kf

    def estimate_pos(self, t: float) -> np.ndarray:
        """Extrapolate the position *t* time-steps into the future.

        Returns a 1-D array ``[x, y]``.
        """
        pos = self.mu_upds[-1][:2, 0]
        vel = self.mu_upds[-1][2:, 0]
        return pos + t * vel

    def predict(self) -> KalmanFilter | None:
        """Run the prediction step.

        If the covariance grows too large or the predicted position leaves the
        map bounds the track is reset and the *achieved* (completed) filter is
        returned; otherwise ``None``.
        """
        mu_prev = self.mu_upds[-1]
        sigma_prev = self.Sigma_upds[-1]
        t = self.ts[-1]

        mu = self.F @ mu_prev
        sigma = self.F @ sigma_prev @ self.F.T + self.Sigma_x

        self.mu_upds.append(mu)
        self.Sigma_upds.append(sigma)
        self.ts.append(t + 1)

        # Check if the track should be terminated
        margin = 10 + self.config.agent_radius
        out_of_bounds = (
            not (margin < mu[0, 0] < self.config.map_size[0] - margin)
            or not (margin < mu[1, 0] < self.config.map_size[1] - margin)
        )
        if sigma[0, 0] >= 150 or out_of_bounds:
            achieved = self.copy()
            self.__init__(self.config)  # type: ignore[misc]
            return achieved
        return None

    def update(self, z: np.ndarray | None) -> list[KalmanFilter]:
        """Run one predict-update cycle.

        Parameters
        ----------
        z : np.ndarray | None
            Measurement ``[x, y]`` or *None* if the agent is not observed.

        Returns
        -------
        list[KalmanFilter]
            Completed (achieved) filter tracks, if any.
        """
        achieved_list: list[KalmanFilter] = []

        if self.active:
            achieved = self.predict()
            if achieved is not None:
                achieved_list.append(achieved)
            if z is not None:
                mu = self.mu_upds[-1]
                sigma = self.Sigma_upds[-1]
                z = z.reshape(-1, 1)

                S = self.Sigma_z + self.H @ sigma @ self.H.T
                K = sigma @ self.H.T @ np.linalg.inv(S)

                self.mu_upds[-1] = mu + K @ (z - self.H @ mu)
                self.Sigma_upds[-1] = (np.eye(4) - K @ self.H) @ sigma

        elif z is not None:
            # First observation — initialise the track
            mu = np.vstack([z.reshape(-1, 1), np.zeros((2, 1))])
            sigma = np.diag([1.0, 1.0, 10.0, 10.0])
            self.mu_upds = [mu]
            self.Sigma_upds = [sigma]
            self.ts = [0.0]
            self.active = True

        return achieved_list
