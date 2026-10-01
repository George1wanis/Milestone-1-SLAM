import numpy as np
import matplotlib

matplotlib.use("Agg")  # render off-screen so plots can be saved to files
import matplotlib.pyplot as plt

import Filter_visualizer


class KalmanFilter2D:
    """Kalman filter for state [px, py, vx, vy] with GPS + speedometer sensors."""

    def __init__(
        self,
        gps_measurement_var,
        speedometere_measurement_var,
        position_process_noise,
        velocity_process_noise,
        dt=1,
    ):
        self.dt = dt

        # State transition (constant velocity)
        self.A = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

        # Control matrix (acceleration input)
        self.B = np.array(
            [
                [0.5 * dt**2, 0],
                [0, 0.5 * dt**2],
                [dt, 0],
                [0, dt],
            ],
            dtype=float,
        )

        # Measurement matrix: GPS sees position, speedometer sees velocity
        self.H = np.eye(4)

        # Process noise covariance
        self.Q = np.diag(
            [
                position_process_noise,
                position_process_noise,
                velocity_process_noise,
                velocity_process_noise,
            ]
        )

        # Measurement noise covariance (GPS -> position, speedometer -> velocity)
        self.R = np.diag(
            [
                gps_measurement_var,
                gps_measurement_var,
                speedometere_measurement_var,
                speedometere_measurement_var,
            ]
        )

    def predict(self, X_past, P_past, U):
        X_pred = self.A @ X_past + self.B @ U
        P_pred = self.A @ P_past @ self.A.T + self.Q
        return X_pred, P_pred

    def correct(self, X_pred, P_pred, Z):
        y = Z - self.H @ X_pred  # innovation
        S = self.H @ P_pred @ self.H.T + self.R  # innovation covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        X_upd = X_pred + K @ y
        P_upd = (np.eye(self.A.shape[0]) - K @ self.H) @ P_pred
        return X_upd, P_upd


# (name, x0, P0_pos, P0_vel, u, z, Q_pos, Q_vel, R_gps, R_speedo)
SCENARIOS = [
    ("scenario_1", [0, 0, 10, 5], 1.0, 1.0, [2.0, 1.0], [10.5, 5.2, 9.8, 4.9], 0.5, 1.0, 4.0, 3.0),
    ("scenario_2", [0, 0, 10, 5], 1.0, 1.0, [2.0, 1.0], [10.5, 5.2, 12.0, 6.0], 0.05, 0.1, 9.0, 9.0),
    ("scenario_3", [0, 0, 10, 5], 1.0, 1.0, [2.0, 1.0], [35.0, 28.0, 10.1, 5.0], 0.05, 0.1, 20.0, 1.0),
    ("scenario_4", [0, 0, 0, 0], 0.01, 0.01, [0.0, 0.0], [0.01, 0.0, 0.0, 0.0], 0.001, 0.01, 0.04, 0.01),
]

PLOTS = [(0, "Position X", "position_x"), (2, "Velocity X", "velocity_x")]


if __name__ == "__main__":
    import os

    os.makedirs("Images", exist_ok=True)

    for name, x0, p_pos, p_vel, u, z, q_pos, q_vel, r_gps, r_spd in SCENARIOS:
        kf = KalmanFilter2D(r_gps, r_spd, q_pos, q_vel, dt=1)

        X0 = np.array(x0, dtype=float).reshape(4, 1)
        P0 = np.diag([p_pos, p_pos, p_vel, p_vel])
        U = np.array(u, dtype=float).reshape(2, 1)
        Z = np.array(z, dtype=float).reshape(4, 1)

        X_pred, P_pred = kf.predict(X0, P0, U)
        X_upd, P_upd = kf.correct(X_pred, P_pred, Z)

        print(f"\n=== {name} ===")
        print("x_pred:", X_pred.ravel(), " diag(P_pred):", np.diag(P_pred))
        print("z     :", Z.ravel(), " diag(R)     :", np.diag(kf.R))
        print("x_upd :", X_upd.ravel(), " diag(P_upd) :", np.diag(P_upd))

        for idx, label, suffix in PLOTS:
            path = os.path.join("Images", f"{name}_{suffix}.png")
            # The visualizer calls plt.show(); redirect it to save the figure.
            plt.show = lambda p=path: plt.savefig(p, dpi=300)
            Filter_visualizer.plot_1d_gaussians(
                X_pred, P_pred, Z, kf.R, X_upd, P_upd, state_idx=idx, label=label
            )
            plt.close("all")
