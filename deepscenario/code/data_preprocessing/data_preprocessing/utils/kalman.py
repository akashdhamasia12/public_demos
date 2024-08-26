import numpy as np
from filterpy.kalman import ExtendedKalmanFilter

from data_preprocessing.utils.generic import wrap_to_pi


def residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    residual = np.subtract(a, b)
    # angular dimensions
    residual[3:6] = wrap_to_pi(residual[3:6])
    return residual


class KalmanFilter:
    def __init__(self, z0: np.ndarray) -> None:
        # state = [x, y, z, roll, pitch, yaw, x_dot, y_dot, z_dot, roll_dot, pitch_dot, yaw_dot]
        # obs = [x, y, z, roll, pitch, yaw]
        self.kf = ExtendedKalmanFilter(dim_x=12, dim_z=6)

        # constant velocity model
        self.kf.F = np.row_stack([np.column_stack([np.eye(6), np.eye(6)]),
                                  np.column_stack([np.zeros((6, 6)), np.eye(6)])])
        self.H = np.column_stack([np.eye(6), np.zeros((6, 6))])
        self.kf.x = np.concatenate([z0, np.zeros(6)])  # initial state
        self.kf.P = 10 * np.diag([1, 1, 1, 1, 1, 1, 1000, 1000, 1000, 1000, 1000, 1000])  # initial state covariance
        self.kf.Q = 1 * np.diag([1, 1, 1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])  # process noise
        self.kf.R = 1 * np.diag([1, 1, 1, 1, 1, 1])  # measurement noise

    def update(self, z: np.ndarray) -> None:
        self.kf.update(z, HJacobian=lambda x: self.H, Hx=lambda x: self.H.dot(x), residual=residual)

    def predict_update(self, z: np.ndarray) -> None:
        self.kf.predict()
        self.update(z)

    def batch_filter(self, zs: list) -> tuple:
        # based on https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/kalman_filter.py#L754
        n = np.size(zs, 0)
        xs = np.zeros((n, self.kf.dim_x))
        Ps = np.zeros((n, self.kf.dim_x, self.kf.dim_x))

        for ii, z in enumerate(zs):
            self.predict_update(z)
            xs[ii, :] = self.kf.x
            Ps[ii, :, :] = self.kf.P

        return xs, Ps

    def rts_smoother(self, xs: np.ndarray, Ps: np.ndarray) -> tuple:
        # based on https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/kalman_filter.py#L923
        if len(xs) != len(Ps):
            raise ValueError('Length of xs and Ps must be the same')

        n = xs.shape[0]
        dim_x = xs.shape[1]

        # smoother gain
        K = np.zeros((n, dim_x, dim_x))

        x, P, xp, Pp = xs.copy(), Ps.copy(), xs.copy(), Ps.copy()
        for k in range(n-2, -1, -1):
            xp[k] = np.dot(self.kf.F, x[k])
            Pp[k] = np.dot(np.dot(self.kf.F, P[k]), self.kf.F.T) + self.kf.Q

            K[k]  = np.dot(np.dot(P[k], self.kf.F.T), np.linalg.inv(Pp[k]))
            x[k] += np.dot(K[k], residual(x[k + 1], xp[k]))
            P[k] += np.dot(np.dot(K[k], P[k+1] - Pp[k]), K[k].T)

        return x, P

    def get_state(self):
        return self.kf.x, self.kf.P
